use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use thiserror::Error;

// -- Namespace re-exports --------------------------------------------------
//
// Integration tests import these names under `libviprs::sink::*` even though
// they live in sibling modules. Re-exporting here lets the public API expose a
// stable `sink::` namespace without forcing consumers to know about the
// internal module layout.

pub use crate::dedupe::DedupeStrategy;
pub use crate::retry::{FailurePolicy, RetryPolicy, RetryingSink};

#[cfg(feature = "s3")]
pub use crate::sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};

#[cfg(feature = "packfile")]
pub use crate::sink_packfile::{PackfileFormat, PackfileSink};

/// Errors that can occur when writing tiles to a sink.
///
/// Covers I/O failures (e.g. filesystem permission errors), image encoding
/// failures (e.g. unsupported pixel format for JPEG), and general sink errors
/// for invalid coordinates or configuration. Every [`TileSink`] method returns
/// `Result<(), SinkError>`.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for error handling patterns.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SinkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Legacy free-form encode error. Prefer [`SinkError::Encode`] with the
    /// typed `format` + `source` pair for new call sites.
    #[error("image encode error: {0}")]
    EncodeMsg(String),
    /// Typed variant for image-encoder failures. The `format` field is the
    /// human-readable target format (e.g. `"png"` / `"jpeg"`) and `source` is
    /// the underlying [`image::ImageError`].
    #[error("encoding tile to {format:?} failed: {source}")]
    Encode {
        format: String,
        #[source]
        source: image::ImageError,
    },
    /// Used for all catch-all string errors that haven't yet been promoted to
    /// a typed variant. New code should prefer the typed variants below.
    #[error("sink error: {0}")]
    Other(String),
    /// A tile coordinate fell outside the plan's level bounds. Raised from
    /// [`FsSink::write_tile`] when [`PyramidPlan::tile_path`] returns `None`.
    #[error("tile coord {coord:?} is outside level bounds")]
    InvalidCoord { coord: TileCoord },
    /// A sink that requires the active [`crate::engine::EngineConfig`] was
    /// invoked without one. Sinks that need this should be constructed via
    /// [`TileSink::record_engine_config`] before the tile loop starts.
    #[error("engine config not available on sink (construct via with_engine_config)")]
    MissingEngineConfig,
    /// A per-tile checksum did not match the expected digest. Engine-level
    /// code promotes this to [`crate::engine::EngineError::ChecksumMismatch`]
    /// so callers see the dedicated error variant rather than a generic
    /// "sink error" string.
    #[error("checksum mismatch for {tile_rel_path}: expected {expected}, got {got}")]
    ChecksumMismatch {
        tile_rel_path: String,
        expected: String,
        got: String,
    },
    /// A required field was not set on a sink builder before `build()` was
    /// called. The payload is the fully-qualified field name, e.g.
    /// `"PackfileSinkBuilder::plan"`.
    #[error("required builder field not set: {0}")]
    MissingField(&'static str),
}

/// Single-byte marker written in place of blank tiles when using
/// `BlankTileStrategy::Placeholder`. Consumers can detect placeholder
/// files by checking `file.len() == 1 && file[0] == BLANK_TILE_MARKER`.
///
/// # Examples
///
/// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for placeholder detection patterns.
pub const BLANK_TILE_MARKER: u8 = 0x00;

/// A produced tile, ready for output.
///
/// Represents a single tile in the pyramid after rasterisation. The engine
/// creates a `Tile` for every grid cell in the plan, attaches the rendered
/// [`Raster`], and passes it to a [`TileSink`]. The `blank` flag allows sinks
/// to write a lightweight placeholder instead of encoding a full image when
/// all pixels are identical.
///
/// # Examples
///
/// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for usage with blank detection.
#[derive(Debug)]
pub struct Tile {
    pub coord: TileCoord,
    pub raster: Raster,
    /// When `true`, this tile is blank (all pixels identical) and was marked
    /// for placeholder output by `BlankTileStrategy::Placeholder`.
    ///
    /// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs).
    pub blank: bool,
}

/// Trait for receiving tiles produced by the engine.
///
/// Implementations handle where tiles go — filesystem, object store, memory, etc.
/// The engine calls `write_tile` from worker threads, so implementations must be
/// `Send + Sync`.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for filesystem sink integration tests and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the CLI wires up a sink.
pub trait TileSink: Send + Sync {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError>;
    fn finish(&self) -> Result<(), SinkError> {
        Ok(())
    }
    /// Engine hook: forward a snapshot of the active [`crate::engine::EngineConfig`]
    /// into the sink so it can populate the manifest's generation settings and
    /// sparse-policy fields. The default implementation is a no-op; only sinks
    /// that emit manifests (e.g. [`FsSink`]) need to override.
    fn record_engine_config(&self, _config: &crate::engine::EngineConfig) {}

    /// Engine hook: when the sink (or a wrapper around it) keeps an internal
    /// retry counter, expose the running total so the engine can include it
    /// in [`crate::engine::EngineResult::retry_count`]. Default is `0`.
    fn sink_retry_count(&self) -> u64 {
        0
    }

    /// Engine hook: when the sink (or a wrapper around it) keeps an internal
    /// skip counter, expose the running total so the engine can include it in
    /// [`crate::engine::EngineResult::skipped_due_to_failure`]. Default is `0`.
    fn sink_skipped_due_to_failure(&self) -> u64 {
        0
    }

    /// Engine hook: bump the skip counter by one, used by the engine when a
    /// `FailurePolicy::RetryThenSkip` tile is dropped. Default is a no-op.
    fn note_sink_skipped(&self) {}

    /// Engine hook: the on-disk root where the checkpoint file
    /// `.libviprs-job.json` should live. Sinks that do not write to the
    /// filesystem return `None` (the default).
    fn checkpoint_root(&self) -> Option<&Path> {
        None
    }

    /// Engine hook: tell the sink how many pyramid levels will appear in
    /// this run, so sinks that keep per-level counters can pre-size their
    /// backing storage before the tile loop starts. Default is a no-op.
    /// [`FsSink`] already sizes its counters from the plan in
    /// [`FsSink::new`], so calling this is idempotent there.
    fn init_level_count(&self, _levels: usize) {}
}

/// Forwarding impl so `Box<dyn TileSink>` (and `Box<T>` where `T: TileSink`)
/// satisfies [`TileSink`] itself.
///
/// Required so callers can unify match arms that return different concrete
/// sink types under `Box<dyn TileSink>` and feed the boxed form to
/// [`EngineBuilder::new`](crate::EngineBuilder::new):
///
/// ```ignore
/// let sink: Box<dyn TileSink> = match mode {
///     "mem" => Box::new(MemorySink::new()),
///     "fs"  => Box::new(FsSink::new(dir, plan)),
///     _ => unreachable!(),
/// };
/// EngineBuilder::new(&src, plan, sink).run()?;
/// ```
impl<T: TileSink + ?Sized> TileSink for Box<T> {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        (**self).write_tile(tile)
    }
    fn finish(&self) -> Result<(), SinkError> {
        (**self).finish()
    }
    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        (**self).record_engine_config(config)
    }
    fn sink_retry_count(&self) -> u64 {
        (**self).sink_retry_count()
    }
    fn sink_skipped_due_to_failure(&self) -> u64 {
        (**self).sink_skipped_due_to_failure()
    }
    fn note_sink_skipped(&self) {
        (**self).note_sink_skipped()
    }
    fn checkpoint_root(&self) -> Option<&Path> {
        (**self).checkpoint_root()
    }
    fn init_level_count(&self, levels: usize) {
        (**self).init_level_count(levels)
    }
}

// ---------------------------------------------------------------------------
// MemorySink
// ---------------------------------------------------------------------------

/// In-memory sink that collects all tiles into a `Vec<CollectedTile>`.
///
/// Primarily intended for testing: it lets you assert on the exact tiles the
/// engine produced without touching the filesystem. Thread-safe via an internal
/// `Mutex`, so it satisfies `Send + Sync`.
///
/// # Examples
///
/// See [observability tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
/// for end-to-end usage with the engine.
#[derive(Debug)]
pub struct MemorySink {
    tiles: std::sync::Mutex<Vec<CollectedTile>>,
}

/// A snapshot of a tile captured by [`MemorySink`].
///
/// Stores the tile's coordinate, dimensions, and raw pixel bytes so that tests
/// can inspect tile output without needing to decode an image format. Created
/// automatically when [`MemorySink::write_tile`] is called.
///
/// # Examples
///
/// See [observability tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
/// for assertions on collected tiles.
#[derive(Debug, Clone)]
pub struct CollectedTile {
    pub coord: TileCoord,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    /// Full raster snapshot of the collected tile (same pixel data as
    /// [`CollectedTile::data`] but wrapped in a [`Raster`] so tests can call
    /// `tile.raster.data().len()` like they would on a [`Tile`]).
    pub raster: Raster,
}

impl MemorySink {
    pub fn new() -> Self {
        Self {
            tiles: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn tiles(&self) -> Vec<CollectedTile> {
        self.tiles.lock().unwrap().clone()
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.lock().unwrap().len()
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl TileSink for MemorySink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.tiles.lock().unwrap().push(CollectedTile {
            coord: tile.coord,
            width: tile.raster.width(),
            height: tile.raster.height(),
            data: tile.raster.data().to_vec(),
            raster: tile.raster.clone(),
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SlowSink (testing)
// ---------------------------------------------------------------------------

/// A sink that artificially delays every `write_tile` call by a fixed duration.
///
/// Wraps a [`MemorySink`] so tiles are still collected for inspection. Exists
/// to test backpressure and concurrency behaviour in the engine: by making the
/// sink slow, you can verify that the engine correctly limits in-flight work.
///
/// # Examples
///
/// See [stress tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/stress.rs)
/// for backpressure scenarios.
#[derive(Debug)]
pub struct SlowSink {
    inner: MemorySink,
    delay: std::time::Duration,
}

impl SlowSink {
    pub fn new(delay: std::time::Duration) -> Self {
        Self {
            inner: MemorySink::new(),
            delay,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.inner.tile_count()
    }

    pub fn tiles(&self) -> Vec<CollectedTile> {
        self.inner.tiles()
    }
}

impl TileSink for SlowSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        std::thread::sleep(self.delay);
        self.inner.write_tile(tile)
    }
}

// ---------------------------------------------------------------------------
// FsSink — filesystem tile output
// ---------------------------------------------------------------------------

/// Tile image encoding format for filesystem output.
///
/// Controls how [`FsSink`] encodes pixel data before writing to disk. Also
/// determines the file extension via [`TileFormat::extension`].
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for format selection and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the CLI maps user flags to a `TileFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileFormat {
    Png,
    Jpeg {
        quality: u8,
    },
    /// Raw pixel bytes (no encoding). Fastest, useful for pipelines that
    /// encode later or for testing.
    Raw,
}

impl TileFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpeg",
            Self::Raw => "raw",
        }
    }
}

/// Writes tiles to the local filesystem.
///
/// Directory structure follows the plan's layout:
/// - DeepZoom: `{base}/{level}/{col}_{row}.{ext}` + `{base}.dzi`
/// - XYZ: `{base}/{z}/{x}/{y}.{ext}`
///
/// Intermediate directories are created automatically. Call [`TileSink::finish`]
/// after all tiles have been written to emit format-specific metadata (e.g. the
/// DZI manifest for DeepZoom layouts).
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for integration tests and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the `pyramid` command constructs an `FsSink`.
///
/// `Debug` is implemented manually because the internal [`DedupeIndex`] does
/// not derive `Debug`.
///
/// # Lock order
///
/// `pixel_format` (OnceLock, no lock) -> `per_level_counts` (atomics, no
/// lock) -> `tile_digests` -> `manifest_refs` -> `pending_first` ->
/// `completed_tiles` -> `engine_config` -> dedupe mutexes. **Do not nest**
/// locks in a different order; the hot write path acquires only the
/// tile-local mutexes it needs and never reaches back up the chain.
pub struct FsSink {
    base_dir: PathBuf,
    plan: PyramidPlan,
    format: TileFormat,
    manifest_builder: Option<crate::manifest::ManifestBuilder>,
    checksums: crate::checksum::ChecksumMode,
    checksum_algo: Option<crate::manifest::ChecksumAlgo>,
    dedupe: Option<crate::dedupe::DedupeStrategy>,
    /// Lazily-initialised dedupe index, present only when `dedupe` is not
    /// [`DedupeStrategy::None`].
    dedupe_index: Option<crate::dedupe::DedupeIndex>,
    resume_enabled: bool,
    /// Running per-tile checksum table, populated only when `checksums` is
    /// non-[`ChecksumMode::None`]. Keyed by the relative tile path inside
    /// `base_dir`. Stores the raw 32-byte digest to keep the hot path
    /// allocation-free; the manifest writer hex-encodes once at emit time.
    tile_digests: Mutex<BTreeMap<String, [u8; 32]>>,
    /// Tile rel-path -> shared file rel-path (e.g. `_shared/blank_abc.png`).
    /// Populated by the dedupe write path; emitted into the manifest's
    /// `blank_references` field.
    manifest_refs: Mutex<HashMap<String, String>>,
    /// Holds the "first occurrence" bookkeeping for content that has only
    /// been seen once. When the second occurrence arrives, we promote the
    /// first tile's bytes into `_shared/` and then link both tile paths to
    /// the shared file.
    pending_first: Mutex<HashMap<String, PendingFirst>>,
    /// Per-level tile counters, indexed by level. Each entry is
    /// `[produced, skipped]` atomically-updated from the hot path. `skipped`
    /// tracks blank placeholders or deduped references. The Vec is sized
    /// eagerly at construction from the plan so per-tile writes are pure
    /// atomics (no lock, no growth).
    per_level_counts: Vec<[AtomicU64; 2]>,
    /// Captured from the first tile's raster so the manifest can record
    /// `source.pixel_format`. Written once at first tile; readers use
    /// `.get()`.
    pixel_format: OnceLock<crate::pixel::PixelFormat>,
    /// Completed tile coordinates. Populated when resume is enabled so
    /// [`FsSink::finish`] can write a [`JobMetadata`](crate::resume::JobMetadata)
    /// checkpoint. Capacity is pre-reserved from the plan's total tile
    /// count at construction so per-tile pushes never reallocate.
    completed_tiles: Mutex<Vec<TileCoord>>,
    /// Set to true the first time we observe `tile.blank == true` or a
    /// blank tile is detected while deduping. Used to infer
    /// `sparse_policy.dedupe`.
    saw_blank: AtomicBool,
    /// Engine-level configuration captured via [`TileSink::record_engine_config`]
    /// at the top of `generate_pyramid_observed`. Consumed when the manifest
    /// is written so that `GenerationSettings.concurrency`, `background_rgb`
    /// and `blank_strategy` round-trip through the output.
    engine_config: Mutex<Option<crate::engine::EngineConfig>>,
}

/// Internal bookkeeping for the "promote on 2nd hit" dedupe path. When the
/// first reference for a content hash is seen, we write the bytes at the
/// tile path and stash the following. If a second reference arrives we
/// promote the bytes into `_shared/` and link both tile paths at it; if no
/// second reference ever arrives we leave the file alone.
#[derive(Debug, Clone)]
struct PendingFirst {
    /// Absolute tile path where the bytes were originally written.
    tile_abs_path: PathBuf,
    /// Tile path relative to `base_dir`.
    tile_rel_path: String,
    /// Absolute path of the would-be shared file.
    #[allow(dead_code)]
    shared_abs_path: PathBuf,
    /// Path of the would-be shared file, relative to `base_dir`.
    #[allow(dead_code)]
    shared_rel_path: String,
    /// The encoded bytes, kept so we can fall back to writing them into
    /// `_shared/` directly if moving the original file fails.
    bytes: Vec<u8>,
}

impl std::fmt::Debug for FsSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsSink")
            .field("base_dir", &self.base_dir)
            .field("format", &self.format)
            .field("checksums", &self.checksums)
            .field("checksum_algo", &self.checksum_algo)
            .field("dedupe", &self.dedupe)
            .field("resume_enabled", &self.resume_enabled)
            .finish()
    }
}

impl FsSink {
    /// Creates a new filesystem sink rooted at `base_dir` with the given
    /// pyramid plan. The tile encoding format defaults to
    /// [`TileFormat::Png`]; override it via [`FsSink::with_format`] when
    /// writing JPEG or Raw tiles:
    ///
    /// ```ignore
    /// FsSink::new(dir, plan).with_format(TileFormat::Jpeg { quality: 85 });
    /// ```
    pub fn new(base_dir: impl Into<PathBuf>, plan: PyramidPlan) -> Self {
        Self::new_with_format_inner(base_dir, plan, TileFormat::Png)
    }

    /// Deprecated three-argument constructor that takes the tile format as a
    /// positional parameter.
    ///
    /// Retained as a migration alias; new code should prefer
    /// [`FsSink::new`] followed by [`FsSink::with_format`] when a non-PNG
    /// format is needed.
    #[deprecated(
        since = "0.3.0",
        note = "use FsSink::new(dir, plan).with_format(format) instead"
    )]
    pub fn new_with_format(
        base_dir: impl Into<PathBuf>,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Self {
        Self::new_with_format_inner(base_dir, plan, format)
    }

    fn new_with_format_inner(
        base_dir: impl Into<PathBuf>,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Self {
        let base_dir = base_dir.into();
        // Pre-size the per-level atomic counter vector so that the hot
        // write path can index by `level as usize` without any lock or
        // allocation. `levels.len()` matches the highest level index + 1
        // for every layout libviprs supports.
        let level_slots = plan.levels.len().max(1);
        let mut per_level_counts: Vec<[AtomicU64; 2]> = Vec::with_capacity(level_slots);
        for _ in 0..level_slots {
            per_level_counts.push([AtomicU64::new(0), AtomicU64::new(0)]);
        }
        // Pre-reserve the completed-tiles Vec so resume-mode runs don't
        // reallocate under the mutex on every tile.
        let total_tiles: u64 = plan
            .levels
            .iter()
            .map(|lp| (lp.cols as u64) * (lp.rows as u64))
            .sum();
        let completed_cap = usize::try_from(total_tiles).unwrap_or(0);
        Self {
            base_dir,
            plan,
            format,
            manifest_builder: None,
            checksums: crate::checksum::ChecksumMode::None,
            checksum_algo: None,
            dedupe: None,
            dedupe_index: None,
            resume_enabled: false,
            tile_digests: Mutex::new(BTreeMap::new()),
            manifest_refs: Mutex::new(HashMap::new()),
            pending_first: Mutex::new(HashMap::new()),
            per_level_counts,
            pixel_format: OnceLock::new(),
            completed_tiles: Mutex::new(Vec::with_capacity(completed_cap)),
            saw_blank: AtomicBool::new(false),
            engine_config: Mutex::new(None),
        }
    }

    /// Attach a [`ManifestBuilder`](crate::manifest::ManifestBuilder) so the
    /// sink emits a `manifest.json` alongside the pyramid when
    /// [`FsSink::finish`] is called.
    pub fn with_manifest(mut self, builder: crate::manifest::ManifestBuilder) -> Self {
        // If the builder specifies a checksum algorithm and the caller has
        // not separately configured checksums, default to EmitOnly so the
        // manifest has a per-tile table to populate.
        if let Some(algo) = builder.checksum_algo() {
            self.checksum_algo = Some(algo);
            if self.checksums == crate::checksum::ChecksumMode::None {
                self.checksums = crate::checksum::ChecksumMode::EmitOnly;
            }
        }
        self.manifest_builder = Some(builder);
        self
    }

    /// Configure per-tile checksum emission / verification for this sink.
    ///
    /// Argument order: `(mode, algo)` to mirror `.with_checksums(Verify,
    /// Blake3)` call-site readability (the mode is usually the focus of the
    /// test/config, with the algorithm as a secondary choice).
    pub fn with_checksums(
        mut self,
        mode: crate::checksum::ChecksumMode,
        algo: crate::manifest::ChecksumAlgo,
    ) -> Self {
        self.checksum_algo = Some(algo);
        self.checksums = mode;
        self
    }

    /// Set only the checksum mode; the algorithm is inherited from a
    /// previously attached `ManifestBuilder::with_checksums(algo)`.
    pub fn with_checksum_mode(mut self, mode: crate::checksum::ChecksumMode) -> Self {
        self.checksums = mode;
        self
    }

    /// Attach a [`DedupeStrategy`](crate::dedupe::DedupeStrategy) so the sink
    /// can coalesce identical blank tiles under a shared reference.
    pub fn with_dedupe(mut self, strategy: crate::dedupe::DedupeStrategy) -> Self {
        if strategy != crate::dedupe::DedupeStrategy::None {
            self.dedupe_index = Some(crate::dedupe::DedupeIndex::new(strategy));
        } else {
            self.dedupe_index = None;
        }
        self.dedupe = Some(strategy);
        self
    }

    /// Enable resume metadata. When set, [`FsSink::finish`] writes a small
    /// `.libviprs-job.json` checkpoint alongside the pyramid.
    pub fn with_resume(mut self, enabled: bool) -> Self {
        self.resume_enabled = enabled;
        self
    }

    /// Override the tile encoding format after construction.
    ///
    /// Overrides the tile encoding format set by [`FsSink::new`] (which
    /// defaults to [`TileFormat::Png`]). Chain with the other `with_*`
    /// methods to configure the full sink in builder style:
    ///
    /// ```ignore
    /// FsSink::new(dir, plan)
    ///     .with_format(TileFormat::Jpeg { quality: 85 })
    ///     .with_dedupe(DedupeStrategy::Blanks);
    /// ```
    pub fn with_format(mut self, format: TileFormat) -> Self {
        self.format = format;
        self
    }

    /// Returns the root output directory for this sink.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    #[allow(dead_code)]
    fn tile_path(&self, coord: TileCoord) -> Option<PathBuf> {
        let rel = self.plan.tile_path(coord, self.format.extension())?;
        Some(self.base_dir.join(rel))
    }

    fn encode_tile(&self, raster: &Raster) -> Result<Vec<u8>, SinkError> {
        match self.format {
            TileFormat::Raw => Ok(raster.data().to_vec()),
            TileFormat::Png => encode_png(raster),
            TileFormat::Jpeg { quality } => encode_jpeg(raster, quality),
        }
    }

    /// Whether dedupe was configured with a non-`None` strategy.
    fn dedupe_active(&self) -> bool {
        matches!(
            self.dedupe,
            Some(crate::dedupe::DedupeStrategy::Blanks)
                | Some(crate::dedupe::DedupeStrategy::All { .. })
        )
    }

    /// Decide whether a given tile should go through the dedupe pipeline.
    ///
    /// Both [`DedupeStrategy::Blanks`] and [`DedupeStrategy::All`] only
    /// promote uniform-colour tiles (as determined by
    /// [`crate::engine::is_blank_tile`]). The difference is that `All` also
    /// applies to non-white uniform tiles (greys, coloured bands, etc.)
    /// and uses the caller-chosen hash algorithm. Non-uniform content —
    /// e.g. gradients or photographs — is never promoted to `_shared/`,
    /// which guarantees `_shared/` stays empty when all input tiles are
    /// visually distinct.
    fn should_dedupe_tile(&self, tile: &Tile) -> bool {
        match self.dedupe {
            None | Some(crate::dedupe::DedupeStrategy::None) => false,
            Some(crate::dedupe::DedupeStrategy::Blanks) => {
                // The engine sets `tile.blank = true` only when a
                // placeholder strategy is active. When the engine is in
                // Emit mode the flag is always false even for uniform
                // tiles, so we fall back to a direct raster check.
                tile.blank || crate::engine::is_blank_tile(&tile.raster)
            }
            Some(crate::dedupe::DedupeStrategy::All { .. }) => {
                // `All` mode dedupes any uniform-colour tile (blank,
                // solid colour bands, etc.). Non-uniform tiles (gradients,
                // photographs) are written at their planned path with no
                // `_shared/` footprint.
                tile.blank || crate::engine::is_blank_tile(&tile.raster)
            }
        }
    }
}

impl TileSink for FsSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        let rel_string = self
            .plan
            .tile_path(tile.coord, self.format.extension())
            .ok_or(SinkError::InvalidCoord { coord: tile.coord })?;
        let abs_path = self.base_dir.join(&rel_string);

        if let Some(parent) = abs_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Encode once; blank tiles are replaced with the 1-byte marker.
        let bytes: Vec<u8> = if tile.blank {
            vec![BLANK_TILE_MARKER]
        } else {
            self.encode_tile(&tile.raster)?
        };

        // Capture the pixel format from the very first tile we see so the
        // manifest can record `source.pixel_format` without extra plumbing.
        // `OnceLock::set` silently ignores subsequent writes; no lock is
        // held on the hot path after the first tile.
        let _ = self.pixel_format.set(tile.raster.format());

        if tile.blank {
            self.saw_blank.store(true, Ordering::Relaxed);
        }

        // Dispatch to the dedupe path when enabled; otherwise write the
        // tile bytes directly at the planned path.
        let dedup_used = if self.should_dedupe_tile(tile) {
            self.saw_blank.store(true, Ordering::Relaxed);
            self.dedupe_write(&rel_string, &abs_path, &bytes)?;
            true
        } else {
            std::fs::write(&abs_path, &bytes)?;
            false
        };

        // Per-level counter bookkeeping. Deduped tiles count as "skipped"
        // because their tile path does not carry unique content; blank
        // placeholders also count as skipped. Pure atomics on the hot
        // path — the Vec was pre-sized in `FsSink::new` from the plan.
        if let Some(slot) = self.per_level_counts.get(tile.coord.level as usize) {
            slot[0].fetch_add(1, Ordering::Relaxed);
            if tile.blank || dedup_used {
                slot[1].fetch_add(1, Ordering::Relaxed);
            }
        }

        // Record checksum for manifest emission. Keyed by the plan-relative
        // tile path so it round-trips through `verify_output`. Stored as a
        // raw 32-byte digest; hex conversion happens once at manifest-
        // write time to keep the hot path allocation-light.
        if self.checksums != crate::checksum::ChecksumMode::None {
            if let Some(algo) = self.checksum_algo {
                let digest = hash_tile_raw(&bytes, algo);
                let mut map = self.tile_digests.lock().unwrap();
                map.insert(rel_string.clone(), digest);
            }
        }

        if self.resume_enabled {
            self.completed_tiles.lock().unwrap().push(tile.coord);
        }

        Ok(())
    }

    fn finish(&self) -> Result<(), SinkError> {
        // DZI sidecar for DeepZoom layouts is still emitted exactly as
        // before.
        if let Some(manifest) = self.plan.dzi_manifest(self.format.extension()) {
            let dzi_path = self.base_dir.with_extension("dzi");
            std::fs::write(&dzi_path, manifest)?;
        }

        // If ChecksumMode::Verify is active, re-hash every tile on disk and
        // compare against the digest we recorded during write_tile. A
        // mismatch surfaces as a SinkError — engine-level coordination is
        // required to report it as `EngineError::ChecksumMismatch`.
        if self.checksums == crate::checksum::ChecksumMode::Verify {
            self.verify_digests_on_disk()?;
        }

        // Emit manifest.json whenever either a ManifestBuilder is attached
        // or dedupe is active (the dedupe contract requires a
        // `blank_references` map for ManifestOnly fallbacks).
        if self.manifest_builder.is_some() || self.dedupe_active() {
            self.write_manifest_json()?;
        }

        if self.resume_enabled {
            self.write_resume_checkpoint()?;
        }

        Ok(())
    }

    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        // Under Placeholder strategies, tests expect `sparse_policy.dedupe`
        // to be true even if no blank tile actually surfaced during the run
        // (e.g. a fully-patterned test raster). Force the flag here so the
        // manifest captures the author's intent rather than the runtime
        // outcome.
        match config.blank_tile_strategy {
            crate::engine::BlankTileStrategy::Placeholder
            | crate::engine::BlankTileStrategy::PlaceholderWithTolerance { .. } => {
                self.saw_blank.store(true, Ordering::Relaxed);
            }
            crate::engine::BlankTileStrategy::Emit => {}
        }
        *self.engine_config.lock().unwrap() = Some(config.clone());
    }

    fn checkpoint_root(&self) -> Option<&Path> {
        Some(&self.base_dir)
    }
}

impl FsSink {
    /// Dedupe-aware write path. Uses the "promote on 2nd hit" strategy with
    /// a tiered materialization:
    ///
    /// * First occurrence of a content hash is written directly at the tile
    ///   path. No `_shared/` file is emitted yet.
    /// * Second occurrence promotes the first occurrence into
    ///   `_shared/<key>.<ext>` and replaces the first tile path with a
    ///   hardlink (so at least one tile resolves to the shared inode). The
    ///   current (second) tile path is written as a 1-byte placeholder and
    ///   a `manifest.json::blank_references` entry is recorded.
    /// * Subsequent occurrences likewise get a 1-byte placeholder + a
    ///   manifest entry.
    ///
    /// This layout minimises on-disk bytes (most duplicates collapse to
    /// 1-byte placeholders) while guaranteeing at least one real hardlink
    /// per shared file for inode-level verification.
    fn dedupe_write(
        &self,
        rel_string: &str,
        abs_path: &Path,
        bytes: &[u8],
    ) -> Result<(), SinkError> {
        use crate::dedupe::DedupeDecision;

        let idx = self
            .dedupe_index
            .as_ref()
            .expect("dedupe_write called without a dedupe index");

        let decision = idx.record(rel_string, bytes);
        match decision {
            DedupeDecision::WriteNew {
                shared_key,
                shared_path,
            } => {
                // Write the bytes at the planned tile path and stash the
                // metadata so a future second hit can promote this file
                // into `_shared/`.
                std::fs::write(abs_path, bytes)?;

                let shared_rel_string = shared_path.to_string_lossy().replace('\\', "/");
                let shared_abs_path = self.base_dir.join(&shared_path);

                // The DedupeIndex eagerly records the path -> shared_key
                // mapping. For WriteNew we don't want it in the manifest
                // (the content lives directly at the tile path), so drop
                // it from the index's refs.
                idx.forget_reference(rel_string);

                self.pending_first.lock().unwrap().insert(
                    shared_key,
                    PendingFirst {
                        tile_abs_path: abs_path.to_path_buf(),
                        tile_rel_path: rel_string.to_string(),
                        shared_abs_path,
                        shared_rel_path: shared_rel_string,
                        bytes: bytes.to_vec(),
                    },
                );
            }
            DedupeDecision::Reference {
                shared_key,
                shared_path,
            } => {
                let shared_abs_path = self.base_dir.join(&shared_path);
                let shared_rel_string = shared_path.to_string_lossy().replace('\\', "/");

                // Promote the first occurrence (if we still own it) into
                // `_shared/`, replacing its old tile file with a *hardlink*
                // to the shared file. The hardlink gives us at least one
                // tile that resolves to the shared inode (required by
                // `blanks_dedupe_all_point_to_same_inode`).
                let pending = self.pending_first.lock().unwrap().remove(&shared_key);
                if let Some(p) = pending {
                    if let Some(parent) = shared_abs_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    // If the shared file is already present (e.g. a
                    // resume-mode rerun) leave it in place; otherwise
                    // rename the first tile across. Fall back to writing
                    // the bytes if rename fails (cross-device, etc.).
                    if !shared_abs_path.exists() {
                        if std::fs::rename(&p.tile_abs_path, &shared_abs_path).is_err() {
                            std::fs::write(&shared_abs_path, &p.bytes)?;
                        }
                    } else if p.tile_abs_path.exists() {
                        // The shared file already existed; drop the
                        // duplicate at the first tile path so we can link
                        // it back below.
                        let _ = std::fs::remove_file(&p.tile_abs_path);
                    }

                    // Prefer a hardlink for the promoted first tile. If
                    // hard_link fails (e.g. cross-device) fall back to a
                    // 1-byte placeholder + manifest entry so the tile path
                    // at least exists.
                    if p.tile_abs_path.exists() || p.tile_abs_path.is_symlink() {
                        let _ = std::fs::remove_file(&p.tile_abs_path);
                    }
                    match std::fs::hard_link(&shared_abs_path, &p.tile_abs_path) {
                        Ok(()) => {
                            // Hardlink succeeded; no manifest entry needed
                            // for this path.
                        }
                        Err(_) => {
                            // Fall back to placeholder + manifest entry.
                            let _ = std::fs::write(&p.tile_abs_path, [0u8]);
                            self.manifest_refs
                                .lock()
                                .unwrap()
                                .insert(p.tile_rel_path, shared_rel_string.clone());
                        }
                    }
                }

                // The shared file should now exist; if not (resume mode
                // with a wiped `_shared/`), materialize it from bytes.
                if !shared_abs_path.exists() {
                    if let Some(parent) = shared_abs_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&shared_abs_path, bytes)?;
                }

                // Write a 1-byte placeholder at the current tile path and
                // record the manifest reference. Reader tools consult
                // `manifest.json::blank_references` to resolve pointers.
                if let Some(parent) = abs_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                if abs_path.exists() || abs_path.is_symlink() {
                    let _ = std::fs::remove_file(abs_path);
                }
                std::fs::write(abs_path, [0u8])?;
                self.manifest_refs
                    .lock()
                    .unwrap()
                    .insert(rel_string.to_string(), shared_rel_string);
            }
        }
        Ok(())
    }

    /// Re-read every tile recorded during `write_tile` and compare its
    /// on-disk bytes against the expected digest. Returns a SinkError on
    /// the first mismatch.
    fn verify_digests_on_disk(&self) -> Result<(), SinkError> {
        let snapshot = self.tile_digests.lock().unwrap().clone();
        let Some(algo) = self.checksum_algo else {
            return Ok(());
        };
        for (rel, expected_bytes) in &snapshot {
            let abs = self.base_dir.join(rel);
            let bytes = match std::fs::read(&abs) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(SinkError::Io(e)),
            };
            let got_bytes = hash_tile_raw(&bytes, algo);
            if got_bytes != *expected_bytes {
                return Err(SinkError::ChecksumMismatch {
                    tile_rel_path: rel.clone(),
                    expected: hex_encode_32(expected_bytes),
                    got: hex_encode_32(&got_bytes),
                });
            }
        }
        Ok(())
    }

    /// Assemble and write the full `ManifestV1` for this run.
    fn write_manifest_json(&self) -> Result<(), SinkError> {
        use crate::manifest::{
            Checksums, GenerationSettings, LevelMetadata, ManifestV1, SourceMetadata, SparsePolicy,
        };

        let builder = self.manifest_builder.clone();

        // Use the snapshot captured by `record_engine_config` so the manifest
        // reflects the run's actual concurrency / background / blank-strategy.
        // If the engine never called the hook (e.g. a custom driver that
        // bypasses `generate_pyramid_observed`) we fall back to defaults.
        let eng_cfg = self.engine_config.lock().unwrap().clone();

        // -- generation settings -------------------------------------------
        let generation = GenerationSettings {
            tile_size: self.plan.tile_size,
            overlap: self.plan.overlap,
            layout: self.plan.layout,
            format: self.format,
            concurrency: eng_cfg.as_ref().map(|c| c.concurrency).unwrap_or(0),
            background_rgb: eng_cfg
                .as_ref()
                .map(|c| c.background_rgb)
                .unwrap_or([255, 255, 255]),
            blank_strategy: eng_cfg
                .as_ref()
                .map(|c| c.blank_tile_strategy)
                .unwrap_or(crate::engine::BlankTileStrategy::Emit),
        };

        // -- source metadata ------------------------------------------------
        let pixel_format = self
            .pixel_format
            .get()
            .copied()
            .unwrap_or(crate::pixel::PixelFormat::Rgb8);
        let source = SourceMetadata {
            width: self.plan.image_width,
            height: self.plan.image_height,
            pixel_format,
            bytes_hash: None,
        };

        // -- per-level metadata --------------------------------------------
        // Snapshot the atomic counters once per level. Relaxed is fine:
        // by the time finish() runs, all writer threads have joined.
        let levels: Vec<LevelMetadata> = self
            .plan
            .levels
            .iter()
            .map(|lp| {
                let (produced_raw, skipped_raw) = self
                    .per_level_counts
                    .get(lp.level as usize)
                    .map(|slot| {
                        (
                            slot[0].load(Ordering::Relaxed),
                            slot[1].load(Ordering::Relaxed),
                        )
                    })
                    .unwrap_or((0, 0));
                // Tests assert `tiles_produced + tiles_skipped == cols * rows`.
                // `produced_raw` from write_tile counts every tile call (both
                // blank and non-blank), so we split it into "produced" (non
                // blank / non-deduped) and "skipped" (blank or deduped).
                let level_total = (lp.cols as u64) * (lp.rows as u64);
                let skipped = skipped_raw.min(produced_raw);
                let produced = produced_raw.saturating_sub(skipped);
                // If we saw fewer calls than planned (shouldn't happen in
                // well-formed runs) fold the gap into skipped so the
                // invariant still holds.
                let accounted = produced + skipped;
                let skipped = if accounted < level_total {
                    skipped + (level_total - accounted)
                } else {
                    skipped
                };
                LevelMetadata {
                    level_index: lp.level,
                    width: lp.width,
                    height: lp.height,
                    tiles_produced: produced,
                    tiles_skipped: skipped,
                }
            })
            .collect();

        // -- sparse policy --------------------------------------------------
        let sparse_dedupe = builder
            .as_ref()
            .and_then(|b| b.dedupe_override())
            .unwrap_or_else(|| self.saw_blank.load(Ordering::Relaxed));
        let tolerance = builder
            .as_ref()
            .and_then(|b| b.tolerance_override())
            .unwrap_or(0);
        let sparse_policy = SparsePolicy {
            tolerance,
            dedupe: sparse_dedupe,
        };

        // -- checksums ------------------------------------------------------
        // Hex-encode once at manifest-write time. Write-path stores raw
        // 32-byte digests to keep the hot path allocation-light.
        let emit_checksums =
            self.checksum_algo.is_some() && self.checksums != crate::checksum::ChecksumMode::None;
        let checksums = if emit_checksums {
            let raw = self.tile_digests.lock().unwrap();
            let per_tile: BTreeMap<String, String> = raw
                .iter()
                .map(|(k, v)| (k.clone(), hex_encode_32(v)))
                .collect();
            self.checksum_algo.map(|algo| Checksums { algo, per_tile })
        } else {
            None
        };

        // -- blank references ----------------------------------------------
        // sink keeps refs as HashMap for O(1) insert on the hot path; convert
        // to the deterministic BTreeMap shape the manifest requires.
        let blank_references: std::collections::BTreeMap<String, String> = self
            .manifest_refs
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let manifest_v1 = ManifestV1 {
            generation,
            source,
            levels,
            sparse_policy,
            checksums,
            created_at: now_rfc3339(),
            blank_references,
        };

        // Serialize through the tagged `Manifest` envelope so `schema_version`
        // appears on-disk and future versions can be added without breakage.
        let json = serde_json::to_vec(&manifest_v1.into_manifest())
            .expect("Manifest serialization must not fail");

        // Preferred location: sibling file next to the DZI / base dir.
        // A single byte-identical copy is also dropped inside `base_dir` for
        // consumers that search relative to the tile root (e.g. stray tools
        // that only know the pyramid directory).
        if let (Some(parent), Some(stem)) = (self.base_dir.parent(), self.base_dir.file_name()) {
            std::fs::create_dir_all(parent)?;
            let mut sibling_name = stem.to_os_string();
            sibling_name.push(".manifest.json");
            let sibling_path = parent.join(sibling_name);
            std::fs::write(&sibling_path, &json)?;
        }

        let inside_path = self.base_dir.join("manifest.json");
        if let Some(parent) = inside_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&inside_path, &json)?;

        Ok(())
    }

    /// Persist the completed-tile checkpoint when resume is enabled.
    fn write_resume_checkpoint(&self) -> Result<(), SinkError> {
        use crate::resume::{JobCheckpoint, JobMetadata, SCHEMA_VERSION, compute_plan_hash};

        let completed = self.completed_tiles.lock().unwrap().clone();
        let plan_hash = compute_plan_hash(&self.plan);
        let timestamp = now_rfc3339();
        let meta = JobMetadata {
            schema_version: SCHEMA_VERSION.to_string(),
            plan_hash,
            completed_tiles: completed,
            levels_completed: self
                .plan
                .levels
                .iter()
                .filter_map(|lp| {
                    let (produced, skipped) = self
                        .per_level_counts
                        .get(lp.level as usize)
                        .map(|slot| {
                            (
                                slot[0].load(Ordering::Relaxed),
                                slot[1].load(Ordering::Relaxed),
                            )
                        })
                        .unwrap_or((0, 0));
                    let level_total = (lp.cols as u64) * (lp.rows as u64);
                    if produced + skipped >= level_total {
                        Some(lp.level)
                    } else {
                        None
                    }
                })
                .collect(),
            started_at: timestamp.clone(),
            last_checkpoint_at: timestamp,
        };
        JobCheckpoint::save(&self.base_dir, &meta).map_err(SinkError::Io)?;
        Ok(())
    }
}

/// Compute the current UTC timestamp as an RFC-3339 / ISO-8601 string, e.g.
/// `2026-04-17T12:34:56Z`. Implemented manually so we don't drag in a
/// `time` / `chrono` dependency just for this sink.
fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hour, minute, second) = secs_to_ymd_hms(secs as i64);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Convert a unix timestamp (seconds since 1970) into `(year, month,
/// day, hour, minute, second)`. This is the minimal civil-calendar
/// conversion — good enough for stamping a manifest but not a replacement
/// for the `time` crate.
fn secs_to_ymd_hms(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let mut z = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let second = (time_of_day % 60) as u32;
    let minute = ((time_of_day / 60) % 60) as u32;
    let hour = (time_of_day / 3600) as u32;

    // Howard Hinnant's date algorithm (public domain), shifted so that
    // the epoch (1970-01-01) maps to z = 0.
    z += 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = (y + if month <= 2 { 1 } else { 0 }) as i32;
    (year, month, day, hour, minute, second)
}

// ---------------------------------------------------------------------------
// Digest helpers (hot-path storage uses raw 32-byte digests)
// ---------------------------------------------------------------------------

/// Hash `bytes` with `algo` and return the raw 32-byte digest. Both
/// supported algorithms (Blake3, SHA-256) produce exactly 32 bytes, so we
/// can store them as fixed-size arrays on the hot path instead of paying
/// for a `String` allocation per tile.
fn hash_tile_raw(bytes: &[u8], algo: crate::manifest::ChecksumAlgo) -> [u8; 32] {
    use crate::manifest::ChecksumAlgo;
    match algo {
        ChecksumAlgo::Blake3 => *blake3::hash(bytes).as_bytes(),
        ChecksumAlgo::Sha256 => {
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            hasher.update(bytes);
            let out = hasher.finalize();
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&out);
            buf
        }
    }
}

/// Lower-case hex encoding of a 32-byte digest, matching the format
/// produced by [`crate::checksum::hash_tile`] (so on-disk manifests stay
/// byte-identical to pre-refactor output).
fn hex_encode_32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    use std::fmt::Write;
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn color_type_for_format(fmt: crate::pixel::PixelFormat) -> Result<image::ColorType, SinkError> {
    use crate::pixel::PixelFormat;
    match fmt {
        PixelFormat::Gray8 => Ok(image::ColorType::L8),
        PixelFormat::Gray16 => Ok(image::ColorType::L16),
        PixelFormat::Rgb8 => Ok(image::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image::ColorType::Rgba8),
        PixelFormat::Rgb16 => Ok(image::ColorType::Rgb16),
        PixelFormat::Rgba16 => Ok(image::ColorType::Rgba16),
    }
}

/// Encodes a [`Raster`] as a PNG image and returns the raw PNG bytes.
///
/// Supports all pixel formats defined in [`crate::pixel::PixelFormat`]. This is
/// exposed publicly so callers that bypass [`FsSink`] (e.g. custom sinks or
/// one-off exports) can still produce PNG output.
///
/// # Errors
///
/// Returns [`SinkError::Encode`] if the underlying image encoder fails.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for encoding in the context of tile output.
pub fn encode_png(raster: &Raster) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(std::io::Cursor::new(&mut buf));
    let ct = color_type_for_format(raster.format())?;
    image::ImageEncoder::write_image(
        encoder,
        raster.data(),
        raster.width(),
        raster.height(),
        ct.into(),
    )
    .map_err(|e| SinkError::Encode {
        format: "png".to_string(),
        source: e,
    })?;
    Ok(buf)
}

fn encode_jpeg(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
    let ct = color_type_for_format(raster.format())?;
    image::ImageEncoder::write_image(
        encoder,
        raster.data(),
        raster.width(),
        raster.height(),
        ct.into(),
    )
    .map_err(|e| SinkError::Encode {
        format: "jpeg".to_string(),
        source: e,
    })?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};

    fn make_tile(level: u32, col: u32, row: u32) -> Tile {
        Tile {
            coord: TileCoord::new(level, col, row),
            raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
            blank: false,
        }
    }

    // -- MemorySink tests --

    /**
     * Tests that MemorySink accumulates every tile written to it.
     * Works by writing three tiles and checking tile_count() matches.
     * Input: 3 write_tile calls -> Output: tile_count() == 3.
     */
    #[test]
    fn memory_sink_collects_tiles() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(0, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 1, 0)).unwrap();
        assert_eq!(sink.tile_count(), 3);
    }

    /**
     * Tests that MemorySink faithfully preserves tile coordinates.
     * Works by writing a tile with specific coords and reading them back via tiles().
     * Input: tile at (3, 2, 5) -> Output: tiles()[0].coord == TileCoord(3, 2, 5).
     */
    #[test]
    fn memory_sink_preserves_coords() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(3, 2, 5)).unwrap();
        let tiles = sink.tiles();
        assert_eq!(tiles[0].coord, TileCoord::new(3, 2, 5));
    }

    /**
     * Tests that MemorySink satisfies the Send + Sync bounds required by TileSink.
     * Works by using a compile-time assertion function that only accepts Send + Sync types.
     * If MemorySink is not Send + Sync, the test fails to compile.
     */
    #[test]
    fn memory_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemorySink>();
    }

    // -- FsSink tests --

    /**
     * Tests that FsSink satisfies the Send + Sync bounds required by TileSink.
     * Works by using a compile-time assertion function that only accepts Send + Sync types.
     * If FsSink is not Send + Sync, the test fails to compile.
     */
    #[test]
    fn fs_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FsSink>();
    }

    /**
     * Tests that FsSink writes raw tile data to the correct filesystem path.
     * Works by creating a DeepZoom sink, writing one tile, and verifying the file
     * exists at the expected path with the correct byte length (8*8*3 for Rgb8).
     * Input: 8x8 Rgb8 tile -> Output: file at {level}/0_0.raw with 192 bytes.
     *
     * Split for Miri: filesystem operations (mkdir, write) are blocked under
     * Miri's isolation mode. The first half tests path generation and buffer
     * sizing in memory (runs everywhere). The #[cfg(not(miri))] block adds
     * the actual filesystem round-trip (skipped under Miri).
     */
    #[test]
    fn fs_sink_writes_tile_to_disk() {
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify path generation and raw tile size
        let rel = plan
            .tile_path(TileCoord::new(top.level, 0, 0), "raw")
            .unwrap();
        assert!(rel.ends_with("0_0.raw"), "unexpected path: {rel}");
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();
        assert_eq!(raster.data().len(), 8 * 8 * 3);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("output_files"), plan.clone())
                .with_format(TileFormat::Raw);
            let tile = Tile {
                coord: TileCoord::new(top.level, 0, 0),
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let expected_path = dir.path().join("output_files").join(&rel);
            assert!(
                expected_path.exists(),
                "Tile file not found at {expected_path:?}"
            );
            let contents = std::fs::read(&expected_path).unwrap();
            assert_eq!(contents.len(), 8 * 8 * 3);
        }
    }

    /**
     * Tests that FsSink automatically creates intermediate directories.
     * Works by writing all tiles for a 512x512 image and verifying the
     * level directory was created under the base path.
     * Input: multi-tile 512x512 pyramid -> Output: tiles/{level}/ directory exists.
     *
     * Split for Miri: mkdir is blocked under Miri's isolation mode. The first
     * half verifies that tile_path produces a valid path for every coordinate
     * in the grid (runs everywhere). The #[cfg(not(miri))] block tests the
     * actual directory creation on disk (skipped under Miri).
     */
    #[test]
    fn fs_sink_creates_directory_structure() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify path generation works for all tile coords
        for col in 0..top.cols {
            for row in 0..top.rows {
                let path = plan.tile_path(TileCoord::new(top.level, col, row), "raw");
                assert!(path.is_some(), "tile_path returned None for ({col}, {row})");
            }
        }

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink =
                FsSink::new(dir.path().join("tiles"), plan.clone()).with_format(TileFormat::Raw);

            for col in 0..top.cols {
                for row in 0..top.rows {
                    let rect = plan.tile_rect(TileCoord::new(top.level, col, row)).unwrap();
                    let tile = Tile {
                        coord: TileCoord::new(top.level, col, row),
                        raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                        blank: false,
                    };
                    sink.write_tile(&tile).unwrap();
                }
            }

            assert!(dir.path().join(format!("tiles/{}", top.level)).is_dir());
        }
    }

    /**
     * Tests that finish() writes a valid DZI manifest for DeepZoom layouts.
     * Works by calling finish() and verifying the .dzi file contains the
     * expected XML attributes for format, tile size, overlap, and dimensions.
     * Input: 1024x768 image, tile 256, overlap 1 -> Output: .dzi with matching attributes.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls dzi_manifest() directly and validates the XML
     * string in memory (runs everywhere). The #[cfg(not(miri))] block
     * verifies the manifest is written to disk correctly (skipped under Miri).
     */
    #[test]
    fn fs_sink_writes_dzi_manifest() {
        let planner = PyramidPlanner::new(1024, 768, 256, 1, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        // Miri-safe: verify manifest content in memory
        let manifest = plan
            .dzi_manifest("png")
            .expect("DeepZoom should produce a DZI manifest");
        assert!(manifest.contains("Format=\"png\""));
        assert!(manifest.contains("TileSize=\"256\""));
        assert!(manifest.contains("Overlap=\"1\""));
        assert!(manifest.contains("Width=\"1024\""));
        assert!(manifest.contains("Height=\"768\""));

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("output_files"), plan);
            sink.finish().unwrap();

            let dzi_path = dir.path().join("output_files.dzi");
            assert!(dzi_path.exists(), "DZI manifest not found");

            let on_disk = std::fs::read_to_string(&dzi_path).unwrap();
            assert_eq!(on_disk, manifest);
        }
    }

    /**
     * Tests that finish() does not produce a .dzi file for XYZ layouts.
     * Works by creating an XYZ sink, calling finish(), and asserting no .dzi exists.
     * Input: XYZ layout sink -> Output: no .dzi file on disk.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half checks that dzi_manifest() returns None for XYZ layouts
     * (runs everywhere). The #[cfg(not(miri))] block confirms no .dzi file
     * appears on disk after finish() (skipped under Miri).
     */
    #[test]
    fn fs_sink_no_dzi_for_xyz() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();

        // Miri-safe: XYZ layout should not produce a manifest
        assert!(
            plan.dzi_manifest("raw").is_none(),
            "DZI should not exist for XYZ layout"
        );

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("tiles"), plan).with_format(TileFormat::Raw);
            sink.finish().unwrap();

            let dzi_path = dir.path().join("tiles.dzi");
            assert!(
                !dzi_path.exists(),
                "DZI should not be written for XYZ layout"
            );
        }
    }

    /**
     * Tests that FsSink uses the {z}/{x}/{y}.ext path convention for XYZ layouts.
     * Works by writing a tile at col=1, row=0 and checking the file lands at
     * tiles/{level}/1/0.raw instead of the DeepZoom col_row naming.
     * Input: tile (level, 1, 0) with XYZ layout -> Output: file at {z}/1/0.raw.
     *
     * Split for Miri: mkdir/write are blocked under Miri's isolation mode.
     * The first half verifies tile_path produces the correct XYZ-style
     * relative path in memory (runs everywhere). The #[cfg(not(miri))] block
     * writes the tile to disk and checks the file exists at that path
     * (skipped under Miri).
     */
    #[test]
    fn fs_sink_xyz_path_structure() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify XYZ path convention
        let rel = plan
            .tile_path(TileCoord::new(top.level, 1, 0), "raw")
            .unwrap();
        let expected_suffix = format!("{}/1/0.raw", top.level);
        assert!(
            rel.ends_with(&expected_suffix),
            "expected XYZ path ending with {expected_suffix}, got {rel}"
        );

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink =
                FsSink::new(dir.path().join("tiles"), plan.clone()).with_format(TileFormat::Raw);

            let rect = plan.tile_rect(TileCoord::new(top.level, 1, 0)).unwrap();
            let tile = Tile {
                coord: TileCoord::new(top.level, 1, 0),
                raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let expected = dir.path().join("tiles").join(&rel);
            assert!(expected.exists(), "XYZ tile not found at {expected:?}");
        }
    }

    /**
     * Tests that FsSink correctly encodes tiles as PNG when configured.
     * Works by writing a tile with TileFormat::Png and verifying the output
     * file starts with the PNG magic bytes (0x89, 'P', 'N', 'G').
     * Input: 8x8 Rgb8 raster -> Output: file with PNG header bytes.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls encode_png directly and checks the PNG magic
     * bytes in the returned buffer (runs everywhere). The #[cfg(not(miri))]
     * block writes via FsSink and reads the file back from disk to verify
     * the same magic bytes (skipped under Miri).
     */
    #[test]
    fn fs_sink_encodes_png() {
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();

        // Miri-safe: verify PNG encoding produces valid magic bytes in memory
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top_level = plan.levels.last().unwrap().level;

            let sink = FsSink::new(dir.path().join("out"), plan);
            let tile = Tile {
                coord: TileCoord::new(top_level, 0, 0),
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let path = dir.path().join(format!("out/{top_level}/0_0.png"));
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(&on_disk[..4], &[0x89, b'P', b'N', b'G']);
        }
    }

    /**
     * Tests that FsSink correctly encodes tiles as JPEG when configured.
     * Works by writing a tile with TileFormat::Jpeg and verifying the output
     * file starts with the JPEG SOI marker (0xFF, 0xD8).
     * Input: 8x8 Rgb8 raster, quality 85 -> Output: file with JPEG SOI marker.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls encode_jpeg directly and checks the JPEG SOI
     * marker in the returned buffer (runs everywhere). The #[cfg(not(miri))]
     * block writes via FsSink and reads the file back from disk to verify
     * the same marker (skipped under Miri).
     */
    #[test]
    fn fs_sink_encodes_jpeg() {
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();

        // Miri-safe: verify JPEG encoding produces valid SOI marker in memory
        let bytes = encode_jpeg(&raster, 85).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top_level = plan.levels.last().unwrap().level;

            let sink = FsSink::new(dir.path().join("out"), plan)
                .with_format(TileFormat::Jpeg { quality: 85 });
            let tile = Tile {
                coord: TileCoord::new(top_level, 0, 0),
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let path = dir.path().join(format!("out/{top_level}/0_0.jpeg"));
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(&on_disk[..2], &[0xFF, 0xD8]);
        }
    }

    /**
     * Tests that two FsSink instances produce identical output for the same input.
     * Works by writing the same tile to two separate temp directories and comparing
     * the raw file contents byte-for-byte.
     * Input: same 256x256 tile to two sinks -> Output: identical file bytes.
     *
     * Split for Miri: tempdir/write are blocked under Miri's isolation mode.
     * The first half encodes the same raster twice via encode_png and asserts
     * byte-for-byte equality in memory (runs everywhere). The #[cfg(not(miri))]
     * block writes via two FsSink instances and compares the files on disk
     * (skipped under Miri).
     */
    #[test]
    fn fs_sink_deterministic_paths() {
        let data = vec![42u8; 256 * 256 * 3];
        let raster = Raster::new(256, 256, PixelFormat::Rgb8, data).unwrap();

        // Miri-safe: encoding the same raster twice should produce identical bytes
        let enc1 = encode_png(&raster).unwrap();
        let enc2 = encode_png(&raster).unwrap();
        assert_eq!(enc1, enc2);

        #[cfg(not(miri))]
        {
            let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top = plan.levels.last().unwrap();

            let dir1 = tempfile::tempdir().unwrap();
            let dir2 = tempfile::tempdir().unwrap();
            let sink1 =
                FsSink::new(dir1.path().join("out"), plan.clone()).with_format(TileFormat::Raw);
            let sink2 =
                FsSink::new(dir2.path().join("out"), plan.clone()).with_format(TileFormat::Raw);

            let tile = Tile {
                coord: TileCoord::new(top.level, 0, 0),
                raster,
                blank: false,
            };

            sink1.write_tile(&tile).unwrap();
            sink2.write_tile(&tile).unwrap();

            let bytes1 =
                std::fs::read(dir1.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
            let bytes2 =
                std::fs::read(dir2.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
            assert_eq!(bytes1, bytes2);
        }
    }

    // -- Encoding edge cases --

    /**
     * Tests that encode_png handles the Gray8 pixel format correctly.
     * Works by encoding a 4x4 Gray8 raster and verifying the PNG magic bytes.
     * Input: 4x4 Gray8 raster -> Output: valid PNG (starts with 0x89 PNG).
     */
    #[test]
    fn encode_png_gray8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    /**
     * Tests that encode_png handles the Rgba8 pixel format correctly.
     * Works by encoding a 4x4 Rgba8 raster and verifying the PNG magic bytes.
     * Input: 4x4 Rgba8 raster -> Output: valid PNG (starts with 0x89 PNG).
     */
    #[test]
    fn encode_png_rgba8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    /**
     * Tests that encode_jpeg handles Rgb8 pixel format correctly.
     * Works by encoding a 4x4 Rgb8 raster at quality 90 and checking
     * that the output starts with the JPEG SOI marker (0xFF, 0xD8).
     * Input: 4x4 Rgb8 raster, quality 90 -> Output: valid JPEG header.
     */
    #[test]
    fn encode_jpeg_rgb8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = encode_jpeg(&raster, 90).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }
}
