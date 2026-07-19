use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
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

#[cfg(feature = "object-store-sink")]
pub use crate::sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};

#[cfg(feature = "packfile")]
pub use crate::sink_packfile::{PackfileFormat, PackfileSink, ZipSink};

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
    /// A sink operation is not implemented in this build because the transport
    /// or capability it needs was not compiled in. Distinct from
    /// [`SinkError::Other`] so callers can tell "unsupported in this build"
    /// apart from a runtime failure, and so a stub cannot masquerade as a
    /// successful-but-empty result. The payload names the operation and what
    /// is missing.
    #[error("unsupported sink operation: {0}")]
    Unsupported(String),
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
    /// A tile whose digest was recorded during `write_tile` had no
    /// corresponding file on disk at verification time. A recorded tile that
    /// vanished (deleted, or never durably written) is a verification
    /// failure rather than something to skip silently, unless it is a
    /// manifest-referenced blank whose content lives in `_shared/` (issue
    /// #93).
    #[error("recorded tile missing from disk: {tile_rel_path}")]
    MissingTile { tile_rel_path: String },
    /// Persisting the resume checkpoint failed while writing a tile through the
    /// resume-aware sink wrapper. A sink can only surface a [`SinkError`], so
    /// this variant carries the underlying [`crate::resume::ResumeError`]
    /// verbatim; the engine promotes it back to
    /// [`crate::engine::EngineError::ResumeFailed`] so a checkpoint failure is
    /// reported with the same variant regardless of code path (issue #140).
    #[error("checkpoint write failed: {0}")]
    Checkpoint(#[source] crate::resume::ResumeError),
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
/// # Write-order independence
///
/// **The order in which `write_tile` is called is not guaranteed.** Under the
/// MapReduce engine with `tile_concurrency > 0`, worker threads extract tiles in
/// parallel and the consumer writes each tile as it arrives on the bounded
/// channel — i.e. in channel-arrival order, not row-major, and interleaved
/// arbitrarily across runs. A sink whose *output* depends on call order (byte
/// layout of a pack/archive that concatenates tiles in arrival sequence, an
/// append-only log keyed on position, etc.) will therefore produce run-to-run
/// different results at `tile_concurrency > 0`.
///
/// Implementations must treat each `write_tile` as an independent, commutative
/// placement keyed by [`Tile::coord`]; the produced pyramid must be identical
/// regardless of the order in which the calls arrive. All in-tree sinks
/// ([`FsSink`], [`MemorySink`], the object-store and packfile sinks) satisfy
/// this because they place each tile by its coordinate rather than by arrival
/// position.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for filesystem sink integration tests and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the CLI wires up a sink.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid)
pub trait TileSink: Send + Sync {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError>;
    fn finish(&self) -> Result<(), SinkError> {
        Ok(())
    }

    /// Transparent-decorator hook: a sink that merely wraps another sink
    /// returns `Some(inner)` here.
    ///
    /// Every engine-bookkeeping method below (`record_engine_config`,
    /// `sink_retry_count`, `sink_skipped_due_to_failure`, `note_sink_skipped`,
    /// `checkpoint_root`, `init_level_count`, `content_format`,
    /// `applies_retry_policy`) has a default that forwards through this hook.
    /// A wrapper therefore only has to override `inner_sink` — and any state it
    /// genuinely owns (e.g. a [`RetryingSink`]'s own retry counter) — instead
    /// of forwarding every bookkeeping method by hand. That removes the
    /// silent-data-loss trap where a wrapper that forgets to forward one method
    /// quietly drops a retry count, the manifest config, the checkpoint root,
    /// or the on-disk format (issue #137).
    ///
    /// Terminal sinks return `None` (the default), which collapses every
    /// bookkeeping default back to its previous no-op / `0` / `None` value, so
    /// leaf-sink behaviour is unchanged.
    fn inner_sink(&self) -> Option<&dyn TileSink> {
        None
    }

    /// Engine hook: forward a snapshot of the active [`crate::engine::EngineConfig`]
    /// into the sink so it can populate the manifest's generation settings and
    /// sparse-policy fields. The default forwards to [`TileSink::inner_sink`]
    /// (a no-op for terminal sinks); only sinks that emit manifests (e.g.
    /// [`FsSink`]) need to override.
    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        if let Some(inner) = self.inner_sink() {
            inner.record_engine_config(config);
        }
    }

    /// Engine hook: when the sink (or a wrapper around it) keeps an internal
    /// retry counter, expose the running total so the engine can include it
    /// in [`crate::engine::EngineResult::retry_count`]. The default forwards to
    /// [`TileSink::inner_sink`], returning `0` for terminal sinks.
    fn sink_retry_count(&self) -> u64 {
        self.inner_sink()
            .map_or(0, |inner| inner.sink_retry_count())
    }

    /// Engine hook: when the sink (or a wrapper around it) keeps an internal
    /// skip counter, expose the running total so the engine can include it in
    /// [`crate::engine::EngineResult::skipped_due_to_failure`]. The default
    /// forwards to [`TileSink::inner_sink`], returning `0` for terminal sinks.
    fn sink_skipped_due_to_failure(&self) -> u64 {
        self.inner_sink()
            .map_or(0, |inner| inner.sink_skipped_due_to_failure())
    }

    /// Engine hook: bump the skip counter by one, used by the engine when a
    /// `FailurePolicy::RetryThenSkip` tile is dropped. The default forwards to
    /// [`TileSink::inner_sink`] (a no-op for terminal sinks).
    fn note_sink_skipped(&self) {
        if let Some(inner) = self.inner_sink() {
            inner.note_sink_skipped();
        }
    }

    /// Engine hook: the on-disk root where the checkpoint file
    /// `.libviprs-job.json` should live. The default forwards to
    /// [`TileSink::inner_sink`]; sinks that do not write to the filesystem
    /// return `None`.
    fn checkpoint_root(&self) -> Option<&Path> {
        self.inner_sink().and_then(|inner| inner.checkpoint_root())
    }

    /// Engine hook: arm durability tracking for a checkpointed run.
    ///
    /// The engine builder calls this once, before the run, whenever it stands
    /// up a [`CheckpointState`](crate::engine) for an on-disk sink. It tells the
    /// sink to record every freshly-written tile path so that the durability
    /// barrier [`TileSink::sync_pending`] — invoked by the engine's checkpoint
    /// writer before it certifies a checkpoint delta — can fsync exactly those
    /// files. Sinks driven without a checkpoint (a plain, non-resume run) are
    /// never armed and pay no per-tile bookkeeping.
    ///
    /// The default forwards to [`TileSink::inner_sink`] (a no-op for terminal
    /// sinks that keep no durability state).
    fn arm_durability_tracking(&self) {
        if let Some(inner) = self.inner_sink() {
            inner.arm_durability_tracking();
        }
    }

    /// Durability barrier (issue #122 / #273): make the bytes of every tile
    /// written so far durable (fsync) so a checkpoint that is about to certify
    /// them never records tiles whose bytes are still only in the page cache.
    ///
    /// The engine's [`CheckpointState`](crate::engine) — the single checkpoint
    /// authority — calls this immediately before it appends a checkpoint delta
    /// to the segment log and publishes the header. A crash after the checkpoint
    /// but before this barrier would otherwise leave the certified checkpoint
    /// pointing at tiles whose bytes never reached stable storage, and resume
    /// would skip those coordinates forever.
    ///
    /// The default forwards to [`TileSink::inner_sink`], bottoming out at `Ok`
    /// for terminal sinks with no durability concept (in-memory sinks, test
    /// doubles). External `TileSink` implementations therefore keep compiling
    /// and behaving unchanged without overriding it.
    fn sync_pending(&self) -> Result<(), SinkError> {
        match self.inner_sink() {
            Some(inner) => inner.sync_pending(),
            None => Ok(()),
        }
    }

    /// Engine hook: tell the sink how many pyramid levels will appear in
    /// this run, so sinks that keep per-level counters can pre-size their
    /// backing storage before the tile loop starts. Default is a no-op.
    /// [`FsSink`] already sizes its counters from the plan in
    /// [`FsSink::new`], so calling this is idempotent there. The default
    /// forwards to [`TileSink::inner_sink`] (a no-op for terminal sinks).
    fn init_level_count(&self, levels: usize) {
        if let Some(inner) = self.inner_sink() {
            inner.init_level_count(levels);
        }
    }

    /// Engine hook: the tile encoding the sink writes, when it has one.
    ///
    /// Folded into the resume plan hash so that resuming a checkpoint with a
    /// changed output format is rejected instead of silently mixing formats
    /// on disk (see [`crate::resume::compute_plan_hash`]). Sinks that do not
    /// commit to a single on-disk format return `None`. The default forwards
    /// to [`TileSink::inner_sink`], so a transparent wrapper exposes the inner
    /// sink's format without forwarding this method by hand.
    fn content_format(&self) -> Option<TileFormat> {
        self.inner_sink().and_then(|inner| inner.content_format())
    }

    /// Engine hook: whether this sink (or a wrapper around it) already runs
    /// its own retry loop inside `write_tile`.
    ///
    /// [`crate::EngineBuilder`] consults this before automatically wrapping a
    /// sink in [`crate::retry::RetryingSink`] for a configured retry policy,
    /// so a caller that pre-wrapped their sink in `RetryingSink` is not
    /// double-wrapped (which would inflate `retry_count` and double-count
    /// `skipped_due_to_failure`). Default is `false`; `RetryingSink` overrides
    /// to `true`, and transparent decorators forward the inner sink's answer.
    /// The default forwards to [`TileSink::inner_sink`] (returning `false` for
    /// terminal sinks), so a plain wrapper around a `RetryingSink` reports
    /// `true` automatically and is not re-wrapped.
    fn applies_retry_policy(&self) -> bool {
        self.inner_sink()
            .is_some_and(|inner| inner.applies_retry_policy())
    }

    /// Engine hook (issue #272): rebuild the sink-side manifest / dedupe /
    /// checksum state a *pre-crash* tile contributes, WITHOUT advancing the
    /// resume checkpoint.
    ///
    /// On resume, [`ResumeAwareSink`](crate::resume) skips the coordinates the
    /// checkpoint already records — they were durably written before the crash.
    /// But the sink's per-run manifest state (`manifest_refs`, `tile_digests`,
    /// the `DedupeIndex`, per-level counts) starts empty, and
    /// [`TileSink::finish`] rebuilds `manifest.json` from it. Without
    /// reconstructing that state a resumed dedupe/checksum run would overwrite
    /// the manifest with a view that omits every pre-crash placeholder and
    /// truncates the checksum table — silent, reader-visible data corruption.
    ///
    /// The engine calls this for each already-completed coordinate, passing the
    /// re-rendered tile, so the sink reconstructs exactly the state an
    /// uninterrupted run would hold; a resumed [`TileSink::finish`] then
    /// reproduces byte-identical output. Implementations MUST NOT advance any
    /// resume checkpoint from here — the caller deliberately does not mark these
    /// coordinates as newly completed. The default forwards to
    /// [`TileSink::inner_sink`], bottoming out at `Ok` for terminal sinks that
    /// keep no manifest state (so external impls keep compiling and a plain,
    /// non-dedupe/checksum resume still short-circuits skipped tiles).
    fn seed_completed_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        match self.inner_sink() {
            Some(inner) => inner.seed_completed_tile(tile),
            None => Ok(()),
        }
    }
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
    fn inner_sink(&self) -> Option<&dyn TileSink> {
        (**self).inner_sink()
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
    fn arm_durability_tracking(&self) {
        (**self).arm_durability_tracking()
    }
    fn sync_pending(&self) -> Result<(), SinkError> {
        (**self).sync_pending()
    }
    fn init_level_count(&self, levels: usize) {
        (**self).init_level_count(levels)
    }
    fn content_format(&self) -> Option<TileFormat> {
        (**self).content_format()
    }
    fn applies_retry_policy(&self) -> bool {
        (**self).applies_retry_policy()
    }
    fn seed_completed_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        (**self).seed_completed_tile(tile)
    }
}

/// Forwarding impl so `&T` satisfies [`TileSink`] wherever `T` does.
///
/// Parallels the [`Box<T>`] impl above. Lets callers feed the
/// [`EngineBuilder`](crate::EngineBuilder) a borrowed sink when they need
/// to keep ownership (e.g. the CLI, which uses the same sink for both the
/// builder path and the resumable free function):
///
/// ```ignore
/// let sink = FsSink::new(dir, plan);
/// EngineBuilder::new(&raster, plan.clone(), &sink).run()?;
/// // `sink` still usable here.
/// ```
impl<T: TileSink + ?Sized> TileSink for &T {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        (*self).write_tile(tile)
    }
    fn finish(&self) -> Result<(), SinkError> {
        (*self).finish()
    }
    fn inner_sink(&self) -> Option<&dyn TileSink> {
        (*self).inner_sink()
    }
    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        (*self).record_engine_config(config)
    }
    fn sink_retry_count(&self) -> u64 {
        (*self).sink_retry_count()
    }
    fn sink_skipped_due_to_failure(&self) -> u64 {
        (*self).sink_skipped_due_to_failure()
    }
    fn note_sink_skipped(&self) {
        (*self).note_sink_skipped()
    }
    fn checkpoint_root(&self) -> Option<&Path> {
        (*self).checkpoint_root()
    }
    fn arm_durability_tracking(&self) {
        (*self).arm_durability_tracking()
    }
    fn sync_pending(&self) -> Result<(), SinkError> {
        (*self).sync_pending()
    }
    fn init_level_count(&self, levels: usize) {
        (*self).init_level_count(levels)
    }
    fn content_format(&self) -> Option<TileFormat> {
        (*self).content_format()
    }
    fn applies_retry_policy(&self) -> bool {
        (*self).applies_retry_policy()
    }
    fn seed_completed_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        (*self).seed_completed_tile(tile)
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
        crate::poison::recover(&self.tiles).clone()
    }

    pub fn tile_count(&self) -> usize {
        crate::poison::recover(&self.tiles).len()
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl TileSink for MemorySink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        crate::poison::recover(&self.tiles).push(CollectedTile {
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
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-format)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TileFormat {
    Png,
    /// JPEG-encoded tiles. The `quality` knob trades off filesize against
    /// visual fidelity (1–100, higher = better quality, larger files).
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-quality)
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
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-sink)
///
/// `Debug` is implemented manually because the internal
/// [`DedupeIndex`](crate::dedupe::DedupeIndex) does not derive `Debug`.
///
/// # Lock discipline
///
/// This type does **not** have a totally-ordered lock hierarchy; it has a
/// two-level rule that is simpler and stronger:
///
/// * `dedupe_promote` is the sole **outer** lock. It guards the whole
///   promote-on-2nd-hit critical section and is only ever taken at the top of
///   [`FsSink::dedupe_write`], never while a field mutex is already held. It is
///   *sharded by content digest* (see [`DEDUPE_PROMOTE_SHARDS`] and
///   [`FsSink::promote_shard`]): all occurrences of a given content map to the
///   same shard — preserving the per-key atomicity the at-least-one-hardlink
///   invariant requires (issue #111) — while distinct content maps to
///   (usually) distinct shards, so tiles of different content no longer
///   serialise on one process-wide lock (issue #296). Exactly one shard is
///   taken per `dedupe_write` call.
/// * Every other mutex — `tile_digests`, `manifest_refs`, `pending_first`,
///   `unsynced_tiles`, `validated_shared`, `engine_config` — is a **leaf**
///   lock. A leaf lock is held only long enough to read, mutate, or snapshot
///   its map/vec (the read paths clone or `mem::take` and release
///   immediately), and is **never** held while acquiring any other leaf lock,
///   a `dedupe_promote` shard, or while calling back into `self`.
///
/// The consequence — and the actual invariant to preserve — is that **at most
/// one leaf lock is ever held on a thread at a time**. The only legal nesting
/// is a `dedupe_promote` shard wrapping a single leaf lock inside
/// `dedupe_write`; no leaf-over-leaf nesting exists, so there is no lock cycle
/// and no AB-BA deadlock is reachable. Because a thread only ever holds one
/// shard at a time (taken at the top of `dedupe_write` and dropped at its end),
/// sharding cannot introduce a shard-over-shard cycle either. `pixel_format`
/// (a `OnceLock`) and `per_level_counts` (atomics) take no mutex and are
/// irrelevant to this rule.
///
/// Field mutexes are acquired through [`FsSink::lock_leaf`], which in debug
/// builds trips a panic the instant a second leaf lock is taken while one is
/// still held — turning an accidental nesting into an immediate, local
/// failure instead of a latent deadlock (issue #112).
///
/// # Sharded promote lock
///
/// The `dedupe_promote` outer lock is striped across
/// [`DEDUPE_PROMOTE_SHARDS`] shards, indexed by a hash of the tile content
/// (see [`FsSink::promote_shard`]). All occurrences of a given content select
/// the same shard, so the per-key atomicity issue #111 relies on is preserved;
/// distinct content selects (usually) distinct shards, so it no longer
/// serialises on one process-wide lock (issue #296).
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
    /// Serialises the dedupe "promote on 2nd hit" critical section so the
    /// index decision, the tile write, the `pending_first` registration and
    /// the promote/link steps form a single atomic unit. Without it a
    /// concurrent duplicate that receives `Reference` could run its
    /// `pending_first.remove` before the first writer's `pending_first.insert`,
    /// silently skipping promotion and breaking the at-least-one-hardlink
    /// invariant (issue #111). This is the outermost lock (see the
    /// type-level `# Lock discipline`); it is only ever taken at the top of
    /// [`FsSink::dedupe_write`] and never while holding a leaf mutex.
    ///
    /// Sharded across [`DEDUPE_PROMOTE_SHARDS`] entries and indexed by content
    /// digest (see [`FsSink::promote_shard`]): occurrences of the same content
    /// hash always take the same shard — so the promote-on-2nd-hit sequence
    /// stays atomic per key and the at-least-one-hardlink invariant holds —
    /// while tiles of distinct content take (usually) distinct shards and no
    /// longer serialise on a single process-wide lock (issue #296).
    dedupe_promote: [Mutex<()>; DEDUPE_PROMOTE_SHARDS],
    /// Shared keys whose `_shared/<key>` blob has already been materialised or
    /// revalidated during this run. Once a key is present, the expensive
    /// [`FsSink::shared_blob_valid`] full-file read + rehash is skipped for
    /// every later duplicate of that content, so a shared blob is read and
    /// hashed at most once per key per run rather than once per duplicate
    /// (issue #296). The pre-existing-blob revalidation the resume path needs
    /// (issue #97) still runs on the *first* touch of each key this run; only
    /// the redundant re-reads that follow are elided. A leaf lock.
    validated_shared: Mutex<HashSet<String>>,
    /// Armed by [`TileSink::arm_durability_tracking`] when the engine stands up
    /// a [`CheckpointState`](crate::engine) for this run. While armed,
    /// [`FsSink::write_tile`] records each freshly-written tile path in
    /// `unsynced_tiles` so [`TileSink::sync_pending`] can fsync it before the
    /// engine certifies the checkpoint delta covering it (issue #122 / #273). A
    /// plain, non-checkpointed run leaves this `false` and pays no per-tile
    /// tracking cost.
    durability_tracking: AtomicBool,
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
    /// Per-shared-key occurrence tracking. Records every tile occurrence of a
    /// given deduplicated content so [`FsSink::canonicalize_dedupe_layout`] can,
    /// at `finish()`, reassign the single full-payload holder to the
    /// coordinate-minimal occurrence — making the on-disk dedupe layout a pure
    /// function of content + coordinates rather than of tile arrival order
    /// (issue #275). A leaf lock.
    dedupe_groups: Mutex<BTreeMap<String, DedupeGroup>>,
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
    /// Absolute paths of tile files written since the last checkpoint flush,
    /// populated only when durability tracking is armed. Before a checkpoint is
    /// certified these are fsynced so the checkpoint never records tiles
    /// whose bytes are still in the page cache (issue #122). Includes the
    /// planned tile paths and any `_shared/` files materialised by the
    /// dedupe path.
    unsynced_tiles: Mutex<Vec<PathBuf>>,
    /// Durability backend used to fsync tile data and the checkpoint
    /// directory. Defaults to real `fsync`; tests inject a recorder to
    /// assert the durability ordering.
    durability: Arc<dyn crate::resume::Durability + Send + Sync>,
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

/// Number of shards backing the `dedupe_promote` outer lock (issue #296).
///
/// The promote-on-2nd-hit critical section must be atomic *per content key*
/// (issue #111), but two tiles of *distinct* content share nothing — different
/// shared blob, different `pending_first` entry, different tile paths — so they
/// need not serialise. Striping the lock across a fixed set of shards, chosen
/// by a hash of the tile content, lets distinct-content writers run
/// concurrently while all occurrences of one content still funnel through a
/// single shard (see [`FsSink::promote_shard`]).
///
/// A power of two so shard selection is a cheap mask. Sized well above any
/// realistic external-writer thread count so distinct-content tiles rarely
/// collide on a shard (a collision only costs an unnecessary wait, never
/// correctness).
const DEDUPE_PROMOTE_SHARDS: usize = 64;

// Per-thread counter of currently-held `FsSink` leaf locks. Only compiled in
// debug builds, where it backs the "at most one leaf lock at a time" assertion
// in `LeafGuard`. A `dedupe_promote` shard is intentionally *not* counted here
// — it is the permitted outer lock (see `FsSink`'s `# Lock discipline`).
#[cfg(debug_assertions)]
thread_local! {
    static FS_SINK_LEAF_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII guard returned by [`FsSink::lock_leaf`]. Wraps a [`MutexGuard`] over a
/// leaf field mutex and, in debug builds, maintains the thread-local
/// [`FS_SINK_LEAF_DEPTH`] so that acquiring a second leaf lock while one is
/// still held panics immediately — surfacing an accidental leaf-over-leaf
/// nesting (the class of mistake the old, fictitious lock-order comment would
/// have invited) at the exact call site rather than as a latent deadlock.
struct LeafGuard<'a, T> {
    inner: MutexGuard<'a, T>,
}

impl<'a, T> LeafGuard<'a, T> {
    #[track_caller]
    fn new(m: &'a Mutex<T>) -> Self {
        #[cfg(debug_assertions)]
        FS_SINK_LEAF_DEPTH.with(|d| {
            assert_eq!(
                d.get(),
                0,
                "FsSink leaf mutex acquired while another leaf lock is held — \
                 violates the at-most-one-leaf-lock invariant (see the type's \
                 `# Lock discipline`, issue #112)"
            );
            d.set(1);
        });
        // Poison recovery (crate::poison policy): the leaf fields are plain
        // in-memory collections that stay structurally valid between
        // operations, so a panicked writer must not cascade its poison into a
        // second panic on every subsequent tile write. We recover the guard and
        // continue; any bookkeeping gap left by the panicked holder is caught
        // downstream by on-disk digest verification, not by tearing down the run.
        LeafGuard {
            inner: crate::poison::recover(m),
        }
    }
}

impl<T> Drop for LeafGuard<'_, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        FS_SINK_LEAF_DEPTH.with(|d| d.set(0));
    }
}

impl<T> std::ops::Deref for LeafGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for LeafGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
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

/// Per-content bookkeeping consumed by [`FsSink::canonicalize_dedupe_layout`]
/// to make dedupe placement a pure function of tile CONTENT + COORDINATE rather
/// than of arrival order (issue #275).
///
/// One entry per distinct shared key. It records every tile occurrence of that
/// content so `finish()` can reassign the single full-payload / hardlink holder
/// deterministically to the coordinate-minimal occurrence, independent of the
/// order in which the tiles reached the sink.
#[derive(Debug, Clone)]
struct DedupeGroup {
    /// Shared file path relative to `base_dir`, e.g. `_shared/blank_<hash>.png`.
    shared_rel: String,
    /// Absolute shared file path (`base_dir/_shared/blank_<hash>.png`).
    shared_abs: PathBuf,
    /// Every tile occurrence of this content: `(coord, tile_rel_path)`.
    occurrences: Vec<(TileCoord, String)>,
}

impl std::fmt::Debug for FsSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsSink")
            .field("base_dir", &self.base_dir)
            .field("format", &self.format)
            .field("checksums", &self.checksums)
            .field("checksum_algo", &self.checksum_algo)
            .field("dedupe", &self.dedupe)
            .field(
                "durability_tracking",
                &self.durability_tracking.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl FsSink {
    /// Acquire one of the field ("leaf") mutexes while upholding the type's
    /// `# Lock discipline`: leaf locks are never nested. In debug builds the
    /// returned [`LeafGuard`] panics if a second leaf lock is taken while one
    /// is still held on this thread, so a stray leaf-over-leaf acquisition
    /// fails loudly at its call site instead of risking a deadlock (issue
    /// #112). `dedupe_promote` is the sole permitted outer lock and is
    /// deliberately acquired directly, not through this helper.
    #[track_caller]
    fn lock_leaf<'a, T>(&self, m: &'a Mutex<T>) -> LeafGuard<'a, T> {
        LeafGuard::new(m)
    }

    /// Creates a new filesystem sink rooted at `base_dir` with the given
    /// pyramid plan. The tile encoding format defaults to
    /// [`TileFormat::Png`]; override it via [`FsSink::with_format`] when
    /// writing JPEG or Raw tiles:
    ///
    /// ```ignore
    /// FsSink::new(dir, plan).with_format(TileFormat::Jpeg { quality: 85 });
    /// ```
    pub fn new(base_dir: impl Into<PathBuf>, plan: PyramidPlan) -> Self {
        let format = TileFormat::Png;
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
        Self {
            base_dir,
            plan,
            format,
            manifest_builder: None,
            checksums: crate::checksum::ChecksumMode::None,
            checksum_algo: None,
            dedupe: None,
            dedupe_index: None,
            dedupe_promote: std::array::from_fn(|_| Mutex::new(())),
            validated_shared: Mutex::new(HashSet::new()),
            durability_tracking: AtomicBool::new(false),
            tile_digests: Mutex::new(BTreeMap::new()),
            manifest_refs: Mutex::new(HashMap::new()),
            pending_first: Mutex::new(HashMap::new()),
            dedupe_groups: Mutex::new(BTreeMap::new()),
            per_level_counts,
            pixel_format: OnceLock::new(),
            unsynced_tiles: Mutex::new(Vec::new()),
            durability: Arc::new(crate::resume::RealDurability),
            saw_blank: AtomicBool::new(false),
            engine_config: Mutex::new(None),
        }
    }

    /// Override the durability backend (test seam).
    ///
    /// Production sinks use the default [`RealDurability`](crate::resume::RealDurability)
    /// installed by [`FsSink::new`]. Tests inject a recorder so the fsync
    /// ordering — tile data synced before the checkpoint that certifies it —
    /// can be asserted, since it is invisible in on-disk state.
    #[cfg(test)]
    pub(crate) fn with_durability(
        mut self,
        durability: Arc<dyn crate::resume::Durability + Send + Sync>,
    ) -> Self {
        self.durability = durability;
        self
    }

    /// Attach a [`ManifestBuilder`](crate::manifest::ManifestBuilder) so the
    /// sink emits a `manifest.json` alongside the pyramid when
    /// [`FsSink::finish`] is called.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-manifest-emit-checksums)
    /// (and [checksum-algo](https://libviprs.org/cli/#flag-checksum-algo)).
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
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-manifest-emit-checksums)
    /// (and [checksum-algo](https://libviprs.org/cli/#flag-checksum-algo)).
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
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
    pub fn with_dedupe(mut self, strategy: crate::dedupe::DedupeStrategy) -> Self {
        if strategy != crate::dedupe::DedupeStrategy::None {
            self.dedupe_index = Some(crate::dedupe::DedupeIndex::new(strategy));
        } else {
            self.dedupe_index = None;
        }
        self.dedupe = Some(strategy);
        self
    }

    /// Arm resume tile-durability tracking for a standalone (non-builder) run.
    ///
    /// When set, [`FsSink::write_tile`] records each freshly-written tile path
    /// so [`TileSink::sync_pending`] can fsync it before a checkpoint certifies
    /// it (issue #122 / #273). The sink no longer publishes its own checkpoint
    /// file — the engine's [`CheckpointState`](crate::engine) is the single
    /// checkpoint authority (issue #277) — so under the documented
    /// [`EngineBuilder::with_resume`](crate::engine::EngineBuilder::with_resume)
    /// path this is armed automatically and calling it here is redundant (but
    /// harmless). Retained so existing callers keep compiling.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
    pub fn with_resume(self, enabled: bool) -> Self {
        self.durability_tracking.store(enabled, Ordering::Relaxed);
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
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-format)
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
            // `dedupe_write` records the digest of the bytes it actually
            // materializes at each tile path (a full payload for the first
            // occurrence, a 1-byte placeholder for references / hardlink
            // fallbacks), so no digest is recorded for the dedupe path here.
            self.dedupe_write(tile.coord, &rel_string, &abs_path, &bytes)?;
            true
        } else {
            std::fs::write(&abs_path, &bytes)?;
            // Non-dedupe path: the full encoded bytes are what lands on disk.
            self.record_tile_digest(&rel_string, &bytes);
            self.track_unsynced(&abs_path);
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

        Ok(())
    }

    fn finish(&self) -> Result<(), SinkError> {
        // Canonicalise dedupe placement so the on-disk layout is a pure
        // function of tile content + coordinates, independent of the order in
        // which tiles reached the sink (issue #275). Must run before checksum
        // verification and manifest emission below so both observe the final,
        // canonical layout.
        if self.dedupe_active() {
            self.canonicalize_dedupe_layout()?;
        }

        // DZI sidecar for DeepZoom layouts is still emitted exactly as
        // before: a sibling of the output directory named `{base}.dzi`.
        if let Some(manifest) = self.plan.dzi_manifest(self.format.extension()) {
            let dzi_path = self.base_dir.with_extension("dzi");
            std::fs::write(&dzi_path, manifest)?;
        }

        // Layout sidecars that live inside the tile directory (Zoomify's
        // `ImageProperties.xml`, IIIF's `info.json`). The planner owns the
        // format and the relative name; the sink only resolves the location.
        if let Some((rel_path, content)) = self.plan.properties_sidecar(self.format.extension()) {
            std::fs::create_dir_all(&self.base_dir)?;
            std::fs::write(self.base_dir.join(rel_path), content)?;
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
        *self.lock_leaf(&self.engine_config) = Some(config.clone());
    }

    fn checkpoint_root(&self) -> Option<&Path> {
        Some(&self.base_dir)
    }

    fn arm_durability_tracking(&self) {
        self.durability_tracking.store(true, Ordering::Relaxed);
    }

    /// Durability barrier (issue #122 / #273): fsync every tile file written
    /// since the last barrier so the checkpoint about to certify them never
    /// records tiles whose bytes are still only in the page cache. Drains the
    /// tracked `unsynced_tiles` set and fsyncs each path via the sink's
    /// [`Durability`](crate::resume::Durability) backend.
    fn sync_pending(&self) -> Result<(), SinkError> {
        // Take the pending set under the leaf lock, then release it before any
        // I/O so the fsyncs never run while a leaf lock is held (the
        // at-most-one-leaf-lock discipline; issue #112).
        let to_sync: Vec<PathBuf> = {
            let mut guard = self.lock_leaf(&self.unsynced_tiles);
            std::mem::take(&mut *guard)
        };
        let mut seen = std::collections::HashSet::new();
        for path in &to_sync {
            if !seen.insert(path.clone()) {
                continue;
            }
            match self.durability.sync_file(path) {
                Ok(()) => {}
                // A path can legitimately be absent by barrier time — e.g. a
                // dedupe first-occurrence file promoted into `_shared/` and
                // replaced by a hardlink. The surviving link/target is synced
                // via its own tracked entry, so a missing path here is not a
                // durability failure.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(SinkError::Io(e)),
            }
        }
        Ok(())
    }

    fn content_format(&self) -> Option<TileFormat> {
        Some(self.format)
    }

    fn seed_completed_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        // Rebuild only the state `finish()` consumes to (re)emit the manifest
        // and dedupe layout: the `DedupeIndex`, `dedupe_groups`, `manifest_refs`,
        // `tile_digests`, per-level counts, `saw_blank` and `pixel_format`. When
        // none of that machinery is active there is nothing for `finish()` to
        // rebuild from a pre-crash tile, so the skipped tile short-circuits as
        // before — preserving resume's I/O savings on the plain path.
        if !(self.dedupe_active()
            || self.checksums != crate::checksum::ChecksumMode::None
            || self.manifest_builder.is_some())
        {
            return Ok(());
        }
        // Re-run the full sink-side write for this already-completed tile so the
        // dedupe index/groups, `manifest_refs`, `tile_digests` and per-level
        // counters end up exactly as an uninterrupted run would leave them —
        // after which `finish()`'s canonicalize + manifest emit reproduce a
        // byte-identical manifest and dedupe layout (issue #272). Tile content
        // is deterministic, so the bytes re-materialised here match what was
        // written pre-crash, and the dedupe write path is already
        // resume-rerun-safe against the on-disk state a prior run left (it
        // revalidates shared blobs and relinks placeholders — issues #93/#97).
        //
        // This intentionally does NOT advance the resume checkpoint:
        // `ResumeAwareSink` calls `mark_tile_completed` only for coordinates it
        // does *not* skip, so seeded coordinates are never re-certified and the
        // checkpoint's `completed_tiles` set stays free of duplicates.
        self.write_tile(tile)
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
    /// Record (or overwrite) the checksum for a tile path, keyed by its
    /// plan-relative path. `materialized` must be the exact bytes that end up
    /// on disk at that path — for deduped placeholders that is the 1-byte
    /// sentinel, not the full encoded payload — so that `ChecksumMode::Verify`
    /// and the post-hoc `verify_output` re-hash of the on-disk file both agree
    /// with the recorded digest (issue #92).
    ///
    /// A no-op when checksums are disabled or no algorithm has been selected.
    /// Record a freshly-written absolute path as needing an `fsync` before
    /// the next checkpoint flush. A no-op unless durability tracking has been
    /// armed (nothing certifies durability on a plain run, so the cost is not
    /// warranted). Issue #122 / #273.
    fn track_unsynced(&self, abs_path: &Path) {
        if !self.durability_tracking.load(Ordering::Relaxed) {
            return;
        }
        self.lock_leaf(&self.unsynced_tiles)
            .push(abs_path.to_path_buf());
    }

    /// Digest algorithm used to name `_shared/blank_<hex>.<ext>` files. Mirrors
    /// `DedupeIndex::effective_algo`: the blank/none strategies always name
    /// shared blobs by their Blake3 digest; `All` honours the caller's choice.
    /// Kept in-sink so a shared blob can be revalidated against the digest
    /// embedded in its own filename without reaching into the index.
    fn dedupe_algo(&self) -> crate::manifest::ChecksumAlgo {
        match self.dedupe_index.as_ref().map(|i| i.strategy()) {
            Some(crate::dedupe::DedupeStrategy::All { algo }) => algo,
            _ => crate::manifest::ChecksumAlgo::Blake3,
        }
    }

    /// True when the file already at `_shared/blank_<hex>.<ext>` hashes to the
    /// `<hex>` digest embedded in its own filename. Used before trusting a
    /// pre-existing shared blob on a resume rerun: a crash mid-write can leave
    /// a short/empty blob that an existence check alone would happily reuse and
    /// point every duplicate tile at (issue #97). A missing file, an unreadable
    /// file, a filename without the `blank_` stem, or a digest mismatch all
    /// return `false` so the caller re-materialises the blob from known-good
    /// bytes.
    fn shared_blob_valid(&self, shared_abs_path: &Path) -> bool {
        let Some(expected_hex) = shared_abs_path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|stem| stem.strip_prefix("blank_"))
        else {
            return false;
        };
        // A well-formed shared stem is exactly the 64-char hex of a 32-byte
        // digest; anything else was not produced by this scheme, so don't
        // pretend to validate it.
        if expected_hex.len() != 64 {
            return false;
        }
        let Ok(bytes) = std::fs::read(shared_abs_path) else {
            return false;
        };
        let got = crate::hex::hex_lower(&hash_tile_raw(&bytes, self.dedupe_algo()));
        got.eq_ignore_ascii_case(expected_hex)
    }

    /// Select the `dedupe_promote` shard for a tile with the given content
    /// `bytes` (issue #296).
    ///
    /// The shard is a pure function of the content, so every occurrence of the
    /// same content selects the same shard — preserving the per-key atomicity
    /// the at-least-one-hardlink invariant depends on (issue #111) — while
    /// distinct content selects (usually) distinct shards, letting
    /// distinct-content tiles promote concurrently instead of serialising on
    /// one process-wide lock. [`std::hash::BuildHasher`]-free
    /// [`std::hash::DefaultHasher`] keeps this deterministic within (and
    /// across) runs; determinism is not required for correctness — only that
    /// identical content maps to one shard — but it keeps behaviour
    /// reproducible.
    fn promote_shard(&self, bytes: &[u8]) -> &Mutex<()> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        // DEDUPE_PROMOTE_SHARDS is a power of two, so the mask is exact.
        let idx = (hasher.finish() as usize) & (DEDUPE_PROMOTE_SHARDS - 1);
        &self.dedupe_promote[idx]
    }

    /// Whether the `_shared/<shared_key>` blob is known-good, consulting the
    /// per-run [`FsSink::validated_shared`] cache before touching disk (issue
    /// #296).
    ///
    /// The first time a key is validated this run — the resume-safety
    /// revalidation of a possibly-pre-existing blob (issue #97) — the full
    /// on-disk read + rehash runs and, on success, the key is remembered.
    /// Every later duplicate of that content short-circuits on the cached
    /// verdict, so a shared blob is read and hashed at most once per key per
    /// run instead of once per duplicate.
    ///
    /// Correctness note: nothing in a single run removes or corrupts a shared
    /// blob after it has been materialised, so a cached "valid" verdict cannot
    /// go stale mid-run; the #97 guard against a *prior* run's truncated blob
    /// still fires on the first touch. Callers must hold the content's
    /// [`FsSink::promote_shard`], so concurrent duplicates of the same content
    /// never race the cache.
    fn shared_blob_cached_valid(&self, shared_key: &str, shared_abs_path: &Path) -> bool {
        if self.lock_leaf(&self.validated_shared).contains(shared_key) {
            return true;
        }
        if self.shared_blob_valid(shared_abs_path) {
            self.mark_shared_validated(shared_key);
            true
        } else {
            false
        }
    }

    /// Record that the `_shared/<shared_key>` blob is materialised and valid
    /// for the remainder of this run, so later duplicates skip the redundant
    /// full-file revalidation (issue #296). Callers must hold the content's
    /// [`FsSink::promote_shard`].
    fn mark_shared_validated(&self, shared_key: &str) {
        self.lock_leaf(&self.validated_shared)
            .insert(shared_key.to_string());
    }

    fn record_tile_digest(&self, rel: &str, materialized: &[u8]) {
        if self.checksums == crate::checksum::ChecksumMode::None {
            return;
        }
        if let Some(algo) = self.checksum_algo {
            let digest = hash_tile_raw(materialized, algo);
            self.lock_leaf(&self.tile_digests)
                .insert(rel.to_string(), digest);
        }
    }

    fn dedupe_write(
        &self,
        coord: TileCoord,
        rel_string: &str,
        abs_path: &Path,
        bytes: &[u8],
    ) -> Result<(), SinkError> {
        use crate::dedupe::DedupeDecision;

        let idx = self
            .dedupe_index
            .as_ref()
            .expect("dedupe_write called without a dedupe index");

        // Hold the promote lock across the entire decision -> write ->
        // register -> promote sequence. The index decision, the first
        // writer's `pending_first.insert`, and a duplicate's
        // `pending_first.remove` would otherwise be three independent
        // critical sections: a `Reference` racing ahead of the first
        // writer's insert would find no pending entry, skip promotion, and
        // leave the first tile as a full private copy — breaking the
        // at-least-one-hardlink invariant (issue #111). Serialising here
        // makes the promote-on-2nd-hit sequence atomic.
        //
        // The lock is sharded by content digest (issue #296): all occurrences
        // of THIS content take the same shard, so the sequence above is still
        // fully serialised per key, but tiles of distinct content take
        // distinct shards and promote concurrently instead of contending on a
        // single process-wide lock. The shard is chosen from `bytes` — a pure
        // function of content — so the same-key funnelling the invariant needs
        // is guaranteed.
        let _promote = crate::poison::recover(self.promote_shard(bytes));

        let decision = idx.record(rel_string, bytes);

        // Record this occurrence against its shared key so `finish()` can pick
        // the full-payload holder deterministically by coordinate order,
        // independent of the arrival order that decided `WriteNew` vs
        // `Reference` above (issue #275). Both decision variants carry the same
        // shared key / path for a given content hash, so the group is stable.
        {
            let (shared_key, shared_rel, shared_abs) = match &decision {
                DedupeDecision::WriteNew {
                    shared_key,
                    shared_path,
                }
                | DedupeDecision::Reference {
                    shared_key,
                    shared_path,
                } => (
                    shared_key.clone(),
                    shared_path.to_string_lossy().replace('\\', "/"),
                    self.base_dir.join(shared_path),
                ),
            };
            self.lock_leaf(&self.dedupe_groups)
                .entry(shared_key)
                .or_insert_with(|| DedupeGroup {
                    shared_rel,
                    shared_abs,
                    occurrences: Vec::new(),
                })
                .occurrences
                .push((coord, rel_string.to_string()));
        }

        match decision {
            DedupeDecision::WriteNew {
                shared_key,
                shared_path,
            } => {
                // Write the bytes at the planned tile path and stash the
                // metadata so a future second hit can promote this file
                // into `_shared/`.
                std::fs::write(abs_path, bytes)?;
                // The full encoded payload lives at the tile path.
                self.record_tile_digest(rel_string, bytes);
                self.track_unsynced(abs_path);

                let shared_rel_string = shared_path.to_string_lossy().replace('\\', "/");
                let shared_abs_path = self.base_dir.join(&shared_path);

                // The DedupeIndex eagerly records the path -> shared_key
                // mapping. For WriteNew we don't want it in the manifest
                // (the content lives directly at the tile path), so drop
                // it from the index's refs.
                idx.forget_reference(rel_string);

                self.lock_leaf(&self.pending_first).insert(
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
                let pending = self.lock_leaf(&self.pending_first).remove(&shared_key);
                if let Some(p) = pending {
                    if let Some(parent) = shared_abs_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    // Reuse a pre-existing shared blob ONLY if its on-disk
                    // content still hashes to the digest in its filename. A
                    // crash mid-write during a prior run can leave a short or
                    // empty blob that an existence check alone would trust and
                    // point every duplicate at (issue #97); such a blob must be
                    // re-materialised. When there is no usable blob, rename the
                    // first tile across (atomic on POSIX; also overwrites a
                    // stale corrupt blob), falling back to an atomic tmp+rename
                    // write if the rename fails (cross-device, etc.).
                    if !self.shared_blob_valid(&shared_abs_path) {
                        if std::fs::rename(&p.tile_abs_path, &shared_abs_path).is_err() {
                            atomic_write(&shared_abs_path, &p.bytes)?;
                        }
                    } else if p.tile_abs_path.exists() {
                        // A valid shared blob already existed; drop the
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
                            // Hardlink succeeded; the promoted tile path now
                            // resolves to the shared file (the full payload),
                            // so its digest stays that of the full bytes. The
                            // digest was already recorded when this tile went
                            // through `WriteNew`, but re-record defensively in
                            // case the two occurrences differ in encoding while
                            // sharing a dedupe key. No manifest entry needed.
                            self.record_tile_digest(&p.tile_rel_path, &p.bytes);
                        }
                        Err(_) => {
                            // Fall back to placeholder + manifest entry. The
                            // tile path now holds the 1-byte sentinel, so its
                            // recorded digest must describe that sentinel — not
                            // the full payload captured during `WriteNew`.
                            // Propagate the write error: an ENOSPC/EACCES here
                            // would otherwise leave no file at the tile path
                            // yet report success (issue #93).
                            std::fs::write(&p.tile_abs_path, [0u8])?;
                            self.record_tile_digest(&p.tile_rel_path, &[0u8]);
                            self.lock_leaf(&self.manifest_refs)
                                .insert(p.tile_rel_path, shared_rel_string.clone());
                        }
                    }

                    // The shared blob is now materialised on disk (renamed
                    // across, atomic-written, or already present and valid).
                    // Remember it so every later duplicate of this content
                    // skips the full revalidation re-read below (issue #296).
                    self.mark_shared_validated(&shared_key);
                }

                // The shared file should now exist and be valid; if it is
                // missing (resume mode with a wiped `_shared/`) or present but
                // corrupt (a crash truncated a prior run's blob), materialize
                // it from the current tile's bytes via an atomic tmp+rename so
                // a reader never sees a partial blob (issue #97). The cached
                // check makes the full-file read + rehash happen at most once
                // per key per run rather than once per duplicate (issue #296).
                if !self.shared_blob_cached_valid(&shared_key, &shared_abs_path) {
                    if let Some(parent) = shared_abs_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    atomic_write(&shared_abs_path, bytes)?;
                    self.mark_shared_validated(&shared_key);
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
                // The tile path holds only the 1-byte sentinel; record the
                // digest of what is actually on disk so Verify/verify_output
                // agree (issue #92).
                self.record_tile_digest(rel_string, &[0u8]);
                // Both the placeholder at the tile path and the shared file
                // that actually holds the content must be durable before the
                // checkpoint certifies this coordinate (issue #122).
                self.track_unsynced(abs_path);
                self.track_unsynced(&shared_abs_path);
                self.lock_leaf(&self.manifest_refs)
                    .insert(rel_string.to_string(), shared_rel_string);
            }
        }
        Ok(())
    }

    /// Make the dedupe on-disk layout a pure function of tile content and
    /// coordinates (issue #275).
    ///
    /// The runtime "promote on 2nd hit" path (see [`FsSink::dedupe_write`])
    /// leaves the *arrival-first* occurrence of each duplicated content as the
    /// full-payload holder — a hardlink to `_shared/<key>` — with every other
    /// occurrence a 1-byte placeholder recorded in
    /// `manifest.json::blank_references`. Which occurrence arrives first is
    /// scheduling-dependent under `tile_concurrency > 0`, so the choice of
    /// holder — and therefore the byte layout and the `blank_references` map —
    /// varied run to run, violating the [`TileSink`] commutative-placement
    /// contract.
    ///
    /// Run once from `finish()` after all writer threads have joined (so there
    /// is no concurrency here), this pass reassigns the single full-payload
    /// holder of every duplicated content to its **coordinate-minimal**
    /// occurrence, sorted by `(level, row, col)` — the same canonical order
    /// [`crate::mapreduce_hot_cache`] uses to make write order deterministic.
    /// The layout *shape* is unchanged: exactly one hardlink per shared blob
    /// plus 1-byte placeholders for the rest; singletons (content seen once)
    /// keep their full file at the tile path and never touch `_shared/`. The
    /// result is identical regardless of the order tiles reached the sink.
    ///
    /// Idempotent: when the holder is already the coordinate-minimal tile (the
    /// common case when tiles arrive in row-major order) it is a pure no-op.
    fn canonicalize_dedupe_layout(&self) -> Result<(), SinkError> {
        // Snapshot the group table, then release the lock before any I/O or
        // other leaf-lock access (lock discipline: at most one leaf lock held
        // at a time — see the type-level `# Lock discipline`).
        let groups: Vec<DedupeGroup> = {
            let g = self.lock_leaf(&self.dedupe_groups);
            g.values().cloned().collect()
        };

        // Only re-read shared bytes to re-record digests when checksums are on.
        let need_digests =
            self.checksums != crate::checksum::ChecksumMode::None && self.checksum_algo.is_some();

        for group in groups {
            // Singletons keep their full file at the tile path; nothing to do.
            if group.occurrences.len() < 2 {
                continue;
            }
            // Reconcile only genuinely promoted content: a valid shared blob
            // must exist. On a filesystem without hardlink support every
            // occurrence is already a placeholder — already order-independent —
            // so there is nothing to reassign.
            if !self.shared_blob_valid(&group.shared_abs) {
                continue;
            }

            // Deterministic target = coordinate-minimal occurrence.
            let mut occ = group.occurrences.clone();
            occ.sort_by_key(|(c, _)| (c.level, c.row, c.col));
            let target_rel = occ[0].1.clone();

            // The current full-payload holder is the sole occurrence NOT
            // recorded as a placeholder in `blank_references`. If every
            // occurrence is a placeholder (no hardlink was materialised) the
            // layout is already order-independent — skip.
            let refs = self.lock_leaf(&self.manifest_refs).clone();
            let current_rel = occ
                .iter()
                .map(|(_, r)| r.clone())
                .find(|r| !refs.contains_key(r));
            let Some(current_rel) = current_rel else {
                continue;
            };
            if current_rel == target_rel {
                continue; // already canonical
            }

            let target_abs = self.base_dir.join(&target_rel);
            let current_abs = self.base_dir.join(&current_rel);

            // Promote the coordinate-minimal tile to the hardlink FIRST, so a
            // (near-impossible, same-filesystem) hardlink failure never leaves
            // the shared blob with no full-payload reference at all.
            if target_abs.exists() || target_abs.is_symlink() {
                let _ = std::fs::remove_file(&target_abs);
            }
            if std::fs::hard_link(&group.shared_abs, &target_abs).is_err() {
                // Could not hardlink the new canonical; leave the existing
                // holder in place. Filesystem hardlink capability is constant
                // for a given run, so this is still deterministic and does not
                // reintroduce order-dependence.
                continue;
            }

            // Target now resolves to the full payload: drop its placeholder ref.
            self.lock_leaf(&self.manifest_refs).remove(&target_rel);
            self.track_unsynced(&target_abs);

            // Demote the previous holder to a 1-byte placeholder + manifest ref.
            if current_abs.exists() || current_abs.is_symlink() {
                let _ = std::fs::remove_file(&current_abs);
            }
            std::fs::write(&current_abs, [0u8])?;
            self.lock_leaf(&self.manifest_refs)
                .insert(current_rel.clone(), group.shared_rel.clone());
            self.track_unsynced(&current_abs);

            // Keep recorded checksums consistent with the rewritten on-disk
            // bytes (issue #92): the new holder now hashes to the full shared
            // payload, the demoted tile to the 1-byte sentinel.
            if need_digests {
                let shared_bytes = std::fs::read(&group.shared_abs)?;
                self.record_tile_digest(&target_rel, &shared_bytes);
                self.record_tile_digest(&current_rel, &[0u8]);
            }
        }
        Ok(())
    }

    /// Re-read every tile recorded during `write_tile` and compare its
    /// on-disk bytes against the expected digest. Returns a SinkError on
    /// the first mismatch.
    fn verify_digests_on_disk(&self) -> Result<(), SinkError> {
        let snapshot = self.lock_leaf(&self.tile_digests).clone();
        let Some(algo) = self.checksum_algo else {
            return Ok(());
        };
        for (rel, expected_bytes) in &snapshot {
            let abs = self.base_dir.join(rel);
            let bytes = match std::fs::read(&abs) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // A recorded tile that is gone from disk is a verification
                    // failure — a tile that was recorded then deleted (or
                    // never durably written) must not pass silently (issue
                    // #93). The sole exemption is a manifest-referenced blank
                    // whose real content lives in `_shared/`; its 1-byte
                    // sentinel may legitimately be absent.
                    if self.lock_leaf(&self.manifest_refs).contains_key(rel) {
                        continue;
                    }
                    return Err(SinkError::MissingTile {
                        tile_rel_path: rel.clone(),
                    });
                }
                Err(e) => return Err(SinkError::Io(e)),
            };
            let got_bytes = hash_tile_raw(&bytes, algo);
            if got_bytes != *expected_bytes {
                return Err(SinkError::ChecksumMismatch {
                    tile_rel_path: rel.clone(),
                    expected: crate::hex::hex_lower(expected_bytes),
                    got: crate::hex::hex_lower(&got_bytes),
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
        let eng_cfg = self.lock_leaf(&self.engine_config).clone();

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
            let raw = self.lock_leaf(&self.tile_digests);
            let per_tile: BTreeMap<String, String> = raw
                .iter()
                .map(|(k, v)| (k.clone(), crate::hex::hex_lower(v)))
                .collect();
            self.checksum_algo.map(|algo| Checksums { algo, per_tile })
        } else {
            None
        };

        // -- blank references ----------------------------------------------
        // sink keeps refs as HashMap for O(1) insert on the hot path; convert
        // to the deterministic BTreeMap shape the manifest requires.
        let blank_references: std::collections::BTreeMap<String, String> = self
            .lock_leaf(&self.manifest_refs)
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
            atomic_write(&sibling_path, &json)?;
        }

        let inside_path = self.base_dir.join("manifest.json");
        atomic_write(&inside_path, &json)?;

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
/// Write `bytes` to `path` atomically: stage them in a sibling `.tmp` file and
/// `rename` it into place. On POSIX the rename is atomic, so a concurrent
/// reader (or a crash) never observes a partially-written `path` — it sees
/// either the old contents or the complete new contents, never a truncated
/// blob (issue #97). Delegates to [`crate::resume::atomic_write`] so shared
/// blobs, the manifest copies, and the checkpoint all publish through one
/// staged-`.tmp` + `rename` helper (issue #124).
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    crate::resume::atomic_write(path, bytes)
}

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
        // Multiband intermediates (from the band ops in `crate::bands`) have
        // no image-crate colour type; reduce or extract to 1/3/4 bands first.
        PixelFormat::Multi8(_) | PixelFormat::Multi16(_) => Err(SinkError::EncodeMsg(format!(
            "multiband raster ({} bands) cannot be encoded as an image tile",
            fmt.channels()
        ))),
        // Float compute intermediates have no PNG/JPEG representation;
        // cast to an unsigned 8/16-bit format before encoding tiles.
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => Err(SinkError::EncodeMsg(format!(
            "float raster ({fmt:?}) cannot be encoded as an image tile; \
             cast to an unsigned 8/16-bit format first"
        ))),
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
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-format)
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

// Crate-visible so extension-dispatched save (`crate::imageio`) reuses the
// sink's JPEG encode path.
pub(crate) fn encode_jpeg(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
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

    /// Reproducer for #117: a poisoned `MemorySink` buffer must not cascade a
    /// second panic into every later `write_tile` / `tiles` / `tile_count`.
    /// Before the fix (`.lock().unwrap()`) the poison re-panicked (RED); after
    /// it the guard is recovered and the already-collected tiles survive (GREEN).
    #[test]
    fn poisoned_memory_sink_recovers_without_cascade() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(0, 0, 0)).unwrap();

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = sink.tiles.lock().unwrap();
            panic!("worker panic while holding the MemorySink buffer lock");
        }));
        assert!(poisoned.is_err());
        assert!(sink.tiles.is_poisoned());

        assert_eq!(sink.tile_count(), 1);
        assert_eq!(sink.tiles().len(), 1);
        sink.write_tile(&make_tile(0, 1, 0)).unwrap();
        assert_eq!(sink.tile_count(), 2);
    }

    /// Reproducer for #117 on the primary cascade site: the `FsSink` leaf-lock
    /// chokepoint. One worker panic that poisons a leaf field mutex must not
    /// turn every subsequent `write_tile` into a second panic (which, on the
    /// write path, would also drop the durability bookkeeping). We poison the
    /// `unsynced_tiles` leaf, then keep writing tiles. Before the fix
    /// (`m.lock().unwrap()` in `LeafGuard::new`) the next write panicked (RED);
    /// after it the guard recovers and the run continues (GREEN).
    #[test]
    fn poisoned_fs_sink_leaf_recovers_without_cascade() {
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let dir = tempfile::tempdir().unwrap();
        // Durability tracking armed so `write_tile` records each tile path into
        // the `unsynced_tiles` leaf through the `lock_leaf` chokepoint.
        let sink = FsSink::new(dir.path().join("out_files"), plan).with_resume(true);

        sink.write_tile(&make_tile(0, 0, 0)).unwrap();

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = sink.unsynced_tiles.lock().unwrap();
            panic!("worker panic while holding an FsSink leaf lock");
        }));
        assert!(poisoned.is_err());
        assert!(sink.unsynced_tiles.is_poisoned());

        // The write path must not cascade the poison into a second panic.
        sink.write_tile(&make_tile(0, 0, 0)).unwrap();
        let recorded = crate::poison::recover(&sink.unsynced_tiles).len();
        assert_eq!(
            recorded, 2,
            "recovered leaf must retain the pre-poison bookkeeping and accept new writes"
        );
    }

    // -- DeepZoom layout variants: Zoomify + IIIF sidecars (libviprs-tests#87) --

    /// Drive an entire plan through `FsSink` (write every tile, then finish)
    /// and return the base output directory for structural assertions.
    fn run_plan_through_fs_sink(plan: PyramidPlan, base: PathBuf) {
        let sink = FsSink::new(base, plan.clone());
        for coord in plan.tile_coords() {
            sink.write_tile(&make_tile(coord.level, coord.col, coord.row))
                .unwrap();
        }
        sink.finish().unwrap();
    }

    /// A Zoomify pyramid produces a `TileGroup0/` directory and an
    /// `ImageProperties.xml` sidecar carrying the source dimensions.
    #[test]
    fn zoomify_run_writes_tilegroup_and_image_properties() {
        let plan = PyramidPlanner::new(300, 200, 128, 0, Layout::Zoomify)
            .unwrap()
            .plan();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("zoomify_out");
        run_plan_through_fs_sink(plan, base.clone());

        assert!(
            base.join("TileGroup0").is_dir(),
            "Zoomify must produce a TileGroup0 directory"
        );
        let props = base.join("ImageProperties.xml");
        assert!(props.exists(), "ImageProperties.xml sidecar must exist");
        let xml = std::fs::read_to_string(&props).unwrap();
        assert!(xml.contains("WIDTH=\"300\""), "got: {xml}");
        assert!(xml.contains("HEIGHT=\"200\""), "got: {xml}");
        // No .dzi sibling for Zoomify.
        assert!(!dir.path().join("zoomify_out.dzi").exists());
    }

    /// An IIIF pyramid produces an `info.json` sidecar carrying the source
    /// dimensions.
    #[test]
    fn iiif_run_writes_info_json() {
        let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::Iiif)
            .unwrap()
            .plan();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("iiif_out");
        run_plan_through_fs_sink(plan, base.clone());

        let info = base.join("info.json");
        assert!(info.exists(), "info.json sidecar must exist");
        let json = std::fs::read_to_string(&info).unwrap();
        assert!(json.contains("\"width\": 512"), "got: {json}");
        assert!(json.contains("\"height\": 512"), "got: {json}");
        // The full-res tile lands under its region directory.
        assert!(
            base.join("0,0,256,256").is_dir(),
            "IIIF region directory must exist"
        );
        assert!(!dir.path().join("iiif_out.dzi").exists());
    }

    /// DeepZoom behaviour is unchanged: a sibling `.dzi` manifest is emitted
    /// and no in-directory properties sidecar appears.
    #[test]
    fn deepzoom_run_still_writes_sibling_dzi_only() {
        let plan = PyramidPlanner::new(300, 200, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("deepzoom_out");
        run_plan_through_fs_sink(plan, base.clone());

        let dzi = dir.path().join("deepzoom_out.dzi");
        assert!(dzi.exists(), "DeepZoom must keep its sibling .dzi manifest");
        let manifest = std::fs::read_to_string(&dzi).unwrap();
        assert!(manifest.contains("Width=\"300\""), "got: {manifest}");
        assert!(manifest.contains("Height=\"200\""), "got: {manifest}");
        assert!(!base.join("ImageProperties.xml").exists());
        assert!(!base.join("info.json").exists());
    }

    // -- Transparent-decorator bookkeeping (issue #137) --

    /// A leaf sink that reports a full, distinct value from every engine
    /// bookkeeping method so a wrapper's forwarding can be observed.
    struct BookkeepingLeaf {
        root: PathBuf,
        skips: std::sync::atomic::AtomicU64,
        levels_seen: std::sync::atomic::AtomicU64,
        config_seen: std::sync::atomic::AtomicBool,
    }

    impl BookkeepingLeaf {
        fn new(root: PathBuf) -> Self {
            Self {
                root,
                skips: std::sync::atomic::AtomicU64::new(0),
                levels_seen: std::sync::atomic::AtomicU64::new(0),
                config_seen: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl TileSink for BookkeepingLeaf {
        fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
            Ok(())
        }
        fn record_engine_config(&self, _config: &crate::engine::EngineConfig) {
            self.config_seen
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        fn sink_retry_count(&self) -> u64 {
            7
        }
        fn sink_skipped_due_to_failure(&self) -> u64 {
            3
        }
        fn note_sink_skipped(&self) {
            self.skips
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn checkpoint_root(&self) -> Option<&Path> {
            Some(&self.root)
        }
        fn init_level_count(&self, levels: usize) {
            self.levels_seen
                .store(levels as u64, std::sync::atomic::Ordering::Relaxed);
        }
        fn content_format(&self) -> Option<TileFormat> {
            Some(TileFormat::Jpeg { quality: 80 })
        }
        fn applies_retry_policy(&self) -> bool {
            true
        }
    }

    /// A transparent decorator that overrides ONLY `write_tile` and
    /// `inner_sink` — it deliberately forwards NONE of the eight engine
    /// bookkeeping methods by hand.
    struct PassThrough<S: TileSink> {
        inner: S,
    }

    impl<S: TileSink> TileSink for PassThrough<S> {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            self.inner.write_tile(tile)
        }
        fn inner_sink(&self) -> Option<&dyn TileSink> {
            Some(&self.inner)
        }
    }

    /// A wrapper that only knows how to name its inner sink (`inner_sink`)
    /// must not silently drop any engine bookkeeping. Before issue #137 a
    /// wrapper had to forward all eight methods by hand or lose retry counts,
    /// the manifest config, the checkpoint root, or the on-disk format; now
    /// the trait defaults forward through `inner_sink`, so overriding that one
    /// method is enough.
    #[test]
    fn decorator_forwards_all_bookkeeping_through_inner_sink() {
        use std::sync::atomic::Ordering;

        let root = PathBuf::from("/tmp/libviprs-137-checkpoint");
        let leaf = BookkeepingLeaf::new(root.clone());
        let wrapper = PassThrough { inner: leaf };

        // Value-returning hooks reflect the inner leaf without any hand-written
        // forwarding on `PassThrough`.
        assert_eq!(wrapper.sink_retry_count(), 7, "retry count must forward");
        assert_eq!(
            wrapper.sink_skipped_due_to_failure(),
            3,
            "skip count must forward"
        );
        assert_eq!(
            wrapper.checkpoint_root(),
            Some(root.as_path()),
            "checkpoint root must forward — else resume silently breaks"
        );
        assert_eq!(
            wrapper.content_format(),
            Some(TileFormat::Jpeg { quality: 80 }),
            "content format must forward — else resume plan-hash mixes formats"
        );
        assert!(
            wrapper.applies_retry_policy(),
            "applies_retry_policy must forward — else the builder double-wraps"
        );

        // Side-effecting hooks reach the inner leaf too.
        wrapper.record_engine_config(&crate::engine::EngineConfig::default());
        assert!(
            wrapper.inner.config_seen.load(Ordering::Relaxed),
            "record_engine_config must forward — else the manifest loses settings"
        );
        wrapper.init_level_count(5);
        assert_eq!(
            wrapper.inner.levels_seen.load(Ordering::Relaxed),
            5,
            "init_level_count must forward — else per-level counters mis-size"
        );
        wrapper.note_sink_skipped();
        wrapper.note_sink_skipped();
        assert_eq!(
            wrapper.inner.skips.load(Ordering::Relaxed),
            2,
            "note_sink_skipped must forward — else skip totals under-count"
        );
    }

    /// A terminal sink that overrides none of the hooks keeps the historical
    /// no-op / `0` / `None` defaults: `inner_sink` returns `None`, so the
    /// forwarding defaults collapse back to their old values.
    #[test]
    fn terminal_sink_keeps_noop_bookkeeping_defaults() {
        let sink = MemorySink::new();
        assert!(sink.inner_sink().is_none());
        assert_eq!(sink.sink_retry_count(), 0);
        assert_eq!(sink.sink_skipped_due_to_failure(), 0);
        assert!(sink.checkpoint_root().is_none());
        assert!(sink.content_format().is_none());
        assert!(!sink.applies_retry_policy());
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

    /**
     * Tests that the tile encoders reject float rasters with a typed
     * error instead of mislabelling their bytes as 8/16-bit samples:
     * PNG/JPEG have no 32-bit float representation here, so callers must
     * cast to an unsigned format before encoding.
     * Input: 4x4 RgbaF32 and FloatF32(1) rasters -> Err(EncodeMsg) from
     * encode_png and encode_jpeg, message naming the float format.
     */
    #[test]
    fn encode_rejects_float_with_typed_error() {
        let rgba = Raster::zeroed(4, 4, PixelFormat::RgbaF32).unwrap();
        let f1 = Raster::zeroed(4, 4, PixelFormat::with_channels(1, 4).unwrap()).unwrap();
        for raster in [&rgba, &f1] {
            match encode_png(raster) {
                Err(SinkError::EncodeMsg(msg)) => {
                    assert!(msg.contains("float raster"), "unexpected message: {msg}")
                }
                other => panic!("expected EncodeMsg for float PNG, got {other:?}"),
            }
            match encode_jpeg(raster, 90) {
                Err(SinkError::EncodeMsg(msg)) => {
                    assert!(msg.contains("float raster"), "unexpected message: {msg}")
                }
                other => panic!("expected EncodeMsg for float JPEG, got {other:?}"),
            }
        }
    }

    /**
     * Regression for issue #92: when dedupe promotes identical uniform tiles
     * under the default `BlankTileStrategy::Emit`, the second (and any hardlink
     * fallback) tile path is materialized as a 1-byte placeholder, but the
     * digest table used to record the digest of the full encoded bytes. That
     * made `ChecksumMode::Verify` re-read the placeholder, hash something
     * different, and fail a perfectly healthy run; `EmitOnly` shipped digests
     * that could never verify.
     *
     * This test drives two identical uniform tiles through a Verify + dedupe
     * sink and asserts finish() succeeds, then re-runs the post-hoc
     * `verify_output` over the emitted manifest and asserts no tile mismatches.
     */
    #[cfg(not(miri))]
    #[test]
    fn verify_dedupe_emit_uniform_tiles_digests_match_disk() {
        use crate::checksum::ChecksumMode;
        use crate::dedupe::DedupeStrategy;
        use crate::manifest::ChecksumAlgo;

        // 16x8 @ tile 8 => the full-resolution level is 2 tiles wide, 1 tall.
        let planner = PyramidPlanner::new(16, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(
            top.cols >= 2,
            "test needs a level with at least two tiles, got cols={}",
            top.cols
        );

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("output_files");
        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Png)
            .with_checksums(ChecksumMode::Verify, ChecksumAlgo::Blake3)
            .with_dedupe(DedupeStrategy::Blanks);

        // Two identical uniform (all-zero) tiles at the same level. Under the
        // default Emit strategy `tile.blank` stays false, but the raster is
        // uniform so `should_dedupe_tile` still routes both through dedupe.
        for col in 0..2 {
            let rect = plan.tile_rect(TileCoord::new(top.level, col, 0)).unwrap();
            let tile = Tile {
                coord: TileCoord::new(top.level, col, 0),
                raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                blank: false,
            };
            sink.write_tile(&tile).unwrap();
        }

        // finish() runs the on-disk Verify pass. Before the fix this returned
        // SinkError::ChecksumMismatch on the deduped placeholder tile.
        sink.finish()
            .expect("Verify + dedupe over identical uniform tiles must pass");

        // Capstone: the emitted manifest's per-tile digests must verify against
        // the bytes actually on disk (placeholders included).
        let report = crate::checksum::verify_output(&base).unwrap();
        assert!(
            report.tiles_mismatched.is_empty(),
            "unexpected mismatches: {:?}",
            report.tiles_mismatched
        );
    }

    /// Locate the single `_shared/blank_*.<ext>` blob under `base`.
    #[cfg(not(miri))]
    fn find_shared_blob(base: &Path) -> PathBuf {
        let shared_dir = base.join("_shared");
        let mut found = None;
        for entry in std::fs::read_dir(&shared_dir)
            .unwrap_or_else(|e| panic!("no _shared/ dir at {shared_dir:?}: {e}"))
        {
            let path = entry.unwrap().path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("blank_"))
            {
                assert!(found.is_none(), "expected exactly one shared blob");
                found = Some(path);
            }
        }
        found.expect("no _shared/blank_* blob was emitted")
    }

    /**
     * Regression for issue #97: shared `_shared/blank_<hex>.<ext>` blobs must
     * be re-validated against the digest embedded in their filename before a
     * resume rerun trusts them. A crash mid-write can leave a short/empty blob;
     * the prior code kept it on an existence check alone and pointed every
     * duplicate tile at the corruption, which — without `ChecksumMode::Verify`
     * — was served silently.
     *
     * This drives one full dedupe run to materialize a correct shared blob,
     * truncates that blob to empty to simulate a crash-interrupted prior run,
     * then reruns the identical job over the same base (a resume). After the
     * rerun the shared blob must again hold the full, correct payload — the new
     * sink must have detected the digest mismatch and re-materialized it.
     * Before the fix the empty blob passed the bare `exists()` check and stayed
     * empty (RED); with the rehash-on-resume guard it is rewritten (GREEN).
     */
    #[cfg(not(miri))]
    #[test]
    fn resume_revalidates_corrupt_shared_blob() {
        use crate::dedupe::DedupeStrategy;

        let planner = PyramidPlanner::new(16, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(top.cols >= 2, "test needs a level with >=2 tiles");

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("output_files");

        let write_two_dupes = |base: &Path| {
            let sink = FsSink::new(base.to_path_buf(), plan.clone())
                .with_format(TileFormat::Png)
                .with_dedupe(DedupeStrategy::Blanks);
            for col in 0..2 {
                let rect = plan.tile_rect(TileCoord::new(top.level, col, 0)).unwrap();
                let tile = Tile {
                    coord: TileCoord::new(top.level, col, 0),
                    raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                    blank: false,
                };
                sink.write_tile(&tile).unwrap();
            }
            sink.finish().unwrap();
        };

        // Run 1: produce a correct shared blob and remember its good bytes.
        write_two_dupes(&base);
        let shared_blob = find_shared_blob(&base);
        let good_bytes = std::fs::read(&shared_blob).unwrap();
        assert!(!good_bytes.is_empty(), "run 1 must emit non-empty payload");

        // Simulate the crash aftermath a resume rerun inherits: a truncated
        // (empty) shared blob left mid-write, while the tile placeholders that
        // referenced it are regenerated from scratch on the rerun. The tile
        // files are removed so the rerun's fresh tile write does not
        // incidentally repair the shared inode through a leftover hardlink —
        // the corrupt `_shared/` blob is the sole survivor, exactly as after a
        // crash between the shared-blob write and its fsync.
        std::fs::write(&shared_blob, b"").unwrap();
        assert_eq!(std::fs::read(&shared_blob).unwrap().len(), 0);
        for col in 0..2 {
            let tile_path = base.join(format!("{}/{col}_0.png", top.level));
            let _ = std::fs::remove_file(&tile_path);
        }

        // Run 2 (resume): the rerun must NOT trust the corrupt blob on its
        // mere existence — it must re-hash it, find the mismatch, and rewrite.
        write_two_dupes(&base);

        let after = std::fs::read(&shared_blob).unwrap();
        assert_eq!(
            after, good_bytes,
            "resume must re-materialize the corrupt shared blob to its full payload"
        );

        // And the restored blob must hash to the digest in its own filename.
        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Png)
            .with_dedupe(DedupeStrategy::Blanks);
        assert!(
            sink.shared_blob_valid(&shared_blob),
            "restored shared blob must validate against its filename digest"
        );
    }

    /**
     * Regression for issue #93: a tile whose digest was recorded during
     * `write_tile` and then removed from disk (deleted, or never durably
     * written) must fail `ChecksumMode::Verify`. Before the fix the on-disk
     * verifier skipped `NotFound` silently, so a recorded-then-deleted tile
     * passed verification and the run reported success over missing content.
     *
     * A manifest-referenced blank (whose real bytes live in `_shared/`) is
     * still exempt: its 1-byte sentinel may be absent without being a failure.
     */
    #[cfg(not(miri))]
    #[test]
    fn verify_recorded_tile_deleted_from_disk_fails() {
        use crate::checksum::ChecksumMode;
        use crate::manifest::ChecksumAlgo;

        let planner = PyramidPlanner::new(8, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        let coord = TileCoord::new(top.level, 0, 0);

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("output_files");
        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Png)
            .with_checksums(ChecksumMode::Verify, ChecksumAlgo::Blake3);

        let rect = plan.tile_rect(coord).unwrap();
        let tile = Tile {
            coord,
            raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
            blank: false,
        };
        sink.write_tile(&tile).unwrap();

        // The digest is now recorded and the file exists. A healthy verify
        // passes.
        sink.verify_digests_on_disk()
            .expect("verify must pass while the recorded tile is present");

        // Delete the recorded tile from disk, then verify again: this must be
        // reported as a failure, not silently skipped.
        let rel = plan.tile_path(coord, TileFormat::Png.extension()).unwrap();
        let abs = base.join(&rel);
        assert!(abs.exists(), "recorded tile should exist before deletion");
        std::fs::remove_file(&abs).unwrap();

        let err = sink
            .verify_digests_on_disk()
            .expect_err("verify must fail when a recorded tile is missing");
        match err {
            SinkError::MissingTile { tile_rel_path } => {
                assert_eq!(tile_rel_path, rel);
            }
            other => panic!("expected MissingTile, got {other:?}"),
        }

        // Exemption: a manifest-referenced blank may be absent without failing.
        sink.manifest_refs
            .lock()
            .unwrap()
            .insert(rel.clone(), "_shared/deadbeef.png".to_string());
        sink.verify_digests_on_disk()
            .expect("manifest-referenced blank may be absent");
    }

    // -- Durability ordering (issue #122) --

    /// One recorded durability operation, in call order.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum DurEvent {
        File(PathBuf),
        Dir(PathBuf),
    }

    /// A [`Durability`](crate::resume::Durability) backend that records the
    /// order of `fsync` calls (and still performs the real syncs) so tests
    /// can assert that tile data is made durable before the checkpoint that
    /// certifies it.
    struct RecordingDurability {
        events: Mutex<Vec<DurEvent>>,
    }

    impl crate::resume::Durability for RecordingDurability {
        fn sync_file(&self, path: &Path) -> std::io::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push(DurEvent::File(path.to_path_buf()));
            let f = std::fs::File::open(path)?;
            f.sync_all()
        }

        fn sync_dir(&self, path: &Path) -> std::io::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push(DurEvent::Dir(path.to_path_buf()));
            #[cfg(unix)]
            {
                let f = std::fs::File::open(path)?;
                f.sync_all()
            }
            #[cfg(not(unix))]
            {
                Ok(())
            }
        }
    }

    /**
     * The durability barrier `TileSink::sync_pending` (issue #122 / #273) must
     * fsync every tile written since the last barrier, so a checkpoint that is
     * about to certify those tiles never records bytes still in the page cache.
     *
     * Arms durability tracking, writes a small pyramid, and calls
     * `sync_pending()` (the exact call the engine's single `CheckpointState`
     * makes before it certifies a delta). The injected recording durability
     * backend must show exactly one file-fsync per written tile — and, because
     * the sink no longer publishes its own checkpoint (the engine's writer is
     * now the sole authority; issue #277), NO directory fsync.
     *
     * RED against the pre-fix code: there was no `sync_pending` barrier at all,
     * and the sink-side writer only fsynced from `finish()` under a
     * standalone-only flag the builder never set.
     */
    #[test]
    #[cfg(not(miri))]
    fn sync_pending_fsyncs_every_written_tile() {
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("out_files");

        let recorder = Arc::new(RecordingDurability {
            events: Mutex::new(Vec::new()),
        });

        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Raw)
            .with_durability(recorder.clone());
        // Arm durability tracking exactly as the engine builder does when it
        // stands up a `CheckpointState` for this sink.
        sink.arm_durability_tracking();

        // Write every tile in the plan.
        let mut written = 0usize;
        for lp in &plan.levels {
            for col in 0..lp.cols {
                for row in 0..lp.rows {
                    let tile = Tile {
                        coord: TileCoord::new(lp.level, col, row),
                        raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
                        blank: false,
                    };
                    sink.write_tile(&tile).unwrap();
                    written += 1;
                }
            }
        }
        assert!(written > 0, "plan should produce at least one tile");

        // The durability barrier the engine invokes before certifying a
        // checkpoint delta.
        sink.sync_pending().unwrap();

        let events = recorder.events.lock().unwrap().clone();

        // No directory fsync: the sink does not publish its own checkpoint
        // anymore (issue #277 — single writer).
        assert!(
            events.iter().all(|e| matches!(e, DurEvent::File(_))),
            "sync_pending must only fsync tile files, never publish a checkpoint \
             directory; got events: {events:?}"
        );

        // Exactly one file fsync per written tile, and every tile path is
        // covered.
        let file_events: Vec<&PathBuf> = events
            .iter()
            .filter_map(|e| match e {
                DurEvent::File(p) => Some(p),
                DurEvent::Dir(_) => None,
            })
            .collect();
        assert_eq!(
            file_events.len(),
            written,
            "expected one tile-data fsync per written tile, got {events:?}"
        );
        for lp in &plan.levels {
            for col in 0..lp.cols {
                for row in 0..lp.rows {
                    let rel = plan
                        .tile_path(TileCoord::new(lp.level, col, row), "raw")
                        .unwrap();
                    let abs = base.join(&rel);
                    assert!(
                        file_events.iter().any(|p| **p == abs),
                        "tile {abs:?} was never fsynced by the durability barrier"
                    );
                }
            }
        }

        // A second barrier with nothing newly written drains to nothing: the
        // pending set was consumed by the first barrier.
        recorder.events.lock().unwrap().clear();
        sink.sync_pending().unwrap();
        assert!(
            recorder.events.lock().unwrap().is_empty(),
            "a barrier with no new writes must fsync nothing"
        );
    }

    /**
     * Regression for issue #111: the dedupe "promote on 2nd hit" path split
     * the index decision, the first tile write, and the `pending_first`
     * registration into separate critical sections. Under concurrent
     * `write_tile` a duplicate receiving `Reference` could run its
     * `pending_first.remove` before the first writer's `pending_first.insert`,
     * silently skipping promotion: the shared file was re-materialised while
     * the first tile kept a full private copy, so no tile ever became a
     * hardlink to the shared inode (the shared file ended up with nlink == 1).
     *
     * This test races two identical uniform tiles through one dedupe sink
     * many times. The at-least-one-hardlink invariant requires that once a
     * `_shared/` file is materialised, at least one tile path is hardlinked
     * to it (nlink >= 2). Before the fix at least one iteration observed a
     * shared file with nlink == 1; after serialising the promote critical
     * section every iteration holds the invariant.
     */
    #[cfg(all(not(miri), unix))]
    #[test]
    fn dedupe_concurrent_promote_keeps_shared_hardlink() {
        use crate::dedupe::DedupeStrategy;
        use std::os::unix::fs::MetadataExt;
        use std::sync::Barrier;

        // 16x8 @ tile 8 => the full-resolution level is 2 tiles wide.
        let planner = PyramidPlanner::new(16, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(
            top.cols >= 2,
            "test needs a level with at least two tiles, got cols={}",
            top.cols
        );

        // Many iterations: the racing insert/remove window is small, so a
        // single pass may miss it. The invariant must hold on every one.
        for iter in 0..128 {
            let dir = tempfile::tempdir().unwrap();
            let base = dir.path().join("output_files");
            let sink = FsSink::new(base.clone(), plan.clone())
                .with_format(TileFormat::Png)
                .with_dedupe(DedupeStrategy::Blanks);

            let make = |col: u32| {
                let rect = plan.tile_rect(TileCoord::new(top.level, col, 0)).unwrap();
                Tile {
                    coord: TileCoord::new(top.level, col, 0),
                    raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                    blank: false,
                }
            };
            let tile_a = make(0);
            let tile_b = make(1);

            // Release both writers simultaneously to maximise the odds the
            // `Reference` writer reaches `pending_first.remove` before the
            // first writer's `pending_first.insert`.
            let barrier = Barrier::new(2);
            std::thread::scope(|s| {
                s.spawn(|| {
                    barrier.wait();
                    sink.write_tile(&tile_a).unwrap();
                });
                s.spawn(|| {
                    barrier.wait();
                    sink.write_tile(&tile_b).unwrap();
                });
            });

            // Exactly one shared file must exist and it must carry the
            // promoted content with at least one hardlink from a tile path.
            let shared_dir = base.join("_shared");
            let shared_files: Vec<_> = std::fs::read_dir(&shared_dir)
                .unwrap_or_else(|e| panic!("iter {iter}: _shared unreadable: {e}"))
                .map(|e| e.unwrap().path())
                .collect();
            assert_eq!(
                shared_files.len(),
                1,
                "iter {iter}: expected exactly one shared file, got {shared_files:?}"
            );
            let shared = &shared_files[0];
            let shared_meta = std::fs::metadata(shared).unwrap();
            assert!(
                shared_meta.nlink() >= 2,
                "iter {iter}: shared file {shared:?} has nlink={}, so no tile is \
                 hardlinked to it (promotion was skipped)",
                shared_meta.nlink()
            );

            // Corroborate: at least one of the two tile paths resolves to the
            // shared inode.
            let shared_ino = shared_meta.ino();
            let shared_dev = shared_meta.dev();
            let mut linked = false;
            for col in 0..2 {
                let rel = plan
                    .tile_path(TileCoord::new(top.level, col, 0), "png")
                    .unwrap();
                let tile_meta = std::fs::metadata(base.join(&rel)).unwrap();
                if tile_meta.ino() == shared_ino && tile_meta.dev() == shared_dev {
                    linked = true;
                }
            }
            assert!(
                linked,
                "iter {iter}: neither tile path is hardlinked to the shared inode"
            );
        }
    }

    /// Build a dedupe-enabled `FsSink` over a throwaway tempdir plus the base
    /// dir it is rooted at, keeping the `TempDir` alive for the caller. Used by
    /// the sharded-promote-lock tests below, which drive `dedupe_write`
    /// directly with synthesised paths rather than through a plan.
    fn dedupe_sink_for_promote_tests() -> (tempfile::TempDir, PathBuf, FsSink) {
        use crate::dedupe::DedupeStrategy;
        use crate::manifest::ChecksumAlgo;
        let planner = PyramidPlanner::new(8, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("out_files");
        std::fs::create_dir_all(base.join("0")).unwrap();
        let sink = FsSink::new(base.clone(), plan)
            .with_format(TileFormat::Png)
            .with_dedupe(DedupeStrategy::All {
                algo: ChecksumAlgo::Blake3,
            });
        (dir, base, sink)
    }

    /// Promote-lock sharding (issue #296): tiles of DISTINCT content take
    /// distinct shards, so one content's held critical section must NOT block a
    /// writer of unrelated content.
    ///
    /// Hold the promote shard for content A, then run a full promote-on-2nd-hit
    /// sequence for content B (mapped to a different shard) on another thread.
    /// It must complete promptly — under the old single process-wide
    /// `dedupe_promote` mutex holding A's lock would block ALL deduped writes,
    /// so B would hang until the shard is released (RED); with the sharded lock
    /// B proceeds concurrently (GREEN).
    #[cfg(not(miri))]
    #[test]
    fn dedupe_distinct_content_promotes_without_serialising() {
        use std::time::Duration;

        let (_dir, base, sink) = dedupe_sink_for_promote_tests();

        // Content A: never written, we only pin its shard.
        let bytes_a = b"promote-shard-content-A".to_vec();
        let shard_a = sink.promote_shard(&bytes_a);

        // Find content B on a different shard so the two genuinely don't
        // contend (a same-shard collision would legitimately serialise).
        let mut bytes_b = None;
        for n in 0..4096u32 {
            let cand = format!("promote-shard-content-B-{n}").into_bytes();
            if !std::ptr::eq(sink.promote_shard(&cand), shard_a) {
                bytes_b = Some(cand);
                break;
            }
        }
        let bytes_b = bytes_b.expect("a content mapping to a different shard must exist");

        // Two distinct occurrences of B force WriteNew then Reference — the
        // full promote sequence, acquiring B's shard.
        let abs_b1 = base.join("0/0_0.png");
        let abs_b2 = base.join("0/0_1.png");
        let sink_ref = &sink;
        let held = crate::poison::recover(sink_ref.promote_shard(&bytes_a));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(move || {
                sink_ref
                    .dedupe_write(TileCoord::new(0, 0, 0), "0/0_0.png", &abs_b1, &bytes_b)
                    .unwrap();
                sink_ref
                    .dedupe_write(TileCoord::new(0, 1, 0), "0/0_1.png", &abs_b2, &bytes_b)
                    .unwrap();
                tx.send(()).unwrap();
            });
            rx.recv_timeout(Duration::from_secs(10)).expect(
                "a distinct-content writer must not be serialised behind an \
                 unrelated content's held promote shard (issue #296)",
            );
        });
        drop(held);
    }

    /// Sharded promote lock still serialises SAME content (issue #111 preserved
    /// under issue #296). Holding content A's shard must block another writer of
    /// the same content until the shard is released — this is what guarantees
    /// the promote-on-2nd-hit sequence stays atomic per key. It also proves the
    /// concurrency in the sibling test is genuine (distinct shards) rather than
    /// vacuous (no locking at all).
    #[cfg(not(miri))]
    #[test]
    fn dedupe_same_content_serialises_on_promote_shard() {
        use std::time::Duration;

        let (_dir, base, sink) = dedupe_sink_for_promote_tests();

        let bytes_a = b"same-content-serialisation-probe".to_vec();
        let abs_a = base.join("0/0_0.png");
        let sink_ref = &sink;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let held = crate::poison::recover(sink_ref.promote_shard(&bytes_a));
            s.spawn(move || {
                sink_ref
                    .dedupe_write(TileCoord::new(0, 0, 0), "0/0_0.png", &abs_a, &bytes_a)
                    .unwrap();
                tx.send(()).unwrap();
            });
            // Still holding A's shard: the same-content writer must be blocked.
            assert!(
                rx.recv_timeout(Duration::from_millis(500)).is_err(),
                "a same-content writer must block on the held promote shard \
                 (issue #111 atomicity)"
            );
            drop(held);
            // Released: it must now make progress.
            rx.recv_timeout(Duration::from_secs(10))
                .expect("writer must proceed once the promote shard is released");
        });
    }

    /// Per-key validation-verdict cache (issue #296): a shared blob is read and
    /// hashed at most once per key per run, not once per duplicate. After
    /// several duplicates of one content the key is recorded in
    /// `validated_shared` (so later duplicates short-circuit the full-file
    /// revalidation) while exactly one shared blob exists and stays valid.
    #[cfg(not(miri))]
    #[test]
    fn dedupe_shared_blob_validated_once_per_key() {
        let (_dir, base, sink) = dedupe_sink_for_promote_tests();

        let bytes = b"validate-once-per-key-content".to_vec();
        // WriteNew + three References of identical content.
        for (col, rel) in [
            (0u32, "0/0_0.png"),
            (1, "0/0_1.png"),
            (2, "0/0_2.png"),
            (3, "0/0_3.png"),
        ] {
            let abs = base.join(rel);
            sink.dedupe_write(TileCoord::new(0, col, 0), rel, &abs, &bytes)
                .unwrap();
        }

        // Exactly one content key validated and cached for the run.
        let cached = crate::poison::recover(&sink.validated_shared);
        assert_eq!(
            cached.len(),
            1,
            "exactly one shared key should be cached as validated, got {cached:?}"
        );

        // Exactly one shared blob on disk, and it is valid (hashes to its name).
        let shared_dir = base.join("_shared");
        let shared_files: Vec<_> = std::fs::read_dir(&shared_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(
            shared_files.len(),
            1,
            "expected exactly one shared blob, got {shared_files:?}"
        );
        assert!(
            sink.shared_blob_valid(&shared_files[0]),
            "the materialised shared blob must be valid"
        );
    }

    /**
     * Enforces the FsSink `# Lock discipline` invariant: at most one leaf
     * field mutex may be held on a thread at a time. Acquiring a second leaf
     * lock through `lock_leaf` while the first is still held is exactly the
     * AB-BA nesting the previously mis-documented "lock order" comment would
     * have invited (issue #112); the debug guard must turn it into an
     * immediate panic at the call site.
     *
     * RED before the fix: no guard existed, so the nested acquisition returned
     * a second guard silently and this `should_panic` test failed. GREEN
     * after: `lock_leaf` trips the depth assertion. Debug-only, since the
     * tracker is compiled under `debug_assertions`.
     */
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "at-most-one-leaf-lock invariant")]
    fn fs_sink_leaf_locks_may_not_nest() {
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        // Construction touches no filesystem, so the path is never written.
        let sink = FsSink::new(std::path::PathBuf::from("unused-no-io"), planner.plan());
        let _outer = sink.lock_leaf(&sink.tile_digests);
        // A second leaf lock while `_outer` is still live must panic.
        let _inner = sink.lock_leaf(&sink.manifest_refs);
    }

    /**
     * The production write/finish paths must uphold the at-most-one-leaf-lock
     * invariant. A full run with dedupe, checksums, and durability tracking all
     * active exercises every leaf mutex — `tile_digests`, `manifest_refs`,
     * `pending_first`, `unsynced_tiles`, `engine_config` —
     * plus the `dedupe_promote` outer lock wrapping leaf acquisitions inside
     * `dedupe_write`. If any path nested two leaf locks the debug guard would
     * panic mid-run; a clean finish proves the documented invariant holds in
     * the real code (issue #112).
     */
    #[cfg(not(miri))]
    #[test]
    fn fs_sink_run_never_nests_leaf_locks() {
        use crate::checksum::ChecksumMode;
        use crate::dedupe::DedupeStrategy;
        use crate::manifest::ChecksumAlgo;

        let planner = PyramidPlanner::new(16, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(top.cols >= 2, "test needs a level with >=2 tiles");

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("output_files");
        let sink = FsSink::new(base, plan.clone())
            .with_format(TileFormat::Png)
            .with_checksums(ChecksumMode::Verify, ChecksumAlgo::Blake3)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_resume(true);

        // engine_config leaf; two identical uniform tiles drive the dedupe
        // promote path (pending_first, manifest_refs, tile_digests); durability
        // tracking drives unsynced_tiles; finish() re-reads
        // tile_digests + manifest_refs + engine_config.
        sink.record_engine_config(&crate::engine::EngineConfig::default());
        for col in 0..2 {
            let rect = plan.tile_rect(TileCoord::new(top.level, col, 0)).unwrap();
            let tile = Tile {
                coord: TileCoord::new(top.level, col, 0),
                raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                blank: false,
            };
            sink.write_tile(&tile).unwrap();
        }
        sink.finish()
            .expect("full dedupe+checksum+resume run must finish without nesting leaf locks");
    }

    /**
     * Issue #275: the tile that holds the full payload (a hardlink to the
     * shared blob) must be the coordinate-minimal occurrence, NOT the first to
     * arrive. Three identical (zeroed => uniform => blank) tiles are fed in
     * REVERSE coordinate order (col 2, then 1, then 0). Before the fix the
     * first arrival — col 2 — kept the full bytes and cols 0/1 became
     * placeholders (RED). After it, `finish()` canonicalises the full-payload
     * holder to col 0 (GREEN).
     */
    #[cfg(not(miri))]
    #[test]
    fn dedupe_full_payload_holder_is_coordinate_minimal_regardless_of_arrival() {
        use crate::dedupe::DedupeStrategy;

        let planner = PyramidPlanner::new(24, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(top.cols >= 3, "test needs a level with >=3 tiles");
        let level = top.level;

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("output_files");
        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Png)
            .with_dedupe(DedupeStrategy::Blanks);

        for col in [2u32, 1, 0] {
            let rect = plan.tile_rect(TileCoord::new(level, col, 0)).unwrap();
            let tile = Tile {
                coord: TileCoord::new(level, col, 0),
                raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                blank: false,
            };
            sink.write_tile(&tile).unwrap();
        }
        sink.finish().unwrap();

        let tile_len = |col: u32| -> u64 {
            let rel = plan
                .tile_path(TileCoord::new(level, col, 0), "png")
                .unwrap();
            std::fs::metadata(base.join(rel)).unwrap().len()
        };
        assert!(
            tile_len(0) > 1,
            "coordinate-minimal tile (col 0) must hold the full payload; got {} bytes \
             (the first tile to arrive under reverse feed was col 2)",
            tile_len(0)
        );
        assert_eq!(tile_len(1), 1, "col 1 must be a 1-byte placeholder");
        assert_eq!(tile_len(2), 1, "col 2 must be a 1-byte placeholder");

        // Exactly one shared blob backs all three occurrences.
        let shared: Vec<_> = std::fs::read_dir(base.join("_shared"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(shared.len(), 1, "expected exactly one shared blob");
    }

    /**
     * Issue #275: replaying the same tiles in opposite arrival orders must
     * yield an identical on-disk dedupe layout — the same full-vs-placeholder
     * assignment at every tile path and a byte-identical shared blob. Before
     * the fix the forward run promotes col 0 while the reverse run promotes
     * col 2, so the tile sizes diverge (RED); after it both canonicalise to the
     * coordinate-minimal holder (GREEN).
     */
    #[cfg(not(miri))]
    #[test]
    fn dedupe_layout_identical_across_arrival_orders() {
        use crate::dedupe::DedupeStrategy;

        let planner = PyramidPlanner::new(24, 8, 8, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let level = plan.levels.last().unwrap().level;

        let run = |base: &Path, cols: &[u32]| {
            let sink = FsSink::new(base.to_path_buf(), plan.clone())
                .with_format(TileFormat::Png)
                .with_dedupe(DedupeStrategy::Blanks);
            for &col in cols {
                let rect = plan.tile_rect(TileCoord::new(level, col, 0)).unwrap();
                let tile = Tile {
                    coord: TileCoord::new(level, col, 0),
                    raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                    blank: false,
                };
                sink.write_tile(&tile).unwrap();
            }
            sink.finish().unwrap();
        };

        let dir_fwd = tempfile::tempdir().unwrap();
        let dir_rev = tempfile::tempdir().unwrap();
        let base_fwd = dir_fwd.path().join("out");
        let base_rev = dir_rev.path().join("out");
        run(&base_fwd, &[0, 1, 2]);
        run(&base_rev, &[2, 1, 0]);

        for col in 0..3u32 {
            let rel = plan
                .tile_path(TileCoord::new(level, col, 0), "png")
                .unwrap();
            let len_fwd = std::fs::metadata(base_fwd.join(&rel)).unwrap().len();
            let len_rev = std::fs::metadata(base_rev.join(&rel)).unwrap().len();
            assert_eq!(
                len_fwd, len_rev,
                "tile col {col} differs in size between arrival orders \
                 ({len_fwd} vs {len_rev}) — dedupe placement is order-dependent"
            );
        }

        let read_shared = |base: &Path| -> Vec<(String, Vec<u8>)> {
            let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(base.join("_shared"))
                .unwrap()
                .map(|e| {
                    let p = e.unwrap().path();
                    (
                        p.file_name().unwrap().to_string_lossy().into_owned(),
                        std::fs::read(&p).unwrap(),
                    )
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        assert_eq!(
            read_shared(&base_fwd),
            read_shared(&base_rev),
            "shared blobs differ between arrival orders"
        );
    }
}
