//! A [`TileSink`] that writes a whole pyramid into one PMTiles v3 archive.
//!
//! [`PmTilesSink`] is the engine-facing half of EPIC F: it takes the tiles a
//! run produces, encodes them, and hands them to the streaming writer in
//! [`crate::pmtiles::writer`], which assembles the header, the directories,
//! the metadata and the tile data and publishes the finished archive with an
//! atomic rename. Nothing appears at the destination until the run finishes.
//!
//! ```
//! use libviprs::planner::{Layout, PyramidPlanner};
//! use libviprs::sink_pmtiles::PmTilesSink;
//! use libviprs::{EngineBuilder, PixelFormat, Raster};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = tempfile::tempdir()?;
//! let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz)?.plan();
//! let src = Raster::new(512, 512, PixelFormat::Rgb8, vec![7u8; 512 * 512 * 3])?;
//!
//! let out = dir.path().join("drawing.pmtiles");
//! let sink = PmTilesSink::builder(&out).plan(plan.clone()).build()?;
//! EngineBuilder::new(&src, plan, sink).run()?;
//!
//! assert!(out.is_file());
//! # Ok(())
//! # }
//! ```
//!
//! # Dedupe here is not the engine's dedupe
//!
//! Collapsing identical payloads into one stored blob is a property of the
//! archive format, not of the run's [`DedupeStrategy`]. `DedupeStrategy`
//! defaults to `None`, under which
//! [`DedupeIndex::record`](crate::dedupe::DedupeIndex::record) answers
//! `WriteNew` for every tile on purpose so that `None` is a true passthrough.
//! A sink that keyed the archive's payload table off those decisions would
//! store every duplicate on the default settings, and "ten thousand blank
//! tiles cost one payload" would only be true for callers who had opted in.
//!
//! So the payload table is keyed on the content digest unconditionally, and
//! the run's strategy is consulted for the digest rather than for the
//! decision: it decides which algorithm the digest is in, which is the only
//! thing about dedupe this sink asks. The digest is the one the engine would
//! have computed, through the same
//! [`content_digest_for`](crate::dedupe::content_digest_for) the index uses;
//! [`Writer::add_tile`](crate::pmtiles::Writer::add_tile) takes it and never
//! re-derives one, so a tile is hashed once however many consumers want the
//! answer.
//!
//! The strategy is held rather than a
//! [`DedupeIndex`](crate::dedupe::DedupeIndex) for a reason worth stating: an
//! index guards its two maps with a mutex, and a sink that reached through one
//! for a hash held that mutex across blake3 over a whole tile payload. The
//! strategy is `Copy`, so it is copied out and the hash runs with nothing
//! locked (issue #1145).
//!
//! # What this sink refuses, and why refusing is the implementation
//!
//! Three things, all of them cases where PMTiles has no representation for
//! what was asked rather than cases where this crate is behind.
//!
//! * **`TileFormat::Raw`.** A PMTiles tile is a self-describing image blob and
//!   raw pixel bytes are not one: there is nowhere in the format to record the
//!   width, the height or the pixel layout that would make them decodable.
//!   [`TileType::try_from_tile_format`](crate::pmtiles::TileType::try_from_tile_format)
//!   already says so.
//! * **Layouts that are not addressed by `(z, x, y)`.** `Layout::Xyz` and
//!   `Layout::Google` are; DeepZoom, Zoomify and IIIF are not, and an archive
//!   built from one of those is addressable but renders nonsense in anything
//!   that opens it.
//! * **`ResumeMode::Resume`.** The writer's staging is not reconstructible
//!   from a checkpoint, so a resumed run would publish an archive with every
//!   pre-crash tile silently absent. `ResumeMode::Verify` was refused beside
//!   it until #1122 and is not any more: the engine asks this sink for a
//!   reader over its own archive and checks the pyramid through that instead
//!   of walking a directory tree. See [`PmTilesSinkBuilder::resume_mode`].

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::dedupe::{DedupeStrategy, content_digest_for};
use crate::engine::EngineConfig;
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::pmtiles::writer::{Writer, WriterOptions};
use crate::pmtiles::{Compression, Header, LibviprsMetadata, Metadata, PmTilesError, TileType};
use crate::resume::{ResumeMode, RunLock};
use crate::sink::{EmissionOrder, SinkError, Tile, TileFormat, TileSink, encode_jpeg, encode_png};

/// Suffix of the sidecar directory a sink creates beside its archive.
///
/// It holds the advisory run lock and nothing else, and it is removed when the
/// sink that owns it is dropped.
const JOB_DIR_SUFFIX: &str = ".job";

// ---------------------------------------------------------------------------
// The coordinate mapping
// ---------------------------------------------------------------------------

/// The PMTiles `(z, x, y)` a plan coordinate addresses.
///
/// PMTiles v3 is ZXY, the convention `Layout::Xyz` and `Layout::Google`
/// already write on disk, so [`TileCoord`]'s three fields map straight across:
/// `level` is `z`, `col` is `x`, `row` is `y`. There is no TMS-style `y`
/// inversion here and no transposition.
///
/// This is a free function, and public, because it is the contract rather than
/// an implementation detail: the sink writes through it and
/// `PmTilesPyramidReader` reads through it, and the one thing that can go
/// wrong is for those two to agree on something the rest of the world does
/// not. `tests/pmtiles_sink.rs` pins it against go-pmtiles' own tile ids.
///
/// # Errors
///
/// [`PmTilesError::ZoomOutOfRange`] for a level above 31, which the `u64`
/// TileID space cannot address, and [`PmTilesError::CoordOutOfRange`] for a
/// column or row outside its level's `2^z` grid. Neither is masked into a
/// different, valid tile the way the reference implementation does.
///
/// # Examples
///
/// ```
/// use libviprs::planner::TileCoord;
/// use libviprs::sink_pmtiles::tile_coord_to_zxy;
///
/// let coord = TileCoord { level: 12, col: 3423, row: 1763 };
/// assert_eq!(tile_coord_to_zxy(coord).unwrap(), (12, 3423, 1763));
/// ```
pub fn tile_coord_to_zxy(coord: TileCoord) -> Result<(u8, u32, u32), PmTilesError> {
    let z = u8::try_from(coord.level).map_err(|_| PmTilesError::ZoomOutOfRange {
        zoom: u8::MAX,
        max: crate::pmtiles::tileid::MAX_ZOOM,
    })?;
    // `zxy_to_tileid` is the authority on what is addressable, so this defers
    // to it rather than repeating the grid arithmetic and drifting from it.
    crate::pmtiles::zxy_to_tileid(z, coord.col, coord.row)?;
    Ok((z, coord.col, coord.row))
}

/// Whether a layout places its tiles at coordinates PMTiles can address.
///
/// Public because it is the gate two different routes into the format have to
/// agree on: this sink refuses a layout here at `build`, and
/// [`migrate_to_pmtiles`](crate::pyramid_migrate::migrate_to_pmtiles) refuses
/// the same layouts before it opens a writer. A second copy of
/// `matches!(layout, Xyz | Google)` in the migration would be a second place
/// to update the day a layout joins or leaves the pair, and the two would
/// disagree for exactly as long as nobody noticed.
///
/// It takes the layout rather than a
/// [`PyramidDescription`](crate::pyramid_reader::PyramidDescription) on
/// purpose. The description's `layout` is an `Option` that is `None` for a
/// backend carrying no plan, and a gate that took it would have to decide what
/// `None` means, which is "wave it through" on the reading that keeps the
/// signature honest.
pub fn layout_is_zxy(layout: Layout) -> bool {
    matches!(layout, Layout::Xyz | Layout::Google)
}

// ---------------------------------------------------------------------------
// PmTilesSink
// ---------------------------------------------------------------------------

/// Where the archive is in its life.
///
/// The writer is created on the first tile rather than at `build()`, so that
/// [`TileSink::record_engine_config`], which the engine calls before any tile,
/// has already landed by the time the archive's metadata is assembled. It also
/// means a sink that is built and dropped without a run leaves nothing on
/// disk but its own lock.
enum WriterState {
    /// Nothing opened yet.
    Pending,
    /// Staging in progress.
    Open(Box<Writer<File>>),
    /// Finalised. `Some` carries the header that was published; `None` means
    /// the finalise failed and there is nothing to report.
    Done(Option<Header>),
}

/// Tile sink that writes an entire pyramid into one PMTiles v3 archive.
///
/// Built through [`PmTilesSink::builder`]. See the [module docs](self) for the
/// dedupe contract and the three things it refuses.
pub struct PmTilesSink {
    out_path: PathBuf,
    job_dir: PathBuf,
    plan: PyramidPlan,
    tile_format: TileFormat,
    /// Everything the writer needs that does not depend on the run. The
    /// metadata is completed from the engine config when the writer opens.
    options: WriterOptions,
    writer: Mutex<WriterState>,
    /// Captured by [`TileSink::record_engine_config`], spent when the writer
    /// opens.
    engine_config: Mutex<Option<EngineConfig>>,
    /// The run's dedupe strategy, which decides the digest algorithm.
    ///
    /// Used for the digest, never for the decision. See the module docs. It is
    /// the strategy rather than a [`DedupeIndex`](crate::dedupe::DedupeIndex)
    /// because the digest is all this sink ever wanted, and an index would put
    /// its own mutex between `write_tile` and the hash (issue #1145).
    dedupe_strategy: Mutex<DedupeStrategy>,
    /// What [`TileSink::emission_order`] answers. Set by
    /// [`PmTilesSinkBuilder::ordered_emission`].
    emission_order: EmissionOrder,
    /// The advisory lock on this archive, held for the sink's whole life.
    lock: Mutex<Option<RunLock>>,
}

impl std::fmt::Debug for PmTilesSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PmTilesSink")
            .field("out_path", &self.out_path)
            .field("tile_format", &self.tile_format)
            .field("layout", &self.plan.layout)
            .finish_non_exhaustive()
    }
}

impl PmTilesSink {
    /// Start a builder for an archive at `path`.
    ///
    /// [`PmTilesSinkBuilder::plan`] is required; everything else has a
    /// default.
    ///
    /// # Examples
    ///
    /// ```
    /// use libviprs::planner::{Layout, PyramidPlanner};
    /// use libviprs::sink::TileFormat;
    /// use libviprs::sink_pmtiles::PmTilesSink;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let dir = tempfile::tempdir()?;
    /// let plan = PyramidPlanner::new(1024, 768, 256, 0, Layout::Xyz)?.plan();
    ///
    /// let sink = PmTilesSink::builder(dir.path().join("out.pmtiles"))
    ///     .plan(plan)
    ///     .tile_format(TileFormat::Jpeg { quality: 85 })
    ///     .build()?;
    /// # let _ = sink;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder(path: impl Into<PathBuf>) -> PmTilesSinkBuilder {
        PmTilesSinkBuilder {
            out_path: path.into(),
            plan: None,
            tile_format: TileFormat::Png,
            metadata: None,
            options: None,
            resume_mode: ResumeMode::Overwrite,
            ordered_emission: false,
        }
    }

    /// Open an archive at `path` for `plan`, encoding tiles as `tile_format`.
    ///
    /// The short form of [`PmTilesSink::builder`].
    ///
    /// # Errors
    ///
    /// See [`PmTilesSinkBuilder::build`].
    pub fn try_new(
        path: impl Into<PathBuf>,
        plan: PyramidPlan,
        tile_format: TileFormat,
    ) -> Result<Self, SinkError> {
        Self::builder(path)
            .plan(plan)
            .tile_format(tile_format)
            .build()
    }

    /// Where the finished archive is published.
    pub fn out_path(&self) -> &Path {
        &self.out_path
    }

    /// The encoding tiles are stored in.
    pub fn tile_format(&self) -> TileFormat {
        self.tile_format
    }

    /// The sidecar directory holding this sink's advisory run lock.
    ///
    /// `<archive>.job`, created when the sink is built and removed when it is
    /// dropped. It exists because [`RunLock`] is directory-scoped by
    /// construction (`dir.join(".libviprs-job.lock")`), so "a lock file beside
    /// the archive" needs a directory to be beside it in. See
    /// [`TileSink::checkpoint_root`] on this type for why it is emphatically
    /// not the archive's own parent.
    pub fn job_dir(&self) -> &Path {
        &self.job_dir
    }

    /// The header of the published archive, once [`TileSink::finish`] has
    /// succeeded.
    ///
    /// `None` before that, and `None` if the finalise failed. It comes from
    /// the writer rather than from re-opening the file, so it is what was
    /// written rather than what a second parse made of it.
    pub fn published_header(&self) -> Option<Header> {
        match &*self.writer.lock().ok()? {
            WriterState::Done(Some(header)) => Some(*header),
            _ => None,
        }
    }

    /// Encode one tile.
    fn encode(&self, tile: &Tile) -> Result<Vec<u8>, SinkError> {
        match self.tile_format {
            TileFormat::Png => encode_png(&tile.raster),
            TileFormat::Jpeg { quality } => encode_jpeg(
                &tile.raster,
                quality,
                crate::sink::background_from(&self.engine_config),
            ),
            TileFormat::Webp => crate::sink::encode_webp(&tile.raster),
            // Refused at `build`, so reaching here would mean a sink was
            // constructed past its own gate.
            TileFormat::Raw => Err(SinkError::PmTiles(PmTilesError::UnsupportedTileFormat {
                format: TileFormat::Raw,
            })),
        }
    }

    /// Take the writer lock, refusing a poisoned one.
    ///
    /// The fragile-write-path half of the crate's poison policy
    /// ([`crate::poison`]): the writer assembles one archive through a
    /// sequence of staged appends, so a holder that panicked part way through
    /// leaves staging nothing can build on. Recovering the guard and carrying
    /// on would publish an archive whose contents nobody can account for, so
    /// the poison becomes a typed error and the run aborts.
    fn lock_writer(&self) -> Result<std::sync::MutexGuard<'_, WriterState>, SinkError> {
        self.writer
            .lock()
            .map_err(|e| SinkError::Other(format!("pmtiles writer mutex poisoned: {e}")))
    }

    /// The metadata object to store, completed from the run's settings.
    fn metadata_for(&self, pixel_format: PixelFormat) -> Metadata {
        let mut metadata = self.options.metadata.clone();
        let config = self
            .engine_config
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        let generation = crate::manifest::GenerationSettings {
            tile_size: self.plan.tile_size,
            overlap: self.plan.overlap,
            layout: self.plan.layout,
            format: self.tile_format,
            concurrency: config.as_ref().map_or(0, |c| c.concurrency),
            background_rgb: config
                .as_ref()
                .map_or([255, 255, 255], |c| c.background_rgb),
            blank_strategy: config
                .as_ref()
                .map_or(crate::engine::BlankTileStrategy::Emit, |c| {
                    c.blank_tile_strategy
                }),
        };
        let source = crate::manifest::SourceMetadata {
            width: self.plan.image_width,
            height: self.plan.image_height,
            pixel_format,
            bytes_hash: None,
        };

        metadata.vnd_libviprs = Some(LibviprsMetadata::new(source, generation));
        if metadata.name.is_none() {
            metadata.name = self
                .out_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned());
        }
        metadata
    }

    /// Run `body` against an open writer, opening one if this is the first
    /// tile.
    fn with_writer<T>(
        &self,
        pixel_format: PixelFormat,
        body: impl FnOnce(&mut Writer<File>) -> Result<T, PmTilesError>,
    ) -> Result<T, SinkError> {
        let mut guard = self.lock_writer()?;
        if matches!(&*guard, WriterState::Pending) {
            let mut options = self.options.clone();
            options.metadata = self.metadata_for(pixel_format);
            let writer = Writer::create(&self.out_path, options).map_err(SinkError::PmTiles)?;
            *guard = WriterState::Open(Box::new(writer));
        }
        match &mut *guard {
            WriterState::Open(writer) => body(writer).map_err(SinkError::PmTiles),
            WriterState::Done(_) => Err(SinkError::Unsupported(
                "the PMTiles sink is already finished".to_string(),
            )),
            WriterState::Pending => unreachable!("just opened above"),
        }
    }
}

impl Drop for PmTilesSink {
    fn drop(&mut self) {
        // Release the lock before removing the directory that holds it.
        // `RunLock`'s own `Drop` unlinks the lock file while the lock is still
        // held, which is what makes the sidecar removable at all.
        if let Ok(mut guard) = self.lock.lock() {
            drop(guard.take());
        }
        // Non-recursive on purpose: it succeeds only when the sidecar is
        // empty, so anything else that ended up in there survives instead of
        // being swept by a sink that does not own it.
        let _ = std::fs::remove_dir(&self.job_dir);
    }
}

// ---------------------------------------------------------------------------
// PmTilesSinkBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for a [`PmTilesSink`].
#[derive(Debug, Clone)]
pub struct PmTilesSinkBuilder {
    out_path: PathBuf,
    plan: Option<PyramidPlan>,
    tile_format: TileFormat,
    metadata: Option<Metadata>,
    options: Option<WriterOptions>,
    resume_mode: ResumeMode,
    ordered_emission: bool,
}

impl PmTilesSinkBuilder {
    /// Attach the pyramid plan. Required.
    pub fn plan(mut self, plan: PyramidPlan) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Set the per-tile encoding. Defaults to [`TileFormat::Png`].
    pub fn tile_format(mut self, tile_format: TileFormat) -> Self {
        self.tile_format = tile_format;
        self
    }

    /// Set the JSON metadata object the archive carries.
    ///
    /// The `vnd.libviprs` namespace is filled in from the plan and the run's
    /// engine config whatever is passed here, so a caller sets the TileJSON
    /// fields (name, attribution, description) and leaves the rest.
    pub fn metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Override the writer's options: leaf size, sort buffer, bounds, centre.
    ///
    /// The tile type and the metadata are set from this builder regardless, so
    /// the sink and the archive cannot disagree about what the tiles are.
    pub fn writer_options(mut self, options: WriterOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Declare the run's resume mode, so an unsupported one is refused before
    /// anything is created.
    ///
    /// [`ResumeMode::Overwrite`] and [`ResumeMode::Verify`] build.
    /// [`ResumeMode::Resume`] does not.
    ///
    /// **`Verify`** used to be refused here too, and the reason it no longer
    /// is says what changed rather than that somebody relaxed a rule. Verify
    /// was dispatched into `raster_verify`, which resolves a checkpoint root
    /// and then stats `root.join(plan.tile_path(coord, ext))` for every
    /// planned coordinate: a loose-file tree walk with no seam a single-file
    /// backend can enter, which pointed at an archive would report the first
    /// coordinate missing and call the archive corrupt. Since #1122 the engine
    /// asks the sink for a reader first
    /// ([`TileSink::open_pyramid_reader`]), and this sink hands back a
    /// [`PmTilesPyramidReader`](crate::pyramid_reader::PmTilesPyramidReader)
    /// over the published archive, so the verify checks the archive instead of
    /// a tree that was never written.
    ///
    /// A Verify run therefore takes the advisory run lock and creates
    /// [`PmTilesSink::job_dir`] like any other, which is new. It is also
    /// correct: two jobs aimed at one archive must not overlap whether or not
    /// either of them intends to write.
    ///
    /// **`Resume`** is a different matter and stays refused. The writer's
    /// staging is not reconstructible from a checkpoint (the content-hash
    /// table, the payload starts and the run boundaries live in memory only),
    /// so a resumed run would publish an archive with every pre-crash tile
    /// silently absent. That is a refusal of the second kind: the format has
    /// no representation for what was asked, so a typed refusal is the correct
    /// implementation rather than a gap to be filled later. Rerun with
    /// `Overwrite`.
    ///
    /// The Resume refusal is also enforced where the engine can reach it, not
    /// only here: [`TileSink::seed_completed_tile`] is the hook a resume calls
    /// for each coordinate it is about to skip, and this sink refuses there
    /// too.
    pub fn resume_mode(mut self, mode: ResumeMode) -> Self {
        self.resume_mode = mode;
        self
    }

    /// Ask the engine for its tiles in ascending tile id order (issue #1145).
    ///
    /// Off by default, because it is not free and most runs do not need it.
    /// A run that turns it on gets an archive that is a pure function of the
    /// tile set rather than of the thread schedule, and, under
    /// [`Layout::Arrival`](crate::pmtiles::Layout), one whose data region is
    /// in tile id order with no reordering pass, no staging file and no copy.
    ///
    /// # What it costs, and where
    ///
    /// The cost is in the engine, not here. The pyramid cascade makes each
    /// level by downscaling the one above it, so the levels can only be
    /// produced from the full-resolution one down, which is the exact reverse
    /// of the order tile ids run in. An ordered run therefore holds every
    /// level's raster at once instead of one at a time, and the levels below
    /// the top sum to a third of it. Nothing else moves: the extraction is
    /// still parallel and the tiles in flight are still bounded by
    /// `EngineConfig::buffer_size`.
    ///
    /// # It is worth nothing under `Layout::TileId`
    ///
    /// The default layout sorts at finalize and writes the data region in
    /// tile id order whatever order the tiles arrived in, so an ordered run
    /// there pays the third and buys an archive it would have produced
    /// anyway. It is accepted rather than refused because it is not wrong,
    /// and because a caller comparing the two layouts wants to hold
    /// everything else fixed. That comparison is what
    /// `an_ordered_arrival_run_is_byte_identical_to_the_tile_id_layout` in
    /// `tests/pmtiles_sink.rs` is.
    pub fn ordered_emission(mut self, ordered: bool) -> Self {
        self.ordered_emission = ordered;
        self
    }

    /// Validate the configuration, take the run lock, and return the sink.
    ///
    /// # Errors
    ///
    /// * [`SinkError::MissingField`] when [`PmTilesSinkBuilder::plan`] was
    ///   never called.
    /// * [`SinkError::UnsupportedResumeMode`] for `Resume`.
    /// * [`SinkError::Unsupported`] for a layout that is not addressed by
    ///   `(z, x, y)`.
    /// * [`SinkError::PmTiles`] for [`TileFormat::Raw`], which PMTiles has no
    ///   tile type for.
    /// * [`SinkError::RunLock`] when another live sink holds this archive.
    pub fn build(self) -> Result<PmTilesSink, SinkError> {
        let plan = self
            .plan
            .ok_or(SinkError::MissingField("PmTilesSinkBuilder::plan"))?;

        // Resume, and only Resume (issue #1122). Verify used to be refused
        // here beside it, because the only verify the engine had walked a
        // directory tree; it now reads the archive back through
        // `TileSink::open_pyramid_reader` below, so there is nothing left for
        // this gate to protect it from.
        //
        // Written as an exhaustive match rather than the shorter
        // `matches!(mode, Resume)`. The two behave identically today and
        // differ the day somebody adds a fourth mode: the match makes that a
        // compile error right here, so the decision gets made, while a
        // `matches!` would wave the new mode through with a refusal written
        // about Resume standing in for one nobody wrote.
        match self.resume_mode {
            ResumeMode::Overwrite | ResumeMode::Verify => {}
            ResumeMode::Resume => {
                return Err(SinkError::UnsupportedResumeMode {
                    mode: self.resume_mode,
                });
            }
        }
        if !layout_is_zxy(plan.layout) {
            return Err(SinkError::Unsupported(format!(
                "{:?} layout is not addressed by (z, x, y), so it has no PMTiles \
                 tile ids; use Layout::Xyz or Layout::Google",
                plan.layout
            )));
        }
        let tile_type =
            TileType::try_from_tile_format(self.tile_format).map_err(SinkError::PmTiles)?;

        let mut options = self.options.unwrap_or_default();
        options.tile_type = tile_type;
        // The blobs are stored exactly as they are encoded, and both encodings
        // are already compressed, so this field describes them rather than
        // asking for anything.
        options.tile_compression = Compression::None;
        if let Some(metadata) = self.metadata {
            options.metadata = metadata;
        }

        let mut job_dir = self.out_path.clone().into_os_string();
        job_dir.push(JOB_DIR_SUFFIX);
        let job_dir = PathBuf::from(job_dir);
        let lock = RunLock::acquire(&job_dir).map_err(SinkError::RunLock)?;

        Ok(PmTilesSink {
            out_path: self.out_path,
            job_dir,
            plan,
            tile_format: self.tile_format,
            options,
            writer: Mutex::new(WriterState::Pending),
            engine_config: Mutex::new(None),
            dedupe_strategy: Mutex::new(DedupeStrategy::default()),
            emission_order: if self.ordered_emission {
                EmissionOrder::TileId
            } else {
                EmissionOrder::Cascade
            },
            lock: Mutex::new(Some(lock)),
        })
    }
}

// ---------------------------------------------------------------------------
// TileSink
// ---------------------------------------------------------------------------

// Seven of the trait's methods are deliberately left at their defaults, which
// for a terminal sink bottom out at a no-op, `0`, `false` or `None`:
// `inner_sink` (this sink wraps nothing), `sink_retry_count`,
// `sink_skipped_due_to_failure`, `note_sink_skipped` and `applies_retry_policy`
// (no retry loop of its own, so `RetryingSink` wraps it the ordinary way),
// `init_level_count` (there are no per-level counters to pre-size: the writer
// keeps one payload table for the whole archive) and `arm_durability_tracking`
// (`sync_pending` below is unconditional, so there is nothing to turn on, and a
// flag nobody reads is worse than no override at all).
impl TileSink for PmTilesSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        // The plan is the authority on which coordinates exist, the same check
        // `FsSink` makes and with the same error, so a caller driving a sink
        // by hand gets one answer whichever backend it points at.
        if self
            .plan
            .tile_path(tile.coord, self.tile_format.extension())
            .is_none()
        {
            return Err(SinkError::InvalidCoord { coord: tile.coord });
        }
        let (z, x, y) = tile_coord_to_zxy(tile.coord).map_err(SinkError::PmTiles)?;

        // Blank tiles are encoded like any other, deliberately. PMTiles has no
        // placeholder concept, so a skipped blank is a hole in the archive and
        // a 1-byte marker is not a decodable tile. The payload table collapses
        // them into one stored blob and the directory into one long run, which
        // is the size win the marker exists for and a better one.
        let bytes = self.encode(tile)?;

        // The guard is dropped before the hash runs, deliberately. The
        // strategy is all the digest needs and it is `Copy`, so holding a
        // mutex across blake3 over a whole tile payload buys nothing and
        // serialises every concurrent `write_tile` on this sink behind one
        // hash at a time (issue #1145).
        let strategy = {
            let guard = self
                .dedupe_strategy
                .lock()
                .map_err(|e| SinkError::Other(format!("pmtiles dedupe mutex poisoned: {e}")))?;
            *guard
        };
        let digest = content_digest_for(strategy, &bytes).1;

        self.with_writer(tile.raster.format(), |writer| {
            writer.add_tile(z, x, y, &bytes, digest)
        })
    }

    fn finish(&self) -> Result<(), SinkError> {
        let mut guard = self.lock_writer()?;
        let writer = match std::mem::replace(&mut *guard, WriterState::Done(None)) {
            WriterState::Open(writer) => *writer,
            WriterState::Pending => {
                // Nothing was ever written. Opening the writer only to hand it
                // straight to `finish` looks redundant, and is not: the
                // "an archive with no tiles has no conformant encoding"
                // refusal lives in the writer, and a second copy of it here
                // would be a guard no test could tell from the real one.
                let mut options = self.options.clone();
                options.metadata = self.metadata_for(PixelFormat::Rgb8);
                Writer::create(&self.out_path, options).map_err(SinkError::PmTiles)?
            }
            WriterState::Done(_) => {
                return Err(SinkError::Unsupported(
                    "the PMTiles sink is already finished".to_string(),
                ));
            }
        };

        let finished = writer.finish().map_err(SinkError::PmTiles)?;
        *guard = WriterState::Done(Some(finished.header));
        Ok(())
    }

    /// The tile encoding this sink commits to.
    ///
    /// Never `None`: the archive's header carries a single `TileType` for the
    /// whole thing, so the sink knows. A `None` here would loosen the resume
    /// plan hash and make the verify path probe every known extension instead
    /// of the one that was written.
    fn content_format(&self) -> Option<TileFormat> {
        Some(self.tile_format)
    }

    /// No checkpoint root, on purpose.
    ///
    /// Whatever a sink returns here is handed to
    /// [`wipe_directory`](crate::engine) on every `Overwrite` run, and that
    /// function's ownership guard refuses any directory that is non-empty and
    /// holds no `.libviprs-job.json` marker. A `PmTilesSink` naming the
    /// archive's own parent would therefore refuse every Overwrite run the
    /// moment the user keeps anything else in that directory, and wipe the
    /// directory when they do not. Neither is a thing a sink should do to a
    /// path the user only asked it to write one file into.
    ///
    /// It is also the honest answer. A checkpoint root is where a resumable
    /// run keeps the record of which tiles are already durable, and this sink
    /// refuses resume (see [`PmTilesSinkBuilder::resume_mode`]), so there is
    /// nothing to keep there. The advisory lock the archive does need lives in
    /// [`PmTilesSink::job_dir`] instead, taken by the sink itself rather than
    /// by the engine, because two sinks aimed at one archive corrupt each
    /// other's staging whether or not a resume policy was configured.
    fn checkpoint_root(&self) -> Option<&Path> {
        None
    }

    /// Open the published archive for reading (issue #1122).
    ///
    /// This is how `ResumeMode::Verify` reaches a single-file backend at all.
    /// The engine asks every sink this before it falls back to the directory
    /// walk, so a verify against an archive checks the archive rather than
    /// stat-ing a tree nobody wrote.
    ///
    /// # It is emphatically not the checkpoint root
    ///
    /// Wiring verify makes [`TileSink::checkpoint_root`] look like the natural
    /// place to hand the engine a path, and the doc above it explains why that
    /// would be wrong: whatever a sink returns there is fed to
    /// [`wipe_directory`](crate::engine) on every `Overwrite` run. Verify
    /// reads the archive; it needs no root, and this method is what makes that
    /// true rather than merely asserted.
    ///
    /// # Errors
    ///
    /// [`SinkError::PyramidRead`] when the archive is absent or is not a
    /// readable v3 archive. `Err`, not `Ok(None)`: a missing archive is a
    /// verify failure with a name, while `Ok(None)` would mean "I have no
    /// reader to offer" and would send the run off to walk a directory tree
    /// that is not there either, which is the "missing tile for coord 0/0/0"
    /// answer this whole path exists to stop giving.
    fn open_pyramid_reader(
        &self,
    ) -> Result<Option<Box<dyn crate::pyramid_reader::PyramidReader>>, SinkError> {
        let reader = crate::pyramid_reader::PmTilesPyramidReader::try_open(&self.out_path)
            .map_err(SinkError::PyramidRead)?;
        Ok(Some(Box::new(reader)))
    }

    /// Make every accepted tile durable.
    ///
    /// The writer's staged payloads and index log are flushed through their
    /// buffers and `sync_data`d. It does not publish anything: the archive
    /// appears at its destination when [`TileSink::finish`] renames it there
    /// and not before, so what a crash keeps is intact staging rather than a
    /// half-usable archive. There is no intermediate state of a single-file
    /// archive that a reader could open, which is the whole reason the
    /// destination stays untouched.
    ///
    /// [`TileSink::arm_durability_tracking`] is deliberately not overridden.
    /// Arming exists so a sink can start recording which tile files to fsync;
    /// this one has one staging file and syncs all of it every time, so there
    /// is nothing to turn on, and a flag nobody reads would be a hook that
    /// looks implemented and does nothing.
    fn sync_pending(&self) -> Result<(), SinkError> {
        let mut guard = self.lock_writer()?;
        match &mut *guard {
            WriterState::Open(writer) => writer.sync_pending().map_err(SinkError::PmTiles),
            // Nothing has been accepted, or everything has already been
            // published and fsynced by the finalise. Either way the barrier
            // has nothing left to make durable.
            WriterState::Pending | WriterState::Done(_) => Ok(()),
        }
    }

    fn record_engine_config(&self, config: &EngineConfig) {
        if let Ok(mut guard) = self.engine_config.lock() {
            *guard = Some(config.clone());
        }
        // The digest algorithm depends on the strategy, so the sink takes the
        // run's rather than staying on the default. Keying the payload table
        // on a digest from one algorithm while the rest of the run uses
        // another would not be wrong, but it would mean hashing twice.
        if let Ok(mut guard) = self.dedupe_strategy.lock() {
            *guard = config.dedupe_strategy.unwrap_or_default();
        }
    }

    /// The order this sink wants its tiles in (issue #1145).
    ///
    /// [`EmissionOrder::Cascade`] unless
    /// [`PmTilesSinkBuilder::ordered_emission`] was set, which is where the
    /// argument for turning it on lives.
    fn emission_order(&self) -> EmissionOrder {
        self.emission_order
    }

    /// Refuse the tile a resume was about to skip.
    ///
    /// The engine calls this for every coordinate a resume short-circuits, to
    /// let the sink rebuild the state that tile would have contributed. A
    /// PMTiles archive cannot: the tile's bytes are not in the archive,
    /// because the archive does not exist until the run finishes. Leaving this
    /// at its `Ok(())` default would publish an archive missing every
    /// pre-crash tile, which is the silent corruption
    /// [`PmTilesSinkBuilder::resume_mode`] refuses at the other end. This is
    /// the end the engine can actually reach.
    fn seed_completed_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        Err(SinkError::UnsupportedResumeMode {
            mode: ResumeMode::Resume,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PyramidPlanner;
    use crate::raster::Raster;

    fn plan() -> PyramidPlan {
        PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz)
            .expect("a square plan is valid")
            .plan()
    }

    fn tile(coord: TileCoord) -> Tile {
        Tile {
            coord,
            raster: Raster::new(8, 8, PixelFormat::Rgb8, vec![3u8; 8 * 8 * 3])
                .expect("a small raster"),
            blank: false,
        }
    }

    /// A panic while the writer lock is held becomes a typed error on the next
    /// call, not a second panic.
    ///
    /// This is the fragile-write-path half of the crate's poison policy: the
    /// archive is assembled through a sequence of staged appends, so a holder
    /// that died mid-append leaves staging that nothing should build on.
    /// `crate::poison::recover` is for bookkeeping maps and would be wrong
    /// here.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_poisoned_writer_lock_is_a_typed_error_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = PmTilesSink::builder(dir.path().join("poison.pmtiles"))
            .plan(plan())
            .build()
            .expect("the sink builds");

        let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = sink.writer.lock().expect("the lock is clean here");
            panic!("a writer thread died mid-archive");
        }));
        assert!(died.is_err(), "the control: the helper really did panic");

        let coord = TileCoord {
            level: 0,
            col: 0,
            row: 0,
        };
        match sink.write_tile(&tile(coord)) {
            Err(SinkError::Other(message)) => assert!(
                message.contains("poisoned"),
                "the error must name the poison, got {message}"
            ),
            other => panic!("a poisoned lock must be a typed error, got {other:?}"),
        }
        match sink.finish() {
            Err(SinkError::Other(message)) => assert!(message.contains("poisoned")),
            other => panic!("finish must refuse a poisoned lock too, got {other:?}"),
        }
    }

    /// A sink that never saw a tile refuses to publish an empty archive, and
    /// leaves nothing at the destination.
    ///
    /// Every PMTiles directory must hold at least one entry, so there is no
    /// conformant encoding of an empty archive and writing one anyway would
    /// push the problem onto whoever opened it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn finishing_without_a_single_tile_publishes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("empty.pmtiles");
        let sink = PmTilesSink::builder(&out)
            .plan(plan())
            .build()
            .expect("the sink builds");

        assert!(sink.finish().is_err(), "an empty archive has no encoding");
        assert!(
            !out.exists(),
            "a refused finalise must leave nothing wearing the archive's name"
        );
        assert_eq!(sink.published_header(), None);
    }

    /// Dropping a sink without finishing leaves the destination untouched.
    ///
    /// This is the shape a killed process most resembles, and it is the whole
    /// point of staging: a partial archive must never wear the name a complete
    /// one would.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn dropping_a_sink_mid_run_leaves_no_archive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("abandoned.pmtiles");
        {
            let sink = PmTilesSink::builder(&out)
                .plan(plan())
                .build()
                .expect("the sink builds");
            for col in 0..2 {
                let coord = TileCoord {
                    level: 9,
                    col,
                    row: 0,
                };
                sink.write_tile(&tile(coord)).expect("the sink takes it");
            }
        }
        assert!(!out.exists(), "an abandoned run publishes nothing");
        assert!(
            !dir.path().join("abandoned.pmtiles.job").exists(),
            "and it takes its sidecar with it"
        );
    }

    /// The published header comes back from the sink rather than from
    /// reopening the file.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_published_header_is_available_without_reopening_the_archive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("header.pmtiles");
        let sink = PmTilesSink::builder(&out)
            .plan(plan())
            .build()
            .expect("the sink builds");

        assert_eq!(sink.published_header(), None, "nothing is published yet");
        for col in 0..2 {
            sink.write_tile(&tile(TileCoord {
                level: 9,
                col,
                row: 0,
            }))
            .expect("the sink takes it");
        }
        sink.finish().expect("the archive publishes");

        let header = sink
            .published_header()
            .expect("a published archive has a header");
        assert_eq!(header.tile_type, TileType::Png);
        assert_eq!(header.addressed_tiles_count, 2);
        assert_eq!(
            header.tile_contents_count, 1,
            "two identical tiles are one payload, on the default strategy"
        );
    }

    /// A coordinate the plan does not have is refused with the same error
    /// `FsSink` uses.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_coordinate_outside_the_plan_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = PmTilesSink::builder(dir.path().join("coord.pmtiles"))
            .plan(plan())
            .build()
            .expect("the sink builds");

        let coord = TileCoord {
            level: 9,
            col: 4_000,
            row: 0,
        };
        match sink.write_tile(&tile(coord)) {
            Err(SinkError::InvalidCoord { coord: got }) => assert_eq!(got, coord),
            other => panic!("an impossible coordinate must be refused, got {other:?}"),
        }
    }

    /// A level above the addressable zoom range is refused rather than
    /// saturated into a valid one.
    #[test]
    fn a_level_above_the_zoom_ceiling_is_refused() {
        for level in [32u32, 256, u32::MAX] {
            let coord = TileCoord {
                level,
                col: 0,
                row: 0,
            };
            assert!(
                tile_coord_to_zxy(coord).is_err(),
                "level {level} is not addressable and must not be masked into one that is"
            );
        }
        // The control: the level just below the ceiling still maps.
        assert_eq!(
            tile_coord_to_zxy(TileCoord {
                level: 31,
                col: 0,
                row: 0
            })
            .expect("zoom 31 is addressable"),
            (31, 0, 0)
        );
    }
}
