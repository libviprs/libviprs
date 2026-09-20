//! Moving a pyramid that already exists into a PMTiles archive, without going
//! back to the source image.
//!
//! PMTiles has been the default storage since 0.5.0
//! ([`PyramidStorage::PmTiles`](crate::storage::PyramidStorage) is
//! `#[default]`), so every pyramid generated before that flip is a tree of
//! loose files, and the whole argument for the flip applies to them: 21851
//! tiles cost 22127 filesystem entries as a tree and 1 as an archive, and a
//! random read goes from 3.17us to 1.00us. Until this module the only route
//! from a tree to an archive was to regenerate from the source, which needs
//! the source image to still exist and repeats every decode and every resample
//! that produced the tree in the first place.
//!
//! Both halves were already here and nothing joined them.
//! [`DirectoryPyramidReader`] reads a tree through
//! [`PyramidPlan::tile_path`](crate::planner::PyramidPlan::tile_path), the
//! same function [`FsSink`](crate::sink::FsSink) writes through, so the two
//! cannot drift on where a tile lives. [`Writer`] takes tiles in any order,
//! stores each distinct payload once by content hash, and publishes with a
//! staged temp file, an fsync and an atomic rename. A migration is a walk from
//! one into the other.
//!
//! # Why this is its own module
//!
//! Not on [`Writer`]: that would drag [`crate::planner`] and
//! [`crate::pyramid_reader`] into the format layer, which today knows about
//! bytes, tile ids and directories and nothing about pyramids, and the
//! dependency would point the wrong way. Not inside
//! [`crate::pyramid_reader`] either: a writer living in the reader module is a
//! category error that makes the next person look for the archive writer in
//! two places. One module per direction-and-backend is how this crate is
//! already laid out (`sink_pmtiles` writes, `pyramid_reader` reads), and this
//! is the third corner of that square.
//!
//! # Why the reader arrives as `&dyn PyramidReader`
//!
//! [`migrate_to_pmtiles`] takes a trait object rather than a generic
//! parameter. The trait is the right width for the job: it answers "the stored
//! bytes at this coordinate, or `None`", which is all a migration needs, and
//! taking it as `dyn` means this module never names a concrete reader type.
//! That matters more than the vtable call, which is once per tile against a
//! blake3 hash of the whole payload. `PmTilesPyramidReader` is being generified
//! over its range reader in a neighbouring lane, and a signature that does not
//! name it cannot be broken by that.
//!
//! It also makes archive-to-archive migration fall out for free: anything
//! implementing [`PyramidReader`] can be the source, including one that is not
//! in this crate.
//!
//! # The plan is a parameter, and that is the whole design
//!
//! [`DirectoryPyramidReader::try_open`] refuses to infer a plan, because a
//! directory of tiles does not say what its level indices mean, how big a tile
//! is or which layout placed it. A migration tool is exactly where somebody
//! adds a `--guess`, and this is the argument against it.
//!
//! There are four independent things to guess wrong (the layout, the level
//! base, the tile size and the row axis) and each of them produces a
//! **structurally perfect archive**: `pmtiles verify` passes it, the header
//! counts add up, the directory is sorted and clustered, and every tile sits
//! at the wrong tile id. A `Layout::Google` plan over a `Layout::Xyz` tree
//! reads `{z}/{y}/{x}` where the tiles are at `{z}/{x}/{y}`, so every read
//! succeeds and hands back the transposed tile. Nothing in the pipeline
//! errors.
//!
//! Worse, reading the result back through this crate cannot catch it. The
//! migration reads through `plan.tile_path` and writes through
//! [`tile_coord_to_zxy`], so opening the archive with the same plan un-applies
//! whatever permutation the wrong plan applied and every tile compares equal.
//! The comparison sits on the identity element of the operation under test.
//! `tests/pyramid_migrate.rs` therefore pins absolute tile ids against
//! go-pmtiles' own vectors and compares payloads against an archive this crate
//! did not write, and never asks a [`PyramidReader`] what it thinks.
//!
//! So the plan is an input. There is no inference here and there is no flag to
//! add one.
//!
//! # What this costs
//!
//! One blake3 hash of every tile's bytes.
//! [`Writer::add_tile`](crate::pmtiles::Writer::add_tile) keys its payload
//! table on a digest the caller supplies and never re-derives one, which is
//! right for the engine (the tile has already been hashed by
//! [`DedupeIndex`](crate::dedupe::DedupeIndex) by the time the sink sees it)
//! and is the real CPU cost here, where the bytes are arriving off disk and
//! nobody has hashed them. It is a whole-payload hash, so a migration is
//! hash-bound rather than I/O-bound on a fast disk. It buys the thing a mostly
//! blank pyramid needs most: ten thousand identical blank tiles collapse to
//! one stored payload and one long directory run.
//!
//! Memory is the writer's, and it is bounded: index records are sorted in
//! [`WriterOptions::sort_buffer_records`](crate::pmtiles::WriterOptions)
//! batches and spilled to a log, so a tree that does not fit in RAM still
//! converts. [`MigrateReport::spilled_run_count`] reports whether the external
//! merge was really reached, because a bounded-memory claim that only the
//! in-memory path ever exercises is a claim nothing checks.
//!
//! # Four refusals, and why refusing is the implementation
//!
//! Three of them are inherited from
//! [`PmTilesSink`](crate::sink_pmtiles::PmTilesSink), by name and with the
//! same error and the same sentence, because they are properties of the
//! destination format rather than of how a run reached it.
//!
//! * **A layout that is not addressed by `(z, x, y)`.** `Xyz` and `Google`
//!   are; DeepZoom, Zoomify and IIIF are not, and there is no tile id for a
//!   DeepZoom coordinate to go to. [`SinkError::Unsupported`], and the gate is
//!   on `plan.layout` rather than on
//!   [`PyramidDescription::layout`](crate::pyramid_reader::PyramidDescription),
//!   which is `None` for a foreign archive and would wave the refusal through.
//! * **[`TileFormat::Raw`].** A PMTiles tile is a self-describing image blob
//!   and raw pixel bytes are not one. This does not match on `Raw` itself: it
//!   calls [`TileType::try_from_tile_format`] and propagates, the same line
//!   `PmTilesSinkBuilder::build` runs, so the day a `Webp` variant lands there
//!   is one place to change rather than two.
//! * **[`ResumeMode::Resume`] and [`ResumeMode::Verify`].** Refused before a
//!   byte moves. An interrupted migration restarts from the beginning, and
//!   that is safe rather than merely tolerable: the writer stages into
//!   `<path>.tmp*` and publishes with a rename, so a half-done migration never
//!   wears the finished archive's name and there is nothing to clean up but
//!   scratch.
//!
//! The fourth is this module's own. A tile format has to be **known**, not
//! guessed: [`PyramidReader::tile_format`] answers `None` for a backend that
//! does not commit to one, and picking PNG there would write a `tile_type`
//! byte nobody measured into an archive that will be read by viewers that
//! believe it. So it is [`SinkError::MissingField`] naming
//! `MigrateOptions::tile_format`, and the caller says.
//!
//! # Examples
//!
//! ```
//! use libviprs::planner::{Layout, PyramidPlanner};
//! use libviprs::pyramid_migrate::{MigrateOptions, migrate_directory_to_pmtiles};
//! use libviprs::pyramid_reader::DirectoryPyramidReader;
//! use libviprs::sink::TileFormat;
//! use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = tempfile::tempdir()?;
//! let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz)?.plan();
//! let src = Raster::new(512, 512, PixelFormat::Rgb8, vec![9u8; 512 * 512 * 3])?;
//!
//! // A tree that already exists, however it got there.
//! let tree = dir.path().join("tiles");
//! EngineBuilder::new(&src, plan.clone(), FsSink::new(&tree, plan.clone())).run()?;
//!
//! // The plan is supplied, never inferred.
//! let reader = DirectoryPyramidReader::try_open(&tree, plan, TileFormat::Png)?;
//! let out = dir.path().join("drawing.pmtiles");
//! let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())?;
//!
//! assert!(out.is_file());
//! assert_eq!(report.tiles_written, report.coords_visited - report.tiles_absent);
//! // A flat source repeats one payload wherever two tiles are the same size,
//! // so the archive stores fewer blobs than it addresses tiles.
//! assert!(report.distinct_payloads < report.tiles_written);
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

use crate::planner::PyramidPlan;
use crate::pmtiles::writer::{Writer, WriterOptions, content_hash};
use crate::pmtiles::{Compression, PmTilesError, TileType};
use crate::pyramid_reader::{DirectoryPyramidReader, PyramidReadError, PyramidReader};
use crate::resume::ResumeMode;
use crate::sink::{SinkError, TileFormat};
use crate::sink_pmtiles::{layout_is_zxy, tile_coord_to_zxy};

// ---------------------------------------------------------------------------
// MigrateError
// ---------------------------------------------------------------------------

/// Everything a migration can refuse or fail on.
///
/// Three variants, split by which half of the walk they came from, so a caller
/// can tell "the tree I was handed is not readable" from "the archive could
/// not be written" without matching on a string. The refusals all arrive as
/// [`MigrateError::Refused`] carrying the [`SinkError`] the sink would have
/// raised for the same input, which is deliberate: a caller that already
/// handles `PmTilesSink::build` failing handles these unchanged.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MigrateError {
    /// The source pyramid could not be read.
    #[error("reading the source pyramid failed: {0}")]
    Read(#[from] PyramidReadError),
    /// The migration was refused, with the sink's own error for the same
    /// input.
    #[error("{0}")]
    Refused(#[from] SinkError),
    /// The archive could not be written.
    #[error("writing the archive failed: {0}")]
    Write(#[from] PmTilesError),
}

// ---------------------------------------------------------------------------
// MigrateOptions
// ---------------------------------------------------------------------------

/// What a migration needs to know that the reader and the plan cannot say.
///
/// Everything here has a defensible default, so the common call is
/// `MigrateOptions::default()`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MigrateOptions {
    /// The encoding the stored bytes are in, overriding whatever the reader
    /// says.
    ///
    /// `None` takes [`PyramidReader::tile_format`], and a reader that does not
    /// commit to one is a refusal rather than a guess. Setting this is how an
    /// archive-to-archive migration out of a backend that knows nothing about
    /// its own payloads gets done at all.
    pub tile_format: Option<TileFormat>,
    /// Which resume mode the caller asked for.
    ///
    /// Only [`ResumeMode::Overwrite`] runs. The field exists so a caller
    /// threading a [`ResumePolicy`](crate::resume::ResumePolicy) through gets
    /// the typed refusal here rather than silently having its mode ignored.
    pub resume_mode: ResumeMode,
    /// The archive writer's own options: metadata, bounds, leaf size and the
    /// sort buffer.
    ///
    /// `tile_type` and `tile_compression` are overwritten from the resolved
    /// tile format, because the payloads are copied through byte for byte and
    /// those two fields describe them rather than instructing anybody.
    pub writer: WriterOptions,
}

impl Default for MigrateOptions {
    fn default() -> Self {
        Self {
            tile_format: None,
            resume_mode: ResumeMode::Overwrite,
            writer: WriterOptions::default(),
        }
    }
}

impl MigrateOptions {
    /// Say what the stored bytes are, instead of asking the reader.
    pub fn with_tile_format(mut self, format: TileFormat) -> Self {
        self.tile_format = Some(format);
        self
    }

    /// Set the resume mode. Only [`ResumeMode::Overwrite`] runs.
    pub fn with_resume_mode(mut self, mode: ResumeMode) -> Self {
        self.resume_mode = mode;
        self
    }

    /// Set the archive writer's options.
    pub fn with_writer(mut self, writer: WriterOptions) -> Self {
        self.writer = writer;
        self
    }
}

// ---------------------------------------------------------------------------
// MigrateReport
// ---------------------------------------------------------------------------

/// What a finished migration did, in numbers.
///
/// These are positive controls rather than progress reporting. A migration
/// that wrote nothing produces an archive that is structurally valid and empty,
/// and so does a migration over an empty tree, and every assertion downstream
/// of either passes vacuously. Separating "coordinates the plan enumerated"
/// from "tiles that were there" from "distinct payloads stored" is what makes
/// those three cases tell themselves apart.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MigrateReport {
    /// Coordinates [`PyramidPlan::tile_coords`](crate::planner::PyramidPlan::tile_coords)
    /// yielded. Zero means the plan describes nothing, whatever is on disk.
    pub coords_visited: u64,
    /// Coordinates the reader had a tile for, which is how many tiles reached
    /// the archive.
    pub tiles_written: u64,
    /// Coordinates the reader answered `None` for. A sparse pyramid
    /// legitimately has holes, so this is a count and not an error, but a run
    /// where it equals `coords_visited` migrated an empty tree.
    pub tiles_absent: u64,
    /// Distinct payloads stored, after the writer's content-hash dedupe. Below
    /// `tiles_written` by exactly what the dedupe saved.
    pub distinct_payloads: u64,
    /// Sorted runs spilled to the index log before finalising.
    ///
    /// Zero means the whole index fitted in the sort buffer and the external
    /// merge was never entered. It is reported because that is the only way to
    /// tell a real bounded-memory run from an in-memory sort that happened to
    /// fit, and a test with no way to check which one it exercised is testing
    /// the easy path and reporting the hard one.
    pub spilled_run_count: usize,
    /// The encoding the tiles were taken to be in, after the override and the
    /// reader have both had their say.
    pub tile_format: TileFormat,
    /// Where the archive was published.
    pub out_path: PathBuf,
}

// ---------------------------------------------------------------------------
// migrate_to_pmtiles
// ---------------------------------------------------------------------------

/// Walk every coordinate `plan` describes, read it from `reader`, and write
/// the result to `out` as one PMTiles v3 archive.
///
/// The plan is what enumerates the work. [`PyramidReader::tile`] takes a
/// [`TileCoord`](crate::planner::TileCoord) and nothing in the trait produces
/// one, which is not an oversight: the coordinates a pyramid holds are a
/// property of the plan that laid it out, and
/// [`PyramidPlan::tile_coords`](crate::planner::PyramidPlan::tile_coords) is
/// the same iterator the engine and `FsSink` walk. Handing the plan in rather
/// than pulling it off the reader is also what lets a foreign archive, which
/// can describe none of this, be a source.
///
/// Nothing appears at `out` until the walk finishes: the writer stages
/// elsewhere and publishes with an atomic rename.
///
/// # Errors
///
/// * [`SinkError::UnsupportedResumeMode`] for `Resume` or `Verify`, before a
///   byte moves.
/// * [`SinkError::Unsupported`] for a layout that is not addressed by
///   `(z, x, y)`.
/// * [`SinkError::MissingField`] when neither
///   [`MigrateOptions::tile_format`] nor [`PyramidReader::tile_format`] says
///   what the stored bytes are.
/// * [`SinkError::PmTiles`] for [`TileFormat::Raw`], which PMTiles has no tile
///   type for.
/// * [`MigrateError::Read`] for a source that could not be read, and
///   [`MigrateError::Write`] for an archive that could not be written. An
///   absent tile is neither: it is counted in
///   [`MigrateReport::tiles_absent`] and the walk carries on.
pub fn migrate_to_pmtiles(
    reader: &dyn PyramidReader,
    plan: &PyramidPlan,
    out: impl AsRef<Path>,
    options: MigrateOptions,
) -> Result<MigrateReport, MigrateError> {
    let out = out.as_ref();

    // Refusal order follows `PmTilesSinkBuilder::build`, so the same bad input
    // names the same thing first whichever route it took.
    if !matches!(options.resume_mode, ResumeMode::Overwrite) {
        return Err(SinkError::UnsupportedResumeMode {
            mode: options.resume_mode,
        }
        .into());
    }
    if !layout_is_zxy(plan.layout) {
        return Err(SinkError::Unsupported(format!(
            "{:?} layout is not addressed by (z, x, y), so it has no PMTiles \
             tile ids; use Layout::Xyz or Layout::Google",
            plan.layout
        ))
        .into());
    }
    let tile_format = options
        .tile_format
        .or_else(|| reader.tile_format())
        .ok_or(SinkError::MissingField("MigrateOptions::tile_format"))?;
    let tile_type = TileType::try_from_tile_format(tile_format).map_err(SinkError::PmTiles)?;

    let mut writer_options = options.writer;
    writer_options.tile_type = tile_type;
    // The payloads are copied through exactly as they are stored, and both
    // encodings are already compressed, so this field describes them rather
    // than asking for anything.
    writer_options.tile_compression = Compression::None;

    let mut writer = Writer::create(out, writer_options)?;

    let mut coords_visited = 0u64;
    let mut tiles_written = 0u64;
    let mut tiles_absent = 0u64;
    for coord in plan.tile_coords() {
        coords_visited += 1;
        let Some(bytes) = reader.tile(coord)? else {
            tiles_absent += 1;
            continue;
        };
        // `tile_coords` only yields coordinates inside their level's grid, so
        // this can only refuse a level past 31 or a grid wider than its own
        // zoom, which is a plan PMTiles cannot address rather than a tile that
        // went missing.
        let (z, x, y) = tile_coord_to_zxy(coord).map_err(SinkError::PmTiles)?;
        writer.add_tile(z, x, y, &bytes, content_hash(&bytes))?;
        tiles_written += 1;
    }

    // Read off the writer before `finish` consumes it. Both are cheap and
    // neither is recoverable afterwards.
    let distinct_payloads = writer.distinct_payload_count() as u64;
    let spilled_run_count = writer.spilled_run_count();
    writer.finish()?;

    Ok(MigrateReport {
        coords_visited,
        tiles_written,
        tiles_absent,
        distinct_payloads,
        spilled_run_count,
        tile_format,
        out_path: out.to_path_buf(),
    })
}

/// [`migrate_to_pmtiles`] for the case it was written for, with the plan taken
/// off the reader that was already holding it.
///
/// [`DirectoryPyramidReader`] is the one backend that always knows its own
/// plan, because it was handed one at `try_open` and refused to open without
/// it. So this is not an inference: it is the plan the caller already
/// supplied, and passing it twice would only be a way to pass two different
/// ones.
///
/// # Errors
///
/// The same set as [`migrate_to_pmtiles`].
pub fn migrate_directory_to_pmtiles(
    reader: &DirectoryPyramidReader,
    out: impl AsRef<Path>,
    options: MigrateOptions,
) -> Result<MigrateReport, MigrateError> {
    migrate_to_pmtiles(reader, reader.plan(), out, options)
}
