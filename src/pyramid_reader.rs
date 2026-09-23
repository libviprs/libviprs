//! Reading a generated pyramid back, whatever it was written to.
//!
//! The engine has always had one way in and several ways out: a run can land
//! in a directory of loose tiles, in a packfile, in an object store or, since
//! #990, in one PMTiles archive. What it has not had is one way to ask "give
//! me the tile at `z/x/y`" that does not care which of those it was. Anything
//! that wanted to read a pyramid back had to know how it was stored, which is
//! why the verify path was a directory walk and why nothing but a viewer ever
//! opened an archive. Since #1122 it is not the only verify path: a sink that
//! can open its own output hands one of these back from
//! [`TileSink::open_pyramid_reader`](crate::sink::TileSink::open_pyramid_reader)
//! and [`pyramid_verify`](crate::verify::pyramid_verify) checks the pyramid
//! through the trait instead. `raster_verify` still owns the tree.
//!
//! [`PyramidReader`] is that one way in. Two implementations ship with it:
//! [`DirectoryPyramidReader`] over a `{z}/{x}/{y}.{ext}` tree, and
//! [`PmTilesPyramidReader`] over a single archive.
//!
//! Since #1121 the archive half no longer means "a file on this disk".
//! [`PmTilesPyramidReader`] carries its transport as a defaulted type
//! parameter, so it still opens a path with nothing spelled out at the call
//! site, and it also opens an object in an injected store through
//! [`try_from_object_store`](PmTilesPyramidReader::try_from_object_store).
//!
//! # An absent tile is not an error
//!
//! [`PyramidReader::tile`] answers `Ok(None)` for a coordinate the pyramid
//! does not have, and reserves `Err` for a pyramid it could not read. The
//! distinction matters more than it looks: a sparse pyramid legitimately has
//! holes, and a reader that cannot tell "there is no tile here" from "I could
//! not find out" turns every hole into a failed run.
//!
//! # The comparison this module makes possible, and the way it lies
//!
//! Two backends that agree tile for tile are evidence that they agree, and
//! nothing more. A sink and a reader that share a coordinate-mapping mistake
//! agree perfectly and are both wrong, and a comparison over an empty
//! coordinate set agrees most perfectly of all. So a cross-backend test needs
//! a positive control on the tile set it compared, and at least one coordinate
//! pinned against something neither backend produced.

use std::path::{Path, PathBuf};

use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::sink::TileFormat;

// ---------------------------------------------------------------------------
// PyramidReadError
// ---------------------------------------------------------------------------

/// Everything that can go wrong reading a pyramid back.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PyramidReadError {
    /// An underlying read failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A PMTiles archive refused something.
    #[error("pmtiles error: {0}")]
    PmTiles(#[from] crate::pmtiles::PmTilesError),
    /// The backend is there but it cannot say what the pyramid is: no plan, no
    /// metadata, nothing to describe it with.
    #[error("the pyramid does not describe itself: {0}")]
    NoDescription(String),
    /// The path handed to a constructor is not the kind of thing that
    /// constructor opens.
    #[error("{path} is not {expected}")]
    NotAPyramid { path: PathBuf, expected: String },
    /// [`PyramidReader::self_check`] walked the storage and found it damaged.
    ///
    /// Distinct from every other variant on purpose. An `Io` or a `PmTiles`
    /// error says the reader could not find out; this one says it did find
    /// out, and the answer is that the pyramid is not sound. A caller that
    /// collapsed the two would report a corrupt archive and an unreadable
    /// disk the same way.
    ///
    /// The payload is the finding list rather than the first symptom, because
    /// a structural walk that stops at the first problem describes one thing
    /// wrong with a file that may have six.
    #[error(
        "the pyramid is structurally damaged ({} finding(s)); the first is: {}",
        .findings.len(),
        .findings.first().map(ToString::to_string).unwrap_or_else(|| "(none recorded)".to_string())
    )]
    StructuralDefects {
        /// The findings themselves, typed rather than rendered.
        ///
        /// These used to be `Vec<String>`. `tests/error_source_typing.rs` is
        /// the file that made "a typed error must not be laundered into a
        /// string" a rule here, and rendering a public enum into display text
        /// broke it: a caller who wants to treat a truncated archive
        /// differently from an entry pointing outside its section was left
        /// substring-matching English.
        findings: Vec<crate::pmtiles::validate::Finding>,
    },
    /// The backend cannot count the tiles it addresses.
    ///
    /// Separate from [`PyramidReadError::NoDescription`] on purpose. That one
    /// means "I cannot say what this pyramid is"; this one means "I know what
    /// it is and I cannot count it". A caller matching the first to decide a
    /// backend carries no metadata should not also catch the second.
    #[error("this pyramid cannot count the tiles it addresses: {0}")]
    NotCountable(String),
    /// The archive carries a `vnd.libviprs` namespace this build cannot parse.
    ///
    /// # Why this variant exists rather than a quiet `None` (issue #1123)
    ///
    /// [`Metadata`](crate::pmtiles::Metadata) parses the whole object or none
    /// of it, and an unknown `format` variant written by a later libviprs
    /// fails it at the outermost object. `describe()` used to swallow that
    /// with `.ok()?` and answer `tile_size: None, layout: None, format: None`
    /// for an archive that records all three.
    ///
    /// The quiet version is worse than it sounds, and not because information
    /// is lost. `format: None` is *already* the legitimate answer for a
    /// foreign go-pmtiles archive that carries no libviprs namespace, so the
    /// two cases were indistinguishable, and they want opposite reactions:
    /// one is "this file was made by another tool, read it as best you can",
    /// the other is "this file was made by libviprs and your libviprs is too
    /// old for it, upgrade". Only the second has an action attached, which is
    /// why this error names the version that wrote the archive.
    ///
    /// Two honest caveats, both from the issue. This cannot fix 0.5.x, which
    /// will do the silent downgrade forever, so the payoff is at the *next*
    /// variant addition rather than at this one. And making an unparseable
    /// namespace not sink the rest of the `Metadata` object (so `name` and
    /// `extra` survive) is a real improvement and a separate decision, not
    /// bundled here.
    #[error(
        "this archive was written by libviprs {libviprs_version} and this build \
         ({}) cannot parse what it recorded: {source}",
        env!("CARGO_PKG_VERSION")
    )]
    MetadataFromANewerLibviprs {
        /// The `libviprs_version` string the archive records, lifted out of
        /// the raw JSON because the typed parse is the thing that failed.
        libviprs_version: String,
        /// The parse failure itself, typed rather than rendered, so a caller
        /// can still see which key serde gave up on.
        #[source]
        source: crate::pmtiles::PmTilesError,
    },
}

// ---------------------------------------------------------------------------
// PyramidDescription
// ---------------------------------------------------------------------------

/// What a pyramid is, as far as its storage can say.
///
/// Every field past the level range is an `Option`, because the two backends
/// know different things. A directory reader holds the plan that produced the
/// tree and can answer all of it; an archive carries whatever its writer chose
/// to record, and a foreign archive may carry none of it. An `Option` is the
/// honest shape for "this backend does not know", and it is better than a
/// plausible default that a caller cannot tell apart from a measurement.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PyramidDescription {
    /// Lowest level with tiles.
    pub min_level: u32,
    /// Highest level with tiles.
    pub max_level: u32,
    /// Tile edge in pixels, when the backend records it.
    pub tile_size: Option<u32>,
    /// The layout the tiles were placed with, when the backend records it.
    pub layout: Option<Layout>,
    /// The encoding the stored bytes are in, when the backend commits to one.
    pub format: Option<TileFormat>,
    /// Width in pixels of the source the pyramid was generated from, when the
    /// backend records it.
    ///
    /// # Why a level range is not this (issue #1130)
    ///
    /// A pyramid's level range is fixed by the longest side rounded up to a
    /// power of two, and its grid at every level is the level's size divided
    /// by the tile size and rounded up. Both of those throw information away,
    /// so a 4000-pixel source and a 4096-pixel one produce the same thirteen
    /// levels, the same grid at each of them and the same 349 coordinates.
    /// Every check a verify had passed on an archive of a different picture,
    /// and the number that tells the two apart was sitting in the archive's
    /// own metadata the whole time.
    pub source_width: Option<u32>,
    /// Height in pixels of the source the pyramid was generated from, when the
    /// backend records it. See [`source_width`](Self::source_width).
    pub source_height: Option<u32>,
    /// Overlap in pixels between neighbouring tiles, when the backend records
    /// it.
    ///
    /// Also from #1130, and the sharper half of it. Overlap does not reach the
    /// grid at all: `tile_grid` only divides by the tile size, so planning a
    /// source at overlap 1 and at overlap 0 produces byte-identical `levels`
    /// vectors. What it changes is `tile_rect`, which is to say every single
    /// tile's pixels. Two archives that disagree about it are entirely
    /// different pyramids that no count can tell apart.
    pub overlap: Option<u32>,
}

impl PyramidDescription {
    /// A description of a pyramid spanning `min_level..=max_level`, with
    /// nothing else recorded yet.
    ///
    /// This exists because the struct is `#[non_exhaustive]` and the trait
    /// that returns it is public. Without a constructor, a downstream
    /// implementor of [`PyramidReader`] had exactly one legal body for
    /// `describe()`, which was to return an error, so the extension point was
    /// public in name only. The crate's own suite demonstrated the asymmetry
    /// in both directions at once: in-crate code built one with a struct
    /// literal (`#[non_exhaustive]` does not apply inside the defining crate)
    /// while a test compiling as an external crate could only refuse.
    ///
    /// The two levels are arguments rather than defaults because there is no
    /// honest "unknown" for them. Every other field defaults to `None`, which
    /// is the honest unknown, and each has a `with_*` setter, matching the
    /// `default()` plus `with_*` convention `tests/non_exhaustive_options.rs`
    /// already holds every public options struct to.
    pub fn new(min_level: u32, max_level: u32) -> Self {
        Self {
            min_level,
            max_level,
            tile_size: None,
            layout: None,
            format: None,
            source_width: None,
            source_height: None,
            overlap: None,
        }
    }

    /// Record the tile edge in pixels.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = Some(tile_size);
        self
    }

    /// Record the layout the tiles were placed with.
    #[must_use]
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Record the encoding the stored bytes are in.
    #[must_use]
    pub fn with_format(mut self, format: TileFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Record the source's pixel dimensions.
    ///
    /// The two travel together because they are useless apart: a pyramid that
    /// knew its source was 4096 wide and had no idea how tall it was could
    /// still be checked in one direction, and a caller would have to reason
    /// about which half of the answer it got. Both or neither.
    #[must_use]
    pub fn with_source_size(mut self, width: u32, height: u32) -> Self {
        self.source_width = Some(width);
        self.source_height = Some(height);
        self
    }

    /// Record the overlap in pixels between neighbouring tiles.
    #[must_use]
    pub fn with_overlap(mut self, overlap: u32) -> Self {
        self.overlap = Some(overlap);
        self
    }
}

// ---------------------------------------------------------------------------
// StructuralSummary
// ---------------------------------------------------------------------------

/// What one walk of a pyramid's own structure found.
///
/// This exists because a verify used to ask two questions that are answered by
/// the same walk, [`PyramidReader::self_check`] and
/// [`PyramidReader::addressed_tiles`], and paid for the walk twice (issue
/// #1130). On a local file nobody notices; over an injected transport it is
/// two full sets of round trips, and past roughly 262144 tiles the reader's
/// leaf cache evicts between them so the second walk refetches what the first
/// one read.
///
/// So the walk is one method now and these are its answers. The two views are
/// still there, defaulted on top of this, so a backend implementing either of
/// them keeps working and a backend implementing this gets both for free.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StructuralSummary {
    /// How many distinct coordinates the storage addresses, or `None` for a
    /// backend that cannot count them.
    ///
    /// `None` rather than `0`, because a backend that cannot count and a
    /// pyramid with nothing in it want opposite reactions and the second one
    /// is a defect.
    pub addressed_tiles: Option<u64>,
    /// Whether the walk checked every offset it read against a size the
    /// storage actually reported.
    ///
    /// This is the guarantee a length cannot give and a payload read can: that
    /// the bytes an index entry points at are inside the object and therefore
    /// reachable. A walk that had the size proved it for every entry at once,
    /// which is what lets a verify take a tile's length instead of its
    /// payload. A walk that did not, or a backend with no structure to walk at
    /// all, leaves `false` here and the verify reads the bytes.
    ///
    /// Defaulting to `false` is the conservative direction on purpose: a
    /// backend that says nothing keeps the behaviour it had.
    pub offsets_bounded: bool,
}

impl StructuralSummary {
    /// A summary of a walk that counted nothing and bounded nothing, which is
    /// the honest starting point for a backend with no structure to check.
    pub fn new() -> Self {
        Self {
            addressed_tiles: None,
            offsets_bounded: false,
        }
    }

    /// Record how many coordinates the storage addresses.
    #[must_use]
    pub fn with_addressed_tiles(mut self, addressed_tiles: u64) -> Self {
        self.addressed_tiles = Some(addressed_tiles);
        self
    }

    /// Record whether the walk bounded every offset against a reported size.
    #[must_use]
    pub fn with_offsets_bounded(mut self, offsets_bounded: bool) -> Self {
        self.offsets_bounded = offsets_bounded;
        self
    }
}

impl Default for StructuralSummary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PyramidReader
// ---------------------------------------------------------------------------

/// One way to read a generated pyramid, whatever it was stored in.
///
/// # Examples
///
/// ```
/// use libviprs::planner::{Layout, PyramidPlanner, TileCoord};
/// use libviprs::pyramid_reader::{DirectoryPyramidReader, PyramidReader};
/// use libviprs::sink::TileFormat;
/// use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dir = tempfile::tempdir()?;
/// let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz)?.plan();
/// let src = Raster::new(256, 256, PixelFormat::Rgb8, vec![9u8; 256 * 256 * 3])?;
///
/// let root = dir.path().join("tiles");
/// EngineBuilder::new(&src, plan.clone(), FsSink::new(&root, plan.clone())).run()?;
///
/// let reader = DirectoryPyramidReader::try_open(&root, plan, TileFormat::Png)?;
/// let described = reader.describe()?;
/// assert_eq!(described.tile_size, Some(256));
///
/// let coord = TileCoord { level: described.max_level, col: 0, row: 0 };
/// assert!(reader.tile(coord)?.is_some());
/// # Ok(())
/// # }
/// ```
pub trait PyramidReader: Send + Sync {
    /// What the pyramid is: levels, tile size, layout, encoding.
    fn describe(&self) -> Result<PyramidDescription, PyramidReadError>;

    /// Walk the storage's own structure once and report what the walk found.
    ///
    /// This is the half of a verify that has nothing to do with what was
    /// asked for: whether the thing on disk is internally consistent, whether
    /// every offset it carries lands inside itself, whether its own counts add
    /// up. A backend with nothing to check answers
    /// [`StructuralSummary::new`], which is the default, and a loose-file tree
    /// genuinely has nothing: a directory of files has no index to disagree
    /// with itself.
    ///
    /// The contract that makes a clean summary meaningful is the one
    /// [`validate::Report::is_valid`](crate::pmtiles::validate::Report::is_valid)
    /// rests on: **anything that stops the walk early must also report a
    /// defect**. Without it a storage that made the walk give up quietly would
    /// answer clean, which is worse than answering with the defect, because
    /// the checks a walk only reaches at the end never ran.
    ///
    /// # Implement this one, not the two below
    ///
    /// [`self_check`](Self::self_check) and
    /// [`addressed_tiles`](Self::addressed_tiles) are defaulted views over
    /// this, because they were two questions about one walk and a verify was
    /// asking both (issue #1130). A backend that overrides them individually
    /// still works exactly as it did, which is why they are still overridable
    /// at all; a backend that overrides this pays for one walk instead of two.
    fn structural_summary(&self) -> Result<StructuralSummary, PyramidReadError> {
        Ok(StructuralSummary::new())
    }

    /// Check the storage's own structure, with no plan to check it against.
    ///
    /// A view over [`structural_summary`](Self::structural_summary): the walk
    /// refuses a damaged storage by returning `Err`, so reaching a summary at
    /// all is the check passing, and the summary's contents are somebody
    /// else's question.
    fn self_check(&self) -> Result<(), PyramidReadError> {
        self.structural_summary().map(|_| ())
    }

    /// How many distinct coordinates the pyramid holds a tile for.
    ///
    /// Recounted from the storage, never read out of a header it also wrote:
    /// a header that lies about its own count is exactly the defect worth
    /// catching, and a count taken from it agrees with itself whatever it
    /// says.
    ///
    /// This is what makes a plan-aware verify possible in the direction a
    /// per-coordinate sweep cannot see. A sweep only asks about coordinates
    /// the plan names, so a pyramid holding *more* than the plan resolves
    /// every question it is asked and is still not the pyramid that plan
    /// produced. Comparing this against `plan.tile_coords().count()` is the
    /// only check that notices.
    ///
    /// # Errors
    ///
    /// [`PyramidReadError::NotCountable`], not `0` and not an `Option`. A
    /// backend that cannot count has to say so loudly, because the failure
    /// mode of a quiet "unknown" is a verify that silently drops its only
    /// both-directions check and stays green. That is what the `None` in
    /// [`StructuralSummary::addressed_tiles`] becomes here.
    fn addressed_tiles(&self) -> Result<u64, PyramidReadError> {
        self.structural_summary()?.addressed_tiles.ok_or_else(|| {
            PyramidReadError::NotCountable(
                "this pyramid cannot count the coordinates it addresses".to_string(),
            )
        })
    }

    /// The stored bytes of one tile, or `None` when the pyramid has no tile
    /// there.
    ///
    /// The bytes come back exactly as they are stored, which for every backend
    /// in this crate means the encoded image, not decoded pixels. An absent
    /// tile is `Ok(None)`, never an error.
    fn tile(&self, coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError>;

    /// How many bytes the pyramid stores for one tile, or `None` when it has
    /// no tile there.
    ///
    /// The same answer [`tile`](Self::tile) would give for
    /// `.map(|bytes| bytes.len())`, which is exactly what the default does, so
    /// no backend breaks by not implementing this. A backend whose index
    /// already carries the length overrides it and stops reading the payload:
    /// that is the difference between reading an index and reading an archive,
    /// and over a ranged transport it is the difference between a directory
    /// walk and one round trip per tile (issue #1130).
    ///
    /// # What this proves, and what it does not
    ///
    /// Present, and how many bytes. **Not** that the bytes are readable. That
    /// last one is the only thing a payload read adds, and whether it is worth
    /// the archive is what [`StructuralSummary::offsets_bounded`] answers: a
    /// walk that bounds-checked every entry against a reported size has
    /// already established reachability for all of them at once.
    fn tile_len(&self, coord: TileCoord) -> Result<Option<u64>, PyramidReadError> {
        Ok(self.tile(coord)?.map(|bytes| bytes.len() as u64))
    }

    /// The encoding the stored bytes are in, when the backend commits to one.
    fn tile_format(&self) -> Option<TileFormat> {
        None
    }
}

// ---------------------------------------------------------------------------
// DirectoryPyramidReader
// ---------------------------------------------------------------------------

/// A pyramid stored as loose files under the layout's own tile paths.
///
/// Reads through [`PyramidPlan::tile_path`], the same function
/// [`FsSink`](crate::sink::FsSink) writes through, so the two cannot drift
/// apart on where a tile lives.
#[derive(Debug)]
pub struct DirectoryPyramidReader {
    base_dir: PathBuf,
    plan: PyramidPlan,
    format: TileFormat,
}

impl DirectoryPyramidReader {
    /// Open the tree at `base_dir` as the pyramid `plan` describes.
    ///
    /// The plan is required rather than inferred. A directory of tiles does
    /// not say what its level indices mean, how big a tile is or which layout
    /// placed it, and a reader that guessed from the directory names would be
    /// inventing a description rather than reporting one.
    ///
    /// # Errors
    ///
    /// [`PyramidReadError::NotAPyramid`] when `base_dir` is not a directory.
    pub fn try_open(
        base_dir: impl Into<PathBuf>,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Result<Self, PyramidReadError> {
        let base_dir = base_dir.into();
        if !base_dir.is_dir() {
            return Err(PyramidReadError::NotAPyramid {
                path: base_dir,
                expected: "a directory of tiles".to_string(),
            });
        }
        Ok(Self {
            base_dir,
            plan,
            format,
        })
    }

    /// The directory the tiles are read from.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// The plan the tree was written with.
    pub fn plan(&self) -> &PyramidPlan {
        &self.plan
    }
}

impl PyramidReader for DirectoryPyramidReader {
    fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
        let mut levels = self.plan.levels.iter().map(|level| level.level);
        let first = levels
            .next()
            .ok_or_else(|| PyramidReadError::NoDescription("the plan has no levels".to_string()))?;
        let (min_level, max_level) = levels.fold((first, first), |(lo, hi), level| {
            (lo.min(level), hi.max(level))
        });

        Ok(PyramidDescription {
            min_level,
            max_level,
            tile_size: Some(self.plan.tile_size),
            layout: Some(self.plan.layout),
            format: Some(self.format),
            source_width: Some(self.plan.image_width),
            source_height: Some(self.plan.image_height),
            overlap: Some(self.plan.overlap),
        })
    }

    fn tile(&self, coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError> {
        // `tile_path` answers `None` for a coordinate outside the plan's grid,
        // which is a tile the pyramid does not have rather than a failure.
        let Some(relative) = self.plan.tile_path(coord, self.format.extension()) else {
            return Ok(None);
        };
        match std::fs::read(self.base_dir.join(relative)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(PyramidReadError::Io(e)),
        }
    }

    /// The file's size, taken from its metadata rather than from its bytes.
    ///
    /// A tree has no index, so this is the closest thing to one it has: the
    /// directory entry already knows how long the file is and reading it
    /// learns nothing else. An absent file is `Ok(None)` on the same terms
    /// [`PyramidReader::tile`] uses, and so is a coordinate the
    /// plan's grid does not contain.
    ///
    /// Note what this does **not** do for a verify: a tree reports no
    /// [`StructuralSummary::offsets_bounded`], because it has no walk to bound
    /// anything with, so `pyramid_verify` still reads the payloads here. That
    /// is deliberate. A length off `stat` says the file exists and how large
    /// it is; it does not say the bytes come back.
    fn tile_len(&self, coord: TileCoord) -> Result<Option<u64>, PyramidReadError> {
        let Some(relative) = self.plan.tile_path(coord, self.format.extension()) else {
            return Ok(None);
        };
        match std::fs::metadata(self.base_dir.join(relative)) {
            Ok(metadata) => Ok(Some(metadata.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(PyramidReadError::Io(e)),
        }
    }

    fn tile_format(&self) -> Option<TileFormat> {
        Some(self.format)
    }
}

// ---------------------------------------------------------------------------
// PmTilesPyramidReader
// ---------------------------------------------------------------------------

/// A pyramid stored as one PMTiles v3 archive.
///
/// Wraps the indexed [`Reader`](crate::pmtiles::Reader) from #988 and maps
/// [`TileCoord`] onto `(z, x, y)` through
/// [`tile_coord_to_zxy`](crate::sink_pmtiles::tile_coord_to_zxy), the same
/// function [`PmTilesSink`](crate::sink_pmtiles::PmTilesSink) writes through,
/// so the two cannot drift apart on where a tile lives.
///
/// # Why the type parameter is defaulted
///
/// `R` is whatever the archive's bytes come from, and it defaults to
/// [`FileRangeReader`](crate::pmtiles::FileRangeReader) so that
/// `PmTilesPyramidReader` keeps meaning exactly what it meant before #1121.
/// Every existing call site spells the type with no parameter and still
/// compiles, and [`try_open`](Self::try_open) still hands back the local-file
/// instantiation.
///
/// This mirrors [`Reader<R>`](crate::pmtiles::Reader) and `Reader::try_open`
/// one level up, and it is deliberately not `Reader<Box<dyn RangeReader>>`.
/// That shape would change the public signatures of
/// [`from_reader`](Self::from_reader) and [`reader`](Self::reader), which is a
/// breaking change for nothing, and `Box<dyn RangeReader>` is not `Debug`, so
/// it would silently drop `Debug` from a public type. Anyone who does want the
/// boxed shape can still have it: `PmTilesPyramidReader<Box<dyn RangeReader>>`
/// works, because `Box<R>` implements the trait.
#[derive(Debug)]
pub struct PmTilesPyramidReader<R: crate::pmtiles::RangeReader = crate::pmtiles::FileRangeReader> {
    reader: crate::pmtiles::Reader<R>,
}

impl PmTilesPyramidReader<crate::pmtiles::FileRangeReader> {
    /// Open the archive at `path`.
    ///
    /// Reads the header and the root directory and nothing else; a tile is
    /// fetched when it is asked for.
    ///
    /// # Errors
    ///
    /// [`PyramidReadError::PmTiles`] for a file that is not a readable v3
    /// archive, and [`PyramidReadError::Io`] if it cannot be opened at all.
    pub fn try_open(path: impl AsRef<Path>) -> Result<Self, PyramidReadError> {
        Ok(Self {
            reader: crate::pmtiles::Reader::try_open(path)?,
        })
    }
}

#[cfg(feature = "object-store-sink")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store-sink")))]
impl PmTilesPyramidReader<crate::pmtiles::ObjectStoreRangeReader> {
    /// Open the archive stored at `key` in an injected object store.
    ///
    /// The counterpart of
    /// [`ObjectStoreSink`](crate::sink_object_store::ObjectStoreSink) on the
    /// way back out. libviprs ships no HTTP or S3 client and #1119 records
    /// that as a permanent decision, so the store is the caller's: anything
    /// that can answer
    /// [`ObjectStore::get_range`](crate::sink_object_store::ObjectStore::get_range)
    /// serves an archive here.
    ///
    /// Two requests happen at open, the header and the root, plus one
    /// [`size`](crate::sink_object_store::ObjectStore::size). A store that
    /// inherits the defaulted refusal for `size` still opens; what it gives up
    /// is the reader's section bounds checks.
    ///
    /// # Errors
    ///
    /// [`PyramidReadError::PmTiles`] when the object is not a readable v3
    /// archive or the store refused a range.
    pub fn try_from_object_store(
        store: std::sync::Arc<dyn crate::sink_object_store::ObjectStore>,
        key: impl Into<String>,
    ) -> Result<Self, PyramidReadError> {
        Ok(Self {
            reader: crate::pmtiles::Reader::try_new(crate::pmtiles::ObjectStoreRangeReader::new(
                store, key,
            ))?,
        })
    }
}

impl<R: crate::pmtiles::RangeReader> PmTilesPyramidReader<R> {
    /// Wrap a reader the caller already opened.
    pub fn from_reader(reader: crate::pmtiles::Reader<R>) -> Self {
        Self { reader }
    }

    /// The archive reader underneath, for the questions this trait does not
    /// ask: the raw header, the root entries, the bounding box.
    pub fn reader(&self) -> &crate::pmtiles::Reader<R> {
        &self.reader
    }

    /// Walk the archive and hand back the report, refusing a damaged one.
    ///
    /// [`PyramidReader::structural_summary`] goes through here, so a caller
    /// that wants the count of a broken archive cannot get one: a count taken
    /// from a walk that raised findings is a count of however far the walk
    /// got.
    ///
    /// Nothing is cached across calls, and that part of the old argument still
    /// holds: the alternative is a structural verdict about a file that can
    /// change underneath it, and a stale verdict is a worse thing to own than
    /// a repeated directory walk.
    ///
    /// What used to be here was a defence of calling this **twice inside one
    /// verify**, once through `self_check` and once through
    /// `addressed_tiles`, ten lines apart, under a run lock that already
    /// guarantees the archive cannot change between them. That was never an
    /// argument about caching. On a local file it is two directory traversals
    /// and nobody notices; over an injected transport it is two full sets of
    /// round trips, and above roughly 262144 tiles the leaf cache (64 pages)
    /// evicts between them so the second walk refetches what the first one
    /// read. One method returning both answers removes the choice, which is
    /// what `structural_summary` is (issue #1130).
    fn structural_report(&self) -> Result<crate::pmtiles::validate::Report, PyramidReadError> {
        let report = crate::pmtiles::validate::validate(
            self.reader.source(),
            &crate::pmtiles::validate::ValidationLimits::default(),
        )?;
        if report.is_valid() {
            return Ok(report);
        }
        Err(PyramidReadError::StructuralDefects {
            findings: report.findings.clone(),
        })
    }

    /// What the archive says libviprs recorded about the run that produced it,
    /// when it was libviprs that produced it.
    fn generation(&self) -> Option<crate::manifest::GenerationSettings> {
        self.reader
            .metadata()
            .ok()?
            .vnd_libviprs
            .as_ref()?
            .generation
            .clone()
    }

    /// What the archive says about the raster it was generated from.
    ///
    /// `None` for a foreign archive, and also for one libviprs assembled from
    /// loose tiles rather than generated from a source, which is the case
    /// [`LibviprsMetadata::source`](crate::pmtiles::LibviprsMetadata::source)
    /// documents.
    fn source(&self) -> Option<crate::manifest::SourceMetadata> {
        self.reader
            .metadata()
            .ok()?
            .vnd_libviprs
            .as_ref()?
            .source
            .clone()
    }

    /// Decide whether a failed metadata parse is this build being out of date.
    ///
    /// `Ok(())` means it is not: either the bytes are unreadable too, or they
    /// carry no `vnd.libviprs` key, and in both cases the archive is somebody
    /// else's and `None` is the honest description. `Err` means the archive
    /// says libviprs wrote it, so the failure is a version gap and the caller
    /// should say so with the version attached.
    ///
    /// Reading the section a second time is the cost. It is paid only on the
    /// failure path (`Reader::metadata` caches its successes and nothing else
    /// calls this), and the alternative is caching two representations of one
    /// section that can disagree.
    ///
    /// The raw scan is a `serde_json::Value` lookup rather than a typed parse
    /// on purpose: a typed parse is the thing that just failed, and the only
    /// question left is whether one key is present and what string sits under
    /// it. A `vnd.libviprs` that is not an object, or that carries no
    /// `libviprs_version`, still counts as ours: the key is the claim, and an
    /// unnamed version becomes `"(unrecorded)"` rather than sending the caller
    /// back to the indistinguishable `None`.
    fn version_gap(
        &self,
        parse_failure: crate::pmtiles::PmTilesError,
    ) -> Result<(), PyramidReadError> {
        let Ok(bytes) = self.reader.metadata_json() else {
            return Ok(());
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return Ok(());
        };
        let Some(vnd) = value.get(crate::pmtiles::metadata::LIBVIPRS_METADATA_KEY) else {
            return Ok(());
        };
        let libviprs_version = vnd
            .get("libviprs_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(unrecorded)")
            .to_string();
        Err(PyramidReadError::MetadataFromANewerLibviprs {
            libviprs_version,
            source: parse_failure,
        })
    }
}

impl<R: crate::pmtiles::RangeReader> PyramidReader for PmTilesPyramidReader<R> {
    /// Walk the archive the way [`validate`](crate::pmtiles::validate) does,
    /// refuse it if the walk found anything, and keep both of the numbers a
    /// verify is going to want.
    ///
    /// Read through [`Report::is_valid`](crate::pmtiles::validate::Report::is_valid)
    /// rather than through a severity filter or a hand-rolled early exit, and
    /// that is the whole point: the walk's invariant is that anything stopping
    /// it early also raises a finding, so "no findings" is the only phrasing
    /// that a truncated walk cannot satisfy. A check that skipped straight to
    /// the counts, or that ignored findings it decided were cosmetic, would
    /// report clean for a file that made the walk give up before it got to
    /// them.
    ///
    /// The count is the run lengths summed, recomputed by this walk. The
    /// header carries an `addressed_tiles_count` and it is not used: it is a
    /// number the writer put in the file, so an archive whose header miscounts
    /// its own tiles would agree with itself perfectly, and the disagreement
    /// between the header's count and what the directories actually cover is
    /// itself one of the findings above.
    ///
    /// `offsets_bounded` comes from the reader's own
    /// [`archive_size`](crate::pmtiles::Reader::archive_size), which is what
    /// the walk bounds every section and every entry against. It is `Some` for
    /// a file and for any transport that answers
    /// [`RangeReader::size`](crate::pmtiles::RangeReader::size).
    ///
    /// Which makes it `true` for every archive that gets this far today, and
    /// that is worth saying rather than leaving as an implication. A backend
    /// that answers `None` produces a
    /// [`Finding::ArchiveSizeUnknown`](crate::pmtiles::validate::Finding::ArchiveSizeUnknown),
    /// and `structural_report` refuses any report carrying a finding, so the
    /// archive never reaches a summary at all. Reading the size here rather
    /// than hard-coding `true` is deliberate: it keeps the flag a statement
    /// about what this walk actually checked, so if that finding is ever
    /// downgraded the conditional in `pyramid_verify` is already correct
    /// instead of quietly wrong.
    fn structural_summary(&self) -> Result<StructuralSummary, PyramidReadError> {
        let report = self.structural_report()?;
        Ok(StructuralSummary::new()
            .with_addressed_tiles(report.addressed_tiles)
            .with_offsets_bounded(self.reader.archive_size().is_some()))
    }

    /// What the archive says it is.
    ///
    /// # An archive from a newer libviprs is named, not blanked (issue #1123)
    ///
    /// Every `Option` here can legitimately be `None`, because a foreign
    /// archive records none of it. That is exactly what made the old
    /// behaviour dangerous: when a metadata object failed to parse, this
    /// returned the same all-`None` description a go-pmtiles archive gets, so
    /// "made by another tool" and "made by libviprs and I am too old" looked
    /// identical and only one of them has an action attached.
    ///
    /// So a parse failure is now interrogated rather than swallowed. If the
    /// raw bytes carry a `vnd.libviprs` key the archive is ours and this
    /// refuses with [`PyramidReadError::MetadataFromANewerLibviprs`], naming
    /// the version that wrote it. If they do not, `None` stays `None` and
    /// every foreign archive reads exactly as it did before.
    fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
        let header = self.reader.header();
        if let Err(parse_failure) = self.reader.metadata() {
            self.version_gap(parse_failure)?;
        }
        let generation = self.generation();
        let source = self.source();
        Ok(PyramidDescription {
            min_level: u32::from(header.min_zoom),
            max_level: u32::from(header.max_zoom),
            tile_size: generation.as_ref().map(|g| g.tile_size),
            layout: generation.as_ref().map(|g| g.layout),
            format: self.tile_format(),
            source_width: source.as_ref().map(|s| s.width),
            source_height: source.as_ref().map(|s| s.height),
            overlap: generation.as_ref().map(|g| g.overlap),
        })
    }

    fn tile(&self, coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError> {
        // A coordinate PMTiles cannot address is a tile this pyramid does not
        // have, the same answer the directory reader gives for a coordinate
        // outside the plan. It is not an error, and it is emphatically not
        // masked into a different, valid tile.
        let Ok((z, x, y)) = crate::sink_pmtiles::tile_coord_to_zxy(coord) else {
            return Ok(None);
        };
        Ok(self.reader.get_tile(z, x, y)?)
    }

    /// The stored length out of the directory entry, with no payload read.
    ///
    /// Answers `None` for a coordinate PMTiles cannot address on the same
    /// terms [`PyramidReader::tile`] does, so the two agree about
    /// which coordinates this pyramid has and disagree only about how much
    /// they cost.
    fn tile_len(&self, coord: TileCoord) -> Result<Option<u64>, PyramidReadError> {
        let Ok((z, x, y)) = crate::sink_pmtiles::tile_coord_to_zxy(coord) else {
            return Ok(None);
        };
        Ok(self
            .reader
            .tile_span(z, x, y)?
            .map(|(_offset, length)| u64::from(length)))
    }

    /// The encoding the stored bytes are in.
    ///
    /// From the `vnd.libviprs` namespace when the archive carries it, because
    /// that is the only place the JPEG quality a [`TileFormat::Jpeg`] carries
    /// is written down. A foreign JPEG archive therefore reports `None` rather
    /// than a quality nobody measured; ask
    /// [`Reader::tile_format`](crate::pmtiles::Reader::tile_format) for the
    /// `TileType`, which is what the archive actually records.
    ///
    /// The fallback answers for the parameterless formats and only those. PNG
    /// has always been one; WebP joined it in issue #1123, and for the same
    /// reason rather than a weaker one: [`TileFormat::Webp`] carries no field
    /// that the header's `TileType` fails to determine, so an archive stamped
    /// `0x04` by any tool at all *is* a WebP pyramid and saying so invents
    /// nothing. JPEG is the odd one out here, not WebP.
    ///
    /// This arm is one of the five sites issue #1123 was written about. It has
    /// a catch-all, so adding a variant to `TileFormat` did not break it and a
    /// WebP archive would have gone on reporting `format: None` indefinitely.
    fn tile_format(&self) -> Option<TileFormat> {
        if let Some(generation) = self.generation() {
            return Some(generation.format);
        }
        match self.reader.tile_format() {
            crate::pmtiles::TileType::Png => Some(TileFormat::Png),
            crate::pmtiles::TileType::Webp => Some(TileFormat::Webp),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PyramidPlanner;

    fn plan() -> PyramidPlan {
        PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz)
            .expect("a square plan is valid")
            .plan()
    }

    /// Opening something that is not a directory is a typed refusal rather
    /// than a reader that answers `None` for everything.
    ///
    /// A reader over a path that does not exist would be indistinguishable
    /// from a reader over an empty pyramid, and every test written against it
    /// would pass.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn opening_a_path_that_is_not_a_directory_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("not-a-tree");
        std::fs::write(&file, b"x").expect("write");

        for candidate in [file, dir.path().join("absent")] {
            match DirectoryPyramidReader::try_open(&candidate, plan(), TileFormat::Png) {
                Err(PyramidReadError::NotAPyramid { .. }) => {}
                other => panic!("{} must be refused, got {other:?}", candidate.display()),
            }
        }
    }

    /// A tile that is simply not on disk is absent, and one the plan does not
    /// have is absent too. Neither is an error.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_missing_tile_and_an_impossible_coordinate_are_both_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reader = DirectoryPyramidReader::try_open(dir.path(), plan(), TileFormat::Png)
            .expect("an empty directory is still a directory");

        let real = TileCoord {
            level: 0,
            col: 0,
            row: 0,
        };
        assert!(
            reader.plan().tile_path(real, "png").is_some(),
            "the positive control: this coordinate is one the plan has, so the \
             `None` below is about the file being missing"
        );
        assert_eq!(reader.tile(real).expect("absence is not an error"), None);

        let impossible = TileCoord {
            level: 99,
            col: 0,
            row: 0,
        };
        assert_eq!(reader.tile(impossible).expect("out of range"), None);
    }

    /// A backend that cannot count the coordinates it addresses says so,
    /// rather than answering zero or `None`.
    ///
    /// The default matters more than it looks. `addressed_tiles` is the only
    /// check that catches a pyramid holding more than the plan asked for, and
    /// a quiet "unknown" would let a verify drop that check and stay green for
    /// every reader that never implemented it. The directory reader is the one
    /// in-tree backend sitting on the default, so it is the one that pins it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_reader_that_cannot_count_its_tiles_refuses_rather_than_guessing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reader = DirectoryPyramidReader::try_open(dir.path(), plan(), TileFormat::Png)
            .expect("an empty directory is still a directory");

        // The control: the same reader answers the questions it can answer, so
        // the refusal below is about the count and not about the reader.
        assert!(reader.describe().is_ok(), "it can still describe itself");
        assert!(reader.self_check().is_ok(), "a tree has no index to damage");

        match reader.addressed_tiles() {
            Err(PyramidReadError::NotCountable(_)) => {}
            other => panic!("a reader that cannot count must say so, got {other:?}"),
        }

        // And it must say so in its OWN words. This used to match
        // `NoDescription`, which is the variant for "I cannot say what this
        // pyramid is". Asserting it here made the test pass while pinning the
        // wrong fact: a caller matching `NoDescription` to decide a backend
        // carries no metadata would also have caught every backend that simply
        // cannot count, and the two want opposite handling. The control above
        // is what makes the distinction visible, because this reader describes
        // itself perfectly well.
        assert!(
            !matches!(
                reader.addressed_tiles(),
                Err(PyramidReadError::NoDescription(_))
            ),
            "the count refusal must not borrow the variant that means \"I cannot describe myself\""
        );
    }
}
