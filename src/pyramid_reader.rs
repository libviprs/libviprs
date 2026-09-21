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
    /// honest "unknown" for them. The three `Option` fields default to `None`,
    /// which is the honest unknown, and each has a `with_*` setter, matching
    /// the `default()` plus `with_*` convention `tests/non_exhaustive_options.rs`
    /// already holds every public options struct to.
    pub fn new(min_level: u32, max_level: u32) -> Self {
        Self {
            min_level,
            max_level,
            tile_size: None,
            layout: None,
            format: None,
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

    /// Check the storage's own structure, with no plan to check it against.
    ///
    /// This is the half of a verify that has nothing to do with what was
    /// asked for: whether the thing on disk is internally consistent, whether
    /// every offset it carries lands inside itself, whether its own counts add
    /// up. A backend with nothing to check answers `Ok(())`, which is the
    /// default, and a loose-file tree genuinely has nothing: a directory of
    /// files has no index to disagree with itself.
    ///
    /// The contract that makes `Ok(())` meaningful is the one
    /// [`validate::Report::is_valid`](crate::pmtiles::validate::Report::is_valid)
    /// rests on: **anything that stops the walk early must also report a
    /// defect**. Without it a storage that made the walk give up quietly would
    /// answer `Ok(())`, which is worse than answering with the defect, because
    /// the checks a walk only reaches at the end never ran.
    fn self_check(&self) -> Result<(), PyramidReadError> {
        Ok(())
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
    /// The default is [`PyramidReadError::NoDescription`], not `0` and not an
    /// `Option`. A backend that cannot count has to say so loudly, because
    /// the failure mode of a quiet "unknown" is a verify that silently drops
    /// its only both-directions check and stays green.
    fn addressed_tiles(&self) -> Result<u64, PyramidReadError> {
        Err(PyramidReadError::NotCountable(
            "this pyramid cannot count the coordinates it addresses".to_string(),
        ))
    }

    /// The stored bytes of one tile, or `None` when the pyramid has no tile
    /// there.
    ///
    /// The bytes come back exactly as they are stored, which for every backend
    /// in this crate means the encoded image, not decoded pixels. An absent
    /// tile is `Ok(None)`, never an error.
    fn tile(&self, coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError>;

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
    /// Both [`PyramidReader::self_check`] and
    /// [`PyramidReader::addressed_tiles`] go through here, so a caller that
    /// wants the count of a broken archive cannot get one: a count taken from
    /// a walk that raised findings is a count of however far the walk got.
    ///
    /// A verify therefore pays this walk twice. That is deliberate and it is
    /// cheap: it reads the header and the directories and no tile payloads,
    /// while the coordinate sweep that follows reads every tile in the
    /// archive. The alternative is caching a report against a file that can
    /// change underneath it, and a stale structural verdict is a worse thing
    /// to own than a second directory walk.
    ///
    /// Be honest about what that costs, though, because "cheap" was doing too
    /// much work in the sentence above. A single `pyramid_verify` calls this
    /// twice, once through `self_check` and once through `addressed_tiles`,
    /// ten lines apart, under a run lock that already guarantees the archive
    /// cannot change between them. On a local file that is two directory
    /// traversals and nobody notices. Over an injected transport it is two
    /// full sets of round trips, and above roughly 262144 tiles the leaf cache
    /// (64 pages) evicts between them, so the second walk refetches what the
    /// first one read. Accepting the double walk is a deliberate trade against
    /// a stale cache, not a claim that it is free, and the shape that removes
    /// the choice entirely is one method returning both answers.
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
}

impl<R: crate::pmtiles::RangeReader> PyramidReader for PmTilesPyramidReader<R> {
    /// Walk the archive the way [`validate`](crate::pmtiles::validate) does
    /// and refuse it if the walk found anything.
    ///
    /// Read through [`Report::is_valid`](crate::pmtiles::validate::Report::is_valid)
    /// rather than through a severity filter or a hand-rolled early exit, and
    /// that is the whole point: the walk's invariant is that anything stopping
    /// it early also raises a finding, so "no findings" is the only phrasing
    /// that a truncated walk cannot satisfy. A check that skipped straight to
    /// the counts, or that ignored findings it decided were cosmetic, would
    /// report clean for a file that made the walk give up before it got to
    /// them.
    fn self_check(&self) -> Result<(), PyramidReadError> {
        self.structural_report().map(|_| ())
    }

    /// The run lengths summed, recomputed by the same walk.
    ///
    /// The header carries an `addressed_tiles_count` and it is not used here.
    /// It is a number the writer put in the file, so an archive whose header
    /// miscounts its own tiles would agree with itself perfectly; the walk
    /// counts what the directories actually cover, and the disagreement
    /// between the two is itself one of the findings above.
    fn addressed_tiles(&self) -> Result<u64, PyramidReadError> {
        self.structural_report()
            .map(|report| report.addressed_tiles)
    }

    fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
        let header = self.reader.header();
        let generation = self.generation();
        Ok(PyramidDescription {
            min_level: u32::from(header.min_zoom),
            max_level: u32::from(header.max_zoom),
            tile_size: generation.as_ref().map(|g| g.tile_size),
            layout: generation.as_ref().map(|g| g.layout),
            format: self.tile_format(),
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

    /// The encoding the stored bytes are in.
    ///
    /// From the `vnd.libviprs` namespace when the archive carries it, because
    /// that is the only place the JPEG quality a [`TileFormat::Jpeg`] carries
    /// is written down. A foreign JPEG archive therefore reports `None` rather
    /// than a quality nobody measured; ask
    /// [`Reader::tile_format`](crate::pmtiles::Reader::tile_format) for the
    /// `TileType`, which is what the archive actually records.
    fn tile_format(&self) -> Option<TileFormat> {
        if let Some(generation) = self.generation() {
            return Some(generation.format);
        }
        match self.reader.tile_format() {
            crate::pmtiles::TileType::Png => Some(TileFormat::Png),
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
