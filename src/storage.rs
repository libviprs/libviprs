//! Where a pyramid lands, and what that choice resolves to.
//!
//! [`PyramidStorage`] is storage *selection*: which of the two artifacts a
//! pyramid becomes, the output path that choice resolves a base to, and the
//! layouts it can hold. It is one enum and three `const fn`s, and it touches
//! no filesystem.
//!
//! It lives here rather than in [`crate::sink`] because that module is sink
//! *mechanism*, thousands of lines of it, and every source-scanning guard in
//! this repository reads a file up to its first `#[cfg(test)] mod`. A policy
//! type parked near the top of `src/sink.rs` with its tests behind it put
//! `FsSink` and twenty other items past that cut, where no guard could see
//! them. The types re-export from the crate root unchanged, so nothing a
//! caller writes moves.

use std::path::{Path, PathBuf};

use crate::planner::Layout;

/// The extension a PMTiles archive carries, without the dot.
///
/// One spelling, so nothing has to spell it a second time.
pub const PMTILES_EXTENSION: &str = "pmtiles";

/// Where a pyramid lands when nothing says otherwise.
///
/// A pyramid is an immutable, indexed artifact, and until 0.5.0 this crate
/// only ever materialised one as a tree of loose files under `{z}/{x}/{y}`
/// ([`FsSink`]). That is fine for one pyramid on a laptop and ruinous at fleet
/// scale: 100k pyramids of 20k tiles each is around 2 billion files, which is
/// inode pressure, a backup that never finishes, and an object-store bill made
/// mostly of request counts. [`PyramidStorage::PmTiles`] puts the whole
/// pyramid, its index and its metadata in one PMTiles v3 archive that still
/// answers a single-tile question in a couple of ranged reads, and it is the
/// default.
///
/// This type is the one place that choice is made, so nothing downstream has
/// to guess at it. It picks the storage and the output path and stops there:
/// the sink is still constructed by name and handed to
/// [`EngineBuilder`](crate::engine_builder::EngineBuilder), so no existing
/// caller changes shape. A `PmTilesSink` aimed at [`output_path`] writes the
/// archive; an [`FsSink`] aimed at the same base writes the tree.
///
/// The match in each method below has no wildcard arm, so a third storage
/// fails to compile here rather than quietly inheriting the archive's answers.
///
/// [`FsSink`]: crate::sink::FsSink
/// [`output_path`]: PyramidStorage::output_path
///
/// # Examples
///
/// <!-- storage-example -->
/// ```
/// use libviprs::{Layout, PyramidStorage};
/// use std::path::Path;
///
/// // Nothing said otherwise, so the pyramid lands in one indexed archive.
/// let storage = PyramidStorage::default();
/// assert_eq!(storage, PyramidStorage::PmTiles);
/// assert_eq!(storage.output_path("city"), Path::new("city.pmtiles"));
///
/// // An archive addresses a tile by (z, x, y), so those are the layouts it takes.
/// assert!(storage.accepts_layout(Layout::Xyz));
/// assert!(storage.accepts_layout(Layout::Google));
/// assert!(!storage.accepts_layout(Layout::DeepZoom));
///
/// // The tree of loose files is still one value away, and it takes all five.
/// let storage = PyramidStorage::Directory;
/// assert_eq!(storage.output_path("city"), Path::new("city"));
/// assert_eq!(storage.extension(), None);
/// assert!(storage.accepts_layout(Layout::DeepZoom));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PyramidStorage {
    /// One PMTiles v3 archive holding every tile, the directories that index
    /// them and the metadata describing the pyramid. The default since 0.5.0.
    #[default]
    PmTiles,
    /// The tree of loose files under `{z}/{x}/{y}` that [`FsSink`] writes,
    /// and the only storage this crate had before 0.5.0.
    ///
    /// [`FsSink`]: crate::sink::FsSink
    Directory,
}

impl PyramidStorage {
    /// The extension this storage's output carries, or `None` for a directory.
    ///
    /// A directory has no extension of its own. DeepZoom's `.dzi` manifest is
    /// a sibling file rather than the output itself, which is why this answers
    /// `None` there rather than `"dzi"`.
    #[must_use]
    pub const fn extension(self) -> Option<&'static str> {
        match self {
            Self::PmTiles => Some(PMTILES_EXTENSION),
            Self::Directory => None,
        }
    }

    /// Whether this storage can hold a pyramid planned with `layout`.
    ///
    /// A directory takes all five. The layout decides the shape of the paths
    /// inside the tree and nothing else.
    ///
    /// An archive takes the two whose level index is a zoom and whose tile is
    /// a `(z, x, y)` triple, [`Layout::Xyz`] and [`Layout::Google`], because
    /// PMTiles v3 addresses a tile by a single `u64` derived from `(z, x, y)`
    /// on a Hilbert curve. Google differs from XYZ in the order it spells a
    /// path on disk (`z/y/x` against `z/x/y`) and not in what it addresses, so
    /// there is no coordinate migration between either of them and an archive.
    ///
    /// [`Layout::DeepZoom`], [`Layout::Zoomify`] and [`Layout::Iiif`] do not
    /// fit. Their level index is a tier rather than a zoom and their tile is
    /// not a `(z, x, y)` triple, so an archive built from one would be
    /// addressable and would render nonsense in anything that opened it.
    ///
    /// This answers the question before the work starts. The sink is what
    /// enforces it, and it refuses when it is built rather than at the first
    /// tile.
    ///
    /// The match on [`Layout`] has no wildcard arm, so a sixth layout fails to
    /// compile here rather than quietly landing on one side.
    #[must_use]
    pub const fn accepts_layout(self, layout: Layout) -> bool {
        match self {
            Self::Directory => true,
            Self::PmTiles => match layout {
                Layout::Xyz | Layout::Google => true,
                Layout::DeepZoom | Layout::Zoomify | Layout::Iiif => false,
            },
        }
    }

    /// Where the output actually lands, given the base a caller asked for.
    ///
    /// A directory is the base, unchanged. An archive is the base with
    /// `.pmtiles` **appended**, unless it already ends in `.pmtiles`
    /// (compared without case, because on macOS and Windows `city.PMTILES`
    /// and `city.PMTILES.pmtiles` name the same file and appending there
    /// would write the archive over the base it came from).
    ///
    /// Appended rather than substituted, deliberately.
    /// [`PathBuf::set_extension`] replaces everything after the last dot, so it
    /// turns `tiles.v2` into `tiles.pmtiles` and loses the `v2`. Appending
    /// gives `tiles.v2.pmtiles`, which is uglier and never surprising. If you
    /// want `city.tif` to become `city.pmtiles`, hand this the stem rather
    /// than the whole name.
    ///
    /// The base is not read, created or checked here. This is path
    /// arithmetic, and the sink is what touches the filesystem.
    #[must_use]
    pub fn output_path(self, base: impl AsRef<Path>) -> PathBuf {
        let base = base.as_ref();
        match self {
            Self::Directory => base.to_path_buf(),
            Self::PmTiles => {
                if base
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case(PMTILES_EXTENSION))
                {
                    return base.to_path_buf();
                }
                let mut name = base.as_os_str().to_os_string();
                name.push(".");
                name.push(PMTILES_EXTENSION);
                PathBuf::from(name)
            }
        }
    }
}

#[cfg(test)]
mod storage_selection_tests {
    use super::*;
    use crate::planner::Layout;

    /// With nothing said, a pyramid lands in one PMTiles archive.
    ///
    /// The tag is the least of it. What a caller actually sees is the
    /// extension the choice carries and the path it resolves an output base
    /// to, so those are asserted here too: a test comparing
    /// `PyramidStorage::default()` against `PyramidStorage::PmTiles` and
    /// nothing else would still pass on a build where both arms resolved to
    /// the same loose-file tree.
    #[test]
    fn the_default_storage_is_one_pmtiles_archive() {
        let storage = PyramidStorage::default();
        assert_eq!(storage, PyramidStorage::PmTiles);
        assert_eq!(storage.extension(), Some(PMTILES_EXTENSION));
        assert_eq!(storage.output_path("city"), Path::new("city.pmtiles"));
        assert!(storage.accepts_layout(Layout::Xyz));
    }

    /// The loose-file tree is one value away, and it answers differently.
    ///
    /// This is the negative control for the test above. Two arms collapsing
    /// onto one answer is the shape that lets a default assertion pass for the
    /// wrong reason, so the two are compared against each other here rather
    /// than each against its own literal.
    #[test]
    fn the_directory_tree_is_reachable_and_is_not_the_default() {
        let tree = PyramidStorage::Directory;
        assert_ne!(tree, PyramidStorage::default());
        assert_eq!(tree.extension(), None);
        assert_eq!(tree.output_path("city"), Path::new("city"));
        assert!(tree.accepts_layout(Layout::DeepZoom));

        assert_ne!(
            tree.output_path("city"),
            PyramidStorage::default().output_path("city"),
            "the two storages must resolve one output base to two different \
             paths, or the default assertion holds on a build where the choice \
             does not reach the output at all"
        );
        assert_ne!(
            tree.accepts_layout(Layout::DeepZoom),
            PyramidStorage::default().accepts_layout(Layout::DeepZoom),
            "the two storages must answer DeepZoom differently, or an \
             `accepts_layout` that said yes to everything would pass here"
        );
    }

    /// The archive takes the two layouts addressed by `(z, x, y)`; the tree
    /// takes all five.
    #[test]
    fn the_archive_takes_the_two_zxy_layouts_and_the_tree_takes_every_one() {
        // `accepts_layout` matches on `Layout` with no wildcard arm, so a
        // sixth layout fails to compile there rather than quietly landing on
        // one side of this table without anyone deciding.
        let table = [
            (Layout::DeepZoom, false),
            (Layout::Xyz, true),
            (Layout::Google, true),
            (Layout::Zoomify, false),
            (Layout::Iiif, false),
        ];
        for (layout, archivable) in table {
            assert_eq!(
                PyramidStorage::PmTiles.accepts_layout(layout),
                archivable,
                "the archive's answer for {layout:?} moved"
            );
            assert!(
                PyramidStorage::Directory.accepts_layout(layout),
                "the tree takes every layout, and it did not take {layout:?}"
            );
        }
        // The control. A table with no refusal in it would pass against an
        // `accepts_layout` that answered `true` for everything, which is the
        // shape this test exists to catch.
        assert!(
            table.iter().any(|(_, archivable)| !archivable),
            "a table with nothing refused in it cannot fail"
        );
    }

    /// The extension is appended, never substituted.
    #[test]
    fn an_output_base_keeps_the_extension_it_already_had() {
        let pm = PyramidStorage::PmTiles;
        // `PathBuf::set_extension` replaces everything after the last dot, so it
        // would turn `tiles.v2` into `tiles.pmtiles` and lose the `v2`.
        assert_eq!(pm.output_path("tiles.v2"), Path::new("tiles.v2.pmtiles"));
        assert_eq!(pm.output_path("city.tif"), Path::new("city.tif.pmtiles"));
        assert_eq!(
            pm.output_path("/var/tiles/city"),
            Path::new("/var/tiles/city.pmtiles")
        );
    }

    /// A base that already names an archive is handed back untouched.
    #[test]
    fn a_base_that_already_names_an_archive_is_left_alone() {
        let pm = PyramidStorage::PmTiles;
        assert_eq!(pm.output_path("city.pmtiles"), Path::new("city.pmtiles"));
        // Case-insensitively. On macOS and Windows `city.PMTILES` and
        // `city.PMTILES.pmtiles` are two names for one file, so appending
        // there would write the archive over the base it was derived from.
        assert_eq!(pm.output_path("city.PMTILES"), Path::new("city.PMTILES"));
    }

    /// An empty base is pinned rather than left to surprise someone.
    #[test]
    fn an_empty_output_base_takes_the_extension_like_any_other() {
        assert_eq!(
            PyramidStorage::PmTiles.output_path(""),
            Path::new(".pmtiles")
        );
        assert_eq!(PyramidStorage::Directory.output_path(""), Path::new(""));
    }
}
