//! Structural validation of a PMTiles v3 archive (issue #991).
//!
//! [`validate`] walks an archive the way a reader would and reports what is
//! wrong with it, rather than answering yes or no. That shape is chosen for
//! the thing it becomes: `viprs pmtiles verify`, where an operator holding a
//! broken 40 GB file wants the list, not the first symptom. So it collects
//! [`Finding`]s and keeps going, and the only thing that stops it is the
//! underlying [`RangeReader`] failing to hand over bytes.
//!
//! # It checks `offset + length`, and the reference does not
//!
//! This is the whole reason the module is not two `if`s. `pmtiles verify` in
//! go-pmtiles v1.31.2 bounds-checks four *lengths* against the file size,
//! `header.RootLength > fileSize` and the same shape for metadata, leaf
//! directories and tile data. It never checks an offset, and it never checks a
//! sum. Measured on the oracle: setting `root_offset` to 999999 in an
//! 1878-byte archive passes every one of those checks, because the root length
//! is still 35 and the whole-archive arithmetic `127 + 35 + 162 + 0 + 1554`
//! still equals 1878. `verify` then walks to offset 999999, hands what it
//! finds to `DeserializeEntries`, which does `reader, _ = gzip.NewReader(data)`
//! and throws the error away, and the next read dereferences a nil pointer.
//! Exit code 2, out of `compress/gzip`.
//!
//! Two rules fall out of that and every check here follows them:
//!
//! * every section bound is `offset.checked_add(length) <= archive_size`, so
//!   an offset past the end is caught whatever its length says, and the
//!   addition itself cannot wrap;
//! * a directory that does not decompress or does not parse is a typed
//!   finding. Passing `pmtiles verify` is not evidence that an archive is safe
//!   to parse, and matching the reference here would be matching a crash.
//!
//! # The leaf offset base
//!
//! A directory entry with `run_length == 0` is a leaf pointer, and its
//! `offset` is relative to `header.leaf_directories_offset`. Not to the start
//! of the file, not to the start of the directory it was found in. This is the
//! one place in the format where a writer and a reader that make the same
//! wrong choice round-trip perfectly and are both wrong, and there is exactly
//! one fixture in the world that exercises it (`leaves-z0z7.pmtiles`, whose
//! six pointers carry 0, 71, 136, 202, 268 and 333 against a
//! `leaf_directories_offset` of 334).
//!
//! # Everything is bounded because the input is hostile
//!
//! [`ValidationLimits`] caps the decompressed size of any one directory, how
//! many leaf directories will be followed, how deep the leaf tree may go, and
//! how many findings accumulate. The distinct-offset count that checks
//! `tile_contents_count` is capped too, and reports [`Report::tile_contents`]
//! as `None` rather than a wrong number when it gives up: an archive with
//! fifty million entries should not cost a set of fifty million `u64` to
//! verify.
//!
//! # Capping each step is not capping the walk
//!
//! Those are all per-step ceilings and their product is not bounded by any of
//! them, which is a hole you can drive a 21 KB file through. A root holding
//! 1,048,577 leaf pointers that every one resolve to the same fat, valid leaf
//! sits inside every one of the caps above: the root is one directory, each
//! leaf read is one directory under `max_directory_bytes`, the pointer count
//! is at the `max_leaf_directories` default of 2^20, the depth is 1, and every
//! entry in every directory is genuinely in bounds so not one finding fires.
//! Multiplied out it is roughly **16 TiB of gzip output and 4.4e12 entry-walk
//! steps out of 21,620 bytes on disk**, with memory flat because each leaf is
//! freed before the next. `viprs pmtiles verify` on defaults simply never
//! returns, which is the same failure the reference implementation is mocked
//! for two sections up, moved from a segfault to a hang.
//!
//! Two things close it, and they are independent:
//!
//! * **a budget across the whole walk, not per directory.** A running total of
//!   decompressed directory bytes and of entries visited, checked before each
//!   directory is read, raising [`Finding::TotalDirectoryBytesExceeded`] or
//!   [`Finding::TotalEntriesExceeded`] and unwinding. The byte budget is sized
//!   from the archive's **own** `root_length + leaf_directories_length`, so it
//!   scales with the file rather than being a constant a large honest archive
//!   could reach;
//! * **leaf offsets are deduplicated.** A leaf is read once however many
//!   pointers reach it, counted once in [`Report::leaves`], counted in
//!   [`Report::shared_leaf_pointers`] and reported as
//!   [`Finding::LeafDirectoryRevisited`].
//!
//! Either one alone collapses the archive above. Both are here because they
//! fail differently: dedupe does nothing against a leaf region holding a
//! million *distinct* fat leaves, and the budget does nothing about the report
//! counting one leaf a million times.
//!
//! [`Report::directory_bytes`] and [`Report::entries_visited`] carry what the
//! walk actually spent, so "this is bounded" is a number a caller can read
//! rather than a sentence in these docs.
//!
//! # Examples
//!
//! ```no_run
//! use libviprs::pmtiles::validate::{ValidationLimits, validate};
//! use libviprs::pmtiles::FileRangeReader;
//!
//! let reader = FileRangeReader::try_open("drawing.pmtiles")?;
//! let report = validate(&reader, &ValidationLimits::default())?;
//! if report.is_valid() {
//!     println!("{} entries, {} tiles", report.tile_entries, report.addressed_tiles);
//! } else {
//!     for finding in &report.findings {
//!         println!("{finding}");
//!     }
//! }
//! # Ok::<(), libviprs::pmtiles::PmTilesError>(())
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::io;

use crate::pmtiles::directory::deserialize_entries;
use crate::pmtiles::{Entry, Header, PmTilesError, RangeReader, header::HEADER_BYTES};

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// The ceilings [`validate`] works under.
///
/// Every one of them exists because the number it bounds comes out of the file
/// being checked. There are no defaults chosen for taste: each is large enough
/// for an archive a conformant writer produces and small enough that a hostile
/// one cannot turn a verify into an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ValidationLimits {
    /// The most bytes one directory may occupy, compressed or decompressed.
    ///
    /// No length field in PMTiles v3 is an uncompressed length, so a reader
    /// cannot pre-size the output buffer and has to cap it instead. The spec
    /// recommends a root directory under 16 KiB; this is generous by three
    /// orders of magnitude so that an unusual but honest archive still
    /// verifies.
    pub max_directory_bytes: usize,

    /// The most leaf directories that will be followed.
    ///
    /// A 40 GB pyramid at zoom 16 needs a few thousand. The ceiling is what
    /// stops a directory claiming millions of leaves from turning one verify
    /// into millions of ranged reads.
    pub max_leaf_directories: usize,

    /// How deep the leaf tree may go. The v3 format allows nesting and every
    /// writer in practice emits one level, so anything past this is a fixture
    /// nobody produced or an archive trying to make a verify recurse forever.
    pub max_leaf_depth: usize,

    /// How many findings to collect before giving up on the walk. A file that
    /// is not an archive at all can produce one finding per entry, and an
    /// operator reading the first fifty already knows.
    pub max_findings: usize,

    /// How many distinct entry offsets to track for the `tile_contents_count`
    /// check before abandoning it. Past this, [`Report::tile_contents`] is
    /// `None` and the count is not compared.
    pub max_tracked_contents: usize,

    /// The most decompressed directory bytes the **whole walk** will inflate,
    /// as an absolute ceiling.
    ///
    /// [`max_directory_bytes`](Self::max_directory_bytes) caps one directory
    /// and says nothing about how many of them there are, which is the hole a
    /// 21 KB archive walked through: a root holding 2^20 + 1 leaf pointers
    /// that all resolve to one fat, valid leaf inflated that leaf once per
    /// pointer, 16 TiB of gzip output with every entry in bounds and not one
    /// finding raised. Memory stayed flat because each leaf is freed before
    /// the next, so it was a hang rather than a kill.
    ///
    /// The budget an individual walk actually works under is the smaller of
    /// this and a figure derived from the archive's own section lengths, so an
    /// honest archive can never reach it: see [`validate`].
    pub max_total_directory_bytes: u64,

    /// The most directory entries the **whole walk** will visit.
    ///
    /// The same amplification measured in operations rather than bytes:
    /// 4.4e12 entry-walk steps out of 21,620 bytes. Every directory body
    /// spends at least four bytes per entry (four parallel varint columns, one
    /// byte minimum each), so
    /// [`max_total_directory_bytes`](Self::max_total_directory_bytes) already
    /// bounds this at a quarter of its value and the two defaults are set to
    /// bind at the same point. It is tracked separately because an operator
    /// reading a report wants to know which ceiling stopped the walk.
    pub max_total_entries: u64,

    /// Whether to keep the decoded entries of every leaf directory in the
    /// report. Off by default: a real archive has millions and a verify does
    /// not need them, but a test comparing against a reference decode does.
    pub collect_entries: bool,
}

/// How far past its own compressed size a walk will let the directory region
/// inflate before it calls the archive hostile.
///
/// The honest bound on a walk's total decompressed directory bytes is the
/// compressed size of the regions those directories live in, `root_length +
/// leaf_directories_length`, times whatever ratio gzip achieved. Directory
/// bodies are columns of small varints and compress at roughly 2:1 to 5:1 in
/// everything go-pmtiles writes, so 64 leaves two orders of magnitude of
/// headroom over a conformant archive while still turning the 794,000:1 the
/// leaf storm asked for into a refusal.
const DIRECTORY_INFLATION_ALLOWANCE: u64 = 64;

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_directory_bytes: 16 * 1024 * 1024,
            max_leaf_directories: 1 << 20,
            max_leaf_depth: 4,
            max_findings: 256,
            max_tracked_contents: 4 * 1024 * 1024,
            max_total_directory_bytes: 1 << 30,
            max_total_entries: 1 << 28,
            collect_entries: false,
        }
    }
}

impl ValidationLimits {
    /// Keep the decoded entries of each leaf directory in the report.
    pub fn with_collect_entries(mut self, collect: bool) -> Self {
        self.collect_entries = collect;
        self
    }

    /// Set the per-directory byte ceiling.
    pub fn with_max_directory_bytes(mut self, bytes: usize) -> Self {
        self.max_directory_bytes = bytes;
        self
    }

    /// Set how many leaf directories will be followed.
    pub fn with_max_leaf_directories(mut self, leaves: usize) -> Self {
        self.max_leaf_directories = leaves;
        self
    }

    /// Set the ceiling on the decompressed directory bytes of the whole walk.
    pub fn with_max_total_directory_bytes(mut self, bytes: u64) -> Self {
        self.max_total_directory_bytes = bytes;
        self
    }

    /// Set the ceiling on the directory entries the whole walk visits.
    pub fn with_max_total_entries(mut self, entries: u64) -> Self {
        self.max_total_entries = entries;
        self
    }
}

// ---------------------------------------------------------------------------
// What a finding points at
// ---------------------------------------------------------------------------

/// One of the four regions the header addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Section {
    /// The root directory.
    Root,
    /// The JSON metadata.
    Metadata,
    /// The leaf directory region, which is one region holding all the leaves.
    LeafDirectories,
    /// The tile blobs.
    TileData,
}

impl fmt::Display for Section {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => f.write_str("the root directory"),
            Self::Metadata => f.write_str("the metadata"),
            Self::LeafDirectories => f.write_str("the leaf directories"),
            Self::TileData => f.write_str("the tile data"),
        }
    }
}

/// Which directory a finding was raised in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DirectoryRef {
    /// The root directory.
    Root,
    /// The nth leaf directory, counted in the order the walk reached them.
    Leaf(usize),
}

impl fmt::Display for DirectoryRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => f.write_str("the root directory"),
            Self::Leaf(index) => write!(f, "leaf directory {index}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

/// One structural problem with an archive.
///
/// Every variant names the numbers it rejected. "Invalid directory" without
/// the value that was invalid is not something an operator can act on, and
/// this is the type a CLI prints straight to a terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Finding {
    /// The file is smaller than the fixed-size header.
    ArchiveTooShort { size: u64, need: u64 },

    /// The backend could not say how large the object is, so no section bound
    /// could be checked against it. Not a defect in the archive; a limit on
    /// what this run was able to check, recorded so a clean report cannot be
    /// mistaken for a complete one.
    ArchiveSizeUnknown,

    /// The first seven bytes were not `PMTiles`.
    BadMagic { found: [u8; 7] },

    /// The magic matched and the version byte was not 3. PMTiles v2 shares the
    /// magic with a completely different layout.
    UnsupportedVersion { found: u8 },

    /// A section runs past the end of the archive. The check is on
    /// `offset + length`, which is the one the reference implementation does
    /// not make.
    SectionOutOfBounds {
        section: Section,
        offset: u64,
        length: u64,
        archive_size: u64,
    },

    /// A section's `offset + length` does not fit in a `u64`.
    SectionOverflow {
        section: Section,
        offset: u64,
        length: u64,
    },

    /// A directory's compressed bytes are larger than the ceiling this run
    /// works under, so it was not read.
    DirectoryTooLarge {
        directory: DirectoryRef,
        length: u64,
        limit: usize,
    },

    /// A directory could not be decompressed or could not be parsed. Carries
    /// the underlying error's text, because "the root directory is broken"
    /// without saying how sends an operator to the wrong tool.
    DirectoryUnreadable {
        directory: DirectoryRef,
        reason: String,
    },

    /// Two entries claim the same tile. The parser refuses a zero id delta, so
    /// entries always ascend; what it cannot see is a run that reaches past
    /// the next entry's id, and then a lookup for a tile in the overlap has
    /// two answers.
    OverlappingRun {
        directory: DirectoryRef,
        index: usize,
        tile_id: u64,
        previous_run_ends_at: u64,
    },

    /// An entry's bytes run past the end of the tile data section. Entry
    /// offsets are relative to `header.tile_data_offset`, so the bound is
    /// `offset + length <= tile_data_length`.
    EntryOutOfTileData {
        directory: DirectoryRef,
        index: usize,
        offset: u64,
        length: u32,
    },

    /// A leaf pointer whose target is outside the leaf directory region, or
    /// outside the archive.
    LeafPointerOutOfBounds {
        directory: DirectoryRef,
        index: usize,
        offset: u64,
        length: u32,
    },

    /// A leaf pointer in an archive whose header says there is no leaf
    /// region. The length is the flag, never the offset: a no-leaf archive
    /// still carries a non-zero `leaf_directories_offset`, equal to
    /// `tile_data_offset`, in everything go-pmtiles writes.
    LeafPointerWithoutLeafSection {
        directory: DirectoryRef,
        index: usize,
    },

    /// More leaf directories than this run will follow.
    TooManyLeafDirectories { found: usize, limit: usize },

    /// The leaf tree is deeper than this run will follow.
    LeafDepthExceeded {
        directory: DirectoryRef,
        index: usize,
        limit: usize,
    },

    /// The entries counted across every directory do not add up to the
    /// header's `tile_entries_count`.
    EntryCountMismatch { counted: u64, header: u64 },

    /// The run lengths do not sum to the header's `addressed_tiles_count`.
    /// This is the invariant that catches a reader or writer mishandling
    /// deduplication: `dupes-z0z3.pmtiles` has 67 entries summing to 85.
    AddressedTilesMismatch { counted: u64, header: u64 },

    /// The distinct entry offsets do not match the header's
    /// `tile_contents_count`.
    ContentsCountMismatch { counted: u64, header: u64 },

    /// A leaf pointer resolved to a leaf this walk had already read.
    ///
    /// Nothing in the spec forbids it and no writer produces it, so it is
    /// reported rather than followed: the leaf's contents cannot depend on
    /// which pointer reached it, and following it again is how one valid 21 KB
    /// archive asked for 16 TiB of gzip output.
    LeafDirectoryRevisited {
        directory: DirectoryRef,
        index: usize,
        absolute_offset: u64,
    },

    /// The walk stopped because it had inflated as many directory bytes as the
    /// whole archive is worth. Carries the budget it was working under, which
    /// is derived from the archive's own section lengths rather than fixed.
    TotalDirectoryBytesExceeded {
        directory: DirectoryRef,
        inflated: u64,
        budget: u64,
    },

    /// The walk stopped because it had visited as many directory entries as it
    /// is willing to.
    TotalEntriesExceeded {
        directory: DirectoryRef,
        visited: u64,
        budget: u64,
    },

    /// The walk stopped because it had collected as many findings as it is
    /// willing to.
    FindingLimitReached { limit: usize },
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArchiveTooShort { size, need } => write!(
                f,
                "the file is {size} bytes, which cannot hold the {need}-byte v3 header"
            ),
            Self::ArchiveSizeUnknown => f.write_str(
                "the backend did not report an object size, so no section bound was checked",
            ),
            Self::BadMagic { found } => {
                write!(f, "not a PMTiles archive: magic is {found:?}")
            }
            Self::UnsupportedVersion { found } => write!(
                f,
                "PMTiles version {found}, and this reader implements version 3"
            ),
            Self::SectionOutOfBounds {
                section,
                offset,
                length,
                archive_size,
            } => write!(
                f,
                "{section} runs from {offset} for {length} bytes, past the end of a \
                 {archive_size}-byte archive"
            ),
            Self::SectionOverflow {
                section,
                offset,
                length,
            } => write!(
                f,
                "{section} claims offset {offset} plus length {length}, which does not fit in a u64"
            ),
            Self::DirectoryTooLarge {
                directory,
                length,
                limit,
            } => write!(
                f,
                "{directory} is {length} bytes, past the {limit} byte ceiling this check works under"
            ),
            Self::DirectoryUnreadable { directory, reason } => {
                write!(f, "{directory} could not be read: {reason}")
            }
            Self::OverlappingRun {
                directory,
                index,
                tile_id,
                previous_run_ends_at,
            } => write!(
                f,
                "{directory} entry {index} claims tile {tile_id}, which the previous run \
                 already covers (it ends at {previous_run_ends_at})"
            ),
            Self::EntryOutOfTileData {
                directory,
                index,
                offset,
                length,
            } => write!(
                f,
                "{directory} entry {index} points at {offset} for {length} bytes, past the end \
                 of the tile data"
            ),
            Self::LeafPointerOutOfBounds {
                directory,
                index,
                offset,
                length,
            } => write!(
                f,
                "{directory} entry {index} is a leaf pointer at {offset} for {length} bytes, \
                 outside the leaf directory region"
            ),
            Self::LeafPointerWithoutLeafSection { directory, index } => write!(
                f,
                "{directory} entry {index} is a leaf pointer, and the header says the leaf \
                 region is empty"
            ),
            Self::TooManyLeafDirectories { found, limit } => write!(
                f,
                "the archive has at least {found} leaf directories, past the {limit} this check \
                 will follow"
            ),
            Self::LeafDepthExceeded {
                directory,
                index,
                limit,
            } => write!(
                f,
                "{directory} entry {index} is a leaf pointer {limit} levels deep, which is as \
                 far as this check follows"
            ),
            Self::EntryCountMismatch { counted, header } => write!(
                f,
                "the directories hold {counted} entries and the header says {header}"
            ),
            Self::AddressedTilesMismatch { counted, header } => write!(
                f,
                "the run lengths sum to {counted} and the header's addressed_tiles_count is \
                 {header}"
            ),
            Self::ContentsCountMismatch { counted, header } => write!(
                f,
                "the entries point at {counted} distinct offsets and the header's \
                 tile_contents_count is {header}"
            ),
            Self::LeafDirectoryRevisited {
                directory,
                index,
                absolute_offset,
            } => write!(
                f,
                "{directory} entry {index} is a leaf pointer at file offset {absolute_offset}, \
                 which another pointer already reached; it was read once"
            ),
            Self::TotalDirectoryBytesExceeded {
                directory,
                inflated,
                budget,
            } => write!(
                f,
                "the walk stopped at {directory} after decompressing {inflated} directory \
                 bytes, past the {budget} this archive's own section lengths allow"
            ),
            Self::TotalEntriesExceeded {
                directory,
                visited,
                budget,
            } => write!(
                f,
                "the walk stopped at {directory} after visiting {visited} directory entries, \
                 past the {budget} this check will walk"
            ),
            Self::FindingLimitReached { limit } => {
                write!(f, "stopped after {limit} findings; there may be more")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// One leaf directory, as the walk found it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafReport {
    /// The offset the pointer carried, **relative to
    /// `header.leaf_directories_offset`**.
    pub offset_in_section: u64,
    /// Where that resolves to in the file.
    pub absolute_offset: u64,
    /// The pointer's length, which is the leaf's compressed size.
    pub compressed_length: u64,
    /// How many entries the leaf decoded to.
    pub entries: usize,
    /// Those entries, when [`ValidationLimits::collect_entries`] asked for
    /// them.
    pub decoded: Option<Vec<Entry>>,
}

/// What [`validate`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Report {
    /// The header, when one decoded. `None` means the file is not a v3
    /// archive and nothing below it was checked.
    pub header: Option<Header>,
    /// Everything wrong with the archive, in the order the walk met it.
    pub findings: Vec<Finding>,
    /// How many entries the root directory held, leaf pointers included.
    pub root_entries: usize,
    /// Tile entries across every directory, leaf pointers **not** included.
    /// This is what `header.tile_entries_count` counts.
    pub tile_entries: u64,
    /// The run lengths summed, which is what `addressed_tiles_count` counts.
    pub addressed_tiles: u64,
    /// Distinct entry offsets, which is what `tile_contents_count` counts, or
    /// `None` when there were more than the run was willing to track.
    pub tile_contents: Option<u64>,
    /// Every **distinct** leaf directory the walk followed. Two pointers at
    /// one leaf produce one entry here and one count in
    /// [`shared_leaf_pointers`](Self::shared_leaf_pointers).
    pub leaves: Vec<LeafReport>,
    /// Decompressed directory bytes the walk inflated, root and leaves
    /// together. This is the number the total-work budget is spent against,
    /// and it is in the report so a bounded-work claim is something a caller
    /// can check rather than something the docs assert.
    pub directory_bytes: u64,
    /// Directory entries the walk visited, leaf pointers included.
    pub entries_visited: u64,
    /// Leaf pointers that resolved to a leaf the walk had already read.
    ///
    /// Nothing in the spec forbids two pointers sharing one leaf and no writer
    /// produces it, so a non-zero value here is either a clever writer or a
    /// file built to make a verify do the same work a million times. Either
    /// way the leaf is read once, and the header's three counts are not
    /// compared afterwards because the walk's totals no longer count what the
    /// header counts.
    pub shared_leaf_pointers: u64,
}

impl Report {
    /// Whether the archive is structurally sound: no findings at all.
    ///
    /// This rests on an invariant the walk keeps: **anything that stops it
    /// early also raises a finding**. Without that, a file that made the walk
    /// give up quietly would report clean, which is worse than reporting the
    /// defect, because the header's three counts are only compared when the
    /// walk reached the end.
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    fn empty() -> Self {
        Self {
            header: None,
            findings: Vec::new(),
            root_entries: 0,
            tile_entries: 0,
            addressed_tiles: 0,
            tile_contents: Some(0),
            leaves: Vec::new(),
            directory_bytes: 0,
            entries_visited: 0,
            shared_leaf_pointers: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// A reader over a slice, for the in-memory entry point
// ---------------------------------------------------------------------------

/// The [`RangeReader`] behind [`validate_bytes`].
///
/// Private on purpose. The indexed reader will want one of these too and this
/// module is not where that decision belongs; what it needs is a way to check
/// an archive it already holds, which a fuzz target and a test both do.
struct SliceReader<'a>(&'a [u8]);

impl RangeReader for SliceReader<'_> {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "offset past this slice"))?;
        let end = start
            .checked_add(len)
            .filter(|end| *end <= self.0.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "range past this slice"))?;
        Ok(self.0[start..end].to_vec())
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.0.len() as u64))
    }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Validate an archive already in memory.
pub fn validate_bytes(bytes: &[u8], limits: &ValidationLimits) -> Result<Report, PmTilesError> {
    validate(&SliceReader(bytes), limits)
}

/// Validate an archive reachable through a [`RangeReader`].
///
/// Returns `Err` only when the reader itself fails. Everything structural is a
/// [`Finding`] in the report, because a validator that stops at the first
/// problem describes one symptom of a file that may have six.
pub fn validate<R: RangeReader + ?Sized>(
    reader: &R,
    limits: &ValidationLimits,
) -> Result<Report, PmTilesError> {
    let mut report = Report::empty();
    let mut walk = Walk {
        limits,
        offsets: BTreeSet::new(),
        tracking_offsets: true,
        visited_leaves: BTreeSet::new(),
        bytes_spent: 0,
        // Filled in below, once the header says how large the directory
        // regions are. Zero until then, and nothing is read until then.
        byte_budget: 0,
        entries_spent: 0,
        entry_budget: limits.max_total_entries,
        exhausted: false,
    };

    let size = reader.size()?;
    let header = match read_header(reader, size, &mut report)? {
        Some(header) => header,
        None => return Ok(report),
    };
    report.header = Some(header);

    if size.is_none() {
        push(&mut report, Finding::ArchiveSizeUnknown, limits);
    }

    // Every section bound, checked as `offset + length` rather than as a bare
    // length. `root_in_bounds` decides whether the walk below happens at all:
    // reading a range this check just rejected is the step the reference
    // takes and segfaults on.
    let sections = [
        (Section::Root, header.root_offset, header.root_length),
        (
            Section::Metadata,
            header.metadata_offset,
            header.metadata_length,
        ),
        (
            Section::LeafDirectories,
            header.leaf_directories_offset,
            header.leaf_directories_length,
        ),
        (
            Section::TileData,
            header.tile_data_offset,
            header.tile_data_length,
        ),
    ];
    let mut root_in_bounds = true;
    for (section, offset, length) in sections {
        if !check_section(&mut report, limits, section, offset, length, size)
            && section == Section::Root
        {
            root_in_bounds = false;
        }
    }

    if !root_in_bounds {
        return Ok(report);
    }

    // The total-work budget, sized against the archive rather than picked.
    //
    // Every directory this walk reads lives in the root region or the leaf
    // region, and leaf offsets are deduplicated below, so the compressed bytes
    // it can legitimately read are bounded by those two lengths. Multiply by
    // the inflation allowance and an honest archive is two orders of magnitude
    // inside the budget, while an archive claiming to inflate a thousandfold
    // is stopped. The floor is one directory's worth, because a tiny archive
    // has a tiny span and still has to be allowed to read its own root.
    walk.byte_budget = header
        .root_length
        .saturating_add(header.leaf_directories_length)
        .saturating_mul(DIRECTORY_INFLATION_ALLOWANCE)
        .max(limits.max_directory_bytes as u64)
        .min(limits.max_total_directory_bytes);

    let Some(entries) = read_directory(
        reader,
        &header,
        DirectoryRef::Root,
        header.root_offset,
        header.root_length,
        &mut report,
        &mut walk,
    )?
    else {
        report.directory_bytes = walk.bytes_spent;
        return Ok(report);
    };
    report.root_entries = entries.len();

    let complete = walk_directory(
        reader,
        &header,
        DirectoryRef::Root,
        &entries,
        0,
        size,
        &mut report,
        &mut walk,
    )?;

    report.tile_contents = if walk.tracking_offsets {
        Some(walk.offsets.len() as u64)
    } else {
        None
    };
    report.directory_bytes = walk.bytes_spent;
    report.entries_visited = walk.entries_spent;

    // The header's three counts, checked only when the walk actually finished.
    // Comparing a partial count against the header would bury the real finding
    // under a mismatch that is a consequence of it.
    if complete {
        let counted_entries = report.tile_entries;
        if counted_entries != header.tile_entries_count {
            push(
                &mut report,
                Finding::EntryCountMismatch {
                    counted: counted_entries,
                    header: header.tile_entries_count,
                },
                limits,
            );
        }
        let counted_tiles = report.addressed_tiles;
        if counted_tiles != header.addressed_tiles_count {
            push(
                &mut report,
                Finding::AddressedTilesMismatch {
                    counted: counted_tiles,
                    header: header.addressed_tiles_count,
                },
                limits,
            );
        }
        if let Some(contents) = report.tile_contents
            && contents != header.tile_contents_count
        {
            push(
                &mut report,
                Finding::ContentsCountMismatch {
                    counted: contents,
                    header: header.tile_contents_count,
                },
                limits,
            );
        }
    }

    Ok(report)
}

/// Mutable state carried through the recursion.
struct Walk<'a> {
    limits: &'a ValidationLimits,
    offsets: BTreeSet<u64>,
    tracking_offsets: bool,
    /// Absolute file positions of every leaf already read, so a second pointer
    /// at one leaf costs a set lookup instead of a second inflate.
    visited_leaves: BTreeSet<u64>,
    /// Decompressed directory bytes spent so far, against `byte_budget`.
    bytes_spent: u64,
    byte_budget: u64,
    /// Directory entries visited so far, against `entry_budget`.
    entries_spent: u64,
    entry_budget: u64,
    /// Set once a budget runs out. Every level of the recursion checks it, so
    /// the walk unwinds instead of finishing the directory it is in.
    exhausted: bool,
}

/// Add a finding, unless the run has already collected as many as it will.
///
/// Returns whether there is room for more, so a caller sweeping thousands of
/// entries can stop rather than build a vector nobody will read.
fn push(report: &mut Report, finding: Finding, limits: &ValidationLimits) -> bool {
    if report.findings.len() >= limits.max_findings {
        return false;
    }
    report.findings.push(finding);
    if report.findings.len() == limits.max_findings {
        report.findings.push(Finding::FindingLimitReached {
            limit: limits.max_findings,
        });
        return false;
    }
    true
}

/// Read and decode the 127-byte header, or record why it could not be.
fn read_header<R: RangeReader + ?Sized>(
    reader: &R,
    size: Option<u64>,
    report: &mut Report,
) -> Result<Option<Header>, PmTilesError> {
    let need = HEADER_BYTES as u64;
    if let Some(size) = size
        && size < need
    {
        report
            .findings
            .push(Finding::ArchiveTooShort { size, need });
        return Ok(None);
    }

    let bytes = match reader.read_range(0, HEADER_BYTES) {
        Ok(bytes) => bytes,
        // A backend that could not say how big it was and then could not hand
        // over 127 bytes is the same defect as a short file, and reporting it
        // as an I/O error would make an unreadable archive indistinguishable
        // from an unreachable one.
        Err(e) if size.is_none() && e.kind() == io::ErrorKind::UnexpectedEof => {
            report
                .findings
                .push(Finding::ArchiveTooShort { size: 0, need });
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };

    match Header::try_decode(&bytes) {
        Ok(header) => Ok(Some(header)),
        Err(PmTilesError::BadMagic { found }) => {
            report.findings.push(Finding::BadMagic { found });
            Ok(None)
        }
        Err(PmTilesError::UnsupportedVersion { found }) => {
            report.findings.push(Finding::UnsupportedVersion { found });
            Ok(None)
        }
        Err(PmTilesError::ShortHeader { got, .. }) => {
            report.findings.push(Finding::ArchiveTooShort {
                size: got as u64,
                need,
            });
            Ok(None)
        }
        Err(other) => Err(other),
    }
}

/// Check one section's `offset + length` against the archive size.
///
/// Returns whether the section is usable, which is what stops the walk from
/// reading a range it has just rejected.
fn check_section(
    report: &mut Report,
    limits: &ValidationLimits,
    section: Section,
    offset: u64,
    length: u64,
    size: Option<u64>,
) -> bool {
    let Some(end) = offset.checked_add(length) else {
        push(
            report,
            Finding::SectionOverflow {
                section,
                offset,
                length,
            },
            limits,
        );
        return false;
    };
    match size {
        Some(size) if end > size => {
            push(
                report,
                Finding::SectionOutOfBounds {
                    section,
                    offset,
                    length,
                    archive_size: size,
                },
                limits,
            );
            false
        }
        // Without a size there is nothing to compare against, which
        // `ArchiveSizeUnknown` already says.
        _ => true,
    }
}

/// Fetch, decompress and parse one directory, recording why not if it fails.
#[allow(clippy::too_many_arguments)]
fn read_directory<R: RangeReader + ?Sized>(
    reader: &R,
    header: &Header,
    which: DirectoryRef,
    offset: u64,
    length: u64,
    report: &mut Report,
    walk: &mut Walk<'_>,
) -> Result<Option<Vec<Entry>>, PmTilesError> {
    let limits = walk.limits;
    // The total-work check, and the reason a 21 KB archive can no longer cost
    // 16 TiB of gzip output. It is deliberately in front of the read rather
    // than inside the decompressor: the budget is a ceiling on the whole walk,
    // so the last directory may overshoot it by at most one
    // `max_directory_bytes`, and in exchange a directory that would have been
    // legal on its own is never reported as unreadable.
    if walk.bytes_spent >= walk.byte_budget {
        push(
            report,
            Finding::TotalDirectoryBytesExceeded {
                directory: which,
                inflated: walk.bytes_spent,
                budget: walk.byte_budget,
            },
            limits,
        );
        walk.exhausted = true;
        return Ok(None);
    }
    let Ok(length_usize) = usize::try_from(length) else {
        push(
            report,
            Finding::DirectoryTooLarge {
                directory: which,
                length,
                limit: limits.max_directory_bytes,
            },
            limits,
        );
        return Ok(None);
    };
    if length_usize > limits.max_directory_bytes {
        push(
            report,
            Finding::DirectoryTooLarge {
                directory: which,
                length,
                limit: limits.max_directory_bytes,
            },
            limits,
        );
        return Ok(None);
    }

    let raw = match reader.read_range(offset, length_usize) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            push(
                report,
                Finding::DirectoryUnreadable {
                    directory: which,
                    reason: e.to_string(),
                },
                limits,
            );
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };

    let plain = match header
        .internal_compression
        .decompress(&raw, limits.max_directory_bytes)
    {
        Ok(plain) => plain,
        Err(e) => {
            push(
                report,
                Finding::DirectoryUnreadable {
                    directory: which,
                    reason: e.to_string(),
                },
                limits,
            );
            return Ok(None);
        }
    };

    walk.bytes_spent = walk.bytes_spent.saturating_add(plain.len() as u64);

    match deserialize_entries(&plain) {
        Ok(entries) => Ok(Some(entries)),
        Err(e) => {
            push(
                report,
                Finding::DirectoryUnreadable {
                    directory: which,
                    reason: e.to_string(),
                },
                limits,
            );
            Ok(None)
        }
    }
}

/// Walk one directory's entries, following leaf pointers.
///
/// Returns whether the subtree under this directory was walked to the end, so
/// the caller knows whether its totals are complete enough to compare against
/// the header.
#[allow(clippy::too_many_arguments)]
fn walk_directory<R: RangeReader + ?Sized>(
    reader: &R,
    header: &Header,
    which: DirectoryRef,
    entries: &[Entry],
    depth: usize,
    size: Option<u64>,
    report: &mut Report,
    walk: &mut Walk<'_>,
) -> Result<bool, PmTilesError> {
    let limits = walk.limits;
    let mut complete = true;

    // The other half of the total-work budget. Charged for the whole directory
    // before a single entry is inspected, because the cost of walking it is
    // already committed by the time the loop starts.
    walk.entries_spent = walk.entries_spent.saturating_add(entries.len() as u64);
    if walk.entries_spent > walk.entry_budget {
        push(
            report,
            Finding::TotalEntriesExceeded {
                directory: which,
                visited: walk.entries_spent,
                budget: walk.entry_budget,
            },
            limits,
        );
        walk.exhausted = true;
        return Ok(false);
    }

    // One past the last tile id the previous entry covers. `deserialize_entries`
    // has already refused a repeated or decreasing id, so the only ordering
    // defect left is a run that reaches past the next entry.
    let mut previous_end: u64 = 0;

    for (index, entry) in entries.iter().enumerate() {
        // A budget that ran out deeper in the tree unwinds the whole walk
        // rather than finishing the directory it happened to be in.
        if walk.exhausted {
            return Ok(false);
        }
        if index > 0 && entry.tile_id < previous_end {
            complete = false;
            if !push(
                report,
                Finding::OverlappingRun {
                    directory: which,
                    index,
                    tile_id: entry.tile_id,
                    previous_run_ends_at: previous_end,
                },
                limits,
            ) {
                return Ok(false);
            }
        }
        previous_end = if entry.is_leaf() {
            entry.tile_id.saturating_add(1)
        } else {
            entry.tile_id.saturating_add(u64::from(entry.run_length))
        };

        if entry.is_leaf() {
            if !follow_leaf(
                reader, header, which, index, entry, depth, size, report, walk,
            )? {
                complete = false;
            }
            continue;
        }

        // A tile entry. Its offset is relative to `tile_data_offset`, so the
        // bound is the section's length and not the archive's size.
        let end = entry.offset.checked_add(u64::from(entry.length));
        if end.is_none_or(|end| end > header.tile_data_length) {
            complete = false;
            if !push(
                report,
                Finding::EntryOutOfTileData {
                    directory: which,
                    index,
                    offset: entry.offset,
                    length: entry.length,
                },
                limits,
            ) {
                return Ok(false);
            }
        }

        report.tile_entries = report.tile_entries.saturating_add(1);
        report.addressed_tiles = report
            .addressed_tiles
            .saturating_add(u64::from(entry.run_length));
        if walk.tracking_offsets {
            if walk.offsets.len() >= limits.max_tracked_contents {
                walk.tracking_offsets = false;
                walk.offsets.clear();
            } else {
                walk.offsets.insert(entry.offset);
            }
        }
    }

    Ok(complete)
}

/// Resolve one leaf pointer and walk what it points at.
///
/// The offset is relative to `header.leaf_directories_offset`. Both bounds are
/// checked: inside the leaf region, and inside the archive, because a header
/// that moves the region can leave a pointer that is fine against the region's
/// length and still lands past the end of the file.
#[allow(clippy::too_many_arguments)]
fn follow_leaf<R: RangeReader + ?Sized>(
    reader: &R,
    header: &Header,
    which: DirectoryRef,
    index: usize,
    entry: &Entry,
    depth: usize,
    size: Option<u64>,
    report: &mut Report,
    walk: &mut Walk<'_>,
) -> Result<bool, PmTilesError> {
    let limits = walk.limits;

    if !header.has_leaves() {
        push(
            report,
            Finding::LeafPointerWithoutLeafSection {
                directory: which,
                index,
            },
            limits,
        );
        return Ok(false);
    }
    if depth >= limits.max_leaf_depth {
        push(
            report,
            Finding::LeafDepthExceeded {
                directory: which,
                index,
                limit: limits.max_leaf_depth,
            },
            limits,
        );
        return Ok(false);
    }
    if report.leaves.len() >= limits.max_leaf_directories {
        push(
            report,
            Finding::TooManyLeafDirectories {
                found: report.leaves.len() + 1,
                limit: limits.max_leaf_directories,
            },
            limits,
        );
        return Ok(false);
    }

    let within_section = entry
        .offset
        .checked_add(u64::from(entry.length))
        .is_some_and(|end| end <= header.leaf_directories_length);
    let absolute = header.leaf_directories_offset.checked_add(entry.offset);
    let within_archive = absolute
        .and_then(|absolute| absolute.checked_add(u64::from(entry.length)))
        .is_some_and(|end| size.is_none_or(|size| end <= size));

    if !within_section || !within_archive {
        push(
            report,
            Finding::LeafPointerOutOfBounds {
                directory: which,
                index,
                offset: entry.offset,
                length: entry.length,
            },
            limits,
        );
        return Ok(false);
    }
    let absolute = absolute.expect("the bound above proved the sum exists");

    // Already read this leaf. Follow it once and only once.
    //
    // A root holding 2^20 + 1 pointers that all resolve to the same valid leaf
    // is the shape that turned a 21,620-byte archive into an unbounded walk:
    // every pointer was in bounds, every entry inside the leaf was in bounds,
    // no finding fired, and the leaf was inflated a million times. The leaf's
    // contents cannot depend on which pointer reached it, so a second visit
    // can only repeat the first one's work and the first one's findings.
    //
    // The walk is marked incomplete because the header's three counts count
    // every reference and this walk now counts each leaf once, so comparing
    // them would manufacture a mismatch that says nothing about the archive.
    // A finding goes with it, because this module's contract is that anything
    // that stops the walk is something the operator gets told about, and an
    // incomplete walk with an empty finding list would let `is_valid` certify
    // an archive whose header counts were never checked.
    if !walk.visited_leaves.insert(absolute) {
        report.shared_leaf_pointers = report.shared_leaf_pointers.saturating_add(1);
        push(
            report,
            Finding::LeafDirectoryRevisited {
                directory: which,
                index,
                absolute_offset: absolute,
            },
            limits,
        );
        return Ok(false);
    }

    let position = report.leaves.len();
    let leaf_ref = DirectoryRef::Leaf(position);
    let Some(entries) = read_directory(
        reader,
        header,
        leaf_ref,
        absolute,
        u64::from(entry.length),
        report,
        walk,
    )?
    else {
        report.leaves.push(LeafReport {
            offset_in_section: entry.offset,
            absolute_offset: absolute,
            compressed_length: u64::from(entry.length),
            entries: 0,
            decoded: None,
        });
        return Ok(false);
    };

    report.leaves.push(LeafReport {
        offset_in_section: entry.offset,
        absolute_offset: absolute,
        compressed_length: u64::from(entry.length),
        entries: entries.len(),
        decoded: limits.collect_entries.then(|| entries.clone()),
    });

    walk_directory(
        reader,
        header,
        leaf_ref,
        &entries,
        depth + 1,
        size,
        report,
        walk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pmtiles::Compression;
    use crate::pmtiles::directory::serialize_entries;

    /// Build the smallest archive that holds `entries` and `tile_data`, with
    /// the header's three counts filled in from the entries so a clean input
    /// really is clean.
    fn archive(entries: &[Entry], tile_data: &[u8]) -> Vec<u8> {
        let directory = serialize_entries(entries).expect("the entries serialize");
        let mut header = Header {
            internal_compression: Compression::None,
            ..Header::default()
        };
        header.root_offset = HEADER_BYTES as u64;
        header.root_length = directory.len() as u64;
        header.metadata_offset = header.root_offset + header.root_length;
        header.metadata_length = 0;
        header.leaf_directories_offset = header.metadata_offset;
        header.leaf_directories_length = 0;
        header.tile_data_offset = header.metadata_offset;
        header.tile_data_length = tile_data.len() as u64;
        header.tile_entries_count = entries.iter().filter(|e| !e.is_leaf()).count() as u64;
        header.addressed_tiles_count = entries.iter().map(|e| u64::from(e.run_length)).sum();
        header.tile_contents_count = entries
            .iter()
            .filter(|e| !e.is_leaf())
            .map(|e| e.offset)
            .collect::<BTreeSet<u64>>()
            .len() as u64;

        let mut out = header.encode().to_vec();
        out.extend_from_slice(&directory);
        out.extend_from_slice(tile_data);
        out
    }

    fn tile(tile_id: u64, offset: u64, length: u32, run_length: u32) -> Entry {
        Entry {
            tile_id,
            offset,
            length,
            run_length,
        }
    }

    /// Build a gzip-internal archive whose root is nothing but leaf pointers.
    ///
    /// `leaves` is the leaf region, laid out in order. `pointers` is one index
    /// into it per root entry, so a repeated index is two pointers at one
    /// leaf, which is the shape the dedupe guard exists for. The header's three
    /// counts are filled in **per pointer**, the way a writer that really
    /// emitted this file would fill them, so the only thing a test is looking
    /// at is the walk.
    fn leafy_archive(leaves: &[Vec<Entry>], pointers: &[usize], tile_data: &[u8]) -> Vec<u8> {
        let gzip = Compression::Gzip;
        let mut bodies = Vec::new();
        let mut offsets = Vec::new();
        let mut at: u64 = 0;
        for leaf in leaves {
            let body = gzip
                .compress(&serialize_entries(leaf).expect("a leaf serializes"))
                .expect("a leaf compresses");
            offsets.push((at, body.len() as u32));
            at += body.len() as u64;
            bodies.push(body);
        }
        let leaf_region_length = at;

        let root: Vec<Entry> = pointers
            .iter()
            .enumerate()
            .map(|(index, &which)| {
                let (offset, length) = offsets[which];
                Entry {
                    // Ascending and unique, which is all `deserialize_entries`
                    // asks of the id column. Two pointers at one leaf are two
                    // different root entries whatever their target.
                    tile_id: index as u64,
                    offset,
                    length,
                    run_length: 0,
                }
            })
            .collect();
        let root_body = gzip
            .compress(&serialize_entries(&root).expect("the root serializes"))
            .expect("the root compresses");

        let mut header = Header {
            internal_compression: gzip,
            ..Header::default()
        };
        header.root_offset = HEADER_BYTES as u64;
        header.root_length = root_body.len() as u64;
        header.metadata_offset = header.root_offset + header.root_length;
        header.metadata_length = 0;
        header.leaf_directories_offset = header.metadata_offset;
        header.leaf_directories_length = leaf_region_length;
        header.tile_data_offset = header.leaf_directories_offset + leaf_region_length;
        header.tile_data_length = tile_data.len() as u64;
        header.tile_entries_count = pointers.iter().map(|&w| leaves[w].len() as u64).sum();
        header.addressed_tiles_count = pointers
            .iter()
            .map(|&w| {
                leaves[w]
                    .iter()
                    .map(|e| u64::from(e.run_length))
                    .sum::<u64>()
            })
            .sum();
        header.tile_contents_count = pointers
            .iter()
            .flat_map(|&w| leaves[w].iter().map(|e| e.offset))
            .collect::<BTreeSet<u64>>()
            .len() as u64;

        let mut out = header.encode().to_vec();
        out.extend_from_slice(&root_body);
        for body in &bodies {
            out.extend_from_slice(body);
        }
        out.extend_from_slice(tile_data);
        out
    }

    /// A leaf of `count` one-byte tile entries starting at `first`.
    fn leaf_of(first: u64, count: u64) -> Vec<Entry> {
        (0..count).map(|i| tile(first + i, 0, 1, 1)).collect()
    }

    #[test]
    fn a_hand_built_archive_validates_clean() {
        let entries = [tile(0, 0, 4, 1), tile(1, 4, 4, 2)];
        let report = validate_bytes(&archive(&entries, &[0u8; 8]), &ValidationLimits::default())
            .expect("a slice reads");
        assert_eq!(report.findings, Vec::new());
        assert_eq!(report.tile_entries, 2);
        assert_eq!(report.addressed_tiles, 3);
        assert_eq!(report.tile_contents, Some(2));
        assert_eq!(report.root_entries, 2);
    }

    #[test]
    fn an_entry_past_the_tile_data_is_a_finding() {
        let entries = [tile(0, 0, 64, 1)];
        let report = validate_bytes(&archive(&entries, &[0u8; 8]), &ValidationLimits::default())
            .expect("a slice reads");
        assert!(report.findings.contains(&Finding::EntryOutOfTileData {
            directory: DirectoryRef::Root,
            index: 0,
            offset: 0,
            length: 64,
        }));
    }

    #[test]
    fn an_overlapping_run_is_a_finding() {
        let entries = [tile(0, 0, 4, 5), tile(2, 4, 4, 1)];
        let report = validate_bytes(&archive(&entries, &[0u8; 8]), &ValidationLimits::default())
            .expect("a slice reads");
        assert!(report.findings.contains(&Finding::OverlappingRun {
            directory: DirectoryRef::Root,
            index: 1,
            tile_id: 2,
            previous_run_ends_at: 5,
        }));
    }

    #[test]
    fn the_finding_limit_stops_the_walk_and_says_so() {
        let entries: Vec<Entry> = (0..64).map(|i| tile(i, 4096, 64, 1)).collect();
        let limits = ValidationLimits {
            max_findings: 4,
            ..ValidationLimits::default()
        };
        let report = validate_bytes(&archive(&entries, &[0u8; 8]), &limits).expect("a slice reads");
        assert_eq!(report.findings.len(), 5, "four findings plus the marker");
        assert_eq!(
            report.findings.last(),
            Some(&Finding::FindingLimitReached { limit: 4 })
        );
    }

    #[test]
    fn a_zero_length_archive_is_too_short_rather_than_an_error() {
        let report = validate_bytes(&[], &ValidationLimits::default()).expect("a slice reads");
        assert_eq!(
            report.findings,
            vec![Finding::ArchiveTooShort { size: 0, need: 127 }]
        );
        assert!(report.header.is_none());
    }

    /// A thousand pointers at one leaf cost one read, not a thousand.
    ///
    /// This is the leaf storm in miniature. Every pointer is in bounds, every
    /// entry inside the leaf is in bounds, and before the dedupe set the walk
    /// inflated the same leaf once per pointer.
    #[test]
    fn a_leaf_reached_by_a_thousand_pointers_is_read_once() {
        let leaves = vec![leaf_of(0, 512)];
        let pointers = vec![0usize; 1000];
        let bytes = leafy_archive(&leaves, &pointers, &[0u8]);

        let report = validate_bytes(&bytes, &ValidationLimits::default()).expect("a slice reads");

        assert_eq!(report.leaves.len(), 1, "one distinct leaf, read once");
        assert_eq!(report.shared_leaf_pointers, 999);
        assert_eq!(
            report.tile_entries, 512,
            "the leaf's entries are counted once, not once per pointer"
        );
        // The whole walk inflated one root and one leaf, so the bytes are on
        // the order of a single leaf rather than a thousand of them.
        assert!(
            report.directory_bytes < 64 * 1024,
            "the walk inflated {} bytes for a 512-entry leaf",
            report.directory_bytes
        );
        assert!(
            report
                .findings
                .iter()
                .any(|f| matches!(f, Finding::LeafDirectoryRevisited { .. })),
            "a repeat has to be reported, not swallowed: {:?}",
            report.findings
        );
    }

    /// The archive the reviewer built, at the limits the CLI ships with.
    ///
    /// 21,620 bytes on disk, a root of 1,048,577 leaf pointers, every one of
    /// them resolving to one valid 16 MiB leaf of 4,194,303 in-bounds entries.
    /// Before the guards this walked forever; the assertion is on the work the
    /// walk reports doing, not on a stopwatch, because a stopwatch on a shared
    /// machine measures the machine.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_leaf_storm_seed_is_bounded_work_at_default_limits() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fuzz/corpus/fuzz_pmtiles_reader/nocrash-leaf-fanout"
        );
        let bytes = std::fs::read(path).expect("the seed is committed next to the fuzz target");
        // Pinned, because the whole point of this test is the shape of these
        // particular bytes and a regenerated fixture is a different test.
        assert_eq!(bytes.len(), 21_620, "the seed changed size");

        let limits = ValidationLimits::default();
        let report = validate_bytes(&bytes, &limits).expect("a slice reads");

        // One leaf read, a million pointers collapsed onto it.
        assert_eq!(report.leaves.len(), 1);
        assert_eq!(report.shared_leaf_pointers, 1_048_576);
        // The budget this archive earns is the floor, one directory's worth,
        // because its own section lengths are tiny. The walk is allowed to
        // overshoot by the directory it was already reading and no further.
        let budget = limits.max_directory_bytes as u64;
        assert!(
            report.directory_bytes <= budget + limits.max_directory_bytes as u64,
            "the walk inflated {} bytes against a {budget} byte budget",
            report.directory_bytes
        );
        // Positive control: it really did walk the archive rather than
        // refusing it at the door, which would satisfy the bound above for the
        // wrong reason.
        assert!(
            report.directory_bytes > 16 * 1024 * 1024,
            "the walk only inflated {} bytes, so it never reached the leaf",
            report.directory_bytes
        );
        assert_eq!(report.tile_entries, 4_194_303);
        assert!(!report.is_valid());
    }

    /// Distinct fat leaves are what dedupe cannot help with, and the budget
    /// can.
    #[test]
    fn the_total_byte_budget_stops_a_region_full_of_distinct_leaves() {
        let leaves: Vec<Vec<Entry>> = (0..32).map(|i| leaf_of(i * 1000, 256)).collect();
        let pointers: Vec<usize> = (0..32).collect();
        let bytes = leafy_archive(&leaves, &pointers, &[0u8]);

        let budget = 4096;
        let limits = ValidationLimits::default().with_max_total_directory_bytes(budget);
        let report = validate_bytes(&bytes, &limits).expect("a slice reads");

        assert!(
            report
                .findings
                .iter()
                .any(|f| matches!(f, Finding::TotalDirectoryBytesExceeded { .. })),
            "no budget finding in {:?}",
            report.findings
        );
        assert!(
            report.leaves.len() < 32,
            "the budget did not stop anything: {} leaves read",
            report.leaves.len()
        );
        // Positive control. A guard that refuses everything satisfies the line
        // above without being a budget at all.
        assert!(
            !report.leaves.is_empty(),
            "the budget refused the first leaf too, so it is not a budget"
        );
        assert!(
            report.directory_bytes <= budget + limits.max_directory_bytes as u64,
            "spent {} against a {budget} byte budget",
            report.directory_bytes
        );
    }

    /// The same walk, measured in entries rather than bytes.
    #[test]
    fn the_total_entry_budget_stops_a_region_full_of_distinct_leaves() {
        let leaves: Vec<Vec<Entry>> = (0..32).map(|i| leaf_of(i * 1000, 256)).collect();
        let pointers: Vec<usize> = (0..32).collect();
        let bytes = leafy_archive(&leaves, &pointers, &[0u8]);

        let limits = ValidationLimits::default().with_max_total_entries(1024);
        let report = validate_bytes(&bytes, &limits).expect("a slice reads");

        assert!(
            report
                .findings
                .iter()
                .any(|f| matches!(f, Finding::TotalEntriesExceeded { .. })),
            "no entry budget finding in {:?}",
            report.findings
        );
        assert!(report.leaves.len() < 32);
        assert!(!report.leaves.is_empty());
        // 32 root pointers plus whole leaves of 256, so the overshoot is one
        // leaf and never more.
        assert!(
            report.entries_visited <= 1024 + 256,
            "visited {} entries against a 1024 budget",
            report.entries_visited
        );
    }

    /// The negative control for both budgets: an ordinary leafy archive at the
    /// default limits must not come anywhere near them.
    ///
    /// The byte budget is derived from the archive's own section lengths, so
    /// this is the assertion that says an honest file cannot trip a guard
    /// aimed at a hostile one.
    #[test]
    fn an_honest_leafy_archive_never_reaches_a_budget() {
        let leaves: Vec<Vec<Entry>> = (0..64).map(|i| leaf_of(i * 1000, 256)).collect();
        let pointers: Vec<usize> = (0..64).collect();
        let bytes = leafy_archive(&leaves, &pointers, &[0u8]);

        let report = validate_bytes(&bytes, &ValidationLimits::default()).expect("a slice reads");

        assert_eq!(report.findings, Vec::new());
        assert_eq!(report.leaves.len(), 64);
        assert_eq!(report.shared_leaf_pointers, 0);
        assert_eq!(report.tile_entries, 64 * 256);
        assert!(report.directory_bytes > 0);
    }

    /// The budget has a floor of one directory's worth, and this is the archive
    /// that needs it.
    ///
    /// The derived half of the budget is the archive's own compressed
    /// directory span times an inflation allowance, which is generous against
    /// anything a real writer emits and stingy against a directory of
    /// identical entries, where gzip gets hundreds to one. The floor is what
    /// keeps a legal, tiny, absurdly compressible archive out of the refusal
    /// path.
    #[test]
    fn a_tiny_archive_whose_directories_compress_hugely_still_verifies() {
        let leaves = vec![leaf_of(0, 50_000), leaf_of(1_000_000, 50_000)];
        let bytes = leafy_archive(&leaves, &[0, 1], &[0u8]);

        let report = validate_bytes(&bytes, &ValidationLimits::default()).expect("a slice reads");

        assert_eq!(report.findings, Vec::new());
        assert_eq!(report.leaves.len(), 2);
        // The control. Without it this test would pass for an archive whose
        // ratio is ordinary, which would make it a test of nothing: the
        // derived budget has to be genuinely below the work involved, so that
        // the floor is the only thing that let the archive through.
        let header = report.header.expect("a header decoded");
        let derived =
            (header.root_length + header.leaf_directories_length) * DIRECTORY_INFLATION_ALLOWANCE;
        assert!(
            derived < report.directory_bytes,
            "the derived budget is {derived} and the archive only inflates {}, so the floor \
             is not what this test is measuring",
            report.directory_bytes
        );
    }

    #[test]
    fn every_finding_renders_something_a_person_can_read() {
        // A guard on the `Display` impl rather than on any one message: a new
        // variant with a `todo!()` or an empty arm would pass every other test
        // in this file.
        let samples = [
            Finding::ArchiveTooShort { size: 0, need: 127 },
            Finding::ArchiveSizeUnknown,
            Finding::BadMagic { found: *b"NOTPMTs" },
            Finding::UnsupportedVersion { found: 2 },
            Finding::SectionOutOfBounds {
                section: Section::Root,
                offset: 999_999,
                length: 35,
                archive_size: 1878,
            },
            Finding::SectionOverflow {
                section: Section::TileData,
                offset: u64::MAX,
                length: 2,
            },
            Finding::DirectoryTooLarge {
                directory: DirectoryRef::Root,
                length: 1 << 40,
                limit: 16,
            },
            Finding::DirectoryUnreadable {
                directory: DirectoryRef::Leaf(3),
                reason: "not gzip".to_owned(),
            },
            Finding::OverlappingRun {
                directory: DirectoryRef::Root,
                index: 1,
                tile_id: 2,
                previous_run_ends_at: 5,
            },
            Finding::EntryOutOfTileData {
                directory: DirectoryRef::Root,
                index: 0,
                offset: 0,
                length: 64,
            },
            Finding::LeafPointerOutOfBounds {
                directory: DirectoryRef::Root,
                index: 0,
                offset: 1,
                length: 2,
            },
            Finding::LeafPointerWithoutLeafSection {
                directory: DirectoryRef::Root,
                index: 0,
            },
            Finding::TooManyLeafDirectories { found: 2, limit: 1 },
            Finding::LeafDepthExceeded {
                directory: DirectoryRef::Leaf(0),
                index: 0,
                limit: 4,
            },
            Finding::EntryCountMismatch {
                counted: 1,
                header: 2,
            },
            Finding::AddressedTilesMismatch {
                counted: 85,
                header: 84,
            },
            Finding::ContentsCountMismatch {
                counted: 63,
                header: 62,
            },
            Finding::LeafDirectoryRevisited {
                directory: DirectoryRef::Root,
                index: 7,
                absolute_offset: 334,
            },
            Finding::TotalDirectoryBytesExceeded {
                directory: DirectoryRef::Leaf(1),
                inflated: 17_000_000,
                budget: 16_777_216,
            },
            Finding::TotalEntriesExceeded {
                directory: DirectoryRef::Leaf(1),
                visited: 5_000_000,
                budget: 4_194_304,
            },
            Finding::FindingLimitReached { limit: 256 },
        ];
        // Twenty-one variants. If a new one is added without a sample here,
        // this count is what notices, because a `Display` arm nobody exercises
        // renders for the first time in front of a user.
        assert_eq!(samples.len(), 21);
        for finding in &samples {
            let text = finding.to_string();
            assert!(!text.is_empty(), "{finding:?} renders as nothing");
            assert!(
                text.len() > 12,
                "{finding:?} renders as {text:?}, which says nothing useful"
            );
        }
    }
}
