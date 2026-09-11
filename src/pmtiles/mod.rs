//! PMTiles v3: the single-file, indexed archive format for tile pyramids.
//!
//! A pyramid is a naturally immutable, indexed artifact, and libviprs has so
//! far materialised it as millions of loose files under a `{z}/{x}/{y}` tree
//! ([`FsSink`](crate::sink::FsSink)). At fleet scale that is catastrophic for
//! inodes, backups and object stores. PMTiles replaces the tree with one file
//! that still answers a single-tile question in a couple of ranged reads, and
//! this module is the format layer every other piece of that work sits on.
//!
//! What is here is the *format*, not the I/O: the 127-byte header, the
//! `(z, x, y)` to `u64` Hilbert [`TileID`](tileid) mapping and its inverse,
//! the bounded LEB128 varint the directories are built from, the directory
//! [`Entry`] model with the spec's column-oriented delta encoding and its
//! run-length rule, the libviprs [`Metadata`] schema, and the [`RangeReader`]
//! abstraction a reader fetches bytes through. The indexed reader, the
//! streaming writer and the `TileSink` integration are separate issues built
//! on top of this one.
//!
//! # Implemented here rather than taken from a crate
//!
//! `pmtiles` exists on crates.io and this crate does not use it. Three
//! reasons, in the order they mattered:
//!
//! * **The writer is the hard half and the crate does not have one.** What
//!   libviprs needs is a bounded-memory streaming writer that can finalise a
//!   multi-gigabyte archive without holding its index in RAM. That is the part
//!   this epic is really about, and it would have had to be written either
//!   way.
//! * **The async model does not fit.** The crate's reader is built around
//!   `async` + `reqwest` + `tokio`. libviprs's engine is synchronous and does
//!   its own scheduling with `std::thread::scope`
//!   ([`streaming_mapreduce`](crate::streaming_mapreduce)); dragging a second
//!   runtime underneath it to read a local file is the wrong shape.
//! * **The dependency budget.** `CONTRIBUTING.md` sets a deliberately high bar
//!   for a new dependency. This module adds **none**: the gzip the spec asks
//!   for is `flate2`, already in the tree, and the varint is fifty lines.
//!
//! # Everything here treats its input as hostile
//!
//! A `.pmtiles` file is data somebody hands you, so every parser in this
//! module is written for input that is actively trying to break it. The rules
//! it follows, each of which has a test:
//!
//! * **no panicking public entry point.** Every fallible operation returns
//!   [`PmTilesError`]. There is no public `unwrap`, no slice index that has
//!   not been bounds-checked first, and no arithmetic that can overflow: every
//!   `+` on a value that came out of a file is a `checked_add`.
//! * **bounded varints.** [`varint::decode_uvarint`] refuses at the eleventh
//!   byte and refuses a tenth byte that would set bits above 63.
//! * **bounded counts.** A directory cannot claim more entries than its own
//!   byte count could possibly hold, so a ten-byte input cannot ask for a
//!   ten-million-entry allocation.
//! * **bounded decompression.** No length field anywhere in PMTiles v3 is an
//!   uncompressed length, so a reader cannot pre-size a decompression buffer
//!   and has to cap the output instead. [`Compression::decompress`] takes that
//!   cap as an argument rather than defaulting it, because the right value
//!   depends on what is being decompressed.
//!
//! # Naming
//!
//! Two conventions, so the module reads consistently:
//!
//! * a **constructor** that can fail is `try_*` ([`Header::try_decode`],
//!   [`FileRangeReader::try_open`]), matching the crate's `try_*` convention;
//! * a **transformation** keeps its plain name and puts its fallibility in the
//!   return type ([`zxy_to_tileid`], [`directory::serialize_entries`]). There
//!   is no panicking twin of any of them to distinguish it from.
//!
//! The types are re-exported here; the free functions and constants stay
//! behind their module path ([`varint::decode_uvarint`],
//! [`tileid::MAX_ZOOM`]), matching the curated crate root in
//! [`lib.rs`](crate). The two tile id functions are the exception, because
//! they are the module's headline arithmetic and every consumer reaches for
//! them by name.
//!
//! # What the spec does not say
//!
//! The v3 specification leaves a number of things to the implementation, and
//! each place this module makes a choice rather than following a rule is
//! marked in the docs with the reasoning. The two biggest:
//!
//! * **There is no algorithm for the TileID mapping.** The spec gives a
//!   one-sentence description, a link to a Wikipedia article that no longer
//!   carries the code it used to, and seven table rows. See [`tileid`] for
//!   what that means for testing, because it is worse than it sounds.
//! * **There is no lookup algorithm**, so the rule that separates a hit from a
//!   miss inside a directory is derived rather than quoted. It lives on
//!   [`Entry::run_contains`], where a reader cannot miss it.
//!
//! # Examples
//!
//! ```
//! use libviprs::pmtiles::{zxy_to_tileid, tileid_to_zxy};
//!
//! // The Hilbert ordering is cumulative across zooms: zoom 1 starts at 1,
//! // zoom 2 at 5, and the four zoom-1 tiles are walked as a "U".
//! assert_eq!(zxy_to_tileid(1, 0, 0).unwrap(), 1);
//! assert_eq!(zxy_to_tileid(1, 0, 1).unwrap(), 2);
//! assert_eq!(zxy_to_tileid(1, 1, 1).unwrap(), 3);
//! assert_eq!(zxy_to_tileid(1, 1, 0).unwrap(), 4);
//! assert_eq!(zxy_to_tileid(2, 0, 0).unwrap(), 5);
//!
//! assert_eq!(tileid_to_zxy(5).unwrap(), (2, 0, 0));
//! ```

pub mod directory;
pub mod header;
pub mod metadata;
pub mod range;
pub mod reader;
pub mod tileid;
pub mod validate;
pub mod varint;
pub mod writer;

pub use directory::Entry;
pub use header::{Compression, Header, TileType};
pub use metadata::{LibviprsMetadata, Metadata};
pub use range::{FileRangeReader, RangeReader};
pub use reader::Reader;
pub use tileid::{tileid_to_zxy, zxy_to_tileid};
pub use writer::{Writer, WriterOptions};

// ---------------------------------------------------------------------------
// PmTilesError
// ---------------------------------------------------------------------------

/// Everything that can go wrong reading, writing or addressing a PMTiles
/// archive.
///
/// One enum for the whole module rather than one per submodule, because the
/// call sites compose: a reader resolving a single tile runs a header decode,
/// a decompression, a varint sweep and a range read, and a caller that had to
/// map four error types through `?` to say "this archive is broken" would be
/// doing bookkeeping instead of handling the failure.
///
/// Every variant names the value it rejected. A malformed archive is somebody
/// else's file, and "invalid directory" without the number that was invalid is
/// not something a user can act on.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PmTilesError {
    /// An underlying read or write failed. Carries the `std::io::Error` so the
    /// `source()` chain survives rather than being flattened into a string.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The first seven bytes were not `PMTiles`. Carries what was found,
    /// because the usual cause is a file that is not a PMTiles archive at all
    /// and the first bytes are the quickest way to see what it is instead.
    #[error("not a PMTiles archive: magic is {found:?}, expected b\"PMTiles\"")]
    BadMagic { found: [u8; 7] },

    /// The magic matched but the version byte was not 3. PMTiles v2 shares the
    /// same seven magic bytes with a completely different layout, so reading a
    /// v2 file as v3 produces plausible-looking garbage rather than an error.
    /// That is exactly why this is a hard refusal.
    #[error("unsupported PMTiles version {found}, this reader implements version 3")]
    UnsupportedVersion { found: u8 },

    /// Fewer than 127 bytes were handed to [`Header::try_decode`].
    #[error("header is {got} bytes, the v3 header is {want} bytes")]
    ShortHeader { got: usize, want: usize },

    /// A zoom level above 31, which the `u64` TileID space cannot address.
    /// See [`tileid::MAX_ZOOM`] for the arithmetic behind the ceiling.
    #[error("zoom {zoom} is above the maximum addressable zoom {max}")]
    ZoomOutOfRange { zoom: u8, max: u8 },

    /// An `x` or `y` outside the `2^z` grid of its own zoom level.
    #[error("tile ({z}, {x}, {y}) is outside the {side}x{side} grid of zoom {z}")]
    CoordOutOfRange { z: u8, x: u32, y: u32, side: u64 },

    /// A TileID at or above the first id of zoom 32, which no `(z, x, y)`
    /// triple can produce. See [`tileid::MAX_TILE_ID`].
    #[error("tile id {id} is above the maximum addressable id {max}")]
    TileIdOutOfRange { id: u64, max: u64 },

    /// A varint ran off the end of its buffer with its continuation bit still
    /// set.
    #[error("varint at offset {offset} is truncated: the buffer ends mid-value")]
    TruncatedVarint { offset: usize },

    /// A varint claimed more than ten bytes, or a tenth byte carrying bits
    /// above 63. Either way the value cannot be a `u64`, and every varint in
    /// PMTiles v3 encodes a 64-bit field.
    #[error("varint at offset {offset} does not fit in a u64")]
    VarintOverflow { offset: usize },

    /// A directory claimed a number of entries its own byte count cannot
    /// hold. Each of the four columns needs at least one byte per entry, so
    /// no honest directory claims more entries than a quarter of the bytes it
    /// has left.
    #[error("directory claims {claimed} entries but only {remaining} bytes remain after the count")]
    DirectoryTooManyEntries { claimed: u64, remaining: usize },

    /// A directory header said zero entries. The spec makes a non-empty
    /// directory a `MUST`, and an empty one has no meaning: a directory exists
    /// to point at something.
    #[error("directory is empty, which the v3 spec forbids")]
    EmptyDirectory,

    /// Two entries carried the same TileID, or a delta pushed the running id
    /// past the addressable range. Entries are delta-encoded with unsigned
    /// varints, so they can only ascend; a zero delta after the first entry
    /// means two entries claim one tile and a lookup cannot say which wins.
    #[error("directory entry {index} has a non-ascending tile id")]
    NonAscendingEntry { index: usize },

    /// A directory entry field that does not fit the width this crate models
    /// it with. `length` and `run_length` are varints on the wire and `u32`
    /// here, which is the same choice every other implementation makes: a
    /// four-gigabyte tile or a run of four billion identical tiles is not a
    /// thing an honest writer produces.
    #[error("directory entry {index} has a {field} of {value}, too large for the field")]
    EntryFieldTooLarge {
        index: usize,
        field: &'static str,
        value: u64,
    },

    /// An entry length of zero. The spec states twice that a length `MUST` be
    /// greater than zero, and a zero-length run would make the contiguous
    /// offset shorthand ambiguous.
    #[error("directory entry {index} has a zero length, which the v3 spec forbids")]
    ZeroLengthEntry { index: usize },

    /// The offset column carried a literal `0` for the first entry. The
    /// shorthand means "contiguous with the previous entry" and the first
    /// entry has no previous entry, so the spec's encoder always writes
    /// `offset + 1` there. Decoding it as `value - 1` underflows a `u64` to
    /// 18446744073709551615 in release builds rather than panicking, which is
    /// why this is refused by name.
    #[error("directory entry 0 uses the contiguous-offset shorthand, which has no previous entry")]
    ContiguousOffsetAtFirstEntry,

    /// Bytes were left over after the four columns were read. The format has
    /// no extension mechanism, so trailing bytes mean the buffer is not the
    /// directory it claims to be.
    #[error("{remaining} bytes left over after the directory's four columns")]
    TrailingDirectoryBytes { remaining: usize },

    /// Arithmetic on values that came out of an archive overflowed. The
    /// payload names the sum that could not be taken, because on a 64-bit
    /// offset the only way to reach here is a hostile or corrupt file.
    #[error("overflow computing {what}")]
    Overflow { what: &'static str },

    /// A compression this build cannot perform. Brotli and Zstd are legal
    /// PMTiles compressions and this crate carries neither codec; `Unknown` is
    /// a writer telling you it does not know what it did.
    #[error("{compression} is not a compression this build can decode")]
    UnsupportedCompression { compression: Compression },

    /// A decompression produced more bytes than the caller allowed. No length
    /// field in PMTiles v3 is an uncompressed length, so this ceiling is the
    /// only thing standing between a reader and a gzip bomb.
    #[error("decompressed output exceeded the {limit} byte ceiling")]
    DecompressionLimit { limit: usize },

    /// The metadata section was not a UTF-8 JSON object, or did not match the
    /// libviprs schema where it claimed to.
    #[error("metadata is not valid libviprs PMTiles metadata: {0}")]
    Metadata(#[from] serde_json::Error),

    /// A libviprs [`TileFormat`](crate::sink::TileFormat) with no PMTiles
    /// [`TileType`] to carry it. `Raw` is the case: a PMTiles tile is a
    /// self-describing image blob and raw pixel bytes are not one.
    #[error("{format:?} has no PMTiles tile type")]
    UnsupportedTileFormat { format: crate::sink::TileFormat },

    /// Two tiles claimed one TileID. The directory encoding cannot express it:
    /// the tile id column holds unsigned deltas, so a repeat would need a zero
    /// delta, and a lookup landing on it could not say which of the two wins.
    /// Surfaced by [`Writer::finish`](crate::pmtiles::Writer::finish) rather
    /// than by `add_tile`, because a streaming writer only learns the ids are
    /// equal once they are in order.
    #[error("tile id {tile_id} was added more than once")]
    DuplicateTile { tile_id: u64 },

    /// Two payloads arrived under one content hash with different lengths.
    /// The writer trusts the caller's digest and never re-derives it, so this
    /// is the one inconsistency it can see: either the hash is not a hash of
    /// the bytes it came with, or two different payloads collided.
    #[error("content hash covers {stored} stored bytes but {length} were offered")]
    ContentHashMismatch { length: u32, stored: u32 },

    /// The root directory would not fit the spec's 16384-byte ceiling even
    /// with every entry pushed down into one leaf, so there is no leaf size
    /// that satisfies the budget.
    #[error("root directory is {length} bytes, over the {budget} byte budget")]
    RootDirectoryOverBudget { length: usize, budget: usize },

    /// A write failed part way through and the writer will not publish.
    ///
    /// [`writer::Writer`] latches the first
    /// failure and refuses everything afterwards, because a `write_all` that
    /// fails has usually written *some* of its bytes. Without the latch those
    /// orphan bytes stay in the staging file, the next accepted tile records an
    /// offset pointing into the middle of them, every later payload is shifted
    /// by the same amount, and `finish` publishes a structurally valid archive
    /// full of the wrong tile bytes. Under a retrying sink and
    /// `FailurePolicy::RetryThenSkip` the run reports success while doing it,
    /// which is the worst possible way to lose data.
    ///
    /// `during` names the step that failed, because "the writer failed" without
    /// it sends an operator to the wrong place. ENOSPC is the realistic cause:
    /// this writer needs roughly twice the archive's size in scratch.
    #[error("the writer failed while {during}, so it will not publish an archive")]
    WriterFailed { during: &'static str },

    /// A section the header describes does not fit inside the archive.
    ///
    /// The check is on `offset + length`, never on the length alone. The
    /// reference implementation's `verify` checks only the length, which is
    /// how a root offset of 999999 in an 1878-byte file passes it and then
    /// crashes the parser behind it.
    #[error(
        "the {section} at offset {offset} for {length} bytes does not fit in a {archive} byte archive"
    )]
    SectionOutOfBounds {
        section: &'static str,
        offset: u64,
        length: u64,
        archive: u64,
    },

    /// The root directory reaches past the first 16384 bytes of the archive.
    /// The spec makes that a `MUST` so a latency-sensitive client can fetch
    /// the header and the whole root in one request.
    #[error("the root directory ends at byte {end}, past the {limit} byte budget")]
    RootDirectoryTooLarge { end: u64, limit: u64 },

    /// A directory entry addresses bytes outside the section it belongs to.
    /// The section is chosen by the entry's kind: the tile data section for a
    /// tile entry, wherever it was found, and the leaf directories section for
    /// a leaf pointer.
    #[error(
        "an entry at offset {offset} for {length} bytes does not fit in the {section} section of {section_length} bytes"
    )]
    EntryOutOfBounds {
        section: &'static str,
        offset: u64,
        length: u64,
        section_length: u64,
    },

    /// A lookup followed leaf pointers past the depth this reader allows. The
    /// spec only discourages nesting and states no limit, so a cycle is
    /// expressible and a reader without a cap of its own follows it forever.
    #[error("a lookup followed more than {limit} levels of leaf directory")]
    LeafDepthExceeded { limit: u8 },
}
