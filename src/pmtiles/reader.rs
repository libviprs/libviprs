//! Random access into a PMTiles archive: header, metadata and one tile.
//!
//! [`Reader`] is the read half of the format. It answers "give me
//! `17/83922/44217`" in a couple of ranged reads against whatever
//! [`RangeReader`] it was built on, without scanning, without decompressing
//! the archive, and without ever holding more than one directory page and one
//! tile in memory. That is the property that makes a 40 GB pyramid on an
//! object store behave like a local file, and it is the reason this module
//! exists at all rather than a `Vec<u8>` and a loop.
//!
//! ```no_run
//! use libviprs::pmtiles::Reader;
//!
//! let reader = Reader::try_open("drawing.pmtiles")?;
//! println!("zooms {} to {}", reader.min_zoom(), reader.max_zoom());
//!
//! match reader.get_tile(2, 3, 3)? {
//!     Some(bytes) => println!("{} bytes", bytes.len()),
//!     None => println!("the archive does not hold that tile"),
//! }
//! # Ok::<(), libviprs::pmtiles::PmTilesError>(())
//! ```
//!
//! # What a lookup costs
//!
//! Opening an archive reads the 127-byte header and then the root directory,
//! which the spec requires to live inside the first 16 KiB precisely so a
//! latency-sensitive client can fetch both at once. Nothing else is read at
//! open time; the metadata section in particular is fetched lazily, because it
//! is the one section a reader can skip entirely and on a vector archive it
//! can be megabytes of `vector_layers`.
//!
//! After that, a tile in a leafless archive is **one** read. A tile in an
//! archive with one level of leaves is two: the leaf, then the payload. A leaf
//! that has already been fetched is served from a small cache
//! ([`MAX_CACHED_LEAVES`]) so a clustered walk does not refetch the same
//! 4096-entry page for every tile in it.
//!
//! # The lookup rule, which the specification does not contain
//!
//! PMTiles v3 describes the directory format and never describes a lookup.
//! Grepping the specification for `lookup`, `search` and `binary search`
//! returns nothing against working positive controls, so the algorithm below
//! is derived from what an entry *means* and it is the part where two
//! implementations can legally differ.
//!
//! To find tile `T` in a directory:
//!
//! 1. take the **largest entry whose `tile_id` is at most `T`**. If there is
//!    none, `T` is below the first entry and the tile is absent;
//! 2. if that entry's run length is 0 it is a leaf pointer. Fetch the leaf and
//!    start again inside it. A miss inside a leaf is final: the parent's entry
//!    only said which leaf could hold `T`;
//! 3. otherwise the tile is present **only if `T < tile_id + run_length`**.
//!
//! Step 3 is the whole answer to "how does a reader tell absent from part of a
//! run", and it is the single easiest thing to get wrong here. Because step 1
//! takes the largest qualifying entry, a lookup for a tile the archive does
//! not hold *lands on the preceding entry* rather than finding nothing. A
//! reader that treats landing as a hit returns the previous tile's bytes for
//! every hole in the archive, and since those bytes are a real, decodable tile
//! it looks like it is working. A round trip that only asks for tiles it wrote
//! never sees it. The check lives on [`Entry::run_contains`].
//!
//! # The base an offset is relative to
//!
//! An entry's offset is relative to a section chosen by the **entry's kind**,
//! not by the directory the entry was found in:
//!
//! * a tile entry, wherever it was found, is relative to
//!   [`Header::tile_data_offset`];
//! * a leaf pointer is relative to [`Header::leaf_directories_offset`].
//!
//! A tile entry six levels down a leaf chain is still relative to the tile
//! data section. The two tempting mistakes are making it relative to the
//! leaf's own start or to the leaf section, and both of them are invisible to
//! a round trip because a writer that makes the same wrong choice agrees with
//! the reader exactly. `Reader::entry_position` is the one place that decision
//! is made, and `tests/pmtiles_reader.rs` pins the resulting absolute byte
//! ranges against an archive go-pmtiles wrote.
//!
//! # Every archive is hostile
//!
//! A `.pmtiles` file is data somebody hands you, and the reference
//! implementation is not a safety model to copy: its `verify` checks
//! `length > fileSize` and never `offset + length`, so a root offset of 999999
//! in an 1878-byte file walks straight through it and then nil-dereferences
//! inside gzip. Passing `pmtiles verify` is not evidence that an archive is
//! safe to parse.
//!
//! So, in this module:
//!
//! * every section is checked as `offset.checked_add(length) <= archive size`,
//!   not as a length on its own, and every entry is checked against the length
//!   of the section it addresses;
//! * every decompression is capped ([`MAX_DIRECTORY_BYTES`],
//!   [`MAX_METADATA_BYTES`]). No length field anywhere in PMTiles v3 is an
//!   uncompressed length, so a cap is the only thing between a reader and a
//!   few hundred bytes that inflate without bound;
//! * leaf following is depth-capped ([`MAX_LEAF_DEPTH`]), because the spec
//!   only *discourages* nesting and states no limit, so a cycle is
//!   expressible;
//! * an unsupported version, an unsupported internal compression and an
//!   out-of-grid coordinate are typed refusals. The reference masks an
//!   out-of-range `x` into a different valid tile and serves it; matching that
//!   would be matching a bug.
//!
//! There is no panicking public entry point. That includes the lock around the
//! leaf cache, which is recovered from poisoning rather than unwrapped.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::pmtiles::directory::deserialize_entries;
use crate::pmtiles::header::HEADER_BYTES;
use crate::pmtiles::writer::DEFAULT_LEAF_ENTRIES;
use crate::pmtiles::{
    Entry, FileRangeReader, Header, Metadata, PmTilesError, RangeReader, TileType, zxy_to_tileid,
};

/// The furthest byte the root directory may reach.
///
/// The spec states this twice and not quite identically: the root "MUST be
/// contained in the first 16,384 bytes", and separately the maximum compressed
/// root is "16384 bytes - 127 bytes". The two agree only when the root starts
/// immediately after the header. `root_offset + root_length <= 16384` is the
/// rule that satisfies both, and it is what this reader enforces.
pub const MAX_ROOT_SPAN: u64 = 16_384;

/// The most a directory may decompress to.
///
/// Chosen rather than derived, because nothing in the format bounds it. Four
/// mebibytes is around a million entries, which is two orders of magnitude
/// past the 4096-entry leaves go-pmtiles writes, so it refuses bombs without
/// refusing anything real. The cap is absolute rather than a ratio, because
/// the compressed size is attacker-controlled too.
pub const MAX_DIRECTORY_BYTES: usize = 4 * 1024 * 1024;

/// The most the metadata section may decompress to.
///
/// The same reasoning as [`MAX_DIRECTORY_BYTES`]. A vector archive's
/// `vector_layers` is the realistic upper end and it is kilobytes.
pub const MAX_METADATA_BYTES: usize = 4 * 1024 * 1024;

/// How many leaf directories a lookup may follow before giving up.
///
/// The spec says only that more than one level is "discouraged" and gives no
/// limit, so a hostile archive can point a leaf at itself. Four is generous
/// against a format whose own writers emit one level.
pub const MAX_LEAF_DEPTH: u8 = 4;

/// Decoded directory entries the leaf cache may hold, summed across every
/// leaf it is holding.
///
/// This is the cache's memory bound and the number the leaf count is derived
/// from, in that order, because a count of leaves is not a bound at all:
/// `read_directory` lets one leaf decode to [`MAX_DIRECTORY_BYTES`] of wire
/// format, and at four bytes an entry that is about a million entries.
///
/// 262144 entries is 6 MiB of [`Entry`] on a 64-bit target, and that figure is
/// the real ceiling rather than a leading one. `remember_leaf` refuses a leaf
/// that is over the budget on its own instead of keeping it, so the sum the
/// cache holds is under this number at every instant rather than under it
/// except for one retained blob. The lookup that decoded such a leaf is
/// holding the [`Arc`] either way, so the cache slot bought nothing and cost
/// the ceiling four times over.
pub const MAX_CACHED_LEAF_ENTRIES: usize = 256 * 1024;

/// How many decoded leaf directories the cache holds.
///
/// Derived from [`MAX_CACHED_LEAF_ENTRIES`] rather than picked, and that is
/// the whole of it. I picked 16 by hand first, against a budget of 262144
/// entries and a writer that puts 4096 entries in a leaf, so the count bound
/// bit at 25% of the memory bound and the memory bound never bound at all on
/// any archive this crate writes. Dividing one by the other means the two
/// cannot disagree again, and the `const _` below says so at compile time.
///
/// Sizing it matters more than it looks, because an LRU over uniformly random
/// leaves is a step and not a slope. Measured on fabricated archives of
/// 4096-entry leaves, 20000 random lookups each: at a cache of 16, sixteen
/// leaves cost 0.38 us a lookup and **seventeen cost 7.37 us**, a nineteenfold
/// jump for one more leaf, because the miss rate of an LRU of `k` over `N`
/// uniformly random leaves is `1 - k/N` and the first miss is the one that
/// pays a ranged read and a directory decode. At 64 leaves the same archive is
/// 93.40 us at a cache of 16 and 0.69 us at a cache of 64. So the cliff does
/// not soften with size, it only moves, and putting it where the memory bound
/// already sits is free.
///
/// What a miss costs, measured on a realistic leaf (9157 stored bytes, 22647
/// plain): 84 to 103 us in total, of which `deserialize_entries` is 52 to 68
/// and the gzip inflate is 32 to 35. The varint decode is about 62% of it and
/// the compression is a third, which is the opposite of what I wrote here
/// first. Anyone reaching for a cheaper miss should go at the decode.
///
/// This is still deliberately off the correctness path: a cold cache changes
/// how many reads happen and never what they return. The key is `(offset,
/// length)` rather than the offset alone for exactly that reason, since the
/// decode is a function of the range and two root entries may point at one
/// offset with two lengths.
pub const MAX_CACHED_LEAVES: usize = MAX_CACHED_LEAF_ENTRIES / DEFAULT_LEAF_ENTRIES;

// The two bounds, held together where a change to either one has to pass.
// Picking the count by hand is what let it sit at a quarter of the budget in
// the first place.
const _: () = assert!(MAX_CACHED_LEAVES * DEFAULT_LEAF_ENTRIES <= MAX_CACHED_LEAF_ENTRIES);
const _: () = assert!(MAX_CACHED_LEAVES > 0);

/// What the leaf cache is keyed on: where a leaf starts and how long it is.
///
/// Both halves, because `read_directory` decodes a range. An archive may
/// legally carry two root entries whose leaf pointers share an offset and
/// differ in length, and the two decode to different directories (or one
/// decodes and the other is a typed refusal). Keying on the offset alone made
/// the second lookup answer with the first lookup's directory, which is a
/// reader whose result depends on what was asked before it.
type LeafKey = (u64, u32);

/// A PMTiles v3 archive opened for random access.
///
/// Generic over its transport so an HTTP or S3 backend drops in without
/// touching anything here: implement [`RangeReader::read_range`], which is the
/// only method [`RangeReader`] requires, and this type works unchanged.
/// [`RangeReader::size`] is defaulted to `Ok(None)` and answering it is what
/// buys the section bounds checks in [`Reader::try_new`]. `Reader` is
/// `Send + Sync` whenever its source is, which the trait requires, so one
/// reader can serve the engine's worker threads.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::{Reader, RangeReader};
/// # use std::io;
///
/// // Any byte source will do. This one is a slice.
/// #[derive(Debug)]
/// struct Bytes(Vec<u8>);
/// impl RangeReader for Bytes {
///     fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
///         let start = offset as usize;
///         self.0
///             .get(start..start + len)
///             .map(<[u8]>::to_vec)
///             .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
///     }
///     fn size(&self) -> io::Result<Option<u64>> {
///         Ok(Some(self.0.len() as u64))
///     }
/// }
///
/// // Not an archive, so opening it is a typed refusal rather than a panic.
/// let err = Reader::try_new(Bytes(vec![0u8; 200])).unwrap_err();
/// assert!(matches!(err, libviprs::pmtiles::PmTilesError::BadMagic { .. }));
/// ```
#[derive(Debug)]
pub struct Reader<R: RangeReader> {
    source: R,
    header: Header,
    root: Vec<Entry>,
    /// The archive's size when the backend knows it. `None` is a real answer
    /// from a streaming transport, and it disables only the checks that need a
    /// total; an entry is still checked against the section that owns it.
    archive_size: Option<u64>,
    metadata: OnceLock<Metadata>,
    /// Most recently used first. Behind a `Mutex` rather than a lock-free
    /// structure because it is touched once per leaf, not once per byte.
    ///
    /// Keyed on the **range**, `(offset, length)`, and not on the offset
    /// alone. What a leaf decodes to is a function of both, because
    /// `read_directory` reads a range, so two root entries pointing at one
    /// offset with two lengths are two different directories and a cache that
    /// could not tell them apart made a warm reader answer what a cold reader
    /// refused.
    leaves: Mutex<Vec<(LeafKey, Arc<Vec<Entry>>)>>,
}

impl Reader<FileRangeReader> {
    /// Open a `.pmtiles` file for random access.
    ///
    /// Reads the header and the root directory and nothing else. `try_` rather
    /// than a bare `open` because it can fail and the crate spells a fallible
    /// constructor that way.
    pub fn try_open(path: impl AsRef<Path>) -> Result<Self, PmTilesError> {
        Self::try_new(FileRangeReader::try_open(path)?)
    }
}

impl<R: RangeReader> Reader<R> {
    /// Open an archive over any byte source.
    ///
    /// Two reads happen here and no more: the 127-byte header, then the root
    /// directory at the offset and length the header gives. Everything the
    /// header claims about where the other sections are is bounds-checked
    /// before any of it is believed.
    pub fn try_new(source: R) -> Result<Self, PmTilesError> {
        let archive_size = source.size()?;
        if let Some(size) = archive_size
            && size < HEADER_BYTES as u64
        {
            return Err(PmTilesError::ShortHeader {
                got: usize::try_from(size).unwrap_or(usize::MAX),
                want: HEADER_BYTES,
            });
        }

        let header = Header::try_decode(&source.read_range(0, HEADER_BYTES)?)?;

        // Every section, checked as a sum. This is the check the reference
        // implementation does not make, and the one that turns its segfault
        // into an error here. It comes before the root's own 16 KiB budget
        // below, so an offset wildly past the end of the file is reported as
        // what it is rather than as a conformance complaint.
        for (name, offset, length) in [
            ("root directory", header.root_offset, header.root_length),
            ("metadata", header.metadata_offset, header.metadata_length),
            (
                "leaf directories",
                header.leaf_directories_offset,
                header.leaf_directories_length,
            ),
            (
                "tile data",
                header.tile_data_offset,
                header.tile_data_length,
            ),
        ] {
            let end = offset
                .checked_add(length)
                .ok_or(PmTilesError::Overflow { what: name })?;
            if let Some(size) = archive_size
                && end > size
            {
                return Err(PmTilesError::SectionOutOfBounds {
                    section: name,
                    offset,
                    length,
                    archive: size,
                });
            }
        }

        let root_end =
            header
                .root_offset
                .checked_add(header.root_length)
                .ok_or(PmTilesError::Overflow {
                    what: "root directory",
                })?;
        if root_end > MAX_ROOT_SPAN {
            return Err(PmTilesError::RootDirectoryTooLarge {
                end: root_end,
                limit: MAX_ROOT_SPAN,
            });
        }

        let root = read_directory(
            &source,
            header.internal_compression,
            header.root_offset,
            header.root_length,
        )?;

        Ok(Self {
            source,
            header,
            root,
            archive_size,
            metadata: OnceLock::new(),
            leaves: Mutex::new(Vec::new()),
        })
    }

    /// The decoded 127-byte header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The byte source underneath, for a caller that wrapped one.
    pub fn source(&self) -> &R {
        &self.source
    }

    /// The root directory's entries, in tile id order.
    ///
    /// Exposed because it is the cheap way to ask structural questions about
    /// an archive (how many entries, what the runs look like, whether the root
    /// is all leaf pointers) without a lookup per tile.
    pub fn root_entries(&self) -> &[Entry] {
        &self.root
    }

    /// The archive's size, when the backend knows it.
    pub fn archive_size(&self) -> Option<u64> {
        self.archive_size
    }

    /// The lowest zoom level the archive claims to hold.
    pub fn min_zoom(&self) -> u8 {
        self.header.min_zoom
    }

    /// The highest zoom level the archive claims to hold.
    pub fn max_zoom(&self) -> u8 {
        self.header.max_zoom
    }

    /// What the tile blobs are.
    ///
    /// Reported as stored, including a value this build does not recognise:
    /// `0x00` means "the writer did not know" while a value past the end of
    /// the enum means "a newer spec knows something we do not", and mapping
    /// the second onto the first throws away the only signal that we are out
    /// of date.
    pub fn tile_format(&self) -> TileType {
        self.header.tile_type
    }

    /// The bounding box as degrees, `(min_lon, min_lat, max_lon, max_lat)`.
    ///
    /// Longitude first in each pair, which is the order the bytes are in. The
    /// spec's prose describes these fields as "the minimum latitude and
    /// minimum longitude", latitude first, and only the byte layout is
    /// normative. Getting it backwards produces a bounding box that still
    /// looks plausible, which is why it is worth saying twice.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        self.header.bounds_degrees()
    }

    /// The JSON metadata, fetched and parsed on first use and kept.
    ///
    /// Lazy because it is the one section a lookup never needs, and capped
    /// because the header carries no uncompressed size for it.
    ///
    /// Unknown keys survive: the format has no other extension point, so a
    /// read that dropped them would destroy another tool's data on the next
    /// write. See [`Metadata::extra`].
    pub fn metadata(&self) -> Result<&Metadata, PmTilesError> {
        if let Some(cached) = self.metadata.get() {
            return Ok(cached);
        }
        let parsed = self.load_metadata()?;
        Ok(self.metadata.get_or_init(|| parsed))
    }

    /// Whether the archive holds `(z, x, y)`.
    ///
    /// Walks the directories exactly as [`Reader::get_tile`] does and stops
    /// before fetching the payload, so asking this and then asking for the
    /// tile costs one extra directory walk and never an extra tile read.
    /// [`Reader::tile_span`] is the same walk with the entry's numbers kept
    /// rather than thrown away.
    ///
    /// An out-of-grid coordinate is an error rather than `false`: "this tile
    /// does not exist" and "that is not a tile" are different answers and only
    /// one of them is about the archive.
    pub fn tile_exists(&self, z: u8, x: u32, y: u32) -> Result<bool, PmTilesError> {
        Ok(self.tile_span(z, x, y)?.is_some())
    }

    /// Where one tile's bytes sit and how many of them there are, without
    /// fetching any of them.
    ///
    /// The directory walk [`Reader::get_tile`] does already produces both
    /// numbers, because an entry carries its payload's offset and its stored
    /// length, so a caller that only needs to know how large a tile is pays
    /// the walk and no payload read at all. That is the whole difference
    /// between reading an index and reading an archive: checking a
    /// 21851-tile pyramid through `get_tile` pulls every byte off the
    /// transport to learn 21851 numbers the directories were already carrying.
    ///
    /// The offset is absolute, so it goes straight into
    /// [`RangeReader::read_range`],
    /// and `Ok(None)` means the archive does not hold the tile on exactly the
    /// terms [`Reader::get_tile`] uses: a pyramid may have holes, and a
    /// coordinate outside the id space is an error rather than an absence.
    ///
    /// # What a span does not prove
    ///
    /// That the bytes at that offset are reachable. A length comes out of a
    /// directory entry and an entry can point anywhere the header's sections
    /// allow, so the thing that makes the cheap answer safe to lean on is
    /// [`validate`](crate::pmtiles::validate) having bounds-checked every
    /// entry against the archive's real size, which it can only do when the
    /// backend reported one.
    pub fn tile_span(&self, z: u8, x: u32, y: u32) -> Result<Option<(u64, u32)>, PmTilesError> {
        let tile_id = zxy_to_tileid(z, x, y)?;
        self.locate(tile_id)
    }

    /// Fetch one tile.
    ///
    /// `Ok(None)` means the archive does not hold it, which is not an error:
    /// a pyramid is allowed to have holes and a caller asking for a tile
    /// outside the covered area is doing something ordinary. `Err` is reserved
    /// for the archive being wrong or the coordinate not being a coordinate.
    ///
    /// The bytes come back exactly as stored, still compressed if
    /// [`Header::tile_compression`] says so. A reader that hands tiles onward
    /// never has to decompress one, which is why an archive with brotli or
    /// zstd tiles is readable by this build even though it carries neither
    /// codec.
    pub fn get_tile(&self, z: u8, x: u32, y: u32) -> Result<Option<Vec<u8>>, PmTilesError> {
        let tile_id = zxy_to_tileid(z, x, y)?;
        let Some((at, length)) = self.locate(tile_id)? else {
            return Ok(None);
        };
        let len = usize::try_from(length).map_err(|_| PmTilesError::Overflow {
            what: "a tile length",
        })?;
        Ok(Some(self.source.read_range(at, len)?))
    }

    /// Walk root then leaves for `tile_id`, returning the payload's absolute
    /// position and stored length.
    ///
    /// Split out from [`Reader::get_tile`] so existence can be answered
    /// without the payload read, and so the directory walk is one piece of
    /// code rather than two that can drift.
    fn locate(&self, tile_id: u64) -> Result<Option<(u64, u32)>, PmTilesError> {
        let mut depth: u8 = 0;
        let mut leaf: Option<Arc<Vec<Entry>>> = None;

        loop {
            let found = {
                let directory: &[Entry] = match leaf.as_deref() {
                    Some(entries) => entries,
                    None => &self.root,
                };
                largest_entry_at_or_below(directory, tile_id)
            };

            // Below the first entry of this directory. Inside a leaf that is
            // final: the pointer that sent us here only said which leaf could
            // hold the tile, so there is nothing to fall back to.
            let Some(entry) = found else {
                return Ok(None);
            };

            if !entry.is_leaf() {
                // Landing on an entry is not a hit. This is the run-end check.
                if !entry.run_contains(tile_id) {
                    return Ok(None);
                }
                let at = self.entry_position(
                    "tile data",
                    self.header.tile_data_offset,
                    self.header.tile_data_length,
                    &entry,
                )?;
                return Ok(Some((at, entry.length)));
            }

            if depth >= MAX_LEAF_DEPTH {
                return Err(PmTilesError::LeafDepthExceeded {
                    limit: MAX_LEAF_DEPTH,
                });
            }
            let at = self.entry_position(
                "leaf directories",
                self.header.leaf_directories_offset,
                self.header.leaf_directories_length,
                &entry,
            )?;
            leaf = Some(self.leaf_directory(at, entry.length)?);
            depth += 1;
        }
    }

    /// Turn an entry's relative offset into an absolute file position, after
    /// checking it fits inside the section it addresses.
    ///
    /// `section_offset` and `section_length` come from the **header**, chosen
    /// by the entry's kind. They are never derived from the directory the
    /// entry was found in, which is the whole content of this function and the
    /// reason it exists rather than being two additions at the call sites: a
    /// tile entry found inside a leaf is relative to the tile data section,
    /// not to the leaf and not to the leaf section.
    fn entry_position(
        &self,
        section: &'static str,
        section_offset: u64,
        section_length: u64,
        entry: &Entry,
    ) -> Result<u64, PmTilesError> {
        let length = u64::from(entry.length);
        let end = entry
            .offset
            .checked_add(length)
            .ok_or(PmTilesError::Overflow {
                what: "an entry's extent",
            })?;
        if end > section_length {
            return Err(PmTilesError::EntryOutOfBounds {
                section,
                offset: entry.offset,
                length,
                section_length,
            });
        }
        section_offset
            .checked_add(entry.offset)
            .ok_or(PmTilesError::Overflow {
                what: "an entry's absolute position",
            })
    }

    /// A leaf directory, from the cache if it is there.
    ///
    /// The cache is consulted with the whole range rather than with the
    /// offset, so a hit is a leaf that was decoded from exactly these bytes.
    fn leaf_directory(&self, at: u64, length: u32) -> Result<Arc<Vec<Entry>>, PmTilesError> {
        let key: LeafKey = (at, length);
        if let Some(cached) = self.cached_leaf(key) {
            return Ok(cached);
        }
        let entries = Arc::new(read_directory(
            &self.source,
            self.header.internal_compression,
            at,
            u64::from(length),
        )?);
        self.remember_leaf(key, Arc::clone(&entries));
        Ok(entries)
    }

    fn cached_leaf(&self, key: LeafKey) -> Option<Arc<Vec<Entry>>> {
        let mut leaves = self.lock_leaves();
        #[cfg(pmtiles_lock_probe)]
        let depth = leaves.len();
        let index = leaves.iter().position(|(cached, _)| *cached == key)?;
        #[cfg(pmtiles_lock_probe)]
        lock_probe::record_hit(depth, index);
        let hit = leaves.remove(index);
        let entries = Arc::clone(&hit.1);
        leaves.insert(0, hit);
        Some(entries)
    }

    fn remember_leaf(&self, key: LeafKey, entries: Arc<Vec<Entry>>) {
        let mut leaves = self.lock_leaves();
        leaves.retain(|(cached, _)| *cached != key);

        // A leaf over the whole budget by itself is handed back and not held.
        // The alternative is to keep it because the lookup in flight is
        // holding it anyway, which is true of the `Arc` and not of the cache
        // slot: the slot keeps it alive after the lookup ends, and that is the
        // difference between a 6 MiB ceiling and a 24 MiB one.
        if entries.len() > MAX_CACHED_LEAF_ENTRIES {
            return;
        }

        leaves.insert(0, (key, entries));
        leaves.truncate(MAX_CACHED_LEAVES);

        // Then the budget, which is the bound the count is derived from.
        // Evicting from the back is the same LRU order the truncate above
        // uses, and the leaf just inserted is at the front, so it is the last
        // thing this could reach and the early return above means it never
        // has to.
        let mut held: usize = leaves.iter().map(|(_, leaf)| leaf.len()).sum();
        while held > MAX_CACHED_LEAF_ENTRIES {
            let Some((_, dropped)) = leaves.pop() else {
                break;
            };
            held -= dropped.len();
        }
    }

    /// The cache lock, recovered rather than unwrapped.
    ///
    /// Nothing in this module can panic while holding it, so poisoning would
    /// have to come from somewhere else entirely, and a cache is not worth a
    /// panicking public entry point on an archive somebody handed us.
    #[cfg(not(pmtiles_lock_probe))]
    fn lock_leaves(&self) -> std::sync::MutexGuard<'_, Vec<(LeafKey, Arc<Vec<Entry>>)>> {
        match self.leaves.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// The same lock, with the wait and the hold timed (see [`lock_probe`]).
    ///
    /// Written as a second whole function rather than as `#[cfg]` lines inside
    /// the first so that the shipped body above is exactly the body that was
    /// there before the probe existed, character for character.
    #[cfg(pmtiles_lock_probe)]
    fn lock_leaves(&self) -> lock_probe::TimedGuard<'_> {
        let before = std::time::Instant::now();
        let guard = match self.leaves.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        lock_probe::TimedGuard::new(guard, before)
    }

    fn load_metadata(&self) -> Result<Metadata, PmTilesError> {
        Metadata::try_from_json(&self.metadata_json()?)
    }

    /// The metadata section decompressed, before anything tries to make sense
    /// of it.
    ///
    /// Exists because [`Metadata`] parses the whole object or none of it, and
    /// "none of it" is a state a reader still has to say something useful
    /// about. A single unknown `format` variant written by a later libviprs
    /// fails the parse at the outermost object, taking `name`, `description`,
    /// `attribution` and the whole `extra` map down with it, and the only way
    /// to tell that archive apart from a foreign one is to look at the bytes
    /// for a `vnd.libviprs` key. That is what
    /// [`PmTilesPyramidReader::describe`](crate::pyramid_reader::PmTilesPyramidReader)
    /// does with this (issue #1123).
    ///
    /// Not cached, unlike [`Reader::metadata`]. The only caller is the failure
    /// path, which by definition has nothing to cache, and caching raw bytes
    /// beside a parsed object would mean two representations of one section
    /// that can disagree.
    pub(crate) fn metadata_json(&self) -> Result<Vec<u8>, PmTilesError> {
        let stored = usize::try_from(self.header.metadata_length).unwrap_or(usize::MAX);
        if stored > MAX_METADATA_BYTES {
            return Err(PmTilesError::DecompressionLimit {
                limit: MAX_METADATA_BYTES,
            });
        }
        let raw = self
            .source
            .read_range(self.header.metadata_offset, stored)?;
        self.header
            .internal_compression
            .decompress(&raw, MAX_METADATA_BYTES)
    }
}

/// The largest entry whose tile id is at most `tile_id`.
///
/// This is step 1 of the lookup and it is not an exact-match search. An exact
/// match finds only the first id of each run and reports every other id in the
/// run as absent, which on a deduplicated pyramid is most of the archive.
fn largest_entry_at_or_below(entries: &[Entry], tile_id: u64) -> Option<Entry> {
    let above = entries.partition_point(|entry| entry.tile_id <= tile_id);
    if above == 0 {
        None
    } else {
        entries.get(above - 1).copied()
    }
}

/// Fetch, decompress and decode one directory.
///
/// The stored length is refused before the read when it is over the
/// decompression ceiling, so a hostile header cannot make a reader pull
/// gigabytes off a network before deciding it did not want them.
fn read_directory<R: RangeReader>(
    source: &R,
    compression: crate::pmtiles::Compression,
    offset: u64,
    length: u64,
) -> Result<Vec<Entry>, PmTilesError> {
    let stored = usize::try_from(length).unwrap_or(usize::MAX);
    if stored > MAX_DIRECTORY_BYTES {
        return Err(PmTilesError::DecompressionLimit {
            limit: MAX_DIRECTORY_BYTES,
        });
    }
    let raw = source.read_range(offset, stored)?;
    let plain = compression.decompress(&raw, MAX_DIRECTORY_BYTES)?;
    deserialize_entries(&plain)
}

/// Lock-wait instrumentation for the leaf cache (issue #1021).
///
/// This exists because the concurrent p99 tail on the leaf-bearing cell was
/// measured as *latency* and blamed on the [`Mutex`] around `leaves`, and a
/// latency number cannot tell the lock apart from anything else in the
/// lookup. Nothing outside this crate can see how long a private mutex was
/// waited on or held, so the only honest way to answer it is from in here.
///
/// It is compiled **only** under `--cfg pmtiles_lock_probe`, which nothing in
/// `ci.yml`, `merge-gate.yml`, the `Makefile`, `tools/local-ci.py` or any
/// published profile sets. Without that cfg every item below disappears and
/// `Reader::lock_leaves` is the `MutexGuard` function it has always been,
/// which is why that one is written out twice rather than sprinkled with
/// `#[cfg]` lines: the shipped body has to stay byte-identical to what it was.
/// The cfg is declared in `Cargo.toml`'s `check-cfg` list beside `cfg(loom)`,
/// which is the same shape for the same reason.
///
/// Counters are thread-local and monotonic, so a caller reads
/// [`snapshot`] before and after a lookup and subtracts. That costs one `Cell`
/// read per lookup instead of an allocation per lock acquisition, which
/// matters when the thing being measured is a few hundred nanoseconds.
///
/// What it perturbs: two `Instant::now` calls per acquisition (a vDSO
/// `clock_gettime`, tens of nanoseconds) and one more inside the critical
/// section when the guard drops. So a hold time reported here is an
/// overestimate by roughly one clock read, and that is the direction that
/// flatters the "the lock is the problem" hypothesis rather than the reverse.
#[cfg(pmtiles_lock_probe)]
pub mod lock_probe {
    use std::cell::Cell;
    use std::time::Duration;

    /// What one thread has spent on the leaf-cache lock so far.
    ///
    /// Everything is a running total except the two maxima, so two snapshots
    /// subtract into "what that lookup cost" and the maxima are read whole.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Stats {
        /// How many times this thread took the lock.
        pub acquisitions: u64,
        /// Nanoseconds spent blocked in `Mutex::lock`, summed.
        pub blocked_ns: u64,
        /// Nanoseconds spent holding the guard, summed.
        pub held_ns: u64,
        /// The longest single block, in nanoseconds.
        pub blocked_max_ns: u64,
        /// The longest single hold, in nanoseconds.
        pub held_max_ns: u64,
        /// Cache hits served, which is the path that reorders.
        pub hits: u64,
        /// Slots the linear scan walked before finding the hit, summed. This
        /// is `index + 1` per hit.
        pub scanned: u64,
        /// Slots the `remove` plus `insert(0)` reorder has to memmove,
        /// summed. That is `index` slots shifted down and then `index` back
        /// up, so `2 * index` per hit.
        pub reordered: u64,
        /// The deepest the cache was seen to be, in slots. This is the number
        /// the "a hit memmoves up to 64 slots" claim rests on.
        pub max_depth: u64,
    }

    impl Stats {
        const ZERO: Self = Self {
            acquisitions: 0,
            blocked_ns: 0,
            held_ns: 0,
            blocked_max_ns: 0,
            held_max_ns: 0,
            hits: 0,
            scanned: 0,
            reordered: 0,
            max_depth: 0,
        };
    }

    thread_local! {
        static STATS: Cell<Stats> = const { Cell::new(Stats::ZERO) };
    }

    /// This thread's totals as they stand.
    pub fn snapshot() -> Stats {
        STATS.with(|stats| stats.get())
    }

    /// Put this thread's totals back to zero.
    pub fn reset() {
        STATS.with(|stats| stats.set(Stats::ZERO));
    }

    fn update(f: impl FnOnce(&mut Stats)) {
        STATS.with(|stats| {
            let mut current = stats.get();
            f(&mut current);
            stats.set(current);
        });
    }

    /// One acquisition, recorded once when the guard drops.
    ///
    /// Both halves land in a single thread-local update on purpose. The
    /// critical section has to carry whatever this costs, and the thing being
    /// measured is a few tens of nanoseconds, so the probe holds the lock for
    /// exactly one clock read on the way in, one on the way out and one `Cell`
    /// round trip, and not a byte more.
    fn record(blocked: Duration, held: Duration) {
        let blocked_ns = blocked.as_nanos() as u64;
        let held_ns = held.as_nanos() as u64;
        update(|stats| {
            stats.acquisitions += 1;
            stats.blocked_ns += blocked_ns;
            stats.blocked_max_ns = stats.blocked_max_ns.max(blocked_ns);
            stats.held_ns += held_ns;
            stats.held_max_ns = stats.held_max_ns.max(held_ns);
        });
    }

    /// A hit at `index` in a cache `depth` slots deep.
    pub(super) fn record_hit(depth: usize, index: usize) {
        update(|stats| {
            stats.hits += 1;
            stats.scanned += index as u64 + 1;
            stats.reordered += 2 * index as u64;
            stats.max_depth = stats.max_depth.max(depth as u64);
        });
    }

    /// The guard `Reader::lock_leaves` hands back under this cfg.
    ///
    /// It derefs to the `Vec` the real guard derefs to, so every call site
    /// reads exactly as it does without the probe, and its `Drop` records the
    /// hold before the inner guard releases the lock.
    pub struct TimedGuard<'a> {
        guard: std::sync::MutexGuard<'a, Vec<(super::LeafKey, std::sync::Arc<Vec<super::Entry>>)>>,
        /// When the caller asked for the lock, and when it got it. The wait is
        /// the difference and the hold runs from the second one.
        asked: std::time::Instant,
        acquired: std::time::Instant,
    }

    impl<'a> TimedGuard<'a> {
        pub(super) fn new(
            guard: std::sync::MutexGuard<
                'a,
                Vec<(super::LeafKey, std::sync::Arc<Vec<super::Entry>>)>,
            >,
            asked: std::time::Instant,
        ) -> Self {
            Self {
                guard,
                asked,
                acquired: std::time::Instant::now(),
            }
        }
    }

    impl std::ops::Deref for TimedGuard<'_> {
        type Target = Vec<(super::LeafKey, std::sync::Arc<Vec<super::Entry>>)>;

        fn deref(&self) -> &Self::Target {
            &self.guard
        }
    }

    impl std::ops::DerefMut for TimedGuard<'_> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.guard
        }
    }

    impl Drop for TimedGuard<'_> {
        fn drop(&mut self) {
            let released = std::time::Instant::now();
            record(
                self.acquired.duration_since(self.asked),
                released.duration_since(self.acquired),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pmtiles::Compression;
    use crate::pmtiles::directory::serialize_entries;
    use std::io;

    /// A byte source with no size, which is what a streaming transport
    /// genuinely reports.
    struct Sized(Vec<u8>, bool);

    impl RangeReader for Sized {
        fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let start = usize::try_from(offset)
                .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            self.0
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
        }

        fn size(&self) -> io::Result<Option<u64>> {
            Ok(self.1.then_some(self.0.len() as u64))
        }
    }

    fn archive(entries: &[Entry], tiles: &[u8]) -> Vec<u8> {
        let root = Compression::Gzip
            .compress(&serialize_entries(entries).expect("the root serialises"))
            .expect("gzip");
        let metadata = Compression::Gzip.compress(b"{}").expect("gzip");
        let metadata_offset = 127 + root.len() as u64;
        let tile_data_offset = metadata_offset + metadata.len() as u64;
        let header = Header {
            root_offset: 127,
            root_length: root.len() as u64,
            metadata_offset,
            metadata_length: metadata.len() as u64,
            leaf_directories_offset: tile_data_offset,
            leaf_directories_length: 0,
            tile_data_offset,
            tile_data_length: tiles.len() as u64,
            ..Header::default()
        };
        let mut out = Vec::new();
        out.extend_from_slice(&header.encode());
        out.extend_from_slice(&root);
        out.extend_from_slice(&metadata);
        out.extend_from_slice(tiles);
        out
    }

    #[test]
    fn the_search_takes_the_largest_entry_at_or_below_and_never_an_exact_match() {
        let entries = [
            Entry {
                tile_id: 5,
                offset: 0,
                length: 1,
                run_length: 3,
            },
            Entry {
                tile_id: 21,
                offset: 1,
                length: 1,
                run_length: 1,
            },
        ];

        assert_eq!(largest_entry_at_or_below(&entries, 4), None);
        assert_eq!(largest_entry_at_or_below(&entries, 5), Some(entries[0]));
        // The point: an id inside the run lands on the run's entry, which an
        // exact-match search would miss.
        assert_eq!(largest_entry_at_or_below(&entries, 7), Some(entries[0]));
        // And an id in the hole after the run still lands on it, which is why
        // the run-end check has to exist separately.
        assert_eq!(largest_entry_at_or_below(&entries, 9), Some(entries[0]));
        assert!(!entries[0].run_contains(9));
        assert_eq!(
            largest_entry_at_or_below(&entries, u64::MAX),
            Some(entries[1])
        );
        assert_eq!(largest_entry_at_or_below(&[], 5), None);
    }

    #[test]
    fn a_backend_that_does_not_know_its_size_still_opens_and_still_bounds_entries() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let bytes = archive(&entries, b"TILE");

        // Known size and unknown size both open, and both serve the tile.
        for knows_its_size in [true, false] {
            let reader =
                Reader::try_new(Sized(bytes.clone(), knows_its_size)).expect("the archive opens");
            assert_eq!(reader.archive_size().is_some(), knows_its_size);
            assert_eq!(
                reader.get_tile(0, 0, 0).expect("a lookup").as_deref(),
                Some(&b"TILE"[..])
            );
        }
    }

    #[test]
    fn an_entry_reaching_past_its_section_is_refused_even_without_an_archive_size() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let mut bytes = archive(&entries, b"TILE");
        // Shrink the declared tile data section to two bytes. Nothing about
        // the archive's own size can catch this, so the section check is the
        // only thing that does.
        bytes[64..72].copy_from_slice(&2u64.to_le_bytes());

        let reader = Reader::try_new(Sized(bytes, false)).expect("the archive opens");
        assert!(matches!(
            reader.get_tile(0, 0, 0),
            Err(PmTilesError::EntryOutOfBounds { .. })
        ));
    }

    /// A leaf key, with the length fixed, for the tests that only vary offset.
    fn at(offset: u64) -> LeafKey {
        (offset, 64)
    }

    #[test]
    fn the_leaf_cache_keeps_the_most_recent_and_forgets_the_oldest() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let reader = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");

        for offset in 0..(MAX_CACHED_LEAVES as u64 + 2) {
            reader.remember_leaf(at(offset), Arc::new(Vec::new()));
        }
        assert_eq!(reader.lock_leaves().len(), MAX_CACHED_LEAVES);
        // The two oldest are gone and the newest is first.
        assert!(reader.cached_leaf(at(0)).is_none());
        assert!(reader.cached_leaf(at(1)).is_none());
        assert!(
            reader
                .cached_leaf(at(MAX_CACHED_LEAVES as u64 + 1))
                .is_some()
        );
    }

    /// Two leaves at one offset with two lengths are two cache entries.
    ///
    /// The unit half of the regression in `tests/pmtiles_reader.rs`, which
    /// builds the archive that makes this reachable through `get_tile`. Here
    /// it is just the key: a cache keyed on the offset would answer the second
    /// lookup with the first leaf's entries, and the entries are what a tile
    /// lookup then resolves against.
    #[test]
    fn two_leaves_at_one_offset_with_two_lengths_do_not_collide() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let reader = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");

        let short = Arc::new(vec![Entry::default(); 1]);
        let long = Arc::new(vec![Entry::default(); 2]);
        reader.remember_leaf((4096, 100), Arc::clone(&short));
        reader.remember_leaf((4096, 200), Arc::clone(&long));

        assert_eq!(
            reader.cached_leaf((4096, 100)).map(|leaf| leaf.len()),
            Some(1),
            "the short range should still answer with the short leaf"
        );
        assert_eq!(
            reader.cached_leaf((4096, 200)).map(|leaf| leaf.len()),
            Some(2),
            "the long range should answer with the long leaf"
        );
        assert!(
            reader.cached_leaf((4096, 300)).is_none(),
            "a range nothing decoded is a miss, not the nearest offset"
        );
    }

    /// The budget evicts before the count does, when the leaves are big.
    ///
    /// Without this the cache is bounded by a leaf count, and a leaf has no
    /// size limit short of [`MAX_DIRECTORY_BYTES`], so a count of leaves is
    /// not a number of bytes at all. The control is the last third: leaves
    /// small enough to fit the budget are all kept, so this is not simply a
    /// cache that evicts everything.
    #[test]
    fn the_leaf_cache_evicts_on_its_entry_budget_before_its_leaf_count() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let reader = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");

        // Three leaves, each 40% of the budget: the third pushes the total
        // over and the oldest goes, long before the count cap.
        let big = MAX_CACHED_LEAF_ENTRIES * 2 / 5;
        let leaf = || Arc::new(vec![Entry::default(); big]);
        for offset in 0..3u64 {
            reader.remember_leaf(at(offset), leaf());
        }
        assert!(
            reader.lock_leaves().len() < 3,
            "three leaves at 40% of the budget each should not all be held"
        );
        assert!(
            reader.cached_leaf(at(2)).is_some(),
            "the leaf just decoded is still held while it fits the budget"
        );
        assert!(
            held_entries(&reader) <= MAX_CACHED_LEAF_ENTRIES,
            "the cache held {} entries over a budget of {MAX_CACHED_LEAF_ENTRIES}",
            held_entries(&reader)
        );

        // One leaf bigger than the whole budget is handed back and not held.
        // Keeping it would make the advertised ceiling a quarter of the real
        // one, and the lookup that decoded it holds the `Arc` regardless.
        reader.remember_leaf(
            at(9),
            Arc::new(vec![Entry::default(); MAX_CACHED_LEAF_ENTRIES + 1]),
        );
        assert!(
            reader.cached_leaf(at(9)).is_none(),
            "a leaf over the whole budget should not take a cache slot"
        );
        assert!(
            held_entries(&reader) <= MAX_CACHED_LEAF_ENTRIES,
            "an over-budget leaf left {} entries in the cache",
            held_entries(&reader)
        );

        // The control: a full cache of ordinary 4096-entry leaves is kept, so
        // this is not simply a cache that evicts everything.
        let fresh = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");
        for offset in 0..MAX_CACHED_LEAVES as u64 {
            fresh.remember_leaf(at(offset), Arc::new(vec![Entry::default(); 4096]));
        }
        assert_eq!(fresh.lock_leaves().len(), MAX_CACHED_LEAVES);
    }

    /// Entries the cache is holding, which is the quantity the budget bounds.
    fn held_entries<R: RangeReader>(reader: &Reader<R>) -> usize {
        reader
            .lock_leaves()
            .iter()
            .map(|(_, leaf)| leaf.len())
            .sum()
    }

    #[test]
    fn a_lookup_hit_moves_its_leaf_back_to_the_front() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let reader = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");

        for offset in 0..MAX_CACHED_LEAVES as u64 {
            reader.remember_leaf(at(offset), Arc::new(Vec::new()));
        }
        // Touch the oldest, then push one more in. Without the move-to-front
        // the touched one would be the one evicted.
        assert!(reader.cached_leaf(at(0)).is_some());
        reader.remember_leaf(at(9999), Arc::new(Vec::new()));
        assert!(reader.cached_leaf(at(0)).is_some());
        assert!(reader.cached_leaf(at(1)).is_none());
    }
}
