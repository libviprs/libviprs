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
//! ([`LEAF_CACHE_ENTRIES`]) so a clustered walk does not refetch the same
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
//! the reader exactly. [`Reader::entry_position`] is the one place that
//! decision is made, and `tests/pmtiles_reader.rs` pins the resulting absolute
//! byte ranges against an archive go-pmtiles wrote.
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

/// How many decoded leaf directories are kept.
///
/// Small on purpose. This is a latency optimisation for a clustered walk,
/// where thousands of consecutive lookups land in the same leaf, and it is
/// deliberately off the correctness path: a cold cache changes how many reads
/// happen and never what they return.
pub const LEAF_CACHE_ENTRIES: usize = 4;

/// A PMTiles v3 archive opened for random access.
///
/// Generic over its transport so an HTTP or S3 backend drops in without
/// touching anything here: implement the three methods of [`RangeReader`] and
/// this type works unchanged. `Reader` is `Send + Sync` whenever its source
/// is, which the trait requires, so one reader can serve the engine's worker
/// threads.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::{Reader, RangeReader};
/// # use std::io;
///
/// // Any byte source will do. This one is a slice.
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
    leaves: Mutex<Vec<(u64, Arc<Vec<Entry>>)>>,
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
    ///
    /// An out-of-grid coordinate is an error rather than `false`: "this tile
    /// does not exist" and "that is not a tile" are different answers and only
    /// one of them is about the archive.
    pub fn tile_exists(&self, z: u8, x: u32, y: u32) -> Result<bool, PmTilesError> {
        let tile_id = zxy_to_tileid(z, x, y)?;
        Ok(self.locate(tile_id)?.is_some())
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
    fn leaf_directory(&self, at: u64, length: u32) -> Result<Arc<Vec<Entry>>, PmTilesError> {
        if let Some(cached) = self.cached_leaf(at) {
            return Ok(cached);
        }
        let entries = Arc::new(read_directory(
            &self.source,
            self.header.internal_compression,
            at,
            u64::from(length),
        )?);
        self.remember_leaf(at, Arc::clone(&entries));
        Ok(entries)
    }

    fn cached_leaf(&self, at: u64) -> Option<Arc<Vec<Entry>>> {
        let mut leaves = self.lock_leaves();
        let index = leaves.iter().position(|(offset, _)| *offset == at)?;
        let hit = leaves.remove(index);
        let entries = Arc::clone(&hit.1);
        leaves.insert(0, hit);
        Some(entries)
    }

    fn remember_leaf(&self, at: u64, entries: Arc<Vec<Entry>>) {
        let mut leaves = self.lock_leaves();
        leaves.retain(|(offset, _)| *offset != at);
        leaves.insert(0, (at, entries));
        leaves.truncate(LEAF_CACHE_ENTRIES);
    }

    /// The cache lock, recovered rather than unwrapped.
    ///
    /// Nothing in this module can panic while holding it, so poisoning would
    /// have to come from somewhere else entirely, and a cache is not worth a
    /// panicking public entry point on an archive somebody handed us.
    fn lock_leaves(&self) -> std::sync::MutexGuard<'_, Vec<(u64, Arc<Vec<Entry>>)>> {
        match self.leaves.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn load_metadata(&self) -> Result<Metadata, PmTilesError> {
        let stored = usize::try_from(self.header.metadata_length).unwrap_or(usize::MAX);
        if stored > MAX_METADATA_BYTES {
            return Err(PmTilesError::DecompressionLimit {
                limit: MAX_METADATA_BYTES,
            });
        }
        let raw = self
            .source
            .read_range(self.header.metadata_offset, stored)?;
        let json = self
            .header
            .internal_compression
            .decompress(&raw, MAX_METADATA_BYTES)?;
        Metadata::try_from_json(&json)
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

    #[test]
    fn the_leaf_cache_keeps_the_most_recent_and_forgets_the_oldest() {
        let entries = [Entry {
            tile_id: 0,
            offset: 0,
            length: 4,
            run_length: 1,
        }];
        let reader = Reader::try_new(Sized(archive(&entries, b"TILE"), true)).expect("opens");

        for offset in 0..(LEAF_CACHE_ENTRIES as u64 + 2) {
            reader.remember_leaf(offset, Arc::new(Vec::new()));
        }
        assert_eq!(reader.lock_leaves().len(), LEAF_CACHE_ENTRIES);
        // The two oldest are gone and the newest is first.
        assert!(reader.cached_leaf(0).is_none());
        assert!(reader.cached_leaf(1).is_none());
        assert!(reader.cached_leaf(LEAF_CACHE_ENTRIES as u64 + 1).is_some());
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

        for offset in 0..LEAF_CACHE_ENTRIES as u64 {
            reader.remember_leaf(offset, Arc::new(Vec::new()));
        }
        // Touch the oldest, then push one more in. Without the move-to-front
        // the touched one would be the one evicted.
        assert!(reader.cached_leaf(0).is_some());
        reader.remember_leaf(99, Arc::new(Vec::new()));
        assert!(reader.cached_leaf(0).is_some());
        assert!(reader.cached_leaf(1).is_none());
    }
}
