//! A PMTiles read is the index and the tile, never the archive (issue #993).
//!
//! The format's whole claim on the directory backend is that a tile costs a
//! bounded number of small ranged reads whatever the archive weighs, which is
//! what lets one file stand in for a filesystem tree of millions. That claim
//! has two halves, and neither is checked by a round-trip test:
//!
//! * **How much is fetched.** A reader that memory-mapped the file, or read it
//!   into a `Vec` and sliced, returns identical bytes for every tile and
//!   passes every equivalence test in this repository. The difference only
//!   shows up as bytes off the transport, so that is what this counts.
//! * **Where it is fetched from.** Entry offsets are `u64`, and an
//!   implementation that narrowed one to `u32` anywhere would be invisible
//!   until an archive crossed 4 GiB. Real archives that big are minutes of
//!   staging to produce, which is why the one in
//!   `tests/pmtiles_bounded_memory.rs` is `#[ignore]`d.
//!
//! # Fabricated, which is the point
//!
//! This file builds the header and the directories of a **6 GiB** archive by
//! hand, through the crate's own [`Header::encode`] and
//! [`serialize_entries`], and serves them through a [`RangeReader`] that
//! synthesises the tile bytes and counts every request. Nothing is written to
//! disk and the whole file runs in microseconds, so the `u32` boundary is
//! covered on every CI run rather than in an opt-in profile.
//!
//! A fabricated archive can lie in one direction: it could describe an
//! archive that is not really 6 GiB. So the source reports its size, the
//! reader's own bounds checks run against that size (every section is checked
//! at open), and [`the_reads_really_land_past_the_four_gibibyte_line`] asserts
//! a request offset above `u32::MAX` was actually issued.

use std::io;
use std::sync::Mutex;

use libviprs::pmtiles::directory::serialize_entries;
use libviprs::pmtiles::{Compression, Entry, Header, RangeReader, Reader, TileType, zxy_to_tileid};

/// The archive this file describes. Six gibibytes of tile data, which is
/// comfortably past the `u32` ceiling of 4 GiB.
const TILE_DATA_LENGTH: u64 = 6 * 1024 * 1024 * 1024;

/// Where the sections sit. Chosen so the root fits the spec's 16 KiB budget
/// for `header + root` and everything else is laid out the way the writer
/// lays it out: root, metadata, leaves, tiles.
const ROOT_OFFSET: u64 = 127;
const METADATA_OFFSET: u64 = 8_192;
const LEAF_OFFSET: u64 = 12_288;
const TILE_DATA_OFFSET: u64 = 65_536;

const METADATA: &[u8] = br#"{"name":"synthetic","vector_layers":[]}"#;

/// The zoom the fabricated tiles live at.
const ZOOM: u8 = 10;

// ---------------------------------------------------------------------------
// The counting source
// ---------------------------------------------------------------------------

/// One request the reader made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Request {
    offset: u64,
    len: usize,
}

/// A byte source that serves a fabricated archive and remembers what was asked
/// for.
///
/// Segments are the parts that really exist as bytes: the header, the root,
/// the metadata and the leaf. Everything inside the tile data section is
/// synthesised from its offset, so a 6 GiB archive costs a few hundred bytes
/// of `Vec`.
struct Counting {
    segments: Vec<(u64, Vec<u8>)>,
    size: u64,
    requests: Mutex<Vec<Request>>,
}

impl Counting {
    fn fetched_bytes(&self) -> u64 {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|r| r.len as u64)
            .sum()
    }

    fn requests(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn forget(&self) {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// The byte a synthetic tile at `offset` is filled with.
///
/// A function of the offset rather than a constant, so a reader that fetched
/// the right length from the wrong place comes back with the wrong bytes
/// instead of with bytes that happen to match.
fn synthetic_byte(offset: u64) -> u8 {
    (offset.wrapping_mul(0x9E37_79B9) >> 24) as u8
}

impl RangeReader for Counting {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Request { offset, len });

        let end = offset.checked_add(len as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflowed")
        })?;
        if end > self.size {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("{offset}..{end} runs past the {} byte archive", self.size),
            ));
        }

        for (start, bytes) in &self.segments {
            let seg_end = start + bytes.len() as u64;
            if offset >= *start && end <= seg_end {
                let from = (offset - start) as usize;
                return Ok(bytes[from..from + len].to_vec());
            }
        }

        if offset >= TILE_DATA_OFFSET {
            // A synthetic tile. Refuse anything the size of a whole-file slurp
            // rather than allocating it: a reader that asked for one should
            // fail here loudly, not quietly succeed and make the byte counter
            // the only witness.
            if len > 1 << 20 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("a {len} byte read of the tile data section is not a tile"),
                ));
            }
            return Ok((offset..end).map(synthetic_byte).collect());
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{offset}..{end} is not inside any section"),
        ))
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.size))
    }
}

// ---------------------------------------------------------------------------
// Building the archive
// ---------------------------------------------------------------------------

/// A tile the fabricated archive holds: where it is addressed and where its
/// bytes are.
#[derive(Debug, Clone, Copy)]
struct Placed {
    x: u32,
    y: u32,
    tile_id: u64,
    /// Relative to the tile data section, which is what an entry carries.
    offset: u64,
    length: u32,
}

impl Placed {
    fn at(x: u32, y: u32, offset: u64, length: u32) -> Self {
        Self {
            x,
            y,
            tile_id: zxy_to_tileid(ZOOM, x, y).expect("inside the zoom's grid"),
            offset,
            length,
        }
    }

    fn absolute(&self) -> u64 {
        TILE_DATA_OFFSET + self.offset
    }

    fn expected(&self) -> Vec<u8> {
        (self.absolute()..self.absolute() + u64::from(self.length))
            .map(synthetic_byte)
            .collect()
    }
}

/// The fabricated archive: two tiles addressed straight from the root and two
/// behind a leaf directory, every one of them past the 4 GiB line.
struct Fabricated {
    source: Counting,
    from_root: [Placed; 2],
    from_leaf: [Placed; 2],
    /// A coordinate the archive deliberately does not hold.
    missing: (u32, u32),
}

fn fabricate() -> Fabricated {
    let four_gib = u64::from(u32::MAX) + 1;

    // Four tiles, every one of them at an offset a `u32` cannot hold, at four
    // offsets spread across the section so a reader that computed the wrong
    // one comes back with the wrong synthetic bytes.
    let mut placed = [
        Placed::at(1, 1, four_gib + 4_096, 512),
        Placed::at(2, 2, four_gib + 1_048_576, 700),
        Placed::at(300, 300, TILE_DATA_LENGTH - 100_000, 1_024),
        Placed::at(301, 301, TILE_DATA_LENGTH - 50_000, 256),
    ];

    // Which two end up in the root and which two behind the leaf is decided by
    // the tile ids, not by the coordinates. PMTiles orders a zoom by its
    // Hilbert curve, so `(300, 300)` is not necessarily above `(1, 1)`, and a
    // directory whose entries do not strictly ascend is a file the format
    // cannot express.
    placed.sort_by_key(|t| t.tile_id);
    let root_tiles = [placed[0], placed[1]];
    let leaf_tiles = [placed[2], placed[3]];
    assert!(
        root_tiles[1].tile_id < leaf_tiles[0].tile_id,
        "the four tile ids should be distinct and ordered"
    );

    let leaf_entries: Vec<Entry> = leaf_tiles
        .iter()
        .map(|t| Entry {
            tile_id: t.tile_id,
            offset: t.offset,
            length: t.length,
            run_length: 1,
        })
        .collect();
    let leaf_bytes = Compression::None
        .compress(&serialize_entries(&leaf_entries).expect("the leaf serialises"))
        .expect("no compression is the identity");

    let mut root_entries: Vec<Entry> = root_tiles
        .iter()
        .map(|t| Entry {
            tile_id: t.tile_id,
            offset: t.offset,
            length: t.length,
            run_length: 1,
        })
        .collect();
    root_entries.push(Entry {
        tile_id: leaf_tiles[0].tile_id,
        offset: 0,
        length: u32::try_from(leaf_bytes.len()).expect("a leaf this small fits a u32"),
        // Zero is what makes an entry a leaf pointer.
        run_length: 0,
    });
    let root_bytes = Compression::None
        .compress(&serialize_entries(&root_entries).expect("the root serialises"))
        .expect("no compression is the identity");

    let header = Header {
        root_offset: ROOT_OFFSET,
        root_length: root_bytes.len() as u64,
        metadata_offset: METADATA_OFFSET,
        metadata_length: METADATA.len() as u64,
        leaf_directories_offset: LEAF_OFFSET,
        leaf_directories_length: leaf_bytes.len() as u64,
        tile_data_offset: TILE_DATA_OFFSET,
        tile_data_length: TILE_DATA_LENGTH,
        addressed_tiles_count: 4,
        tile_entries_count: 4,
        tile_contents_count: 4,
        clustered: true,
        internal_compression: Compression::None,
        tile_compression: Compression::None,
        tile_type: TileType::Png,
        min_zoom: ZOOM,
        max_zoom: ZOOM,
        ..Header::default()
    };

    let source = Counting {
        segments: vec![
            (0, header.encode().to_vec()),
            (ROOT_OFFSET, root_bytes),
            (METADATA_OFFSET, METADATA.to_vec()),
            (LEAF_OFFSET, leaf_bytes),
        ],
        size: TILE_DATA_OFFSET + TILE_DATA_LENGTH,
        requests: Mutex::new(Vec::new()),
    };

    // A coordinate the archive deliberately does not hold. Asserted rather
    // than assumed: the Hilbert ordering means a "clearly different" pair of
    // coordinates is not obviously a different tile id.
    let missing = (900u32, 900u32);
    let missing_id = zxy_to_tileid(ZOOM, missing.0, missing.1).expect("inside the zoom's grid");
    assert!(
        placed.iter().all(|t| t.tile_id != missing_id),
        "the coordinate chosen as a miss is one of the four the archive holds"
    );

    Fabricated {
        source,
        from_root: root_tiles,
        from_leaf: leaf_tiles,
        missing,
    }
}

// ---------------------------------------------------------------------------
// The assertions
// ---------------------------------------------------------------------------

/// Opening a 6 GiB archive reads the header and the root and stops.
#[test]
fn opening_an_archive_reads_the_header_and_the_root_and_nothing_else() {
    let fabricated = fabricate();
    let archive_size = fabricated.source.size;
    let reader = Reader::try_new(fabricated.source).expect("the fabricated archive opens");

    let fetched = reader.source().fetched_bytes();
    assert!(
        fetched < 16_384,
        "opening fetched {fetched} bytes, which is past the spec's 16 KiB header-plus-root budget"
    );
    assert!(
        archive_size / fetched > 100_000,
        "opening a {archive_size} byte archive fetched {fetched} bytes, a ratio of {}",
        archive_size / fetched
    );
    assert_eq!(reader.archive_size(), Some(archive_size));
    assert_eq!(reader.root_entries().len(), 3);
}

/// A tile addressed straight from the root costs one read of exactly its own
/// length, at an offset past 4 GiB.
#[test]
fn a_root_addressed_tile_costs_one_read_of_its_own_length() {
    let fabricated = fabricate();
    let expected: Vec<(Vec<u8>, u64, usize)> = fabricated
        .from_root
        .iter()
        .map(|t| (t.expected(), t.absolute(), t.length as usize))
        .collect();
    let reader = Reader::try_new(fabricated.source).expect("the fabricated archive opens");

    for (tile, (bytes, absolute, length)) in fabricated.from_root.iter().zip(expected) {
        reader.source().forget();
        let got = reader
            .get_tile(ZOOM, tile.x, tile.y)
            .expect("the lookup succeeds")
            .expect("the archive holds this tile");

        assert_eq!(got, bytes, "the tile came back from the wrong offset");
        assert_eq!(
            reader.source().requests(),
            vec![Request {
                offset: absolute,
                len: length
            }],
            "a root-addressed tile should be exactly one read"
        );
    }
}

/// A tile behind a leaf pointer costs the leaf and the tile, and the leaf only
/// once.
#[test]
fn a_leaf_addressed_tile_costs_the_leaf_once_and_then_only_the_tile() {
    let fabricated = fabricate();
    let first = fabricated.from_leaf[0];
    let second = fabricated.from_leaf[1];
    let reader = Reader::try_new(fabricated.source).expect("the fabricated archive opens");

    reader.source().forget();
    let got = reader
        .get_tile(ZOOM, first.x, first.y)
        .expect("the lookup succeeds")
        .expect("the archive holds this tile");
    assert_eq!(got, first.expected());
    let cold = reader.source().requests();
    assert_eq!(
        cold.len(),
        2,
        "a cold leaf lookup should be the leaf and the tile, got {cold:?}"
    );
    assert_eq!(cold[0].offset, LEAF_OFFSET, "the leaf comes first");
    assert_eq!(cold[1].offset, first.absolute());

    reader.source().forget();
    let got = reader
        .get_tile(ZOOM, second.x, second.y)
        .expect("the lookup succeeds")
        .expect("the archive holds this tile");
    assert_eq!(got, second.expected());
    assert_eq!(
        reader.source().requests(),
        vec![Request {
            offset: second.absolute(),
            len: second.length as usize
        }],
        "the second lookup into a cached leaf should be the tile alone"
    );
}

/// The requests really cross the line this file exists for.
///
/// Without this, every assertion above would pass on an archive whose offsets
/// all fit a `u32`, and the fabrication would be proving nothing about the
/// arithmetic it was built to exercise.
#[test]
fn the_reads_really_land_past_the_four_gibibyte_line() {
    let fabricated = fabricate();
    let tile = fabricated.from_root[0];
    let reader = Reader::try_new(fabricated.source).expect("the fabricated archive opens");
    reader.source().forget();
    reader
        .get_tile(ZOOM, tile.x, tile.y)
        .expect("the lookup succeeds")
        .expect("the archive holds this tile");

    let requests = reader.source().requests();
    assert!(
        requests.iter().any(|r| r.offset > u64::from(u32::MAX)),
        "no request landed past 4 GiB, so this fabrication proves nothing: {requests:?}"
    );
    for tile in fabricated.from_leaf {
        assert!(
            tile.absolute() > u64::from(u32::MAX),
            "the leaf tiles should also be past 4 GiB"
        );
    }
}

/// Every lookup this archive supports, and the metadata is still untouched.
///
/// The metadata section is a real segment in the fabricated source, so a
/// reader that fetched it eagerly would succeed and only this counts it. It is
/// the cheapest available proxy for "nothing is read that the lookup did not
/// need".
#[test]
fn a_whole_workload_never_fetches_the_metadata() {
    let fabricated = fabricate();
    let archive_size = fabricated.source.size;
    let coords: Vec<(u32, u32)> = fabricated
        .from_root
        .iter()
        .chain(fabricated.from_leaf.iter())
        .map(|t| (t.x, t.y))
        .collect();
    let missing = fabricated.missing;
    let reader = Reader::try_new(fabricated.source).expect("the fabricated archive opens");

    for _ in 0..8 {
        for (x, y) in &coords {
            reader
                .get_tile(ZOOM, *x, *y)
                .expect("the lookup succeeds")
                .expect("the archive holds this tile");
        }
    }
    assert!(
        reader
            .get_tile(ZOOM, missing.0, missing.1)
            .expect("a miss is not an error")
            .is_none(),
        "the archive should not hold the coordinate chosen as a miss"
    );

    let requests = reader.source().requests();
    assert!(
        !requests.iter().any(|r| r.offset == METADATA_OFFSET),
        "the metadata was fetched by a workload that never asked for it: {requests:?}"
    );

    let fetched = reader.source().fetched_bytes();
    assert!(
        fetched < 64 * 1024,
        "33 lookups over a {archive_size} byte archive fetched {fetched} bytes"
    );
    assert!(
        archive_size / fetched > 100_000,
        "the whole workload read 1 byte for every {} in the archive, which is not index-only",
        archive_size / fetched
    );
}

// ---------------------------------------------------------------------------
// The leaf cache, counted rather than timed
// ---------------------------------------------------------------------------

/// How many leaf directories the archive below has.
///
/// Eight, which is over the four the leaf cache held before issue #993 and
/// inside the sixteen it holds now. Both halves of that matter: at four this
/// test fails, and at more than `LEAF_CACHE_ENTRIES` it would be asserting
/// something no cache promises.
const LEAVES: usize = 8;

/// Tiles inside each of them.
const TILES_PER_LEAF: usize = 4;

/// An archive whose root holds nothing but leaf pointers.
fn fabricate_leafy() -> (Counting, Vec<Vec<Placed>>) {
    let four_gib = u64::from(u32::MAX) + 1;
    let mut placed: Vec<Placed> = (0..(LEAVES * TILES_PER_LEAF) as u32)
        .map(|i| {
            Placed::at(
                i,
                (i * 3) % 1024,
                four_gib + u64::from(i) * 4_096,
                64 + i % 7,
            )
        })
        .collect();
    placed.sort_by_key(|t| t.tile_id);

    let groups: Vec<Vec<Placed>> = placed
        .chunks(TILES_PER_LEAF)
        .map(<[Placed]>::to_vec)
        .collect();
    assert_eq!(groups.len(), LEAVES);

    let mut leaf_section: Vec<u8> = Vec::new();
    let mut pointers: Vec<Entry> = Vec::new();
    for group in &groups {
        let entries: Vec<Entry> = group
            .iter()
            .map(|t| Entry {
                tile_id: t.tile_id,
                offset: t.offset,
                length: t.length,
                run_length: 1,
            })
            .collect();
        let body = Compression::None
            .compress(&serialize_entries(&entries).expect("a leaf serialises"))
            .expect("no compression is the identity");
        pointers.push(Entry {
            tile_id: group[0].tile_id,
            offset: leaf_section.len() as u64,
            length: u32::try_from(body.len()).expect("a leaf this small fits a u32"),
            run_length: 0,
        });
        leaf_section.extend_from_slice(&body);
    }

    let root_bytes = Compression::None
        .compress(&serialize_entries(&pointers).expect("the root serialises"))
        .expect("no compression is the identity");

    let header = Header {
        root_offset: ROOT_OFFSET,
        root_length: root_bytes.len() as u64,
        metadata_offset: METADATA_OFFSET,
        metadata_length: METADATA.len() as u64,
        leaf_directories_offset: LEAF_OFFSET,
        leaf_directories_length: leaf_section.len() as u64,
        tile_data_offset: TILE_DATA_OFFSET,
        tile_data_length: TILE_DATA_LENGTH,
        addressed_tiles_count: (LEAVES * TILES_PER_LEAF) as u64,
        tile_entries_count: (LEAVES * TILES_PER_LEAF) as u64,
        tile_contents_count: (LEAVES * TILES_PER_LEAF) as u64,
        clustered: true,
        internal_compression: Compression::None,
        tile_compression: Compression::None,
        tile_type: TileType::Png,
        min_zoom: ZOOM,
        max_zoom: ZOOM,
        ..Header::default()
    };

    let source = Counting {
        segments: vec![
            (0, header.encode().to_vec()),
            (ROOT_OFFSET, root_bytes),
            (METADATA_OFFSET, METADATA.to_vec()),
            (LEAF_OFFSET, leaf_section),
        ],
        size: TILE_DATA_OFFSET + TILE_DATA_LENGTH,
        requests: Mutex::new(Vec::new()),
    };
    (source, groups)
}

/// Every leaf of an eight-leaf archive stays cached, so a second pass over all
/// of them reads tiles and no directories.
///
/// This is the regression guard on the leaf cache size (issue #993). The cache
/// held four leaves, which is right for a clustered walk and wrong for random
/// access: on a real 21851-tile archive with six leaves, 20000 random lookups
/// cost 1699 ms against 127 ms for the same 20000 walked in order, because a
/// third of them missed and paid a ranged read plus a gzip inflate of a
/// 4096-entry directory. The number is sixteen now.
///
/// It counts reads rather than timing them, so it says the same thing on a
/// loaded machine as on an idle one, and it fails for the one reason it is
/// about rather than for load. A timing assertion here would be the sampling
/// mistake dressed up as a benchmark.
#[test]
fn every_leaf_of_a_multi_leaf_archive_stays_cached() {
    use libviprs::pmtiles::reader::LEAF_CACHE_ENTRIES;

    assert!(
        LEAVES > 4,
        "this archive has to have more leaves than the cache used to hold, or it proves nothing"
    );
    assert!(
        LEAVES <= LEAF_CACHE_ENTRIES,
        "the cache holds {LEAF_CACHE_ENTRIES} leaves and this archive has {LEAVES}, so a miss \
         on the second pass would be the cache working as designed"
    );

    let (source, groups) = fabricate_leafy();
    let leaf_span = source.size;
    let reader = Reader::try_new(source).expect("the fabricated archive opens");
    assert_eq!(
        reader.root_entries().len(),
        LEAVES,
        "the root should hold one pointer per leaf and no tile entries"
    );
    let _ = leaf_span;

    // First pass: one tile out of every leaf, so every leaf is decoded once.
    reader.source().forget();
    for group in &groups {
        let tile = group[0];
        let got = reader
            .get_tile(ZOOM, tile.x, tile.y)
            .expect("the lookup succeeds")
            .expect("the archive holds this tile");
        assert_eq!(got, tile.expected());
    }
    let warming = reader.source().requests();
    let leaf_reads = warming
        .iter()
        .filter(|r| r.offset >= LEAF_OFFSET && r.offset < TILE_DATA_OFFSET)
        .count();
    assert_eq!(
        leaf_reads, LEAVES,
        "the first pass should decode each leaf exactly once, got {warming:?}"
    );

    // Second pass: a different tile from each leaf, walked backwards, which is
    // the order an LRU that is one short would evict in.
    reader.source().forget();
    for group in groups.iter().rev() {
        let tile = group[TILES_PER_LEAF - 1];
        let got = reader
            .get_tile(ZOOM, tile.x, tile.y)
            .expect("the lookup succeeds")
            .expect("the archive holds this tile");
        assert_eq!(got, tile.expected());
    }
    let second = reader.source().requests();
    assert_eq!(
        second.len(),
        LEAVES,
        "the second pass should be one read per tile and nothing else, got {second:?}"
    );
    assert!(
        second
            .iter()
            .all(|r| r.offset >= TILE_DATA_OFFSET + u64::from(u32::MAX)),
        "every read in the second pass should be a tile past 4 GiB, got {second:?}"
    );
}
