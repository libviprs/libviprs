//! The PMTiles indexed reader, pinned against go-pmtiles (issue #988).
//!
//! # Why none of this round-trips against our own writer
//!
//! A reader and a writer that share a misreading of the specification
//! round-trip perfectly and both look correct. The reader is therefore pinned
//! against `tests/fixtures/pmtiles/`, which holds three archives every byte of
//! which was written by `go-pmtiles` v1.31.2, and five JSON files of reference
//! values that program produced. Nothing in this file compares our reader
//! against our writer, and nothing in it is a number somebody transcribed out
//! of our own decoder.
//!
//! The vectors are **loaded by path at run time**, not pasted in as literals,
//! because a transcribed constant is indistinguishable from an invented one.
//! Each file's sha256 is checked before it is parsed and the number of rows
//! parsed out of it is checked against a pinned count, so a parse that quietly
//! yields nothing fails loudly instead of passing every assertion made over
//! the empty set.
//!
//! # The five things a reader gets wrong, and where each one is caught
//!
//! * **The base an offset inside a leaf is relative to.** A tile entry found
//!   in a leaf is relative to `tile_data_offset`, not to the leaf's own start
//!   and not to `leaf_directory_offset`. `leaves-z0z7.pmtiles` is the only
//!   fixture anywhere that exercises it: entry offsets 0 and 72 must land at
//!   absolute 725 and 797. Caught by
//!   [`the_leaf_path_runs_and_every_entry_behind_it_resolves`], which asserts
//!   the exact set of byte ranges the reader fetched.
//!   `all_twenty_one_thousand_entries_behind_the_leaves_come_back_correct`
//!   is the same pin at full width.
//! * **Absence versus a run.** The lookup finds the largest entry whose tile
//!   id is at most the one asked for, so a miss lands on the preceding entry
//!   rather than finding nothing. Only `T < tile_id + run_length` separates a
//!   hit from a miss, and a reader that skips it returns a real, decodable,
//!   wrong tile for every hole in the archive. Caught by
//!   [`a_coordinate_the_archive_does_not_hold_is_absent_rather_than_the_previous_tile`].
//! * **Both shapes of deduplication.** Identical tiles at consecutive ids
//!   become one entry with a run length above 1; identical tiles that are not
//!   consecutive become separate entries pointing at the same offset. Caught
//!   by [`both_shapes_of_deduplication_resolve_and_both_are_exercised`], which
//!   counts how many of each it actually resolved rather than assuming the
//!   fixture reached them.
//! * **Bounds.** The reference implementation checks `length > fileSize` and
//!   never `offset + length`, so it walks past a root offset of 999999 in an
//!   1878-byte file and then nil-dereferences. Caught by
//!   [`a_root_offset_past_the_end_is_refused_where_the_reference_nil_dereferences`].
//! * **Reading more than it has to.** Caught by
//!   [`a_single_tile_read_touches_the_header_the_root_and_one_payload_range`],
//!   through a [`RangeReader`] that logs every range it is asked for.
//!
//! # The instrumented reader is an execution counter, not only an assertion
//!
//! A suite driven by the two leafless goldens never runs the leaf traversal at
//! all and goes green anyway. So the tests that care about a code path assert
//! that the path executed: a read landing inside
//! `[leaf_directory_offset, +leaf_directory_length)` for the leaf tests, and a
//! non-zero count of each dedupe shape for `dupes-z0z3`. A guard that still
//! passes when the thing it inspects disappears is not a guard.
//!
//! # Where a fixture is built here rather than taken from the oracle
//!
//! The hardening tests need archives go-pmtiles will not write: a root that is
//! not gzip, a metadata section that expands without bound, a leaf chain five
//! deep. Those are built in this file out of `Header::encode` and
//! `serialize_entries`, and they are only ever used to check what the reader
//! *refuses*. No assertion about what the format *means* rests on one.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use libviprs::pmtiles::directory::serialize_entries;
use libviprs::pmtiles::reader::{
    MAX_DIRECTORY_BYTES, MAX_LEAF_DEPTH, MAX_METADATA_BYTES, MAX_ROOT_SPAN, Reader,
};
use libviprs::pmtiles::{
    Compression, Entry, FileRangeReader, Header, LibviprsMetadata, Metadata, PmTilesError,
    RangeReader, TileType, tileid_to_zxy, zxy_to_tileid,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// The fixtures, and the pins that say they are the ones the oracle produced
// ---------------------------------------------------------------------------

/// `(file name, sha256)` for every fixture this file reads.
///
/// The archive hashes are the ones `tests/fixtures/pmtiles/PROVENANCE.md`
/// records; the vector hashes are in `vectors/PROVENANCE.md`. A fixture
/// regenerated from a different go-pmtiles release fails here rather than
/// somewhere subtle.
const RASTER: &str = "raster-z0z2.pmtiles";
const DUPES: &str = "dupes-z0z3.pmtiles";
const LEAVES: &str = "leaves-z0z7.pmtiles";

const FIXTURE_SHA256: &[(&str, &str)] = &[
    (
        RASTER,
        "e2ed5e64f3c29efa3ec3b679ec5f1b06569c1b234c6eea762fb9f02fc23e9c12",
    ),
    (
        DUPES,
        "bfc9db4c6ce6a04194e02b3d4815814adb05209f1aaba8591e4e1332f6e56a27",
    ),
    (
        LEAVES,
        "fe5c9636be61abc60046d7f13837f8a3efb20ce3c38303644dac0cbec8248b8d",
    ),
    (
        "vectors/header.json",
        "99258d11ea1fa9cd99c8b28a74ea1bf217e0dea87b4ee00776a6b0c1ea36f1c3",
    ),
    (
        "vectors/directory.json",
        "9f01472702fd4e93c3025bd9897cc336429a1bc05385f35ae555f238490e4d47",
    ),
    (
        "vectors/directory-leaves.json",
        "acc033de338def6a600a806a03cf803cacfe87470a3a8a059f0bace3b9320d58",
    ),
    (
        "vectors/tiles.json",
        "efaebeee9399d9e1e6e0395059caf38c659f134353442bfe29fd0065e5ae6581",
    ),
    (
        "vectors/tileid.json",
        "a486b48b09ab1b9d8f20208b992fc47b89ba67c235506f5f47cccd808e82b265",
    ),
];

/// `tests/fixtures/pmtiles`.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pmtiles")
}

fn fixture_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The sha256 this file pins for `name`.
fn pinned_sha256(name: &str) -> &'static str {
    FIXTURE_SHA256
        .iter()
        .find(|(fixture, _)| *fixture == name)
        .map(|(_, sha)| *sha)
        .unwrap_or_else(|| panic!("{name} has no pinned sha256 in FIXTURE_SHA256"))
}

/// Read a fixture and check it is the file the oracle produced.
fn fixture_bytes(name: &str) -> Vec<u8> {
    let path = fixture_path(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert_eq!(
        sha256_hex(&bytes),
        pinned_sha256(name),
        "{name} is not the fixture this test was written against"
    );
    bytes
}

/// Parse a reference vector file, after checking its hash.
fn vectors(name: &str) -> Value {
    let bytes = fixture_bytes(&format!("vectors/{name}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing vectors/{name}: {e}"))
}

/// A JSON number as a `u64`, with the path named in the failure.
fn u64_at(value: &Value, key: &str) -> u64 {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not an unsigned integer: {}", value[key]))
}

fn str_at(value: &Value, key: &str) -> String {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string: {}", value[key]))
        .to_string()
}

fn rows<'a>(value: &'a Value, key: &str) -> &'a Vec<Value> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is not an array"))
}

// ---------------------------------------------------------------------------
// A RangeReader that records what it was asked for
// ---------------------------------------------------------------------------

/// Wraps any [`RangeReader`] and logs every `(offset, len)` it serves.
///
/// This is the execution counter the acceptance criteria ask for. It answers
/// two different questions with the same log: "did the reader stay off the
/// rest of the archive" and "did the code path I think I am testing actually
/// run".
struct CountingReader<R: RangeReader> {
    inner: R,
    log: Mutex<Vec<(u64, usize)>>,
}

impl<R: RangeReader> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    /// Every range served so far, in order.
    fn reads(&self) -> Vec<(u64, usize)> {
        self.log.lock().expect("read log is not poisoned").clone()
    }

    fn clear(&self) {
        self.log.lock().expect("read log is not poisoned").clear();
    }

    /// Total bytes handed out, which is what "never reads the whole archive"
    /// is really a claim about.
    fn bytes_read(&self) -> usize {
        self.reads().iter().map(|(_, len)| len).sum()
    }

    /// The ranges served that fall inside `[start, start + length)`.
    fn reads_within(&self, start: u64, length: u64) -> Vec<(u64, usize)> {
        self.reads()
            .into_iter()
            .filter(|(offset, _)| *offset >= start && *offset < start + length)
            .collect()
    }
}

impl<R: RangeReader> RangeReader for CountingReader<R> {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.log
            .lock()
            .expect("read log is not poisoned")
            .push((offset, len));
        self.inner.read_range(offset, len)
    }

    fn size(&self) -> io::Result<Option<u64>> {
        self.inner.size()
    }
}

/// A [`RangeReader`] over bytes already in memory, for the archives this file
/// builds rather than reads.
struct InMemory(Vec<u8>);

impl RangeReader for InMemory {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset past usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflows"))?;
        if end > self.0.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "range past the end of the archive",
            ));
        }
        Ok(self.0[start..end].to_vec())
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.0.len() as u64))
    }
}

/// Open a golden through a logging reader, over the real file.
fn counted(name: &str) -> Reader<CountingReader<FileRangeReader>> {
    // Checking the hash here is what makes every assertion downstream a
    // statement about the archive the oracle produced.
    let _ = fixture_bytes(name);
    let source = FileRangeReader::try_open(fixture_path(name))
        .unwrap_or_else(|e| panic!("opening {name}: {e}"));
    Reader::try_new(CountingReader::new(source))
        .unwrap_or_else(|e| panic!("reading the header of {name}: {e}"))
}

/// Open bytes we built here.
fn memory_reader(bytes: Vec<u8>) -> Result<Reader<InMemory>, PmTilesError> {
    Reader::try_new(InMemory(bytes))
}

// ---------------------------------------------------------------------------
// The fixtures are what they claim to be
// ---------------------------------------------------------------------------

/// Row counts pinned so a parse that silently yields nothing cannot pass.
///
/// An empty result has two explanations and only one of them is the code being
/// right, so every vector file this suite reads is checked for the number of
/// rows it is supposed to carry before anything is asserted over those rows.
#[test]
#[cfg_attr(miri, ignore)]
fn the_vector_files_are_the_ones_the_oracle_produced_and_they_parse_to_the_rows_they_claim() {
    // Hashes first: `fixture_bytes` checks each one, so reading all eight is
    // the check.
    for (name, _) in FIXTURE_SHA256 {
        let bytes = fixture_bytes(name);
        assert!(!bytes.is_empty(), "{name} is empty");
    }

    // `tileid.json` carries its own count block, so it is checked against
    // itself as well as against this file.
    let tileid = vectors("tileid.json");
    let counts = &tileid["counts"];
    assert_eq!(
        rows(&tileid, "convention_discriminators").len() as u64,
        u64_at(counts, "convention_discriminators"),
        "the discriminator rows parsed do not match the file's own count"
    );
    assert_eq!(
        u64_at(counts, "convention_discriminators"),
        12,
        "the twelve rows that tell the Hilbert conventions apart"
    );

    let tiles = vectors("tiles.json");
    let expected: &[(&str, usize, usize, usize)] =
        &[(RASTER, 11, 2, 3), (DUPES, 8, 1, 1), (LEAVES, 6, 1, 1)];
    for (archive, hits, absent, out_of_range) in expected {
        let a = &tiles["archives"][archive];
        assert_eq!(rows(a, "hits").len(), *hits, "{archive} hits");
        assert_eq!(rows(a, "absent").len(), *absent, "{archive} absent");
        assert_eq!(
            rows(a, "out_of_range").len(),
            *out_of_range,
            "{archive} out_of_range"
        );
    }

    let directory = vectors("directory.json");
    assert_eq!(
        rows(&directory["archives"][RASTER]["root_directory"], "entries").len(),
        21
    );
    assert_eq!(
        rows(&directory["archives"][DUPES]["root_directory"], "entries").len(),
        67
    );

    let leaves = vectors("directory-leaves.json");
    assert_eq!(rows(&leaves["root_directory"], "entries").len(), 6);
    let leaf_entries: usize = rows(&leaves, "leaf_directories")
        .iter()
        .map(|leaf| rows(&leaf["entries_columnar"], "tile_id").len())
        .sum();
    assert_eq!(
        leaf_entries, 21844,
        "the 21844 entries behind the six leaves"
    );

    let header = vectors("header.json");
    assert_eq!(
        header["archives"]
            .as_object()
            .expect("archives is an object")
            .len(),
        3
    );
}

// ---------------------------------------------------------------------------
// Header, and the accessors that read out of it
// ---------------------------------------------------------------------------

/// Every header field, and the four accessors on top of them, against what
/// `pmtiles.DeserializeHeader` reports for the same bytes.
#[test]
#[cfg_attr(miri, ignore)]
fn the_header_decodes_to_what_go_pmtiles_reports_for_every_golden() {
    let vectors_json = vectors("header.json");
    for name in [RASTER, DUPES, LEAVES] {
        let want = &vectors_json["archives"][name]["decoded_by_pmtiles_DeserializeHeader"];
        let reader = counted(name);
        let got: &Header = reader.header();

        assert_eq!(got.root_offset, u64_at(want, "root_offset"), "{name}");
        assert_eq!(got.root_length, u64_at(want, "root_length"), "{name}");
        assert_eq!(
            got.metadata_offset,
            u64_at(want, "metadata_offset"),
            "{name}"
        );
        assert_eq!(
            got.metadata_length,
            u64_at(want, "metadata_length"),
            "{name}"
        );
        assert_eq!(
            got.leaf_directories_offset,
            u64_at(want, "leaf_directory_offset"),
            "{name}"
        );
        assert_eq!(
            got.leaf_directories_length,
            u64_at(want, "leaf_directory_length"),
            "{name}"
        );
        assert_eq!(
            got.tile_data_offset,
            u64_at(want, "tile_data_offset"),
            "{name}"
        );
        assert_eq!(
            got.tile_data_length,
            u64_at(want, "tile_data_length"),
            "{name}"
        );
        assert_eq!(
            got.addressed_tiles_count,
            u64_at(want, "addressed_tiles_count"),
            "{name}"
        );
        assert_eq!(
            got.tile_entries_count,
            u64_at(want, "tile_entries_count"),
            "{name}"
        );
        assert_eq!(
            got.tile_contents_count,
            u64_at(want, "tile_contents_count"),
            "{name}"
        );
        assert_eq!(
            got.clustered,
            want["clustered"].as_bool().expect("clustered is a bool"),
            "{name}"
        );
        assert_eq!(
            got.internal_compression.to_byte() as u64,
            u64_at(want, "internal_compression"),
            "{name}"
        );
        assert_eq!(
            got.tile_compression.to_byte() as u64,
            u64_at(want, "tile_compression"),
            "{name}"
        );

        assert_eq!(u64::from(reader.min_zoom()), u64_at(want, "min_zoom"));
        assert_eq!(u64::from(reader.max_zoom()), u64_at(want, "max_zoom"));
        assert_eq!(
            u64::from(reader.tile_format().to_byte()),
            u64_at(want, "tile_type")
        );
        assert_eq!(reader.tile_format(), TileType::Png, "{name}");

        let (min_lon, min_lat, max_lon, max_lat) = reader.bounds();
        let e7 = |degrees: f64| (degrees * 1e7).round() as i64;
        assert_eq!(e7(min_lon), want["min_lon_e7"].as_i64().expect("i64"));
        assert_eq!(e7(min_lat), want["min_lat_e7"].as_i64().expect("i64"));
        assert_eq!(e7(max_lon), want["max_lon_e7"].as_i64().expect("i64"));
        assert_eq!(e7(max_lat), want["max_lat_e7"].as_i64().expect("i64"));
    }
}

/// Opening an archive reads the header and the root directory and nothing
/// else. Metadata in particular is lazy: it is the one section a reader can
/// skip entirely, and on a vector archive it can be megabytes.
#[test]
#[cfg_attr(miri, ignore)]
fn opening_an_archive_reads_the_header_and_the_root_and_stops() {
    let reader = counted(RASTER);
    let header = *reader.header();
    let reads = reader.source().reads();

    assert_eq!(
        reads.len(),
        2,
        "header then root, and nothing else: {reads:?}"
    );
    assert_eq!(reads[0], (0, 127), "the 127-byte header");
    assert_eq!(
        reads[1],
        (header.root_offset, header.root_length as usize),
        "the root directory, exactly as the header sizes it"
    );

    // The metadata section was not touched.
    assert!(
        reader
            .source()
            .reads_within(header.metadata_offset, header.metadata_length)
            .is_empty(),
        "opening an archive must not read its metadata"
    );
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

/// Every coordinate go-pmtiles serves a payload for comes back byte for byte.
#[test]
#[cfg_attr(miri, ignore)]
fn every_tile_go_pmtiles_serves_comes_back_byte_for_byte() {
    let tiles = vectors("tiles.json");
    let mut checked = 0usize;
    for name in [RASTER, DUPES, LEAVES] {
        let reader = counted(name);
        for hit in rows(&tiles["archives"][name], "hits") {
            let z = u64_at(hit, "z") as u8;
            let x = u64_at(hit, "x") as u32;
            let y = u64_at(hit, "y") as u32;
            let want_len = u64_at(hit, "length") as usize;
            let want_sha = str_at(hit, "sha256");

            let got = reader
                .get_tile(z, x, y)
                .unwrap_or_else(|e| panic!("{name} {z}/{x}/{y}: {e}"))
                .unwrap_or_else(|| panic!("{name} {z}/{x}/{y} came back absent"));

            assert_eq!(got.len(), want_len, "{name} {z}/{x}/{y} length");
            assert_eq!(sha256_hex(&got), want_sha, "{name} {z}/{x}/{y} payload");
            assert!(reader.tile_exists(z, x, y).expect("existence is decidable"));
            checked += 1;
        }
    }
    assert_eq!(checked, 25, "11 + 8 + 6 rows across the three goldens");
}

/// A coordinate the archive does not hold is `Ok(None)`, not the previous
/// tile's bytes.
///
/// The oracle's own `absent` rows are the zoom-below and zoom-above cases. The
/// rest are derived, and they are the ones that matter: a hole *inside* the id
/// space, which is exactly where a reader missing the run-end check returns a
/// real, decodable, wrong tile.
#[test]
#[cfg_attr(miri, ignore)]
fn a_coordinate_the_archive_does_not_hold_is_absent_rather_than_the_previous_tile() {
    let tiles = vectors("tiles.json");
    let mut checked = 0usize;
    for name in [RASTER, DUPES, LEAVES] {
        let reader = counted(name);
        for absent in rows(&tiles["archives"][name], "absent") {
            let z = u64_at(absent, "z") as u8;
            let x = u64_at(absent, "x") as u32;
            let y = u64_at(absent, "y") as u32;
            assert_eq!(
                reader.get_tile(z, x, y).expect("a miss is not an error"),
                None,
                "{name} {z}/{x}/{y}"
            );
            assert!(!reader.tile_exists(z, x, y).expect("existence is decidable"));
            checked += 1;
        }
    }
    assert_eq!(checked, 4, "the oracle's own absent probes");

    // The derived half. `raster-z0z2` holds zooms 0 to 2 and nothing above, so
    // every zoom-3 id is a hole *after* the last entry, and a reader that
    // treats "landed on an entry" as a hit serves the z=2 tile that precedes
    // it. There is no zoom-3 entry at all, so the lookup lands on the last
    // entry of zoom 2 and must then fail the run-end check.
    let reader = counted(RASTER);
    for (z, x, y) in [(3, 0, 0), (3, 7, 7), (3, 4, 2), (4, 0, 0), (5, 31, 31)] {
        assert_eq!(
            reader.get_tile(z, x, y).expect("a miss is not an error"),
            None,
            "{z}/{x}/{y} is past the last entry of the archive"
        );
    }

    // And the positive control for that negative: the tiles either side of the
    // hole are present, so the assertion above is not passing because the
    // reader refuses everything.
    assert!(reader.get_tile(2, 3, 3).expect("a hit").is_some());
    assert!(reader.get_tile(0, 0, 0).expect("a hit").is_some());
}

/// A hole between two runs, built rather than found.
///
/// None of the three goldens has an interior hole: go-pmtiles' `convert`
/// writes complete pyramids. So this builds the archive the spec digest
/// describes, tiles 5 and 9 with 6, 7 and 8 missing, and asks for the gap. A
/// reader without the run-end check returns tile 5's bytes for all three.
#[test]
fn a_hole_between_two_runs_is_absent_and_not_the_run_before_it() {
    let a = b"AAAA".to_vec();
    let b = b"BBBB".to_vec();
    let mut tile_data = a.clone();
    tile_data.extend_from_slice(&b);

    let root = vec![
        Entry {
            tile_id: 5,
            offset: 0,
            length: 4,
            run_length: 1,
        },
        Entry {
            tile_id: 9,
            offset: 4,
            length: 4,
            run_length: 1,
        },
    ];
    let archive = build_archive(&root, &[], b"{}", &tile_data, |_| {});
    let reader = memory_reader(archive).expect("the archive opens");

    let at = |id: u64| {
        let (z, x, y) = tileid_to_zxy(id).expect("an addressable id");
        reader.get_tile(z, x, y).expect("a lookup is not an error")
    };

    assert_eq!(at(5).as_deref(), Some(&a[..]), "tile 5 is present");
    assert_eq!(at(9).as_deref(), Some(&b[..]), "tile 9 is present");
    for id in [6, 7, 8] {
        assert_eq!(at(id), None, "tile {id} is a hole, not tile 5 again");
    }
    // Below the first entry, which is the other absence shape.
    assert_eq!(at(4), None, "tile 4 is below the first entry");
    assert_eq!(at(0), None, "tile 0 is below the first entry");
}

/// A run serves every id it covers, and stops at its end.
#[test]
fn a_run_serves_its_whole_span_and_not_one_id_past_it() {
    let payload = b"RUNRUNRUN".to_vec();
    let root = vec![Entry {
        tile_id: 5,
        offset: 0,
        length: payload.len() as u32,
        run_length: 3,
    }];
    let archive = build_archive(&root, &[], b"{}", &payload, |_| {});
    let reader = memory_reader(archive).expect("the archive opens");

    for id in [5, 6, 7] {
        let (z, x, y) = tileid_to_zxy(id).expect("an addressable id");
        assert_eq!(
            reader.get_tile(z, x, y).expect("a lookup").as_deref(),
            Some(&payload[..]),
            "tile {id} is inside the run"
        );
    }
    let (z, x, y) = tileid_to_zxy(8).expect("an addressable id");
    assert_eq!(
        reader.get_tile(z, x, y).expect("a lookup"),
        None,
        "tile 8 is one past the run and must be absent"
    );
}

/// An out-of-grid coordinate is refused, where go-pmtiles masks it into a
/// different valid tile and serves that tile's bytes.
///
/// `vectors/tiles.json`'s `out_of_range` rows are behaviour and not a target:
/// the file says so itself. The assertion is that we refuse, and that we do
/// not return the payload the reference served.
#[test]
#[cfg_attr(miri, ignore)]
fn a_coordinate_outside_its_own_zoom_grid_is_refused_where_go_pmtiles_masks_it() {
    let tiles = vectors("tiles.json");
    let mut checked = 0usize;
    for name in [RASTER, DUPES, LEAVES] {
        let reader = counted(name);
        for row in rows(&tiles["archives"][name], "out_of_range") {
            let z = u64_at(row, "z") as u8;
            let x = u64_at(row, "x") as u32;
            let y = u64_at(row, "y") as u32;
            // go-pmtiles exited 0 and wrote a real payload for every one of
            // these, which is the point.
            assert_eq!(u64_at(row, "exit_code"), 0, "{name} {z}/{x}/{y}");
            assert!(u64_at(row, "length") > 0, "{name} {z}/{x}/{y}");

            let got = reader.get_tile(z, x, y);
            assert!(
                matches!(&got, Err(PmTilesError::CoordOutOfRange { .. })),
                "{name} {z}/{x}/{y} must be refused, got {got:?}"
            );
            assert!(reader.tile_exists(z, x, y).is_err());
            checked += 1;
        }
    }
    assert_eq!(checked, 5, "the oracle's five out-of-range probes");
}

/// A zoom above 31 cannot be addressed by a `u64` tile id and is refused
/// rather than saturated, which is what the reference does.
#[test]
#[cfg_attr(miri, ignore)]
fn a_zoom_the_id_space_cannot_address_is_refused_rather_than_saturated() {
    let reader = counted(RASTER);
    for z in [32u8, 33, 63, 127, 255] {
        let got = reader.get_tile(z, 0, 0);
        assert!(
            matches!(&got, Err(PmTilesError::ZoomOutOfRange { .. })),
            "zoom {z} must be refused, got {got:?}"
        );
    }
    // Positive control: zoom 31 is addressable, so the refusal above is about
    // the ceiling and not about every large zoom.
    assert_eq!(
        reader.get_tile(31, 0, 0).expect("zoom 31 is addressable"),
        None
    );
}

// ---------------------------------------------------------------------------
// The read pattern
// ---------------------------------------------------------------------------

/// One `get_tile` touches the header, the root, and one payload range.
///
/// The acceptance criterion is "demonstrated single-tile read touches only
/// header + directory pages + one payload range", so this counts the ranges
/// rather than timing anything.
#[test]
#[cfg_attr(miri, ignore)]
fn a_single_tile_read_touches_the_header_the_root_and_one_payload_range() {
    let archive_len = fixture_bytes(RASTER).len();
    let reader = counted(RASTER);
    let header = *reader.header();

    // Two reads so far: header and root. Clear them, so what follows is the
    // cost of the tile alone.
    assert_eq!(reader.source().reads().len(), 2);
    reader.source().clear();

    let tile = reader
        .get_tile(2, 3, 3)
        .expect("a lookup")
        .expect("z2 (3,3) is in the archive");

    let reads = reader.source().reads();
    assert_eq!(
        reads.len(),
        1,
        "a leafless archive needs exactly one read for the payload: {reads:?}"
    );
    let (offset, len) = reads[0];
    assert_eq!(
        len,
        tile.len(),
        "the read is sized by the entry, not padded"
    );
    assert!(
        offset >= header.tile_data_offset
            && offset + len as u64 <= header.tile_data_offset + header.tile_data_length,
        "the payload read lands inside the tile data section"
    );

    // The whole archive was never read. 127 + 35 + 74 is 236 of 1878.
    assert!(
        reader.source().bytes_read() < archive_len,
        "the reader must not have read the whole archive"
    );
    assert!(
        reader.source().bytes_read() <= 236,
        "header, root and one 74-byte payload: {} bytes",
        reader.source().bytes_read()
    );
}

/// `tile_exists` answers without fetching the payload.
#[test]
#[cfg_attr(miri, ignore)]
fn tile_exists_answers_without_reading_the_payload() {
    let reader = counted(RASTER);
    let header = *reader.header();
    reader.source().clear();

    assert!(reader.tile_exists(2, 3, 3).expect("existence is decidable"));
    assert!(
        reader
            .source()
            .reads_within(header.tile_data_offset, header.tile_data_length)
            .is_empty(),
        "tile_exists must not fetch the tile: {:?}",
        reader.source().reads()
    );
}

// ---------------------------------------------------------------------------
// Deduplication, both shapes
// ---------------------------------------------------------------------------

/// Every one of the 85 addressed tiles of `dupes-z0z3` resolves, and both
/// shapes of deduplication were actually exercised while doing it.
///
/// The counters are the point. A green sweep over an archive that happened to
/// contain neither shape would prove nothing, so this counts how many lookups
/// were served by a run rather than by their own entry, and how many landed on
/// an offset shared by entries that are not adjacent, and refuses to pass
/// unless both happened.
#[test]
#[cfg_attr(miri, ignore)]
fn both_shapes_of_deduplication_resolve_and_both_are_exercised() {
    let reader = counted(DUPES);
    let header = *reader.header();
    let entries = reader.root_entries().to_vec();

    assert_eq!(
        entries.len() as u64,
        header.tile_entries_count,
        "67 entries"
    );
    let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
    assert_eq!(
        addressed, header.addressed_tiles_count,
        "run lengths sum to the header's addressed count"
    );
    assert_eq!(addressed, 85);

    // Offsets carried by more than one entry are the non-adjacent dedupe
    // shape. `dupes-z0z3` has offset 148 on four entries 21, 49, 63 and 76.
    let shared_offsets: Vec<u64> = {
        let mut offsets: Vec<u64> = entries.iter().map(|e| e.offset).collect();
        offsets.sort_unstable();
        let mut shared = Vec::new();
        for window in offsets.windows(2) {
            if window[0] == window[1] && !shared.contains(&window[0]) {
                shared.push(window[0]);
            }
        }
        shared
    };
    assert!(
        shared_offsets.contains(&148),
        "the fixture's four non-adjacent duplicates share offset 148, found {shared_offsets:?}"
    );
    assert!(
        shared_offsets.contains(&0),
        "the z0 tile and the whole of z2 are the same red PNG at offset 0, found {shared_offsets:?}"
    );

    let mut served_by_a_run = 0usize;
    let mut served_from_a_shared_offset = 0usize;
    let mut resolved = 0usize;

    for entry in &entries {
        for step in 0..u64::from(entry.run_length) {
            let id = entry.tile_id + step;
            let (z, x, y) = tileid_to_zxy(id).expect("an addressable id");

            reader.source().clear();
            let tile = reader
                .get_tile(z, x, y)
                .unwrap_or_else(|e| panic!("tile {id}: {e}"))
                .unwrap_or_else(|| panic!("tile {id} came back absent"));

            let reads = reader.source().reads();
            assert_eq!(reads.len(), 1, "one payload read for tile {id}: {reads:?}");
            assert_eq!(
                reads[0],
                (
                    header.tile_data_offset + entry.offset,
                    entry.length as usize
                ),
                "tile {id} read the range its entry names"
            );
            assert_eq!(tile.len(), entry.length as usize);

            if step > 0 {
                served_by_a_run += 1;
            }
            if shared_offsets.contains(&entry.offset) {
                served_from_a_shared_offset += 1;
            }
            resolved += 1;
        }
    }

    assert_eq!(resolved, 85, "every addressed tile resolved");
    assert!(
        served_by_a_run >= 1,
        "no lookup was served by a run length above 1, so that path never ran"
    );
    assert!(
        served_from_a_shared_offset >= 1,
        "no lookup landed on a shared offset, so that path never ran"
    );
    // The fixture's numbers, so a future regeneration that quietly loses one
    // of the two shapes fails here rather than weakening the test in silence.
    // 18 is the run shape: a run of 4 and a run of 16 each serve every id
    // after their first from the same entry, so 3 + 15.
    assert_eq!(served_by_a_run, 18, "85 addressed tiles across 67 entries");
    // 21 is the shared-offset shape: the four offset-148 tiles, plus the z0
    // tile and the sixteen z2 tiles that are all the same red PNG at offset 0.
    assert_eq!(
        served_from_a_shared_offset, 21,
        "4 at offset 148, 1 + 16 at offset 0"
    );
}

/// The root directory of the two leafless goldens is the entry list
/// go-pmtiles decoded, field for field.
#[test]
#[cfg_attr(miri, ignore)]
fn the_root_directory_is_the_entry_list_go_pmtiles_decoded() {
    let directory = vectors("directory.json");
    for name in [RASTER, DUPES] {
        let reader = counted(name);
        let want = rows(&directory["archives"][name]["root_directory"], "entries");
        let got = reader.root_entries();
        assert_eq!(got.len(), want.len(), "{name} entry count");
        for (index, (entry, row)) in got.iter().zip(want).enumerate() {
            assert_eq!(entry.tile_id, u64_at(row, "tile_id"), "{name}[{index}] id");
            assert_eq!(
                entry.offset,
                u64_at(row, "offset"),
                "{name}[{index}] offset"
            );
            assert_eq!(
                u64::from(entry.length),
                u64_at(row, "length"),
                "{name}[{index}] length"
            );
            assert_eq!(
                u64::from(entry.run_length),
                u64_at(row, "run_length"),
                "{name}[{index}] run length"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Leaves
// ---------------------------------------------------------------------------

/// The leaf path runs, and the ranges it fetched are the ones the oracle
/// recorded.
///
/// This is the leaf offset base pinned as tightly as it can be. Two sets of
/// ranges are asserted exactly:
///
/// * the six leaf reads, at absolute 334, 405, 470, 536, 602 and 667, which is
///   `leaf_directory_offset` plus the pointer's relative offset;
/// * the payload reads, at absolute 725 and 797, which is `tile_data_offset`
///   plus the entry's offset. A reader that used the leaf's own start or
///   `leaf_directory_offset` as the base for a tile entry found inside a leaf
///   produces different numbers here and fails.
#[test]
#[cfg_attr(miri, ignore)]
fn the_leaf_path_runs_and_every_entry_behind_it_resolves() {
    let leaves = vectors("directory-leaves.json");
    let reader = counted(LEAVES);
    let header = *reader.header();

    assert!(header.leaf_directories_length > 0, "the golden has leaves");
    assert_eq!(header.leaf_directories_offset, 334);
    assert_eq!(header.tile_data_offset, 725);

    // The root is six leaf pointers and nothing else.
    let want_root = rows(&leaves["root_directory"], "entries");
    let got_root = reader.root_entries();
    assert_eq!(got_root.len(), 6);
    for (index, (entry, row)) in got_root.iter().zip(want_root).enumerate() {
        assert_eq!(entry.tile_id, u64_at(row, "tile_id"), "root[{index}]");
        assert_eq!(entry.offset, u64_at(row, "offset"), "root[{index}]");
        assert_eq!(
            u64::from(entry.length),
            u64_at(row, "length"),
            "root[{index}]"
        );
        assert_eq!(entry.run_length, 0, "root[{index}] is a leaf pointer");
        assert!(entry.is_leaf(), "root[{index}] is a leaf pointer");
    }

    // One tile out of every leaf, so every leaf is fetched at least once.
    reader.source().clear();
    for row in want_root {
        let id = u64_at(row, "tile_id");
        let (z, x, y) = tileid_to_zxy(id).expect("an addressable id");
        let tile = reader
            .get_tile(z, x, y)
            .unwrap_or_else(|e| panic!("tile {id}: {e}"))
            .unwrap_or_else(|| panic!("tile {id} came back absent"));
        assert_eq!(tile.len(), 72);
    }
    // The first entry of all six leaves happens to carry entry offset 0, so
    // the loop above only ever reaches the first of the two payloads. These
    // are two of the oracle's own hit rows, both recorded at entry offset 72,
    // and without them the exact payload set below would be half a test.
    for (z, x, y) in [(5u8, 10u32, 11u32), (7, 63, 64)] {
        assert!(
            reader.get_tile(z, x, y).expect("a lookup").is_some(),
            "{z}/{x}/{y} is one of the oracle's offset-72 rows"
        );
    }

    let leaf_reads = reader.source().reads_within(
        header.leaf_directories_offset,
        header.leaf_directories_length,
    );
    assert!(
        !leaf_reads.is_empty(),
        "no read landed inside the leaf section, so the leaf path never ran"
    );

    let want_leaf_ranges: Vec<(u64, usize)> = rows(&leaves, "leaf_directories")
        .iter()
        .map(|leaf| {
            (
                u64_at(leaf, "absolute_offset"),
                u64_at(leaf, "compressed_length") as usize,
            )
        })
        .collect();
    assert_eq!(
        want_leaf_ranges,
        vec![
            (334, 71),
            (405, 65),
            (470, 66),
            (536, 66),
            (602, 65),
            (667, 58)
        ],
        "the oracle's six leaf ranges"
    );
    let mut got_leaf_ranges = leaf_reads.clone();
    got_leaf_ranges.sort_unstable();
    got_leaf_ranges.dedup();
    assert_eq!(
        got_leaf_ranges, want_leaf_ranges,
        "the reader fetched exactly the six leaves the oracle recorded"
    );

    // The payloads, which is where the base matters.
    let mut payload_reads = reader
        .source()
        .reads_within(header.tile_data_offset, header.tile_data_length);
    payload_reads.sort_unstable();
    payload_reads.dedup();
    assert_eq!(
        payload_reads,
        vec![(725, 72), (797, 72)],
        "entry offsets 0 and 72 are relative to tile_data_offset, so they land at 725 and 797"
    );
}

/// Every one of the 21844 entries behind the six leaves resolves to the
/// payload the oracle recorded for it.
///
/// The sweep matters because the six pointers only prove the first entry of
/// each leaf. This walks the whole id space of the archive, 0 through 21844,
/// and checks each against the entry columns in `directory-leaves.json`. A
/// binary search that is off by one inside a leaf, or a run that is resolved
/// one id too far, shows up here and nowhere else.
#[test]
#[cfg_attr(miri, ignore)]
fn all_twenty_one_thousand_entries_behind_the_leaves_come_back_correct() {
    let leaves = vectors("directory-leaves.json");
    let reader = counted(LEAVES);
    let header = *reader.header();

    // Expand the oracle's entry columns into id -> (offset, length).
    let mut expected: Vec<Option<(u64, u64)>> = vec![None; 21_845];
    let mut entry_count = 0usize;
    for leaf in rows(&leaves, "leaf_directories") {
        let columns = &leaf["entries_columnar"];
        let ids = rows(columns, "tile_id");
        let offsets = rows(columns, "offset");
        let lengths = rows(columns, "length");
        let runs = rows(columns, "run_length");
        assert_eq!(ids.len(), offsets.len());
        assert_eq!(ids.len(), lengths.len());
        assert_eq!(ids.len(), runs.len());
        for index in 0..ids.len() {
            let id = ids[index].as_u64().expect("a tile id");
            let offset = offsets[index].as_u64().expect("an offset");
            let length = lengths[index].as_u64().expect("a length");
            let run = runs[index].as_u64().expect("a run length");
            assert!(run > 0, "a leaf of this golden holds no nested pointers");
            for step in 0..run {
                let slot = usize::try_from(id + step).expect("an id inside the archive");
                assert!(expected[slot].is_none(), "id {slot} is covered twice");
                expected[slot] = Some((offset, length));
            }
            entry_count += 1;
        }
    }
    assert_eq!(entry_count, 21_844, "the oracle's entry count");
    assert!(
        expected.iter().all(Option::is_some),
        "the archive addresses every id from 0 to 21844"
    );

    // Both payloads, read straight out of the tile data section, so the
    // comparison below is against bytes and not against our own decode.
    let archive = fixture_bytes(LEAVES);
    let payload_at = |offset: u64, length: u64| -> Vec<u8> {
        let start = (header.tile_data_offset + offset) as usize;
        archive[start..start + length as usize].to_vec()
    };

    let mut leaf_reads = 0usize;
    for (id, want) in expected.iter().enumerate() {
        let (offset, length) = want.expect("every id is addressed");
        let (z, x, y) = tileid_to_zxy(id as u64).expect("an addressable id");

        reader.source().clear();
        let tile = reader
            .get_tile(z, x, y)
            .unwrap_or_else(|e| panic!("tile {id} ({z}/{x}/{y}): {e}"))
            .unwrap_or_else(|| panic!("tile {id} ({z}/{x}/{y}) came back absent"));
        assert_eq!(tile, payload_at(offset, length), "tile {id} payload");

        let reads = reader.source().reads();
        let payload_reads: Vec<(u64, usize)> = reads
            .iter()
            .copied()
            .filter(|(at, _)| *at >= header.tile_data_offset)
            .collect();
        assert_eq!(
            payload_reads,
            vec![(header.tile_data_offset + offset, length as usize)],
            "tile {id} read exactly the range its entry names, relative to tile_data_offset"
        );
        leaf_reads += reader
            .source()
            .reads_within(
                header.leaf_directories_offset,
                header.leaf_directories_length,
            )
            .len();
    }

    assert!(
        leaf_reads >= 6,
        "the sweep crossed six leaves and fetched {leaf_reads} of them"
    );
}

/// A leaf fetched once is not fetched again for the next tile in it.
///
/// The cache is a performance feature and it is deliberately off the
/// correctness path, but "the reader re-reads the leaf for every tile" is the
/// kind of regression nothing else in this file would notice.
#[test]
#[cfg_attr(miri, ignore)]
fn a_leaf_read_once_serves_the_next_tile_without_a_second_fetch() {
    let reader = counted(LEAVES);
    let header = *reader.header();

    reader.source().clear();
    // Tile id 0. The first leaf covers ids 0 to 4096.
    assert!(reader.get_tile(0, 0, 0).expect("a lookup").is_some());
    let first = reader
        .source()
        .reads_within(
            header.leaf_directories_offset,
            header.leaf_directories_length,
        )
        .len();
    assert_eq!(first, 1, "the first tile fetched its leaf");

    reader.source().clear();
    // Tile id 1, which is in that same leaf. Zoom 7 is not: its first tile is
    // id 5461, which lives in the second leaf.
    assert_eq!(zxy_to_tileid(1, 0, 0).expect("an addressable id"), 1);
    assert!(reader.get_tile(1, 0, 0).expect("a lookup").is_some());
    let second = reader
        .source()
        .reads_within(
            header.leaf_directories_offset,
            header.leaf_directories_length,
        )
        .len();
    assert_eq!(second, 0, "the second tile reused the cached leaf");
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// The goldens' metadata parses, and every key in it survives.
///
/// All three are **foreign** metadata objects: go-pmtiles wrote them and not
/// one of their keys is a field our struct names, so this is the case that
/// proves the passthrough map is doing its job rather than serde quietly
/// dropping what it does not recognise.
#[test]
#[cfg_attr(miri, ignore)]
fn the_metadata_of_a_foreign_archive_parses_and_keeps_its_keys() {
    let directory = vectors("directory.json");
    let leaves = vectors("directory-leaves.json");

    let expectations: Vec<(&str, String)> = vec![
        (
            RASTER,
            str_at(&directory["archives"][RASTER]["metadata"], "decompressed"),
        ),
        (
            DUPES,
            str_at(&directory["archives"][DUPES]["metadata"], "decompressed"),
        ),
        (LEAVES, str_at(&leaves["metadata"], "decompressed")),
    ];

    for (name, raw) in expectations {
        let want: Value = serde_json::from_str(&raw).expect("the oracle's metadata is JSON");
        let want_keys: Vec<&String> = want.as_object().expect("an object").keys().collect();
        assert!(!want_keys.is_empty(), "{name} metadata has keys to lose");

        let reader = counted(name);
        let meta: &Metadata = reader.metadata().unwrap_or_else(|e| panic!("{name}: {e}"));

        // Round-tripping it back to JSON is how a dropped key becomes visible.
        let got: Value =
            serde_json::from_slice(&meta.to_json().expect("serialising")).expect("valid JSON");
        for key in want_keys {
            assert_eq!(
                got.get(key),
                want.get(key),
                "{name} metadata lost or changed {key}"
            );
        }
        assert!(
            meta.vnd_libviprs.is_none(),
            "{name} was not written by libviprs"
        );
    }
}

/// Metadata is fetched at most once, and only when it is asked for.
#[test]
#[cfg_attr(miri, ignore)]
fn metadata_is_read_lazily_and_only_once() {
    let reader = counted(RASTER);
    let header = *reader.header();
    reader.source().clear();

    let first = reader.metadata().expect("metadata parses").clone();
    let reads = reader.source().reads();
    assert_eq!(
        reads,
        vec![(header.metadata_offset, header.metadata_length as usize)],
        "one read, sized by the header"
    );

    reader.source().clear();
    let second = reader.metadata().expect("metadata parses");
    assert_eq!(&first, second, "the same object comes back");
    assert!(
        reader.source().reads().is_empty(),
        "the second call must not go back to the archive"
    );
}

/// The libviprs namespace survives a write and a read through a real archive.
#[test]
fn the_libviprs_namespace_round_trips_through_an_archive() {
    let mut metadata =
        Metadata::try_from_json(br#"{"name":"drawing"}"#).expect("the seed metadata parses");
    metadata.vnd_libviprs = Some(LibviprsMetadata::default());
    let json = metadata.to_json().expect("serialising metadata");

    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let archive = build_archive(&root, &[], &json, b"TILE", |_| {});
    let reader = memory_reader(archive).expect("the archive opens");

    let got = reader.metadata().expect("metadata parses");
    assert_eq!(got.name.as_deref(), Some("drawing"));
    let namespace = got
        .vnd_libviprs
        .as_ref()
        .expect("the vnd.libviprs key survived");
    assert_eq!(namespace.coordinate_convention, "zxy");
    assert_eq!(namespace.libviprs_meta_version, 1);
}

// ---------------------------------------------------------------------------
// Hostile input
// ---------------------------------------------------------------------------

/// Build a well-formed archive, then hand `tweak` the header to break.
///
/// Layout is header, root, metadata, leaves, tile data, which is the order the
/// spec's diagram shows. Every section is reached through its own offset, so
/// the order is a convenience rather than something a reader may rely on.
fn build_archive(
    root: &[Entry],
    leaves: &[Vec<Entry>],
    metadata_json: &[u8],
    tile_data: &[u8],
    tweak: impl FnOnce(&mut Header),
) -> Vec<u8> {
    let root_bytes = Compression::Gzip
        .compress(&serialize_entries(root).expect("the root serialises"))
        .expect("gzip");
    let metadata_bytes = Compression::Gzip.compress(metadata_json).expect("gzip");
    let leaf_blobs: Vec<Vec<u8>> = leaves
        .iter()
        .map(|leaf| {
            Compression::Gzip
                .compress(&serialize_entries(leaf).expect("a leaf serialises"))
                .expect("gzip")
        })
        .collect();

    let root_offset = 127u64;
    let metadata_offset = root_offset + root_bytes.len() as u64;
    let leaf_offset = metadata_offset + metadata_bytes.len() as u64;
    let leaf_length: u64 = leaf_blobs.iter().map(|b| b.len() as u64).sum();
    let tile_data_offset = leaf_offset + leaf_length;

    let mut header = Header {
        root_offset,
        root_length: root_bytes.len() as u64,
        metadata_offset,
        metadata_length: metadata_bytes.len() as u64,
        leaf_directories_offset: leaf_offset,
        leaf_directories_length: leaf_length,
        tile_data_offset,
        tile_data_length: tile_data.len() as u64,
        addressed_tiles_count: root.iter().map(|e| u64::from(e.run_length)).sum(),
        tile_entries_count: root.iter().filter(|e| e.run_length > 0).count() as u64,
        tile_contents_count: 0,
        clustered: true,
        tile_type: TileType::Png,
        min_zoom: 0,
        max_zoom: 7,
        ..Header::default()
    };
    tweak(&mut header);

    let mut out = Vec::new();
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&root_bytes);
    out.extend_from_slice(&metadata_bytes);
    for blob in &leaf_blobs {
        out.extend_from_slice(blob);
    }
    out.extend_from_slice(tile_data);
    out
}

/// The relative offsets of the leaves `build_archive` would lay out.
fn leaf_offsets(leaves: &[Vec<Entry>]) -> Vec<(u64, u32)> {
    let mut at = 0u64;
    let mut out = Vec::new();
    for leaf in leaves {
        let blob = Compression::Gzip
            .compress(&serialize_entries(leaf).expect("a leaf serialises"))
            .expect("gzip");
        out.push((at, blob.len() as u32));
        at += blob.len() as u64;
    }
    out
}

#[test]
fn something_that_is_not_a_pmtiles_archive_is_refused_by_its_magic() {
    let mut bytes = vec![0u8; 400];
    bytes[..7].copy_from_slice(b"NOTPMTS");
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::BadMagic { .. })),
        "got {:?}",
        got.err()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_version_two_archive_is_refused_by_the_version_it_carries() {
    let mut bytes = fixture_bytes(RASTER);
    bytes[7] = 2;
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::UnsupportedVersion { found: 2 })),
        "a v2 archive shares the magic and has a different layout, so it must be refused by name"
    );
}

#[test]
fn an_archive_shorter_than_a_header_is_refused_rather_than_read_short() {
    for len in [0usize, 1, 7, 126] {
        let got = memory_reader(vec![0u8; len]);
        assert!(
            matches!(&got, Err(PmTilesError::ShortHeader { .. })),
            "{len} bytes is not an archive, got {:?}",
            got.err()
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_truncated_archive_is_refused_rather_than_read_short() {
    let mut bytes = fixture_bytes(RASTER);
    bytes.truncate(900);
    let got = memory_reader(bytes);
    // The tile data section is the one that no longer fits. Naming the variant
    // matters: an archive this broken is refused by several layers and only
    // the bounds check is the one this test is about.
    assert!(
        matches!(&got, Err(PmTilesError::SectionOutOfBounds { .. })),
        "got {:?}",
        got.err()
    );
}

/// The case that makes `pmtiles verify` segfault.
///
/// Its four bounds checks are all `length > fileSize` and none of them is
/// `offset + length > fileSize`, so a root offset of 999999 in an 1878-byte
/// file passes every one, and `DeserializeEntries` then discards the gzip
/// error and nil-dereferences the reader. We check the sum.
#[test]
#[cfg_attr(miri, ignore)]
fn a_root_offset_past_the_end_is_refused_where_the_reference_nil_dereferences() {
    let mut bytes = fixture_bytes(RASTER);
    assert_eq!(bytes.len(), 1878);
    bytes[8..16].copy_from_slice(&999_999u64.to_le_bytes());
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::SectionOutOfBounds { .. })),
        "got {:?}",
        got.err()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_section_whose_offset_and_length_overflow_is_refused_without_wrapping() {
    // Root offset near the top of the range with a non-zero length, so
    // `offset + length` wraps unless the addition is checked.
    let mut bytes = fixture_bytes(RASTER);
    bytes[8..16].copy_from_slice(&(u64::MAX - 8).to_le_bytes());
    bytes[16..24].copy_from_slice(&64u64.to_le_bytes());
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::Overflow { .. })),
        "a root whose offset and length wrap must be refused, got {:?}",
        got.err()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_root_length_past_the_end_is_refused() {
    let mut bytes = fixture_bytes(RASTER);
    bytes[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::SectionOutOfBounds { .. })),
        "got {:?}",
        got.err()
    );
}

/// A root that is not gzip is a typed error. This is the other half of the
/// reference's crash: it writes `reader, _ = gzip.NewReader(data)` and then
/// dereferences a nil reader.
#[test]
#[cfg_attr(miri, ignore)]
fn a_root_that_is_not_gzip_is_a_typed_error_and_not_a_panic() {
    let mut bytes = fixture_bytes(RASTER);
    let root_offset = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
    let root_length = u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")) as usize;
    for byte in &mut bytes[root_offset..root_offset + root_length] {
        *byte = 0x41;
    }
    let got = memory_reader(bytes);
    assert!(
        matches!(&got, Err(PmTilesError::Io(_))),
        "a root that will not inflate is the gzip error, surfaced, got {:?}",
        got.err()
    );
}

/// The root must fit in the first 16384 bytes, which the spec makes a MUST so
/// a latency-sensitive client can fetch the header and the whole root at once.
///
/// The archive here is genuinely big enough to hold the root where the header
/// says it is, so the only thing wrong with it is the budget. An archive that
/// was also too short would be caught by the section bounds first and this
/// test would be measuring the wrong rule.
#[test]
fn a_root_reaching_past_the_sixteen_kilobyte_budget_is_refused() {
    let entries = [Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let root = Compression::Gzip
        .compress(&serialize_entries(&entries).expect("the root serialises"))
        .expect("gzip");

    // Start the root four bytes short of the budget, so it ends past it.
    let root_offset = MAX_ROOT_SPAN - 4;
    let tail = root_offset + root.len() as u64;
    let header = Header {
        root_offset,
        root_length: root.len() as u64,
        metadata_offset: tail,
        metadata_length: 0,
        leaf_directories_offset: tail,
        leaf_directories_length: 0,
        tile_data_offset: tail,
        tile_data_length: 4,
        ..Header::default()
    };

    let mut archive = vec![0u8; (tail + 4) as usize];
    archive[..127].copy_from_slice(&header.encode());
    archive[root_offset as usize..tail as usize].copy_from_slice(&root);
    assert!(
        archive.len() as u64 > MAX_ROOT_SPAN,
        "the archive is long enough that only the budget is violated"
    );

    let got = memory_reader(archive);
    assert!(
        matches!(&got, Err(PmTilesError::RootDirectoryTooLarge { .. })),
        "got {:?}",
        got.err()
    );
}

/// A metadata section that expands without bound is refused at the ceiling
/// rather than inflated into memory. No length field in PMTiles v3 is an
/// uncompressed length, so the cap is the only defence there is.
#[test]
fn a_metadata_bomb_is_refused_at_the_ceiling_rather_than_inflated() {
    let bomb_source = vec![b'{'; MAX_METADATA_BYTES + 4096];
    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let archive = build_archive(&root, &[], &bomb_source, b"TILE", |_| {});
    // The compressed metadata is tiny; the point is that the reader never
    // finds out how big it would have been.
    assert!(
        archive.len() < 64 * 1024,
        "the bomb compresses to {} bytes, which is the whole point",
        archive.len()
    );

    let reader = memory_reader(archive).expect("the archive opens, the metadata is lazy");
    let got = reader.metadata();
    assert!(
        matches!(&got, Err(PmTilesError::DecompressionLimit { .. })),
        "got {:?}",
        got.err()
    );
}

/// The same for a directory, which is the one a reader cannot make lazy.
#[test]
fn a_directory_bomb_is_refused_at_the_ceiling_rather_than_inflated() {
    let bomb = Compression::Gzip
        .compress(&vec![0u8; MAX_DIRECTORY_BYTES + 4096])
        .expect("gzip");
    let tail = 127 + bomb.len() as u64;
    assert!(
        tail <= MAX_ROOT_SPAN,
        "the compressed bomb is {} bytes, which would trip the root budget first",
        bomb.len()
    );
    let header = Header {
        root_offset: 127,
        root_length: bomb.len() as u64,
        metadata_offset: tail,
        metadata_length: 0,
        leaf_directories_offset: tail,
        tile_data_offset: tail,
        ..Header::default()
    };

    let mut archive = Vec::new();
    archive.extend_from_slice(&header.encode());
    archive.extend_from_slice(&bomb);

    let got = memory_reader(archive);
    assert!(
        matches!(&got, Err(PmTilesError::DecompressionLimit { .. })),
        "got {:?}",
        got.err()
    );
}

/// A tile entry naming bytes outside the tile data section is refused rather
/// than served from wherever the sum happens to land.
#[test]
fn a_tile_entry_pointing_outside_the_tile_data_section_is_refused() {
    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let archive = build_archive(&root, &[], b"{}", b"TILE", |header| {
        // Shrink the declared section so the entry reaches past its end.
        header.tile_data_length = 2;
    });
    let reader = memory_reader(archive).expect("the archive opens");
    let got = reader.get_tile(0, 0, 0);
    assert!(
        matches!(&got, Err(PmTilesError::EntryOutOfBounds { .. })),
        "got {got:?}"
    );
}

/// A leaf pointer naming bytes outside the leaf section, likewise.
#[test]
fn a_leaf_pointer_pointing_outside_the_leaf_section_is_refused() {
    let leaf = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let leaves = vec![leaf];
    let placed = leaf_offsets(&leaves);
    let root = vec![Entry {
        tile_id: 0,
        offset: placed[0].0,
        length: placed[0].1,
        run_length: 0,
    }];
    let archive = build_archive(&root, &leaves, b"{}", b"TILE", |header| {
        header.leaf_directories_length = 2;
    });
    let reader = memory_reader(archive).expect("the archive opens");
    let got = reader.get_tile(0, 0, 0);
    assert!(
        matches!(&got, Err(PmTilesError::EntryOutOfBounds { .. })),
        "got {got:?}"
    );
}

/// A leaf chain deeper than the cap is refused rather than followed forever.
///
/// The spec only "discourages" more than one level and states no limit, so a
/// reader that does not impose one of its own can be walked in a circle by a
/// hostile archive. This is the tightest circle there is: one leaf whose only
/// entry is a pointer to itself.
///
/// The self-reference is a fixed point, because the length written inside the
/// leaf is the compressed length of the leaf that contains it, so it is solved
/// by iterating rather than computed. Convergence is asserted, not assumed.
#[test]
fn a_leaf_chain_deeper_than_the_cap_is_refused() {
    let gzipped = |leaf: &[Entry]| {
        Compression::Gzip
            .compress(&serialize_entries(leaf).expect("a leaf serialises"))
            .expect("gzip")
    };

    let mut leaf = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 1,
        run_length: 0,
    }];
    for _ in 0..8 {
        let len = gzipped(&leaf).len() as u32;
        if leaf[0].length == len {
            break;
        }
        leaf[0].length = len;
    }
    assert_eq!(
        leaf[0].length,
        gzipped(&leaf).len() as u32,
        "the self-referencing leaf has to describe its own size"
    );

    let leaves = vec![leaf.clone()];
    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: leaf[0].length,
        run_length: 0,
    }];
    let archive = build_archive(&root, &leaves, b"{}", b"TILE", |_| {});
    let reader = memory_reader(archive).expect("the archive opens");
    let got = reader.get_tile(0, 0, 0);
    assert!(
        matches!(&got, Err(PmTilesError::LeafDepthExceeded { .. })),
        "a cycle must stop at the depth cap of {MAX_LEAF_DEPTH}, got {got:?}"
    );
}

/// A directory claiming more entries than its bytes can hold is refused
/// before anything is allocated for them.
#[test]
fn a_directory_claiming_more_entries_than_its_bytes_can_hold_is_refused() {
    // A hand-built directory body: the count varint says a million, and the
    // buffer is twenty bytes long.
    let mut body = Vec::new();
    libviprs::pmtiles::varint::encode_uvarint(1_000_000, &mut body);
    body.extend_from_slice(&[1u8; 20]);
    let compressed = Compression::Gzip.compress(&body).expect("gzip");

    let tail = 127 + compressed.len() as u64;
    let header = Header {
        root_offset: 127,
        root_length: compressed.len() as u64,
        metadata_offset: tail,
        leaf_directories_offset: tail,
        tile_data_offset: tail,
        ..Header::default()
    };

    let mut archive = Vec::new();
    archive.extend_from_slice(&header.encode());
    archive.extend_from_slice(&compressed);

    let got = memory_reader(archive);
    assert!(
        matches!(&got, Err(PmTilesError::DirectoryTooManyEntries { .. })),
        "got {:?}",
        got.err()
    );
}

/// An internal compression this build cannot decode is a named refusal, not a
/// wrong answer.
#[test]
fn an_internal_compression_this_build_cannot_decode_is_refused_by_name() {
    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    for compression in [Compression::Unknown, Compression::Brotli, Compression::Zstd] {
        let archive = build_archive(&root, &[], b"{}", b"TILE", |header| {
            header.internal_compression = compression;
        });
        let got = memory_reader(archive);
        assert!(
            matches!(&got, Err(PmTilesError::UnsupportedCompression { .. })),
            "{compression:?}: got {:?}",
            got.err()
        );
    }
}

/// A tile compression this build cannot decode is fine, because a reader that
/// hands the stored bytes onward never has to decompress a tile.
#[test]
fn a_tile_compression_this_build_cannot_decode_still_serves_the_stored_bytes() {
    let root = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 4,
        run_length: 1,
    }];
    let archive = build_archive(&root, &[], b"{}", b"TILE", |header| {
        header.tile_compression = Compression::Brotli;
    });
    let reader = memory_reader(archive).expect("the archive opens");
    assert_eq!(
        reader.get_tile(0, 0, 0).expect("a lookup").as_deref(),
        Some(&b"TILE"[..]),
        "the stored bytes come back untouched"
    );
    assert_eq!(reader.header().tile_compression, Compression::Brotli);
}

/// The reader is usable from several threads at once, which is what the whole
/// `RangeReader: Send + Sync` shape is for.
#[test]
#[cfg_attr(miri, ignore)]
fn one_reader_serves_several_threads_at_once() {
    let reader = counted(RASTER);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..8 {
                    assert!(reader.get_tile(2, 3, 3).expect("a lookup").is_some());
                }
            });
        }
    });
}

/// `Reader` is the type a downstream caller names, so this is the compile-time
/// half of the surface check: it must be `Send + Sync` for the engine to share
/// one across workers.
#[test]
fn the_reader_type_is_shareable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Reader<InMemory>>();
    assert_send_sync::<Reader<FileRangeReader>>();
}

/// The convenience constructor reaches an archive by path, which is the entry
/// point everything above the reader will actually use.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_opens_by_path_through_the_convenience_constructor() {
    let _ = fixture_bytes(RASTER);
    let reader = Reader::try_open(fixture_path(RASTER)).expect("the golden opens by path");
    assert_eq!(reader.min_zoom(), 0);
    assert_eq!(reader.max_zoom(), 2);
    assert_eq!(reader.tile_format(), TileType::Png);
    assert!(reader.get_tile(0, 0, 0).expect("a lookup").is_some());
}
