//! The PMTiles v3 format primitives, pinned from outside the crate (issue #987).
//!
//! An integration test compiles as an *external* crate, so everything here is
//! reachable exactly the way a downstream caller reaches it. That makes this
//! file the acceptance check for three things the unit tests inside
//! `src/pmtiles/` cannot state:
//!
//! * the public surface really is public. `libviprs::pmtiles::{Header,
//!   TileType, Compression, Entry, RangeReader, FileRangeReader, PmTilesError,
//!   Metadata}` are named here by path, so a missing `pub use` fails the build
//!   rather than the docs;
//! * [`RangeReader`] is **object-safe**. A `Box<dyn RangeReader>` is what lets
//!   a future HTTP or S3 backend slot in without touching the reader or the
//!   writer, and object safety is a property of the trait definition that only
//!   a `dyn` coercion proves;
//! * the enums are `#[non_exhaustive]`, which is observable only from another
//!   crate (a trailing `_` arm is `unreachable_patterns` for an exhaustive
//!   enum and required for a non-exhaustive one).
//!
//! # The tile ids are the part that has to be right
//!
//! Every other piece of this module can be checked against itself. The Hilbert
//! mapping cannot: an encoder and a decoder that share a wrong orientation
//! round-trip perfectly and agree with nothing else in the world. So the
//! vectors below come from three places that are not each other, and
//! [`tile_ids_match_the_published_reference_vectors`] carries the six the
//! PMTiles project publishes in its own test suites.
//!
//! A transposition is the failure this guards hardest, because it survives
//! every symmetric fixture: `(3, 5, 2)` and `(3, 2, 5)` are the same tile with
//! `x` and `y` swapped, and they must land on different ids.

use std::io::Write;

use libviprs::pmtiles::directory::{deserialize_entries, serialize_entries};
use libviprs::pmtiles::tileid::{MAX_TILE_ID, MAX_ZOOM};
use libviprs::pmtiles::{
    Compression, Entry, FileRangeReader, Header, Metadata, PmTilesError, RangeReader, TileType,
    tileid_to_zxy, zxy_to_tileid,
};

/// The six `(z, x, y) -> id` pairs the PMTiles project publishes as the
/// worked example of its Hilbert ordering, plus the level-2 and level-3 tiles
/// that separate a correct curve from a transposed or mirrored one.
///
/// The first six are the published set. The rest are this lane's own
/// derivation, cross-checked against a second, independent implementation
/// (a recursive quadrant construction rather than the bit-twiddling form)
/// before they were written down.
const TILE_ID_VECTORS: &[(u8, u32, u32, u64)] = &[
    // Published.
    (0, 0, 0, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 2),
    (1, 1, 1, 3),
    (1, 1, 0, 4),
    (2, 0, 0, 5),
    // Derived. Asymmetric on purpose: each one changes if the curve is
    // transposed, mirrored, or rotated the wrong way at a recursion step.
    (2, 1, 0, 6),
    (2, 3, 1, 17),
    (2, 0, 3, 10),
    (3, 5, 2, 76),
    (3, 2, 5, 50),
    (5, 17, 9, 1213),
    (7, 37, 27, 7473),
    (9, 301, 178, 309_426),
    (14, 9001, 3, 335_872_411),
    (17, 83922, 44217, 20_571_255_868),
    (20, 999_983, 7, 1_461_989_237_823),
    (31, 1, 2, 1_537_228_672_809_129_314),
    (31, 2_147_483_647, 2_147_483_647, 4_611_686_018_427_387_903),
];

#[test]
fn tile_ids_match_the_published_reference_vectors() {
    for &(z, x, y, want) in TILE_ID_VECTORS {
        let got = zxy_to_tileid(z, x, y).expect("vector is inside the addressable range");
        assert_eq!(got, want, "zxy_to_tileid({z}, {x}, {y})");
        assert_eq!(
            tileid_to_zxy(want).expect("a vector id decodes"),
            (z, x, y),
            "tileid_to_zxy({want})"
        );
    }
}

/// A transposed curve round-trips with itself and matches nothing else, so the
/// only fixtures that can catch it are ones where `x != y`.
#[test]
fn swapping_x_and_y_changes_the_tile_id() {
    let a = zxy_to_tileid(3, 5, 2).unwrap();
    let b = zxy_to_tileid(3, 2, 5).unwrap();
    assert_ne!(a, b, "(3,5,2) and (3,2,5) must not share a tile id");
    assert_eq!((a, b), (76, 50));
}

#[test]
fn tile_id_round_trips_over_a_sampled_grid() {
    let mut seen = std::collections::BTreeSet::new();
    for z in 0..=6u8 {
        let side = 1u32 << z;
        for y in 0..side {
            for x in 0..side {
                let id = zxy_to_tileid(z, x, y).unwrap();
                assert!(seen.insert(id), "id {id} is used twice");
                assert_eq!(tileid_to_zxy(id).unwrap(), (z, x, y));
            }
        }
    }
    // A positive control on the loop above: an empty or short sweep would let
    // every assertion in it pass vacuously. 4^0 + ... + 4^6 = 5461.
    assert_eq!(seen.len(), 5461, "the sweep did not visit the whole grid");
}

#[test]
fn coordinates_outside_the_addressable_range_are_refused_not_wrapped() {
    // Zoom 32 has no representable id: the level starts past `u64::MAX`.
    assert!(matches!(
        zxy_to_tileid(MAX_ZOOM + 1, 0, 0),
        Err(PmTilesError::ZoomOutOfRange { .. })
    ));
    // x and y are bounded by the level, independently.
    assert!(matches!(
        zxy_to_tileid(3, 8, 0),
        Err(PmTilesError::CoordOutOfRange { .. })
    ));
    assert!(matches!(
        zxy_to_tileid(3, 0, 8),
        Err(PmTilesError::CoordOutOfRange { .. })
    ));
    // The positive control: one step inside the bound is fine, so the two
    // refusals above are about the bound and not about the whole level.
    assert!(zxy_to_tileid(3, 7, 7).is_ok());

    // And the id side of the same bound.
    assert!(tileid_to_zxy(MAX_TILE_ID).is_ok());
    assert!(matches!(
        tileid_to_zxy(MAX_TILE_ID + 1),
        Err(PmTilesError::TileIdOutOfRange { .. })
    ));
}

#[test]
fn header_round_trips_through_exactly_127_bytes() {
    let header = sample_header();
    let bytes = header.encode();
    assert_eq!(bytes.len(), 127, "the v3 header is 127 bytes");
    assert_eq!(&bytes[..7], b"PMTiles");
    assert_eq!(bytes[7], 3);

    let decoded = Header::try_decode(&bytes).expect("a header this crate wrote decodes");
    assert_eq!(decoded, header);
}

#[test]
fn a_foreign_or_truncated_header_is_refused_with_a_typed_error() {
    let good = sample_header().encode();

    let mut bad_magic = good;
    bad_magic[0] = b'X';
    assert!(matches!(
        Header::try_decode(&bad_magic),
        Err(PmTilesError::BadMagic { .. })
    ));

    let mut bad_version = good;
    bad_version[7] = 4;
    assert!(matches!(
        Header::try_decode(&bad_version),
        Err(PmTilesError::UnsupportedVersion { .. })
    ));

    assert!(matches!(
        Header::try_decode(&good[..126]),
        Err(PmTilesError::ShortHeader { .. })
    ));

    // The positive control: the unmodified bytes still decode, so the three
    // refusals above are about what was changed and not about the fixture.
    assert!(Header::try_decode(&good).is_ok());
}

#[test]
fn a_directory_round_trips_through_the_column_oriented_form() {
    let entries = vec![
        Entry {
            tile_id: 0,
            offset: 0,
            length: 10,
            run_length: 1,
        },
        // Contiguous with the entry before it: the offset column stores a 0.
        Entry {
            tile_id: 1,
            offset: 10,
            length: 20,
            run_length: 2,
        },
        // A gap, and a leaf pointer (`run_length` 0).
        Entry {
            tile_id: 5,
            offset: 100,
            length: 30,
            run_length: 0,
        },
    ];

    let bytes = serialize_entries(&entries).expect("a directory this small serializes");
    let back = deserialize_entries(&bytes).expect("what this crate wrote, it reads");
    assert_eq!(back, entries);
}

#[test]
fn metadata_carries_the_namespaced_libviprs_object() {
    let json = sample_metadata_json();
    let meta = Metadata::try_from_json(json.as_bytes()).expect("the sample metadata parses");
    let vnd = meta
        .vnd_libviprs
        .as_ref()
        .expect("the libviprs namespace survives a parse");
    assert_eq!(vnd.coordinate_convention, "zxy");
    assert!(vnd.libviprs_meta_version >= 1);

    // Re-serialising keeps the namespaced key spelled the way the spec's
    // `vnd.` convention wants it, which a struct field name alone would not.
    let out = meta.to_json().expect("metadata re-serialises");
    let text = String::from_utf8(out).unwrap();
    assert!(
        text.contains("\"vnd.libviprs\""),
        "the namespaced key must survive a round trip, got: {text}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn file_range_reader_returns_exactly_the_bytes_asked_for() {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    let payload: Vec<u8> = (0..=255u8).collect();
    file.write_all(&payload).expect("write");
    file.flush().expect("flush");

    let reader = FileRangeReader::try_open(file.path()).expect("open");
    assert_eq!(reader.size().unwrap(), Some(256));

    assert_eq!(reader.read_range(0, 4).unwrap(), &payload[0..4]);
    assert_eq!(reader.read_range(200, 56).unwrap(), &payload[200..256]);
    assert_eq!(reader.read_range(255, 1).unwrap(), vec![255]);
    assert!(reader.read_range(0, 0).unwrap().is_empty());

    // A short read is a failure, never a truncated buffer: the reader has no
    // way to tell the caller it returned less than it was asked for.
    assert!(reader.read_range(250, 10).is_err());
    assert!(reader.read_range(256, 1).is_err());
    // The positive control: the same length one byte earlier succeeds.
    assert_eq!(reader.read_range(246, 10).unwrap().len(), 10);
}

/// A `Box<dyn RangeReader>` is what a future HTTP or S3 backend slots into, so
/// object safety is part of the contract rather than an implementation detail.
#[test]
#[cfg_attr(miri, ignore)]
fn range_reader_is_object_safe() {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(b"PMTiles").expect("write");
    file.flush().expect("flush");

    let boxed: Box<dyn RangeReader> =
        Box::new(FileRangeReader::try_open(file.path()).expect("open"));
    assert_eq!(boxed.read_range(0, 7).unwrap(), b"PMTiles");
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_tile_type_non_exhaustive(v: &TileType) {
    match v {
        TileType::Unknown => {}
        TileType::Mvt => {}
        TileType::Png => {}
        TileType::Jpeg => {}
        TileType::Webp => {}
        TileType::Avif => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_compression_non_exhaustive(v: &Compression) {
    match v {
        Compression::Unknown => {}
        Compression::None => {}
        Compression::Gzip => {}
        Compression::Brotli => {}
        Compression::Zstd => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_pmtiles_error_non_exhaustive(v: &PmTilesError) {
    match v {
        PmTilesError::Io(_) => {}
        PmTilesError::BadMagic { .. } => {}
        PmTilesError::UnsupportedVersion { .. } => {}
        PmTilesError::ShortHeader { .. } => {}
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A header with a distinct value in every field, so a test that compares two
/// of them cannot pass by two fields happening to hold the same number.
fn sample_header() -> Header {
    Header {
        root_offset: 127,
        root_length: 301,
        metadata_offset: 428,
        metadata_length: 902,
        leaf_directories_offset: 1330,
        leaf_directories_length: 4096,
        tile_data_offset: 5426,
        tile_data_length: 9_876_543_210,
        addressed_tiles_count: 71,
        tile_entries_count: 53,
        tile_contents_count: 47,
        clustered: true,
        internal_compression: Compression::Gzip,
        tile_compression: Compression::None,
        tile_type: TileType::Webp,
        min_zoom: 3,
        max_zoom: 17,
        min_lon_e7: -123_456_789,
        min_lat_e7: -45_678_901,
        max_lon_e7: 109_876_543,
        max_lat_e7: 78_901_234,
        center_zoom: 11,
        center_lon_e7: -12_345_678,
        center_lat_e7: 87_654_321,
    }
}

fn sample_metadata_json() -> String {
    r#"{
      "name": "drawing",
      "vnd.libviprs": {
        "libviprs_meta_version": 1,
        "libviprs_version": "0.4.0",
        "coordinate_convention": "zxy",
        "source": {
          "width": 4001,
          "height": 2999,
          "pixel_format": "rgb8"
        },
        "generation": {
          "tile_size": 512,
          "overlap": 1,
          "layout": "xyz",
          "format": { "kind": "png" },
          "concurrency": 7,
          "background_rgb": [1, 2, 3],
          "blank_strategy": { "kind": "emit" }
        }
      }
    }"#
    .to_string()
}

// ---------------------------------------------------------------------------
// The go-pmtiles oracle, loaded from the committed vectors at run time
// ---------------------------------------------------------------------------
//
// Everything above this line is this lane's own derivation of the format,
// which is worth having and cannot settle anything on its own: an encoder and
// a decoder that share a misreading agree with each other perfectly. What
// follows pins the same code against bytes written by `protomaps/go-pmtiles`
// v1.31.2, which never saw libviprs.
//
// The rows are read out of `tests/fixtures/pmtiles/vectors/*.json` at run time
// rather than transcribed into constants here. A transcribed number carries no
// evidence of where it came from, and the usual way a pinned test dies is
// somebody editing the constant instead of the code. See
// `tests/common/pmtiles_oracle.rs` for the sha256 pinning and the count
// cross-check that keep a silent parse failure from reading as a pass.

#[path = "common/pmtiles_oracle.rs"]
mod oracle;

use libviprs::pmtiles::header::HEADER_BYTES;

/// A generous but finite ceiling for the directory gunzips below. Nothing in
/// the goldens decompresses past 16 KiB; the point of naming a limit at all is
/// that PMTiles v3 stores no uncompressed length anywhere, so a reader has to
/// cap the output rather than pre-size it.
const DECOMPRESS_CEILING: usize = 1 << 20;

/// The file this whole oracle section is pinned against is the file that was
/// measured, and the parse reaches every row in it.
///
/// The count comes from the `counts` block that go-pmtiles' dump program
/// wrote, not from this parser, so a parse that silently found half the rows
/// fails here instead of passing six loops vacuously.
#[test]
#[cfg_attr(miri, ignore)]
fn the_oracle_tile_id_vectors_are_the_file_this_suite_was_pinned_against() {
    let vectors = oracle::tileid_vectors();

    // Six sections of `(z, x, y) -> id` rows. `level_bases` is a different
    // shape and `out_of_range_observations` is behaviour rather than vectors,
    // so neither is counted into the distinct-pair total the file declares.
    let sections = [
        "first_and_last_of_level",
        "orientation_boundaries",
        "off_grid",
        "hilbert_order_z2",
        "hilbert_order_z3",
        "convention_discriminators",
    ];
    let mut distinct = std::collections::BTreeSet::new();
    for section in sections {
        for row in oracle::tile_id_rows(&vectors, section) {
            distinct.insert((row.z, row.x, row.y, row.tile_id));
        }
    }
    // `out_of_range_observations` is deliberately not in the sum. Those rows
    // are behaviour rather than vectors, and adding them takes the total to
    // 172, which is how I know the file's own 162 counts the six sections
    // above and nothing else.
    assert_eq!(
        oracle::tile_id_rows(&vectors, "out_of_range_observations").len(),
        10,
        "the out-of-range rows are still ten, so 162 + 10 is still 172"
    );

    let declared = oracle::declared_count(&vectors, "distinct_zxy_to_tileid_pairs");
    assert_eq!(
        distinct.len(),
        declared,
        "the file declares {declared} distinct pairs and the parse found {}",
        distinct.len()
    );
    assert_eq!(
        declared, 162,
        "the pinned vector file has 162 distinct pairs"
    );
}

/// The twelve rows that tell the candidate conventions apart.
///
/// This is the test that matters and it is deliberately on its own. 150 of the
/// 162 pairs in the vector file are structural coordinates (level firsts and
/// lasts, quadrant corners, the tiles straddling the centre seam) and several
/// different mappings agree on all of them, **plain Z-order with no rotation
/// included**. A suite that pins only those has 150 green rows that cannot
/// fail.
///
/// These twelve run from zoom 9 to zoom 15, sit off every quadrant boundary,
/// have `x` and `y` of differing parity, and four of them are arranged as
/// swapped pairs so a mapping that is accidentally symmetric does not survive
/// either. `(12, 3423, 1763) -> 19078479` is the row the specification itself
/// publishes and the one that rules out 21314735, 19217573 and 20757839.
#[test]
#[cfg_attr(miri, ignore)]
fn the_twelve_convention_discriminators_from_the_oracle_hold() {
    let vectors = oracle::tileid_vectors();
    let rows = oracle::tile_id_rows(&vectors, "convention_discriminators");
    assert_eq!(rows.len(), 12, "the discriminating set is twelve rows");

    for row in &rows {
        assert!(
            row.roundtrip_ok,
            "({}, {}, {}) is a discriminator, so the oracle must round-trip it",
            row.z, row.x, row.y
        );
        assert_eq!(
            zxy_to_tileid(row.z, row.x, row.y).expect("a discriminator is addressable"),
            row.tile_id,
            "zxy_to_tileid({}, {}, {})",
            row.z,
            row.x,
            row.y
        );
        assert_eq!(
            tileid_to_zxy(row.tile_id).expect("a discriminator id decodes"),
            (row.z, row.x, row.y),
            "tileid_to_zxy({})",
            row.tile_id
        );
    }

    // The swapped pairs, called out by name. A symmetric mapping passes every
    // row above taken one at a time and collides here.
    let by_coord: std::collections::BTreeMap<(u8, u32, u32), u64> =
        rows.iter().map(|r| ((r.z, r.x, r.y), r.tile_id)).collect();
    let mut swapped = 0;
    for (&(z, x, y), &id) in &by_coord {
        if let Some(&other) = by_coord.get(&(z, y, x))
            && x != y
        {
            assert_ne!(id, other, "({z},{x},{y}) and ({z},{y},{x}) share an id");
            swapped += 1;
        }
    }
    // Six rows, which is three pairs seen from both ends. The oracle's own
    // FINDINGS.md and this epic's brief both say "four of them are swapped
    // pairs"; the file has three, at z=12, z=13 and z=15. I counted rather
    // than inherited the number, which is the only reason it is right here.
    assert_eq!(swapped, 6, "three swapped pairs, seen from both ends");
}

/// The level firsts and lasts, the quadrant corners and the centre seam: 76
/// rows that a wrong convention still passes.
///
/// This test is here to be **green under the Z-order mutation**, which is the
/// only way the discriminator tests can be shown to be doing the work. I
/// measured which rows separate the two mappings rather than assuming it: with
/// `rotate` turned into a no-op, `first_and_last_of_level` gives 0 mismatches
/// out of 32 and `orientation_boundaries` gives 0 out of 44. So these 76 rows
/// are the control half, and their value is not that they are green, it is
/// that they say what the mapping does at the edges.
#[test]
#[cfg_attr(miri, ignore)]
fn the_structural_oracle_rows_hold() {
    let vectors = oracle::tileid_vectors();
    let mut checked = 0;
    for section in ["first_and_last_of_level", "orientation_boundaries"] {
        for row in oracle::tile_id_rows(&vectors, section) {
            assert_eq!(
                zxy_to_tileid(row.z, row.x, row.y).expect("a structural row is addressable"),
                row.tile_id,
                "{section}: zxy_to_tileid({}, {}, {})",
                row.z,
                row.x,
                row.y
            );
            assert_eq!(
                tileid_to_zxy(row.tile_id).expect("a structural id decodes"),
                (row.z, row.x, row.y),
                "{section}: tileid_to_zxy({})",
                row.tile_id
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 32 + 44, "the structural sweep lost rows");
}

/// The `off_grid` rows, which are not structural and are not advertised as
/// discriminators either.
///
/// They are here in their own test because I measured them: 12 of the 17 change
/// under a no-rotation mapping, so this section discriminates nearly as hard as
/// the twelve rows that are named for it. Filing them with the structural rows
/// would have made the control half of the negative control go red and told me
/// nothing about why.
#[test]
#[cfg_attr(miri, ignore)]
fn the_off_grid_oracle_rows_hold() {
    let vectors = oracle::tileid_vectors();
    let rows = oracle::tile_id_rows(&vectors, "off_grid");
    assert_eq!(rows.len(), 17, "the off-grid section is seventeen rows");
    for row in &rows {
        assert_eq!(
            zxy_to_tileid(row.z, row.x, row.y).expect("an off-grid row is addressable"),
            row.tile_id,
            "zxy_to_tileid({}, {}, {})",
            row.z,
            row.x,
            row.y
        );
        assert_eq!(
            tileid_to_zxy(row.tile_id).expect("an off-grid id decodes"),
            (row.z, row.x, row.y),
            "tileid_to_zxy({})",
            row.tile_id
        );
    }
}

/// Zoom 2 and zoom 3 enumerated in tile id order, from the oracle.
///
/// A Hilbert curve visits ids 5, 6, 7, 8 as `(0,0), (1,0), (1,1), (0,1)` and
/// Z-order visits them as `(0,0), (0,1), (1,0), (1,1)`, so these two sections
/// separate the two curves at the cheapest possible zoom. They are grouped
/// with the discriminators rather than the structural rows for exactly that
/// reason.
#[test]
#[cfg_attr(miri, ignore)]
fn the_enumerated_low_zoom_orders_are_hilbert_and_not_z_order() {
    let vectors = oracle::tileid_vectors();
    let mut checked = 0;
    for (section, first) in [("hilbert_order_z2", 5u64), ("hilbert_order_z3", 21)] {
        let rows = oracle::tile_id_rows(&vectors, section);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(
                row.tile_id,
                first + index as u64,
                "{section} is supposed to be in id order"
            );
            assert_eq!(
                zxy_to_tileid(row.z, row.x, row.y).expect("addressable"),
                row.tile_id,
                "{section}: zxy_to_tileid({}, {}, {})",
                row.z,
                row.x,
                row.y
            );
            assert_eq!(
                tileid_to_zxy(row.tile_id).expect("decodes"),
                (row.z, row.x, row.y),
                "{section}: tileid_to_zxy({})",
                row.tile_id
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 16 + 64, "the enumerated sweep lost rows");
}

/// `level_bases`: the first id, last id and tile count of every level the
/// oracle dumped, including zoom 31 where `u64` runs out.
#[test]
#[cfg_attr(miri, ignore)]
fn the_oracle_level_bases_are_where_each_zoom_starts_and_ends() {
    let vectors = oracle::tileid_vectors();
    let rows = vectors["level_bases"]
        .as_array()
        .expect("level_bases is an array");
    assert_eq!(
        rows.len(),
        oracle::declared_count(&vectors, "level_bases"),
        "level_bases lost rows in the parse"
    );

    for row in rows {
        let z = u8::try_from(row["z"].as_u64().expect("z")).expect("a zoom fits in a u8");
        let first = row["first_tile_id"].as_u64().expect("first_tile_id");
        let last = row["last_tile_id"].as_u64().expect("last_tile_id");
        let count = row["tile_count"].as_u64().expect("tile_count");

        assert_eq!(
            zxy_to_tileid(z, 0, 0).expect("(z, 0, 0) is addressable"),
            first,
            "zoom {z} starts at the wrong id"
        );
        assert_eq!(last - first + 1, count, "zoom {z}'s own arithmetic");
        assert_eq!(
            count,
            1u64 << (2 * u32::from(z)),
            "zoom {z} holds 4^z tiles"
        );
        // The last id of the level decodes back into the level, and the id one
        // past it does not, which is what makes the base a boundary rather
        // than a number that happens to be right.
        let (dz, _, _) = tileid_to_zxy(last).expect("the last id of a level decodes");
        assert_eq!(dz, z, "the last id of zoom {z} decodes into another zoom");
        if z < 31 {
            let (nz, _, _) = tileid_to_zxy(last + 1).expect("the next id decodes");
            assert_eq!(nz, z + 1, "the id after zoom {z} is not zoom {}", z + 1);
        }
    }
}

/// The rows where go-pmtiles answers a question it was not asked.
///
/// `ZxyToID` masks `x` and `y` into the grid instead of checking them, so
/// `(2, 4, 0)` comes back as tile id 5, which is the `(2, 0, 0)` tile, and it
/// reaches the CLI: `pmtiles tile raster-z0z2.pmtiles 2 4 0` exits 0 and writes
/// a real 74-byte PNG for a coordinate that does not exist. It saturates `z`
/// the same way, so zooms 32, 33 and 63 all answer 6148914691236517205.
///
/// These are evidence about the reference, not a target. This is the oracle
/// posture where the reference accepts the input and produces garbage, so
/// refusing is more faithful than matching, and each refusal is asserted **by
/// name** rather than as "some error".
#[test]
#[cfg_attr(miri, ignore)]
fn the_rows_go_pmtiles_masks_are_refused_by_name() {
    let vectors = oracle::tileid_vectors();
    let rows = oracle::tile_id_rows(&vectors, "out_of_range_observations");
    assert_eq!(rows.len(), 10, "the out-of-range observations are ten rows");

    let mut zoom_refusals = 0;
    let mut coord_refusals = 0;
    for row in &rows {
        assert!(
            !row.roundtrip_ok,
            "({}, {}, {}) is supposed to be a row the reference mangles",
            row.z, row.x, row.y
        );
        match zxy_to_tileid(row.z, row.x, row.y) {
            Err(PmTilesError::ZoomOutOfRange { zoom, max }) => {
                assert_eq!(zoom, row.z);
                assert_eq!(max, MAX_ZOOM);
                zoom_refusals += 1;
            }
            Err(PmTilesError::CoordOutOfRange { z, x, y, side }) => {
                assert_eq!((z, x, y), (row.z, row.x, row.y));
                assert_eq!(side, 1u64 << row.z);
                coord_refusals += 1;
            }
            other => panic!(
                "({}, {}, {}) must be refused by name, not answered; go-pmtiles masks it to {}. Got {other:?}",
                row.z, row.x, row.y, row.tile_id
            ),
        }
    }
    // Both refusals are represented, so a single over-broad check cannot be
    // carrying the whole table.
    assert_eq!(
        zoom_refusals, 4,
        "four of the ten rows are out-of-range zooms"
    );
    assert_eq!(
        coord_refusals, 6,
        "six of the ten rows are out-of-range coordinates"
    );

    // The tightest positive control there is, and it costs nothing because the
    // oracle already measured it: the tile the reference silently substituted
    // is a real tile, and we address it correctly. `roundtrip_z/x/y` on a
    // masked row is `IDToZxy` of the id the reference answered, so asserting
    // `zxy_to_tileid(that) == that id` says the refusal above is about the
    // out-of-range input and not about the id or the level.
    let mut substituted = 0;
    let mut unaddressable = 0;
    for row in &rows {
        if row.z > MAX_ZOOM {
            // The four saturating-zoom rows have no substituted tile to check:
            // `ZxyToID` computes `(1 << (z * 2) - 1) / 3` on a `uint8` z, so
            // `z * 2` wraps at 256 and the answer is `(2^64 - 1) / 3`, which is
            // the first id past what a `u64` can address. Three of them decode
            // back to "zoom 127" and the fourth to zoom 0, neither of which is
            // the tile that was asked for. Our side refuses the id too, and
            // that is asserted by name rather than left implied.
            assert!(
                matches!(
                    tileid_to_zxy(row.tile_id),
                    Err(PmTilesError::TileIdOutOfRange { .. })
                ),
                "the id go-pmtiles saturates to ({}) is not addressable",
                row.tile_id
            );
            unaddressable += 1;
            continue;
        }
        let (rz, rx, ry) = row.roundtrip;
        let (rz, rx, ry) = (rz as u8, rx as u32, ry as u32);
        assert_eq!(
            zxy_to_tileid(rz, rx, ry).expect("the substituted tile is a real tile"),
            row.tile_id,
            "({rz}, {rx}, {ry}) is the tile go-pmtiles answered with, and it must address"
        );
        substituted += 1;
    }
    assert_eq!(
        substituted, 6,
        "the substituted-tile control ran on the wrong number of rows"
    );
    assert_eq!(
        unaddressable, 4,
        "four rows saturate the zoom rather than mask a coordinate"
    );

    // And the broader control: 76 in-range rows in the same file still answer,
    // so "everything is refused" cannot pass here either. Deliberately the two
    // sections that do not depend on the curve's orientation, so a convention
    // bug reddens the convention tests and not this one, which is about
    // refusal.
    let mut answered = 0;
    for section in ["first_and_last_of_level", "orientation_boundaries"] {
        for row in oracle::tile_id_rows(&vectors, section) {
            assert_eq!(
                zxy_to_tileid(row.z, row.x, row.y).expect("a live oracle row still answers"),
                row.tile_id
            );
            answered += 1;
        }
    }
    assert_eq!(
        answered,
        32 + 44,
        "the positive control ran on nothing like the whole file"
    );
}

/// The sha256 this suite expects of each golden, by file name.
fn golden_sha256(name: &str) -> &'static str {
    match name {
        "raster-z0z2.pmtiles" => oracle::RASTER_GOLDEN_SHA256,
        "dupes-z0z3.pmtiles" => oracle::DUPES_GOLDEN_SHA256,
        "leaves-z0z7.pmtiles" => oracle::LEAVES_GOLDEN_SHA256,
        "distinct-z0z7.pmtiles" => oracle::DISTINCT_GOLDEN_SHA256,
        other => panic!("no pinned sha256 for the golden {other:?}"),
    }
}

/// Every field of every golden's header, against what
/// `pmtiles.DeserializeHeader` reported for the same 127 bytes, and then the
/// encoder against those bytes again.
///
/// The re-encode is the half that makes this a writer test as well as a reader
/// test. `Header::encode` has to reproduce the reference's bytes exactly, so a
/// field written at the wrong offset, or in the wrong endianness, fails here
/// even though a decode-then-encode round trip inside this crate would not
/// notice either.
///
/// The header is little-endian throughout, `u64` and `i32` alike, and the four
/// bounds fields are what prove it: `min_lon_e7` is -1800000000 and it decodes
/// little-endian at offset 102 and big-endian nowhere in the buffer. The two
/// centre fields are **zero** in all four goldens and therefore match at about
/// fifty offsets in both endiannesses, which is the zero-has-two-explanations
/// trap in its purest form. A check built on the centre alone proves nothing.
///
/// The bounds are the same whole world in all four as well, which is worth
/// saying because it bounds what this cell can show: it pins the decoder and
/// the encoder against each other and against the reference, on values that
/// happen to be symmetric. What it cannot see is the writer choosing the wrong
/// field for a value nobody recorded, and
/// `pmtiles_writer::the_bounds_the_writer_is_given_are_the_bounds_the_archive_carries`
/// is the cell for that.
#[test]
#[cfg_attr(miri, ignore)]
fn every_golden_header_decodes_and_re_encodes_byte_for_byte() {
    let vectors = oracle::header_vectors();
    let archives = vectors["archives"]
        .as_object()
        .expect("header.json has an archives object");
    assert_eq!(archives.len(), 4, "four goldens are pinned");

    let mut checked = 0;
    for (name, entry) in archives {
        let hex = entry["header_bytes_hex"]
            .as_str()
            .expect("header_bytes_hex is a string");
        let bytes = oracle::unhex(hex);
        assert_eq!(
            bytes.len(),
            HEADER_BYTES,
            "{name}'s header is not 127 bytes"
        );

        // The vector file and the archive have to agree about the archive,
        // which is a cross-check between two fixtures rather than an
        // assertion about our code.
        let archive = oracle::golden(name, golden_sha256(name));
        assert_eq!(
            archive.len() as u64,
            entry["golden_size"].as_u64().expect("golden_size"),
            "{name} is not the size the vectors say"
        );
        assert_eq!(
            &archive[..HEADER_BYTES],
            &bytes[..],
            "{name}'s first 127 bytes are not the header the vectors carry"
        );

        let header = Header::try_decode(&bytes).expect("a real archive's header decodes");
        let want = &entry["decoded_by_pmtiles_DeserializeHeader"];

        let u64_field = |key: &str| want[key].as_u64().unwrap_or_else(|| panic!("{name}.{key}"));
        let i64_field = |key: &str| want[key].as_i64().unwrap_or_else(|| panic!("{name}.{key}"));

        assert_eq!(header.root_offset, u64_field("root_offset"), "{name}");
        assert_eq!(header.root_length, u64_field("root_length"), "{name}");
        assert_eq!(
            header.metadata_offset,
            u64_field("metadata_offset"),
            "{name}"
        );
        assert_eq!(
            header.metadata_length,
            u64_field("metadata_length"),
            "{name}"
        );
        assert_eq!(
            header.leaf_directories_offset,
            u64_field("leaf_directory_offset"),
            "{name}"
        );
        assert_eq!(
            header.leaf_directories_length,
            u64_field("leaf_directory_length"),
            "{name}"
        );
        assert_eq!(
            header.tile_data_offset,
            u64_field("tile_data_offset"),
            "{name}"
        );
        assert_eq!(
            header.tile_data_length,
            u64_field("tile_data_length"),
            "{name}"
        );
        assert_eq!(
            header.addressed_tiles_count,
            u64_field("addressed_tiles_count"),
            "{name}"
        );
        assert_eq!(
            header.tile_entries_count,
            u64_field("tile_entries_count"),
            "{name}"
        );
        assert_eq!(
            header.tile_contents_count,
            u64_field("tile_contents_count"),
            "{name}"
        );
        assert_eq!(
            header.clustered,
            want["clustered"].as_bool().expect("clustered"),
            "{name}"
        );
        assert_eq!(
            header.internal_compression,
            Compression::from_byte(u64_field("internal_compression") as u8),
            "{name}"
        );
        assert_eq!(
            header.tile_compression,
            Compression::from_byte(u64_field("tile_compression") as u8),
            "{name}"
        );
        assert_eq!(
            header.tile_type,
            TileType::from_byte(u64_field("tile_type") as u8),
            "{name}"
        );
        assert_eq!(header.min_zoom, u64_field("min_zoom") as u8, "{name}");
        assert_eq!(header.max_zoom, u64_field("max_zoom") as u8, "{name}");
        assert_eq!(header.center_zoom, u64_field("center_zoom") as u8, "{name}");
        assert_eq!(
            i64::from(header.min_lon_e7),
            i64_field("min_lon_e7"),
            "{name}"
        );
        assert_eq!(
            i64::from(header.min_lat_e7),
            i64_field("min_lat_e7"),
            "{name}"
        );
        assert_eq!(
            i64::from(header.max_lon_e7),
            i64_field("max_lon_e7"),
            "{name}"
        );
        assert_eq!(
            i64::from(header.max_lat_e7),
            i64_field("max_lat_e7"),
            "{name}"
        );
        assert_eq!(
            i64::from(header.center_lon_e7),
            i64_field("center_lon_e7"),
            "{name}"
        );
        assert_eq!(
            i64::from(header.center_lat_e7),
            i64_field("center_lat_e7"),
            "{name}"
        );

        // The four bounds fields are the positive control for the endianness,
        // because they are the only position fields in these archives that are
        // not zero.
        assert_ne!(
            header.min_lon_e7, 0,
            "{name}'s bounds are all zero, so this archive cannot settle the endianness"
        );
        assert_ne!(
            header.max_lat_e7, 0,
            "{name}'s bounds are all zero, so this archive cannot settle the endianness"
        );

        assert_eq!(
            header.encode().to_vec(),
            bytes,
            "{name} does not re-encode to the reference's bytes"
        );
        checked += 1;
    }
    assert_eq!(
        checked, 4,
        "the header sweep did not visit all four goldens"
    );
}

/// The root directory of the two flat goldens, decoded and re-serialised
/// against go-pmtiles' own bytes.
///
/// `reserialize_roundtrip_ok` in the vectors records that go-pmtiles'
/// `SerializeEntries` reproduces `decompressed_hex` exactly from the entries
/// `DeserializeEntries` had just decoded, so those bytes are a legitimate
/// byte-for-byte target for a serializer rather than a convenience for a
/// parser. That is also what makes this the whole no-zigzag proof: if any
/// column here were zigzagged, or if the offset column were delta encoded
/// rather than storing `offset + 1` with a literal `0` meaning "continues
/// where the last entry ended", the bytes would differ and this would fail.
#[test]
#[cfg_attr(miri, ignore)]
fn the_golden_root_directories_decode_and_re_serialise_byte_for_byte() {
    let vectors = oracle::directory_vectors();
    let archives = vectors["archives"]
        .as_object()
        .expect("directory.json has an archives object");
    assert_eq!(archives.len(), 2, "two flat goldens are pinned here");

    let mut checked = 0;
    for (name, entry) in archives {
        let root = &entry["root_directory"];
        let raw = oracle::unhex(root["raw_gzip_hex"].as_str().expect("raw_gzip_hex"));
        let plain = oracle::unhex(root["decompressed_hex"].as_str().expect("decompressed_hex"));
        assert_eq!(
            plain.len() as u64,
            root["decompressed_len"].as_u64().expect("decompressed_len"),
            "{name}: the decompressed hex is not the length the vectors say"
        );

        // The archive itself carries those compressed bytes where the header
        // says it does, which ties the vector file to the file it describes.
        let archive = oracle::golden(name, golden_sha256(name));
        let at = root["absolute_offset"].as_u64().expect("absolute_offset") as usize;
        let len = root["compressed_length"]
            .as_u64()
            .expect("compressed_length") as usize;
        assert_eq!(
            &archive[at..at + len],
            &raw[..],
            "{name}: the archive's root directory bytes are not the ones the vectors carry"
        );
        assert_eq!(
            at as u64,
            entry["header_root_offset"]
                .as_u64()
                .expect("header_root_offset"),
            "{name}: the root directory is not where the header says"
        );

        // Gunzip through this crate's own bounded decompressor.
        assert_eq!(
            Compression::Gzip
                .decompress(&raw, DECOMPRESS_CEILING)
                .expect("a real archive's root directory decompresses"),
            plain,
            "{name}: gunzip did not reproduce the reference's plain bytes"
        );

        let entries = deserialize_entries(&plain).expect("a real archive's root directory parses");
        let want = root["entries"].as_array().expect("entries is an array");
        assert_eq!(
            entries.len(),
            want.len(),
            "{name}: entry count disagrees with the vectors"
        );
        assert_eq!(
            entries.len() as u64,
            root["entry_count"].as_u64().expect("entry_count"),
            "{name}: entry count disagrees with the file's own count"
        );
        assert_eq!(
            entries.len() as u64,
            entry["header_tile_entries_count"]
                .as_u64()
                .expect("header_tile_entries_count"),
            "{name}: entry count disagrees with the header"
        );

        for (index, (got, expected)) in entries.iter().zip(want).enumerate() {
            assert_eq!(
                *got,
                Entry {
                    tile_id: expected["tile_id"].as_u64().expect("tile_id"),
                    offset: expected["offset"].as_u64().expect("offset"),
                    length: expected["length"].as_u64().expect("length") as u32,
                    run_length: expected["run_length"].as_u64().expect("run_length") as u32,
                },
                "{name}: entry {index}"
            );
        }

        // The three header counts, each a different question about the same
        // entry list. Run lengths sum to the addressed tiles; distinct offsets
        // count the tile contents.
        let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
        assert_eq!(
            addressed,
            entry["header_addressed_tiles_count"]
                .as_u64()
                .expect("header_addressed_tiles_count"),
            "{name}: the run lengths do not sum to the addressed tile count"
        );
        let distinct: std::collections::BTreeSet<u64> = entries.iter().map(|e| e.offset).collect();
        assert_eq!(
            distinct.len() as u64,
            entry["header_tile_contents_count"]
                .as_u64()
                .expect("header_tile_contents_count"),
            "{name}: the distinct offsets do not count the tile contents"
        );

        assert!(
            root["reserialize_roundtrip_ok"]
                .as_bool()
                .expect("reserialize_roundtrip_ok"),
            "{name}: the vectors do not claim these bytes are a serializer target"
        );
        assert_eq!(
            serialize_entries(&entries).expect("the entries serialise"),
            plain,
            "{name}: this crate's serializer does not reproduce the reference's bytes"
        );
        checked += 1;
    }
    assert_eq!(checked, 2, "the directory sweep did not visit both goldens");
}

/// Both shapes deduplication takes, from `dupes-z0z3.pmtiles`.
///
/// `convert` collapses identical tiles two different ways and a reader that
/// handles one and not the other looks correct on most archives. Identical
/// tiles at consecutive ids become one entry with `run_length > 1`; identical
/// tiles that are not consecutive stay separate entries pointing at the same
/// offset. This golden has both on purpose, and the second is the one that is
/// easy to miss.
#[test]
#[cfg_attr(miri, ignore)]
fn the_dupes_golden_carries_both_shapes_of_deduplication() {
    let vectors = oracle::directory_vectors();
    let entry = &vectors["archives"]["dupes-z0z3.pmtiles"];
    let plain = oracle::unhex(
        entry["root_directory"]["decompressed_hex"]
            .as_str()
            .expect("decompressed_hex"),
    );
    let entries = deserialize_entries(&plain).expect("the dupes root directory parses");

    // Shape one: runs. 67 entries covering 85 addressed tiles.
    let runs: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.run_length).collect();
    assert_eq!(
        runs,
        [1u32, 4, 16].into_iter().collect(),
        "the run lengths are not the three the oracle measured"
    );
    assert!(entries.len() < 85, "nothing collapsed into a run at all");

    // Shape two: one offset shared by four entries that are nowhere near each
    // other. Their ids are 21, 49, 63 and 76, so no run could ever form.
    let shared: Vec<(usize, u64)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.offset == 148)
        .map(|(i, e)| (i, e.tile_id))
        .collect();
    assert_eq!(
        shared,
        vec![(3, 21), (31, 49), (45, 63), (58, 76)],
        "offset 148 is not shared by the four non-adjacent entries"
    );
    for (index, _) in &shared {
        assert_eq!(
            entries[*index].run_length, 1,
            "a shared-offset entry is a run"
        );
    }

    // And the backwards offset jump, which is where a signed or zigzagged
    // offset column would have to show itself. Entry 1 sits at 74 and entry 2
    // at 0. The raw uvarint in the bytes for entry 2 is 1, not the 147 a
    // zigzagged -74 would be, and the byte-for-byte re-serialisation in
    // `the_golden_root_directories_decode_and_re_serialise_byte_for_byte` is
    // what holds that claim.
    assert!(
        entries[2].offset < entries[1].offset,
        "the golden has lost its backwards offset jump"
    );
}

/// The leaf golden, at the directory-codec level.
///
/// `leaves-z0z7.pmtiles` is the only fixture anywhere with real leaf
/// directories, so it is the only thing that can catch a writer and a reader
/// agreeing on the wrong offset base. Traversal belongs to the reader (#988);
/// what this asserts is the half the codec owns: 6 pointer entries with
/// `run_length == 0`, 21844 tile entries spread over six leaves, every leaf
/// re-serialising to the reference's own bytes, and the arithmetic that says
/// the tile entries inside those leaves are relative to `tile_data_offset`
/// rather than to the leaf or to the leaf section.
#[test]
#[cfg_attr(miri, ignore)]
fn the_leaf_golden_decodes_and_re_serialises_every_one_of_its_six_leaves() {
    let vectors = oracle::directory_leaves_vectors();
    let header_v = &vectors["header"];
    let leaf_section_offset = header_v["leaf_directory_offset"].as_u64().expect("offset");
    let leaf_section_length = header_v["leaf_directory_length"].as_u64().expect("length");
    let tile_data_offset = header_v["tile_data_offset"]
        .as_u64()
        .expect("tile_data_offset");
    let tile_data_length = header_v["tile_data_length"]
        .as_u64()
        .expect("tile_data_length");
    let archive = oracle::golden("leaves-z0z7.pmtiles", oracle::LEAVES_GOLDEN_SHA256);

    // The root is six leaf pointers and nothing else.
    let root = &vectors["root_directory"];
    let root_plain = oracle::unhex(root["decompressed_hex"].as_str().expect("decompressed_hex"));
    let pointers = deserialize_entries(&root_plain).expect("the leaf golden's root parses");
    assert_eq!(pointers.len(), 6, "the root is six leaf pointers");
    for (index, pointer) in pointers.iter().enumerate() {
        assert_eq!(
            pointer.run_length, 0,
            "root entry {index} is not a leaf pointer"
        );
        assert!(
            pointer.is_leaf(),
            "root entry {index} does not report as a leaf"
        );
    }
    assert_eq!(
        serialize_entries(&pointers).expect("the pointers serialise"),
        root_plain,
        "the root of the leaf golden does not re-serialise to the reference's bytes"
    );

    // Every leaf: the pointer says where it is, the archive has those bytes,
    // they gunzip, they parse, and they serialise back byte for byte.
    let leaves = vectors["leaf_directories"]
        .as_array()
        .expect("leaf_directories is an array");
    assert_eq!(leaves.len(), 6, "six leaves");

    let mut total_entries = 0usize;
    let mut lengths_sum = 0u64;
    let mut tile_offsets = std::collections::BTreeSet::new();
    for (index, leaf) in leaves.iter().enumerate() {
        let relative = leaf["offset_in_leaf_section"]
            .as_u64()
            .expect("relative offset");
        let absolute = leaf["absolute_offset"].as_u64().expect("absolute offset");
        let compressed = leaf["compressed_length"]
            .as_u64()
            .expect("compressed_length");

        // The pointer's offset is relative to the leaf section, not to the
        // file. This is the arithmetic a reader gets wrong, stated here where
        // the codec can see it.
        assert_eq!(
            pointers[index].offset, relative,
            "leaf {index}: the pointer does not carry the section-relative offset"
        );
        assert_eq!(
            leaf_section_offset + relative,
            absolute,
            "leaf {index}: section offset plus relative offset is not the absolute offset"
        );
        assert_eq!(
            u64::from(pointers[index].length),
            compressed,
            "leaf {index}: the pointer's length is not the leaf's compressed length"
        );

        let raw = oracle::unhex(leaf["raw_gzip_hex"].as_str().expect("raw_gzip_hex"));
        let plain = oracle::unhex(leaf["decompressed_hex"].as_str().expect("decompressed_hex"));
        assert_eq!(
            &archive[absolute as usize..(absolute + compressed) as usize],
            &raw[..],
            "leaf {index}: the archive does not hold these bytes there"
        );
        assert_eq!(
            Compression::Gzip
                .decompress(&raw, DECOMPRESS_CEILING)
                .expect("a leaf decompresses"),
            plain,
            "leaf {index}: gunzip did not reproduce the reference's plain bytes"
        );

        let entries = deserialize_entries(&plain).expect("a leaf parses");
        let columnar = &leaf["entries_columnar"];
        let ids = columnar["tile_id"].as_array().expect("tile_id column");
        let offsets = columnar["offset"].as_array().expect("offset column");
        let lengths = columnar["length"].as_array().expect("length column");
        let runs = columnar["run_length"]
            .as_array()
            .expect("run_length column");
        assert_eq!(
            entries.len(),
            leaf["entry_count"].as_u64().expect("entry_count") as usize,
            "leaf {index}: entry count"
        );
        assert_eq!(entries.len(), ids.len(), "leaf {index}: id column length");

        for (position, got) in entries.iter().enumerate() {
            assert_eq!(
                *got,
                Entry {
                    tile_id: ids[position].as_u64().expect("tile_id"),
                    offset: offsets[position].as_u64().expect("offset"),
                    length: lengths[position].as_u64().expect("length") as u32,
                    run_length: runs[position].as_u64().expect("run_length") as u32,
                },
                "leaf {index}: entry {position}"
            );
            // Every entry inside a leaf here is a tile entry, and a tile entry
            // is relative to `tile_data_offset` whatever directory it was found
            // in. The proof is arithmetic rather than assertion: the archive's
            // whole tile data section is 144 bytes, so an offset that fits
            // inside it under this base and would land in the leaf directory
            // region under either tempting wrong base is not ambiguous.
            assert_ne!(
                got.run_length, 0,
                "leaf {index}: entry {position} is a leaf pointer inside a leaf"
            );
            assert!(
                got.offset + u64::from(got.length) <= tile_data_length,
                "leaf {index}: entry {position} runs past the tile data section"
            );
            tile_offsets.insert(got.offset);
        }

        assert_eq!(
            serialize_entries(&entries).expect("a leaf serialises"),
            plain,
            "leaf {index}: this crate's serializer does not reproduce the reference's bytes"
        );

        total_entries += entries.len();
        lengths_sum += compressed;
    }

    assert_eq!(
        total_entries as u64,
        header_v["tile_entries_count"]
            .as_u64()
            .expect("tile_entries_count"),
        "the six leaves do not hold the header's entry count"
    );
    assert_eq!(total_entries, 21844, "21844 entries over six leaves");
    assert_eq!(
        lengths_sum, leaf_section_length,
        "the six leaf lengths do not sum to the leaf section length"
    );
    assert_eq!(
        leaf_section_offset + leaf_section_length,
        tile_data_offset,
        "the leaf section does not end where the tile data begins"
    );
    // Two payloads, at 0 and 72, inside a 144-byte tile data section. Under a
    // leaf-relative or leaf-section-relative base these would land in the
    // directory region instead.
    assert_eq!(
        tile_offsets.iter().copied().collect::<Vec<u64>>(),
        vec![0, 72],
        "the leaf tile entries do not point at the two payloads"
    );
    assert_eq!(
        header_v["tile_contents_count"]
            .as_u64()
            .expect("tile_contents_count"),
        2,
        "the archive is supposed to hold two distinct payloads"
    );
}
