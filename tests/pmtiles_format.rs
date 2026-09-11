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
