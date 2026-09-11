//! The structural validator, against the `go-pmtiles` goldens and against
//! archives deliberately broken in the ways the reference does not survive
//! (issue #991).
//!
//! `libviprs::pmtiles::validate` is the core of the future
//! `viprs pmtiles verify`, so it is written to answer "what is wrong with this
//! archive" rather than "is it broken": it collects findings and keeps
//! walking, and only an I/O failure in the underlying reader stops it. A
//! validator that returns on the first problem tells an operator about one
//! symptom of a file with six.
//!
//! # What it has to be better at than the reference
//!
//! `pmtiles verify` bounds-checks four *lengths* against the file size and
//! never checks `offset + length`. Measured on the oracle: setting
//! `root_offset` to 999999 in an 1878-byte archive passes every one of its
//! checks, because the root *length* is still 35 and the whole-archive size
//! arithmetic still adds up. It then walks to that offset, hands bytes that do
//! not start with a gzip header to `DeserializeEntries`, which does
//! `reader, _ = gzip.NewReader(data)` and drops the error, and the next read
//! dereferences a nil pointer. Exit code 2, `panic: runtime error: invalid
//! memory address or nil pointer dereference`, out of `compress/gzip`.
//!
//! So [`the_root_offset_that_walks_past_the_end_of_the_file_is_flagged`] is
//! the test that matters most in this file. Every section check here is
//! `offset.checked_add(length) <= size`, and a directory that is not
//! decompressible is a typed finding rather than a crash.
//!
//! # The leaf offset base, which nothing else in the world tests
//!
//! A root entry with `run_length == 0` is a leaf pointer and its `offset` is
//! relative to `header.leaf_directories_offset`, not to the start of the file
//! and not to the start of the root directory. `leaves-z0z7.pmtiles` is the
//! only fixture anywhere that exercises it, the PMTiles specification
//! repository's own fixtures included, and it is the one place where a writer
//! and a reader that make the same wrong choice round-trip perfectly and are
//! both wrong. [`the_leaf_pointers_resolve_relative_to_the_leaf_section`]
//! pins the six absolute offsets against what `go-pmtiles` reported, so the
//! agreement is with something outside this crate.
//!
//! # What this file does not do
//!
//! It does not re-pin the directory bytes or the tile id mapping.
//! `tests/pmtiles_format.rs` (issue #987) owns the oracle vectors and pins
//! those, and duplicating them here would be two copies of one claim that can
//! drift apart. Everything here runs through [`validate_bytes`], so a test
//! that mentions a golden is asserting something about the *walk*.

use libviprs::pmtiles::validate::{
    DirectoryRef, Finding, Section, ValidationLimits, validate_bytes,
};
use libviprs::pmtiles::{Compression, Header};
use serde_json::Value;

#[path = "common/pmtiles_oracle.rs"]
mod pmtiles_oracle;

use pmtiles_oracle::{
    DUPES_GOLDEN_SHA256, LEAVES_GOLDEN_SHA256, RASTER_GOLDEN_SHA256, directory_leaves_vectors,
    golden,
};

/// The three goldens with the digests the shared loader pins them by.
const GOLDENS: &[(&str, &str)] = &[
    ("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256),
    ("dupes-z0z3.pmtiles", DUPES_GOLDEN_SHA256),
    ("leaves-z0z7.pmtiles", LEAVES_GOLDEN_SHA256),
];

/// Where the 64-bit little-endian header fields sit, for the tests that break
/// one on purpose. From the v3 layout: seven bytes of magic, a version byte,
/// then eleven `u64`s.
const OFF_ROOT_OFFSET: usize = 8;
const OFF_ROOT_LENGTH: usize = 16;
const OFF_LEAF_OFFSET: usize = 40;
const OFF_ADDRESSED_TILES: usize = 72;

/// Overwrite a little-endian `u64` in a copy of an archive.
fn with_u64(bytes: &[u8], offset: usize, value: u64) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    out
}

/// One leaf directory as the oracle dumped it.
///
/// Parsed here rather than in `tests/common/pmtiles_oracle.rs` because this is
/// the only caller: the shared loader is read by three lanes at once and a
/// helper only one of them uses is a merge conflict for the other two.
#[derive(Debug, Clone)]
struct OracleLeaf {
    offset_in_section: u64,
    absolute_offset: u64,
    compressed_length: u64,
    /// `(tile_id, offset, length, run_length)` per entry.
    entries: Vec<(u64, u64, u32, u32)>,
}

fn u64_at(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("a leaf block has no unsigned {key}"))
}

/// Every leaf of `leaves-z0z7.pmtiles`, with each leaf's entry count checked
/// against the `entry_count` the dump program wrote beside it.
///
/// That cross-check is the positive control: a parse that silently produced
/// empty columns would make every comparison in the caller's loop vacuous.
fn oracle_leaves() -> Vec<OracleLeaf> {
    let doc = directory_leaves_vectors();
    let leaves = doc
        .get("leaf_directories")
        .and_then(Value::as_array)
        .expect("directory-leaves.json has a leaf_directories array");
    assert!(
        !leaves.is_empty(),
        "the leaf-bearing golden parsed to zero leaves, so nothing below could fail"
    );

    leaves
        .iter()
        .map(|leaf| {
            let declared =
                usize::try_from(u64_at(leaf, "entry_count")).expect("a count fits in a usize");
            let columns = leaf
                .get("entries_columnar")
                .expect("a leaf block has entries_columnar");
            let column = |name: &str| -> Vec<u64> {
                columns
                    .get(name)
                    .and_then(Value::as_array)
                    .unwrap_or_else(|| panic!("a leaf block has no {name} column"))
                    .iter()
                    .map(|v| v.as_u64().expect("a column holds unsigned integers"))
                    .collect()
            };
            let ids = column("tile_id");
            let offsets = column("offset");
            let lengths = column("length");
            let runs = column("run_length");
            assert_eq!(
                ids.len(),
                declared,
                "a leaf's tile_id column has {} values against a declared entry_count of \
                 {declared}",
                ids.len()
            );
            for (name, col) in [
                ("offset", &offsets),
                ("length", &lengths),
                ("run_length", &runs),
            ] {
                assert_eq!(
                    col.len(),
                    ids.len(),
                    "a leaf's {name} column is a different length from its tile_id column"
                );
            }
            OracleLeaf {
                offset_in_section: u64_at(leaf, "offset_in_leaf_section"),
                absolute_offset: u64_at(leaf, "absolute_offset"),
                compressed_length: u64_at(leaf, "compressed_length"),
                entries: (0..ids.len())
                    .map(|i| {
                        (
                            ids[i],
                            offsets[i],
                            u32::try_from(lengths[i]).expect("a length fits in a u32"),
                            u32::try_from(runs[i]).expect("a run length fits in a u32"),
                        )
                    })
                    .collect(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The goldens
// ---------------------------------------------------------------------------

/// All three archives `go-pmtiles` wrote validate with no findings, and the
/// totals the walk counted are the ones their headers claim.
#[test]
#[cfg_attr(miri, ignore)]
fn every_golden_archive_validates_clean() {
    let mut checked = 0;
    for (name, sha) in GOLDENS {
        let bytes = golden(name, sha);
        let report = validate_bytes(&bytes, &ValidationLimits::default())
            .unwrap_or_else(|e| panic!("{name}: the validator could not read the archive: {e}"));
        assert!(
            report.findings.is_empty(),
            "{name} is an archive the reference implementation wrote and it produced findings: \
             {:?}",
            report.findings
        );
        assert!(report.is_valid());

        let header = report.header.expect("a clean archive has a header");
        assert_eq!(
            report.tile_entries, header.tile_entries_count,
            "{name}: counted entries against the header's tile_entries_count"
        );
        assert_eq!(
            report.addressed_tiles, header.addressed_tiles_count,
            "{name}: the run lengths must sum to addressed_tiles_count"
        );
        assert_eq!(
            report.tile_contents,
            Some(header.tile_contents_count),
            "{name}: distinct entry offsets against tile_contents_count"
        );
        checked += 1;
    }
    assert_eq!(checked, 3, "all three goldens were validated");
}

/// The leaf-bearing golden is walked through its six leaves, and every leaf
/// lands where `go-pmtiles` says it does.
///
/// The pointers carry 0, 71, 136, 202, 268 and 333, which are offsets into the
/// leaf section rather than into the file. Read as absolute they would point
/// into the header, which at least fails loudly; the failure this catches is
/// the quiet one, where a writer emits them absolute and a reader reads them
/// absolute and the pair agrees with itself.
#[test]
#[cfg_attr(miri, ignore)]
fn the_leaf_pointers_resolve_relative_to_the_leaf_section() {
    let bytes = golden("leaves-z0z7.pmtiles", LEAVES_GOLDEN_SHA256);
    let report = validate_bytes(&bytes, &ValidationLimits::default()).expect("the archive reads");
    assert!(report.findings.is_empty(), "{:?}", report.findings);

    let oracle = oracle_leaves();
    assert_eq!(oracle.len(), 6, "the oracle recorded six leaves");
    assert_eq!(
        report.leaves.len(),
        oracle.len(),
        "the validator found {} leaves against the oracle's {}",
        report.leaves.len(),
        oracle.len()
    );

    let header = report.header.expect("a clean archive has a header");
    assert_eq!(header.leaf_directories_offset, 334);
    assert_eq!(header.leaf_directories_length, 391);

    let mut lengths = 0u64;
    for (found, expected) in report.leaves.iter().zip(&oracle) {
        assert_eq!(
            found.offset_in_section, expected.offset_in_section,
            "a leaf pointer's relative offset"
        );
        assert_eq!(
            found.absolute_offset, expected.absolute_offset,
            "a leaf pointer resolved to the wrong place in the file"
        );
        assert_eq!(
            found.absolute_offset,
            header.leaf_directories_offset + expected.offset_in_section,
            "the base for a leaf offset is header.leaf_directories_offset"
        );
        assert_eq!(found.compressed_length, expected.compressed_length);
        assert_eq!(found.entries, expected.entries.len());
        lengths += found.compressed_length;
    }

    // Two totals the goldens' provenance calls out, and both are cheap:
    assert_eq!(
        lengths, header.leaf_directories_length,
        "the six leaf lengths must sum to leaf_directories_length"
    );
    assert_eq!(
        header.leaf_directories_offset + header.leaf_directories_length,
        header.tile_data_offset,
        "the leaf section ends where the tile data begins"
    );
    assert_eq!(report.tile_entries, 21844);
    assert_eq!(report.addressed_tiles, 21845);
}

/// Every one of the 21844 entries the walk pulled out of the six leaves is the
/// entry `go-pmtiles` decoded there.
///
/// A per-entry comparison rather than a total, because a total is satisfied by
/// two errors that cancel. This is about the *walk* rather than the directory
/// parser: `tests/pmtiles_format.rs` already pins the leaf bytes, and what is
/// being checked here is that following six pointers lands on the right six
/// directories in the right order.
#[test]
#[cfg_attr(miri, ignore)]
fn the_walked_leaf_entries_are_the_ones_go_pmtiles_decoded() {
    let bytes = golden("leaves-z0z7.pmtiles", LEAVES_GOLDEN_SHA256);
    let report = validate_bytes(
        &bytes,
        &ValidationLimits::default().with_collect_entries(true),
    )
    .expect("the archive reads");

    let oracle = oracle_leaves();
    let mut compared = 0usize;
    for (index, (found, expected)) in report.leaves.iter().zip(&oracle).enumerate() {
        let decoded = found
            .decoded
            .as_ref()
            .expect("collect_entries was asked for");
        assert_eq!(
            decoded.len(),
            expected.entries.len(),
            "leaf {index} decoded to {} entries against the oracle's {}",
            decoded.len(),
            expected.entries.len()
        );
        for (position, (got, want)) in decoded.iter().zip(&expected.entries).enumerate() {
            assert_eq!(
                (got.tile_id, got.offset, got.length, got.run_length),
                *want,
                "leaf {index} entry {position}"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared, 21844,
        "the comparison covered {compared} entries, not the whole archive"
    );
}

/// The walk's three totals on the archive that deduplicates two different
/// ways.
///
/// Runs collapse identical *consecutive* tiles into one entry with
/// `run_length > 1`; identical tiles that are not consecutive stay separate
/// entries with `run_length` 1 pointing at the same offset. A walk that
/// handled only the first would count 85 entries rather than 67, and one that
/// handled only the second would count 63 addressed tiles rather than 85.
#[test]
#[cfg_attr(miri, ignore)]
fn the_walk_counts_both_shapes_of_deduplication() {
    let report = validate_bytes(
        &golden("dupes-z0z3.pmtiles", DUPES_GOLDEN_SHA256),
        &ValidationLimits::default(),
    )
    .expect("the archive reads");
    assert!(report.findings.is_empty(), "{:?}", report.findings);
    assert_eq!(
        report.root_entries, 67,
        "entries after run-length collapsing"
    );
    assert_eq!(report.addressed_tiles, 85, "tiles the archive addresses");
    assert_eq!(report.tile_contents, Some(63), "distinct tile payloads");
}

// ---------------------------------------------------------------------------
// Archives broken on purpose
// ---------------------------------------------------------------------------

/// The one the reference crashes on.
///
/// A root offset of 999999 in an 1878-byte file. `pmtiles verify` checks
/// `root_length > fileSize` and nothing about the offset, so it accepts this
/// and then segfaults inside gzip. This must be a finding that names the
/// section, the offset and the size.
#[test]
#[cfg_attr(miri, ignore)]
fn the_root_offset_that_walks_past_the_end_of_the_file_is_flagged() {
    let broken = with_u64(
        &golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256),
        OFF_ROOT_OFFSET,
        999_999,
    );
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("bytes are readable");
    assert!(
        report.findings.contains(&Finding::SectionOutOfBounds {
            section: Section::Root,
            offset: 999_999,
            length: 35,
            archive_size: 1878,
        }),
        "a root offset past the end of the file was not flagged: {:?}",
        report.findings
    );
    assert!(!report.is_valid());
}

/// `offset + length` overflowing a `u64` is a finding, not a wrap.
#[test]
#[cfg_attr(miri, ignore)]
fn a_section_whose_offset_plus_length_overflows_is_flagged() {
    let broken = with_u64(
        &with_u64(
            &golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256),
            OFF_ROOT_OFFSET,
            u64::MAX - 4,
        ),
        OFF_ROOT_LENGTH,
        64,
    );
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("bytes are readable");
    assert!(
        report.findings.contains(&Finding::SectionOverflow {
            section: Section::Root,
            offset: u64::MAX - 4,
            length: 64,
        }),
        "an overflowing section was not flagged: {:?}",
        report.findings
    );
}

/// A root length larger than the whole file, which is the one shape the
/// reference does catch.
#[test]
#[cfg_attr(miri, ignore)]
fn a_root_length_larger_than_the_archive_is_flagged() {
    let broken = with_u64(
        &golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256),
        OFF_ROOT_LENGTH,
        1 << 40,
    );
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("bytes are readable");
    assert!(
        report.findings.iter().any(|f| matches!(
            f,
            Finding::SectionOutOfBounds {
                section: Section::Root,
                ..
            }
        )),
        "{:?}",
        report.findings
    );
}

/// A truncated archive is flagged on the section that no longer fits, and the
/// header itself still decodes.
#[test]
#[cfg_attr(miri, ignore)]
fn a_truncated_archive_is_flagged() {
    let full = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    let report =
        validate_bytes(&full[..900], &ValidationLimits::default()).expect("bytes are readable");
    assert!(
        report.findings.iter().any(|f| matches!(
            f,
            Finding::SectionOutOfBounds {
                section: Section::TileData,
                ..
            }
        )),
        "a file cut off inside its tile data was not flagged: {:?}",
        report.findings
    );
    assert!(report.header.is_some(), "the header is still readable");
}

/// Bytes that are not a PMTiles archive at all.
#[test]
#[cfg_attr(miri, ignore)]
fn wrong_magic_and_wrong_version_are_each_their_own_finding() {
    let mut wrong_magic = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    wrong_magic[..7].copy_from_slice(b"NOTPMTs");
    let report = validate_bytes(&wrong_magic, &ValidationLimits::default()).expect("readable");
    assert_eq!(
        report.findings,
        vec![Finding::BadMagic { found: *b"NOTPMTs" }],
        "a non-archive should produce exactly one finding and stop"
    );
    assert!(report.header.is_none());

    let mut wrong_version = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    wrong_version[7] = 2;
    let report = validate_bytes(&wrong_version, &ValidationLimits::default()).expect("readable");
    assert_eq!(
        report.findings,
        vec![Finding::UnsupportedVersion { found: 2 }],
        "PMTiles v2 shares the magic and has a completely different layout"
    );
}

/// A file too short to hold a header is a finding rather than an I/O error.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_shorter_than_the_header_is_flagged() {
    let full = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    for len in [0usize, 1, 7, 8, 126] {
        let report = validate_bytes(&full[..len], &ValidationLimits::default()).expect("readable");
        assert_eq!(
            report.findings,
            vec![Finding::ArchiveTooShort {
                size: len as u64,
                need: 127
            }],
            "a {len}-byte input"
        );
    }
}

/// A directory pointing at bytes that are not what the header's internal
/// compression says is a typed finding.
///
/// This is the second half of the reference's crash: `DeserializeEntries`
/// throws away the `gzip.NewReader` error and dereferences the nil reader on
/// the next read, so *any* directory offset landing on non-gzip bytes is a
/// segfault there rather than a rare interaction.
#[test]
#[cfg_attr(miri, ignore)]
fn a_directory_that_does_not_decompress_is_a_finding_and_not_a_crash() {
    let archive = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    let header = Header::try_decode(&archive).expect("the header decodes");
    assert_eq!(header.internal_compression, Compression::Gzip);
    // Point the root at the tile data, which is PNG bytes and not gzip.
    let broken = with_u64(&archive, OFF_ROOT_OFFSET, header.tile_data_offset);

    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("readable");
    assert!(
        report.findings.iter().any(|f| matches!(
            f,
            Finding::DirectoryUnreadable {
                directory: DirectoryRef::Root,
                ..
            }
        )),
        "a root directory that is not gzip was not reported: {:?}",
        report.findings
    );
}

/// A leaf pointer whose target is outside the archive is flagged, and the walk
/// does not follow it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_leaf_pointer_outside_the_leaf_section_is_flagged() {
    // Move the leaf region to the end of the file; the six pointers, whose
    // offsets are relative to it, then resolve past the archive.
    let broken = with_u64(
        &golden("leaves-z0z7.pmtiles", LEAVES_GOLDEN_SHA256),
        OFF_LEAF_OFFSET,
        860,
    );
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("readable");
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f, Finding::LeafPointerOutOfBounds { .. })),
        "{:?}",
        report.findings
    );
}

/// A header count that does not match what the directories hold.
#[test]
#[cfg_attr(miri, ignore)]
fn a_header_count_that_disagrees_with_the_directories_is_flagged() {
    let broken = with_u64(
        &golden("dupes-z0z3.pmtiles", DUPES_GOLDEN_SHA256),
        OFF_ADDRESSED_TILES,
        84,
    );
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("readable");
    assert!(
        report.findings.contains(&Finding::AddressedTilesMismatch {
            counted: 85,
            header: 84,
        }),
        "the run lengths sum to 85 and the header claims 84: {:?}",
        report.findings
    );
}

/// Runs that overlap, which `deserialize_entries` cannot see on its own.
///
/// The parser refuses a zero delta, so two entries cannot share an id, but
/// nothing there stops an entry with `run_length` 5 from being followed by one
/// two ids later. Both then claim the same tile and a lookup cannot say which
/// wins.
#[test]
#[cfg_attr(miri, ignore)]
fn overlapping_runs_are_flagged() {
    use libviprs::pmtiles::Entry;

    let entries = vec![
        Entry {
            tile_id: 0,
            offset: 0,
            length: 10,
            run_length: 5,
        },
        Entry {
            tile_id: 2,
            offset: 10,
            length: 10,
            run_length: 1,
        },
    ];
    let report = validate_bytes(
        &archive_around(&entries, &[0u8; 20]),
        &ValidationLimits::default(),
    )
    .expect("readable");
    assert!(
        report.findings.iter().any(|f| matches!(
            f,
            Finding::OverlappingRun {
                directory: DirectoryRef::Root,
                index: 1,
                ..
            }
        )),
        "two entries claiming tile 2 were not flagged: {:?}",
        report.findings
    );
}

/// An entry whose bytes run past the end of the tile data section.
#[test]
#[cfg_attr(miri, ignore)]
fn an_entry_pointing_outside_the_tile_data_is_flagged() {
    use libviprs::pmtiles::Entry;

    let entries = vec![Entry {
        tile_id: 0,
        offset: 0,
        length: 64,
        run_length: 1,
    }];
    // Only 20 bytes of tile data, and the entry claims 64.
    let report = validate_bytes(
        &archive_around(&entries, &[0u8; 20]),
        &ValidationLimits::default(),
    )
    .expect("readable");
    assert!(
        report.findings.iter().any(|f| matches!(
            f,
            Finding::EntryOutOfTileData {
                directory: DirectoryRef::Root,
                index: 0,
                ..
            }
        )),
        "{:?}",
        report.findings
    );
}

/// No input, however malformed, makes the validator panic or hand back an
/// error it has no name for.
///
/// Two sweeps over a real archive: every truncation, and a single byte flipped
/// at every position. Both are cheap on a 1878-byte file and both are the
/// shapes a fuzzer finds first.
#[test]
#[cfg_attr(miri, ignore)]
fn no_mutation_of_a_real_archive_makes_the_validator_panic() {
    let archive = golden("raster-z0z2.pmtiles", RASTER_GOLDEN_SHA256);
    let limits = ValidationLimits::default();

    let mut truncations = 0;
    for len in 0..=archive.len() {
        let report = validate_bytes(&archive[..len], &limits).expect("a slice is always readable");
        if len == archive.len() {
            assert!(report.findings.is_empty(), "the untouched archive is clean");
        }
        truncations += 1;
    }
    assert_eq!(truncations, archive.len() + 1);

    let mut flips = 0;
    for position in 0..archive.len() {
        let mut broken = archive.clone();
        broken[position] ^= 0xFF;
        let _ = validate_bytes(&broken, &limits).expect("a slice is always readable");
        flips += 1;
    }
    assert_eq!(
        flips, 1878,
        "the flip sweep covered every byte of the archive"
    );
}

/// Wrap a directory in the smallest archive that can hold it, so a test can
/// build one by hand and hand it to the validator.
///
/// Uncompressed internally, which is legal (`Compression::None` is `0x01`) and
/// keeps the fixture readable in a hex dump. The header's three counts are
/// derived from the entries, so an archive built this way is clean unless the
/// test deliberately made it otherwise.
fn archive_around(entries: &[libviprs::pmtiles::Entry], tile_data: &[u8]) -> Vec<u8> {
    use std::collections::BTreeSet;

    use libviprs::pmtiles::directory::serialize_entries;

    let directory = serialize_entries(entries).expect("the entries serialize");
    let mut header = Header {
        internal_compression: Compression::None,
        tile_compression: Compression::None,
        ..Header::default()
    };
    header.root_offset = 127;
    header.root_length = directory.len() as u64;
    header.metadata_offset = 127 + directory.len() as u64;
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
