//! The structural validator, against the `go-pmtiles` goldens and against
//! archives deliberately broken in the ways the reference implementation does
//! not survive (issue #991).
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

use std::collections::BTreeSet;

use libviprs::pmtiles::directory::{deserialize_entries, serialize_entries};
use libviprs::pmtiles::validate::{
    DirectoryRef, Finding, Section, ValidationLimits, validate_bytes,
};
use libviprs::pmtiles::{Compression, Header};

#[path = "common/pmtiles_oracle.rs"]
mod pmtiles_oracle;

use pmtiles_oracle::{GOLDEN_DIGESTS, golden, leaf_directories, root_directory};

/// Where the 64-bit little-endian header fields sit, for the tests that break
/// one on purpose. Taken from the v3 layout, which puts the seven-byte magic
/// and the version byte first and then eleven `u64`s.
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

// ---------------------------------------------------------------------------
// The goldens
// ---------------------------------------------------------------------------

/// All three archives `go-pmtiles` wrote validate with no findings, and the
/// totals the walk counted are the ones their headers claim.
#[test]
#[cfg_attr(miri, ignore)]
fn every_golden_archive_validates_clean() {
    let mut checked = 0;
    for (name, _) in GOLDEN_DIGESTS {
        let bytes = golden(name);
        let report = validate_bytes(&bytes, &ValidationLimits::default())
            .unwrap_or_else(|e| panic!("{name}: the validator could not read the archive: {e}"));
        assert!(
            report.findings.is_empty(),
            "{name} is an archive the reference implementation wrote and it \
             produced findings: {:?}",
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
    let bytes = golden("leaves-z0z7.pmtiles");
    let report = validate_bytes(&bytes, &ValidationLimits::default()).expect("the archive reads");
    assert!(report.findings.is_empty(), "{:?}", report.findings);

    let oracle = leaf_directories();
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
            found.offset_in_section, expected.offset_in_leaf_section,
            "a leaf pointer's relative offset"
        );
        assert_eq!(
            found.absolute_offset, expected.absolute_offset,
            "a leaf pointer resolved to the wrong place in the file"
        );
        assert_eq!(
            found.absolute_offset,
            header.leaf_directories_offset + expected.offset_in_leaf_section,
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

/// Every one of the 21844 leaf entries is the entry `go-pmtiles` decoded.
///
/// A per-leaf comparison rather than a total, because a total is satisfied by
/// two errors that cancel.
#[test]
#[cfg_attr(miri, ignore)]
fn the_leaf_entries_are_the_ones_go_pmtiles_decoded() {
    let bytes = golden("leaves-z0z7.pmtiles");
    let report = validate_bytes(&bytes, &ValidationLimits::default().with_collect_entries(true))
        .expect("the archive reads");

    let oracle = leaf_directories();
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
                (want.tile_id, want.offset, want.length, want.run_length),
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

/// The decompressed root directory is byte-for-byte what `go-pmtiles` wrote,
/// and re-serialising the entries reproduces those same bytes.
///
/// The second half is what makes this a target for a writer rather than a
/// fixture for a parser: the oracle recorded that `SerializeEntries` over the
/// entries `DeserializeEntries` had just decoded gives the directory body
/// back exactly, so a serializer that produces anything else is producing a
/// different file.
#[test]
#[cfg_attr(miri, ignore)]
fn root_directory_bytes_match_the_reference_byte_for_byte() {
    let mut checked = 0;
    for name in ["raster-z0z2.pmtiles", "dupes-z0z3.pmtiles"] {
        let archive = golden(name);
        let header = Header::try_decode(&archive).expect("a golden header decodes");
        let start = usize::try_from(header.root_offset).unwrap();
        let end = start + usize::try_from(header.root_length).unwrap();
        let plain = header
            .internal_compression
            .decompress(&archive[start..end], 1 << 20)
            .expect("the root directory decompresses");

        let (want_bytes, want_entries) = root_directory(name);
        assert_eq!(plain, want_bytes, "{name}: decompressed root directory");

        let entries = deserialize_entries(&plain).expect("the root directory parses");
        assert_eq!(entries.len(), want_entries.len(), "{name}: entry count");
        for (got, want) in entries.iter().zip(&want_entries) {
            assert_eq!(
                (got.tile_id, got.offset, got.length, got.run_length),
                (want.tile_id, want.offset, want.length, want.run_length),
                "{name}"
            );
        }
        assert_eq!(
            serialize_entries(&entries).expect("the entries re-serialize"),
            plain,
            "{name}: our serializer does not reproduce the reference's bytes"
        );
        checked += 1;
    }
    assert_eq!(checked, 2);
}

/// Both deduplication shapes in `dupes-z0z3.pmtiles` survive the walk.
///
/// They are genuinely different and a reader can handle one without the other.
/// Runs collapse identical *consecutive* tiles into one entry with
/// `run_length > 1`. Identical tiles that are not consecutive stay separate
/// entries with `run_length` 1 that point at the same offset. A reader that
/// only handles runs returns the wrong bytes for the second shape while
/// looking correct on most archives.
#[test]
#[cfg_attr(miri, ignore)]
fn both_deduplication_shapes_are_present_and_counted() {
    let (_, entries) = root_directory("dupes-z0z3.pmtiles");
    assert_eq!(entries.len(), 67);

    let runs: Vec<u32> = entries
        .iter()
        .map(|e| e.run_length)
        .filter(|r| *r > 1)
        .collect();
    assert!(
        runs.contains(&4) && runs.contains(&16),
        "the four z1 tiles and sixteen z2 tiles should be runs of 4 and 16, got {runs:?}"
    );

    let at_148: Vec<u64> = entries
        .iter()
        .filter(|e| e.offset == 148)
        .map(|e| e.tile_id)
        .collect();
    assert_eq!(
        at_148,
        vec![21, 49, 63, 76],
        "four non-adjacent entries share offset 148"
    );

    let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
    assert_eq!(addressed, 85, "the run lengths sum to addressed_tiles_count");
    let distinct: BTreeSet<u64> = entries.iter().map(|e| e.offset).collect();
    assert_eq!(distinct.len(), 63, "distinct offsets are tile_contents_count");

    let report = validate_bytes(&golden("dupes-z0z3.pmtiles"), &ValidationLimits::default())
        .expect("the archive reads");
    assert!(report.findings.is_empty(), "{:?}", report.findings);
    assert_eq!(report.addressed_tiles, 85);
    assert_eq!(report.tile_entries, 67);
    assert_eq!(report.tile_contents, Some(63));
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
    let broken = with_u64(&golden("raster-z0z2.pmtiles"), OFF_ROOT_OFFSET, 999_999);
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
        &with_u64(&golden("raster-z0z2.pmtiles"), OFF_ROOT_OFFSET, u64::MAX - 4),
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
    let broken = with_u64(&golden("raster-z0z2.pmtiles"), OFF_ROOT_LENGTH, 1 << 40);
    let report = validate_bytes(&broken, &ValidationLimits::default()).expect("bytes are readable");
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f, Finding::SectionOutOfBounds { section: Section::Root, .. })),
        "{:?}",
        report.findings
    );
}

/// A truncated archive is flagged on the section that no longer fits, and the
/// header itself still decodes.
#[test]
#[cfg_attr(miri, ignore)]
fn a_truncated_archive_is_flagged() {
    let full = golden("raster-z0z2.pmtiles");
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
    let mut wrong_magic = golden("raster-z0z2.pmtiles");
    wrong_magic[..7].copy_from_slice(b"NOTPMTs");
    let report = validate_bytes(&wrong_magic, &ValidationLimits::default()).expect("readable");
    assert_eq!(
        report.findings,
        vec![Finding::BadMagic {
            found: *b"NOTPMTs"
        }],
        "a non-archive should produce exactly one finding and stop"
    );
    assert!(report.header.is_none());

    let mut wrong_version = golden("raster-z0z2.pmtiles");
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
    for len in [0usize, 1, 7, 8, 126] {
        let report = validate_bytes(&golden("raster-z0z2.pmtiles")[..len], &ValidationLimits::default())
            .expect("readable");
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
    let mut broken = golden("raster-z0z2.pmtiles");
    // Point the root at the tile data, which is PNG bytes and not gzip.
    let header = Header::try_decode(&broken).expect("the header decodes");
    assert_eq!(header.internal_compression, Compression::Gzip);
    broken = with_u64(&broken, OFF_ROOT_OFFSET, header.tile_data_offset);

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

/// A leaf pointer whose target is outside the leaf section is flagged, and the
/// walk does not follow it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_leaf_pointer_outside_the_leaf_section_is_flagged() {
    // Shrink the leaf section to nothing by moving its offset to the end of
    // the file; the six pointers then resolve past the archive.
    let broken = with_u64(&golden("leaves-z0z7.pmtiles"), OFF_LEAF_OFFSET, 860);
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
    let broken = with_u64(&golden("dupes-z0z3.pmtiles"), OFF_ADDRESSED_TILES, 84);
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
    let directory = serialize_entries(&entries).expect("these entries serialize");
    let archive = archive_around(&directory, &[0u8; 20]);

    let report = validate_bytes(&archive, &ValidationLimits::default()).expect("readable");
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
    let directory = serialize_entries(&entries).expect("these entries serialize");
    // Only 20 bytes of tile data, and the entry claims 64.
    let archive = archive_around(&directory, &[0u8; 20]);

    let report = validate_bytes(&archive, &ValidationLimits::default()).expect("readable");
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
/// Two sweeps over a real archive: every truncation, and a single byte
/// flipped at every position. Both are cheap on a 1878-byte file and both are
/// the shapes a fuzzer finds first.
#[test]
#[cfg_attr(miri, ignore)]
fn no_mutation_of_a_real_archive_makes_the_validator_panic() {
    let archive = golden("raster-z0z2.pmtiles");
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
    assert_eq!(flips, 1878, "the flip sweep covered every byte of the archive");
}

/// Wrap a serialised directory in the smallest archive that can hold it, so a
/// test can build a directory by hand and hand it to the validator.
///
/// Uncompressed internally, which is legal (`Compression::None` is `0x01`) and
/// keeps the fixture readable in a hex dump.
fn archive_around(directory: &[u8], tile_data: &[u8]) -> Vec<u8> {
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

    let entries = deserialize_entries(directory).expect("the directory parses");
    header.tile_entries_count = entries.iter().filter(|e| !e.is_leaf()).count() as u64;
    header.addressed_tiles_count = entries.iter().map(|e| u64::from(e.run_length)).sum();
    header.tile_contents_count = entries
        .iter()
        .filter(|e| !e.is_leaf())
        .map(|e| e.offset)
        .collect::<BTreeSet<u64>>()
        .len() as u64;

    let mut out = header.encode().to_vec();
    out.extend_from_slice(directory);
    out.extend_from_slice(tile_data);
    out
}
