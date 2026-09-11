//! PMTiles v3 format invariants, as properties and as `go-pmtiles` vectors
//! (issue #991).
//!
//! Two kinds of test share this file because they answer two halves of one
//! question, and either half alone is misleading.
//!
//! The **properties** say the primitives are each other's inverse over a wide
//! input space: `zxy_to_tileid` and `tileid_to_zxy`, `encode_uvarint` and
//! `decode_uvarint`, `serialize_entries` and `deserialize_entries`,
//! `Header::encode` and `Header::try_decode`. That catches an off-by-one at a
//! zoom nobody wrote a fixture for, and it catches a decoder that drifts after
//! the contiguous-offset shorthand, which is the kind of bug a three-entry
//! example never reaches.
//!
//! It cannot catch a *wrong convention*. A Hilbert mapping and its transpose
//! are both perfect inverses of themselves, and a varint reader that read the
//! bytes backwards would round-trip with a writer that wrote them backwards.
//! So the **vectors** pin the same primitives to numbers that came out of the
//! real `go-pmtiles` v1.31.2, loaded from the committed JSON under
//! `tests/fixtures/pmtiles/vectors/` at run time rather than copied into a
//! table here, because a transcribed literal and an invented one are the same
//! thing in a diff.
//!
//! # Twelve of the 162 vector rows are the ones that mean anything
//!
//! This is the part worth reading before adding a row. The oracle carries 162
//! distinct `(z, x, y)` pairs, and 150 of them are *structural*: the first and
//! last tile of a level, the four quadrant corners, the tiles straddling the
//! centre. Four different candidate conventions agree on every one of those,
//! plain Z-order with no rotation included, so a suite that pins all 150 and
//! passes has demonstrated nothing about which curve this crate implements. It
//! has a large green table and no discrimination.
//!
//! The twelve rows in `convention_discriminators` are the ones that separate
//! them: zoom 9 to 15, off every quadrant boundary, `x` and `y` of differing
//! parity, and four of them arranged as swapped pairs, so
//! `(13, 5107, 2884) = 79053962` against `(13, 2884, 5107) = 53847284` also
//! rules out any symmetric mapping.
//! [`the_twelve_convention_discriminators_match_go_pmtiles`]
//! is therefore the test in this file that would go red on a wrong curve.
//!
//! I measured how much the rest are worth rather than assuming it. Against a
//! candidate mapping that keeps the quadrant order and drops the Hilbert
//! reflection, all twelve discriminators change, all 76 rows in
//! [`the_purely_structural_rows_agree_and_could_not_have_failed`] stay the
//! same, and the 97 rows in [`the_scattered_and_ordering_rows_agree_too`] are
//! in between, with 44 of them changing. Three tests rather than two, so a
//! failure says which kind of wrong the curve is.
//!
//! # And ten rows are evidence *against* the reference
//!
//! `out_of_range_observations` records what `ZxyToID` does when the coordinate
//! is impossible, which is not to refuse it. It masks `x` and `y` into a
//! different valid tile, so `(2, 4, 0)` comes back as id 5, which is really
//! `(2, 0, 0)`, and `pmtiles tile archive 2 4 0` will happily hand over the
//! payload of `(2, 0, 0)` with exit code 0. Above zoom 31 it saturates instead,
//! and z=32, z=33 and z=63 all return one id that `IDToZxy` then decodes as
//! zoom 127.
//!
//! libviprs refuses every one of those, per the crate's `try_*` convention, so
//! [`the_rows_where_the_reference_masks_or_saturates_are_refused`] asserts an
//! error and never a value. Pinning them as expectations would be building a
//! reader that reproduces a reference bug, and it would look *better* than
//! getting it right, because the table would be bigger.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use libviprs::pmtiles::directory::{deserialize_entries, push_entry, serialize_entries};
use libviprs::pmtiles::tileid::{MAX_TILE_ID, MAX_ZOOM, first_tileid_of_zoom};
use libviprs::pmtiles::varint::{MAX_UVARINT_LEN, decode_uvarint, encode_uvarint, uvarint_len};
use libviprs::pmtiles::{
    Compression, Entry, Header, PmTilesError, TileType, tileid_to_zxy, zxy_to_tileid,
};
use proptest::prelude::*;

#[path = "common/pmtiles_oracle.rs"]
mod pmtiles_oracle;

use pmtiles_oracle::{header_u64, level_bases, oracle_header, tileid_section};

// ---------------------------------------------------------------------------
// How many cases, and why that many
// ---------------------------------------------------------------------------

/// Cases for the cheap integer properties.
///
/// Every one of these is a few hundred nanoseconds of arithmetic, so the
/// number is chosen for coverage rather than for a time budget: 4096 draws
/// over 32 zoom levels puts roughly 128 in each. The whole file runs in under
/// a second, which matters because this repo already has a proptest that takes
/// about ten seconds per case, and a suite nobody will wait for is a suite
/// nobody runs.
const CHEAP_CASES: u32 = 4096;

/// Cases for the properties that build and re-parse a whole directory or
/// header. Each one allocates and sweeps four varint columns, so they are
/// worth a few microseconds rather than a few hundred nanoseconds.
const STRUCTURAL_CASES: u32 = 512;

/// The most entries a generated directory holds. Large enough that the
/// contiguous-offset shorthand and a run both appear many times over the run,
/// small enough that 512 cases stay in milliseconds.
const MAX_GENERATED_ENTRIES: usize = 64;

/// `cases` cases, and **no failure-persistence file**.
///
/// proptest's default writes a `.proptest-regressions` file next to the
/// source on the first failure. In an integration test it cannot even find
/// the crate root to write it ("FileFailurePersistence::SourceParallel set,
/// but failed to find lib.rs or main.rs"), and where it can, it drops an
/// untracked file into the tree that the next run then treats as input. A
/// failing property here should print its minimal input and nothing else.
fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------------
// The vectors
// ---------------------------------------------------------------------------

/// The twelve rows that tell a real Hilbert mapping apart from the plausible
/// imitations, forward and inverse, from `go-pmtiles` v1.31.2.
///
/// A failure here is a wrong curve. A failure in
/// [`the_structural_vector_rows_agree_as_well`] and not here is a wrong
/// *detail* in a curve that is otherwise the right shape.
#[test]
#[cfg_attr(miri, ignore)]
fn the_twelve_convention_discriminators_match_go_pmtiles() {
    let rows = tileid_section("convention_discriminators");
    assert_eq!(
        rows.len(),
        12,
        "the discriminator set is the whole point of this test"
    );

    let mut swapped_pairs = 0;
    for row in &rows {
        assert!(
            row.roundtrip_ok,
            "a discriminator row that does not round-trip in the reference is \
             not a target: {row:?}"
        );
        let got = zxy_to_tileid(row.z, row.x, row.y)
            .unwrap_or_else(|e| panic!("({}, {}, {}) was refused: {e}", row.z, row.x, row.y));
        assert_eq!(
            got, row.tile_id,
            "zxy_to_tileid({}, {}, {}) is {got}, go-pmtiles says {}",
            row.z, row.x, row.y, row.tile_id
        );
        assert_eq!(
            tileid_to_zxy(row.tile_id).expect("a reference id decodes"),
            (row.z, row.x, row.y),
            "tileid_to_zxy({}) does not come back to the coordinate that \
             produced it",
            row.tile_id
        );
        if rows.iter().any(|other| other.x == row.y && other.y == row.x) && row.x != row.y {
            swapped_pairs += 1;
        }
    }

    // The swapped pairs are what rule out a symmetric mapping, so their
    // presence is part of what this test claims. Three pairs, counted from
    // both ends, is six rows: (12, 3423, 1763), (13, 5107, 2884) and
    // (15, 20749, 9310), each with its transpose.
    assert_eq!(
        swapped_pairs, 6,
        "the discriminators no longer contain the swapped pairs that rule out \
         a symmetric mapping, so this test has quietly stopped discriminating"
    );
}

/// Check one group of `tileid.json` sections forward and backward, and return
/// how many rows it covered.
fn check_sections(sections: &[&str]) -> usize {
    let mut checked = 0usize;
    for section in sections {
        for row in tileid_section(section) {
            assert!(row.roundtrip_ok, "{section} row {row:?} is an observation");
            assert_eq!(
                zxy_to_tileid(row.z, row.x, row.y).expect("an in-range vector"),
                row.tile_id,
                "{section}: zxy_to_tileid({}, {}, {})",
                row.z,
                row.x,
                row.y
            );
            assert_eq!(
                tileid_to_zxy(row.tile_id).expect("a reference id decodes"),
                (row.z, row.x, row.y),
                "{section}: tileid_to_zxy({})",
                row.tile_id
            );
            checked += 1;
        }
    }
    checked
}

/// The 76 rows that genuinely cannot discriminate: the first and last tile of
/// every level, and the quadrant and centre boundaries.
///
/// Worth having as a regression net and worth not overrating. I measured this
/// rather than taking it on faith: against a candidate mapping that keeps the
/// quadrant order and drops the Hilbert reflection, **every one of these 76
/// rows still agrees**, so a suite built only on them would be green on a
/// mapping that is wrong at every zoom above 2.
#[test]
#[cfg_attr(miri, ignore)]
fn the_purely_structural_rows_agree_and_could_not_have_failed() {
    let checked = check_sections(&["first_and_last_of_level", "orientation_boundaries"]);
    assert_eq!(checked, 76, "32 first-and-last rows plus 44 boundary rows");
}

/// The 97 rows that turn out to discriminate after all: the scattered
/// coordinates and the two full zoom orderings.
///
/// The oracle's own notes group these with the structural rows as
/// non-discriminating, and measured against the no-reflection candidate that
/// is not quite right: 12 of the 17 `off_grid` rows, 4 of the 16 `z2`
/// orderings and 28 of the 64 `z3` orderings change. They are weaker than the
/// twelve discriminators, which all change, and they are not nothing, so they
/// are in their own test rather than pooled with the 76 above.
#[test]
#[cfg_attr(miri, ignore)]
fn the_scattered_and_ordering_rows_agree_too() {
    let checked = check_sections(&["off_grid", "hilbert_order_z2", "hilbert_order_z3"]);
    assert_eq!(checked, 97, "17 off-grid rows plus 16 z2 plus 64 z3");
}

/// The first and last id of every zoom level, and the level's tile count.
#[test]
#[cfg_attr(miri, ignore)]
fn level_bases_match_go_pmtiles() {
    let rows = level_bases();
    assert_eq!(rows.len(), 16, "the oracle covers sixteen zoom levels");
    for (z, first, last, count) in rows {
        assert_eq!(
            first_tileid_of_zoom(z).expect("an in-range zoom"),
            first,
            "the first id of zoom {z}"
        );
        assert_eq!(
            tileid_to_zxy(first).expect("a level base decodes"),
            (z, 0, 0),
            "the first id of zoom {z} is not (z, 0, 0)"
        );
        assert_eq!(
            last - first + 1,
            count,
            "zoom {z}'s span does not match its tile count"
        );
        let (decoded_z, _, _) = tileid_to_zxy(last).expect("a level's last id decodes");
        assert_eq!(decoded_z, z, "the last id of zoom {z} decoded to another zoom");
    }
}

/// The rows where `go-pmtiles` masks or saturates rather than refusing.
///
/// The assertion is that libviprs returns an error, and specifically **not**
/// the id the reference returned. Both halves matter: an implementation that
/// happened to return a different wrong id would satisfy "not equal" while
/// being just as wrong.
#[test]
#[cfg_attr(miri, ignore)]
fn the_rows_where_the_reference_masks_or_saturates_are_refused() {
    let rows = tileid_section("out_of_range_observations");
    assert_eq!(rows.len(), 10, "the oracle recorded ten such rows");

    let mut masked = 0;
    let mut saturated = 0;
    for row in &rows {
        assert!(
            !row.roundtrip_ok,
            "an out-of-range row that round-trips in the reference is not one \
             of these: {row:?}"
        );
        let got = zxy_to_tileid(row.z, row.x, row.y);
        match &got {
            Err(PmTilesError::ZoomOutOfRange { zoom, max }) => {
                assert!(row.z > MAX_ZOOM, "{row:?} was refused for the wrong reason");
                assert_eq!((*zoom, *max), (row.z, MAX_ZOOM));
                saturated += 1;
            }
            Err(PmTilesError::CoordOutOfRange { z, x, y, .. }) => {
                assert!(
                    row.z <= MAX_ZOOM,
                    "{row:?} should have been refused on its zoom"
                );
                assert_eq!((*z, *x, *y), (row.z, row.x, row.y));
                masked += 1;
            }
            other => panic!(
                "({}, {}, {}) gave {other:?}. go-pmtiles answers {} here, which \
                 is a different, valid tile; we refuse instead.",
                row.z, row.x, row.y, row.tile_id
            ),
        }
    }

    // A positive control on the match above: if every row landed in one arm,
    // half of what this test claims to cover would not be covered at all.
    assert_eq!(masked, 6, "six rows are an out-of-grid x or y");
    assert_eq!(saturated, 4, "four rows are a zoom above 31");

    // The reference's three saturating zooms collapse onto one id. Ours has no
    // id to collapse onto, and this is the sharpest statement of the
    // difference: 6148914691236517205 is one past the last addressable id.
    assert_eq!(MAX_TILE_ID + 1, 6_148_914_691_236_517_205);
    assert!(tileid_to_zxy(MAX_TILE_ID + 1).is_err());
}

/// The header fields `go-pmtiles` decoded, against ours, on all three goldens.
///
/// This is a header-only read, so it needs the archive bytes but none of the
/// reader: `Header::try_decode` takes the first 127 bytes.
#[test]
#[cfg_attr(miri, ignore)]
fn golden_headers_decode_to_the_fields_go_pmtiles_reports() {
    let mut checked = 0;
    for (name, _) in pmtiles_oracle::GOLDEN_DIGESTS {
        let bytes = pmtiles_oracle::golden(name);
        let header = Header::try_decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let fields = oracle_header(name);

        assert_eq!(header.root_offset, header_u64(&fields, "root_offset"));
        assert_eq!(header.root_length, header_u64(&fields, "root_length"));
        assert_eq!(
            header.metadata_offset,
            header_u64(&fields, "metadata_offset")
        );
        assert_eq!(
            header.metadata_length,
            header_u64(&fields, "metadata_length")
        );
        assert_eq!(
            header.leaf_directories_offset,
            header_u64(&fields, "leaf_directory_offset")
        );
        assert_eq!(
            header.leaf_directories_length,
            header_u64(&fields, "leaf_directory_length")
        );
        assert_eq!(
            header.tile_data_offset,
            header_u64(&fields, "tile_data_offset")
        );
        assert_eq!(
            header.tile_data_length,
            header_u64(&fields, "tile_data_length")
        );
        assert_eq!(
            header.addressed_tiles_count,
            header_u64(&fields, "addressed_tiles_count")
        );
        assert_eq!(
            header.tile_entries_count,
            header_u64(&fields, "tile_entries_count")
        );
        assert_eq!(
            header.tile_contents_count,
            header_u64(&fields, "tile_contents_count")
        );
        assert_eq!(
            u64::from(header.min_zoom),
            header_u64(&fields, "min_zoom"),
            "{name} min zoom"
        );
        assert_eq!(
            u64::from(header.max_zoom),
            header_u64(&fields, "max_zoom"),
            "{name} max zoom"
        );
        assert_eq!(
            u64::from(header.internal_compression.to_byte()),
            header_u64(&fields, "internal_compression")
        );
        assert_eq!(
            u64::from(header.tile_compression.to_byte()),
            header_u64(&fields, "tile_compression")
        );
        assert_eq!(
            u64::from(header.tile_type.to_byte()),
            header_u64(&fields, "tile_type")
        );

        // The finding worth pinning from this file rather than reading in
        // prose: a no-leaf archive does NOT carry a zero leaf offset. Both
        // small goldens set it to `tile_data_offset`, so anything deciding
        // "are there leaves" from the offset answers yes for every archive
        // go-pmtiles writes.
        assert_eq!(
            header.has_leaves(),
            header.leaf_directories_length > 0,
            "{name}: has_leaves must read the length"
        );
        if !header.has_leaves() {
            assert_eq!(
                header.leaf_directories_offset, header.tile_data_offset,
                "{name}: a no-leaf archive still carries a non-zero leaf offset"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 3, "all three goldens were compared");
}

// ---------------------------------------------------------------------------
// The properties
// ---------------------------------------------------------------------------
//
// Every proptest below counts its own invocations into a static and asserts
// the count afterwards. A `proptest!` block whose strategy produced nothing,
// or which was filtered down to nothing, passes silently, and so does one
// whose case count somebody set to zero. The counter is the positive control
// that says the assertions actually ran.

static TILEID_CASES: AtomicUsize = AtomicUsize::new(0);
static TILEID_ZOOMS: AtomicU32 = AtomicU32::new(0);

/// `zxy_to_tileid` and `tileid_to_zxy` are inverses over the whole
/// addressable space.
///
/// The coordinate is drawn as a raw `u32` and folded into the level's grid, so
/// the draw is uniform across the level rather than clustered near the origin,
/// which is where a fixture-based suite always is.
#[test]
fn zxy_and_tileid_are_inverses() {
    proptest!(
        config(CHEAP_CASES),
        |(z in 0u8..=MAX_ZOOM, raw_x in any::<u32>(), raw_y in any::<u32>())| {
            TILEID_CASES.fetch_add(1, Ordering::Relaxed);
            TILEID_ZOOMS.fetch_or(1u32 << z, Ordering::Relaxed);

            let side = 1u64 << z;
            let x = u32::try_from(u64::from(raw_x) % side).unwrap();
            let y = u32::try_from(u64::from(raw_y) % side).unwrap();

            let id = zxy_to_tileid(z, x, y).unwrap();
            prop_assert!(id <= MAX_TILE_ID);
            prop_assert!(id >= first_tileid_of_zoom(z).unwrap(), "id {} is below its level's base", id);
            prop_assert_eq!(tileid_to_zxy(id).unwrap(), (z, x, y));

            // A neighbouring tile must not share the id. This is the cheap
            // form of "the mapping is injective", and it is the assertion a
            // transposed or collapsed curve fails.
            if x + 1 < u32::try_from(side).unwrap_or(u32::MAX) {
                prop_assert_ne!(zxy_to_tileid(z, x + 1, y).unwrap(), id);
            }
        }
    );

    let cases = TILEID_CASES.load(Ordering::Relaxed);
    assert!(
        cases >= CHEAP_CASES as usize,
        "the property body ran {cases} times against a configured \
         {CHEAP_CASES}, so most of it was never exercised"
    );
    let zooms = TILEID_ZOOMS.load(Ordering::Relaxed).count_ones();
    assert!(
        zooms >= 16,
        "only {zooms} distinct zoom levels were generated out of 32, so the \
         strategy has collapsed and the property covers a sliver of the space"
    );
}

static ID_CASES: AtomicUsize = AtomicUsize::new(0);

/// The same inverse from the other side: every addressable id decodes to a
/// coordinate that encodes back to it.
#[test]
fn every_addressable_tile_id_decodes_and_re_encodes() {
    proptest!(
        config(CHEAP_CASES),
        |(id in 0u64..=MAX_TILE_ID)| {
            ID_CASES.fetch_add(1, Ordering::Relaxed);
            let (z, x, y) = tileid_to_zxy(id).unwrap();
            prop_assert!(z <= MAX_ZOOM);
            let side = 1u64 << z;
            prop_assert!(u64::from(x) < side, "x {} escaped the grid of zoom {}", x, z);
            prop_assert!(u64::from(y) < side, "y {} escaped the grid of zoom {}", y, z);
            prop_assert_eq!(zxy_to_tileid(z, x, y).unwrap(), id);
        }
    );
    assert!(ID_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

static REFUSAL_CASES: AtomicUsize = AtomicUsize::new(0);

/// A coordinate outside its level's grid is refused, never folded into a
/// different tile.
///
/// This is the property the reference implementation does not have, and it is
/// the one that keeps `pmtiles tile archive 2 4 0` from quietly returning the
/// payload of `(2, 0, 0)`.
#[test]
fn coordinates_outside_the_grid_are_always_refused() {
    proptest!(
        config(CHEAP_CASES),
        |(z in 0u8..=MAX_ZOOM, over in 0u32..1024, swap in any::<bool>())| {
            REFUSAL_CASES.fetch_add(1, Ordering::Relaxed);
            let side = 1u64 << z;
            // `side` is at most 2^31, so this stays inside a u32 for every
            // zoom the mapping accepts.
            let Ok(outside) = u32::try_from(side + u64::from(over)) else {
                return Ok(());
            };
            let (x, y) = if swap { (outside, 0) } else { (0, outside) };
            prop_assert!(
                matches!(
                    zxy_to_tileid(z, x, y),
                    Err(PmTilesError::CoordOutOfRange { .. })
                ),
                "({}, {}, {}) was not refused; the reference folds it into a \
                 different valid tile instead", z, x, y
            );
        }
    );
    assert!(REFUSAL_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

static ZOOM_REFUSAL_CASES: AtomicUsize = AtomicUsize::new(0);

/// Every zoom above 31 is refused, and so is every id above the last one a
/// coordinate can produce.
///
/// The reference saturates here instead: zoom 32, 33 and 63 all give one id,
/// which its own inverse then reads back as zoom 127.
#[test]
fn zooms_and_ids_past_the_addressable_range_are_always_refused() {
    proptest!(
        config(CHEAP_CASES),
        |(z in (MAX_ZOOM + 1)..=u8::MAX, id in (MAX_TILE_ID + 1)..=u64::MAX)| {
            ZOOM_REFUSAL_CASES.fetch_add(1, Ordering::Relaxed);
            prop_assert!(
                matches!(
                    zxy_to_tileid(z, 0, 0),
                    Err(PmTilesError::ZoomOutOfRange { .. })
                ),
                "zoom {} was not refused; the reference saturates here instead", z
            );
            prop_assert!(
                matches!(
                    first_tileid_of_zoom(z),
                    Err(PmTilesError::ZoomOutOfRange { .. })
                ),
                "first_tileid_of_zoom({}) was not refused", z
            );
            prop_assert!(
                matches!(
                    tileid_to_zxy(id),
                    Err(PmTilesError::TileIdOutOfRange { .. })
                ),
                "id {} is past the last addressable one and was not refused", id
            );
        }
    );
    assert!(ZOOM_REFUSAL_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

static VARINT_CASES: AtomicUsize = AtomicUsize::new(0);

/// `encode_uvarint` and `decode_uvarint` are inverses, the encoding is the
/// shortest one, and `uvarint_len` agrees with what the encoder wrote.
///
/// The trailing-byte case is in here rather than in its own test because it is
/// the same property: a decoder that consumed the whole buffer instead of one
/// value would pass a round-trip on a buffer holding exactly one varint and
/// fail on every real directory column.
#[test]
fn varints_round_trip_and_stop_where_they_should() {
    proptest!(
        config(CHEAP_CASES),
        |(value in any::<u64>(), tail in proptest::collection::vec(any::<u8>(), 0..8))| {
            VARINT_CASES.fetch_add(1, Ordering::Relaxed);

            let mut buf = Vec::new();
            encode_uvarint(value, &mut buf);
            prop_assert!(!buf.is_empty());
            prop_assert!(buf.len() <= MAX_UVARINT_LEN);
            prop_assert_eq!(buf.len(), uvarint_len(value));

            // Shortest form: only the final byte may be a bare zero, and no
            // byte before it may lack its continuation bit.
            for byte in &buf[..buf.len() - 1] {
                prop_assert!(byte & 0x80 != 0);
            }
            prop_assert!(buf[buf.len() - 1] & 0x80 == 0);
            if buf.len() > 1 {
                prop_assert!(buf[buf.len() - 1] != 0, "a trailing zero byte is not minimal");
            }

            let (decoded, used) = decode_uvarint(&buf, 0).unwrap();
            prop_assert_eq!(decoded, value);
            prop_assert_eq!(used, buf.len());

            // The same bytes with arbitrary junk behind them decode to the
            // same value and consume the same number of bytes.
            let mut with_tail = buf.clone();
            with_tail.extend_from_slice(&tail);
            let (again, used_again) = decode_uvarint(&with_tail, 0).unwrap();
            prop_assert_eq!(again, value);
            prop_assert_eq!(used_again, used);
        }
    );
    assert!(VARINT_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

static VARINT_REFUSAL_CASES: AtomicUsize = AtomicUsize::new(0);

/// A varint that does not fit a `u64` is refused, and a truncated one is
/// refused differently.
///
/// Three shapes, and they have to stay distinguishable because a caller
/// debugging a corrupt archive needs to know whether the buffer ended or the
/// value overflowed:
///
/// * more than ten continuation bytes: [`PmTilesError::VarintOverflow`];
/// * exactly ten bytes whose last carries bits above 63: the same;
/// * fewer than ten bytes, all with the continuation bit set:
///   [`PmTilesError::TruncatedVarint`].
#[test]
fn over_long_and_truncated_varints_are_refused() {
    proptest!(
        config(CHEAP_CASES),
        |(extra in 1usize..16, short in 0usize..MAX_UVARINT_LEN, last in 2u8..=0x7F)| {
            VARINT_REFUSAL_CASES.fetch_add(1, Ordering::Relaxed);

            // Eleven or more bytes that never terminate.
            let over_long = vec![0xFFu8; MAX_UVARINT_LEN + extra];
            prop_assert!(
                matches!(
                    decode_uvarint(&over_long, 0),
                    Err(PmTilesError::VarintOverflow { .. })
                ),
                "{} continuation bytes were not reported as an overflow",
                over_long.len()
            );

            // Exactly ten bytes, terminating, but the tenth carries bits that
            // would land above bit 63.
            let mut too_wide = vec![0xFFu8; MAX_UVARINT_LEN - 1];
            too_wide.push(last);
            prop_assert!(
                matches!(
                    decode_uvarint(&too_wide, 0),
                    Err(PmTilesError::VarintOverflow { .. })
                ),
                "a tenth byte of {} carries bits above 63 and was accepted", last
            );

            // The largest legal ten-byte varint, for contrast: the same shape
            // with a bare 1 in the tenth byte is u64::MAX and must be accepted.
            let mut widest_legal = vec![0xFFu8; MAX_UVARINT_LEN - 1];
            widest_legal.push(1);
            prop_assert_eq!(decode_uvarint(&widest_legal, 0).unwrap(), (u64::MAX, MAX_UVARINT_LEN));

            // A buffer that ends mid-value.
            let truncated = vec![0x80u8; short];
            prop_assert!(
                matches!(
                    decode_uvarint(&truncated, 0),
                    Err(PmTilesError::TruncatedVarint { .. })
                ),
                "a {}-byte unterminated varint was not reported as truncated", short
            );
        }
    );
    assert!(VARINT_REFUSAL_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

static ARBITRARY_VARINT_CASES: AtomicUsize = AtomicUsize::new(0);

/// Arbitrary bytes never panic the decoder, and whatever it does return is
/// consistent with what the encoder would have written.
///
/// It is **not** `encode(decode(bytes)) == bytes`, and that is a real property
/// of this decoder rather than a weakening to make a test pass. LEB128 has
/// non-minimal spellings: `[0x80, 0x00]` is a perfectly decodable zero, and
/// the decoder accepts it and reports two bytes consumed where the encoder
/// would have written one. Go's `binary.Uvarint`, which is what the reference
/// implementation reads directories with, accepts them too, so refusing them
/// would make this crate stricter than the format's own tooling on archives
/// nothing in the wild produces. What must hold is that the canonical form is
/// never *longer* than what was consumed, which is what catches a decoder that
/// silently dropped a byte or ran a byte past its value.
#[test]
fn arbitrary_bytes_never_panic_the_varint_decoder() {
    proptest!(
        config(CHEAP_CASES),
        |(bytes in proptest::collection::vec(any::<u8>(), 0..24))| {
            ARBITRARY_VARINT_CASES.fetch_add(1, Ordering::Relaxed);
            if let Ok((value, used)) = decode_uvarint(&bytes, 0) {
                prop_assert!(used >= 1 && used <= bytes.len());
                prop_assert!(used <= MAX_UVARINT_LEN);
                prop_assert!(
                    uvarint_len(value) <= used,
                    "decoding {:?} gave {} out of {} bytes, but its canonical \
                     encoding is {} bytes long",
                    &bytes[..used], value, used, uvarint_len(value)
                );
                // Every byte before the last must carry a continuation bit and
                // the last must not, or the decoder stopped in the wrong place.
                for byte in &bytes[..used - 1] {
                    prop_assert!(byte & 0x80 != 0);
                }
                prop_assert!(bytes[used - 1] & 0x80 == 0);

                // The canonical encoding of the value it returned decodes back
                // to the same value, which is the half that has to round-trip.
                let mut round = Vec::new();
                encode_uvarint(value, &mut round);
                prop_assert_eq!(decode_uvarint(&round, 0).unwrap(), (value, round.len()));
            }
        }
    );
    assert!(ARBITRARY_VARINT_CASES.load(Ordering::Relaxed) >= CHEAP_CASES as usize);
}

/// A strategy for a directory that a conformant writer could have produced:
/// strictly ascending ids, non-zero lengths, and a run length that is
/// sometimes zero so leaf pointers are covered too.
///
/// The `contiguous` flag is the part that earns its place. A tile laid down
/// immediately after the previous one is the common case in a real archive and
/// it is what the offset column's `0` sentinel exists for, and drawing an
/// offset independently of the previous entry's end produces it essentially
/// never: the first version of this generator picked both from `0..4096` and
/// the run of 512 cases hit the shorthand once by luck and then, on the next
/// seed, not at all. That made the positive control below flaky, which is
/// worse than not having one.
fn directory_strategy() -> impl Strategy<Value = Vec<Entry>> {
    proptest::collection::vec(
        (1u64..4096, 0u64..4096, 1u32..4096, 0u32..8, any::<bool>()),
        1..MAX_GENERATED_ENTRIES,
    )
    .prop_map(|rows| {
        let mut id: u64 = 0;
        let mut next_byte: u64 = 0;
        let mut entries = Vec::with_capacity(rows.len());
        for (gap, drawn_offset, length, run_length, contiguous) in rows {
            let offset = if !entries.is_empty() && contiguous {
                next_byte
            } else {
                drawn_offset
            };
            entries.push(Entry {
                tile_id: id,
                offset,
                length,
                run_length,
            });
            id += gap;
            next_byte = offset + u64::from(length);
        }
        entries
    })
}

/// The contiguous-offset shorthand, deterministically, so the coverage does
/// not rest on a random draw.
///
/// Three entries where each starts exactly where the last one ended, which
/// means entries 1 and 2 both encode their offset as a literal `0`. Two
/// shorthand entries in a row is the shape that separates a decoder tracking
/// the running position correctly from one that only advances it on the long
/// form: the second `0` is where the drift shows.
#[test]
fn two_contiguous_entries_in_a_row_round_trip() {
    let entries = vec![
        Entry {
            tile_id: 0,
            offset: 0,
            length: 10,
            run_length: 1,
        },
        Entry {
            tile_id: 1,
            offset: 10,
            length: 20,
            run_length: 1,
        },
        Entry {
            tile_id: 2,
            offset: 30,
            length: 5,
            run_length: 1,
        },
    ];
    let bytes = serialize_entries(&entries).expect("these entries serialize");
    // The last three bytes are the offset column: 1 for entry 0 (offset + 1),
    // then the sentinel twice.
    assert_eq!(&bytes[bytes.len() - 3..], &[1, 0, 0], "the offset column");
    assert_eq!(deserialize_entries(&bytes).expect("it parses"), entries);
}

static DIRECTORY_CASES: AtomicUsize = AtomicUsize::new(0);
static DIRECTORY_SHORTHAND_HITS: AtomicUsize = AtomicUsize::new(0);
static DIRECTORY_LEAF_HITS: AtomicUsize = AtomicUsize::new(0);

/// `serialize_entries` and `deserialize_entries` are inverses, run lengths
/// and leaf pointers included.
///
/// The interesting half is the offset column, which is not a delta: it stores
/// `offset + 1`, with a literal `0` meaning "this tile starts where the
/// previous one ended". A decoder that advanced its running position only on
/// the long form drifts the moment two shorthand entries meet, and no
/// three-entry fixture reaches that.
#[test]
fn directories_round_trip_through_their_column_encoding() {
    proptest!(
        config(STRUCTURAL_CASES),
        |(entries in directory_strategy())| {
            DIRECTORY_CASES.fetch_add(1, Ordering::Relaxed);

            for window in entries.windows(2) {
                if window[1].offset == window[0].offset + u64::from(window[0].length) {
                    DIRECTORY_SHORTHAND_HITS.fetch_add(1, Ordering::Relaxed);
                }
            }
            for entry in &entries {
                if entry.run_length == 0 {
                    DIRECTORY_LEAF_HITS.fetch_add(1, Ordering::Relaxed);
                }
            }

            let bytes = serialize_entries(&entries).unwrap();
            let decoded = deserialize_entries(&bytes).unwrap();
            prop_assert_eq!(&decoded, &entries);

            // Serialising what came back must reproduce the same bytes. A
            // round-trip through the model can hide an encoder with two
            // spellings for one directory; a round-trip through the bytes
            // cannot.
            prop_assert_eq!(serialize_entries(&decoded).unwrap(), bytes);
        }
    );

    let cases = DIRECTORY_CASES.load(Ordering::Relaxed);
    assert!(cases >= STRUCTURAL_CASES as usize, "only {cases} directories were built");
    // Positive controls. If the generator stopped producing contiguous
    // offsets or leaf pointers, this test would still pass while covering
    // neither of the two shapes it exists for.
    assert!(
        DIRECTORY_SHORTHAND_HITS.load(Ordering::Relaxed) > 0,
        "no generated directory used the contiguous-offset shorthand"
    );
    assert!(
        DIRECTORY_LEAF_HITS.load(Ordering::Relaxed) > 0,
        "no generated directory contained a leaf pointer"
    );
}

static RUN_CASES: AtomicUsize = AtomicUsize::new(0);
static RUN_COLLAPSES: AtomicUsize = AtomicUsize::new(0);

/// Building a directory with `push_entry` collapses identical consecutive
/// tiles into runs, and the run lengths still sum to the number of tiles that
/// went in.
///
/// That sum is the invariant the oracle names as the one that catches a reader
/// mishandling deduplication: it is `header.addressed_tiles_count`, and for
/// `dupes-z0z3.pmtiles` it is 67 entries summing to 85.
#[test]
fn run_length_collapsing_preserves_the_addressed_tile_count() {
    proptest!(
        config(STRUCTURAL_CASES),
        |(blobs in proptest::collection::vec((0u64..4, 1u32..4), 1..MAX_GENERATED_ENTRIES))| {
            RUN_CASES.fetch_add(1, Ordering::Relaxed);

            let mut entries: Vec<Entry> = Vec::new();
            for (index, (offset, length)) in blobs.iter().enumerate() {
                push_entry(&mut entries, index as u64, *offset, *length).unwrap();
            }

            let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
            prop_assert_eq!(addressed, blobs.len() as u64,
                "the run lengths stopped summing to the tiles that went in");
            prop_assert!(entries.len() <= blobs.len());
            if entries.len() < blobs.len() {
                RUN_COLLAPSES.fetch_add(1, Ordering::Relaxed);
            }

            // Every tile that went in is still findable through the run rule.
            for id in 0..blobs.len() as u64 {
                prop_assert!(
                    entries.iter().any(|e| e.run_contains(id)),
                    "tile {} fell out of the directory", id
                );
            }

            let decoded = deserialize_entries(&serialize_entries(&entries).unwrap()).unwrap();
            prop_assert_eq!(decoded, entries);
        }
    );
    assert!(RUN_CASES.load(Ordering::Relaxed) >= STRUCTURAL_CASES as usize);
    assert!(
        RUN_COLLAPSES.load(Ordering::Relaxed) > 0,
        "no generated sequence ever collapsed into a run, so the property \
         never exercised the thing it is about"
    );
}

static HEADER_CASES: AtomicUsize = AtomicUsize::new(0);

/// `Header::encode` and `Header::try_decode` are inverses for every field.
///
/// The enums go in through `from_byte` rather than being picked from the named
/// variants, so `Other(0x2A)` is generated as often as `Png` is and the
/// unknown-value path is covered rather than assumed.
#[test]
fn headers_round_trip_through_their_127_bytes() {
    proptest!(
        config(STRUCTURAL_CASES),
        |(
            offsets in proptest::collection::vec(any::<u64>(), 11),
            flags in proptest::collection::vec(any::<u8>(), 6),
            coords in proptest::collection::vec(any::<i32>(), 6),
            clustered in any::<bool>(),
        )| {
            HEADER_CASES.fetch_add(1, Ordering::Relaxed);

            let header = Header {
                root_offset: offsets[0],
                root_length: offsets[1],
                metadata_offset: offsets[2],
                metadata_length: offsets[3],
                leaf_directories_offset: offsets[4],
                leaf_directories_length: offsets[5],
                tile_data_offset: offsets[6],
                tile_data_length: offsets[7],
                addressed_tiles_count: offsets[8],
                tile_entries_count: offsets[9],
                tile_contents_count: offsets[10],
                clustered,
                internal_compression: Compression::from_byte(flags[0]),
                tile_compression: Compression::from_byte(flags[1]),
                tile_type: TileType::from_byte(flags[2]),
                min_zoom: flags[3],
                max_zoom: flags[4],
                center_zoom: flags[5],
                min_lon_e7: coords[0],
                min_lat_e7: coords[1],
                max_lon_e7: coords[2],
                max_lat_e7: coords[3],
                center_lon_e7: coords[4],
                center_lat_e7: coords[5],
            };

            let bytes = header.encode();
            prop_assert_eq!(bytes.len(), 127);
            prop_assert_eq!(&bytes[..7], b"PMTiles");
            prop_assert_eq!(bytes[7], 3);
            prop_assert_eq!(Header::try_decode(&bytes).unwrap(), header);

            // A longer buffer is accepted and only the first 127 bytes are
            // read, which is what lets a reader fetch header plus root
            // directory in one range and hand the whole thing over.
            let mut padded = bytes.to_vec();
            padded.extend_from_slice(&[0xAB; 64]);
            prop_assert_eq!(Header::try_decode(&padded).unwrap(), header);

            // One byte short is a refusal, not a partial decode.
            prop_assert!(
                matches!(
                    Header::try_decode(&bytes[..126]),
                    Err(PmTilesError::ShortHeader { got: 126, want: 127 })
                ),
                "126 bytes decoded as a header instead of being refused"
            );

            // The enum bytes survive a value no revision defines.
            prop_assert_eq!(header.internal_compression.to_byte(), flags[0]);
            prop_assert_eq!(header.tile_compression.to_byte(), flags[1]);
            prop_assert_eq!(header.tile_type.to_byte(), flags[2]);
        }
    );
    assert!(HEADER_CASES.load(Ordering::Relaxed) >= STRUCTURAL_CASES as usize);
}

static BAD_HEADER_CASES: AtomicUsize = AtomicUsize::new(0);

/// Arbitrary bytes are either a header or a typed refusal, never a panic, and
/// the magic and version are the only two things that can refuse them.
#[test]
fn arbitrary_bytes_are_never_decoded_as_a_header_by_accident() {
    proptest!(
        config(STRUCTURAL_CASES),
        |(bytes in proptest::collection::vec(any::<u8>(), 0..300))| {
            BAD_HEADER_CASES.fetch_add(1, Ordering::Relaxed);
            match Header::try_decode(&bytes) {
                Ok(_) => {
                    prop_assert!(bytes.len() >= 127);
                    prop_assert_eq!(&bytes[..7], b"PMTiles");
                    prop_assert_eq!(bytes[7], 3);
                }
                Err(PmTilesError::ShortHeader { got, want }) => {
                    prop_assert_eq!((got, want), (bytes.len(), 127));
                }
                Err(PmTilesError::BadMagic { found }) => {
                    prop_assert_ne!(&found[..], b"PMTiles");
                }
                Err(PmTilesError::UnsupportedVersion { found }) => {
                    prop_assert_eq!(&bytes[..7], b"PMTiles");
                    prop_assert_ne!(found, 3);
                }
                Err(other) => prop_assert!(false, "unexpected refusal: {:?}", other),
            }
        }
    );
    assert!(BAD_HEADER_CASES.load(Ordering::Relaxed) >= STRUCTURAL_CASES as usize);
}
