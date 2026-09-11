#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::pmtiles::directory::deserialize_entries;
use libviprs::pmtiles::tileid::MAX_TILE_ID;
use libviprs::pmtiles::validate::{ValidationLimits, validate_bytes};
use libviprs::pmtiles::varint::{decode_uvarint, encode_uvarint, uvarint_len};
use libviprs::pmtiles::{Header, tileid_to_zxy, zxy_to_tileid};

/// Fuzz every PMTiles surface that eats bytes somebody else wrote.
///
/// A `.pmtiles` archive is a file handed over by whoever made it, usually
/// fetched over HTTP from a bucket nobody in this process controls, and the
/// format is built out of exactly the primitives that go wrong on hostile
/// input: variable-length integers, four parallel columns of them, offsets and
/// lengths that index into the same file, and a gzip stream whose decompressed
/// size no field states.
///
/// The reference implementation does not survive this. `pmtiles verify` in
/// go-pmtiles v1.31.2 bounds-checks four lengths against the file size and
/// never checks `offset + length`, so a root offset of 999999 in an 1878-byte
/// archive walks straight through the validator, and `DeserializeEntries`
/// then does `reader, _ = gzip.NewReader(data)` and dereferences the nil
/// reader. That is a clean SIGSEGV out of `compress/gzip` on a file anyone can
/// make, and `nocrash-root-offset-past-the-end` in the seed corpus is exactly
/// it. This target exists so the same input is a typed finding here.
///
/// Four entry points, reached directly rather than through one wrapper,
/// because a fuzzer that has to guess a 127-byte header before it can reach
/// the varint decoder spends its whole budget on the header:
///
/// * [`validate_bytes`], which is the whole-archive walk: header, section
///   bounds, root directory, every leaf behind it;
/// * `Header::try_decode`, on the raw input;
/// * `deserialize_entries`, with the input taken as an already-decompressed
///   directory body, which is what a reader hands it after the gzip;
/// * `decode_uvarint`, the primitive the other three are built from.
///
/// The assertions are properties rather than fixtures, so a failure is a real
/// contradiction and not a stale expectation.
///
/// # Two passes, and the second one is the one that ships
///
/// The walk runs twice over the same input. The first pass uses deliberately
/// small limits, because a fuzzer will find a directory claiming a huge entry
/// count in seconds and the point there is to prove the claim is refused
/// rather than to watch it be honoured. Those small limits used to be the only
/// pass, and that was a hole: `max_leaf_directories = 64` and
/// `max_directory_bytes = 1 << 20` are precisely the two knobs whose **default**
/// values multiply into an unbounded walk, so the target structurally could
/// not reach the bug the CLI ships with. A 21,620-byte archive whose root
/// holds 1,048,577 leaf pointers that all resolve to one valid 16 MiB leaf
/// asks for roughly 16 TiB of gzip output at `ValidationLimits::default()`,
/// and for 64 MiB under the limits above, which is over in a blink and looks
/// like nothing.
///
/// So the second pass runs at `ValidationLimits::default()`, which is what
/// `viprs pmtiles verify` uses. It costs nothing on ordinary input, because
/// only an input that actually carries a fat gzip stream can spend the budget,
/// and the budget is what makes even that bounded. `nocrash-leaf-fanout` in
/// the seed corpus is the archive above.
///
/// # The seed corpus
///
/// `fuzz/corpus/fuzz_pmtiles_reader/` holds the three golden archives
/// go-pmtiles wrote (`valid-*`), inputs that must be refused at the first
/// check (`rejected-*`), and inputs that must produce findings without
/// crashing (`nocrash-*`). Each `nocrash-` seed is a named structural defect:
/// a section offset past the end of the file, a section whose offset plus
/// length overflows a `u64`, a directory pointed at bytes that are not gzip, a
/// leaf pointer resolving outside the leaf region, a directory header claiming
/// more entries than its own bytes could hold, and a varint that never
/// terminates.
///
/// `nocrash-leaf-fanout` is the odd one out and the most valuable: nothing
/// about it is malformed. 21,620 bytes, a header that decodes, a root of
/// 1,048,577 leaf pointers every one of which is in bounds, and one valid
/// 16 MiB leaf of 4,194,303 in-bounds entries behind all of them. It is the
/// amplification seed, and the thing it is a regression test for is the walk
/// returning at all.
fuzz_target!(|data: &[u8]| {
    // Small on purpose. `max_directory_bytes` is the decompression ceiling,
    // and a gzip bomb is the cheapest thing for a fuzzer to stumble into.
    let limits = ValidationLimits::default()
        .with_max_directory_bytes(1 << 20)
        .with_max_leaf_directories(64);

    if let Ok(report) = validate_bytes(data, &limits) {
        // A clean report claims the archive is structurally sound, so the
        // header it walked past must itself decode. If this ever fires, the
        // walk is reporting on an archive it did not actually parse.
        if report.is_valid() {
            assert!(
                report.header.is_some(),
                "a report with no findings and no header"
            );
            assert!(
                Header::try_decode(data).is_ok(),
                "the validator found nothing wrong with bytes the header decoder refuses"
            );
        }
        // Leaf bookkeeping has to stay consistent with the limits it works
        // under, or the bound is not a bound.
        assert!(report.leaves.len() <= 64);
    }

    // The same walk at the limits the CLI ships with. This is the pass that
    // would have caught the leaf storm, and the assertions are about the work
    // the walk admits to doing rather than about the verdict.
    let shipped = ValidationLimits::default();
    if let Ok(report) = validate_bytes(data, &shipped) {
        // The total-work budget is what makes this pass affordable. It is
        // derived from the archive's own section lengths and floored at one
        // directory, and the walk may overshoot it by the directory it was
        // already reading when it ran out.
        let floor = shipped.max_directory_bytes as u64;
        let budget = report
            .header
            .as_ref()
            .map(|header| {
                header
                    .root_length
                    .saturating_add(header.leaf_directories_length)
                    .saturating_mul(64)
                    .max(floor)
                    .min(shipped.max_total_directory_bytes)
            })
            .unwrap_or(floor);
        assert!(
            report.directory_bytes <= budget.saturating_add(floor),
            "the walk inflated {} bytes against a {budget} byte budget",
            report.directory_bytes
        );
        // One directory's worth of overshoot, and a directory body spends at
        // least four bytes an entry, so the ceiling on that overshoot is the
        // per-directory byte cap.
        assert!(
            report.entries_visited <= shipped.max_total_entries.saturating_add(floor),
            "the walk visited {} entries against a {} budget",
            report.entries_visited,
            shipped.max_total_entries
        );
        // Anything that stopped the walk early has to have said so, or a
        // clean-looking report is one nobody can act on.
        assert!(
            report.shared_leaf_pointers == 0 || !report.findings.is_empty(),
            "leaf pointers were collapsed and nothing was reported"
        );
    }

    // The header, on the raw bytes. Total over every input: either a header or
    // a typed refusal.
    if let Ok(header) = Header::try_decode(data) {
        assert!(data.len() >= 127);
        // `has_leaves` must read the length and never the offset. Every
        // archive go-pmtiles writes without leaves still carries a non-zero
        // leaf offset, equal to `tile_data_offset`.
        assert_eq!(header.has_leaves(), header.leaf_directories_length > 0);
        // Encoding what was decoded and decoding that again must land on the
        // same header. Not a byte-for-byte comparison against the input: the
        // clustered field is a bool on the wire in name only, any non-zero
        // byte reads as true, and the encoder writes 1, so an input carrying
        // 0x42 there is a legitimate archive whose re-encoding differs by one
        // byte. What has to hold is that no field was lost.
        assert_eq!(Header::try_decode(&header.encode()).ok(), Some(header));
    }

    // The input as an already-decompressed directory body.
    if let Ok(entries) = deserialize_entries(data) {
        assert!(!entries.is_empty(), "an empty directory must be refused");
        let mut previous: Option<u64> = None;
        for entry in &entries {
            assert!(entry.length > 0, "a zero-length entry must be refused");
            assert!(
                entry.tile_id <= MAX_TILE_ID,
                "an unaddressable tile id must be refused"
            );
            if let Some(previous) = previous {
                assert!(
                    entry.tile_id > previous,
                    "the id column is an unsigned delta, so ids can only ascend"
                );
            }
            previous = Some(entry.tile_id);
        }
    }

    // The varint primitive, on the raw bytes.
    if let Ok((value, used)) = decode_uvarint(data, 0) {
        assert!(used >= 1 && used <= data.len());
        assert!(used <= 10, "a u64 varint is at most ten bytes");
        // LEB128 has non-minimal spellings and this decoder accepts them, the
        // way Go's `binary.Uvarint` does, so the canonical form may be shorter
        // than what was consumed. It must never be longer: that would mean the
        // decoder stopped early and lost bits.
        assert!(uvarint_len(value) <= used);
        let mut round = Vec::new();
        encode_uvarint(value, &mut round);
        let again = decode_uvarint(&round, 0).expect("a canonical encoding decodes");
        assert_eq!(again, (value, round.len()));
    }

    // The tile id mapping, driven from the front of the input so a fuzzer can
    // steer it. Both directions, because an encoder and a decoder that share a
    // wrong orientation agree with each other perfectly.
    if data.len() >= 8 {
        let raw = u64::from_le_bytes(data[..8].try_into().expect("eight bytes"));
        let id = raw % (MAX_TILE_ID + 1);
        let (z, x, y) = tileid_to_zxy(id).expect("an id inside the addressable range decodes");
        assert!(z <= 31);
        assert!(u64::from(x) < 1u64 << z, "x escaped its level's grid");
        assert!(u64::from(y) < 1u64 << z, "y escaped its level's grid");
        assert_eq!(
            zxy_to_tileid(z, x, y).expect("a decoded coordinate re-encodes"),
            id
        );
        // Above the addressable range is a refusal, never a wrap. The
        // reference saturates here instead and returns one id for every zoom
        // above 31.
        assert!(tileid_to_zxy(MAX_TILE_ID.wrapping_add(1 + (raw % 4096))).is_err());
    }
});
