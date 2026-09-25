//! The PMTiles v3 streaming writer, pinned against go-pmtiles (issue #989).
//!
//! # Why none of this round-trips through our own reader
//!
//! F1.2 is writing the indexed reader while this is being written, and the two
//! are halves of one contract. **A reader and a writer that share a spec
//! misreading round-trip perfectly and both look correct.** So nothing here
//! asks our reader anything. Every expectation comes from the three golden
//! archives under `tests/fixtures/pmtiles/`, which were produced by
//! `go-pmtiles` v1.31.2 (`protomaps/go-pmtiles`, commit
//! `a3e4951ea6a0477b784c27c1dcbfd9c130878c5a`) and are pinned here by sha256.
//!
//! Each golden is parsed with nothing but `flate2` and the format module, its
//! own header counts are checked against what the parse produced (so a broken
//! parse fails loudly instead of passing on the empty set), and the tiles it
//! contains are then fed back into [`Writer`] in a shuffled order. What comes
//! out is compared **byte for byte** against the bytes go-pmtiles wrote.
//!
//! That comparison is possible because of a measurement the oracle lane made:
//! `pmtiles.SerializeEntries`, run over the entries `DeserializeEntries` had
//! just decoded, reproduces the gunzipped directory body exactly on all three
//! goldens. The decompressed directory is therefore a real byte-level target
//! and not merely a convenient dump.
//!
//! # The four things a writer gets wrong, and where each is caught
//!
//! * **The offset column is not delta encoded.** It stores `offset + 1`, with
//!   a literal `0` meaning "this blob starts where the previous one ended".
//!   Only the tile id column is delta encoded, and **there is no zigzag
//!   anywhere** despite the delta encoding. `dupes-z0z3` is the fixture that
//!   proves it: entry 1 sits at offset 74 and entry 2 at offset 0, a backwards
//!   jump, and the raw uvarint in the bytes is `1` rather than the `147` a
//!   zigzagged `-74` would need. Caught by
//!   [`the_root_directory_is_byte_identical_to_the_one_go_pmtiles_wrote`].
//! * **The leaf offset base.** A tile entry found inside a leaf is relative to
//!   `tile_data_offset`, not to the leaf's own start and not to
//!   `leaf_directories_offset`. `leaves-z0z7` is the only fixture anywhere
//!   that exercises it. Caught by
//!   [`a_tile_entry_inside_a_leaf_is_relative_to_the_tile_data_section`].
//! * **Both shapes of dedupe.** Identical tiles at consecutive ids collapse
//!   into one entry with `run_length > 1`; identical tiles that are *not*
//!   consecutive stay separate entries pointing at the **same offset**. A
//!   writer that only does the first produces a valid but different archive.
//!   `dupes-z0z3` has both, deliberately. Caught by
//!   [`non_adjacent_duplicates_share_one_payload_without_sharing_an_entry`].
//! * **`leaf_directories_offset` is non-zero when there are no leaves.** It
//!   equals `tile_data_offset`, and the *length* is the flag. Caught by
//!   [`the_leaf_offset_is_not_the_flag_the_leaf_length_is`].

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use libviprs::pmtiles::directory::{deserialize_entries, serialize_entries};
use libviprs::pmtiles::writer::{DEDUPE_WINDOW_WAYS, Layout, Writer, WriterOptions, content_hash};
use libviprs::pmtiles::{Compression, Entry, Header, Metadata, PmTilesError, TileType};

// ---------------------------------------------------------------------------
// The goldens, and the parse that keeps itself honest
// ---------------------------------------------------------------------------

/// The three go-pmtiles archives this file is pinned to, with the sha256 of
/// the bytes on disk.
///
/// The hash is not decoration. These fixtures are the only thing separating
/// this suite from testing our writer against our own reading of the spec, so
/// a fixture that quietly changed would take the whole argument with it and
/// leave every assertion below still green.
const GOLDENS: &[(&str, &str)] = &[
    (
        "raster-z0z2.pmtiles",
        "e2ed5e64f3c29efa3ec3b679ec5f1b06569c1b234c6eea762fb9f02fc23e9c12",
    ),
    (
        "dupes-z0z3.pmtiles",
        "bfc9db4c6ce6a04194e02b3d4815814adb05209f1aaba8591e4e1332f6e56a27",
    ),
    (
        "leaves-z0z7.pmtiles",
        "fe5c9636be61abc60046d7f13837f8a3efb20ce3c38303644dac0cbec8248b8d",
    ),
];

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pmtiles")
}

/// Read one golden and refuse it if the bytes are not the ones this file was
/// written against.
fn golden(name: &str) -> Vec<u8> {
    let want = GOLDENS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, h)| *h)
        .unwrap_or_else(|| panic!("{name} is not one of the pinned goldens"));

    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("golden {} is not readable: {e}", path.display()));

    use sha2::Digest;
    let got: String = sha2::Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        got, want,
        "golden {name} is not the archive this suite pins"
    );
    bytes
}

/// Everything this suite needs out of a golden archive: its header, every
/// directory entry with runs still packed, and the tile payload behind each
/// addressed tile id.
struct Golden {
    header: Header,
    /// Entries from the root and every leaf, concatenated in tile id order.
    entries: Vec<Entry>,
    /// One `(tile_id, payload)` per addressed tile, runs expanded, ascending.
    tiles: Vec<(u64, Vec<u8>)>,
    bytes: Vec<u8>,
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_end(&mut out)
        .expect("a golden's gzip members decompress");
    out
}

/// Parse a golden with nothing but `flate2` and the format module, then check
/// the parse against the three counts the archive states about itself.
///
/// The self-check is the positive control. Every assertion downstream is
/// derived from `entries` and `tiles`, so a parse that produced nothing would
/// make all of them vacuous; requiring the parse to reproduce
/// `addressed_tiles_count`, `tile_entries_count` and `tile_contents_count`
/// means a silent mis-parse fails here instead of passing everywhere.
fn parse_golden(name: &str) -> Golden {
    let bytes = golden(name);
    let header = Header::try_decode(&bytes[..127]).expect("a golden has a decodable header");

    let root_slice = section(&bytes, header.root_offset, header.root_length);
    let mut entries = Vec::new();
    for entry in deserialize_entries(&gunzip(root_slice)).expect("a golden's root decodes") {
        if entry.is_leaf() {
            let leaf_at = header.leaf_directories_offset + entry.offset;
            let leaf = section(&bytes, leaf_at, u64::from(entry.length));
            entries.extend(deserialize_entries(&gunzip(leaf)).expect("a golden's leaf decodes"));
        } else {
            entries.push(entry);
        }
    }

    let mut tiles = Vec::new();
    let mut contents = BTreeSet::new();
    for entry in &entries {
        assert!(
            !entry.is_leaf(),
            "no golden here nests leaves inside leaves"
        );
        let at = header.tile_data_offset + entry.offset;
        let payload = section(&bytes, at, u64::from(entry.length)).to_vec();
        contents.insert((entry.offset, entry.length));
        for k in 0..u64::from(entry.run_length) {
            tiles.push((entry.tile_id + k, payload.clone()));
        }
    }

    assert_eq!(
        entries.len() as u64,
        header.tile_entries_count,
        "{name}: parsed a different number of entries than the header states"
    );
    assert_eq!(
        tiles.len() as u64,
        header.addressed_tiles_count,
        "{name}: the run lengths do not sum to the addressed tile count"
    );
    assert_eq!(
        contents.len() as u64,
        header.tile_contents_count,
        "{name}: parsed a different number of distinct blobs than the header states"
    );
    assert!(
        tiles.windows(2).all(|w| w[0].0 < w[1].0),
        "{name}: tile ids are not strictly ascending"
    );

    Golden {
        header,
        entries,
        tiles,
        bytes,
    }
}

fn section(bytes: &[u8], offset: u64, length: u64) -> &[u8] {
    let start = usize::try_from(offset).expect("a golden fits in memory");
    let end = start + usize::try_from(length).expect("a golden fits in memory");
    &bytes[start..end]
}

/// Shuffle deterministically, so "any arrival order" is exercised without the
/// suite being able to pass or fail differently run to run.
fn shuffled<T>(items: &mut [T], seed: u64) {
    // xorshift64*, enough to scramble an insertion order and reproducible
    // without a dependency.
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Feed a golden's own tiles into a fresh [`Writer`], in a shuffled order, and
/// return the archive that comes out.
fn rewrite(g: &Golden, dir: &Path, seed: u64) -> (PathBuf, Vec<u8>) {
    let mut tiles: Vec<(u64, Vec<u8>)> = g.tiles.clone();
    shuffled(&mut tiles, seed);

    let out = dir.join(format!("rewritten-{seed}.pmtiles"));
    let mut writer = Writer::create(
        &out,
        WriterOptions::default()
            .with_tile_type(g.header.tile_type)
            .with_tile_compression(g.header.tile_compression),
    )
    .expect("a writer opens");
    for (tile_id, payload) in &tiles {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
        writer
            .add_tile(z, x, y, payload, content_hash(payload))
            .expect("a tile is accepted");
    }
    let finished = writer.finish().expect("the archive finalises");
    assert_eq!(finished.path.as_deref(), Some(out.as_path()));

    let bytes = std::fs::read(&out).expect("the archive is on disk");
    (out, bytes)
}

/// Pull the root and every leaf out of an archive we wrote, the same way
/// [`parse_golden`] does, so the two can be compared field by field.
fn parse_ours(bytes: &[u8]) -> Golden {
    let header = Header::try_decode(&bytes[..127]).expect("our header decodes");
    let mut entries = Vec::new();
    for entry in deserialize_entries(&gunzip(section(
        bytes,
        header.root_offset,
        header.root_length,
    )))
    .expect("our root decodes")
    {
        if entry.is_leaf() {
            let leaf = section(
                bytes,
                header.leaf_directories_offset + entry.offset,
                u64::from(entry.length),
            );
            entries.extend(deserialize_entries(&gunzip(leaf)).expect("our leaf decodes"));
        } else {
            entries.push(entry);
        }
    }
    let mut tiles = Vec::new();
    for entry in &entries {
        let payload = section(
            bytes,
            header.tile_data_offset + entry.offset,
            u64::from(entry.length),
        )
        .to_vec();
        for k in 0..u64::from(entry.run_length) {
            tiles.push((entry.tile_id + k, payload.clone()));
        }
    }
    Golden {
        header,
        entries,
        tiles,
        bytes: bytes.to_vec(),
    }
}

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().expect("a scratch directory")
}

// ---------------------------------------------------------------------------
// The byte-for-byte pin
// ---------------------------------------------------------------------------

/// The decompressed root directory our writer produces is **identical, byte
/// for byte**, to the one go-pmtiles produced from the same tiles.
///
/// This is the strongest evidence available to this lane and it is the reason
/// the writer lays its data region out in tile id order. go-pmtiles' `convert`
/// assigns a payload's offset the first time that payload is reached in tile
/// id order, so a writer doing the same thing lands every blob at the same
/// offset, and every column of the directory then has to agree with the
/// reference or this fails.
///
/// It is also the negative control for the offset column. Encoding that column
/// as a plain delta, or as a zigzag delta, still round-trips through our own
/// decoder and still produces ascending tile ids, and this is the test that
/// says no.
#[test]
#[cfg_attr(miri, ignore)]
fn the_root_directory_is_byte_identical_to_the_one_go_pmtiles_wrote() {
    let dir = scratch();
    for name in ["raster-z0z2.pmtiles", "dupes-z0z3.pmtiles"] {
        let g = parse_golden(name);
        assert_eq!(
            g.header.leaf_directories_length, 0,
            "{name} is supposed to be a leafless golden"
        );

        let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0001);
        let theirs = gunzip(section(
            &g.bytes,
            g.header.root_offset,
            g.header.root_length,
        ));
        let mine = parse_ours(&ours);
        let mine_root = gunzip(section(
            &ours,
            mine.header.root_offset,
            mine.header.root_length,
        ));

        assert_eq!(
            mine_root.len(),
            theirs.len(),
            "{name}: our root directory is a different length than go-pmtiles'"
        );
        assert_eq!(
            mine_root, theirs,
            "{name}: our root directory is not the bytes go-pmtiles wrote"
        );

        // And the tile data section, which is what makes the offsets in that
        // directory mean the same thing.
        let their_data = section(
            &g.bytes,
            g.header.tile_data_offset,
            g.header.tile_data_length,
        );
        let our_data = section(
            &ours,
            mine.header.tile_data_offset,
            mine.header.tile_data_length,
        );
        assert_eq!(
            our_data, their_data,
            "{name}: our tile data section is not the bytes go-pmtiles wrote"
        );
    }
}

/// The same pin stated the other way round: our serializer, handed the entries
/// go-pmtiles' own decoder produced, reproduces the reference bytes.
///
/// This one does not depend on any layout decision the writer makes, so it
/// isolates the encoding itself. The oracle measured that
/// `pmtiles.SerializeEntries` does exactly this, which is what makes the
/// decompressed directory a legitimate target rather than a dump.
#[test]
#[cfg_attr(miri, ignore)]
fn serialising_the_reference_entries_reproduces_the_reference_bytes() {
    for name in [
        "raster-z0z2.pmtiles",
        "dupes-z0z3.pmtiles",
        "leaves-z0z7.pmtiles",
    ] {
        let g = parse_golden(name);
        let root = gunzip(section(
            &g.bytes,
            g.header.root_offset,
            g.header.root_length,
        ));
        let decoded = deserialize_entries(&root).expect("the root decodes");
        assert!(!decoded.is_empty(), "{name}: the root decoded to nothing");
        assert_eq!(
            serialize_entries(&decoded).expect("the root re-serialises"),
            root,
            "{name}: re-serialising the root did not reproduce its bytes"
        );

        // Every leaf too, which is where the offset column's shorthand gets a
        // real workout: `leaves-z0z7` alternates between two payloads, so half
        // its entries take the contiguous branch and half do not.
        for entry in &decoded {
            if !entry.is_leaf() {
                continue;
            }
            let leaf = gunzip(section(
                &g.bytes,
                g.header.leaf_directories_offset + entry.offset,
                u64::from(entry.length),
            ));
            let leaf_entries = deserialize_entries(&leaf).expect("a leaf decodes");
            assert!(
                !leaf_entries.is_empty(),
                "{name}: a leaf decoded to nothing"
            );
            assert_eq!(
                serialize_entries(&leaf_entries).expect("a leaf re-serialises"),
                leaf,
                "{name}: re-serialising a leaf did not reproduce its bytes"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Leaves
// ---------------------------------------------------------------------------

/// A tile entry inside a leaf is relative to `tile_data_offset`.
///
/// Not the leaf's own start, and not `leaf_directories_offset`. The base is
/// chosen by the entry's *kind*, not by the directory it was found in, and
/// this is the only fixture anywhere that can tell the three apart: a leafless
/// archive has `leaf_directories_offset == tile_data_offset`, so two of the
/// three wrong answers coincide with the right one.
///
/// The test carries its own negative control. It resolves a leaf-held tile
/// through each of the three candidate bases and requires the other two to
/// disagree, because an archive whose leaf section happened to sit at the same
/// place as its tile data would make this pass for the wrong reason.
#[test]
#[cfg_attr(miri, ignore)]
fn a_tile_entry_inside_a_leaf_is_relative_to_the_tile_data_section() {
    let dir = scratch();
    let g = parse_golden("leaves-z0z7.pmtiles");
    assert!(
        g.header.leaf_directories_length > 0,
        "leaves-z0z7 is supposed to have leaves"
    );

    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0002);
    let header = Header::try_decode(&ours[..127]).expect("our header decodes");
    assert!(
        header.leaf_directories_length > 0,
        "our archive of 21845 tiles should not have fitted in the root alone"
    );

    let root = deserialize_entries(&gunzip(section(
        &ours,
        header.root_offset,
        header.root_length,
    )))
    .expect("our root decodes");
    assert!(
        root.iter().all(|e| e.is_leaf()),
        "a root that spilled into leaves holds only leaf pointers"
    );

    // Take the last leaf's last entry, the furthest possible thing from the
    // start of the file.
    let pointer = *root.last().expect("at least one leaf");
    let leaf_start = header.leaf_directories_offset + pointer.offset;
    let leaf = deserialize_entries(&gunzip(section(
        &ours,
        leaf_start,
        u64::from(pointer.length),
    )))
    .expect("our leaf decodes");
    let entry = *leaf.last().expect("a leaf is never empty");

    let want: &Vec<u8> = &g
        .tiles
        .iter()
        .find(|(id, _)| *id == entry.tile_id)
        .expect("the golden holds that tile")
        .1;

    let right = section(
        &ours,
        header.tile_data_offset + entry.offset,
        u64::from(entry.length),
    );
    assert_eq!(right, &want[..], "the tile data offset is the base");

    // Negative control: the other two candidate bases must land somewhere
    // else. If they did not, this test could not tell them apart and would be
    // green for a writer that used either.
    assert_ne!(
        header.tile_data_offset, header.leaf_directories_offset,
        "this fixture cannot discriminate the bases unless the sections differ"
    );
    assert_ne!(
        header.tile_data_offset, leaf_start,
        "this fixture cannot discriminate the bases unless the leaf sits elsewhere"
    );
}

/// Every tile in every leaf resolves through `tile_data_offset`, on an archive
/// whose leaves do **not** all start at offset 0.
///
/// # Why the golden cannot do this on its own
///
/// I found this with a mutation and it is worth writing down, because it is a
/// hole in the only leaf-bearing fixture that exists. `leaves-z0z7` holds two
/// distinct payloads, at offsets 0 and 72, alternating, and **all six of its
/// leaves have a first entry at offset 0**. So "relative to the leaf's own
/// start" and "relative to the tile data section" produce byte-identical
/// archives for it, and the test above, which pins the golden, stays green for
/// a writer that rebases its leaf entries onto the leaf. I mutated the writer
/// to do exactly that and nothing went red.
///
/// This builds the fixture the golden is not: 16384 tiles with 16384 distinct
/// payloads, forced into eight leaves, so leaf `k`'s first entry sits far from
/// zero and a rebase changes every offset in seven leaves out of eight. The
/// positive control below is the one the golden fails, and it is what makes
/// the sweep over every entry mean something.
#[test]
#[cfg_attr(miri, ignore)]
fn every_tile_in_every_leaf_resolves_through_the_tile_data_offset() {
    let dir = scratch();
    let out = dir.path().join("many-leaves.pmtiles");

    // Zoom 7 is exactly 16384 tiles, ids 5461..=21844, which is also the point
    // at which the writer stops trying to fit everything in the root.
    let ids: Vec<u64> = (5461u64..5461 + 16384).collect();
    let payload_for = |id: u64| format!("a distinct payload for tile {id}").into_bytes();

    let mut w = Writer::create(
        &out,
        WriterOptions::default()
            .with_tile_type(TileType::Png)
            .with_leaf_entries(2048),
    )
    .unwrap();
    for id in &ids {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*id).unwrap();
        let payload = payload_for(*id);
        w.add_tile(z, x, y, &payload, content_hash(&payload))
            .unwrap();
    }
    w.finish().unwrap();

    let bytes = std::fs::read(&out).unwrap();
    let header = Header::try_decode(&bytes[..127]).unwrap();
    assert!(
        header.leaf_directories_length > 0,
        "this should have spilled"
    );

    let root = deserialize_entries(&gunzip(section(
        &bytes,
        header.root_offset,
        header.root_length,
    )))
    .unwrap();
    assert!(
        root.len() >= 8,
        "expected at least eight leaves, got {}",
        root.len()
    );

    let mut leaf_starts = Vec::new();
    let mut checked = 0usize;
    let mut leaf_base_would_differ = 0usize;
    for pointer in &root {
        assert!(pointer.is_leaf());
        let leaf_start = header.leaf_directories_offset + pointer.offset;
        let leaf = deserialize_entries(&gunzip(section(
            &bytes,
            leaf_start,
            u64::from(pointer.length),
        )))
        .unwrap();
        leaf_starts.push(leaf[0].offset);
        for entry in &leaf {
            let want = payload_for(entry.tile_id);
            let got = section(
                &bytes,
                header.tile_data_offset + entry.offset,
                u64::from(entry.length),
            );
            assert_eq!(
                got,
                &want[..],
                "tile {} resolved to the wrong bytes",
                entry.tile_id
            );
            checked += 1;
            // The negative control, counted rather than asserted one at a
            // time: resolving through the leaf's own first entry instead of
            // the tile data section has to land somewhere else for this
            // fixture to be discriminating at all.
            if leaf[0].offset != 0 {
                leaf_base_would_differ += 1;
            }
        }
    }

    assert_eq!(checked, 16384, "the sweep did not visit every tile");
    // The positive control the golden cannot give: leaves that start away from
    // zero. Without this the whole sweep above passes for a rebasing writer.
    let away_from_zero = leaf_starts.iter().filter(|o| **o != 0).count();
    assert!(
        away_from_zero >= 7,
        "only {away_from_zero} of {} leaves start away from offset 0, so this \
         fixture cannot tell the two bases apart",
        leaf_starts.len()
    );
    assert!(leaf_base_would_differ >= 14000);
}

/// `leaf_directories_offset` is not zero when there are no leaves: it equals
/// `tile_data_offset`, and the **length** is the flag.
///
/// go-pmtiles writes it that way on every leafless archive it produces, so a
/// reader testing `leaf_directories_offset != 0` to decide whether leaves
/// exist gets the wrong answer on all of them, and a writer that zeroed the
/// offset would be the one archive in the world where that reader worked.
#[test]
#[cfg_attr(miri, ignore)]
fn the_leaf_offset_is_not_the_flag_the_leaf_length_is() {
    let dir = scratch();
    let g = parse_golden("raster-z0z2.pmtiles");
    assert_eq!(g.header.leaf_directories_length, 0);
    assert_eq!(g.header.leaf_directories_offset, g.header.tile_data_offset);

    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0003);
    let header = Header::try_decode(&ours[..127]).unwrap();
    assert_eq!(header.leaf_directories_length, 0);
    assert_ne!(
        header.leaf_directories_offset, 0,
        "a leafless archive still points its leaf section somewhere real"
    );
    assert_eq!(header.leaf_directories_offset, header.tile_data_offset);
    assert!(!header.has_leaves());
}

/// The root directory stays inside the first 16384 bytes of the archive, which
/// is the one size rule the spec makes a `MUST`.
#[test]
#[cfg_attr(miri, ignore)]
fn the_root_directory_fits_the_sixteen_kilobyte_budget() {
    let dir = scratch();
    let g = parse_golden("leaves-z0z7.pmtiles");
    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0004);
    let header = Header::try_decode(&ours[..127]).unwrap();
    assert_eq!(header.root_offset, 127);
    assert!(
        header.root_offset + header.root_length <= 16384,
        "root at {} for {} bytes runs past the 16 KiB ceiling",
        header.root_offset,
        header.root_length
    );
    // Positive control: an archive that did not need leaves would satisfy the
    // bound trivially, so require this one to have actually spilled.
    assert!(header.leaf_directories_length > 0);
}

// ---------------------------------------------------------------------------
// Dedupe, in both of its shapes
// ---------------------------------------------------------------------------

/// Identical tiles at consecutive ids become **one** entry with a run length,
/// and identical tiles that are not consecutive become separate entries
/// sharing **one** offset.
///
/// `dupes-z0z3` has both on purpose: run lengths of 4 and 16, and offset 148
/// shared by four entries that are not adjacent. A writer that only collapses
/// adjacent runs stores the non-adjacent duplicates again and produces a
/// valid but different archive, which is why the counts are checked against
/// the golden's rather than merely against each other.
#[test]
#[cfg_attr(miri, ignore)]
fn non_adjacent_duplicates_share_one_payload_without_sharing_an_entry() {
    let dir = scratch();
    let g = parse_golden("dupes-z0z3.pmtiles");
    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0005);
    let mine = parse_ours(&ours);

    assert_eq!(
        mine.entries, g.entries,
        "our entry list differs from theirs"
    );
    assert_eq!(
        mine.header.addressed_tiles_count,
        g.header.addressed_tiles_count
    );
    assert_eq!(mine.header.tile_entries_count, g.header.tile_entries_count);
    assert_eq!(
        mine.header.tile_contents_count,
        g.header.tile_contents_count
    );

    // Shape one: runs. Positive control on the fixture itself, so a golden
    // that lost its runs would fail here rather than make the check vacuous.
    let runs: Vec<u32> = mine
        .entries
        .iter()
        .map(|e| e.run_length)
        .filter(|r| *r > 1)
        .collect();
    assert_eq!(runs, vec![4, 16], "the two runs this fixture carries");

    // Shape two: non-adjacent entries pointing at one offset.
    let mut by_offset: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, entry) in mine.entries.iter().enumerate() {
        by_offset.entry(entry.offset).or_default().push(index);
    }
    let non_adjacent: Vec<(u64, usize)> = by_offset
        .iter()
        .filter(|(_, ix)| ix.len() > 1 && ix.windows(2).any(|w| w[1] != w[0] + 1))
        .map(|(off, ix)| (*off, ix.len()))
        .collect();
    assert!(
        non_adjacent.contains(&(148, 4)),
        "offset 148 should be shared by four non-adjacent entries, got {non_adjacent:?}"
    );

    // And the payload really is stored once: 63 blobs of 74 bytes for 85
    // addressed tiles.
    assert_eq!(mine.header.tile_data_length, g.header.tile_data_length);
    assert!(
        mine.header.tile_contents_count < mine.header.tile_entries_count,
        "dedupe found nothing, so this fixture is not testing dedupe"
    );
}

/// Archive-level dedupe is unconditional and keyed on the content hash, with
/// no `DedupeStrategy` anywhere near it.
///
/// `DedupeStrategy` defaults to `None`, and under `None` `DedupeIndex::record`
/// returns `WriteNew` for every call **by design**: it is a true passthrough
/// and must not collapse anything. A writer that drove its payload table off
/// those decisions would therefore store every duplicate on default settings,
/// and "duplicate tiles produce exactly one stored payload" would hold only
/// for callers who had turned dedupe on. The archive's payload table is a
/// property of the archive, not of the engine's blank-tile policy.
///
/// # "within the window" is doing real work in that name
///
/// This test used to be called
/// `identical_payloads_are_stored_once_whatever_the_engine_dedupe_strategy_is`
/// and it read as unconditional, because the writer kept a hash of every
/// payload it had ever seen and the promise really was unconditional. Issue
/// #1137 replaced that with a fixed-capacity window, so two identical payloads
/// far enough apart are now stored twice, and the name had to say so before
/// the behaviour changed under it. What is unconditional is the part this test
/// is actually about: dedupe happens with `DedupeStrategy::None` and without
/// the caller asking. How far it reaches is
/// [`both_edges_of_the_dedupe_window_are_where_the_window_says_they_are`].
#[test]
#[cfg_attr(miri, ignore)]
fn identical_payloads_within_the_window_are_stored_once_whatever_the_engine_dedupe_strategy_is() {
    let dir = scratch();
    let out = dir.path().join("dupes.pmtiles");
    let ocean = b"the same forty-two bytes, over and over ok".to_vec();
    let land = b"something else entirely".to_vec();

    let mut w = Writer::create(&out, WriterOptions::default().with_tile_type(TileType::Png))
        .expect("a writer opens");
    // Zoom 0 is id 0, zoom 1 is ids 1..=4, zoom 2 starts at id 5. Putting the
    // ocean blob at zoom 0 and at two zoom 2 tiles, with four zoom 1 tiles of
    // something else in between, produces both shapes at once: the zoom 2 pair
    // is a run, and the zoom 0 tile is a separate entry pointing at the same
    // offset with four entries' worth of ids between them.
    w.add_tile(0, 0, 0, &ocean, content_hash(&ocean)).unwrap();
    for (x, y) in [(0, 0), (0, 1), (1, 1), (1, 0)] {
        w.add_tile(1, x, y, &land, content_hash(&land)).unwrap();
    }
    w.add_tile(2, 0, 0, &ocean, content_hash(&ocean)).unwrap();
    w.add_tile(2, 1, 0, &ocean, content_hash(&ocean)).unwrap();
    let done = w.finish().unwrap();

    assert_eq!(
        done.header.tile_data_length,
        (ocean.len() + land.len()) as u64,
        "seven tiles over two distinct payloads stored more than two blobs"
    );
    assert_eq!(done.header.tile_contents_count, 2);
    assert_eq!(done.header.addressed_tiles_count, 7);
    // One entry for id 0, one for the zoom 1 run of four, one for the zoom 2
    // run of two. The runs are the first shape of dedupe; ids 0 and 5 sharing
    // offset 0 across a gap is the second.
    assert_eq!(done.header.tile_entries_count, 3);

    let bytes = std::fs::read(&out).unwrap();
    let mine = parse_ours(&bytes);
    assert_eq!(
        mine.entries,
        vec![
            Entry {
                tile_id: 0,
                offset: 0,
                length: ocean.len() as u32,
                run_length: 1
            },
            Entry {
                tile_id: 1,
                offset: ocean.len() as u64,
                length: land.len() as u32,
                run_length: 4
            },
            Entry {
                tile_id: 5,
                offset: 0,
                length: ocean.len() as u32,
                run_length: 2
            },
        ]
    );
    // Positive control on the gap: without it the three entries would have
    // collapsed into one run and this test would be about runs alone.
    assert_eq!(mine.entries[0].offset, mine.entries[2].offset);
    assert!(mine.entries[2].tile_id > mine.entries[0].tile_id + 1);
}

/// Both edges of the dedupe window, in one archive: a duplicate inside it
/// shares an offset and one beyond it gets its own.
///
/// This is the trade issue #1137 makes and it is the one thing about this
/// writer that got weaker rather than stronger. The content-hash table was
/// exact at any distance and cost 48 bytes for every distinct payload for the
/// whole run, which at ten million payloads was the largest allocation in the
/// process and the whole of the remaining unbounded growth. The window is a
/// fixed-capacity 8-way set-associative table with LRU inside the set, so a
/// hit is still exact and a miss stages a payload an identical one may already
/// have staged long ago.
///
/// The two cases that matter are untouched. A photograph has no duplicates to
/// miss. A blank or solid-colour tile recurs constantly, so it never leaves
/// the window. What pays is a pyramid with many distinct payloads repeating at
/// long range.
///
/// The budget here is zero, which buys the smallest window there is: one set
/// of [`DEDUPE_WINDOW_WAYS`] ways, which is a plain LRU of that many payloads
/// and needs no assumption about which set a hash lands in. So the eviction
/// below is arithmetic rather than a probability.
#[test]
#[cfg_attr(miri, ignore)]
fn both_edges_of_the_dedupe_window_are_where_the_window_says_they_are() {
    let dir = scratch();
    let out = dir.path().join("window.pmtiles");
    let ocean = b"the same forty-two bytes, over and over ok".to_vec();
    let filler = |n: usize| format!("filler number {n:04}, distinct from every other").into_bytes();

    let mut w = Writer::create(
        &out,
        WriterOptions::default()
            .with_tile_type(TileType::Png)
            .with_dedupe_memory_bytes(0),
    )
    .expect("a writer opens");

    // Ids ascend with arrival, so the layout the archive ends up in is the
    // order the window saw.
    let mut id = 21u64;
    let add = |w: &mut Writer<std::fs::File>, id: &mut u64, payload: &[u8]| {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*id).unwrap();
        w.add_tile(z, x, y, payload, content_hash(payload)).unwrap();
        let used = *id;
        *id += 1;
        used
    };

    let first = add(&mut w, &mut id, &ocean);
    // Two short of filling the window, so the ocean tile is still in it.
    for n in 0..DEDUPE_WINDOW_WAYS - 2 {
        add(&mut w, &mut id, &filler(n));
    }
    let inside = add(&mut w, &mut id, &ocean);
    // A full window's worth of distinct payloads after that. The reference
    // above made the ocean tile the most recent, so it takes exactly this many
    // to walk it back out of the set.
    for n in 0..DEDUPE_WINDOW_WAYS {
        add(&mut w, &mut id, &filler(100 + n));
    }
    let beyond = add(&mut w, &mut id, &ocean);

    let done = w.finish().unwrap();
    let bytes = std::fs::read(&out).unwrap();
    let mine = parse_ours(&bytes);
    let offset_of = |tile_id: u64| {
        mine.entries
            .iter()
            .find(|e| e.tile_id == tile_id)
            .unwrap_or_else(|| panic!("tile {tile_id} is missing from the archive"))
            .offset
    };

    // Inside the window: one blob, two entries pointing at it.
    assert_eq!(
        offset_of(inside),
        offset_of(first),
        "a duplicate inside the window should share the first one's blob"
    );
    // Beyond it: its own blob.
    assert_ne!(
        offset_of(beyond),
        offset_of(first),
        "a duplicate past the window should have been stored again"
    );

    // And the second copy really is the same bytes, so what changed is where
    // they are and not what they are.
    let at = |offset: u64| {
        section(
            &bytes,
            mine.header.tile_data_offset + offset,
            ocean.len() as u64,
        )
        .to_vec()
    };
    assert_eq!(at(offset_of(beyond)), ocean);
    assert_eq!(at(offset_of(first)), ocean);

    // Counted from the other side: one tile, one blob, except the one the
    // window caught.
    let tiles = 3 + 2 * DEDUPE_WINDOW_WAYS as u64 - 2;
    assert_eq!(done.header.addressed_tiles_count, tiles);
    assert_eq!(
        done.header.tile_contents_count,
        tiles - 1,
        "exactly one of the three ocean tiles should have been deduplicated"
    );
}

/// A run of identical adjacent tiles is one entry with `run_length = N`, and
/// the run stops where the payload changes.
#[test]
#[cfg_attr(miri, ignore)]
fn a_run_of_identical_adjacent_tiles_is_one_entry() {
    let dir = scratch();
    let out = dir.path().join("runs.pmtiles");
    let ocean = b"ocean".to_vec();
    let land = b"land!".to_vec();

    let mut w = Writer::create(&out, WriterOptions::default().with_tile_type(TileType::Png))
        .expect("a writer opens");
    // Zoom 3 ids run 21..=84. Give ids 21..=40 the same blob and 41 a
    // different one.
    for id in 21u64..=41 {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(id).unwrap();
        let payload = if id <= 40 { &ocean } else { &land };
        w.add_tile(z, x, y, payload, content_hash(payload)).unwrap();
    }
    let done = w.finish().unwrap();
    assert_eq!(done.header.addressed_tiles_count, 21);
    assert_eq!(done.header.tile_entries_count, 2);
    assert_eq!(done.header.tile_contents_count, 2);

    let bytes = std::fs::read(&out).unwrap();
    let mine = parse_ours(&bytes);
    assert_eq!(
        mine.entries[0].run_length, 20,
        "the run should cover 21..=40"
    );
    assert_eq!(mine.entries[0].tile_id, 21);
    assert_eq!(mine.entries[1].tile_id, 41);
    assert_eq!(mine.entries[1].run_length, 1);
    assert_ne!(
        mine.entries[0].offset, mine.entries[1].offset,
        "two different payloads must not share an offset"
    );
}

/// What the gzip level costs at the size the root budget is about.
///
/// Issue #1142 suggests dropping `Compression::compress` from
/// `flate2::Compression::best()` to the default level 6, on the evidence that
/// re-gzipping the golden fixtures' roots at both levels gives identical
/// lengths. Those roots hold 63, 85 and a few hundred entries, and four small
/// fixtures agreeing is not the same thing as the level being free. The only
/// place the level can change anything is a root near the 16257-byte budget,
/// because a directory that fits at one level and not at the other spills into
/// leaves, which changes the archive's shape, its header, every offset in it
/// and what a reader pays to open it.
///
/// # What it measured, and what I did about it
///
/// At a full 16383-entry root, flat tiles, leaf pointers and sparse ids all
/// come out byte-for-byte the same length at both levels, and a root of
/// dedupe back references comes out 20 bytes *smaller* at level 6, 71,476
/// against 71,496. At the budget, level 6 fits one more entry than level 9 on
/// leaf pointers (8740 against 8739) and one more on back references (3872
/// against 3871), and the same count on the other two.
///
/// So the claim holds in the direction it was made: level 6 never produced a
/// larger root here, and the fixtures were not lying, they were just small.
/// **The default stays at `best()` anyway**, for a reason the issue does not
/// weigh. This writer matches go-pmtiles deliberately, at the 4096-entry leaf
/// start and the 16384-entry root cutoff, and go-pmtiles compresses its
/// directories at `gzip.BestCompression`. Matching that is part of why a
/// golden archive is a usable target. Moving our level changes `root_length`
/// for some inputs, which shifts every section offset after it, in exchange
/// for some directory compression time that is a small share of a finalize.
/// That is a bad trade for the one property these tests are built on.
///
/// This is a measurement rather than a reproduction. It is here so the
/// decision has something behind it that can be taken again, and so that a
/// future change making the levels agree completely shows up as a failure
/// rather than as nothing.
#[test]
#[cfg_attr(miri, ignore)]
fn what_the_gzip_level_costs_at_the_root_budget() {
    // `ROOT_CEILING - HEADER_BYTES`, restated because it is private to the
    // writer and this test is about the number rather than about the writer.
    const ROOT_BUDGET: usize = 16384 - 127;

    /// A little deterministic noise, so the columns carry the entropy a real
    /// directory's do without dragging in a dependency.
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// Tiles in a flat root: consecutive ids, contiguous blobs of the size
    /// encoded imagery comes out at.
    fn flat(count: u64) -> Vec<Entry> {
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        let mut offset = 0u64;
        (0..count)
            .map(|index| {
                let length = 2048 + (next(&mut rng) % 16384) as u32;
                let entry = Entry {
                    tile_id: index,
                    offset,
                    length,
                    run_length: 1,
                };
                offset += u64::from(length);
                entry
            })
            .collect()
    }

    /// A root of leaf pointers: `run_length` zero, ids a leaf apart, lengths
    /// the size a compressed leaf comes out at.
    fn leaf_pointers(count: u64) -> Vec<Entry> {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut offset = 0u64;
        (0..count)
            .map(|index| {
                let length = 1000 + (next(&mut rng) % 6000) as u32;
                let entry = Entry {
                    tile_id: index * 4096,
                    offset,
                    length,
                    run_length: 0,
                };
                offset += u64::from(length);
                entry
            })
            .collect()
    }

    /// A root with back references, which is what dedupe produces and what
    /// puts real values in the offset column instead of the contiguous
    /// shorthand.
    fn back_references(count: u64) -> Vec<Entry> {
        let mut rng = 0xdead_beef_1234_5678u64;
        let mut offset = 0u64;
        let mut seen: Vec<(u64, u32)> = Vec::new();
        (0..count)
            .map(|index| {
                let reuse = !seen.is_empty() && next(&mut rng) % 5 < 2;
                let (at, length) = if reuse {
                    seen[(next(&mut rng) as usize) % seen.len()]
                } else {
                    let length = 2048 + (next(&mut rng) % 16384) as u32;
                    let at = offset;
                    offset += u64::from(length);
                    seen.push((at, length));
                    (at, length)
                };
                Entry {
                    tile_id: index,
                    offset: at,
                    length,
                    run_length: 1,
                }
            })
            .collect()
    }

    /// A sparse pyramid, where the tile-id column carries multi-byte deltas.
    fn with_gaps(count: u64) -> Vec<Entry> {
        let mut rng = 0x0123_4567_89ab_cdefu64;
        let mut offset = 0u64;
        let mut tile_id = 0u64;
        (0..count)
            .map(|_| {
                tile_id += 1 + next(&mut rng) % (1 << 20);
                let length = 2048 + (next(&mut rng) % 16384) as u32;
                let entry = Entry {
                    tile_id,
                    offset,
                    length,
                    run_length: 1,
                };
                offset += u64::from(length);
                entry
            })
            .collect()
    }

    let gzip = |bytes: &[u8], level: u32| -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(bytes).expect("gzip takes the bytes");
        encoder.finish().expect("gzip finishes")
    };

    // The largest root of this shape that fits the budget at this level.
    let fits = |build: &dyn Fn(u64) -> Vec<Entry>, level: u32| -> u64 {
        let mut count = 0u64;
        let mut step = 8192u64;
        while step > 0 {
            let plain = serialize_entries(&build(count + step)).expect("entries serialise");
            if gzip(&plain, level).len() <= ROOT_BUDGET {
                count += step;
            } else {
                step /= 2;
            }
        }
        count
    };

    /// A named way of building a root of `n` entries.
    type Shape<'a> = (&'a str, &'a dyn Fn(u64) -> Vec<Entry>);

    let shapes: [Shape; 4] = [
        ("flat tiles", &flat),
        ("leaf pointers", &leaf_pointers),
        ("back references", &back_references),
        ("sparse ids", &with_gaps),
    ];

    // The full root first: one under the 16384-entry cutoff this writer shares
    // with go-pmtiles, which is the largest root it will ever try to build.
    let mut lengths_differ = false;
    for (name, build) in &shapes {
        let plain = serialize_entries(&build(16_383)).expect("16383 entries serialise");
        let best = gzip(&plain, 9).len();
        let default = gzip(&plain, 6).len();
        lengths_differ |= best != default;
        println!(
            "16383-entry root, {name}: {} raw, {best} at level 9, {default} at level 6",
            plain.len()
        );
    }

    let mut budget_differs = false;
    for (name, build) in &shapes {
        let at_best = fits(*build, 9);
        let at_default = fits(*build, 6);
        budget_differs |= at_best != at_default;
        println!(
            "entries fitting the root budget, {name}: {at_best} at level 9, {at_default} at level 6"
        );
    }

    assert!(
        lengths_differ || budget_differs,
        "level 6 and level 9 agreed on every shape and every size measured here, so the level \
         really would be free and the only thing keeping `Compression::compress` on best() \
         would be matching go-pmtiles"
    );
}

// ---------------------------------------------------------------------------
// Determinism, clustering, and the header's honesty
// ---------------------------------------------------------------------------

/// Two shuffled insertion orders produce a byte-identical archive.
///
/// This is the criterion that forced the layout decision. Issue #989 asks for
/// arrival-order data *and* for byte-identical output from two arrival orders,
/// and those cannot both hold. The tiebreaker is structural: the root has to
/// fit in the first 16384 bytes, so its size is unknown until the entries are
/// sorted, so the data region is copied at finalize whichever choice is made.
/// Given the copy happens anyway, doing it in tile id order costs one pass and
/// buys determinism, an honest `clustered = true`, and read locality.
#[test]
#[cfg_attr(miri, ignore)]
fn two_shuffled_insertion_orders_produce_a_byte_identical_archive() {
    let dir = scratch();
    let g = parse_golden("dupes-z0z3.pmtiles");
    let (_, a) = rewrite(&g, dir.path(), 0x5eed_1111);
    let (_, b) = rewrite(&g, dir.path(), 0x7eed_2222);
    assert_eq!(
        a.len(),
        b.len(),
        "two orders produced different sized archives"
    );
    assert_eq!(a, b, "the archive is not a pure function of its tile set");

    // Positive control: the two orders really were different, otherwise this
    // test compares one order with itself.
    let mut one: Vec<(u64, Vec<u8>)> = g.tiles.clone();
    let mut two = one.clone();
    shuffled(&mut one, 0x5eed_1111);
    shuffled(&mut two, 0x7eed_2222);
    let ids_one: Vec<u64> = one.iter().map(|(id, _)| *id).collect();
    let ids_two: Vec<u64> = two.iter().map(|(id, _)| *id).collect();
    assert_ne!(ids_one, ids_two, "the two shuffles produced the same order");
}

/// And it stops being byte-identical the moment the window cannot hold the
/// tile set.
///
/// The test above runs `dupes-z0z3` at the default window, which tracks
/// 129,056 payloads against a fixture holding 63. Three orders of magnitude of
/// slack means it is green now and stays green whatever happens to the
/// property, which makes it a poor guard for a property that just became
/// conditional. This is the other half: whether two identical payloads share
/// one blob is a function of how far apart they **arrived**, so at a window
/// too small to hold the set, two orders of the same tiles produce different
/// archives.
///
/// That is the contract issue #1137 changed, stated as an executable fact
/// rather than as a caveat in prose. A caller who needs the archive to be a
/// pure function of its tile set sizes
/// `WriterOptions::dedupe_memory_bytes` past the payload count, and the test
/// above is what that buys them.
#[test]
#[cfg_attr(miri, ignore)]
fn two_orders_stop_agreeing_once_the_window_cannot_hold_the_tile_set() {
    let dir = scratch();
    let ocean = b"the same forty-two bytes, over and over ok".to_vec();
    let filler = |n: usize| format!("filler number {n:04}, distinct from every other").into_bytes();

    // Three tiles carrying one payload, far enough apart in ascending id order
    // that a window of one set cannot keep it, and close enough together in
    // the other order that it never leaves.
    let mut tiles: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut id = 21u64;
    let push = |tiles: &mut Vec<(u64, Vec<u8>)>, id: &mut u64, bytes: Vec<u8>| {
        tiles.push((*id, bytes));
        *id += 1;
    };
    push(&mut tiles, &mut id, ocean.clone());
    for n in 0..DEDUPE_WINDOW_WAYS - 2 {
        push(&mut tiles, &mut id, filler(n));
    }
    push(&mut tiles, &mut id, ocean.clone());
    for n in 0..DEDUPE_WINDOW_WAYS {
        push(&mut tiles, &mut id, filler(100 + n));
    }
    push(&mut tiles, &mut id, ocean.clone());

    let write = |order: &[(u64, Vec<u8>)], name: &str| -> (Vec<u8>, Header) {
        let out = dir.path().join(name);
        let mut w = Writer::create(
            &out,
            WriterOptions::default()
                .with_tile_type(TileType::Png)
                .with_dedupe_memory_bytes(0),
        )
        .expect("a writer opens");
        for (tile_id, payload) in order {
            let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
            w.add_tile(z, x, y, payload, content_hash(payload)).unwrap();
        }
        let done = w.finish().expect("the archive finalises");
        (
            std::fs::read(&out).expect("the archive is on disk"),
            done.header,
        )
    };

    // Ascending, which is the order that walks the shared payload out of the
    // window between its second and third tile.
    let (ascending, ascending_header) = write(&tiles, "ascending.pmtiles");

    // The same tiles, with the three that share a payload fed together, so the
    // window never loses it.
    let mut grouped: Vec<(u64, Vec<u8>)> = tiles
        .iter()
        .filter(|(_, bytes)| *bytes == ocean)
        .cloned()
        .collect();
    grouped.extend(tiles.iter().filter(|(_, bytes)| *bytes != ocean).cloned());
    let (regrouped, regrouped_header) = write(&grouped, "regrouped.pmtiles");

    // The positive control on the fixture: both runs really did take the same
    // tiles, so what follows is about the window and not about the input.
    let mut one: Vec<u64> = tiles.iter().map(|(id, _)| *id).collect();
    let mut two: Vec<u64> = grouped.iter().map(|(id, _)| *id).collect();
    assert_ne!(one, two, "the two orders are the same order");
    one.sort_unstable();
    two.sort_unstable();
    assert_eq!(one, two, "the two orders are not the same tile set");
    assert_eq!(
        ascending_header.addressed_tiles_count,
        regrouped_header.addressed_tiles_count
    );

    // Three ocean tiles: ascending catches one duplicate, regrouped catches
    // both.
    assert_eq!(
        ascending_header.tile_contents_count,
        regrouped_header.tile_contents_count + 1,
        "the two orders should disagree by exactly the duplicate the window lost"
    );
    assert_ne!(
        ascending, regrouped,
        "the archive is still a pure function of its tile set at a window that cannot hold it, \
         so the contract issue #1137 changed did not change"
    );
}

/// `clustered` is set honestly, and the two things it promises hold.
///
/// The spec's definition is operational: offsets are contiguous with the
/// previous `offset + length` or refer to a lesser offset (which is what
/// dedupe produces), and the first tile entry has offset 0. A writer that
/// claimed clustering without laying the data out that way tells a reader it
/// may skip work it cannot skip.
#[test]
#[cfg_attr(miri, ignore)]
fn clustered_is_true_and_the_layout_backs_it_up() {
    let dir = scratch();
    let g = parse_golden("dupes-z0z3.pmtiles");
    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0006);
    let mine = parse_ours(&ours);
    assert!(
        mine.header.clustered,
        "we lay the data out in tile id order"
    );

    assert_eq!(
        mine.entries[0].offset, 0,
        "the first tile entry starts the section"
    );
    let mut next_byte = 0u64;
    for entry in &mine.entries {
        // Spelled as two named conditions rather than as `<=`, because the
        // spec's definition of clustered is two separate permissions and
        // collapsing them loses which one each entry is using.
        let contiguous_with_the_previous_blob = entry.offset == next_byte;
        let back_reference_to_a_deduplicated_one = entry.offset < next_byte;
        assert!(
            contiguous_with_the_previous_blob || back_reference_to_a_deduplicated_one,
            "entry at {} is neither contiguous nor a back reference",
            entry.tile_id
        );
        next_byte = next_byte.max(entry.offset + u64::from(entry.length));
    }
    assert_eq!(
        next_byte, mine.header.tile_data_length,
        "the blobs do not tile the data section exactly"
    );
}

/// The header's three counts hold their stated relationship and are filled in
/// rather than left at the "unknown" sentinel.
#[test]
#[cfg_attr(miri, ignore)]
fn the_three_counts_are_filled_in_and_consistent() {
    let dir = scratch();
    let g = parse_golden("dupes-z0z3.pmtiles");
    let (_, ours) = rewrite(&g, dir.path(), 0x5eed_0007);
    let h = parse_ours(&ours).header;
    assert!(h.tile_contents_count > 0 && h.tile_entries_count > 0 && h.addressed_tiles_count > 0);
    assert!(h.tile_contents_count <= h.tile_entries_count);
    assert!(h.tile_entries_count <= h.addressed_tiles_count);
    assert_eq!(h.min_zoom, 0);
    assert_eq!(h.max_zoom, 3);
    assert_eq!(h.internal_compression, Compression::Gzip);
    assert_eq!(h.tile_type, TileType::Png);
}

/// The metadata section is a JSON object carrying the `vnd.libviprs`
/// namespace, and it round-trips.
#[test]
#[cfg_attr(miri, ignore)]
fn the_metadata_section_carries_the_libviprs_namespace() {
    let dir = scratch();
    let out = dir.path().join("meta.pmtiles");
    let mut meta = Metadata::default();
    meta.name = Some("a drawing".to_string());
    meta.vnd_libviprs = Some(libviprs::pmtiles::LibviprsMetadata::default());

    let mut w = Writer::create(
        &out,
        WriterOptions::default()
            .with_tile_type(TileType::Png)
            .with_metadata(meta),
    )
    .unwrap();
    w.add_tile(0, 0, 0, b"tile", content_hash(b"tile")).unwrap();
    let done = w.finish().unwrap();

    let bytes = std::fs::read(&out).unwrap();
    let raw = gunzip(section(
        &bytes,
        done.header.metadata_offset,
        done.header.metadata_length,
    ));
    let back = Metadata::try_from_json(&raw).expect("our metadata parses");
    assert_eq!(back.name.as_deref(), Some("a drawing"));
    assert_eq!(
        back.vnd_libviprs
            .as_ref()
            .map(|m| m.coordinate_convention.as_str()),
        Some("zxy")
    );
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// An archive with no tiles is refused rather than written.
///
/// Every directory `MUST` have more than zero entries and a conformant archive
/// has a root directory, so there is no legal encoding of an empty archive.
/// Writing one anyway would produce a file no conformant reader can open.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_with_no_tiles_is_refused() {
    let dir = scratch();
    let out = dir.path().join("empty.pmtiles");
    let w = Writer::create(&out, WriterOptions::default()).unwrap();
    let err = w
        .finish()
        .expect_err("an empty archive is not representable");
    assert!(matches!(err, PmTilesError::EmptyDirectory), "got {err:?}");
    assert!(!out.exists(), "a refused finalize must not publish a file");
}

/// The same tile twice is a refusal, not a silent last-writer-wins.
#[test]
#[cfg_attr(miri, ignore)]
fn adding_one_tile_twice_is_refused() {
    let dir = scratch();
    let out = dir.path().join("dupe-id.pmtiles");
    let mut w = Writer::create(&out, WriterOptions::default()).unwrap();
    w.add_tile(1, 0, 1, b"a", content_hash(b"a")).unwrap();
    w.add_tile(1, 0, 1, b"b", content_hash(b"b")).unwrap();
    let err = w
        .finish()
        .expect_err("two entries cannot claim one tile id");
    assert!(
        matches!(err, PmTilesError::DuplicateTile { tile_id: 2 }),
        "got {err:?}"
    );
    assert!(!out.exists());
}

/// A coordinate outside its own zoom's grid is refused rather than masked.
#[test]
#[cfg_attr(miri, ignore)]
fn an_out_of_range_coordinate_is_refused_not_wrapped() {
    let dir = scratch();
    let out = dir.path().join("oob.pmtiles");
    let mut w = Writer::create(&out, WriterOptions::default()).unwrap();
    let err = w
        .add_tile(2, 4, 0, b"x", content_hash(b"x"))
        .expect_err("(2, 4, 0) is not a tile");
    assert!(
        matches!(err, PmTilesError::CoordOutOfRange { .. }),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Large offsets
// ---------------------------------------------------------------------------

/// Offsets past 4 GiB survive the arithmetic that carries them.
///
/// **This is arithmetic only.** It builds a synthetic entry set whose offsets
/// and lengths sit above `u32::MAX` and runs it through the real directory
/// serializer and the real header encoder; it does not write four gigabytes,
/// so it proves the `u64` maths and says nothing about the I/O path at that
/// size.
#[test]
fn offsets_past_four_gibibytes_survive_the_arithmetic() {
    const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
    let entries = vec![
        Entry {
            tile_id: 0,
            offset: FOUR_GIB - 1,
            length: 3,
            run_length: 1,
        },
        // Contiguous with the previous entry, so this one takes the shorthand
        // branch on an offset that does not fit in 32 bits.
        Entry {
            tile_id: 1,
            offset: FOUR_GIB + 2,
            length: 7,
            run_length: 2,
        },
        Entry {
            tile_id: 5,
            offset: FOUR_GIB * 3,
            length: u32::MAX,
            run_length: 1,
        },
    ];
    let bytes = serialize_entries(&entries).expect("large offsets serialise");
    assert_eq!(
        deserialize_entries(&bytes).expect("large offsets decode"),
        entries
    );

    let header = Header {
        root_offset: 127,
        root_length: bytes.len() as u64,
        tile_data_offset: FOUR_GIB,
        tile_data_length: FOUR_GIB * 5,
        addressed_tiles_count: 4,
        tile_entries_count: 3,
        tile_contents_count: 3,
        clustered: true,
        tile_type: TileType::Png,
        ..Header::default()
    };
    let encoded = header.encode();
    let back = Header::try_decode(&encoded).expect("a large header decodes");
    assert_eq!(back.tile_data_length, FOUR_GIB * 5);
    assert_eq!(back, header);
}

// ---------------------------------------------------------------------------
// Interrupted finalize
// ---------------------------------------------------------------------------

/// The environment variable that turns this test binary into the child half of
/// [`killing_the_process_leaves_no_final_archive`].
const KILL_CHILD_DIR: &str = "LIBVIPRS_PMTILES_KILL_CHILD_DIR";

/// Killing the process after N tiles leaves staging behind and **no** final
/// archive, and the same writer run to completion does produce one.
///
/// This is the easiest vacuous test in the issue. "Kill before anything is
/// written, assert the final path is absent" passes for a writer that never
/// writes anything at all, so the positive control is the whole point: the
/// child is killed **after** eight tiles have gone in, the staging file it was
/// appending to has to still be on disk, and a second child that finishes
/// normally has to produce the final path in the same directory. Without those
/// two the absence proves nothing.
///
/// The kill is a real `SIGABRT` from a real child process rather than a
/// simulated one, because what is being tested is that nothing between
/// `add_tile` and the rename ever makes `<path>` appear.
#[test]
#[cfg_attr(miri, ignore)]
fn killing_the_process_leaves_no_final_archive() {
    if let Ok(dir) = std::env::var(KILL_CHILD_DIR) {
        // Child half.
        let out = Path::new(&dir).join("killed.pmtiles");
        let mut w = Writer::create(&out, WriterOptions::default().with_tile_type(TileType::Png))
            .expect("the child opens a writer");
        for id in 21u64..29 {
            let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(id).unwrap();
            let payload = format!("tile {id}").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .expect("the child adds a tile");
        }
        // Die with the writer live, its staging file open and its destructor
        // unrun. `abort` rather than `exit` so nothing gets a chance to clean
        // up on the way out.
        std::process::abort();
    }

    let dir = scratch();
    let status = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            "killing_the_process_leaves_no_final_archive",
            "--nocapture",
        ])
        .env(KILL_CHILD_DIR, dir.path())
        .status()
        .expect("the child runs");
    assert!(
        !status.success(),
        "the child was supposed to abort, not to pass"
    );

    let out = dir.path().join("killed.pmtiles");
    assert!(!out.exists(), "a killed run published {}", out.display());

    // Positive control. Without this the assertion above is satisfied by a
    // child that crashed before it got anywhere, and by a writer that never
    // touches the disk at all.
    let staged: Vec<PathBuf> = std::fs::read_dir(dir.path())
        .expect("the scratch directory is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    assert!(
        !staged.is_empty(),
        "the child died after eight tiles and left nothing staged, so it cannot have been writing"
    );
    assert!(
        staged.iter().any(|p| p
            .file_name()
            .map(|n| n.to_string_lossy().starts_with("killed.pmtiles.tmp"))
            == Some(true)),
        "expected a killed.pmtiles.tmp* staging file, found {staged:?}"
    );

    // Second control: the same code, run to completion in the same directory,
    // does publish. Otherwise "no final archive" is what this writer always
    // does.
    let ok = dir.path().join("survivor.pmtiles");
    let mut w =
        Writer::create(&ok, WriterOptions::default().with_tile_type(TileType::Png)).unwrap();
    w.add_tile(0, 0, 0, b"tile", content_hash(b"tile")).unwrap();
    w.finish().unwrap();
    assert!(ok.exists(), "a completed run must publish");
}

/// Dropping a writer without finishing leaves no archive and no staging
/// litter.
#[test]
#[cfg_attr(miri, ignore)]
fn dropping_a_writer_without_finishing_cleans_up_after_itself() {
    let dir = scratch();
    let out = dir.path().join("abandoned.pmtiles");
    {
        let mut w =
            Writer::create(&out, WriterOptions::default().with_tile_type(TileType::Png)).unwrap();
        w.add_tile(0, 0, 0, b"tile", content_hash(b"tile")).unwrap();
        // Positive control: staging really is on disk while the writer lives,
        // so the emptiness asserted below is a cleanup and not a no-op.
        let during: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        assert!(
            during
                .iter()
                .any(|n| n.starts_with("abandoned.pmtiles.tmp")),
            "nothing was staged while the writer was live: {during:?}"
        );
        assert!(!out.exists(), "the final path appeared before finish()");
    }
    let after: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    assert!(after.is_empty(), "a dropped writer left {after:?} behind");
}

/// A finalize that cannot publish reports the failure and leaves nothing
/// half-written where the archive was supposed to go.
#[test]
#[cfg_attr(miri, ignore)]
fn a_finalize_that_cannot_publish_leaves_no_partial_archive() {
    let dir = scratch();
    let out = dir.path().join("blocked.pmtiles");
    // A directory sitting on the destination name makes the rename fail after
    // every byte of the archive has already been assembled, which is the
    // window the atomicity is for.
    std::fs::create_dir(&out).unwrap();

    let mut w =
        Writer::create(&out, WriterOptions::default().with_tile_type(TileType::Png)).unwrap();
    w.add_tile(0, 0, 0, b"tile", content_hash(b"tile")).unwrap();
    let err = w.finish().expect_err("publishing over a directory fails");
    assert!(matches!(err, PmTilesError::Io(_)), "got {err:?}");

    assert!(out.is_dir(), "the destination should be untouched");
    let left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n != "blocked.pmtiles")
        .collect();
    assert!(left.is_empty(), "a failed finalize left {left:?} behind");
}

// ---------------------------------------------------------------------------
// Bounded memory
// ---------------------------------------------------------------------------

/// An archive built through a real external merge is byte-identical to one
/// built with everything sorted in memory.
///
/// The sort buffer is an option rather than a constant precisely so this test
/// can exist: the default is a million records and writing a million tiles to
/// reach the merge would make the suite unusable, so this writes 3000 with a
/// 128-record buffer and gets 24 runs out of it.
///
/// The positive control is `spilled_run_count`. Without it the test passes
/// unchanged against a writer that never spills at all, which is exactly the
/// path it is supposed to be avoiding.
#[test]
#[cfg_attr(miri, ignore)]
fn an_external_merge_produces_the_same_archive_as_an_in_memory_sort() {
    let dir = scratch();

    // Zoom 6 is ids 1365..=5460. Take 3000 of them and SHUFFLE them.
    //
    // Reversing them was the obvious thing and it was wrong, which a mutation
    // caught: replacing the run sort with `reverse()` is the identity on a
    // perfectly reverse-ordered input, so the test stayed green against a
    // writer that does not sort at all. A shuffle has no such symmetry.
    let mut ids: Vec<u64> = (1365u64..1365 + 3000).collect();
    shuffled(&mut ids, 0x5eed_0009);

    let build = |name: &str, buffer: usize| -> (Vec<u8>, usize) {
        let out = dir.path().join(name);
        let mut w = Writer::create(
            &out,
            WriterOptions::default()
                .with_tile_type(TileType::Png)
                .with_sort_buffer_records(buffer),
        )
        .unwrap();
        for id in &ids {
            let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*id).unwrap();
            // Four distinct payloads, so dedupe and the RLE both have work to
            // do and the merge is not sorting a set of identical records.
            let payload = format!("payload {}", id % 4).into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }
        let runs = w.spilled_run_count();
        w.finish().unwrap();
        (std::fs::read(&out).unwrap(), runs)
    };

    let (merged, merged_runs) = build("merged.pmtiles", 128);
    let (in_memory, in_memory_runs) = build("in-memory.pmtiles", usize::MAX);

    assert!(
        merged_runs >= 23,
        "3000 records at 128 to a run should spill 23 runs, got {merged_runs}"
    );
    assert_eq!(
        in_memory_runs, 0,
        "the in-memory build should not have spilled a run before finish"
    );
    assert_eq!(
        merged, in_memory,
        "the external merge produced a different archive than the in-memory sort"
    );

    // And the archive is actually right, not merely consistent with itself.
    let mine = parse_ours(&merged);
    assert_eq!(mine.header.addressed_tiles_count, 3000);
    assert_eq!(mine.header.tile_contents_count, 4);
    assert!(
        mine.entries.windows(2).all(|w| w[0].tile_id < w[1].tile_id),
        "the merge did not produce ascending tile ids"
    );
    assert_eq!(mine.tiles.len(), 3000);
}

// ---------------------------------------------------------------------------
// The generic sink
// ---------------------------------------------------------------------------

/// The writer works over any `Write + Seek`, not only a file, and the archive
/// it produces there is the same one it would have published.
#[test]
#[cfg_attr(miri, ignore)]
fn the_writer_works_over_any_write_and_seek_sink() {
    let dir = scratch();
    let g = parse_golden("raster-z0z2.pmtiles");
    let (_, published) = rewrite(&g, dir.path(), 0x5eed_0008);

    let mut sink = std::io::Cursor::new(Vec::new());
    {
        let scratch_dir = scratch();
        let mut w = Writer::try_new(
            &mut sink,
            scratch_dir.path(),
            WriterOptions::default()
                .with_tile_type(g.header.tile_type)
                .with_tile_compression(g.header.tile_compression),
        )
        .unwrap();
        let mut tiles = g.tiles.clone();
        shuffled(&mut tiles, 0x1234_5678);
        for (tile_id, payload) in &tiles {
            let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
            w.add_tile(z, x, y, payload, content_hash(payload)).unwrap();
        }
        let done = w.finish().unwrap();
        assert_eq!(done.path, None, "an in-memory sink has nothing to publish");
    }
    assert_eq!(
        sink.into_inner(),
        published,
        "the same tiles through a Cursor produced a different archive"
    );
}

// ---------------------------------------------------------------------------
// The header we write, not the one we meant
// ---------------------------------------------------------------------------

/// The bounds the writer is asked for are the bounds the archive carries, in
/// the fields they belong in.
///
/// # The hole this closes
///
/// The decoder's positions are pinned twice over:
/// `pmtiles_reader::the_header_decodes_to_what_go_pmtiles_reports_for_every_golden`
/// compares all six `i32`s against what `DeserializeHeader` returned, and
/// `pmtiles_format::every_golden_header_decodes_and_re_encodes_byte_for_byte`
/// catches a read that disagrees with the write. Nothing read back the
/// positions of a header **we** produced. Measured: exchange latitude and
/// longitude inside `Header::set_bounds_degrees` and all 128 tests in this
/// crate's PMTiles suite pass, while every archive libviprs writes carries
/// bounds that are nonsense and every consumer renders the wrong extent.
///
/// Two halves, and they catch different mistakes. The values have to come back
/// as asked, which catches a field written to the wrong offset or at the wrong
/// scale. And they have to be **in range for what they are**, which needs no
/// expectation at all: longitude runs to 180 and latitude stops at 90, so a
/// longitude sitting in a latitude field is out of range on its face. The
/// second half is what survives somebody changing the numbers below.
///
/// The numbers are deliberately not symmetric and not each other's negation.
/// A fixture sitting on the identity element of the operation under test is
/// the mistake this whole area has already made once.
#[test]
#[cfg_attr(miri, ignore)]
fn the_bounds_the_writer_is_given_are_the_bounds_the_archive_carries() {
    const WEST: f64 = -12.25;
    const SOUTH: f64 = 4.5;
    const EAST: f64 = 33.75;
    const NORTH: f64 = 51.125;
    const CENTRE: (f64, f64) = (7.5, 22.25);

    // The control for the control. If any two of these were equal, or one were
    // another's negation, an exchange or a sign flip would be the identity.
    let corners = [WEST, SOUTH, EAST, NORTH, CENTRE.0, CENTRE.1];
    for (i, a) in corners.iter().enumerate() {
        for b in corners.iter().skip(i + 1) {
            assert_ne!(a, b, "two of the bounds are the same number");
            assert_ne!(*a, -*b, "two of the bounds are each other's negation");
        }
    }

    let dir = scratch();
    let out = dir.path().join("bounded.pmtiles");
    let tile = b"a payload, of some length or other".to_vec();

    let mut w = Writer::create(
        &out,
        WriterOptions::default()
            .with_tile_type(TileType::Png)
            .with_bounds_degrees([WEST, SOUTH, EAST, NORTH])
            .with_center_degrees(CENTRE.0, CENTRE.1),
    )
    .expect("a writer opens");
    for (z, x, y) in [
        (0u8, 0u32, 0u32),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
    ] {
        w.add_tile(z, x, y, &tile, content_hash(&tile)).unwrap();
    }
    w.finish().expect("the writer finishes");

    let bytes = std::fs::read(&out).expect("read the archive back");
    let header = Header::try_decode(&bytes[..127]).expect("decode the header we wrote");
    let (west, south, east, north) = header.bounds_degrees();
    let (lon, lat) = header.center_degrees();

    for (label, got, want) in [
        ("west", west, WEST),
        ("south", south, SOUTH),
        ("east", east, EAST),
        ("north", north, NORTH),
        ("centre longitude", lon, CENTRE.0),
        ("centre latitude", lat, CENTRE.1),
    ] {
        assert!(
            (got - want).abs() < 1e-6,
            "the writer was asked for a {label} of {want} and the archive \
             carries {got}"
        );
    }

    for (label, value) in [("west", west), ("east", east), ("centre longitude", lon)] {
        assert!(
            value.is_finite() && (-180.0..=180.0).contains(&value),
            "the {label} in the archive is {value}, which is not a longitude. \
             A latitude and a longitude exchanged on the way out reads exactly \
             like this."
        );
    }
    for (label, value) in [("south", south), ("north", north), ("centre latitude", lat)] {
        assert!(
            value.is_finite() && (-90.0..=90.0).contains(&value),
            "the {label} in the archive is {value}, which is not a latitude. \
             A latitude and a longitude exchanged on the way out reads exactly \
             like this."
        );
    }

    assert!(west <= east, "the bounds run west {west} to east {east}");
    assert!(
        south <= north,
        "the bounds run south {south} to north {north}"
    );
    assert!(
        (west..=east).contains(&lon) && (south..=north).contains(&lat),
        "the centre ({lon}, {lat}) is outside the bounds"
    );
    assert!(
        header.min_zoom <= header.max_zoom
            && (header.min_zoom..=header.max_zoom).contains(&header.center_zoom),
        "the zoom range is {}..={} with a centre zoom of {}",
        header.min_zoom,
        header.max_zoom,
        header.center_zoom
    );
}

// ---------------------------------------------------------------------------
// Layout::Arrival (issue #1143)
// ---------------------------------------------------------------------------

/// The options an arrival-order rewrite of `g` uses.
fn arrival_options(g: &Golden) -> WriterOptions {
    WriterOptions::default()
        .with_tile_type(g.header.tile_type)
        .with_tile_compression(g.header.tile_compression)
        .with_layout(Layout::Arrival)
}

/// Feed a golden's own tiles into a writer at `out`, shuffled by `seed`, and
/// hand back the order they went in.
///
/// The order is the return value because under [`Layout::Arrival`] it is what
/// decides the data region, so a test that cannot see it cannot check the
/// layout it is asking for.
fn write_shuffled(
    g: &Golden,
    out: &Path,
    seed: u64,
    options: WriterOptions,
) -> Vec<(u64, Vec<u8>)> {
    let mut tiles: Vec<(u64, Vec<u8>)> = g.tiles.clone();
    shuffled(&mut tiles, seed);
    let mut writer = Writer::create(out, options).expect("a writer opens");
    for (tile_id, payload) in &tiles {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
        writer
            .add_tile(z, x, y, payload, content_hash(payload))
            .expect("a tile is accepted");
    }
    writer.finish().expect("the archive finalises");
    tiles
}

/// Where each distinct payload lands in the data region if payloads are placed
/// the first time they **arrive**, and the total that makes.
///
/// Keyed on the payload bytes, which is what the writer's window is keyed on
/// once the window is large enough to hold the whole tile set. Every fixture
/// here is, by three orders of magnitude, so the two agree.
fn arrival_offsets(order: &[(u64, Vec<u8>)]) -> (BTreeMap<u64, u64>, u64) {
    let mut placed: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    let mut per_tile: BTreeMap<u64, u64> = BTreeMap::new();
    let mut next = 0u64;
    for (tile_id, payload) in order {
        let offset = *placed.entry(payload.clone()).or_insert_with(|| {
            let at = next;
            next += payload.len() as u64;
            at
        });
        per_tile.insert(*tile_id, offset);
    }
    (per_tile, next)
}

/// The same thing in tile id order, which is what the default layout produces.
///
/// Only ever used as a fixed-point check. A shuffle that happened to agree
/// with tile id order would make the arrival assertions pass against either
/// layout, and a test that cannot tell the two apart is not testing the one it
/// names.
fn tile_id_offsets(order: &[(u64, Vec<u8>)]) -> BTreeMap<u64, u64> {
    let mut sorted: Vec<(u64, Vec<u8>)> = order.to_vec();
    sorted.sort_by_key(|(id, _)| *id);
    arrival_offsets(&sorted).0
}

/// An arrival-order archive puts the tile data straight after the reserved
/// prefix, and the metadata and the leaves after the tile data.
///
/// This is the layout the whole issue is about. v3 fixes the header's position
/// and requires the root inside the first 16384 bytes; it fixes nothing else,
/// so the sections in front of the tile data can be a reservation rather than
/// a thing that has to be written first.
///
/// Both fixtures matter. `dupes-z0z3` is 85 tiles and fits the root, and
/// `leaves-z0z7` is 21845 and does not, so it has a real leaf section: the
/// section that moves furthest here, from in front of the tile data to behind
/// it. A root-only fixture on its own would never place one.
#[test]
#[cfg_attr(miri, ignore)]
fn an_arrival_archive_puts_the_tile_data_before_the_metadata_and_the_leaves() {
    let dir = scratch();

    for (name, expect_leaves) in [("dupes-z0z3.pmtiles", false), ("leaves-z0z7.pmtiles", true)] {
        let label = name.trim_end_matches(".pmtiles");
        let g = parse_golden(name);
        let out = dir.path().join(format!("arrival-{label}.pmtiles"));
        write_shuffled(&g, &out, 0x5eed_1143, arrival_options(&g));
        let bytes = std::fs::read(&out).expect("the archive is on disk");
        let h = Header::try_decode(&bytes[..127]).expect("our header decodes");

        assert_eq!(h.root_offset, 127, "{label}: the root follows the header");
        assert!(
            h.root_offset + h.root_length <= 16384,
            "{label}: the root runs to {}, past the 16384 the spec allows it",
            h.root_offset + h.root_length
        );
        assert_eq!(
            h.tile_data_offset, 16384,
            "{label}: the tile data starts at {} rather than at the reserved \
             prefix's end, so the payloads were placed after the directories \
             and had to be copied there",
            h.tile_data_offset
        );
        assert_eq!(
            h.metadata_offset,
            h.tile_data_offset + h.tile_data_length,
            "{label}: the metadata is not immediately after the tile data"
        );
        assert_eq!(
            h.leaf_directories_offset,
            h.metadata_offset + h.metadata_length,
            "{label}: the leaf section is not immediately after the metadata"
        );
        assert_eq!(
            h.leaf_directories_offset + h.leaf_directories_length,
            bytes.len() as u64,
            "{label}: the archive does not end where its last section does"
        );

        // go-pmtiles' own arithmetic, spelled out rather than implied by the
        // three assertions above, because it is the constraint a future change
        // here is most likely to break without noticing. `Verify` accepts a
        // file whose size is either `127 + root + meta + leaf + tiles` or
        // `16384 + meta + leaf + tiles`, and nothing else
        // (`pmtiles/verify.go:84`, v1.31.2). The second is this layout and it
        // is the only padded size the reference knows about, so reserving a
        // different prefix, or aligning the tile data, or leaving a gap
        // anywhere, is rejected by the reference and accepted by ours.
        assert_eq!(
            bytes.len() as u64,
            16384 + h.metadata_length + h.leaf_directories_length + h.tile_data_length,
            "{label}: the archive is not the size go-pmtiles computes for a \
             padded one"
        );
        // This seed shuffles, and #1144 turned the flag into a measurement
        // rather than a constant, so the `false` below has to be earned now.
        // The walk over the archive's own entries is what says it was: before
        // #1144 this assertion held for every arrival archive ever written and
        // could not have caught anything.
        let walk = clustering_of(&parse_ours(&bytes).entries);
        assert!(
            walk.is_err(),
            "{label}: this seed did not break the tile id ordering, so the \
             assertion below cannot tell the two verdicts apart: {walk:?}"
        );
        assert!(
            !h.clustered,
            "{label}: the archive claims clustering its own entries deny"
        );

        assert_eq!(
            h.leaf_directories_length > 0,
            expect_leaves,
            "{label}: the fixture did not produce the leaf structure it was picked for"
        );
    }
}

/// Every tile in an arrival-order archive comes back through this crate's own
/// reader, and the blobs are where arrival order says they are.
///
/// The second half is what stops the first from being free. A round trip
/// through our reader passes over an archive in **either** layout, so on its
/// own it would say nothing about which one was written. Pinning each entry's
/// offset to the position the insertion order predicts is what makes it a test
/// of `Layout::Arrival` rather than of the writer in general, and
/// [`tile_id_offsets`] is the control that says the two predictions differ for
/// this fixture and this seed.
#[test]
#[cfg_attr(miri, ignore)]
fn an_arrival_archive_round_trips_every_tile_and_its_blobs_are_in_arrival_order() {
    let dir = scratch();
    for name in ["dupes-z0z3.pmtiles", "leaves-z0z7.pmtiles"] {
        round_trips_in_arrival_order(name, dir.path());
    }
}

/// One fixture's worth of [`an_arrival_archive_round_trips_every_tile_and_its_blobs_are_in_arrival_order`].
fn round_trips_in_arrival_order(name: &str, dir: &Path) {
    let g = parse_golden(name);
    let out = dir.join(format!("arrival-roundtrip-{name}"));
    let order = write_shuffled(&g, &out, 0x5eed_1144, arrival_options(&g));

    let (expected, total) = arrival_offsets(&order);
    let by_tile_id = tile_id_offsets(&order);
    assert!(
        expected != by_tile_id,
        "{name}: this seed put the tiles in tile id order, so the two layouts \
         predict the same offsets and nothing below can tell them apart"
    );

    let reader = libviprs::pmtiles::Reader::try_open(&out).expect("the archive opens");
    assert_eq!(
        reader.header().tile_data_length,
        total,
        "{name}: the data region is not the distinct payloads laid end to end"
    );

    for (tile_id, payload) in &g.tiles {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
        let got = reader.get_tile(z, x, y).expect("a read succeeds");
        assert_eq!(
            got.as_deref(),
            Some(payload.as_slice()),
            "{name}: tile {tile_id} did not come back as it went in"
        );
        let (at, length) = reader
            .tile_span(z, x, y)
            .expect("a span resolves")
            .expect("the tile is present");
        assert_eq!(
            length as usize,
            payload.len(),
            "{name}: tile {tile_id} changed length"
        );
        assert_eq!(
            at,
            reader.header().tile_data_offset + expected[tile_id],
            "{name}: tile {tile_id} is at {at}, which is where tile id order \
             would put it rather than where it arrived",
        );
    }

    let bytes = std::fs::read(&out).expect("the archive is on disk");
    let report = libviprs::pmtiles::validate::validate_bytes(
        &bytes,
        &libviprs::pmtiles::validate::ValidationLimits::default(),
    )
    .expect("the validator runs");
    assert!(
        report.is_valid(),
        "{name}: the validator found {:?}",
        report.findings
    );
}

/// Write arrival-order archives out where `go-pmtiles` can be pointed at them.
///
/// Not a gate, and `#[ignore]`d so it is never one. The gate for #1143 is the
/// three cells above plus `pmtiles verify`, and this is the part of that last
/// one that has to happen inside this crate: the reference tool cannot build an
/// arrival-order archive, so something has to hand it one.
///
/// It is here rather than in a script because the tiles come from the pinned
/// goldens and the writer is this crate's, so a capture living anywhere else
/// would need its own copy of both. `LIBVIPRS_ARRIVAL_CAPTURE_DIR` says where
/// the archives go; without it they go to the system temp directory and are
/// named on stdout.
///
/// ```text
/// LIBVIPRS_ARRIVAL_CAPTURE_DIR=/work/out \
///   cargo test --test pmtiles_writer -- --ignored --nocapture capture_an_arrival
/// ```
///
/// Each golden is written twice, once each way, because a `pmtiles show` of
/// the arrival archive means little without the tile id one beside it: the two
/// have to report the same tile count, the same zoom range and the same
/// metadata, and differ only in where the sections sit.
#[test]
#[ignore = "a capture tool for the go-pmtiles oracle, not a gate"]
#[cfg_attr(miri, ignore)] // writes archives to the filesystem, which Miri isolation blocks
fn capture_an_arrival_archive_for_the_go_pmtiles_oracle() {
    let dir = std::env::var("LIBVIPRS_ARRIVAL_CAPTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    std::fs::create_dir_all(&dir).expect("the capture directory is writable");

    for name in ["dupes-z0z3.pmtiles", "leaves-z0z7.pmtiles"] {
        let label = name.trim_end_matches(".pmtiles");
        let g = parse_golden(name);
        // Three archives rather than two since #1144. `arrival-inorder` is the
        // one the reference has never been shown before: an archive laid out
        // in arrival order whose header claims `clustered`, which is the claim
        // `pmtiles extract` acts on and `verify` checks against the entries.
        for (suffix, layout, shuffle) in [
            ("arrival", Layout::Arrival, true),
            ("arrival-inorder", Layout::Arrival, false),
            ("tileid", Layout::TileId, true),
        ] {
            let out = dir.join(format!("{label}-{suffix}.pmtiles"));
            let options = WriterOptions::default()
                .with_tile_type(g.header.tile_type)
                .with_tile_compression(g.header.tile_compression)
                .with_layout(layout);
            if shuffle {
                write_shuffled(&g, &out, 0x5eed_1143, options);
            } else {
                let order: Vec<(u64, &[u8])> =
                    g.tiles.iter().map(|(id, p)| (*id, p.as_slice())).collect();
                write_in_this_order(&out, &order, options);
            }
            let bytes = std::fs::read(&out).expect("the archive is on disk");
            let h = Header::try_decode(&bytes[..127]).expect("our header decodes");
            println!(
                "{} {} bytes, tile data at {} for {}, metadata at {}, leaves at {} for {}, clustered {}",
                out.display(),
                bytes.len(),
                h.tile_data_offset,
                h.tile_data_length,
                h.metadata_offset,
                h.leaf_directories_offset,
                h.leaf_directories_length,
                h.clustered,
            );
        }
    }
}

/// A dropped arrival run leaves no archive, and the tile bytes it had already
/// written go with it.
///
/// `dropping_a_writer_without_finishing_cleans_up_after_itself` says this for
/// the default layout, where what is on disk mid-run is a `.data` file that
/// was never going to be published. Arrival order changes what is at stake:
/// the thing holding the payloads **is** the archive, opened at the first
/// tile, so an abandoned run has a partial PMTiles file sitting there. It has
/// to keep its temporary name until `finish` renames it and it has to be
/// removed on the way out, and the positive control in the middle is what
/// stops this passing for a writer that opened nothing at all.
#[test]
#[cfg_attr(miri, ignore)]
fn a_dropped_arrival_run_takes_its_partial_archive_with_it() {
    let dir = scratch();
    let out = dir.path().join("abandoned-arrival.pmtiles");
    {
        let mut w = Writer::create(
            &out,
            WriterOptions::default()
                .with_tile_type(TileType::Png)
                .with_layout(Layout::Arrival),
        )
        .unwrap();
        w.add_tile(0, 0, 0, b"tile", content_hash(b"tile")).unwrap();

        let staged = dir.path().join("abandoned-arrival.pmtiles.tmp");
        let size = std::fs::metadata(&staged)
            .expect("the destination is open and holding the payload")
            .len();
        assert_eq!(
            size,
            16384 + 4,
            "the reserved prefix plus the one four-byte payload is what should \
             be on disk mid-run"
        );
        assert!(!out.exists(), "the final path appeared before finish()");
    }
    let after: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    assert!(
        after.is_empty(),
        "a dropped arrival writer left {after:?} behind"
    );
}

// ---------------------------------------------------------------------------
// clustered, earned rather than bought (issue #1144)
// ---------------------------------------------------------------------------

/// What a walk of an archive's own entries found about the way its data
/// section is laid out.
///
/// The two counts are shape controls. A cell asserting `clustered` over a tile
/// set that turned out to hold no duplicates has not covered the
/// back-reference permission at all, and a count is what says so out loud
/// rather than leaving it to be assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Clustering {
    /// Entries that started exactly where the blobs before them ended.
    contiguous: usize,
    /// Entries that pointed back inside them, which is what dedupe produces.
    back_references: usize,
}

/// The spec's operational definition of `clustered`, applied to the bytes.
///
/// Walking the entries in tile id order, each blob either starts exactly where
/// the blobs before it ended or lies wholly inside them. Anything else leaves
/// a stretch of the data section the walk stepped over, and a reader told the
/// archive is clustered is precisely the reader that will not go back for it.
///
/// This is not the writer's code path. It reads the decoded directory, so a
/// writer that computed the flag out of its own bookkeeping and got it wrong
/// is caught here. What it does share with the writer is one reading of the
/// spec, and go-pmtiles settles that half: `verify` refuses an archive whose
/// header claims clustering its entries do not back up, with "out-of-order
/// entry %v in clustered archive".
fn clustering_of(entries: &[Entry]) -> Result<Clustering, String> {
    let mut laid_down = 0u64;
    let mut found = Clustering {
        contiguous: 0,
        back_references: 0,
    };
    for entry in entries {
        let end = entry.offset.saturating_add(u64::from(entry.length));
        if entry.offset == laid_down {
            laid_down = end;
            found.contiguous += 1;
        } else if end <= laid_down {
            found.back_references += 1;
        } else {
            return Err(format!(
                "the entry for tile {} is at offset {} for {} bytes, which is \
                 neither contiguous with the {laid_down} bytes laid down \
                 before it nor a back reference inside them",
                entry.tile_id, entry.offset, entry.length
            ));
        }
    }
    Ok(found)
}

/// Three payloads at three lengths, so an offset in the directory says which
/// blob it is without anyone having to go and read the bytes.
const BLOB_A: &[u8] = b"aaaa";
const BLOB_B: &[u8] = b"bbbbbb";
const BLOB_C: &[u8] = b"cc";

/// Six tiles carrying every shape that decides clustering.
///
/// * Tiles 1 and 2 are consecutive ids on one blob, so in tile id order they
///   fold into a single entry with `run_length` 2 spanning both.
/// * Tile 4 is deliberately absent, so tiles 3 and 5 are neighbours in the
///   directory without being consecutive ids. They carry the same blob, so
///   their two entries hold the **same** offset: an offset that repeats rather
///   than descends, which a back-reference test spelled `offset < previous`
///   refuses and nothing else in this fixture would catch.
/// * Tile 6 repeats tile 0's blob with two other blobs written in between, so
///   its entry is a real descent in the offset column.
fn clustering_tiles() -> Vec<(u64, &'static [u8])> {
    vec![
        (0, BLOB_A),
        (1, BLOB_B),
        (2, BLOB_B),
        (3, BLOB_C),
        (5, BLOB_C),
        (6, BLOB_A),
    ]
}

/// What the synthetic tile sets below are written with: a tile type, because
/// `Unknown` is a legal value the reference tool then has nothing to say
/// about, and the layout under test.
///
/// The dedupe budget is not decoration. The two sweeps below write 1440
/// archives between them, and at the 8 MiB default each one allocates and
/// zeroes an eight-megabyte window to put three payloads in. That alone was
/// 219 of the 248 seconds this file took, against 29 before the sweeps
/// existed. 64 KiB still buys about a thousand payload slots, which is three
/// orders of magnitude more than any fixture here needs, so nothing evicts and
/// the archives are the ones the default budget produced.
fn synthetic_options(layout: Layout) -> WriterOptions {
    WriterOptions::default()
        .with_tile_type(TileType::Png)
        .with_layout(layout)
        .with_dedupe_memory_bytes(64 * 1024)
}

/// Feed exactly this arrival order into a writer and hand back what it
/// published, parsed.
fn write_in_this_order(out: &Path, order: &[(u64, &[u8])], options: WriterOptions) -> Golden {
    let mut writer = Writer::create(out, options).expect("a writer opens");
    for (tile_id, payload) in order {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
        writer
            .add_tile(z, x, y, payload, content_hash(payload))
            .expect("a tile is accepted");
    }
    writer.finish().expect("the archive finalises");
    parse_ours(&std::fs::read(out).expect("the archive is on disk"))
}

/// Every ordering of `items`, which for the six tiles above is 720 of them.
fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

/// The tile ids of an arrival order, for a failure message that can be read.
fn ids_of(order: &[(u64, &[u8])]) -> Vec<u64> {
    order.iter().map(|(id, _)| *id).collect()
}

/// An arrival run whose tiles happened to arrive in tile id order earns
/// `clustered`, and the archive backs the claim up.
///
/// This is #1144 in one cell. Before it the flag was read off the layout, so
/// this archive said `false` while being every bit as clustered as the tile id
/// one beside it, and `pmtiles extract` refused an input it could have taken.
#[test]
#[cfg_attr(miri, ignore)]
fn an_arrival_run_in_tile_id_order_earns_clustered() {
    let dir = scratch();
    let g = parse_golden("dupes-z0z3.pmtiles");
    let out = dir.path().join("in-order-arrival.pmtiles");

    assert!(
        g.tiles.windows(2).all(|w| w[0].0 < w[1].0),
        "the golden's tiles come out of the parse ascending, which is what \
         makes feeding them straight in an in-order arrival run"
    );
    let mut writer = Writer::create(&out, arrival_options(&g)).expect("a writer opens");
    for (tile_id, payload) in &g.tiles {
        let (z, x, y) = libviprs::pmtiles::tileid_to_zxy(*tile_id).unwrap();
        writer
            .add_tile(z, x, y, payload, content_hash(payload))
            .expect("a tile is accepted");
    }
    writer.finish().expect("the archive finalises");

    let ours = parse_ours(&std::fs::read(&out).expect("the archive is on disk"));
    assert_eq!(
        ours.header.tile_data_offset, 16384,
        "this is not the arrival layout, so nothing below is about it"
    );
    let walk = clustering_of(&ours.entries).expect("the data section is laid out clustered");
    assert!(
        walk.back_references > 0,
        "dupes-z0z3 stopped producing duplicates, so this cell no longer \
         covers the permission a back reference uses: {walk:?}"
    );
    assert!(
        ours.header.clustered,
        "the tiles arrived in tile id order and the data section is laid out \
         clustered ({walk:?}), so the archive has no business telling a reader \
         otherwise"
    );
}

/// Across every arrival order of one tile set, the flag says what the archive
/// says.
///
/// The cells around this one each pin a shape by name. This one is the reason
/// to believe there is not a seventh shape nobody thought of: 720 orders, each
/// laying the data region out differently, every one of them checked against a
/// walk of its own directory rather than against an expectation written here.
///
/// The direction that matters is a `true` the walk denies. A spurious `false`
/// costs a reader work it could have skipped; a spurious `true` tells
/// go-pmtiles it may skip work it cannot, and `pmtiles extract` believes it.
#[test]
#[cfg_attr(miri, ignore)]
fn the_clustered_flag_agrees_with_the_archive_for_every_arrival_order() {
    let dir = scratch();
    let mut earned = 0usize;
    let mut refused = 0usize;

    for (n, order) in permutations(&clustering_tiles()).into_iter().enumerate() {
        let out = dir.path().join(format!("perm-{n}.pmtiles"));
        let ours = write_in_this_order(&out, &order, synthetic_options(Layout::Arrival));
        let ids = ids_of(&order);
        match (clustering_of(&ours.entries), ours.header.clustered) {
            (Ok(_), true) => earned += 1,
            (Err(_), false) => refused += 1,
            (Ok(found), false) => panic!(
                "arrival order {ids:?} laid the data section out clustered \
                 ({found:?}) and the archive says it did not"
            ),
            (Err(why), true) => panic!(
                "arrival order {ids:?} claims clustering its own entries deny: \
                 {why}"
            ),
        }
    }

    assert!(
        earned > 0 && refused > 0,
        "this tile set produced {earned} clustered and {refused} unclustered \
         archives, so one of the two verdicts was never exercised and the \
         agreement above came for free"
    );
}

/// A dedupe back reference to a lesser offset is legal and does not cost an
/// arrival run its clustering.
///
/// The spec grants two permissions, not one, and a tracker that only checked
/// the offset column never goes backwards would refuse this archive its flag.
/// Tile 6 carries tile 0's blob with two other blobs written in between, so
/// its entry points at offset 0 from a position where the blobs laid down
/// already run past it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_dedupe_back_reference_does_not_cost_an_arrival_run_its_clustering() {
    let dir = scratch();
    let out = dir.path().join("back-reference.pmtiles");
    let ours = write_in_this_order(
        &out,
        &clustering_tiles(),
        synthetic_options(Layout::Arrival),
    );

    let last = *ours.entries.last().expect("the archive has entries");
    assert_eq!(last.tile_id, 6, "the fixture's last entry moved");
    let in_front = ours.entries[..ours.entries.len() - 1]
        .iter()
        .map(|e| e.offset)
        .max()
        .expect("there is something in front of it");
    assert!(
        last.offset < in_front,
        "tile 6's entry is at offset {} with {in_front} in front of it, so \
         this fixture no longer holds a descent in the offset column and the \
         assertion below is about nothing",
        last.offset
    );

    let walk = clustering_of(&ours.entries).expect("the data section is laid out clustered");
    assert!(
        ours.header.clustered,
        "a back reference to a lesser offset is the spec's second permission \
         rather than a break in the ordering ({walk:?})"
    );
}

/// Two entries carrying one offset are a back reference too, not a descent.
///
/// Tile 4 is absent from the fixture, so tiles 3 and 5 are neighbours in the
/// directory without being consecutive ids: they cannot fold into a run, and
/// they carry the same blob, so the offset column repeats rather than moving.
/// A back-reference test spelled `offset < previous` refuses exactly this one
/// and nothing else.
#[test]
#[cfg_attr(miri, ignore)]
fn two_entries_at_one_offset_are_not_a_descent() {
    let dir = scratch();
    let out = dir.path().join("repeated-offset.pmtiles");
    let ours = write_in_this_order(
        &out,
        &clustering_tiles(),
        synthetic_options(Layout::Arrival),
    );

    let at = |tile_id: u64| {
        ours.entries
            .iter()
            .position(|e| e.tile_id == tile_id)
            .unwrap_or_else(|| panic!("tile {tile_id} has an entry of its own"))
    };
    let (three, five) = (at(3), at(5));
    assert_eq!(five, three + 1, "tiles 3 and 5 are no longer neighbours");
    assert_eq!(
        ours.entries[three].offset, ours.entries[five].offset,
        "tiles 3 and 5 stopped sharing a blob, so the offset column no longer \
         repeats here and this cell covers nothing"
    );
    assert_eq!(
        (
            ours.entries[three].run_length,
            ours.entries[five].run_length
        ),
        (1, 1),
        "the two folded into runs, which is the shape the cell below this one \
         is for"
    );

    let walk = clustering_of(&ours.entries).expect("the data section is laid out clustered");
    assert!(
        ours.header.clustered,
        "an offset that repeats is a back reference into bytes already laid \
         down, not a step backwards ({walk:?})"
    );
}

/// A run of identical adjacent tiles is one entry over many tile ids, and it
/// clusters.
///
/// Four consecutive ids on one blob collapse into a single entry with
/// `run_length` 4, so the directory has three entries for six tiles and the
/// walk sees one offset standing for four tile ids.
#[test]
#[cfg_attr(miri, ignore)]
fn a_run_of_identical_adjacent_tiles_still_clusters_under_arrival() {
    let dir = scratch();
    let out = dir.path().join("run-arrival.pmtiles");
    let tiles: Vec<(u64, &[u8])> = vec![
        (0, BLOB_A),
        (1, BLOB_B),
        (2, BLOB_B),
        (3, BLOB_B),
        (4, BLOB_B),
        (5, BLOB_C),
    ];
    let ours = write_in_this_order(&out, &tiles, synthetic_options(Layout::Arrival));

    assert_eq!(
        ours.entries.len(),
        3,
        "six tiles did not fold into three entries, so there is no run here"
    );
    assert_eq!(
        (ours.entries[1].tile_id, ours.entries[1].run_length),
        (1, 4),
        "the middle entry is not the four-tile run this cell is about"
    );

    let walk = clustering_of(&ours.entries).expect("the data section is laid out clustered");
    assert!(
        ours.header.clustered,
        "a run is one blob covering four tile ids, not four steps through the \
         data section ({walk:?})"
    );
}

/// A run whose own tiles arrived backwards still clusters, because the blob
/// was placed once and the placement is what the flag is about.
///
/// The arrival sequence here descends four times and the archive is clustered
/// anyway: tiles 4, 3, 2 and 1 share one blob, so only the first of them
/// places anything. A tracker that watched the arrival sequence for descents
/// rather than the placements would call this unclustered, which is the safe
/// direction and still wrong.
#[test]
#[cfg_attr(miri, ignore)]
fn a_run_whose_tiles_arrived_backwards_still_clusters() {
    let dir = scratch();
    let out = dir.path().join("run-backwards.pmtiles");
    let order: Vec<(u64, &[u8])> = vec![
        (0, BLOB_A),
        (4, BLOB_B),
        (3, BLOB_B),
        (2, BLOB_B),
        (1, BLOB_B),
        (5, BLOB_C),
    ];
    let descents = order.windows(2).filter(|w| w[1].0 < w[0].0).count();
    assert_eq!(
        descents, 3,
        "this arrival order no longer goes backwards, so it cannot tell a \
         placement tracker apart from an arrival one"
    );

    let ours = write_in_this_order(&out, &order, synthetic_options(Layout::Arrival));
    let walk = clustering_of(&ours.entries).expect("the data section is laid out clustered");
    assert!(
        ours.header.clustered,
        "the three descents in the arrival order all landed on a blob that was \
         already placed, so nothing about the data section moved ({walk:?})"
    );
}

/// Exactly one descent, at the very last tile, still clears the flag.
///
/// The pair is the point. Two runs over the same eight tiles differing only in
/// where tile 0 arrives, and the flag has to come out differently. Every
/// payload is distinct on purpose: a last tile that deduplicated into a blob
/// already placed would be a back reference, and the archive would genuinely
/// still be clustered.
#[test]
#[cfg_attr(miri, ignore)]
fn one_descent_at_the_last_tile_clears_clustered() {
    let dir = scratch();
    let payloads: Vec<Vec<u8>> = (0..8u64)
        .map(|i| format!("tile {i} carries its own bytes").into_bytes())
        .collect();
    let ascending: Vec<(u64, &[u8])> = (0..8u64)
        .map(|i| (i, payloads[i as usize].as_slice()))
        .collect();
    let mut moved = ascending.clone();
    let first = moved.remove(0);
    moved.push(first);

    let descents = |order: &[(u64, &[u8])]| order.windows(2).filter(|w| w[1].0 < w[0].0).count();
    assert_eq!(descents(&ascending), 0, "the control is not in order");
    assert_eq!(
        descents(&moved),
        1,
        "moving tile 0 to the end has to leave exactly one descent, or this \
         cell is not about the boundary it names"
    );
    assert_eq!(
        moved.last().expect("the order is not empty").0,
        0,
        "the one descent is not at the last arrival"
    );

    let control = write_in_this_order(
        &dir.path().join("ascending.pmtiles"),
        &ascending,
        synthetic_options(Layout::Arrival),
    );
    let broken = write_in_this_order(
        &dir.path().join("moved.pmtiles"),
        &moved,
        synthetic_options(Layout::Arrival),
    );

    let walk = clustering_of(&control.entries).expect("the in-order run lays out clustered");
    assert_eq!(
        walk.back_references, 0,
        "the payloads stopped being distinct, so the moved tile below could \
         deduplicate into place and the run would still be clustered: {walk:?}"
    );
    assert!(
        control.header.clustered,
        "the in-order control lost its flag"
    );

    let why = clustering_of(&broken.entries)
        .expect_err("one tile arriving last has to break the ordering");
    assert!(
        !broken.header.clustered,
        "the archive claims clustering its own entries deny: {why}"
    );
    assert_eq!(
        control.entries.len(),
        broken.entries.len(),
        "the two runs no longer carry the same tiles"
    );
}

/// `Layout::TileId` earns the flag whatever order its tiles arrived in, which
/// is a check on the tracker rather than on the archive.
///
/// That layout assigns offsets walking the entries in tile id order, taking
/// the next free one for a blob it has not placed and an earlier one for a
/// blob it has, so there is no path through it that lays out an unclustered
/// data region. Before #1144 the flag was read off the layout and this could
/// not have failed. It can now, and the way it fails is a tracker fed the
/// staged offset a payload arrived at instead of the offset it was placed at.
#[test]
#[cfg_attr(miri, ignore)]
fn layout_tile_id_earns_clustered_whatever_the_arrival_order() {
    let dir = scratch();
    for (n, order) in permutations(&clustering_tiles()).into_iter().enumerate() {
        let out = dir.path().join(format!("tileid-perm-{n}.pmtiles"));
        let ours = write_in_this_order(&out, &order, synthetic_options(Layout::TileId));
        let ids = ids_of(&order);
        let walk = clustering_of(&ours.entries).unwrap_or_else(|why| {
            panic!("tile id order laid out an unclustered data region from arrival order {ids:?}: {why}")
        });
        assert!(
            ours.header.clustered,
            "tile id order did not claim the clustering it produced from \
             arrival order {ids:?} ({walk:?})"
        );
    }
}
