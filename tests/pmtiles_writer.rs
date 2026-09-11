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
use libviprs::pmtiles::writer::{Writer, WriterOptions, content_hash};
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
#[test]
#[cfg_attr(miri, ignore)]
fn identical_payloads_are_stored_once_whatever_the_engine_dedupe_strategy_is() {
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
        assert!(
            entry.offset == next_byte || entry.offset < next_byte,
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
