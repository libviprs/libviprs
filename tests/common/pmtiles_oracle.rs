//! Loading the committed go-pmtiles oracle vectors, by path, at run time.
//!
//! Everything under `tests/fixtures/pmtiles/` came out of the real
//! `protomaps/go-pmtiles` v1.31.2 (commit
//! `a3e4951ea6a0477b784c27c1dcbfd9c130878c5a`). `PROVENANCE.md` beside the
//! files says how. This module is how a test reaches them, and it exists
//! because the reader (#988), the writer (#989) and the correctness suite
//! (#991) all pin against the same files and should not each grow their own
//! parser for them.
//!
//! # Why these are loaded rather than transcribed
//!
//! A transcribed literal is indistinguishable from an invented one. A number
//! copied out of a JSON file into a Rust `const` carries no evidence of where
//! it came from, and if someone later "fixes" a failing test by editing the
//! constant, nothing anywhere notices. Reading the committed file at run time
//! keeps the oracle and the expectation the same object, and `git log` on the
//! fixture is the audit trail.
//!
//! # Three ways this could quietly stop checking anything, and the guard for each
//!
//! * **The file could be the wrong file.** Every load verifies the sha256 of
//!   the bytes it just read against a pinned digest and panics on a mismatch,
//!   so editing a vector to make a test pass fails loudly at the next load
//!   rather than silently moving the target.
//! * **The parse could come back empty.** A loop over zero rows passes every
//!   assertion inside it. So [`tile_id_rows`] cross-checks the number of rows
//!   it parsed against the `counts` block *inside the same file*, which is a
//!   number go-pmtiles' dump program wrote and this parser did not, and
//!   refuses if they disagree or if the count is missing.
//! * **The section could be the wrong section.** Section names are passed in
//!   by the caller, and a typo would otherwise read as "no rows". A missing
//!   section is a panic naming the sections that do exist, never an empty
//!   vector.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// sha256 of `tests/fixtures/pmtiles/vectors/tileid.json`.
pub const TILEID_JSON_SHA256: &str =
    "a486b48b09ab1b9d8f20208b992fc47b89ba67c235506f5f47cccd808e82b265";
/// sha256 of `tests/fixtures/pmtiles/vectors/header.json`.
pub const HEADER_JSON_SHA256: &str =
    "92e0d18f2b10781249f8b731cafe54aa16936404270cfd30c9dc28e58559af39";
/// sha256 of `tests/fixtures/pmtiles/vectors/directory.json`.
pub const DIRECTORY_JSON_SHA256: &str =
    "9f01472702fd4e93c3025bd9897cc336429a1bc05385f35ae555f238490e4d47";
/// sha256 of `tests/fixtures/pmtiles/vectors/directory-leaves.json`.
pub const DIRECTORY_LEAVES_JSON_SHA256: &str =
    "acc033de338def6a600a806a03cf803cacfe87470a3a8a059f0bace3b9320d58";
/// sha256 of `tests/fixtures/pmtiles/vectors/tiles.json`.
pub const TILES_JSON_SHA256: &str =
    "efaebeee9399d9e1e6e0395059caf38c659f134353442bfe29fd0065e5ae6581";

/// sha256 of `tests/fixtures/pmtiles/raster-z0z2.pmtiles`.
pub const RASTER_GOLDEN_SHA256: &str =
    "e2ed5e64f3c29efa3ec3b679ec5f1b06569c1b234c6eea762fb9f02fc23e9c12";
/// sha256 of `tests/fixtures/pmtiles/dupes-z0z3.pmtiles`.
pub const DUPES_GOLDEN_SHA256: &str =
    "bfc9db4c6ce6a04194e02b3d4815814adb05209f1aaba8591e4e1332f6e56a27";
/// sha256 of `tests/fixtures/pmtiles/leaves-z0z7.pmtiles`.
pub const LEAVES_GOLDEN_SHA256: &str =
    "fe5c9636be61abc60046d7f13837f8a3efb20ce3c38303644dac0cbec8248b8d";
/// sha256 of `tests/fixtures/pmtiles/distinct-z0z7.pmtiles`.
///
/// The leaf golden with leaves that start somewhere. Every leaf in
/// `leaves-z0z7` holds tile entries starting at offset 0, which makes
/// "rebase each leaf on its own first entry" the identity, and measured, a
/// reader that does exactly that passes all 128 tests here. This archive's five
/// leaves start at 0, 49164, 98324, 147497 and 196597, so the mistake moves
/// bytes in four of the five.
pub const DISTINCT_GOLDEN_SHA256: &str =
    "a32dce77a93a304dbd27b80d72160b29455446931b477b5b1b7a84155ecd2dd5";

/// The release the whole fixture set was produced by, asserted out of the
/// `produced_by` block of every vector file so a fixture swapped for one from
/// a different version of the tool fails rather than passing quietly.
pub const ORACLE_RELEASE_TAG: &str = "v1.31.2";
/// The go-pmtiles commit behind [`ORACLE_RELEASE_TAG`].
pub const ORACLE_SOURCE_COMMIT: &str = "a3e4951ea6a0477b784c27c1dcbfd9c130878c5a";

/// `tests/fixtures/pmtiles/`, absolute.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pmtiles")
}

/// Hex-encode, lowercase.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase hex string, panicking with the offending text rather
/// than returning a shorter buffer, because a silently short fixture is the
/// empty-result trap again.
pub fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(
        text.len() % 2,
        0,
        "hex string has an odd length ({})",
        text.len()
    );
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16)
                .unwrap_or_else(|_| panic!("not hex at byte {}: {:?}", i / 2, &text[i..i + 2]))
        })
        .collect()
}

/// Read one fixture and check its sha256 before handing the bytes back.
pub fn read_checked(relative: &str, want_sha256: &str) -> Vec<u8> {
    let path = fixtures_dir().join(relative);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read the committed fixture {}: {e}", path.display()));
    let got = hex(&Sha256::digest(&bytes));
    assert_eq!(
        got,
        want_sha256,
        "{} is not the file this test was pinned against",
        path.display()
    );
    bytes
}

/// One of the golden archives, bytes, sha256-checked.
pub fn golden(name: &str, want_sha256: &str) -> Vec<u8> {
    read_checked(name, want_sha256)
}

/// One of the vector files, parsed, sha256-checked, with its `produced_by`
/// block confirmed to name the pinned release.
pub fn vectors(name: &str, want_sha256: &str) -> Value {
    let bytes = read_checked(&format!("vectors/{name}"), want_sha256);
    let value: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("vectors/{name} is not JSON: {e}"));
    let produced = value
        .get("produced_by")
        .unwrap_or_else(|| panic!("vectors/{name} has no produced_by block"));
    assert_eq!(
        produced.get("release_tag").and_then(Value::as_str),
        Some(ORACLE_RELEASE_TAG),
        "vectors/{name} was produced by a different go-pmtiles release"
    );
    assert_eq!(
        produced.get("source_commit").and_then(Value::as_str),
        Some(ORACLE_SOURCE_COMMIT),
        "vectors/{name} was produced from a different go-pmtiles commit"
    );
    value
}

/// The whole of `vectors/tileid.json`.
pub fn tileid_vectors() -> Value {
    vectors("tileid.json", TILEID_JSON_SHA256)
}

/// The whole of `vectors/header.json`.
pub fn header_vectors() -> Value {
    vectors("header.json", HEADER_JSON_SHA256)
}

/// The whole of `vectors/directory.json`.
pub fn directory_vectors() -> Value {
    vectors("directory.json", DIRECTORY_JSON_SHA256)
}

/// The whole of `vectors/directory-leaves.json`.
pub fn directory_leaves_vectors() -> Value {
    vectors("directory-leaves.json", DIRECTORY_LEAVES_JSON_SHA256)
}

/// One `(z, x, y) -> tile_id` row as the oracle dumped it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileIdRow {
    pub z: u8,
    pub x: u32,
    pub y: u32,
    pub tile_id: u64,
    /// `IDToZxy(tile_id) == (z, x, y)`, as the dump program computed it. It is
    /// false exactly on the out-of-range rows, where the reference masked the
    /// input and answered a different tile's id.
    pub roundtrip_ok: bool,
    /// What `IDToZxy(tile_id)` answered. On an in-range row it is `(z, x, y)`
    /// again; on a masked row it is the *other* tile the reference silently
    /// substituted, which is what makes it a usable positive control.
    pub roundtrip: (u64, u64, u64),
}

fn u64_at(row: &Value, key: &str, section: &str, index: usize) -> u64 {
    row.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{section}[{index}] has no unsigned {key}"))
}

/// Every row of one named section of `tileid.json`, cross-checked against the
/// `counts` block in the same file.
///
/// Panics rather than returning an empty vector on a missing section, a
/// missing count, or a count that disagrees with what was parsed. All three
/// are the same failure from the reader's point of view (a loop that runs zero
/// times) and all three have to be loud.
pub fn tile_id_rows(vectors: &Value, section: &str) -> Vec<TileIdRow> {
    let array = vectors
        .get(section)
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            let have: Vec<&str> = vectors
                .as_object()
                .map(|m| m.keys().map(String::as_str).collect())
                .unwrap_or_default();
            panic!("tileid.json has no array section {section:?}; it has {have:?}")
        });

    let rows: Vec<TileIdRow> = array
        .iter()
        .enumerate()
        .map(|(index, row)| TileIdRow {
            z: u8::try_from(u64_at(row, "z", section, index)).expect("a zoom fits in a u8"),
            x: u32::try_from(u64_at(row, "x", section, index)).expect("an x fits in a u32"),
            y: u32::try_from(u64_at(row, "y", section, index)).expect("a y fits in a u32"),
            tile_id: u64_at(row, "tile_id", section, index),
            roundtrip_ok: row
                .get("roundtrip_ok")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| panic!("{section}[{index}] has no roundtrip_ok")),
            roundtrip: (
                u64_at(row, "roundtrip_z", section, index),
                u64_at(row, "roundtrip_x", section, index),
                u64_at(row, "roundtrip_y", section, index),
            ),
        })
        .collect();

    let declared = declared_count(vectors, section);
    assert_eq!(
        rows.len(),
        declared,
        "parsed {} rows of {section} but the file's own counts block says {declared}",
        rows.len()
    );
    assert!(!rows.is_empty(), "{section} is empty");
    rows
}

/// The `counts` entry for a section, which the dump program wrote and this
/// parser did not, so it is an independent check on the parse.
pub fn declared_count(vectors: &Value, section: &str) -> usize {
    let counts = vectors
        .get("counts")
        .and_then(Value::as_object)
        .expect("tileid.json has no counts block");
    let declared = counts
        .get(section)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            let have: Vec<&str> = counts.keys().map(String::as_str).collect();
            panic!("counts has no entry for {section:?}; it has {have:?}")
        });
    usize::try_from(declared).expect("a count fits in a usize")
}
