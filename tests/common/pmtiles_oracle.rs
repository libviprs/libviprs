//! Loading the committed `go-pmtiles` reference vectors, with their digests
//! pinned (issue #991).
//!
//! Every number these helpers hand out came from the real `go-pmtiles`
//! v1.31.2 and is read out of a JSON file under
//! `tests/fixtures/pmtiles/vectors/` at run time. The alternative, copying the
//! numbers into a Rust `const` table, is what this exists to avoid: a
//! transcribed literal and an invented one look exactly the same in a diff,
//! and the only thing separating them is whether somebody actually ran the
//! reference implementation. Reading the file keeps the provenance attached to
//! the value.
//!
//! # Two things it refuses to do quietly
//!
//! **A vector file whose bytes have changed is not the vector file.** Each one
//! is pinned by sha256 in [`VECTOR_DIGESTS`] and [`read_pinned`] fails naming
//! both digests. An oracle you are allowed to edit is not an oracle, and the
//! edit that matters is the easy one: a test goes red, somebody "fixes" the
//! expectation, and the suite is now pinned to libviprs' own output.
//!
//! **A parse that yields nothing must not pass.** Every section accessor
//! checks the number of rows it parsed against the `counts` block inside the
//! same file, so a renamed key or a changed shape fails loudly instead of
//! handing back an empty `Vec` that every assertion in the caller's loop then
//! agrees with. That is the positive control the evidence bar asks for, and it
//! is not hypothetical: a `for` loop over zero rows is green.
//!
//! # The rows that are evidence rather than expectations
//!
//! `tileid.json`'s `out_of_range_observations` section records what
//! `ZxyToID` *does* with an impossible coordinate, which is to mask `x` and
//! `y` into a different valid tile and to saturate `z` above 31 so that z=32,
//! 33 and 63 all return one id. libviprs refuses all of those. So
//! [`TileIdRow::roundtrip_ok`] is carried through rather than dropped, and the
//! caller is expected to assert a refusal for those rows rather than a value.
//! Pinning them as targets would build a reader that reproduces a reference
//! bug, with a bigger green table than the one that got it right.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The reference vector files and the sha256 each must still have.
///
/// Measured on the copies committed under `tests/fixtures/pmtiles/vectors/`,
/// which are byte-for-byte the capture's own output. `PROVENANCE.md` beside
/// them carries the same table.
pub const VECTOR_DIGESTS: &[(&str, &str)] = &[
    (
        "tileid.json",
        "a486b48b09ab1b9d8f20208b992fc47b89ba67c235506f5f47cccd808e82b265",
    ),
    (
        "header.json",
        "99258d11ea1fa9cd99c8b28a74ea1bf217e0dea87b4ee00776a6b0c1ea36f1c3",
    ),
    (
        "directory.json",
        "9f01472702fd4e93c3025bd9897cc336429a1bc05385f35ae555f238490e4d47",
    ),
    (
        "directory-leaves.json",
        "acc033de338def6a600a806a03cf803cacfe87470a3a8a059f0bace3b9320d58",
    ),
];

/// The three golden archives and the sha256 each must still have.
///
/// The archives themselves are committed by the format lane (#987); these
/// digests are the ones its `PROVENANCE.md` records, and they are repeated
/// here so a test that reads an archive fails on a changed one rather than
/// on the assertion downstream of it.
pub const GOLDEN_DIGESTS: &[(&str, &str)] = &[
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

/// `tests/fixtures/pmtiles/`, where the archives live.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pmtiles")
}

/// `tests/fixtures/pmtiles/vectors/`, where the JSON lives.
pub fn vector_dir() -> PathBuf {
    fixture_dir().join("vectors")
}

/// Lowercase hex, because `crate::hex` is `pub(crate)` and an integration
/// test is a different crate.
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The sha256 of `bytes`, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

/// Read a file and refuse it if its digest is not the one pinned.
///
/// Panics rather than returning a `Result`, because every caller is a test and
/// the only useful response to "the oracle has been edited" is to stop.
pub fn read_pinned(path: &Path, want_sha256: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "cannot read the pinned fixture {}: {e}. The three golden archives \
             are committed by issue #987; if they are not in the tree yet, this \
             test cannot run.",
            path.display()
        )
    });
    let got = sha256_hex(&bytes);
    assert_eq!(
        got,
        want_sha256,
        "{} has changed: sha256 is {got}, the pin says {want_sha256}. \
         These bytes came out of go-pmtiles v1.31.2 and are not ours to edit; \
         if the capture was genuinely redone, re-pin it in \
         tests/common/pmtiles_oracle.rs and say so in the provenance note.",
        path.display()
    );
    bytes
}

/// One vector file, parsed, with its digest checked first.
pub fn vectors(name: &str) -> Value {
    let want = VECTOR_DIGESTS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("{name} is not one of the pinned vector files"))
        .1;
    let bytes = read_pinned(&vector_dir().join(name), want);
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name} is not valid JSON: {e}"))
}

/// One golden archive's bytes, with its digest checked first.
pub fn golden(name: &str) -> Vec<u8> {
    let want = GOLDEN_DIGESTS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("{name} is not one of the pinned goldens"))
        .1;
    read_pinned(&fixture_dir().join(name), want)
}

/// One `(z, x, y) -> tile_id` row out of `tileid.json`.
///
/// `roundtrip_ok` is the reference's own answer to "does `IDToZxy` of this id
/// give the coordinate back", and it is `false` for exactly the rows where
/// `ZxyToID` masked or saturated something. Carrying it is what lets a caller
/// tell a target from an observation without hard-coding which section is
/// which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileIdRow {
    pub z: u8,
    pub x: u32,
    pub y: u32,
    pub tile_id: u64,
    pub roundtrip_ok: bool,
    pub note: String,
}

fn u64_at(row: &Value, key: &str, section: &str) -> u64 {
    row.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{section} row is missing an integer {key}: {row}"))
}

/// Every row of one section of `tileid.json`, with the row count checked
/// against the file's own `counts` block.
pub fn tileid_section(section: &str) -> Vec<TileIdRow> {
    let doc = vectors("tileid.json");
    let declared = doc
        .get("counts")
        .and_then(|c| c.get(section))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("tileid.json has no counts entry for {section}"));

    let rows = doc
        .get(section)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("tileid.json has no {section} array"));

    let parsed: Vec<TileIdRow> = rows
        .iter()
        .map(|row| TileIdRow {
            z: u8::try_from(u64_at(row, "z", section)).expect("a zoom fits in a u8"),
            x: u32::try_from(u64_at(row, "x", section)).expect("an x fits in a u32"),
            y: u32::try_from(u64_at(row, "y", section)).expect("a y fits in a u32"),
            tile_id: u64_at(row, "tile_id", section),
            roundtrip_ok: row
                .get("roundtrip_ok")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| panic!("{section} row is missing roundtrip_ok: {row}")),
            note: row
                .get("note")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        })
        .collect();

    assert_eq!(
        parsed.len() as u64,
        declared,
        "parsed {} rows from tileid.json's {section} but the file's own counts \
         block says {declared}. A parse that silently yields fewer rows makes \
         every assertion in the caller's loop vacuous, which is the failure \
         this check exists for.",
        parsed.len()
    );
    assert!(
        declared > 0,
        "the {section} section is empty, so nothing in it can fail"
    );
    parsed
}

/// The `(zoom, first_tile_id, last_tile_id, tile_count)` rows of
/// `tileid.json`'s `level_bases`, count-checked the same way.
pub fn level_bases() -> Vec<(u8, u64, u64, u64)> {
    let doc = vectors("tileid.json");
    let declared = doc
        .get("counts")
        .and_then(|c| c.get("level_bases"))
        .and_then(Value::as_u64)
        .expect("tileid.json has a counts entry for level_bases");
    let rows = doc
        .get("level_bases")
        .and_then(Value::as_array)
        .expect("tileid.json has a level_bases array");
    let parsed: Vec<(u8, u64, u64, u64)> = rows
        .iter()
        .map(|row| {
            (
                u8::try_from(u64_at(row, "z", "level_bases")).expect("a zoom fits in a u8"),
                u64_at(row, "first_tile_id", "level_bases"),
                u64_at(row, "last_tile_id", "level_bases"),
                u64_at(row, "tile_count", "level_bases"),
            )
        })
        .collect();
    assert_eq!(
        parsed.len() as u64,
        declared,
        "parsed {} level_bases rows against a declared {declared}",
        parsed.len()
    );
    parsed
}

/// One directory entry as `go-pmtiles` decoded it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleEntry {
    pub tile_id: u64,
    pub offset: u64,
    pub length: u32,
    pub run_length: u32,
}

/// Decode a hex string from a vector file into bytes.
pub fn unhex(hex: &str) -> Vec<u8> {
    assert!(
        hex.len() % 2 == 0,
        "a hex string with an odd length is not bytes"
    );
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .unwrap_or_else(|e| panic!("{} is not hex: {e}", &hex[i..i + 2]))
        })
        .collect()
}

/// The `root_directory` block of one archive in `directory.json`, as
/// `(decompressed bytes, entries)`.
///
/// The entry count is checked against the block's own `entry_count`, so a
/// shape change fails here rather than producing an empty comparison.
pub fn root_directory(archive: &str) -> (Vec<u8>, Vec<OracleEntry>) {
    let doc = vectors("directory.json");
    let block = doc
        .get("archives")
        .and_then(|a| a.get(archive))
        .and_then(|a| a.get("root_directory"))
        .unwrap_or_else(|| panic!("directory.json has no root_directory for {archive}"));
    let bytes = unhex(
        block
            .get("decompressed_hex")
            .and_then(Value::as_str)
            .expect("the root directory block has decompressed_hex"),
    );
    let declared = block
        .get("entry_count")
        .and_then(Value::as_u64)
        .expect("the root directory block has an entry_count");
    let entries: Vec<OracleEntry> = block
        .get("entries")
        .and_then(Value::as_array)
        .expect("the root directory block has entries")
        .iter()
        .map(|e| OracleEntry {
            tile_id: u64_at(e, "tile_id", archive),
            offset: u64_at(e, "offset", archive),
            length: u32::try_from(u64_at(e, "length", archive)).expect("a length fits in a u32"),
            run_length: u32::try_from(u64_at(e, "run_length", archive))
                .expect("a run length fits in a u32"),
        })
        .collect();
    assert_eq!(
        entries.len() as u64,
        declared,
        "parsed {} entries for {archive} against a declared entry_count of {declared}",
        entries.len()
    );
    assert!(!entries.is_empty(), "{archive} decoded to no entries at all");
    (bytes, entries)
}

/// One leaf directory of `leaves-z0z7.pmtiles`, out of
/// `directory-leaves.json`.
#[derive(Debug, Clone)]
pub struct OracleLeaf {
    /// The id of the root entry that points at this leaf.
    pub pointer_tile_id: u64,
    /// The offset the root entry carries, which is **relative to
    /// `header.leaf_directory_offset`** and not to the start of the file.
    pub offset_in_leaf_section: u64,
    /// Where that lands in the file.
    pub absolute_offset: u64,
    pub compressed_length: u64,
    pub entries: Vec<OracleEntry>,
}

/// Every leaf directory of the leaf-bearing golden, with the per-leaf entry
/// count checked against the file's own `entry_count`.
pub fn leaf_directories() -> Vec<OracleLeaf> {
    let doc = vectors("directory-leaves.json");
    let leaves = doc
        .get("leaf_directories")
        .and_then(Value::as_array)
        .expect("directory-leaves.json has a leaf_directories array");
    assert!(
        !leaves.is_empty(),
        "the leaf-bearing golden parsed to zero leaves, so nothing below can fail"
    );
    leaves
        .iter()
        .map(|leaf| {
            let declared = leaf
                .get("entry_count")
                .and_then(Value::as_u64)
                .expect("a leaf block has an entry_count");
            let columns = leaf
                .get("entries_columnar")
                .expect("a leaf block has entries_columnar");
            let column = |name: &str| -> Vec<u64> {
                columns
                    .get(name)
                    .and_then(Value::as_array)
                    .unwrap_or_else(|| panic!("a leaf block has no {name} column"))
                    .iter()
                    .map(|v| v.as_u64().expect("a column holds integers"))
                    .collect()
            };
            let ids = column("tile_id");
            let offsets = column("offset");
            let lengths = column("length");
            let runs = column("run_length");
            assert_eq!(
                ids.len() as u64,
                declared,
                "a leaf's tile_id column has {} values against a declared \
                 entry_count of {declared}",
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
                pointer_tile_id: leaf
                    .get("pointer_tile_id")
                    .and_then(Value::as_u64)
                    .expect("a leaf block has a pointer_tile_id"),
                offset_in_leaf_section: leaf
                    .get("offset_in_leaf_section")
                    .and_then(Value::as_u64)
                    .expect("a leaf block has an offset_in_leaf_section"),
                absolute_offset: leaf
                    .get("absolute_offset")
                    .and_then(Value::as_u64)
                    .expect("a leaf block has an absolute_offset"),
                compressed_length: leaf
                    .get("compressed_length")
                    .and_then(Value::as_u64)
                    .expect("a leaf block has a compressed_length"),
                entries: (0..ids.len())
                    .map(|i| OracleEntry {
                        tile_id: ids[i],
                        offset: offsets[i],
                        length: u32::try_from(lengths[i]).expect("a length fits in a u32"),
                        run_length: u32::try_from(runs[i]).expect("a run length fits in a u32"),
                    })
                    .collect(),
            }
        })
        .collect()
}

/// The header fields `go-pmtiles` decoded for one archive, out of
/// `header.json`.
pub fn oracle_header(archive: &str) -> serde_json::Map<String, Value> {
    let doc = vectors("header.json");
    doc.get("archives")
        .and_then(|a| a.get(archive))
        .and_then(|a| a.get("decoded_by_pmtiles_DeserializeHeader"))
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("header.json has no decoded header for {archive}"))
        .clone()
}

/// One `u64` field out of an oracle header block.
pub fn header_u64(fields: &serde_json::Map<String, Value>, key: &str) -> u64 {
    fields
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("the oracle header has no u64 field {key}"))
}
