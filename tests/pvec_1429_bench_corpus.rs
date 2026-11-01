//! N8.1 (#1429, slice of #1064): the six benchmark cells are pinned in
//! `tests/fixtures/pvec_bench_corpus.json`, and this suite holds the manifest and the generators
//! together. It goes red on a missing cell, a drifted checksum, two corpus revisions, a cell with
//! no measured entity count, a seventh cell, and a `malformed` cell that does not cover an N3.6
//! limit or whose inputs are not refused by the limit they name.
//!
//! Regenerating the manifest numbers is one command, which also prints the rows:
//!
//! ```text
//! cargo test --release --features pvec-convert --test pvec_1429_bench_corpus -- \
//!     --ignored --nocapture measure_cells
//! ```
//!
//! Nothing here runs a converter, needs a proprietary tool, or touches LibreDWG.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use pvec_bench::{
    CELLS, MANIFEST, Regen, cells, count, generate, generate_malformed, manifest, problems,
    sha256_hex,
};
use serde_json::{Value, json};

#[test]
fn the_manifest_is_a_good_corpus() {
    let bad = problems(&manifest(), Regen::Small);
    assert!(bad.is_empty(), "{MANIFEST}:\n{}", bad.join("\n"));
}

#[test]
fn six_cells_resolve_by_name_to_a_checksum() {
    let m = manifest();
    let c = cells(&m);
    assert_eq!(
        c.keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>(),
        CELLS.into_iter().collect()
    );
    for (name, cell) in c.iter().filter(|(n, _)| *n != "malformed") {
        assert_eq!(cell["sha256"].as_str().map(str::len), Some(64), "{name}");
    }
}

#[test]
fn a_missing_cell_turns_it_red() {
    let mut m = manifest();
    m["cells"].as_object_mut().unwrap().remove("block_heavy");
    assert!(
        problems(&m, Regen::No)
            .iter()
            .any(|p| p.contains("block_heavy is missing"))
    );
}

#[test]
fn a_drifted_checksum_turns_it_red() {
    let mut m = manifest();
    m["cells"]["simple"]["sha256"] = json!("0".repeat(64));
    assert!(
        problems(&m, Regen::Small)
            .iter()
            .any(|p| p.contains("cell simple") && p.contains("checksum"))
    );
}

#[test]
fn two_corpus_revisions_turn_it_red() {
    let mut m = manifest();
    m["cells"]["spline_heavy"]["corpus_revision"] = json!("r1-open");
    assert!(
        problems(&m, Regen::No)
            .iter()
            .any(|p| p.contains("cell spline_heavy") && p.contains("corpus revision"))
    );
}

#[test]
fn a_revision_other_than_the_vendored_pin_turns_it_red() {
    let mut m = manifest();
    m["corpus_revision"] = json!("r9-open");
    assert!(
        problems(&m, Regen::No)
            .iter()
            .any(|p| p.contains("vendored pin"))
    );
}

#[test]
fn a_seventh_cell_without_a_measured_count_turns_it_red() {
    let mut m = manifest();
    let mut extra = m["cells"]["simple"].clone();
    extra["entities"] = Value::Null;
    m["cells"]
        .as_object_mut()
        .unwrap()
        .insert("seventh".into(), extra);
    let bad = problems(&m, Regen::No);
    assert!(
        bad.iter()
            .any(|p| p.contains("cell seventh") && p.contains("not one of the six"))
    );
    assert!(
        bad.iter()
            .any(|p| p.contains("cell seventh") && p.contains("no measured entity count"))
    );
}

#[test]
fn a_cell_with_no_measured_count_turns_it_red() {
    let mut m = manifest();
    m["cells"]["paper_space"]["entities"] = Value::Null;
    assert!(
        problems(&m, Regen::No)
            .iter()
            .any(|p| p.contains("cell paper_space") && p.contains("no measured entity count"))
    );
}

#[test]
fn a_malformed_cell_that_skips_an_n3_6_limit_turns_it_red() {
    let mut m = manifest();
    let inputs = m["cells"]["malformed"]["inputs"].as_array_mut().unwrap();
    inputs.retain(|i| i["limit"] != "max_spline_knots");
    assert!(
        problems(&m, Regen::No)
            .iter()
            .any(|p| p.contains("no input trips max_spline_knots"))
    );
}

#[test]
fn a_malformed_input_the_limit_does_not_refuse_turns_it_red() {
    let mut m = manifest();
    m["cells"]["malformed"]["inputs"][0]["limits"] = json!({});
    m["cells"]["malformed"]["inputs"][0]["params"] = json!({"levels": 3});
    let bad = problems(&m, Regen::Small);
    assert!(
        bad.iter()
            .any(|p| p.contains("malformed") && p.contains("must be refused")),
        "{bad:?}"
    );
}

#[test]
fn the_generators_are_deterministic() {
    let m = manifest();
    let c = cells(&m);
    let simple = &c["simple"];
    let g = simple["generator"].as_str().unwrap();
    assert_eq!(
        sha256_hex(&generate(g, &simple["params"])),
        sha256_hex(&generate(g, &simple["params"]))
    );
    let i = &c["malformed"]["inputs"][0];
    let k = i["kind"].as_str().unwrap();
    assert_eq!(
        generate_malformed(k, &i["params"]),
        generate_malformed(k, &i["params"])
    );
}

/// Prints the manifest rows for the current generators. Not a test of anything: it is the recorded
/// command that produces every `bytes`, `sha256` and `entities` figure in the manifest.
#[test]
#[ignore = "regenerates the manifest numbers, minutes in a debug build"]
fn measure_cells() {
    let m = manifest();
    let dir = tempfile::tempdir().unwrap();
    for (name, c) in cells(&m).iter().filter(|(n, _)| *n != "malformed") {
        let t = std::time::Instant::now();
        let bytes = generate(c["generator"].as_str().unwrap(), &c["params"]);
        let n = count(&bytes, dir.path()).expect("converts");
        println!(
            "ROW {}",
            json!({"cell": name, "bytes": bytes.len(), "sha256": sha256_hex(&bytes), "encountered": n.encountered, "written": n.written, "artifact_bytes": n.artifact_bytes, "secs": t.elapsed().as_secs_f64()})
        );
    }
    for i in cells(&m)["malformed"]["inputs"].as_array().unwrap() {
        let b = generate_malformed(i["kind"].as_str().unwrap(), &i["params"]);
        println!(
            "ROW {}",
            json!({"input": i["id"], "bytes": b.len(), "sha256": sha256_hex(&b)})
        );
    }
}

/// The full pass, large cell included.
#[test]
#[ignore = "regenerates the 10^6 entity cell"]
fn the_manifest_is_a_good_corpus_including_the_large_cell() {
    let bad = problems(&manifest(), Regen::All);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
