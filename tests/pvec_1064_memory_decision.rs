//! N8.4 (slice of #1064): the strategy decision is written with its numbers, and the spill
//! strategy is a row in the committed baseline.
//!
//! These read the committed document and baseline, so they are red until the spill strategy has
//! been measured on the native x86_64 host and the decision written beside the figures.

use std::path::PathBuf;

fn read(path: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn section(doc: &str) -> &str {
    let start = doc
        .find("## Bounded-memory generation")
        .expect("the bounded-memory section");
    let rest = &doc[start + 1..];
    &doc[start..rest.find("\n## ").map_or(doc.len(), |i| start + 1 + i)]
}

#[test]
fn the_spill_strategy_is_measured_not_marked_unbuilt() {
    let s = section(&read("docs/pvec-benchmarks.md")).to_string();
    assert!(
        !s.contains("not built"),
        "the spill is built (pvec#235), so no row may say it is not"
    );
    assert!(
        s.contains("| spill |"),
        "a spill row per cell is in the table"
    );
}

#[test]
fn the_decision_names_a_strategy_and_cites_figures() {
    let s = section(&read("docs/pvec-benchmarks.md")).to_string();
    let at = s.find("### Decision").expect("a Decision heading");
    let d = &s[at..];
    assert!(
        d.contains("libviprs ships"),
        "the decision says which strategy libviprs ships"
    );
    assert!(d.contains("MiB"), "the decision cites the measured figures");
    assert!(
        d.contains("two-pass"),
        "the decision says why the two-pass strategy is not built"
    );
}

#[test]
fn the_baseline_has_spill_rows_for_every_cell_from_the_native_host() {
    let b: serde_json::Value =
        serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json"))
            .expect("the baseline parses");
    for cell in ["large_industrial", "spline_heavy", "block_heavy"] {
        let key = format!("bounded_memory/spill/{cell}/x1.0/x86_64-linux-mars");
        let r = &b["rows"][&key];
        assert!(r.is_object(), "{key} is in the baseline");
        for col in [
            "bytes_spilled",
            "bytes_reread",
            "spill_count",
            "spill_budget",
        ] {
            assert!(!r[col].is_null(), "{key} has {col}");
        }
    }
}

#[test]
fn the_budget_sweep_is_recorded() {
    let b: serde_json::Value =
        serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json"))
            .expect("the baseline parses");
    for budget in [64u64, 128] {
        let key =
            format!("bounded_memory/spill_budget/large_industrial/b{budget}MiB/x86_64-linux-mars");
        assert!(b["rows"][&key].is_object(), "{key} is in the baseline");
    }
}
