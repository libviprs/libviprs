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
#[ignore = "benchmark plumbing, opt-in: cargo test --test pvec_1064_memory_decision -- --ignored"]
#[cfg_attr(miri, ignore)]
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
#[ignore = "benchmark plumbing, opt-in: cargo test --test pvec_1064_memory_decision -- --ignored"]
#[cfg_attr(miri, ignore)]
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
#[ignore = "benchmark plumbing, opt-in: cargo test --test pvec_1064_memory_decision -- --ignored"]
#[cfg_attr(miri, ignore)]
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
#[ignore = "benchmark plumbing, opt-in: cargo test --test pvec_1064_memory_decision -- --ignored"]
#[cfg_attr(miri, ignore)]
fn the_budget_sweep_is_recorded() {
    let b: serde_json::Value =
        serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json"))
            .expect("the baseline parses");
    for budget in [64u64, 128, 256] {
        let key =
            format!("bounded_memory/spill_budget/large_industrial/b{budget}MiB/x86_64-linux-mars");
        assert!(b["rows"][&key].is_object(), "{key} is in the baseline");
    }
}

/// The spill rows in the document are the baseline's, rounded up to the MiB, so the page and the
/// file cannot drift apart by a rerun that only touched one of them.
#[test]
#[ignore = "benchmark plumbing, opt-in: cargo test --test pvec_1064_memory_decision -- --ignored"]
#[cfg_attr(miri, ignore)]
fn the_spill_table_agrees_with_the_baseline() {
    let b: serde_json::Value =
        serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json"))
            .expect("the baseline parses");
    let doc = read("docs/pvec-benchmarks.md");
    let up = |bytes: &serde_json::Value| {
        ((bytes.as_f64().unwrap() / 1_048_576.0).ceil() as u64).to_string()
    };
    let mut checked = 0;
    for line in section(&doc).lines().filter(|l| l.starts_with("| spill |")) {
        let c: Vec<&str> = line.split('|').map(str::trim).collect();
        let key = format!("bounded_memory/spill/{}/x{}/x86_64-linux-mars", c[2], c[3]);
        let r = &b["rows"][&key];
        assert!(r.is_object(), "{key} is not in the baseline");
        let want = [
            up(&r["tracked_peak"]),
            up(&r["peak_rss_bytes"]),
            up(&r["bytes_spilled"]),
            r["spill_count"].to_string(),
        ];
        let got = [
            c[6].replace(',', ""),
            c[7].replace(',', ""),
            c[8].replace(',', ""),
            c[9].to_string(),
        ];
        assert_eq!(got, want, "{key}: the document and the baseline disagree");
        checked += 1;
    }
    assert!(checked >= 5, "the spill table has its rows ({checked})");
}
