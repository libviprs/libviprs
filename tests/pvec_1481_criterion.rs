//! N8.4 close-out (#1481): the memory criterion is restated to what the conversion does, and the
//! page, the baseline and the contract all say so with the same numbers.
//!
//! These read the committed document, baseline and bench source, so they are red until the
//! three-scale rows have been taken after the stack and the text is written beside them.

use std::path::PathBuf;

use serde_json::Value;

const GIB: f64 = (1u64 << 30) as f64;
const ARM: &str = "aarch64-linux-mercury";
const NAS: &str = "x86_64-linux-nas";
const SCALES: [&str; 3] = ["0.5", "1.0", "2.0"];

fn read(path: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn rows() -> Value {
    let b: Value = serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json")).unwrap();
    b["rows"].clone()
}

fn direct(scale: &str) -> Value {
    let key = format!("bounded_memory/direct/large_industrial/x{scale}/{ARM}");
    let r = rows()[&key].clone();
    assert!(!r.is_null(), "no baseline row {key}");
    r
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_restated_criterion_holds_on_the_three_scale_rows() {
    let mut per_entity = Vec::new();
    for scale in SCALES {
        let r = direct(scale);
        let (tracked, entities) = (
            r["tracked_peak"].as_f64().unwrap(),
            r["entities"].as_f64().unwrap(),
        );
        per_entity.push(tracked / entities);
        assert!(
            r["peak_rss_bytes"].as_u64().is_some(),
            "x{scale} has no RSS"
        );
        assert!(
            tracked / entities <= 1450.0,
            "x{scale}: {}",
            tracked / entities
        );
    }
    let one = direct("1.0");
    assert_eq!(one["entities"], 1_025_000);
    assert!(one["tracked_peak"].as_f64().unwrap() <= 1.4 * GIB);
    assert!(one["peak_rss_bytes"].as_f64().unwrap() <= 1.75 * GIB);
    let (lo, hi) = per_entity
        .iter()
        .fold((f64::MAX, 0f64), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert!(
        hi <= lo * 1.10,
        "the per-entity figures differ by more than 10%: {per_entity:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_provider_document_alone_is_measured_at_the_three_scales() {
    for scale in SCALES {
        let key = format!("bounded_memory_stages/direct/large_industrial/x{scale}/{NAS}");
        let r = rows()[&key].clone();
        assert!(!r.is_null(), "no baseline row {key}");
        let per = r["stages"]["open"]["held"].as_f64().unwrap() / r["entities"].as_f64().unwrap();
        assert!(
            (420.0..=430.0).contains(&per),
            "x{scale}: {per} bytes an entity"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_bench_checks_the_restated_ceilings() {
    let src = read("tests/pvec_bounded_memory_bench.rs");
    assert!(src.contains("const RSS_CEILING: u64 = 7 << 28;"));
    assert!(src.contains("const TRACKED_CEILING: u64 = 1_503_238_554;"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_page_withdraws_the_original_claim_and_states_the_new_one() {
    let doc = read("docs/pvec-benchmarks.md");
    let at = doc
        .find("## Bounded-memory generation")
        .expect("the section");
    let s = &doc[at..];
    for needed in [
        "were not met",
        "withdrawn",
        "1.4 GiB",
        "1.75 GiB",
        "1,450 bytes",
        "within 10%",
        "### Not done",
        "report_rows(false)",
        "killed by the operating system",
        "no guard",
        "424",
        "1,323 MiB",
    ] {
        assert!(s.contains(needed), "the page does not say {needed:?}");
    }
    assert!(!s.contains("That work is tracked in #1481."));
}
