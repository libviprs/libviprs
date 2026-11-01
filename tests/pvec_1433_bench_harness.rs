//! N8.2 (#1433, slice of #1064): the plumbing the provider benchmark stands on, tested without
//! taking a measurement. The measuring itself is `tests/pvec_provider_bench.rs`, `#[ignore]`d.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use pvec_bench::{Stat, blank_columns, provider_regressions, quiet_after};
use serde_json::json;

#[test]
fn a_stat_is_the_median_with_the_extremes() {
    let s = Stat::of(&[3.0, 1.0, 2.0]);
    assert_eq!((s.median, s.min, s.max), (2.0, 1.0, 3.0));
    assert_eq!(Stat::of(&[5.0]).spread(), 0.0);
}

#[test]
fn a_spread_over_twenty_percent_is_unstable() {
    assert!(!Stat::of(&[0.92, 1.0, 1.08]).unstable());
    assert!(Stat::of(&[0.80, 1.0, 1.30]).unstable());
}

#[test]
#[should_panic]
fn no_samples_is_a_panic_not_a_zero() {
    let _ = Stat::of(&[]);
}

#[test]
fn quiet_needs_three_polls_in_a_row_under_the_cpu_count() {
    assert_eq!(quiet_after([1.0, 2.0, 3.0], 28), Some(3));
    assert_eq!(
        quiet_after([30.0, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0], 28),
        Some(7)
    );
    assert_eq!(
        quiet_after([1.0, 1.0, 28.0, 1.0, 1.0], 28),
        None,
        "28 is not under 28"
    );
    assert_eq!(quiet_after([], 28), None);
}

#[test]
fn a_blank_column_is_named() {
    let row = json!({"a": 1, "b": null});
    assert_eq!(blank_columns(&row, &["a", "b", "c"]), vec!["b", "c"]);
    assert!(blank_columns(&row, &["a"]).is_empty());
}

fn row(eps: f64, peak: u64, unstable: bool) -> serde_json::Value {
    json!({"entities_per_s": {"median": eps, "min": eps, "max": eps}, "tracked_peak": peak, "unstable": unstable})
}

#[test]
fn ten_percent_less_throughput_or_fifteen_percent_more_peak_is_a_regression() {
    let base = row(1000.0, 1000, false);
    assert!(provider_regressions(&base, &row(905.0, 1140, false)).is_empty());
    assert_eq!(
        provider_regressions(&base, &row(890.0, 1000, false)).len(),
        1
    );
    assert_eq!(
        provider_regressions(&base, &row(1000.0, 1160, false)).len(),
        1
    );
    assert_eq!(
        provider_regressions(&base, &row(800.0, 2000, false)).len(),
        2
    );
}

#[test]
fn an_unstable_row_never_gates() {
    let base = row(1000.0, 1000, false);
    assert!(provider_regressions(&base, &row(10.0, 9999, true)).is_empty());
    assert!(provider_regressions(&row(1000.0, 1000, true), &row(10.0, 9999, false)).is_empty());
}

/// The document names every baseline row's cell and the platform it ran on, and the baseline's
/// rows each carry the columns the harness refuses to print blank.
#[test]
#[cfg_attr(miri, ignore)]
fn the_document_and_the_baseline_describe_the_same_rows() {
    let doc = std::fs::read_to_string(pvec_bench::root().join("docs/pvec-benchmarks.md")).unwrap();
    let rows = pvec_bench::baseline_rows();
    let rows = rows.as_object().expect("the baseline has rows");
    assert!(!rows.is_empty(), "no baseline row is committed");
    for (key, row) in rows {
        let parts: Vec<&str> = key.split('/').collect();
        let platform = *parts.last().unwrap();
        // The word each kind of row is found under in the document's tables.
        let name = match parts[..] {
            ["provider", cell, _] | ["normalization", cell, _] => cell,
            ["normalization", "transform", class, _] => class,
            ["normalization", cell, _mix, _] => cell,
            _ => panic!("{key}: a baseline key this suite does not know"),
        };
        assert!(
            doc.contains(&format!("| {name} |")) || doc.contains(&format!("`{name}`")),
            "{key}: the document has no row for {name}"
        );
        assert!(
            doc.contains(platform),
            "{key}: the document never names {platform}"
        );
        assert_eq!(row["platform"], platform, "{key}");
        let required: &[&str] = if parts[0] == "provider" {
            &[
                "cell",
                "entities",
                "entities_per_s",
                "tracked_peak",
                "peak_rss_bytes",
                "block_frames",
                "platform",
            ]
        } else {
            &["cell", "stage_ns", "platform"]
        };
        let blank = blank_columns(row, required);
        assert!(blank.is_empty(), "{key}: {blank:?}");
    }
}

#[test]
fn a_stage_that_grew_ten_percent_is_a_regression() {
    let row = |a: f64, b: f64, unstable: bool| json!({"stage_ns": {"visibility": a, "bounds": b}, "unstable": unstable});
    let base = row(100.0, 50.0, false);
    assert!(pvec_bench::stage_regressions(&base, &row(109.0, 54.0, false)).is_empty());
    let bad = pvec_bench::stage_regressions(&base, &row(111.0, 50.0, false));
    assert_eq!(bad.len(), 1);
    assert!(bad[0].contains("visibility"), "{bad:?}");
    assert_eq!(
        pvec_bench::stage_regressions(&base, &row(200.0, 100.0, false)).len(),
        2
    );
    assert!(pvec_bench::stage_regressions(&base, &row(200.0, 100.0, true)).is_empty());
}

#[test]
fn the_unattributed_remainder_is_what_no_stage_accounts_for() {
    let r = pvec_bench::unattributed(&[0.3, 0.5], 1.0);
    assert!((r - 0.2).abs() < 1e-12, "{r}");
    assert!(pvec_bench::unattributed(&[0.6, 0.5], 1.0) < 0.0);
}

#[test]
fn generation_regressions_are_ten_percent_time_five_percent_bytes_two_points_of_ratio() {
    let row = |wall: f64, bytes: f64, pct: f64, unstable: bool| json!({"wall_secs": {"median": wall}, "artifact_bytes": bytes, "artifact_pct_of_source": pct, "unstable": unstable});
    let base = row(1.0, 1000.0, 20.0, false);
    assert!(pvec_bench::generation_regressions(&base, &row(1.09, 1049.0, 21.9, false)).is_empty());
    assert_eq!(
        pvec_bench::generation_regressions(&base, &row(1.2, 1000.0, 20.0, false)).len(),
        1
    );
    assert_eq!(
        pvec_bench::generation_regressions(&base, &row(1.0, 1060.0, 20.0, false)).len(),
        1
    );
    assert_eq!(
        pvec_bench::generation_regressions(&base, &row(1.0, 1000.0, 22.5, false)).len(),
        1
    );
    // A noisy wall time does not gate, the deterministic bytes still do.
    let noisy = pvec_bench::generation_regressions(&base, &row(9.0, 2000.0, 20.0, true));
    assert_eq!(noisy.len(), 1, "{noisy:?}");
}
