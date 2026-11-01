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
        let mut parts = key.split('/');
        let (bench, cell, platform) = (parts.next(), parts.next().unwrap(), parts.next().unwrap());
        assert_eq!(bench, Some("provider"), "{key}");
        assert!(
            doc.contains(&format!("| {cell} |")),
            "{key}: the document has no row for {cell}"
        );
        assert!(
            doc.contains(platform),
            "{key}: the document never names {platform}"
        );
        assert_eq!(row["platform"], platform, "{key}");
        let blank = blank_columns(
            row,
            &[
                "cell",
                "entities",
                "entities_per_s",
                "tracked_peak",
                "peak_rss_bytes",
                "block_frames",
                "platform",
            ],
        );
        assert!(blank.is_empty(), "{key}: {blank:?}");
    }
}
