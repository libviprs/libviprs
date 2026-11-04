//! N8.4a (#1515, slice of #1481): the conversion peak is split by stage, and what opening the
//! drawing alone costs is a figure in the baseline.
//!
//! These read the committed document and baseline, so they are red until the stage rows have been
//! taken on the native x86_64 host and written beside the figures.

use std::path::PathBuf;

const STAGES: [&str; 6] = ["open", "view", "flatten", "classify", "write", "report"];
const SCALES: [&str; 3] = ["0.5", "1.0", "2.0"];

fn read(path: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn baseline() -> serde_json::Value {
    serde_json::from_str(&read("tests/fixtures/pvec_bench_baseline.json"))
        .expect("the baseline parses")
}

fn stage_key(scale: &str) -> String {
    format!("bounded_memory_stages/direct/large_industrial/x{scale}/x86_64-linux-mars")
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_baseline_has_a_stage_row_per_scale_with_every_stage() {
    let b = baseline();
    for scale in SCALES {
        let key = stage_key(scale);
        let r = &b["rows"][&key];
        assert!(r.is_object(), "{key} is in the baseline");
        assert!(
            r["entities"].as_u64().is_some_and(|e| e > 0),
            "{key}: entities"
        );
        assert!(r["tracked_peak"].as_u64().is_some(), "{key}: tracked_peak");
        for stage in STAGES {
            for col in ["held", "peak"] {
                assert!(
                    r["stages"][stage][col].as_u64().is_some(),
                    "{key}: stages.{stage}.{col}"
                );
            }
        }
    }
}

/// A stage cannot end holding more than the most it ever held, and the whole run's tracked peak is
/// the largest stage peak, so a stage row that disagrees with its own total was not taken from one run.
#[test]
#[cfg_attr(miri, ignore)]
fn the_stage_rows_are_consistent_with_themselves() {
    let b = baseline();
    for scale in SCALES {
        let key = stage_key(scale);
        let r = &b["rows"][&key];
        let mut top = 0;
        for stage in STAGES {
            let (held, peak) = (
                r["stages"][stage]["held"].as_u64().unwrap_or(u64::MAX),
                r["stages"][stage]["peak"].as_u64().unwrap_or(0),
            );
            assert!(
                held <= peak,
                "{key}: {stage} holds {held} but peaked at {peak}"
            );
            top = top.max(peak);
        }
        assert_eq!(
            r["tracked_peak"].as_u64(),
            Some(top),
            "{key}: the tracked peak is the largest stage peak"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_provider_rows_carry_what_opening_alone_costs() {
    let b = baseline();
    for cell in ["simple", "block_heavy", "spline_heavy", "large_industrial"] {
        let key = format!("provider/{cell}/x86_64-linux-mars");
        let r = &b["rows"][&key];
        for col in ["open_only_held", "open_only_tracked_peak"] {
            assert!(r[col].as_u64().is_some(), "{key} has {col}");
        }
    }
}

fn stage_section(doc: &str) -> &str {
    let start = doc
        .find("### Peak by stage")
        .expect("a Peak by stage heading");
    let rest = &doc[start + 1..];
    &doc[start..rest.find("\n#").map_or(doc.len(), |i| start + 1 + i)]
}

/// The rows in the document are the baseline's, rounded up to the MiB, so the page and the file
/// cannot drift apart by a rerun that only touched one of them.
#[test]
#[cfg_attr(miri, ignore)]
fn the_stage_table_agrees_with_the_baseline() {
    let b = baseline();
    let doc = read("docs/pvec-benchmarks.md");
    let up = |bytes: u64| bytes.div_ceil(1_048_576).to_string();
    let mut checked = 0;
    for line in stage_section(&doc)
        .lines()
        .filter(|l| l.starts_with("| stage |"))
    {
        let c: Vec<&str> = line.split('|').map(str::trim).collect();
        let key = stage_key(c[3]);
        let r = &b["rows"][&key];
        assert!(r.is_object(), "{key} is not in the baseline");
        let s = &r["stages"][c[4]];
        let (held, peak) = (s["held"].as_u64().unwrap(), s["peak"].as_u64().unwrap());
        let per_entity = (held as f64 / r["entities"].as_f64().unwrap()).round();
        assert_eq!(
            [
                c[5].replace(',', ""),
                c[6].replace(',', ""),
                c[7].replace(',', "")
            ],
            [up(held), up(peak), format!("{per_entity}")],
            "{key} {}: the document and the baseline disagree",
            c[4]
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        SCALES.len() * STAGES.len(),
        "a row per scale and stage"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_document_says_what_the_open_alone_costs() {
    let doc = read("docs/pvec-benchmarks.md");
    let s = stage_section(&doc);
    assert!(
        s.contains("opening the drawing alone"),
        "the open share is stated in words"
    );
    assert!(s.contains("MiB"), "the share cites the measured figure");
}
