//! N7.8 (#1444, slice of #1070): the corpus regression report and its gate.
//!
//! Every fixture of the pinned `pvec-gold-fixtures` revision is converted through
//! `PvecConversion`, and three sections are recorded per fixture: the mathematical one (the
//! conversion classes, with the evidence class printed beside the row, and a
//! `DUAL_ORACLE_EQUIVALENT` row labelled as converter agreement and never as source truth), the
//! view policy one (what the profile kept, what it excluded and why) and the size one (entities,
//! artifact bytes). Wall time and peak RSS, which differ from run to run, are in the ignored
//! report only.
//!
//! The deterministic columns are `tests/fixtures/pvec_corpus_baseline.json`, and the diff against
//! it runs in the ordinary suite (`the_corpus_does_not_regress`). Regenerating it, and printing
//! the report with the timing columns, is one command:
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars LIBVIPRS_CORPUS_BASELINE_OUT=tests/fixtures/pvec_corpus_baseline.json \
//!     cargo test --release --features pvec-convert --test pvec_corpus_regression -- \
//!     --ignored --nocapture corpus_report
//! ```
//!
//! Nothing here runs a converter, needs one, or touches LibreDWG: the corpus is read from the
//! repository.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;
#[path = "common/pvec_corpus.rs"]
mod pvec_corpus;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::time::Instant;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::fidelity::FidelityClass;
use libviprs::pvec_convert::PvecConversion;
use pvec_corpus::{conformance, document, regressions};
use pvec_gold::{Corpus, EvidenceClass, Fixture};
use serde_json::{Value, json};

const BASELINE: &str = "tests/fixtures/pvec_corpus_baseline.json";

/// One fixture's row and, for the report, how long it took.
fn row(fx: &Fixture, dir: &std::path::Path) -> (Value, f64) {
    let bytes = fx.dwg();
    let dest = dir.join(format!("{}.pvec", fx.id));
    let t = Instant::now();
    let run = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert();
    let secs = t.elapsed().as_secs_f64();
    let head = json!({
        "fixture": fx.id,
        "group": fx.group,
        "evidence": fx.class.tag(),
        "evidence_scope": fx.scope.tag(),
        // A converter agreement is not source truth, and the report says which this row is.
        "what_the_class_means": match fx.class {
            EvidenceClass::KnownTruthExact => "independently authored expectation",
            EvidenceClass::DualOracleEquivalent => "converter agreement, not source truth",
            EvidenceClass::RealWorldDifferential => "regression only, no expectation",
            EvidenceClass::Quarantined => "runs and reports, never gates",
        },
    });
    let mut r = head.as_object().unwrap().clone();
    match run {
        Ok(out) => {
            let by = serde_json::to_value(&out.report.totals.by_fidelity).unwrap();
            let c = conformance(by.as_object().unwrap());
            r.insert("outcome".into(), json!("converted"));
            r.insert(
                "mathematical".into(),
                json!({"by_fidelity": by, "conformance": c}),
            );
            r.insert(
                "view".into(),
                json!({
                    "space": out.report.view.space,
                    "included": out.report.view.included,
                    "excluded_by_field": out.report.view.excluded_by_field,
                    "xrefs_unresolved": out.report.xrefs.unresolved.len(),
                }),
            );
            r.insert(
                "size".into(),
                json!({
                    "entities": out.report.totals.encountered,
                    "written": out.written,
                    "artifact_bytes": std::fs::metadata(&out.artifact).map_or(0, |m| m.len()),
                }),
            );
        }
        Err(e) => {
            // Nothing was written, but the classes are still the report: a probe counts them.
            let probe = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
                .unit(SourceUnit::Meters)
                .probe();
            let (by, reasons, met) = match &probe {
                Ok(f) => (
                    FidelityClass::ALL
                        .iter()
                        .map(|c| (c.tag().to_string(), json!(f.count(*c))))
                        .collect::<serde_json::Map<_, _>>(),
                    f.reasons()
                        .map(|(r, class, n)| {
                            json!({"reason": r.code(), "source_class": class, "count": n})
                        })
                        .collect::<Vec<_>>(),
                    f.encountered(),
                ),
                Err(_) => (serde_json::Map::new(), Vec::new(), 0),
            };
            let mut why = format!("refused: {e}");
            let mut source = std::error::Error::source(&e);
            while let Some(cause) = source {
                why += &format!(" <- {cause}");
                source = cause.source();
            }
            r.insert("outcome".into(), json!(why));
            r.insert(
                "mathematical".into(),
                json!({"by_fidelity": by, "conformance": "NoGeometry", "reasons": reasons}),
            );
            r.insert("view".into(), json!({}));
            r.insert(
                "size".into(),
                json!({"entities": met, "written": 0, "artifact_bytes": 0}),
            );
        }
    }
    (Value::Object(r), secs)
}

fn current() -> (Value, Vec<f64>) {
    let corpus = Corpus::open();
    let dir = tempfile::tempdir().unwrap();
    let mut rows = Vec::new();
    let mut secs = Vec::new();
    for fx in corpus.all() {
        let (r, s) = row(fx, dir.path());
        rows.push((fx.id.clone(), r));
        secs.push(s);
    }
    (document(&corpus.revision, rows), secs)
}

fn baseline() -> Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(BASELINE);
    serde_json::from_str(
        &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{BASELINE} is missing: {e}")),
    )
    .unwrap_or_else(|e| panic!("{BASELINE} is not JSON: {e}"))
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_corpus_does_not_regress() {
    let (now, _) = current();
    let bad = regressions(&baseline(), &now);
    assert!(bad.is_empty(), "{BASELINE}:\n{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_baseline_has_one_row_per_fixture_and_no_empty_cell() {
    let b = baseline();
    let corpus = Corpus::open();
    assert_eq!(b["corpus_revision"], corpus.revision.as_str());
    let rows = b["rows"].as_object().unwrap();
    assert_eq!(rows.len(), corpus.all().len());
    for (id, r) in rows {
        for section in ["mathematical", "view", "size"] {
            assert!(
                r.get(section).is_some_and(|v| !v.is_null()),
                "{id}: no {section} section"
            );
        }
        assert!(
            r["evidence"].is_string() && r["outcome"].is_string(),
            "{id}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_fixture_that_moves_from_exact_to_equivalent_fails_the_gate() {
    let mut cur = baseline();
    let id = cur["rows"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, r)| {
            r["mathematical"]["conformance"] == "SupportedExact" && r["evidence"] != "QUARANTINED"
        })
        .map(|(id, _)| id.clone())
        .expect("a gating fixture that is SupportedExact");
    cur["rows"][&id]["mathematical"]["conformance"] = json!("SupportedEquivalent");
    let bad = regressions(&baseline(), &cur);
    assert!(
        bad.iter()
            .any(|p| p.contains(&id)
                && p.contains("moved from SupportedExact to SupportedEquivalent")),
        "{bad:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_quarantined_fixture_is_in_the_report_and_never_gates() {
    let mut cur = baseline();
    let ids: Vec<String> = cur["rows"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, r)| r["evidence"] == "QUARANTINED")
        .map(|(id, _)| id.clone())
        .collect();
    assert!(!ids.is_empty(), "the corpus has quarantined fixtures");
    for id in &ids {
        cur["rows"][id]["mathematical"]["conformance"] = json!("NoGeometry");
        cur["rows"][id]["size"]["artifact_bytes"] = json!(u64::MAX / 2);
    }
    assert!(regressions(&baseline(), &cur).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn artifact_growth_of_two_percent_in_total_or_five_on_one_fixture_fails() {
    let base = baseline();
    let gating: Vec<String> = base["rows"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, r)| {
            r["evidence"] != "QUARANTINED"
                && r["size"]["artifact_bytes"].as_u64().unwrap_or(0) > 300
        })
        .map(|(id, _)| id.clone())
        .collect();
    let mut one = base.clone();
    let big = &gating[0];
    let bytes = base["rows"][big]["size"]["artifact_bytes"]
        .as_f64()
        .unwrap();
    one["rows"][big]["size"]["artifact_bytes"] = json!((bytes * 1.06) as u64);
    assert!(
        regressions(&base, &one)
            .iter()
            .any(|p| p.contains(big) && p.contains("5%"))
    );
    let mut all = base.clone();
    for id in &gating {
        let b = base["rows"][id]["size"]["artifact_bytes"].as_f64().unwrap();
        all["rows"][id]["size"]["artifact_bytes"] = json!((b * 1.04) as u64);
    }
    let bad = regressions(&base, &all);
    assert!(
        bad.iter().any(|p| p.contains("total") && p.contains("2%")),
        "{bad:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_dual_oracle_row_is_labelled_as_agreement_and_never_as_truth() {
    let b = baseline();
    for (id, r) in b["rows"].as_object().unwrap() {
        if r["evidence"] == "DUAL_ORACLE_EQUIVALENT" {
            assert_eq!(
                r["what_the_class_means"], "converter agreement, not source truth",
                "{id}"
            );
        }
        assert!(
            !r["what_the_class_means"]
                .as_str()
                .unwrap()
                .contains("exact"),
            "{id}"
        );
    }
}

/// The report: all three sections with the timing columns, tagged with the platform. With
/// `LIBVIPRS_CORPUS_BASELINE_OUT` it also writes the deterministic baseline.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "prints the report and regenerates the baseline"]
fn corpus_report() {
    let platform = pvec_bench::platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let (now, secs) = current();
    for ((id, r), s) in now["rows"].as_object().unwrap().iter().zip(&secs) {
        println!(
            "| {id} | {} | {} | {} | {} | {} | {:.1} | {platform} |",
            r["evidence"].as_str().unwrap(),
            r["mathematical"]["conformance"].as_str().unwrap(),
            r["view"]["included"],
            r["size"]["entities"],
            r["size"]["artifact_bytes"],
            s * 1000.0
        );
    }
    println!("PEAK_RSS {:?}", pvec_bench::peak_rss());
    if let Ok(out) = std::env::var("LIBVIPRS_CORPUS_BASELINE_OUT") {
        std::fs::write(&out, serde_json::to_string_pretty(&now).unwrap() + "\n").unwrap();
    }
}

/// The document lists every fixture the corpus has, and the command that regenerates the baseline.
#[test]
#[cfg_attr(miri, ignore)]
fn the_document_has_a_row_for_every_fixture() {
    let doc = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/pvec-corpus-regression.md"),
    )
    .expect("docs/pvec-corpus-regression.md");
    for fx in Corpus::open().all() {
        assert!(
            doc.contains(&format!("| {} |", fx.id)),
            "the document has no row for {}",
            fx.id
        );
    }
    assert!(doc.contains("LIBVIPRS_CORPUS_BASELINE_OUT=tests/fixtures/pvec_corpus_baseline.json"));
}
