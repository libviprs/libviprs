//! Issue #1079 (N4.8/N2.7): the provider matrix runs over the pinned O2.9 corpus, not over a
//! hand-picked pair of drawings, and nothing in a fixture is dropped without a row saying so.
//!
//! Needs `--features "dwg-acadrust pvec-normalized"`.
#![cfg(all(feature = "dwg-acadrust", feature = "pvec-normalized"))]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::path::PathBuf;

use libviprs::pvec::differential::{Deltas, Fixture, Matrix, Verdict};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use pvec_gold::{Corpus, EvidenceClass};

/// Every non-rejection fixture of the pinned corpus that carries a drawing.
fn corpus_fixtures(c: &Corpus) -> Vec<Fixture> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold/fixtures");
    c.all()
        .iter()
        .filter(|f| !f.expects_rejection && f.has("source.dwg"))
        .map(|f| Fixture {
            id: f.id.clone(),
            path: root.join(&f.id).join("source.dwg"),
            evidence: f.class.tag().to_string(),
        })
        .collect()
}

fn matrix(c: &Corpus, only: &[&str]) -> (Matrix, Vec<Fixture>) {
    let fx: Vec<_> = corpus_fixtures(c)
        .into_iter()
        .filter(|f| only.is_empty() || only.contains(&f.id.as_str()))
        .collect();
    let mut m = Matrix::new(c.revision.clone());
    m.run::<AcadrustProvider>(&fx).expect("the corpus runs");
    (m, fx)
}

#[test]
fn the_matrix_covers_every_known_truth_fixture_of_the_pinned_corpus() {
    let c = Corpus::open();
    let (m, fx) = matrix(&c, &[]);
    let known: Vec<_> = c
        .of_class(EvidenceClass::KnownTruthExact)
        .into_iter()
        .filter(|f| !f.expects_rejection)
        .collect();
    assert!(known.len() >= 30, "{}", known.len());
    for f in known {
        assert!(fx.iter().any(|x| x.id == f.id), "{} is not run", f.id);
        assert!(m.rows("acadrust", &f.id).is_some(), "{} has no rows", f.id);
    }
}

#[test]
fn a_corpus_run_with_one_provider_is_partial_and_names_the_corpus_revision() {
    let c = Corpus::open();
    let (mut m, _) = matrix(&c, &["line-001", "circle-001"]);
    m.absent("acadsharp-rs", "no adapter yet (libviprs-dep#161)");
    let none = Deltas::parse("").unwrap();
    assert!(matches!(m.verdict(&none), Verdict::Partial { .. }));
    let r = m.report(&none);
    assert!(r.contains(&format!("corpus={}", c.revision)), "{r}");
    assert!(r.contains("evidence=KNOWN_TRUTH_EXACT"), "{r}");
    assert!(!r.contains("DUAL_ORACLE_EQUIVALENT agreement"), "{r}");
}

#[test]
fn layer_state_rows_all_carry_a_layer_name() {
    let c = Corpus::open();
    let (m, _) = matrix(&c, &["layer-state-001"]);
    let rows = m.rows("acadrust", "layer-state-001").unwrap();
    assert!(!rows.is_empty());
    for r in rows {
        assert!(!r.layer.is_empty() || r.class_name == "INSERT", "{r:?}");
    }
}

/// #1239: a DIMENSION's inserts are drawn or reported, never silently skipped.
#[test]
fn a_dimension_is_drawn_or_reported_and_never_silently_absent() {
    let c = Corpus::open();
    let (m, _) = matrix(&c, &["dimension-001"]);
    let rows = m.rows("acadrust", "dimension-001").unwrap();
    assert!(!rows.is_empty(), "the dimension left no row at all");
    for r in rows {
        let drawn = r.bucket == "SupportedExact" || r.bucket == "SupportedEquivalent";
        assert!(
            drawn || !r.reason.is_empty(),
            "{r:?} is neither drawn nor explained"
        );
    }
}

#[test]
fn every_fixture_row_set_is_the_same_on_a_second_run() {
    let c = Corpus::open();
    let ids = ["nested-blocks-001", "dimension-001", "layer-state-001"];
    let (a, _) = matrix(&c, &ids);
    let (b, _) = matrix(&c, &ids);
    let none = Deltas::parse("").unwrap();
    assert_eq!(a.report(&none), b.report(&none));
}
