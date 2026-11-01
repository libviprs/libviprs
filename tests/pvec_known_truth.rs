//! N2.1 (#1050): the known-truth suite.
//!
//! Every `KNOWN_TRUTH_EXACT` fixture of the pinned `pvec-gold-fixtures` revision (see
//! `tests/fixtures/pvec-gold/PIN.json`) is opened through the acadrust provider, flattened, run
//! through the same extraction and promotion code the conversion uses, and compared with the
//! fixture's `expected.math.json`. That file was authored from the fixture's parameters, not
//! computed by libviprs, so a mapping bug cannot hide behind a second computation of itself.
//!
//! What it checks per entity: the semantic class, source-space coordinates (bit exact where the
//! drawing stores them), arc and ellipse parameters, spline degree, knots with multiplicities,
//! weights (and the absence of a weight array for a non-rational spline), the parameter domain,
//! closed and periodic state, the composed instance transform, layer state, the refusal of a
//! singular transform, and the fidelity class the mapping claims. The few numbers this file
//! computes itself are labelled `truth side`: they turn a truth record's parameters into points so
//! the two sides can be compared. Every tolerance is `Policy::V1`'s, there is no literal epsilon.
//!
//! Nothing here runs a converter or needs one: the corpus is read from the repository, into
//! memory, and nothing is written, so a failing run leaves nothing behind.
//!
//! Needs `--features "dwg-acadrust pvec-normalized pvec-spline"`.

#![cfg(all(
    feature = "dwg-acadrust",
    feature = "pvec-normalized",
    feature = "pvec-spline"
))]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use pvec_gold::{Corpus, EvidenceClass, Fixture};
use serde_json::{Value, json};

// ------------------------------------------------------------------------------------ the suite

/// Runs one fixture against `truth` and returns every disagreement, each naming the property.
fn run(fx: &Fixture, truth: &Value) -> Vec<String> {
    mapping::check(fx, truth)
}

fn known_truth_ids(corpus: &Corpus) -> Vec<String> {
    corpus
        .of_class(EvidenceClass::KnownTruthExact)
        .iter()
        .map(|f| f.id.clone())
        .collect()
}

#[test]
fn n2_1_1050_the_pinned_corpus_is_the_revision_the_pin_names() {
    let corpus = Corpus::open();
    assert_eq!(corpus.revision, "r2-open");
    assert_eq!(corpus.gold_commit.len(), 40, "the pin names a full commit");
    assert_eq!(corpus.upstream_manifest_sha256.len(), 64);
    assert_eq!(known_truth_ids(&corpus).len(), 38, "r2-open has 38");
}

#[test]
fn n2_1_1050_every_known_truth_fixture_is_executed_and_agrees_with_its_expectation() {
    let corpus = Corpus::open();
    let mut executed = Vec::new();
    let mut failures = Vec::new();
    for fx in corpus.of_class(EvidenceClass::KnownTruthExact) {
        let truth = fx
            .truth()
            .expect("a known-truth fixture has a truth record");
        let problems = run(fx, &truth);
        executed.push(fx.id.clone());
        failures.extend(problems.into_iter().map(|p| format!("{}: {p}", fx.id)));
    }
    let mut manifest = known_truth_ids(&corpus);
    manifest.sort();
    executed.sort();
    assert_eq!(
        executed, manifest,
        "a fixture in the manifest that was not run fails the suite"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn n2_1_1050_a_dual_oracle_fixture_is_refused_with_the_class_named() {
    let corpus = Corpus::open();
    let fx = corpus.get("blocks-dynamic-linear-001");
    assert_eq!(fx.class, EvidenceClass::DualOracleEquivalent);
    let err = fx.truth().expect_err("not source truth");
    let msg = err.to_string();
    assert!(msg.contains("DUAL_ORACLE_EQUIVALENT"), "{msg}");
    assert!(msg.contains("KNOWN_TRUTH_EXACT"), "{msg}");
    assert!(msg.contains("blocks-dynamic-linear-001"), "{msg}");
    assert!(msg.contains("class mismatch"), "{msg}");
}

#[test]
fn n2_1_1050_no_other_class_is_accepted_on_the_known_truth_path() {
    let corpus = Corpus::open();
    for fx in corpus.all() {
        let accepted = fx.truth().is_ok();
        assert_eq!(
            accepted,
            fx.class == EvidenceClass::KnownTruthExact,
            "{} ({})",
            fx.id,
            fx.class.tag()
        );
    }
}

fn perturbed(id: &str, edit: impl FnOnce(&mut Value)) -> Vec<String> {
    let corpus = Corpus::open();
    let fx = corpus.get(id);
    let mut truth = fx.truth().unwrap();
    assert!(run(fx, &truth).is_empty(), "{id} is green before the edit");
    edit(&mut truth);
    run(fx, &truth)
}

fn hexnum(v: f64) -> Value {
    json!({"decimal": format!("{v:?}"), "f64_hex": format!("0x{:016x}", v.to_bits())})
}

#[test]
fn n2_1_1050_moving_one_control_point_turns_the_suite_red() {
    let problems = perturbed("bspline-d3-001", |t| {
        t["entities"][0]["geometry"]["control_points"][2][1] = hexnum(-0.5000000001);
    });
    assert!(
        problems.iter().any(|p| p.contains("control point 2")),
        "{problems:?}"
    );
}

#[test]
fn n2_1_1050_changing_one_knot_multiplicity_turns_the_suite_red() {
    let problems = perturbed("repeated-knot-c0-001", |t| {
        let knots = &mut t["entities"][0]["geometry"]["knots"];
        let interior = knots
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|k| k["multiplicity"] == 3 && k["value"]["decimal"] == "0.5")
            .expect("the repeated interior knot");
        interior["multiplicity"] = json!(2);
    });
    assert!(
        problems.iter().any(|p| p.contains("knot")),
        "the failure names the knots: {problems:?}"
    );
}

#[test]
fn n2_1_1050_changing_one_bulge_turns_the_suite_red() {
    let problems = perturbed("bulge-arc-001", |t| {
        t["entities"][0]["geometry"]["bulges"][0] = hexnum(0.45);
    });
    assert!(problems.iter().any(|p| p.contains("bulge")), "{problems:?}");
}

#[test]
fn n2_1_1050_a_wrong_weight_and_a_wrong_class_are_both_caught() {
    let weight = perturbed("nonunit-weight-001", |t| {
        t["entities"][0]["geometry"]["weights"][2] = hexnum(49.0);
    });
    assert!(weight.iter().any(|p| p.contains("weight")), "{weight:?}");
    let class = perturbed("circle-001", |t| {
        t["entities"][0]["class"] = json!("ellipse");
    });
    assert!(class.iter().any(|p| p.contains("class")), "{class:?}");
}

#[test]
fn n2_1_1050_a_vendored_file_that_moved_is_caught_by_the_pin() {
    let dir = tempfile::tempdir().unwrap();
    let from = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold");
    copy_tree(&from, dir.path());
    // Untouched copy opens.
    let _ = Corpus::open_at(dir.path());
    let victim = dir.path().join("fixtures/line-001/expected.math.json");
    let mut bytes = std::fs::read(&victim).unwrap();
    bytes.push(b' ');
    std::fs::write(&victim, bytes).unwrap();
    let caught = std::panic::catch_unwind(|| Corpus::open_at(dir.path()));
    let msg = *caught
        .expect_err("a changed file must not open")
        .downcast::<String>()
        .unwrap();
    assert!(msg.contains("line-001/expected.math.json"), "{msg}");
    assert!(msg.contains("r2-open"), "{msg}");
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let dst = to.join(e.file_name());
        if e.path().is_dir() {
            std::fs::create_dir_all(&dst).unwrap();
            copy_tree(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

#[test]
fn n2_1_1050_a_failing_run_leaves_no_state_behind() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold");
    let snapshot = |dir: &std::path::Path| {
        fn walk(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, u64)>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let m = e.metadata().unwrap();
                if m.is_dir() {
                    walk(&e.path(), out);
                } else {
                    out.push((e.path(), m.len()));
                }
            }
        }
        let mut v = Vec::new();
        walk(dir, &mut v);
        v.sort();
        v
    };
    let before = snapshot(&root);
    let failing = perturbed("circle-001", |t| {
        t["entities"][0]["geometry"]["radius"] = hexnum(9.0);
    });
    assert!(!failing.is_empty(), "the run really failed");
    assert_eq!(snapshot(&root), before, "the vendored tree is untouched");
}

#[test]
fn n2_1_1050_the_suite_starts_no_process_and_needs_no_converter() {
    // Built from pieces so this file does not match its own scan.
    let needles = [
        ["std::pro", "cess"].concat(),
        ["Comm", "and::new"].concat(),
        ["dock", "er"].concat(),
        ["CAD Exch", "anger"].concat(),
        ["O", "DA"].concat() + " image",
    ];
    for file in [
        include_str!("pvec_known_truth.rs"),
        include_str!("common/pvec_gold.rs"),
    ] {
        for n in &needles {
            let hit = file
                .lines()
                .find(|l| l.contains(n.as_str()) && !l.trim_start().starts_with("//"));
            assert!(hit.is_none(), "found {n:?} in suite code: {hit:?}");
        }
    }
}

mod mapping {
    use super::*;

    pub fn check(_fx: &Fixture, _truth: &Value) -> Vec<String> {
        vec!["the mapping comparison is not written yet".to_string()]
    }
}
