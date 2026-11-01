//! N2.2 (#1050): the dual-oracle suite.
//!
//! Real drawings have no independent ground truth, so the corpus admits them as differential
//! regressions only: two open converters agreeing (`acadsharp-dwg2step` and `libredwg-dwg2step`,
//! ADR 0010) is `DUAL_ORACLE_EQUIVALENT` evidence, never source truth. This suite takes every
//! fixture of the pinned revision that is not `KNOWN_TRUTH_EXACT` (that tier is
//! `pvec_known_truth`), decodes it through the same path the conversion uses, and compares the
//! world-space result with the converters' neutral STEP signature, one differential per geometry
//! class. The result of each fixture goes into a report with its evidence class and its status.
//!
//! * `QUARANTINED` fixtures run and report, and can never fail the gate.
//! * No converter runs here and none is needed: the signatures are published corpus artifacts,
//!   vendored under `tests/fixtures/pvec-gold`.
//! * The report is checked for wording: it labels results `DUAL_ORACLE_EQUIVALENT` and never
//!   calls converter agreement source truth or exactness.
//!
//! Set `PVEC_CONFORMANCE_REPORT_DIR` to also write the report to
//! `<dir>/dual-oracle.txt` (the local gate collects it). Needs `--features pvec-conformance`.

#![cfg(feature = "pvec-conformance")]

#[path = "common/pvec_gold_decode.rs"]
mod decode;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use pvec_geometry::{Curve, Curve2d};
use pvec_gold::{Corpus, EvidenceClass, EvidenceScope, Fixture, nums, points};
use serde_json::Value;

/// What libviprs and the converters made of one geometry class of one fixture.
#[derive(Debug, Default, Clone, PartialEq)]
struct Differential {
    compared: usize,
    matched: usize,
    deviations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    /// Compared against the converters' signature, per geometry class.
    Compared(BTreeMap<String, Differential>),
    /// The converters wrote nothing comparable, or the corpus holds no signature for it.
    NoStepEvidence(String),
    /// libviprs could not open the drawing.
    NotDecoded(String),
}

#[derive(Debug, Clone)]
struct Row {
    id: String,
    class: EvidenceClass,
    outcome: Outcome,
}

impl Row {
    fn status(&self) -> &'static str {
        if self.class == EvidenceClass::Quarantined {
            "quarantined"
        } else {
            "accepted"
        }
    }

    /// Whether this row lets the gate pass. A quarantined fixture runs and reports and never
    /// fails it; an accepted one fails on any deviation.
    fn gate_passes(&self) -> bool {
        match &self.outcome {
            _ if self.class == EvidenceClass::Quarantined => true,
            Outcome::Compared(d) => d.values().all(|d| d.deviations.is_empty()),
            Outcome::NoStepEvidence(_) => true,
            Outcome::NotDecoded(_) => false,
        }
    }
}

// ------------------------------------------------------------------------- the comparison

/// libviprs' leaves of a drawing as (class, world curve) pairs or the reason one has none, and
/// the metres per drawing unit: libviprs keeps source units (the unit travels beside the
/// geometry), while the converters write STEP in metres. The factors are physical constants,
/// written here and not taken from libviprs.
type Leaf = Result<(&'static str, Curve), String>;

fn libviprs_side(fx: &Fixture) -> Result<(Vec<Leaf>, f64), String> {
    let bytes = fx.dwg();
    let provider = decode::open(&bytes)?;
    let flat = decode::flatten(&provider, &fx.id).map_err(|e| format!("flatten refused: {e}"))?;
    let metres = match provider.source_unit().insunits_code() {
        Some(1) => 0.0254,
        Some(2) => 0.3048,
        Some(4) => 0.001,
        Some(5) => 0.01,
        Some(6) | Some(0) | None => 1.0,
        Some(other) => {
            return Err(format!(
                "INSUNITS {other} has no metre factor in this suite"
            ));
        }
    };
    Ok((flat.entities.iter().map(decode::world).collect(), metres))
}

/// How one converter entity differs from one libviprs curve, `None` when it does not.
fn differs(class: &str, curve: &Curve, g: &Value, k: f64) -> Option<String> {
    let pts: Vec<[f64; 2]> = decode::sample(curve, 8)
        .iter()
        .map(|p| [p[0] * k, p[1] * k])
        .collect();
    let ends = (pts[0], pts[pts.len() - 1]);
    match class {
        "line" => {
            let (s, e) = (nums(&g["start"]), nums(&g["end"]));
            let sc = decode::scale_of(&[s[0], s[1], e[0], e[1]]);
            let (s, e) = ([s[0], s[1]], [e[0], e[1]]);
            if decode::near_pt(ends.0, s, sc) && decode::near_pt(ends.1, e, sc) {
                None
            } else if decode::near_pt(ends.0, e, sc) && decode::near_pt(ends.1, s, sc) {
                Some("the line runs the other way".into())
            } else {
                Some(format!(
                    "the line runs {:?} to {:?}, the converter {s:?} to {e:?}",
                    ends.0, ends.1
                ))
            }
        }
        "circle" | "ellipse" | "circular_arc" | "elliptical_arc" => {
            let c = nums(&g["center"]);
            let (a, b) = if class.starts_with("circ") || class == "circle" {
                (pvec_gold::num(&g["radius"]), pvec_gold::num(&g["radius"]))
            } else {
                (
                    pvec_gold::num(&g["semi_major"]),
                    pvec_gold::num(&g["semi_minor"]),
                )
            };
            let axis = if g.get("major_axis").is_some() {
                nums(&g["major_axis"])
            } else {
                nums(&g["x_axis"])
            };
            let n = axis[0].hypot(axis[1]);
            let u = [axis[0] / n, axis[1] / n];
            let sc = decode::scale_of(&[c[0], c[1], a, b]);
            pts.iter()
                .find(|p| !decode::on_ellipse(**p, [c[0], c[1]], u, a, b, sc))
                .map(|p| format!("the point {p:?} is off the converter's {class}"))
        }
        "polyline" => {
            let want = points(&g["vertices"]);
            let Curve::Polyline(p) = curve else {
                return Some(format!(
                    "libviprs has a {:?}, the converter a polyline",
                    curve.kind()
                ));
            };
            let got = p.vertices();
            if got.len() < want.len() {
                return Some(format!(
                    "{} vertices against the converter's {}",
                    got.len(),
                    want.len()
                ));
            }
            want.iter().zip(got).enumerate().find_map(|(i, (w, v))| {
                let sc = decode::scale_of(&[w[0], w[1]]);
                (!decode::near_pt([v.x * k, v.y * k], [w[0], w[1]], sc)).then(|| {
                    format!(
                        "vertex {i} is ({}, {}), the converter's ({}, {})",
                        v.x * k,
                        v.y * k,
                        w[0],
                        w[1]
                    )
                })
            })
        }
        "bspline" | "nurbs" | "bezier" => {
            let want = points(&g["control_points"]);
            let got: Vec<[f64; 2]> = match curve {
                Curve::BSpline(s) => s
                    .control_points()
                    .iter()
                    .map(|p| [p.x * k, p.y * k])
                    .collect(),
                Curve::RationalBSpline(s) => s
                    .control_points()
                    .iter()
                    .map(|p| [p.x * k, p.y * k])
                    .collect(),
                Curve::Bezier(s) => s
                    .control_points()
                    .iter()
                    .map(|p| [p.x * k, p.y * k])
                    .collect(),
                other => {
                    return Some(format!(
                        "libviprs has a {:?}, the converter a spline",
                        other.kind()
                    ));
                }
            };
            if got.len() != want.len() {
                return Some(format!(
                    "{} control points against the converter's {}",
                    got.len(),
                    want.len()
                ));
            }
            got.iter().zip(&want).enumerate().find_map(|(i, (v, w))| {
                let sc = decode::scale_of(&[w[0], w[1]]);
                (!decode::near_pt(*v, [w[0], w[1]], sc))
                    .then(|| format!("control point {i} is {v:?}, the converter's {w:?}"))
            })
        }
        other => Some(format!("no comparison for the converter's class {other}")),
    }
}

/// Compares libviprs' leaves with a converter signature, one differential per geometry class.
fn compare(ours: &[Leaf], metres: f64, signature: &Value) -> BTreeMap<String, Differential> {
    let mut out: BTreeMap<String, Differential> = BTreeMap::new();
    let theirs: Vec<&Value> = signature["entities"]
        .as_array()
        .map_or(vec![], |v| v.iter().collect());
    let mut by_class: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for t in &theirs {
        by_class
            .entry(t["class"].as_str().unwrap_or("?"))
            .or_default()
            .push(t);
    }
    // The converters do not promise libviprs' entity order, so an entity matches any converter
    // entity of its class that has not been matched yet.
    let mut taken: BTreeMap<&str, Vec<bool>> = by_class
        .iter()
        .map(|(c, v)| (*c, vec![false; v.len()]))
        .collect();
    for r in ours {
        match r {
            Ok((class, curve)) => {
                let slot = out.entry((*class).to_string()).or_default();
                slot.compared += 1;
                let candidates = by_class.get(class).map_or(&[][..], |v| v.as_slice());
                let flags = taken.entry(class).or_default();
                let mut first_miss = None;
                let hit = candidates.iter().enumerate().find(|(i, t)| {
                    if flags[*i] {
                        return false;
                    }
                    let miss = differs(class, curve, &t["geometry"], metres);
                    let found = miss.is_none();
                    if first_miss.is_none() {
                        first_miss = miss;
                    }
                    found
                });
                match hit {
                    Some((i, _)) => {
                        flags[i] = true;
                        slot.matched += 1;
                    }
                    None => slot.deviations.push(
                        first_miss.unwrap_or_else(|| "the converters wrote no such entity".into()),
                    ),
                }
            }
            Err(why) => {
                let slot = out.entry("unmapped".to_string()).or_default();
                slot.compared += 1;
                slot.deviations.push(format!("libviprs {why}"));
            }
        }
    }
    for (class, flags) in &taken {
        let left = flags.iter().filter(|t| !**t).count();
        if left > 0 {
            let slot = out.entry((*class).to_string()).or_default();
            slot.deviations.push(format!(
                "the converters wrote {left} {class} entities libviprs did not produce"
            ));
        }
    }
    out
}

fn row(fx: &Fixture) -> Row {
    let outcome = if !fx.has("primary.signature.json") {
        let why = fx
            .has("verdict.json")
            .then(|| fx.json("verdict.json"))
            .and_then(|v| v["reason"].as_str().map(str::to_string))
            .unwrap_or_else(|| "the corpus holds no signature".into());
        Outcome::NoStepEvidence(why)
    } else {
        match libviprs_side(fx) {
            Err(why) => Outcome::NotDecoded(why),
            Ok((ours, metres)) => {
                Outcome::Compared(compare(&ours, metres, &fx.json("primary.signature.json")))
            }
        }
    };
    Row {
        id: fx.id.clone(),
        class: fx.class,
        outcome,
    }
}

/// Every fixture of the pinned revision that this suite owns.
fn rows(corpus: &Corpus) -> Vec<Row> {
    corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::KnownTruthExact)
        .map(row)
        .collect()
}

/// The report. Evidence class and status of every executed fixture, then its differentials.
fn report(corpus: &Corpus, rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "pvec dual-oracle differential, corpus {} @ {}",
        corpus.revision, corpus.gold_commit
    );
    let _ = writeln!(
        out,
        "oracles: acadsharp-dwg2step, libredwg-dwg2step (OPEN_CONVERTERS, ADR 0010)"
    );
    for r in rows {
        let _ = writeln!(
            out,
            "fixture {} evidence_class={} status={}",
            r.id,
            r.class.tag(),
            r.status()
        );
        match &r.outcome {
            Outcome::Compared(d) => {
                for (class, d) in d {
                    let _ = writeln!(
                        out,
                        "  {class}: compared={} matched={} deviations={}",
                        d.compared,
                        d.matched,
                        d.deviations.len()
                    );
                    for v in &d.deviations {
                        let _ = writeln!(out, "    deviation: {v}");
                    }
                }
            }
            Outcome::NoStepEvidence(why) => {
                let _ = writeln!(out, "  no converter evidence to compare: {why}");
            }
            Outcome::NotDecoded(why) => {
                let _ = writeln!(out, "  libviprs did not decode it: {why}");
            }
        }
        let _ = writeln!(
            out,
            "  gate: {}",
            if r.gate_passes() { "pass" } else { "fail" }
        );
    }
    out
}

fn forbidden_phrases(text: &str) -> Vec<&'static str> {
    let lower = text.to_lowercase();
    [
        "source truth",
        "structurally exact",
        "structural exactness",
        "exactness",
        "known_truth",
    ]
    .into_iter()
    .filter(|p| lower.contains(p))
    .collect()
}

// ------------------------------------------------------------------------------------ tests

#[test]
fn n2_2_1050_every_fixture_that_is_not_known_truth_is_executed_and_reported() {
    let corpus = Corpus::open();
    let rows = rows(&corpus);
    let want: Vec<&str> = corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::KnownTruthExact)
        .map(|f| f.id.as_str())
        .collect();
    assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), want);
    let text = report(&corpus, &rows);
    for r in &rows {
        let head = format!(
            "fixture {} evidence_class={} status={}",
            r.id,
            r.class.tag(),
            r.status()
        );
        assert!(text.contains(&head), "the report omits {head}");
    }
    if let Ok(dir) = std::env::var("PVEC_CONFORMANCE_REPORT_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join("dual-oracle.txt"), &text).unwrap();
    }
    let failing: Vec<&str> = rows
        .iter()
        .filter(|r| !r.gate_passes())
        .map(|r| r.id.as_str())
        .collect();
    assert!(failing.is_empty(), "the gate fails on {failing:?}\n{text}");
}

#[test]
fn n2_2_1050_the_accepted_dual_oracle_fixture_is_compared_not_skipped() {
    let corpus = Corpus::open();
    let fx = corpus.get("blocks-dynamic-linear-001");
    assert_eq!(fx.class, EvidenceClass::DualOracleEquivalent);
    assert_eq!(fx.scope, EvidenceScope::StepOracle);
    let Outcome::Compared(d) = row(fx).outcome else {
        panic!("a dual-oracle fixture with a signature must be compared");
    };
    assert!(d.values().map(|d| d.compared).sum::<usize>() > 0, "{d:?}");
}

#[test]
fn n2_2_1050_a_quarantined_fixture_that_fails_its_comparison_leaves_the_gate_green_and_is_reported()
{
    let corpus = Corpus::open();
    let mut quarantined = row(corpus.get("zero-radius-001"));
    assert_eq!(quarantined.class, EvidenceClass::Quarantined);
    let Outcome::Compared(ref mut d) = quarantined.outcome else {
        panic!("zero-radius-001 has a signature and must be compared");
    };
    d.entry("circle".to_string())
        .or_default()
        .deviations
        .push("forced deviation".into());
    assert!(quarantined.gate_passes(), "quarantine never gates");
    let text = report(&corpus, std::slice::from_ref(&quarantined));
    assert!(text.contains("deviation: forced deviation"), "{text}");
    assert!(text.contains("status=quarantined"), "{text}");
    // The same deviation on an accepted fixture does fail it.
    let mut accepted = quarantined.clone();
    accepted.class = EvidenceClass::DualOracleEquivalent;
    assert!(!accepted.gate_passes());
}

#[test]
fn n2_2_1050_the_report_never_calls_agreement_source_truth_or_exactness() {
    let corpus = Corpus::open();
    let text = report(&corpus, &rows(&corpus));
    assert!(
        forbidden_phrases(&text).is_empty(),
        "{:?}\n{text}",
        forbidden_phrases(&text)
    );
    assert!(text.contains("DUAL_ORACLE_EQUIVALENT"));
    // The check itself bites.
    assert_eq!(
        forbidden_phrases("two converters agree, so this is source truth"),
        vec!["source truth"]
    );
    assert!(!forbidden_phrases("this is structurally exact").is_empty());
}

#[test]
fn n2_2_1050_the_suite_starts_no_process_and_needs_no_converter() {
    let needles = [
        ["std::pro", "cess"].concat(),
        ["Comm", "and::new"].concat(),
        ["dock", "er"].concat(),
        ["CAD Exch", "anger"].concat(),
    ];
    for file in [
        include_str!("pvec_dual_oracle.rs"),
        include_str!("common/pvec_gold_decode.rs"),
    ] {
        for n in &needles {
            let hit = file
                .lines()
                .find(|l| l.contains(n.as_str()) && !l.trim_start().starts_with("//"));
            assert!(hit.is_none(), "found {n:?} in suite code: {hit:?}");
        }
    }
}

#[test]
fn n2_2_1050_the_pin_summary_is_the_index_and_a_bump_reads_as_per_class_changes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold");
    let summary: Value =
        serde_json::from_slice(&std::fs::read(root.join("summary.json")).unwrap()).unwrap();
    let corpus = Corpus::open();
    let mut derived: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in corpus.all() {
        derived
            .entry(format!("{} / {}", f.class.tag(), f.scope.tag()))
            .or_default()
            .push(f.id.clone());
    }
    for v in derived.values_mut() {
        v.sort();
    }
    let recorded: BTreeMap<String, Vec<String>> =
        serde_json::from_value(summary["by_class"].clone()).unwrap();
    assert_eq!(
        recorded, derived,
        "summary.json is stale: re-run tools/pvec-gold-vendor.py"
    );
    assert_eq!(summary["corpus_revision"], corpus.revision.as_str());

    // A bump is a diff of two such summaries; this is what a reviewer reads.
    let mut next = recorded.clone();
    let moved = next
        .get_mut("QUARANTINED / STEP_ORACLE")
        .unwrap()
        .pop()
        .unwrap();
    next.entry("DUAL_ORACLE_EQUIVALENT / STEP_ORACLE".to_string())
        .or_default()
        .push(moved.clone());
    let changes = summary_diff(&recorded, &next);
    assert_eq!(
        changes,
        vec![
            "DUAL_ORACLE_EQUIVALENT / STEP_ORACLE: +".to_string() + &moved,
            "QUARANTINED / STEP_ORACLE: -".to_string() + &moved,
        ]
    );
}

/// The per-class lines of a pin bump: ids that joined (`+`) or left (`-`) each class.
fn summary_diff(
    old: &BTreeMap<String, Vec<String>>,
    new: &BTreeMap<String, Vec<String>>,
) -> Vec<String> {
    let mut out = Vec::new();
    for class in old
        .keys()
        .chain(new.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (a, b) = (
            old.get(class).cloned().unwrap_or_default(),
            new.get(class).cloned().unwrap_or_default(),
        );
        for id in b.iter().filter(|i| !a.contains(i)) {
            out.push(format!("{class}: +{id}"));
        }
        for id in a.iter().filter(|i| !b.contains(i)) {
            out.push(format!("{class}: -{id}"));
        }
    }
    out
}
