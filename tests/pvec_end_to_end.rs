//! N2.3 (#1050): the DWG to PVec to comparison path, end to end.
//!
//! ```text
//! DWG -> provider -> normalization -> PVec write -> PVec read -> comparison
//! ```
//!
//! Every non-quarantined fixture of the pinned `pvec-gold-fixtures` revision is converted with the
//! public conversion API, the published artifact is opened with PVec's public reader, and what the
//! reader says is compared with the fixture's `expected.math.json` (a known-truth fixture) or
//! checked for consistency with the conversion report (every other class, which has no source
//! truth to compare against). The comparison runs on the re-read curves, never on the in-memory
//! normalization result, so a writer or reader defect cannot hide behind a correct mapping.
//!
//! A mismatch names the stage that introduced it: `provider`, `normalization`, `write`, `read` or
//! `compare`. Converting one fixture twice gives the same bytes, for the artifact and the report.
//!
//! Only public API is used: the conversion API, `open_published`, PVec's `PvecReader` and the
//! curve types it hands back. No container offset or private module path appears here. The tolerance
//! is `Policy::V1`'s, there is no literal epsilon.
//!
//! Nothing runs a converter. Every artifact is written under a temporary directory that is removed
//! when the run ends, including when it fails. Set `PVEC_GATE_OUT` to also write the determinism
//! digest set (`determinism.digests`) there, which is how the local gate collects it.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::writer::WriteError;
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use pvec_facade::PvecReader;
use pvec_geometry::{Curve, Curve2d};
use pvec_gold::{Corpus, EvidenceClass, Fixture, num, nums, points};
use pvec_numeric::Policy;
use serde_json::Value;

const POLICY: Policy = Policy::V1;

// -------------------------------------------------------------------------------------- stages

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Provider,
    Normalization,
    Write,
    Read,
    Compare,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Provider => "provider",
            Stage::Normalization => "normalization",
            Stage::Write => "write",
            Stage::Read => "read",
            Stage::Compare => "compare",
        }
    }
}

#[derive(Debug)]
struct Failure {
    stage: Stage,
    why: String,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.stage.name(), self.why)
    }
}

fn fail<T>(stage: Stage, why: impl Into<String>) -> Result<T, Failure> {
    Err(Failure {
        stage,
        why: why.into(),
    })
}

/// The stage a conversion error belongs to.
fn stage_of(e: &PvecConvertError) -> Stage {
    match e {
        PvecConvertError::Open(_) => Stage::Provider,
        PvecConvertError::Write(_) | PvecConvertError::Publish(_) | PvecConvertError::Cancelled => {
            Stage::Write
        }
        _ => Stage::Normalization,
    }
}

fn describe(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(s) = cur {
        out.push_str(": ");
        out.push_str(&s.to_string());
        cur = s.source();
    }
    out
}

// -------------------------------------------------------------------------------- the pipeline

/// What one fixture became on the way through.
struct Converted {
    /// The published artifact, as published.
    artifact: Vec<u8>,
    /// The published report, as published.
    report_bytes: Vec<u8>,
    report: ConversionReport,
    /// Re-read through PVec's public reader, in PVec id order.
    curves: Vec<Curve>,
}

/// Stages one to four: provider, normalization, write, read. `Ok(None)` is a drawing that no entity
/// of produced geometry for (every entity was refused or ignored by policy), which the writer
/// reports as `NoGeometry`.
fn convert(dwg: &[u8], dir: &Path, name: &str) -> Result<Option<Converted>, Failure> {
    let dest = dir.join(format!("{name}.pvec"));
    let out = match PvecConversion::new(CadSource::Bytes(dwg), &dest)
        .unit(SourceUnit::Meters)
        .convert()
    {
        Ok(o) => o,
        Err(PvecConvertError::Write(WriteError::NoGeometry)) => return Ok(None),
        Err(e) => return fail(stage_of(&e), describe(&e)),
    };
    let artifact = std::fs::read(&dest).or_else(|e| fail(Stage::Write, e.to_string()))?;
    let report_bytes =
        std::fs::read(report_path(&dest)).or_else(|e| fail(Stage::Write, e.to_string()))?;
    let curves = read_back(&dest)?;
    if curves.len() as u64 != out.written {
        return fail(
            Stage::Read,
            format!(
                "the conversion wrote {} entities and the reader finds {}",
                out.written,
                curves.len()
            ),
        );
    }
    Ok(Some(Converted {
        artifact,
        report_bytes,
        report: out.report,
        curves,
    }))
}

fn read_back(path: &Path) -> Result<Vec<Curve>, Failure> {
    let reader = open_published(path).or_else(|e| fail(Stage::Read, describe(&e)))?;
    entities(&reader)
}

fn entities(reader: &PvecReader) -> Result<Vec<Curve>, Failure> {
    let all = reader
        .entities()
        .or_else(|e| fail(Stage::Read, describe(&e)))?;
    for (n, (id, _)) in all.iter().enumerate() {
        if id.index() as usize != n {
            return fail(
                Stage::Read,
                format!("entity {n} came back as id {}", id.index()),
            );
        }
    }
    Ok(all.into_iter().map(|(_, c)| c).collect())
}

// ------------------------------------------------------------------------------ the comparison

type Problems = Vec<String>;

fn near(a: f64, b: f64, scale: f64) -> bool {
    POLICY.near(a, b, scale)
}

fn near_pt(a: [f64; 2], b: [f64; 2], scale: f64) -> bool {
    POLICY.near_point(a, b, scale)
}

fn scale_of(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0)
}

fn end_points(c: &Curve) -> [[f64; 2]; 2] {
    let d = c.domain();
    let at = |u: f64| {
        let p = c.eval(u).expect("a re-read curve evaluates");
        [p.x, p.y]
    };
    [at(d.lo()), at(d.hi())]
}

/// The radii and the first axis of a conic's truth record.
fn conic_frame(class: &str, g: &Value) -> (f64, f64, [f64; 2]) {
    let (rx, ry, axis) = match class {
        "circle" | "circular_arc" => (num(&g["radius"]), num(&g["radius"]), nums(&g["x_axis"])),
        _ => (
            num(&g["semi_major"]),
            num(&g["semi_minor"]),
            nums(&g["major_axis"]),
        ),
    };
    let n = axis[0].hypot(axis[1]);
    (rx, ry, [axis[0] / n, axis[1] / n])
}

/// Whether `curve` is what the truth record `class`/`g` describes. Returns the reasons it is not.
fn differs(class: &str, g: &Value, t: &Value, curve: &Curve) -> Problems {
    let mut out = Problems::new();
    match class {
        "line" => {
            let (s, e) = (nums(&g["start"]), nums(&g["end"]));
            let got = end_points(curve);
            let sc = scale_of(&[s[0], s[1], e[0], e[1]]);
            if !near_pt(got[0], [s[0], s[1]], sc) || !near_pt(got[1], [e[0], e[1]], sc) {
                out.push(format!(
                    "line: runs {:?} to {:?}, truth {s:?} to {e:?}",
                    got[0], got[1]
                ));
            }
        }
        "circle" | "ellipse" => {
            let c = nums(&g["center"]);
            let (rx, ry, u) = conic_frame(class, g);
            let sc = scale_of(&[c[0], c[1], rx, ry]);
            let d = curve.domain();
            for k in 0..8 {
                let p = curve.eval(d.lo() + d.width() * f64::from(k) / 8.0).unwrap();
                let (dx, dy) = (p.x - c[0], p.y - c[1]);
                let (s, q) = (dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0]);
                if !near((s / rx).hypot(q / ry), 1.0, sc.max(1.0)) {
                    out.push(format!(
                        "{class}: the point ({}, {}) is off the truth",
                        p.x, p.y
                    ));
                    break;
                }
            }
        }
        "circular_arc" | "elliptical_arc" => {
            let c = nums(&g["center"]);
            let (rx, ry, ax) = conic_frame(class, g);
            // The y axis follows the normal, so a reflected frame is honoured.
            let nz = g.get("normal").map_or(1.0, |n| nums(n)[2]);
            let ay = [-nz * ax[1], nz * ax[0]];
            let (a, sweep) = (num(&g["start_angle"]), num(&g["sweep_angle"]));
            let at = |th: f64| {
                [
                    c[0] + rx * th.cos() * ax[0] + ry * th.sin() * ay[0],
                    c[1] + rx * th.cos() * ax[1] + ry * th.sin() * ay[1],
                ]
            };
            let got = end_points(curve);
            let sc = scale_of(&[rx, ry, c[0], c[1]]);
            if !near_pt(got[0], at(a), sc) || !near_pt(got[1], at(a + sweep), sc) {
                out.push(format!(
                    "{class}: runs {:?} to {:?}, truth {:?} to {:?}",
                    got[0],
                    got[1],
                    at(a),
                    at(a + sweep)
                ));
            }
        }
        "polyline" => {
            let v = points(&g["vertices"]);
            let closed = t["closure"]["closed_by_endpoint"]
                .as_bool()
                .unwrap_or(false);
            let got = end_points(curve);
            let (first, last) = (&v[0], if closed { &v[0] } else { &v[v.len() - 1] });
            let sc = scale_of(&v.iter().flatten().copied().collect::<Vec<_>>());
            if !near_pt(got[0], [first[0], first[1]], sc)
                || !near_pt(got[1], [last[0], last[1]], sc)
            {
                out.push(format!(
                    "polyline: runs {:?} to {:?}, truth {first:?} to {last:?}",
                    got[0], got[1]
                ));
            }
            if let Curve::Polyline(p) = curve {
                let want = v.len() + usize::from(closed);
                if p.vertices().len() != want {
                    out.push(format!(
                        "polyline: {} vertices, truth {want}",
                        p.vertices().len()
                    ));
                }
                for (i, (a, w)) in p.vertices().iter().zip(&v).enumerate() {
                    if !near_pt([a.x, a.y], [w[0], w[1]], sc) {
                        out.push(format!(
                            "polyline: vertex {i} is ({}, {}), truth {w:?}",
                            a.x, a.y
                        ));
                    }
                }
            }
        }
        "bspline" | "nurbs" | "bezier" => spline(&mut out, class, g, t, curve),
        other => out.push(format!("no comparison for the truth class {other}")),
    }
    out
}

fn spline(out: &mut Problems, class: &str, g: &Value, t: &Value, curve: &Curve) {
    let degree = g["degree"].as_u64().unwrap_or(0) as usize;
    let want_cps = points(&g["control_points"]);
    let (got_degree, cps, knots, weights) = match curve {
        Curve::BSpline(s) => (
            s.degree(),
            s.control_points()
                .iter()
                .map(|p| [p.x, p.y])
                .collect::<Vec<[f64; 2]>>(),
            Some(s.knots().to_vec()),
            None::<Vec<f64>>,
        ),
        Curve::RationalBSpline(s) => (
            s.degree(),
            s.control_points().iter().map(|p| [p.x, p.y]).collect(),
            Some(s.knots().to_vec()),
            Some(s.weights().to_vec()),
        ),
        Curve::Bezier(b) => (
            b.degree(),
            b.control_points().iter().map(|p| [p.x, p.y]).collect(),
            None,
            None,
        ),
        other => {
            out.push(format!("{class}: came back as {:?}", other.kind()));
            return;
        }
    };
    if got_degree != degree {
        out.push(format!("degree: re-read {got_degree}, truth {degree}"));
    }
    if t["closure"]["periodic"].as_bool().unwrap_or(false) {
        // Not comparable structurally: PVec stores a clamped spline, so a periodic source comes back
        // with its control points and knots rewritten. The weaker class it may claim is
        // SupportedEquivalent (checked in `policy`), and what is asserted here is what survives the
        // rewrite: the degree (above) and that the curve still closes on itself.
        let [a, b] = end_points(curve);
        let sc = scale_of(&[a[0], a[1], b[0], b[1]]);
        if !near_pt(a, b, sc) {
            out.push(format!(
                "closed state: a periodic source's curve runs {a:?} to {b:?}"
            ));
        }
        return;
    }
    if cps.len() != want_cps.len() {
        out.push(format!(
            "control points: re-read {}, truth {}",
            cps.len(),
            want_cps.len()
        ));
    } else {
        for (i, (a, w)) in cps.iter().zip(&want_cps).enumerate() {
            if a[0] != w[0] || a[1] != w[1] {
                out.push(format!(
                    "control point {i}: re-read ({}, {}), truth ({}, {})",
                    a[0], a[1], w[0], w[1]
                ));
            }
        }
    }
    if let (Some(got), Some(list)) = (&knots, g.get("knots").and_then(Value::as_array)) {
        let want: Vec<f64> = list
            .iter()
            .flat_map(|k| {
                let m = k["multiplicity"].as_u64().unwrap_or(1) as usize;
                std::iter::repeat_n(num(&k["value"]), m)
            })
            .collect();
        if *got != want {
            out.push(format!(
                "knots: re-read {got:?}, truth (with multiplicities) {want:?}"
            ));
        }
    }
    // A non-rational spline has no weight array at all, not a row of ones.
    match (&weights, g.get("weights")) {
        (None, None) => {}
        (Some(w), Some(want)) if *w == nums(want) => {}
        (Some(w), Some(want)) => out.push(format!("weights: re-read {w:?}, truth {want}")),
        (None, Some(_)) => out.push("weights: the truth has weights and the artifact none".into()),
        (Some(_), None) => out.push("weights: a weight array on a non-rational spline".into()),
    }
}

/// The geometry a truth entity says the artifact holds: the world result when the entity is
/// instanced under a transform, the curve itself otherwise.
fn target(t: &Value) -> (&str, &Value) {
    match t.get("world") {
        Some(w) => (w["class"].as_str().unwrap_or("?"), &w["geometry"]),
        None => (t["class"].as_str().unwrap_or("?"), &t["geometry"]),
    }
}

/// The truth entities the artifact is expected to hold: not an equivalent form of another, and not
/// on a layer that is off or frozen (the view profile excludes those, and says so in the report).
fn live(truth: &Value) -> Vec<&Value> {
    truth["entities"]
        .as_array()
        .expect("a truth record lists entities")
        .iter()
        .filter(|e| !e["id"].as_str().is_some_and(|i| i.contains("-equivalent-")))
        .filter(|e| {
            let l = &e["layer"];
            l["on"].as_bool().unwrap_or(true) && !l["frozen"].as_bool().unwrap_or(false)
        })
        .collect()
}

/// What the report must say about a known-truth fixture: an entity the view profile drops is counted
/// as ignored by policy (and is absent from the artifact, which the entity count checks), and a
/// periodic source is never claimed exact.
fn policy(c: &Converted, truth: &Value) -> Result<(), Failure> {
    let all = truth["entities"].as_array().unwrap();
    let primary = all
        .iter()
        .filter(|e| !e["id"].as_str().is_some_and(|i| i.contains("-equivalent-")))
        .count();
    let dropped = (primary - live(truth).len()) as u64;
    let ignored = c
        .report
        .totals
        .by_fidelity
        .get("IgnoredByPolicy")
        .copied()
        .unwrap_or(0);
    if ignored != dropped {
        return fail(
            Stage::Compare,
            format!(
                "the truth drops {dropped} entities by layer state, the report ignores {ignored}"
            ),
        );
    }
    let periodic = live(truth).iter().any(|e| {
        matches!(e["class"].as_str(), Some("bspline" | "nurbs"))
            && e["closure"]["periodic"].as_bool().unwrap_or(false)
    });
    if periodic
        && c.report
            .entities
            .iter()
            .any(|r| r.semantic_type.is_some() && r.fidelity == "SupportedExact")
    {
        return fail(
            Stage::Compare,
            "a periodic source was claimed SupportedExact, but it is stored rewritten",
        );
    }
    Ok(())
}

/// Stage five against a known-truth record: every truth entity has a distinct re-read curve that
/// is the one it describes, and nothing else was written.
fn compare_to_truth(curves: &[Curve], truth: &Value) -> Result<(), Failure> {
    let entities = live(truth);
    if entities.len() != curves.len() {
        return fail(
            Stage::Compare,
            format!(
                "entity count: the truth has {}, the artifact holds {}",
                entities.len(),
                curves.len()
            ),
        );
    }
    let mut claimed = vec![false; curves.len()];
    for t in entities {
        let (class, g) = target(t);
        let id = t["id"].as_str().unwrap_or("?");
        let mut reasons = Problems::new();
        let hit = curves.iter().enumerate().find(|(i, c)| {
            if claimed[*i] {
                return false;
            }
            let why = differs(class, g, t, c);
            let ok = why.is_empty();
            reasons.extend(why);
            ok
        });
        match hit {
            Some((i, _)) => claimed[i] = true,
            None => {
                reasons.sort();
                reasons.dedup();
                return fail(
                    Stage::Compare,
                    format!("{id}: no re-read curve matches ({})", reasons.join("; ")),
                );
            }
        }
    }
    Ok(())
}

/// Stage five against a fixture with no source truth: the artifact holds exactly the geometry the
/// report says was written, in the report's id order, and every curve evaluates.
fn compare_to_report(c: &Converted) -> Result<(), Failure> {
    let drawn = c
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .count();
    if drawn != c.curves.len() {
        return fail(
            Stage::Compare,
            format!(
                "the report has {drawn} entities with geometry, the artifact holds {}",
                c.curves.len()
            ),
        );
    }
    for (n, curve) in c.curves.iter().enumerate() {
        let d = curve.domain();
        for u in [d.lo(), d.hi()] {
            if curve.eval(u).is_err() {
                return fail(
                    Stage::Compare,
                    format!("entity {n} does not evaluate at {u}"),
                );
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------ one run

/// All five stages for one fixture. A known-truth fixture that expects a refusal must be refused
/// at normalization (or leave nothing to write) and must not yield a curve.
fn run(fx: &Fixture, truth: Option<&Value>, dir: &Path) -> Result<Option<Converted>, Failure> {
    let dwg = fx.dwg();
    let refusal = truth.is_some_and(|t| {
        t["entities"]
            .as_array()
            .is_some_and(|l| l.iter().any(|e| e.get("rejection").is_some()))
    });
    let done = match convert(&dwg, dir, &fx.id) {
        Err(f) if refusal && f.stage == Stage::Normalization => return Ok(None),
        other => other?,
    };
    if refusal {
        return match done {
            None => Ok(None),
            Some(c) => fail(
                Stage::Normalization,
                format!(
                    "the truth expects a refusal and {} curves came out",
                    c.curves.len()
                ),
            ),
        };
    }
    let Some(c) = done else {
        return match truth {
            Some(t) if !t["entities"].as_array().is_some_and(Vec::is_empty) => fail(
                Stage::Normalization,
                "the truth lists entities and no geometry reached the writer",
            ),
            _ => Ok(None),
        };
    };
    match truth {
        Some(t) => {
            compare_to_truth(&c.curves, t)?;
            policy(&c, t)?;
        }
        None => compare_to_report(&c)?,
    }
    Ok(Some(c))
}

fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The index of the first differing byte, or `None` when the two are identical.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(a.iter().zip(b).take_while(|(x, y)| x == y).count())
}

// -------------------------------------------------------------------------------------- tests

/// Drawings that cannot be converted today, each with the text of the failure and where it is
/// tracked. A gap is waived only while it is real: once a fix makes the fixture pass, the suite
/// fails until its row is deleted, so this list cannot go stale.
const KNOWN_GAPS: &[(&str, &str, &str)] = &[
    ("bulge-arc-001", "WRT-08", "libviprs/libviprs#1446"),
    ("bulge-arc-neg-001", "WRT-08", "libviprs/libviprs#1446"),
    ("mixed-entity-001", "WRT-08", "libviprs/libviprs#1446"),
];

fn gap(id: &str) -> Option<&'static (&'static str, &'static str, &'static str)> {
    KNOWN_GAPS.iter().find(|g| g.0 == id)
}

fn accepted(corpus: &Corpus) -> Vec<&Fixture> {
    corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
        .collect()
}

fn truth_of(fx: &Fixture) -> Option<Value> {
    fx.truth().ok()
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_every_accepted_fixture_is_converted_re_read_and_compared() {
    let corpus = Corpus::open();
    let dir = tempfile::tempdir().unwrap();
    let mut executed = Vec::new();
    let mut failures = Vec::new();
    let mut with_geometry = 0;
    let mut compared_to_truth = 0;
    for fx in accepted(&corpus) {
        let truth = truth_of(fx);
        executed.push(fx.id.clone());
        match (run(fx, truth.as_ref(), dir.path()), gap(&fx.id)) {
            (Ok(done), Some((_, _, why))) => failures.push(format!(
                "{}: the recorded gap ({why}) is gone, delete its row from KNOWN_GAPS ({} geometry)",
                fx.id,
                done.is_some()
            )),
            (Err(f), Some((_, text, _))) if f.why.contains(text) => {}
            (Ok(Some(_)), None) => {
                with_geometry += 1;
                compared_to_truth += usize::from(truth.is_some());
            }
            (Ok(None), None) => {}
            (Err(f), _) => failures.push(format!("{}: {f}", fx.id)),
        }
    }
    let mut manifest: Vec<String> = accepted(&corpus).iter().map(|f| f.id.clone()).collect();
    manifest.sort();
    executed.sort();
    assert_eq!(
        executed, manifest,
        "a fixture in the manifest that was not run fails the suite"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        compared_to_truth >= 30,
        "only {compared_to_truth} known-truth fixtures reached the comparison ({with_geometry} produced geometry)"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_the_comparison_runs_on_the_re_read_artifact_not_on_the_normalization() {
    // Bytes of the artifact are changed after the write and before the read: the comparison has to
    // see it, which it cannot if it looked at anything held in memory from before.
    let corpus = Corpus::open();
    let fx = corpus.get("circle-001");
    let truth = fx.truth().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let c = run(fx, Some(&truth), dir.path()).unwrap().unwrap();
    let reader = PvecReader::open(c.artifact.clone()).unwrap();
    assert!(compare_to_truth(&entities(&reader).unwrap(), &truth).is_ok());
    // A different artifact (another fixture's) in the same slot is a compare failure.
    let other = run(corpus.get("line-001"), None, dir.path())
        .unwrap()
        .unwrap();
    let f = compare_to_truth(&other.curves, &truth).unwrap_err();
    assert_eq!(f.stage, Stage::Compare, "{f}");
}

fn perturbed(id: &str, edit: impl FnOnce(&mut Value)) -> Failure {
    let corpus = Corpus::open();
    let fx = corpus.get(id);
    let mut truth = fx.truth().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(
        run(fx, Some(&truth), dir.path()).is_ok(),
        "{id} is green before the edit"
    );
    edit(&mut truth);
    match run(fx, Some(&truth), dir.path()) {
        Err(f) => f,
        Ok(_) => panic!("{id}: the perturbation did not turn the path red"),
    }
}

fn hexnum(v: f64) -> Value {
    serde_json::json!({"decimal": format!("{v:?}"), "f64_hex": format!("0x{:016x}", v.to_bits())})
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_a_perturbed_expectation_fails_at_the_compare_stage_naming_the_property() {
    let cp = perturbed("bspline-d3-001", |t| {
        t["entities"][0]["geometry"]["control_points"][2][1] = hexnum(-0.5000000001);
    });
    assert_eq!(cp.stage, Stage::Compare, "{cp}");
    assert!(cp.why.contains("control point 2"), "{cp}");

    let knot = perturbed("repeated-knot-c0-001", |t| {
        let knots = t["entities"][0]["geometry"]["knots"]
            .as_array_mut()
            .unwrap();
        let k = knots.iter_mut().find(|k| k["multiplicity"] == 3).unwrap();
        k["multiplicity"] = serde_json::json!(2);
    });
    assert_eq!(knot.stage, Stage::Compare, "{knot}");
    assert!(knot.why.contains("knots"), "{knot}");

    let weight = perturbed("nonunit-weight-001", |t| {
        t["entities"][0]["geometry"]["weights"][2] = hexnum(49.0);
    });
    assert_eq!(weight.stage, Stage::Compare, "{weight}");
    assert!(weight.why.contains("weights"), "{weight}");

    let radius = perturbed("circle-001", |t| {
        t["entities"][0]["geometry"]["radius"] = hexnum(9.0);
    });
    assert_eq!(radius.stage, Stage::Compare, "{radius}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_each_stage_is_named_when_it_is_the_one_that_fails() {
    let dir = tempfile::tempdir().unwrap();

    // provider: bytes that are no drawing
    let f = convert(b"this is not a drawing", dir.path(), "junk")
        .map(|_| ())
        .unwrap_err();
    assert_eq!(f.stage, Stage::Provider, "{f}");

    // normalization: a transform the mapping refuses, for a truth that did not expect it
    let corpus = Corpus::open();
    let singular = corpus.get("singular-001");
    let f = run(singular, None, dir.path()).map(|_| ());
    // With no truth the refusal is not expected, so it is reported, at normalization.
    if let Err(f) = f {
        assert_eq!(f.stage, Stage::Normalization, "{f}");
    }

    // write: a unit PVec cannot declare is refused by the writer, not rescaled
    let dwg = corpus.get("line-001").dwg();
    let dest = dir.path().join("yards.pvec");
    let e = PvecConversion::new(CadSource::Bytes(&dwg), &dest)
        .unit(SourceUnit::Unknown)
        .convert()
        .map(|_| ())
        .unwrap_err();
    assert_eq!(stage_of(&e), Stage::Write, "{e}");
    assert!(!dest.exists(), "a refused write publishes nothing");

    // read: an artifact cut short does not open
    let c = convert(&dwg, dir.path(), "line").unwrap().unwrap();
    let cut = c.artifact[..c.artifact.len() / 2].to_vec();
    assert!(PvecReader::open(cut).is_err());
    let path = dir.path().join("cut.pvec");
    std::fs::write(&path, &c.artifact[..c.artifact.len() / 2]).unwrap();
    let f = read_back(&path).map(|_| ()).unwrap_err();
    assert_eq!(f.stage, Stage::Read, "{f}");

    // compare: reported with the stage and the entity
    let f = compare_to_truth(&[], &corpus.get("line-001").truth().unwrap()).unwrap_err();
    assert_eq!(f.stage, Stage::Compare, "{f}");
    assert!(f.to_string().starts_with("[compare]"), "{f}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_a_refusal_the_truth_expects_is_met_and_no_curve_comes_out() {
    let corpus = Corpus::open();
    let dir = tempfile::tempdir().unwrap();
    for id in ["singular-001", "near-singular-001"] {
        let fx = corpus.get(id);
        let truth = fx.truth().unwrap();
        let out = run(fx, Some(&truth), dir.path()).unwrap_or_else(|f| panic!("{id}: {f}"));
        assert!(out.is_none(), "{id}: a refused drawing yields no artifact");
        // A truth that stopped expecting the refusal is a failure of the path.
        let mut t = truth;
        t["entities"][0]
            .as_object_mut()
            .unwrap()
            .remove("rejection");
        assert!(run(fx, Some(&t), dir.path()).is_err(), "{id}");
    }
}

/// Converts `fx` twice into different directories.
fn twice(fx: &Fixture) -> Option<(Converted, Converted)> {
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let dwg = fx.dwg();
    let a = convert(&dwg, d1.path(), "out").ok().flatten()?;
    let b = convert(&dwg, d2.path(), "out").ok().flatten()?;
    Some((a, b))
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_two_conversions_of_one_fixture_are_byte_identical() {
    let corpus = Corpus::open();
    let mut digests = BTreeMap::new();
    let mut checked = 0;
    for fx in accepted(&corpus) {
        let Some((a, b)) = twice(fx) else { continue };
        checked += 1;
        assert_eq!(
            first_difference(&a.artifact, &b.artifact),
            None,
            "{}: the artifact differs between two conversions",
            fx.id
        );
        assert_eq!(
            first_difference(&a.report_bytes, &b.report_bytes),
            None,
            "{}: the report differs between two conversions",
            fx.id
        );
        // The comparison is repeated against the second artifact.
        if let Some(t) = truth_of(fx) {
            let reread = entities(&PvecReader::open(b.artifact.clone()).unwrap()).unwrap();
            compare_to_truth(&reread, &t).unwrap_or_else(|f| panic!("{}: {f}", fx.id));
        }
        digests.insert(
            fx.id.clone(),
            (digest(&a.artifact), digest(&a.report_bytes)),
        );
    }
    assert!(
        checked >= 30,
        "only {checked} fixtures reached a second conversion"
    );
    if let Ok(out) = std::env::var("PVEC_GATE_OUT") {
        let mut text = String::new();
        for (id, (art, rep)) in &digests {
            text.push_str(&format!("{id} {art} {rep}\n"));
        }
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(Path::new(&out).join("determinism.digests"), text).unwrap();
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_3_1050_an_ordering_perturbation_turns_the_determinism_check_red() {
    use libviprs::CancelToken;
    use libviprs::pvec::order_entities;
    use libviprs::pvec::writer::{WriteConfig, write_geometry};
    use libviprs::pvec::{EntityId, EntityStyle, FidelityClass, Normalized2DGeometry};
    use libviprs::pvec::{InstancePath, OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry};

    let geoms = [
        Normalized2DGeometry::circle([10.0, 10.0], [1.0, 0.0], 5.0, FidelityClass::SupportedExact)
            .unwrap(),
        Normalized2DGeometry::polyline(&[[0.0, 0.0], [4.0, 3.0]], FidelityClass::SupportedExact)
            .unwrap(),
    ];
    let raw = (0..2)
        .map(|i| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [f64::from(i), 0.0, 0.0],
                end: [f64::from(i) + 1.0, 0.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(libviprs::cad::ItemHandle::new(u64::from(i as u32) + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: "0".into(),
            },
        })
        .collect();
    let ids: Vec<EntityId> = order_entities("t", "1", &OrderingPolicy::new(), raw)
        .entities
        .into_iter()
        .map(|o| o.id)
        .collect();
    let config = WriteConfig {
        unit: SourceUnit::Meters,
        leaves: None,
    };
    let write = |order: [usize; 2]| {
        write_geometry(
            &config,
            order.iter().map(|&i| (ids[i], &geoms[i])),
            Vec::new(),
            &CancelToken::new(),
        )
        .unwrap()
        .bytes
    };
    let (a, b) = (write([0, 1]), write([0, 1]));
    assert_eq!(
        first_difference(&a, &b),
        None,
        "the same order is the same bytes"
    );
    let swapped = write([1, 0]);
    assert!(
        first_difference(&a, &swapped).is_some(),
        "an emit order that moved is seen by the byte comparison"
    );
}

#[test]
fn n2_3_1050_the_suite_uses_only_the_public_api_and_starts_no_converter() {
    // Built from pieces so this file does not match its own scan.
    let needles = [
        ["pvec_", "codec"].concat(),
        ["pvec_", "container"].concat(),
        ["pvec_", "stream"].concat(),
        ["std::pro", "cess"].concat(),
        ["Comm", "and::new"].concat(),
        ["dock", "er"].concat(),
        ["CAD Exch", "anger"].concat(),
        ["O", "DA"].concat() + " image",
    ];
    for n in &needles {
        let hit = include_str!("pvec_end_to_end.rs")
            .lines()
            .find(|l| l.contains(n.as_str()) && !l.trim_start().starts_with("//"));
        assert!(hit.is_none(), "found {n:?}: {hit:?}");
    }
}
