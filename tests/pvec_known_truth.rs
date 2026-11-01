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
//! Needs `--features pvec-conformance`.

#![cfg(feature = "pvec-conformance")]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use pvec_gold::{Corpus, EvidenceClass, Fixture};
use serde_json::{Value, json};

// ------------------------------------------------------------------------------------ the suite

/// Disagreements that are known and recorded, one per line: fixture, the text of the problem, and
/// where it is written down. A gap is waived only while it is still real: when a fix makes the
/// problem disappear the suite fails until its row is deleted, so this list cannot go stale.
const KNOWN_GAPS: &[(&str, &str, &str)] = &[(
    "bezier-d3-001",
    "the mapping lands in Some(BSpline), the truth is bezier",
    "docs/spikes/dwg-mapping.md M21: a single-span SPLINE maps to a B-spline and is graded equivalent (#1375)",
)];

/// Runs one fixture against `truth` and returns every disagreement that is not a recorded gap,
/// each naming the property, plus a line for every recorded gap that no longer exists.
fn run(fx: &Fixture, truth: &Value) -> Vec<String> {
    let all = compare(fx, truth);
    let waived = |p: &String| {
        KNOWN_GAPS
            .iter()
            .any(|(id, text, _)| *id == fx.id && p.contains(text))
    };
    let mut out: Vec<String> = all.iter().filter(|p| !waived(p)).cloned().collect();
    // Only the unperturbed record can say a gap is gone.
    if truth == &fx.json("expected.math.json") {
        for (id, text, why) in KNOWN_GAPS.iter().filter(|g| g.0 == fx.id) {
            if !all.iter().any(|p| p.contains(text)) {
                out.push(format!(
                    "the recorded gap {id}/{text:?} ({why}) is gone, delete its row from KNOWN_GAPS"
                ));
            }
        }
    }
    out
}

fn compare(fx: &Fixture, truth: &Value) -> Vec<String> {
    // A truth record this suite cannot read is a defect of the suite, reported with the fixture.
    std::panic::catch_unwind(|| mapping::check(fx, truth)).unwrap_or_else(|e| {
        let why = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(ToString::to_string))
            .unwrap_or_default();
        vec![format!("the comparison itself broke: {why}")]
    })
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
#[cfg_attr(miri, ignore)] // reads the vendored corpus from disk
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

#[test]
fn n2_1_1050_a_wrong_world_result_a_wrong_matrix_and_a_wrong_layer_are_caught() {
    let world = perturbed("nonuniform-scale-circle-001", |t| {
        t["entities"][0]["world"]["geometry"]["semi_minor"] = hexnum(2.5);
    });
    assert!(world.iter().any(|p| p.contains("world")), "{world:?}");
    let matrix = perturbed("translation-001", |t| {
        t["entities"][0]["transform"]["matrix"][3] = hexnum(3.75);
    });
    assert!(matrix.iter().any(|p| p.contains("transform")), "{matrix:?}");
    let layer = perturbed("layer-state-001", |t| {
        t["entities"][1]["layer"]["on"] = json!(true);
    });
    assert!(layer.iter().any(|p| p.contains("layer")), "{layer:?}");
}

#[test]
fn n2_1_1050_a_refused_transform_emits_no_curve_and_a_curve_is_never_expected() {
    let corpus = Corpus::open();
    for id in ["singular-001", "near-singular-001"] {
        let fx = corpus.get(id);
        assert!(fx.expects_rejection, "{id}");
        assert!(run(fx, &fx.truth().unwrap()).is_empty(), "{id}");
    }
    // If the truth stopped asking for a refusal, the (still refused) drawing would disagree.
    let problems = perturbed("singular-001", |t| {
        t["entities"][0]
            .as_object_mut()
            .unwrap()
            .remove("rejection");
    });
    assert!(
        !problems.is_empty(),
        "a refusal the truth does not expect is a failure"
    );
}

#[test]
fn n2_1_1050_every_recorded_gap_names_a_fixture_in_the_corpus() {
    let corpus = Corpus::open();
    for (id, _, why) in KNOWN_GAPS {
        assert_eq!(corpus.get(id).class, EvidenceClass::KnownTruthExact, "{id}");
        assert!(why.contains("docs/spikes/dwg-mapping.md"), "{id}: {why}");
    }
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
#[cfg_attr(miri, ignore)] // reads the vendored corpus from disk
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
    use crate::pvec_gold::{num, nums, points};
    use libviprs::cad::{CadSource, ItemHandle};
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::dwg_basic::{self, BasicKind, ExactBasic, SegmentKind};
    use libviprs::pvec::dwg_spline::{self, ExactSpline, SplineForm};
    use libviprs::pvec::flatten::{
        BlockResolver, FlatEntity, FlattenBounds, FlattenError, LineageStore,
    };
    use libviprs::pvec::mapping::{NormalizedKind, kind_of};
    use libviprs::pvec::provenance::NoProvenance;
    use libviprs::pvec::transform::{ComposedTransform, TransformReason, promote};
    use libviprs::pvec::{
        DwgProvider, EntityScope, FidelityClass, InstanceStep, LayoutKind, Normalized2DGeometry,
        RawGeometry, ViewProfile,
    };
    use pvec_geometry::{Curve, Curve2d};
    use pvec_numeric::Policy;

    const POLICY: Policy = Policy::V1;

    type Problems = Vec<String>;

    fn near(a: f64, b: f64, scale: f64) -> bool {
        POLICY.near(a, b, scale)
    }

    fn near_pt(a: [f64; 2], b: [f64; 2], scale: f64) -> bool {
        POLICY.near_point(a, b, scale)
    }

    /// The size of the features a truth record talks about, for `Policy::near`.
    fn scale_of(values: &[f64]) -> f64 {
        values.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0)
    }

    pub fn check(fx: &Fixture, truth: &Value) -> Problems {
        let mut out = Problems::new();
        let bytes = fx.dwg();
        let provider = match AcadrustProvider::open(CadSource::Bytes(&bytes)) {
            Ok(p) => p,
            Err(e) => return vec![format!("the drawing does not open: {e}")],
        };
        let layouts = provider.layouts().expect("layouts");
        let model = layouts
            .iter()
            .find(|l| l.kind == LayoutKind::Model)
            .expect("model space");
        let resolver = BlockResolver::new(
            &provider,
            FlattenBounds::from_profile(&ViewProfile::default_profile()),
        );
        let flat = resolver.flatten(
            &EntityScope::Layout(model.id),
            &fx.id,
            &model.name,
            &mut NoProvenance,
            &mut LineageStore::new(),
        );
        let truth_entities = truth["entities"].as_array().expect("entities");

        // A truth record that expects a refusal: no curve may come out.
        if let Some(rej) = truth_entities.iter().find(|e| e.get("rejection").is_some()) {
            let class = rej["rejection"]["classification"].as_str().unwrap_or("");
            match &flat {
                Err(FlattenError::Compose(e)) => {
                    // N1.5 freezes one code for singular and near-singular (NP-19), so the
                    // finer truth names SINGULAR_TRANSFORM / NEAR_SINGULAR_TRANSFORM both
                    // land on it. What both demand is that no curve is emitted.
                    if e.reason() != Some(TransformReason::Singular) {
                        out.push(format!(
                            "rejection {class}: refused with {:?}, not TRANSFORM_SINGULAR",
                            e.reason()
                        ));
                    }
                }
                Err(other) => out.push(format!("rejection {class}: wrong refusal: {other}")),
                Ok(f) => out.push(format!(
                    "rejection {class}: {} entities were emitted, none may be",
                    f.entities.len()
                )),
            }
            return out;
        }
        let flat = match flat {
            Ok(f) => f,
            Err(e) => return vec![format!("flatten refused a drawing the truth accepts: {e}")],
        };
        if !flat.skipped.is_empty() {
            out.push(format!("{} inserts were skipped", flat.skipped.len()));
        }

        let derived = |e: &&Value| e["id"].as_str().is_some_and(|i| i.contains("-equivalent-"));
        let primary: Vec<&Value> = truth_entities.iter().filter(|e| !derived(e)).collect();
        let extra: Vec<&Value> = truth_entities.iter().filter(derived).collect();
        if primary.len() != flat.entities.len() {
            out.push(format!(
                "entity count: the truth has {}, the mapping produced {}",
                primary.len(),
                flat.entities.len()
            ));
            return out;
        }
        for (t, f) in primary.iter().zip(&flat.entities) {
            let id = t["id"].as_str().unwrap_or("?");
            let mut p = Problems::new();
            entity(&mut p, &provider, t, f, &extra);
            out.extend(p.into_iter().map(|m| format!("{id}: {m}")));
        }
        out
    }

    fn entity(
        out: &mut Problems,
        provider: &AcadrustProvider,
        t: &Value,
        f: &FlatEntity,
        extra: &[&Value],
    ) {
        let class = t["class"].as_str().unwrap_or("?");
        let g = &t["geometry"];
        match class {
            "line" => line(out, t, f),
            "circle" | "circular_arc" | "ellipse" | "elliptical_arc" => conic(out, class, t, f),
            "polyline" => polyline(out, t, f, extra),
            "bspline" | "nurbs" | "bezier" => spline(out, class, t, f),
            other => out.push(format!("class: no comparison for truth class {other}")),
        }
        if let Some(layer) = t.get("layer") {
            layer_state(out, provider, layer, f);
        }
        let _ = g;
        transform_and_world(out, t, f);
    }

    // -------------------------------------------------------------------------------- layer

    fn layer_state(out: &mut Problems, provider: &AcadrustProvider, layer: &Value, f: &FlatEntity) {
        let name = layer["name"].as_str().unwrap_or("?");
        if f.entity.provenance.layer != name {
            out.push(format!(
                "layer: the entity is on {:?}, the truth says {name}",
                f.entity.provenance.layer
            ));
        }
        let layers = provider.layers().expect("layer table");
        match layers.iter().find(|l| l.name == name) {
            None => out.push(format!("layer: {name} is not in the layer table")),
            Some(l) => {
                if l.off == layer["on"].as_bool().unwrap_or(true) {
                    out.push(format!(
                        "layer: {name} off flag is {}, truth on={}",
                        l.off, layer["on"]
                    ));
                }
                if l.frozen != layer["frozen"].as_bool().unwrap_or(false) {
                    out.push(format!(
                        "layer: {name} frozen flag is {}, truth {}",
                        l.frozen, layer["frozen"]
                    ));
                }
            }
        }
    }

    // ------------------------------------------------------------------------------- basics

    fn basic(out: &mut Problems, f: &FlatEntity, want: BasicKind) -> Option<ExactBasic> {
        match dwg_basic::extract(&f.entity) {
            Ok(b) if b.kind == want => Some(b),
            Ok(b) => {
                out.push(format!(
                    "class: extracted {:?}, the truth is {want:?}",
                    b.kind
                ));
                None
            }
            Err(e) => {
                out.push(format!("class: extraction refused: {e}"));
                None
            }
        }
    }

    /// The fidelity rule of the known-truth tier, from the truth record alone: the mapping may
    /// claim `SupportedExact` only if it reproduces the truth's parameter domain exactly, and
    /// `SupportedEquivalent` is always a true (weaker) claim. Anything below that is wrong for a
    /// fixture whose geometry the truth states.
    fn fidelity(out: &mut Problems, t: &Value, got: FidelityClass, domain: Option<(f64, f64)>) {
        if !matches!(
            got,
            FidelityClass::SupportedExact | FidelityClass::SupportedEquivalent
        ) {
            out.push(format!("fidelity: {got:?}, the truth states the geometry"));
            return;
        }
        let want = (
            num(&t["parameter_domain"]["start"]),
            num(&t["parameter_domain"]["end"]),
        );
        if let Some(d) = domain
            && got == FidelityClass::SupportedExact
            && d != want
        {
            out.push(format!(
                "fidelity: SupportedExact claimed but the parameter domain {d:?} is not the truth's {want:?}"
            ));
        }
    }

    fn line(out: &mut Problems, t: &Value, f: &FlatEntity) {
        let Some(b) = basic(out, f, BasicKind::Line) else {
            return;
        };
        let g = &t["geometry"];
        let (s, e) = (nums(&g["start"]), nums(&g["end"]));
        let RawGeometry::Line { start, end } = &f.entity.geometry else {
            out.push("class: the decoder did not report a line".into());
            return;
        };
        if start[..2] != s[..2] || end[..2] != e[..2] {
            out.push(format!(
                "coordinates: decoded {start:?} {end:?}, truth {s:?} {e:?}"
            ));
        }
        let d = b.curve.domain();
        let (p0, p1) = (b.curve.eval(d.lo()).unwrap(), b.curve.eval(d.hi()).unwrap());
        let scale = scale_of(&[s[0], s[1], e[0], e[1]]);
        if !near_pt([p0.x, p0.y], [s[0], s[1]], scale)
            || !near_pt([p1.x, p1.y], [e[0], e[1]], scale)
        {
            out.push(format!(
                "coordinates: the curve runs {p0:?} to {p1:?}, truth {s:?} to {e:?}"
            ));
        }
        fidelity(out, t, b.classification.class(), Some((d.lo(), d.hi())));
    }

    fn conic(out: &mut Problems, class: &str, t: &Value, f: &FlatEntity) {
        let want = match class {
            "circle" => BasicKind::Circle,
            "circular_arc" => BasicKind::Arc,
            "ellipse" => BasicKind::Ellipse,
            _ => BasicKind::EllipticalArc,
        };
        let Some(b) = basic(out, f, want) else { return };
        let g = &t["geometry"];
        let c = nums(&g["center"]);
        match &f.entity.geometry {
            RawGeometry::Circle { center, radius, .. }
            | RawGeometry::Arc { center, radius, .. } => {
                if center[..2] != c[..2] {
                    out.push(format!("centre: decoded {center:?}, truth {c:?}"));
                }
                if *radius != num(&g["radius"]) {
                    out.push(format!(
                        "radius: decoded {radius}, truth {}",
                        num(&g["radius"])
                    ));
                }
            }
            RawGeometry::Ellipse {
                center,
                major_axis,
                minor_ratio,
                ..
            } => {
                if center[..2] != c[..2] {
                    out.push(format!("centre: decoded {center:?}, truth {c:?}"));
                }
                let m = nums(&g["major_axis"]);
                let semi_major = num(&g["semi_major"]);
                let len = major_axis[0].hypot(major_axis[1]);
                let scale = scale_of(&[semi_major]);
                if !near(len, semi_major, scale) {
                    out.push(format!("semi-major: decoded {len}, truth {semi_major}"));
                }
                if !near(len * minor_ratio, num(&g["semi_minor"]), scale) {
                    out.push(format!(
                        "semi-minor: decoded {}, truth {}",
                        len * minor_ratio,
                        num(&g["semi_minor"])
                    ));
                }
                if !near_pt(
                    [major_axis[0] / len, major_axis[1] / len],
                    [m[0], m[1]],
                    1.0,
                ) {
                    out.push(format!("major axis: decoded {major_axis:?}, truth {m:?}"));
                }
            }
            _ => out.push("class: the decoder reported another geometry".into()),
        }
        if let (Some((a, e)), "circular_arc" | "elliptical_arc") = (b.source_range, class) {
            // Parameters exactly as stored (`start_angle`), and the sweep they imply.
            let start = num(&g["start_angle"]);
            let sweep = num(&g["sweep_angle"]);
            if a != start {
                out.push(format!("start angle: decoded {a}, truth {start}"));
            }
            let got = (e - a).rem_euclid(std::f64::consts::TAU);
            if !near(got, sweep, std::f64::consts::TAU) {
                out.push(format!("sweep: decoded {got}, truth {sweep}"));
            }
        }
        // Where the curve actually runs, from the truth's parameters (truth side).
        let normal = nums(&g["normal"]);
        let _ = normal;
        let d = b.curve.domain();
        let sc = scale_of(&c);
        let pt = |u: f64| {
            let p = b.curve.eval(u).unwrap();
            [p.x, p.y]
        };
        let (rx, ry) = radii(class, g);
        let at = |theta: f64| {
            let axis = x_axis(class, g);
            [
                c[0] + rx * theta.cos() * axis[0] - ry * theta.sin() * axis[1],
                c[1] + rx * theta.cos() * axis[1] + ry * theta.sin() * axis[0],
            ]
        };
        match class {
            "circle" | "ellipse" => {
                for k in 0..8 {
                    let u = d.lo() + d.width() * f64::from(k) / 8.0;
                    let p = pt(u);
                    if !on_ellipse(p, c[0..2].try_into().unwrap(), x_axis(class, g), rx, ry, sc) {
                        out.push(format!(
                            "geometry: the point {p:?} is off the truth's {class}"
                        ));
                        break;
                    }
                }
            }
            _ => {
                let (a, sweep) = (num(&g["start_angle"]), num(&g["sweep_angle"]));
                let (want0, want1) = (at(a), at(a + sweep));
                let (got0, got1) = (pt(d.lo()), pt(d.hi()));
                let sc = scale_of(&[rx, ry, c[0], c[1]]);
                if !near_pt(got0, want0, sc) || !near_pt(got1, want1, sc) {
                    out.push(format!(
                        "geometry: the arc runs {got0:?} to {got1:?}, truth {want0:?} to {want1:?}"
                    ));
                }
            }
        }
        let d = b.curve.domain();
        fidelity(out, t, b.classification.class(), Some((d.lo(), d.hi())));
    }

    fn radii(class: &str, g: &Value) -> (f64, f64) {
        match class {
            "circle" | "circular_arc" => (num(&g["radius"]), num(&g["radius"])),
            _ => (num(&g["semi_major"]), num(&g["semi_minor"])),
        }
    }

    fn x_axis(class: &str, g: &Value) -> [f64; 2] {
        let v = match class {
            "circle" | "circular_arc" => nums(&g["x_axis"]),
            _ => nums(&g["major_axis"]),
        };
        let n = v[0].hypot(v[1]);
        [v[0] / n, v[1] / n]
    }

    /// Whether `p` lies on the ellipse (truth side: the implicit equation of the truth record).
    fn on_ellipse(p: [f64; 2], c: [f64; 2], u: [f64; 2], a: f64, b: f64, scale: f64) -> bool {
        let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
        let (s, t) = (dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0]);
        near((s / a).hypot(t / b), 1.0, scale.max(1.0))
    }

    fn polyline(out: &mut Problems, t: &Value, f: &FlatEntity, extra: &[&Value]) {
        let Some(b) = basic(out, f, BasicKind::Polyline) else {
            return;
        };
        let g = &t["geometry"];
        let want = points(&g["vertices"]);
        let bulges = nums(&g["bulges"]);
        let RawGeometry::Polyline {
            vertices, closed, ..
        } = &f.entity.geometry
        else {
            out.push("class: the decoder did not report a polyline".into());
            return;
        };
        if vertices.len() != want.len() {
            out.push(format!(
                "vertex count: decoded {}, truth {}",
                vertices.len(),
                want.len()
            ));
            return;
        }
        for (i, (v, w)) in vertices.iter().zip(&want).enumerate() {
            if v.x != w[0] || v.y != w[1] {
                out.push(format!(
                    "vertex {i}: decoded ({}, {}), truth ({}, {})",
                    v.x, v.y, w[0], w[1]
                ));
            }
            if v.bulge != bulges[i] {
                out.push(format!(
                    "bulge {i}: decoded {}, truth {}",
                    v.bulge, bulges[i]
                ));
            }
        }
        let want_closed = t["closure"]["closed_by_endpoint"]
            .as_bool()
            .unwrap_or(false);
        if *closed != want_closed || b.closed != want_closed {
            out.push(format!(
                "closed state: decoded {closed}, mapped {}, truth {want_closed}",
                b.closed
            ));
        }
        // Segment kinds follow the bulge signs (which the truth fixes).
        let n_seg = if want_closed {
            want.len()
        } else {
            want.len() - 1
        };
        let kinds: Vec<SegmentKind> = bulges[..n_seg]
            .iter()
            .map(|&k| {
                if k == 0.0 {
                    SegmentKind::Straight
                } else if k > 0.0 {
                    SegmentKind::ArcCounterClockwise
                } else {
                    SegmentKind::ArcClockwise
                }
            })
            .collect();
        if b.segments != kinds {
            out.push(format!(
                "bulge: segments {:?}, the truth's bulges give {kinds:?}",
                b.segments
            ));
        }
        // The arc a bulge describes, when the truth carries one.
        for arc in extra {
            let ag = &arc["geometry"];
            let Curve::Composite(comp) = &b.curve else {
                out.push(format!(
                    "bulge: expected a composite, got {:?}",
                    b.curve.kind()
                ));
                continue;
            };
            let Some(Curve::Trimmed(tr)) = comp
                .segments()
                .iter()
                .find(|s| matches!(s, Curve::Trimmed(_)))
            else {
                out.push("bulge: no arc segment in the composite".into());
                continue;
            };
            let Curve::Circle(circle) = tr.basis() else {
                out.push("bulge: the arc is not over a circle".into());
                continue;
            };
            let c = nums(&ag["center"]);
            let r = num(&ag["radius"]);
            let o = circle.placement().origin();
            let sc = scale_of(&[c[0], c[1], r]);
            if !near_pt([o.x, o.y], [c[0], c[1]], sc) {
                out.push(format!(
                    "bulge: arc centre ({}, {}), truth ({}, {})",
                    o.x, o.y, c[0], c[1]
                ));
            }
            if !near(circle.radius(), r, sc) {
                out.push(format!("bulge: arc radius {}, truth {r}", circle.radius()));
            }
            let (a, sweep) = (num(&ag["start_angle"]), num(&ag["sweep_angle"]));
            let trim = tr.trim();
            if !near(trim.start, a, std::f64::consts::TAU)
                || !near(trim.span, sweep, std::f64::consts::TAU)
            {
                out.push(format!(
                    "bulge: arc start {} span {}, truth {a} {sweep}",
                    trim.start, trim.span
                ));
            }
        }
        let curve_len = match &b.curve {
            Curve::Polyline(p) => Some(p.vertices().len()),
            _ => None,
        };
        if let Some(n) = curve_len {
            let expect = if want_closed {
                want.len() + 1
            } else {
                want.len()
            };
            if n != expect {
                out.push(format!(
                    "vertex count: the curve has {n}, expected {expect}"
                ));
            }
        }
        let class = b.classification.class();
        let all_straight = kinds.iter().all(|k| *k == SegmentKind::Straight);
        let d = b.curve.domain();
        let dom = all_straight.then_some((d.lo(), d.hi()));
        fidelity(out, t, class, dom);
    }

    // ------------------------------------------------------------------------------ splines

    fn spline(out: &mut Problems, class: &str, t: &Value, f: &FlatEntity) {
        let g = &t["geometry"];
        let degree = g["degree"].as_u64().unwrap_or(0) as u32;
        let cps = points(&g["control_points"]);
        // Knots: truth carries (value, multiplicity) pairs, a bezier carries none (clamped 0..1).
        let want_knots: Vec<f64> = match g.get("knots").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .flat_map(|k| {
                    let v = num(&k["value"]);
                    let m = k["multiplicity"].as_u64().unwrap_or(1) as usize;
                    std::iter::repeat_n(v, m)
                })
                .collect(),
            None => {
                let n = degree as usize + 1;
                let mut v = vec![0.0; n];
                v.extend(std::iter::repeat_n(1.0, n));
                v
            }
        };
        let cl = &t["closure"];
        let (lo, hi) = (
            num(&t["parameter_domain"]["start"]),
            num(&t["parameter_domain"]["end"]),
        );

        // What the decoder reports, which every spline mapping starts from.
        let RawGeometry::Spline {
            degree: d,
            control_points,
            knots,
            weights,
            periodic,
            closed,
        } = &f.entity.geometry
        else {
            out.push("class: the decoder did not report a spline".into());
            return;
        };
        if *d != degree {
            out.push(format!("degree: decoded {d}, truth {degree}"));
        }
        if control_points.len() != cps.len() {
            out.push(format!(
                "control points: decoded {}, truth {}",
                control_points.len(),
                cps.len()
            ));
        } else {
            for (i, (a, w)) in control_points.iter().zip(&cps).enumerate() {
                if a[0] != w[0] || a[1] != w[1] {
                    out.push(format!(
                        "control point {i}: decoded ({}, {}), truth ({}, {})",
                        a[0], a[1], w[0], w[1]
                    ));
                }
            }
        }
        match knots.present() {
            Some(k) if *k == want_knots => {}
            other => out.push(format!(
                "knots: decoded {other:?}, truth (with multiplicities) {want_knots:?}"
            )),
        }
        // A non-rational spline carries no weight array at all, not a row of ones.
        match (weights.present(), g.get("weights")) {
            (None, None) => {}
            (Some(w), Some(want)) if *w == nums(want) => {}
            (Some(w), Some(want)) => out.push(format!("weights: decoded {w:?}, truth {want}")),
            (None, Some(_)) => {
                out.push("weights: the truth has weights and none were decoded".into())
            }
            (Some(_), None) => out.push("weights: a weight array for a non-rational spline".into()),
        }
        if periodic.present() != cl["periodic"].as_bool().as_ref() {
            out.push(format!(
                "periodic state: decoded {periodic:?}, truth {}",
                cl["periodic"]
            ));
        }
        if *closed != cl["closed_by_knots"].as_bool().unwrap_or(false) {
            out.push(format!(
                "closed state: decoded {closed}, truth closed_by_knots {}",
                cl["closed_by_knots"]
            ));
        }
        let by_endpoint = control_points
            .first()
            .zip(control_points.last())
            .is_some_and(|(a, b)| a == b);
        if by_endpoint != cl["closed_by_endpoint"].as_bool().unwrap_or(false) {
            out.push(format!(
                "closed state: endpoints coincide {by_endpoint}, truth closed_by_endpoint {}",
                cl["closed_by_endpoint"]
            ));
        }

        // What the mapping makes of it.
        let s: ExactSpline = match dwg_spline::extract(&f.entity) {
            Ok(s) => s,
            Err(e) => {
                out.push(format!("mapping: spline extraction refused: {e}"));
                return;
            }
        };
        if s.degree != degree || s.knots != want_knots {
            out.push(format!("knots: mapped {:?}, truth {want_knots:?}", s.knots));
        }
        if s.control_points
            .iter()
            .zip(&cps)
            .any(|(a, w)| a[0] != w[0] || a[1] != w[1])
        {
            out.push("control points: the mapping moved one".into());
        }
        match (&s.form, g.get("weights")) {
            (SplineForm::NonRational, None) => {}
            (SplineForm::Rational(w), Some(want)) if *w == nums(want) => {}
            _ => out.push("weights: the mapping's form disagrees with the truth".into()),
        }
        if s.domain.lo != lo || s.domain.hi != hi {
            out.push(format!(
                "parameter domain: mapped [{}, {}], truth [{lo}, {hi}]",
                s.domain.lo, s.domain.hi
            ));
        }
        if s.periodic != cl["periodic"].as_bool().unwrap_or(false) || s.closed != *closed {
            out.push("closed state: the mapping changed a flag".into());
        }
        // The mapped record and the class it lands in (profile sections 11 and 12).
        let pts: Vec<[f64; 2]> = s.control_points.iter().map(|p| [p[0], p[1]]).collect();
        let record = match &s.form {
            SplineForm::NonRational => Normalized2DGeometry::b_spline(
                degree as usize,
                &pts,
                &s.knots,
                FidelityClass::SupportedExact,
            ),
            SplineForm::Rational(w) => Normalized2DGeometry::rational_b_spline(
                degree as usize,
                &pts,
                &s.knots,
                w,
                FidelityClass::SupportedExact,
            ),
            _ => return,
        };
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                out.push(format!("class: the record cannot be built: {e}"));
                return;
            }
        };
        let kind = kind_of(&record);
        let want_kind = match class {
            "nurbs" => NormalizedKind::RationalBSpline,
            "bspline" => NormalizedKind::BSpline,
            _ => NormalizedKind::Bezier,
        };
        if kind != Some(want_kind) {
            out.push(format!(
                "class: the mapping lands in {kind:?}, the truth is {class}"
            ));
        }
    }

    // ------------------------------------------------------------- transform, world result

    fn transform_and_world(out: &mut Problems, t: &Value, f: &FlatEntity) {
        let structure = t["structure"].as_str().unwrap_or("CURVE_ONLY");
        let m = f.matrix();
        if structure == "CURVE_ONLY" {
            if m != [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
                out.push(format!("transform: a curve-only entity carries {m:?}"));
            }
            return;
        }
        let composed = t
            .get("transform")
            .or_else(|| t.get("transform_chain").map(|c| &c["composed"]))
            .expect("a transformed entity states its transform");
        let tm = nums(&composed["matrix"]);
        let want = [tm[0], tm[1], tm[4], tm[5], tm[3], tm[7]];
        let sc = scale_of(&want);
        if !m.iter().zip(&want).all(|(a, b)| near(*a, *b, sc)) {
            out.push(format!("transform: composed {m:?}, truth {want:?}"));
        }
        // The truth states the determinant of its 4x4 matrix, so the z scale is part of it.
        let det = (m[0] * m[3] - m[1] * m[2]) * tm[10];
        if !near(det, num(&composed["determinant"]), sc) {
            out.push(format!(
                "transform: determinant {det}, truth {}",
                num(&composed["determinant"])
            ));
        }
        let world = &t["world"];
        // Promote through the composed map, the way the conversion does.
        let Some(record) = promoted(out, f) else {
            return;
        };
        let Some(curve) = record.geometry() else {
            out.push("world: the promoted record carries no geometry".into());
            return;
        };
        let wclass = world["class"].as_str().unwrap_or("?");
        let wg = &world["geometry"];
        let d = curve.domain();
        let pt = |u: f64| {
            let p = curve.eval(u).unwrap();
            [p.x, p.y]
        };
        match wclass {
            "line" => {
                let (s, e) = (nums(&wg["start"]), nums(&wg["end"]));
                let sc = scale_of(&[s[0], s[1], e[0], e[1]]);
                if !near_pt(pt(d.lo()), [s[0], s[1]], sc) || !near_pt(pt(d.hi()), [e[0], e[1]], sc)
                {
                    out.push(format!(
                        "world: line runs {:?} to {:?}, truth {s:?} to {e:?}",
                        pt(d.lo()),
                        pt(d.hi())
                    ));
                }
                if kind_of(&record) != Some(NormalizedKind::Line) {
                    out.push(format!("world class: {:?}, truth line", kind_of(&record)));
                }
            }
            "circle" | "ellipse" => {
                let c = nums(&wg["center"]);
                let (rx, ry) = radii(wclass, wg);
                let u = x_axis(wclass, wg);
                for k in 0..8 {
                    let p = pt(d.lo() + d.width() * f64::from(k) / 8.0);
                    if !on_ellipse(p, [c[0], c[1]], u, rx, ry, scale_of(&[c[0], c[1], rx, ry])) {
                        out.push(format!(
                            "world: the point {p:?} is off the truth's {wclass}"
                        ));
                        break;
                    }
                }
                let want = if wclass == "circle" {
                    NormalizedKind::Circle
                } else {
                    NormalizedKind::Ellipse
                };
                if kind_of(&record) != Some(want) {
                    out.push(format!("world class: {:?}, truth {wclass} (a circle under this map must become an ellipse)", kind_of(&record)));
                }
            }
            "circular_arc" => {
                let c = nums(&wg["center"]);
                let r = num(&wg["radius"]);
                let ax = nums(&wg["x_axis"]);
                let nz = nums(&wg["normal"]);
                let (a, sweep) = (num(&wg["start_angle"]), num(&wg["sweep_angle"]));
                // y axis = normal x x_axis, so a reflected frame (normal -Z) is honoured.
                let ay = [-nz[2] * ax[1], nz[2] * ax[0]];
                let at = |th: f64| {
                    [
                        c[0] + r * (th.cos() * ax[0] + th.sin() * ay[0]),
                        c[1] + r * (th.cos() * ax[1] + th.sin() * ay[1]),
                    ]
                };
                let sc = scale_of(&[c[0], c[1], r]);
                let (w0, w1) = (at(a), at(a + sweep));
                if !near_pt(pt(d.lo()), w0, sc) || !near_pt(pt(d.hi()), w1, sc) {
                    out.push(format!("world: arc runs {:?} to {:?}, truth {w0:?} to {w1:?} (end points and their order)", pt(d.lo()), pt(d.hi())));
                }
                if kind_of(&record) != Some(NormalizedKind::CircularArc) {
                    out.push(format!(
                        "world class: {:?}, truth circular_arc",
                        kind_of(&record)
                    ));
                }
            }
            other => out.push(format!("world: no comparison for class {other}")),
        }
        let want_orientation = t["orientation"]["parameter_direction"].as_str();
        if want_orientation != Some("START_TO_END") && want_orientation.is_some() {
            out.push(format!("orientation: truth {want_orientation:?}"));
        }
    }

    /// The leaf, mapped and promoted through its composed instance transform.
    fn promoted(out: &mut Problems, f: &FlatEntity) -> Option<Normalized2DGeometry> {
        let curve = match dwg_basic::extract(&f.entity) {
            Ok(b) => b.curve,
            Err(e) => {
                out.push(format!("world: the leaf cannot be mapped: {e}"));
                return None;
            }
        };
        let record =
            match Normalized2DGeometry::from_curve(curve, FidelityClass::SupportedEquivalent) {
                Ok(r) => r,
                Err(e) => {
                    out.push(format!("world: {e}"));
                    return None;
                }
            };
        let step = InstanceStep {
            block_name: "composed".to_string(),
            insert_handle: ItemHandle::new(0),
        };
        let composed = match ComposedTransform::identity().push_inner_coefficients(step, f.matrix())
        {
            Ok(c) => c,
            Err(e) => {
                out.push(format!("world: the composed map is refused: {e}"));
                return None;
            }
        };
        match promote(&record, &composed) {
            Ok(p) => Some(p.record().clone()),
            Err(e) => {
                out.push(format!("world: promotion failed: {e:?}"));
                None
            }
        }
    }
}
