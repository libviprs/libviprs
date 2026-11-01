//! N2.5 (#1055): geometric equivalence under the shared numeric policy.
//!
//! Where two representations are the same locus without being structurally identical, comparison
//! needs tolerance and conditioning rules. Those rules are PVec's (`pvec_numeric::Policy::V1`,
//! `docs/NUMERIC_POLICY.md`), and libviprs introduces no epsilon of its own: every comparison in
//! this file goes through `Policy::near` (NP-04 the tolerance triple, NP-05 equality), and the
//! derivative cases follow NP-16 (continuity and one-sided derivatives). The guard test at the end
//! holds the normalization and conformance sources to that.
//!
//! The expectations are the pinned corpus' `expected.math.json` records. The few numbers computed
//! in this file are labelled `truth side`: they turn a truth record's parameters into points.
//!
//! Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

#[path = "common/pvec_gold_decode.rs"]
mod decode;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use decode::{near, near_pt, on_ellipse, scale_of};
use libviprs::pvec::dwg_basic;
use libviprs::pvec::mapping::ConformanceClass;
use libviprs::pvec::{
    EntityStyle, InstancePath, Point3, ProvenanceStub, RawEntity, RawGeometry, RawVertex,
};
use pvec_geometry::{Curve, Curve2d, Point2};
use pvec_gold::{Corpus, num, nums};
use pvec_numeric::Side;
use serde_json::Value;

const TAU: f64 = std::f64::consts::TAU;

fn arc_of_composite(c: &Curve) -> (Point2, f64, f64, f64, bool) {
    let Curve::Composite(comp) = c else {
        panic!("a bulge polyline is a composite, got {:?}", c.kind());
    };
    let Some(Curve::Trimmed(t)) = comp
        .segments()
        .iter()
        .find(|s| matches!(s, Curve::Trimmed(_)))
    else {
        panic!("no arc segment");
    };
    let Curve::Circle(circle) = t.basis() else {
        panic!("not over a circle");
    };
    let trim = t.trim();
    (
        circle.placement().origin(),
        circle.radius(),
        trim.start,
        trim.span,
        trim.reversed,
    )
}

fn leaf_curve(corpus: &Corpus, id: &str) -> (Curve, Value) {
    let fx = corpus.get(id);
    let provider = decode::open(&fx.dwg()).unwrap();
    let flat = decode::flatten(&provider, id).unwrap();
    let (_, curve) = decode::world(&flat.entities[0]).unwrap();
    (curve, fx.truth().unwrap())
}

// -------------------------------------------------------------------- bulge arc against the truth arc

fn bulge_arc_matches(id: &str, reversed: bool) {
    let corpus = Corpus::open();
    let fx = corpus.get(id);
    let provider = decode::open(&fx.dwg()).unwrap();
    let flat = decode::flatten(&provider, id).unwrap();
    let b = dwg_basic::extract(&flat.entities[0].entity).unwrap();
    let (c, r, start, span, rev) = arc_of_composite(&b.curve);
    let truth = fx.truth().unwrap();
    let arc = truth["entities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["class"] == "circular_arc")
        .unwrap();
    let g = &arc["geometry"];
    let tc = nums(&g["center"]);
    let sc = scale_of(&[tc[0], tc[1], num(&g["radius"])]);
    // NP-05 at the feature size of the arc.
    assert!(
        near_pt([c.x, c.y], [tc[0], tc[1]], sc),
        "{id}: centre ({}, {})",
        c.x,
        c.y
    );
    assert!(near(r, num(&g["radius"]), sc), "{id}: radius {r}");
    assert!(
        near(start, num(&g["start_angle"]), TAU),
        "{id}: start {start}"
    );
    assert!(near(span, num(&g["sweep_angle"]), TAU), "{id}: span {span}");
    assert_eq!(rev, reversed, "{id}: sense");
}

#[test]
fn n2_5_1056_a_positive_bulge_arc_is_the_fixture_arc() {
    bulge_arc_matches("bulge-arc-001", false);
}

#[test]
fn n2_5_1056_a_negative_bulge_arc_is_the_fixture_arc_run_clockwise() {
    bulge_arc_matches("bulge-arc-neg-001", true);
}

// ------------------------------------------------------- circle under non-uniform scale, an ellipse

fn promoted_ellipse_matches(id: &str) {
    let corpus = Corpus::open();
    let (curve, truth) = leaf_curve(&corpus, id);
    assert!(
        matches!(curve, Curve::Ellipse(_)),
        "{id}: a circle under this map is an ellipse, got {:?}",
        curve.kind()
    );
    let w = &truth["entities"][0]["world"]["geometry"];
    let c = nums(&w["center"]);
    let (a, b) = (num(&w["semi_major"]), num(&w["semi_minor"]));
    let m = nums(&w["major_axis"]);
    let n = m[0].hypot(m[1]);
    let u = [m[0] / n, m[1] / n];
    let sc = scale_of(&[c[0], c[1], a, b]);
    for p in decode::sample(&curve, 16) {
        assert!(
            on_ellipse(p, [c[0], c[1]], u, a, b, sc),
            "{id}: {p:?} is off the truth's ellipse"
        );
    }
}

#[test]
fn n2_5_1056_a_circle_promoted_under_non_uniform_scale_is_the_truth_ellipse() {
    promoted_ellipse_matches("nonuniform-scale-circle-001");
}

#[test]
fn n2_5_1056_a_rotated_non_uniform_scale_gives_the_same_ellipse_class() {
    promoted_ellipse_matches("nonuniform-scale-rotated-001");
}

// ------------------------------------------------------------ an arc as a rational spline, same locus

#[test]
fn n2_5_1056_an_arc_written_as_a_rational_spline_is_the_same_arc() {
    let corpus = Corpus::open();
    let (arc, truth) = leaf_curve(&corpus, "arc-seam-001");
    let rational = arc
        .to_rational_b_spline()
        .expect("an arc has an exact rational form");
    assert!(matches!(rational, Curve::RationalBSpline(_)));
    let g = &truth["entities"][0]["geometry"];
    let c = nums(&g["center"]);
    let r = num(&g["radius"]);
    let sc = scale_of(&[c[0], c[1], r]);
    // Same end points, and every point of the rational form lies on the truth circle.
    let (a, b) = (decode::sample(&arc, 8), decode::sample(&rational, 8));
    assert!(
        near_pt(a[0], b[0], sc),
        "start {:?} against {:?}",
        a[0],
        b[0]
    );
    assert!(near_pt(a[8], b[8], sc), "end {:?} against {:?}", a[8], b[8]);
    for p in decode::sample(&rational, 32) {
        assert!(
            on_ellipse(p, [c[0], c[1]], [1.0, 0.0], r, r, sc),
            "{p:?} is off the circle"
        );
    }
}

// ------------------------------------------------------------- joins: end points and derivatives

#[test]
fn n2_5_1056_segment_joins_agree_on_their_end_point() {
    let corpus = Corpus::open();
    let (curve, _) = leaf_curve(&corpus, "bulge-arc-001");
    let Curve::Composite(comp) = &curve else {
        panic!("composite expected")
    };
    for pair in comp.segments().windows(2) {
        let (d0, d1) = (pair[0].domain(), pair[1].domain());
        let end = pair[0].eval(d0.hi()).unwrap();
        let start = pair[1].eval(d1.lo()).unwrap();
        let sc = scale_of(&[end.x, end.y]);
        assert!(
            near_pt([end.x, end.y], [start.x, start.y], sc),
            "{end:?} against {start:?}"
        );
    }
}

#[test]
fn n2_5_1056_a_derivative_at_a_discontinuity_follows_np_16() {
    // NP-16: where only one-sided limits exist, a `Side::Both` request for an order the curve is
    // not smooth enough for is refused, and a one-sided request is answered. The corpus names
    // those parameters in `derivative_exclusions`; every other parameter answers both-sided.
    let corpus = Corpus::open();
    for id in ["polyline-001", "repeated-knot-c0-001"] {
        let (curve, truth) = leaf_curve(&corpus, id);
        let d = curve.domain();
        let excluded: Vec<f64> = truth["derivative_exclusions"]
            .as_array()
            .expect("the truth names its derivative exclusions")
            .iter()
            .map(|e| num(&e["parameter"]))
            .collect();
        assert!(!excluded.is_empty(), "{id}");
        for t in &excluded {
            assert!(
                curve.derivative(*t, 1, Side::Both).is_err(),
                "{id}: both-sided derivative at {t} must be refused (NP-16)"
            );
            let (left, right) = (
                curve.derivative(*t, 1, Side::Left).expect("left limit"),
                curve.derivative(*t, 1, Side::Right).expect("right limit"),
            );
            assert!(
                left.length().is_finite() && right.length().is_finite(),
                "{id}: one-sided limits at {t}"
            );
        }
        // A parameter between two exclusions, away from every knot or vertex, is smooth.
        let mid = d.lo() + 0.25 * d.width();
        if !excluded.iter().any(|e| near(*e, mid, 1.0)) {
            assert!(
                curve.derivative(mid, 1, Side::Both).is_ok(),
                "{id}: smooth at {mid}"
            );
        }
    }
}

// ------------------------------------------------------------------ two scales, one policy

fn raw(vertices: Vec<RawVertex>) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Polyline {
            vertices,
            closed: false,
            elevation: 0.0,
            normal: [0.0, 0.0, 1.0] as Point3,
        },
        provenance: ProvenanceStub {
            handle: None,
            class_name: "LWPOLYLINE".to_string(),
            block_path: InstancePath::root(),
            layer: "0".to_string(),
        },
    }
}

/// The bulge arc of `bulge-arc-001` drawn at `scale` times its size, compared with the truth arc
/// at the same factor, by one function that has no notion of which scale it is looking at.
fn same_arc_at(scale: f64) {
    let corpus = Corpus::open();
    let truth = corpus.get("bulge-arc-001").truth().unwrap();
    let g = &truth["entities"][1]["geometry"];
    let p = &truth["entities"][0]["geometry"];
    let vs = pvec_gold::points(&p["vertices"]);
    let bs = nums(&p["bulges"]);
    let e = raw(vec![
        RawVertex {
            x: vs[0][0] * scale,
            y: vs[0][1] * scale,
            bulge: bs[0],
        },
        RawVertex {
            x: vs[1][0] * scale,
            y: vs[1][1] * scale,
            bulge: bs[1],
        },
    ]);
    let b = dwg_basic::extract(&e).unwrap();
    let (c, r, start, span, _) = arc_of_composite(&b.curve);
    let tc = nums(&g["center"]);
    let want_r = num(&g["radius"]) * scale; // truth side: the same drawing, scaled
    let sc = scale_of(&[tc[0] * scale, tc[1] * scale, want_r]);
    assert!(
        near_pt([c.x, c.y], [tc[0] * scale, tc[1] * scale], sc),
        "centre at scale {scale}"
    );
    assert!(near(r, want_r, sc), "radius at scale {scale}");
    assert!(
        near(start, num(&g["start_angle"]), TAU),
        "start at scale {scale}"
    );
    assert!(
        near(span, num(&g["sweep_angle"]), TAU),
        "span at scale {scale}"
    );
}

#[test]
fn n2_5_1056_the_same_policy_holds_at_two_source_scales_a_million_apart() {
    let (small, large) = (1.0, 1.0e6);
    assert!(large / small >= 1.0e6);
    same_arc_at(small);
    same_arc_at(large);
}

// ------------------------------------------------------------------- what a result may claim

/// The strongest class a comparison supports: parameter for parameter, the same locus, or only a
/// display approximation. A result that passes only the last is display fidelity.
fn claim(structural: bool, geometric: bool, display: bool) -> Option<ConformanceClass> {
    if structural {
        Some(ConformanceClass::StructuralExactness)
    } else if geometric {
        Some(ConformanceClass::GeometricExactness)
    } else if display {
        Some(ConformanceClass::DisplayFidelity)
    } else {
        None
    }
}

#[test]
fn n2_5_1056_a_display_only_pass_is_reported_as_display_fidelity() {
    assert_eq!(
        claim(false, false, true),
        Some(ConformanceClass::DisplayFidelity)
    );
    assert_eq!(
        claim(false, true, true),
        Some(ConformanceClass::GeometricExactness)
    );
    assert_eq!(
        claim(true, true, true),
        Some(ConformanceClass::StructuralExactness)
    );
    assert_eq!(claim(false, false, false), None);
    // An arc written as a rational spline passes the locus check and not the parameter one.
    let corpus = Corpus::open();
    let (arc, _) = leaf_curve(&corpus, "arc-seam-001");
    let rational = arc.to_rational_b_spline().unwrap();
    let same_kind = std::mem::discriminant(&arc) == std::mem::discriminant(&rational);
    assert_eq!(
        claim(same_kind, true, true),
        Some(ConformanceClass::GeometricExactness)
    );
}

// ------------------------------------------------------------------------- the guard

/// Violations of "no literal is a comparison threshold": a scientific literal with a negative
/// exponent, `EPSILON`, or a comparison against a decimal that starts `0.00`, outside comments.
fn threshold_literals(file: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        let b = code.as_bytes();
        let sci = (1..b.len().saturating_sub(2)).any(|i| {
            (b[i] == b'e' || b[i] == b'E')
                && b[i + 1] == b'-'
                && b[i - 1].is_ascii_digit()
                && b[i + 2].is_ascii_digit()
        });
        let eps = code.contains(&["EPS", "ILON"].concat());
        let tiny = ["<", ">", "<=", ">="]
            .iter()
            .any(|op| code.contains(&format!("{op} 0.00")) || code.contains(&format!("{op}0.00")));
        if sci || eps || tiny {
            out.push(format!("{file}:{}: {}", n + 1, line.trim()));
        }
    }
    out
}

#[test]
fn n2_5_1056_no_numeric_literal_is_a_comparison_threshold_in_the_normalization_and_conformance_sources()
 {
    let sources: &[(&str, &str)] = &[
        (
            "src/pvec/normalized.rs",
            include_str!("../src/pvec/normalized.rs"),
        ),
        (
            "src/pvec/mapping.rs",
            include_str!("../src/pvec/mapping.rs"),
        ),
        (
            "src/pvec/transform.rs",
            include_str!("../src/pvec/transform.rs"),
        ),
        (
            "src/pvec/dwg_basic.rs",
            include_str!("../src/pvec/dwg_basic.rs"),
        ),
        (
            "src/pvec/dwg_spline.rs",
            include_str!("../src/pvec/dwg_spline.rs"),
        ),
        (
            "src/pvec/flatten.rs",
            include_str!("../src/pvec/flatten.rs"),
        ),
        (
            "src/pvec/fidelity.rs",
            include_str!("../src/pvec/fidelity.rs"),
        ),
        ("src/pvec/bounds.rs", include_str!("../src/pvec/bounds.rs")),
        (
            "src/pvec/ordering.rs",
            include_str!("../src/pvec/ordering.rs"),
        ),
        ("src/pvec/view.rs", include_str!("../src/pvec/view.rs")),
        (
            "src/pvec/view_profile.rs",
            include_str!("../src/pvec/view_profile.rs"),
        ),
        (
            "tests/pvec_known_truth.rs",
            include_str!("pvec_known_truth.rs"),
        ),
        (
            "tests/pvec_structural.rs",
            include_str!("pvec_structural.rs"),
        ),
        ("tests/pvec_geometric.rs", include_str!("pvec_geometric.rs")),
        (
            "tests/common/pvec_gold.rs",
            include_str!("common/pvec_gold.rs"),
        ),
        (
            "tests/common/pvec_gold_decode.rs",
            include_str!("common/pvec_gold_decode.rs"),
        ),
    ];
    let found: Vec<String> = sources
        .iter()
        .flat_map(|(f, t)| threshold_literals(f, t))
        .collect();
    assert!(
        found.is_empty(),
        "a literal threshold crept in:\n{}",
        found.join("\n")
    );
}

#[test]
fn n2_5_1056_the_guard_fails_on_a_literal_threshold_and_names_the_file() {
    let planted = ["if (a - b).abs() < 1e", "-9 { }"].concat();
    let v = threshold_literals("src/pvec/transform.rs", &planted);
    assert_eq!(v.len(), 1, "{v:?}");
    assert!(v[0].starts_with("src/pvec/transform.rs:1:"), "{v:?}");
    let planted = ["let d = f64::EPS", "ILON;"].concat();
    assert_eq!(threshold_literals("x.rs", &planted).len(), 1);
    let planted = ["x ", "< 0.0", "01"].concat();
    assert_eq!(threshold_literals("x.rs", &planted).len(), 1);
    assert!(threshold_literals("x.rs", "let n = 1e6; // 1e-9 in a comment").is_empty());
}
