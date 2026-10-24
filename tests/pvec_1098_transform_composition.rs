//! Issue #1098 (N3.4): transform composition and promotion.
//!
//! The matrix under test is `docs/adr/pvec-transform-matrix.md`, measured in
//! `docs/spikes/transform-matrix.md` (evidence ids in the comments). Every number that is a
//! tolerance comes from `Policy::V1` by name, here and nowhere in the module.
#![cfg(feature = "pvec-normalized")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::transform::{
    Cell, ComposedTransform, PromotionError, PromotionMatrix, Shape, SourceRow, TransformReason,
    promote, promote_with,
};
use libviprs::pvec::{
    FidelityClass, InstanceStep, Normalized2DGeometry, ProvenanceHandle, SourceTransform,
};
use pvec_geometry::{
    BSpline, Circle, Composite, Curve, Ellipse, Line, Placement2, Point2, Polyline,
    RationalBSpline, Transform2, Trimmed, Vector2,
};
use pvec_numeric::Policy;

const SRC: &str = include_str!("../src/pvec/transform.rs");
const EXACT: FidelityClass = FidelityClass::SupportedExact;

fn step(name: &str, handle: u64) -> InstanceStep {
    InstanceStep {
        block_name: name.to_string(),
        insert_handle: ItemHandle::new(handle),
    }
}

fn st(m: [f64; 6]) -> SourceTransform {
    SourceTransform::new(m).unwrap()
}

fn one(m: [f64; 6]) -> ComposedTransform {
    ComposedTransform::compose([(step("B", 1), st(m))]).unwrap()
}

fn p(x: f64, y: f64) -> Point2 {
    Point2::new(x, y)
}

fn placement(cx: f64, cy: f64, dx: f64, dy: f64) -> Placement2 {
    Placement2::new(p(cx, cy), Vector2::new(dx, dy)).unwrap()
}

// The fixtures F1 to F9 of the spike.
fn fixtures() -> Vec<(SourceRow, Curve)> {
    let line = Line::new(p(1.0, 2.0), Vector2::new(3.0, 4.0)).unwrap();
    let circle = Circle::new(placement(3.0, -2.0, 1.0, 0.0), 5.0).unwrap();
    let ellipse = Ellipse::new(placement(1.0, 1.0, 0.8, 0.6), 4.0, 2.0).unwrap();
    let bulge = Composite::new(vec![
        Curve::Trimmed(
            Trimmed::new(
                Curve::Line(Line::new(p(0.0, 0.0), Vector2::new(1.0, 0.0)).unwrap()),
                0.0,
                4.0,
                true,
            )
            .unwrap(),
        ),
        Curve::Trimmed(
            Trimmed::new(
                Curve::Circle(Circle::new(placement(4.0, 2.0, 1.0, 0.0), 2.0).unwrap()),
                -std::f64::consts::FRAC_PI_2,
                0.0,
                true,
            )
            .unwrap(),
        ),
    ])
    .unwrap();
    let pts = |v: &[(f64, f64)]| v.iter().map(|&(x, y)| p(x, y)).collect::<Vec<_>>();
    vec![
        (
            SourceRow::Line,
            Curve::Trimmed(Trimmed::new(Curve::Line(line), 0.5, 6.0, true).unwrap()),
        ),
        (
            SourceRow::Polyline,
            Curve::Polyline(
                Polyline::new(pts(&[(0.0, 0.0), (4.0, 0.0), (5.0, 3.0), (2.0, 5.0)])).unwrap(),
            ),
        ),
        (SourceRow::PolylineWithBulge, Curve::Composite(bulge)),
        (
            SourceRow::CircularArc,
            Curve::Trimmed(Trimmed::new(Curve::Circle(circle), 0.3, 2.1, true).unwrap()),
        ),
        (SourceRow::Circle, Curve::Circle(circle)),
        (SourceRow::Ellipse, Curve::Ellipse(ellipse)),
        (
            SourceRow::EllipticalArc,
            Curve::Trimmed(Trimmed::new(Curve::Ellipse(ellipse), 0.4, 4.0, true).unwrap()),
        ),
        (
            SourceRow::BSpline,
            Curve::BSpline(
                BSpline::new(
                    3,
                    pts(&[
                        (0.0, 0.0),
                        (1.0, 2.0),
                        (2.0, -1.0),
                        (3.0, 3.0),
                        (4.0, 0.0),
                        (5.0, 2.0),
                    ]),
                    vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                )
                .unwrap(),
            ),
        ),
        (
            SourceRow::Nurbs,
            Curve::RationalBSpline(
                RationalBSpline::new(
                    2,
                    pts(&[(0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0), (5.0, 1.0)]),
                    vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    vec![1.0, 0.7, 1.4, 1.0, 0.9],
                )
                .unwrap(),
            ),
        ),
    ]
}

fn record(c: &Curve) -> Normalized2DGeometry {
    Normalized2DGeometry::from_curve(c.clone(), EXACT).unwrap()
}

// T1 translation, T2 rotation, T3 uniform scale, T4 non-uniform scale, T5 reflection, T6 general.
fn classes() -> [(&'static str, [f64; 6]); 6] {
    [
        ("translation", [1.0, 0.0, 0.0, 1.0, 7.5, -3.25]),
        ("rotation", Transform2::rotate(0.6).unwrap().matrix()),
        ("uniform scale", [2.5, 0.0, 0.0, 2.5, 0.0, 0.0]),
        ("non-uniform scale", [3.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
        ("reflection", [1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        ("general affine", [1.3, 0.4, -0.7, 0.9, 2.0, -1.0]),
    ]
}

// ---- composition order ---------------------------------------------------------------------

#[test]
fn order_is_outermost_first_and_reversing_it_changes_the_matrix() {
    // Outer: translate by 10 in x. Inner: scale by 2. A point of the innermost block is scaled
    // first, then placed, so (1, 0) lands at x = 12 and the translation stays 10.
    let outer = st([1.0, 0.0, 0.0, 1.0, 10.0, 0.0]);
    let inner = st([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let c = ComposedTransform::compose([(step("OUT", 1), outer), (step("IN", 2), inner)]).unwrap();
    assert_eq!(c.matrix(), [2.0, 0.0, 0.0, 2.0, 10.0, 0.0]);
    let at = c.transform().apply_point(p(1.0, 0.0));
    assert_eq!((at.x, at.y), (12.0, 0.0));

    // The same two levels the other way round give a different matrix, so the test fails if the
    // order is reversed.
    let rev =
        ComposedTransform::compose([(step("IN", 2), inner), (step("OUT", 1), outer)]).unwrap();
    assert_eq!(rev.matrix(), [2.0, 0.0, 0.0, 2.0, 20.0, 0.0]);
    assert_ne!(c.matrix(), rev.matrix());
}

#[test]
fn push_inner_matches_composing_the_whole_nest() {
    let a = st([1.0, 0.0, 0.0, 1.0, 10.0, 0.0]);
    let b = st(Transform2::rotate(0.6).unwrap().matrix());
    let d = st([2.0, 0.0, 0.0, 3.0, 1.0, 1.0]);
    let whole =
        ComposedTransform::compose([(step("A", 1), a), (step("B", 2), b), (step("C", 3), d)])
            .unwrap();
    let stepwise = ComposedTransform::identity()
        .push_inner(step("A", 1), &a)
        .unwrap()
        .push_inner(step("B", 2), &b)
        .unwrap()
        .push_inner(step("C", 3), &d)
        .unwrap();
    assert_eq!(whole.matrix(), stepwise.matrix());
}

#[test]
fn an_empty_nest_is_the_identity() {
    let c = ComposedTransform::compose(Vec::<(InstanceStep, SourceTransform)>::new()).unwrap();
    assert_eq!(c.matrix(), Transform2::IDENTITY.matrix());
    assert!(c.lineage().is_empty());
}

// ---- finite checking -----------------------------------------------------------------------

#[test]
fn a_non_finite_coefficient_names_the_insert_and_emits_no_geometry() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let ok = ComposedTransform::identity()
            .push_inner(step("OUTER", 10), &st([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
            .unwrap()
            .push_inner(step("MIDDLE", 20), &st([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
            .unwrap();
        let err = ok
            .push_inner_coefficients(step("INNER", 30), [1.0, 0.0, 0.0, 1.0, bad, 0.0])
            .unwrap_err();
        assert_eq!(err.insert().block_name, "INNER");
        assert_eq!(err.insert().insert_handle, ItemHandle::new(30));
        assert_eq!(err.depth(), 2);
        assert_eq!(err.reason(), Some(TransformReason::NonFinite));
        assert_eq!(err.error().rule(), "NP-01");
        // The branch has no composed transform, so nothing can be promoted through it, and the
        // message names the insert.
        assert!(err.to_string().contains("INNER"), "{err}");
    }
}

#[test]
fn the_non_finite_level_is_found_wherever_it_sits_in_the_nest() {
    for at in 0..3 {
        let mut levels: Vec<[f64; 6]> = vec![[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]; 3];
        levels[at][5] = f64::NAN;
        let mut c = ComposedTransform::identity();
        let mut found = None;
        for (i, m) in levels.iter().enumerate() {
            match c.push_inner_coefficients(step(&format!("L{i}"), i as u64), *m) {
                Ok(n) => c = n,
                Err(e) => {
                    found = Some(e);
                    break;
                }
            }
        }
        let e = found.expect("refused");
        assert_eq!(e.insert().block_name, format!("L{at}"));
        assert_eq!(e.depth(), at);
    }
}

// ---- degeneracy: singular and near-singular, through PVec ----------------------------------

#[test]
fn a_singular_transform_is_refused_with_the_matrix_diagnostic() {
    // T7 of the spike: determinant exactly zero (E55 to E63).
    let err = ComposedTransform::identity()
        .push_inner(step("FLAT", 7), &st([1.0, 2.0, 2.0, 4.0, 0.0, 0.0]))
        .unwrap_err();
    assert_eq!(err.reason(), Some(TransformReason::Singular));
    assert_eq!(err.error().rule(), "NP-19");
    assert_eq!(err.insert().block_name, "FLAT");
    assert_eq!(TransformReason::Singular.code(), "TRANSFORM_SINGULAR");
    assert_eq!(TransformReason::Singular.rule(), "NP-19");
}

#[test]
fn a_near_singular_map_is_classified_by_the_policy_not_by_a_literal() {
    // One side of the NP-19 bound each, derived from the policy's own constant (T9, T10).
    let rel = Policy::V1.rel;
    let below = [1.0, 0.0, 0.0, 0.5 * rel, 0.0, 0.0];
    let above = [1.0, 0.0, 0.0, 2.0 * rel, 0.0, 0.0];
    let e = ComposedTransform::identity()
        .push_inner(step("THIN", 1), &st(below))
        .unwrap_err();
    assert_eq!(e.reason(), Some(TransformReason::Singular));
    assert!(
        ComposedTransform::identity()
            .push_inner(step("THIN", 1), &st(above))
            .is_ok()
    );
}

#[test]
fn a_composition_can_become_singular_when_no_single_level_is() {
    // Each level is fine, the product is not. PVec decides where, and the insert named is the one
    // whose step crossed the line.
    let s = Policy::V1.rel.sqrt().sqrt();
    let level = [1.0, 0.0, 0.0, 10.0 * s, 0.0, 0.0];
    let mut c = ComposedTransform::identity();
    let mut refused_at = None;
    for i in 0..8 {
        match c.push_inner_coefficients(step(&format!("N{i}"), i), level) {
            Ok(n) => c = n,
            Err(e) => {
                assert_eq!(e.reason(), Some(TransformReason::Singular));
                refused_at = Some(e.depth());
                break;
            }
        }
    }
    let depth = refused_at.expect("the product eventually collapses");
    assert!(depth >= 1, "no single level was singular");
}

// ---- lineage -------------------------------------------------------------------------------

#[test]
fn the_composed_matrix_of_a_two_level_nest_equals_the_product_of_the_recorded_levels() {
    let outer = st([1.0, 0.0, 0.0, 1.0, 10.0, 5.0]);
    let inner = st(Transform2::rotate(0.6).unwrap().matrix());
    let c = ComposedTransform::compose([(step("OUT", 1), outer), (step("IN", 2), inner)]).unwrap();

    let lineage = c.lineage();
    assert_eq!(lineage.len(), 2);
    assert_eq!(lineage[0].insert, step("OUT", 1));
    assert_eq!(lineage[0].transform, outer.coefficients());
    assert_eq!(lineage[1].insert, step("IN", 2));
    assert_eq!(lineage[1].transform, inner.coefficients());

    // Product of the recorded levels through PVec: the innermost applies first.
    let o = Transform2::new(lineage[0].transform).unwrap();
    let i = Transform2::new(lineage[1].transform).unwrap();
    let product = i.then(&o).unwrap();
    assert_eq!(c.matrix(), product.matrix());
    assert_eq!(c.source_transform().coefficients(), product.matrix());
}

// ---- promotion: the matrix ----------------------------------------------------------------

#[test]
fn every_cell_of_the_frozen_matrix_is_pinned() {
    use Shape::*;
    // Columns: translation, rotation, uniform scale, non-uniform scale, reflection, general.
    let expected: [(SourceRow, [Shape; 6]); 9] = [
        (SourceRow::Line, [TrimmedLine; 6]),
        (SourceRow::Polyline, [Shape::Polyline; 6]),
        (SourceRow::PolylineWithBulge, [Composite; 6]),
        (
            SourceRow::CircularArc,
            [
                TrimmedCircle,
                TrimmedCircle,
                TrimmedCircle,
                TrimmedEllipse,
                TrimmedCircle,
                TrimmedEllipse,
            ],
        ),
        (
            SourceRow::Circle,
            [Circle, Circle, Circle, Ellipse, Circle, Ellipse],
        ),
        (SourceRow::Ellipse, [Ellipse; 6]),
        (SourceRow::EllipticalArc, [TrimmedEllipse; 6]),
        (SourceRow::BSpline, [BSpline; 6]),
        (SourceRow::Nurbs, [RationalBSpline; 6]),
    ];
    let fx = fixtures();
    for (row, shapes) in expected {
        let (frow, curve) = fx.iter().find(|(r, _)| *r == row).unwrap();
        assert_eq!(SourceRow::of(curve), Some(*frow));
        for ((name, m), want) in classes().iter().zip(shapes) {
            let out = promote(&record(curve), &one(*m)).unwrap();
            let got = Shape::of(out.record().geometry().expect("geometry")).unwrap();
            assert_eq!(got, want, "{row:?} under {name}");
            assert_eq!(out.record().fidelity(), EXACT, "{row:?} under {name}");
            assert!(out.reason().is_none());
            assert!(out.record().source_bounds().is_some());
        }
    }
}

#[test]
fn a_circle_under_non_uniform_scale_is_an_ellipse_never_a_circle() {
    // E32: semi-axes 15 and 2.5 for the radius 5 circle under scale(3, 0.5).
    let fx = fixtures();
    let circle = &fx.iter().find(|(r, _)| *r == SourceRow::Circle).unwrap().1;
    let out = promote(&record(circle), &one([3.0, 0.0, 0.0, 0.5, 0.0, 0.0])).unwrap();
    match out.record().geometry().unwrap() {
        Curve::Ellipse(e) => {
            let (a, b) = (e.semi_axis_1(), e.semi_axis_2());
            assert!(
                (a.max(b) - 15.0).abs() < 1e-9 && (a.min(b) - 2.5).abs() < 1e-9,
                "{a} {b}"
            );
        }
        other => panic!("expected an ellipse, got {other:?}"),
    }
}

#[test]
fn the_bulge_composite_keeps_its_segments_and_promotes_the_arc_inside_it() {
    // E30 and E48: the arc segment becomes a trimmed ellipse, the line stays a line, and the
    // segment count is unchanged.
    let fx = fixtures();
    let bulge = &fx
        .iter()
        .find(|(r, _)| *r == SourceRow::PolylineWithBulge)
        .unwrap()
        .1;
    for m in [
        [3.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [1.3, 0.4, -0.7, 0.9, 2.0, -1.0],
    ] {
        let out = promote(&record(bulge), &one(m)).unwrap();
        let Curve::Composite(c) = out.record().geometry().unwrap() else {
            panic!("not a composite")
        };
        let kinds: Vec<_> = c.segments().iter().map(|s| Shape::of(s).unwrap()).collect();
        assert_eq!(kinds, [Shape::TrimmedLine, Shape::TrimmedEllipse]);
    }
    let out = promote(&record(bulge), &one([2.5, 0.0, 0.0, 2.5, 0.0, 0.0])).unwrap();
    let Curve::Composite(c) = out.record().geometry().unwrap() else {
        panic!("not a composite")
    };
    let kinds: Vec<_> = c.segments().iter().map(|s| Shape::of(s).unwrap()).collect();
    assert_eq!(kinds, [Shape::TrimmedLine, Shape::TrimmedCircle]);
}

#[test]
fn a_reflection_keeps_the_end_points_and_their_order() {
    // E40: start maps to start, end to end, and the sweep flips.
    let fx = fixtures();
    let arc = &fx
        .iter()
        .find(|(r, _)| *r == SourceRow::CircularArc)
        .unwrap()
        .1;
    let m = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
    let out = promote(&record(arc), &one(m)).unwrap();
    let t = Transform2::new(m).unwrap();
    use pvec_geometry::Curve2d;
    let (s0, s1) = (arc.domain().lo(), arc.domain().hi());
    let new = out.record().geometry().unwrap();
    let (n0, n1) = (new.domain().lo(), new.domain().hi());
    let a = t.apply_point(arc.eval(s0).unwrap());
    let b = new.eval(n0).unwrap();
    assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
    let a = t.apply_point(arc.eval(s1).unwrap());
    let b = new.eval(n1).unwrap();
    assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
}

#[test]
fn splines_keep_degree_knots_and_weights_and_only_move_their_control_points() {
    let fx = fixtures();
    let m = [1.3, 0.4, -0.7, 0.9, 2.0, -1.0];
    let bs = &fx.iter().find(|(r, _)| *r == SourceRow::BSpline).unwrap().1;
    let (Curve::BSpline(before), Curve::BSpline(after)) = (
        bs,
        promote(&record(bs), &one(m))
            .unwrap()
            .record()
            .geometry()
            .unwrap()
            .clone(),
    ) else {
        panic!("kind changed")
    };
    assert_eq!(before.degree(), after.degree());
    assert_eq!(before.knots(), after.knots());
    let nu = &fx.iter().find(|(r, _)| *r == SourceRow::Nurbs).unwrap().1;
    let (Curve::RationalBSpline(before), Curve::RationalBSpline(after)) = (
        nu,
        promote(&record(nu), &one(m))
            .unwrap()
            .record()
            .geometry()
            .unwrap()
            .clone(),
    ) else {
        panic!("kind changed")
    };
    assert_eq!(before.weights(), after.weights());
    assert_eq!(before.knots(), after.knots());
}

// ---- promotion: refusals and classifications -----------------------------------------------

#[test]
fn a_conic_pushed_past_the_conditioning_bound_is_unsupported_and_other_kinds_are_not() {
    let ratio = Policy::V1.axis_ratio_max;
    let inside = [1.0, 0.0, 0.0, 4.0 / ratio, 0.0, 0.0]; // T12
    let outside = [1.0, 0.0, 0.0, 0.25 / ratio, 0.0, 0.0]; // T13
    for (row, curve) in fixtures() {
        let conic = matches!(
            row,
            SourceRow::Circle
                | SourceRow::Ellipse
                | SourceRow::CircularArc
                | SourceRow::EllipticalArc
                | SourceRow::PolylineWithBulge
        );
        assert!(
            promote(&record(&curve), &one(inside))
                .unwrap()
                .record()
                .geometry()
                .is_some(),
            "{row:?} inside"
        );
        let out = promote(&record(&curve), &one(outside)).unwrap();
        if conic {
            assert!(out.record().geometry().is_none(), "{row:?} outside");
            assert_eq!(out.record().fidelity(), FidelityClass::Unsupported);
            assert_eq!(out.reason(), Some(TransformReason::ConicIllConditioned));
            assert_eq!(TransformReason::ConicIllConditioned.rule(), "NP-07");
            assert_eq!(
                TransformReason::ConicIllConditioned.code(),
                "TRANSFORM_CONIC_ILL_CONDITIONED"
            );
        } else {
            assert!(out.record().geometry().is_some(), "{row:?} outside");
        }
    }
}

#[test]
fn promotion_keeps_the_provenance_handle() {
    let fx = fixtures();
    let c = &fx[4].1;
    let rec = record(c).with_provenance(ProvenanceHandle::new(42));
    let ok = promote(&rec, &one([3.0, 0.0, 0.0, 0.5, 0.0, 0.0])).unwrap();
    assert_eq!(ok.record().provenance(), Some(ProvenanceHandle::new(42)));
    let ratio = Policy::V1.axis_ratio_max;
    let refused = promote(&rec, &one([1.0, 0.0, 0.0, 0.25 / ratio, 0.0, 0.0])).unwrap();
    assert_eq!(
        refused.record().provenance(),
        Some(ProvenanceHandle::new(42))
    );
}

#[test]
fn a_record_without_geometry_passes_through_unchanged() {
    let rec = Normalized2DGeometry::without_geometry(FidelityClass::IgnoredByPolicy).unwrap();
    let out = promote(&rec, &one([2.0, 0.0, 0.0, 2.0, 0.0, 0.0])).unwrap();
    assert!(out.record().geometry().is_none());
    assert_eq!(out.record().fidelity(), FidelityClass::IgnoredByPolicy);
}

fn all_require_capability(_: SourceRow) -> Cell {
    Cell::RequiresPVecCapability
}

#[test]
fn a_cell_that_requires_a_pvec_capability_is_classified_not_approximated() {
    let m = PromotionMatrix::from_lookup(all_require_capability);
    let fx = fixtures();
    let rec = record(&fx[4].1).with_provenance(ProvenanceHandle::new(1));
    let out = promote_with(&m, &rec, &one([3.0, 0.0, 0.0, 0.5, 0.0, 0.0])).unwrap();
    assert!(out.record().geometry().is_none());
    assert_eq!(
        out.record().fidelity(),
        FidelityClass::RequiresPVecCapability
    );
    assert_eq!(out.record().provenance(), Some(ProvenanceHandle::new(1)));
}

fn circle_is_a_polyline(row: SourceRow) -> Cell {
    match row {
        SourceRow::Circle => Cell::Shapes(&[Shape::Polyline]),
        _ => Cell::Shapes(&[Shape::Circle]),
    }
}

#[test]
fn a_shape_the_cell_does_not_allow_is_a_violation_and_not_passed_on() {
    let m = PromotionMatrix::from_lookup(circle_is_a_polyline);
    let fx = fixtures();
    let err =
        promote_with(&m, &record(&fx[4].1), &one([2.5, 0.0, 0.0, 2.5, 0.0, 0.0])).unwrap_err();
    assert!(
        matches!(
            err,
            PromotionError::MatrixViolation {
                row: SourceRow::Circle,
                got: Some(Shape::Circle)
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_bare_line_has_no_row() {
    let l = Curve::Line(Line::new(p(0.0, 0.0), Vector2::new(1.0, 0.0)).unwrap());
    assert_eq!(SourceRow::of(&l), None);
    assert_eq!(Shape::of(&l), None);
}

// ---- no numbers of our own ----------------------------------------------------------------

fn code_only(src: &str) -> String {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_module_has_no_numeric_literal_threshold() {
    let code = code_only(SRC);
    let bytes: Vec<char> = code.chars().collect();
    for i in 1..bytes.len().saturating_sub(1) {
        let (a, b, c) = (bytes[i - 1], bytes[i], bytes[i + 1]);
        assert!(
            !(a.is_ascii_digit() && b == '.' && c.is_ascii_digit()),
            "a float literal near byte {i}: {}",
            &code[i.saturating_sub(20)..(i + 20).min(code.len())]
        );
        assert!(
            !(a.is_ascii_digit() && (b == 'e' || b == 'E') && (c == '-' || c.is_ascii_digit())),
            "an exponent literal near byte {i}"
        );
    }
    for banned in [
        "EPSILON",
        "epsilon",
        "tolerance",
        "f64::MIN_POSITIVE",
        "Policy::",
    ] {
        assert!(!code.contains(banned), "`{banned}` in the module");
    }
}
