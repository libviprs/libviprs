//! Exact extraction of lines, polylines, bulges, circles, arcs and ellipses (N4.2, #1073).
//!
//! The known-truth numbers come from `pvec-gold-fixtures` (`corpus/r0-open`, the
//! `params.json` and `expected.math.json` of `line-001`, `bulge-arc-001`, `bulge-arc-neg-001`,
//! `circle-001`, `ellipse-001`, `elliptical-arc-001`, `arc-seam-001` and
//! `lwpolyline-closed-001`). They are written out here, so this file needs nothing outside the
//! repository. Parameters are compared as parameters (centre, radius, angles, ratio), and every
//! tolerance is PVec's `Policy::V1` rather than a number of mine.
//!
//! Needs `--features pvec-normalized`.

#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::dwg_basic::{
    BasicError, BasicKind, ExactBasic, SegmentKind, extract, extract_arc, extract_circle,
    extract_ellipse, extract_line, extract_polyline,
};
use libviprs::pvec::fidelity::{FidelityClass, ReasonCode};
use libviprs::pvec::{InstancePath, Point3, ProvenanceStub, RawEntity, RawGeometry, RawVertex};
use pvec_geometry::{Curve, Curve2d, CurveKind, Point2};
use pvec_numeric::{MAX_POLYLINE_VERTICES, Policy};

const POLICY: Policy = Policy::V1;
const Z: Point3 = [0.0, 0.0, 1.0];
const TAU: f64 = std::f64::consts::TAU;

fn entity(class: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        geometry,
        provenance: ProvenanceStub {
            handle: None,
            class_name: class.to_string(),
            block_path: InstancePath::root(),
            layer: "0".to_string(),
        },
    }
}

fn near(a: f64, b: f64, scale: f64) -> bool {
    POLICY.near(a, b, scale)
}

fn near_pt(p: Point2, x: f64, y: f64) -> bool {
    POLICY.near_point([p.x, p.y], [x, y], 1.0)
}

fn line(start: Point3, end: Point3) -> RawEntity {
    entity("LINE", RawGeometry::Line { start, end })
}

fn vertex(x: f64, y: f64, bulge: f64) -> RawVertex {
    RawVertex { x, y, bulge }
}

fn polyline(vertices: Vec<RawVertex>, closed: bool) -> RawEntity {
    entity(
        "LWPOLYLINE",
        RawGeometry::Polyline {
            vertices,
            closed,
            elevation: 0.0,
            normal: Z,
        },
    )
}

fn circle(centre: Point3, radius: f64) -> RawEntity {
    entity(
        "CIRCLE",
        RawGeometry::Circle {
            center: centre,
            radius,
            normal: Z,
        },
    )
}

fn arc(centre: Point3, radius: f64, start: f64, end: f64) -> RawEntity {
    entity(
        "ARC",
        RawGeometry::Arc {
            center: centre,
            radius,
            start_angle: start,
            end_angle: end,
            normal: Z,
        },
    )
}

fn ellipse(centre: Point3, major: Point3, ratio: f64, t0: f64, t1: f64) -> RawEntity {
    entity(
        "ELLIPSE",
        RawGeometry::Ellipse {
            center: centre,
            major_axis: major,
            minor_ratio: ratio,
            start_parameter: t0,
            end_parameter: t1,
            normal: Z,
        },
    )
}

/// The arc a bulge segment became: `(centre, radius, start, span, reversed)` read off PVec's
/// trimmed circle, not recomputed here.
fn arc_of(c: &Curve) -> (Point2, f64, f64, f64, bool) {
    let Curve::Trimmed(t) = c else {
        panic!("not a trimmed curve: {:?}", c.kind());
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

fn single_arc(b: &ExactBasic) -> (Point2, f64, f64, f64, bool) {
    let Curve::Composite(comp) = &b.curve else {
        panic!("a bulge polyline is a composite, got {:?}", b.curve.kind());
    };
    assert_eq!(comp.segments().len(), 1);
    arc_of(&comp.segments()[0])
}

// ---------------------------------------------------------------- LINE

#[test]
fn line_001_is_a_trimmed_line_through_both_end_points() {
    // line-001: (0.5, 1.25) to (4.0, 3.5) at z 0.
    let b = extract_line(&line([0.5, 1.25, 0.0], [4.0, 3.5, 0.0])).unwrap();
    assert_eq!(b.kind, BasicKind::Line);
    let Curve::Trimmed(t) = &b.curve else {
        panic!("a line is a trimmed line, got {:?}", b.curve.kind());
    };
    assert_eq!(t.basis().kind(), CurveKind::Line);
    let d = b.curve.domain();
    let (p0, p1) = (b.curve.eval(d.lo()).unwrap(), b.curve.eval(d.hi()).unwrap());
    assert_eq!((p0.x, p0.y), (0.5, 1.25), "the start point is bit exact");
    assert!(near_pt(p1, 4.0, 3.5));
    assert_eq!(b.classification.class(), FidelityClass::SupportedEquivalent);
    assert_eq!(b.classification.source_class(), "LINE");
}

#[test]
fn a_line_with_two_z_values_is_not_flattened() {
    let e = extract_line(&line([0.0, 0.0, 0.0], [1.0, 1.0, 2.0])).unwrap_err();
    assert_eq!(e, BasicError::NonPlanar);
    let c = e.classification("LINE");
    assert_eq!(c.class(), FidelityClass::RequiresPVecCapability);
    assert_eq!(c.reason(), Some(ReasonCode::Geometry3d));
}

#[test]
fn a_line_with_one_z_keeps_it_as_the_elevation() {
    let b = extract_line(&line([0.0, 0.0, 3.5], [1.0, 1.0, 3.5])).unwrap();
    assert_eq!(b.elevation, 3.5);
}

#[test]
fn a_line_of_zero_length_is_classified_coincident() {
    let e = extract_line(&line([2.0, 2.0, 0.0], [2.0, 2.0, 0.0])).unwrap_err();
    assert!(matches!(e, BasicError::CoincidentVertices { index: 0 }));
    assert_eq!(
        e.classification("LINE").reason(),
        Some(ReasonCode::CoincidentVertices)
    );
}

#[test]
fn a_non_finite_line_coordinate_is_refused() {
    let e = extract_line(&line([f64::NAN, 0.0, 0.0], [1.0, 1.0, 0.0])).unwrap_err();
    assert!(matches!(e, BasicError::Degenerate { .. }), "{e:?}");
}

// --------------------------------------------------------- POLYLINE

#[test]
fn lwpolyline_closed_001_closes_with_a_copy_of_the_first_vertex() {
    // lwpolyline-closed-001: four vertices, closed by flag, all bulges zero.
    let vs = [(0.0, 0.0), (5.0, 0.5), (4.25, 3.75), (-1.0, 2.5)];
    let e = polyline(vs.iter().map(|v| vertex(v.0, v.1, 0.0)).collect(), true);
    let b = extract_polyline(&e).unwrap();
    assert_eq!(b.kind, BasicKind::Polyline);
    assert!(b.closed);
    let Curve::Polyline(p) = &b.curve else {
        panic!("all-straight is a polyline, got {:?}", b.curve.kind());
    };
    assert_eq!(p.vertices().len(), 5, "four vertices plus the closing copy");
    for (i, v) in vs.iter().enumerate() {
        assert_eq!(
            (p.vertices()[i].x, p.vertices()[i].y),
            *v,
            "vertex {i} is bit exact"
        );
    }
    assert_eq!(
        (p.vertices()[4].x, p.vertices()[4].y),
        (0.0, 0.0),
        "the closing vertex is the first, copied"
    );
    assert_eq!(b.segments, vec![SegmentKind::Straight; 4]);
    assert_eq!(b.classification.class(), FidelityClass::SupportedExact);
}

#[test]
fn an_open_polyline_keeps_every_vertex_and_adds_none() {
    let e = polyline(
        vec![
            vertex(0.0, 0.0, 0.0),
            vertex(2.0, 0.0, 0.0),
            vertex(2.0, 3.0, 0.0),
        ],
        false,
    );
    let b = extract_polyline(&e).unwrap();
    let Curve::Polyline(p) = &b.curve else {
        panic!("polyline expected");
    };
    assert_eq!(p.vertices().len(), 3);
    assert!(!b.closed);
    assert_eq!(b.segments.len(), 2);
}

#[test]
fn old_style_polyline_goes_through_the_same_path() {
    let mut e = polyline(vec![vertex(0.0, 0.0, 0.0), vertex(1.0, 1.0, 0.0)], false);
    e.provenance.class_name = "POLYLINE".to_string();
    let b = extract(&e).unwrap();
    assert_eq!(b.classification.source_class(), "POLYLINE");
}

#[test]
fn a_zero_bulge_stays_a_straight_segment_between_two_arcs() {
    let e = polyline(
        vec![
            vertex(0.0, 0.0, 0.4),
            vertex(2.0, 0.0, 0.0),
            vertex(4.0, 0.0, -0.3),
            vertex(6.0, 1.0, 0.0),
        ],
        false,
    );
    let b = extract_polyline(&e).unwrap();
    assert_eq!(
        b.segments,
        vec![
            SegmentKind::ArcCounterClockwise,
            SegmentKind::Straight,
            SegmentKind::ArcClockwise
        ]
    );
    let Curve::Composite(c) = &b.curve else {
        panic!("composite expected");
    };
    assert_eq!(c.segments().len(), 3);
    assert_eq!(c.segments()[1].kind(), CurveKind::Polyline);
    assert_eq!(b.classification.class(), FidelityClass::SupportedEquivalent);
}

#[test]
fn bulge_arc_001_matches_the_expected_arc() {
    // bulge-arc-001: p0 (1, 0.5), p1 (4.5, 2.25), bulge 2/5. Expected arc from the truth record:
    // centre (1.83125, 3.2125), radius 2.837011246, start 4.415023835, sweep 1.522025508, CCW.
    let e = polyline(
        vec![vertex(1.0, 0.5, 0.4), vertex(4.5, 2.25, 0.0)],
        false,
    );
    let b = extract_polyline(&e).unwrap();
    let (c, r, start, span, reversed) = single_arc(&b);
    assert!(near_pt(c, 1.83125, 3.2125), "centre {c:?}");
    assert!(near(r, 2.837_011_246_452_858_5, 3.0), "radius {r}");
    assert!(near(start, 4.415_023_835_160_766, TAU), "start {start}");
    assert!(near(span, 1.522_025_508_449_459_6, TAU), "span {span}");
    assert!(!reversed, "a positive bulge runs counter-clockwise");
}

#[test]
fn bulge_arc_neg_001_is_clockwise_with_the_centre_on_the_right() {
    // bulge-arc-neg-001: p0 (3, -1), p1 (0.5, 2.5), bulge -3/10. Expected: centre
    // (4.404166667, 2.645833333), radius 3.906889392, start 3.178928551, sweep 1.165827178, CW.
    let e = polyline(vec![vertex(3.0, -1.0, -0.3), vertex(0.5, 2.5, 0.0)], false);
    let b = extract_polyline(&e).unwrap();
    let (c, r, start, span, reversed) = single_arc(&b);
    assert!(near_pt(c, 4.404_166_666_666_667, 2.645_833_333_333_333_5), "centre {c:?}");
    assert!(near(r, 3.906_889_392_115_193_3, 4.0), "radius {r}");
    assert!(near(start, 3.178_928_550_616_880_5, TAU), "start {start}");
    assert!(near(span, 1.165_827_177_911_468_4, TAU), "span {span}");
    assert!(reversed, "a negative bulge runs clockwise");
    assert_eq!(b.segments, vec![SegmentKind::ArcClockwise]);
}

#[test]
fn tan_eighth_pi_on_a_unit_chord_is_a_quarter_circle() {
    // chord (0,0)-(1,0), bulge tan(pi/8): the included angle is pi/2, the radius is 1/sqrt(2),
    // the centre (0.5, 0.5), and the arc starts at 5pi/4 on that circle.
    let b = std::f64::consts::FRAC_PI_8.tan();
    let e = polyline(vec![vertex(0.0, 0.0, b), vertex(1.0, 0.0, 0.0)], false);
    let x = extract_polyline(&e).unwrap();
    let (c, r, start, span, reversed) = single_arc(&x);
    assert!(near_pt(c, 0.5, 0.5), "centre {c:?}");
    assert!(near(r, std::f64::consts::FRAC_1_SQRT_2, 1.0), "radius {r}");
    assert!(near(span, std::f64::consts::FRAC_PI_2, TAU), "span {span}");
    assert!(near(start, 5.0 * std::f64::consts::FRAC_PI_4, TAU));
    assert!(!reversed);
    // Direction is checked on the curve itself: it starts at p0 and ends at p1.
    let d = x.curve.domain();
    let (p0, p1) = (x.curve.eval(d.lo()).unwrap(), x.curve.eval(d.hi()).unwrap());
    assert!(near_pt(p0, 0.0, 0.0), "{p0:?}");
    assert!(near_pt(p1, 1.0, 0.0), "{p1:?}");
}

#[test]
fn flipping_the_bulge_sign_flips_the_direction_and_the_centre_side() {
    let b = std::f64::consts::FRAC_PI_8.tan();
    let pos = extract_polyline(&polyline(
        vec![vertex(0.0, 0.0, b), vertex(1.0, 0.0, 0.0)],
        false,
    ))
    .unwrap();
    let neg = extract_polyline(&polyline(
        vec![vertex(0.0, 0.0, -b), vertex(1.0, 0.0, 0.0)],
        false,
    ))
    .unwrap();
    let (cp, _, _, _, rp) = single_arc(&pos);
    let (cn, _, _, _, rn) = single_arc(&neg);
    assert!(!rp && rn);
    assert!(near_pt(cp, 0.5, 0.5));
    assert!(near_pt(cn, 0.5, -0.5), "the centre moves to the right of the chord");
}

#[test]
fn the_closing_segment_of_a_closed_polyline_honours_the_last_bulge() {
    // A unit square with the closing segment bulged outward by a semicircle (bulge 1).
    let e = polyline(
        vec![
            vertex(0.0, 0.0, 0.0),
            vertex(1.0, 0.0, 0.0),
            vertex(1.0, 1.0, 0.0),
            vertex(0.0, 1.0, 1.0),
        ],
        true,
    );
    let b = extract_polyline(&e).unwrap();
    assert!(b.closed);
    assert_eq!(b.segments.len(), 4);
    assert_eq!(b.segments[3], SegmentKind::ArcCounterClockwise);
    let Curve::Composite(c) = &b.curve else {
        panic!("composite expected");
    };
    assert!(c.is_declared_closed());
    let (centre, radius, _, span, reversed) = arc_of(c.segments().last().unwrap());
    assert!(near_pt(centre, 0.0, 0.5), "centre {centre:?}");
    assert!(near(radius, 0.5, 1.0));
    assert!(near(span, std::f64::consts::PI, TAU));
    assert!(!reversed);
}

#[test]
fn the_last_bulge_of_an_open_polyline_is_not_a_segment() {
    let e = polyline(vec![vertex(0.0, 0.0, 0.0), vertex(1.0, 0.0, 0.7)], false);
    let b = extract_polyline(&e).unwrap();
    assert_eq!(b.segments, vec![SegmentKind::Straight]);
}

#[test]
fn coincident_vertices_are_classified_not_dropped() {
    let e = polyline(
        vec![
            vertex(0.0, 0.0, 0.0),
            vertex(1.0, 1.0, 0.0),
            vertex(1.0, 1.0, 0.0),
            vertex(2.0, 0.0, 0.0),
        ],
        false,
    );
    let err = extract_polyline(&e).unwrap_err();
    assert_eq!(err, BasicError::CoincidentVertices { index: 1 });
    let c = err.classification("LWPOLYLINE");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::CoincidentVertices));
}

#[test]
fn a_closed_polyline_whose_last_vertex_repeats_the_first_is_coincident() {
    let e = polyline(
        vec![
            vertex(0.0, 0.0, 0.0),
            vertex(2.0, 0.0, 0.0),
            vertex(2.0, 2.0, 0.0),
            vertex(0.0, 0.0, 0.0),
        ],
        true,
    );
    let err = extract_polyline(&e).unwrap_err();
    assert_eq!(err, BasicError::CoincidentVertices { index: 3 });
}

#[test]
fn coincident_vertices_around_a_bulge_are_classified() {
    let e = polyline(
        vec![vertex(1.0, 1.0, 0.5), vertex(1.0, 1.0, 0.0)],
        false,
    );
    assert_eq!(
        extract_polyline(&e).unwrap_err(),
        BasicError::CoincidentVertices { index: 0 }
    );
}

#[test]
fn non_finite_bulges_are_classified() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let e = polyline(vec![vertex(0.0, 0.0, bad), vertex(1.0, 0.0, 0.0)], false);
        let err = extract_polyline(&e).unwrap_err();
        assert!(
            matches!(err, BasicError::DegenerateBulge { index: 0, .. }),
            "{err:?}"
        );
        let c = err.classification("LWPOLYLINE");
        assert_eq!(c.reason(), Some(ReasonCode::DegenerateBulge));
        assert_eq!(c.class(), FidelityClass::Unsupported);
    }
}

#[test]
fn a_bulge_pvec_cannot_build_is_classified_as_a_bulge() {
    // The bulge is finite, and a chord of 1e12 with it puts the centre past PVec's bound.
    let e = polyline(
        vec![vertex(0.0, 0.0, 1.0e-300), vertex(1.0e12, 0.0, 0.0)],
        false,
    );
    let err = extract_polyline(&e).unwrap_err();
    assert!(
        matches!(err, BasicError::DegenerateBulge { index: 0, .. }),
        "{err:?}"
    );
}

#[test]
fn a_polyline_over_the_vertex_limit_is_refused_with_both_numbers() {
    let n = MAX_POLYLINE_VERTICES + 1;
    let vs = (0..n).map(|i| vertex(i as f64, 0.0, 0.0)).collect();
    let err = extract_polyline(&polyline(vs, false)).unwrap_err();
    assert_eq!(
        err,
        BasicError::TooManyVertices {
            limit: MAX_POLYLINE_VERTICES,
            observed: n
        }
    );
    let text = err.to_string();
    assert!(text.contains(&MAX_POLYLINE_VERTICES.to_string()), "{text}");
    assert!(text.contains(&n.to_string()), "{text}");
    assert_eq!(
        err.classification("LWPOLYLINE").reason(),
        Some(ReasonCode::PolylineLimitExceeded)
    );
}

#[test]
fn a_polyline_of_one_vertex_is_refused() {
    let err = extract_polyline(&polyline(vec![vertex(0.0, 0.0, 0.0)], false)).unwrap_err();
    assert!(matches!(err, BasicError::Degenerate { .. }), "{err:?}");
}

#[test]
fn a_polyline_with_a_non_plus_z_normal_is_not_mapped() {
    let mut e = polyline(vec![vertex(0.0, 0.0, 0.0), vertex(1.0, 0.0, 0.0)], false);
    if let RawGeometry::Polyline { normal, .. } = &mut e.geometry {
        *normal = [0.0, 0.0, -1.0];
    }
    let err = extract_polyline(&e).unwrap_err();
    assert_eq!(
        err,
        BasicError::ExtrusionNotApplied {
            normal: [0.0, 0.0, -1.0]
        }
    );
    assert_eq!(
        err.classification("LWPOLYLINE").reason(),
        Some(ReasonCode::ExtrusionNotApplied)
    );
}

// -------------------------------------------------------- CIRCLE, ARC

#[test]
fn circle_001_keeps_centre_and_radius_bit_for_bit() {
    // circle-001: centre (1.5, -2.25), radius 2.75.
    let b = extract_circle(&circle([1.5, -2.25, 0.0], 2.75)).unwrap();
    assert_eq!(b.kind, BasicKind::Circle);
    let Curve::Circle(c) = &b.curve else {
        panic!("a circle stays a circle, got {:?}", b.curve.kind());
    };
    assert_eq!(c.radius().to_bits(), 2.75_f64.to_bits());
    assert_eq!(
        (c.placement().origin().x, c.placement().origin().y),
        (1.5, -2.25)
    );
    assert_eq!(b.classification.class(), FidelityClass::SupportedExact);
}

#[test]
fn a_zero_radius_circle_is_classified_and_has_no_entity() {
    let err = extract_circle(&circle([0.0, 0.0, 0.0], 0.0)).unwrap_err();
    assert_eq!(err, BasicError::DegenerateRadius { radius: 0.0 });
    let c = err.classification("CIRCLE");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::DegenerateRadius));
    assert_eq!(c.source_class(), "CIRCLE");
    let row = err.warning("CIRCLE");
    assert!(row.contains("CIRCLE") && row.contains("degenerate-radius"), "{row}");
}

#[test]
fn negative_and_non_finite_radii_are_classified() {
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = extract_circle(&circle([0.0, 0.0, 0.0], bad)).unwrap_err();
        assert!(matches!(err, BasicError::DegenerateRadius { .. }), "{err:?}");
        let err = extract_arc(&arc([0.0, 0.0, 0.0], bad, 0.0, 1.0)).unwrap_err();
        assert!(matches!(err, BasicError::DegenerateRadius { .. }), "{err:?}");
    }
}

#[test]
fn arc_seam_001_keeps_the_counter_clockwise_angles() {
    // arc-seam-001: centre origin, radius 3, from 5.5 rad with a sweep of 1.5 rad, so the end
    // angle stored is 7.0 - 2pi = 0.716814693.
    let end = 7.0 - TAU;
    let b = extract_arc(&arc([0.0, 0.0, 0.0], 3.0, 5.5, end)).unwrap();
    assert_eq!(b.kind, BasicKind::Arc);
    let (c, r, start, span, reversed) = arc_of(&b.curve);
    assert!(near_pt(c, 0.0, 0.0));
    assert!(near(r, 3.0, 3.0));
    assert!(near(start, 5.5, TAU), "start {start}");
    assert!(near(span, 1.5, TAU), "span {span}");
    assert!(!reversed, "a DWG arc is counter-clockwise");
    let (s, e) = b.source_range.expect("the stored angles are kept");
    assert_eq!((s, e), (5.5, end), "radians as stored, not normalised");
    // It starts at the start angle and ends at the end angle.
    let d = b.curve.domain();
    let (p0, p1) = (b.curve.eval(d.lo()).unwrap(), b.curve.eval(d.hi()).unwrap());
    assert!(near_pt(p0, 3.0 * 5.5_f64.cos(), 3.0 * 5.5_f64.sin()), "{p0:?}");
    assert!(near_pt(p1, 3.0 * end.cos(), 3.0 * end.sin()), "{p1:?}");
}

#[test]
fn an_arc_whose_end_is_below_its_start_wraps_counter_clockwise() {
    // 270 degrees to 90 degrees, in radians: the arc runs through 0, a half turn.
    let b = extract_arc(&arc(
        [1.0, 1.0, 0.0],
        2.0,
        3.0 * std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
    ))
    .unwrap();
    let (_, _, _, span, reversed) = arc_of(&b.curve);
    assert!(near(span, std::f64::consts::PI, TAU));
    assert!(!reversed);
}

#[test]
fn an_arc_of_zero_sweep_is_classified() {
    let err = extract_arc(&arc([0.0, 0.0, 0.0], 1.0, 1.0, 1.0)).unwrap_err();
    assert!(matches!(err, BasicError::Degenerate { .. }), "{err:?}");
    assert_eq!(
        err.classification("ARC").reason(),
        Some(ReasonCode::DegenerateGeometry)
    );
}

#[test]
fn no_angle_is_converted_through_degrees() {
    // The same arc handed in as radians and as the radians of its degree value gives one answer,
    // and an angle of 90 deg expressed in radians is read as pi/2, not as 90 radians.
    let b = extract_arc(&arc([0.0, 0.0, 0.0], 1.0, 0.0, std::f64::consts::FRAC_PI_2)).unwrap();
    let (_, _, start, span, _) = arc_of(&b.curve);
    assert_eq!(start, 0.0);
    assert_eq!(span.to_bits(), std::f64::consts::FRAC_PI_2.to_bits());
}

// ------------------------------------------------------------ ELLIPSE

#[test]
fn ellipse_001_is_an_ellipse_with_the_stored_ratio() {
    // ellipse-001: centre (2, 1), major axis along (0.96, 0.28) with semi-major 3, ratio 0.98.
    let b = extract_ellipse(&ellipse(
        [2.0, 1.0, 0.0],
        [2.88, 0.84, 0.0],
        0.98,
        0.0,
        TAU,
    ))
    .unwrap();
    assert_eq!(b.kind, BasicKind::Ellipse);
    let Curve::Ellipse(el) = &b.curve else {
        panic!("a whole ellipse stays an ellipse, got {:?}", b.curve.kind());
    };
    assert!(near(el.semi_axis_1(), 3.0, 3.0), "{}", el.semi_axis_1());
    assert!(near(el.semi_axis_2(), 2.94, 3.0), "{}", el.semi_axis_2());
    let p = el.placement();
    assert!(near_pt(p.origin(), 2.0, 1.0));
    assert!(POLICY.near_point([p.x_axis().x, p.x_axis().y], [0.96, 0.28], 1.0));
    assert_eq!(b.classification.class(), FidelityClass::SupportedExact);
    assert_eq!(b.source_range, Some((0.0, TAU)));
}

#[test]
fn elliptical_arc_001_keeps_its_parameters_not_points() {
    // elliptical-arc-001: the same ellipse from 0.3 rad (eccentric anomaly) with a sweep of 2.2.
    let b = extract_ellipse(&ellipse(
        [2.0, 1.0, 0.0],
        [2.88, 0.84, 0.0],
        0.98,
        0.3,
        2.5,
    ))
    .unwrap();
    assert_eq!(b.kind, BasicKind::EllipticalArc);
    let Curve::Trimmed(t) = &b.curve else {
        panic!("an elliptical arc is a trimmed ellipse, got {:?}", b.curve.kind());
    };
    assert_eq!(t.basis().kind(), CurveKind::Ellipse);
    let trim = t.trim();
    assert!(near(trim.start, 0.3, TAU), "{}", trim.start);
    assert!(near(trim.span, 2.2, TAU), "{}", trim.span);
    assert!(!trim.reversed);
    assert_eq!(b.source_range, Some((0.3, 2.5)));
    assert_eq!(b.classification.class(), FidelityClass::SupportedEquivalent);
}

#[test]
fn a_zero_ratio_ellipse_is_classified() {
    let err = extract_ellipse(&ellipse([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 0.0, TAU))
        .unwrap_err();
    assert_eq!(err, BasicError::DegenerateAxisRatio { ratio: 0.0 });
    let c = err.classification("ELLIPSE");
    assert_eq!(c.reason(), Some(ReasonCode::DegenerateAxisRatio));
    assert_eq!(c.class(), FidelityClass::Unsupported);
}

#[test]
fn a_non_finite_or_oversized_ratio_is_classified() {
    for bad in [f64::NAN, f64::INFINITY, -0.5, 1.5] {
        let err = extract_ellipse(&ellipse([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], bad, 0.0, TAU))
            .unwrap_err();
        assert!(
            matches!(err, BasicError::DegenerateAxisRatio { .. }),
            "{bad}: {err:?}"
        );
    }
}

#[test]
fn a_zero_major_axis_is_degenerate() {
    let err = extract_ellipse(&ellipse([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5, 0.0, TAU))
        .unwrap_err();
    assert!(matches!(err, BasicError::Degenerate { .. }), "{err:?}");
    assert_eq!(
        err.classification("ELLIPSE").reason(),
        Some(ReasonCode::DegenerateGeometry)
    );
}

// ---------------------------------------------------------- dispatch

#[test]
fn extract_dispatches_on_the_geometry_and_refuses_the_rest() {
    assert_eq!(
        extract(&line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])).unwrap().kind,
        BasicKind::Line
    );
    assert_eq!(
        extract(&circle([0.0, 0.0, 0.0], 1.0)).unwrap().kind,
        BasicKind::Circle
    );
    let unmapped = entity(
        "ACAD_PROXY_ENTITY",
        RawGeometry::Unmapped {
            reason: "proxy".to_string(),
        },
    );
    assert_eq!(extract(&unmapped).unwrap_err(), BasicError::NotBasic);
    // A per-class function refuses another class rather than guessing.
    assert_eq!(
        extract_circle(&line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])).unwrap_err(),
        BasicError::NotBasic
    );
}

#[test]
fn every_refusal_has_a_classification_that_matches_its_report_row() {
    let errors = [
        BasicError::NotBasic,
        BasicError::DegenerateRadius { radius: 0.0 },
        BasicError::CoincidentVertices { index: 2 },
        BasicError::DegenerateAxisRatio { ratio: 0.0 },
        BasicError::DegenerateBulge {
            index: 1,
            bulge: f64::NAN,
        },
        BasicError::TooManyVertices {
            limit: 3,
            observed: 4,
        },
        BasicError::NonPlanar,
        BasicError::ExtrusionNotApplied { normal: Z },
    ];
    for e in errors {
        let c = e.classification("X");
        let reason = c.reason().expect("a refusal carries a reason");
        let row = e.warning("X");
        assert!(row.contains(reason.code()), "{row} lacks {}", reason.code());
        assert!(row.contains('X'));
        assert!(!c.class().produces_geometry() || c.class() == FidelityClass::RequiresExtension);
    }
}

// ------------------------------------------------------- source scan

/// The extraction path carries no `f32` and no degree conversion: the module is scanned, comments
/// excluded.
#[test]
fn the_module_has_no_f32_and_no_degree_conversion() {
    let src = include_str!("../src/pvec/dwg_basic.rs");
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in ["f32", "to_degrees", "to_radians", "180.0", "360.0", "as f32"] {
        assert!(!code.contains(banned), "dwg_basic.rs contains `{banned}`");
    }
}

/// This task defines no epsilon of its own: the module has no float literal that is a tolerance.
#[test]
fn the_module_defines_no_tolerance() {
    let src = include_str!("../src/pvec/dwg_basic.rs");
    for (n, l) in src.lines().enumerate() {
        let t = l.trim_start();
        if t.starts_with("//") {
            continue;
        }
        for banned in ["1e-", "1.0e-", "EPSILON", "epsilon", "f64::MIN_POSITIVE"] {
            assert!(!t.contains(banned), "line {}: `{banned}` in `{t}`", n + 1);
        }
    }
}
