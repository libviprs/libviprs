//! Issue #1567 (the geometry half of #1565, layouts slice 6): curves cut exactly against a clip.
//!
//! Written red, before the implementation. A circle, arc or ellipse that crosses a viewport clip
//! comes back as arcs of the same carrier over the kept parameter intervals, a spline as
//! sub-ranges of itself (knot insertion, PVec's), a polyline as the kept runs of its own
//! vertices. Nothing is tessellated, and every piece starts and ends on the clip boundary.
//!
//! Sampling lives here and only here: the tests sample a curve to check the pieces, the clipper
//! never does. Tolerances are built from `Policy::V1`, scaled by the clip and, for a spline, by
//! the parameter tolerance the clipper reports.
#![cfg(feature = "pvec-normalized")]

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::atomic::{AtomicU32, Ordering};

use libviprs::pvec::curve_clip::{CurveClip, CurveClipError, clip_curve};
use libviprs::pvec::fidelity::FidelityClass;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use pvec_geometry::{
    BSpline, Bezier, Circle, Composite, Curve, Curve2d, CurveKind, Ellipse, Line, Placement2,
    Point2, Polyline, RationalBSpline, Trimmed, Vector2,
};
use pvec_numeric::{Policy, detmath};

const CASES: u32 = 64;

fn p(x: f64, y: f64) -> Point2 {
    Point2::checked(x, y).unwrap()
}
fn place(cx: f64, cy: f64, dx: f64, dy: f64) -> Placement2 {
    Placement2::new(p(cx, cy), Vector2::new(dx, dy)).unwrap()
}
fn circle(cx: f64, cy: f64, r: f64) -> Curve {
    Curve::Circle(Circle::new(place(cx, cy, 1.0, 0.0), r).unwrap())
}
fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Point2> {
    vec![p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)]
}
fn square() -> Vec<Point2> {
    rect(0.0, 0.0, 10.0, 10.0)
}
/// An L-shaped, non-convex clip: the square with its top right quarter cut away.
fn ell() -> Vec<Point2> {
    vec![
        p(0.0, 0.0),
        p(10.0, 0.0),
        p(10.0, 5.0),
        p(5.0, 5.0),
        p(5.0, 10.0),
        p(0.0, 10.0),
    ]
}

/// The even-odd rule, written out here so the tests do not lean on the clipper's own.
fn inside(ring: &[Point2], q: Point2) -> bool {
    let mut odd = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + n - 1) % n]);
        if (a.y > q.y) != (b.y > q.y) {
            let x = a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if q.x < x {
                odd = !odd;
            }
        }
    }
    odd
}

/// The distance from `q` to the nearest edge of the ring.
fn boundary_distance(ring: &[Point2], q: Point2) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (ex, ey) = (b.x - a.x, b.y - a.y);
            let t = (((q.x - a.x) * ex + (q.y - a.y) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
            detmath::hypot(q.x - (a.x + t * ex), q.y - (a.y + t * ey))
        })
        .fold(f64::INFINITY, f64::min)
}

/// The tolerance a point of a piece is held to: `Policy::V1` at the scale of the clip.
fn tol(ring: &[Point2]) -> f64 {
    let mag = ring
        .iter()
        .fold(0.0f64, |m, q| m.max(q.x.abs()).max(q.y.abs()));
    Policy::V1.tol(mag, mag)
}

fn ends(c: &Curve) -> (Point2, Point2) {
    let d = c.domain();
    (c.eval(d.lo()).unwrap(), c.eval(d.hi()).unwrap())
}

fn split(r: CurveClip) -> (Vec<Curve>, FidelityClass, f64) {
    match r {
        CurveClip::Split {
            pieces,
            fidelity,
            parameter_tolerance,
        } => (pieces, fidelity, parameter_tolerance),
        other => panic!("expected a split, got {other:?}"),
    }
}

/// Every piece starts and ends on the boundary, unless that end is an end of the source, and its
/// middle is inside the clip.
fn assert_pieces_sit_in(ring: &[Point2], source: &Curve, pieces: &[Curve], slack: f64) {
    let (s0, s1) = ends(source);
    let own_end = |q: Point2| {
        [s0, s1]
            .iter()
            .any(|e| Policy::V1.near_point([q.x, q.y], [e.x, e.y], slack))
    };
    for c in pieces {
        let (a, b) = ends(c);
        for q in [a, b] {
            assert!(
                own_end(q) || boundary_distance(ring, q) <= slack,
                "a piece ends at ({}, {}), {} off the boundary",
                q.x,
                q.y,
                boundary_distance(ring, q)
            );
        }
        let d = c.domain();
        assert!(inside(ring, c.eval(0.5 * (d.lo() + d.hi())).unwrap()));
    }
}

fn trim_of(c: &Curve) -> pvec_numeric::Trim {
    match c {
        Curve::Trimmed(t) => t.trim(),
        other => panic!("expected a trimmed curve, got {:?}", other.kind()),
    }
}

// ---------------------------------------------------------------------------------------------
// Conics: split by parameter interval
// ---------------------------------------------------------------------------------------------

#[test]
fn a_circle_wholly_inside_is_kept_as_it_is() {
    assert_eq!(
        clip_curve(&circle(5.0, 5.0, 2.0), &square()).unwrap(),
        CurveClip::Inside
    );
}

#[test]
fn a_circle_wholly_outside_or_around_the_clip_is_dropped() {
    assert_eq!(
        clip_curve(&circle(20.0, 5.0, 2.0), &square()).unwrap(),
        CurveClip::Outside
    );
    // The clip lies inside the circle: no crossing, and the circle itself is outside.
    assert_eq!(
        clip_curve(&circle(5.0, 5.0, 50.0), &square()).unwrap(),
        CurveClip::Outside
    );
}

#[test]
fn a_circle_over_one_edge_keeps_one_arc_of_the_same_circle() {
    let source = circle(10.0, 5.0, 2.0);
    let (pieces, fidelity, ptol) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedExact);
    assert_eq!(ptol, 0.0);
    assert_eq!(pieces.len(), 1);
    let Curve::Trimmed(t) = &pieces[0] else {
        panic!("expected an arc")
    };
    // The carrier is the source circle, bit for bit: only the interval is new.
    assert_eq!(t.basis(), &source);
    // The crossings are (10, 7) and (10, 3), at angles pi/2 and 3 pi/2, from detmath::atan2.
    assert_eq!(t.trim().start, detmath::atan2(1.0, 0.0));
    assert!(Policy::V1.near(t.trim().span, PI, TAU));
    assert!(!t.trim().reversed);
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()));
}

#[test]
fn a_circle_over_the_seam_of_its_parameter_is_one_arc_not_two() {
    // Centred on the left edge: the kept half runs from -pi/2 through 0 to pi/2.
    let source = circle(0.0, 5.0, 2.0);
    let (pieces, _, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(pieces.len(), 1);
    let trim = trim_of(&pieces[0]);
    assert!(Policy::V1.near(trim.span, PI, TAU));
    assert!(Policy::V1.near(trim.start, 3.0 * FRAC_PI_2, TAU));
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()));
}

#[test]
fn a_circle_on_each_corner_keeps_the_quarter_inside() {
    for (cx, cy, start) in [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, FRAC_PI_2),
        (10.0, 10.0, PI),
        (0.0, 10.0, 3.0 * FRAC_PI_2),
    ] {
        let (pieces, _, _) = split(clip_curve(&circle(cx, cy, 1.0), &square()).unwrap());
        assert_eq!(pieces.len(), 1, "corner ({cx}, {cy})");
        let trim = trim_of(&pieces[0]);
        assert!(
            Policy::V1.near(trim.span, FRAC_PI_2, TAU),
            "corner ({cx}, {cy})"
        );
        assert!(
            Policy::V1.near(trim.start.rem_euclid(TAU), start, TAU)
                || Policy::V1.near(trim.start, TAU, TAU) && start == 0.0,
            "corner ({cx}, {cy}) starts at {}",
            trim.start
        );
    }
}

#[test]
fn a_circle_through_all_four_edges_keeps_four_arcs_in_order() {
    let source = circle(5.0, 5.0, 6.0);
    let (pieces, _, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(pieces.len(), 4);
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()));
    let starts: Vec<f64> = pieces.iter().map(|c| trim_of(c).start).collect();
    let mut sorted = starts.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(starts, sorted, "pieces come in parameter order");
}

#[test]
fn an_arc_keeps_its_direction_and_its_carrier() {
    // A clockwise arc (sense false) of a circle centred on the right edge, from just past 3 pi/2
    // down through pi to just short of pi/2: both ends are outside, the left half is kept.
    let source = Curve::Trimmed(
        Trimmed::new(
            circle(10.0, 5.0, 2.0),
            3.0 * FRAC_PI_2 + 0.5,
            FRAC_PI_2 - 0.5,
            false,
        )
        .unwrap(),
    );
    let (pieces, fidelity, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedExact);
    assert_eq!(pieces.len(), 1);
    let trim = trim_of(&pieces[0]);
    assert!(trim.reversed, "the piece runs the same way as the source");
    assert!(Policy::V1.near(trim.span, PI, TAU));
    // The piece starts where the source enters the clip: at (10, 3) for a clockwise run.
    let (a, b) = ends(&pieces[0]);
    assert!(boundary_distance(&square(), a) <= tol(&square()));
    assert!(
        a.y < b.y,
        "a clockwise run from the bottom crossing to the top one"
    );
    assert!(matches!(&pieces[0], Curve::Trimmed(t) if t.basis() == &circle(10.0, 5.0, 2.0)));
}

#[test]
fn an_arc_wholly_inside_is_kept_even_when_its_circle_crosses() {
    let source = Curve::Trimmed(Trimmed::new(circle(10.0, 5.0, 2.0), 2.5, 3.5, true).unwrap());
    assert_eq!(clip_curve(&source, &square()).unwrap(), CurveClip::Inside);
}

#[test]
fn a_rotated_ellipse_keeps_arcs_of_the_same_ellipse() {
    let source = Curve::Ellipse(Ellipse::new(place(10.0, 5.0, 0.8, 0.6), 4.0, 1.5).unwrap());
    let (pieces, fidelity, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedExact);
    assert_eq!(pieces.len(), 1);
    assert!(matches!(&pieces[0], Curve::Trimmed(t) if t.basis() == &source));
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()));
}

#[test]
fn a_polygon_clip_cuts_a_circle_at_its_concave_corner() {
    // Centred on the inner corner (5, 5) of the L: three quarters are inside.
    let source = circle(5.0, 5.0, 2.0);
    let (pieces, _, _) = split(clip_curve(&source, &ell()).unwrap());
    assert_eq!(pieces.len(), 1);
    assert!(Policy::V1.near(trim_of(&pieces[0]).span, 3.0 * FRAC_PI_2, TAU));
    assert_pieces_sit_in(&ell(), &source, &pieces, tol(&ell()));
}

#[test]
fn a_triangle_clip_cuts_a_circle_into_three_arcs() {
    let tri = vec![p(0.0, 0.0), p(12.0, 0.0), p(6.0, 10.0)];
    let source = circle(6.0, 4.0, 4.5);
    let (pieces, _, _) = split(clip_curve(&source, &tri).unwrap());
    assert_eq!(pieces.len(), 3);
    assert_pieces_sit_in(&tri, &source, &pieces, tol(&tri));
}

// ---------------------------------------------------------------------------------------------
// Lines and polylines against a polygon
// ---------------------------------------------------------------------------------------------

#[test]
fn a_polyline_keeps_its_own_vertices_and_gains_the_crossings() {
    let source = Curve::Polyline(
        Polyline::new(vec![p(-5.0, 2.0), p(3.0, 2.0), p(7.0, 8.0), p(15.0, 8.0)]).unwrap(),
    );
    let (pieces, fidelity, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedExact);
    assert_eq!(pieces.len(), 1);
    let Curve::Polyline(pl) = &pieces[0] else {
        panic!("expected a polyline")
    };
    let v = pl.vertices();
    assert_eq!(v.len(), 4);
    assert_eq!((v[0].x, v[0].y), (0.0, 2.0));
    assert_eq!(v[1], p(3.0, 2.0), "a source vertex is kept bit for bit");
    assert_eq!(v[2], p(7.0, 8.0));
    assert_eq!((v[3].x, v[3].y), (10.0, 8.0));
}

#[test]
fn a_polyline_through_the_notch_of_the_l_comes_out_in_two_runs() {
    let source =
        Curve::Polyline(Polyline::new(vec![p(1.0, 7.0), p(9.0, 7.0), p(9.0, 1.0)]).unwrap());
    let (pieces, _, _) = split(clip_curve(&source, &ell()).unwrap());
    assert_eq!(pieces.len(), 2);
    assert_pieces_sit_in(&ell(), &source, &pieces, tol(&ell()));
}

#[test]
fn a_trimmed_line_is_cut_to_the_clip() {
    let line = Curve::Line(Line::new(p(-5.0, 5.0), Vector2::new(1.0, 0.0)).unwrap());
    let source = Curve::Trimmed(Trimmed::new(line.clone(), 0.0, 20.0, true).unwrap());
    let (pieces, _, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(pieces.len(), 1);
    let trim = trim_of(&pieces[0]);
    assert_eq!((trim.start, trim.span), (5.0, 10.0));
    assert!(matches!(&pieces[0], Curve::Trimmed(t) if t.basis() == &line));
}

#[test]
fn an_unbounded_line_becomes_the_segment_inside() {
    let line = Curve::Line(Line::new(p(-5.0, 5.0), Vector2::new(1.0, 0.0)).unwrap());
    let (pieces, _, _) = split(clip_curve(&line, &square()).unwrap());
    assert_eq!(pieces.len(), 1);
    let (a, b) = ends(&pieces[0]);
    assert_eq!((a.x, a.y, b.x, b.y), (0.0, 5.0, 10.0, 5.0));
}

#[test]
fn a_bulged_polyline_keeps_whole_segments_together() {
    // Line from (2, 2) to (8, 2), a half circle bulging out past x = 10 back to (8, 8), then a line
    // home: the clip cuts the bulge, and the line plus the arc's first part stay one composite.
    let l1 = Curve::Trimmed(
        Trimmed::new(
            Curve::Line(Line::new(p(2.0, 2.0), Vector2::new(1.0, 0.0)).unwrap()),
            0.0,
            6.0,
            true,
        )
        .unwrap(),
    );
    let bulge = Curve::Trimmed(Trimmed::from_bulge(p(8.0, 2.0), p(8.0, 8.0), 1.0).unwrap());
    let l2 = Curve::Trimmed(
        Trimmed::new(
            Curve::Line(Line::new(p(8.0, 8.0), Vector2::new(-1.0, 0.0)).unwrap()),
            0.0,
            6.0,
            true,
        )
        .unwrap(),
    );
    let source = Curve::Composite(Composite::new(vec![l1, bulge, l2]).unwrap());
    let (pieces, fidelity, _) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedExact);
    assert_eq!(pieces.len(), 2);
    assert!(matches!(&pieces[0], Curve::Composite(c) if c.segments().len() == 2));
    assert!(matches!(&pieces[1], Curve::Composite(c) if c.segments().len() == 2));
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()));
}

// ---------------------------------------------------------------------------------------------
// Splines: knot insertion, SupportedEquivalent
// ---------------------------------------------------------------------------------------------

fn spline() -> BSpline {
    BSpline::new(
        3,
        vec![
            p(2.0, 2.0),
            p(6.0, 9.0),
            p(14.0, 7.0),
            p(16.0, 2.0),
            p(8.0, 1.0),
            p(4.0, 5.0),
        ],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
    )
    .unwrap()
}

#[test]
fn a_spline_is_cut_into_sub_ranges_of_itself() {
    let source = Curve::BSpline(spline());
    let (pieces, fidelity, ptol) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedEquivalent);
    assert!(ptol > 0.0, "the parameter tolerance is reported");
    assert_eq!(pieces.len(), 2);
    for c in &pieces {
        assert_eq!(c.kind(), CurveKind::BSpline, "the kind is kept");
        // A sub-range keeps the parameter, so every point of the piece is the source's own.
        let d = c.domain();
        for k in 0..=8 {
            let t = d.lo() + (d.hi() - d.lo()) * f64::from(k) / 8.0;
            let (a, b) = (c.eval(t).unwrap(), source.eval(t).unwrap());
            assert!(Policy::V1.near_point([a.x, a.y], [b.x, b.y], 16.0));
        }
    }
    // An end found to `ptol` in the parameter is within `ptol` times the speed of the boundary.
    let speed = 64.0;
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()) + ptol * speed);
}

#[test]
fn a_bezier_stays_a_bezier() {
    let source = Curve::Bezier(
        Bezier::new(vec![p(2.0, 5.0), p(8.0, 14.0), p(14.0, 5.0), p(18.0, 0.0)]).unwrap(),
    );
    let (pieces, fidelity, ptol) = split(clip_curve(&source, &square()).unwrap());
    assert_eq!(fidelity, FidelityClass::SupportedEquivalent);
    assert!(!pieces.is_empty());
    assert!(pieces.iter().all(|c| c.kind() == CurveKind::Bezier));
    assert_pieces_sit_in(&square(), &source, &pieces, tol(&square()) + ptol * 64.0);
}

#[test]
fn a_spline_wholly_inside_or_outside_is_kept_or_dropped_whole() {
    let inner = BSpline::new(
        2,
        vec![p(2.0, 2.0), p(5.0, 8.0), p(8.0, 2.0)],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    assert_eq!(
        clip_curve(&Curve::BSpline(inner), &square()).unwrap(),
        CurveClip::Inside
    );
    let outer = BSpline::new(
        2,
        vec![p(20.0, 2.0), p(25.0, 8.0), p(28.0, 2.0)],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    assert_eq!(
        clip_curve(&Curve::BSpline(outer), &square()).unwrap(),
        CurveClip::Outside
    );
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

#[test]
fn a_rational_spline_is_refused_by_kind() {
    let r = RationalBSpline::new(
        2,
        vec![p(2.0, 2.0), p(12.0, 8.0), p(8.0, 2.0)],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![1.0, 0.5, 1.0],
    )
    .unwrap();
    assert_eq!(
        clip_curve(&Curve::RationalBSpline(r), &square()),
        Err(CurveClipError::Unsupported(CurveKind::RationalBSpline))
    );
}

#[test]
fn a_clip_of_fewer_than_three_vertices_is_refused() {
    assert_eq!(
        clip_curve(&circle(5.0, 5.0, 2.0), &[p(0.0, 0.0), p(1.0, 1.0)]),
        Err(CurveClipError::Ring)
    );
}

#[test]
fn the_same_input_gives_the_same_bits() {
    let source = Curve::Ellipse(Ellipse::new(place(9.0, 4.0, 0.6, 0.8), 3.0, 1.25).unwrap());
    let ring = vec![p(0.0, 0.0), p(11.0, 1.0), p(10.0, 9.0), p(1.0, 10.0)];
    assert_eq!(
        clip_curve(&source, &ring).unwrap(),
        clip_curve(&source, &ring).unwrap()
    );
}

// ---------------------------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------------------------

/// The source parameter intervals the pieces cover. A trimmed piece maps back through its basis
/// interval (the same carrier as the source's), a spline piece keeps the source parameter.
fn covered(source: &Curve, pieces: &[Curve]) -> Vec<(f64, f64)> {
    pieces
        .iter()
        .map(|c| match (source, c) {
            (Curve::Circle(_) | Curve::Ellipse(_), Curve::Trimmed(t)) => {
                (t.trim().start, t.trim().start + t.trim().span)
            }
            (Curve::BSpline(_), Curve::BSpline(_)) => (c.domain().lo(), c.domain().hi()),
            other => panic!("no coverage rule for {:?}", other.1.kind()),
        })
        .collect()
}

fn is_covered(intervals: &[(f64, f64)], t: f64, periodic: bool) -> bool {
    intervals.iter().any(|&(a, b)| {
        if periodic {
            let along = (t - a).rem_euclid(TAU);
            along <= b - a
        } else {
            t >= a && t <= b
        }
    })
}

/// A point of the source is kept exactly when it is inside the clip, checked on samples that sit
/// clearly off the boundary.
fn check_kept_iff_inside(source: &Curve, ring: &[Point2], periodic: bool) {
    let r = clip_curve(source, ring).unwrap();
    let d = source.domain();
    let margin = 1e3 * tol(ring);
    let intervals = match &r {
        CurveClip::Inside => vec![(d.lo(), d.hi())],
        CurveClip::Outside => vec![],
        CurveClip::Split { pieces, .. } => {
            assert_pieces_sit_in(ring, source, pieces, margin);
            covered(source, pieces)
        }
    };
    for k in 0..256 {
        let t = d.lo() + (d.hi() - d.lo()) * (f64::from(k) + 0.5) / 256.0;
        let q = source.eval(t).unwrap();
        if boundary_distance(ring, q) <= margin {
            continue;
        }
        assert_eq!(
            is_covered(&intervals, t, periodic),
            inside(ring, q),
            "parameter {t} at ({}, {})",
            q.x,
            q.y
        );
    }
}

fn quad() -> impl Strategy<Value = Vec<Point2>> {
    (
        [-2.0f64..2.0, -2.0f64..2.0, -2.0f64..2.0, -2.0f64..2.0],
        [-2.0f64..2.0, -2.0f64..2.0, -2.0f64..2.0, -2.0f64..2.0],
    )
        .prop_map(|(dx, dy)| {
            vec![
                p(dx[0], dy[0]),
                p(10.0 + dx[1], dy[1]),
                p(10.0 + dx[2], 10.0 + dy[2]),
                p(dx[3], 10.0 + dy[3]),
            ]
        })
}

#[test]
fn property_a_conic_keeps_exactly_what_is_inside() {
    let seen = AtomicU32::new(0);
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    let strategy = (
        -4.0f64..14.0,
        -4.0f64..14.0,
        0.5f64..9.0,
        0.2f64..1.0,
        0.0f64..TAU,
        quad(),
    );
    runner
        .run(&strategy, |(cx, cy, a, ratio, angle, ring)| {
            seen.fetch_add(1, Ordering::Relaxed);
            let f = place(cx, cy, angle.cos(), angle.sin());
            let source = Curve::Ellipse(Ellipse::new(f, a, a * ratio).unwrap());
            check_kept_iff_inside(&source, &ring, true);
            check_kept_iff_inside(&circle(cx, cy, a), &ring, true);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen.load(Ordering::Relaxed), CASES);
}

#[test]
fn property_a_spline_keeps_exactly_what_is_inside() {
    let seen = AtomicU32::new(0);
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    let strategy = (
        proptest::collection::vec((-6.0f64..16.0, -6.0f64..16.0), 4..9),
        1usize..4,
        quad(),
    );
    runner
        .run(&strategy, |(cps, degree, ring)| {
            seen.fetch_add(1, Ordering::Relaxed);
            let n = cps.len();
            let inner = n - degree - 1;
            let mut knots = vec![0.0; degree + 1];
            knots.extend((1..=inner).map(|i| i as f64));
            knots.extend(std::iter::repeat_n((inner + 1) as f64, degree + 1));
            let pts = cps.iter().map(|&(x, y)| p(x, y)).collect();
            let source = Curve::BSpline(BSpline::new(degree, pts, knots).unwrap());
            check_kept_iff_inside(&source, &ring, false);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen.load(Ordering::Relaxed), CASES);
}
