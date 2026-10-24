//! Issue #1091 (N6.4): spatial bounds and entity-reference inputs.
//!
//! Written red, before the implementation. Sampling lives here and only here: the bounds
//! themselves must never be sampled.
#![cfg(feature = "pvec-normalized")]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use libviprs::pvec::{BoundsError, BoundsMethod, bounds_inputs, curve_bounds};
use pvec_entity::EntityId;
use pvec_geometry::{
    BSpline, Bezier, Circle, Curve, Curve2d, Ellipse, Placement2, Point2, Polyline,
    RationalBSpline, Trimmed, Vector2,
};

const BOUNDS_SRC: &str = include_str!("../src/pvec/bounds.rs");

fn p(x: f64, y: f64) -> Point2 {
    Point2::checked(x, y).unwrap()
}
fn place(cx: f64, cy: f64, dx: f64, dy: f64) -> Placement2 {
    Placement2::new(p(cx, cy), Vector2::new(dx, dy)).unwrap()
}
fn circle(cx: f64, cy: f64, r: f64) -> Curve {
    Curve::Circle(Circle::new(place(cx, cy, 1.0, 0.0), r).unwrap())
}
fn arc(r: f64, t1: f64, t2: f64) -> Curve {
    Curve::Trimmed(Trimmed::new(circle(0.0, 0.0, r), t1, t2, true).unwrap())
}
fn ellipse() -> Curve {
    Curve::Ellipse(Ellipse::new(place(2.0, -1.0, 0.8, 0.6), 5.0, 2.0).unwrap())
}
fn cubic() -> Curve {
    Curve::Bezier(
        Bezier::new(vec![
            p(0.0, 0.0),
            p(10.0, 30.0),
            p(20.0, -30.0),
            p(30.0, 0.0),
        ])
        .unwrap(),
    )
}
fn quartic() -> Curve {
    Curve::Bezier(
        Bezier::new(vec![
            p(0.0, 0.0),
            p(5.0, 20.0),
            p(10.0, -25.0),
            p(15.0, 20.0),
            p(20.0, 0.0),
        ])
        .unwrap(),
    )
}
fn bspline() -> Curve {
    Curve::BSpline(
        BSpline::new(
            3,
            vec![
                p(0.0, 0.0),
                p(1.0, 4.0),
                p(3.0, -3.0),
                p(5.0, 5.0),
                p(7.0, 0.0),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap(),
    )
}
fn nurbs() -> Curve {
    Curve::RationalBSpline(
        RationalBSpline::new(
            2,
            vec![p(0.0, 0.0), p(2.0, 4.0), p(4.0, 0.0)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 1.0],
        )
        .unwrap(),
    )
}
fn polyline() -> Curve {
    Curve::Polyline(Polyline::new(vec![p(0.0, 0.0), p(4.0, 9.0), p(-3.0, 2.0)]).unwrap())
}

#[test]
fn circle_at_origin_is_exactly_minus_r_to_r_bit_for_bit() {
    for r in [1.0, 0.1, 3.7, 12345.678, 1e-9, 1e9] {
        let (b, m) = curve_bounds(&circle(0.0, 0.0, r)).unwrap();
        assert_eq!(m, BoundsMethod::Exact);
        for (got, want) in [
            (b.min().x, -r),
            (b.min().y, -r),
            (b.max().x, r),
            (b.max().y, r),
        ] {
            assert_eq!(got.to_bits(), want.to_bits(), "radius {r}");
        }
    }
}

#[test]
fn a_quarter_arc_is_tighter_than_the_full_circle() {
    let (full, _) = curve_bounds(&circle(0.0, 0.0, 10.0)).unwrap();
    let (q, m) = curve_bounds(&arc(10.0, 0.0, FRAC_PI_2)).unwrap();
    assert_eq!(m, BoundsMethod::Exact);
    assert!(q.min().x > full.min().x && q.min().y > full.min().y);
    assert!(q.max().x <= full.max().x && q.max().y <= full.max().y);
    assert!(
        q.min().x >= -1e-9 && q.min().y >= -1e-9,
        "first quadrant only"
    );
    assert_eq!(q.max().x.to_bits(), 10.0f64.to_bits());
    assert_eq!(q.max().y.to_bits(), 10.0f64.to_bits());
}

fn assert_encloses(name: &str, c: &Curve) {
    let (b, _) = curve_bounds(c).unwrap();
    let d = c.domain();
    let n = 20_000;
    for i in 0..=n {
        let t = d.lo() + d.width() * (i as f64 / n as f64);
        let q = c.eval(t).unwrap();
        assert!(
            q.x >= b.min().x && q.x <= b.max().x && q.y >= b.min().y && q.y <= b.max().y,
            "{name}: sample at t={t} ({}, {}) outside {:?}",
            q.x,
            q.y,
            b
        );
    }
}

#[test]
fn dense_sampling_never_leaves_the_reported_bounds() {
    assert_encloses("circle", &circle(3.0, -2.0, 7.5));
    assert_encloses("arc", &arc(4.0, 0.3, 4.0));
    assert_encloses("arc wraps", &arc(4.0, 5.0, 0.7 + TAU));
    assert_encloses("ellipse", &ellipse());
    assert_encloses("polyline", &polyline());
    assert_encloses("cubic", &cubic());
    assert_encloses("quartic", &quartic());
    assert_encloses("bspline", &bspline());
    assert_encloses("nurbs", &nurbs());
}

#[test]
fn closed_form_kinds_are_exact_and_splines_are_labelled_hull() {
    for (n, c) in [
        ("circle", circle(0.0, 0.0, 2.0)),
        ("arc", arc(2.0, 0.0, PI)),
        ("ellipse", ellipse()),
        ("polyline", polyline()),
        ("cubic", cubic()),
    ] {
        assert_eq!(curve_bounds(&c).unwrap().1, BoundsMethod::Exact, "{n}");
    }
    for (n, c) in [
        ("quartic", quartic()),
        ("bspline", bspline()),
        ("nurbs", nurbs()),
    ] {
        assert_eq!(
            curve_bounds(&c).unwrap().1,
            BoundsMethod::HullUpperBound,
            "{n}"
        );
    }
}

#[test]
fn a_hull_bound_is_never_recorded_as_exact_in_the_report() {
    let curves = [
        circle(0.0, 0.0, 1.0),
        quartic(),
        bspline(),
        nurbs(),
        cubic(),
    ];
    let items: Vec<_> = curves
        .iter()
        .enumerate()
        .map(|(i, c)| (EntityId::new(i as u32).unwrap(), c))
        .collect();
    let r = bounds_inputs(items).unwrap();
    assert_eq!(r.entries().len(), 5);
    assert_eq!(r.exact_count(), 2);
    assert_eq!(r.hull_count(), 3);
    for (e, hull) in r.entries().iter().zip([false, true, true, true, false]) {
        assert_eq!(
            e.method() == BoundsMethod::HullUpperBound,
            hull,
            "{}",
            e.id()
        );
    }
}

#[test]
fn a_composite_is_exact_only_when_every_part_is() {
    use pvec_geometry::Composite;
    // the arc ends at (0, 1), so each tail starts there
    let tail = Curve::Polyline(Polyline::new(vec![p(0.0, 1.0), p(4.0, 9.0)]).unwrap());
    let spline = Curve::BSpline(
        BSpline::new(
            3,
            vec![
                p(0.0, 1.0),
                p(1.0, 4.0),
                p(3.0, -3.0),
                p(5.0, 5.0),
                p(7.0, 0.0),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap(),
    );
    let ok = Curve::Composite(Composite::new(vec![arc(1.0, 0.0, FRAC_PI_2), tail]).unwrap());
    assert_eq!(curve_bounds(&ok).unwrap().1, BoundsMethod::Exact);
    let hull = Curve::Composite(Composite::new(vec![arc(1.0, 0.0, FRAC_PI_2), spline]).unwrap());
    assert_eq!(curve_bounds(&hull).unwrap().1, BoundsMethod::HullUpperBound);
    // a spline trimmed from a longer one is still only a hull
    assert_encloses("composite", &hull);
}

#[test]
fn the_report_carries_pvec_ids_and_the_drawing_union() {
    let a = circle(0.0, 0.0, 1.0);
    let b = circle(10.0, 20.0, 2.0);
    let r = bounds_inputs([
        (EntityId::new(7).unwrap(), &a),
        (EntityId::new(3).unwrap(), &b),
    ])
    .unwrap();
    assert_eq!(r.entries()[0].id(), EntityId::new(7).unwrap());
    assert_eq!(r.entries()[1].id(), EntityId::new(3).unwrap());
    let d = r.drawing_bounds().unwrap();
    assert_eq!(
        (d.min().x, d.min().y, d.max().x, d.max().y),
        (-1.0, -1.0, 12.0, 22.0)
    );
}

#[test]
fn an_empty_input_has_no_drawing_bounds() {
    let r = bounds_inputs(Vec::<(EntityId, &Curve)>::new()).unwrap();
    assert!(r.entries().is_empty() && r.drawing_bounds().is_none());
}

#[test]
fn an_unbounded_line_is_a_typed_error_not_a_box() {
    let line = Curve::Line(pvec_geometry::Line::new(p(0.0, 0.0), Vector2::new(1.0, 0.0)).unwrap());
    assert_eq!(curve_bounds(&line).unwrap_err(), BoundsError::Unbounded);
}

#[test]
fn bounds_are_computed_without_sampling_in_src() {
    let code: String = BOUNDS_SRC
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in [
        ".eval(",
        "derivative(",
        "sample",
        "tessell",
        "for i in 0..",
        "powi(",
        "cos(",
        "sin(",
    ] {
        assert!(
            !code.contains(banned),
            "src/pvec/bounds.rs contains `{banned}`"
        );
    }
}
