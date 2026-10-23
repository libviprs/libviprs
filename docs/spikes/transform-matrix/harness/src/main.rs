//! N1.5 measurement harness: one fixture of each source kind through one instance of each
//! transform class, via PVec's `Curve2d::transform`. Prints a TSV on stdout.
//!
//! What this is not: an evaluator, a tessellator or a tolerance. Every curve operation is a call
//! into `pvec-geometry`, and every comparison goes through `pvec_numeric::Policy::V1`. The
//! reference for a measured point is "map the original point with the transform, evaluate the
//! original curve with PVec", so for splines the check of the transform itself is the exact
//! control-point comparison, and the pointwise one is self-consistency of PVec's evaluator. An
//! independent evaluator is the job of the geometry oracle (P2.x) and the probes of O1.6.

use pvec_geometry::{
    Circle, Composite, Curve, Curve2d, Ellipse, GeometryError, Line, ParamMap, Placement2, Point2,
    Polyline, RationalBSpline, BSpline, Side, Transform2, Trimmed, Vector2,
};
use pvec_numeric::Policy;
use std::f64::consts::{PI, TAU};

const SAMPLES: usize = 256;

type Res<T> = Result<T, GeometryError>;

struct Fixture {
    id: &'static str,
    row: &'static str,
    curve: Curve,
}

fn p(x: f64, y: f64) -> Point2 {
    Point2::new(x, y)
}

fn fixtures() -> Res<Vec<Fixture>> {
    let circle_frame = Placement2::new(p(3.0, -2.0), Vector2::new(1.0, 0.0))?;
    let circle = Circle::new(circle_frame, 5.0)?;
    let ellipse_frame = Placement2::new(p(1.0, 1.0), Vector2::new(0.8, 0.6))?;
    let ellipse = Ellipse::new(ellipse_frame, 4.0, 2.0)?;
    let line = Line::new(p(1.0, 2.0), Vector2::new(3.0, 4.0))?;
    let bulge_circle = Circle::new(Placement2::new(p(4.0, 2.0), Vector2::new(1.0, 0.0))?, 2.0)?;
    let bulge = Composite::new(vec![
        Curve::Trimmed(Trimmed::new(Curve::Line(Line::new(p(0.0, 0.0), Vector2::new(1.0, 0.0))?), 0.0, 4.0, true)?),
        Curve::Trimmed(Trimmed::new(Curve::Circle(bulge_circle), 1.5 * PI, TAU, true)?),
    ])?;
    Ok(vec![
        Fixture { id: "F1", row: "line", curve: Curve::Trimmed(Trimmed::new(Curve::Line(line), 0.5, 6.0, true)?) },
        Fixture {
            id: "F2",
            row: "polyline",
            curve: Curve::Polyline(Polyline::new(vec![p(0.0, 0.0), p(4.0, 0.0), p(5.0, 3.0), p(2.0, 5.0)])?),
        },
        Fixture { id: "F3", row: "polyline with bulge", curve: Curve::Composite(bulge) },
        Fixture { id: "F4", row: "circular arc", curve: Curve::Trimmed(Trimmed::new(Curve::Circle(circle), 0.3, 2.1, true)?) },
        Fixture { id: "F5", row: "circle", curve: Curve::Circle(circle) },
        Fixture { id: "F6", row: "ellipse", curve: Curve::Ellipse(ellipse) },
        Fixture { id: "F7", row: "elliptical arc", curve: Curve::Trimmed(Trimmed::new(Curve::Ellipse(ellipse), 0.4, 4.0, true)?) },
        Fixture {
            id: "F8",
            row: "b-spline",
            curve: Curve::BSpline(BSpline::new(
                3,
                vec![p(0.0, 0.0), p(1.0, 2.0), p(3.0, 3.0), p(4.0, 0.0), p(6.0, -1.0), p(7.0, 2.0)],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            )?),
        },
        Fixture {
            id: "F9",
            row: "nurbs",
            curve: Curve::RationalBSpline(RationalBSpline::new(
                2,
                vec![p(0.0, 0.0), p(2.0, 3.0), p(5.0, 3.0), p(6.0, 0.0), p(8.0, 1.0)],
                vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                vec![1.0, 0.7, 1.4, 1.0, 0.9],
            )?),
        },
    ])
}

/// One instance of each transform class. `build` is a closure so a refusal at construction is a
/// measured outcome and not a harness failure.
struct Xf {
    id: &'static str,
    class: &'static str,
    build: Box<dyn Fn() -> Res<Transform2>>,
    band: &'static str,
}

fn transforms() -> Vec<Xf> {
    let rel = Policy::V1.rel;
    vec![
        Xf { id: "T1", class: "translation", build: Box::new(|| Transform2::translate(7.5, -3.25)), band: "" },
        Xf { id: "T2", class: "rotation", build: Box::new(|| Transform2::rotate(0.6)), band: "" },
        Xf { id: "T3", class: "uniform scale", build: Box::new(|| Transform2::scale(2.5, 2.5)), band: "" },
        Xf { id: "T4", class: "non-uniform scale", build: Box::new(|| Transform2::scale(3.0, 0.5)), band: "" },
        Xf { id: "T5", class: "reflection", build: Box::new(Transform2::mirror_x), band: "" },
        Xf {
            id: "T6",
            class: "general affine",
            build: Box::new(|| Transform2::new([1.3, 0.4, -0.7, 0.9, 2.0, -1.0])),
            band: "",
        },
        // Rows 1 and 2 are dependent, so the determinant is exactly zero.
        Xf { id: "T7", class: "singular", build: Box::new(|| Transform2::new([1.0, 2.0, 2.0, 4.0, 0.0, 0.0])), band: "" },
        Xf { id: "T8", class: "shear", build: Box::new(|| Transform2::shear(0.7, 0.0)), band: "" },
        // Near-singular: one side of the NP-19 boundary each, derived from the policy constant.
        Xf {
            id: "T9",
            class: "near-singular below the NP-19 bound",
            build: Box::new(move || Transform2::scale(1.0, rel * 0.5)),
            band: "near-singular",
        },
        Xf {
            id: "T10",
            class: "near-singular above the NP-19 bound",
            build: Box::new(move || Transform2::scale(1.0, rel * 2.0)),
            band: "near-singular",
        },
        // A coefficient that is not a number never builds a map (NP-01), so N3.4 can name the insert.
        Xf {
            id: "T11",
            class: "non-finite coefficient",
            build: Box::new(|| Transform2::new([f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0])),
            band: "",
        },
    ]
}

fn kind_name(c: &Curve) -> String {
    match c {
        Curve::Line(_) => "line".into(),
        Curve::Circle(_) => "circle".into(),
        Curve::Ellipse(_) => "ellipse".into(),
        Curve::Polyline(_) => "polyline".into(),
        Curve::Trimmed(t) => format!("trimmed_curve over {}", kind_name(t.basis())),
        Curve::Composite(_) => "composite_curve".into(),
        Curve::Bezier(_) => "bezier_curve".into(),
        Curve::BSpline(_) => "b_spline_curve_with_knots".into(),
        Curve::RationalBSpline(_) => "rational_b_spline_curve".into(),
        _ => "unknown".into(),
    }
}

fn dist(a: Point2, b: Point2) -> f64 {
    a.distance(b)
}

fn fmt_e(x: f64) -> String {
    format!("{x:.3e}")
}

fn fmt_p(x: f64) -> String {
    // Parameters and coordinates in the worked example, to nine decimals.
    format!("{:.9}", x + 0.0)
}

struct Rec {
    id: String,
    fixture: &'static str,
    transform: &'static str,
    class: &'static str,
    outcome: &'static str,
    result: String,
    rule: String,
    error: String,
    worst: String,
    tol: String,
    ratio: String,
    notes: Vec<(String, String)>,
}

fn emit(rec: &Rec) {
    let notes: Vec<String> = rec.notes.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        rec.id, rec.fixture, rec.transform, rec.class, rec.outcome, rec.result, rec.rule, rec.error, rec.worst,
        rec.tol, rec.ratio, SAMPLES, notes.join(";")
    );
}

fn sweep(c: &Curve) -> &'static str {
    // Turning direction of the traversal at the middle of the domain: the sign of the cross
    // product of the first and second derivative. Exactly zero means no turning (a straight run).
    let d = c.domain();
    let mid = d.lo() + 0.5 * d.width();
    let (Ok(d1), Ok(d2)) = (c.derivative(mid, 1, Side::Both), c.derivative(mid, 2, Side::Both)) else {
        return "n/a";
    };
    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross > 0.0 {
        "ccw"
    } else if cross < 0.0 {
        "cw"
    } else {
        "straight"
    }
}

fn measure(f: &Fixture, x: &Xf, id: String, m: Transform2) -> Rec {
    let mut rec = Rec {
        id,
        fixture: f.id,
        transform: x.id,
        class: x.class,
        outcome: "ok",
        result: String::new(),
        rule: "-".into(),
        error: "-".into(),
        worst: "-".into(),
        tol: "-".into(),
        ratio: "-".into(),
        notes: Vec::new(),
    };
    if !x.band.is_empty() {
        rec.notes.push(("band".into(), x.band.into()));
        rec.notes.push(("derived_from".into(), "Policy::V1.rel".into()));
    }
    rec.notes.push((
        "det_sign".into(),
        if m.determinant() > 0.0 { "positive" } else { "negative" }.into(),
    ));
    let (new, map): (Curve, ParamMap) = match f.curve.transform(&m) {
        Ok(v) => v,
        Err(e) => {
            rec.outcome = "refused";
            rec.result = "-".into();
            rec.error = format!("{e:?}");
            rec.rule = e.rule().to_string();
            rec.notes.push(("curve_emitted".into(), "false".into()));
            return rec;
        }
    };
    rec.result = kind_name(&new);
    rec.notes.push(("param_offset".into(), fmt_p(map.offset)));
    rec.notes.push(("param_scale".into(), fmt_p(map.scale)));

    let d = f.curve.domain();
    let periodic = matches!(new, Curve::Circle(_) | Curve::Ellipse(_));
    let mut worst = 0.0f64;
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    let mut mag = 0.0f64;
    let mut ends = [(Point2::default(), Point2::default()); 2];
    for i in 0..=SAMPLES {
        let t = d.lo() + d.width() * (i as f64) / (SAMPLES as f64);
        let (Ok(old_pt), t2) = (f.curve.eval(t), map.apply(t)) else {
            rec.notes.push(("eval_error".into(), "original".into()));
            continue;
        };
        let t2 = if periodic { t2.rem_euclid(TAU) } else { t2 };
        let want = m.apply_point(old_pt);
        let got = match new.eval(t2) {
            Ok(g) => g,
            Err(e) => {
                rec.notes.push(("eval_error".into(), format!("{e:?}")));
                continue;
            }
        };
        worst = worst.max(dist(want, got));
        lo = [lo[0].min(want.x), lo[1].min(want.y)];
        hi = [hi[0].max(want.x), hi[1].max(want.y)];
        mag = mag.max(want.x.abs()).max(want.y.abs());
        if i == 0 {
            ends[0] = (want, got);
        }
        if i == SAMPLES {
            ends[1] = (want, got);
        }
    }
    // The feature scale of NP-04 is the extent of the mapped sample set, magnitude is its largest
    // coordinate. Both are read off the samples, so no curve mathematics is done here.
    let scale = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt();
    let tol = Policy::V1.tol(scale, mag);
    rec.worst = fmt_e(worst);
    rec.tol = fmt_e(tol);
    rec.ratio = fmt_e(worst / tol);

    let start_ok = Policy::V1.near_point([ends[0].0.x, ends[0].0.y], [ends[0].1.x, ends[0].1.y], scale);
    let end_ok = Policy::V1.near_point([ends[1].0.x, ends[1].0.y], [ends[1].1.x, ends[1].1.y], scale);
    rec.notes.push(("start_preserved".into(), start_ok.to_string()));
    rec.notes.push(("end_preserved".into(), end_ok.to_string()));

    match (&f.curve, &new) {
        (Curve::BSpline(a), Curve::BSpline(b)) => {
            spline_notes(&mut rec, a.degree(), a.control_points(), a.knots(), None, b.degree(), b.control_points(), b.knots(), None, &m)
        }
        (Curve::RationalBSpline(a), Curve::RationalBSpline(b)) => spline_notes(
            &mut rec,
            a.degree(),
            a.control_points(),
            a.knots(),
            Some(a.weights()),
            b.degree(),
            b.control_points(),
            b.knots(),
            Some(b.weights()),
            &m,
        ),
        (Curve::Trimmed(a), Curve::Trimmed(b)) => {
            rec.notes.push(("reversed_before".into(), a.trim().reversed.to_string()));
            rec.notes.push(("reversed_after".into(), b.trim().reversed.to_string()));
            rec.notes.push(("trim_start_before".into(), fmt_p(a.trim().start)));
            rec.notes.push(("trim_span_before".into(), fmt_p(a.trim().span)));
            rec.notes.push(("trim_start_after".into(), fmt_p(b.trim().start)));
            rec.notes.push(("trim_span_after".into(), fmt_p(b.trim().span)));
            rec.notes.push(("sweep_before".into(), sweep(&f.curve).into()));
            rec.notes.push(("sweep_after".into(), sweep(&new).into()));
            match (a.basis(), b.basis()) {
                (Curve::Circle(x0), Curve::Circle(x1)) => {
                    rec.notes.push(("centre_before".into(), format!("({},{})", fmt_p(x0.placement().origin().x), fmt_p(x0.placement().origin().y))));
                    rec.notes.push(("centre_after".into(), format!("({},{})", fmt_p(x1.placement().origin().x), fmt_p(x1.placement().origin().y))));
                    rec.notes.push(("x_axis_before".into(), format!("({},{})", fmt_p(x0.placement().x_axis().x), fmt_p(x0.placement().x_axis().y))));
                    rec.notes.push(("x_axis_after".into(), format!("({},{})", fmt_p(x1.placement().x_axis().x), fmt_p(x1.placement().x_axis().y))));
                    rec.notes.push(("radius_after".into(), fmt_p(x1.radius())));
                }
                (Curve::Ellipse(x0), Curve::Ellipse(x1)) => {
                    rec.notes.push(("semi_axes_before".into(), format!("({},{})", fmt_p(x0.semi_axis_1()), fmt_p(x0.semi_axis_2()))));
                    rec.notes.push(("semi_axes_after".into(), format!("({},{})", fmt_p(x1.semi_axis_1()), fmt_p(x1.semi_axis_2()))));
                }
                _ => {}
            }
        }
        (Curve::Circle(_), Curve::Ellipse(e)) | (Curve::Ellipse(_), Curve::Ellipse(e)) => {
            rec.notes.push(("semi_axes_after".into(), format!("({},{})", fmt_p(e.semi_axis_1()), fmt_p(e.semi_axis_2()))));
            rec.notes.push(("x_axis_after".into(), format!("({},{})", fmt_p(e.placement().x_axis().x), fmt_p(e.placement().x_axis().y))));
        }
        (Curve::Circle(_), Curve::Circle(c)) => {
            rec.notes.push(("radius_after".into(), fmt_p(c.radius())));
        }
        (Curve::Ellipse(_), Curve::Circle(c)) => {
            rec.notes.push(("radius_after".into(), fmt_p(c.radius())));
        }
        _ => {}
    }
    rec
}

#[allow(clippy::too_many_arguments)]
fn spline_notes(
    rec: &mut Rec,
    d0: usize,
    c0: &[Point2],
    k0: &[f64],
    w0: Option<&[f64]>,
    d1: usize,
    c1: &[Point2],
    k1: &[f64],
    w1: Option<&[f64]>,
    m: &Transform2,
) {
    let cp_dev = c0.iter().zip(c1).map(|(a, b)| dist(m.apply_point(*a), *b)).fold(0.0f64, f64::max);
    rec.notes.push(("degree_identical".into(), (d0 == d1).to_string()));
    rec.notes.push(("knots_identical".into(), (k0 == k1).to_string()));
    rec.notes.push(("control_points_max_dev".into(), fmt_e(cp_dev)));
    if let (Some(a), Some(b)) = (w0, w1) {
        rec.notes.push(("weights_identical".into(), (a == b).to_string()));
    }
}

fn main() -> Res<()> {
    println!("id\tfixture\ttransform\tclass\toutcome\tresult\trule\terror\tworst_dev\ttol\tratio\tsamples\tnotes");
    let fx = fixtures()?;
    let xs = transforms();
    let mut n = 0usize;
    for x in &xs {
        for f in &fx {
            n += 1;
            let id = format!("E{n}");
            match (x.build)() {
                Ok(m) => emit(&measure(f, x, id, m)),
                Err(e) => {
                    // A map that cannot be built never reaches a curve (P3.8), so no curve is emitted.
                    let mut notes = vec![("curve_emitted".to_string(), "false".to_string())];
                    if !x.band.is_empty() {
                        notes.push(("band".into(), x.band.into()));
                        notes.push(("derived_from".into(), "Policy::V1.rel".into()));
                    }
                    emit(&Rec {
                        id,
                        fixture: f.id,
                        transform: x.id,
                        class: x.class,
                        outcome: "refused",
                        result: "-".into(),
                        rule: e.rule().to_string(),
                        error: format!("{e:?}"),
                        worst: "-".into(),
                        tol: "-".into(),
                        ratio: "-".into(),
                        notes,
                    });
                }
            }
        }
    }
    // The exact rational form on request, for the conic rows (SupportedEquivalent entry point).
    let shear = Transform2::shear(0.7, 0.0)?;
    for f in fx.iter().filter(|f| matches!(f.row, "circle" | "ellipse" | "circular arc" | "elliptical arc")) {
        for (label, curve) in [("original", f.curve.clone()), ("after T8", f.curve.transform(&shear)?.0)] {
            n += 1;
            let mut notes = vec![("on".to_string(), label.to_string()), ("entry_point".into(), "Curve::to_rational_b_spline".into())];
            let (result, outcome, rule, error) = match curve.to_rational_b_spline() {
                Ok(r) => {
                    if let Curve::RationalBSpline(rb) = &r {
                        notes.push(("control_points".into(), rb.control_points().len().to_string()));
                    }
                    (kind_name(&r), "ok", "-".to_string(), "-".to_string())
                }
                Err(e) => ("-".to_string(), "refused", e.rule().to_string(), format!("{e:?}")),
            };
            emit(&Rec {
                id: format!("E{n}"),
                fixture: f.id,
                transform: "rational-form",
                class: "to_rational_b_spline",
                outcome,
                result,
                rule,
                error,
                worst: "-".into(),
                tol: "-".into(),
                ratio: "-".into(),
                notes,
            });
        }
    }
    Ok(())
}
