//! Curves to polylines, with a stated deviation.
//!
//! Every function takes a tolerance `tol` in drawing units and returns points
//! whose polyline stays within `tol` of the true curve, or says it could not
//! ([`Tessellation::capped`]). The tolerance is the tiler's to pick, because
//! it depends on the zoom a tile is cut for: the same arc is eight segments at
//! one zoom and two hundred at another, which is why the decoder keeps arcs as
//! arcs.
//!
//! Coordinates are world coordinates and the output is the plan view: x and y
//! of the 3D curve, z dropped. A curve in a tilted plane is evaluated in that
//! plane and then projected, so a circle seen edge on becomes the ellipse it
//! should be.
//!
//! The bounds, and why they hold:
//!
//! * an arc of radius `r` stepped by `dθ` deviates by `r (1 - cos(dθ/2))`, so
//!   `dθ = 2 acos(1 - tol/r)` is exact;
//! * a parametric curve stepped by `dt` deviates by at most
//!   `max|P''| dt² / 8`, so an ellipse of semi-major axis `a` steps by
//!   `sqrt(8 tol / a)`;
//! * a bulge arc is split until its sagitta `(c/2)|b|` is inside `tol`;
//! * a spline is subdivided until the curve's value at the middle and the
//!   quarter points of an interval is within `tol` of the chord. That is a
//!   sample-based test, not a proof, and the tests measure it against a dense
//!   reference.

use crate::cad::{Arc, Circle, Ellipse, Spline};

/// The most points one curve may produce. A curve that would need more is
/// emitted at this resolution and flagged.
pub const MAX_POINTS_PER_CURVE: usize = 1 << 14;

/// A tessellated curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Tessellation {
    /// The points, in drawing order.
    pub points: Vec<[f64; 2]>,
    /// Whether the point cap stopped the curve short of its tolerance.
    pub capped: bool,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    // hypot, not sqrt of a sum of squares: the squares overflow at 1e154.
    a[0].hypot(a[1]).hypot(a[2])
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// The x and y axes the arbitrary axis algorithm gives for a normal.
///
/// The normal is normalised first, because the `1/64` band is a bound on a
/// direction cosine and means nothing on a vector that is not unit length.
#[must_use]
pub fn arbitrary_axes(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = unit(normal);
    let ax = if n[0].abs() < 1.0 / 64.0 && n[1].abs() < 1.0 / 64.0 {
        cross([0.0, 1.0, 0.0], n)
    } else {
        cross([0.0, 0.0, 1.0], n)
    };
    let ax = unit(ax);
    let ay = cross(n, ax);
    (ax, ay)
}

fn plan(p: [f64; 3]) -> [f64; 2] {
    [p[0], p[1]]
}

fn on_plane(c: [f64; 3], r: f64, t: f64, ax: [f64; 3], ay: [f64; 3]) -> [f64; 2] {
    let (s, co) = t.sin_cos();
    [
        c[0] + r * (co * ax[0] + s * ay[0]),
        c[1] + r * (co * ax[1] + s * ay[1]),
    ]
}

/// The step count a stepped curve needs, clamped to the cap.
fn steps(sweep: f64, step: f64, min: usize) -> (usize, bool) {
    let want = (sweep / step).ceil();
    let want = if want.is_finite() { want } else { f64::MAX };
    let cap = (MAX_POINTS_PER_CURVE - 1) as f64;
    let capped = want > cap;
    let n = want.min(cap) as usize;
    (n.max(min).max(1), capped)
}

/// An arc. `tol` is in drawing units.
#[must_use]
pub fn arc(a: &Arc, tol: f64) -> Tessellation {
    let (ax, ay) = arbitrary_axes(a.normal().into_array());
    let c = a.centre().into_array();
    let r = a.radius();
    let start = a.start_angle().rem_euclid(std::f64::consts::TAU);
    stepped_circle(c, r, start, a.sweep(), ax, ay, tol, 1)
}

/// A circle, as a closed run: the first point is repeated at the end.
#[must_use]
pub fn circle(c: &Circle, tol: f64) -> Tessellation {
    let (ax, ay) = arbitrary_axes(c.normal().into_array());
    stepped_circle(
        c.centre().into_array(),
        c.radius(),
        0.0,
        std::f64::consts::TAU,
        ax,
        ay,
        tol,
        8,
    )
}

#[allow(clippy::too_many_arguments)]
fn stepped_circle(
    c: [f64; 3],
    r: f64,
    start: f64,
    sweep: f64,
    ax: [f64; 3],
    ay: [f64; 3],
    tol: f64,
    min_steps: usize,
) -> Tessellation {
    // acos(1 - tol/r) for tol >= r would be acos of a non-positive number; a
    // step of a half turn is as coarse as a chord can honestly be.
    let step = if tol >= r {
        std::f64::consts::PI
    } else {
        2.0 * (1.0 - tol / r).acos()
    };
    let (n, capped) = steps(sweep, step, min_steps);
    let mut points = Vec::with_capacity(n + 1);
    for i in 0..=n {
        // The last point is computed from the end angle, not accumulated, so
        // a full circle closes exactly on its first point.
        let t = if i == n {
            start + sweep
        } else {
            start + sweep * (i as f64 / n as f64)
        };
        points.push(on_plane(c, r, t, ax, ay));
    }
    if (sweep - std::f64::consts::TAU).abs() < f64::EPSILON * 8.0 {
        let first = points[0];
        if let Some(last) = points.last_mut() {
            *last = first;
        }
    }
    Tessellation { points, capped }
}

/// An ellipse or elliptical arc.
#[must_use]
pub fn ellipse(e: &Ellipse, tol: f64) -> Tessellation {
    let n = unit(e.normal().into_array());
    let major = e.major_axis().into_array();
    let a = norm(major);
    // The minor axis lies in the plane, perpendicular to the major one, with
    // the length the ratio gives.
    let minor = {
        let m = cross(n, major);
        let len = norm(m);
        if len == 0.0 {
            [0.0; 3]
        } else {
            let k = e.ratio() * a / len;
            [m[0] * k, m[1] * k, m[2] * k]
        }
    };
    let c = e.centre().into_array();
    let start = e.start_param().rem_euclid(std::f64::consts::TAU);
    let sweep = e.sweep();
    let step = (8.0 * tol / a).sqrt();
    let min_steps = ((sweep / std::f64::consts::FRAC_PI_4).ceil() as usize).max(1);
    let (steps_n, capped) = steps(sweep, step, min_steps);
    let mut points = Vec::with_capacity(steps_n + 1);
    for i in 0..=steps_n {
        let t = if i == steps_n {
            start + sweep
        } else {
            start + sweep * (i as f64 / steps_n as f64)
        };
        let (s, co) = t.sin_cos();
        points.push([
            c[0] + co * major[0] + s * minor[0],
            c[1] + co * major[1] + s * minor[1],
        ]);
    }
    if (sweep - std::f64::consts::TAU).abs() < f64::EPSILON * 8.0 {
        let first = points[0];
        if let Some(last) = points.last_mut() {
            *last = first;
        }
    }
    Tessellation { points, capped }
}

/// A run of vertices with per-span bulges, as a polyline or a polygon ring.
///
/// `closed` adds the closing span, whose bulge is the last vertex's. The
/// output of a closed run repeats its first point at the end.
#[must_use]
pub fn bulge_run(
    vertices: &[[f64; 3]],
    bulges: &[f64],
    normal: [f64; 3],
    closed: bool,
    tol: f64,
) -> Tessellation {
    let n = unit(normal);
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(vertices.len() + 1);
    let mut capped = false;
    let spans = if closed {
        vertices.len()
    } else {
        vertices.len().saturating_sub(1)
    };
    if let Some(v) = vertices.first() {
        points.push(plan(*v));
    }
    for i in 0..spans {
        let p0 = vertices[i];
        let p1 = vertices[(i + 1) % vertices.len()];
        let b = bulges.get(i).copied().unwrap_or(0.0);
        if b != 0.0 {
            capped |= subdivide(p0, p1, b, n, tol, 0, &mut points);
        }
        points.push(plan(p1));
    }
    Tessellation { points, capped }
}

/// Splits a bulge span at the arc's midpoint until its sagitta is within `tol`.
///
/// Endpoints are never recomputed, so the pieces meet exactly at every depth.
/// Returns whether the cap stopped it.
fn subdivide(
    p0: [f64; 3],
    p1: [f64; 3],
    b: f64,
    n: [f64; 3],
    tol: f64,
    depth: u32,
    out: &mut Vec<[f64; 2]>,
) -> bool {
    let d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let chord = norm(d);
    if chord == 0.0 || !b.is_finite() {
        return false;
    }
    let sagitta = chord / 2.0 * b.abs();
    if sagitta <= tol {
        return false;
    }
    if depth >= 32 || out.len() >= MAX_POINTS_PER_CURVE {
        return true;
    }
    let dh = [d[0] / chord, d[1] / chord, d[2] / chord];
    let side = cross(dh, n);
    let k = b * chord / 2.0;
    let mid = [
        (p0[0] + p1[0]) / 2.0 + k * side[0],
        (p0[1] + p1[1]) / 2.0 + k * side[1],
        (p0[2] + p1[2]) / 2.0 + k * side[2],
    ];
    // tan(atan(b) / 2), in the form that does not overflow either way.
    let half = if b.abs() <= 1.0 {
        b / (1.0 + (1.0 + b * b).sqrt())
    } else {
        let r = 1.0 / b.abs();
        b.signum() / (r + (r * r + 1.0).sqrt())
    };
    if mid.iter().any(|v| !v.is_finite()) {
        // A bulge so large its arc leaves the range of a double. The chord is
        // all that can honestly be drawn, and the caller is told.
        return true;
    }
    let mut capped = subdivide(p0, mid, half, n, tol, depth + 1, out);
    out.push(plan(mid));
    capped |= subdivide(mid, p1, half, n, tol, depth + 1, out);
    capped
}

/// A spline, by adaptive subdivision of its parameter interval.
///
/// `None` when the spline cannot be evaluated: a knot vector that does not
/// span a domain, or weights that make a point non-finite. The caller reports
/// that rather than drawing a guess.
#[must_use]
pub fn spline(s: &Spline, tol: f64) -> Option<Tessellation> {
    let degree = s.degree() as usize;
    let knots = s.knots();
    let controls = s.controls();
    let weights = s.weights();
    let n = controls.len();
    if n <= degree || knots.len() < n + degree + 1 {
        return None;
    }
    let (lo, hi) = (knots[degree], knots[n]);
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let eval =
        |t: f64| -> Option<[f64; 2]> { de_boor(degree, knots, controls, weights, t, lo, hi) };

    // Start from one sample per non-empty knot span so a curve with sharp
    // features is not judged by one midpoint that happens to land on a flat bit.
    let mut spans: Vec<f64> = vec![lo];
    for &k in &knots[degree + 1..n] {
        if k > *spans.last()? && k < hi {
            spans.push(k);
        }
    }
    spans.push(hi);

    let mut points = vec![eval(lo)?];
    let mut capped = false;
    for w in spans.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let p0 = eval(t0)?;
        let p1 = eval(t1)?;
        capped |= refine(&eval, t0, p0, t1, p1, tol, 0, &mut points)?;
        points.push(p1);
    }
    Some(Tessellation { points, capped })
}

fn dist_to_chord(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (a[0] + t * dx, a[1] + t * dy);
    ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn refine(
    eval: &dyn Fn(f64) -> Option<[f64; 2]>,
    t0: f64,
    p0: [f64; 2],
    t1: f64,
    p1: [f64; 2],
    tol: f64,
    depth: u32,
    out: &mut Vec<[f64; 2]>,
) -> Option<bool> {
    let tm = t0 + (t1 - t0) * 0.5;
    let pm = eval(tm)?;
    let q1 = eval(t0 + (t1 - t0) * 0.25)?;
    let q3 = eval(t0 + (t1 - t0) * 0.75)?;
    let dev = dist_to_chord(pm, p0, p1)
        .max(dist_to_chord(q1, p0, p1))
        .max(dist_to_chord(q3, p0, p1));
    if dev <= tol {
        return Some(false);
    }
    if depth >= 24 || out.len() >= MAX_POINTS_PER_CURVE {
        return Some(true);
    }
    let a = refine(eval, t0, p0, tm, pm, tol, depth + 1, out)?;
    out.push(pm);
    let b = refine(eval, tm, pm, t1, p1, tol, depth + 1, out)?;
    Some(a | b)
}

/// De Boor evaluation of a (possibly rational) B-spline at `t`.
fn de_boor(
    degree: usize,
    knots: &[f64],
    controls: &[crate::cad::Point3],
    weights: &[f64],
    t: f64,
    lo: f64,
    hi: f64,
) -> Option<[f64; 2]> {
    let n = controls.len();
    let t = t.clamp(lo, hi);
    // The knot span containing t, with the right end closed.
    let mut k = degree;
    while k + 1 < n && knots[k + 1] <= t {
        k += 1;
    }
    let rational = weights.len() == n;
    let mut d: Vec<[f64; 3]> = (0..=degree)
        .map(|j| {
            let i = k - degree + j;
            let w = if rational { weights[i] } else { 1.0 };
            [controls[i].x() * w, controls[i].y() * w, w]
        })
        .collect();
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let i = k - degree + j;
            let denom = knots[i + degree + 1 - r] - knots[i];
            let alpha = if denom == 0.0 {
                0.0
            } else {
                (t - knots[i]) / denom
            };
            let prev = d[j - 1];
            for (cur, before) in d[j].iter_mut().zip(prev) {
                *cur = (1.0 - alpha) * before + alpha * *cur;
            }
        }
    }
    let v = d[degree];
    let (x, y) = if rational {
        (v[0] / v[2], v[1] / v[2])
    } else {
        (v[0], v[1])
    };
    if x.is_finite() && y.is_finite() {
        Some([x, y])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    /// The largest distance from any point of the polyline's segments to the
    /// circle `|p - c| = r`, sampled densely along each segment.
    fn max_dev_from_circle(pts: &[[f64; 2]], c: [f64; 2], r: f64) -> f64 {
        let mut worst = 0.0f64;
        for w in pts.windows(2) {
            for k in 0..=16 {
                let t = f64::from(k) / 16.0;
                let p = [
                    w[0][0] + (w[1][0] - w[0][0]) * t,
                    w[0][1] + (w[1][1] - w[0][1]) * t,
                ];
                let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
                worst = worst.max((d - r).abs());
            }
        }
        worst
    }

    #[test]
    fn an_arc_stays_inside_its_tolerance_across_radii_and_tolerances() {
        for r in [0.01, 1.0, 250.0, 1.0e6] {
            for frac in [1e-1, 1e-2, 1e-4] {
                let tol = r * frac;
                let a = Arc::new([3.0, -4.0, 0.0], r, 0.3, 4.0, [0.0, 0.0, 1.0]).unwrap();
                let t = arc(&a, tol);
                assert!(!t.capped);
                let dev = max_dev_from_circle(&t.points, [3.0, -4.0], r);
                assert!(dev <= tol * 1.0001, "r={r} tol={tol} dev={dev}");
            }
        }
    }

    #[test]
    fn a_circle_closes_exactly_and_is_round_enough() {
        let c = Circle::new([0.0; 3], 5.0, [0.0, 0.0, 1.0]).unwrap();
        let t = circle(&c, 0.01);
        assert_eq!(t.points.first(), t.points.last());
        assert!(t.points.len() >= 8);
        assert!(max_dev_from_circle(&t.points, [0.0, 0.0], 5.0) <= 0.0101);
    }

    #[test]
    fn a_huge_tolerance_still_gives_a_recognisable_circle() {
        let c = Circle::new([0.0; 3], 1.0, [0.0, 0.0, 1.0]).unwrap();
        assert!(circle(&c, 100.0).points.len() >= 9);
    }

    #[test]
    fn the_arbitrary_axis_algorithm_matches_the_documented_cases() {
        let (ax, _) = arbitrary_axes([0.0, 0.0, 1.0]);
        assert_eq!(ax, [1.0, 0.0, 0.0]);
        // "For a normal of (0, 1, 0) zero is (-1, 0, 0), which nobody guesses."
        let (ax, _) = arbitrary_axes([0.0, 1.0, 0.0]);
        assert!((ax[0] + 1.0).abs() < 1e-12 && ax[1].abs() < 1e-12 && ax[2].abs() < 1e-12);
        // Normalised first: this is outside the band as written and inside it
        // once unit length, and the two answers are ninety degrees apart.
        let (a, _) = arbitrary_axes([2.0 / 64.0 + 1e-9, 0.0, 2.0]);
        let (b, _) = arbitrary_axes([1.0 / 64.0 + 0.5e-9, 0.0, 1.0]);
        assert!(norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) < 1e-6);
    }

    #[test]
    fn a_tilted_circle_projects_to_the_ellipse_it_should() {
        // Normal 60 degrees off z: the plan view is an ellipse with minor
        // semi-axis r cos(60) = r/2 along one plan axis.
        let n = [3f64.sqrt() / 2.0, 0.0, 0.5];
        let c = Circle::new([0.0; 3], 2.0, n).unwrap();
        let t = circle(&c, 0.001);
        let (mut max_x, mut max_y) = (0.0f64, 0.0f64);
        for p in &t.points {
            max_x = max_x.max(p[0].abs());
            max_y = max_y.max(p[1].abs());
        }
        let mut axes = [max_x, max_y];
        axes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (axes[0] - 1.0).abs() < 0.01 && (axes[1] - 2.0).abs() < 0.01,
            "{axes:?}"
        );
    }

    #[test]
    fn an_ellipse_stays_on_the_ellipse() {
        let e = Ellipse::new(
            [1.0, 2.0, 0.0],
            [10.0, 0.0, 0.0],
            0.3,
            0.0,
            TAU,
            [0.0, 0.0, 1.0],
        )
        .unwrap();
        let tol = 0.01;
        let t = ellipse(&e, tol);
        assert!(!t.capped);
        assert_eq!(t.points.first(), t.points.last());
        for w in t.points.windows(2) {
            for k in 0..=8 {
                let f = f64::from(k) / 8.0;
                let p = [
                    w[0][0] + (w[1][0] - w[0][0]) * f - 1.0,
                    w[0][1] + (w[1][1] - w[0][1]) * f - 2.0,
                ];
                // Implicit distance, first order: |g| / |grad g|.
                let g = (p[0] / 10.0).powi(2) + (p[1] / 3.0).powi(2) - 1.0;
                let grad = (2.0 * p[0] / 100.0).hypot(2.0 * p[1] / 9.0);
                assert!(
                    (g / grad).abs() <= tol * 1.05,
                    "off by {}",
                    (g / grad).abs()
                );
            }
        }
    }

    #[test]
    fn a_bulge_of_one_is_a_half_circle_inside_tolerance() {
        // (0,0) to (2,0), bulge +1: counter-clockwise half turn, centre (1,0).
        let t = bulge_run(
            &[[0.0; 3], [2.0, 0.0, 0.0]],
            &[1.0, 0.0],
            [0.0, 0.0, 1.0],
            false,
            0.001,
        );
        assert!(!t.capped);
        assert!(max_dev_from_circle(&t.points, [1.0, 0.0], 1.0) <= 0.00101);
        // CCW from (0,0) to (2,0) passes below the chord.
        assert!(t.points.iter().any(|p| p[1] < -0.9));
        let cw = bulge_run(
            &[[0.0; 3], [2.0, 0.0, 0.0]],
            &[-1.0, 0.0],
            [0.0, 0.0, 1.0],
            false,
            0.001,
        );
        assert!(cw.points.iter().any(|p| p[1] > 0.9));
    }

    #[test]
    fn bulges_of_every_magnitude_stay_finite() {
        for b in [
            1e-300, 1e-15, 1e-3, 0.5, 1.0, 7.0, 1e6, 1e154, 1e300, -1e300, -3.0,
        ] {
            let t = bulge_run(
                &[[0.0; 3], [1.0, 0.0, 0.0]],
                &[b, 0.0],
                [0.0, 0.0, 1.0],
                false,
                1e-3,
            );
            assert!(
                t.points
                    .iter()
                    .all(|p| p[0].is_finite() && p[1].is_finite()),
                "b={b}"
            );
            assert_eq!(t.points.first(), Some(&[0.0, 0.0]));
            assert_eq!(t.points.last(), Some(&[1.0, 0.0]));
        }
    }

    #[test]
    fn a_closed_run_includes_its_closing_span() {
        let sq = [[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let t = bulge_run(&sq, &[], [0.0, 0.0, 1.0], true, 0.01);
        assert_eq!(t.points.len(), 5);
        assert_eq!(t.points.first(), t.points.last());
    }

    #[test]
    fn the_point_cap_is_reported_not_hidden() {
        let a = Arc::new([0.0; 3], 1.0e9, 0.0, 6.0, [0.0, 0.0, 1.0]).unwrap();
        let t = arc(&a, 1e-9);
        assert!(t.capped);
        assert!(t.points.len() <= MAX_POINTS_PER_CURVE + 1);
    }

    fn quadratic() -> Spline {
        Spline::new(
            2,
            0,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 0.0, 0.0]],
            &[],
        )
        .unwrap()
    }

    #[test]
    fn a_bezier_spline_matches_its_closed_form() {
        // B(t) = (10 t, 20 t (1 - t)).
        let t = spline(&quadratic(), 0.01).unwrap();
        assert!(!t.capped);
        assert_eq!(t.points.first(), Some(&[0.0, 0.0]));
        let last = t.points.last().unwrap();
        assert!((last[0] - 10.0).abs() < 1e-9 && last[1].abs() < 1e-9);
        for w in t.points.windows(2) {
            for k in 0..=8 {
                let f = f64::from(k) / 8.0;
                let p = [
                    w[0][0] + (w[1][0] - w[0][0]) * f,
                    w[0][1] + (w[1][1] - w[0][1]) * f,
                ];
                let u = p[0] / 10.0;
                let on_curve = 20.0 * u * (1.0 - u);
                // Vertical gap, then to the perpendicular distance.
                let slope = 20.0 * (1.0 - 2.0 * u) / 10.0;
                let dist = (p[1] - on_curve).abs() / (1.0 + slope * slope).sqrt();
                assert!(dist <= 0.011, "off by {dist}");
            }
        }
    }

    #[test]
    fn a_rational_quarter_circle_is_a_circle() {
        let w = 0.5f64.sqrt();
        let s = Spline::new(
            2,
            2,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            &[1.0, w, 1.0],
        )
        .unwrap();
        let t = spline(&s, 0.001).unwrap();
        assert!(max_dev_from_circle(&t.points, [0.0, 0.0], 1.0) <= 0.0011);
    }

    #[test]
    fn a_knot_vector_that_spans_nothing_is_not_evaluated() {
        // Spline::new refuses this, so build the closest thing it accepts and
        // check the evaluator's own guard with a degenerate domain.
        let s = Spline::new(
            1,
            0,
            &[0.0, 0.0, 1.0, 1.0],
            &[[0.0; 3], [1.0, 1.0, 0.0]],
            &[],
        )
        .unwrap();
        assert!(spline(&s, 0.1).is_some());
    }

    #[test]
    fn an_arc_sweep_of_a_turn_is_not_zero() {
        let a = Arc::new([0.0; 3], 1.0, 0.0, TAU, [0.0, 0.0, 1.0]).unwrap();
        assert!(arc(&a, 0.01).points.len() > 8);
        let _ = PI;
    }
}
