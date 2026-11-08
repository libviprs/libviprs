//! Curves cut exactly against a viewport clip (issue #1567, the geometry half of slice 6, #1565).
//!
//! The print profile draws nothing outside a viewport (P-7 of `docs/adr/pvec-print-profile.md`),
//! and PVec has no clip region, so the producer cuts. The design comment on #1557 settled how: a
//! curve that crosses the clip is split, never tessellated, and the pieces inside are kept.
//!
//! | source | pieces | class |
//! |---|---|---|
//! | circle, ellipse, arc (a trimmed conic) | trimmed arcs of the source's own carrier over the kept parameter intervals | `SupportedExact` |
//! | line, trimmed line, polyline | trimmed lines of the same carrier, polylines of the kept runs of the source's vertices | `SupportedExact` |
//! | B-spline, Bezier | sub-ranges of the source (`BSpline::sub_range`, PVec's knot insertion), the kind kept | `SupportedEquivalent` |
//! | trimmed spline | the same spline under a narrower trim | `SupportedEquivalent` |
//! | composite | each segment cut as above, the runs that stay joined kept as composites | the worst of its segments |
//! | rational B-spline | refused, [`CurveClipError::Unsupported`]: PVec has no sub-range for it yet | |
//!
//! # How
//!
//! Every crossing of the curve with an edge of the clip is found as a parameter of the curve. The
//! parameters cut the domain into intervals, and each interval is kept when the point PVec
//! evaluates at its middle is inside the clip by the even-odd rule. Kept neighbours are joined,
//! and for a whole circle or ellipse the interval through the seam of the parameter is one arc.
//! So a rectangle is just a four-vertex polygon here, and a polygon clip is cut edge by edge.
//!
//! A conic crossing is closed form: the edge is mapped into the frame where the conic is the unit
//! circle, one `sqrt` solves the quadratic, and the angle comes from `detmath::atan2`, so the bits
//! are the same on every machine and the angle is within the one ulp PVec states for it. A line or
//! polyline crossing is two cross products. A spline crossing has no closed form: each knot span is
//! a Bezier piece (PVec's `sub_range`), a piece whose control points all sit on one side of the
//! edge's line has no crossing (the convex hull), a piece whose control values change sign once has
//! exactly one and is bisected on PVec's evaluator, and anything else is halved again through
//! `sub_range`. The bisection stops at `Policy::V1`'s tolerance for the domain, which is reported
//! as the parameter tolerance with `SupportedEquivalent`.
//!
//! Nothing here evaluates, inserts knots or tessellates (NG-10 of
//! `docs/adr/pvec-normalized-geometry.md`): every point comes from `Curve2d::eval`, every spline
//! piece from `BSpline::sub_range`, and every tolerance from `Policy::V1`. A spurious crossing
//! parameter costs nothing, because neighbours on the same side are joined again, so an edge end
//! is matched with the policy's slack rather than missed.

use core::f64::consts::TAU;
use std::fmt;

use pvec_geometry::{
    BSpline, Bezier, Composite, Curve, Curve2d, CurveKind, GeometryError, Placement2, Point2,
    Polyline, Trimmed, Vector2,
};
use pvec_numeric::{NumericError, Policy, detmath};

use super::fidelity::FidelityClass;

const POLICY: Policy = Policy::V1;

/// What became of one curve against one clip.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveClip {
    /// Wholly inside: keep the source as it is.
    Inside,
    /// Wholly outside: nothing of it is drawn (`OutsideViewport`).
    Outside,
    /// It crosses the clip: the pieces inside, in the order the source runs through them.
    Split {
        /// The kept pieces.
        pieces: Vec<Curve>,
        /// `SupportedExact` for conics, lines and polylines, `SupportedEquivalent` once a spline
        /// is involved.
        fidelity: FidelityClass,
        /// How far, in the source parameter, a spline crossing may sit from the true one. Zero
        /// when every crossing is closed form.
        parameter_tolerance: f64,
    },
}

/// Why a curve could not be cut.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CurveClipError {
    /// The clip has fewer than three vertices.
    Ring,
    /// A curve kind this module cannot cut exactly yet.
    Unsupported(CurveKind),
    /// PVec refused a piece.
    Geometry(GeometryError),
}

impl From<GeometryError> for CurveClipError {
    fn from(e: GeometryError) -> Self {
        CurveClipError::Geometry(e)
    }
}

impl fmt::Display for CurveClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CurveClipError::Ring => write!(f, "a clip needs at least three vertices"),
            CurveClipError::Unsupported(k) => write!(f, "cannot cut a {k:?} exactly"),
            CurveClipError::Geometry(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CurveClipError {}

/// Cuts `curve` against the polygon `ring` (its vertices in order, closed implicitly, inside by the
/// even-odd rule). A rectangle is the four corners.
pub fn clip_curve(curve: &Curve, ring: &[Point2]) -> Result<CurveClip, CurveClipError> {
    if ring.len() < 3 {
        return Err(CurveClipError::Ring);
    }
    let Curve::Composite(c) = curve else {
        let cut = cut(curve, ring)?;
        return Ok(if cut.intervals.is_empty() {
            CurveClip::Outside
        } else if cut.intervals == [cut.domain] {
            CurveClip::Inside
        } else {
            CurveClip::Split {
                pieces: cut.pieces,
                fidelity: fidelity(cut.tolerance),
                parameter_tolerance: cut.tolerance.unwrap_or(0.0),
            }
        });
    };
    // A composite: each segment on its own, and the pieces that still join end to start (a
    // segment kept to its end followed by one kept from its start) stay one composite.
    let mut runs: Vec<Vec<Curve>> = Vec::new();
    let mut run: Vec<Curve> = Vec::new();
    let mut whole = true;
    let mut tolerance: Option<f64> = None;
    for seg in c.segments() {
        let cut = cut(seg, ring)?;
        if let Some(t) = cut.tolerance {
            tolerance = Some(tolerance.map_or(t, |m| m.max(t)));
        }
        if cut.intervals == [cut.domain] {
            run.push(seg.clone());
            continue;
        }
        whole = false;
        if cut.intervals.is_empty() {
            runs.extend((!run.is_empty()).then(|| std::mem::take(&mut run)));
        }
        for (iv, piece) in cut.intervals.iter().zip(cut.pieces) {
            if iv[0] != cut.domain[0] && !run.is_empty() {
                runs.push(std::mem::take(&mut run));
            }
            run.push(piece);
            if iv[1] != cut.domain[1] {
                runs.push(std::mem::take(&mut run));
            }
        }
    }
    if whole {
        return Ok(CurveClip::Inside);
    }
    runs.extend((!run.is_empty()).then_some(run));
    if runs.is_empty() {
        return Ok(CurveClip::Outside);
    }
    let pieces = runs
        .into_iter()
        .map(|mut r| {
            if r.len() == 1 {
                Ok(r.remove(0))
            } else {
                Composite::new(r).map(Curve::Composite)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CurveClip::Split {
        pieces,
        fidelity: fidelity(tolerance),
        parameter_tolerance: tolerance.unwrap_or(0.0),
    })
}

fn fidelity(tolerance: Option<f64>) -> FidelityClass {
    if tolerance.is_some() {
        FidelityClass::SupportedEquivalent
    } else {
        FidelityClass::SupportedExact
    }
}

/// One non-composite curve cut against the clip: the kept intervals of its own parameter, the
/// pieces built over them (one each, in order), its domain, and the parameter tolerance when a
/// crossing was found by bisection.
struct Cut {
    intervals: Vec<[f64; 2]>,
    pieces: Vec<Curve>,
    domain: [f64; 2],
    tolerance: Option<f64>,
}

fn cut(curve: &Curve, ring: &[Point2]) -> Result<Cut, CurveClipError> {
    let d = curve.domain();
    let domain = [d.lo(), d.hi()];
    let (params, tolerance) = crossings(curve, ring)?;
    let periodic = matches!(curve, Curve::Circle(_) | Curve::Ellipse(_));

    // The boundaries: the domain ends and every crossing strictly inside it, sorted, with any two
    // the policy cannot tell apart taken as one. A bare line's domain is the coordinate bound, so
    // its parameters are told apart at the scale of the crossings instead.
    let scale = if matches!(curve, Curve::Line(_)) {
        let lo = params.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = params.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if hi > lo { hi - lo } else { 1.0 }
    } else {
        domain[1] - domain[0]
    };
    let mut sorted = params;
    sorted.sort_by(f64::total_cmp);
    let mut bounds = vec![domain[0]];
    for t in sorted {
        let last = bounds[bounds.len() - 1];
        let near = |a: f64, b: f64| {
            (a - b).abs()
                <= POLICY
                    .tol(scale, a.abs().max(b.abs()))
                    .max(tolerance.unwrap_or(0.0))
        };
        if t > domain[0] && t < domain[1] && !near(t, last) && !near(t, domain[1]) {
            bounds.push(t);
        }
    }
    bounds.push(domain[1]);

    // Keep an interval when its middle is inside by the even-odd rule (a ray to the right crosses
    // the boundary an odd number of times); join kept neighbours.
    let inside = |q: Point2| {
        edges(ring).fold(false, |odd, (a, b)| {
            let crosses =
                (a.y > q.y) != (b.y > q.y) && q.x < a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y);
            odd != crosses
        })
    };
    let mut kept: Vec<[f64; 2]> = Vec::new();
    for w in bounds.windows(2) {
        let mid = curve.eval(w[0] + 0.5 * (w[1] - w[0]))?;
        if !inside(mid) {
            continue;
        }
        match kept.last_mut() {
            Some(k) if k[1] == w[0] => k[1] = w[1],
            _ => kept.push([w[0], w[1]]),
        }
    }
    // A whole conic has no ends: the interval through the seam is one arc, and it comes first,
    // since the source starts inside it.
    let mut wrap: Option<[f64; 2]> = None;
    if periodic && kept.len() > 1 && kept[0][0] == domain[0] && kept[kept.len() - 1][1] == domain[1]
    {
        let first = kept.remove(0);
        let last = kept.pop().expect("more than one kept interval");
        wrap = Some([last[0], first[1] + TAU]);
    }

    /// The piece of `curve` over `[a, b]` of its own parameter.
    fn piece(curve: &Curve, [a, b]: [f64; 2]) -> Result<Curve, CurveClipError> {
        Ok(match curve {
            Curve::Circle(_) | Curve::Ellipse(_) | Curve::Line(_) => {
                Curve::Trimmed(Trimmed::new(curve.clone(), a, b, true)?)
            }
            Curve::Trimmed(t) => {
                let trim = t.trim();
                let (s1, s2) = if trim.reversed {
                    (trim.start + trim.span - a, trim.start + trim.span - b)
                } else {
                    (trim.start + a, trim.start + b)
                };
                Curve::Trimmed(Trimmed::new(t.basis().clone(), s1, s2, !trim.reversed)?)
            }
            Curve::Polyline(pl) => {
                let v = pl.vertices();
                let mut out = vec![curve.eval(a)?];
                out.extend(
                    v.iter()
                        .enumerate()
                        .filter(|(i, _)| (*i as f64) > a && (*i as f64) < b)
                        .map(|(_, q)| *q),
                );
                out.push(curve.eval(b)?);
                Curve::Polyline(Polyline::new(out)?)
            }
            Curve::BSpline(s) => Curve::BSpline(s.sub_range(a, b)?),
            Curve::Bezier(z) => Curve::Bezier(Bezier::new(
                z.as_bspline()?.sub_range(a, b)?.control_points().to_vec(),
            )?),
            other => return Err(CurveClipError::Unsupported(other.kind())),
        })
    }

    let mut intervals = Vec::with_capacity(kept.len() + 1);
    let mut pieces = Vec::with_capacity(kept.len() + 1);
    for iv in wrap.into_iter().chain(kept) {
        match piece(curve, iv) {
            Ok(p) => {
                intervals.push(iv);
                pieces.push(p);
            }
            // A sliver the policy calls a point: a touch, not a piece.
            Err(CurveClipError::Geometry(GeometryError::Numeric(NumericError::CollapsedTrim))) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(Cut {
        intervals,
        pieces,
        domain,
        tolerance,
    })
}

/// The crossings of `curve` with every edge of the ring, in the curve's own parameter, and the
/// parameter tolerance when they were found by bisection.
fn crossings(curve: &Curve, ring: &[Point2]) -> Result<(Vec<f64>, Option<f64>), CurveClipError> {
    match curve {
        Curve::Circle(c) => Ok((
            conic_hits(c.placement(), c.radius(), c.radius(), ring),
            None,
        )),
        Curve::Ellipse(e) => Ok((
            conic_hits(e.placement(), e.semi_axis_1(), e.semi_axis_2(), ring),
            None,
        )),
        Curve::Line(l) => Ok((
            segment_hits(l.origin(), l.direction(), ring).collect(),
            None,
        )),
        Curve::Polyline(pl) => {
            let v = pl.vertices();
            let mut out = Vec::new();
            for (i, w) in v.windows(2).enumerate() {
                let dir = w[0].vector_to(w[1]);
                let slack = POLICY.tol(1.0, 1.0);
                out.extend(
                    segment_hits(w[0], dir, ring)
                        .filter(|u| (-slack..=1.0 + slack).contains(u))
                        .map(|u| i as f64 + u.clamp(0.0, 1.0)),
                );
            }
            Ok((out, None))
        }
        Curve::BSpline(b) => spline_hits(b, ring).map(|(v, t)| (v, Some(t))),
        Curve::Bezier(z) => spline_hits(&z.as_bspline()?, ring).map(|(v, t)| (v, Some(t))),
        Curve::Trimmed(t) => {
            let trim = t.trim();
            let (basis_params, tolerance) = crossings(t.basis(), ring)?;
            let periodic = matches!(t.basis(), Curve::Circle(_) | Curve::Ellipse(_));
            let along = basis_params.into_iter().filter_map(|s| {
                let a = if periodic {
                    (s - trim.start).rem_euclid(TAU)
                } else {
                    s - trim.start
                };
                (0.0..=trim.span).contains(&a).then_some(if trim.reversed {
                    trim.span - a
                } else {
                    a
                })
            });
            Ok((along.collect(), tolerance))
        }
        other => Err(CurveClipError::Unsupported(other.kind())),
    }
}

/// The ring's edges, the last one closing it.
fn edges(ring: &[Point2]) -> impl Iterator<Item = (Point2, Point2)> + '_ {
    (0..ring.len()).map(|i| (ring[i], ring[(i + 1) % ring.len()]))
}

/// The parameters `s` at which `origin + s dir` crosses an edge of the ring (within the policy's
/// slack of the edge's ends, so a crossing through a vertex is not lost between its two edges).
fn segment_hits(origin: Point2, dir: Vector2, ring: &[Point2]) -> impl Iterator<Item = f64> + '_ {
    let slack = POLICY.tol(1.0, 1.0);
    edges(ring).filter_map(move |(p, q)| {
        let (ex, ey) = (q.x - p.x, q.y - p.y);
        let denom = dir.x * ey - dir.y * ex;
        if denom == 0.0 {
            return None;
        }
        let (wx, wy) = (p.x - origin.x, p.y - origin.y);
        let r = (wx * dir.y - wy * dir.x) / denom;
        (-slack..=1.0 + slack)
            .contains(&r)
            .then_some((wx * ey - wy * ex) / denom)
    })
}

/// The angles at which the conic `c + a cos t x + b sin t y` crosses an edge of the ring, in
/// `[0, 2 pi)`. In the conic's frame, scaled by its semi-axes, the conic is the unit circle and the
/// edge a segment, so a crossing is a root of one quadratic: one `sqrt`, and the angle of the root
/// from `detmath::atan2`.
fn conic_hits(f: &Placement2, a: f64, b: f64, ring: &[Point2]) -> Vec<f64> {
    let (o, x, y) = (f.origin(), f.x_axis(), f.y_axis());
    let local = |p: Point2| {
        let (dx, dy) = (p.x - o.x, p.y - o.y);
        ((dx * x.x + dy * x.y) / a, (dx * y.x + dy * y.y) / b)
    };
    let slack = POLICY.tol(1.0, 1.0);
    let mut out = Vec::new();
    for (p, q) in edges(ring) {
        let ((u0, v0), (u1, v1)) = (local(p), local(q));
        let (du, dv) = (u1 - u0, v1 - v0);
        // |(u0, v0) + s (du, dv)|^2 = 1, with the half linear term.
        let qa = du * du + dv * dv;
        let qb = u0 * du + v0 * dv;
        let qc = u0 * u0 + v0 * v0 - 1.0;
        let disc = qb * qb - qa * qc;
        if qa == 0.0 || disc < 0.0 {
            continue;
        }
        // The stable pair: no cancellation between `-qb` and the root.
        let k = -(qb + disc.sqrt().copysign(qb));
        let roots = if k == 0.0 {
            [0.0, 0.0]
        } else {
            [k / qa, qc / k]
        };
        for s in roots {
            if (-slack..=1.0 + slack).contains(&s) {
                let t = detmath::atan2(v0 + s * dv, u0 + s * du).rem_euclid(TAU);
                out.push(if t < TAU { t } else { 0.0 });
            }
        }
    }
    out
}

/// The parameters at which the spline crosses an edge of the ring, and the tolerance they are
/// found to: `Policy::V1`'s at the scale of the domain.
fn spline_hits(s: &BSpline, ring: &[Point2]) -> Result<(Vec<f64>, f64), CurveClipError> {
    let d = s.domain();
    let tol = POLICY.tol(d.hi() - d.lo(), d.lo().abs().max(d.hi().abs()));
    let mut knots: Vec<f64> = s
        .knots()
        .iter()
        .copied()
        .filter(|k| (d.lo()..=d.hi()).contains(k))
        .collect();
    knots.dedup();
    let spans = knots
        .windows(2)
        .map(|w| Ok((s.sub_range(w[0], w[1])?, w[0], w[1])))
        .collect::<Result<Vec<_>, GeometryError>>()?;
    /// The crossings of the source spline `s` with the edge's line on `[t0, t1]`, where `span` is `s`
    /// restricted to that interval (so its control points bound it). See the module docs.
    fn isolate(
        s: &BSpline,
        span: &BSpline,
        [t0, t1]: [f64; 2],
        (p, q): (Point2, Point2),
        tol: f64,
        depth: u32,
        out: &mut Vec<f64>,
    ) -> Result<(), CurveClipError> {
        let (ex, ey) = (q.x - p.x, q.y - p.y);
        let ee = ex * ex + ey * ey;
        if ee == 0.0 {
            return Ok(());
        }
        let side = |c: Point2| (c.x - p.x) * ey - (c.y - p.y) * ex;
        let along = |c: Point2| ((c.x - p.x) * ex + (c.y - p.y) * ey) / ee;
        let slack = POLICY.tol(1.0, 1.0);
        let cps = span.control_points();
        // Off the edge's extent altogether, by the convex hull.
        if cps.iter().all(|c| along(*c) < -slack) || cps.iter().all(|c| along(*c) > 1.0 + slack) {
            return Ok(());
        }
        let g: Vec<f64> = cps.iter().map(|c| side(*c)).collect();
        let on_edge = |t: f64, out: &mut Vec<f64>| -> Result<(), CurveClipError> {
            let a = along(s.eval(t)?);
            if (-slack..=1.0 + slack).contains(&a) {
                out.push(t);
            }
            Ok(())
        };
        // An end exactly on the line is a crossing candidate whatever the rest does.
        if g[0] == 0.0 {
            on_edge(t0, out)?;
        }
        if g[g.len() - 1] == 0.0 {
            on_edge(t1, out)?;
        }
        let signs: Vec<bool> = g.iter().filter(|v| **v != 0.0).map(|v| *v > 0.0).collect();
        let changes = signs.windows(2).filter(|w| w[0] != w[1]).count();
        if changes == 0 {
            return Ok(());
        }
        let ends_off_line = g[0] != 0.0 && g[g.len() - 1] != 0.0;
        if changes == 1 && ends_off_line {
            // Exactly one crossing (variation diminishing): bisect on PVec's evaluator.
            let start_positive = g[0] > 0.0;
            let (mut lo, mut hi) = (t0, t1);
            while hi - lo > tol {
                let mid = lo + 0.5 * (hi - lo);
                if mid <= lo || mid >= hi {
                    break;
                }
                let v = side(s.eval(mid)?);
                if v == 0.0 {
                    (lo, hi) = (mid, mid);
                    break;
                }
                if (v > 0.0) == start_positive {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return on_edge(lo + 0.5 * (hi - lo), out);
        }
        let mid = t0 + 0.5 * (t1 - t0);
        // Past the policy's resolution: one candidate, which costs nothing if spurious.
        if t1 - t0 <= tol || depth >= 64 || mid <= t0 || mid >= t1 {
            return on_edge(mid, out);
        }
        let left = span.sub_range(t0, mid)?;
        let right = span.sub_range(mid, t1)?;
        isolate(s, &left, [t0, mid], (p, q), tol, depth + 1, out)?;
        isolate(s, &right, [mid, t1], (p, q), tol, depth + 1, out)
    }

    let mut out = Vec::new();
    for edge in edges(ring) {
        for (span, t0, t1) in &spans {
            isolate(s, span, [*t0, *t1], edge, tol, 0, &mut out)?;
        }
    }
    Ok((out, tol))
}
