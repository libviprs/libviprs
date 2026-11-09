//! Closed-ring fill cutting against a convex clip (issue #1578, L5).
//!
//! A fill under a clip that is not a rectangle (a circle, a triangle, a half circle) is cut as a
//! closed ring, never as a polyline: a polyline clip can split a ring into open pieces that
//! triangulate wrongly. A convex clip cuts a ring correctly under both fill rules, so a cut ring
//! keeps its rule. This is Sutherland-Hodgman against a convex clip ring.

/// Cuts the closed `ring` (first point equal to last) to the convex closed `clip` ring, either
/// winding. Returns the cut ring, closed, or an empty vector when nothing is left. Returns `None`
/// when any coordinate is not finite, when `clip` is not convex, or when it is not a closed ring of
/// three or more corners: the caller keeps the fill uncut then, as display fidelity.
#[must_use]
pub fn cut_ring(ring: &[[f64; 2]], clip: &[[f64; 2]]) -> Option<Vec<[f64; 2]>> {
    if ring
        .iter()
        .chain(clip)
        .any(|p| !p[0].is_finite() || !p[1].is_finite())
    {
        return None;
    }
    let mut corners = clip.to_vec();
    corners.dedup();
    while corners.len() > 1 && corners.first() == corners.last() {
        corners.pop();
    }
    let n = corners.len();
    if n < 3 {
        return None;
    }
    // Convex means every turn goes the same way and the ring turns once around: a pentagram
    // turns the same way at every corner but goes round twice.
    let mut turning = 0.0f64;
    let mut sign = 0.0f64;
    for i in 0..n {
        let (a, b, c) = (corners[i], corners[(i + 1) % n], corners[(i + 2) % n]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
        let cross = u[0] * v[1] - u[1] * v[0];
        if cross == 0.0 && u[0] * v[0] + u[1] * v[1] >= 0.0 {
            continue;
        }
        if sign != 0.0 && cross * sign < 0.0 {
            return None;
        }
        if cross != 0.0 {
            sign = cross.signum();
        }
        turning += cross.atan2(u[0] * v[0] + u[1] * v[1]);
    }
    if sign == 0.0 || (turning.abs() - std::f64::consts::TAU).abs() > 1e-6 {
        return None;
    }
    let mut out = ring.to_vec();
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    for i in 0..n {
        let (a, b) = (corners[i], corners[(i + 1) % n]);
        // Positive on the inside, whichever way the clip is wound.
        let side =
            |p: [f64; 2]| sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]));
        let input = std::mem::take(&mut out);
        for (j, &current) in input.iter().enumerate() {
            let previous = input[(j + input.len() - 1) % input.len()];
            let (s_current, s_previous) = (side(current), side(previous));
            let crossing = || {
                let t = s_previous / (s_previous - s_current);
                [
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                ]
            };
            if s_current >= 0.0 {
                if s_previous < 0.0 {
                    out.push(crossing());
                }
                out.push(current);
            } else if s_previous >= 0.0 {
                out.push(crossing());
            }
        }
        if out.is_empty() {
            return Some(out);
        }
    }
    out.dedup();
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        return Some(Vec::new());
    }
    out.push(out[0]);
    Some(out)
}
