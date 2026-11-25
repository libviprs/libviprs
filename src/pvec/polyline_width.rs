//! What a polyline's widths plot as (#1718).
//!
//! A polyline of one width is a stroke of that width, and the path stays as drawn (a bulged side
//! keeps its arc). A polyline whose width changes along it, or is wide on some sides and not on
//! others, is the outline of the polygon it covers: each side is a band between its start and end
//! widths, a bulged side follows its arc with the width running along the arc, and the left edges
//! run forward and the right edges back to close the ring. The widths are in the polyline's own
//! units, so the caller scales a stroke's width by the map the polyline is drawn through.

use super::provider::RawVertex;

/// What a polyline's widths plot as.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Widths {
    /// Every side has this one width: stroke the path at it.
    Stroke(f64),
    /// The widths differ along the polyline: this closed ring, with no bulges, is what it covers.
    Outline(Vec<RawVertex>),
}

/// What the widths of the polyline through `vertices` plot as, or `None` when it has none.
/// `widths` is the start and end width of each vertex, in the vertices' order: a vertex's pair
/// is the width of the side that leaves it.
pub(crate) fn plan(vertices: &[RawVertex], closed: bool, widths: &[[f64; 2]]) -> Option<Widths> {
    /// One point along a side, the left normal of the path there and the width at it.
    type Sample = ([f64; 2], [f64; 2], f64);

    /// The samples along the side from `a` to `b` with `bulge`, widening from `start` to `end`.
    fn side(a: [f64; 2], b: [f64; 2], bulge: f64, start: f64, end: f64) -> Vec<Sample> {
        // The angle one flattened piece of an arc turns through, at most.
        const ARC_STEP: f64 = std::f64::consts::PI / 24.0;
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let chord = dx.hypot(dy);
        if !(chord.is_finite() && chord > 0.0) {
            return Vec::new();
        }
        let left = [-dy / chord, dx / chord];
        if !(bulge.is_finite() && bulge != 0.0) {
            return vec![(a, left, start), (b, left, end)];
        }
        // The included angle is 4 atan(bulge). The centre is on the chord's left for a positive
        // bulge, at (chord / 2) / tan(angle / 2) = chord (1 - bulge^2) / (4 bulge) from its middle.
        let angle = 4.0 * bulge.atan();
        let offset = chord * (1.0 - bulge * bulge) / (4.0 * bulge);
        let centre = [
            (a[0] + b[0]) / 2.0 + left[0] * offset,
            (a[1] + b[1]) / 2.0 + left[1] * offset,
        ];
        let (rx, ry) = (a[0] - centre[0], a[1] - centre[1]);
        let radius = rx.hypot(ry);
        let pieces = ((angle.abs() / ARC_STEP).ceil() as usize).clamp(1, 96);
        (0..=pieces)
            .map(|j| {
                let t = j as f64 / pieces as f64;
                let (sin, cos) = (angle * t).sin_cos();
                let out = [
                    (rx * cos - ry * sin) / radius,
                    (rx * sin + ry * cos) / radius,
                ];
                // Travelling counter-clockwise the left of the path is towards the centre.
                let normal = if bulge > 0.0 { [-out[0], -out[1]] } else { out };
                let at = [centre[0] + out[0] * radius, centre[1] + out[1] * radius];
                (at, normal, start + (end - start) * t)
            })
            .collect()
    }
    // A width that is not a positive number is none.
    let clean = |w: f64| if w.is_finite() && w > 0.0 { w } else { 0.0 };
    let sides = match vertices.len() {
        0 | 1 => return None,
        n if closed && n > 2 => n,
        n => n - 1,
    };
    let at = |i: usize| {
        widths
            .get(i)
            .map_or([0.0, 0.0], |w| [clean(w[0]), clean(w[1])])
    };
    let first = at(0);
    if (0..sides).all(|i| at(i) == [0.0, 0.0]) {
        return None;
    }
    if first[0] > 0.0 && (0..sides).all(|i| at(i) == [first[0], first[0]]) {
        return Some(Widths::Stroke(first[0]));
    }
    let (mut left, mut right): (Vec<[f64; 2]>, Vec<[f64; 2]>) = (Vec::new(), Vec::new());
    for i in 0..sides {
        let (a, b) = (&vertices[i], &vertices[(i + 1) % vertices.len()]);
        let [start, end] = at(i);
        for (p, n, w) in side([a.x, a.y], [b.x, b.y], a.bulge, start, end) {
            left.push([p[0] + n[0] * w / 2.0, p[1] + n[1] * w / 2.0]);
            right.push([p[0] - n[0] * w / 2.0, p[1] - n[1] * w / 2.0]);
        }
    }
    let mut ring: Vec<[f64; 2]> = Vec::with_capacity(left.len() * 2);
    for p in left.into_iter().chain(right.into_iter().rev()) {
        let close = |q: &[f64; 2]| (q[0] - p[0]).abs() <= 1e-12 && (q[1] - p[1]).abs() <= 1e-12;
        if !ring.last().is_some_and(close) {
            ring.push(p);
        }
    }
    while ring.len() > 1 && ring.first().zip(ring.last()).is_some_and(|(f, l)| f == l) {
        ring.pop();
    }
    (ring.len() >= 3).then(|| {
        Widths::Outline(
            ring.into_iter()
                .map(|[x, y]| RawVertex { x, y, bulge: 0.0 })
                .collect(),
        )
    })
}
