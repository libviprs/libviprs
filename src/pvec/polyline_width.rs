//! What a polyline's widths plot as (#1718).
//!
//! A polyline of one width is a stroke of that width, and the path stays as drawn (a bulged side
//! keeps its arc). A polyline whose width changes along it, or is wide on some sides and not on
//! others, is the outline of the polygon it covers: each side is a band between its start and end
//! widths, a bulged side follows its arc with the width running along the arc, and the left edges
//! run forward and the right edges back to close the ring. The widths are in the polyline's own
//! units, so the caller scales a stroke's width by the map the polyline is drawn through.

use super::provider::RawVertex;

/// The most points one outline ring may have. A polyline that would need more is stroked at its
/// widest width instead, so a hostile drawing cannot ask for a ring of hundreds of millions.
const MAX_RING_POINTS: usize = 1_000_000;

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
        // Basic arithmetic, not libm's hypot, which differs between architectures in the last bit.
        let chord = (dx * dx + dy * dy).sqrt();
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
        let radius = (rx * rx + ry * ry).sqrt();
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
    // What the polyline falls back to when its outline is too big or not a number: its widest
    // width as a stroke.
    let widest = (0..sides)
        .map(|i| at(i)[0].max(at(i)[1]))
        .fold(0.0, f64::max);
    let stroke = (widest > 0.0).then_some(Widths::Stroke(widest));
    let (mut left, mut right): (Vec<[f64; 2]>, Vec<[f64; 2]>) = (Vec::new(), Vec::new());
    for i in 0..sides {
        // Each point is on the ring twice, once on each edge, so the ring is twice what is held.
        if left.len() > MAX_RING_POINTS / 2 {
            return stroke;
        }
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
    if ring.len() > MAX_RING_POINTS || ring.iter().any(|p| !(p[0].is_finite() && p[1].is_finite()))
    {
        return stroke;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, bulge: f64) -> RawVertex {
        RawVertex { x, y: 0.0, bulge }
    }

    #[test]
    fn a_polyline_that_would_need_over_the_cap_falls_back_to_its_widest_stroke() {
        let vertices: Vec<RawVertex> = (0..12_000).map(|i| v(f64::from(i), 20.0)).collect();
        let widths = vec![[0.2, 0.6]; vertices.len()];
        assert_eq!(plan(&vertices, false, &widths), Some(Widths::Stroke(0.6)));
    }

    #[test]
    fn a_polyline_under_the_cap_is_an_outline_of_at_most_the_cap() {
        let vertices: Vec<RawVertex> = (0..1_000).map(|i| v(f64::from(i), 1.0)).collect();
        let widths = vec![[0.2, 0.6]; vertices.len()];
        let Some(Widths::Outline(ring)) = plan(&vertices, false, &widths) else {
            panic!("an outline");
        };
        assert!(ring.len() > 1_000 && ring.len() <= MAX_RING_POINTS);
    }

    #[test]
    fn a_bulge_or_width_that_overflows_never_gives_a_non_finite_outline() {
        for bulge in [1e200, -1e200, 1e-300, 1e-15, f64::MAX] {
            let vertices = [v(0.0, bulge), v(10.0, 0.0)];
            match plan(&vertices, false, &[[1.0, 2.0], [0.0, 0.0]]) {
                Some(Widths::Outline(r)) => {
                    assert!(
                        r.iter().all(|p| p.x.is_finite() && p.y.is_finite()),
                        "{bulge}"
                    );
                }
                Some(Widths::Stroke(w)) => assert!(w.is_finite()),
                None => panic!("a width is not dropped for a bulge of {bulge}"),
            }
        }
        let vertices = [v(0.0, 0.0), v(10.0, 0.0)];
        match plan(&vertices, false, &[[f64::MAX, f64::MAX / 2.0], [0.0, 0.0]]) {
            Some(Widths::Outline(r)) => {
                assert!(r.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
            }
            Some(Widths::Stroke(w)) => assert!(w.is_finite()),
            None => panic!("a huge width is not dropped"),
        }
    }
}
