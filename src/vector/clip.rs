//! Clipping and simplifying polylines, on the plain `f64` grid.
//!
//! Both operations are iterative: a hostile drawing can hand over a polyline of
//! a million points, and recursion depth must not follow the point count.

/// Clips an open polyline to the slab `lo <= p[axis] <= hi`.
///
/// Returns the pieces inside, in order. A polyline that leaves and re-enters
/// comes back as several pieces; one that never touches the slab comes back
/// as none. Pieces always have at least two points. Endpoints where the
/// polyline crosses the slab edge are computed once per crossing, from the
/// segment's own endpoints, so two neighbouring slabs agree on where a line
/// crosses their shared boundary to the last bit.
#[must_use]
pub fn clip_slab(points: &[[f64; 2]], axis: usize, lo: f64, hi: f64) -> Vec<Vec<[f64; 2]>> {
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    if points.len() < 2 {
        return out;
    }
    let inside = |p: [f64; 2]| p[axis] >= lo && p[axis] <= hi;
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (da, db) = (a[axis], b[axis]);
        // Parameter range of the segment inside the slab.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        if da == db {
            if !inside(a) {
                flush(&mut cur, &mut out);
                continue;
            }
        } else {
            let ta = (lo - da) / (db - da);
            let tb = (hi - da) / (db - da);
            let (enter, leave) = if ta < tb { (ta, tb) } else { (tb, ta) };
            t0 = t0.max(enter);
            t1 = t1.min(leave);
            if t0 >= t1 {
                flush(&mut cur, &mut out);
                continue;
            }
        }
        let at = |t: f64, end: [f64; 2], start: [f64; 2]| -> [f64; 2] {
            if t == 0.0 {
                start
            } else if t == 1.0 {
                end
            } else {
                [
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t,
                ]
            }
        };
        let p0 = at(t0, b, a);
        let p1 = at(t1, b, a);
        if cur.is_empty() {
            cur.push(p0);
        } else if cur.last() != Some(&p0) {
            // The previous piece ended where this segment does not begin.
            flush(&mut cur, &mut out);
            cur.push(p0);
        }
        cur.push(p1);
        if t1 < 1.0 {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn flush(cur: &mut Vec<[f64; 2]>, out: &mut Vec<Vec<[f64; 2]>>) {
    if cur.len() >= 2 {
        out.push(std::mem::take(cur));
    } else {
        cur.clear();
    }
}

/// Ramer-Douglas-Peucker with an explicit stack, keeping both ends.
///
/// `eps` is the largest distance a dropped point may be from the kept
/// polyline, in the units of the points.
#[must_use]
pub fn simplify(points: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    let n = points.len();
    if n <= 2 || eps <= 0.0 {
        return points.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (a, b) = (points[i], points[j]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let mut worst = (0.0f64, i);
        for (k, p) in points.iter().enumerate().take(j).skip(i + 1) {
            let d = if len2 == 0.0 {
                ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt()
            } else {
                // Distance to the chord's line, which is what the tolerance
                // is about for a stroke.
                ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len2.sqrt()
            };
            if d > worst.0 {
                worst = (d, k);
            }
        }
        if worst.0 > eps {
            keep[worst.1] = true;
            stack.push((i, worst.1));
            stack.push((worst.1, j));
        }
    }
    points
        .iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_across_a_slab_is_cut_at_both_edges() {
        let out = clip_slab(&[[-5.0, 1.0], [15.0, 1.0]], 0, 0.0, 10.0);
        assert_eq!(out, vec![vec![[0.0, 1.0], [10.0, 1.0]]]);
    }

    #[test]
    fn a_polyline_that_leaves_and_returns_is_two_pieces() {
        let out = clip_slab(
            &[[1.0, 0.0], [20.0, 0.0], [20.0, 5.0], [2.0, 5.0]],
            0,
            0.0,
            10.0,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![[1.0, 0.0], [10.0, 0.0]]);
        assert_eq!(out[1], vec![[10.0, 5.0], [2.0, 5.0]]);
    }

    #[test]
    fn a_segment_wholly_outside_or_touching_one_point_gives_nothing() {
        assert!(clip_slab(&[[11.0, 0.0], [20.0, 0.0]], 0, 0.0, 10.0).is_empty());
        assert!(clip_slab(&[[10.0, 0.0], [20.0, 0.0]], 0, 0.0, 10.0).is_empty());
        assert!(clip_slab(&[[3.0, 0.0]], 0, 0.0, 10.0).is_empty());
    }

    #[test]
    fn a_line_along_the_edge_is_inside() {
        let out = clip_slab(&[[0.0, 0.0], [0.0, 9.0]], 0, 0.0, 10.0);
        assert_eq!(out, vec![vec![[0.0, 0.0], [0.0, 9.0]]]);
    }

    #[test]
    fn a_continuous_run_inside_stays_one_piece() {
        let pts = [[1.0, 1.0], [2.0, 5.0], [3.0, 2.0], [9.0, 9.0]];
        assert_eq!(clip_slab(&pts, 0, 0.0, 10.0), vec![pts.to_vec()]);
    }

    #[test]
    fn neighbouring_slabs_agree_on_the_crossing() {
        let line = [[0.3, 0.1], [7.9, 3.7]];
        let left = clip_slab(&line, 0, 0.0, 4.0);
        let right = clip_slab(&line, 0, 4.0, 8.0);
        assert_eq!(left[0].last(), right[0].first());
    }

    #[test]
    fn simplify_keeps_ends_and_corners() {
        let pts = [
            [0.0, 0.0],
            [1.0, 0.001],
            [2.0, 0.0],
            [2.0, 5.0],
            [2.0, 10.0],
        ];
        let s = simplify(&pts, 0.01);
        assert_eq!(s, vec![[0.0, 0.0], [2.0, 0.0], [2.0, 10.0]]);
    }

    #[test]
    fn simplify_survives_a_million_points() {
        let pts: Vec<[f64; 2]> = (0..1_000_000)
            .map(|i| [f64::from(i), if i % 2 == 0 { 0.0 } else { 0.4 }])
            .collect();
        let s = simplify(&pts, 0.5);
        assert_eq!(s.len(), 2);
    }
}
