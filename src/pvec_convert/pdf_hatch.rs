//! The lines of a pattern hatch (#1670): each line family of the pattern, cut to the hatch's boundary
//! rings (even-odd, so an island is left out) and to its dashes. Everything is in one space, the
//! page's points. The work is bounded: a pattern whose lines, or whose dashes, would be more than
//! [`MAX_LINES`] or [`MAX_SEGMENTS`] is refused with [`TooDense`], and the plot fills the hatch
//! instead, as a plotter does for a hatch it cannot space.

use super::PlacedPatternLine;

/// Most parallel lines one family may lay over a boundary.
pub(super) const MAX_LINES: f64 = 20_000.0;
/// Most strokes one hatch may be drawn with.
pub(super) const MAX_SEGMENTS: usize = 250_000;
/// Most line-against-edge tests one hatch may take.
const MAX_TESTS: f64 = 60_000_000.0;

/// The pattern is too dense (or too fine) to draw as lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TooDense;

/// The strokes, as (x0, y0, x1, y1), of `families` over `rings`, which are closed polygons.
pub(super) fn expand(
    rings: &[Vec<[f64; 2]>],
    families: &[PlacedPatternLine],
) -> Result<Vec<[f64; 4]>, TooDense> {
    let edges: f64 = rings.iter().map(|r| r.len() as f64).sum();
    let mut out: Vec<[f64; 4]> = Vec::new();
    for family in families {
        let [dx, dy] = family.direction;
        let normal = [-dy, dx];
        let across = family.step[0] * normal[0] + family.step[1] * normal[1];
        if !across.is_finite() || across.abs() < 1e-12 {
            // No spacing across: the family is one line, or none; it adds nothing to draw.
            continue;
        }
        let project = |p: [f64; 2]| p[0] * normal[0] + p[1] * normal[1];
        let (mut low, mut high) = (f64::MAX, f64::MIN);
        for p in rings.iter().flatten() {
            let v = project(*p);
            low = low.min(v);
            high = high.max(v);
        }
        if low > high {
            continue;
        }
        let o = project(family.origin);
        // The lines are at o + k * across; k runs over what lies between low and high.
        let (k0, k1) = {
            let (a, b) = ((low - o) / across, (high - o) / across);
            (a.min(b).ceil(), a.max(b).floor())
        };
        let count = k1 - k0 + 1.0;
        if count > MAX_LINES || count * edges > MAX_TESTS {
            return Err(TooDense);
        }
        let period: f64 = family.dashes.iter().map(|d| d.abs()).sum();
        let dashed = !family.dashes.is_empty() && period > 0.0;
        if dashed && !period.is_finite() {
            return Err(TooDense);
        }
        let mut k = k0;
        while k <= k1 {
            let origin = [
                family.origin[0] + k * family.step[0],
                family.origin[1] + k * family.step[1],
            ];
            let at = |p: [f64; 2]| (p[0] - origin[0]) * dx + (p[1] - origin[1]) * dy;
            let side =
                |p: [f64; 2]| (p[0] - origin[0]) * normal[0] + (p[1] - origin[1]) * normal[1];
            let mut crossings: Vec<f64> = Vec::new();
            for ring in rings {
                for (i, a) in ring.iter().enumerate() {
                    let b = ring[(i + 1) % ring.len()];
                    let (sa, sb) = (side(*a), side(b));
                    if (sa > 0.0) != (sb > 0.0) {
                        let u = sa / (sa - sb);
                        let (ta, tb) = (at(*a), at(b));
                        crossings.push(ta + u * (tb - ta));
                    }
                }
            }
            crossings.sort_by(f64::total_cmp);
            for pair in crossings.chunks_exact(2) {
                let (t0, t1) = (pair[0], pair[1]);
                if t1 <= t0 {
                    continue;
                }
                let mut push = |a: f64, b: f64| -> Result<(), TooDense> {
                    if out.len() >= MAX_SEGMENTS {
                        return Err(TooDense);
                    }
                    out.push([
                        origin[0] + a * dx,
                        origin[1] + a * dy,
                        origin[0] + b * dx,
                        origin[1] + b * dy,
                    ]);
                    Ok(())
                };
                if !dashed {
                    push(t0, t1)?;
                    continue;
                }
                // The dashes in this piece: the work is its length over the period, times the
                // elements of a period.
                if (t1 - t0) / period * family.dashes.len() as f64 > MAX_SEGMENTS as f64 {
                    return Err(TooDense);
                }
                // The cycle starts at t = 0 on the line's origin.
                let mut at_start = (t0 / period).floor() * period;
                'cycles: loop {
                    let mut pos = at_start;
                    for d in &family.dashes {
                        let len = d.abs();
                        let (a, b) = (pos, pos + len);
                        pos = b;
                        if *d >= 0.0 {
                            let (lo, hi) = (a.max(t0), b.min(t1));
                            if hi > lo || (len == 0.0 && a >= t0 && a <= t1) {
                                push(lo, hi.max(lo))?;
                            }
                        }
                        if a >= t1 {
                            break 'cycles;
                        }
                    }
                    at_start += period;
                    if at_start > t1 {
                        break;
                    }
                }
            }
            k += 1.0;
        }
    }
    Ok(out)
}
