//! TEXT, MTEXT, ATTDEF, HATCH and SOLID as PVec's text and hatch streams (#1609), and the text of a
//! MULTILEADER, a TOLERANCE, or a DIMENSION whose block is missing (#1658).
//!
//! Neither is a curve, so neither goes into the geometry. Each rides on entities that are:
//!
//! * a text is anchored on a polyline along its baseline, from where the run starts to where it
//!   ends. Its length is the run's printed width (planviewer scales the glyphs to it, as it does
//!   with a PDF's `/Widths`), worked out from Helvetica's widths and the style's width factor
//!   because the drawing's own face is not read. An MTEXT is one text and anchor per printed
//!   line, and the start is the printed baseline's, after the TEXT's justification or the
//!   MTEXT's attachment point (#1621);
//! * a hatch, or a SOLID (a filled quadrilateral), is its boundary: one closed polyline per loop,
//!   and the hatch record lists them.
//!
//! The line style stream marks all of those hidden, so a viewer draws the text and the fill and not
//! the anchor and the outline. That is why the whole thing needs that stream (see
//! [`PvecConversion::annotations`](crate::pvec_convert::PvecConversion::annotations)).
//!
//! What it does not carry: a text's font and oblique angle, an MTEXT's formatting beyond the
//! characters it shows (and the word wrap its reference width would cause), and
//! the exact curve of a boundary: the container stores straight polylines, so an arc, an elliptic
//! arc or a spline in a boundary is carried as short chords. A hatch with a loop the provider could
//! not read at all (an edge that is not a curve, or edges that do not join) is not carried, since a
//! part of a region is not the region. A hatch pattern is carried by name, angle and scale, not by
//! its lines.

use super::components::{FillBody, PlacedPatternLine, TextBody};
use super::dwg_acadrust::run_advance;
use super::flatten::FlatEntity;
use super::provider::{
    Annotation, Point3, RawEntity, RawGeometry, RawVertex, TextAnchorV, z_axis_flip,
};

/// What a mapped annotation puts in its stream once the ids of its carriers are known.
pub(crate) enum Body {
    /// One text record per printed line, each on the carrier at its index.
    Text(Vec<TextBody>),
    /// A hatch record over every carrier.
    Fill(FillBody),
    /// Several annotations of one entity (an OLE2FRAME's cells and text, #1705): each with how many
    /// carriers it took, in order.
    Runs(Vec<(usize, Body)>),
}

/// Why [`map`] returns nothing for a text with no line inside the clip it was asked to keep to.
pub(crate) const CLIPPED: &str = "it is outside the sheet or viewport it is drawn through";

/// [`map`] for each of `runs`, whose carriers and records go together under one body (#1705). A run
/// that maps to nothing (cut away by the clip, or not drawable) is left out; what is left is an error
/// only when nothing is.
pub(crate) fn map_runs(
    leaf: &FlatEntity,
    runs: Vec<Annotation>,
    keep: &dyn Fn([f64; 2]) -> bool,
    clip: Option<[f64; 4]>,
) -> Result<(Vec<RawEntity>, Body), &'static str> {
    let (mut carriers, mut bodies, mut clipped) = (Vec::new(), Vec::new(), false);
    for run in runs {
        match map(leaf, run, keep, clip) {
            Ok((c, body)) => {
                bodies.push((c.len(), body));
                carriers.extend(c);
            }
            Err(CLIPPED) => clipped = true,
            Err(_) => {}
        }
    }
    if bodies.is_empty() {
        return Err(if clipped {
            CLIPPED
        } else {
            "none of its cells or text can be drawn"
        });
    }
    Ok((carriers, Body::Runs(bodies)))
}

/// The hidden entities to write for `annotation`, in block space like every other leaf, and the
/// stream record that goes with them, or why it cannot be mapped. A text is laid out in its own
/// plane, one anchor and record per printed line, and `keep` says whether a line starting at a
/// point of the drawing (after the leaf's map) is wanted, so a text seen through a viewport or on
/// a sheet is cut to it line by line.
pub(crate) fn map(
    leaf: &FlatEntity,
    annotation: Annotation,
    keep: &dyn Fn([f64; 2]) -> bool,
    clip: Option<[f64; 4]>,
) -> Result<(Vec<RawEntity>, Body), &'static str> {
    /// How far below its baseline a line's descenders reach, in text heights. Capitals stand one text
    /// height above the baseline.
    const DESCENT: f64 = 0.25;

    let carrier = |geometry: RawGeometry| RawEntity {
        geometry,
        provenance: leaf.entity.provenance.clone(),
        style: leaf.entity.style.clone(),
    };
    match annotation {
        Annotation::Text {
            position,
            height,
            rotation,
            content,
            normal,
            width_factor,
            anchor,
            fit,
            block_width,
            line_spacing,
        } => {
            let flip = z_axis_flip(normal).ok_or("its normal is not the Z axis")?;
            // The entity's own plane to the block's: a -Z normal mirrors x.
            let to_block = |x: f64, y: f64| if flip { [-x, y, 0.0] } else { [x, y, 0.0] };
            let [m00, m01, m10, m11, ..] = leaf.matrix();
            let linear = |v: Point3| (m00 * v[0] + m01 * v[1], m10 * v[0] + m11 * v[1]);
            let (sin, cos) = rotation.sin_cos();
            // The baseline direction and the text's own up, as the drawing sees them.
            let (dx, dy) = linear(to_block(cos, sin));
            let (ux, uy) = linear(to_block(-sin, cos));
            let scaled = |h: f64| h * (ux * ux + uy * uy).sqrt();
            if !(scaled(height).is_finite() && scaled(height) > 0.0 && (dx != 0.0 || dy != 0.0)) {
                return Err("its placement collapses it");
            }
            let lines: Vec<&str> = content.split('\n').map(str::trim).collect();
            let factor = if width_factor > 0.0 {
                width_factor
            } else {
                1.0
            };
            let natural: Vec<f64> = lines
                .iter()
                .map(|l| run_advance(l) * height * factor)
                .collect();
            // An aligned or fitted TEXT is one line stretched between its two points; aligned
            // scales the height with the width.
            let (height, widths) = match fit {
                Some((span, scales)) if natural[0] > 0.0 => {
                    let k = span / natural[0];
                    (if scales { height * k } else { height }, vec![span])
                }
                _ => (height, natural),
            };
            let spacing = if line_spacing > 0.0 {
                line_spacing
            } else {
                1.0
            };
            // AutoCAD's single spacing puts one baseline 5/3 of a text height under the last.
            let pitch = height * spacing * 5.0 / 3.0;
            let count = lines.len() as f64;
            let block = block_width.max(widths.iter().copied().fold(0.0, f64::max));
            let left = -anchor.0 * block;
            let span_down = height * (1.0 + DESCENT) + (count - 1.0) * pitch;
            // The first baseline relative to the position, up positive.
            let first = match anchor.1 {
                TextAnchorV::Baseline => 0.0,
                TextAnchorV::Top => -height,
                TextAnchorV::Middle => span_down / 2.0 - height,
                TextAnchorV::Bottom => span_down - height,
            };
            let mut carriers = Vec::new();
            let mut records = Vec::new();
            for (i, line) in lines.iter().enumerate() {
                let width = widths[i.min(widths.len() - 1)];
                if line.is_empty() || !(width.is_finite() && width > 0.0) {
                    continue;
                }
                let up = first - i as f64 * pitch;
                let start = [
                    position[0] + cos * left - sin * up,
                    position[1] + sin * left + cos * up,
                ];
                let [x, y] = leaf.map_point(to_block(start[0], start[1]));
                if !keep([x, y]) {
                    continue;
                }
                carriers.push(carrier(RawGeometry::Polyline {
                    vertices: vec![
                        RawVertex {
                            x: start[0],
                            y: start[1],
                            bulge: 0.0,
                        },
                        RawVertex {
                            x: start[0] + cos * width,
                            y: start[1] + sin * width,
                            bulge: 0.0,
                        },
                    ],
                    closed: false,
                    elevation: 0.0,
                    normal,
                }));
                records.push(TextBody {
                    x,
                    y,
                    height: scaled(height),
                    rotation: dy.atan2(dx),
                    text: (*line).to_owned(),
                });
            }
            if records.is_empty() {
                return Err(if lines.iter().any(|l| !l.is_empty()) {
                    CLIPPED
                } else {
                    "it prints nothing"
                });
            }
            Ok((carriers, Body::Text(records)))
        }
        Annotation::Fill {
            rings,
            elevation,
            normal,
            pattern,
            rgba,
            unread_loops,
            pattern_lines,
        } => {
            if unread_loops > 0 {
                return Err("a boundary loop could not be read as a ring of curves");
            }
            if rings.is_empty() {
                return Err("it has no boundary");
            }
            let flip = z_axis_flip(normal).ok_or("its normal is not the Z axis")?;
            let to_block = |x: f64, y: f64| if flip { [-x, y, 0.0] } else { [x, y, 0.0] };
            // PVec's container stores straight polylines and refuses a ring with arcs in it, so an
            // arc becomes short chords. A corner repeated up to rounding noise is one corner, and
            // a ring left with fewer than three has no area, so it adds nothing to the fill.
            fn straight(ring: &[RawVertex]) -> Option<Vec<RawVertex>> {
                let mut points: Vec<[f64; 2]> = Vec::new();
                for (i, a) in ring.iter().enumerate() {
                    let b = &ring[(i + 1) % ring.len()];
                    points.push([a.x, a.y]);
                    let theta = 4.0 * a.bulge.atan();
                    let chord = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
                    if a.bulge == 0.0 || !theta.is_finite() || chord == 0.0 {
                        continue;
                    }
                    // The centre sits on the chord's left for a counter-clockwise span.
                    let h = chord / 2.0 * (1.0 - a.bulge * a.bulge) / (2.0 * a.bulge);
                    let centre = [
                        (a.x + b.x) / 2.0 - (b.y - a.y) / chord * h,
                        (a.y + b.y) / 2.0 + (b.x - a.x) / chord * h,
                    ];
                    let steps = (theta.abs() / (std::f64::consts::PI / 48.0))
                        .ceil()
                        .max(1.0) as usize;
                    for k in 1..steps {
                        let (sin, cos) = (theta * k as f64 / steps as f64).sin_cos();
                        let (dx, dy) = (a.x - centre[0], a.y - centre[1]);
                        points.push([
                            centre[0] + dx * cos - dy * sin,
                            centre[1] + dx * sin + dy * cos,
                        ]);
                    }
                }
                let reach = points.iter().flatten().fold(1.0_f64, |m, v| m.max(v.abs()));
                // PVec refuses a span shorter than 1e-12 or 8 ulps of the coordinates, so noise up
                // to a few times that is merged and a real sliver, orders of magnitude longer, is not.
                let floor = (reach * 1e-14).max(2e-12);
                let same = |p: [f64; 2], q: [f64; 2]| {
                    (p[0] - q[0]).abs().max((p[1] - q[1]).abs()) <= floor
                };
                let mut kept: Vec<[f64; 2]> = Vec::new();
                for p in points {
                    if kept.last().is_none_or(|q| !same(*q, p)) {
                        kept.push(p);
                    }
                }
                while kept.len() > 1 && same(kept[0], kept[kept.len() - 1]) {
                    kept.pop();
                }
                (kept.len() >= 3).then(|| {
                    kept.into_iter()
                        .map(|[x, y]| RawVertex { x, y, bulge: 0.0 })
                        .collect()
                })
            }
            // A ring wholly inside what it is seen through is carried as it is, one wholly outside is
            // left out, and one that crosses the edge of a rectangle is cut to it (Sutherland and
            // Hodgman, in the drawing's space, then mapped back into the entity's plane). A ring
            // seen through a polygon is carried or left out whole.
            let [ma, mb, mc, md, mx, my] = leaf.matrix();
            let det = ma * md - mb * mc;
            let cut = |vertices: Vec<RawVertex>| -> Option<Vec<RawVertex>> {
                let seen: Vec<[f64; 2]> = vertices
                    .iter()
                    .map(|v| leaf.map_point(to_block(v.x, v.y)))
                    .collect();
                if seen.iter().all(|p| keep(*p)) {
                    return Some(vertices);
                }
                let b = clip?;
                if det == 0.0 || !det.is_finite() {
                    return None;
                }
                let mut ring = seen;
                for edge in 0..4 {
                    let inside = |p: [f64; 2]| match edge {
                        0 => p[0] >= b[0],
                        1 => p[0] <= b[2],
                        2 => p[1] >= b[1],
                        _ => p[1] <= b[3],
                    };
                    let cross = |p: [f64; 2], q: [f64; 2]| {
                        let (axis, at) = match edge {
                            0 => (0, b[0]),
                            1 => (0, b[2]),
                            2 => (1, b[1]),
                            _ => (1, b[3]),
                        };
                        let t = (at - p[axis]) / (q[axis] - p[axis]);
                        let mut r = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
                        r[axis] = at;
                        r
                    };
                    let mut next: Vec<[f64; 2]> = Vec::new();
                    for (i, p) in ring.iter().enumerate() {
                        let q = ring[(i + 1) % ring.len()];
                        match (inside(*p), inside(q)) {
                            (true, true) => next.push(q),
                            (true, false) => next.push(cross(*p, q)),
                            (false, true) => {
                                next.push(cross(*p, q));
                                next.push(q);
                            }
                            (false, false) => {}
                        }
                    }
                    if next.len() < 3 {
                        return None;
                    }
                    ring = next;
                }
                // Back to the entity's plane: the inverse of the leaf's map, then the plane's flip.
                Some(
                    ring.iter()
                        .map(|p| {
                            let (px, py) = (p[0] - mx, p[1] - my);
                            let bx = (md * px - mb * py) / det;
                            let by = (-mc * px + ma * py) / det;
                            RawVertex {
                                x: if flip { -bx } else { bx },
                                y: by,
                                bulge: 0.0,
                            }
                        })
                        .collect(),
                )
            };
            let carriers: Vec<RawEntity> = rings
                .iter()
                .filter_map(|ring| straight(ring))
                .filter_map(cut)
                .map(|vertices| {
                    carrier(RawGeometry::Polyline {
                        vertices,
                        closed: true,
                        elevation,
                        normal,
                    })
                })
                .collect();
            if carriers.is_empty() {
                // Rings that had an area and none of it in view are outside, not flat.
                if rings.iter().any(|r| straight(r).is_some()) {
                    return Err(CLIPPED);
                }
                return Err("its corners are on one line, so there is no area to fill");
            }
            // The pattern's line families in the drawing's space (a viewport's map included): the
            // direction and the step by the leaf's linear part, the dash lengths by what it does to
            // a unit along the line.
            let lines: Vec<PlacedPatternLine> = if pattern.is_some() {
                let linear = |v: [f64; 2]| {
                    let [bx, by, _] = to_block(v[0], v[1]);
                    [ma * bx + mb * by, mc * bx + md * by]
                };
                pattern_lines
                    .iter()
                    .filter_map(|l| {
                        let (s, c) = l.angle.sin_cos();
                        let along = linear([c, s]);
                        let length = pvec_numeric::detmath::hypot(along[0], along[1]);
                        if !(length.is_finite() && length > 0.0) {
                            return None;
                        }
                        let (ox, oy) = (l.offset[0], l.offset[1]);
                        let step = linear([ox * c - oy * s, ox * s + oy * c]);
                        let origin = leaf.map_point(to_block(l.base[0], l.base[1]));
                        let ok = origin.iter().chain(&step).all(|v| v.is_finite())
                            && l.dashes.iter().all(|d| d.is_finite());
                        ok.then(|| PlacedPatternLine {
                            origin,
                            direction: [along[0] / length, along[1] / length],
                            step,
                            dashes: l.dashes.iter().map(|d| d * length).collect(),
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Ok((
                carriers,
                Body::Fill(FillBody {
                    pattern,
                    rgba,
                    lines,
                }),
            ))
        }
    }
}
