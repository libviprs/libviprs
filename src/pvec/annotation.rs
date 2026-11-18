//! TEXT, MTEXT, ATTDEF, HATCH and SOLID as PVec's text and hatch streams (#1609).
//!
//! Neither is a curve, so neither goes into the geometry. Each rides on entities that are:
//!
//! * a text is anchored on a short polyline, from its insertion point one text height along its
//!   baseline, which gives the text a position in the drawing's box and something for the stream
//!   record to refer to;
//! * a hatch, or a SOLID (a filled quadrilateral), is its boundary: one closed polyline per loop,
//!   and the hatch record lists them.
//!
//! The line style stream marks all of those hidden, so a viewer draws the text and the fill and not
//! the anchor and the outline. That is why the whole thing needs that stream (see
//! [`PvecConversion::annotations`](crate::pvec_convert::PvecConversion::annotations)).
//!
//! What it does not carry: a text's font, width factor, oblique angle and justification (the record
//! has one anchor, the insertion point), an MTEXT's formatting beyond the characters it shows, and
//! the exact curve of a boundary: the container stores straight polylines, so an arc, an elliptic
//! arc or a spline in a boundary is carried as short chords. A hatch with a loop the provider could
//! not read at all (an edge that is not a curve, or edges that do not join) is not carried, since a
//! part of a region is not the region. A hatch pattern is carried by name, angle and scale, not by
//! its lines.

use super::components::{FillBody, TextBody};
use super::flatten::FlatEntity;
use super::provider::{Annotation, Point3, RawEntity, RawGeometry, RawVertex, z_axis_flip};

/// What a mapped annotation puts in its stream once the ids of its carriers are known.
pub(crate) enum Body {
    /// A text record on the first carrier.
    Text(TextBody),
    /// A hatch record over every carrier.
    Fill(FillBody),
}

/// The hidden entities to write for `annotation`, in block space like every other leaf, and the
/// stream record that goes with them, or why it cannot be mapped.
pub(crate) fn map(
    leaf: &FlatEntity,
    annotation: Annotation,
) -> Result<(Vec<RawEntity>, Body), &'static str> {
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
            let scaled = height * (ux * ux + uy * uy).sqrt();
            if !(scaled.is_finite() && scaled > 0.0 && (dx != 0.0 || dy != 0.0)) {
                return Err("its placement collapses it");
            }
            let [x, y] = leaf.map_point(to_block(position[0], position[1]));
            let anchor = carrier(RawGeometry::Polyline {
                vertices: vec![
                    RawVertex {
                        x: position[0],
                        y: position[1],
                        bulge: 0.0,
                    },
                    RawVertex {
                        x: position[0] + cos * height,
                        y: position[1] + sin * height,
                        bulge: 0.0,
                    },
                ],
                closed: false,
                elevation: 0.0,
                normal,
            });
            let body = TextBody {
                x,
                y,
                height: scaled,
                rotation: dy.atan2(dx),
                text: content,
            };
            Ok((vec![anchor], Body::Text(body)))
        }
        Annotation::Fill {
            rings,
            elevation,
            normal,
            pattern,
            rgba,
            unread_loops,
        } => {
            if unread_loops > 0 {
                return Err("a boundary loop could not be read as a ring of curves");
            }
            if rings.is_empty() {
                return Err("it has no boundary");
            }
            z_axis_flip(normal).ok_or("its normal is not the Z axis")?;
            // PVec's container stores straight polylines and refuses a ring with arcs in it, so an
            // arc becomes short chords. A corner repeated up to rounding noise is one corner, and
            // a ring left with fewer than three has no area, so it adds nothing to the fill.
            fn straight(ring: &[RawVertex]) -> Option<Vec<RawVertex>> {
                let mut points: Vec<[f64; 2]> = Vec::new();
                for (i, a) in ring.iter().enumerate() {
                    let b = &ring[(i + 1) % ring.len()];
                    points.push([a.x, a.y]);
                    let theta = 4.0 * a.bulge.atan();
                    let chord = (b.x - a.x).hypot(b.y - a.y);
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
            let carriers: Vec<RawEntity> = rings
                .iter()
                .filter_map(|ring| straight(ring))
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
                return Err("its corners are on one line, so there is no area to fill");
            }
            Ok((carriers, Body::Fill(FillBody { pattern, rgba })))
        }
    }
}
