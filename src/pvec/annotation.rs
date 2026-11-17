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
//! a hatch that has a loop of elliptic or spline edges (a part of a region is not the region). A
//! hatch pattern is carried by name, angle and scale, not by its lines.

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
                return Err("a boundary loop is not a ring of lines and arcs");
            }
            if rings.is_empty() || rings.iter().any(|r| r.len() < 3) {
                return Err("it has no boundary");
            }
            z_axis_flip(normal).ok_or("its normal is not the Z axis")?;
            let carriers = rings
                .into_iter()
                .map(|vertices| {
                    carrier(RawGeometry::Polyline {
                        vertices,
                        closed: true,
                        elevation,
                        normal,
                    })
                })
                .collect();
            Ok((carriers, Body::Fill(FillBody { pattern, rgba })))
        }
    }
}
