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

use super::components::{FillBody, TextBody};
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
}

/// Why [`map`] returns nothing for a text with no line inside the clip it was asked to keep to.
pub(crate) const CLIPPED: &str = "it is outside the sheet or viewport it is drawn through";

/// The hidden entities to write for `annotation`, in block space like every other leaf, and the
/// stream record that goes with them, or why it cannot be mapped. A text is laid out in its own
/// plane, one anchor and record per printed line, and `keep` says whether a line starting at a
/// point of the drawing (after the leaf's map) is wanted, so a text seen through a viewport or on
/// a sheet is cut to it line by line.
pub(crate) fn map(
    leaf: &FlatEntity,
    annotation: Annotation,
    keep: &dyn Fn([f64; 2]) -> bool,
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
