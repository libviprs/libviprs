//! The first DWG provider: acadrust, behind the `dwg-acadrust` feature.
//!
//! This is the one place in libviprs that names acadrust. Everything that leaves it
//! is a [`DwgProvider`], an [`EntityCursor`] or one of the boundary types of
//! [`provider`](crate::pvec::provider): no `acadrust` type appears in a public
//! signature, and `tests/pvec_provider_boundary.rs` checks that.
//!
//! ```text
//! acadrust::CadDocument
//!   layers        ->  LayerState      (off, frozen, locked, as the table stores them)
//!   Layout objects ->  LayoutState     (model first, then paper layouts in tab order)
//!   block record's entities -> EntityCursor -> RawEntity   (storage order, not expanded)
//!   entity invisible flag  ->  EntityFacts      (own flag, and the provider's own answer)
//!   Layout plot settings   ->  LayoutPlot       (paper, units, scale, margins, CTB name, flags)
//!   paper-space viewports  ->  ViewportEntity   (visibility, view, plot flags, clip by kind, frozen layers)
//! ```
//!
//! # Where the facts come from
//!
//! Every facet is read from acadrust's object model directly, which is the
//! [`FacetSource::DecoderDirect`] row of [`ProviderCaps`]. The existing
//! [`CadDecoder`](crate::cad::CadDecoder) adapter carries none of them (no layer
//! name, no block name, no insert chain), and it is untouched.
//!
//! # What this provider does not do
//!
//! It does not expand INSERTs, apply layer visibility, or flatten a curve. A spline
//! with no control points (a fit-point spline) is [`RawGeometry::Unmapped`], and so
//! is every class with no mapping yet: counted, with the class name, never dropped.
//! A reader panic on a hostile file is caught and reported as
//! [`ProviderError::Open`], because the open failure is the thing Gate 0 counts.
//!
//! # Supported DWG versions
//!
//! [`probe`](DwgProvider::probe) accepts the six byte stamps the existing decoder
//! accepts, [`SUPPORTED_DWG_VERSIONS`] (AC1014 through AC1032), because those are the
//! versions the corpus of `docs/spikes/acadrust-gate0.md` was run over.
//!
//! # DXF
//!
//! A DXF opens too (#1561). acadrust's DXF reader yields the same `CadDocument` its
//! DWG reader does, so everything after the open is the DWG path unchanged. The sniff
//! looks at the first kilobyte: the binary form's 22 byte sentinel, or the text form's
//! first group pair being `0` and `SECTION`, after any `999` comment pairs. It takes what
//! acadrust's text reader takes and no more: the code line may carry spaces, the value
//! line only its line end, and a byte order mark or a blank line before the first pair
//! is not a DXF (the reader could not open it). No DXF version is gated here: the
//! version sits in the header section, not at a fixed offset, and the reader refuses
//! what it cannot read with an [`Open`](ProviderError::Open) error.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};

use acadrust as acr;
use acr::objects::ObjectType;
use acr::{DwgReader, DxfReader, EntityType};

use crate::cad::{CadSource, Fill, ItemHandle, decode_escapes, display_text};
use crate::pvec::provider::{
    Annotation, ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityFacts,
    EntityScope, EntityStyle, Facet, FacetSource, InstancePath, LayerState, LayoutId, LayoutKind,
    LayoutPlot, LayoutState, Point3, ProvenanceStub, ProviderCaps, ProviderError, RawEntity,
    RawGeometry, RawVertex, SourceColour, SourceFact, SourceLinetype, SourceLineweight,
    SourceTransform, TextStyleRef, ViewportClip, ViewportEntity, XrefAttachment,
};
use crate::pvec::units::SourceUnit;

/// What this provider calls itself in [`ProviderError`].
const NAME: &str = "acadrust";

/// The DWG version stamps accepted: R14, 2000, 2004, 2007, 2010, 2013 and 2018.
pub const SUPPORTED_DWG_VERSIONS: [&[u8; 6]; 7] = [
    b"AC1014", b"AC1015", b"AC1018", b"AC1021", b"AC1024", b"AC1027", b"AC1032",
];

fn stamp_supported(head: &[u8]) -> bool {
    head.len() >= 6 && SUPPORTED_DWG_VERSIONS.iter().any(|v| head[..6] == v[..])
}

/// Whether `head` starts a DXF, binary or text (see the module's DXF section).
fn dxf_sniffed(head: &[u8]) -> bool {
    if head.starts_with(b"AutoCAD Binary DXF\r\n\x1a\x00") {
        return true;
    }
    // A pair needs the line after its value to have begun, so a value cut off at the end
    // of the head (`SECT`) is never taken for a whole one. The reader trims a code line
    // and only drops the `\r` of a value line, so the sniff does the same.
    let mut lines = head.split(|b| *b == b'\n').peekable();
    loop {
        let code = lines.next().map(<[u8]>::trim_ascii);
        let value = lines.next().map(|v| v.strip_suffix(b"\r").unwrap_or(v));
        match (code, value, lines.peek()) {
            (Some(b"999"), Some(_), Some(_)) => {}
            (Some(b"0"), Some(b"SECTION"), Some(_)) => return true,
            _ => return false,
        }
    }
}

/// The first kilobyte: six bytes are the DWG stamp, and the rest leaves a DXF's
/// leading comments room before its first `SECTION`.
fn read_head(source: CadSource<'_>) -> Result<Vec<u8>, ProviderError> {
    match source {
        CadSource::Bytes(b) => Ok(b[..b.len().min(1024)].to_vec()),
        CadSource::Path(p) => {
            let mut head = Vec::with_capacity(1024);
            let file = std::fs::File::open(p).map_err(|e| ProviderError::Open {
                provider: NAME,
                detail: format!("{}: {e}", p.display()),
            })?;
            file.take(1024)
                .read_to_end(&mut head)
                .map_err(|e| ProviderError::Open {
                    provider: NAME,
                    detail: e.to_string(),
                })?;
            Ok(head)
        }
    }
}

/// A TEXT or MTEXT entity's own fields, in the coordinates of the block it sits in.
///
/// `RawGeometry` has no text variant (a text entity comes through the cursor as
/// `Unmapped`), so the `cad::CadDecoder` over this provider asks for it by handle.
#[derive(Debug, Clone, PartialEq)]
pub struct TextFact {
    /// The insertion point.
    pub position: Point3,
    /// The text height.
    pub height: f64,
    /// The rotation in radians.
    pub rotation: f64,
    /// The stored string, with its escapes and (for MTEXT) its formatting codes.
    pub content: String,
    /// True for MTEXT, whose content carries formatting codes.
    pub multiline: bool,
    /// The extrusion normal.
    pub normal: Point3,
    /// The MTEXT attachment point as its DXF number (71), 1 to 9, or [`None`] for anything that
    /// is not MTEXT.
    pub attachment: Option<u8>,
}

/// What a HATCH, LEADER, MULTILEADER or TOLERANCE draws, read by handle for the
/// `cad::CadDecoder` over this provider. Like [`TextFact`], these classes come through the cursor
/// as `Unmapped`, so the decoder asks for their geometry here.
///
/// Hatch rings are in the hatch's own plane at `elevation`; strokes and texts are as stored.
#[derive(Debug, Clone, Default)]
pub struct DrawnFact {
    /// One closed ring per hatch boundary loop, with the bulge of the span leaving each vertex. A
    /// loop of straight and circular spans is exact; an elliptic or spline edge is sampled into
    /// short straight spans.
    pub rings: Vec<Vec<RawVertex>>,
    /// The plane height of the hatch rings.
    pub elevation: f64,
    /// The hatch's extrusion normal.
    pub normal: Point3,
    /// How each ring is drawn, one per ring: filled (a solid HATCH or a 2D SOLID) or an outline.
    pub fills: Vec<Fill>,
    /// A HATCH that is a pattern, whose lines this reading does not draw.
    pub pattern_only: bool,
    /// Open runs of points: a leader's lines.
    pub strokes: Vec<Vec<Point3>>,
    /// Text the entity shows.
    pub texts: Vec<TextFact>,
    /// What the entity has that this reading leaves out, each in a sentence.
    pub skipped: Vec<String>,
}

/// A circular boundary edge as its start point, end point and the bulge of the span, or
/// [`None`] for a degenerate or full-circle edge.
fn arc_span(e: &acr::entities::CircularArcEdge) -> Option<([f64; 2], [f64; 2], f64)> {
    use std::f64::consts::TAU;
    let (a0, a1) = if e.counter_clockwise {
        (e.start_angle, e.end_angle)
    } else {
        (-e.start_angle, -e.end_angle)
    };
    let sweep = if e.counter_clockwise {
        (a1 - a0).rem_euclid(TAU)
    } else {
        -((a0 - a1).rem_euclid(TAU))
    };
    if e.radius.is_nan() || e.radius <= 0.0 || sweep == 0.0 {
        return None;
    }
    let at = |a: f64| {
        [
            e.center.x + e.radius * a.cos(),
            e.center.y + e.radius * a.sin(),
        ]
    };
    Some((at(a0), at(a1), (sweep / 4.0).tan()))
}

/// An elliptic boundary edge as points along it, from its start to its end, or [`None`] for a
/// degenerate one. The parameters follow the sign convention of [`arc_span`]: a clockwise edge runs
/// from the negated parameters.
fn ellipse_points(e: &acr::entities::EllipticArcEdge) -> Option<Vec<[f64; 2]>> {
    use std::f64::consts::TAU;
    let (mx, my) = (e.major_axis_endpoint.x, e.major_axis_endpoint.y);
    let ratio = e.minor_axis_ratio;
    if !(ratio > 0.0 && ratio.is_finite() && mx.is_finite() && my.is_finite())
        || (mx == 0.0 && my == 0.0)
    {
        return None;
    }
    let (t0, t1) = if e.counter_clockwise {
        (e.start_angle, e.end_angle)
    } else {
        (-e.start_angle, -e.end_angle)
    };
    // A whole ellipse is stored as a full turn, which `rem_euclid` would read as no sweep.
    let sweep = if (t1 - t0).abs() >= TAU - 1e-9 {
        TAU.copysign(t1 - t0)
    } else if e.counter_clockwise {
        (t1 - t0).rem_euclid(TAU)
    } else {
        -((t0 - t1).rem_euclid(TAU))
    };
    if !sweep.is_finite() || sweep == 0.0 {
        return None;
    }
    // A step of at most a 96th of a turn keeps a chord within 0.06% of the longer radius.
    let steps = ((sweep.abs() / TAU * 96.0).ceil() as usize).clamp(8, 192);
    let (nx, ny) = (-my * ratio, mx * ratio);
    Some(
        (0..=steps)
            .map(|i| {
                let (sin, cos) = (t0 + sweep * i as f64 / steps as f64).sin_cos();
                [
                    e.center.x + mx * cos + nx * sin,
                    e.center.y + my * cos + ny * sin,
                ]
            })
            .collect(),
    )
}

/// A spline boundary edge as points along it, from its start to its end, or [`None`] when its
/// control points, degree and knots do not describe a curve. Without control points the fit points
/// are joined as given.
fn spline_points(e: &acr::entities::SplineEdge) -> Option<Vec<[f64; 2]>> {
    if e.control_points.is_empty() {
        let fit: Vec<[f64; 2]> = e.fit_points.iter().map(|p| [p.x, p.y]).collect();
        return (fit.len() >= 2 && fit.iter().flatten().all(|v| v.is_finite())).then_some(fit);
    }
    let degree = usize::try_from(e.degree).ok().filter(|d| *d >= 1)?;
    let n = e.control_points.len();
    let knots = &e.knots;
    if n <= degree
        || knots.len() != n + degree + 1
        || !knots.iter().all(|k| k.is_finite())
        || knots.windows(2).any(|w| w[1] < w[0])
    {
        return None;
    }
    let (lo, hi) = (knots[degree], knots[n]);
    if hi <= lo {
        return None;
    }
    // Homogeneous control points: a weight is only read when the spline says it is rational.
    let mut homogeneous = Vec::with_capacity(n);
    for p in &e.control_points {
        let w = if e.rational { p.z } else { 1.0 };
        if !(p.x.is_finite() && p.y.is_finite() && w > 0.0 && w.is_finite()) {
            return None;
        }
        homogeneous.push([p.x * w, p.y * w, w]);
    }
    let steps = ((n - degree) * 16).clamp(16, 512);
    Some(
        (0..=steps)
            .map(|i| {
                let u = if i == steps {
                    hi
                } else {
                    lo + (hi - lo) * i as f64 / steps as f64
                };
                // The knot span holding `u`, the last one at the end of the domain (de Boor).
                let mut k = degree;
                while k < n - 1 && u >= knots[k + 1] {
                    k += 1;
                }
                let mut d: Vec<[f64; 3]> =
                    (0..=degree).map(|j| homogeneous[j + k - degree]).collect();
                for r in 1..=degree {
                    for j in (r..=degree).rev() {
                        let at = j + k - degree;
                        let gap = knots[at + degree + 1 - r] - knots[at];
                        let alpha = if gap > 0.0 {
                            (u - knots[at]) / gap
                        } else {
                            0.0
                        };
                        let (before, here) = (d[j - 1], d[j]);
                        d[j] = std::array::from_fn(|c| (1.0 - alpha) * before[c] + alpha * here[c]);
                    }
                }
                let [x, y, w] = d[degree];
                [x / w, y / w]
            })
            .collect(),
    )
}

fn hatch_rings(h: &acr::entities::Hatch, out: &mut DrawnFact) {
    use acr::entities::BoundaryEdge;
    out.elevation = h.elevation;
    out.normal = p3(h.normal);
    out.pattern_only = !h.is_solid;
    for (n, path) in h.paths.iter().enumerate() {
        let mut ring: Vec<RawVertex> = Vec::new();
        let mut end: Option<[f64; 2]> = None;
        let mut whole = true;
        for edge in &path.edges {
            // The points of a sampled edge, after its first and before its last.
            let mut between: Vec<[f64; 2]> = Vec::new();
            let mut sampled = |points: Option<Vec<[f64; 2]>>| {
                let points = points?;
                let (first, last) = (*points.first()?, *points.last()?);
                between = points[1..points.len() - 1].to_vec();
                Some((first, last, 0.0))
            };
            let span = match edge {
                BoundaryEdge::Polyline(pl) => {
                    ring.extend(pl.vertices.iter().map(|v| RawVertex {
                        x: v.x,
                        y: v.y,
                        bulge: v.z,
                    }));
                    end = pl.vertices.last().map(|v| [v.x, v.y]);
                    continue;
                }
                BoundaryEdge::Line(l) => Some(([l.start.x, l.start.y], [l.end.x, l.end.y], 0.0)),
                // A whole circle is stored as a full turn, which has no span: it is two half turns.
                BoundaryEdge::CircularArc(a)
                    if a.radius > 0.0
                        && a.radius.is_finite()
                        && (a.end_angle - a.start_angle).abs() >= std::f64::consts::TAU - 1e-9 =>
                {
                    let bulge = if a.counter_clockwise { 1.0 } else { -1.0 };
                    let at = |angle: f64| {
                        let (sin, cos) = angle.sin_cos();
                        [a.center.x + a.radius * cos, a.center.y + a.radius * sin]
                    };
                    let start = if a.counter_clockwise {
                        a.start_angle
                    } else {
                        -a.start_angle
                    };
                    let [x, y] = at(start);
                    let [x2, y2] = at(start + std::f64::consts::PI);
                    ring.push(RawVertex { x, y, bulge });
                    ring.push(RawVertex {
                        x: x2,
                        y: y2,
                        bulge,
                    });
                    end = Some([x, y]);
                    continue;
                }
                BoundaryEdge::CircularArc(a) => arc_span(a),
                BoundaryEdge::EllipticArc(a) => sampled(ellipse_points(a)),
                BoundaryEdge::Spline(sp) => sampled(spline_points(sp)),
            };
            let Some((from, to, bulge)) = span else {
                whole = false;
                break;
            };
            // Edges run head to tail, so a gap means this reading would draw a wrong shape.
            let tolerance = 1e-6 * (1.0 + from[0].abs().max(from[1].abs()));
            if end.is_some_and(|e| {
                (e[0] - from[0]).abs() > tolerance || (e[1] - from[1]).abs() > tolerance
            }) {
                whole = false;
                break;
            }
            ring.push(RawVertex {
                x: from[0],
                y: from[1],
                bulge,
            });
            ring.extend(between.iter().map(|p| RawVertex {
                x: p[0],
                y: p[1],
                bulge: 0.0,
            }));
            end = Some(to);
        }
        if whole && ring.len() >= 2 {
            out.rings.push(ring);
        } else {
            out.skipped.push(format!(
                "boundary loop {n} has an edge that is not a curve, or edges that do not join"
            ));
        }
    }
    out.fills = hatch_fills(h, out.rings.len(), out.skipped.is_empty());
}

/// How each ring of a solid HATCH is drawn.
///
/// A loop that could not be read leaves a hole in the picture where the fill would have been, so
/// a hatch with one is drawn as outlines, and the skipped loop is already reported. Otherwise the
/// hatch style decides. "Normal" and "outer" fill the loops together by even-odd, so an island is
/// a hole: outer is AutoCAD's default island detection, and an island in it is not filled.
/// "Ignore" fills each loop by itself, so an island is filled over.
fn hatch_fills(h: &acr::entities::Hatch, rings: usize, all_read: bool) -> Vec<Fill> {
    use acr::entities::HatchStyleType;
    if !h.is_solid || !all_read {
        return vec![Fill::Outline; rings];
    }
    match h.style {
        HatchStyleType::Normal | HatchStyleType::Outer => {
            (0..rings).map(|i| Fill::Solid { opens: i == 0 }).collect()
        }
        HatchStyleType::Ignore => vec![Fill::Solid { opens: true }; rings],
    }
}

/// The angle of `along`, or `fallback` when it has no direction in the plane.
fn angle_of(along: acr::Vector3, fallback: f64) -> f64 {
    if along.x == 0.0 && along.y == 0.0 {
        fallback
    } else {
        along.y.atan2(along.x)
    }
}

/// The first provider.
pub struct AcadrustProvider {
    doc: acr::CadDocument,
    caps: ProviderCaps,
    layouts: Vec<LayoutState>,
    /// The plot settings of each of `layouts`, by position; `None` for a layout named after its
    /// block record, which has no Layout object to hold them.
    plots: Vec<Option<LayoutPlot>>,
}

impl AcadrustProvider {
    /// Every block definition's name, in the block table's order.
    ///
    /// Not part of [`DwgProvider`]: a consumer follows INSERTs by name and has no
    /// need of the list. It is here for a report that wants to say how many blocks
    /// a drawing exposes.
    #[must_use]
    pub fn block_names(&self) -> Vec<String> {
        self.doc
            .block_records
            .iter()
            .map(|b| b.name.clone())
            .collect()
    }

    /// The unit of the drawing's coordinates, read from the header's `$INSUNITS`.
    ///
    /// Header only: no inference from the coordinates and no default. The value is
    /// handed on as the drawing stores it, and the coordinates the cursor yields
    /// are not scaled by it.
    #[must_use]
    pub fn source_unit(&self) -> SourceUnit {
        SourceUnit::from_insunits(Some(self.doc.header.insertion_units))
    }

    /// The model space extents the header stores, as `[min_x, min_y, max_x, max_y]`. A drawing
    /// that never set them reads back the inverted box, which is not a usable extent.
    #[must_use]
    pub fn model_space_extents(&self) -> [f64; 4] {
        let h = &self.doc.header;
        [
            h.model_space_extents_min.x,
            h.model_space_extents_min.y,
            h.model_space_extents_max.x,
            h.model_space_extents_max.y,
        ]
    }

    /// The text fields of the TEXT or MTEXT entity with this handle, or [`None`] for any other
    /// entity.
    #[must_use]
    pub fn text_fact(&self, handle: ItemHandle) -> Option<TextFact> {
        match self.doc.get_entity(acr::Handle::new(handle.get()))? {
            EntityType::Text(t) => Some(TextFact {
                position: p3(t.insertion_point),
                height: t.height,
                rotation: t.rotation,
                content: t.value.clone(),
                multiline: false,
                normal: p3(t.normal),
                attachment: None,
            }),
            EntityType::MText(m) => Some(TextFact {
                position: p3(m.insertion_point),
                height: m.height,
                rotation: m.rotation,
                content: m.value.clone(),
                multiline: true,
                normal: p3(m.normal),
                attachment: Some(m.attachment_point as u8),
            }),
            _ => None,
        }
    }

    /// What the HATCH, LEADER, MULTILEADER or TOLERANCE with this handle draws, or [`None`] for
    /// any other entity.
    #[must_use]
    pub fn drawn_fact(&self, handle: ItemHandle) -> Option<DrawnFact> {
        let mut out = DrawnFact::default();
        match self.doc.get_entity(acr::Handle::new(handle.get()))? {
            EntityType::Hatch(h) => hatch_rings(h, &mut out),
            EntityType::Leader(l) => out
                .strokes
                .push(l.vertices.iter().copied().map(p3).collect()),
            EntityType::MultiLeader(m) => {
                let c = &m.context;
                // A leader line is stored as its points from the arrow tip; it ends at the root's
                // connection point, where the text or block attaches.
                for root in &c.leader_roots {
                    for line in &root.lines {
                        let mut run: Vec<Point3> = line.points.iter().copied().map(p3).collect();
                        run.push(p3(root.connection_point));
                        out.strokes.push(run);
                    }
                }
                if c.has_text_contents && !c.text_string.is_empty() {
                    out.texts.push(TextFact {
                        position: p3(c.text_location),
                        height: c.text_height,
                        rotation: angle_of(c.text_direction, c.text_rotation),
                        content: c.text_string.clone(),
                        multiline: true,
                        normal: p3(c.text_normal),
                        attachment: None,
                    });
                }
                if c.has_block_contents {
                    out.skipped.push("its block content is not drawn".into());
                }
            }
            EntityType::Tolerance(t) => out.texts.push(TextFact {
                position: p3(t.insertion_point),
                height: t.text_height,
                rotation: angle_of(t.direction, 0.0),
                content: t.text.clone(),
                multiline: true,
                normal: p3(t.normal),
                attachment: None,
            }),
            // A 3D polyline is its vertices joined in order, seen from above like every other
            // 3D line here.
            EntityType::Polyline(p) => {
                let mut run: Vec<Point3> = p.vertices.iter().map(|v| p3(v.location)).collect();
                if p.is_closed() {
                    run.extend(run.first().copied());
                }
                out.strokes.push(run);
            }
            EntityType::Polyline3D(p) => {
                let mut run: Vec<Point3> = p.vertices.iter().map(|v| p3(v.position)).collect();
                if p.is_closed() {
                    run.extend(run.first().copied());
                }
                out.strokes.push(run);
            }
            // A 3DFACE is the edges its flags leave visible. A triangle repeats a corner, which
            // makes one edge a point, and a point is not drawn.
            EntityType::Face3D(f) => {
                let c = [
                    f.first_corner,
                    f.second_corner,
                    f.third_corner,
                    f.fourth_corner,
                ];
                let hidden = [
                    f.invisible_edges.is_first_invisible(),
                    f.invisible_edges.is_second_invisible(),
                    f.invisible_edges.is_third_invisible(),
                    f.invisible_edges.is_fourth_invisible(),
                ];
                for i in 0..4 {
                    let (a, b) = (c[i], c[(i + 1) % 4]);
                    if !hidden[i] && a != b {
                        out.strokes.push(vec![p3(a), p3(b)]);
                    }
                }
            }
            EntityType::Solid(s) => {
                // The stored order of a quadrilateral is first, second, third, fourth, and it is
                // drawn first, second, fourth, third.
                let corners = s.boundary_corners();
                out.elevation = corners.first().map_or(0.0, |c| c.z);
                out.normal = p3(s.normal);
                out.rings.push(
                    corners
                        .iter()
                        .map(|c| RawVertex {
                            x: c.x,
                            y: c.y,
                            bulge: 0.0,
                        })
                        .collect(),
                );
                out.fills.push(Fill::Solid { opens: true });
            }
            _ => return None,
        }
        Some(out)
    }

    /// The text of the visible attributes of the INSERT with this handle, in the coordinates the
    /// INSERT sits in.
    #[must_use]
    pub fn attribute_texts(&self, handle: ItemHandle) -> Vec<TextFact> {
        let Some(EntityType::Insert(i)) = self.doc.get_entity(acr::Handle::new(handle.get()))
        else {
            return Vec::new();
        };
        i.attributes
            .iter()
            .filter(|a| a.is_visible())
            .map(|a| TextFact {
                position: p3(a.insertion_point),
                height: a.height,
                rotation: a.rotation,
                content: a.value.clone(),
                multiline: false,
                normal: p3(a.normal),
                attachment: None,
            })
            .collect()
    }

    /// The anonymous block a DIMENSION entity draws itself from, which holds its lines and text.
    #[must_use]
    pub fn dimension_block(&self, handle: ItemHandle) -> Option<String> {
        match self.doc.get_entity(acr::Handle::new(handle.get()))? {
            EntityType::Dimension(d) if !d.base().block_name.is_empty() => {
                Some(d.base().block_name.clone())
            }
            _ => None,
        }
    }

    /// The colour a fill is drawn in, as red, green, blue and alpha: its own colour, or its layer's
    /// for BYLAYER. A fill whose colour is BYBLOCK (the INSERT's, which the provider cannot see)
    /// or has none is black.
    fn fill_rgba(&self, common: &acr::entities::EntityCommon) -> u32 {
        let layer_colour = || {
            let found = self.doc.layers.get(&common.layer).or_else(|| {
                self.doc
                    .layers
                    .iter()
                    .find(|l| l.name.eq_ignore_ascii_case(&common.layer))
            });
            found.map(|l| l.color)
        };
        let colour = match common.color {
            acr::Color::ByLayer => layer_colour().unwrap_or(acr::Color::None),
            c => c,
        };
        let (r, g, b) = colour.rgb().unwrap_or((0, 0, 0));
        u32::from_be_bytes([r, g, b, 255])
    }

    fn capabilities() -> ProviderCaps {
        Facet::ALL.iter().fold(ProviderCaps::new(), |caps, f| {
            caps.with(*f, Availability::Reported, FacetSource::DecoderDirect)
        })
    }

    fn collect_layouts(doc: &acr::CadDocument) -> Vec<(LayoutState, Option<LayoutPlot>)> {
        let block_of = |handle: acr::Handle| {
            doc.block_records
                .iter()
                .find(|b| b.handle == handle)
                .map(|b| b.name.clone())
        };
        let mut found: Vec<(i16, u64, String, String, Option<LayoutPlot>)> = Vec::new();
        for o in doc.objects.values() {
            if let ObjectType::Layout(l) = o
                && let Some(block) = block_of(l.block_record)
            {
                let plot = LayoutPlot {
                    paper_size: l.paper_size.clone(),
                    paper_width: l.paper_width,
                    paper_height: l.paper_height,
                    paper_units: l.plot_paper_units,
                    plot_type: l.plot_type,
                    window: [
                        l.plot_window_min_x,
                        l.plot_window_min_y,
                        l.plot_window_max_x,
                        l.plot_window_max_y,
                    ],
                    scale_numerator: l.plot_scale_numerator,
                    scale_denominator: l.plot_scale_denominator,
                    rotation: l.plot_rotation,
                    margins: [
                        l.plot_margin_left,
                        l.plot_margin_bottom,
                        l.plot_margin_right,
                        l.plot_margin_top,
                    ],
                    origin: [l.plot_origin_x, l.plot_origin_y],
                    style_sheet: l.plot_style_sheet.clone(),
                    print_lineweights: l.plot_flags.print_lineweights,
                    scale_lineweights: l.plot_flags.scale_lineweights,
                    plot_plot_styles: l.plot_flags.plot_plot_styles,
                    plot_viewport_borders: l.plot_flags.plot_viewport_borders,
                    draw_viewports_first: l.plot_flags.draw_viewports_first,
                };
                found.push((
                    l.tab_order,
                    l.handle.value(),
                    l.name.clone(),
                    block,
                    Some(plot),
                ));
            }
        }
        if found.is_empty() {
            // Drawings before R2000 have no Layout objects: the block table is all
            // there is, so name the layouts after their block records.
            for b in doc.block_records.iter() {
                let lower = b.name.to_ascii_lowercase();
                if lower.starts_with("*model_space") || lower.starts_with("*paper_space") {
                    let name = if lower.starts_with("*model_space") {
                        "Model".to_string()
                    } else {
                        b.name.clone()
                    };
                    found.push((0, b.handle.value(), name, b.name.clone(), None));
                }
            }
        }
        // The objects live in a hash map, so the order is made here, not inherited.
        let is_model = |block: &str| block.eq_ignore_ascii_case("*Model_Space");
        found.sort_by_key(|a| (!is_model(&a.3), a.0, a.1));
        found
            .into_iter()
            .enumerate()
            .map(|(i, (_, _, name, block, plot))| {
                let state = LayoutState {
                    id: LayoutId(u32::try_from(i).unwrap_or(u32::MAX)),
                    name,
                    kind: if is_model(&block) {
                        LayoutKind::Model
                    } else {
                        LayoutKind::Paper
                    },
                    block_name: block,
                };
                (state, plot)
            })
            .collect()
    }
}

impl DwgProvider for AcadrustProvider {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        let head = read_head(source)?;
        Ok(stamp_supported(&head) || dxf_sniffed(&head))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let head = read_head(source)?;
        let dwg = stamp_supported(&head);
        if !dwg && !dxf_sniffed(&head) {
            return Err(ProviderError::Unrecognised {
                provider: NAME,
                detail: format!(
                    "neither a supported DWG version stamp nor a DXF header: {:?}",
                    String::from_utf8_lossy(&head[..head.len().min(6)])
                ),
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| match (source, dwg) {
            (CadSource::Path(p), true) => DwgReader::from_file(p).and_then(|mut r| r.read()),
            (CadSource::Bytes(b), true) => DwgReader::from_stream(Cursor::new(b)).read(),
            (CadSource::Path(p), false) => DxfReader::from_file(p).and_then(DxfReader::read),
            // The DXF reader wants an owned stream, so the bytes are copied once here.
            (CadSource::Bytes(b), false) => {
                DxfReader::from_reader(Cursor::new(b.to_vec())).and_then(DxfReader::read)
            }
        }));
        let doc = match result {
            Ok(Ok(doc)) => doc,
            Ok(Err(e)) => {
                return Err(ProviderError::Open {
                    provider: NAME,
                    detail: e.to_string(),
                });
            }
            Err(_) => {
                return Err(ProviderError::Open {
                    provider: NAME,
                    detail: "the reader panicked".into(),
                });
            }
        };
        let (layouts, plots) = Self::collect_layouts(&doc).into_iter().unzip();
        Ok(Self {
            doc,
            caps: Self::capabilities(),
            layouts,
            plots,
        })
    }

    fn name(&self) -> &'static str {
        NAME
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(self.layouts.clone())
    }

    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        let plot = self
            .plots
            .get(layout.0 as usize)
            .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {}", layout.0)))?;
        Ok(plot.clone().map_or(SourceFact::Absent, SourceFact::Present))
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(self
            .doc
            .layers
            .iter()
            .map(|l| LayerState {
                name: l.name.clone(),
                handle: item_handle(l.handle),
                off: l.is_off(),
                frozen: l.is_frozen(),
                locked: l.is_locked(),
                plottable: l.is_plottable,
                color: fact_colour(l.color).present().copied(),
            })
            .collect())
    }

    fn entity_facts(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        let (record, _) = self.resolve(scope)?;
        let mut out = BTreeMap::new();
        for handle in &record.entity_handles {
            let Some(e) = self.doc.get_entity(*handle) else {
                continue;
            };
            if matches!(e, EntityType::Block(_) | EntityType::BlockEnd(_)) {
                continue;
            }
            let Some(key) = item_handle(*handle) else {
                continue;
            };
            let common = e.common();
            // A drawing stores one layer per entity, so there is no second one to report.
            // That is a fact about the drawing, not a gap in the reader.
            let layer_hides = self
                .doc
                .layers
                .get(&common.layer)
                .is_some_and(|l| l.is_off() || l.is_frozen());
            out.insert(
                key,
                EntityFacts {
                    visible: SourceFact::Present(!common.invisible),
                    layer_override: SourceFact::Absent,
                    provider_visible: SourceFact::Present(!common.invisible && !layer_hides),
                },
            );
        }
        Ok(out)
    }

    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        let mut out = Vec::new();
        for layout in self.layouts.iter().filter(|l| l.kind == LayoutKind::Paper) {
            let Some(record) = self.doc.block_records.get(&layout.block_name) else {
                continue;
            };
            for handle in &record.entity_handles {
                let (Some(EntityType::Viewport(v)), Some(key)) =
                    (self.doc.get_entity(*handle), item_handle(*handle))
                else {
                    continue;
                };
                out.push(ViewportEntity {
                    handle: key,
                    layout: layout.id,
                    id: v.id,
                    visible: v.status.is_on && !v.common.invisible,
                    clip: self.viewport_clip(v),
                    layer: v.common.layer.clone(),
                    center: [v.center.x, v.center.y],
                    width: v.width,
                    height: v.height,
                    view_center: [v.view_center.x, v.view_center.y],
                    view_height: v.view_height,
                    view_target: p3(v.view_target),
                    view_direction: p3(v.view_direction),
                    twist: v.twist_angle,
                    perspective: v.status.perspective,
                    hide_plot: v.status.hide_plot,
                    non_rectangular_clipping: v.status.non_rectangular_clipping,
                    frozen_layers: v
                        .frozen_layers
                        .iter()
                        .filter_map(|h| {
                            self.doc
                                .layers
                                .iter()
                                .find(|l| l.handle == *h)
                                .map(|l| l.name.clone())
                        })
                        .collect(),
                });
            }
        }
        Ok(out)
    }

    fn xref_blocks(&self) -> Result<Vec<XrefAttachment>, ProviderError> {
        Ok(self
            .doc
            .block_records
            .iter()
            .filter(|b| b.flags.is_xref || b.flags.is_xref_overlay)
            .map(|b| XrefAttachment {
                block_name: b.name.clone(),
                path: b.xref_path.clone(),
                overlay: b.flags.is_xref_overlay,
                unloaded: b.flags.is_xref_unloaded,
            })
            .collect())
    }

    fn text_anchor(&self, handle: ItemHandle) -> Option<(String, Point3)> {
        self.text_fact(handle).map(|t| (t.content, t.position))
    }

    fn inserted_attribute(&self, attdef: ItemHandle, insert: ItemHandle) -> Option<Annotation> {
        let EntityType::AttributeDefinition(a) =
            self.doc.get_entity(acr::Handle::new(attdef.get()))?
        else {
            return None;
        };
        // A constant definition carries its own value into every INSERT. Any other shows what the
        // INSERT's ATTRIB with that tag holds, and nothing when there is none: the tag is the
        // block editor's placeholder, not something AutoCAD draws in a drawing.
        let content = if a.flags.constant {
            (!a.flags.invisible).then(|| a.default_value.clone())?
        } else {
            let EntityType::Insert(i) = self.doc.get_entity(acr::Handle::new(insert.get()))? else {
                return None;
            };
            i.attributes
                .iter()
                .find(|v| v.tag.eq_ignore_ascii_case(&a.tag))
                .filter(|v| v.is_visible())?
                .value
                .clone()
        };
        text(
            a.insertion_point,
            a.height,
            a.rotation,
            a.normal,
            decode_escapes(&content),
        )
    }

    fn annotation(&self, handle: ItemHandle) -> Option<Annotation> {
        let entity = self.doc.get_entity(acr::Handle::new(handle.get()))?;
        match entity {
            EntityType::Text(t) => text(
                t.insertion_point,
                t.height,
                t.rotation,
                t.normal,
                decode_escapes(&t.value),
            ),
            EntityType::MText(m) => text(
                m.insertion_point,
                m.height,
                m.rotation,
                m.normal,
                decode_escapes(&display_text(&m.value).text),
            ),
            // A definition outside any INSERT shows its tag, and an invisible one never. Inside an
            // INSERT it goes through `inserted_attribute`.
            EntityType::AttributeDefinition(a) if !a.flags.invisible => text(
                a.insertion_point,
                a.height,
                a.rotation,
                a.normal,
                decode_escapes(if a.tag.is_empty() {
                    &a.default_value
                } else {
                    &a.tag
                }),
            ),
            EntityType::Hatch(h) => {
                let mut drawn = DrawnFact::default();
                hatch_rings(h, &mut drawn);
                Some(Annotation::Fill {
                    rings: drawn.rings,
                    elevation: drawn.elevation,
                    normal: drawn.normal,
                    pattern: (!h.is_solid).then(|| {
                        let scale = h.pattern_scale;
                        (
                            h.pattern.name.clone(),
                            h.pattern_angle,
                            if scale > 0.0 && scale.is_finite() {
                                scale
                            } else {
                                1.0
                            },
                        )
                    }),
                    rgba: self.fill_rgba(&h.common),
                    unread_loops: u32::try_from(drawn.skipped.len()).unwrap_or(u32::MAX),
                })
            }
            EntityType::Solid(s) => {
                // A triangle repeats a corner, and a ring with a repeated vertex is not a boundary.
                let mut ring: Vec<RawVertex> = Vec::new();
                for c in s.boundary_corners() {
                    if ring.last().is_none_or(|v| (v.x, v.y) != (c.x, c.y)) {
                        ring.push(RawVertex {
                            x: c.x,
                            y: c.y,
                            bulge: 0.0,
                        });
                    }
                }
                if ring.len() > 1
                    && (ring[0].x, ring[0].y) == (ring[ring.len() - 1].x, ring[ring.len() - 1].y)
                {
                    ring.pop();
                }
                Some(Annotation::Fill {
                    elevation: s.boundary_corners().first().map_or(0.0, |c| c.z),
                    rings: vec![ring],
                    normal: p3(s.normal),
                    pattern: None,
                    rgba: self.fill_rgba(&s.common),
                    unread_loops: 0,
                })
            }
            _ => None,
        }
    }

    fn text_styles(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, TextStyleRef>, ProviderError> {
        let (record, _) = self.resolve(scope)?;
        let mut out = BTreeMap::new();
        for handle in &record.entity_handles {
            let (Some(e), Some(key)) = (self.doc.get_entity(*handle), item_handle(*handle)) else {
                continue;
            };
            let name = match e {
                EntityType::Text(t) => &t.style,
                EntityType::MText(m) => &m.style,
                _ => continue,
            };
            let entry = self.doc.text_styles.get(name).or_else(|| {
                self.doc
                    .text_styles
                    .iter()
                    .find(|s| s.name.eq_ignore_ascii_case(name))
            });
            out.insert(
                key,
                TextStyleRef {
                    style_name: name.clone(),
                    font_file: entry.map(|s| s.font_file.clone()).unwrap_or_default(),
                    big_font_file: entry.map(|s| s.big_font_file.clone()).unwrap_or_default(),
                    true_type_font: entry.map(|s| s.true_type_font.clone()).unwrap_or_default(),
                },
            );
        }
        Ok(out)
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let (record, path) = self.resolve(scope)?;
        Ok(Box::new(AcadrustCursor {
            doc: &self.doc,
            handles: record.entity_handles.iter(),
            path,
            base_points: BTreeMap::new(),
        }))
    }
}

/// A run of text as an [`Annotation`], or `None` for one that draws nothing: with no height or no
/// characters, which the stream refuses.
fn text(
    position: acr::Vector3,
    height: f64,
    rotation: f64,
    normal: acr::Vector3,
    content: String,
) -> Option<Annotation> {
    (height > 0.0 && height.is_finite() && !content.trim().is_empty()).then(|| Annotation::Text {
        position: p3(position),
        height,
        rotation,
        content,
        normal: p3(normal),
    })
}

impl AcadrustProvider {
    fn resolve(
        &self,
        scope: &EntityScope,
    ) -> Result<(&acr::tables::BlockRecord, InstancePath), ProviderError> {
        let (block, path) = match scope {
            EntityScope::Layout(id) => {
                let l = self
                    .layouts
                    .get(id.0 as usize)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {}", id.0)))?;
                (l.block_name.clone(), InstancePath::root())
            }
            EntityScope::BlockDefinition { name, reached_via } => {
                (name.clone(), reached_via.clone())
            }
        };
        let record = self
            .doc
            .block_records
            .get(&block)
            .ok_or_else(|| ProviderError::NoSuchScope(format!("block {block}")))?;
        Ok((record, path))
    }

    /// What the viewport is clipped to: the boundary entity it names, by kind, else the
    /// rectangle its centre, width and height describe. A boundary that is not a polyline is
    /// named, never replaced by the rectangle (#1560).
    fn viewport_clip(&self, v: &acr::entities::Viewport) -> ViewportClip {
        if let Some(handle) = item_handle(v.clip_boundary_handle) {
            let vertex = |x: f64, y: f64, bulge: f64| RawVertex { x, y, bulge };
            return match self.doc.get_entity(v.clip_boundary_handle) {
                Some(EntityType::LwPolyline(p)) => ViewportClip::Polygon(
                    p.vertices
                        .iter()
                        .map(|q| vertex(q.location.x, q.location.y, q.bulge))
                        .collect(),
                ),
                Some(EntityType::Polyline2D(p)) => ViewportClip::Polygon(
                    p.vertices
                        .iter()
                        .map(|q| vertex(q.location.x, q.location.y, q.bulge))
                        .collect(),
                ),
                Some(EntityType::Circle(_)) => ViewportClip::Circle(handle),
                Some(EntityType::Ellipse(_)) => ViewportClip::Ellipse(handle),
                Some(EntityType::Spline(_)) => ViewportClip::Spline(handle),
                Some(e) => ViewportClip::Other {
                    handle,
                    class_name: e.as_entity().entity_type().to_string(),
                },
                None => ViewportClip::Other {
                    handle,
                    class_name: "UNKNOWN".into(),
                },
            };
        }
        let (cx, cy) = (v.center.x, v.center.y);
        let (hw, hh) = (v.width / 2.0, v.height / 2.0);
        ViewportClip::Rectangle([
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ])
    }
}

struct AcadrustCursor<'a> {
    doc: &'a acr::CadDocument,
    handles: std::slice::Iter<'a, acr::Handle>,
    path: InstancePath,
    /// Block base points looked up so far, by block name.
    base_points: BTreeMap<String, [f64; 3]>,
}

impl EntityCursor for AcadrustCursor<'_> {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        for handle in self.handles.by_ref() {
            let Some(e) = self.doc.get_entity(*handle) else {
                return Ok(Some(RawEntity {
                    geometry: RawGeometry::Unmapped {
                        reason: "the block lists a handle the document has no entity for".into(),
                    },
                    provenance: ProvenanceStub {
                        handle: item_handle(*handle),
                        class_name: "UNKNOWN".into(),
                        block_path: self.path.clone(),
                        layer: String::new(),
                    },
                    style: EntityStyle::absent(),
                }));
            };
            if matches!(e, EntityType::Block(_) | EntityType::BlockEnd(_)) {
                continue;
            }
            return Ok(Some(self.convert(e)));
        }
        Ok(None)
    }
}

fn item_handle(h: acr::Handle) -> Option<ItemHandle> {
    if h.is_null() {
        None
    } else {
        Some(ItemHandle::new(h.value()))
    }
}

/// A stored colour as a fact. `ByLayer` and `None` carry no colour of their own.
fn fact_colour(c: acr::Color) -> SourceFact<SourceColour> {
    match c {
        acr::Color::Index(i) => SourceFact::Present(SourceColour::Aci(i)),
        acr::Color::Rgb { r, g, b } => SourceFact::Present(SourceColour::True { r, g, b }),
        acr::Color::ByBlock => SourceFact::Present(SourceColour::ByBlock),
        acr::Color::ByLayer | acr::Color::None => SourceFact::Absent,
    }
}

/// A stored lineweight as a fact. `ByLayer` carries no weight of its own, and a
/// negative value that is not one of the three markers is not a weight.
fn fact_lineweight(w: acr::LineWeight) -> SourceFact<SourceLineweight> {
    match w {
        acr::LineWeight::Value(v) => u16::try_from(v).map_or(SourceFact::Absent, |h| {
            SourceFact::Present(SourceLineweight::Hundredths(h))
        }),
        acr::LineWeight::Default => SourceFact::Present(SourceLineweight::Default),
        acr::LineWeight::ByBlock => SourceFact::Present(SourceLineweight::ByBlock),
        acr::LineWeight::ByLayer => SourceFact::Absent,
    }
}

fn p3(v: acr::Vector3) -> Point3 {
    [v.x, v.y, v.z]
}

impl AcadrustCursor<'_> {
    fn base_point(&mut self, block: &str) -> [f64; 3] {
        if let Some(b) = self.base_points.get(block) {
            return *b;
        }
        let b = self.doc.block_records.get(block).map_or([0.0; 3], |r| {
            [r.base_point.x, r.base_point.y, r.base_point.z]
        });
        self.base_points.insert(block.to_string(), b);
        b
    }

    /// The entity's colour and lineweight, BYLAYER resolved through the layer table.
    ///
    /// BYBLOCK is handed on as it is: this cursor sees one block definition and does
    /// not know the INSERT that is drawing it, so a layer whose own value is BYBLOCK
    /// gives `Present(ByBlock)` too, and BYLAYER on layer "0" inside a block gives
    /// `Present(ByLayer)`. A BYLAYER entity on a layer the table does not
    /// have, or on one whose own value is BYLAYER or no colour, has nothing to
    /// resolve to and reports [`SourceFact::Absent`].
    fn style_of(&self, common: &acr::entities::EntityCommon) -> EntityStyle {
        let layer = self.doc.layers.get(&common.layer).or_else(|| {
            self.doc
                .layers
                .iter()
                .find(|l| l.name.eq_ignore_ascii_case(&common.layer))
        });
        let colour = match common.color {
            // Layer "0" inside a block is drawn on the layer of the INSERT that places it, which
            // this cursor cannot see, so BYLAYER there is handed on for the consumer to resolve (#1638).
            acr::Color::ByLayer if common.layer == "0" && !self.path.is_root() => {
                SourceFact::Present(SourceColour::ByLayer)
            }
            acr::Color::ByLayer => layer.map_or(SourceFact::Absent, |l| fact_colour(l.color)),
            c => fact_colour(c),
        };
        let lineweight = match common.line_weight {
            acr::LineWeight::ByLayer => {
                layer.map_or(SourceFact::Absent, |l| fact_lineweight(l.line_weight))
            }
            w => fact_lineweight(w),
        };
        let linetype = {
            // An empty name is BYLAYER, as the drawing stores it.
            let name =
                if common.linetype.is_empty() || common.linetype.eq_ignore_ascii_case("ByLayer") {
                    layer.map(|l| l.line_type.as_str())
                } else {
                    Some(common.linetype.as_str())
                };
            match name {
                None => SourceFact::Absent,
                Some(n) if n.eq_ignore_ascii_case("ByBlock") => {
                    SourceFact::Present(SourceLinetype::ByBlock)
                }
                Some(n) if n.eq_ignore_ascii_case("Continuous") => {
                    SourceFact::Present(SourceLinetype::Pattern {
                        name: n.to_owned(),
                        elements: Vec::new(),
                        scale: common.linetype_scale,
                    })
                }
                Some(n) => self.doc.line_types.get(n).map_or(SourceFact::Absent, |t| {
                    SourceFact::Present(SourceLinetype::Pattern {
                        name: n.to_owned(),
                        elements: t.elements.iter().map(|e| e.length).collect(),
                        scale: common.linetype_scale,
                    })
                }),
            }
        };
        EntityStyle {
            colour,
            lineweight,
            linetype,
        }
    }

    fn convert(&mut self, e: &EntityType) -> RawEntity {
        let common = e.common();
        let class_name = e.as_entity().entity_type().to_string();
        let provenance = ProvenanceStub {
            handle: item_handle(common.handle),
            class_name: class_name.clone(),
            block_path: self.path.clone(),
            layer: common.layer.clone(),
        };
        let unmapped = |reason: String| RawGeometry::Unmapped { reason };
        let geometry = match e {
            EntityType::Line(l) => RawGeometry::Line {
                start: p3(l.start),
                end: p3(l.end),
            },
            EntityType::Circle(c) => RawGeometry::Circle {
                center: p3(c.center),
                radius: c.radius,
                normal: p3(c.normal),
            },
            EntityType::Arc(a) => RawGeometry::Arc {
                center: p3(a.center),
                radius: a.radius,
                start_angle: a.start_angle,
                end_angle: a.end_angle,
                normal: p3(a.normal),
            },
            EntityType::Ellipse(el) => RawGeometry::Ellipse {
                center: p3(el.center),
                major_axis: p3(el.major_axis),
                minor_ratio: el.minor_axis_ratio,
                start_parameter: el.start_parameter,
                end_parameter: el.end_parameter,
                normal: p3(el.normal),
            },
            EntityType::LwPolyline(p) => RawGeometry::Polyline {
                vertices: p
                    .vertices
                    .iter()
                    .map(|v| RawVertex {
                        x: v.location.x,
                        y: v.location.y,
                        bulge: v.bulge,
                    })
                    .collect(),
                closed: p.is_closed,
                elevation: p.elevation,
                normal: p3(p.normal),
            },
            EntityType::Polyline2D(p) => RawGeometry::Polyline {
                vertices: p
                    .vertices
                    .iter()
                    .map(|v| RawVertex {
                        x: v.location.x,
                        y: v.location.y,
                        bulge: v.bulge,
                    })
                    .collect(),
                closed: p.flags.is_closed(),
                elevation: p.elevation,
                normal: p3(p.normal),
            },
            EntityType::Spline(s) => match u32::try_from(s.degree) {
                _ if s.control_points.is_empty() => unmapped(
                    "a spline with no control points (fit points only) has no control polygon to report"
                        .into(),
                ),
                Err(_) => unmapped(format!("a spline of degree {}", s.degree)),
                Ok(degree) => RawGeometry::Spline {
                    degree,
                    control_points: s.control_points.iter().copied().map(p3).collect(),
                    knots: if s.knots.is_empty() {
                        SourceFact::Absent
                    } else {
                        SourceFact::Present(s.knots.clone())
                    },
                    weights: if s.weights.is_empty() {
                        SourceFact::Absent
                    } else {
                        SourceFact::Present(s.weights.clone())
                    },
                    periodic: SourceFact::Present(s.flags.periodic),
                    closed: s.flags.closed,
                },
            },
            EntityType::Insert(i) => {
                let base = self.base_point(&i.block_name);
                let (sin, cos) = i.rotation.sin_cos();
                let (sx, sy) = (i.x_scale(), i.y_scale());
                let (m00, m01, m10, m11) = (cos * sx, -sin * sy, sin * sx, cos * sy);
                let tx = i.insert_point.x - (m00 * base[0] + m01 * base[1]);
                let ty = i.insert_point.y - (m10 * base[0] + m11 * base[1]);
                match SourceTransform::new([m00, m01, m10, m11, tx, ty]) {
                    Ok(transform) => RawGeometry::Insert(BlockRef {
                        block_name: i.block_name.clone(),
                        transform,
                        normal: p3(i.normal),
                        grid: ArrayGrid {
                            rows: u32::from(i.row_count).max(1),
                            columns: u32::from(i.column_count).max(1),
                            row_spacing: i.row_spacing,
                            column_spacing: i.column_spacing,
                        },
                        elevation: i.insert_point.z - i.z_scale() * base[2],
                        z_scale: i.z_scale(),
                    }),
                    Err(err) => unmapped(format!("an INSERT of {}: {err}", i.block_name)),
                }
            }
            // An IMAGE has no pixels any reader here will fetch (it names a path), so it is
            // reported with its frame and no placement is written (issue #1328, ADR 0018).
            EntityType::RasterImage(img) => {
                let loaded = img
                    .definition_handle
                    .and_then(|h| self.doc.objects.get(&h))
                    .is_some_and(|o| {
                        matches!(o, acr::objects::ObjectType::ImageDefinition(d) if d.is_loaded)
                    });
                unmapped(format!(
                    "IMAGE frame insert=({}, {}, {}) u=({}, {}, {}) v=({}, {}, {}) size=({}, {}) \
                     show_image={} definition_loaded={loaded}; no placement is written",
                    img.insertion_point.x,
                    img.insertion_point.y,
                    img.insertion_point.z,
                    img.u_vector.x,
                    img.u_vector.y,
                    img.u_vector.z,
                    img.v_vector.x,
                    img.v_vector.y,
                    img.v_vector.z,
                    img.size.x,
                    img.size.y,
                    img.is_visible(),
                ))
            }
            _ => unmapped(format!("no mapping for {class_name}")),
        };
        let style = self.style_of(common);
        RawEntity {
            geometry,
            provenance,
            style,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DrawnFact, arc_span, hatch_rings};
    use acadrust::entities::{
        BoundaryEdge, BoundaryPath, CircularArcEdge, EllipticArcEdge, Hatch, LineEdge, SplineEdge,
    };
    use acadrust::types::{Vector2, Vector3};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn edge(start: f64, end: f64, counter_clockwise: bool) -> CircularArcEdge {
        CircularArcEdge {
            center: Vector2 { x: 0.0, y: 0.0 },
            radius: 2.0,
            start_angle: start,
            end_angle: end,
            counter_clockwise,
        }
    }

    #[test]
    fn a_counter_clockwise_quarter_turn_has_a_bulge_of_tan_an_eighth() {
        let (from, to, bulge) = arc_span(&edge(0.0, FRAC_PI_2, true)).unwrap();
        assert!((from[0] - 2.0).abs() < 1e-12 && from[1].abs() < 1e-12);
        assert!(to[0].abs() < 1e-12 && (to[1] - 2.0).abs() < 1e-12);
        assert!((bulge - (FRAC_PI_2 / 4.0).tan()).abs() < 1e-12);
    }

    #[test]
    fn a_clockwise_edge_runs_from_the_negated_angles_and_bulges_negative() {
        // Stored as 270 and 90 degrees, it runs clockwise from 90 to 270 degrees: a half turn
        // through 0 degrees.
        let (from, to, bulge) = arc_span(&edge(1.5 * PI, FRAC_PI_2, false)).unwrap();
        assert!(
            from[0].abs() < 1e-9 && (from[1] - 2.0).abs() < 1e-9,
            "{from:?}"
        );
        assert!(to[0].abs() < 1e-9 && (to[1] + 2.0).abs() < 1e-9, "{to:?}");
        assert!((bulge + 1.0).abs() < 1e-9, "{bulge}");
    }

    #[test]
    fn an_edge_with_no_sweep_or_no_radius_is_not_a_span() {
        assert!(arc_span(&edge(1.0, 1.0, true)).is_none());
        let mut e = edge(0.0, 1.0, true);
        e.radius = 0.0;
        assert!(arc_span(&e).is_none());
    }

    fn rings_of(edges: Vec<BoundaryEdge>) -> DrawnFact {
        let mut path = BoundaryPath::new();
        for e in edges {
            path.add_edge(e);
        }
        let mut hatch = Hatch::solid();
        hatch.add_path(path);
        let mut out = DrawnFact::default();
        hatch_rings(&hatch, &mut out);
        out
    }

    fn half_ellipse(center: [f64; 2], major: [f64; 2], ratio: f64, from: f64) -> BoundaryEdge {
        BoundaryEdge::EllipticArc(EllipticArcEdge {
            center: Vector2 {
                x: center[0],
                y: center[1],
            },
            major_axis_endpoint: Vector2 {
                x: major[0],
                y: major[1],
            },
            minor_axis_ratio: ratio,
            start_angle: from,
            end_angle: from + PI,
            counter_clockwise: true,
        })
    }

    #[test]
    fn issue_1609_a_loop_of_two_elliptic_arcs_is_one_ring_on_the_ellipse() {
        let out = rings_of(vec![
            half_ellipse([10.0, 5.0], [4.0, 0.0], 0.5, 0.0),
            half_ellipse([10.0, 5.0], [4.0, 0.0], 0.5, PI),
        ]);
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);
        assert_eq!(out.rings.len(), 1);
        let ring = &out.rings[0];
        assert!(
            ring.len() >= 32,
            "{} vertices is a polygon, not a curve",
            ring.len()
        );
        for v in ring {
            let on = ((v.x - 10.0) / 4.0).powi(2) + ((v.y - 5.0) / 2.0).powi(2);
            assert!(
                (on - 1.0).abs() < 1e-9,
                "({}, {}) is off the ellipse",
                v.x,
                v.y
            );
            assert_eq!(v.bulge, 0.0);
        }
        assert_eq!(out.fills.len(), 1);
    }

    #[test]
    fn issue_1609_an_elliptic_arc_follows_a_tilted_major_axis_and_closes_on_a_line() {
        // The major axis points along (0, 4), so the ellipse is 1 wide and 4 tall around (0, 0).
        // The arc runs a quarter turn from the major axis to the minor one, and two lines close it.
        let arc = BoundaryEdge::EllipticArc(EllipticArcEdge {
            center: Vector2 { x: 0.0, y: 0.0 },
            major_axis_endpoint: Vector2 { x: 0.0, y: 4.0 },
            minor_axis_ratio: 0.25,
            start_angle: 0.0,
            end_angle: FRAC_PI_2,
            counter_clockwise: true,
        });
        let line = |a: [f64; 2], b: [f64; 2]| {
            BoundaryEdge::Line(LineEdge {
                start: Vector2 { x: a[0], y: a[1] },
                end: Vector2 { x: b[0], y: b[1] },
            })
        };
        // Parameter 0 is the major axis end (0, 4), parameter 90 degrees is (-1, 0).
        let out = rings_of(vec![
            arc,
            line([-1.0, 0.0], [0.0, 0.0]),
            line([0.0, 0.0], [0.0, 4.0]),
        ]);
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);
        let ring = &out.rings[0];
        let (first, last_arc) = (&ring[0], &ring[ring.len() - 3]);
        assert!(
            first.x.abs() < 1e-9 && (first.y - 4.0).abs() < 1e-9,
            "{first:?}"
        );
        assert!(
            ring.iter()
                .all(|v| (v.x * v.x + (v.y / 4.0).powi(2) - 1.0).abs() < 1e-9
                    || v.x == 0.0 && (v.y == 0.0 || v.y == 4.0))
        );
        assert!(last_arc.x < 0.0);
    }

    fn spline(degree: i32, knots: &[f64], points: &[[f64; 3]]) -> BoundaryEdge {
        BoundaryEdge::Spline(SplineEdge {
            degree,
            rational: points.iter().any(|p| p[2] != 1.0),
            periodic: false,
            knots: knots.to_vec(),
            control_points: points
                .iter()
                .map(|p| Vector3::new(p[0], p[1], p[2]))
                .collect(),
            fit_points: Vec::new(),
            start_tangent: Vector2 { x: 0.0, y: 0.0 },
            end_tangent: Vector2 { x: 0.0, y: 0.0 },
        })
    }

    #[test]
    fn issue_1609_a_spline_edge_is_sampled_along_its_curve() {
        // A quadratic Bezier from (0, 0) over (5, 10) to (10, 0) is the parabola y = 2x(1 - x/10).
        let curve = spline(
            2,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[[0.0, 0.0, 1.0], [5.0, 10.0, 1.0], [10.0, 0.0, 1.0]],
        );
        let close = BoundaryEdge::Line(LineEdge {
            start: Vector2 { x: 10.0, y: 0.0 },
            end: Vector2 { x: 0.0, y: 0.0 },
        });
        let out = rings_of(vec![curve, close]);
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);
        let ring = &out.rings[0];
        assert!(ring.len() >= 16, "{}", ring.len());
        assert!(
            ring.iter()
                .all(|v| (v.y - 2.0 * v.x * (1.0 - v.x / 10.0)).abs() < 1e-9 || v.y == 0.0)
        );
        let top = ring.iter().map(|v| v.y).fold(f64::MIN, f64::max);
        assert!((top - 5.0).abs() < 0.05, "the arch peaks at 5, not {top}");
    }

    #[test]
    fn issue_1609_a_rational_spline_edge_is_a_circle_arc() {
        // A quarter circle of radius 3 as a rational quadratic: weight cos(45 degrees) in the middle.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let curve = spline(
            2,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[[3.0, 0.0, 1.0], [3.0, 3.0, w], [0.0, 3.0, 1.0]],
        );
        let back = BoundaryEdge::Line(LineEdge {
            start: Vector2 { x: 0.0, y: 3.0 },
            end: Vector2 { x: 3.0, y: 0.0 },
        });
        let out = rings_of(vec![curve, back]);
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);
        assert!(
            out.rings[0]
                .iter()
                .filter(|v| v.x * v.x + v.y * v.y > 1e-9)
                .all(|v| ((v.x * v.x + v.y * v.y).sqrt() - 3.0).abs() < 1e-9
                    || (v.x == 0.0 || v.y == 0.0))
        );
    }

    #[test]
    fn issue_1609_a_spline_edge_that_does_not_join_its_neighbour_still_leaves_the_loop_unread() {
        let curve = spline(
            2,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[[0.0, 0.0, 1.0], [5.0, 10.0, 1.0], [10.0, 0.0, 1.0]],
        );
        let far = BoundaryEdge::Line(LineEdge {
            start: Vector2 { x: 50.0, y: 50.0 },
            end: Vector2 { x: 0.0, y: 0.0 },
        });
        let out = rings_of(vec![curve, far]);
        assert!(out.rings.is_empty());
        assert_eq!(out.skipped.len(), 1);
    }

    #[test]
    fn issue_1609_a_spline_with_a_broken_knot_vector_is_left_unread_not_guessed() {
        let curve = spline(
            2,
            &[0.0, 1.0],
            &[[0.0, 0.0, 1.0], [5.0, 10.0, 1.0], [10.0, 0.0, 1.0]],
        );
        let out = rings_of(vec![curve]);
        assert!(out.rings.is_empty());
        assert_eq!(out.skipped.len(), 1);
    }

    #[test]
    fn issue_1609_a_circle_edge_is_a_ring_of_two_half_turns() {
        let full = |ccw: bool| {
            BoundaryEdge::CircularArc(CircularArcEdge {
                center: Vector2 { x: 5.0, y: 5.0 },
                radius: 3.0,
                start_angle: 0.0,
                end_angle: 2.0 * PI,
                counter_clockwise: ccw,
            })
        };
        let out = rings_of(vec![full(true)]);
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);
        let ring = &out.rings[0];
        assert_eq!(ring.len(), 2);
        assert!((ring[0].x - 8.0).abs() < 1e-9 && (ring[1].x - 2.0).abs() < 1e-9);
        assert_eq!((ring[0].bulge, ring[1].bulge), (1.0, 1.0));
        let clockwise = rings_of(vec![full(false)]);
        assert_eq!(clockwise.rings[0][0].bulge, -1.0);
    }
}
