//! The paint order (stream 23), the fill rule (stream 25) and the hatch records of a PDF page
//! (issue #1578, L3). `pdf_page_to_pvec` hands over what it wrote in the order the PDF painted it,
//! and this builds the streams. Kept apart from the stroke colour (L2) and the ring cutting (L5),
//! which work on the same loop.

use super::components::{
    Fill, FillAssignment, FillPaintStream, FillRule, FillSource, HatchStream, ItemKind,
    PaintOrderStream, PaintRun, Stream,
};
use super::pdf_shading::{PdfFillPattern, PdfGradientKind, PdfShading};
use super::pdf_tiling::{PdfTiling, PdfTilingPrimitive};
use super::pdf_vector::{PdfFillRule, PdfVectorPage};
use pvec_capabilities::Registry;
use pvec_components::{Gradient, GradientCoords, HatchRecord, Stop, Tiling, TilingPrimitive};
use pvec_entity::EntityId;

/// What `pdf_page_to_pvec` painted, in content-stream order: one hatch record per filled path
/// (its rings, hidden, are the boundary, so a compound path keeps its holes as rings of the one
/// record) and the paint items in the order the PDF drew them. A `B` path puts its fill before its
/// outline.
pub(super) struct PaintLog {
    hatch: HatchStream,
    gradients: Vec<Gradient>,
    tilings: Vec<Tiling>,
    rules: Vec<FillAssignment>,
    hidden: Vec<EntityId>,
    items: Vec<(ItemKind, u32)>,
}

impl PaintLog {
    pub(super) fn new() -> Self {
        Self {
            hatch: HatchStream {
                records: Vec::new(),
            },
            gradients: Vec::new(),
            tilings: Vec::new(),
            rules: Vec::new(),
            hidden: Vec::new(),
            items: Vec::new(),
        }
    }

    /// A filled path: `rings` are the polygon entities written for it, `rgb` and `alpha` its fill
    /// colour (`ca`). With a `pattern` (a shading or a tiling of `page`) the fill is that gradient
    /// or tiling in stream 25, and the hatch record keeps the colour in the middle of the ramp (or
    /// the mean colour of the cell) for a reader that does not know 25. A pattern that cannot be
    /// written (a zero length axis, a singular matrix) is the flat colour. Returns the numbers of
    /// the hatch records it became.
    pub(super) fn fill(
        &mut self,
        rings: &[EntityId],
        rule: PdfFillRule,
        (rgb, alpha): ([u8; 3], u8),
        pattern: Option<(PdfFillPattern, &PdfVectorPage)>,
    ) -> std::ops::Range<u32> {
        let first = self.hatch.records.len() as u32;
        let rule = match rule {
            PdfFillRule::EvenOdd => FillRule::EvenOdd,
            _ => FillRule::NonZero,
        };
        let height = pattern.map_or(0.0, |(_, page)| page.height);
        let (source, [r, g, b]) = match pattern {
            Some((PdfFillPattern::Shading(index), page)) => page
                .shadings
                .get(index)
                .and_then(|shading| Some((shading, gradient(shading, height, alpha)?)))
                .map(|(shading, gradient)| {
                    let index = self
                        .gradients
                        .iter()
                        .position(|known| *known == gradient)
                        .unwrap_or_else(|| {
                            self.gradients.push(gradient);
                            self.gradients.len() - 1
                        });
                    (FillSource::Gradient(index as u32), middle(shading))
                }),
            Some((PdfFillPattern::Tiling(index), page)) => page
                .tilings
                .get(index)
                .and_then(|tiling| Some((tiling, tiling_cell(tiling, height, alpha)?)))
                .map(|(tiling, cell)| {
                    let index = self
                        .tilings
                        .iter()
                        .position(|known| *known == cell)
                        .unwrap_or_else(|| {
                            self.tilings.push(cell);
                            self.tilings.len() - 1
                        });
                    (FillSource::Tiling(index as u32), mean(tiling, rgb))
                }),
            None => None,
        }
        .unwrap_or((FillSource::Own, rgb));
        // A record holds at most 4096 boundary entities.
        for rings in rings.chunks(4096) {
            let hatch = self.hatch.records.len() as u32;
            self.hatch.records.push(HatchRecord {
                boundary: rings.to_vec(),
                fill: Fill::Solid {
                    rgba: u32::from_be_bytes([r, g, b, alpha]),
                },
            });
            self.rules.push(FillAssignment {
                hatch,
                rule,
                paint: source,
            });
            self.items.push((ItemKind::Fill, hatch));
        }
        self.hidden.extend_from_slice(rings);
        first..self.hatch.records.len() as u32
    }

    /// A stroke (its entity id), a text record or a raster (its `RasterId`), painted after
    /// everything logged before it. Fills are logged by `fill`.
    pub(super) fn item(&mut self, kind: ItemKind, number: u32) {
        self.items.push((kind, number));
    }

    /// The rings of every fill, which `pdf_page_to_pvec` hides through the one line style stream.
    pub(super) fn hidden_rings(&self) -> &[EntityId] {
        &self.hidden
    }

    /// The streams to push: hatch (17), fill paint (25) and paint order (23). Nothing for a page with no paint.
    pub(super) fn streams(self, registry: &Registry) -> Result<Vec<Stream>, String> {
        let refused = |error: &dyn std::fmt::Display| error.to_string();
        let mut out = Vec::new();
        if !self.hatch.records.is_empty() {
            out.push(self.hatch.into_stream(registry).map_err(|e| refused(&e))?);
            let fill_paint = FillPaintStream {
                gradients: self.gradients,
                tilings: self.tilings,
                assignments: self.rules,
            };
            out.push(fill_paint.into_stream(registry).map_err(|e| refused(&e))?);
        }
        let mut runs: Vec<PaintRun> = Vec::new();
        for (kind, number) in self.items {
            match runs.last_mut() {
                Some(run) if run.kind == kind && run.first + run.count == number => run.count += 1,
                _ => runs.push(PaintRun {
                    kind,
                    first: number,
                    count: 1,
                }),
            }
        }
        if !runs.is_empty() {
            let order = PaintOrderStream { runs };
            out.push(order.into_stream(registry).map_err(|e| refused(&e))?);
        }
        Ok(out)
    }
}

/// The shading as a gradient in source space (inches, y down), with `alpha` on every stop. The
/// matrix of the shading maps gradient space to the page in points with y up; the gradient's maps
/// it to source space in PVec's `[a b c d tx ty]` order (rows `[a b tx]` and `[c d ty]`). `None`
/// when PVec would refuse it.
fn gradient(shading: &PdfShading, page_height: f64, alpha: u8) -> Option<Gradient> {
    let transform = to_source(shading.matrix, page_height);
    let det = transform[0] * transform[3] - transform[1] * transform[2];
    let coords = match (shading.kind, shading.coords.as_slice()) {
        (PdfGradientKind::Axial, &[x0, y0, x1, y1]) if (x0, y0) != (x1, y1) => {
            GradientCoords::Axial { x0, y0, x1, y1 }
        }
        (PdfGradientKind::Radial, &[x0, y0, r0, x1, y1, r1])
            if r0 >= 0.0 && r1 >= 0.0 && (x0, y0, r0) != (x1, y1, r1) =>
        {
            GradientCoords::Radial {
                x0,
                y0,
                r0,
                x1,
                y1,
                r1,
            }
        }
        _ => return None,
    };
    (det != 0.0 && det.is_finite() && transform.iter().all(|v| v.is_finite())).then(|| Gradient {
        coords,
        extend_before: shading.extend[0],
        extend_after: shading.extend[1],
        transform,
        stops: shading
            .stops
            .iter()
            .map(|stop| Stop {
                offset: stop.offset,
                rgba: u32::from_be_bytes([stop.rgb[0], stop.rgb[1], stop.rgb[2], alpha]),
            })
            .collect(),
    })
}

/// The colour halfway along the shading's ramp, which a reader that does not know stream 25 draws.
fn middle(shading: &PdfShading) -> [u8; 3] {
    let stops = &shading.stops;
    let Some(after) = stops.iter().position(|stop| stop.offset >= 0.5) else {
        return stops.last().map_or([0; 3], |stop| stop.rgb);
    };
    let (next, before) = (&stops[after], after.checked_sub(1).map(|i| &stops[i]));
    match before {
        Some(before) if next.offset > 0.5 => {
            let t = (0.5 - before.offset) / (next.offset - before.offset);
            std::array::from_fn(|i| {
                (f64::from(before.rgb[i]) + t * (f64::from(next.rgb[i]) - f64::from(before.rgb[i])))
                    .round() as u8
            })
        }
        _ => next.rgb,
    }
}

/// A map to the page in points with y up, `[a b c d e f]` in PDF's row-vector order, as the map
/// to source space (inches, y down) in PVec's `[a b c d tx ty]` order (rows `[a b tx]`, `[c d ty]`).
fn to_source([a, b, c, d, e, f]: [f64; 6], page_height: f64) -> [f64; 6] {
    [
        a / 72.0,
        c / 72.0,
        -b / 72.0,
        -d / 72.0,
        e / 72.0,
        (page_height - f) / 72.0,
    ]
}

/// The cell as a PVec tiling with `alpha` on every colour. `None` when PVec would refuse it.
fn tiling_cell(tiling: &PdfTiling, page_height: f64, alpha: u8) -> Option<Tiling> {
    let transform = to_source(tiling.matrix, page_height);
    let det = transform[0] * transform[3] - transform[1] * transform[2];
    let word = |rgb: [u8; 3]| u32::from_be_bytes([rgb[0], rgb[1], rgb[2], alpha]);
    let primitives = tiling
        .primitives
        .iter()
        .map(|primitive| match primitive {
            PdfTilingPrimitive::Polygon { rule, rgb, points } if points.len() >= 3 => {
                Some(TilingPrimitive::Polygon {
                    rule: match rule {
                        PdfFillRule::EvenOdd => FillRule::EvenOdd,
                        _ => FillRule::NonZero,
                    },
                    rgba: word(*rgb),
                    points: points.clone(),
                })
            }
            PdfTilingPrimitive::Polyline { rgb, width, points }
                if points.len() >= 2 && width.is_finite() && *width >= 0.0 =>
            {
                Some(TilingPrimitive::Polyline {
                    rgba: word(*rgb),
                    width: *width,
                    points: points.clone(),
                })
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    (det != 0.0 && det.is_finite() && transform.iter().all(|v| v.is_finite())).then(|| Tiling {
        coloured: tiling.coloured,
        x_step: tiling.steps[0],
        y_step: tiling.steps[1],
        cell: tiling.cell,
        transform,
        primitives,
    })
}

/// The mean colour of the cell's primitives, which a reader that does not know stream 25 draws.
/// An uncoloured cell is painted in `fill`.
fn mean(tiling: &PdfTiling, fill: [u8; 3]) -> [u8; 3] {
    let colours: Vec<[u8; 3]> = tiling
        .primitives
        .iter()
        .map(|primitive| match primitive {
            PdfTilingPrimitive::Polygon { rgb, .. } | PdfTilingPrimitive::Polyline { rgb, .. } => {
                *rgb
            }
        })
        .collect();
    if !tiling.coloured || colours.is_empty() {
        return fill;
    }
    std::array::from_fn(|i| {
        (colours.iter().map(|c| u32::from(c[i])).sum::<u32>() / colours.len() as u32) as u8
    })
}
