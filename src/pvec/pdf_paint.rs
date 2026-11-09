//! The paint order (stream 23), the fill rule (stream 25) and the hatch records of a PDF page
//! (issue #1578, L3). `pdf_page_to_pvec` hands over what it wrote in the order the PDF painted it,
//! and this builds the streams. Kept apart from the stroke colour (L2) and the ring cutting (L5),
//! which work on the same loop.

use super::components::{
    Fill, FillAssignment, FillPaintStream, FillRule, FillSource, HatchStream, ItemKind,
    PaintOrderStream, PaintRun, Stream,
};
use super::pdf_vector::PdfFillRule;
use pvec_capabilities::Registry;
use pvec_entity::EntityId;

/// What `pdf_page_to_pvec` painted, in content-stream order: one hatch record per filled path
/// (its rings, hidden, are the boundary, so a compound path keeps its holes as rings of the one
/// record) and the paint items in the order the PDF drew them. A `B` path puts its fill before its
/// outline.
pub(super) struct PaintLog {
    hatch: HatchStream,
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
            rules: Vec::new(),
            hidden: Vec::new(),
            items: Vec::new(),
        }
    }

    /// A filled path: `rings` are the polygon entities written for it, `rgb` its fill colour.
    /// Returns the numbers of the hatch records it became.
    pub(super) fn fill(
        &mut self,
        rings: &[EntityId],
        rule: PdfFillRule,
        rgb: [u8; 3],
    ) -> std::ops::Range<u32> {
        let first = self.hatch.records.len() as u32;
        let rule = match rule {
            PdfFillRule::EvenOdd => FillRule::EvenOdd,
            _ => FillRule::NonZero,
        };
        let [r, g, b] = rgb;
        // A record holds at most 4096 boundary entities.
        for rings in rings.chunks(4096) {
            let hatch = self.hatch.records.len() as u32;
            // The facade does not re-export `HatchRecord`, so a record is made from one decoded
            // from a one-ring payload (count, boundary kind, one entity, solid tag, colour).
            let mut record = HatchStream::decode(&[
                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            ])
            .expect("a one-ring solid hatch payload")
            .records
            .remove(0);
            record.boundary = rings.to_vec();
            record.fill = Fill::Solid {
                rgba: u32::from_be_bytes([r, g, b, 255]),
            };
            self.hatch.records.push(record);
            self.rules.push(FillAssignment {
                hatch,
                rule,
                paint: FillSource::Own,
            });
            self.items.push((ItemKind::Fill, hatch));
        }
        self.hidden.extend_from_slice(rings);
        first..self.hatch.records.len() as u32
    }

    /// A stroke entity, painted after everything logged before it.
    pub(super) fn stroke(&mut self, entity: EntityId) {
        self.items.push((ItemKind::Stroke, entity.index()));
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
                gradients: Vec::new(),
                tilings: Vec::new(),
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
