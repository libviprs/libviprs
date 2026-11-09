//! Non-rectangular clips as capability 26 (issue #1578, L4; `docs/PAINT.md` 2.5 and 6.4 in
//! libviprs/pvec).
//!
//! A clip that is an axis-aligned rectangle is cut into the geometry by `pdf_page_to_pvec`. Any
//! other `W` or `W*` path is kept whole: its subpaths are written as hidden closed ring entities
//! and a clip record names them with the winding rule. Clips nest through the parent of a record,
//! so a chain of clips in force is the chain of records, outermost first, and a stroke or fill is assigned
//! to the record of its innermost clip. A chain is cut off at eight records (the stencil bits a
//! renderer has), so the items under a deeper clip are drawn with the first eight.
//!
//! Strokes (by entity id) and fills (by hatch record number, from `pdf_paint`) are assigned. Text
//! and rasters follow the same way once they are written. A clip whose path has no ring of three distinct
//! points (a line, a point) is left unapplied, like a clip this module does not know.

use super::components::{ClipRecord, ClipRun, ClipStream, FillRule, ItemKind};
use super::pdf_clip::ClipReport;
use super::pdf_to_pvec::{source_point, subpaths};
use super::pdf_vector::{PdfFillRule, PdfVectorPage};
use pvec_entity::EntityId;
use std::collections::BTreeMap;

/// A general clip: its rule and its rings, closed, in inches from the top left.
pub(super) struct GeneralClip {
    pub rule: FillRule,
    pub rings: Vec<Vec<[f64; 2]>>,
}

/// The clips of the page that are not rectangles, by the `sequence` of the entity that defined
/// them. A clip that is empty is not here: its entities are dropped as fully clipped.
pub(super) fn general_clips(
    page: &PdfVectorPage,
    report: &ClipReport,
) -> BTreeMap<u64, GeneralClip> {
    let mut found = BTreeMap::new();
    for clip in report.clips.iter().filter(|c| c.rect.is_none() && !c.empty) {
        let Some(entity) = page
            .entities
            .iter()
            .find(|e| e.provenance.sequence == clip.sequence)
        else {
            continue;
        };
        let rings: Vec<Vec<[f64; 2]>> = subpaths(entity)
            .into_iter()
            .filter_map(|(mut points, _)| {
                points.dedup();
                if points.len() > 1 && points.first() == points.last() {
                    points.pop();
                }
                if points.len() < 3 {
                    return None;
                }
                points.push(points[0]);
                Some(
                    points
                        .iter()
                        .map(|&p| source_point(p, page.height))
                        .collect(),
                )
            })
            .collect();
        if !rings.is_empty() {
            found.insert(
                clip.sequence,
                GeneralClip {
                    rule: match clip.rule {
                        PdfFillRule::NonZero => FillRule::NonZero,
                        PdfFillRule::EvenOdd => FillRule::EvenOdd,
                    },
                    rings,
                },
            );
        }
    }
    found
}

/// The clip stream for the strokes in `stroked` (an entity id and its chain, ascending by id) and
/// the fills in `filled` (a hatch record number and its chain, ascending). `write_rings` writes the hidden rings of one clip, once per clip, and returns the rule and the
/// ids of the ring entities.
pub(super) fn build<E>(
    stroked: &[(EntityId, Vec<u64>)],
    filled: &[(u32, Vec<u64>)],
    mut write_rings: impl FnMut(u64) -> Result<(FillRule, Vec<EntityId>), E>,
) -> Result<ClipStream, E> {
    let mut rings: BTreeMap<u64, (FillRule, Vec<EntityId>)> = BTreeMap::new();
    let mut index: BTreeMap<&[u64], u32> = BTreeMap::new();
    let mut clips: Vec<ClipRecord> = Vec::new();
    let mut runs: Vec<ClipRun> = Vec::new();
    let items = stroked
        .iter()
        .map(|(id, chain)| (ItemKind::Stroke, id.index(), chain))
        .chain(
            filled
                .iter()
                .map(|(hatch, chain)| (ItemKind::Fill, *hatch, chain)),
        );
    for (kind, number, chain) in items {
        for end in 1..=chain.len() {
            let prefix = &chain[..end];
            if index.contains_key(prefix) {
                continue;
            }
            let sequence = chain[end - 1];
            if !rings.contains_key(&sequence) {
                rings.insert(sequence, write_rings(sequence)?);
            }
            let (rule, boundary) = rings[&sequence].clone();
            index.insert(prefix, clips.len() as u32);
            clips.push(ClipRecord {
                rule,
                parent: (end > 1).then(|| index[&chain[..end - 1]]),
                boundary,
            });
        }
        let clip = index[chain.as_slice()];
        match runs.last_mut() {
            Some(run)
                if run.kind == kind && run.clip == clip && run.first + run.count == number =>
            {
                run.count += 1
            }
            _ => runs.push(ClipRun {
                kind,
                first: number,
                count: 1,
                clip,
            }),
        }
    }
    Ok(ClipStream { clips, runs })
}
