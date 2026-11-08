//! `pdf_page_to_pvec`: a PDF page's paths as a PVec artifact (issue #1578, L1).
//!
//! This is the mapping the server's `convert_pdf` carried before it moved here, so the server and
//! PlanViewer's test generator share one copy. Each painted path becomes polylines in inches with
//! the origin at the top left (style 2 for a fill, 1 for a stroke), curves flattened to 16 steps.
//! The bytes are the ones the server wrote, which `tests/pvec_1578_pdf_page_to_pvec.rs` pins on
//! `blueprint-mix.pdf`. Nothing from the paint design's new streams is written here yet.

use super::pdf_vector::{PdfPathSegment, PdfVectorPage};
use pvec_container::{
    Bounds, LengthUnit, Metadata,
    writer::{ContainerWriter, EntityMeta, WriterConfig},
};
use pvec_entity::EntityId;
use thiserror::Error;

/// Why a page could not be written as a PVec.
#[derive(Debug, Error)]
pub enum PdfToPvecError {
    /// No path on the page is painted with a fill or a stroke.
    #[error("The PDF has no supported vector paths for PlanViewer.")]
    NoPaths,
    /// The PVec writer refused an entity.
    #[error("PVec entity failed: {0}")]
    Entity(String),
    /// The PVec container could not be finished.
    #[error("PVec writing failed: {0}")]
    Write(String),
}

/// Writes `page` as a PVec artifact and returns its bytes. The bounds are the page in inches.
pub fn pdf_page_to_pvec(
    page: &PdfVectorPage,
    producer_id: &str,
    producer_version: &str,
) -> Result<Vec<u8>, PdfToPvecError> {
    let inches = |point: [f64; 2]| [point[0] / 72.0, (page.height - point[1]) / 72.0];
    let bounds = Bounds {
        min: [0.0, 0.0],
        max: [page.width / 72.0, page.height / 72.0],
    };
    let mut writer = ContainerWriter::new(WriterConfig {
        unit: LengthUnit::Inch,
        metadata: Metadata {
            producer_id: producer_id.into(),
            producer_version: producer_version.into(),
            corpus_revision: None,
        },
        bounds: Some(bounds),
        capabilities: vec![],
    });
    let mut next_id = 1u32;
    for entity in &page.entities {
        if entity.paint.fill.is_none() && !entity.paint.stroke {
            continue;
        }
        let style = if entity.paint.fill.is_some() { 2 } else { 1 };
        let mut current = None;
        let mut subpath_start = None;
        for segment in &entity.segments {
            let points: Vec<[f64; 2]> = match *segment {
                PdfPathSegment::MoveTo(point) => {
                    let point = inches(point);
                    current = Some(point);
                    subpath_start = Some(point);
                    continue;
                }
                PdfPathSegment::LineTo(point) => {
                    let end = inches(point);
                    let line = current.map(|start| vec![start, end]);
                    current = Some(end);
                    match line {
                        Some(line) => line,
                        None => continue,
                    }
                }
                PdfPathSegment::CurveTo { c1, c2, end } => {
                    let (c1, c2, end) = (inches(c1), inches(c2), inches(end));
                    let curve = current.map(|start| {
                        (0..=16)
                            .map(|step| {
                                let t = step as f64 / 16.0;
                                let u = 1.0 - t;
                                let blend = |axis: usize| {
                                    u * u * u * start[axis]
                                        + 3.0 * u * u * t * c1[axis]
                                        + 3.0 * u * t * t * c2[axis]
                                        + t * t * t * end[axis]
                                };
                                [blend(0), blend(1)]
                            })
                            .collect()
                    });
                    current = Some(end);
                    match curve {
                        Some(curve) => curve,
                        None => continue,
                    }
                }
                PdfPathSegment::Close => {
                    if let (Some(start), Some(end)) = (current, subpath_start) {
                        current = Some(end);
                        if start != end {
                            vec![start, end]
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
            };
            push_polyline(&mut writer, &mut next_id, style, &points, bounds)?;
        }
    }
    if next_id == 1 {
        return Err(PdfToPvecError::NoPaths);
    }
    writer
        .finish()
        .map_err(|error| PdfToPvecError::Write(error.to_string()))
}

/// One polyline entity: repeated points dropped, a single point (or none) skipped, the cell the
/// centre of its bounds falls in on a 16 by 16 grid over the page.
fn push_polyline(
    writer: &mut ContainerWriter,
    next_id: &mut u32,
    style: u32,
    points: &[[f64; 2]],
    page_bounds: Bounds,
) -> Result<(), PdfToPvecError> {
    let points = points
        .iter()
        .copied()
        .fold(Vec::new(), |mut points, point| {
            if points.last() != Some(&point) {
                points.push(point);
            }
            points
        });
    if points.len() < 2 {
        return Ok(());
    }
    let first = points[0];
    let bounds = points.iter().fold(
        Bounds {
            min: first,
            max: first,
        },
        |bounds, point| Bounds {
            min: [bounds.min[0].min(point[0]), bounds.min[1].min(point[1])],
            max: [bounds.max[0].max(point[0]), bounds.max[1].max(point[1])],
        },
    );
    let center_x = (bounds.min[0] + bounds.max[0]) / 2.0;
    let center_y = (bounds.min[1] + bounds.max[1]) / 2.0;
    let column = (center_x / page_bounds.max[0] * 16.0)
        .floor()
        .clamp(0.0, 15.0) as u32;
    let row = (center_y / page_bounds.max[1] * 16.0)
        .floor()
        .clamp(0.0, 15.0) as u32;
    let id = EntityId::new(*next_id).map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
    writer
        .push_polyline(
            EntityMeta {
                id,
                style,
                cell: row * 16 + column + 1,
                bounds,
            },
            &points,
        )
        .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
    *next_id += 1;
    Ok(())
}
