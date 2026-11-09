//! `pdf_page_to_pvec`: a PDF page's paths as a PVec artifact (issue #1578, L4).
//!
//! This is the mapping iasbuilt/server's `convert_pdf` runs on main (the white-fill, closed-polygon
//! and rectangular-clip handling from server #336), so the server and PlanViewer's test generator
//! share one copy. Each painted path becomes polylines in inches with the origin at the top left
//! (style 2 for a filled polygon, 1 for a stroke), curves flattened to 16 steps. A white fill is
//! dropped (a stroked one keeps its outline), and a path inside a rectangular clip is cut to it
//! through `apply_clip_policy`. `tests/pvec_1578_pdf_page_to_pvec.rs` pins the bytes on
//! `blueprint-mix.pdf`. The stroke colour and alpha (stream 24) and the stroke width and dashes (the line style stream) are written for every stroke (L2); the geometry blocks are the ones server main writes, the test pins the whole file.
//!
//! This module flattens curves and compares coordinates exactly, which ADR NG-10 (NG-B2, NG-B6)
//! keeps out of normalization and which `pdf_vector` and `pdf_bezier` say they never do. It is
//! allowed here because the output has to match the bytes the server already writes, so the
//! flattening and the clip cuts are copied as they are. It is a stopgap, and it goes away when the
//! stream work of #1578 replaces this mapping.
//!
//! The paint order (stream 23), the fill rule (stream 25) and the hatch record of every fill come from `pdf_paint` (L3). A clip that is not a rectangle is written as capability 26 for the strokes and fills under it (L4, `pdf_clip_stream`).

use super::components::{
    ItemKind, LineStyleDef, LineStyleStream, Stream, StrokePaintStream, StrokeWidth,
};
use super::limits::SourceLimits;
use super::pdf_clip::{ClipCase, ClipLimits, ExactBounds, LinesOnlyBounds, apply_clip_policy};
use super::pdf_clip_stream::{build as build_clip_stream, general_clips};
use super::pdf_image::{PdfImageOptions, bake_page_images};
use super::pdf_paint::PaintLog;
use super::pdf_ring_cut::cut_ring;
use super::pdf_shading::PdfFillPattern;
use super::pdf_vector::{
    PdfFillRule, PdfPaintIntent, PdfPathEntity, PdfPathSegment, PdfProvenance, PdfStrokeParams,
    PdfVectorPage,
};
use pvec_capabilities::Registry;
use pvec_components::{RASTER_CAPABILITY, RasterId, TextRecord, TextStream, encode_rasters};
use pvec_container::{
    Bounds, LengthUnit, Metadata,
    writer::{ContainerWriter, EntityMeta, WriterConfig},
};
use pvec_entity::EntityId;
use std::collections::BTreeMap;
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

/// Writes `page` as a PVec artifact, with its text but without its images, and returns its bytes.
/// The bounds are the page in inches. See [`pdf_page_to_pvec_with`] to bake the images as well.
pub fn pdf_page_to_pvec(
    page: &PdfVectorPage,
    producer_id: &str,
    producer_version: &str,
) -> Result<Vec<u8>, PdfToPvecError> {
    pdf_page_to_pvec_with(page, producer_id, producer_version, false)
}

/// [`pdf_page_to_pvec`], and with `bake_images` the page's images baked into the raster capability
/// as well. Baking decodes and tiles every image, so a scanned sheet takes minutes in a debug build.
pub fn pdf_page_to_pvec_with(
    page: &PdfVectorPage,
    producer_id: &str,
    producer_version: &str,
    bake_images: bool,
) -> Result<Vec<u8>, PdfToPvecError> {
    /// The polylines one PDF path becomes, as (style, points in inches). Style 2 is a closed fill
    /// polygon the viewer can fill, style 1 a stroke. `clip` is the page-space rectangle the path is
    /// drawn inside, when there is one.
    ///
    /// Limits: every filled subpath is its own solid polygon, so holes in a compound path (rings,
    /// letter counters) fill solid. A fill colour set through a pattern or an unknown colour space is
    /// not tracked by libviprs, so the last plain colour decides whether it counts as white.
    fn entity_runs(
        entity: &PdfPathEntity,
        clip: Option<ExactBounds>,
        page_height: f64,
    ) -> Vec<(u32, Vec<[f64; 2]>)> {
        /// The pieces of a polyline that lie inside the rectangle, each continuous.
        fn clip_polyline(points: &[[f64; 2]], rect: ExactBounds) -> Vec<Vec<[f64; 2]>> {
            /// Liang-Barsky: the part of the segment inside the rectangle.
            fn clip_segment(
                a: [f64; 2],
                b: [f64; 2],
                rect: ExactBounds,
            ) -> Option<([f64; 2], [f64; 2])> {
                let (mut t0, mut t1) = (0.0f64, 1.0f64);
                let d = [b[0] - a[0], b[1] - a[1]];
                for (p, q) in [
                    (-d[0], a[0] - rect.min[0]),
                    (d[0], rect.max[0] - a[0]),
                    (-d[1], a[1] - rect.min[1]),
                    (d[1], rect.max[1] - a[1]),
                ] {
                    if p == 0.0 {
                        if q < 0.0 {
                            return None;
                        }
                    } else {
                        let t = q / p;
                        if p < 0.0 {
                            if t > t1 {
                                return None;
                            }
                            t0 = t0.max(t);
                        } else {
                            if t < t0 {
                                return None;
                            }
                            t1 = t1.min(t);
                        }
                    }
                }
                let at = |t: f64| [a[0] + t * d[0], a[1] + t * d[1]];
                Some((
                    if t0 == 0.0 { a } else { at(t0) },
                    if t1 == 1.0 { b } else { at(t1) },
                ))
            }
            let mut pieces: Vec<Vec<[f64; 2]>> = Vec::new();
            for pair in points.windows(2) {
                match clip_segment(pair[0], pair[1], rect) {
                    Some((start, end)) => match pieces.last_mut() {
                        Some(piece) if piece.last() == Some(&start) => piece.push(end),
                        _ => pieces.push(vec![start, end]),
                    },
                    None => pieces.push(Vec::new()),
                }
            }
            pieces.retain(|piece| !piece.is_empty());
            pieces
        }

        // The viewer paints the page white, so a white fill adds nothing. One that is also stroked
        // keeps its outline.
        // A shading fill is not white whatever the last plain colour was.
        let fills = entity.paint.fill.is_some()
            && (entity.fill_pattern.is_some() || entity.fill_colour.iter().any(|&c| c < 250));
        let strokes = entity.paint.stroke;
        let mut runs = Vec::new();
        for (mut points, closed) in subpaths(entity) {
            points.dedup();
            // A fill closes its subpath implicitly, so three distinct points are a polygon whether or
            // not it has an `h`. A trailing point equal to the first is the closing edge already.
            let mut ring = points.clone();
            if ring.len() > 1 && ring.first() == ring.last() {
                ring.pop();
            }
            let filled = fills && ring.len() >= 3;
            if filled {
                ring.push(ring[0]);
                // A fill is cut as a closed ring (a rectangle is convex, so the cut keeps the fill
                // rule), never as a polyline: that can split a ring into open pieces.
                if let Some(clip) = clip {
                    let [(x0, y0), (x1, y1)] =
                        [(clip.min[0], clip.min[1]), (clip.max[0], clip.max[1])];
                    ring = cut_ring(&ring, &[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
                        .unwrap_or_default();
                }
                if ring.len() >= 4 {
                    runs.push((
                        2,
                        ring.iter().map(|&p| source_point(p, page_height)).collect(),
                    ));
                }
            }
            // A stroked fill (`B`) keeps its outline as well as its polygon.
            if strokes {
                if closed && points.first() != points.last() {
                    points.push(points[0]);
                }
                let pieces = match clip {
                    Some(clip) => clip_polyline(&points, clip),
                    None => vec![points],
                };
                for piece in pieces.into_iter().filter(|piece| piece.len() >= 2) {
                    runs.push((
                        1,
                        piece
                            .iter()
                            .map(|&p| source_point(p, page_height))
                            .collect(),
                    ));
                }
            }
        }
        runs
    }

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
    // What was painted, in the PDF's own order: the hatch records, the fill rules and stream 23.
    let mut log = PaintLog::new();
    // What each stroke entity says about its paint: a colour word (`0xRRGGBBAA`, the alpha byte is
    // the stroke alpha), and its line style (width and dashes in inches, the PDF's own `w` and `d`
    // times the scale of the CTM it was painted under).
    let mut stroke_colours: Vec<(EntityId, u32)> = Vec::new();
    let mut stroke_styles: Vec<(EntityId, LineStyleDef)> = Vec::new();
    // A clip that is a plain rectangle is applied to the geometry. A general clip path is kept
    // unclipped (display fidelity only), as the clip policy records. When clip analysis fails
    // (more clips in force at once than the policy allows), the page still converts, unclipped.
    let clips = apply_clip_policy(
        page,
        &ClipLimits::from(&SourceLimits::default()),
        &LinesOnlyBounds,
    )
    .ok();
    let general = clips
        .as_ref()
        .map(|clips| general_clips(page, clips))
        .unwrap_or_default();
    // Each stroke (by entity id), fill (by hatch record number), text record and raster (by
    // `RasterId`) written under general clips, with the clips in force (outermost first, cut off
    // at eight, the depth a clip stream allows).
    let mut clipped: Vec<(ItemKind, u32, Vec<u64>)> = Vec::new();
    let general_chain = |active: &[u64]| -> Vec<u64> {
        active
            .iter()
            .copied()
            .filter(|sequence| general.contains_key(sequence))
            .take(8)
            .collect()
    };
    // Images (L6), only when the caller asks: baking decodes and tiles every image, which takes
    // minutes on a scanned sheet. They go into the raster capability in content-stream order, each
    // placed by its CTM. The pyramid of the whole page stays the first picture under them.
    let bake = bake_images
        .then(|| {
            bake_page_images(
                page,
                &PdfImageOptions {
                    page_to_source: [1.0 / 72.0, 0.0, 0.0, -1.0 / 72.0, 0.0, page.height / 72.0],
                    ..PdfImageOptions::default()
                },
            )
        })
        .filter(|bake| !bake.written.is_empty());
    // The text runs that get a record, in order: a baseline with a length and a height, and not
    // under a clip that leaves nothing.
    let empty_clips: Vec<u64> = clips
        .iter()
        .flat_map(|clips| &clips.clips)
        .filter(|clip| clip.empty)
        .map(|clip| clip.sequence)
        .collect();
    let texts: Vec<_> = page
        .texts
        .iter()
        .filter_map(|run| {
            let start = source_point(run.start, page.height);
            let end = source_point(run.end, page.height);
            let height = run.cap_height / 72.0;
            let visible = start != end && height > 0.0 && height.is_finite();
            (visible && !run.active_clips.iter().any(|c| empty_clips.contains(c)))
                .then_some((run, start, end, height))
        })
        .collect();
    // The rasters and the text records are painted where their `sequence` says (the number of
    // entities before them): an image before a text run of the same number, both before the entity
    // of that number.
    let mut extras: Vec<(u64, ItemKind, u32, Vec<u64>)> = Vec::new();
    for (raster, baked) in bake.iter().flat_map(|bake| &bake.written).enumerate() {
        let image = &page.images[baked.index];
        extras.push((
            image.provenance.sequence,
            ItemKind::Raster,
            raster as u32,
            general_chain(&image.active_clips),
        ));
    }
    for (record, (run, ..)) in texts.iter().enumerate() {
        extras.push((
            run.sequence,
            ItemKind::Text,
            record as u32,
            general_chain(&run.active_clips),
        ));
    }
    extras.sort_by_key(|extra| extra.0);
    let mut extras = extras.into_iter().peekable();
    let decisions: BTreeMap<u64, _> = clips
        .iter()
        .flat_map(|clips| &clips.decisions)
        .map(|decision| (decision.entity, decision))
        .collect();
    let rectangles: BTreeMap<u64, _> = clips
        .iter()
        .flat_map(|clips| &clips.clips)
        .filter_map(|clip| Some((clip.sequence, clip.rect?)))
        .collect();
    // An `sh` paints a shading over the clip in force: a fill of the page rectangle with that
    // shading, in the place the PDF painted it (`sequence` is the number of entities before it).
    let shading_fills: Vec<PdfPathEntity> = page
        .shading_ops
        .iter()
        .map(|op| PdfPathEntity {
            segments: [
                [0.0, 0.0],
                [page.width, 0.0],
                [page.width, page.height],
                [0.0, page.height],
            ]
            .into_iter()
            .enumerate()
            .map(|(corner, point)| match corner {
                0 => PdfPathSegment::MoveTo(point),
                _ => PdfPathSegment::LineTo(point),
            })
            .chain([PdfPathSegment::Close])
            .collect(),
            paint: PdfPaintIntent {
                fill: Some(PdfFillRule::NonZero),
                stroke: false,
                close: false,
            },
            clip: None,
            fill_colour: [0, 0, 0],
            fill_pattern: Some(PdfFillPattern::Shading(op.shading)),
            stroke_colour: [0, 0, 0],
            fill_alpha: 255,
            stroke_alpha: 255,
            ctm_scale: 1.0,
            ctm_anisotropy: 1.0,
            active_clips: op.active_clips.clone(),
            soft_masked: op.soft_masked,
            text_clipped: false,
            stroke_params: PdfStrokeParams::default(),
            provenance: PdfProvenance {
                sequence: op.provenance.sequence,
                ..op.provenance.clone()
            },
        })
        .collect();
    let mut painted: Vec<(&PdfPathEntity, bool)> = page
        .entities
        .iter()
        .map(|entity| (entity, false))
        .chain(shading_fills.iter().map(|entity| (entity, true)))
        .collect();
    // An `sh` goes before the entity that had its sequence number, and after one painted earlier.
    painted.sort_by_key(|&(entity, is_sh)| (entity.provenance.sequence, !is_sh));
    for (entity, is_sh) in painted {
        // Rasters and text painted before this entity (an `sh` has the number of the entity it
        // goes before, and comes first among them).
        let sequence = entity.provenance.sequence;
        while let Some((_, kind, number, chain)) =
            extras.next_if(|extra| extra.0 < sequence || (!is_sh && extra.0 == sequence))
        {
            log.item(kind, number);
            if !chain.is_empty() {
                clipped.push((kind, number, chain));
            }
        }
        let mut clip = None;
        let mut chain: Vec<u64> = Vec::new();
        if is_sh {
            // Clips in force at the `sh`: the rectangles cut it, the others are its clip chain.
            chain = general_chain(&entity.active_clips);
            clip = entity
                .active_clips
                .iter()
                .filter_map(|sequence| rectangles.get(sequence).copied())
                .fold(None, |both: Option<ExactBounds>, rect| {
                    Some(both.map_or(rect, |both| ExactBounds {
                        min: [both.min[0].max(rect.min[0]), both.min[1].max(rect.min[1])],
                        max: [both.max[0].min(rect.max[0]), both.max[1].min(rect.max[1])],
                    }))
                });
            if clip.is_some_and(|clip| clip.min[0] > clip.max[0] || clip.min[1] > clip.max[1]) {
                continue;
            }
        } else if let Some(clips) = &clips {
            let Some(decision) = decisions.get(&entity.provenance.sequence) else {
                continue;
            };
            if decision.case == ClipCase::FullyClipped {
                continue;
            }
            chain = general_chain(&decision.clips);
            let rects = decision
                .clips
                .iter()
                .filter_map(|sequence| rectangles.get(sequence).copied())
                .chain(
                    decision
                        .form_clips
                        .iter()
                        .filter_map(|&index| clips.form_clips[index].rect),
                );
            clip = rects.fold(None, |both: Option<ExactBounds>, rect| {
                Some(both.map_or(rect, |both| ExactBounds {
                    min: [both.min[0].max(rect.min[0]), both.min[1].max(rect.min[1])],
                    max: [both.max[0].min(rect.max[0]), both.max[1].min(rect.max[1])],
                }))
            });
            if clip.is_some_and(|clip| clip.min[0] > clip.max[0] || clip.min[1] > clip.max[1]) {
                continue;
            }
        }
        // The PDF paints a path's fill and then its stroke, so the rings come first.
        let mut runs = entity_runs(entity, clip, page.height);
        runs.sort_by_key(|(style, _)| std::cmp::Reverse(*style));
        let mut rings = Vec::new();
        let mut strokes = Vec::new();
        for (style, points) in runs {
            // One polyline entity: repeated points dropped, a single point (or none) skipped, the
            // cell the centre of its bounds falls in on a 16 by 16 grid over the page.
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
                continue;
            }
            let (entity_bounds, cell) = bounds_and_cell(&points, bounds);
            let id = EntityId::new(next_id)
                .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
            if style == 1 {
                let [r, g, b] = entity.stroke_colour;
                stroke_colours.push((id, u32::from_be_bytes([r, g, b, entity.stroke_alpha])));
                // PDF user space is 1/72 inch by the scale of the CTM. A width of 0 is the thinnest
                // line the output can draw.
                let to_inches = entity.ctm_scale / 72.0;
                let width = entity.stroke_params.width.unwrap_or(1.0) * to_inches;
                let (dashes, phase) = entity.stroke_params.dash.as_ref().map_or_else(
                    || (Vec::new(), 0.0),
                    |dash| {
                        (
                            dash.array.iter().map(|d| d * to_inches).collect::<Vec<_>>(),
                            dash.phase * to_inches,
                        )
                    },
                );
                // An all-zero or negative array draws solid in a PDF viewer, as no array does.
                let solid = dashes.len() > 16 || dashes.iter().all(|d| *d <= 0.0);
                stroke_styles.push((
                    id,
                    LineStyleDef {
                        name: "pdf-stroke".into(),
                        dashes: if solid { Vec::new() } else { dashes },
                        width: Some(if width > 0.0 {
                            StrokeWidth::Width(width)
                        } else {
                            StrokeWidth::Thinnest
                        }),
                        phase: if solid { 0.0 } else { phase },
                        visible: true,
                    },
                ));
            }
            writer
                .push_polyline(
                    EntityMeta {
                        id,
                        style,
                        cell,
                        bounds: entity_bounds,
                    },
                    &points,
                )
                .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
            if style == 1 && !chain.is_empty() {
                clipped.push((ItemKind::Stroke, id.index(), chain.clone()));
            }
            next_id += 1;
            if style == 2 {
                rings.push(id);
            } else {
                strokes.push(id);
            }
        }
        if let (false, Some(rule)) = (rings.is_empty(), entity.paint.fill) {
            let hatches = log.fill(
                &rings,
                rule,
                (entity.fill_colour, entity.fill_alpha),
                entity.fill_pattern.map(|pattern| (pattern, page)),
            );
            if !chain.is_empty() {
                clipped.extend(hatches.map(|hatch| (ItemKind::Fill, hatch, chain.clone())));
            }
        }
        for id in strokes {
            log.item(ItemKind::Stroke, id.index());
        }
    }
    // What is left was painted after every entity.
    for (_, kind, number, chain) in extras {
        log.item(kind, number);
        if !chain.is_empty() {
            clipped.push((kind, number, chain));
        }
    }
    if next_id == 1 {
        return Err(PdfToPvecError::NoPaths);
    }
    let registry = Registry::v1();
    let refused = |error: &dyn std::fmt::Display| PdfToPvecError::Write(error.to_string());
    // The rings of the general clips are hidden closed polylines, written after the page's own
    // entities, with the clip stream that names them.
    if !clipped.is_empty() {
        // Strokes first, then fills, text and rasters, so the records come in the order they did
        // before text and rasters were assigned.
        clipped.sort_by_key(|(kind, ..)| match kind {
            ItemKind::Stroke => 0,
            ItemKind::Fill => 1,
            ItemKind::Text => 2,
            ItemKind::Raster => 3,
        });
        let stream = build_clip_stream(&clipped, |sequence| {
            let clip = &general[&sequence];
            let mut boundary = Vec::new();
            for ring in &clip.rings {
                let id = EntityId::new(next_id)
                    .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
                let (ring_bounds, cell) = bounds_and_cell(ring, bounds);
                writer
                    .push_polyline(
                        EntityMeta {
                            id,
                            style: 1,
                            cell,
                            bounds: ring_bounds,
                        },
                        ring,
                    )
                    .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
                stroke_styles.push((
                    id,
                    LineStyleDef {
                        name: "pdf-clip-ring".into(),
                        dashes: Vec::new(),
                        width: Some(StrokeWidth::Thinnest),
                        phase: 0.0,
                        visible: false,
                    },
                ));
                boundary.push(id);
                next_id += 1;
            }
            Ok((clip.rule, boundary))
        })?;
        writer
            .push_stream(stream.into_stream(&registry).map_err(|e| refused(&e))?)
            .map_err(|e| refused(&e))?;
    }
    // Text (L7): a record per run, on a hidden two point polyline along its baseline, which gives
    // the text its bounds and the colour of stream 24 something to hang on. The rotation is the
    // direction of the baseline in source space, which has y down.
    let mut records = Vec::new();
    for &(run, start, end, height) in &texts {
        let (anchor_bounds, cell) = bounds_and_cell(&[start, end], bounds);
        let id =
            EntityId::new(next_id).map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
        writer
            .push_polyline(
                EntityMeta {
                    id,
                    style: 1,
                    cell,
                    bounds: anchor_bounds,
                },
                &[start, end],
            )
            .map_err(|error| PdfToPvecError::Entity(error.to_string()))?;
        next_id += 1;
        let [r, g, b] = run.fill_colour;
        stroke_colours.push((id, u32::from_be_bytes([r, g, b, 255])));
        stroke_styles.push((
            id,
            LineStyleDef {
                name: "pdf-text-anchor".into(),
                dashes: Vec::new(),
                width: Some(StrokeWidth::Thinnest),
                phase: 0.0,
                visible: false,
            },
        ));
        let mut text = run.text.as_str();
        if text.len() > 4096 {
            text = &text[..text.floor_char_boundary(4096)];
        }
        records.push(TextRecord {
            entity: id,
            x: start[0],
            y: start[1],
            height,
            rotation: (end[1] - start[1]).atan2(end[0] - start[0]),
            text: text.to_owned(),
        });
    }
    if !records.is_empty() {
        let stream = TextStream { records }
            .into_stream(&registry)
            .map_err(|e| refused(&e))?;
        writer.push_stream(stream).map_err(|e| refused(&e))?;
    }
    if let Some(bake) = bake {
        let placements = bake
            .written
            .iter()
            .map(|baked| crate::raster_pvec::container::placement(&baked.pyramid))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| refused(&e))?;
        let payload = encode_rasters(&placements).map_err(|e| refused(&e))?;
        let version = registry
            .get(RASTER_CAPABILITY)
            .map(|c| c.version)
            .ok_or_else(|| refused(&"the registry has no raster capability"))?;
        writer
            .push_stream(Stream {
                capability: RASTER_CAPABILITY,
                version,
                payload,
            })
            .map_err(|e| refused(&e))?;
        for (index, baked) in bake.written.iter().enumerate() {
            let raster = RasterId::new(index as u32).map_err(|e| refused(&e))?;
            let tiles = baked
                .pyramid
                .tiles()
                .map(
                    |(level, col, row, bytes)| pvec_container::blocks::raster::TileIn {
                        level,
                        col,
                        row,
                        bytes: bytes.to_vec(),
                    },
                )
                .collect();
            writer
                .push_raster_tiles(raster, tiles)
                .map_err(|e| refused(&e))?;
        }
    }
    if !stroke_colours.is_empty() {
        // The colour table, one entry per distinct word, in first-seen order.
        let mut colours: Vec<u32> = Vec::new();
        let mut colour_index: BTreeMap<u32, u32> = BTreeMap::new();
        let assignments = stroke_colours
            .iter()
            .map(|&(id, word)| {
                let index = *colour_index.entry(word).or_insert_with(|| {
                    colours.push(word);
                    (colours.len() - 1) as u32
                });
                (id, index)
            })
            .collect();
        let paint = StrokePaintStream {
            colours,
            assignments,
        };
        writer
            .push_stream(paint.into_stream(&registry).map_err(|e| refused(&e))?)
            .map_err(|e| refused(&e))?;
    }
    // One line style stream: the stroke widths and dashes, and the hidden style of the fill rings.
    stroke_styles.extend(log.hidden_rings().iter().map(|&id| {
        (
            id,
            LineStyleDef {
                name: "pdf-fill-ring".into(),
                dashes: Vec::new(),
                width: None,
                phase: 0.0,
                visible: false,
            },
        )
    }));
    if !stroke_styles.is_empty() {
        let mut styles: Vec<LineStyleDef> = Vec::new();
        let mut style_index: BTreeMap<(String, Vec<u64>), u32> = BTreeMap::new();
        let assignments: Vec<(EntityId, u32)> = stroke_styles
            .into_iter()
            .map(|(id, def)| {
                // The name is part of the key, so the text anchors, the clip rings and the fill rings
                // (all hidden) each keep a style of their own.
                let key = [u64::from(def.visible), u64::from(def.width.is_some())]
                    .into_iter()
                    .chain(std::iter::once(match def.width {
                        Some(StrokeWidth::Width(w)) => w.to_bits(),
                        _ => 0,
                    }))
                    .chain(std::iter::once(def.phase.to_bits()))
                    .chain(def.dashes.iter().map(|d| d.to_bits()))
                    .collect::<Vec<u64>>();
                let index = *style_index
                    .entry((def.name.clone(), key))
                    .or_insert_with(|| {
                        styles.push(def);
                        (styles.len() - 1) as u32
                    });
                (id, index)
            })
            .collect();
        let line_style = LineStyleStream {
            styles,
            assignments,
        };
        writer
            .push_stream(line_style.into_stream(&registry).map_err(|e| refused(&e))?)
            .map_err(|e| refused(&e))?;
    }
    for stream in log.streams(&registry).map_err(PdfToPvecError::Write)? {
        writer
            .push_stream(stream)
            .map_err(|error| PdfToPvecError::Write(error.to_string()))?;
    }
    writer
        .finish()
        .map_err(|error| PdfToPvecError::Write(error.to_string()))
}

/// The subpaths of a path in PDF points, curves flattened to 16 steps, each with whether an `h`
/// closed it.
pub(super) fn subpaths(entity: &PdfPathEntity) -> Vec<(Vec<[f64; 2]>, bool)> {
    fn cubic_points(start: [f64; 2], c1: [f64; 2], c2: [f64; 2], end: [f64; 2]) -> Vec<[f64; 2]> {
        (0..=16)
            .map(|step| {
                let t = step as f64 / 16.0;
                let u = 1.0 - t;
                [
                    u * u * u * start[0]
                        + 3.0 * u * u * t * c1[0]
                        + 3.0 * u * t * t * c2[0]
                        + t * t * t * end[0],
                    u * u * u * start[1]
                        + 3.0 * u * u * t * c1[1]
                        + 3.0 * u * t * t * c2[1]
                        + t * t * t * end[1],
                ]
            })
            .collect()
    }

    let mut subpaths: Vec<(Vec<[f64; 2]>, bool)> = Vec::new();
    for segment in &entity.segments {
        match *segment {
            PdfPathSegment::MoveTo(point) => subpaths.push((vec![point], false)),
            PdfPathSegment::LineTo(point) => {
                if let Some((points, _)) = subpaths.last_mut() {
                    points.push(point);
                }
            }
            PdfPathSegment::CurveTo { c1, c2, end } => {
                if let Some((points, _)) = subpaths.last_mut() {
                    if let Some(&start) = points.last() {
                        points.extend(cubic_points(start, c1, c2, end).into_iter().skip(1));
                    }
                }
            }
            PdfPathSegment::Close => {
                if let Some((_, closed)) = subpaths.last_mut() {
                    *closed = true;
                }
            }
        }
    }
    subpaths
}

/// The bounds of a polyline and the cell the centre of its bounds falls in on a 16 by 16 grid over
/// the page.
fn bounds_and_cell(points: &[[f64; 2]], page: Bounds) -> (Bounds, u32) {
    let first = points[0];
    let entity_bounds = points.iter().fold(
        Bounds {
            min: first,
            max: first,
        },
        |bounds, point| Bounds {
            min: [bounds.min[0].min(point[0]), bounds.min[1].min(point[1])],
            max: [bounds.max[0].max(point[0]), bounds.max[1].max(point[1])],
        },
    );
    let center_x = (entity_bounds.min[0] + entity_bounds.max[0]) / 2.0;
    let center_y = (entity_bounds.min[1] + entity_bounds.max[1]) / 2.0;
    let column = (center_x / page.max[0] * 16.0).floor().clamp(0.0, 15.0) as u32;
    let row = (center_y / page.max[1] * 16.0).floor().clamp(0.0, 15.0) as u32;
    (entity_bounds, row * 16 + column + 1)
}

pub(super) fn source_point(point: [f64; 2], page_height: f64) -> [f64; 2] {
    [point[0] / 72.0, (page_height - point[1]) / 72.0]
}
