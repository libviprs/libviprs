//! The body of `fuzz_pvec_clip` (N7.6, #1437), shared with `tests/pvec_1437_fuzz_seeds.rs`.
//!
//! A vector page built from arbitrary paths: compound paths with any run of move, line, curve and
//! close, clips that are empty, self-intersecting, nested past the limit or point at sequences that
//! do not exist, Form XObject boxes with any corners, and coordinates that are NaN or infinite. It
//! is run through the clip policy with both bounds sources. The policy must return a report or a
//! typed error, never panic.

use arbitrary::{Result as AResult, Unstructured};
use libviprs::pvec::SourceLimits;
use libviprs::pvec::pdf_clip::{AnalyticBounds, ClipLimits, LinesOnlyBounds, apply_clip_policy};
use libviprs::pvec::pdf_vector::{
    PdfFillRule, PdfFormClip, PdfPaintIntent, PdfPathEntity, PdfPathSegment, PdfPoint,
    PdfProvenance, PdfStrokeParams, PdfVectorPage,
};

fn pt(u: &mut Unstructured<'_>) -> AResult<PdfPoint> {
    // Mostly sane page coordinates so clips and paths overlap, sometimes anything at all.
    if u.ratio(1, 8)? {
        Ok([u.arbitrary()?, u.arbitrary()?])
    } else {
        Ok([
            f64::from(u.int_in_range(0..=200u8)?),
            f64::from(u.int_in_range(0..=200u8)?),
        ])
    }
}

fn rule(u: &mut Unstructured<'_>) -> AResult<PdfFillRule> {
    Ok(if u.arbitrary()? {
        PdfFillRule::NonZero
    } else {
        PdfFillRule::EvenOdd
    })
}

fn entity(u: &mut Unstructured<'_>, sequence: u64) -> AResult<PdfPathEntity> {
    let n = u.int_in_range(0..=12usize)?;
    let mut segments = Vec::new();
    for _ in 0..n {
        segments.push(match u.int_in_range(0..=3)? {
            0 => PdfPathSegment::MoveTo(pt(u)?),
            1 => PdfPathSegment::LineTo(pt(u)?),
            2 => PdfPathSegment::CurveTo {
                c1: pt(u)?,
                c2: pt(u)?,
                end: pt(u)?,
            },
            _ => PdfPathSegment::Close,
        });
    }
    let k = u.int_in_range(0..=4usize)?;
    let mut active_clips = Vec::new();
    for _ in 0..k {
        // A sequence that exists, or one that does not.
        active_clips.push(u.int_in_range(0..=sequence + 2)?);
    }
    Ok(PdfPathEntity {
        segments,
        paint: PdfPaintIntent {
            fill: if u.arbitrary()? { Some(rule(u)?) } else { None },
            stroke: u.arbitrary()?,
            close: u.arbitrary()?,
        },
        clip: if u.ratio(1, 3)? { Some(rule(u)?) } else { None },
        active_clips,
        soft_masked: u.arbitrary()?,
        text_clipped: u.arbitrary()?,
        stroke_params: PdfStrokeParams::default(),
        provenance: PdfProvenance {
            page: 1,
            operator: "S".into(),
            form_path: Vec::new(),
            sequence,
        },
    })
}

fn build(u: &mut Unstructured<'_>) -> AResult<PdfVectorPage> {
    let n = u.int_in_range(0..=40u64)?;
    let mut entities = Vec::new();
    for s in 0..n {
        entities.push(entity(u, s)?);
    }
    let m = u.int_in_range(0..=4usize)?;
    let mut form_clips = Vec::new();
    for _ in 0..m {
        form_clips.push(PdfFormClip {
            form_path: vec!["Fm0".into()],
            corners: [pt(u)?, pt(u)?, pt(u)?, pt(u)?],
            first_sequence: u.int_in_range(0..=n + 2)?,
            end_sequence: u.int_in_range(0..=n + 2)?,
        });
    }
    Ok(PdfVectorPage {
        page: 1,
        width: 200.0,
        height: 200.0,
        rotation_degrees: 0,
        entities,
        form_clips,
        unbalanced_restores: 0,
        colour_operators: Default::default(),
        text: Default::default(),
        shading_paints: 0,
        pattern_selections: 0,
        images: Vec::new(),
    })
}

pub fn run(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let Ok(page) = build(&mut u) else { return };
    let mut limits = SourceLimits::default();
    limits.max_block_depth = u.int_in_range(0..=6u32).unwrap_or(4);
    let limits = ClipLimits::from(&limits);
    let _ = apply_clip_policy(&page, &limits, &LinesOnlyBounds);
    let _ = apply_clip_policy(&page, &limits, &AnalyticBounds);
}
