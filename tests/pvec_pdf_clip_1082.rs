//! The PDF clipping and compound-path policy (N5.3, #1082), over fixtures built in memory with
//! `lopdf` so the content streams under test are visible in the test itself.

use std::collections::HashSet;

use libviprs::pvec::fidelity::FidelityClass;
use libviprs::pvec::limits::SourceLimits;
use libviprs::pvec::pdf_clip::{
    ClipCase, ClipConformance, ClipError, ClipLimits, ClipReport, ExactBounds, ExactBoundsSource,
    LinesOnlyBounds, apply_clip_policy,
};
use libviprs::pvec::pdf_vector::{
    PdfFillRule, PdfPathEntity, PdfPathSegment, PdfVectorLimits, PdfVectorPage,
    extract_page_paths_from_bytes,
};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

fn pdf(body: &str, rotate: Option<i64>, soft_mask: bool) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 200.into(), 100.into()],
        "Contents" => content,
    };
    if let Some(r) = rotate {
        page.set("Rotate", r);
    }
    if soft_mask {
        let group = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Form",
                "BBox" => vec![0.into(), 0.into(), 200.into(), 100.into()],
            },
            b"0 0 10 10 re f".to_vec(),
        ));
        page.set(
            "Resources",
            dictionary! {
                "ExtGState" => dictionary! {
                    "GS1" => dictionary! {
                        "Type" => "ExtGState",
                        "SMask" => dictionary! {
                            "Type" => "Mask", "S" => "Alpha", "G" => group,
                        },
                    },
                    "GS0" => dictionary! { "Type" => "ExtGState", "SMask" => "None" },
                },
            },
        );
    }
    let page_id = doc.add_object(page);
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn page_of(bytes: &[u8]) -> PdfVectorPage {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).unwrap()
}

fn report(body: &str) -> ClipReport {
    let page = page_of(&pdf(body, None, false));
    apply_clip_policy(
        &page,
        &ClipLimits::from(&SourceLimits::default()),
        &LinesOnlyBounds,
    )
    .unwrap()
}

/// The decision for the last painted entity.
fn last(r: &ClipReport) -> &libviprs::pvec::pdf_clip::EntityClipDecision {
    r.decisions.last().expect("a painted entity")
}

/// A source that knows a fixed box for every entity with a curve.
struct Fixed(ExactBounds);
impl ExactBoundsSource for Fixed {
    fn exact_bounds(&self, e: &PdfPathEntity) -> Option<ExactBounds> {
        e.segments
            .iter()
            .any(|s| matches!(s, PdfPathSegment::CurveTo { .. }))
            .then_some(self.0)
    }
}

const NO_CLIP: &str = "10 10 50 20 re f";
const RECT: &str = "0 0 100 50 re W n 10 10 20 20 re f";
const TRIANGLE: &str = "0 0 m 100 0 l 50 80 l h W n 10 10 5 5 re f";
const OUTSIDE: &str = "0 0 50 50 re W n 100 100 10 10 re f";

#[test]
fn n5_3_no_clip_is_recorded_as_no_clip() {
    let r = report(NO_CLIP);
    assert_eq!(r.decisions.len(), 1);
    assert_eq!(last(&r).case, ClipCase::NoClip);
    assert!(last(&r).clips.is_empty());
    assert!(r.clips.is_empty());
}

#[test]
fn n5_3_axis_aligned_rectangle_is_an_exact_clip_and_keeps_the_geometry() {
    let r = report(RECT);
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
    assert_eq!(
        last(&r).conformance,
        Some(ClipConformance::GeometricallyExact)
    );
    assert!(last(&r).classification.is_none());
    assert_eq!(r.clips.len(), 1);
    assert_eq!(
        r.clips[0].rect,
        Some(ExactBounds {
            min: [0.0, 0.0],
            max: [100.0, 50.0]
        })
    );
    assert_eq!(r.clips[0].rule, PdfFillRule::NonZero);
}

#[test]
fn n5_3_rectangle_drawn_with_m_l_h_is_still_rectangular() {
    let r = report("0 0 m 100 0 l 100 50 l 0 50 l h W n 10 10 20 20 re f");
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
}

#[test]
fn n5_3_rectangle_stays_rectangular_under_page_rotation() {
    let page = page_of(&pdf(RECT, Some(90), false));
    let r = apply_clip_policy(
        &page,
        &ClipLimits::from(&SourceLimits::default()),
        &LinesOnlyBounds,
    )
    .unwrap();
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
}

#[test]
fn n5_3_a_tilted_rectangle_is_not_axis_aligned() {
    let r = report("0.866 0.5 -0.5 0.866 50 0 cm 0 0 40 20 re W n 5 5 5 5 re f");
    assert_eq!(last(&r).case, ClipCase::BoundedConservative);
}

#[test]
fn n5_3_general_path_is_bounded_conservative_and_never_exact() {
    let r = report(TRIANGLE);
    assert_eq!(last(&r).case, ClipCase::BoundedConservative);
    assert_eq!(last(&r).conformance, Some(ClipConformance::DisplayFidelity));
    assert_eq!(r.clips[0].conformance, ClipConformance::DisplayFidelity);
    assert_eq!(r.clips[0].rect, None);
    // The geometry is kept, unclipped.
    assert!(last(&r).classification.is_none());
}

#[test]
fn n5_3_no_bounded_conservative_record_is_labelled_exact() {
    for body in [
        TRIANGLE,
        "0 0 m 50 100 100 100 100 0 c h W n 10 10 5 5 re f",
    ] {
        let r = report(body);
        for c in &r.clips {
            assert_ne!(c.conformance, ClipConformance::GeometricallyExact);
        }
        for d in &r.decisions {
            assert_ne!(d.conformance, Some(ClipConformance::GeometricallyExact));
        }
    }
}

#[test]
fn n5_3_entity_outside_the_clip_is_dropped_by_policy_with_a_reason() {
    let r = report(OUTSIDE);
    let d = last(&r);
    assert_eq!(d.case, ClipCase::FullyClipped);
    let c = d.classification.as_ref().expect("a classification");
    assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
    assert_eq!(c.policy_field(), Some("OutsideClip"));
}

#[test]
fn n5_3_a_stroked_entity_outside_the_path_extent_is_not_dropped() {
    // Stroke width is not known here, so the path extent says nothing about the painted extent.
    let r = report("0 0 50 50 re W n 55 10 m 55 40 l S");
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
}

#[test]
fn n5_3_a_curve_is_never_dropped_on_its_control_hull() {
    // The hull reaches y = 80, outside the clip (y <= 70). The exact extent stops at 62.5.
    let body = "0 0 100 70 re W n 10 10 m 10 80 90 80 90 10 c f";
    let page = page_of(&pdf(body, None, false));
    let limits = ClipLimits::from(&SourceLimits::default());
    // No source for curves: retained.
    let r = apply_clip_policy(&page, &limits, &LinesOnlyBounds).unwrap();
    assert_ne!(last(&r).case, ClipCase::FullyClipped);
    // A source whose exact extent is inside: retained.
    let inside = Fixed(ExactBounds {
        min: [10.0, 10.0],
        max: [90.0, 62.5],
    });
    let r = apply_clip_policy(&page, &limits, &inside).unwrap();
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
    // A source whose exact extent is outside: dropped.
    let outside = Fixed(ExactBounds {
        min: [110.0, 10.0],
        max: [190.0, 60.0],
    });
    let r = apply_clip_policy(&page, &limits, &outside).unwrap();
    assert_eq!(last(&r).case, ClipCase::FullyClipped);
}

#[test]
fn n5_3_a_soft_mask_makes_the_entity_unsupported_with_a_report_row() {
    let page = page_of(&pdf("/GS1 gs 10 10 20 20 re f", None, true));
    let r = apply_clip_policy(
        &page,
        &ClipLimits::from(&SourceLimits::default()),
        &LinesOnlyBounds,
    )
    .unwrap();
    let d = last(&r);
    assert_eq!(d.case, ClipCase::Unsupported);
    let c = d.classification.as_ref().expect("a classification");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert!(c.reason().is_some());
}

#[test]
fn n5_3_clearing_the_soft_mask_or_restoring_the_state_ends_it() {
    let page = page_of(&pdf("/GS1 gs /GS0 gs 10 10 5 5 re f", None, true));
    let limits = ClipLimits::from(&SourceLimits::default());
    let r = apply_clip_policy(&page, &limits, &LinesOnlyBounds).unwrap();
    assert_eq!(last(&r).case, ClipCase::NoClip);
    let page = page_of(&pdf("q /GS1 gs 1 1 5 5 re f Q 10 10 5 5 re f", None, true));
    let r = apply_clip_policy(&page, &limits, &LinesOnlyBounds).unwrap();
    assert_eq!(r.decisions[0].case, ClipCase::Unsupported);
    assert_eq!(r.decisions[1].case, ClipCase::NoClip);
}

#[test]
fn n5_3_every_case_is_produced_by_a_committed_fixture() {
    let page = page_of(&pdf("/GS1 gs 10 10 20 20 re f", None, true));
    let limits = ClipLimits::from(&SourceLimits::default());
    let mut seen: HashSet<ClipCase> = HashSet::new();
    for body in [NO_CLIP, RECT, TRIANGLE, OUTSIDE] {
        seen.extend(report(body).decisions.iter().map(|d| d.case));
    }
    seen.extend(
        apply_clip_policy(&page, &limits, &LinesOnlyBounds)
            .unwrap()
            .decisions
            .iter()
            .map(|d| d.case),
    );
    for case in ClipCase::ALL {
        assert!(seen.contains(&case), "no fixture produces {case:?}");
    }
    assert_eq!(ClipCase::ALL.len(), 5);
}

#[test]
fn n5_3_the_clip_count_equals_the_recorded_decisions() {
    let body = "q 0 0 100 100 re W n q 10 10 50 50 re W* n 20 20 5 5 re f Q \
                0 0 m 50 0 l 25 40 l h W n 30 30 5 5 re f Q 40 40 5 5 re f";
    let page = page_of(&pdf(body, None, false));
    let in_stream = page.entities.iter().filter(|e| e.clip.is_some()).count();
    assert_eq!(in_stream, 3);
    let r = apply_clip_policy(
        &page,
        &ClipLimits::from(&SourceLimits::default()),
        &LinesOnlyBounds,
    )
    .unwrap();
    assert_eq!(r.clips.len(), in_stream);
    // Every clip a decision names has a record.
    let known: HashSet<u64> = r.clips.iter().map(|c| c.sequence).collect();
    for d in &r.decisions {
        for c in &d.clips {
            assert!(known.contains(c));
        }
    }
}

#[test]
fn n5_3_q_and_q_scope_the_active_clips() {
    let body = "q 0 0 100 100 re W n q 10 10 50 50 re W n 20 20 5 5 re f Q 30 30 5 5 re f Q \
                40 40 5 5 re f";
    let r = report(body);
    let depths: Vec<usize> = r.decisions.iter().map(|d| d.clips.len()).collect();
    assert_eq!(depths, vec![2, 1, 0]);
    assert_eq!(r.max_depth, 2);
}

#[test]
fn n5_3_the_winding_rule_is_recorded_and_never_flattened() {
    let r = report("0 0 100 100 re 20 20 60 60 re W* n 50 50 5 5 re f");
    assert_eq!(r.clips[0].rule, PdfFillRule::EvenOdd);
    assert_eq!(r.clips[0].subpaths, 2);
    assert_eq!(last(&r).case, ClipCase::BoundedConservative);

    let r = report("0 0 100 100 re 20 20 60 60 re W n 50 50 5 5 re f");
    assert_eq!(r.clips[0].rule, PdfFillRule::NonZero);

    let r = report("0 0 100 100 re 20 20 60 60 re f*");
    assert_eq!(last(&r).fill_rule, Some(PdfFillRule::EvenOdd));
    assert_eq!(last(&r).subpaths, 2);
    let r = report("0 0 100 100 re 20 20 60 60 re f");
    assert_eq!(last(&r).fill_rule, Some(PdfFillRule::NonZero));
}

#[test]
fn n5_3_nesting_past_the_limit_is_a_typed_error_naming_limit_and_depth() {
    let body = "q 0 0 90 90 re W n q 0 0 80 80 re W n q 0 0 70 70 re W n 1 1 2 2 re f Q Q Q";
    let page = page_of(&pdf(body, None, false));
    let err =
        apply_clip_policy(&page, &ClipLimits { max_clip_depth: 2 }, &LinesOnlyBounds).unwrap_err();
    assert_eq!(
        err,
        ClipError::NestingExceeded {
            limit: 2,
            limit_name: "max_block_depth",
            observed: 3
        }
    );
    let text = err.to_string();
    assert!(
        text.contains("max_block_depth=2") && text.contains("depth 3"),
        "{text}"
    );
    // At the limit is fine.
    assert!(apply_clip_policy(&page, &ClipLimits { max_clip_depth: 3 }, &LinesOnlyBounds).is_ok());
}

#[test]
fn n5_3_the_depth_limit_comes_from_the_source_limits() {
    let l = SourceLimits::default();
    assert_eq!(ClipLimits::from(&l).max_clip_depth, l.max_block_depth);
}

#[test]
fn n5_3_the_policy_is_deterministic() {
    let a = report(TRIANGLE);
    let b = report(TRIANGLE);
    assert_eq!(a, b);
}
