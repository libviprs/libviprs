//! Issue #1380 (N5.3b, slice of #1086 and #1282): exact bounds in the PDF clip policy, the
//! Form XObject `/BBox` as a clip, text clipping (`Tr` 4 to 7) and the empty clip path.
//!
//! The pages are built in memory with `lopdf`, so the content streams under test are in the
//! test itself. Needs `--features pvec-normalized`.
#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::fidelity::FidelityClass;
use libviprs::pvec::limits::SourceLimits;
use libviprs::pvec::pdf_clip::{
    AnalyticBounds, ClipCase, ClipConformance, ClipLimits, ClipReport, ExactBoundsSource,
    LinesOnlyBounds, apply_clip_policy,
};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// A page whose content is `body`, with Form XObjects `F1` (`bbox`, optional `/Matrix`) and `F2`
/// (the same, drawing a 10 by 10 square at 100,100) in its resources.
fn pdf(body: &str, form_matrix: Option<Vec<Object>>) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let mut f1 = dictionary! {
        "Type" => "XObject", "Subtype" => "Form",
        "BBox" => vec![0.into(), 0.into(), 50.into(), 50.into()],
    };
    if let Some(m) = form_matrix {
        f1.set("Matrix", m);
    }
    // Two squares: one inside the BBox, one outside it.
    let form = doc.add_object(Stream::new(
        f1,
        b"10 10 10 10 re f 100 100 10 10 re f".to_vec(),
    ));
    let page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 300.into(), 300.into()],
        "Contents" => content,
        "Resources" => dictionary! { "XObject" => dictionary! { "F1" => form } },
    };
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

fn page(body: &str) -> PdfVectorPage {
    extract_page_paths_from_bytes(&pdf(body, None), 1, &PdfVectorLimits::default()).unwrap()
}

fn report_with(page: &PdfVectorPage, bounds: &dyn ExactBoundsSource) -> ClipReport {
    apply_clip_policy(page, &ClipLimits::from(&SourceLimits::default()), bounds).unwrap()
}

fn report(body: &str) -> ClipReport {
    report_with(&page(body), &AnalyticBounds)
}

fn last(r: &ClipReport) -> &libviprs::pvec::pdf_clip::EntityClipDecision {
    r.decisions.last().expect("a painted entity")
}

const CLIP: &str = "0 0 100 100 re W n";

// ---- analytic bounds ----------------------------------------------------------------------

/// x(t) of this cubic never goes below 104.25, so it lies wholly right of the clip, yet its
/// control polygon reaches x = 99 and overlaps the clip.
const HULL_OVERLAPS_EXTENT_DOES_NOT: &str = "120 10 m 99 30 99 70 120 90 c f";
/// y(t) peaks at 92.5 inside the clip, while its control points reach y = 120 outside it.
const EXTENT_INSIDE_HULL_OUTSIDE: &str = "10 10 m 10 120 90 120 90 10 c f";

#[test]
fn a_curve_whose_hull_overlaps_the_clip_but_whose_extent_does_not_is_fully_clipped() {
    let p = page(&format!("{CLIP} {HULL_OVERLAPS_EXTENT_DOES_NOT}"));
    let exact = report_with(&p, &AnalyticBounds);
    assert_eq!(last(&exact).case, ClipCase::FullyClipped);
    let c = last(&exact)
        .classification
        .as_ref()
        .expect("classification");
    assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
    assert_eq!(c.policy_field(), Some("OutsideClip"));
    // The lines-only source cannot say for a curve, so it keeps the entity.
    let lines = report_with(&p, &LinesOnlyBounds);
    assert_ne!(last(&lines).case, ClipCase::FullyClipped);
}

#[test]
fn a_curve_whose_extent_is_inside_the_clip_is_kept_though_its_hull_leaves_it() {
    let r = report(&format!("{CLIP} {EXTENT_INSIDE_HULL_OUTSIDE}"));
    assert_eq!(last(&r).case, ClipCase::ExactRectangular);
    assert!(last(&r).classification.is_none());
}

#[test]
fn analytic_bounds_are_the_closed_form_extremum_not_a_sample() {
    let p = page(HULL_OVERLAPS_EXTENT_DOES_NOT);
    let b = AnalyticBounds.exact_bounds(&p.entities[0]).expect("bounds");
    assert!((b.min[0] - 104.25).abs() < 1e-12, "{b:?}");
    assert_eq!(b.max[0], 120.0);
    assert_eq!(b.min[1], 10.0);
    assert_eq!(b.max[1], 90.0);
}

#[test]
fn analytic_bounds_cover_straight_paths_like_the_lines_only_source() {
    let p = page("5 6 m 25 6 l 25 16 l h f");
    assert_eq!(
        AnalyticBounds.exact_bounds(&p.entities[0]),
        LinesOnlyBounds.exact_bounds(&p.entities[0])
    );
}

// ---- Form XObject /BBox -------------------------------------------------------------------

#[test]
fn a_form_bbox_is_recorded_as_a_clip_and_the_square_outside_it_is_fully_clipped() {
    let r = report("q /F1 Do Q");
    assert_eq!(r.form_clips.len(), 1, "{:?}", r.form_clips);
    let f = &r.form_clips[0];
    assert_eq!(f.form_path, vec!["F1".to_string()]);
    assert_eq!(f.conformance, ClipConformance::GeometricallyExact);
    assert_eq!(
        f.rect.map(|b| (b.min, b.max)),
        Some(([0.0, 0.0], [50.0, 50.0]))
    );
    assert_eq!(r.decisions.len(), 2);
    assert_eq!(r.decisions[0].case, ClipCase::ExactRectangular);
    assert_eq!(r.decisions[0].form_clips, vec![0]);
    assert_eq!(r.decisions[1].case, ClipCase::FullyClipped);
}

#[test]
fn a_rotated_form_bbox_is_a_display_fidelity_clip_and_keeps_the_geometry() {
    let (c, s) = (0.866_025_4_f32, 0.5_f32);
    let m = vec![
        c.into(),
        s.into(),
        (-s).into(),
        c.into(),
        0.into(),
        0.into(),
    ];
    let p =
        extract_page_paths_from_bytes(&pdf("q /F1 Do Q", Some(m)), 1, &PdfVectorLimits::default())
            .unwrap();
    let r = report_with(&p, &AnalyticBounds);
    assert_eq!(r.form_clips.len(), 1);
    assert!(
        r.form_clips[0].rect.is_none(),
        "a rotated rectangle is not axis aligned"
    );
    assert_eq!(
        r.form_clips[0].conformance,
        ClipConformance::DisplayFidelity
    );
    assert_eq!(r.decisions[0].case, ClipCase::BoundedConservative);
}

#[test]
fn the_form_clip_does_not_leak_to_what_the_page_paints_after_the_form() {
    let r = report("q /F1 Do Q 150 150 10 10 re f");
    let after = last(&r);
    assert_eq!(after.case, ClipCase::NoClip);
    assert!(after.form_clips.is_empty());
}

#[test]
fn two_invocations_of_one_form_are_two_clips() {
    let r = report("q /F1 Do Q q 1 0 0 1 100 100 cm /F1 Do Q");
    assert_eq!(r.form_clips.len(), 2);
    assert_ne!(r.form_clips[0].rect, r.form_clips[1].rect);
    // The second square of the second invocation lands at 200,200, outside its clip at 100..150.
    assert_eq!(r.decisions.len(), 4);
    assert_eq!(r.decisions[2].form_clips, vec![1]);
}

// ---- text clipping ------------------------------------------------------------------------

#[test]
fn text_render_mode_clip_makes_later_geometry_display_fidelity_only() {
    for mode in 4..=7 {
        let r = report(&format!("BT {mode} Tr (A) Tj ET 10 10 20 20 re f"));
        assert!(last(&r).text_clip, "Tr {mode}");
        assert_eq!(last(&r).case, ClipCase::BoundedConservative, "Tr {mode}");
        assert_eq!(
            last(&r).conformance,
            Some(ClipConformance::DisplayFidelity),
            "Tr {mode}"
        );
    }
}

#[test]
fn text_render_modes_without_clip_leave_geometry_alone() {
    for mode in 0..=3 {
        let r = report(&format!("BT {mode} Tr (A) Tj ET 10 10 20 20 re f"));
        assert!(!last(&r).text_clip, "Tr {mode}");
        assert_eq!(last(&r).case, ClipCase::NoClip, "Tr {mode}");
    }
}

#[test]
fn a_text_clip_ends_with_the_graphics_state_that_made_it() {
    let r = report("q BT 7 Tr (A) Tj ET 10 10 20 20 re f Q 40 40 20 20 re f");
    assert!(r.decisions[0].text_clip);
    assert!(!r.decisions[1].text_clip);
    assert_eq!(r.decisions[1].case, ClipCase::NoClip);
}

// ---- empty clip path ----------------------------------------------------------------------

#[test]
fn an_empty_path_under_w_n_clips_everything_after_it() {
    let r = report("W n 10 10 20 20 re f");
    assert_eq!(r.clips.len(), 1, "the empty clip is seen");
    assert!(r.clips[0].empty);
    assert_eq!(r.clips[0].conformance, ClipConformance::GeometricallyExact);
    assert_eq!(last(&r).case, ClipCase::FullyClipped);
}

#[test]
fn a_nonempty_clip_is_not_marked_empty() {
    let r = report(&format!("{CLIP} 10 10 20 20 re f"));
    assert!(!r.clips[0].empty);
}
