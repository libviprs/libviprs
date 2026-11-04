//! Issue #1490: the fill report must not say a PDF image is not placed when the bake places it.
//!
//! Since #1332 `bake_page_images` writes a placement for every image it can decode, yet
//! `FILL_POLICY` still reports the image constructs as `*-not-placed`. Needs
//! `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::fill_policy::{FillConstruct, FillReport, policy_for};
use libviprs::pvec::pdf_image::{PdfImageOptions, bake_page_images};

fn written_one() -> (Vec<u8>, usize) {
    let bytes = one(rgb8(2, 2, &[9u8; 12]), "q 20 0 0 20 5 5 cm /Im1 Do Q");
    let bake = bake_page_images(&page(&bytes), &PdfImageOptions::default());
    let n = bake.written.len();
    (bytes, n)
}

#[test]
fn the_bake_places_the_image_this_test_is_about() {
    let (_, written) = written_one();
    assert_eq!(written, 1);
}

#[test]
fn no_pdf_image_row_says_not_placed() {
    for c in [
        FillConstruct::PdfImageXObject,
        FillConstruct::PdfInlineImage,
    ] {
        let reason = policy_for(c).reason;
        assert!(
            !reason.contains("not-placed"),
            "{c:?} is reported `{reason}` but the raster bake places it"
        );
    }
}

#[test]
fn the_report_of_a_placed_image_does_not_claim_otherwise() {
    let (bytes, written) = written_one();
    let mut report = FillReport::new();
    report.count_pdf_page(&page(&bytes));
    assert_eq!(report.count(FillConstruct::PdfImageXObject), written as u64);
    for row in report.rows() {
        assert!(
            !row.policy.reason.contains("not-placed"),
            "{:?}: {}",
            row.construct,
            row.policy.reason
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_hatch_policy_doc_matches_the_code() {
    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/pvec-hatch-policy.md"),
    )
    .unwrap();
    for c in [
        FillConstruct::PdfImageXObject,
        FillConstruct::PdfInlineImage,
    ] {
        let reason = policy_for(c).reason;
        assert!(doc.contains(&format!("`{reason}`")), "doc lacks `{reason}`");
        assert!(!doc.contains("`image-pdf-xobject-not-placed`"));
        assert!(!doc.contains("`image-pdf-inline-not-placed`"));
    }
    // The doc says what the rows do and do not know, and where the placed count is.
    assert!(doc.contains("counts every image"));
    assert!(doc.contains("`rasters`") && doc.contains("`images_refused`"));
}
