//! Issue #1615: the image report says whether the page was baked, and `pdf_page_to_pvec_report`
//! hands it back with the bytes. Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::cad::CadSource;
use libviprs::pvec::pdf_image::{PdfImageOptions, bake_page_images};
use libviprs::pvec::{pdf_page_to_pvec_report, pdf_page_to_pvec_with};
use libviprs::pvec_convert::PdfConversion;

const RULE: &str = "10 10 m 100 10 l S ";
const IMAGE: &str = "q 100 0 0 50 50 200 cm /Im1 Do Q";

fn with_image() -> Vec<u8> {
    one(gray8(4, 4, &[7u8; 16]), &format!("{RULE}{IMAGE}"))
}

#[test]
fn the_report_of_a_baked_page_is_baked() {
    let (_, report) = pdf_page_to_pvec_report(&page(&with_image()), "test", "0", true).unwrap();
    assert_eq!((report.images, report.written), (1, 1));
    assert!(report.baked());
    assert_eq!(report.not_baked, None);
}

#[test]
fn a_page_asked_not_to_bake_reports_not_baked() {
    let (bytes, report) =
        pdf_page_to_pvec_report(&page(&with_image()), "test", "0", false).unwrap();
    assert!(!report.baked());
    assert_eq!(report.not_baked.as_deref(), Some("not-requested"));
    assert_eq!(
        bytes,
        pdf_page_to_pvec_with(&page(&with_image()), "test", "0", false).unwrap()
    );
}

#[test]
fn a_page_with_no_image_is_baked_whatever_was_asked() {
    let bytes = pdf(doc(), Resources::images(Vec::new()), RULE.as_bytes());
    for bake in [true, false] {
        let (_, report) = pdf_page_to_pvec_report(&page(&bytes), "test", "0", bake).unwrap();
        assert!(report.baked(), "bake={bake}");
    }
}

#[test]
fn over_max_images_nothing_is_baked_and_the_limit_is_named() {
    let mut options = PdfImageOptions::default();
    options.max_images = 0;
    let bake = bake_page_images(&page(&with_image()), &options);
    assert!(bake.written.is_empty());
    assert!(!bake.report.baked());
    assert_eq!(bake.report.not_baked.as_deref(), Some("max-images"));
}

#[test]
fn over_max_image_bytes_nothing_is_baked_and_the_limit_is_named() {
    let mut options = PdfImageOptions::default();
    options.max_image_bytes = 1;
    let bake = bake_page_images(&page(&with_image()), &options);
    assert!(bake.written.is_empty());
    assert_eq!(bake.report.not_baked.as_deref(), Some("max-image-bytes"));
}

#[test]
fn within_the_limits_the_limit_fields_stay_out_of_the_way() {
    let bake = bake_page_images(&page(&with_image()), &PdfImageOptions::default());
    assert_eq!(bake.report.written, 1);
    assert_eq!(bake.report.not_baked, None);
}

#[test]
fn over_the_image_limit_the_report_names_the_limit() {
    let bytes = one(gray8(4, 4, &[7u8; 16]), &format!("{RULE}{IMAGE}"));
    let mut options = PdfImageOptions::default();
    options.max_images = 0;
    let json = {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pvec");
        let out = PdfConversion::new(CadSource::Bytes(&bytes), &dest)
            .image_options(options)
            .convert()
            .unwrap();
        serde_json::from_str::<serde_json::Value>(&out.report.to_json_string().unwrap()).unwrap()
    };
    assert_eq!(json["pdf"]["rasters"], 0);
    assert_eq!(json["pdf"]["images_baked"], false, "{json}");
    assert_eq!(json["pdf"]["images_not_baked_reason"], "max-images");
}
