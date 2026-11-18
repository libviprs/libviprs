//! Issue #1615: a conversion bakes the page's images by default, and the JSON report says whether
//! every image was baked (`pdf.images_baked`). Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::cad::CadSource;
use libviprs::pvec::pdf_image::PdfImageOptions;
use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec_convert::PdfConversion;
use lopdf::dictionary;
use pvec_facade::PvecReader;

const RULE: &str = "10 10 m 100 10 l S ";
const IMAGE: &str = "q 100 0 0 50 50 200 cm /Im1 Do Q";

fn report_json(bytes: &[u8], options: Option<PdfImageOptions>) -> serde_json::Value {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let mut conversion = PdfConversion::new(CadSource::Bytes(bytes), &dest);
    if let Some(options) = options {
        conversion = conversion.image_options(options);
    }
    let out = conversion.convert().unwrap();
    serde_json::from_str(&out.report.to_json_string().unwrap()).unwrap()
}

#[test]
fn pdf_page_to_pvec_bakes_the_image_without_being_asked() {
    let bytes = one(gray8(4, 4, &[7u8; 16]), &format!("{RULE}{IMAGE}"));
    let artifact = pdf_page_to_pvec(&page(&bytes), "test", "0").unwrap();
    let placements = PvecReader::open(artifact)
        .unwrap()
        .raster_placements()
        .unwrap();
    assert_eq!(placements.len(), 1, "the image was left in the pyramid");
}

#[test]
fn the_baked_image_keeps_its_own_resolution() {
    let bytes = one(gray8(4, 4, &[7u8; 16]), &format!("{RULE}{IMAGE}"));
    let artifact = pdf_page_to_pvec(&page(&bytes), "test", "0").unwrap();
    let placements = PvecReader::open(artifact)
        .unwrap()
        .raster_placements()
        .unwrap();
    assert_eq!(
        placements.first().map(|p| (p.width, p.height)),
        Some((4, 4))
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_says_the_images_were_baked() {
    let bytes = one(gray8(4, 4, &[7u8; 16]), &format!("{RULE}{IMAGE}"));
    let json = report_json(&bytes, None);
    assert_eq!(json["pdf"]["images"], 1);
    assert_eq!(json["pdf"]["rasters"], 1);
    assert_eq!(json["pdf"]["images_baked"], true, "{json}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_page_with_no_image_counts_as_baked() {
    let bytes = pdf(doc(), Resources::images(Vec::new()), RULE.as_bytes());
    assert_eq!(report_json(&bytes, None)["pdf"]["images_baked"], true);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_refused_image_means_the_page_is_not_baked() {
    let jpx = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "JPXDecode" },
        vec![1, 2, 3],
    );
    let bytes = one(jpx, &format!("{RULE}{IMAGE}"));
    let json = report_json(&bytes, None);
    assert_eq!(json["pdf"]["rasters"], 0);
    assert_eq!(json["pdf"]["images_baked"], false, "{json}");
}
