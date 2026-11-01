//! Issue #1332 (RL2.1): the report counts every outcome, and the images on the page are exactly
//! the images written plus the images refused.

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageOptions, bake_page_images};
use lopdf::{Object, dictionary};

fn jpeg_16() -> Vec<u8> {
    let mut out = Vec::new();
    let data = vec![100u8; 16 * 16 * 3];
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90)
        .encode(&data, 16, 16, image::ExtendedColorType::Rgb8)
        .unwrap();
    out
}

#[test]
fn a_page_with_no_image_reports_zeros() {
    let bytes = pdf(doc(), Resources::images(Vec::new()), b"0 0 10 10 re f");
    let b = bake_page_images(&page(&bytes), &PdfImageOptions::default());
    assert_eq!(b.report.images, 0);
    assert_eq!(b.report.written, 0);
    assert_eq!(b.report.refused_total(), 0);
    assert_eq!(
        (
            b.report.colour_approximated,
            b.report.bit_depth_reduced,
            b.report.re_encoded_lossy,
            b.report.interleaved,
            b.report.clipped
        ),
        (0, 0, 0, 0, 0)
    );
    assert!(b.written.is_empty() && b.refused.is_empty());
}

#[test]
fn the_counts_per_outcome_add_up_to_the_images_on_the_page() {
    let mut d = doc();
    let icc = d.add_object(lopdf::Stream::new(dictionary! { "N" => 1 }, vec![0u8; 4]));
    let icc_gray = image(
        1,
        1,
        dictionary! {
            "ColorSpace" => vec![name("ICCBased"), Object::Reference(icc)],
            "BitsPerComponent" => 8,
        },
        vec![9],
    );
    let wide = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 16 },
        vec![0x12, 0x34],
    );
    let jpx = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "JPXDecode" },
        vec![1, 2, 3],
    );
    let jbig = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "JBIG2Decode" },
        vec![1, 2, 3],
    );
    let dct = image(
        16,
        16,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode" },
        jpeg_16(),
    );
    let bytes = pdf(
        d,
        Resources::images(vec![
            ("Gray", Object::Stream(gray8(1, 1, &[5]))),
            ("Icc", Object::Stream(icc_gray)),
            ("Wide", Object::Stream(wide)),
            ("Jpx", Object::Stream(jpx)),
            ("Jbig", Object::Stream(jbig)),
            ("Dct", Object::Stream(dct)),
        ]),
        b"0 0 3 3 re f \
          q 4 0 0 4 0 0 cm /Gray Do Q q 4 0 0 4 10 0 cm /Icc Do Q \
          q 4 0 0 4 20 0 cm /Wide Do Q q 4 0 0 4 30 0 cm /Jpx Do Q \
          q 4 0 0 4 40 0 cm /Jbig Do Q q 4 0 0 4 50 0 cm /Dct Do Q \
          5 5 3 3 re f",
    );
    let p = page(&bytes);
    let b = bake_page_images(&p, &PdfImageOptions::default());
    let r = &b.report;
    assert_eq!(r.images, p.images.len() as u64);
    assert_eq!(r.images, 6);
    assert_eq!(r.written, 4);
    assert_eq!(r.images, r.written + r.refused_total());
    // With `jp2k` the three-byte JPX stream is tried and refused (#1364), without it never tried.
    let jpx_reason = if cfg!(feature = "jp2k") {
        "decode-failed"
    } else {
        "filter-not-enabled:JPXDecode"
    };
    assert_eq!(r.refused.get(jpx_reason), Some(&1));
    assert_eq!(r.refused.get("filter-not-enabled:JBIG2Decode"), Some(&1));
    assert_eq!(r.colour_approximated, 1);
    assert_eq!(r.bit_depth_reduced, 1);
    assert_eq!(r.re_encoded_lossy, 1);
    assert_eq!(
        r.interleaved, 4,
        "every written image sits between the two paths"
    );
    assert_eq!(b.written.len() + b.refused.len(), p.images.len());
    // Written and refused together cover each index once, in content order.
    let mut idx: Vec<usize> = b
        .written
        .iter()
        .map(|w| w.index)
        .chain(b.refused.iter().map(|r| r.index))
        .collect();
    idx.sort_unstable();
    assert_eq!(idx, (0..6).collect::<Vec<_>>());
}

#[test]
fn every_refusal_keeps_its_source_chain() {
    use std::error::Error;
    let jbig2 = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "JBIG2Decode" },
        vec![1],
    );
    let b = bake_page_images(
        &page(&one(jbig2, "q 4 0 0 4 0 0 cm /Im1 Do Q")),
        &PdfImageOptions::default(),
    );
    let e = &b.refused[0].error;
    assert!(e.to_string().contains("JBIG2Decode"));
    // A refusal that wraps another error exposes it; a leaf has no source and says so.
    assert!(e.source().is_none());

    // A tile edge PVec does not accept is the pyramid's refusal, with its error as the source.
    let mut o = PdfImageOptions::default();
    o.tile_edge = 300;
    let b = bake_page_images(
        &page(&one(gray8(1, 1, &[1]), "q 4 0 0 4 0 0 cm /Im1 Do Q")),
        &o,
    );
    let e = &b.refused[0].error;
    assert!(e.source().is_some(), "{e:?}");
}
