//! Issue #1362 (RL2.1 follow-up 1/6): the PDF filter seam, with no new codec.
//!
//! One `FilterInput` in, one `FilterOutput` out, one `match` that knows which build has which
//! codec. These tests pin what callers can see of it: the three codec names are named as
//! "not in this build" and no longer as "nobody decodes it", an unknown name is still refused
//! as before, `/DecodeParms` follows its filter by position, one pixel budget bounds every
//! codec, a per-stream byte cap sits beside the page total, and the report says which filter
//! produced each written image. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use std::error::Error;
use std::io::Write;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageError, PdfImageOptions, bake_page_images};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Object, dictionary};

fn bake(bytes: &[u8]) -> PdfImageBake {
    bake_page_images(&page(bytes), &PdfImageOptions::default())
}

fn flate(raw: &[u8]) -> Vec<u8> {
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(raw).unwrap();
    z.finish().unwrap()
}

fn jpeg(w: u32, h: u32) -> Vec<u8> {
    let mut src = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            src.extend_from_slice(&[(x % 251) as u8, (y % 241) as u8, 128]);
        }
    }
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80)
        .encode(&src, w, h, image::ExtendedColorType::Rgb8)
        .unwrap();
    out
}

fn refused(bytes: &[u8]) -> PdfImageError {
    let b = bake(bytes);
    assert!(b.written.is_empty(), "{:?}", b.written);
    assert_eq!(b.refused.len(), 1);
    b.refused.into_iter().next().unwrap().error
}

fn filtered(filter: Object, data: Vec<u8>) -> Vec<u8> {
    one(
        image(
            2,
            2,
            dictionary! {
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => filter,
            },
            data,
        ),
        "q 2 0 0 2 0 0 cm /Im1 Do Q",
    )
}

#[test]
fn a_2_mpx_flate_dct_image_under_a_1_mpx_budget_is_too_large_and_never_a_source_error() {
    let s = image(
        2000,
        1000,
        dictionary! {
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8,
            "Filter" => vec![name("FlateDecode"), name("DCTDecode")],
        },
        flate(&jpeg(2000, 1000)),
    );
    let bytes = one(s, "q 200 0 0 100 0 0 cm /Im1 Do Q");
    let mut o = PdfImageOptions::default();
    o.max_pixels = 1_000_000;
    let b = bake_page_images(&page(&bytes), &o);
    assert!(b.written.is_empty());
    let e = &b.refused[0].error;
    assert!(
        matches!(
            e,
            PdfImageError::TooLarge {
                pixels: 2_000_000,
                limit: 1_000_000
            }
        ),
        "{e:?}"
    );
    assert_eq!(b.report.refused.get("too-large"), Some(&1));
}

#[test]
fn the_pixel_budget_is_the_one_the_dct_decoder_runs_under() {
    // The JPEG is 2000 x 1000 but the dictionary says 10 x 10, so the declared size passes the
    // early check. The decoder must still be held to `max_pixels`, and must say so as a size
    // refusal or a decode failure of the filter, never as the decoder's own default budget.
    let s = image(
        10,
        10,
        dictionary! {
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode",
        },
        jpeg(2000, 1000),
    );
    let bytes = one(s, "q 10 0 0 10 0 0 cm /Im1 Do Q");
    let mut o = PdfImageOptions::default();
    o.max_pixels = 1_000_000;
    let b = bake_page_images(&page(&bytes), &o);
    assert!(b.written.is_empty());
    let e = &b.refused[0].error;
    assert!(
        matches!(e, PdfImageError::Filter { filter, .. } if filter == "DCTDecode"),
        "{e:?}"
    );
    assert_eq!(e.reason(), "decode-failed");
}

#[test]
fn the_three_codec_names_report_filter_not_enabled_and_the_error_names_the_feature() {
    let mut names = vec![("CCITTFaxDecode", "ccitt"), ("JBIG2Decode", "jbig2")];
    // With `jp2k` the JPX stage is a codec that tries the stream (#1364), not a missing one.
    if !cfg!(feature = "jp2k") {
        names.push(("JPXDecode", "jp2k"));
    }
    for (filter, feature) in names {
        let e = refused(&filtered(name(filter), vec![0u8; 4]));
        let reason = format!("filter-not-enabled:{filter}");
        assert_eq!(e.reason(), reason);
        match &e {
            PdfImageError::FilterNotEnabled {
                filter: f,
                feature: ft,
            } => {
                assert_eq!(f, filter);
                assert_eq!(*ft, feature);
            }
            other => panic!("{other:?}"),
        }
        assert!(e.to_string().contains(feature), "{e}");
        assert!(e.to_string().contains(filter), "{e}");
    }
}

#[test]
fn a_codec_after_flate_is_also_not_enabled_and_is_counted_in_the_report() {
    let bytes = filtered(
        Object::Array(vec![name("FlateDecode"), name("CCITTFaxDecode")]),
        flate(&[0u8; 4]),
    );
    let b = bake(&bytes);
    assert_eq!(
        b.report.refused.get("filter-not-enabled:CCITTFaxDecode"),
        Some(&1)
    );
    assert!(b.report.written_by_filter.is_empty());
}

#[test]
fn an_unknown_filter_is_still_unsupported_by_name() {
    for chain in [
        name("LZWDecode"),
        Object::Array(vec![name("ASCII85Decode"), name("FlateDecode")]),
    ] {
        let e = refused(&filtered(chain, vec![0u8; 4]));
        assert!(matches!(&e, PdfImageError::UnsupportedFilter(_)), "{e:?}");
        assert!(
            e.reason().starts_with("unsupported-filter:"),
            "{}",
            e.reason()
        );
    }
    let e = refused(&filtered(name("LZWDecode"), vec![0u8; 4]));
    assert_eq!(e.reason(), "unsupported-filter:LZWDecode");
}

#[test]
fn decode_parms_as_an_array_is_matched_to_its_filter_by_position() {
    // One Flate filter and a two entry array: the first entry is the Flate stage's (null), so no
    // predictor applies. The old reading took the first dictionary it found, which is the second
    // entry, and unpredicted rows that were never predicted.
    let raw = [10u8, 20, 30, 40];
    let s = image(
        2,
        2,
        dictionary! {
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            "Filter" => vec![name("FlateDecode")],
            "DecodeParms" => vec![
                Object::Null,
                Object::Dictionary(dictionary! {
                    "Predictor" => 12, "Colors" => 1, "BitsPerComponent" => 8, "Columns" => 2,
                }),
            ],
        },
        flate(&raw),
    );
    let b = bake(&one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q"));
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
}

#[test]
fn decode_parms_as_an_array_still_reaches_a_flate_predictor_in_its_own_slot() {
    // Row 0: Sub, 10 then +5. Row 1: Up, +1 then +2 over row 0.
    let raw = [1u8, 10, 5, 2, 1, 2];
    let s = image(
        2,
        2,
        dictionary! {
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            "Filter" => vec![name("FlateDecode")],
            "DecodeParms" => vec![Object::Dictionary(dictionary! {
                "Predictor" => 15, "Colors" => 1, "BitsPerComponent" => 8, "Columns" => 2,
            })],
        },
        flate(&raw),
    );
    let b = bake(&one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q"));
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
}

#[test]
fn written_by_filter_counts_each_written_image_under_its_terminal_filter() {
    let plain = gray8(1, 1, &[7]);
    let zipped = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "FlateDecode" },
        flate(&[7]),
    );
    let dct = image(
        16,
        16,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode" },
        jpeg(16, 16),
    );
    let zdct = image(
        16,
        16,
        dictionary! {
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8,
            "Filter" => vec![name("FlateDecode"), name("DCTDecode")],
        },
        flate(&jpeg(16, 16)),
    );
    let ccitt = image(
        1,
        1,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1, "Filter" => "CCITTFaxDecode" },
        vec![0],
    );
    let bytes = pdf(
        doc(),
        Resources::images(vec![
            ("A", Object::Stream(plain)),
            ("B", Object::Stream(zipped)),
            ("C", Object::Stream(dct)),
            ("D", Object::Stream(zdct)),
            ("E", Object::Stream(ccitt)),
        ]),
        b"q 4 0 0 4 0 0 cm /A Do Q q 4 0 0 4 10 0 cm /B Do Q q 4 0 0 4 20 0 cm /C Do Q \
          q 4 0 0 4 30 0 cm /D Do Q q 4 0 0 4 40 0 cm /E Do Q",
    );
    let b = bake(&bytes);
    let r = &b.report;
    assert_eq!(r.written, 4);
    assert_eq!(r.images, r.written + r.refused_total());
    assert_eq!(r.written_by_filter.get("none"), Some(&1));
    assert_eq!(r.written_by_filter.get("FlateDecode"), Some(&1));
    assert_eq!(r.written_by_filter.get("DCTDecode"), Some(&2));
    assert_eq!(r.written_by_filter.values().sum::<u64>(), r.written);
    assert!(!r.written_by_filter.contains_key("CCITTFaxDecode"));
}

#[test]
fn a_stream_over_the_per_image_cap_is_one_refused_image_and_not_a_failed_page() {
    let d = PdfVectorLimits::default();
    assert_eq!(d.max_image_stream_bytes, 64 << 20);
    assert!(d.max_image_stream_bytes < d.max_image_bytes);

    let big = gray8(16, 16, &[9u8; 256]);
    let small = gray8(1, 1, &[5]);
    let bytes = pdf(
        doc(),
        Resources::images(vec![
            ("Big", Object::Stream(big)),
            ("Small", Object::Stream(small)),
        ]),
        b"q 4 0 0 4 0 0 cm /Big Do Q q 4 0 0 4 10 0 cm /Small Do Q",
    );
    let limits = PdfVectorLimits {
        max_image_stream_bytes: 100,
        ..PdfVectorLimits::default()
    };
    let p = extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap();
    let b = bake_page_images(&p, &PdfImageOptions::default());
    assert_eq!(b.report.images, 2);
    assert_eq!(b.report.written, 1);
    assert_eq!(b.refused[0].index, 0);
    assert_eq!(b.refused[0].error.reason(), "too-large");
}

#[test]
fn a_filter_error_keeps_the_codec_error_as_its_source() {
    // A JPEG that is not one: tried, so `decode-failed`, with the codec's error underneath.
    let e = refused(&one(
        image(
            2,
            2,
            dictionary! {
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "DCTDecode",
            },
            vec![0xFF, 0xD8, 0x00, 0x01],
        ),
        "q 2 0 0 2 0 0 cm /Im1 Do Q",
    ));
    assert_eq!(e.reason(), "decode-failed");
    assert!(
        matches!(&e, PdfImageError::Filter { filter, .. } if filter == "DCTDecode"),
        "{e:?}"
    );
    assert!(e.source().is_some());
}
