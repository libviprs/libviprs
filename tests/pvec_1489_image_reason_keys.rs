//! Issue #1489: every reason key the PDF image bake reports is named in a doc.
//!
//! `docs/pvec-raster-producer.md` lists the refusals by key. `PdfImageError::reason` returns more
//! keys than that page names (`fully-clipped`, `too-large`, `invalid-placement`,
//! `unsupported-colour-space:<name>`, `pyramid-refused`), so a reader of a report cannot look them
//! up. A `name:argument` key is documented by its prefix. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageError, PdfImageNote, PdfImageOptions, bake_page_images};
use libviprs::raster_pvec::pyramid::RasterPyramidError;

fn docs() -> String {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
    let mut all = String::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap().to_string_lossy().into_owned();
        if n.starts_with("pvec") && n.ends_with(".md") {
            all.push_str(&std::fs::read_to_string(p).unwrap());
        }
    }
    all
}

fn documented(docs: &str, key: &str) -> bool {
    docs.contains(&format!("`{}", key.split(':').next().unwrap()))
}

#[test]
fn every_refusal_key_is_in_a_doc() {
    let docs = docs();
    let every_variant = vec![
        PdfImageError::Decode("x".into()),
        PdfImageError::TooLarge {
            pixels: 2,
            limit: 1,
        },
        PdfImageError::UnsupportedFilter("LZWDecode".into()),
        PdfImageError::FilterNotEnabled {
            filter: "JBIG2Decode".into(),
            feature: "jbig2",
        },
        PdfImageError::UnsupportedColourSpace("Separation".into()),
        PdfImageError::GraphicsStateSoftMask,
        PdfImageError::SingularCtm,
        PdfImageError::InvalidPlacement("determinant"),
        PdfImageError::FullyClipped,
        PdfImageError::AlphaNeedsLosslessTiles,
        PdfImageError::Pyramid(RasterPyramidError::TileSize(100)),
    ];
    let missing: Vec<String> = every_variant
        .iter()
        .map(PdfImageError::reason)
        .filter(|k| !documented(&docs, k))
        .collect();
    assert!(missing.is_empty(), "keys in no doc: {missing:?}");
}

#[test]
fn the_five_keys_the_blueprint_hit_or_could_hit_are_named() {
    let docs = docs();
    for key in [
        "fully-clipped",
        "too-large",
        "invalid-placement",
        "unsupported-colour-space",
        "pyramid-refused",
    ] {
        assert!(documented(&docs, key), "`{key}` is in no doc");
    }
}

#[test]
fn every_note_key_is_in_a_doc() {
    let docs = docs();
    for n in [
        PdfImageNote::ColourSpaceApproximated,
        PdfImageNote::BitDepthReduced,
        PdfImageNote::ReEncodedLossy,
        PdfImageNote::Interleaved,
        PdfImageNote::Clipped,
        PdfImageNote::MaskIgnored,
        PdfImageNote::Bilevel,
    ] {
        assert!(
            documented(&docs, n.as_str()),
            "`{}` is in no doc",
            n.as_str()
        );
    }
}

#[test]
fn a_real_refusal_from_a_page_is_reported_under_a_documented_key() {
    // An 8 by 8 image under a clip that leaves nothing of it.
    let bytes = one(
        rgb8(8, 8, &[7u8; 8 * 8 * 3]),
        "q 20 20 5 5 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q",
    );
    let bake = bake_page_images(&page(&bytes), &PdfImageOptions::default());
    let docs = docs();
    assert_eq!(bake.report.refused.len(), 1);
    for key in bake.report.refused.keys() {
        assert!(documented(&docs, key), "report key `{key}` is in no doc");
    }
}
