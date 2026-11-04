//! Issue #1491: one call turns a PDF into a `.pvec` with its vectors and its rasters.
//!
//! `PdfConversion` is to a PDF page what `PvecConversion` is to a DWG. Given a page with a stroked
//! path and an image it publishes an artifact that holds the entity and the raster, with a report
//! that counts both. Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::CancelToken;
use libviprs::cad::CadSource;
use libviprs::pvec::pdf_image::PdfImageNote;
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec_convert::{PdfConversion, PdfConvertError};

fn pdf_with_path_and_image() -> Vec<u8> {
    one(
        rgb8(2, 2, &[9u8; 12]),
        "10 10 m 200 10 l 200 200 l S q 20 0 0 20 5 5 cm /Im1 Do Q",
    )
}

fn dest(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1491-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join("out.pvec")
}

/// Removes the directory `dest` made.
fn clean(out: &std::path::Path) {
    std::fs::remove_dir_all(out.parent().unwrap()).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_pdf_converts_to_an_artifact_with_an_entity_and_a_raster() {
    let bytes = pdf_with_path_and_image();
    let out = dest("both");
    let done = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .convert()
        .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    // The path is two segments, and each is one entity.
    assert_eq!((done.written, done.rasters), (2, 1));
    let reader = open_published(&done.artifact).unwrap();
    assert_eq!(reader.entity_count().unwrap(), done.written);
    assert_eq!(reader.raster_placements().unwrap().len(), 1);
    clean(&out);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_counts_the_image_and_the_vector_entities() {
    let bytes = pdf_with_path_and_image();
    let out = dest("report");
    let done = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .convert()
        .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    assert_eq!(done.report.totals.encountered, done.fidelity.encountered());
    assert_eq!(done.report.totals.encountered, done.written);
    let pdf = done
        .report
        .pdf
        .as_ref()
        .expect("a PDF report has a pdf section");
    assert_eq!((pdf.page, pdf.images, pdf.rasters), (1, 1, 1));
    assert!(pdf.images_refused.is_empty());
    // The published report is the one returned, and it says the same.
    let json = std::fs::read_to_string(report_path(&done.artifact)).unwrap();
    assert_eq!(json, done.report.to_json_string().unwrap());
    assert!(json.contains("\"rasters\": 1"), "{json}");
    clean(&out);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_same_pdf_twice_gives_the_same_bytes() {
    let bytes = pdf_with_path_and_image();
    let (a, b) = (dest("det-a"), dest("det-b"));
    for d in [&a, &b] {
        PdfConversion::new(CadSource::Bytes(&bytes), d)
            .convert()
            .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    }
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    clean(&a);
    clean(&b);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_refused_image_is_a_report_line_under_its_key_and_the_vectors_still_convert() {
    // An image under a clip that leaves nothing of it.
    let bytes = one(
        rgb8(8, 8, &[7u8; 8 * 8 * 3]),
        "10 10 m 200 10 l S q 20 20 5 5 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q",
    );
    let out = dest("refused");
    let done = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .convert()
        .unwrap();
    assert!(done.written >= 1, "{}", done.written);
    assert_eq!(done.rasters, 0);
    let pdf = done.report.pdf.as_ref().unwrap();
    assert_eq!(pdf.images_refused.get("fully-clipped"), Some(&1));
    clean(&out);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_image_notes_are_the_bakes_own_note_keys() {
    // Painted between two paths, so interleaved, and under a rectangle clip, so clipped.
    let bytes = one(
        rgb8(8, 8, &[7u8; 8 * 8 * 3]),
        "10 10 m 200 10 l S q 0 0 6 6 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q 10 20 m 200 20 l S",
    );
    let out = dest("notes");
    let done = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .convert()
        .unwrap();
    let notes = &done.report.pdf.as_ref().unwrap().image_notes;
    let known: Vec<&str> = [
        PdfImageNote::ColourSpaceApproximated,
        PdfImageNote::BitDepthReduced,
        PdfImageNote::ReEncodedLossy,
        PdfImageNote::Interleaved,
        PdfImageNote::Clipped,
        PdfImageNote::MaskIgnored,
        PdfImageNote::Bilevel,
    ]
    .map(PdfImageNote::as_str)
    .to_vec();
    for key in notes.keys() {
        assert!(known.contains(&key.as_str()), "`{key}` is not a note key");
    }
    assert_eq!(notes.get(PdfImageNote::Interleaved.as_str()), Some(&1));
    assert_eq!(notes.get(PdfImageNote::Clipped.as_str()), Some(&1));
    clean(&out);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_page_without_a_path_is_refused_and_nothing_is_published() {
    let bytes = one(rgb8(2, 2, &[9u8; 12]), "q 20 0 0 20 5 5 cm /Im1 Do Q");
    let out = dest("nopath");
    let err = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PdfConvertError::NoGeometry), "{err}");
    assert!(!out.exists() && !report_path(&out).exists());
    clean(&out);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_missing_page_and_a_cancelled_run_publish_nothing() {
    let bytes = pdf_with_path_and_image();
    let out = dest("refusals");
    let err = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .page(2)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PdfConvertError::Open(_)), "{err}");
    let token = CancelToken::new();
    token.cancel();
    let err = PdfConversion::new(CadSource::Bytes(&bytes), &out)
        .cancel(token)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PdfConvertError::Cancelled), "{err}");
    assert!(!out.exists() && !report_path(&out).exists());
    clean(&out);
}
