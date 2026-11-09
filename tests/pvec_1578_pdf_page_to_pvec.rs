//! Issue #1578 (the L4 mapping, the part that needs none of the new streams; L1 to L3 and L5 to
//! L7 stay open): `pvec::pdf_page_to_pvec` is the mapping iasbuilt/server main's `convert_pdf`
//! runs since server #336 (white fills dropped, closed fills as polygons, rectangular clips
//! applied), moved into the library. The bytes it writes for `blueprint-mix.pdf` are pinned to
//! what server main's `convert_pdf` writes for the same page, producer `iasbuilt-server` 1.0.3
//! (server commit 33e48f5). Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::pdf_vector::extract_page_paths_from_bytes;
use libviprs::pvec::{PdfToPvecError, PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

// These build one-page PDFs in memory, so they run without the fixture.
fn convert(body: &str) -> Result<Vec<u8>, PdfToPvecError> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::from(page)],
            "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut pdf = Vec::new();
    doc.save_to(&mut pdf).unwrap();
    let page = extract_page_paths_from_bytes(&pdf, 1, &PdfVectorLimits::default()).unwrap();
    pdf_page_to_pvec(&page, "test", "0.0.0")
}

const BOX: &str = "10 10 m 60 10 l 60 60 l 10 60 l h";

#[test]
fn a_white_fill_is_dropped() {
    let error = convert(&format!("1 g {BOX} f")).unwrap_err();
    assert!(matches!(error, PdfToPvecError::NoPaths));
    assert!(convert(&format!("0 g {BOX} f")).is_ok());
}

#[test]
fn a_stroked_white_fill_keeps_only_its_outline() {
    let both = convert(&format!("1 g {BOX} B")).unwrap();
    let outline = convert(&format!("{BOX} S")).unwrap();
    assert_eq!(both, outline);
}

#[test]
fn a_rectangular_clip_cuts_a_stroke_to_the_rectangle() {
    let clipped = convert("q 25 0 50 100 re W n 0 50 m 100 50 l S Q").unwrap();
    let direct = convert("25 50 m 75 50 l S").unwrap();
    assert_eq!(clipped, direct);
}

#[test]
fn a_rectangular_clip_cuts_a_fill_to_the_rectangle() {
    let clipped = convert("q 0 0 50 100 re W n 0 g 25 25 m 75 25 l 75 75 l 25 75 l h f Q").unwrap();
    let direct = convert("0 g 25 25 m 50 25 l 50 75 l 25 75 l h f").unwrap();
    assert_eq!(clipped, direct);
}

#[test]
fn a_path_wholly_outside_its_clip_is_skipped() {
    let error = convert("q 0 0 10 10 re W n 50 50 m 90 50 l S Q").unwrap_err();
    assert!(matches!(error, PdfToPvecError::NoPaths));
}

fn blueprint_mix() -> Option<PathBuf> {
    let fixtures = std::env::var_os("LIBVIPRS_TESTS_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../libviprs-tests/tests/fixtures")
        });
    let path = fixtures.join("blueprint-mix.pdf");
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipped: {} is not there (set LIBVIPRS_TESTS_FIXTURES)",
            path.display()
        );
        None
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn blueprint_mix_bytes_are_pinned_with_the_stroke_and_paint_streams() {
    let Some(path) = blueprint_mix() else { return };
    let mut page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    // The paths alone: server main writes no text and no rasters (#1578 L6 and L7 add them, and
    // `pvec_1578_e_text_images.rs` covers those).
    page.texts.clear();
    page.images.clear();
    let bytes = pdf_page_to_pvec(&page, "iasbuilt-server", "1.0.3").unwrap();
    assert_eq!(bytes.len(), 16448);
    let hex: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        "d7f0e5268926be226defcf9115c0e6b315f39864642e5575ea4ac374eb7674a4"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_page_with_nothing_painted_is_refused_with_the_servers_message() {
    let Some(path) = blueprint_mix() else { return };
    let mut page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    page.entities.clear();
    let error = pdf_page_to_pvec(&page, "iasbuilt-server", "0.0.0").unwrap_err();
    assert_eq!(
        error.to_string(),
        "The PDF has no supported vector paths for PlanViewer."
    );
}
