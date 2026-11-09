//! Issue #1578 (L6): `pdf_page_to_pvec` writes the text of the page (text records, each on a
//! hidden anchor entity that carries the text colour). `blueprint-mix.pdf` is the oracle: its title
//! block reads "Campus Map 1.0 1 of 09 Revision 1.01/2025 - April 7, 2025." (pdftotext). Needs
//! `--features pvec-write,pvec-raster`. Image baking is checked in `pvec_1578_e_text_state`.

#![cfg(all(feature = "pvec-write", feature = "pvec-raster"))]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use pvec_facade::components::{ItemKind, PaintOrderStream, StrokePaintStream, TextStream};
use pvec_facade::{EntityId, PvecReader};
use std::path::PathBuf;

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

fn payload(reader: &PvecReader, capability: pvec_capabilities::CapabilityId) -> Vec<u8> {
    let (streams, _) = reader.streams().unwrap();
    streams
        .into_iter()
        .find(|s| s.capability == capability)
        .unwrap_or_else(|| panic!("no stream for {capability:?}"))
        .payload
}

fn written() -> Option<(libviprs::pvec::PdfVectorPage, Vec<u8>)> {
    let path = blueprint_mix()?;
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "libviprs-1578-e", "1").unwrap();
    Some((page, bytes))
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_title_block_is_a_text_record_on_a_hidden_anchor() {
    let Some((_, bytes)) = written() else { return };
    let reader = PvecReader::open(bytes).unwrap();
    let texts = TextStream::decode(&payload(&reader, TextStream::CAPABILITY)).unwrap();
    let all: String = texts
        .records
        .iter()
        .map(|r| r.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    for needle in ["Campus Map", "1.0", "April 7, 2025"] {
        assert!(all.contains(needle), "`{needle}` is missing from `{all}`");
    }
    for record in &texts.records {
        assert!(record.height > 0.0 && record.height < 2.0, "{record:?}");
        assert!(record.rotation.is_finite());
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_text_anchor_is_an_entity_with_the_text_colour() {
    let Some((_, bytes)) = written() else { return };
    let reader = PvecReader::open(bytes).unwrap();
    let texts = TextStream::decode(&payload(&reader, TextStream::CAPABILITY)).unwrap();
    let paint =
        StrokePaintStream::decode(&payload(&reader, StrokePaintStream::CAPABILITY)).unwrap();
    assert!(!texts.records.is_empty());
    for record in &texts.records {
        let entity: EntityId = record.entity;
        assert!(
            paint.assignments.iter().any(|(id, _)| *id == entity),
            "text anchor {entity:?} has no colour"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_text_record_of_the_title_block_is_named_once_in_the_paint_order() {
    let Some((_, bytes)) = written() else { return };
    let reader = PvecReader::open(bytes).unwrap();
    let texts = TextStream::decode(&payload(&reader, TextStream::CAPABILITY)).unwrap();
    let order = PaintOrderStream::decode(&payload(&reader, PaintOrderStream::CAPABILITY)).unwrap();
    let mut named: Vec<u32> = order
        .runs
        .iter()
        .filter(|run| run.kind == ItemKind::Text)
        .flat_map(|run| run.first..run.first + run.count)
        .collect();
    named.sort_unstable();
    assert_eq!(named, (0..texts.records.len() as u32).collect::<Vec<_>>());
}
