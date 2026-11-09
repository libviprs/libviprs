//! Issue #1578 (L6 and L7): `pdf_page_to_pvec` writes the text of the page (text records, each on a
//! hidden anchor entity that carries the text colour) and bakes the page's images into the raster
//! capability, ordered with the fills. `blueprint-mix.pdf` is the oracle: its title block reads
//! "Campus Map 1.0 1 of 09 Revision 1.01/2025 - April 7, 2025." (pdftotext) and its scan strips
//! are 12738 by 220 images with soft masks. Needs `--features pvec-write,pvec-raster`.

#![cfg(all(feature = "pvec-write", feature = "pvec-raster"))]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use pvec_facade::components::{StrokePaintStream, TextStream};
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
fn the_baked_images_are_rasters_placed_where_the_pdf_paints_them() {
    let Some((page, bytes)) = written() else {
        return;
    };
    let reader = PvecReader::open(bytes).unwrap();
    let placements = reader.raster_placements().unwrap();
    assert!(!placements.is_empty(), "no image became a raster");
    // Pixel (0, 0) is the top left of the image, which is (0, 1) in its unit square, and the
    // far corner is (1, 0). Source space is inches with the origin at the top left of the page.
    let paints = |placement: &pvec_facade::unstable::Placement,
                  image: &libviprs::pvec::PdfImagePlacement| {
        let [a, b, c, d, e, f] = image.ctm;
        let (w, h) = (f64::from(placement.width), f64::from(placement.height));
        [(0.0, 0.0, 0.0, 1.0), (w, h, 1.0, 0.0)]
            .iter()
            .all(|&(px, py, u, v)| {
                let [pa, pb, pc, pd, ptx, pty] = placement.pixel_to_source;
                let got = [pa * px + pb * py + ptx, pc * px + pd * py + pty];
                let want = [
                    (a * u + c * v + e) / 72.0,
                    (page.height - (b * u + d * v + f)) / 72.0,
                ];
                (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6
            })
    };
    for (raster, placement) in placements.iter().enumerate() {
        assert!(
            page.images.iter().any(|image| paints(placement, image)),
            "raster {raster} is not where any image of the PDF is painted"
        );
    }
}
