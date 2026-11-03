//! Issue #1491: one call turns a PDF into a `.pvec` with its vectors and its rasters.
//!
//! `PvecConversion` is the one conversion surface. Given a PDF page with a stroked path and an
//! image it should publish an artifact that holds the entity and the raster, with a report that
//! counts both. Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec_convert::PvecConversion;

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

#[test]
#[cfg_attr(miri, ignore)]
fn a_pdf_converts_to_an_artifact_with_an_entity_and_a_raster() {
    let bytes = pdf_with_path_and_image();
    let out = dest("both");
    let done = PvecConversion::new(CadSource::Bytes(&bytes), &out)
        .unit(SourceUnit::Millimeters)
        .convert()
        .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    assert!(done.written >= 2, "{}", done.written);
    let reader = open_published(&done.artifact).unwrap();
    assert!(reader.entity_count().unwrap() >= 2);
    assert_eq!(reader.raster_placements().unwrap().len(), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_counts_the_image_and_the_vector_entities() {
    let bytes = pdf_with_path_and_image();
    let out = dest("report");
    let done = PvecConversion::new(CadSource::Bytes(&bytes), &out)
        .unit(SourceUnit::Millimeters)
        .convert()
        .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    let report = std::fs::read_to_string(report_path(&done.artifact)).unwrap();
    assert!(report.contains("raster"), "{report}");
    assert_eq!(done.report.totals.encountered, done.fidelity.encountered());
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_same_pdf_twice_gives_the_same_bytes() {
    let bytes = pdf_with_path_and_image();
    let (a, b) = (dest("det-a"), dest("det-b"));
    for d in [&a, &b] {
        PvecConversion::new(CadSource::Bytes(&bytes), d)
            .unit(SourceUnit::Millimeters)
            .convert()
            .unwrap_or_else(|e| panic!("a PDF is refused: {e}"));
    }
    assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
}
