//! Issue #1328 (R2.4): a DWG `IMAGE` is reported with its frame as the acadrust `Unmapped` record,
//! and no placement is written, whether or not the image is shown or loaded.
//!
//! The drawings are written by acadrust's `DwgWriter` and read back through the provider, so the
//! path under test is the real one. Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use acadrust::entities::{ImageDisplayFlags, RasterImage};
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::fill_policy::{FillConstruct, FillReport};
use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind, RawEntity, RawGeometry};

fn read_back(image: RasterImage) -> Vec<RawEntity> {
    let mut doc = CadDocument::new();
    doc.add_entity(EntityType::RasterImage(image)).expect("add");
    let bytes = DwgWriter::write_to_vec(&doc).expect("write");
    let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).expect("open");
    let layouts = p.layouts().unwrap();
    let model = layouts
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    p.entities(&EntityScope::Layout(model.id))
        .unwrap()
        .collect_all()
        .unwrap()
}

fn image() -> RasterImage {
    let mut img = RasterImage::new("scan.tif", Vector3::new(10.0, 20.0, 0.0), 400.0, 300.0);
    img.u_vector = Vector3::new(0.5, 0.0, 0.0);
    img.v_vector = Vector3::new(0.0, 0.5, 0.0);
    img
}

fn only(entities: &[RawEntity]) -> &RawEntity {
    let images: Vec<_> = entities
        .iter()
        .filter(|e| e.provenance.class_name == "IMAGE")
        .collect();
    assert_eq!(images.len(), 1, "{entities:?}");
    images[0]
}

#[test]
fn an_image_is_unmapped_and_its_frame_is_in_the_record() {
    let entities = read_back(image());
    let RawGeometry::Unmapped { reason } = &only(&entities).geometry else {
        panic!(
            "an IMAGE must be Unmapped, got {:?}",
            only(&entities).geometry
        );
    };
    for needle in [
        "insert=(10, 20, 0)",
        "u=(0.5, 0, 0)",
        "v=(0, 0.5, 0)",
        "size=(400, 300)",
        "show_image=true",
        "no placement",
    ] {
        assert!(reason.contains(needle), "{needle} missing from {reason:?}");
    }
}

#[test]
fn an_image_with_show_image_clear_is_reported_the_same_way() {
    let mut img = image();
    img.flags.remove(ImageDisplayFlags::SHOW_IMAGE);
    let entities = read_back(img);
    let RawGeometry::Unmapped { reason } = &only(&entities).geometry else {
        panic!()
    };
    assert!(reason.contains("show_image=false"), "{reason:?}");
    assert!(reason.contains("no placement"), "{reason:?}");
}

#[test]
fn an_image_whose_definition_is_not_loaded_says_so() {
    // No IMAGEDEF object accompanies the image, so the definition cannot be resolved as loaded.
    let entities = read_back(image());
    let RawGeometry::Unmapped { reason } = &only(&entities).geometry else {
        panic!()
    };
    assert!(reason.contains("definition_loaded=false"), "{reason:?}");
}

#[test]
fn the_report_counts_the_frame() {
    let entities = read_back(image());
    let mut report = FillReport::new();
    report.count_dwg_entities(&entities);
    assert_eq!(report.count(FillConstruct::DwgImage), 1);
}
