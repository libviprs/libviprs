//! Issue #1430: `PvecConversion::convert` carries a planar SPLINE the way
//! `Normalized2DGeometry::from_spline` maps it, and refuses the ones it cannot, with the reason
//! code `dwg_spline` gives. The drawings are written by acadrust's `DwgWriter`.
//!
//! Needs `--features pvec-convert`.
#![cfg(feature = "pvec-convert")]

use acadrust::entities::Spline;
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::fidelity::FidelityClass;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;

fn pts(n: usize, z: f64) -> Vec<Vector3> {
    (0..n)
        .map(|k| Vector3::new(k as f64 * 2.0, ((k * 7) % 5) as f64, z))
        .collect()
}

/// A one-span cubic (a Bezier), a cubic with several spans, a degree 2, and a 3D cubic.
fn drawing() -> Vec<u8> {
    let mut doc = CadDocument::new();
    for (degree, n, z) in [(3, 4, 0.0), (3, 9, 0.0), (2, 6, 0.0), (3, 6, 5.0)] {
        doc.add_entity(EntityType::Spline(Spline::from_control_points(
            degree,
            pts(n, z),
        )))
        .unwrap();
    }
    DwgWriter::write_to_vec(&doc).expect("write")
}

#[test]
fn planar_splines_are_written_and_a_3d_one_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("s.pvec");
    let bytes = drawing();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("a drawing of splines converts");
    assert_eq!(out.report.totals.encountered, 4);
    assert_eq!(out.written, 3, "the three planar splines are written");
    assert_eq!(open_published(&dest).unwrap().entity_count().unwrap(), 3);
    assert_eq!(out.fidelity.count(FidelityClass::SupportedExact), 3);
    let reasons: Vec<_> = out
        .fidelity
        .reasons()
        .map(|(r, c, n)| (r.code(), c.to_string(), n))
        .collect();
    assert_eq!(
        reasons,
        vec![("spline-3d", "SPLINE".to_string(), 1)],
        "{reasons:?}"
    );
}

#[test]
fn no_spline_is_reported_as_an_unmapped_source_class() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = drawing();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dir.path().join("s.pvec"))
        .unit(SourceUnit::Meters)
        .convert()
        .unwrap();
    assert!(
        out.fidelity
            .reasons()
            .all(|(r, _, _)| r.code() != "UNMAPPED_SOURCE_CLASS"),
        "{:?}",
        out.fidelity.reasons().collect::<Vec<_>>()
    );
}
