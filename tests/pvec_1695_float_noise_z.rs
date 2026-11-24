//! A LINE or ELLIPSE whose Z differs by float noise is planar, not geometry-3d (#1695).
//!
//! A drawing computes its end points, so a 2D line comes back with a Z of 1.87e-13 at one end and
//! -1.75e-13 at the other. The planarity test compared the two Z values exactly, so such a line was
//! refused as `geometry-3d` and left off the sheet, while a line whose ends happened to share the
//! same tiny Z converted. The test is now relative, the way #1660 reads float noise in a viewport's
//! direction: a Z difference of at most 1e-9 of the longer of the entity's plan length and the
//! coordinates' size, with that length floored at one drawing unit, is noise. A real out of plane
//! segment is still refused. The drawings are written with acadrust. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;

use acadrust::entities::{Circle, Ellipse, Line, Spline};
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec_convert::PvecConversion;

/// What the report says about the one entity of `class` in a drawing holding just `entity`: its
/// fidelity tag and reason code, and how many entities were written beyond the circle.
fn outcome(tag: &str, class: &str, entity: EntityType) -> (String, Option<String>, u64) {
    let mut doc = CadDocument::new();
    doc.add_entity(entity).unwrap();
    // A drawing with nothing written is an error, so a refused entity needs a circle beside it.
    doc.add_entity(EntityType::Circle(Circle::from_center_radius(
        Vector3::new(500.0, 500.0, 0.0),
        5.0,
    )))
    .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dir = std::env::temp_dir().join(format!("libviprs-1695-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("o.pvec"))
        .unit(SourceUnit::Millimeters)
        .convert();
    let _ = fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let rows: Vec<_> = out
        .report
        .entities
        .iter()
        .filter(|r| r.source_class == class)
        .collect();
    assert_eq!(rows.len(), 1, "{tag}: one {class} row expected: {rows:?}");
    (
        rows[0].fidelity.clone(),
        rows[0].reason.clone(),
        out.written - 1,
    )
}

fn line(z0: f64, z1: f64, length: f64) -> EntityType {
    EntityType::Line(Line::from_coords(10.0, 20.0, z0, 10.0 + length, 20.0, z1))
}

fn drawn(tag: &str, class: &str, entity: EntityType) {
    let (fidelity, reason, written) = outcome(tag, class, entity);
    assert_eq!(reason, None, "{tag}: refused as {fidelity}");
    assert_eq!(written, 1, "{tag}: the {class} was not written");
}

fn refused_3d(tag: &str, class: &str, entity: EntityType, reason: &str) {
    let (fidelity, got, written) = outcome(tag, class, entity);
    assert_eq!(got.as_deref(), Some(reason), "{tag}: {fidelity}");
    assert_eq!(fidelity, "RequiresPVecCapability", "{tag}");
    assert_eq!(written, 0, "{tag}: a refused entity writes no geometry");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_line_with_the_same_tiny_z_at_both_ends_converts() {
    drawn("same", "LINE", line(1.87e-13, 1.87e-13, 300.0));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_line_whose_z_differs_by_float_noise_converts() {
    drawn("noise", "LINE", line(1.87e-13, -1.75e-13, 300.0));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_line_that_really_leaves_the_plane_is_still_geometry_3d() {
    refused_3d("real", "LINE", line(0.0, 1.0, 300.0), "geometry-3d");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_short_line_is_judged_against_the_floor_not_its_own_length() {
    drawn("short", "LINE", line(1.87e-13, -1.75e-13, 1e-3));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_noise_is_relative_to_the_coordinates_far_from_the_origin() {
    // One ulp at 251705.56 is about 3e-11, which is well over a fixed 1e-12 but under 1e-9 of it.
    drawn("far", "LINE", line(251_705.56, 251_705.56 + 3.0e-11, 300.0));
    refused_3d(
        "far-real",
        "LINE",
        line(251_705.56, 251_706.56, 300.0),
        "geometry-3d",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_vertical_segment_is_not_noise() {
    // No plan length at all: the floor keeps the tolerance at 1e-9, a metre of Z is real.
    let up = EntityType::Line(Line::from_coords(5.0, 5.0, 0.0, 5.0, 5.0, 1.0));
    refused_3d("vertical", "LINE", up, "geometry-3d");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_zero_length_line_with_noise_is_coincident_not_3d() {
    let (_, reason, _) = outcome(
        "zero",
        "LINE",
        EntityType::Line(Line::from_coords(7.0, 7.0, 1.87e-13, 7.0, 7.0, -1.75e-13)),
    );
    assert_eq!(reason.as_deref(), Some("coincident-vertices"));
}

fn ellipse(axis_z: f64) -> EntityType {
    EntityType::Ellipse(Ellipse::from_center_axes(
        Vector3::new(50.0, 50.0, 0.0),
        Vector3::new(40.0, 10.0, axis_z),
        0.5,
    ))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_an_ellipse_whose_major_axis_has_float_noise_in_z_converts() {
    drawn("ellipse-noise", "ELLIPSE", ellipse(3.6e-13));
    drawn("ellipse-noise-neg", "ELLIPSE", ellipse(-1.75e-13));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_an_ellipse_with_a_tilted_major_axis_is_still_geometry_3d() {
    refused_3d("ellipse-real", "ELLIPSE", ellipse(5.0), "geometry-3d");
}

fn spline(z: [f64; 4]) -> EntityType {
    let pts = [[0.0, 0.0], [10.0, 20.0], [30.0, 20.0], [40.0, 0.0]];
    EntityType::Spline(Spline::from_control_points(
        3,
        pts.iter()
            .zip(z)
            .map(|(p, z)| Vector3::new(p[0], p[1], z))
            .collect(),
    ))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_spline_with_float_noise_in_its_control_point_z_converts() {
    drawn(
        "spline-noise",
        "SPLINE",
        spline([1.87e-13, -1.75e-13, 0.0, 2.0e-13]),
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1695_a_spline_with_a_real_z_spread_is_still_spline_3d() {
    refused_3d(
        "spline-real",
        "SPLINE",
        spline([0.0, 0.0, 5.0, 0.0]),
        "spline-3d",
    );
}
