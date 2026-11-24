//! A restarted polyline side has its dash pattern fitted to it (#1694).
//!
//! #1691 restarts a PLINEGEN-off polyline's pattern at every vertex, but only aligns it at the start
//! of each side, so a side whose length is not a dash plus a whole number of periods reaches its
//! corner in a gap where the plot has a dash. The rule here: a side is fitted so it ends on a dash.
//! With the pattern's first dash d and period p, a side of length L takes n = floor((L - d) / p)
//! whole periods and the pattern is stretched by L / (n p + d), every element by the same factor, so
//! the shape is kept and the last dash ends exactly at the corner. A side that already is d plus a
//! whole number of periods, a PLINEGEN-on polyline and a LINE are not touched. Fixtures are written
//! with acadrust. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::collections::BTreeMap;
use std::fs;

use acadrust::entities::{Line, LwPolyline, Polyline2D, Vertex2D};
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::Curve;
use pvec_facade::components::LineStyleStream;

const STREAMS: StreamSelection = StreamSelection {
    line_style: true,
    ..StreamSelection::NONE
};

/// HIDDEN's dash and gap in the fixtures, and so its first dash and its period.
const DASH: f64 = 0.25;
const PERIOD: f64 = 0.375;

/// One record: its two ends when it is a two-point polyline (zeros otherwise) and the dash array of
/// its style (empty for none).
type Record = ([f64; 2], [f64; 2], Vec<f64>);

fn convert(tag: &str, bytes: &[u8]) -> Vec<Record> {
    let dir = std::env::temp_dir().join(format!("libviprs-1694-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(STREAMS)
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let reader = open_published(&dest).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let style = LineStyleStream::decode(
        &streams
            .iter()
            .find(|s| s.capability == LineStyleStream::CAPABILITY)
            .expect("the line style stream")
            .payload,
    )
    .unwrap();
    let dashes: BTreeMap<u64, Vec<f64>> = style
        .assignments
        .iter()
        .map(|(id, k)| {
            (
                u64::from(id.index()),
                style.styles[*k as usize].dashes.clone(),
            )
        })
        .collect();
    let out = reader
        .entities()
        .unwrap()
        .iter()
        .map(|(id, curve)| {
            let (a, b) = match curve {
                Curve::Polyline(p) if p.vertices().len() == 2 => {
                    let v = p.vertices();
                    ([v[0].x, v[0].y], [v[1].x, v[1].y])
                }
                // Anything else (a LINE, a run of several vertices) has no side of its own.
                _ => ([0.0; 2], [0.0; 2]),
            };
            (
                a,
                b,
                dashes
                    .get(&u64::from(id.index()))
                    .cloned()
                    .unwrap_or_default(),
            )
        })
        .collect();
    let _ = fs::remove_dir_all(dir);
    out
}

fn doc() -> CadDocument {
    let mut doc = CadDocument::new();
    doc.header.polyline_linetype_generation = false;
    let mut hidden = LineType::new("HIDDEN");
    hidden.elements = vec![LineTypeElement::dash(DASH), LineTypeElement::space(0.125)];
    doc.line_types.add(hidden).unwrap();
    doc
}

fn lw(points: &[[f64; 2]], closed: bool, plinegen: bool) -> EntityType {
    let mut p = LwPolyline::new();
    for q in points {
        p.add_point_with_bulge(Vector2::new(q[0], q[1]), 0.0);
    }
    p.is_closed = closed;
    p.plinegen = plinegen;
    p.common.linetype = "HIDDEN".into();
    EntityType::LwPolyline(p)
}

fn rect(w: f64, h: f64) -> [[f64; 2]; 4] {
    [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

fn straight(tag: &str, entity: EntityType) -> Vec<Record> {
    let mut d = doc();
    d.add_entity(entity).unwrap();
    convert(tag, &DwgWriter::write_to_vec(&d).unwrap())
}

fn length(r: &Record) -> f64 {
    (r.1[0] - r.0[0]).hypot(r.1[1] - r.0[1])
}

/// Whether the record's pattern ends exactly on the end of a dash at its far corner: its length is
/// the first dash plus a whole number of periods, to float noise.
fn ends_on_a_dash(r: &Record) -> bool {
    let [d, g] = [r.2[0], r.2[1]];
    let n = (length(r) - d) / (d + g);
    (n - n.round()).abs() < 1e-6 && n.round() >= 0.0
}

/// The factor the pattern was stretched by, from its first dash.
fn stretch(r: &Record) -> f64 {
    r.2[0] / DASH
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_every_side_of_a_closed_polyline_ends_on_a_dash() {
    let r = straight("closed", lw(&rect(10.1, 10.1), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert_eq!(side.2.len(), 2, "{side:?}");
        assert!(ends_on_a_dash(side), "this side ends in a gap: {side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_the_pattern_is_stretched_by_length_over_whole_periods_plus_a_dash() {
    // 10.1 holds 26 periods and a dash (10.0), so it stretches by 1.01. 7.3 holds 18 and a dash
    // (7.0), so it stretches by 7.3 / 7.0.
    let r = straight("factor", lw(&rect(10.1, 7.3), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        let l = length(side);
        let n = ((l - DASH) / PERIOD).floor();
        let want = l / (n * PERIOD + DASH);
        assert!(
            (stretch(side) - want).abs() < 1e-6,
            "side {l} stretches by {} and not {want}: {side:?}",
            stretch(side)
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_the_pattern_keeps_its_shape_when_stretched() {
    let r = straight("shape", lw(&rect(10.3, 10.3), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert!(
            (side.2[0] / side.2[1] - 2.0).abs() < 1e-6,
            "the dash and the gap grow by the same factor: {side:?}"
        );
        assert!(stretch(side) > 1.0, "a side only ever stretches: {side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_an_open_polyline_fits_every_side_too() {
    let pts = [[0.0, 0.0], [10.1, 0.0], [10.1, 7.3]];
    let r = straight("open", lw(&pts, false, false));
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(r.iter().all(ends_on_a_dash), "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_a_2d_polyline_is_fitted_the_same_way() {
    let mut p = Polyline2D::new();
    for q in rect(10.1, 10.1) {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    p.close();
    p.common.linetype = "HIDDEN".into();
    let r = straight("2d", EntityType::Polyline2D(p));
    assert_eq!(r.len(), 4, "{r:?}");
    assert!(r.iter().all(ends_on_a_dash), "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_a_side_that_is_a_dash_and_whole_periods_is_left_alone() {
    // 7.75 is a dash and 20 periods (0.25 + 20 * 0.375), exact in binary.
    let r = straight("exact", lw(&rect(7.75, 7.75), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert_eq!(side.2, vec![DASH, 0.125], "{side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_a_polyline_with_plinegen_on_keeps_its_pattern_as_drawn() {
    let r = straight("on", lw(&rect(10.1, 10.1), true, true));
    // One continuous run, so no two-point sides and nothing fitted.
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].2, vec![DASH, 0.125], "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_a_line_keeps_its_pattern_as_drawn() {
    let mut l = Line::from_coords(0.0, 0.0, 0.0, 10.1, 0.0, 0.0);
    l.common.linetype = "HIDDEN".into();
    let r = straight("line", EntityType::Line(l));
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].2, vec![DASH, 0.125], "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1694_a_side_shorter_than_a_dash_is_left_alone() {
    let r = straight("short", lw(&rect(0.2, 0.2), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert_eq!(side.2, vec![DASH, 0.125], "{side:?}");
    }
}
