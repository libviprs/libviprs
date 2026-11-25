//! A polyline's width plots as a width, not a hairline (#1718).
//!
//! LWPOLYLINE carries a constant width and a start and end width on each vertex, the 2D POLYLINE
//! carries a start and end width on each vertex and a default for the entity. The converter read
//! none of them, so a wide polyline plotted as a hairline. A polyline of one width plots as a
//! stroke of that width in paper millimetres, after the plot scale. A polyline whose width changes
//! along it is drawn as the outline of the polygon it covers, with a bulged side following its
//! arc. A width of 0 changes nothing. The width is geometry, so a layout that does not print
//! lineweights still plots it. The drawings are written with acadrust on the paper layout
//! `Sheet`. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::fs;

use acadrust::entities::{Line, LwPolyline, LwVertex, Polyline2D, Vertex2D};
use acadrust::objects::ObjectType;
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PvecConversion;
use plan::scratch;
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, StrokeWidth};
use pvec_geometry::Curve2d;

/// What one record plots: its stroke width in paper millimetres (`None` for the thinnest line) and
/// its vertices in paper millimetres when it is a polyline, and the height of its box.
type Record = (Option<f64>, Option<Vec<[f64; 2]>>, f64);

/// A sheet whose plot scale is `paper` paper millimetres to `drawing` drawing units, holding just
/// `entity` and a plain line far from it, so the drawing is never empty.
fn sheet(entity: EntityType, paper: f64, drawing: f64, print_lineweights: bool) -> Vec<u8> {
    let mut doc = CadDocument::new();
    let layout = doc.add_layout("Sheet").unwrap();
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ANSI_B_(11.00_x_17.00_Inches)".into();
    l.paper_width = 279.4;
    l.paper_height = 431.8;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = paper;
    l.plot_scale_denominator = drawing;
    l.plot_rotation = 0;
    l.plot_margin_left = 0.0;
    l.plot_margin_bottom = 0.0;
    l.plot_origin_x = 0.0;
    l.plot_origin_y = 0.0;
    l.plot_flags.print_lineweights = print_lineweights;
    doc.add_entity_to_layout(entity, "Sheet").unwrap();
    doc.add_entity_to_layout(
        EntityType::Line(Line::from_coords(100.0, 100.0, 0.0, 110.0, 100.0, 0.0)),
        "Sheet",
    )
    .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

/// The records of `entity` on a sheet at `paper` mm to `drawing` units, the plain line left out.
fn plot(tag: &str, entity: EntityType, paper: f64, drawing: f64, lineweights: bool) -> Vec<Record> {
    let bytes = sheet(entity, paper, drawing, lineweights);
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .components(StreamSelection::ALL)
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let reader = open_published(&dest).unwrap();
    let mm = reader.manifest().metres_per_unit * 1000.0;
    let streams = reader.streams().unwrap().0;
    let style = LineStyleStream::decode(
        &streams
            .iter()
            .find(|s| s.capability == LineStyleStream::CAPABILITY)
            .expect("the line style stream")
            .payload,
    )
    .unwrap();
    let widths: std::collections::BTreeMap<u32, Option<f64>> = style
        .assignments
        .iter()
        .map(|(id, k)| {
            let w = match style.styles[*k as usize].width {
                Some(StrokeWidth::Width(w)) => Some(w * mm),
                _ => None,
            };
            (id.index(), w)
        })
        .collect();
    let per_unit = paper / drawing;
    let out = reader
        .entities()
        .unwrap()
        .iter()
        .filter_map(|(id, curve)| {
            let b = curve.bounds().unwrap();
            // the plain line lies 100 units up, far from every polyline here
            if b.min().y / mm / per_unit > 50.0 {
                return None;
            }
            let vertices = match curve {
                Curve::Polyline(p) => {
                    Some(p.vertices().iter().map(|v| [v.x * mm, v.y * mm]).collect())
                }
                _ => None,
            };
            let height = (b.max().y - b.min().y) * mm;
            Some((widths.get(&id.index()).copied().flatten(), vertices, height))
        })
        .collect();
    let _ = fs::remove_dir_all(dir);
    out
}

fn lw(
    points: &[[f64; 2]],
    bulges: &[f64],
    widths: &[[f64; 2]],
    constant: f64,
    closed: bool,
) -> EntityType {
    let mut p = LwPolyline::new();
    for (i, q) in points.iter().enumerate() {
        let mut v = LwVertex::new(Vector2::new(q[0], q[1]));
        v.bulge = bulges.get(i).copied().unwrap_or(0.0);
        let [s, e] = widths.get(i).copied().unwrap_or([0.0, 0.0]);
        v.start_width = s;
        v.end_width = e;
        p.add_vertex(v);
    }
    p.constant_width = constant;
    p.is_closed = closed;
    EntityType::LwPolyline(p)
}

/// Twice the signed area the vertices enclose, taken as a closed ring, as an area.
fn area(v: &[[f64; 2]]) -> f64 {
    let n = v.len();
    (0..n)
        .map(|i| v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1])
        .sum::<f64>()
        .abs()
        / 2.0
}

fn only(r: &[Record]) -> &Record {
    assert_eq!(r.len(), 1, "one record for the polyline, got {r:?}");
    &r[0]
}

fn assert_near(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} +- {tol}"
    );
}

const RUN: [[f64; 2]; 3] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_constant_width_is_the_stroke_width() {
    let r = plot("constant", lw(&RUN, &[], &[], 0.4, false), 1.0, 1.0, true);
    let (w, v, _) = only(&r);
    assert_near(
        w.expect("a stroke width, not a hairline"),
        0.4,
        1e-6,
        "width in mm",
    );
    assert_eq!(
        v.as_ref().expect("a polyline").len(),
        3,
        "the path is kept as drawn: {r:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_equal_start_and_end_widths_on_the_vertices_are_the_same_stroke() {
    let widths = [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]];
    let r = plot(
        "vertex-equal",
        lw(&RUN, &[], &widths, 0.0, false),
        1.0,
        1.0,
        true,
    );
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.4,
        1e-6,
        "width in mm",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_the_width_is_paper_millimetres_after_the_plot_scale() {
    // 1 : 0.04 is 25 mm of paper to a drawing unit, so 0.016 units is 0.4 mm.
    let r = plot("scaled", lw(&RUN, &[], &[], 0.016, false), 1.0, 0.04, true);
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.4,
        1e-6,
        "width in mm",
    );
    // 1 : 50 is 0.02 mm of paper to a unit, so 20 units is 0.4 mm.
    let r = plot(
        "scaled-down",
        lw(&RUN, &[], &[], 20.0, false),
        1.0,
        50.0,
        true,
    );
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.4,
        1e-6,
        "width in mm",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_closed_polyline_of_one_width_is_one_stroke_with_its_closing_side() {
    let r = plot(
        "closed",
        lw(
            &[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            &[],
            &[],
            0.4,
            true,
        ),
        1.0,
        1.0,
        true,
    );
    let (w, v, _) = only(&r);
    assert_near(w.expect("a stroke width"), 0.4, 1e-6, "width in mm");
    assert!(v.as_ref().expect("a polyline").len() >= 4, "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_the_width_prints_when_the_layout_does_not_print_lineweights() {
    let r = plot(
        "no-lineweights",
        lw(&RUN, &[], &[], 0.4, false),
        1.0,
        1.0,
        false,
    );
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.4,
        1e-6,
        "width in mm",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_constant_width_on_a_bulged_side_keeps_the_arc_and_the_width() {
    let r = plot(
        "bulged-constant",
        lw(&[[0.0, 0.0], [10.0, 0.0]], &[1.0, 0.0], &[], 0.4, false),
        1.0,
        1.0,
        true,
    );
    let (w, _, height) = only(&r);
    assert_near(w.expect("a stroke width"), 0.4, 1e-6, "width in mm");
    // a semicircle of radius 5 reaches 5 mm off its chord
    assert_near(*height, 5.0, 0.05, "the arc's height");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_taper_is_the_outline_of_its_polygon() {
    // 2 wide at the start, a point at the end, over 10: a triangle of area 10.
    let r = plot(
        "taper-point",
        lw(
            &[[0.0, 0.0], [10.0, 0.0]],
            &[],
            &[[2.0, 0.0], [0.0, 0.0]],
            0.0,
            false,
        ),
        1.0,
        1.0,
        true,
    );
    let v = only(&r).1.as_ref().expect("an outline");
    assert_near(area(v), 10.0, 1e-6, "area of the taper");
    let (lo, hi) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), p| (l.min(p[1]), h.max(p[1])));
    assert_near(lo, -1.0, 1e-6, "lowest y");
    assert_near(hi, 1.0, 1e-6, "highest y");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_taper_between_two_widths_is_a_trapezoid() {
    // 1 wide to 3 wide over 10: area (1 + 3) / 2 * 10.
    let r = plot(
        "taper-trapezoid",
        lw(
            &[[0.0, 0.0], [10.0, 0.0]],
            &[],
            &[[1.0, 3.0], [0.0, 0.0]],
            0.0,
            false,
        ),
        1.0,
        1.0,
        true,
    );
    assert_near(
        area(only(&r).1.as_ref().expect("an outline")),
        20.0,
        1e-6,
        "area of the taper",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_taper_is_scaled_by_the_plot_scale() {
    // 25 mm a unit: a 1 by 3 taper over 10 units is 25 by 75 over 250 mm, area 12500.
    let r = plot(
        "taper-scaled",
        lw(
            &[[0.0, 0.0], [10.0, 0.0]],
            &[],
            &[[1.0, 3.0], [0.0, 0.0]],
            0.0,
            false,
        ),
        1.0,
        0.04,
        true,
    );
    assert_near(
        area(only(&r).1.as_ref().expect("an outline")),
        12500.0,
        1.0,
        "area in mm2",
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_taper_on_a_bulged_side_follows_the_arc() {
    // A semicircle of radius 5 that widens from 0.5 to 1.5: the area it covers is the mean width
    // 1 times the arc's length 5 pi, up to the arc's flattening.
    let r = plot(
        "taper-bulged",
        lw(
            &[[0.0, 0.0], [10.0, 0.0]],
            &[1.0, 0.0],
            &[[0.5, 1.5], [0.0, 0.0]],
            0.0,
            false,
        ),
        1.0,
        1.0,
        true,
    );
    let v = only(&r).1.as_ref().expect("an outline");
    assert_near(
        area(v),
        5.0 * std::f64::consts::PI,
        0.3,
        "area of the curved taper",
    );
    let top = v.iter().map(|p| p[1].abs()).fold(0.0, f64::max);
    assert!(
        top > 5.0,
        "the outline follows the arc out to its height, not the chord: {top}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_polyline_that_is_wide_on_one_side_only_is_an_outline() {
    // The first side is 0.4 wide and the second has none: the width changes along the polyline, so
    // the outline of the wide side is drawn and the area is that side's 0.4 by 10.
    let widths = [[0.4, 0.4], [0.0, 0.0], [0.0, 0.0]];
    let r = plot(
        "one-side",
        lw(&RUN, &[], &widths, 0.0, false),
        1.0,
        1.0,
        true,
    );
    let covered: f64 = r
        .iter()
        .filter_map(|(_, v, _)| v.as_ref())
        .map(|v| area(v))
        .sum();
    assert_near(covered, 4.0, 0.05, "area covered by the wide side");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_width_of_zero_changes_nothing() {
    let r = plot("zero", lw(&RUN, &[], &[], 0.0, false), 1.0, 1.0, true);
    let (w, v, _) = only(&r);
    assert_eq!(*w, None, "no width: {r:?}");
    assert_eq!(v.as_ref().expect("a polyline").len(), 3, "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1718_a_2d_polyline_takes_its_vertex_widths_and_its_default_width() {
    let mut p = Polyline2D::new();
    for q in RUN {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    for v in &mut p.vertices {
        v.start_width = 0.4;
        v.end_width = 0.4;
    }
    let r = plot("pl2d-vertex", EntityType::Polyline2D(p), 1.0, 1.0, true);
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.4,
        1e-6,
        "width in mm",
    );

    let mut p = Polyline2D::new();
    for q in RUN {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    p.start_width = 0.6;
    p.end_width = 0.6;
    let r = plot("pl2d-default", EntityType::Polyline2D(p), 1.0, 1.0, true);
    assert_near(
        only(&r).0.expect("a stroke width"),
        0.6,
        1e-6,
        "default width in mm",
    );
}
