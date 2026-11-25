//! A polyline with arc sides is plotted, not refused (#1717).
//!
//! A polyline whose sides carry a bulge is one composite curve, and PVec's writer refuses a
//! composite (`WRT-08`), so a plot that holds only such polylines fails with "PVec's writer
//! refused every curve" and one that holds more loses the polyline. #1691 already writes a
//! polyline as one record per side when PLINEGEN is off; this is the same for a polyline with arcs
//! whatever its PLINEGEN. Each case plots one layout holding one polyline (or converts one drawing)
//! and expects every side, straight or curved, on the page. The drawings are written with
//! acadrust. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;

use acadrust::entities::{Line, LwPolyline};
use acadrust::objects::ObjectType;
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::types::Vector2;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;
use lopdf::content::Operation;

const PT: f64 = 72.0 / 25.4;

fn sheet(entity: EntityType, hidden: bool) -> Vec<u8> {
    let mut doc = CadDocument::new();
    if hidden {
        let mut lt = LineType::new("HIDDEN");
        lt.elements = vec![LineTypeElement::dash(5.0), LineTypeElement::space(2.5)];
        doc.line_types.add(lt).unwrap();
    }
    doc.add_layout("A3").unwrap();
    let layout = doc
        .objects
        .values_mut()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == "A3" => Some(l),
            _ => None,
        })
        .unwrap();
    layout.paper_width = 420.0;
    layout.paper_height = 297.0;
    layout.plot_flags.print_lineweights = true;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "USER_(420.00_x_297.00_MM)".into();
    doc.add_entity_to_layout(entity, "A3").unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn plot(tag: &str, entity: EntityType, hidden: bool) -> Vec<Operation> {
    let dir = std::env::temp_dir().join(format!("libviprs-1717-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let bytes = sheet(entity, hidden);
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .convert_with::<libviprs::pvec::dwg_acadrust::AcadrustProvider>()
        .unwrap_or_else(|e| panic!("{tag}: the plot failed: {e}"));
    let doc = Document::load(&dest).unwrap();
    let id = doc.get_pages()[&1];
    let ops = doc.get_and_decode_page_content(id).unwrap().operations;
    let _ = fs::remove_dir_all(dir);
    ops
}

fn lw(
    points: &[[f64; 2]],
    bulges: &[f64],
    closed: bool,
    linetype: &str,
    plinegen: bool,
) -> EntityType {
    let mut p = LwPolyline::new();
    for (i, q) in points.iter().enumerate() {
        p.add_point_with_bulge(
            Vector2::new(q[0], q[1]),
            bulges.get(i).copied().unwrap_or(0.0),
        );
    }
    p.is_closed = closed;
    p.plinegen = plinegen;
    p.common.linetype = linetype.into();
    EntityType::LwPolyline(p)
}

fn count(ops: &[Operation], op: &str) -> usize {
    ops.iter().filter(|o| o.operator == op).count()
}

/// Every point the page draws to, in points, as (operator, x, y) for the path operators.
fn points(ops: &[Operation]) -> Vec<(String, f64, f64)> {
    let n = |o: &lopdf::Object| o.as_float().map(f64::from).unwrap_or(f64::NAN);
    ops.iter()
        .filter_map(|o| {
            let a = &o.operands;
            match o.operator.as_str() {
                "m" | "l" => Some((o.operator.clone(), n(&a[0]), n(&a[1]))),
                "c" => Some((o.operator.clone(), n(&a[4]), n(&a[5]))),
                _ => None,
            }
        })
        .collect()
}

/// The farthest a curve's end points lie from the horizontal line through where its first curve
/// starts, in millimetres: the sagitta of a half circle on a horizontal chord is its radius.
fn bulge_depth_mm(ops: &[Operation]) -> f64 {
    let pts = points(ops);
    let first = pts.iter().position(|p| p.0 == "c").expect("a curve");
    let y0 = pts[..first].last().expect("a start").2;
    pts.iter()
        .filter(|p| p.0 == "c")
        .map(|p| (p.2 - y0).abs() / PT)
        .fold(0.0, f64::max)
}

const TRI: [[f64; 2]; 3] = [[50.0, 50.0], [150.0, 50.0], [100.0, 150.0]];

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_a_closed_triangle_with_one_bulged_side_plots_the_arc_and_both_straight_sides() {
    // Bulge -1 on the first side: a half circle of radius 50 over the chord, then two straight sides.
    let ops = plot("tri", lw(&TRI, &[-1.0], true, "Continuous", false), false);
    assert_eq!(count(&ops, "l"), 2, "the two straight sides");
    assert_eq!(count(&ops, "c"), 2, "a half circle is two quarter turns");
    let depth = bulge_depth_mm(&ops);
    assert!(
        (depth - 50.0).abs() < 0.5,
        "the arc bulges by its radius, got {depth}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_a_bulge_of_zero_stays_a_straight_polyline() {
    let ops = plot(
        "zero",
        lw(&TRI, &[0.0, 0.0, 0.0], true, "Continuous", false),
        false,
    );
    assert_eq!(count(&ops, "l"), 3, "three straight sides");
    assert_eq!(count(&ops, "c"), 0, "no curve");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_an_open_polyline_with_a_half_circle_side_keeps_its_straight_side() {
    let pts = [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0]];
    let ops = plot("half", lw(&pts, &[1.0], false, "Continuous", false), false);
    assert_eq!(count(&ops, "l"), 1, "the straight side");
    assert_eq!(count(&ops, "c"), 2, "the half circle");
    let depth = bulge_depth_mm(&ops);
    assert!((depth - 50.0).abs() < 0.5, "got {depth}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_a_bulged_side_cut_by_the_sheet_edge_keeps_the_straight_side() {
    // Bulge +1 on a side 25 mm above the bottom edge: the half circle dips 50 mm and leaves the paper.
    let pts = [[50.0, 25.0], [150.0, 25.0], [150.0, 120.0]];
    let ops = plot("edge", lw(&pts, &[1.0], false, "Continuous", false), false);
    assert_eq!(
        count(&ops, "l"),
        1,
        "the straight side is not lost with the arc"
    );
    assert!(count(&ops, "c") >= 1, "the part of the arc on the paper");
    for (op, x, y) in points(&ops) {
        assert!(
            x >= -0.01 && y >= -0.01 && x <= 420.0 * PT + 0.01 && y <= 297.0 * PT + 0.01,
            "{op} {x} {y} is off the sheet"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_a_patterned_polyline_with_arcs_is_plotted_with_plinegen_on_too() {
    // PLINEGEN on asks for one continuous pattern, which a record per side cannot carry. The polyline
    // must still be drawn, every side of it, rather than dropped.
    let ops = plot("on", lw(&TRI, &[-1.0], true, "HIDDEN", true), true);
    assert!(count(&ops, "c") >= 2, "the arc is drawn");
    assert!(count(&ops, "l") >= 2, "and so are the straight sides");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1717_a_sheet_with_a_line_and_a_polyline_with_arcs_draws_both() {
    // The polyline used to be dropped and the line plotted alone, with nothing to say so.
    let dir = std::env::temp_dir().join(format!("libviprs-1717-both-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    let layout = doc
        .objects
        .values_mut()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == "A3" => Some(l),
            _ => None,
        })
        .unwrap();
    layout.paper_width = 420.0;
    layout.paper_height = 297.0;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "USER_(420.00_x_297.00_MM)".into();
    doc.add_entity_to_layout(
        EntityType::Line(Line::from_coords(10.0, 10.0, 0.0, 200.0, 10.0, 0.0)),
        "A3",
    )
    .unwrap();
    doc.add_entity_to_layout(lw(&TRI, &[-1.0], true, "Continuous", false), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .convert_with::<libviprs::pvec::dwg_acadrust::AcadrustProvider>()
        .unwrap_or_else(|e| panic!("the plot failed: {e}"));
    let pdf = Document::load(&dest).unwrap();
    let id = pdf.get_pages()[&1];
    let ops = pdf.get_and_decode_page_content(id).unwrap().operations;
    assert_eq!(count(&ops, "l"), 3, "the line and the two straight sides");
    assert_eq!(count(&ops, "c"), 2, "the half circle");
    let _ = fs::remove_dir_all(dir);
}
