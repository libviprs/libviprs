//! Issue #1723: a polyline's own width is geometry, so the built-in default plot style (black at
//! 0.25 mm, #1670) leaves it alone. A constant width plots at that width with no table, with an
//! identity table and with an explicit table of another pen width. A polyline of width 0 still
//! takes the table's width (the default's 0.25 mm with none). The drawings are written in memory
//! with acadrust on one A3 layout at 1:1 and plotted with `PdfPlot`. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::{Line, LwPolyline, LwVertex};
use acadrust::objects::ObjectType;
use acadrust::plot_style::PlotStyleTable;
use acadrust::types::{Color, LineWeight, Vector2};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;
use lopdf::content::Operation;

const PT_PER_MM: f64 = 72.0 / 25.4;

/// A polyline along y = `y` with `n` vertices 40 units apart, of constant width `width`, in ACI 1
/// with a 0.70 mm lineweight of its own.
fn polyline(y: f64, n: usize, width: f64) -> EntityType {
    let mut p = LwPolyline::new();
    for i in 0..n {
        p.add_vertex(LwVertex::new(Vector2::new(20.0 + 40.0 * i as f64, y)));
    }
    p.constant_width = width;
    p.common.color = Color::Index(1);
    p.common.line_weight = LineWeight::Value(70);
    EntityType::LwPolyline(p)
}

/// A3, 1:1: a 0.8 mm wide polyline of three vertices, a 1.2 mm wide one of two (a straight wide
/// polyline), one of width 0 and a plain line with the same 0.70 mm lineweight.
fn drawing() -> Vec<u8> {
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
    layout.plot_flags.print_lineweights = true;
    layout.plot_flags.plot_plot_styles = true;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "USER_(420.00_x_297.00_MM)".into();
    let mut line = Line::from_coords(20.0, 120.0, 0.0, 100.0, 120.0, 0.0);
    line.common.color = Color::Index(1);
    line.common.line_weight = LineWeight::Value(70);
    for e in [
        polyline(30.0, 3, 0.8),
        polyline(60.0, 2, 1.2),
        polyline(90.0, 3, 0.0),
        EntityType::Line(line),
    ] {
        doc.add_entity_to_layout(e, "A3").unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1723-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The distinct widths in millimetres the paths are stroked at, sorted. A polyline with a table is
/// one record a side (#1691), so the same width can come more than once.
fn widths(tag: &str, table: Option<PlotStyleTable>) -> Vec<f64> {
    let dir = scratch(tag);
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let mut p = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"]);
    if let Some(t) = table {
        p = p.plot_style_table(t);
    }
    p.convert_with::<AcadrustProvider>().unwrap();
    let doc = Document::load(&dest).unwrap();
    let id = doc.get_pages()[&1];
    let ops: Vec<Operation> = doc.get_and_decode_page_content(id).unwrap().operations;
    let real = |o: &lopdf::Object| {
        o.as_float()
            .map(f64::from)
            .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
    };
    let mut w = 0.0;
    let mut out = Vec::new();
    for o in &ops {
        match o.operator.as_str() {
            "w" => w = real(&o.operands[0]),
            "S" => out.push(w / PT_PER_MM),
            _ => {}
        }
    }
    let _ = fs::remove_dir_all(dir);
    out.sort_by(f64::total_cmp);
    out.dedup_by(|a, b| (*a - *b).abs() < 0.005);
    out
}

fn assert_widths(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 0.005, "{what}: {got:?} against {want:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_with_no_table_a_constant_width_polyline_plots_at_its_own_width() {
    let got = widths("none", None);
    assert!(
        got.iter().any(|w| (w - 0.8).abs() < 0.005),
        "the 0.8 mm polyline is not 0.8 mm in {got:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_with_no_table_a_straight_wide_polyline_plots_at_its_own_width() {
    let got = widths("none-straight", None);
    assert!(
        got.iter().any(|w| (w - 1.2).abs() < 0.005),
        "the 1.2 mm straight polyline is not 1.2 mm in {got:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_with_no_table_everything_else_stays_at_the_defaults_quarter_millimetre() {
    // The width 0 polyline and the 0.70 mm line take the default's 0.25 mm: the polyline's own
    // width is the only thing that is let through.
    let got = widths("none-rest", None);
    assert_widths(&got, &[0.25, 0.8, 1.2], "no table");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_with_an_identity_table_a_constant_width_polyline_plots_at_its_own_width() {
    let got = widths("identity", Some(PlotStyleTable::identity("own.ctb")));
    assert_widths(&got, &[0.7, 0.8, 1.2], "identity table");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_with_a_table_of_another_pen_width_a_constant_width_polyline_keeps_its_own_width() {
    let mut t = PlotStyleTable::identity("wide.ctb");
    // Lineweight 15 is the table's entry 14.
    let pen = f64::from(t.lineweights[14]);
    assert!(pen > 0.4 && pen < 0.7, "the pen is {pen}");
    t.aci_entries[1].lineweight = 15;
    let got = widths("explicit", Some(t));
    assert_widths(&got, &[pen, 0.8, 1.2], "explicit table");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1723_a_width_zero_polyline_still_takes_the_tables_width() {
    // None: the default's 0.25 mm. Identity: the object's own 0.70 mm. Explicit: the table's pen.
    let none = widths("zero-none", None);
    assert!(
        none.iter().any(|w| (w - 0.25).abs() < 0.005),
        "no 0.25 mm stroke with no table in {none:?}"
    );
    let mut t = PlotStyleTable::identity("wide.ctb");
    let pen = f64::from(t.lineweights[14]);
    t.aci_entries[1].lineweight = 15;
    let explicit = widths("zero-explicit", Some(t));
    assert!(
        explicit.iter().any(|w| (w - pen).abs() < 0.005),
        "no {pen} mm stroke with the explicit table in {explicit:?}"
    );
    assert!(
        !explicit.iter().any(|w| (w - 0.25).abs() < 0.005),
        "the default's 0.25 mm leaked into {explicit:?}"
    );
}
