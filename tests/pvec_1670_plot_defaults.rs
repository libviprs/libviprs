//! Issue #1670: a plot with no plot style table is monochrome. Every stroke, text and solid fill is
//! black, and every stroke is one width, 0.25 mm, whatever colour or lineweight the object has (a true
//! colour object included). A table the caller sets still wins, and PlotStyleTable::identity is the
//! way back to the objects' own colours and widths. The drawing is built in memory, so the suite is
//! hermetic. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::{Line, Solid, Text};
use acadrust::objects::ObjectType;
use acadrust::plot_style::PlotStyleTable;
use acadrust::types::{Color, LineWeight, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PdfPlot, PdfPlotOutput};
use lopdf::Document;
use lopdf::content::Operation;

const PT_PER_MM: f64 = 72.0 / 25.4;

fn line(y: f64, color: Color, weight: i16) -> EntityType {
    let mut l = Line::from_coords(10.0, y, 0.0, 100.0, y, 0.0);
    l.common.color = color;
    l.common.line_weight = LineWeight::Value(weight);
    EntityType::Line(l)
}

/// One A3 layout: lines in ACI 1 (0.25 mm), ACI 5 (0.5 mm), ACI 7 (0.13 mm), a true colour line
/// (0.7 mm), a line with a zero lineweight, a red TEXT and a green SOLID.
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
    let mut text = Text::with_value("Plan", Vector3::new(20.0, 20.0, 0.0)).with_height(5.0);
    text.common.color = Color::Index(1);
    let mut solid = Solid::new(
        Vector3::new(300.0, 50.0, 0.0),
        Vector3::new(330.0, 50.0, 0.0),
        Vector3::new(300.0, 80.0, 0.0),
        Vector3::new(330.0, 80.0, 0.0),
    );
    solid.common.color = Color::Index(3);
    for e in [
        line(10.0, Color::Index(1), 25),
        line(20.0, Color::Index(5), 50),
        line(30.0, Color::Index(7), 13),
        line(
            40.0,
            Color::Rgb {
                r: 200,
                g: 30,
                b: 30,
            },
            70,
        ),
        line(50.0, Color::Index(2), 0),
        EntityType::Text(text),
        EntityType::Solid(solid),
    ] {
        doc.add_entity_to_layout(e, "A3").unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1670d-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Plots the drawing, with a table when one is given, and returns the output and the page's operators.
fn plot(tag: &str, table: Option<PlotStyleTable>) -> (PdfPlotOutput, Vec<Operation>) {
    let dir = scratch(tag);
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let mut p = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .texts(true)
        .fills(true);
    if let Some(t) = table {
        p = p.plot_style_table(t);
    }
    let out = p.convert_with::<AcadrustProvider>().unwrap();
    let doc = Document::load(&dest).unwrap();
    let id = doc.get_pages()[&1];
    let ops = doc.get_and_decode_page_content(id).unwrap().operations;
    (out, ops)
}

fn real(o: &lopdf::Object) -> f64 {
    o.as_float()
        .map(f64::from)
        .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
}

/// Each stroked path's colour (as 0 to 255 bytes) and width in points.
fn strokes(ops: &[Operation]) -> Vec<([u8; 3], f64)> {
    let (mut rgb, mut w) = ([0u8; 3], 0.0);
    let mut out = Vec::new();
    for o in ops {
        match o.operator.as_str() {
            "RG" => {
                let v: Vec<f64> = o.operands.iter().map(real).collect();
                rgb = [v[0], v[1], v[2]].map(|c| (c * 255.0).round() as u8);
            }
            "w" => w = real(&o.operands[0]),
            "S" => out.push((rgb, w)),
            _ => {}
        }
    }
    out
}

/// The non-stroking colour in force at each fill ("f", "f*") and each text show ("Tj").
fn fills_and_texts(ops: &[Operation]) -> Vec<(&str, [u8; 3])> {
    let mut rgb = [0u8; 3];
    let mut out = Vec::new();
    for o in ops {
        match o.operator.as_str() {
            "rg" => {
                let v: Vec<f64> = o.operands.iter().map(real).collect();
                rgb = [v[0], v[1], v[2]].map(|c| (c * 255.0).round() as u8);
            }
            "f" | "f*" => out.push(("fill", rgb)),
            "Tj" => out.push(("text", rgb)),
            _ => {}
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_no_table_strokes_every_object_black_at_a_quarter_millimetre() {
    let (_, ops) = plot("mono", None);
    let s = strokes(&ops);
    assert!(s.len() >= 5, "five lines and a fill outline: {s:?}");
    for (colour, width) in &s {
        assert_eq!(*colour, [0, 0, 0], "a stroke that is not black in {s:?}");
        assert!(
            (width - 0.25 * PT_PER_MM).abs() < 0.01,
            "a stroke that is not 0.25 mm ({} pt) in {s:?}",
            0.25 * PT_PER_MM
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_no_table_a_true_colour_line_and_a_zero_lineweight_line_are_black_at_a_quarter_millimetre()
 {
    // The red true colour line is 0.7 mm and the ACI 2 line has a zero lineweight (the thinnest line):
    // under the built-in default neither keeps what it had.
    let (_, ops) = plot("true", None);
    let s = strokes(&ops);
    assert!(
        !s.iter()
            .any(|(c, w)| *c == [200, 30, 30] || *w == 0.0 || *w > 0.3 * PT_PER_MM),
        "an object kept its own colour or width in {s:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_no_table_text_and_a_solid_fill_are_black() {
    let (_, ops) = plot("textfill", None);
    let f = fills_and_texts(&ops);
    assert!(f.iter().any(|(k, _)| *k == "text"), "no text in {f:?}");
    assert!(f.iter().any(|(k, _)| *k == "fill"), "no fill in {f:?}");
    assert!(
        f.iter().all(|(_, c)| *c == [0, 0, 0]),
        "a text or fill that is not black in {f:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_an_identity_table_still_draws_the_objects_own_colours_and_widths() {
    let (_, ops) = plot("identity", Some(PlotStyleTable::identity("own.ctb")));
    let s = strokes(&ops);
    let has = |c: [u8; 3], mm: f64| {
        s.iter()
            .any(|(sc, w)| *sc == c && (w - mm * PT_PER_MM).abs() < 0.01)
    };
    assert!(has([255, 0, 0], 0.25), "no red 0.25 mm stroke in {s:?}");
    assert!(has([0, 0, 255], 0.5), "no blue 0.5 mm stroke in {s:?}");
    assert!(
        has([200, 30, 30], 0.7),
        "no true colour 0.7 mm stroke in {s:?}"
    );
    let f = fills_and_texts(&ops);
    assert!(
        f.iter().any(|(_, c)| *c != [0, 0, 0]),
        "the text and fill are black under an identity table: {f:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_table_the_caller_sets_beats_the_built_in_default() {
    let mut t = PlotStyleTable::identity("blue.ctb");
    t.aci_entries[1].color = Some([10, 20, 30]);
    let (_, ops) = plot("explicit", Some(t));
    let s = strokes(&ops);
    assert!(
        s.iter().any(|(c, _)| *c == [10, 20, 30]),
        "ACI 1 is not 10 20 30 in {s:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_notes_name_the_built_in_default_only_when_it_was_used() {
    let (none, _) = plot("note-none", None);
    assert!(
        none.notes
            .iter()
            .any(|n| n == "plot-style-default-monochrome"),
        "{:?}",
        none.notes
    );
    let (some, _) = plot("note-some", Some(PlotStyleTable::identity("own.ctb")));
    assert!(
        !some
            .notes
            .iter()
            .any(|n| n == "plot-style-default-monochrome"),
        "{:?}",
        some.notes
    );
}
