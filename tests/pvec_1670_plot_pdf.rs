//! Issue #1670: a drawing's print layouts plotted to one PDF with a plot style table, as the page a
//! plotter would draw. The drawing, the .ctb and the .pc3 are all built in memory (acadrust writes the
//! DWG, `PlotStyleTable::to_bytes` the CTB), so the suite is hermetic and needs no private file. The
//! owner's own drawing is checked by `pvec_1670_owner_layouts.rs`, which skips unless it is pointed at
//! the files. Needs `--features pvec-convert`.
//!
//! Page sizes and strokes are `PdfPlot`'s from #1636. What these tests pin is the rest of the page: a
//! circle is a Bezier path and not 96 chords, a dashed linetype is dashed, a zero lineweight is the
//! thinnest line, a SOLID is filled, text is text, a rotated layout is a rotated page, and a plot
//! can be cancelled, bounded and read its unit from the file.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::{Circle, Line, RasterImage, Solid, Text, Viewport};
use acadrust::objects::{ImageDefinition, ObjectType};
use acadrust::plot_style::PlotStyleTable;
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::types::{Color, LineWeight, Transparency, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceLimits, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PdfPlot, PdfPlotOutput, PvecConvertError};
use lopdf::Document;
use lopdf::content::Operation;

const PT_PER_MM: f64 = 72.0 / 25.4;

/// One layout's page setup: name, paper in mm, rotation code.
struct Sheet(&'static str, f64, f64, i16);

fn set_plot(doc: &mut CadDocument, s: &Sheet) {
    let layout = doc
        .objects
        .values_mut()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == s.0 => Some(l),
            _ => None,
        })
        .unwrap();
    layout.paper_width = s.1;
    layout.paper_height = s.2;
    layout.plot_rotation = s.3;
    layout.plot_flags.print_lineweights = true;
    layout.plot_flags.plot_plot_styles = true;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = format!("USER_({:.2}_x_{:.2}_MM)", s.1, s.2);
}

fn line(x0: f64, y0: f64, x1: f64, y1: f64, aci: u8, linetype: &str) -> EntityType {
    let mut l = Line::from_coords(x0, y0, 0.0, x1, y1, 0.0);
    l.common.color = Color::Index(aci);
    l.common.line_weight = LineWeight::Value(25);
    l.common.linetype = linetype.into();
    EntityType::Line(l)
}

/// Two paper layouts: `A3` (420 x 297, landscape, unrotated) holding a circle, a red line, a dashed
/// blue line, a SOLID and a TEXT, and `A4` (210 x 297, rotation 1) holding one line.
fn drawing() -> Vec<u8> {
    let mut doc = CadDocument::new();
    let mut dashed = LineType::new("DASHED");
    dashed.elements = vec![LineTypeElement::dash(10.0), LineTypeElement::space(5.0)];
    doc.line_types.add(dashed).unwrap();
    doc.add_layout("A3").unwrap();
    doc.add_layout("A4").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    set_plot(&mut doc, &Sheet("A4", 210.0, 297.0, 1));
    let mut circle = Circle::from_coords(200.0, 150.0, 0.0, 40.0);
    circle.common.color = Color::Index(7);
    let mut text = Text::with_value("Plan", Vector3::new(20.0, 20.0, 0.0)).with_height(5.0);
    text.common.color = Color::Index(7);
    let solid = Solid::new(
        Vector3::new(300.0, 50.0, 0.0),
        Vector3::new(330.0, 50.0, 0.0),
        Vector3::new(300.0, 80.0, 0.0),
        Vector3::new(330.0, 80.0, 0.0),
    );
    for e in [
        EntityType::Circle(circle),
        line(10.0, 10.0, 100.0, 10.0, 1, "Continuous"),
        line(10.0, 30.0, 100.0, 30.0, 5, "DASHED"),
        EntityType::Solid(solid),
        EntityType::Text(text),
    ] {
        doc.add_entity_to_layout(e, "A3").unwrap();
    }
    doc.add_entity_to_layout(line(10.0, 10.0, 100.0, 200.0, 7, "Continuous"), "A4")
        .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

/// A CTB: ACI 1 plots black at 0.35 mm, ACI 5 keeps its colour at lineweight index 0 (the object's,
/// then the thinnest the device draws once the object's own is zero), the rest is the object's.
fn ctb() -> PlotStyleTable {
    let mut t = PlotStyleTable::identity("mono.ctb");
    t.lineweights = vec![0.0, 0.35, 0.5];
    t.aci_entries[1].color = Some([0, 0, 0]);
    t.aci_entries[1].lineweight = 2;
    t
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1670-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn plot(
    tag: &str,
    configure: impl for<'a> FnOnce(PdfPlot<'a>) -> PdfPlot<'a>,
) -> (Result<PdfPlotOutput, PvecConvertError>, PathBuf) {
    let dir = scratch(tag);
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let p = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .plot_style_table(ctb())
        .layouts(["A3", "A4"])
        .texts(true)
        .fills(true);
    (
        configure(p).convert_with::<AcadrustProvider>(),
        dir.join("out.pdf"),
    )
}

fn ops(pdf: &PathBuf, page: u32) -> Vec<Operation> {
    let doc = Document::load(pdf).unwrap();
    let id = doc.get_pages()[&page];
    doc.get_and_decode_page_content(id).unwrap().operations
}

fn count(ops: &[Operation], name: &str) -> usize {
    ops.iter().filter(|o| o.operator == name).count()
}

fn real(o: &lopdf::Object) -> f64 {
    o.as_float()
        .map(f64::from)
        .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
}

// ---- the page ---------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_one_page_per_layout_at_the_layouts_paper() {
    let (out, pdf) = plot("pages", |p| p);
    assert_eq!(out.unwrap().pages, 2);
    let info = libviprs::pdf_info(&pdf).unwrap();
    let near = |i: usize, w: f64, h: f64| {
        let p = &info.pages[i];
        assert!(
            (p.width_pts - w * PT_PER_MM).abs() < 0.2 && (p.height_pts - h * PT_PER_MM).abs() < 0.2,
            "page {i}: {} x {} pt is not {w} x {h} mm",
            p.width_pts,
            p.height_pts
        );
    };
    near(0, 420.0, 297.0);
    // stored 210 x 297 at rotation 1: shown as the landscape frame, from a portrait box
    near(1, 297.0, 210.0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_circle_is_bezier_segments_not_chords() {
    let (out, pdf) = plot("circle", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    let curves = count(&o, "c");
    let lines = count(&o, "l");
    assert!(
        (4..=16).contains(&curves),
        "{curves} curve operators for a circle"
    );
    assert!(
        lines < 40,
        "{lines} line operators: the circle is drawn as chords"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_dashed_linetype_sets_a_dash_pattern() {
    let (out, pdf) = plot("dash", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    let dashes: Vec<&Operation> = o
        .iter()
        .filter(|o| o.operator == "d" && o.operands[0].as_array().is_ok_and(|a| !a.is_empty()))
        .collect();
    assert_eq!(dashes.len(), 1, "one dashed stroke");
    let pattern: Vec<f64> = dashes[0].operands[0]
        .as_array()
        .unwrap()
        .iter()
        .map(real)
        .collect();
    assert!(
        (pattern[0] - 10.0 * PT_PER_MM).abs() < 0.1 && (pattern[1] - 5.0 * PT_PER_MM).abs() < 0.1,
        "{pattern:?} pt is not the 10 mm dash and 5 mm gap"
    );
}

/// Each stroked path's colour (as 0 to 255 bytes), width in points and dash array.
fn strokes(ops: &[Operation]) -> Vec<([u8; 3], f64, Vec<f64>)> {
    let (mut rgb, mut w, mut dash) = ([0u8; 3], 0.0, Vec::new());
    let mut out = Vec::new();
    for o in ops {
        match o.operator.as_str() {
            "RG" => {
                let v: Vec<f64> = o.operands.iter().map(real).collect();
                rgb = [v[0], v[1], v[2]].map(|c| (c * 255.0).round() as u8);
            }
            "w" => w = real(&o.operands[0]),
            "d" => dash = o.operands[0].as_array().unwrap().iter().map(real).collect(),
            "S" => out.push((rgb, w, dash.clone())),
            _ => {}
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_ctb_sets_each_colours_stroke_colour_and_width() {
    // ACI 1 plots black at the table's second weight and ACI 5 keeps blue at the object's 0.25 mm.
    let mut t = ctb();
    t.lineweights = vec![0.0, 0.35, 0.35];
    let (out, pdf) = plot("ctb", |p| p.plot_style_table(t));
    out.unwrap();
    let s = strokes(&ops(&pdf, 1));
    assert!(
        s.iter()
            .any(|(c, w, _)| *c == [0, 0, 0] && (w - 0.35 * PT_PER_MM).abs() < 0.01),
        "no black 0.35 mm stroke in {s:?}"
    );
    assert!(
        s.iter()
            .any(|(c, w, _)| *c == [0, 0, 255] && (w - 0.25 * PT_PER_MM).abs() < 0.01),
        "no blue 0.25 mm stroke in {s:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_zero_lineweight_is_the_thinnest_line() {
    // The table's weight for ACI 5 is its first entry, 0.0 mm: the thinnest line the device draws.
    let mut t = ctb();
    t.aci_entries[5].lineweight = 1;
    t.lineweights = vec![0.0, 0.0, 0.5];
    let (out, pdf) = plot("zero", |p| p.plot_style_table(t));
    out.unwrap();
    let s = strokes(&ops(&pdf, 1));
    assert!(
        s.iter().any(|(c, w, _)| *c == [0, 0, 255] && *w == 0.0),
        "no blue 0 w stroke in {s:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_solid_is_filled() {
    let (out, pdf) = plot("fill", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    assert!(count(&o, "f") + count(&o, "f*") >= 1, "no fill operator");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_text_is_text() {
    let (out, pdf) = plot("text", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    assert_eq!(count(&o, "BT"), 1);
    let shown: Vec<u8> = o
        .iter()
        .filter(|o| o.operator == "Tj")
        .flat_map(|o| o.operands[0].as_str().unwrap().to_vec())
        .collect();
    assert_eq!(shown, b"Plan");
    let doc = Document::load(&pdf).unwrap();
    let id = doc.get_pages()[&1];
    let fonts = doc.get_page_fonts(id).unwrap();
    assert_eq!(fonts.len(), 1, "one font on the page");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_rotated_layout_is_a_rotated_page() {
    // Layout A4 is portrait paper with plot rotation 1: the plotted sheet (the frame) is landscape. The
    // box is the portrait paper and the page carries /Rotate, the way a plotter's own PDF does.
    let (out, pdf) = plot("rotate", |p| p);
    out.unwrap();
    let doc = Document::load(&pdf).unwrap();
    let id = doc.get_pages()[&2];
    let page = doc.get_dictionary(id).unwrap();
    let rotate = page.get(b"Rotate").and_then(|r| r.as_i64()).unwrap();
    assert!(rotate == 90 || rotate == 270, "/Rotate {rotate}");
    let media: Vec<f64> = page
        .get(b"MediaBox")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(real)
        .collect();
    assert!(
        (media[2] - 210.0 * PT_PER_MM).abs() < 0.2 && (media[3] - 297.0 * PT_PER_MM).abs() < 0.2,
        "{media:?}"
    );
    let first = doc.get_dictionary(doc.get_pages()[&1]).unwrap();
    assert_eq!(first.get(b"Rotate").and_then(|r| r.as_i64()).unwrap(), 0);
}

// ---- bounds, cancel and the unit --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_cancelled_plot_stops_and_leaves_no_file() {
    let token = libviprs::CancelToken::new();
    token.cancel();
    let (out, pdf) = plot("cancel", |p| p.cancel(token));
    assert!(matches!(out, Err(PvecConvertError::Cancelled)), "{out:?}");
    assert!(!pdf.exists());
    assert!(!pdf.with_extension("plot-work").exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_source_limits_bound_the_plot() {
    let mut limits = SourceLimits::default();
    limits.max_entities = 2;
    let (out, pdf) = plot("limit", |p| p.limits(limits));
    assert!(
        out.is_err(),
        "a limit of two entities plots a layout of five"
    );
    assert!(!pdf.exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_unit_comes_from_the_file_when_none_is_given() {
    let dir = scratch("unit");
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let out = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .convert()
        .unwrap();
    assert_eq!(out.pages, 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_without_unit_convert_with_still_asks_for_one() {
    let dir = scratch("nounit");
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let out = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .view_profile(ViewProfile::print_profile("A3"))
        .convert_with::<AcadrustProvider>();
    assert!(matches!(out, Err(PvecConvertError::UnitRequired)));
}

// ---- images -----------------------------------------------------------------------------

/// A sheet like `A3` with a 4 x 4 pixel IMAGE of 10 mm a pixel, its lower-left corner at (50, 40), whose
/// file is `logo.png` in the returned directory.
fn image_plot(tag: &str, with_dir: bool) -> PathBuf {
    let dir = scratch(tag);
    image::RgbaImage::from_fn(4, 4, |x, y| {
        image::Rgba([(60 * x + 30) as u8, (60 * y + 30) as u8, 128, 255])
    })
    .save(dir.join("logo.png"))
    .unwrap();
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let mut def = ImageDefinition::new("logo.png");
    def.handle = doc.allocate_handle();
    def.size_in_pixels = (4, 4);
    def.pixel_size = (1.0, 1.0);
    def.is_loaded = true;
    let handle = def.handle;
    doc.objects.insert(handle, ObjectType::ImageDefinition(def));
    let mut img = RasterImage::new("logo.png", Vector3::new(50.0, 40.0, 0.0), 4.0, 4.0);
    img.u_vector = Vector3::new(10.0, 0.0, 0.0);
    img.v_vector = Vector3::new(0.0, 10.0, 0.0);
    img.definition_handle = Some(handle);
    doc.add_entity_to_layout(EntityType::RasterImage(img), "A3")
        .unwrap();
    doc.add_entity_to_layout(line(10.0, 10.0, 100.0, 10.0, 1, "Continuous"), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    let mut p = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"]);
    if with_dir {
        p = p.images(&dir);
    }
    p.convert_with::<AcadrustProvider>().unwrap();
    dest
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_placed_image_is_an_image_on_the_page() {
    let pdf = image_plot("image", true);
    let o = ops(&pdf, 1);
    assert_eq!(count(&o, "Do"), 1, "one image drawn");
    // the cm before the Do places the unit square: 40 mm across and up from (50, 40) mm
    let at = o.iter().position(|o| o.operator == "Do").unwrap();
    let cm = o[..at].iter().rev().find(|o| o.operator == "cm").unwrap();
    let m: Vec<f64> = cm.operands.iter().map(real).collect();
    let mm = |v: f64| v / PT_PER_MM;
    assert!(
        (mm(m[0]) - 40.0).abs() < 0.1 && mm(m[1]).abs() < 0.1,
        "{m:?}"
    );
    assert!(
        mm(m[2]).abs() < 0.1 && (mm(m[3]) - 40.0).abs() < 0.1,
        "{m:?}"
    );
    assert!(
        (mm(m[4]) - 50.0).abs() < 0.1 && (mm(m[5]) - 40.0).abs() < 0.1,
        "{m:?}"
    );
    let doc = Document::load(&pdf).unwrap();
    let page = doc.get_pages()[&1];
    let (resources, _) = doc.get_page_resources(page).unwrap();
    let xobjects = resources
        .unwrap()
        .get(b"XObject")
        .unwrap()
        .as_dict()
        .unwrap();
    let (_, id) = xobjects.iter().next().unwrap();
    let image = &doc
        .get_object(id.as_reference().unwrap())
        .unwrap()
        .as_stream()
        .unwrap()
        .dict;
    assert_eq!(image.get(b"Width").unwrap().as_i64().unwrap(), 4);
    assert_eq!(image.get(b"Height").unwrap().as_i64().unwrap(), 4);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_without_an_image_directory_no_image_is_drawn() {
    let pdf = image_plot("noimage", false);
    assert_eq!(count(&ops(&pdf, 1), "Do"), 0);
}

// ---- fills behind a viewport ------------------------------------------------------------

/// Sheet `A3` with a 200 x 150 mm viewport centred at (210, 148) showing model space at 1 : 1 around the
/// origin, and in model space a SOLID of 30 mm sides at (10, 10): wholly inside the window.
fn viewport_fill_plot(tag: &str) -> PathBuf {
    let dir = scratch(tag);
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let mut window = Viewport::with_size(Vector3::new(210.0, 148.0, 0.0), 200.0, 150.0);
    window.id = 2;
    window.view_height = 150.0;
    window.view_center = Vector3::new(0.0, 0.0, 0.0);
    doc.add_entity_to_layout(EntityType::Viewport(Box::new(window)), "A3")
        .unwrap();
    let solid = Solid::new(
        Vector3::new(10.0, 10.0, 0.0),
        Vector3::new(40.0, 10.0, 0.0),
        Vector3::new(10.0, 40.0, 0.0),
        Vector3::new(40.0, 40.0, 0.0),
    );
    doc.add_entity(EntityType::Solid(solid)).unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .fills(true)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    dest
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_fill_behind_a_viewport_is_drawn_where_the_viewport_puts_it() {
    let pdf = viewport_fill_plot("vpfill");
    let o = ops(&pdf, 1);
    assert_eq!(
        count(&o, "f*"),
        1,
        "the SOLID is filled through the viewport"
    );
    // its corners are the viewport's: model (10, 10) is paper (210 + 10, 148 + 10) mm
    let at = o.iter().position(|o| o.operator == "f*").unwrap();
    let first_move = o[..at]
        .iter()
        .rev()
        .take_while(|o| o.operator != "rg")
        .last()
        .unwrap();
    let _ = first_move;
    let xs: Vec<f64> = o[..at]
        .iter()
        .filter(|o| o.operator == "m" || o.operator == "l")
        .map(|o| real(&o.operands[0]) / PT_PER_MM)
        .collect();
    assert!(xs.iter().any(|x| (x - 220.0).abs() < 0.5), "{xs:?}");
    assert!(xs.iter().any(|x| (x - 250.0).abs() < 0.5), "{xs:?}");
}

// ---- the lineweight a CTB style names (#1692) -------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_a_ctb_lineweight_is_one_more_than_the_index_of_the_list() {
    // The editor's list is "use the object's", then the table from its first entry, so style 1 is the
    // table's 0.0 mm (the thinnest line) and style 2 its 0.5 mm. ACI 1 is style 1, ACI 5 is style 2.
    let mut t = PlotStyleTable::identity("list.ctb");
    t.lineweights = vec![0.0, 0.5];
    t.aci_entries[1].lineweight = 1;
    t.aci_entries[5].lineweight = 2;
    let (out, pdf) = plot("list", |p| p.plot_style_table(t));
    out.unwrap();
    let s = strokes(&ops(&pdf, 1));
    let red = s
        .iter()
        .find(|(c, _, _)| *c == [255, 0, 0] || *c == [0, 0, 0] && false);
    let _ = red;
    // ACI 1 keeps its colour here (no colour set), ACI 5 is blue
    assert!(
        s.iter().any(|(c, w, _)| *c == [255, 0, 0] && *w == 0.0),
        "ACI 1 is not at 0 w in {s:?}"
    );
    assert!(
        s.iter()
            .any(|(c, w, _)| *c == [0, 0, 255] && (w - 0.5 * PT_PER_MM).abs() < 0.01),
        "ACI 5 is not at 0.5 mm in {s:?}"
    );
}

// ---- a fill's transparency --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_transparent_fill_is_thinned_toward_the_paper() {
    // A red SOLID that is 60 percent transparent is drawn at 40 percent red on white: (255, 153, 153).
    let dir = scratch("alpha");
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let mut solid = Solid::new(
        Vector3::new(300.0, 50.0, 0.0),
        Vector3::new(330.0, 50.0, 0.0),
        Vector3::new(300.0, 80.0, 0.0),
        Vector3::new(330.0, 80.0, 0.0),
    );
    solid.common.color = Color::Index(1);
    solid.common.transparency = Transparency::Explicit(153);
    doc.add_entity_to_layout(EntityType::Solid(solid), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .plot_style_table(PlotStyleTable::identity("own.ctb"))
        .fills(true)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let o = ops(&dest, 1);
    let at = o.iter().position(|o| o.operator == "f*").expect("a fill");
    let rg = o[..at].iter().rev().find(|o| o.operator == "rg").unwrap();
    let c: Vec<u8> = rg
        .operands
        .iter()
        .map(|v| (real(v) * 255.0).round() as u8)
        .collect();
    assert_eq!(c, [255, 153, 153], "{rg:?}");
}

// ---- a fill that crosses the viewport's edge --------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_fill_crossing_the_viewport_edge_is_cut_to_it() {
    // The viewport spans paper x 110 to 310 mm. A SOLID at model x 80 to 130 is at 290 to 340 on the
    // sheet, so its fill is cut at 310.
    let dir = scratch("vpcut");
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let mut window = Viewport::with_size(Vector3::new(210.0, 148.0, 0.0), 200.0, 150.0);
    window.id = 2;
    window.view_height = 150.0;
    window.view_center = Vector3::new(0.0, 0.0, 0.0);
    doc.add_entity_to_layout(EntityType::Viewport(Box::new(window)), "A3")
        .unwrap();
    let solid = Solid::new(
        Vector3::new(80.0, 10.0, 0.0),
        Vector3::new(130.0, 10.0, 0.0),
        Vector3::new(80.0, 40.0, 0.0),
        Vector3::new(130.0, 40.0, 0.0),
    );
    doc.add_entity(EntityType::Solid(solid)).unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .fills(true)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let o = ops(&dest, 1);
    let at = o
        .iter()
        .position(|o| o.operator == "f*")
        .expect("the part inside is filled");
    let xs: Vec<f64> = o[..at]
        .iter()
        .filter(|o| o.operator == "m" || o.operator == "l")
        .map(|o| real(&o.operands[0]) / PT_PER_MM)
        .collect();
    let (lo, hi) = xs
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    assert!(
        (lo - 290.0).abs() < 0.5 && (hi - 310.0).abs() < 0.5,
        "{xs:?}"
    );
}

// ---- a solid fill's outline ---------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_solid_fill_is_outlined_at_the_thinnest_line_in_its_own_colour() {
    // A plotter draws a solid hatch or SOLID filled and stroked, so its edge is the same colour as the
    // fill and as thin as a line gets: a closed path stroked at 0 w right after the fill.
    let (out, pdf) = plot("outline", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    let at = o.iter().position(|o| o.operator == "f*").expect("a fill");
    let fill = o[..at].iter().rev().find(|o| o.operator == "rg").unwrap();
    let after = &o[at..];
    let stroke = after
        .iter()
        .position(|o| o.operator == "S")
        .expect("an outline");
    let outline = &after[..stroke];
    assert!(
        outline
            .iter()
            .any(|o| o.operator == "RG" && o.operands == fill.operands),
        "the outline is not in the fill's colour: {outline:?}"
    );
    assert!(
        outline
            .iter()
            .any(|o| o.operator == "w" && real(&o.operands[0]) == 0.0),
        "the outline is not at 0 w: {outline:?}"
    );
    assert!(
        outline.iter().any(|o| o.operator == "h"),
        "the outline is not closed"
    );
}

// ---- the size of text -------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_text_is_sized_so_a_capital_stands_one_text_height() {
    // A text height in a drawing is the height of a capital, and Helvetica's capitals stand 0.716 em
    // high, so a 5 mm text is set in a 5 / 0.716 mm face.
    let (out, pdf) = plot("textsize", |p| p);
    out.unwrap();
    let o = ops(&pdf, 1);
    let tf = o.iter().find(|o| o.operator == "Tf").expect("text is set");
    let size = real(&tf.operands[1]) / PT_PER_MM;
    assert!((size - 5.0 / 0.716).abs() < 0.05, "{size} mm");
}

// ---- a line that is not flat --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_sloped_line_is_drawn_as_it_looks_from_above() {
    // A plan shows a line whose ends are at different heights as the segment it covers.
    let dir = scratch("sloped");
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let l = Line::from_coords(10.0, 10.0, 0.0, 100.0, 50.0, 30.0);
    doc.add_entity_to_layout(EntityType::Line(l), "A3").unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let o = ops(&dest, 1);
    let at = o.iter().position(|o| o.operator == "S").expect("a stroke");
    let pts: Vec<(f64, f64)> = o[..at]
        .iter()
        .filter(|o| o.operator == "m" || o.operator == "l")
        .map(|o| {
            (
                real(&o.operands[0]) / PT_PER_MM,
                real(&o.operands[1]) / PT_PER_MM,
            )
        })
        .collect();
    assert_eq!(pts.len(), 2, "{pts:?}");
    assert!(
        (pts[0].0 - 10.0).abs() < 0.05 && (pts[0].1 - 10.0).abs() < 0.05,
        "{pts:?}"
    );
    assert!(
        (pts[1].0 - 100.0).abs() < 0.05 && (pts[1].1 - 50.0).abs() < 0.05,
        "{pts:?}"
    );
}

// ---- a fill takes the plot style table's colour -------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_fill_takes_the_plot_style_tables_colour_for_its_aci() {
    // The table plots ACI 1 black: a SOLID in ACI 1 is filled black, as its lines would be.
    let dir = scratch("fillctb");
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    set_plot(&mut doc, &Sheet("A3", 420.0, 297.0, 0));
    let mut solid = Solid::new(
        Vector3::new(300.0, 50.0, 0.0),
        Vector3::new(330.0, 50.0, 0.0),
        Vector3::new(300.0, 80.0, 0.0),
        Vector3::new(330.0, 80.0, 0.0),
    );
    solid.common.color = Color::Index(1);
    doc.add_entity_to_layout(EntityType::Solid(solid), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .plot_style_table(ctb())
        .fills(true)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let o = ops(&dest, 1);
    let at = o.iter().position(|o| o.operator == "f*").expect("a fill");
    let rg = o[..at].iter().rev().find(|o| o.operator == "rg").unwrap();
    let c: Vec<u8> = rg
        .operands
        .iter()
        .map(|v| (real(v) * 255.0).round() as u8)
        .collect();
    assert_eq!(c, [0, 0, 0], "{rg:?}");
}

// ---- strokes below a width are hairlines --------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_strokes_thinner_than_the_hairline_limit_are_drawn_at_0_w() {
    // A plotter's PDF writes every line under some width as the thinnest line; the owner's has only 0 w
    // and 5 w though its table has 0.10 mm for ACI 7. With a limit of 0.2 mm the 0.10 mm line is 0 w and
    // the 0.35 mm one keeps its width.
    let mut t = PlotStyleTable::identity("limit.ctb");
    t.lineweights = vec![0.10, 0.35];
    t.aci_entries[1].lineweight = 1;
    t.aci_entries[5].lineweight = 2;
    let (out, pdf) = plot("hairline", |p| p.plot_style_table(t).hairline_below(0.2));
    out.unwrap();
    let s = strokes(&ops(&pdf, 1));
    assert!(
        s.iter().any(|(c, w, _)| *c == [255, 0, 0] && *w == 0.0),
        "{s:?}"
    );
    assert!(
        s.iter()
            .any(|(c, w, _)| *c == [0, 0, 255] && (w - 0.35 * PT_PER_MM).abs() < 0.01),
        "{s:?}"
    );
}
