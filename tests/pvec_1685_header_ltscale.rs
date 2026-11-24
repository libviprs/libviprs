//! The header's LTSCALE and PSLTSCALE reach the dash patterns (#1685).
//!
//! #1683 made the sheet's own dashes paper millimetres but never read the header: LTSCALE, the
//! global linetype scale, was applied to nothing, and a model-space line seen through a viewport
//! drew with its pattern in bare sheet units. With PSLTSCALE 1 AutoCAD plots a viewport's
//! linetypes like the sheet's own (pattern x LTSCALE x the entity's scale, in paper units), and
//! with PSLTSCALE 0 the pattern lives in model units and the viewport scales it. A layout at 1:1
//! with LTSCALE 1 must not change. The drawings are written with acadrust on the paper layout
//! `Sheet`: a HIDDEN line (pattern 0.375) on the sheet, a CENTER2 line (1.125) and a DASHDOT2 line
//! (0.5) in model space seen through one window at a viewport scale of 1:2. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::collections::BTreeMap;
use std::fs;

use acadrust::entities::{Line, Viewport};
use acadrust::objects::ObjectType;
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PvecConversion;
use plan::scratch;
use pvec_facade::components::LineStyleStream;

fn linetype(doc: &mut CadDocument, name: &str, elements: [f64; 4]) {
    let mut lt = LineType::new(name);
    lt.elements = vec![
        LineTypeElement::dash(elements[0]),
        LineTypeElement::space(elements[1]),
        LineTypeElement::dash(elements[2]),
        LineTypeElement::space(elements[3]),
    ];
    doc.line_types.add(lt).unwrap();
}

fn line(doc: &mut CadDocument, layout: Option<&str>, linetype: &str, a: [f64; 2], b: [f64; 2]) {
    let mut l = Line::from_coords(a[0], a[1], 0.0, b[0], b[1], 0.0);
    l.common.linetype = linetype.into();
    let e = EntityType::Line(l);
    match layout {
        Some(name) => doc.add_entity_to_layout(e, name).unwrap(),
        None => doc.add_entity(e).unwrap(),
    };
}

/// The sheet is Tabloid in millimetres at `n` paper millimetres to `d` units. HIDDEN (0.25 on,
/// 0.125 off) is on the sheet, CENTER2 (0.75, 0.125, 0.125, 0.125, 1.125 a period) and DASHDOT2
/// (0.25, 0.125, a dot, 0.125, 0.5 a period) are in model space. The one window is 8 by 10 sheet
/// units and shows 20 model units of height, a viewport scale of 1:2.
fn drawing(n: f64, d: f64, ltscale: f64, psltscale: bool) -> Vec<u8> {
    let mut doc = CadDocument::new();
    doc.header.linetype_scale = ltscale;
    doc.header.paper_space_linetype_scaling = psltscale;
    linetype(&mut doc, "HIDDEN", [0.25, 0.125, 0.0, 0.0]);
    linetype(&mut doc, "CENTER2", [0.75, 0.125, 0.125, 0.125]);
    linetype(&mut doc, "DASHDOT2", [0.25, 0.125, 0.0, 0.125]);
    let layout = doc.add_layout("Sheet").unwrap();
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ANSI_B_(11.00_x_17.00_Inches)".into();
    l.paper_width = 279.4;
    l.paper_height = 431.8;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = n;
    l.plot_scale_denominator = d;
    l.plot_rotation = 0;
    l.plot_margin_left = 0.0;
    l.plot_margin_bottom = 0.0;
    l.plot_origin_x = 0.0;
    l.plot_origin_y = 0.0;
    l.plot_flags.print_lineweights = true;
    line(&mut doc, Some("Sheet"), "HIDDEN", [1.0, 1.0], [2.0, 1.0]);
    let mut window = Viewport::with_size(Vector3::new(5.0, 8.0, 0.0), 8.0, 10.0);
    window.id = 2;
    window.view_height = 20.0;
    window.view_center = Vector3::new(0.0, 0.0, 0.0);
    doc.add_entity_to_layout(EntityType::Viewport(Box::new(window)), "Sheet")
        .unwrap();
    line(&mut doc, None, "CENTER2", [-3.0, 0.0], [3.0, 0.0]);
    line(&mut doc, None, "DASHDOT2", [-3.0, 1.0], [3.0, 1.0]);
    DwgWriter::write_to_vec(&doc).unwrap()
}

/// The pattern period of each linetype drawn, in millimetres of the published sheet.
fn periods_mm(tag: &str, profile: Option<ViewProfile>, bytes: &[u8]) -> BTreeMap<String, f64> {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let mut c = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(StreamSelection::ALL);
    if let Some(p) = profile {
        c = c.view_profile(p);
    }
    c.convert().expect("the drawing converts");
    let r = open_published(&dest).unwrap();
    let mm_per_unit = r.manifest().metres_per_unit * 1000.0;
    let streams = r.streams().unwrap().0;
    let stream = streams
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
        .expect("the line style stream");
    let style = LineStyleStream::decode(&stream.payload).unwrap();
    let out = style
        .assignments
        .iter()
        .map(|(_, k)| {
            let s = &style.styles[*k as usize];
            (s.name.clone(), s.dashes.iter().sum::<f64>() * mm_per_unit)
        })
        .collect();
    let _ = fs::remove_dir_all(dir);
    out
}

fn sheet(tag: &str, n: f64, d: f64, ltscale: f64, psltscale: bool) -> BTreeMap<String, f64> {
    periods_mm(
        tag,
        Some(ViewProfile::print_profile("Sheet")),
        &drawing(n, d, ltscale, psltscale),
    )
}

fn assert_periods(got: &BTreeMap<String, f64>, want: [(&str, f64); 3]) {
    for (name, mm) in want {
        let g = got.get(name).copied().unwrap_or(f64::NAN);
        assert!(
            (g - mm).abs() < 1e-6,
            "{name}: periods {got:?} mm, expected {mm} mm"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1685_the_sheets_own_dashes_carry_the_header_ltscale() {
    // 0.375 x 0.5 x 25 mm a unit.
    let got = sheet("own", 1.0, 0.04, 0.5, true);
    assert!((got["HIDDEN"] - 4.6875).abs() < 1e-6, "{got:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1685_with_psltscale_1_a_viewports_dashes_are_paper_millimetres_too() {
    // 1.125 x 0.5 x 25 = 14.0625 and 0.5 x 0.5 x 25 = 6.25, whatever the viewport scale is.
    let got = sheet("ps1", 1.0, 0.04, 0.5, true);
    assert_periods(
        &got,
        [("HIDDEN", 4.6875), ("CENTER2", 14.0625), ("DASHDOT2", 6.25)],
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1685_with_psltscale_0_a_viewports_dashes_are_scaled_by_the_viewport() {
    // Model units, so the window's 1:2 and the 25 mm a unit both apply: 12.5 mm a model unit.
    let got = sheet("ps0", 1.0, 0.04, 0.5, false);
    assert_periods(
        &got,
        [
            ("HIDDEN", 4.6875),
            ("CENTER2", 7.03125),
            ("DASHDOT2", 3.125),
        ],
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1685_a_layout_at_1_to_1_with_ltscale_1_keeps_its_patterns_as_drawn() {
    // A viewport scale doesn't enter with PSLTSCALE 1, and PSLTSCALE 0 only adds it.
    let got = sheet("unit", 1.0, 1.0, 1.0, true);
    assert_periods(
        &got,
        [("HIDDEN", 0.375), ("CENTER2", 1.125), ("DASHDOT2", 0.5)],
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1685_ltscale_applies_to_model_space_patterns_without_a_sheet() {
    let got = periods_mm("model", None, &drawing(1.0, 1.0, 2.0, true));
    assert!((got["CENTER2"] - 2.25).abs() < 1e-6, "{got:?}");
    assert!((got["DASHDOT2"] - 1.0).abs() < 1e-6, "{got:?}");
}
