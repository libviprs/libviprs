//! Dash patterns of a layout plotted at a scale other than 1:1 are paper millimetres too (#1683).
//!
//! #1665 draws a sheet's linework at its plot scale, so a drawing unit of a 1 : 0.04 layout is 25
//! mm of paper. The line-style stream's dash lengths were left in drawing units, so a dash and
//! gap of 0.5 and 0.25 units printed 0.75 mm of paper where the plot has 18.75 mm. The pattern
//! period must be what the plot shows, with the linetype scale (entity and global) still applied
//! the way the drawing defines it. A layout at 1:1 must not change. The drawings are written with
//! acadrust on the paper layout `Sheet`. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::fs;

use acadrust::entities::Line;
use acadrust::objects::ObjectType;
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PvecConversion;
use plan::scratch;
use pvec_facade::components::LineStyleStream;

/// One DASHED line (0.5 on, 0.25 off, 0.75 a period), one DASHDOT line (0.5, 0.25, a dot, 0.25:
/// 1.0 a period) and one DASHED line with a linetype scale of 2, on a Tabloid sheet in
/// millimetres whose plot scale is `n` paper millimetres to `d` units.
fn drawing(n: f64, d: f64) -> Vec<u8> {
    let mut doc = CadDocument::new();
    let mut dashed = LineType::new("DASHED");
    dashed.elements = vec![LineTypeElement::dash(0.5), LineTypeElement::space(0.25)];
    doc.line_types.add(dashed).unwrap();
    let mut dash_dot = LineType::new("DASHDOT");
    dash_dot.elements = vec![
        LineTypeElement::dash(0.5),
        LineTypeElement::space(0.25),
        LineTypeElement::dash(0.0),
        LineTypeElement::space(0.25),
    ];
    doc.line_types.add(dash_dot).unwrap();
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
    for (i, (linetype, scale)) in [("DASHED", 1.0), ("DASHDOT", 1.0), ("DASHED", 2.0)]
        .into_iter()
        .enumerate()
    {
        let y = i as f64 + 1.0;
        let mut line = Line::from_coords(1.0, y, 0.0, 9.0, y, 0.0);
        line.common.linetype = linetype.into();
        line.common.linetype_scale = scale;
        doc.add_entity_to_layout(EntityType::Line(line), "Sheet")
            .unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

/// The pattern period of each of the three lines in paper millimetres, in drawing order.
fn periods_mm(tag: &str, n: f64, d: f64) -> Vec<f64> {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let bytes = drawing(n, d);
    PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .components(StreamSelection::ALL)
        .convert()
        .expect("the sheet converts");
    let r = open_published(&dest).unwrap();
    let mm_per_unit = r.manifest().metres_per_unit * 1000.0;
    let streams = r.streams().unwrap().0;
    let stream = streams
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
        .expect("the line style stream");
    let style = LineStyleStream::decode(&stream.payload).unwrap();
    let mut out: Vec<(u32, f64)> = style
        .assignments
        .iter()
        .map(|(id, k)| {
            let period: f64 = style.styles[*k as usize].dashes.iter().sum();
            (id.index(), period * mm_per_unit)
        })
        .collect();
    out.sort_by_key(|p| p.0);
    let _ = fs::remove_dir_all(dir);
    out.into_iter().map(|p| p.1).collect()
}

fn assert_periods(got: &[f64], want: [f64; 3]) {
    assert_eq!(got.len(), 3, "one style per line, got {got:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() < 1e-6,
            "periods {got:?} mm, expected {want:?} mm"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1683_dash_and_dash_dot_periods_are_paper_millimetres_at_1_to_0_04() {
    // 25 mm a unit: 0.75 units is 18.75 mm, 1.0 is 25 mm, and a linetype scale of 2 doubles it.
    assert_periods(&periods_mm("scaled", 1.0, 0.04), [18.75, 25.0, 37.5]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1683_a_layout_at_1_to_1_keeps_its_patterns_as_drawn() {
    assert_periods(&periods_mm("unit", 1.0, 1.0), [0.75, 1.0, 1.5]);
}
