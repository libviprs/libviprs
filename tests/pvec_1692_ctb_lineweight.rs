//! Issue #1692: the lineweight a CTB style names, as a conversion applies it. In the plot style editor
//! the list reads "Use object lineweight" first and then the table from its first entry, so a style's
//! `lineweight` is one more than the table index it picks: 0 is the object's, 1 the table's first
//! entry, 2 its second. `PlotStyleTable::resolve_lineweight` reads the entry at the style's number
//! itself (acadrust#73), so the conversion reads the list itself. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::fs;

use acadrust::entities::Line;
use acadrust::plot_style::PlotStyleTable;
use acadrust::types::{Color, LineWeight};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PvecConversion;
use plan::scratch;
use pvec_facade::components::{LineStyleStream, StrokeWidth};

/// A table whose list is 0.10, 0.20, 0.30, 0.40 mm and whose styles for ACI 1 to 4 pick the numbers 1
/// to 4. The scale factor is off.
fn table() -> PlotStyleTable {
    let mut t = PlotStyleTable::identity("synthetic.ctb");
    t.lineweights = vec![0.10, 0.20, 0.30, 0.40];
    t.apply_factor = false;
    for aci in 1..=4u8 {
        t.aci_entries[usize::from(aci)].lineweight = aci;
    }
    t
}

/// The width in millimetres each of ACI 1 to 6 is plotted at, for lines the object draws at 0.25 mm,
/// or `None` for a thinnest line. ACI 5 and 6 are the styles the table leaves at 0 and past its end.
fn widths(tag: &str, mut t: PlotStyleTable) -> Vec<Option<f64>> {
    t.aci_entries[5].lineweight = 0;
    t.aci_entries[6].lineweight = 9;
    let mut doc = CadDocument::new();
    doc.add_layout("Sheet").unwrap();
    for aci in 1..=6u8 {
        let mut l = Line::from_coords(f64::from(aci), 0.0, 0.0, f64::from(aci), 1.0, 0.0);
        l.common.color = Color::Index(aci);
        l.common.line_weight = LineWeight::Value(25);
        doc.add_entity_to_layout(EntityType::Line(l), "Sheet")
            .unwrap();
    }
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .components(StreamSelection::ALL)
        .plot_style_table(t)
        .convert()
        .unwrap();
    let reader = open_published(&dest).unwrap();
    let streams = reader.streams().unwrap().0;
    let s = streams
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
        .unwrap();
    let style = LineStyleStream::decode(&s.payload).unwrap();
    let mm = reader.manifest().metres_per_unit * 1000.0;
    let mut out: Vec<Option<f64>> = vec![None; 6];
    for (id, n) in &style.assignments {
        // the entities are written in the order they were added, so the id is the ACI less one
        let w = match style.styles[*n as usize].width {
            Some(StrokeWidth::Width(w)) => Some(w * mm),
            _ => None,
        };
        out[id.index() as usize] = w;
    }
    let _ = fs::remove_dir_all(dir);
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_style_one_is_the_first_entry_of_the_list() {
    let w = widths("one", table());
    assert!((w[0].unwrap() - 0.10).abs() < 1e-6, "{w:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_style_two_is_the_second_entry() {
    let w = widths("two", table());
    assert!((w[1].unwrap() - 0.20).abs() < 1e-6, "{w:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_the_last_entry_is_reachable_by_its_number() {
    let w = widths("last", table());
    assert!((w[3].unwrap() - 0.40).abs() < 1e-6, "{w:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_a_first_entry_of_zero_is_the_thinnest_line() {
    let mut t = table();
    t.lineweights = vec![0.0, 0.5];
    t.aci_entries[1].lineweight = 1;
    let w = widths("zero", t);
    assert_eq!(w[0], None, "{w:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1692_zero_and_a_number_past_the_list_keep_the_objects_lineweight() {
    let w = widths("object", table());
    assert!((w[4].unwrap() - 0.25).abs() < 1e-6, "style 0: {w:?}");
    assert!(
        (w[5].unwrap() - 0.25).abs() < 1e-6,
        "style past the end: {w:?}"
    );
}
