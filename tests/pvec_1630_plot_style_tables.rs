//! Issue #1630 (C4.4 of iasbuilt/webapp #1434, spec section 6 P-8 and the Phase C4 addendum): a
//! print conversion takes a plot style table (`PvecConversion::plot_style_table`) and applies it
//! to every stroke it writes, per ACI for a CTB: colour (with the grayscale policy bit), screening,
//! lineweight through the table's lineweight list and `apply_factor`, and linetype. An entry whose
//! colour is the object colour keeps the entity's own colour, and so does an entity with a true
//! colour, which has no ACI to look up.
//!
//! Each honoured property gets its own table so a failure names the property. The real table is
//! `Langara_2024-LW0.ctb`: ACI 1 to 9, 209 and 251 plot black, ACI 7 at 0.13 mm, everything else
//! keeps its object colour and lineweight. The drawings are written with acadrust, one stroke per
//! case, on the paper layout `Sheet` so the print profile walks them. Widths are compared as
//! ratios, so the test does not depend on the unit the widths are written in. Needs
//! `--features pvec-convert`.
//!
//! Not covered here: STB named styles. The provider does not report an entity's plot style name
//! yet, so there is nothing to look a named style up by (said on the issue).

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use acadrust::entities::Line;
use acadrust::plot_style::PlotStyleTable;
use acadrust::tables::{Layer, LineType, LineTypeElement};
use acadrust::types::{Color, LineWeight};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput};
use plan::scratch;
use pvec_facade::components::{LineStyleStream, Stream, StrokePaintStream, StrokeWidth};

const CTB: &[u8] = include_bytes!("fixtures/cad/langara/Langara_2024-LW0.ctb");

/// The stroke each case draws: its own ACI, or a true colour, an object lineweight and linetype.
struct Case {
    aci: Option<u8>,
    true_colour: Option<[u8; 3]>,
    lineweight_hundredths: i16,
    dashed: bool,
}

fn case(aci: u8) -> Case {
    Case {
        aci: Some(aci),
        true_colour: None,
        lineweight_hundredths: 25,
        dashed: false,
    }
}

/// A drawing with one line per case in the paper layout `Sheet`, and each line's handle.
fn drawing(cases: &[Case]) -> (Vec<u8>, Vec<u64>) {
    let mut doc = CadDocument::new();
    let mut dashed = LineType::new("DASHED");
    dashed.elements = vec![LineTypeElement::dash(0.5), LineTypeElement::space(0.25)];
    doc.line_types.add(dashed).unwrap();
    doc.layers
        .add(Layer::with_color("PLAN", Color::Index(3)))
        .unwrap();
    doc.add_layout("Sheet").unwrap();
    let mut handles = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let x = i as f64 + 1.0;
        let mut l = Line::from_coords(x, 0.0, 0.0, x, 1.0, 0.0);
        l.common.layer = "PLAN".into();
        l.common.color = match (c.aci, c.true_colour) {
            (_, Some([r, g, b])) => Color::Rgb { r, g, b },
            (Some(aci), None) => Color::Index(aci),
            (None, None) => Color::ByLayer,
        };
        l.common.line_weight = LineWeight::Value(c.lineweight_hundredths);
        if c.dashed {
            l.common.linetype = "DASHED".into();
        }
        let h = doc
            .add_entity_to_layout(EntityType::Line(l), "Sheet")
            .unwrap();
        handles.push(h.value());
    }
    (DwgWriter::write_to_vec(&doc).unwrap(), handles)
}

struct Run {
    out: PvecConversionOutput,
    dest: PathBuf,
    dir: PathBuf,
}

fn convert(tag: &str, bytes: &[u8], table: Option<PlotStyleTable>) -> Run {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let mut c = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .components(StreamSelection::ALL);
    if let Some(t) = table {
        c = c.plot_style_table(t);
    }
    let out = c.convert().unwrap();
    Run { out, dest, dir }
}

fn streams(run: &Run) -> Vec<Stream> {
    open_published(&run.dest).unwrap().streams().unwrap().0
}

/// The PVec id of each written entity is its place among the report rows that wrote geometry.
fn kept(run: &Run) -> Vec<u64> {
    run.out
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .map(|r| r.source_handle.expect("a handle"))
        .collect()
}

/// The paint word of each source handle that has one.
fn words(run: &Run) -> BTreeMap<u64, u32> {
    let s = streams(run);
    let Some(p) = s
        .iter()
        .find(|s| s.capability == StrokePaintStream::CAPABILITY)
    else {
        return BTreeMap::new();
    };
    let paint = StrokePaintStream::decode(&p.payload).unwrap();
    let kept = kept(run);
    paint
        .assignments
        .iter()
        .map(|(id, n)| (kept[id.index() as usize], paint.colours[*n as usize]))
        .collect()
}

/// The stroke width and dash array of each source handle that has a style.
fn styles(run: &Run) -> BTreeMap<u64, (Option<StrokeWidth>, Vec<f64>)> {
    let s = streams(run);
    let Some(p) = s
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
    else {
        return BTreeMap::new();
    };
    let style = LineStyleStream::decode(&p.payload).unwrap();
    let kept = kept(run);
    style
        .assignments
        .iter()
        .map(|(id, n)| {
            let def = &style.styles[*n as usize];
            (kept[id.index() as usize], (def.width, def.dashes.clone()))
        })
        .collect()
}

fn width(w: &BTreeMap<u64, (Option<StrokeWidth>, Vec<f64>)>, h: u64) -> f64 {
    match w.get(&h).and_then(|s| s.0) {
        Some(StrokeWidth::Width(x)) => x,
        other => panic!("{h:#X} has a positive width, got {other:?}"),
    }
}

fn aci_word(aci: u8) -> u32 {
    let (r, g, b) = acadrust::types::Color::Index(aci).rgb().unwrap();
    u32::from_be_bytes([r, g, b, 255])
}

/// A CTB with every entry at the object's value, which the test then edits.
fn identity() -> PlotStyleTable {
    PlotStyleTable::identity("test.ctb")
}

fn rgb(word: u32) -> [f64; 3] {
    let [r, g, b, _] = word.to_be_bytes();
    [f64::from(r), f64::from(g), f64::from(b)]
}

/// How light a stroke looks on white paper: its colour composited over white by its alpha.
fn on_white(word: u32) -> [f64; 3] {
    let a = f64::from(word & 0xFF) / 255.0;
    rgb(word).map(|c| c * a + 255.0 * (1.0 - a))
}

fn lum(c: [f64; 3]) -> f64 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn done(run: Run) {
    let _ = fs::remove_dir_all(run.dir);
}

// ---- the real table ----------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn print_profile_applies_ctb_colour_screening_lineweight_linetype() {
    // The Langara table: ACI 1 to 9, 209 and 251 black, ACI 7 at 0.13 mm, the rest by object.
    let table = PlotStyleTable::from_bytes("Langara_2024-LW0.ctb", CTB).unwrap();
    let acis = [1u8, 2, 5, 7, 9, 30, 100, 209, 251];
    let cases: Vec<Case> = acis.iter().map(|a| case(*a)).collect();
    let (bytes, h) = drawing(&cases);
    let plain = convert("langara-plain", &bytes, None);
    let run = convert("langara", &bytes, Some(table));
    let w = words(&run);
    let wp = words(&plain);
    for (i, aci) in acis.iter().enumerate() {
        let black = [1u8, 2, 5, 7, 9, 209, 251].contains(aci);
        if black {
            assert_eq!(w[&h[i]], 0x0000_00FF, "ACI {aci} plots black");
        } else {
            assert_eq!(
                w[&h[i]],
                aci_word(*aci),
                "ACI {aci} keeps its object colour"
            );
        }
    }
    assert_ne!(
        w[&h[0]], wp[&h[0]],
        "without the table ACI 1 is red, so the table is what changed it"
    );
    let s = styles(&run);
    let object = width(&s, h[1]);
    assert!(
        (width(&s, h[3]) / object - 0.13 / 0.25).abs() < 1e-9,
        "ACI 7 is 0.13 mm where the object is 0.25 mm"
    );
    for i in [0, 1, 2, 4, 5, 6, 7, 8] {
        assert_eq!(
            width(&s, h[i]),
            object,
            "ACI {} keeps its lineweight",
            acis[i]
        );
    }
    done(plain);
    done(run);
}

// ---- one case per honoured property ---------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn a_ctb_colour_replaces_the_object_colour() {
    let mut t = identity();
    t.aci_entries[1].color = Some([0, 0, 255]);
    let (bytes, h) = drawing(&[case(1), case(2)]);
    let w = words(&convert("colour", &bytes, Some(t)));
    assert_eq!(w[&h[0]], 0x0000_FFFF, "ACI 1 plots the entry's blue");
    assert_eq!(
        w[&h[1]],
        aci_word(2),
        "an entry with the object colour leaves ACI 2"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_grayscale_policy_bit_plots_the_object_colour_in_gray() {
    let mut t = identity();
    t.aci_entries[3].color_policy |= 2;
    let (bytes, h) = drawing(&[case(3), case(5)]);
    let w = words(&convert("gray", &bytes, Some(t)));
    let [r, g, b] = rgb(w[&h[0]]);
    assert!(r == g && g == b, "gray has equal channels: {r} {g} {b}");
    let want = lum(rgb(aci_word(3)));
    assert!((r - want).abs() <= 1.0, "luminance of ACI 3: {r} vs {want}");
    assert_eq!(w[&h[1]], aci_word(5), "another ACI stays in colour");
}

#[test]
#[cfg_attr(miri, ignore)]
fn screening_lightens_the_stroke() {
    let mut t = identity();
    t.aci_entries[5].screening = 50;
    t.aci_entries[6].screening = 0;
    let (bytes, h) = drawing(&[case(5), case(6), case(5), case(1)]);
    // The two ACI 5 strokes are the same stroke but for the table: one entry each is not what is
    // asserted, the table's own entry is, so compare against an unscreened ACI 1.
    let w = words(&convert("screen", &bytes, Some(t)));
    let full = lum(on_white(aci_word(5)));
    let half = lum(on_white(w[&h[0]]));
    assert!(
        half > full + 1.0 && half < 255.0,
        "50 percent is lighter than {full} and not white: {half}"
    );
    assert!(
        (lum(on_white(w[&h[1]])) - 255.0).abs() < 1.0,
        "0 percent is white paper"
    );
    assert_eq!(
        lum(on_white(w[&h[3]])),
        lum(on_white(aci_word(1))),
        "100 percent is untouched"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_lineweight_list_sets_the_width() {
    let mut t = identity();
    // Index 13 of the standard list is 0.50 mm.
    assert!((t.lineweights[13] - 0.50).abs() < 1e-6);
    t.aci_entries[6].lineweight = 13;
    let (bytes, h) = drawing(&[case(6), case(8)]);
    let s = styles(&convert("lineweight", &bytes, Some(t)));
    assert!(
        (width(&s, h[0]) / width(&s, h[1]) - 0.50 / 0.25).abs() < 1e-9,
        "0.50 mm against the object's 0.25 mm"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn apply_factor_scales_a_table_lineweight_and_not_the_object_one() {
    let mut t = identity();
    t.aci_entries[6].lineweight = 13;
    t.apply_factor = true;
    t.scale_factor = 2.0;
    let (bytes, h) = drawing(&[case(6), case(8)]);
    let s = styles(&convert("factor", &bytes, Some(t)));
    assert!(
        (width(&s, h[0]) / width(&s, h[1]) - 1.00 / 0.25).abs() < 1e-9,
        "0.50 mm times 2 against the object's 0.25 mm"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_solid_entry_removes_an_object_linetype_and_a_dashed_entry_adds_one() {
    let mut t = identity();
    // The plot style linetype numbers: 0 is solid, 1 is dashed, 31 is "use object linetype".
    t.aci_entries[2].linetype = 0;
    t.aci_entries[4].linetype = 1;
    let dashed = |aci: u8| Case {
        dashed: true,
        ..case(aci)
    };
    let (bytes, h) = drawing(&[dashed(2), case(4), dashed(8), case(9)]);
    let s = styles(&convert("linetype", &bytes, Some(t)));
    let dashes = |i: usize| s.get(&h[i]).map(|x| x.1.clone()).unwrap_or_default();
    assert!(dashes(0).is_empty(), "solid beats the object's DASHED");
    assert!(
        !dashes(1).is_empty(),
        "dashed beats the object's continuous"
    );
    assert!(!dashes(2).is_empty(), "an object entry keeps DASHED");
    assert!(dashes(3).is_empty(), "an object entry keeps continuous");
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_object_colour_entry_and_a_true_colour_keep_the_entitys_own_colour() {
    // ACI 1 is forced black, so a true-colour stroke, which has no ACI, must not be.
    let mut t = identity();
    t.aci_entries[1].color = Some([0, 0, 0]);
    let truecolour = Case {
        aci: None,
        true_colour: Some([12, 34, 56]),
        ..case(1)
    };
    let (bytes, h) = drawing(&[case(1), truecolour, case(30)]);
    let w = words(&convert("object", &bytes, Some(t)));
    assert_eq!(w[&h[0]], 0x0000_00FF);
    assert_eq!(w[&h[1]], 0x0C22_38FF, "a true colour is its own");
    assert_eq!(
        w[&h[2]],
        aci_word(30),
        "an object-colour entry keeps ACI 30"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_recorded_properties_do_not_change_what_is_drawn() {
    // Dither, pens, adaptive, end, join and fill are recorded only (C4.1 keeps them): a table that
    // differs from the identity in those alone converts to the same bytes as the identity.
    let mut t = identity();
    for e in t.aci_entries.iter_mut().skip(1) {
        e.physical_pen_number = 7;
        e.virtual_pen_number = 9;
        e.adaptive_linetype = false;
        e.end_style = 1;
        e.join_style = 2;
        e.fill_style = 70;
        e.linepattern_size = 2.0;
    }
    let (bytes, _) = drawing(&[case(1), case(5), case(7)]);
    let plain = convert("rec-plain", &bytes, Some(identity()));
    let recorded = convert("rec", &bytes, Some(t));
    assert_eq!(words(&plain), words(&recorded));
    assert_eq!(styles(&plain), styles(&recorded));
    done(plain);
    done(recorded);
}

#[test]
#[cfg_attr(miri, ignore)]
fn without_a_table_the_conversion_is_what_it_was() {
    let (bytes, h) = drawing(&[case(1), case(7)]);
    let a = convert("none-a", &bytes, None);
    let b = convert("none-b", &bytes, None);
    assert_eq!(words(&a), words(&b));
    assert_eq!(words(&a)[&h[0]], aci_word(1), "ACI 1 is red with no table");
    done(a);
    done(b);
}

#[test]
#[cfg_attr(miri, ignore)]
fn screening_is_written_opaque_and_blended_toward_white() {
    // A plotter prints a screened colour as a lighter ink, not a see-through one, so the word is
    // opaque and the colour is the blend `255 - (255 - c) * s`.
    let mut t = identity();
    t.aci_entries[5].screening = 50;
    let (bytes, h) = drawing(&[case(5)]);
    let w = words(&convert("opaque", &bytes, Some(t)));
    assert_eq!(w[&h[0]] & 0xFF, 0xFF, "no alpha");
    let blended = rgb(aci_word(5)).map(|c| (255.0 - (255.0 - c) * 0.5).round());
    assert_eq!(rgb(w[&h[0]]), blended);
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_stb_applies_nothing_and_the_report_says_so() {
    // The provider reports no plot style name per entity yet, so a named style table has nothing
    // to look up: it changes no stroke and the sheet notes the gap (said on #1630).
    let mut t = identity();
    t.is_stb = true;
    t.aci_entries[1].color = Some([0, 0, 255]);
    let (bytes, h) = drawing(&[case(1)]);
    let run = convert("stb", &bytes, Some(t));
    assert_eq!(words(&run)[&h[0]], aci_word(1), "ACI 1 stays red");
    let notes = &run.out.report.sheet.as_ref().expect("a sheet").notes;
    assert!(
        notes.iter().any(|n| n == "plot-style-stb-unsupported"),
        "{notes:?}"
    );
    let plain = convert("stb-plain", &bytes, None);
    let notes = &plain.out.report.sheet.as_ref().expect("a sheet").notes;
    assert!(!notes.iter().any(|n| n == "plot-style-stb-unsupported"));
    done(plain);
    done(run);
}
