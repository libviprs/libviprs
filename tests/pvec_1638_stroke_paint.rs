//! Issue #1638: a DWG conversion writes the stroke paint stream (`0xRRGGBBAA`, one entry per
//! stroke whose colour the drawing settles), the way the PDF path does since #1578. The colour is
//! the effective one: its own ACI or true colour, BYLAYER on the effective layer (an entity on
//! layer 0 inside a block takes the INSERT's layer), BYBLOCK on the nearest INSERT that does not say
//! BYBLOCK itself, and ACI 7 at the top, written black the way a plot prints it. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use acadrust::entities::{Insert, Line};
use acadrust::tables::{BlockRecord, Layer};
use acadrust::types::{Color, Handle, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput};
use plan::scratch;
use pvec_facade::components::{Stream, StrokePaintStream};

const BLACK: u32 = 0x0000_00FF;

fn word(c: Color) -> u32 {
    let (r, g, b) = c.rgb().expect("a colour with a value");
    u32::from_be_bytes([r, g, b, 255])
}

/// A drawing under construction: handles of what it holds are kept by name.
struct Drawing {
    doc: CadDocument,
    owners: BTreeMap<&'static str, Handle>,
}

impl Drawing {
    fn new(layers: &[(&str, Color)]) -> Self {
        let mut doc = CadDocument::new();
        for (name, colour) in layers {
            doc.layers.add(Layer::with_color(*name, *colour)).unwrap();
        }
        Drawing {
            doc,
            owners: BTreeMap::new(),
        }
    }

    fn block(&mut self, name: &'static str) {
        let mut r = BlockRecord::new(name);
        r.handle = self.doc.allocate_handle();
        self.owners.insert(name, r.handle);
        self.doc.block_records.add(r).unwrap();
    }

    /// A line at `x`, in `block` (or in model space), and its handle.
    fn line(&mut self, block: Option<&'static str>, x: f64, layer: &str, colour: Color) -> u64 {
        let mut e = EntityType::Line(Line::from_coords(x, 0.0, 0.0, x, 1.0, 0.0));
        e.common_mut().layer = layer.into();
        e.common_mut().color = colour;
        if let Some(b) = block {
            e.common_mut().owner_handle = self.owners[b];
        }
        self.doc.add_entity(e).unwrap().value()
    }

    fn insert(
        &mut self,
        into: Option<&'static str>,
        name: &str,
        layer: &str,
        colour: Color,
    ) -> u64 {
        let mut e = EntityType::Insert(Insert::new(name, Vector3::new(0.0, 0.0, 0.0)));
        e.common_mut().layer = layer.into();
        e.common_mut().color = colour;
        if let Some(b) = into {
            e.common_mut().owner_handle = self.owners[b];
        }
        self.doc.add_entity(e).unwrap().value()
    }

    fn bytes(&self) -> Vec<u8> {
        DwgWriter::write_to_vec(&self.doc).unwrap()
    }
}

struct Run {
    out: PvecConversionOutput,
    dest: PathBuf,
    dir: PathBuf,
}

fn convert(tag: &str, bytes: &[u8], select: StreamSelection) -> Run {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(select)
        .convert()
        .unwrap();
    Run { out, dest, dir }
}

fn streams_of(dest: &Path) -> Vec<Stream> {
    let (streams, skipped) = open_published(dest).unwrap().streams().unwrap();
    assert!(skipped.is_empty(), "{skipped:?}");
    streams
}

fn paint_of(streams: &[Stream]) -> Option<StrokePaintStream> {
    streams
        .iter()
        .find(|s| s.capability == StrokePaintStream::CAPABILITY)
        .map(|s| StrokePaintStream::decode(&s.payload).unwrap())
}

/// The colour word of each source handle that has one, read back through PVec's reader.
fn words(run: &Run) -> BTreeMap<u64, u32> {
    let paint = paint_of(&streams_of(&run.dest)).expect("a stroke paint stream");
    // The PVec id of an entity is its place among the rows that wrote geometry.
    let kept: Vec<u64> = run
        .out
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .map(|r| r.source_handle.expect("a handle"))
        .collect();
    paint
        .assignments
        .iter()
        .map(|(id, n)| (kept[id.index() as usize], paint.colours[*n as usize]))
        .collect()
}

const ALL_BUT_PAINT: StreamSelection = StreamSelection {
    stroke_paint: false,
    ..StreamSelection::ALL
};

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_strokes_on_layers_of_different_colours_get_different_entries() {
    let mut d = Drawing::new(&[
        ("L-RED", Color::Index(1)),
        ("L-BLUE", Color::Rgb { r: 0, g: 0, b: 255 }),
    ]);
    let a = d.line(None, 1.0, "L-RED", Color::ByLayer);
    let b = d.line(None, 2.0, "L-BLUE", Color::ByLayer);
    let c = d.line(None, 3.0, "L-RED", Color::ByLayer);
    let run = convert("layers", &d.bytes(), StreamSelection::ALL);
    let paint = paint_of(&streams_of(&run.dest)).expect("a stroke paint stream");
    // One table entry per distinct word, in the order the strokes were first seen.
    assert_eq!(paint.colours.len(), 2, "{paint:?}");
    assert_eq!(paint.assignments.len(), 3);
    let w = words(&run);
    assert_eq!(w[&a], word(Color::Index(1)));
    assert_eq!(w[&b], 0x0000_FFFF);
    assert_eq!(w[&c], w[&a]);
    assert_ne!(w[&a], w[&b]);
    let _ = fs::remove_dir_all(run.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_bylayer_inside_an_insert_resolves_on_the_effective_layer() {
    // A line on layer 0 inside a block is drawn on the layer of the INSERT that places it, so
    // BYLAYER there is the INSERT layer's colour, not layer 0's.
    let mut d = Drawing::new(&[("WALLS", Color::Index(3)), ("DOORS", Color::Index(5))]);
    d.block("B1");
    let inside = d.line(Some("B1"), 1.0, "0", Color::ByLayer);
    let own = d.line(Some("B1"), 2.0, "DOORS", Color::ByLayer);
    d.insert(None, "B1", "WALLS", Color::ByLayer);
    let run = convert("effective", &d.bytes(), StreamSelection::ALL);
    let w = words(&run);
    assert_eq!(w[&inside], word(Color::Index(3)), "the INSERT's layer");
    assert_eq!(
        w[&own],
        word(Color::Index(5)),
        "a layer of its own is its own"
    );
    let _ = fs::remove_dir_all(run.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_byblock_resolves_through_nested_inserts_and_is_aci_7_at_the_top() {
    let mut d = Drawing::new(&[("WALLS", Color::Index(3))]);
    d.block("INNER");
    d.block("OUTER");
    let deep = d.line(Some("INNER"), 1.0, "WALLS", Color::ByBlock);
    // INNER is placed inside OUTER with BYBLOCK, so it takes what OUTER's INSERT says.
    d.insert(Some("OUTER"), "INNER", "WALLS", Color::ByBlock);
    let direct = d.line(Some("OUTER"), 2.0, "WALLS", Color::ByBlock);
    d.insert(None, "OUTER", "WALLS", Color::Index(2));
    let run = convert("byblock", &d.bytes(), StreamSelection::ALL);
    let w = words(&run);
    assert_eq!(w[&deep], word(Color::Index(2)), "through two INSERTs");
    assert_eq!(w[&direct], word(Color::Index(2)));
    let _ = fs::remove_dir_all(run.dir);

    // The nearest INSERT that is not BYBLOCK wins; an INSERT BYLAYER takes its own layer's colour.
    let mut d = Drawing::new(&[("WALLS", Color::Index(3)), ("DOORS", Color::Index(5))]);
    d.block("INNER");
    d.block("OUTER");
    let deep = d.line(Some("INNER"), 1.0, "WALLS", Color::ByBlock);
    d.insert(Some("OUTER"), "INNER", "DOORS", Color::ByLayer);
    d.insert(None, "OUTER", "WALLS", Color::Index(2));
    let run = convert("byblock-bylayer", &d.bytes(), StreamSelection::ALL);
    assert_eq!(words(&run)[&deep], word(Color::Index(5)));
    let _ = fs::remove_dir_all(run.dir);

    // Nothing above says a colour: ACI 7, written black.
    let mut d = Drawing::new(&[("WALLS", Color::Index(3))]);
    d.block("OUTER");
    let top = d.line(Some("OUTER"), 1.0, "WALLS", Color::ByBlock);
    let bare = d.line(None, 5.0, "WALLS", Color::ByBlock);
    d.insert(None, "OUTER", "WALLS", Color::ByBlock);
    let run = convert("byblock-top", &d.bytes(), StreamSelection::ALL);
    let w = words(&run);
    assert_eq!(w[&top], BLACK);
    assert_eq!(w[&bare], BLACK);
    let _ = fs::remove_dir_all(run.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_a_true_colour_is_written_as_it_is_and_aci_7_is_black() {
    let mut d = Drawing::new(&[("WALLS", Color::Index(3))]);
    let t = d.line(
        None,
        1.0,
        "WALLS",
        Color::Rgb {
            r: 10,
            g: 20,
            b: 30,
        },
    );
    let seven = d.line(None, 2.0, "WALLS", Color::Index(7));
    let one = d.line(None, 3.0, "WALLS", Color::Index(1));
    let run = convert("true", &d.bytes(), StreamSelection::ALL);
    let w = words(&run);
    assert_eq!(w[&t], 0x0A14_1EFF);
    assert_eq!(w[&seven], BLACK);
    assert_eq!(w[&one], word(Color::Index(1)));
    // The alpha byte is always opaque.
    assert!(w.values().all(|v| v & 0xFF == 0xFF));
    let _ = fs::remove_dir_all(run.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_the_stream_is_its_own_flag_and_leaves_every_other_byte_alone() {
    let mut d = Drawing::new(&[("L-RED", Color::Index(1))]);
    d.line(None, 1.0, "L-RED", Color::ByLayer);
    let bytes = d.bytes();
    let without = convert("flag-off", &bytes, ALL_BUT_PAINT);
    assert!(paint_of(&streams_of(&without.dest)).is_none());
    assert!(
        !without
            .out
            .report
            .components
            .as_ref()
            .unwrap()
            .streams
            .contains(&"stroke-paint".to_owned())
    );
    let only = convert(
        "flag-only",
        &bytes,
        StreamSelection {
            stroke_paint: true,
            ..StreamSelection::NONE
        },
    );
    let streams = streams_of(&only.dest);
    assert_eq!(streams.len(), 1);
    assert!(paint_of(&streams).is_some());
    assert_eq!(
        only.out.report.components.as_ref().unwrap().streams,
        ["stroke-paint"]
    );
    // The same geometry with and without the stream.
    assert_eq!(without.out.written, only.out.written);
    let _ = fs::remove_dir_all(without.dir);
    let _ = fs::remove_dir_all(only.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_a_provider_with_no_colours_writes_no_stream() {
    // The plan fixture's lines carry no colour but one: only that one gets an entry, and a
    // stroke the provider says nothing about is not painted and not counted as ignored.
    let dir = scratch("mock");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .convert_with::<plan::Mock>()
        .unwrap();
    let paint = paint_of(&streams_of(&dest)).expect("the one coloured line");
    assert_eq!(paint.assignments.len(), 1, "{paint:?}");
    assert_eq!(paint.colours, [word(Color::Index(3))]);
    assert_eq!(out.report.components.unwrap().colours_ignored, 0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_colours_ignored_counts_only_the_strokes_that_could_not_be_resolved() {
    // BYLAYER on layer 0 inside a block placed on a layer whose own colour is BYLAYER: the effective
    // layer names no colour to take, so that stroke is unresolved. The others are not.
    let mut d = Drawing::new(&[("WALLS", Color::Index(3)), ("NO-COLOUR", Color::ByLayer)]);
    d.block("B1");
    let lost = d.line(Some("B1"), 1.0, "0", Color::ByLayer);
    d.insert(None, "B1", "NO-COLOUR", Color::ByLayer);
    let fine = d.line(None, 2.0, "WALLS", Color::ByLayer);
    let run = convert("ignored", &d.bytes(), StreamSelection::ALL);
    let w = words(&run);
    assert!(!w.contains_key(&lost));
    assert_eq!(w[&fine], word(Color::Index(3)));
    assert_eq!(
        run.out.report.components.as_ref().unwrap().colours_ignored,
        1
    );
    let _ = fs::remove_dir_all(run.dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1638_a_sheet_paints_the_model_strokes_seen_through_its_viewport() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/sample_AC1018.dwg");
    let dir = scratch("sheet");
    let dest = dir.join("sheet.pvec");
    let out = PvecConversion::new(CadSource::Path(&path), &dest)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .components(StreamSelection::ALL)
        .convert()
        .unwrap();
    let through = out
        .report
        .entities
        .iter()
        .filter(|r| r.viewport.is_some() && r.semantic_type.is_some())
        .count();
    assert!(through > 0, "the viewport shows model strokes");
    let paint = paint_of(&streams_of(&dest)).expect("a stroke paint stream");
    let kept: Vec<_> = out
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .collect();
    let painted_through = paint
        .assignments
        .iter()
        .filter(|(id, _)| kept[id.index() as usize].viewport.is_some())
        .count();
    assert!(painted_through > 0, "{paint:?}");
    assert!(paint.colours.iter().all(|w| w & 0xFF == 0xFF));
    let _ = fs::remove_dir_all(dir);
}
