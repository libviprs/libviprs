//! Issue #1705: an OLE2FRAME (an embedded spreadsheet table, say) is drawn from the preview it
//! carries. A DWG keeps the picture the application cached for display next to the object, and a plot
//! ignored it, so a sheet made of such tables came out blank.
//!
//! Everything is built in memory: acadrust writes a DWG with one OLE2FRAME whose compound file holds a
//! small synthetic EMF (a table: a frame of lines, one filled cell and a text), and the conversion
//! and the plot are the real ones. No private file is read here; the owner's drawing is checked by
//! `pvec_1705_owner_ole.rs`, which skips unless it is pointed at the files. Needs
//! `--features pvec-convert`.
//!
//! The frame sits on an A3 sheet (millimetres, 1 : 1) with its upper-left corner at (100, 200) and its
//! lower-right corner at (150, 175), and the table's metafile is 200 by 100 device pixels, so one
//! pixel is a quarter of a millimetre across and a quarter down.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::compound_file::{
    BinaryRecord, CompoundEntry, CompoundFile, CompoundStorage, CompoundStream,
    CompoundStreamContent,
};
use acadrust::entities::{Line, Ole2Frame, OleFrameEnvelope, OleObjectType};
use acadrust::objects::ObjectType;
use acadrust::types::{Color, LineWeight, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::emf::{EmfRefusal, EmfShape, play};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PdfPlot, PvecConversion, PvecConversionOutput};
use lopdf::Document;
use lopdf::content::Operation;

const SHEET: &str = "Sheet A";
const PT_PER_MM: f64 = 72.0 / 25.4;
/// The frame's corners on the sheet, millimetres.
const UPPER_LEFT: [f64; 2] = [100.0, 200.0];
const LOWER_RIGHT: [f64; 2] = [150.0, 175.0];

// ---- a synthetic EMF ---------------------------------------------------------------------

fn rec(kind: u32, body: &[u8]) -> Vec<u8> {
    let mut out = kind.to_le_bytes().to_vec();
    out.extend(((8 + body.len()) as u32).to_le_bytes());
    out.extend(body);
    out
}

fn words(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// EMR_HEADER for a picture of `w` by `h` device pixels (bounds 0..w-1, 0..h-1).
fn header(w: i32, h: i32, bytes: u32, records: u32) -> Vec<u8> {
    let mut b = words(&[0, 0, w - 1, h - 1, 0, 0, w * 5, h * 5]);
    b.extend(0x464D_4520u32.to_le_bytes());
    b.extend(0x0001_0000u32.to_le_bytes());
    b.extend(bytes.to_le_bytes());
    b.extend(records.to_le_bytes());
    b.extend(3u16.to_le_bytes());
    b.extend(0u16.to_le_bytes());
    b.extend(words(&[0, 0, 0, w, h, w / 4, h / 4]));
    rec(1, &b)
}

fn eof() -> Vec<u8> {
    rec(14, &words(&[0, 16, 20]))
}

fn create_pen(ih: i32, style: i32) -> Vec<u8> {
    rec(38, &words(&[ih, style, 1, 0, 0]))
}

fn create_brush(ih: i32, rgb: [u8; 3]) -> Vec<u8> {
    let c = i32::from_le_bytes([rgb[0], rgb[1], rgb[2], 0]);
    rec(39, &words(&[ih, 0, c, 0]))
}

fn create_font(ih: i32, height: i32) -> Vec<u8> {
    let mut b = words(&[ih, height, 0, 0, 0, 400]);
    b.extend([0u8; 92 - 20]);
    rec(82, &b)
}

fn select(ih: i32) -> Vec<u8> {
    rec(37, &words(&[ih]))
}

fn move_to(x: i32, y: i32) -> Vec<u8> {
    rec(27, &words(&[x, y]))
}

fn line_to(x: i32, y: i32) -> Vec<u8> {
    rec(54, &words(&[x, y]))
}

fn text_align(a: i32) -> Vec<u8> {
    rec(22, &words(&[a]))
}

/// A BitBlt with a pattern ROP (the current brush) over x, y, w, h.
fn pat_blt(x: i32, y: i32, w: i32, h: i32) -> Vec<u8> {
    let mut b = words(&[0, 0, 0, 0, x, y, w, h]);
    b.extend(0x00F0_0021u32.to_le_bytes());
    b.extend([0u8; 100 - 8 - 16 - 16 - 4]);
    rec(76, &b)
}

fn text_out(x: i32, y: i32, s: &str) -> Vec<u8> {
    let units: Vec<u16> = s.encode_utf16().collect();
    let mut b = words(&[0, 0, 0, 0, 1]);
    b.extend(1.0f32.to_le_bytes());
    b.extend(1.0f32.to_le_bytes());
    b.extend(words(&[x, y, units.len() as i32, 76, 0, 0, 0, 0, 0]));
    b.extend(0u32.to_le_bytes());
    let mut tail: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
    while tail.len() % 4 != 0 {
        tail.push(0);
    }
    let off_dx = 76 + tail.len();
    let n = b.len();
    b[n - 4..].copy_from_slice(&(off_dx as u32).to_le_bytes());
    b.extend(tail);
    b.extend(vec![0u8; units.len() * 4]);
    rec(84, &b)
}

/// The table: a frame of four lines, a red cell, the text "Hello" 20 pixels tall on a baseline, and
/// a record type the player does not know (an EMR_SETMITERLIMIT, 58).
fn table() -> Vec<u8> {
    assemble(&[
        create_pen(1, 0),
        select(1),
        move_to(0, 0),
        line_to(199, 0),
        line_to(199, 99),
        line_to(0, 99),
        line_to(0, 0),
        create_brush(2, [255, 0, 0]),
        select(2),
        pat_blt(10, 10, 80, 40),
        create_font(3, -20),
        select(3),
        text_align(24),
        text_out(20, 80, "Hello"),
        rec(58, &words(&[10])),
    ])
}

/// A header, `records` and an end record, as an EMF with consistent counts.
fn assemble(records: &[Vec<u8>]) -> Vec<u8> {
    let body: usize = records.iter().map(Vec::len).sum();
    let total = 88 + body + 20;
    let mut out = header(200, 100, total as u32, records.len() as u32 + 2);
    for r in records {
        out.extend(r);
    }
    out.extend(eof());
    out
}

// ---- a DWG holding an OLE2FRAME ----------------------------------------------------------

fn stream(name: &str, bytes: &[u8]) -> CompoundEntry {
    CompoundEntry::Stream(CompoundStream {
        name: name.to_string(),
        class_id: [0; 16],
        state_bits: 0,
        creation_time: 0,
        modified_time: 0,
        content: CompoundStreamContent::Binary(BinaryRecord::split(bytes, 4096)),
    })
}

/// The frame with the streams of its compound file.
fn frame(streams: &[(&str, Vec<u8>)]) -> Ole2Frame {
    let mut ole = Ole2Frame::new();
    ole.upper_left_corner = Vector3::new(UPPER_LEFT[0], UPPER_LEFT[1], 0.0);
    ole.lower_right_corner = Vector3::new(LOWER_RIGHT[0], LOWER_RIGHT[1], 0.0);
    ole.ole_object_type = OleObjectType::Embedded;
    ole.is_paper_space = true;
    ole.envelope = OleFrameEnvelope::Geometry {
        marker: 0x0080,
        extension_records: Vec::new(),
    };
    let mut root = CompoundStorage::root();
    for (name, bytes) in streams {
        root.entries.push(stream(name, bytes));
    }
    ole.storage.compound_file = Some(CompoundFile {
        root,
        ..CompoundFile::default()
    });
    ole
}

fn sheet_with(ole: Ole2Frame) -> Vec<u8> {
    let mut doc = CadDocument::new();
    let layout = doc.add_layout(SHEET).expect("layout");
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    l.paper_width = 420.0;
    l.paper_height = 297.0;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = 1.0;
    l.plot_scale_denominator = 1.0;
    l.plot_rotation = 0;
    l.plot_margin_left = 0.0;
    l.plot_margin_bottom = 0.0;
    l.plot_origin_x = 0.0;
    l.plot_origin_y = 0.0;
    l.plot_flags.print_lineweights = true;
    let mut rule = Line::from_coords(10.0, 10.0, 0.0, 60.0, 10.0, 0.0);
    rule.common.color = Color::Index(7);
    rule.common.line_weight = LineWeight::Value(25);
    doc.add_entity_to_layout(EntityType::Line(rule), SHEET)
        .expect("a line");
    doc.add_entity_to_layout(EntityType::Ole2Frame(Box::new(ole)), SHEET)
        .expect("the frame");
    DwgWriter::write_to_vec(&doc).expect("write")
}

fn table_sheet() -> Vec<u8> {
    sheet_with(frame(&[("CONTENTS", table())]))
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1705-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn convert(bytes: &[u8], tag: &str, on: bool) -> PvecConversionOutput {
    let dir = scratch(tag);
    PvecConversion::new(CadSource::Bytes(bytes), &dir.join("sheet.pvec"))
        .view_profile(ViewProfile::print_profile(SHEET))
        .ole_previews(on)
        .convert()
        .unwrap_or_else(|e| panic!("the sheet converts: {e:?}"))
}

/// The warnings of every row of the OLE2FRAME: its own and the ones its lines and text came as.
fn ole_warnings(out: &PvecConversionOutput) -> Vec<&String> {
    out.report
        .entities
        .iter()
        .filter(|r| r.source_class == "OLE2FRAME")
        .flat_map(|r| r.warnings.iter())
        .collect()
}

fn ole_says(out: &PvecConversionOutput, what: &str) -> bool {
    ole_warnings(out)
        .iter()
        .any(|w| w.starts_with("OLE2FRAME") && w.contains(what))
}

fn plot(bytes: &[u8], tag: &str, on: bool) -> PathBuf {
    let dir = scratch(tag);
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile(SHEET))
        .layouts([SHEET])
        .texts(true)
        .fills(true)
        .ole_previews(on)
        .convert_with::<AcadrustProvider>()
        .unwrap_or_else(|e| panic!("the sheet plots: {e:?}"));
    dest
}

fn ops(pdf: &PathBuf) -> Vec<Operation> {
    let doc = Document::load(pdf).unwrap();
    let id = doc.get_pages()[&1];
    doc.get_and_decode_page_content(id).unwrap().operations
}

fn real(o: &lopdf::Object) -> f64 {
    o.as_float()
        .map(f64::from)
        .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
}

/// Every point a path operator names, in millimetres.
fn path_points(o: &[Operation]) -> Vec<[f64; 2]> {
    o.iter()
        .filter(|o| o.operator == "m" || o.operator == "l")
        .map(|o| {
            [
                real(&o.operands[0]) / PT_PER_MM,
                real(&o.operands[1]) / PT_PER_MM,
            ]
        })
        .collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.02
}

// ---- the player --------------------------------------------------------------------------

#[test]
fn issue_1705_the_player_maps_the_table_onto_the_corners() {
    let drawing = play(&table(), UPPER_LEFT, LOWER_RIGHT).expect("the table plays");
    let lines: Vec<&Vec<[f64; 2]>> = drawing
        .shapes
        .iter()
        .filter_map(|s| match s {
            EmfShape::Line(p) => Some(p),
            _ => None,
        })
        .collect();
    let points: Vec<[f64; 2]> = lines.iter().flat_map(|l| l.iter().copied()).collect();
    assert!(!points.is_empty(), "the frame of the table is drawn");
    let (lo, hi) = points
        .iter()
        .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1])],
                [hi[0].max(p[0]), hi[1].max(p[1])],
            )
        });
    // The picture's pixel (0, 0) is the upper-left corner and (199, 99) the lower-right one.
    assert!(near(lo[0], 100.0) && near(hi[0], 150.0), "x {lo:?} {hi:?}");
    assert!(near(lo[1], 175.0) && near(hi[1], 200.0), "y {lo:?} {hi:?}");
}

#[test]
fn issue_1705_a_pattern_blt_is_a_solid_fill_in_the_brush_colour() {
    let drawing = play(&table(), UPPER_LEFT, LOWER_RIGHT).unwrap();
    let fills: Vec<(&Vec<[f64; 2]>, u32)> = drawing
        .shapes
        .iter()
        .filter_map(|s| match s {
            EmfShape::Fill { ring, rgba } => Some((ring, *rgba)),
            _ => None,
        })
        .collect();
    assert_eq!(fills.len(), 1, "one cell");
    assert_eq!(fills[0].1, 0xFF00_00FF, "red, opaque");
    let xs: Vec<f64> = fills[0].0.iter().map(|p| p[0]).collect();
    // Pixels 10 to 90 of 199 across 50 mm.
    let (x0, x1) = (100.0 + 10.0 * 50.0 / 199.0, 100.0 + 90.0 * 50.0 / 199.0);
    assert!(
        xs.iter().any(|x| near(*x, x0)) && xs.iter().any(|x| near(*x, x1)),
        "{xs:?}"
    );
}

#[test]
fn issue_1705_text_is_a_text_run_at_its_baseline() {
    let drawing = play(&table(), UPPER_LEFT, LOWER_RIGHT).unwrap();
    let runs: Vec<_> = drawing
        .shapes
        .iter()
        .filter_map(|s| match s {
            EmfShape::Text {
                at,
                height,
                text,
                anchor,
                ..
            } => Some((*at, *height, text.clone(), *anchor)),
            _ => None,
        })
        .collect();
    assert_eq!(runs.len(), 1);
    let (at, height, text, anchor) = &runs[0];
    assert_eq!(text, "Hello");
    assert!(near(at[0], 100.0 + 20.0 * 50.0 / 199.0), "x {at:?}");
    assert!(near(at[1], 200.0 - 80.0 * 25.0 / 99.0), "y {at:?}");
    assert_eq!(*anchor, 0.0, "the run starts at the reference point");
    // A 20 pixel face is 20 * 25 / 99 mm, and its capitals stand 0.716 of it.
    let cap = 20.0 * 25.0 / 99.0 * 0.716;
    assert!((height - cap).abs() < 0.05, "{height} against {cap}");
}

#[test]
fn issue_1705_a_record_the_player_does_not_know_is_counted_in_the_notes() {
    let drawing = play(&table(), UPPER_LEFT, LOWER_RIGHT).unwrap();
    assert!(
        drawing
            .notes
            .iter()
            .any(|n| n.contains("17") && n.contains('1')),
        "{:?}",
        drawing.notes
    );
}

#[test]
fn issue_1705_a_preview_that_is_not_an_emf_is_refused() {
    let err = play(&[0u8; 200], UPPER_LEFT, LOWER_RIGHT).unwrap_err();
    assert_eq!(err, EmfRefusal::NotAMetafile);
}

#[test]
fn issue_1705_a_record_that_runs_past_the_end_is_refused() {
    let mut bytes = table();
    // The size of the first record after the header claims more than is there.
    bytes[88 + 4..88 + 8].copy_from_slice(&0x00FF_FF00u32.to_le_bytes());
    let err = play(&bytes, UPPER_LEFT, LOWER_RIGHT).unwrap_err();
    assert!(matches!(err, EmfRefusal::BadRecord { index: 1 }), "{err:?}");
}

#[test]
fn issue_1705_a_record_of_a_size_that_is_not_a_multiple_of_four_is_refused() {
    let mut bytes = table();
    bytes[88 + 4..88 + 8].copy_from_slice(&13u32.to_le_bytes());
    assert!(matches!(
        play(&bytes, UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::BadRecord { .. })
    ));
}

#[test]
fn issue_1705_more_records_than_the_limit_are_refused() {
    let records: Vec<Vec<u8>> = (0..300_000).map(|_| text_align(0)).collect();
    let bytes = assemble(&records);
    assert!(matches!(
        play(&bytes, UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::TooManyRecords { .. })
    ));
}

#[test]
fn issue_1705_a_preview_over_the_byte_limit_is_refused() {
    let mut bytes = table();
    bytes.resize(9 << 20, 0);
    assert!(matches!(
        play(&bytes, UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::TooLarge { .. })
    ));
}

#[test]
fn issue_1705_a_preview_that_draws_more_shapes_than_the_limit_is_refused() {
    let mut records = vec![create_pen(1, 0), select(1)];
    for i in 0..5000 {
        records.push(move_to(0, i % 100));
        records.push(line_to(199, i % 100));
    }
    assert!(matches!(
        play(&assemble(&records), UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::TooManyShapes { .. })
    ));
}

#[test]
fn issue_1705_a_picture_with_no_extent_is_refused() {
    let mut bytes = table();
    // Bounds right = left: no width to map onto the corners.
    bytes[8 + 8..8 + 12].copy_from_slice(&0i32.to_le_bytes());
    assert!(matches!(
        play(&bytes, UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::NoExtent)
    ));
}

#[test]
fn issue_1705_an_unbounded_save_stack_is_refused_not_grown() {
    let records: Vec<Vec<u8>> = (0..5000).map(|_| rec(33, &[])).collect();
    assert!(matches!(
        play(&assemble(&records), UPPER_LEFT, LOWER_RIGHT),
        Err(EmfRefusal::TooDeep)
    ));
}

#[test]
fn issue_1705_a_selected_object_that_was_never_made_does_not_panic() {
    let records = vec![
        select(999),
        move_to(0, 0),
        line_to(10, 10),
        rec(40, &words(&[7_000_000])),
    ];
    // Either it plays (an unknown handle selects nothing) or it is refused, but it never panics.
    let _ = play(&assemble(&records), UPPER_LEFT, LOWER_RIGHT);
}

fn map_mode(m: i32) -> Vec<u8> {
    rec(17, &words(&[m]))
}

fn window_org(x: i32, y: i32) -> Vec<u8> {
    rec(10, &words(&[x, y]))
}

fn window_ext(x: i32, y: i32) -> Vec<u8> {
    rec(9, &words(&[x, y]))
}

fn viewport_org(x: i32, y: i32) -> Vec<u8> {
    rec(12, &words(&[x, y]))
}

fn viewport_ext(x: i32, y: i32) -> Vec<u8> {
    rec(11, &words(&[x, y]))
}

/// SETWORLDTRANSFORM (35) with eM11, eM12, eM21, eM22, eDx, eDy.
fn world(m: [f32; 6]) -> Vec<u8> {
    rec(
        35,
        &m.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
}

/// The corners a frame of one black line across the picture reaches.
fn line_extent(records: &[Vec<u8>]) -> ([f64; 2], [f64; 2]) {
    let drawing = play(&assemble(records), UPPER_LEFT, LOWER_RIGHT).expect("plays");
    let points: Vec<[f64; 2]> = drawing
        .shapes
        .iter()
        .filter_map(|s| match s {
            EmfShape::Line(p) => Some(p.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(!points.is_empty(), "a line is drawn");
    let lo = points
        .iter()
        .fold([f64::MAX; 2], |a, p| [a[0].min(p[0]), a[1].min(p[1])]);
    let hi = points
        .iter()
        .fold([f64::MIN; 2], |a, p| [a[0].max(p[0]), a[1].max(p[1])]);
    (lo, hi)
}

// ---- window, viewport and world transform (the review of #1710) ----------------------------

#[test]
fn issue_1705_a_window_origin_and_extents_map_the_logical_picture_onto_the_device() {
    // The window is 100 by 50 logical units at (1000, 500) and the viewport is 199 by 99 pixels, so
    // a line from the window's corner to its far corner crosses the whole picture.
    let (lo, hi) = line_extent(&[
        map_mode(8),
        window_org(1000, 500),
        window_ext(100, 50),
        viewport_org(0, 0),
        viewport_ext(199, 99),
        create_pen(1, 0),
        select(1),
        move_to(1000, 500),
        line_to(1100, 550),
    ]);
    assert!(near(lo[0], 100.0) && near(hi[0], 150.0), "x {lo:?} {hi:?}");
    assert!(near(lo[1], 175.0) && near(hi[1], 200.0), "y {lo:?} {hi:?}");
}

#[test]
fn issue_1705_a_world_transform_scales_and_moves_what_follows() {
    // x doubled, y moved down 10: a line (0, 0) to (99, 89) is the picture's diagonal.
    let (lo, hi) = line_extent(&[
        world([2.0, 0.0, 0.0, 1.0, 1.0, 10.0]),
        create_pen(1, 0),
        select(1),
        move_to(0, 0),
        line_to(99, 89),
    ]);
    let x = |px: f64| 100.0 + px * 50.0 / 199.0;
    let y = |py: f64| 200.0 - py * 25.0 / 99.0;
    assert!(
        near(lo[0], x(1.0)) && near(hi[0], x(199.0)),
        "x {lo:?} {hi:?}"
    );
    assert!(
        near(lo[1], y(99.0)) && near(hi[1], y(10.0)),
        "y {lo:?} {hi:?}"
    );
}

#[test]
fn issue_1705_a_modified_world_transform_composes() {
    // Identity, then a left-multiplied scale by 2 in x: the same picture as setting it outright.
    let mut modify = words(&[0; 0]);
    modify.extend(
        [2.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes()),
    );
    modify.extend(2u32.to_le_bytes());
    let (lo, hi) = line_extent(&[
        rec(36, &modify),
        create_pen(1, 0),
        select(1),
        move_to(0, 0),
        line_to(99, 99),
    ]);
    assert!(
        near(lo[0], 100.0) && near(hi[0], 100.0 + 198.0 * 50.0 / 199.0),
        "x {lo:?} {hi:?}"
    );
}

#[test]
fn issue_1705_text_past_its_limit_says_so_and_is_not_a_shape_count() {
    let long = "a".repeat(4096);
    let records: Vec<Vec<u8>> = (0..40).map(|i| text_out(10, 10 + i, &long)).collect();
    let err = play(&assemble(&records), UPPER_LEFT, LOWER_RIGHT).unwrap_err();
    assert!(matches!(err, EmfRefusal::TooMuchText { .. }), "{err:?}");
}

// ---- the conversion ----------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_off_by_default_the_frame_is_counted_and_nothing_is_drawn() {
    let out = convert(&table_sheet(), "off", false);
    assert!(
        !ole_warnings(&out).iter().any(|w| w.contains("drawn")),
        "{:?}",
        ole_warnings(&out)
    );
    let reader = open_published(&out.artifact).unwrap();
    // The only curve is the sheet's own rule.
    assert_eq!(reader.entities().unwrap().len(), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_with_previews_on_the_table_is_in_the_artifact() {
    let out = convert(&table_sheet(), "on", true);
    assert!(
        ole_says(&out, "drawn from its EMF preview"),
        "{:?}",
        ole_warnings(&out)
    );
    assert!(
        ole_says(&out, "17"),
        "an unknown record is named: {:?}",
        ole_warnings(&out)
    );
    let reader = open_published(&out.artifact).unwrap();
    assert!(
        reader.entities().unwrap().len() > 1,
        "the table's lines come in beside the rule"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_frame_with_no_preview_has_a_row_that_says_so() {
    let bytes = sheet_with(frame(&[("Other", vec![1, 2, 3, 4])]));
    let out = convert(&bytes, "none", true);
    assert!(ole_says(&out, "no preview"), "{:?}", ole_warnings(&out));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_windows_metafile_preview_is_not_played_and_says_so() {
    // A placeable WMF header and one end record: the object's cached picture, not an EMF.
    let mut wmf = 0x9AC6_CDD7u32.to_le_bytes().to_vec();
    wmf.extend([0u8; 18]);
    wmf.extend([
        1, 0, 9, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
    ]);
    let bytes = sheet_with(frame(&[("CONTENTS", wmf)]));
    let out = convert(&bytes, "wmf", true);
    assert!(ole_says(&out, "not drawn"), "{:?}", ole_warnings(&out));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_hostile_preview_is_a_typed_row_and_the_sheet_still_converts() {
    let mut bad = table();
    bad[88 + 4..88 + 8].copy_from_slice(&0x00FF_FF00u32.to_le_bytes());
    let out = convert(&sheet_with(frame(&[("CONTENTS", bad)])), "hostile", true);
    assert!(ole_says(&out, "not drawn"), "{:?}", ole_warnings(&out));
    let reader = open_published(&out.artifact).unwrap();
    assert_eq!(reader.entities().unwrap().len(), 1, "only the sheet's rule");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_png_preview_is_placed_like_an_image_at_the_corners() {
    let img = image::RgbaImage::from_fn(4, 2, |x, y| {
        image::Rgba([(60 * x + 30) as u8, (100 * y + 30) as u8, 128, 255])
    });
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png).unwrap();
    let bytes = sheet_with(frame(&[("CONTENTS", png.into_inner())]));
    let out = convert(&bytes, "png", true);
    assert!(
        ole_says(&out, "placed as a raster"),
        "{:?}",
        ole_warnings(&out)
    );
    let reader = open_published(&out.artifact).unwrap();
    let placed = reader.raster_placements().unwrap();
    assert_eq!(placed.len(), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_png_over_the_pixel_limit_is_refused_before_it_is_decoded() {
    // A header that declares 100000 by 100000 pixels with no pixels behind it.
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
    png.extend(b"IHDR");
    png.extend(100_000u32.to_be_bytes());
    png.extend(100_000u32.to_be_bytes());
    png.extend([8, 2, 0, 0, 0, 0, 0, 0, 0]);
    let out = convert(&sheet_with(frame(&[("CONTENTS", png)])), "pngbomb", true);
    assert!(ole_says(&out, "not drawn"), "{:?}", ole_warnings(&out));
}

// ---- the plot ----------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_plot_draws_the_table_at_the_frame_corners() {
    let pdf = plot(&table_sheet(), "plot", true);
    let o = ops(&pdf);
    let pts = path_points(&o);
    // The table frame reaches both corners (the sheet's own rule is at y 10).
    let table: Vec<&[f64; 2]> = pts.iter().filter(|p| p[1] > 100.0).collect();
    assert!(
        table.iter().any(|p| near(p[0], 100.0) && near(p[1], 200.0)),
        "{pts:?}"
    );
    assert!(
        table.iter().any(|p| near(p[0], 150.0) && near(p[1], 175.0)),
        "{pts:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_plot_shows_the_cell_text_as_text_and_fills_the_cell() {
    let pdf = plot(&table_sheet(), "plottext", true);
    let o = ops(&pdf);
    let shown: Vec<u8> = o
        .iter()
        .filter(|o| o.operator == "Tj")
        .flat_map(|o| o.operands[0].as_str().unwrap().to_vec())
        .collect();
    assert_eq!(shown, b"Hello");
    let tm = o
        .iter()
        .find(|o| o.operator == "Tm")
        .expect("text is placed");
    let x = real(&tm.operands[4]) / PT_PER_MM;
    let y = real(&tm.operands[5]) / PT_PER_MM;
    assert!(
        near(x, 100.0 + 20.0 * 50.0 / 199.0) && near(y, 200.0 - 80.0 * 25.0 / 99.0),
        "{x} {y}"
    );
    assert!(
        o.iter().any(|o| o.operator == "f" || o.operator == "f*"),
        "the red cell is filled"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_plot_without_the_option_draws_none_of_it() {
    let pdf = plot(&table_sheet(), "plotoff", false);
    let o = ops(&pdf);
    assert!(
        path_points(&o).iter().all(|p| p[1] < 100.0),
        "only the rule"
    );
    assert!(!o.iter().any(|o| o.operator == "Tj"));
}

// ---- the allocation budget ---------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_raster_preview_is_charged_to_the_allocation_budget_before_it_is_decoded() {
    let img = image::RgbaImage::from_pixel(600, 600, image::Rgba([10, 20, 30, 255]));
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png).unwrap();
    let bytes = sheet_with(frame(&[("CONTENTS", png.into_inner())]));
    let dir = scratch("budget");
    // 600 x 600 x 4 is 1.44 MB of pixels, over a 1 MiB budget.
    let mut limits = libviprs::pvec::SourceLimits::default();
    limits.max_allocation_bytes = 1 << 20;
    let e = PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("sheet.pvec"))
        .view_profile(ViewProfile::print_profile(SHEET))
        .limits(limits)
        .ole_previews(true)
        .convert()
        .unwrap_err();
    assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES", "{e}");
}
