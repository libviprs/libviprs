//! Issue #1621: DWG text in the PVec text stream matches what the drawing prints. With
//! `annotations(true)` and the line style stream, a TEXT, MTEXT or attribute is carried in model
//! space and on a print layout, its anchor is as long as the printed run (planviewer scales the
//! glyphs to that length), its position is the printed baseline start after justification and
//! attachment, and an MTEXT is one record per printed line.
//!
//! The drawings here are written with acadrust. The last tests read Rom's drawing from the path in
//! `LIBVIPRS_DWG_TEXT_FIXTURE` (24 MiB, so it is not committed) and, once they sit beside it, the
//! per-layout PDFs it was printed to; each part skips with its reason when its input is missing.
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use acadrust::entities::{
    AttachmentPoint, MText, Text, TextHorizontalAlignment, TextVerticalAlignment,
};
use acadrust::objects::ObjectType;
use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
use libviprs::cad::{CadSource, display_text};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::dwg_acadrust::{AcadrustProvider, TextFact};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PvecConversion, PvecLayoutSet};
use pvec_facade::Curve;
use pvec_facade::components::TextStream;

/// Line style for the hidden anchors, and the text stream.
const STREAMS: StreamSelection = StreamSelection {
    line_style: true,
    ..StreamSelection::NONE
};

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1621-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// What a text record says, read out of the stream's own record type (the facade does not name
/// it).
struct Record {
    text: String,
    x: f64,
    y: f64,
    height: f64,
}

/// A text record with the length of the polyline it is anchored on.
struct Carried {
    record: Record,
    advance: f64,
}

/// The text records of the artifact at `path`, each with its anchor's length.
fn carried(path: &Path) -> Vec<Carried> {
    let reader = open_published(path).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let Some(stream) = streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)
    else {
        return Vec::new();
    };
    let entities = reader.entities().unwrap();
    TextStream::decode(&stream.payload)
        .unwrap()
        .records
        .into_iter()
        .map(|record| {
            let advance =
                entities
                    .iter()
                    .find(|(id, _)| *id == record.entity)
                    .map_or(0.0, |(_, curve)| match curve {
                        Curve::Polyline(p) => {
                            let v = p.vertices();
                            let (a, z) = (&v[0], &v[v.len() - 1]);
                            (z.x - a.x).hypot(z.y - a.y)
                        }
                        other => panic!("a text anchor is a polyline, not {other:?}"),
                    });
            Carried {
                record: Record {
                    text: record.text,
                    x: record.x,
                    y: record.y,
                    height: record.height,
                },
                advance,
            }
        })
        .collect()
}

/// Model space of `doc`, converted with the text stream on, in millimetres.
fn model(tag: &str, doc: &CadDocument) -> Vec<Carried> {
    let bytes = DwgWriter::write_to_vec(doc).unwrap();
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(STREAMS)
        .annotations(true)
        .convert()
        .unwrap();
    let out = carried(&dest);
    let _ = fs::remove_dir_all(dir);
    out
}

fn one<'a>(all: &'a [Carried], text: &str) -> &'a Carried {
    let found: Vec<_> = all.iter().filter(|c| c.record.text == text).collect();
    assert_eq!(
        found.len(),
        1,
        "one record {text:?} in {:?}",
        all.iter().map(|c| &c.record.text).collect::<Vec<_>>()
    );
    found[0]
}

fn text(value: &str, at: [f64; 2], height: f64) -> Text {
    Text::with_value(value, Vector3::new(at[0], at[1], 0.0)).with_height(height)
}

fn mtext(value: &str, at: [f64; 2], height: f64, attach: AttachmentPoint) -> MText {
    let mut m = MText::with_value(value, Vector3::new(at[0], at[1], 0.0)).with_height(height);
    m.attachment_point = attach;
    m
}

/// A line so the drawing has geometry whatever is carried.
fn frame(doc: &mut CadDocument) {
    doc.add_entity(EntityType::Line(acadrust::entities::Line::from_points(
        Vector3::new(-500.0, -500.0, 0.0),
        Vector3::new(500.0, 500.0, 0.0),
    )))
    .unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_text_anchor_is_as_long_as_the_printed_run() {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    doc.add_entity(EntityType::Text(text("MECHANICAL ROOM", [0.0, 0.0], 2.5)))
        .unwrap();
    let all = model("advance", &doc);
    let room = one(&all, "MECHANICAL ROOM");
    // Any font prints 15 capitals at height 2.5 somewhere between half and 1.1 heights a
    // character; one height for the whole run (the anchor until #1621) squeezes it tenfold.
    let per_char = room.advance / (15.0 * 2.5);
    assert!(
        (0.5..=1.1).contains(&per_char),
        "advance {} for 15 characters at height 2.5 ({per_char} heights a character)",
        room.advance
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_the_width_factor_scales_the_run() {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    let mut narrow = text("NARROW", [0.0, 20.0], 2.0);
    narrow.width_factor = 0.5;
    doc.add_entity(EntityType::Text(narrow)).unwrap();
    doc.add_entity(EntityType::Text(text("NARROW1", [0.0, 40.0], 2.0)))
        .unwrap();
    let all = model("width", &doc);
    let half = one(&all, "NARROW").advance;
    let full = one(&all, "NARROW1").advance;
    // "NARROW1" is "NARROW" and one more character, so at factor 1 it is longer than twice the
    // factor 0.5 run by the width of that character alone.
    assert!(
        half > 0.0 && full > 2.0 * half,
        "{half} at 0.5, {full} at 1"
    );
    assert!(
        full < 2.0 * half * 7.0 / 6.0 * 1.25,
        "{half} at 0.5, {full} at 1"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_centred_text_is_drawn_from_its_printed_start() {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    let mut centred = text("CORRIDOR", [100.0, 0.0], 3.0);
    centred.horizontal_alignment = TextHorizontalAlignment::Center;
    centred.vertical_alignment = TextVerticalAlignment::Baseline;
    centred.alignment_point = Some(Vector3::new(100.0, 0.0, 0.0));
    doc.add_entity(EntityType::Text(centred)).unwrap();
    let mut right = text("STAIR", [100.0, 30.0], 3.0);
    right.horizontal_alignment = TextHorizontalAlignment::Right;
    right.alignment_point = Some(Vector3::new(100.0, 30.0, 0.0));
    doc.add_entity(EntityType::Text(right)).unwrap();
    let all = model("justify", &doc);
    let c = one(&all, "CORRIDOR");
    // The middle of the printed run is on the alignment point, on its baseline.
    assert!(
        (c.record.x + c.advance / 2.0 - 100.0).abs() < 1e-6 && c.record.y.abs() < 1e-6,
        "starts at ({}, {}), {} long",
        c.record.x,
        c.record.y,
        c.advance
    );
    let r = one(&all, "STAIR");
    assert!(
        (r.record.x + r.advance - 100.0).abs() < 1e-6 && (r.record.y - 30.0).abs() < 1e-6,
        "starts at ({}, {}), {} long",
        r.record.x,
        r.record.y,
        r.advance
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_an_mtext_is_one_record_per_printed_line() {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    doc.add_entity(EntityType::MText(Box::new(mtext(
        "FIRST LINE\\PSECOND {\\fArial|b1;LINE}",
        [0.0, 50.0],
        2.0,
        AttachmentPoint::TopLeft,
    ))))
    .unwrap();
    let all = model("paragraphs", &doc);
    let first = one(&all, "FIRST LINE");
    let second = one(&all, "SECOND LINE");
    // Top left attachment: the first baseline is under the insertion point by about a cap height,
    // and the next line is under it.
    assert!(
        first.record.x.abs() < 1e-6 && second.record.x.abs() < 1e-6,
        "{} and {}",
        first.record.x,
        second.record.x
    );
    assert!(
        first.record.y < 50.0 && first.record.y > 50.0 - 2.0 * 2.0,
        "first baseline at {}",
        first.record.y
    );
    assert!(
        second.record.y < first.record.y - 2.0,
        "second baseline {} under the first {}",
        second.record.y,
        first.record.y
    );
    assert!(all.iter().all(|c| !c.record.text.contains('\n')));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_top_centre_mtext_hangs_centred_under_its_insertion_point() {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    doc.add_entity(EntityType::MText(Box::new(mtext(
        "ROOM 101",
        [200.0, 50.0],
        2.5,
        AttachmentPoint::TopCenter,
    ))))
    .unwrap();
    let all = model("attach", &doc);
    let room = one(&all, "ROOM 101");
    assert!(
        (room.record.x + room.advance / 2.0 - 200.0).abs() < 1e-6,
        "starts at {}, {} long",
        room.record.x,
        room.advance
    );
    assert!(
        room.record.y < 50.0 && room.record.y > 50.0 - 2.0 * 2.5,
        "baseline at {}",
        room.record.y
    );
}

/// An A3 layout in millimetres plotted 1:1, holding `texts` in its own paper space.
fn sheet_with(texts: Vec<EntityType>) -> Vec<u8> {
    let mut doc = CadDocument::new();
    frame(&mut doc);
    let layout = doc.add_layout("Sheet A").unwrap();
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
    doc.add_entity_to_layout(
        EntityType::Line(acadrust::entities::Line::from_points(
            Vector3::new(10.0, 10.0, 0.0),
            Vector3::new(410.0, 10.0, 0.0),
        )),
        "Sheet A",
    )
    .unwrap();
    for t in texts {
        doc.add_entity_to_layout(t, "Sheet A").unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_text_on_a_print_layout_is_in_that_sheets_text_stream() {
    let bytes = sheet_with(vec![
        EntityType::Text(text("SHEET TITLE", [20.0, 20.0], 5.0)),
        EntityType::MText(Box::new(mtext(
            "DRAWN BY",
            [300.0, 40.0],
            2.5,
            AttachmentPoint::BottomLeft,
        ))),
    ]);
    let dir = scratch("sheet");
    let dest = dir.join("sheet.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .view_profile(ViewProfile::print_profile("Sheet A"))
        .components(STREAMS)
        .annotations(true)
        .convert()
        .expect("the sheet converts");
    let all = carried(&dest);
    let counts = out.report.components.map(|c| c.annotations);
    let title = all
        .iter()
        .find(|c| c.record.text == "SHEET TITLE")
        .unwrap_or_else(|| panic!("SHEET TITLE on the sheet; annotation counts {counts:?}"));
    assert!(
        (title.record.x - 20.0).abs() < 1e-6 && (title.record.y - 20.0).abs() < 1e-6,
        "({}, {})",
        title.record.x,
        title.record.y
    );
    assert!((title.record.height - 5.0).abs() < 1e-9);
    assert!(
        all.iter().any(|c| c.record.text == "DRAWN BY"),
        "DRAWN BY on the sheet; annotation counts {counts:?}"
    );
    let _ = fs::remove_dir_all(dir);
}

// ---- Rom's drawing ---------------------------------------------------------------------------

/// The drawing at `LIBVIPRS_DWG_TEXT_FIXTURE`, or `None` (with the reason on stderr) when the
/// variable is not set or names no file. Rom's is `MASTER_FLOOR_PLAN_Rev.6.0_NOV_08_2026.dwg`,
/// sha256 e11e3b1130d64bf7a9bf34bfc10b5312fa78d7f17105a30d3dd3a9978a83cceb.
fn fixture() -> Option<PathBuf> {
    let Some(path) = std::env::var_os("LIBVIPRS_DWG_TEXT_FIXTURE") else {
        eprintln!(
            "skipped: LIBVIPRS_DWG_TEXT_FIXTURE is not set (point it at Rom's MASTER_FLOOR_PLAN DWG)"
        );
        return None;
    };
    let path = PathBuf::from(path);
    if !path.is_file() {
        eprintln!(
            "skipped: LIBVIPRS_DWG_TEXT_FIXTURE names {} and that is not a file",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// The lines a text prints: MTEXT codes read, one entry per line, blank lines left out.
fn lines(fact: &TextFact) -> Vec<String> {
    let shown = if fact.multiline {
        display_text(&fact.content).text
    } else {
        fact.content.clone()
    };
    shown
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Every line printed by the TEXT and MTEXT entities and the attributes of the INSERTs that sit
/// directly in `scope`, counted. Text follows the layer rules of Rom's call of 2026-11-19 in both
/// modes: a TEXT or MTEXT on a layer that is on and plots is expected, frozen or not; an INSERT's
/// attributes only when the INSERT's own layer is on, thawed and plots (a frozen INSERT takes them
/// with it, and the attribute's own layer is not known here, so this stays a lower bound).
fn expected(provider: &AcadrustProvider, scope: &EntityScope) -> BTreeMap<String, u64> {
    let layers = provider.layers().unwrap();
    let shown = |name: &str, frozen_hides: bool| {
        layers
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(name))
            .is_none_or(|l| !l.off && l.plottable && !(frozen_hides && l.frozen))
    };
    let mut want = BTreeMap::new();
    for raw in provider.entities(scope).unwrap().collect_all().unwrap() {
        let Some(handle) = raw.provenance.handle else {
            continue;
        };
        let layer = raw.provenance.layer.as_str();
        let facts: Vec<TextFact> = match raw.provenance.class_name.as_str() {
            "TEXT" | "MTEXT" if shown(layer, false) => {
                provider.text_fact(handle).into_iter().collect()
            }
            "INSERT" if shown(layer, true) => provider.attribute_texts(handle),
            _ => continue,
        };
        for line in facts.iter().flat_map(lines) {
            *want.entry(line).or_default() += 1;
        }
    }
    want
}

/// The lines in the text stream of the artifact at `path`, counted.
fn got(path: &Path) -> BTreeMap<String, u64> {
    let mut have = BTreeMap::new();
    for c in carried(path) {
        for line in c
            .record
            .text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            *have.entry(line.to_owned()).or_default() += 1;
        }
    }
    have
}

/// What `want` has more of than `have`.
fn missing(want: &BTreeMap<String, u64>, have: &BTreeMap<String, u64>) -> Vec<(String, u64)> {
    want.iter()
        .filter_map(|(k, n)| {
            let short = n.saturating_sub(have.get(k).copied().unwrap_or(0));
            (short > 0).then(|| (k.clone(), short))
        })
        .collect()
}

fn drawing_unit(provider: &AcadrustProvider) -> SourceUnit {
    match provider.source_unit() {
        SourceUnit::Unitless | SourceUnit::Unknown => SourceUnit::Millimeters,
        other => other,
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_roms_drawing_model_space_text_is_all_in_the_stream() {
    let Some(path) = fixture() else { return };
    let provider = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
    let model = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model space");
    let want = expected(&provider, &EntityScope::Layout(model.id));
    let total: u64 = want.values().sum();
    assert!(total > 1000, "the drawing has its text: {total} lines");
    let dir = scratch("rom-model");
    let dest = dir.join("model.pvec");
    PvecConversion::new(CadSource::Path(&path), &dest)
        .unit(drawing_unit(&provider))
        .components(STREAMS)
        .annotations(true)
        .convert()
        .unwrap();
    let short = missing(&want, &got(&dest));
    let lost: u64 = short.iter().map(|(_, n)| n).sum();
    assert!(
        short.is_empty(),
        "{lost} of {total} model space lines are not in the text stream, e.g. {:?}",
        &short[..short.len().min(20)]
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_roms_drawing_each_layouts_own_text_is_on_its_sheet() {
    let Some(path) = fixture() else { return };
    let provider = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
    let dir = scratch("rom-sheets");
    let out = PvecLayoutSet::new(CadSource::Path(&path), &dir)
        .sheets(|c| c.components(STREAMS).annotations(true))
        .convert()
        .unwrap();
    let layouts = provider.layouts().unwrap();
    let mut report = Vec::new();
    let mut checked = 0;
    for sheet in out.written() {
        let layout = layouts
            .iter()
            .find(|l| l.name == sheet.layout)
            .expect("the sheet's layout");
        let want = expected(&provider, &EntityScope::Layout(layout.id));
        if want.is_empty() {
            continue;
        }
        checked += 1;
        let short = missing(&want, &got(&sheet.file));
        if !short.is_empty() {
            let total: u64 = want.values().sum();
            let lost: u64 = short.iter().map(|(_, n)| n).sum();
            report.push(format!(
                "{}: {lost} of {total} lines missing, e.g. {:?}",
                sheet.layout,
                &short[..short.len().min(5)]
            ));
        }
    }
    assert!(checked > 0, "some layout has text of its own");
    assert!(report.is_empty(), "{}", report.join("\n"));
    let _ = fs::remove_dir_all(dir);
}

/// The PDFs beside the drawing, which Rom prints one per layout. Matched to a layout when the
/// file stem, lowercased with everything but letters and digits dropped, ends with the layout
/// name treated the same way.
#[cfg(feature = "pdfium")]
fn prints(path: &Path) -> Vec<PathBuf> {
    let mut pdfs: Vec<PathBuf> = fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("pdf")))
        .collect();
    pdfs.sort();
    pdfs
}

#[cfg(feature = "pdfium")]
fn squash(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(feature = "pdfium")]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_roms_drawing_each_sheet_shows_the_text_its_pdf_prints() {
    use libviprs::pvec::{PdfVectorLimits, extract_page_paths};

    let Some(path) = fixture() else { return };
    let pdfs = prints(&path);
    if pdfs.is_empty() {
        eprintln!(
            "skipped: no per-layout PDF beside {} yet (Rom adds them on 2026-11-20)",
            path.display()
        );
        return;
    }
    let dir = scratch("rom-prints");
    let out = PvecLayoutSet::new(CadSource::Path(&path), &dir)
        .sheets(|c| c.components(STREAMS).annotations(true))
        .convert()
        .unwrap();
    let mut report = Vec::new();
    let mut compared = 0;
    for pdf in &pdfs {
        let stem = squash(&pdf.file_stem().unwrap().to_string_lossy());
        let Some(sheet) = out
            .written()
            .filter(|s| stem.ends_with(&squash(&s.layout)))
            .max_by_key(|s| squash(&s.layout).len())
        else {
            report.push(format!("{}: no converted layout matches it", pdf.display()));
            continue;
        };
        compared += 1;
        let page = extract_page_paths(pdf, 1, &PdfVectorLimits::default()).unwrap();
        // Word by word: a PDF run and a DWG string need not split a line the same way.
        let words = |s: &str| s.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        let mut want: BTreeMap<String, u64> = BTreeMap::new();
        for run in &page.texts {
            for w in words(&run.text) {
                *want.entry(w).or_default() += 1;
            }
        }
        let mut have: BTreeMap<String, u64> = BTreeMap::new();
        let records = carried(&sheet.file);
        for c in &records {
            for w in words(&c.record.text) {
                *have.entry(w).or_default() += 1;
            }
        }
        let total: u64 = want.values().sum();
        let short = missing(&want, &have);
        let lost: u64 = short.iter().map(|(_, n)| n).sum();
        // Hatch pattern names and the odd field the plotter evaluates differently aside, the sheet
        // shows what the print shows.
        if total > 0 && lost * 100 > total {
            report.push(format!(
                "{} ({}): {lost} of {total} printed words not on the sheet, e.g. {:?}",
                pdf.display(),
                sheet.layout,
                &short[..short.len().min(10)]
            ));
        }
        // Where a whole run matches one record, it starts where the print starts it, within half
        // the text height. The PDF is in points; the sheet is in the unit its manifest names.
        let metres_per_unit = open_published(&sheet.file)
            .unwrap()
            .manifest()
            .metres_per_unit;
        let to_sheet = 0.0254 / 72.0 / metres_per_unit;
        let mut far = 0;
        let mut matched = 0;
        for run in &page.texts {
            let Some(c) = records
                .iter()
                .find(|c| c.record.text.trim() == run.text.trim())
            else {
                continue;
            };
            matched += 1;
            let (dx, dy) = (
                run.start[0] * to_sheet - c.record.x,
                run.start[1] * to_sheet - c.record.y,
            );
            if dx.hypot(dy) > 0.5 * c.record.height.max(1.0) {
                far += 1;
            }
        }
        if far * 20 > matched {
            report.push(format!(
                "{} ({}): {far} of {matched} matched runs start away from the print",
                pdf.display(),
                sheet.layout
            ));
        }
    }
    assert!(compared > 0, "some PDF matches a layout: {report:?}");
    assert!(report.is_empty(), "{}", report.join("\n"));
    let _ = fs::remove_dir_all(dir);
}
