//! Issue #1658: DWG to PVec carries the text of DIMENSION, MULTILEADER and TOLERANCE.
//!
//! A DIMENSION draws itself from the anonymous block it names (the lines, the arrowheads and the
//! string AutoCAD printed, as an MTEXT), so the provider reports it as an INSERT of that block and
//! everything in it is converted like any other block content. A dimension whose block is missing
//! falls back to its measurement, formatted from its style. A MULTILEADER's text and a TOLERANCE's
//! feature control frame text go in the text stream the way TEXT and MTEXT do, behind the same
//! texts option. Each oracle is read out of the drawing with acadrust rather than written down
//! here, except where the point of the test is the string this crate builds (the tolerance's GD&T
//! symbols and the fallback measurement).
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use acadrust::entities::{Line, MultiLeader, TextAttachmentPointType};
use acadrust::tables::Layer;
use acadrust::{CadDocument, DwgReader, DwgWriter, EntityType, Vector3};
use libviprs::cad::{CadSource, display_text};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::{DwgProvider, EntityScope, InstancePath, LayoutKind, RawGeometry, SourceUnit};
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::Curve;
use pvec_facade::components::{HatchStream, TextStream};

/// Line style for the hidden anchors and boundaries; the text and hatch streams come with the
/// texts and fills options.
const STREAMS: StreamSelection = StreamSelection {
    line_style: true,
    ..StreamSelection::NONE
};

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(rel)
}

fn read(rel: &str) -> CadDocument {
    DwgReader::from_file(fixture(rel)).unwrap().read().unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1658-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// One text record, with the length of the polyline it is anchored on (the printed run).
struct Record {
    text: String,
    x: f64,
    y: f64,
    height: f64,
    rotation: f64,
    advance: f64,
}

struct Run {
    records: Vec<Record>,
    hatches: usize,
    report: ConversionReport,
    took: Duration,
}

/// Model space of `bytes`, converted in millimetres with the texts and fills choices given.
fn convert(tag: &str, bytes: &[u8], texts: bool, fills: bool) -> Run {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let start = Instant::now();
    let out = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(STREAMS)
        .texts(texts)
        .fills(fills)
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let took = start.elapsed();
    let reader = open_published(&dest).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let entities = reader.entities().unwrap();
    let records = streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)
        .map(|s| TextStream::decode(&s.payload).unwrap().records)
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            let advance = entities.iter().find(|(id, _)| *id == r.entity).map_or(
                0.0,
                |(_, curve)| match curve {
                    Curve::Polyline(p) => {
                        let v = p.vertices();
                        let (a, z) = (&v[0], &v[v.len() - 1]);
                        (z.x - a.x).hypot(z.y - a.y)
                    }
                    other => panic!("a text anchor is a polyline, not {other:?}"),
                },
            );
            Record {
                text: r.text,
                x: r.x,
                y: r.y,
                height: r.height,
                rotation: r.rotation,
                advance,
            }
        })
        .collect();
    let hatches = streams
        .iter()
        .find(|s| s.capability == HatchStream::CAPABILITY)
        .map_or(0, |s| {
            HatchStream::decode(&s.payload).unwrap().records.len()
        });
    let _ = fs::remove_dir_all(dir);
    Run {
        records,
        hatches,
        report: out.report,
        took,
    }
}

/// A fixture converted as [`convert`] does; `tag` keeps each test's scratch directory its own.
fn convert_fixture(tag: &str, rel: &str, texts: bool, fills: bool) -> Run {
    convert(tag, &fs::read(fixture(rel)).unwrap(), texts, fills)
}

fn texts(run: &Run) -> Vec<&str> {
    run.records.iter().map(|r| r.text.as_str()).collect()
}

fn count(run: &Run, key: &str) -> u64 {
    run.report
        .components
        .as_ref()
        .and_then(|c| c.annotations.get(key).copied())
        .unwrap_or(0)
}

/// The rows of `class`, and that there is at least one.
fn rows<'a>(run: &'a Run, class: &str) -> Vec<&'a libviprs::pvec::report::EntityRow> {
    let found: Vec<_> = run
        .report
        .entities
        .iter()
        .filter(|r| r.source_class == class)
        .collect();
    assert!(!found.is_empty(), "no {class} row");
    found
}

fn is_dimension(class: &str) -> bool {
    class.starts_with("DIMENSION") || class.ends_with("_DIMENSION")
}

/// What a person reads in a stored MTEXT string, one entry per printed line.
fn shown(raw: &str) -> Vec<String> {
    display_text(raw)
        .text
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The entities model space owns directly.
fn model(doc: &CadDocument) -> Vec<&EntityType> {
    let record = doc.block_records.get("*Model_Space").unwrap();
    record
        .entity_handles
        .iter()
        .filter_map(|h| doc.get_entity(*h))
        .collect()
}

/// The entities a block owns directly.
fn block<'a>(doc: &'a CadDocument, name: &str) -> Vec<&'a EntityType> {
    let record = doc.block_records.get(name).unwrap();
    record
        .entity_handles
        .iter()
        .filter_map(|h| doc.get_entity(*h))
        .collect()
}

/// A middle-centre text record of height `h` is centred on `at`: half its block (its run, or the
/// MTEXT's reference width when that is wider) along its baseline from where it starts lands
/// within a text height and a half of `at`, the baseline sitting a little under the middle.
fn centred_on(r: &Record, at: [f64; 2], h: f64, width: f64) -> bool {
    let half = r.advance.max(width) / 2.0;
    let (sin, cos) = r.rotation.sin_cos();
    (r.x + cos * half - at[0]).hypot(r.y + sin * half - at[1]) < 1.5 * h
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_every_dimension_block_text_is_a_text_record() {
    let doc = read("cad/sample_AC1018.dwg");
    // Every printed line of every MTEXT in each model space DIMENSION's block, where it sits, its
    // height and its reference width.
    let mut expected: Vec<(String, [f64; 2], f64, f64)> = Vec::new();
    let mut dimensions = 0;
    for e in model(&doc) {
        let EntityType::Dimension(d) = e else {
            continue;
        };
        dimensions += 1;
        for inner in block(&doc, &d.base().block_name) {
            if let EntityType::MText(m) = inner {
                for line in shown(&m.value) {
                    expected.push((
                        line,
                        [m.insertion_point.x, m.insertion_point.y],
                        m.height,
                        m.rectangle_width,
                    ));
                }
            }
        }
    }
    assert_eq!(dimensions, 11, "the sample's model space dimensions");
    assert_eq!(expected.len(), dimensions, "one printed line per dimension");

    let run = convert_fixture("dim-block-text", "cad/sample_AC1018.dwg", true, false);
    let mut used = vec![false; run.records.len()];
    for (text, at, h, width) in &expected {
        let hit = run
            .records
            .iter()
            .enumerate()
            .position(|(i, r)| !used[i] && r.text == *text && centred_on(r, *at, *h, *width));
        match hit {
            Some(i) => used[i] = true,
            None => panic!(
                "no record {text:?} centred on {at:?} (height {h}, width {width}): {:?}",
                run.records
                    .iter()
                    .filter(|r| r.text == *text)
                    .map(|r| (r.x, r.y, r.height, r.rotation, r.advance))
                    .collect::<Vec<_>>()
            ),
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_only_drawing_converts_with_its_lines() {
    let doc = read("cad/g13/g13_dimension.dwg");
    let dims: Vec<_> = model(&doc)
        .into_iter()
        .filter_map(|e| match e {
            EntityType::Dimension(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(dims.len(), 1);
    let inner = block(&doc, &dims[0].base().block_name);
    let lines: Vec<u64> = inner
        .iter()
        .filter(|e| matches!(e, EntityType::Line(_)))
        .map(|e| e.common().handle.value())
        .collect();
    let solids = inner
        .iter()
        .filter(|e| matches!(e, EntityType::Solid(_)))
        .count();
    assert_eq!(
        (lines.len(), solids),
        (3, 2),
        "the fixture's dimension block"
    );

    let run = convert_fixture("dim-only", "cad/g13/g13_dimension.dwg", true, true);
    for h in &lines {
        let row = run
            .report
            .entities
            .iter()
            .find(|r| r.source_handle == Some(*h))
            .unwrap_or_else(|| panic!("no row for the block's LINE {h:x}"));
        assert_eq!(row.source_class, "LINE");
        assert!(
            row.semantic_type.is_some(),
            "LINE {h:x} is geometry: {row:?}"
        );
    }
    assert_eq!(
        run.hatches, solids,
        "each arrowhead SOLID is a hatch record"
    );
    assert_eq!(texts(&run), ["10"]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_no_dimension_with_a_block_is_reported_unmapped() {
    let run = convert_fixture("dim-unmapped", "cad/sample_AC1018.dwg", true, false);
    let unmapped: Vec<_> = run
        .report
        .entities
        .iter()
        .filter(|r| is_dimension(&r.source_class))
        .filter(|r| r.reason.as_deref() == Some("unmapped-source-class"))
        .map(|r| (&r.source_class, r.source_handle))
        .collect();
    assert!(unmapped.is_empty(), "{unmapped:?}");
    // They have no rows of their own now, so the report counts them (#1658 B2).
    assert_eq!(count(&run, "DIMENSION.expanded"), 11);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_reached_through_a_dimension_is_not_expanded() {
    // 3000 dimensions, each the content of the one before's block.
    let run = convert_fixture("dim-deep", "cad/g13/g13_dimension_deep.dwg", true, false);
    assert!(run.took < Duration::from_secs(20), "took {:?}", run.took);
    assert_eq!(texts(&run), ["10"], "only the outer dimension is drawn");
    assert_eq!(count(&run, "DIMENSION.expanded"), 1);
    let dims: Vec<_> = run
        .report
        .entities
        .iter()
        .filter(|r| is_dimension(&r.source_class))
        .collect();
    assert_eq!(dims.len(), 1, "{dims:?}");
    assert_eq!(dims[0].reason.as_deref(), Some("unmapped-source-class"));
    assert!(
        dims[0]
            .warnings
            .iter()
            .any(|w| w.contains("a dimension inside a dimension's block is not expanded")),
        "{:?}",
        dims[0].warnings
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_without_its_block_carries_its_measurement() {
    let rel = "pvec-gold/fixtures/dimension-001/source.dwg";
    let doc = read(rel);
    let [EntityType::Dimension(d)] = model(&doc)[..] else {
        panic!("the fixture holds one dimension");
    };
    let b = d.base();
    assert!(
        doc.block_records.get(&b.block_name).is_none(),
        "its block is missing"
    );

    // Standard: two decimals with trailing zeros suppressed (DIMZIN 8), so 5.00 prints as 5.
    let run = convert_fixture("dim-no-block", rel, true, false);
    assert_eq!(texts(&run), ["5"]);
    let r = &run.records[0];
    assert!(
        (r.height - 0.18).abs() < 1e-9,
        "DIMTXT times DIMSCALE: {}",
        r.height
    );
    assert!(
        centred_on(
            r,
            [b.text_middle_point.x, b.text_middle_point.y],
            r.height,
            0.0
        ),
        "at ({}, {})",
        r.x,
        r.y
    );
    for row in rows(&run, "DIMENSION_LINEAR") {
        assert!(
            row.warnings
                .iter()
                .any(|w| w.contains("rebuilt from the measurement")),
            "{:?}",
            row.warnings
        );
    }
}

/// Whether a dimension carries style overrides of its own (the ACAD application's DSTYLE list),
/// which the fallback does not read.
fn overridden(e: &EntityType) -> bool {
    e.common()
        .extended_data
        .get_record("ACAD")
        .is_some_and(|r| {
            r.values
                .iter()
                .any(|v| matches!(v, acadrust::xdata::XDataValue::String(s) if s == "DSTYLE"))
        })
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_without_its_block_prints_what_its_block_would_have() {
    let mut doc = read("cad/sample_AC1018.dwg");
    // What each dimension's block prints, and where the dimension says its text goes.
    let mut expected: Vec<(String, [f64; 2], f64)> = Vec::new();
    let mut handles = Vec::new();
    let mut skipped = 0;
    for e in model(&doc) {
        let EntityType::Dimension(d) = e else {
            continue;
        };
        handles.push(e.common().handle);
        if overridden(e) {
            skipped += 1;
            continue;
        }
        let b = d.base();
        let style = doc.dim_styles.get(&b.style_name).unwrap();
        let height = style.dimtxt
            * if style.dimscale > 0.0 {
                style.dimscale
            } else {
                1.0
            };
        for inner in block(&doc, &b.block_name) {
            if let EntityType::MText(m) = inner {
                for line in shown(&m.value) {
                    expected.push((line, [b.text_middle_point.x, b.text_middle_point.y], height));
                }
            }
        }
    }
    assert!(expected.len() >= 8, "{expected:?}");
    // Every dimension loses its block.
    for h in handles {
        if let Some(EntityType::Dimension(d)) = doc.get_entity_mut(h) {
            d.base_mut().block_name = "*MISSING".into();
        }
    }
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let run = convert("no-blocks", &bytes, true, false);
    let mut used = vec![false; run.records.len()];
    for (text, at, h) in &expected {
        let hit = run.records.iter().enumerate().position(|(i, r)| {
            !used[i]
                && r.text == *text
                && (r.height - h).abs() < 1e-9
                && centred_on(r, *at, *h, 0.0)
        });
        match hit {
            Some(i) => used[i] = true,
            None => panic!("no record {text:?} at {at:?} in {:?}", texts(&run)),
        }
    }
    let rebuilt = run
        .report
        .entities
        .iter()
        .filter(|r| is_dimension(&r.source_class))
        .filter(|r| {
            r.warnings
                .iter()
                .any(|w| w.contains("rebuilt from the measurement"))
        })
        .count();
    assert_eq!(
        rebuilt,
        expected.len(),
        "{skipped} skipped for their overrides"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_multileader_text_is_a_text_record() {
    let doc = read("cad/sample_AC1018.dwg");
    let mut expected: std::collections::BTreeMap<String, usize> = Default::default();
    let mut with_text = 0;
    for e in model(&doc) {
        if let EntityType::MultiLeader(m) = e
            && m.context.has_text_contents
            && !m.context.text_string.is_empty()
        {
            with_text += 1;
            for line in shown(&m.context.text_string) {
                *expected.entry(line).or_default() += 1;
            }
        }
    }
    assert_eq!(with_text, 15, "the sample's multileaders with text");

    let run = convert_fixture("mleader-text", "cad/sample_AC1018.dwg", true, false);
    for (line, n) in &expected {
        let got = run.records.iter().filter(|r| &r.text == line).count();
        assert_eq!(got, *n, "records {line:?} in {:?}", texts(&run));
    }
    assert_eq!(count(&run, "MULTILEADER.mapped"), with_text as u64);
    for row in rows(&run, "MULTILEADER") {
        // The lines are drawn too since #1671, so the row no longer says they are left out.
        assert!(
            !row.warnings
                .iter()
                .any(|w| w.contains("leader lines are not drawn")),
            "{:?}",
            row.warnings
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_multileader_text_sits_at_its_attachment_point() {
    let mut doc = CadDocument::new();
    doc.add_entity(EntityType::Line(Line::from_points(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(200.0, 0.0, 0.0),
    )))
    .unwrap();
    for (text, x, at) in [
        ("Left", 10.0, TextAttachmentPointType::Left),
        ("Middle", 60.0, TextAttachmentPointType::Center),
        ("Right", 120.0, TextAttachmentPointType::Right),
    ] {
        let mut m = MultiLeader::with_text(
            text,
            Vector3::new(x, 20.0, 0.0),
            vec![Vector3::new(x - 5.0, 10.0, 0.0)],
        );
        m.context.text_height = 2.5;
        m.context.text_attachment_point = at;
        m.context.text_direction = Vector3::new(1.0, 0.0, 0.0);
        m.context.text_normal = Vector3::new(0.0, 0.0, 1.0);
        m.text_attachment_point = at;
        doc.add_entity(EntityType::MultiLeader(Box::new(m)))
            .unwrap();
    }
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let run = convert("mleader-attach", &bytes, true, false);
    let one = |t: &str| {
        let found: Vec<_> = run.records.iter().filter(|r| r.text == t).collect();
        assert_eq!(found.len(), 1, "one {t:?} in {:?}", texts(&run));
        found[0]
    };
    // The text location is the top of the text, at its left, centre or right.
    let (left, middle, right) = (one("Left"), one("Middle"), one("Right"));
    for r in [left, middle, right] {
        assert!((r.height - 2.5).abs() < 1e-9, "{}", r.height);
        assert!(
            (r.y - (20.0 - 2.5)).abs() < 1e-6,
            "{} baseline {}",
            r.text,
            r.y
        );
    }
    assert!((left.x - 10.0).abs() < 1e-6, "{}", left.x);
    assert!(
        (middle.x + middle.advance / 2.0 - 60.0).abs() < 1e-6,
        "{}",
        middle.x
    );
    assert!(
        (right.x + right.advance - 120.0).abs() < 1e-6,
        "{}",
        right.x
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_tolerance_text_is_a_text_record_without_codes() {
    let doc = read("cad/g13/g13_tolerance.dwg");
    let tolerances = doc
        .entities()
        .filter(|e| matches!(e, EntityType::Tolerance(_)))
        .count();
    assert!(tolerances >= 3, "the fixture's tolerances");

    let run = convert_fixture("tolerance", "cad/g13/g13_tolerance.dwg", true, false);
    // Two in model space and one through an INSERT, each a two row frame: position, diameter 0.25
    // at maximum material condition to datum A, then perpendicular within 0.5 to datum B.
    let mut got = texts(&run);
    got.sort_unstable();
    assert_eq!(
        got,
        [
            "⊥ | 0.5 | B",
            "⊥ | 0.5 | B",
            "⊥ | 0.5 | B",
            "⌖ | ⌀0.25Ⓜ | A",
            "⌖ | ⌀0.25Ⓜ | A",
            "⌖ | ⌀0.25Ⓜ | A",
        ]
    );
    for r in &run.records {
        assert!(
            !r.text.contains("%%") && !r.text.contains('\\') && !r.text.contains('{'),
            "a code is left in {:?}",
            r.text
        );
    }
    // The insertion point is the left end of the frame and the middle of its rows, at (4, 6) for
    // the one model space draws directly (the others go through scaled INSERTs).
    assert!(
        run.records.iter().any(|r| r.text.starts_with('⌖')
            && (r.x - 4.0).abs() < 1e-6
            && (r.y - 6.0).abs() < r.height),
        "{:?}",
        run.records.iter().map(|r| (r.x, r.y)).collect::<Vec<_>>()
    );
    // The one with a tilted normal is refused as any text off the Z axis is; the carried ones say
    // what they leave out.
    assert_eq!(count(&run, "TOLERANCE.mapped"), 3);
    let carried: Vec<_> = rows(&run, "TOLERANCE")
        .into_iter()
        .filter(|r| {
            r.warnings
                .iter()
                .any(|w| w.contains("carried in a PVec stream"))
        })
        .collect();
    assert_eq!(carried.len(), run.records.len(), "a row per printed row");
    // The frame is drawn too since #1671, so the rows no longer say it is left out.
    for row in carried {
        assert!(
            !row.warnings
                .iter()
                .any(|w| w.contains("its frame is not drawn")),
            "{:?}",
            row.warnings
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_with_texts_off_they_stay_unmapped_and_say_so() {
    // Fills on, texts off: with neither choice nothing is annotated at all (#1624).
    let run = convert_fixture("texts-off", "cad/sample_AC1018.dwg", false, true);
    assert!(run.records.is_empty(), "{:?}", texts(&run));
    for class in ["MULTILEADER", "TOLERANCE"] {
        // The entity's own rows: its lines (#1671) are geometry and are drawn either way.
        for row in rows(&run, class)
            .into_iter()
            .filter(|r| r.semantic_type.is_none())
        {
            assert_eq!(row.reason.as_deref(), Some("unmapped-source-class"));
            assert!(
                row.warnings.iter().any(|w| w.contains("texts are off")),
                "{class}: {:?}",
                row.warnings
            );
        }
    }
}

/// g13_dimension with its dimension moved onto a new layer, frozen or not, and a LINE of its own
/// so the drawing has something to write either way.
fn dimension_on_its_own_layer(frozen: bool) -> Vec<u8> {
    let mut doc = read("cad/g13/g13_dimension.dwg");
    let mut layer = Layer::new("DIMS");
    layer.flags.frozen = frozen;
    doc.layers.add(layer).unwrap();
    let handle = model(&doc)
        .into_iter()
        .find(|e| matches!(e, EntityType::Dimension(_)))
        .map(|e| e.common().handle)
        .unwrap();
    doc.get_entity_mut(handle).unwrap().common_mut().layer = "DIMS".into();
    doc.add_entity(EntityType::Line(Line::from_points(
        Vector3::new(0.0, -10.0, 0.0),
        Vector3::new(10.0, -10.0, 0.0),
    )))
    .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_on_a_frozen_layer_is_dropped_whole() {
    let thawed = convert(
        "dim-thawed",
        &dimension_on_its_own_layer(false),
        true,
        false,
    );
    assert_eq!(texts(&thawed), ["10"], "on a thawed layer it is drawn");
    let lines = |run: &Run| rows(run, "LINE").len();
    assert!(lines(&thawed) > 1, "its lines are drawn with the extra one");

    let frozen = convert("dim-frozen", &dimension_on_its_own_layer(true), true, false);
    assert!(frozen.records.is_empty(), "{:?}", texts(&frozen));
    let drawn: Vec<_> = frozen
        .report
        .entities
        .iter()
        .filter(|r| r.source_class == "LINE" && r.semantic_type.is_some())
        .collect();
    assert_eq!(drawn.len(), 1, "only the extra LINE is drawn: {drawn:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_the_cursor_reports_a_dimension_as_an_insert_of_its_block() {
    let open = |rel: &str| {
        let bytes = fs::read(fixture(rel)).unwrap();
        AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap()
    };
    let model_scope = |p: &AcadrustProvider| {
        let id = p
            .layouts()
            .unwrap()
            .into_iter()
            .find(|l| l.kind == LayoutKind::Model)
            .unwrap()
            .id;
        EntityScope::Layout(id)
    };
    let dimension = |p: &AcadrustProvider, scope: &EntityScope| {
        let mut found = p
            .entities(scope)
            .unwrap()
            .collect_all()
            .unwrap()
            .into_iter()
            .filter(|e| is_dimension(&e.provenance.class_name))
            .collect::<Vec<_>>();
        assert_eq!(found.len(), 1);
        found.remove(0)
    };

    // With its block: an INSERT of it, one instance, nothing moved, and still named a dimension.
    let p = open("cad/g13/g13_dimension.dwg");
    let d = dimension(&p, &model_scope(&p));
    assert_eq!(d.provenance.class_name, "DIMENSION_LINEAR");
    let RawGeometry::Insert(b) = &d.geometry else {
        panic!("{:?}", d.geometry);
    };
    assert_eq!(b.block_name, "*D0");
    assert_eq!(b.transform.coefficients(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!((b.grid.rows, b.grid.columns), (1, 1));
    assert_eq!((b.z_scale, b.elevation), (1.0, 0.0));
    assert_eq!(b.normal, [0.0, 0.0, 1.0]);

    // Inside a dimension's block, another dimension is not followed.
    let p = open("cad/g13/g13_dimension_deep.dwg");
    let outer = dimension(&p, &model_scope(&p));
    let inner_scope = EntityScope::BlockDefinition {
        name: "*D0".into(),
        reached_via: InstancePath::root().child("*D0", outer.provenance.handle.unwrap()),
    };
    let inner = dimension(&p, &inner_scope);
    match &inner.geometry {
        RawGeometry::Unmapped { reason } => assert!(
            reason.contains("a dimension inside a dimension's block is not expanded"),
            "{reason}"
        ),
        other => panic!("{other:?}"),
    }

    // Without its block: unmapped, and the reason names the block.
    let p = open("pvec-gold/fixtures/dimension-001/source.dwg");
    let d = dimension(&p, &model_scope(&p));
    match &d.geometry {
        RawGeometry::Unmapped { reason } => assert!(reason.contains("*U0"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

/// g13_dimension written back with acadrust after `edit` changes its dimension.
fn edited_dimension(edit: impl Fn(&mut acadrust::entities::DimensionBase)) -> Vec<u8> {
    let mut doc = read("cad/g13/g13_dimension.dwg");
    let handle = model(&doc)
        .into_iter()
        .find(|e| matches!(e, EntityType::Dimension(_)))
        .map(|e| e.common().handle)
        .unwrap();
    if let Some(EntityType::Dimension(d)) = doc.get_entity_mut(handle) {
        edit(d.base_mut());
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1658_a_dimension_off_its_default_placement_draws_where_it_is_placed() {
    // Where the middle of the one "10" run lands.
    let middle = |run: &Run| {
        let found: Vec<_> = run.records.iter().filter(|r| r.text == "10").collect();
        assert_eq!(found.len(), 1, "{:?}", texts(run));
        let (sin, cos) = found[0].rotation.sin_cos();
        let half = found[0].advance / 2.0;
        [found[0].x + cos * half, found[0].y + sin * half]
    };
    let plain = middle(&convert(
        "dim-plain",
        &edited_dimension(|_| {}),
        true,
        false,
    ));
    // The insertion point carries the whole block with it, as an INSERT's does.
    let moved = middle(&convert(
        "dim-moved",
        &edited_dimension(|b| b.insertion_point = Vector3::new(100.0, 50.0, 0.0)),
        true,
        false,
    ));
    assert!(
        (moved[0] - plain[0] - 100.0).abs() < 1e-6 && (moved[1] - plain[1] - 50.0).abs() < 1e-6,
        "{plain:?} moved to {moved:?}"
    );
    // A -Z normal mirrors the block across the Y axis, as it does an INSERT's.
    let flipped = middle(&convert(
        "dim-flipped",
        &edited_dimension(|b| b.normal = Vector3::new(0.0, 0.0, -1.0)),
        true,
        false,
    ));
    assert!(
        (flipped[0] + plain[0]).abs() < 1e-6 && (flipped[1] - plain[1]).abs() < 1e-6,
        "{plain:?} flipped to {flipped:?}"
    );
}
