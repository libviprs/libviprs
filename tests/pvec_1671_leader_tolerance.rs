//! Issue #1671: DWG to PVec draws the leader lines of LEADER and MULTILEADER entities, with their
//! arrowheads, and the frame of a TOLERANCE, next to the text #1658 carries.
//!
//! One small drawing holds one of each (plus a LINE so there is always something to write), built
//! with acadrust. A leader line is an open polyline through the stored points (a MULTILEADER's
//! ends at the root's connection point, where the text attaches), its arrowhead a small closed
//! triangle at the first point, and a TOLERANCE's frame is one closed rectangle per row with a
//! divider between its cells. None of it depends on the texts option: a line is geometry.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::{Leader, Line, MultiLeader, Tolerance};
use acadrust::tables::Layer;
use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::Curve;
use pvec_facade::components::TextStream;

const STREAMS: StreamSelection = StreamSelection {
    line_style: true,
    ..StreamSelection::NONE
};

/// One polyline as its points, with the first point repeated at the end when it is closed.
type Run = Vec<[f64; 2]>;

struct Converted {
    runs: Vec<(String, Run)>,
    texts: Vec<String>,
    report: ConversionReport,
}

impl Converted {
    /// The visible polylines of `class`: not the hidden anchors a text rides on.
    fn of(&self, class: &str) -> Vec<&Run> {
        self.runs
            .iter()
            .filter(|(c, _)| c == class)
            .map(|(_, r)| r)
            .collect()
    }
}

fn convert(tag: &str, bytes: &[u8], texts: bool) -> Converted {
    let dir = std::env::temp_dir().join(format!("libviprs-1671-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest: PathBuf = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(STREAMS)
        .texts(texts)
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let reader = open_published(&dest).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let entities = reader.entities().unwrap();
    let mut runs = Vec::new();
    for row in &out.report.entities {
        let carried = row
            .warnings
            .iter()
            .any(|w| w.contains("carried in a PVec stream"));
        if row.semantic_type.is_none() || carried {
            continue;
        }
        let Some((_, Curve::Polyline(p))) = entities
            .iter()
            .find(|(id, _)| u64::from(id.index()) == row.id)
        else {
            continue;
        };
        let run = p.vertices().iter().map(|v| [v.x, v.y]).collect();
        runs.push((row.source_class.clone(), run));
    }
    let texts = streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)
        .map(|s| TextStream::decode(&s.payload).unwrap().records)
        .unwrap_or_default()
        .into_iter()
        .map(|r| r.text)
        .collect();
    let _ = fs::remove_dir_all(dir);
    Converted {
        runs,
        texts,
        report: out.report,
    }
}

/// A LEADER, a MULTILEADER and a TOLERANCE, each on layer `layer`, and a LINE of its own.
fn drawing(layer: &str) -> Vec<u8> {
    let mut doc = CadDocument::new();
    if layer != "0" {
        let mut l = Layer::new(layer);
        l.flags.frozen = true;
        doc.layers.add(l).unwrap();
    }
    doc.add_entity(EntityType::Line(Line::from_points(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(200.0, 0.0, 0.0),
    )))
    .unwrap();

    let mut leader = Leader::from_vertices(vec![
        Vector3::new(10.0, 10.0, 0.0),
        Vector3::new(30.0, 20.0, 0.0),
        Vector3::new(40.0, 20.0, 0.0),
    ]);
    leader.arrow_enabled = true;
    leader.arrow_size = 2.0;
    leader.common.layer = layer.into();
    doc.add_entity(EntityType::Leader(leader)).unwrap();

    let mut m = MultiLeader::with_text(
        "Note",
        Vector3::new(80.0, 30.0, 0.0),
        vec![Vector3::new(60.0, 10.0, 0.0), Vector3::new(70.0, 20.0, 0.0)],
    );
    m.context.text_height = 2.5;
    m.context.text_direction = Vector3::new(1.0, 0.0, 0.0);
    m.context.text_normal = Vector3::new(0.0, 0.0, 1.0);
    m.arrowhead_size = 2.0;
    m.context.arrowhead_size = 2.0;
    m.context.leader_roots[0].lines[0].arrowhead_size = 2.0;
    m.common.layer = layer.into();
    doc.add_entity(EntityType::MultiLeader(Box::new(m)))
        .unwrap();

    // Two rows of three cells, 2.5 high text and a 0.625 gap: each row is 3.75 high.
    let mut t = Tolerance::with_text(
        Vector3::new(100.0, 50.0, 0.0),
        "{\\Fgdt;j}%%v0.5%%vA\\P{\\Fgdt;b}%%v0.1%%vB",
    );
    t.text_height = 2.5;
    t.dimension_gap = 0.625;
    t.common.layer = layer.into();
    doc.add_entity(EntityType::Tolerance(t)).unwrap();

    DwgWriter::write_to_vec(&doc).unwrap()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

fn has_point(run: &Run, at: [f64; 2]) -> bool {
    run.iter().any(|p| near(p[0], at[0]) && near(p[1], at[1]))
}

/// The run that is exactly `points`.
fn through<'a>(runs: &[&'a Run], points: &[[f64; 2]]) -> Option<&'a Run> {
    runs.iter().copied().find(|r| {
        r.len() == points.len()
            && r.iter()
                .zip(points)
                .all(|(a, b)| near(a[0], b[0]) && near(a[1], b[1]))
    })
}

/// The small closed shape at `tip`: an arrowhead, every point within `size` of the tip.
fn arrowhead_at(runs: &[&Run], tip: [f64; 2], size: f64) -> bool {
    runs.iter().any(|r| {
        r.len() >= 3
            && r.first() == r.last()
            && has_point(r, tip)
            && r.iter()
                .all(|p| (p[0] - tip[0]).hypot(p[1] - tip[1]) <= size + 1e-6)
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1671_a_leader_is_drawn_as_its_line_with_an_arrowhead() {
    let run = convert("leader", &drawing("0"), true);
    let runs = run.of("LEADER");
    assert!(
        through(&runs, &[[10.0, 10.0], [30.0, 20.0], [40.0, 20.0]]).is_some(),
        "the leader's line is not drawn: {runs:?}"
    );
    assert!(
        arrowhead_at(&runs, [10.0, 10.0], 2.0),
        "no arrowhead at the first vertex: {runs:?}"
    );
    let row = run
        .report
        .entities
        .iter()
        .find(|r| r.source_class == "LEADER" && r.semantic_type.is_some())
        .expect("a LEADER row that produced geometry");
    assert_ne!(row.reason.as_deref(), Some("unmapped-source-class"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1671_a_multileader_is_drawn_as_its_line_to_the_text_with_an_arrowhead() {
    let run = convert("mleader", &drawing("0"), true);
    let runs = run.of("MULTILEADER");
    assert!(
        through(&runs, &[[60.0, 10.0], [70.0, 20.0], [80.0, 30.0]]).is_some(),
        "the leader line does not run from the arrow tip to the text: {runs:?}"
    );
    assert!(
        arrowhead_at(&runs, [60.0, 10.0], 2.0),
        "no arrowhead at the arrow tip: {runs:?}"
    );
    // The text is still carried, and its row no longer says the lines are left out.
    assert!(run.texts.iter().any(|t| t == "Note"), "{:?}", run.texts);
    for row in run
        .report
        .entities
        .iter()
        .filter(|r| r.source_class == "MULTILEADER")
    {
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
fn issue_1671_a_tolerance_is_drawn_as_its_frame_one_rectangle_per_row() {
    let run = convert("tolerance", &drawing("0"), true);
    assert_eq!(
        run.texts.iter().filter(|t| t.contains(" | ")).count(),
        2,
        "{:?}",
        run.texts
    );
    let runs = run.of("TOLERANCE");
    // The insertion point is the left end of the frame and the middle of its rows: two rows of
    // 3.75 stand from 46.25 to 53.75 about y = 50.
    let rect = |lo: f64, hi: f64| {
        runs.iter().copied().find(|r| {
            r.len() == 5
                && r.first() == r.last()
                && r.iter().all(|p| near(p[1], lo) || near(p[1], hi))
                && r.iter().any(|p| near(p[1], lo))
                && r.iter().any(|p| near(p[1], hi))
                && r.iter().any(|p| near(p[0], 100.0))
        })
    };
    let (top, bottom) = (rect(50.0, 53.75), rect(46.25, 50.0));
    let (top, bottom) = (
        top.unwrap_or_else(|| panic!("no top row rectangle: {runs:?}")),
        bottom.unwrap_or_else(|| panic!("no bottom row rectangle: {runs:?}")),
    );
    for r in [top, bottom] {
        let right = r.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
        assert!(right > 100.0 + 3.75, "the frame has a width: {r:?}");
    }
    // Three cells a row, so two dividers a row, each as tall as its row and inside the rectangle.
    let dividers = |lo: f64, hi: f64, r: &Run| {
        let left = r.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
        let right = r.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
        runs.iter()
            .filter(|d| {
                d.len() == 2
                    && near(d[0][0], d[1][0])
                    && d[0][0] > left + 1e-6
                    && d[0][0] < right - 1e-6
                    && near(d[0][1].min(d[1][1]), lo)
                    && near(d[0][1].max(d[1][1]), hi)
            })
            .count()
    };
    assert_eq!(dividers(50.0, 53.75, top), 2, "{runs:?}");
    assert_eq!(dividers(46.25, 50.0, bottom), 2, "{runs:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1671_the_lines_do_not_need_the_texts_option() {
    let run = convert("texts-off", &drawing("0"), false);
    assert!(run.texts.is_empty(), "{:?}", run.texts);
    assert!(!run.of("LEADER").is_empty(), "no LEADER line");
    assert!(!run.of("MULTILEADER").is_empty(), "no MULTILEADER line");
    assert!(!run.of("TOLERANCE").is_empty(), "no TOLERANCE frame");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1671_what_is_on_a_frozen_layer_loses_its_lines_with_it() {
    let run = convert("frozen", &drawing("HIDDEN"), true);
    for class in ["LEADER", "MULTILEADER", "TOLERANCE"] {
        assert!(run.of(class).is_empty(), "{class} drew {:?}", run.of(class));
    }
    assert!(run.texts.is_empty(), "{:?}", run.texts);
    assert!(
        run.report
            .entities
            .iter()
            .any(|r| r.source_class == "LINE" && r.semantic_type.is_some()),
        "the LINE is still drawn"
    );
}
