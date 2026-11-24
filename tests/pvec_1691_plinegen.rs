//! A polyline's dash pattern restarts at every vertex when PLINEGEN is off (#1691).
//!
//! AutoCAD plots a polyline with PLINEGEN off by starting its linetype anew on every segment, so
//! each side begins and ends with a dash and the corners are solid. With PLINEGEN on one pattern
//! runs through the whole polyline. libviprs never read the flag, so a dashed frame left a gap at
//! its corners. The entity's own flag (LWPOLYLINE bit 128, 2D POLYLINE bit 128) is what the plot
//! follows, the header's PLINEGEN only seeds new polylines. Each side of a restarted polyline is
//! its own record with the entity's style and provenance, so the pattern restarts on it. The
//! drawings are written with acadrust. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::collections::BTreeMap;
use std::fs;

use acadrust::entities::{Line, LwPolyline, Polyline2D, Vertex2D};
use acadrust::tables::{LineType, LineTypeElement};
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::Curve;
use pvec_facade::components::LineStyleStream;

const STREAMS: StreamSelection = StreamSelection {
    line_style: true,
    ..StreamSelection::NONE
};

/// What one record draws: its linetype's name (empty for none) and its vertices when it is a
/// polyline, or nothing when it is a curve of another kind (an arc).
type Record = (String, Option<Vec<[f64; 2]>>);

fn convert(tag: &str, bytes: &[u8]) -> Vec<Record> {
    let dir = std::env::temp_dir().join(format!("libviprs-1691-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .components(STREAMS)
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the conversion failed: {e}"));
    let reader = open_published(&dest).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let style = LineStyleStream::decode(
        &streams
            .iter()
            .find(|s| s.capability == LineStyleStream::CAPABILITY)
            .expect("the line style stream")
            .payload,
    )
    .unwrap();
    let names: BTreeMap<u64, String> = style
        .assignments
        .iter()
        .map(|(id, k)| {
            (
                u64::from(id.index()),
                style.styles[*k as usize].name.clone(),
            )
        })
        .collect();
    let out = reader
        .entities()
        .unwrap()
        .iter()
        .map(|(id, curve)| {
            let vertices = match curve {
                Curve::Polyline(p) => Some(p.vertices().iter().map(|v| [v.x, v.y]).collect()),
                _ => None,
            };
            (
                names
                    .get(&u64::from(id.index()))
                    .cloned()
                    .unwrap_or_default(),
                vertices,
            )
        })
        .collect();
    let _ = fs::remove_dir_all(dir);
    out
}

fn doc(header_plinegen: bool) -> CadDocument {
    let mut doc = CadDocument::new();
    doc.header.polyline_linetype_generation = header_plinegen;
    let mut hidden = LineType::new("HIDDEN");
    hidden.elements = vec![LineTypeElement::dash(0.25), LineTypeElement::space(0.125)];
    doc.line_types.add(hidden).unwrap();
    doc
}

fn write(doc: &CadDocument) -> Vec<u8> {
    DwgWriter::write_to_vec(doc).unwrap()
}

const SQUARE: [[f64; 2]; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

fn lw(
    points: &[[f64; 2]],
    bulges: &[f64],
    closed: bool,
    plinegen: bool,
    linetype: &str,
) -> EntityType {
    let mut p = LwPolyline::new();
    for (i, q) in points.iter().enumerate() {
        p.add_point_with_bulge(
            Vector2::new(q[0], q[1]),
            bulges.get(i).copied().unwrap_or(0.0),
        );
    }
    p.is_closed = closed;
    p.plinegen = plinegen;
    p.common.linetype = linetype.into();
    EntityType::LwPolyline(p)
}

fn pl2d(points: &[[f64; 2]], closed: bool, plinegen: bool) -> EntityType {
    let mut p = Polyline2D::new();
    for q in points {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    if closed {
        p.close();
    }
    if plinegen {
        p.flags |= acadrust::entities::PolylineFlags::LINETYPE_CONTINUOUS;
    }
    p.common.linetype = "HIDDEN".into();
    EntityType::Polyline2D(p)
}

/// The hidden-line records of a drawing holding just `entity` (and a plain line so the
/// drawing is never empty).
fn hidden(tag: &str, header: bool, entity: EntityType) -> Vec<Record> {
    let mut d = doc(header);
    d.add_entity(entity).unwrap();
    d.add_entity(EntityType::Line(Line::from_coords(
        50.0, 50.0, 0.0, 60.0, 50.0, 0.0,
    )))
    .unwrap();
    convert(tag, &write(&d))
        .into_iter()
        .filter(|(n, _)| n == "HIDDEN")
        .collect()
}

fn sides(records: &[Record]) -> Vec<Vec<[f64; 2]>> {
    records.iter().filter_map(|(_, v)| v.clone()).collect()
}

fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
}

/// Whether some record is exactly the side from `a` to `b`.
fn has_side(records: &[Record], a: [f64; 2], b: [f64; 2]) -> bool {
    sides(records)
        .iter()
        .any(|v| v.len() == 2 && near(v[0], a) && near(v[1], b))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_a_closed_polyline_with_plinegen_off_is_one_record_per_side() {
    let r = hidden("closed-off", false, lw(&SQUARE, &[], true, false, "HIDDEN"));
    assert_eq!(r.len(), 4, "{r:?}");
    for k in 0..4 {
        assert!(
            has_side(&r, SQUARE[k], SQUARE[(k + 1) % 4]),
            "side {k} is missing from {r:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_the_same_polyline_with_plinegen_on_keeps_its_one_continuous_pattern() {
    let r = hidden("closed-on", false, lw(&SQUARE, &[], true, true, "HIDDEN"));
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(sides(&r)[0].len(), 5, "the closed run keeps all its points");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_an_open_polyline_with_plinegen_off_restarts_at_each_inner_vertex() {
    let pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
    let r = hidden("open-off", false, lw(&pts, &[], false, false, "HIDDEN"));
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(
        has_side(&r, pts[0], pts[1]) && has_side(&r, pts[1], pts[2]),
        "{r:?}"
    );
    let on = hidden("open-on", false, lw(&pts, &[], false, true, "HIDDEN"));
    assert_eq!(on.len(), 1, "{on:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_an_arc_segment_restarts_the_pattern_too() {
    // The second side is a half circle (bulge 1), the other three are straight.
    let bulges = [0.0, 1.0, 0.0, 0.0];
    let off = hidden(
        "arc-off",
        false,
        lw(&SQUARE, &bulges, true, false, "HIDDEN"),
    );
    assert_eq!(off.len(), 4, "{off:?}");
    assert_eq!(
        off.iter().filter(|(_, v)| v.is_none()).count(),
        1,
        "exactly the arc is not a straight piece: {off:?}"
    );
    assert!(has_side(&off, SQUARE[0], SQUARE[1]) && has_side(&off, SQUARE[2], SQUARE[3]));
    // The writer does not take a composite curve yet, so with PLINEGEN on the polyline is not
    // written at all today. Either way it must never be cut into pieces.
    let on = hidden("arc-on", false, lw(&SQUARE, &bulges, true, true, "HIDDEN"));
    assert!(on.len() <= 1, "{on:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_a_2d_polyline_follows_its_own_flag() {
    let off = hidden("2d-off", false, pl2d(&SQUARE, true, false));
    assert_eq!(off.len(), 4, "{off:?}");
    let on = hidden("2d-on", false, pl2d(&SQUARE, true, true));
    assert_eq!(on.len(), 1, "{on:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_the_entitys_flag_wins_over_the_headers() {
    // The header only seeds polylines drawn later, so a drawing with it on can hold a polyline
    // that plots restarted, and one with it off a polyline that plots continuous.
    let off = hidden(
        "hdr-on-ent-off",
        true,
        lw(&SQUARE, &[], true, false, "HIDDEN"),
    );
    assert_eq!(off.len(), 4, "{off:?}");
    let on = hidden(
        "hdr-off-ent-on",
        false,
        lw(&SQUARE, &[], true, true, "HIDDEN"),
    );
    assert_eq!(on.len(), 1, "{on:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_a_solid_polyline_stays_one_record_whatever_its_flag() {
    // No pattern, nothing to restart, and one record keeps the entity whole.
    let mut d = doc(false);
    d.add_entity(lw(&SQUARE, &[], true, false, "Continuous"))
        .unwrap();
    let all = convert("solid", &write(&d));
    assert_eq!(all.len(), 1, "{all:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1691_every_side_keeps_the_entitys_linetype_and_a_line_is_untouched() {
    let r = hidden("style", false, lw(&SQUARE, &[], true, false, "HIDDEN"));
    assert!(r.iter().all(|(n, _)| n == "HIDDEN"), "{r:?}");
    // The plain LINE beside it is one record, whatever PLINEGEN says.
    let mut d = doc(false);
    d.add_entity(EntityType::Line({
        let mut l = Line::from_coords(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
        l.common.linetype = "HIDDEN".into();
        l
    }))
    .unwrap();
    assert_eq!(convert("line", &write(&d)).len(), 1);
}
