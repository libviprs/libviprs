//! A restarted polyline side keeps its pattern period and centres the pattern (#1699, replacing
//! the stretch of #1694).
//!
//! #1691 restarts a PLINEGEN-off polyline's pattern at every vertex. #1694 stretched each side's
//! pattern so the side ended on a dash, and a plotted side measures otherwise: the period stays the
//! linetype's own and the spare length is shared by the two end dashes. With the pattern's dash d,
//! its gap g and its period p = d + g, a side of length L holds n = floor((L + g) / p) whole
//! periods, the remainder r = L - (n p - g) is split in two, and the first and the last dash are
//! each d + r / 2 long, with every dash between them starting exactly p after the one before. The
//! numbers are the plotted ones, 3.125 mm and 1.5625 mm (a period of 4.6875 mm) on a side of
//! 93.6875 mm, where the plot's first dash is 3.87 mm.
//!
//! PVec's line style is one cyclic dash array and a start phase per record, and a phase can start
//! inside a dash but cannot lengthen it, so a side is written as three records that join at the two
//! end dashes: a solid lead of r / 2, the pattern at its own period from there to the last dash's
//! end, and a solid tail of r / 2. A side that is exactly n periods less a gap, a side shorter than
//! one pattern, a PLINEGEN-on polyline and a LINE are not touched. Fixtures are written with
//! acadrust. Needs `--features pvec-convert`.

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

/// HIDDEN's dash and gap in the fixtures at the plot's scale (mm), and so its period.
const DASH: f64 = 3.125;
const GAP: f64 = 1.5625;
const PERIOD: f64 = 4.6875;
/// A side of 20 periods less a gap plus a remainder of 1.5, so each end dash is 0.75 longer: the
/// first dash is 3.875 mm. Every number here is a multiple of 1/16, so the arithmetic is exact.
const PLOT_SIDE: f64 = 20.0 * PERIOD - GAP + 1.5;

/// One record: its two ends when it is a two-point polyline (zeros otherwise) and the dash array of
/// its style (empty for none).
type Record = ([f64; 2], [f64; 2], Vec<f64>);

fn convert(tag: &str, bytes: &[u8]) -> Vec<Record> {
    let dir = std::env::temp_dir().join(format!("libviprs-1694-{tag}-{}", std::process::id()));
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
    let dashes: BTreeMap<u64, Vec<f64>> = style
        .assignments
        .iter()
        .map(|(id, k)| {
            (
                u64::from(id.index()),
                style.styles[*k as usize].dashes.clone(),
            )
        })
        .collect();
    let out = reader
        .entities()
        .unwrap()
        .iter()
        .map(|(id, curve)| {
            let (a, b) = match curve {
                Curve::Polyline(p) if p.vertices().len() == 2 => {
                    let v = p.vertices();
                    ([v[0].x, v[0].y], [v[1].x, v[1].y])
                }
                // Anything else (a LINE, a run of several vertices) has no side of its own.
                _ => ([0.0; 2], [0.0; 2]),
            };
            (
                a,
                b,
                dashes
                    .get(&u64::from(id.index()))
                    .cloned()
                    .unwrap_or_default(),
            )
        })
        .collect();
    let _ = fs::remove_dir_all(dir);
    out
}

fn doc() -> CadDocument {
    let mut doc = CadDocument::new();
    doc.header.polyline_linetype_generation = false;
    let mut hidden = LineType::new("HIDDEN");
    hidden.elements = vec![LineTypeElement::dash(DASH), LineTypeElement::space(GAP)];
    doc.line_types.add(hidden).unwrap();
    doc
}

fn lw(points: &[[f64; 2]], closed: bool, plinegen: bool) -> EntityType {
    let mut p = LwPolyline::new();
    for q in points {
        p.add_point_with_bulge(Vector2::new(q[0], q[1]), 0.0);
    }
    p.is_closed = closed;
    p.plinegen = plinegen;
    p.common.linetype = "HIDDEN".into();
    EntityType::LwPolyline(p)
}

fn rect(w: f64, h: f64) -> [[f64; 2]; 4] {
    [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

fn straight(tag: &str, entity: EntityType) -> Vec<Record> {
    let mut d = doc();
    d.add_entity(entity).unwrap();
    convert(tag, &DwgWriter::write_to_vec(&d).unwrap())
}

fn length(r: &Record) -> f64 {
    (r.1[0] - r.0[0]).hypot(r.1[1] - r.0[1])
}

/// The records grouped into sides: consecutive records that run the same way along the same line
/// are one side (a corner turns, so the next side is a new group).
fn sides(records: &[Record]) -> Vec<&[Record]> {
    let dir = |r: &Record| {
        let l = length(r).max(1e-12);
        [
            ((r.1[0] - r.0[0]) / l).round(),
            ((r.1[1] - r.0[1]) / l).round(),
        ]
    };
    let mut out: Vec<&[Record]> = Vec::new();
    let mut from = 0;
    for k in 1..=records.len() {
        if k == records.len() || dir(&records[k]) != dir(&records[from]) {
            out.push(&records[from..k]);
            from = k;
        }
    }
    out
}

/// The dashes a side draws, as (start, end) distances along it from its first point, with the
/// dashes that touch (the end of a solid lead and the first dash of the pattern) joined. A record
/// with no dash array is solid, and one with an array repeats it from its start (every phase here
/// is zero).
fn drawn(side: &[Record]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut at = 0.0;
    for r in side {
        let l = length(r);
        if r.2.is_empty() {
            out.push((at, at + l));
        } else {
            let mut x = 0.0;
            'cycle: loop {
                for (k, e) in r.2.iter().enumerate() {
                    if x >= l - 1e-9 {
                        break 'cycle;
                    }
                    if k % 2 == 0 {
                        out.push((at + x, at + (x + e).min(l)));
                    }
                    x += e;
                }
            }
        }
        at += l;
    }
    let mut joined: Vec<(f64, f64)> = Vec::new();
    for d in out {
        match joined.last_mut() {
            Some(last) if d.0 - last.1 < 1e-9 => last.1 = d.1,
            _ => joined.push(d),
        }
    }
    joined
}

/// What a side of length `l` must draw by the rule alone: n periods, a remainder, and the two end
/// dashes each d + r / 2 long.
fn centred(l: f64) -> Vec<(f64, f64)> {
    let n = ((l + GAP) / PERIOD + 1e-9).floor();
    let half = (l - (n * PERIOD - GAP)) / 2.0;
    let mut want = vec![(0.0, DASH + half)];
    for k in 1..n as usize - 1 {
        let s = half + k as f64 * PERIOD;
        want.push((s, s + DASH));
    }
    want.push((l - DASH - half, l));
    want
}

fn same(got: &[(f64, f64)], want: &[(f64, f64)]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (g.0 - w.0).abs() < 1e-6 && (g.1 - w.1).abs() < 1e-6)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_the_plots_side_starts_on_a_dash_of_3_875_and_keeps_the_period() {
    let r = straight("plot", lw(&rect(PLOT_SIDE, PLOT_SIDE), true, false));
    let s = sides(&r);
    assert_eq!(s.len(), 4, "{r:?}");
    for side in s {
        let d = drawn(side);
        assert!((d[0].1 - d[0].0 - 3.875).abs() < 1e-9, "first dash: {d:?}");
        assert!((d[0].0).abs() < 1e-9, "it starts at the corner: {d:?}");
        // The second dash starts a gap after the first, then every period.
        for (k, w) in d.windows(2).enumerate() {
            assert!(
                (w[1].0 - w[0].0 - if k == 0 { 3.875 + GAP } else { PERIOD }).abs() < 1e-9,
                "dash {k} to the next: {d:?}"
            );
        }
        let last = d[d.len() - 1];
        assert!(
            (last.1 - PLOT_SIDE).abs() < 1e-9,
            "it ends at the corner: {d:?}"
        );
        assert!((last.1 - last.0 - 3.875).abs() < 1e-9, "last dash: {d:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_every_side_draws_what_the_rule_says() {
    // 20 x 10 mm: n = 4 and r = 2.8125 on the long sides, n = 2 and r = 2.1875 on the short ones.
    let r = straight("rule", lw(&rect(20.0, 10.0), true, false));
    let s = sides(&r);
    assert_eq!(s.len(), 4, "{r:?}");
    for side in s {
        let l: f64 = side.iter().map(length).sum();
        let (got, want) = (drawn(side), centred(l));
        assert!(same(&got, &want), "side {l}: drew {got:?}, want {want:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_the_pattern_is_never_stretched() {
    let r = straight("period", lw(&rect(PLOT_SIDE, 30.0), true, false));
    assert!(r.len() >= 4, "{r:?}");
    for rec in r.iter().filter(|rec| !rec.2.is_empty()) {
        assert_eq!(
            rec.2,
            vec![DASH, GAP],
            "the period is the linetype's own: {rec:?}"
        );
    }
    // And the end dashes, not the pattern, take the remainder: no dash is shorter than d.
    for side in sides(&r) {
        assert!(
            drawn(side).iter().all(|d| d.1 - d.0 > DASH - 1e-9),
            "{side:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_an_open_polyline_centres_every_side_too() {
    let pts = [[0.0, 0.0], [PLOT_SIDE, 0.0], [PLOT_SIDE, 30.0]];
    let r = straight("open", lw(&pts, false, false));
    let s = sides(&r);
    assert_eq!(s.len(), 2, "{r:?}");
    for side in s {
        let l: f64 = side.iter().map(length).sum();
        assert!(same(&drawn(side), &centred(l)), "side {l}: {side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_a_2d_polyline_is_centred_the_same_way() {
    let mut p = Polyline2D::new();
    for q in rect(PLOT_SIDE, PLOT_SIDE) {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    p.close();
    p.common.linetype = "HIDDEN".into();
    let r = straight("2d", EntityType::Polyline2D(p));
    let s = sides(&r);
    assert_eq!(s.len(), 4, "{r:?}");
    for side in s {
        assert!(same(&drawn(side), &centred(PLOT_SIDE)), "{side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_a_side_that_is_whole_periods_less_a_gap_is_left_alone() {
    let l = 20.0 * PERIOD - GAP;
    let r = straight("exact", lw(&rect(l, l), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert_eq!(side.2, vec![DASH, GAP], "{side:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_a_polyline_with_plinegen_on_keeps_its_pattern_as_drawn() {
    let r = straight("on", lw(&rect(PLOT_SIDE, PLOT_SIDE), true, true));
    // One continuous run, so no two-point sides and nothing centred.
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].2, vec![DASH, GAP], "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_a_line_keeps_its_pattern_as_drawn() {
    let mut l = Line::from_coords(0.0, 0.0, 0.0, PLOT_SIDE, 0.0, 0.0);
    l.common.linetype = "HIDDEN".into();
    let r = straight("line", EntityType::Line(l));
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].2, vec![DASH, GAP], "{r:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1699_a_side_shorter_than_one_pattern_is_left_alone() {
    // 3 mm holds no whole period (n = 0), so it keeps the pattern from its start.
    let r = straight("short", lw(&rect(3.0, 3.0), true, false));
    assert_eq!(r.len(), 4, "{r:?}");
    for side in &r {
        assert_eq!(side.2, vec![DASH, GAP], "{side:?}");
    }
}
