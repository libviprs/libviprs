//! `paper-space-multi-viewport-001` through the print profile (#1579, follows slice 4, #1563).
//!
//! The gold probe is one model-space line, (0, 0) to (12, 9), seen through twenty rectangular
//! plan viewports on `Layout1`. Since libviprs/pvec-gold-fixtures#102 every viewport's target is
//! the origin and its view centre is the model point at its middle, so each one is placed by
//! `paper = centre + s * (model - view centre)`, `s = height / view height`, and cut to its
//! rectangle. The expected segment of each viewport comes from the authored record
//! (`expected.math.json`), never from the reader. The DXF twin is the same drawing and draws the
//! same way. A target off the origin stays refused (`pvec_1563_viewports`). Needs
//! `--features pvec-convert`.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::pvec::ViewProfile;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow};
use libviprs::pvec::{ReasonCode, SourceUnit};
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::Curve;
use serde_json::Value;

const PROBE: &[u8] =
    include_bytes!("fixtures/pvec-gold/fixtures/paper-space-multi-viewport-001/source.dwg");
const RECORD: &str =
    include_str!("fixtures/pvec-gold/fixtures/paper-space-multi-viewport-001/expected.math.json");

/// One authored viewport: its paper rectangle and the model window it shows.
struct Authored {
    id: String,
    centre: [f64; 2],
    size: [f64; 2],
    view_centre: [f64; 2],
    view_height: f64,
}

fn f(v: &Value) -> f64 {
    let hex = v["f64_hex"].as_str().expect("f64_hex");
    f64::from_bits(u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap())
}

fn authored() -> (Vec<Authored>, [[f64; 2]; 2]) {
    let record: Value = serde_json::from_str(RECORD).unwrap();
    let vps = record["viewports"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| Authored {
            id: v["id"].as_str().unwrap().to_owned(),
            centre: [f(&v["paper"]["center"][0]), f(&v["paper"]["center"][1])],
            size: [f(&v["paper"]["width"]), f(&v["paper"]["height"])],
            view_centre: [f(&v["view"]["center"][0]), f(&v["view"]["center"][1])],
            view_height: f(&v["view"]["height"]),
        })
        .collect();
    let g = &record["entities"][0]["geometry"];
    let line = [
        [f(&g["start"][0]), f(&g["start"][1])],
        [f(&g["end"][0]), f(&g["end"][1])],
    ];
    (vps, line)
}

/// The part of the model line a viewport shows, placed on the sheet, or `None` when it shows none
/// of it. Liang-Barsky against the model window, then the viewport's similarity.
fn expected_segment(v: &Authored, line: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let s = v.size[1] / v.view_height;
    let half = [v.size[0] / s / 2.0, v.view_height / 2.0];
    let (a, d) = (line[0], [line[1][0] - line[0][0], line[1][1] - line[0][1]]);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for axis in 0..2 {
        let lo = v.view_centre[axis] - half[axis];
        let hi = v.view_centre[axis] + half[axis];
        for (p, q) in [(-d[axis], a[axis] - lo), (d[axis], hi - a[axis])] {
            if p == 0.0 {
                if q < 0.0 {
                    return None;
                }
            } else {
                let r = q / p;
                if p < 0.0 {
                    t0 = t0.max(r);
                } else {
                    t1 = t1.min(r);
                }
            }
        }
    }
    if t0 >= t1 {
        return None;
    }
    let place = |t: f64| {
        let m = [a[0] + t * d[0], a[1] + t * d[1]];
        [
            v.centre[0] + s * (m[0] - v.view_centre[0]),
            v.centre[1] + s * (m[1] - v.view_centre[1]),
        ]
    };
    Some([place(t0), place(t1)])
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1579-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A trimmed line as its two ends.
fn ends(c: &Curve) -> [[f64; 2]; 2] {
    let Curve::Trimmed(t) = c else {
        panic!("a trimmed line, not {c:?}");
    };
    let Curve::Line(l) = t.basis() else {
        panic!("a trimmed line, not {c:?}");
    };
    let (o, d, s) = (l.origin(), l.direction(), t.trim().span);
    [[o.x, o.y], [o.x + d.x * s, o.y + d.y * s]]
}

fn near(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| (x - y).abs() <= 1e-9 * (1.0 + y.abs()))
}

/// Converts `source` as `Layout1` and checks every viewport against the authored record.
fn draws_every_viewport(tag: &str, source: CadSource<'_>) {
    let dir = scratch(tag);
    let dest = dir.join("sheet.pvec");
    let report: ConversionReport = PvecConversion::new(source, &dest)
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .convert()
        .unwrap_or_else(|e| panic!("{tag}: the sheet converts: {e:?}"))
        .report;
    let sheet = report.sheet.as_ref().expect("a sheet section");
    let (vps, line) = authored();
    assert_eq!(sheet.viewports.len(), vps.len(), "{:#?}", sheet.viewports);

    let curves = open_published(&dest).unwrap().entities().unwrap();
    let kept: Vec<&EntityRow> = report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .collect();
    assert_eq!(kept.len(), curves.len(), "{:#?}", report.entities);

    for v in &vps {
        let (hw, hh) = (v.size[0] / 2.0, v.size[1] / 2.0);
        let want_box = [
            v.centre[0] - hw,
            v.centre[1] - hh,
            v.centre[0] + hw,
            v.centre[1] + hh,
        ];
        let row = sheet
            .viewports
            .iter()
            .find(|r| {
                r.paper_box.as_ref().is_some_and(|b| {
                    b.iter()
                        .map(|x| x.parse::<f64>().unwrap())
                        .eq(want_box.iter().copied())
                })
            })
            .unwrap_or_else(|| panic!("{tag} {}: no drawn row: {:#?}", v.id, sheet.viewports));
        assert_eq!(
            (row.status.as_str(), row.reason.as_deref()),
            ("drawn", None),
            "{tag} {}",
            v.id
        );
        assert_eq!(
            row.scale.as_deref(),
            Some(format!("1:{}", v.view_height / v.size[1]).as_str()),
            "{tag} {}",
            v.id
        );
        let through: Vec<[[f64; 2]; 2]> = kept
            .iter()
            .zip(&curves)
            .filter(|(r, _)| r.viewport == Some(row.handle))
            .map(|(_, (_, c))| ends(c))
            .collect();
        match expected_segment(v, line) {
            Some(want) => {
                assert_eq!(through.len(), 1, "{tag} {}: {through:?}", v.id);
                let got = through[0];
                assert!(
                    near(got, want) || near([got[1], got[0]], want),
                    "{tag} {}: drawn {got:?}, authored {want:?}",
                    v.id
                );
                assert_eq!(row.entities, 1, "{tag} {}", v.id);
            }
            None => assert!(through.is_empty(), "{tag} {}: {through:?}", v.id),
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1579_the_gold_probe_draws_its_line_through_every_viewport_where_the_record_puts_it() {
    draws_every_viewport("dwg", CadSource::Bytes(PROBE));
}

#[test]
#[cfg_attr(miri, ignore)] // reads the twin from disk, which Miri isolation blocks
fn issue_1579_the_dxf_twin_of_the_probe_draws_the_same() {
    let twin = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-dxf/paper-space-multi-viewport-001.dxf");
    let bytes = fs::read(twin).unwrap();
    draws_every_viewport("dxf", CadSource::Bytes(&bytes));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1579_probing_the_gold_probe_refuses_no_viewport() {
    let probed = PvecConversion::new(CadSource::Bytes(PROBE), Path::new("x.pvec"))
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .probe()
        .expect("Layout1 is walked");
    let refused: Vec<_> = probed
        .reasons()
        .filter(|(r, _, _)| *r == ReasonCode::ViewNotEnabled)
        .collect();
    assert!(refused.is_empty(), "{refused:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1579_every_probe_viewport_has_its_target_at_the_origin() {
    use libviprs::pvec::DwgProvider;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    let provider = AcadrustProvider::open(CadSource::Bytes(PROBE)).unwrap();
    let vps = provider.viewports().unwrap();
    assert_eq!(vps.len(), 20);
    for v in vps {
        assert_eq!(v.view_target, [0.0, 0.0, 0.0], "{:?}", v.handle);
    }
}
