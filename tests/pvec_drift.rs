//! The PVec drift check (EPIC N, follows #1050 and #1060): a `.pvec` against the fixture it was
//! generated from.
//!
//! Every drift kind is shown by writing a deliberately perturbed artifact with PVec's public writer
//! and checking that the library names the kind and the entity. An unperturbed artifact passes. The
//! committed digests (`tests/fixtures/pvec-drift/`) are what makes a real change fail the ordinary
//! suite; `PVEC_GATE_BLESS=1` rewrites them (see `docs/pvec-drift-check.md`).
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gate.rs"]
mod pvec_gate;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::drift::{self, Against, DriftKind, DriftReport};
use libviprs::pvec_convert::PvecConversion;
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Circle, Curve2d, Ellipse, Placement2, Point2, Polyline};
use pvec_gold::{Corpus, EvidenceClass, Fixture};
use serde_json::{Value, json};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-drift")
}

/// The sections of a dump file, `== <fixture>` followed by the dump, as (fixture, dump).
fn sections(text: &str) -> Vec<(&str, &str)> {
    text.split("== ")
        .filter(|s| !s.is_empty())
        .map(|s| s.split_once('\n').expect("a section has a heading line"))
        .collect()
}

fn override_path() -> Option<PathBuf> {
    (std::env::consts::ARCH != "x86_64")
        .then(|| dir().join(format!("accepted.{}.dump", std::env::consts::ARCH)))
}

/// The accepted output for this architecture: `accepted.dump` is the x86_64 one, and an
/// `accepted.<arch>.dump` replaces the sections of the fixtures whose geometry comes out in other
/// bits there (#1471, `docs/adr/pvec-artifact-reproducibility.md`).
fn accepted_file() -> String {
    let base = std::fs::read_to_string(dir().join("accepted.dump")).unwrap_or_else(|e| {
        panic!(
            "tests/fixtures/pvec-drift/accepted.dump is missing ({e}); PVEC_GATE_BLESS=1 writes it"
        )
    });
    let over = override_path()
        .filter(|p| p.exists())
        .map(|p| std::fs::read_to_string(p).unwrap());
    let over = over.as_deref().map(sections).unwrap_or_default();
    sections(&base)
        .into_iter()
        .map(|(id, body)| {
            let body = over.iter().find(|(o, _)| *o == id).map_or(body, |(_, b)| b);
            format!("== {id}\n{body}")
        })
        .collect()
}

// ------------------------------------------------------------------------------- the artifacts

/// A converted fixture, or `None` when the conversion refuses it (no artifact to check).
fn convert(fx: &Fixture) -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(format!("{}.pvec", fx.id));
    let dwg = fx.dwg();
    PvecConversion::new(CadSource::Bytes(&dwg), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .ok()?;
    Some(std::fs::read(&dest).unwrap())
}

/// An artifact opened for editing: PVec's own manifest and curves, written back with PVec's writer.
struct Art {
    manifest: Manifest,
    curves: Vec<(u32, Curve)>,
}

impl Art {
    fn open(bytes: &[u8]) -> Art {
        let r = PvecReader::open(bytes.to_vec()).unwrap();
        Art {
            manifest: r.manifest().clone(),
            curves: r
                .entities()
                .unwrap()
                .into_iter()
                .map(|(id, c)| (id.index(), c))
                .collect(),
        }
    }

    fn write(&self) -> Vec<u8> {
        let mut w = PvecWriter::new(self.manifest.clone()).unwrap();
        for (id, c) in &self.curves {
            w.push(EntityId::new(*id).unwrap(), c).unwrap();
        }
        w.finish().unwrap()
    }
}

/// Sets the manifest bounds to those of the geometry, as a writer would after the edit.
fn refit(art: &mut Art) {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for (_, c) in &art.curves {
        let b = c.bounds().unwrap();
        lo = [lo[0].min(b.min().x), lo[1].min(b.min().y)];
        hi = [hi[0].max(b.max().x), hi[1].max(b.max().y)];
    }
    art.manifest.bounds = Bounds2 { min: lo, max: hi };
}

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y)
}

/// A synthetic artifact of three polylines, with the truth record that describes them.
fn three_polylines() -> (Value, Art) {
    let shapes: [[(f64, f64); 3]; 3] = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        [(2.0, 2.0), (3.0, 2.5), (4.0, 2.0)],
        [(5.0, 0.0), (6.0, 1.0), (7.0, 0.0)],
    ];
    let num =
        |v: f64| json!({"decimal": v.to_string(), "f64_hex": format!("0x{:016x}", v.to_bits())});
    let entities: Vec<Value> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "id": format!("poly-{i}"),
                "class": "polyline",
                "geometry": {"vertices": s.iter().map(|p| json!([num(p.0), num(p.1), num(0.0)])).collect::<Vec<_>>()},
                "closure": {"closed_by_endpoint": false, "periodic": false},
            })
        })
        .collect();
    let truth = json!({
        "fixture_id": "three-polylines",
        "evidence_class": "KNOWN_TRUTH_EXACT",
        "units": {"scale_to_canonical": num(1.0)},
        "entities": entities,
    });
    let curves = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let v = s.iter().map(|p| pt(p.0, p.1)).collect();
            (i as u32, Curve::Polyline(Polyline::new(v).unwrap()))
        })
        .collect();
    let manifest = Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [7.0, 2.5],
        },
        capabilities: Vec::new(),
        producer_id: "test".into(),
        producer_version: "0".into(),
    };
    (truth, Art { manifest, curves })
}

fn report_for_truth(truth: &Value, art: &Art) -> DriftReport {
    drift::check(
        "three-polylines",
        "KNOWN_TRUTH_EXACT",
        Some(truth),
        None,
        &art.write(),
    )
    .unwrap()
}

fn kinds(r: &DriftReport) -> Vec<(Option<u32>, DriftKind)> {
    r.drifts.iter().map(|d| (d.entity, d.kind)).collect()
}

// --------------------------------------------------------------------- the unperturbed passes

#[test]
fn n_1471_an_unperturbed_artifact_has_no_drift_against_its_truth() {
    let (truth, art) = three_polylines();
    let r = report_for_truth(&truth, &art);
    assert!(r.is_clean(), "{}", r.to_text());
    assert!(!r.fails());
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_every_converted_fixture_has_no_drift_against_its_truth_or_its_accepted_output() {
    let accepted = accepted_file();
    let corpus = Corpus::open();
    let mut checked = 0;
    for fx in corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
    {
        let Some(bytes) = convert(fx) else { continue };
        let truth = fx
            .has("expected.math.json")
            .then(|| fx.json("expected.math.json"));
        let section = drift::accepted_section(&accepted, &fx.id);
        let r = drift::check(&fx.id, fx.class.tag(), truth.as_ref(), section, &bytes)
            .unwrap_or_else(|e| panic!("{}: {e}", fx.id));
        assert!(r.is_clean(), "{}", r.to_text());
        assert!(section.is_some(), "{} has no accepted dump", fx.id);
        checked += 1;
    }
    assert!(checked >= 30, "only {checked} fixtures were checked");
}

// ------------------------------------------------- one perturbed artifact per drift kind, truth

#[test]
fn n_1471_a_moved_vertex_is_caught_and_names_the_entity() {
    let (truth, mut art) = three_polylines();
    let Curve::Polyline(p) = &art.curves[1].1 else {
        unreachable!()
    };
    let mut v = p.vertices().to_vec();
    v[1] = pt(v[1].x + 0.001, v[1].y);
    art.curves[1].1 = Curve::Polyline(Polyline::new(v).unwrap());
    let r = report_for_truth(&truth, &art);
    assert_eq!(
        kinds(&r),
        [(Some(1), DriftKind::MovedVertex)],
        "{}",
        r.to_text()
    );
    assert!(r.fails());
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_a_changed_radius_is_caught() {
    let fx_id = "circle-001";
    let corpus = Corpus::open();
    let fx = corpus.get(fx_id);
    let mut art = Art::open(&convert(fx).unwrap());
    let Curve::Circle(c) = &art.curves[0].1 else {
        panic!("circle-001 is not a circle")
    };
    art.curves[0].1 = Curve::Circle(Circle::new(*c.placement(), c.radius() * 1.001).unwrap());
    refit(&mut art);
    let r = drift::check(
        fx_id,
        "KNOWN_TRUTH_EXACT",
        Some(&fx.json("expected.math.json")),
        None,
        &art.write(),
    )
    .unwrap();
    assert_eq!(
        kinds(&r),
        [(Some(0), DriftKind::ChangedRadius)],
        "{}",
        r.to_text()
    );
}

#[test]
fn n_1471_a_dropped_entity_is_caught() {
    let (truth, mut art) = three_polylines();
    art.curves.pop();
    refit(&mut art);
    let r = report_for_truth(&truth, &art);
    assert_eq!(
        kinds(&r),
        [(Some(2), DriftKind::DroppedEntity)],
        "{}",
        r.to_text()
    );
}

#[test]
fn n_1471_an_extra_entity_is_caught() {
    let (truth, mut art) = three_polylines();
    art.curves.push((
        3,
        Curve::Polyline(Polyline::new(vec![pt(0.5, 0.5), pt(0.6, 0.6)]).unwrap()),
    ));
    let r = report_for_truth(&truth, &art);
    assert_eq!(
        kinds(&r),
        [(Some(3), DriftKind::ExtraEntity)],
        "{}",
        r.to_text()
    );
}

#[test]
fn n_1471_reordered_ids_are_caught() {
    let (truth, mut art) = three_polylines();
    let (a, b) = (art.curves[0].1.clone(), art.curves[2].1.clone());
    art.curves[0].1 = b;
    art.curves[2].1 = a;
    let r = report_for_truth(&truth, &art);
    let mut got = kinds(&r);
    got.sort();
    assert_eq!(
        got,
        [
            (Some(0), DriftKind::ReorderedIds),
            (Some(2), DriftKind::ReorderedIds)
        ],
        "{}",
        r.to_text()
    );
}

#[test]
fn n_1471_changed_bounds_are_caught() {
    let (truth, mut art) = three_polylines();
    art.manifest.bounds.max[0] += 1.0;
    let r = report_for_truth(&truth, &art);
    assert_eq!(
        kinds(&r),
        [(None, DriftKind::ChangedBounds)],
        "{}",
        r.to_text()
    );
}

#[test]
fn n_1471_a_changed_unit_is_caught() {
    let (truth, mut art) = three_polylines();
    art.manifest.metres_per_unit = 0.3048; // feet
    let r = report_for_truth(&truth, &art);
    assert_eq!(
        kinds(&r),
        [(None, DriftKind::ChangedUnit)],
        "{}",
        r.to_text()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_a_swapped_class_is_caught_even_when_the_geometry_would_pass() {
    let corpus = Corpus::open();
    let fx = corpus.get("circle-001");
    let mut art = Art::open(&convert(fx).unwrap());
    let Curve::Circle(c) = &art.curves[0].1 else {
        panic!("circle-001 is not a circle")
    };
    // an ellipse is not the circle the fixture says, whatever its semi-axes
    let e = Ellipse::new(
        Placement2::new(c.placement().origin(), c.placement().x_axis()).unwrap(),
        c.radius(),
        c.radius() * 0.5,
    )
    .unwrap();
    art.curves[0].1 = Curve::Ellipse(e);
    refit(&mut art);
    let r = drift::check(
        &fx.id,
        "KNOWN_TRUTH_EXACT",
        Some(&fx.json("expected.math.json")),
        None,
        &art.write(),
    )
    .unwrap();
    assert_eq!(
        kinds(&r),
        [(Some(0), DriftKind::SwappedClass)],
        "{}",
        r.to_text()
    );
}

// ----------------------------------------------- the same kinds against the accepted output

/// `blocks-dynamic-linear-001` has two entities and no truth: only its accepted output can be
/// compared with.
fn accepted_pair() -> (String, Art) {
    let corpus = Corpus::open();
    let fx = corpus.get("blocks-dynamic-linear-001");
    assert_eq!(fx.class, EvidenceClass::DualOracleEquivalent);
    let art = Art::open(&convert(fx).unwrap());
    assert_eq!(art.curves.len(), 2);
    let all = accepted_file();
    (
        drift::accepted_section(&all, &fx.id)
            .expect("accepted section")
            .to_owned(),
        art,
    )
}

fn against_accepted(accepted: &str, art: &Art) -> DriftReport {
    drift::check(
        "blocks-dynamic-linear-001",
        "DUAL_ORACLE_EQUIVALENT",
        None,
        Some(accepted),
        &art.write(),
    )
    .unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_a_fixture_with_no_truth_is_checked_against_the_accepted_output() {
    let (accepted, art) = accepted_pair();
    assert!(against_accepted(&accepted, &art).is_clean());
    // a fixture with no truth and no accepted output cannot be checked at all
    let err = drift::check("x", "DUAL_ORACLE_EQUIVALENT", None, None, &art.write());
    assert!(err.is_err());
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_every_drift_kind_is_caught_against_the_accepted_output() {
    let (accepted, art) = accepted_pair();
    let line_moved = |a: &mut Art| {
        // move the first entity's data by editing whichever curve kind it is
        a.curves[0].1 = match &a.curves[0].1 {
            Curve::Polyline(p) => {
                let mut v = p.vertices().to_vec();
                v[0] = pt(v[0].x + 0.01, v[0].y);
                Curve::Polyline(Polyline::new(v).unwrap())
            }
            Curve::Circle(c) => Curve::Circle(
                Circle::new(
                    Placement2::new(
                        pt(c.placement().origin().x + 0.01, c.placement().origin().y),
                        c.placement().x_axis(),
                    )
                    .unwrap(),
                    c.radius(),
                )
                .unwrap(),
            ),
            other => panic!("unexpected {other:?}"),
        };
    };
    type Edit<'a> = Box<dyn Fn(&mut Art) + 'a>;
    let cases: Vec<(DriftKind, Edit<'_>)> = vec![
        (DriftKind::MovedVertex, Box::new(line_moved)),
        (
            DriftKind::DroppedEntity,
            Box::new(|a| {
                a.curves.pop();
            }),
        ),
        (
            DriftKind::ExtraEntity,
            Box::new(|a| {
                a.curves.push((
                    2,
                    Curve::Polyline(Polyline::new(vec![pt(80.0, 80.0), pt(90.0, 90.0)]).unwrap()),
                ))
            }),
        ),
        (
            DriftKind::ReorderedIds,
            Box::new(|a| {
                let c = a.curves[0].1.clone();
                a.curves[0].1 = a.curves[1].1.clone();
                a.curves[1].1 = c;
            }),
        ),
        (
            DriftKind::ChangedBounds,
            Box::new(|a| a.manifest.bounds.min[1] -= 1.0),
        ),
        (
            DriftKind::ChangedUnit,
            Box::new(|a| a.manifest.metres_per_unit = 0.001),
        ),
        (
            DriftKind::SwappedClass,
            Box::new(|a| {
                a.curves[0].1 =
                    Curve::Polyline(Polyline::new(vec![pt(0.5, 0.5), pt(0.75, 0.5)]).unwrap())
            }),
        ),
    ];
    for (want, edit) in cases {
        let mut a = Art {
            manifest: art.manifest.clone(),
            curves: art.curves.clone(),
        };
        edit(&mut a);
        let r = against_accepted(&accepted, &a);
        assert!(
            r.drifts
                .iter()
                .any(|d| d.kind == want && d.against == Against::Accepted),
            "{}: not reported:\n{}",
            want.name(),
            r.to_text()
        );
        assert!(r.fails());
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_a_quarantined_fixture_reports_drift_and_never_gates() {
    let (accepted, mut art) = accepted_pair();
    art.curves.pop();
    let r = drift::check("x", "QUARANTINED", None, Some(&accepted), &art.write()).unwrap();
    assert!(!r.is_clean());
    assert!(!r.fails());
    assert!(r.to_text().contains("not gating"));
}

// ---------------------------------------------------------------------------- the reports

#[test]
fn n_1471_the_reports_are_machine_readable_and_for_people() {
    let (truth, mut art) = three_polylines();
    art.curves.pop();
    refit(&mut art);
    let r = report_for_truth(&truth, &art);
    let j: Value = serde_json::from_str(&r.to_json()).expect("the machine report is json");
    assert_eq!(j["fixture"], "three-polylines");
    assert_eq!(j["clean"], false);
    assert_eq!(j["drifts"][0]["kind"], "dropped-entity");
    assert_eq!(j["drifts"][0]["entity"], 2);
    assert_eq!(j["drifts"][0]["against"], "known-truth");
    assert_eq!(j["digest"].as_str().unwrap().len(), 64);
    let t = r.to_text();
    assert!(
        t.contains("entity 2") && t.contains("dropped-entity"),
        "{t}"
    );
}

#[test]
fn n_1471_the_digest_is_over_the_canonical_dump_and_moves_with_one_bit() {
    let (_, art) = three_polylines();
    let a = drift::canonical_dump("f", &art.write()).unwrap();
    let mut b = Art {
        manifest: art.manifest.clone(),
        curves: art.curves.clone(),
    };
    let Curve::Polyline(p) = &b.curves[0].1 else {
        unreachable!()
    };
    let mut v = p.vertices().to_vec();
    v[1] = pt(f64::from_bits(v[1].x.to_bits() + 1), v[1].y);
    b.curves[0].1 = Curve::Polyline(Polyline::new(v).unwrap());
    let b = drift::canonical_dump("f", &b.write()).unwrap();
    assert_ne!(drift::digest(&a), drift::digest(&b));
    assert_eq!(
        drift::digest(&a),
        drift::digest(&drift::canonical_dump("f", &art.write()).unwrap())
    );
}

// ----------------------------------------------------------- the committed digests, the suite

/// The accepted dumps (one `== <fixture>` section each) and the digest lines, for every fixture the
/// conversion produces an artifact for.
fn current() -> (String, String) {
    let corpus = Corpus::open();
    let (mut dumps, mut digests) = (String::new(), String::new());
    for fx in corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
    {
        if let Some(bytes) = convert(fx) {
            let dump = drift::canonical_dump(&fx.id, &bytes).unwrap();
            dumps.push_str(&format!("== {}\n{dump}", fx.id));
            digests.push_str(&format!("{} {}\n", fx.id, drift::digest(&dump)));
        }
    }
    (dumps, digests)
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_the_digest_of_every_converted_fixture_matches_the_committed_one() {
    let (dumps, digests) = current();
    if std::env::var_os("PVEC_GATE_BLESS").is_some() {
        // x86_64 writes the accepted output, any other architecture writes only the sections that
        // differ from it (the digest rows are written by `check` the same way)
        let base = std::fs::read_to_string(dir().join("accepted.dump")).ok();
        match (override_path(), base) {
            (Some(over), Some(base)) => {
                let base = sections(&base);
                let changed: String = sections(&dumps)
                    .into_iter()
                    .filter(|s| !base.contains(s))
                    .map(|(id, body)| format!("== {id}\n{body}"))
                    .collect();
                if changed.is_empty() {
                    let _ = std::fs::remove_file(over);
                } else {
                    std::fs::write(over, changed).unwrap();
                }
            }
            _ => std::fs::write(dir().join("accepted.dump"), &dumps).unwrap(),
        }
    }
    pvec_gate::check("geometry drift", "drift", &digests);
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_the_accepted_dump_hashes_to_the_committed_digest() {
    let accepted = accepted_file();
    let baseline = pvec_gate::baseline_text("geometry drift", "drift");
    assert!(baseline.lines().count() >= 30);
    for line in baseline.lines() {
        let (id, want) = line.split_once(' ').unwrap();
        let section = drift::accepted_section(&accepted, id)
            .unwrap_or_else(|| panic!("no accepted dump for {id}"));
        assert_eq!(
            drift::digest(section),
            want,
            "{id}: the accepted dump and its digest disagree"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_the_fixtures_whose_bits_differ_between_architectures_agree_within_the_numeric_policy() {
    // The x86_64 and aarch64 accepted outputs of these fixtures hash differently, and neither is a
    // drift from the other: a check against either gives a clean report on both architectures.
    let Ok(over) = std::fs::read_to_string(dir().join("accepted.aarch64.dump")) else {
        panic!("tests/fixtures/pvec-drift/accepted.aarch64.dump is missing");
    };
    let base = std::fs::read_to_string(dir().join("accepted.dump")).unwrap();
    let corpus = Corpus::open();
    assert!(!sections(&over).is_empty());
    for (id, arm) in sections(&over) {
        let fx = corpus.get(id);
        let x64 = drift::accepted_section(&base, id).unwrap();
        assert_ne!(drift::digest(x64), drift::digest(arm));
        let bytes = convert(fx).unwrap();
        let truth = fx
            .has("expected.math.json")
            .then(|| fx.json("expected.math.json"));
        for accepted in [x64, arm] {
            let r =
                drift::check(id, fx.class.tag(), truth.as_ref(), Some(accepted), &bytes).unwrap();
            assert!(r.is_clean(), "{}", r.to_text());
        }
    }
}

// ------------------------------------------------------------------------- the example binary

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_the_example_prints_both_reports_and_fails_on_drift() {
    let exe = std::env::current_exe().unwrap();
    let example = exe
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/pvec_drift");
    if !example.exists() {
        panic!(
            "the pvec_drift example is not built at {}",
            example.display()
        );
    }
    let corpus = Corpus::open();
    let fx = corpus.get("circle-001");
    let tmp = tempfile::tempdir().unwrap();
    let good = tmp.path().join("good.pvec");
    std::fs::write(&good, convert(fx).unwrap()).unwrap();
    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold/fixtures/circle-001");
    let run = |file: &Path, json: bool| {
        let mut c = std::process::Command::new(&example);
        c.arg(&fixture_dir).arg(file);
        if json {
            c.arg("--json");
        }
        c.output().unwrap()
    };
    let ok = run(&good, true);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let j: Value = serde_json::from_slice(&ok.stdout).unwrap();
    assert_eq!(j["clean"], true);

    let mut art = Art::open(&std::fs::read(&good).unwrap());
    art.manifest.metres_per_unit = 0.3048;
    let bad = tmp.path().join("bad.pvec");
    std::fs::write(&bad, art.write()).unwrap();
    let out = run(&bad, false);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("changed-unit"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn n_1471_the_library_uses_pvecs_public_api_and_none_of_its_maths() {
    let src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pvec/drift.rs"))
            .unwrap();
    for banned in [
        "de_boor",
        "basis_fun",
        "find_span",
        "pvec_container",
        "pvec_codec",
        "pvec::unstable",
    ] {
        assert!(!src.contains(banned), "src/pvec/drift.rs mentions {banned}");
    }
    assert!(src.contains("Policy::V1"), "the numeric policy is PVec's");
}
