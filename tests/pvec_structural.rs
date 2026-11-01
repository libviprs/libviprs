//! N2.4 (#1055): structural exactness, asserted exactly where the source and the decoder expose
//! comparable parameters, and recorded where they do not.
//!
//! One row per property of the issue, each with at least one named fixture of the pinned
//! `pvec-gold-fixtures` corpus. The values come from the decoder (`RawGeometry`) and the mapping
//! (`ExactBasic`, `ExactSpline`), the expectation from the fixture's `expected.math.json`. Nothing
//! in this file uses a tolerance: structural comparison is exact by construction (`==` on the
//! bits), and a comparison that needs a tolerance belongs to `pvec_geometric` (N2.5).
//!
//! Each entity class carries a comparability record: whether the source format exposes the
//! parameter, whether the decoder surfaces it, and when it does not, the weaker conformance class
//! the fixture may claim instead. Structural exactness is never asserted on a property the decoder
//! does not expose, and the record says which class applies instead of skipping quietly.
//!
//! Needs `--features pvec-conformance`.

#![cfg(feature = "pvec-conformance")]

#[path = "common/pvec_gold_decode.rs"]
mod decode;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use libviprs::pvec::dwg_basic::{self, BasicKind};
use libviprs::pvec::dwg_spline::{self, SplineForm, TrimState};
use libviprs::pvec::flatten::FlatEntity;
use libviprs::pvec::mapping::ConformanceClass;
use libviprs::pvec::{RawGeometry, SourceFact};
use pvec_gold::{Corpus, Fixture, num, nums, points};
use serde_json::{Value, json};

// ------------------------------------------------------------------------ comparability records

/// What can be compared exactly for one property of one entity class, and what happens when it
/// cannot.
struct Comparability {
    class: &'static str,
    property: &'static str,
    /// Whether the source format (DWG) carries the parameter at all.
    source_exposes: bool,
    /// Whether the decoder surfaces it.
    decoder_surfaces: bool,
    /// The weaker class the fixture may claim when the property is not comparable.
    fallback: Option<ConformanceClass>,
    why: &'static str,
}

const COMPARABILITY: &[Comparability] = &[
    Comparability {
        class: "spline",
        property: "degree",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "spline",
        property: "control points",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "spline",
        property: "knot vector and multiplicities",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "rational spline",
        property: "weights",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "non-rational spline",
        property: "weights",
        source_exposes: false,
        decoder_surfaces: true,
        fallback: None,
        why: "a non-rational spline has no weight array: its absence is the assertion",
    },
    Comparability {
        class: "arc",
        property: "trim / parameter domain",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "elliptical arc",
        property: "trim / parameter domain",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "spline",
        property: "trim / parameter domain",
        source_exposes: false,
        decoder_surfaces: false,
        fallback: Some(ConformanceClass::GeometricExactness),
        why: "a DWG SPLINE runs over its whole knot domain and has no trim (the mapping reports \
              TrimState::FullDomain for every spline), so only the evaluated domain is compared, and a trimmed spline may claim \
              geometric exactness at most",
    },
    Comparability {
        class: "polyline",
        property: "closed / periodic state",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "",
    },
    Comparability {
        class: "spline",
        property: "closed / periodic state",
        source_exposes: true,
        decoder_surfaces: true,
        fallback: None,
        why: "the decoder reports both flags; the mapping of a periodic spline is refused today \
              (#1374), which the known-truth suite records",
    },
];

#[test]
fn n2_4_1055_a_property_the_source_does_not_expose_names_its_weaker_class() {
    for c in COMPARABILITY {
        let comparable = c.source_exposes && c.decoder_surfaces;
        if comparable {
            assert!(
                c.fallback.is_none(),
                "{} / {}: comparable yet downgraded",
                c.class,
                c.property
            );
        } else if c.property == "weights" {
            // The absence of a weight array is itself compared exactly (see the weights row).
            assert!(c.fallback.is_none() && c.decoder_surfaces);
        } else {
            let class = c.fallback.unwrap_or_else(|| {
                panic!(
                    "{} / {}: not comparable and no weaker class named",
                    c.class, c.property
                )
            });
            assert_ne!(
                class,
                ConformanceClass::StructuralExactness,
                "{} / {}: a property nobody exposes cannot be structurally exact",
                c.class,
                c.property
            );
            assert!(!c.why.is_empty(), "{} / {}: say why", c.class, c.property);
            eprintln!(
                "{} / {}: not comparable, claims {}",
                c.class,
                c.property,
                class.tag()
            );
        }
    }
}

#[test]
fn n2_4_1055_a_dwg_spline_has_no_trim_so_its_record_says_so() {
    // The claim in the record, held against the mapping: every spline comes back untrimmed.
    let corpus = Corpus::open();
    for id in [
        "bspline-d3-001",
        "nurbs-d3-001",
        "bspline-nonunit-domain-001",
        "closed-by-endpoint-001",
    ] {
        let f = leaf(corpus.get(id));
        let mapped = dwg_spline::extract(&f.entity).unwrap();
        assert_eq!(mapped.trim, TrimState::FullDomain, "{id}");
    }
    let row = COMPARABILITY
        .iter()
        .find(|c| c.class == "spline" && c.property == "trim / parameter domain")
        .unwrap();
    assert!(!row.source_exposes && !row.decoder_surfaces);
    assert_eq!(row.fallback, Some(ConformanceClass::GeometricExactness));
}

// ------------------------------------------------------------------------------- the rows

type Check = fn(&Fixture, &Value) -> Result<(), String>;

/// One property of the issue's table, one fixture that asserts it.
struct Row {
    property: &'static str,
    fixture: &'static str,
    check: Check,
}

const ROWS: &[Row] = &[
    Row {
        property: "degree",
        fixture: "bspline-d3-001",
        check: degree,
    },
    Row {
        property: "degree",
        fixture: "bspline-d5-001",
        check: degree,
    },
    Row {
        property: "degree",
        fixture: "nurbs-d2-offconic-001",
        check: degree,
    },
    Row {
        property: "control points",
        fixture: "nurbs-d3-001",
        check: control_points,
    },
    Row {
        property: "control points",
        fixture: "bspline-d3-001",
        check: control_points,
    },
    Row {
        property: "knot vector and multiplicities",
        fixture: "repeated-knot-c0-001",
        check: knots,
    },
    Row {
        property: "knot vector and multiplicities",
        fixture: "bspline-d3-001",
        check: knots,
    },
    Row {
        property: "weights",
        fixture: "nonunit-weight-001",
        check: weights,
    },
    Row {
        property: "weights",
        fixture: "nurbs-d3-001",
        check: weights,
    },
    Row {
        property: "weights",
        fixture: "bspline-d3-001",
        check: weights,
    },
    Row {
        property: "trim / parameter domain",
        fixture: "arc-seam-001",
        check: arc_trim,
    },
    Row {
        property: "trim / parameter domain",
        fixture: "elliptical-arc-001",
        check: arc_trim,
    },
    Row {
        property: "trim / parameter domain",
        fixture: "bspline-nonunit-domain-001",
        check: spline_domain,
    },
    Row {
        property: "closed / periodic state",
        fixture: "lwpolyline-closed-001",
        check: closed,
    },
    Row {
        property: "closed / periodic state",
        fixture: "closed-by-endpoint-001",
        check: closed,
    },
    Row {
        property: "closed / periodic state",
        fixture: "closed-periodic-001",
        check: closed,
    },
];

fn leaf(fx: &Fixture) -> FlatEntity {
    let provider = decode::open(&fx.dwg()).expect("the drawing opens");
    let mut flat = decode::flatten(&provider, &fx.id)
        .expect("the drawing flattens")
        .entities;
    assert_eq!(flat.len(), 1, "{}: one leaf expected", fx.id);
    flat.remove(0)
}

/// What the decoder reports of a spline: degree, control points, knots, weights (absent for a
/// non-rational one), periodic flag, closed flag.
type DecodedSpline = (u32, Vec<[f64; 3]>, Vec<f64>, Option<Vec<f64>>, bool, bool);

fn spline_of(f: &FlatEntity) -> Result<DecodedSpline, String> {
    let RawGeometry::Spline {
        degree,
        control_points,
        knots,
        weights,
        periodic,
        closed,
    } = &f.entity.geometry
    else {
        return Err("the decoder did not report a spline".into());
    };
    let knots = knots
        .present()
        .ok_or("the decoder did not surface the knots")?
        .clone();
    let periodic = *periodic
        .present()
        .ok_or("the decoder did not surface the periodic flag")?;
    let weights = match weights {
        SourceFact::Present(w) => Some(w.clone()),
        SourceFact::Absent => None,
        _ => return Err("the decoder did not surface the weights".into()),
    };
    Ok((
        *degree,
        control_points.clone(),
        knots,
        weights,
        periodic,
        *closed,
    ))
}

fn degree(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let (d, ..) = spline_of(&f)?;
    let want = t["entities"][0]["geometry"]["degree"].as_u64().unwrap() as u32;
    if d != want {
        return Err(format!("degree: decoded {d}, truth {want}"));
    }
    let mapped =
        dwg_spline::extract(&f.entity).map_err(|e| format!("degree: mapping refused: {e}"))?;
    if mapped.degree != want {
        return Err(format!("degree: mapped {}, truth {want}", mapped.degree));
    }
    Ok(())
}

fn control_points(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let mapped = dwg_spline::extract(&f.entity)
        .map_err(|e| format!("control points: mapping refused: {e}"))?;
    let want = points(&t["entities"][0]["geometry"]["control_points"]);
    if mapped.control_points.len() != want.len() {
        return Err(format!(
            "control points: {} mapped, {} in the truth",
            mapped.control_points.len(),
            want.len()
        ));
    }
    for (i, (a, w)) in mapped.control_points.iter().zip(&want).enumerate() {
        if a[..] != w[..] {
            return Err(format!(
                "control point {i}: mapped {a:?}, truth {w:?}, in source order"
            ));
        }
    }
    Ok(())
}

/// The knot vector as (value, multiplicity) runs.
fn runs(knots: &[f64]) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = Vec::new();
    for &k in knots {
        match out.last_mut() {
            Some((v, m)) if *v == k => *m += 1,
            _ => out.push((k, 1)),
        }
    }
    out
}

fn knots(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let mapped =
        dwg_spline::extract(&f.entity).map_err(|e| format!("knot vector: mapping refused: {e}"))?;
    let want: Vec<(f64, usize)> = t["entities"][0]["geometry"]["knots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            (
                num(&k["value"]),
                k["multiplicity"].as_u64().unwrap() as usize,
            )
        })
        .collect();
    let got = runs(&mapped.knots);
    if got != want {
        return Err(format!(
            "knot vector and multiplicities: mapped {got:?}, truth {want:?}"
        ));
    }
    Ok(())
}

fn weights(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let mapped =
        dwg_spline::extract(&f.entity).map_err(|e| format!("weights: mapping refused: {e}"))?;
    match (&mapped.form, t["entities"][0]["geometry"].get("weights")) {
        (SplineForm::Rational(w), Some(want)) if *w == nums(want) => Ok(()),
        (SplineForm::NonRational, None) => {
            // Absence, not a row of ones: the decoder reports no weight array either.
            let (.., w, _, _) = spline_of(&f)?;
            if w.is_some() {
                return Err(
                    "weights: the decoder invented a weight array for a non-rational spline".into(),
                );
            }
            Ok(())
        }
        (SplineForm::NonRational, Some(_)) => {
            Err("weights: the truth has weights, the mapping none".into())
        }
        (SplineForm::Rational(w), None) => Err(format!(
            "weights: {w:?} for a spline the truth calls non-rational"
        )),
        (SplineForm::Rational(w), Some(want)) => {
            Err(format!("weights: mapped {w:?}, truth {:?}", nums(want)))
        }
        _ => Err("weights: a form this suite does not know".into()),
    }
}

fn arc_trim(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let b = dwg_basic::extract(&f.entity).map_err(|e| format!("trim: mapping refused: {e}"))?;
    if !matches!(b.kind, BasicKind::Arc | BasicKind::EllipticalArc) {
        return Err(format!("trim: {:?} is not an arc", b.kind));
    }
    let (start, end) = b.source_range.ok_or("trim: no source range")?;
    let g = &t["entities"][0]["geometry"];
    let want = num(&g["start_angle"]);
    if start != want {
        return Err(format!(
            "trim / parameter domain: start {start}, truth {want}"
        ));
    }
    // The drawing stores an end angle, the truth the unwrapped end of the domain. They are the same
    // number unless the trim wraps through a full turn (the stored end is then below the start),
    // and a wrapped end is not comparable bit for bit, so only its start is.
    let (dstart, dend) = (
        num(&t["entities"][0]["parameter_domain"]["start"]),
        num(&t["entities"][0]["parameter_domain"]["end"]),
    );
    if start != dstart {
        return Err(format!(
            "trim / parameter domain: start {start}, truth domain {dstart}"
        ));
    }
    if end.to_bits() != dend.to_bits() && end >= start {
        return Err(format!(
            "trim / parameter domain: end {end}, truth domain end {dend}"
        ));
    }
    Ok(())
}

fn spline_domain(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let mapped = dwg_spline::extract(&f.entity)
        .map_err(|e| format!("parameter domain: mapping refused: {e}"))?;
    let d = &t["entities"][0]["parameter_domain"];
    let want = (num(&d["start"]), num(&d["end"]));
    if (mapped.domain.lo, mapped.domain.hi) != want {
        return Err(format!(
            "trim / parameter domain: mapped [{}, {}], truth {want:?}",
            mapped.domain.lo, mapped.domain.hi
        ));
    }
    Ok(())
}

fn closed(fx: &Fixture, t: &Value) -> Result<(), String> {
    let f = leaf(fx);
    let cl = &t["entities"][0]["closure"];
    let by_endpoint = cl["closed_by_endpoint"].as_bool().unwrap();
    let by_knots = cl["closed_by_knots"].as_bool().unwrap();
    let periodic = cl["periodic"].as_bool().unwrap();
    match &f.entity.geometry {
        RawGeometry::Polyline { closed, .. } => {
            if *closed != by_endpoint {
                return Err(format!(
                    "closed / periodic state: decoded closed {closed}, truth {by_endpoint}"
                ));
            }
            let b = dwg_basic::extract(&f.entity)
                .map_err(|e| format!("closed state: mapping refused: {e}"))?;
            if b.closed != by_endpoint {
                return Err(format!(
                    "closed / periodic state: mapped closed {}, truth {by_endpoint}",
                    b.closed
                ));
            }
            Ok(())
        }
        RawGeometry::Spline { .. } => {
            let (_, cps, _, _, p, c) = spline_of(&f)?;
            let ends = cps.first() == cps.last();
            if p != periodic || c != by_knots || ends != by_endpoint {
                return Err(format!(
                    "closed / periodic state: decoded periodic {p} closed {c} endpoints-coincide {ends}, \
                     truth periodic {periodic} closed_by_knots {by_knots} closed_by_endpoint {by_endpoint}"
                ));
            }
            // The mapping keeps the flags it is given, wherever it accepts the spline at all.
            if let Ok(m) = dwg_spline::extract(&f.entity)
                && (m.periodic != p || m.closed != c)
            {
                return Err("closed / periodic state: the mapping changed a flag".into());
            }
            Ok(())
        }
        _ => Err("closed / periodic state: not a polyline or a spline".into()),
    }
}

fn run_row(corpus: &Corpus, row: &Row, edit: impl FnOnce(&mut Value)) -> Result<(), String> {
    let fx = corpus.get(row.fixture);
    let mut truth = fx.truth().expect("a structural fixture is known truth");
    edit(&mut truth);
    (row.check)(fx, &truth).map_err(|e| format!("[{}] {}: {e}", row.property, row.fixture))
}

// ------------------------------------------------------------------------------- the tests

#[test]
fn n2_4_1055_every_property_row_has_a_named_fixture_that_passes() {
    let corpus = Corpus::open();
    let wanted = [
        "degree",
        "control points",
        "knot vector and multiplicities",
        "weights",
        "trim / parameter domain",
        "closed / periodic state",
    ];
    for property in wanted {
        let rows: Vec<&Row> = ROWS.iter().filter(|r| r.property == property).collect();
        assert!(!rows.is_empty(), "no fixture asserts {property}");
        for r in rows {
            run_row(&corpus, r, |_| {}).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}

#[test]
fn n2_4_1055_a_non_rational_spline_asserts_that_no_weight_array_is_present() {
    let corpus = Corpus::open();
    let row = ROWS
        .iter()
        .find(|r| r.fixture == "bspline-d3-001" && r.property == "weights")
        .unwrap();
    run_row(&corpus, row, |_| {}).unwrap();
    // Claiming weights of one for it is wrong, not equivalent.
    let err = run_row(&corpus, row, |t| {
        t["entities"][0]["geometry"]["weights"] = json!([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    })
    .unwrap_err();
    assert!(err.contains("weights"), "{err}");
}

#[test]
fn n2_4_1055_mutating_one_knot_multiplicity_names_the_property() {
    let corpus = Corpus::open();
    let row = ROWS
        .iter()
        .find(|r| r.fixture == "repeated-knot-c0-001")
        .unwrap();
    let err = run_row(&corpus, row, |t| {
        let ks = t["entities"][0]["geometry"]["knots"]
            .as_array_mut()
            .unwrap();
        let k = ks
            .iter_mut()
            .find(|k| k["multiplicity"] == 3 && k["value"]["decimal"] == "0.5")
            .unwrap();
        k["multiplicity"] = json!(2);
    })
    .unwrap_err();
    assert!(err.contains("knot vector and multiplicities"), "{err}");
    assert!(err.contains("repeated-knot-c0-001"), "{err}");
}

#[test]
fn n2_4_1055_degree_control_point_weight_and_trim_mutations_name_their_property() {
    let corpus = Corpus::open();
    let find = |f: &str, p: &str| {
        ROWS.iter()
            .find(|r| r.fixture == f && r.property == p)
            .unwrap()
    };
    let err = run_row(&corpus, find("bspline-d3-001", "degree"), |t| {
        t["entities"][0]["geometry"]["degree"] = json!(4);
    })
    .unwrap_err();
    assert!(err.contains("degree"), "{err}");
    let err = run_row(&corpus, find("nurbs-d3-001", "control points"), |t| {
        let p = &mut t["entities"][0]["geometry"]["control_points"];
        p.as_array_mut().unwrap().swap(1, 2);
    })
    .unwrap_err();
    assert!(err.contains("control point"), "{err}");
    let err = run_row(&corpus, find("nonunit-weight-001", "weights"), |t| {
        t["entities"][0]["geometry"]["weights"][2] =
            json!({"decimal": "49", "f64_hex": format!("0x{:016x}", 49.0_f64.to_bits())});
    })
    .unwrap_err();
    assert!(err.contains("weights"), "{err}");
    let err = run_row(
        &corpus,
        find("arc-seam-001", "trim / parameter domain"),
        |t| {
            t["entities"][0]["geometry"]["start_angle"] =
                json!({"decimal": "5.4", "f64_hex": format!("0x{:016x}", 5.4_f64.to_bits())});
        },
    )
    .unwrap_err();
    assert!(err.contains("trim / parameter domain"), "{err}");
    let err = run_row(
        &corpus,
        find("closed-by-endpoint-001", "closed / periodic state"),
        |t| {
            t["entities"][0]["closure"]["periodic"] = json!(true);
        },
    )
    .unwrap_err();
    assert!(err.contains("closed / periodic state"), "{err}");
}

#[test]
fn n2_4_1055_no_assertion_in_this_file_uses_a_tolerance() {
    // Built from pieces so this file does not match its own scan.
    let needles = [
        ["Pol", "icy"].concat(),
        [".nea", "r("].concat(),
        ["abs() ", "<"].concat(),
        ["EPS", "ILON"].concat(),
        ["1e", "-"].concat(),
    ];
    for n in &needles {
        let hit = include_str!("pvec_structural.rs")
            .lines()
            .find(|l| l.contains(n.as_str()) && !l.trim_start().starts_with("//"));
        assert!(
            hit.is_none(),
            "found {n:?} in a structural assertion: {hit:?}"
        );
    }
}
