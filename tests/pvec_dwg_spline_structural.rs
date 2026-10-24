//! Exact B-spline extraction (N4.3, #1074): the structural parameters survive as
//! authored, or the spline is refused with a typed error.
//!
//! Needs `--features pvec-spline`. The last test also needs `dwg-acadrust` and reads
//! a real drawing through the first provider.

#![cfg(feature = "pvec-spline")]

use libviprs::pvec::dwg_spline::{
    Requires, SplineClass, SplineError, SplineField, SplineForm, TrimState, classify, extract,
};
use libviprs::pvec::{InstancePath, Point3, ProvenanceStub, RawEntity, RawGeometry, SourceFact};
use pvec_numeric::{MAX_CONTROL_POINTS, MAX_DEGREE, MAX_KNOTS};

fn entity(class: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        geometry,
        provenance: ProvenanceStub {
            handle: None,
            class_name: class.to_string(),
            block_path: InstancePath::root(),
            layer: "0".to_string(),
        },
    }
}

fn pts(n: usize) -> Vec<Point3> {
    (0..n)
        .map(|i| [i as f64, (i * i) as f64 * 0.5, 0.0])
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn spline(
    degree: u32,
    control_points: Vec<Point3>,
    knots: SourceFact<Vec<f64>>,
    weights: SourceFact<Vec<f64>>,
    periodic: SourceFact<bool>,
    closed: bool,
) -> RawEntity {
    entity(
        "SPLINE",
        RawGeometry::Spline {
            degree,
            control_points,
            knots,
            weights,
            periodic,
            closed,
        },
    )
}

fn plain(degree: u32, n: usize, knots: Vec<f64>) -> RawEntity {
    spline(
        degree,
        pts(n),
        SourceFact::Present(knots),
        SourceFact::Absent,
        SourceFact::Present(false),
        false,
    )
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn clamped_spline_keeps_counts_and_knots() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let s = extract(&plain(3, 6, knots.clone())).unwrap();
    assert_eq!(s.degree, 3);
    assert_eq!(s.control_points.len(), 6);
    assert_eq!(
        s.knots.len(),
        s.control_points.len() + s.degree as usize + 1
    );
    assert_eq!(bits(&s.knots), bits(&knots));
    assert_eq!((s.domain.lo, s.domain.hi), (0.0, 3.0));
    assert_eq!(s.trim, TrimState::FullDomain);
    assert!(matches!(s.form, SplineForm::NonRational));
    assert!(!s.closed && !s.periodic);
}

#[test]
fn authored_interior_multiplicity_survives_byte_for_byte() {
    // Interior knot 2.0 twice: C1 at that break for a cubic.
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let s = extract(&plain(3, 7, knots.clone())).unwrap();
    assert_eq!(bits(&s.knots), bits(&knots));
    assert_eq!(s.knots.iter().filter(|k| **k == 2.0).count(), 2);
    assert_eq!(s.knots.len(), 7 + 3 + 1);
}

#[test]
fn periodic_flag_is_kept_and_nothing_is_unrolled() {
    // Uniform unclamped knots, the stored form of a periodic cubic.
    let knots: Vec<f64> = (0..10).map(f64::from).collect();
    let mut e = plain(3, 6, knots.clone());
    if let RawGeometry::Spline { periodic, .. } = &mut e.geometry {
        *periodic = SourceFact::Present(true);
    }
    let s = extract(&e).unwrap();
    assert!(s.periodic);
    assert_eq!(
        s.control_points.len(),
        6,
        "periodicity must not duplicate control points"
    );
    assert_eq!(
        bits(&s.knots),
        bits(&knots),
        "knots are not rewritten either"
    );
}

#[test]
fn closed_flag_is_a_flag_not_a_repeated_point() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let mut e = plain(3, 6, knots);
    if let RawGeometry::Spline { closed, .. } = &mut e.geometry {
        *closed = true;
    }
    let s = extract(&e).unwrap();
    assert!(s.closed);
    assert_eq!(s.control_points.len(), 6);
    assert_ne!(s.control_points.first(), s.control_points.last());
}

#[test]
fn rational_and_non_rational_stay_distinct() {
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut e = plain(2, 3, knots);
    let non = extract(&e).unwrap();
    assert!(matches!(non.form, SplineForm::NonRational));
    // Explicit unit weights are still rational: they are authored, not defaulted.
    if let RawGeometry::Spline { weights, .. } = &mut e.geometry {
        *weights = SourceFact::Present(vec![1.0, 0.5, 1.0]);
    }
    match extract(&e).unwrap().form {
        SplineForm::Rational(w) => assert_eq!(bits(&w), bits(&[1.0, 0.5, 1.0])),
        other => panic!("expected rational, got {other:?}"),
    }
}

#[test]
fn bad_weights_are_refused_naming_the_control_point() {
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    for (w, idx) in [
        (vec![1.0, 1.0, 0.0], 2),
        (vec![1.0, -0.5, 1.0], 1),
        (vec![f64::NAN, 1.0, 1.0], 0),
        (vec![1.0, f64::INFINITY, 1.0], 1),
    ] {
        let mut e = plain(2, 3, knots.clone());
        if let RawGeometry::Spline { weights, .. } = &mut e.geometry {
            *weights = SourceFact::Present(w);
        }
        match extract(&e) {
            Err(SplineError::BadWeight { index, .. }) => assert_eq!(index, idx),
            other => panic!("expected BadWeight at {idx}, got {other:?}"),
        }
    }
}

#[test]
fn weight_count_mismatch_is_typed() {
    let mut e = plain(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    if let RawGeometry::Spline { weights, .. } = &mut e.geometry {
        *weights = SourceFact::Present(vec![1.0, 1.0]);
    }
    assert_eq!(
        extract(&e).unwrap_err(),
        SplineError::WeightCount {
            expected: 3,
            observed: 2
        }
    );
}

#[test]
fn too_many_control_points_is_refused_with_limit_and_count() {
    let n = MAX_CONTROL_POINTS + 1;
    let e = plain(3, n, vec![0.0; n + 4]);
    assert_eq!(
        extract(&e).unwrap_err(),
        SplineError::TooManyControlPoints {
            limit: MAX_CONTROL_POINTS,
            observed: n
        }
    );
}

#[test]
fn too_many_knots_is_refused_with_limit_and_count() {
    let observed = MAX_KNOTS + 1;
    let e = plain(3, 10, vec![0.0; observed]);
    assert_eq!(
        extract(&e).unwrap_err(),
        SplineError::TooManyKnots {
            limit: MAX_KNOTS,
            observed
        }
    );
}

#[test]
fn degree_outside_the_supported_range_is_refused() {
    let e = plain(0, 3, vec![0.0, 0.0, 1.0, 1.0]);
    assert!(matches!(
        extract(&e),
        Err(SplineError::DegreeOutOfRange { degree: 0, .. })
    ));
    let d = MAX_DEGREE as u32 + 1;
    let n = MAX_DEGREE + 3;
    let e = plain(d, n, vec![0.0; n + MAX_DEGREE + 2]);
    assert!(matches!(
        extract(&e),
        Err(SplineError::DegreeOutOfRange { limit, .. }) if limit == MAX_DEGREE
    ));
}

#[test]
fn knot_structure_errors_are_typed() {
    assert_eq!(
        extract(&plain(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0])).unwrap_err(),
        SplineError::KnotCount {
            expected: 6,
            observed: 5
        }
    );
    assert_eq!(
        extract(&plain(2, 3, vec![0.0, 0.0, 0.0, 1.0, 0.5, 1.0])).unwrap_err(),
        SplineError::KnotsDecrease { index: 4 }
    );
    assert_eq!(
        extract(&plain(2, 3, vec![0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0])).unwrap_err(),
        SplineError::NonFiniteKnot { index: 2 }
    );
}

#[test]
fn a_knot_repeated_beyond_what_the_degree_allows_goes_to_the_policy() {
    // Interior multiplicity 4 on a cubic: more than the degree.
    let e = plain(
        3,
        8,
        vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
    );
    assert!(matches!(extract(&e), Err(SplineError::Policy(_))));
}

#[test]
fn unreported_facts_name_the_missing_field() {
    let mk = |knots, periodic| spline(2, pts(3), knots, SourceFact::Absent, periodic, false);
    for knots in [SourceFact::NotReportable, SourceFact::Absent] {
        assert_eq!(
            extract(&mk(knots, SourceFact::Present(false))).unwrap_err(),
            SplineError::MissingField {
                field: SplineField::Knots,
                requires: Requires::Extension
            }
        );
    }
    let ok = SourceFact::Present(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        extract(&mk(ok, SourceFact::NotReportable)).unwrap_err(),
        SplineError::MissingField {
            field: SplineField::Periodic,
            requires: Requires::Extension
        }
    );
}

#[test]
fn unreportable_weights_are_not_read_as_non_rational() {
    let mut e = plain(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    if let RawGeometry::Spline { weights, .. } = &mut e.geometry {
        *weights = SourceFact::NotReportable;
    }
    assert_eq!(
        extract(&e).unwrap_err(),
        SplineError::MissingField {
            field: SplineField::Weights,
            requires: Requires::Extension
        }
    );
}

#[test]
fn three_d_and_helix_are_classified_not_projected() {
    let mut e = plain(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    assert_eq!(classify(&e), SplineClass::Planar);
    if let RawGeometry::Spline { control_points, .. } = &mut e.geometry {
        control_points[1][2] = 4.0;
    }
    assert_eq!(classify(&e), SplineClass::ThreeD);
    assert_eq!(
        extract(&e).unwrap_err(),
        SplineError::Unsupported3d {
            class: SplineClass::ThreeD,
            requires: Requires::PVecCapability
        }
    );
    let helix = entity(
        "HELIX",
        RawGeometry::Unmapped {
            reason: "helix".into(),
        },
    );
    assert_eq!(classify(&helix), SplineClass::Helix);
    assert!(matches!(
        extract(&helix),
        Err(SplineError::Unsupported3d {
            class: SplineClass::Helix,
            ..
        })
    ));
    let line = entity(
        "LINE",
        RawGeometry::Line {
            start: [0.0; 3],
            end: [1.0; 3],
        },
    );
    assert_eq!(classify(&line), SplineClass::NotSpline);
    assert_eq!(extract(&line).unwrap_err(), SplineError::NotSpline);
}

/// The emitted semantic types and the module's code never mention a polyline, an arc
/// or a sampling step.
#[test]
fn no_code_path_turns_a_spline_into_polylines_or_arcs() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pvec/dwg_spline.rs"),
    )
    .unwrap();
    let code: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in [
        "Polyline",
        "RawVertex",
        "Arc",
        "tessellat",
        "flatten",
        "sample",
        "resample",
        "fit",
    ] {
        assert!(!code.contains(banned), "dwg_spline.rs mentions {banned}");
    }
}

#[cfg(feature = "dwg-acadrust")]
#[test]
fn a_real_drawing_goes_through_unchanged() {
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cad/g13/g13_spline.dwg");
    let p = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
    let layouts = p.layouts().unwrap();
    let m = layouts
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let all = p
        .entities(&EntityScope::Layout(m.id))
        .unwrap()
        .collect_all()
        .unwrap();
    let mut seen = 0;
    for e in &all {
        let RawGeometry::Spline {
            degree,
            control_points,
            knots,
            ..
        } = &e.geometry
        else {
            continue;
        };
        let s = extract(e).unwrap_or_else(|err| panic!("{err}"));
        seen += 1;
        assert_eq!(s.degree, *degree);
        assert_eq!(s.control_points.len(), control_points.len());
        assert_eq!(bits(&s.knots), bits(knots.present().unwrap()));
    }
    assert!(seen > 0, "the fixture has a spline");
}
