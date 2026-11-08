//! Issue #1546 (slice 1 of #1543): the calibration a caller can hand libviprs. A scale claim and a
//! georeference, each checked against the rules PVec's calibration block will hold them to, and a
//! projective transform refused with its own error because the block cannot carry one. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use libviprs::pvec_convert::{
    Calibration, CalibrationError, ControlPoint, CrsDefinition, CrsKind, DefinitionFormat,
    Georeference, GeoreferenceMethod, ScaleClaim, ScaleProvenance,
};

fn plan_scale() -> ScaleClaim {
    // 1:100 on a sheet drawn in millimetres: one unit stands for 0.1 m.
    ScaleClaim {
        metres_per_source_unit: 0.1,
        provenance: ScaleProvenance::PlanScale {
            numerator: 1,
            denominator: 100,
            paper_metres_per_unit: 0.001,
        },
        uncertainty: None,
        note: "1:100".to_owned(),
    }
}

fn georeference() -> Georeference {
    Georeference {
        affine: [0.5, 0.0, 0.0, 0.5, 530_000.0, 180_000.0],
        crs_kind: CrsKind::Projected,
        method: GeoreferenceMethod::Declared,
        crs_id: Some("EPSG:27700".to_owned()),
        definition: None,
        control_points: vec![
            ControlPoint {
                source: [0.0, 0.0],
                world: [530_000.0, 180_000.0],
            },
            ControlPoint {
                source: [100.0, 50.0],
                world: [530_050.0, 180_025.0],
            },
        ],
        residual_max: 0.0,
    }
}

fn refused(scale: Option<ScaleClaim>, georef: Option<Georeference>) -> CalibrationError {
    Calibration::new(scale, georef, None).expect_err("the calibration should be refused")
}

#[test]
fn a_plan_scale_and_a_georeference_are_accepted_and_kept() {
    let c = Calibration::new(
        Some(plan_scale()),
        Some(georeference()),
        Some("site-42".to_owned()),
    )
    .unwrap();
    assert_eq!(c.scale(), Some(&plan_scale()));
    assert_eq!(c.georeference(), Some(&georeference()));
    assert_eq!(c.external_id(), Some("site-42"));
}

#[test]
fn a_calibration_needs_a_scale_or_a_georeference() {
    assert!(matches!(refused(None, None), CalibrationError::Empty));
}

#[test]
fn a_measured_scale_is_the_distance_over_the_source_length() {
    let s = ScaleClaim::measured([0.0, 0.0], [3.0, 4.0], 10.0).unwrap();
    assert_eq!(s.metres_per_source_unit, 2.0);
    assert!(Calibration::new(Some(s), None, None).is_ok());
}

#[test]
fn a_measured_pair_on_one_point_is_refused() {
    assert!(matches!(
        ScaleClaim::measured([1.0, 1.0], [1.0, 1.0], 10.0),
        Err(CalibrationError::Degenerate)
    ));
}

#[test]
fn a_scale_outside_two_to_the_forty_either_way_is_refused() {
    for bad in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        2f64.powi(41),
        2f64.powi(-41),
    ] {
        let s = ScaleClaim {
            metres_per_source_unit: bad,
            ..plan_scale()
        };
        assert!(
            matches!(refused(Some(s), None), CalibrationError::OutOfRange { .. }),
            "{bad} should be out of range"
        );
    }
}

#[test]
fn an_uncertainty_must_be_in_zero_to_one() {
    for bad in [-0.1, 1.0, f64::NAN] {
        let s = ScaleClaim {
            uncertainty: Some(bad),
            ..plan_scale()
        };
        assert!(matches!(
            refused(Some(s), None),
            CalibrationError::OutOfRange { .. }
        ));
    }
}

#[test]
fn evidence_must_match_the_provenance() {
    let zero_ratio = ScaleClaim {
        provenance: ScaleProvenance::PlanScale {
            numerator: 0,
            denominator: 100,
            paper_metres_per_unit: 0.001,
        },
        ..plan_scale()
    };
    assert!(matches!(
        refused(Some(zero_ratio), None),
        CalibrationError::Evidence
    ));
}

#[test]
fn a_singular_affine_is_refused() {
    let g = Georeference {
        affine: [1.0, 2.0, 2.0, 4.0, 0.0, 0.0],
        ..georeference()
    };
    assert!(matches!(refused(None, Some(g)), CalibrationError::Affine));
}

#[test]
fn an_affine_with_a_linear_entry_over_two_to_the_twenty_is_refused() {
    let g = Georeference {
        affine: [2f64.powi(21), 0.0, 0.0, 1.0, 0.0, 0.0],
        ..georeference()
    };
    assert!(matches!(refused(None, Some(g)), CalibrationError::Affine));
}

#[test]
fn a_crs_id_outside_its_class_or_over_64_bytes_is_refused() {
    for bad in ["EPSG 27700", "EPSG:27700\n", &"E".repeat(65)] {
        let g = Georeference {
            crs_id: Some(bad.to_owned()),
            ..georeference()
        };
        assert!(
            matches!(refused(None, Some(g)), CalibrationError::Text { .. }),
            "{bad:?} should be refused"
        );
    }
    let urn = Georeference {
        crs_id: Some("urn:ogc:def:crs:EPSG::27700".to_owned()),
        ..georeference()
    };
    assert!(Calibration::new(None, Some(urn), None).is_ok());
}

#[test]
fn a_georeference_names_its_crs_by_id_or_definition() {
    let none = Georeference {
        crs_id: None,
        definition: None,
        ..georeference()
    };
    assert!(matches!(refused(None, Some(none)), CalibrationError::NoCrs));
    let wkt = Georeference {
        crs_id: None,
        definition: Some(CrsDefinition {
            format: DefinitionFormat::Wkt1,
            text: "GEOGCS[\"WGS 84\",\n\tDATUM[\"WGS_1984\"]]".to_owned(),
        }),
        ..georeference()
    };
    assert!(Calibration::new(None, Some(wkt), None).is_ok());
}

#[test]
fn a_definition_over_16_kib_or_with_a_bidi_override_is_refused() {
    for text in ["G".repeat(16 * 1024 + 1), "GEOGCS[\u{202e}]".to_owned()] {
        let g = Georeference {
            definition: Some(CrsDefinition {
                format: DefinitionFormat::Projjson,
                text,
            }),
            ..georeference()
        };
        assert!(matches!(
            refused(None, Some(g)),
            CalibrationError::Text { .. }
        ));
    }
}

#[test]
fn a_note_with_a_control_character_or_over_255_bytes_is_refused() {
    for note in [
        "a\u{7}b".to_owned(),
        "n".repeat(256),
        "a\u{2028}b".to_owned(),
    ] {
        let s = ScaleClaim {
            note,
            ..plan_scale()
        };
        assert!(matches!(
            refused(Some(s), None),
            CalibrationError::Text { .. }
        ));
    }
    let long_id = Calibration::new(Some(plan_scale()), None, Some("x".repeat(256)));
    assert!(matches!(long_id, Err(CalibrationError::Text { .. })));
}

#[test]
fn more_than_1024_control_points_are_refused() {
    let g = Georeference {
        control_points: (0..1025)
            .map(|n| ControlPoint {
                source: [f64::from(n), 0.0],
                world: [f64::from(n), 0.0],
            })
            .collect(),
        ..georeference()
    };
    assert!(matches!(
        refused(None, Some(g)),
        CalibrationError::TooMany { .. }
    ));
}

#[test]
fn one_source_point_with_two_world_points_is_refused() {
    let mut g = georeference();
    g.control_points.push(ControlPoint {
        source: [0.0, 0.0],
        world: [1.0, 1.0],
    });
    assert!(matches!(
        refused(None, Some(g)),
        CalibrationError::Contradictory { .. }
    ));
}

#[test]
fn a_residual_is_only_kept_for_a_fit_of_three_points_or_more() {
    let g = Georeference {
        residual_max: 0.5,
        ..georeference()
    };
    assert!(matches!(refused(None, Some(g)), CalibrationError::Evidence));
}

#[test]
fn an_affine_matrix_is_taken_apart_in_pvec_order() {
    let m = [[0.5, 0.25, 100.0], [0.125, -0.5, 50.0], [0.0, 0.0, 1.0]];
    assert_eq!(
        Georeference::affine_from_matrix(m).unwrap(),
        [0.5, 0.25, 0.125, -0.5, 100.0, 50.0]
    );
}

#[test]
fn a_projective_matrix_is_refused_with_a_typed_error() {
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]];
    assert!(matches!(
        Georeference::affine_from_matrix(m),
        Err(CalibrationError::Projective)
    ));
}

#[test]
fn a_scale_derived_from_the_georeference_needs_the_georeference() {
    let s = ScaleClaim {
        provenance: ScaleProvenance::Georeference,
        ..plan_scale()
    };
    assert!(matches!(
        refused(Some(s.clone()), None),
        CalibrationError::Evidence
    ));
    assert!(Calibration::new(Some(s), Some(georeference()), None).is_ok());
}

#[test]
fn a_fitted_georeference_needs_two_points() {
    let g = Georeference {
        method: GeoreferenceMethod::Fitted,
        control_points: georeference().control_points[..1].to_vec(),
        ..georeference()
    };
    assert!(matches!(refused(None, Some(g)), CalibrationError::Evidence));
}

#[test]
fn negative_zero_is_stored_as_zero() {
    let g = Georeference {
        affine: [0.5, -0.0, -0.0, 0.5, -0.0, 180_000.0],
        ..georeference()
    };
    let s = ScaleClaim {
        uncertainty: Some(-0.0),
        ..plan_scale()
    };
    let c = Calibration::new(Some(s), Some(g), None).unwrap();
    let affine = c.georeference().unwrap().affine;
    assert!(affine.iter().all(|v| !(*v == 0.0 && v.is_sign_negative())));
    let u = c.scale().unwrap().uncertainty.unwrap();
    assert!(!u.is_sign_negative());
}
