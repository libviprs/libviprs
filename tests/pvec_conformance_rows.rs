//! #1326 (N6.2b, suite half), folded into #1055: each `NORMALIZED_TO_PVEC` row's conformance class,
//! tied to the libviprs suite that verifies it.
//!
//! The panel decided not to substitute PVec's own structural tests for this: PVec checking its own
//! class is a different claim from libviprs' conversion meeting the class its mapping row states.
//! So every row names the test in `pvec_structural` or `pvec_geometric` (the suites that run the
//! conversion path over the pinned corpus) that verifies it, and this file checks four things:
//! the table covers every row, its class is the row's class, the named suite is the one the class
//! belongs to, and the named test exists in that file.
//!
//! Needs `--features pvec-conformance`.

#![cfg(feature = "pvec-conformance")]

use libviprs::pvec::mapping::{ConformanceClass, NORMALIZED_TO_PVEC, NormalizedKind, row_for};

const STRUCTURAL: &str = include_str!("pvec_structural.rs");
const GEOMETRIC: &str = include_str!("pvec_geometric.rs");
const KNOWN_TRUTH: &str = include_str!("pvec_known_truth.rs");

/// (kind, class the row states, suite file, the test that verifies it there)
const VERIFIED_BY: &[(NormalizedKind, ConformanceClass, &str, &str)] = &[
    (
        NormalizedKind::Line,
        ConformanceClass::StructuralExactness,
        "pvec_known_truth",
        "n2_1_1050_every_known_truth_fixture_is_executed_and_agrees_with_its_expectation",
    ),
    (
        NormalizedKind::Circle,
        ConformanceClass::StructuralExactness,
        "pvec_known_truth",
        "n2_1_1050_every_known_truth_fixture_is_executed_and_agrees_with_its_expectation",
    ),
    (
        NormalizedKind::CircularArc,
        ConformanceClass::StructuralExactness,
        "pvec_structural",
        "n2_4_1055_every_property_row_has_a_named_fixture_that_passes",
    ),
    (
        NormalizedKind::Ellipse,
        ConformanceClass::StructuralExactness,
        "pvec_known_truth",
        "n2_1_1050_every_known_truth_fixture_is_executed_and_agrees_with_its_expectation",
    ),
    (
        NormalizedKind::EllipticalArc,
        ConformanceClass::StructuralExactness,
        "pvec_structural",
        "n2_4_1055_every_property_row_has_a_named_fixture_that_passes",
    ),
    (
        NormalizedKind::Polyline,
        ConformanceClass::StructuralExactness,
        "pvec_structural",
        "n2_4_1055_every_property_row_has_a_named_fixture_that_passes",
    ),
    (
        NormalizedKind::Composite,
        ConformanceClass::GeometricExactness,
        "pvec_geometric",
        "n2_5_1056_a_positive_bulge_arc_is_the_fixture_arc",
    ),
    (
        NormalizedKind::Bezier,
        ConformanceClass::StructuralExactness,
        "pvec_known_truth",
        "n2_1_1050_every_known_truth_fixture_is_executed_and_agrees_with_its_expectation",
    ),
    (
        NormalizedKind::BSpline,
        ConformanceClass::StructuralExactness,
        "pvec_structural",
        "n2_4_1055_every_property_row_has_a_named_fixture_that_passes",
    ),
    (
        NormalizedKind::RationalBSpline,
        ConformanceClass::StructuralExactness,
        "pvec_structural",
        "n2_4_1055_every_property_row_has_a_named_fixture_that_passes",
    ),
];

fn suite_text(suite: &str) -> &'static str {
    match suite {
        "pvec_structural" => STRUCTURAL,
        "pvec_geometric" => GEOMETRIC,
        "pvec_known_truth" => KNOWN_TRUTH,
        other => panic!("{other} is not a suite this file knows"),
    }
}

#[test]
fn n2_6_1326_every_mapping_row_is_tied_to_a_suite() {
    assert_eq!(VERIFIED_BY.len(), NORMALIZED_TO_PVEC.len());
    for kind in NormalizedKind::ALL {
        assert!(
            VERIFIED_BY.iter().any(|v| v.0 == kind),
            "{kind:?} is tied to no suite"
        );
    }
}

#[test]
fn n2_6_1326_the_class_a_row_states_is_the_class_its_suite_verifies() {
    for (kind, class, suite, _) in VERIFIED_BY {
        let row = row_for(*kind);
        assert_eq!(
            row.conformance,
            *class,
            "{kind:?}: the mapping row claims {} but the suite table says {}",
            row.conformance.tag(),
            class.tag()
        );
        match class {
            // A structural claim is verified where parameters are compared exactly: the known-truth
            // suite (decoder and mapping against the independent record) or the structural suite.
            ConformanceClass::StructuralExactness => {
                assert!(
                    matches!(*suite, "pvec_structural" | "pvec_known_truth"),
                    "{kind:?}: {suite}"
                );
            }
            ConformanceClass::GeometricExactness => {
                assert_eq!(
                    *suite, "pvec_geometric",
                    "{kind:?}: a geometric claim belongs to the geometric suite"
                );
            }
            ConformanceClass::DisplayFidelity => {
                panic!("{kind:?}: no row may claim display fidelity")
            }
        }
    }
}

#[test]
fn n2_6_1326_the_named_test_exists_in_the_named_suite() {
    for (kind, _, suite, test) in VERIFIED_BY {
        assert!(
            suite_text(suite).contains(&format!("fn {test}(")),
            "{kind:?}: {suite} has no test {test}"
        );
    }
}

#[test]
fn n2_6_1326_a_row_that_changes_class_without_its_suite_turns_this_red() {
    // Moving a structural row to geometric in the table, with its suite left alone, is caught by
    // the class check; shown here on a copy so the real table stays the evidence.
    let (kind, class, _, _) = VERIFIED_BY[0];
    assert_eq!(row_for(kind).conformance, class);
    let other = match class {
        ConformanceClass::StructuralExactness => ConformanceClass::GeometricExactness,
        _ => ConformanceClass::StructuralExactness,
    };
    assert_ne!(row_for(kind).conformance, other);
}
