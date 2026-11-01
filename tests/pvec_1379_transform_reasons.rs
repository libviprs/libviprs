//! Issue #1379 (N3.5c, slice of #1079 and #1323): the three frozen transform refusals are
//! reason codes of the fidelity classification, so a refused INSERT map classifies with a code
//! of its own and never with a generic one.
#![cfg(feature = "pvec-normalized")]

use std::collections::HashSet;

use libviprs::cad::ItemHandle;
use libviprs::pvec::transform::{ComposedTransform, TransformReason};
use libviprs::pvec::{FidelityClass, InstanceStep, ReasonCode, SourceTransform};

const ALL: [TransformReason; 3] = [
    TransformReason::NonFinite,
    TransformReason::Singular,
    TransformReason::ConicIllConditioned,
];

fn step() -> InstanceStep {
    InstanceStep {
        block_name: "B".to_string(),
        insert_handle: ItemHandle::new(7),
    }
}

#[test]
fn each_transform_refusal_has_a_reason_code_that_is_enumerated() {
    for t in ALL {
        let r = t.reason_code();
        assert!(
            ReasonCode::ALL.contains(&r),
            "{t:?} -> {r:?} is not in ReasonCode::ALL"
        );
        assert_ne!(
            r,
            ReasonCode::UnmappedSourceClass,
            "{t:?} needs a code of its own"
        );
        assert_eq!(r.class(), FidelityClass::Unsupported, "{t:?}");
    }
}

#[test]
fn the_three_codes_are_distinct_and_spelled_after_the_diagnostic() {
    let codes: HashSet<_> = ALL.iter().map(|t| t.reason_code()).collect();
    assert_eq!(codes.len(), 3, "each refusal needs a code of its own");
    for t in ALL {
        let expected = t.code().to_ascii_lowercase().replace('_', "-");
        assert_eq!(t.reason_code().code(), expected, "{t:?}");
    }
}

#[test]
fn a_singular_insert_map_classifies_with_transform_singular() {
    let singular = SourceTransform::new([1.0, 2.0, 2.0, 4.0, 0.0, 0.0]).unwrap();
    let err = ComposedTransform::compose([(step(), singular)]).unwrap_err();
    let c = err.classification("INSERT");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::TransformSingular));
    assert_eq!(c.source_class(), "INSERT");
}

#[test]
fn a_non_finite_insert_map_classifies_with_transform_non_finite() {
    let nonfinite = ComposedTransform::identity()
        .push_inner_coefficients(step(), [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap_err();
    let c = nonfinite.classification("INSERT");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::TransformNonFinite));
}
