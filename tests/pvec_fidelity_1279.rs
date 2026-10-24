//! N3.5b (#1279): the view profile hands its field names to the classification, and the
//! unresolved-block reason is a code of its own.

use libviprs::pvec::fidelity::Classification;
use libviprs::pvec::{Exclusion, FidelityClass, LayerStateRule, ReasonCode, ViewProfile};

#[test]
fn n3_5b_1279_every_exclusion_names_a_field_the_profile_has() {
    let knobs: Vec<_> = ViewProfile::default_profile()
        .knobs()
        .into_iter()
        .map(|k| k.field)
        .collect();
    for e in Exclusion::ALL {
        assert!(knobs.contains(&e.field()), "{e:?} names {}", e.field());
    }
}

#[test]
fn n3_5b_1279_default_profile_names_every_field_that_can_exclude() {
    let mut fields = ViewProfile::default_profile().exclusion_fields();
    fields.sort_unstable();
    assert_eq!(
        fields,
        vec![
            "layers.frozen",
            "layers.off",
            "nested_inserts.max_depth",
            "xrefs"
        ]
    );
}

#[test]
fn n3_5b_1279_a_rule_set_to_ignore_can_no_longer_exclude() {
    let mut p = ViewProfile::default_profile();
    p.layers.frozen = LayerStateRule::Ignore;
    let fields = p.exclusion_fields();
    assert!(!fields.contains(&"layers.frozen"));
    assert!(fields.contains(&"layers.off"));
}

#[test]
fn n3_5b_1279_policy_exclusions_classify_with_the_profile_field() {
    for e in [
        Exclusion::LayerOff,
        Exclusion::LayerFrozen,
        Exclusion::DepthLimit,
    ] {
        let c = e.classification("LINE");
        assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
        assert_eq!(c.policy_field(), Some(e.field()));
        assert_eq!(c.source_class(), "LINE");
    }
}

#[test]
fn n3_5b_1279_unresolved_block_is_unsupported_with_its_own_code() {
    let c = Exclusion::Unresolved.classification("INSERT");
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::UnresolvedBlock));
    assert_eq!(ReasonCode::UnresolvedBlock.code(), "unresolved-block");
    assert!(ReasonCode::ALL.contains(&ReasonCode::UnresolvedBlock));
}

#[test]
fn n3_5b_1279_classification_agrees_with_exclusion_fidelity() {
    for e in Exclusion::ALL {
        assert_eq!(e.classification("X").class(), e.fidelity(), "{e:?}");
    }
}

#[test]
fn n3_5b_1279_ignored_by_policy_rejects_a_field_the_profile_lacks() {
    let p = ViewProfile::default_profile();
    assert!(Classification::ignored_by_profile(&p, "layers.off", "LINE").is_some());
    assert!(Classification::ignored_by_profile(&p, "include_frozen_layers", "LINE").is_none());
}
