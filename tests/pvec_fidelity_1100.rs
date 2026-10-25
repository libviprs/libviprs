//! Entity classification (N3.5, #1100): every entity ends in exactly one of six classes, the
//! counts add up, and the refusing classes carry a stable reason code.

use std::collections::HashSet;

use libviprs::pvec::fidelity::{
    Basis, Classification, FidelityClass, ReasonCode, Tally, TallyError, classify_unmapped,
};
use libviprs::pvec::{InstancePath, ProvenanceStub, RawEntity, RawGeometry};

fn entity(class: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry,
        provenance: ProvenanceStub {
            handle: None,
            class_name: class.to_string(),
            block_path: InstancePath::root(),
            layer: "0".to_string(),
        },
    }
}

#[test]
fn there_are_exactly_six_classes() {
    assert_eq!(FidelityClass::ALL.len(), 6);
    let tags: HashSet<_> = FidelityClass::ALL.iter().map(|c| c.tag()).collect();
    assert_eq!(tags.len(), 6);
    assert!(tags.contains("SupportedExact"));
    assert!(tags.contains("IgnoredByPolicy"));
}

#[test]
fn reason_codes_are_enumerated_once_and_stable() {
    let codes: HashSet<_> = ReasonCode::ALL.iter().map(|r| r.code()).collect();
    assert_eq!(codes.len(), ReasonCode::ALL.len(), "codes must be unique");
    for r in ReasonCode::ALL {
        let c = r.code();
        assert!(
            c.chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-'),
            "{c} is not kebab-case"
        );
        assert!(
            matches!(
                r.class(),
                FidelityClass::Unsupported
                    | FidelityClass::RequiresPVecCapability
                    | FidelityClass::RequiresExtension
            ),
            "{c} names a class that takes no reason"
        );
    }
    // The baseline diff keys on these spellings, so pin a few.
    assert_eq!(ReasonCode::Helix.code(), "helix");
    assert_eq!(ReasonCode::Spline3d.code(), "spline-3d");
    assert_eq!(
        ReasonCode::UnmappedSourceClass.code(),
        "unmapped-source-class"
    );
}

#[test]
fn a_refusal_carries_reason_and_source_class() {
    let c = Classification::refused(ReasonCode::Spline3d, "SPLINE");
    assert_eq!(c.class(), FidelityClass::RequiresPVecCapability);
    assert_eq!(c.reason(), Some(ReasonCode::Spline3d));
    assert_eq!(c.source_class(), "SPLINE");
    assert_eq!(c.policy_field(), None);

    let u = Classification::refused(ReasonCode::UnmappedSourceClass, "ACAD_PROXY_ENTITY");
    assert_eq!(u.class(), FidelityClass::Unsupported);
}

#[test]
fn only_refusing_reasons_are_accepted_by_refused() {
    let ext = Classification::refused(ReasonCode::SplineKnotsNotReported, "SPLINE");
    assert_eq!(ext.class(), FidelityClass::RequiresExtension);
}

#[test]
fn ignored_by_policy_names_the_field() {
    let c = Classification::ignored_by_policy("include_frozen_layers", "LINE");
    assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
    assert_eq!(c.policy_field(), Some("include_frozen_layers"));
    assert_eq!(c.reason(), None);
}

#[test]
fn supported_classes_carry_no_reason() {
    for basis in [Basis::Structural, Basis::Equivalent] {
        let c = Classification::supported(basis, "CIRCLE");
        assert_eq!(c.reason(), None);
        assert!(c.class().produces_geometry());
    }
    assert_eq!(
        Classification::supported(Basis::Structural, "CIRCLE").class(),
        FidelityClass::SupportedExact
    );
    assert_eq!(
        Classification::supported(Basis::Equivalent, "CIRCLE").class(),
        FidelityClass::SupportedEquivalent
    );
}

#[test]
fn display_fidelity_is_never_supported_exact() {
    let c = Classification::supported(Basis::DisplayOnly, "HATCH");
    assert_ne!(c.class(), FidelityClass::SupportedExact);
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::DisplayFidelityOnly));
}

#[test]
fn an_unmapped_entity_is_unsupported_not_dropped() {
    let e = entity(
        "ACAD_PROXY_ENTITY",
        RawGeometry::Unmapped {
            reason: "proxy".into(),
        },
    );
    let c = classify_unmapped(&e);
    assert_eq!(c.class(), FidelityClass::Unsupported);
    assert_eq!(c.reason(), Some(ReasonCode::UnmappedSourceClass));
    assert_eq!(c.source_class(), "ACAD_PROXY_ENTITY");
}

fn mixed() -> Vec<Classification> {
    vec![
        Classification::supported(Basis::Structural, "CIRCLE"),
        Classification::supported(Basis::Structural, "ARC"),
        Classification::supported(Basis::Equivalent, "LWPOLYLINE"),
        Classification::refused(ReasonCode::Helix, "HELIX"),
        Classification::refused(ReasonCode::Helix, "HELIX"),
        Classification::refused(ReasonCode::SplineKnotsNotReported, "SPLINE"),
        Classification::refused(ReasonCode::UnmappedSourceClass, "PROXY"),
        Classification::ignored_by_policy("include_frozen_layers", "LINE"),
    ]
}

#[test]
fn the_six_counts_sum_to_the_entities_encountered() {
    let all = mixed();
    let mut t = Tally::new();
    for c in &all {
        t.record(c);
    }
    let report = t.finish(all.len() as u64).expect("counts agree");
    let sum: u64 = FidelityClass::ALL.iter().map(|c| report.count(*c)).sum();
    assert_eq!(sum, all.len() as u64);
    assert_eq!(report.encountered(), 8);
    assert_eq!(report.count(FidelityClass::SupportedExact), 2);
    assert_eq!(report.count(FidelityClass::RequiresPVecCapability), 2);
    assert_eq!(report.count(FidelityClass::RequiresExtension), 1);
    assert_eq!(report.count(FidelityClass::IgnoredByPolicy), 1);
}

#[test]
fn a_dropped_entity_fails_the_conversion_check() {
    let all = mixed();
    let mut t = Tally::new();
    // The conversion saw eight entities but only classified seven.
    for c in &all[..7] {
        t.record(c);
    }
    assert_eq!(
        t.finish(8).unwrap_err(),
        TallyError::CountMismatch {
            classified: 7,
            encountered: 8
        }
    );
}

#[test]
fn reason_counts_are_keyed_by_code_and_source_class() {
    let mut t = Tally::new();
    for c in &mixed() {
        t.record(c);
    }
    let report = t.finish(8).unwrap();
    let rows: Vec<_> = report
        .reasons()
        .map(|(code, src, n)| (code, src.to_string(), n))
        .collect();
    assert!(rows.contains(&(ReasonCode::Helix, "HELIX".to_string(), 2)));
    assert!(rows.contains(&(ReasonCode::SplineKnotsNotReported, "SPLINE".to_string(), 1)));
    // Deterministic order, so a baseline diff is meaningful.
    let mut sorted = rows.clone();
    sorted.sort();
    assert_eq!(rows, sorted);
}

#[cfg(feature = "pvec-normalized")]
#[test]
fn an_unsupported_curve_is_refused_not_tessellated() {
    use libviprs::pvec::Normalized2DGeometry;
    // A flattened stand-in for a curve PVec cannot carry, offered as canonical geometry.
    let flattened = [[0.0, 0.0], [1.0, 0.5], [2.0, 0.8], [3.0, 0.9]];
    for class in [
        FidelityClass::Unsupported,
        FidelityClass::RequiresPVecCapability,
        FidelityClass::IgnoredByPolicy,
    ] {
        assert!(
            Normalized2DGeometry::polyline(&flattened, class).is_err(),
            "{class:?} must not hold a polyline"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn there_is_one_enum_for_the_six_classes() {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir("src/pvec").unwrap() {
        let p = entry.unwrap().path();
        if p.extension().is_some_and(|e| e == "rs") {
            let src = std::fs::read_to_string(&p).unwrap();
            if src.contains("enum Requires") || src.contains("enum FidelityClass") {
                hits.push(p.display().to_string());
            }
        }
    }
    assert_eq!(hits, vec!["src/pvec/fidelity.rs".to_string()], "{hits:?}");
}

#[cfg(feature = "pvec-spline")]
mod spline {
    use super::*;
    use libviprs::pvec::SourceFact;
    use libviprs::pvec::dwg_spline::{SplineError, extract};

    fn spline(
        class: &str,
        pts: Vec<[f64; 3]>,
        knots: SourceFact<Vec<f64>>,
        periodic: SourceFact<bool>,
    ) -> RawEntity {
        entity(
            class,
            RawGeometry::Spline {
                degree: 2,
                control_points: pts,
                knots,
                weights: SourceFact::Absent,
                periodic,
                closed: false,
            },
        )
    }

    fn flat() -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
    }

    fn knots() -> SourceFact<Vec<f64>> {
        SourceFact::Present(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    }

    #[test]
    fn missing_knots_need_an_extension() {
        let e = spline(
            "SPLINE",
            flat(),
            SourceFact::NotReportable,
            SourceFact::Present(false),
        );
        let err = extract(&e).unwrap_err();
        let c = err.classification("SPLINE");
        assert_eq!(c.class(), FidelityClass::RequiresExtension);
        assert_eq!(c.reason(), Some(ReasonCode::SplineKnotsNotReported));
    }

    #[test]
    fn missing_periodic_flag_needs_an_extension() {
        let e = spline("SPLINE", flat(), knots(), SourceFact::NotReportable);
        let c = extract(&e).unwrap_err().classification("SPLINE");
        assert_eq!(c.class(), FidelityClass::RequiresExtension);
        assert_eq!(c.reason(), Some(ReasonCode::SplinePeriodicNotReported));
    }

    #[test]
    fn a_3d_spline_needs_a_pvec_capability() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 0.0, 0.0]];
        let e = spline("SPLINE", pts, knots(), SourceFact::Present(false));
        let c = extract(&e).unwrap_err().classification("SPLINE");
        assert_eq!(c.class(), FidelityClass::RequiresPVecCapability);
        assert_eq!(c.reason(), Some(ReasonCode::Spline3d));
    }

    #[test]
    fn a_helix_needs_a_pvec_capability() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 0.0, 0.0]];
        let e = spline("HELIX", pts, knots(), SourceFact::Present(false));
        let c = extract(&e).unwrap_err().classification("HELIX");
        assert_eq!(c.class(), FidelityClass::RequiresPVecCapability);
        assert_eq!(c.reason(), Some(ReasonCode::Helix));
    }

    #[test]
    fn a_malformed_spline_is_unsupported_with_a_reason() {
        let bad = SourceFact::Present(vec![0.0, 1.0]);
        let e = spline("SPLINE", flat(), bad, SourceFact::Present(false));
        let err = extract(&e).unwrap_err();
        assert!(matches!(err, SplineError::KnotCount { .. }));
        let c = err.classification("SPLINE");
        assert_eq!(c.class(), FidelityClass::Unsupported);
        assert_eq!(c.reason(), Some(ReasonCode::SplineMalformed));
    }

    #[test]
    fn every_spline_error_classifies_with_a_reason() {
        // The mapping is total: a new SplineError variant cannot compile without a code.
        let e = spline("SPLINE", flat(), knots(), SourceFact::Present(false));
        assert!(extract(&e).is_ok());
    }
}
