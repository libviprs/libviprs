//! Issue #1311 (N6.2b, registry half): the mapping table checked against PVec's capability
//! registry. The registry is a dev-dependency, so `src/` never names it.
#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::mapping::NORMALIZED_TO_PVEC;
use pvec_capabilities::{CapabilityId, Class, Registry};

/// Registry geometry capabilities with no row, and why. A new PVec capability lands here or in
/// the table, or the second test fails.
const NOT_MAPPED: [(u16, &str); 5] = [
    (
        1,
        "cartesian_point: a building block of every curve, not a normalized kind",
    ),
    (
        2,
        "direction_vector: a building block of every curve, not a normalized kind",
    ),
    (
        3,
        "axis_placement_2d: carried inside circle and ellipse, not a normalized kind",
    ),
    (
        4,
        "line: only exists bounded, so the Line row uses trimmed_curve",
    ),
    (
        13,
        "affine_transform_2d: libviprs applies transforms before normalizing, no row",
    ),
];

#[test]
fn every_row_names_an_id_and_name_the_registry_has() {
    let reg = Registry::v1();
    for row in &NORMALIZED_TO_PVEC {
        let cap = reg
            .get(CapabilityId::new(u32::from(row.capability.id)))
            .unwrap_or_else(|| {
                panic!(
                    "{:?}: id {} is not in the registry",
                    row.kind, row.capability.id
                )
            });
        assert_eq!(
            cap.name, row.capability.name,
            "{:?}: name differs from the registry",
            row.kind
        );
        assert_eq!(
            cap.class,
            Class::Geometry,
            "{:?}: not a geometry capability",
            row.kind
        );
        assert!(
            !reg.is_retired(cap.id),
            "{:?}: id {} is retired",
            row.kind,
            cap.id
        );
    }
}

#[test]
fn every_registry_geometry_capability_is_mapped_or_explicitly_not() {
    let reg = Registry::v1();
    let used: Vec<u32> = NORMALIZED_TO_PVEC
        .iter()
        .map(|r| u32::from(r.capability.id))
        .collect();
    for cap in reg.entries().iter().filter(|c| c.class == Class::Geometry) {
        let listed = NOT_MAPPED
            .iter()
            .any(|(id, _)| u32::from(*id) == cap.id.get());
        assert!(
            used.contains(&cap.id.get()) || listed,
            "registry capability {} ({}) has no row and no NOT_MAPPED entry",
            cap.id,
            cap.name
        );
    }
}

#[test]
fn not_mapped_entries_are_real_unused_and_explained() {
    let reg = Registry::v1();
    let used: Vec<u32> = NORMALIZED_TO_PVEC
        .iter()
        .map(|r| u32::from(r.capability.id))
        .collect();
    for (id, reason) in &NOT_MAPPED {
        let id = u32::from(*id);
        assert!(
            reg.get(CapabilityId::new(id)).is_some(),
            "NOT_MAPPED id {id} is not in the registry"
        );
        assert!(!used.contains(&id), "NOT_MAPPED id {id} also has a row");
        assert!(
            !reason.trim().is_empty(),
            "NOT_MAPPED id {id} has no reason"
        );
    }
}
