//! Issue #1414 (slice of #1048, N7.2): the allowed-capability gate, its three policies, the three
//! profiles and `probe`. Needs `--features pvec-convert`.
//!
//! The drawing is a mock provider so the entities are known exactly: a line (handle 1, needs
//! `trimmed_curve`, 8, and extraction classes it `SupportedEquivalent`) and two circles (handles 2
//! and 4, needs `circle`, 6, `SupportedExact`).

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    FidelityClass, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceUnit,
};
use libviprs::pvec_convert::{
    CapabilityPolicy, ConversionProfile, PvecConversion, PvecConvertError,
};
use pvec_facade::{CapabilityId, Curve};

struct Mock {
    entities: Vec<RawEntity>,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn entity(handle: u64, class: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry,
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: class.into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

fn line() -> RawEntity {
    entity(
        1,
        "LINE",
        RawGeometry::Line {
            start: [0.0, 0.0, 0.0],
            end: [10.0, 0.0, 0.0],
        },
    )
}

fn circle() -> RawEntity {
    entity(
        2,
        "CIRCLE",
        RawGeometry::Circle {
            center: [5.0, 5.0, 0.0],
            radius: 2.0,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

fn second_circle() -> RawEntity {
    entity(
        4,
        "CIRCLE",
        RawGeometry::Circle {
            center: [20.0, 5.0, 0.0],
            radius: 1.0,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

impl DwgProvider for Mock {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK")))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let CadSource::Bytes(b) = source else {
            return Err(ProviderError::Unrecognised {
                provider: "mock",
                detail: "bytes only".into(),
            });
        };
        let entities = match b {
            b"MOCK-exact" => vec![circle(), second_circle()],
            _ => vec![line(), circle(), second_circle()],
        };
        let caps = Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
            c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
        });
        Ok(Mock { entities, caps })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }])
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            handle: None,
            off: false,
            frozen: false,
            locked: false,
        }])
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        match scope {
            EntityScope::Layout(_) => Ok(Box::new(Cursor(self.entities.clone().into_iter()))),
            _ => Err(ProviderError::NoSuchScope("no blocks".into())),
        }
    }
}

const MIXED: &[u8] = b"MOCK-mixed";
const EXACT: &[u8] = b"MOCK-exact";

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1414-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn conversion<'a>(bytes: &'a [u8], dest: &'a std::path::Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Bytes(bytes), dest).unit(SourceUnit::Meters)
}

/// Every capability but the circle's (6).
fn without_circle() -> Vec<CapabilityId> {
    ConversionProfile::Default
        .capabilities()
        .into_iter()
        .filter(|c| c.get() != 6)
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_reject_input_names_the_capability_and_the_entity_and_leaves_no_file() {
    let dir = scratch("reject-input");
    let dest = dir.join("o.pvec");
    let err = conversion(MIXED, &dest)
        .allowed_capabilities(without_circle())
        .capability_policy(CapabilityPolicy::RejectInput)
        .convert_with::<Mock>()
        .unwrap_err();
    let PvecConvertError::CapabilityNotAllowed {
        capability,
        name,
        handle,
        ..
    } = &err
    else {
        panic!("{err:?}")
    };
    assert_eq!((capability.get(), *name, *handle), (6, "circle", Some(2)));
    assert!(!dest.exists() && !report_path(&dest).exists());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_reject_entity_converts_leaves_the_entity_out_and_counts_it() {
    let dir = scratch("reject-entity");
    let dest = dir.join("o.pvec");
    let out = conversion(MIXED, &dest)
        .allowed_capabilities(without_circle())
        .capability_policy(CapabilityPolicy::RejectEntity)
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.report.totals.by_fidelity["RequiresPVecCapability"], 2);
    for handle in [2, 4] {
        let row = out
            .report
            .entities
            .iter()
            .find(|r| r.source_handle == Some(handle))
            .unwrap();
        assert_eq!(row.reason.as_deref(), Some("capability-not-allowed"));
        assert!(row.semantic_type.is_none());
    }
    // The artifact, not the in-memory class: no circle was written, and nothing stands in for it.
    let entities = open_published(&dest).unwrap().entities().unwrap();
    assert_eq!(entities.len() as u64, out.written);
    assert_eq!(entities.len(), 1);
    assert!(entities.iter().all(|(_, c)| !matches!(c, Curve::Circle(_))));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_probe_and_convert_return_the_same_counts_field_by_field() {
    let dir = scratch("probe");
    let dest = dir.join("o.pvec");
    fn build(dest: &std::path::Path) -> PvecConversion<'_> {
        conversion(MIXED, dest)
            .allowed_capabilities(without_circle())
            .capability_policy(CapabilityPolicy::RejectEntity)
    }
    let probed = build(&dest).probe_with::<Mock>().unwrap();
    assert!(!dest.exists(), "a probe writes nothing");
    let out = build(&dest).convert_with::<Mock>().unwrap();
    assert_eq!(probed.encountered(), out.fidelity.encountered());
    for class in FidelityClass::ALL {
        assert_eq!(probed.count(class), out.fidelity.count(class), "{class:?}");
    }
    assert_eq!(
        probed.reasons().collect::<Vec<_>>(),
        out.fidelity.reasons().collect::<Vec<_>>()
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_strict_refuses_an_equivalent_entity_and_accepts_the_all_exact_drawing() {
    let dir = scratch("strict");
    let dest = dir.join("o.pvec");
    let err = conversion(MIXED, &dest)
        .profile(ConversionProfile::Strict)
        .convert_with::<Mock>()
        .unwrap_err();
    let PvecConvertError::CapabilityNotAllowed { name, handle, .. } = &err else {
        panic!("{err:?}")
    };
    assert_eq!((*name, *handle), ("trimmed_curve", Some(1)));
    assert!(!dest.exists());

    let out = conversion(EXACT, &dest)
        .profile(ConversionProfile::Strict)
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.written, 2);
    assert_eq!(out.report.totals.by_fidelity["SupportedEquivalent"], 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_downgrade_records_that_none_exists_and_never_tessellates() {
    let dir = scratch("downgrade");
    let dest = dir.join("o.pvec");
    let out = conversion(MIXED, &dest)
        .allowed_capabilities(without_circle())
        .capability_policy(CapabilityPolicy::Downgrade)
        .convert_with::<Mock>()
        .unwrap();
    let row = out
        .report
        .entities
        .iter()
        .find(|r| r.source_handle == Some(2))
        .unwrap();
    assert!(
        row.warnings
            .iter()
            .any(|w| w.contains("downgrade-unavailable")),
        "{:?}",
        row.warnings
    );
    let entities = open_published(&dest).unwrap().entities().unwrap();
    // A tessellated circle would come back as a polyline.
    assert_eq!(entities.len(), 1);
    assert!(
        entities
            .iter()
            .all(|(_, c)| !matches!(c, Curve::Circle(_) | Curve::Polyline(_)))
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_the_profiles_are_written_down_once_and_the_report_records_the_one_used() {
    let strict = ConversionProfile::Strict.capabilities();
    let default = ConversionProfile::Default.capabilities();
    let permissive = ConversionProfile::Permissive.capabilities();
    assert!(strict.is_subset(&default) && strict != default);
    assert!(
        !strict.iter().any(|c| c.get() == 9),
        "composite_curve is geometric, not exact"
    );
    assert_eq!(default, permissive);
    assert_eq!(default.len(), 8, "ids 5 to 12");

    let dir = scratch("profile");
    let dest = dir.join("o.pvec");
    let out = conversion(EXACT, &dest)
        .profile(ConversionProfile::Permissive)
        .convert_with::<Mock>()
        .unwrap();
    let c = out.report.capabilities.as_ref().unwrap();
    assert_eq!(
        (c.profile.as_str(), c.policy.as_str()),
        ("Permissive", "Downgrade")
    );
    assert_eq!(
        c.allowed,
        permissive.iter().map(|c| c.get()).collect::<Vec<_>>()
    );
    let on_disk = fs::read_to_string(report_path(&dest)).unwrap();
    assert!(on_disk.contains("\"capabilities\""));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1414_the_capability_type_is_pvecs_own_and_no_local_copy_exists() {
    assert!(
        std::any::type_name::<CapabilityId>().contains("pvec_capabilities"),
        "{}",
        std::any::type_name::<CapabilityId>()
    );
    let src = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/pvec_convert/capability.rs"
    ))
    .unwrap();
    assert!(!src.contains("struct CapabilityId") && !src.contains("type CapabilityId"));
    assert!(src.contains("writer::CapabilityId"));
}
