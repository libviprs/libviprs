//! Issue #1378 (N4.5b, slice of #1079 and #1283): the provider's xref facet and the
//! open-and-flatten step of `XrefSession`.
//!
//! The drawings are written by acadrust's `DwgWriter` and read back through the provider, so the
//! path under test is the real one. Needs `--features "dwg-acadrust pvec-normalized"`.

#![cfg(all(feature = "dwg-acadrust", feature = "pvec-normalized"))]

use std::fs;
use std::path::Path;

use acadrust::entities::{Insert, Line};
use acadrust::tables::{BlockRecord, Layer};
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore, SkipReason};
use libviprs::pvec::limits::{SourceGuard, SourceLimits};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::xref::{XrefError, XrefOpened, XrefPolicy, XrefRef, XrefSession};
use libviprs::pvec::{
    Availability, DwgProvider, EntityScope, Facet, LayoutKind, ProviderError, RawEntity,
    RawGeometry,
};

fn xref_record(name: &str, path: &str, overlay: bool) -> BlockRecord {
    let mut r = BlockRecord::new(name);
    r.flags.is_xref = !overlay;
    r.flags.is_xref_overlay = overlay;
    r.xref_path = path.to_string();
    r
}

/// A drawing with one LINE from `(0,0)` to `(len,0)`, plus an INSERT of each `(block, path)` at x=10.
fn drawing(len: f64, xrefs: &[(&str, &str)]) -> Vec<u8> {
    let mut doc = CadDocument::new();
    doc.add_entity(EntityType::Line(Line::from_coords(
        0.0, 0.0, 0.0, len, 0.0, 0.0,
    )))
    .unwrap();
    if !xrefs.is_empty() {
        doc.layers.add(Layer::new("A-XREF")).unwrap();
    }
    for (name, path) in xrefs {
        doc.block_records
            .add(xref_record(name, path, false))
            .unwrap();
        let mut ins = Insert::new(*name, Vector3::new(10.0, 20.0, 0.0));
        ins.common.layer = "A-XREF".to_string();
        doc.add_entity(EntityType::Insert(ins)).unwrap();
    }
    DwgWriter::write_to_vec(&doc).expect("write")
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    fs::write(dir.join(name), bytes).unwrap();
}

fn open(path: &Path) -> AcadrustProvider {
    AcadrustProvider::open(CadSource::Path(path)).expect("open")
}

fn model(p: &AcadrustProvider) -> Vec<RawEntity> {
    let m = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    p.entities(&EntityScope::Layout(m.id))
        .unwrap()
        .collect_all()
        .unwrap()
}

fn guard() -> SourceGuard {
    SourceGuard::new(SourceLimits::default())
}

#[test]
fn pvec_1378_the_provider_lists_xref_block_records_and_claims_the_facet() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    let p = open(&dir.path().join("host.dwg"));
    assert_eq!(
        p.caps().availability(Facet::XrefAttachments),
        Availability::Reported
    );
    let xs = p.xref_blocks().unwrap();
    assert_eq!(xs.len(), 1, "{xs:?}");
    assert_eq!(xs[0].block_name, "WING");
    assert_eq!(xs[0].path, "wing.dwg");
    assert!(!xs[0].overlay);
}

#[test]
fn pvec_1378_a_drawing_with_no_xref_reports_an_empty_list_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "plain.dwg", &drawing(5.0, &[]));
    let p = open(&dir.path().join("plain.dwg"));
    assert!(p.xref_blocks().unwrap().is_empty());
}

#[test]
fn pvec_1378_discovery_builds_the_xref_ref_from_the_insert_and_the_block_record() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    let p = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&p, &model(&p)).unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    let x = &found[0];
    assert_eq!(x.name, "WING");
    assert_eq!(x.path, "wing.dwg");
    assert_eq!(x.layer, "A-XREF");
    assert_eq!(x.transform.coefficients()[4], 10.0);
    assert_eq!(x.transform.coefficients()[5], 20.0);
}

#[test]
fn pvec_1378_a_provider_without_the_facet_cannot_discover() {
    struct Blind;
    impl DwgProvider for Blind {
        fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
            Ok(false)
        }
        fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
            Ok(Blind)
        }
        fn name(&self) -> &'static str {
            "blind"
        }
        fn caps(&self) -> &libviprs::pvec::ProviderCaps {
            static CAPS: std::sync::OnceLock<libviprs::pvec::ProviderCaps> =
                std::sync::OnceLock::new();
            CAPS.get_or_init(libviprs::pvec::ProviderCaps::new)
        }
        fn layouts(&self) -> Result<Vec<libviprs::pvec::LayoutState>, ProviderError> {
            Ok(vec![])
        }
        fn layers(&self) -> Result<Vec<libviprs::pvec::LayerState>, ProviderError> {
            Ok(vec![])
        }
        fn entities<'a>(
            &'a self,
            _: &EntityScope,
        ) -> Result<Box<dyn libviprs::pvec::EntityCursor + 'a>, ProviderError> {
            Err(ProviderError::NoSuchScope("none".into()))
        }
    }
    let e = XrefRef::discover(&Blind, &[]).unwrap_err();
    assert_eq!(e, ProviderError::CannotReport(Facet::XrefAttachments));
}

#[test]
fn pvec_1378_the_flattener_reports_an_xref_insert_instead_of_expanding_it_to_nothing() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    let p = open(&dir.path().join("host.dwg"));
    let m = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let out = BlockResolver::new(&p, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(m.id),
            "host.dwg",
            "Model",
            &mut NoProvenance,
            &mut LineageStore::new(),
        )
        .unwrap();
    assert_eq!(out.entities.len(), 1, "only the host's own line");
    assert_eq!(out.skipped.len(), 1, "{:?}", out.skipped);
    assert_eq!(out.skipped[0].reason, SkipReason::ExternalReference);
    assert_eq!(out.skipped[0].block_name, "WING");
}

fn flatten_len(o: &XrefOpened) -> usize {
    match o {
        XrefOpened::Flattened(f) => f.flattened.entities.len(),
        other => panic!("expected a flattened xref, got {other:?}"),
    }
}

#[test]
fn pvec_1378_resolve_opens_and_flattens_the_attached_drawing() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    write(dir.path(), "wing.dwg", &drawing(7.0, &[]));
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let mut g = guard();
    let opened = s
        .open_and_flatten::<AcadrustProvider>(&mut g, &found[0], FlattenBounds::default())
        .unwrap();
    assert_eq!(flatten_len(&opened), 1);
    let XrefOpened::Flattened(f) = opened else {
        unreachable!()
    };
    assert_eq!(f.name, "WING");
    assert_eq!(f.depth, 1);
    assert!(f.nested.is_empty());
    let RawGeometry::Line { end, .. } = &f.flattened.entities[0].entity.geometry else {
        panic!("{:?}", f.flattened.entities[0].entity.geometry)
    };
    assert_eq!(
        end[0], 7.0,
        "the attached drawing's own line, not the host's"
    );
    assert_eq!(s.report().resolved, vec!["WING".to_string()]);
    assert_eq!(
        g.depth(),
        0,
        "the depth slot is released when the walk is done"
    );
}

#[test]
fn pvec_1378_external_dependency_opens_nothing_and_leaves_a_row() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::ExternalDependency);
    let o = s
        .open_and_flatten::<AcadrustProvider>(&mut guard(), &found[0], FlattenBounds::default())
        .unwrap();
    assert!(
        matches!(o, XrefOpened::Omitted(0)),
        "wing.dwg does not exist and was not touched"
    );
    assert_eq!(s.report().unresolved[0].name, "WING");
}

#[test]
fn pvec_1378_a_file_that_is_not_a_drawing_is_a_typed_error_and_frees_the_slot() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    write(dir.path(), "wing.dwg", b"this is not a drawing");
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let mut g = guard();
    let e = s
        .open_and_flatten::<AcadrustProvider>(&mut g, &found[0], FlattenBounds::default())
        .unwrap_err();
    assert!(
        matches!(e, XrefError::Open { ref name, .. } if name == "WING"),
        "{e:?}"
    );
    assert_eq!(g.depth(), 0);
}

#[test]
fn pvec_1378_nested_xrefs_are_flattened_one_level_deeper() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    write(
        dir.path(),
        "wing.dwg",
        &drawing(7.0, &[("ANNEX", "annex.dwg")]),
    );
    write(dir.path(), "annex.dwg", &drawing(9.0, &[]));
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let mut g = guard();
    let XrefOpened::Flattened(f) = s
        .open_and_flatten::<AcadrustProvider>(&mut g, &found[0], FlattenBounds::default())
        .unwrap()
    else {
        panic!("expected a flattened xref")
    };
    assert_eq!(f.nested.len(), 1);
    assert_eq!(f.nested[0].name, "ANNEX");
    assert_eq!(f.nested[0].depth, 2);
    assert_eq!(f.nested[0].flattened.entities.len(), 1);
    assert_eq!(
        s.report().resolved,
        vec!["WING".to_string(), "ANNEX".to_string()]
    );
    assert_eq!(g.depth(), 0);
}

#[test]
fn pvec_1378_a_cycle_through_the_attached_drawings_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    write(
        dir.path(),
        "wing.dwg",
        &drawing(7.0, &[("BACK", "host.dwg")]),
    );
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let e = s
        .open_and_flatten::<AcadrustProvider>(&mut guard(), &found[0], FlattenBounds::default())
        .unwrap_err();
    assert!(matches!(e, XrefError::Cycle { .. }), "{e:?}");
}

#[test]
fn pvec_1378_depth_past_the_shared_limit_names_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "host.dwg",
        &drawing(5.0, &[("WING", "wing.dwg")]),
    );
    write(
        dir.path(),
        "wing.dwg",
        &drawing(7.0, &[("ANNEX", "annex.dwg")]),
    );
    write(dir.path(), "annex.dwg", &drawing(9.0, &[]));
    let host = open(&dir.path().join("host.dwg"));
    let found = XrefRef::discover(&host, &model(&host)).unwrap();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 1;
    let e = s
        .open_and_flatten::<AcadrustProvider>(
            &mut SourceGuard::new(limits),
            &found[0],
            FlattenBounds::new(limits, 1_000_000),
        )
        .unwrap_err();
    assert!(matches!(e, XrefError::Limit(_)), "{e:?}");
}
