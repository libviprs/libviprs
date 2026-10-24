//! Issue #1096 (N3.2): source provenance records, held in a side store.
//!
//! One test per acceptance criterion. The geometry comparison is done on PVec's own `Debug`
//! rendering of the curve and box, so it is a byte comparison and not a tolerance.
#![cfg(feature = "pvec-normalized")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::provenance::{
    MAX_BLOCK_PATH_DEPTH, MAX_NAME_BYTES, MAX_WARNING_BYTES, MAX_WARNINGS, NoProvenance,
    Provenance, ProvenanceSink, ProvenanceStore,
};
use libviprs::pvec::{
    FidelityClass, InstancePath, Normalized2DGeometry, ProvenanceStub, SourceTransform,
};

const SRC: &str = include_str!("../src/pvec/provenance.rs");

fn stub(handle: u64, class: &str, layer: &str, path: &InstancePath) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(ItemHandle::new(handle)),
        class_name: class.into(),
        block_path: path.clone(),
        layer: layer.into(),
    }
}

fn xf(tx: f64) -> SourceTransform {
    SourceTransform::new([1.0, 0.0, 0.0, 1.0, tx, 0.0]).unwrap()
}

/// A block-heavy drawing: a block "TREE" holding a circle and a polyline, instanced six times,
/// two of them nested one level inside "ROOM". Returns what a normaliser would produce, with
/// provenance recorded through whichever sink it is given.
fn run<S: ProvenanceSink>(sink: &mut S) -> Vec<Normalized2DGeometry> {
    let mut out = Vec::new();
    let room = InstancePath::root().child("ROOM", ItemHandle::new(0x10));
    let mut paths = vec![room.child("TREE", ItemHandle::new(0x21))];
    paths.push(room.child("TREE", ItemHandle::new(0x22)));
    for i in 0..4u64 {
        paths.push(InstancePath::root().child("TREE", ItemHandle::new(0x30 + i)));
    }
    for (i, path) in paths.iter().enumerate() {
        let shift = i as f64 * 10.0;
        for (class, circle) in [("CIRCLE", true), ("LWPOLYLINE", false)] {
            let s = stub(0x4C, class, "TREES", path);
            let h = sink.record(|| Provenance::from_stub(&s, "plan.dwg", "Model", xf(shift)));
            let g = if circle {
                Normalized2DGeometry::circle(
                    [shift, 1.0],
                    [1.0, 0.0],
                    2.5,
                    FidelityClass::SupportedExact,
                )
            } else {
                Normalized2DGeometry::polyline(
                    &[[shift, 0.0], [shift + 1.0, 3.0]],
                    FidelityClass::SupportedExact,
                )
            }
            .unwrap();
            sink.warn(h, "mapped with a default ref direction");
            out.push(match h {
                Some(h) => g.with_provenance(h),
                None => g,
            });
        }
    }
    out
}

fn geometry_bytes(records: &[Normalized2DGeometry]) -> String {
    records
        .iter()
        .map(|r| format!("{:?}|{:?}|{:?}", r.geometry(), r.source_bounds(), r.fidelity()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn geometry_is_byte_identical_with_provenance_on_and_off() {
    let mut store = ProvenanceStore::new();
    let on = run(&mut store);
    let off = run(&mut NoProvenance);
    assert_eq!(on.len(), 12);
    assert_eq!(geometry_bytes(&on), geometry_bytes(&off));
    assert!(on.iter().all(|r| r.provenance().is_some()));
    assert!(off.iter().all(|r| r.provenance().is_none()));
    assert_eq!(store.len(), 12);
}

#[test]
fn off_mode_never_builds_a_record() {
    let mut built = 0;
    let h = NoProvenance.record(|| {
        built += 1;
        Provenance::from_stub(&stub(1, "LINE", "0", &InstancePath::root()), "a", "b", xf(0.0))
    });
    assert!(h.is_none());
    assert_eq!(built, 0, "the closure must not run when provenance is off");
}

#[test]
fn geometry_record_grows_by_at_most_one_optional_index() {
    // The record holds the handle and nothing else for provenance: no owned payload type.
    assert!(std::mem::size_of::<Option<libviprs::pvec::ProvenanceHandle>>() <= 16);
    let body: String = SRC
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !body.contains("Normalized2DGeometry {"),
        "provenance.rs must not define or extend the geometry record"
    );
}

#[test]
fn block_path_is_root_to_leaf_and_tells_two_instances_apart() {
    let mut store = ProvenanceStore::new();
    let outer = InstancePath::root().child("OUTER", ItemHandle::new(0x58));
    let a = outer.child("INNER", ItemHandle::new(0x60));
    let b = outer.child("INNER", ItemHandle::new(0x61));
    let ha = store
        .record(|| Provenance::from_stub(&stub(0x4C, "LINE", "L", &a), "p.dwg", "Model", xf(1.0)))
        .unwrap();
    let hb = store
        .record(|| Provenance::from_stub(&stub(0x4C, "LINE", "L", &b), "p.dwg", "Model", xf(2.0)))
        .unwrap();
    let (pa, pb) = (store.get(ha).unwrap(), store.get(hb).unwrap());
    let names: Vec<_> = pa.block_path().steps().iter().map(|s| s.block_name.as_str()).collect();
    assert_eq!(names, ["OUTER", "INNER"]);
    assert_eq!(pa.block_path().to_string(), "OUTER@58/INNER@60");
    assert_ne!(pa.block_path(), pb.block_path());
    assert_eq!(pa.handle(), pb.handle(), "same leaf handle, different instance");
    assert_ne!(ha, hb);
}

#[test]
fn record_carries_every_field_the_issue_lists() {
    let mut store = ProvenanceStore::new();
    let path = InstancePath::root().child("B", ItemHandle::new(7));
    let h = store
        .record(|| Provenance::from_stub(&stub(0xAB, "ARC", "WALLS", &path), "plan.dwg", "Layout1", xf(5.0)))
        .unwrap();
    let p = store.get(h).unwrap();
    assert_eq!(p.source(), "plan.dwg");
    assert_eq!(p.layout(), "Layout1");
    assert_eq!(p.handle(), Some(ItemHandle::new(0xAB)));
    assert_eq!(p.class_name(), "ARC");
    assert_eq!(p.layer(), "WALLS");
    assert_eq!(p.transform().coefficients()[4], 5.0);
    assert!(p.warnings().is_empty());
}

#[test]
fn handles_are_a_dense_index() {
    let mut store = ProvenanceStore::new();
    for i in 0..5u64 {
        let h = store
            .record(|| Provenance::from_stub(&stub(i, "LINE", "0", &InstancePath::root()), "a", "b", xf(0.0)))
            .unwrap();
        assert_eq!(h.get(), i);
    }
    assert!(store.get(libviprs::pvec::ProvenanceHandle::new(5)).is_none());
}

#[test]
fn string_fields_are_truncated_at_the_documented_bound() {
    let long = "x".repeat(10_000_000);
    let multibyte = "é".repeat(MAX_NAME_BYTES); // 2 bytes each, must cut on a boundary
    let mut path = InstancePath::root();
    for i in 0..(MAX_BLOCK_PATH_DEPTH as u64 + 40) {
        path = path.child(long.as_str(), ItemHandle::new(i));
    }
    let s = ProvenanceStub {
        handle: None,
        class_name: long.clone(),
        block_path: path,
        layer: multibyte,
    };
    let p = Provenance::from_stub(&s, &long, &long, xf(0.0));
    assert!(p.source().len() <= MAX_NAME_BYTES);
    assert!(p.layout().len() <= MAX_NAME_BYTES);
    assert!(p.class_name().len() <= MAX_NAME_BYTES);
    assert!(p.layer().len() <= MAX_NAME_BYTES);
    assert!(p.layer().is_char_boundary(p.layer().len()));
    assert_eq!(p.block_path().depth(), MAX_BLOCK_PATH_DEPTH);
    assert!(p.block_path().steps().iter().all(|s| s.block_name.len() <= MAX_NAME_BYTES));
    assert!(p.is_truncated());
    let mut report = String::new();
    p.write_report(&mut report).unwrap();
    assert!(report.len() < 64 * 1024, "report of one record stays small, got {}", report.len());
}

#[test]
fn an_untruncated_record_says_so() {
    let p = Provenance::from_stub(&stub(1, "LINE", "0", &InstancePath::root()), "a", "b", xf(0.0));
    assert!(!p.is_truncated());
}

#[test]
fn a_warning_is_retrievable_through_the_entity_provenance() {
    let mut store = ProvenanceStore::new();
    let p = InstancePath::root();
    let h1 = store.record(|| Provenance::from_stub(&stub(1, "SPLINE", "0", &p), "a", "b", xf(0.0)));
    let h2 = store.record(|| Provenance::from_stub(&stub(2, "LINE", "0", &p), "a", "b", xf(0.0)));
    store.warn(h1, "knot vector clamped");
    let w1 = store.get(h1.unwrap()).unwrap().warnings();
    assert_eq!(w1, ["knot vector clamped"]);
    assert!(store.get(h2.unwrap()).unwrap().warnings().is_empty());
    // A warning against no handle (provenance off) is a no-op.
    store.warn(None, "ignored");
    assert_eq!(store.len(), 2);
}

#[test]
fn warnings_are_bounded_in_count_and_length() {
    let mut store = ProvenanceStore::new();
    let h = store.record(|| Provenance::from_stub(&stub(1, "X", "0", &InstancePath::root()), "a", "b", xf(0.0)));
    let big = "w".repeat(1_000_000);
    for _ in 0..(MAX_WARNINGS * 10) {
        store.warn(h, &big);
    }
    let p = store.get(h.unwrap()).unwrap();
    assert_eq!(p.warnings().len(), MAX_WARNINGS);
    assert!(p.warnings().iter().all(|w| w.len() <= MAX_WARNING_BYTES));
    assert!(p.is_truncated());
}

#[test]
fn report_lines_are_escaped_and_stable() {
    let mut store = ProvenanceStore::new();
    let path = InstancePath::root().child("B", ItemHandle::new(0x1F));
    let s = stub(0xAB, "ARC", "we\"ird\nlayer", &path);
    let h = store.record(|| Provenance::from_stub(&s, "plan.dwg", "Model", xf(2.0)));
    store.warn(h, "bad\nthing");
    let mut out = String::new();
    store.write_report(&mut out).unwrap();
    assert_eq!(out.lines().count(), 1, "one record is one line: {out:?}");
    assert!(out.contains("handle=AB"));
    assert!(out.contains("path=B@1F"));
    assert!(out.contains(r#"layer="we\"ird\nlayer""#));
    let mut again = String::new();
    store.write_report(&mut again).unwrap();
    assert_eq!(out, again);
}

#[test]
fn module_does_no_geometry_math() {
    let body: String = SRC
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in ["sqrt", "atan2", "to_radians", "EPSILON", "tolerance"] {
        assert!(!body.contains(banned), "provenance.rs must not use `{banned}`");
    }
}
