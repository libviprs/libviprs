//! Issue #1102 (N3.7): deterministic source ordering and identity.

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    InstancePath, OrderingBasis, OrderingPolicy, Point3, ProvenanceStub, RawEntity, RawGeometry,
    canonical_cmp, order_entities,
};

fn pt(x: f64) -> Point3 {
    [x, 0.0, 0.0]
}

fn ent(handle: Option<u64>, path: InstancePath, layer: &str, x: f64) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Line {
            start: pt(x),
            end: pt(x + 1.0),
        },
        provenance: ProvenanceStub {
            handle: handle.map(ItemHandle::new),
            class_name: "LINE".into(),
            block_path: path,
            layer: layer.into(),
        },
    }
}

/// A fixture with one block instantiated twice: the leaf handle 0x20 repeats per instance.
fn fixture() -> Vec<RawEntity> {
    let a = InstancePath::root().child("B", ItemHandle::new(0x50));
    let b = InstancePath::root().child("B", ItemHandle::new(0x51));
    vec![
        ent(Some(0x10), InstancePath::root(), "0", 0.0),
        ent(Some(0x20), a.clone(), "0", 1.0),
        ent(Some(0x21), a, "0", 2.0),
        ent(Some(0x20), b.clone(), "0", 1.0),
        ent(Some(0x21), b, "0", 2.0),
        ent(None, InstancePath::root(), "0", 9.0),
    ]
}

fn render(v: &[libviprs::pvec::OrderedEntity]) -> String {
    v.iter()
        .map(|o| format!("{} {:?}\n", o.id.get(), o.entity))
        .collect()
}

fn canon() -> OrderingPolicy {
    OrderingPolicy::new()
}

#[test]
fn two_runs_are_byte_identical_with_repeated_block_instance() {
    let a = order_entities("p", "1", &canon(), fixture());
    let b = order_entities("p", "1", &canon(), fixture());
    assert_eq!(render(&a.entities), render(&b.entities));
}

#[test]
fn canonical_ids_do_not_depend_on_arrival_order() {
    let base = order_entities("p", "1", &canon(), fixture());
    let mut rev = fixture();
    rev.reverse();
    let other = order_entities("p", "1", &canon(), rev);
    assert_eq!(render(&base.entities), render(&other.entities));
}

#[test]
fn ids_are_dense_from_zero_in_output_order() {
    let o = order_entities("p", "1", &canon(), fixture());
    let ids: Vec<u64> = o.entities.iter().map(|e| e.id.get()).collect();
    assert_eq!(ids, (0..6).collect::<Vec<_>>());
}

#[test]
fn canonical_order_is_path_then_handle_then_index() {
    let root = InstancePath::root();
    let a = InstancePath::root().child("B", ItemHandle::new(1));
    let e_root_hi = ent(Some(9), root.clone(), "0", 0.0);
    let e_root_lo = ent(Some(2), root.clone(), "0", 0.0);
    let e_none = ent(None, root, "0", 0.0);
    let e_deep = ent(Some(1), a, "0", 0.0);
    assert_eq!(canonical_cmp(0, &e_root_lo, 0, &e_root_hi), Ordering::Less);
    assert_eq!(canonical_cmp(0, &e_none, 0, &e_root_lo), Ordering::Less);
    assert_eq!(canonical_cmp(0, &e_root_hi, 0, &e_deep), Ordering::Less);
    assert_eq!(canonical_cmp(1, &e_root_lo, 2, &e_root_lo), Ordering::Less);
}

#[test]
fn same_handle_and_index_still_orders_deterministically() {
    let p = InstancePath::root();
    let x = ent(Some(5), p.clone(), "A", 0.0);
    let y = ent(Some(5), p.clone(), "B", 0.0);
    let z = ent(Some(5), p, "A", 3.0);
    for (l, r) in [(&x, &y), (&x, &z), (&y, &z)] {
        let fwd = canonical_cmp(7, l, 7, r);
        assert_ne!(fwd, Ordering::Equal);
        assert_eq!(canonical_cmp(7, r, 7, l), fwd.reverse());
    }
    assert_eq!(canonical_cmp(7, &x, 7, &x), Ordering::Equal);
    // transitivity over the three
    assert_eq!(canonical_cmp(7, &x, 7, &y), Ordering::Less);
    assert_eq!(canonical_cmp(7, &y, 7, &z), Ordering::Greater);
    assert_eq!(canonical_cmp(7, &x, 7, &z), Ordering::Less);
}

#[test]
fn declared_traversal_keeps_arrival_order_and_says_so() {
    let policy = OrderingPolicy::new().declare("p", "2", OrderingBasis::DecoderTraversal);
    let mut rev = fixture();
    rev.reverse();
    let o = order_entities("p", "2", &policy, rev.clone());
    assert_eq!(o.report.basis(), OrderingBasis::DecoderTraversal);
    let got: Vec<_> = o
        .entities
        .iter()
        .map(|e| format!("{:?}", e.entity))
        .collect();
    let want: Vec<_> = rev.iter().map(|e| format!("{e:?}")).collect();
    assert_eq!(got, want);
    let ids: Vec<u64> = o.entities.iter().map(|e| e.id.get()).collect();
    assert_eq!(ids, (0..6).collect::<Vec<_>>());
}

#[test]
fn undeclared_provider_or_version_is_canonicalised() {
    let policy = OrderingPolicy::new().declare("p", "2", OrderingBasis::DecoderTraversal);
    assert_eq!(policy.basis_for("p", "3"), OrderingBasis::Canonicalised);
    assert_eq!(policy.basis_for("q", "2"), OrderingBasis::Canonicalised);
    assert_eq!(policy.basis_for("p", "2"), OrderingBasis::DecoderTraversal);
}

#[test]
fn report_names_provider_version_and_basis() {
    let o = order_entities("acadrust", "0.4.1", &canon(), fixture());
    let mut s = String::new();
    o.report.write_report(&mut s).unwrap();
    assert_eq!(
        s,
        "ordering provider=\"acadrust\" version=\"0.4.1\" basis=canonicalised entities=6"
    );
    let policy =
        OrderingPolicy::new().declare("acadrust", "0.5.0", OrderingBasis::DecoderTraversal);
    let o = order_entities("acadrust", "0.5.0", &policy, fixture());
    let mut s = String::new();
    o.report.write_report(&mut s).unwrap();
    assert!(s.contains("basis=decoder-traversal"), "{s}");
}

#[test]
fn ids_do_not_depend_on_provenance_recording() {
    // Ordering takes entities only: a provenance sink is not an input, so ids with
    // provenance on or off are the same function of the same entities.
    let off = order_entities("p", "1", &canon(), fixture());
    #[cfg(feature = "pvec-normalized")]
    {
        use libviprs::pvec::SourceTransform;
        use libviprs::pvec::provenance::{Provenance, ProvenanceSink, ProvenanceStore};
        let mut store = ProvenanceStore::new();
        for o in &off.entities {
            let _ = store.record(|| {
                Provenance::from_stub(
                    &o.entity.provenance,
                    "f.dwg",
                    "Model",
                    SourceTransform::new([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
                )
            });
        }
    }
    let on = order_entities("p", "1", &canon(), fixture());
    assert_eq!(render(&off.entities), render(&on.entities));
}

/// Lines of `src` that name a hash container outside comments.
fn hash_uses(src: &str) -> Vec<String> {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .filter(|l| l.contains("HashMap") || l.contains("HashSet"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn hash_use_detector_flags_a_hashmap() {
    assert_eq!(
        hash_uses("let m: HashMap<u8,u8> = x;\n// HashSet here\n").len(),
        1
    );
    assert!(hash_uses("use std::collections::BTreeMap;\n").is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn no_hash_iteration_in_emit_path() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pvec");
    let mut bad = Vec::new();
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "rs") {
            for l in hash_uses(&fs::read_to_string(&p).unwrap()) {
                bad.push(format!("{}: {}", p.display(), l.trim()));
            }
        }
    }
    assert!(bad.is_empty(), "hash containers in src/pvec: {bad:#?}");
}
