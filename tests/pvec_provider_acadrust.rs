//! `AcadrustProvider`, the first DWG provider (N4.1, #1072), on the two acceptance
//! fixtures `vc_layers` and `vc_nested`.
//!
//! **What these fixtures are, and are not.** Both were written by ACadSharp's
//! `DwgWriter` (the N1.4 spike, #1235) and are read here by acadrust, a different
//! implementation. That is a cross-family check: it catches the two readers
//! disagreeing about a file one of them wrote, and it says nothing about what
//! AutoCAD would make of the file. The expectations below are hand computed from the
//! drawings' construction (see `docs/spikes/view-context.md` on #1235, the two worked
//! examples). They are **not known truth**. Known truth is the AutoCAD-authored
//! corpus, which is Gate 0's table in `docs/spikes/acadrust-gate0.md`.
//!
//! Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use std::path::PathBuf;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{
    Availability, DwgProvider, EntityScope, Facet, FacetSource, InstancePath, LayoutKind,
    ProviderError, RawEntity, RawGeometry,
};
use sha2::{Digest, Sha256};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider")
        .join(name)
}

fn open(name: &str) -> AcadrustProvider {
    let path = fixture(name);
    AcadrustProvider::open(CadSource::Path(&path)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn model(p: &AcadrustProvider) -> Vec<RawEntity> {
    let layouts = p.layouts().unwrap();
    let m = layouts
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model space");
    p.entities(&EntityScope::Layout(m.id))
        .unwrap()
        .collect_all()
        .unwrap()
}

fn block(p: &AcadrustProvider, name: &str, via: &InstancePath) -> Vec<RawEntity> {
    p.entities(&EntityScope::BlockDefinition {
        name: name.to_string(),
        reached_via: via.clone(),
    })
    .unwrap()
    .collect_all()
    .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn handle(e: &RawEntity) -> u64 {
    e.provenance.handle.expect("handle").get()
}

fn xf(e: &RawEntity) -> [f64; 6] {
    match &e.geometry {
        RawGeometry::Insert(r) => r.transform.coefficients(),
        other => panic!("not an insert: {other:?}"),
    }
}

// ---- the fixtures are what the notes say they are ----------------------------------

#[test]
fn the_fixtures_are_the_ones_n1_4_recorded_and_say_what_they_are() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec/provider");
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).expect("SHA256SUMS");
    for name in ["vc_layers.dwg", "vc_nested.dwg"] {
        let bytes = std::fs::read(dir.join(name)).expect(name);
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert!(
            sums.lines()
                .any(|l| l.starts_with(&got) && l.ends_with(name)),
            "{name}: sha256 {got} is not in SHA256SUMS"
        );
    }
    let note = std::fs::read_to_string(dir.join("PROVENANCE.md")).expect("PROVENANCE.md");
    assert!(
        note.contains("cross-family"),
        "the note must say cross-family"
    );
    assert!(
        note.contains("not known truth"),
        "the note must say not known truth"
    );
    assert!(note.contains("ACadSharp"), "the note must name the writer");
}

// ---- probe and open ----------------------------------------------------------------

#[test]
fn probe_answers_from_the_version_stamp_alone() {
    let p = fixture("vc_layers.dwg");
    assert!(AcadrustProvider::probe(CadSource::Path(&p)).unwrap());
    assert!(AcadrustProvider::probe(CadSource::Bytes(b"AC1032\0\0\0\0")).unwrap());
    assert!(!AcadrustProvider::probe(CadSource::Bytes(b"AC1009\0\0\0\0")).unwrap());
    assert!(!AcadrustProvider::probe(CadSource::Bytes(b"  0\nSECTION\n")).unwrap());
    assert!(!AcadrustProvider::probe(CadSource::Bytes(b"")).unwrap());
}

#[test]
fn open_refuses_what_is_not_a_drawing_with_a_typed_error() {
    for bytes in [
        &b""[..],
        b"hello",
        b"AC1032",
        b"AC1032 and then nothing that parses",
    ] {
        match AcadrustProvider::open(CadSource::Bytes(bytes)) {
            Err(ProviderError::Unrecognised { .. } | ProviderError::Open { .. }) => {}
            Err(other) => panic!("unexpected error kind: {other:?}"),
            Ok(_) => panic!("opened {bytes:?}"),
        }
    }
    let missing = fixture("does-not-exist.dwg");
    assert!(AcadrustProvider::open(CadSource::Path(&missing)).is_err());
}

#[test]
fn open_works_from_bytes_and_from_a_path_alike() {
    let bytes = std::fs::read(fixture("vc_nested.dwg")).unwrap();
    let a = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
    let path = fixture("vc_nested.dwg");
    let b = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
    assert_eq!(a.name(), "acadrust");
    assert_eq!(format!("{:?}", model(&a)), format!("{:?}", model(&b)));
}

// ---- vc_layers: layer state, layouts, entities, insert chain -----------------------

#[test]
fn vc_layers_exposes_the_layer_table_with_its_flags() {
    let p = open("vc_layers.dwg");
    let layers = p.layers().unwrap();
    assert_eq!(layers.len(), 4);
    let by = |n: &str| {
        layers
            .iter()
            .find(|l| l.name == n)
            .unwrap_or_else(|| panic!("layer {n}"))
    };
    assert!(by("0").off && !by("0").frozen);
    assert!(!by("VC_ON").off && !by("VC_ON").frozen);
    assert!(by("VC_OFF").off && !by("VC_OFF").frozen);
    assert!(!by("VC_FROZEN").off && by("VC_FROZEN").frozen);
    assert!(layers.iter().all(|l| !l.locked));
    assert!(layers.iter().all(|l| l.handle.is_some()));
    // The same drawing twice gives the same table in the same order.
    assert_eq!(layers, p.layers().unwrap());
}

#[test]
fn vc_layers_exposes_model_space_first_then_the_paper_layout() {
    let p = open("vc_layers.dwg");
    let l = p.layouts().unwrap();
    assert_eq!(l.len(), 2);
    assert_eq!(
        (l[0].name.as_str(), l[0].kind, l[0].block_name.as_str()),
        ("Model", LayoutKind::Model, "*Model_Space")
    );
    assert_eq!(
        (l[1].name.as_str(), l[1].kind, l[1].block_name.as_str()),
        ("Layout1", LayoutKind::Paper, "*Paper_Space")
    );
    assert_ne!(l[0].id, l[1].id);
}

#[test]
fn vc_layers_model_space_is_unfiltered_and_in_storage_order() {
    // Visibility is not applied here: layer state is data, the rule is N1.4's.
    let p = open("vc_layers.dwg");
    let e = model(&p);
    let handles: Vec<u64> = e.iter().map(handle).collect();
    assert_eq!(handles, vec![0x4B, 0x4C, 0x4D, 0x4E, 0x55, 0x56, 0x57]);
    let layers: Vec<&str> = e.iter().map(|x| x.provenance.layer.as_str()).collect();
    assert_eq!(
        layers,
        [
            "VC_ON",
            "VC_OFF",
            "VC_FROZEN",
            "0",
            "VC_ON",
            "VC_OFF",
            "VC_FROZEN"
        ]
    );
    let classes: Vec<&str> = e.iter().map(|x| x.provenance.class_name.as_str()).collect();
    assert_eq!(
        classes,
        ["LINE", "LINE", "LINE", "LINE", "INSERT", "INSERT", "INSERT"]
    );
    assert!(e.iter().all(|x| x.provenance.block_path.is_root()));

    match &e[1].geometry {
        RawGeometry::Line { start, end } => {
            assert_eq!((start[0], start[1], end[0], end[1]), (0.0, 2.0, 10.0, 2.0));
        }
        other => panic!("{other:?}"),
    }
    // Three INSERTs of one block, at x = 20, 30, 40.
    for (k, x) in [(4, 20.0), (5, 30.0), (6, 40.0)] {
        match &e[k].geometry {
            RawGeometry::Insert(r) => assert_eq!(r.block_name, "VC_BLK"),
            other => panic!("{other:?}"),
        }
        let t = xf(&e[k]);
        assert!(close(t[0], 1.0) && close(t[3], 1.0) && close(t[1], 0.0) && close(t[2], 0.0));
        assert!(close(t[4], x) && close(t[5], 0.0), "{t:?}");
    }
}

#[test]
fn vc_layers_block_definition_keeps_the_entities_own_layers_and_the_path_it_was_given() {
    let p = open("vc_layers.dwg");
    let via = InstancePath::root().child("VC_BLK", libviprs::cad::ItemHandle::new(0x55));
    let e = block(&p, "VC_BLK", &via);
    assert_eq!(
        e.iter().map(handle).collect::<Vec<_>>(),
        vec![0x52, 0x53, 0x54]
    );
    // b1 is on layer "0": its own layer, not the INSERT's. Effective layer is N1.4's V1-2.
    let layers: Vec<&str> = e.iter().map(|x| x.provenance.layer.as_str()).collect();
    assert_eq!(layers, ["0", "VC_ON", "VC_OFF"]);
    assert!(e.iter().all(|x| x.provenance.block_path == via));
    assert!(p.block_names().iter().any(|n| n == "VC_BLK"));
    assert!(p.block_names().iter().any(|n| n == "*Model_Space"));
}

#[test]
fn an_unknown_block_or_layout_is_no_such_scope() {
    let p = open("vc_layers.dwg");
    assert!(matches!(
        p.entities(&EntityScope::BlockDefinition {
            name: "NOPE".into(),
            reached_via: InstancePath::root()
        }),
        Err(ProviderError::NoSuchScope(_))
    ));
    assert!(matches!(
        p.entities(&EntityScope::Layout(libviprs::pvec::LayoutId(99))),
        Err(ProviderError::NoSuchScope(_))
    ));
}

// ---- vc_nested: instance paths and transforms against the worked example -----------

type Seg = ([f64; 2], [f64; 2]);

/// Walks the drawing through the traits, applies the composed transforms innermost
/// first, and returns `(path, world segment)` for every line.
fn flatten(p: &AcadrustProvider) -> Vec<(String, Seg)> {
    fn apply(t: &[f64; 6], v: [f64; 2]) -> [f64; 2] {
        [
            t[0] * v[0] + t[1] * v[1] + t[4],
            t[2] * v[0] + t[3] * v[1] + t[5],
        ]
    }
    fn go(
        p: &AcadrustProvider,
        scope: &EntityScope,
        chain: &[[f64; 6]],
        out: &mut Vec<(String, Seg)>,
    ) {
        for e in p.entities(scope).unwrap().collect_all().unwrap() {
            match &e.geometry {
                RawGeometry::Insert(r) => {
                    let mut next = chain.to_vec();
                    next.push(r.transform.coefficients());
                    go(
                        p,
                        &EntityScope::BlockDefinition {
                            name: r.block_name.clone(),
                            reached_via: e.child_path().unwrap(),
                        },
                        &next,
                        out,
                    );
                }
                RawGeometry::Line { start, end } => {
                    let world = |pt: [f64; 3]| {
                        let mut v = [pt[0], pt[1]];
                        for t in chain.iter().rev() {
                            v = apply(t, v);
                        }
                        v
                    };
                    out.push((
                        e.provenance.block_path.to_string(),
                        (world(*start), world(*end)),
                    ));
                }
                _ => {}
            }
        }
    }
    let layouts = p.layouts().unwrap();
    let m = layouts
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let mut out = Vec::new();
    go(p, &EntityScope::Layout(m.id), &[], &mut out);
    out
}

#[test]
fn vc_nested_inserts_carry_the_transforms_their_fields_describe() {
    let p = open("vc_nested.dwg");
    let e = model(&p);
    assert_eq!(
        e.iter().map(handle).collect::<Vec<_>>(),
        vec![0x56, 0x57, 0x58]
    );
    let blocks: Vec<String> = e
        .iter()
        .map(|x| match &x.geometry {
            RawGeometry::Insert(r) => r.block_name.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(blocks, ["VC_INNER", "VC_INNER", "VC_OUTER"]);
    // The same block twice at two scales: two different instances, one block.
    let (a, b, c) = (xf(&e[0]), xf(&e[1]), xf(&e[2]));
    assert!(close(a[0], 1.0) && close(a[3], 1.0) && close(a[4], 10.0));
    assert!(close(b[0], 3.0) && close(b[3], 3.0) && close(b[4], 20.0));
    assert!(close(c[0], 2.0) && close(c[3], 2.0) && close(c[4], 100.0) && close(c[5], 100.0));
}

#[test]
fn vc_nested_reaches_the_same_block_through_distinct_instance_paths() {
    let p = open("vc_nested.dwg");
    let got = flatten(&p);
    let paths: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        paths,
        [
            "VC_INNER@56",
            "VC_INNER@57",
            "VC_OUTER@58/VC_MID@55",
            "VC_OUTER@58/VC_MID@55/VC_INNER@51",
        ]
    );
}

#[test]
fn vc_nested_composed_leaves_land_where_the_hand_computation_puts_them() {
    // The worked example in docs/spikes/view-context.md, computed by hand from the
    // construction of the drawing. Hand computed, writer ACadSharp, reader acadrust:
    // cross-family, not known truth.
    let p = open("vc_nested.dwg");
    let got = flatten(&p);
    let expect: [Seg; 4] = [
        ([10.0, 0.0], [11.0, 0.0]),
        ([20.0, 0.0], [23.0, 0.0]),
        ([106.0, 100.0], [104.0, 100.0]),
        ([106.0, 102.0], [102.0, 102.0]),
    ];
    assert_eq!(got.len(), 4);
    for (k, ((_, seg), want)) in got.iter().zip(expect).enumerate() {
        for i in 0..2 {
            assert!(
                close(seg.0[i], want.0[i]),
                "leaf {k} start[{i}]: {seg:?} against {want:?}"
            );
            assert!(
                close(seg.1[i], want.1[i]),
                "leaf {k} end[{i}]: {seg:?} against {want:?}"
            );
        }
    }
}

// ---- determinism and caps ----------------------------------------------------------

#[test]
fn two_traversals_of_every_scope_agree_entity_by_entity() {
    for name in ["vc_layers.dwg", "vc_nested.dwg"] {
        let p = open(name);
        let mut scopes: Vec<EntityScope> = p
            .layouts()
            .unwrap()
            .iter()
            .map(|l| EntityScope::Layout(l.id))
            .collect();
        for b in p.block_names() {
            scopes.push(EntityScope::BlockDefinition {
                name: b,
                reached_via: InstancePath::root(),
            });
        }
        assert!(scopes.len() >= 4, "{name}: {} scopes", scopes.len());
        for scope in scopes {
            let a = p.entities(&scope).unwrap().collect_all().unwrap();
            let b = p.entities(&scope).unwrap().collect_all().unwrap();
            assert_eq!(a.len(), b.len(), "{name} {scope:?}");
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                // Debug prints floats with the shortest text that round-trips the bits.
                assert_eq!(
                    format!("{x:?}"),
                    format!("{y:?}"),
                    "{name} {scope:?} entity {i}"
                );
            }
        }
        // A fresh open gives the same sequence too, not just the same handle.
        let q = open(name);
        assert_eq!(
            format!("{:?}", model(&p)),
            format!("{:?}", model(&q)),
            "{name}: reopen"
        );
    }
}

#[test]
fn caps_claim_what_the_reader_exposes_and_name_the_source() {
    let p = open("vc_layers.dwg");
    let caps = p.caps();
    for f in Facet::ALL {
        let row = caps.get(f).unwrap_or_else(|| panic!("no row for {f:?}"));
        assert_eq!(row.availability, Availability::Reported, "{f:?}");
        // acadrust is read directly, not through the CadDecoder adapter.
        assert_eq!(row.source, FacetSource::DecoderDirect, "{f:?}");
    }
    for e in model(&p) {
        assert!(caps.violations(&e).is_empty(), "{:?}", e.provenance);
    }
}
