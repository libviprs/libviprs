//! Issue #1103 (N3.8): the debug and inspection export of normalized geometry.
//!
//! One group per acceptance criterion. The dump is text, so the tests read it as text.
#![cfg(feature = "pvec-normalized")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::dump::{DUMP_VERSION, NormalizedDump, dump_to_string};
use libviprs::pvec::provenance::{Provenance, ProvenanceSink, ProvenanceStore};
use libviprs::pvec::{
    Basis, Classification, FidelityClass, InstancePath, Normalized2DGeometry, ProvenanceStub,
    ReasonCode, SourceTransform, Tally,
};

const SRC: &str = include_str!("../src/pvec/dump.rs");
const DOC: &str = include_str!("../docs/normalized-dump.md");

fn xf(tx: f64) -> SourceTransform {
    SourceTransform::new([1.0, 0.0, 0.0, 1.0, tx, 0.0]).unwrap()
}

/// A fixture with a block instanced three times (one nested), one entity per fidelity route,
/// and awkward coordinates. Provenance goes to `store`.
fn fixture(store: &mut ProvenanceStore) -> Vec<Normalized2DGeometry> {
    let room = InstancePath::root().child("ROOM", ItemHandle::new(0x10));
    let paths = [
        room.child("TREE", ItemHandle::new(0x21)),
        InstancePath::root().child("TREE", ItemHandle::new(0x30)),
        InstancePath::root().child("TREE", ItemHandle::new(0x31)),
    ];
    let mut out = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let shift = i as f64 * 10.0;
        let stub = ProvenanceStub {
            handle: Some(ItemHandle::new(0x4C)),
            class_name: "CIRCLE".into(),
            block_path: path.clone(),
            layer: "TREES".into(),
        };
        let h = store.record(|| Provenance::from_stub(&stub, "plan.dwg", "Model", xf(shift)));
        let g = Normalized2DGeometry::circle(
            [shift + 0.1, 1.0 / 3.0],
            [1.0, 0.0],
            2.5,
            FidelityClass::SupportedExact,
        )
        .unwrap();
        out.push(g.with_provenance(h.unwrap()));
    }
    out.push(
        Normalized2DGeometry::polyline(
            &[[0.0, 0.0], [1.0000000000000002, 3.0], [123456789.12345678, -7.5e-5]],
            FidelityClass::SupportedExact,
        )
        .unwrap(),
    );
    out.push(
        Normalized2DGeometry::circular_arc(
            [0.0, 0.0],
            [1.0, 0.0],
            4.0,
            0.25,
            1.75,
            true,
            FidelityClass::SupportedEquivalent,
        )
        .unwrap(),
    );
    out.push(
        Normalized2DGeometry::ellipse([5.0, 5.0], [1.0, 0.0], 3.0, 1.5, FidelityClass::SupportedExact)
            .unwrap(),
    );
    out.push(
        Normalized2DGeometry::b_spline(
            2,
            &[[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]],
            &[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            FidelityClass::SupportedExact,
        )
        .unwrap(),
    );
    out.push(Normalized2DGeometry::without_geometry(FidelityClass::Unsupported).unwrap());
    out.push(Normalized2DGeometry::without_geometry(FidelityClass::IgnoredByPolicy).unwrap());
    out
}

fn dump_with_provenance(store: &ProvenanceStore, es: &[Normalized2DGeometry]) -> String {
    let mut d = NormalizedDump::new(String::new()).unwrap();
    for e in es {
        d.push_stored(e, store).unwrap();
    }
    d.finish().unwrap()
}

fn value_after<'a>(line: &'a str, key: &str) -> &'a str {
    let start = line.find(key).unwrap_or_else(|| panic!("no {key} in {line}")) + key.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| c == ' ' || c == ',' || c == ')')
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn the_format_has_a_versioned_header_and_an_end_line() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    let s = dump_with_provenance(&store, &es);
    let first = s.lines().next().unwrap();
    assert_eq!(first, format!("# libviprs normalized dump v{DUMP_VERSION}"));
    assert!(s.lines().last().unwrap().starts_with("end entities="), "{s}");
    assert!(s.ends_with('\n'));
}

#[test]
fn two_dumps_of_one_fixture_are_byte_identical() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    let a = dump_with_provenance(&store, &es);
    let b = dump_with_provenance(&store, &es);
    assert!(!a.is_empty());
    assert_eq!(a.as_bytes(), b.as_bytes());

    // A second, independently built fixture dumps the same too.
    let mut store2 = ProvenanceStore::new();
    let es2 = fixture(&mut store2);
    assert_eq!(a, dump_with_provenance(&store2, &es2));
}

#[test]
fn coordinates_reparse_to_bit_identical_f64() {
    let c = Normalized2DGeometry::circle(
        [0.1, 1.0 / 3.0],
        [1.0, 0.0],
        1.0000000000000002,
        FidelityClass::SupportedExact,
    )
    .unwrap();
    let s = dump_to_string([(&c, None)]);
    let line = s.lines().find(|l| l.contains("radius=")).expect(&s);
    let radius: f64 = value_after(line, "radius=").parse().unwrap();
    assert_eq!(radius.to_bits(), 1.0000000000000002f64.to_bits());
    let ox: f64 = value_after(line, "origin=(").parse().unwrap();
    assert_eq!(ox.to_bits(), 0.1f64.to_bits());
    let oy: f64 = line
        .split("origin=(")
        .nth(1)
        .unwrap()
        .split(',')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(oy.to_bits(), (1.0f64 / 3.0).to_bits());

    let p = Normalized2DGeometry::polyline(
        &[[0.0, 0.0], [123456789.12345678, -7.5e-5]],
        FidelityClass::SupportedExact,
    )
    .unwrap();
    let s = dump_to_string([(&p, None)]);
    assert!(s.contains("123456789.12345678"), "{s}");
    assert!(s.contains("-7.5e-5") || s.contains("-0.000075"), "{s}");
}

#[test]
fn every_entity_carries_type_fidelity_and_bounds() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    let s = dump_with_provenance(&store, &es);
    let heads: Vec<&str> = s.lines().filter(|l| l.starts_with("entity ")).collect();
    assert_eq!(heads.len(), es.len());
    for (i, (h, e)) in heads.iter().zip(&es).enumerate() {
        let parts: Vec<&str> = h.split(' ').collect();
        assert_eq!(parts.len(), 4, "{h}");
        assert_eq!(parts[1], i.to_string(), "ids are positions, see #1281");
        assert_eq!(parts[3], e.fidelity().tag());
    }
    for kind in ["circle", "polyline", "trimmed", "ellipse", "bspline", "none"] {
        assert!(heads.iter().any(|h| h.split(' ').nth(2) == Some(kind)), "{kind}\n{s}");
    }
    assert!(s.contains("  bounds min=("), "{s}");
    assert!(s.contains("  bounds none"), "{s}");
}

#[test]
fn the_entity_count_equals_the_fidelity_encountered_count() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    let mut tally = Tally::new();
    for e in &es {
        let c = match e.fidelity() {
            FidelityClass::Unsupported => {
                Classification::refused(ReasonCode::UnmappedSourceClass, "PROXY")
            }
            FidelityClass::IgnoredByPolicy => Classification::ignored_by_policy("layers", "TEXT"),
            _ => Classification::supported(Basis::Exact, "CIRCLE"),
        };
        tally.record(&c);
    }
    let report = tally.finish(es.len() as u64).unwrap();
    let mut d = NormalizedDump::new(String::new()).unwrap();
    for e in &es {
        d.push_stored(e, &store).unwrap();
    }
    assert_eq!(d.entities_written(), report.encountered());
    let s = d.finish().unwrap();
    assert!(s.contains(&format!("end entities={}", report.encountered())), "{s}");
}

#[test]
fn a_failure_after_entity_n_still_leaves_n_entities_in_the_sink() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    // The sink outlives the writer: a conversion that dies in PVec writing drops the dump
    // without calling `finish`, and the sink already holds what was normalized so far.
    struct Shared<'a>(&'a mut String);
    impl std::fmt::Write for Shared<'_> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.push_str(s);
            Ok(())
        }
    }
    let mut sink = String::new();
    {
        let mut d = NormalizedDump::new(Shared(&mut sink)).unwrap();
        for e in es.iter().take(3) {
            d.push_stored(e, &store).unwrap();
        }
        // the PVec writing stage fails here; `d` is dropped, never finished
    }
    assert_eq!(sink.lines().filter(|l| l.starts_with("entity ")).count(), 3, "{sink}");
    assert!(sink.starts_with("# libviprs normalized dump"));
    assert!(!sink.contains("\nend entities="), "an unfinished dump has no end line");
}

#[test]
fn provenance_is_printed_when_present_and_absent_when_off() {
    let mut store = ProvenanceStore::new();
    let es = fixture(&mut store);
    let with = dump_with_provenance(&store, &es);
    assert!(with.contains("  provenance source=\"plan.dwg\""), "{with}");
    assert!(with.contains("layer=\"TREES\""));
    assert!(with.contains("handle=4C"));
    assert!(with.contains("path=ROOM@10/TREE@21"));
    assert!(with.contains("xf=[1.0, 0.0, 0.0, 1.0, 10.0, 0.0]"));

    let without = dump_to_string(es.iter().map(|e| (e, None)));
    assert!(!without.contains("provenance"), "{without}");
    // The geometry lines do not change when provenance goes away.
    let strip = |s: &str| {
        s.lines()
            .filter(|l| !l.starts_with("  provenance"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&with), strip(&without));
}

#[test]
fn the_dump_is_independent_of_pvec_serialization() {
    // No PVec codec or container type is named by the writer, and there is no new dependency.
    for banned in ["pvec_codec", "pvec_container", "pvec_entity", "HashMap", "HashSet"] {
        assert!(!SRC.contains(banned), "dump.rs mentions {banned}");
    }
}

#[test]
fn nothing_reads_the_format_back() {
    for banned in ["fn parse", "fn load", "fn read", "FromStr", "fn from_dump"] {
        assert!(!SRC.contains(banned), "dump.rs has a loader-shaped item: {banned}");
    }
}

#[test]
fn the_document_describes_the_format_and_disclaims_interchange() {
    assert!(DOC.contains("not an interchange format"), "{DOC}");
    assert!(DOC.contains("no loader"), "{DOC}");
    for key in ["entity", "bounds", "provenance", "end entities=", "#1281"] {
        assert!(DOC.contains(key), "docs/normalized-dump.md does not mention {key}");
    }
}
