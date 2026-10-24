//! Units and source-coordinate handling (N4.7, #1078).
//!
//! * The `$INSUNITS` mapping is one closed table, code 0 included, and a missing or
//!   out-of-table value is `Unknown` rather than a guess.
//! * `docs/pvec-units.md` has a row for every code the enum admits.
//! * Nothing under `src/pvec/` refers to the raster-side georeference types,
//!   asserted over the sources.
//! * Needs `--features dwg-acadrust` for the provider half: coordinates reach the
//!   cursor bit-identical to what the file stores, and the same drawing in inches
//!   and in millimetres differs in its `SourceUnit` and nowhere else.

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::pvec::{DrawingExtents, INSUNITS_TABLE, SourceSpace, SourceUnit};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn every_code_in_the_table_round_trips_and_code_zero_is_unitless() {
    assert_eq!(INSUNITS_TABLE.len(), 25);
    for (i, (code, unit, name)) in INSUNITS_TABLE.iter().enumerate() {
        assert_eq!(usize::try_from(*code).unwrap(), i, "table is in code order");
        assert_eq!(SourceUnit::from_insunits(Some(*code)), *unit);
        assert_eq!(unit.insunits_code(), Some(*code));
        assert_eq!(unit.name(), *name);
    }
    assert_eq!(SourceUnit::from_insunits(Some(0)), SourceUnit::Unitless);
    assert_eq!(SourceUnit::from_insunits(Some(1)), SourceUnit::Inches);
    assert_eq!(SourceUnit::from_insunits(Some(4)), SourceUnit::Millimeters);
    assert_eq!(SourceUnit::from_insunits(Some(6)), SourceUnit::Meters);
}

#[test]
fn a_missing_or_unlisted_value_is_unknown_never_a_default() {
    assert_eq!(SourceUnit::from_insunits(None), SourceUnit::Unknown);
    for bad in [-1_i16, 25, 26, 100, i16::MAX, i16::MIN] {
        assert_eq!(SourceUnit::from_insunits(Some(bad)), SourceUnit::Unknown);
    }
    assert_eq!(SourceUnit::Unknown.insunits_code(), None);
    assert_ne!(SourceUnit::Unknown, SourceUnit::Unitless);
}

#[test]
fn only_a_named_length_unit_is_physical() {
    assert!(!SourceUnit::Unknown.is_physical());
    assert!(!SourceUnit::Unitless.is_physical());
    for (_, unit, _) in INSUNITS_TABLE.iter().filter(|(c, _, _)| *c != 0) {
        assert!(unit.is_physical(), "{unit:?}");
    }
}

#[test]
fn the_handoff_metadata_carries_unit_and_bounds_untouched() {
    let b = DrawingExtents {
        min: [0.5, -2.25, 0.0],
        max: [1024.0625, 3.0, 0.0],
    };
    let s = SourceSpace::new(SourceUnit::Millimeters).with_bounds(b);
    assert_eq!(s.unit, SourceUnit::Millimeters);
    let got = s.bounds.expect("bounds");
    assert_eq!(got.min[0].to_bits(), 0.5_f64.to_bits());
    assert_eq!(got.max[0].to_bits(), 1024.0625_f64.to_bits());
    assert!(SourceSpace::new(SourceUnit::Unknown).bounds.is_none());
}

#[test]
#[cfg_attr(miri, ignore)] // reads repository files, which Miri isolation blocks
fn the_units_doc_has_a_row_for_every_code_including_zero() {
    let doc = fs::read_to_string(root().join("docs/pvec-units.md")).expect("docs/pvec-units.md");
    for (code, _, name) in INSUNITS_TABLE.iter() {
        let row = doc
            .lines()
            .find(|l| {
                let mut cells = l.split('|').map(str::trim).skip(1);
                cells.next() == Some(code.to_string().as_str())
            })
            .unwrap_or_else(|| panic!("no table row for code {code}"));
        assert!(row.contains(name), "row for {code} lacks {name:?}: {row}");
    }
    assert!(doc.contains("Unknown"));
    assert!(doc.contains("never rescaled") || doc.contains("never rescales"));
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Whether `src` mentions the raster-side georeference module or its types.
fn mentions_geo(src: &str) -> Option<&'static str> {
    [
        "crate::geo",
        "libviprs::geo",
        "super::geo",
        "GeoTransform",
        "GeoCoord",
        "GeoBounds",
    ]
    .into_iter()
    .find(|needle| src.contains(needle))
}

#[test]
fn the_scan_can_fail() {
    assert!(mentions_geo("use crate::geo::GeoBounds;").is_some());
    assert!(mentions_geo("let t: GeoTransform;").is_some());
    assert!(mentions_geo("use crate::cad::CadSource;").is_none());
}

#[test]
#[cfg_attr(miri, ignore)] // reads repository files, which Miri isolation blocks
fn nothing_under_src_pvec_refers_to_the_geo_module() {
    let mut files = Vec::new();
    rs_files(&root().join("src/pvec"), &mut files);
    assert!(
        files.iter().any(|f| f.ends_with("units.rs")),
        "units.rs is part of the scan"
    );
    for f in files {
        let src = fs::read_to_string(&f).unwrap();
        if let Some(hit) = mentions_geo(&src) {
            panic!("{} refers to {hit}", f.display());
        }
    }
}

#[cfg(feature = "dwg-acadrust")]
mod provider {
    use acadrust::entities::Line;
    use acadrust::types::Vector3;
    use acadrust::{CadDocument, DwgWriter, EntityType};
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind, RawGeometry, SourceUnit};

    /// Exactly representable in binary, so a tolerance would only hide a rescale.
    const START: [f64; 3] = [1.5, -2.25, 0.0];
    const END: [f64; 3] = [1_048_576.5, 0.001953125, 4096.0625];

    fn drawing(insunits: i16) -> Vec<u8> {
        let mut doc = CadDocument::new();
        doc.header.insertion_units = insunits;
        let mut line = Line::new();
        line.start = Vector3 {
            x: START[0],
            y: START[1],
            z: START[2],
        };
        line.end = Vector3 {
            x: END[0],
            y: END[1],
            z: END[2],
        };
        doc.add_entity(EntityType::Line(line)).unwrap();
        DwgWriter::write_to_vec(&doc).unwrap()
    }

    type Segment = ([f64; 3], [f64; 3]);

    fn read(bytes: &[u8]) -> (SourceUnit, Vec<Segment>) {
        let p = AcadrustProvider::open(CadSource::Bytes(bytes)).unwrap();
        let model = p
            .layouts()
            .unwrap()
            .into_iter()
            .find(|l| l.kind == LayoutKind::Model)
            .unwrap();
        let lines = p
            .entities(&EntityScope::Layout(model.id))
            .unwrap()
            .collect_all()
            .unwrap()
            .into_iter()
            .filter_map(|e| match e.geometry {
                RawGeometry::Line { start, end } => Some((start, end)),
                _ => None,
            })
            .collect();
        (p.source_unit(), lines)
    }

    fn bits(p: [f64; 3]) -> [u64; 3] {
        p.map(f64::to_bits)
    }

    #[test]
    fn coordinates_reach_pvec_bit_identical_to_the_source() {
        let (_, lines) = read(&drawing(4));
        assert_eq!(lines.len(), 1);
        assert_eq!(bits(lines[0].0), bits(START));
        assert_eq!(bits(lines[0].1), bits(END));
    }

    #[test]
    fn inches_and_millimetres_differ_in_unit_only() {
        let (inch_unit, inch) = read(&drawing(1));
        let (mm_unit, mm) = read(&drawing(4));
        assert_eq!(inch_unit, SourceUnit::Inches);
        assert_eq!(mm_unit, SourceUnit::Millimeters);
        assert_ne!(inch_unit, mm_unit);
        assert_eq!(inch.len(), mm.len());
        for (a, b) in inch.iter().zip(&mm) {
            assert_eq!(bits(a.0), bits(b.0));
            assert_eq!(bits(a.1), bits(b.1));
        }
    }

    #[test]
    fn code_zero_is_unitless_and_an_unlisted_code_is_unknown_and_neither_fails() {
        let (unit, lines) = read(&drawing(0));
        assert_eq!(unit, SourceUnit::Unitless);
        assert_eq!(lines.len(), 1);
        let (unit, lines) = read(&drawing(99));
        assert_eq!(unit, SourceUnit::Unknown);
        assert!(!unit.is_physical());
        assert_eq!(lines.len(), 1, "an unknown unit does not fail the read");
        assert_eq!(bits(lines[0].0), bits(START));
    }
}
