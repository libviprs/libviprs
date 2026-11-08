//! DXF through the acadrust provider (#1561, part of the layouts epic #1557).
//!
//! acadrust reads DXF into the same `CadDocument` it reads a DWG into, so the provider sniffs a
//! DXF (the text form's first `0`/`SECTION` pair, or the binary form's 22 byte sentinel) beside
//! the DWG version stamp and opens it with the DXF reader. Everything after the open is the DWG
//! path unchanged.
//!
//! The twins in `tests/fixtures/pvec-dxf` are written from gold fixtures' `source.dwg` by the
//! pinned acadrust's own DXF writer, so a twin is the same drawing as its DWG by construction and
//! the gate baselines carry a row for each next to the DWG's. `PVEC_DXF_TWINS_BLESS=1` rewrites
//! them; run that with `--test pvec_1561_dxf the_twins_are` so no other cell reads a twin while it
//! is being written. They are a round trip through one implementation, not known truth about DXF.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gate.rs"]
mod pvec_gate;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::io::Cursor;
use std::path::Path;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind, ProviderError};
use pvec_gold::Corpus;

/// What a drawing looks like through the provider with the handles left out, which a writer is
/// free to renumber: layers, layouts and every model-space entity's geometry and style.
fn shape(p: &AcadrustProvider) -> String {
    let mut layers: Vec<String> = p
        .layers()
        .unwrap()
        .iter()
        .map(|l| format!("{} off={} frozen={}", l.name, l.off, l.frozen))
        .collect();
    layers.sort();
    let all = p.layouts().unwrap();
    let layouts: Vec<String> = all
        .iter()
        .map(|l| format!("{} {:?}", l.name, l.kind))
        .collect();
    let model = all
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model space");
    let entities: Vec<String> = p
        .entities(&EntityScope::Layout(model.id))
        .unwrap()
        .collect_all()
        .unwrap()
        .iter()
        .map(|e| format!("{:?} {:?}", e.geometry, e.style))
        .collect();
    format!("{layers:#?}\n{layouts:#?}\n{entities:#?}")
}

#[test]
#[cfg_attr(miri, ignore)] // reads and may write the twins on disk, which Miri isolation blocks
fn the_twins_are_the_listed_gold_drawings_written_by_the_pinned_dxf_writer() {
    let text = [
        "bspline-d3-001",
        "circle-001",
        "ellipse-001",
        "layer-state-001",
        "line-001",
        "lwpolyline-closed-001",
        "mixed-entity-001",
        "nested-blocks-001",
        "paper-space-multi-viewport-001",
        "viewport-clipped-001",
    ];
    let binary = ["mixed-entity-001"];
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-dxf");
    if std::env::var_os("PVEC_DXF_TWINS_BLESS").is_some() {
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = Corpus::open();
        let read = |id: &str| {
            acadrust::DwgReader::from_stream(Cursor::new(corpus.get(id).dwg()))
                .read()
                .unwrap_or_else(|e| panic!("{id}: {e}"))
        };
        for id in text {
            let bytes = acadrust::DxfWriter::new(&read(id)).write_to_vec().unwrap();
            std::fs::write(dir.join(format!("{id}.dxf")), bytes).unwrap();
        }
        for id in binary {
            let bytes = acadrust::DxfWriter::new_binary(&read(id))
                .write_to_vec()
                .unwrap();
            std::fs::write(dir.join(format!("{id}.binary.dxf")), bytes).unwrap();
        }
    }
    let mut want: Vec<String> = text
        .iter()
        .map(|id| format!("{id}.dxf"))
        .chain(binary.iter().map(|id| format!("{id}.binary.dxf")))
        .collect();
    want.sort();
    let twins = pvec_gate::dxf_twins();
    let rows: Vec<&String> = twins.iter().map(|t| &t.0).collect();
    assert_eq!(rows, want.iter().collect::<Vec<_>>());
    for (row, _, bytes) in &twins {
        let binary_form = bytes.starts_with(b"AutoCAD Binary DXF\r\n\x1a\x00");
        assert_eq!(binary_form, row.ends_with(".binary.dxf"), "{row}");
    }
}

#[test]
fn probe_accepts_text_and_binary_dxf_beside_the_dwg_stamps() {
    let yes: [&[u8]; 6] = [
        b"0\nSECTION\n",
        b"  0\nSECTION\n  2\nHEADER\n",
        b"  0\r\nSECTION\r\n  2\r\nHEADER\r\n",
        b"999\nwritten by hand\n  0\nSECTION\n",
        b"AutoCAD Binary DXF\r\n\x1a\x00\x00\x00SECTION\x00",
        b"AC1032\0\0\0\0",
    ];
    for bytes in yes {
        assert!(
            AcadrustProvider::probe(CadSource::Bytes(bytes)).unwrap(),
            "{:?}",
            String::from_utf8_lossy(bytes)
        );
    }
    // The reader cannot open a byte order mark, a blank line before the first pair or a
    // padded value, so none of them is sniffed as a DXF.
    let no: [&[u8]; 12] = [
        b"",
        b"\xEF\xBB\xBF  0\nSECTION\n",
        b"\n\n  0\nSECTION\n",
        b"  0\nSECTION \n",
        b"0",
        b"0\nSECT",
        b"  0\nENTITIES\n",
        b"  1\nSECTION\n",
        b"999\nonly a comment\n",
        b"AutoCAD Binary DXF\r\n",
        b"AC1009\0\0\0\0",
        b"%PDF-1.7",
    ];
    for bytes in no {
        assert!(
            !AcadrustProvider::probe(CadSource::Bytes(bytes)).unwrap(),
            "{:?}",
            String::from_utf8_lossy(bytes)
        );
    }
}

#[test]
fn open_tells_a_dxf_header_with_nothing_after_it_from_a_file_that_is_not_a_drawing() {
    match AcadrustProvider::open(CadSource::Bytes(b"  0\nSECTION\n")) {
        Err(ProviderError::Open { .. }) => {}
        Err(e) => panic!("a sniffed DXF that does not read is an Open error: {e:?}"),
        Ok(_) => panic!("opened a DXF header with nothing after it"),
    }
    match AcadrustProvider::open(CadSource::Bytes(b"  0\nNOTHING\n")) {
        Err(ProviderError::Unrecognised { detail, .. }) => {
            assert!(detail.contains("DXF"), "the refusal names DXF: {detail}");
        }
        Err(e) => panic!("not a drawing is Unrecognised: {e:?}"),
        Ok(_) => panic!("opened something that is not a drawing"),
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the twins from disk
fn every_twin_probes_and_opens_from_bytes_and_from_a_path_alike() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-dxf");
    for (row, _, bytes) in pvec_gate::dxf_twins() {
        let path = dir.join(&row);
        assert!(
            AcadrustProvider::probe(CadSource::Path(&path)).unwrap(),
            "{row}"
        );
        assert!(
            AcadrustProvider::probe(CadSource::Bytes(&bytes)).unwrap(),
            "{row}"
        );
        let a = AcadrustProvider::open(CadSource::Bytes(&bytes))
            .unwrap_or_else(|e| panic!("{row} from bytes: {e}"));
        let b = AcadrustProvider::open(CadSource::Path(&path))
            .unwrap_or_else(|e| panic!("{row} from a path: {e}"));
        assert_eq!(a.name(), "acadrust");
        assert_eq!(shape(&a), shape(&b), "{row}");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the twins and the gold drawings from disk
fn every_twin_shows_the_same_layers_layouts_and_model_space_as_its_dwg() {
    let corpus = Corpus::open();
    for (row, fixture, bytes) in pvec_gate::dxf_twins() {
        let dwg = corpus.get(&fixture).dwg();
        let from_dwg = AcadrustProvider::open(CadSource::Bytes(&dwg)).unwrap();
        let from_dxf = AcadrustProvider::open(CadSource::Bytes(&bytes))
            .unwrap_or_else(|e| panic!("{row}: {e}"));
        assert_eq!(shape(&from_dxf), shape(&from_dwg), "{row}");
        assert_eq!(
            from_dxf.source_unit(),
            from_dwg.source_unit(),
            "{row}: the unit"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the checked-in baselines
fn the_twin_rows_of_the_classification_and_view_baselines_equal_their_dwg_rows() {
    let twins = pvec_gate::dxf_twins();
    for name in ["classification", "view-decisions"] {
        let text = std::fs::read_to_string(pvec_gate::baseline_path(name)).unwrap();
        let row = |id: &str| -> String {
            text.lines()
                .find(|l| l.split_whitespace().next() == Some(id))
                .unwrap_or_else(|| panic!("{name}: no row for {id}"))
                .split_once(' ')
                .map_or(String::new(), |(_, rest)| rest.to_owned())
        };
        for (twin, fixture, _) in &twins {
            assert_eq!(row(twin), row(fixture), "{name}: {twin}");
        }
    }
}
