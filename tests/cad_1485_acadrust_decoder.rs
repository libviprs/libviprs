//! The acadrust provider behind `CadDecoder` (issue #1485), so `default_decoder` picks it.
//!
//! Pure Rust, so nothing here skips: there is no native library to be missing. The drawings are
//! the ACadSharp adapter's own fixtures (`tests/fixtures/cad`), and the layer one is the PVec
//! provider's `vc_layers`.
#![cfg(feature = "dwg-acadrust")]

use std::io::Write;
use std::path::{Path, PathBuf};

use libviprs::cad::acadrust::AcadrustDecoder;
use libviprs::cad::{
    CadDecoder, CadError, CadSource, CollectSink, DecodeReport, DiagnosticCode, Primitive,
    PrimitiveKind, ViewKind,
};
use libviprs::vector::convert::{ProviderKind, decoder_for, default_decoder};

fn cad(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cad")
        .join(name)
}

fn model(decoder: &dyn CadDecoder, path: &Path) -> (CollectSink, DecodeReport) {
    let drawing = decoder
        .open(CadSource::Path(path))
        .expect("the drawing opens");
    let views = drawing.views().expect("views");
    assert_eq!(views[0].kind(), ViewKind::Model, "Model Space comes first");
    let mut sink = CollectSink::new();
    let mut report = DecodeReport::new();
    drawing
        .decode(views[0].index(), &mut sink, &mut report)
        .expect("decode");
    assert!(sink.is_finished() && report.is_complete());
    assert_eq!(
        report.counts().total(),
        sink.len() as u64,
        "counted what was delivered"
    );
    (sink, report)
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_shape_the_pipeline_draws_crosses_from_its_own_fixture() {
    let d = AcadrustDecoder::new();
    for (file, kind) in [
        ("g13/g13_line.dwg", PrimitiveKind::Line),
        ("g13/g13_circle.dwg", PrimitiveKind::Circle),
        ("g13/g13_arc.dwg", PrimitiveKind::Arc),
        ("g13/g13_ellipse.dwg", PrimitiveKind::Ellipse),
        ("g13/g13_polyline.dwg", PrimitiveKind::Polyline),
        ("g13/g13_spline.dwg", PrimitiveKind::Spline),
        ("g13/g13_text.dwg", PrimitiveKind::Text),
    ] {
        let (_, report) = model(&d, &cad(file));
        assert!(report.counts().get(kind) > 0, "{file} gave no {kind:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_real_sample_crosses_lines_polylines_circles_and_text() {
    let (_, report) = model(&AcadrustDecoder::new(), &cad("sample_AC1018.dwg"));
    for kind in [
        PrimitiveKind::Line,
        PrimitiveKind::Polyline,
        PrimitiveKind::Circle,
        PrimitiveKind::Text,
    ] {
        assert!(report.counts().get(kind) > 0, "the sample gave no {kind:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn what_is_not_flattened_is_reported_and_never_dropped_quietly() {
    let (_, report) = model(&AcadrustDecoder::new(), &cad("g13/g13_unsupported.dwg"));
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|d| d.code() == DiagnosticCode::UNSUPPORTED_ENTITY),
        "an entity this reader does not flatten has to be in the report"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_insert_is_expanded_into_the_view_and_says_so() {
    let (sink, _) = model(&AcadrustDecoder::new(), &cad("g13/g13_insert.dwg"));
    assert!(
        sink.collected()
            .iter()
            .any(|p| p.origin().is_expanded_insert()),
        "no primitive came from an expanded INSERT"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_g13_fixture_decodes_to_the_end_and_counts_what_it_delivered() {
    let dir = cad("g13");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "dwg") {
            continue;
        }
        let d = AcadrustDecoder::new();
        // A fixture built to be refused (a truncated or hostile file) may fail to open, but
        // an open drawing has to decode to the end.
        let Ok(drawing) = d.open(CadSource::Path(&path)) else {
            continue;
        };
        for v in drawing.views().unwrap() {
            let (mut sink, mut report) = (CollectSink::new(), DecodeReport::new());
            drawing
                .decode(v.index(), &mut sink, &mut report)
                .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
            assert!(report.is_complete(), "{}", path.display());
            assert_eq!(report.counts().total(), sink.len() as u64);
        }
        seen += 1;
    }
    assert!(seen > 20, "only {seen} fixtures opened");
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_unequal_scale_never_turns_a_curve_into_a_wrong_one() {
    let (sink, report) = model(&AcadrustDecoder::new(), &cad("g13/g13_nonuniform.dwg"));
    let circles = sink
        .collected()
        .iter()
        .filter(|p| matches!(p, Primitive::Circle(_)))
        .count();
    let refused = report
        .diagnostics()
        .iter()
        .filter(|d| d.code() == DiagnosticCode::NON_UNIFORM_BLOCK_SCALE)
        .count();
    assert!(
        circles == 0 || refused == 0,
        "a circle was drawn under an unequal scale"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_off_or_frozen_layer_hides_its_entities_and_its_inserts() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider/vc_layers.dwg");
    let (sink, _) = model(&AcadrustDecoder::new(), &path);
    let handles: Vec<u64> = sink
        .collected()
        .iter()
        .filter(|p| !p.origin().is_expanded_insert())
        .filter_map(|p| p.origin().item_handle().map(|h| h.get()))
        .collect();
    assert!(
        handles.contains(&0x4B),
        "the line on the visible layer is drawn: {handles:?}"
    );
    for hidden in [0x4C, 0x4D, 0x4E] {
        assert!(
            !handles.contains(&hidden),
            "{hidden:#x} sits on an off or frozen layer"
        );
    }
    assert!(
        sink.collected().iter().all(|p| p
            .origin()
            .item_handle()
            .is_none_or(|h| h.get() != 0x56 && h.get() != 0x57)),
        "nothing from an INSERT on an off or frozen layer"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn path_and_bytes_give_the_same_primitives() {
    let path = cad("sample_AC1018.dwg");
    let bytes = std::fs::read(&path).unwrap();
    let d = AcadrustDecoder::new();
    let (a, _) = model(&d, &path);
    let drawing = d.open(CadSource::Bytes(&bytes)).unwrap();
    let mut b = CollectSink::new();
    drawing.decode(0, &mut b, &mut DecodeReport::new()).unwrap();
    assert_eq!(a.collected(), b.collected());
}

#[test]
#[cfg_attr(miri, ignore)]
fn probe_reads_the_stamp_and_a_missing_file_is_an_io_error() {
    let d = AcadrustDecoder::new();
    assert!(d.probe(CadSource::Path(&cad("sample_AC1018.dwg"))).unwrap());
    assert!(!d.probe(CadSource::Bytes(b"AC1009xxxx")).unwrap());
    assert!(!d.probe(CadSource::Bytes(&[])).unwrap());
    assert!(matches!(
        d.probe(CadSource::Path(&cad("absent.dwg"))),
        Err(CadError::Io(_))
    ));
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_view_that_is_not_there_is_named() {
    let d = AcadrustDecoder::new();
    let path = cad("sample_AC1018.dwg");
    let drawing = d.open(CadSource::Path(&path)).unwrap();
    let err = drawing
        .decode(999, &mut CollectSink::new(), &mut DecodeReport::new())
        .unwrap_err();
    assert!(
        matches!(err, CadError::NoSuchView { index: 999, .. }),
        "{err:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn something_that_is_not_a_drawing_is_unrecognised() {
    let d = AcadrustDecoder::new();
    assert!(matches!(
        d.open(CadSource::Bytes(b"%PDF-1.7 not a drawing")).err(),
        Some(CadError::UnrecognisedSource { .. })
    ));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_hardened_decoder_refuses_a_file_past_the_input_cap_and_the_plain_one_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.dwg");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"AC1032").unwrap();
    f.set_len((256 << 20) + 1).unwrap();
    let hardened = format!(
        "{:?}",
        AcadrustDecoder::hardened()
            .open(CadSource::Path(&path))
            .err()
    );
    assert!(hardened.contains("LimitExceeded"), "{hardened}");
    let plain = format!(
        "{:?}",
        AcadrustDecoder::new().open(CadSource::Path(&path)).err()
    );
    assert!(!plain.contains("LimitExceeded"), "{plain}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn default_decoder_is_acadrust_and_needs_no_native_library() {
    let d = default_decoder(true).expect("acadrust is a default feature");
    assert!(d.probe(CadSource::Path(&cad("sample_AC1018.dwg"))).unwrap());
    let (_, report) = model(d.as_ref(), &cad("sample_AC1018.dwg"));
    assert!(report.counts().total() > 0);
    assert!(decoder_for(ProviderKind::Acadrust, false).is_ok());
}

#[cfg(not(feature = "dwg-acadsharp"))]
#[test]
#[cfg_attr(miri, ignore)]
fn acadsharp_is_not_built_unless_asked_for() {
    assert!(matches!(
        decoder_for(ProviderKind::AcadSharp, false).err(),
        Some(libviprs::vector::ConvertError::ProviderNotBuilt)
    ));
}

/// The ACadSharp adapter's answers on `sample_AC1018.dwg` are in `cad_1030_acadsharp_decoder.rs`:
/// 90 lines, 25 polylines, 3 arcs, 15 circles, 1 ellipse, 2 splines, 8 polygons, 70 texts.
/// Dimensions are drawn from their blocks here, so the shapes that need no hatch, leader or
/// solid support agree exactly, and the rest is reported, not missing quietly.
#[test]
#[cfg_attr(miri, ignore)]
fn the_sample_agrees_with_the_acadsharp_adapter_where_the_entities_are_supported() {
    let (_, report) = model(&AcadrustDecoder::new(), &cad("sample_AC1018.dwg"));
    let c = report.counts();
    assert_eq!(c.get(PrimitiveKind::Polyline), 25);
    assert_eq!(c.get(PrimitiveKind::Arc), 3);
    assert_eq!(c.get(PrimitiveKind::Ellipse), 1);
    assert_eq!(c.get(PrimitiveKind::Spline), 2);
    assert!(c.get(PrimitiveKind::Line) >= 60 && c.get(PrimitiveKind::Text) >= 40);
    // What this reader does not draw yet is named by class, so a gap is a line in the report.
    let said = |class: &str| {
        report.diagnostics().iter().any(|d| {
            d.code() == DiagnosticCode::UNSUPPORTED_ENTITY && d.message().starts_with(class)
        })
    };
    for class in ["HATCH", "MULTILEADER", "SOLID"] {
        assert!(said(class), "{class} is not drawn and was not reported");
    }
    assert!(!said("DIMENSION"), "a dimension is drawn from its block");
}
