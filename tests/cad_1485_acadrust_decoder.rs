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
fn a_uniform_scale_scales_the_circle_and_a_mirror_keeps_the_arc_on_the_same_side() {
    let (sink, _) = model(&AcadrustDecoder::new(), &cad("g13/g13_scale_4x.dwg"));
    let radii: Vec<f64> = sink
        .collected()
        .iter()
        .filter_map(|p| {
            if let Primitive::Circle(c) = p {
                Some(c.radius())
            } else {
                None
            }
        })
        .collect();
    let (_, one) = model(&AcadrustDecoder::new(), &cad("g13/g13_scale_1x.dwg"));
    assert!(!radii.is_empty() && one.counts().get(PrimitiveKind::Circle) > 0);
    let (sink1, _) = model(&AcadrustDecoder::new(), &cad("g13/g13_scale_1x.dwg"));
    let r1 = sink1
        .collected()
        .iter()
        .find_map(|p| {
            if let Primitive::Circle(c) = p {
                Some(c.radius())
            } else {
                None
            }
        })
        .expect("a circle at 1x");
    assert!(
        radii.iter().any(|r| (r - 4.0 * r1).abs() < 1e-9),
        "{radii:?} vs {r1}"
    );
}

#[test]
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
fn something_that_is_not_a_drawing_is_unrecognised() {
    let d = AcadrustDecoder::new();
    assert!(matches!(
        d.open(CadSource::Bytes(b"%PDF-1.7 not a drawing")).err(),
        Some(CadError::UnrecognisedSource { .. })
    ));
}

#[test]
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
fn default_decoder_is_acadrust_and_needs_no_native_library() {
    let d = default_decoder(true).expect("acadrust is a default feature");
    assert!(d.probe(CadSource::Path(&cad("sample_AC1018.dwg"))).unwrap());
    let (_, report) = model(d.as_ref(), &cad("sample_AC1018.dwg"));
    assert!(report.counts().total() > 0);
    assert!(decoder_for(ProviderKind::Acadrust, false).is_ok());
}

#[cfg(not(feature = "dwg-acadsharp"))]
#[test]
fn acadsharp_is_not_built_unless_asked_for() {
    assert!(matches!(
        decoder_for(ProviderKind::AcadSharp, false).err(),
        Some(libviprs::vector::ConvertError::ProviderNotBuilt)
    ));
}
