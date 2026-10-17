//! The `acadsharp-rs` provider behind `CadDecoder` (issue #1030).
//!
//! Two kinds of test live here. The ones that need no native library (the
//! version stamp, the feature wiring) always run. The ones that open a drawing
//! need the NativeAOT archive `acadsharp-rs` links; without one they say so and
//! return, unless `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1` is set, in which case a
//! missing archive is a failure. A skipped lane is the same colour as a passing
//! one, so the gate that is meant to exercise the native path sets the variable.
#![cfg(feature = "dwg-acadsharp")]

use std::path::Path;

use libviprs::cad::acadsharp::{
    AcadSharpDecoder, ProviderLimits, SUPPORTED_DWG_VERSIONS, is_supported_version,
};
use libviprs::cad::{
    CadDecoder, CadError, CadSource, CollectSink, DecodeReport, DiagnosticCode, PrimitiveKind,
    ViewKind,
};

const SAMPLE: &[u8] = include_bytes!("fixtures/cad/sample_AC1018.dwg");

fn decoder() -> Option<AcadSharpDecoder> {
    match AcadSharpDecoder::new() {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "the native lane was required and the decoder did not start: {e:?}"
            );
            eprintln!("skipped: no native acadsharp library linked ({e})");
            None
        }
    }
}

#[test]
fn the_supported_stamps_are_r14_through_2018() {
    for stamp in SUPPORTED_DWG_VERSIONS {
        assert!(is_supported_version(stamp.as_slice()));
    }
    for bad in [
        &b"AC1009"[..],
        b"AC1032 ",
        b"AC1035",
        b"AC1012",
        b"ac1018",
        b"",
        b"AC10",
        b"%PDF-1",
    ] {
        assert!(!is_supported_version(bad), "{bad:?} must not be accepted");
    }
}

#[test]
fn probe_reads_only_the_stamp() {
    let Some(d) = decoder() else { return };
    assert!(d.probe(CadSource::Bytes(SAMPLE)).unwrap());
    assert!(!d.probe(CadSource::Bytes(b"AC1035xxxxxx")).unwrap());
    assert!(!d.probe(CadSource::Bytes(b"AC10")).unwrap());
    assert!(!d.probe(CadSource::Bytes(&[])).unwrap());

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/sample_AC1018.dwg");
    assert!(d.probe(CadSource::Path(&path)).unwrap());

    let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/absent.dwg");
    assert!(matches!(
        d.probe(CadSource::Path(&missing)),
        Err(CadError::Io(_))
    ));
}

#[test]
fn a_real_dwg_opens_lists_views_and_streams_primitives() {
    let Some(d) = decoder() else { return };
    let drawing = d.open(CadSource::Bytes(SAMPLE)).expect("the sample opens");
    let views = drawing.views().expect("views");
    assert!(!views.is_empty());
    assert_eq!(views[0].kind(), ViewKind::Model, "Model Space comes first");
    assert_eq!(views[0].name(), "Model");

    let mut sink = CollectSink::new();
    let mut report = DecodeReport::new();
    drawing.decode(0, &mut sink, &mut report).expect("decode");

    assert!(sink.is_finished(), "the implementation finishes the sink");
    assert!(report.is_complete(), "the provider's totals agreed");
    assert!(!report.is_truncated());
    assert!(!sink.is_empty());
    assert_eq!(
        report.counts().total(),
        sink.len() as u64,
        "everything delivered was counted, and nothing else"
    );
    // The sample has lines and text; a decoder that produced only one kind of
    // shape has lost something.
    assert!(report.counts().get(PrimitiveKind::Line) > 0);
    assert!(report.counts().get(PrimitiveKind::Text) > 0);
    assert!(
        report.counts().get(PrimitiveKind::Polygon) > 0,
        "closed boundaries cross as Polygons"
    );
    // Measured on native x86_64 against the `acadsharp-3.7.1-viprs.1` archive.
    // The numbers are the sample's own, so a change here is a change in what
    // the provider emits and wants a reason.
    let counts: Vec<_> = report.counts().iter().collect();
    assert_eq!(
        counts,
        vec![
            (PrimitiveKind::Line, 90),
            (PrimitiveKind::Polyline, 25),
            (PrimitiveKind::Arc, 3),
            (PrimitiveKind::Circle, 15),
            (PrimitiveKind::Ellipse, 1),
            (PrimitiveKind::Spline, 2),
            (PrimitiveKind::Polygon, 8),
            (PrimitiveKind::Text, 70),
        ]
    );
    let mut by_code = std::collections::BTreeMap::<u32, usize>::new();
    for diag in report.diagnostics() {
        *by_code.entry(diag.code().get()).or_default() += 1;
    }
    assert_eq!(
        by_code,
        std::collections::BTreeMap::from([
            (DiagnosticCode::UNSUPPORTED_ENTITY.get(), 88),
            (DiagnosticCode::READER_NOTIFICATION.get(), 24),
            (DiagnosticCode::TEXT_NOT_RECOVERED.get(), 8),
        ]),
        "nothing the reader could not draw may go unreported"
    );
    assert_eq!(report.dropped_diagnostics(), 0);
}

#[test]
fn path_and_bytes_give_the_same_primitives() {
    let Some(d) = decoder() else { return };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/sample_AC1018.dwg");
    let mut a = CollectSink::new();
    let mut b = CollectSink::new();
    d.open(CadSource::Path(&path))
        .unwrap()
        .decode(0, &mut a, &mut DecodeReport::new())
        .unwrap();
    d.open(CadSource::Bytes(SAMPLE))
        .unwrap()
        .decode(0, &mut b, &mut DecodeReport::new())
        .unwrap();
    assert_eq!(a.collected(), b.collected());
}

#[test]
fn warnings_become_diagnostics_with_their_codes_and_the_view() {
    let Some(d) = decoder() else { return };
    let drawing = d.open(CadSource::Bytes(SAMPLE)).unwrap();
    let mut report = DecodeReport::new();
    drawing
        .decode(0, &mut CollectSink::new(), &mut report)
        .unwrap();
    assert!(
        !report.diagnostics().is_empty(),
        "the sample draws entities the reader cannot represent, and that has to be said"
    );
    for diag in report.diagnostics() {
        assert_eq!(diag.view(), Some(0));
        assert_ne!(
            diag.code(),
            DiagnosticCode::DECODE_FAILED,
            "a decode that finished must not file a failure: {}",
            diag.message()
        );
    }
    // The provider's own codes keep their numbers.
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|d| d.code() == DiagnosticCode::UNSUPPORTED_ENTITY),
        "the sample has entities the reader names as unsupported"
    );
}

#[test]
fn garbage_is_unrecognised_not_a_panic() {
    let Some(d) = decoder() else { return };
    for bytes in [
        &b"not a drawing at all"[..],
        b"AC1018",
        b"AC1035\0\0\0\0\0\0",
    ] {
        let err = d.open(CadSource::Bytes(bytes)).err().expect("must refuse");
        assert!(
            matches!(
                err,
                CadError::UnrecognisedSource { .. } | CadError::Provider { .. }
            ),
            "{err:?}"
        );
    }
    assert!(matches!(
        d.open(CadSource::Bytes(&[])).err().unwrap(),
        CadError::UnrecognisedSource { .. }
    ));
}

#[test]
fn a_truncated_dwg_is_an_error_or_a_reported_gap_never_a_clean_complete() {
    let Some(d) = decoder() else { return };
    let cut = &SAMPLE[..SAMPLE.len() / 2];
    match d.open(CadSource::Bytes(cut)) {
        Err(_) => {}
        Ok(drawing) => {
            let mut report = DecodeReport::new();
            let r = drawing.decode(0, &mut CollectSink::new(), &mut report);
            assert!(
                r.is_err() || !report.diagnostics().is_empty() || !report.is_complete(),
                "half a file decoded as if nothing had gone wrong"
            );
        }
    }
}

#[test]
fn a_view_that_does_not_exist_is_no_such_view() {
    let Some(d) = decoder() else { return };
    let drawing = d.open(CadSource::Bytes(SAMPLE)).unwrap();
    let n = drawing.views().unwrap().len() as u32;
    let err = drawing
        .decode(n + 40, &mut CollectSink::new(), &mut DecodeReport::new())
        .unwrap_err();
    assert!(matches!(err, CadError::NoSuchView { index, views } if index == n + 40 && views == n));
}

#[test]
fn limits_reach_the_reader() {
    let Some(d) = decoder() else { return };
    let tight = d.with_limits(ProviderLimits {
        max_input_bytes: Some(1024),
        ..ProviderLimits::default()
    });
    assert!(tight.open(CadSource::Bytes(SAMPLE)).is_err());
}

#[test]
fn the_synthetic_document_round_trips() {
    let Some(d) = decoder() else { return };
    let drawing = d.open(CadSource::Bytes(b"VIPRSSYN")).unwrap();
    assert!(!drawing.views().unwrap().is_empty());
    let mut report = DecodeReport::new();
    let mut sink = CollectSink::new();
    drawing.decode(0, &mut sink, &mut report).unwrap();
    assert!(report.is_complete());
    let _ = DiagnosticCode::UNSUPPORTED_ENTITY;
}
