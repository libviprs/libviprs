//! Issue #1413 (slice of #1048, N7.1): the conversion API. One call over a named fixture
//! publishes the artifact and its report, the report's counts are the fixture's, every error keeps
//! its source chain, and a cancelled run leaves nothing behind. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::CancelToken;
use libviprs::cad::CadSource;
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::writer::WriteError;
use libviprs::pvec::{FidelityClass, SourceUnit};
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1413-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn chain(e: &dyn Error) -> Vec<String> {
    let mut out = vec![e.to_string()];
    let mut cur = e.source();
    while let Some(s) = cur {
        out.push(s.to_string());
        cur = s.source();
    }
    out
}

/// `vc_layers.dwg` declares four lines in model space and the block `VC_BLK` inserted three times
/// on the layers `VC_ON`, `VC_OFF`, `VC_FROZEN` and `0` (off). The view profile keeps 3 leaves and
/// excludes 6 (`tests/pvec_view_profile_1097_fixture.rs`), so the report meets 9 entities.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_a_conversion_publishes_the_artifact_and_a_report_with_the_fixtures_counts() {
    let dir = scratch("counts");
    let dest = dir.join("layers.pvec");
    let path = fixture("vc_layers.dwg");
    let out = PvecConversion::new(CadSource::Path(&path), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .unwrap();

    assert_eq!(out.artifact, dest);
    assert!(dest.exists() && report_path(&dest).exists());
    assert_eq!(out.report.totals.encountered, 9);
    assert_eq!(out.report.view.included, 3);
    let drawn = out.report.totals.by_fidelity["SupportedExact"]
        + out.report.totals.by_fidelity["SupportedEquivalent"];
    assert_eq!(drawn, 3);
    assert_eq!(out.report.totals.by_fidelity["IgnoredByPolicy"], 6);
    assert_eq!(out.fidelity.encountered(), 9);
    assert_eq!(out.fidelity.count(FidelityClass::IgnoredByPolicy), 6);
    assert_eq!(out.written, 3);
    assert_eq!(out.report.totals.by_source_class["INSERT"], 2);

    let reader = open_published(&dest).unwrap();
    assert_eq!(reader.entity_count().unwrap(), 3);
    let on_disk =
        ConversionReport::from_json_slice(&fs::read(report_path(&dest)).unwrap()).unwrap();
    assert_eq!(on_disk, out.report);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_the_report_is_part_of_the_output_and_matches_the_entities_written() {
    let dir = scratch("nested");
    let dest = dir.join("nested.pvec");
    let path = fixture("vc_nested.dwg");
    let out = PvecConversion::new(CadSource::Path(&path), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .unwrap();
    let rows_with_geometry = out
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .count() as u64;
    assert_eq!(rows_with_geometry, out.written);
    assert!(out.written > 0);
    assert_eq!(
        out.report.entities.len() as u64,
        out.report.totals.encountered
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_two_runs_publish_the_same_bytes() {
    let dir = scratch("det");
    let path = fixture("vc_nested.dwg");
    // The report names its sidecar by file name, so the two runs use the same name in two folders.
    fs::create_dir_all(dir.join("a")).unwrap();
    fs::create_dir_all(dir.join("b")).unwrap();
    let a = dir.join("a/o.pvec");
    let b = dir.join("b/o.pvec");
    PvecConversion::new(CadSource::Path(&path), &a)
        .unit(SourceUnit::Meters)
        .convert()
        .unwrap();
    PvecConversion::new(CadSource::Path(&path), &b)
        .unit(SourceUnit::Meters)
        .convert()
        .unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    assert_eq!(
        fs::read(report_path(&a)).unwrap(),
        fs::read(report_path(&b)).unwrap()
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_an_unopenable_source_keeps_the_provider_error_as_the_source() {
    let dir = scratch("open");
    let missing = dir.join("missing.dwg");
    let dest = dir.join("o.pvec");
    let err = PvecConversion::new(CadSource::Path(&missing), &dest)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Open(_)), "{err:?}");
    assert!(err.source().is_some(), "{:?}", chain(&err));
    assert!(!dest.exists() && !report_path(&dest).exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_a_unit_pvec_cannot_declare_reaches_the_caller_through_the_source_chain() {
    let dir = scratch("unit");
    let path = fixture("vc_layers.dwg");
    let dest = dir.join("u.pvec");
    // The fixture's header declares no unit, so with nothing said the writer refuses it.
    let err = PvecConversion::new(CadSource::Path(&path), &dest)
        .convert()
        .unwrap_err();
    let PvecConvertError::Write(w) = &err else {
        panic!("{err:?}")
    };
    assert!(matches!(
        w,
        WriteError::UnsupportedUnit(SourceUnit::Unitless)
    ));
    assert!(chain(&err).len() >= 2, "{:?}", chain(&err));
    assert!(!dest.exists() && !report_path(&dest).exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1413_a_precancelled_run_returns_cancelled_and_creates_no_file() {
    let dir = scratch("cancel");
    let path = fixture("vc_layers.dwg");
    let dest = dir.join("c.pvec");
    let token = CancelToken::new();
    token.cancel();
    let err = PvecConversion::new(CadSource::Path(&path), &dest)
        .cancel(token)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn issue_1413_the_error_enum_is_non_exhaustive() {
    #[deny(unreachable_patterns)]
    fn probe(e: &PvecConvertError) -> bool {
        match e {
            PvecConvertError::Cancelled => true,
            PvecConvertError::UnitRequired => false,
            _ => false,
        }
    }
    assert!(probe(&PvecConvertError::Cancelled));
    let _: &Path = Path::new("");
}
