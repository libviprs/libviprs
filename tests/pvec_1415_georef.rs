//! Issue #1415 (slice of #1048, N7.3): the georeference handoff. A calibration changes the sidecar
//! and nothing else, the sidecar round-trips and refuses a version it does not know, its bounds are
//! the artifact's bit for bit, an undeclared unit stays `Unknown`, and no GIS transform reaches the
//! artifact. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceUnit,
};
use libviprs::pvec_convert::{
    GeoreferenceError, GeoreferenceHandoff, PvecConversion, PvecConvertError, georef_path,
};

/// Two circles, centres (5, 5) and (20, 5), radii 2 and 1: the drawing box is exactly
/// (3, 3) to (21, 7).
struct Mock {
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn circle(handle: u64, x: f64, r: f64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Circle {
            center: [x, 5.0, 0.0],
            radius: r,
            normal: [0.0, 0.0, 1.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "CIRCLE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Ok(Mock {
            caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            }),
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }])
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(Vec::new())
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        match scope {
            EntityScope::Layout(_) => Ok(Box::new(Cursor(
                vec![circle(1, 5.0, 2.0), circle(2, 20.0, 1.0)].into_iter(),
            ))),
            _ => Err(ProviderError::NoSuchScope("no blocks".into())),
        }
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1415-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn mock<'a>(dest: &'a Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Bytes(b"x"), dest).unit(SourceUnit::Meters)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_two_calibrations_give_identical_artifact_bytes_and_different_sidecars() {
    let dir = scratch("twice");
    let (a, b) = (dir.join("a.pvec"), dir.join("b.pvec"));
    mock(&a)
        .calibration("site-survey-1")
        .control_points([[3.0, 3.0], [21.0, 7.0]])
        .convert_with::<Mock>()
        .unwrap();
    mock(&b)
        .calibration("site-survey-2")
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    assert_ne!(
        fs::read(georef_path(&a)).unwrap(),
        fs::read(georef_path(&b)).unwrap()
    );
    let sa = GeoreferenceHandoff::read_from(&georef_path(&a)).unwrap();
    assert_eq!(sa.as_v1().calibration_id.as_deref(), Some("site-survey-1"));
    assert_eq!(sa.as_v1().control_points, vec![[3.0, 3.0], [21.0, 7.0]]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_recalibrating_rewrites_the_sidecar_and_nothing_else() {
    let dir = scratch("recal");
    let dest = dir.join("o.pvec");
    mock(&dest)
        .calibration("old")
        .convert_with::<Mock>()
        .unwrap();
    let (artifact, report) = (
        fs::read(&dest).unwrap(),
        fs::read(report_path(&dest)).unwrap(),
    );
    GeoreferenceHandoff::recalibrate(&georef_path(&dest), Some("new".into()), vec![[4.0, 4.0]])
        .unwrap();
    assert_eq!(fs::read(&dest).unwrap(), artifact);
    assert_eq!(fs::read(report_path(&dest)).unwrap(), report);
    let s = GeoreferenceHandoff::read_from(&georef_path(&dest)).unwrap();
    assert_eq!(s.as_v1().calibration_id.as_deref(), Some("new"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_the_sidecar_round_trips_carries_a_version_and_refuses_an_unknown_one() {
    let dir = scratch("round");
    let dest = dir.join("o.pvec");
    mock(&dest).convert_with::<Mock>().unwrap();
    let path = georef_path(&dest);
    let first = GeoreferenceHandoff::read_from(&path).unwrap();
    let copy = dir.join("copy.georef.json");
    first.write_to(&copy).unwrap();
    assert_eq!(fs::read(&copy).unwrap(), fs::read(&path).unwrap());
    assert_eq!(GeoreferenceHandoff::read_from(&copy).unwrap(), first);
    let json = fs::read_to_string(&path).unwrap();
    assert!(json.contains("\"schema_version\": \"1\""), "{json}");

    let future = json.replace("\"schema_version\": \"1\"", "\"schema_version\": \"2\"");
    let err = GeoreferenceHandoff::from_json_slice(future.as_bytes()).unwrap_err();
    assert!(matches!(err, GeoreferenceError::UnknownVersion(v) if v == "2"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_the_sidecar_bounds_are_the_artifacts_bit_for_bit() {
    let dir = scratch("bounds");
    let dest = dir.join("o.pvec");
    mock(&dest).convert_with::<Mock>().unwrap();
    let s = GeoreferenceHandoff::read_from(&georef_path(&dest)).unwrap();
    let b = s.as_v1().bounds.unwrap();
    let bits = |p: [f64; 2]| p.map(f64::to_bits);
    assert_eq!(bits(b.min), bits([3.0, 3.0]));
    assert_eq!(bits(b.max), bits([21.0, 7.0]));
    let manifest = open_published(&dest).unwrap();
    let declared = manifest.manifest().bounds;
    assert_eq!(bits(b.min), bits(declared.min));
    assert_eq!(bits(b.max), bits(declared.max));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_an_undeclared_unit_stays_unknown_and_no_default_is_substituted() {
    let dir = scratch("unit");
    let dest = dir.join("o.pvec");
    // The provider cannot report a unit: the record says Unknown, and the caller's is applied.
    mock(&dest).convert_with::<Mock>().unwrap();
    let u = GeoreferenceHandoff::read_from(&georef_path(&dest))
        .unwrap()
        .as_v1()
        .unit
        .clone();
    assert_eq!(u.source, "Unknown");
    assert_eq!((u.insunits, u.physical), (None, false));
    assert!(u.evidence.contains("no unit"), "{}", u.evidence);
    assert_eq!(u.applied, "Meters");

    // A real DWG whose header says unitless (INSUNITS 0): Unitless, not Meters.
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider/vc_layers.dwg");
    let dest2 = dir.join("dwg.pvec");
    PvecConversion::new(CadSource::Path(&src), &dest2)
        .unit(SourceUnit::Millimeters)
        .convert()
        .unwrap();
    let u = GeoreferenceHandoff::read_from(&georef_path(&dest2))
        .unwrap()
        .as_v1()
        .unit
        .clone();
    assert_eq!((u.source.as_str(), u.insunits), ("Unitless", Some(0)));
    assert_eq!(u.applied, "Millimeters");
    assert!(!u.physical);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_no_transform_crs_or_calibration_reaches_the_artifact() {
    let dir = scratch("scan");
    let dest = dir.join("o.pvec");
    mock(&dest)
        .calibration("calibration-id-that-must-not-leak")
        .control_points([[3.0, 3.0]])
        .convert_with::<Mock>()
        .unwrap();
    let bytes = fs::read(&dest).unwrap();
    for needle in [
        &b"calibration-id-that-must-not-leak"[..],
        b"EPSG",
        b"GeoTransform",
        b"crs",
        b"PROJCS",
    ] {
        assert!(
            !bytes.windows(needle.len()).any(|w| w == needle),
            "{:?} found in the artifact",
            String::from_utf8_lossy(needle)
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_the_report_names_the_sidecar_and_a_bad_control_point_is_refused() {
    let dir = scratch("report");
    let dest = dir.join("o.pvec");
    let out = mock(&dest).convert_with::<Mock>().unwrap();
    assert_eq!(
        out.report.georeference.as_deref(),
        Some("o.pvec.georef.json")
    );
    let on_disk =
        ConversionReport::from_json_slice(&fs::read(report_path(&dest)).unwrap()).unwrap();
    assert_eq!(on_disk.georeference, out.report.georeference);

    let bad = dir.join("bad.pvec");
    let err = mock(&bad)
        .control_points([[0.0, f64::NAN]])
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(
        err,
        PvecConvertError::Georeference(GeoreferenceError::NonFinite { index: 0 })
    ));
    assert!(!bad.exists() && !georef_path(&bad).exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1415_a_failed_run_leaves_the_previous_sidecar_as_it_was() {
    let dir = scratch("atomic");
    let dest = dir.join("o.pvec");
    mock(&dest)
        .calibration("first")
        .convert_with::<Mock>()
        .unwrap();
    let before = fs::read(georef_path(&dest)).unwrap();
    let token = libviprs::CancelToken::new();
    token.cancel();
    let err = mock(&dest)
        .calibration("second")
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled));
    assert_eq!(fs::read(georef_path(&dest)).unwrap(), before);
    assert!(!dir.join("o.pvec.georef.json.partial").exists());
    let _ = fs::remove_dir_all(&dir);
}
