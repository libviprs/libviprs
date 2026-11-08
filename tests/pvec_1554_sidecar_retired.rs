//! Issue #1554 (slice 5 of #1543): the `.georef.json` sidecar is retired. A conversion writes none,
//! the calibration lives only in the artifact, and an old sidecar still imports for one minor
//! release through the application's resolver, which turns its opaque `calibration_id` into a
//! calibration. When nothing resolves, nothing is written. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceUnit,
};
use libviprs::pvec_convert::{
    Calibration, GeoreferenceError, PvecConversion, ScaleClaim, ScaleProvenance, SidecarImport,
    georef_path, import_sidecar,
};

/// Two circles, the drawing box (3, 3) to (21, 7), as in the #1415 tests.
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
    let d = std::env::temp_dir().join(format!("libviprs-1554-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts the mock drawing into `dir/o.pvec`, with no calibration.
fn converted(dir: &Path) -> PathBuf {
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(b"x"), &dest)
        .unit(SourceUnit::Meters)
        .convert_with::<Mock>()
        .unwrap();
    dest
}

/// A sidecar as a schema 1 conversion wrote it.
fn old_sidecar(dest: &Path, id: Option<&str>) -> PathBuf {
    let id = id.map_or("null".to_owned(), |i| format!("\"{i}\""));
    let json = format!(
        r#"{{
  "schema_version": "1",
  "bounds": {{ "min": [3.0, 3.0], "max": [21.0, 7.0] }},
  "unit": {{ "source": "Unknown", "insunits": null, "physical": false,
            "evidence": "the provider reports no unit for this drawing", "applied": "Meters" }},
  "calibration_id": {id},
  "control_points": [[3.0, 3.0]]
}}"#
    );
    let path = georef_path(dest);
    fs::write(&path, json).unwrap();
    path
}

fn survey() -> Calibration {
    let scale = ScaleClaim {
        metres_per_source_unit: 0.001,
        provenance: ScaleProvenance::Manual,
        uncertainty: None,
        note: "from the survey record".to_owned(),
    };
    Calibration::new(Some(scale), None, None).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_conversion_writes_no_sidecar_and_the_report_names_none() {
    let dir = scratch("none");
    let out = PvecConversion::new(CadSource::Bytes(b"x"), &dir.join("o.pvec"))
        .unit(SourceUnit::Meters)
        .convert_with::<Mock>()
        .unwrap();
    assert!(!georef_path(&dir.join("o.pvec")).exists());
    assert_eq!(out.report.georeference, None);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn without_a_calibration_no_transform_or_crs_reaches_the_artifact() {
    let dir = scratch("scan");
    let bytes = fs::read(converted(&dir)).unwrap();
    for needle in [&b"EPSG"[..], b"GeoTransform", b"crs", b"PROJCS"] {
        assert!(
            !bytes.windows(needle.len()).any(|w| w == needle),
            "{:?} found in the artifact",
            String::from_utf8_lossy(needle)
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_old_sidecar_imports_through_the_resolver_with_its_id_kept() {
    let dir = scratch("import");
    let dest = converted(&dir);
    let sidecar = old_sidecar(&dest, Some("site-survey-1"));
    let mut asked = None;
    let done = import_sidecar(&dest, &sidecar, |id| {
        asked = Some(id.to_owned());
        Some(survey())
    })
    .unwrap();
    assert_eq!(done, SidecarImport::Imported);
    assert_eq!(asked.as_deref(), Some("site-survey-1"));
    let c = open_published(&dest)
        .unwrap()
        .calibration()
        .unwrap()
        .unwrap();
    assert_eq!(c.external_id, "site-survey-1");
    assert_eq!(c.scale.unwrap().metres_per_source_unit, 0.001);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_id_nothing_resolves_writes_nothing() {
    let dir = scratch("unresolved");
    let dest = converted(&dir);
    let before = fs::read(&dest).unwrap();
    let sidecar = old_sidecar(&dest, Some("gone"));
    let done = import_sidecar(&dest, &sidecar, |_| None).unwrap();
    assert_eq!(done, SidecarImport::Unresolved("gone".to_owned()));
    assert_eq!(fs::read(&dest).unwrap(), before);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_sidecar_with_no_id_has_nothing_to_import() {
    let dir = scratch("no-id");
    let dest = converted(&dir);
    let before = fs::read(&dest).unwrap();
    let sidecar = old_sidecar(&dest, None);
    let done = import_sidecar(&dest, &sidecar, |_| panic!("nothing to resolve")).unwrap();
    assert_eq!(done, SidecarImport::NoCalibrationId);
    assert_eq!(fs::read(&dest).unwrap(), before);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_sidecar_of_an_unknown_version_is_refused_and_writes_nothing() {
    let dir = scratch("future");
    let dest = converted(&dir);
    let before = fs::read(&dest).unwrap();
    let sidecar = old_sidecar(&dest, Some("site-survey-1"));
    let json = fs::read_to_string(&sidecar)
        .unwrap()
        .replace("\"schema_version\": \"1\"", "\"schema_version\": \"2\"");
    fs::write(&sidecar, json).unwrap();
    let err = import_sidecar(&dest, &sidecar, |_| Some(survey())).unwrap_err();
    assert!(matches!(err, GeoreferenceError::UnknownVersion(v) if v == "2"));
    assert_eq!(fs::read(&dest).unwrap(), before);
    fs::remove_dir_all(&dir).unwrap();
}
