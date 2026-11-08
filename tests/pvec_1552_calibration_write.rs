//! Issue #1552 (slice 4 of #1543): the calibration goes into the `.pvec`. A PDF writes what its page
//! says, a DWG writes what the caller hands it and nothing otherwise, the fixture's control points come
//! back out of the written artifact, and `recalibrate` adds, replaces and removes the block without
//! moving anything else. Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use std::fs;
use std::path::{Path, PathBuf};

use fixture::*;
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceUnit,
};
use libviprs::pvec_convert::{
    Calibration, ControlPoint, CrsKind, Georeference, GeoreferenceMethod, PdfConversion,
    PvecConversion, ScaleClaim, ScaleProvenance, recalibrate,
};
use lopdf::{Dictionary, Document, Object, dictionary};
use pvec_facade::unstable::{
    Calibration as Stored, DefinitionFormat as StoredFormat, FitMethod, Provenance,
};

const CONTENT: &str = "10 10 m 600 10 l 600 780 l 10 780 l h S \
                       0.5 g 100 100 200 150 re f \
                       q 200 0 0 100 300 400 cm /Im1 Do Q";

fn pdf_with(entries: Vec<(&str, Object)>) -> Vec<u8> {
    let bytes = one(rgb8(2, 2, &[9u8; 12]), CONTENT);
    let mut doc = Document::load_mem(&bytes).unwrap();
    let page = *doc.get_pages().get(&1).unwrap();
    let dict = doc.get_object_mut(page).unwrap().as_dict_mut().unwrap();
    for (k, v) in entries {
        dict.set(k, v);
    }
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn geopdf(gcs: Dictionary) -> Vec<u8> {
    let geo = dictionary! {
        "Type" => "Measure", "Subtype" => "GEO",
        "GPTS" => Object::Array(
            [51.0f32, -1.0, 51.1, -1.0, 51.1, -0.9, 51.0, -0.9].map(Object::Real).to_vec()
        ),
        "LPTS" => Object::Array([0, 0, 0, 1, 1, 1, 1, 0].map(Object::Integer).to_vec()),
        "GCS" => gcs,
    };
    let vp = Object::Dictionary(dictionary! {
        "Type" => "Viewport",
        "BBox" => Object::Array([0, 0, 612, 792].map(Object::Integer).to_vec()),
        "Measure" => geo,
    });
    pdf_with(vec![("VP", Object::Array(vec![vp]))])
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1552-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn stored(path: &Path) -> Option<Stored> {
    open_published(path).unwrap().calibration().unwrap()
}

fn convert_pdf(bytes: &[u8], out: &Path) {
    PdfConversion::new(CadSource::Bytes(bytes), out)
        .convert()
        .unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_pdf_with_no_viewport_stores_the_point_as_its_declared_unit() {
    let dir = scratch("points");
    let out = dir.join("out.pvec");
    convert_pdf(&pdf_with(Vec::new()), &out);
    let c = stored(&out).expect("a calibration block");
    let scale = c.scale.unwrap();
    assert_eq!(scale.provenance, Provenance::DeclaredUnit);
    assert_eq!(scale.metres_per_source_unit, 0.0254 / 72.0);
    assert!(c.georeference.is_none());
    fs::remove_dir_all(&dir).unwrap();
}

/// The #1547 fixture read back from the artifact: the four GeoPDF corners, in page space and
/// (longitude, latitude), and the fitted affine.
#[test]
#[cfg_attr(miri, ignore)]
fn the_geopdf_control_points_are_read_back_from_the_artifact() {
    let dir = scratch("geo");
    let out = dir.join("out.pvec");
    convert_pdf(
        &geopdf(dictionary! { "Type" => "GEOGCS", "EPSG" => 4326 }),
        &out,
    );
    let g = stored(&out).unwrap().georeference.unwrap();
    assert_eq!(g.crs_id, "EPSG:4326");
    assert_eq!(g.method, FitMethod::Fitted);
    let sources: Vec<[f64; 2]> = g.control_points.iter().map(|p| p.source).collect();
    assert_eq!(
        sources,
        vec![[0.0, 0.0], [0.0, 792.0], [612.0, 792.0], [612.0, 0.0]]
    );
    let world = g.control_points[2].world;
    assert!((world[0] + 0.9).abs() < 1e-5 && (world[1] - 51.1).abs() < 1e-5);
    let [a, b, c, d, tx, ty] = g.source_to_crs;
    let (x, y) = (a * 612.0 + b * 792.0 + tx, c * 612.0 + d * 792.0 + ty);
    assert!((x - world[0]).abs() < 1e-9 && (y - world[1]).abs() < 1e-9);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_geopdf_wkt_is_stored_as_wkt1() {
    let dir = scratch("wkt");
    let out = dir.join("out.pvec");
    let wkt = "GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]]]";
    convert_pdf(
        &geopdf(dictionary! { "Type" => "GEOGCS", "WKT" => literal(wkt.as_bytes()) }),
        &out,
    );
    let def = stored(&out)
        .unwrap()
        .georeference
        .unwrap()
        .definition
        .unwrap();
    assert_eq!(def.format, StoredFormat::Wkt1);
    assert_eq!(def.text, wkt);
    fs::remove_dir_all(&dir).unwrap();
}

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

fn site() -> Calibration {
    Calibration::new(
        Some(ScaleClaim::measured([3.0, 3.0], [21.0, 3.0], 36.0).unwrap()),
        Some(Georeference {
            affine: [2.0, 0.0, 0.0, 2.0, 530_000.0, 180_000.0],
            crs_kind: CrsKind::Projected,
            method: GeoreferenceMethod::Manual,
            crs_id: Some("EPSG:27700".to_owned()),
            definition: None,
            control_points: vec![ControlPoint {
                source: [3.0, 3.0],
                world: [530_006.0, 180_006.0],
            }],
            residual_max: 0.0,
        }),
        Some("site-survey-1".to_owned()),
    )
    .unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_dwg_stores_the_calibration_its_caller_supplies() {
    let dir = scratch("dwg");
    let out = dir.join("out.pvec");
    PvecConversion::new(CadSource::Bytes(b"x"), &out)
        .unit(SourceUnit::Meters)
        .calibration(site())
        .convert_with::<Mock>()
        .unwrap();
    let c = stored(&out).unwrap();
    assert_eq!(c.external_id, "site-survey-1");
    let scale = c.scale.unwrap();
    assert_eq!(scale.metres_per_source_unit, 2.0);
    assert_eq!(
        scale.provenance,
        Provenance::Measured {
            p0: [3.0, 3.0],
            p1: [21.0, 3.0],
            distance_metres: 36.0
        }
    );
    let g = c.georeference.unwrap();
    assert_eq!(g.crs_id, "EPSG:27700");
    assert_eq!(g.source_to_crs, [2.0, 0.0, 0.0, 2.0, 530_000.0, 180_000.0]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_dwg_with_no_calibration_stores_none() {
    let dir = scratch("dwg-none");
    let out = dir.join("out.pvec");
    PvecConversion::new(CadSource::Bytes(b"x"), &out)
        .unit(SourceUnit::Meters)
        .convert_with::<Mock>()
        .unwrap();
    assert!(stored(&out).is_none());
    fs::remove_dir_all(&dir).unwrap();
}

/// Removing the block and putting the same one back gives the bytes the conversion wrote: the
/// geometry never moved, and a recalibrated file has one byte form.
#[test]
#[cfg_attr(miri, ignore)]
fn recalibrate_removes_adds_and_replaces_without_moving_the_geometry() {
    let dir = scratch("recal");
    let out = dir.join("out.pvec");
    PvecConversion::new(CadSource::Bytes(b"x"), &out)
        .unit(SourceUnit::Meters)
        .calibration(site())
        .convert_with::<Mock>()
        .unwrap();
    let written = fs::read(&out).unwrap();
    recalibrate(&out, None).unwrap();
    assert!(stored(&out).is_none());
    recalibrate(&out, Some(&site())).unwrap();
    assert_eq!(fs::read(&out).unwrap(), written);
    let manual = Calibration::new(
        Some(ScaleClaim {
            metres_per_source_unit: 0.5,
            provenance: ScaleProvenance::Manual,
            uncertainty: Some(0.01),
            note: "taped on site".to_owned(),
        }),
        None,
        None,
    )
    .unwrap();
    recalibrate(&out, Some(&manual)).unwrap();
    let c = stored(&out).unwrap();
    assert_eq!(c.scale.unwrap().provenance, Provenance::Manual);
    assert!(c.georeference.is_none());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn recalibrating_a_missing_file_is_an_error_and_writes_nothing() {
    let dir = scratch("missing");
    let out = dir.join("absent.pvec");
    assert!(recalibrate(&out, Some(&site())).is_err());
    assert!(!out.exists());
    fs::remove_dir_all(&dir).unwrap();
}
