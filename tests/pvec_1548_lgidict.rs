//! Issue #1548 (slice 3 of #1543): a page georeferenced the older way, with an `/LGIDict`. Its
//! `/CTM` maps the page into the projection, and when the projection is UTM on WGS 84 it maps to an
//! EPSG code without a table; any other projection is reported and not mapped. When a page has a
//! GeoPDF viewport as well, the viewport wins. Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::cad::CadSource;
use libviprs::pvec_convert::{CrsKind, GeoreferenceMethod, PdfConversion, PdfConversionOutput};
use lopdf::{Dictionary, Document, Object, dictionary};

/// The same page as the viewport fixture of #1547: an outline, a grey fill and a placed raster.
const CONTENT: &str = "10 10 m 600 10 l 600 780 l 10 780 l h S \
                       0.5 g 100 100 200 150 re f \
                       q 200 0 0 100 300 400 cm /Im1 Do Q";

fn fixture(entries: Vec<(&str, Object)>) -> Vec<u8> {
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

/// LGIDict writes its numbers as strings, so the fixture does too.
fn strings(v: &[&str]) -> Object {
    Object::Array(v.iter().map(|s| literal(s.as_bytes())).collect())
}

/// Two metres a point, the page's lower-left at (500000, 5700000).
const CTM: [&str; 6] = ["2", "0", "0", "2", "500000", "5700000"];

fn lgidict(projection: Dictionary) -> Object {
    Object::Dictionary(dictionary! {
        "Type" => "LGIDict",
        "Version" => literal(b"2.1"),
        "CTM" => strings(&CTM),
        "Projection" => projection,
    })
}

fn utm(zone: i64, hemisphere: &str) -> Dictionary {
    dictionary! {
        "Type" => "Projection",
        "ProjectionType" => literal(b"UT"),
        "Datum" => literal(b"WGE"),
        "Zone" => zone,
        "Hemisphere" => literal(hemisphere.as_bytes()),
    }
}

fn convert(bytes: &[u8], tag: &str) -> PdfConversionOutput {
    let dir = std::env::temp_dir().join(format!("libviprs-1548-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out.pvec");
    let done = PdfConversion::new(CadSource::Bytes(bytes), &out)
        .convert()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    done
}

fn apply(affine: [f64; 6], p: [f64; 2]) -> [f64; 2] {
    let [a, b, c, d, tx, ty] = affine;
    [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty]
}

#[test]
fn the_lgidict_is_read_with_its_ctm_in_page_space() {
    let p = page(&fixture(vec![("LGIDict", lgidict(utm(30, "N")))]));
    let [l] = p.lgidicts.as_slice() else {
        panic!("one LGIDict expected, got {:?}", p.lgidicts);
    };
    assert_eq!(l.affine, [2.0, 0.0, 0.0, 2.0, 500_000.0, 5_700_000.0]);
    assert_eq!(l.projection_type, "UT");
    assert_eq!(l.datum.as_deref(), Some("WGE"));
    assert_eq!(l.zone, Some(30));
    assert_eq!(l.hemisphere.as_deref(), Some("N"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn utm_north_on_wgs84_is_epsg_326_and_the_zone() {
    let done = convert(
        &fixture(vec![("LGIDict", lgidict(utm(30, "N")))]),
        "north",
    );
    let g = done.calibration.georeference().expect("a georeference");
    assert_eq!(g.crs_id.as_deref(), Some("EPSG:32630"));
    assert_eq!(g.crs_kind, CrsKind::Projected);
    assert_eq!(g.method, GeoreferenceMethod::Declared);
    assert_eq!(g.residual_max, 0.0);
    // Pinned: the page's corners in the projection.
    assert_eq!(apply(g.affine, [0.0, 0.0]), [500_000.0, 5_700_000.0]);
    assert_eq!(apply(g.affine, [612.0, 792.0]), [501_224.0, 5_701_584.0]);
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert_eq!(summary.lgidicts, 1);
    assert_eq!(summary.georeference.as_deref(), Some("EPSG:32630"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn utm_south_is_epsg_327_and_an_array_of_dictionaries_is_read() {
    let done = convert(
        &fixture(vec![(
            "LGIDict",
            Object::Array(vec![lgidict(utm(33, "S"))]),
        )]),
        "south",
    );
    assert_eq!(
        done.calibration.georeference().unwrap().crs_id.as_deref(),
        Some("EPSG:32733")
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn another_projection_is_reported_and_not_mapped() {
    let lambert = dictionary! {
        "Type" => "Projection", "ProjectionType" => literal(b"LE"), "Datum" => literal(b"WGE"),
    };
    let done = convert(&fixture(vec![("LGIDict", lgidict(lambert))]), "lambert");
    assert!(done.calibration.georeference().is_none());
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert!(summary.notes.contains(&"lgidict-not-mapped".to_owned()));
}

#[test]
#[cfg_attr(miri, ignore)]
fn utm_on_another_datum_is_not_mapped() {
    let mut nad27 = utm(17, "N");
    nad27.set("Datum", literal(b"NAS-C"));
    let done = convert(&fixture(vec![("LGIDict", lgidict(nad27))]), "nad27");
    assert!(done.calibration.georeference().is_none());
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_ctm_follows_the_page_rotation() {
    let done = convert(
        &fixture(vec![
            ("Rotate", Object::Integer(90)),
            ("LGIDict", lgidict(utm(30, "N"))),
        ]),
        "rotate",
    );
    let g = done.calibration.georeference().unwrap();
    // The unrotated corner (612, 792) is displayed at (792, 0), and still lands where the CTM puts it.
    assert_eq!(apply(g.affine, [792.0, 0.0]), [501_224.0, 5_701_584.0]);
    assert_eq!(apply(g.affine, [0.0, 612.0]), [500_000.0, 5_700_000.0]);
}

/// The fixture both ways: with a GeoPDF viewport too, the ISO viewport is the one used.
#[test]
#[cfg_attr(miri, ignore)]
fn a_geopdf_viewport_wins_over_the_lgidict() {
    let geo = dictionary! {
        "Type" => "Measure", "Subtype" => "GEO",
        "GPTS" => Object::Array(
            [51.0f32, -1.0, 51.1, -1.0, 51.1, -0.9, 51.0, -0.9].map(Object::Real).to_vec()
        ),
        "LPTS" => Object::Array([0, 0, 0, 1, 1, 1, 1, 0].map(Object::Integer).to_vec()),
        "GCS" => dictionary! { "Type" => "GEOGCS", "EPSG" => 4326 },
    };
    let vp = Object::Dictionary(dictionary! {
        "Type" => "Viewport",
        "BBox" => Object::Array([0, 0, 612, 792].map(Object::Integer).to_vec()),
        "Measure" => geo,
    });
    let done = convert(
        &fixture(vec![
            ("VP", Object::Array(vec![vp])),
            ("LGIDict", lgidict(utm(30, "N"))),
        ]),
        "both",
    );
    assert_eq!(
        done.calibration.georeference().unwrap().crs_id.as_deref(),
        Some("EPSG:4326")
    );
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert!(summary.notes.contains(&"lgidict-ignored".to_owned()));
}
