//! Issue #1547 (slice 2 of #1543): a PDF says its own scale and place. The page's `/VP` viewports
//! are read, the one that covers the page is chosen, a `/Measure /RL` gives a plan scale, a
//! `/Measure /GEO` gives control points in the same post-`/Rotate` page space as the geometry and a
//! fitted affine, and every PDF without a plan scale still says its unit is the point.
//! Needs `--features pvec-convert,pvec-raster`.

#![cfg(all(feature = "pvec-convert", feature = "pvec-raster"))]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::cad::CadSource;
use libviprs::pvec::pdf_vector::{PdfMeasure, PdfViewport};
use libviprs::pvec_convert::{
    Calibration, CrsKind, DefinitionFormat, GeoreferenceMethod, PdfConversion, PdfConversionOutput,
    ScaleProvenance,
};
use lopdf::{Dictionary, Document, Object, dictionary};

const POINT: f64 = 0.0254 / 72.0;

/// The fixture page: a vector layer (a stroked outline and a filled block) and a placed raster.
const CONTENT: &str = "10 10 m 600 10 l 600 780 l 10 780 l h S \
                       0.5 g 100 100 200 150 re f \
                       q 200 0 0 100 300 400 cm /Im1 Do Q";

/// The fixture PDF with `entries` set on its page dictionary.
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

/// PDF reals as `lopdf` writes them, `f32`, which is why the world points are compared to 1e-5
/// (51.1 as an `f32` is 51.0999985).
fn nums(v: &[f64]) -> Object {
    Object::Array(v.iter().map(|&x| Object::Real(x as f32)).collect())
}

fn viewport(bbox: [f64; 4], measure: Dictionary) -> Object {
    Object::Dictionary(dictionary! {
        "Type" => "Viewport",
        "BBox" => nums(&bbox),
        "Name" => literal(b"Plan"),
        "Measure" => measure,
    })
}

/// 1 in = 10 ft: one point of paper is 10 / 72 ft.
fn rl() -> Dictionary {
    dictionary! {
        "Type" => "Measure", "Subtype" => "RL",
        "R" => literal(b"1 in = 10 ft"),
        "X" => vec![Object::Dictionary(dictionary! {
            "Type" => "NumberFormat", "U" => literal(b"ft"), "C" => Object::Real(10.0 / 72.0),
        })],
    }
}

/// A GeoPDF measure over the whole viewport: the unit square's corners and their (lat, lon).
const GPTS: [f64; 8] = [51.0, -1.0, 51.1, -1.0, 51.1, -0.9, 51.0, -0.9];

fn geo(gcs: Dictionary) -> Dictionary {
    dictionary! {
        "Type" => "Measure", "Subtype" => "GEO",
        "GPTS" => nums(&GPTS),
        "LPTS" => nums(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        "GCS" => gcs,
    }
}

fn wgs84() -> Dictionary {
    dictionary! { "Type" => "GEOGCS", "EPSG" => 4326 }
}

const WGS84_WKT: &str = "GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]]";

const PAGE: [f64; 4] = [0.0, 0.0, 612.0, 792.0];

fn convert(bytes: &[u8], tag: &str) -> PdfConversionOutput {
    let dir = std::env::temp_dir().join(format!("libviprs-1547-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out.pvec");
    let done = PdfConversion::new(CadSource::Bytes(bytes), &out)
        .convert()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    done
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn a_page_without_viewports_has_none_and_reads_no_measure() {
    assert!(page(&fixture(Vec::new())).viewports.is_empty());
}

#[test]
fn a_viewport_is_read_in_page_space_with_its_measure() {
    let bytes = fixture(vec![(
        "VP",
        Object::Array(vec![viewport([0.0, 0.0, 612.0, 792.0], geo(wgs84()))]),
    )]);
    let p = page(&bytes);
    let [
        PdfViewport {
            bbox,
            name,
            measure: Some(PdfMeasure::Geo {
                local, geographic, ..
            }),
        },
    ] = p.viewports.as_slice()
    else {
        panic!("one GEO viewport expected, got {:?}", p.viewports);
    };
    assert_eq!(*bbox, PAGE);
    assert_eq!(name.as_deref(), Some("Plan"));
    assert_eq!(
        local,
        &vec![[0.0, 0.0], [0.0, 792.0], [612.0, 792.0], [612.0, 0.0]]
    );
    assert_eq!(geographic.len(), 4);
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_pdf_says_its_unit_is_the_point_when_nothing_better_is_stated() {
    let done = convert(&fixture(Vec::new()), "points");
    let scale = done.calibration.scale().expect("a declared-unit scale");
    assert_eq!(scale.provenance, ScaleProvenance::DeclaredUnit);
    assert_eq!(scale.metres_per_source_unit, POINT);
    assert!(done.calibration.georeference().is_none());
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert_eq!(summary.viewport, None);
    assert_eq!(summary.scale, "declared-unit");
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_plan_scale_replaces_the_point_and_keeps_its_ratio() {
    let bytes = fixture(vec![("VP", Object::Array(vec![viewport(PAGE, rl())]))]);
    let done = convert(&bytes, "rl");
    let scale = done.calibration.scale().unwrap();
    assert!(near(
        scale.metres_per_source_unit,
        10.0 / 72.0 * 0.3048,
        1e-8
    ));
    let ScaleProvenance::PlanScale {
        numerator,
        denominator,
        paper_metres_per_unit,
    } = scale.provenance
    else {
        panic!("a plan scale expected, got {:?}", scale.provenance);
    };
    assert_eq!((numerator, denominator), (1, 120));
    assert_eq!(paper_metres_per_unit, POINT);
    assert_eq!(scale.note, "1 in = 10 ft");
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert_eq!(summary.viewport, Some(0));
    assert_eq!(summary.scale, "plan-scale");
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_plan_scale_in_a_unit_nobody_knows_is_reported_and_the_point_stays() {
    let mut m = rl();
    m.set(
        "X",
        vec![Object::Dictionary(dictionary! {
            "Type" => "NumberFormat", "U" => literal(b"cubits"), "C" => 1.0,
        })],
    );
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, m)]))]),
        "cubits",
    );
    assert_eq!(
        done.calibration.scale().unwrap().provenance,
        ScaleProvenance::DeclaredUnit
    );
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert!(summary.notes.contains(&"rl-unit-unknown".to_owned()));
}

/// The fixture both ways round: the GeoPDF points come back in page space, in (lon, lat) order, and
/// the affine puts each source point on its world point.
#[test]
#[cfg_attr(miri, ignore)]
fn a_geopdf_viewport_gives_fitted_control_points_and_its_epsg_code() {
    let bytes = fixture(vec![(
        "VP",
        Object::Array(vec![viewport(PAGE, geo(wgs84()))]),
    )]);
    let done = convert(&bytes, "geo");
    assert_eq!(done.rasters, 1);
    assert!(done.written >= 2);
    let g = done.calibration.georeference().expect("a georeference");
    assert_eq!(g.crs_kind, CrsKind::Geographic);
    assert_eq!(g.method, GeoreferenceMethod::Fitted);
    assert_eq!(g.crs_id.as_deref(), Some("EPSG:4326"));
    let pinned = [
        ([0.0, 0.0], [-1.0, 51.0]),
        ([0.0, 792.0], [-1.0, 51.1]),
        ([612.0, 792.0], [-0.9, 51.1]),
        ([612.0, 0.0], [-0.9, 51.0]),
    ];
    assert_eq!(g.control_points.len(), 4);
    let [a, b, c, d, tx, ty] = g.affine;
    for (p, (source, world)) in g.control_points.iter().zip(pinned) {
        assert_eq!(p.source, source);
        assert!(near(p.world[0], world[0], 1e-5) && near(p.world[1], world[1], 1e-5));
        let x = a * source[0] + b * source[1] + tx;
        let y = c * source[0] + d * source[1] + ty;
        assert!(near(x, p.world[0], 1e-9) && near(y, p.world[1], 1e-9));
    }
    assert!(g.residual_max < 1e-9);
    // GEO has no plan scale, so the unit stays the point.
    assert_eq!(
        done.calibration.scale().unwrap().provenance,
        ScaleProvenance::DeclaredUnit
    );
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert_eq!(summary.georeference.as_deref(), Some("EPSG:4326"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_gcs_given_only_as_wkt_is_kept_as_its_definition() {
    let gcs = dictionary! { "Type" => "GEOGCS", "WKT" => literal(WGS84_WKT.as_bytes()) };
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, geo(gcs))]))]),
        "wkt",
    );
    let g = done.calibration.georeference().unwrap();
    assert_eq!(g.crs_id, None);
    let def = g.definition.as_ref().unwrap();
    assert_eq!(def.format, DefinitionFormat::Wkt);
    assert_eq!(def.text, WGS84_WKT);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_projected_gcs_is_named_by_its_geographic_base() {
    let wkt = format!(
        "PROJCS[\"British_National_Grid\",{WGS84_WKT},PROJECTION[\"Transverse_Mercator\"],UNIT[\"Meter\",1.0]]"
    );
    let gcs = dictionary! {
        "Type" => "PROJCS", "EPSG" => 27700, "WKT" => literal(wkt.as_bytes()),
    };
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, geo(gcs))]))]),
        "projcs",
    );
    let g = done.calibration.georeference().unwrap();
    assert_eq!(g.crs_kind, CrsKind::Geographic);
    assert_eq!(g.crs_id, None);
    assert_eq!(g.definition.as_ref().unwrap().text, WGS84_WKT);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_projected_gcs_with_no_wkt_is_reported_and_not_mapped() {
    let gcs = dictionary! { "Type" => "PROJCS", "EPSG" => 27700 };
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, geo(gcs))]))]),
        "projcs-bare",
    );
    assert!(done.calibration.georeference().is_none());
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert!(
        summary
            .notes
            .contains(&"geo-projected-without-definition".to_owned())
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn control_points_follow_the_page_rotation() {
    let bytes = fixture(vec![
        ("Rotate", Object::Integer(90)),
        ("VP", Object::Array(vec![viewport(PAGE, geo(wgs84()))])),
    ]);
    let done = convert(&bytes, "rotate");
    let g = done.calibration.georeference().unwrap();
    // Displayed 792 wide and 612 high: (x, y) in the unrotated page goes to (y, 612 - x).
    let sources: Vec<[f64; 2]> = g.control_points.iter().map(|p| p.source).collect();
    assert_eq!(
        sources,
        vec![[0.0, 612.0], [792.0, 612.0], [792.0, 0.0], [0.0, 0.0]]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_viewport_that_covers_the_page_wins_over_a_smaller_one() {
    let bytes = fixture(vec![(
        "VP",
        Object::Array(vec![
            viewport([400.0, 20.0, 600.0, 120.0], rl()),
            viewport(PAGE, geo(wgs84())),
        ]),
    )]);
    let done = convert(&bytes, "choice");
    assert!(done.calibration.georeference().is_some());
    assert_eq!(
        done.report.pdf.unwrap().calibration.unwrap().viewport,
        Some(1)
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn with_no_covering_viewport_the_largest_is_used() {
    let bytes = fixture(vec![(
        "VP",
        Object::Array(vec![
            viewport([0.0, 0.0, 300.0, 300.0], rl()),
            viewport([400.0, 20.0, 600.0, 120.0], geo(wgs84())),
        ]),
    )]);
    let done = convert(&bytes, "largest");
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert_eq!(summary.viewport, Some(0));
    assert_eq!(summary.scale, "plan-scale");
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_control_points_give_a_similarity_with_no_residual() {
    let mut m = geo(wgs84());
    m.set("GPTS", nums(&[51.0, -1.0, 51.1, -0.9]));
    m.set("LPTS", nums(&[0.0, 0.0, 1.0, 1.0]));
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, m)]))]),
        "two",
    );
    let g = done.calibration.georeference().unwrap();
    let [a, b, c, d, ..] = g.affine;
    assert!(near(a, d, 1e-15) && near(b, -c, 1e-15));
    assert_eq!(g.residual_max, 0.0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn points_an_affine_cannot_meet_leave_a_residual() {
    let mut m = geo(wgs84());
    m.set(
        "GPTS",
        nums(&[51.0, -1.0, 51.1, -1.0, 51.12, -0.88, 51.0, -0.9]),
    );
    let done = convert(
        &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, m)]))]),
        "residual",
    );
    let g = done.calibration.georeference().unwrap();
    assert!(g.residual_max > 1e-3, "{}", g.residual_max);
    assert!(Calibration::new(None, Some(g.clone()), None).is_ok());
}

/// A hostile `/C` below zero is refused as a plan scale, and the conversion finishes.
#[test]
#[cfg_attr(miri, ignore)]
fn a_negative_plan_scale_factor_is_refused_and_the_conversion_finishes() {
    let mut m = rl();
    m.set(
        "X",
        vec![Object::Dictionary(dictionary! {
            "Type" => "NumberFormat", "U" => literal(b"ft"), "C" => Object::Real(-10.0 / 72.0),
        })],
    );
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let done = convert(
            &fixture(vec![("VP", Object::Array(vec![viewport(PAGE, m)]))]),
            "negative",
        );
        tx.send(done).unwrap();
    });
    let done = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the conversion should finish");
    assert_eq!(
        done.calibration.scale().unwrap().provenance,
        ScaleProvenance::DeclaredUnit
    );
    let summary = done.report.pdf.unwrap().calibration.unwrap();
    assert!(summary.notes.contains(&"rl-refused".to_owned()));
}
