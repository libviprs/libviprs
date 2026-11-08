//! Layouts slice 2 (#1560): the acadrust provider hands over what placing model space on a sheet
//! needs. A layout's plot settings, each viewport's view (centre, height, target, direction,
//! twist, perspective), its plot flags and its own layer, a layer's plottable flag, and a clip
//! boundary that says what kind it is instead of turning into the viewport's rectangle.
//!
//! The three viewport probes of the gold corpus are read from the vendored files (compiled in, so
//! nothing touches the disk at run time); the rest is written with acadrust and read back through
//! a real DWG.
//!
//! Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use std::io::Cursor;

use acadrust::entities::{Circle, LwPolyline, Viewport};
use acadrust::objects::ObjectType;
use acadrust::types::Vector2;
use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{
    Availability, DwgProvider, Facet, FacetSource, LayoutId, LayoutKind, ProviderError, SourceFact,
    ViewportClip, ViewportEntity,
};

const MULTI: &[u8] =
    include_bytes!("fixtures/pvec-gold/fixtures/paper-space-multi-viewport-001/source.dwg");
const MULTI_MATH: &str =
    include_str!("fixtures/pvec-gold/fixtures/paper-space-multi-viewport-001/expected.math.json");
const CLIPPED: &[u8] =
    include_bytes!("fixtures/pvec-gold/fixtures/viewport-clipped-001/source.dwg");
const CLIPPED_MATH: &str =
    include_str!("fixtures/pvec-gold/fixtures/viewport-clipped-001/expected.math.json");
const BOUNDED: &[u8] =
    include_bytes!("fixtures/pvec-gold/fixtures/viewport-clipped-bounded-001/source.dwg");
const BOUNDED_MATH: &str =
    include_str!("fixtures/pvec-gold/fixtures/viewport-clipped-bounded-001/expected.math.json");

fn open(bytes: &[u8]) -> AcadrustProvider {
    AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens")
}

/// An exact number of the expected file, from its hex bits.
fn exact(v: &serde_json::Value) -> f64 {
    let hex = v["f64_hex"].as_str().expect("f64_hex");
    f64::from_bits(u64::from_str_radix(hex.trim_start_matches("0x"), 16).expect("hex"))
}

fn pair(v: &serde_json::Value) -> [f64; 2] {
    [exact(&v[0]), exact(&v[1])]
}

/// The viewports the probe recorded, in the order it wrote them.
fn expected_viewports(math: &str) -> Vec<serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_str(math).expect("json");
    doc["viewports"].as_array().expect("viewports").clone()
}

fn paper_viewports(p: &AcadrustProvider) -> Vec<ViewportEntity> {
    p.viewports().expect("viewports")
}

#[test]
fn each_viewport_of_the_multi_viewport_probe_carries_its_paper_rectangle_and_its_view() {
    let p = open(MULTI);
    let got = paper_viewports(&p);
    let want = expected_viewports(MULTI_MATH);
    // The probe has no paper-space view of its own, so every viewport it wrote is a window onto
    // the model, the one the reader numbers 1 included.
    assert_eq!(got.len(), want.len(), "every viewport the probe wrote");
    let mut ids: Vec<i16> = got.iter().map(|v| v.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (1..=20).collect::<Vec<i16>>());
    for w in &want {
        let centre = pair(&w["paper"]["center"]);
        let v = got
            .iter()
            .find(|v| v.center == centre)
            .unwrap_or_else(|| panic!("no viewport centred on {centre:?}: {got:#?}"));
        assert_eq!(v.width, exact(&w["paper"]["width"]), "{}", w["id"]);
        assert_eq!(v.height, exact(&w["paper"]["height"]), "{}", w["id"]);
        assert_eq!(v.view_center, pair(&w["view"]["center"]), "{}", w["id"]);
        assert_eq!(v.view_height, exact(&w["view"]["height"]), "{}", w["id"]);
        // The probe draws plan views, untwisted, not in perspective, all of them printed.
        assert_eq!(v.view_direction, [0.0, 0.0, 1.0], "{}", w["id"]);
        assert_eq!(v.twist, 0.0, "{}", w["id"]);
        assert!(!v.perspective && !v.hide_plot, "{}", w["id"]);
        assert!(!v.layer.is_empty(), "{}: a viewport is on a layer", w["id"]);
        assert!(
            matches!(v.clip, ViewportClip::Rectangle(_)),
            "{}: no boundary entity, so its own rectangle: {:?}",
            w["id"],
            v.clip
        );
    }
}

fn clipped_polygon_is_kept_vertex_for_vertex(bytes: &[u8], math: &str) {
    let p = open(bytes);
    let got = paper_viewports(&p);
    let want = expected_viewports(math);
    assert_eq!(got.len(), 1);
    let (v, w) = (&got[0], &want[0]);
    assert_eq!(v.center, pair(&w["paper"]["center"]));
    assert_eq!(v.width, exact(&w["paper"]["width"]));
    assert_eq!(v.height, exact(&w["paper"]["height"]));
    assert_eq!(v.view_center, pair(&w["view"]["center"]));
    assert_eq!(v.view_height, exact(&w["view"]["height"]));
    let ViewportClip::Polygon(vertices) = &v.clip else {
        panic!("a polyline boundary is a polygon: {:?}", v.clip);
    };
    let polygon: Vec<[f64; 2]> = w["clip"]["polygon"]
        .as_array()
        .expect("polygon")
        .iter()
        .map(pair)
        .collect();
    let read: Vec<[f64; 2]> = vertices.iter().map(|q| [q.x, q.y]).collect();
    assert_eq!(read, polygon);
    assert!(vertices.iter().all(|q| q.bulge == 0.0));
}

#[test]
fn the_clipped_probe_keeps_its_polygon_and_its_rectangle_apart() {
    clipped_polygon_is_kept_vertex_for_vertex(CLIPPED, CLIPPED_MATH);
}

#[test]
fn the_bounded_probe_reads_the_same_polygon_which_is_inside_its_bounds() {
    clipped_polygon_is_kept_vertex_for_vertex(BOUNDED, BOUNDED_MATH);
}

#[test]
fn the_probes_layouts_have_plot_settings_and_a_layout_id_nobody_listed_is_no_scope() {
    for bytes in [MULTI, CLIPPED, BOUNDED] {
        let p = open(bytes);
        for l in p.layouts().unwrap() {
            assert!(
                matches!(p.layout_plot(l.id), Ok(SourceFact::Present(_))),
                "{}: every layout since R2000 embeds plot settings",
                l.name
            );
        }
        let missing = LayoutId(u32::try_from(p.layouts().unwrap().len()).unwrap());
        assert!(matches!(
            p.layout_plot(missing),
            Err(ProviderError::NoSuchScope(_))
        ));
    }
}

#[test]
fn the_caps_claim_layout_plot_from_the_decoder_directly() {
    let p = open(CLIPPED);
    let row = p.caps().get(Facet::LayoutPlot).expect("a row");
    assert_eq!(row.availability, Availability::Reported);
    assert_eq!(row.source, FacetSource::DecoderDirect);
}

/// A drawing with one sheet whose plot settings are all off their defaults, a no-plot layer, a
/// viewport with every view field set and two clipped viewports, one to a circle.
struct Sheet {
    bytes: Vec<u8>,
    plain: u64,
    circled: u64,
    bulged: u64,
    circle: u64,
}

fn sheet() -> Sheet {
    let mut doc = CadDocument::new();
    let mut defpoints = acadrust::tables::Layer::new("Defpoints");
    defpoints.is_plottable = false;
    defpoints.handle = doc.allocate_handle();
    doc.layers.add(defpoints).expect("layer");
    let mut vports = acadrust::tables::Layer::new("VPORTS");
    vports.handle = doc.allocate_handle();
    doc.layers.add(vports).expect("layer");

    let layout = doc.add_layout("Sheet A").expect("layout");
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    l.paper_width = 420.0;
    l.paper_height = 297.0;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_window_min_x = 1.0;
    l.plot_window_min_y = 2.0;
    l.plot_window_max_x = 3.0;
    l.plot_window_max_y = 4.0;
    l.plot_scale_numerator = 1.0;
    l.plot_scale_denominator = 2.0;
    l.plot_rotation = 1;
    l.plot_margin_left = 5.0;
    l.plot_margin_bottom = 6.0;
    l.plot_margin_right = 7.0;
    l.plot_margin_top = 8.0;
    l.plot_origin_x = 9.0;
    l.plot_origin_y = 10.0;
    l.plot_style_sheet = "monochrome.ctb".into();
    l.plot_flags.print_lineweights = true;
    l.plot_flags.scale_lineweights = true;
    l.plot_flags.plot_plot_styles = true;
    l.plot_flags.plot_viewport_borders = false;
    l.plot_flags.draw_viewports_first = true;

    let mut vp = Viewport::with_size(Vector3::new(100.0, 80.0, 0.0), 120.0, 90.0);
    vp.id = 2;
    vp.common.layer = "VPORTS".into();
    vp.view_center = Vector3::new(12.5, -3.25, 0.0);
    vp.view_height = 45.0;
    vp.view_target = Vector3::new(1.0, 2.0, 0.0);
    vp.view_direction = Vector3::new(0.0, 0.0, 1.0);
    vp.twist_angle = 0.5;
    vp.status.hide_plot = true;
    vp.status.perspective = true;
    let plain = doc
        .add_entity_to_layout(EntityType::Viewport(Box::new(vp)), "Sheet A")
        .expect("viewport")
        .value();

    let ring = Circle::from_center_radius(Vector3::new(300.0, 80.0, 0.0), 30.0);
    let circle = doc
        .add_entity_to_layout(EntityType::Circle(ring), "Sheet A")
        .expect("circle");
    let mut vp = Viewport::with_size(Vector3::new(300.0, 80.0, 0.0), 60.0, 60.0);
    vp.id = 3;
    vp.status.non_rectangular_clipping = true;
    vp.clip_boundary_handle = circle;
    let circled = doc
        .add_entity_to_layout(EntityType::Viewport(Box::new(vp)), "Sheet A")
        .expect("viewport")
        .value();

    let mut pl = LwPolyline::from_points(vec![
        Vector2::new(0.0, 200.0),
        Vector2::new(40.0, 200.0),
        Vector2::new(40.0, 240.0),
    ]);
    pl.vertices[1].bulge = 0.5;
    pl.is_closed = true;
    let boundary = doc
        .add_entity_to_layout(EntityType::LwPolyline(pl), "Sheet A")
        .expect("polyline");
    let mut vp = Viewport::with_size(Vector3::new(20.0, 220.0, 0.0), 40.0, 40.0);
    vp.id = 4;
    vp.status.non_rectangular_clipping = true;
    vp.clip_boundary_handle = boundary;
    let bulged = doc
        .add_entity_to_layout(EntityType::Viewport(Box::new(vp)), "Sheet A")
        .expect("viewport")
        .value();

    let mut bytes = Vec::new();
    DwgWriter::write_to_writer(Cursor::new(&mut bytes), &doc).expect("write");
    Sheet {
        bytes,
        plain,
        circled,
        bulged,
        circle: circle.value(),
    }
}

fn viewport(p: &AcadrustProvider, handle: u64) -> ViewportEntity {
    paper_viewports(p)
        .into_iter()
        .find(|v| v.handle.get() == handle)
        .unwrap_or_else(|| panic!("no viewport {handle:X}"))
}

#[test]
fn a_layouts_plot_settings_cross_the_boundary_as_stored() {
    let s = sheet();
    let p = open(&s.bytes);
    let sheet = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.name == "Sheet A")
        .expect("Sheet A");
    assert_eq!(sheet.kind, LayoutKind::Paper);
    let SourceFact::Present(plot) = p.layout_plot(sheet.id).expect("reported") else {
        panic!("Sheet A has plot settings");
    };
    assert_eq!(plot.paper_size, "ISO_A3_(420.00_x_297.00_MM)");
    assert_eq!((plot.paper_width, plot.paper_height), (420.0, 297.0));
    assert_eq!(plot.paper_units, 1);
    assert_eq!(plot.plot_type, 5);
    assert_eq!(plot.window, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!((plot.scale_numerator, plot.scale_denominator), (1.0, 2.0));
    assert_eq!(plot.rotation, 1);
    assert_eq!(plot.margins, [5.0, 6.0, 7.0, 8.0]);
    assert_eq!(plot.origin, [9.0, 10.0]);
    assert_eq!(plot.style_sheet, "monochrome.ctb");
    assert!(plot.print_lineweights && plot.scale_lineweights && plot.plot_plot_styles);
    assert!(!plot.plot_viewport_borders);
    assert!(plot.draw_viewports_first);
}

#[test]
fn a_no_plot_layer_says_so_and_the_others_print() {
    let p = open(&sheet().bytes);
    let layers = p.layers().unwrap();
    let get = |name: &str| {
        layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("no layer {name}"))
    };
    assert!(!get("Defpoints").plottable);
    assert!(get("VPORTS").plottable);
    assert!(get("0").plottable);
}

#[test]
fn a_viewports_view_plot_flags_and_layer_cross_the_boundary() {
    let s = sheet();
    let p = open(&s.bytes);
    let v = viewport(&p, s.plain);
    assert_eq!(v.id, 2);
    assert_eq!(v.layer, "VPORTS");
    assert_eq!(v.center, [100.0, 80.0]);
    assert_eq!((v.width, v.height), (120.0, 90.0));
    assert_eq!(v.view_center, [12.5, -3.25]);
    assert_eq!(v.view_height, 45.0);
    assert_eq!(v.view_target, [1.0, 2.0, 0.0]);
    assert_eq!(v.view_direction, [0.0, 0.0, 1.0]);
    assert_eq!(v.twist, 0.5);
    assert!(v.perspective);
    assert!(v.hide_plot);
    assert!(!v.non_rectangular_clipping);
    assert_eq!(
        v.clip,
        ViewportClip::Rectangle([[40.0, 35.0], [160.0, 35.0], [160.0, 125.0], [40.0, 125.0]])
    );
}

#[test]
fn a_circle_boundary_is_named_a_circle_and_not_turned_into_the_rectangle() {
    let s = sheet();
    let p = open(&s.bytes);
    let v = viewport(&p, s.circled);
    assert!(v.non_rectangular_clipping);
    match v.clip {
        ViewportClip::Circle(h) => assert_eq!(h.get(), s.circle),
        other => panic!("a circle boundary, not {other:?}"),
    }
}

#[test]
fn a_polyline_boundary_keeps_its_bulges() {
    let s = sheet();
    let p = open(&s.bytes);
    let v = viewport(&p, s.bulged);
    let ViewportClip::Polygon(vertices) = &v.clip else {
        panic!("a polyline boundary is a polygon: {:?}", v.clip);
    };
    let bulges: Vec<f64> = vertices.iter().map(|q| q.bulge).collect();
    assert_eq!(bulges, vec![0.0, 0.5, 0.0]);
}
