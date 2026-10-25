//! N4.6b (#1296): `AcadrustProvider` fills the entity visibility, layer override and viewport
//! facets from a real DWG. The drawing is built in memory with acadrust, written out as a DWG
//! and read back through the provider, so the facts have crossed a real file.
//!
//! Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use std::io::Cursor;

use acadrust::entities::Viewport;
use acadrust::{CadDocument, DwgWriter, EntityType, Line, Vector3};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{
    Availability, DwgProvider, EntityScope, EntityStates, Facet, FacetSource, LayoutKind,
    PolicyReason, SourceFact, ViewProfile, ViewportFreezeMode, VisibilityFilter,
};

struct Drawing {
    bytes: Vec<u8>,
    plain: u64,
    hidden: u64,
    on_off_layer: u64,
    viewport: u64,
}

fn line_on(layer: &str, invisible: bool) -> EntityType {
    let mut l = Line::from_points(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    l.common.layer = layer.to_string();
    l.common.invisible = invisible;
    EntityType::Line(l)
}

fn drawing() -> Drawing {
    let mut doc = CadDocument::new();
    let mut off = acadrust::tables::Layer::new("OffLayer");
    off.turn_off();
    doc.layers.add(off).expect("layer");
    let mut cold = acadrust::tables::Layer::new("ColdInSheet");
    doc.layers.add(cold.clone()).expect("layer");
    cold.handle = doc.layers.get("ColdInSheet").expect("layer").handle;

    let plain = doc.add_entity(line_on("0", false)).expect("plain").value();
    let hidden = doc.add_entity(line_on("0", true)).expect("hidden").value();
    let on_off_layer = doc
        .add_entity(line_on("OffLayer", false))
        .expect("off")
        .value();

    doc.add_layout("Sheet A").expect("layout");
    let mut vp = Viewport::with_size(Vector3::new(100.0, 100.0, 0.0), 80.0, 40.0);
    vp.id = 2;
    vp.freeze_layer(cold.handle);
    let viewport = doc
        .add_entity_to_layout(EntityType::Viewport(vp), "Sheet A")
        .expect("viewport")
        .value();

    let mut bytes = Vec::new();
    DwgWriter::write_to_writer(Cursor::new(&mut bytes), &doc).expect("write");
    Drawing {
        bytes,
        plain,
        hidden,
        on_off_layer,
        viewport,
    }
}

fn open(d: &Drawing) -> AcadrustProvider {
    AcadrustProvider::open(CadSource::Bytes(&d.bytes)).expect("opens")
}

fn model_scope(p: &AcadrustProvider) -> EntityScope {
    let m = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model");
    EntityScope::Layout(m.id)
}

#[test]
fn pvec_1296_caps_claim_the_three_facets_from_the_decoder_directly() {
    let p = open(&drawing());
    for f in [
        Facet::EntityVisibility,
        Facet::LayerOverride,
        Facet::ViewportEntities,
    ] {
        let row = p
            .caps()
            .get(f)
            .unwrap_or_else(|| panic!("no row for {f:?}"));
        assert_eq!(row.availability, Availability::Reported, "{f:?}");
        assert_eq!(row.source, FacetSource::DecoderDirect, "{f:?}");
    }
}

#[test]
fn pvec_1296_entity_facts_carry_the_invisible_flag_by_handle() {
    let d = drawing();
    let p = open(&d);
    let facts = p.entity_facts(&model_scope(&p)).expect("facts");
    let vis = |h: u64| {
        facts
            .get(&libviprs::cad::ItemHandle::new(h))
            .unwrap_or_else(|| panic!("no facts for {h:X}"))
            .visible
            .clone()
    };
    assert!(matches!(vis(d.plain), SourceFact::Present(true)));
    assert!(matches!(vis(d.hidden), SourceFact::Present(false)));
}

#[test]
fn pvec_1296_the_providers_own_answer_counts_layer_state() {
    let d = drawing();
    let p = open(&d);
    let facts = p.entity_facts(&model_scope(&p)).expect("facts");
    let own = |h: u64| {
        facts[&libviprs::cad::ItemHandle::new(h)]
            .provider_visible
            .clone()
    };
    assert!(matches!(own(d.plain), SourceFact::Present(true)));
    assert!(matches!(own(d.hidden), SourceFact::Present(false)));
    assert!(matches!(own(d.on_off_layer), SourceFact::Present(false)));
}

#[test]
fn pvec_1296_a_drawing_has_no_second_layer_so_the_override_is_absent_not_unreadable() {
    let d = drawing();
    let p = open(&d);
    let facts = p.entity_facts(&model_scope(&p)).expect("facts");
    for f in facts.values() {
        assert!(matches!(f.layer_override, SourceFact::Absent));
    }
}

#[test]
fn pvec_1296_viewport_entities_carry_visibility_clip_and_frozen_layer_names() {
    let d = drawing();
    let p = open(&d);
    let vps = p.viewports().expect("viewports");
    let vp = vps
        .iter()
        .find(|v| v.handle.get() == d.viewport)
        .expect("the viewport we wrote");
    assert!(vp.visible);
    assert_eq!(vp.frozen_layers, vec!["ColdInSheet".to_string()]);
    // 80 wide, 40 high, centred on (100, 100).
    let xs: Vec<f64> = vp.clip.iter().map(|c| c[0]).collect();
    let ys: Vec<f64> = vp.clip.iter().map(|c| c[1]).collect();
    assert_eq!(vp.clip.len(), 4);
    assert!((xs.iter().cloned().fold(f64::MAX, f64::min) - 60.0).abs() < 1e-9);
    assert!((xs.iter().cloned().fold(f64::MIN, f64::max) - 140.0).abs() < 1e-9);
    assert!((ys.iter().cloned().fold(f64::MAX, f64::min) - 80.0).abs() < 1e-9);
    assert!((ys.iter().cloned().fold(f64::MIN, f64::max) - 120.0).abs() < 1e-9);
    let sheet = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.name == "Sheet A")
        .expect("Sheet A");
    assert_eq!(vp.layout, sheet.id);
}

#[test]
fn pvec_1296_the_filter_drops_what_the_drawing_hid_with_no_inputs_handed_in() {
    let d = drawing();
    let p = open(&d);
    let out = VisibilityFilter::run_from_provider(
        &p,
        &ViewProfile::default_profile(),
        ViewportFreezeMode::ReportOnly,
    )
    .expect("runs");
    let dropped: Vec<(u64, PolicyReason)> = out
        .ignored
        .iter()
        .filter_map(|r| r.handle.map(|h| (h.get(), r.reason)))
        .collect();
    assert!(dropped.contains(&(d.hidden, PolicyReason::EntityInvisible)));
    assert!(dropped.contains(&(d.on_off_layer, PolicyReason::LayerOff)));
    assert!(
        out.kept
            .iter()
            .any(|e| e.provenance.handle.map(|h| h.get()) == Some(d.plain))
    );
}

#[test]
fn pvec_1296_entity_states_for_a_block_scope_come_from_the_same_provider() {
    let d = drawing();
    let p = open(&d);
    let st = EntityStates::from_provider(&p, &model_scope(&p)).expect("states");
    let s = st.get(Some(libviprs::cad::ItemHandle::new(d.hidden)));
    assert!(!s.visible);
}
