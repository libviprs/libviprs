//! Issue #1626 (C1.1 of iasbuilt/webapp #1434): the CAD manifest and the one visibility resolver.
//! `cad_manifest` lists a drawing without converting it, `resolve_visibility` applies the truth
//! table of spec section 1 (tests 9.1 to 9.6). Needs `--features pvec-convert`.
//!
//! The big-file test reads the drawing named by `IASBUILT_DWG_LAYOUTS_FIXTURE` and says so when
//! the variable is unset.

#![cfg(feature = "pvec-convert")]

use std::collections::BTreeMap;
use std::path::Path;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec_convert::{
    CadChoices, CadManifest, CadMode, LayerOverride, ManifestLayer, ManifestLayerState,
    ManifestLayout, ManifestMask, ManifestViewport, PlotsVisibility, StateLayer, Visibility, VpRef,
    cad_manifest, resolve_visibility,
};

const LAYOUT: &str = "SHEET_001";
const VIEWPORT: &str = "2A3";

fn choices() -> CadChoices {
    serde_json::from_str(r#"{"schema":1,"mode":"layouts"}"#).expect("minimal choices parse")
}

/// One layer, one layout with one viewport that freezes the layer when `vp_frozen`.
fn manifest(name: &str, flags: [bool; 5]) -> CadManifest {
    let [on, frozen, locked, plot, vp_frozen] = flags;
    let layer = ManifestLayer {
        name: name.into(),
        on,
        frozen,
        locked,
        plot,
        vp_frozen_in: if vp_frozen {
            vec![VpRef {
                layout: LAYOUT.into(),
                viewport: VIEWPORT.into(),
            }]
        } else {
            vec![]
        },
        ..ManifestLayer::default()
    };
    let viewport = ManifestViewport {
        id: VIEWPORT.into(),
        index: 2,
        on: true,
        frozen_layers: if vp_frozen { vec![name.into()] } else { vec![] },
        ..ManifestViewport::default()
    };
    CadManifest {
        schema: 1,
        layers: vec![layer],
        layouts: vec![ManifestLayout {
            name: LAYOUT.into(),
            tab: 1,
            sheet: true,
            viewports: vec![viewport],
            ..ManifestLayout::default()
        }],
        ..CadManifest::default()
    }
}

fn vp_key() -> String {
    format!("{LAYOUT}/{VIEWPORT}")
}

fn vis(b: bool) -> Visibility {
    if b {
        Visibility::Visible
    } else {
        Visibility::Hidden
    }
}

// 9.1 resolver-truth-table
#[test]
fn resolver_truth_table() {
    for bits in 0u8..32 {
        let flags = [0, 1, 2, 3, 4].map(|i| bits & (1 << i) != 0);
        let [on, frozen, _locked, plot, vp_frozen] = flags;
        let got = resolve_visibility(&manifest("A-WALL", flags), &choices());
        let l = &got[0];
        let base = on && !frozen && plot;
        assert_eq!(l.model, vis(on && !frozen), "model for {flags:?}");
        let plots = match (base, vp_frozen) {
            (false, _) => PlotsVisibility::Hidden,
            (true, false) => PlotsVisibility::Visible,
            (true, true) => PlotsVisibility::Viewport,
        };
        assert_eq!(l.plots, plots, "plots for {flags:?}");
        assert_eq!(
            l.per_viewport.get(&vp_key()).copied(),
            Some(vis(base && !vp_frozen)),
            "through the viewport for {flags:?}"
        );
        if plots == PlotsVisibility::Viewport {
            assert_eq!(
                (l.plots_count.frozen_in, l.plots_count.of),
                (1, 1),
                "{flags:?}"
            );
        }
    }
}

// 9.1, the Defpoints row
#[test]
fn resolver_defpoints_never_plots() {
    let got = resolve_visibility(
        &manifest("Defpoints", [true, false, false, true, false]),
        &choices(),
    );
    assert_eq!(got[0].model, Visibility::Visible);
    assert_eq!(got[0].plots, PlotsVisibility::Hidden);
    assert_eq!(
        got[0].per_viewport.get(&vp_key()),
        Some(&Visibility::Hidden)
    );
}

// 9.2 resolver-override-wins
#[test]
fn resolver_override_wins() {
    for name in ["A-WALL", "DEFPOINTS"] {
        for bits in 0u8..32 {
            let flags = [0, 1, 2, 3, 4].map(|i| bits & (1 << i) != 0);
            for (want, vis_want, plots_want) in [
                (
                    LayerOverride::Visible,
                    Visibility::Visible,
                    PlotsVisibility::Visible,
                ),
                (
                    LayerOverride::Hidden,
                    Visibility::Hidden,
                    PlotsVisibility::Hidden,
                ),
            ] {
                let mut c = choices();
                c.layers.insert(name.to_ascii_lowercase(), want);
                let l = &resolve_visibility(&manifest(name, flags), &c)[0];
                assert_eq!(l.model, vis_want, "{name} {flags:?}");
                assert_eq!(l.plots, plots_want, "{name} {flags:?}");
                assert_eq!(
                    l.per_viewport.get(&vp_key()),
                    Some(&vis_want),
                    "{name} {flags:?}"
                );
            }
        }
    }
}

fn state(mask: ManifestMask, layers: Vec<StateLayer>) -> ManifestLayerState {
    ManifestLayerState {
        name: "Floor Plan".into(),
        mask,
        layers,
        missing_layers: vec!["OLD-DEMO".into()],
        unlisted_layers: vec!["A-NEW".into()],
        ..ManifestLayerState::default()
    }
}

// 9.3 resolver-state-mask
#[test]
fn resolver_state_mask() {
    let recorded = StateLayer {
        name: "A-WALL".into(),
        on: false,
        frozen: false,
        locked: false,
        plot: false,
        vp_frozen: false,
    };
    let mut m = manifest("A-WALL", [true, false, false, true, false]);
    let mut c = choices();
    c.layer_state = Some("Floor Plan".into());

    // The mask restores On only: the state's plot flag (false) must not leak in.
    let only_on = ManifestMask {
        on: true,
        ..ManifestMask::default()
    };
    m.layer_states = vec![state(only_on, vec![recorded.clone()])];
    let l = &resolve_visibility(&m, &c)[0];
    assert_eq!(
        l.model,
        Visibility::Hidden,
        "the state switches the layer off"
    );

    let only_plot = ManifestMask {
        plot: true,
        ..ManifestMask::default()
    };
    m.layer_states = vec![state(only_plot, vec![recorded.clone()])];
    let l = &resolve_visibility(&m, &c)[0];
    assert_eq!(l.model, Visibility::Visible, "On is not in the mask");
    assert_eq!(
        l.plots,
        PlotsVisibility::Hidden,
        "Plot is in the mask and the state says no-plot"
    );

    let no_bits = ManifestMask::default();
    m.layer_states = vec![state(no_bits, vec![recorded])];
    let l = &resolve_visibility(&m, &c)[0];
    assert_eq!(
        (l.model, l.plots),
        (Visibility::Visible, PlotsVisibility::Visible)
    );
}

#[test]
fn resolver_turn_off_layers_not_in_state() {
    let mut m = manifest("A-NEW", [true, false, false, true, false]);
    m.layer_states = vec![state(
        ManifestMask {
            on: true,
            ..ManifestMask::default()
        },
        vec![],
    )];
    let mut c = choices();
    c.layer_state = Some("Floor Plan".into());
    assert_eq!(resolve_visibility(&m, &c)[0].model, Visibility::Visible);
    c.turn_off_layers_not_in_state = true;
    let l = &resolve_visibility(&m, &c)[0];
    assert_eq!(
        (l.model, l.plots),
        (Visibility::Hidden, PlotsVisibility::Hidden)
    );
}

// 9.4 resolver-case-insensitive
#[test]
fn resolver_case_insensitive() {
    let mut c = choices();
    c.layers = BTreeMap::from([("a-wall".to_string(), LayerOverride::Hidden)]);
    let l = &resolve_visibility(&manifest("A-WALL", [true, false, false, true, false]), &c)[0];
    assert_eq!(l.model, Visibility::Hidden);
}

// 9.5 resolver-xref-layer
#[test]
fn resolver_xref_layer() {
    let mut m = manifest("XR|A-WALL", [true, false, false, true, false]);
    m.layers[0].xref = Some("XR".into());
    let l = &resolve_visibility(&m, &choices())[0];
    assert_eq!(
        (l.model, l.plots),
        (Visibility::Visible, PlotsVisibility::Visible)
    );
    assert_eq!(l.name, "XR|A-WALL");
}

#[test]
fn choices_refuse_unknown_fields_and_manifest_does_not() {
    assert!(
        serde_json::from_str::<CadChoices>(r#"{"schema":1,"mode":"model","bogus":1}"#).is_err()
    );
    let c = serde_json::from_str::<CadChoices>(
        r#"{"schema":1,"mode":"layouts","layerState":"Floor Plan","layers":{"A-DIMS":"visible"}}"#,
    )
    .expect("choices parse");
    assert_eq!(c.mode, CadMode::Layouts);
    assert_eq!(c.layers.get("A-DIMS"), Some(&LayerOverride::Visible));
    let m = serde_json::from_str::<CadManifest>(
        r#"{"schema":1,"extra":true,"layers":[],"layouts":[]}"#,
    );
    assert!(m.is_ok(), "the manifest ignores fields it does not know");
}

// A small DXF with a paper layout and viewports: the listing agrees with the provider.
#[test]
fn cad_manifest_lists_a_dxf_without_converting() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-dxf/paper-space-multi-viewport-001.dxf");
    let m = cad_manifest::<AcadrustProvider>(CadSource::Path(&path)).expect("the DXF lists");
    assert_eq!(m.schema, 1);
    assert!(!m.layers.is_empty(), "a drawing has layers");
    assert!(!m.layouts.is_empty(), "the fixture has a paper layout");
    let tabs: Vec<u32> = m.layouts.iter().map(|l| l.tab).collect();
    assert_eq!(
        tabs,
        (1..=tabs.len() as u32).collect::<Vec<_>>(),
        "tab order"
    );
    assert!(
        m.layouts.iter().any(|l| l.viewports.len() > 1),
        "the fixture has more than one viewport on a layout"
    );
}

// 9.6 cad-manifest-lists-without-converting
#[test]
fn cad_manifest_lists_without_converting() {
    let Some(path) = std::env::var_os("IASBUILT_DWG_LAYOUTS_FIXTURE") else {
        eprintln!("skipped: IASBUILT_DWG_LAYOUTS_FIXTURE is not set (the 66-layout drawing)");
        return;
    };
    let m = cad_manifest::<AcadrustProvider>(CadSource::Path(Path::new(&path)))
        .expect("the drawing lists");
    assert_eq!(m.layouts.len(), 66, "66 layouts, skipped ones included");
    let at = |tab: usize| m.layouts[tab - 1].name.as_str();
    assert_eq!(
        (at(21), at(65), at(66)),
        ("SHEET_010", "SHEET_011", "SHEET_012")
    );
    assert!(
        m.layouts
            .iter()
            .enumerate()
            .all(|(i, l)| l.tab as usize == i + 1)
    );
    let viewports: usize = m.layouts.iter().map(|l| l.viewports.len()).sum();
    assert_eq!(viewports, 113, "113 viewports");
    assert!(!m.layers.is_empty() && m.layers.iter().all(|l| !l.name.is_empty()));
    assert!(m.model_space.entity_count > 0, "the count pass ran");
    let sheet = m
        .layouts
        .iter()
        .find(|l| l.name == "SHEET_011")
        .expect("SHEET_011 is listed");
    let paper = sheet.paper.as_ref().expect("SHEET_011 lists its paper");
    assert!((paper.width_mm - 215.9).abs() < 0.5 && (paper.height_mm - 279.4).abs() < 0.5);
    assert_eq!(paper.orientation, "portrait");
}

// What a server needs to request one thumbnail per layout: name, tab order, paper, plot area.
#[test]
fn cad_manifest_carries_what_a_thumbnail_needs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-dxf/paper-space-multi-viewport-001.dxf");
    let m = cad_manifest::<AcadrustProvider>(CadSource::Path(&path)).expect("the DXF lists");
    for l in m.layouts.iter().filter(|l| l.sheet) {
        let paper = l.paper.as_ref().expect("a sheet lists its paper");
        assert!(
            paper.width_mm >= 0.0 && paper.height_mm >= 0.0,
            "{}",
            l.name
        );
        assert!(l.plot.is_some(), "a sheet lists its plot area: {}", l.name);
        assert!(l.tab >= 1, "{}", l.name);
    }
}

fn dxf(name: &str) -> CadManifest {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-cad-import")
        .join(name);
    cad_manifest::<AcadrustProvider>(CadSource::Path(&path)).expect("the DXF lists")
}

fn layer<'a>(m: &'a CadManifest, name: &str) -> &'a ManifestLayer {
    m.layers
        .iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("{name} is listed"))
}

// The flags fixture: every flag is read and the xref warning fires.
#[test]
fn cad_import_flags_fixture_lists_every_flag() {
    let m = dxf("cad-import-flags.dxf");
    assert!(layer(&m, "A-ON").on && !layer(&m, "A-OFF").on);
    assert!(layer(&m, "A-FROZEN").frozen && !layer(&m, "A-ON").frozen);
    assert!(layer(&m, "A-LOCKED").locked);
    assert!(!layer(&m, "A-NOPLOT").plot && layer(&m, "A-ON").plot);
    assert!(layer(&m, "A-NEWVP").frozen_in_new_viewports);
    assert_eq!(layer(&m, "XR|A-WALL").xref.as_deref(), Some("XR"));
    let codes: Vec<&str> = m.warnings.iter().map(|w| w.code.as_str()).collect();
    assert!(codes.contains(&"XREF_LAYERS"), "{codes:?}");

    let resolved = resolve_visibility(&m, &choices());
    let of = |n: &str| resolved.iter().find(|r| r.name == n).expect("resolved");
    assert_eq!(of("A-OFF").model, Visibility::Hidden);
    assert_eq!(of("A-FROZEN").model, Visibility::Hidden);
    assert_eq!(of("A-NOPLOT").model, Visibility::Visible);
    assert_eq!(of("A-NOPLOT").plots, PlotsVisibility::Hidden);
    assert_eq!(of("Defpoints").plots, PlotsVisibility::Hidden);
}

// The plot fixture: a viewport freezes a layer, so the layer is "Visible (Viewport)".
#[test]
fn cad_import_plot_fixture_resolves_a_viewport_freeze() {
    let m = dxf("cad-import-plot.dxf");
    let frozen = layer(&m, "A-VPFROZEN");
    assert_eq!(
        frozen.vp_frozen_in.len(),
        1,
        "one viewport freezes it: {:?}",
        frozen.vp_frozen_in
    );
    let resolved = resolve_visibility(&m, &choices());
    let of = |n: &str| resolved.iter().find(|r| r.name == n).expect("resolved");
    assert_eq!(of("A-VPFROZEN").model, Visibility::Visible);
    assert_eq!(of("A-VPFROZEN").plots, PlotsVisibility::Viewport);
    assert_eq!(of("A-VPFROZEN").plots_count.frozen_in, 1);
    assert_eq!(of("A-PLOTTED").plots, PlotsVisibility::Visible);
    assert_eq!(of("A-NOPLOT").plots, PlotsVisibility::Hidden);
}
