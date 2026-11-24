//! Issue #1675: a model space conversion follows the drawing's saved layer table, or a named
//! layer state applied under its mask (design on libviprs#1670, decision D6).
//!
//! The drawing is small and written here with acadrust: six layers with one LINE each in model
//! space, and two stored states, `Saved` (the table as the file has it) and `Review`, which
//! restores only on/off and frozen. Each case runs the whole path: the manifest read back from
//! the DXF bytes, `apply_choices` on the default profile, then a conversion whose report says
//! which entities a policy dropped. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::collections::BTreeSet;
use std::fs;

use acadrust::entities::Line;
use acadrust::tables::Layer;
use acadrust::{CadDocument, DxfWriter, EntityType, LayerStateMask};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::{
    CadChoices, CadManifest, PlotsVisibility, PvecConversion, Visibility, apply_choices,
    cad_manifest, check_layer_state, resolve_visibility,
};

/// `(layer, off, frozen, locked, plots)`.
const LAYERS: [(&str, bool, bool, bool, bool); 6] = [
    ("WALLS", false, false, false, true),
    ("DOORS", false, false, false, true),
    ("OFF-L", true, false, false, true),
    ("FROZEN-L", false, true, false, true),
    ("LOCKED-L", false, false, true, true),
    ("NOPLOT-L", false, false, false, false),
];

fn drawing() -> CadDocument {
    let mut doc = CadDocument::new();
    for (name, off, frozen, locked, plots) in LAYERS
        .iter()
        .copied()
        .chain([("ANNOT", false, false, false, true)])
    {
        doc.layers
            .add(Layer::new(name))
            .expect("the layer is added");
        let layer = doc.layers.get_mut(name).expect("the layer exists");
        layer.flags.off = off;
        layer.flags.frozen = frozen;
        layer.flags.locked = locked;
        layer.is_plottable = plots;
    }
    doc.assign_table_entry_handles();
    for (n, (name, ..)) in LAYERS
        .iter()
        .chain(&[("ANNOT", false, false, false, true)])
        .enumerate()
    {
        let y = n as f64 * 10.0;
        let mut line = Line::from_coords(0.0, y, 0.0, 10.0, y, 0.0);
        line.common.layer = (*name).into();
        doc.add_entity(EntityType::Line(line))
            .expect("the line is added");
    }
    // The file's own table, then a state that restores only on/off and frozen: WALLS goes off,
    // OFF-L comes on, FROZEN-L thaws. The lock and plot flips and the state's missing ANNOT
    // are there to show what the mask and the list leave alone.
    doc.capture_layer_state("Saved", "the table as the file has it");
    let mut review = doc.layer_state("Saved").expect("Saved is stored");
    review.name = "Review".into();
    review.mask = LayerStateMask::ON | LayerStateMask::FROZEN;
    for l in &mut review.layers {
        match l.layer_name.as_str() {
            "WALLS" => l.off = true,
            "OFF-L" => l.off = false,
            "FROZEN-L" => l.frozen = false,
            "LOCKED-L" => l.locked = false,
            "NOPLOT-L" => l.plottable = true,
            _ => {}
        }
    }
    review.layers.retain(|l| l.layer_name != "ANNOT");
    doc.store_layer_state(review);
    doc
}

fn bytes() -> Vec<u8> {
    DxfWriter::new(&drawing())
        .write_to_vec()
        .expect("the DXF writes")
}

/// Each LINE's handle and the layer it is on.
fn lines() -> Vec<(u64, String)> {
    drawing()
        .entities()
        .map(|e| (e.common().handle.value(), e.common().layer.clone()))
        .collect()
}

fn manifest() -> CadManifest {
    cad_manifest::<AcadrustProvider>(CadSource::Bytes(&bytes())).expect("the drawing lists")
}

fn choices(json: &str) -> CadChoices {
    serde_json::from_str(json).expect("the choices parse")
}

/// The layers whose LINE the conversion kept, under the default profile after `choices`.
fn drawn(tag: &str, json: &str) -> BTreeSet<String> {
    let profile = apply_choices(ViewProfile::default_profile(), &manifest(), &choices(json));
    let dir = std::env::temp_dir().join(format!("libviprs-1675-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let out = PvecConversion::new(CadSource::Bytes(&bytes()), &dir.join("out.pvec"))
        .unit(SourceUnit::Meters)
        .view_profile(profile)
        .convert_with::<AcadrustProvider>()
        .expect("model space converts");
    let _ = fs::remove_dir_all(dir);
    let lines = lines();
    out.report
        .entities
        .iter()
        .filter(|r| r.fidelity != "IgnoredByPolicy")
        .filter_map(|r| lines.iter().find(|(h, _)| Some(*h) == r.source_handle))
        .map(|(_, layer)| layer.clone())
        .collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1675_with_no_state_the_saved_layer_table_decides() {
    // Off and frozen drop. Locked changes nothing, and a no-plot layer still shows in model space.
    assert_eq!(
        drawn("table", r#"{"schema":1,"mode":"model"}"#),
        set(&["WALLS", "DOORS", "LOCKED-L", "NOPLOT-L", "ANNOT"])
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1675_a_named_state_applies_only_what_its_mask_restores() {
    // Review restores on/off and frozen: WALLS goes off, OFF-L and FROZEN-L come back. Its lock
    // and plot rows are outside the mask, and ANNOT is not listed, so both stay as the file has them.
    assert_eq!(
        drawn(
            "review",
            r#"{"schema":1,"mode":"model","layerState":"Review"}"#
        ),
        set(&[
            "DOORS", "OFF-L", "FROZEN-L", "LOCKED-L", "NOPLOT-L", "ANNOT"
        ])
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1675_a_state_name_matches_without_regard_to_case() {
    assert_eq!(
        drawn(
            "case",
            r#"{"schema":1,"mode":"model","layerState":"rEVIEW"}"#
        ),
        drawn(
            "exact",
            r#"{"schema":1,"mode":"model","layerState":"Review"}"#
        )
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1675_turning_off_the_layers_a_state_does_not_list_hides_them() {
    let got = drawn(
        "unlisted",
        r#"{"schema":1,"mode":"model","layerState":"Review","turnOffLayersNotInState":true}"#,
    );
    assert!(!got.contains("ANNOT"), "ANNOT is not in Review: {got:?}");
    assert!(got.contains("DOORS") && got.contains("OFF-L"), "{got:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1675_a_user_override_still_beats_the_state() {
    let got = drawn(
        "override",
        r#"{"schema":1,"mode":"model","layerState":"Review","layers":{"WALLS":"visible","DOORS":"hidden"}}"#,
    );
    assert!(got.contains("WALLS") && !got.contains("DOORS"), "{got:?}");
}

#[test]
fn issue_1675_locked_and_no_plot_layers_show_in_model_space() {
    let resolved = resolve_visibility(&manifest(), &choices(r#"{"schema":1,"mode":"model"}"#));
    let of = |n: &str| {
        resolved
            .iter()
            .find(|r| r.name == n)
            .expect("the layer is listed")
    };
    assert_eq!(of("LOCKED-L").model, Visibility::Visible);
    assert_eq!(of("LOCKED-L").plots, PlotsVisibility::Visible);
    // No-plot shows in model space and is out of the plots (D6).
    assert_eq!(of("NOPLOT-L").model, Visibility::Visible);
    assert_eq!(of("NOPLOT-L").plots, PlotsVisibility::Hidden);
}

#[test]
fn issue_1675_an_unknown_state_is_an_error_naming_the_ones_there_are() {
    let m = manifest();
    let err = check_layer_state(
        &m,
        &choices(r#"{"schema":1,"mode":"model","layerState":"Night"}"#),
    )
    .expect_err("Night is not stored");
    assert_eq!(err.code(), "LAYER_STATE_NOT_FOUND");
    assert_eq!(err.name, "Night");
    assert_eq!(err.available, ["Saved", "Review"]);
    let said = err.to_string();
    assert!(
        said.contains("Night") && said.contains("Saved") && said.contains("Review"),
        "{said}"
    );
}

#[test]
fn issue_1675_a_known_state_or_none_passes_the_check() {
    let m = manifest();
    for json in [
        r#"{"schema":1,"mode":"model"}"#,
        r#"{"schema":1,"mode":"model","layerState":"review"}"#,
    ] {
        assert_eq!(check_layer_state(&m, &choices(json)), Ok(()), "{json}");
    }
}

#[test]
fn issue_1675_viewport_overrides_are_for_sheets_and_a_model_space_profile_has_none() {
    let json = |mode: &str| {
        format!(r#"{{"schema":1,"mode":"{mode}","viewports":{{"Sheet/2A":{{"WALLS":"hidden"}}}}}}"#)
    };
    let model = apply_choices(
        ViewProfile::default_profile(),
        &manifest(),
        &choices(&json("model")),
    );
    assert!(
        model.layers.viewport_overrides.is_empty(),
        "{:?}",
        model.layers.viewport_overrides
    );
    // On a sheet they are kept, so the case above fails for the right reason.
    let sheet = apply_choices(
        ViewProfile::print_profile("Sheet"),
        &manifest(),
        &choices(&json("layouts")),
    );
    assert_eq!(sheet.layers.viewport_overrides.len(), 1);
}
