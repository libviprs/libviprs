//! Issue #1626: the manifest read through acadrust from drawings built here, so every case spec 9
//! lists is on the page. The drawings are written to DXF text in memory and read back by the
//! provider, which means layer states, layouts, viewports and their frozen lists come from the
//! reader and not from a hand-built manifest. No binary fixture is needed. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use acadrust::entities::{Line, Text, Viewport};
use acadrust::objects::ObjectType;
use acadrust::tables::Layer;
use acadrust::types::{Handle, Vector3};
use acadrust::{CadDocument, DxfWriter, EntityType, LayerStateMask};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec_convert::{
    CadChoices, CadManifest, ManifestLayer, ManifestLayout, PlotsVisibility, Visibility,
    cad_manifest, resolve_visibility,
};

/// `(name, off, frozen, locked, plots)`, one layer per row of the truth table that does not need a
/// viewport, then the three that do.
const LAYERS: [(&str, bool, bool, bool, bool); 13] = [
    ("ON-THAW-PLOT", false, false, false, true),
    ("OFF", true, false, false, true),
    ("FROZEN", false, true, false, true),
    ("LOCKED", false, false, true, true),
    ("NOPLOT", false, false, false, false),
    ("VPFROZEN-ONE", false, false, false, true),
    ("VPFROZEN-ALL", false, false, false, true),
    ("OFF-FROZEN", true, true, false, true),
    ("FROZEN-NOPLOT", false, true, false, false),
    ("LOCK-NOPLOT", false, false, true, false),
    ("Defpoints", false, false, false, true),
    ("XR|LAYER", false, false, false, true),
    ("PAPER-ONLY", false, false, false, true),
];

fn line_on(layer: &str) -> EntityType {
    let mut line = Line::from_coords(0.0, 0.0, 0.0, 10.0, 10.0, 0.0);
    line.common.layer = layer.into();
    EntityType::Line(line)
}

fn handle_of(doc: &CadDocument, layer: &str) -> Handle {
    doc.layers.get(layer).expect("the layer exists").handle
}

/// A model-space window onto the model at `id` (2 and up), plan view, rectangular clip.
fn window(id: i16, frozen: &[Handle]) -> Viewport {
    let mut v = Viewport::with_size(Vector3::new(100.0, 100.0, 0.0), 80.0, 60.0);
    v.id = id;
    v.view_height = 30.0;
    v.view_center = Vector3::new(5.0, 5.0, 0.0);
    v.frozen_layers = frozen.to_vec();
    v
}

fn put(doc: &mut CadDocument, layout: &str, entity: EntityType) {
    doc.add_entity_to_layout(entity, layout)
        .expect("the entity is added to the layout");
}

fn set_plot(doc: &mut CadDocument, layout: &str, width: f64, height: f64, rotation: i16) {
    for object in doc.objects.values_mut() {
        if let ObjectType::Layout(l) = object
            && l.name == layout
        {
            l.paper_width = width;
            l.paper_height = height;
            l.plot_rotation = rotation;
            l.paper_size = format!("{width}x{height}");
        }
    }
}

/// The drawing of spec 9: one layer per truth-table row, a LINE and a TEXT in model space for
/// each, a LINE in paper space for each, and layouts that cover every skip and viewport case.
fn drawing() -> CadDocument {
    let mut doc = CadDocument::new();
    for (name, off, frozen, locked, plots) in LAYERS {
        if doc.layers.get(name).is_none() {
            doc.layers
                .add(Layer::new(name))
                .expect("the layer is added");
        }
        let layer = doc.layers.get_mut(name).expect("the layer exists");
        layer.flags.off = off;
        layer.flags.frozen = frozen;
        layer.flags.locked = locked;
        layer.is_plottable = plots;
    }
    doc.assign_table_entry_handles();
    for (name, ..) in LAYERS.iter().filter(|l| l.0 != "PAPER-ONLY") {
        doc.add_entity(line_on(name)).expect("model line");
        let mut text = Text::with_value("note", Vector3::new(1.0, 1.0, 0.0));
        text.common.layer = (*name).into();
        doc.add_entity(EntityType::Text(text)).expect("model text");
    }

    for layout in [
        "SHEET_A", "SHEET_B", "SHEET_C", "TITLE", "EMPTY", "BLANK", "ROTATED", "ROT3",
    ] {
        doc.add_layout(layout).expect("the layout is added");
    }

    // SHEET_A: two windows. One layer is frozen in the first only, one in both, and a paper-space
    // only layer is "frozen" in the first too, which changes nothing.
    let (one, all, paper_only) = (
        handle_of(&doc, "VPFROZEN-ONE"),
        handle_of(&doc, "VPFROZEN-ALL"),
        handle_of(&doc, "PAPER-ONLY"),
    );
    for (name, ..) in &LAYERS {
        put(&mut doc, "SHEET_A", line_on(name));
    }
    put(
        &mut doc,
        "SHEET_A",
        EntityType::Viewport(Box::new(window(2, &[one, all, paper_only]))),
    );
    put(
        &mut doc,
        "SHEET_A",
        EntityType::Viewport(Box::new(window(3, &[all]))),
    );
    set_plot(&mut doc, "SHEET_A", 420.0, 297.0, 0);

    // SHEET_B: the layer frozen everywhere is frozen here too; the second window is off.
    let mut off = window(3, &[all]);
    off.status.is_on = false;
    put(
        &mut doc,
        "SHEET_B",
        EntityType::Viewport(Box::new(window(2, &[all]))),
    );
    put(&mut doc, "SHEET_B", EntityType::Viewport(Box::new(off)));
    set_plot(&mut doc, "SHEET_B", 210.0, 297.0, 0);

    // SHEET_C: the viewports a print leaves out or refuses, one each.
    let mut hide = window(3, &[]);
    hide.status.hide_plot = true;
    let mut twist = window(4, &[]);
    twist.twist_angle = 0.5;
    let mut plan = window(5, &[]);
    plan.view_direction = Vector3::new(0.0, 1.0, 0.0);
    let mut perspective = window(6, &[]);
    perspective.status.perspective = true;
    let mut gone = window(2, &[]);
    gone.status.is_on = false;
    for v in [gone, hide, twist, plan, perspective] {
        put(&mut doc, "SHEET_C", EntityType::Viewport(Box::new(v)));
    }
    set_plot(&mut doc, "SHEET_C", 297.0, 210.0, 0);

    // TITLE: a title block and no window. EMPTY: nothing, its paper view is off. BLANK: only its
    // paper view. Layout1, which the drawing starts with, has none. ROTATED and ROT3: a quarter turn each way.
    put(&mut doc, "TITLE", line_on("0"));
    set_plot(&mut doc, "TITLE", 297.0, 210.0, 0);
    let paper_view = doc
        .objects
        .values()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == "EMPTY" => Some(l.viewport),
            _ => None,
        })
        .expect("EMPTY has its paper view");
    let Some(EntityType::Viewport(v)) = doc.get_entity_mut(paper_view) else {
        panic!("the paper view is a viewport");
    };
    v.status.is_on = false;
    put(&mut doc, "ROTATED", line_on("0"));
    set_plot(&mut doc, "ROTATED", 210.0, 297.0, 1);
    put(&mut doc, "ROT3", line_on("0"));
    set_plot(&mut doc, "ROT3", 420.0, 297.0, 3);

    // The states: everything the file has; a mask that leaves Plot out and flips three flags; one
    // that names layers the drawing no longer has; one with viewport settings.
    doc.capture_layer_state("Full", "every property");
    let full = doc.layer_state("Full").expect("Full is stored");
    let mut floor = full.clone();
    floor.name = "Floor Plan".into();
    floor.mask = LayerStateMask::ON | LayerStateMask::FROZEN;
    for l in &mut floor.layers {
        match l.layer_name.as_str() {
            "OFF" => l.off = false,
            "FROZEN" => l.frozen = false,
            "NOPLOT" => l.plottable = true,
            _ => {}
        }
    }
    floor.layers.retain(|l| l.layer_name != "LOCKED");
    doc.store_layer_state(floor);
    let mut missing = full.clone();
    missing.name = "Old Demo".into();
    let mut stale = missing.layers[0].clone();
    stale.layer_name = "OLD-DEMO".into();
    stale.layer_handle = Handle::new(0x9999);
    missing.layers.push(stale);
    doc.store_layer_state(missing);
    let mut viewport = full;
    viewport.name = "Viewport".into();
    viewport.mask = LayerStateMask::SUPPORTED | LayerStateMask::CURRENT_VIEWPORT;
    viewport.hidden = true;
    doc.store_layer_state(viewport);
    doc
}

fn listed(doc: &CadDocument) -> CadManifest {
    let bytes = DxfWriter::new(doc).write_to_vec().expect("the DXF writes");
    cad_manifest::<AcadrustProvider>(CadSource::Bytes(&bytes)).expect("the drawing lists")
}

fn layer<'a>(m: &'a CadManifest, name: &str) -> &'a ManifestLayer {
    m.layers
        .iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("{name} is listed"))
}

fn layout<'a>(m: &'a CadManifest, name: &str) -> &'a ManifestLayout {
    m.layouts
        .iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("{name} is listed"))
}

fn choices(json: &str) -> CadChoices {
    serde_json::from_str(json).expect("the choices parse")
}

// The D4 rule: the drawing has what the other tests assert on before they assert.
#[test]
fn the_drawing_has_what_the_tests_below_assert_on() {
    let doc = drawing();
    for (name, ..) in &LAYERS {
        assert!(doc.layers.get(name).is_some(), "{name} is in the table");
    }
    assert_eq!(doc.layer_states().len(), 4, "four layer states");
    let m = listed(&doc);
    assert_eq!(m.layouts.len(), 9, "nine paper layouts");
    assert!(m.model_space.entity_count >= 24, "model space has entities");
}

#[test]
fn layer_flags_and_counts_come_through_the_reader() {
    let m = listed(&drawing());
    for (name, off, frozen, locked, plots) in LAYERS {
        let l = layer(&m, name);
        assert_eq!(
            (!l.on, l.frozen, l.locked, l.plot),
            (off, frozen, locked, plots),
            "{name}"
        );
        let (model, text) = if name == "PAPER-ONLY" { (0, 0) } else { (2, 1) };
        assert_eq!(
            (
                l.entity_count.model,
                l.entity_count.paper,
                l.entity_count.text
            ),
            (model, 1, text),
            "{name}: one line and one text in model space, one line in paper space"
        );
    }
    assert_eq!(layer(&m, "XR|LAYER").xref.as_deref(), Some("XR"));
    assert!(layer(&m, "Defpoints").plot, "Defpoints has plot on here");
    assert_eq!(m.model_space.entity_count, 24);
    assert_eq!(m.model_space.text_count, 12);
}

#[test]
fn viewport_freezes_are_listed_per_layout_and_viewport() {
    let m = listed(&drawing());
    let in_layouts = |name: &str| {
        let mut v: Vec<&str> = layer(&m, name)
            .vp_frozen_in
            .iter()
            .map(|r| r.layout.as_str())
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(in_layouts("VPFROZEN-ONE"), ["SHEET_A"]);
    assert_eq!(
        in_layouts("VPFROZEN-ALL"),
        ["SHEET_A", "SHEET_A", "SHEET_B", "SHEET_B"]
    );
    assert!(in_layouts("ON-THAW-PLOT").is_empty());
}

#[test]
fn layouts_say_which_are_sheets_and_why_the_others_are_not() {
    let m = listed(&drawing());
    let tabs: Vec<(&str, u32)> = m.layouts.iter().map(|l| (l.name.as_str(), l.tab)).collect();
    assert_eq!(
        tabs,
        [
            ("Layout1", 1),
            ("SHEET_A", 2),
            ("SHEET_B", 3),
            ("SHEET_C", 4),
            ("TITLE", 5),
            ("EMPTY", 6),
            ("BLANK", 7),
            ("ROTATED", 8),
            ("ROT3", 9),
        ]
    );
    for name in ["SHEET_A", "SHEET_B", "SHEET_C", "TITLE", "ROTATED", "ROT3"] {
        let l = layout(&m, name);
        assert!(l.sheet && l.skip.is_none(), "{name} is a sheet");
        assert!(
            l.paper.is_some() && l.plot.is_some(),
            "{name} lists its plot"
        );
    }
    let skipped = |name: &str| {
        let l = layout(&m, name);
        assert!(!l.sheet && l.paper.is_none(), "{name} is skipped");
        l.skip.clone()
    };
    assert_eq!(skipped("EMPTY").as_deref(), Some("empty"));
    assert_eq!(
        skipped("Layout1").as_deref(),
        Some("empty"),
        "the drawing's own, no paper view"
    );
    assert_eq!(skipped("BLANK").as_deref(), Some("emptyAfterView"));
    assert_eq!(
        layout(&m, "TITLE").viewports.len(),
        1,
        "only its paper view"
    );
}

#[test]
fn viewports_are_dropped_or_refused_from_the_record_alone() {
    let m = listed(&drawing());
    let status = |name: &str| -> Vec<(String, Option<String>)> {
        layout(&m, name)
            .viewports
            .iter()
            .map(|v| (v.status.clone(), v.reason.clone()))
            .collect()
    };
    let s = |a: &str, b: Option<&str>| (a.to_owned(), b.map(str::to_owned));
    assert_eq!(
        status("SHEET_A"),
        [s("paper", None), s("drawn", None), s("drawn", None)]
    );
    assert_eq!(
        status("SHEET_B"),
        [
            s("paper", None),
            s("drawn", None),
            s("dropped", Some("off"))
        ]
    );
    assert_eq!(
        status("SHEET_C"),
        [
            s("paper", None),
            s("dropped", Some("off")),
            s("dropped", Some("hide-plot")),
            s("refused", Some("twist")),
            s("refused", Some("non-plan")),
            s("refused", Some("perspective")),
        ]
    );
}

// Finding 1 of the #1643 review: the handle is the key a layout keeps across revisions.
#[test]
fn every_layout_carries_the_handle_of_its_layout_object() {
    let doc = drawing();
    let m = listed(&doc);
    let mut seen = Vec::new();
    for l in &m.layouts {
        let handle = l
            .handle
            .as_deref()
            .unwrap_or_else(|| panic!("{} has a handle", l.name));
        let object = doc
            .objects
            .values()
            .find_map(|o| match o {
                ObjectType::Layout(o) if o.name == l.name => Some(o.handle.value()),
                _ => None,
            })
            .expect("the layout object exists");
        assert_eq!(handle, format!("{object:X}"), "{}", l.name);
        assert!(!seen.contains(&handle), "{} repeats a handle", l.name);
        seen.push(handle);
    }
}

// Finding 4: the manifest reports the sheet the conversion writes, so a quarter turn swaps the sides.
#[test]
fn paper_size_and_orientation_follow_the_plot_rotation() {
    let m = listed(&drawing());
    let paper = |name: &str| {
        let p = layout(&m, name).paper.as_ref().expect("a sheet has paper");
        (p.width_mm, p.height_mm, p.orientation.clone(), p.rotation)
    };
    assert_eq!(paper("SHEET_A"), (420.0, 297.0, "landscape".into(), 0));
    assert_eq!(paper("SHEET_B"), (210.0, 297.0, "portrait".into(), 0));
    assert_eq!(paper("ROTATED"), (297.0, 210.0, "landscape".into(), 90));
    assert_eq!(paper("ROT3"), (297.0, 420.0, "portrait".into(), 270));
}

#[test]
fn layer_states_are_read_with_their_masks_and_missing_layers() {
    let m = listed(&drawing());
    let names: Vec<&str> = m.layer_states.iter().map(|s| s.name.as_str()).collect();
    for want in ["Full", "Floor Plan", "Old Demo", "Viewport"] {
        assert!(names.contains(&want), "{want} in {names:?}");
    }
    let state = |name: &str| {
        m.layer_states
            .iter()
            .find(|s| s.name == name)
            .expect("listed")
    };

    let full = state("Full");
    assert!(full.mask.on && full.mask.frozen && full.mask.plot && !full.mask.viewport);
    assert!(
        full.missing_layers.is_empty() && full.unlisted_layers.is_empty(),
        "{:?} {:?}",
        full.missing_layers,
        full.unlisted_layers
    );
    assert!(full.layers.iter().any(|l| l.name == "OFF" && !l.on));

    let floor = state("Floor Plan");
    assert!(floor.mask.on && floor.mask.frozen && !floor.mask.plot);
    assert!(floor.layers.iter().any(|l| l.name == "OFF" && l.on));
    assert_eq!(
        floor.unlisted_layers,
        ["LOCKED"],
        "LOCKED was left out of it"
    );

    let old = state("Old Demo");
    // The reader names a state row by looking its handle up, so a deleted layer comes back with
    // no name and only its handle.
    assert!(old.missing_layers.is_empty(), "{:?}", old.missing_layers);
    assert_eq!(old.missing_layer_handles, ["9999"]);
    assert!(old.layers.iter().all(|l| !l.name.is_empty()));

    let viewport = state("Viewport");
    assert!(viewport.mask.viewport && viewport.hidden);
}

// Finding 2: a layer only paper space uses has nothing a viewport can hide.
#[test]
fn a_paper_space_only_layer_is_visible_whatever_a_viewport_freezes() {
    let m = listed(&drawing());
    let resolved = resolve_visibility(&m, &choices(r#"{"schema":1,"mode":"layouts"}"#));
    let of = |n: &str| resolved.iter().find(|r| r.name == n).expect("resolved");

    let only = of("PAPER-ONLY");
    assert_eq!(only.plots, PlotsVisibility::Visible, "no viewport chip");
    assert!(only.per_layout.is_empty(), "{:?}", only.per_layout);
    assert!(
        only.per_viewport
            .values()
            .all(|v| *v == Visibility::Visible),
        "{:?}",
        only.per_viewport
    );
    assert_eq!(only.plots_count.frozen_in, 0);

    let one = of("VPFROZEN-ONE");
    assert_eq!(one.plots, PlotsVisibility::Viewport);
    assert_eq!(one.plots_count.frozen_in, 1);
    assert_eq!(of("VPFROZEN-ALL").plots_count.frozen_in, 4);
    assert_eq!(of("ON-THAW-PLOT").plots, PlotsVisibility::Visible);
    assert_eq!(of("NOPLOT").plots, PlotsVisibility::Hidden);
    assert_eq!(of("Defpoints").plots, PlotsVisibility::Hidden);
    assert_eq!(of("OFF").model, Visibility::Hidden);
    assert_eq!(
        of("LOCKED").model,
        Visibility::Visible,
        "lock changes nothing"
    );
}

#[test]
fn a_chosen_state_applies_under_its_mask_to_what_the_reader_gave() {
    let m = listed(&drawing());
    let of = |r: &[libviprs::pvec_convert::ResolvedLayer], n: &str| {
        r.iter().find(|r| r.name == n).expect("resolved").clone()
    };
    let resolved = resolve_visibility(
        &m,
        &choices(r#"{"schema":1,"mode":"layouts","layerState":"Floor Plan"}"#),
    );
    assert_eq!(
        of(&resolved, "OFF").model,
        Visibility::Visible,
        "the state turns it on"
    );
    assert_eq!(
        of(&resolved, "FROZEN").model,
        Visibility::Visible,
        "and thaws this one"
    );
    assert_eq!(
        of(&resolved, "NOPLOT").plots,
        PlotsVisibility::Hidden,
        "the mask leaves Plot out, so the file's no-plot stays"
    );

    let strict = resolve_visibility(
        &m,
        &choices(
            r#"{"schema":1,"mode":"layouts","layerState":"Floor Plan","turnOffLayersNotInState":true}"#,
        ),
    );
    assert_eq!(
        of(&strict, "LOCKED").model,
        Visibility::Hidden,
        "LOCKED is not in the state"
    );
}
