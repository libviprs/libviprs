//! PDF optional content (issue #1637, C5.1): the manifest of the groups, the PDF truth table for
//! view and print, membership dictionaries, and the rewrite of the default configuration that
//! the vector conversion reads.
//!
//! The main fixture is `tests/fixtures/pvec/pdf-layers-1637.pdf`, authored as text. Page 1 draws
//! one 20 point square per group at x = 40, 70, 100, 130, 160, 190 (Walls, Hidden, View off,
//! Print off, Option A, Option B), three more at x = 220, 250, 280 under a membership dictionary
//! (AnyOn, AllOn, a visibility expression), a form at x = 310 marked with Walls, and one square at
//! x = 10 under no marking at all. Page 2 is a rotated page with nothing on it.

use libviprs::pvec::pdf_layers::{
    PdfLayerChoices, PdfLayerMode, PdfLayerOverride, PdfManifest, apply_pdf_layers, pdf_manifest,
};
use libviprs::pvec::pdf_vector::{
    PdfPathSegment, PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes,
};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

const FIXTURE: &[u8] = include_bytes!("fixtures/pvec/pdf-layers-1637.pdf");

fn manifest(bytes: &[u8]) -> PdfManifest {
    pdf_manifest(bytes).expect("the fixture reads")
}

fn layer<'a>(m: &'a PdfManifest, name: &str) -> &'a libviprs::pvec::pdf_layers::PdfLayer {
    m.layers
        .iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("no layer named {name}: {:?}", m.layers))
}

/// The x of the first corner of every painted path on page 1, in content order.
fn xs(bytes: &[u8]) -> Vec<i64> {
    let page: PdfVectorPage =
        extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).expect("page 1");
    page.entities
        .iter()
        .filter_map(|e| match e.segments.first() {
            Some(PdfPathSegment::MoveTo(p)) => Some(p[0].round() as i64),
            _ => None,
        })
        .collect()
}

fn choices(mode: PdfLayerMode, layers: &[(&str, PdfLayerOverride)]) -> PdfLayerChoices {
    PdfLayerChoices {
        mode,
        layers: layers.iter().map(|(n, o)| ((*n).to_owned(), *o)).collect(),
    }
}

#[test]
fn pdf_manifest_lists_ocgs_default_config_usage_order_rbgroups() {
    let m = manifest(FIXTURE);

    assert_eq!(m.pages.len(), 2);
    assert_eq!((m.pages[0].width, m.pages[0].height), (612.0, 792.0));
    assert_eq!(m.pages[0].rotation_degrees, 0);
    // Page 2 is 300 by 400 turned a quarter, so it displays 400 by 300.
    assert_eq!((m.pages[1].width, m.pages[1].height), (400.0, 300.0));
    assert_eq!(m.pages[1].rotation_degrees, 90);

    let names: Vec<&str> = m.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "Walls",
            "Hidden",
            "View off",
            "Print off",
            "Option A",
            "Option B"
        ]
    );
    assert_eq!(m.layers[0].id, 10);

    // /D /BaseState /ON with /OFF [Hidden, Option B].
    let on: Vec<bool> = m.layers.iter().map(|l| l.on).collect();
    assert_eq!(on, [true, false, true, true, true, false]);

    // Usage applies because /AS lists the event for the group.
    assert!(layer(&m, "View off").view_hidden && !layer(&m, "View off").print_hidden);
    assert!(layer(&m, "Print off").print_hidden && !layer(&m, "Print off").view_hidden);
    assert!(!layer(&m, "Walls").view_hidden && !layer(&m, "Walls").print_hidden);

    assert_eq!(layer(&m, "Walls").intent, ["View"]);

    // /Order: [[(Plan) Walls [(Details) View off]] Hidden [Print off] [(Options) Option A]].
    assert_eq!(layer(&m, "Walls").order, Some(vec!["Plan".to_owned()]));
    assert_eq!(
        layer(&m, "View off").order,
        Some(vec!["Plan".to_owned(), "Details".to_owned()])
    );
    assert_eq!(layer(&m, "Hidden").order, Some(vec![]));
    // An array with no label hangs under the group before it.
    assert_eq!(
        layer(&m, "Print off").order,
        Some(vec!["Hidden".to_owned()])
    );
    assert_eq!(
        layer(&m, "Option A").order,
        Some(vec!["Options".to_owned()])
    );
    // Not in /Order at all.
    assert_eq!(layer(&m, "Option B").order, None);

    // /RBGroups [[Option A, Option B]].
    assert_eq!(m.rb_groups, vec![vec![14, 15]]);
    assert_eq!(layer(&m, "Option A").rb_groups, vec![0]);
    assert_eq!(layer(&m, "Option B").rb_groups, vec![0]);
    assert!(layer(&m, "Walls").rb_groups.is_empty());

    // Direct naming only: a marked-content span, a form, an annotation.
    assert_eq!(layer(&m, "Walls").spans, 2);
    assert_eq!(layer(&m, "Print off").spans, 2);
    assert_eq!(layer(&m, "Hidden").spans, 1);
    assert_eq!(layer(&m, "Option B").spans, 1);

    assert_eq!(m.configs, ["Alternate A"]);
}

/// One group in a document of its own, so each row of the truth table is one document.
fn one_group(
    base: &str,
    in_on: bool,
    in_off: bool,
    usage: &[(&str, &str)],
    as_events: &[&str],
) -> Vec<u8> {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut ocg = dictionary! { "Type" => "OCG", "Name" => Object::string_literal("G") };
    if !usage.is_empty() {
        let mut u = Dictionary::new();
        for (event, key) in usage {
            u.set(*event, dictionary! { *key => "OFF" });
        }
        ocg.set("Usage", u);
    }
    let g = doc.add_object(ocg);
    let list = |yes: bool| -> Vec<Object> { if yes { vec![g.into()] } else { vec![] } };
    let mut d = dictionary! {
        "BaseState" => base,
        "ON" => list(in_on),
        "OFF" => list(in_off),
    };
    if !as_events.is_empty() {
        let entries: Vec<Object> = as_events
            .iter()
            .map(|e| {
                dictionary! { "Event" => *e, "Category" => vec![Object::Name(e.as_bytes().to_vec())], "OCGs" => vec![Object::from(g)] }
                    .into()
            })
            .collect();
        d.set("AS", entries);
    }
    let content = doc.add_object(Stream::new(Dictionary::new(), b"1 1 5 5 re f".to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1 }.into(),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog", "Pages" => pages_id,
        "OCProperties" => dictionary! { "OCGs" => vec![Object::from(g)], "D" => d },
    });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

#[test]
fn pdf_resolve_truth_table_view_and_print() {
    // (what the row says, document, view, print)
    let rows: Vec<(&str, Vec<u8>, bool, bool)> = vec![
        (
            "BaseState ON, in neither list",
            one_group("ON", false, false, &[], &[]),
            true,
            true,
        ),
        (
            "in /OFF",
            one_group("ON", false, true, &[], &[]),
            false,
            false,
        ),
        (
            "BaseState OFF, not in /ON",
            one_group("OFF", false, false, &[], &[]),
            false,
            false,
        ),
        (
            "in /ON, ViewState OFF",
            one_group(
                "OFF",
                true,
                false,
                &[("View", "ViewState")],
                &["View", "Print"],
            ),
            false,
            true,
        ),
        (
            "in /ON, PrintState OFF",
            one_group(
                "OFF",
                true,
                false,
                &[("Print", "PrintState")],
                &["View", "Print"],
            ),
            true,
            false,
        ),
        (
            "ViewState OFF and PrintState OFF",
            one_group(
                "ON",
                false,
                false,
                &[("View", "ViewState"), ("Print", "PrintState")],
                &["View", "Print"],
            ),
            false,
            false,
        ),
        (
            "/AS lists neither event, usage ignored",
            one_group(
                "ON",
                false,
                false,
                &[("View", "ViewState"), ("Print", "PrintState")],
                &[],
            ),
            true,
            true,
        ),
        (
            "/AS lists only Print, ViewState ignored",
            one_group(
                "ON",
                false,
                false,
                &[("View", "ViewState"), ("Print", "PrintState")],
                &["Print"],
            ),
            true,
            false,
        ),
    ];
    for (row, bytes, view, print) in rows {
        let m = manifest(&bytes);
        let g = &m.layers[0];
        assert_eq!((g.resolved.view, g.resolved.print), (view, print), "{row}");
    }

    // The fixture agrees: Walls, Hidden, View off, Print off, Option A, Option B.
    let m = manifest(FIXTURE);
    let view: Vec<bool> = m.layers.iter().map(|l| l.resolved.view).collect();
    let print: Vec<bool> = m.layers.iter().map(|l| l.resolved.print).collect();
    assert_eq!(view, [true, false, false, true, true, false]);
    assert_eq!(print, [true, false, true, false, true, false]);
}

#[test]
fn pdf_ocmd_any_on_all_on_ve() {
    let m = manifest(FIXTURE);
    assert_eq!(m.ocmds.len(), 3);
    let by_policy = |p: &str| m.ocmds.iter().find(|o| o.policy == p).unwrap();

    // AnyOn over {Walls on, Hidden off} shows, AllOn hides, and the expression
    // And(Walls, Not(Hidden)) shows.
    let any = by_policy("AnyOn");
    assert_eq!(any.groups, ["Walls", "Hidden"]);
    assert!(any.resolved.view && any.resolved.print);
    assert_eq!(any.spans, 1);
    let all = by_policy("AllOn");
    assert!(!all.resolved.view && !all.resolved.print);
    let ve = by_policy("VE");
    assert!(ve.resolved.view && ve.resolved.print);

    // Evaluated per span in the vector output: the squares at 220 and 280 draw, 250 does not.
    let drawn = xs(FIXTURE);
    assert!(drawn.contains(&220) && drawn.contains(&280));
    assert!(!drawn.contains(&250));
}

#[test]
fn pdf_apply_layers_rewrites_d_and_strips_as() {
    let out = apply_pdf_layers(
        FIXTURE,
        &choices(
            PdfLayerMode::View,
            &[
                ("Walls", PdfLayerOverride::Hidden),
                ("Hidden", PdfLayerOverride::Visible),
                ("View off", PdfLayerOverride::Visible),
            ],
        ),
    )
    .unwrap();

    // Everything else is byte for byte: the rewrite is appended.
    assert!(out.starts_with(FIXTURE));
    assert!(out.len() > FIXTURE.len());

    let doc = Document::load_mem(&out).unwrap();
    let props = doc
        .catalog()
        .unwrap()
        .get(b"OCProperties")
        .and_then(|o| doc.dereference(o).map(|(_, o)| o))
        .and_then(Object::as_dict)
        .unwrap();
    let d = props.get(b"D").and_then(Object::as_dict).unwrap();
    assert!(d.get(b"AS").is_err(), "/AS is stripped");
    assert_eq!(
        d.get(b"BaseState").and_then(Object::as_name).unwrap(),
        b"ON"
    );
    let off: Vec<u32> = d
        .get(b"OFF")
        .and_then(Object::as_array)
        .unwrap()
        .iter()
        .map(|o| o.as_reference().unwrap().0)
        .collect();
    // Walls (override), Print off is visible in view mode, Option B was already off, Hidden is
    // forced on. View off is forced on.
    assert!(off.contains(&10) && off.contains(&15));
    assert!(!off.contains(&11) && !off.contains(&12));

    let m = manifest(&out);
    let view: Vec<bool> = m.layers.iter().map(|l| l.resolved.view).collect();
    assert_eq!(view, [false, true, true, true, true, false]);
    // Usage cannot flip the answer back, in either mode.
    let print: Vec<bool> = m.layers.iter().map(|l| l.resolved.print).collect();
    assert_eq!(print, view);

    // Print mode takes PrintState instead of ViewState.
    let print_out = apply_pdf_layers(FIXTURE, &choices(PdfLayerMode::Print, &[])).unwrap();
    let m = manifest(&print_out);
    let view: Vec<bool> = m.layers.iter().map(|l| l.resolved.view).collect();
    assert_eq!(view, [true, false, true, false, true, false]);

    // An override inside a radio group turns the sibling off.
    let radio = apply_pdf_layers(
        FIXTURE,
        &choices(
            PdfLayerMode::View,
            &[("Option B", PdfLayerOverride::Visible)],
        ),
    )
    .unwrap();
    let m = manifest(&radio);
    assert!(layer(&m, "Option B").resolved.view);
    assert!(!layer(&m, "Option A").resolved.view);
}

#[test]
fn pdf_apply_layers_hides_spans_in_vector_output() {
    // As authored, in view mode: the unmarked square, Walls, Print off, Option A, the two
    // membership dictionaries that show, and the Walls form.
    assert_eq!(xs(FIXTURE), [10, 40, 130, 160, 220, 280, 310]);

    let out = apply_pdf_layers(
        FIXTURE,
        &choices(
            PdfLayerMode::View,
            &[
                ("Walls", PdfLayerOverride::Hidden),
                ("Hidden", PdfLayerOverride::Visible),
                ("View off", PdfLayerOverride::Visible),
            ],
        ),
    )
    .unwrap();
    // Walls goes, with the form that carries it; Hidden and View off come back; the expression
    // And(Walls, Not(Hidden)) is now false, and AnyOn over {off, on} still shows.
    assert_eq!(xs(&out), [10, 70, 100, 130, 160, 220]);

    // Print mode hides Print off and shows View off.
    let print = apply_pdf_layers(FIXTURE, &choices(PdfLayerMode::Print, &[])).unwrap();
    assert_eq!(xs(&print), [10, 40, 100, 160, 220, 280, 310]);
}

#[test]
fn pdf_without_ocproperties_has_no_layers() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), b"10 10 20 20 re f".to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 200.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1 }.into(),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();

    let m = manifest(&bytes);
    assert!(m.layers.is_empty() && m.ocmds.is_empty() && m.configs.is_empty());
    assert_eq!(m.pages.len(), 1);
    assert_eq!((m.pages[0].width, m.pages[0].height), (100.0, 200.0));

    // Nothing to rewrite, and it converts as it did.
    let out = apply_pdf_layers(&bytes, &choices(PdfLayerMode::Print, &[])).unwrap();
    assert_eq!(out, bytes);
    assert_eq!(xs(&out), [10]);
}

/// `/OCProperties` and `/D` are objects of their own in many exporters' files, and the group
/// carries a `ViewState` pdfium reads without asking `/AS`.
#[test]
fn pdf_apply_layers_rewrites_indirect_d_and_clears_usage_state() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let g = doc.add_object(dictionary! {
        "Type" => "OCG", "Name" => Object::string_literal("G"),
        "Usage" => dictionary! { "View" => dictionary! { "ViewState" => "OFF" } },
    });
    let d = doc.add_object(dictionary! {
        "BaseState" => "ON",
        "AS" => vec![Object::from(dictionary! {
            "Event" => "View", "Category" => vec![Object::Name(b"View".to_vec())],
            "OCGs" => vec![Object::from(g)],
        })],
    });
    let props = doc.add_object(dictionary! { "OCGs" => vec![Object::from(g)], "D" => d });
    let content = doc.add_object(Stream::new(Dictionary::new(), b"1 1 5 5 re f".to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1 }.into(),
    );
    let catalog = doc.add_object(
        dictionary! { "Type" => "Catalog", "Pages" => pages_id, "OCProperties" => props },
    );
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    assert!(!manifest(&bytes).layers[0].resolved.view);

    let out = apply_pdf_layers(
        &bytes,
        &choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Visible)]),
    )
    .unwrap();
    assert!(out.starts_with(&bytes));
    assert!(manifest(&out).layers[0].resolved.view);

    let re = Document::load_mem(&out).unwrap();
    let group = re.get_dictionary(g).unwrap();
    let usage = group.get(b"Usage").and_then(Object::as_dict).unwrap();
    let view = usage.get(b"View").and_then(Object::as_dict).unwrap();
    assert!(
        view.get(b"ViewState").is_err(),
        "pdfium would read it past /AS"
    );
    assert!(re.get_dictionary(d).unwrap().get(b"AS").is_err());
}

#[test]
fn pdf_apply_layers_clears_state_when_usage_and_view_are_indirect() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let view = doc.add_object(dictionary! { "ViewState" => "OFF" });
    let usage = doc.add_object(dictionary! { "View" => view });
    let g = doc.add_object(dictionary! {
        "Type" => "OCG", "Name" => Object::string_literal("G"), "Usage" => usage,
    });
    let d = doc.add_object(dictionary! {
        "BaseState" => "ON",
        "AS" => vec![Object::from(dictionary! {
            "Event" => "View", "Category" => vec![Object::Name(b"View".to_vec())],
            "OCGs" => vec![Object::from(g)],
        })],
    });
    let props = doc.add_object(dictionary! { "OCGs" => vec![Object::from(g)], "D" => d });
    let content = doc.add_object(Stream::new(Dictionary::new(), b"1 1 5 5 re f".to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1 }.into(),
    );
    let catalog = doc.add_object(
        dictionary! { "Type" => "Catalog", "Pages" => pages_id, "OCProperties" => props },
    );
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    assert!(!manifest(&bytes).layers[0].resolved.view);

    let out = apply_pdf_layers(
        &bytes,
        &choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Visible)]),
    )
    .unwrap();
    let re = Document::load_mem(&out).unwrap();
    assert!(!re.get_dictionary(view).unwrap().has(b"ViewState"));
}

#[cfg(feature = "serde")]
#[test]
fn manifest_and_choices_serialise_as_camel_case_json() {
    let m = manifest(FIXTURE);
    let v = serde_json::to_value(&m).unwrap();
    assert!(v["layers"][0].get("viewHidden").is_some());
    assert!(v["layers"][0]["resolved"].get("print").is_some());
    assert!(v.get("rbGroups").is_some());
    let c: PdfLayerChoices =
        serde_json::from_str(r#"{"mode":"print","layers":{"A":"hidden"}}"#).unwrap();
    assert_eq!(c.mode, PdfLayerMode::Print);
    assert_eq!(c.layers["A"], PdfLayerOverride::Hidden);
}

/// A one-page document with one group `G` (carrying `ocg_extra`) that marks a square at x = 10,
/// and an unmarked square at x = 60.
fn marked_square(ocg_extra: Dictionary, in_off: bool) -> (Vec<u8>, lopdf::ObjectId) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut ocg = dictionary! { "Type" => "OCG", "Name" => Object::string_literal("G") };
    for (k, v) in ocg_extra.iter() {
        ocg.set(k.clone(), v.clone());
    }
    let g = doc.add_object(ocg);
    let off = if in_off {
        vec![Object::from(g)]
    } else {
        vec![]
    };
    let content = doc.add_object(Stream::new(
        Dictionary::new(),
        b"/OC /P1 BDC 10 10 20 20 re f EMC 60 10 20 20 re f".to_vec(),
    ));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => content,
        "Resources" => dictionary! { "Properties" => dictionary! { "P1" => g } },
    });
    doc.objects.insert(
        pages_id,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1 }.into(),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog", "Pages" => pages_id,
        "OCProperties" => dictionary! {
            "OCGs" => vec![Object::from(g)],
            "D" => dictionary! { "BaseState" => "ON", "OFF" => off },
        },
    });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    (out, g)
}

#[test]
fn a_design_intent_group_is_shown_until_the_user_hides_it() {
    let design = dictionary! { "Intent" => "Design" };
    let (bytes, _) = marked_square(design.clone(), true);
    // Listed OFF, but its intent is not View, so a viewer ignores the state.
    assert_eq!(xs(&bytes), [10, 60]);

    // The user's word wins in both paths: hiding it hides the span and the manifest agrees.
    let hidden = choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Hidden)]);
    let out = apply_pdf_layers(&bytes, &hidden).unwrap();
    assert_eq!(xs(&out), [60]);
    let m = manifest(&out);
    assert!(!m.layers[0].resolved.view);
    assert_eq!(m.layers[0].intent, ["View"]);
}

#[test]
fn showing_a_design_intent_group_keeps_it_shown_and_leaves_other_groups_alone() {
    let (bytes, _) = marked_square(dictionary! { "Intent" => "Design" }, true);
    let shown = choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Visible)]);
    let out = apply_pdf_layers(&bytes, &shown).unwrap();
    assert_eq!(xs(&out), [10, 60]);
    assert!(manifest(&out).layers[0].resolved.view);

    // A group that is not overridden keeps the intent it had.
    let untouched = apply_pdf_layers(&bytes, &choices(PdfLayerMode::View, &[])).unwrap();
    assert_eq!(manifest(&untouched).layers[0].intent, ["Design"]);
}

#[test]
fn a_state_key_that_says_on_is_dropped_too() {
    let usage = dictionary! { "View" => dictionary! { "ViewState" => "ON" } };
    let (bytes, g) = marked_square(dictionary! { "Usage" => usage }, false);
    assert_eq!(xs(&bytes), [10, 60]);

    let hidden = choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Hidden)]);
    let out = apply_pdf_layers(&bytes, &hidden).unwrap();
    assert_eq!(xs(&out), [60]);
    // pdfium reads the group's own ViewState before it reads /OFF, so it must be gone.
    let re = Document::load_mem(&out).unwrap();
    let view = re
        .get_dictionary(g)
        .and_then(|d| d.get(b"Usage"))
        .and_then(Object::as_dict)
        .and_then(|u| u.get(b"View"))
        .and_then(Object::as_dict)
        .unwrap();
    assert!(!view.has(b"ViewState"));
}

/// The check the strip exists for: the raster pdfium draws agrees with the vector path. Needs
/// libpdfium (feature `pdfium`, `PDFIUM_PATH` or a system library); without one it says so and
/// passes, like the other pdfium cells here.
#[cfg(feature = "pdfium")]
#[test]
#[cfg_attr(miri, ignore)]
fn pdfium_draws_what_the_rewritten_layers_say() {
    let usage = dictionary! { "View" => dictionary! { "ViewState" => "ON" } };
    let (bytes, _) = marked_square(dictionary! { "Usage" => usage }, false);
    let out = apply_pdf_layers(
        &bytes,
        &choices(PdfLayerMode::View, &[("G", PdfLayerOverride::Hidden)]),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let render = |name: &str, pdf: &[u8]| {
        let path = dir.path().join(name);
        std::fs::write(&path, pdf).unwrap();
        libviprs::render_page_pdfium(&path, 1, 72)
    };
    let before = match render("before.pdf", &bytes) {
        Ok(r) => r,
        Err(libviprs::PdfError::Pdfium(e)) => {
            eprintln!("skipped: no libpdfium ({e})");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let after = render("after.pdf", &out).unwrap();
    // The page is 100 points square at 72 dpi; the squares sit at x 10 and x 60, y 10 to 30
    // from the bottom, so their centres are at (20, 80) and (70, 80) from the top.
    let dark = |r: &libviprs::Raster, x: u32, y: u32| {
        let bpp = r.data().len() / (r.width() * r.height()) as usize;
        let i = (y * r.width() + x) as usize * bpp;
        r.data()[i] < 128
    };
    assert!(dark(&before, 20, 80) && dark(&before, 70, 80));
    assert!(!dark(&after, 20, 80), "pdfium still draws the hidden group");
    assert!(dark(&after, 70, 80));
}
