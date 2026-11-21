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
    // And(Walls, Not(Hidden)) is now false and so is AnyOn over {off, on}... which is true.
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
