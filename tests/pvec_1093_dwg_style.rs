//! Issue #1093 (N5.4b): lineweight, linetype and colour index come from the DWG provider and feed
//! the line-style component. Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use acadrust::entities::Line;
use acadrust::tables::{Layer, LineType, LineTypeElement};
use acadrust::types::{Color, LineWeight, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::style::{StrokeWidth, StyleReport, dwg_line_style, dwg_style_input};
use libviprs::pvec::{
    Availability, DwgProvider, EntityScope, Facet, LayoutKind, RawEntity, RawGeometry,
    SourceLinetype,
};

fn drawing() -> Vec<u8> {
    let mut doc = CadDocument::new();
    let mut dashed = LineType::new("DASHED");
    dashed.elements = vec![LineTypeElement::dash(0.5), LineTypeElement::space(0.25)];
    doc.line_types.add(dashed).unwrap();
    let mut layer = Layer::with_color("L-DASH", Color::Index(5));
    layer.line_type = "DASHED".into();
    doc.layers.add(layer).unwrap();
    // x = 1: every facet set on the entity. x = 2: all BYLAYER. x = 3: BYBLOCK linetype.
    for (x, colour, weight, linetype) in [
        (1.0, Color::Index(3), LineWeight::Value(25), "DASHED"),
        (2.0, Color::ByLayer, LineWeight::ByLayer, ""),
        (3.0, Color::ByLayer, LineWeight::ByLayer, "ByBlock"),
    ] {
        let mut line = Line::new();
        line.start = Vector3 { x, y: 0.0, z: 0.0 };
        line.end = Vector3 { x, y: 1.0, z: 0.0 };
        line.common.layer = "L-DASH".into();
        line.common.color = colour;
        line.common.line_weight = weight;
        line.common.linetype = linetype.into();
        line.common.linetype_scale = 2.0;
        doc.add_entity(EntityType::Line(line)).unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn read() -> (AcadrustProvider, Vec<RawEntity>) {
    let bytes = drawing();
    let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
    let model = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let all = p
        .entities(&EntityScope::Layout(model.id))
        .unwrap()
        .collect_all()
        .unwrap();
    (p, all)
}

fn at(all: &[RawEntity], x: f64) -> &RawEntity {
    all.iter()
        .find(|e| matches!(&e.geometry, RawGeometry::Line { start, .. } if start[0] == x))
        .unwrap()
}

#[test]
fn issue_1093_the_provider_claims_the_linetype_facet() {
    let (p, _) = read();
    assert_eq!(
        p.caps().availability(Facet::Linetype),
        Availability::Reported
    );
}

#[test]
fn issue_1093_a_linetype_is_read_as_its_elements_and_the_entity_scale() {
    let (_, all) = read();
    assert_eq!(
        at(&all, 1.0).style.linetype.present(),
        Some(&SourceLinetype::Pattern {
            elements: vec![0.5, -0.25],
            scale: 2.0
        })
    );
}

#[test]
fn issue_1093_bylayer_resolves_through_the_layer_and_byblock_is_left_alone() {
    let (_, all) = read();
    assert_eq!(
        at(&all, 2.0).style.linetype.present(),
        Some(&SourceLinetype::Pattern {
            elements: vec![0.5, -0.25],
            scale: 2.0
        })
    );
    assert_eq!(
        at(&all, 3.0).style.linetype.present(),
        Some(&SourceLinetype::ByBlock)
    );
}

#[test]
fn issue_1093_a_fixture_with_lineweight_linetype_and_colour_makes_a_component_and_counts_the_colour()
 {
    let (_, all) = read();
    let e = at(&all, 1.0);
    let input = dwg_style_input(&e.style, true);
    assert_eq!(input.colour_index, Some(3));
    let id =
        libviprs::pvec::order_entities("t", "1", &Default::default(), all.clone()).entities[0].id;
    let c = dwg_line_style(id, &input).unwrap();
    assert_eq!(c.width, Some(StrokeWidth::Width(0.25)));
    let dash = c.dash.expect("a dash pattern");
    assert_eq!(dash.array(), [1.0, 0.5]);
    let mut report = StyleReport::default();
    report.count_colour("LINE", u64::from(input.colour_index.is_some()));
    assert_eq!(report.colour_total(), 1);
}
