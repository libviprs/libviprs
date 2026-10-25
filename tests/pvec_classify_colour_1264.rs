//! #1264: colour and lineweight on the provider boundary, and the classifier rules
//! that read them. Follow-up to N4.9 (#1233), parent #1039.
//!
//! The first half builds entities by hand through the boundary types, so it needs no
//! feature. The second half writes a small drawing with known colours and lineweights
//! and reads it back through the acadrust provider (`--features dwg-acadrust`).

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    ClassSource, ElementClass, ElementRules, EntityStyle, Facet, InstancePath, Point3,
    ProvenanceStub, RawEntity, RawGeometry, RuleMatch, SourceColour, SourceFact, SourceLineweight,
};

fn pt(x: f64, y: f64) -> Point3 {
    [x, y, 0.0]
}

fn styled(
    handle: u64,
    layer: &str,
    colour: SourceFact<SourceColour>,
    lineweight: SourceFact<SourceLineweight>,
) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Line {
            start: pt(handle as f64, 0.0),
            end: pt(handle as f64, 10.0),
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
        style: EntityStyle { colour, lineweight },
    }
}

fn coloured(c: SourceFact<SourceColour>) -> RawEntity {
    styled(1, "MISC", c, SourceFact::Absent)
}

fn weighted(w: SourceFact<SourceLineweight>) -> RawEntity {
    styled(1, "MISC", SourceFact::Absent, w)
}

const RED: SourceColour = SourceColour::Aci(1);

#[test]
fn a_colour_rule_matches_the_exact_colour_and_says_so() {
    let rules = ElementRules::new().with_rule(RuleMatch::Colour(RED), ElementClass::Wall);
    let c = rules.classify(&coloured(SourceFact::Present(RED)));
    assert_eq!(c.class, ElementClass::Wall);
    assert_eq!(c.source, Some(ClassSource::ColourRule));
    assert_eq!(c.rule_index, Some(0));
    assert_eq!(ClassSource::ColourRule.tag(), "colour-rule");
}

#[test]
fn a_colour_rule_does_not_match_another_colour() {
    let rules = ElementRules::new().with_rule(RuleMatch::Colour(RED), ElementClass::Wall);
    let c = rules.classify(&coloured(SourceFact::Present(SourceColour::Aci(2))));
    assert_eq!(c.class, ElementClass::Unclassified);
    assert_eq!(c.source, None);
}

#[test]
fn an_aci_index_and_a_true_colour_that_look_alike_are_different_colours() {
    // ACI 1 is red, and so is true colour 255,0,0. The drawing wrote two different
    // things, and nothing here guesses that they mean the same.
    let rules = ElementRules::new().with_rule(RuleMatch::Colour(RED), ElementClass::Wall);
    let true_red = SourceColour::True { r: 255, g: 0, b: 0 };
    assert_eq!(
        rules
            .classify(&coloured(SourceFact::Present(true_red)))
            .class,
        ElementClass::Unclassified
    );
    let rules = ElementRules::new().with_rule(RuleMatch::Colour(true_red), ElementClass::Column);
    assert_eq!(
        rules
            .classify(&coloured(SourceFact::Present(true_red)))
            .class,
        ElementClass::Column
    );
    assert_eq!(
        rules.classify(&coloured(SourceFact::Present(RED))).class,
        ElementClass::Unclassified
    );
}

#[test]
fn byblock_absent_and_not_reportable_colours_never_match_a_colour_rule() {
    let rules = ElementRules::new()
        .with_rule(RuleMatch::Colour(RED), ElementClass::Wall)
        .with_rule(
            RuleMatch::Colour(SourceColour::ByBlock),
            ElementClass::Column,
        );
    for c in [
        SourceFact::Present(SourceColour::ByBlock),
        SourceFact::Absent,
        SourceFact::NotReportable,
    ] {
        let got = rules.classify(&coloured(c.clone()));
        assert_eq!(got.class, ElementClass::Unclassified, "{c:?}");
    }
}

#[test]
fn a_lineweight_rule_matches_the_exact_weight_and_says_so() {
    let rules = ElementRules::new().with_rule(RuleMatch::Lineweight(50), ElementClass::Wall);
    let c = rules.classify(&weighted(SourceFact::Present(
        SourceLineweight::Hundredths(50),
    )));
    assert_eq!(c.class, ElementClass::Wall);
    assert_eq!(c.source, Some(ClassSource::LineweightRule));
    assert_eq!(ClassSource::LineweightRule.tag(), "lineweight-rule");
    let c = rules.classify(&weighted(SourceFact::Present(
        SourceLineweight::Hundredths(35),
    )));
    assert_eq!(c.class, ElementClass::Unclassified);
}

#[test]
fn default_byblock_absent_and_not_reportable_weights_never_match_a_lineweight_rule() {
    let rules = ElementRules::new().with_rule(RuleMatch::Lineweight(0), ElementClass::Wall);
    for w in [
        SourceFact::Present(SourceLineweight::Default),
        SourceFact::Present(SourceLineweight::ByBlock),
        SourceFact::Absent,
        SourceFact::NotReportable,
    ] {
        let got = rules.classify(&weighted(w.clone()));
        assert_eq!(got.class, ElementClass::Unclassified, "{w:?}");
    }
}

#[test]
fn layer_rules_beat_colour_rules_and_colour_rules_beat_lineweight_rules() {
    // One entity every kind of rule matches, rules listed in the opposite order to
    // the precedence, so the list order cannot be what decides it.
    let rules = ElementRules::new()
        .with_rule(RuleMatch::Lineweight(50), ElementClass::Unclassified)
        .with_rule(RuleMatch::Lineweight(50), ElementClass::Column)
        .with_rule(RuleMatch::Colour(RED), ElementClass::Wall)
        .with_rule(RuleMatch::Layer("MISC".into()), ElementClass::Column);
    let e = styled(
        1,
        "MISC",
        SourceFact::Present(RED),
        SourceFact::Present(SourceLineweight::Hundredths(50)),
    );
    let c = rules.classify(&e);
    assert_eq!(c.class, ElementClass::Column);
    assert_eq!(c.source, Some(ClassSource::LayerRule));
    assert_eq!(c.rule_index, Some(3));

    let rules = ElementRules::new()
        .with_rule(RuleMatch::Lineweight(50), ElementClass::Column)
        .with_rule(RuleMatch::Colour(RED), ElementClass::Wall);
    let c = rules.classify(&e);
    assert_eq!(c.class, ElementClass::Wall);
    assert_eq!(c.source, Some(ClassSource::ColourRule));
    assert_eq!(c.rule_index, Some(1));

    let rules = ElementRules::new()
        .with_rule(RuleMatch::Lineweight(50), ElementClass::Column)
        .with_rule(RuleMatch::Colour(SourceColour::Aci(9)), ElementClass::Wall);
    let c = rules.classify(&e);
    assert_eq!(c.class, ElementClass::Column);
    assert_eq!(c.source, Some(ClassSource::LineweightRule));
    assert_eq!(c.rule_index, Some(0));
}

#[test]
fn the_first_matching_colour_rule_wins_within_its_kind() {
    let rules = ElementRules::new()
        .with_rule(
            RuleMatch::Colour(SourceColour::Aci(9)),
            ElementClass::Column,
        )
        .with_rule(RuleMatch::Colour(RED), ElementClass::Wall)
        .with_rule(RuleMatch::Colour(RED), ElementClass::Column);
    let c = rules.classify(&coloured(SourceFact::Present(RED)));
    assert_eq!((c.class, c.rule_index), (ElementClass::Wall, Some(1)));
}

#[test]
fn the_default_profile_is_unchanged_by_colour_and_lineweight() {
    // Adding the facets must not make anything classify that did not before.
    let e = styled(
        1,
        "MISC",
        SourceFact::Present(RED),
        SourceFact::Present(SourceLineweight::Hundredths(50)),
    );
    assert_eq!(
        ElementRules::default_profile().classify(&e).class,
        ElementClass::Unclassified
    );
}

#[test]
fn the_facets_are_named_and_a_hand_built_entity_reports_them_by_what_it_carries() {
    assert!(Facet::ALL.contains(&Facet::Colour));
    assert!(Facet::ALL.contains(&Facet::Lineweight));
    let e = styled(
        1,
        "MISC",
        SourceFact::Present(RED),
        SourceFact::NotReportable,
    );
    assert!(e.reported().contains(&Facet::Colour));
    assert!(!e.not_reportable().contains(&Facet::Colour));
    assert!(e.not_reportable().contains(&Facet::Lineweight));
    assert!(!e.reported().contains(&Facet::Lineweight));
    let none = EntityStyle::not_reportable();
    assert!(none.colour.is_not_reportable() && none.lineweight.is_not_reportable());
}

#[cfg(feature = "dwg-acadrust")]
mod provider {
    use acadrust::entities::Line;
    use acadrust::tables::Layer;
    use acadrust::types::{Color, LineWeight, Vector3};
    use acadrust::{CadDocument, DwgWriter, EntityType};
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{
        Availability, DwgProvider, ElementClass, ElementRules, EntityScope, Facet, LayoutKind,
        RawEntity, RawGeometry, RuleMatch, SourceColour, SourceFact, SourceLineweight,
    };

    /// (x of the line, layer, colour, lineweight). The x is how a line is found again.
    type Row = (f64, &'static str, Color, LineWeight);

    const ROWS: [Row; 7] = [
        (1.0, "L-RED", Color::ByLayer, LineWeight::ByLayer),
        (2.0, "L-RED", Color::Index(3), LineWeight::Value(25)),
        (
            3.0,
            "L-RED",
            Color::Rgb {
                r: 10,
                g: 20,
                b: 30,
            },
            LineWeight::Default,
        ),
        (4.0, "L-RED", Color::ByBlock, LineWeight::ByBlock),
        (5.0, "L-BLUE", Color::ByLayer, LineWeight::ByLayer),
        (6.0, "L-RED", Color::Index(1), LineWeight::Value(0)),
        (7.0, "L-RED", Color::Index(7), LineWeight::Value(211)),
    ];

    fn drawing() -> Vec<u8> {
        let mut doc = CadDocument::new();
        let mut red = Layer::with_color("L-RED", Color::Index(1));
        red.line_weight = LineWeight::Value(50);
        doc.layers.add(red).unwrap();
        let mut blue = Layer::with_color("L-BLUE", Color::Rgb { r: 0, g: 0, b: 255 });
        blue.line_weight = LineWeight::Default;
        doc.layers.add(blue).unwrap();
        for (x, layer, colour, weight) in ROWS {
            let mut line = Line::new();
            line.start = Vector3 { x, y: 0.0, z: 0.0 };
            line.end = Vector3 { x, y: 1.0, z: 0.0 };
            line.common.layer = layer.to_string();
            line.common.color = colour;
            line.common.line_weight = weight;
            doc.add_entity(EntityType::Line(line)).unwrap();
        }
        DwgWriter::write_to_vec(&doc).unwrap()
    }

    fn read() -> Vec<RawEntity> {
        let bytes = drawing();
        let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
        let model = p
            .layouts()
            .unwrap()
            .into_iter()
            .find(|l| l.kind == LayoutKind::Model)
            .unwrap();
        p.entities(&EntityScope::Layout(model.id))
            .unwrap()
            .collect_all()
            .unwrap()
    }

    fn at(all: &[RawEntity], x: f64) -> &RawEntity {
        all.iter()
            .find(|e| matches!(&e.geometry, RawGeometry::Line { start, .. } if start[0] == x))
            .unwrap_or_else(|| panic!("no line at x={x}"))
    }

    fn colour(e: &RawEntity) -> Option<SourceColour> {
        e.style.colour.present().copied()
    }

    fn weight(e: &RawEntity) -> Option<SourceLineweight> {
        e.style.lineweight.present().copied()
    }

    #[test]
    fn the_provider_claims_both_facets() {
        let bytes = drawing();
        let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
        for f in [Facet::Colour, Facet::Lineweight] {
            assert_eq!(p.caps().availability(f), Availability::Reported, "{f:?}");
        }
    }

    #[test]
    fn an_explicit_colour_and_weight_are_reported_as_stored() {
        let all = read();
        let e = at(&all, 2.0);
        assert_eq!(colour(e), Some(SourceColour::Aci(3)));
        assert_eq!(weight(e), Some(SourceLineweight::Hundredths(25)));
        let e = at(&all, 3.0);
        assert_eq!(
            colour(e),
            Some(SourceColour::True {
                r: 10,
                g: 20,
                b: 30
            })
        );
        assert_eq!(weight(e), Some(SourceLineweight::Default));
        let e = at(&all, 7.0);
        assert_eq!(colour(e), Some(SourceColour::Aci(7)));
        assert_eq!(weight(e), Some(SourceLineweight::Hundredths(211)));
    }

    #[test]
    fn a_zero_weight_is_a_weight_and_not_an_absent_one() {
        let all = read();
        assert_eq!(weight(at(&all, 6.0)), Some(SourceLineweight::Hundredths(0)));
    }

    #[test]
    fn bylayer_resolves_through_the_layer_table() {
        let all = read();
        let e = at(&all, 1.0);
        assert_eq!(colour(e), Some(SourceColour::Aci(1)));
        assert_eq!(weight(e), Some(SourceLineweight::Hundredths(50)));
        let e = at(&all, 5.0);
        assert_eq!(colour(e), Some(SourceColour::True { r: 0, g: 0, b: 255 }));
        assert_eq!(weight(e), Some(SourceLineweight::Default));
    }

    #[test]
    fn byblock_is_reported_as_byblock_and_not_resolved_here() {
        let all = read();
        let e = at(&all, 4.0);
        assert_eq!(colour(e), Some(SourceColour::ByBlock));
        assert_eq!(weight(e), Some(SourceLineweight::ByBlock));
    }

    #[test]
    fn no_entity_comes_back_not_reportable_from_a_provider_that_reads_the_facets() {
        for e in read() {
            assert!(
                !e.style.colour.is_not_reportable() && !e.style.lineweight.is_not_reportable(),
                "{:?}",
                e.provenance.handle
            );
            assert!(
                !matches!(e.style.colour, SourceFact::Absent),
                "colour absent"
            );
        }
    }

    #[test]
    fn rules_classify_a_real_drawing_by_the_colours_it_carries() {
        let all = read();
        let rules = ElementRules::new()
            .with_rule(RuleMatch::Colour(SourceColour::Aci(1)), ElementClass::Wall)
            .with_rule(RuleMatch::Lineweight(211), ElementClass::Column);
        let class = |x: f64| rules.classify(at(&all, x)).class;
        // Lines 1 and 6 are red (one by layer, one explicit), line 7 is the heavy one.
        assert_eq!(class(1.0), ElementClass::Wall);
        assert_eq!(class(6.0), ElementClass::Wall);
        assert_eq!(class(7.0), ElementClass::Column);
        assert_eq!(class(2.0), ElementClass::Unclassified);
        assert_eq!(class(4.0), ElementClass::Unclassified);
    }
}
