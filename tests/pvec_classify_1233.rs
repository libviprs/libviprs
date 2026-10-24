//! N4.9 (#1233): wall and column classification from layer and block rules.
//!
//! The entities are built by hand through the provider boundary types, so nothing
//! here needs a CAD crate or a feature. The "fixture" is a small known-truth
//! drawing: every entity's intended class is written next to it.

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    ArrayGrid, BlockRef, ClassSource, ClassificationReport, ElementClass, ElementRules,
    InstancePath, Point3, ProvenanceStub, RawEntity, RawGeometry, RuleMatch, SourceTransform,
};

fn pt(x: f64, y: f64) -> Point3 {
    [x, y, 0.0]
}

fn line(handle: u64, layer: &str, path: InstancePath) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Line {
            start: pt(handle as f64, 0.0),
            end: pt(handle as f64, 10.0),
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: path,
            layer: layer.into(),
        },
    }
}

fn insert(handle: u64, layer: &str, block: &str) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::IDENTITY,
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
        }),
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "INSERT".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

fn in_blocks(names: &[&str]) -> InstancePath {
    let mut p = InstancePath::root();
    for (i, n) in names.iter().enumerate() {
        p = p.child(*n, ItemHandle::new(0x100 + i as u64));
    }
    p
}

/// (entity, truth)
fn fixture() -> Vec<(RawEntity, ElementClass)> {
    use ElementClass::*;
    let root = InstancePath::root;
    vec![
        (line(1, "A-WALL", root()), Wall),
        (line(2, "A-WALL-FULL", root()), Wall),
        (line(3, "a-wall-prht", root()), Wall),
        (line(4, "S-COLS", root()), Column),
        (line(5, "S-COL-CONC", root()), Column),
        (line(6, "A-DOOR", root()), Unclassified),
        (line(7, "A-ANNO-TEXT", root()), Unclassified),
        (line(8, "0", root()), Unclassified),
        (line(9, "A-WALLET-NOT", root()), Unclassified),
        (insert(10, "0", "COLUMN_RC"), Column),
        (insert(11, "A-FURN", "CHAIR"), Unclassified),
    ]
}

fn classes(rules: &ElementRules, es: &[RawEntity]) -> Vec<ElementClass> {
    es.iter().map(|e| rules.classify(e).class).collect()
}

#[test]
fn n49_every_wall_and_column_is_classified_and_nothing_else() {
    let fx = fixture();
    let es: Vec<RawEntity> = fx.iter().map(|(e, _)| e.clone()).collect();
    let truth: Vec<ElementClass> = fx.iter().map(|(_, t)| *t).collect();
    assert_eq!(classes(&ElementRules::default_profile(), &es), truth);
}

#[test]
fn n49_default_profile_is_documented_and_replaceable() {
    let d = ElementRules::default_profile();
    assert!(!d.rules().is_empty());
    // A caller replaces it wholesale: an empty profile classifies nothing.
    let es: Vec<RawEntity> = fixture().into_iter().map(|(e, _)| e).collect();
    assert!(
        classes(&ElementRules::new(), &es)
            .iter()
            .all(|c| *c == ElementClass::Unclassified)
    );
}

#[test]
fn n49_same_input_and_profile_give_identical_classes() {
    let es: Vec<RawEntity> = fixture().into_iter().map(|(e, _)| e).collect();
    let rules = ElementRules::default_profile();
    let a = rules.classify_all(&es);
    let b = rules.classify_all(&es);
    assert_eq!(a, b);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn n49_two_matching_rules_resolve_by_list_order() {
    let e = line(1, "A-COL-WALL", InstancePath::root());
    let wall_first = ElementRules::new()
        .with_rule(RuleMatch::Layer("A-*".into()), ElementClass::Wall)
        .with_rule(RuleMatch::Layer("*COL*".into()), ElementClass::Column);
    let col_first = ElementRules::new()
        .with_rule(RuleMatch::Layer("*COL*".into()), ElementClass::Column)
        .with_rule(RuleMatch::Layer("A-*".into()), ElementClass::Wall);
    let a = wall_first.classify(&e);
    assert_eq!((a.class, a.rule_index), (ElementClass::Wall, Some(0)));
    let b = col_first.classify(&e);
    assert_eq!((b.class, b.rule_index), (ElementClass::Column, Some(0)));
}

#[test]
fn n49_block_rule_beats_layer_rule_whatever_the_list_order() {
    // A column block inserted on a wall layer is a column.
    let e = insert(1, "A-WALL", "COLUMN_RC");
    let layer_first = ElementRules::new()
        .with_rule(RuleMatch::Layer("A-WALL*".into()), ElementClass::Wall)
        .with_rule(RuleMatch::Block("COLUMN*".into()), ElementClass::Column);
    let c = layer_first.classify(&e);
    assert_eq!(c.class, ElementClass::Column);
    assert_eq!(c.source, Some(ClassSource::BlockRule));
}

#[test]
fn n49_source_tag_names_the_kind_of_rule() {
    let rules = ElementRules::default_profile();
    let by_layer = rules.classify(&line(1, "A-WALL", InstancePath::root()));
    assert_eq!(by_layer.source, Some(ClassSource::LayerRule));
    let by_block = rules.classify(&insert(2, "0", "COLUMN_RC"));
    assert_eq!(by_block.source, Some(ClassSource::BlockRule));
    let none = rules.classify(&line(3, "A-DOOR", InstancePath::root()));
    assert_eq!(none.source, None);
    assert_eq!(none.rule_index, None);
}

#[test]
fn n49_nested_inserts_keep_the_innermost_match_and_the_lineage() {
    let rules = ElementRules::default_profile();
    // A column block reached through a wall unit: innermost is the column.
    let leaf = line(7, "0", in_blocks(&["A-WALL-UNIT", "COLUMN_RC"]));
    let c = rules.classify(&leaf);
    assert_eq!(c.class, ElementClass::Column);
    assert_eq!(c.block_path, leaf.provenance.block_path);
    assert_eq!(c.block_path.depth(), 2);
    assert_eq!(c.handle, leaf.provenance.handle);
    // The other way round: a wall block inside a column block is a wall.
    let rules = ElementRules::new()
        .with_rule(RuleMatch::Block("COLUMN*".into()), ElementClass::Column)
        .with_rule(RuleMatch::Block("WALLUNIT*".into()), ElementClass::Wall);
    let leaf = line(8, "0", in_blocks(&["COLUMN_RC", "WALLUNIT_A"]));
    assert_eq!(rules.classify(&leaf).class, ElementClass::Wall);
    // Only the outer block matches: it still decides.
    let leaf = line(9, "0", in_blocks(&["COLUMN_RC", "PLAIN"]));
    assert_eq!(rules.classify(&leaf).class, ElementClass::Column);
    // An INSERT's own block name is the innermost step of its own entity.
    let nested_insert = {
        let mut e = insert(10, "0", "WALLUNIT_B");
        e.provenance.block_path = in_blocks(&["COLUMN_RC"]);
        e
    };
    assert_eq!(rules.classify(&nested_insert).class, ElementClass::Wall);
}

#[test]
fn n49_unmatched_entities_are_unclassified_counted_and_never_dropped() {
    let es: Vec<RawEntity> = fixture().into_iter().map(|(e, _)| e).collect();
    let (out, report) = ElementRules::default_profile().classify_all(&es);
    assert_eq!(out.len(), es.len());
    assert_eq!(
        report,
        ClassificationReport {
            walls: 3,
            columns: 3,
            unclassified: 5,
        }
    );
    for (c, e) in out.iter().zip(&es) {
        assert_eq!(c.handle, e.provenance.handle);
    }
}

#[test]
fn n49_matching_is_case_insensitive_and_globs_are_anchored() {
    let r = ElementRules::new()
        .with_rule(RuleMatch::Layer("a-wall".into()), ElementClass::Wall)
        .with_rule(RuleMatch::Layer("S-C?L".into()), ElementClass::Column);
    let c = |l: &str| r.classify(&line(1, l, InstancePath::root())).class;
    assert_eq!(c("A-WALL"), ElementClass::Wall);
    assert_eq!(c("A-WALL-X"), ElementClass::Unclassified);
    assert_eq!(c("XA-WALL"), ElementClass::Unclassified);
    assert_eq!(c("s-col"), ElementClass::Column);
    assert_eq!(c("S-COOL"), ElementClass::Unclassified);
}

#[test]
fn n49_changing_the_profile_changes_classes_and_never_geometry() {
    let es: Vec<RawEntity> = fixture().into_iter().map(|(e, _)| e).collect();
    let before: Vec<String> = es.iter().map(|e| format!("{:?}", e.geometry)).collect();
    let a = classes(&ElementRules::default_profile(), &es);
    let swapped = ElementRules::new()
        .with_rule(RuleMatch::Layer("A-WALL*".into()), ElementClass::Column)
        .with_rule(RuleMatch::Layer("S-COL*".into()), ElementClass::Wall);
    let b = classes(&swapped, &es);
    assert_ne!(a, b);
    let after: Vec<String> = es.iter().map(|e| format!("{:?}", e.geometry)).collect();
    assert_eq!(before, after);
}
