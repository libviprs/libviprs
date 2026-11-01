//! N2.6 (#1055): view, visibility and transform behaviour made executable.
//!
//! One named case per row of the two phase-1 documents: the decision table of
//! `docs/adr/pvec-view-profile.md` (N1.4) and the promotion and degeneracy matrix of
//! `docs/adr/pvec-transform-matrix.md` (N1.5). A policy change cannot land without a named test
//! moving here. Visibility is asserted against libviprs' own decision (`ViewProfile::apply`
//! re-derives it from the layer table), not against whatever the decoder chose to emit.
//!
//! The corpus fixtures are the pinned `pvec-gold-fixtures` ones. The "clipped viewport" case of
//! N1.4 has no fixture there (libviprs/pvec-gold-fixtures#96 asks for one), and v1 does not draw
//! paper space at all (V1-1), so that row is asserted as the refusal it is and nothing more.
//!
//! Needs `--features pvec-conformance`.

#![cfg(feature = "pvec-conformance")]

#[path = "common/pvec_gold_decode.rs"]
mod decode;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::cell::Cell;
use std::collections::BTreeMap;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, FlattenError, LineageStore};
use libviprs::pvec::provenance::ProvenanceStore;
use libviprs::pvec::transform::TransformReason;
use libviprs::pvec::{
    ArrayGrid, BlockRef, DwgProvider, EntityCursor, EntityScope, Exclusion, FidelityClass,
    InsertLayerRule, InstancePath, LayerState, LayerStateRule, LayoutId, LayoutKind, LayoutState,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceTransform,
    SpaceSelection, ViewProfile, ViewProfileError, ViewportClipping, ViewportFreezeRule,
    XrefPolicy,
};
use pvec_geometry::Curve;
use pvec_gold::Corpus;

fn model_view(id: &str) -> Result<libviprs::pvec::ViewOutcome, ViewProfileError> {
    let corpus = Corpus::open();
    let provider = decode::open(&corpus.get(id).dwg()).unwrap();
    ViewProfile::default_profile().apply(&provider)
}

// ------------------------------------------------------------------ N1.4: the decision table

#[test]
fn n2_6_1055_view_row_model_space_is_the_default_and_the_only_space_drawn() {
    let p = ViewProfile::default_profile();
    assert_eq!(p.space, SpaceSelection::Model);
    let out = model_view("layer-state-001").unwrap();
    assert!(
        out.report.space_name.to_lowercase().contains("model"),
        "{}",
        out.report.space_name
    );
}

#[test]
fn n2_6_1055_view_row_paper_space_is_refused_never_merged_and_never_skipped() {
    // A drawing that has a paper layout: the first corpus or provider fixture that does.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let candidates = [
        root.join("cad/sample_AC1018.dwg"),
        root.join("pvec/provider/vc_layers.dwg"),
        root.join("pvec/provider/vc_nested.dwg"),
    ];
    let mut found = None;
    for path in &candidates {
        let provider =
            libviprs::pvec::dwg_acadrust::AcadrustProvider::open(CadSource::Path(path)).unwrap();
        if let Some(l) = provider
            .layouts()
            .unwrap()
            .into_iter()
            .find(|l| l.kind == LayoutKind::Paper)
        {
            found = Some((provider, l.name));
            break;
        }
    }
    let (provider, layout) = found.expect("one of the fixtures has a paper layout");
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout(layout.clone());
    match profile.apply(&provider) {
        Err(ViewProfileError::PaperSpaceUnsupported { layout: l }) => assert_eq!(l, layout),
        other => panic!("paper space must be refused with a diagnostic, got {other:?}"),
    }
    profile.space = SpaceSelection::PaperLayout("NoSuchLayout".into());
    assert!(matches!(
        profile.apply(&provider),
        Err(ViewProfileError::LayoutNotFound { .. })
    ));
}

#[test]
fn n2_6_1055_view_row_viewport_clipping_exact_and_bounded_only_are_deferred_with_paper_space() {
    // No corpus fixture has a viewport (pvec-gold-fixtures#96), and v1 refuses paper space, so
    // both the exactly-clipped and the bounded-only case are one refusal: the knob says so.
    let p = ViewProfile::default_profile();
    assert_eq!(p.viewport_clipping, ViewportClipping::Unsupported);
    let knob = p
        .knobs()
        .into_iter()
        .find(|k| k.row == "viewport-clip-exact")
        .unwrap();
    assert_eq!(knob.value, "unsupported");
}

#[test]
fn n2_6_1055_view_row_layer_off_is_ignored_by_policy_reported_and_absent_from_the_output() {
    let out = model_view("layer-state-001").unwrap();
    let off = out
        .excluded
        .iter()
        .find(|e| e.layer == "OFF_L")
        .expect("OFF_L is in the report");
    assert_eq!(off.reason, Exclusion::LayerOff);
    assert_eq!(off.reason.fidelity(), FidelityClass::IgnoredByPolicy);
    assert_eq!(off.reason.code(), "LAYER_HIDDEN");
    // Both halves: reported as excluded with the reason, and not in what goes on to PVec.
    assert!(out.included.iter().all(|v| v.effective_layer != "OFF_L"));
    assert_eq!(out.report.excluded_by_layer.get("OFF_L"), Some(&1));
}

#[test]
fn n2_6_1055_view_row_layer_frozen_is_ignored_by_policy_reported_and_absent_from_the_output() {
    let out = model_view("layer-state-001").unwrap();
    let frozen = out
        .excluded
        .iter()
        .find(|e| e.layer == "FROZEN_L")
        .expect("FROZEN_L is in the report");
    assert_eq!(frozen.reason, Exclusion::LayerFrozen);
    assert_eq!(frozen.reason.fidelity(), FidelityClass::IgnoredByPolicy);
    assert!(out.included.iter().all(|v| v.effective_layer != "FROZEN_L"));
    assert_eq!(out.included.len(), 1);
    assert_eq!(out.included[0].effective_layer, "SHOWN_L");
    assert_eq!(out.report.included, 1);
    assert_eq!(out.report.excluded_by_field.get("layers.frozen"), Some(&1));
    assert_eq!(out.report.excluded_by_field.get("layers.off"), Some(&1));
}

#[test]
fn n2_6_1055_flipping_the_frozen_default_turns_the_frozen_case_red() {
    let corpus = Corpus::open();
    let provider = decode::open(&corpus.get("layer-state-001").dwg()).unwrap();
    let mut flipped = ViewProfile::default_profile();
    flipped.layers.frozen = LayerStateRule::Ignore;
    let out = flipped.apply(&provider).unwrap();
    // With the default the frozen entity is excluded (the test above); flipping the default
    // draws it, and the case above would fail on `included.len() == 1`.
    assert_eq!(out.included.len(), 2);
    assert!(out.included.iter().any(|v| v.effective_layer == "FROZEN_L"));
    assert_eq!(
        ViewProfile::default_profile().layers.frozen,
        LayerStateRule::Exclude
    );
    assert_eq!(
        ViewProfile::default_profile().layers.off,
        LayerStateRule::Exclude
    );
}

#[test]
fn n2_6_1055_view_row_layer_frozen_in_viewport_is_ignored_and_reported() {
    let p = ViewProfile::default_profile();
    assert_eq!(
        p.layers.frozen_in_viewport,
        ViewportFreezeRule::IgnoreAndReport
    );
    let knob = p
        .knobs()
        .into_iter()
        .find(|k| k.row == "layer-frozen-in-viewport")
        .unwrap();
    assert_eq!(knob.value, "ignore-and-report");
}

#[test]
fn n2_6_1055_view_row_insert_layer_override_inherits_on_layer_zero() {
    let p = ViewProfile::default_profile();
    assert_eq!(
        p.layers.insert_override,
        InsertLayerRule::InheritOnLayerZero
    );
    let knob = p
        .knobs()
        .into_iter()
        .find(|k| k.row == "insert-layer-override")
        .unwrap();
    assert_eq!(knob.value, "inherit-on-layer-zero");
}

#[test]
fn n2_6_1055_view_row_nested_inserts_are_followed_to_the_depth_limit() {
    assert_eq!(ViewProfile::default_profile().nested_inserts.max_depth, 64);
    let out = model_view("nested-blocks-001").unwrap();
    assert_eq!(
        out.included.len(),
        1,
        "the three-level chain ends in one leaf"
    );
    assert_eq!(out.included[0].entity.provenance.block_path.depth(), 3);
    // A limit below the nesting excludes the leaf and says why.
    let corpus = Corpus::open();
    let provider = decode::open(&corpus.get("nested-blocks-001").dwg()).unwrap();
    let mut shallow = ViewProfile::default_profile();
    shallow.nested_inserts.max_depth = 1;
    let out = shallow.apply(&provider).unwrap();
    assert!(out.included.is_empty());
    assert!(
        out.excluded
            .iter()
            .any(|e| e.reason == Exclusion::DepthLimit)
    );
}

#[test]
fn n2_6_1055_view_row_xref_contributes_nothing_and_is_reported_external() {
    assert_eq!(
        ViewProfile::default_profile().xrefs,
        XrefPolicy::ReportExternal
    );
    // The report half (UNRESOLVED_BLOCK, 105, field `xrefs`) is pinned on a mock provider by
    // `n3_3_1097_unresolved_block_contributes_nothing_and_is_reported`. On the corpus drawing the
    // acadrust provider hands over no INSERT for the xref at all today (the provider-side xref
    // facet is #1079), so nothing is drawn and nothing is reported. That is recorded here as it
    // is: the day the provider surfaces it, this assertion goes red and moves to the report.
    let out = model_view("xref-001").unwrap();
    assert!(out.included.is_empty(), "an xref draws nothing");
    assert!(
        out.excluded.is_empty(),
        "the provider surfaces no xref INSERT yet (#1079)"
    );
    assert_eq!(Exclusion::Unresolved.code(), "UNRESOLVED_BLOCK");
    assert_eq!(Exclusion::Unresolved.fidelity(), FidelityClass::Unsupported);
}

#[test]
fn n2_6_1055_every_row_of_the_decision_table_has_a_knob_and_every_knob_a_test_above() {
    let rows: Vec<&str> = ViewProfile::default_profile()
        .knobs()
        .iter()
        .map(|k| k.row)
        .collect();
    assert_eq!(
        rows,
        [
            "model-space",
            "viewport-clip-exact",
            "layer-off",
            "layer-frozen",
            "layer-frozen-in-viewport",
            "insert-layer-override",
            "nested-insert",
            "xref"
        ]
    );
    let me = include_str!("pvec_view_transform.rs");
    for needle in [
        "view_row_model_space",
        "view_row_paper_space",
        "view_row_viewport_clipping",
        "view_row_layer_off",
        "view_row_layer_frozen_is",
        "view_row_layer_frozen_in_viewport",
        "view_row_insert_layer_override",
        "view_row_nested_inserts",
        "view_row_xref",
    ] {
        assert!(
            me.contains(&format!("fn n2_6_1055_{needle}")),
            "no named test for {needle}"
        );
    }
}

// ----------------------------------------------------- N1.5: promotion and degeneracy matrix

/// The curve a leaf of `id` becomes in world space, as a name the matrix uses.
fn promoted_shape(id: &str) -> &'static str {
    let corpus = Corpus::open();
    let provider = decode::open(&corpus.get(id).dwg()).unwrap();
    let flat = decode::flatten(&provider, id).unwrap();
    let (_, curve) = decode::world(&flat.entities[0]).unwrap();
    match &curve {
        Curve::Trimmed(t) => match t.basis() {
            Curve::Line(_) => "trimmed line",
            Curve::Circle(_) => "trimmed circle",
            Curve::Ellipse(_) => "trimmed ellipse",
            _ => "trimmed other",
        },
        Curve::Circle(_) => "circle",
        Curve::Ellipse(_) => "ellipse",
        Curve::Polyline(_) => "polyline",
        _ => "other",
    }
}

#[test]
fn n2_6_1055_matrix_cell_translation_keeps_a_line_a_line() {
    assert_eq!(promoted_shape("translation-001"), "trimmed line");
}

#[test]
fn n2_6_1055_matrix_cell_rotation_keeps_a_line_a_line() {
    assert_eq!(promoted_shape("rotation-irrational-001"), "trimmed line");
}

#[test]
fn n2_6_1055_matrix_cell_uniform_scale_keeps_a_circle_a_circle() {
    assert_eq!(promoted_shape("uniform-scale-001"), "circle");
}

#[test]
fn n2_6_1055_matrix_cell_non_uniform_scale_turns_a_circle_into_an_ellipse() {
    assert_eq!(promoted_shape("nonuniform-scale-circle-001"), "ellipse");
    assert_eq!(promoted_shape("nonuniform-scale-rotated-001"), "ellipse");
}

#[test]
fn n2_6_1055_matrix_cell_reflection_keeps_an_arc_a_trimmed_circle() {
    assert_eq!(promoted_shape("reflection-001"), "trimmed circle");
}

#[test]
fn n2_6_1055_matrix_cell_shear_and_general_affine_keep_a_line_a_line() {
    assert_eq!(promoted_shape("shear-chain-001"), "trimmed line");
    assert_eq!(promoted_shape("general-affine-chain-001"), "trimmed line");
}

fn refusal(id: &str) -> TransformReason {
    let corpus = Corpus::open();
    let provider = decode::open(&corpus.get(id).dwg()).unwrap();
    match decode::flatten(&provider, id) {
        Err(FlattenError::Compose(e)) => e.reason().expect("a named reason"),
        other => panic!(
            "{id}: a singular map must be refused, got {:?}",
            other.map(|f| f.entities.len())
        ),
    }
}

#[test]
fn n2_6_1055_matrix_cell_singular_is_a_refusal_not_a_best_effort_curve() {
    assert_eq!(refusal("singular-001"), TransformReason::Singular);
    assert_eq!(TransformReason::Singular.code(), "TRANSFORM_SINGULAR");
}

#[test]
fn n2_6_1055_matrix_cell_near_singular_is_classified_by_the_conditioning_rule() {
    // P1.2's conditioning rule (NP-19) folds near-singular into the singular code; the corpus
    // truth calls it NEAR_SINGULAR (QUARANTINE). What both agree on is that no curve is emitted.
    assert_eq!(refusal("near-singular-001"), TransformReason::Singular);
    assert_eq!(TransformReason::Singular.rule(), "NP-19");
}

#[test]
fn n2_6_1055_every_cell_class_of_the_matrix_has_a_named_test() {
    let me = include_str!("pvec_view_transform.rs");
    for cell in [
        "translation",
        "rotation",
        "uniform_scale",
        "non_uniform_scale",
        "reflection",
        "shear_and_general_affine",
        "singular",
        "near_singular",
    ] {
        assert!(
            me.contains(&format!("fn n2_6_1055_matrix_cell_{cell}")),
            "no test for the {cell} cell"
        );
    }
}

// ------------------------------------------------- one block at two scales, lineage in provenance

fn h(raw: u64) -> ItemHandle {
    ItemHandle::new(raw)
}

fn stub(handle: u64, class: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(h(handle)),
        class_name: class.to_string(),
        block_path: InstancePath::root(),
        layer: "0".to_string(),
    }
}

fn insert(handle: u64, block: &str, m: [f64; 6]) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.to_string(),
            transform: SourceTransform::new(m).unwrap(),
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
        }),
        provenance: stub(handle, "INSERT"),
    }
}

struct Mock {
    model: Vec<RawEntity>,
    blocks: BTreeMap<String, Vec<RawEntity>>,
    caps: ProviderCaps,
    opened: Cell<u64>,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Err(ProviderError::Unrecognised {
            provider: "mock",
            detail: String::new(),
        })
    }
    fn name(&self) -> &'static str {
        "mock"
    }
    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(Vec::new())
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(Vec::new())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        self.opened.set(self.opened.get() + 1);
        let (items, path) = match scope {
            EntityScope::Layout(_) => (self.model.clone(), InstancePath::root()),
            EntityScope::BlockDefinition { name, reached_via } => (
                self.blocks
                    .get(name)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("block {name}")))?
                    .clone(),
                reached_via.clone(),
            ),
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        };
        let stamped: Vec<RawEntity> = items
            .into_iter()
            .map(|mut e| {
                e.provenance.block_path = path.clone();
                e
            })
            .collect();
        Ok(Box::new(Cursor(stamped.into_iter())))
    }
}

#[test]
fn n2_6_1055_one_block_instanced_at_two_scales_gives_two_distinct_results_with_lineage() {
    let line = RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 0.0, 0.0],
        },
        provenance: stub(0x40, "LINE"),
    };
    let mock = Mock {
        model: vec![
            insert(0x10, "OUTER", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            insert(0x11, "OUTER", [1.0, 0.0, 0.0, 1.0, 10.0, 0.0]),
        ],
        blocks: BTreeMap::from([
            (
                "OUTER".to_string(),
                vec![insert(0x20, "INNER", [2.0, 0.0, 0.0, 2.0, 0.0, 0.0])],
            ),
            ("INNER".to_string(), vec![line]),
        ]),
        caps: ProviderCaps::new(),
        opened: Cell::new(0),
    };
    // Two-level nesting, one block instanced twice with different outer offsets, and the inner
    // scale applied to both: the two world results differ, and the lineage names every level.
    let mut store = ProvenanceStore::new();
    let mut lineage = LineageStore::new();
    let out = BlockResolver::new(&mock, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(LayoutId(0)),
            "fixture.dwg",
            "Model",
            &mut store,
            &mut lineage,
        )
        .unwrap();
    assert_eq!(out.entities.len(), 2);
    let (a, b) = (&out.entities[0], &out.entities[1]);
    assert_ne!(a.matrix(), b.matrix(), "two instances, two composed maps");
    assert_eq!(a.map_point([1.0, 0.0, 0.0]), [2.0, 0.0]);
    assert_eq!(b.map_point([1.0, 0.0, 0.0]), [12.0, 0.0]);
    for leaf in [a, b] {
        let handle = leaf
            .lineage_handle()
            .expect("lineage is kept when a store is passed");
        let levels = lineage.get(handle).expect("the lineage record").levels();
        assert_eq!(levels.len(), 2, "outer and inner insert");
    }
}
