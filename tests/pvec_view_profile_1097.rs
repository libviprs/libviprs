//! N3.3 (#1097): the view-profile API.
//!
//! The drawing is built by hand through the provider boundary types, so nothing here needs a
//! CAD crate. It is the `vc_layers` construction of the N1.4 spike (#1043), whose worked example
//! gives the expected numbers: 3 visible leaves and 6 excluded entities under the v1 rule.
//! One test per acceptance criterion, named with the issue number.

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, Exclusion, Facet,
    FacetSource, InsertLayerRule, InstancePath, LayerState, LayerStateRule, LayoutId, LayoutKind,
    LayoutState, ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry,
    SourceTransform, SpaceSelection, ViewProfile, ViewProfileError,
};

// ---- a hand-built provider ------------------------------------------------------------

struct Mock {
    caps: ProviderCaps,
    layouts: Vec<LayoutState>,
    layers: Vec<LayerState>,
    model: Vec<RawEntity>,
    blocks: Vec<(String, Vec<RawEntity>)>,
}

struct VecCursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for VecCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: libviprs::cad::CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: libviprs::cad::CadSource<'_>) -> Result<Self, ProviderError> {
        Err(ProviderError::Open {
            provider: "mock",
            detail: "hand built".into(),
        })
    }
    fn name(&self) -> &'static str {
        "mock"
    }
    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(self.layouts.clone())
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        if self.caps.availability(Facet::LayerState) == Availability::CannotReport {
            return Err(ProviderError::CannotReport(Facet::LayerState));
        }
        Ok(self.layers.clone())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => self.model.clone(),
            EntityScope::Layout(id) => return Err(ProviderError::NoSuchScope(format!("{id:?}"))),
            EntityScope::BlockDefinition { name, reached_via } => {
                let (_, es) = self
                    .blocks
                    .iter()
                    .find(|(n, _)| n == name)
                    .ok_or_else(|| ProviderError::NoSuchScope(name.clone()))?;
                es.iter()
                    .cloned()
                    .map(|mut e| {
                        e.provenance.block_path = reached_via.clone();
                        e
                    })
                    .collect()
            }
            _ => return Err(ProviderError::NoSuchScope("?".into())),
        };
        Ok(Box::new(VecCursor(list.into_iter())))
    }
}

fn line(handle: u64, layer: &str) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [handle as f64, 0.0, 0.0],
            end: [handle as f64, 1.0, 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

fn insert_grid(handle: u64, layer: &str, block: &str, grid: ArrayGrid) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::IDENTITY,
            normal: [0.0, 0.0, 1.0],
            grid,
        }),
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "INSERT".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

fn insert(handle: u64, layer: &str, block: &str) -> RawEntity {
    insert_grid(handle, layer, block, ArrayGrid::SINGLE)
}

fn layer(name: &str, off: bool, frozen: bool) -> LayerState {
    LayerState {
        name: name.into(),
        handle: None,
        off,
        frozen,
        locked: false,
    }
}

fn layout(id: u32, name: &str, kind: LayoutKind) -> LayoutState {
    LayoutState {
        id: LayoutId(id),
        name: name.into(),
        kind,
        block_name: format!("*{name}"),
    }
}

/// `vc_layers`: lines m1..m4 on ON, OFF, FRZ and "0" (which is off), block VC_BLK holding b1 on
/// "0", b2 on ON and b3 on OFF, inserted on ON, OFF and FRZ.
fn vc_layers() -> Mock {
    Mock {
        caps: ProviderCaps::new()
            .with(
                Facet::LayerState,
                Availability::Reported,
                FacetSource::DecoderDirect,
            )
            .with(
                Facet::LayoutState,
                Availability::Reported,
                FacetSource::DecoderDirect,
            ),
        layouts: vec![
            layout(0, "Model", LayoutKind::Model),
            layout(1, "Layout1", LayoutKind::Paper),
            layout(2, "Sheet B", LayoutKind::Paper),
        ],
        layers: vec![
            layer("0", true, false),
            layer("VC_ON", false, false),
            layer("VC_OFF", true, false),
            layer("VC_FROZEN", false, true),
        ],
        model: vec![
            line(1, "VC_ON"),
            line(2, "VC_OFF"),
            line(3, "VC_FROZEN"),
            line(4, "0"),
            insert(10, "VC_ON", "VC_BLK"),
            insert(11, "VC_OFF", "VC_BLK"),
            insert(12, "VC_FROZEN", "VC_BLK"),
        ],
        blocks: vec![(
            "VC_BLK".into(),
            vec![line(21, "0"), line(22, "VC_ON"), line(23, "VC_OFF")],
        )],
    }
}

fn handles(o: &libviprs::pvec::ViewOutcome) -> Vec<u64> {
    o.included
        .iter()
        .map(|v| v.entity.provenance.handle.unwrap().get())
        .collect()
}

// ---- the default profile --------------------------------------------------------------

#[test]
fn n3_3_1097_default_profile_is_pinned_field_by_field() {
    use libviprs::pvec::{
        LayerVisibility, NestedInsertPolicy, ViewportClipping, ViewportFreezeRule, XrefPolicy,
    };
    let expected = ViewProfile {
        space: SpaceSelection::Model,
        viewport_clipping: ViewportClipping::Unsupported,
        layers: LayerVisibility {
            off: LayerStateRule::Exclude,
            frozen: LayerStateRule::Exclude,
            frozen_in_viewport: ViewportFreezeRule::IgnoreAndReport,
            insert_override: InsertLayerRule::InheritOnLayerZero,
        },
        nested_inserts: NestedInsertPolicy { max_depth: 64 },
        xrefs: XrefPolicy::ReportExternal,
        limits: libviprs::pvec::SourceLimits::default(),
    };
    assert_eq!(ViewProfile::default_profile(), expected);
}

#[test]
fn n3_3_1097_default_profile_prints_every_knob_with_its_n1_4_row() {
    let text = ViewProfile::default_profile().to_string();
    for needle in [
        "space = model",
        "layers.off = exclude",
        "layers.frozen = exclude",
        "layers.frozen_in_viewport = ignore-and-report",
        "layers.insert_override = inherit-on-layer-zero",
        "nested_inserts.max_depth = 64",
        "xrefs = report-external",
        "viewport_clipping = unsupported",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n3_3_1097_every_knob_names_a_row_of_the_n1_4_decision_table() {
    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/spikes/view-context.md"),
    )
    .expect("spike doc");
    let knobs = ViewProfile::default_profile().knobs();
    assert_eq!(
        knobs.len(),
        8,
        "a knob was added or removed: justify it against N1.4"
    );
    let mut fields: Vec<_> = knobs.iter().map(|k| k.field).collect();
    fields.sort_unstable();
    fields.dedup();
    assert_eq!(fields.len(), 8, "duplicate field path");
    for k in knobs {
        assert!(
            doc.contains(&format!("| `{}` |", k.row)),
            "{} names row {:?}, which is not in the decision table",
            k.field,
            k.row
        );
    }
}

// ---- space and layout selection --------------------------------------------------------

#[test]
fn n3_3_1097_absent_layout_is_a_typed_error_naming_the_layouts_that_exist() {
    let mut p = ViewProfile::default_profile();
    p.space = SpaceSelection::PaperLayout("Nope".into());
    let err = p.apply(&vc_layers()).unwrap_err();
    assert_eq!(
        err,
        ViewProfileError::LayoutNotFound {
            requested: "Nope".into(),
            available: vec!["Layout1".into(), "Sheet B".into()],
        }
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Nope") && msg.contains("Layout1") && msg.contains("Sheet B"),
        "{msg}"
    );
}

#[test]
fn n3_3_1097_present_paper_layout_is_refused_not_merged_or_emptied() {
    let mut p = ViewProfile::default_profile();
    p.space = SpaceSelection::PaperLayout("layout1".into());
    let err = p.apply(&vc_layers()).unwrap_err();
    assert_eq!(
        err,
        ViewProfileError::PaperSpaceUnsupported {
            layout: "Layout1".into()
        }
    );
    assert!(err.to_string().contains("508"), "{err}");
}

#[test]
fn n3_3_1097_a_drawing_without_model_space_is_a_typed_error() {
    let mut m = vc_layers();
    m.layouts.retain(|l| l.kind == LayoutKind::Paper);
    let err = ViewProfile::default_profile().apply(&m).unwrap_err();
    assert!(
        matches!(err, ViewProfileError::NoModelSpace { .. }),
        "{err:?}"
    );
}

// ---- applying the rule ----------------------------------------------------------------

#[test]
fn n3_3_1097_default_profile_reproduces_the_n1_4_worked_example() {
    let o = ViewProfile::default_profile().apply(&vc_layers()).unwrap();
    // m1, then b1 (layer "0" takes VC_ON from its INSERT) and b2 from the first INSERT.
    assert_eq!(handles(&o), vec![1, 21, 22]);
    assert_eq!(o.included[1].effective_layer, "VC_ON");
    assert_eq!(o.report.included, 3);
    // m2, m3, m4, the two excluded INSERTs, and b3 inside the visible one.
    assert_eq!(o.excluded.len(), 6);
    assert_eq!(o.report.excluded_by_field["layers.off"], 4);
    assert_eq!(o.report.excluded_by_field["layers.frozen"], 2);
    assert_eq!(o.report.excluded_by_layer["VC_OFF"], 3);
    assert_eq!(o.report.excluded_by_layer["VC_FROZEN"], 2);
    assert_eq!(o.report.excluded_by_layer["0"], 1);
    assert_eq!(o.report.space_name, "Model");
}

#[test]
fn n3_3_1097_an_excluded_entity_names_the_profile_field_that_excluded_it() {
    let o = ViewProfile::default_profile().apply(&vc_layers()).unwrap();
    let m3 = o
        .excluded
        .iter()
        .find(|e| e.handle == Some(ItemHandle::new(3)))
        .unwrap();
    assert_eq!(m3.reason, Exclusion::LayerFrozen);
    assert_eq!(m3.reason.field(), "layers.frozen");
    assert_eq!(m3.reason.code(), "LAYER_HIDDEN");
    let i11 = o
        .excluded
        .iter()
        .find(|e| e.handle == Some(ItemHandle::new(11)))
        .unwrap();
    assert_eq!(i11.class_name, "INSERT");
    assert_eq!(i11.reason.field(), "layers.off");
}

#[cfg(feature = "pvec-normalized")]
#[test]
fn n3_3_1097_policy_exclusions_classify_as_ignored_by_policy() {
    use libviprs::pvec::FidelityClass;
    for r in [
        Exclusion::LayerOff,
        Exclusion::LayerFrozen,
        Exclusion::DepthLimit,
    ] {
        assert_eq!(r.fidelity(), FidelityClass::IgnoredByPolicy, "{r:?}");
    }
    // ADR pvec-unsupported-policy: an unresolved block is Unsupported, UNRESOLVED_BLOCK.
    assert_eq!(Exclusion::Unresolved.fidelity(), FidelityClass::Unsupported);
}

#[test]
fn n3_3_1097_two_profiles_differing_only_in_the_frozen_rule_give_different_counts() {
    let strict = ViewProfile::default_profile();
    let mut lax = strict.clone();
    lax.layers.frozen = LayerStateRule::Ignore;
    let a = strict.apply(&vc_layers()).unwrap();
    let b = lax.apply(&vc_layers()).unwrap();
    assert_eq!(a.report.included, 3);
    assert_eq!(b.report.included, 6);
    assert_ne!(a.report.included, b.report.included);
    assert_eq!(a.report.excluded_by_field.get("layers.frozen"), Some(&2));
    assert_eq!(b.report.excluded_by_field.get("layers.frozen"), None);
    assert_eq!(a.report.profile, strict);
    assert_eq!(b.report.profile, lax);
}

#[test]
fn n3_3_1097_layer_zero_inheritance_is_a_knob() {
    let mut p = ViewProfile::default_profile();
    p.layers.insert_override = InsertLayerRule::OwnLayerOnly;
    let o = p.apply(&vc_layers()).unwrap();
    // b1 keeps its own layer "0", which is off.
    assert_eq!(handles(&o), vec![1, 22]);
}

#[test]
fn n3_3_1097_unresolved_block_contributes_nothing_and_is_reported() {
    let mut m = vc_layers();
    m.model = vec![line(1, "VC_ON"), insert(30, "VC_ON", "OTHER.DWG|BLK")];
    let o = ViewProfile::default_profile().apply(&m).unwrap();
    assert_eq!(handles(&o), vec![1]);
    assert_eq!(o.excluded.len(), 1);
    assert_eq!(o.excluded[0].reason, Exclusion::Unresolved);
    assert_eq!(o.excluded[0].reason.code(), "UNRESOLVED_BLOCK");
    assert_eq!(o.excluded[0].reason.field(), "xrefs");
}

#[test]
fn n3_3_1097_a_block_that_contains_itself_stops_at_the_depth_limit() {
    let mut m = vc_layers();
    m.model = vec![insert(40, "VC_ON", "LOOP")];
    m.blocks = vec![(
        "LOOP".into(),
        vec![line(41, "VC_ON"), insert(42, "VC_ON", "LOOP")],
    )];
    let mut p = ViewProfile::default_profile();
    p.nested_inserts.max_depth = 3;
    let o = p.apply(&m).unwrap();
    assert_eq!(o.report.included, 3);
    assert_eq!(o.excluded.len(), 1);
    assert_eq!(o.excluded[0].reason, Exclusion::DepthLimit);
    assert_eq!(o.excluded[0].reason.field(), "nested_inserts.max_depth");
}

#[test]
fn n3_3_1097_a_minsert_multiplies_the_instances_of_its_leaves() {
    let mut m = vc_layers();
    let grid = ArrayGrid {
        rows: 2,
        columns: 3,
        row_spacing: 1.0,
        column_spacing: 1.0,
    };
    m.model = vec![insert_grid(50, "VC_ON", "VC_BLK", grid)];
    let o = ViewProfile::default_profile().apply(&m).unwrap();
    assert_eq!(handles(&o), vec![21, 22]);
    assert!(o.included.iter().all(|v| v.instances == 6));
}

#[test]
fn n3_3_1097_missing_layer_state_is_an_error_when_the_profile_needs_it() {
    let mut m = vc_layers();
    m.caps = ProviderCaps::new();
    let err = ViewProfile::default_profile().apply(&m).unwrap_err();
    assert_eq!(
        err,
        ViewProfileError::Provider(ProviderError::CannotReport(Facet::LayerState))
    );
    // A profile that acts on no layer flag does not need the table.
    let mut p = ViewProfile::default_profile();
    p.layers.off = LayerStateRule::Ignore;
    p.layers.frozen = LayerStateRule::Ignore;
    assert_eq!(p.apply(&m).unwrap().report.included, 13);
}

#[test]
fn n3_3_1097_apply_is_deterministic() {
    let a = ViewProfile::default_profile().apply(&vc_layers()).unwrap();
    let b = ViewProfile::default_profile().apply(&vc_layers()).unwrap();
    assert_eq!(handles(&a), handles(&b));
    assert_eq!(a.excluded, b.excluded);
    assert_eq!(a.report, b.report);
}
