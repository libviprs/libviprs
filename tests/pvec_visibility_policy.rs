//! N4.6 (#1077): visibility and layout filtering as libviprs policy.
//!
//! The drawing is hand built through the provider boundary types. One test per acceptance
//! criterion plus the edges of the order the filter evaluates in, all named with the issue
//! number. A filter that trusts the provider's own visibility answer turns these red.

use std::collections::BTreeMap;

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityState, EntityStates, Facet,
    FacetSource, FidelityClass, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState,
    PolicyReason, ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry,
    SpaceSelection, ViewProfile, ViewProfileError, ViewportFreezeMode, ViewportInput,
    VisibilityFilter, select_layout,
};

struct Mock {
    caps: ProviderCaps,
    layouts: Vec<LayoutState>,
    layers: Vec<LayerState>,
    by_layout: Vec<(u32, Vec<RawEntity>)>,
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
        match scope {
            EntityScope::Layout(LayoutId(id)) => {
                let (_, es) = self
                    .by_layout
                    .iter()
                    .find(|(i, _)| i == id)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("{id}")))?;
                Ok(Box::new(VecCursor(es.clone().into_iter())))
            }
            _ => Err(ProviderError::NoSuchScope("?".into())),
        }
    }
}

fn h(n: u64) -> ItemHandle {
    ItemHandle::new(n)
}

fn line(handle: u64, layer: &str) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::not_reportable(),
        geometry: RawGeometry::Line {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 0.0, 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(h(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
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

fn layouts() -> Vec<LayoutState> {
    vec![
        layout(0, "Model", LayoutKind::Model),
        layout(1, "Sheet A", LayoutKind::Paper),
        layout(2, "Sheet B", LayoutKind::Paper),
    ]
}

fn layers() -> Vec<LayerState> {
    vec![
        layer("0", false, false),
        layer("WALLS", false, false),
        layer("Hidden", true, false),
        layer("Frozen", false, true),
        layer("Both", true, true),
    ]
}

fn provider(model: Vec<RawEntity>) -> Mock {
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
        layouts: layouts(),
        layers: layers(),
        by_layout: vec![(0, model), (1, Vec::new()), (2, Vec::new())],
    }
}

fn run(p: &Mock, states: &EntityStates) -> libviprs::pvec::VisibilityOutcome {
    VisibilityFilter::run(
        p,
        &ViewProfile::default_profile(),
        states,
        Vec::new(),
        ViewportFreezeMode::ReportOnly,
    )
    .expect("filter runs")
}

#[test]
fn pvec_1077_frozen_layer_entities_are_dropped_and_counted() {
    // The provider hands back entities on a frozen layer, as a decoder that ignores layer state
    // does. The output must not change because of it.
    let p = provider(vec![line(1, "WALLS"), line(2, "Frozen"), line(3, "Frozen")]);
    let out = run(&p, &EntityStates::new());
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.ignored.len(), 2);
    assert!(
        out.ignored
            .iter()
            .all(|r| r.reason == PolicyReason::LayerFrozen)
    );
    assert_eq!(out.counts_by_reason()[&PolicyReason::LayerFrozen], 2);
}

#[test]
fn pvec_1077_off_and_frozen_stay_distinct_reasons() {
    let p = provider(vec![line(1, "Hidden"), line(2, "Frozen"), line(3, "Both")]);
    let out = run(&p, &EntityStates::new());
    let reasons: Vec<_> = out.ignored.iter().map(|r| (r.handle, r.reason)).collect();
    assert_eq!(
        reasons,
        vec![
            (Some(h(1)), PolicyReason::LayerOff),
            (Some(h(2)), PolicyReason::LayerFrozen),
            // Off is checked before frozen, and the entity still has exactly one reason.
            (Some(h(3)), PolicyReason::LayerOff),
        ]
    );
}

#[test]
fn pvec_1077_every_dropped_entity_has_exactly_one_reason_and_counts_sum() {
    let model = vec![
        line(1, "WALLS"),
        line(2, "Hidden"),
        line(3, "Frozen"),
        line(4, "WALLS"),
        line(5, "WALLS"),
        line(6, "WALLS"),
        line(7, "Both"),
    ];
    let p = provider(model.clone());
    let states = EntityStates::new()
        .with(
            h(4),
            EntityState {
                visible: false,
                ..EntityState::default()
            },
        )
        .with(
            h(5),
            EntityState {
                layout: Some(LayoutId(1)),
                ..EntityState::default()
            },
        )
        .with(
            h(6),
            EntityState {
                outside_viewport: true,
                ..EntityState::default()
            },
        );
    let out = run(&p, &states);
    assert_eq!(out.kept.len() as u64 + out.dropped(), model.len() as u64);
    let counts = out.counts_by_reason();
    assert_eq!(counts.values().sum::<u64>(), out.dropped());
    assert_eq!(counts[&PolicyReason::LayerOff], 2);
    assert_eq!(counts[&PolicyReason::LayerFrozen], 1);
    assert_eq!(counts[&PolicyReason::EntityInvisible], 1);
    assert_eq!(counts[&PolicyReason::NotInSelectedLayout], 1);
    assert_eq!(counts[&PolicyReason::OutsideViewport], 1);
    // One row per handle, no entity twice.
    let mut seen: BTreeMap<u64, u32> = BTreeMap::new();
    for r in &out.ignored {
        *seen.entry(r.handle.unwrap().get()).or_insert(0) += 1;
    }
    assert!(seen.values().all(|n| *n == 1));
}

#[test]
fn pvec_1077_entity_flag_is_checked_after_layer_state() {
    let p = provider(vec![line(1, "Hidden")]);
    let states = EntityStates::new().with(
        h(1),
        EntityState {
            visible: false,
            ..EntityState::default()
        },
    );
    let out = run(&p, &states);
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerOff);
}

#[test]
fn pvec_1077_layer_override_replaces_the_provenance_layer() {
    let p = provider(vec![line(1, "WALLS"), line(2, "Hidden")]);
    let states = EntityStates::new()
        .with(
            h(1),
            EntityState {
                layer_override: Some("Frozen".into()),
                ..EntityState::default()
            },
        )
        .with(
            h(2),
            EntityState {
                layer_override: Some("walls".into()),
                ..EntityState::default()
            },
        );
    let out = run(&p, &states);
    assert_eq!(out.ignored.len(), 1);
    assert_eq!(out.ignored[0].handle, Some(h(1)));
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerFrozen);
    assert_eq!(out.ignored[0].layer, "Frozen");
    assert_eq!(out.kept[0].provenance.handle, Some(h(2)));
}

#[test]
fn pvec_1077_missing_layout_is_a_typed_error_naming_the_ones_that_exist() {
    let p = provider(Vec::new());
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout("Sheet Z".into());
    let err = VisibilityFilter::run(
        &p,
        &profile,
        &EntityStates::new(),
        Vec::new(),
        ViewportFreezeMode::ReportOnly,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ViewProfileError::LayoutNotFound {
            requested: "Sheet Z".into(),
            available: vec!["Sheet A".into(), "Sheet B".into()],
        }
    );
    assert!(err.to_string().contains("Sheet A"));
}

#[test]
fn pvec_1077_model_is_the_default_and_a_named_paper_layout_is_explicit() {
    let ls = layouts();
    assert_eq!(
        select_layout(&ls, &SpaceSelection::Model).unwrap().name,
        "Model"
    );
    let paper = select_layout(&ls, &SpaceSelection::PaperLayout("sheet b".into())).unwrap();
    assert_eq!((paper.id, paper.kind), (LayoutId(2), LayoutKind::Paper));
    assert!(matches!(
        select_layout(&ls[1..], &SpaceSelection::Model),
        Err(ViewProfileError::NoModelSpace { .. })
    ));
}

#[test]
fn pvec_1077_paper_layout_run_reads_that_layouts_entities() {
    let mut p = provider(vec![line(1, "WALLS")]);
    p.by_layout[1].1 = vec![line(10, "WALLS"), line(11, "Frozen")];
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout("Sheet A".into());
    let out = VisibilityFilter::run(
        &p,
        &profile,
        &EntityStates::new(),
        Vec::new(),
        ViewportFreezeMode::ReportOnly,
    )
    .unwrap();
    assert_eq!(out.layout.name, "Sheet A");
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.kept[0].provenance.handle, Some(h(10)));
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerFrozen);
}

#[test]
fn pvec_1077_provider_says_visible_on_frozen_layer_is_a_warning_and_policy_wins() {
    let p = provider(vec![line(1, "Frozen"), line(2, "WALLS")]);
    let states = EntityStates::new()
        .with(
            h(1),
            EntityState {
                provider_visible: Some(true),
                ..EntityState::default()
            },
        )
        .with(
            h(2),
            EntityState {
                provider_visible: Some(true),
                ..EntityState::default()
            },
        );
    let out = run(&p, &states);
    // Entity 1 is dropped whatever the provider said, entity 2 agrees and is silent.
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.kept[0].provenance.handle, Some(h(2)));
    assert_eq!(out.disagreements.len(), 1);
    let d = &out.disagreements[0];
    assert_eq!(d.handle, Some(h(1)));
    assert!(d.provider_visible);
    assert!(!d.policy_visible);
    assert_eq!(d.policy_reason, Some(PolicyReason::LayerFrozen));
}

#[test]
fn pvec_1077_provider_says_hidden_on_visible_layer_is_a_warning_and_entity_is_kept() {
    let p = provider(vec![line(1, "WALLS")]);
    let states = EntityStates::new().with(
        h(1),
        EntityState {
            provider_visible: Some(false),
            ..EntityState::default()
        },
    );
    let out = run(&p, &states);
    assert_eq!(out.kept.len(), 1);
    assert!(out.ignored.is_empty());
    assert_eq!(out.disagreements.len(), 1);
    assert!(!out.disagreements[0].provider_visible);
    assert!(out.disagreements[0].policy_visible);
    assert_eq!(out.disagreements[0].policy_reason, None);
}

fn viewport(handle: u64, layout: u32, frozen: &[&str]) -> ViewportInput {
    ViewportInput {
        handle: h(handle),
        layout: LayoutId(layout),
        visible: true,
        clip: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        frozen_layers: frozen.iter().map(|s| (*s).to_string()).collect(),
    }
}

#[test]
fn pvec_1077_viewport_freeze_is_reported_and_not_applied_by_default() {
    let mut p = provider(Vec::new());
    p.by_layout[1].1 = vec![line(10, "WALLS")];
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout("Sheet A".into());
    let states = EntityStates::new().with(
        h(10),
        EntityState {
            viewport: Some(h(100)),
            ..EntityState::default()
        },
    );
    let vps = vec![viewport(100, 1, &["walls"]), viewport(200, 2, &[])];
    let out =
        VisibilityFilter::run(&p, &profile, &states, vps, ViewportFreezeMode::ReportOnly).unwrap();
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.ignored_freezes.len(), 1);
    assert_eq!(out.ignored_freezes[0].viewport, h(100));
    // Only the selected layout's viewports are collected, with their boundary intact.
    assert_eq!(out.viewports.len(), 1);
    assert_eq!(out.viewports[0].clip.len(), 4);
    assert!(out.viewports[0].visible);
}

#[test]
fn pvec_1077_viewport_freeze_apply_mode_drops_as_viewport_frozen() {
    let mut p = provider(Vec::new());
    p.by_layout[1].1 = vec![line(10, "WALLS"), line(11, "0")];
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout("Sheet A".into());
    let vp = Some(h(100));
    let states = EntityStates::new()
        .with(
            h(10),
            EntityState {
                viewport: vp,
                ..EntityState::default()
            },
        )
        .with(
            h(11),
            EntityState {
                viewport: vp,
                ..EntityState::default()
            },
        );
    let out = VisibilityFilter::run(
        &p,
        &profile,
        &states,
        vec![viewport(100, 1, &["WALLS"])],
        ViewportFreezeMode::Apply,
    )
    .unwrap();
    assert_eq!(out.ignored.len(), 1);
    assert_eq!(out.ignored[0].reason, PolicyReason::ViewportFrozen);
    assert!(out.ignored_freezes.is_empty());
    assert_eq!(out.kept.len(), 1);
}

#[test]
fn pvec_1077_dropped_rows_classify_as_ignored_by_policy_with_the_field() {
    let p = provider(vec![line(1, "Hidden"), line(2, "Frozen")]);
    let out = run(&p, &EntityStates::new());
    let fields: Vec<_> = out
        .ignored
        .iter()
        .map(|r| {
            let c = r.classification();
            assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
            assert_eq!(r.reason.fidelity(), FidelityClass::IgnoredByPolicy);
            c.policy_field().unwrap().to_string()
        })
        .collect();
    assert_eq!(fields, vec!["layers.off", "layers.frozen"]);
    assert_eq!(PolicyReason::LayerOff.code(), "LAYER_HIDDEN");
    assert_eq!(PolicyReason::LayerFrozen.code(), "LAYER_HIDDEN");
}

#[test]
fn pvec_1077_profile_rule_ignore_keeps_the_layer_visible() {
    use libviprs::pvec::LayerStateRule;
    let p = provider(vec![line(1, "Hidden"), line(2, "Frozen")]);
    let mut profile = ViewProfile::default_profile();
    profile.layers.off = LayerStateRule::Ignore;
    let out = VisibilityFilter::run(
        &p,
        &profile,
        &EntityStates::new(),
        Vec::new(),
        ViewportFreezeMode::ReportOnly,
    )
    .unwrap();
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerFrozen);
}

#[test]
fn pvec_1077_provider_that_cannot_report_layers_is_an_error_not_an_unfiltered_drawing() {
    let mut p = provider(vec![line(1, "Frozen")]);
    p.caps = ProviderCaps::new();
    let err = VisibilityFilter::run(
        &p,
        &ViewProfile::default_profile(),
        &EntityStates::new(),
        Vec::new(),
        ViewportFreezeMode::ReportOnly,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ViewProfileError::Provider(ProviderError::CannotReport(Facet::LayerState))
    ));
}
