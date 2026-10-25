//! N4.6b (#1296): entity visibility, layer override and viewport entities cross the provider
//! boundary and feed the N4.6 filter, so nothing has to be handed in beside the entity.
//!
//! The provider here is hand built. It answers `entity_facts` and `viewports` the way a real
//! one would, and the tests read the filter's outcome, never the mock's own answers.

use std::collections::BTreeMap;

use libviprs::cad::ItemHandle;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityFacts, EntityScope, EntityStates, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, PolicyReason,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact,
    SpaceSelection, ViewProfile, ViewProfileError, ViewportEntity, ViewportFreezeMode,
    VisibilityFilter,
};

struct Mock {
    facts: Option<BTreeMap<ItemHandle, EntityFacts>>,
    vports: Option<Vec<ViewportEntity>>,
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
    fn entity_facts(
        &self,
        _scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        self.facts
            .clone()
            .ok_or(ProviderError::CannotReport(Facet::EntityVisibility))
    }
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        self.vports
            .clone()
            .ok_or(ProviderError::CannotReport(Facet::ViewportEntities))
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
        facts: Some(BTreeMap::new()),
        vports: Some(Vec::new()),
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

fn facts(visible: Option<bool>, over: Option<&str>, provider_visible: Option<bool>) -> EntityFacts {
    let f = |o: Option<bool>| o.map_or(SourceFact::Absent, SourceFact::Present);
    EntityFacts {
        visible: f(visible),
        layer_override: over.map_or(SourceFact::Absent, |s| SourceFact::Present(s.into())),
        provider_visible: f(provider_visible),
    }
}

fn run(p: &Mock) -> Result<libviprs::pvec::VisibilityOutcome, ViewProfileError> {
    VisibilityFilter::run_from_provider(
        p,
        &ViewProfile::default_profile(),
        ViewportFreezeMode::ReportOnly,
    )
}

#[test]
fn pvec_1296_the_three_facets_are_in_the_boundary_and_a_provider_must_claim_them() {
    for f in [
        Facet::EntityVisibility,
        Facet::LayerOverride,
        Facet::ViewportEntities,
    ] {
        assert!(Facet::ALL.contains(&f), "{f:?} is missing from Facet::ALL");
    }
    // A provider that declares nothing cannot report them.
    let caps = ProviderCaps::new();
    assert_eq!(
        caps.availability(Facet::EntityVisibility),
        Availability::CannotReport
    );
}

#[test]
fn pvec_1296_a_provider_with_no_override_says_cannot_report_instead_of_empty() {
    struct Bare;
    impl DwgProvider for Bare {
        fn probe(_: libviprs::cad::CadSource<'_>) -> Result<bool, ProviderError> {
            Ok(false)
        }
        fn open(_: libviprs::cad::CadSource<'_>) -> Result<Self, ProviderError> {
            Ok(Bare)
        }
        fn name(&self) -> &'static str {
            "bare"
        }
        fn caps(&self) -> &ProviderCaps {
            unreachable!()
        }
        fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
            Ok(vec![])
        }
        fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
            Ok(vec![])
        }
        fn entities<'a>(
            &'a self,
            _: &EntityScope,
        ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
            Err(ProviderError::NoSuchScope("none".into()))
        }
    }
    assert!(matches!(
        Bare.entity_facts(&EntityScope::Layout(LayoutId(0))),
        Err(ProviderError::CannotReport(Facet::EntityVisibility))
    ));
    assert!(matches!(
        Bare.viewports(),
        Err(ProviderError::CannotReport(Facet::ViewportEntities))
    ));
}

#[test]
fn pvec_1296_an_invisible_entity_is_dropped_from_what_the_provider_reports() {
    let mut p = provider(vec![line(1, "WALLS"), line(2, "WALLS")]);
    p.facts = Some(BTreeMap::from([(h(2), facts(Some(false), None, None))]));
    let out = run(&p).expect("runs");
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.ignored.len(), 1);
    assert_eq!(out.ignored[0].handle, Some(h(2)));
    assert_eq!(out.ignored[0].reason, PolicyReason::EntityInvisible);
}

#[test]
fn pvec_1296_a_layer_override_decides_the_layer_the_filter_uses() {
    // Stored on WALLS, overridden onto an off layer: dropped as LayerOff, on the override.
    let mut p = provider(vec![line(1, "WALLS")]);
    p.facts = Some(BTreeMap::from([(h(1), facts(None, Some("Hidden"), None))]));
    let out = run(&p).expect("runs");
    assert_eq!(out.ignored.len(), 1);
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerOff);
    assert_eq!(out.ignored[0].layer, "Hidden");
}

#[test]
fn pvec_1296_the_providers_own_answer_is_compared_and_never_trusted() {
    // The provider says the Frozen-layer line is visible. Policy drops it and records the clash.
    let mut p = provider(vec![line(1, "Frozen")]);
    p.facts = Some(BTreeMap::from([(h(1), facts(None, None, Some(true)))]));
    let out = run(&p).expect("runs");
    assert!(out.kept.is_empty());
    assert_eq!(out.disagreements.len(), 1);
    assert!(out.disagreements[0].provider_visible);
    assert!(!out.disagreements[0].policy_visible);
}

#[test]
fn pvec_1296_entities_without_facts_keep_the_default_state() {
    let p = provider(vec![line(1, "WALLS"), line(2, "Frozen")]);
    let out = run(&p).expect("runs");
    assert_eq!(out.kept.len(), 1);
    assert_eq!(out.ignored[0].reason, PolicyReason::LayerFrozen);
}

#[test]
fn pvec_1296_entity_states_are_built_from_the_provider() {
    let mut p = provider(vec![line(1, "WALLS")]);
    p.facts = Some(BTreeMap::from([(
        h(1),
        facts(Some(false), Some("Hidden"), Some(true)),
    )]));
    let st = EntityStates::from_provider(&p, &EntityScope::Layout(LayoutId(0))).expect("states");
    let s = st.get(Some(h(1)));
    assert!(!s.visible);
    assert_eq!(s.layer_override.as_deref(), Some("Hidden"));
    assert_eq!(s.provider_visible, Some(true));
    assert_eq!(s.layout, Some(LayoutId(0)));
}

#[test]
fn pvec_1296_a_not_reportable_fact_is_an_error_not_a_default() {
    let mut p = provider(vec![line(1, "WALLS")]);
    p.facts = Some(BTreeMap::from([(h(1), EntityFacts::NOT_REPORTABLE)]));
    let err = EntityStates::from_provider(&p, &EntityScope::Layout(LayoutId(0)))
        .expect_err("must not default");
    assert!(matches!(err, ProviderError::CannotReport(_)), "{err:?}");
}

#[test]
fn pvec_1296_viewports_of_the_selected_paper_layout_reach_the_outcome() {
    let mut p = provider(vec![]);
    p.by_layout[1].1 = vec![line(10, "WALLS")];
    p.vports = Some(vec![
        ViewportEntity {
            handle: h(50),
            layout: LayoutId(1),
            visible: true,
            clip: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
            frozen_layers: vec!["WALLS".into()],
        },
        ViewportEntity {
            handle: h(51),
            layout: LayoutId(2),
            visible: false,
            clip: vec![],
            frozen_layers: vec![],
        },
    ]);
    let mut profile = ViewProfile::default_profile();
    profile.space = SpaceSelection::PaperLayout("Sheet A".into());
    let out = VisibilityFilter::run_from_provider(&p, &profile, ViewportFreezeMode::ReportOnly)
        .expect("runs");
    assert_eq!(out.viewports.len(), 1);
    assert_eq!(out.viewports[0].handle, h(50));
    assert_eq!(out.viewports[0].frozen_layers, vec!["WALLS".to_string()]);
    assert_eq!(out.viewports[0].clip.len(), 4);
}

#[test]
fn pvec_1296_a_provider_that_cannot_report_viewports_fails_with_that_facet() {
    let mut p = provider(vec![line(1, "WALLS")]);
    p.vports = None;
    match run(&p) {
        Err(ViewProfileError::Provider(ProviderError::CannotReport(Facet::ViewportEntities))) => {}
        other => panic!("expected CannotReport(ViewportEntities), got {other:?}"),
    }
}

#[test]
fn pvec_1296_a_provider_that_cannot_report_entity_facts_fails_with_that_facet() {
    let mut p = provider(vec![line(1, "WALLS")]);
    p.facts = None;
    match run(&p) {
        Err(ViewProfileError::Provider(ProviderError::CannotReport(Facet::EntityVisibility))) => {}
        other => panic!("expected CannotReport(EntityVisibility), got {other:?}"),
    }
}
