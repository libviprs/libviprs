//! Visibility and layout filtering as libviprs policy (N4.6, #1077).
//!
//! Which entities are visible is a question two systems answer differently: the drawing has
//! layer and entity state, and the provider has its own idea of what to hand back. Panel
//! finding 3 of #1039 says the answer belongs on this side of the boundary, so
//! [`VisibilityFilter`] re-derives it from the state it was given and records every decision.
//!
//! # What the filter reads
//!
//! The layer table ([`LayerState`]), the selected layout ([`SpaceSelection`] resolved against
//! [`LayoutState`]), the [`ViewProfile`] rules for off and frozen layers, and one
//! [`EntityState`] per entity the caller has facts about (the entity's own visibility flag, a
//! per-entity layer override, the layout it really belongs to, the viewport it is seen
//! through, and what the provider claimed). They come in beside the entity, keyed by handle, see
//! [`EntityStates`]. A provider that reports [`DwgProvider::entity_facts`] and
//! [`DwgProvider::viewports`] fills them: [`EntityStates::from_provider`] and
//! [`VisibilityFilter::run_from_provider`] build them without the caller handing anything in.
//!
//! # Order, and exactly one reason
//!
//! An entity is checked in a fixed order and the first match is its only reason:
//! [`PolicyReason::NotInSelectedLayout`], then layer state on the effective layer (the layer
//! override when there is one): [`PolicyReason::LayerOff`], [`PolicyReason::LayerFrozen`],
//! [`PolicyReason::ViewportFrozen`], then [`PolicyReason::EntityInvisible`], then
//! [`PolicyReason::OutsideViewport`]. Off and frozen stay separate reasons, they are never
//! folded into one "invisible". Each dropped entity is one [`IgnoredByPolicy`] row, so the
//! per-reason counts always sum to the dropped count.
//!
//! # Viewports are collected, not applied
//!
//! A paper-space viewport's visibility, clip boundary and frozen layers are kept as
//! [`ViewportInput`] for N3.3's clipping stage. A viewport-only freeze follows V1-3 of the view
//! profile ADR: by default it is reported ([`ViewportFreezeNote`]) and the layer is still
//! drawn. [`ViewportFreezeMode::Apply`] is the explicit opt-in that drops the entity as
//! [`PolicyReason::ViewportFrozen`]. [`PolicyReason::OutsideViewport`] is reported by the
//! clipping stage through [`EntityState::outside_viewport`], this module does no geometry.
//!
//! # Provider disagreement
//!
//! When [`EntityState::provider_visible`] differs from the policy answer the filter adds a
//! [`VisibilityDisagreement`] warning and applies the policy answer.

use std::collections::BTreeMap;

use super::fidelity::{Classification, FidelityClass};
use super::provider::{
    DwgProvider, EntityScope, Facet, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState,
    ProviderError, RawEntity, ViewportEntity,
};
use super::view_profile::{LayerStateRule, SpaceSelection, ViewProfile, ViewProfileError};
use crate::cad::ItemHandle;

/// Why an entity was dropped by policy. A closed set, one reason per dropped entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PolicyReason {
    /// Its effective layer is switched off.
    LayerOff,
    /// Its effective layer is frozen.
    LayerFrozen,
    /// Its effective layer is frozen in the viewport it is seen through (opt-in).
    ViewportFrozen,
    /// The entity's own visibility flag is off.
    EntityInvisible,
    /// It belongs to a layout other than the selected one.
    NotInSelectedLayout,
    /// The clipping stage reported it outside its viewport.
    OutsideViewport,
}

impl PolicyReason {
    /// Every reason, in evaluation order.
    pub const ALL: [PolicyReason; 6] = [
        PolicyReason::NotInSelectedLayout,
        PolicyReason::LayerOff,
        PolicyReason::LayerFrozen,
        PolicyReason::ViewportFrozen,
        PolicyReason::EntityInvisible,
        PolicyReason::OutsideViewport,
    ];

    /// The view-profile field responsible, as [`ViewProfile::knobs`] spells it.
    #[must_use]
    pub fn field(self) -> &'static str {
        match self {
            PolicyReason::LayerOff => "layers.off",
            PolicyReason::LayerFrozen => "layers.frozen",
            PolicyReason::ViewportFrozen => "layers.frozen_in_viewport",
            PolicyReason::EntityInvisible => "entity.visible",
            PolicyReason::NotInSelectedLayout | PolicyReason::OutsideViewport => "space",
        }
    }

    /// The diagnostic code the report carries. `LAYER_HIDDEN` and `VIEWPORT_STATE_IGNORED` are
    /// the codes N1.4 proposed and this task allocates, as strings until `DiagnosticCode` takes
    /// them.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            PolicyReason::LayerOff | PolicyReason::LayerFrozen => "LAYER_HIDDEN",
            PolicyReason::ViewportFrozen => "VIEWPORT_STATE_IGNORED",
            PolicyReason::EntityInvisible => "ENTITY_INVISIBLE",
            PolicyReason::NotInSelectedLayout => "NOT_IN_SELECTED_LAYOUT",
            PolicyReason::OutsideViewport => "OUTSIDE_VIEWPORT",
        }
    }

    /// The N3.5 class: always [`FidelityClass::IgnoredByPolicy`].
    #[must_use]
    pub fn fidelity(self) -> FidelityClass {
        FidelityClass::IgnoredByPolicy
    }
}

/// One dropped entity: the `IgnoredByPolicy { reason }` row N6.7 reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoredByPolicy {
    /// Why.
    pub reason: PolicyReason,
    /// The entity's handle, when it has one.
    pub handle: Option<ItemHandle>,
    /// The drawing's class name for it.
    pub class_name: String,
    /// The INSERT chain it was reached through.
    pub block_path: InstancePath,
    /// The effective layer the decision used.
    pub layer: String,
}

impl IgnoredByPolicy {
    /// The N3.5 classification of this row, naming the profile field responsible.
    #[must_use]
    pub fn classification(&self) -> Classification {
        Classification::ignored_by_policy(self.reason.field(), self.class_name.as_str())
    }
}

/// Facts about one entity that the provider boundary does not carry yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityState {
    /// The entity's own visibility flag. `true` unless the drawing says otherwise.
    pub visible: bool,
    /// A per-entity layer that replaces the one in the entity's provenance.
    pub layer_override: Option<String>,
    /// The layout the entity really belongs to, when it is known.
    pub layout: Option<LayoutId>,
    /// The viewport the entity is seen through, when it is seen through one.
    pub viewport: Option<ItemHandle>,
    /// Set by the clipping stage when the entity lies outside its viewport.
    pub outside_viewport: bool,
    /// What the provider claimed about visibility, when it claimed anything.
    pub provider_visible: Option<bool>,
}

impl Default for EntityState {
    fn default() -> Self {
        Self {
            visible: true,
            layer_override: None,
            layout: None,
            viewport: None,
            outside_viewport: false,
            provider_visible: None,
        }
    }
}

/// [`EntityState`] per entity handle. An entity with no entry has the default state.
#[derive(Debug, Clone, Default)]
pub struct EntityStates {
    map: BTreeMap<ItemHandle, EntityState>,
}

impl EntityStates {
    /// No facts.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the state of one entity.
    #[must_use]
    pub fn with(mut self, handle: ItemHandle, state: EntityState) -> Self {
        self.map.insert(handle, state);
        self
    }

    /// The state of `handle`, or the default.
    #[must_use]
    pub fn get(&self, handle: Option<ItemHandle>) -> EntityState {
        handle
            .and_then(|h| self.map.get(&h))
            .cloned()
            .unwrap_or_default()
    }
}

impl EntityStates {
    /// Builds the states of one scope's entities from what `provider` reports.
    ///
    /// # Errors
    ///
    /// The provider's own error, including [`ProviderError::CannotReport`].
    pub fn from_provider<P: DwgProvider + ?Sized>(
        provider: &P,
        scope: &EntityScope,
    ) -> Result<Self, ProviderError> {
        let mut states = Self::new();
        let layout = match scope {
            EntityScope::Layout(id) => Some(*id),
            EntityScope::BlockDefinition { .. } => None,
        };
        for (handle, f) in provider.entity_facts(scope)? {
            if f.visible.is_not_reportable() || f.provider_visible.is_not_reportable() {
                return Err(ProviderError::CannotReport(Facet::EntityVisibility));
            }
            if f.layer_override.is_not_reportable() {
                return Err(ProviderError::CannotReport(Facet::LayerOverride));
            }
            states = states.with(
                handle,
                EntityState {
                    visible: f.visible.present().copied().unwrap_or(true),
                    layer_override: f.layer_override.present().cloned(),
                    layout,
                    viewport: None,
                    outside_viewport: false,
                    provider_visible: f.provider_visible.present().copied(),
                },
            );
        }
        Ok(states)
    }
}

impl From<ViewportEntity> for ViewportInput {
    fn from(v: ViewportEntity) -> Self {
        Self {
            handle: v.handle,
            layout: v.layout,
            visible: v.visible,
            clip: v.clip,
            frozen_layers: v.frozen_layers,
        }
    }
}

/// A paper-space viewport, collected as an input for N3.3's clipping. Nothing here is applied.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewportInput {
    /// The viewport entity's handle.
    pub handle: ItemHandle,
    /// The layout that holds it.
    pub layout: LayoutId,
    /// The viewport's own visibility.
    pub visible: bool,
    /// The clip boundary, paper-space vertices in order.
    pub clip: Vec<[f64; 2]>,
    /// Layers frozen in this viewport only.
    pub frozen_layers: Vec<String>,
}

/// What a viewport-only layer freeze does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ViewportFreezeMode {
    /// V1-3: draw the layer and report that a freeze was ignored.
    #[default]
    ReportOnly,
    /// Drop the entity as [`PolicyReason::ViewportFrozen`].
    Apply,
}

/// A viewport freeze that was ignored under [`ViewportFreezeMode::ReportOnly`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewportFreezeNote {
    /// The entity that was drawn anyway.
    pub entity: Option<ItemHandle>,
    /// The viewport that froze its layer.
    pub viewport: ItemHandle,
    /// The layer.
    pub layer: String,
}

/// The provider and the policy disagree about whether an entity is visible. The policy wins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibilityDisagreement {
    /// The entity.
    pub handle: Option<ItemHandle>,
    /// Its class name.
    pub class_name: String,
    /// The effective layer.
    pub layer: String,
    /// What the provider said.
    pub provider_visible: bool,
    /// What the policy decided, and what was applied.
    pub policy_visible: bool,
    /// The policy's reason when it dropped the entity.
    pub policy_reason: Option<PolicyReason>,
}

/// The result of filtering one layout.
#[derive(Debug, Clone)]
pub struct VisibilityOutcome {
    /// The layout that was selected.
    pub layout: LayoutState,
    /// The entities that stay.
    pub kept: Vec<RawEntity>,
    /// One row per dropped entity.
    pub ignored: Vec<IgnoredByPolicy>,
    /// Provider-versus-policy disagreements (warnings).
    pub disagreements: Vec<VisibilityDisagreement>,
    /// Viewport-only freezes that were reported and not applied.
    pub ignored_freezes: Vec<ViewportFreezeNote>,
    /// The viewports of the selected layout, for N3.3.
    pub viewports: Vec<ViewportInput>,
}

impl VisibilityOutcome {
    /// How many entities were dropped.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.ignored.len() as u64
    }

    /// Dropped entities per reason. Every reason has a key, so the values sum to
    /// [`dropped`](Self::dropped).
    #[must_use]
    pub fn counts_by_reason(&self) -> BTreeMap<PolicyReason, u64> {
        let mut m: BTreeMap<PolicyReason, u64> =
            PolicyReason::ALL.iter().map(|r| (*r, 0)).collect();
        for row in &self.ignored {
            *m.entry(row.reason).or_insert(0) += 1;
        }
        m
    }
}

/// The policy filter. Build it with [`VisibilityFilter::new`], then [`decide`](Self::decide)
/// per entity or [`run`](Self::run) over a provider.
#[derive(Debug, Clone)]
pub struct VisibilityFilter {
    layers: Vec<LayerState>,
    layout: LayoutState,
    off: LayerStateRule,
    frozen: LayerStateRule,
    freeze_mode: ViewportFreezeMode,
    viewports: Vec<ViewportInput>,
}

/// Resolves the profile's space against the drawing's layouts.
///
/// Model space is the default, a named paper layout is an explicit request, and a layout the
/// drawing does not have is an error naming the ones it does.
///
/// # Errors
///
/// [`ViewProfileError::LayoutNotFound`] (with the paper layouts that exist) or
/// [`ViewProfileError::NoModelSpace`].
pub fn select_layout(
    layouts: &[LayoutState],
    space: &SpaceSelection,
) -> Result<LayoutState, ViewProfileError> {
    match space {
        SpaceSelection::Model => layouts
            .iter()
            .find(|l| l.kind == LayoutKind::Model)
            .cloned()
            .ok_or_else(|| ViewProfileError::NoModelSpace {
                available: layouts.iter().map(|l| l.name.clone()).collect(),
            }),
        SpaceSelection::PaperLayout(wanted) => layouts
            .iter()
            .find(|l| l.kind == LayoutKind::Paper && l.name.eq_ignore_ascii_case(wanted))
            .cloned()
            .ok_or_else(|| ViewProfileError::LayoutNotFound {
                requested: wanted.clone(),
                available: layouts
                    .iter()
                    .filter(|l| l.kind == LayoutKind::Paper)
                    .map(|l| l.name.clone())
                    .collect(),
            }),
    }
}

/// What [`VisibilityFilter::decide`] concluded for one entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// The layer the decision used.
    pub layer: String,
    /// The reason it is dropped, or `None` when it stays.
    pub dropped: Option<PolicyReason>,
}

impl VisibilityFilter {
    /// A filter for `layout` under `profile`'s layer rules, over the drawing's layer table.
    #[must_use]
    pub fn new(profile: &ViewProfile, layout: LayoutState, layers: Vec<LayerState>) -> Self {
        Self {
            layers,
            layout,
            off: profile.layers.off,
            frozen: profile.layers.frozen,
            freeze_mode: ViewportFreezeMode::default(),
            viewports: Vec::new(),
        }
    }

    /// Sets what a viewport-only freeze does.
    #[must_use]
    pub fn with_freeze_mode(mut self, mode: ViewportFreezeMode) -> Self {
        self.freeze_mode = mode;
        self
    }

    /// Adds the viewports of the selected layout. Only those on that layout are kept.
    #[must_use]
    pub fn with_viewports(mut self, viewports: Vec<ViewportInput>) -> Self {
        self.viewports = viewports
            .into_iter()
            .filter(|v| v.layout == self.layout.id)
            .collect();
        self
    }

    fn layer_state(&self, name: &str) -> Option<&LayerState> {
        self.layers
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(name))
    }

    /// Decides one entity from the state given, never from a provider's own answer.
    #[must_use]
    pub fn decide(&self, entity: &RawEntity, state: &EntityState) -> Decision {
        let layer = state
            .layer_override
            .clone()
            .unwrap_or_else(|| entity.provenance.layer.clone());
        let dropped = self.reason(&layer, state);
        Decision { layer, dropped }
    }

    fn reason(&self, layer: &str, state: &EntityState) -> Option<PolicyReason> {
        if state.layout.is_some_and(|l| l != self.layout.id) {
            return Some(PolicyReason::NotInSelectedLayout);
        }
        if let Some(ls) = self.layer_state(layer) {
            if ls.off && self.off == LayerStateRule::Exclude {
                return Some(PolicyReason::LayerOff);
            }
            if ls.frozen && self.frozen == LayerStateRule::Exclude {
                return Some(PolicyReason::LayerFrozen);
            }
        }
        if self.freeze_mode == ViewportFreezeMode::Apply && self.viewport_freezes(layer, state) {
            return Some(PolicyReason::ViewportFrozen);
        }
        if !state.visible {
            return Some(PolicyReason::EntityInvisible);
        }
        if state.outside_viewport {
            return Some(PolicyReason::OutsideViewport);
        }
        None
    }

    fn viewport_for(&self, state: &EntityState) -> Option<&ViewportInput> {
        let h = state.viewport?;
        self.viewports.iter().find(|v| v.handle == h)
    }

    fn viewport_freezes(&self, layer: &str, state: &EntityState) -> bool {
        self.viewport_for(state).is_some_and(|v| {
            v.frozen_layers
                .iter()
                .any(|f| f.eq_ignore_ascii_case(layer))
        })
    }

    /// Filters `entities` (one layout's worth) and records every decision.
    #[must_use]
    pub fn filter(&self, entities: Vec<RawEntity>, states: &EntityStates) -> VisibilityOutcome {
        let mut out = VisibilityOutcome {
            layout: self.layout.clone(),
            kept: Vec::new(),
            ignored: Vec::new(),
            disagreements: Vec::new(),
            ignored_freezes: Vec::new(),
            viewports: self.viewports.clone(),
        };
        for e in entities {
            let st = states.get(e.provenance.handle);
            let d = self.decide(&e, &st);
            if let Some(p) = st.provider_visible
                && p != d.dropped.is_none()
            {
                out.disagreements.push(VisibilityDisagreement {
                    handle: e.provenance.handle,
                    class_name: e.provenance.class_name.clone(),
                    layer: d.layer.clone(),
                    provider_visible: p,
                    policy_visible: d.dropped.is_none(),
                    policy_reason: d.dropped,
                });
            }
            if self.freeze_mode == ViewportFreezeMode::ReportOnly
                && d.dropped.is_none()
                && self.viewport_freezes(&d.layer, &st)
                && let Some(vp) = self.viewport_for(&st)
            {
                out.ignored_freezes.push(ViewportFreezeNote {
                    entity: e.provenance.handle,
                    viewport: vp.handle,
                    layer: d.layer.clone(),
                });
            }
            match d.dropped {
                Some(reason) => out.ignored.push(IgnoredByPolicy {
                    reason,
                    handle: e.provenance.handle,
                    class_name: e.provenance.class_name.clone(),
                    block_path: e.provenance.block_path.clone(),
                    layer: d.layer,
                }),
                None => out.kept.push(e),
            }
        }
        out
    }

    /// Resolves the layout, reads the layer table and the layout's entities from `provider`,
    /// and filters them.
    ///
    /// # Errors
    ///
    /// The errors of [`select_layout`], and [`ViewProfileError::Provider`] when the provider
    /// fails or cannot report layers or layouts.
    pub fn run<P: DwgProvider + ?Sized>(
        provider: &P,
        profile: &ViewProfile,
        states: &EntityStates,
        viewports: Vec<ViewportInput>,
        mode: ViewportFreezeMode,
    ) -> Result<VisibilityOutcome, ViewProfileError> {
        let layouts = provider.layouts()?;
        let layout = select_layout(&layouts, &profile.space)?;
        let layers = provider.layers()?;
        let mut cursor = provider.entities(&EntityScope::Layout(layout.id))?;
        let entities = cursor.collect_all()?;
        drop(cursor);
        Ok(Self::new(profile, layout, layers)
            .with_freeze_mode(mode)
            .with_viewports(viewports)
            .filter(entities, states))
    }

    /// Like [`run`](Self::run), with the entity states and viewports read from `provider`
    /// instead of handed in.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run); a provider that cannot report the facets is
    /// [`ViewProfileError::Provider`].
    pub fn run_from_provider<P: DwgProvider + ?Sized>(
        provider: &P,
        profile: &ViewProfile,
        mode: ViewportFreezeMode,
    ) -> Result<VisibilityOutcome, ViewProfileError> {
        let layouts = provider.layouts()?;
        let layout = select_layout(&layouts, &profile.space)?;
        let states = EntityStates::from_provider(provider, &EntityScope::Layout(layout.id))?;
        let viewports = provider
            .viewports()?
            .into_iter()
            .map(ViewportInput::from)
            .collect();
        Self::run(provider, profile, &states, viewports, mode)
    }
}
