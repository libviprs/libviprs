//! The view-profile options type (N3.3, #1097).
//!
//! "The drawing" is not one thing: layers hide entities, blocks nest, layouts carry viewports and
//! xrefs point at other files. [`ViewProfile`] is the explicit input that says which answer a
//! conversion wants, so no provider default decides it by accident (panel finding 3 of #1039).
//! The v1 answer is the frozen rule of `docs/adr/pvec-view-profile.md` (clauses V1-1 to V1-5),
//! measured in `docs/spikes/view-context.md` (N1.4).
//!
//! # Every knob is a row of the N1.4 decision table
//!
//! | field | N1.4 row | v1 default |
//! |---|---|---|
//! | `space` | `model-space`, `paper-space-layout` | model space |
//! | `viewport_clipping` | `viewport-clip-exact`, `viewport-clip-bounded` | unsupported |
//! | `layers.off` | `layer-off` | exclude |
//! | `layers.frozen` | `layer-frozen` | exclude |
//! | `layers.frozen_in_viewport` | `layer-frozen-in-viewport` | ignore and report |
//! | `layers.insert_override` | `insert-layer-override` | layer "0" inherits the INSERT's layer |
//! | `nested_inserts.max_depth` | `nested-insert` | 64, the decoder's own limit |
//! | `xrefs` | `xref` | report as external |
//!
//! [`ViewProfile::knobs`] returns that table as data, and a test checks every row name against
//! the spike document, so a knob without a row is a red test. Locked and no-plot layers are
//! visible under V1-3 and have no knob.
//!
//! # Policy is applied here, not trusted
//!
//! [`ViewProfile::apply`] walks a [`DwgProvider`]'s entities for the selected space, descends
//! into INSERTs and checks every entity against the profile, whatever the provider did or did not
//! filter. Each excluded entity comes back as an [`ExcludedEntity`] naming the profile field
//! that excluded it ([`Exclusion::field`]), and the [`ViewReport`] counts them per field and per
//! layer, so "nothing was on that layer" and "the filter lost it" stay distinguishable.
//!
//! What this does not do: it composes no transforms (N3.4), maps no geometry to PVec and resolves
//! no xref (N4.5). It stops at an INSERT of a block the provider does not have and reports it.
//! A leaf's effective layer and its [`InstancePath`] are what N3.4 and N4.4 need next.
//!
//! # Fidelity classes
//!
//! A layer or depth exclusion is `IgnoredByPolicy`. An unresolved block is `Unsupported` with
//! `UNRESOLVED_BLOCK` (105), as `docs/adr/pvec-unsupported-policy.md` freezes it. `LAYER_HIDDEN`
//! is the proposed code of N1.4 and N4.6 allocates it; here it is a string, not a
//! `DiagnosticCode`.

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

use super::limits::SourceLimits;

use super::provider::{
    DwgProvider, EntityScope, InstancePath, LayerState, LayoutKind, ProviderError, RawEntity,
    RawGeometry,
};
use crate::cad::ItemHandle;

/// Which space is drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpaceSelection {
    /// Model space.
    Model,
    /// A named paper-space layout.
    PaperLayout(String),
}

/// What a layer-state flag does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayerStateRule {
    /// Entities on such a layer are excluded.
    Exclude,
    /// The flag is ignored and the entities are drawn.
    Ignore,
}

/// Viewport-only layer freezes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewportFreezeRule {
    /// Draw the layer and report that a freeze was ignored.
    IgnoreAndReport,
}

/// Viewport clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewportClipping {
    /// Refuse.
    Unsupported,
}

/// Layer-0 inheritance inside an INSERT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertLayerRule {
    /// Layer "0" takes the enclosing INSERT's layer.
    InheritOnLayerZero,
    /// Every entity keeps its own layer.
    OwnLayerOnly,
}

/// Xrefs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum XrefPolicy {
    /// Contribute nothing, report as external.
    ReportExternal,
}

/// Layer visibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerVisibility {
    /// Off layers.
    pub off: LayerStateRule,
    /// Frozen layers.
    pub frozen: LayerStateRule,
    /// Viewport freezes.
    pub frozen_in_viewport: ViewportFreezeRule,
    /// Layer-0 inheritance.
    pub insert_override: InsertLayerRule,
}

/// Nested inserts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedInsertPolicy {
    /// Deepest INSERT chain followed.
    pub max_depth: u32,
}

/// The view context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewProfile {
    /// Space.
    pub space: SpaceSelection,
    /// Clipping.
    pub viewport_clipping: ViewportClipping,
    /// Layers.
    pub layers: LayerVisibility,
    /// Nested inserts.
    pub nested_inserts: NestedInsertPolicy,
    /// Xrefs.
    pub xrefs: XrefPolicy,
    /// N3.6's resource ceilings for the conversion. Not a view knob (it has no N1.4 row and
    /// is absent from [`ViewProfile::knobs`]); it rides along so a caller overrides it through
    /// the same options value, and flattening builds its guard from it.
    pub limits: SourceLimits,
}

/// One knob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Knob {
    /// Field path.
    pub field: &'static str,
    /// Value.
    pub value: String,
    /// N1.4 row.
    pub row: &'static str,
}

/// Why an entity left the drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Exclusion {
    /// Off layer.
    LayerOff,
    /// Frozen layer.
    LayerFrozen,
    /// Too deep.
    DepthLimit,
    /// Unresolved block.
    Unresolved,
}

/// One excluded entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedEntity {
    /// Handle.
    pub handle: Option<ItemHandle>,
    /// Class name.
    pub class_name: String,
    /// Path.
    pub block_path: InstancePath,
    /// Effective layer.
    pub layer: String,
    /// Reason.
    pub reason: Exclusion,
}

/// One kept leaf.
#[derive(Debug, Clone)]
pub struct ViewedEntity {
    /// The entity.
    pub entity: RawEntity,
    /// Effective layer.
    pub effective_layer: String,
    /// Instances.
    pub instances: u64,
}

/// Counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewReport {
    /// Profile used.
    pub profile: ViewProfile,
    /// Space drawn.
    pub space_name: String,
    /// Leaves kept.
    pub included: u64,
    /// Excluded, by field.
    pub excluded_by_field: BTreeMap<&'static str, u64>,
    /// Excluded, by layer.
    pub excluded_by_layer: BTreeMap<String, u64>,
}

/// Result of applying.
#[derive(Debug, Clone)]
pub struct ViewOutcome {
    /// Kept.
    pub included: Vec<ViewedEntity>,
    /// Excluded.
    pub excluded: Vec<ExcludedEntity>,
    /// Report.
    pub report: ViewReport,
}

/// Why a profile could not be applied.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ViewProfileError {
    /// No such paper layout.
    #[error("paper layout {requested:?} not found, the paper layouts that exist are {available:?}")]
    LayoutNotFound {
        /// Asked for.
        requested: String,
        /// Paper layouts that exist.
        available: Vec<String>,
    },
    /// No model space.
    #[error("the drawing has no model space, the layouts that exist are {available:?}")]
    NoModelSpace {
        /// Layouts that exist.
        available: Vec<String>,
    },
    /// Paper space is out of v1.
    #[error("paper layout {layout:?} exists but paper space is out of v1 (VIEW_NOT_ENABLED, 508)")]
    PaperSpaceUnsupported {
        /// The layout.
        layout: String,
    },
    /// The provider failed, or cannot report something the profile needs.
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),
}

impl ViewProfile {
    /// The v1 "visible flattened drawing" rule, spelled out field by field.
    ///
    /// Constructed explicitly so a caller can see what they accepted. A test pins every value,
    /// so changing one is a red test and a new ADR, not a quiet edit.
    #[must_use]
    pub fn default_profile() -> Self {
        Self {
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
            limits: SourceLimits::default(),
        }
    }

    /// This profile with `limits` as its resource ceilings.
    #[must_use]
    pub fn with_limits(mut self, limits: SourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Every knob with its value and the N1.4 decision-table row it comes from.
    #[must_use]
    pub fn knobs(&self) -> Vec<Knob> {
        let rule = |r: LayerStateRule| match r {
            LayerStateRule::Exclude => "exclude",
            LayerStateRule::Ignore => "ignore",
        };
        let (space, space_row) = match &self.space {
            SpaceSelection::Model => ("model".to_string(), "model-space"),
            SpaceSelection::PaperLayout(n) => (format!("paper:{n}"), "paper-space-layout"),
        };
        vec![
            Knob {
                field: "space",
                value: space,
                row: space_row,
            },
            Knob {
                field: "viewport_clipping",
                value: match self.viewport_clipping {
                    ViewportClipping::Unsupported => "unsupported",
                }
                .into(),
                row: "viewport-clip-exact",
            },
            Knob {
                field: "layers.off",
                value: rule(self.layers.off).into(),
                row: "layer-off",
            },
            Knob {
                field: "layers.frozen",
                value: rule(self.layers.frozen).into(),
                row: "layer-frozen",
            },
            Knob {
                field: "layers.frozen_in_viewport",
                value: match self.layers.frozen_in_viewport {
                    ViewportFreezeRule::IgnoreAndReport => "ignore-and-report",
                }
                .into(),
                row: "layer-frozen-in-viewport",
            },
            Knob {
                field: "layers.insert_override",
                value: match self.layers.insert_override {
                    InsertLayerRule::InheritOnLayerZero => "inherit-on-layer-zero",
                    InsertLayerRule::OwnLayerOnly => "own-layer-only",
                }
                .into(),
                row: "insert-layer-override",
            },
            Knob {
                field: "nested_inserts.max_depth",
                value: self.nested_inserts.max_depth.to_string(),
                row: "nested-insert",
            },
            Knob {
                field: "xrefs",
                value: match self.xrefs {
                    XrefPolicy::ReportExternal => "report-external",
                }
                .into(),
                row: "xref",
            },
        ]
    }

    /// The fields of this profile that can exclude an entity, as [`Knob::field`] spells them.
    /// A layer rule set to `Ignore` cannot exclude, so it is not listed.
    #[must_use]
    pub fn exclusion_fields(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.layers.off == LayerStateRule::Exclude {
            out.push(Exclusion::LayerOff.field());
        }
        if self.layers.frozen == LayerStateRule::Exclude {
            out.push(Exclusion::LayerFrozen.field());
        }
        out.push(Exclusion::DepthLimit.field());
        out.push(Exclusion::Unresolved.field());
        out
    }

    /// Applies the profile to one drawing.
    ///
    /// The provider's entities for the selected space are walked in the provider's own order,
    /// INSERTs are descended depth first (a visible INSERT is not itself a leaf), and every leaf
    /// is checked against the layer rules (V1-2, V1-3). An INSERT that is excluded takes its
    /// whole expansion with it and counts once.
    ///
    /// # Errors
    ///
    /// [`ViewProfileError::LayoutNotFound`] for a paper layout the drawing does not have,
    /// [`ViewProfileError::PaperSpaceUnsupported`] for one it does (v1 draws model space only),
    /// [`ViewProfileError::NoModelSpace`], and [`ViewProfileError::Provider`] when the provider
    /// fails or cannot report layer state while the profile acts on it.
    pub fn apply<P: DwgProvider + ?Sized>(
        &self,
        provider: &P,
    ) -> Result<ViewOutcome, ViewProfileError> {
        let layouts = provider.layouts()?;
        let names = |kind: LayoutKind| -> Vec<String> {
            layouts
                .iter()
                .filter(|l| l.kind == kind)
                .map(|l| l.name.clone())
                .collect()
        };
        let model = match &self.space {
            SpaceSelection::Model => layouts
                .iter()
                .find(|l| l.kind == LayoutKind::Model)
                .ok_or_else(|| ViewProfileError::NoModelSpace {
                    available: layouts.iter().map(|l| l.name.clone()).collect(),
                })?,
            SpaceSelection::PaperLayout(wanted) => {
                return Err(
                    match layouts.iter().find(|l| {
                        l.kind == LayoutKind::Paper && l.name.eq_ignore_ascii_case(wanted)
                    }) {
                        Some(l) => ViewProfileError::PaperSpaceUnsupported {
                            layout: l.name.clone(),
                        },
                        None => ViewProfileError::LayoutNotFound {
                            requested: wanted.clone(),
                            available: names(LayoutKind::Paper),
                        },
                    },
                );
            }
        };

        let needs_layers = self.layers.off == LayerStateRule::Exclude
            || self.layers.frozen == LayerStateRule::Exclude;
        let layer_table: Vec<LayerState> = if needs_layers {
            provider.layers()?
        } else {
            Vec::new()
        };

        let mut walk = Walk {
            profile: self,
            provider,
            layers: layer_table,
            included: Vec::new(),
            excluded: Vec::new(),
        };
        let mut cursor = provider.entities(&EntityScope::Layout(model.id))?;
        let mut top = Vec::new();
        while let Some(e) = cursor.next_entity()? {
            top.push(e);
        }
        drop(cursor);
        for e in top {
            walk.visit(e, 0, None, 1)?;
        }

        let mut by_field = BTreeMap::new();
        let mut by_layer = BTreeMap::new();
        for x in &walk.excluded {
            *by_field.entry(x.reason.field()).or_insert(0u64) += 1;
            *by_layer.entry(x.layer.clone()).or_insert(0u64) += 1;
        }
        let report = ViewReport {
            profile: self.clone(),
            space_name: model.name.clone(),
            included: walk.included.len() as u64,
            excluded_by_field: by_field,
            excluded_by_layer: by_layer,
        };
        Ok(ViewOutcome {
            included: walk.included,
            excluded: walk.excluded,
            report,
        })
    }
}

struct Walk<'a, P: ?Sized> {
    profile: &'a ViewProfile,
    provider: &'a P,
    layers: Vec<LayerState>,
    included: Vec<ViewedEntity>,
    excluded: Vec<ExcludedEntity>,
}

impl<P: DwgProvider + ?Sized> Walk<'_, P> {
    fn layer_state(&self, name: &str) -> Option<&LayerState> {
        // Symbol-table names are case-insensitive in a DWG.
        self.layers
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(name))
    }

    /// The reason a layer hides an entity under this profile, if it does.
    fn layer_exclusion(&self, layer: &str) -> Option<Exclusion> {
        let state = self.layer_state(layer)?;
        if state.off && self.profile.layers.off == LayerStateRule::Exclude {
            Some(Exclusion::LayerOff)
        } else if state.frozen && self.profile.layers.frozen == LayerStateRule::Exclude {
            Some(Exclusion::LayerFrozen)
        } else {
            None
        }
    }

    fn visit(
        &mut self,
        entity: RawEntity,
        depth: u32,
        container_layer: Option<&str>,
        instances: u64,
    ) -> Result<(), ViewProfileError> {
        // V1-2: a leaf (or INSERT) on layer "0" inside an INSERT takes that INSERT's layer.
        let effective = match container_layer {
            Some(c)
                if self.profile.layers.insert_override == InsertLayerRule::InheritOnLayerZero
                    && entity.provenance.layer == "0" =>
            {
                c.to_string()
            }
            _ => entity.provenance.layer.clone(),
        };

        if let Some(reason) = self.layer_exclusion(&effective) {
            self.exclude(&entity, effective, reason);
            return Ok(());
        }

        let RawGeometry::Insert(block) = &entity.geometry else {
            self.included.push(ViewedEntity {
                entity,
                effective_layer: effective,
                instances,
            });
            return Ok(());
        };

        if depth >= self.profile.nested_inserts.max_depth {
            self.exclude(&entity, effective, Exclusion::DepthLimit);
            return Ok(());
        }
        let via = entity.child_path().unwrap_or_else(|| {
            entity
                .provenance
                .block_path
                .child(block.block_name.clone(), crate::cad::ItemHandle::new(0))
        });
        let scope = EntityScope::BlockDefinition {
            name: block.block_name.clone(),
            reached_via: via,
        };
        let mut cursor = match self.provider.entities(&scope) {
            Ok(c) => c,
            Err(ProviderError::NoSuchScope(_)) => {
                self.exclude(&entity, effective, Exclusion::Unresolved);
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        let mut children = Vec::new();
        while let Some(c) = cursor.next_entity()? {
            children.push(c);
        }
        drop(cursor);
        let n = u64::from(block.grid.rows.max(1)) * u64::from(block.grid.columns.max(1));
        let child_instances = instances.saturating_mul(n);
        for c in children {
            self.visit(c, depth + 1, Some(&effective), child_instances)?;
        }
        Ok(())
    }

    fn exclude(&mut self, e: &RawEntity, layer: String, reason: Exclusion) {
        self.excluded.push(ExcludedEntity {
            handle: e.provenance.handle,
            class_name: e.provenance.class_name.clone(),
            block_path: e.provenance.block_path.clone(),
            layer,
            reason,
        });
    }
}

impl Exclusion {
    /// Every way an entity can be excluded.
    pub const ALL: [Exclusion; 4] = [
        Exclusion::LayerOff,
        Exclusion::LayerFrozen,
        Exclusion::DepthLimit,
        Exclusion::Unresolved,
    ];

    /// The classification of an entity excluded this way: `IgnoredByPolicy` naming
    /// [`Exclusion::field`], or `Unsupported` with `UnresolvedBlock` for an unresolved block.
    #[must_use]
    pub fn classification(
        self,
        source_class: impl Into<String>,
    ) -> super::fidelity::Classification {
        use super::fidelity::{Classification, ReasonCode};
        match self {
            Exclusion::Unresolved => {
                Classification::refused(ReasonCode::UnresolvedBlock, source_class)
            }
            _ => Classification::ignored_by_policy(self.field(), source_class),
        }
    }

    /// The profile field that excluded the entity, as [`Knob::field`] spells it.
    #[must_use]
    pub fn field(self) -> &'static str {
        match self {
            Exclusion::LayerOff => "layers.off",
            Exclusion::LayerFrozen => "layers.frozen",
            Exclusion::DepthLimit => "nested_inserts.max_depth",
            Exclusion::Unresolved => "xrefs",
        }
    }

    /// The diagnostic the report carries: `LAYER_HIDDEN` (proposed by N1.4, allocated by N4.6),
    /// `BLOCK_DEPTH_EXCEEDED` (proposed here, same owner) or `UNRESOLVED_BLOCK` (105).
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Exclusion::LayerOff | Exclusion::LayerFrozen => "LAYER_HIDDEN",
            Exclusion::DepthLimit => "BLOCK_DEPTH_EXCEEDED",
            Exclusion::Unresolved => "UNRESOLVED_BLOCK",
        }
    }
}

#[cfg(feature = "pvec-normalized")]
impl Exclusion {
    /// The N3.5 fidelity class of an entity excluded this way.
    #[must_use]
    pub fn fidelity(self) -> super::normalized::FidelityClass {
        use super::normalized::FidelityClass;
        match self {
            Exclusion::LayerOff | Exclusion::LayerFrozen | Exclusion::DepthLimit => {
                FidelityClass::IgnoredByPolicy
            }
            Exclusion::Unresolved => FidelityClass::Unsupported,
        }
    }
}

impl fmt::Display for ViewProfile {
    /// One `field = value` line per knob, so a report shows what the caller accepted.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for k in self.knobs() {
            writeln!(f, "{} = {}", k.field, k.value)?;
        }
        Ok(())
    }
}
