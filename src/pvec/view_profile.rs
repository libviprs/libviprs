//! The view-profile options type (N3.3, #1097).
//!
//! Skeleton: types only, behaviour follows.

use std::collections::BTreeMap;
use std::fmt;

use super::provider::{DwgProvider, InstancePath, ProviderError, RawEntity};
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewProfileError {
    /// No such paper layout.
    LayoutNotFound {
        /// Asked for.
        requested: String,
        /// Paper layouts that exist.
        available: Vec<String>,
    },
    /// No model space.
    NoModelSpace {
        /// Layouts that exist.
        available: Vec<String>,
    },
    /// Paper space is out of v1.
    PaperSpaceUnsupported {
        /// The layout.
        layout: String,
    },
    /// The provider failed.
    Provider(ProviderError),
}

impl fmt::Display for ViewProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ViewProfileError {}

impl ViewProfile {
    /// The v1 default.
    #[must_use]
    pub fn default_profile() -> Self {
        todo!("N3.3")
    }

    /// Every knob.
    #[must_use]
    pub fn knobs(&self) -> Vec<Knob> {
        todo!("N3.3")
    }

    /// Applies the profile.
    ///
    /// # Errors
    ///
    /// See [`ViewProfileError`].
    pub fn apply<P: DwgProvider + ?Sized>(&self, _provider: &P) -> Result<ViewOutcome, ViewProfileError> {
        todo!("N3.3")
    }
}

impl Exclusion {
    /// Profile field.
    #[must_use]
    pub fn field(self) -> &'static str {
        todo!("N3.3")
    }
    /// Diagnostic code.
    #[must_use]
    pub fn code(self) -> &'static str {
        todo!("N3.3")
    }
}

#[cfg(feature = "pvec-normalized")]
impl Exclusion {
    /// Fidelity class.
    #[must_use]
    pub fn fidelity(self) -> super::normalized::FidelityClass {
        todo!("N3.3")
    }
}

impl fmt::Display for ViewProfile {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!("N3.3")
    }
}
