//! Nested block and INSERT flattening with bounded recursion and a transform lineage (N4.4, #1075).
//!
//! A drawing's geometry mostly lives inside blocks, and INSERTs nest and multiply without limit.
//! [`BlockResolver`] walks them depth-first through the [`DwgProvider`] traits, in the provider's own
//! entity order, and emits the leaves as [`FlatEntity`] values.
//!
//! # Bounds
//!
//! The ceilings come in through [`FlattenBounds`], which wraps the [`SourceLimits`] of N3.6 (depth,
//! total entities, expanded instances, grid size, allocation) and adds the one ceiling that set does
//! not have, entities per block. Nothing is hard-coded here. A breach is a typed [`FlattenError`]
//! naming the limit, the observed value and the block path. A block that reaches itself or an
//! ancestor is refused by name, as [`FlattenError::BlockRecursionLimit`], the moment the second
//! reference is met, so the budget is never spent on it.
//!
//! The walk keeps its own stack instead of recursing, so a raised depth limit cannot overflow the
//! thread's stack.
//!
//! # Transforms
//!
//! Every level's INSERT map is composed once, through N3.4's [`ComposedTransform`], into a single
//! `f64` affine. A leaf carries that matrix, and [`FlatEntity::map_point`] applies it in one step, so
//! a coordinate is rounded once and not once per level. Promotion of the composed map (the matrix of
//! N1.5) is `transform::promote`'s, not repeated here.
//!
//! An INSERT whose extrusion normal is `-Z` is the OCS mirror `x -> -x`, applied outside the INSERT's
//! own map. A normal that is not parallel to Z needs a 3D placement the 2D adapter does not carry, so
//! that INSERT is skipped and classified `RequiresExtension`, never dropped.
//!
//! # Lineage
//!
//! The ordered block path, each level's INSERT handle and each level's transform go to the side
//! stream: the N3.2 [`ProvenanceSink`] receives the usual record (with the full block path and the
//! composed map), and a [`LineageStore`] keyed by the same [`ProvenanceHandle`] receives the
//! per-level [`TransformLineage`]. Neither is a field of the geometry record, so
//! `size_of::<Normalized2DGeometry>()` does not move whether lineage is recorded or not, and a sink
//! that keeps nothing ([`NoProvenance`](super::provenance::NoProvenance)) costs no lineage at all.

use std::fmt;

use thiserror::Error;

use crate::cad::ItemHandle;

use super::fidelity::FidelityClass;
use super::limits::{Limit, LimitError, SourceGuard, SourceLimits};
use super::normalized::ProvenanceHandle;
use super::provenance::{Provenance, ProvenanceSink};
use super::provider::{
    BlockRef, DwgProvider, EntityCursor, EntityScope, InstancePath, InstanceStep, Point3,
    ProviderError, RawEntity, RawGeometry,
};
use super::transform::{ComposeError, ComposedTransform, LineageLevel};

/// The ceilings one flatten run works under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FlattenBounds {
    /// N3.6's limit set: nesting depth, total entities, expanded instances, grid size, allocation.
    pub limits: SourceLimits,
    /// The most entities one instance of one block may hold.
    pub max_entities_per_block: u64,
}

impl FlattenBounds {
    /// Bounds over `limits`, with the per-block ceiling at `max_entities_per_block`.
    #[must_use]
    pub const fn new(limits: SourceLimits, max_entities_per_block: u64) -> Self {
        Self {
            limits,
            max_entities_per_block,
        }
    }
}

impl Default for FlattenBounds {
    fn default() -> Self {
        Self::new(SourceLimits::default(), 1_000_000)
    }
}

/// Why a flatten run stopped.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum FlattenError {
    /// A block references itself or an ancestor. `depth` is the nesting depth of the offending
    /// INSERT and `block_path` the INSERT chain including it.
    #[error("block recursion at depth {depth}: {block_path}")]
    BlockRecursionLimit {
        /// How deep the offending INSERT sits.
        depth: u32,
        /// The INSERT chain, outermost first, ending with the offending INSERT.
        block_path: InstancePath,
    },
    /// One of N3.6's limits was exceeded.
    #[error("limit {} exceeded at {block_path}: observed {observed}, ceiling {ceiling}", .limit.name())]
    Limit {
        /// Which limit.
        limit: Limit,
        /// The value that broke it.
        observed: u64,
        /// The configured ceiling.
        ceiling: u64,
        /// Where the walk was.
        block_path: InstancePath,
    },
    /// One instance of a block holds more entities than the per-block ceiling.
    #[error("block `{block}` at {block_path} holds more than {ceiling} entities (observed {observed})")]
    EntitiesPerBlock {
        /// The block's name.
        block: String,
        /// The count that broke the ceiling.
        observed: u64,
        /// The configured ceiling.
        ceiling: u64,
        /// The INSERT chain that reached the block.
        block_path: InstancePath,
    },
    /// An INSERT's map could not be composed (non-finite or singular).
    #[error("{0}")]
    Compose(ComposeError),
    /// The provider failed.
    #[error("provider: {0}")]
    Provider(ProviderError),
}

/// Why an INSERT was not expanded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SkipReason {
    /// The provider has no content for an anonymous or dynamic block (`*U`, `*D`, ...).
    AnonymousBlockUnresolved,
    /// The provider has no block of that name.
    BlockNotFound,
    /// The INSERT's normal is not parallel to Z.
    NormalNotZ,
    /// The INSERT has no handle, so no lineage step can name it.
    MissingInsertHandle,
}

impl SkipReason {
    /// A stable machine-readable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::AnonymousBlockUnresolved => "anonymous-block-unresolved",
            Self::BlockNotFound => "block-not-found",
            Self::NormalNotZ => "insert-normal-not-z",
            Self::MissingInsertHandle => "insert-missing-handle",
        }
    }
}

/// An INSERT the walk met and did not expand. Counted, never dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct SkippedInsert {
    /// The INSERT chain including this INSERT (without a handle, the chain up to its parent).
    pub block_path: InstancePath,
    /// The block it referenced.
    pub block_name: String,
    /// Its handle, when it has one.
    pub insert_handle: Option<ItemHandle>,
    /// The class it falls in.
    pub class: FidelityClass,
    /// Why.
    pub reason: SkipReason,
}

/// One leaf entity with the single map from its block's coordinates to the layout's.
#[derive(Debug, Clone)]
pub struct FlatEntity {
    /// The leaf as the provider reported it, coordinates still in block space. Its provenance stub
    /// carries the full block path.
    pub entity: RawEntity,
    matrix: [f64; 6],
    lineage: Option<ProvenanceHandle>,
}

impl FlatEntity {
    /// The composed map, `[m00, m01, m10, m11, tx, ty]`.
    #[must_use]
    pub const fn matrix(&self) -> [f64; 6] {
        self.matrix
    }

    /// The handle of this leaf's provenance and lineage records, when the sink kept any.
    #[must_use]
    pub const fn lineage_handle(&self) -> Option<ProvenanceHandle> {
        self.lineage
    }

    /// Applies the composed map to a block-space point, once.
    #[must_use]
    pub fn map_point(&self, p: Point3) -> [f64; 2] {
        let [a, b, c, d, tx, ty] = self.matrix;
        [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty]
    }
}

/// The per-level transforms of one leaf, outermost first.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformLineage {
    levels: Vec<LineageLevel>,
}

impl TransformLineage {
    /// The levels, outermost first. Empty for a leaf a layout owns directly.
    #[must_use]
    pub fn levels(&self) -> &[LineageLevel] {
        &self.levels
    }

    /// How many INSERTs deep.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// The block path the levels spell.
    #[must_use]
    pub fn block_path(&self) -> InstancePath {
        self.levels.iter().fold(InstancePath::root(), |p, l| {
            p.child(l.insert.block_name.clone(), l.insert.insert_handle)
        })
    }
}

/// The lineage side table, keyed by the provenance handle of the leaf it describes.
#[derive(Debug, Clone, Default)]
pub struct LineageStore {
    records: std::collections::BTreeMap<u64, TransformLineage>,
}

impl LineageStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many leaves have a record.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The lineage of the leaf `handle` names.
    #[must_use]
    pub fn get(&self, handle: ProvenanceHandle) -> Option<&TransformLineage> {
        self.records.get(&handle.get())
    }
}

/// What one flatten run produced.
#[derive(Debug, Clone, Default)]
pub struct Flattened {
    /// The leaves, in the deterministic depth-first order of the walk.
    pub entities: Vec<FlatEntity>,
    /// The INSERTs that were not expanded.
    pub skipped: Vec<SkippedInsert>,
    /// How many block instances were opened (a MINSERT counts rows times columns).
    pub instances: u64,
    /// Bytes the guard charged for the run, always within the configured allocation ceiling.
    pub allocated_bytes: u64,
}

/// Walks INSERTs through a provider and flattens them.
pub struct BlockResolver<'a, P: DwgProvider + ?Sized> {
    provider: &'a P,
    bounds: FlattenBounds,
}

impl<P: DwgProvider + ?Sized> fmt::Debug for BlockResolver<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockResolver")
            .field("provider", &self.provider.name())
            .field("bounds", &self.bounds)
            .finish()
    }
}

impl<'a, P: DwgProvider + ?Sized> BlockResolver<'a, P> {
    /// A resolver over `provider` under `bounds`.
    #[must_use]
    pub fn new(provider: &'a P, bounds: FlattenBounds) -> Self {
        Self { provider, bounds }
    }

    /// Flattens everything `scope` reaches.
    ///
    /// `source` and `layout` fill the provenance records the `sink` keeps, and `lineage` receives
    /// one record per kept provenance handle.
    ///
    /// # Errors
    ///
    /// A [`FlattenError`] naming the first bound that broke, a cycle, an INSERT map that cannot be
    /// composed, or a provider failure.
    pub fn flatten<S: ProvenanceSink>(
        &self,
        scope: &EntityScope,
        source: &str,
        layout: &str,
        sink: &mut S,
        lineage: &mut LineageStore,
    ) -> Result<Flattened, FlattenError> {
        let _ = (scope, source, layout, sink, lineage, &self.bounds);
        let _: Option<(
            &dyn EntityCursor,
            SourceGuard,
            Provenance,
            BlockRef,
            RawGeometry,
            LimitError,
            ComposedTransform,
            InstanceStep,
        )> = None;
        todo!("N4.4 walk")
    }
}
