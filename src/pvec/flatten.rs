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
//! own map. A normal that is the Z axis up to rounding counts as the Z axis (#1609). A normal that is
//! not parallel to Z needs a 3D placement the 2D adapter does not carry, so by default that INSERT is
//! skipped and classified `RequiresExtension`, never dropped.
//!
//! # Out of the plan's plane
//!
//! A resolver made [`projecting`](BlockResolver::projecting) draws such a block the way a plan sees
//! it from above instead (#1609). From that INSERT down, each level's map is the full 3D one (the
//! arbitrary axis algorithm, the INSERT's elevation and Z scale), and a leaf's map to the plan is
//! that map with z dropped. In a tilted plane the map is still invertible on the leaf's own plane,
//! so a curve keeps an exact affine image. In a vertical plane every curve is seen edge on: its
//! image is the segment it covers, which is written as a line in plan coordinates, so PVec's 2x3
//! maps never need to be singular. A curve that is a point from above, and anything that is not a
//! line, arc, circle, ellipse or polyline, has nothing to draw and is listed in
//! [`Flattened::unplaced`].
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

use pvec_geometry::Curve2d;
use thiserror::Error;

use crate::cad::ItemHandle;

use super::fidelity::FidelityClass;
use super::limits::{Limit, LimitError, SourceGuard, SourceLimits};
use super::normalized::ProvenanceHandle;
use super::provenance::{Provenance, ProvenanceSink};

use super::provider::{
    BlockRef, DwgProvider, EntityCursor, EntityScope, InstancePath, InstanceStep, Point3,
    ProviderError, RawEntity, RawGeometry, z_axis_flip,
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

impl FlattenBounds {
    /// Bounds taking their [`SourceLimits`] from `profile`, with the default per-block ceiling.
    #[must_use]
    pub fn from_profile(profile: &super::view_profile::ViewProfile) -> Self {
        Self::new(profile.limits, Self::default().max_entities_per_block)
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
    #[error(
        "block `{block}` at {block_path} holds more than {ceiling} entities (observed {observed})"
    )]
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
    /// The INSERT places an external reference: its geometry lives in another file.
    ExternalReference,
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
            Self::ExternalReference => "external-reference",
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
    edge_on: bool,
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

    /// This leaf with `outer` composed after its map, so `outer` applies last: a viewport's map
    /// onto the sheet (#1563). The matrix stays private, so composing is done here.
    #[cfg(feature = "pvec-convert")]
    #[must_use]
    pub(crate) fn through(&self, outer: [f64; 6]) -> FlatEntity {
        let [p, q, r, s, u, v] = outer;
        let [a, b, c, d, tx, ty] = self.matrix;
        FlatEntity {
            entity: self.entity.clone(),
            matrix: [
                p * a + q * c,
                p * b + q * d,
                r * a + s * c,
                r * b + s * d,
                p * tx + q * ty + u,
                r * tx + s * ty + v,
            ],
            lineage: self.lineage,
            edge_on: self.edge_on,
        }
    }

    /// Whether this leaf is a curve of a block seen edge on, rewritten as the segment it projects
    /// to in plan coordinates (#1609). Only a [`projecting`](BlockResolver::projecting) resolver
    /// makes one.
    #[must_use]
    pub const fn edge_on(&self) -> bool {
        self.edge_on
    }
}

impl FlatEntity {
    /// A leaf put back together from what a compact store kept of it (#1726): it has no lineage, as
    /// a leaf of a walk that records none does not.
    #[cfg(feature = "pvec-convert")]
    #[must_use]
    pub(crate) const fn assembled(
        entity: RawEntity,
        matrix: [f64; 6],
        edge_on: bool,
    ) -> FlatEntity {
        FlatEntity {
            entity,
            matrix,
            lineage: None,
            edge_on,
        }
    }

    /// This leaf's map and lineage over another entity: what stands in for it when it is carried by
    /// entities of its own (#1609).
    #[cfg(feature = "pvec-convert")]
    #[must_use]
    pub(crate) fn carrying(&self, entity: RawEntity) -> FlatEntity {
        FlatEntity {
            entity,
            matrix: self.matrix,
            lineage: self.lineage,
            edge_on: self.edge_on,
        }
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

/// A leaf of a block out of the plan's plane that has no image in the plan, with why.
#[derive(Debug, Clone)]
pub struct UnplacedLeaf {
    /// The leaf as the provider reported it, its provenance carrying the full block path.
    pub entity: RawEntity,
    /// Why nothing is drawn for it.
    pub why: &'static str,
}

/// What one flatten run produced.
#[derive(Debug, Clone, Default)]
pub struct Flattened {
    /// The leaves, in the deterministic depth-first order of the walk.
    pub entities: Vec<FlatEntity>,
    /// The INSERTs that were not expanded.
    pub skipped: Vec<SkippedInsert>,
    /// The leaves of blocks out of the plan's plane that a [`projecting`](BlockResolver::projecting)
    /// resolver met and could not draw. Always empty otherwise.
    pub unplaced: Vec<UnplacedLeaf>,
    /// How many block instances were opened (a MINSERT counts rows times columns).
    pub instances: u64,
    /// Bytes the guard charged for the run, always within the configured allocation ceiling.
    pub allocated_bytes: u64,
}

/// Walks INSERTs through a provider and flattens them.
pub struct BlockResolver<'a, P: DwgProvider + ?Sized> {
    provider: &'a P,
    bounds: FlattenBounds,
    project: bool,
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
        Self {
            provider,
            bounds,
            project: false,
        }
    }

    /// This resolver, drawing a block whose normal is not the Z axis as the plan sees it when `on`,
    /// instead of skipping its INSERT (see the module docs, #1609).
    ///
    /// Leaves placed under such an INSERT are not sent to the provenance sink and carry no
    /// lineage (`FlatEntity::lineage` is `None`), so a caller with a real `ProvenanceSink` gets
    /// no records for them. The conversion passes `NoProvenance`, so nothing is lost there.
    #[must_use]
    pub fn projecting(mut self, on: bool) -> Self {
        self.project = on;
        self
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
        // An xref block record owns no entities, so expanding it would be silent. A provider
        // that cannot list them leaves the set empty, as before.
        let xrefs = self
            .provider
            .xref_blocks()
            .map(|v| {
                v.into_iter()
                    .map(|x| x.block_name.to_ascii_lowercase())
                    .collect()
            })
            .unwrap_or_default();
        let mut walk = Walk {
            project: self.project,
            xrefs,
            guard: SourceGuard::new(self.bounds.limits),
            sink,
            lineage,
            source,
            layout,
            out: Flattened::default(),
        };
        let root = self.open(scope).map_err(FlattenError::Provider)?;
        let mut stack = vec![Frame::Cursor {
            cursor: root,
            composed: ComposedTransform::identity(),
            plan: None,
            path: InstancePath::root(),
            block: None,
            seen: 0,
        }];
        while let Some(top) = stack.last_mut() {
            match top {
                Frame::Cursor {
                    cursor,
                    composed,
                    plan,
                    path,
                    block,
                    seen,
                } => {
                    let Some(mut entity) = cursor.next_entity().map_err(FlattenError::Provider)?
                    else {
                        stack.pop();
                        continue;
                    };
                    *seen += 1;
                    if let Some(name) = block.as_deref()
                        && *seen > self.bounds.max_entities_per_block
                    {
                        return Err(FlattenError::EntitiesPerBlock {
                            block: name.to_string(),
                            observed: *seen,
                            ceiling: self.bounds.max_entities_per_block,
                            block_path: path.clone(),
                        });
                    }
                    walk.guard
                        .check_entity(&entity)
                        .map_err(|e| limit_error(e, path, path.depth()))?;
                    if let RawGeometry::Insert(r) = &entity.geometry {
                        let (parent, parent_path) = (composed.clone(), path.clone());
                        if let Some(grid) = walk.enter(&entity, r, parent, *plan, &parent_path)? {
                            stack.push(grid);
                        }
                    } else if let Some(g) = plan {
                        entity.provenance.block_path = path.clone();
                        match place(&entity, g) {
                            Ok((entity, matrix, edge_on)) => walk.out.entities.push(FlatEntity {
                                entity,
                                matrix,
                                lineage: None,
                                edge_on,
                            }),
                            Err(why) => walk.out.unplaced.push(UnplacedLeaf { entity, why }),
                        }
                    } else {
                        walk.leaf(entity, composed, path)?;
                    }
                }
                Frame::Grid(g) => {
                    if g.next >= g.cells {
                        stack.pop();
                        walk.guard.exit_block();
                        continue;
                    }
                    let index = g.next;
                    g.next += 1;
                    let (composed, plan, cell_path) = g.cell(index)?;
                    let name = g.name.clone();
                    let handle = g.handle;
                    let scope = EntityScope::BlockDefinition {
                        name: name.clone(),
                        reached_via: cell_path.clone(),
                    };
                    match self.open(&scope) {
                        Ok(cursor) => {
                            walk.out.instances += 1;
                            stack.push(Frame::Cursor {
                                cursor,
                                composed,
                                plan,
                                path: cell_path,
                                block: Some(name),
                                seen: 0,
                            });
                        }
                        Err(ProviderError::NoSuchScope(_)) => {
                            let reason = if name.starts_with('*') {
                                SkipReason::AnonymousBlockUnresolved
                            } else {
                                SkipReason::BlockNotFound
                            };
                            let class = match reason {
                                SkipReason::AnonymousBlockUnresolved => {
                                    FidelityClass::RequiresExtension
                                }
                                _ => FidelityClass::Unsupported,
                            };
                            walk.out.skipped.push(SkippedInsert {
                                block_path: cell_path,
                                block_name: name,
                                insert_handle: Some(handle),
                                class,
                                reason,
                            });
                            stack.pop();
                            walk.guard.exit_block();
                        }
                        Err(e) => return Err(FlattenError::Provider(e)),
                    }
                }
            }
        }
        walk.out.allocated_bytes = walk.guard.allocated_bytes();
        Ok(walk.out)
    }

    fn open(&self, scope: &EntityScope) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        self.provider.entities(scope)
    }
}

/// A map from a block's 3D coordinates to the plan: rows x and y over `[x, y, z, 1]`.
type PlanMap = [[f64; 4]; 2];

enum Frame<'a> {
    /// One block instance (or the root scope) being read.
    Cursor {
        cursor: Box<dyn EntityCursor + 'a>,
        composed: ComposedTransform,
        /// The map from this block's 3D coordinates to the plan, rows x and y, once an INSERT out
        /// of the plan's plane was met above. `composed` stops at that INSERT.
        plan: Option<PlanMap>,
        path: InstancePath,
        block: Option<String>,
        seen: u64,
    },
    /// One INSERT, stepping through its grid cells.
    Grid(GridFrame),
}

struct GridFrame {
    name: String,
    handle: ItemHandle,
    parent: ComposedTransform,
    parent_path: InstancePath,
    coefficients: [f64; 6],
    mirror_x: bool,
    /// The parent's map to the plan, for an INSERT met out of the plan's plane or below one.
    plan: Option<PlanMap>,
    normal: Point3,
    elevation: f64,
    z_scale: f64,
    columns: u64,
    row_spacing: f64,
    column_spacing: f64,
    cells: u64,
    next: u64,
}

impl GridFrame {
    /// The composed map, the map to the plan out of the plan's plane, and the block path of one
    /// cell. Cells run row-major.
    fn cell(
        &self,
        index: u64,
    ) -> Result<(ComposedTransform, Option<PlanMap>, InstancePath), FlattenError> {
        let mut m = self.coefficients;
        if self.cells > 1 {
            let (row, col) = (index / self.columns, index % self.columns);
            // Spacings run along the INSERT's own (rotated) axes, which are the unit directions of
            // the linear part's two columns.
            // Basic arithmetic and `sqrt` only: libm's `hypot` gives another last bit on x86_64
            // and on arm64, and this result goes into a stored translation (#1471).
            let unit = |x: f64, y: f64| {
                let m = x.abs().max(y.abs());
                if m > 0.0 && m.is_finite() {
                    let (sx, sy) = (x / m, y / m);
                    let n = (sx * sx + sy * sy).sqrt();
                    (sx / n, sy / n)
                } else {
                    (0.0, 0.0)
                }
            };
            let (ux, uy) = unit(m[0], m[2]);
            let (vx, vy) = unit(m[1], m[3]);
            #[allow(clippy::cast_precision_loss)]
            let (dc, dr) = (
                col as f64 * self.column_spacing,
                row as f64 * self.row_spacing,
            );
            m[4] += dc * ux + dr * vx;
            m[5] += dc * uy + dr * vy;
        }
        let path = self.parent_path.child(self.name.clone(), self.handle);
        if let Some(g) = self.plan {
            // OCS to the container through the arbitrary axis algorithm, then the parent's map.
            let (ax, ay) = crate::vector::tessellate::arbitrary_axes(self.normal);
            let n = [
                ax[1] * ay[2] - ax[2] * ay[1],
                ax[2] * ay[0] - ax[0] * ay[2],
                ax[0] * ay[1] - ax[1] * ay[0],
            ];
            let local = [
                [m[0], m[1], 0.0, m[4]],
                [m[2], m[3], 0.0, m[5]],
                [0.0, 0.0, self.z_scale, self.elevation],
            ];
            let mut out = [[0.0; 4]; 2];
            for (row, g) in out.iter_mut().zip(g) {
                // What one unit along each OCS axis moves the parent's row by.
                let along = [ax, ay, n].map(|v| g[0] * v[0] + g[1] * v[1] + g[2] * v[2]);
                for (j, x) in row.iter_mut().enumerate() {
                    *x = (0..3).map(|k| along[k] * local[k][j]).sum::<f64>();
                }
                row[3] += g[3];
            }
            return Ok((self.parent.clone(), Some(out), path));
        }
        if self.mirror_x {
            m = [-m[0], -m[1], m[2], m[3], -m[4], m[5]];
        }
        let step = InstanceStep {
            block_name: self.name.clone(),
            insert_handle: self.handle,
        };
        let composed = self
            .parent
            .push_inner_coefficients(step, m)
            .map_err(FlattenError::Compose)?;
        Ok((composed, None, path))
    }
}

/// What a leaf below an INSERT out of the plan's plane draws, under its map `g` to the plan: the
/// entity to classify, the 2x3 map that goes with it, and whether it is seen edge on. `Err` says
/// why there is nothing to draw.
fn place(entity: &RawEntity, g: &PlanMap) -> Result<(RawEntity, [f64; 6], bool), &'static str> {
    const POINT: &str = "seen edge on from above, it is a point";
    const NOT_A_CURVE: &str =
        "only its lines, arcs, circles, ellipses and polylines have an image in the plan";
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let policy = pvec_numeric::Policy::V1;
    let [r0, r1] = g;
    let (a, b, c, d) = (r0[0], r0[1], r1[0], r1[1]);
    // Whether the plan sees the block edge on is PVec's singularity rule (NP-19), and whether a
    // segment is a point is its length floor (NP-06): no tolerance of this crate's own.
    let flat = matches!(
        policy.validate_affine([a, b, c, d, 0.0, 0.0]),
        Err(pvec_numeric::NumericError::SingularTransform)
    );
    let point = |p: [f64; 2], q: [f64; 2]| {
        let mag = [p[0], p[1], q[0], q[1]]
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let len = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
        policy.degenerate_length(len, mag)
    };
    let at = |p: Point3| {
        [
            r0[0] * p[0] + r0[1] * p[1] + r0[2] * p[2] + r0[3],
            r1[0] * p[0] + r1[1] * p[1] + r1[2] * p[2] + r1[3],
        ]
    };
    let segment = |p: [f64; 2], q: [f64; 2]| {
        let mut e = entity.clone();
        e.geometry = RawGeometry::Line {
            start: [p[0], p[1], 0.0],
            end: [q[0], q[1], 0.0],
        };
        e
    };
    // A line is its two ends under any map, so even a 3D one is exact.
    if let RawGeometry::Line { start, end } = entity.geometry {
        let (p, q) = (at(start), at(end));
        if point(p, q) {
            return Err(POINT);
        }
        return Ok((segment(p, q), IDENTITY, flat));
    }
    let basic = super::dwg_basic::extract(entity).map_err(|_| NOT_A_CURVE)?;
    let z = basic.elevation;
    let shift = [r0[2] * z + r0[3], r1[2] * z + r1[3]];
    if !flat {
        return Ok((entity.clone(), [a, b, c, d, shift[0], shift[1]], false));
    }
    // Rank one: both rows are multiples of the longer one, `w`, so the plan point is `dir` times
    // `w . p` plus the shift, and the curve covers the span of `w . p` along its length.
    let (w, dir) = if a * a + b * b >= c * c + d * d {
        let k = (a * c + b * d) / (a * a + b * b);
        ([a, b], [1.0, k])
    } else {
        let k = (a * c + b * d) / (c * c + d * d);
        ([c, d], [k, 1.0])
    };
    let len = (w[0] * w[0] + w[1] * w[1]).sqrt();
    if len.is_nan() || len == 0.0 {
        return Err(POINT);
    }
    let (ux, uy) = (w[0] / len, w[1] / len);
    let turned = pvec_geometry::Transform2::new([ux, uy, -uy, ux, 0.0, 0.0])
        .and_then(|turn| basic.curve.transform(&turn))
        .and_then(|(turned, _)| turned.bounds());
    let Ok(turned) = turned else {
        return Err(NOT_A_CURVE);
    };
    let (lo, hi) = (turned.min().x * len, turned.max().x * len);
    let p = [shift[0] + dir[0] * lo, shift[1] + dir[1] * lo];
    let q = [shift[0] + dir[0] * hi, shift[1] + dir[1] * hi];
    if point(p, q) {
        return Err(POINT);
    }
    Ok((segment(p, q), IDENTITY, true))
}

struct Walk<'s, S: ProvenanceSink> {
    /// Whether an INSERT out of the plan's plane is drawn as the plan sees it.
    project: bool,
    /// Lowercased names of the external-reference block records.
    xrefs: std::collections::BTreeSet<String>,
    guard: SourceGuard,
    sink: &'s mut S,
    lineage: &'s mut LineageStore,
    source: &'s str,
    layout: &'s str,
    out: Flattened,
}

impl<S: ProvenanceSink> Walk<'_, S> {
    /// Meets an INSERT: either a grid frame to push, or a recorded skip.
    fn enter<'a>(
        &mut self,
        entity: &RawEntity,
        r: &BlockRef,
        parent: ComposedTransform,
        parent_plan: Option<PlanMap>,
        parent_path: &InstancePath,
    ) -> Result<Option<Frame<'a>>, FlattenError> {
        let skip = |walk: &mut Self, reason: SkipReason, class: FidelityClass| {
            walk.out.skipped.push(SkippedInsert {
                block_path: match entity.provenance.handle {
                    Some(hd) => parent_path.child(r.block_name.clone(), hd),
                    None => parent_path.clone(),
                },
                block_name: r.block_name.clone(),
                insert_handle: entity.provenance.handle,
                class,
                reason,
            });
        };
        if self.xrefs.contains(&r.block_name.to_ascii_lowercase()) {
            skip(
                self,
                SkipReason::ExternalReference,
                FidelityClass::Unsupported,
            );
            return Ok(None);
        }
        let Some(handle) = entity.provenance.handle else {
            skip(
                self,
                SkipReason::MissingInsertHandle,
                FidelityClass::Unsupported,
            );
            return Ok(None);
        };
        let mirror_x = z_axis_flip(r.normal);
        // Once a level is out of the plan's plane, every level below it is placed in 3D.
        let plan = match (parent_plan, mirror_x) {
            (Some(g), _) => Some(g),
            (None, Some(_)) => None,
            (None, None) if self.project => {
                let [a, b, c, d, tx, ty] = parent.matrix();
                Some([[a, b, 0.0, tx], [c, d, 0.0, ty]])
            }
            (None, None) => {
                skip(
                    self,
                    SkipReason::NormalNotZ,
                    FidelityClass::RequiresExtension,
                );
                return Ok(None);
            }
        };
        let child = parent_path.child(r.block_name.clone(), handle);
        self.guard
            .enter_block(&r.block_name, r.grid)
            .map_err(|e| limit_error(e, &child, child.depth()))?;
        let cells = u64::from(r.grid.rows).saturating_mul(u64::from(r.grid.columns));
        Ok(Some(Frame::Grid(GridFrame {
            name: r.block_name.clone(),
            handle,
            parent,
            parent_path: parent_path.clone(),
            coefficients: r.transform.coefficients(),
            mirror_x: mirror_x.unwrap_or(false),
            plan,
            normal: r.normal,
            elevation: r.elevation,
            z_scale: r.z_scale,
            columns: u64::from(r.grid.columns).max(1),
            row_spacing: r.grid.row_spacing,
            column_spacing: r.grid.column_spacing,
            cells,
            next: 0,
        })))
    }

    fn leaf(
        &mut self,
        mut entity: RawEntity,
        composed: &ComposedTransform,
        path: &InstancePath,
    ) -> Result<(), FlattenError> {
        entity.provenance.block_path = path.clone();
        let transform = composed.source_transform();
        let (source, layout) = (self.source, self.layout);
        let stub = entity.provenance.clone();
        let handle = self
            .sink
            .record(|| Provenance::from_stub(&stub, source, layout, transform));
        if let Some(handle) = handle {
            let levels = composed.lineage().to_vec();
            let bytes = u64::try_from(
                levels
                    .len()
                    .saturating_mul(std::mem::size_of::<LineageLevel>()),
            )
            .unwrap_or(u64::MAX);
            self.guard
                .charge(bytes)
                .map_err(|e| limit_error(e, path, path.depth()))?;
            self.lineage
                .records
                .insert(handle.get(), TransformLineage { levels });
        }
        self.out.entities.push(FlatEntity {
            entity,
            matrix: composed.matrix(),
            lineage: handle,
            edge_on: false,
        });
        Ok(())
    }
}

fn limit_error(e: LimitError, path: &InstancePath, depth: usize) -> FlattenError {
    match e {
        LimitError::Exceeded {
            limit,
            observed,
            ceiling,
        } => FlattenError::Limit {
            limit,
            observed,
            ceiling,
            block_path: path.clone(),
        },
        LimitError::BlockCycle { .. } => FlattenError::BlockRecursionLimit {
            depth: u32::try_from(depth).unwrap_or(u32::MAX),
            block_path: path.clone(),
        },
    }
}
