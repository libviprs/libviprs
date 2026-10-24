//! Resource limits for CAD ingestion (issue #1101, N3.6).
//!
//! A drawing is hostile input. The crate already bounds the other formats it reads
//! (`pdf::DEFAULT_MAX_RENDER_PIXELS`, `MAX_PDF_DIM`, the 64 level `/Parent` walk, the
//! varint length cap in `pmtiles`), and this is the same treatment for the provider
//! boundary: [`SourceLimits`] is the set, [`SourceGuard`] is the meter a consumer
//! feeds as it walks a drawing, and [`LimitError`] is the typed refusal. Every
//! breach names the limit and the observed value. Nothing here panics, and nothing
//! truncates: a consumer that gets an error stops, it does not carry on with a
//! prefix of the drawing.
//!
//! The guard is deliberately free of any reader type. It looks at [`RawEntity`] and
//! at block names, so it works the same behind any [`DwgProvider`](super::DwgProvider).
//! N4.4 (flattening) drives `enter_block` and `exit_block` as it descends and
//! `check_entity` on every leaf. PDF limits are `PdfVectorLimits` and stay there.

use thiserror::Error;

use super::provider::{ArrayGrid, RawEntity};

/// The ceilings for one conversion.
///
/// The defaults are backstops against crafted input, not tight bounds: a real
/// drawing, including a large site plan, stays far below all of them. Raise one
/// when a legitimate input trips it, or use [`SourceLimits::raised`] to switch the
/// checks to their loosest values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceLimits {
    /// Deepest INSERT nesting. Layout level entities are depth 0. 64 matches the
    /// `/Parent` walk cap in `pdf`, and real drawings nest a handful of levels.
    pub max_block_depth: u32,
    /// Most entities one conversion may look at. 20 million: a city scale survey
    /// has a few million, and each entity costs at least a few hundred bytes
    /// downstream, so this is already several GiB.
    pub max_entities: u64,
    /// Most vertices in one polyline. 2^20, a polyline longer than that is a
    /// point cloud and belongs somewhere else.
    pub max_polyline_vertices: usize,
    /// Most control points in one spline. 2^20, as for polylines.
    pub max_spline_control_points: usize,
    /// Longest knot vector (and weight array) of one spline. A valid vector has
    /// control points plus degree plus one knots, so this is the control point
    /// ceiling with 64 spare for the degree.
    pub max_spline_knots: usize,
    /// Most instances one INSERT may place (MINSERT rows times columns). The file
    /// format allows 32767 by 32767, over a billion, from a few bytes.
    pub max_insert_grid: u64,
    /// Most block instances one conversion may expand in total, counting MINSERT
    /// grids and multiplying through nesting. This catches the input that is legal
    /// at every level (ten INSERTs of ten INSERTs of ...) and unbounded together.
    pub max_expanded_instances: u64,
    /// Estimated bytes of entity data one conversion may hold or produce. 2 GiB,
    /// half of what the raster ceiling in `pdf` allows.
    pub max_allocation_bytes: u64,
    /// Longest layer, block or class name, in bytes. DWG itself caps names at 255
    /// characters, so a longer one is not from a conforming writer.
    pub max_name_bytes: usize,
}

impl Default for SourceLimits {
    fn default() -> Self {
        Self {
            max_block_depth: 64,
            max_entities: 20_000_000,
            max_polyline_vertices: 1 << 20,
            max_spline_control_points: 1 << 20,
            max_spline_knots: (1 << 20) + 64,
            max_insert_grid: 1_000_000,
            max_expanded_instances: 10_000_000,
            max_allocation_bytes: 2 << 30,
            max_name_bytes: 1020,
        }
    }
}

impl SourceLimits {
    /// Every ceiling at its loosest. For a trusted source, and for the test that a
    /// conversion inside the defaults is unaffected by the limits.
    #[must_use]
    pub const fn raised() -> Self {
        Self {
            max_block_depth: u32::MAX,
            max_entities: u64::MAX,
            max_polyline_vertices: usize::MAX,
            max_spline_control_points: usize::MAX,
            max_spline_knots: usize::MAX,
            max_insert_grid: u64::MAX,
            max_expanded_instances: u64::MAX,
            max_allocation_bytes: u64::MAX,
            max_name_bytes: usize::MAX,
        }
    }
}

/// Which limit tripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Limit {
    /// [`SourceLimits::max_block_depth`].
    BlockDepth,
    /// [`SourceLimits::max_entities`].
    Entities,
    /// [`SourceLimits::max_polyline_vertices`].
    PolylineVertices,
    /// [`SourceLimits::max_spline_control_points`].
    SplineControlPoints,
    /// [`SourceLimits::max_spline_knots`], knots.
    SplineKnots,
    /// [`SourceLimits::max_spline_knots`], weights.
    SplineWeights,
    /// [`SourceLimits::max_insert_grid`].
    InsertGrid,
    /// [`SourceLimits::max_expanded_instances`].
    ExpandedInstances,
    /// [`SourceLimits::max_allocation_bytes`].
    AllocationBytes,
    /// [`SourceLimits::max_name_bytes`].
    NameBytes,
}

impl Limit {
    /// The name of the [`SourceLimits`] field behind it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BlockDepth => "max_block_depth",
            Self::Entities => "max_entities",
            Self::PolylineVertices => "max_polyline_vertices",
            Self::SplineControlPoints => "max_spline_control_points",
            Self::SplineKnots => "max_spline_knots",
            Self::SplineWeights => "max_spline_knots (weights)",
            Self::InsertGrid => "max_insert_grid",
            Self::ExpandedInstances => "max_expanded_instances",
            Self::AllocationBytes => "max_allocation_bytes",
            Self::NameBytes => "max_name_bytes",
        }
    }
}

/// A refused source.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LimitError {
    /// A ceiling was passed.
    #[error("limit {} exceeded: observed {observed}, ceiling {ceiling}", .limit.name())]
    Exceeded {
        /// Which one.
        limit: Limit,
        /// The value the source asked for.
        observed: u64,
        /// The ceiling that was in force.
        ceiling: u64,
    },
    /// A block reaches itself through its own INSERTs.
    #[error("block cycle: {}", .chain.join(" -> "))]
    BlockCycle {
        /// The block that was entered a second time.
        block: String,
        /// The chain of block names from the first entry to the repeat, inclusive.
        chain: Vec<String>,
    },
}

/// The meter for one conversion.
#[derive(Debug, Clone)]
pub struct SourceGuard {
    limits: SourceLimits,
}

impl SourceGuard {
    /// A guard with nothing counted yet.
    #[must_use]
    pub fn new(limits: SourceLimits) -> Self {
        let _ = limits;
        todo!()
    }

    /// The limits in force.
    #[must_use]
    pub fn limits(&self) -> &SourceLimits {
        &self.limits
    }

    /// Entities counted so far.
    #[must_use]
    pub fn entities_seen(&self) -> u64 {
        todo!()
    }

    /// Estimated bytes charged so far.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        todo!()
    }

    /// Instances expanded so far.
    #[must_use]
    pub fn expanded_instances(&self) -> u64 {
        todo!()
    }

    /// The current block nesting depth.
    #[must_use]
    pub fn depth(&self) -> u32 {
        todo!()
    }

    /// Count one entity and check its sub-element counts, its names and the
    /// allocation it implies.
    ///
    /// # Errors
    ///
    /// [`LimitError::Exceeded`] naming the first limit passed.
    pub fn check_entity(&mut self, entity: &RawEntity) -> Result<(), LimitError> {
        let _ = entity;
        todo!()
    }

    /// Check a name taken from the source (a layer, a block).
    ///
    /// # Errors
    ///
    /// [`LimitError::Exceeded`] with [`Limit::NameBytes`].
    pub fn check_name(&self, name: &str) -> Result<(), LimitError> {
        let _ = name;
        todo!()
    }

    /// Charge `bytes` against the allocation ceiling.
    ///
    /// # Errors
    ///
    /// [`LimitError::Exceeded`] with [`Limit::AllocationBytes`].
    pub fn charge(&mut self, bytes: u64) -> Result<(), LimitError> {
        let _ = bytes;
        todo!()
    }

    /// Descend into block `name` through an INSERT that places `grid`.
    ///
    /// Checked in this order: the name, a cycle, the depth, the grid, the
    /// expanded instance total. On an error nothing is pushed, so the caller does
    /// not call [`exit_block`](Self::exit_block) for it.
    ///
    /// # Errors
    ///
    /// [`LimitError::BlockCycle`] or [`LimitError::Exceeded`].
    pub fn enter_block(&mut self, name: &str, grid: ArrayGrid) -> Result<(), LimitError> {
        let _ = (name, grid);
        todo!()
    }

    /// Leave the block entered last. A call with nothing entered does nothing.
    pub fn exit_block(&mut self) {
        todo!()
    }
}
