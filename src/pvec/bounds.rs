//! Spatial bounds and entity references for PVec's spatial directory (issue #1091, N6.4).
//!
//! STUB: the signatures only, so the red tests compile and fail.

use pvec_entity::EntityId as PvecEntityId;
use pvec_geometry::{Bounds, Curve, GeometryError};


/// The box PVec reports for a curve.
pub type SourceBounds = Bounds;

/// How a box was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsMethod {
    /// Closed form, tight.
    Exact,
    /// Control hull: encloses the curve, not tight.
    HullUpperBound,
}

/// Why a box could not be produced.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundsError {
    /// Not implemented yet.
    NotImplemented,
    /// PVec refused.
    Refused(GeometryError),
    /// The curve has no finite extent.
    Unbounded,
    /// The id does not fit PVec's entity limit.
    IdOutOfRange(u64),
}

/// One entity's box, its PVec id and how the box was obtained.
#[derive(Debug, Clone, Copy)]
pub struct EntityBounds {
    id: PvecEntityId,
    bounds: SourceBounds,
    method: BoundsMethod,
}

impl EntityBounds {
    /// The PVec id.
    pub fn id(&self) -> PvecEntityId {
        self.id
    }
    /// The box.
    pub fn bounds(&self) -> &SourceBounds {
        &self.bounds
    }
    /// The method.
    pub fn method(&self) -> BoundsMethod {
        self.method
    }
}

/// All boxes plus the drawing-level union.
#[derive(Debug, Clone, Default)]
pub struct BoundsReport {
    entries: Vec<EntityBounds>,
    drawing: Option<SourceBounds>,
}

impl BoundsReport {
    /// Per-entity entries in input order.
    pub fn entries(&self) -> &[EntityBounds] {
        &self.entries
    }
    /// Union of every box.
    pub fn drawing_bounds(&self) -> Option<&SourceBounds> {
        self.drawing.as_ref()
    }
    /// Entries with an exact box.
    pub fn exact_count(&self) -> usize {
        0
    }
    /// Entries with a hull box.
    pub fn hull_count(&self) -> usize {
        0
    }
}

/// The box for one curve and how it was obtained.
pub fn curve_bounds(_curve: &Curve) -> Result<(SourceBounds, BoundsMethod), BoundsError> {
    Err(BoundsError::NotImplemented)
}

/// Boxes for a set of entities.
pub fn bounds_inputs<'a>(
    _items: impl IntoIterator<Item = (PvecEntityId, &'a Curve)>,
) -> Result<BoundsReport, BoundsError> {
    Err(BoundsError::NotImplemented)
}
