//! Spatial bounds and entity-reference inputs for PVec's spatial directory (issue #1091, N6.4).
//!
//! PVec builds the directory (P5.1 to P5.3). What it needs from here is one box per entity, in
//! source space, and a way to say how much to trust each one. This module hands those over and
//! builds no index, directory or straddler structure.
//!
//! # Where the mathematics lives
//!
//! The issue asks for analytic bounds, never sampled ones. PVec's `Curve2d::bounds` already is
//! that: closed form for lines once trimmed, arcs, ellipses, polylines and Bezier curves up to
//! degree 3, and the control hull for splines and higher-degree Bezier curves, with nothing
//! sampled. Re-deriving the extrema here would be a second implementation of something PVec owns
//! (NG-01, NG-10 of `docs/adr/pvec-normalized-geometry.md`), so I delegate and add the one thing
//! PVec's `Bounds` does not carry: [`BoundsMethod`], recorded per entity.
//!
//! A hull is a valid conservative box. It is labelled [`BoundsMethod::HullUpperBound`] and is
//! never reported as exact. A spline whose degree admits a closed form still comes back as a hull
//! until PVec offers a tight box for it; the label stays truthful in the meantime.
//!
//! # Identity
//!
//! Entities are referred to by `pvec_entity::EntityId` (P3.6), which is PVec's. [`pvec_entity_id`]
//! maps the dense position `ordering` assigns onto it and refuses what PVec's entity limit does
//! not admit, so no id is made up here.
//!
//! # Finite checking
//!
//! A curve PVec holds is already finite and non-degenerate (P1.2 is applied on construction), and
//! a curve with no finite extent (a bare line) is refused with [`BoundsError::Unbounded`]. No
//! tolerance is defined in this module.
//!
//! The drawing-level box is the union of the entity boxes. It is georeference metadata for
//! N4.7 and N7.3 and is not applied to any geometry.

use std::fmt;

use pvec_entity::EntityId as PvecEntityId;
use pvec_geometry::{Bounds, Curve, Curve2d, GeometryError, Point2};

use super::ordering::EntityId;

/// The box PVec reports for a curve: enclosing, tight only when [`BoundsMethod::Exact`].
pub type SourceBounds = Bounds;

/// How a box was obtained. Recorded for every entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundsMethod {
    /// Closed form: the box is the tight extent of the curve.
    Exact,
    /// The control hull: encloses the curve and is valid for pruning, but is not tight.
    HullUpperBound,
}

impl BoundsMethod {
    /// The label used in reports.
    pub fn as_str(self) -> &'static str {
        match self {
            BoundsMethod::Exact => "exact",
            BoundsMethod::HullUpperBound => "hull-upper-bound",
        }
    }
}

/// Why a box could not be produced.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundsError {
    /// PVec refused the curve; the error is PVec's.
    Refused(GeometryError),
    /// The curve has no finite extent, so there is no box to hand over.
    Unbounded,
    /// The id does not fit PVec's entity limit.
    IdOutOfRange(u64),
}

impl fmt::Display for BoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundsError::Refused(e) => write!(f, "PVec refused the curve: {e:?}"),
            BoundsError::Unbounded => f.write_str("the curve has no finite extent"),
            BoundsError::IdOutOfRange(n) => {
                write!(f, "entity id {n} is outside PVec's entity limit")
            }
        }
    }
}

impl std::error::Error for BoundsError {}

impl From<GeometryError> for BoundsError {
    fn from(e: GeometryError) -> Self {
        match e {
            GeometryError::Unbounded => BoundsError::Unbounded,
            other => BoundsError::Refused(other),
        }
    }
}

/// One entity's box, its PVec id and how the box was obtained.
#[derive(Debug, Clone, Copy)]
pub struct EntityBounds {
    id: PvecEntityId,
    bounds: SourceBounds,
    method: BoundsMethod,
}

impl EntityBounds {
    /// The entity's file-local PVec id.
    pub fn id(&self) -> PvecEntityId {
        self.id
    }
    /// The source-space box.
    pub fn bounds(&self) -> &SourceBounds {
        &self.bounds
    }
    /// How the box was obtained.
    pub fn method(&self) -> BoundsMethod {
        self.method
    }
}

/// Every entity's box plus the drawing-level union.
#[derive(Debug, Clone, Default)]
pub struct BoundsReport {
    entries: Vec<EntityBounds>,
    drawing: Option<SourceBounds>,
}

impl BoundsReport {
    /// Entries in input order.
    pub fn entries(&self) -> &[EntityBounds] {
        &self.entries
    }
    /// The union of every entity box, `None` for no entities. Georeference metadata only.
    pub fn drawing_bounds(&self) -> Option<&SourceBounds> {
        self.drawing.as_ref()
    }
    /// How many entries carry an exact box.
    pub fn exact_count(&self) -> usize {
        self.count(BoundsMethod::Exact)
    }
    /// How many entries carry a hull box.
    pub fn hull_count(&self) -> usize {
        self.count(BoundsMethod::HullUpperBound)
    }
    /// The entries whose box is only a hull.
    pub fn hull_entries(&self) -> impl Iterator<Item = &EntityBounds> {
        self.entries
            .iter()
            .filter(|e| e.method == BoundsMethod::HullUpperBound)
    }

    fn count(&self, m: BoundsMethod) -> usize {
        self.entries.iter().filter(|e| e.method == m).count()
    }
}

/// Maps the dense position `ordering` assigned onto PVec's file-local id.
pub fn pvec_entity_id(id: EntityId) -> Result<PvecEntityId, BoundsError> {
    let raw = id.get();
    u32::try_from(raw)
        .ok()
        .and_then(|n| PvecEntityId::new(n).ok())
        .ok_or(BoundsError::IdOutOfRange(raw))
}

/// The largest Bezier degree PVec gives a closed-form box for.
const CLOSED_FORM_BEZIER_DEGREE: usize = 3;

fn method_of(curve: &Curve) -> BoundsMethod {
    let exact = match curve {
        Curve::Line(_) | Curve::Circle(_) | Curve::Ellipse(_) | Curve::Polyline(_) => true,
        Curve::Bezier(b) => b.degree() <= CLOSED_FORM_BEZIER_DEGREE,
        Curve::Trimmed(t) => matches!(
            t.basis(),
            Curve::Line(_) | Curve::Circle(_) | Curve::Ellipse(_)
        ),
        Curve::Composite(c) => c
            .segments()
            .iter()
            .all(|s| method_of(s) == BoundsMethod::Exact),
        Curve::BSpline(_) | Curve::RationalBSpline(_) => false,
        // A kind added to PVec later is a hull until it is classified here.
        _ => false,
    };
    if exact {
        BoundsMethod::Exact
    } else {
        BoundsMethod::HullUpperBound
    }
}

/// The box for one curve and how it was obtained. The box is PVec's, untouched.
pub fn curve_bounds(curve: &Curve) -> Result<(SourceBounds, BoundsMethod), BoundsError> {
    let b = curve.bounds()?;
    Ok((b, method_of(curve)))
}

fn union(a: &SourceBounds, b: &SourceBounds) -> Result<SourceBounds, BoundsError> {
    let lo = Point2::checked(a.min().x.min(b.min().x), a.min().y.min(b.min().y))?;
    let hi = Point2::checked(a.max().x.max(b.max().x), a.max().y.max(b.max().y))?;
    Ok(Bounds::new(lo, hi)?)
}

/// Boxes for a set of entities, each referred to by its PVec id, plus the drawing-level union.
pub fn bounds_inputs<'a>(
    items: impl IntoIterator<Item = (PvecEntityId, &'a Curve)>,
) -> Result<BoundsReport, BoundsError> {
    let mut report = BoundsReport::default();
    for (id, curve) in items {
        let (bounds, method) = curve_bounds(curve)?;
        report.drawing = Some(match &report.drawing {
            Some(d) => union(d, &bounds)?,
            None => bounds,
        });
        report.entries.push(EntityBounds { id, bounds, method });
    }
    Ok(report)
}
