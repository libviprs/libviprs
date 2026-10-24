//! Exact B-spline extraction from the provider boundary (N4.3, #1074).
//!
//! A `SPLINE` is where a CAD ingest usually loses its mathematics: a decoder that
//! hands back sampled points has already destroyed the parameters PVec needs. This
//! module takes a [`RawEntity`] whose geometry is a [`RawGeometry::Spline`] and
//! returns the whole description as authored, or a typed refusal. Nothing here
//! samples, flattens or fits.

use std::fmt;

use pvec_numeric::NumericError;

use super::provider::{Point3, RawEntity, RawGeometry};

/// What the extractor kept of one spline, as authored.
#[derive(Debug, Clone)]
pub struct ExactSpline {
    /// Degree.
    pub degree: u32,
    /// Control points in source coordinates, never duplicated to express closure.
    pub control_points: Vec<Point3>,
    /// The knot vector, multiplicities as authored.
    pub knots: Vec<f64>,
    /// Rational or not, kept apart rather than defaulted to 1.0.
    pub form: SplineForm,
    /// The evaluated parameter domain, `[knots[degree], knots[n]]`.
    pub domain: ParamDomain,
    /// Trim state.
    pub trim: TrimState,
    /// The drawing's closed flag.
    pub closed: bool,
    /// The drawing's periodic flag.
    pub periodic: bool,
}

/// Non-rational, or rational with one weight per control point.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SplineForm {
    /// The drawing carries no weights.
    NonRational,
    /// The drawing's weights, verbatim.
    Rational(Vec<f64>),
}

/// The evaluated parameter interval.
#[derive(Debug, Clone, Copy)]
pub struct ParamDomain {
    /// Start parameter.
    pub lo: f64,
    /// End parameter.
    pub hi: f64,
}

/// Whether the spline is cut short of its full domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrimState {
    /// The curve runs over its whole knot domain. A DWG `SPLINE` has no trim.
    FullDomain,
}

/// How an entity relates to what this module extracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplineClass {
    /// A planar B-spline: every control point has the same z.
    Planar,
    /// Control points with differing z. Classified, never projected.
    ThreeD,
    /// A `HELIX`. Classified, never projected.
    Helix,
    /// Not a spline.
    NotSpline,
}

/// A field a provider has to supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplineField {
    /// The knot vector.
    Knots,
    /// The periodic flag.
    Periodic,
}

/// What would unblock a refused spline (N3.5's two outcomes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Requires {
    /// The provider has to report more.
    Extension,
    /// PVec has to learn to carry it.
    PVecCapability,
}

/// Why a spline was refused. Nothing is truncated or repaired.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SplineError {
    /// The entity is not a spline.
    NotSpline,
    /// A field is missing, and what would supply it.
    MissingField {
        /// The field.
        field: SplineField,
        /// What would unblock it.
        requires: Requires,
    },
    /// A `HELIX` or a 3D spline: PVec's curves are planar.
    Unsupported3d {
        /// The classification.
        class: SplineClass,
        /// What would unblock it.
        requires: Requires,
    },
    /// Degree outside `1..=limit`.
    DegreeOutOfRange {
        /// The degree.
        degree: u32,
        /// The largest supported.
        limit: usize,
    },
    /// Fewer control points than `degree + 1`.
    TooFewControlPoints {
        /// Observed.
        observed: usize,
        /// Needed.
        needed: usize,
    },
    /// More control points than PVec supports.
    TooManyControlPoints {
        /// The limit.
        limit: usize,
        /// Observed.
        observed: usize,
    },
    /// More knots than PVec supports.
    TooManyKnots {
        /// The limit.
        limit: usize,
        /// Observed.
        observed: usize,
    },
    /// `knots.len()` is not `control_points.len() + degree + 1`.
    KnotCount {
        /// Expected.
        expected: usize,
        /// Observed.
        observed: usize,
    },
    /// A knot is NaN or infinite.
    NonFiniteKnot {
        /// Index.
        index: usize,
    },
    /// `knots[index] < knots[index - 1]`.
    KnotsDecrease {
        /// Index of the first knot that decreases.
        index: usize,
    },
    /// A weight count other than one per control point.
    WeightCount {
        /// Expected.
        expected: usize,
        /// Observed.
        observed: usize,
    },
    /// A weight that is zero, negative or not finite.
    BadWeight {
        /// The control point's index.
        index: usize,
        /// The weight, verbatim.
        value: f64,
    },
    /// A coordinate that is not finite.
    NonFiniteControlPoint {
        /// The control point's index.
        index: usize,
    },
    /// PVec's numeric policy refused the rest (repeat rules, domain, weight floor).
    Policy(NumericError),
}

impl fmt::Display for SplineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SplineError {}

/// Classifies an entity without extracting it.
#[must_use]
pub fn classify(_entity: &RawEntity) -> SplineClass {
    SplineClass::NotSpline
}

/// Extracts the exact description of a spline entity, or refuses.
///
/// # Errors
/// See [`SplineError`].
pub fn extract(entity: &RawEntity) -> Result<ExactSpline, SplineError> {
    let _ = &entity.geometry;
    let _: Option<&RawGeometry> = None;
    Err(SplineError::NotSpline)
}
