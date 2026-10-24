//! Exact B-spline extraction from the provider boundary (N4.3, #1074).
//!
//! A `SPLINE` is where a CAD ingest usually loses its mathematics: a decoder that
//! hands back sampled points has already destroyed the parameters PVec needs. This
//! module takes a [`RawEntity`] whose geometry is a [`RawGeometry::Spline`] and
//! returns the whole description as authored, or a typed refusal. Nothing here
//! samples, flattens or fits.

use std::fmt;

use pvec_numeric::{MAX_CONTROL_POINTS, MAX_DEGREE, MAX_KNOTS, NumericError, Policy};

use super::provider::{Point3, RawEntity, RawGeometry, SourceFact};

const POLICY: Policy = Policy::V1;

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
    /// The weights.
    Weights,
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
///
/// A `HELIX` is recognised by the drawing's class name. A spline is planar when every
/// control point has the same z (`-0.0` and `0.0` count as the same plane), and 3D otherwise.
#[must_use]
pub fn classify(entity: &RawEntity) -> SplineClass {
    if entity.provenance.class_name.eq_ignore_ascii_case("HELIX") {
        return SplineClass::Helix;
    }
    match &entity.geometry {
        RawGeometry::Spline { control_points, .. } => {
            let flat = control_points.windows(2).all(|w| w[0][2] == w[1][2]);
            if flat {
                SplineClass::Planar
            } else {
                SplineClass::ThreeD
            }
        }
        _ => SplineClass::NotSpline,
    }
}

fn missing(field: SplineField) -> SplineError {
    SplineError::MissingField {
        field,
        requires: Requires::Extension,
    }
}

/// Extracts the exact description of a spline entity, or refuses.
///
/// The checks run in an order where nothing is sized by an unchecked number: counts
/// against PVec's limits first, then the shape of the knot vector, then the weights,
/// then PVec's own rules for what is left (repeat limits, empty or ambiguous domain,
/// weight floor). A knot vector, a weight or a flag is never filled in, rebuilt from
/// points or rounded.
///
/// Periodic and closed splines come back with the flags set and the control points and
/// knots exactly as the drawing stores them. Turning a periodic spline into PVec's
/// unwrapped form is the codec side's call, and it needs the flag to make it.
///
/// # Errors
/// See [`SplineError`].
pub fn extract(entity: &RawEntity) -> Result<ExactSpline, SplineError> {
    match classify(entity) {
        SplineClass::NotSpline => return Err(SplineError::NotSpline),
        class @ SplineClass::Helix => {
            return Err(SplineError::Unsupported3d {
                class,
                requires: Requires::PVecCapability,
            });
        }
        SplineClass::Planar | SplineClass::ThreeD => {}
    }
    let RawGeometry::Spline {
        degree,
        control_points,
        knots,
        weights,
        periodic,
        closed,
    } = &entity.geometry
    else {
        return Err(SplineError::NotSpline);
    };

    // A coordinate that is not a number is reported as that, not as a 3D spline.
    if let Some(index) = control_points
        .iter()
        .position(|q| q.iter().any(|c| !c.is_finite()))
    {
        return Err(SplineError::NonFiniteControlPoint { index });
    }
    if classify(entity) == SplineClass::ThreeD {
        return Err(SplineError::Unsupported3d {
            class: SplineClass::ThreeD,
            requires: Requires::PVecCapability,
        });
    }

    // Facts the provider could not or did not give.
    let knots = knots.present().ok_or(missing(SplineField::Knots))?;
    let periodic = *periodic.present().ok_or(missing(SplineField::Periodic))?;

    // Counts and degree, before anything is sized by them.
    let n = control_points.len();
    if n > MAX_CONTROL_POINTS {
        return Err(SplineError::TooManyControlPoints {
            limit: MAX_CONTROL_POINTS,
            observed: n,
        });
    }
    if knots.len() > MAX_KNOTS {
        return Err(SplineError::TooManyKnots {
            limit: MAX_KNOTS,
            observed: knots.len(),
        });
    }
    let p = *degree as usize;
    if !(1..=MAX_DEGREE).contains(&p) {
        return Err(SplineError::DegreeOutOfRange {
            degree: *degree,
            limit: MAX_DEGREE,
        });
    }
    if n < p + 1 {
        return Err(SplineError::TooFewControlPoints {
            observed: n,
            needed: p + 1,
        });
    }
    if knots.len() != n + p + 1 {
        return Err(SplineError::KnotCount {
            expected: n + p + 1,
            observed: knots.len(),
        });
    }

    // Values.
    if let Some(index) = knots.iter().position(|k| !k.is_finite()) {
        return Err(SplineError::NonFiniteKnot { index });
    }
    if let Some(i) = knots.windows(2).position(|w| w[1] < w[0]) {
        return Err(SplineError::KnotsDecrease { index: i + 1 });
    }
    let form = match weights {
        SourceFact::NotReportable => return Err(missing(SplineField::Weights)),
        SourceFact::Absent => SplineForm::NonRational,
        SourceFact::Present(w) => {
            if w.len() != n {
                return Err(SplineError::WeightCount {
                    expected: n,
                    observed: w.len(),
                });
            }
            if let Some(index) = w.iter().position(|x| !x.is_finite() || *x <= 0.0) {
                return Err(SplineError::BadWeight {
                    index,
                    value: w[index],
                });
            }
            POLICY.validate_weights(w).map_err(SplineError::Policy)?;
            SplineForm::Rational(w.clone())
        }
    };
    let (lo, hi) = POLICY
        .validate_knots(p, n, knots)
        .map_err(SplineError::Policy)?;

    Ok(ExactSpline {
        degree: *degree,
        control_points: control_points.clone(),
        knots: knots.clone(),
        form,
        domain: ParamDomain { lo, hi },
        trim: TrimState::FullDomain,
        closed: *closed,
        periodic,
    })
}
