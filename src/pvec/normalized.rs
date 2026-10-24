//! `Normalized2DGeometry`: the record every source path produces and every PVec path consumes
//! (issue #1095, N3.1; specification `docs/adr/pvec-normalized-geometry.md`).
//!
//! # The delegation rule
//!
//! This record is a thin adapter over PVec's public semantic API (NG-01). It holds values that
//! `pvec_geometry` constructors have already accepted, hands them on, and records what was done to
//! get them. It does not compute with them. Every mathematical operation delegates to
//! `pvec_geometry`: validity, bounds, the finite-and-normal question and every degeneracy rule are
//! PVec's, and a refusal comes back as PVec's own error with its rule identifier (NG-09).
//!
//! So this module contains no evaluator, no tessellator, no knot manipulation, no curve-kind
//! promotion, no tolerance and no comparison threshold, and it has no equality of its own
//! (NG-B1 to NG-B6). Replacing a `pvec_geometry` call here with a local reimplementation is a
//! reviewable violation of the ADR, not a style preference: point at NG-01 and the NG-10 row it
//! matches. `tests/pvec_1095_normalized_geometry.rs` scans this file for the constructs the ADR
//! bans.
//!
//! Where PVec has no entry point for a mapping (a line from two end points, a bulge segment, an
//! ellipse from a major-axis vector and a ratio, see NG-11) there is no constructor here either.
//! The caller records that outcome with [`Normalized2DGeometry::without_geometry`] and
//! [`FidelityClass::RequiresPVecCapability`].
//!
//! # Coordinates
//!
//! Source-space `f64`, unconverted (NG-04). A constructor hands each coordinate to PVec as it
//! arrived, so a non-finite one is refused on entry (PVec's NP-01) and no record can hold one.
//!
//! # What is not here
//!
//! Provenance payloads (N3.2), view options (N3.3), transform composition (N3.4), the six-way
//! classification's reason codes (N3.5), identity and ordering (N3.7) and the debug dump (N3.8).
//! The provenance handle and the fidelity enum below are the smallest shapes the record needs, so
//! that it compiles and is testable; the issues that own them extend them.

use std::fmt;

use pvec_geometry::{
    BSpline, Bounds, Circle, Curve, Curve2d, Ellipse, GeometryError, Placement2, Point2, Polyline,
    RationalBSpline, Trimmed, Vector2,
};
use pvec_numeric::NumericError;

/// The box PVec reports for an entity's curve (NG-05): enclosing, not always tight, and input for
/// the spatial index only.
pub type SourceBounds = Bounds;

/// An opaque handle to a provenance payload held outside the record (NG-06).
///
/// The payload is N3.2's. Nothing in the record is derived from the handle, and the record is
/// complete without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProvenanceHandle(u64);

impl ProvenanceHandle {
    /// Wraps a store-assigned value.
    pub const fn new(value: u64) -> ProvenanceHandle {
        ProvenanceHandle(value)
    }

    /// The wrapped value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// What became of a source entity (NG-07): the six classes of N3.5.
///
/// Fieldless on purpose here. N3.5 owns the reason codes and any growth of the set, hence
/// `non_exhaustive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FidelityClass {
    /// The PVec value is the source construct, structurally.
    SupportedExact,
    /// The PVec value is the same locus under a different but equivalent construct.
    SupportedEquivalent,
    /// The mapping needs mathematics PVec has no entry point for yet. No geometry.
    RequiresPVecCapability,
    /// The geometry exists and needs an extension to be carried in full.
    RequiresExtension,
    /// The source construct has no mapping. No geometry.
    Unsupported,
    /// Dropped on purpose by policy. No geometry.
    IgnoredByPolicy,
}

impl FidelityClass {
    /// Whether a record of this class carries a PVec curve (NG-07).
    pub const fn produces_geometry(self) -> bool {
        matches!(
            self,
            FidelityClass::SupportedExact
                | FidelityClass::SupportedEquivalent
                | FidelityClass::RequiresExtension
        )
    }
}

/// Why a record could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormalizeError {
    /// PVec refused the value, under the rule `error.rule()` names. `field` says which input of
    /// the constructor it was handed in.
    Refused {
        /// The constructor input PVec refused.
        field: &'static str,
        /// PVec's refusal, unchanged.
        error: GeometryError,
    },
    /// The fidelity class and the presence of geometry disagree (NG-07).
    FidelityGeometryMismatch {
        /// The class that was offered.
        fidelity: FidelityClass,
    },
}

impl NormalizeError {
    /// Whether PVec refused a coordinate or parameter as NaN or infinite.
    pub fn is_non_finite(&self) -> bool {
        matches!(
            self,
            NormalizeError::Refused { error, .. }
                if matches!(error.as_numeric(), Some(NumericError::NotFinite))
        )
    }

    /// The PVec rule identifier behind a refusal, `None` for a class mismatch.
    pub fn rule(&self) -> Option<&'static str> {
        match self {
            NormalizeError::Refused { error, .. } => Some(error.rule()),
            NormalizeError::FidelityGeometryMismatch { .. } => None,
        }
    }
}

impl fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormalizeError::Refused { field, error } => {
                write!(f, "PVec refused `{field}`: {error}")
            }
            NormalizeError::FidelityGeometryMismatch { fidelity } => {
                write!(f, "{fidelity:?} and the presence of geometry disagree")
            }
        }
    }
}

impl std::error::Error for NormalizeError {}

fn refuse(field: &'static str) -> impl FnOnce(GeometryError) -> NormalizeError {
    move |error| NormalizeError::Refused { field, error }
}

fn point(field: &'static str, p: [f64; 2]) -> Result<Point2, NormalizeError> {
    Point2::checked(p[0], p[1]).map_err(refuse(field))
}

fn points(field: &'static str, ps: &[[f64; 2]]) -> Result<Vec<Point2>, NormalizeError> {
    ps.iter().map(|p| point(field, *p)).collect()
}

fn placement(centre: [f64; 2], ref_direction: [f64; 2]) -> Result<Placement2, NormalizeError> {
    let origin = point("centre", centre)?;
    Placement2::new(origin, Vector2::new(ref_direction[0], ref_direction[1]))
        .map_err(refuse("ref_direction"))
}

/// One source entity, normalised: a PVec curve (or none), PVec's box for it, an optional
/// provenance handle and a fidelity class (NG-02).
///
/// Deliberately without `PartialEq` and `Hash` (NG-B6): equality of coordinates needs PVec's
/// scale term, and identity is `EntityKey`'s job (N3.7).
#[derive(Debug, Clone)]
pub struct Normalized2DGeometry {
    geometry: Option<Curve>,
    source_bounds: Option<SourceBounds>,
    provenance: Option<ProvenanceHandle>,
    fidelity: FidelityClass,
}

impl Normalized2DGeometry {
    /// Wraps a curve PVec already accepted. The box is PVec's, taken from `Curve::bounds`.
    ///
    /// Refused when `fidelity` is a class that produces no geometry, and when PVec reports no
    /// box for the curve (a bare, unbounded line).
    pub fn from_curve(curve: Curve, fidelity: FidelityClass) -> Result<Self, NormalizeError> {
        if !fidelity.produces_geometry() {
            return Err(NormalizeError::FidelityGeometryMismatch { fidelity });
        }
        let bounds = curve.bounds().map_err(refuse("curve"))?;
        Ok(Normalized2DGeometry {
            geometry: Some(curve),
            source_bounds: Some(bounds),
            provenance: None,
            fidelity,
        })
    }

    /// A record that exists but holds no geometry: `RequiresPVecCapability`, `Unsupported` or
    /// `IgnoredByPolicy`. It keeps the entity counted (NG-07).
    pub fn without_geometry(fidelity: FidelityClass) -> Result<Self, NormalizeError> {
        if fidelity.produces_geometry() {
            return Err(NormalizeError::FidelityGeometryMismatch { fidelity });
        }
        Ok(Normalized2DGeometry {
            geometry: None,
            source_bounds: None,
            provenance: None,
            fidelity,
        })
    }

    /// A circle (profile section 6): centre, reference direction and radius, as given.
    pub fn circle(
        centre: [f64; 2],
        ref_direction: [f64; 2],
        radius: f64,
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let circle =
            Circle::new(placement(centre, ref_direction)?, radius).map_err(refuse("radius"))?;
        Self::from_curve(Curve::Circle(circle), fidelity)
    }

    /// A circular arc: a trimmed circle (profile sections 6 and 8), trim parameters and sense as
    /// given.
    #[allow(clippy::too_many_arguments)]
    pub fn circular_arc(
        centre: [f64; 2],
        ref_direction: [f64; 2],
        radius: f64,
        t1: f64,
        t2: f64,
        sense: bool,
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let circle =
            Circle::new(placement(centre, ref_direction)?, radius).map_err(refuse("radius"))?;
        let arc = Trimmed::new(Curve::Circle(circle), t1, t2, sense).map_err(refuse("trim"))?;
        Self::from_curve(Curve::Trimmed(arc), fidelity)
    }

    /// An ellipse (profile section 7): placement and the two semi-axes, as given.
    pub fn ellipse(
        centre: [f64; 2],
        ref_direction: [f64; 2],
        semi_axis_1: f64,
        semi_axis_2: f64,
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let e = Ellipse::new(placement(centre, ref_direction)?, semi_axis_1, semi_axis_2)
            .map_err(refuse("semi_axis"))?;
        Self::from_curve(Curve::Ellipse(e), fidelity)
    }

    /// A polyline (profile section 5): vertices only. A closed one repeats its first vertex.
    pub fn polyline(
        vertices: &[[f64; 2]],
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let p = Polyline::new(points("vertices", vertices)?).map_err(refuse("vertices"))?;
        Self::from_curve(Curve::Polyline(p), fidelity)
    }

    /// A B-spline with explicit knots (profile section 11), passed through whole.
    pub fn b_spline(
        degree: usize,
        control_points: &[[f64; 2]],
        knots: &[f64],
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let s = BSpline::new(
            degree,
            points("control_points", control_points)?,
            knots.to_vec(),
        )
        .map_err(refuse("knots"))?;
        Self::from_curve(Curve::BSpline(s), fidelity)
    }

    /// A rational B-spline (profile section 12): the same plus one weight per control point.
    /// Weights present means rational, never demoted.
    pub fn rational_b_spline(
        degree: usize,
        control_points: &[[f64; 2]],
        knots: &[f64],
        weights: &[f64],
        fidelity: FidelityClass,
    ) -> Result<Self, NormalizeError> {
        Self::check_class(fidelity)?;
        let s = RationalBSpline::new(
            degree,
            points("control_points", control_points)?,
            knots.to_vec(),
            weights.to_vec(),
        )
        .map_err(refuse("weights"))?;
        Self::from_curve(Curve::RationalBSpline(s), fidelity)
    }

    fn check_class(fidelity: FidelityClass) -> Result<(), NormalizeError> {
        if fidelity.produces_geometry() {
            Ok(())
        } else {
            Err(NormalizeError::FidelityGeometryMismatch { fidelity })
        }
    }

    /// Attaches a provenance handle. Nothing else on the record changes.
    #[must_use]
    pub fn with_provenance(mut self, handle: ProvenanceHandle) -> Self {
        self.provenance = Some(handle);
        self
    }

    /// The PVec curve, `None` for the three classes that produce none.
    pub fn geometry(&self) -> Option<&Curve> {
        self.geometry.as_ref()
    }

    /// PVec's box for the curve: enclosing, not always tight, for the spatial index only.
    pub fn source_bounds(&self) -> Option<&SourceBounds> {
        self.source_bounds.as_ref()
    }

    /// The provenance handle, if one was attached.
    pub fn provenance(&self) -> Option<ProvenanceHandle> {
        self.provenance
    }

    /// The fidelity class.
    pub fn fidelity(&self) -> FidelityClass {
        self.fidelity
    }
}
