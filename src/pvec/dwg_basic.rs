//! Exact extraction of the basic DWG entities (N4.2, #1073).
//!
//! A `LINE`, a `LWPOLYLINE`, a 2D `POLYLINE`, a `CIRCLE`, an `ARC` and an `ELLIPSE` each have a
//! closed-form PVec equivalent, so nothing here samples, fits or tessellates. Every construction
//! is a call into `pvec_geometry`, and every tolerance is PVec's numeric policy speaking through
//! the refusal it returns. A refusal is classified (see [`BasicError::classification`]) and gets a
//! one-line report row ([`BasicError::warning`]), never repaired.

use std::fmt;

use pvec_geometry::{Curve, GeometryError};

use super::fidelity::Classification;
use super::provider::{Point3, RawEntity};

/// Which basic source class an extraction came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BasicKind {
    /// `LINE`.
    Line,
    /// `LWPOLYLINE` or a 2D `POLYLINE`, with or without bulges.
    Polyline,
    /// `CIRCLE`.
    Circle,
    /// `ARC`.
    Arc,
    /// `ELLIPSE` over its whole period.
    Ellipse,
    /// `ELLIPSE` cut to an arc.
    EllipticalArc,
}

/// What one segment of a polyline became.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SegmentKind {
    /// A straight segment (bulge exactly zero).
    Straight,
    /// A counter-clockwise arc (positive bulge).
    ArcCounterClockwise,
    /// A clockwise arc (negative bulge).
    ArcClockwise,
}

/// One basic entity, as PVec carries it plus the source numbers PVec's canonical form moves.
#[derive(Debug, Clone)]
pub struct ExactBasic {
    /// The source class.
    pub kind: BasicKind,
    /// The PVec curve.
    pub curve: Curve,
    /// One entry per polyline segment, the closing segment of a closed polyline last. Empty for
    /// every other kind.
    pub segments: Vec<SegmentKind>,
    /// The drawing's closed flag (polylines), `false` otherwise.
    pub closed: bool,
    /// The angles of an arc, or the parameters of an elliptical arc, exactly as stored, in the
    /// drawing's counter-clockwise convention. PVec's trim is a normalised increasing interval
    /// and the numbers here are what the drawing said.
    pub source_range: Option<(f64, f64)>,
    /// The z of the entity's plane.
    pub elevation: f64,
    /// The outcome under N3.5's classes.
    pub classification: Classification,
}

/// Why a basic entity was refused. Nothing is repaired.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BasicError {
    /// The entity is not a basic one (or not the class the function extracts).
    NotBasic,
    /// A radius that is zero, negative, not finite or below PVec's floor.
    DegenerateRadius {
        /// The radius as stored.
        radius: f64,
    },
    /// Two consecutive end points PVec treats as one point.
    CoincidentVertices {
        /// The index of the vertex that starts the segment.
        index: usize,
    },
    /// An ellipse ratio PVec refuses.
    DegenerateAxisRatio {
        /// The ratio as stored.
        ratio: f64,
    },
    /// A bulge that is not finite or that PVec refuses.
    DegenerateBulge {
        /// The index of the vertex that carries it.
        index: usize,
        /// The bulge as stored.
        bulge: f64,
    },
    /// PVec refused the rest.
    Degenerate {
        /// Which input.
        field: &'static str,
        /// PVec's refusal, unchanged.
        error: GeometryError,
    },
    /// More polyline vertices than PVec supports.
    TooManyVertices {
        /// The limit.
        limit: usize,
        /// Observed.
        observed: usize,
    },
    /// Different z values on one entity.
    NonPlanar,
    /// An extrusion normal that is not +z.
    ExtrusionNotApplied {
        /// The normal as stored.
        normal: Point3,
    },
}

impl fmt::Display for BasicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BasicError {}

impl BasicError {
    /// What became of the entity, with a stable reason code.
    #[must_use]
    pub fn classification(&self, source_class: impl Into<String>) -> Classification {
        let _ = source_class.into();
        unimplemented!("N4.2 red phase")
    }

    /// The row the conversion report (N6.7) carries for this refusal.
    #[must_use]
    pub fn warning(&self, source_class: &str) -> String {
        let _ = source_class;
        unimplemented!("N4.2 red phase")
    }
}

/// Extracts a `LINE`.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_line(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}

/// Extracts a `LWPOLYLINE` or a 2D `POLYLINE`.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_polyline(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}

/// Extracts a `CIRCLE`.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_circle(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}

/// Extracts an `ARC`.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_arc(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}

/// Extracts an `ELLIPSE`.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_ellipse(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}

/// Extracts whichever basic class the entity's geometry is.
///
/// # Errors
/// See [`BasicError`].
pub fn extract(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let _ = entity;
    unimplemented!("N4.2 red phase")
}
