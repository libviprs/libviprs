//! Transform composition and promotion (issue #1098, N3.4; matrix frozen by
//! `docs/adr/pvec-transform-matrix.md`).
//!
//! STUB: red tests first, bodies follow.

use std::fmt;

use pvec_geometry::{Curve, GeometryError, ParamMap, Transform2};

use super::normalized::{NormalizeError, Normalized2DGeometry};
use super::provider::{InstanceStep, SourceTransform};

/// The three frozen reason codes of the matrix (all `Unsupported`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransformReason {
    /// A coefficient is not finite (NP-01).
    NonFinite,
    /// Singular or near-singular (NP-19).
    Singular,
    /// A conic image too eccentric (NP-07).
    ConicIllConditioned,
}

impl TransformReason {
    /// The stable reason code.
    pub const fn code(self) -> &'static str {
        unimplemented!()
    }
    /// The PVec rule behind it.
    pub const fn rule(self) -> &'static str {
        unimplemented!()
    }
}

/// One level of a nest as recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct LineageLevel {
    /// The insert.
    pub insert: InstanceStep,
    /// Its transform as the provider gave it.
    pub transform: [f64; 6],
}

/// Why a composition step was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeError {
    insert: InstanceStep,
    depth: usize,
    reason: Option<TransformReason>,
    error: GeometryError,
}

impl ComposeError {
    /// The insert whose transform was refused.
    pub fn insert(&self) -> &InstanceStep {
        &self.insert
    }
    /// Its position in the nest, 0 outermost.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// The matrix reason code, when PVec's rule is one of the frozen three.
    pub fn reason(&self) -> Option<TransformReason> {
        self.reason
    }
    /// PVec's refusal.
    pub fn error(&self) -> &GeometryError {
        &self.error
    }
}

impl fmt::Display for ComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.insert.block_name)
    }
}

impl std::error::Error for ComposeError {}

/// A composed transform with the lineage that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposedTransform {
    levels: Vec<LineageLevel>,
    composed: Transform2,
}

impl ComposedTransform {
    /// The empty nest.
    pub fn identity() -> Self {
        unimplemented!()
    }
    /// Composes a nest, outermost first.
    pub fn compose<I>(levels: I) -> Result<Self, ComposeError>
    where
        I: IntoIterator<Item = (InstanceStep, SourceTransform)>,
    {
        let _ = levels;
        unimplemented!()
    }
    /// One more level, inside this one.
    pub fn push_inner(
        &self,
        insert: InstanceStep,
        transform: &SourceTransform,
    ) -> Result<Self, ComposeError> {
        let _ = (insert, transform);
        unimplemented!()
    }
    /// As `push_inner`, from raw coefficients.
    pub fn push_inner_coefficients(
        &self,
        insert: InstanceStep,
        coefficients: [f64; 6],
    ) -> Result<Self, ComposeError> {
        let _ = (insert, coefficients);
        unimplemented!()
    }
    /// The composed matrix.
    pub fn matrix(&self) -> [f64; 6] {
        unimplemented!()
    }
    /// The composed map as PVec's type.
    pub fn transform(&self) -> &Transform2 {
        &self.composed
    }
    /// The composed map as the provenance type.
    pub fn source_transform(&self) -> SourceTransform {
        unimplemented!()
    }
    /// The recorded levels, outermost first.
    pub fn lineage(&self) -> &[LineageLevel] {
        &self.levels
    }
}

/// A row of the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceRow {
    /// Trimmed line.
    Line,
    /// Polyline.
    Polyline,
    /// Bulge composite.
    PolylineWithBulge,
    /// Circular arc.
    CircularArc,
    /// Circle.
    Circle,
    /// Ellipse.
    Ellipse,
    /// Elliptical arc.
    EllipticalArc,
    /// B-spline.
    BSpline,
    /// NURBS.
    Nurbs,
}

/// A PVec semantic type as the matrix names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shape {
    /// `trimmed_curve over line`.
    TrimmedLine,
    /// `polyline`.
    Polyline,
    /// `composite_curve`.
    Composite,
    /// `trimmed_curve over circle`.
    TrimmedCircle,
    /// `circle`.
    Circle,
    /// `trimmed_curve over ellipse`.
    TrimmedEllipse,
    /// `ellipse`.
    Ellipse,
    /// `b_spline_curve_with_knots`.
    BSpline,
    /// `rational_b_spline_curve`.
    RationalBSpline,
}

impl Shape {
    /// The shape of a PVec curve, `None` for kinds outside the matrix.
    pub fn of(curve: &Curve) -> Option<Shape> {
        let _ = curve;
        unimplemented!()
    }
}

impl SourceRow {
    /// The row of a PVec curve, `None` for kinds outside the matrix.
    pub fn of(curve: &Curve) -> Option<SourceRow> {
        let _ = curve;
        unimplemented!()
    }
}

/// A cell of the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// PVec returns one of these shapes.
    Shapes(&'static [Shape]),
    /// PVec has no entry point for it.
    RequiresPVecCapability,
}

/// The frozen matrix, one cell per row for every non-singular map.
#[derive(Debug, Clone, Copy)]
pub struct PromotionMatrix {
    cell: fn(SourceRow) -> Cell,
}

impl PromotionMatrix {
    /// The matrix of `docs/adr/pvec-transform-matrix.md`.
    pub const FROZEN: PromotionMatrix = PromotionMatrix { cell: frozen_cell };
    /// A matrix from a lookup, for tests of the other outcomes.
    pub const fn from_lookup(cell: fn(SourceRow) -> Cell) -> Self {
        PromotionMatrix { cell }
    }
    /// The cell for a row.
    pub fn cell(&self, row: SourceRow) -> Cell {
        (self.cell)(row)
    }
}

fn frozen_cell(_row: SourceRow) -> Cell {
    unimplemented!()
}

/// What `promote` could not do.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PromotionError {
    /// The curve's kind has no row.
    NoMatrixRow,
    /// PVec returned a shape the cell does not allow.
    MatrixViolation {
        /// The row.
        row: SourceRow,
        /// What came back.
        got: Option<Shape>,
    },
    /// PVec refused for a rule that is not one of the three reason codes.
    Refused(GeometryError),
    /// The record could not be rebuilt.
    Normalize(NormalizeError),
}

/// The result of promoting one record.
#[derive(Debug, Clone)]
pub struct Promotion {
    record: Normalized2DGeometry,
    param_map: Option<ParamMap>,
    reason: Option<TransformReason>,
}

impl Promotion {
    /// The record.
    pub fn record(&self) -> &Normalized2DGeometry {
        &self.record
    }
    /// The parameter map PVec returned.
    pub fn param_map(&self) -> Option<ParamMap> {
        self.param_map
    }
    /// Why the record carries no geometry, when that is a transform refusal.
    pub fn reason(&self) -> Option<TransformReason> {
        self.reason
    }
}

/// Promotes a record through a composed transform, against the frozen matrix.
pub fn promote(
    record: &Normalized2DGeometry,
    composed: &ComposedTransform,
) -> Result<Promotion, PromotionError> {
    promote_with(&PromotionMatrix::FROZEN, record, composed)
}

/// As [`promote`], against a given matrix.
pub fn promote_with(
    matrix: &PromotionMatrix,
    record: &Normalized2DGeometry,
    composed: &ComposedTransform,
) -> Result<Promotion, PromotionError> {
    let _ = (matrix, record, composed);
    unimplemented!()
}
