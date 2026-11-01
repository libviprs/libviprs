//! Transform composition and promotion (issue #1098, N3.4; matrix frozen by
//! `docs/adr/pvec-transform-matrix.md`).
//!
//! A nested INSERT reaches geometry through several affine maps. This module composes them and
//! pushes a [`Normalized2DGeometry`] through the result. It owns the order, the lineage and the
//! lookup against the frozen matrix. It owns no mathematics: the product is
//! [`Transform2::then`], the singularity and finiteness rules are the ones
//! [`Transform2::new`] applies (NP-01, NP-19), the image of a curve is
//! [`Curve2d::transform`], and a conic that comes out too eccentric is refused there (NP-07).
//! There is no threshold, epsilon or comparison in this file, and
//! `tests/pvec_1098_transform_composition.rs` scans it for a numeric literal.
//!
//! # Composition order
//!
//! Levels are given **outermost insert first**, the order a traversal meets them. A point of the
//! innermost block is mapped by the innermost insert first and the outermost insert last, so for
//! levels `L0` (outer) to `Ln` (inner) the composed map is "`Ln`, then `Ln-1`, ..., then `L0`".
//! [`ComposedTransform::push_inner`] adds one level inside the current ones, which is what a
//! depth-first walk does on the way in, and composing a whole nest with
//! [`ComposedTransform::compose`] is the same thing level by level. The same levels in the
//! other order give a different matrix (a translation outside a scale is not a scale outside a
//! translation), and the test pins one nest where it does.
//!
//! # Refusals
//!
//! Every step goes through PVec, so every step is checked: a level that is not finite is refused
//! under NP-01, one that is singular or near-singular under NP-19, and so is a product that
//! became so although no single level was. The refusal names the insert whose step it was
//! ([`ComposeError`]) and no composed transform exists for that branch, so nothing can be
//! promoted through it. The matrix's frozen reason codes ([`TransformReason`]) are looked up
//! from the PVec rule, never from a value measured here.
//!
//! # Promotion
//!
//! [`promote`] does not classify a map as uniform, non-uniform or a reflection. It hands the
//! curve to PVec and checks the kind that comes back against the row's cell of the frozen matrix
//! ([`PromotionMatrix::FROZEN`]). A circle stays a circle under a similarity and becomes an
//! ellipse otherwise because PVec says so, and a kind outside the cell is a
//! [`PromotionError::MatrixViolation`], not something to pass on. A cell that says
//! [`Cell::RequiresPVecCapability`] yields that classification and no geometry.
//!
//! # Lineage
//!
//! [`ComposedTransform::lineage`] keeps each level as given, so a diagnostic can show the nest
//! that produced a curve and the composed matrix can be recomputed from it. N3.2's provenance
//! store records the composed map through [`ComposedTransform::source_transform`] and the inserts
//! through [`ComposedTransform::instance_path`].

use std::fmt;

use pvec_geometry::{Curve, Curve2d, GeometryError, ParamMap, Transform2};

use super::fidelity::{Classification, ReasonCode};
use super::normalized::{FidelityClass, NormalizeError, Normalized2DGeometry};
use super::provider::{InstancePath, InstanceStep, SourceTransform};

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
        match self {
            TransformReason::NonFinite => "TRANSFORM_NON_FINITE",
            TransformReason::Singular => "TRANSFORM_SINGULAR",
            TransformReason::ConicIllConditioned => "TRANSFORM_CONIC_ILL_CONDITIONED",
        }
    }
    /// The classification reason this refusal carries.
    pub const fn reason_code(self) -> ReasonCode {
        match self {
            TransformReason::NonFinite => ReasonCode::TransformNonFinite,
            TransformReason::Singular => ReasonCode::TransformSingular,
            TransformReason::ConicIllConditioned => ReasonCode::TransformConicIllConditioned,
        }
    }

    /// The PVec rule behind it.
    pub const fn rule(self) -> &'static str {
        match self {
            TransformReason::NonFinite => "NP-01",
            TransformReason::Singular => "NP-19",
            TransformReason::ConicIllConditioned => "NP-07",
        }
    }

    /// The reason code behind a PVec refusal, `None` when the rule is not one of the three.
    ///
    /// NP-02 is folded into `Singular` because `Transform2::new` reports an entry out of range
    /// under it, and NP-19 describes that case as part of an unusable map.
    pub fn from_refusal(error: &GeometryError) -> Option<TransformReason> {
        match error.rule() {
            "NP-01" => Some(TransformReason::NonFinite),
            "NP-19" | "NP-02" => Some(TransformReason::Singular),
            "NP-07" => Some(TransformReason::ConicIllConditioned),
            _ => None,
        }
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

    /// The classification of the INSERT whose map was refused.
    #[must_use]
    pub fn classification(&self, source_class: &str) -> Classification {
        // A refusal outside the frozen three has no matrix code. It is still counted, as an
        // unmapped class, so nothing is dropped.
        let reason = self.reason.map_or(
            ReasonCode::UnmappedSourceClass,
            TransformReason::reason_code,
        );
        Classification::refused(reason, source_class)
    }
}

impl fmt::Display for ComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "insert of block `{}` (handle {}, level {}): ",
            self.insert.block_name,
            self.insert.insert_handle.get(),
            self.depth
        )?;
        match self.reason {
            Some(r) => write!(f, "{} ({}): {}", r.code(), r.rule(), self.error),
            None => write!(f, "{}", self.error),
        }
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
    /// The empty nest: no insert, the identity.
    pub fn identity() -> Self {
        ComposedTransform {
            levels: Vec::new(),
            composed: Transform2::IDENTITY,
        }
    }

    /// Composes a nest given outermost insert first.
    ///
    /// # Errors
    ///
    /// The first level whose step PVec refuses, named in the error. Nothing is returned for a
    /// refused nest.
    pub fn compose<I>(levels: I) -> Result<Self, ComposeError>
    where
        I: IntoIterator<Item = (InstanceStep, SourceTransform)>,
    {
        levels
            .into_iter()
            .try_fold(Self::identity(), |acc, (insert, t)| {
                acc.push_inner(insert, &t)
            })
    }

    /// One more level inside the ones already composed.
    ///
    /// # Errors
    ///
    /// [`ComposeError`] naming `insert` when PVec refuses the level or the product.
    pub fn push_inner(
        &self,
        insert: InstanceStep,
        transform: &SourceTransform,
    ) -> Result<Self, ComposeError> {
        self.push_inner_coefficients(insert, transform.coefficients())
    }

    /// As [`push_inner`](Self::push_inner), from six raw coefficients `[m00, m01, m10, m11, tx,
    /// ty]` that have not been through [`SourceTransform`]. The finite check happens here, in
    /// PVec, and names `insert`.
    ///
    /// # Errors
    ///
    /// [`ComposeError`] naming `insert` when PVec refuses the level or the product.
    pub fn push_inner_coefficients(
        &self,
        insert: InstanceStep,
        coefficients: [f64; 6],
    ) -> Result<Self, ComposeError> {
        let depth = self.levels.len();
        let refuse = |insert: &InstanceStep, error: GeometryError| ComposeError {
            insert: insert.clone(),
            depth,
            reason: TransformReason::from_refusal(&error),
            error,
        };
        let level = Transform2::new(coefficients).map_err(|e| refuse(&insert, e))?;
        // The new level is applied first and everything outside it afterwards.
        let composed = level.then(&self.composed).map_err(|e| refuse(&insert, e))?;
        let mut levels = self.levels.clone();
        levels.push(LineageLevel {
            insert,
            transform: coefficients,
        });
        Ok(ComposedTransform { levels, composed })
    }

    /// The composed matrix `[m00, m01, m10, m11, tx, ty]`.
    pub fn matrix(&self) -> [f64; 6] {
        self.composed.matrix()
    }

    /// The composed map as PVec's type.
    pub fn transform(&self) -> &Transform2 {
        &self.composed
    }

    /// The composed map as the provider-boundary type N3.2's provenance record holds.
    pub fn source_transform(&self) -> SourceTransform {
        // A `Transform2` is finite by construction, so this cannot refuse.
        SourceTransform::new(self.composed.matrix()).expect("a Transform2 is finite")
    }

    /// The inserts of the nest as an [`InstancePath`], outermost first, for a provenance stub.
    pub fn instance_path(&self) -> InstancePath {
        self.levels.iter().fold(InstancePath::root(), |path, l| {
            path.child(l.insert.block_name.clone(), l.insert.insert_handle)
        })
    }

    /// The levels as recorded, outermost first.
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
        match curve {
            Curve::Polyline(_) => Some(Shape::Polyline),
            Curve::Composite(_) => Some(Shape::Composite),
            Curve::Circle(_) => Some(Shape::Circle),
            Curve::Ellipse(_) => Some(Shape::Ellipse),
            Curve::BSpline(_) => Some(Shape::BSpline),
            Curve::RationalBSpline(_) => Some(Shape::RationalBSpline),
            Curve::Trimmed(t) => match t.basis() {
                Curve::Line(_) => Some(Shape::TrimmedLine),
                Curve::Circle(_) => Some(Shape::TrimmedCircle),
                Curve::Ellipse(_) => Some(Shape::TrimmedEllipse),
                _ => None,
            },
            _ => None,
        }
    }
}

impl SourceRow {
    /// The row of a PVec curve, `None` for kinds outside the matrix.
    pub fn of(curve: &Curve) -> Option<SourceRow> {
        match curve {
            Curve::Polyline(_) => Some(SourceRow::Polyline),
            Curve::Composite(_) => Some(SourceRow::PolylineWithBulge),
            Curve::Circle(_) => Some(SourceRow::Circle),
            Curve::Ellipse(_) => Some(SourceRow::Ellipse),
            Curve::BSpline(_) => Some(SourceRow::BSpline),
            Curve::RationalBSpline(_) => Some(SourceRow::Nurbs),
            Curve::Trimmed(t) => match t.basis() {
                Curve::Line(_) => Some(SourceRow::Line),
                Curve::Circle(_) => Some(SourceRow::CircularArc),
                Curve::Ellipse(_) => Some(SourceRow::EllipticalArc),
                _ => None,
            },
            _ => None,
        }
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

/// The frozen matrix, one cell per row. Every non-singular column of a row shares the cell: which
/// of the two shapes a circle or an arc comes back as is PVec's call, made under its own equality
/// rule, and the per-column outcomes are pinned in the tests against the spike's measurements.
/// The singular column never reaches a cell, because no composed map exists for it.
fn frozen_cell(row: SourceRow) -> Cell {
    match row {
        SourceRow::Line => Cell::Shapes(&[Shape::TrimmedLine]),
        SourceRow::Polyline => Cell::Shapes(&[Shape::Polyline]),
        SourceRow::PolylineWithBulge => Cell::Shapes(&[Shape::Composite]),
        SourceRow::CircularArc => Cell::Shapes(&[Shape::TrimmedCircle, Shape::TrimmedEllipse]),
        SourceRow::Circle => Cell::Shapes(&[Shape::Circle, Shape::Ellipse]),
        SourceRow::Ellipse => Cell::Shapes(&[Shape::Ellipse]),
        SourceRow::EllipticalArc => Cell::Shapes(&[Shape::TrimmedEllipse]),
        SourceRow::BSpline => Cell::Shapes(&[Shape::BSpline]),
        SourceRow::Nurbs => Cell::Shapes(&[Shape::RationalBSpline]),
    }
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
    let Some(curve) = record.geometry() else {
        // Nothing to promote: the class and the reason stay as they were.
        return Ok(Promotion {
            record: record.clone(),
            param_map: None,
            reason: None,
        });
    };
    let row = SourceRow::of(curve).ok_or(PromotionError::NoMatrixRow)?;
    if matches!(matrix.cell(row), Cell::RequiresPVecCapability) {
        return classified(record, FidelityClass::RequiresPVecCapability, None);
    }
    match curve.transform(composed.transform()) {
        Err(error) => match TransformReason::from_refusal(&error) {
            Some(reason) => classified(record, FidelityClass::Unsupported, Some(reason)),
            None => Err(PromotionError::Refused(error)),
        },
        Ok((image, param_map)) => {
            if let Verdict::RequiresCapability = check(matrix, curve, &image)? {
                return classified(record, FidelityClass::RequiresPVecCapability, None);
            }
            let mut out = Normalized2DGeometry::from_curve(image, record.fidelity())
                .map_err(PromotionError::Normalize)?;
            if let Some(h) = record.provenance() {
                out = out.with_provenance(h);
            }
            Ok(Promotion {
                record: out,
                param_map: Some(param_map),
                reason: None,
            })
        }
    }
}

enum Verdict {
    Allowed,
    RequiresCapability,
}

/// Checks PVec's image against the cell of the source's row, and for a composite against the cell
/// of each segment's row, so an arc inside a bulge composite is held to the same table.
fn check(
    matrix: &PromotionMatrix,
    source: &Curve,
    image: &Curve,
) -> Result<Verdict, PromotionError> {
    let row = SourceRow::of(source).ok_or(PromotionError::NoMatrixRow)?;
    let allowed = match matrix.cell(row) {
        Cell::RequiresPVecCapability => return Ok(Verdict::RequiresCapability),
        Cell::Shapes(shapes) => shapes,
    };
    let got = Shape::of(image);
    if !got.is_some_and(|g| allowed.contains(&g)) {
        return Err(PromotionError::MatrixViolation { row, got });
    }
    if let (Curve::Composite(s), Curve::Composite(i)) = (source, image) {
        if s.segments().len() != i.segments().len() {
            return Err(PromotionError::MatrixViolation { row, got });
        }
        for (a, b) in s.segments().iter().zip(i.segments()) {
            if let Verdict::RequiresCapability = check(matrix, a, b)? {
                return Ok(Verdict::RequiresCapability);
            }
        }
    }
    Ok(Verdict::Allowed)
}

/// A record with no geometry, keeping the provenance handle so the entity stays counted and
/// traceable.
fn classified(
    record: &Normalized2DGeometry,
    class: FidelityClass,
    reason: Option<TransformReason>,
) -> Result<Promotion, PromotionError> {
    let mut out =
        Normalized2DGeometry::without_geometry(class).map_err(PromotionError::Normalize)?;
    if let Some(h) = record.provenance() {
        out = out.with_provenance(h);
    }
    Ok(Promotion {
        record: out,
        param_map: None,
        reason,
    })
}
