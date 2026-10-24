//! The clipping and compound-path policy for the PDF side (N5.3, #1082).
//!
//! Stub: the red tests in `tests/pvec_pdf_clip_1082.rs` come first.

use super::fidelity::Classification;
use super::limits::SourceLimits;
use super::pdf_vector::{PdfFillRule, PdfPathEntity, PdfPoint, PdfVectorPage};

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactBounds {
    /// Lower-left corner.
    pub min: PdfPoint,
    /// Upper-right corner.
    pub max: PdfPoint,
}

/// Where exact bounds come from.
pub trait ExactBoundsSource {
    /// The exact extent of the entity's path, or `None` when this source cannot say.
    fn exact_bounds(&self, entity: &PdfPathEntity) -> Option<ExactBounds>;
}

/// Answers only for straight-segment paths.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinesOnlyBounds;

impl ExactBoundsSource for LinesOnlyBounds {
    fn exact_bounds(&self, _entity: &PdfPathEntity) -> Option<ExactBounds> {
        todo!()
    }
}

/// Clip nesting bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipLimits {
    /// Deepest number of simultaneously active clips.
    pub max_clip_depth: u32,
}

impl From<&SourceLimits> for ClipLimits {
    fn from(_: &SourceLimits) -> Self {
        todo!()
    }
}

/// The closed set of outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClipCase {
    /// No active clip.
    NoClip,
    /// Every active clip is an axis-aligned rectangle.
    ExactRectangular,
    /// A general clip path, display fidelity only.
    BoundedConservative,
    /// Exact bounds outside the clip.
    FullyClipped,
    /// Depends on a construct that is not representable.
    Unsupported,
}

impl ClipCase {
    /// All five.
    pub const ALL: [ClipCase; 5] = [
        ClipCase::NoClip,
        ClipCase::ExactRectangular,
        ClipCase::BoundedConservative,
        ClipCase::FullyClipped,
        ClipCase::Unsupported,
    ];
}

/// How faithful a recorded clip is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClipConformance {
    /// Exact with respect to the PDF.
    GeometricallyExact,
    /// For display only.
    DisplayFidelity,
}

/// One clip boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipRecord {
    /// Sequence of the entity that defined it.
    pub sequence: u64,
    /// Winding rule of the `W` or `W*`.
    pub rule: PdfFillRule,
    /// Subpaths in the boundary.
    pub subpaths: usize,
    /// The rectangle, when it is one.
    pub rect: Option<ExactBounds>,
    /// Conformance.
    pub conformance: ClipConformance,
}

/// One painted entity's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityClipDecision {
    /// The entity's sequence.
    pub entity: u64,
    /// The case.
    pub case: ClipCase,
    /// Sequences of the active clips.
    pub clips: Vec<u64>,
    /// Set when the entity is dropped or unsupported.
    pub classification: Option<Classification>,
    /// Conformance of the clipping applied to it.
    pub conformance: Option<ClipConformance>,
    /// Winding rule of the entity's own fill, per compound group.
    pub fill_rule: Option<PdfFillRule>,
    /// Subpaths in the entity.
    pub subpaths: usize,
}

/// The policy's output.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipReport {
    /// Every clip boundary.
    pub clips: Vec<ClipRecord>,
    /// Every painted entity.
    pub decisions: Vec<EntityClipDecision>,
    /// Deepest nesting seen.
    pub max_depth: usize,
}

/// Why the policy refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ClipError {
    /// Too many active clips.
    #[error("clip nesting limit {limit_name}={limit} exceeded at depth {observed}")]
    NestingExceeded {
        /// The limit.
        limit: u32,
        /// Its name.
        limit_name: &'static str,
        /// Observed depth.
        observed: usize,
    },
}

/// Decide every entity of a page.
///
/// # Errors
///
/// See [`ClipError`].
pub fn apply_clip_policy(
    _page: &PdfVectorPage,
    _limits: &ClipLimits,
    _bounds: &dyn ExactBoundsSource,
) -> Result<ClipReport, ClipError> {
    todo!()
}
