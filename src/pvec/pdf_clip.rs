//! The clipping and compound-path policy for the PDF side (N5.3, #1082).
//!
//! A PDF clip is the intersection of arbitrary paths applied to everything drawn afterwards.
//! Computing that intersection exactly needs boolean arithmetic on Beziers, which v1 does not
//! have and does not fake. So this module does not clip anything. It decides, for every painted
//! entity of a [`PdfVectorPage`], what the clips in force mean for it, and records the decision
//! so no clip is ever dropped silently.
//!
//! | [`ClipCase`] | when | result |
//! |---|---|---|
//! | `NoClip` | no clip in force | geometry as authored |
//! | `ExactRectangular` | every clip in force is an axis-aligned rectangle in page space | geometry kept, clips recorded as exact boundaries |
//! | `BoundedConservative` | some clip is a general path | geometry kept unclipped, clips recorded with their winding rule as display fidelity only |
//! | `FullyClipped` | the entity's exact bounds lie outside a clip's exact bounds | dropped, `IgnoredByPolicy` with field `OutsideClip` |
//! | `Unsupported` | a soft mask is in force | `Unsupported` with [`ReasonCode::PdfSoftMask`] |
//!
//! # Exactness
//!
//! A clip is exact only where the PDF made it exact: a rectangle. A general clip is recorded as
//! [`ClipConformance::DisplayFidelity`] and is never labelled otherwise. The rectangle test is
//! exact equality on page-space coordinates, there is no epsilon here: a rectangle that only
//! becomes axis-aligned up to rounding is a general path.
//!
//! # Dropping an entity
//!
//! `FullyClipped` needs the entity's exact bounds, and a sampled or control-hull bound is not
//! one: a curve whose hull leaves the clip can still sit entirely inside it. Bounds come through
//! [`ExactBoundsSource`]. [`LinesOnlyBounds`] answers only for straight-segment paths, where the
//! endpoints are the extent. Curve extrema are N6.4's (#1091), and until a source for them is
//! plugged in a curve is retained. A stroked entity is retained too, because its painted extent
//! is its path extent plus half a stroke width, and stroke width is N5.4's.
//!
//! # Compound paths
//!
//! The winding rule (nonzero from `f`/`W`, even-odd from `f*`/`W*`) is recorded with the number
//! of subpaths it governs, on both the clips and the painted entities. Nothing flattens a
//! compound path into one outline to resolve the rule.
//!
//! # Nesting
//!
//! The number of clips in force at once is bounded by [`ClipLimits::max_clip_depth`], taken from
//! N3.6's `max_block_depth` (it has no clip-specific field), and a path that nests deeper is a
//! typed [`ClipError`] naming the limit and the observed depth.

use std::collections::BTreeMap;

use thiserror::Error;

use super::fidelity::{Classification, ReasonCode};
use super::limits::{Limit, SourceLimits};
use super::pdf_vector::{PdfFillRule, PdfPathEntity, PdfPathSegment, PdfPoint, PdfVectorPage};

/// The source class name carried on classifications made here.
const SOURCE_CLASS: &str = "pdf-path";
/// The `IgnoredByPolicy` field for an entity outside its clip.
const OUTSIDE_CLIP: &str = "OutsideClip";

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactBounds {
    /// Lower-left corner.
    pub min: PdfPoint,
    /// Upper-right corner.
    pub max: PdfPoint,
}

impl ExactBounds {
    /// Whether the two boxes share no point. Touching is not disjoint.
    fn disjoint(&self, other: &ExactBounds) -> bool {
        self.max[0] < other.min[0]
            || other.max[0] < self.min[0]
            || self.max[1] < other.min[1]
            || other.max[1] < self.min[1]
    }
}

/// Where exact bounds come from. An implementation must never return a sampled or hull bound:
/// `None` means "cannot say", and the entity is then retained.
pub trait ExactBoundsSource {
    /// The exact extent of the entity's path, or `None` when this source cannot say.
    fn exact_bounds(&self, entity: &PdfPathEntity) -> Option<ExactBounds>;
}

/// Answers only for straight-segment paths, whose extent is the extent of their points.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinesOnlyBounds;

impl ExactBoundsSource for LinesOnlyBounds {
    fn exact_bounds(&self, entity: &PdfPathEntity) -> Option<ExactBounds> {
        let mut bounds: Option<ExactBounds> = None;
        for seg in &entity.segments {
            let p = match seg {
                PdfPathSegment::MoveTo(p) | PdfPathSegment::LineTo(p) => *p,
                PdfPathSegment::CurveTo { .. } => return None,
                PdfPathSegment::Close => continue,
            };
            bounds = Some(match bounds {
                None => ExactBounds { min: p, max: p },
                Some(b) => ExactBounds {
                    min: [b.min[0].min(p[0]), b.min[1].min(p[1])],
                    max: [b.max[0].max(p[0]), b.max[1].max(p[1])],
                },
            });
        }
        bounds
    }
}

/// The bound on clip nesting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipLimits {
    /// Most clips in force at once.
    pub max_clip_depth: u32,
}

impl From<&SourceLimits> for ClipLimits {
    fn from(limits: &SourceLimits) -> Self {
        Self {
            max_clip_depth: limits.max_block_depth,
        }
    }
}

/// The closed set of outcomes for a painted entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClipCase {
    /// No clip in force.
    NoClip,
    /// Every clip in force is an axis-aligned rectangle.
    ExactRectangular,
    /// A general clip path: kept unclipped, display fidelity only.
    BoundedConservative,
    /// Exact bounds outside a clip.
    FullyClipped,
    /// Depends on a construct that is not representable.
    Unsupported,
}

impl ClipCase {
    /// All five, in table order.
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
    /// For display only, never reported as exact.
    DisplayFidelity,
}

/// One clip boundary, as the PDF defined it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipRecord {
    /// `sequence` of the entity that defined it.
    pub sequence: u64,
    /// The winding rule of its `W` or `W*`.
    pub rule: PdfFillRule,
    /// Subpaths in the boundary.
    pub subpaths: usize,
    /// The rectangle, when the boundary is exactly one axis-aligned rectangle.
    pub rect: Option<ExactBounds>,
    /// `GeometricallyExact` for a rectangle, `DisplayFidelity` otherwise.
    pub conformance: ClipConformance,
}

/// What the policy decided for one painted entity.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityClipDecision {
    /// The entity's `sequence`.
    pub entity: u64,
    /// The case.
    pub case: ClipCase,
    /// `sequence` of every clip in force, outermost first.
    pub clips: Vec<u64>,
    /// Set for `FullyClipped` and `Unsupported`.
    pub classification: Option<Classification>,
    /// The weakest conformance among the clips applied, `None` when none applies.
    pub conformance: Option<ClipConformance>,
    /// The winding rule of the entity's own fill.
    pub fill_rule: Option<PdfFillRule>,
    /// Subpaths in the entity, which `fill_rule` governs as a group.
    pub subpaths: usize,
}

/// The policy's output for one page.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipReport {
    /// Every clip boundary, in stream order.
    pub clips: Vec<ClipRecord>,
    /// Every painted entity, in stream order.
    pub decisions: Vec<EntityClipDecision>,
    /// The most clips in force at once.
    pub max_depth: usize,
}

/// Why the policy refused.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ClipError {
    /// More clips in force at once than the limit.
    #[error("clip nesting limit {limit_name}={limit} exceeded at depth {observed}")]
    NestingExceeded {
        /// The configured limit.
        limit: u32,
        /// The `SourceLimits` field it comes from.
        limit_name: &'static str,
        /// The depth reached.
        observed: usize,
    },
}

fn subpath_count(entity: &PdfPathEntity) -> usize {
    entity
        .segments
        .iter()
        .filter(|s| matches!(s, PdfPathSegment::MoveTo(_)))
        .count()
}

/// The box of a path that is exactly one axis-aligned rectangle: a move, three lines (or four,
/// the last back to the start) and an optional close. Exact comparison, no epsilon.
pub(crate) fn exact_rect(entity: &PdfPathEntity) -> Option<ExactBounds> {
    if subpath_count(entity) != 1 {
        return None;
    }
    let mut pts: Vec<PdfPoint> = Vec::with_capacity(5);
    for (i, seg) in entity.segments.iter().enumerate() {
        match (i, seg) {
            (0, PdfPathSegment::MoveTo(p)) => pts.push(*p),
            (1..=4, PdfPathSegment::LineTo(p)) => pts.push(*p),
            (_, PdfPathSegment::Close) if i == entity.segments.len() - 1 => {}
            _ => return None,
        }
    }
    if pts.len() == 5 {
        if pts[4] != pts[0] {
            return None;
        }
        pts.pop();
    }
    let [p0, p1, p2, p3] = <[PdfPoint; 4]>::try_from(pts).ok()?;
    let horizontal_first = p0[1] == p1[1] && p1[0] == p2[0] && p2[1] == p3[1] && p3[0] == p0[0];
    let vertical_first = p0[0] == p1[0] && p1[1] == p2[1] && p2[0] == p3[0] && p3[1] == p0[1];
    if !(horizontal_first || vertical_first) {
        return None;
    }
    Some(ExactBounds {
        min: [p0[0].min(p2[0]), p0[1].min(p2[1])],
        max: [p0[0].max(p2[0]), p0[1].max(p2[1])],
    })
}

/// Decide every painted entity of a page, and record every clip boundary.
///
/// # Errors
///
/// [`ClipError::NestingExceeded`] when more than `limits.max_clip_depth` clips are in force.
pub fn apply_clip_policy(
    page: &PdfVectorPage,
    limits: &ClipLimits,
    bounds: &dyn ExactBoundsSource,
) -> Result<ClipReport, ClipError> {
    let by_sequence: BTreeMap<u64, &PdfPathEntity> = page
        .entities
        .iter()
        .map(|e| (e.provenance.sequence, e))
        .collect();

    let mut clips = Vec::new();
    let mut rects: BTreeMap<u64, Option<ExactBounds>> = BTreeMap::new();
    let mut decisions = Vec::new();
    let mut max_depth = 0usize;

    for entity in &page.entities {
        // A clip takes effect after the path that defines it, so it counts one deeper there.
        let depth = entity.active_clips.len() + usize::from(entity.clip.is_some());
        if depth > limits.max_clip_depth as usize {
            return Err(ClipError::NestingExceeded {
                limit: limits.max_clip_depth,
                limit_name: Limit::BlockDepth.name(),
                observed: depth,
            });
        }
        max_depth = max_depth.max(depth);

        if let Some(rule) = entity.clip {
            let rect = exact_rect(entity);
            rects.insert(entity.provenance.sequence, rect);
            clips.push(ClipRecord {
                sequence: entity.provenance.sequence,
                rule,
                subpaths: subpath_count(entity),
                rect,
                conformance: if rect.is_some() {
                    ClipConformance::GeometricallyExact
                } else {
                    ClipConformance::DisplayFidelity
                },
            });
        }

        // A path that only defines a clip (`W n`) is not drawn.
        if entity.paint.fill.is_none() && !entity.paint.stroke {
            continue;
        }

        let mut d = EntityClipDecision {
            entity: entity.provenance.sequence,
            case: ClipCase::NoClip,
            clips: entity.active_clips.clone(),
            classification: None,
            conformance: None,
            fill_rule: entity.paint.fill,
            subpaths: subpath_count(entity),
        };
        if entity.soft_masked {
            d.case = ClipCase::Unsupported;
            d.classification = Some(Classification::refused(
                ReasonCode::PdfSoftMask,
                SOURCE_CLASS,
            ));
        } else if !entity.active_clips.is_empty() {
            let outside = !entity.paint.stroke
                && bounds.exact_bounds(entity).is_some_and(|eb| {
                    entity.active_clips.iter().any(|c| {
                        let clip_bounds = match rects.get(c) {
                            Some(Some(r)) => Some(*r),
                            _ => by_sequence.get(c).and_then(|ce| bounds.exact_bounds(ce)),
                        };
                        clip_bounds.is_some_and(|cb| eb.disjoint(&cb))
                    })
                });
            let all_rect = entity
                .active_clips
                .iter()
                .all(|c| matches!(rects.get(c), Some(Some(_))));
            if outside {
                d.case = ClipCase::FullyClipped;
                d.classification = Some(Classification::ignored_by_policy(
                    OUTSIDE_CLIP,
                    SOURCE_CLASS,
                ));
            } else if all_rect {
                d.case = ClipCase::ExactRectangular;
                d.conformance = Some(ClipConformance::GeometricallyExact);
            } else {
                d.case = ClipCase::BoundedConservative;
                d.conformance = Some(ClipConformance::DisplayFidelity);
            }
        }
        decisions.push(d);
    }

    Ok(ClipReport {
        clips,
        decisions,
        max_depth,
    })
}
