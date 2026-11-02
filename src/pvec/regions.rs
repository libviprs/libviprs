//! Derived room and corridor regions (issue #1093, N6.9).
//!
//! Stub: the types are here, the derivation follows the red tests.

use pvec_facade::components::{Origin, RegionKind, Stream};
use thiserror::Error;

use super::ordering::EntityId;

/// One wall segment, by the entity it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallEdge {
    /// The wall entity.
    pub entity: EntityId,
    /// Its start.
    pub start: [f64; 2],
    /// Its end.
    pub end: [f64; 2],
}

/// A text or block-attribute anchor that may name the room it sits in.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomLabel {
    /// Where it sits.
    pub at: [f64; 2],
    /// What it says.
    pub text: String,
}

/// A room the drawing states itself: a closed boundary polyline, or a tagged block.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitRoom {
    /// The boundary polygon, not repeating the first point.
    pub polygon: Vec<[f64; 2]>,
    /// The entities that draw it, in ring order.
    pub boundary: Vec<EntityId>,
    /// Its name, when it has one.
    pub name: Option<String>,
}

/// The knobs of the derivation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionConfig {
    /// The widest gap between two wall ends that is closed, in drawing units.
    pub gap_tolerance: f64,
    /// A face whose short side is at most this is a corridor candidate.
    pub corridor_max_width: f64,
    /// ... and whose long side is at least this many times the short one.
    pub corridor_min_aspect: f64,
    /// The most faces the step builds before it refuses.
    pub max_faces: usize,
    /// Keep faces no rule labels, as regions with no kind. Off omits them.
    pub keep_unlabelled: bool,
}

/// What a region boundary refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryRef {
    /// A source wall entity.
    Source(EntityId),
    /// The `n`th gap-closing segment of [`Regions::derived`].
    Derived(usize),
}

/// One derived region.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Room or corridor, `None` for a face no rule labels.
    pub kind: Option<RegionKind>,
    /// Inferred with a confidence, or explicit.
    pub origin: Origin,
    /// The ring, in order.
    pub boundary: Vec<BoundaryRef>,
    /// A name found inside the region.
    pub label: Option<String>,
}

/// A segment created by closing a gap. Never source geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedSegment {
    /// Start.
    pub start: [f64; 2],
    /// End.
    pub end: [f64; 2],
}

/// Counts for the conversion report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegionReport {
    /// Rooms found.
    pub rooms: u64,
    /// Corridors found.
    pub corridors: u64,
    /// Gaps closed.
    pub gaps_closed: u64,
    /// Faces no rule labelled.
    pub unlabelled: u64,
}

/// The result of the step.
#[derive(Debug, Clone, PartialEq)]
pub struct Regions {
    /// The regions.
    pub regions: Vec<Region>,
    /// The gap-closing segments, flagged derived by being here and not in the source geometry.
    pub derived: Vec<DerivedSegment>,
    /// The counts.
    pub report: RegionReport,
}

/// Why the step refused.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegionsError {
    /// More faces than [`RegionConfig::max_faces`].
    #[error("the walls enclose more than {limit} faces")]
    TooManyFaces {
        /// The limit.
        limit: usize,
    },
    /// A tolerance or limit that is not a usable number.
    #[error("bad region configuration: {0}")]
    BadConfig(&'static str),
}

impl RegionsError {
    /// The stable reason code.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            RegionsError::TooManyFaces { .. } => "regions.face-limit",
            RegionsError::BadConfig(_) => "regions.bad-config",
        }
    }
}

/// Builds the regions the walls enclose.
///
/// # Errors
///
/// [`RegionsError`].
pub fn derive_regions(
    walls: &[WallEdge],
    labels: &[RoomLabel],
    explicit: &[ExplicitRoom],
    config: &RegionConfig,
) -> Result<Regions, RegionsError> {
    let _ = (walls, labels, explicit, config);
    todo!("N6.9")
}

/// The semantic class stream for `regions`: PVec entity `i` is `source_ids[i]`, and the derived
/// segments were appended after the source geometry, so `Derived(n)` is entity
/// `source_ids.len() + n`. A region that points at an entity that was not written is left out.
///
/// # Errors
///
/// A message when PVec's encoder refuses the payload.
pub fn semantic_stream(regions: &Regions, source_ids: &[EntityId]) -> Result<Stream, String> {
    let _ = (regions, source_ids, RegionKind::Room);
    todo!("N6.9")
}
