//! The one funnel every PVec geometry write passes through (issue #1314, N6.1b).
//!
//! The signatures are in place and the body refuses everything, so the tests that describe the
//! funnel run and fail before the implementation lands.

use pvec_facade::components::Stream;
use thiserror::Error;

use super::bounds::{BoundsError, BoundsReport};
use super::normalized::Normalized2DGeometry;
use super::ordering::EntityId;
use super::units::SourceUnit;
use crate::cancel::CancelToken;

/// The leaf size and the Tier 0 preview bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeafConfig {
    /// Entities per leaf.
    pub cell_capacity: u32,
    /// Bound on a Tier 0 preview's error.
    pub tier0_max_error: Option<f64>,
}

/// What a write needs besides the entities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WriteConfig {
    /// The drawing's unit.
    pub unit: SourceUnit,
    /// Leaves and Tier 0.
    pub leaves: Option<LeafConfig>,
}

/// An artifact and what went into it.
#[derive(Debug)]
pub struct GeometryArtifact {
    /// The artifact.
    pub bytes: Vec<u8>,
    /// Boxes.
    pub bounds: BoundsReport,
    /// Entities written.
    pub written: u64,
    /// Entities skipped.
    pub skipped: u64,
}

/// Why a write failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WriteError {
    /// Not written yet.
    #[error("the funnel is not implemented")]
    Unimplemented,
    /// Unit.
    #[error("unit {0:?}")]
    UnsupportedUnit(SourceUnit),
    /// Nothing.
    #[error("no geometry")]
    NoGeometry,
    /// Cancelled.
    #[error("cancelled")]
    Cancelled,
    /// Box.
    #[error("bounds")]
    Bounds(#[source] BoundsError),
    /// PVec.
    #[error("pvec")]
    Pvec(#[source] pvec_facade::PvecError),
}

/// Write `items` as one artifact.
///
/// # Errors
///
/// Always [`WriteError::Unimplemented`] for now.
pub fn write_geometry<'a>(
    _config: &WriteConfig,
    _items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    _streams: Vec<Stream>,
    _cancel: &CancelToken,
) -> Result<GeometryArtifact, WriteError> {
    Err(WriteError::Unimplemented)
}
