//! The one funnel every PVec geometry write passes through (issue #1314, N6.1b).
//!
//! Normalized geometry goes in, a PVec artifact comes out, and the bytes are written by the
//! facade's `PvecWriter` and nothing else. `tests/pvec_api_surface.rs` holds the line: a writer
//! call anywhere else under `src/` (the two raster-stream files apart, which write tiles) turns it
//! red, so the set of things libviprs asks PVec's writer to do is the set in this file.
//!
//! # What the funnel decides, and what it leaves to PVec
//!
//! It maps. An entity that carries no geometry (an unsupported or ignored one) is counted in
//! [`GeometryArtifact::skipped`] and not written. The others get PVec's file-local ids **dense from
//! zero, in the order given** (Tier 0 needs ids `0..n` with no gaps), and
//! [`GeometryArtifact::source_ids`] is the way back: element `i` is the libviprs id of PVec entity
//! `i`. Component streams refer to entities by PVec's id, so they are built through it. The drawing unit must be one PVec can declare, and the drawing box that the manifest
//! needs is the union of the entity boxes PVec itself reports ([`bounds_inputs`]).
//!
//! Everything about layout is PVec's. With a [`LeafConfig`] the funnel asks `PvecWriter::new_indexed`
//! for a spatial directory and Tier 0 blocks and gives it the two numbers it takes, the leaf size
//! and the preview bound. libviprs assigns no tier and defines no tolerance of its own: which
//! entity is exact in Tier 0 and which gets a preview is PVec's `Tier0Builder`, and a preview never
//! replaces the analytic curve in the artifact (see `docs/adr/pvec-tiers-and-spool-rescope.md`).
//!
//! # Memory
//!
//! The facade's writer keeps what it is given until `finish`, and `finish` returns the artifact
//! as one `Vec<u8>`. Peak memory is therefore linear in the entity count, and this funnel cannot
//! change that. The measured curve is in the ADR above.
//!
//! # Units
//!
//! PVec declares six length units. A drawing in any other unit (yards, miles, a unitless or an
//! unknown `$INSUNITS`) is refused with [`WriteError::UnsupportedUnit`] rather than rescaled:
//! rescaling would change every coordinate. The caller that wants a policy for unitless drawings
//! (N7.2) passes the unit it chose.

use pvec_facade::components::Stream;
use pvec_facade::{Bounds2, EntityId as PvecEntityId, Manifest, PvecError, PvecWriter};
use thiserror::Error;

use super::bounds::{BoundsError, BoundsReport, bounds_inputs};
use super::normalized::Normalized2DGeometry;
use super::ordering::EntityId;
use super::units::SourceUnit;
use crate::cancel::CancelToken;

/// How often the write looks at the cancel token, in entities.
const CANCEL_EVERY: usize = 1024;

/// The leaf size and the Tier 0 preview bound, the two numbers PVec's indexed writer takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeafConfig {
    /// Entities per leaf of PVec's spatial directory.
    pub cell_capacity: u32,
    /// The bound, in source units, on a Tier 0 preview's error. `None` writes no Tier 0 blocks.
    /// PVec's own default for a drawing is `PREVIEW_BUDGET_PX * extent / 1920`.
    pub tier0_max_error: Option<f64>,
}

/// What a write needs besides the entities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WriteConfig {
    /// The drawing's unit.
    pub unit: SourceUnit,
    /// Lay a spatial directory and Tier 0 blocks into the artifact. `None` writes a plain file.
    pub leaves: Option<LeafConfig>,
}

/// An artifact and what went into it.
#[derive(Debug)]
pub struct GeometryArtifact {
    /// The artifact.
    pub bytes: Vec<u8>,
    /// One box per written entity, how it was obtained, and the drawing box.
    pub bounds: BoundsReport,
    /// Entities written.
    pub written: u64,
    /// Entities that carried no geometry and were not written.
    pub skipped: u64,
    /// The libviprs id of each written entity; the index is the entity's PVec id.
    pub source_ids: Vec<EntityId>,
}

/// Why a write failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WriteError {
    /// The drawing's unit is not one PVec can declare, and it is not rescaled.
    #[error("PVec cannot declare the unit {0:?}")]
    UnsupportedUnit(SourceUnit),
    /// No entity carried geometry, so there is no artifact to write.
    #[error("no entity carries geometry")]
    NoGeometry,
    /// The cancel token fired. Nothing was written.
    #[error("cancelled")]
    Cancelled,
    /// A box or an id could not be produced.
    #[error("an entity's box or id could not be produced")]
    Bounds(#[source] BoundsError),
    /// PVec's writer refused an entity, a stream or the artifact.
    #[error("PVec's writer refused: {0}")]
    Pvec(#[source] PvecError),
}

/// PVec's metres-per-unit for the units it can declare. The values are bit-equal to PVec's own,
/// which `Manifest` checks when the writer is built and the tests pin on the read-back.
fn metres_per_unit(unit: SourceUnit) -> Result<f64, WriteError> {
    match unit {
        SourceUnit::Meters => Ok(1.0),
        SourceUnit::Millimeters => Ok(0.001),
        SourceUnit::Centimeters => Ok(0.01),
        SourceUnit::Kilometers => Ok(1000.0),
        SourceUnit::Inches => Ok(0.0254),
        SourceUnit::Feet => Ok(0.3048),
        other => Err(WriteError::UnsupportedUnit(other)),
    }
}

/// Write `items` (in the order given, ids from [`order_entities`](super::ordering::order_entities))
/// and then `streams` as one artifact.
///
/// The result is deterministic: the same items, config and streams give the same bytes.
///
/// # Errors
///
/// [`WriteError`]: an unsupported unit, nothing to write, a cancellation (checked every
/// [`CANCEL_EVERY`] entities and before the artifact is sealed), or PVec's refusal with its rule
/// as the `source()`.
pub fn write_geometry<'a>(
    config: &WriteConfig,
    items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    streams: Vec<Stream>,
    cancel: &CancelToken,
) -> Result<GeometryArtifact, WriteError> {
    let metres_per_unit = metres_per_unit(config.unit)?;
    let mut skipped = 0u64;
    let mut curves = Vec::new();
    let mut source_ids = Vec::new();
    for (id, geometry) in items {
        match geometry.geometry() {
            Some(curve) => {
                let next = curves.len() as u64;
                let pvec_id = u32::try_from(next)
                    .ok()
                    .and_then(|n| PvecEntityId::new(n).ok())
                    .ok_or(WriteError::Bounds(BoundsError::IdOutOfRange(next)))?;
                curves.push((pvec_id, curve));
                source_ids.push(id);
            }
            None => skipped += 1,
        }
    }
    let bounds = bounds_inputs(curves.iter().copied()).map_err(WriteError::Bounds)?;
    let drawing = bounds.drawing_bounds().ok_or(WriteError::NoGeometry)?;
    let manifest = Manifest {
        metres_per_unit,
        bounds: Bounds2 {
            min: [drawing.min().x, drawing.min().y],
            max: [drawing.max().x, drawing.max().y],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
    };
    let mut writer = match config.leaves {
        Some(l) => PvecWriter::new_indexed(manifest, l.cell_capacity, l.tier0_max_error),
        None => PvecWriter::new(manifest),
    }
    .map_err(WriteError::Pvec)?;
    for (n, (id, curve)) in curves.iter().enumerate() {
        if n % CANCEL_EVERY == 0 && cancel.is_cancelled() {
            return Err(WriteError::Cancelled);
        }
        writer.push(*id, curve).map_err(WriteError::Pvec)?;
    }
    for stream in streams {
        writer.components(stream).map_err(WriteError::Pvec)?;
    }
    if cancel.is_cancelled() {
        return Err(WriteError::Cancelled);
    }
    let written = curves.len() as u64;
    Ok(GeometryArtifact {
        bytes: writer.finish().map_err(WriteError::Pvec)?,
        bounds,
        written,
        skipped,
        source_ids,
    })
}
