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
//! `i`. A curve PVec's container refuses (`WRT-08`, one it cannot store exactly and
//! deterministically) is not written and takes no id: it comes back in
//! [`GeometryArtifact::refused`] with the rule, and the drawing box is that of what was written.
//! Component streams refer to entities by PVec's id, so they are built through it. The drawing unit must be one PVec can declare, and the drawing box that the manifest
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
//! [`write_geometry`] returns the artifact as one `Vec<u8>` from a writer that keeps what it is given
//! until `finish`, so its peak memory is linear in the entity count. [`write_geometry_to`] is the bounded
//! path: PVec's `with_spill` keeps the pushes in files under a root the caller names and `finish_to`
//! builds one block at a time and streams the artifact to a sink, so an indexed write stays inside its
//! budget however many entities it has (85 MiB at 1M and 83 MiB at 4M for a 64 MiB budget, against
//! 561 and 2,160 MiB in memory, `docs/BENCHMARKS.md` in `libviprs/pvec`). The curve through this
//! funnel is in `docs/adr/pvec-tiers-and-spool-rescope.md`.
//!
//! # Units
//!
//! PVec declares six length units. A drawing in any other unit (yards, miles, a unitless or an
//! unknown `$INSUNITS`) is refused with [`WriteError::UnsupportedUnit`] rather than rescaled:
//! rescaling would change every coordinate. The caller that wants a policy for unitless drawings
//! (N7.2) passes the unit it chose.

/// PVec's capability identifier, re-exported here because this file is the one place the crate
/// names the facade. The conversion API takes its allowed capabilities in this type.
pub use pvec_facade::CapabilityId;
use pvec_facade::components::Stream;
use pvec_facade::{Bounds2, EntityId as PvecEntityId, Manifest, PvecError, PvecWriter};
use std::io::{self, Write};
use std::path::PathBuf;

use thiserror::Error;

use super::bounds::{BoundsError, BoundsReport, bounds_inputs};
use super::normalized::Normalized2DGeometry;
use super::ordering::EntityId;
use super::units::SourceUnit;
use crate::cancel::CancelToken;

/// The writer rule for a curve PVec's container cannot store exactly and deterministically.
const REFUSED_UNSTORABLE: &str = "WRT-08";

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
    /// Entities PVec's writer refused (a curve its container cannot store exactly and
    /// deterministically), each with the writer rule that refused it. They are not written and
    /// take no PVec id.
    pub refused: Vec<(EntityId, String)>,
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

/// An artifact whose geometry is written and that is not sealed yet: the writer with everything pushed, and
/// the facts about what went in. [`open_geometry`] makes one so that a caller can add raster blocks through
/// [`OpenArtifact::rasters`] before [`OpenArtifact::finish`] seals it, while the geometry still goes through
/// this one funnel and the facade's writer stays inside it.
pub(crate) struct OpenArtifact {
    writer: PvecWriter,
    bounds: BoundsReport,
    written: u64,
    skipped: u64,
    source_ids: Vec<EntityId>,
    refused: Vec<(EntityId, String)>,
}

/// Maps `items` onto PVec ids, builds the manifest from PVec's own boxes, and pushes everything into a
/// writer, with a spill root and budget when the write streams. The write is cancelled between entities,
/// and `on_chunk` is called after each batch of [`CANCEL_EVERY`] pushed entities (and the last, short one).
fn prepare<'a>(
    config: &WriteConfig,
    items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    area: Option<[f64; 4]>,
    spill: Option<&SpillConfig>,
    cancel: &CancelToken,
    on_chunk: &mut dyn FnMut(u64, u64),
) -> Result<OpenArtifact, WriteError> {
    let metres_per_unit = metres_per_unit(config.unit)?;
    let mut skipped = 0u64;
    let mut kept = Vec::new();
    for (id, geometry) in items {
        match geometry.geometry() {
            Some(curve) => kept.push((id, curve)),
            None => skipped += 1,
        }
    }
    let mut refused = Vec::new();
    // A curve PVec refuses is dropped and the rest are written again, so the manifest box is the
    // union of what is in the file and the ids stay dense. The common run is one pass, and a
    // second one only follows a refusal, which the first pass names.
    let (writer, bounds, written) = loop {
        let curves = kept
            .iter()
            .enumerate()
            .map(|(n, (_, curve))| {
                let n = n as u64;
                u32::try_from(n)
                    .ok()
                    .and_then(|n| PvecEntityId::new(n).ok())
                    .map(|id| (id, *curve))
                    .ok_or(WriteError::Bounds(BoundsError::IdOutOfRange(n)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bounds = bounds_inputs(curves.iter().copied()).map_err(WriteError::Bounds)?;
        let drawing = bounds.drawing_bounds().ok_or(WriteError::NoGeometry)?;
        let manifest = Manifest {
            metres_per_unit,
            // The drawing's box, grown to take in `area` when the caller has content of its own beyond the
            // entities (a page's rasters).
            bounds: Bounds2 {
                min: [
                    area.map_or(drawing.min().x, |a| a[0].min(drawing.min().x)),
                    area.map_or(drawing.min().y, |a| a[1].min(drawing.min().y)),
                ],
                max: [
                    area.map_or(drawing.max().x, |a| a[2].max(drawing.max().x)),
                    area.map_or(drawing.max().y, |a| a[3].max(drawing.max().y)),
                ],
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
        if let Some(s) = spill {
            writer = writer
                .with_spill(&s.root, s.budget)
                .map_err(WriteError::Pvec)?;
        }
        // Positions in `kept` of the curves this pass could not write. Later curves are pushed
        // under the id they would have if the refused ones were already gone.
        let mut refused_here = Vec::new();
        let mut next = 0u32;
        for (n, (_, curve)) in curves.iter().enumerate() {
            if n % CANCEL_EVERY == 0 && cancel.is_cancelled() {
                return Err(WriteError::Cancelled);
            }
            let dense = PvecEntityId::new(next)
                .map_err(|_| WriteError::Bounds(BoundsError::IdOutOfRange(u64::from(next))))?;
            match writer.push(dense, curve) {
                Ok(()) => next += 1,
                Err(PvecError::Write { rule }) if rule == REFUSED_UNSTORABLE => {
                    refused_here.push((n, rule));
                }
                Err(e) => return Err(WriteError::Pvec(e)),
            }
            if (n + 1) % CANCEL_EVERY == 0 || n + 1 == curves.len() {
                on_chunk(n as u64 + 1, 0);
                if cancel.is_cancelled() {
                    return Err(WriteError::Cancelled);
                }
            }
        }
        if !refused_here.is_empty() {
            for (n, rule) in refused_here.into_iter().rev() {
                refused.push((kept.remove(n).0, rule));
            }
            continue;
        }
        break (writer, bounds, curves.len() as u64);
    };
    refused.sort_by_key(|(id, _)| *id);
    Ok(OpenArtifact {
        writer,
        bounds,
        written,
        skipped,
        source_ids: kept.iter().map(|(id, _)| *id).collect(),
        refused,
    })
}

/// Write `items` (in the order given, ids from [`order_entities`](super::ordering::order_entities))
/// and then `streams` as one artifact held in memory.
///
/// The result is deterministic: the same items, config and streams give the same bytes.
///
/// # Errors
///
/// [`WriteError`]: an unsupported unit, nothing to write, a cancellation (checked every
/// [`CANCEL_EVERY`] entities and before the artifact is sealed), or PVec's refusal with its rule
/// as the `source()`.
///
/// `on_chunk(entities, bytes)` is called after each batch of [`CANCEL_EVERY`] entities is pushed, with the
/// entities pushed so far in the pass and `0` bytes (nothing is written out until the end). If a pass has
/// to restart because PVec refused a curve, the batches of the new pass are reported again. A cancel that
/// fires inside `on_chunk` stops the write before the next batch.
pub fn write_geometry<'a>(
    config: &WriteConfig,
    items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    streams: Vec<Stream>,
    cancel: &CancelToken,
    on_chunk: &mut dyn FnMut(u64, u64),
) -> Result<GeometryArtifact, WriteError> {
    open_geometry(config, items, streams, None, None, cancel, on_chunk)?.finish()
}

/// As [`write_geometry`] up to the seal: the entities and `streams` are written and the artifact is handed
/// back open, so raster blocks can be added with [`OpenArtifact::rasters`] before [`OpenArtifact::finish`].
/// The manifest box is the entities' box, grown to take in `area` (`[min x, min y, max x, max y]`) when one
/// is given.
///
/// # Errors
///
/// As [`write_geometry`].
pub(crate) fn open_geometry<'a>(
    config: &WriteConfig,
    items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    streams: Vec<Stream>,
    area: Option<[f64; 4]>,
    spill: Option<&SpillConfig>,
    cancel: &CancelToken,
    on_chunk: &mut dyn FnMut(u64, u64),
) -> Result<OpenArtifact, WriteError> {
    let mut open = prepare(config, items, area, spill, cancel, on_chunk)?;
    open.components(streams, cancel)?;
    Ok(open)
}

impl OpenArtifact {
    /// Writes the raster pyramids into the artifact, `RasterId` by position (see
    /// [`write_rasters`](crate::raster_pvec::container::write_rasters)). Returns how many were written. Call
    /// it once: the ids are positions within the call, so a second call would start again at 0.
    ///
    /// # Errors
    ///
    /// [`RasterWriteError`](crate::raster_pvec::container::RasterWriteError).
    #[cfg(all(feature = "pvec-raster", feature = "pvec-convert"))]
    pub(crate) fn rasters<'a>(
        &mut self,
        pyramids: impl IntoIterator<Item = &'a crate::raster_pvec::pyramid::RasterPyramid>,
    ) -> Result<usize, crate::raster_pvec::container::RasterWriteError> {
        crate::raster_pvec::container::write_rasters(&mut self.writer, pyramids)
            .map(|ids| ids.len())
    }

    /// The libviprs id of each entity written; the index is the entity's PVec id. A component stream is
    /// built against these, because a refused curve moves every id after it.
    pub(crate) fn source_ids(&self) -> &[EntityId] {
        &self.source_ids
    }

    /// Adds component streams after the geometry.
    ///
    /// # Errors
    ///
    /// [`WriteError::Pvec`] when PVec refuses a stream, [`WriteError::Cancelled`] when the token fired.
    pub(crate) fn components(
        &mut self,
        streams: Vec<Stream>,
        cancel: &CancelToken,
    ) -> Result<(), WriteError> {
        for stream in streams {
            self.writer.components(stream).map_err(WriteError::Pvec)?;
        }
        if cancel.is_cancelled() {
            return Err(WriteError::Cancelled);
        }
        Ok(())
    }

    /// Seals the artifact.
    ///
    /// # Errors
    ///
    /// [`WriteError::Pvec`] when PVec refuses to seal it.
    pub(crate) fn finish(self) -> Result<GeometryArtifact, WriteError> {
        Ok(GeometryArtifact {
            bytes: self.writer.finish().map_err(WriteError::Pvec)?,
            bounds: self.bounds,
            written: self.written,
            skipped: self.skipped,
            source_ids: self.source_ids,
            refused: self.refused,
        })
    }
}

impl OpenArtifact {
    /// Seals the artifact straight into `out`, as [`write_geometry_to`] does: a cancel is seen at every
    /// write, and `on_chunk` hears each one. The writer must have been opened with a spill config.
    ///
    /// # Errors
    ///
    /// As [`write_geometry_to`].
    pub(crate) fn seal_into(
        self,
        cancel: &CancelToken,
        on_chunk: &mut dyn FnMut(u64, u64),
        out: &mut dyn Write,
    ) -> Result<StreamedArtifact, WriteError> {
        let length = self
            .writer
            .finish_to(&mut ChunkWrite {
                inner: out,
                cancel,
                on_chunk,
                entities: self.written,
                bytes: 0,
            })
            .map_err(|e| {
                if cancel.is_cancelled() {
                    WriteError::Cancelled
                } else {
                    WriteError::Pvec(e)
                }
            })?;
        if cancel.is_cancelled() {
            return Err(WriteError::Cancelled);
        }
        Ok(StreamedArtifact {
            length,
            bounds: self.bounds,
            written: self.written,
            skipped: self.skipped,
            source_ids: self.source_ids,
            refused: self.refused,
        })
    }
}

/// Where a streamed write keeps its spill files, and how much memory it may use before it spills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillConfig {
    /// The directory the spill files go under. Nothing is written anywhere else.
    pub root: PathBuf,
    /// The memory budget in bytes.
    pub budget: u64,
}

/// What a streamed write leaves behind besides the bytes: the same facts as [`GeometryArtifact`].
#[derive(Debug)]
pub struct StreamedArtifact {
    /// Bytes written to the sink.
    pub length: u64,
    /// One box per written entity and the drawing box.
    pub bounds: BoundsReport,
    /// Entities written.
    pub written: u64,
    /// Entities that carried no geometry.
    pub skipped: u64,
    /// The libviprs id of each written entity.
    pub source_ids: Vec<EntityId>,
    /// Entities PVec's writer refused, with the rule, as for [`GeometryArtifact::refused`].
    pub refused: Vec<(EntityId, String)>,
}

/// A sink that reports each write as a chunk and stops writing once the cancel token has fired, so a
/// flush ends at the next write.
struct ChunkWrite<'a> {
    inner: &'a mut dyn Write,
    cancel: &'a CancelToken,
    on_chunk: &'a mut dyn FnMut(u64, u64),
    entities: u64,
    bytes: u64,
}

impl Write for ChunkWrite<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            // Not `Interrupted`: `write_all` retries that kind forever.
            return Err(io::Error::other("cancelled"));
        }
        let n = self.inner.write(buf)?;
        self.bytes += n as u64;
        (self.on_chunk)(self.entities, self.bytes);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Write `items` and `streams` as one artifact straight to `out`, keeping the writer's pushes in files under
/// `spill.root` and its memory inside `spill.budget` (PVec's `with_spill` and `finish_to`, pvec#235). The
/// bytes are the ones [`write_geometry`] returns, whatever the budget.
///
/// The memory is bounded for an indexed write ([`WriteConfig::leaves`] set). A plain write spills too, but
/// holds one block per geometry type, which is the whole drawing, so it is not flat.
///
/// A cancel is seen between entities and at every write to `out`, so a flush ends at the next write with
/// [`WriteError::Cancelled`], and one that fires on the last write is seen once the flush returns.
/// `on_chunk(entities, bytes)` is called after each batch of pushed entities (bytes `0`) and after each write
/// the flush makes to `out` (all entities, bytes written so far); for a fixed input, config and budget the
/// calls are the same every run. PVec removes its spill files on success, on error and on drop, so none is left
/// whichever way this ends. What `out` has received on an error is a prefix of the artifact, which the caller
/// discards (`publish_streamed` does).
///
/// # Errors
///
/// [`WriteError`], as for [`write_geometry`]; a failing `out` or spill file is [`WriteError::Pvec`] with
/// PVec's `Io` error and the `io::Error` as its source.
pub fn write_geometry_to<'a>(
    config: &WriteConfig,
    items: impl IntoIterator<Item = (EntityId, &'a Normalized2DGeometry)>,
    streams: Vec<Stream>,
    spill: &SpillConfig,
    cancel: &CancelToken,
    on_chunk: &mut dyn FnMut(u64, u64),
    out: &mut dyn Write,
) -> Result<StreamedArtifact, WriteError> {
    open_geometry(config, items, streams, None, Some(spill), cancel, on_chunk)?
        .seal_into(cancel, on_chunk, out)
}
