//! A [`TileSink`] that streams a raster pyramid into a PVec artifact (issue #1331, RL1.3).
//!
//! [`PvecSink`] is the engine-facing half of the raster producer, beside
//! [`PmTilesSink`](crate::sink_pmtiles::PmTilesSink). The engine hands it tiles in any
//! [`EmissionOrder`](crate::sink::EmissionOrder) and from any number of workers; each one is
//! encoded, held to the legal table and LIMIT-15 ([`crate::raster_pvec::tiles`]), and given to
//! PVec's writer with its `(level, col, row)`. The writer owns the order: PVec lays the tiles out
//! in Hilbert order whatever order they came in, so the artifact is the same bytes for one worker
//! or eight, which a test holds.
//!
//! ```no_run
//! use libviprs::raster_pvec::pyramid::{RasterPlacement, RasterPyramidOptions, prepare_raster};
//! use libviprs::sink_pvec::PvecSink;
//! use libviprs::{EngineBuilder, Layout, PyramidPlanner, Raster};
//! # fn run(raster: Raster, writer: pvec_facade::PvecWriter) -> Result<(), Box<dyn std::error::Error>> {
//! let plan = PyramidPlanner::new(raster.width(), raster.height(), 512, 0, Layout::Google)?.plan();
//! let options = RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]));
//! let job = prepare_raster(&raster, &plan, &options)?;
//! let sink = PvecSink::new(writer, &job.pyramid, &options)?;
//! EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink).run()?;
//! sink.finish_to("scan.pvec".as_ref(), "/var/tmp".as_ref())?;
//! # Ok(())
//! # }
//! ```
//!
//! # Streamed, not spooled
//!
//! The issue offered two shapes: stream if PVec's raster write takes tiles incrementally, spool to
//! a staged temp file and replay if it wants the set. `PvecWriter::raster_tiles(raster, tiles)`
//! (`pvec::unstable`, `crates/pvec/src/unstable.rs` in libviprs/pvec) takes "any number of calls
//! and any order", so this sink streams and writes **no spool file**. A spool would not have
//! helped anyway, because the replay has to hand the same bytes to the same in-memory writer.
//!
//! # What "bounded" means here
//!
//! The pixel side is bounded as it is for every sink: the engine's strip budget
//! ([`StreamingConfig`](crate::streaming::StreamingConfig)) holds, and a tile is encoded and
//! dropped. The encoded side is not: PVec's writer keeps the encoded tiles of the artifact until
//! `PvecWriter::finish` and returns the artifact as one `Vec<u8>`, so the sink's peak is the
//! working set plus the encoded pyramid (plus the writer's index, about 24 bytes a tile) and the
//! copy `finish` makes. Making that flat needs a writer that emits chunks as they fill, which is
//! PVec's side (libviprs/pvec#218 and #220) and out of scope here. The measured numbers are in
//! `tests/pvec_1331_bounded_memory.rs` and `docs/pmtiles-benchmarks.md`.
//!
//! # One raster, and the file
//!
//! A sink places one raster, `RasterId` 0 (PVec's `rasters` takes the placements of a whole
//! artifact at once, and [`write_rasters`](crate::raster_pvec::container::write_rasters) is the
//! several-raster path). The artifact still needs an entity (PVec's WRT-01), so the caller builds
//! the writer, pushes the sheet's linework, and hands it in.
//!
//! [`PvecSink::finish_to`] finishes the writer and publishes the bytes: written to a staging file
//! under a caller-supplied root (never the destination directory), synced, and renamed over the
//! destination. Nothing appears at the destination until the whole artifact is written, and the
//! staging file is removed on every exit path. The staging root has to be on the destination's
//! filesystem for the rename to work.

use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pvec_facade::PvecWriter;
use pvec_facade::unstable::{RasterId, TileIn};
use thiserror::Error;

use crate::engine::EngineConfig;
use crate::engine::is_blank_tile;
use crate::planner::PyramidPlan;
use crate::raster_pvec::container::{RasterWriteError, placement};
use crate::raster_pvec::pyramid::{RasterPyramid, RasterPyramidOptions, cut_tile};
use crate::raster_pvec::tiles::check_tile_len;
use crate::sink::{SinkError, Tile, TileFormat, TileSink};

/// Why the artifact could not be finished or published.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PvecSinkError {
    /// PVec's writer refused the artifact at `finish`.
    #[error("PVec's writer refused the artifact")]
    Pvec(#[source] pvec_facade::PvecError),
    /// Staging or publishing the bytes failed.
    #[error("publishing the artifact failed")]
    Io(#[source] std::io::Error),
}

/// Streams the tiles of one raster into a PVec writer. See the module docs.
pub struct PvecSink {
    writer: Mutex<PvecWriter>,
    plan: PyramidPlan,
    format: TileFormat,
    skip_blanks: bool,
    config: Mutex<Option<EngineConfig>>,
    stored: AtomicU64,
    dropped: AtomicU64,
}

impl PvecSink {
    /// Attach the placement of `pyramid` (from
    /// [`prepare_raster`](crate::raster_pvec::pyramid::prepare_raster)) to `writer` as raster 0,
    /// and encode tiles the way `options` says. `options.skip_blanks` is applied here and not only
    /// by the engine, because the streaming engines ignore it: a uniform tile is not stored.
    ///
    /// # Errors
    ///
    /// [`RasterWriteError`] when the placement does not fit PVec's fields or PVec's writer
    /// refuses it (a singular map, an attribution that breaks RASTER-07).
    pub fn new(
        mut writer: PvecWriter,
        pyramid: &RasterPyramid,
        options: &RasterPyramidOptions,
    ) -> Result<Self, RasterWriteError> {
        writer
            .rasters(&[placement(pyramid)?])
            .map_err(RasterWriteError::Pvec)?;
        Ok(Self {
            writer: Mutex::new(writer),
            plan: pyramid.plan.clone(),
            format: options.tile_format,
            skip_blanks: options.skip_blanks,
            config: Mutex::new(None),
            stored: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        })
    }

    /// Tiles handed to PVec's writer so far.
    #[must_use]
    pub fn stored_tiles(&self) -> u64 {
        self.stored.load(Ordering::Relaxed)
    }

    /// Tiles of the engine's Google grid beyond a level's `cols` by `rows`, dropped as padding.
    #[must_use]
    pub fn dropped_padding_tiles(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// The writer, with the raster's tiles in it, for the caller to add to and finish.
    #[must_use]
    pub fn into_writer(self) -> PvecWriter {
        self.writer
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Finish the artifact and publish it at `dest`, staging the bytes under `staging` first.
    ///
    /// # Errors
    ///
    /// [`PvecSinkError::Pvec`] when the writer refuses the artifact, [`PvecSinkError::Io`] when
    /// staging, syncing or the rename fails (a staging root on another filesystem included).
    /// Nothing is left in `staging` or at `dest` after an error.
    pub fn finish_to(self, dest: &Path, staging: &Path) -> Result<(), PvecSinkError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let bytes = self.into_writer().finish().map_err(PvecSinkError::Pvec)?;
        let staged = staging.join(format!(
            ".pvec-sink-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let publish = || -> std::io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&staged, dest)
        };
        publish().map_err(|e| {
            // Whatever failed, the staging file does not outlive the call.
            let _ = std::fs::remove_file(&staged);
            PvecSinkError::Io(e)
        })
    }
}

impl TileSink for PvecSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        if self.skip_blanks && is_blank_tile(&tile.raster) {
            return Ok(());
        }
        let Some(bytes) = cut_tile(&self.plan, self.format, &self.config, tile)? else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        check_tile_len(self.plan.tile_size, bytes.len())
            .map_err(|e| SinkError::Other(e.to_string()))?;
        let c = tile.coord;
        let tile = TileIn {
            level: c.level,
            col: c.col,
            row: c.row,
            bytes,
        };
        let id = RasterId::new(0).expect("0 is a valid RasterId");
        crate::poison::recover(&self.writer)
            .raster_tiles(id, vec![tile])
            .map_err(|e| SinkError::Other(format!("PVec's writer refused a tile: {e}")))?;
        self.stored.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn record_engine_config(&self, config: &EngineConfig) {
        *crate::poison::recover(&self.config) = Some(config.clone());
    }
}
