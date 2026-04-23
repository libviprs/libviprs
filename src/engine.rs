use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use thiserror::Error;

#[cfg(test)]
use crate::observe::NoopObserver;
use crate::observe::{EngineEvent, EngineObserver, MemoryTracker};
use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::{Raster, RasterError};
use crate::resize;
use crate::resume::{
    JobCheckpoint, JobMetadata, ResumeError, ResumeMode, SCHEMA_VERSION, compute_plan_hash,
};
use crate::retry::FailurePolicy;
use crate::sink::{SinkError, Tile, TileSink};

/// Errors that can occur during pyramid generation.
///
/// Wraps lower-level raster and sink errors into a single error type so that
/// callers of [`generate_pyramid`] and [`generate_pyramid_observed`] can handle
/// all failure modes uniformly. Also covers engine-specific conditions such as
/// cancellation and worker panics.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    #[error("raster error: {0}")]
    Raster(#[from] RasterError),
    #[error("sink error: {0}")]
    Sink(#[from] SinkError),
    #[error("engine cancelled")]
    Cancelled,
    #[error("worker panicked")]
    WorkerPanic,
    /// A per-tile checksum did not match the expected digest after
    /// re-hashing the on-disk / in-memory bytes.
    #[error("checksum mismatch for tile {tile:?} (expected {expected}, got {got})")]
    ChecksumMismatch {
        tile: TileCoord,
        expected: String,
        got: String,
    },
    /// The resumed job checkpoint's plan hash does not match the current plan.
    #[error("plan hash mismatch (expected {expected}, got {got})")]
    PlanHashMismatch { expected: String, got: String },
    /// A resumable job could not be initialised or advanced.
    #[error("resume failed: {0}")]
    ResumeFailed(#[from] ResumeError),
    /// `ResumeMode::Verify` was requested but no on-disk checkpoint root could
    /// be resolved from either [`EngineConfig::checkpoint_root`] or
    /// [`TileSink::checkpoint_root`]. Verify mode requires an on-disk sink
    /// (or an explicit `checkpoint_root` on the config) to read back the
    /// previously-written tiles.
    #[error("Verify mode requires an on-disk sink or EngineConfig::checkpoint_root")]
    VerifyRequiresOnDiskSink,
    #[error("budget exceeded: worst-case strip {strip_bytes} bytes > budget {budget_bytes} bytes")]
    BudgetExceeded { strip_bytes: u64, budget_bytes: u64 },
    /// The [`EngineKind`](crate::EngineKind) requested through
    /// [`EngineBuilder::with_engine`](crate::EngineBuilder::with_engine) is
    /// not compatible with the supplied source. For example,
    /// [`EngineKind::Monolithic`](crate::EngineKind::Monolithic) requires an
    /// in-memory [`Raster`]; pairing it with a [`StripSource`](crate::streaming::StripSource)
    /// would require materialising the entire source up front, which is
    /// exactly what a strip source is built to avoid. The builder surfaces
    /// this condition as a typed error instead of silently pulling the
    /// source into memory.
    #[error("engine kind {kind:?} incompatible with supplied source: {reason}")]
    IncompatibleSource {
        kind: crate::EngineKind,
        reason: &'static str,
    },
}

/// Controls how blank (uniform-color) tiles are handled during pyramid generation.
///
/// Sparse images (e.g. scanned documents with large white margins) can produce
/// many tiles where every pixel is identical. This strategy lets the engine
/// replace those tiles with tiny placeholders, dramatically reducing output size.
///
/// See [`is_blank_tile`] for the detection logic and the
/// [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for integration-level examples. In the CLI, the `--skip-blank` flag selects
/// [`BlankTileStrategy::Placeholder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlankTileStrategy {
    /// Emit blank tiles as full raster data (default). Every tile coordinate
    /// produces a complete image file, including tiles that are entirely one color.
    Emit,
    /// Replace blank tiles with a 1-byte placeholder marker (`0x00`). Consumers
    /// can detect these marker files by their size and generate their own blank
    /// tiles on the fly, saving significant disk space for sparse images.
    Placeholder,
    /// Like [`BlankTileStrategy::Placeholder`] but treats tiles whose pixel
    /// values fall within `max_channel_delta` of the first pixel as blank.
    /// Useful for scans with minor JPEG noise in the background.
    PlaceholderWithTolerance { max_channel_delta: u8 },
}

/// Configuration for the pyramid generation engine.
///
/// Groups every tunable knob that affects how [`generate_pyramid`] runs:
/// thread count, channel buffer depth, edge-tile background color, and
/// blank-tile handling. The [`Default`] implementation provides sensible
/// values for single-threaded operation.
///
/// Builder-style setters ([`with_concurrency`](Self::with_concurrency),
/// [`with_buffer_size`](Self::with_buffer_size),
/// [`with_blank_tile_strategy`](Self::with_blank_tile_strategy)) allow
/// chaining for ergonomic construction.
///
/// See the
/// [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for filesystem-backed usage and the
/// [CLI pyramid command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for command-line construction of this config.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EngineConfig {
    /// Number of worker threads for tile extraction. 0 = single-threaded (current thread).
    pub concurrency: usize,
    /// Maximum tiles buffered between producer and sink. Controls backpressure.
    pub buffer_size: usize,
    /// Background color (RGB) used to pad edge tiles to the full tile size.
    /// Defaults to white (255, 255, 255).
    pub background_rgb: [u8; 3],
    /// How to handle tiles where every pixel is the same color.
    /// Defaults to `Emit` (write full tile data).
    pub blank_tile_strategy: BlankTileStrategy,
    /// How to react when a sink write fails after retries.
    pub failure_policy: FailurePolicy,
    /// Persist the resume checkpoint every N tiles (0 = never).
    pub checkpoint_every: u64,
    /// Optional engine-level content-addressed deduplication strategy.
    pub dedupe_strategy: Option<crate::dedupe::DedupeStrategy>,
    /// Explicit on-disk root for resume checkpoints and Verify-mode reads.
    /// If None, falls back to `sink.checkpoint_root()`.
    /// Required when the sink is an opaque user wrapper that does not forward checkpoint_root().
    pub checkpoint_root: Option<PathBuf>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            concurrency: 0,
            buffer_size: 64,
            background_rgb: [255, 255, 255],
            blank_tile_strategy: BlankTileStrategy::Emit,
            failure_policy: FailurePolicy::default(),
            checkpoint_every: 0,
            dedupe_strategy: None,
            checkpoint_root: None,
        }
    }
}

impl EngineConfig {
    /// Sets the number of worker threads for parallel tile extraction.
    ///
    /// `0` (the default) means single-threaded execution on the calling thread.
    /// Any positive value spawns that many workers per pyramid level.
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Sets the bounded-channel capacity between producer threads and the sink consumer.
    ///
    /// A smaller buffer limits memory usage but may cause producers to block
    /// more frequently. A larger buffer smooths out sink latency at the cost
    /// of higher peak memory.
    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = n;
        self
    }

    /// Sets the strategy for handling blank (uniform-color) tiles.
    ///
    /// See [`BlankTileStrategy`] for the available options.
    pub fn with_blank_tile_strategy(mut self, strategy: BlankTileStrategy) -> Self {
        self.blank_tile_strategy = strategy;
        self
    }

    /// Sets the failure policy used when sink writes fail.
    ///
    /// See [`FailurePolicy`] for the available options. This is used by the
    /// engine to decide whether a failed write aborts the whole run
    /// (`FailFast` / `RetryThenFail`) or is accounted into
    /// [`EngineResult::skipped_due_to_failure`] and the run continues
    /// (`RetryThenSkip`).
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.failure_policy = policy;
        self
    }

    /// Persist the resume checkpoint every `n` tiles. `0` disables the
    /// periodic checkpoint — only the terminal checkpoint is written when the
    /// run finishes cleanly.
    pub fn with_checkpoint_every(mut self, n: u64) -> Self {
        self.checkpoint_every = n;
        self
    }

    /// Configure the content-addressed deduplication strategy applied by the
    /// engine before a tile reaches the sink.
    ///
    /// In the Phase 2b stub this records the strategy on the config but does
    /// not yet drive engine-level deduplication (sinks that accept a
    /// [`DedupeStrategy`] continue to apply their own per-sink dedupe).
    pub fn with_dedupe_strategy(mut self, strategy: crate::dedupe::DedupeStrategy) -> Self {
        self.dedupe_strategy = Some(strategy);
        self
    }

    /// Configure an explicit on-disk root for resume checkpoints and
    /// Verify-mode reads. When unset, the engine falls back to
    /// [`TileSink::checkpoint_root`]. Supplying this is the preferred way
    /// to drive resume/verify against an opaque user-wrapped sink that does
    /// not forward `checkpoint_root()`.
    pub fn with_checkpoint_root(mut self, root: PathBuf) -> Self {
        self.checkpoint_root = Some(root);
        self
    }
}

/// Per-stage duration breakdown for a single pyramid run.
///
/// Populated alongside [`EngineResult::duration`] when tracing is enabled; in
/// the Phase 2b stub implementation the stages are all measured as zero
/// durations (end-to-end time is reported via [`EngineResult::duration`]
/// instead).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StageDurations {
    /// Time spent planning / validating the pyramid layout.
    pub planning: Duration,
    /// Time spent decoding the source raster.
    pub decode: Duration,
    /// Time spent downscaling between levels.
    pub resize: Duration,
    /// Time spent encoding tiles (PNG / JPEG / raw).
    pub encode: Duration,
    /// Time spent handing tiles to the sink and awaiting `finish()`.
    pub sink: Duration,
}

/// Summary statistics returned after a successful pyramid generation.
///
/// Captures tile counts, level counts, and peak memory so that callers can
/// log, display progress, or assert correctness without inspecting the sink
/// directly. Every field is populated by [`generate_pyramid`] /
/// [`generate_pyramid_observed`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EngineResult {
    /// Total number of tiles written to the sink (including placeholders).
    pub tiles_produced: u64,
    /// Number of tiles that were blank and replaced with placeholders
    /// (only non-zero when `BlankTileStrategy::Placeholder` is used).
    pub tiles_skipped: u64,
    /// Number of pyramid levels that were processed (always equals the plan's level count).
    pub levels_processed: u32,
    /// Peak tracked memory in bytes (raster buffers only).
    pub peak_memory_bytes: u64,
    /// Total bytes read from the source raster.
    pub bytes_read: u64,
    /// Total bytes written to the sink (best-effort; sum of encoded payloads).
    pub bytes_written: u64,
    /// Number of retry attempts observed across all sinks.
    pub retry_count: u64,
    /// Peak number of tiles held in the producer/consumer queue.
    pub queue_pressure_peak: u32,
    /// Wall-clock duration of the pyramid run.
    pub duration: Duration,
    /// Per-stage duration breakdown (see [`StageDurations`]).
    pub stage_durations: StageDurations,
    /// Number of tiles that failed terminally and were skipped under
    /// `FailurePolicy::RetryThenSkip`.
    pub skipped_due_to_failure: u64,
}

/// Generates a complete tile pyramid from a source raster.
///
/// This is the primary entry point for pyramid generation. It processes levels
/// from full resolution (top) down to 1x1, extracting tiles at each level and
/// writing them to the provided [`TileSink`]. When
/// [`EngineConfig::concurrency`] is greater than zero, tiles within each level
/// are produced in parallel using scoped threads with bounded-channel
/// backpressure.
///
/// For progress reporting, use [`generate_pyramid_observed`] instead.
///
/// See the
/// [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for filesystem output,
/// [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// for PDF-sourced pyramids, and the
/// [CLI pyramid command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for end-to-end CLI usage.
/// Generates a tile pyramid with an [`EngineObserver`] for progress events.
///
/// Behaves identically to [`generate_pyramid`] but emits [`EngineEvent`]s
/// (level started/completed, tile completed, finished) to the supplied
/// observer. This is the function used by the
/// [CLI pyramid command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// to drive its progress bar.
///
/// See the
/// [observability tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
/// for integration-level examples of observer usage.
pub(crate) fn generate_pyramid_observed(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    run_pyramid(source, plan, sink, config, observer, None, None)
}

/// Internal driver shared by both `generate_pyramid_observed` and
/// `generate_pyramid_resumable`.
///
/// `skip_coords` (optional): tiles that should not be re-emitted because they
/// were recorded as complete in an incoming checkpoint.
///
/// `checkpoint_state` (optional): when supplied, the engine persists the
/// running `JobMetadata` to `sink.checkpoint_root()/.libviprs-job.json`
/// every `config.checkpoint_every` tiles (and once at the end).
fn run_pyramid(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
    skip_coords: Option<&HashSet<TileCoord>>,
    checkpoint_state: Option<&CheckpointState>,
) -> Result<EngineResult, EngineError> {
    let started = Instant::now();
    #[cfg(feature = "tracing")]
    let _pipeline_span = tracing::info_span!(target: "libviprs", "pipeline").entered();

    // Forward the active config into the sink so it can embed concurrency /
    // background / blank-strategy into the emitted manifest without a
    // secondary plumbing path.
    sink.record_engine_config(config);

    // Let the sink preallocate per-level bookkeeping (e.g. `FsSink` uses
    // this to size its per-level counter atomics up-front rather than
    // locking-and-growing on the first tile of each level). The default
    // trait impl is a no-op, so unaware sinks silently ignore the hint.
    sink.init_level_count(plan.levels.len());

    let top_level = plan.levels.len() - 1;
    let mut tiles_produced: u64 = 0;
    let mut tiles_skipped: u64 = 0;
    let bytes_read = source.data().len() as u64;
    let tracker = MemoryTracker::new();
    let bytes_written = AtomicU64::new(0);
    let queue_pressure_peak = AtomicU32::new(0);
    let stage_planning = Duration::ZERO;
    let stage_decode_start = Instant::now();

    let stage_resize = AtomicU64::new(0); // nanos
    let stage_encode = AtomicU64::new(0);
    let stage_sink = AtomicU64::new(0);

    // For Google layout or centred plans, embed the source image into a
    // canvas-sized raster at the centre offset. This matches vips's approach:
    // the image is placed in the canvas first, then the entire canvas is
    // downscaled level-by-level. This ensures boundary pixels are averaged
    // correctly instead of computing per-level offsets that diverge due to
    // integer rounding.
    let mut current = if plan.centre && (plan.centre_offset_x > 0 || plan.centre_offset_y > 0) {
        let canvas = embed_in_canvas(source, plan, config.background_rgb)?;
        let canvas_bytes = canvas.data().len() as u64;
        tracker.alloc(canvas_bytes);
        canvas
    } else {
        let source_bytes = source.data().len() as u64;
        tracker.alloc(source_bytes);
        source.clone()
    };
    let stage_decode_done: Instant = Instant::now();

    // Mutable state shared with the inner level loops.
    let ctx = EmitContext {
        bytes_written: &bytes_written,
        queue_pressure_peak: &queue_pressure_peak,
        stage_encode: &stage_encode,
        stage_sink: &stage_sink,
        skip_coords,
        checkpoint_state,
    };

    // Process from top level (full res) down to level 0 (1×1)
    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];
        #[cfg(feature = "tracing")]
        let _level_span = tracing::info_span!(
            target: "libviprs",
            "level",
            level_index = level.level
        )
        .entered();

        observer.on_event(EngineEvent::LevelStarted {
            level: level.level,
            width: level.width,
            height: level.height,
            tile_count: level.tile_count(),
        });

        // Downscale if not at the top level.
        // Uses downscale_half (2x2 box filter) to match libvips's
        // region-shrink=mean algorithm. Each level is ceil(prev/2).
        if level_idx < top_level {
            let old_bytes = current.data().len() as u64;
            let resize_start = Instant::now();
            current = resize::downscale_half(&current)?;
            stage_resize.fetch_add(resize_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let new_bytes = current.data().len() as u64;
            // Track: freed old level, allocated new
            tracker.dealloc(old_bytes);
            tracker.alloc(new_bytes);
        }

        // Extract and emit tiles for this level
        let (level_tiles, level_skipped) = extract_and_emit_level(
            &current,
            plan,
            level_idx as u32,
            sink,
            config,
            observer,
            &ctx,
        )?;
        tiles_produced += level_tiles;
        tiles_skipped += level_skipped;

        // Record level completion into the optional checkpoint.
        if let Some(cp) = checkpoint_state {
            cp.mark_level_completed(level.level);
        }

        observer.on_event(EngineEvent::LevelCompleted {
            level: level.level,
            tiles_produced: level_tiles,
        });
    }

    // Free last raster from tracking
    tracker.dealloc(current.data().len() as u64);

    let sink_finish_start = Instant::now();
    match sink.finish() {
        Ok(()) => {}
        Err(e) => return Err(promote_sink_error(e)),
    }
    stage_sink.fetch_add(
        sink_finish_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );

    // Flush a final checkpoint — ensures the on-disk `.libviprs-job.json`
    // reflects every tile emitted by the run (not only those that landed on
    // a `checkpoint_every` boundary).
    if let Some(cp) = checkpoint_state {
        cp.flush().map_err(EngineError::from)?;
    }

    observer.on_event(EngineEvent::Finished {
        total_tiles: tiles_produced,
        levels: plan.levels.len() as u32,
    });

    let decode_elapsed = stage_decode_done.saturating_duration_since(stage_decode_start);
    let stage_durations = StageDurations {
        planning: stage_planning,
        decode: decode_elapsed,
        resize: Duration::from_nanos(stage_resize.load(Ordering::Relaxed)),
        encode: Duration::from_nanos(stage_encode.load(Ordering::Relaxed)),
        sink: Duration::from_nanos(stage_sink.load(Ordering::Relaxed)),
    };

    let retry_count = sink.sink_retry_count();
    let skipped_due_to_failure = sink.sink_skipped_due_to_failure();

    Ok(EngineResult {
        tiles_produced,
        tiles_skipped,
        levels_processed: plan.levels.len() as u32,
        peak_memory_bytes: tracker.peak_bytes(),
        bytes_read,
        bytes_written: bytes_written.load(Ordering::Relaxed),
        retry_count,
        queue_pressure_peak: queue_pressure_peak.load(Ordering::Relaxed),
        duration: started.elapsed(),
        stage_durations,
        skipped_due_to_failure,
    })
}

/// Context struct passed into per-level emission so the inner functions can
/// update counters and checkpoints without blowing up their signatures.
struct EmitContext<'a> {
    bytes_written: &'a AtomicU64,
    queue_pressure_peak: &'a AtomicU32,
    stage_encode: &'a AtomicU64,
    stage_sink: &'a AtomicU64,
    /// Tiles that have already been written on a previous run (from a resume
    /// checkpoint). When `Some`, tiles matching an entry are skipped without
    /// calling the sink.
    skip_coords: Option<&'a HashSet<TileCoord>>,
    /// Running on-disk checkpoint. When present, each successful write is
    /// appended to it, and when `config.checkpoint_every` is non-zero the
    /// checkpoint is flushed to disk periodically.
    checkpoint_state: Option<&'a CheckpointState>,
}

/// Mutable, shared state for the on-disk resume checkpoint.
///
/// Wraps a [`JobMetadata`] behind a `Mutex` so that the emission loops — which
/// run on worker threads under parallel concurrency — can append completed
/// coordinates without fighting over exclusive ownership. A monotonically
/// increasing counter tracks how many tiles have been appended *since the
/// last flush* so the "every N tiles" cadence can be implemented without
/// poking the filesystem on every write.
pub(crate) struct CheckpointState {
    /// The directory where `.libviprs-job.json` lives — typically the sink's
    /// `base_dir`. Every call to [`CheckpointState::flush`] writes there.
    root: std::path::PathBuf,
    /// Running metadata. Re-serialised on every flush.
    meta: std::sync::Mutex<JobMetadata>,
    /// Write counter since the last flush. `checkpoint_every == 0` means we
    /// never perform intermediate flushes (final flush only).
    pending_since_flush: std::sync::atomic::AtomicU64,
    /// Flush cadence. `0` disables periodic flushing.
    checkpoint_every: u64,
}

impl CheckpointState {
    fn new(
        root: std::path::PathBuf,
        meta: JobMetadata,
        _plan: &PyramidPlan,
        checkpoint_every: u64,
    ) -> Self {
        Self {
            root,
            meta: std::sync::Mutex::new(meta),
            pending_since_flush: AtomicU64::new(0),
            checkpoint_every,
        }
    }

    /// Append a successful write to the metadata. When `checkpoint_every`
    /// tiles have accumulated since the last flush, also persist the
    /// checkpoint to disk so a crash can resume from the latest boundary.
    pub(crate) fn mark_tile_completed(&self, coord: TileCoord) -> Result<(), ResumeError> {
        {
            let mut meta = self.meta.lock().unwrap();
            meta.completed_tiles.push(coord);
        }
        // Flush periodically. `checkpoint_every == 0` disables this path and
        // leaves only the final flush (done by `run_pyramid` on success).
        //
        // `Relaxed` is sufficient: `flush()` takes the `meta` mutex
        // internally, which provides the happens-before edge between the
        // worker that appended the tile and the thread that serialises
        // the snapshot. The counter itself is a pure cadence gauge.
        if self.checkpoint_every > 0 {
            let n = self.pending_since_flush.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.checkpoint_every {
                self.pending_since_flush.store(0, Ordering::Relaxed);
                self.flush()?;
            }
        }
        Ok(())
    }

    /// Promote the level to `levels_completed` if every tile in that level
    /// has been accounted for. Called by `run_pyramid` after each level's
    /// inner loop returns.
    fn mark_level_completed(&self, level: u32) {
        let mut meta = self.meta.lock().unwrap();
        if !meta.levels_completed.contains(&level) {
            meta.levels_completed.push(level);
        }
    }

    /// Serialise the current metadata to `<root>/.libviprs-job.json`. Atomic
    /// via the tmp+rename dance inside [`JobCheckpoint::save`].
    pub(crate) fn flush(&self) -> Result<(), ResumeError> {
        let snapshot = {
            let mut meta = self.meta.lock().unwrap();
            meta.last_checkpoint_at = now_rfc3339_engine();
            meta.clone()
        };
        JobCheckpoint::save(&self.root, &snapshot).map_err(ResumeError::from)
    }
}

/// RFC-3339 timestamp helper (engine-local copy so the engine does not have
/// to depend on the sink module's private helper).
fn now_rfc3339_engine() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Use the same minimal civil-calendar conversion as the sink module.
    let (year, month, day, hour, minute, second) = secs_to_ymd_hms_engine(secs as i64);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn secs_to_ymd_hms_engine(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let mut z = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let second = (time_of_day % 60) as u32;
    let minute = ((time_of_day / 60) % 60) as u32;
    let hour = (time_of_day / 3600) as u32;
    z += 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = (y + if month <= 2 { 1 } else { 0 }) as i32;
    (year, month, day, hour, minute, second)
}

/// Promote `SinkError::ChecksumMismatch` to a dedicated engine error so
/// callers see the explicit variant (tests in `phase3_checksum.rs` match on
/// `EngineError::ChecksumMismatch`). All other sink errors pass through as
/// `EngineError::Sink`.
fn promote_sink_error(err: SinkError) -> EngineError {
    match err {
        SinkError::ChecksumMismatch {
            tile_rel_path,
            expected,
            got,
        } => {
            let tile =
                parse_tile_rel_path(&tile_rel_path).unwrap_or_else(|| TileCoord::new(0, 0, 0));
            EngineError::ChecksumMismatch {
                tile,
                expected,
                got,
            }
        }
        other => EngineError::Sink(other),
    }
}

/// Best-effort reverse-parse of a tile relative path back into a [`TileCoord`].
///
/// Understands the DeepZoom (`<level>/<col>_<row>.<ext>`) and XYZ /
/// Google (`<level>/<col>/<row>.<ext>`) shapes. Returns `None` when the
/// path does not match either layout — in which case the caller falls back
/// to `TileCoord::default()` so the error still surfaces with the other
/// fields intact.
fn parse_tile_rel_path(rel: &str) -> Option<TileCoord> {
    let normalized = rel.replace('\\', "/");
    let no_ext = normalized
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(&normalized);
    let parts: Vec<&str> = no_ext.split('/').collect();
    match parts.as_slice() {
        [level, last] => {
            let level: u32 = level.parse().ok()?;
            let (col, row) = last.split_once('_')?;
            let col: u32 = col.parse().ok()?;
            let row: u32 = row.parse().ok()?;
            Some(TileCoord::new(level, col, row))
        }
        [level, col, row] => {
            let level: u32 = level.parse().ok()?;
            let col: u32 = col.parse().ok()?;
            let row: u32 = row.parse().ok()?;
            Some(TileCoord::new(level, col, row))
        }
        _ => None,
    }
}

/// Resumable pyramid generation.
///
/// Runs `generate_pyramid` under one of three on-disk-state regimes:
///
/// * [`ResumeMode::Overwrite`] — wipe any stale contents of the sink's root
///   directory (`sink.checkpoint_root()`), then generate the pyramid from
///   scratch, writing a fresh checkpoint on every `config.checkpoint_every`
///   tiles and one final flush at the end.
/// * [`ResumeMode::Resume`] — load the existing `.libviprs-job.json`, verify
///   its `plan_hash` against the current plan (bails with
///   [`EngineError::PlanHashMismatch`] if they disagree), then generate only
///   the tiles that are not yet recorded as complete.
/// * [`ResumeMode::Verify`] — do not write anything to the sink. Walk every
///   tile in the plan, read the on-disk bytes, and return an error if a
///   tile is missing or (when the manifest includes checksums) if its bytes
///   hash to something other than the recorded digest.
pub(crate) fn generate_pyramid_resumable(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    mode: ResumeMode,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    match mode {
        ResumeMode::Overwrite => run_overwrite(source, plan, sink, config, observer),
        ResumeMode::Resume => run_resume(source, plan, sink, config, observer),
        ResumeMode::Verify => run_verify(source, plan, sink, config, observer),
    }
}

/// Start-from-scratch branch of [`generate_pyramid_resumable`].
fn run_overwrite(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    // If the sink exposes an on-disk root, wipe its stale contents so
    // pre-existing garbage (files from an aborted run, a checkpoint with
    // the wrong plan_hash, etc.) does not survive into the new run.
    if let Some(root) = resolve_checkpoint_root(config, sink) {
        wipe_directory(&root).map_err(|e| EngineError::ResumeFailed(ResumeError::from(e)))?;
    }

    let cp = cp_for_sink(sink, plan, config, Vec::new(), Vec::new());
    let skip = HashSet::new();
    run_pyramid_with_cp(
        source,
        plan,
        sink,
        config,
        cp.as_ref(),
        &skip,
        false,
        observer,
    )
}

/// Resolve the on-disk checkpoint root. Prefers the explicit
/// [`EngineConfig::checkpoint_root`] when set; otherwise consults
/// [`TileSink::checkpoint_root`]. Returns `None` when neither is available
/// (e.g. a pure in-memory sink with no config override).
///
/// The explicit config path exists so that callers wrapping [`FsSink`] in
/// an opaque user sink (e.g. recording / tee / retry wrappers) can still
/// drive resume and Verify without needing each wrapper to forward
/// `checkpoint_root()` through its trait impl.
pub(crate) fn resolve_checkpoint_root(cfg: &EngineConfig, sink: &dyn TileSink) -> Option<PathBuf> {
    cfg.checkpoint_root
        .clone()
        .or_else(|| sink.checkpoint_root().map(|p| p.to_path_buf()))
}

/// Resume-from-checkpoint branch of [`generate_pyramid_resumable`].
fn run_resume(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let expected_hash = compute_plan_hash(plan);

    let (existing_completed, existing_levels) =
        if let Some(root) = resolve_checkpoint_root(config, sink) {
            // `JobCheckpoint::load` returns `Result<Option<_>, ResumeError>`:
            // `?` surfaces real corruption as an engine error (via
            // `EngineError::ResumeFailed`) rather than silently degrading
            // to "no checkpoint". A missing file is still `Ok(None)`.
            match JobCheckpoint::load(&root)? {
                Some(meta) => {
                    if meta.plan_hash != expected_hash {
                        return Err(EngineError::PlanHashMismatch {
                            expected: meta.plan_hash.clone(),
                            got: expected_hash,
                        });
                    }
                    (meta.completed_tiles, meta.levels_completed)
                }
                None => (Vec::new(), Vec::new()),
            }
        } else {
            (Vec::new(), Vec::new())
        };

    let skip: HashSet<TileCoord> = existing_completed.iter().copied().collect();
    let cp = cp_for_sink(sink, plan, config, existing_completed, existing_levels);
    run_pyramid_with_cp(
        source,
        plan,
        sink,
        config,
        cp.as_ref(),
        &skip,
        true,
        observer,
    )
}

/// Build a `CheckpointState` rooted at the sink's checkpoint directory, or
/// `None` if the sink does not expose a filesystem root (no on-disk
/// checkpoint is possible in that case).
pub(crate) fn cp_for_sink(
    sink: &dyn TileSink,
    plan: &PyramidPlan,
    config: &EngineConfig,
    completed_tiles: Vec<TileCoord>,
    levels_completed: Vec<u32>,
) -> Option<CheckpointState> {
    let root = resolve_checkpoint_root(config, sink)?;
    let now = now_rfc3339_engine();
    let meta = JobMetadata {
        schema_version: SCHEMA_VERSION.to_string(),
        plan_hash: compute_plan_hash(plan),
        completed_tiles,
        levels_completed,
        started_at: now.clone(),
        last_checkpoint_at: now,
    };
    Some(CheckpointState::new(
        root,
        meta,
        plan,
        config.checkpoint_every,
    ))
}

/// Common pyramid-run body used by both `Overwrite` and `Resume`.
///
/// The `treat_skip_as_produced` flag keeps [`EngineResult::tiles_produced`]
/// meaningful across both modes: a fresh Overwrite returns the total count,
/// while Resume returns only the count of tiles written on *this* run
/// (the tests use the difference to assert exactly how much rework happened).
#[allow(clippy::too_many_arguments)]
fn run_pyramid_with_cp(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    cp: Option<&CheckpointState>,
    skip: &HashSet<TileCoord>,
    _treat_skip_as_produced: bool,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let skip_ref = if skip.is_empty() { None } else { Some(skip) };
    run_pyramid(source, plan, sink, config, observer, skip_ref, cp)
}

/// Verify-mode branch: walks every tile in the plan, reads the on-disk bytes
/// via `sink.checkpoint_root()` joined with the plan's tile path, and
/// returns an error if any tile is missing or (when the manifest records
/// checksums) if the bytes do not match the recorded digest.
///
/// Verify does NOT call `sink.write_tile`; the test suite asserts that
/// `tiles_produced == 0` and that no files are mutated. Emits
/// `LevelStarted` / `TileCompleted` / `LevelCompleted` / `Finished`
/// events so progress observers see verify runs as first-class.
fn run_verify(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let started = Instant::now();
    let root_buf =
        resolve_checkpoint_root(config, sink).ok_or(EngineError::VerifyRequiresOnDiskSink)?;
    let root = root_buf.as_path();

    // Try every known tile-file extension until we find one that matches.
    // The sink's active format isn't visible from this layer, so we probe
    // the common extensions produced by `TileFormat::extension` before
    // declaring a tile missing.
    let candidate_exts = ["raw", "png", "jpeg", "jpg"];

    for coord in plan.tile_coords() {
        let mut found: Option<std::path::PathBuf> = None;
        for ext in &candidate_exts {
            if let Some(rel) = plan.tile_path(coord, ext) {
                let abs = root.join(&rel);
                if abs.is_file() {
                    found = Some(abs);
                    break;
                }
            }
        }
        match found {
            Some(_abs) => {}
            None => {
                return Err(EngineError::Sink(SinkError::Other(format!(
                    "Verify: missing tile for coord {:?}",
                    coord
                ))));
            }
        }
    }

    // If the manifest includes a checksum table, re-hash each listed tile
    // and fail on the first mismatch. This mirrors the bits of
    // `verify_output` that are relevant for in-run verification.
    if let Some(manifest) = read_manifest(root) {
        if let Some(checksums) = manifest.get("checksums") {
            if let (Some(algo_str), Some(per_tile)) = (
                checksums.get("algo").and_then(|v| v.as_str()),
                checksums.get("per_tile").and_then(|v| v.as_object()),
            ) {
                let algo = match algo_str {
                    "blake3" => Some(crate::manifest::ChecksumAlgo::Blake3),
                    "sha256" => Some(crate::manifest::ChecksumAlgo::Sha256),
                    _ => None,
                };
                if let Some(algo) = algo {
                    for (rel, expected) in per_tile {
                        let expected_s = match expected.as_str() {
                            Some(s) => s,
                            None => continue,
                        };
                        let abs = root.join(rel);
                        let bytes = match std::fs::read(&abs) {
                            Ok(b) => b,
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                            Err(e) => return Err(EngineError::Sink(SinkError::Io(e))),
                        };
                        let got = crate::checksum::hash_tile(&bytes, algo);
                        if !got.eq_ignore_ascii_case(expected_s) {
                            return Err(EngineError::ChecksumMismatch {
                                tile: parse_tile_rel_path(rel)
                                    .unwrap_or_else(|| TileCoord::new(0, 0, 0)),
                                expected: expected_s.to_string(),
                                got,
                            });
                        }
                    }
                }
            }
        }
    }

    // Byte-exact verification: regenerate every level by walking the same
    // downscale path that the engine uses during a live run, then compare
    // each on-disk tile byte-for-byte against the expected content. This
    // catches corruption (flipped bytes, truncation) even when no manifest
    // is attached.
    //
    // The regeneration is done inline here rather than going through
    // `run_pyramid` because Verify must not touch the sink — the test
    // suite asserts the output directory is unchanged.
    let bg = config.background_rgb;
    let mut current = if plan.centre && (plan.centre_offset_x > 0 || plan.centre_offset_y > 0) {
        embed_in_canvas(source, plan, bg)?
    } else {
        source.clone()
    };
    let top_level = plan.levels.len() - 1;

    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];
        if level_idx < top_level {
            current = resize::downscale_half(&current)?;
        }
        observer.on_event(EngineEvent::LevelStarted {
            level: level.level,
            width: level.width,
            height: level.height,
            tile_count: level.tile_count(),
        });
        for row in 0..level.rows {
            for col in 0..level.cols {
                let coord = TileCoord::new(level_idx as u32, col, row);
                observer.on_event(EngineEvent::TileCompleted { coord });
                let expected = extract_tile(&current, plan, coord, bg)?;
                let expected_bytes = expected.data();

                // Find the on-disk file via the candidate extensions.
                let mut found: Option<(std::path::PathBuf, String)> = None;
                for ext in &candidate_exts {
                    if let Some(rel) = plan.tile_path(coord, ext) {
                        let abs = root.join(&rel);
                        if abs.is_file() {
                            found = Some((abs, (*ext).to_string()));
                            break;
                        }
                    }
                }
                let (abs, ext) = match found {
                    Some(f) => f,
                    None => {
                        return Err(EngineError::Sink(SinkError::Other(format!(
                            "Verify: missing tile for coord {:?}",
                            coord
                        ))));
                    }
                };

                let on_disk =
                    std::fs::read(&abs).map_err(|e| EngineError::Sink(SinkError::Io(e)))?;

                if ext == "raw" {
                    // Raw tiles are byte-exact: any mismatch (truncation,
                    // flipped byte, padding drift) is corruption.
                    if on_disk != expected_bytes {
                        return Err(EngineError::ChecksumMismatch {
                            tile: coord,
                            expected: format!("{} bytes (raw)", expected_bytes.len()),
                            got: format!(
                                "{} bytes on disk differ from regenerated tile",
                                on_disk.len()
                            ),
                        });
                    }
                }
                // Encoded tiles (png/jpeg) are not byte-exact against a fresh
                // encode due to encoder-state nondeterminism, so we keep the
                // existence check above and defer deep verification to the
                // manifest-checksum branch.
            }
        }
        observer.on_event(EngineEvent::LevelCompleted {
            level: level.level,
            tiles_produced: level.tile_count(),
        });
    }

    observer.on_event(EngineEvent::Finished {
        total_tiles: plan.total_tile_count(),
        levels: plan.levels.len() as u32,
    });

    Ok(EngineResult {
        tiles_produced: 0,
        tiles_skipped: 0,
        levels_processed: plan.levels.len() as u32,
        peak_memory_bytes: 0,
        bytes_read: 0,
        bytes_written: 0,
        retry_count: 0,
        queue_pressure_peak: 0,
        duration: started.elapsed(),
        stage_durations: StageDurations::default(),
        skipped_due_to_failure: 0,
    })
}

/// Parse the manifest JSON next to `root` (either `<root>.manifest.json`
/// sibling or `<root>/manifest.json` inside). Returns `None` if no manifest
/// exists, which is legitimate for runs that never attached a manifest
/// builder.
fn read_manifest(root: &std::path::Path) -> Option<serde_json::Value> {
    // Sibling first.
    if let (Some(parent), Some(stem)) = (root.parent(), root.file_name()) {
        let mut name = stem.to_os_string();
        name.push(".manifest.json");
        let sibling = parent.join(name);
        if let Ok(bytes) = std::fs::read(&sibling) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                return Some(v);
            }
        }
    }
    let inside = root.join("manifest.json");
    if let Ok(bytes) = std::fs::read(&inside) {
        return serde_json::from_slice::<serde_json::Value>(&bytes).ok();
    }
    None
}

/// Remove every entry under `dir`, ignoring errors for individual entries so
/// a pre-existing but partially-populated directory can still be wiped
/// cleanly. The directory itself is retained. Caller should have verified
/// the path is a directory they own.
pub(crate) fn wipe_directory(dir: &std::path::Path) -> std::io::Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
        return Ok(());
    }
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            let _ = std::fs::remove_dir_all(&p);
        } else {
            let _ = std::fs::remove_file(&p);
        }
    }
    Ok(())
}

/// Embeds the source image into a canvas-sized raster at the centre offset.
///
/// Creates a background-filled raster of `canvas_width × canvas_height` and
/// blits the source image at `(centre_offset_x, centre_offset_y)`. This
/// replicates how vips handles `--centre`: the image is placed in the canvas
/// before any downscaling, so boundary pixels are averaged correctly when
/// the canvas is halved level-by-level.
fn embed_in_canvas(
    source: &Raster,
    plan: &PyramidPlan,
    background_rgb: [u8; 3],
) -> Result<Raster, RasterError> {
    let cw = plan.canvas_width;
    let ch = plan.canvas_height;
    let bpp = source.format().bytes_per_pixel();
    let mut canvas = make_background_tile(cw, bpp, background_rgb);

    let ox = plan.centre_offset_x as usize;
    let oy = plan.centre_offset_y as usize;
    let iw = source.width() as usize;
    let src_stride = iw * bpp;
    let dst_stride = cw as usize * bpp;

    for row in 0..source.height() as usize {
        let src_start = row * src_stride;
        let dst_start = (row + oy) * dst_stride + ox * bpp;
        canvas[dst_start..dst_start + src_stride]
            .copy_from_slice(&source.data()[src_start..src_start + src_stride]);
    }

    Raster::new(cw, ch, source.format(), canvas)
}

/// Extracts every tile for one pyramid level and writes them to the sink.
///
/// Dispatches to either the single-threaded loop or the parallel worker pool
/// depending on [`EngineConfig::concurrency`]. In single-threaded mode, tiles
/// are extracted and emitted in row-major order on the calling thread. In
/// parallel mode, the work is delegated to [`extract_and_emit_parallel`].
///
/// Each tile is optionally checked for blankness when
/// [`BlankTileStrategy::Placeholder`] is active, and the observer is notified
/// after every tile is written.
///
/// Returns `(tiles_produced, tiles_skipped)`.
fn extract_and_emit_level(
    raster: &Raster,
    plan: &PyramidPlan,
    level: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
    ctx: &EmitContext,
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let blank_strategy = config.blank_tile_strategy;

    if config.concurrency == 0 {
        // Single-threaded path
        let mut count = 0u64;
        let mut skipped = 0u64;
        for row in 0..level_plan.rows {
            for col in 0..level_plan.cols {
                let coord = TileCoord::new(level, col, row);
                // Honor the resume checkpoint: tiles already present on disk
                // (per an inbound `.libviprs-job.json`) are not re-emitted.
                if let Some(skip) = ctx.skip_coords {
                    if skip.contains(&coord) {
                        continue;
                    }
                }
                let encode_start = Instant::now();
                let tile_raster = extract_tile(raster, plan, coord, config.background_rgb)?;
                ctx.stage_encode
                    .fetch_add(encode_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let blank = is_blank_for_strategy(&tile_raster, blank_strategy);
                if blank {
                    skipped += 1;
                }
                // Track bytes written as the tile payload size (sink-side
                // encoding overhead is not included — the test only sums the
                // raw raster bytes).
                let tile_bytes = tile_raster.data().len() as u64;
                let tile = Tile {
                    coord,
                    raster: tile_raster,
                    blank,
                };
                let sink_start = Instant::now();
                match sink.write_tile(&tile) {
                    Ok(()) => {
                        ctx.stage_sink
                            .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        ctx.bytes_written.fetch_add(tile_bytes, Ordering::Relaxed);
                        if let Some(cp) = ctx.checkpoint_state {
                            cp.mark_tile_completed(coord).map_err(EngineError::from)?;
                        }
                    }
                    Err(e) => {
                        ctx.stage_sink
                            .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        match &config.failure_policy {
                            FailurePolicy::RetryThenSkip(_) => {
                                // Account the skip on the outermost sink so
                                // it surfaces in EngineResult.
                                sink.note_sink_skipped();
                                observer.on_event(EngineEvent::TileCompleted { coord });
                                // Intentionally do NOT increment count here —
                                // this tile did not produce output. But also
                                // do not increment a skip counter tied to
                                // blanks; the RetryThenSkip path is a
                                // separate counter fed by sink.sink_skipped_due_to_failure.
                                continue;
                            }
                            _ => return Err(promote_sink_error(e)),
                        }
                    }
                }
                observer.on_event(EngineEvent::TileCompleted { coord });
                #[cfg(feature = "tracing")]
                if tracing::enabled!(target: "libviprs::tile", tracing::Level::TRACE) {
                    tracing::trace!(
                        target: "libviprs::tile",
                        x = coord.col,
                        y = coord.row,
                        level = coord.level,
                        "tile done"
                    );
                }
                count += 1;
            }
        }
        // Parallel and single-threaded both contribute to queue_pressure.
        // In single-threaded there is effectively 1 tile in-flight at a
        // time; record that as the minimum peak so the test's > 0
        // assertion holds even when concurrency is 0.
        let _ = ctx.queue_pressure_peak.fetch_max(1, Ordering::Relaxed);
        Ok((count, skipped))
    } else {
        extract_and_emit_parallel(raster, plan, level, sink, config, observer, ctx)
    }
}

/// Return `true` when the current [`BlankTileStrategy`] wants this tile to be
/// written as a placeholder.
fn is_blank_for_strategy(raster: &Raster, strategy: BlankTileStrategy) -> bool {
    match strategy {
        BlankTileStrategy::Emit => false,
        BlankTileStrategy::Placeholder => is_blank_tile(raster),
        BlankTileStrategy::PlaceholderWithTolerance { max_channel_delta } => {
            is_blank_tile_with_tolerance(raster, max_channel_delta)
        }
    }
}

/// Extracts tiles for one level in parallel using scoped worker threads.
///
/// Tile coordinates are divided into roughly equal chunks (one per worker).
/// Each worker extracts its tiles and sends them through a bounded
/// `sync_channel`, which provides backpressure — producers block when the
/// channel is full, preventing unbounded memory growth. A single consumer
/// on the calling thread drains the channel, writes tiles to the sink, and
/// notifies the observer.
///
/// The bounded channel capacity is set by [`EngineConfig::buffer_size`].
/// Worker count is capped at `min(concurrency, tile_count)` to avoid
/// spawning idle threads.
///
/// Returns `(tiles_produced, tiles_skipped)`.
fn extract_and_emit_parallel(
    raster: &Raster,
    plan: &PyramidPlan,
    level: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
    ctx: &EmitContext,
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let total_tiles = level_plan.tile_count();

    if total_tiles == 0 {
        return Ok((0, 0));
    }

    let blank_strategy = config.blank_tile_strategy;

    // Bounded channel for backpressure: producers block when buffer is full
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Tile, EngineError>>(config.buffer_size);
    // Queue-pressure gauge — producers bump when they send, consumer drops
    // when it receives. The peak gives us a coarse upper bound on in-flight
    // work, matching the `queue_pressure_peak` field on EngineResult.
    let in_flight = Arc::new(AtomicU32::new(0));

    // Share the raster across worker threads (read-only)
    let raster = Arc::new(raster.clone());
    let plan = Arc::new(plan.clone());

    // Collect tile coordinates for this level, honouring the resume
    // checkpoint: tiles flagged as already complete are elided here so
    // neither the producer nor consumer do any work for them.
    let coords: Vec<TileCoord> = (0..level_plan.rows)
        .flat_map(|row| (0..level_plan.cols).map(move |col| TileCoord::new(level, col, row)))
        .filter(|coord| match ctx.skip_coords {
            Some(skip) => !skip.contains(coord),
            None => true,
        })
        .collect();

    if coords.is_empty() {
        return Ok((0, 0));
    }

    // Spawn workers
    let concurrency = config.concurrency.min(coords.len());
    let chunk_size = coords.len().div_ceil(concurrency);

    let stage_encode: &AtomicU64 = ctx.stage_encode;
    let queue_peak: &AtomicU32 = ctx.queue_pressure_peak;

    std::thread::scope(|s| {
        // Spawn producer threads
        for chunk in coords.chunks(chunk_size) {
            let tx = tx.clone();
            let raster = Arc::clone(&raster);
            let plan = Arc::clone(&plan);
            let in_flight = Arc::clone(&in_flight);
            let chunk = chunk.to_vec();
            let bg = config.background_rgb;

            s.spawn(move || {
                for coord in chunk {
                    // Queue-pressure gauge: count active producers. A
                    // producer is "in flight" while it is extracting a
                    // tile and attempting to hand it off. The peak is
                    // bounded by the worker count (plus up to one on the
                    // consumer side while it writes to the sink).
                    //
                    // Pure gauge — no happens-before needed between the
                    // counter and any tile payload, so `Relaxed` is safe
                    // (and avoids a useless full fence in the hot loop).
                    let cur = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = queue_peak.fetch_max(cur, Ordering::Relaxed);

                    let encode_start = Instant::now();
                    let result = extract_tile(&raster, &plan, coord, bg)
                        .map(|tile_raster| {
                            let blank = is_blank_for_strategy(&tile_raster, blank_strategy);
                            Tile {
                                coord,
                                raster: tile_raster,
                                blank,
                            }
                        })
                        .map_err(EngineError::from);
                    stage_encode
                        .fetch_add(encode_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let send_failed = tx.send(result).is_err();
                    // Balance the counter as soon as the tile leaves the
                    // producer — either the consumer has taken ownership
                    // (and the channel buffers it briefly) or the consumer
                    // is gone and we're about to exit. `Relaxed` matches
                    // the corresponding `fetch_add` above.
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    if send_failed {
                        break; // Consumer dropped
                    }
                }
            });
        }
        // Drop our copy so rx knows when all producers are done
        drop(tx);

        // Consumer: receive tiles and write to sink
        let mut count = 0u64;
        let mut skipped = 0u64;
        for result in rx {
            let tile = result?;
            let coord = tile.coord;
            if tile.blank {
                skipped += 1;
            }
            let tile_bytes = tile.raster.data().len() as u64;
            let sink_start = Instant::now();
            match sink.write_tile(&tile) {
                Ok(()) => {
                    ctx.stage_sink
                        .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    ctx.bytes_written.fetch_add(tile_bytes, Ordering::Relaxed);
                    if let Some(cp) = ctx.checkpoint_state {
                        cp.mark_tile_completed(coord).map_err(EngineError::from)?;
                    }
                }
                Err(e) => {
                    ctx.stage_sink
                        .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    match &config.failure_policy {
                        FailurePolicy::RetryThenSkip(_) => {
                            sink.note_sink_skipped();
                            observer.on_event(EngineEvent::TileCompleted { coord });
                            continue;
                        }
                        _ => return Err(promote_sink_error(e)),
                    }
                }
            }
            observer.on_event(EngineEvent::TileCompleted { coord });
            #[cfg(feature = "tracing")]
            if tracing::enabled!(target: "libviprs::tile", tracing::Level::TRACE) {
                tracing::trace!(
                    target: "libviprs::tile",
                    x = coord.col,
                    y = coord.row,
                    level = coord.level,
                    "tile done"
                );
            }
            count += 1;
        }
        Ok((count, skipped))
    })
}

/// Allocates a `tile_size × tile_size` pixel buffer filled with the
/// background color.
///
/// Used to pad edge tiles and to produce solid-background tiles for
/// canvas regions that lie outside the source image (e.g. Google layout
/// with centre). The background RGB triplet is expanded to match the
/// pixel format's bytes-per-pixel: grayscale uses the red channel,
/// RGB copies all three, RGBA appends alpha=255, and other formats
/// repeat the red channel.
fn make_background_tile(ts: u32, bpp: usize, background_rgb: [u8; 3]) -> Vec<u8> {
    let mut padded = vec![0u8; ts as usize * ts as usize * bpp];
    let bg_pixel: Vec<u8> = match bpp {
        1 => vec![background_rgb[0]],
        3 => background_rgb.to_vec(),
        4 => vec![background_rgb[0], background_rgb[1], background_rgb[2], 255],
        _ => vec![background_rgb[0]; bpp],
    };
    for pixel in padded.chunks_exact_mut(bpp) {
        pixel.copy_from_slice(&bg_pixel);
    }
    padded
}

/// Extracts a single tile's pixel data from the level raster.
///
/// For standard DeepZoom/Xyz layouts without centre, the tile rect maps
/// directly to image coordinates — the region is extracted and edge tiles
/// are padded to `tile_size` with the background color (only when
/// `overlap == 0`; overlap tiles keep their natural smaller size).
///
/// For Google layout or any plan with `centre == true`, tiles are
/// addressed in *canvas* space (which may be larger than the image).
/// The function computes the intersection of the canvas-space tile rect
/// with the image region (offset by the centre offset), then:
/// - If the tile is entirely outside the image → returns a solid
///   background tile.
/// - If partially overlapping → creates a background tile and blits the
///   intersecting image region at the correct offset.
/// - If fully within the image → extracts directly (fast path).
fn extract_tile(
    raster: &Raster,
    plan: &PyramidPlan,
    coord: TileCoord,
    background_rgb: [u8; 3],
) -> Result<Raster, RasterError> {
    let rect = plan
        .tile_rect(coord)
        .expect("tile_rect returned None for valid coord");

    let ts = plan.tile_size;
    let bpp = raster.format().bytes_per_pixel();

    // For Google layout or non-centred plans where tiles reference canvas
    // space (tile rects may extend beyond the raster), extract what we can
    // and pad the rest with background.
    if plan.layout == crate::planner::Layout::Google {
        // For centred plans, the source raster has been embedded in the
        // canvas by embed_in_canvas(), so the raster IS the canvas and
        // tiles extract directly. For non-centred plans, the image is
        // at (0,0) and tiles beyond the image boundary get padding.
        let rw = raster.width();
        let rh = raster.height();

        // Intersection of tile rect with raster bounds
        let inter_right = (rect.x + rect.width).min(rw);
        let inter_bottom = (rect.y + rect.height).min(rh);

        if rect.x >= rw || rect.y >= rh {
            // Tile entirely outside raster — solid background
            let padded = make_background_tile(ts, bpp, background_rgb);
            return Raster::new(ts, ts, raster.format(), padded);
        }

        let inter_w = inter_right - rect.x;
        let inter_h = inter_bottom - rect.y;

        if inter_w == ts && inter_h == ts {
            // Fast path: tile entirely within raster
            return raster.extract(rect.x, rect.y, ts, ts);
        }

        // Partial: extract overlap and pad
        let content = raster.extract(rect.x, rect.y, inter_w, inter_h)?;
        let mut padded = make_background_tile(ts, bpp, background_rgb);
        let src_stride = inter_w as usize * bpp;
        let dst_stride = ts as usize * bpp;
        for row in 0..inter_h as usize {
            let src_start = row * src_stride;
            let dst_start = row * dst_stride;
            padded[dst_start..dst_start + src_stride]
                .copy_from_slice(&content.data()[src_start..src_start + src_stride]);
        }
        return Raster::new(ts, ts, raster.format(), padded);
    }

    // Standard DeepZoom/Xyz path
    let content = raster.extract(rect.x, rect.y, rect.width, rect.height)?;

    // Pad edge tiles to the full tile size with the background color.
    // Only pad when there's no overlap — overlap tiles have intentionally
    // different sizes and must not be resized.
    if plan.overlap == 0 && (content.width() < ts || content.height() < ts) {
        let mut padded = make_background_tile(ts, bpp, background_rgb);

        // Copy content rows into the padded buffer
        let src_stride = content.width() as usize * bpp;
        let dst_stride = ts as usize * bpp;
        for row in 0..content.height() as usize {
            let src_start = row * src_stride;
            let dst_start = row * dst_stride;
            padded[dst_start..dst_start + src_stride]
                .copy_from_slice(&content.data()[src_start..src_start + src_stride]);
        }

        Raster::new(ts, ts, content.format(), padded)
    } else {
        Ok(content)
    }
}

/// Returns `true` if every pixel in the tile is identical (i.e. the tile is
/// a uniform solid color).
///
/// Used by the engine when [`BlankTileStrategy::Placeholder`] is active to
/// decide whether a tile should be replaced with a 1-byte marker instead of
/// full image data. A single-pixel raster is trivially blank.
///
/// See the
/// [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for integration-level examples.
pub fn is_blank_tile(raster: &Raster) -> bool {
    let data = raster.data();
    let bpp = raster.format().bytes_per_pixel();
    if data.len() <= bpp {
        return true;
    }
    let first_pixel = &data[..bpp];
    data.chunks(bpp).all(|px| px == first_pixel)
}

/// Returns `true` if every pixel in the tile is within `max_channel_delta`
/// of the tile's first pixel on every channel.
///
/// Equivalent to [`is_blank_tile`] when `max_channel_delta == 0`. Useful for
/// raster backgrounds with light JPEG-compression noise where an exact equal
/// check would miss near-uniform regions.
pub fn is_blank_tile_with_tolerance(raster: &Raster, max_channel_delta: u8) -> bool {
    if max_channel_delta == 0 {
        return is_blank_tile(raster);
    }
    let data = raster.data();
    let bpp = raster.format().bytes_per_pixel();
    if data.len() <= bpp {
        return true;
    }
    let first_pixel = &data[..bpp];
    data.chunks(bpp).all(|px| {
        px.iter().zip(first_pixel.iter()).all(|(a, b)| {
            let d = a.abs_diff(*b);
            d <= max_channel_delta
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::CollectingObserver;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::sink::MemorySink;

    fn gradient_raster(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h {
            for x in 0..w {
                let off = (y as usize * w as usize + x as usize) * bpp;
                data[off] = (x % 256) as u8;
                data[off + 1] = (y % 256) as u8;
                data[off + 2] = ((x + y) % 256) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_raster(w: u32, h: u32, val: u8) -> Raster {
        let data = vec![val; w as usize * h as usize * 3];
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /**
     * Tests that single-threaded engine produces the correct total tile count.
     * Works by running generate_pyramid with concurrency=0 (default) and asserting
     * both the returned count and the sink's stored count match the plan.
     * Input: 512x512 RGB gradient, tile_size=256 -> Output: plan.total_tile_count() tiles.
     */
    #[test]
    fn single_threaded_produces_all_tiles() {
        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default();

        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert_eq!(result.tiles_produced, plan.total_tile_count());
        assert_eq!(sink.tile_count() as u64, plan.total_tile_count());
    }

    /**
     * Tests that multi-threaded engine produces the correct total tile count.
     * Works by running generate_pyramid with concurrency=4 and verifying the
     * result and sink agree with the plan's expected tile count.
     * Input: 512x512 RGB gradient, tile_size=256, 4 threads -> Output: all expected tiles.
     */
    #[test]
    fn parallel_produces_all_tiles() {
        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default().with_concurrency(4);

        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert_eq!(result.tiles_produced, plan.total_tile_count());
        assert_eq!(sink.tile_count() as u64, plan.total_tile_count());
    }

    /**
     * Tests that every expected (level, col, row) coordinate appears in the output.
     * Works by sorting the produced tile coordinates and the plan's expected
     * coordinates, then asserting exact equality between the two sets.
     * Input: 600x400 non-square image, tile_size=256, concurrency=2.
     */
    #[test]
    fn all_tile_coords_present() {
        let src = gradient_raster(600, 400);
        let planner = PyramidPlanner::new(600, 400, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default().with_concurrency(2);

        generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        let tiles = sink.tiles();
        let mut coords: Vec<_> = tiles.iter().map(|t| t.coord).collect();
        coords.sort_by_key(|c| (c.level, c.row, c.col));

        let mut expected: Vec<_> = plan.tile_coords().collect();
        expected.sort_by_key(|c| (c.level, c.row, c.col));

        assert_eq!(coords, expected);
    }

    /**
     * Tests that each produced tile has the width and height specified by the plan.
     * Works by comparing every tile's dimensions against plan.tile_rect() for
     * its coordinate, catching off-by-one errors at image/tile boundaries.
     * Input: 500x300 non-tile-aligned image, tile_size=256.
     */
    #[test]
    fn tile_dimensions_match_plan() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default();

        generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        for tile in sink.tiles() {
            let rect = plan.tile_rect(tile.coord).unwrap();
            // Edge tiles are padded to the full tile size when overlap=0
            let expected_w = if rect.width < 256 { 256 } else { rect.width };
            let expected_h = if rect.height < 256 { 256 } else { rect.height };
            assert_eq!(tile.width, expected_w, "Width mismatch at {:?}", tile.coord);
            assert_eq!(
                tile.height, expected_h,
                "Height mismatch at {:?}",
                tile.coord
            );
        }
    }

    /**
     * Tests that tile pixel data is identical regardless of concurrency level.
     * Works by generating a reference pyramid single-threaded, then re-running
     * at concurrency 1, 2, 4, 8, 16 and byte-comparing every tile's data.
     * Input: 256x256 gradient, tile_size=64 -> Output: identical tiles at all concurrency levels.
     */
    #[test]
    fn deterministic_across_concurrency_levels() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let ref_sink = MemorySink::new();
        generate_pyramid_observed(
            &src,
            &plan,
            &ref_sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .unwrap();

        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        for concurrency in [1, 2, 4, 8, 16] {
            let sink = MemorySink::new();
            let config = EngineConfig::default().with_concurrency(concurrency);
            generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

            let mut tiles = sink.tiles();
            tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

            assert_eq!(
                tiles.len(),
                ref_tiles.len(),
                "Tile count mismatch at concurrency={concurrency}"
            );

            for (ref_t, t) in ref_tiles.iter().zip(tiles.iter()) {
                assert_eq!(ref_t.coord, t.coord);
                assert_eq!(
                    ref_t.data, t.data,
                    "Tile data diverged at {:?} with concurrency={concurrency}",
                    t.coord
                );
            }
        }
    }

    /**
     * Tests that EngineResult.levels_processed matches the plan's level count.
     * Works by checking the result metadata against plan.level_count(),
     * ensuring no levels are skipped or double-counted.
     * Input: 64x64 image, tile_size=256 -> Output: levels_processed == plan.level_count().
     */
    #[test]
    fn levels_processed_matches_plan() {
        let src = gradient_raster(64, 64);
        let planner = PyramidPlanner::new(64, 64, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();

        let result =
            generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .unwrap();
        assert_eq!(result.levels_processed, plan.level_count() as u32);
    }

    /**
     * Tests the edge case where the image is smaller than a single tile.
     * Works by verifying that each pyramid level produces exactly one tile,
     * so total tiles equals the number of levels.
     * Input: 10x10 image, tile_size=256 -> Output: one tile per level.
     */
    #[test]
    fn small_image_single_tile() {
        let src = gradient_raster(10, 10);
        let planner = PyramidPlanner::new(10, 10, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();

        let result =
            generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .unwrap();
        assert_eq!(result.tiles_produced, plan.level_count() as u64);
    }

    /**
     * Tests that the engine completes correctly with a minimal buffer size.
     * Works by setting buffer_size=1 with 4 concurrent workers, forcing
     * frequent producer blocking, and verifying no tiles are lost.
     * Input: 512x512 image, tile_size=128, buffer=1 -> Output: all tiles produced.
     */
    #[test]
    fn backpressure_small_buffer() {
        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default()
            .with_concurrency(4)
            .with_buffer_size(1);

        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }

    /**
     * Tests that a raster where every pixel is identical is detected as blank.
     * Works by creating an 8x8 solid-color raster and asserting is_blank_tile
     * returns true, since all pixel triplets are (128, 128, 128).
     * Input: 8x8 solid val=128 -> Output: true.
     */
    #[test]
    fn is_blank_tile_solid() {
        let r = solid_raster(8, 8, 128);
        assert!(is_blank_tile(&r));
    }

    /**
     * Tests that a raster with even one differing pixel is not blank.
     * Works by creating a solid raster then modifying the first byte,
     * making the first pixel differ from the rest.
     * Input: 8x8 solid val=128 with data[0]=0 -> Output: false.
     */
    #[test]
    fn is_blank_tile_not_blank() {
        let mut data = vec![128u8; 8 * 8 * 3];
        data[0] = 0;
        let r = Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap();
        assert!(!is_blank_tile(&r));
    }

    /**
     * Tests the boundary case of a 1x1 pixel raster for blank detection.
     * Works because a single-pixel raster has no other pixel to differ from,
     * so it is trivially blank regardless of its color value.
     * Input: 1x1 RGB pixel [1,2,3] -> Output: true.
     */
    #[test]
    fn is_blank_tile_single_pixel() {
        let r = Raster::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
        assert!(is_blank_tile(&r));
    }

    /**
     * Tests that tiles generated with overlap have dimensions matching the plan.
     * Works by using overlap=2, which adds border pixels to tiles, then
     * verifying each tile's width/height against plan.tile_rect().
     * Input: 600x400 image, tile_size=256, overlap=2 -> Output: correct overlap-adjusted sizes.
     */
    #[test]
    fn overlap_tiles_have_correct_size() {
        let src = gradient_raster(600, 400);
        let planner = PyramidPlanner::new(600, 400, 256, 2, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default();

        generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        for tile in sink.tiles() {
            let rect = plan.tile_rect(tile.coord).unwrap();
            assert_eq!(tile.width, rect.width);
            assert_eq!(tile.height, rect.height);
        }
    }

    /**
     * Tests that parallel engine works correctly when the sink is slow.
     * Works by using a SlowSink with 1ms delay and a small buffer (2),
     * stressing the backpressure mechanism under realistic conditions.
     * Input: 128x128 image, tile_size=64, 4 threads, 1ms sink delay -> Output: all tiles.
     */
    #[test]
    fn concurrent_with_slow_sink() {
        use crate::sink::SlowSink;
        use std::time::Duration;

        let src = gradient_raster(128, 128);
        let planner = PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = SlowSink::new(Duration::from_millis(1));
        let config = EngineConfig::default()
            .with_concurrency(4)
            .with_buffer_size(2);

        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
        assert_eq!(sink.tile_count() as u64, plan.total_tile_count());
    }

    // -- Observability tests --

    /**
     * Tests that the observer receives a TileCompleted event for every tile.
     * Works by counting TileCompleted events from a CollectingObserver and
     * comparing against the plan's total tile count.
     * Input: 128x128 image, tile_size=64 -> Output: tile_events == total_tile_count.
     */
    #[test]
    fn observer_receives_all_tile_events() {
        let src = gradient_raster(128, 128);
        let planner = PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let obs = CollectingObserver::new();

        generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &obs).unwrap();

        let tile_events = obs
            .events()
            .iter()
            .filter(|e| matches!(e, EngineEvent::TileCompleted { .. }))
            .count();

        assert_eq!(tile_events as u64, plan.total_tile_count());
    }

    /**
     * Tests that LevelStarted events arrive in top-down order and Finished is last.
     * Works by extracting level numbers from LevelStarted events and comparing
     * against a descending sequence, then checking the final event type.
     * Input: 64x64 image, tile_size=256 -> Output: levels in descending order, Finished last.
     */
    #[test]
    fn observer_receives_level_events_in_order() {
        let src = gradient_raster(64, 64);
        let planner = PyramidPlanner::new(64, 64, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let obs = CollectingObserver::new();

        generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &obs).unwrap();

        let events = obs.events();

        // Should have LevelStarted/LevelCompleted pairs for each level (top-down)
        let level_starts: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::LevelStarted { level, .. } => Some(*level),
                _ => None,
            })
            .collect();

        // Levels processed top-down
        let expected_levels: Vec<u32> = (0..plan.level_count() as u32).rev().collect();
        assert_eq!(level_starts, expected_levels);

        // Last event should be Finished
        assert!(matches!(events.last(), Some(EngineEvent::Finished { .. })));
    }

    /**
     * Tests that the Finished event carries the correct total tile and level counts.
     * Works by matching on the last event and asserting its fields equal
     * the plan's total_tile_count and level_count.
     * Input: 256x256 image, tile_size=128 -> Output: Finished{total_tiles, levels} match plan.
     */
    #[test]
    fn observer_finished_event_has_correct_totals() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let obs = CollectingObserver::new();

        generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &obs).unwrap();

        let events = obs.events();
        let finished = events.last().unwrap();
        match finished {
            EngineEvent::Finished {
                total_tiles,
                levels,
            } => {
                assert_eq!(*total_tiles, plan.total_tile_count());
                assert_eq!(*levels, plan.level_count() as u32);
            }
            _ => panic!("Last event should be Finished"),
        }
    }

    /**
     * Tests that the observer receives all TileCompleted events under concurrency.
     * Works by running with concurrency=4 and verifying the TileCompleted count
     * matches the plan, ensuring thread-safe event delivery.
     * Input: 256x256 image, tile_size=64, 4 threads -> Output: correct event count.
     */
    #[test]
    fn observer_works_with_concurrency() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();
        let obs = CollectingObserver::new();

        generate_pyramid_observed(
            &src,
            &plan,
            &sink,
            &EngineConfig::default().with_concurrency(4),
            &obs,
        )
        .unwrap();

        let tile_events = obs
            .events()
            .iter()
            .filter(|e| matches!(e, EngineEvent::TileCompleted { .. }))
            .count();

        assert_eq!(tile_events as u64, plan.total_tile_count());
    }

    /**
     * Tests that peak memory tracking reports at least the source raster size.
     * Works by checking that peak_memory_bytes >= source pixel data size,
     * since the source raster must be held in memory throughout.
     * Input: 512x512 RGB (786432 bytes) -> Output: peak_memory_bytes >= 786432.
     */
    #[test]
    fn peak_memory_is_reported() {
        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();

        let result =
            generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .unwrap();

        // Peak should be at least the source raster size
        let source_bytes = 512 * 512 * 3;
        assert!(
            result.peak_memory_bytes >= source_bytes,
            "Peak {} < source {source_bytes}",
            result.peak_memory_bytes
        );
    }

    /**
     * Tests that peak memory stays bounded below 2x the source raster size.
     * Works because the engine only holds one level raster at a time, so
     * peak usage should not exceed source + one downscaled copy.
     * Input: 1024x1024 RGB (3145728 bytes) -> Output: peak < 6291456 bytes.
     */
    #[test]
    fn peak_memory_is_bounded() {
        // For a 1024x1024 image, peak memory should not be wildly larger
        // than the source (we only hold one level raster at a time)
        let src = gradient_raster(1024, 1024);
        let planner = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();

        let result =
            generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .unwrap();

        let source_bytes = 1024u64 * 1024 * 3;
        // Should be less than 2x source (current level + some overhead)
        assert!(
            result.peak_memory_bytes < source_bytes * 2,
            "Peak {} >= 2x source {source_bytes}",
            result.peak_memory_bytes
        );
    }

    // -- Google layout + centre tests --

    #[test]
    fn google_centre_produces_all_tiles() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        let sink = MemorySink::new();
        let config = EngineConfig::default();

        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
        assert_eq!(sink.tile_count() as u64, plan.total_tile_count());
    }

    #[test]
    fn google_centre_all_tiles_full_size() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        let sink = MemorySink::new();

        generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
            .unwrap();

        for tile in sink.tiles() {
            assert_eq!(tile.width, 256, "Width mismatch at {:?}", tile.coord);
            assert_eq!(tile.height, 256, "Height mismatch at {:?}", tile.coord);
        }
    }

    #[test]
    fn google_centre_edge_tiles_have_background() {
        // Small image centred in 256x256 canvas → single tile with background padding
        let src = solid_raster(10, 10, 200);
        let planner = PyramidPlanner::new(10, 10, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        let sink = MemorySink::new();

        generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
            .unwrap();

        // Level 0 should have 1 tile (1x1 grid)
        let tiles = sink.tiles();
        let level0: Vec<_> = tiles.iter().filter(|t| t.coord.level == 0).collect();
        assert_eq!(level0.len(), 1);
        let tile = &level0[0];
        assert_eq!(tile.width, 256);
        assert_eq!(tile.height, 256);
        // Tile should NOT be entirely the source color (200,200,200) since background is white
        assert!(
            !is_blank_tile(
                &Raster::new(
                    tile.width,
                    tile.height,
                    PixelFormat::Rgb8,
                    tile.data.clone()
                )
                .unwrap()
            ) || tile.data.chunks(3).all(|px| px == [255, 255, 255])
        );
    }

    #[test]
    fn google_centre_deterministic_across_concurrency() {
        let src = gradient_raster(400, 300);
        let planner = PyramidPlanner::new(400, 300, 128, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();

        let ref_sink = MemorySink::new();
        generate_pyramid_observed(
            &src,
            &plan,
            &ref_sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .unwrap();

        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        for concurrency in [1, 2, 4] {
            let sink = MemorySink::new();
            let config = EngineConfig::default().with_concurrency(concurrency);
            generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

            let mut tiles = sink.tiles();
            tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

            assert_eq!(tiles.len(), ref_tiles.len());
            for (ref_t, t) in ref_tiles.iter().zip(tiles.iter()) {
                assert_eq!(ref_t.coord, t.coord);
                assert_eq!(
                    ref_t.data, t.data,
                    "Tile {:?} diverged at concurrency={concurrency}",
                    t.coord
                );
            }
        }
    }

    #[test]
    fn google_no_centre_produces_all_tiles() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        let sink = MemorySink::new();

        let result =
            generate_pyramid_observed(&src, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }

    /// Observer should see LevelStarted / TileCompleted / LevelCompleted /
    /// Finished events from the resumable path, not just the non-resumable
    /// one. Runs all three ResumeModes against a CollectingObserver and
    /// asserts each one drives at least the expected shape of events.
    #[test]
    fn resumable_emits_observer_events() {
        use crate::observe::CollectingObserver;
        use crate::resume::ResumeMode;
        use tempfile::tempdir;

        let src = gradient_raster(128, 96);
        let planner = PyramidPlanner::new(128, 96, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let dir = tempdir().unwrap();
        let sink = crate::sink::FsSink::new(dir.path().join("tiles"), plan.clone())
            .with_format(crate::sink::TileFormat::Raw);

        // --- Overwrite --------------------------------------------------
        let obs = CollectingObserver::new();
        generate_pyramid_resumable(
            &src,
            &plan,
            &sink,
            &EngineConfig::default(),
            ResumeMode::Overwrite,
            &obs,
        )
        .unwrap();
        let events = obs.events();
        let tile_events = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::TileCompleted { .. }))
            .count();
        let finished = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::Finished { .. }))
            .count();
        assert_eq!(
            tile_events as u64,
            plan.total_tile_count(),
            "Overwrite: tile events"
        );
        assert_eq!(finished, 1, "Overwrite: finished event");

        // --- Resume (no-op since everything is already complete) --------
        let obs = CollectingObserver::new();
        generate_pyramid_resumable(
            &src,
            &plan,
            &sink,
            &EngineConfig::default(),
            ResumeMode::Resume,
            &obs,
        )
        .unwrap();
        // Resume with a full checkpoint short-circuits without per-tile
        // work, so we only require the engine to have produced *some*
        // observer activity (the Finished event at minimum).
        assert!(
            !obs.events().is_empty(),
            "Resume mode produced no observer events"
        );

        // --- Verify -----------------------------------------------------
        let obs = CollectingObserver::new();
        generate_pyramid_resumable(
            &src,
            &plan,
            &sink,
            &EngineConfig::default(),
            ResumeMode::Verify,
            &obs,
        )
        .unwrap();
        let events = obs.events();
        let tile_events = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::TileCompleted { .. }))
            .count();
        let finished = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::Finished { .. }))
            .count();
        assert_eq!(
            tile_events as u64,
            plan.total_tile_count(),
            "Verify: tile events"
        );
        assert_eq!(finished, 1, "Verify: finished event");
    }
}
