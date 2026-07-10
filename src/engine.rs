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
use crate::resume::{JobCheckpoint, JobMetadata, ResumeError, SCHEMA_VERSION, compute_plan_hash};
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
    /// A [`StripSource`](crate::streaming::StripSource) failed to produce a
    /// strip. This wraps the source's own typed error (e.g. a
    /// [`PdfError`](crate::pdf::PdfError), or any external source's error)
    /// while preserving the [`std::error::Error::source`] chain, instead of
    /// stringifying it into a [`SinkError::Other`] and mis-attributing a
    /// source failure to storage (issue #140).
    #[error("source error: {0}")]
    Source(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
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
    /// The supplied [`PyramidPlan`](crate::PyramidPlan) describes an image
    /// whose dimensions do not match the source raster it was paired with.
    /// The engine validates this at entry so a mismatch surfaces as a typed
    /// error instead of an out-of-bounds slice copy inside the tiling /
    /// canvas-embedding path (a library-level denial of service on untrusted
    /// input).
    #[error(
        "plan/source dimension mismatch: plan describes {plan_width}x{plan_height} \
         but source is {source_width}x{source_height}"
    )]
    PlanSourceMismatch {
        plan_width: u32,
        plan_height: u32,
        source_width: u32,
        source_height: u32,
    },
    /// The supplied [`PyramidPlan`](crate::PyramidPlan) is structurally
    /// invalid (for example, it has no levels). A plan is normally produced by
    /// [`PyramidPlanner::plan`](crate::PyramidPlanner::plan), which upholds
    /// these invariants; the engine re-checks them at entry so a malformed
    /// plan cannot trigger an arithmetic underflow or a silent zero-tile run.
    #[error("invalid plan: {reason}")]
    InvalidPlan { reason: &'static str },
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
///
/// **See also:** the [interactive CLI generator](https://libviprs.org/cli/#cli-generator)
/// composes a runnable program from these same knobs.
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
    /// Optional content digest identifying the source raster this run reads
    /// from. When set, it is folded into the resume plan hash so that
    /// resuming an on-disk checkpoint against a *different* source (even one
    /// with identical dimensions) is rejected instead of silently
    /// interleaving two images. `None` (the default) leaves source identity
    /// out of the hash, preserving the geometry-only behaviour for callers
    /// that do not compute a digest.
    pub source_content_hash: Option<String>,
    /// Optional cooperative-cancellation token. When set, the engine polls it
    /// at level / tile / strip boundaries (and the retry backoff sleeps in
    /// short slices so it can be interrupted) and stops with
    /// [`EngineError::Cancelled`] once the token is cancelled. `None` (the
    /// default) is a run that cannot be cancelled.
    pub cancel: Option<crate::cancel::CancelToken>,
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
            source_content_hash: None,
            cancel: None,
        }
    }
}

impl EngineConfig {
    /// Sets the number of worker threads for parallel tile extraction.
    ///
    /// `0` (the default) means single-threaded execution on the calling thread.
    /// Any positive value spawns that many workers per pyramid level.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-concurrency).
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Sets the bounded-channel capacity between producer threads and the sink consumer.
    ///
    /// A smaller buffer limits memory usage but may cause producers to block
    /// more frequently. A larger buffer smooths out sink latency at the cost
    /// of higher peak memory.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-buffer-size).
    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = n;
        self
    }

    /// Sets the strategy for handling blank (uniform-color) tiles.
    ///
    /// See [`BlankTileStrategy`] for the available options.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-skip-blank)
    /// (and the related [`--blank-tolerance`](https://libviprs.org/cli/#flag-blank-tolerance) flag).
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
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-failure-policy).
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
    /// A non-[`None`](crate::dedupe::DedupeStrategy::None) strategy drives
    /// engine-level deduplication of blank tiles: every exactly-uniform tile is
    /// collapsed into the shared 1-byte placeholder marker instead of a full
    /// payload, so a run over a sparse image no longer writes one complete file
    /// per identical blank tile. The collapse is reported through
    /// [`EngineResult::tiles_skipped`] and, for on-disk sinks, materialised as
    /// the [`BLANK_TILE_MARKER`](crate::sink::BLANK_TILE_MARKER). It is
    /// non-lossy: the marker regenerates to the same uniform tile, so
    /// Verify-mode reconstruction still matches (see
    /// [`regenerated_tile_matches_marker`]).
    ///
    /// Both [`Blanks`](crate::dedupe::DedupeStrategy::Blanks) and
    /// [`All`](crate::dedupe::DedupeStrategy::All) promote uniform content at
    /// this layer (mirroring
    /// [`FsSink::should_dedupe_tile`](crate::sink::FsSink)); they differ only in
    /// the sink-side shared-key hash algorithm, which does not change the
    /// engine's emit decision. Sinks that were themselves given a
    /// [`DedupeStrategy`] continue to apply their own per-sink dedupe on top of
    /// this.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
    /// (and the related [`--dedupe-all`](https://libviprs.org/cli/#flag-dedupe-all) flag).
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

    /// Records a content digest for the source raster so that resume can
    /// reject a checkpoint produced from a different source.
    ///
    /// The digest is opaque to the engine — any string that uniquely
    /// identifies the source bytes works (e.g. the hex BLAKE3 digest exposed
    /// via [`crate::manifest::SourceMetadata::bytes_hash`]). It is folded
    /// into [`crate::resume::compute_plan_hash`]; two runs that pass
    /// different digests will not resume one another's checkpoints.
    pub fn with_source_content_hash(mut self, digest: impl Into<String>) -> Self {
        self.source_content_hash = Some(digest.into());
        self
    }

    /// Attach a [`CancelToken`](crate::cancel::CancelToken) so the run can be
    /// cooperatively cancelled. See the [`cancel`](crate::cancel) module for
    /// the polling contract.
    pub fn with_cancel(mut self, token: crate::cancel::CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Returns `Err(EngineError::Cancelled)` when a cancel token is attached
    /// and has been cancelled; otherwise `Ok(())`. Callers poll this at
    /// cooperative checkpoints in the tiling loops.
    #[inline]
    pub(crate) fn check_cancelled(&self) -> Result<(), EngineError> {
        match &self.cancel {
            Some(token) if token.is_cancelled() => Err(EngineError::Cancelled),
            _ => Ok(()),
        }
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
    /// Time spent extracting tile rasters out of each level (cropping the
    /// region, padding edge tiles, and deciding blankness) before they are
    /// handed to the sink.
    pub extract: Duration,
    /// Time spent encoding tiles (PNG / JPEG / raw).
    ///
    /// Encoding happens inside the sink's `write_tile`, so it is currently
    /// folded into [`StageDurations::sink`] rather than measured separately;
    /// this field stays zero until the engine gains a dedicated encode stage.
    /// It is deliberately **not** where tile *extraction* time is booked — that
    /// lives in [`StageDurations::extract`].
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
    run_pyramid(source, plan, sink, config, observer)
}

/// Internal driver behind [`generate_pyramid_observed`].
///
/// Resume is handled one layer up, in [`crate::engine_builder`]: the builder
/// wraps the user sink in a `ResumeAwareSink` that filters already-completed
/// coordinates and advances the on-disk [`CheckpointState`]. The engine driver
/// itself is intentionally resume-oblivious — it walks the whole plan and hands
/// every tile to the (possibly wrapping) sink.
fn run_pyramid(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
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
    let stage_extract = AtomicU64::new(0);
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
        stage_extract: &stage_extract,
        stage_sink: &stage_sink,
    };

    // Process from top level (full res) down to level 0 (1×1)
    for level_idx in (0..plan.levels.len()).rev() {
        // Cooperative cancellation: stop cleanly at the level boundary before
        // committing to (potentially expensive) downscale + tile emission.
        config.check_cancelled()?;
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

    observer.on_event(EngineEvent::Finished {
        total_tiles: tiles_produced,
        levels: plan.levels.len() as u32,
    });

    let decode_elapsed = stage_decode_done.saturating_duration_since(stage_decode_start);
    let stage_durations = StageDurations {
        planning: stage_planning,
        decode: decode_elapsed,
        resize: Duration::from_nanos(stage_resize.load(Ordering::Relaxed)),
        extract: Duration::from_nanos(stage_extract.load(Ordering::Relaxed)),
        // Encoding is done by the sink's `write_tile`, so its cost is folded
        // into `sink` rather than measured separately here (issue #115).
        encode: Duration::ZERO,
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
    /// Accumulated nanoseconds spent extracting tile rasters (see
    /// [`StageDurations::extract`]). This is **not** encode time — extraction
    /// crops/pads a region, it does not PNG/JPEG-encode it (issue #115).
    stage_extract: &'a AtomicU64,
    stage_sink: &'a AtomicU64,
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
    /// Running metadata. Only its small scalar fields (schema, plan hash,
    /// completed levels, timestamps, format) plus the fixed `inline` slice of
    /// coordinates are re-serialised into the header on every flush; the bulk
    /// of `completed_tiles` is persisted through the append-only segment log.
    meta: std::sync::Mutex<JobMetadata>,
    /// Half-open range `inline_start..inline_end` inside `meta.completed_tiles`
    /// identifying the coordinates that live *inline* in the JSON header rather
    /// than in the segment log. These are the coordinates a resume loaded from
    /// a header that did not already have them in the segment file (a legacy or
    /// hand-written checkpoint). The range is fixed for the life of the run, so
    /// the header stays a bounded size regardless of how many new tiles the run
    /// completes (issue #127). For a fresh run and for a resume off an
    /// engine-written segment log, this range is empty and the header carries
    /// no coordinates at all.
    inline_start: usize,
    inline_end: usize,
    /// Serialises flushes and tracks how many of `meta.completed_tiles` are
    /// already durable in the segment log. Held across the append + header
    /// rewrite so two workers cannot interleave partial segment frames. The
    /// stored value is the exclusive upper index into `completed_tiles`
    /// covered by the log so far; it starts at `inline_end` (everything known
    /// at construction is either inline in the header or already in the log).
    seg_cursor: std::sync::Mutex<usize>,
    /// Monotonically increasing count of tiles marked completed over the whole
    /// run. A flush is triggered whenever this counter lands on a multiple of
    /// `checkpoint_every`; it is never reset. Because [`AtomicU64::fetch_add`]
    /// hands every concurrent caller a *unique* value, exactly one caller
    /// observes each boundary — so no increment can be clobbered and the "every
    /// N tiles" cadence is preserved under parallel marking (issue #113).
    /// `checkpoint_every == 0` means we never perform intermediate flushes
    /// (final flush only).
    completed_counter: std::sync::atomic::AtomicU64,
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
        // Coordinates already durable in the segment log occupy the front of
        // `completed_tiles` (see `JobCheckpoint::load`, which lists segment
        // coordinates first). Everything after that came from the header inline
        // and must keep being written inline until it is migrated into the log
        // by the first flush. Counting physical frames is an O(1) metadata read.
        let segment_frames = crate::resume::count_segment_frames(
            &crate::resume::JobCheckpoint::segments_path(&root),
        )
        .unwrap_or(0);
        let total = meta.completed_tiles.len();
        let inline_start = segment_frames.min(total);
        let inline_end = total;
        Self {
            root,
            meta: std::sync::Mutex::new(meta),
            inline_start,
            inline_end,
            seg_cursor: std::sync::Mutex::new(inline_end),
            completed_counter: AtomicU64::new(0),
            checkpoint_every,
        }
    }

    /// Append a successful write to the metadata. When `checkpoint_every`
    /// tiles have accumulated since the last flush, also persist the
    /// checkpoint to disk so a crash can resume from the latest boundary.
    pub(crate) fn mark_tile_completed(&self, coord: TileCoord) -> Result<(), ResumeError> {
        {
            let mut meta = crate::poison::recover(&self.meta);
            meta.completed_tiles.push(coord);
        }
        // Flush periodically. `checkpoint_every == 0` disables this path and
        // leaves only the final flush (done by `run_pyramid` on success).
        if self.cadence_reached() {
            self.flush()?;
        }
        Ok(())
    }

    /// Advance the flush-cadence counter for one completed tile and report
    /// whether this tile lands exactly on a flush boundary.
    ///
    /// The counter is monotonic and never reset: a flush is due precisely when
    /// the post-increment value is a multiple of `checkpoint_every`. Compared
    /// to the previous check-then-reset scheme (`fetch_add`, compare against
    /// the threshold, then `store(0)`), this is race-free under concurrent
    /// marking. [`AtomicU64::fetch_add`] serialises the increment and returns a
    /// value unique to each caller, so exactly one caller sees each boundary
    /// (`n % checkpoint_every == 0`). The old reset could clobber an increment
    /// interleaved between another worker's `fetch_add` and its `store(0)`,
    /// dropping a tile from the tally and pushing the next checkpoint past the
    /// intended N-tile boundary (issue #113).
    ///
    /// `Relaxed` is sufficient: [`Self::flush`] takes the `meta` mutex
    /// internally, which provides the happens-before edge between the worker
    /// that appended the tile and the thread that serialises the snapshot. The
    /// counter itself is a pure cadence gauge.
    fn cadence_reached(&self) -> bool {
        if self.checkpoint_every == 0 {
            return false;
        }
        let n = self.completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
        n % self.checkpoint_every == 0
    }

    /// Promote the level to `levels_completed` only when every tile in that
    /// level is present in `completed_tiles`. Called by the builder's resume
    /// path once the run returns, with `expected_tiles` set to the level's
    /// full [`LevelPlan::tile_count`].
    ///
    /// A level in which [`FailurePolicy::RetryThenSkip`] dropped one or more
    /// tiles never has all of its coordinates recorded (skipped tiles call
    /// `note_sink_skipped` and never reach [`Self::mark_tile_completed`]), so
    /// such a level must **not** be recorded as completed. Otherwise the
    /// `levels_completed` invariant ("every tile in the level is present in
    /// `completed_tiles`") would be violated, and a consumer honouring the
    /// documented "skip whole levels" resume optimisation would treat the
    /// failure-skipped tiles as done — permanently unrecoverable (issue #125).
    pub(crate) fn mark_level_completed(&self, level: u32, expected_tiles: u64) {
        let mut meta = crate::poison::recover(&self.meta);
        let recorded = meta
            .completed_tiles
            .iter()
            .filter(|c| c.level == level)
            .count() as u64;
        if recorded < expected_tiles {
            return;
        }
        if !meta.levels_completed.contains(&level) {
            meta.levels_completed.push(level);
        }
    }

    /// Persist the checkpoint with *bounded* per-flush I/O (issue #127).
    ///
    /// Rather than re-serialising the whole `completed_tiles` vector into the
    /// JSON header on every flush (which made per-flush cost grow with all
    /// completed tiles, cumulatively O(n²/k)), this appends only the
    /// coordinates completed since the previous flush to the append-only
    /// segment log, then rewrites a small header carrying just the scalar
    /// fields plus the fixed inline coordinate range. The append is a single
    /// bounded write; the header stays a bounded size.
    ///
    /// Ordering: the delta coordinates are appended and fsynced *before* the
    /// header that summarises them is renamed into place, so a crash between
    /// the two leaves the segment log holding coordinates the header does not
    /// yet mention. Those tiles were genuinely written (the coordinate is only
    /// marked after a successful write), so recovering them as completed is
    /// safe; a resume simply skips already-done work.
    ///
    /// The `seg_cursor` lock serialises concurrent flushes so two workers
    /// cannot interleave partial segment frames or double-append the same
    /// delta.
    pub(crate) fn flush(&self) -> Result<(), ResumeError> {
        let mut cursor = crate::poison::recover(&self.seg_cursor);

        // Snapshot only the bounded parts we need under the meta lock: the
        // frozen inline coordinate slice, the newly-completed delta, and the
        // scalar header fields. The full vector is never cloned.
        let (header_meta, delta, new_cursor) = {
            let mut meta = crate::poison::recover(&self.meta);
            meta.last_checkpoint_at = now_rfc3339_engine();
            let len = meta.completed_tiles.len();
            let inline_end = self.inline_end.min(len);
            let inline_start = self.inline_start.min(inline_end);
            let inline = meta.completed_tiles[inline_start..inline_end].to_vec();
            let delta = meta.completed_tiles[(*cursor).min(len)..len].to_vec();
            let header_meta = JobMetadata {
                schema_version: meta.schema_version.clone(),
                plan_hash: meta.plan_hash.clone(),
                completed_tiles: inline,
                levels_completed: meta.levels_completed.clone(),
                started_at: meta.started_at.clone(),
                last_checkpoint_at: meta.last_checkpoint_at.clone(),
                content_format: meta.content_format,
            };
            (header_meta, delta, len)
        };

        // Append the delta first so its bytes are durable before the header
        // that certifies the completed set is published.
        crate::resume::append_segments(&self.root, &delta, &crate::resume::RealDurability)
            .map_err(ResumeError::from)?;
        JobCheckpoint::save(&self.root, &header_meta).map_err(ResumeError::from)?;
        *cursor = new_cursor;
        Ok(())
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
pub(crate) fn promote_sink_error(err: SinkError) -> EngineError {
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
        // A checkpoint-write failure that reached us via the resume-aware sink
        // wrapper must surface with the same variant as the monolithic path,
        // which maps `ResumeError` straight to `EngineError::ResumeFailed`
        // (issue #140).
        SinkError::Checkpoint(e) => EngineError::ResumeFailed(e),
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
    let contract = crate::resume::PlanContract::from_engine(config, sink);
    let meta = JobMetadata {
        schema_version: SCHEMA_VERSION.to_string(),
        plan_hash: compute_plan_hash(plan, &contract),
        completed_tiles,
        levels_completed,
        started_at: now.clone(),
        last_checkpoint_at: now,
        content_format: contract.format,
    };
    Some(CheckpointState::new(
        root,
        meta,
        plan,
        config.checkpoint_every,
    ))
}

/// Verify mode for the monolithic raster path: walks every tile in the
/// plan, reads the on-disk bytes via `sink.checkpoint_root()` joined with
/// the plan's tile path, and returns an error if any tile is missing or
/// (when the manifest records checksums) if the bytes do not match the
/// recorded digest.
///
/// Does NOT call `sink.write_tile`; the test suite asserts that
/// `tiles_produced == 0` and that no files are mutated. Emits
/// `LevelStarted` / `TileCompleted` / `LevelCompleted` / `Finished`
/// events so progress observers see verify runs as first-class.
///
/// Strip-source equivalents live in [`crate::stream_verify`].
pub fn raster_verify(
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

    // If a checkpoint exists, its `plan_hash` must match the plan we're
    // verifying against — otherwise we'd walk every tile just to report a
    // byte mismatch on the first one, which is strictly less useful than
    // failing fast with the structural error.
    if let Some(meta) = JobCheckpoint::load(root)? {
        if let Err(got) = crate::resume::verify_checkpoint_contract(&meta, plan, config, sink) {
            return Err(EngineError::PlanHashMismatch {
                expected: meta.plan_hash,
                got,
            });
        }
    }

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
                // Route through the single shared parser. An unknown / future
                // / typo'd algorithm is a hard verification failure here, not
                // something to silently skip — otherwise a manifest stamped
                // with a bogus algo would pass with zero digests checked
                // (issue #95).
                let algo = crate::manifest::ChecksumAlgo::from_manifest_str(algo_str).ok_or_else(
                    || {
                        EngineError::Sink(SinkError::Other(format!(
                            "Verify: unknown checksum algorithm {algo_str:?} in manifest"
                        )))
                    },
                )?;
                {
                    // A recorded tile that is gone from disk is a verification
                    // failure, not something to skip — unless it is a
                    // manifest-referenced blank whose content lives in
                    // `_shared/` (issue #93).
                    let blank_refs = manifest.get("blank_references").and_then(|v| v.as_object());
                    for (rel, expected) in per_tile {
                        let expected_s = match expected.as_str() {
                            Some(s) => s,
                            None => continue,
                        };
                        // Reject traversal / absolute / prefixed manifest keys
                        // before any filesystem access, and stream the tile
                        // through the hasher to cap memory (see #79).
                        let abs = match crate::checksum::safe_manifest_join(root, rel) {
                            Some(p) => p,
                            None => {
                                return Err(EngineError::Sink(SinkError::Other(format!(
                                    "Verify: manifest tile path escapes checkpoint root: {rel}"
                                ))));
                            }
                        };
                        let got = match crate::checksum::hash_file(&abs, algo) {
                            Ok(g) => g,
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                                if blank_refs.is_some_and(|m| m.contains_key(rel)) {
                                    continue;
                                }
                                return Err(EngineError::Sink(SinkError::MissingTile {
                                    tile_rel_path: rel.clone(),
                                }));
                            }
                            Err(e) => return Err(EngineError::Sink(SinkError::Io(e))),
                        };
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

    // Tile paths whose raw content is a 1-byte placeholder pointing at a
    // deduped payload under `_shared/` (issue #93). These are legitimate
    // markers even when the regenerated tile is not itself blank.
    let blank_ref_paths: std::collections::HashSet<String> = read_manifest(root)
        .and_then(|m| {
            m.get("blank_references")
                .and_then(|v| v.as_object())
                .map(|o| o.keys().cloned().collect())
        })
        .unwrap_or_default();

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
                    if on_disk.len() == 1 && on_disk[0] == crate::sink::BLANK_TILE_MARKER {
                        // A 1-byte placeholder: either a blank-tile marker
                        // (`BlankTileStrategy::Placeholder*`) or a dedupe
                        // reference whose payload lives in `_shared/`. Validate
                        // the regenerated tile's blankness / manifest reference
                        // instead of byte-comparing the marker (issue #94).
                        let is_dedupe_ref = plan
                            .tile_path(coord, &ext)
                            .is_some_and(|rel| blank_ref_paths.contains(&rel));
                        if !is_dedupe_ref && !regenerated_tile_matches_marker(&expected, config) {
                            return Err(EngineError::ChecksumMismatch {
                                tile: coord,
                                expected: "blank tile (placeholder marker)".to_string(),
                                got: "regenerated tile is not blank".to_string(),
                            });
                        }
                    } else if on_disk != expected_bytes {
                        // Raw tiles are byte-exact: any mismatch (truncation,
                        // flipped byte, padding drift) is corruption.
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
/// cleanly. The directory itself is retained.
///
/// Refuses to wipe a directory this crate does not own: a destructive
/// Overwrite must only clear output we plausibly produced, never an
/// arbitrary directory a caller mis-pointed the sink at. A directory is
/// considered owned when it is empty or already holds our checkpoint marker
/// ([`crate::resume::CHECKPOINT_FILENAME`]). Anything else yields
/// [`std::io::ErrorKind::PermissionDenied`] and is left untouched.
pub(crate) fn wipe_directory(dir: &std::path::Path) -> std::io::Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
        return Ok(());
    }
    if !dir.is_dir() {
        return Ok(());
    }

    // Ownership guard: only proceed when the directory is empty or carries
    // our marker. A single pass records both facts.
    let mut is_empty = true;
    let mut owned = false;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        is_empty = false;
        if entry.file_name() == crate::resume::CHECKPOINT_FILENAME {
            owned = true;
            break;
        }
    }
    if !is_empty && !owned {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing to Overwrite-wipe {}: directory is not empty and holds no {} marker",
                dir.display(),
                crate::resume::CHECKPOINT_FILENAME
            ),
        ));
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
    let mut canvas = make_background_tile(cw, ch, bpp, background_rgb);

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
                // Cooperative cancellation: poll before extracting each tile
                // so a long single-threaded level can be stopped promptly.
                config.check_cancelled()?;
                let coord = TileCoord::new(level, col, row);
                let extract_start = Instant::now();
                let tile_raster = extract_tile(raster, plan, coord, config.background_rgb)?;
                ctx.stage_extract
                    .fetch_add(extract_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let blank = blank_for_output(&tile_raster, blank_strategy, config.dedupe_strategy);
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
                    }
                    Err(e) => {
                        ctx.stage_sink
                            .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        // A write that failed because its retry backoff was
                        // interrupted by a cancellation must surface as
                        // Cancelled, not be swallowed by RetryThenSkip or
                        // reported as a plain sink error.
                        config.check_cancelled()?;
                        match &config.failure_policy {
                            FailurePolicy::RetryThenSkip(_) => {
                                // Account the skip on the outermost sink so
                                // it surfaces in EngineResult.
                                sink.note_sink_skipped();
                                // A tile that exhausted RetryThenSkip produced
                                // no output; report it as TileFailed, never as
                                // TileCompleted, so observers that pair
                                // completions with sink writes stay consistent.
                                observer.on_event(EngineEvent::TileFailed {
                                    coord,
                                    error: e.to_string(),
                                });
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
        // There is no producer/consumer channel on the single-threaded path:
        // a tile is extracted, written, and dropped before the next one is
        // touched, so at most one tile is ever "held" at a time. Record a peak
        // occupancy of 1 to reflect that (issue #115).
        let _ = ctx.queue_pressure_peak.fetch_max(1, Ordering::Relaxed);
        Ok((count, skipped))
    } else {
        extract_and_emit_parallel(raster, plan, level, sink, config, observer, ctx)
    }
}

/// Return `true` when the current [`BlankTileStrategy`] wants this tile to be
/// written as a placeholder.
pub(crate) fn is_blank_for_strategy(raster: &Raster, strategy: BlankTileStrategy) -> bool {
    match strategy {
        BlankTileStrategy::Emit => false,
        BlankTileStrategy::Placeholder => is_blank_tile(raster),
        BlankTileStrategy::PlaceholderWithTolerance { max_channel_delta } => {
            is_blank_tile_with_tolerance(raster, max_channel_delta)
        }
    }
}

/// Decide whether a produced tile is emitted as a placeholder marker rather
/// than a full payload, combining the explicit [`BlankTileStrategy`] with the
/// engine-level [`EngineConfig::dedupe_strategy`].
///
/// This is the single point where [`EngineConfig::dedupe_strategy`] takes
/// effect: a non-[`None`](crate::dedupe::DedupeStrategy::None) strategy
/// collapses every exactly-uniform tile into the shared
/// [`BLANK_TILE_MARKER`](crate::sink::BLANK_TILE_MARKER), so identical blank
/// tiles no longer each occupy a full file. Without this the field was a
/// silent no-op — accepted through [`EngineConfig::with_dedupe_strategy`] but
/// never routed into the emit path (issue #130).
///
/// Restricting the dedupe collapse to [`is_blank_tile`] (exact uniformity)
/// keeps the marker regenerable, so Verify-mode reconstruction still matches
/// (see [`regenerated_tile_matches_marker`]). Both
/// [`Blanks`](crate::dedupe::DedupeStrategy::Blanks) and
/// [`All`](crate::dedupe::DedupeStrategy::All) promote uniform content here,
/// matching [`FsSink::should_dedupe_tile`](crate::sink::FsSink); the algorithm
/// carried by `All` only influences the sink-side shared-key naming, not the
/// engine's emit decision.
fn blank_for_output(
    raster: &Raster,
    blank_strategy: BlankTileStrategy,
    dedupe: Option<crate::dedupe::DedupeStrategy>,
) -> bool {
    if is_blank_for_strategy(raster, blank_strategy) {
        return true;
    }
    let dedupe_collapses_blanks = matches!(
        dedupe,
        Some(crate::dedupe::DedupeStrategy::Blanks)
            | Some(crate::dedupe::DedupeStrategy::All { .. })
    );
    dedupe_collapses_blanks && is_blank_tile(raster)
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
    // Engine-level dedupe of blank tiles (issue #130). Captured up front so the
    // per-worker closures can fold it into their blank decision alongside
    // `blank_strategy`; `Option<DedupeStrategy>` is `Copy`.
    let dedupe_strategy = config.dedupe_strategy;

    // Bounded channel for backpressure: producers block when buffer is full.
    // Routed through `crate::sync_queue` so the loom suite can model-check the
    // exact send/recv/teardown protocol this path relies on.
    let (tx, rx) = crate::sync_queue::bounded::<Result<Tile, EngineError>>(config.buffer_size);
    // Queue-occupancy gauge — a producer bumps it right before it hands a tile
    // to the channel, and the consumer drops it as soon as it receives one, so
    // the running value tracks the tiles actually sitting in (or in transit
    // through) the bounded channel and its peak is the true "tiles held in the
    // queue" reported by `queue_pressure_peak`. Incrementing *before* the send
    // (rather than counting active producers) makes the peak sensitive to a
    // backed-up buffer instead of being capped at the worker count (issue #115).
    let in_flight = Arc::new(AtomicU32::new(0));

    // Workers run under `std::thread::scope`, so they cannot outlive this
    // frame; share the raster and plan by borrow rather than deep-cloning
    // them into an `Arc`. The old clone held a full extra copy of the level
    // raster alive for the whole emission (a hidden ~1x-source spike the
    // MemoryTracker never charged); borrowing keeps the real peak in line
    // with `peak_memory_bytes`.

    // Collect every tile coordinate for this level. Resume filtering happens
    // in the builder's `ResumeAwareSink`, not here — the engine walks the
    // whole plan and lets the wrapping sink short-circuit already-completed
    // writes.
    let coords: Vec<TileCoord> = (0..level_plan.rows)
        .flat_map(|row| (0..level_plan.cols).map(move |col| TileCoord::new(level, col, row)))
        .collect();

    if coords.is_empty() {
        return Ok((0, 0));
    }

    // Spawn workers
    let concurrency = config.concurrency.min(coords.len());
    let chunk_size = coords.len().div_ceil(concurrency);

    let stage_extract: &AtomicU64 = ctx.stage_extract;
    let queue_peak: &AtomicU32 = ctx.queue_pressure_peak;

    std::thread::scope(|s| {
        // Spawn producer threads
        for chunk in coords.chunks(chunk_size) {
            let tx = tx.clone();
            let in_flight = Arc::clone(&in_flight);
            let chunk = chunk.to_vec();
            let bg = config.background_rgb;

            s.spawn(move || {
                for coord in chunk {
                    // Cooperative cancellation: stop feeding tiles into the
                    // channel as soon as the run is cancelled. The consumer
                    // observes the cancel independently and returns Cancelled.
                    if config.check_cancelled().is_err() {
                        break;
                    }
                    let extract_start = Instant::now();
                    let result = extract_tile(raster, plan, coord, bg)
                        .map(|tile_raster| {
                            let blank =
                                blank_for_output(&tile_raster, blank_strategy, dedupe_strategy);
                            Tile {
                                coord,
                                raster: tile_raster,
                                blank,
                            }
                        })
                        .map_err(EngineError::from);
                    stage_extract
                        .fetch_add(extract_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    // Queue-occupancy gauge: the tile is about to enter the
                    // channel. Bump *before* the (possibly blocking) send so a
                    // producer stalled on a full buffer is still counted, then
                    // record the peak. The matching decrement happens on the
                    // consumer once it receives the tile — the send→recv edge
                    // guarantees the increment happens-before that decrement, so
                    // the gauge never underflows.
                    //
                    // Pure gauge — no happens-before is needed between the
                    // counter and any tile payload, so `Relaxed` is safe (and
                    // avoids a useless full fence in the hot loop).
                    let cur = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = queue_peak.fetch_max(cur, Ordering::Relaxed);

                    let send_failed = tx.send(result).is_err();
                    if send_failed {
                        // The tile never reached the consumer, so the consumer
                        // will never decrement for it — balance the counter here
                        // before winding down.
                        in_flight.fetch_sub(1, Ordering::Relaxed);
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
            // A tile just left the channel: drop the queue-occupancy gauge the
            // producer bumped before sending it (issue #115). Done first so the
            // running count reflects the drain even if we bail out below.
            in_flight.fetch_sub(1, Ordering::Relaxed);
            // Cooperative cancellation: stop draining the channel and return
            // Cancelled. Producers observe the same token and wind down; the
            // scope join then reaps them.
            config.check_cancelled()?;
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
                }
                Err(e) => {
                    ctx.stage_sink
                        .fetch_add(sink_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    // A write that failed because its retry backoff was
                    // interrupted by a cancellation must surface as Cancelled,
                    // not be swallowed by RetryThenSkip or reported as a plain
                    // sink error.
                    config.check_cancelled()?;
                    match &config.failure_policy {
                        FailurePolicy::RetryThenSkip(_) => {
                            sink.note_sink_skipped();
                            // A tile that exhausted RetryThenSkip produced no
                            // output; report it as TileFailed, never as
                            // TileCompleted, so observers that pair completions
                            // with sink writes stay consistent.
                            observer.on_event(EngineEvent::TileFailed {
                                coord,
                                error: e.to_string(),
                            });
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

/// Allocates a `width × height` pixel buffer filled with the background
/// color.
///
/// Used to pad edge tiles (square, `tile_size × tile_size`) and to build
/// the full-resolution centre canvas (which may be non-square). The
/// background RGB triplet is expanded to match the pixel format's
/// bytes-per-pixel: grayscale uses the red channel, RGB copies all three,
/// RGBA appends alpha=255, and other formats repeat the red channel.
fn make_background_tile(width: u32, height: u32, bpp: usize, background_rgb: [u8; 3]) -> Vec<u8> {
    let mut padded = vec![0u8; width as usize * height as usize * bpp];
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
            let padded = make_background_tile(ts, ts, bpp, background_rgb);
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
        let mut padded = make_background_tile(ts, ts, bpp, background_rgb);
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
        let mut padded = make_background_tile(ts, ts, bpp, background_rgb);

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

/// Decides whether a 1-byte [`crate::sink::BLANK_TILE_MARKER`] on disk is a
/// legitimate stand-in for the regenerated `expected` tile during raw-format
/// Verify.
///
/// The sink writes the marker only for tiles it considered blank under the
/// active [`BlankTileStrategy`], so Verify must re-apply the *same* blankness
/// predicate rather than byte-comparing the marker against the full
/// regenerated tile (issue #94). Under [`BlankTileStrategy::Emit`] no marker
/// should ever have been written by the strategy itself, but dedupe can still
/// emit markers independently; those are resolved by the caller via
/// `blank_references`, and the strict [`is_blank_tile`] check here is a safe
/// last resort.
pub(crate) fn regenerated_tile_matches_marker(expected: &Raster, config: &EngineConfig) -> bool {
    match config.blank_tile_strategy {
        BlankTileStrategy::Placeholder | BlankTileStrategy::Emit => is_blank_tile(expected),
        BlankTileStrategy::PlaceholderWithTolerance { max_channel_delta } => {
            is_blank_tile_with_tolerance(expected, max_channel_delta)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::CollectingObserver;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::sink::MemorySink;

    /// Issue #127 (acceptance criterion 1): checkpoint I/O per flush must be
    /// bounded, i.e. NOT proportional to the total number of completed tiles.
    ///
    /// Today `CheckpointState::flush` re-serialises the entire
    /// `completed_tiles` vector into `.libviprs-job.json` on every flush, so
    /// the on-disk checkpoint grows linearly with the number of completed
    /// tiles and the per-flush write cost is O(n) — cumulatively O(n^2/k).
    ///
    /// This ratchet asserts the header size stays bounded regardless of how
    /// many tiles have been recorded. The fix (issue #127) moves the
    /// coordinate log out of the single JSON header into an append-only
    /// segment file, so each periodic flush appends only the delta and the
    /// header stays small. `JobCheckpoint::load` merges the segment log back
    /// in, so the recorded set is still fully recoverable. The
    /// `libviprs-tests` integration suite was updated in lockstep to read the
    /// checkpoint through `JobCheckpoint::load` rather than raw-parsing the
    /// header.
    #[test]
    fn checkpoint_flush_io_is_bounded_per_flush() {
        use crate::resume::{CHECKPOINT_FILENAME, JobCheckpoint, JobMetadata};

        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let meta = JobMetadata::new("deadbeef".to_string(), "1970-01-01T00:00:00Z".into());
        let cp = CheckpointState::new(dir.path().to_path_buf(), meta, &plan, 1);

        let header = dir.path().join(CHECKPOINT_FILENAME);

        cp.mark_tile_completed(TileCoord::new(0, 0, 0)).unwrap();
        let size_after_few = std::fs::metadata(&header).unwrap().len();

        for i in 0..5_000u32 {
            cp.mark_tile_completed(TileCoord::new(0, i % 2, i / 2))
                .unwrap();
        }
        let size_after_many = std::fs::metadata(&header).unwrap().len();

        // The header must not balloon as more tiles are recorded.
        assert!(
            size_after_many <= size_after_few + 512,
            "per-flush checkpoint I/O is unbounded: header grew from {size_after_few} to \
             {size_after_many} bytes as completed-tile count increased"
        );

        // And the recorded set must still be fully recoverable.
        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        assert!(loaded.completed_tiles.len() >= 5_000);
    }

    /// Reproducer for #117: a poisoned checkpoint-`meta` lock must not cascade
    /// a second panic into every later `mark_tile_completed` — which, on the
    /// write path, would also abort the final checkpoint flush. We poison the
    /// meta mutex under `catch_unwind`, then keep marking tiles. Before the fix
    /// (`.lock().unwrap()`) the next mark panicked (RED); after it the guard is
    /// recovered and the pre-poison completions survive (GREEN).
    #[test]
    fn poisoned_checkpoint_meta_recovers_without_cascade() {
        use crate::resume::JobMetadata;

        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let meta = JobMetadata::new("deadbeef".to_string(), "1970-01-01T00:00:00Z".into());
        // checkpoint_every = 0 keeps the test off the disk-flush path so it
        // isolates the in-memory `meta` lock recovery.
        let cp = CheckpointState::new(dir.path().to_path_buf(), meta, &plan, 0);

        cp.mark_tile_completed(TileCoord::new(0, 0, 0)).unwrap();

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cp.meta.lock().unwrap();
            panic!("worker panic while holding the checkpoint meta lock");
        }));
        assert!(poisoned.is_err());
        assert!(cp.meta.is_poisoned());

        // The lifecycle bookkeeping must keep working after the poison.
        cp.mark_tile_completed(TileCoord::new(0, 1, 0)).unwrap();
        cp.mark_level_completed(0, 1);
        let recorded = crate::poison::recover(&cp.meta).completed_tiles.len();
        assert_eq!(
            recorded, 2,
            "recovered meta must retain the pre-poison completion and accept new ones"
        );
    }

    /// Issue #140: a checkpoint-write failure must surface as *one* variant
    /// regardless of the code path. The monolithic tile loop maps
    /// `ResumeError` straight to `EngineError::ResumeFailed`; the resume-wrapper
    /// sink can only return a `SinkError`, so it now carries the typed
    /// `SinkError::Checkpoint`. `promote_sink_error` must lift that back to the
    /// same `EngineError::ResumeFailed` variant the monolithic path produces —
    /// otherwise the identical failure is reported as two different errors.
    #[test]
    fn checkpoint_failure_unifies_to_resume_failed() {
        let resume = || ResumeError::SchemaMismatch {
            expected: "1",
            found: "99".to_string(),
        };

        // Path A: monolithic loop maps ResumeError -> EngineError directly.
        let monolithic: EngineError = EngineError::from(resume());
        assert!(matches!(monolithic, EngineError::ResumeFailed(_)));

        // Path B: resume-wrapper sink returns SinkError::Checkpoint, which the
        // engine promotes on the way out.
        let via_sink = promote_sink_error(SinkError::Checkpoint(resume()));
        assert!(
            matches!(via_sink, EngineError::ResumeFailed(_)),
            "checkpoint failure via the sink path must promote to the same \
             EngineError::ResumeFailed variant as the monolithic path, got {via_sink:?}"
        );
    }

    /// Issue #125: a level in which `RetryThenSkip` dropped a tile must NOT be
    /// recorded in `levels_completed`, whose documented invariant is that every
    /// tile of a completed level is present in `completed_tiles`. A skipped
    /// tile calls `note_sink_skipped` and never reaches `mark_tile_completed`,
    /// so the level is genuinely incomplete; recording it would let a consumer
    /// honouring the "skip whole levels" resume optimisation treat the dropped
    /// tile as done — permanently unrecoverable.
    ///
    /// Before the fix `mark_level_completed` pushed unconditionally, so the
    /// partial level was recorded (RED). After the fix the push is gated on
    /// every tile of the level being present in `completed_tiles`, so it is
    /// withheld until the level truly completes (GREEN).
    #[test]
    fn partial_level_is_not_recorded_completed() {
        use crate::resume::JobMetadata;

        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        // Pick a level with more than one tile so a single dropped tile leaves
        // a detectable gap.
        let level = plan
            .levels
            .iter()
            .find(|l| l.tile_count() >= 2)
            .expect("plan must contain a multi-tile level");
        let level_id = level.level;
        let total = level.tile_count();

        let meta = JobMetadata::new("deadbeef".to_string(), "1970-01-01T00:00:00Z".into());
        // checkpoint_every == 0 keeps this a pure in-memory bookkeeping test
        // (no intermediate disk flushes).
        let cp = CheckpointState::new(dir.path().to_path_buf(), meta, &plan, 0);

        // Enumerate every tile of the level, then hold one back — emulating a
        // tile dropped by `FailurePolicy::RetryThenSkip`, which records the
        // skip on the sink but never calls `mark_tile_completed`.
        let mut coords = Vec::new();
        for row in 0..level.rows {
            for col in 0..level.cols {
                coords.push(TileCoord::new(level_id, col, row));
            }
        }
        let dropped = coords.pop().unwrap();
        for c in &coords {
            cp.mark_tile_completed(*c).unwrap();
        }
        assert_eq!(coords.len() as u64, total - 1);

        // A level still missing a tile must not be recorded as completed.
        cp.mark_level_completed(level_id, total);
        assert!(
            !cp.meta.lock().unwrap().levels_completed.contains(&level_id),
            "level {level_id} was recorded completed while tile {dropped:?} is \
             still missing from completed_tiles — violates the levels_completed \
             invariant (issue #125)"
        );

        // Once the final tile lands, the level genuinely completes and may be
        // recorded.
        cp.mark_tile_completed(dropped).unwrap();
        cp.mark_level_completed(level_id, total);
        assert!(
            cp.meta.lock().unwrap().levels_completed.contains(&level_id),
            "level {level_id} has all {total} tiles recorded but was not marked completed"
        );
    }

    /// Issue #113: the flush cadence must be preserved under concurrent
    /// marking — no increment may be lost or double-counted, so over a run of
    /// `T` completed tiles at cadence `N` exactly `floor(T / N)` flush
    /// boundaries are hit, independent of how the worker threads interleave.
    ///
    /// The previous implementation used a non-atomic check-then-reset
    /// (`fetch_add(1)`, compare `>= N`, then `store(0)`). A third increment
    /// interleaved between one worker's `fetch_add` and its `store(0)` was
    /// clobbered by the reset, dropping tiles from the tally (fewer boundaries
    /// than expected), while two workers straddling the threshold before either
    /// reset both fired (more boundaries than expected). Either way the count
    /// diverged from `floor(T / N)` (RED). The monotonic modulo counter hands
    /// each caller a unique value and never resets, so exactly one caller sees
    /// each boundary and the count is exact (GREEN).
    #[test]
    fn checkpoint_cadence_preserved_under_concurrency() {
        use crate::resume::JobMetadata;
        use std::sync::Barrier;
        use std::sync::atomic::AtomicU64;

        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let meta = JobMetadata::new("deadbeef".to_string(), "1970-01-01T00:00:00Z".into());

        const EVERY: u64 = 4;
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 250_000;
        let total = THREADS * PER_THREAD;
        let expected = total / EVERY;

        let cp = Arc::new(CheckpointState::new(
            dir.path().to_path_buf(),
            meta,
            &plan,
            EVERY,
        ));
        // Count cadence boundaries directly off the atomic decision so the
        // stress loop stays in-memory (no per-tile disk flush) and hammers the
        // exact increment path that raced. All threads start together to
        // maximise contention on the shared counter.
        let boundaries = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(THREADS as usize));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let cp = Arc::clone(&cp);
                let boundaries = Arc::clone(&boundaries);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..PER_THREAD {
                        if cp.cadence_reached() {
                            boundaries.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let hit = boundaries.load(Ordering::Relaxed);
        assert_eq!(
            hit, expected,
            "flush cadence lost or duplicated under concurrent marking (issue #113): \
             {total} tiles at every={EVERY} must hit exactly {expected} flush boundaries, got {hit}"
        );
    }

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

    /// Issue #130: `EngineConfig::dedupe_strategy` must not be a silent no-op.
    ///
    /// A non-`None` strategy has to route into the engine's emit path and
    /// collapse uniform (blank) tiles into placeholder markers. Over a solid
    /// raster every tile is uniform, so with dedupe disabled nothing is
    /// skipped, and with dedupe enabled every tile is collapsed. Before the fix
    /// the strategy was accepted but never consulted, so `tiles_skipped` stayed
    /// `0` for `Blanks`/`All` and these assertions failed (RED); after the fix
    /// they hold (GREEN). Runs both the single-threaded and parallel emit paths.
    #[test]
    fn dedupe_strategy_drives_blank_tile_collapse() {
        use crate::dedupe::DedupeStrategy;

        let src = solid_raster(512, 512, 255);
        let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let total = plan.total_tile_count();
        assert!(total > 1, "need multiple tiles to demonstrate a collapse");

        for concurrency in [0usize, 4] {
            let run = |dedupe: Option<DedupeStrategy>| {
                let mut config = EngineConfig::default().with_concurrency(concurrency);
                if let Some(ds) = dedupe {
                    config = config.with_dedupe_strategy(ds);
                }
                let sink = MemorySink::new();
                let result =
                    generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
                // Every tile is still produced — the collapse is non-lossy, it
                // only changes how a tile is represented, not whether it exists.
                assert_eq!(result.tiles_produced, total);
                assert_eq!(sink.tile_count() as u64, total);
                result.tiles_skipped
            };

            // Baseline: the knob defaults to a no-op, so no tile is collapsed.
            let skipped_none = run(None);
            assert_eq!(
                skipped_none, 0,
                "concurrency={concurrency}: without a dedupe strategy no tile should be collapsed"
            );

            // Each active strategy must change behaviour versus the baseline.
            let skipped_blanks = run(Some(DedupeStrategy::Blanks));
            assert_eq!(
                skipped_blanks, total,
                "concurrency={concurrency}: DedupeStrategy::Blanks must collapse every uniform tile"
            );
            assert_ne!(
                skipped_blanks, skipped_none,
                "concurrency={concurrency}: DedupeStrategy::Blanks must differ from no dedupe"
            );

            let skipped_all = run(Some(DedupeStrategy::All {
                algo: crate::manifest::ChecksumAlgo::Blake3,
            }));
            assert_eq!(
                skipped_all, total,
                "concurrency={concurrency}: DedupeStrategy::All must collapse every uniform tile"
            );
            assert_ne!(
                skipped_all, skipped_none,
                "concurrency={concurrency}: DedupeStrategy::All must differ from no dedupe"
            );
        }
    }

    /// Issue #128: a tile that terminally fails under `RetryThenSkip` must be
    /// reported as [`EngineEvent::TileFailed`], never as
    /// [`EngineEvent::TileCompleted`]. The failure/skip vocabulary landed in
    /// #199; this pins the engine emission site — both the single-threaded and
    /// the parallel consumer path — to actually emit it.
    ///
    /// Before the fix the `RetryThenSkip` arm emitted `TileCompleted` for the
    /// dropped tile, so an observer pairing completions with sink writes
    /// over-counted and never learned the tile failed (RED). After the fix the
    /// dropped tile surfaces as `TileFailed { coord, error }` with a non-empty
    /// error and no `TileCompleted` is emitted for it (GREEN).
    #[test]
    fn retry_then_skip_emits_tile_failed_not_completed() {
        use crate::retry::{FailurePolicy, RetryPolicy};

        // A sink that terminally fails writing exactly one coordinate and
        // accepts every other tile, so we get a mix of failed and completed
        // tiles in the same run.
        struct FailCoordSink {
            target: TileCoord,
        }
        impl TileSink for FailCoordSink {
            fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
                if tile.coord == self.target {
                    Err(SinkError::Other(
                        "simulated terminal write failure".to_string(),
                    ))
                } else {
                    Ok(())
                }
            }
        }

        let src = gradient_raster(512, 512);
        let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        // The top level is a 2×2 grid of tiles, so a single failing coord
        // leaves sibling tiles that must still complete normally.
        let top = (plan.levels.len() - 1) as u32;
        let target = TileCoord::new(top, 1, 1);

        // Cover both emission sites: single-threaded (concurrency == 0) and the
        // parallel consumer loop (concurrency > 0).
        for concurrency in [0usize, 4] {
            let config = EngineConfig::default()
                .with_concurrency(concurrency)
                // `fail_fast` retry policy: no waiting, straight to the skip arm.
                .with_failure_policy(FailurePolicy::RetryThenSkip(RetryPolicy::fail_fast()));
            let sink = FailCoordSink { target };
            let obs = CollectingObserver::new();

            // The run itself succeeds — RetryThenSkip swallows the drop.
            generate_pyramid_observed(&src, &plan, &sink, &config, &obs).unwrap();
            let events = obs.events();

            // The dropped tile must NOT be reported as completed.
            assert!(
                !events.iter().any(|e| matches!(
                    e,
                    EngineEvent::TileCompleted { coord } if *coord == target
                )),
                "concurrency={concurrency}: dropped tile {target:?} was emitted as \
                 TileCompleted — violates issue #128"
            );

            // It must surface exactly once as TileFailed, carrying the error.
            let failed: Vec<String> = events
                .iter()
                .filter_map(|e| match e {
                    EngineEvent::TileFailed { coord, error } if *coord == target => {
                        Some(error.clone())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                failed.len(),
                1,
                "concurrency={concurrency}: expected exactly one TileFailed for {target:?}, \
                 got {failed:?}"
            );
            assert!(
                !failed[0].is_empty(),
                "concurrency={concurrency}: TileFailed must carry a non-empty error description"
            );

            // Sibling tiles on the same level still complete normally.
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    EngineEvent::TileCompleted { coord }
                        if coord.level == top && *coord != target
                )),
                "concurrency={concurrency}: sibling tiles should still emit TileCompleted"
            );
        }
    }

    /// Issue #130: the engine-level collapse must reach the sink as real
    /// placeholder markers, not just a counter. On disk (raw format) a
    /// collapsed uniform tile is the single-byte
    /// [`BLANK_TILE_MARKER`](crate::sink::BLANK_TILE_MARKER); without a dedupe
    /// strategy the same tile carries its full raw payload.
    #[test]
    fn dedupe_strategy_writes_placeholder_markers_on_disk() {
        use crate::dedupe::DedupeStrategy;
        use crate::sink::{BLANK_TILE_MARKER, FsSink, TileFormat};

        let src = solid_raster(256, 256, 255);
        let plan = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let count_marker_files = |dir: &std::path::Path| -> (usize, usize) {
            let mut markers = 0usize;
            let mut full = 0usize;
            for entry in walkdir(dir) {
                if entry.extension().and_then(|e| e.to_str()) == Some("raw") {
                    let len = std::fs::metadata(&entry).unwrap().len();
                    if len == 1 {
                        let byte = std::fs::read(&entry).unwrap()[0];
                        assert_eq!(byte, BLANK_TILE_MARKER, "1-byte tile must be the marker");
                        markers += 1;
                    } else {
                        full += 1;
                    }
                }
            }
            (markers, full)
        };

        // Baseline: no dedupe -> every raw tile is a full payload.
        let base = tempfile::tempdir().unwrap();
        let sink = FsSink::new(base.path().join("out"), plan.clone()).with_format(TileFormat::Raw);
        let config = EngineConfig::default();
        generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        sink.finish().unwrap();
        let (base_markers, base_full) = count_marker_files(&base.path().join("out"));
        assert_eq!(
            base_markers, 0,
            "no dedupe must not write any 1-byte markers"
        );
        assert!(base_full > 0, "expected full raw tiles without dedupe");

        // Dedupe enabled -> every uniform tile is collapsed to a 1-byte marker.
        let dd = tempfile::tempdir().unwrap();
        let sink = FsSink::new(dd.path().join("out"), plan.clone()).with_format(TileFormat::Raw);
        let config = EngineConfig::default().with_dedupe_strategy(DedupeStrategy::Blanks);
        generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        sink.finish().unwrap();
        let (dd_markers, dd_full) = count_marker_files(&dd.path().join("out"));
        assert!(
            dd_markers > 0,
            "dedupe strategy must materialise placeholder markers on disk"
        );
        assert_eq!(
            dd_full, 0,
            "every uniform tile of a solid raster must collapse to a marker under dedupe"
        );
        assert_ne!(
            dd_markers, base_markers,
            "dedupe strategy must change on-disk output versus no dedupe"
        );
    }

    /// Minimal recursive walk of a directory tree, returning every file path.
    /// Kept local to the test module so the reproducer does not pull in an
    /// extra dependency.
    fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
        out
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
    /// Finished events from every resume path, not just the non-resumable
    /// one. Runs all three ResumeModes via `EngineBuilder::with_resume`
    /// against a CollectingObserver and asserts each one drives at least
    /// the expected shape of events.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resumable_emits_observer_events() {
        use std::sync::Arc;

        use crate::observe::CollectingObserver;
        use crate::resume::ResumePolicy;
        use crate::{EngineBuilder, EngineKind};
        use tempfile::tempdir;

        let src = gradient_raster(128, 96);
        let planner = PyramidPlanner::new(128, 96, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let dir = tempdir().unwrap();
        let sink = crate::sink::FsSink::new(dir.path().join("tiles"), plan.clone())
            .with_format(crate::sink::TileFormat::Raw);

        // --- Overwrite --------------------------------------------------
        let obs = Arc::new(CollectingObserver::new());
        EngineBuilder::new(&src, plan.clone(), &sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite())
            .with_observer_arc(obs.clone())
            .run()
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
        let obs = Arc::new(CollectingObserver::new());
        EngineBuilder::new(&src, plan.clone(), &sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .with_observer_arc(obs.clone())
            .run()
            .unwrap();
        // Resume with a full checkpoint short-circuits without per-tile
        // work, so we only require the engine to have produced *some*
        // observer activity (the Finished event at minimum).
        assert!(
            !obs.events().is_empty(),
            "Resume mode produced no observer events"
        );

        // --- Verify -----------------------------------------------------
        let obs = Arc::new(CollectingObserver::new());
        EngineBuilder::new(&src, plan.clone(), &sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::verify())
            .with_observer_arc(obs.clone())
            .run()
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

    /// Overwrite must clear the sink's OWN output directory (removing stale
    /// tiles) while leaving the unrelated contents of a caller-supplied
    /// `checkpoint_root` untouched. The previous behaviour wiped whatever
    /// `resolve_checkpoint_root` resolved to — which prefers the config's
    /// `checkpoint_root` — deleting the user's metadata dir and leaving the
    /// stale tiles behind (issue #123).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn overwrite_clears_sink_not_user_checkpoint_root() {
        use crate::resume::{CHECKPOINT_FILENAME, ResumePolicy};
        use crate::{EngineBuilder, EngineKind};
        use tempfile::tempdir;

        let src = gradient_raster(128, 96);
        let planner = PyramidPlanner::new(128, 96, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        // The sink writes tiles here. Pre-populate it with a stale tile plus
        // our ownership marker so Overwrite is entitled to clear it.
        let out = tempdir().unwrap();
        let out_dir = out.path().join("tiles");
        std::fs::create_dir_all(&out_dir).unwrap();
        std::fs::write(out_dir.join(CHECKPOINT_FILENAME), b"{}").unwrap();
        let stale = out_dir.join("stale_tile.raw");
        std::fs::write(&stale, b"old bytes").unwrap();

        // A SEPARATE, user-supplied checkpoint_root holding unrelated files
        // the caller expects to keep.
        let ckpt = tempdir().unwrap();
        let ckpt_dir = ckpt.path().to_path_buf();
        let sentinel = ckpt_dir.join("keep-me.txt");
        std::fs::write(&sentinel, b"precious").unwrap();

        let sink = crate::sink::FsSink::new(out_dir.clone(), plan.clone())
            .with_format(crate::sink::TileFormat::Raw);

        let policy = ResumePolicy::overwrite().with_checkpoint_root(ckpt_dir.clone());
        EngineBuilder::new(&src, plan.clone(), &sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(policy)
            .run()
            .unwrap();

        assert!(
            sentinel.exists(),
            "Overwrite wiped an unrelated file in the user's checkpoint_root"
        );
        assert!(
            !stale.exists(),
            "Overwrite left a stale tile in the sink's own output directory"
        );
        assert!(
            out_dir.read_dir().unwrap().next().is_some(),
            "Overwrite produced no output in the sink dir"
        );
    }

    /// `with_config` must never fabricate a destructive Overwrite policy just
    /// because the config carried checkpoint knobs. With no explicit resume
    /// mode attached, the run must be non-destructive and leave the
    /// checkpoint_root's contents intact (issue #123).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn with_config_does_not_fabricate_destructive_overwrite() {
        use crate::{EngineBuilder, EngineKind};
        use tempfile::tempdir;

        let src = gradient_raster(128, 96);
        let planner = PyramidPlanner::new(128, 96, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let out = tempdir().unwrap();
        let out_dir = out.path().join("tiles");

        let ckpt = tempdir().unwrap();
        let sentinel = ckpt.path().join("keep-me.txt");
        std::fs::write(&sentinel, b"precious").unwrap();

        let cfg = EngineConfig {
            checkpoint_every: 4,
            checkpoint_root: Some(ckpt.path().to_path_buf()),
            ..EngineConfig::default()
        };

        let sink = crate::sink::FsSink::new(out_dir.clone(), plan.clone())
            .with_format(crate::sink::TileFormat::Raw);

        // No explicit resume policy — only `with_config` carrying the knobs.
        EngineBuilder::new(&src, plan.clone(), &sink)
            .with_engine(EngineKind::Monolithic)
            .with_config(cfg)
            .run()
            .unwrap();

        assert!(
            sentinel.exists(),
            "with_config implicitly enabled a destructive Overwrite and wiped the checkpoint_root"
        );
    }

    /// The wipe helper must refuse a directory it does not own — one that is
    /// neither empty nor holds our `.libviprs-job.json` marker — so a
    /// mis-pointed sink dir can never delete unrelated user files (issue #123).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn wipe_directory_refuses_non_owned_dir() {
        use tempfile::tempdir;

        let d = tempdir().unwrap();
        let foreign = d.path().join("not-ours.txt");
        std::fs::write(&foreign, b"data").unwrap();

        let res = wipe_directory(d.path());
        assert!(
            res.is_err(),
            "wipe_directory must refuse a directory it does not own"
        );
        assert!(
            foreign.exists(),
            "wipe_directory deleted a file in a non-owned directory"
        );
    }

    /// The wipe helper still fully clears a directory it DOES own — one that
    /// carries our marker — including nested tile dirs and the marker itself.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn wipe_directory_clears_owned_dir() {
        use crate::resume::CHECKPOINT_FILENAME;
        use tempfile::tempdir;

        let d = tempdir().unwrap();
        std::fs::write(d.path().join(CHECKPOINT_FILENAME), b"{}").unwrap();
        std::fs::write(d.path().join("0_0.raw"), b"tile").unwrap();
        std::fs::create_dir(d.path().join("1")).unwrap();
        std::fs::write(d.path().join("1").join("0_0.raw"), b"tile").unwrap();

        wipe_directory(d.path()).unwrap();
        assert!(d.path().exists(), "the directory itself must be retained");
        assert_eq!(
            d.path().read_dir().unwrap().count(),
            0,
            "an owned directory must be fully cleared"
        );
    }

    /// Resume must reject a run whose output contract changed, not just its
    /// geometry. Here run 1 writes a full PNG pyramid + checkpoint; run 2
    /// resumes the same directory but asks for JPEG tiles. Because the tile
    /// format is part of the content contract, the resume gate must fail
    /// with [`EngineError::PlanHashMismatch`] instead of silently keeping
    /// the `.png` tiles under a manifest that now declares `.jpeg`.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_rejects_tile_format_change() {
        use crate::resume::ResumePolicy;
        use crate::sink::{FsSink, TileFormat};
        use crate::{EngineBuilder, EngineKind};
        use tempfile::tempdir;

        let src = gradient_raster(128, 96);
        let plan = PyramidPlanner::new(128, 96, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let dir = tempdir().unwrap();
        let root = dir.path().join("tiles");

        // Run 1: full pyramid as PNG, writing a resume checkpoint.
        let sink_png = FsSink::new(root.clone(), plan.clone()).with_format(TileFormat::Png);
        EngineBuilder::new(&src, plan.clone(), &sink_png)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite())
            .run()
            .unwrap();

        // Run 2: resume the SAME directory but with a different tile format.
        let sink_jpeg =
            FsSink::new(root.clone(), plan.clone()).with_format(TileFormat::Jpeg { quality: 80 });
        let err = EngineBuilder::new(&src, plan.clone(), &sink_jpeg)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run()
            .expect_err("resume with a changed tile format must be rejected");
        assert!(
            matches!(err, EngineError::PlanHashMismatch { .. }),
            "expected PlanHashMismatch on tile-format change, got {err:?}"
        );
    }

    /**
     * Tests that a centred non-square canvas embeds and renders successfully.
     * `embed_in_canvas` must allocate a `canvas_width × canvas_height` buffer;
     * a square allocation makes `Raster::new` reject any non-square canvas with
     * `BufferSizeMismatch`. Works by building a centred DeepZoom plan whose
     * canvas is 1024x256, embedding the source directly, and running the full
     * pyramid to the sink.
     * Input: 1000x200 image, tile_size=256, centre=true -> canvas 1024x256.
     */
    #[test]
    fn centred_non_square_canvas_renders() {
        let src = gradient_raster(1000, 200);
        let plan = PyramidPlanner::new(1000, 200, 256, 0, Layout::DeepZoom)
            .unwrap()
            .with_centre(true)
            .plan();

        // Sanity: the plan really is a non-square centred canvas that offsets
        // the image on both axes (so embed_in_canvas is exercised).
        assert_ne!(
            plan.canvas_width, plan.canvas_height,
            "expected a non-square canvas"
        );
        assert!(plan.centre_offset_x > 0 && plan.centre_offset_y > 0);

        // Direct embed must succeed and produce a canvas-sized raster.
        let canvas = embed_in_canvas(&src, &plan, EngineConfig::default().background_rgb).unwrap();
        assert_eq!(canvas.width(), plan.canvas_width);
        assert_eq!(canvas.height(), plan.canvas_height);

        // The source pixels must survive at the centre offset, and the
        // background must fill the padding above the image.
        let bpp = src.format().bytes_per_pixel();
        let ox = plan.centre_offset_x as usize;
        let oy = plan.centre_offset_y as usize;
        let dst_stride = plan.canvas_width as usize * bpp;
        let embedded = &canvas.data()[oy * dst_stride + ox * bpp..oy * dst_stride + ox * bpp + bpp];
        let src_first = &src.data()[..bpp];
        assert_eq!(embedded, src_first, "source pixel lost at centre offset");

        // End-to-end: the whole pyramid renders without error.
        let sink = MemorySink::new();
        let config = EngineConfig::default();
        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
        assert_eq!(sink.tile_count() as u64, plan.total_tile_count());
    }

    /// Raw-format Verify must recognize the 1-byte `BLANK_TILE_MARKER` that
    /// `BlankTileStrategy::Placeholder` writes for blank tiles, rather than
    /// byte-comparing the marker against the regenerated full tile (issue
    /// #94). A fully-uniform source produces an all-placeholder pyramid, so
    /// every on-disk tile is a marker.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn raster_verify_accepts_raw_placeholder_markers() {
        use crate::sink::{FsSink, TileFormat};

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (w, h, ts) = (256u32, 256u32, 128u32);
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let data = vec![7u8; w as usize * h as usize * bpp];
        let src = Raster::new(w, h, PixelFormat::Rgb8, data).unwrap();
        let plan = PyramidPlanner::new(w, h, ts, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let sink = FsSink::new(&out, plan.clone()).with_format(TileFormat::Raw);
        // Match the background to the solid fill so every edge-padded tile is
        // also uniform, producing an all-placeholder pyramid.
        let mut cfg =
            EngineConfig::default().with_blank_tile_strategy(BlankTileStrategy::Placeholder);
        cfg.background_rgb = [7, 7, 7];
        generate_pyramid_observed(&src, &plan, &sink, &cfg, &NoopObserver).unwrap();

        // Setup sanity: the run really did emit 1-byte markers on disk.
        let first = plan.tile_coords().next().unwrap();
        let rel = plan.tile_path(first, "raw").unwrap();
        let on_disk = std::fs::read(out.join(&rel)).unwrap();
        assert_eq!(
            on_disk,
            vec![crate::sink::BLANK_TILE_MARKER],
            "placeholder run should write a 1-byte marker on disk"
        );

        raster_verify(&src, &plan, &sink, &cfg, &NoopObserver)
            .expect("raw placeholder pyramid must verify");
    }

    /// Issue #95: the monolithic `raster_verify` path must FAIL when the
    /// manifest records an unknown checksum algorithm, rather than mapping it
    /// to `None` and skipping the per-tile digest phase (which reported an
    /// intact pyramid as verified with zero digests checked).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn raster_verify_rejects_unknown_algo() {
        use crate::checksum::ChecksumMode;
        use crate::manifest::{ChecksumAlgo, ManifestBuilder};
        use crate::sink::{FsSink, TileFormat};
        use tempfile::tempdir;

        let src = gradient_raster(256, 256);
        let plan = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let out = tempdir().unwrap();
        let out_dir = out.path().join("tiles");
        let sink = FsSink::new(out_dir.clone(), plan.clone())
            .with_format(TileFormat::Raw)
            .with_manifest(ManifestBuilder::new())
            .with_checksums(ChecksumMode::EmitOnly, ChecksumAlgo::Blake3);

        let cfg = EngineConfig::default();
        generate_pyramid_observed(&src, &plan, &sink, &cfg, &NoopObserver).unwrap();

        // Re-stamp both emitted manifest copies with a bogus algorithm,
        // leaving the correct per-tile digests intact.
        for path in [
            out.path().join("tiles.manifest.json"),
            out_dir.join("manifest.json"),
        ] {
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v.get_mut("checksums")
                .and_then(|c| c.as_object_mut())
                .expect("manifest must record a checksums table")
                .insert("algo".into(), serde_json::json!("totally-bogus-algo"));
            std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        }

        let err = raster_verify(&src, &plan, &sink, &cfg, &NoopObserver)
            .expect_err("verify must reject a manifest with an unknown checksum algorithm");
        match err {
            EngineError::Sink(SinkError::Other(msg)) => assert!(
                msg.contains("unknown checksum algorithm"),
                "unexpected error message: {msg}"
            ),
            other => panic!("expected SinkError::Other for unknown algo, got {other:?}"),
        }
    }

    /// Issue #115: tile *extraction* time must be booked into the `extract`
    /// stage, not mislabeled as `encode`.
    ///
    /// [`StageDurations::encode`] is documented as PNG/JPEG/raw *encoding*,
    /// which happens inside the sink (and is folded into `sink`). The engine's
    /// per-tile crop/pad work is extraction, not encoding. Before the fix that
    /// wall time was accumulated into the `encode` field, so `encode > 0` and
    /// `extract` did not exist. After the fix extraction lands in `extract`
    /// while `encode` stays zero. Exercises both the single-threaded (0) and
    /// parallel (4) emit paths.
    #[test]
    fn extraction_time_is_booked_into_extract_not_encode() {
        let src = gradient_raster(512, 512);
        let plan = PyramidPlanner::new(512, 512, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert!(
            plan.total_tile_count() > 1,
            "need several tiles so extraction takes measurable time"
        );

        for concurrency in [0usize, 4] {
            let config = EngineConfig::default().with_concurrency(concurrency);
            let sink = MemorySink::new();
            let result =
                generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();
            let stages = &result.stage_durations;

            assert!(
                stages.extract > Duration::ZERO,
                "concurrency={concurrency}: extraction time must be recorded in the `extract` \
                 stage, got {:?}",
                stages.extract
            );
            assert_eq!(
                stages.encode,
                Duration::ZERO,
                "concurrency={concurrency}: `encode` is sink-side PNG/JPEG time and must not be \
                 fed by tile extraction (issue #115); got {:?}",
                stages.encode
            );
        }
    }

    /// Issue #115: `queue_pressure_peak` must reflect the tiles actually held in
    /// the producer/consumer channel, not merely the count of active producers.
    ///
    /// The old gauge counted producers currently extracting/sending, so its peak
    /// was capped by the worker count and blind to a backed-up buffer. Here the
    /// consumer is deliberately slow while a comfortably-sized buffer feeds two
    /// workers, so the channel fills well past the worker count. Before the fix
    /// the peak could not exceed `concurrency`; after it, it tracks true channel
    /// occupancy and climbs into the buffer.
    #[test]
    fn queue_pressure_peak_tracks_buffer_occupancy_not_worker_count() {
        use std::sync::atomic::AtomicU64;

        /// Sink whose every write sleeps briefly so the bounded channel backs
        /// up behind a slow consumer.
        struct SlowSink {
            writes: AtomicU64,
        }
        impl TileSink for SlowSink {
            fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
                self.writes.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(2));
                Ok(())
            }
        }

        let concurrency = 2usize;
        let buffer_size = 16usize;

        // A single top level with many small tiles keeps every worker fed while
        // the slow sink drains one tile at a time, so the buffer saturates.
        let src = gradient_raster(1024, 32);
        let plan = PyramidPlanner::new(1024, 32, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let total = plan.total_tile_count();
        assert!(
            total > (buffer_size as u64 + concurrency as u64),
            "need more tiles than the buffer can hold to force it to saturate"
        );

        let config = EngineConfig::default()
            .with_concurrency(concurrency)
            .with_buffer_size(buffer_size);
        let sink = SlowSink {
            writes: AtomicU64::new(0),
        };
        let result = generate_pyramid_observed(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert!(
            result.queue_pressure_peak > concurrency as u32,
            "queue_pressure_peak must track buffered tiles, not be capped at the worker count \
             ({concurrency}); got {}",
            result.queue_pressure_peak
        );
        // Sanity: occupancy can never exceed what the channel plus in-flight
        // senders can hold.
        assert!(
            result.queue_pressure_peak <= (buffer_size + concurrency) as u32,
            "queue_pressure_peak {} exceeds the maximum possible occupancy {}",
            result.queue_pressure_peak,
            buffer_size + concurrency
        );
    }
}
