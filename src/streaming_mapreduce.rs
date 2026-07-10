//! MapReduce streaming pyramid engine.
//!
//! A parallel variant of the streaming engine that processes multiple strips
//! concurrently while respecting a memory ceiling. The design follows a
//! MapReduce pattern:
//!
//! - **Map phase**: Render K strips in parallel (bounded by memory budget).
//! - **Reduce phase**: Emit tiles and propagate downscales sequentially.
//!
//! The existing sequential streaming engine in [`crate::streaming`] remains
//! unchanged. This module provides a parallel alternative that achieves higher
//! throughput on multi-core systems by overlapping strip rendering.
//!
//! ## Parallelism model
//!
//! 1. **Strip-level (Map)** — up to K strips rendered concurrently, where K
//!    is bounded by `floor(memory_budget / per_strip_cost)`.
//! 2. **Tile-level (within each strip)** — scoped-thread tile extraction
//!    with bounded-channel backpressure, same pattern as the monolithic engine.
//! 3. **Sequential reduce (propagation)** — half-strips feed into
//!    [`propagate_down`](crate::streaming::propagate_down) in order, since
//!    the pairing dependency requires sequential processing.
//!
//! ## Entry points
//!
//! - [`generate_pyramid_mapreduce`] — explicit MapReduce with a [`StripSource`].
//! - [`generate_pyramid_mapreduce_auto`] — auto-selects monolithic or MapReduce
//!   based on the budget vs. estimated monolithic peak memory.

use crate::engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, is_blank_for_strategy,
};
use crate::observe::{EngineEvent, EngineObserver, MemoryTracker};
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::resize;
use crate::sink::{Tile, TileSink};
#[cfg(test)]
use crate::streaming::RasterStripSource;
use crate::streaming::{
    StripSource, compute_strip_height, emit_full_level_tiles, emit_strip_tiles,
    fill_background_rows, find_monolithic_threshold, obtain_canvas_strip, propagate_down,
};

/// Configuration for the MapReduce streaming engine.
///
/// Controls memory budget, per-strip tile concurrency, channel backpressure,
/// and tile handling options. The budget determines how many strips can be
/// in flight simultaneously during the Map phase.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel)
/// (memory budget is documented at
/// [`#flag-memory-budget`](https://libviprs.org/cli/#flag-memory-budget))
#[derive(Debug, Clone)]
pub struct MapReduceConfig {
    /// Soft memory budget in bytes (covers all in-flight strips + accumulators).
    pub memory_budget_bytes: u64,
    /// Maximum worker threads for tile extraction within each strip.
    /// 0 = single-threaded tile emission (strip-level parallelism only).
    pub tile_concurrency: usize,
    /// Bounded channel capacity for tile backpressure.
    pub buffer_size: usize,
    /// Background colour for edge tile padding.
    pub background_rgb: [u8; 3],
    /// Blank tile handling strategy.
    pub blank_tile_strategy: BlankTileStrategy,
    /// Optional cooperative-cancellation token. When set, the engine polls it
    /// before each batch of strips and stops with
    /// [`EngineError::Cancelled`] once cancelled (#133).
    pub cancel: Option<crate::cancel::CancelToken>,
    /// How terminal write failures are handled. Threaded from the builder so
    /// the MapReduce engine honors the same `FailurePolicy` as the monolithic
    /// and streaming engines (issue #134). Under `RetryThenSkip` a tile whose
    /// retries are exhausted is skipped and accounted, not propagated.
    pub failure_policy: crate::retry::FailurePolicy,
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 64 * 1024 * 1024,
            tile_concurrency: 0,
            buffer_size: 64,
            background_rgb: [255, 255, 255],
            blank_tile_strategy: BlankTileStrategy::Emit,
            cancel: None,
            failure_policy: crate::retry::FailurePolicy::default(),
        }
    }
}

impl MapReduceConfig {
    fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            concurrency: self.tile_concurrency,
            buffer_size: self.buffer_size,
            background_rgb: self.background_rgb,
            blank_tile_strategy: self.blank_tile_strategy,
            failure_policy: self.failure_policy.clone(),
            checkpoint_every: 0,
            dedupe_strategy: None,
            checkpoint_root: None,
            source_content_hash: None,
            cancel: self.cancel.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Estimate the accumulator cost across all levels above the monolithic threshold.
fn estimate_accumulator_cost(plan: &PyramidPlan, format: PixelFormat, strip_height: u32) -> u64 {
    let bpp = format.bytes_per_pixel() as u64;
    let threshold = find_monolithic_threshold(plan, format, strip_height);
    let mut total: u64 = 0;
    let mut w = plan.canvas_width as u64;
    let mut h = strip_height as u64;

    for level_idx in (0..plan.levels.len()).rev() {
        if level_idx <= threshold {
            break;
        }
        total += w * h.div_ceil(2) * bpp;
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    total
}

/// Estimate the cost of the largest monolithic level buffer.
fn estimate_mono_buffer_cost(plan: &PyramidPlan, format: PixelFormat) -> u64 {
    let bpp = format.bytes_per_pixel() as u64;
    let strip_budget = plan.canvas_width as u64 * plan.tile_size as u64 * 2 * bpp;
    for level_idx in (0..plan.levels.len()).rev() {
        let (lw, lh) = if plan.layout == Layout::Google {
            plan.canvas_size_at_level(plan.levels[level_idx].level)
        } else {
            (plan.levels[level_idx].width, plan.levels[level_idx].height)
        };
        let level_bytes = lw as u64 * lh as u64 * bpp;
        if level_bytes <= strip_budget {
            return level_bytes;
        }
    }
    0
}

/// Compute the number of in-flight strips that fit within a memory budget.
///
/// The budget must accommodate *every* live buffer the Map phase holds at
/// once: all `K` decoded strips, the per-level accumulators and monolithic
/// buffer (`fixed_cost`), and the bounded tile channel backlog
/// (`channel_bytes`). Charging the channel and all `K` strips — rather than a
/// single strip — is what keeps the real RSS within the configured ceiling
/// (issue #103); previously only the accumulators were charged, so `K` strips
/// plus the channel could push the peak toward 2× budget.
///
/// Returns at least 1. When even a single strip does not fit, callers rely on
/// the pre-flight [`EngineError::BudgetExceeded`] check in
/// [`generate_pyramid_mapreduce`] to reject the budget up front.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel)
pub fn compute_inflight_strips(
    plan: &PyramidPlan,
    format: PixelFormat,
    strip_height: u32,
    channel_bytes: u64,
    memory_budget_bytes: u64,
) -> u32 {
    let bpp = format.bytes_per_pixel() as u64;
    let strip_cost = plan.canvas_width as u64 * strip_height as u64 * bpp;
    if strip_cost == 0 {
        return 1;
    }
    let fixed_cost = estimate_accumulator_cost(plan, format, strip_height)
        + estimate_mono_buffer_cost(plan, format)
        + channel_bytes;
    let available = memory_budget_bytes.saturating_sub(fixed_cost);
    let k = available / strip_cost;
    // The image only spans this many strips (`strip_cost != 0` above implies
    // `strip_height >= 1`, so the `div_ceil` is safe). There is no reason to
    // admit more in-flight strips than actually exist, and using it as the
    // upper clamp bound also keeps the return within `u32` range.
    let total_strips = (plan.canvas_height.div_ceil(strip_height) as u64).max(1);
    // Clamp *before* the `u32` cast: `k` is a `u64`, so a huge budget can yield
    // a value at or above `2^32`. `k as u32` would then truncate — any nonzero
    // multiple of `2^32` wraps to 0 — and `strip_specs.chunks(0)` panics with
    // "chunk size must be non-zero". Clamping first guarantees `1..=total_strips`
    // (issue #106).
    k.clamp(1, total_strips) as u32
}

/// Estimate peak memory for the MapReduce streaming engine.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel)
///
/// `channel_bytes` charges the bounded tile channel backlog (up to
/// `buffer_size + concurrency` decoded tiles held in flight during parallel
/// tile emission). It is included so the estimate reflects every live buffer,
/// not just the strips and accumulators (issue #103).
pub fn estimate_mapreduce_peak_memory(
    plan: &PyramidPlan,
    format: PixelFormat,
    strip_height: u32,
    inflight_strips: u32,
    channel_bytes: u64,
) -> u64 {
    let bpp = format.bytes_per_pixel() as u64;
    let strip_cost = plan.canvas_width as u64 * strip_height as u64 * bpp;
    let fixed_cost = estimate_accumulator_cost(plan, format, strip_height)
        + estimate_mono_buffer_cost(plan, format)
        + channel_bytes;
    let peak = inflight_strips as u64 * strip_cost + fixed_cost;
    peak + peak / 10
}

// ---------------------------------------------------------------------------
// Parallel tile emission within a strip
// ---------------------------------------------------------------------------

/// Emit tiles from a strip using parallel worker threads.
fn emit_strip_tiles_parallel(
    strip: &Raster,
    plan: &PyramidPlan,
    level: u32,
    strip_canvas_y: u32,
    sink: &dyn TileSink,
    config: &MapReduceConfig,
    observer: &dyn EngineObserver,
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let ts = plan.tile_size;
    // Honor the full `BlankTileStrategy` — including
    // `PlaceholderWithTolerance` — via the same predicate the sequential and
    // monolithic paths use. An exact-equality test against `Placeholder`
    // silently dropped the tolerance variant whenever `tile_concurrency > 0`,
    // flipping blank flags and `tiles_skipped` between concurrency settings and
    // breaking the byte-identical-output guarantee (issue #107).
    let blank_strategy = config.blank_tile_strategy;

    let first_row = strip_canvas_y / ts;
    let last_row = (strip_canvas_y + strip.height())
        .div_ceil(ts)
        .min(level_plan.rows);

    if first_row >= last_row {
        return Ok((0, 0));
    }

    let coords: Vec<TileCoord> = (first_row..last_row)
        .flat_map(|row| (0..level_plan.cols).map(move |col| TileCoord::new(level, col, row)))
        .collect();

    if coords.is_empty() {
        return Ok((0, 0));
    }

    // Routed through `crate::sync_queue` (loom-modellable under `--cfg loom`)
    // rather than `std::sync::mpsc::sync_channel` directly, so the shipped
    // backpressure/teardown protocol is what the loom suite exercises.
    let (tx, rx) = crate::sync_queue::bounded::<Result<Tile, EngineError>>(config.buffer_size);

    // Workers run under `std::thread::scope` and therefore cannot outlive
    // this frame, so they borrow `strip` and `plan` directly. The previous
    // `Arc::new(strip.clone())` held a full extra copy of the strip alive for
    // the whole emission — memory the budget estimate never accounted for.
    let concurrency = config.tile_concurrency.min(coords.len());
    let chunk_size = coords.len().div_ceil(concurrency);

    std::thread::scope(|s| {
        for chunk in coords.chunks(chunk_size) {
            let tx = tx.clone();
            let chunk = chunk.to_vec();
            let bg = config.background_rgb;

            s.spawn(move || {
                for coord in chunk {
                    let result = crate::streaming::extract_tile_from_strip(
                        strip,
                        plan,
                        coord,
                        strip_canvas_y,
                        bg,
                    )
                    .map(|tile_raster| {
                        let blank = is_blank_for_strategy(&tile_raster, blank_strategy);
                        Tile {
                            coord,
                            raster: tile_raster,
                            blank,
                        }
                    })
                    .map_err(EngineError::from);
                    if tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut count = 0u64;
        let mut skipped = 0u64;
        for result in rx {
            let tile = result?;
            let coord = tile.coord;
            if tile.blank {
                skipped += 1;
            }
            if let Err(e) = sink.write_tile(&tile) {
                // Honor the configured FailurePolicy so the MapReduce engine
                // matches the monolithic and streaming engines (issue #134). A
                // write whose retry backoff was interrupted by a cancellation
                // must surface as Cancelled, not be swallowed by RetryThenSkip.
                if let Some(token) = &config.cancel {
                    if token.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                }
                match &config.failure_policy {
                    crate::retry::FailurePolicy::RetryThenSkip(_) => {
                        sink.note_sink_skipped();
                        observer.on_event(EngineEvent::TileCompleted { coord });
                        continue;
                    }
                    _ => return Err(crate::engine::promote_sink_error(e)),
                }
            }
            observer.on_event(EngineEvent::TileCompleted { coord });
            count += 1;
        }
        Ok((count, skipped))
    })
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Generate a tile pyramid using the MapReduce streaming engine.
///
/// Processes strips in parallel batches. Within each batch, strip rendering
/// can happen concurrently (when `tile_concurrency > 0`, batch size > 1, and
/// the source opts in via [`StripSource::permits_concurrent_strips`]); sources
/// that require the default sequential, increasing-`y` access pattern are
/// rendered one strip at a time. Tile emission and propagation remain
/// sequential regardless, to preserve the deterministic strip ordering required
/// by `propagate_down`.
///
/// # Pixel parity
///
/// Produces byte-identical output to the sequential streaming engine and
/// the monolithic engine. The reduce phase uses the same `propagate_down`
/// logic and monolithic flush.
pub(crate) fn generate_pyramid_mapreduce(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &MapReduceConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let format = source.format();
    let bpp = format.bytes_per_pixel();
    let engine_cfg = config.engine_config();

    // Pre-flight (parity with the sequential engine, `streaming.rs`): the
    // worst-case strip is one minimum aligned unit (2 × tile_size rows) at
    // canvas width. If the budget cannot fit it, the engine cannot honour the
    // budget no matter how it slices the work, so reject it up front instead
    // of silently proceeding with an over-budget minimum strip (issue #103).
    let min_strip_height = 2 * plan.tile_size;
    let worst_case_strip_bytes = plan.canvas_width as u64 * min_strip_height as u64 * bpp as u64;
    if worst_case_strip_bytes > config.memory_budget_bytes {
        return Err(EngineError::BudgetExceeded {
            strip_bytes: worst_case_strip_bytes,
            budget_bytes: config.memory_budget_bytes,
        });
    }

    let strip_height =
        compute_strip_height(plan, format, config.memory_budget_bytes).unwrap_or(min_strip_height);

    // The parallel tile-emission path (`emit_strip_tiles_parallel`) holds up to
    // `buffer_size` decoded tiles in its bounded channel, plus one per worker
    // in flight. Charge that backlog against the budget so the in-flight strip
    // count leaves room for it and the peak estimate stays honest (issue #103).
    let channel_bytes = if config.tile_concurrency > 0 {
        let tile_bytes = plan.tile_size as u64 * plan.tile_size as u64 * bpp as u64;
        (config.buffer_size as u64 + config.tile_concurrency as u64) * tile_bytes
    } else {
        0
    };
    let inflight = compute_inflight_strips(
        plan,
        format,
        strip_height,
        channel_bytes,
        config.memory_budget_bytes,
    );

    let ch = plan.canvas_height;
    let top_level = plan.levels.len() - 1;
    let tracker = MemoryTracker::new();

    let mut tiles_produced: u64 = 0;
    let mut tiles_skipped: u64 = 0;

    let mut accumulators: Vec<Option<Raster>> = vec![None; plan.levels.len()];
    let monolithic_threshold = find_monolithic_threshold(plan, format, strip_height);
    let mut mono_accumulators: Vec<Vec<u8>> = plan.levels.iter().map(|_| Vec::new()).collect();

    // Emit LevelStarted for all levels upfront
    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];
        observer.on_event(EngineEvent::LevelStarted {
            level: level.level,
            width: level.width,
            height: level.height,
            tile_count: level.tile_count(),
        });
    }

    // Pre-compute strip offsets
    let total_strips = ch.div_ceil(strip_height);
    let strip_specs: Vec<(u32, u32)> = (0..total_strips)
        .map(|i| {
            let y = i * strip_height;
            let h = strip_height.min(ch - y);
            (y, h)
        })
        .collect();

    let total_batches = strip_specs.len().div_ceil(inflight as usize) as u32;

    // ===================================================================
    // Process in batches
    // ===================================================================
    let mut strip_index_offset: u32 = 0;
    for (batch_idx, batch_specs) in strip_specs.chunks(inflight as usize).enumerate() {
        // Cooperative cancellation: stop cleanly at the batch boundary before
        // rendering (and downscaling) another batch of strips.
        engine_cfg.check_cancelled()?;
        observer.on_event(EngineEvent::BatchStarted {
            batch_index: batch_idx as u32,
            strips_in_batch: batch_specs.len() as u32,
            total_batches,
        });

        let mut batch_tiles: u64 = 0;

        // MAP: render all strips in this batch (parallel when beneficial).
        //
        // Concurrent rendering issues `render_strip` from several threads at
        // once and therefore out of `y` order. That only honours the
        // `StripSource` contract for sources that opt in via
        // `permits_concurrent_strips`; a default (cursor-based) source is
        // promised sequential, strictly-increasing-`y` access, so it must take
        // the sequential branch even when concurrency is enabled (issue #105).
        let rendered_strips = if config.tile_concurrency > 0
            && batch_specs.len() > 1
            && source.permits_concurrent_strips()
        {
            let mut strips: Vec<Option<Raster>> = vec![None; batch_specs.len()];
            std::thread::scope(|s| -> Result<(), EngineError> {
                let mut handles = Vec::with_capacity(batch_specs.len());
                for &(y, sh) in batch_specs {
                    let engine_cfg = &engine_cfg;
                    handles.push(s.spawn(move || -> Result<Raster, EngineError> {
                        obtain_canvas_strip(source, plan, y, sh, engine_cfg)
                    }));
                }
                for (i, handle) in handles.into_iter().enumerate() {
                    let strip = handle.join().map_err(|_| EngineError::WorkerPanic)??;
                    strips[i] = Some(strip);
                }
                Ok(())
            })?;
            strips.into_iter().map(|s| s.unwrap()).collect::<Vec<_>>()
        } else {
            batch_specs
                .iter()
                .map(|&(y, sh)| obtain_canvas_strip(source, plan, y, sh, &engine_cfg))
                .collect::<Result<Vec<_>, _>>()?
        };

        // All strips in this batch are now materialised simultaneously (the
        // Map phase renders them concurrently, or sequentially into one Vec).
        // Charge every one against the tracker here so the reported peak
        // reflects the whole in-flight batch, not just the single strip being
        // reduced. Each strip is released again after it is downscaled below,
        // so the tracked total falls back as the batch drains (issue #103).
        for strip in &rendered_strips {
            tracker.alloc(strip.data().len() as u64);
        }

        // REDUCE: for each rendered strip, emit tiles and propagate (sequential)
        for (i, strip) in rendered_strips.into_iter().enumerate() {
            let &(y, _) = &batch_specs[i];
            let strip_idx = strip_index_offset + i as u32;

            observer.on_event(EngineEvent::StripRendered {
                strip_index: strip_idx,
                total_strips,
            });

            let strip_bytes = strip.data().len() as u64;

            // Emit tiles at the top level
            let (tp, ts_skip) = if config.tile_concurrency > 0 {
                emit_strip_tiles_parallel(
                    &strip,
                    plan,
                    top_level as u32,
                    y,
                    sink,
                    config,
                    observer,
                )?
            } else {
                emit_strip_tiles(
                    &strip,
                    plan,
                    top_level as u32,
                    y,
                    sink,
                    &engine_cfg,
                    observer,
                )?
            };
            tiles_produced += tp;
            tiles_skipped += ts_skip;
            batch_tiles += tp;

            // Downscale for reduce propagation
            let half = resize::downscale_half(&strip)?;
            tracker.dealloc(strip_bytes);
            let half_bytes = half.data().len() as u64;
            tracker.alloc(half_bytes);

            // Propagate half-strip into lower levels.
            //
            // A single-level plan (top_level == 0) has no level below the
            // top: the top-level tiles were already emitted above, so there
            // is nothing left to reduce. Guard the recursion to avoid the
            // `top_level - 1` underflow (debug panic / release usize::MAX ->
            // out-of-bounds index into `accumulators`).
            if top_level > 0 {
                propagate_down(
                    half,
                    top_level - 1,
                    y / 2,
                    &mut accumulators,
                    &mut mono_accumulators,
                    monolithic_threshold,
                    plan,
                    sink,
                    &engine_cfg,
                    observer,
                    &tracker,
                    &mut tiles_produced,
                    &mut tiles_skipped,
                )?;
            }
        }

        strip_index_offset += batch_specs.len() as u32;

        observer.on_event(EngineEvent::BatchCompleted {
            batch_index: batch_idx as u32,
            tiles_produced: batch_tiles,
        });
    }

    // ===================================================================
    // Phase 2: Flush unpaired strip accumulators
    // ===================================================================
    for level_idx in (monolithic_threshold + 1..plan.levels.len()).rev() {
        if let Some(leftover) = accumulators[level_idx].take() {
            let (_, lh) = if plan.layout == Layout::Google {
                plan.canvas_size_at_level(plan.levels[level_idx].level)
            } else {
                (plan.levels[level_idx].width, plan.levels[level_idx].height)
            };
            let leftover_y = lh.saturating_sub(leftover.height());

            let (tp, ts_skip) = emit_strip_tiles(
                &leftover,
                plan,
                level_idx as u32,
                leftover_y,
                sink,
                &engine_cfg,
                observer,
            )?;
            tiles_produced += tp;
            tiles_skipped += ts_skip;

            if level_idx > 0 {
                let further_half = resize::downscale_half(&leftover)?;
                propagate_down(
                    further_half,
                    level_idx - 1,
                    leftover_y / 2,
                    &mut accumulators,
                    &mut mono_accumulators,
                    monolithic_threshold,
                    plan,
                    sink,
                    &engine_cfg,
                    observer,
                    &tracker,
                    &mut tiles_produced,
                    &mut tiles_skipped,
                )?;
            }
        }
    }

    // ===================================================================
    // Phase 3: Monolithic flush — assemble and emit small levels
    // ===================================================================
    {
        let top_mono = monolithic_threshold.min(plan.levels.len() - 1);
        let mut prev_raster: Option<Raster> = None;

        for level_idx in (0..=top_mono).rev() {
            let level = &plan.levels[level_idx];
            let (lw, lh) = if plan.layout == Layout::Google {
                plan.canvas_size_at_level(level.level)
            } else {
                (level.width, level.height)
            };
            if lw == 0 || lh == 0 {
                continue;
            }

            let raster = if let Some(prev) = prev_raster.take() {
                resize::downscale_half(&prev)?
            } else {
                let mut acc_data = std::mem::take(&mut mono_accumulators[level_idx]);
                if acc_data.is_empty() {
                    continue;
                }
                let expected = lw as usize * lh as usize * bpp;

                if acc_data.len() > expected {
                    acc_data.truncate(expected);
                }
                if acc_data.len() < expected {
                    let filled_rows = acc_data.len() / (lw as usize * bpp);
                    acc_data.resize(expected, 0);
                    fill_background_rows(
                        &mut acc_data,
                        filled_rows,
                        lw,
                        lh,
                        bpp,
                        engine_cfg.background_rgb,
                    );
                }
                Raster::new(lw, lh, format, acc_data)?
            };

            let (tp, ts_skip) = emit_full_level_tiles(
                &raster,
                plan,
                level_idx as u32,
                sink,
                &engine_cfg,
                observer,
            )?;
            tiles_produced += tp;
            tiles_skipped += ts_skip;

            prev_raster = Some(raster);
        }
    }

    // Emit LevelCompleted for all levels
    for level in &plan.levels {
        observer.on_event(EngineEvent::LevelCompleted {
            level: level.level,
            tiles_produced: level.tile_count(),
        });
    }

    sink.finish()?;

    observer.on_event(EngineEvent::Finished {
        total_tiles: tiles_produced,
        levels: plan.levels.len() as u32,
    });

    Ok(EngineResult {
        tiles_produced,
        tiles_skipped,
        levels_processed: plan.levels.len() as u32,
        peak_memory_bytes: tracker.peak_bytes(),
        bytes_read: 0,
        bytes_written: 0,
        retry_count: 0,
        queue_pressure_peak: 0,
        duration: std::time::Duration::ZERO,
        stage_durations: crate::engine::StageDurations::default(),
        skipped_due_to_failure: sink.sink_skipped_due_to_failure(),
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::NoopObserver;
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

    #[test]
    fn compute_inflight_strips_at_least_one() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let k = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 0, 1);
        assert!(k >= 1);
    }

    #[test]
    fn compute_inflight_strips_huge_budget_no_chunks_zero_panic() {
        // A caller-supplied budget that is a nonzero multiple of 2^32 (or the
        // saturating maximum) used to make `k` land on a multiple of 2^32 and
        // truncate to 0 in the `u32` cast, so `strip_specs.chunks(inflight)`
        // panicked with "chunk size must be non-zero" (issue #106). The count
        // must instead stay in `1..=total_strips` for *any* budget.
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let strip_height = 512u32;
        // Exact number of strips the batch loop will chunk over.
        let total_strips = plan.canvas_height.div_ceil(strip_height);

        for budget in [1u64 << 32, 2u64 << 32, u64::MAX] {
            let inflight =
                compute_inflight_strips(&plan, PixelFormat::Rgb8, strip_height, 0, budget);
            assert!(
                inflight >= 1,
                "budget {budget}: inflight must be >= 1 to avoid chunks(0), got {inflight}",
            );
            assert!(
                inflight <= total_strips,
                "budget {budget}: inflight {inflight} must not exceed total strips {total_strips}",
            );

            // Prove the returned count is a legal `chunks` argument: the call
            // below panics if `inflight` ever truncated to 0.
            let strip_specs: Vec<u32> = (0..total_strips).collect();
            let batches = strip_specs.chunks(inflight as usize).count();
            assert!(
                batches >= 1,
                "budget {budget}: chunking produced no batches"
            );
        }
    }

    #[test]
    fn compute_inflight_strips_grows_with_budget() {
        let planner = PyramidPlanner::new(4096, 4096, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let small = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 0, 1_000_000);
        let large = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 0, 100_000_000);
        assert!(large >= small);
    }

    #[test]
    fn estimate_mapreduce_peak_monotonic() {
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let est_1 = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 1, 0);
        let est_4 = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 4, 0);
        assert!(est_4 > est_1);
    }

    #[test]
    fn mapreduce_basic_parity() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let ref_sink = MemorySink::new();
        crate::engine::generate_pyramid_observed(
            &src,
            &plan,
            &ref_sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .unwrap();
        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        let mr_sink = MemorySink::new();
        // Budget fits the minimum aligned strip (256×256×3 = 196_608 bytes) so
        // the run is admitted; it still exercises the streaming/reduce path.
        let config = MapReduceConfig {
            memory_budget_bytes: 1_000_000,
            ..MapReduceConfig::default()
        };
        let strip_src = RasterStripSource::new(&src);
        generate_pyramid_mapreduce(&strip_src, &plan, &mr_sink, &config, &NoopObserver).unwrap();
        let mut mr_tiles = mr_sink.tiles();
        mr_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        assert_eq!(ref_tiles.len(), mr_tiles.len());
        for (r, m) in ref_tiles.iter().zip(mr_tiles.iter()) {
            assert_eq!(r.coord, m.coord);
            assert_eq!(r.data, m.data, "tile data diverged at {:?}", m.coord);
        }
    }

    #[test]
    fn mapreduce_rejects_budget_too_small_for_strip() {
        // A 512×512 RGB8 image with tile_size 256 has a minimum aligned strip
        // of 2×256 = 512 rows → 512×512×3 = 786_432 bytes. A budget below that
        // cannot honour the memory ceiling, so the engine must reject it up
        // front with `BudgetExceeded` — parity with the sequential streaming
        // engine — instead of silently proceeding with an over-budget minimum
        // strip.
        let src = gradient_raster(512, 512);
        let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let config = MapReduceConfig {
            memory_budget_bytes: 100_000,
            ..MapReduceConfig::default()
        };
        let sink = MemorySink::new();
        let err = generate_pyramid_mapreduce(
            &RasterStripSource::new(&src),
            &plan,
            &sink,
            &config,
            &NoopObserver,
        )
        .unwrap_err();
        assert!(
            matches!(err, EngineError::BudgetExceeded { .. }),
            "expected BudgetExceeded, got {err:?}",
        );
    }

    #[test]
    fn estimate_accounts_for_channel_backlog() {
        // The peak estimate must grow by (at least) the charged channel
        // backlog — previously the channel-held tiles were invisible to the
        // estimate, so the reported peak understated real RSS (issue #103).
        let plan = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let without = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 2, 0);
        let with = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 2, 4_000_000);
        assert!(with > without);
        assert!(with - without >= 4_000_000);
    }

    #[test]
    fn inflight_strips_shrink_when_channel_charged() {
        // Charging the channel backlog leaves less room for in-flight strips,
        // so K must be non-increasing as the channel charge grows — this is
        // what keeps K strips + channel within the ceiling (issue #103).
        let plan = PyramidPlanner::new(8192, 8192, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let budget = 200_000_000u64;
        let k_no_channel = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 0, budget);
        let k_big_channel =
            compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 100_000_000, budget);
        assert!(
            k_no_channel > 1,
            "test needs a case where several strips fit"
        );
        assert!(k_big_channel <= k_no_channel);
        assert!(k_big_channel >= 1);
    }

    #[test]
    fn mapreduce_real_peak_within_budget() {
        // End-to-end: with the in-flight strips now charged (both by the
        // budgeter and the memory tracker), a constrained run keeps its real
        // tracked peak within the configured budget instead of spiking toward
        // 2× budget mid-batch (issue #103).
        let src = gradient_raster(1024, 1024);
        let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let budget = 12_000_000u64;
        let sink = MemorySink::new();
        let config = MapReduceConfig {
            memory_budget_bytes: budget,
            ..MapReduceConfig::default()
        };
        let result = generate_pyramid_mapreduce(
            &RasterStripSource::new(&src),
            &plan,
            &sink,
            &config,
            &NoopObserver,
        )
        .unwrap();
        assert!(
            result.peak_memory_bytes > 0,
            "tracker should record the in-flight strips"
        );
        assert!(
            result.peak_memory_bytes <= budget,
            "real peak {} exceeded budget {budget}",
            result.peak_memory_bytes,
        );
    }

    /// A raster that is uniform within a small channel delta but *not* exactly
    /// uniform: every pixel is the base colour with a per-pixel wobble of at
    /// most `delta`. `PlaceholderWithTolerance { max_channel_delta >= delta }`
    /// treats each full tile as blank; exact `Placeholder` does not.
    fn near_uniform_raster(w: u32, h: u32, base: u8, delta: u8) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h {
            for x in 0..w {
                let off = (y as usize * w as usize + x as usize) * bpp;
                // Deterministic wobble in `0..=delta`, so the tile is blank
                // within tolerance but every pixel is not identical.
                let wobble = ((x + y) % (delta as u32 + 1)) as u8;
                data[off] = base + wobble;
                data[off + 1] = base + wobble;
                data[off + 2] = base + wobble;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    #[test]
    fn placeholder_with_tolerance_identical_across_tile_concurrency() {
        // Regression for issue #107: the parallel tile-emission path tested
        // `blank_tile_strategy == Placeholder` with exact equality, so
        // `PlaceholderWithTolerance` silently degraded to "never blank" the
        // moment `tile_concurrency > 0` — flipping blank flags and
        // `tiles_skipped` between concurrency settings and breaking the
        // byte-identical-output guarantee. The base level is emitted through
        // that path, so its near-uniform tiles are where the two runs diverge.
        let src = near_uniform_raster(256, 256, 100, 2);
        let plan = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let run = |tile_concurrency: usize| {
            let sink = MemorySink::new();
            let config = MapReduceConfig {
                memory_budget_bytes: 1_000_000,
                tile_concurrency,
                blank_tile_strategy: BlankTileStrategy::PlaceholderWithTolerance {
                    max_channel_delta: 2,
                },
                ..MapReduceConfig::default()
            };
            let result = generate_pyramid_mapreduce(
                &RasterStripSource::new(&src),
                &plan,
                &sink,
                &config,
                &NoopObserver,
            )
            .unwrap();
            let mut tiles = sink.tiles();
            tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));
            (result.tiles_skipped, tiles)
        };

        let (seq_skipped, seq_tiles) = run(0);
        let (par_skipped, par_tiles) = run(4);

        // The tolerance must produce *some* blank tiles, otherwise the test
        // would pass vacuously (exact-equality bug and correct code agree on
        // "never blank").
        assert!(
            seq_skipped > 0,
            "sequential path must skip near-uniform tiles under the tolerance"
        );
        assert_eq!(
            seq_skipped, par_skipped,
            "tiles_skipped must be identical at tile_concurrency 0 and >= 2 \
             (parallel path ignored PlaceholderWithTolerance, issue #107)"
        );

        assert_eq!(seq_tiles.len(), par_tiles.len());
        for (s, p) in seq_tiles.iter().zip(par_tiles.iter()) {
            assert_eq!(s.coord, p.coord);
            assert_eq!(
                s.data, p.data,
                "tile bytes diverged between concurrency settings at {:?}",
                p.coord
            );
        }
    }

    #[test]
    fn mapreduce_auto_selects_monolithic_for_large_budget() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let config = MapReduceConfig {
            memory_budget_bytes: u64::MAX,
            ..MapReduceConfig::default()
        };
        let sink = MemorySink::new();
        let result = generate_pyramid_mapreduce(
            &RasterStripSource::new(&src),
            &plan,
            &sink,
            &config,
            &NoopObserver,
        )
        .unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }
}

// ---------------------------------------------------------------------------
// Single-level plan underflow guard (issue #102)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod single_level_tests {
    use super::*;
    use crate::observe::NoopObserver;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlan, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::MemorySink;

    fn gradient(w: u32, h: u32) -> Raster {
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

    fn run(plan: &PyramidPlan, src: &Raster) -> MemorySink {
        let sink = MemorySink::new();
        // Modest budget that still fits a minimum aligned strip for these
        // single-tile plans, so the strip loop and propagate_down run without
        // tripping the pre-flight budget check.
        let config = MapReduceConfig {
            memory_budget_bytes: 1_000_000,
            ..MapReduceConfig::default()
        };
        generate_pyramid_mapreduce(
            &RasterStripSource::new(src),
            plan,
            &sink,
            &config,
            &NoopObserver,
        )
        .expect("single-level plan must run to completion without underflow panic");
        sink
    }

    #[test]
    fn google_single_tile_image_runs_to_completion() {
        let src = gradient(256, 256);
        let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::Google)
            .unwrap()
            .plan();
        assert_eq!(plan.levels.len(), 1, "expected a single-level plan");
        let sink = run(&plan, &src);
        assert!(
            !sink.tiles().is_empty(),
            "single-level plan should still emit its top-level tile(s)"
        );
    }

    #[test]
    fn deepzoom_one_by_one_source_runs_to_completion() {
        let src = gradient(1, 1);
        let plan = PyramidPlanner::new(1, 1, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert_eq!(plan.levels.len(), 1, "expected a single-level plan");
        let sink = run(&plan, &src);
        assert!(
            !sink.tiles().is_empty(),
            "single-level plan should still emit its top-level tile(s)"
        );
    }
}
