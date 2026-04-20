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

use std::sync::Arc;

use crate::engine::{BlankTileStrategy, EngineConfig, EngineError, EngineResult, is_blank_tile};
use crate::observe::{EngineEvent, EngineObserver, MemoryTracker};
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::resize;
use crate::sink::{Tile, TileSink};
use crate::streaming::{
    RasterStripSource, StripSource, compute_strip_height, emit_full_level_tiles, emit_strip_tiles,
    fill_background_rows, find_monolithic_threshold, obtain_canvas_strip, propagate_down,
};

/// Configuration for the MapReduce streaming engine.
///
/// Controls memory budget, per-strip tile concurrency, channel backpressure,
/// and tile handling options. The budget determines how many strips can be
/// in flight simultaneously during the Map phase.
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
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 64 * 1024 * 1024,
            tile_concurrency: 0,
            buffer_size: 64,
            background_rgb: [255, 255, 255],
            blank_tile_strategy: BlankTileStrategy::Emit,
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
            failure_policy: crate::retry::FailurePolicy::default(),
            checkpoint_every: 0,
            dedupe_strategy: None,
            checkpoint_root: None,
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
/// Returns at least 1.
pub fn compute_inflight_strips(
    plan: &PyramidPlan,
    format: PixelFormat,
    strip_height: u32,
    memory_budget_bytes: u64,
) -> u32 {
    let bpp = format.bytes_per_pixel() as u64;
    let strip_cost = plan.canvas_width as u64 * strip_height as u64 * bpp;
    if strip_cost == 0 {
        return 1;
    }
    let fixed_cost = estimate_accumulator_cost(plan, format, strip_height)
        + estimate_mono_buffer_cost(plan, format);
    let available = memory_budget_bytes.saturating_sub(fixed_cost);
    let k = available / strip_cost;
    k.max(1) as u32
}

/// Estimate peak memory for the MapReduce streaming engine.
pub fn estimate_mapreduce_peak_memory(
    plan: &PyramidPlan,
    format: PixelFormat,
    strip_height: u32,
    inflight_strips: u32,
) -> u64 {
    let bpp = format.bytes_per_pixel() as u64;
    let strip_cost = plan.canvas_width as u64 * strip_height as u64 * bpp;
    let fixed_cost = estimate_accumulator_cost(plan, format, strip_height)
        + estimate_mono_buffer_cost(plan, format);
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
    let use_placeholders = config.blank_tile_strategy == BlankTileStrategy::Placeholder;

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

    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Tile, EngineError>>(config.buffer_size);

    let strip_arc = Arc::new(strip.clone());
    let plan_arc = Arc::new(plan.clone());

    let concurrency = config.tile_concurrency.min(coords.len());
    let chunk_size = coords.len().div_ceil(concurrency);

    std::thread::scope(|s| {
        for chunk in coords.chunks(chunk_size) {
            let tx = tx.clone();
            let strip_arc = Arc::clone(&strip_arc);
            let plan_arc = Arc::clone(&plan_arc);
            let chunk = chunk.to_vec();
            let bg = config.background_rgb;

            s.spawn(move || {
                for coord in chunk {
                    let result = crate::streaming::extract_tile_from_strip(
                        &strip_arc,
                        &plan_arc,
                        coord,
                        strip_canvas_y,
                        bg,
                    )
                    .map(|tile_raster| {
                        let blank = use_placeholders && is_blank_tile(&tile_raster);
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
            sink.write_tile(&tile)?;
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
/// can happen concurrently (when `tile_concurrency > 0` and batch size > 1),
/// while tile emission and propagation remain sequential to preserve the
/// deterministic strip ordering required by `propagate_down`.
///
/// # Pixel parity
///
/// Produces byte-identical output to the sequential streaming engine and
/// the monolithic engine. The reduce phase uses the same `propagate_down`
/// logic and monolithic flush.
pub fn generate_pyramid_mapreduce(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &MapReduceConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let format = source.format();
    let bpp = format.bytes_per_pixel();
    let engine_cfg = config.engine_config();

    let strip_height = compute_strip_height(plan, format, config.memory_budget_bytes)
        .unwrap_or(2 * plan.tile_size);
    let inflight = compute_inflight_strips(plan, format, strip_height, config.memory_budget_bytes);

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
        observer.on_event(EngineEvent::BatchStarted {
            batch_index: batch_idx as u32,
            strips_in_batch: batch_specs.len() as u32,
            total_batches,
        });

        let mut batch_tiles: u64 = 0;

        // MAP: render all strips in this batch (parallel when beneficial)
        let rendered_strips = if config.tile_concurrency > 0 && batch_specs.len() > 1 {
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

        // REDUCE: for each rendered strip, emit tiles and propagate (sequential)
        for (i, strip) in rendered_strips.into_iter().enumerate() {
            let &(y, _) = &batch_specs[i];
            let strip_idx = strip_index_offset + i as u32;

            observer.on_event(EngineEvent::StripRendered {
                strip_index: strip_idx,
                total_strips,
            });

            let strip_bytes = strip.data().len() as u64;
            tracker.alloc(strip_bytes);

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

            // Propagate half-strip into lower levels
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
        skipped_due_to_failure: 0,
    })
}

/// Auto-selecting entry point: monolithic if budget allows, MapReduce otherwise.
pub fn generate_pyramid_mapreduce_auto(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &MapReduceConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let mono_peak = plan.estimate_peak_memory_for_format(source.format());

    if mono_peak <= config.memory_budget_bytes {
        let engine_cfg = config.engine_config();
        crate::engine::generate_pyramid_observed(source, plan, sink, &engine_cfg, observer)
    } else {
        let strip_source = RasterStripSource::new(source);
        generate_pyramid_mapreduce(&strip_source, plan, sink, config, observer)
    }
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
        let k = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 1);
        assert!(k >= 1);
    }

    #[test]
    fn compute_inflight_strips_grows_with_budget() {
        let planner = PyramidPlanner::new(4096, 4096, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let small = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 1_000_000);
        let large = compute_inflight_strips(&plan, PixelFormat::Rgb8, 512, 100_000_000);
        assert!(large >= small);
    }

    #[test]
    fn estimate_mapreduce_peak_monotonic() {
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let est_1 = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 1);
        let est_4 = estimate_mapreduce_peak_memory(&plan, PixelFormat::Rgb8, 512, 4);
        assert!(est_4 > est_1);
    }

    #[test]
    fn mapreduce_basic_parity() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let ref_sink = MemorySink::new();
        crate::engine::generate_pyramid(&src, &plan, &ref_sink, &EngineConfig::default()).unwrap();
        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        let mr_sink = MemorySink::new();
        let config = MapReduceConfig {
            memory_budget_bytes: 100_000,
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
    fn mapreduce_auto_selects_monolithic_for_large_budget() {
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let config = MapReduceConfig {
            memory_budget_bytes: u64::MAX,
            ..MapReduceConfig::default()
        };
        let sink = MemorySink::new();
        let result =
            generate_pyramid_mapreduce_auto(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }
}
