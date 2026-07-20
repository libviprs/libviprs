//! MapReduce hot-cache pyramid engine (issue #67).
//!
//! A local-only variant of the MapReduce engine that trades memory for
//! throughput: the MAP/REDUCE phases run unchanged, but every produced tile
//! is retained in an in-memory hot cache instead of being written to the
//! caller's sink as it arrives. When the whole pyramid has been rendered, the
//! cache is drained to the user sink in one batched pass, in canonical
//! `(level, row, col)` order, followed by a single `finish()`.
//!
//! ## Why
//!
//! Sinks whose per-write cost dominates (network stores, archive writers,
//! sinks holding coarse locks) benefit from receiving the whole pyramid as
//! one ordered burst at the end of the run instead of interleaved with
//! rendering. The canonical flush order also makes the *write order* to the
//! sink deterministic even at `tile_concurrency > 0`, which the plain
//! MapReduce engine deliberately does not promise (it writes in
//! channel-arrival order).
//!
//! ## Determinism
//!
//! Byte-identical to the streaming and MapReduce engines for the same inputs:
//! the tile bytes are produced by the exact same MAP/REDUCE code (this module
//! reuses [`generate_pyramid_mapreduce`] wholesale against the cache), and the
//! flush replays them without transformation.
//!
//! ## Memory
//!
//! The configured memory budget bounds the MAP/REDUCE working set exactly as
//! it does for [`EngineKind::MapReduce`](crate::EngineKind::MapReduce),
//! including the pre-flight [`EngineError::BudgetExceeded`] rejection. The
//! hot cache itself is *additional*, deliberately unbounded residency (the
//! whole pyramid's raster bytes stay in RAM until the flush): that is the
//! opt-in tradeoff this engine exists for, and it is why
//! [`EngineKind::Auto`](crate::EngineKind::Auto) never selects it. The cached
//! bytes are charged into [`EngineResult::peak_memory_bytes`] so the reported
//! peak reflects the real residency.
//!
//! ## Observer semantics
//!
//! `TileCompleted` fires when a tile is *produced* (lands in the hot cache),
//! not when it reaches the caller's sink; the sink receives everything during
//! the final flush. `Finished` keeps its "comes last" contract: the inner
//! run's `Finished` is withheld and re-emitted only after the flush and the
//! user sink's `finish()` have completed.
//!
//! ## Entry point
//!
//! Reached through the fluent [`EngineBuilder`](crate::EngineBuilder) via
//! [`EngineKind::MapReduceHotCache`](crate::EngineKind::MapReduceHotCache).
//! Explicit opt-in only.

use std::sync::Mutex;

use crate::engine::{EngineError, EngineResult, promote_sink_error};
use crate::observe::{EngineEvent, EngineObserver};
use crate::planner::PyramidPlan;
use crate::sink::{SinkError, Tile, TileSink};
use crate::streaming::StripSource;
use crate::streaming_mapreduce::{MapReduceConfig, WorkExecutor, generate_pyramid_mapreduce};

/// Terminal in-memory sink holding every produced tile until the final flush.
///
/// Unlike [`MemorySink`](crate::sink::MemorySink) it preserves the full
/// [`Tile`] (including the `blank` flag), so the flush replays exactly what
/// the engine produced and placeholder-writing sinks behave identically to a
/// direct run.
#[derive(Debug, Default)]
struct HotCacheSink {
    tiles: Mutex<Vec<Tile>>,
}

impl HotCacheSink {
    fn new() -> Self {
        Self::default()
    }

    fn into_tiles(self) -> Vec<Tile> {
        self.tiles
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl TileSink for HotCacheSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        crate::poison::recover(&self.tiles).push(Tile {
            coord: tile.coord,
            raster: tile.raster.clone(),
            blank: tile.blank,
        });
        Ok(())
    }
}

/// Observer adapter that withholds the inner run's `Finished` event.
///
/// The hot-cache driver runs the MapReduce engine to completion against the
/// cache and only then flushes the user sink, so the `Finished` event the
/// inner run emits would reach observers *before* any tile lands in the real
/// sink. The driver suppresses it here and re-emits `Finished` itself once
/// the flush (and the user sink's `finish()`) has completed, preserving the
/// documented "Finished comes last" ordering. Every other event forwards
/// unchanged.
struct DeferFinished<'a> {
    inner: &'a dyn EngineObserver,
}

impl EngineObserver for DeferFinished<'_> {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::Finished { .. }) {
            return;
        }
        self.inner.on_event(event);
    }

    fn on_extensions(&self, extensions: &crate::extensions::Extensions) {
        self.inner.on_extensions(extensions);
    }
}

/// Generate a tile pyramid with the MapReduce hot-cache engine.
///
/// Runs [`generate_pyramid_mapreduce`] unchanged against an in-memory cache,
/// then drains the cache to `sink` in canonical `(level, row, col)` order and
/// calls `sink.finish()` once. See the [module docs](self) for the contract.
///
/// # Pixel parity
///
/// Byte-identical to the streaming and MapReduce engines: the tiles are
/// produced by the same code and replayed verbatim.
pub(crate) fn generate_pyramid_mapreduce_hot_cache(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &MapReduceConfig,
    observer: &dyn EngineObserver,
    executor: &dyn WorkExecutor,
) -> Result<EngineResult, EngineError> {
    // Phase 1: render the whole pyramid into the hot cache. The inner run's
    // `Finished` is withheld; it is re-emitted after the flush below.
    let cache = HotCacheSink::new();
    let defer = DeferFinished { inner: observer };
    let mut result = generate_pyramid_mapreduce(source, plan, &cache, config, &defer, executor)?;

    // Phase 2: single-pass batched flush, canonical order. Sorting here is
    // what makes the user-visible *write order* deterministic even when the
    // cache was filled in channel-arrival order at `tile_concurrency > 0`.
    let mut tiles = cache.into_tiles();
    tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));
    let cached_bytes: u64 = tiles.iter().map(|t| t.raster.data().len() as u64).sum();

    let mut flush_skipped = 0u64;
    for tile in &tiles {
        // Cooperative cancellation between writes, matching the polling
        // cadence of the other engines' write loops.
        if let Some(token) = &config.cancel {
            if token.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
        }
        if let Err(e) = sink.write_tile(tile) {
            // Same terminal-failure handling as the MapReduce emit paths: a
            // write whose retry backoff was interrupted by cancellation
            // surfaces as Cancelled, RetryThenSkip skips and accounts, and
            // every other policy aborts the run.
            if let Some(token) = &config.cancel {
                if token.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
            }
            match &config.failure_policy {
                crate::retry::FailurePolicy::RetryThenSkip(_) => {
                    sink.note_sink_skipped();
                    flush_skipped += 1;
                    continue;
                }
                _ => return Err(promote_sink_error(e)),
            }
        }
    }

    sink.finish()?;

    // A tile dropped at flush time produced no output, so it must not be
    // counted as produced (parity with the direct engines, where a skipped
    // write never increments the produced count).
    result.tiles_produced = result.tiles_produced.saturating_sub(flush_skipped);
    result.skipped_due_to_failure = sink.sink_skipped_due_to_failure();
    // Charge the cache residency into the reported peak: the cached pyramid
    // is held alongside the MAP/REDUCE working set, so the sum is a
    // conservative upper bound on the run's real raster residency.
    result.peak_memory_bytes = result.peak_memory_bytes.saturating_add(cached_bytes);

    observer.on_event(EngineEvent::Finished {
        total_tiles: result.tiles_produced,
        levels: plan.levels.len() as u32,
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::observe::{CollectingObserver, NoopObserver};
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::MemorySink;
    use crate::streaming::RasterStripSource;
    use crate::streaming_mapreduce::LocalWorkExecutor;

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
    fn hot_cache_parity_with_monolithic_reference() {
        let src = gradient_raster(256, 256);
        let plan = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

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

        let hot_sink = MemorySink::new();
        let config = MapReduceConfig {
            memory_budget_bytes: 1_000_000,
            ..MapReduceConfig::default()
        };
        let strip_src = RasterStripSource::new(&src);
        generate_pyramid_mapreduce_hot_cache(
            &strip_src,
            &plan,
            &hot_sink,
            &config,
            &NoopObserver,
            &LocalWorkExecutor,
        )
        .unwrap();
        let mut hot_tiles = hot_sink.tiles();
        hot_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        assert_eq!(ref_tiles.len(), hot_tiles.len());
        for (r, h) in ref_tiles.iter().zip(hot_tiles.iter()) {
            assert_eq!(r.coord, h.coord);
            assert_eq!(r.data, h.data, "tile data diverged at {:?}", h.coord);
        }
    }

    #[test]
    fn hot_cache_emits_finished_exactly_once_and_last() {
        let src = gradient_raster(256, 256);
        let plan = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let observer = CollectingObserver::new();
        let sink = MemorySink::new();
        let config = MapReduceConfig {
            memory_budget_bytes: 1_000_000,
            ..MapReduceConfig::default()
        };
        let result = generate_pyramid_mapreduce_hot_cache(
            &RasterStripSource::new(&src),
            &plan,
            &sink,
            &config,
            &observer,
            &LocalWorkExecutor,
        )
        .unwrap();

        let events = observer.events();
        let finished: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::Finished { .. }))
            .collect();
        assert_eq!(
            finished.len(),
            1,
            "the withheld inner Finished must not leak alongside the re-emitted one"
        );
        assert!(
            matches!(events.last(), Some(EngineEvent::Finished { .. })),
            "Finished must be the final event"
        );
        assert!(
            matches!(
                events.last(),
                Some(EngineEvent::Finished { total_tiles, .. }) if *total_tiles == result.tiles_produced
            ),
            "Finished must carry the post-flush produced count"
        );
    }

    #[test]
    fn hot_cache_rejects_budget_too_small_for_strip() {
        // Parity with the MapReduce pre-flight: a budget below the minimum
        // aligned strip is rejected up front, before anything is cached.
        let src = gradient_raster(512, 512);
        let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let config = MapReduceConfig {
            memory_budget_bytes: 100_000,
            ..MapReduceConfig::default()
        };
        let err = generate_pyramid_mapreduce_hot_cache(
            &RasterStripSource::new(&src),
            &plan,
            &MemorySink::new(),
            &config,
            &NoopObserver,
            &LocalWorkExecutor,
        )
        .unwrap_err();
        assert!(
            matches!(err, EngineError::BudgetExceeded { .. }),
            "expected BudgetExceeded, got {err:?}"
        );
    }
}
