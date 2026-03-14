use std::sync::Arc;

use thiserror::Error;

use crate::observe::{EngineEvent, EngineObserver, MemoryTracker, NoopObserver};
use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::{Raster, RasterError};
use crate::resize;
use crate::sink::{SinkError, Tile, TileSink};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("raster error: {0}")]
    Raster(#[from] RasterError),
    #[error("sink error: {0}")]
    Sink(#[from] SinkError),
    #[error("engine cancelled")]
    Cancelled,
    #[error("worker panicked")]
    WorkerPanic,
}

/// Configuration for the execution engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of worker threads for tile extraction. 0 = single-threaded (current thread).
    pub concurrency: usize,
    /// Maximum tiles buffered between producer and sink. Controls backpressure.
    pub buffer_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            concurrency: 0,
            buffer_size: 64,
        }
    }
}

impl EngineConfig {
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = n;
        self
    }
}

/// Result of a completed pyramid generation.
#[derive(Debug)]
pub struct EngineResult {
    pub tiles_produced: u64,
    pub levels_processed: u32,
    /// Peak tracked memory in bytes (raster buffers only).
    pub peak_memory_bytes: u64,
}

/// Generate a tile pyramid from a source raster.
///
/// Processes levels from full resolution (top) down to 1×1.
/// At each level, tiles are extracted and sent to the sink.
/// When `concurrency > 0`, tiles within each level are produced in parallel.
pub fn generate_pyramid(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
) -> Result<EngineResult, EngineError> {
    generate_pyramid_observed(source, plan, sink, config, &NoopObserver)
}

/// Generate a tile pyramid with an observer for progress events.
pub fn generate_pyramid_observed(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let top_level = plan.levels.len() - 1;
    let mut tiles_produced: u64 = 0;
    let tracker = MemoryTracker::new();

    // Track source raster
    let source_bytes = source.data().len() as u64;
    tracker.alloc(source_bytes);

    // Start with the source raster at full resolution
    let mut current = source.clone();

    // Process from top level (full res) down to level 0 (1×1)
    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];

        observer.on_event(EngineEvent::LevelStarted {
            level: level.level,
            width: level.width,
            height: level.height,
            tile_count: level.tile_count(),
        });

        // Downscale if not at the top level
        if level_idx < top_level {
            let old_bytes = current.data().len() as u64;
            current = resize::downscale_to(&current, level.width, level.height)?;
            let new_bytes = current.data().len() as u64;
            // Track: freed old level, allocated new
            tracker.dealloc(old_bytes);
            tracker.alloc(new_bytes);
        }

        // Extract and emit tiles for this level
        let level_tiles =
            extract_and_emit_level(&current, plan, level_idx as u32, sink, config, observer)?;
        tiles_produced += level_tiles;

        observer.on_event(EngineEvent::LevelCompleted {
            level: level.level,
            tiles_produced: level_tiles,
        });
    }

    // Free last raster from tracking
    tracker.dealloc(current.data().len() as u64);

    sink.finish()?;

    observer.on_event(EngineEvent::Finished {
        total_tiles: tiles_produced,
        levels: plan.levels.len() as u32,
    });

    Ok(EngineResult {
        tiles_produced,
        levels_processed: plan.levels.len() as u32,
        peak_memory_bytes: tracker.peak_bytes(),
    })
}

/// Extract all tiles for a single level and send them to the sink.
fn extract_and_emit_level(
    raster: &Raster,
    plan: &PyramidPlan,
    level: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<u64, EngineError> {
    let level_plan = &plan.levels[level as usize];

    if config.concurrency == 0 {
        // Single-threaded path
        let mut count = 0u64;
        for row in 0..level_plan.rows {
            for col in 0..level_plan.cols {
                let coord = TileCoord::new(level, col, row);
                let tile_raster = extract_tile(raster, plan, coord)?;
                sink.write_tile(&Tile {
                    coord,
                    raster: tile_raster,
                })?;
                observer.on_event(EngineEvent::TileCompleted { coord });
                count += 1;
            }
        }
        Ok(count)
    } else {
        extract_and_emit_parallel(raster, plan, level, sink, config, observer)
    }
}

/// Parallel tile extraction using a bounded channel for backpressure.
fn extract_and_emit_parallel(
    raster: &Raster,
    plan: &PyramidPlan,
    level: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<u64, EngineError> {
    let level_plan = &plan.levels[level as usize];
    let total_tiles = level_plan.tile_count();

    if total_tiles == 0 {
        return Ok(0);
    }

    // Bounded channel for backpressure: producers block when buffer is full
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Tile, EngineError>>(config.buffer_size);

    // Share the raster across worker threads (read-only)
    let raster = Arc::new(raster.clone());
    let plan = Arc::new(plan.clone());

    // Collect tile coordinates for this level
    let coords: Vec<TileCoord> = (0..level_plan.rows)
        .flat_map(|row| (0..level_plan.cols).map(move |col| TileCoord::new(level, col, row)))
        .collect();

    // Spawn workers
    let concurrency = config.concurrency.min(coords.len());
    let chunk_size = coords.len().div_ceil(concurrency);

    std::thread::scope(|s| {
        // Spawn producer threads
        for chunk in coords.chunks(chunk_size) {
            let tx = tx.clone();
            let raster = Arc::clone(&raster);
            let plan = Arc::clone(&plan);
            let chunk = chunk.to_vec();

            s.spawn(move || {
                for coord in chunk {
                    let result = extract_tile(&raster, &plan, coord)
                        .map(|tile_raster| Tile {
                            coord,
                            raster: tile_raster,
                        })
                        .map_err(EngineError::from);
                    if tx.send(result).is_err() {
                        break; // Consumer dropped
                    }
                }
            });
        }
        // Drop our copy so rx knows when all producers are done
        drop(tx);

        // Consumer: receive tiles and write to sink
        let mut count = 0u64;
        for result in rx {
            let tile = result?;
            let coord = tile.coord;
            sink.write_tile(&tile)?;
            observer.on_event(EngineEvent::TileCompleted { coord });
            count += 1;
        }
        Ok(count)
    })
}

/// Extract a single tile from a level raster.
fn extract_tile(
    raster: &Raster,
    plan: &PyramidPlan,
    coord: TileCoord,
) -> Result<Raster, RasterError> {
    let rect = plan
        .tile_rect(coord)
        .expect("tile_rect returned None for valid coord");

    raster.extract(rect.x, rect.y, rect.width, rect.height)
}

/// Check if a tile is "blank" — all pixels are identical.
/// Useful for blank tile skipping optimization.
pub fn is_blank_tile(raster: &Raster) -> bool {
    let data = raster.data();
    let bpp = raster.format().bytes_per_pixel();
    if data.len() <= bpp {
        return true;
    }
    let first_pixel = &data[..bpp];
    data.chunks(bpp).all(|px| px == first_pixel)
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

        let result = generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        let result = generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        generate_pyramid(&src, &plan, &sink, &config).unwrap();

        for tile in sink.tiles() {
            let rect = plan.tile_rect(tile.coord).unwrap();
            assert_eq!(tile.width, rect.width, "Width mismatch at {:?}", tile.coord);
            assert_eq!(
                tile.height, rect.height,
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
        generate_pyramid(&src, &plan, &ref_sink, &EngineConfig::default()).unwrap();

        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        for concurrency in [1, 2, 4, 8, 16] {
            let sink = MemorySink::new();
            let config = EngineConfig::default().with_concurrency(concurrency);
            generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        let result = generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();
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

        let result = generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();
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

        let result = generate_pyramid(&src, &plan, &sink, &config).unwrap();
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

        generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        let result = generate_pyramid(&src, &plan, &sink, &config).unwrap();
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

        let result = generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();

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

        let result = generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();

        let source_bytes = 1024u64 * 1024 * 3;
        // Should be less than 2x source (current level + some overhead)
        assert!(
            result.peak_memory_bytes < source_bytes * 2,
            "Peak {} >= 2x source {source_bytes}",
            result.peak_memory_bytes
        );
    }
}
