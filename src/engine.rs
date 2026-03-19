use std::sync::Arc;

use thiserror::Error;

use crate::observe::{EngineEvent, EngineObserver, MemoryTracker, NoopObserver};
use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::{Raster, RasterError};
use crate::resize;
use crate::sink::{SinkError, Tile, TileSink};

/// Errors that can occur during pyramid generation.
///
/// Wraps lower-level raster and sink errors into a single error type so that
/// callers of [`generate_pyramid`] and [`generate_pyramid_observed`] can handle
/// all failure modes uniformly. Also covers engine-specific conditions such as
/// cancellation and worker panics.
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
pub enum BlankTileStrategy {
    /// Emit blank tiles as full raster data (default). Every tile coordinate
    /// produces a complete image file, including tiles that are entirely one color.
    Emit,
    /// Replace blank tiles with a 1-byte placeholder marker (`0x00`). Consumers
    /// can detect these marker files by their size and generate their own blank
    /// tiles on the fly, saving significant disk space for sparse images.
    Placeholder,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            concurrency: 0,
            buffer_size: 64,
            background_rgb: [255, 255, 255],
            blank_tile_strategy: BlankTileStrategy::Emit,
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
}

/// Summary statistics returned after a successful pyramid generation.
///
/// Captures tile counts, level counts, and peak memory so that callers can
/// log, display progress, or assert correctness without inspecting the sink
/// directly. Every field is populated by [`generate_pyramid`] /
/// [`generate_pyramid_observed`].
#[derive(Debug)]
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
pub fn generate_pyramid(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
) -> Result<EngineResult, EngineError> {
    generate_pyramid_observed(source, plan, sink, config, &NoopObserver)
}

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
pub fn generate_pyramid_observed(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let top_level = plan.levels.len() - 1;
    let mut tiles_produced: u64 = 0;
    let mut tiles_skipped: u64 = 0;
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
        let (level_tiles, level_skipped) =
            extract_and_emit_level(&current, plan, level_idx as u32, sink, config, observer)?;
        tiles_produced += level_tiles;
        tiles_skipped += level_skipped;

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
        tiles_skipped,
        levels_processed: plan.levels.len() as u32,
        peak_memory_bytes: tracker.peak_bytes(),
    })
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
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let use_placeholders = config.blank_tile_strategy == BlankTileStrategy::Placeholder;

    if config.concurrency == 0 {
        // Single-threaded path
        let mut count = 0u64;
        let mut skipped = 0u64;
        for row in 0..level_plan.rows {
            for col in 0..level_plan.cols {
                let coord = TileCoord::new(level, col, row);
                let tile_raster = extract_tile(raster, plan, coord, config.background_rgb)?;
                let blank = use_placeholders && is_blank_tile(&tile_raster);
                if blank {
                    skipped += 1;
                }
                sink.write_tile(&Tile {
                    coord,
                    raster: tile_raster,
                    blank,
                })?;
                observer.on_event(EngineEvent::TileCompleted { coord });
                count += 1;
            }
        }
        Ok((count, skipped))
    } else {
        extract_and_emit_parallel(raster, plan, level, sink, config, observer)
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
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let total_tiles = level_plan.tile_count();

    if total_tiles == 0 {
        return Ok((0, 0));
    }

    let use_placeholders = config.blank_tile_strategy == BlankTileStrategy::Placeholder;

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
            let bg = config.background_rgb;

            s.spawn(move || {
                for coord in chunk {
                    let result = extract_tile(&raster, &plan, coord, bg)
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
            sink.write_tile(&tile)?;
            observer.on_event(EngineEvent::TileCompleted { coord });
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

    // For Google layout or plans with centre, tiles reference canvas space.
    // We need to map from canvas coords to image coords.
    if plan.layout == crate::planner::Layout::Google || plan.centre {
        let (off_x, off_y) = plan.centre_offset_at_level(coord.level);
        let img_w = raster.width();
        let img_h = raster.height();

        // Canvas-space tile rect
        let tile_x = rect.x;
        let tile_y = rect.y;
        let tile_w = rect.width;
        let tile_h = rect.height;

        // Image region in canvas space
        let img_left = off_x;
        let img_top = off_y;
        let img_right = off_x + img_w;
        let img_bottom = off_y + img_h;

        // Intersection of tile rect with image region
        let inter_left = tile_x.max(img_left);
        let inter_top = tile_y.max(img_top);
        let inter_right = (tile_x + tile_w).min(img_right);
        let inter_bottom = (tile_y + tile_h).min(img_bottom);

        if inter_left >= inter_right || inter_top >= inter_bottom {
            // Tile is entirely outside the image — solid background
            let padded = make_background_tile(ts, bpp, background_rgb);
            return Raster::new(ts, ts, raster.format(), padded);
        }

        let inter_w = inter_right - inter_left;
        let inter_h = inter_bottom - inter_top;

        // Source coords in image space
        let src_x = inter_left - off_x;
        let src_y = inter_top - off_y;

        // Destination offset within the tile
        let dst_x = inter_left - tile_x;
        let dst_y = inter_top - tile_y;

        if dst_x == 0 && dst_y == 0 && inter_w == tile_w && inter_h == tile_h {
            // Fast path: tile entirely within image
            let content = raster.extract(src_x, src_y, inter_w, inter_h)?;
            if content.width() == ts && content.height() == ts {
                return Ok(content);
            }
            // Shouldn't happen for Google, but handle gracefully
            let mut padded = make_background_tile(ts, bpp, background_rgb);
            let src_stride = content.width() as usize * bpp;
            let dst_stride = ts as usize * bpp;
            for row in 0..content.height() as usize {
                let src_start = row * src_stride;
                let dst_start = row * dst_stride;
                padded[dst_start..dst_start + src_stride]
                    .copy_from_slice(&content.data()[src_start..src_start + src_stride]);
            }
            return Raster::new(ts, ts, raster.format(), padded);
        }

        // Partial overlap: create background tile and blit the intersection
        let content = raster.extract(src_x, src_y, inter_w, inter_h)?;
        let mut padded = make_background_tile(ts, bpp, background_rgb);
        let src_stride = content.width() as usize * bpp;
        let dst_stride = ts as usize * bpp;
        for row in 0..inter_h as usize {
            let src_start = row * src_stride;
            let dst_start = (row + dst_y as usize) * dst_stride + dst_x as usize * bpp;
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

        let result = generate_pyramid(&src, &plan, &sink, &config).unwrap();
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

        generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();

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

        generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();

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
        generate_pyramid(&src, &plan, &ref_sink, &EngineConfig::default()).unwrap();

        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        for concurrency in [1, 2, 4] {
            let sink = MemorySink::new();
            let config = EngineConfig::default().with_concurrency(concurrency);
            generate_pyramid(&src, &plan, &sink, &config).unwrap();

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

        let result = generate_pyramid(&src, &plan, &sink, &EngineConfig::default()).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }
}
