//! Strip-based streaming pyramid engine.
//!
//! The monolithic engine materialises the entire canvas before generating tiles,
//! requiring O(canvas_w × canvas_h) memory. For large rasters (e.g. 20k×30k
//! blueprint scans at 300 DPI), this can exceed available RAM.
//!
//! This module provides an alternative pipeline that processes the pyramid in
//! horizontal bands ("strips"), reducing peak memory to O(canvas_w × strip_h)
//! where strip_h is a configurable fraction of the canvas height.
//!
//! ## Architecture
//!
//! The engine divides the canvas into horizontal strips of uniform height and
//! processes each strip through a three-phase pipeline:
//!
//! 1. **Tile emission** — extract tiles that intersect the strip at the top
//!    pyramid level and write them to the sink.
//! 2. **Downscale** — apply the 2×2 box filter ([`crate::resize::downscale_half`])
//!    to halve the strip dimensions.
//! 3. **Propagation** — feed the halved strip into the next lower level, either
//!    pairing it with a previously stored half-strip or accumulating it into a
//!    monolithic buffer for small levels that fit entirely in memory.
//!
//! Levels below a size threshold ("monolithic threshold") are not processed via
//! strip pairing. Instead, the topmost monolithic level accumulates row data from
//! incoming half-strips, and during a final flush phase, the full raster for that
//! level is assembled, tiles are emitted, and each subsequent smaller level is
//! produced by downscaling the previous one. This mirrors the monolithic engine's
//! downscale chain and guarantees **pixel-exact parity** at every level.
//!
//! ## Strip height selection
//!
//! [`compute_strip_height`] chooses the tallest strip height that fits within
//! the caller's memory budget, constrained to multiples of `2 × tile_size` so
//! that tile rows and downscale pairs align cleanly at every level.
//!
//! ## Entry points
//!
//! - [`generate_pyramid_streaming`] — explicit streaming with a [`StripSource`].
//! - [`generate_pyramid_auto`] — auto-selects monolithic or streaming based on
//!   the budget vs. estimated monolithic peak memory.

use crate::engine::{BlankTileStrategy, EngineConfig, EngineError, EngineResult};
use crate::observe::{EngineEvent, EngineObserver, MemoryTracker};
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::raster::{Raster, RasterError};
use crate::resize;
use crate::sink::{Tile, TileSink};

/// Configuration for the streaming pyramid engine.
///
/// Wraps the standard [`EngineConfig`] with an additional memory budget that
/// controls strip height selection. The budget is a soft upper bound — the
/// engine uses it to maximise strip height (and therefore throughput) while
/// keeping estimated peak memory below the target.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Soft memory budget in bytes.
    ///
    /// [`compute_strip_height`] divides this by the per-unit cost to find the
    /// tallest strip that fits. If the budget is too small for even the minimum
    /// strip (2 × tile_size rows), the engine falls back to that minimum rather
    /// than refusing to run.
    pub memory_budget_bytes: u64,
    /// Standard engine settings: background colour, blank tile strategy, etc.
    pub engine: EngineConfig,
}

/// Provider of horizontal pixel bands for the streaming engine.
///
/// The streaming engine iterates top-to-bottom across the source image,
/// requesting one strip at a time via [`render_strip`](Self::render_strip).
/// Implementations decide *how* those pixels are produced:
///
/// - [`RasterStripSource`] — extracts bands from an in-memory [`Raster`].
/// - [`PdfiumStripSource`] (requires `pdfium` feature) — lazily renders a
///   PDF page and extracts bands from the cached render.
///
/// Custom implementations can back this trait with tiled TIFF decoders,
/// network streaming sources, or any producer that can emit rows on demand.
///
/// # Contract
///
/// - `render_strip(y, h)` must return a raster of width [`width()`](Self::width)
///   and height ≤ `h`, starting at row `y` in the full-resolution image.
/// - The engine calls strips in strictly increasing `y` order, but may request
///   different heights for the last strip if the image height is not a multiple
///   of the strip height.
/// - Implementations must be `Send + Sync` so they can be shared across threads
///   in future parallel-strip extensions.
pub trait StripSource: Send + Sync {
    /// Render or extract the horizontal band starting at `y_offset` with up to
    /// `height` rows, in full-resolution image coordinates.
    fn render_strip(&self, y_offset: u32, height: u32) -> Result<Raster, EngineError>;
    /// Full image width in pixels.
    fn width(&self) -> u32;
    /// Full image height in pixels.
    fn height(&self) -> u32;
    /// Pixel format of every strip returned by [`render_strip`](Self::render_strip).
    fn format(&self) -> PixelFormat;
}

/// [`StripSource`] backed by an in-memory [`Raster`].
///
/// Extracts row bands via [`Raster::extract`], which copies the requested
/// rows into a new buffer without touching the rest of the source.
///
/// This is the default source used by [`generate_pyramid_auto`] when the
/// monolithic path would exceed the memory budget. The source raster still
/// lives in memory, but the pyramid generation pipeline avoids the large
/// canvas-sized working allocation that the monolithic engine requires.
pub struct RasterStripSource<'a> {
    raster: &'a Raster,
}

impl<'a> RasterStripSource<'a> {
    /// Wrap an existing raster as a strip source.
    pub fn new(raster: &'a Raster) -> Self {
        Self { raster }
    }
}

impl<'a> StripSource for RasterStripSource<'a> {
    fn render_strip(&self, y_offset: u32, height: u32) -> Result<Raster, EngineError> {
        let h = height.min(self.raster.height() - y_offset);
        self.raster
            .extract(0, y_offset, self.raster.width(), h)
            .map_err(EngineError::from)
    }

    fn width(&self) -> u32 {
        self.raster.width()
    }

    fn height(&self) -> u32 {
        self.raster.height()
    }

    fn format(&self) -> PixelFormat {
        self.raster.format()
    }
}

// ---------------------------------------------------------------------------
// PdfiumStripSource — PDF strip rendering via pdfium
// ---------------------------------------------------------------------------

/// [`StripSource`] backed by pdfium rendering.
///
/// Renders the PDF page lazily on first `render_strip` call and caches the
/// full raster. Subsequent strip requests extract from the cached raster.
///
/// This avoids re-rendering the PDF for every strip while still enabling
/// the streaming pyramid engine to process the image in horizontal bands.
/// The memory win comes from the pyramid generation side (no canvas-sized
/// allocation), not from the PDF rendering itself.
///
/// For truly massive PDF pages where even a single full render is too large,
/// a matrix-based sub-page rendering approach using
/// `FPDF_RenderPageBitmapWithMatrix` would be needed. This is documented
/// as a future optimisation path.
#[cfg(feature = "pdfium")]
pub struct PdfiumStripSource {
    raster: std::sync::Mutex<Option<Raster>>,
    path: std::path::PathBuf,
    page: usize,
    dpi: u32,
    full_width: u32,
    full_height: u32,
}

#[cfg(feature = "pdfium")]
impl PdfiumStripSource {
    /// Create a new `PdfiumStripSource` for the given PDF page.
    ///
    /// Does not render the page immediately — the render is deferred until
    /// the first call to [`render_strip`](StripSource::render_strip).
    ///
    /// # Arguments
    ///
    /// * `path` — Path to the PDF file.
    /// * `page` — 1-based page number.
    /// * `dpi` — Render resolution in dots per inch.
    /// * `full_width` — Expected full-resolution pixel width.
    /// * `full_height` — Expected full-resolution pixel height.
    pub fn new(
        path: impl Into<std::path::PathBuf>,
        page: usize,
        dpi: u32,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        Self {
            raster: std::sync::Mutex::new(None),
            path: path.into(),
            page,
            dpi,
            full_width,
            full_height,
        }
    }

    /// Lazily render the PDF page, caching the result.
    fn ensure_rendered(&self) -> Result<(), EngineError> {
        let mut guard = self.raster.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let raster = crate::pdf::render_page_pdfium(&self.path, self.page, self.dpi)
            .map_err(|e| EngineError::Sink(crate::sink::SinkError::Other(e.to_string())))?;
        *guard = Some(raster);
        Ok(())
    }

    /// Get a reference to the cached raster, extracting a strip from it.
    fn extract_strip(&self, y_offset: u32, height: u32) -> Result<Raster, EngineError> {
        let guard = self.raster.lock().unwrap();
        let raster = guard
            .as_ref()
            .expect("ensure_rendered must be called first");
        let h = height.min(raster.height().saturating_sub(y_offset));
        if h == 0 {
            let bpp = raster.format().bytes_per_pixel();
            let data = vec![255u8; self.full_width as usize * height as usize * bpp];
            return Raster::new(self.full_width, height, raster.format(), data)
                .map_err(EngineError::from);
        }
        raster
            .extract(0, y_offset, raster.width(), h)
            .map_err(EngineError::from)
    }
}

#[cfg(feature = "pdfium")]
impl StripSource for PdfiumStripSource {
    fn render_strip(&self, y_offset: u32, height: u32) -> Result<Raster, EngineError> {
        self.ensure_rendered()?;
        self.extract_strip(y_offset, height)
    }

    fn width(&self) -> u32 {
        self.full_width
    }

    fn height(&self) -> u32 {
        self.full_height
    }

    fn format(&self) -> PixelFormat {
        // Default to RGBA8 since pdfium renders to RGBA
        let guard = self.raster.lock().unwrap();
        guard
            .as_ref()
            .map(|r| r.format())
            .unwrap_or(PixelFormat::Rgba8)
    }
}

// ---------------------------------------------------------------------------
// Strip height computation
// ---------------------------------------------------------------------------

/// Compute the tallest strip height that fits within a memory budget.
///
/// Strip height is constrained to multiples of `2 × tile_size`. This
/// alignment guarantees three properties the engine depends on:
///
/// - Each strip covers an integral number of tile rows, so every tile
///   at the top level is fully contained within exactly one strip.
/// - After the 2×2 box-filter downscale, the half-strip height is a
///   multiple of `tile_size`, keeping tile alignment at the next level.
/// - Pairs of half-strips combine into an even-height raster, allowing
///   the next downscale to split cleanly without orphaned rows.
///
/// The function estimates per-unit cost via [`estimate_streaming_memory`]
/// and divides the budget by that cost to find the maximum number of
/// units. If even a single unit exceeds the budget, returns `None`.
///
/// # Returns
///
/// `Some(strip_height)` — the largest aligned strip height within budget,
/// capped at the canvas height. `None` if the budget is insufficient.
pub fn compute_strip_height(plan: &PyramidPlan, format: PixelFormat, budget: u64) -> Option<u32> {
    let ch = plan.canvas_height as u64;
    let ts = plan.tile_size;
    // Minimum alignment unit: two tile rows, ensuring downscale pairs align
    let unit = 2 * ts;

    if plan.canvas_width == 0 || unit == 0 {
        return None;
    }

    // Estimate cost for the smallest possible strip (one unit). This walks
    // the full geometric series of live buffers across all pyramid levels,
    // matching the model used by estimate_streaming_peak_memory.
    let cost_per_unit = estimate_streaming_memory(plan, format, unit);
    if cost_per_unit == 0 {
        return None;
    }

    // How many units fit in the budget?
    let max_units = budget / cost_per_unit;
    if max_units == 0 {
        return None;
    }

    // Convert to pixel height, capped at the canvas
    let strip_height = (max_units * u64::from(unit)).min(ch) as u32;
    // Round down to ensure exact multiple of the alignment unit
    let strip_height = (strip_height / unit) * unit;
    if strip_height == 0 {
        return None;
    }
    Some(strip_height)
}

/// Estimate peak memory for streaming at a given strip height.
///
/// Delegates to [`PyramidPlan::estimate_streaming_peak_memory`], which
/// walks the geometric series of live buffers (strip + accumulator at
/// each level, dimensions halving) and adds a 10% safety margin.
///
/// Use this to validate that a chosen strip height stays within budget,
/// or to display estimated memory usage to callers before committing.
pub fn estimate_streaming_memory(
    plan: &PyramidPlan,
    format: PixelFormat,
    strip_height: u32,
) -> u64 {
    plan.estimate_streaming_peak_memory(format, strip_height)
}

// ---------------------------------------------------------------------------
// Core streaming loop
// ---------------------------------------------------------------------------

/// Generate a tile pyramid using the strip-based streaming pipeline.
///
/// This is the main entry point for memory-constrained pyramid generation.
/// The function walks the source image top-to-bottom in horizontal strips,
/// emitting tiles at the highest pyramid level from each strip, then
/// recursively downscaling and propagating the strip data through lower
/// levels.
///
/// # Processing phases
///
/// 1. **Strip loop** — iterate strips across the canvas height:
///    - Obtain the strip in canvas space (handling centring / padding).
///    - Emit tiles at the top level for tile rows that intersect the strip.
///    - Downscale the strip via the 2×2 box filter.
///    - Propagate the half-strip to lower levels.
///
/// 2. **Unpaired-strip flush** — for levels above the monolithic threshold
///    where an odd number of half-strips arrived, the leftover strip is
///    emitted and propagated. This happens when the canvas height is not an
///    exact multiple of the strip height.
///
/// 3. **Monolithic flush** — small levels (at or below the monolithic
///    threshold) are flushed from top to bottom. The topmost monolithic level
///    has its full raster assembled from accumulated row data. Each subsequent
///    level is produced by downscaling the previous level's raster, mirroring
///    the monolithic engine's pipeline for pixel-exact parity.
///
/// # Errors
///
/// Propagates errors from the source, sink, raster operations, and the
/// 2×2 downscale filter.
pub fn generate_pyramid_streaming(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &StreamingConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let format = source.format();
    let strip_height = match compute_strip_height(plan, format, config.memory_budget_bytes) {
        Some(h) => h,
        None => {
            // Budget too small for even one aligned unit — fall back to the
            // minimum (2 × tile_size). The engine will still work, just with
            // potentially higher memory than the budget intended.
            2 * plan.tile_size
        }
    };

    let ch = plan.canvas_height;
    let top_level = plan.levels.len() - 1;
    let tracker = MemoryTracker::new();
    let bpp = format.bytes_per_pixel();

    let mut tiles_produced: u64 = 0;
    let mut tiles_skipped: u64 = 0;

    // --- Accumulator state ---
    //
    // Two accumulator arrays serve different roles:
    //
    // `accumulators` (per-level Option<Raster>):
    //   Used for strip pairing at levels *above* the monolithic threshold.
    //   At each such level, half-strips arrive one at a time from propagation.
    //   The first half-strip is stored; when the second arrives, the pair is
    //   concatenated vertically, tiles are emitted, and the combined raster
    //   is downscaled for the next level.
    //
    // `mono_accumulators` (per-level Vec<u8>):
    //   Used for levels *at or below* the monolithic threshold. These levels
    //   are small enough that the full raster fits in memory alongside the
    //   strip working set. Only the topmost monolithic level actually
    //   accumulates data from propagation; lower levels are produced by
    //   downscaling during the flush phase.
    let mut accumulators: Vec<Option<Raster>> = vec![None; plan.levels.len()];

    // Partition levels into "large" (strip-paired) and "small" (monolithic).
    // The threshold is the highest level whose full raster ≤ one strip's budget.
    let monolithic_threshold = find_monolithic_threshold(plan, format, strip_height);

    let mut mono_accumulators: Vec<Vec<u8>> = plan.levels.iter().map(|_| Vec::new()).collect();

    // Emit LevelStarted for all levels upfront. The streaming engine
    // processes levels in an interleaved fashion, so per-level start/end
    // events bracket the entire run rather than individual level passes.
    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];
        observer.on_event(EngineEvent::LevelStarted {
            level: level.level,
            width: level.width,
            height: level.height,
            tile_count: level.tile_count(),
        });
    }

    // ===================================================================
    // Phase 1: Strip loop — iterate horizontal bands top-to-bottom
    // ===================================================================
    let total_strips = ch.div_ceil(strip_height);
    let mut strip_index: u32 = 0;
    let mut y: u32 = 0;
    while y < ch {
        // Clamp the last strip if the canvas height isn't a multiple of
        // strip_height. The shorter strip is handled correctly by all
        // downstream functions.
        let sh = strip_height.min(ch - y);

        // Step 1: Obtain the strip in canvas coordinates. For Google layout
        // with centring this embeds the source rows into a canvas-width
        // background; for DeepZoom the strip is the raw source rows.
        let strip = obtain_canvas_strip(source, plan, y, sh, &config.engine)?;

        observer.on_event(EngineEvent::StripRendered {
            strip_index,
            total_strips,
        });
        strip_index += 1;
        let strip_bytes = strip.data().len() as u64;
        tracker.alloc(strip_bytes);

        // Step 2: Emit tiles at the top pyramid level for tile rows that
        // intersect this strip's Y range.
        let (tp, ts_skip) = emit_strip_tiles(
            &strip,
            plan,
            top_level as u32,
            y,
            sink,
            &config.engine,
            observer,
        )?;
        tiles_produced += tp;
        tiles_skipped += ts_skip;

        // Step 3: Downscale the strip with the 2×2 box filter, producing
        // a half-strip at the next level. Free the original strip's memory
        // charge before accounting the new one.
        let half = resize::downscale_half(&strip)?;
        tracker.dealloc(strip_bytes);
        let half_bytes = half.data().len() as u64;
        tracker.alloc(half_bytes);

        // Step 4: Push the half-strip into the recursive propagation tree.
        // It will be paired with another half-strip at the next level, or
        // appended to a monolithic accumulator if the level is small enough.
        propagate_down(
            half,
            top_level - 1,
            y / 2,
            &mut accumulators,
            &mut mono_accumulators,
            monolithic_threshold,
            plan,
            sink,
            &config.engine,
            observer,
            &tracker,
            &mut tiles_produced,
            &mut tiles_skipped,
        )?;

        y += sh;
    }

    // ===================================================================
    // Phase 2: Flush unpaired strip accumulators
    // ===================================================================
    //
    // When the canvas height isn't evenly divisible by the strip height,
    // the last half-strip at some levels arrives without a partner. These
    // leftovers sit in `accumulators[level_idx]` and must be emitted and
    // propagated before the monolithic flush.
    //
    // Process from highest to lowest so that propagation feeds data into
    // levels below, which may themselves be monolithic.
    for level_idx in (monolithic_threshold + 1..plan.levels.len()).rev() {
        if let Some(leftover) = accumulators[level_idx].take() {
            // The leftover covers the bottom slice of this level. Its Y
            // position is the level height minus the strip's own height.
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
                &config.engine,
                observer,
            )?;
            tiles_produced += tp;
            tiles_skipped += ts_skip;

            // Continue the downscale chain into lower levels
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
                    &config.engine,
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
    //
    // Only the topmost monolithic level has accumulated raw row data via
    // propagate_down. All levels below it are empty — they are produced
    // here by downscaling the level above, exactly as the monolithic
    // engine does. This guarantees pixel-exact parity: each pixel goes
    // through the same sequence of 2×2 box-filter averages regardless
    // of whether the pyramid was built monolithically or via streaming.
    {
        let top_mono = monolithic_threshold.min(plan.levels.len() - 1);
        let mut prev_raster: Option<Raster> = None;

        // Walk from the largest monolithic level (top_mono) down to level 0
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
                // Produce this level by downscaling the level above,
                // mirroring the monolithic engine's downscale chain
                resize::downscale_half(&prev)?
            } else {
                // Topmost monolithic level — assemble from accumulated rows.
                let mut acc_data = std::mem::take(&mut mono_accumulators[level_idx]);
                if acc_data.is_empty() {
                    continue;
                }
                let expected = lw as usize * lh as usize * bpp;

                // The accumulated data may exceed the expected size when
                // the source has odd dimensions and downscale_half produces
                // div_ceil rows from each strip fragment. Truncate to the
                // exact expected byte count.
                if acc_data.len() > expected {
                    acc_data.truncate(expected);
                }
                // Conversely, if the last strip was shorter than expected,
                // pad remaining rows with the background colour.
                if acc_data.len() < expected {
                    let filled_rows = acc_data.len() / (lw as usize * bpp);
                    acc_data.resize(expected, 0);
                    fill_background_rows(
                        &mut acc_data,
                        filled_rows,
                        lw,
                        lh,
                        bpp,
                        config.engine.background_rgb,
                    );
                }
                Raster::new(lw, lh, format, acc_data)?
            };

            let (tp, ts_skip) = emit_full_level_tiles(
                &raster,
                plan,
                level_idx as u32,
                sink,
                &config.engine,
                observer,
            )?;
            tiles_produced += tp;
            tiles_skipped += ts_skip;

            // Retain this raster so the next iteration can downscale it
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
    })
}

/// Auto-selecting entry point that chooses between monolithic and streaming.
///
/// Compares the estimated peak memory of the monolithic engine against the
/// configured budget. If the monolithic path fits, it delegates to
/// [`generate_pyramid_observed`](crate::engine::generate_pyramid_observed)
/// for maximum throughput. Otherwise, wraps the source raster in a
/// [`RasterStripSource`] and calls [`generate_pyramid_streaming`].
///
/// This is the recommended entry point when the caller has an in-memory
/// [`Raster`] and wants automatic memory management without committing
/// to a specific engine implementation.
pub fn generate_pyramid_auto(
    source: &Raster,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &StreamingConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let mono_peak = plan.estimate_peak_memory_for_format(source.format());

    if mono_peak <= config.memory_budget_bytes {
        crate::engine::generate_pyramid_observed(source, plan, sink, &config.engine, observer)
    } else {
        let strip_source = RasterStripSource::new(source);
        generate_pyramid_streaming(&strip_source, plan, sink, config, observer)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Find the highest level index at or below which we accumulate monolithically.
///
/// Levels are classified as "small" when the full-level raster (width × height
/// × bytes_per_pixel) fits within the memory footprint of one top-level strip.
/// Such levels are cheap to hold entirely in memory, so the engine accumulates
/// their row data into a flat byte buffer rather than strip-pairing them.
///
/// During the monolithic flush phase, only the topmost monolithic level's
/// accumulator is populated directly. All smaller levels are produced by
/// cascading downscale_half, which replicates the monolithic engine's behavior.
///
/// Scanning from the top (largest) level downward, the first level whose
/// full raster fits within the strip budget becomes the threshold.
fn find_monolithic_threshold(plan: &PyramidPlan, format: PixelFormat, strip_height: u32) -> usize {
    let bpp = format.bytes_per_pixel() as u64;
    // Reference budget: one full-width strip at the top level
    let strip_budget = plan.canvas_width as u64 * strip_height as u64 * bpp;

    for level_idx in (0..plan.levels.len()).rev() {
        let (lw, lh) = if plan.layout == Layout::Google {
            plan.canvas_size_at_level(plan.levels[level_idx].level)
        } else {
            let lp = &plan.levels[level_idx];
            (lp.width, lp.height)
        };
        let level_bytes = lw as u64 * lh as u64 * bpp;
        if level_bytes <= strip_budget {
            return level_idx;
        }
    }
    // Every level is larger than a strip — treat level 0 as monolithic anyway
    0
}

/// Obtain a horizontal strip in canvas coordinate space.
///
/// The canvas may differ from the raw source in two ways:
///
/// - **Centring offset** (Google layout with `centre = true`): the source
///   image is placed at an offset inside a power-of-2-sized canvas. This
///   function creates a background-filled strip of `canvas_width × height`
///   and blits the intersecting source rows at the correct offset.
///
/// - **Width padding** (Google layout without centring): the source image
///   sits at (0, 0) but the canvas extends to the power-of-2 width. Source
///   rows are copied into the left portion; the remainder is background.
///
/// - **DeepZoom / Xyz**: the canvas equals the image, so the strip is just
///   the raw source rows with no padding or offset.
fn obtain_canvas_strip(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    y: u32,
    height: u32,
    config: &EngineConfig,
) -> Result<Raster, EngineError> {
    let cw = plan.canvas_width;
    let format = source.format();
    let bpp = format.bytes_per_pixel();

    if plan.centre && (plan.centre_offset_x > 0 || plan.centre_offset_y > 0) {
        // Canvas-space strip with centring
        embed_strip_in_canvas(source, plan, y, height, config.background_rgb)
    } else if plan.layout == Layout::Google {
        // Google layout without centring: image at (0,0), pad to canvas width
        let src_h = source.height();
        if y >= src_h {
            // Entirely below the source — solid background
            let data = make_background_buffer(cw, height, bpp, config.background_rgb);
            Raster::new(cw, height, format, data).map_err(EngineError::from)
        } else {
            let avail_h = (src_h - y).min(height);
            let src_strip = source.render_strip(y, avail_h)?;

            if src_strip.width() == cw && avail_h == height {
                return Ok(src_strip);
            }

            // Need to embed in canvas-width strip
            let mut data = make_background_buffer(cw, height, bpp, config.background_rgb);
            let src_row_bytes = src_strip.width() as usize * bpp;
            let dst_stride = cw as usize * bpp;
            for row in 0..avail_h as usize {
                let src_start = row * src_row_bytes;
                let dst_start = row * dst_stride;
                data[dst_start..dst_start + src_row_bytes]
                    .copy_from_slice(&src_strip.data()[src_start..src_start + src_row_bytes]);
            }
            Raster::new(cw, height, format, data).map_err(EngineError::from)
        }
    } else {
        // DeepZoom/Xyz: strip is in image space, width = source width
        let src_h = source.height();
        let avail_h = (src_h - y).min(height);
        source.render_strip(y, avail_h)
    }
}

/// Embed source rows into a canvas-width background strip for centred layouts.
///
/// Allocates a `canvas_width × strip_h` background buffer, computes the
/// intersection of the strip's Y range with the offset source image, and
/// copies the overlapping rows at the correct (x, y) offset. Rows above
/// or below the source image remain filled with the background colour.
fn embed_strip_in_canvas(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    canvas_y: u32,
    strip_h: u32,
    background_rgb: [u8; 3],
) -> Result<Raster, EngineError> {
    let cw = plan.canvas_width;
    let format = source.format();
    let bpp = format.bytes_per_pixel();
    let ox = plan.centre_offset_x;
    let oy = plan.centre_offset_y;
    let src_w = source.width();
    let src_h = source.height();

    let mut data = make_background_buffer(cw, strip_h, bpp, background_rgb);

    // Intersection of canvas strip [canvas_y, canvas_y + strip_h) with
    // image region [oy, oy + src_h) in canvas space.
    let strip_top = canvas_y;
    let strip_bottom = canvas_y + strip_h;
    let img_top = oy;
    let img_bottom = oy + src_h;

    let inter_top = strip_top.max(img_top);
    let inter_bottom = strip_bottom.min(img_bottom);

    if inter_top < inter_bottom {
        let src_y = inter_top - oy;
        let src_rows = inter_bottom - inter_top;
        let src_strip = source.render_strip(src_y, src_rows)?;

        let dst_stride = cw as usize * bpp;
        let src_row_bytes = src_w.min(src_strip.width()) as usize * bpp;
        let local_y = (inter_top - canvas_y) as usize;

        for row in 0..src_rows as usize {
            let src_start = row * src_strip.stride();
            let dst_start = (local_y + row) * dst_stride + ox as usize * bpp;
            let copy_len = src_row_bytes.min(data.len() - dst_start);
            data[dst_start..dst_start + copy_len]
                .copy_from_slice(&src_strip.data()[src_start..src_start + copy_len]);
        }
    }

    Raster::new(cw, strip_h, format, data).map_err(EngineError::from)
}

/// Allocate a `w × h` pixel buffer filled with the background colour.
///
/// Handles 1-channel (grayscale), 3-channel (RGB), and 4-channel (RGBA,
/// alpha = 255) formats. Used for padding tiles and canvas strips.
fn make_background_buffer(w: u32, h: u32, bpp: usize, background_rgb: [u8; 3]) -> Vec<u8> {
    let size = w as usize * h as usize * bpp;
    let mut buf = vec![0u8; size];
    let bg_pixel: Vec<u8> = match bpp {
        1 => vec![background_rgb[0]],
        3 => background_rgb.to_vec(),
        4 => vec![background_rgb[0], background_rgb[1], background_rgb[2], 255],
        _ => vec![background_rgb[0]; bpp],
    };
    for pixel in buf.chunks_exact_mut(bpp) {
        pixel.copy_from_slice(&bg_pixel);
    }
    buf
}

/// Fill background colour into unfilled rows of a partially-populated buffer.
///
/// Used during the monolithic flush when the accumulated data for a level
/// is shorter than the expected full-level byte count (e.g. the last strip
/// was shorter than a full strip height). Rows `[0, filled_rows)` are left
/// untouched; rows `[filled_rows, h)` are overwritten with the background.
fn fill_background_rows(
    buf: &mut [u8],
    filled_rows: usize,
    w: u32,
    h: u32,
    bpp: usize,
    background_rgb: [u8; 3],
) {
    let bg_pixel: Vec<u8> = match bpp {
        1 => vec![background_rgb[0]],
        3 => background_rgb.to_vec(),
        4 => vec![background_rgb[0], background_rgb[1], background_rgb[2], 255],
        _ => vec![background_rgb[0]; bpp],
    };
    let stride = w as usize * bpp;
    for row in filled_rows..h as usize {
        let start = row * stride;
        let end = start + stride;
        if end > buf.len() {
            break;
        }
        for pixel in buf[start..end].chunks_exact_mut(bpp) {
            pixel.copy_from_slice(&bg_pixel);
        }
    }
}

/// Emit tiles for the tile rows that a strip covers at a given pyramid level.
///
/// Iterates all tile columns across each tile row that overlaps the strip's
/// vertical range `[strip_canvas_y, strip_canvas_y + strip.height())`. For
/// each tile, extracts the pixel region via [`extract_tile_from_strip`],
/// checks for blank-tile optimisation, and writes the tile to the sink.
///
/// Returns `(tiles_produced, tiles_skipped)` where `tiles_skipped` counts
/// tiles marked as blank placeholders.
fn emit_strip_tiles(
    strip: &Raster,
    plan: &PyramidPlan,
    level: u32,
    strip_canvas_y: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<(u64, u64), EngineError> {
    let level_plan = &plan.levels[level as usize];
    let ts = plan.tile_size;
    let use_placeholders = config.blank_tile_strategy == BlankTileStrategy::Placeholder;

    let first_row = strip_canvas_y / ts;
    let last_row = (strip_canvas_y + strip.height()).div_ceil(ts);
    let last_row = last_row.min(level_plan.rows);

    let mut count = 0u64;
    let mut skipped = 0u64;

    for row in first_row..last_row {
        for col in 0..level_plan.cols {
            let coord = TileCoord::new(level, col, row);
            let tile_raster =
                extract_tile_from_strip(strip, plan, coord, strip_canvas_y, config.background_rgb)?;
            let blank = use_placeholders && crate::engine::is_blank_tile(&tile_raster);
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
}

/// Extract a single tile from a strip, translating canvas coordinates to
/// strip-local coordinates and handling edge padding.
///
/// This is the streaming analogue of `engine::extract_tile`. The key
/// difference is that the source raster is a narrow horizontal strip rather
/// than the full level, so tile Y coordinates must be translated by
/// `strip_canvas_y` and the tile's vertical extent may only partially
/// overlap the strip.
///
/// ## Edge tile padding (DeepZoom / Xyz)
///
/// For layouts other than Google, the planner's `tile_rect` returns the
/// content region that may be smaller than `tile_size` at the right / bottom
/// edges of the image. When `overlap == 0` and the content is smaller than
/// the tile size in either dimension, the tile is padded to full
/// `tile_size × tile_size` with the background colour. This matches the
/// monolithic engine's `extract_tile` behavior exactly and ensures
/// pixel-exact parity between the two pipelines.
///
/// ## Google layout
///
/// Google tiles are always `tile_size × tile_size`. Regions outside the
/// source image are filled with background.
fn extract_tile_from_strip(
    strip: &Raster,
    plan: &PyramidPlan,
    coord: TileCoord,
    strip_canvas_y: u32,
    background_rgb: [u8; 3],
) -> Result<Raster, RasterError> {
    let rect = plan
        .tile_rect(coord)
        .expect("tile_rect returned None for valid coord");
    let ts = plan.tile_size;
    let bpp = strip.format().bytes_per_pixel();

    // --- Coordinate translation ---
    // The tile rect is in level-space (canvas) coordinates. The strip starts
    // at canvas row `strip_canvas_y`, so subtract that to get the strip-local
    // row offset for the tile's top edge.
    let local_y = rect.y.saturating_sub(strip_canvas_y);
    let strip_h = strip.height();
    let strip_w = strip.width();

    // How many rows of the tile fall before the start of this strip? This
    // happens when the tile straddles the boundary between two strips.
    let avail_y_start = strip_canvas_y.saturating_sub(rect.y);
    // Available rows of tile content within this strip
    let avail_h = if local_y + rect.height > strip_h {
        strip_h.saturating_sub(local_y)
    } else {
        rect.height
    };

    // --- Determine content and output dimensions ---
    // Google tiles are always square at tile_size. DeepZoom/Xyz edge tiles
    // have smaller content (rect.width × rect.height).
    let tile_w = if plan.layout == Layout::Google {
        ts
    } else {
        rect.width
    };
    let tile_h = if plan.layout == Layout::Google {
        ts
    } else {
        rect.height
    };

    // Horizontal intersection: clamp tile rect against strip width
    let avail_x = rect.x.min(strip_w);
    let avail_w = (rect.x + rect.width).min(strip_w).saturating_sub(avail_x);

    // --- Determine final output dimensions ---
    // DeepZoom edge tiles (overlap == 0, content < tile_size) must be padded
    // to the full tile_size. This matches the monolithic engine's behavior.
    let needs_edge_pad = |dim: u32| -> u32 {
        if plan.layout == Layout::Google || (plan.overlap == 0 && dim < ts) {
            ts
        } else {
            dim
        }
    };

    // --- Case 1: tile entirely outside the strip ---
    if avail_w == 0 || avail_h == 0 || avail_x >= strip_w || local_y >= strip_h {
        let out_w = needs_edge_pad(tile_w);
        let out_h = needs_edge_pad(tile_h);
        let padded = make_background_buffer(out_w, out_h, bpp, background_rgb);
        return Raster::new(out_w, out_h, strip.format(), padded);
    }

    // --- Case 2: tile fully within strip (fast path) ---
    if avail_x == rect.x
        && avail_w == tile_w
        && local_y == 0
        && avail_h == tile_h
        && avail_y_start == 0
    {
        let content = strip.extract(avail_x, local_y, tile_w, tile_h)?;

        // Even on the fast path, DeepZoom edge tiles may need padding to
        // reach tile_size × tile_size when overlap is zero.
        if plan.overlap == 0
            && plan.layout != Layout::Google
            && (content.width() < ts || content.height() < ts)
        {
            let mut padded = make_background_buffer(ts, ts, bpp, background_rgb);
            let src_stride = content.width() as usize * bpp;
            let dst_stride = ts as usize * bpp;
            for row in 0..content.height() as usize {
                let src_start = row * src_stride;
                let dst_start = row * dst_stride;
                padded[dst_start..dst_start + src_stride]
                    .copy_from_slice(&content.data()[src_start..src_start + src_stride]);
            }
            return Raster::new(ts, ts, strip.format(), padded);
        }
        return Ok(content);
    }

    // --- Case 3: partial overlap — extract available region and pad ---
    let content = strip.extract(avail_x, local_y, avail_w, avail_h)?;

    let out_w = needs_edge_pad(tile_w);
    let out_h = needs_edge_pad(tile_h);

    // Start with a fully background-filled output buffer, then blit the
    // available content pixels at the correct offset.
    let mut padded = make_background_buffer(out_w, out_h, bpp, background_rgb);
    let src_stride = avail_w as usize * bpp;
    let dst_stride = out_w as usize * bpp;
    // Horizontal offset within the output tile (non-zero when the tile rect
    // starts before the strip's left edge, which shouldn't happen but is
    // handled defensively).
    let dx = (avail_x - rect.x.min(avail_x)) as usize * bpp;
    // Vertical offset: rows of the tile that precede this strip
    let dy = avail_y_start as usize;

    for row in 0..avail_h as usize {
        let src_start = row * src_stride;
        let dst_start = (row + dy) * dst_stride + dx;
        if dst_start + src_stride <= padded.len() {
            padded[dst_start..dst_start + src_stride]
                .copy_from_slice(&content.data()[src_start..src_start + src_stride]);
        }
    }
    Raster::new(out_w, out_h, strip.format(), padded)
}

/// Emit all tiles for a complete level raster.
///
/// Used during the monolithic flush phase when a full-level raster has been
/// assembled (either from accumulated data or by downscaling the level above).
/// Delegates to [`emit_strip_tiles`] with `strip_canvas_y = 0`, since the
/// raster covers the entire level.
fn emit_full_level_tiles(
    raster: &Raster,
    plan: &PyramidPlan,
    level: u32,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<(u64, u64), EngineError> {
    emit_strip_tiles(raster, plan, level, 0, sink, config, observer)
}

/// Recursively propagate a downscaled half-strip to lower pyramid levels.
///
/// This is the core recursive function of the streaming engine. Each call
/// receives a half-strip (the output of `downscale_half` from the level
/// above) and routes it into one of two pipelines:
///
/// ## Monolithic path (level_idx ≤ monolithic_threshold)
///
/// The half-strip's raw pixel bytes are appended to the monolithic
/// accumulator. No further propagation occurs — lower levels will be
/// produced during the flush phase by downscaling the full assembled raster.
/// This is critical for pixel-exact parity: independently downscaling
/// fragments and concatenating produces different rounding than downscaling
/// the full level at once (the 2×2 box filter averages different row pairs
/// when fragments have odd heights).
///
/// ## Strip-pairing path (level_idx > monolithic_threshold)
///
/// Half-strips arrive in pairs from consecutive top-level strips. The first
/// half-strip is stored in `accumulators[level_idx]`. When the second
/// arrives, the pair is concatenated vertically into a combined strip, tiles
/// are emitted for the tile rows it covers, and the combined strip is
/// downscaled and propagated to the next lower level.
///
/// If the canvas height is not evenly divisible by the strip height, the
/// last half-strip at some levels will arrive without a partner. These
/// leftovers are handled in Phase 2 (unpaired-strip flush) of the main
/// `generate_pyramid_streaming` function.
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn propagate_down(
    half_strip: Raster,
    level_idx: usize,
    strip_y_at_level: u32,
    accumulators: &mut Vec<Option<Raster>>,
    mono_accumulators: &mut Vec<Vec<u8>>,
    monolithic_threshold: usize,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
    tracker: &MemoryTracker,
    tiles_produced: &mut u64,
    tiles_skipped: &mut u64,
) -> Result<(), EngineError> {
    if level_idx <= monolithic_threshold {
        // Monolithic path: append rows and stop. The flush phase will
        // assemble the full raster and cascade downscales from there.
        let acc = &mut mono_accumulators[level_idx];
        acc.extend_from_slice(half_strip.data());
        return Ok(());
    }

    // Strip-pairing path: check if we already have a stored first half.
    match accumulators[level_idx].take() {
        None => {
            // First of pair — store and wait for the partner
            accumulators[level_idx] = Some(half_strip);
        }
        Some(prev) => {
            // Second of pair — stack top-over-bottom to reconstruct the
            // full strip height at this level
            let combined = concat_vertical(&prev, &half_strip)?;
            // The combined strip's Y offset is where the first half started
            let combined_y = strip_y_at_level.saturating_sub(prev.height());

            // Emit tiles for the tile rows covered by the combined strip
            let (tp, ts_skip) = emit_strip_tiles(
                &combined,
                plan,
                level_idx as u32,
                combined_y,
                sink,
                config,
                observer,
            )?;
            *tiles_produced += tp;
            *tiles_skipped += ts_skip;

            // Continue the downscale chain into the next lower level
            if level_idx > 0 {
                let further_half = resize::downscale_half(&combined)?;
                propagate_down(
                    further_half,
                    level_idx - 1,
                    combined_y / 2,
                    accumulators,
                    mono_accumulators,
                    monolithic_threshold,
                    plan,
                    sink,
                    config,
                    observer,
                    tracker,
                    tiles_produced,
                    tiles_skipped,
                )?;
            }
        }
    }

    Ok(())
}

/// Concatenate two rasters vertically (same width, stacked top-over-bottom).
///
/// Used to combine paired half-strips into a single strip before tile
/// emission and further downscaling. Both rasters must have the same width
/// and pixel format. The result has height = `top.height() + bottom.height()`.
fn concat_vertical(top: &Raster, bottom: &Raster) -> Result<Raster, RasterError> {
    debug_assert_eq!(top.width(), bottom.width());
    debug_assert_eq!(top.format(), bottom.format());

    let w = top.width();
    let h = top.height() + bottom.height();
    // Row data is contiguous for both rasters, so a simple byte concatenation
    // produces the correct layout for the combined raster.
    let mut data = Vec::with_capacity(top.data().len() + bottom.data().len());
    data.extend_from_slice(top.data());
    data.extend_from_slice(bottom.data());
    Raster::new(w, h, top.format(), data)
}

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

    fn solid_raster(w: u32, h: u32, val: u8) -> Raster {
        let data = vec![val; w as usize * h as usize * 3];
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    // -- Phase 1: Pure function tests --

    #[test]
    fn compute_strip_height_basic() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        // Large budget → strip height == canvas height
        let sh = compute_strip_height(&plan, PixelFormat::Rgb8, u64::MAX);
        assert!(sh.is_some());
        let sh = sh.unwrap();
        assert!(sh >= plan.tile_size * 2);
        assert!(sh <= plan.canvas_height);
    }

    #[test]
    fn compute_strip_height_tight_budget() {
        let planner = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        // Tiny budget → should return minimum strip or None
        let sh = compute_strip_height(&plan, PixelFormat::Rgb8, 1);
        assert!(sh.is_none());
    }

    #[test]
    fn compute_strip_height_is_multiple_of_2ts() {
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        // Moderate budget
        let budget = 50_000_000u64; // 50 MB
        if let Some(sh) = compute_strip_height(&plan, PixelFormat::Rgb8, budget) {
            assert_eq!(sh % (2 * plan.tile_size), 0);
        }
    }

    #[test]
    fn estimate_streaming_memory_monotonic() {
        let planner = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let m1 = estimate_streaming_memory(&plan, PixelFormat::Rgb8, 512);
        let m2 = estimate_streaming_memory(&plan, PixelFormat::Rgb8, 1024);
        assert!(m2 >= m1, "Larger strip should use more memory");
    }

    #[test]
    fn estimate_peak_memory_for_format_scales_with_bpp() {
        let planner = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let rgb = plan.estimate_peak_memory_for_format(PixelFormat::Rgb8);
        let rgba = plan.estimate_peak_memory_for_format(PixelFormat::Rgba8);
        // RGBA (4 bpp) should be > RGB (3 bpp)
        assert!(rgba > rgb);
    }

    // -- Phase 2: Parity tests --

    #[test]
    fn streaming_parity_deepzoom_small() {
        // Bit-exact parity: streaming vs monolithic on a small image
        let src = gradient_raster(256, 256);
        let planner = PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        // Monolithic
        let ref_sink = MemorySink::new();
        crate::engine::generate_pyramid(&src, &plan, &ref_sink, &EngineConfig::default()).unwrap();
        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        // Streaming with generous budget (should still produce identical output)
        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: u64::MAX,
            engine: EngineConfig::default(),
        };
        generate_pyramid_auto(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        let mut tiles = sink.tiles();
        tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        // With huge budget, auto should pick monolithic — tiles must match exactly
        assert_eq!(ref_tiles.len(), tiles.len(), "tile count mismatch");
        for (r, t) in ref_tiles.iter().zip(tiles.iter()) {
            assert_eq!(r.coord, t.coord);
            assert_eq!(r.data, t.data, "Tile data mismatch at {:?}", t.coord);
        }
    }

    #[test]
    fn streaming_produces_all_tiles_deepzoom() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: 1_000_000, // 1 MB — force streaming
            engine: EngineConfig::default(),
        };
        let strip_src = RasterStripSource::new(&src);
        let result =
            generate_pyramid_streaming(&strip_src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert_eq!(
            result.tiles_produced,
            plan.total_tile_count(),
            "Not all tiles produced"
        );
    }

    #[test]
    fn streaming_produces_all_tiles_google_centre() {
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: 2_000_000,
            engine: EngineConfig::default(),
        };
        let strip_src = RasterStripSource::new(&src);
        let result =
            generate_pyramid_streaming(&strip_src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert_eq!(
            result.tiles_produced,
            plan.total_tile_count(),
            "Not all tiles produced for Google+centre"
        );
    }

    #[test]
    fn auto_selects_monolithic_for_large_budget() {
        let src = gradient_raster(128, 128);
        let planner = PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let ref_sink = MemorySink::new();
        crate::engine::generate_pyramid(&src, &plan, &ref_sink, &EngineConfig::default()).unwrap();
        let mut ref_tiles = ref_sink.tiles();
        ref_tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: u64::MAX,
            engine: EngineConfig::default(),
        };
        generate_pyramid_auto(&src, &plan, &sink, &config, &NoopObserver).unwrap();
        let mut tiles = sink.tiles();
        tiles.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));

        assert_eq!(ref_tiles.len(), tiles.len());
        for (r, t) in ref_tiles.iter().zip(tiles.iter()) {
            assert_eq!(r.coord, t.coord);
            assert_eq!(r.data, t.data, "Tile data mismatch at {:?}", t.coord);
        }
    }

    #[test]
    fn auto_selects_streaming_for_small_budget() {
        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: 1_000, // Extremely small — forces streaming
            engine: EngineConfig::default(),
        };
        let result = generate_pyramid_auto(&src, &plan, &sink, &config, &NoopObserver).unwrap();

        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }

    #[test]
    fn concat_vertical_works() {
        let top = solid_raster(4, 2, 100);
        let bottom = solid_raster(4, 3, 200);
        let combined = concat_vertical(&top, &bottom).unwrap();
        assert_eq!(combined.width(), 4);
        assert_eq!(combined.height(), 5);
        // First 2 rows should be 100, next 3 should be 200
        let bpp = 3;
        let stride = 4 * bpp;
        for row in 0..2 {
            for byte in &combined.data()[row * stride..(row + 1) * stride] {
                assert_eq!(*byte, 100);
            }
        }
        for row in 2..5 {
            for byte in &combined.data()[row * stride..(row + 1) * stride] {
                assert_eq!(*byte, 200);
            }
        }
    }

    #[test]
    fn raster_strip_source_extracts_correctly() {
        let src = gradient_raster(100, 200);
        let strip_src = RasterStripSource::new(&src);
        assert_eq!(strip_src.width(), 100);
        assert_eq!(strip_src.height(), 200);

        let strip = strip_src.render_strip(50, 30).unwrap();
        assert_eq!(strip.width(), 100);
        assert_eq!(strip.height(), 30);

        // First pixel should match source at (0, 50)
        assert_eq!(strip.data()[0], 0); // x=0
        assert_eq!(strip.data()[1], 50); // y=50
        assert_eq!(strip.data()[2], 50); // x+y=50
    }

    #[test]
    fn streaming_emits_strip_rendered_events() {
        use crate::observe::CollectingObserver;

        let src = gradient_raster(512, 512);
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: 1_000, // Force streaming with small strips
            engine: EngineConfig::default(),
        };
        let observer = CollectingObserver::new();
        let strip_src = RasterStripSource::new(&src);
        generate_pyramid_streaming(&strip_src, &plan, &sink, &config, &observer).unwrap();

        let strip_events: Vec<_> = observer
            .events()
            .into_iter()
            .filter(|e| matches!(e, EngineEvent::StripRendered { .. }))
            .collect();
        assert!(!strip_events.is_empty(), "expected StripRendered events");

        // Verify sequential indexing and consistent total
        for (i, e) in strip_events.iter().enumerate() {
            if let EngineEvent::StripRendered {
                strip_index,
                total_strips,
            } = e
            {
                assert_eq!(*strip_index, i as u32);
                assert_eq!(*total_strips, strip_events.len() as u32);
            }
        }
    }

    #[test]
    fn streaming_odd_dimensions() {
        // Non-power-of-2 size
        let src = gradient_raster(500, 300);
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink = MemorySink::new();
        let config = StreamingConfig {
            memory_budget_bytes: 500_000,
            engine: EngineConfig::default(),
        };
        let strip_src = RasterStripSource::new(&src);
        let result =
            generate_pyramid_streaming(&strip_src, &plan, &sink, &config, &NoopObserver).unwrap();
        assert_eq!(result.tiles_produced, plan.total_tile_count());
    }
}
