//! Strip-based verify implementation.
//!
//! This module mirrors the byte-exact verify path in
//! [`crate::engine::run_verify`](crate::engine) without requiring the full
//! source raster to live in memory up front. The caller supplies a
//! [`StripSource`](crate::streaming::StripSource); the verify walker pulls
//! the top pyramid level strip-by-strip, assembles it into a
//! `canvas_width × canvas_height` raster, and then replays the monolithic
//! engine's level-by-level downscale / tile-extract / tile-compare loop.
//!
//! The *only* memory cost above that of the monolithic verify is the top
//! level itself — and that's an unavoidable cost of byte-exact verification
//! against on-disk tiles, because the lower levels must be produced by the
//! same downscale chain the engine used at generation time. What we avoid
//! is materialising the full source image ahead of time, which for large
//! PDFs or tiled TIFFs would otherwise dominate peak memory.

use std::path::PathBuf;
use std::time::Instant;

use crate::engine::{EngineConfig, EngineError, EngineResult, StageDurations};
use crate::observe::{EngineEvent, EngineObserver};
use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::resize;
use crate::sink::{SinkError, TileSink};
use crate::streaming::{StripSource, obtain_canvas_strip};

/// Candidate tile file extensions probed when looking for a tile on disk.
///
/// The sink's active format is not visible from this layer (the `TileSink`
/// trait object doesn't expose it), so we probe the extensions produced by
/// every [`TileFormat`](crate::sink::TileFormat) variant before declaring a
/// tile missing. This matches the behaviour of `engine::run_verify`.
const CANDIDATE_EXTS: [&str; 4] = ["raw", "png", "jpeg", "jpg"];

/// Verify an on-disk pyramid against a streaming source.
///
/// Walks every tile listed in `plan`, reads the corresponding file from the
/// checkpoint root (resolved via [`EngineConfig::checkpoint_root`] or the
/// sink's own root), and confirms it matches what the engine would produce
/// if it re-rendered the pyramid from `source`.
///
/// This is the streaming analogue of the byte-exact verify loop in
/// `engine::run_verify`. Behaviour is identical in every respect *except*
/// that the top-level raster is assembled on demand from
/// [`StripSource::render_strip`] calls rather than copied from a pre-loaded
/// [`Raster`]. Lower levels are produced by the same
/// [`resize::downscale_half`] chain the monolithic engine uses, guaranteeing
/// pixel-exact parity.
///
/// # Verification strategy per extension
///
/// * `raw` — byte-exact comparison against the regenerated tile. Any
///   mismatch (truncation, flipped byte, padding drift) is reported as
///   [`EngineError::ChecksumMismatch`].
/// * `png` / `jpeg` / `jpg` — existence check only. Encoded tiles cannot be
///   re-encoded bit-identically from fresh pixel data (encoder-state
///   nondeterminism), so deeper verification is deferred to the
///   manifest-checksum branch.
///
/// # Manifest checksums
///
/// If a `manifest.json` sits next to or inside the checkpoint root and
/// includes a `checksums.per_tile` table, every listed tile is re-hashed
/// with the manifest's declared algorithm and compared against the
/// recorded digest. A mismatch returns [`EngineError::ChecksumMismatch`].
///
/// # Events
///
/// Emits the same progression events as `run_verify`:
/// [`LevelStarted`](EngineEvent::LevelStarted) →
/// [`TileCompleted`](EngineEvent::TileCompleted) per tile →
/// [`LevelCompleted`](EngineEvent::LevelCompleted) per level →
/// [`Finished`](EngineEvent::Finished) once. Strip accumulation does *not*
/// emit [`StripRendered`](EngineEvent::StripRendered) — verify runs are
/// observationally equivalent to the monolithic path.
///
/// # Errors
///
/// * [`EngineError::VerifyRequiresOnDiskSink`] — neither
///   [`EngineConfig::checkpoint_root`] nor [`TileSink::checkpoint_root`]
///   yields a readable directory.
/// * [`EngineError::Sink`] wrapping [`SinkError::Other`] — a tile listed in
///   the plan is missing from disk.
/// * [`EngineError::Sink`] wrapping [`SinkError::Io`] — reading a tile
///   file's bytes failed.
/// * [`EngineError::ChecksumMismatch`] — the on-disk bytes differ from the
///   regenerated tile (raw) or from the manifest digest (any format).
/// * [`EngineError::Raster`] — the downscale chain or tile extraction
///   failed structurally (e.g. invalid extracted region).
///
/// # Returned `EngineResult`
///
/// Matches `run_verify`: `tiles_produced = 0`, `levels_processed =
/// plan.levels.len()`, `duration = started.elapsed()`, every other counter
/// zero. Verify never writes to the sink and never retries, so the
/// write-side counters have no meaningful value to report.
pub(crate) fn verify_from_strip_source(
    source: &dyn StripSource,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    config: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let started = Instant::now();
    let root_buf = resolve_root(config, sink).ok_or(EngineError::VerifyRequiresOnDiskSink)?;
    let root = root_buf.as_path();
    let bg = config.background_rgb;

    // Fast-fail when the on-disk checkpoint was produced from a different
    // plan. Mirrors the Monolithic raster_verify path so verify errors on
    // plan divergence surface structurally instead of as per-tile byte
    // mismatches.
    if let Some(meta) = crate::resume::JobCheckpoint::load(root)? {
        let expected = crate::resume::compute_plan_hash(plan);
        if meta.plan_hash != expected {
            return Err(EngineError::PlanHashMismatch {
                expected: meta.plan_hash,
                got: expected,
            });
        }
    }

    // ------------------------------------------------------------------
    // Phase 1: existence pass.
    //
    // Walk every plan tile and confirm *some* candidate extension resolves
    // to an on-disk file. This mirrors the first loop in `run_verify` and
    // gives fast feedback when the output directory is clearly wrong
    // (e.g. pointed at a stale run) before we spend time re-rendering.
    // ------------------------------------------------------------------
    for coord in plan.tile_coords() {
        if find_tile_on_disk(root, plan, coord).is_none() {
            return Err(EngineError::Sink(SinkError::Other(format!(
                "Verify: missing tile for coord {coord:?}"
            ))));
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: manifest-checksum branch.
    //
    // Copy of `run_verify`'s manifest handling: if a manifest.json is
    // present and records per-tile checksums, re-hash the on-disk bytes
    // and fail on the first mismatch.
    // ------------------------------------------------------------------
    if let Some(manifest) = read_manifest(root) {
        if let Some(checksums) = manifest.get("checksums") {
            let algo_str = checksums.get("algo").and_then(|v| v.as_str());
            let per_tile = checksums.get("per_tile").and_then(|v| v.as_object());
            if let (Some(algo_str), Some(per_tile)) = (algo_str, per_tile) {
                let algo = match algo_str {
                    "blake3" => Some(crate::manifest::ChecksumAlgo::Blake3),
                    "sha256" => Some(crate::manifest::ChecksumAlgo::Sha256),
                    _ => None,
                };
                if let Some(algo) = algo {
                    for (rel, expected) in per_tile {
                        let Some(expected_s) = expected.as_str() else {
                            continue;
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

    // ------------------------------------------------------------------
    // Phase 3: assemble the top-level raster from `source` strips.
    //
    // `obtain_canvas_strip` already encapsulates every layout concern
    // (Google padding, centring, DeepZoom/Xyz raw rows), so we defer to
    // it instead of re-implementing the centre-blit that `embed_in_canvas`
    // performs in the monolithic path. The helper returns canvas-space
    // strips, which makes the concatenated buffer byte-identical to
    // `embed_in_canvas(source, plan, bg)` even when centring is active.
    // ------------------------------------------------------------------
    let top_level_idx = plan.levels.len() - 1;
    let top = &plan.levels[top_level_idx];
    let format = source.format();
    let bpp = format.bytes_per_pixel();

    let cw = plan.canvas_width;
    let ch = plan.canvas_height;
    let dst_stride = cw as usize * bpp;
    let mut canvas = vec![0u8; dst_stride * ch as usize];

    // Strip height: `obtain_canvas_strip` contract only requires that
    // strips be requested in increasing Y with monotonically incrementing
    // heights. The exact height is a performance knob, not a correctness
    // one — a single strip per `2 × tile_size` rows keeps peak auxiliary
    // memory bounded to that same size, independent of canvas height.
    let strip_h = (2 * plan.tile_size).min(ch).max(1);
    let mut y: u32 = 0;
    while y < ch {
        let sh = strip_h.min(ch - y);
        let strip = obtain_canvas_strip(source, plan, y, sh, config)?;
        // `obtain_canvas_strip` guarantees the returned raster has width
        // `canvas_width` for Google / centred layouts. For DeepZoom/Xyz
        // the strip width equals the source width, which also equals
        // `canvas_width` in those layouts (no canvas padding is applied).
        debug_assert_eq!(strip.width(), cw);
        debug_assert_eq!(strip.format(), format);
        let strip_rows = strip.height() as usize;
        let src_row_bytes = strip.width() as usize * bpp;
        let src_stride = strip.stride();
        let data = strip.data();
        for row in 0..strip_rows {
            let src_start = row * src_stride;
            let dst_start = (y as usize + row) * dst_stride;
            canvas[dst_start..dst_start + src_row_bytes]
                .copy_from_slice(&data[src_start..src_start + src_row_bytes]);
        }
        y += sh;
    }

    let mut current = Raster::new(cw, ch, format, canvas)?;

    // Sanity check: the assembled raster must match the top plan level's
    // recorded dimensions, otherwise the downstream downscale chain will
    // diverge from what the engine wrote. For every layout libviprs
    // supports this is identity; guard it defensively.
    debug_assert_eq!(current.width(), top.width);
    debug_assert_eq!(current.height(), top.height);

    // ------------------------------------------------------------------
    // Phase 4: byte-exact verification, level-by-level, top to bottom.
    //
    // This is a direct transcription of `run_verify`'s second loop. The
    // downscale cadence, tile ordering, and event emission order are all
    // preserved; only the top-level source changes (strip-assembled
    // rather than `embed_in_canvas`-produced).
    // ------------------------------------------------------------------
    for level_idx in (0..plan.levels.len()).rev() {
        let level = &plan.levels[level_idx];
        if level_idx < top_level_idx {
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

                // `extract_tile_from_strip` with `strip_canvas_y = 0`
                // applied to a full-level raster is byte-equivalent to the
                // private `engine::extract_tile`: same rect projection,
                // same edge padding, same Google vs. DeepZoom branching.
                let expected =
                    crate::streaming::extract_tile_from_strip(&current, plan, coord, 0, bg)?;
                let expected_bytes = expected.data();

                let (abs, ext) = match find_tile_on_disk(root, plan, coord) {
                    Some(found) => found,
                    None => {
                        return Err(EngineError::Sink(SinkError::Other(format!(
                            "Verify: missing tile for coord {coord:?}"
                        ))));
                    }
                };

                let on_disk =
                    std::fs::read(&abs).map_err(|e| EngineError::Sink(SinkError::Io(e)))?;

                if ext == "raw" && on_disk != expected_bytes {
                    return Err(EngineError::ChecksumMismatch {
                        tile: coord,
                        expected: format!("{} bytes (raw)", expected_bytes.len()),
                        got: format!(
                            "{} bytes on disk differ from regenerated tile",
                            on_disk.len()
                        ),
                    });
                }
                // Encoded formats (png/jpeg) fall through: existence
                // check already passed, and fresh re-encoding is not
                // byte-stable, so we don't compare pixel data here.
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

// ---------------------------------------------------------------------------
// Internal helpers
//
// These reproduce logic that `engine.rs` keeps private. They are intentionally
// kept file-local so this module can be built without touching the engine.
// ---------------------------------------------------------------------------

/// Locate a tile on disk by probing the candidate extensions.
///
/// Returns `Some((absolute_path, extension))` for the first candidate that
/// exists as a regular file, or `None` if none match.
fn find_tile_on_disk(
    root: &std::path::Path,
    plan: &PyramidPlan,
    coord: TileCoord,
) -> Option<(PathBuf, &'static str)> {
    for ext in &CANDIDATE_EXTS {
        if let Some(rel) = plan.tile_path(coord, ext) {
            let abs = root.join(&rel);
            if abs.is_file() {
                return Some((abs, *ext));
            }
        }
    }
    None
}

/// Resolve the on-disk checkpoint root, preferring the config override.
///
/// Duplicates `engine::resolve_checkpoint_root`, which is `pub(crate)` in
/// the engine module but imported here via its full path instead of
/// re-export to keep the module graph explicit.
fn resolve_root(cfg: &EngineConfig, sink: &dyn TileSink) -> Option<PathBuf> {
    crate::engine::resolve_checkpoint_root(cfg, sink)
}

/// Parse a relative tile path (as stored in `manifest.json`) back into a
/// [`TileCoord`].
///
/// Accepts both DeepZoom-style paths (`"<level>/<col>_<row>.<ext>"`) and
/// XYZ/Google-style paths (`"<level>/<col>/<row>.<ext>"`). Windows path
/// separators are normalised. Returns `None` for any other shape.
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

/// Read the manifest JSON next to the checkpoint root.
///
/// Probes the sibling `<root>.manifest.json` first, then `<root>/manifest.json`
/// inside. Returns `None` if no file is found or if the contents don't parse
/// as JSON — a missing or corrupt manifest is not a verify error on its own;
/// it just means the checksum branch is skipped.
fn read_manifest(root: &std::path::Path) -> Option<serde_json::Value> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::NoopObserver;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::sink::{FsSink, TileFormat};
    use crate::streaming::RasterStripSource;

    /// Build a deterministic RGB gradient raster of size `w × h`.
    ///
    /// Using a gradient (rather than a solid fill) ensures that
    /// `downscale_half` produces distinct bytes at every level, so a
    /// corruption test that flips a single byte cannot be masked by
    /// blank-tile placeholders or incidentally-matching background fill.
    fn gradient(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h {
            for x in 0..w {
                let off = (y as usize * w as usize + x as usize) * bpp;
                data[off] = (x % 256) as u8;
                data[off + 1] = (y % 256) as u8;
                data[off + 2] = ((x.wrapping_add(y)) % 256) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// Generate a raw-format pyramid into a temp dir and return the
    /// pieces a verify test needs: the sink, the plan, and the source.
    fn build_raw_pyramid(
        dir: &std::path::Path,
        w: u32,
        h: u32,
        tile_size: u32,
    ) -> (FsSink, PyramidPlan, Raster) {
        let src = gradient(w, h);
        let plan = PyramidPlanner::new(w, h, tile_size, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let sink = FsSink::new(dir, plan.clone()).with_format(TileFormat::Raw);
        crate::engine::generate_pyramid_observed(
            &src,
            &plan,
            &sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .unwrap();
        (sink, plan, src)
    }

    /// Happy path: an intact raw pyramid verifies cleanly via the stream
    /// path, reports zero tiles produced, and flags every level as
    /// processed.
    #[test]
    fn stream_verify_happy_path_raw() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (sink, plan, src) = build_raw_pyramid(&out, 256, 256, 128);
        let strip_src = RasterStripSource::new(&src);

        let res = verify_from_strip_source(
            &strip_src,
            &plan,
            &sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .expect("verify should succeed on an untouched pyramid");

        assert_eq!(res.tiles_produced, 0, "verify must not write tiles");
        assert_eq!(res.levels_processed, plan.levels.len() as u32);
    }

    /// Missing-tile path: deleting a single on-disk tile must surface as
    /// a `SinkError::Other` wrapped in `EngineError::Sink`, matching the
    /// contract of `engine::run_verify`.
    #[test]
    fn stream_verify_reports_missing_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (sink, plan, src) = build_raw_pyramid(&out, 256, 256, 128);

        // Pick any tile from the plan — the first one iterated — and
        // delete every candidate-extension file that matches it on disk.
        let victim = plan
            .tile_coords()
            .next()
            .expect("plan has at least one tile");
        for ext in &CANDIDATE_EXTS {
            if let Some(rel) = plan.tile_path(victim, ext) {
                let abs = out.join(&rel);
                let _ = std::fs::remove_file(abs);
            }
        }

        let strip_src = RasterStripSource::new(&src);
        let err = verify_from_strip_source(
            &strip_src,
            &plan,
            &sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .expect_err("verify should fail when a tile is missing");

        match err {
            EngineError::Sink(SinkError::Other(msg)) => {
                assert!(
                    msg.starts_with("Verify: missing tile"),
                    "unexpected missing-tile message: {msg}"
                );
            }
            other => panic!("expected SinkError::Other for missing tile, got {other:?}"),
        }
    }

    /// Byte-corruption path: flipping a byte in one raw tile must be
    /// detected as `EngineError::ChecksumMismatch` with the offending
    /// `TileCoord` populated.
    #[test]
    fn stream_verify_detects_raw_corruption() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (sink, plan, src) = build_raw_pyramid(&out, 256, 256, 128);

        // Corrupt the first raw tile we can find — pick a tile at the
        // bottom (level 0) so we traverse multiple downscale iterations
        // before reaching it, exercising the full verify loop.
        let mut corrupted: Option<TileCoord> = None;
        'outer: for coord in plan.tile_coords() {
            if let Some(rel) = plan.tile_path(coord, "raw") {
                let abs = out.join(&rel);
                if let Ok(mut bytes) = std::fs::read(&abs) {
                    if !bytes.is_empty() {
                        bytes[0] ^= 0xFF;
                        std::fs::write(&abs, &bytes).unwrap();
                        corrupted = Some(coord);
                        break 'outer;
                    }
                }
            }
        }
        let corrupted = corrupted.expect("pyramid should contain at least one raw tile");

        let strip_src = RasterStripSource::new(&src);
        let err = verify_from_strip_source(
            &strip_src,
            &plan,
            &sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .expect_err("verify should fail on byte-corrupted raw tile");

        match err {
            EngineError::ChecksumMismatch { tile, .. } => {
                assert_eq!(tile, corrupted, "mismatch reported on wrong tile");
            }
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
    }
}
