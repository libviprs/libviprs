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
/// Fallback probe set used only when the sink does not pin a concrete format
/// (a transparent wrapper that returns `None` from
/// [`TileSink::content_format`](crate::sink::TileSink::content_format)). When
/// the format *is* known, [`active_candidate_exts`] narrows the probe to the
/// active format so Verify does not validate a stale sibling file (issue #139).
/// This matches the behaviour of `engine::raster_verify`.
const CANDIDATE_EXTS: [&str; 4] = ["raw", "png", "jpeg", "jpg"];

/// A [`StripSource`](crate::streaming::StripSource) (or the canvas-embedding
/// helper it is routed through) returned a strip whose layout does not match
/// the top-level canvas the verify walker is assembling.
///
/// The strip → canvas blit in Phase 3 trusts three invariants: the strip
/// width equals the canvas width, the strip pixel format equals the source
/// format, and the strip yields no more rows than were requested for the
/// current band. Every in-tree layout upholds them, but a third-party
/// `StripSource` implementation can violate them. These invariants were once
/// guarded only by `debug_assert_eq!`, which compiles out in release builds —
/// turning a violation into either a release-build panic (a slice index runs
/// past the canvas buffer) or silent cross-row corruption of the canvas
/// `Vec`. Surfacing the mismatch as a typed [`EngineError::Source`] keeps the
/// blit's load-bearing layout invariant enforced in every build profile
/// (issue #81).
#[derive(Debug)]
struct StripLayoutMismatch(String);

impl std::fmt::Display for StripLayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StripLayoutMismatch {}

/// Build the typed error returned when a strip violates a Phase-3 blit
/// invariant. Wraps the human-readable reason in [`EngineError::Source`],
/// attributing the failure to the strip source rather than to storage.
fn strip_layout_error(reason: String) -> EngineError {
    EngineError::Source(Box::new(StripLayoutMismatch(reason)))
}

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
pub fn verify_from_strip_source(
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
    // Probe only the sink's active on-disk format so a stale sibling file from
    // a previous run in a different format cannot pass Verify (issue #139).
    let active_exts = active_candidate_exts(sink);

    // Fast-fail when the on-disk checkpoint was produced from a different
    // plan. Mirrors the Monolithic raster_verify path so verify errors on
    // plan divergence surface structurally instead of as per-tile byte
    // mismatches.
    if let Some(meta) = crate::resume::JobCheckpoint::load(root)?
        && let Err(current) = crate::resume::verify_checkpoint_contract(&meta, plan, config, sink)
    {
        return Err(EngineError::PlanHashMismatch {
            expected: current,
            actual: meta.plan_hash,
        });
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
        if find_tile_on_disk(root, plan, coord, &active_exts).is_none() {
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
    if let Some(manifest) = read_manifest(root)
        && let Some(checksums) = manifest.get("checksums")
    {
        let algo_str = checksums.get("algo").and_then(|v| v.as_str());
        let per_tile = checksums.get("per_tile").and_then(|v| v.as_object());
        if let (Some(algo_str), Some(per_tile)) = (algo_str, per_tile) {
            // Route through the single shared parser. An unknown / future
            // / typo'd algorithm is a hard verification failure here, not
            // something to silently skip — otherwise a manifest stamped
            // with a bogus algo would pass with zero digests checked
            // (issue #95).
            let algo =
                crate::manifest::ChecksumAlgo::from_manifest_str(algo_str).ok_or_else(|| {
                    EngineError::Sink(SinkError::Other(format!(
                        "Verify: unknown checksum algorithm {algo_str:?} in manifest"
                    )))
                })?;
            {
                // A recorded tile that is gone from disk is a verification
                // failure, not something to skip — unless it is a
                // manifest-referenced blank whose content lives in
                // `_shared/` (issue #93).
                let blank_refs = manifest.get("blank_references").and_then(|v| v.as_object());
                for (rel, expected) in per_tile {
                    let Some(expected_s) = expected.as_str() else {
                        continue;
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
                            tile: coord_for_manifest_rel(plan, rel),
                            expected: expected_s.to_string(),
                            got,
                        });
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

    // Tile paths whose raw content is a 1-byte placeholder pointing at a
    // deduped payload under `_shared/` (issue #93). Legitimate markers even
    // when the regenerated tile is not itself blank.
    let blank_ref_paths: std::collections::HashSet<String> = read_manifest(root)
        .and_then(|m| {
            m.get("blank_references")
                .and_then(|v| v.as_object())
                .map(|o| o.keys().cloned().collect())
        })
        .unwrap_or_default();

    let cw = plan.canvas_width;
    let ch = plan.canvas_height;
    let dst_stride = cw as usize * bpp;
    // Allocate the top-level canvas through the checked/fallible raster
    // constructor: `canvas_width × canvas_height` derive from the (untrusted)
    // plan, so a `vec![0u8; …]` here would abort the process on an
    // over-budget or unsatisfiable size. `Raster::zeroed` enforces the
    // allocation budget and uses `try_reserve`, surfacing a typed
    // `EngineError::Raster` instead (issue #73).
    let mut canvas_raster = Raster::zeroed(cw, ch, format)?;
    let canvas = canvas_raster.data_mut();

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
        //
        // These are the load-bearing invariants of the row-by-row blit below:
        // a strip wider than the canvas, in a different pixel format, or with
        // more rows than this band requested would drive `dst_start +
        // src_row_bytes` past the canvas length (a release-build panic) or
        // spill rows into the neighbouring band (silent canvas corruption).
        // Slice bounds keep the operation memory-safe, but nothing else
        // enforces the layout contract for a third-party `StripSource`, so the
        // former `debug_assert_eq!` guards are promoted to hard checks that
        // return a typed error in every build profile (issue #81).
        let strip_rows = strip.height() as usize;
        if strip.width() != cw {
            return Err(strip_layout_error(format!(
                "strip source returned a strip {} px wide but the canvas is {} px \
                 wide; the strip-to-canvas blit requires them equal",
                strip.width(),
                cw
            )));
        }
        if strip.format() != format {
            return Err(strip_layout_error(format!(
                "strip source returned a strip in pixel format {:?} but the source \
                 format is {:?}",
                strip.format(),
                format
            )));
        }
        if strip_rows > sh as usize {
            return Err(strip_layout_error(format!(
                "strip source returned {strip_rows} rows for a band of at most {sh} \
                 rows; the surplus rows would overrun the canvas"
            )));
        }
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

    let current = canvas_raster;

    // Sanity check: the assembled raster must match the top plan level's
    // recorded dimensions, otherwise the downstream downscale chain will
    // diverge from what the engine wrote. For every layout libviprs
    // supports this is identity; guard it defensively.
    debug_assert_eq!(current.width(), top.width);
    debug_assert_eq!(current.height(), top.height);

    // ------------------------------------------------------------------
    // Phase 4: byte-exact verification, level-by-level, top to bottom.
    //
    // The walk skeleton is the shared `level_walk::walk_levels_down` (the
    // same one `raster_verify` and the live engines drive). The downscale
    // cadence, tile ordering, and event emission order are all preserved;
    // only the top-level source changes (strip-assembled rather than
    // `embed_in_canvas`-produced).
    // ------------------------------------------------------------------
    crate::level_walk::walk_levels_down::<EngineError, _, _, _, _>(
        current,
        plan.levels.len(),
        |_| Ok(()),
        // Step: the same tile op the live engines apply.
        |_, prev| Ok(resize::downscale_half(&prev)?),
        |level_idx, current| {
            let level = &plan.levels[level_idx];

            observer.on_event(EngineEvent::LevelStarted {
                level: level.level,
                width: level.width,
                height: level.height,
                tile_count: level.tile_count(),
            });

            for row in 0..level.rows {
                for col in 0..level.cols {
                    let coord = TileCoord::new(level_idx as u32, col, row);
                    observer.on_event(EngineEvent::tile_completed(coord));

                    // `extract_tile_from_strip` with `strip_canvas_y = 0`
                    // applied to a full-level raster is byte-equivalent to the
                    // private `engine::extract_tile`: same rect projection,
                    // same edge padding, same Google vs. DeepZoom branching.
                    let expected =
                        crate::streaming::extract_tile_from_strip(current, plan, coord, 0, bg)?;
                    let expected_bytes = expected.data();

                    let (abs, ext) = match find_tile_on_disk(root, plan, coord, &active_exts) {
                        Some(found) => found,
                        None => {
                            return Err(EngineError::Sink(SinkError::Other(format!(
                                "Verify: missing tile for coord {coord:?}"
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
                                .tile_path(coord, ext)
                                .is_some_and(|rel| blank_ref_paths.contains(&rel));
                            if !is_dedupe_ref
                                && !crate::engine::regenerated_tile_matches_marker(
                                    &expected, config,
                                )
                            {
                                return Err(EngineError::ChecksumMismatch {
                                    tile: coord,
                                    expected: "blank tile (placeholder marker)".to_string(),
                                    got: "regenerated tile is not blank".to_string(),
                                });
                            }
                        } else if on_disk != expected_bytes {
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
                    // Encoded formats (png/jpeg) fall through: existence
                    // check already passed, and fresh re-encoding is not
                    // byte-stable, so we don't compare pixel data here.
                }
            }

            observer.on_event(EngineEvent::LevelCompleted {
                level: level.level,
                tiles_produced: level.tile_count(),
            });
            Ok(())
        },
    )?;

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

/// The tile-file extensions to probe for the sink's active on-disk format.
///
/// A previous run in a different format can leave a stale sibling file (e.g. a
/// `.png` when this run writes `.raw`); probing every extension and taking the
/// first hit would validate that stale file and let a missing active-format
/// tile pass. When the sink pins a concrete format we probe only it; a
/// transparent wrapper that reports `None` falls back to every known extension
/// (issue #139).
fn active_candidate_exts(sink: &dyn TileSink) -> Vec<&'static str> {
    match sink.content_format() {
        Some(crate::sink::TileFormat::Jpeg { .. }) => vec!["jpeg", "jpg"],
        Some(fmt) => vec![fmt.extension()],
        None => CANDIDATE_EXTS.to_vec(),
    }
}

/// Locate a tile on disk by probing `exts` in order.
///
/// Returns `Some((absolute_path, extension))` for the first candidate that
/// exists as a regular file, or `None` if none match.
fn find_tile_on_disk(
    root: &std::path::Path,
    plan: &PyramidPlan,
    coord: TileCoord,
    exts: &[&'static str],
) -> Option<(PathBuf, &'static str)> {
    for ext in exts {
        if let Some(rel) = plan.tile_path(coord, ext) {
            let abs = root.join(&rel);
            if abs.is_file() {
                return Some((abs, *ext));
            }
        }
    }
    None
}

/// Resolve the [`TileCoord`] a manifest tile key refers to, so a checksum
/// mismatch is attributed to the offending tile instead of a fabricated
/// `TileCoord(0, 0, 0)` (or a col/row-transposed coordinate).
///
/// The authoritative match scans the plan for the coordinate whose
/// [`PyramidPlan::tile_path`] equals the key; this is exact for every layout,
/// including Google (`{level}/{row}/{col}`), which the layout-agnostic
/// [`parse_tile_rel_path`] would otherwise transpose. The structural parse is
/// kept only as a fallback for a foreign key the plan does not produce (issue
/// #139).
fn coord_for_manifest_rel(plan: &PyramidPlan, rel: &str) -> TileCoord {
    let normalized = rel.replace('\\', "/");
    for coord in plan.tile_coords() {
        for ext in CANDIDATE_EXTS {
            if plan.tile_path(coord, ext).is_some_and(|p| p == normalized) {
                return coord;
            }
        }
    }
    parse_tile_rel_path(rel).unwrap_or_else(|| TileCoord::new(0, 0, 0))
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
        if let Ok(bytes) = std::fs::read(&sibling)
            && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            return Some(v);
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

    /// Like [`build_raw_pyramid`] but attaches a manifest with a per-tile
    /// checksum table (BLAKE3) so the manifest-checksum branch of verify is
    /// exercised. Returns the same pieces plus the pyramid output dir so the
    /// caller can tamper the emitted `manifest.json`.
    fn build_raw_pyramid_with_checksums(
        dir: &std::path::Path,
        w: u32,
        h: u32,
        tile_size: u32,
    ) -> (FsSink, PyramidPlan, Raster) {
        use crate::checksum::ChecksumMode;
        use crate::manifest::{ChecksumAlgo, ManifestBuilder};
        let src = gradient(w, h);
        let plan = PyramidPlanner::new(w, h, tile_size, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let sink = FsSink::new(dir, plan.clone())
            .with_format(TileFormat::Raw)
            .with_manifest(ManifestBuilder::new())
            .with_checksums(ChecksumMode::EmitOnly, ChecksumAlgo::Blake3);
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

    /// Overwrite the `checksums.algo` field of every emitted `manifest.json`
    /// (sibling `<dir>.manifest.json` and the in-dir copy) with `algo`,
    /// leaving the `per_tile` table intact.
    fn rewrite_manifest_algo(dir: &std::path::Path, algo: &str) {
        let mut targets = Vec::new();
        if let (Some(parent), Some(stem)) = (dir.parent(), dir.file_name()) {
            let mut name = stem.to_os_string();
            name.push(".manifest.json");
            targets.push(parent.join(name));
        }
        targets.push(dir.join("manifest.json"));

        let mut rewritten = 0;
        for path in targets {
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v.get_mut("checksums")
                .and_then(|c| c.as_object_mut())
                .expect("manifest must record a checksums table")
                .insert("algo".into(), serde_json::json!(algo));
            std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
            rewritten += 1;
        }
        assert!(rewritten > 0, "no manifest.json was found to rewrite");
    }

    /// Issue #95: a manifest stamped with an unknown checksum algorithm must
    /// FAIL streaming verify, not be silently skipped. Before the fix the
    /// manifest-checksum branch mapped an unrecognised algo to `None` and
    /// skipped the entire per-tile digest phase, so an intact pyramid whose
    /// manifest was re-stamped with a bogus algo reported success with zero
    /// digests checked.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn stream_verify_rejects_unknown_algo() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (sink, plan, src) = build_raw_pyramid_with_checksums(&out, 256, 256, 128);

        // Re-stamp the manifest with a future / typo'd algorithm the engine
        // does not support, keeping the (correct) per-tile digests in place.
        rewrite_manifest_algo(&out, "totally-bogus-algo");

        let strip_src = RasterStripSource::new(&src);
        let err = verify_from_strip_source(
            &strip_src,
            &plan,
            &sink,
            &EngineConfig::default(),
            &NoopObserver,
        )
        .expect_err("verify must reject a manifest with an unknown checksum algorithm");

        match err {
            EngineError::Sink(SinkError::Other(msg)) => assert!(
                msg.contains("unknown checksum algorithm"),
                "unexpected error message: {msg}"
            ),
            other => panic!("expected SinkError::Other for unknown algo, got {other:?}"),
        }
    }

    /// Happy path: an intact raw pyramid verifies cleanly via the stream
    /// path, reports zero tiles produced, and flags every level as
    /// processed.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
                if let Ok(mut bytes) = std::fs::read(&abs)
                    && !bytes.is_empty()
                {
                    bytes[0] ^= 0xFF;
                    std::fs::write(&abs, &bytes).unwrap();
                    corrupted = Some(coord);
                    break 'outer;
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

    /// Raw-format strip Verify must recognize the 1-byte `BLANK_TILE_MARKER`
    /// that `BlankTileStrategy::Placeholder` writes, rather than
    /// byte-comparing it against the regenerated full tile (issue #94). A
    /// fully-uniform source yields an all-placeholder pyramid.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn stream_verify_accepts_raw_placeholder_markers() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (w, h, ts) = (256u32, 256u32, 128u32);
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let src = Raster::new(
            w,
            h,
            PixelFormat::Rgb8,
            vec![7u8; w as usize * h as usize * bpp],
        )
        .unwrap();
        let plan = PyramidPlanner::new(w, h, ts, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let sink = FsSink::new(&out, plan.clone()).with_format(TileFormat::Raw);
        // Match the background to the solid fill so every edge-padded tile is
        // also uniform, producing an all-placeholder pyramid.
        let mut cfg = EngineConfig::default()
            .with_blank_tile_strategy(crate::engine::BlankTileStrategy::Placeholder);
        cfg.background_rgb = [7, 7, 7];
        crate::engine::generate_pyramid_observed(&src, &plan, &sink, &cfg, &NoopObserver).unwrap();

        // Setup sanity: at least the first tile is a 1-byte marker on disk.
        let first = plan.tile_coords().next().unwrap();
        let rel = plan.tile_path(first, "raw").unwrap();
        let on_disk = std::fs::read(out.join(&rel)).unwrap();
        assert_eq!(
            on_disk,
            vec![crate::sink::BLANK_TILE_MARKER],
            "placeholder run should write a 1-byte marker on disk"
        );

        let strip_src = RasterStripSource::new(&src);
        verify_from_strip_source(&strip_src, &plan, &sink, &cfg, &NoopObserver)
            .expect("raw placeholder pyramid must verify via strip path");
    }

    /// A [`StripSource`] whose `render_strip` deliberately violates the
    /// strip → canvas blit contract. The trait's `width` / `height` / `format`
    /// accessors report the true, plan-matching dimensions (so plan validation
    /// and the strip-request arithmetic behave normally), while `render_strip`
    /// returns a raster that is either wider than the canvas, in the wrong
    /// pixel format, or taller than the requested band. This is exactly the
    /// shape of a buggy or adversarial third-party source. Issue #81.
    struct MalformedStripSource {
        w: u32,
        h: u32,
        format: PixelFormat,
        strip_format: PixelFormat,
        extra_width: u32,
        extra_rows: u32,
    }

    impl MalformedStripSource {
        fn new(w: u32, h: u32) -> Self {
            Self {
                w,
                h,
                format: PixelFormat::Rgb8,
                strip_format: PixelFormat::Rgb8,
                extra_width: 0,
                extra_rows: 0,
            }
        }
    }

    impl StripSource for MalformedStripSource {
        fn render_strip(&self, _y: u32, height: u32) -> Result<Raster, EngineError> {
            let w = self.w + self.extra_width;
            let h = height + self.extra_rows;
            let bpp = self.strip_format.bytes_per_pixel();
            let data = vec![0u8; w as usize * h as usize * bpp];
            Raster::new(w, h, self.strip_format, data).map_err(EngineError::from)
        }
        fn width(&self) -> u32 {
            self.w
        }
        fn height(&self) -> u32 {
            self.h
        }
        fn format(&self) -> PixelFormat {
            self.format
        }
    }

    /// Assert that streaming verify rejects `bad` with the typed strip-layout
    /// error rather than panicking or corrupting the canvas. Builds a valid
    /// raw pyramid up front (so the existence pass passes) and then replays
    /// verify against the non-conforming source.
    fn assert_strip_layout_rejected(bad: MalformedStripSource, expect_substr: &str) {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiles");
        let (sink, plan, _src) = build_raw_pyramid(&out, bad.w, bad.h, 128);

        let err =
            verify_from_strip_source(&bad, &plan, &sink, &EngineConfig::default(), &NoopObserver)
                .expect_err("verify must reject a strip that violates the canvas layout");

        match err {
            EngineError::Source(inner) => {
                let msg = inner.to_string();
                assert!(
                    msg.contains(expect_substr),
                    "unexpected strip-layout error message: {msg}"
                );
            }
            other => {
                panic!("expected EngineError::Source for strip layout mismatch, got {other:?}")
            }
        }
    }

    /// Issue #81: a strip whose width exceeds the canvas width must surface a
    /// typed error in release, not a slice-bounds panic / silent corruption
    /// that the old `debug_assert_eq!` masked away outside debug builds.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn stream_verify_rejects_overwide_strip() {
        let mut bad = MalformedStripSource::new(256, 256);
        bad.extra_width = 1;
        assert_strip_layout_rejected(bad, "wide");
    }

    /// Issue #81: a strip in a different pixel format than the source must be
    /// rejected — a wider bpp would read/write a different number of bytes per
    /// row and desynchronise the whole canvas.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn stream_verify_rejects_wrong_format_strip() {
        let mut bad = MalformedStripSource::new(256, 256);
        // Same declared width, different (wider) pixel format.
        bad.strip_format = PixelFormat::Rgba8;
        assert_strip_layout_rejected(bad, "pixel format");
    }

    /// Issue #81: a strip that returns more rows than the band requested must
    /// be rejected. The surplus rows would drive `dst_start` past the canvas
    /// buffer on the final band (a release panic) or overwrite the following
    /// band's rows.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn stream_verify_rejects_overtall_strip() {
        let mut bad = MalformedStripSource::new(256, 256);
        bad.extra_rows = 1;
        assert_strip_layout_rejected(bad, "rows");
    }
}
