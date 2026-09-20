//! Verify-mode entry points.
//!
//! Three flavours share one namespace so `EngineBuilder` can route verify
//! runs without the rest of the crate hard-coding which helper to call. Two of
//! them re-render the pyramid from the source and compare; the third reads the
//! finished pyramid back and checks it against the plan.
//!
//! * [`raster_verify`] — walks every level via `downscale_half` against
//!   the full in-memory source raster. Used when the caller has a
//!   `&Raster` and picks `EngineKind::Monolithic`.
//! * [`verify_from_strip_source`]
//!   — strip-driven verify for pull-based sources or when the caller
//!   explicitly picks `EngineKind::Streaming` / `EngineKind::MapReduce`.
//! * [`pyramid_verify`] reads the pyramid back through [`PyramidReader`], for
//!   a sink that can open its own output. Used when the output is not a tree
//!   of files, so there is nothing for the two above to stat.
//!
//! All three emit the same `LevelStarted` / `TileCompleted` / `LevelCompleted`
//! / `Finished` event stream so observers see verify runs as first-class, and
//! all three report `tiles_produced: 0`.

use std::time::Instant;

use crate::engine::{EngineError, EngineResult, StageDurations};
use crate::observe::{EngineEvent, EngineObserver};
use crate::planner::{PyramidPlan, TileCoord};
use crate::pyramid_reader::{PyramidReadError, PyramidReader};
use crate::sink::{SinkError, TileSink};

pub use crate::engine::raster_verify;
pub use crate::stream_verify::verify_from_strip_source;

/// A verify refusal that is about the pyramid rather than about a failure to
/// read it.
///
/// The wording of the missing-tile case is deliberately the same sentence
/// [`raster_verify`] produces, so a caller's message does not change depending
/// on where the tiles went.
fn refused(message: String) -> EngineError {
    EngineError::Sink(SinkError::Other(message))
}

/// A reader that could not answer, as distinct from a pyramid that is wrong.
fn unreadable(error: PyramidReadError) -> EngineError {
    EngineError::Sink(SinkError::PyramidRead(error))
}

/// Verify a finished pyramid against the plan that produced it, by reading it
/// back (issue #1122).
///
/// This is the verify for a sink whose output is not a directory of files.
/// [`raster_verify`] resolves a checkpoint root and stats
/// `root.join(plan.tile_path(coord, ext))` for every planned coordinate, which
/// a single-file archive has no seam to enter; the engine asks the sink for a
/// [`PyramidReader`] first ([`TileSink::open_pyramid_reader`]) and comes here
/// when it gets one. It is a sibling of the directory walk and not a
/// replacement for it: `raster_verify` re-renders from the source and compares
/// bytes, which is more than a reader can offer and is what keeps the tree's
/// verify contract where it is.
///
/// Writes nothing, keeps no checkpoint, and reports `tiles_produced: 0`.
///
/// # Four checks, because one of them is a trap
///
/// The implementation this function is not is `reader.tile(coord).is_some()`
/// for every planned coordinate. It looks like the whole job and it is green
/// for at least three pyramids that are wrong:
///
/// 1. **A PMTiles entry carries a `run_length`** over Hilbert-ordered id
///    space, so a single entry answers `Some` for a stretch of coordinates
///    nobody wrote. An archive of one blank tile with one long run answers
///    `Some` at every coordinate in its range, at every zoom.
/// 2. **A zero-length payload is `Some(vec![])`**, which is not a tile.
/// 3. **A plan narrower than the pyramid** resolves every coordinate it asks
///    about and goes green while extra levels sit in the file. No
///    per-coordinate sweep can ever see this, because the sweep only asks
///    about coordinates the plan names.
///
/// So there are four checks and each one catches something the others cannot:
///
/// * **the self-description against the plan** ([`PyramidReader::describe`]):
///   the level range, and the tile size, layout and encoding the pyramid
///   recorded for itself. This is the archive's stand-in for
///   `resume::verify_checkpoint_contract` (crate-private, so no link),
///   and it is the check an implementation drops first, because the sweep
///   passes without it: a 256-pixel tile at `0/0/0` is a perfectly readable
///   tile whatever the plan meant by a tile;
/// * **the structural self-check** ([`PyramidReader::self_check`]), run before
///   the sweep and read through the storage's own "no findings" verdict, so
///   that a walk which gave up early cannot report clean;
/// * **the addressed-tile count against the plan's**, which is the only check
///   that sees case 3, and it is an equality rather than a `>=` on purpose.
///   The short direction is normally reported by the sweep, which can name the
///   coordinate; the wide direction has nowhere else to be caught;
/// * **the count of what was actually probed**, asserted non-zero, because a
///   verify over an empty coordinate set agrees most perfectly of all.
///
/// The optional fifth is a decode per tile. It is not done here: it closes
/// case 2 more thoroughly than the empty-payload check below, at the cost of
/// decoding the whole pyramid, and the four above already refuse every pyramid
/// that could otherwise pass.
///
/// # Errors
///
/// [`EngineError::Sink`] wrapping [`SinkError::PyramidRead`] when the reader
/// could not answer, and wrapping [`SinkError::Other`] when it answered and
/// the answer disagrees with the plan.
pub fn pyramid_verify(
    reader: &dyn PyramidReader,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let started = Instant::now();

    // Check 1. Structural soundness first, because everything below reads the
    // same storage: a sweep over a pyramid whose index does not add up
    // produces confident per-coordinate answers out of a structure that is
    // already known to be wrong.
    reader.self_check().map_err(unreadable)?;

    // Check 2. What the pyramid says it is, against what the plan asked for.
    describe_matches_the_plan(reader, plan, sink)?;

    // Check 3. The sweep, top level first, which is the order `raster_verify`
    // walks in and therefore the order an observer already expects.
    let mut probed: u64 = 0;
    for level_idx in (0..plan.levels.len()).rev() {
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
                let bytes = reader
                    .tile(coord)
                    .map_err(unreadable)?
                    .ok_or_else(|| refused(format!("Verify: missing tile for coord {coord:?}")))?;
                // Not a decode, and not trying to be one. It is the single
                // payload length that cannot be an image in any encoding this
                // crate writes, and it is exactly what `is_some()` waves
                // through.
                if bytes.is_empty() {
                    return Err(refused(format!(
                        "Verify: the tile at {coord:?} is stored with no bytes at all, \
                         which is not an encoded tile in any format"
                    )));
                }
                probed += 1;
                observer.on_event(EngineEvent::tile_completed(coord));
            }
        }
        observer.on_event(EngineEvent::LevelCompleted {
            level: level.level,
            tiles_produced: level.tile_count(),
        });
    }

    // Check 4. The counts, in both directions and from two independent
    // derivations: `probed` is what the loop above resolved and `planned` is
    // what the plan's own iterator yields, so a sweep that walked a different
    // set of coordinates than the plan describes is caught here rather than
    // reported as a clean run.
    let planned = plan.tile_coords().count() as u64;
    if probed != planned {
        return Err(refused(format!(
            "Verify: the sweep resolved {probed} coordinates and the plan has \
             {planned}; the walk and the plan disagree about which tiles exist"
        )));
    }
    if probed == 0 {
        return Err(refused(
            "Verify: nothing was checked. A verify over an empty coordinate set \
             agrees with everything, so it is reported as a failure rather than \
             as a pass"
                .to_string(),
        ));
    }

    // And the direction the sweep is blind to. A pyramid addressing MORE than
    // the plan answered every question it was asked and is still not the
    // pyramid this plan produced.
    let addressed = reader.addressed_tiles().map_err(unreadable)?;
    if addressed != planned {
        return Err(refused(format!(
            "Verify: the pyramid addresses {addressed} tiles and the plan has \
             {planned}; a pyramid holding tiles the plan does not name is not \
             the pyramid this plan produced"
        )));
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

/// The pyramid's own account of itself, against the plan and the sink's
/// configured encoding.
///
/// A pyramid that cannot say how it was generated is refused rather than
/// verified loosely. The alternative reads as tolerant and is not: it would
/// report a clean verify for a foreign pyramid nobody's plan produced, which
/// is the loudest possible disagreement and the one a caller most wants
/// named. The encoding is the one exception, because
/// [`TileSink::content_format`] is legitimately `None` for a sink that does
/// not commit to a format, and there is then nothing to compare against
/// rather than something being withheld.
fn describe_matches_the_plan(
    reader: &dyn PyramidReader,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
) -> Result<(), EngineError> {
    let described = reader.describe().map_err(unreadable)?;

    let plan_min = plan
        .levels
        .iter()
        .map(|level| level.level)
        .min()
        .ok_or_else(|| refused("Verify: the plan has no levels to check".to_string()))?;
    let plan_max = plan
        .levels
        .iter()
        .map(|level| level.level)
        .max()
        .unwrap_or(plan_min);
    if (described.min_level, described.max_level) != (plan_min, plan_max) {
        return Err(refused(format!(
            "Verify: the pyramid covers levels {}..={} and the plan covers \
             {plan_min}..={plan_max}",
            described.min_level, described.max_level
        )));
    }

    match described.tile_size {
        Some(size) if size == plan.tile_size => {}
        Some(size) => {
            return Err(refused(format!(
                "Verify: the pyramid was generated at tile size {size} and the \
                 plan asks for {}",
                plan.tile_size
            )));
        }
        None => {
            return Err(refused(
                "Verify: the pyramid does not record the tile size it was \
                 generated at, so there is nothing to check this plan against"
                    .to_string(),
            ));
        }
    }

    match described.layout {
        Some(layout) if layout == plan.layout => {}
        Some(layout) => {
            return Err(refused(format!(
                "Verify: the pyramid was generated with {layout:?} layout and \
                 the plan asks for {:?}",
                plan.layout
            )));
        }
        None => {
            return Err(refused(
                "Verify: the pyramid does not record the layout it was generated \
                 with, so there is nothing to check this plan against"
                    .to_string(),
            ));
        }
    }

    match (described.format, sink.content_format()) {
        (_, None) => {}
        (Some(stored), Some(configured)) if stored == configured => {}
        (Some(stored), Some(configured)) => {
            return Err(refused(format!(
                "Verify: the pyramid stores {stored:?} tiles and this run is \
                 configured for {configured:?}"
            )));
        }
        (None, Some(configured)) => {
            return Err(refused(format!(
                "Verify: the pyramid does not record what its tiles are encoded \
                 as, and this run is configured for {configured:?}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::NoopObserver;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::pyramid_reader::PyramidDescription;
    use crate::sink::{Tile, TileFormat};

    /// A pyramid that answers whatever the test wants it to.
    ///
    /// Three of the branches in [`pyramid_verify`] cannot be reached with an
    /// archive this crate is able to write: our own PMTiles writer refuses a
    /// zero-length payload by name, always records `vnd.libviprs`, and always
    /// knows how many tiles it addressed. A reader that does none of those is
    /// not hypothetical (a foreign archive is exactly that, and our validator
    /// does not flag a zero-length entry), and a double is the honest way to
    /// reach the branches without hand-assembling a hostile file.
    struct FakePyramid {
        description: PyramidDescription,
        /// What `tile` answers for every coordinate.
        stored: Option<Vec<u8>>,
        /// What `addressed_tiles` answers, or `None` for a reader that cannot
        /// count.
        addressed: Option<u64>,
    }

    impl PyramidReader for FakePyramid {
        fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
            Ok(self.description.clone())
        }

        fn tile(&self, _coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError> {
            Ok(self.stored.clone())
        }

        fn addressed_tiles(&self) -> Result<u64, PyramidReadError> {
            match self.addressed {
                Some(count) => Ok(count),
                None => Err(PyramidReadError::NoDescription(
                    "this double cannot count".to_string(),
                )),
            }
        }
    }

    /// A terminal sink that pins its encoding, which is what a real one that
    /// offers a reader does.
    struct PngSink;

    impl TileSink for PngSink {
        fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
            Ok(())
        }
        fn finish(&self) -> Result<(), SinkError> {
            Ok(())
        }
        fn content_format(&self) -> Option<TileFormat> {
            Some(TileFormat::Png)
        }
    }

    fn plan() -> PyramidPlan {
        PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz)
            .expect("a square plan is valid")
            .plan()
    }

    fn description(plan: &PyramidPlan) -> PyramidDescription {
        PyramidDescription {
            min_level: 0,
            max_level: plan.levels.len() as u32 - 1,
            tile_size: Some(plan.tile_size),
            layout: Some(plan.layout),
            format: Some(TileFormat::Png),
        }
    }

    fn verify(reader: &FakePyramid, plan: &PyramidPlan) -> Result<EngineResult, EngineError> {
        pyramid_verify(reader, plan, &PngSink, &NoopObserver)
    }

    /// A tile stored with no bytes is `Some(vec![])`, and it is not a tile.
    ///
    /// This is the second half of the `is_some()` trap: the coordinate
    /// resolves, the count adds up, the structure is sound, and what comes
    /// back decodes to nothing. The control beside it is the same pyramid with
    /// bytes in it, so the refusal is about the payload rather than about the
    /// double.
    #[test]
    fn a_tile_stored_with_no_bytes_is_refused_although_it_resolves() {
        let plan = plan();
        let planned = plan.tile_coords().count() as u64;

        let sound = FakePyramid {
            description: description(&plan),
            stored: Some(vec![0x89, b'P', b'N', b'G']),
            addressed: Some(planned),
        };
        verify(&sound, &plan).expect("the control verifies");

        let hollow = FakePyramid {
            description: description(&plan),
            stored: Some(Vec::new()),
            addressed: Some(planned),
        };
        let message = verify(&hollow, &plan)
            .expect_err("a zero-length payload is not a tile")
            .to_string();
        assert!(
            message.contains("no bytes"),
            "the refusal must say what is wrong with the tile, got: {message}"
        );
    }

    /// A reader that cannot count the coordinates it addresses cannot be
    /// verified against a plan.
    ///
    /// The count is the only check that sees a pyramid holding more than the
    /// plan asked for, so a reader without one gets a refusal rather than a
    /// verify with that check quietly dropped. Everything else about this
    /// pyramid is correct, which is the point: it is refused for the one thing
    /// it cannot answer.
    #[test]
    fn a_reader_that_cannot_count_its_tiles_cannot_verify() {
        let plan = plan();
        let uncountable = FakePyramid {
            description: description(&plan),
            stored: Some(vec![1, 2, 3]),
            addressed: None,
        };
        match verify(&uncountable, &plan) {
            Err(EngineError::Sink(SinkError::PyramidRead(PyramidReadError::NoDescription(_)))) => {}
            other => panic!("a verify with no count is not a verify, got {other:?}"),
        }
    }

    /// A pyramid that does not record how it was generated is refused rather
    /// than verified loosely.
    ///
    /// Reading as tolerant here would mean reporting a clean verify for a
    /// pyramid nobody's plan produced, which is the loudest possible
    /// disagreement and the one a caller most wants named.
    #[test]
    fn a_pyramid_that_cannot_say_how_it_was_generated_is_refused() {
        let plan = plan();
        let mut description = description(&plan);
        description.tile_size = None;
        let anonymous = FakePyramid {
            description,
            stored: Some(vec![1, 2, 3]),
            addressed: Some(plan.tile_coords().count() as u64),
        };
        let message = verify(&anonymous, &plan)
            .expect_err("a pyramid with no generation record cannot be checked")
            .to_string();
        assert!(
            message.contains("tile size"),
            "the refusal must name what the pyramid could not say, got: {message}"
        );
    }

    /// The count check is an equality, and the direction nothing else catches
    /// is the one where the pyramid is bigger than the plan.
    #[test]
    fn a_pyramid_addressing_more_than_the_plan_is_refused() {
        let plan = plan();
        let planned = plan.tile_coords().count() as u64;
        let wide = FakePyramid {
            description: description(&plan),
            stored: Some(vec![1, 2, 3]),
            addressed: Some(planned + 1),
        };
        let message = verify(&wide, &plan)
            .expect_err("a pyramid wider than the plan cannot verify")
            .to_string();
        assert!(
            message.contains(&(planned + 1).to_string()) && message.contains(&planned.to_string()),
            "the refusal must name both counts, got: {message}"
        );
    }
}
