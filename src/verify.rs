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
use crate::sink::{SinkError, TileFormat};

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

/// What a [`pyramid_verify`] run's per-tile probe actually established
/// (issue #1130).
///
/// The sweep asks two things of every coordinate the plan names: that the
/// pyramid holds a tile there, and that what it holds is not zero bytes. Both
/// answers are in the storage's index, and reading the payload as well proves
/// exactly one more thing, that the bytes at that offset come back.
///
/// That extra thing is worth the whole archive sometimes and not others, so
/// the run picks and then says which it picked. This enum is that sentence. It
/// is here rather than left implicit because a verify that read every byte and
/// a verify that read none of them both return `Ok`, and a caller deciding how
/// much to trust a green run needs to know which one it got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TileEvidence {
    /// Every tile's stored bytes came off the storage, so they are reachable
    /// as well as present and non-empty.
    ///
    /// This is what a run gets when the structural walk could not bound the
    /// index's offsets against a size the storage reported, and when the
    /// backend has no structural walk at all, which is the loose-file tree.
    PayloadsRead,
    /// Only each tile's stored length was taken, out of an index the
    /// structural walk had already bounds-checked against a reported size.
    ///
    /// Reachability is not given up here, it is established once for every
    /// entry at once instead of one payload at a time. What the run gives up
    /// is `bytes_read`, which is `0`, because it did not read any.
    LengthsFromTheIndex,
}

/// Verify a finished pyramid against the plan that produced it, by reading it
/// back (issue #1122).
///
/// This is the verify for a sink whose output is not a directory of files.
/// [`raster_verify`] resolves a checkpoint root and stats
/// `root.join(plan.tile_path(coord, ext))` for every planned coordinate, which
/// a single-file archive has no seam to enter; the engine asks the sink for a
/// [`PyramidReader`] first ([`crate::sink::TileSink::open_pyramid_reader`]) and comes here
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
/// * **the structural walk** ([`PyramidReader::structural_summary`]), run
///   before the sweep and read through the storage's own "no findings"
///   verdict, so that a walk which gave up early cannot report clean;
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
/// # What the sweep reads, and what that buys (issue #1130)
///
/// The sweep used to call [`PyramidReader::tile`] for every planned
/// coordinate and use the result only for `bytes.is_empty()`, then drop the
/// `Vec`. Verifying a 21851-tile pyramid therefore read the whole archive off
/// storage to learn 21851 numbers the index was already carrying, and over the
/// ranged transport #1121 opened that is one round trip per tile, serially.
///
/// So it takes [`PyramidReader::tile_len`] instead, but only when the
/// storage's own walk earned it. Reading a payload does prove one thing a
/// length cannot, that the bytes at that offset are reachable, and that is
/// redundant only when the structural walk bounds-checked every entry against
/// a size the storage actually reported. It is **not** redundant when the
/// backend could not say how large it is, which is the streaming case, nor for
/// a backend with no structure to walk at all.
/// [`StructuralSummary`](crate::pyramid_reader::StructuralSummary) carries that
/// answer and the run reports which guarantee it got as
/// [`EngineResult::tile_evidence`].
///
/// The structural walk itself happens once. `self_check` and
/// `addressed_tiles` were two questions about one walk, asked ten lines apart
/// under a run lock that guarantees the archive cannot change between them, so
/// this asks [`PyramidReader::structural_summary`] once and reads both answers
/// off it. `self_check` is deprecated as a result: nothing calls it, and an
/// override of it would be a walk that silently stopped running.
///
/// # Errors
///
/// [`EngineError::Sink`] wrapping [`SinkError::PyramidRead`] when the reader
/// could not answer, and wrapping [`SinkError::Other`] when it answered and
/// the answer disagrees with the plan.
pub fn pyramid_verify(
    reader: &dyn PyramidReader,
    plan: &PyramidPlan,
    configured_format: Option<TileFormat>,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    let started = Instant::now();

    // Check 1. Structural soundness first, because everything below reads the
    // same storage: a sweep over a pyramid whose index does not add up
    // produces confident per-coordinate answers out of a structure that is
    // already known to be wrong.
    //
    // This is the only structural walk in the function. The addressed count at
    // the bottom comes off the same summary rather than out of a second one.
    let structure = reader.structural_summary().map_err(unreadable)?;

    // Check 2. What the pyramid says it is, against what the plan asked for.
    describe_matches_the_plan(reader, plan, configured_format)?;

    // Which probe the sweep uses, decided once for the run rather than per
    // tile, because it is a property of the walk that already happened and not
    // of any coordinate.
    let evidence = if structure.offsets_bounded {
        TileEvidence::LengthsFromTheIndex
    } else {
        TileEvidence::PayloadsRead
    };

    // Check 3. The sweep, top level first, which is the order `raster_verify`
    // walks in and therefore the order an observer already expects.
    let mut probed: u64 = 0;
    let mut bytes_read: u64 = 0;
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
                let coord = TileCoord::new(level.level, col, row);
                let stored = match evidence {
                    TileEvidence::LengthsFromTheIndex => {
                        reader.tile_len(coord).map_err(unreadable)?
                    }
                    TileEvidence::PayloadsRead => {
                        let length = reader
                            .tile(coord)
                            .map_err(unreadable)?
                            .map(|bytes| bytes.len() as u64);
                        // Counted here rather than below because this is the
                        // only arm that reads a byte. A run that took lengths
                        // off the index reports `bytes_read: 0`, which is the
                        // truth: it did not read any.
                        bytes_read += length.unwrap_or(0);
                        length
                    }
                };
                let length = stored
                    .ok_or_else(|| refused(format!("Verify: missing tile for coord {coord:?}")))?;
                // Not a decode, and not trying to be one. It is the single
                // payload length that cannot be an image in any encoding this
                // crate writes, and it is exactly what `is_some()` waves
                // through.
                if length == 0 {
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

    // Check 4. The counts, and only the ones that can fail.
    //
    // There used to be a `probed != planned` check here, described as "two
    // independent derivations". It was neither independent nor able to fail:
    // `probed` counts one increment per cell of `plan.levels`, and `planned`
    // was `plan.tile_coords().count()`, which sums the same rows times columns
    // over the same vector. No input makes them differ, so the branch was dead
    // while reading as the thing that catches a sweep walking the wrong
    // coordinates. A count could not have caught that anyway: counts are blind
    // to permutations, which is the same argument the migration module makes
    // about tile-id sets.
    //
    // What replaces it is structural rather than arithmetic. The sweep now
    // builds its coordinate from `level.level`, the identical field
    // `plan.tile_coords()` reads, so the two cannot name different tiles. And
    // the check below compares against a number that comes out of the
    // ARCHIVE rather than out of the plan, which is what independent was
    // supposed to mean.
    let planned = plan.total_tile_count();
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
    //
    // Off the summary when the walk counted, and asked directly when it did
    // not. That second arm is not a fallback for a backend that cannot count:
    // it is the backend that counts without walking, which is every reader
    // written before #1130, and treating its `None` here as "cannot count"
    // would drop this check for a reader that answers it perfectly well.
    let addressed = match structure.addressed_tiles {
        Some(addressed) => addressed,
        None => reader.addressed_tiles().map_err(unreadable)?,
    };
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
        bytes_read,
        bytes_written: 0,
        retry_count: 0,
        queue_pressure_peak: 0,
        duration: started.elapsed(),
        stage_durations: StageDurations::default(),
        skipped_due_to_failure: 0,
        tile_evidence: Some(evidence),
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
/// the caller's configured format is legitimately `None` for a sink that does
/// not commit to a format, and there is then nothing to compare against
/// rather than something being withheld.
///
/// # The source size and the overlap, which nothing else can see (issue #1130)
///
/// The level range and the grid both round: a pyramid's levels are fixed by
/// its longest side rounded up to a power of two, and each level's grid is its
/// size divided by the tile size and rounded up. So a 4000-pixel source and a
/// 4096-pixel one plan thirteen identical levels with identical grids and the
/// same 349 coordinates, and every check in this function passed on an archive
/// generated from a different picture.
///
/// The overlap is worse, because it does not reach the grid at all.
/// `tile_grid` divides by the tile size and nothing else, so planning at
/// overlap 1 and at overlap 0 produces byte-identical `levels` vectors while
/// `tile_rect` moves every tile's rectangle. Two pyramids that disagree about
/// it share every number a sweep or a count can compare and have no pixel in
/// common.
///
/// Both are refused for `None` the way the tile size is, and for the same
/// reason: a pyramid that will not say cannot be checked, and "cannot be
/// checked" is a refusal rather than a pass.
///
/// # What this check is worth per backend
///
/// Everything here compares the reader's `describe()` against the plan, so it
/// is only as independent as the two sources are.
/// [`DirectoryPyramidReader::describe`](crate::pyramid_reader::DirectoryPyramidReader)
/// fills all five fields out of the plan it was opened with, so for a
/// loose-file tree this compares a plan with itself and cannot fail. That is
/// not a defect to fix here, it is what a directory of tiles can say about
/// itself: a tree carries no metadata, and the reader is handed the plan
/// precisely because nothing in the tree records one. The check bites on an
/// archive, which records its own generation settings independently of
/// whatever plan is checking it.
///
/// The arms are ordered cheapest-claim-first rather than by history, so an
/// archive carrying no `vnd.libviprs` namespace at all now gets refused for
/// its source size where it used to be refused for its tile size. Both are
/// true of it and the operator-facing string changed, which is worth knowing
/// if anything matches on that text.
fn describe_matches_the_plan(
    reader: &dyn PyramidReader,
    plan: &PyramidPlan,
    configured_format: Option<TileFormat>,
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

    match (described.source_width, described.source_height) {
        (Some(width), Some(height)) if (width, height) == (plan.image_width, plan.image_height) => {
        }
        (Some(width), Some(height)) => {
            return Err(refused(format!(
                "Verify: the pyramid was generated from a {width}x{height} source \
                 and the plan is for {}x{}",
                plan.image_width, plan.image_height
            )));
        }
        _ => {
            return Err(refused(
                "Verify: the pyramid does not record the source size it was \
                 generated from, so there is nothing to check this plan against"
                    .to_string(),
            ));
        }
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

    match described.overlap {
        Some(overlap) if overlap == plan.overlap => {}
        Some(overlap) => {
            return Err(refused(format!(
                "Verify: the pyramid was generated with an overlap of {overlap} \
                 pixels and the plan asks for {}",
                plan.overlap
            )));
        }
        None => {
            return Err(refused(
                "Verify: the pyramid does not record the overlap it was \
                 generated with, so there is nothing to check this plan against"
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

    match (described.format, configured_format) {
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::observe::NoopObserver;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::pyramid_reader::{PyramidDescription, StructuralSummary};
    use crate::sink::{Tile, TileSink};

    /// A pyramid that answers whatever the test wants it to, and remembers
    /// which of the two probes it was asked for.
    ///
    /// Three of the branches in [`pyramid_verify`] cannot be reached with an
    /// archive this crate is able to write: our own PMTiles writer refuses a
    /// zero-length payload by name, always records `vnd.libviprs`, and always
    /// knows how many tiles it addressed. A reader that does none of those is
    /// not hypothetical (a foreign archive is exactly that, and our validator
    /// does not flag a zero-length entry), and a double is the honest way to
    /// reach the branches without hand-assembling a hostile file.
    ///
    /// Since #1130 it also stands in for the two shapes a backend can take:
    /// one that answers the whole structural walk at once, and one that only
    /// knows how to count. `tile_len` here is deliberately **not** the trait's
    /// default body, because the default reaches `tile` and would count both
    /// probes as one.
    struct FakePyramid {
        description: PyramidDescription,
        /// What `tile` answers for every coordinate.
        stored: Option<Vec<u8>>,
        /// What `addressed_tiles` answers, or `None` for a reader that cannot
        /// count.
        addressed: Option<u64>,
        /// What `structural_summary` answers, or `None` to sit on the trait's
        /// default, which is the shape of every backend written before #1130.
        summary: Option<StructuralSummary>,
        payload_reads: AtomicUsize,
        length_reads: AtomicUsize,
        counts: AtomicUsize,
    }

    impl FakePyramid {
        fn new(
            description: PyramidDescription,
            stored: Option<Vec<u8>>,
            addressed: Option<u64>,
        ) -> Self {
            Self {
                description,
                stored,
                addressed,
                summary: None,
                payload_reads: AtomicUsize::new(0),
                length_reads: AtomicUsize::new(0),
                counts: AtomicUsize::new(0),
            }
        }

        fn with_summary(mut self, summary: StructuralSummary) -> Self {
            self.summary = Some(summary);
            self
        }

        fn payload_reads(&self) -> usize {
            self.payload_reads.load(Ordering::SeqCst)
        }

        fn length_reads(&self) -> usize {
            self.length_reads.load(Ordering::SeqCst)
        }

        fn counts(&self) -> usize {
            self.counts.load(Ordering::SeqCst)
        }
    }

    impl PyramidReader for FakePyramid {
        fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
            Ok(self.description.clone())
        }

        fn tile(&self, _coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError> {
            self.payload_reads.fetch_add(1, Ordering::SeqCst);
            Ok(self.stored.clone())
        }

        fn tile_len(&self, _coord: TileCoord) -> Result<Option<u64>, PyramidReadError> {
            self.length_reads.fetch_add(1, Ordering::SeqCst);
            Ok(self.stored.as_ref().map(|bytes| bytes.len() as u64))
        }

        fn structural_summary(&self) -> Result<StructuralSummary, PyramidReadError> {
            match &self.summary {
                Some(summary) => Ok(summary.clone()),
                None => Ok(StructuralSummary::new()),
            }
        }

        fn addressed_tiles(&self) -> Result<u64, PyramidReadError> {
            self.counts.fetch_add(1, Ordering::SeqCst);
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
            source_width: Some(plan.image_width),
            source_height: Some(plan.image_height),
            overlap: Some(plan.overlap),
        }
    }

    fn verify(reader: &FakePyramid, plan: &PyramidPlan) -> Result<EngineResult, EngineError> {
        pyramid_verify(reader, plan, PngSink.content_format(), &NoopObserver)
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

        let sound = FakePyramid::new(
            description(&plan),
            Some(vec![0x89, b'P', b'N', b'G']),
            Some(planned),
        );
        verify(&sound, &plan).expect("the control verifies");

        let hollow = FakePyramid::new(description(&plan), Some(Vec::new()), Some(planned));
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
        let uncountable = FakePyramid::new(description(&plan), Some(vec![1, 2, 3]), None);
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
    ///
    /// Three fields, one per thing a pyramid can decline to say about itself,
    /// because each is checked in its own arm and a cell carrying only the
    /// first proves nothing about the other two. The source size and the
    /// overlap arrived with #1130, and they are the two that nothing else in
    /// this function can see: the level range and the grid both round, so a
    /// 4000-pixel source and a 4096-pixel one are indistinguishable, and
    /// overlap does not reach the grid at all.
    #[test]
    fn a_pyramid_that_cannot_say_how_it_was_generated_is_refused() {
        let plan = plan();
        let planned = plan.tile_coords().count() as u64;

        for (withhold, expected) in [
            (
                Box::new(|d: &mut PyramidDescription| d.tile_size = None)
                    as Box<dyn Fn(&mut PyramidDescription)>,
                "tile size",
            ),
            (
                Box::new(|d: &mut PyramidDescription| d.source_width = None),
                "source size",
            ),
            (
                Box::new(|d: &mut PyramidDescription| d.overlap = None),
                "overlap",
            ),
        ] {
            let mut description = description(&plan);
            withhold(&mut description);
            let anonymous = FakePyramid::new(description, Some(vec![1, 2, 3]), Some(planned));
            let message = verify(&anonymous, &plan)
                .expect_err("a pyramid with no generation record cannot be checked")
                .to_string();
            assert!(
                message.contains(expected),
                "the refusal must name what the pyramid could not say \
                 ({expected}), got: {message}"
            );
        }
    }

    /// The count check is an equality, and the direction nothing else catches
    /// is the one where the pyramid is bigger than the plan.
    #[test]
    fn a_pyramid_addressing_more_than_the_plan_is_refused() {
        let plan = plan();
        let planned = plan.tile_coords().count() as u64;
        let wide = FakePyramid::new(description(&plan), Some(vec![1, 2, 3]), Some(planned + 1));
        let message = verify(&wide, &plan)
            .expect_err("a pyramid wider than the plan cannot verify")
            .to_string();
        assert!(
            message.contains(&(planned + 1).to_string()) && message.contains(&planned.to_string()),
            "the refusal must name both counts, got: {message}"
        );
    }

    /// A walk that bounded every offset it read makes the sweep take lengths,
    /// and the run says which guarantee that leaves it with.
    ///
    /// The three assertions after the first are the ones that matter. No
    /// payload was fetched, every coordinate was still probed, and the count
    /// came off the summary rather than out of a second walk.
    #[test]
    fn a_bounded_walk_makes_the_sweep_read_lengths_rather_than_payloads() {
        let plan = plan();
        let planned = plan.tile_coords().count();
        let bounded = FakePyramid::new(description(&plan), Some(vec![1, 2, 3]), None).with_summary(
            StructuralSummary::new()
                .with_addressed_tiles(planned as u64)
                .with_offsets_bounded(true),
        );

        let result = verify(&bounded, &plan).expect("a sound pyramid verifies");

        assert_eq!(
            bounded.payload_reads(),
            0,
            "the sweep read {} payloads to learn {planned} lengths the index \
             already carries",
            bounded.payload_reads()
        );
        assert_eq!(
            bounded.length_reads(),
            planned,
            "the positive control: a sweep that probed nothing would also have \
             read no payloads"
        );
        assert_eq!(
            bounded.counts(),
            0,
            "the summary carried the count, so asking for it again is the \
             second walk this change removed"
        );
        assert_eq!(
            result.tile_evidence,
            Some(TileEvidence::LengthsFromTheIndex)
        );
        assert_eq!(
            result.bytes_read, 0,
            "a run that read no payload must not report bytes it did not read"
        );
    }

    /// A walk that bounded nothing keeps reading payloads.
    ///
    /// This one is green before #1130 as well as after, and that is what it is
    /// for: it is the arm that must **not** move. A length says a tile is
    /// there and how large it is; only a payload read says the bytes come
    /// back, and nothing has established that for a backend whose walk had no
    /// size to check against. The cells that were red are in
    /// `tests/pmtiles_verify_reads_the_index.rs`.
    ///
    /// It doubles as the check that a backend which only knows how to count
    /// still gets counted: this double sits on the default summary, so the
    /// count has to be asked for directly, exactly once.
    #[test]
    fn an_unbounded_walk_keeps_reading_payloads() {
        let plan = plan();
        let planned = plan.tile_coords().count();
        let unbounded = FakePyramid::new(
            description(&plan),
            Some(vec![1, 2, 3]),
            Some(planned as u64),
        );

        let result = verify(&unbounded, &plan).expect("a sound pyramid verifies");

        assert_eq!(
            unbounded.payload_reads(),
            planned,
            "every coordinate's bytes have to be read when nothing else has \
             proved they are reachable"
        );
        assert_eq!(
            unbounded.length_reads(),
            0,
            "and the cheap probe must not be used behind a walk that earned \
             nothing"
        );
        assert_eq!(
            unbounded.counts(),
            1,
            "the count was asked for directly, once"
        );
        assert_eq!(result.tile_evidence, Some(TileEvidence::PayloadsRead));
        assert_eq!(
            result.bytes_read,
            planned as u64 * 3,
            "the payload path reports what it read"
        );
    }
}
