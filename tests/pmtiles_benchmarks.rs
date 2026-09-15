//! PMTiles against the directory backend, measured (issue #993).
//!
//! Generation and reads, both backends, one clock and one memory basis, with
//! the result exported as the JSON the libviprs.org benchmark page reads.
//!
//! # Why this is `#[ignore]`d tests and not `criterion`
//!
//! This crate has no `benches/` and no `criterion`, and issue #993 asks for a
//! documented reason before it gains one. The reason not to is that criterion
//! measures the wrong thing here. Its model is "call this function many times,
//! report the distribution", which is right for a pure function and wrong for
//! a pyramid run: the interesting numbers are peak RSS, output size and
//! filesystem-entry count, all of which are properties of one run rather than
//! of a sample of many, and the cold read numbers want a cache that repeated
//! iterations destroy by construction. So this reuses the `#[ignore]`
//! wall-clock convention `src/colour.rs` already uses and the
//! [`MemoryTracker`](libviprs::MemoryTracker) the engine already reports
//! through `EngineResult::peak_memory_bytes`.
//!
//! # Every measurement is a fresh process
//!
//! Peak RSS is a high-water mark the kernel never lowers on its own, so two
//! backends measured in one process hand the second one the first one's peak.
//! libviprs-bench learned this the expensive way (its issue #153 and its
//! `tests/rss_isolation.rs`), and the fix there is the fix here: the parent
//! test re-executes **this test binary** once per cell, and a cell reports its
//! own process's numbers. Inside a cell, the read phase additionally resets the
//! high-water mark through `/proc/self/clear_refs`, so generation's peak does
//! not become the read scenario's floor.
//!
//! On a platform with no `/proc`, both mechanisms are unavailable and every
//! `peak_rss_mb` is `null`, along with the two ratios computed from it. That is
//! a column the platform cannot fill, not a number invented to fill it, and it
//! is why the published figures come from the Linux container. It used to be
//! `0.0`, which was worse than useless on `resource_cost`, where zero is the
//! best possible score and a failed measurement therefore published a record.
//!
//! # Running it
//!
//! ```text
//! cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
//! LIBVIPRS_BENCH_PROFILE=large LIBVIPRS_BENCH_JSON=/tmp/pmtiles.json \
//!   cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
//! ```
//!
//! `docs/pmtiles-benchmarks.md` has the full procedure and what each column
//! means.
//!
//! # What issue #1021 changed here
//!
//! #1021 is a measurement issue and says so, so nothing on the read path moved
//! and nothing here is an optimisation. Four things changed about what gets
//! measured:
//!
//! * A **brink cell**, whose root stops fourteen entries under the largest
//!   flat root the writer emits. The other cells sit at 93, 1373, 5469 and 6
//!   root entries, so the sweep bracketed the worst case without touching it
//!   and the published peak was a line fitted through three points.
//! * The cold row is **split** into the six phases a cold open is made of, and
//!   published next to the combined row rather than instead of it. The phases
//!   have to add up, which is what stops the split being six numbers about
//!   some other piece of work.
//! * `read_concurrent` became a **curve**, one row at each of 1, 2, 4 and 8
//!   threads, with T=1 as the control. One thread count cannot say whether a
//!   slow row is contention or per-lookup cost.
//! * The envelope carries **provenance**, because the numbers this issue is
//!   correcting were published with no host attached at all.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pmtiles::{Entry, FileRangeReader, Header, RangeReader};
use libviprs::pyramid_reader::{DirectoryPyramidReader, PmTilesPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::{EngineBuilder, FsSink};

#[path = "common/pmtiles_bench.rs"]
mod bench;

use bench::{Measurement, Profile, Splitmix};

/// Environment variable a child cell reads to learn what to measure.
const CELL_VAR: &str = "LIBVIPRS_BENCH_CELL";

/// Prefix a child puts on the lines it wants the parent to reprint.
///
/// A child's stdout goes nowhere unless it fails, so anything it says about
/// the pyramid it just built is lost. This is the one line per cell worth
/// keeping: what shape the archive's directory came out.
const SAY_PREFIX: &str = "CELL ";

/// Environment variable naming the file a child writes its rows into.
///
/// A file rather than a line on stdout, because a row is pretty-printed JSON
/// and so is several lines. The first version of this printed each row behind
/// a prefix and the parent picked the prefixed lines back out, which captured
/// the opening `[` of every row and nothing else, and the failure surfaced as
/// `expected value, line 2 column 1` from the parent's own parse rather than
/// as anything about the protocol. `a_child_row_file_splices_into_the_document`
/// is the control that now covers the hand-off without running a benchmark.
const CELL_OUT_VAR: &str = "LIBVIPRS_BENCH_CELL_OUT";

/// The two backends, under the names the exported rows carry.
const DIRECTORY: &str = "directory";
const PMTILES: &str = "pmtiles";

/// The writer's cutoff, from `ROOT_ONLY_MAX_ENTRIES` in
/// `src/pmtiles/writer.rs`, which is also go-pmtiles'. At or above it the
/// entries cannot all live in the root and the archive grows leaf
/// directories.
const ROOT_ONLY_MAX_ENTRIES: usize = 16_384;

/// The largest root this writer will actually emit, which is one less.
///
/// `writer.rs:1156` is `if plan.entry_count < ROOT_ONLY_MAX_ENTRIES`, a strict
/// comparison, so 16384 entries already spill into leaves and 16383 is the
/// biggest flat root there is. Issue #1021 says 16384 and puts its
/// extrapolated peak there; it is one entry out, which changes nothing about
/// the argument and everything about what a cell that claims to sit on the
/// cutoff has to plan.
///
/// `entry_count` counts **run-length-encoded entries and not tiles**. The
/// writer's `plan_entries` opens a run when a tile's id follows the last one
/// and its payload is byte-identical, so a pyramid of repeated tiles has far
/// fewer entries than tiles. The benchmark's source is a gradient, whose tiles
/// are all distinct, so for these cells the two numbers are equal, and
/// `the_brink_cells_root_stops_just_under_the_writers_cutoff` checks that
/// rather than assuming it.
const LARGEST_FLAT_ROOT: usize = ROOT_ONLY_MAX_ENTRIES - 1;

/// How many cold samples a read scenario takes.
///
/// Small, because each one opens a reader and throws it away, and the point of
/// the row is the shape of a first lookup rather than a tight confidence
/// interval on it.
const COLD_SAMPLES: usize = 64;

/// The thread counts the concurrent scenario walks.
///
/// **1 is the control and it is the reason this is a ladder rather than a
/// number.** The old row measured one thread count, picked from
/// `available_parallelism`, so a slow concurrent row could have been
/// contention or could have been per-lookup cost that was always there, and
/// nothing in the export could tell the two apart. With T=1 measured through
/// the same code path, the shape of the curve against it is the answer.
///
/// 8 runs even on a box with fewer cores. That is oversubscription rather than
/// parallelism, and it is left in deliberately: the provenance records `ncpu`,
/// so a reader can see which points on the curve had a core to themselves.
const THREAD_LADDER: [usize; 4] = [1, 2, 4, 8];

// ---------------------------------------------------------------------------
// Peak RSS, and resetting it
// ---------------------------------------------------------------------------

/// Ask the kernel to drop this process's peak-RSS high-water mark back to its
/// current RSS.
///
/// `5` is `CLEAR_REFS_MM_HIWATER_RSS`. It exists so a long-lived process can
/// measure a phase rather than its whole history, which is exactly the problem
/// here. Answers whether it worked; everywhere without `/proc` it does not,
/// and the caller reports `0.0` rather than a stale peak.
fn reset_peak_rss() -> bool {
    std::fs::write("/proc/self/clear_refs", "5\n").is_ok()
}

/// Peak RSS in bytes for a phase that began with a successful
/// [`reset_peak_rss`], or `None` where the platform has no answer.
///
/// `None` rather than `0`, all the way out to a `null` in the exported row.
/// A zero here divided into `resource_cost`, where lower is better, so every
/// platform without `/proc` published the best possible score on that column
/// as though it had measured it.
fn phase_peak_rss(reset_worked: bool) -> Option<u64> {
    if reset_worked {
        bench::peak_rss_bytes()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Generating
// ---------------------------------------------------------------------------

fn plan_for(width: u32, height: u32, tile_size: u32) -> PyramidPlan {
    PyramidPlanner::new(width, height, tile_size, 0, Layout::Xyz)
        .expect("a plan is valid")
        .plan()
}

/// Every coordinate the plan covers, in level then row then column order.
fn coordinates(plan: &PyramidPlan) -> Vec<TileCoord> {
    let mut out = Vec::new();
    for level in &plan.levels {
        for row in 0..level.rows {
            for col in 0..level.cols {
                out.push(TileCoord {
                    level: level.level,
                    col,
                    row,
                });
            }
        }
    }
    out
}

/// One cell of the sweep: a canvas at a tile size.
#[derive(Debug, Clone, Copy)]
struct Cell {
    width: u32,
    height: u32,
    tile_size: u32,
}

impl Cell {
    fn spec(&self) -> String {
        format!("{}x{}@{}", self.width, self.height, self.tile_size)
    }
}

/// What one generation run produced.
struct Generated {
    row: Measurement,
    /// The archive, or the root of the tree.
    output: PathBuf,
    plan: PyramidPlan,
}

/// Run one canvas into one backend and measure it.
fn generate(storage: &str, profile: Profile, dir: &Path, cell: Cell) -> Generated {
    let plan = plan_for(cell.width, cell.height, cell.tile_size);
    let source = bench::gradient(cell.width, cell.height);

    let reset = reset_peak_rss();
    let (result, output) = match storage {
        PMTILES => {
            let archive = dir.join("pyramid.pmtiles");
            let sink = PmTilesSink::builder(&archive)
                .plan(plan.clone())
                .tile_format(TileFormat::Png)
                .build()
                .expect("the archive sink builds");
            let result = EngineBuilder::new(&source, plan.clone(), sink)
                .run()
                .expect("the archive run succeeds");
            (result, archive)
        }
        DIRECTORY => {
            let root = dir.join("tree");
            let sink = FsSink::new(&root, plan.clone()).with_format(TileFormat::Png);
            let result = EngineBuilder::new(&source, plan.clone(), sink)
                .run()
                .expect("the directory run succeeds");
            (result, root)
        }
        other => panic!("unknown storage backend {other:?}"),
    };
    let rss = phase_peak_rss(reset);

    let row = Measurement::new(
        "generate",
        storage,
        profile,
        cell.width,
        cell.height,
        cell.tile_size,
        1,
        result.duration,
        Some(result.peak_memory_bytes),
        rss,
        result.tiles_produced,
    )
    .with_output(bench::occupancy(&output));

    Generated { row, output, plan }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

fn open_reader(storage: &str, output: &Path, plan: &PyramidPlan) -> Box<dyn PyramidReader> {
    match storage {
        PMTILES => {
            Box::new(PmTilesPyramidReader::try_open(output).expect("the archive opens for reading"))
        }
        DIRECTORY => Box::new(
            DirectoryPyramidReader::try_open(output, plan.clone(), TileFormat::Png)
                .expect("the tree opens for reading"),
        ),
        other => panic!("unknown storage backend {other:?}"),
    }
}

/// One read pass: per-lookup latencies and the tile bytes returned.
struct Pass {
    elapsed: Duration,
    latencies: Vec<Duration>,
    bytes: u64,
    hits: u64,
}

fn read_pass(reader: &dyn PyramidReader, coords: &[TileCoord]) -> Pass {
    let mut latencies = Vec::with_capacity(coords.len());
    let mut bytes = 0;
    let mut hits = 0;
    let started = Instant::now();
    for coord in coords {
        let at = Instant::now();
        let tile = reader.tile(*coord).expect("a lookup succeeds");
        latencies.push(at.elapsed());
        if let Some(tile) = tile {
            bytes += tile.len() as u64;
            hits += 1;
        }
    }
    Pass {
        elapsed: started.elapsed(),
        latencies,
        bytes,
        hits,
    }
}

/// A pass where every lookup gets a reader that has never been used.
///
/// The open is inside the timed section on purpose. For PMTiles it is a header
/// fetch and a root-directory fetch, which is what a client really pays before
/// its first tile; for the directory backend it reads nothing at all, because
/// a tree has nothing to read up front. Leaving it out would price one backend
/// for work the other does not do.
fn cold_pass(storage: &str, output: &Path, plan: &PyramidPlan, coords: &[TileCoord]) -> Pass {
    let mut latencies = Vec::with_capacity(coords.len());
    let mut bytes = 0;
    let mut hits = 0;
    let started = Instant::now();
    for coord in coords {
        let at = Instant::now();
        let reader = open_reader(storage, output, plan);
        let tile = reader.tile(*coord).expect("a lookup succeeds");
        latencies.push(at.elapsed());
        if let Some(tile) = tile {
            bytes += tile.len() as u64;
            hits += 1;
        }
    }
    Pass {
        elapsed: started.elapsed(),
        latencies,
        bytes,
        hits,
    }
}

/// One cold pass with the open taken apart into the phases it is made of.
///
/// Each iteration walks `Reader::try_new`'s own steps by hand, in its order,
/// timing each one, and then times a lookup through a real reader. Doing it by
/// hand rather than instrumenting `src/pmtiles/reader.rs` keeps this a
/// measurement issue: nothing on the product's hot path changes to be measured
/// here, and the check that the hand-rolled version really is the same work is
/// that its phases add up to what the combined row measures.
///
/// The second open, the one the lookup phase runs against, is not timed. It is
/// real work and it makes this pass slower than the combined one in wall time,
/// which is why `wall_time_ms` on a phase row is the phase and not the pass.
struct SplitPass {
    /// One sample vector per entry in [`bench::COLD_PHASES`], in that order.
    phases: Vec<Vec<Duration>>,
    /// Per iteration, the sum of that iteration's six phases.
    totals: Vec<Duration>,
    /// Tile bytes the lookup phase returned, which is the only phase that
    /// returns any.
    bytes: u64,
    hits: u64,
}

impl SplitPass {
    /// The median of the per-iteration sums, in microseconds.
    fn total_p50_us(&mut self) -> Option<f64> {
        bench::percentile_micros(&mut self.totals)
    }
}

/// A [`RangeReader`] that records every `(offset, len)` it is asked for.
///
/// The reconciliation guard needs to know which byte ranges a cold open
/// actually touches, and asking the source is the only way to know it that
/// does not involve a clock. `tests/pmtiles_reader.rs` has its own copy for
/// its own questions; they are separate test binaries and consolidating them
/// is a change for its own PR rather than a rider on this one.
///
/// `size` is deliberately not logged. `Reader::try_new` calls it to
/// bounds-check the header and it moves no bytes, so counting it would put an
/// entry in the log that no phase of the split corresponds to.
struct ReadLog<R: RangeReader> {
    inner: R,
    log: ReadTally,
}

/// A read log that several sources write into and a caller can read back after
/// every one of them has been dropped.
///
/// Shared rather than owned because the split builds its source inside the
/// phase that times the build, so the guard never holds that source and cannot
/// ask it anything afterwards. The log outlives it.
type ReadTally = std::sync::Arc<std::sync::Mutex<Vec<(u64, usize)>>>;

fn read_tally() -> ReadTally {
    std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// Every range served into `log` so far, in order.
fn tallied(log: &ReadTally) -> Vec<(u64, usize)> {
    log.lock().expect("the read log is not poisoned").clone()
}

impl<R: RangeReader> ReadLog<R> {
    fn sharing(inner: R, log: ReadTally) -> Self {
        Self { inner, log }
    }
}

impl<R: RangeReader> RangeReader for ReadLog<R> {
    fn read_range(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.log
            .lock()
            .expect("the read log is not poisoned")
            .push((offset, len));
        self.inner.read_range(offset, len)
    }

    fn size(&self) -> std::io::Result<Option<u64>> {
        self.inner.size()
    }
}

/// What one split iteration produced, beside its six durations.
///
/// The products are here because the reconciliation guard needs them. Three of
/// the six phases do no I/O at all, so nothing a read log can see tells an
/// inflate that ran from one that was skipped; what tells them apart is the
/// bytes that came out. Carrying them costs the measurement nothing, because
/// every timed region is closed before this is built.
struct SplitIteration {
    /// In [`bench::COLD_PHASES`] order.
    phases: [Duration; 6],
    /// What the header phase decoded.
    header: Header,
    /// What the inflate and decode phases turned the root into.
    entries: Vec<Entry>,
    /// What the lookup phase returned.
    tile: Option<Vec<u8>>,
}

/// One iteration of the split: `Reader::try_new`'s own steps, by hand, in its
/// order, each one timed.
///
/// Split out of [`cold_split_pass`] so that
/// `the_cold_split_accounts_for_the_whole_combined_row` can run these steps
/// rather than a second copy of them. That is the whole reason the function
/// exists: a guard that re-implemented this sequence would be asserting that
/// its own copy matches the reader, which is the one thing never in doubt,
/// while the copy that actually drifts went unchecked.
fn split_iteration(output: &Path, coord: TileCoord) -> SplitIteration {
    split_iteration_over(output, coord, |path| {
        FileRangeReader::try_open(path).expect("the archive opens")
    })
}

/// [`split_iteration`], with the byte source its hand-rolled phases read
/// through supplied by the caller.
///
/// The reconciliation guard passes a source that writes down what it was asked
/// for, which is how the split's own reads become comparable with the combined
/// row's. Products alone would leave a split that read the right bytes twice
/// looking exactly like one that read them once, and a phase doing more work
/// than the row does is the same defect as a phase doing less.
fn split_iteration_over<R: RangeReader>(
    output: &Path,
    coord: TileCoord,
    open_source: impl Fn(&Path) -> R,
) -> SplitIteration {
    use libviprs::pmtiles::directory::deserialize_entries;
    use libviprs::pmtiles::header::HEADER_BYTES;
    use libviprs::pmtiles::reader::MAX_DIRECTORY_BYTES;

    let at = Instant::now();
    let source = open_source(output);
    std::hint::black_box(source.size().expect("the archive has a size"));
    let open = at.elapsed();

    let at = Instant::now();
    let header = Header::try_decode(
        &source
            .read_range(0, HEADER_BYTES)
            .expect("the header can be read"),
    )
    .expect("the header decodes");
    let header_time = at.elapsed();

    let at = Instant::now();
    let raw = source
        .read_range(
            header.root_offset,
            usize::try_from(header.root_length).expect("a root length fits a usize"),
        )
        .expect("the root can be read");
    let fetch = at.elapsed();

    let at = Instant::now();
    let plain = header
        .internal_compression
        .decompress(&raw, MAX_DIRECTORY_BYTES)
        .expect("the root inflates");
    let inflate = at.elapsed();

    let at = Instant::now();
    let entries = std::hint::black_box(deserialize_entries(&plain).expect("the root decodes"));
    let decode = at.elapsed();
    assert!(!entries.is_empty(), "a root of no entries is not a root");

    // Untimed: the lookup phase has to run against a reader the crate built,
    // because that is the code path a caller takes, and rebuilding `locate`
    // here would be measuring this file instead of the crate.
    let reader = PmTilesPyramidReader::try_open(output).expect("the archive opens for reading");
    let at = Instant::now();
    let tile = reader.tile(coord).expect("a lookup succeeds");
    let lookup = at.elapsed();

    SplitIteration {
        phases: [open, header_time, fetch, inflate, decode, lookup],
        header,
        entries,
        tile,
    }
}

fn cold_split_pass(output: &Path, coords: &[TileCoord]) -> SplitPass {
    let mut phases: Vec<Vec<Duration>> = bench::COLD_PHASES
        .iter()
        .map(|_| Vec::with_capacity(coords.len()))
        .collect();
    let mut totals = Vec::with_capacity(coords.len());
    let mut bytes = 0;
    let mut hits = 0;

    for coord in coords {
        let iteration = split_iteration(output, *coord);
        if let Some(tile) = &iteration.tile {
            bytes += tile.len() as u64;
            hits += 1;
        }
        totals.push(iteration.phases.iter().copied().sum());
        for (slot, value) in phases.iter_mut().zip(iteration.phases) {
            slot.push(value);
        }
    }

    SplitPass {
        phases,
        totals,
        bytes,
        hits,
    }
}

/// The same as [`read_pass`], spread over `threads` threads, each taking a
/// contiguous slice.
fn concurrent_pass(reader: &dyn PyramidReader, coords: &[TileCoord], threads: usize) -> Pass {
    let chunk = coords.len().div_ceil(threads.max(1));
    let started = Instant::now();
    let parts: Vec<Pass> = std::thread::scope(|scope| {
        let handles: Vec<_> = coords
            .chunks(chunk.max(1))
            .map(|slice| scope.spawn(move || read_pass(reader, slice)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a reader thread does not panic"))
            .collect()
    });
    let mut latencies = Vec::new();
    let mut bytes = 0;
    let mut hits = 0;
    for part in parts {
        latencies.extend(part.latencies);
        bytes += part.bytes;
        hits += part.hits;
    }
    Pass {
        elapsed: started.elapsed(),
        latencies,
        bytes,
        hits,
    }
}

/// Build a row out of a finished pass.
#[allow(clippy::too_many_arguments)]
fn read_row(
    scenario: &str,
    storage: &str,
    profile: Profile,
    cell: Cell,
    concurrency: usize,
    rss: Option<u64>,
    root_entries: Option<u64>,
    mut pass: Pass,
) -> Measurement {
    let bytes = pass.bytes;
    Measurement::new(
        scenario,
        storage,
        profile,
        cell.width,
        cell.height,
        cell.tile_size,
        concurrency,
        pass.elapsed,
        // The engine's tracker charges raster buffers, and a read allocates
        // none, so this column has no answer on a read row rather than a zero.
        None,
        rss,
        pass.hits,
    )
    .with_root_entries(root_entries)
    .with_latencies(&mut pass.latencies, bytes)
}

/// One row per phase of the split cold open.
///
/// `tiles_produced` on a phase row is how many cold opens that phase served,
/// which is the same number the combined row reports, because each iteration
/// of the split does every phase exactly once for one tile. `wall_time_ms` is
/// the phase's own time summed over those iterations and not the pass's, so
/// six phase rows' wall times add up to what the split cost rather than to six
/// times it.
///
/// Only the lookup phase returns tile bytes. Every other phase publishes
/// `tile_bytes_returned: null` rather than `0`, because a zero there reads as
/// a phase that fetched a tile for nothing.
fn cold_split_rows(
    storage: &str,
    profile: Profile,
    cell: Cell,
    rss: Option<u64>,
    root_entries: Option<u64>,
    pass: &mut SplitPass,
) -> Vec<Measurement> {
    let mut rows = Vec::with_capacity(bench::COLD_PHASES.len());
    for (index, scenario) in bench::COLD_PHASES.iter().enumerate() {
        let samples = &mut pass.phases[index];
        let elapsed: Duration = samples.iter().copied().sum();
        let is_lookup = *scenario == "read_cold_lookup";
        let row = Measurement::new(
            scenario,
            storage,
            profile,
            cell.width,
            cell.height,
            cell.tile_size,
            1,
            elapsed,
            None,
            rss,
            samples.len() as u64,
        )
        .with_root_entries(root_entries);
        rows.push(if is_lookup {
            row.with_latencies(samples, pass.bytes)
        } else {
            row.with_latency_samples(samples)
        });
    }
    rows
}

/// Every read scenario, against one already-generated pyramid.
///
/// "Cold" is a reader opened a moment ago whose in-process caches hold
/// nothing. It is **not** a cold OS page cache. Dropping that needs root on
/// Linux and has no portable equivalent, and a benchmark that claimed a cold
/// page cache without dropping one would be reporting a warm number under a
/// cold label. The gap between the cold and warm rows is therefore the
/// in-process caches plus the archive's own open cost, which is the part
/// libviprs controls.
fn read_scenarios(
    storage: &str,
    profile: Profile,
    cell: Cell,
    output: &Path,
    plan: &PyramidPlan,
    root_entries: Option<u64>,
) -> Vec<Measurement> {
    let all = coordinates(plan);
    assert!(!all.is_empty(), "a plan with no tiles is not a pyramid");
    let samples = profile.read_samples().min(all.len());
    let sequential: Vec<TileCoord> = all.iter().copied().take(samples).collect();

    let mut rng = Splitmix::new(0x5EED_1234_ABCD_0001);
    let random: Vec<TileCoord> = (0..samples).map(|_| all[rng.below(all.len())]).collect();

    let cold_coords: Vec<TileCoord> = sequential
        .iter()
        .copied()
        .take(COLD_SAMPLES.min(samples))
        .collect();

    let mut rows = Vec::new();
    let reset = reset_peak_rss();

    let cold = cold_pass(storage, output, plan, &cold_coords);
    assert_eq!(
        cold.hits,
        cold_coords.len() as u64,
        "every planned coordinate should be present in the pyramid"
    );
    let combined_p50 = {
        let mut latencies = cold.latencies.clone();
        bench::percentile_micros(&mut latencies)
    };
    rows.push(read_row(
        "read_cold",
        storage,
        profile,
        cell,
        1,
        phase_peak_rss(reset),
        root_entries,
        cold,
    ));

    // The same cold open again, taken apart. PMTiles only: the directory
    // backend's open is one `is_dir()` stat with no header, no ranged read and
    // no index to decode, so splitting it would produce four rows of nothing
    // and one that is the whole cost. That asymmetry is the finding, and the
    // combined row already carries it.
    if storage == PMTILES {
        let mut split = cold_split_pass(output, &cold_coords);
        assert_eq!(
            split.hits,
            cold_coords.len() as u64,
            "the split pass should find every coordinate the combined pass found"
        );
        if let (Some(split_p50), Some(combined_p50)) = (split.total_p50_us(), combined_p50) {
            println!(
                "{SAY_PREFIX}{storage} {}: cold split p50 {split_p50:.2} us against combined \
                 {combined_p50:.2} us, {:+.1}%",
                cell.spec(),
                100.0 * (split_p50 - combined_p50) / combined_p50,
            );
        }
        rows.extend(cold_split_rows(
            storage,
            profile,
            cell,
            phase_peak_rss(reset),
            root_entries,
            &mut split,
        ));
    }

    // Warm: one reader, the same coordinates the cold pass just walked.
    let reader = open_reader(storage, output, plan);
    let _ = read_pass(reader.as_ref(), &cold_coords);
    let warm = read_pass(reader.as_ref(), &cold_coords);
    rows.push(read_row(
        "read_warm",
        storage,
        profile,
        cell,
        1,
        phase_peak_rss(reset),
        root_entries,
        warm,
    ));

    let sequential_pass = read_pass(reader.as_ref(), &sequential);
    rows.push(read_row(
        "read_sequential",
        storage,
        profile,
        cell,
        1,
        phase_peak_rss(reset),
        root_entries,
        sequential_pass,
    ));

    let random_pass = read_pass(reader.as_ref(), &random);
    rows.push(read_row(
        "read_random",
        storage,
        profile,
        cell,
        1,
        phase_peak_rss(reset),
        root_entries,
        random_pass,
    ));

    // A curve rather than a point. One thread is the control: without it a
    // slow eight-thread row could be contention or could be per-lookup cost
    // that was there all along, and the export could not say which.
    for threads in THREAD_LADDER {
        let concurrent = concurrent_pass(reader.as_ref(), &random, threads);
        rows.push(read_row(
            "read_concurrent",
            storage,
            profile,
            cell,
            threads,
            phase_peak_rss(reset),
            root_entries,
            concurrent,
        ));
    }

    rows
}

/// Print what shape the archive's directory came out, for a PMTiles cell.
///
/// Nothing in a result row carries this and every read number depends on it.
/// Under the writer's `ROOT_ONLY_MAX_ENTRIES` the whole directory lives in the
/// root, so the leaf lookup, the leaf cache and the second ranged read never
/// run at all, and a sweep whose cells are all in that regime measures one
/// half of the read path and reports it as the read path. It is printed rather
/// than asserted because it is a property of the cell rather than a
/// requirement on it; `the_large_profile_reaches_the_leaf_directory_path` is
/// the assertion.
fn report_directory_shape(storage: &str, cell: Cell, output: &Path) -> Option<u64> {
    if storage != PMTILES {
        return None;
    }
    let (entries, leaves) = root_shape(output);
    println!(
        "{SAY_PREFIX}{storage} {}: root holds {entries} entries, {leaves} of them leaf pointers",
        cell.spec(),
    );
    Some(entries)
}

/// How many entries an archive's root holds and how many of them are leaf
/// pointers, asked of the archive.
///
/// Every question this file asks about an archive's directory shape goes
/// through here rather than through arithmetic over the planner, because the
/// planner's tile count and the writer's entry count are different numbers
/// whenever a run of tiles shares a payload, and because arithmetic over the
/// writer's halving stops being true the day the writer changes.
fn root_shape(output: &Path) -> (u64, u64) {
    let reader = PmTilesPyramidReader::try_open(output).expect("the archive opens for reading");
    let root = reader.reader().root_entries();
    let leaves = root.iter().filter(|entry| entry.is_leaf()).count();
    (root.len() as u64, leaves as u64)
}

/// One cell: generate into one backend, then read it back every way.
fn run_cell(storage: &str, profile: Profile, cell: Cell) -> Vec<Measurement> {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let generated = generate(storage, profile, dir.path(), cell);
    let root_entries = report_directory_shape(storage, cell, &generated.output);
    let mut rows = vec![generated.row.with_root_entries(root_entries)];
    rows.extend(read_scenarios(
        storage,
        profile,
        cell,
        &generated.output,
        &generated.plan,
        root_entries,
    ));
    rows
}

// ---------------------------------------------------------------------------
// The child cell
// ---------------------------------------------------------------------------

/// Parse the `<storage>:<width>x<height>@<tile>` a parent hands a child.
fn parse_spec(spec: &str) -> (String, Cell) {
    let (storage, rest) = spec.split_once(':').unwrap_or_else(|| {
        panic!("a cell spec is <storage>:<width>x<height>@<tile>, got {spec:?}")
    });
    let (canvas, tile) = rest
        .split_once('@')
        .unwrap_or_else(|| panic!("a cell needs a tile size, got {rest:?}"));
    let (width, height) = canvas
        .split_once('x')
        .unwrap_or_else(|| panic!("a canvas is <width>x<height>, got {canvas:?}"));
    (
        storage.to_string(),
        Cell {
            width: width.parse().expect("a canvas width"),
            height: height.parse().expect("a canvas height"),
            tile_size: tile.parse().expect("a tile size"),
        },
    )
}

/// Runs one cell in its own process and prints its rows.
///
/// Selected by [`CELL_VAR`], which the parent sets. Without it this is a
/// no-op, so `cargo test -- --ignored` on this file runs the suite below and
/// nothing here.
#[test]
#[ignore = "a child process of pmtiles_benchmarks; selected by LIBVIPRS_BENCH_CELL"]
#[cfg_attr(miri, ignore)]
fn benchmark_cell() {
    let Ok(spec) = std::env::var(CELL_VAR) else {
        return;
    };
    let out = std::env::var(CELL_OUT_VAR)
        .unwrap_or_else(|_| panic!("{CELL_VAR} is set and {CELL_OUT_VAR} is not"));
    let (storage, cell) = parse_spec(&spec);
    let rows = run_cell(&storage, Profile::from_env(), cell);
    // A bare array, because the envelope belongs to the parent that owns every
    // cell's rows rather than to one cell.
    std::fs::write(&out, bench::rows_to_json(&rows)).expect("a cell can write its rows");
    println!(
        "{storage} {} wrote {} rows to {out}",
        cell.spec(),
        rows.len()
    );
}

/// Splice several children's row arrays into one array.
///
/// Each child writes a complete JSON array of its own rows. Stripping the
/// brackets and joining keeps every record's field order and pretty printing,
/// which re-serialising through `serde_json` would not: its map is a
/// `BTreeMap`, so a round trip alphabetises the columns.
fn splice(documents: &[String]) -> String {
    let bodies: Vec<&str> = documents
        .iter()
        .map(|doc| {
            doc.trim()
                .strip_prefix('[')
                .and_then(|rest| rest.strip_suffix(']'))
                .unwrap_or_else(|| panic!("a cell document is a JSON array, got {doc:?}"))
                // Only the newlines the brackets sat on, so the records keep
                // the two-space indent the rest of the document is written at.
                .trim_start_matches('\n')
                .trim_end()
        })
        .filter(|body| !body.is_empty())
        .collect();
    format!("[\n{}\n]\n", bodies.join(",\n"))
}

/// Run one cell in a fresh copy of this test binary and read back its rows.
fn spawn_cell(storage: &str, cell: Cell, into: &Path) -> String {
    let exe = std::env::current_exe().expect("the test binary knows where it is");
    let output = std::process::Command::new(exe)
        .args(["--exact", "benchmark_cell", "--ignored", "--nocapture"])
        .env(CELL_VAR, format!("{storage}:{}", cell.spec()))
        .env(CELL_OUT_VAR, into)
        .output()
        .expect("the child cell starts");
    assert!(
        output.status.success(),
        "the {storage} {} cell failed:\n{}\n{}",
        cell.spec(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for line in String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix(SAY_PREFIX))
    {
        println!("  {line}");
    }
    let document = std::fs::read_to_string(into).unwrap_or_else(|e| {
        panic!(
            "the {storage} {} cell wrote no rows to {}: {e}\n{}",
            cell.spec(),
            into.display(),
            String::from_utf8_lossy(&output.stdout)
        )
    });
    assert!(
        document.contains("\"scenario\""),
        "the {storage} {} cell wrote a document with no rows in it:\n{document}",
        cell.spec()
    );
    document
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

/// The whole comparison, exported as JSON.
#[test]
#[ignore = "wall-clock benchmark; run with --ignored --nocapture"]
#[cfg_attr(miri, ignore)]
fn pmtiles_versus_directory() {
    let profile = Profile::from_env();
    // Sampled before a cell runs, so the load average is the host's and not
    // this benchmark's own.
    let provenance = bench::Provenance::capture();
    for warning in provenance.measurement_condition_warnings() {
        eprintln!("{warning}");
    }
    println!(
        "measured at {} on {} {}, {} CPUs, load {}, {}",
        provenance.commit.as_deref().unwrap_or("an unknown commit"),
        provenance.host.os,
        provenance.host.arch,
        provenance
            .host
            .ncpu
            .map(|n| n.to_string())
            .unwrap_or_else(|| "an unknown number of".to_string()),
        provenance.load_average_line(),
        provenance
            .rustc_version
            .as_deref()
            .unwrap_or("an unknown rustc"),
    );
    let staging = tempfile::tempdir().expect("a scratch directory for the cell documents");
    let mut documents: Vec<String> = Vec::new();
    for (index, (width, height, tile_size)) in profile.canvases().iter().enumerate() {
        let cell = Cell {
            width: *width,
            height: *height,
            tile_size: *tile_size,
        };
        for storage in [DIRECTORY, PMTILES] {
            let into = staging.path().join(format!("{index}-{storage}.json"));
            documents.push(spawn_cell(storage, cell, &into));
        }
    }
    let document = bench::document(&splice(&documents), &provenance);
    let path = bench::write_document(&document);

    let envelope: serde_json::Value =
        serde_json::from_str(&document).expect("the exported document is JSON");
    let parsed = envelope["rows"]
        .as_array()
        .expect("the document carries a rows array")
        .clone();
    assert!(
        !parsed.is_empty(),
        "the sweep produced no rows at all, which is not a benchmark"
    );

    println!("wrote {} rows to {}", parsed.len(), path.display());
    println!(
        "{:<10} {:<6} {:<22} {:<10} {:>3} {:>12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "canvas",
        "tile",
        "scenario",
        "storage",
        "T",
        "wall_ms",
        "rss_mb",
        "tiles/s",
        "bytes",
        "entries",
        "root",
        "p50_us",
    );
    // A column the row did not measure prints as a dash, the same way it
    // exports as `null`. A zero in this table would read as a measurement.
    let number = |value: &serde_json::Value, places: usize| match value.as_f64() {
        Some(value) => format!("{value:.places$}"),
        None => "-".to_string(),
    };
    let count = |value: &serde_json::Value| match value.as_u64() {
        Some(value) => value.to_string(),
        None => "-".to_string(),
    };
    for row in &parsed {
        println!(
            "{:<10} {:<6} {:<22} {:<10} {:>3} {:>12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}",
            format!(
                "{}x{}",
                row["width"].as_u64().unwrap_or(0),
                row["height"].as_u64().unwrap_or(0)
            ),
            row["tile_size"].as_u64().unwrap_or(0),
            row["scenario"].as_str().unwrap_or("?"),
            row["storage"].as_str().unwrap_or("?"),
            count(&row["concurrency"]),
            number(&row["wall_time_ms"], 2),
            number(&row["peak_rss_mb"], 1),
            number(&row["tiles_per_second"], 1),
            count(&row["output_bytes"]),
            count(&row["filesystem_entries"]),
            count(&row["root_entries"]),
            number(&row["p50_latency_us"], 2),
        );
    }
}

// ---------------------------------------------------------------------------
// The guards that run on every CI job
// ---------------------------------------------------------------------------

const CI_CELL: Cell = Cell {
    width: 512,
    height: 512,
    tile_size: 256,
};

/// The cell the cold-split guard measures against.
///
/// Not a published cell and not in either profile. It exists because the
/// reconciliation needs a root whose decode dominates the open, and
/// [`CI_CELL`]'s root holds twelve entries, so a split that skipped decoding
/// it entirely would still land inside any sane tolerance. 2048 pixels at a 64
/// pixel tile plans about fourteen hundred entries, which is the same root
/// size as the sweep's second cell, and it generates in about the time
/// [`CI_CELL`] does twice over.
const SPLIT_GUARD_CELL: Cell = Cell {
    width: 2048,
    height: 2048,
    tile_size: 64,
};

/// The harness measures something, at a size that costs milliseconds.
///
/// Every `#[ignore]`d test above is invisible to CI by construction, so this is
/// the one that keeps the harness compiling and working. It is also the
/// positive control on the numbers: a row of zeroes and an empty pyramid
/// satisfy every shape assertion, so this asserts the tile counts match the
/// plan, that both backends produced the same tiles, and that the
/// filesystem-entry gap the whole issue is about is really there.
#[test]
#[cfg_attr(miri, ignore)]
fn both_backends_measure_the_same_pyramid() {
    let profile = Profile::Ci;
    let dir = tempfile::tempdir().expect("a scratch directory");

    let tree = generate(DIRECTORY, profile, dir.path(), CI_CELL);
    let archive = generate(PMTILES, profile, dir.path(), CI_CELL);

    let planned = coordinates(&tree.plan).len() as u64;
    assert!(planned > 1, "a 512x512 pyramid has more than one tile");
    assert_eq!(
        tree.row.tiles_produced, planned,
        "the directory run should produce every planned tile"
    );
    assert_eq!(
        archive.row.tiles_produced, planned,
        "the archive run should produce every planned tile"
    );

    assert_eq!(
        archive.row.filesystem_entries,
        Some(1),
        "a PMTiles pyramid is one file"
    );
    let tree_entries = tree
        .row
        .filesystem_entries
        .expect("the tree's occupancy was measured");
    assert!(
        tree_entries > planned,
        "a directory pyramid is at least one entry per tile plus its directories, got \
         {tree_entries}"
    );
    assert!(
        tree.row.output_bytes.is_some_and(|bytes| bytes > 0)
            && archive.row.output_bytes.is_some_and(|bytes| bytes > 0),
        "both backends should have written bytes"
    );
    assert!(
        tree.row.tiles_per_second.is_some_and(|tps| tps > 0.0)
            && archive.row.tiles_per_second.is_some_and(|tps| tps > 0.0),
        "a run that took no measurable time is not a measurement"
    );
    assert_eq!(archive.row.tile_size, CI_CELL.tile_size);
    // The two columns that used to hold the same string.
    assert_eq!(archive.row.engine, bench::ENGINE);
    assert_eq!(archive.row.storage, PMTILES);
    assert_eq!(tree.row.storage, DIRECTORY);

    // The two pyramids hold the same tiles. Byte equality across backends is
    // `tests/pmtiles_pyramid_reader.rs`' job; what this needs is that the
    // benchmark opened two readers over the same content rather than over one
    // full and one empty pyramid.
    let from_tree = open_reader(DIRECTORY, &tree.output, &tree.plan);
    let from_archive = open_reader(PMTILES, &archive.output, &archive.plan);
    for coord in coordinates(&tree.plan) {
        let a = from_tree.tile(coord).expect("a lookup succeeds");
        let b = from_archive.tile(coord).expect("a lookup succeeds");
        assert_eq!(
            a.is_some(),
            b.is_some(),
            "the two backends disagree on whether {coord:?} exists"
        );
    }
}

/// Every read scenario produces a usable row at CI size.
#[test]
#[cfg_attr(miri, ignore)]
fn every_read_scenario_reports_a_row() {
    let profile = Profile::Ci;
    let dir = tempfile::tempdir().expect("a scratch directory");
    let generated = generate(PMTILES, profile, dir.path(), CI_CELL);

    let (root_entries, _) = root_shape(&generated.output);
    let rows = read_scenarios(
        PMTILES,
        profile,
        CI_CELL,
        &generated.output,
        &generated.plan,
        Some(root_entries),
    );
    let scenarios: Vec<&str> = rows.iter().map(|r| r.scenario.as_str()).collect();
    let mut expected = vec!["read_cold"];
    expected.extend(bench::COLD_PHASES);
    expected.extend(["read_warm", "read_sequential", "read_random"]);
    expected.extend(THREAD_LADDER.iter().map(|_| "read_concurrent"));
    assert_eq!(scenarios, expected);

    // The thread curve, with its control. A ladder that lost T=1 still looks
    // like a curve and can no longer separate contention from per-lookup cost,
    // which is the one question it exists to answer.
    let ladder: Vec<usize> = rows
        .iter()
        .filter(|r| r.scenario == "read_concurrent")
        .map(|r| r.concurrency)
        .collect();
    assert_eq!(ladder, THREAD_LADDER.to_vec());
    assert_eq!(
        ladder.first(),
        Some(&1),
        "the concurrent curve has no single-thread control, so nothing in it is a baseline"
    );

    for row in &rows {
        assert!(
            row.tiles_produced > 0,
            "{} read no tiles at all",
            row.scenario
        );
        let phase_without_a_tile = bench::COLD_PHASES.contains(&row.scenario.as_str())
            && row.scenario != "read_cold_lookup";
        if phase_without_a_tile {
            // A phase that does not fetch a tile publishes a hole rather than
            // a zero, which would read as a tile fetched for free.
            assert_eq!(
                row.tile_bytes_returned, None,
                "{} claims to have returned tile bytes",
                row.scenario
            );
        } else {
            assert!(
                row.tile_bytes_returned.is_some_and(|bytes| bytes > 0),
                "{} returned no tile bytes",
                row.scenario
            );
        }
        let p50 = row.p50_latency_us.expect("a read row has a median");
        let p99 = row.p99_latency_us.expect("a read row has a p99");
        assert!(p99 >= p50, "{}: p99 {p99} is below p50 {p50}", row.scenario);
        // A read row measures no pyramid and no raster buffers, and says so.
        assert_eq!(
            row.output_bytes, None,
            "{} priced the pyramid",
            row.scenario
        );
        assert_eq!(
            row.tracked_memory_mb, None,
            "{} charged the raster tracker",
            row.scenario
        );
        assert_eq!(
            row.root_entries,
            Some(root_entries),
            "{} lost the root entry count the ramp is measured against",
            row.scenario
        );
    }
    assert!(
        rows.iter().any(|r| r.concurrency > 1),
        "the concurrent scenario should report its thread count"
    );
}

/// The split's phases are the work the combined row does, shown without a clock.
///
/// This is the assertion that makes the split worth having. Six phases that do
/// not reconcile with the combined row are six numbers about some other piece
/// of work, and they would look exactly as plausible on a chart.
///
/// # Why this is no longer a timing comparison
///
/// It used to measure the cell both ways and allow the sum of the phases to
/// sit within 25% of the combined row. That reads as a reconciliation and is
/// really a race between two independent samples: the split is measured, then
/// the combined is measured, and on a host doing anything else the two land in
/// different conditions. Five consecutive runs of that assertion at one commit
/// on one machine came out at +6.3%, -0.8%, +19.0%, -20.3% and -2.9%: a
/// 39-point spread against a 25% allowance. It failed on a busy host and
/// passed on a quiet one for reasons that had nothing to do with the code.
/// `docs/pmtiles-benchmarks.md` put the p50 noise floor near 8% and chose 25%
/// as three times that, but 8% is a quiet-host number and 25% is not three
/// times the spread this actually has under load.
///
/// Widening the allowance only moves the load at which it lies, and each
/// widening makes it mean less. What is being claimed is that the split walks
/// the same steps over the same bytes, and that is a fact about I/O and about
/// decoded values rather than about duration. Checked that way it holds on any
/// host at any load, and it is a stronger statement than the percentage was:
/// the old form passed just as happily on a split that read the right bytes
/// and threw them away.
///
/// # What the phases are held to
///
/// Three of the six move bytes and are checked against the reads the combined
/// row's own path makes, one for one. Two move no bytes at all, so no read log
/// can see them and they are checked by what they produced instead. The sixth
/// is the open, which nothing needs to assert because every other step reads
/// through the source it returns: a missing open is a missing everything.
///
/// The quantitative form of this claim, that the six durations sum to the
/// combined duration, is worth having and does not belong in this repository.
/// It needs repetitions, a dispersion the run itself measured, and a host that
/// holds still, which is what the storage family in `libviprs-bench` is for.
/// This is the guard left behind.
///
/// The cell is not one the sweep publishes. What the guard needs is a root big
/// enough that a dropped phase moves the total and a generation that costs
/// milliseconds, and a 64 pixel tile over a 2048 pixel canvas gives both. The
/// root size is asserted rather than assumed, because a cell that quietly
/// shrank would leave a guard that passes against a split with a phase
/// missing.
#[test]
#[cfg_attr(miri, ignore)]
fn the_cold_split_accounts_for_the_whole_combined_row() {
    let profile = Profile::Ci;
    let dir = tempfile::tempdir().expect("a scratch directory");
    let generated = generate(PMTILES, profile, dir.path(), SPLIT_GUARD_CELL);
    let (root_entries, leaves) = root_shape(&generated.output);
    assert_eq!(leaves, 0, "the split guard wants a flat root to decode");
    assert!(
        root_entries >= 1_000,
        "the split guard's cell decodes {root_entries} entries, which is too small a root for a \
         dropped phase to show up in the total"
    );

    let coords: Vec<TileCoord> = coordinates(&generated.plan)
        .into_iter()
        .take(COLD_SAMPLES)
        .collect();
    let coord = *coords.first().expect("the plan produced a coordinate");

    // The combined row's own path, over a source that writes down what it was
    // asked for. The substitution is exact rather than close:
    // `PmTilesPyramidReader::try_open` is `pmtiles::Reader::try_open` and its
    // `tile` is `tile_coord_to_zxy` then `get_tile` (`src/pyramid_reader.rs`),
    // and `Reader::try_open(path)` is `Reader::try_new(FileRangeReader::
    // try_open(path)?)` (`src/pmtiles/reader.rs`). Wrapping the file source
    // changes which reads are written down and none of which reads happen.
    let combined_log = read_tally();
    let reader = libviprs::pmtiles::Reader::try_new(ReadLog::sharing(
        FileRangeReader::try_open(&generated.output).expect("the archive opens"),
        std::sync::Arc::clone(&combined_log),
    ))
    .expect("the archive opens for reading");
    let opening = tallied(&combined_log);
    let (z, x, y) = libviprs::sink_pmtiles::tile_coord_to_zxy(coord)
        .expect("a coordinate the plan produced is addressable in PMTiles");
    let combined_tile = reader.get_tile(z, x, y).expect("a lookup succeeds");
    let lookup: Vec<(u64, usize)> = tallied(&combined_log)[opening.len()..].to_vec();

    // The split's own steps, run rather than described a second time, through
    // a source that keeps the same kind of log.
    let split_log = read_tally();
    let split = {
        let log = std::sync::Arc::clone(&split_log);
        split_iteration_over(&generated.output, coord, move |path| {
            ReadLog::sharing(
                FileRangeReader::try_open(path).expect("the archive opens"),
                std::sync::Arc::clone(&log),
            )
        })
    };

    // The reads the split's timed phases make are the reads a cold open makes:
    // same ranges, same order, and the same number of them. This is the
    // reconciliation the percentage was standing in for, and it is the half a
    // products-only check would miss, because a phase that reads the right
    // bytes twice is doing work the combined row does not.
    assert_eq!(
        tallied(&split_log),
        opening,
        "the split's timed phases do not read what opening the archive reads"
    );

    // The split's lookup phase runs through `PmTilesPyramidReader`, which is
    // the crate's own code and therefore cannot read differently from the
    // combined row's `get_tile`; what it returned is checked below instead.

    // Phase 2, the header: the combined row's first read, and the same header
    // came back out of it.
    assert_eq!(
        opening.first().copied(),
        Some((0, libviprs::pmtiles::header::HEADER_BYTES)),
        "a cold open starts by reading the header, and the split times a read of that same range"
    );
    assert_eq!(
        split.header,
        *reader.header(),
        "the split decoded a different header from the one the reader went on to use, so every \
         phase after it is addressing a different archive"
    );

    // Phase 3, the root fetch: the combined row's second read is exactly the
    // range the split's own header says the root lives at.
    assert_eq!(
        opening.get(1).copied(),
        Some((
            split.header.root_offset,
            usize::try_from(split.header.root_length).expect("a root length fits a usize")
        )),
        "the combined row's second read is not the root range the split fetched"
    );
    assert_eq!(
        opening.len(),
        2,
        "a cold open is the header and the root and nothing else, so a split of six phases that \
         covers more than two reads before the lookup is covering work the row does not do: \
         {opening:?}"
    );

    // Phases 4 and 5, inflate and decode. Neither touches the archive, so the
    // only thing that can say they ran is what came out of them. A skipped
    // inflate hands the decoder gzip and a skipped decode has no entries, and
    // both land here rather than inside a percentage.
    assert_eq!(
        split.entries.as_slice(),
        reader.root_entries(),
        "the split's inflate and decode did not reproduce the root directory the reader built"
    );

    // Phase 6, the lookup: one payload read, and the same bytes.
    assert_eq!(
        lookup.len(),
        1,
        "a leafless archive needs exactly one read for the payload: {lookup:?}"
    );
    assert!(
        combined_tile.is_some(),
        "the guard's coordinate has to be a tile the archive holds, or the lookup phase is timing \
         a miss and the payload read above is not the one a caller pays for"
    );
    assert_eq!(
        split.tile, combined_tile,
        "the split's lookup returned different bytes from the combined row's"
    );

    // And every phase really ran, for every coordinate the pass walks. The
    // reconciliation above is one iteration; this is the shape of all of them,
    // and it is what catches a phase cheap enough to hide.
    let pass = cold_split_pass(&generated.output, &coords);
    assert_eq!(pass.phases.len(), bench::COLD_PHASES.len());
    for (phase, samples) in bench::COLD_PHASES.iter().zip(&pass.phases) {
        assert_eq!(
            samples.len(),
            coords.len(),
            "{phase} measured {} of {} opens",
            samples.len(),
            coords.len()
        );
    }
}

/// The large profile reaches an archive with leaf directories, and the CI
/// profile deliberately does not.
///
/// This is the one property of the sweep that cannot be seen by reading a
/// result row, and it is the one most likely to be wrong. Under
/// `ROOT_ONLY_MAX_ENTRIES` entries the writer puts the whole directory in the
/// root, so an archive of a few thousand tiles never exercises the leaf
/// lookup, the leaf cache or the second ranged read at all. A sweep whose
/// largest cell stayed under the cutoff would be measuring one half of the
/// read path and reporting it as the read path.
///
/// It counts planned tiles rather than directory entries, which can only be
/// fewer: a run of consecutive tile ids carrying one payload collapses into
/// one entry. The benchmark's source is a gradient, whose tiles are distinct,
/// so the two numbers are the same here.
#[test]
fn the_large_profile_reaches_the_leaf_directory_path() {
    let biggest = Profile::Large
        .canvases()
        .iter()
        .map(|(w, h, t)| coordinates(&plan_for(*w, *h, *t)).len())
        .max()
        .expect("the large profile has cells");
    assert!(
        biggest > ROOT_ONLY_MAX_ENTRIES,
        "the largest cell in the large profile plans {biggest} tiles, under the \
         {ROOT_ONLY_MAX_ENTRIES} entry cutoff, so no cell in the sweep produces a leaf directory"
    );

    // And it reaches the other side of the same cutoff. Before the brink cell
    // the sweep's root-only cells sat at 93, 1373 and 5469 entries, so it
    // bracketed the worst case from a third of the way down and the peak of
    // the ramp was a line fitted through three points.
    let closest = Profile::Large
        .canvases()
        .iter()
        .map(|(w, h, t)| {
            planned_tiles(Cell {
                width: *w,
                height: *h,
                tile_size: *t,
            })
        })
        .filter(|tiles| *tiles <= LARGEST_FLAT_ROOT)
        .max()
        .expect("the large profile has a cell inside the cutoff");
    assert!(
        LARGEST_FLAT_ROOT - closest < LARGEST_FLAT_ROOT / 100,
        "the closest the large profile gets to the {LARGEST_FLAT_ROOT} entry cutoff from below \
         is {closest} tiles, so the worst case is still an extrapolation"
    );

    for (w, h, t) in Profile::Ci.canvases() {
        let tiles = coordinates(&plan_for(*w, *h, *t)).len();
        assert!(
            tiles < ROOT_ONLY_MAX_ENTRIES,
            "the CI profile's {w}x{h}@{t} cell plans {tiles} tiles, which is not the cheap \
             profile it is meant to be"
        );
    }
}

/// How many tiles a cell plans, without materialising every coordinate.
fn planned_tiles(cell: Cell) -> usize {
    plan_for(cell.width, cell.height, cell.tile_size)
        .levels
        .iter()
        .map(|level| level.cols as usize * level.rows as usize)
        .sum()
}

/// The cell the brink search picks, and the space it searches.
///
/// Tile sizes from 16 pixels, because below that a tile stops resembling
/// anything anybody ships and its payload stops resembling a tile's. Canvases
/// up to 4096 pixels on the short edge and twice that on the long one, so the
/// brink cell costs less to generate than the sweep's existing biggest cell
/// rather than more. Heights are found by binary search, which is sound
/// because a plan's tile count never falls as the canvas grows.
///
/// It is arithmetic, and arithmetic is exactly what must not be trusted to pin
/// the cell: this says which cell to build and
/// `the_brink_cells_root_stops_just_under_the_writers_cutoff` says what the
/// archive actually came out as.
fn brink_search() -> (Cell, usize) {
    let mut best: Option<(Cell, usize)> = None;
    for tile_size in 16u32..=256 {
        for width in [256u32, 512, 1024, 2048, 4096] {
            let fits = |height: u32| {
                planned_tiles(Cell {
                    width,
                    height,
                    tile_size,
                }) <= LARGEST_FLAT_ROOT
            };
            if !fits(width) {
                continue;
            }
            let (mut low, mut high) = (width, 2 * width);
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if fits(mid) {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            let cell = Cell {
                width,
                height: low,
                tile_size,
            };
            let tiles = planned_tiles(cell);
            if best.is_none_or(|(_, most)| tiles > most) {
                best = Some((cell, tiles));
            }
        }
    }
    best.expect("some cell in the search space plans a pyramid")
}

/// The writer's cutoff is still the number this file thinks it is.
///
/// [`ROOT_ONLY_MAX_ENTRIES`] here is a copy: the writer's is private, so
/// nothing links the two and a change there would leave every assertion in
/// this file naming a cutoff that moved. The source is read instead, the way
/// `tests/pmtiles_release_readiness.rs` reads the files it makes claims about,
/// so the copy cannot drift in silence.
///
/// The comparison is scraped too, and it is the reason
/// [`LARGEST_FLAT_ROOT`] is one less than the constant. `<` means 16384
/// entries already spill into leaves; a `<=` would move the brink by one and
/// nothing else in this file would notice.
#[test]
fn the_writers_root_only_cutoff_is_the_one_this_file_copied() {
    const WRITER: &str = include_str!("../src/pmtiles/writer.rs");
    let declaration = format!("const ROOT_ONLY_MAX_ENTRIES: u64 = {ROOT_ONLY_MAX_ENTRIES};");
    assert!(
        WRITER.contains(&declaration),
        "src/pmtiles/writer.rs no longer declares `{declaration}`, so every cutoff this file \
         names is a number from somewhere else"
    );
    assert!(
        WRITER.contains("if plan.entry_count < ROOT_ONLY_MAX_ENTRIES"),
        "the writer no longer decides a flat root with `plan.entry_count < \
         ROOT_ONLY_MAX_ENTRIES`, so the largest flat root may not be \
         {LARGEST_FLAT_ROOT} any more"
    );
}

/// The brink cell is still the best the search space has.
///
/// Cheap: the whole sweep is fifteen thousand plans and a couple of
/// milliseconds, because a plan is arithmetic over level sizes and nothing is
/// generated. It is the guard on the *choice* of cell rather than on what the
/// archive came out as, and it fails the day the planner's rounding changes
/// and some other cell gets closer to the cutoff.
#[test]
fn the_brink_cell_is_the_largest_root_the_search_space_reaches() {
    let (found, tiles) = brink_search();
    let configured = Cell {
        width: bench::BRINK_CANVAS.0,
        height: bench::BRINK_CANVAS.1,
        tile_size: bench::BRINK_CANVAS.2,
    };
    assert_eq!(
        (found.width, found.height, found.tile_size),
        (configured.width, configured.height, configured.tile_size),
        "the search space's best cell is now {}, planning {tiles} tiles, and the profile still \
         walks {}",
        found.spec(),
        configured.spec(),
    );
    assert!(
        tiles <= LARGEST_FLAT_ROOT,
        "the brink cell plans {tiles} tiles, at or over the {LARGEST_FLAT_ROOT} entries the \
         writer will still keep in a flat root"
    );
    // And it really is at the brink rather than merely under it. 1% of the
    // cutoff is about 164 entries, and stepping this cell's canvas by a pixel
    // moves its tile count by roughly 150, so anything much wider than that
    // means the search stopped finding what it used to.
    let short_by = LARGEST_FLAT_ROOT - tiles;
    assert!(
        short_by * 100 < LARGEST_FLAT_ROOT,
        "the brink cell plans {tiles} tiles, {short_by} short of the {LARGEST_FLAT_ROOT} entry \
         cutoff, which is far enough from it that the cell no longer measures the peak of the \
         ramp"
    );
}

/// The brink cell's archive really does stop one step under the cutoff.
///
/// This is the pin, and it is a pin because it asks the archive. The cell is a
/// planner's tile count; what the ramp is a function of is the writer's
/// **entry** count, and those two are the same number only while no run of
/// neighbouring tiles shares a payload. Asserting `root_entries().len()`
/// against the writer's own cutoff catches the day either half moves: a
/// planner that rounds differently, a writer whose cutoff changes, or a source
/// whose tiles start deduplicating.
///
/// `#[ignore]`d for the same reason everything else at this scale is: it
/// generates a fifty megabyte archive, which is half a second in release and
/// long enough in a debug CI job to be worth nobody's time. The large profile
/// runs the same check on every sweep through
/// [`report_directory_shape`]'s count, and this is the way to ask it on
/// demand:
///
/// ```text
/// cargo test --release --test pmtiles_benchmarks -- --ignored \
///   the_brink_cells_root_stops_just_under_the_writers_cutoff --nocapture
/// ```
#[test]
#[ignore = "generates the brink archive; run with --ignored"]
#[cfg_attr(miri, ignore)]
fn the_brink_cells_root_stops_just_under_the_writers_cutoff() {
    let cell = Cell {
        width: bench::BRINK_CANVAS.0,
        height: bench::BRINK_CANVAS.1,
        tile_size: bench::BRINK_CANVAS.2,
    };
    let dir = tempfile::tempdir().expect("a scratch directory");
    let generated = generate(PMTILES, Profile::Large, dir.path(), cell);
    let (entries, leaves) = root_shape(&generated.output);
    println!(
        "{} planned {} tiles and its root holds {entries} entries, {leaves} of them leaf \
         pointers, {} under the {LARGEST_FLAT_ROOT} the writer still keeps flat",
        cell.spec(),
        planned_tiles(cell),
        LARGEST_FLAT_ROOT as u64 - entries,
    );

    assert_eq!(
        leaves, 0,
        "the brink cell's archive grew leaf directories, so its root is a handful of pointers \
         and it measures the far side of the cliff rather than the top of the ramp"
    );
    assert!(
        (entries as usize) <= LARGEST_FLAT_ROOT,
        "the brink cell's root holds {entries} entries, at or past the {LARGEST_FLAT_ROOT} the \
         writer will keep flat, which is not a root this writer emits"
    );
    let short_by = LARGEST_FLAT_ROOT as u64 - entries;
    assert!(
        short_by * 100 < LARGEST_FLAT_ROOT as u64,
        "the brink cell's root holds {entries} entries, {short_by} under the \
         {LARGEST_FLAT_ROOT} entry cutoff, so the sweep brackets the worst case again instead \
         of measuring it"
    );

    // Tiles and entries are the same number here, and that is a measurement
    // rather than an assumption: the writer counts run-length-encoded entries,
    // so a source whose neighbouring tiles shared a payload would put far
    // fewer entries in the root than the planner plans tiles.
    assert_eq!(
        entries as usize,
        planned_tiles(cell),
        "the gradient's tiles are meant to be all distinct, so every planned tile should cost \
         one root entry; a run collapsed, and the root is smaller than the cell's tile count"
    );
}

/// The exported document is the shape libviprs.org reads, parsed by something
/// other than the code that wrote it.
///
/// This is the contract with libviprs.org (issue #62). A field renamed here is
/// a chart that silently stops drawing there, and nothing else in either
/// repository would catch it.
#[test]
#[cfg_attr(miri, ignore)]
fn the_exported_json_carries_every_field_the_site_reads() {
    let rows = vec![
        Measurement::new(
            "generate",
            PMTILES,
            Profile::Ci,
            2048,
            2048,
            256,
            1,
            Duration::from_millis(1234),
            Some(5 * 1024 * 1024),
            Some(90 * 1024 * 1024),
            349,
        )
        .with_output(Some((4_194_304, 1)))
        // The root the 16384 pixel cell's archive came out with, so the
        // fixture carries a real count rather than a round number.
        .with_root_entries(Some(5_469)),
        Measurement::new(
            "read_random",
            DIRECTORY,
            Profile::Large,
            2048,
            2048,
            64,
            8,
            Duration::from_micros(9_876),
            None,
            None,
            512,
        )
        .with_latencies(
            &mut [
                Duration::from_micros(3),
                Duration::from_micros(11),
                Duration::from_micros(40),
            ],
            123_456,
        ),
    ];

    let text = bench::to_json(&rows, &bench::Provenance::capture());
    let parsed: serde_json::Value =
        serde_json::from_str(&text).expect("the exported document is JSON");
    let envelope = parsed.as_object().expect("the document is an object");

    // The envelope, which is what lets a consumer refuse a shape it does not
    // understand rather than read a renamed column as absent. The key set is
    // read off the parsed value and the order off the text, because a
    // `serde_json::Map` is a `BTreeMap` and has no order left to check.
    let mut keys: Vec<&str> = envelope.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["provenance", "rows", "schema"],
        "the document is a schema, where it came from, and its rows, and nothing else"
    );
    assert!(
        text.find("\"schema\"") < text.find("\"provenance\""),
        "the schema should come first, so a consumer can decide before it reads anything else"
    );
    assert!(
        text.find("\"provenance\"") < text.find("\"rows\""),
        "the provenance should come before the rows it describes"
    );
    assert_eq!(envelope["schema"], serde_json::json!(bench::SCHEMA_VERSION));
    let array = envelope["rows"].as_array().expect("rows is an array");
    assert_eq!(array.len(), 2, "one object per measurement");

    for record in array {
        let object = record.as_object().expect("every record is an object");
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        for field in bench::FIELDS {
            assert!(
                object.contains_key(field),
                "the exported record is missing {field}; it has {keys:?}"
            );
        }
        assert_eq!(
            object.len(),
            bench::FIELDS.len(),
            "the record carries a field bench::FIELDS does not list: {keys:?}"
        );

        for field in bench::FIELDS {
            let value = &object[field];
            if bench::STRING_FIELDS.contains(&field) {
                assert!(value.is_string(), "{field} should be a string, got {value}");
                continue;
            }
            if value.is_null() {
                assert!(
                    bench::NULLABLE_FIELDS.contains(&field),
                    "{field} is null, and it is not a column a row is allowed to leave unmeasured"
                );
                continue;
            }
            assert!(value.is_number(), "{field} should be a number, got {value}");
            assert!(
                value.as_f64().expect("a number").is_finite(),
                "{field} is not finite: {value}"
            );
        }
    }

    // `engine` says which engine, `storage` says which backend, and they are
    // no longer the same string under two names.
    let generation = array[0].as_object().expect("an object");
    assert_eq!(generation["engine"], bench::ENGINE);
    assert_eq!(generation["storage"], PMTILES);
    assert_ne!(generation["engine"], generation["storage"]);
    assert_eq!(generation["filesystem_entries"], 1);
    assert_eq!(generation["tile_size"], 256);
    assert!(
        generation["p50_latency_us"].is_null(),
        "a generation row measures no latency, so it publishes no latency"
    );

    // The derived columns are derived the way the existing producer derives
    // them, which is what makes a row here comparable with a row there.
    let rss_mb = generation["peak_rss_mb"].as_f64().expect("a number");
    let tps = generation["tiles_per_second"].as_f64().expect("a number");
    let per_mb = generation["tiles_per_second_per_mb"]
        .as_f64()
        .expect("a number");
    assert!(
        (per_mb - tps / rss_mb).abs() < 1e-9,
        "tiles_per_second_per_mb should be tiles_per_second over peak_rss_mb"
    );

    // And a row that measured no RSS publishes holes, not the flattering zero
    // that used to be the best score on `resource_cost`.
    let read = array[1].as_object().expect("an object");
    for field in [
        "peak_rss_mb",
        "tracked_memory_mb",
        "tiles_per_second_per_mb",
        "resource_cost",
        "output_bytes",
        "filesystem_entries",
        // A directory row has no root to decode, and a root of nothing would
        // be a free open on the column the whole ramp is plotted against.
        "root_entries",
    ] {
        assert!(
            read[field].is_null(),
            "{field} on an unmeasured row is {} rather than null",
            read[field]
        );
    }
    assert_eq!(read["tile_bytes_returned"], 123_456);
    assert_eq!(generation["root_entries"], 5_469);
}

/// The envelope says where the numbers came from.
///
/// The published PMTiles figures carried no commit, no host, no CPU count and
/// no load average, and their own prose said "amd64 container" on a machine
/// where that means Rosetta. A ramp measured on a host nobody can identify is
/// the mistake issue #1021 exists to correct, so the shape of the attestation
/// is checked the same way the row shape is: parsed back, key by key.
#[test]
#[cfg_attr(miri, ignore)]
fn the_envelope_says_which_host_produced_the_numbers() {
    let provenance = bench::Provenance::capture();
    let text = bench::to_json(&[sample_row("generate", PMTILES)], &provenance);
    let parsed: serde_json::Value =
        serde_json::from_str(&text).expect("the exported document is JSON");
    let object = parsed["provenance"]
        .as_object()
        .expect("the provenance is an object");

    let keys: Vec<&str> = object.keys().map(String::as_str).collect();
    for field in bench::PROVENANCE_FIELDS {
        assert!(
            object.contains_key(field),
            "the provenance is missing {field}; it has {keys:?}"
        );
    }
    assert_eq!(
        object.len(),
        bench::PROVENANCE_FIELDS.len(),
        "the provenance carries a field bench::PROVENANCE_FIELDS does not list: {keys:?}"
    );

    let host = object["host"].as_object().expect("the host is an object");
    let host_keys: Vec<&str> = host.keys().map(String::as_str).collect();
    for field in bench::PROVENANCE_HOST_FIELDS {
        assert!(
            host.contains_key(field),
            "the provenance's host is missing {field}; it has {host_keys:?}"
        );
    }
    assert_eq!(host.len(), bench::PROVENANCE_HOST_FIELDS.len());

    // The two that are known on every platform this builds for, and the one
    // that is a compile-time fact rather than a guess.
    assert_eq!(host["arch"], std::env::consts::ARCH);
    assert_eq!(host["os"], std::env::consts::OS);
    assert_eq!(
        object["build_profile"],
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    assert!(host["in_container"].is_boolean());

    // The commit is a string or a hole and never anything else. Which one it
    // is depends on how this checkout is mounted rather than on the code:
    // a linked worktree bind-mounted into a container has a `.git` file
    // pointing outside the mount, and git answers "not a git repository"
    // there. `a_repository_with_a_commit_is_read_back` is the control that the
    // reading works.
    assert!(
        object["commit"].is_string() || object["commit"].is_null(),
        "commit is {} rather than a commit or a hole",
        object["commit"]
    );
    assert_eq!(
        object["commit"].is_null(),
        object["dirty"].is_null(),
        "a commit with no dirty flag, or the reverse, is half an attestation"
    );

    // And a run that measured nothing about the host says so in holes rather
    // than in zeroes: an ncpu of 0 is a machine that cannot run a benchmark.
    let unknown = bench::Provenance::unknown();
    let text = bench::to_json(&[sample_row("generate", PMTILES)], &unknown);
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    for field in ["commit", "dirty", "rustc_version", "load_average"] {
        assert!(
            parsed["provenance"][field].is_null(),
            "{field} on an unattested run is {} rather than null",
            parsed["provenance"][field]
        );
    }
    assert!(parsed["provenance"]["host"]["ncpu"].is_null());
    assert!(parsed["provenance"]["host"]["cpu_model"].is_null());
}

/// A tree with a commit in it reads back as that commit, and an edit to it
/// reads back as dirty.
///
/// The control on the provenance's git half. It builds its own repository
/// rather than asking about this one, because how this one is checked out is
/// not a property of the code: in a linked worktree mounted into a container
/// the commit is genuinely unreadable and `null` is the right answer, so an
/// assertion against this tree would be testing the mount.
///
/// Where git is missing entirely the control asserts the other half, that the
/// harness answers holes. It never skips: a skipped test is the same colour as
/// a passing one, and this is the field whose whole job is to say "nobody can
/// reproduce this".
#[test]
#[cfg_attr(miri, ignore)]
fn a_repository_with_a_commit_is_read_back() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let repo = dir.path();

    if !bench::git_is_available() {
        assert_eq!(
            bench::commit_and_dirty(repo),
            (None, None),
            "with no git to ask, the provenance must publish holes rather than invent a commit"
        );
        return;
    }

    for args in [
        vec!["init", "--quiet"],
        vec!["config", "user.email", "bench@example.invalid"],
        vec!["config", "user.name", "bench"],
    ] {
        bench::git_in(repo, &args).unwrap_or_else(|| panic!("git {args:?} should succeed"));
    }
    std::fs::write(repo.join("measured.txt"), "one\n").expect("the file is written");
    bench::git_in(repo, &["add", "measured.txt"]).expect("the file stages");
    bench::git_in(repo, &["commit", "--quiet", "-m", "one"]).expect("the commit lands");

    let (commit, dirty) = bench::commit_and_dirty(repo);
    let commit = commit.expect("a repository with a commit names one");
    assert!(
        commit.len() >= 7 && commit.chars().all(|c| c.is_ascii_hexdigit()),
        "{commit:?} does not look like a short commit"
    );
    assert_eq!(dirty, Some(false), "a freshly committed tree is not dirty");

    std::fs::write(repo.join("measured.txt"), "two\n").expect("the file is edited");
    let (again, dirty) = bench::commit_and_dirty(repo);
    assert_eq!(
        again,
        Some(commit),
        "editing a tracked file does not move the commit, which is the whole problem"
    );
    assert_eq!(
        dirty,
        Some(true),
        "a tree edited since its commit is dirty, and the commit no longer describes it"
    );
}

/// A run taken under conditions that spoil it says so out loud.
#[test]
fn a_run_whose_conditions_spoil_it_warns() {
    let mut prov = bench::Provenance::unknown();
    // Nothing known at all: the commit warning fires and nothing else claims
    // to know something it does not.
    let warnings = prov.measurement_condition_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("no commit")),
        "a document with no commit should say so: {warnings:?}"
    );
    assert!(
        !prov.host_looked_contended(),
        "a host with no load sample and no CPU count must not be called contended"
    );

    prov.commit = Some("d4f13924".to_string());
    prov.host.ncpu = Some(8);
    prov.load_average = Some(bench::LoadAverage {
        one_min: 9.5,
        five_min: 8.0,
        fifteen_min: 4.0,
    });
    prov.dirty = Some(true);
    prov.build_profile = "release";
    assert!(prov.host_looked_contended());
    let warnings = prov.measurement_condition_warnings();
    assert!(warnings.iter().any(|w| w.contains("1-minute host load")));
    assert!(warnings.iter().any(|w| w.contains("uncommitted changes")));

    // An idle host on a clean tree in release has nothing to say.
    prov.dirty = Some(false);
    prov.load_average = Some(bench::LoadAverage {
        one_min: 0.5,
        five_min: 0.4,
        fifteen_min: 0.3,
    });
    assert_eq!(prov.measurement_condition_warnings(), Vec::<String>::new());
}

/// Both platforms' spellings of a load average parse.
///
/// Linux writes `0.52 0.58 0.59 1/523 12345` and macOS' `sysctl` writes
/// `{ 1.83 1.92 1.98 }`. A parser that split on whitespace alone would read
/// macOS' first field as `{` and answer `None` on the host this repository is
/// developed on, which is the one place the miss would look like "this
/// platform has no load average".
#[test]
fn a_load_average_parses_on_both_platforms_spellings() {
    let linux = bench::parse_load_average("0.52 0.58 0.59 1/523 12345\n")
        .expect("the Linux spelling parses");
    assert!((linux.one_min - 0.52).abs() < 1e-9);
    assert!((linux.fifteen_min - 0.59).abs() < 1e-9);

    let macos =
        bench::parse_load_average("{ 1.83 1.92 1.98 }\n").expect("the macOS spelling parses");
    assert!((macos.one_min - 1.83).abs() < 1e-9);
    assert!((macos.five_min - 1.92).abs() < 1e-9);
    assert!((macos.fifteen_min - 1.98).abs() < 1e-9);

    assert!(bench::parse_load_average("").is_none());
    assert!(bench::parse_load_average("1.0 2.0").is_none());
}

/// A row measured on a platform with no `/proc` publishes a hole.
#[test]
fn a_platform_without_a_peak_rss_reports_null_not_a_flattering_zero() {
    assert_eq!(phase_peak_rss(false), None);
    let row = Measurement::new(
        "generate",
        PMTILES,
        Profile::Ci,
        512,
        512,
        256,
        1,
        Duration::from_millis(10),
        None,
        phase_peak_rss(false),
        5,
    );
    assert_eq!(row.peak_rss_mb, None);
    // And the two ratios that divide by it are holes rather than zeroes. On
    // `resource_cost` lower is better, so a zero was the best possible score
    // and every platform without `/proc` published one as a measurement.
    assert_eq!(row.tiles_per_second_per_mb, None);
    assert_eq!(row.resource_cost, None);

    let text = bench::to_json(&[row], &bench::Provenance::unknown());
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    for field in ["peak_rss_mb", "tiles_per_second_per_mb", "resource_cost"] {
        assert!(parsed["rows"][0][field].is_null(), "{field} should be null");
    }
}

/// A pyramid the harness cannot stat is a hole, not one entry of nothing.
#[test]
#[cfg_attr(miri, ignore)]
fn an_unmeasurable_pyramid_publishes_no_entry_count() {
    let missing = std::path::Path::new("/this/path/does/not/exist/pyramid.pmtiles");
    assert_eq!(
        bench::occupancy(missing),
        None,
        "a path that cannot be stat'd has no occupancy"
    );

    let row = Measurement::new(
        "generate",
        PMTILES,
        Profile::Ci,
        512,
        512,
        256,
        1,
        Duration::from_millis(10),
        None,
        None,
        5,
    )
    .with_output(bench::occupancy(missing));
    assert_eq!(row.filesystem_entries, None);
    assert_eq!(row.output_bytes, None);

    // The control: a path that does exist is measured rather than skipped.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let file = dir.path().join("pyramid.pmtiles");
    std::fs::write(&file, b"not really an archive").expect("the file is written");
    assert_eq!(bench::occupancy(&file), Some((21, 1)));
}

/// The 64 pixel cell plans the tile count the doc and the CHANGELOG publish.
///
/// The doc said 21845 in its prose and 21851 in its tables, and both numbers
/// are real: 21845 is the full-pyramid count for 8192 pixels at 64 pixel
/// tiles if the pyramid stops at a one-tile level, and the planner keeps
/// halving the source down to a single pixel, which adds six more one-tile
/// levels. Nothing checked either. This is the three-line guard that settles
/// it, and it is the shape of guard this repository already writes.
#[test]
fn the_eight_thousand_pixel_cell_plans_the_tile_count_the_doc_publishes() {
    assert_eq!(coordinates(&plan_for(8192, 8192, 64)).len(), 21_851);
    // The 21845 the prose used to carry, and where it comes from: the levels
    // whose source is at least one tile wide.
    let full_levels: usize = (0..8).map(|level| 1usize << (2 * level)).sum();
    assert_eq!(full_levels, 21_845);
    assert_eq!(
        plan_for(8192, 8192, 64).levels.len(),
        14,
        "the planner halves to a single pixel, which is where the other six tiles come from"
    );
}

/// A child's document splices into the parent's, parsed rather than eyeballed.
///
/// This is the hand-off the first version of this file got wrong, and the
/// thing about it worth remembering is that both halves were individually
/// fine: the child serialised valid JSON and the parent parsed valid JSON.
/// What was broken was the carrier between them, which no test touched
/// because neither side owns it. It is also a two-microsecond test standing in
/// for a ten-minute benchmark run, which is the only reason the bug cost ten
/// minutes rather than one afternoon.
#[test]
fn a_child_row_file_splices_into_the_document() {
    let one = bench::rows_to_json(&[sample_row("generate", PMTILES)]);
    let two = bench::rows_to_json(&[
        sample_row("read_cold", DIRECTORY),
        sample_row("read_warm", DIRECTORY),
    ]);

    // Each half is a document in its own right.
    for document in [&one, &two] {
        serde_json::from_str::<Vec<serde_json::Value>>(document)
            .expect("a cell writes a JSON array");
    }
    assert!(
        one.contains('\n'),
        "a row is pretty printed over several lines"
    );

    let spliced = splice(&[one, two]);
    let document = bench::document(&spliced, &bench::Provenance::unknown());
    let envelope: serde_json::Value =
        serde_json::from_str(&document).expect("the spliced document is JSON");
    assert_eq!(envelope["schema"], serde_json::json!(bench::SCHEMA_VERSION));
    let parsed = envelope["rows"].as_array().expect("rows is an array");
    assert_eq!(parsed.len(), 3, "one record per row, from both documents");
    assert_eq!(parsed[0]["scenario"], "generate");
    assert_eq!(parsed[2]["scenario"], "read_warm");
    // Field order survives the splice, which is the reason it is a splice
    // rather than a re-serialisation. It has to be read off the text: a
    // `serde_json::Map` is a `BTreeMap`, so the parsed value has no order left
    // to check.
    let width_at = document.find("\"width\"").expect("the width column");
    let engine_at = document.find("\"engine\"").expect("the engine column");
    let cost_at = document
        .find("\"resource_cost\"")
        .expect("the resource_cost column");
    assert!(
        width_at < engine_at && engine_at < cost_at,
        "the exported text should keep the column order bench::FIELDS lists"
    );
}

fn sample_row(scenario: &str, storage: &str) -> Measurement {
    Measurement::new(
        scenario,
        storage,
        Profile::Ci,
        2048,
        2048,
        256,
        1,
        Duration::from_millis(7),
        Some(1024),
        Some(2048),
        42,
    )
}

/// A cell spec survives the trip through the environment variable.
///
/// The parent and the child agree on this string and nothing else, so a
/// format change on one side is a child that panics on a spec it cannot read,
/// inside a process whose stderr the parent only prints on failure.
#[test]
fn a_cell_spec_round_trips_through_its_string_form() {
    let cell = Cell {
        width: 8192,
        height: 4096,
        tile_size: 64,
    };
    let (storage, parsed) = parse_spec(&format!("{PMTILES}:{}", cell.spec()));
    assert_eq!(storage, PMTILES);
    assert_eq!(parsed.width, cell.width);
    assert_eq!(parsed.height, cell.height);
    assert_eq!(parsed.tile_size, cell.tile_size);
}
