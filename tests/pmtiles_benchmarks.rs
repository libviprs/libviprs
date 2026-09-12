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
//! of a sample of many, and the read numbers want a cold cache that repeated
//! iterations destroy by construction. So this reuses the `#[ignore]`
//! wall-clock convention `src/colour.rs` already uses and the
//! [`MemoryTracker`](libviprs::MemoryTracker) the engine already reports
//! through [`EngineResult::peak_memory_bytes`].
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
//! `peak_rss_mb` is `0.0`. That is a column the platform cannot fill, not a
//! number invented to fill it, and it is why the published figures come from
//! the Linux container.
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

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pyramid_reader::{DirectoryPyramidReader, PmTilesPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::{EngineBuilder, FsSink};

#[path = "common/pmtiles_bench.rs"]
mod bench;

use bench::{Measurement, Profile, Splitmix};

/// Environment variable a child cell reads to learn what to measure.
const CELL_VAR: &str = "LIBVIPRS_BENCH_CELL";

/// Prefix a child prints one serialised row behind.
const ROW_PREFIX: &str = "BENCHROW ";

/// The two backends, under the names the exported rows carry.
const DIRECTORY: &str = "directory";
const PMTILES: &str = "pmtiles";

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
/// [`reset_peak_rss`], or `0` where the platform has no answer.
fn phase_peak_rss(reset_worked: bool) -> u64 {
    if reset_worked {
        bench::peak_rss_bytes_or_zero()
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Generating
// ---------------------------------------------------------------------------

fn plan_for(width: u32, height: u32) -> PyramidPlan {
    PyramidPlanner::new(width, height, 256, 0, Layout::Xyz)
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

/// What one generation run produced.
struct Generated {
    row: Measurement,
    /// The archive, or the root of the tree.
    output: PathBuf,
    plan: PyramidPlan,
}

/// Run one canvas into one backend and measure it.
fn generate(storage: &str, profile: Profile, dir: &Path, width: u32, height: u32) -> Generated {
    let plan = plan_for(width, height);
    let source = bench::gradient(width, height);

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

    let (bytes, entries) = bench::occupancy(&output);
    let row = Measurement::new(
        "generate",
        storage,
        profile,
        width,
        height,
        1,
        result.duration,
        result.peak_memory_bytes,
        rss,
        result.tiles_produced,
    )
    .with_output(bytes, entries);

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

/// The same, spread over `threads` threads, each taking a contiguous slice.
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
    width: u32,
    height: u32,
    concurrency: usize,
    rss: u64,
    mut pass: Pass,
) -> Measurement {
    let bytes = pass.bytes;
    Measurement::new(
        scenario,
        storage,
        profile,
        width,
        height,
        concurrency,
        pass.elapsed,
        0,
        rss,
        pass.hits,
    )
    .with_latencies(&mut pass.latencies, bytes)
}

/// Every read scenario, against one already-generated pyramid.
///
/// "Cold" is a reader opened a moment ago whose in-process caches hold
/// nothing: the PMTiles leaf cache is empty and the directory reader has no
/// open handles. It is **not** a cold OS page cache. Dropping that needs root
/// on Linux and has no portable equivalent, and a benchmark that claimed a
/// cold page cache without dropping one would be reporting a warm number under
/// a cold label. The gap between `read_cold` and `read_warm` here is therefore
/// the in-process caches alone, which is the part libviprs controls.
fn read_scenarios(
    storage: &str,
    profile: Profile,
    width: u32,
    height: u32,
    output: &Path,
    plan: &PyramidPlan,
) -> Vec<Measurement> {
    let all = coordinates(plan);
    let samples = profile.read_samples().min(all.len().max(1));
    let sequential: Vec<TileCoord> = all.iter().copied().take(samples).collect();

    let mut rng = Splitmix::new(0x5EED_1234_ABCD_0001);
    let random: Vec<TileCoord> = (0..samples).map(|_| all[rng.below(all.len())]).collect();

    let mut rows = Vec::new();

    // Cold: one lookup on a reader that has never been used.
    let reset = reset_peak_rss();
    let reader = open_reader(storage, output, plan);
    let single = vec![sequential[0]];
    let cold = read_pass(reader.as_ref(), &single);
    assert_eq!(cold.hits, 1, "the first coordinate should be present");
    rows.push(read_row(
        "read_cold",
        storage,
        profile,
        width,
        height,
        1,
        phase_peak_rss(reset),
        cold,
    ));

    // Warm: the same coordinate again on the same reader.
    let warm = read_pass(reader.as_ref(), &single);
    rows.push(read_row(
        "read_warm",
        storage,
        profile,
        width,
        height,
        1,
        phase_peak_rss(reset),
        warm,
    ));

    let sequential_pass = read_pass(reader.as_ref(), &sequential);
    rows.push(read_row(
        "read_sequential",
        storage,
        profile,
        width,
        height,
        1,
        phase_peak_rss(reset),
        sequential_pass,
    ));

    let random_pass = read_pass(reader.as_ref(), &random);
    rows.push(read_row(
        "read_random",
        storage,
        profile,
        width,
        height,
        1,
        phase_peak_rss(reset),
        random_pass,
    ));

    // At least two, because a "concurrent" row measured on one thread is a
    // sequential row wearing the wrong label, and a container pinned to one
    // CPU is exactly where that would happen unnoticed.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 8);
    let concurrent = concurrent_pass(reader.as_ref(), &random, threads);
    rows.push(read_row(
        "read_concurrent",
        storage,
        profile,
        width,
        height,
        threads,
        phase_peak_rss(reset),
        concurrent,
    ));

    rows
}

/// One cell: generate into one backend, then read it back every way.
fn run_cell(storage: &str, profile: Profile, width: u32, height: u32) -> Vec<Measurement> {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let generated = generate(storage, profile, dir.path(), width, height);
    let mut rows = vec![generated.row];
    rows.extend(read_scenarios(
        storage,
        profile,
        width,
        height,
        &generated.output,
        &generated.plan,
    ));
    rows
}

// ---------------------------------------------------------------------------
// The child cell
// ---------------------------------------------------------------------------

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
    let (storage, canvas) = spec
        .split_once(':')
        .unwrap_or_else(|| panic!("a cell spec is <storage>:<width>x<height>, got {spec:?}"));
    let (width, height) = canvas
        .split_once('x')
        .unwrap_or_else(|| panic!("a canvas is <width>x<height>, got {canvas:?}"));
    let width: u32 = width.parse().expect("a canvas width");
    let height: u32 = height.parse().expect("a canvas height");

    for row in run_cell(storage, Profile::from_env(), width, height) {
        println!("{ROW_PREFIX}{}", bench::to_json(std::slice::from_ref(&row)));
    }
}

/// Run one cell in a fresh copy of this test binary and collect its rows.
fn spawn_cell(storage: &str, width: u32, height: u32) -> Vec<String> {
    let exe = std::env::current_exe().expect("the test binary knows where it is");
    let output = std::process::Command::new(exe)
        .args(["--exact", "benchmark_cell", "--ignored", "--nocapture"])
        .env(CELL_VAR, format!("{storage}:{width}x{height}"))
        .output()
        .expect("the child cell starts");
    assert!(
        output.status.success(),
        "the {storage} {width}x{height} cell failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let rows: Vec<String> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix(ROW_PREFIX))
        .map(|json| json.trim().to_string())
        .collect();
    assert!(
        !rows.is_empty(),
        "the {storage} {width}x{height} cell printed no rows:\n{stdout}"
    );
    rows
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
    let mut rows: Vec<String> = Vec::new();
    for (width, height) in profile.canvases() {
        for storage in [DIRECTORY, PMTILES] {
            rows.extend(spawn_cell(storage, *width, *height));
        }
    }

    // Each child printed its row as a one-element array; splice them into one.
    let bodies: Vec<String> = rows
        .iter()
        .map(|row| {
            row.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_string()
        })
        .collect();
    let document = format!("[\n{}\n]\n", bodies.join(",\n"));

    let path = bench::results_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("the results directory can be created");
    }
    std::fs::write(&path, &document).expect("the results file can be written");

    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&document).expect("the exported document is JSON");
    assert_eq!(parsed.len(), rows.len());

    println!("wrote {} rows to {}", parsed.len(), path.display());
    for row in &parsed {
        println!(
            "  {:<16} {:<10} {:>10.1} ms  rss {:>8.1} MB  {:>10.1} tiles/s  {:>12} bytes  {:>9} entries",
            row["scenario"].as_str().unwrap_or("?"),
            row["storage"].as_str().unwrap_or("?"),
            row["wall_time_ms"].as_f64().unwrap_or(0.0),
            row["peak_rss_mb"].as_f64().unwrap_or(0.0),
            row["tiles_per_second"].as_f64().unwrap_or(0.0),
            row["output_bytes"].as_u64().unwrap_or(0),
            row["filesystem_entries"].as_u64().unwrap_or(0),
        );
    }
}

// ---------------------------------------------------------------------------
// The guards that run on every CI job
// ---------------------------------------------------------------------------

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

    let tree = generate(DIRECTORY, profile, dir.path(), 512, 512);
    let archive = generate(PMTILES, profile, dir.path(), 512, 512);

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
        archive.row.filesystem_entries, 1,
        "a PMTiles pyramid is one file"
    );
    assert!(
        tree.row.filesystem_entries > planned,
        "a directory pyramid is at least one entry per tile plus its directories, got {}",
        tree.row.filesystem_entries
    );
    assert!(
        tree.row.output_bytes > 0 && archive.row.output_bytes > 0,
        "both backends should have written bytes"
    );
    assert!(
        tree.row.tiles_per_second > 0.0 && archive.row.tiles_per_second > 0.0,
        "a run that took no measurable time is not a measurement"
    );

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
    let generated = generate(PMTILES, profile, dir.path(), 512, 512);

    let rows = read_scenarios(
        PMTILES,
        profile,
        512,
        512,
        &generated.output,
        &generated.plan,
    );
    let scenarios: Vec<&str> = rows.iter().map(|r| r.scenario.as_str()).collect();
    assert_eq!(
        scenarios,
        vec![
            "read_cold",
            "read_warm",
            "read_sequential",
            "read_random",
            "read_concurrent"
        ]
    );
    for row in &rows {
        assert!(
            row.tiles_produced > 0,
            "{} read no tiles at all",
            row.scenario
        );
        assert!(
            row.bytes_fetched > 0,
            "{} returned no tile bytes",
            row.scenario
        );
        assert!(
            row.p99_latency_us >= row.p50_latency_us,
            "{}: p99 {} is below p50 {}",
            row.scenario,
            row.p99_latency_us,
            row.p50_latency_us
        );
    }
    assert!(
        rows.iter().any(|r| r.concurrency > 1),
        "the concurrent scenario should report its thread count"
    );
}

/// The exported document is the shape `scalability_results.json` is, parsed by
/// something other than the code that wrote it.
///
/// This is the contract with libviprs.org (issue #62). A field renamed here is
/// a chart that silently stops drawing there, and nothing else in either
/// repository would catch it.
#[test]
fn the_exported_json_carries_every_field_the_site_reads() {
    let rows = vec![
        Measurement::new(
            "generate",
            PMTILES,
            Profile::Ci,
            2048,
            2048,
            1,
            Duration::from_millis(1234),
            5 * 1024 * 1024,
            90 * 1024 * 1024,
            349,
        )
        .with_output(4_194_304, 1),
        Measurement::new(
            "read_random",
            DIRECTORY,
            Profile::Large,
            2048,
            2048,
            8,
            Duration::from_micros(9_876),
            0,
            0,
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

    let document = bench::to_json(&rows);
    let parsed: serde_json::Value =
        serde_json::from_str(&document).expect("the exported document is JSON");
    let array = parsed
        .as_array()
        .expect("the document is a top-level array");
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
        // The twelve the existing consumer reads have to be the types it reads
        // them as: numbers everywhere except `engine`.
        for field in bench::SCALABILITY_FIELDS {
            let value = &object[field];
            if field == "engine" {
                assert!(value.is_string(), "{field} should be a string, got {value}");
            } else {
                assert!(value.is_number(), "{field} should be a number, got {value}");
                assert!(
                    value.as_f64().expect("a number").is_finite(),
                    "{field} is not finite: {value}"
                );
            }
        }
    }

    // The derived columns are derived the way the existing producer derives
    // them, which is what makes a row here comparable with a row there.
    let generation = array[0].as_object().expect("an object");
    let rss_mb = generation["peak_rss_mb"].as_f64().expect("a number");
    let tps = generation["tiles_per_second"].as_f64().expect("a number");
    let per_mb = generation["tiles_per_second_per_mb"]
        .as_f64()
        .expect("a number");
    assert!(
        (per_mb - tps / rss_mb).abs() < 1e-9,
        "tiles_per_second_per_mb should be tiles_per_second over peak_rss_mb"
    );
    assert_eq!(generation["engine"], generation["storage"]);
    assert_eq!(generation["filesystem_entries"], 1);
}

/// A row measured on a platform with no `/proc` reports zero rather than a
/// stale peak.
#[test]
fn a_platform_without_a_peak_rss_reports_zero_not_a_stale_number() {
    assert_eq!(phase_peak_rss(false), 0);
    let row = Measurement::new(
        "generate",
        PMTILES,
        Profile::Ci,
        512,
        512,
        1,
        Duration::from_millis(10),
        0,
        phase_peak_rss(false),
        5,
    );
    assert_eq!(row.peak_rss_mb, 0.0);
    // And the two ratios that divide by it stay finite rather than becoming an
    // infinity no JSON parser accepts.
    assert_eq!(row.tiles_per_second_per_mb, 0.0);
    assert_eq!(row.resource_cost, 0.0);
}
