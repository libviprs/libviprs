//! Feature tests for issue #67: the `WorkExecutor` plug-in seam and the
//! `EngineKind::MapReduceHotCache` engine.
//!
//! Written test-first: these tests compile against the intended public
//! surface (`WorkExecutor`, `StripWorkUnit`, `WorkContext`,
//! `LocalWorkExecutor`, `EngineBuilder::with_executor`,
//! `EngineKind::MapReduceHotCache`) and fail until that surface lands.
//!
//! Contracts pinned here:
//!
//! * `LocalWorkExecutor` (and the implicit default) is byte-identical to the
//!   pre-seam MapReduce engine.
//! * A custom executor receives strip work units in canonical order (strictly
//!   increasing `canvas_y`, covering the whole canvas, all at the top level)
//!   and its output feeds the unchanged REDUCE phase byte-identically.
//! * An executor failure propagates as the `EngineError` it returned.
//! * `MapReduceHotCache` is byte-identical to the Streaming and MapReduce
//!   engines, flushes the user sink in a single pass at the end (canonical
//!   `(level, row, col)` write order, `finish` exactly once, after every
//!   write), and stays deterministic across tile-concurrency settings.
//! * `MapReduceHotCache` keeps MapReduce's pre-flight `BudgetExceeded`
//!   rejection.

use std::sync::Arc;
use std::sync::Mutex;

use libviprs::streaming_mapreduce::{LocalWorkExecutor, StripWorkUnit, WorkContext, WorkExecutor};
use libviprs::{
    EngineBuilder, EngineError, EngineKind, Layout, MemorySink, PixelFormat, PyramidPlanner,
    Raster, SinkError, Tile, TileCoord, TileSink,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic non-uniform raster (same pattern as the in-crate parity
/// tests) so byte-level comparisons are meaningful.
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

/// Collect a run's tiles sorted by canonical coordinate for byte comparison.
fn sorted_tiles(sink: &MemorySink) -> Vec<(TileCoord, Vec<u8>)> {
    let mut tiles: Vec<(TileCoord, Vec<u8>)> = sink
        .tiles()
        .into_iter()
        .map(|t| (t.coord, t.data))
        .collect();
    tiles.sort_by_key(|(c, _)| (c.level, c.row, c.col));
    tiles
}

fn run_engine(
    src: &Raster,
    plan: &libviprs::PyramidPlan,
    kind: EngineKind,
    budget: u64,
) -> Vec<(TileCoord, Vec<u8>)> {
    let sink = MemorySink::new();
    let (_result, sink) = EngineBuilder::new(src, plan.clone(), sink)
        .with_engine(kind)
        .with_memory_budget(budget)
        .run_collect()
        .expect("engine run should succeed");
    sorted_tiles(&sink)
}

// ---------------------------------------------------------------------------
// WorkExecutor seam
// ---------------------------------------------------------------------------

/// Explicitly installing `LocalWorkExecutor` must be byte-identical to the
/// default MapReduce run (no executor installed).
#[test]
fn local_executor_byte_identical_to_default_mapreduce() {
    let src = gradient_raster(1024, 1024);
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let budget = 6_000_000u64;

    let reference = run_engine(&src, &plan, EngineKind::MapReduce, budget);

    let sink = MemorySink::new();
    let (_result, sink) = EngineBuilder::new(&src, plan.clone(), sink)
        .with_engine(EngineKind::MapReduce)
        .with_memory_budget(budget)
        .with_executor(Arc::new(LocalWorkExecutor))
        .run_collect()
        .expect("run with explicit LocalWorkExecutor should succeed");
    let via_seam = sorted_tiles(&sink);

    assert_eq!(reference.len(), via_seam.len());
    for ((rc, rd), (sc, sd)) in reference.iter().zip(via_seam.iter()) {
        assert_eq!(rc, sc);
        assert_eq!(rd, sd, "tile bytes diverged at {sc:?}");
    }
}

/// A recording executor that delegates to the local implementation.
struct RecordingExecutor {
    seen: Mutex<Vec<StripWorkUnit>>,
}

impl WorkExecutor for RecordingExecutor {
    fn execute(&self, spec: StripWorkUnit, ctx: WorkContext<'_>) -> Result<Raster, EngineError> {
        self.seen.lock().unwrap().push(spec);
        LocalWorkExecutor.execute(spec, ctx)
    }
}

/// A custom executor sees every strip work unit, in canonical dispatch order
/// (sequential MAP path), covering the whole canvas at the top level, and the
/// pyramid it feeds into REDUCE is byte-identical to the built-in path.
#[test]
fn custom_executor_receives_canonical_specs_and_preserves_output() {
    let src = gradient_raster(1024, 1024);
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    // Budget sized so the canvas spans several strips (same choice as the
    // in-crate parity test), so the executor is dispatched more than once.
    let budget = 6_000_000u64;
    let top_level = (plan.levels.len() - 1) as u32;

    let reference = run_engine(&src, &plan, EngineKind::MapReduce, budget);

    let executor = Arc::new(RecordingExecutor {
        seen: Mutex::new(Vec::new()),
    });
    let sink = MemorySink::new();
    let (_result, sink) = EngineBuilder::new(&src, plan.clone(), sink)
        .with_engine(EngineKind::MapReduce)
        .with_memory_budget(budget)
        .with_executor(executor.clone())
        .run_collect()
        .expect("run with recording executor should succeed");

    let seen = executor.seen.lock().unwrap().clone();
    assert!(
        seen.len() >= 2,
        "test needs multiple strips to exercise dispatch ordering, got {}",
        seen.len()
    );

    // Canonical order: strictly increasing canvas_y, top level, gap-free
    // coverage of the full canvas height.
    let mut expected_y = 0u32;
    for spec in &seen {
        assert_eq!(
            spec.canvas_y, expected_y,
            "strip dispatch must be canonical and gap-free"
        );
        assert_eq!(
            spec.level, top_level,
            "MAP work is dispatched at the top level"
        );
        assert!(spec.height > 0);
        expected_y += spec.height;
    }
    assert_eq!(
        expected_y, plan.canvas_height,
        "dispatched strips must cover the whole canvas"
    );

    // And the output is unchanged.
    let via_seam = sorted_tiles(&sink);
    assert_eq!(reference.len(), via_seam.len());
    for ((rc, rd), (sc, sd)) in reference.iter().zip(via_seam.iter()) {
        assert_eq!(rc, sc);
        assert_eq!(rd, sd, "tile bytes diverged at {sc:?}");
    }
}

/// An executor that always fails.
struct FailingExecutor;

impl WorkExecutor for FailingExecutor {
    fn execute(&self, _spec: StripWorkUnit, _ctx: WorkContext<'_>) -> Result<Raster, EngineError> {
        Err(EngineError::WorkerPanic)
    }
}

/// An executor failure must abort the run with the error it returned.
#[test]
fn failing_executor_error_propagates() {
    let src = gradient_raster(512, 512);
    let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let err = EngineBuilder::new(&src, plan, MemorySink::new())
        .with_engine(EngineKind::MapReduce)
        .with_memory_budget(4_000_000)
        .with_executor(Arc::new(FailingExecutor))
        .run()
        .unwrap_err();
    assert!(
        matches!(err, EngineError::WorkerPanic),
        "expected the executor's WorkerPanic to propagate, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// MapReduceHotCache engine
// ---------------------------------------------------------------------------

/// The hot-cache engine is byte-identical to Streaming and MapReduce for the
/// same inputs, and its reported peak accounts for the cached tiles.
#[test]
fn hot_cache_byte_parity_with_streaming_and_mapreduce() {
    let src = gradient_raster(1024, 1024);
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let budget = 6_000_000u64;

    let streaming = run_engine(&src, &plan, EngineKind::Streaming, budget);
    let mapreduce = run_engine(&src, &plan, EngineKind::MapReduce, budget);

    let sink = MemorySink::new();
    let (result, sink) = EngineBuilder::new(&src, plan.clone(), sink)
        .with_engine(EngineKind::MapReduceHotCache)
        .with_memory_budget(budget)
        .run_collect()
        .expect("hot-cache run should succeed");
    let hot = sorted_tiles(&sink);

    assert_eq!(result.tiles_produced, plan.total_tile_count());
    assert_eq!(streaming.len(), hot.len());
    assert_eq!(mapreduce.len(), hot.len());
    for (((sc, sd), (mc, md)), (hc, hd)) in streaming.iter().zip(mapreduce.iter()).zip(hot.iter()) {
        assert_eq!(sc, hc);
        assert_eq!(mc, hc);
        assert_eq!(sd, hd, "hot-cache bytes diverged from streaming at {hc:?}");
        assert_eq!(md, hd, "hot-cache bytes diverged from mapreduce at {hc:?}");
    }

    // The cached pyramid is held in RAM until the flush; the reported peak
    // must account for (at least) those tile bytes.
    let cached_bytes: u64 = hot.iter().map(|(_, d)| d.len() as u64).sum();
    assert!(
        result.peak_memory_bytes >= cached_bytes,
        "peak {} must charge the {} bytes held by the hot cache",
        result.peak_memory_bytes,
        cached_bytes
    );
}

/// Sink that records the order of every `write_tile` / `finish` call.
#[derive(Debug, Default)]
struct CallOrderSink {
    calls: Mutex<Vec<CallRecord>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CallRecord {
    Write(TileCoord),
    Finish,
}

impl TileSink for CallOrderSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.calls
            .lock()
            .unwrap()
            .push(CallRecord::Write(tile.coord));
        Ok(())
    }
    fn finish(&self) -> Result<(), SinkError> {
        self.calls.lock().unwrap().push(CallRecord::Finish);
        Ok(())
    }
}

/// The hot-cache engine drains the user sink in one pass at the end: every
/// write arrives in canonical `(level, row, col)` order, `finish` is called
/// exactly once and only after the last write, and the write order is
/// identical across tile-concurrency settings (the determinism plain
/// MapReduce cannot promise at `tile_concurrency > 0`).
#[test]
fn hot_cache_single_pass_canonical_flush_and_cross_concurrency_determinism() {
    let src = gradient_raster(1024, 1024);
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let budget = 6_000_000u64;

    let run = |tile_concurrency: usize| {
        let sink = CallOrderSink::default();
        let (_result, sink) = EngineBuilder::new(&src, plan.clone(), sink)
            .with_engine(EngineKind::MapReduceHotCache)
            .with_memory_budget(budget)
            .with_concurrency(tile_concurrency)
            .run_collect()
            .expect("hot-cache run should succeed");
        sink.calls.lock().unwrap().clone()
    };

    let sequential = run(0);
    let parallel = run(4);

    for calls in [&sequential, &parallel] {
        // finish exactly once, as the final call.
        let finishes = calls
            .iter()
            .filter(|c| matches!(c, CallRecord::Finish))
            .count();
        assert_eq!(finishes, 1, "finish must be called exactly once");
        assert_eq!(
            calls.last(),
            Some(&CallRecord::Finish),
            "finish must come after every write"
        );

        // Writes arrive in canonical order.
        let coords: Vec<TileCoord> = calls
            .iter()
            .filter_map(|c| match c {
                CallRecord::Write(coord) => Some(*coord),
                CallRecord::Finish => None,
            })
            .collect();
        assert_eq!(coords.len() as u64, plan.total_tile_count());
        let mut expected = coords.clone();
        expected.sort_by_key(|c| (c.level, c.row, c.col));
        assert_eq!(
            coords, expected,
            "hot-cache flush must write in canonical (level, row, col) order"
        );
    }

    assert_eq!(
        sequential, parallel,
        "hot-cache write order must be identical across tile-concurrency settings"
    );
}

/// The hot-cache engine keeps MapReduce's pre-flight budget rejection: a
/// budget below the minimum aligned strip fails up front with
/// `BudgetExceeded` instead of silently over-committing.
#[test]
fn hot_cache_rejects_budget_below_minimum_strip() {
    let src = gradient_raster(512, 512);
    let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let err = EngineBuilder::new(&src, plan, MemorySink::new())
        .with_engine(EngineKind::MapReduceHotCache)
        .with_memory_budget(100_000)
        .run()
        .unwrap_err();
    assert!(
        matches!(err, EngineError::BudgetExceeded { .. }),
        "expected BudgetExceeded, got {err:?}"
    );
}

/// The hot-cache engine honours the executor seam too: the same recording
/// executor sees the canonical dispatch and output parity holds.
#[test]
fn hot_cache_uses_executor_seam() {
    let src = gradient_raster(1024, 1024);
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let budget = 6_000_000u64;

    let reference = run_engine(&src, &plan, EngineKind::MapReduce, budget);

    let executor = Arc::new(RecordingExecutor {
        seen: Mutex::new(Vec::new()),
    });
    let sink = MemorySink::new();
    let (_result, sink) = EngineBuilder::new(&src, plan.clone(), sink)
        .with_engine(EngineKind::MapReduceHotCache)
        .with_memory_budget(budget)
        .with_executor(executor.clone())
        .run_collect()
        .expect("hot-cache run with executor should succeed");

    assert!(
        !executor.seen.lock().unwrap().is_empty(),
        "hot-cache MAP phase must dispatch through the installed executor"
    );

    let hot = sorted_tiles(&sink);
    assert_eq!(reference.len(), hot.len());
    for ((rc, rd), (hc, hd)) in reference.iter().zip(hot.iter()) {
        assert_eq!(rc, hc);
        assert_eq!(rd, hd, "tile bytes diverged at {hc:?}");
    }
}
