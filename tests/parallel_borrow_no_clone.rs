//! Regression test for issue #104.
//!
//! The parallel monolithic path used to deep-copy the whole level raster
//! (`Arc::new(raster.clone())`) before handing it to scoped worker threads.
//! Because the workers run under `std::thread::scope` they can borrow the
//! raster directly, so the clone is pure waste: it holds an extra full copy of
//! the largest (top) level alive for the entire emission, pushing the true peak
//! toward ~3x the source while `MemoryTracker` never charges it.
//!
//! This test installs a process-wide counting allocator and asserts that the
//! peak *live* heap growth during a parallel run stays below the size of a
//! second full-resolution raster. On the buggy code the extra clone blows past
//! that bound; once the workers borrow instead of cloning it comfortably fits.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use libviprs::{
    EngineBuilder, EngineKind, Layout as PyLayout, PyramidPlanner, SinkError, Tile, TileSink,
    generate_test_raster,
};

// --- counting global allocator ----------------------------------------------

struct CountingAlloc;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                let live = LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size
                    - layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

// --- discarding sink ---------------------------------------------------------

/// A sink that immediately drops every tile so accumulated output does not
/// pollute the heap measurement — only engine-internal buffers remain live.
#[derive(Default)]
struct NullSink {
    count: AtomicUsize,
}

impl TileSink for NullSink {
    fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

// --- the test ----------------------------------------------------------------

#[test]
fn parallel_monolithic_does_not_clone_the_level_raster() {
    // A large single-source raster so the top-level clone dominates any
    // per-tile / channel buffering noise. 1536x1536 RGB = ~7.08 MiB.
    let dim = 1536u32;
    let src = generate_test_raster(dim, dim).expect("raster");
    let source_bytes = (dim as usize) * (dim as usize) * 3;

    let plan = PyramidPlanner::new(dim, dim, 256, 0, PyLayout::DeepZoom)
        .expect("planner")
        .plan();

    // Warm up any lazy one-time allocations (thread pools, format tables, …)
    // on a throwaway single-threaded run so they are not attributed to the
    // measured window.
    {
        let sink = NullSink::default();
        EngineBuilder::new(&src, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_concurrency(0)
            .with_buffer_size(4)
            .run()
            .expect("warmup run");
    }

    // Measure the parallel run: snapshot the live baseline (which already
    // includes `src`), reset the peak to it, run, then read the high-water
    // mark reached during emission.
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);

    let sink = NullSink::default();
    let result = EngineBuilder::new(&src, plan, sink)
        .with_engine(EngineKind::Monolithic)
        .with_concurrency(4)
        .with_buffer_size(4)
        .run()
        .expect("parallel run");
    assert!(result.tiles_produced > 0);

    let peak = PEAK.load(Ordering::SeqCst);
    let growth = peak.saturating_sub(baseline);

    // The engine legitimately holds ONE working copy of the top level
    // (`current = source.clone()`), so ~1x source of growth is expected. The
    // buggy path holds a SECOND full copy (the discarded `Arc::new(clone)`),
    // pushing growth past 2x. Assert we stay under 2x source; the deleted
    // clone is the only thing that could breach it.
    assert!(
        growth < source_bytes * 2,
        "peak live-heap growth {growth} >= 2x source {source_bytes}: the level \
         raster is being cloned for the workers instead of borrowed (issue #104)"
    );
}
