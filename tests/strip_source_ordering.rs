//! Regression test for issue #105.
//!
//! The `StripSource` contract documents that the engine requests strips
//! sequentially, in strictly increasing `y` order. The MapReduce MAP phase,
//! however, used to spawn concurrent scoped threads that called
//! `render_strip` in parallel (and therefore out of `y` order) for *every*
//! source, silently breaking cursor-based sources (a streamed TIFF decoder or
//! a network byte stream) that rely on the documented ordering.
//!
//! This test installs a `StripSource` that leaves `permits_concurrent_strips`
//! at its default (`false`) — i.e. an ordinary cursor-style source — and
//! asserts the engine honours the sequential, monotonic contract even under
//! `EngineKind::MapReduce` with concurrency enabled. It records two kinds of
//! violation:
//!
//! * two `render_strip` calls overlapping in time (concurrency), and
//! * a call whose `y_offset` is not strictly greater than the previous one.
//!
//! On the pre-fix engine the MAP phase renders the first batch's strips on
//! parallel threads, so both flags trip and the test fails. Once the engine
//! only parallelises sources that opt in via `permits_concurrent_strips`, a
//! default source is driven sequentially and the invariant holds.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::time::Duration;

use libviprs::streaming::StripSource;
use libviprs::{
    EngineBuilder, EngineError, EngineKind, Layout, PixelFormat, PyramidPlanner, Raster, SinkError,
    Tile, TileSink, generate_test_raster,
};

/// A cursor-style strip source that verifies the engine's access pattern.
///
/// Backed by an in-memory raster (so the pixels are real and the pyramid
/// pipeline runs to completion), it additionally asserts, at runtime, that no
/// two `render_strip` calls overlap and that `y_offset` never goes backwards —
/// exactly the guarantees the trait promises and a real streaming decoder
/// would depend on.
struct OrderTrackingSource {
    raster: Raster,
    /// Number of `render_strip` calls currently executing. A cursor source
    /// can tolerate exactly one; anything higher is a concurrency violation.
    in_flight: AtomicUsize,
    /// Highest `y_offset` seen so far (`-1` before the first call).
    last_y: AtomicI64,
    concurrent_violation: AtomicBool,
    order_violation: AtomicBool,
}

impl OrderTrackingSource {
    fn new(raster: Raster) -> Self {
        Self {
            raster,
            in_flight: AtomicUsize::new(0),
            last_y: AtomicI64::new(-1),
            concurrent_violation: AtomicBool::new(false),
            order_violation: AtomicBool::new(false),
        }
    }
}

impl StripSource for OrderTrackingSource {
    // NOTE: `permits_concurrent_strips` is intentionally left at its default
    // (`false`): this models a source that requires the documented sequential,
    // increasing-`y` access pattern.

    fn render_strip(&self, y_offset: u32, height: u32) -> Result<Raster, EngineError> {
        // Detect overlapping calls: bump the in-flight counter, hold it for a
        // moment so a genuinely concurrent sibling call is caught in the act,
        // then drop it back.
        let concurrent = self.in_flight.fetch_add(1, Ordering::SeqCst);
        if concurrent != 0 {
            self.concurrent_violation.store(true, Ordering::SeqCst);
        }

        // Detect out-of-order access: y must strictly increase across calls.
        let prev = self.last_y.swap(y_offset as i64, Ordering::SeqCst);
        if (y_offset as i64) <= prev {
            self.order_violation.store(true, Ordering::SeqCst);
        }

        // Widen the window so overlapping calls reliably observe each other.
        std::thread::sleep(Duration::from_millis(20));

        self.in_flight.fetch_sub(1, Ordering::SeqCst);

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
        PixelFormat::Rgb8
    }
}

#[derive(Default)]
struct CountingSink {
    count: AtomicUsize,
}

impl TileSink for CountingSink {
    fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

// The engine consumes the source by value, so the runtime flags are read back
// out through a shared `Arc` handle after the run completes. Dimensions +
// budget are chosen (via a probe against `compute_strip_height` /
// `compute_inflight_strips`) so the MAP phase forms a batch of >1 strip with
// concurrency enabled — i.e. the parallel path is actually taken.
#[test]
fn mapreduce_never_calls_render_strip_concurrently_or_out_of_order() {
    use std::sync::Arc;

    struct SharedSource(Arc<OrderTrackingSource>);
    impl StripSource for SharedSource {
        fn render_strip(&self, y: u32, height: u32) -> Result<Raster, EngineError> {
            self.0.render_strip(y, height)
        }
        fn width(&self) -> u32 {
            self.0.width()
        }
        fn height(&self) -> u32 {
            self.0.height()
        }
        fn format(&self) -> PixelFormat {
            self.0.format()
        }
    }

    let (w, h) = (1024u32, 8192u32);
    let raster = generate_test_raster(w, h).expect("raster");
    let inner = Arc::new(OrderTrackingSource::new(raster));

    let plan = PyramidPlanner::new(w, h, 256, 0, Layout::DeepZoom)
        .expect("planner")
        .plan();

    let sink = CountingSink::default();
    EngineBuilder::new(SharedSource(inner.clone()), plan, sink)
        .with_engine(EngineKind::MapReduce)
        .with_concurrency(4)
        .with_buffer_size(2)
        .with_memory_budget(30_000_000)
        .run()
        .expect("mapreduce run");

    assert!(
        !inner.concurrent_violation.load(Ordering::SeqCst),
        "render_strip was called concurrently from multiple threads, violating \
         the StripSource sequential-access contract (issue #105)"
    );
    assert!(
        !inner.order_violation.load(Ordering::SeqCst),
        "render_strip was called with non-increasing y_offset, violating the \
         StripSource increasing-y contract (issue #105)"
    );
}
