//! Cooperative cancellation: a run must be stoppable via a shared cancel
//! token and surface [`EngineError::Cancelled`], and an in-flight retry
//! backoff must be interruptible instead of sleeping out its full schedule.
//!
//! These exercise the public `EngineBuilder::with_cancel` / `CancelToken`
//! surface end-to-end (issue #133). Before the fix `EngineError::Cancelled`
//! was declared but never constructed: there was no cancel token plumbed into
//! the level / strip loops or the retry backoff, so a long run could not be
//! stopped cooperatively.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use libviprs::sink::SinkError;
use libviprs::{
    CancelToken, EngineBuilder, EngineError, EngineEvent, EngineKind, EngineObserver,
    FailurePolicy, Layout, MemorySink, PixelFormat, PyramidPlanner, Raster, RetryPolicy, Tile,
    TileSink,
};

fn gradient(w: u32, h: u32) -> Raster {
    let bpp = 3usize;
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

/// A pre-cancelled token must abort the monolithic run immediately with
/// [`EngineError::Cancelled`], never running to completion.
#[test]
fn precancelled_monolithic_run_returns_cancelled() {
    let src = gradient(512, 512);
    let plan = PyramidPlanner::new(512, 512, 128, 0, Layout::DeepZoom)
        .unwrap()
        .plan();

    let cancel = CancelToken::new();
    cancel.cancel();

    let result = EngineBuilder::new(&src, plan, MemorySink::new())
        .with_engine(EngineKind::Monolithic)
        .with_cancel(cancel)
        .run();

    assert!(
        matches!(result, Err(EngineError::Cancelled)),
        "a pre-cancelled run must return EngineError::Cancelled, got {result:?}"
    );
}

/// An observer that cancels the run after observing a fixed number of
/// completed tiles, modelling a user hitting Ctrl-C part-way through.
struct CancelAfter {
    token: CancelToken,
    seen: AtomicU64,
    after: u64,
}

impl EngineObserver for CancelAfter {
    fn on_event(&self, event: EngineEvent) {
        if let EngineEvent::TileCompleted { .. } = event {
            let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
            if n >= self.after {
                self.token.cancel();
            }
        }
    }
}

/// Cancelling mid-run must stop the engine cooperatively: the run returns
/// `Cancelled` and the sink holds strictly fewer tiles than the full plan.
#[test]
fn midrun_cancel_stops_before_completion() {
    let src = gradient(512, 512);
    let plan = PyramidPlanner::new(512, 512, 64, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let total = plan.total_tile_count();
    assert!(
        total > 10,
        "need a plan with enough tiles to cancel mid-way"
    );

    let cancel = CancelToken::new();
    let obs = CancelAfter {
        token: cancel.clone(),
        seen: AtomicU64::new(0),
        after: 3,
    };

    let (result, sink) = EngineBuilder::new(&src, plan, MemorySink::new())
        .with_engine(EngineKind::Monolithic)
        .with_cancel(cancel)
        .with_observer(obs)
        .run_collect();

    assert!(
        matches!(result, Err(EngineError::Cancelled)),
        "a mid-run cancel must return EngineError::Cancelled, got {result:?}"
    );
    assert!(
        (sink.tile_count() as u64) < total,
        "cancellation must stop the run early: wrote {} of {} tiles",
        sink.tile_count(),
        total
    );
}

/// A sink whose `write_tile` always fails with a transient error, forcing the
/// retry loop to exercise its backoff.
struct AlwaysFailSink {
    calls: AtomicU64,
}

impl TileSink for AlwaysFailSink {
    fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(SinkError::Other("always fails".into()))
    }
}

/// An in-flight retry backoff must be interruptible: with a long backoff
/// schedule, cancelling from another thread has to return promptly with
/// `EngineError::Cancelled` rather than sleeping out the full schedule.
#[test]
fn retry_backoff_is_interruptible() {
    let src = gradient(64, 64);
    let plan = PyramidPlanner::new(64, 64, 64, 0, Layout::DeepZoom)
        .unwrap()
        .plan();

    let cancel = CancelToken::new();
    let sink = AlwaysFailSink {
        calls: AtomicU64::new(0),
    };

    // A deliberately long backoff: if it is not interruptible the run would
    // block for many seconds before the retry budget is exhausted.
    let policy = RetryPolicy::new(10, Duration::from_secs(5))
        .with_multiplier(1.0)
        .with_max_backoff(Duration::from_secs(5))
        .with_jitter(false);

    // Cancel shortly after the run starts sleeping in its first backoff.
    let canceller = cancel.clone();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        canceller.cancel();
    });

    let start = Instant::now();
    let result = EngineBuilder::new(&src, plan, sink)
        .with_engine(EngineKind::Monolithic)
        .with_failure_policy(FailurePolicy::RetryThenFail(policy))
        .with_cancel(cancel)
        .run();
    let elapsed = start.elapsed();

    handle.join().unwrap();

    assert!(
        matches!(result, Err(EngineError::Cancelled)),
        "an interrupted retry backoff must surface Cancelled, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "cancellation must cut the 5s backoff short; took {elapsed:?}"
    );
}
