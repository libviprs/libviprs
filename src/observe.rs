use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::planner::TileCoord;

/// Events emitted during pyramid generation.
///
/// Each variant represents a distinct lifecycle moment in the tile-pyramid
/// engine. Observers receive these events in order via
/// [`EngineObserver::on_event`], enabling progress reporting, logging,
/// and post-hoc analysis without coupling the engine to any specific UI
/// or metrics backend.
///
/// # Example usage
///
/// - [progress_events_match_tile_count](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
///   asserts that the total number of `TileCompleted` events equals the
///   sum reported in `Finished`.
/// - [level_started_before_tile_completed](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
///   verifies the ordering invariant between `LevelStarted` and
///   `TileCompleted` events.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-trace-level)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    // -- Pipeline-level events (emitted by callers for full-pipeline observability) --
    /// The source file is about to be loaded (decoded, extracted, or rendered).
    ///
    /// Callers emit this before starting source acquisition so observers can
    /// track the full pipeline from input to output. The `source_description`
    /// is a free-form string identifying the input (e.g. a file path, "stdin",
    /// or "PDF page 3 at 300 DPI").
    SourceLoadStarted { source_description: String },

    /// The source raster is ready.
    ///
    /// Callers emit this after decoding / rendering completes but before
    /// planning or tiling begins.
    SourceLoaded {
        width: u32,
        height: u32,
        format: crate::pixel::PixelFormat,
        size_bytes: u64,
    },

    /// The pyramid plan has been created.
    ///
    /// Callers emit this after planning so observers know the scope of the
    /// tiling work ahead.
    PlanCreated {
        levels: u32,
        total_tiles: u64,
        canvas_width: u32,
        canvas_height: u32,
    },

    // -- Tiling-phase events (emitted by the engine) --
    /// A level is about to be processed.
    LevelStarted {
        level: u32,
        width: u32,
        height: u32,
        tile_count: u64,
    },
    /// A tile was produced and sent to the sink.
    TileCompleted { coord: TileCoord },
    /// A level finished processing.
    LevelCompleted { level: u32, tiles_produced: u64 },

    // -- Streaming-engine events --
    /// A horizontal strip was rendered / extracted from the source.
    ///
    /// Emitted by the streaming engine each time a strip is obtained from
    /// the [`StripSource`](crate::streaming::StripSource). Observers can
    /// use `strip_index` / `total_strips` for progress reporting.
    StripRendered { strip_index: u32, total_strips: u32 },

    // -- MapReduce-engine events --
    /// A batch of strips is about to be processed in parallel.
    ///
    /// Emitted by the MapReduce engine at the start of each batch.
    /// `strips_in_batch` may be less than the computed in-flight count
    /// for the final batch.
    BatchStarted {
        batch_index: u32,
        strips_in_batch: u32,
        total_batches: u32,
    },

    /// A batch of strips has finished processing (map + reduce).
    BatchCompleted {
        batch_index: u32,
        tiles_produced: u64,
    },

    // -- Completion events --
    /// The tiling phase is done.
    ///
    /// Emitted by the engine at the end of pyramid generation.
    Finished { total_tiles: u64, levels: u32 },

    /// The entire pipeline is complete, control is returning to the caller.
    ///
    /// Callers emit this as the last event after all post-processing
    /// (manifest writing, cleanup, etc.) is finished. The engine never
    /// emits this — it is purely a caller-side bookend to `SourceLoadStarted`.
    PipelineComplete,
}

/// Trait for receiving engine progress events.
///
/// Implementations must be `Send + Sync` since events may be emitted from
/// worker threads. The engine holds a shared reference to the observer and
/// calls [`on_event`](Self::on_event) synchronously on the thread that
/// produced the event.
///
/// # Example usage
///
/// - [progress_events_match_tile_count](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
///   uses a [`CollectingObserver`] (which implements this trait) to capture
///   and inspect all events emitted during a test pyramid build.
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   implements this trait to print live progress to stderr.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-trace-level)
pub trait EngineObserver: Send + Sync {
    fn on_event(&self, event: EngineEvent);
}

/// A no-op observer that discards all events.
///
/// This is the default observer used when the caller does not need progress
/// feedback. Because `on_event` is an empty function, the compiler can
/// inline and eliminate it entirely, adding zero overhead to the hot
/// tile-generation loop.
pub struct NoopObserver;

impl EngineObserver for NoopObserver {
    fn on_event(&self, _event: EngineEvent) {}
}

/// An observer that collects all events in order.
///
/// Stores every [`EngineEvent`] it receives in a `Mutex<Vec<_>>`, making
/// the full event history available for post-hoc assertions. This is the
/// primary observer used in integration tests and is also useful for
/// debugging production pipelines (attach it alongside a logging observer).
///
/// # Example usage
///
/// - [progress_events_match_tile_count](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
///   and related tests create a `CollectingObserver`, run a pyramid build,
///   then inspect the captured events.
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   wraps a `CollectingObserver` for its `--verbose` mode.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-trace-level)
#[derive(Debug)]
pub struct CollectingObserver {
    events: std::sync::Mutex<Vec<EngineEvent>>,
}

impl CollectingObserver {
    /// Create a new, empty `CollectingObserver`.
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all collected events so far, cloned from the
    /// internal mutex-protected buffer.
    pub fn events(&self) -> Vec<EngineEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Return the number of events collected so far.
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

impl Default for CollectingObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineObserver for CollectingObserver {
    fn on_event(&self, event: EngineEvent) {
        self.events.lock().unwrap().push(event);
    }
}

/// Tracks peak memory usage during pyramid generation.
///
/// Uses a simple atomic counter that the engine updates at key allocation points.
/// This is not a full allocator -- it tracks logical memory (raster buffers held),
/// not every malloc.
///
/// # Example usage
///
/// - [peak_memory_bounded_for_medium_image](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
///   attaches a `MemoryTracker` to an engine run and asserts the peak stays
///   below an expected ceiling.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-trace-level)
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    current: Arc<AtomicU64>,
    peak: Arc<AtomicU64>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            current: Arc::new(AtomicU64::new(0)),
            peak: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record an allocation of `bytes`.
    pub fn alloc(&self, bytes: u64) {
        let new = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(new, Ordering::Relaxed);
    }

    /// Record a deallocation of `bytes`.
    pub fn dealloc(&self, bytes: u64) {
        self.current.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current tracked memory in bytes.
    pub fn current_bytes(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// Peak tracked memory in bytes.
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Reset the tracker.
    pub fn reset(&self) {
        self.current.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * Tests that CollectingObserver records every event it receives.
     * Works by sending three distinct event types and checking event_count().
     * Input: LevelStarted, TileCompleted, LevelCompleted → Output: event_count() == 3.
     */
    #[test]
    fn collecting_observer_captures_events() {
        let obs = CollectingObserver::new();
        obs.on_event(EngineEvent::LevelStarted {
            level: 0,
            width: 1,
            height: 1,
            tile_count: 1,
        });
        obs.on_event(EngineEvent::TileCompleted {
            coord: TileCoord::new(0, 0, 0),
        });
        obs.on_event(EngineEvent::LevelCompleted {
            level: 0,
            tiles_produced: 1,
        });
        assert_eq!(obs.event_count(), 3);
    }

    /**
     * Tests that CollectingObserver preserves insertion order of events.
     * Works by emitting LevelStarted then Finished and pattern-matching
     * the returned vec by index to confirm ordering.
     * Input: LevelStarted(level=5), Finished(total=10) → Output: events()[0] is LevelStarted, events()[1] is Finished.
     */
    #[test]
    fn collecting_observer_preserves_order() {
        let obs = CollectingObserver::new();
        obs.on_event(EngineEvent::LevelStarted {
            level: 5,
            width: 100,
            height: 100,
            tile_count: 4,
        });
        obs.on_event(EngineEvent::Finished {
            total_tiles: 10,
            levels: 6,
        });
        let events = obs.events();
        assert!(matches!(
            events[0],
            EngineEvent::LevelStarted { level: 5, .. }
        ));
        assert!(matches!(
            events[1],
            EngineEvent::Finished {
                total_tiles: 10,
                ..
            }
        ));
    }

    /**
     * Tests that NoopObserver accepts events without panicking.
     * Works by calling on_event; the test passing (no crash) confirms
     * the no-op implementation is valid.
     */
    #[test]
    fn noop_observer_compiles() {
        let obs = NoopObserver;
        obs.on_event(EngineEvent::Finished {
            total_tiles: 0,
            levels: 0,
        });
    }

    /**
     * Tests that NoopObserver satisfies Send + Sync bounds.
     * Works by calling a generic function constrained with Send + Sync;
     * a compile failure would indicate the type cannot be shared across threads.
     */
    #[test]
    fn noop_observer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoopObserver>();
    }

    /**
     * Tests that CollectingObserver satisfies Send + Sync bounds.
     * Works via a compile-time check; the Mutex interior makes this
     * non-trivial to guarantee.
     */
    #[test]
    fn collecting_observer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CollectingObserver>();
    }

    /**
     * Tests alloc/dealloc tracking and peak-memory watermark.
     * Works by performing a sequence of alloc/dealloc calls and asserting
     * current and peak at each step.
     * Input: alloc(100), alloc(200), dealloc(150), dealloc(150) → Output: current=0, peak=300.
     */
    #[test]
    fn memory_tracker_basic() {
        let t = MemoryTracker::new();
        assert_eq!(t.current_bytes(), 0);
        assert_eq!(t.peak_bytes(), 0);

        t.alloc(100);
        assert_eq!(t.current_bytes(), 100);
        assert_eq!(t.peak_bytes(), 100);

        t.alloc(200);
        assert_eq!(t.current_bytes(), 300);
        assert_eq!(t.peak_bytes(), 300);

        t.dealloc(150);
        assert_eq!(t.current_bytes(), 150);
        assert_eq!(t.peak_bytes(), 300); // Peak unchanged

        t.dealloc(150);
        assert_eq!(t.current_bytes(), 0);
        assert_eq!(t.peak_bytes(), 300);
    }

    /**
     * Tests that reset() zeroes both current and peak counters.
     * Works by allocating 500 bytes, resetting, then asserting both return 0.
     * Input: alloc(500), reset() → Output: current=0, peak=0.
     */
    #[test]
    fn memory_tracker_reset() {
        let t = MemoryTracker::new();
        t.alloc(500);
        t.reset();
        assert_eq!(t.current_bytes(), 0);
        assert_eq!(t.peak_bytes(), 0);
    }

    /**
     * Tests thread safety of MemoryTracker under concurrent access.
     * Works by spawning 8 threads that each alloc/dealloc equal amounts,
     * then verifying current returns to 0 and peak is non-zero.
     * Input: 8 threads × 100 alloc(10)/dealloc(10) → Output: current=0, peak>0.
     */
    #[test]
    fn memory_tracker_concurrent() {
        use std::thread;

        let t = MemoryTracker::new();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let t = t.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        t.alloc(10);
                    }
                    for _ in 0..100 {
                        t.dealloc(10);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(t.current_bytes(), 0);
        assert!(t.peak_bytes() > 0);
    }
}
