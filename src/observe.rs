use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::planner::TileCoord;

/// Events emitted during pyramid generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
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
    /// The entire pyramid is done.
    Finished { total_tiles: u64, levels: u32 },
}

/// Trait for receiving engine progress events.
///
/// Implementations must be `Send + Sync` since events may be emitted from
/// worker threads.
pub trait EngineObserver: Send + Sync {
    fn on_event(&self, event: EngineEvent);
}

/// A no-op observer that discards all events. Used when no observer is configured.
pub struct NoopObserver;

impl EngineObserver for NoopObserver {
    fn on_event(&self, _event: EngineEvent) {}
}

/// An observer that collects all events in order. Useful for testing.
#[derive(Debug)]
pub struct CollectingObserver {
    events: std::sync::Mutex<Vec<EngineEvent>>,
}

impl CollectingObserver {
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<EngineEvent> {
        self.events.lock().unwrap().clone()
    }

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
/// This is not a full allocator — it tracks logical memory (raster buffers held),
/// not every malloc.
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

    #[test]
    fn noop_observer_compiles() {
        let obs = NoopObserver;
        obs.on_event(EngineEvent::Finished {
            total_tiles: 0,
            levels: 0,
        });
    }

    #[test]
    fn noop_observer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoopObserver>();
    }

    #[test]
    fn collecting_observer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CollectingObserver>();
    }

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

    #[test]
    fn memory_tracker_reset() {
        let t = MemoryTracker::new();
        t.alloc(500);
        t.reset();
        assert_eq!(t.current_bytes(), 0);
        assert_eq!(t.peak_bytes(), 0);
    }

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
