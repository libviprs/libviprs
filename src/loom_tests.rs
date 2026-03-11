/// Loom-based concurrency model checking for the engine's high-risk primitives.
///
/// These tests exhaustively explore thread interleavings for:
/// 1. Bounded tile queue (producer/consumer, no lost tiles)
/// 2. Level barrier (all tiles at level N complete before N+1)
/// 3. Backpressure (producer blocks when queue is full, unblocks on drain)
///
/// Run with: RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
///
/// These tests are gated behind `cfg(loom)` because Loom replaces std primitives
/// and is incompatible with normal test runs.
#[cfg(loom)]
mod tests {
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::sync::{Arc, Mutex, Condvar};
    use loom::thread;

    /// A bounded queue simulating the engine's tile channel.
    /// Uses Mutex + Condvar (same pattern as sync_channel internals).
    struct BoundedQueue<T> {
        inner: Mutex<BoundedQueueInner<T>>,
        not_full: Condvar,
        not_empty: Condvar,
    }

    struct BoundedQueueInner<T> {
        buf: Vec<T>,
        capacity: usize,
        closed: bool,
    }

    impl<T> BoundedQueue<T> {
        fn new(capacity: usize) -> Self {
            Self {
                inner: Mutex::new(BoundedQueueInner {
                    buf: Vec::new(),
                    capacity,
                    closed: false,
                }),
                not_full: Condvar::new(),
                not_empty: Condvar::new(),
            }
        }

        /// Push a value, blocking if at capacity. Returns false if closed.
        fn push(&self, val: T) -> bool {
            let mut inner = self.inner.lock().unwrap();
            while inner.buf.len() >= inner.capacity && !inner.closed {
                inner = self.not_full.wait(inner).unwrap();
            }
            if inner.closed {
                return false;
            }
            inner.buf.push(val);
            self.not_empty.notify_one();
            true
        }

        /// Pop a value, blocking if empty. Returns None when closed and drained.
        fn pop(&self) -> Option<T> {
            let mut inner = self.inner.lock().unwrap();
            loop {
                if let Some(val) = inner.buf.pop() {
                    self.not_full.notify_one();
                    return Some(val);
                }
                if inner.closed {
                    return None;
                }
                inner = self.not_empty.wait(inner).unwrap();
            }
        }

        fn close(&self) {
            let mut inner = self.inner.lock().unwrap();
            inner.closed = true;
            self.not_full.notify_all();
            self.not_empty.notify_all();
        }
    }

    // ---- Test 1: Tile queue - no lost tiles under concurrent push/pop ----

    #[test]
    fn loom_tile_queue_no_lost_items() {
        loom::model(|| {
            let queue = Arc::new(BoundedQueue::new(2));
            let received = Arc::new(AtomicUsize::new(0));

            let q1 = Arc::clone(&queue);
            let t1 = thread::spawn(move || {
                q1.push(1);
                q1.push(2);
            });

            let q2 = Arc::clone(&queue);
            let t2 = thread::spawn(move || {
                q2.push(3);
            });

            let q3 = Arc::clone(&queue);
            let r = Arc::clone(&received);
            let t3 = thread::spawn(move || {
                let mut count = 0;
                // We know exactly 3 items will be pushed, then queue closed
                while let Some(_val) = q3.pop() {
                    count += 1;
                    r.fetch_add(1, Ordering::Relaxed);
                }
                count
            });

            t1.join().unwrap();
            t2.join().unwrap();
            queue.close();

            let popped = t3.join().unwrap();
            assert_eq!(popped + received.load(Ordering::Relaxed) - popped, 3);
            assert_eq!(received.load(Ordering::Relaxed), 3);
        });
    }

    // ---- Test 2: Level barrier - all producers finish before consumer proceeds ----

    #[test]
    fn loom_level_barrier() {
        loom::model(|| {
            let completed = Arc::new(AtomicUsize::new(0));
            let total_workers = 2;

            let barrier_done = Arc::new((Mutex::new(false), Condvar::new()));

            // Worker 1
            let c1 = Arc::clone(&completed);
            let b1 = Arc::clone(&barrier_done);
            let w1 = thread::spawn(move || {
                // "Process tile"
                let prev = c1.fetch_add(1, Ordering::Release);
                if prev + 1 == total_workers {
                    let (lock, cvar) = &*b1;
                    let mut done = lock.lock().unwrap();
                    *done = true;
                    cvar.notify_all();
                }
            });

            // Worker 2
            let c2 = Arc::clone(&completed);
            let b2 = Arc::clone(&barrier_done);
            let w2 = thread::spawn(move || {
                let prev = c2.fetch_add(1, Ordering::Release);
                if prev + 1 == total_workers {
                    let (lock, cvar) = &*b2;
                    let mut done = lock.lock().unwrap();
                    *done = true;
                    cvar.notify_all();
                }
            });

            // Consumer waits for barrier
            let (lock, cvar) = &*barrier_done;
            let mut done = lock.lock().unwrap();
            while !*done {
                done = cvar.wait(done).unwrap();
            }

            w1.join().unwrap();
            w2.join().unwrap();

            // After barrier, all workers must have completed
            assert_eq!(completed.load(Ordering::Acquire), total_workers);
        });
    }

    // ---- Test 3: Backpressure - producer blocks when queue full, unblocks on drain ----

    #[test]
    fn loom_backpressure() {
        loom::model(|| {
            // Capacity 1: producer must block after first push
            let queue = Arc::new(BoundedQueue::new(1));
            let producer_progress = Arc::new(AtomicUsize::new(0));

            let q = Arc::clone(&queue);
            let pp = Arc::clone(&producer_progress);
            let producer = thread::spawn(move || {
                q.push(1);
                pp.fetch_add(1, Ordering::Release);
                // This should block until consumer pops
                q.push(2);
                pp.fetch_add(1, Ordering::Release);
            });

            let q = Arc::clone(&queue);
            let consumer = thread::spawn(move || {
                let v1 = q.pop();
                let v2 = q.pop();
                assert!(v1.is_some());
                assert!(v2.is_some());
            });

            producer.join().unwrap();
            queue.close();
            consumer.join().unwrap();

            // Both items were pushed and consumed
            assert_eq!(producer_progress.load(Ordering::Acquire), 2);
        });
    }

    // ---- Test 4: Multiple producers with bounded queue ----

    #[test]
    fn loom_multi_producer_bounded() {
        loom::model(|| {
            let queue = Arc::new(BoundedQueue::new(1));
            let sum = Arc::new(AtomicUsize::new(0));

            let q1 = Arc::clone(&queue);
            let p1 = thread::spawn(move || {
                q1.push(10);
            });

            let q2 = Arc::clone(&queue);
            let p2 = thread::spawn(move || {
                q2.push(20);
            });

            let q3 = Arc::clone(&queue);
            let s = Arc::clone(&sum);
            let consumer = thread::spawn(move || {
                while let Some(val) = q3.pop() {
                    s.fetch_add(val, Ordering::Relaxed);
                }
            });

            p1.join().unwrap();
            p2.join().unwrap();
            queue.close();
            consumer.join().unwrap();

            assert_eq!(sum.load(Ordering::Relaxed), 30);
        });
    }
}
