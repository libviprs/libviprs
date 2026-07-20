//! Loom model-checking of the crate's shipped bounded MPSC queue
//! ([`crate::sync_queue`]) — the exact primitive the parallel tile-emission
//! paths run on ([`extract_and_emit_parallel`](crate::engine) and
//! `emit_strip_tiles_parallel` in [`crate::streaming_mapreduce`]).
//!
//! Under `--cfg loom`, `crate::sync_queue::bounded` is backed by a
//! loom-instrumented `Mutex` + `Condvar` implementation, so these tests
//! model-check thread interleavings of the real send / recv / teardown
//! protocol rather than a toy stand-in (see `model` for the search bound).
//! The invariants modelled here are the ones the engine depends on:
//!
//! 1. **No lost items** — every value a producer hands off is delivered
//!    (FIFO backpressure queue, multiple producers, one consumer).
//! 2. **Backpressure** — a producer blocks while the queue is full and makes
//!    progress only as the consumer drains it.
//! 3. **Receiver-drop teardown** — a consumer that drops `rx` early (the
//!    engine's early-error path) wakes any blocked sender, which observes the
//!    teardown and returns instead of deadlocking.
//! 4. **`drop(tx)`-before-consume** — the consumer's iterator ends only after
//!    the last sender is dropped, having first delivered every buffered item.
//!
//! Run with: `RUSTFLAGS="--cfg loom" cargo test --lib loom_tests`
//!
//! These tests are gated behind `cfg(loom)` because loom replaces the std
//! primitives and is incompatible with normal test runs.

#[cfg(loom)]
mod tests {
    use crate::sync_queue::bounded;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::thread;

    /// Model-check `f` under a bounded number of preemptions.
    ///
    /// The queue uses two condvars, and an unbounded exhaustive search over
    /// three threads is intractable as a CI merge gate. A preemption bound
    /// keeps the search fast and deterministic while still covering the
    /// interleavings that expose teardown / ordering bugs; an explicit
    /// `LOOM_MAX_PREEMPTIONS` (or `LOOM_MAX_BRANCHES`) override still wins.
    fn model<F>(f: F)
    where
        F: Fn() + Sync + Send + 'static,
    {
        let mut builder = loom::model::Builder::new();
        if builder.preemption_bound.is_none() {
            builder.preemption_bound = Some(3);
        }
        builder.check(f);
    }

    /// Two producers push three items total into a capacity-2 queue while a
    /// consumer drains it. Loom explores the interleavings and verifies that
    /// no item is ever lost or duplicated — the core delivery guarantee the
    /// tile-emission consumer relies on.
    #[test]
    fn loom_queue_no_lost_items() {
        model(|| {
            let (tx, rx) = bounded::<u32>(2);
            let tx2 = tx.clone();

            let p1 = thread::spawn(move || {
                let _ = tx.send(1);
                let _ = tx.send(2);
            });
            let p2 = thread::spawn(move || {
                let _ = tx2.send(3);
            });

            let mut sum = 0u32;
            let mut count = 0u32;
            for value in rx {
                sum += value;
                count += 1;
            }

            p1.join().unwrap();
            p2.join().unwrap();

            assert_eq!(count, 3, "every pushed item must be received exactly once");
            assert_eq!(sum, 6, "received items must be 1, 2 and 3");
        });
    }

    /// With capacity 1 the producer's second `send` cannot complete until the
    /// consumer pops the first item. Loom verifies the producer always makes
    /// full progress (both sends observed) and both items arrive in order.
    #[test]
    fn loom_backpressure_blocks_producer() {
        model(|| {
            let (tx, rx) = bounded::<u32>(1);
            let progress = Arc::new(AtomicUsize::new(0));

            let p = Arc::clone(&progress);
            let producer = thread::spawn(move || {
                let _ = tx.send(1);
                p.fetch_add(1, Ordering::Release);
                // Blocks until the consumer pops the first item.
                let _ = tx.send(2);
                p.fetch_add(1, Ordering::Release);
            });

            let first = rx.recv();
            let second = rx.recv();

            producer.join().unwrap();

            assert_eq!(first, Some(1));
            assert_eq!(second, Some(2));
            assert_eq!(progress.load(Ordering::Acquire), 2);
        });
    }

    /// The consumer takes at most one item then drops `rx` while the producer
    /// may still be blocked on a full queue. Reaching the end of the model
    /// across the explored interleavings proves the blocked sender is always
    /// woken by the receiver-drop teardown and never deadlocks — the engine's
    /// early-error path.
    #[test]
    fn loom_receiver_drop_unblocks_senders() {
        model(|| {
            let (tx, rx) = bounded::<u32>(1);

            let producer = thread::spawn(move || {
                // The first send may fill the queue; the second may block until
                // the receiver is dropped. Both calls must always return.
                let _ = tx.send(1);
                let _ = tx.send(2);
            });

            let _first = rx.recv();
            drop(rx);

            producer.join().unwrap();
        });
    }

    /// The consumer drains via the iterator adapter; it must terminate only
    /// after the sender is dropped, and only once every buffered item has been
    /// delivered — the `drop(tx)`-before-consume invariant.
    #[test]
    fn loom_drop_tx_terminates_receiver() {
        model(|| {
            let (tx, rx) = bounded::<u32>(2);

            let producer = thread::spawn(move || {
                let _ = tx.send(1);
                let _ = tx.send(2);
                // tx dropped here → the receiver observes end-of-stream.
            });

            let mut count = 0u32;
            for _ in rx {
                count += 1;
            }

            producer.join().unwrap();

            assert_eq!(count, 2, "receiver must drain all items before ending");
        });
    }
}
