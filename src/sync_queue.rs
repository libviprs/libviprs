//! Bounded MPSC queue used by the parallel tile-emission paths.
//!
//! `extract_and_emit_parallel` ([`crate::engine`]) and
//! `emit_strip_tiles_parallel` ([`crate::streaming_mapreduce`]) hand tiles
//! from a pool of scoped worker threads to a single consumer over this queue.
//! It provides the three properties those paths rely on: FIFO delivery,
//! producer backpressure at a fixed capacity, and clean teardown — dropping
//! the receiver wakes any blocked sender (the consumer early-error path), and
//! dropping the last sender ends the consumer's iterator (the
//! `drop(tx)`-before-consume invariant).
//!
//! Production builds delegate to [`std::sync::mpsc::sync_channel`], preserving
//! its exact semantics and performance. Under `--cfg loom` the identical API is
//! backed by a hand-written `Mutex` + `Condvar` queue that loom can instrument,
//! so the loom suite ([`crate::loom_tests`]) model-checks the very primitive the
//! engine ships on instead of a toy stand-in.

/// Error returned by [`Sender::send`] when the receiver has been dropped.
///
/// Carries the value that could not be delivered, mirroring
/// [`std::sync::mpsc::SendError`].
#[derive(Debug)]
pub struct SendError<T>(pub T);

pub use imp::bounded;

#[cfg(not(loom))]
mod imp {
    use super::SendError;
    use std::sync::mpsc::{Receiver as StdReceiver, SyncSender, sync_channel};

    /// Create a bounded MPSC queue with room for `capacity` in-flight items.
    ///
    /// A `capacity` of zero yields a rendezvous queue (each `send` blocks until
    /// a `recv` takes the item), matching [`sync_channel`].
    pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let (tx, rx) = sync_channel(capacity);
        (Sender(tx), Receiver(rx))
    }

    /// Sending half of a [`bounded`] queue. Cloneable for multiple producers.
    pub struct Sender<T>(SyncSender<T>);

    impl<T> Clone for Sender<T> {
        fn clone(&self) -> Self {
            Sender(self.0.clone())
        }
    }

    impl<T> Sender<T> {
        /// Send a value, blocking while the queue is at capacity. Returns
        /// [`SendError`] once the receiver has been dropped.
        pub fn send(&self, value: T) -> Result<(), SendError<T>> {
            self.0.send(value).map_err(|e| SendError(e.0))
        }
    }

    /// Receiving half of a [`bounded`] queue.
    pub struct Receiver<T>(StdReceiver<T>);

    impl<T> Receiver<T> {
        /// Receive the next value, blocking until one is available. Returns
        /// `None` once the queue is empty and every sender has been dropped.
        pub fn recv(&self) -> Option<T> {
            self.0.recv().ok()
        }
    }

    impl<T> IntoIterator for Receiver<T> {
        type Item = T;
        type IntoIter = IntoIter<T>;
        fn into_iter(self) -> IntoIter<T> {
            IntoIter(self)
        }
    }

    /// Blocking iterator that drains a [`Receiver`] until every sender is gone.
    pub struct IntoIter<T>(Receiver<T>);

    impl<T> Iterator for IntoIter<T> {
        type Item = T;
        fn next(&mut self) -> Option<T> {
            self.0.recv()
        }
    }
}

#[cfg(loom)]
mod imp {
    use super::SendError;
    use loom::sync::{Arc, Condvar, Mutex};
    use std::collections::VecDeque;

    struct Shared<T> {
        inner: Mutex<Inner<T>>,
        not_full: Condvar,
        not_empty: Condvar,
    }

    struct Inner<T> {
        buf: VecDeque<T>,
        capacity: usize,
        senders: usize,
        receiver_alive: bool,
    }

    /// Create a bounded MPSC queue with room for `capacity` in-flight items.
    ///
    /// The loom model treats a zero capacity as one; the production
    /// (non-loom) queue provides real rendezvous semantics.
    pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                buf: VecDeque::new(),
                capacity: capacity.max(1),
                senders: 1,
                receiver_alive: true,
            }),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        });
        (
            Sender {
                shared: shared.clone(),
            },
            Receiver { shared },
        )
    }

    /// Sending half of a [`bounded`] queue. Cloneable for multiple producers.
    pub struct Sender<T> {
        shared: Arc<Shared<T>>,
    }

    impl<T> Clone for Sender<T> {
        fn clone(&self) -> Self {
            {
                let mut inner = self.shared.inner.lock().unwrap();
                inner.senders += 1;
            }
            Sender {
                shared: self.shared.clone(),
            }
        }
    }

    impl<T> Sender<T> {
        /// Send a value, blocking while the queue is at capacity. Returns
        /// [`SendError`] once the receiver has been dropped.
        pub fn send(&self, value: T) -> Result<(), SendError<T>> {
            let mut inner = self.shared.inner.lock().unwrap();
            while inner.buf.len() >= inner.capacity && inner.receiver_alive {
                inner = self.shared.not_full.wait(inner).unwrap();
            }
            if !inner.receiver_alive {
                return Err(SendError(value));
            }
            inner.buf.push_back(value);
            drop(inner);
            self.shared.not_empty.notify_one();
            Ok(())
        }
    }

    impl<T> Drop for Sender<T> {
        fn drop(&mut self) {
            let last = {
                let mut inner = self.shared.inner.lock().unwrap();
                inner.senders -= 1;
                inner.senders == 0
            };
            // Wake the consumer so it can observe end-of-stream once the final
            // producer is gone (the `drop(tx)`-before-consume invariant).
            if last {
                self.shared.not_empty.notify_all();
            }
        }
    }

    /// Receiving half of a [`bounded`] queue.
    pub struct Receiver<T> {
        shared: Arc<Shared<T>>,
    }

    impl<T> Receiver<T> {
        /// Receive the next value, blocking until one is available. Returns
        /// `None` once the queue is empty and every sender has been dropped.
        pub fn recv(&self) -> Option<T> {
            let mut inner = self.shared.inner.lock().unwrap();
            loop {
                if let Some(value) = inner.buf.pop_front() {
                    drop(inner);
                    self.shared.not_full.notify_one();
                    return Some(value);
                }
                if inner.senders == 0 {
                    return None;
                }
                inner = self.shared.not_empty.wait(inner).unwrap();
            }
        }
    }

    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            {
                let mut inner = self.shared.inner.lock().unwrap();
                inner.receiver_alive = false;
            }
            // Wake every blocked sender so it observes teardown and returns
            // `SendError` instead of deadlocking (the consumer early-error path).
            self.shared.not_full.notify_all();
        }
    }

    impl<T> IntoIterator for Receiver<T> {
        type Item = T;
        type IntoIter = IntoIter<T>;
        fn into_iter(self) -> IntoIter<T> {
            IntoIter(self)
        }
    }

    /// Blocking iterator that drains a [`Receiver`] until every sender is gone.
    pub struct IntoIter<T>(Receiver<T>);

    impl<T> Iterator for IntoIter<T> {
        type Item = T;
        fn next(&mut self) -> Option<T> {
            self.0.recv()
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::{bounded, SendError};
    use std::thread;
    use std::time::Duration;

    // No items may be lost or reordered when a single producer feeds a bounded
    // queue and a consumer drains it through the iterator adapter.
    #[test]
    fn fifo_single_producer_no_lost_items() {
        let (tx, rx) = bounded::<u32>(2);
        let producer = thread::spawn(move || {
            for i in 0..100 {
                tx.send(i).unwrap();
            }
        });
        let got: Vec<u32> = rx.into_iter().collect();
        producer.join().unwrap();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }

    // Dropping the last sender must terminate the consumer's iterator after it
    // has observed every buffered item — the `drop(tx)`-before-consume invariant
    // the engine relies on to know when a level is fully emitted.
    #[test]
    fn drop_tx_terminates_consumer() {
        let (tx, rx) = bounded::<u32>(4);
        let producer = thread::spawn(move || {
            for i in 0..8 {
                tx.send(i).unwrap();
            }
            // tx dropped here.
        });
        let mut count = 0;
        for _ in rx {
            count += 1;
        }
        producer.join().unwrap();
        assert_eq!(count, 8);
    }

    // A sender blocked on a full queue must wake and observe teardown once the
    // receiver is dropped, rather than deadlocking — the consumer early-error
    // path in the engine drops `rx` while producers are still blocked.
    #[test]
    fn receiver_drop_unblocks_blocked_sender() {
        let (tx, rx) = bounded::<u32>(1);
        tx.send(0).unwrap(); // buffer now full
        let sender = thread::spawn(move || tx.send(1));
        thread::sleep(Duration::from_millis(50));
        drop(rx);
        let res: Result<(), SendError<u32>> = sender.join().unwrap();
        assert!(
            res.is_err(),
            "a blocked sender must observe receiver teardown"
        );
    }
}
