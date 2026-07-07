//! Bounded MPSC queue used by the parallel tile-emission paths.

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
