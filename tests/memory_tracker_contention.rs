//! `MemoryTracker`'s saturating counter under real thread contention
//! (issue #1157).
//!
//! #1157 moves `alloc` and `dealloc` from `AtomicU64::fetch_update`, which
//! stable 1.99 deprecates, to `AtomicU64::update`, so the crate builds on 1.99
//! again. That's supposed to change nothing anyone can observe, and these hold
//! it to that: they pass on the old code and have to keep passing on the new.
//! The unit tests in `src/observe.rs` pin the saturating arithmetic on one
//! thread; these pin that it stays atomic when several threads race on the
//! same counter, which is the half a wrong call could break (a lost update, or
//! a clamp applied to a stale value). They're smoke tests, not a proof: a race
//! that loses an update shows up often on a multi-core box, never on a single
//! core.

use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use libviprs::MemoryTracker;

const THREADS: usize = 8;
const ROUNDS: u64 = 5_000;

/// Run `body` on `THREADS` threads that all start together, each with its own
/// handle on the same tracker.
fn race(t: &MemoryTracker, body: impl Fn(&MemoryTracker) + Sync) {
    let barrier = Barrier::new(THREADS);
    thread::scope(|s| {
        for _ in 0..THREADS {
            let t = t.clone();
            let (barrier, body) = (&barrier, &body);
            s.spawn(move || {
                barrier.wait();
                body(&t);
            });
        }
    });
}

/**
 * No update gets lost when every thread only allocates.
 * Works by racing 8 threads that each alloc(3) 5000 times; a read-then-write
 * that isn't one atomic step would drop some of them.
 * Input: 8 x 5000 x alloc(3) -> Output: current = peak = 120000 exactly.
 */
#[test]
fn concurrent_allocs_add_up_exactly() {
    let t = MemoryTracker::new();
    race(&t, |t| {
        for _ in 0..ROUNDS {
            t.alloc(3);
        }
    });
    let total = THREADS as u64 * ROUNDS * 3;
    assert_eq!(
        t.current_bytes(),
        total,
        "an alloc was lost under contention"
    );
    assert_eq!(t.peak_bytes(), total);
}

/**
 * Balanced pairs come back to zero, and the peak never claims more than could
 * have been held at once.
 * Works by racing 8 threads that each alloc(7) then dealloc(7) 5000 times, so
 * at most 8 allocations are ever outstanding together.
 * Input: 8 x 5000 x (alloc(7), dealloc(7)) -> Output: current = 0,
 * 7 <= peak <= 56.
 */
#[test]
fn balanced_pairs_return_to_zero_with_a_bounded_peak() {
    let t = MemoryTracker::new();
    race(&t, |t| {
        for _ in 0..ROUNDS {
            t.alloc(7);
            t.dealloc(7);
        }
    });
    assert_eq!(t.current_bytes(), 0);
    let peak = t.peak_bytes();
    assert!(
        (7..=THREADS as u64 * 7).contains(&peak),
        "peak {peak} is outside what 8 outstanding allocations of 7 can reach"
    );
}

/**
 * Racing allocs past `u64::MAX` clamp at the ceiling and never wrap or panic.
 * Works by starting 1000 below the ceiling and racing 8 threads that each
 * alloc(1) 5000 times, 40000 in all.
 * Input: alloc(u64::MAX - 1000), then 8 x 5000 x alloc(1) ->
 * Output: current = peak = u64::MAX.
 */
#[test]
fn racing_allocs_saturate_at_the_ceiling() {
    let t = MemoryTracker::new();
    t.alloc(u64::MAX - 1000);
    race(&t, |t| {
        for _ in 0..ROUNDS {
            t.alloc(1);
        }
    });
    assert_eq!(
        t.current_bytes(),
        u64::MAX,
        "current wrapped instead of clamping"
    );
    assert_eq!(t.peak_bytes(), u64::MAX);
}

/**
 * Racing deallocs past zero clamp at zero and leave the peak alone.
 * Works by allocating 1000 and racing 8 threads that each dealloc(1) 5000
 * times, then allocating once more to show `current` restarted from zero
 * rather than from a wrapped value.
 * Input: alloc(1000), 8 x 5000 x dealloc(1), alloc(5) -> Output: current = 5,
 * peak = 1000.
 */
#[test]
fn racing_deallocs_saturate_at_zero() {
    let t = MemoryTracker::new();
    t.alloc(1000);
    race(&t, |t| {
        for _ in 0..ROUNDS {
            t.dealloc(1);
        }
    });
    assert_eq!(t.current_bytes(), 0, "current wrapped below zero");
    t.alloc(5);
    assert_eq!(t.current_bytes(), 5);
    assert_eq!(t.peak_bytes(), 1000, "a wrapped current leaked into peak");
}

/**
 * Over-deallocating threads racing each other never push `current` or `peak`
 * past what's really outstanding, checked while the race runs.
 * Works by having 4 threads alloc(10) then dealloc(25) in a loop while a fifth
 * reads both counters, all five released together by one barrier so the
 * reader is already looking when the first write lands. Each thread holds at
 * most 10 at a time and `dealloc` clamps at zero, so neither counter can ever
 * exceed 4 x 10, whatever the interleaving; a clamp applied to a stale value,
 * or a wrap, would.
 * Input: 4 x 5000 x (alloc(10), dealloc(25)) with a concurrent reader ->
 * Output: every read <= 40, current = 0 at the end.
 */
#[test]
fn over_deallocation_under_contention_stays_within_what_is_outstanding() {
    const WRITERS: u64 = 4;
    let t = MemoryTracker::new();
    let done = AtomicBool::new(false);
    let start = Barrier::new(WRITERS as usize + 1);
    let bound = WRITERS * 10;
    let worst = thread::scope(|s| {
        let reader = s.spawn(|| {
            start.wait();
            let mut worst = 0;
            while !done.load(Ordering::Acquire) {
                worst = worst.max(t.current_bytes()).max(t.peak_bytes());
            }
            worst
        });
        let writers: Vec<_> = (0..WRITERS)
            .map(|_| {
                let t = t.clone();
                let start = &start;
                s.spawn(move || {
                    start.wait();
                    for _ in 0..ROUNDS {
                        t.alloc(10);
                        t.dealloc(25);
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().expect("a writer panicked");
        }
        done.store(true, Ordering::Release);
        reader.join().expect("the reader panicked")
    });
    assert!(
        worst <= bound,
        "a counter read {worst}, past the {bound} that can be outstanding"
    );
    assert!(t.peak_bytes() <= bound);
    assert_eq!(t.current_bytes(), 0);
}
