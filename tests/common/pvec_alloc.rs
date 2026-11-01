//! The counting allocator the conversion benchmarks share (N8.2 to N8.5, #1064).
//!
//! `#[global_allocator]` has to be declared in the test binary itself, so a suite that wants the
//! numbers writes `#[global_allocator] static ALLOC: pvec_alloc::Counting = pvec_alloc::Counting;`
//! and reads them through [`mark`] and [`since`]. The basis is the live heap, not process RSS:
//! RSS never falls, so it cannot separate one run's peak from the drawing the harness itself
//! holds. Both are reported, because they answer different questions.

#![allow(dead_code)] // each suite uses its own part

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// Forwards to `System` and counts.
pub struct Counting;

fn grow(by: u64) {
    TOTAL.fetch_add(by, Ordering::Relaxed);
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` and only adds to atomic counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grow(l.size() as u64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grow(l.size() as u64);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as u64, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                grow((new - l.size()) as u64);
            } else {
                LIVE.fetch_sub((l.size() - new) as u64, Ordering::Relaxed);
            }
        }
        q
    }
}

/// Where a measurement started: the live heap then, with the peak and the running total reset to it.
#[derive(Debug, Clone, Copy)]
pub struct Mark {
    live: u64,
    total: u64,
}

/// Starts a measurement.
pub fn mark() -> Mark {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    Mark {
        live,
        total: TOTAL.load(Ordering::Relaxed),
    }
}

/// What the allocator saw since `m`: the peak of the live heap above where it stood, and the bytes
/// requested in all.
pub fn since(m: Mark) -> (u64, u64) {
    (
        PEAK.load(Ordering::Relaxed).saturating_sub(m.live),
        TOTAL.load(Ordering::Relaxed) - m.total,
    )
}
