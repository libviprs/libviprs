//! Crate-wide mutex-poison policy.
//!
//! A `std::sync::Mutex` becomes *poisoned* when a thread panics while holding
//! its guard. By default the next `lock()` returns `Err(PoisonError)`, and the
//! common `.lock().unwrap()` / `.expect(...)` idioms turn that into a *second*
//! panic. In a tile run the worker threads all share the same sink / engine /
//! dedupe / observer locks, so one panicking worker used to cascade: its poison
//! re-panicked every subsequent `write_tile`, checkpoint append and observer
//! callback, tearing down the whole run and losing the final checkpoint on the
//! write path (issue #117).
//!
//! # The single policy
//!
//! **A poisoned lock is never re-raised as a panic.** How the poison is handled
//! depends on whether the guarded state stays usable after a holder panics:
//!
//! * **Consistent-state locks — recover.** When the data behind the lock is a
//!   plain in-memory collection, counter, or bookkeeping map (event logs,
//!   collected tiles, checkpoint metadata, the dedupe `refs`/`seen` maps, the
//!   `FsSink` leaf fields), a panic can only leave it *logically incomplete*,
//!   never structurally corrupt — a `Vec`/`BTreeMap` is still a valid
//!   `Vec`/`BTreeMap` between operations. These sites recover the guard via
//!   [`recover`] (`unwrap_or_else(|p| p.into_inner())`) and carry on. Any
//!   semantic gap (a missing digest, a not-yet-recorded reference) is caught
//!   downstream by on-disk digest verification and checkpoint validation rather
//!   than by aborting the run.
//!
//! * **Fragile write-path locks — fail cleanly.** When a panic can leave the
//!   guarded state part-way through a mutation that later operations cannot
//!   safely build on — the packfile's sequential archive writer, where a
//!   half-written tar/zip entry would corrupt every subsequent append — the
//!   poison is surfaced as a typed error on the fallible path
//!   (`SinkError::Other`, see [`crate::sink_packfile`]) so the operation aborts
//!   cleanly instead of recovering onto an unusable writer.
//!
//! Either way the outcome is the same: **one worker's panic can no longer
//! cascade into unrelated panics elsewhere in the run.**
//!
//! New lock sites in `sink`/`engine`/`dedupe`/`observe` must follow this
//! policy: call [`recover`] for consistent state, or convert the poison to a
//! typed error on a fragile write path. Never `.lock().unwrap()` /
//! `.lock().expect(...)`.

use std::sync::{Mutex, MutexGuard};

/// Acquire `mutex`, recovering the guard if the lock is poisoned.
///
/// This is the [consistent-state](self#the-single-policy) half of the crate
/// poison policy: a prior holder's panic leaves the guarded value logically
/// incomplete but structurally valid, so we take the inner guard via
/// [`std::sync::PoisonError::into_inner`] and continue instead of re-panicking.
#[inline]
pub(crate) fn recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
