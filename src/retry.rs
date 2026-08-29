//! Retry wrapper and failure policy for tile sinks.
//!
//! This module provides a [`RetryingSink`] that wraps any [`TileSink`] and
//! retries failed `write_tile` calls with an exponential backoff. The
//! companion [`FailurePolicy`] is consumed at the engine level to decide
//! whether retry exhaustion should fail the entire pyramid (`FailFast` /
//! `RetryThenFail`) or skip the offending tile and continue
//! (`RetryThenSkip`).
//!
//! # Design
//!
//! * [`RetryPolicy`] is a plain, `Clone`-able value type with public fields,
//!   so callers can build it inline in tests or config files.
//! * [`RetryingSink`] is transparent for healthy sinks — on success it
//!   forwards to the inner sink with no allocation and no atomic writes
//!   beyond the single atomic read in the happy path.
//! * Backoff is computed by the free function [`compute_backoff`], which is
//!   deterministic (jitter-free) so unit tests can pin exact values.
//! * Jitter is produced from a cheap xorshift64 PRNG seeded once per
//!   process (via `RandomState::new().hash_one(&())`) and mixed with a
//!   per-sink monotonic counter — no external `rand` dependency and no
//!   per-call syscalls after the first invocation.
//!
//! # Example
//!
//! ```
//! use libviprs::retry::{FailurePolicy, RetryPolicy, RetryingSink};
//! use libviprs::sink::MemorySink;
//!
//! let policy = RetryPolicy::default();
//! let sink = RetryingSink::new(MemorySink::new(), policy);
//! ```
//!
//! The engine itself inspects the [`FailurePolicy`] carried in
//! `EngineConfig` to decide how to interpret a terminal error from
//! `write_tile`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use crate::sink::{SinkError, Tile, TileSink};

// ---------------------------------------------------------------------------
// Process-wide PRNG seed
// ---------------------------------------------------------------------------

/// Returns a per-process random seed drawn once from OS entropy via
/// [`std::hash::RandomState`]. Subsequent calls reuse the cached value, so
/// jitter sampling is syscall-free after the first invocation.
fn process_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        use std::hash::{BuildHasher, RandomState};
        RandomState::new().hash_one(()) // one OS-entropy draw per process
    })
}

/// Draws a distinct nonce for each [`RetryingSink`] constructed in this
/// process, so two sinks sharing the same [`process_seed`] and both ticking
/// from zero do not emit identical jitter streams. Monotonic and wrapping;
/// only distinctness matters, not the value.
fn next_sink_nonce() -> u64 {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    NONCE.fetch_add(1, Ordering::Relaxed)
}

/// Cheap xorshift64-based pseudo-random nanosecond value in `[0, max_nanos)`.
///
/// Combines the per-process seed with a per-sink nonce and a monotonic
/// per-sink counter to de-correlate jitter across both calls and sinks
/// without touching the OS on every invocation. Mixing `sink_nonce` is what
/// keeps two sinks in one process from producing the same sequence even
/// though `process_seed()` is shared and each `jitter_tick` starts at zero.
/// Good enough for jitter; not cryptographic.
fn sample_jitter(max_nanos: u64, jitter_tick: &AtomicU64, sink_nonce: u64) -> u64 {
    if max_nanos == 0 {
        return 0;
    }
    let tick = jitter_tick.fetch_add(1, Ordering::Relaxed);
    let mut x = process_seed()
        .wrapping_add(sink_nonce.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(tick.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // xorshift64
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x % max_nanos
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// Parameters controlling the exponential-backoff retry loop in
/// [`RetryingSink`].
///
/// * `max_retries` — number of **additional** attempts made after the first
///   failed write. A value of `3` means up to 4 total attempts.
/// * `initial_backoff` — sleep before retry #1.
/// * `multiplier` — applied geometrically to produce retry #2, #3, ….
/// * `max_backoff` — hard cap; the computed backoff is clamped to this value
///   before any jitter is applied.
/// * `jitter` — when `true`, a uniformly-distributed random slice in
///   `[0, backoff / 2]` is added to each sleep. Jitter helps de-synchronise
///   many parallel workers hammering the same flaky endpoint.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-retry-max)
/// (and [`--retry-backoff`](https://libviprs.org/cli/#flag-retry-backoff) for the backoff arg)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub multiplier: f32,
    pub max_backoff: Duration,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(50),
            multiplier: 2.0,
            max_backoff: Duration::from_secs(5),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Construct a policy with explicit retry count and initial backoff,
    /// defaulting the remaining fields. Combine with the `with_*` builders
    /// to tune `multiplier`, `max_backoff`, and `jitter`.
    pub fn new(max_retries: u32, initial_backoff: Duration) -> Self {
        Self {
            max_retries,
            initial_backoff,
            ..Self::default()
        }
    }

    /// Construct a policy that never retries (shorthand for `max_retries = 0`).
    ///
    /// Use this when you want [`FailurePolicy::FailFast`]-style behaviour but
    /// still want to feed a `RetryPolicy` value through an API that requires
    /// one (e.g. a `match` arm returning `RetryPolicy` from every branch).
    pub fn fail_fast() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Short-form alias for [`RetryPolicy::with_max_retries`].
    pub fn with_max(self, n: u32) -> Self {
        self.with_max_retries(n)
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_initial_backoff(mut self, d: Duration) -> Self {
        self.initial_backoff = d;
        self
    }

    pub fn with_multiplier(mut self, m: f32) -> Self {
        self.multiplier = m;
        self
    }

    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    pub fn with_jitter(mut self, enabled: bool) -> Self {
        self.jitter = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// FailurePolicy
// ---------------------------------------------------------------------------

/// How the engine should react when a `write_tile` call terminally fails.
///
/// * [`FailurePolicy::FailFast`] — propagate the first error; no retries.
/// * [`FailurePolicy::RetryThenFail`] — retry per the embedded policy, and
///   propagate the last error if every retry is exhausted.
/// * [`FailurePolicy::RetryThenSkip`] — retry per the embedded policy; on
///   exhaustion, account the tile in
///   `EngineResult::skipped_due_to_failure` and continue.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-failure-policy)
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FailurePolicy {
    /// Propagate the first error; no retries.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-failure-policy)
    #[default]
    FailFast,
    /// Retry per the embedded policy; propagate the last error if every
    /// retry is exhausted.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-failure-policy)
    RetryThenFail(RetryPolicy),
    /// Retry per the embedded policy; on exhaustion, skip the tile and
    /// continue.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-failure-policy)
    RetryThenSkip(RetryPolicy),
}

// ---------------------------------------------------------------------------
// Backoff computation
// ---------------------------------------------------------------------------

/// Deterministic backoff computation (no jitter).
///
/// Returns `policy.initial_backoff * multiplier.powi(attempt)` clamped to
/// `policy.max_backoff`. `attempt` is zero-based: `attempt == 0` gives the
/// wait before the very first retry.
///
/// Used by [`RetryingSink`] and is directly unit-testable — tests lean on
/// this function to assert the geometric progression without having to
/// observe real sleeps.
pub fn compute_backoff(policy: &RetryPolicy, attempt: u32) -> Duration {
    let base_nanos = policy.initial_backoff.as_nanos() as f64;
    let multiplier = policy.multiplier as f64;
    // `powi` with a potentially large `attempt` can overflow to +inf; the cap
    // below handles that cleanly via saturation.
    let scaled = base_nanos * multiplier.powi(attempt as i32);

    let max_nanos = policy.max_backoff.as_nanos() as f64;
    let clamped = if !scaled.is_finite() || scaled > max_nanos {
        max_nanos
    } else if scaled < 0.0 {
        0.0
    } else {
        scaled
    };

    // Round to the nearest nanosecond before casting: `f64` cannot exactly
    // represent every product of `initial * multiplier^attempt`, so a naive
    // truncating cast can produce 39_999_999 ns where 40_000_000 is intended.
    let nanos = clamped.round() as u128;
    duration_from_nanos_u128(nanos)
}

/// Build a `Duration` from a `u128` nanosecond count, saturating at
/// `Duration::MAX`. Keeps the arithmetic branchless on the happy path.
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;
    let secs = (nanos / NANOS_PER_SEC) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

// ---------------------------------------------------------------------------
// RetryingSink
// ---------------------------------------------------------------------------

/// Sink decorator that retries failed `write_tile` calls with exponential
/// backoff.
///
/// Wrap any [`TileSink`] to get automatic retry behaviour. The retry loop
/// runs **inside** `write_tile`, so from the engine's point of view a
/// transient error is transparent — the engine only sees the terminal
/// outcome (success, or the last error after exhausting retries).
///
/// # Counters
///
/// Two atomic counters record activity for the engine to aggregate:
///
/// * [`RetryingSink::retry_count`] — number of retry attempts (the first
///   try does not count; only subsequent retries do).
/// * [`RetryingSink::skipped_due_to_failure`] — incremented by the engine
///   (not by `RetryingSink` itself) when `RetryThenSkip` drops a tile.
///   Exposed here so the engine can stash the running total without a
///   second data structure.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-retry-max)
pub struct RetryingSink<S: TileSink> {
    inner: S,
    policy: RetryPolicy,
    retry_count: AtomicU64,
    skipped_due_to_failure: AtomicU64,
    /// Per-sink monotonic tick used to de-correlate jitter across calls.
    jitter_tick: AtomicU64,
    /// Per-sink nonce mixed into the jitter seed so distinct sinks in one
    /// process draw independent jitter streams despite the shared
    /// [`process_seed`] and both ticks starting at zero.
    jitter_nonce: u64,
    /// Optional cooperative-cancellation token. When set, the exponential
    /// backoff sleeps in short slices and aborts between them, so an in-flight
    /// retry does not have to run its full schedule before the run can stop.
    cancel: Option<crate::cancel::CancelToken>,
}

impl<S: TileSink> RetryingSink<S> {
    /// Wrap `inner` with the given retry `policy`.
    pub fn new(inner: S, policy: RetryPolicy) -> Self {
        Self {
            inner,
            policy,
            retry_count: AtomicU64::new(0),
            skipped_due_to_failure: AtomicU64::new(0),
            jitter_tick: AtomicU64::new(0),
            jitter_nonce: next_sink_nonce(),
            cancel: None,
        }
    }

    /// Attach a [`CancelToken`](crate::cancel::CancelToken) so an in-flight
    /// backoff can be interrupted. The engine wires this from
    /// [`EngineConfig::cancel`](crate::engine::EngineConfig::cancel); a
    /// cancelled token stops the retry loop, and the engine then reports the
    /// run as [`EngineError::Cancelled`](crate::engine::EngineError::Cancelled).
    pub fn with_cancel(mut self, token: Option<crate::cancel::CancelToken>) -> Self {
        self.cancel = token;
        self
    }

    /// Total number of retry attempts observed by this sink so far.
    pub fn retry_count(&self) -> u64 {
        self.retry_count.load(Ordering::Relaxed)
    }

    /// Total number of tiles the engine marked as skipped via this sink
    /// under a `RetryThenSkip` failure policy.
    pub fn skipped_due_to_failure(&self) -> u64 {
        self.skipped_due_to_failure.load(Ordering::Relaxed)
    }

    /// Accessor for the wrapped sink — useful for integration tests that
    /// need to inspect side effects recorded by the inner sink (e.g. a
    /// `RecordingRetrySink`'s timestamps).
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Bump the skip counter. Called by the engine, not by the retry loop.
    #[doc(hidden)]
    pub fn note_skipped(&self) {
        self.skipped_due_to_failure.fetch_add(1, Ordering::Relaxed);
    }

    /// Borrow the retry policy this sink was configured with.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Sleep for the computed backoff, adding jitter if enabled.
    ///
    /// When a [`CancelToken`](crate::cancel::CancelToken) is attached, the
    /// sleep is broken into short slices and abandoned as soon as the token is
    /// cancelled. Returns `true` if the full backoff elapsed, `false` if it was
    /// cut short by cancellation.
    fn backoff_sleep(&self, attempt: u32) -> bool {
        let base = compute_backoff(&self.policy, attempt);
        let total = if self.policy.jitter {
            let max_jitter_nanos = (base / 2).as_nanos() as u64;
            let jitter_nanos =
                sample_jitter(max_jitter_nanos, &self.jitter_tick, self.jitter_nonce);
            base + Duration::from_nanos(jitter_nanos)
        } else {
            base
        };
        if total.is_zero() {
            return true;
        }
        // No cancel token: sleep the whole duration in one go (fast path).
        let Some(cancel) = &self.cancel else {
            thread::sleep(total);
            return true;
        };
        // Interruptible sleep: poll the token between short slices so an
        // in-flight backoff can be aborted well before its full schedule.
        const SLICE: Duration = Duration::from_millis(25);
        let mut remaining = total;
        while !remaining.is_zero() {
            if cancel.is_cancelled() {
                return false;
            }
            let step = remaining.min(SLICE);
            thread::sleep(step);
            remaining -= step;
        }
        !cancel.is_cancelled()
    }
}

impl<S: TileSink> TileSink for RetryingSink<S> {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        // First attempt — the common case on healthy sinks. Fast path: no
        // atomic writes, no allocation.
        match self.inner.write_tile(tile) {
            Ok(()) => Ok(()),
            Err(first_err) => {
                if self.policy.max_retries == 0 {
                    return Err(first_err);
                }
                let mut last_err = first_err;
                for attempt in 0..self.policy.max_retries {
                    // If cancellation fires during (or before) the backoff,
                    // stop retrying and return the last error immediately. The
                    // engine polls the same token on the write-error path and
                    // promotes this to EngineError::Cancelled.
                    if !self.backoff_sleep(attempt) {
                        return Err(last_err);
                    }
                    if self.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                        return Err(last_err);
                    }
                    self.retry_count.fetch_add(1, Ordering::Relaxed);
                    match self.inner.write_tile(tile) {
                        Ok(()) => return Ok(()),
                        Err(e) => last_err = e,
                    }
                }
                Err(last_err)
            }
        }
    }

    fn finish(&self) -> Result<(), SinkError> {
        self.inner.finish()
    }

    /// Expose the wrapped sink so the trait's bookkeeping defaults forward
    /// through it. `RetryingSink` genuinely owns two counters
    /// (`sink_retry_count`, `sink_skipped_due_to_failure`) and its own
    /// retry-loop marker (`applies_retry_policy`), so those stay overridden
    /// below; every purely-transparent hook (`record_engine_config`,
    /// `checkpoint_root`, `init_level_count`, `content_format`) is served by
    /// the default that reads this inner sink, so the wrapper cannot silently
    /// drop one (issue #137).
    fn inner_sink(&self) -> Option<&dyn TileSink> {
        Some(&self.inner)
    }

    fn sink_retry_count(&self) -> u64 {
        self.retry_count.load(Ordering::Relaxed) + self.inner.sink_retry_count()
    }

    fn sink_skipped_due_to_failure(&self) -> u64 {
        self.skipped_due_to_failure.load(Ordering::Relaxed)
            + self.inner.sink_skipped_due_to_failure()
    }

    fn note_sink_skipped(&self) {
        // Record the skip on this wrapper's own counter only — do NOT forward
        // to the inner sink. The engine calls `note_sink_skipped` exactly once
        // per dropped tile, on the outermost sink, and `sink_skipped_due_to_failure`
        // already recurses (self + inner) to aggregate the whole chain. Forwarding
        // here as well would let one skip through an N-deep stack of RetryingSinks
        // report N (mirrors how `sink_retry_count` sums per-level counters that are
        // each bumped only by their own retry loop).
        self.skipped_due_to_failure.fetch_add(1, Ordering::Relaxed);
    }

    fn applies_retry_policy(&self) -> bool {
        // This decorator runs the retry loop itself, so a caller that
        // pre-wrapped their sink must not be wrapped again by the builder.
        true
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::TileCoord;
    use crate::raster::Raster;
    use crate::sink::MemorySink;
    use std::sync::atomic::AtomicU32;

    // Compile-time assertion that `RetryingSink<S>` is `Send + Sync` for a
    // concrete `Send + Sync` inner sink. If this ever breaks, the engine's
    // parallel use of `RetryingSink` will stop compiling — catch it here.
    const _: fn() = || {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RetryingSink<MemorySink>>();
    };

    fn dummy_tile() -> Tile {
        let raster = Raster::new(1, 1, PixelFormat::Rgb8, vec![0, 0, 0]).unwrap();
        Tile {
            coord: TileCoord {
                level: 0,
                col: 0,
                row: 0,
            },
            raster,
            blank: false,
        }
    }

    struct CountingFailSink {
        budget: AtomicU32,
        calls: AtomicU64,
    }

    impl CountingFailSink {
        fn new(fail_times: u32) -> Self {
            Self {
                budget: AtomicU32::new(fail_times),
                calls: AtomicU64::new(0),
            }
        }
    }

    impl TileSink for CountingFailSink {
        fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let prev = self.budget.load(Ordering::SeqCst);
            if prev > 0
                && self
                    .budget
                    .compare_exchange(prev, prev - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                Err(SinkError::Other("fail".into()))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn default_policy_matches_spec() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.initial_backoff, Duration::from_millis(50));
        assert!((p.multiplier - 2.0).abs() < f32::EPSILON);
        assert_eq!(p.max_backoff, Duration::from_secs(5));
        assert!(p.jitter);
    }

    #[test]
    fn default_failure_policy_is_fail_fast() {
        assert_eq!(FailurePolicy::default(), FailurePolicy::FailFast);
    }

    #[test]
    fn compute_backoff_is_geometric() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff: Duration::from_millis(10),
            multiplier: 2.0,
            max_backoff: Duration::from_secs(60),
            jitter: false,
        };
        assert_eq!(compute_backoff(&policy, 0), Duration::from_millis(10));
        assert_eq!(compute_backoff(&policy, 1), Duration::from_millis(20));
        assert_eq!(compute_backoff(&policy, 2), Duration::from_millis(40));
        assert_eq!(compute_backoff(&policy, 3), Duration::from_millis(80));
    }

    #[test]
    fn compute_backoff_is_capped() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff: Duration::from_secs(1),
            multiplier: 10.0,
            max_backoff: Duration::from_secs(3),
            jitter: false,
        };
        assert_eq!(compute_backoff(&policy, 0), Duration::from_secs(1));
        assert_eq!(compute_backoff(&policy, 1), Duration::from_secs(3));
        assert_eq!(compute_backoff(&policy, 2), Duration::from_secs(3));
        // Exceedingly large attempt must still saturate at the cap.
        assert_eq!(compute_backoff(&policy, 100), Duration::from_secs(3));
    }

    #[test]
    fn retries_until_success() {
        let inner = CountingFailSink::new(2);
        let policy = RetryPolicy {
            max_retries: 5,
            initial_backoff: Duration::from_micros(1),
            multiplier: 1.0,
            max_backoff: Duration::from_millis(1),
            jitter: false,
        };
        let sink = RetryingSink::new(inner, policy);
        let tile = dummy_tile();
        sink.write_tile(&tile)
            .expect("should succeed after retries");
        assert_eq!(sink.retry_count(), 2);
        assert_eq!(sink.inner().calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn returns_last_error_when_exhausted() {
        let inner = CountingFailSink::new(100);
        let policy = RetryPolicy {
            max_retries: 2,
            initial_backoff: Duration::from_micros(1),
            multiplier: 1.0,
            max_backoff: Duration::from_millis(1),
            jitter: false,
        };
        let sink = RetryingSink::new(inner, policy);
        let tile = dummy_tile();
        let err = sink.write_tile(&tile).unwrap_err();
        match err {
            SinkError::Other(msg) => assert_eq!(msg, "fail"),
            other => panic!("unexpected error: {other:?}"),
        }
        // max_retries=2 → 1 initial + 2 retries = 3 total attempts.
        assert_eq!(sink.inner().calls.load(Ordering::SeqCst), 3);
        assert_eq!(sink.retry_count(), 2);
    }

    #[test]
    fn nested_wrappers_count_a_single_skip_once() {
        // Engine calls `note_sink_skipped` exactly once per dropped tile, on the
        // outermost sink. Through a two-deep RetryingSink chain that single skip
        // must aggregate to 1 — not 2. (Before the fix, `note_sink_skipped`
        // forwarded into the inner wrapper while `sink_skipped_due_to_failure`
        // also summed self+inner, double-counting.)
        let inner = RetryingSink::new(MemorySink::new(), RetryPolicy::default());
        let outer = RetryingSink::new(inner, RetryPolicy::default());

        outer.note_sink_skipped();

        assert_eq!(
            outer.sink_skipped_due_to_failure(),
            1,
            "one skip through nested RetryingSinks must count exactly once"
        );
    }

    #[test]
    fn jitter_is_decorrelated_across_sinks() {
        // Two sinks constructed in the same process share `process_seed()` and
        // both tick from zero. Without a per-sink nonce mixed into the seed they
        // would emit byte-for-byte identical jitter streams, defeating the
        // cross-sink de-correlation the jitter is meant to provide.
        let a = RetryingSink::new(MemorySink::new(), RetryPolicy::default());
        let b = RetryingSink::new(MemorySink::new(), RetryPolicy::default());

        // A wide range keeps the (already vanishing) chance of an incidental
        // per-tick collision negligible across the sampled sequence.
        let max = u32::MAX as u64;
        let seq_a: Vec<u64> = (0..16)
            .map(|_| sample_jitter(max, &a.jitter_tick, a.jitter_nonce))
            .collect();
        let seq_b: Vec<u64> = (0..16)
            .map(|_| sample_jitter(max, &b.jitter_tick, b.jitter_nonce))
            .collect();

        assert_ne!(
            seq_a, seq_b,
            "distinct sinks must draw independent jitter streams"
        );
    }

    #[test]
    fn zero_retries_returns_first_error_immediately() {
        let inner = CountingFailSink::new(1);
        let policy = RetryPolicy {
            max_retries: 0,
            initial_backoff: Duration::from_micros(1),
            multiplier: 2.0,
            max_backoff: Duration::from_millis(1),
            jitter: false,
        };
        let sink = RetryingSink::new(inner, policy);
        let tile = dummy_tile();
        assert!(sink.write_tile(&tile).is_err());
        assert_eq!(sink.retry_count(), 0);
        assert_eq!(sink.inner().calls.load(Ordering::SeqCst), 1);
    }
}
