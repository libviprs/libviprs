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
//! ```ignore
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

/// Cheap xorshift64-based pseudo-random nanosecond value in `[0, max_nanos)`.
///
/// Combines the per-process seed with a monotonic per-sink counter to
/// de-correlate jitter across both calls and sinks without touching the OS
/// on every invocation. Good enough for jitter; not cryptographic.
fn sample_jitter(max_nanos: u64, jitter_tick: &AtomicU64) -> u64 {
    if max_nanos == 0 {
        return 0;
    }
    let tick = jitter_tick.fetch_add(1, Ordering::Relaxed);
    let mut x = process_seed().wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15));
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FailurePolicy {
    FailFast,
    RetryThenFail(RetryPolicy),
    RetryThenSkip(RetryPolicy),
}

impl Default for FailurePolicy {
    fn default() -> Self {
        Self::FailFast
    }
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

    // Safe: `clamped` is non-negative and bounded by `max_nanos`, which fits
    // in u128 by construction (it came from a Duration).
    let nanos = clamped as u128;
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
pub struct RetryingSink<S: TileSink> {
    inner: S,
    policy: RetryPolicy,
    retry_count: AtomicU64,
    skipped_due_to_failure: AtomicU64,
    /// Per-sink monotonic tick used to de-correlate jitter across calls.
    jitter_tick: AtomicU64,
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
        }
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
    fn backoff_sleep(&self, attempt: u32) {
        let base = compute_backoff(&self.policy, attempt);
        let total = if self.policy.jitter {
            let max_jitter_nanos = (base / 2).as_nanos() as u64;
            let jitter_nanos = sample_jitter(max_jitter_nanos, &self.jitter_tick);
            base + Duration::from_nanos(jitter_nanos)
        } else {
            base
        };
        if !total.is_zero() {
            thread::sleep(total);
        }
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
                    self.backoff_sleep(attempt);
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

    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        self.inner.record_engine_config(config);
    }

    fn sink_retry_count(&self) -> u64 {
        self.retry_count.load(Ordering::Relaxed) + self.inner.sink_retry_count()
    }

    fn sink_skipped_due_to_failure(&self) -> u64 {
        self.skipped_due_to_failure.load(Ordering::Relaxed)
            + self.inner.sink_skipped_due_to_failure()
    }

    fn note_sink_skipped(&self) {
        self.skipped_due_to_failure.fetch_add(1, Ordering::Relaxed);
        // Forward so inner wrappers (e.g. another RetryingSink) can also
        // observe the skip. An actual terminal sink ignores the hook via the
        // default no-op implementation.
        self.inner.note_sink_skipped();
    }

    fn checkpoint_root(&self) -> Option<&std::path::Path> {
        self.inner.checkpoint_root()
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
