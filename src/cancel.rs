//! Cooperative cancellation for long-running pyramid generation.
//!
//! A [`CancelToken`] is a cheap, cloneable handle around a shared
//! [`AtomicBool`]. Clone it, hand one copy to the engine via
//! [`EngineBuilder::with_cancel`](crate::EngineBuilder::with_cancel) (or
//! [`EngineConfig::with_cancel`](crate::EngineConfig::with_cancel)) and keep
//! the other; calling [`CancelToken::cancel`] from any thread — a signal
//! handler, a UI "stop" button, a watchdog timer — asks the running engine to
//! stop at the next cooperative checkpoint and return
//! [`EngineError::Cancelled`](crate::EngineError::Cancelled).
//!
//! The engines check the token at coarse-grained boundaries where stopping is
//! cheap and leaves the output in a well-defined partial state:
//!
//! * the monolithic engine checks at the start of every pyramid level and
//!   before extracting each tile (and each parallel tile chunk);
//! * the streaming engine checks before rendering each strip;
//! * the map-reduce engine checks before each batch of strips;
//! * the retry backoff sleeps in short slices and aborts between them, so an
//!   in-flight exponential backoff does not have to run to completion before
//!   the run can stop.
//!
//! Cancellation is cooperative: a token set mid-tile takes effect at the next
//! checkpoint, not instantly. No work already handed to the sink is rolled
//! back — a resumable run's checkpoint still records every tile that was
//! durably written before the stop, so a later `--resume` picks up cleanly.
//!
//! # Example
//!
//! ```
//! use libviprs::CancelToken;
//!
//! let token = CancelToken::new();
//! assert!(!token.is_cancelled());
//!
//! let worker = token.clone();
//! worker.cancel(); // from any thread
//! assert!(token.is_cancelled());
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// A cloneable cooperative-cancellation handle shared with a running engine.
///
/// All clones observe the same underlying flag, so cancelling any clone
/// cancels the run. Cheap to clone (one `Arc` bump) and safe to share across
/// threads. See the [module docs](self) for where the engines poll it.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    /// A wall-clock bound a conversion put on its own copy of the token (#1674). It never reaches the
    /// shared flag, so a deadline does not cancel the caller's other clones.
    deadline: Option<Instant>,
}

impl PartialEq for CancelToken {
    /// Two tokens are equal when they are clones of one another, because that
    /// is the only way two of them can ever observe the same cancellation.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.flag, &other.flag)
    }
}

impl Eq for CancelToken {}

impl CancelToken {
    /// Create a fresh, un-cancelled token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            deadline: None,
        }
    }

    /// A copy of this token that also reads as cancelled once `at` has passed (the earlier of `at`
    /// and a deadline it already has). The copy shares the flag, and the deadline stays on the copy.
    #[cfg(feature = "pvec-convert")]
    pub(crate) fn with_deadline(&self, at: Instant) -> Self {
        Self {
            flag: self.flag.clone(),
            deadline: Some(self.deadline.map_or(at, |d| d.min(at))),
        }
    }

    /// `true` when the deadline passed and the flag was not the reason: a cancel by the caller wins.
    #[cfg(feature = "pvec-convert")]
    pub(crate) fn deadline_passed(&self) -> bool {
        !self.flag.load(Ordering::SeqCst) && self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Request cancellation. Idempotent — calling it more than once, or from
    /// several threads, is harmless. Every clone of this token observes the
    /// change.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Returns `true` once [`cancel`](Self::cancel) has been called on this
    /// token or any of its clones (or, for the copy a conversion keeps while it has a
    /// deadline, once the deadline passed).
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh token is not cancelled; cancelling flips the flag.
    #[test]
    fn new_token_is_not_cancelled() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        t.cancel();
        assert!(t.is_cancelled());
    }

    /// Cancelling one clone is observable through every other clone: all
    /// clones share the same underlying flag.
    #[test]
    fn clones_share_the_same_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled(), "cancelling one clone must cancel all");
    }

    /// The `Default` impl behaves like `new` — an un-cancelled token.
    #[test]
    fn default_is_uncancelled() {
        let t = CancelToken::default();
        assert!(!t.is_cancelled());
    }

    /// `cancel` is idempotent across repeated calls and threads.
    #[test]
    fn cancel_is_idempotent() {
        let t = CancelToken::new();
        let t2 = t.clone();
        let h = std::thread::spawn(move || {
            t2.cancel();
            t2.cancel();
        });
        h.join().unwrap();
        t.cancel();
        assert!(t.is_cancelled());
    }
}
