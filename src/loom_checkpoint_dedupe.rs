//! Loom model-checking of the checkpoint + dedupe interleavings where the
//! resume cluster's shipped concurrency bugs actually occurred (issue #278).
//!
//! The existing loom suite ([`crate::loom_tests`]) model-checks only the bounded
//! MPSC queue ([`crate::sync_queue`]). Every concurrency defect the resume
//! cluster fixed lived **outside** the queue, on the seam between the engine's
//! [`CheckpointState`](crate::engine), the sink's durability tracking, and the
//! [`DedupeIndex`](crate::dedupe) — exactly the interactions this module models:
//!
//! * **#273 / #277 — checkpoint never certifies an un-fsynced tile.**
//!   [`crate::engine::CheckpointState::flush`] snapshots the completed-tile
//!   *delta*, then invokes the sink durability barrier
//!   ([`crate::sink::TileSink::sync_pending`]) to fsync the tracked tile bytes,
//!   and only then appends the segment delta + publishes the header that
//!   certifies those coordinates. The ordering guarantee (flush doc, `engine.rs`)
//!   is that the delta is snapshotted **before** the barrier, so every
//!   coordinate the flush certifies had its tile path recorded (during
//!   `write_tile`, which precedes `mark_tile_completed`) before the barrier
//!   drained the pending set — so the barrier fsyncs at least every tile the
//!   delta certifies, even while a concurrent worker is writing more tiles.
//!
//! * **#275 — dedupe placement is content-addressed / coordinate-canonical.**
//!   [`crate::dedupe::DedupeIndex::record`] returns `WriteNew` for whichever
//!   occurrence of a content hash it sees first — under `tile_concurrency > 0`
//!   that is producer-completion order, non-deterministic. The full-payload
//!   holder is fixed only at `finish()` by
//!   [`crate::sink::FsSink::canonicalize_dedupe_layout`], which reassigns it to
//!   the **coordinate-minimal** occurrence. The final layout (holder +
//!   `blank_references`) must therefore be identical regardless of arrival order.
//!
//! * **#272 — resume seeding reconstructs identical state.**
//!   On resume, [`crate::sink::TileSink::seed_completed_tile`] replays each
//!   already-completed coordinate through the same record path, so a resumed
//!   `finish()` canonicalizes to the same placement an uninterrupted run
//!   produces — independent of which occurrence held the full payload before the
//!   crash and of the seeding order.
//!
//! These are **faithful protocol models**, not calls into the product types:
//! `CheckpointState` and `DedupeIndex` guard their state with `std::sync`
//! primitives, which loom cannot instrument, so the models reproduce the exact
//! lock/ordering protocol over `loom::sync` primitives and assert the invariants
//! the cluster fixes established. Each model is parameterised by an ordering enum
//! whose *correct* variant mirrors the shipped code (and passes under every
//! interleaving loom explores) and whose *buggy* variant reproduces the pre-fix
//! behaviour (loom finds a violating interleaving) — so the models have provable
//! teeth. The committed tests run the correct variant; the buggy variant is a
//! live, compiled reproduction reachable by flipping the enum.
//!
//! Run with: `RUSTFLAGS="--cfg loom" cargo test --lib loom_checkpoint_dedupe`
//!
//! Gated behind `cfg(loom)`: loom replaces the std primitives and is
//! incompatible with normal test runs.

#![cfg(loom)]

use loom::sync::{Arc, Mutex};
use loom::thread;

/// Model-check `f` under a bounded number of preemptions.
///
/// Like [`crate::loom_tests`], an unbounded exhaustive search over the two
/// worker threads plus the internal flush/record critical sections is
/// intractable as a CI merge gate, so a preemption bound keeps the search fast
/// and deterministic while still covering the interleavings that expose the
/// ordering/determinism defects. An explicit `LOOM_MAX_PREEMPTIONS` /
/// `LOOM_MAX_BRANCHES` override still wins.
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

// ===========================================================================
// Model 1 — checkpoint never certifies an un-fsynced tile (#273 / #277)
// ===========================================================================

/// Ordering of the durability barrier relative to the completed-tile delta
/// snapshot inside a checkpoint flush.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FlushOrder {
    /// The shipped ordering: snapshot the completed delta first, *then* drain +
    /// fsync the sink's pending set, *then* certify. Every certified coordinate
    /// is guaranteed durable. (`engine.rs::CheckpointState::flush`.)
    SnapshotThenBarrier,
    /// The pre-#273 hazard: drain + fsync the pending set *before* snapshotting
    /// the delta. A concurrent worker that marks a tile between the barrier and
    /// the snapshot lands in the certified delta without ever being fsynced.
    /// Retained as a live reproduction — loom finds the violating interleaving.
    #[allow(dead_code)]
    BarrierThenSnapshot,
}

/// The shared state a checkpoint flush coordinates, mirroring the real split
/// across [`crate::engine::CheckpointState`] (`meta.completed_tiles`,
/// `seg_cursor`) and [`crate::sink::FsSink`] (`unsynced_tiles`). A tile id
/// stands in for a `TileCoord` / tile path.
struct CheckpointModel {
    /// `CheckpointState::meta.completed_tiles` — coordinates marked complete.
    completed: Mutex<Vec<u32>>,
    /// `FsSink::unsynced_tiles` — tile paths written but not yet fsynced.
    unsynced: Mutex<Vec<u32>>,
    /// Tiles whose bytes the durability barrier has fsynced.
    durable: Mutex<Vec<u32>>,
    /// Coordinates the published header has certified as complete.
    certified: Mutex<Vec<u32>>,
    /// `CheckpointState::seg_cursor` — serialises flushes and tracks the
    /// exclusive upper index into `completed` already covered by the log.
    seg_cursor: Mutex<usize>,
}

impl CheckpointModel {
    fn new() -> Self {
        Self {
            completed: Mutex::new(Vec::new()),
            unsynced: Mutex::new(Vec::new()),
            durable: Mutex::new(Vec::new()),
            certified: Mutex::new(Vec::new()),
            seg_cursor: Mutex::new(0),
        }
    }

    /// `FsSink::write_tile` followed by `CheckpointState::mark_tile_completed`:
    /// the tile bytes are written (its path pushed onto the unsynced set)
    /// **before** the coordinate is marked complete. Two independent lock
    /// acquisitions — a flush on another worker can interleave between them.
    fn write_then_mark(&self, tile: u32) {
        // write_tile: track the freshly-written (page-cache-only) tile path.
        self.unsynced.lock().unwrap().push(tile);
        // mark_tile_completed: record the coordinate as complete.
        self.completed.lock().unwrap().push(tile);
    }

    /// `CheckpointState::flush`. Holds `seg_cursor` for the whole flush so two
    /// workers cannot interleave partial segment frames or double-certify a
    /// delta. Under [`FlushOrder::SnapshotThenBarrier`] the completed delta is
    /// snapshotted before the barrier, guaranteeing every certified coordinate
    /// was fsynced first.
    fn flush(&self, order: FlushOrder) {
        let mut cursor = self.seg_cursor.lock().unwrap();

        let snapshot_delta = |cursor: usize| -> (Vec<u32>, usize) {
            let completed = self.completed.lock().unwrap();
            let len = completed.len();
            (completed[cursor.min(len)..len].to_vec(), len)
        };
        // Drain the whole pending set and mark those tiles durable (fsync).
        let barrier = || {
            let drained: Vec<u32> = std::mem::take(&mut *self.unsynced.lock().unwrap());
            let mut durable = self.durable.lock().unwrap();
            for t in drained {
                durable.push(t);
            }
        };

        let (delta, new_cursor) = match order {
            FlushOrder::SnapshotThenBarrier => {
                // Snapshot BEFORE the barrier: every coordinate in `delta` had
                // its path pushed onto `unsynced` (in write_tile, before the
                // mark this snapshot observed), so the subsequent drain fsyncs
                // it. This is the shipped ordering.
                let (delta, len) = snapshot_delta(*cursor);
                barrier();
                (delta, len)
            }
            FlushOrder::BarrierThenSnapshot => {
                // Barrier BEFORE snapshot: a concurrent worker that marks a tile
                // after the drain but before this snapshot slips into `delta`
                // un-fsynced. This is the pre-#273 defect.
                barrier();
                snapshot_delta(*cursor)
            }
        };

        // Certify the delta (append segment + publish header). Every certified
        // coordinate must already be durable — the invariant #273/#277 restore.
        let durable = self.durable.lock().unwrap();
        let mut certified = self.certified.lock().unwrap();
        for tile in &delta {
            assert!(
                durable.contains(tile),
                "checkpoint certified tile {tile} whose bytes were never fsynced \
                 (durable={:?}, delta={delta:?}) — a periodic checkpoint recorded \
                 a tile still only in the page cache (issue #273/#277)",
                *durable
            );
            certified.push(*tile);
        }
        *cursor = new_cursor;
    }
}

/// Drive two workers that each write+mark a distinct tile and then flush,
/// under `order`. Model-checks that no flush ever certifies a tile that was not
/// fsynced first, that the two contending flushes serialise on `seg_cursor`
/// (no coordinate is certified twice), and that every written tile is certified
/// exactly once by the time both workers join.
fn run_checkpoint_model(order: FlushOrder) {
    model(move || {
        let m = Arc::new(CheckpointModel::new());

        let m1 = Arc::clone(&m);
        let w1 = thread::spawn(move || {
            m1.write_then_mark(1);
            m1.flush(order);
        });
        let m2 = Arc::clone(&m);
        let w2 = thread::spawn(move || {
            m2.write_then_mark(2);
            m2.flush(order);
        });

        w1.join().unwrap();
        w2.join().unwrap();

        // A final flush (as `run_pyramid` performs on success) certifies any
        // tail the periodic flushes left. After it, every tile is certified.
        m.flush(order);

        let mut certified = m.certified.lock().unwrap().clone();
        certified.sort_unstable();
        assert_eq!(
            certified,
            vec![1, 2],
            "every written tile must be certified exactly once across the \
             serialised flushes (no loss, no double-certify): got {certified:?}"
        );
    });
}

/// **#273 / #277 exit gate.** Under the shipped flush ordering
/// (snapshot-then-barrier) no interleaving of two concurrent workers +
/// contending flushes ever certifies a tile whose bytes were not fsynced first,
/// and the two flushes serialise cleanly so every tile is certified exactly
/// once.
///
/// To reproduce the pre-#273 defect, pass [`FlushOrder::BarrierThenSnapshot`]
/// here: loom finds an interleaving where a worker marks its tile between the
/// barrier and the delta snapshot, so the certify step trips the durability
/// assertion.
#[test]
fn loom_checkpoint_never_certifies_unsynced_tile() {
    run_checkpoint_model(FlushOrder::SnapshotThenBarrier);
}

// ===========================================================================
// Model 2 — dedupe placement is deterministic (#275)
// ===========================================================================

/// Whether the sink canonicalises dedupe placement at `finish()`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// The shipped behaviour: at `finish()` reassign the full-payload holder of
    /// each duplicated content to its coordinate-minimal occurrence, independent
    /// of arrival order. (`FsSink::canonicalize_dedupe_layout`.)
    Canonicalized,
    /// The pre-#275 hazard: keep whichever occurrence `DedupeIndex::record` saw
    /// first as the full-payload holder — a function of thread scheduling.
    /// Retained as a live reproduction — loom finds an arrival order that
    /// diverges from the canonical placement.
    #[allow(dead_code)]
    ArrivalFirst,
}

/// The dedupe state relevant to placement determinism, modelling one content
/// hash with several occurrences (distinct tile coordinates that share bytes).
/// Mirrors the split in [`crate::dedupe::DedupeIndex`] (`seen`) and
/// [`crate::sink::FsSink`] (`dedupe_groups`, `manifest_refs`, the
/// `dedupe_promote` lock).
struct DedupeModel {
    /// `DedupeIndex::seen` for this content hash: the first occurrence recorded
    /// keeps the full payload at record time (the arrival-order `WriteNew`
    /// holder).
    holder: Mutex<Option<u32>>,
    /// `FsSink::dedupe_groups[hash].occurrences` — every occurrence recorded.
    occurrences: Mutex<Vec<u32>>,
    /// `manifest.json::blank_references` keys — occurrences materialised as
    /// 1-byte placeholders (everything except the full-payload holder).
    refs: Mutex<Vec<u32>>,
    /// `FsSink::dedupe_promote` — serialises the record→register sequence so a
    /// `Reference` never races ahead of the first writer's insert (issue #111).
    promote: Mutex<()>,
}

impl DedupeModel {
    fn new() -> Self {
        Self {
            holder: Mutex::new(None),
            occurrences: Mutex::new(Vec::new()),
            refs: Mutex::new(Vec::new()),
            promote: Mutex::new(()),
        }
    }

    /// `FsSink::dedupe_write` → `DedupeIndex::record`: under the promote lock,
    /// the first occurrence of the content becomes the `WriteNew` holder and the
    /// rest become `Reference` placeholders. Every occurrence is registered in
    /// the group so `finish()` can pick the canonical holder later.
    fn record(&self, coord: u32) {
        let _promote = self.promote.lock().unwrap();
        let mut holder = self.holder.lock().unwrap();
        let is_reference = holder.is_some();
        if holder.is_none() {
            *holder = Some(coord); // WriteNew: this occurrence holds the payload
        }
        self.occurrences.lock().unwrap().push(coord);
        if is_reference {
            self.refs.lock().unwrap().push(coord); // placeholder + manifest ref
        }
    }

    /// `FsSink::canonicalize_dedupe_layout`, run once from `finish()` after all
    /// writers have joined (no concurrency here). Reassigns the full-payload
    /// holder of the duplicated content to its coordinate-minimal occurrence and
    /// rewrites the placeholder set accordingly.
    fn canonicalize(&self) {
        let mut occ = self.occurrences.lock().unwrap().clone();
        if occ.len() < 2 {
            return; // singletons keep their file; nothing to reassign
        }
        occ.sort_unstable(); // coordinate order (level, row, col) proxy
        let target = occ[0];
        let mut holder = self.holder.lock().unwrap();
        let current = holder.expect("a recorded group always has a holder");
        if current == target {
            return; // already canonical
        }
        // Promote the coordinate-minimal occurrence to the full-payload holder
        // and demote the previous holder to a placeholder.
        *holder = Some(target);
        let mut refs = self.refs.lock().unwrap();
        refs.retain(|c| *c != target);
        refs.push(current);
    }

    /// The final placement: `(full_payload_holder, sorted placeholder refs)`.
    fn placement(&self) -> (u32, Vec<u32>) {
        let holder = self.holder.lock().unwrap().expect("group has a holder");
        let mut refs = self.refs.lock().unwrap().clone();
        refs.sort_unstable();
        (holder, refs)
    }
}

/// Record the three occurrences `[coords[0], coords[1], coords[2]]` of one
/// shared content across two workers (so arrival order is scheduler-decided),
/// then finish under `placement`. Returns the resulting placement.
fn record_and_finish(coords: [u32; 3], placement: Placement) -> (u32, Vec<u32>) {
    let d = Arc::new(DedupeModel::new());

    let d1 = Arc::clone(&d);
    let w1 = thread::spawn(move || {
        d1.record(coords[0]);
    });
    let d2 = Arc::clone(&d);
    let w2 = thread::spawn(move || {
        d2.record(coords[1]);
        d2.record(coords[2]);
    });
    w1.join().unwrap();
    w2.join().unwrap();

    if placement == Placement::Canonicalized {
        d.canonicalize();
    }
    d.placement()
}

/// **#275 exit gate.** Three byte-identical tiles at coordinates {1,2,3} are
/// recorded in a scheduler-decided arrival order across two workers; after the
/// canonicalising `finish()` the full-payload holder is always the
/// coordinate-minimal occurrence (1) and the placeholder set is always {2,3},
/// for *every* interleaving loom explores. The on-disk placement is a pure
/// function of content + coordinates.
///
/// To reproduce the pre-#275 defect, pass [`Placement::ArrivalFirst`]: loom
/// explores the interleaving where occurrence 2 or 3 records before occurrence
/// 1, so the arrival-first holder is not the coordinate-minimal one and this
/// assertion fails.
#[test]
fn loom_dedupe_placement_is_deterministic() {
    model(|| {
        let placement = record_and_finish([1, 2, 3], Placement::Canonicalized);
        assert_eq!(
            placement,
            (1, vec![2, 3]),
            "dedupe placement must be coordinate-canonical (holder = min coord, \
             the rest placeholders) regardless of tile arrival order (issue #275)"
        );
    });
}

// ===========================================================================
// Model 3 — resume seeding reconstructs identical placement (#272 + #275)
// ===========================================================================

/// **#272 + #275 exit gate.** A resumed run reconstructs byte-identical dedupe
/// placement to an uninterrupted run, independent of which occurrence held the
/// full payload before the crash and of the seeding order.
///
/// The pre-crash run left occurrence 2 (an arbitrary, arrival-decided holder)
/// as the full-payload tile. On resume, [`crate::sink::TileSink::seed_completed_tile`]
/// replays every already-completed coordinate through the same record path —
/// modelled here as one worker "seeding" the pre-crash occurrence while another
/// records the freshly-rendered post-crash occurrences. After the canonicalising
/// `finish()` the placement must match the clean run's canonical placement
/// (holder = coordinate-minimal occurrence 1, placeholders {2,3}) for every
/// interleaving.
///
/// This is the model form of the crash-resume determinism harness in
/// `libviprs-tests/tests/resume_determinism_harness.rs`: seeding replays the
/// same record path, and canonicalisation makes the final layout
/// order-independent, so a resumed `finish()` is byte-identical to a clean one.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn loom_resume_seeding_reconstructs_identical_placement() {
    // Reference: the uninterrupted run's canonical placement.
    let clean = (1u32, vec![2u32, 3u32]);

    model(move || {
        let d = Arc::new(DedupeModel::new());

        // Worker S: resume seeding of the pre-crash holder (occurrence 2), which
        // replays through the identical record path (seed_completed_tile ->
        // write_tile -> dedupe_write -> record).
        let ds = Arc::clone(&d);
        let seed = thread::spawn(move || {
            ds.record(2);
        });
        // Worker P: the freshly re-rendered post-crash occurrences (1 and 3).
        let dp = Arc::clone(&d);
        let post = thread::spawn(move || {
            dp.record(1);
            dp.record(3);
        });
        seed.join().unwrap();
        post.join().unwrap();

        d.canonicalize();

        assert_eq!(
            d.placement(),
            clean,
            "a resumed run's dedupe placement must be byte-identical to the \
             uninterrupted run's, independent of the pre-crash holder and the \
             seeding order (issues #272 + #275)"
        );
    });
}
