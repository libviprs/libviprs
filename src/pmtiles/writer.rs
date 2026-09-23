//! The streaming, bounded-memory PMTiles v3 writer.
//!
//! [`Writer`] takes tiles in **any order**, stores each distinct payload
//! exactly once, and assembles a spec-correct archive at [`Writer::finish`]
//! with a staged temp file, an `fsync` and an atomic rename.
//!
//! # What the memory actually is
//!
//! Peak memory does not grow with the number of **tiles**: the per-tile index
//! is spilled to an append-only log on disk and externally sorted at finalize,
//! and the directories are built one leaf at a time. It does not grow with the
//! number of **distinct payloads** either, and that half is new. It used to,
//! linearly, which mattered because tiling a large photograph produces
//! essentially no duplicate tiles, so "distinct payloads" and "tiles" are the
//! same figure for this crate's main job.
//!
//! Peak RSS of one process per row, every payload distinct, measured by
//! `one_rss_row` in `tests/pmtiles_bounded_memory.rs` rather than remembered.
//! The right-hand column is what this page used to quote, from the writer
//! before EPIC #1135:
//!
//! | distinct payloads | peak RSS | before the epic |
//! |---|---|---|
//! | 500,000 | 22.8 MB | 79.0 MB |
//! | 1,000,000 | 34.9 MB | 155.4 MB |
//! | 2,000,000 | 35.8 MB | 288.5 MB |
//! | 4,000,000 | 35.9 MB | 563.2 MB |
//! | 10,000,000 | **36.0 MB** | 1,083.8 MB |
//!
//! So a ten-million-tile photograph costs 36 MB rather than about a gigabyte,
//! and the figure stops moving once the sort buffer has filled, at a million.
//! Most of what is left **is** the sort buffer, 24 MB of `Vec` at the default
//! `sort_buffer_records`, and the dedupe window is 8 MB of the rest. Both are
//! numbers the caller sets.
//!
//! Four tables used to scale with the payload count and this page named three
//! of them. The content-hash table went in issue #1137, replaced by a
//! fixed-capacity window the caller sizes. The payload table and the
//! final-offset lookup went in issue #1138, which put the staged offset in the
//! index record itself: the record is the same 20 bytes on disk either way,
//! and an offset answers both questions those two were keeping the answers to.
//! The write order was the fourth, the one the doc never mentioned, and it
//! spills now at twelve bytes a record, read back once while the archive is
//! written.
//!
//! A pyramid of mostly blank tiles has very few distinct payloads, which is
//! the case this design exists for. A photograph is the case the window's
//! budget is chosen for.
//!
//! The lifecycle is [`PackfileSink`](crate::sink_packfile::PackfileSink)'s:
//! open once, feed it, finish once. The atomicity is
//! [`resume::atomic_write`](crate::resume)'s: stage into a sibling, flush it
//! to disk, rename it into place, and never let a partial archive wear the
//! final name.
//!
//! # The layout decision, which the issue asks for twice in two ways
//!
//! Issue #989 asks for the data region in **arrival order** with
//! `clustered = false`, and also for two shuffled insertion orders to produce
//! a **byte-identical archive**. Those cannot both hold: arrival order means
//! the bytes depend on arrival order.
//!
//! This page used to say the tiebreaker was structural: that the root's size
//! is unknown until every entry is sorted, so the sections cannot be laid out
//! until the last tile has arrived, so **the staged payloads get copied at
//! finalize whatever layout is chosen**. That argument is wrong and it is
//! worth saying why, because it made a real choice look like a fact of the
//! format.
//!
//! v3 fixes the position of one thing, the 127-byte header, and requires the
//! root directory to live inside the first 16384 bytes. Every other section
//! may be relocated arbitrarily. And a tile entry's offset is relative to the
//! start of the **tile data section**, not to the file, so a payload's offset
//! is final the moment it is staged: nothing that happens to the sections
//! around it can move it. A writer that reserved the first 16384 bytes,
//! appended payloads straight into the destination behind them and wrote the
//! directories after the tile data would copy nothing at all. That is a real
//! layout and this writer does not offer it yet.
//!
//! What such a writer cannot offer is the two things issue #989 also asks
//! for. Arrival order means the bytes depend on arrival order, so two shuffled
//! insertion orders stop producing a byte-identical archive, and `clustered`
//! stops being true, which `pmtiles extract` requires of its input. Tile id
//! order buys a deterministic archive, an honest `clustered = true`, and a
//! reader whose sequential scan of a zoom level is a sequential scan of the
//! file.
//!
//! So this writer stages payloads in arrival order, sorts at finalize, and
//! writes the data region in tile id order. `clustered` is `true` and it is
//! true. The copy is what that costs, and it is a price rather than an
//! inevitability.
//!
//! # Dedupe is the archive's, not the engine's
//!
//! [`DedupeStrategy`](crate::dedupe::DedupeStrategy) defaults to
//! [`None`](crate::dedupe::DedupeStrategy::None), and under `None`
//! [`DedupeIndex::record`](crate::dedupe::DedupeIndex::record) returns
//! `WriteNew` for **every** call by design: it is a true passthrough and must
//! not collapse anything, because the tile tree it governs is supposed to have
//! one file per tile. A writer that drove its payload table off those
//! decisions would store every duplicate on default settings.
//!
//! Storing one blob per distinct payload is a property of the archive format,
//! not of the engine's blank-tile policy, so it happens here unconditionally
//! and is keyed on the content hash the caller passes in. The engine already
//! computes that hash ([`DedupeIndex::content_digest`](crate::dedupe::DedupeIndex::content_digest)),
//! so nothing is hashed twice; a caller without one can use [`content_hash`].
//!
//! Both shapes of dedupe fall out of it. Identical payloads at **consecutive**
//! tile ids collapse into one entry with a run length, and identical payloads
//! at ids that are **not** consecutive stay separate entries pointing at the
//! same offset. Both are ordinary PMTiles and a reader has to handle each.
//!
//! It reaches as far as the window and no further. The writer remembers the
//! last [`WriterOptions::dedupe_memory_bytes`] worth of payloads, so two
//! identical payloads further apart than that are stored twice. A photograph
//! has no duplicates to miss and a blank tile recurs constantly so it never
//! leaves the window, which is why the default is a number nobody has to think
//! about; a pyramid with many distinct payloads that also repeat at long range
//! is the case that pays, and the option is there for it.
//!
//! # What is bounded and what is not
//!
//! Bounded, and independent of the tile count: the per-tile index (spilled,
//! sorted in fixed-size runs), the entry list (spilled, streamed into leaves),
//! the leaf section (spilled), the write order (spilled), the payload copy
//! (one blob at a time), and the external merge, which reads every run through
//! **one** file descriptor with a capped fan-in rather than holding one open
//! file per run.
//!
//! Bounded, and independent of the payload count: the dedupe window and the
//! repeat table, which are a fixed capacity the caller sizes with
//! [`WriterOptions::dedupe_memory_bytes`].
//!
//! Nothing here grows with either number. The two figures that move are the
//! two the caller chose, and `bound_for` in `tests/pmtiles_bounded_memory.rs`
//! is that sentence written as arithmetic, with a test that fails when it
//! stops being true.
//!
//! # A failed write is never published
//!
//! [`add_tile`](Writer::add_tile) latches the first I/O failure and
//! [`finish`](Writer::finish) refuses from there with
//! [`PmTilesError::WriterFailed`]. A `write_all` that fails has usually
//! written some of its bytes, and without the latch those orphan bytes sit in
//! the staging file unaccounted for, the next accepted payload records an
//! offset pointing into the middle of them, and every later payload is shifted
//! by the same amount. That is a structurally valid archive full of the wrong
//! tile bytes, and under a retrying sink with
//! [`FailurePolicy::RetryThenSkip`](crate::sink::FailurePolicy) the run reports
//! success while producing it. ENOSPC is the realistic trigger, because this
//! writer needs roughly twice the archive's size in scratch.
//!
//! # Examples
//!
//! ```no_run
//! use libviprs::pmtiles::TileType;
//! use libviprs::pmtiles::writer::{Writer, WriterOptions, content_hash};
//!
//! let mut w = Writer::create(
//!     "pyramid.pmtiles",
//!     WriterOptions::default().with_tile_type(TileType::Png),
//! )?;
//! let png: &[u8] = b"\x89PNG...";
//! w.add_tile(0, 0, 0, png, content_hash(png))?;
//! let done = w.finish()?;
//! assert_eq!(done.header.addressed_tiles_count, 1);
//! # Ok::<(), libviprs::pmtiles::PmTilesError>(())
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::pmtiles::directory::serialize_entries;
use crate::pmtiles::header::HEADER_BYTES;
use crate::pmtiles::tileid::zxy_to_tileid;
use crate::pmtiles::{Compression, Entry, Header, Metadata, PmTilesError, TileType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The spec's ceiling on the whole of `header + root directory`, so a
/// latency-sensitive client can fetch both in one request.
const ROOT_CEILING: u64 = 16384;

/// The largest compressed root this writer will emit, given that it always
/// places the root immediately after the header. `ROOT_CEILING - 127`.
const ROOT_BUDGET: usize = (ROOT_CEILING as usize) - HEADER_BYTES;

/// Above this many entries, do not even try to fit them all in the root.
///
/// This is a cutoff, not an arithmetic impossibility, and it is worth being
/// precise about which: 21844 entries of a two-payload pyramid gzip to a few
/// hundred bytes and **would** fit the 16257-byte budget comfortably. What the
/// cutoff buys is that the try itself is bounded, because finding out means
/// holding the whole entry list in memory, which is the one allocation this
/// writer exists to avoid.
///
/// It is also go-pmtiles' cutoff, at the same value, and that is why an
/// archive built here from a given tile set has the same leaf structure as one
/// the reference builds from it rather than a flat root the reference would
/// never produce.
const ROOT_ONLY_MAX_ENTRIES: u64 = 16384;

/// Entries per leaf directory, before the doubling loop in
/// [`build_directories`] widens it.
///
/// 4096 is what go-pmtiles uses as its starting point, and matching it means
/// an archive this writer produces from the same tiles has the same leaf
/// structure as one the reference produces, which is what makes a golden
/// archive a usable target rather than merely a plausible one.
///
/// `pub(crate)` because the reader sizes its leaf cache off it: the count of
/// leaves it holds is the entry budget divided by this, so the two numbers
/// cannot drift apart the way they did in issue #993.
pub(crate) const DEFAULT_LEAF_ENTRIES: usize = 4096;

/// How many spill records are sorted in memory before a run is written out.
///
/// A record is 20 bytes on disk and **24 in memory**, because `Spill` is a
/// `u64`, a `u64` and a `u32` and the compiler pads that to the alignment of
/// its widest field. So 1 Mi records is 20 MB written and 24 MB of `Vec`, and
/// the buffer is the larger of the two. The earlier comment here quoted 20 MB
/// for both and understated the live figure by a fifth.
///
/// It also keeps the run count low, which used to matter a great deal more
/// than it does: the merge held one open file handle per run, so a long enough
/// job ran out of descriptors. It reads through a single handle now
/// ([`SortedSpill`]) and caps its fan-in ([`MAX_MERGE_FANIN`]), so the run
/// count costs merge passes rather than a failure.
const SORT_RUN_RECORDS: usize = 1 << 20;

/// One spilled index record: `tile_id`, staged offset, length.
const SPILL_RECORD_BYTES: usize = 8 + 8 + 4;

/// How many runs one merge pass will read at once.
///
/// The merge used to open every run at once and hold the descriptors for the
/// whole pass. macOS ships a soft `RLIMIT_NOFILE` of 256 and Linux usually
/// 1024, so a job with enough runs died of `EMFILE` inside `finish`, after the
/// entire run, with nothing salvageable because the archive only exists at the
/// rename. Reading through one handle fixed the descriptors; this caps what is
/// left, which is the per-cursor read buffers and the heap, and turns any
/// number of runs into repeated passes instead of one impossible one.
const MAX_MERGE_FANIN: usize = 128;

/// Records one merge cursor buffers at a time.
///
/// 256 records is 5 KB a cursor, so a full-width pass holds 640 KB of buffers
/// however many runs the job produced.
const MERGE_CURSOR_RECORDS: usize = 256;

/// One spilled directory entry: `tile_id`, final offset, length, run length.
const ENTRY_RECORD_BYTES: usize = 8 + 8 + 4 + 4;

/// One spilled write-order record: staged offset, length.
const ORDER_RECORD_BYTES: usize = 8 + 4;

/// Copy buffer for moving staged payloads and leaf bytes into the archive.
const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// Ways per set in the dedupe window.
///
/// Eight, which is one 64-byte cache line's worth of the hashes the lookup
/// compares and the width at which set-associative caches stop gaining much.
/// Public because the size of the window is a contract a caller can reason
/// about: a budget buys a whole number of sets of this many payloads, and the
/// smallest window there is holds exactly this many.
pub const DEDUPE_WINDOW_WAYS: usize = 8;

/// What one tracked payload costs, across the window and the repeat table.
///
/// A [`WindowSlot`] is 48 bytes, a [`RepeatSlot`] is 16, and each table spends
/// four bytes of recency order on every set of [`DEDUPE_WINDOW_WAYS`], which
/// is one more byte a payload between them.
/// `the_dedupe_budget_buys_what_it_says_it_buys` holds this against
/// `size_of`, so it cannot drift from the types it describes.
const DEDUPE_BYTES_PER_PAYLOAD: usize = 65;

/// How much memory the dedupe window spends when the caller says nothing.
///
/// Eight mebibytes, which tracks 129056 payloads. The number is a judgement
/// about which jobs the window should still be exact for rather than an
/// arithmetic result, so here is the judgement. A photograph has no duplicate
/// tiles at all, so any window is the right size for it. A pyramid of mostly
/// blank tiles has a handful of distinct payloads that recur constantly, so
/// they never leave a window of any size. What this number decides is the case
/// in between, a pyramid with more than 129056 distinct payloads that also
/// repeat at long range, and eight mebibytes is a tenth of the sort buffer's
/// default and small enough that no caller has to think about it.
///
/// A caller who knows their input repeats at long range raises it, and pays
/// what they raise it by.
const DEFAULT_DEDUPE_MEMORY_BYTES: usize = 8 * 1024 * 1024;

/// "This payload has not been placed in the data region yet."
///
/// A sentinel rather than an `Option<u64>` because the `Option` would double
/// the table it lives in, and it cannot collide with a real offset: an offset
/// is assigned before its length is added, the addition is checked, and every
/// entry is at least one byte, so a run that assigned `u64::MAX` fails on the
/// next `checked_add` before the value is ever stored.
const UNPLACED: u64 = u64::MAX;

/// Disambiguates the scratch prefix of two writers sharing one directory.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

/// What one `finish` really did, counted, so the tests below can assert on it
/// instead of on the source.
///
/// Three of the claims in EPIC #1135 are about work rather than about output:
/// how many durability barriers a `finish` issues, whether the payload copy
/// seeks per payload, and whether the leaf loop rebuilds every leaf once per
/// doubling. None of them changes a byte of the archive, so nothing that reads
/// the archive can tell whether they hold, and a test that reads the source
/// instead is a test of the source.
///
/// The counters are **thread-local**, because `cargo test` runs this binary's
/// tests in parallel threads of one process and a global counter would be
/// measuring whichever other writer happened to be finishing at the same
/// moment. A writer never leaves the thread that drives it, so a thread-local
/// is exact and needs no lock.
///
/// Everything here compiles to nothing outside `cfg(test)`.
mod probe {
    #[cfg(test)]
    use std::cell::Cell;

    #[cfg(test)]
    thread_local! {
        /// Durability barriers issued on this thread.
        static SYNCS: Cell<usize> = const { Cell::new(0) };
        /// Seeks of the staged payload file during the archive write.
        static STAGED_SEEKS: Cell<usize> = const { Cell::new(0) };
        /// Positioned reads of the staged payload file during the archive
        /// write. Reads through the merge's own handle are not counted: they
        /// are a different file and a different question.
        static STAGED_READS: Cell<usize> = const { Cell::new(0) };
        /// Times the leaf loop built the whole leaf section.
        static LEAF_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
        /// Entries per leaf the loop settled on.
        static LEAF_ENTRIES_USED: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn sync() {
        #[cfg(test)]
        SYNCS.with(|c| c.set(c.get() + 1));
    }

    pub(super) fn staged_seek() {
        #[cfg(test)]
        STAGED_SEEKS.with(|c| c.set(c.get() + 1));
    }

    // No caller until the positioned-read copy in #1142 lands, and the test
    // that counts it is red until then. That is the point of it.
    #[expect(dead_code, reason = "the copy that calls it is issue #1142")]
    pub(super) fn staged_read() {
        #[cfg(test)]
        STAGED_READS.with(|c| c.set(c.get() + 1));
    }

    pub(super) fn leaf_attempt() {
        #[cfg(test)]
        LEAF_ATTEMPTS.with(|c| c.set(c.get() + 1));
    }

    pub(super) fn leaf_entries_used(entries: usize) {
        #[cfg(test)]
        LEAF_ENTRIES_USED.with(|c| c.set(entries));
        #[cfg(not(test))]
        let _ = entries;
    }

    #[cfg(test)]
    pub(super) fn reset() {
        SYNCS.with(|c| c.set(0));
        STAGED_SEEKS.with(|c| c.set(0));
        STAGED_READS.with(|c| c.set(0));
        LEAF_ATTEMPTS.with(|c| c.set(0));
        LEAF_ENTRIES_USED.with(|c| c.set(0));
    }

    #[cfg(test)]
    pub(super) fn syncs() -> usize {
        SYNCS.with(Cell::get)
    }

    #[cfg(test)]
    pub(super) fn staged_seeks() -> usize {
        STAGED_SEEKS.with(Cell::get)
    }

    #[cfg(test)]
    pub(super) fn staged_reads() -> usize {
        STAGED_READS.with(Cell::get)
    }

    #[cfg(test)]
    pub(super) fn leaf_attempts() -> usize {
        LEAF_ATTEMPTS.with(Cell::get)
    }

    #[cfg(test)]
    pub(super) fn leaf_entries() -> usize {
        LEAF_ENTRIES_USED.with(Cell::get)
    }
}

/// `sync_data`, counted.
fn sync_data(file: &File) -> std::io::Result<()> {
    probe::sync();
    file.sync_data()
}

/// `sync_all`, counted. This is the one before the rename.
fn sync_all(file: &File) -> std::io::Result<()> {
    probe::sync();
    file.sync_all()
}

// ---------------------------------------------------------------------------
// The dedupe window
// ---------------------------------------------------------------------------

/// One payload the window still remembers.
///
/// `length` doubles as the occupancy flag: [`Writer::add_tile`] refuses an
/// empty payload, twice, because the spec forbids one, so no live entry can
/// ever wear a zero here and the alternative is a byte of padding in a 48-byte
/// slot that is already exactly six words.
#[derive(Clone, Copy)]
struct WindowSlot {
    hash: [u8; 32],
    data_offset: u64,
    length: u32,
}

impl WindowSlot {
    const EMPTY: Self = Self {
        hash: [0; 32],
        data_offset: 0,
        length: 0,
    };

    fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// A set's ways, most recently used first, one nibble each.
///
/// Eight ways of three bits would fit in 24 bits, but nibbles make the shift
/// arithmetic a shift by four and the whole thing fits a `u32` either way.
/// `0x7654_3210` is the identity, so an untouched set evicts way 7 first and
/// fills from the top down, which is why [`DedupeWindow::insert`] can take the
/// eviction candidate and the first free slot from the same end.
const RECENCY_IDENTITY: u32 = 0x7654_3210;

/// Move `way` to the front of `order`, leaving the rest in their relative
/// positions.
fn touch(order: u32, way: usize) -> u32 {
    let way = way as u32;
    let mut out = way;
    let mut shift = 4;
    for slot in 0..DEDUPE_WINDOW_WAYS {
        let other = (order >> (4 * slot)) & 0xF;
        if other != way {
            out |= other << shift;
            shift += 4;
        }
    }
    out
}

/// The way `order` says has gone longest without a hit.
fn least_recent(order: u32) -> usize {
    ((order >> (4 * (DEDUPE_WINDOW_WAYS - 1))) & 0xF) as usize
}

/// The payloads this writer can still deduplicate against.
///
/// Fixed capacity, [`DEDUPE_WINDOW_WAYS`]-way set associative, LRU inside the
/// set. It replaced a `HashMap<[u8; 32], u64>` holding every payload the
/// writer had ever seen, which was the whole of its remaining unbounded growth
/// and, at ten million distinct payloads, the largest single allocation in the
/// process (issue #1137).
///
/// What that costs is exactness at long range: two identical payloads far
/// enough apart are stored twice now. What it buys, besides the bound, is
/// speed. There is no rehash and no doubling transient, and a lookup touches
/// one cache line instead of chasing a table of hundreds of megabytes.
struct DedupeWindow {
    slots: Box<[WindowSlot]>,
    recency: Box<[u32]>,
    sets: usize,
}

impl DedupeWindow {
    fn with_sets(sets: usize) -> Self {
        Self {
            slots: vec![WindowSlot::EMPTY; sets * DEDUPE_WINDOW_WAYS].into_boxed_slice(),
            recency: vec![RECENCY_IDENTITY; sets].into_boxed_slice(),
            sets,
        }
    }

    /// Which set a hash lands in.
    ///
    /// Multiply-shift rather than a mask, so the set count is whatever the
    /// budget bought instead of the largest power of two under it. Half a
    /// window is a real cost and this is two instructions.
    fn set_of(&self, hash: &[u8; 32]) -> usize {
        let key = u64::from_le_bytes(hash[..8].try_into().expect("8 bytes"));
        ((u128::from(key) * self.sets as u128) >> 64) as usize
    }

    /// Where this hash was last staged, and how long it is, if the window
    /// still holds it.
    fn get(&mut self, hash: &[u8; 32]) -> Option<(u64, u32)> {
        let set = self.set_of(hash);
        let base = set * DEDUPE_WINDOW_WAYS;
        for way in 0..DEDUPE_WINDOW_WAYS {
            let slot = self.slots[base + way];
            if !slot.is_empty() && slot.hash == *hash {
                self.recency[set] = touch(self.recency[set], way);
                return Some((slot.data_offset, slot.length));
            }
        }
        None
    }

    /// Remember a payload, evicting the set's least recently used entry if
    /// every way is taken.
    fn insert(&mut self, hash: [u8; 32], data_offset: u64, length: u32) {
        let set = self.set_of(&hash);
        let base = set * DEDUPE_WINDOW_WAYS;
        let way = (0..DEDUPE_WINDOW_WAYS)
            .find(|way| self.slots[base + way].is_empty())
            .unwrap_or_else(|| least_recent(self.recency[set]));
        self.slots[base + way] = WindowSlot {
            hash,
            data_offset,
            length,
        };
        self.recency[set] = touch(self.recency[set], way);
    }
}

/// One staged offset a second tile has already pointed at.
///
/// `staged` is the key and [`NO_STAGED_OFFSET`] marks the slot empty: a real
/// staged offset cannot be `u64::MAX`, because the payload at it would have to
/// end past the end of a `u64` and the `checked_add` in
/// [`Writer::add_tile`] refuses that before the offset is ever stored.
#[derive(Clone, Copy)]
struct RepeatSlot {
    staged: u64,
    placed: u64,
}

/// "No payload is staged here."
const NO_STAGED_OFFSET: u64 = u64::MAX;

/// The staged offsets that more than one tile points at.
///
/// Finalization needs to know, for a payload it is about to place, whether it
/// has placed that payload already. Asking that of every payload means a table
/// with an entry per payload, which is the growth this epic exists to remove.
/// So `add_tile` marks an offset here the moment a second tile points at it,
/// and finalization only has to look the marked ones up: a record whose offset
/// is absent was referenced exactly once, so it cannot already have been
/// placed (issue #1137, consumed by #1138).
///
/// It is capped like the window, and for the same reason. An offset that falls
/// out of it is stored twice rather than shared, which is the same trade the
/// window makes and produces the same ordinary archive.
struct RepeatTable {
    slots: Box<[RepeatSlot]>,
    recency: Box<[u32]>,
    sets: usize,
}

impl RepeatTable {
    fn with_sets(sets: usize) -> Self {
        Self {
            slots: vec![
                RepeatSlot {
                    staged: NO_STAGED_OFFSET,
                    placed: UNPLACED,
                };
                sets * DEDUPE_WINDOW_WAYS
            ]
            .into_boxed_slice(),
            recency: vec![RECENCY_IDENTITY; sets].into_boxed_slice(),
            sets,
        }
    }

    /// Which set an offset lands in.
    ///
    /// Staged offsets are dense and ascending, so they are their own worst
    /// hash: the low bits repeat with the payload size. The multiply spreads
    /// them before the reduction does its work.
    fn set_of(&self, staged: u64) -> usize {
        let key = staged.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((u128::from(key) * self.sets as u128) >> 64) as usize
    }

    /// Record that more than one tile points at this offset.
    fn mark(&mut self, staged: u64) {
        let set = self.set_of(staged);
        let base = set * DEDUPE_WINDOW_WAYS;
        for way in 0..DEDUPE_WINDOW_WAYS {
            if self.slots[base + way].staged == staged {
                self.recency[set] = touch(self.recency[set], way);
                return;
            }
        }
        let way = (0..DEDUPE_WINDOW_WAYS)
            .find(|way| self.slots[base + way].staged == NO_STAGED_OFFSET)
            .unwrap_or_else(|| least_recent(self.recency[set]));
        self.slots[base + way] = RepeatSlot {
            staged,
            placed: UNPLACED,
        };
        self.recency[set] = touch(self.recency[set], way);
    }

    /// Where a marked offset was placed in the data region, as a slot to read
    /// or fill in. [`UNPLACED`] means it is marked but has not been reached
    /// yet; `None` means nothing ever marked it, so exactly one tile points at
    /// it and it cannot have been placed.
    ///
    /// It inserts nothing and evicts nothing, which is the property
    /// finalization needs: the table is whatever the add phase left, so
    /// `finish` allocates not one byte per payload, and it cannot lose a
    /// placement it has already made.
    fn placement(&mut self, staged: u64) -> Option<&mut u64> {
        let set = self.set_of(staged);
        let base = set * DEDUPE_WINDOW_WAYS;
        let way = (0..DEDUPE_WINDOW_WAYS).find(|way| self.slots[base + way].staged == staged)?;
        Some(&mut self.slots[base + way].placed)
    }
}

/// How many sets a deduplication budget buys.
///
/// At least one, whatever the budget. A window of no ways would store every
/// duplicate twice, which is a writer nobody asked for, and the floor makes
/// `with_dedupe_memory_bytes(0)` mean "the smallest window there is" rather
/// than "no window".
fn dedupe_sets(budget: usize) -> usize {
    (budget / (DEDUPE_WINDOW_WAYS * DEDUPE_BYTES_PER_PAYLOAD)).max(1)
}

// ---------------------------------------------------------------------------
// content_hash
// ---------------------------------------------------------------------------

/// The content hash [`Writer::add_tile`] keys its payload table on, for a
/// caller that does not already have one.
///
/// Blake3, matching what [`DedupeIndex`](crate::dedupe::DedupeIndex) computes
/// for every tile the engine produces. A caller driving this from the engine
/// should pass that digest straight through
/// ([`DedupeIndex::content_digest`](crate::dedupe::DedupeIndex::content_digest))
/// rather than hashing the same bytes a second time.
///
/// The writer never re-derives the hash from the payload, so the only thing
/// that matters is that the same bytes always arrive with the same hash and
/// different bytes do not. Any 32-byte digest with those properties works,
/// which is why the parameter is a plain `[u8; 32]` and not a type that pins
/// an algorithm.
pub fn content_hash(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

// ---------------------------------------------------------------------------
// WriterOptions
// ---------------------------------------------------------------------------

/// What a [`Writer`] needs to know that it cannot work out from the tiles.
///
/// Everything here has a defensible default, so the common call is
/// `WriterOptions::default().with_tile_type(...)`. The one field with no
/// honest default is the tile type, and `Unknown` is a legal value meaning
/// exactly that: the writer did not say.
///
/// # The bounds have no "unspecified"
///
/// PMTiles gives the three counts a `0`-means-unknown sentinel and gives the
/// bounds, the centre and the zoom range nothing of the kind. All-zero bounds
/// are a degenerate box at (0, 0), not "I did not compute them". So the
/// default here is the whole Web Mercator extent rather than zero, which is
/// the honest answer for a pyramid that covers whatever it covers, and a
/// caller who knows better sets it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WriterOptions {
    /// What the tile blobs are. `Unknown` says the writer did not know.
    pub tile_type: TileType,
    /// How the tile blobs are compressed. The blobs are stored exactly as
    /// they are handed to [`Writer::add_tile`], so this field describes them
    /// rather than instructing the writer: a PNG is already compressed and is
    /// stored with `None`.
    pub tile_compression: Compression,
    /// How the root, the metadata and every leaf are compressed. Gzip is the
    /// only value with an implementation here and the only one anything else
    /// reliably reads.
    pub internal_compression: Compression,
    /// The JSON metadata object. Defaults to `{}`, which is the minimum a
    /// conformant archive may carry.
    pub metadata: Metadata,
    /// West, south, east, north, in degrees. Defaults to the whole Web
    /// Mercator extent.
    pub bounds_degrees: [f64; 4],
    /// Centre longitude and latitude, in degrees. Purely advisory.
    pub center_degrees: (f64, f64),
    /// Zoom a viewer may open at, or `None` to take the midpoint of the zoom
    /// range the tiles turned out to cover. Purely advisory.
    pub center_zoom: Option<u8>,
    /// Entries per leaf directory, before the doubling loop widens it to make
    /// the root fit. Lower it to make a leaf cheaper to fetch, raise it to
    /// make the root smaller.
    pub leaf_entries: usize,
    /// How many index records are sorted in memory before a sorted run is
    /// written out to the log.
    ///
    /// This is the writer's memory ceiling for the sort, at 20 bytes a record,
    /// and it is an option rather than a constant so a test can reach the
    /// external merge without writing a million tiles. A bounded-memory claim
    /// that only an unreachable constant can exercise is a claim nothing
    /// checks.
    pub sort_buffer_records: usize,
    /// How much memory the writer may spend remembering which payloads it has
    /// already staged.
    ///
    /// Not honoured yet: this is the budget the fixed-capacity dedupe window
    /// spends, and the window itself is issue #1137. The field is here first
    /// because the tests that prove both edges of that window have to be able
    /// to ask for a window small enough to have edges.
    pub dedupe_memory_bytes: usize,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            tile_type: TileType::Unknown,
            tile_compression: Compression::None,
            internal_compression: Compression::Gzip,
            metadata: Metadata::default(),
            bounds_degrees: [-180.0, -85.051_128_7, 180.0, 85.051_128_7],
            center_degrees: (0.0, 0.0),
            center_zoom: None,
            leaf_entries: DEFAULT_LEAF_ENTRIES,
            sort_buffer_records: SORT_RUN_RECORDS,
            dedupe_memory_bytes: DEFAULT_DEDUPE_MEMORY_BYTES,
        }
    }
}

impl WriterOptions {
    /// Set what the tile blobs are.
    pub fn with_tile_type(mut self, tile_type: TileType) -> Self {
        self.tile_type = tile_type;
        self
    }

    /// Set how the tile blobs are already compressed.
    pub fn with_tile_compression(mut self, compression: Compression) -> Self {
        self.tile_compression = compression;
        self
    }

    /// Set how the root, the metadata and the leaves are compressed.
    pub fn with_internal_compression(mut self, compression: Compression) -> Self {
        self.internal_compression = compression;
        self
    }

    /// Set the JSON metadata object.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the bounds, as west, south, east, north in degrees.
    pub fn with_bounds_degrees(mut self, bounds: [f64; 4]) -> Self {
        self.bounds_degrees = bounds;
        self
    }

    /// Set the advisory centre, in degrees.
    pub fn with_center_degrees(mut self, lon: f64, lat: f64) -> Self {
        self.center_degrees = (lon, lat);
        self
    }

    /// Set the advisory opening zoom. `None` takes the midpoint of the zoom
    /// range the tiles cover.
    pub fn with_center_zoom(mut self, zoom: Option<u8>) -> Self {
        self.center_zoom = zoom;
        self
    }

    /// Set how many entries a leaf directory starts out holding.
    pub fn with_leaf_entries(mut self, entries: usize) -> Self {
        self.leaf_entries = entries;
        self
    }

    /// Set how many index records are sorted in memory before a run is
    /// spilled.
    pub fn with_sort_buffer_records(mut self, records: usize) -> Self {
        self.sort_buffer_records = records;
        self
    }

    /// Set how much memory the dedupe window may spend.
    pub fn with_dedupe_memory_bytes(mut self, bytes: usize) -> Self {
        self.dedupe_memory_bytes = bytes;
        self
    }
}

// ---------------------------------------------------------------------------
// Finish
// ---------------------------------------------------------------------------

/// What [`Writer::finish`] hands back.
///
/// The issue sketches `finish() -> Result<PathBuf>`, which only answers for
/// the path-based flavour. Returning the header as well costs nothing and
/// gives the caller the three counts, the zoom range and the section offsets
/// without reopening the file it just wrote, which is what the sink in #990
/// needs for its manifest.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Finish {
    /// The 127-byte header exactly as it was written.
    pub header: Header,
    /// Where the archive was published, for a writer made by
    /// [`Writer::create`]. `None` for one made by [`Writer::try_new`], which
    /// wrote into a sink the caller owns and has no path to publish to.
    pub path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Where the assembled archive goes.
///
/// Two cases rather than one generic, because only the staged case can
/// `sync_all` and rename, and `W: Write + Seek` cannot express either.
enum Sink<W> {
    /// A sink the caller owns. Nothing is published; the archive is written
    /// where the sink is pointing.
    Foreign(W),
    /// A temp file this writer created and will rename into place.
    Staged(File),
}

impl<W: Write + Seek> Sink<W> {
    fn as_write(&mut self) -> &mut dyn Write {
        match self {
            Self::Foreign(w) => w,
            Self::Staged(f) => f,
        }
    }
}

/// Where payloads are staged, and where the index log is appended.
///
/// One real variant and one that only exists under `cfg(test)`. The failure
/// this writer has to survive is a `write_all` that writes some of its bytes
/// and then errors, and there is no portable way to make a real `File` do that
/// on demand: ENOSPC is the production trigger and a test cannot fill a
/// filesystem. The test variant writes its accepted prefix into the same real
/// file, so the orphan bytes land on disk exactly as they would in the field
/// and the test can measure them.
enum Staging {
    Real(BufWriter<File>),
    #[cfg(test)]
    FailsAfter(tests::FailAfter),
}

impl Staging {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Real(w) => w.write_all(bytes),
            #[cfg(test)]
            Self::FailsAfter(w) => w.write_all(bytes),
        }
    }

    /// Push everything written so far through to the device, without closing.
    ///
    /// The barrier lives on `Staging` rather than at the call site so a caller
    /// does not have to know which variant it is holding. This arrived with
    /// `sync_pending` rather than with `Staging` itself, deliberately: a
    /// capability with no caller is `dead_code` under `-D warnings`, and the two
    /// halves were written on separate branches, which is how `sync_pending`
    /// came to reach past this type for `flush` and `get_ref` and only failed
    /// once the branches met.
    fn sync(&mut self) -> std::io::Result<()> {
        match self {
            Self::Real(w) => {
                w.flush()?;
                sync_data(w.get_ref())
            }
            #[cfg(test)]
            Self::FailsAfter(w) => w.sync(),
        }
    }

    /// Push everything through to the device and close.
    fn finish(self) -> std::io::Result<()> {
        match self {
            Self::Real(w) => sync_data(&w.into_inner().map_err(|e| e.into_error())?),
            #[cfg(test)]
            Self::FailsAfter(w) => w.finish(),
        }
    }
}

/// One record in the append-only index log: which tile, where its payload is
/// staged, and how long it is.
///
/// `data_offset` used to be a dense payload index, which meant the writer had
/// to keep a `payload_starts: Vec<u64>` to turn that index back into an offset
/// and a length, and a `final_offsets: Vec<u64>` to translate it again at
/// placement. Both of those grew one entry per distinct payload. The offset is
/// the same eight bytes as the index, so the record is the same 20 on disk and
/// the same 24 in memory, and the two vectors are simply gone (issue #1138).
///
/// It also makes placement cheaper rather than dearer. A record whose offset
/// is absent from [`RepeatTable`] is referenced exactly once, so it cannot
/// already have been placed: assign the next offset and move on, with no
/// lookup at all. That is every tile of a photograph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Spill {
    tile_id: u64,
    data_offset: u64,
    length: u32,
}

impl Spill {
    fn encode(&self) -> [u8; SPILL_RECORD_BYTES] {
        let mut out = [0u8; SPILL_RECORD_BYTES];
        out[0..8].copy_from_slice(&self.tile_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.data_offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.length.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8; SPILL_RECORD_BYTES]) -> Self {
        Self {
            tile_id: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
            data_offset: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
            length: u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")),
        }
    }
}

/// One sorted run inside the index log.
#[derive(Debug, Clone, Copy)]
struct Run {
    /// Byte offset of the run's first record in the log.
    start: u64,
    /// How many records the run holds.
    count: u64,
}

/// A streaming, bounded-memory PMTiles v3 writer.
///
/// See the [module docs](self) for the layout decision and the memory bounds.
pub struct Writer<W: Write + Seek> {
    sink: Option<Sink<W>>,
    options: WriterOptions,

    /// Scratch paths this writer created and must remove, whatever happens.
    scratch: Vec<PathBuf>,
    /// Common prefix of every scratch path, and of the staged archive.
    base: PathBuf,
    /// Where the archive is published on a successful finish.
    destination: Option<PathBuf>,

    /// Staged payloads, appended in arrival order.
    staged: Option<Staging>,
    /// The append-only index log.
    log: Option<Staging>,
    log_len: u64,

    /// The payloads this writer can still deduplicate against.
    ///
    /// Fixed capacity: see [`DedupeWindow`]. It used to be a
    /// `HashMap<[u8; 32], u64>` of every payload ever seen, which is the
    /// allocation issue #1137 removed.
    window: Option<DedupeWindow>,
    /// Staged offsets that more than one tile points at, so finalization can
    /// tell a payload it has placed already from one it has not.
    repeats: RepeatTable,
    /// How many bytes of payload are staged, which is where the next one
    /// goes.
    staged_len: u64,
    /// How many payloads have been staged.
    staged_payloads: u64,
    /// Records not yet written to a run.
    sort_buffer: Vec<Spill>,
    /// Runs already written to the log.
    runs: Vec<Run>,

    /// The step a write failed at, if one has. Set once and never cleared: see
    /// [`PmTilesError::WriterFailed`].
    failed: Option<&'static str>,

    tile_count: u64,
    min_zoom: u8,
    max_zoom: u8,
}

impl<W: Write + Seek> std::fmt::Debug for Writer<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("base", &self.base)
            .field("destination", &self.destination)
            .field("tiles", &self.tile_count)
            .field("staged_payloads", &self.staged_payloads)
            .field("staged_bytes", &self.staged_len)
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}

impl Writer<File> {
    /// Create an archive at `path`, staging into `<path>.tmp` and its
    /// siblings.
    ///
    /// Nothing appears at `path` until [`finish`](Writer::finish) succeeds. A
    /// run that is interrupted, that fails, or that is dropped without
    /// finishing leaves the destination untouched, which is the whole point:
    /// a partial archive must never wear the name a complete one would.
    ///
    /// One destination has one writer. Two writers aimed at the same `path`
    /// share the same `<path>.tmp*` staging names and will corrupt each
    /// other's, the same contract [`PackfileSink`](crate::sink_packfile::PackfileSink)
    /// has for its own output file.
    pub fn create(path: impl AsRef<Path>, options: WriterOptions) -> Result<Self, PmTilesError> {
        let destination = path.as_ref().to_path_buf();
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut base = destination.clone().into_os_string();
        base.push(".tmp");
        let base = PathBuf::from(base);

        let mut writer = Self::open_scratch(base, options)?;
        writer.destination = Some(destination);
        // The staged archive itself is created at finish, not now: an
        // interrupted run should leave the index and the payloads, which are
        // evidence that it was working, and not an empty file that looks like
        // a half-written archive.
        writer.sink = None;
        Ok(writer)
    }
}

impl<W: Write + Seek> Writer<W> {
    /// Write an archive into a sink the caller owns, using `scratch_dir` for
    /// the index log and the staged payloads.
    ///
    /// The archive starts at the sink's current position and every offset in
    /// its header is relative to that, as the spec requires: offsets are
    /// "relative to the first byte of the archive", which is not necessarily
    /// the first byte of whatever the archive is embedded in.
    ///
    /// Nothing is published, because there is nothing to publish to. Use
    /// [`Writer::create`] for the atomic-rename flavour.
    pub fn try_new(
        out: W,
        scratch_dir: impl AsRef<Path>,
        options: WriterOptions,
    ) -> Result<Self, PmTilesError> {
        let dir = scratch_dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
        let base = dir.join(format!("pmtiles-{}-{seq}.tmp", std::process::id()));
        let mut writer = Self::open_scratch(base, options)?;
        writer.sink = Some(Sink::Foreign(out));
        Ok(writer)
    }

    /// Open the two scratch files every flavour needs.
    fn open_scratch(base: PathBuf, options: WriterOptions) -> Result<Self, PmTilesError> {
        let sets = dedupe_sets(options.dedupe_memory_bytes);
        let data_path = suffixed(&base, ".data");
        let log_path = suffixed(&base, ".idx");
        let staged = File::create(&data_path)?;
        let log = File::create(&log_path)?;
        Ok(Self {
            sink: None,
            options,
            scratch: vec![data_path, log_path],
            base,
            destination: None,
            staged: Some(Staging::Real(BufWriter::new(staged))),
            log: Some(Staging::Real(BufWriter::new(log))),
            log_len: 0,
            window: Some(DedupeWindow::with_sets(sets)),
            repeats: RepeatTable::with_sets(sets),
            staged_len: 0,
            staged_payloads: 0,
            sort_buffer: Vec::new(),
            runs: Vec::new(),
            failed: None,
            tile_count: 0,
            min_zoom: u8::MAX,
            max_zoom: 0,
        })
    }

    /// Add one tile.
    ///
    /// `content_hash` is the digest of `bytes`, which the engine has already
    /// computed. The writer trusts it: two calls carrying the same hash are
    /// two references to one payload and only the first is stored. It is never
    /// re-derived, which is the point, so a caller that hands in a hash of
    /// something else gets an archive whose tiles are not the ones it passed.
    ///
    /// Refuses a coordinate outside its own zoom's grid rather than masking it
    /// into a different, valid tile the way the reference implementation does,
    /// and refuses an empty payload, which the spec forbids twice.
    ///
    /// Adding the same `(z, x, y)` twice is refused, at
    /// [`finish`](Writer::finish) rather than here: the duplicate is only
    /// visible once the ids are in order, and finding it earlier would mean
    /// keeping every id in memory, which is the thing this writer exists not
    /// to do.
    pub fn add_tile(
        &mut self,
        z: u8,
        x: u32,
        y: u32,
        bytes: &[u8],
        content_hash: [u8; 32],
    ) -> Result<(), PmTilesError> {
        if let Some(during) = self.failed {
            return Err(PmTilesError::WriterFailed { during });
        }
        let tile_id = zxy_to_tileid(z, x, y)?;
        if bytes.is_empty() {
            return Err(PmTilesError::ZeroLengthEntry {
                index: self.tile_count as usize,
            });
        }
        let length = u32::try_from(bytes.len()).map_err(|_| PmTilesError::EntryFieldTooLarge {
            index: self.tile_count as usize,
            field: "length",
            value: bytes.len() as u64,
        })?;

        let staged = self
            .window
            .as_mut()
            .expect("a live writer has its dedupe window")
            .get(&content_hash);
        let data_offset = match staged {
            Some((offset, stored)) => {
                if stored != length {
                    return Err(PmTilesError::ContentHashMismatch { length, stored });
                }
                // A second tile is pointing at this payload, so finalization
                // will have to recognise it the second time it reaches it.
                // Nothing is marked for a payload only one tile ever names,
                // which is every tile of a photograph.
                self.repeats.mark(offset);
                offset
            }
            None => {
                let offset = self.staged_len;
                // Everything from here down can leave bytes behind, so it is
                // latched.
                let written = self
                    .staged
                    .as_mut()
                    .expect("a live writer has its staging file")
                    .write_all(bytes);
                self.latch("staging a payload", written)?;
                let end = offset
                    .checked_add(u64::from(length))
                    .ok_or(PmTilesError::Overflow {
                        what: "the staged payload region",
                    });
                let end = self.latch("staging a payload", end)?;
                self.staged_len = end;
                self.staged_payloads += 1;
                self.window
                    .as_mut()
                    .expect("a live writer has its dedupe window")
                    .insert(content_hash, offset, length);
                offset
            }
        };

        // `flush_run` latches its own write, so this is only a `?`.
        self.push_spill(Spill {
            tile_id,
            data_offset,
            length,
        })?;

        self.tile_count += 1;
        self.min_zoom = self.min_zoom.min(z);
        self.max_zoom = self.max_zoom.max(z);
        Ok(())
    }

    /// Record that a step failed after it could have written something.
    ///
    /// Called on the I/O paths of [`add_tile`](Self::add_tile) and nowhere
    /// else, on purpose. The refusals in front of them, a coordinate outside
    /// its own grid, an empty payload, a content hash covering a different
    /// length, all happen before a byte moves, and latching those would turn
    /// an engine's `FailurePolicy::Skip` into a failed run.
    fn latch<T, E: Into<PmTilesError>>(
        &mut self,
        during: &'static str,
        result: Result<T, E>,
    ) -> Result<T, PmTilesError> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.failed.get_or_insert(during);
                Err(error.into())
            }
        }
    }

    /// How many tiles have been added so far. Runs are not collapsed until
    /// [`finish`](Writer::finish), so this counts addressed tiles.
    pub fn tile_count(&self) -> u64 {
        self.tile_count
    }

    /// How many payloads are staged so far.
    ///
    /// One per distinct payload the dedupe window caught. Two identical
    /// payloads far enough apart that the window forgot the first are two
    /// payloads here, which is the trade the window makes (see
    /// [`WriterOptions::dedupe_memory_bytes`]) and the reason this is no
    /// longer called a count of *distinct* payloads.
    pub fn distinct_payload_count(&self) -> usize {
        self.staged_payloads as usize
    }

    /// How many sorted runs have been spilled to the index log so far.
    ///
    /// Public because it is the only way to tell a real external merge from an
    /// in-memory sort that happened to fit, and a bounded-memory test with no
    /// way to check that it exercised the merge is testing the easy path and
    /// reporting the hard one.
    pub fn spilled_run_count(&self) -> usize {
        self.runs.len()
    }

    /// Make every tile accepted so far durable, without finalising anything.
    ///
    /// The durability barrier a checkpointed engine run needs
    /// ([`TileSink::sync_pending`](crate::sink::TileSink::sync_pending)): the
    /// records still in the sort buffer are appended to the index log as a
    /// run, both scratch files are pushed through their buffers, and both are
    /// `sync_data`d. After it returns, every `add_tile` that has been accepted
    /// has its payload and its index record on stable storage.
    ///
    /// It does **not** make an archive appear. Nothing exists at the
    /// destination until [`finish`](Writer::finish) renames it there, by
    /// design, so what this buys a crashed run is that its staging is intact
    /// and not that its output is half usable. A single-file archive has no
    /// intermediate state a reader could open, which is the whole reason the
    /// destination stays untouched until the end.
    ///
    /// Flushing the sort buffer as a run costs nothing: the external merge
    /// takes any number of sorted runs, so a barrier that lands mid-buffer
    /// produces a shorter run and no other difference.
    pub fn sync_pending(&mut self) -> Result<(), PmTilesError> {
        self.flush_run()?;
        if let Some(staged) = self.staged.as_mut() {
            staged.sync()?;
        }
        if let Some(log) = self.log.as_mut() {
            log.sync()?;
        }
        Ok(())
    }

    fn push_spill(&mut self, record: Spill) -> Result<(), PmTilesError> {
        self.sort_buffer.push(record);
        if self.sort_buffer.len() >= self.options.sort_buffer_records.max(1) {
            self.flush_run()?;
        }
        Ok(())
    }

    /// Sort what is in memory and append it to the log as one run.
    ///
    /// The write is latched here rather than at the call site, because this is
    /// reached from more than one place and a half-written run leaves the log
    /// in the same shape a half-written payload leaves the staging file: bytes
    /// nothing accounts for, which the next run would be recorded after. The
    /// durability barrier F1.4's sink calls (`sync_pending`, issue 990) lands
    /// on this method, so it inherits the latch rather than having to remember
    /// it.
    fn flush_run(&mut self) -> Result<(), PmTilesError> {
        if self.sort_buffer.is_empty() {
            return Ok(());
        }
        self.sort_buffer.sort_unstable();
        let start = self.log_len;
        let count = self.sort_buffer.len() as u64;
        let written = {
            let Self {
                log, sort_buffer, ..
            } = self;
            let log = log.as_mut().expect("a live writer has its log");
            sort_buffer
                .iter()
                .try_for_each(|record| log.write_all(&record.encode()))
        };
        self.latch("appending to the index log", written)?;
        self.log_len += count * SPILL_RECORD_BYTES as u64;
        self.runs.push(Run { start, count });
        self.sort_buffer.clear();
        Ok(())
    }

    /// Assemble the archive and publish it.
    ///
    /// Returns the header that was written and, for a writer made by
    /// [`Writer::create`], the path it was published to.
    ///
    /// An archive with no tiles is refused: every directory `MUST` hold more
    /// than zero entries and a conformant archive has a root directory, so
    /// there is no legal encoding of an empty one, and writing something
    /// non-conformant rather than failing would push the problem onto whoever
    /// tried to open it.
    pub fn finish(mut self) -> Result<Finish, PmTilesError> {
        // `Drop` does the cleanup whichever way this goes, so the body works
        // through `&mut self` and never moves a field out.
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<Finish, PmTilesError> {
        // A run that could not write a tile does not get to publish one.
        //
        // `add_tile` advances its bookkeeping only after `write_all` returns,
        // so a write that fails part way leaves orphan bytes in the staging
        // file that nothing accounts for. The next accepted payload would be
        // recorded at an offset pointing into the middle of them and every
        // later payload would be shifted by the same amount, which is a
        // structurally perfect archive full of the wrong tile bytes. A
        // retrying sink under `FailurePolicy::RetryThenSkip` turns that into a
        // *successful* run, so the refusal has to be here rather than left to
        // the caller noticing an error it was told to tolerate.
        if let Some(during) = self.failed {
            return Err(PmTilesError::WriterFailed { during });
        }

        // There is deliberately no early "did you add any tiles" check here.
        // A zero-tile archive is refused by `serialize_entries`, which is
        // where the spec's "MUST be greater than 0" lives, and a second
        // refusal in front of it would be a guard no test can distinguish
        // from the real one: removing it changes no observable behaviour,
        // which makes it exactly the kind of code that rots unnoticed.
        self.flush_run()?;

        // The window has no reader left. `add_tile` is the only thing that
        // ever looked at it, so from here it is the budget the caller gave us
        // sitting resident through the whole of finalize for nothing, which is
        // exactly where the old writer took its peak. The repeat table stays:
        // placement reads it.
        self.window = None;

        // Both scratch files are done being written. Flush them through their
        // buffers and close the handles before anything reads them back.
        if let Some(staged) = self.staged.take() {
            staged.finish()?;
        }
        if let Some(log) = self.log.take() {
            log.finish()?;
        }

        let plan = self.plan_entries()?;
        let (root, leaves) = self.build_directories(&plan)?;
        let metadata = self
            .options
            .internal_compression
            .compress(&self.options.metadata.to_json()?)?;

        let header = self.assemble_header(&plan, &root, &metadata, &leaves)?;
        self.write_archive(&header, &root, &metadata, &leaves, &plan)?;
        self.publish(header)
    }
}

/// Everything the sorting pass worked out, none of which is the entries
/// themselves: those went back to disk.
struct Plan {
    /// Path of the spilled entry list.
    entries_path: PathBuf,
    entry_count: u64,
    addressed_tiles: u64,
    /// Path of the spilled write order.
    order_path: PathBuf,
    /// How many payloads the data region carries, which is how many records
    /// that file holds.
    contents_count: u64,
    tile_data_length: u64,
}

/// One payload the data region carries, as the archive write needs it.
///
/// A staged offset and a length, which is everything: the bytes are at that
/// offset in the staging file and there are that many of them. This used to
/// be an index into a table the writer had to keep beside it.
///
/// They go to disk rather than into a `Vec`. The `Vec` was the fourth table
/// scaling with distinct payloads and the module doc named only three of them,
/// so at ten million payloads it was 80 MB nobody had counted (issue #1139).
/// Twelve bytes a record, written in placement order by [`Writer::plan_entries`]
/// and read back once, sequentially, by [`Writer::write_archive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    data_offset: u64,
    length: u32,
}

impl Placement {
    fn encode(&self) -> [u8; ORDER_RECORD_BYTES] {
        let mut out = [0u8; ORDER_RECORD_BYTES];
        out[0..8].copy_from_slice(&self.data_offset.to_le_bytes());
        out[8..12].copy_from_slice(&self.length.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8; ORDER_RECORD_BYTES]) -> Self {
        Self {
            data_offset: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
            length: u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")),
        }
    }
}

/// The leaf section, when there is one.
struct Leaves {
    path: PathBuf,
    length: u64,
}

impl<W: Write + Seek> Writer<W> {
    /// Sort the index log, assign each distinct payload its final offset, run
    /// the RLE, and spill the resulting entry list.
    ///
    /// This is the one pass that sees every tile, and it holds at most one
    /// entry plus the payload maps while doing it.
    fn plan_entries(&mut self) -> Result<Plan, PmTilesError> {
        let entries_path = suffixed(&self.base, ".ent");
        self.scratch.push(entries_path.clone());
        let mut out = BufWriter::new(File::create(&entries_path)?);

        let order_path = suffixed(&self.base, ".ord");
        self.scratch.push(order_path.clone());
        let mut order = BufWriter::new(File::create(&order_path)?);
        let mut contents_count: u64 = 0;
        let mut next_offset: u64 = 0;

        let mut entry_count: u64 = 0;
        let mut addressed: u64 = 0;
        let mut open: Option<Entry> = None;
        let mut previous_id: Option<u64> = None;

        let (log_path, runs) = self.reduce_runs()?;
        let mut source = SortedSpill::open(&log_path, &runs)?;
        while let Some(record) = source.next_record()? {
            if previous_id == Some(record.tile_id) {
                return Err(PmTilesError::DuplicateTile {
                    tile_id: record.tile_id,
                });
            }
            previous_id = Some(record.tile_id);

            // Where this payload goes. A staged offset the repeat table has
            // never heard of is referenced exactly once, so it cannot have
            // been placed already and there is nothing to look up and nothing
            // to remember: take the next offset and move on. That is every
            // tile of a photograph, and it is why finalization no longer
            // allocates anything per payload.
            let offset = match self.repeats.placement(record.data_offset) {
                Some(slot) if *slot != UNPLACED => *slot,
                marked => {
                    let offset = next_offset;
                    next_offset = next_offset.checked_add(u64::from(record.length)).ok_or(
                        PmTilesError::Overflow {
                            what: "the tile data section",
                        },
                    )?;
                    if let Some(slot) = marked {
                        *slot = offset;
                    }
                    order.write_all(
                        &Placement {
                            data_offset: record.data_offset,
                            length: record.length,
                        }
                        .encode(),
                    )?;
                    contents_count += 1;
                    offset
                }
            };

            addressed += 1;
            match open.as_mut() {
                // The run absorbs this tile only if the ids are consecutive
                // and it is the same blob, which is the same rule
                // `directory::push_entry` applies.
                Some(last)
                    if last.offset == offset
                        && last.length == record.length
                        && last.tile_id.checked_add(u64::from(last.run_length))
                            == Some(record.tile_id) =>
                {
                    last.run_length =
                        last.run_length
                            .checked_add(1)
                            .ok_or(PmTilesError::Overflow {
                                what: "a run length",
                            })?;
                }
                _ => {
                    if let Some(done) = open.take() {
                        out.write_all(&encode_entry(&done))?;
                        entry_count += 1;
                    }
                    open = Some(Entry {
                        tile_id: record.tile_id,
                        offset,
                        length: record.length,
                        run_length: 1,
                    });
                }
            }
        }
        if let Some(done) = open.take() {
            out.write_all(&encode_entry(&done))?;
            entry_count += 1;
        }
        sync_data(&out.into_inner().map_err(|e| e.into_error())?)?;
        order.into_inner().map_err(|e| e.into_error())?;

        Ok(Plan {
            entries_path,
            entry_count,
            addressed_tiles: addressed,
            order_path,
            contents_count,
            tile_data_length: next_offset,
        })
    }

    /// Fold the run list down until one merge pass can take it.
    ///
    /// Returns the log the merge should read and the runs inside it. With
    /// `MAX_MERGE_FANIN` at 128 this is a no-op below 128 runs, one extra pass
    /// below 16384, two below 2 Mi and three below 268 Mi, so even a run list
    /// produced by a checkpoint every thousand tiles over a billion-tile job
    /// costs three sequential rewrites of the index rather than a failure.
    ///
    /// Two scratch files, used alternately, because a pass cannot write into
    /// the file it is reading. Both are registered for cleanup the first time
    /// they are created.
    fn reduce_runs(&mut self) -> Result<(PathBuf, Vec<Run>), PmTilesError> {
        let mut path = suffixed(&self.base, ".idx");
        let mut runs = self.runs.clone();
        let mut pass = 0usize;
        while runs.len() > MAX_MERGE_FANIN {
            let out_path = suffixed(
                &self.base,
                if pass.is_multiple_of(2) {
                    ".mrg0"
                } else {
                    ".mrg1"
                },
            );
            if !self.scratch.contains(&out_path) {
                self.scratch.push(out_path.clone());
            }
            let mut out = BufWriter::new(File::create(&out_path)?);
            let mut folded = Vec::with_capacity(runs.len().div_ceil(MAX_MERGE_FANIN));
            let mut at: u64 = 0;
            for group in runs.chunks(MAX_MERGE_FANIN) {
                let mut source = SortedSpill::open(&path, group)?;
                let mut count: u64 = 0;
                while let Some(record) = source.next_record()? {
                    out.write_all(&record.encode())?;
                    count += 1;
                }
                folded.push(Run { start: at, count });
                at += count * SPILL_RECORD_BYTES as u64;
            }
            sync_data(&out.into_inner().map_err(|e| e.into_error())?)?;
            path = out_path;
            runs = folded;
            pass += 1;
        }
        Ok((path, runs))
    }

    /// Build the root directory, spilling into leaves when the root will not
    /// fit its budget.
    ///
    /// The spec says a sophisticated writer "might need several attempts to
    /// optimize this", which is the loop below: build the leaves at one size,
    /// see whether the resulting root fits, and widen the leaves if it does
    /// not. Starting at 4096 entries and doubling is what go-pmtiles does, and
    /// matching it means an archive built here from a given tile set has the
    /// same shape as one the reference builds from it.
    fn build_directories(
        &mut self,
        plan: &Plan,
    ) -> Result<(Vec<u8>, Option<Leaves>), PmTilesError> {
        if plan.entry_count < ROOT_ONLY_MAX_ENTRIES {
            let mut entries = Vec::with_capacity(plan.entry_count as usize);
            let mut reader = EntryReader::open(&plan.entries_path)?;
            while let Some(entry) = reader.next_entry()? {
                entries.push(entry);
            }
            let root = self
                .options
                .internal_compression
                .compress(&serialize_entries(&entries)?)?;
            if root.len() <= ROOT_BUDGET {
                return Ok((root, None));
            }
        }

        let leaf_path = suffixed(&self.base, ".leaf");
        self.scratch.push(leaf_path.clone());
        let mut leaf_entries = self.options.leaf_entries.max(1);
        loop {
            probe::leaf_attempt();
            let mut leaf_file = BufWriter::new(File::create(&leaf_path)?);
            let mut pointers: Vec<Entry> = Vec::new();
            let mut leaf_offset: u64 = 0;
            let mut reader = EntryReader::open(&plan.entries_path)?;
            let mut chunk: Vec<Entry> = Vec::with_capacity(leaf_entries);
            loop {
                chunk.clear();
                while chunk.len() < leaf_entries {
                    match reader.next_entry()? {
                        Some(entry) => chunk.push(entry),
                        None => break,
                    }
                }
                if chunk.is_empty() {
                    break;
                }
                let body = self
                    .options
                    .internal_compression
                    .compress(&serialize_entries(&chunk)?)?;
                let length =
                    u32::try_from(body.len()).map_err(|_| PmTilesError::EntryFieldTooLarge {
                        index: pointers.len(),
                        field: "leaf length",
                        value: body.len() as u64,
                    })?;
                pointers.push(Entry {
                    tile_id: chunk[0].tile_id,
                    offset: leaf_offset,
                    length,
                    // Zero is the only thing that marks an entry a leaf
                    // pointer. There is no type tag.
                    run_length: 0,
                });
                leaf_file.write_all(&body)?;
                leaf_offset += body.len() as u64;
            }
            sync_data(&leaf_file.into_inner().map_err(|e| e.into_error())?)?;

            let root = self
                .options
                .internal_compression
                .compress(&serialize_entries(&pointers)?)?;
            if root.len() <= ROOT_BUDGET {
                probe::leaf_entries_used(leaf_entries);
                return Ok((
                    root,
                    Some(Leaves {
                        path: leaf_path,
                        length: leaf_offset,
                    }),
                ));
            }
            if pointers.len() <= 1 {
                // One leaf holding everything and a root of one pointer that
                // still does not fit means the budget cannot be met at any
                // leaf size, so doubling again would spin forever.
                return Err(PmTilesError::RootDirectoryOverBudget {
                    length: root.len(),
                    budget: ROOT_BUDGET,
                });
            }
            leaf_entries = leaf_entries.checked_mul(2).ok_or(PmTilesError::Overflow {
                what: "a leaf size",
            })?;
        }
    }

    fn assemble_header(
        &self,
        plan: &Plan,
        root: &[u8],
        metadata: &[u8],
        leaves: &Option<Leaves>,
    ) -> Result<Header, PmTilesError> {
        let root_offset = HEADER_BYTES as u64;
        let root_length = root.len() as u64;
        if root_offset + root_length > ROOT_CEILING {
            return Err(PmTilesError::RootDirectoryOverBudget {
                length: root.len(),
                budget: ROOT_BUDGET,
            });
        }
        let metadata_offset = root_offset + root_length;
        let metadata_length = metadata.len() as u64;
        let leaf_directories_offset = metadata_offset + metadata_length;
        let leaf_directories_length = leaves.as_ref().map(|l| l.length).unwrap_or(0);
        // When there are no leaves this lands exactly on the tile data, which
        // is what go-pmtiles writes and what makes the *length* the flag
        // rather than the offset.
        let tile_data_offset = leaf_directories_offset
            .checked_add(leaf_directories_length)
            .ok_or(PmTilesError::Overflow {
                what: "the tile data offset",
            })?;

        let [west, south, east, north] = self.options.bounds_degrees;
        let center_zoom = self
            .options
            .center_zoom
            // `saturating_sub` because the zoom range is only meaningful once
            // a tile has arrived, and the empty case reaches here in the
            // refusal path with `min_zoom` still at its sentinel.
            .unwrap_or_else(|| self.min_zoom + self.max_zoom.saturating_sub(self.min_zoom) / 2);

        let mut header = Header {
            root_offset,
            root_length,
            metadata_offset,
            metadata_length,
            leaf_directories_offset,
            leaf_directories_length,
            tile_data_offset,
            tile_data_length: plan.tile_data_length,
            addressed_tiles_count: plan.addressed_tiles,
            tile_entries_count: plan.entry_count,
            tile_contents_count: plan.contents_count,
            // Earned, not claimed: the data region below is written in tile id
            // order, so the first tile entry is at offset 0 and every later
            // offset is either contiguous with the previous blob's end or a
            // back reference to a deduplicated one.
            clustered: true,
            internal_compression: self.options.internal_compression,
            tile_compression: self.options.tile_compression,
            tile_type: self.options.tile_type,
            min_zoom: self.min_zoom,
            max_zoom: self.max_zoom,
            // The six position fields are filled in by the two setters
            // immediately below, which own the degrees-to-e7 conversion. They
            // are zero here only because a struct literal has to name every
            // field.
            min_lon_e7: 0,
            min_lat_e7: 0,
            max_lon_e7: 0,
            max_lat_e7: 0,
            center_zoom,
            center_lon_e7: 0,
            center_lat_e7: 0,
        };
        header.set_bounds_degrees(west, south, east, north);
        header.set_center_degrees(self.options.center_degrees.0, self.options.center_degrees.1);
        Ok(header)
    }

    /// Write the five sections, in the order the spec lists them.
    fn write_archive(
        &mut self,
        header: &Header,
        root: &[u8],
        metadata: &[u8],
        leaves: &Option<Leaves>,
        plan: &Plan,
    ) -> Result<(), PmTilesError> {
        if self.sink.is_none() {
            // The staged flavour creates its archive file here rather than at
            // `create`, so an interrupted run leaves the index and the
            // payloads and not an empty file wearing an archive's name.
            let staged_archive = self.base.clone();
            self.scratch.push(staged_archive.clone());
            self.sink = Some(Sink::Staged(File::create(&staged_archive)?));
        }
        let staged_data = File::open(suffixed(&self.base, ".data"))?;
        let mut staged_data = BufReader::new(staged_data);
        let mut buffer = vec![0u8; COPY_BUFFER_BYTES];

        let sink = self.sink.as_mut().expect("a sink exists by now");
        let out = sink.as_write();

        out.write_all(&header.encode())?;
        out.write_all(root)?;
        out.write_all(metadata)?;
        if let Some(leaves) = leaves {
            let mut leaf_file = BufReader::new(File::open(&leaves.path)?);
            copy_exactly(&mut leaf_file, out, leaves.length, &mut buffer)?;
        }
        let mut order = OrderReader::open(&plan.order_path)?;
        while let Some(placement) = order.next_placement()? {
            probe::staged_seek();
            staged_data.seek(SeekFrom::Start(placement.data_offset))?;
            copy_exactly(
                &mut staged_data,
                out,
                u64::from(placement.length),
                &mut buffer,
            )?;
        }
        out.flush()?;
        Ok(())
    }

    /// Flush to disk and rename into place, or hand the sink back untouched.
    fn publish(&mut self, header: Header) -> Result<Finish, PmTilesError> {
        let sink = self.sink.take().expect("a sink exists by now");
        match (sink, self.destination.take()) {
            (Sink::Staged(mut file), Some(destination)) => {
                // Flush the payload to the device before publishing, so the
                // renamed file is not left empty or short by a power loss
                // between the rename and the writeback.
                file.flush()?;
                sync_all(&file)?;
                // Close before the rename: some filesystems refuse to rename
                // over an open handle.
                drop(file);
                std::fs::rename(&self.base, &destination)?;
                Ok(Finish {
                    header,
                    path: Some(destination),
                })
            }
            (Sink::Staged(mut file), None) => {
                file.flush()?;
                Ok(Finish { header, path: None })
            }
            (Sink::Foreign(mut out), _) => {
                out.flush()?;
                Ok(Finish { header, path: None })
            }
        }
    }
}

impl<W: Write + Seek> Drop for Writer<W> {
    /// Remove every scratch file this writer created.
    ///
    /// Runs on the success path too, where the staged archive has already been
    /// renamed away and its `remove_file` is a no-op. It does **not** run when
    /// the process is killed, which is deliberate: what is left behind then is
    /// the evidence that the run was working, under names that are obviously
    /// temporary, and never the archive's own name.
    fn drop(&mut self) {
        // Close the buffered handles before unlinking, so nothing is still
        // writing into a file that is about to go away.
        self.staged = None;
        self.log = None;
        self.sink = None;
        for path in &self.scratch {
            let _ = std::fs::remove_file(path);
        }
    }
}

// ---------------------------------------------------------------------------
// Spilled record I/O
// ---------------------------------------------------------------------------

fn suffixed(base: &Path, suffix: &str) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn encode_entry(entry: &Entry) -> [u8; ENTRY_RECORD_BYTES] {
    let mut out = [0u8; ENTRY_RECORD_BYTES];
    out[0..8].copy_from_slice(&entry.tile_id.to_le_bytes());
    out[8..16].copy_from_slice(&entry.offset.to_le_bytes());
    out[16..20].copy_from_slice(&entry.length.to_le_bytes());
    out[20..24].copy_from_slice(&entry.run_length.to_le_bytes());
    out
}

fn decode_entry(bytes: &[u8; ENTRY_RECORD_BYTES]) -> Entry {
    Entry {
        tile_id: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
        offset: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
        length: u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")),
        run_length: u32::from_le_bytes(bytes[20..24].try_into().expect("4 bytes")),
    }
}

/// Copy exactly `length` bytes, refusing a short read rather than writing a
/// shorter section than the header promised.
fn copy_exactly<R: Read>(
    from: &mut R,
    to: &mut dyn Write,
    length: u64,
    buffer: &mut [u8],
) -> Result<(), PmTilesError> {
    let mut left = length;
    while left > 0 {
        let want = usize::try_from(left.min(buffer.len() as u64)).expect("bounded by the buffer");
        from.read_exact(&mut buffer[..want])?;
        to.write_all(&buffer[..want])?;
        left -= want as u64;
    }
    Ok(())
}

/// Read exactly `buf.len()` bytes from `at`, without touching a shared cursor.
///
/// This is what lets the whole merge run on one open file. On Unix it is one
/// `pread`; elsewhere it is a seek and a read, which is equivalent here because
/// [`SortedSpill`] is not shared between threads.
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, at)
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    let mut file = file;
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(buf)
}

/// One run's cursor during the k-way merge.
///
/// It owns a position and a buffer and **not** a file handle. The version that
/// owned a `BufReader<File>` each was the whole of the descriptor problem:
/// `SortedSpill::open` called `File::open` once per run and held every one of
/// them for the length of the merge, so a job with more runs than the process
/// had descriptors died of `EMFILE` inside `finish`, after all the work, with
/// nothing to salvage because the archive only exists at the rename.
struct RunCursor {
    /// Next byte of the log this cursor will read.
    at: u64,
    /// Records not yet decoded, buffered ones included.
    left: u64,
    buffer: Vec<u8>,
    /// Bytes of `buffer` already handed out, and bytes of it that are valid.
    taken: usize,
    filled: usize,
    head: Option<Spill>,
}

impl RunCursor {
    fn open(file: &File, run: Run) -> Result<Self, PmTilesError> {
        let mut cursor = Self {
            at: run.start,
            left: run.count,
            buffer: vec![0u8; MERGE_CURSOR_RECORDS * SPILL_RECORD_BYTES],
            taken: 0,
            filled: 0,
            head: None,
        };
        cursor.advance(file)?;
        Ok(cursor)
    }

    fn advance(&mut self, file: &File) -> Result<(), PmTilesError> {
        if self.left == 0 {
            self.head = None;
            return Ok(());
        }
        if self.taken == self.filled {
            let records = self.left.min(MERGE_CURSOR_RECORDS as u64) as usize;
            let want = records * SPILL_RECORD_BYTES;
            read_exact_at(file, &mut self.buffer[..want], self.at)?;
            self.at += want as u64;
            self.taken = 0;
            self.filled = want;
        }
        let bytes: [u8; SPILL_RECORD_BYTES] = self.buffer
            [self.taken..self.taken + SPILL_RECORD_BYTES]
            .try_into()
            .expect("one record's worth");
        self.taken += SPILL_RECORD_BYTES;
        self.left -= 1;
        self.head = Some(Spill::decode(&bytes));
        Ok(())
    }
}

/// The index log, read back in tile id order.
///
/// One run means everything fitted in the sort buffer and there is nothing to
/// merge; that is the common case and it costs one sequential read. More than
/// one means a real k-way merge, which is what keeps a billion-tile archive
/// inside the sort buffer.
///
/// It holds **one** file handle whatever the run count, and refuses more than
/// [`MAX_MERGE_FANIN`] runs in a pass. [`Writer::reduce_runs`] is what makes
/// that refusal unreachable in practice, by folding a long run list down in
/// passes first.
struct SortedSpill {
    file: File,
    cursors: Vec<RunCursor>,
    queue: BinaryHeap<Reverse<(u64, usize)>>,
}

impl SortedSpill {
    fn open(path: &Path, runs: &[Run]) -> Result<Self, PmTilesError> {
        if runs.len() > MAX_MERGE_FANIN {
            return Err(PmTilesError::Overflow {
                what: "the merge fan-in",
            });
        }
        let file = File::open(path)?;
        let mut cursors = Vec::with_capacity(runs.len());
        let mut queue = BinaryHeap::with_capacity(runs.len());
        for (index, run) in runs.iter().enumerate() {
            let cursor = RunCursor::open(&file, *run)?;
            if let Some(head) = cursor.head {
                queue.push(Reverse((head.tile_id, index)));
            }
            cursors.push(cursor);
        }
        Ok(Self {
            file,
            cursors,
            queue,
        })
    }

    fn next_record(&mut self) -> Result<Option<Spill>, PmTilesError> {
        let Some(Reverse((_, index))) = self.queue.pop() else {
            return Ok(None);
        };
        let record = self.cursors[index].head.expect("a queued run has a head");
        self.cursors[index].advance(&self.file)?;
        if let Some(head) = self.cursors[index].head {
            self.queue.push(Reverse((head.tile_id, index)));
        }
        Ok(Some(record))
    }
}

/// The spilled write order, read back in order.
struct OrderReader {
    reader: BufReader<File>,
}

impl OrderReader {
    fn open(path: &Path) -> Result<Self, PmTilesError> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
        })
    }

    fn next_placement(&mut self) -> Result<Option<Placement>, PmTilesError> {
        let mut bytes = [0u8; ORDER_RECORD_BYTES];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => Ok(Some(Placement::decode(&bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// The spilled entry list, read back in order.
struct EntryReader {
    reader: BufReader<File>,
}

impl EntryReader {
    fn open(path: &Path) -> Result<Self, PmTilesError> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
        })
    }

    fn next_entry(&mut self) -> Result<Option<Entry>, PmTilesError> {
        let mut bytes = [0u8; ENTRY_RECORD_BYTES];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => Ok(Some(decode_entry(&bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("a scratch directory")
    }

    /// A staging writer that accepts `budget` bytes into the real file and
    /// then reports the disk full.
    ///
    /// The accepted prefix really is written, so a `write_all` that crosses the
    /// budget leaves bytes behind and returns an error, which is the shape of
    /// the failure this writer has to survive. `BufWriter` is deliberately not
    /// in the way: buffering would make the point at which the error surfaces
    /// depend on `BufWriter`'s internals instead of on the budget.
    pub(super) struct FailAfter {
        into: File,
        budget: usize,
        written: usize,
    }

    impl FailAfter {
        fn over(base: &Path, suffix: &str, budget: usize) -> Self {
            let into = std::fs::OpenOptions::new()
                .write(true)
                .open(suffixed(base, suffix))
                .expect("a scratch file this writer just created");
            Self {
                into,
                budget,
                written: 0,
            }
        }

        pub(super) fn finish(self) -> std::io::Result<()> {
            sync_data(&self.into)
        }

        /// The durability barrier, honoured here too.
        ///
        /// A stand-in that skipped it would let a partial-write test pass for
        /// the wrong reason: the bytes the assertion reads would not have
        /// reached the file it reads them from.
        pub(super) fn sync(&mut self) -> std::io::Result<()> {
            sync_data(&self.into)
        }
    }

    impl Write for FailAfter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.written >= self.budget {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    "no space left on device",
                ));
            }
            let room = (self.budget - self.written).min(buf.len());
            let n = self.into.write(&buf[..room])?;
            self.written += n;
            Ok(n)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.into.flush()
        }
    }

    impl<W: Write + Seek> Writer<W> {
        /// Swap the staging writer for one that runs out of room, keeping the
        /// same underlying file.
        fn stage_into_a_full_disk_after(&mut self, budget: usize) {
            self.staged = Some(Staging::FailsAfter(FailAfter::over(
                &self.base, ".data", budget,
            )));
        }

        /// The same, for the index log.
        fn log_into_a_full_disk_after(&mut self, budget: usize) {
            self.log = Some(Staging::FailsAfter(FailAfter::over(
                &self.base, ".idx", budget,
            )));
        }
    }

    /// How many descriptors this process holds open **on `path`**.
    ///
    /// Counting every open descriptor instead would be measuring the rest of
    /// the suite: `cargo test` runs these in parallel threads of one process
    /// and the total moves under you between two reads of it. Per-path is
    /// exact, and `/proc/self/fd` is where the target of a descriptor is
    /// readable, so this is Linux-only and the gate is where it runs.
    #[cfg(target_os = "linux")]
    fn descriptors_on(path: &Path) -> usize {
        let want = path.canonicalize().expect("the log exists");
        std::fs::read_dir("/proc/self/fd")
            .expect("/proc is mounted")
            .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
            .filter(|target| *target == want)
            .count()
    }

    /// The per-payload figure the budget is divided by is the one the types
    /// really cost.
    ///
    /// `DEDUPE_BYTES_PER_PAYLOAD` is a constant a caller's budget is divided
    /// by, so a slot that grew a field would quietly hand out a window bigger
    /// than the budget asked for. Nothing else in the writer would notice.
    #[test]
    fn the_dedupe_budget_buys_what_it_says_it_buys() {
        let per_payload = std::mem::size_of::<WindowSlot>()
            + std::mem::size_of::<RepeatSlot>()
            // Two recency words a set, spread over the set's ways.
            + 2 * std::mem::size_of::<u32>() / DEDUPE_WINDOW_WAYS;
        assert_eq!(
            per_payload, DEDUPE_BYTES_PER_PAYLOAD,
            "the slots cost {per_payload} bytes a payload and the budget is divided by \
             {DEDUPE_BYTES_PER_PAYLOAD}"
        );

        // And the division lands where it should, at the two ends that matter.
        assert_eq!(dedupe_sets(0), 1, "the smallest window is still a window");
        assert_eq!(
            dedupe_sets(DEDUPE_BYTES_PER_PAYLOAD * DEDUPE_WINDOW_WAYS * 100),
            100
        );
        let sets = dedupe_sets(DEFAULT_DEDUPE_MEMORY_BYTES);
        let spent = sets * DEDUPE_WINDOW_WAYS * DEDUPE_BYTES_PER_PAYLOAD;
        assert!(
            spent <= DEFAULT_DEDUPE_MEMORY_BYTES,
            "the default window spends {spent} against a budget of {DEFAULT_DEDUPE_MEMORY_BYTES}"
        );
    }

    /// The window evicts the way that has gone longest without a hit, and a
    /// hit is what makes a way recent.
    ///
    /// Driven directly rather than through an archive, because the archive
    /// test can only see the two ends of this and the thing in the middle,
    /// that a *hit* renews an entry rather than only an insert, is what makes
    /// a blank tile survive a window it does not fit in.
    #[test]
    fn a_hit_renews_an_entry_and_the_oldest_way_is_the_one_that_goes() {
        let hash = |n: u8| [n; 32];
        let mut window = DedupeWindow::with_sets(1);
        for n in 0..DEDUPE_WINDOW_WAYS as u8 {
            window.insert(hash(n), u64::from(n), 1);
        }
        // Way 0 is the oldest, so renew it and fill the set once more. What
        // goes is way 1, which is now the oldest.
        assert_eq!(window.get(&hash(0)), Some((0, 1)));
        window.insert(hash(100), 100, 1);
        assert_eq!(
            window.get(&hash(0)),
            Some((0, 1)),
            "a hit should have renewed the oldest entry"
        );
        assert_eq!(
            window.get(&hash(1)),
            None,
            "the entry that had gone longest without a hit should be the one evicted"
        );
        // And everything else is still there, so the eviction took one.
        for n in 2..DEDUPE_WINDOW_WAYS as u8 {
            assert_eq!(window.get(&hash(n)), Some((u64::from(n), 1)), "way {n}");
        }
    }

    #[test]
    fn a_spill_record_round_trips_through_its_twenty_bytes() {
        let record = Spill {
            tile_id: u64::MAX - 3,
            data_offset: 1 << 40,
            length: u32::MAX,
        };
        assert_eq!(Spill::decode(&record.encode()), record);
        assert_eq!(record.encode().len(), SPILL_RECORD_BYTES);
    }

    #[test]
    fn an_order_record_round_trips_through_its_twelve_bytes() {
        let placement = Placement {
            data_offset: 1 << 40,
            length: u32::MAX,
        };
        assert_eq!(Placement::decode(&placement.encode()), placement);
        assert_eq!(placement.encode().len(), ORDER_RECORD_BYTES);
    }

    #[test]
    fn an_entry_record_round_trips_through_its_twenty_four_bytes() {
        let entry = Entry {
            tile_id: 19_078_479,
            offset: 5 * 1024 * 1024 * 1024,
            length: 4096,
            run_length: 17,
        };
        assert_eq!(decode_entry(&encode_entry(&entry)), entry);
        assert_eq!(encode_entry(&entry).len(), ENTRY_RECORD_BYTES);
    }

    /// The merge really merges: three runs with interleaved ids come back in
    /// one ascending sequence.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_external_merge_puts_interleaved_runs_back_in_order() {
        let dir = temp_dir();
        let path = dir.path().join("idx");
        let runs_in: Vec<Vec<u64>> = vec![vec![0, 3, 9], vec![1, 4, 5, 8], vec![2, 6, 7]];

        let mut file = File::create(&path).unwrap();
        let mut runs = Vec::new();
        let mut at = 0u64;
        for ids in &runs_in {
            for id in ids {
                file.write_all(
                    &Spill {
                        tile_id: *id,
                        data_offset: *id * 10,
                        length: 1,
                    }
                    .encode(),
                )
                .unwrap();
            }
            runs.push(Run {
                start: at,
                count: ids.len() as u64,
            });
            at += (ids.len() * SPILL_RECORD_BYTES) as u64;
        }
        file.sync_all().unwrap();
        drop(file);

        let mut merged = SortedSpill::open(&path, &runs).unwrap();
        let mut got = Vec::new();
        while let Some(record) = merged.next_record().unwrap() {
            got.push(record.tile_id);
        }
        assert_eq!(got, (0..10).collect::<Vec<u64>>());
    }

    /// A merge over a single run is still a merge, and it is the common case.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_external_merge_handles_one_run_and_no_runs() {
        let dir = temp_dir();
        let path = dir.path().join("idx");
        File::create(&path).unwrap().sync_all().unwrap();
        let mut empty = SortedSpill::open(&path, &[]).unwrap();
        assert!(empty.next_record().unwrap().is_none());
    }

    /// `copy_exactly` refuses a short source rather than writing a shorter
    /// section than the header promised.
    #[test]
    fn copying_refuses_a_source_that_runs_out() {
        let mut source = std::io::Cursor::new(vec![1u8, 2, 3]);
        let mut sink: Vec<u8> = Vec::new();
        let mut buffer = vec![0u8; 2];
        let err = copy_exactly(&mut source, &mut sink, 4, &mut buffer)
            .expect_err("four bytes are not there");
        assert!(matches!(err, PmTilesError::Io(_)), "got {err:?}");
    }

    /// The scratch prefix of two writers in one directory does not collide.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn two_writers_in_one_scratch_directory_get_distinct_prefixes() {
        let dir = temp_dir();
        let a: Writer<std::io::Cursor<Vec<u8>>> = Writer::try_new(
            std::io::Cursor::new(Vec::new()),
            dir.path(),
            WriterOptions::default(),
        )
        .unwrap();
        let b: Writer<std::io::Cursor<Vec<u8>>> = Writer::try_new(
            std::io::Cursor::new(Vec::new()),
            dir.path(),
            WriterOptions::default(),
        )
        .unwrap();
        assert_ne!(a.base, b.base);
    }

    /// A tile that could not be written means no archive, not a wrong one.
    ///
    /// This is the whole failure. `add_tile` advances its bookkeeping only
    /// after `write_all` returns, so a write that fails part way leaves bytes
    /// in the staging file that nothing accounts for. `PmTilesSink` does not
    /// apply its own retry policy, so `EngineBuilder` wraps it in
    /// `RetryingSink`: without the latch the retry appends after the orphan and
    /// records an offset pointing into the middle of it, every later payload is
    /// shifted by the same amount, and under `FailurePolicy::RetryThenSkip` the
    /// run *succeeds* and publishes a structurally valid archive full of the
    /// wrong tile bytes.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_partial_write_latches_and_finish_refuses_to_publish() {
        let dir = temp_dir();
        let destination = dir.path().join("pyramid.pmtiles");
        let mut w = Writer::create(&destination, WriterOptions::default()).unwrap();
        // 6000 bytes of room and 4096-byte payloads, so the second tile is
        // accepted for 1904 bytes and then refused.
        w.stage_into_a_full_disk_after(6000);

        let first = vec![1u8; 4096];
        w.add_tile(3, 0, 0, &first, content_hash(&first))
            .expect("the first tile fits");

        let second = vec![2u8; 4096];
        let err = w
            .add_tile(3, 1, 0, &second, content_hash(&second))
            .expect_err("the second tile runs out of room");
        assert!(matches!(err, PmTilesError::Io(_)), "got {err:?}");

        // The partial write really happened: the staging file is longer than
        // the writer's own account of it. Those extra bytes are the orphan the
        // latch exists for, and without it the next payload's offset would be
        // recorded on the far side of them.
        let staged_path = suffixed(&w.base, ".data");
        let on_disk = std::fs::metadata(&staged_path).unwrap().len();
        assert_eq!(w.staged_len, 4096, "only the first payload is accounted");
        assert!(
            on_disk > w.staged_len,
            "the staging file is {on_disk} bytes and the writer accounts for {}, so no \
             partial write happened and this test is not testing anything",
            w.staged_len
        );

        // The retry is refused rather than appended after the orphan.
        let third = vec![3u8; 16];
        let err = w
            .add_tile(3, 2, 0, &third, content_hash(&third))
            .expect_err("a latched writer accepts nothing more");
        assert!(
            matches!(
                err,
                PmTilesError::WriterFailed {
                    during: "staging a payload"
                }
            ),
            "got {err:?}"
        );

        let err = w.finish().expect_err("a latched writer must not publish");
        assert!(
            matches!(
                err,
                PmTilesError::WriterFailed {
                    during: "staging a payload"
                }
            ),
            "got {err:?}"
        );
        assert!(
            !destination.exists(),
            "a run that could not write a tile published an archive anyway"
        );
    }

    /// The index log gets the same treatment, latched inside `flush_run`.
    ///
    /// The latch lives in `flush_run` rather than at its call site because
    /// `add_tile` is not the only thing that reaches it: F1.4's sink calls a
    /// durability barrier that flushes the sort buffer as a run, and a barrier
    /// that failed half way through writing one would otherwise leave a
    /// writer that still publishes.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_partial_write_to_the_index_log_latches_too() {
        let dir = temp_dir();
        let destination = dir.path().join("pyramid.pmtiles");
        // One record a run, so every tile writes 20 bytes of log. 30 bytes of
        // room means the second run is accepted for 10 and then refused.
        let mut w = Writer::create(
            &destination,
            WriterOptions::default().with_sort_buffer_records(1),
        )
        .unwrap();
        w.log_into_a_full_disk_after(30);

        let first = b"one".as_slice();
        w.add_tile(3, 0, 0, first, content_hash(first))
            .expect("the first tile fits");

        let second = b"two".as_slice();
        let err = w
            .add_tile(3, 1, 0, second, content_hash(second))
            .expect_err("the log runs out of room");
        assert!(matches!(err, PmTilesError::Io(_)), "got {err:?}");

        let log_path = suffixed(&w.base, ".idx");
        let on_disk = std::fs::metadata(&log_path).unwrap().len();
        assert_eq!(
            on_disk, 30,
            "the log should hold one whole record and half of another"
        );
        assert_eq!(
            w.log_len, SPILL_RECORD_BYTES as u64,
            "and the writer should only account for the whole one"
        );

        let third = b"three".as_slice();
        let err = w
            .add_tile(3, 2, 0, third, content_hash(third))
            .expect_err("a latched writer accepts nothing more");
        assert!(
            matches!(
                err,
                PmTilesError::WriterFailed {
                    during: "appending to the index log"
                }
            ),
            "got {err:?}"
        );

        let err = w.finish().expect_err("a latched writer must not publish");
        assert!(
            matches!(
                err,
                PmTilesError::WriterFailed {
                    during: "appending to the index log"
                }
            ),
            "got {err:?}"
        );
        assert!(!destination.exists());
    }

    /// A refusal that happens before any byte moves does not latch.
    ///
    /// The distinction matters: an engine running `FailurePolicy::Skip` is
    /// entitled to hand this writer a tile it will refuse and carry on, and a
    /// latch on those would turn every skipped tile into a failed run.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_refusal_before_the_write_leaves_the_writer_usable() {
        let dir = temp_dir();
        let destination = dir.path().join("pyramid.pmtiles");
        let mut w = Writer::create(&destination, WriterOptions::default()).unwrap();

        let empty: &[u8] = &[];
        assert!(w.add_tile(3, 0, 0, empty, content_hash(empty)).is_err());
        // A coordinate outside its own zoom's grid, refused rather than masked.
        let payload = b"tile".as_slice();
        assert!(
            w.add_tile(3, 99, 0, payload, content_hash(payload))
                .is_err()
        );

        w.add_tile(3, 0, 0, payload, content_hash(payload))
            .expect("the writer is still usable");
        let done = w.finish().expect("and it still publishes");
        assert_eq!(done.header.addressed_tiles_count, 1);
        assert!(destination.exists());
    }

    /// The merge holds one descriptor, whatever the run count.
    ///
    /// The count itself is Linux-only, because `/proc/self/fd` is where a
    /// descriptor's target is readable and counting descriptors any other way
    /// would be counting the rest of the suite. The merge's output is checked
    /// everywhere.
    ///
    /// It used to hold one per run and keep them for the whole pass, so a job
    /// with more runs than the process had descriptors died of `EMFILE` inside
    /// `finish`, after all the work, with nothing to salvage because the
    /// archive only exists at the rename. macOS ships a soft `RLIMIT_NOFILE` of
    /// 256 and Linux usually 1024, and a checkpoint every thousand tiles forces
    /// a run boundary, so it was a quarter of a million tiles away on a laptop.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_merge_opens_one_file_however_many_runs_it_reads() {
        let dir = temp_dir();
        let path = dir.path().join("idx");
        let mut file = File::create(&path).unwrap();
        let mut runs = Vec::new();
        let mut at = 0u64;
        // The widest pass the merge will take, so the descriptor count is
        // measured at the limit rather than somewhere comfortable.
        for run in 0..MAX_MERGE_FANIN as u64 {
            for step in 0..4u64 {
                file.write_all(
                    &Spill {
                        tile_id: run + step * MAX_MERGE_FANIN as u64,
                        data_offset: run,
                        length: 1,
                    }
                    .encode(),
                )
                .unwrap();
            }
            runs.push(Run {
                start: at,
                count: 4,
            });
            at += 4 * SPILL_RECORD_BYTES as u64;
        }
        file.sync_all().unwrap();
        drop(file);

        #[cfg(target_os = "linux")]
        let before = descriptors_on(&path);
        let mut merged = SortedSpill::open(&path, &runs).unwrap();
        #[cfg(target_os = "linux")]
        {
            let during = descriptors_on(&path);
            // Nothing else in this process has the log open, which is what
            // makes the second number meaningful rather than a coincidence.
            assert_eq!(before, 0, "something already had the log open");
            assert_eq!(
                during,
                1,
                "{} runs cost {during} descriptors on the log",
                runs.len()
            );
        }

        // And it still merges: every id once, ascending.
        let mut ids = Vec::new();
        while let Some(record) = merged.next_record().unwrap() {
            ids.push(record.tile_id);
        }
        assert_eq!(ids.len(), MAX_MERGE_FANIN * 4);
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "the merge is not sorted"
        );
    }

    /// One pass refuses a run list wider than the fan-in rather than opening
    /// it.
    ///
    /// `reduce_runs` is what makes this unreachable from `finish`, and that is
    /// exactly why the refusal needs its own test: a guard whose caller always
    /// satisfies it has nothing else watching it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_merge_pass_refuses_more_runs_than_its_fan_in() {
        let dir = temp_dir();
        let path = dir.path().join("idx");
        let mut file = File::create(&path).unwrap();
        let mut runs = Vec::new();
        let mut at = 0u64;
        for id in 0..=MAX_MERGE_FANIN as u64 {
            file.write_all(
                &Spill {
                    tile_id: id,
                    data_offset: id,
                    length: 1,
                }
                .encode(),
            )
            .unwrap();
            runs.push(Run {
                start: at,
                count: 1,
            });
            at += SPILL_RECORD_BYTES as u64;
        }
        file.sync_all().unwrap();
        drop(file);

        assert_eq!(runs.len(), MAX_MERGE_FANIN + 1);
        let Err(err) = SortedSpill::open(&path, &runs) else {
            panic!("one run too many was accepted");
        };
        assert!(
            matches!(
                err,
                PmTilesError::Overflow {
                    what: "the merge fan-in"
                }
            ),
            "got {err:?}"
        );
        // The positive control: one fewer run is accepted, so the refusal is
        // about the width and not about the file.
        assert!(
            SortedSpill::open(&path, &runs[..MAX_MERGE_FANIN]).is_ok(),
            "the full width should be fine"
        );
    }

    /// More runs than one pass will take is folded down rather than refused.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_run_list_wider_than_the_fan_in_is_merged_in_passes() {
        let dir = temp_dir();
        let destination = dir.path().join("wide.pmtiles");
        // One record a run, so the run count is the tile count.
        let tiles = MAX_MERGE_FANIN * 2 + 45;
        let mut w = Writer::create(
            &destination,
            WriterOptions::default().with_sort_buffer_records(1),
        )
        .unwrap();
        // Added back to front, so an unsorted merge is visible in the output.
        for id in (0..tiles as u64).rev() {
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(id + 5).unwrap();
            let payload = format!("tile {id}").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }
        assert_eq!(
            w.spilled_run_count(),
            tiles,
            "the point of this test is a run list wider than the fan-in"
        );
        assert!(w.spilled_run_count() > MAX_MERGE_FANIN);

        let done = w.finish().expect("a wide run list still finishes");
        assert_eq!(done.header.addressed_tiles_count, tiles as u64);
        assert_eq!(done.header.tile_contents_count, tiles as u64);
        assert!(destination.exists());

        // And the archive really is in tile id order, checked by decoding its
        // root rather than by trusting the header's own count. This entry count
        // is below `ROOT_ONLY_MAX_ENTRIES`, so the root is flat.
        let bytes = std::fs::read(&destination).unwrap();
        let header = Header::try_decode(&bytes).expect("the header decodes");
        let root =
            &bytes[header.root_offset as usize..(header.root_offset + header.root_length) as usize];
        let plain = header
            .internal_compression
            .decompress(root, 1 << 20)
            .expect("the root decompresses");
        let entries =
            crate::pmtiles::directory::deserialize_entries(&plain).expect("the root parses");
        assert_eq!(
            entries.len(),
            tiles,
            "the root is not flat, adjust the test"
        );
        assert!(
            entries.windows(2).all(|w| w[0].tile_id < w[1].tile_id),
            "the entries are not in tile id order"
        );
        assert!(
            entries.iter().all(|e| e.run_length == 1),
            "distinct payloads must not collapse into runs"
        );
    }

    /// The window answers both questions the payload table used to.
    ///
    /// `payload_starts` was a `Vec<u64>` of where every payload began, plus a
    /// sentinel, so an offset and a length both came out of it by
    /// subtraction. It grew one entry per payload for the whole run. The
    /// window carries both numbers for the payloads it still remembers, and
    /// the staged length is a counter.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_window_carries_every_offset_and_length() {
        let dir = temp_dir();
        let mut sink = std::io::Cursor::new(Vec::new());
        let mut w = Writer::try_new(&mut sink, dir.path(), WriterOptions::default()).unwrap();

        let payloads: Vec<Vec<u8>> = (1..=5u8).map(|n| vec![n; n as usize * 3]).collect();
        for (index, payload) in payloads.iter().enumerate() {
            w.add_tile(3, index as u32, 0, payload, content_hash(payload))
                .unwrap();
        }
        // A duplicate, which must not stage a second payload.
        w.add_tile(3, 6, 0, &payloads[2], content_hash(&payloads[2]))
            .unwrap();

        assert_eq!(w.staged_payloads, 5);
        assert_eq!(w.distinct_payload_count(), 5);
        let mut expected_offset = 0u64;
        for payload in &payloads {
            assert_eq!(
                w.window
                    .as_mut()
                    .expect("the window is live before finish")
                    .get(&content_hash(payload)),
                Some((expected_offset, payload.len() as u32)),
                "the window should know where this payload is and how long it is"
            );
            expected_offset += payload.len() as u64;
        }
        assert_eq!(w.staged_len, expected_offset);
    }

    /// A `finish` issues exactly one durability barrier.
    ///
    /// Every other `sync_data` in `finish` is on a scratch file that `Drop`
    /// deletes and that no resume path ever reads back: `PmTilesSink` refuses
    /// resume outright, `checkpoint_root` returns `None` and
    /// `seed_completed_tile` errors. So the durability was bought and never
    /// spent, and against a cell whose whole job is a few dozen PNG encodes,
    /// four or five `fsync`s is a real share of the wall time (issue #1141).
    ///
    /// The one that stays is the `sync_all` in `publish`, before the rename,
    /// which is the only one with anything to protect: it is what stops a
    /// power loss between the rename and the writeback leaving a complete
    /// archive's name on an empty or short file.
    ///
    /// The profile is deliberately awkward. One record a run means 300 runs
    /// and a merge fold; one entry a leaf means the leaf loop runs. Both of
    /// those used to sync.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_finish_issues_exactly_one_durability_barrier() {
        let dir = temp_dir();
        let destination = dir.path().join("barriers.pmtiles");
        let mut w = Writer::create(
            &destination,
            WriterOptions::default()
                .with_sort_buffer_records(1)
                .with_leaf_entries(1),
        )
        .unwrap();
        for id in 21u64..=320 {
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(id).unwrap();
            let payload = format!("tile number {id}, distinct from every other").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }
        assert!(
            w.spilled_run_count() > MAX_MERGE_FANIN,
            "this profile is supposed to reach the merge fold"
        );

        probe::reset();
        w.finish().expect("the archive finishes");

        assert_eq!(
            probe::syncs(),
            1,
            "a finish should issue one durability barrier, the sync_all before the rename"
        );
        assert!(destination.exists());
    }

    /// `sync_pending` still syncs, because that one was asked for.
    ///
    /// The control on the test above. A writer that simply stopped syncing
    /// anything would pass it, and would quietly drop the barrier a
    /// checkpointed engine run explicitly calls for.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_barrier_a_caller_asks_for_is_still_issued() {
        let dir = temp_dir();
        let mut sink = std::io::Cursor::new(Vec::new());
        let mut w = Writer::try_new(&mut sink, dir.path(), WriterOptions::default()).unwrap();
        let payload = b"one tile".as_slice();
        w.add_tile(3, 0, 0, payload, content_hash(payload)).unwrap();

        probe::reset();
        w.sync_pending().expect("the barrier lands");
        assert_eq!(
            probe::syncs(),
            2,
            "sync_pending syncs the staged payloads and the index log"
        );
    }

    /// The payload copy reads each blob where it lies instead of seeking to
    /// it.
    ///
    /// `write_archive` used to `seek` a `BufReader` and then read. `BufReader`
    /// discards its buffer on a seek by documented contract, so every payload
    /// cost an `lseek`, a thrown-away readahead and a fresh read: three
    /// syscalls a payload, ten million times on a ten-million-payload archive.
    /// The positioned-read helper the merge already runs on does it in one
    /// (issue #1142).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_payload_copy_reads_each_payload_where_it_lies_rather_than_seeking_to_it() {
        const PAYLOADS: usize = 64;
        let dir = temp_dir();
        let mut sink = std::io::Cursor::new(Vec::new());
        let mut w = Writer::try_new(&mut sink, dir.path(), WriterOptions::default()).unwrap();
        for index in 0..PAYLOADS as u64 {
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(21 + index).unwrap();
            let payload = format!("payload {index}").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }

        probe::reset();
        w.finish().expect("the archive finishes");

        assert_eq!(
            probe::staged_seeks(),
            0,
            "the copy should not move a shared cursor at all"
        );
        assert_eq!(
            probe::staged_reads(),
            PAYLOADS,
            "one positioned read a payload, and these payloads are far under the copy buffer"
        );
    }

    /// The leaf loop does not rebuild every leaf once per doubling.
    ///
    /// `build_directories` starts at the configured leaf size and doubles
    /// until the root fits, re-gzipping every leaf on every attempt. The size
    /// it settles on says how many attempts a pure doubling loop needs, so the
    /// assertion is against that number rather than against a constant, and
    /// the control below refuses a fixture that does not force enough
    /// doublings to be able to tell the two apart (issue #1142).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_leaf_loop_does_not_rebuild_every_leaf_once_per_doubling() {
        const ENTRIES: u64 = 40_000;
        let dir = temp_dir();
        let mut sink = std::io::Cursor::new(Vec::new());
        let mut w = Writer::try_new(
            &mut sink,
            dir.path(),
            WriterOptions::default().with_leaf_entries(1),
        )
        .unwrap();

        // Widely spaced ids, so the root's tile-id column carries real
        // entropy. This is the whole difficulty of building a fixture that
        // forces doublings: a root of leaf pointers has an all-zero run-length
        // column, an offset column that is all zeros because the leaves are
        // contiguous, and a length column of compressed leaf sizes that barely
        // move, so gzip takes the lot to well under a byte a pointer. Measured
        // on contiguous ids, twenty thousand pointers fit the 16257-byte root
        // budget with room to spare. Four-byte deltas are what make a pointer
        // cost something.
        let mut id = 21u64;
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..ENTRIES {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(id).unwrap();
            let size = 32 + (rng >> 33) as usize % 96;
            let mut payload = vec![0u8; size];
            payload[..8].copy_from_slice(&id.to_le_bytes());
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
            id += 1 + (rng >> 38);
        }

        probe::reset();
        w.finish().expect("the archive finishes");

        let attempts = probe::leaf_attempts();
        let settled = probe::leaf_entries();
        assert!(settled >= 1, "the leaf loop never reported a size");
        // A pure doubling loop from 1 reaches `settled` in this many attempts,
        // counting the first.
        let doublings = settled.trailing_zeros() as usize + 1;
        println!("leaf loop: {attempts} attempts, settled on {settled} entries a leaf");
        assert!(
            doublings >= 4,
            "this fixture settled on {settled} entries a leaf, which is only {doublings} \
             doublings away from the start, so it cannot tell a doubling loop from anything else"
        );
        assert!(
            attempts < doublings,
            "the leaf section was built {attempts} times to settle on {settled} entries a \
             leaf, which is what plain doubling costs"
        );
    }

    /// The sort buffer spills once it fills, and the merge still produces one
    /// ascending sequence. This one forces the boundary by hand; the run size
    /// is configurable through
    /// [`WriterOptions::with_sort_buffer_records`](super::WriterOptions::with_sort_buffer_records)
    /// and `a_run_list_wider_than_the_fan_in_is_merged_in_passes` drives it
    /// that way instead.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_writer_that_spilled_several_runs_still_sorts() {
        let dir = temp_dir();
        let mut sink = std::io::Cursor::new(Vec::new());
        let mut w = Writer::try_new(&mut sink, dir.path(), WriterOptions::default()).unwrap();
        // Ids 21..=84 are zoom 3, added back to front.
        for id in (21u64..=84).rev() {
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(id).unwrap();
            let payload = format!("{id}").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }
        // Force two runs out of one writer.
        w.flush_run().unwrap();
        for id in 5u64..=20 {
            let (z, x, y) = crate::pmtiles::tileid_to_zxy(id).unwrap();
            let payload = format!("{id}").into_bytes();
            w.add_tile(z, x, y, &payload, content_hash(&payload))
                .unwrap();
        }
        assert_eq!(w.runs.len(), 1, "one run should be on disk before finish");
        let done = w.finish().unwrap();
        assert_eq!(done.header.addressed_tiles_count, 80);
        assert_eq!(done.header.min_zoom, 2);
        assert_eq!(done.header.max_zoom, 3);
    }
}
