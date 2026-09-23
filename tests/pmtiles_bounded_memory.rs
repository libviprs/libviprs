//! The PMTiles writer's memory bound, measured rather than asserted in prose
//! (issue #993).
//!
//! `src/pmtiles/writer.rs` claims a streaming writer whose live memory is
//! "bounded, and independent of the tile count". It used to claim that with
//! one stated exception, the tables that scale with the number of **distinct
//! payloads**, and EPIC #1135 removed the exception: the dedupe window has a
//! fixed capacity the caller sizes, the index record carries its own staged
//! offset, and the write order is spilled. Nothing the writer holds grows with
//! the payload count any more.
//!
//! Until this file, nothing measured either half of that sentence, and the
//! half that was an exception is exactly the half a benchmark is most likely
//! to hide, because a synthetic pyramid of identical tiles has one distinct
//! payload however many tiles it has.
//!
//! # What is measured, and with what
//!
//! A `#[global_allocator]` that counts bytes in flight. Not the engine's
//! [`MemoryTracker`](libviprs::MemoryTracker), which charges raster buffers
//! and knows nothing about the writer's `Vec`s, and not process RSS, which is
//! a high-water mark the allocator never gives back and so cannot show a peak
//! falling. Live heap bytes is the only basis on which "bounded during
//! finalize" is a statement with a truth value.
//!
//! The counter is process-wide, so every measuring test takes [`MEASURING`]
//! first and the binary holds nothing but measuring tests. A test blocked on
//! that mutex has already allocated its own frame, which is a few hundred
//! bytes against bounds in the megabytes.
//!
//! # Three assertions, because one number proves nothing
//!
//! A single peak under a single ceiling is satisfied by a writer that lost
//! every tile, and satisfied by a writer whose bound is really the tile count
//! with a ceiling that happens to be generous. So:
//!
//! * [`the_finalize_peak_does_not_move_when_the_tile_count_quadruples`] holds
//!   the sort buffer and the distinct-payload count fixed and multiplies the
//!   tiles by four. This is the "not the tile count" half.
//! * [`the_finalize_peak_grows_with_the_sort_buffer_it_was_given`] holds the
//!   tiles fixed and multiplies the sort buffer by 64, and asserts the peak
//!   moves by roughly the buffer's own size. This is the "scales with the sort
//!   buffer" half, and it is what stops the first test from passing on a
//!   writer whose peak is a constant nothing can move.
//! * [`the_finalize_peak_stays_under_the_stated_bound`] checks the absolute
//!   number against a formula, so the two relative tests cannot both pass on a
//!   writer that is bounded at a gigabyte.
//! * [`the_add_phase_footprint_does_not_grow_with_the_distinct_payload_count`]
//!   and [`the_finalize_growth_does_not_follow_the_distinct_payload_count`]
//!   are the pair issue #1140 exists for. The formula in [`bound_for`] used to
//!   carry a `PAYLOAD_TABLE_BYTES * distinct_payloads` term, which is to say
//!   the bound scaled with the exact quantity it was written to bound, so the
//!   test certified the growth instead of catching it and
//!   `the_finalize_peak_stays_under_the_stated_bound` passed throughout. These
//!   two multiply the distinct payloads by sixteen and hold everything else
//!   still.
//!
//! Every one of them also asserts the header the writer produced, because a
//! run that silently dropped tiles would beat all three bounds.
//!
//! # The offsets past 4 GiB
//!
//! Entry offsets are `u64` and the arithmetic that turns one into a file
//! position is in
//! [`Reader::entry_position`](libviprs::pmtiles::reader). Proving it at a real
//! `u32` boundary needs an archive whose tile data section is over 4 GiB,
//! which on the write side means really staging 4 GiB and is therefore the
//! opt-in [`an_archive_past_four_gibibytes_finalizes_in_bounded_memory`]. The
//! read side needs no such thing: `tests/pmtiles_index_only_reads.rs`
//! fabricates the header and the root directory of a 6 GiB archive and serves
//! it through a counting [`RangeReader`](libviprs::pmtiles::RangeReader), for
//! free and at exact offsets.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use libviprs::pmtiles::writer::{Writer, WriterOptions, content_hash};
use libviprs::pmtiles::{Header, TileType, zxy_to_tileid};

// ---------------------------------------------------------------------------
// The counting allocator
// ---------------------------------------------------------------------------

static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Serialises the measuring tests against each other, since the counter above
/// is one per process.
static MEASURING: Mutex<()> = Mutex::new(());

struct Counting;

impl Counting {
    fn charge(size: usize) {
        let now = IN_FLIGHT.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
        PEAK.fetch_max(now, Ordering::Relaxed);
    }

    fn discharge(size: usize) {
        IN_FLIGHT.fetch_sub(size as u64, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System`, which upholds the `GlobalAlloc`
// contract, and returns exactly what it returned. The counters are relaxed
// atomics read only for reporting, so they cannot affect the pointers handed
// back or the layouts passed on.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            Self::charge(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::charge(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        Self::discharge(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            // Charge the new block before discharging the old one, so the peak
            // this records is the **sum** of the two and not the larger of
            // them. That is deliberate and it is not what the allocator really
            // holds: `System.realloc` may grow a block in place and hold one.
            // It is the conservative direction for an upper-bound test, which
            // is the only kind of test here, and the cost is that a peak
            // dominated by one big growing `Vec` reads high. Measured on the
            // 262144-record sort buffer, this order reports 9454340 bytes and
            // discharging first reports 6809585, so 28% of that figure is the
            // wrapper rather than the process. The 4096-record cell is
            // identical either way and the absolute-bound cell moves by 1550
            // bytes on 650215, which is why nothing published here moves.
            Self::charge(new_size);
            Self::discharge(layout.size());
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Take the measuring lock and zero the high-water mark.
///
/// The peak is seeded from the live total rather than from zero, so what a
/// measurement reports is growth over the quiescent baseline rather than the
/// absolute footprint of a test binary.
fn start_measuring() -> (MutexGuard<'static, ()>, u64) {
    let guard = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let baseline = IN_FLIGHT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    (guard, baseline)
}

fn peak_over(baseline: u64) -> u64 {
    PEAK.load(Ordering::Relaxed).saturating_sub(baseline)
}

fn live_over(baseline: u64) -> u64 {
    IN_FLIGHT.load(Ordering::Relaxed).saturating_sub(baseline)
}

// ---------------------------------------------------------------------------
// The bound
// ---------------------------------------------------------------------------

/// What one spilled index record costs in memory, from the writer's own note
/// over `SORT_RUN_RECORDS`: 20 bytes on disk, 24 in memory after padding.
const SPILL_BYTES_IN_MEMORY: u64 = 24;

/// Headroom factor on the sort buffer, because `Vec` grows by doubling and a
/// buffer that filled to `n` records may hold capacity for up to `2n`.
const SORT_BUFFER_SLACK: u64 = 2;

/// The deduplication budget every measurement here hands the writer.
///
/// It is passed explicitly rather than left at the default because it is the
/// one allocation in the writer that a caller sizes, and a bound that quoted
/// the default would move whenever the default did. Small on purpose: the
/// point of these tests is what the writer does per payload, and a budget
/// large enough to hide that would hide it.
const DEDUPE_BUDGET_BYTES: usize = 1024 * 1024;

/// Everything that is neither the sort buffer nor a payload table: the merge's
/// capped fan-in of buffered readers, one leaf directory's worth of entries
/// and its serialised form, the root directory, and the copy buffer. All of it
/// is fixed by constants in the writer rather than by anything the caller
/// passes.
///
/// Measured at about 550 KiB on 262144 tiles with a 4096-record sort buffer
/// and 257 distinct payloads (a 650 KiB peak, 98 KiB of which is the buffer
/// itself). Carried at 4 MiB, which is seven times that: enough headroom for
/// the merge's extra passes above a 128-run fan-in and for a different
/// allocator's rounding, and not so much that a writer which started holding
/// the whole entry list would slip under it.
const FIXED_OVERHEAD_BYTES: u64 = 4 * 1024 * 1024;

/// The writer's whole memory bound: the sort buffer the caller asked for, the
/// dedupe budget the caller asked for, and a constant.
///
/// There is **no term in the number of distinct payloads**, which is the
/// entire point of issue #1140. The term that used to be here,
/// `PAYLOAD_TABLE_BYTES * distinct_payloads`, made the bound scale with the
/// quantity it exists to bound, so the assertion could not fail for the one
/// thing it was written about. Anything this writer spends per payload now
/// shows up as a failure rather than as a wider ceiling.
fn bound_for(sort_buffer_records: usize, dedupe_memory_bytes: usize) -> u64 {
    SORT_BUFFER_SLACK * SPILL_BYTES_IN_MEMORY * sort_buffer_records as u64
        + dedupe_memory_bytes as u64
        + FIXED_OVERHEAD_BYTES
}

// ---------------------------------------------------------------------------
// A sink that keeps the position and drops the bytes
// ---------------------------------------------------------------------------

/// A `Write + Seek` that tracks where it is and throws the bytes away.
///
/// `Writer::try_new` takes any `W: Write + Seek`, and a bounded-memory
/// measurement should not also be a measurement of how fast this machine's
/// filesystem is. Dropping the output keeps the run honest in the one way that
/// matters here: the writer still performs every copy, every seek and every
/// offset computation, and the archive's length is still tracked, so the
/// header it emits is the header it would have emitted to a file.
#[derive(Debug, Default)]
struct Discard {
    position: u64,
    length: u64,
}

impl Write for Discard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.position += buf.len() as u64;
        self.length = self.length.max(self.position);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for Discard {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let next = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(n) => self.length as i128 + i128::from(n),
            SeekFrom::Current(n) => self.position as i128 + i128::from(n),
        };
        if next < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before the start of the sink",
            ));
        }
        self.position = next as u64;
        self.length = self.length.max(self.position);
        Ok(self.position)
    }
}

// ---------------------------------------------------------------------------
// One measured write
// ---------------------------------------------------------------------------

/// What one measured write reports.
#[derive(Debug)]
struct Measured {
    /// Peak live heap over the baseline, across the whole write.
    peak: u64,
    /// Peak live heap over the live total at the instant `finish` was called,
    /// so it is what finalize itself added on top of what the add phase left
    /// resident.
    finalize_growth: u64,
    /// Live heap the add phase left resident at the instant `finish` was
    /// called. This is the half the dedupe window and the payload table live
    /// in, and it is invisible in `finalize_growth`, which subtracts it.
    add_phase_live: u64,
    header: Header,
}

/// Write `tiles` tiles cycling over `distinct` payloads, and report what it
/// cost.
///
/// The payloads cycle rather than repeat, so consecutive tile ids carry
/// different bytes and the directory cannot collapse the whole pyramid into
/// one run-length entry. A pyramid of identical tiles is one entry and no leaf
/// directories at any size, which would make every number here a measurement
/// of nothing.
fn measure_write(tiles: u64, distinct: u64, sort_buffer_records: usize) -> Measured {
    let scratch = tempfile::tempdir().expect("a scratch directory");

    let payloads: Vec<(Vec<u8>, [u8; 32])> = (0..distinct)
        .map(|i| {
            let mut bytes = vec![0u8; 64];
            bytes[..8].copy_from_slice(&i.to_le_bytes());
            let hash = content_hash(&bytes);
            (bytes, hash)
        })
        .collect();

    // The smallest zoom whose grid addresses `tiles`, so one zoom holds the
    // whole profile and the tile ids stay contiguous. z=9 is a 512x512 grid
    // and covers 262144, which is every cheap profile here; the two-million
    // cell lands on z=11.
    let zoom: u8 = (9..=14)
        .find(|z| (1u64 << z) * (1u64 << z) >= tiles)
        .unwrap_or_else(|| panic!("no zoom up to 14 addresses {tiles} tiles"));
    let side: u64 = 1 << zoom;

    let options = WriterOptions::default()
        .with_tile_type(TileType::Png)
        .with_sort_buffer_records(sort_buffer_records)
        .with_dedupe_memory_bytes(DEDUPE_BUDGET_BYTES);

    let (_guard, baseline) = start_measuring();

    let mut writer =
        Writer::try_new(Discard::default(), scratch.path(), options).expect("the writer opens");
    for index in 0..tiles {
        let x = (index % side) as u32;
        let y = (index / side) as u32;
        let (bytes, hash) = &payloads[(index % distinct) as usize];
        writer
            .add_tile(zoom, x, y, bytes, *hash)
            .expect("every tile is inside the grid");
    }
    let before_finalize = live_over(baseline);
    let finished = writer.finish().expect("the archive finishes");
    let peak = peak_over(baseline);

    Measured {
        peak,
        finalize_growth: peak.saturating_sub(before_finalize),
        add_phase_live: before_finalize,
        header: finished.header,
    }
}

/// Assert the run really produced the archive it claimed to.
///
/// Without this every bound below is also satisfied by a writer that dropped
/// the tiles on the floor, which is the cheapest possible way to be bounded.
fn assert_really_wrote(measured: &Measured, tiles: u64, distinct: u64) {
    assert_eq!(
        measured.header.addressed_tiles_count, tiles,
        "the header should address every tile that was added"
    );
    assert_eq!(
        measured.header.tile_contents_count, distinct,
        "the header should count one blob per distinct payload"
    );
    assert!(
        measured.header.tile_entries_count > 0,
        "an archive with tiles has entries"
    );
}

// ---------------------------------------------------------------------------
// The three assertions
// ---------------------------------------------------------------------------

/// Four times the tiles, same sort buffer, same payloads: the peak must not
/// follow the tile count.
#[test]
#[cfg_attr(miri, ignore)]
fn the_finalize_peak_does_not_move_when_the_tile_count_quadruples() {
    const RECORDS: usize = 4096;
    const DISTINCT: u64 = 3;

    let small = measure_write(65_536, DISTINCT, RECORDS);
    let large = measure_write(262_144, DISTINCT, RECORDS);

    assert_really_wrote(&small, 65_536, DISTINCT);
    assert_really_wrote(&large, 262_144, DISTINCT);
    println!(
        "tile-count independence: 65536 tiles peak={} finalize=+{}, 262144 tiles peak={} finalize=+{}",
        small.peak, small.finalize_growth, large.peak, large.finalize_growth
    );

    // Four times the tiles may cost a little: the run table is 16 bytes per
    // spilled run and the tile count decides how many runs there are. At this
    // sort buffer that is 16 runs against 64, so 768 bytes. Measured, the two
    // peaks differ by under 2 KB. The slack is 256 KiB, over a hundred times
    // the measured difference and eighteen times under the 4.7 MB a peak that
    // really tracked the tile count would add at 24 bytes a record.
    let slack = 256 * 1024;
    assert!(
        large.peak <= small.peak + slack,
        "the peak followed the tile count: {} tiles peaked at {} bytes, {} tiles at {} bytes",
        65_536,
        small.peak,
        262_144,
        large.peak
    );
}

/// Sixty-four times the sort buffer, same tiles: the peak must follow the
/// buffer.
///
/// This is the control on the test above. A writer that allocated one fixed
/// megabyte and streamed everything would pass "does not follow the tile
/// count" perfectly while having no sort buffer at all, and this is the
/// assertion it fails.
#[test]
#[cfg_attr(miri, ignore)]
fn the_finalize_peak_grows_with_the_sort_buffer_it_was_given() {
    const TILES: u64 = 262_144;
    const DISTINCT: u64 = 3;
    const SMALL: usize = 4_096;
    const LARGE: usize = 262_144;

    let small = measure_write(TILES, DISTINCT, SMALL);
    let large = measure_write(TILES, DISTINCT, LARGE);

    assert_really_wrote(&small, TILES, DISTINCT);
    assert_really_wrote(&large, TILES, DISTINCT);
    println!(
        "sort-buffer scaling: {SMALL} records peak={}, {LARGE} records peak={}, delta={}",
        small.peak,
        large.peak,
        large.peak.saturating_sub(small.peak)
    );

    // The buffer itself is 24 bytes a record, so the difference between the
    // two is at least 6 MiB of `Vec`. Half of that is the floor, because the
    // larger buffer also spills 64 times fewer runs and so gives some back.
    let buffer_delta = SPILL_BYTES_IN_MEMORY * (LARGE - SMALL) as u64;
    assert!(
        large.peak >= small.peak + buffer_delta / 2,
        "a 64x sort buffer moved the peak by only {} bytes ({} -> {}), which is less than half \
         of the {buffer_delta} bytes the buffer itself costs",
        large.peak.saturating_sub(small.peak),
        small.peak,
        large.peak
    );
}

/// The absolute peak, against the formula the module documents.
#[test]
#[cfg_attr(miri, ignore)]
fn the_finalize_peak_stays_under_the_stated_bound() {
    const TILES: u64 = 262_144;
    const DISTINCT: u64 = 257;
    const RECORDS: usize = 4_096;

    let measured = measure_write(TILES, DISTINCT, RECORDS);
    assert_really_wrote(&measured, TILES, DISTINCT);

    let bound = bound_for(RECORDS, DEDUPE_BUDGET_BYTES);
    println!(
        "absolute bound: peak={} finalize=+{} bound={bound} ({RECORDS} records, {DISTINCT} distinct)",
        measured.peak, measured.finalize_growth
    );
    assert!(
        measured.peak <= bound,
        "peak {} bytes over a bound of {bound} bytes for {RECORDS} sort-buffer records and \
         {DISTINCT} distinct payloads",
        measured.peak
    );
    assert!(
        measured.finalize_growth <= bound,
        "finalize alone added {} bytes on top of the add phase, over a bound of {bound}",
        measured.finalize_growth
    );
}

// ---------------------------------------------------------------------------
// The two that issue #1140 exists for
// ---------------------------------------------------------------------------

/// Sixteen times the distinct payloads, same tiles, same everything else: what
/// the add phase leaves resident must not follow the payload count.
///
/// This is the half the old bound hid. The writer held a
/// `HashMap<[u8; 32], u64>` of every payload it had ever seen plus a
/// `payload_starts: Vec<u64>` beside it, both of them growing one entry per
/// distinct payload for the whole run, and the formula in `bound_for` carried
/// a term for exactly that. So the number went up, the ceiling went up with
/// it, and the assertion held.
///
/// Sixteen rather than four, because at four a real per-payload cost can still
/// hide under the constants.
#[test]
#[cfg_attr(miri, ignore)]
fn the_add_phase_footprint_does_not_grow_with_the_distinct_payload_count() {
    const RECORDS: usize = 4_096;
    const FEW: u64 = 16_384;
    const MANY: u64 = 262_144;

    let few = measure_write(FEW, FEW, RECORDS);
    let many = measure_write(MANY, MANY, RECORDS);

    assert_really_wrote(&few, FEW, FEW);
    assert_really_wrote(&many, MANY, MANY);
    println!(
        "add-phase footprint: {FEW} distinct live={}, {MANY} distinct live={}, delta={}",
        few.add_phase_live,
        many.add_phase_live,
        many.add_phase_live.saturating_sub(few.add_phase_live)
    );

    // 256 KiB of slack against a difference that was 20 MB before this epic:
    // the hash table alone was about 48 bytes an entry at `hashbrown`'s load
    // factor, so 246 thousand more payloads bought about 11 MB of it, and
    // `payload_starts` another 2 MB on top.
    let slack = 256 * 1024;
    assert!(
        many.add_phase_live <= few.add_phase_live + slack,
        "the add phase followed the payload count: {FEW} distinct payloads left {} bytes \
         resident and {MANY} left {} bytes",
        few.add_phase_live,
        many.add_phase_live
    );
}

/// The same sixteenfold step, measured on what finalize allocates on top.
///
/// `plan_entries` used to allocate a `final_offsets: Vec<u64>` and an
/// `order: Vec<u64>`, both one entry per distinct payload, *inside* `finish`
/// and on top of everything the add phase was still holding. Nothing in
/// `finish` reads the dedupe window, so that was three per-payload tables live
/// at once at the moment the writer is supposed to be at its cheapest.
#[test]
#[cfg_attr(miri, ignore)]
fn the_finalize_growth_does_not_follow_the_distinct_payload_count() {
    const RECORDS: usize = 4_096;
    const FEW: u64 = 16_384;
    const MANY: u64 = 262_144;

    let few = measure_write(FEW, FEW, RECORDS);
    let many = measure_write(MANY, MANY, RECORDS);

    assert_really_wrote(&few, FEW, FEW);
    assert_really_wrote(&many, MANY, MANY);
    println!(
        "finalize growth: {FEW} distinct +{}, {MANY} distinct +{}, delta={}",
        few.finalize_growth,
        many.finalize_growth,
        many.finalize_growth.saturating_sub(few.finalize_growth)
    );

    let slack = 256 * 1024;
    assert!(
        many.finalize_growth <= few.finalize_growth + slack,
        "finalize followed the payload count: {FEW} distinct payloads added {} bytes and \
         {MANY} added {} bytes",
        few.finalize_growth,
        many.finalize_growth
    );
}

/// The absolute bound again, at two million distinct payloads.
///
/// `the_finalize_peak_stays_under_the_stated_bound` runs at 257 distinct
/// payloads, which is small enough that a per-payload cost of a hundred bytes
/// is 25 KB and disappears under `FIXED_OVERHEAD_BYTES`. Two million is the
/// count issue #1140 names, and at the old cost it is about 200 MB against a
/// bound of 4.4 MB.
///
/// It is not `#[ignore]`d. It stages 128 MB through the scratch directory and
/// takes a few seconds, which is the price of the one cell where the number
/// this epic is about is big enough to see.
#[test]
#[cfg_attr(miri, ignore)]
fn the_finalize_peak_stays_under_the_stated_bound_at_two_million_payloads() {
    const TILES: u64 = 2_000_000;
    const RECORDS: usize = 4_096;

    let measured = measure_write(TILES, TILES, RECORDS);
    assert_really_wrote(&measured, TILES, TILES);

    let bound = bound_for(RECORDS, DEDUPE_BUDGET_BYTES);
    println!(
        "two million distinct payloads: peak={} finalize=+{} add-phase={} bound={bound}",
        measured.peak, measured.finalize_growth, measured.add_phase_live
    );
    assert!(
        measured.peak <= bound,
        "peak {} bytes over a bound of {bound} bytes at {TILES} distinct payloads",
        measured.peak
    );
}

// ---------------------------------------------------------------------------
// The opt-in profile
// ---------------------------------------------------------------------------

/// A real archive whose tile data section is past 4 GiB, finalized under the
/// same bound.
///
/// `#[ignore]` because it stages over 4 GiB through the scratch directory and
/// reads every byte of it back, which is minutes and gigabytes rather than the
/// second the three tests above cost. Run it with
/// `cargo test --test pmtiles_bounded_memory -- --ignored --nocapture`.
///
/// What it adds over the cheap tests is the `u32` boundary: entry offsets past
/// 4 294 967 295, a `tile_data_length` that does not fit a `u32`, and a header
/// that still reports them. The read half of the same boundary is proved for
/// free in `tests/pmtiles_index_only_reads.rs`.
#[test]
#[ignore = "stages over 4 GiB through the scratch directory; run with --ignored"]
#[cfg_attr(miri, ignore)]
fn an_archive_past_four_gibibytes_finalizes_in_bounded_memory() {
    const PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
    const PAYLOADS: u64 = 272; // 272 * 16 MiB = 4.25 GiB
    const RECORDS: usize = 4_096;

    let scratch = tempfile::tempdir().expect("a scratch directory");
    let options = WriterOptions::default()
        .with_tile_type(TileType::Png)
        .with_sort_buffer_records(RECORDS)
        .with_dedupe_memory_bytes(DEDUPE_BUDGET_BYTES);

    // One 16 MiB buffer, rewritten in place per tile, so the test's own
    // footprint is a constant and the measurement is of the writer.
    //
    // Allocated **before** the baseline is taken, which is the whole point of
    // where this line sits. It used to be below `start_measuring()`, so the
    // harness's own 16 MiB landed inside the measured peak and the bound had
    // to be padded to accommodate it: 54.8 MB of headroom over a writer
    // contributing 410368 bytes, a factor of 136, and a regression that made
    // the writer hold fifty megabytes would have passed.
    let mut payload = vec![0u8; PAYLOAD_BYTES];

    let (_guard, baseline) = start_measuring();

    let mut writer =
        Writer::try_new(Discard::default(), scratch.path(), options).expect("the writer opens");

    let zoom: u8 = 9;
    let side: u64 = 1 << zoom;
    for index in 0..PAYLOADS {
        payload[..8].copy_from_slice(&index.to_le_bytes());
        let hash = content_hash(&payload);
        let x = (index % side) as u32;
        let y = (index / side) as u32;
        writer
            .add_tile(zoom, x, y, &payload, hash)
            .expect("every tile is inside the grid");
    }
    let finished = writer.finish().expect("the archive finishes");
    let peak = peak_over(baseline);

    let expected = PAYLOADS * PAYLOAD_BYTES as u64;
    assert_eq!(
        finished.header.tile_data_length, expected,
        "the tile data section should be the sum of the distinct payloads"
    );
    assert!(
        finished.header.tile_data_length > u64::from(u32::MAX),
        "this profile exists to cross 4 GiB and only reached {}",
        finished.header.tile_data_length
    );
    assert_eq!(finished.header.addressed_tiles_count, PAYLOADS);

    // The same formula the cheap tests use, with nothing added for the
    // harness: the payload buffer is outside the measurement now, and the
    // writer copies payloads through a 64 KiB buffer rather than holding one.
    // Measured at 410368 bytes against this bound of 4425728, which is the
    // same order of headroom `FIXED_OVERHEAD_BYTES` carries everywhere else
    // and eleven times tighter than the 136 this test used to allow.
    let bound = bound_for(RECORDS, DEDUPE_BUDGET_BYTES);
    assert!(
        peak <= bound,
        "a 4.25 GiB archive peaked at {peak} bytes, over a bound of {bound}"
    );

    // And the harness's own buffer really is outside the measurement, so the
    // bound above is not quietly paying for it again.
    assert!(
        peak < PAYLOAD_BYTES as u64,
        "the measured peak {peak} is at least the {PAYLOAD_BYTES} byte payload buffer, so the          harness's own allocation is inside the measurement"
    );

    println!(
        "4 GiB profile: tile_data_length={} peak_heap_bytes={peak} bound={bound}",
        finished.header.tile_data_length
    );
}

/// The flat index this file writes really covers distinct tiles.
///
/// A guard on the harness rather than on the writer: `measure_write` maps a
/// flat index onto `(x, y)` by hand, and a mapping that wrapped would hand the
/// same coordinate in twice. `finish` refuses a duplicate, so a wrapped
/// mapping fails loudly rather than silently, but it would fail as "the writer
/// is broken" rather than as "the harness is".
///
/// The ids are deliberately not asserted to ascend with the index. PMTiles
/// orders a zoom by its Hilbert curve, so row-major `(x, y)` walks that curve
/// out of order, and the writer sorting them is the whole reason it has a sort
/// buffer to bound.
#[test]
fn the_flat_index_maps_onto_distinct_tiles() {
    let _guard = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let zoom: u8 = 9;
    let side: u64 = 1 << zoom;
    let mut seen = std::collections::BTreeSet::new();
    for index in [0u64, 1, side - 1, side, side + 1, side * side - 1] {
        let x = (index % side) as u32;
        let y = (index / side) as u32;
        assert!(
            u64::from(x) < side && u64::from(y) < side,
            "inside the grid"
        );
        let id = zxy_to_tileid(zoom, x, y).expect("inside the grid");
        assert!(seen.insert(id), "the flat index handed {id} in twice");
    }
    assert_eq!(seen.len(), 6, "six indices should map onto six tiles");
}
