//! Where the concurrent p99 tail on the leaf-bearing archive goes (issue #1021).
//!
//! #1021 reports that at eight threads on `8192x8192@64`, the one cell whose
//! archive has leaf directories, the PMTiles reader has a p99 the directory
//! backend does not: 62.08 / 38.67 / 43.58 us against 7.62 / 8.50 / 9.08,
//! while the p50 still goes the other way at 2.21 against 4.00. It names a
//! suspect, the `Mutex` around the leaf cache, and it is careful to say the
//! suspect is unproved because it measured **latency** and not **lock wait**.
//!
//! A latency number cannot tell those apart. So this measures both, on the
//! same lookups, and attributes each lookup's time.
//!
//! # Why attribution rather than another A/B
//!
//! The harness has a measured noise floor and it is savage on exactly this
//! statistic: both benchmark profiles walk `2048x2048@256`, so the committed
//! sweep measures one cell twice on an idle host with identical code, and
//! across that replicate pair p99 moves by up to 74.5% (wall 10.7%, p50 7.1%,
//! peak RSS 0.05%). Any finding of the form "config A's p99 is bigger than
//! config B's" has to clear that, and a single pair clears nothing.
//!
//! The way out is to stop comparing runs. Every lookup here carries its own
//! total latency, the nanoseconds it spent blocked on the leaf-cache mutex,
//! the nanoseconds it spent holding it and the nanoseconds it spent inside
//! `RangeReader::read_range`. So the question "what is the tail made of" is
//! answered **inside one run**, by taking the slowest 1% of lookups and adding
//! up where their microseconds went. Run-to-run dispersion cannot move a share
//! that is computed from the same lookups it describes, and the replicate
//! rounds below are there to show the share itself is stable rather than to
//! carry the argument.
//!
//! # The four arms
//!
//! * `pm-shared`: one `Reader`, N threads, which is exactly what
//!   `concurrent_pass` in `tests/pmtiles_benchmarks.rs` does and therefore the
//!   configuration the issue measured.
//! * `pm-perthread`: N threads, N `Reader`s over the same file. Same reads,
//!   same decode, same allocation, and every mutex is private to one thread so
//!   it can never be contended. This is the ablation: if the tail is the lock,
//!   it has to go away here.
//! * `dir`: the directory backend, for the baseline the tail is defined
//!   against.
//! * `pm-deep`: a second archive built to hold about sixty leaf directories
//!   instead of six, so the cache fills to its `MAX_CACHED_LEAVES` bound and
//!   the linear scan and the `remove` plus `insert(0)` reorder run at their
//!   full length. The shipped default is not touched: the *archive* changes,
//!   not the constant. This is the honest way to ask what the 16-to-64 cache
//!   sizing did to the reorder, because on the real cell the cache never holds
//!   more than six slots and so never reaches the 64 the claim is about.
//!
//! # Running it
//!
//! ```text
//! LIBVIPRS_PROBE_DIR=/work/probe LIBVIPRS_PROBE_OUT=/work/probe/out.jsonl \
//!   cargo test --release --test pmtiles_lock_probe -- --ignored --nocapture
//! ```
//!
//! With `RUSTFLAGS='--cfg pmtiles_lock_probe'` the blocked and held columns
//! are real; without it they are zero and the run still produces the latency
//! and `read_range` columns, which is the control that the instrumentation is
//! not itself making the tail.

use std::cell::Cell;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pmtiles::{FileRangeReader, RangeReader, Reader, Writer, WriterOptions};
use libviprs::pyramid_reader::{DirectoryPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::sink_pmtiles::{PmTilesSink, tile_coord_to_zxy};
use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};

/// The leaf-bearing cell, the only one of the sweep's four whose archive has
/// leaf directories at all and the only one the tail shows up on.
const CELL: (u32, u32, u32) = (8192, 8192, 64);

/// What the large profile's read scenarios sample, from `Profile::read_samples`.
const READ_SAMPLES: usize = 20_000;

/// The seed `read_scenarios` uses for its random coordinate list, so this walks
/// the same coordinates in the same order as the row being investigated.
const RANDOM_SEED: u64 = 0x5EED_1234_ABCD_0001;

// ---------------------------------------------------------------------------
// Timing a lookup's reads from outside the crate
// ---------------------------------------------------------------------------

thread_local! {
    /// `(calls, nanoseconds)` this thread has spent inside `read_range`.
    static READS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

fn reads_snapshot() -> (u64, u64) {
    READS.with(Cell::get)
}

/// A `RangeReader` that times the real one.
///
/// This is the half of the attribution that needs no product code at all: the
/// transport is a public trait and `Reader::try_new` takes any implementation
/// of it, so the pread can be timed from a test. The lock cannot, which is why
/// the other half is `cfg`-gated instrumentation inside `reader.rs`.
struct TimedSource(FileRangeReader);

impl RangeReader for TimedSource {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let at = Instant::now();
        let out = self.0.read_range(offset, len);
        let ns = at.elapsed().as_nanos() as u64;
        READS.with(|cell| {
            let (calls, total) = cell.get();
            cell.set((calls + 1, total + ns));
        });
        out
    }

    fn size(&self) -> io::Result<Option<u64>> {
        self.0.size()
    }
}

// ---------------------------------------------------------------------------
// One measured lookup
// ---------------------------------------------------------------------------

/// Everything one lookup cost, split by where it went.
#[derive(Clone, Copy, Default)]
struct Rec {
    lat_ns: u64,
    read_ns: u64,
    reads: u64,
    blocked_ns: u64,
    held_ns: u64,
    acquisitions: u64,
}

#[cfg(pmtiles_lock_probe)]
fn lock_snapshot() -> libviprs::pmtiles::reader::lock_probe::Stats {
    libviprs::pmtiles::reader::lock_probe::snapshot()
}

/// What the lock probe saw, as `(blocked_ns, held_ns, acquisitions)`, or zeros
/// in a build that does not carry it.
#[cfg(pmtiles_lock_probe)]
fn lock_delta(
    before: libviprs::pmtiles::reader::lock_probe::Stats,
    after: libviprs::pmtiles::reader::lock_probe::Stats,
) -> (u64, u64, u64) {
    (
        after.blocked_ns - before.blocked_ns,
        after.held_ns - before.held_ns,
        after.acquisitions - before.acquisitions,
    )
}

/// The stand-in for a snapshot in a build with no probe in it.
///
/// A named zero-sized type rather than `()`, because `let before = f();` on a
/// unit-returning function is `clippy::let_unit_value` and the lint job denies
/// warnings. The call sites then read identically under both cfgs.
#[cfg(not(pmtiles_lock_probe))]
#[derive(Clone, Copy)]
struct NoLockStats;

#[cfg(not(pmtiles_lock_probe))]
fn lock_snapshot() -> NoLockStats {
    NoLockStats
}

#[cfg(not(pmtiles_lock_probe))]
fn lock_delta(_before: NoLockStats, _after: NoLockStats) -> (u64, u64, u64) {
    (0, 0, 0)
}

/// What the leaf cache looked like while a pass ran.
///
/// This is the half of the hypothesis that is a claim about a *shape* rather
/// than about time: "a hit memmoves up to 64 slots". `max_depth` is how many
/// slots the cache ever held, `scanned` is how many it walked to find a hit
/// and `reordered` is how many the `remove` plus `insert(0)` shifted.
#[derive(Clone, Copy, Default)]
struct Shape {
    max_depth: u64,
    scanned: u64,
    reordered: u64,
    hits: u64,
    /// Times a worker thread gave up its CPU of its own accord, summed over
    /// the threads of one pass. An uncontended `std::sync::Mutex` is a pair of
    /// atomics and no syscall, so a thread only parks when it really has to
    /// wait, which makes this an independent witness on the lock that owes
    /// nothing to the instrumentation and is therefore readable in the
    /// uninstrumented build too.
    voluntary: u64,
    involuntary: u64,
}

impl Shape {
    fn merge(self, other: Self) -> Self {
        Self {
            max_depth: self.max_depth.max(other.max_depth),
            scanned: self.scanned + other.scanned,
            reordered: self.reordered + other.reordered,
            hits: self.hits + other.hits,
            voluntary: self.voluntary + other.voluntary,
            involuntary: self.involuntary + other.involuntary,
        }
    }

    /// This thread's own switch counts.
    ///
    /// `/proc/thread-self` and not `/proc/self`: the latter answers for the
    /// thread group leader, which in this test is the thread that spawned the
    /// workers and then blocked in `join`, so it reported zero for every arm
    /// including the ones that were parking thousands of times. A zero that
    /// cannot move is not a measurement.
    fn with_thread_switches(mut self) -> Self {
        if let Ok(status) = std::fs::read_to_string("/proc/thread-self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("voluntary_ctxt_switches:") {
                    self.voluntary = rest.trim().parse().unwrap_or(0);
                } else if let Some(rest) = line.strip_prefix("nonvoluntary_ctxt_switches:") {
                    self.involuntary = rest.trim().parse().unwrap_or(0);
                }
            }
        }
        self
    }
}

#[cfg(pmtiles_lock_probe)]
fn shape_now() -> Shape {
    let stats = libviprs::pmtiles::reader::lock_probe::snapshot();
    Shape {
        max_depth: stats.max_depth,
        scanned: stats.scanned,
        reordered: stats.reordered,
        hits: stats.hits,
        ..Shape::default()
    }
}

#[cfg(not(pmtiles_lock_probe))]
fn shape_now() -> Shape {
    Shape::default()
}

/// Run `coords` over `threads` threads, one object per thread taken from
/// `targets` (a one-element slice shares one object with every thread).
fn timed_pass<T: Sync>(
    targets: &[T],
    coords: &[TileCoord],
    threads: usize,
    lookup: &(dyn Fn(&T, TileCoord) -> usize + Sync),
) -> (Vec<Rec>, Shape) {
    let chunk = coords.len().div_ceil(threads.max(1));
    let parts: Vec<(Vec<Rec>, Shape)> = std::thread::scope(|scope| {
        let handles: Vec<_> = coords
            .chunks(chunk.max(1))
            .enumerate()
            .map(|(index, slice)| {
                let target = &targets[index % targets.len()];
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(slice.len());
                    for coord in slice {
                        let reads_before = reads_snapshot();
                        let lock_before = lock_snapshot();
                        let at = Instant::now();
                        let bytes = lookup(target, *coord);
                        let lat_ns = at.elapsed().as_nanos() as u64;
                        let lock_after = lock_snapshot();
                        let reads_after = reads_snapshot();
                        assert!(bytes > 0, "a planned coordinate is present");
                        let (blocked_ns, held_ns, acquisitions) =
                            lock_delta(lock_before, lock_after);
                        out.push(Rec {
                            lat_ns,
                            read_ns: reads_after.1 - reads_before.1,
                            reads: reads_after.0 - reads_before.0,
                            blocked_ns,
                            held_ns,
                            acquisitions,
                        });
                    }
                    // Read after the loop: a scoped thread starts with its
                    // thread-locals at zero, so this is this pass's own total
                    // rather than a running one.
                    (out, shape_now().with_thread_switches())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a probe thread does not panic"))
            .collect()
    });
    let mut recs = Vec::with_capacity(coords.len());
    let mut shape = Shape::default();
    for (part, seen) in parts {
        recs.extend(part);
        shape = shape.merge(seen);
    }
    (recs, shape)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Nearest-rank percentile, the same rule `percentile_micros_at` uses in the
/// benchmark harness, so a number here is comparable with a number there.
fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() as f64) * p).ceil() as usize;
    let index = rank.clamp(1, sorted.len()) - 1;
    sorted[index] as f64 / 1000.0
}

fn quantiles(values: &mut [u64]) -> String {
    values.sort_unstable();
    format!(
        "{{\"p50\":{:.3},\"p90\":{:.3},\"p99\":{:.3},\"p999\":{:.3},\"max\":{:.3}}}",
        percentile(values, 0.50),
        percentile(values, 0.90),
        percentile(values, 0.99),
        percentile(values, 0.999),
        percentile(values, 1.0),
    )
}

fn mean_us(values: impl Iterator<Item = u64>, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    values.sum::<u64>() as f64 / count as f64 / 1000.0
}

/// The line one replicate contributes.
///
/// The tail block is the finding: the slowest 1% of lookups, and what their
/// microseconds were spent on. `blocked_share` is the fraction of the tail's
/// total latency that went into waiting for the leaf-cache mutex, which is the
/// number the hypothesis lives or dies on.
fn summarise(
    arm: &str,
    threads: usize,
    round: usize,
    recs: &[Rec],
    shape: Shape,
    note: &str,
) -> String {
    let mut lat: Vec<u64> = recs.iter().map(|r| r.lat_ns).collect();
    let mut blocked: Vec<u64> = recs.iter().map(|r| r.blocked_ns).collect();
    let mut held: Vec<u64> = recs.iter().map(|r| r.held_ns).collect();
    let mut read: Vec<u64> = recs.iter().map(|r| r.read_ns).collect();

    let lat_block = quantiles(&mut lat);
    let blocked_block = quantiles(&mut blocked);
    let held_block = quantiles(&mut held);
    let read_block = quantiles(&mut read);

    let mut by_latency: Vec<Rec> = recs.to_vec();
    by_latency.sort_unstable_by_key(|r| r.lat_ns);
    let tail_from = by_latency.len() - by_latency.len() / 100;
    let tail = &by_latency[tail_from..];
    let tail_lat: u64 = tail.iter().map(|r| r.lat_ns).sum();
    let tail_blocked: u64 = tail.iter().map(|r| r.blocked_ns).sum();
    let tail_held: u64 = tail.iter().map(|r| r.held_ns).sum();
    let tail_read: u64 = tail.iter().map(|r| r.read_ns).sum();

    let all_lat: u64 = recs.iter().map(|r| r.lat_ns).sum();
    let all_blocked: u64 = recs.iter().map(|r| r.blocked_ns).sum();
    let all_held: u64 = recs.iter().map(|r| r.held_ns).sum();
    let all_read: u64 = recs.iter().map(|r| r.read_ns).sum();
    let acquisitions: u64 = recs.iter().map(|r| r.acquisitions).sum();
    let reads: u64 = recs.iter().map(|r| r.reads).sum();

    let share = |part: u64, whole: u64| {
        if whole == 0 {
            0.0
        } else {
            part as f64 / whole as f64
        }
    };

    format!(
        "{{\"arm\":\"{arm}\",\"threads\":{threads},\"round\":{round},\"n\":{n},\
\"lat_us\":{lat_block},\"blocked_us\":{blocked_block},\"held_us\":{held_block},\
\"read_us\":{read_block},\
\"tail\":{{\"n\":{tail_n},\"lat_mean_us\":{tail_lat_mean:.3},\
\"blocked_mean_us\":{tail_blocked_mean:.3},\"held_mean_us\":{tail_held_mean:.3},\
\"read_mean_us\":{tail_read_mean:.3},\"blocked_share\":{tail_blocked_share:.4},\
\"held_share\":{tail_held_share:.4},\"read_share\":{tail_read_share:.4}}},\
\"whole\":{{\"blocked_share\":{whole_blocked_share:.4},\"held_share\":{whole_held_share:.4},\
\"read_share\":{whole_read_share:.4},\"acq_per_lookup\":{acq:.3},\
\"reads_per_lookup\":{rd:.3}}},\
\"cache\":{{\"max_depth\":{depth},\"scanned_per_hit\":{scan:.2},\
\"moved_per_hit\":{moved:.2},\"hits\":{hits}}},\
\"parks\":{{\"voluntary\":{vol},\"involuntary\":{invol}}},\"note\":\"{note}\"}}",
        n = recs.len(),
        tail_n = tail.len(),
        tail_lat_mean = mean_us(tail.iter().map(|r| r.lat_ns), tail.len()),
        tail_blocked_mean = mean_us(tail.iter().map(|r| r.blocked_ns), tail.len()),
        tail_held_mean = mean_us(tail.iter().map(|r| r.held_ns), tail.len()),
        tail_read_mean = mean_us(tail.iter().map(|r| r.read_ns), tail.len()),
        tail_blocked_share = share(tail_blocked, tail_lat),
        tail_held_share = share(tail_held, tail_lat),
        tail_read_share = share(tail_read, tail_lat),
        whole_blocked_share = share(all_blocked, all_lat),
        whole_held_share = share(all_held, all_lat),
        whole_read_share = share(all_read, all_lat),
        acq = acquisitions as f64 / recs.len().max(1) as f64,
        rd = reads as f64 / recs.len().max(1) as f64,
        depth = shape.max_depth,
        scan = shape.scanned as f64 / shape.hits.max(1) as f64,
        moved = shape.reordered as f64 / shape.hits.max(1) as f64,
        hits = shape.hits,
        vol = shape.voluntary,
        invol = shape.involuntary,
    )
}

// ---------------------------------------------------------------------------
// The pyramids
// ---------------------------------------------------------------------------

fn plan_for(width: u32, height: u32, tile_size: u32) -> PyramidPlan {
    PyramidPlanner::new(width, height, tile_size, 0, Layout::Xyz)
        .expect("a plan is valid")
        .plan()
}

fn coordinates(plan: &PyramidPlan) -> Vec<TileCoord> {
    let mut out = Vec::new();
    for level in &plan.levels {
        for row in 0..level.rows {
            for col in 0..level.cols {
                out.push(TileCoord {
                    level: level.level,
                    col,
                    row,
                });
            }
        }
    }
    out
}

/// The benchmark harness's `Splitmix`, repeated here rather than imported so
/// this file does not build on `tests/common/pmtiles_bench.rs`, which another
/// lane is editing for the same issue.
struct Splitmix(u64);

impl Splitmix {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let value = z ^ (z >> 31);
        if bound == 0 {
            0
        } else {
            (value % bound as u64) as usize
        }
    }
}

fn gradient(width: u32, height: u32) -> Raster {
    let mut data = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height {
        for x in 0..width {
            let off = (y as usize * width as usize + x as usize) * 3;
            data[off] = (x % 251) as u8;
            data[off + 1] = (y % 241) as u8;
            data[off + 2] = ((x * 7 + y * 13) % 239) as u8;
        }
    }
    Raster::new(width, height, PixelFormat::Rgb8, data).expect("a gradient raster is well formed")
}

/// Build the archive and the tree the sweep builds, once, and keep them.
///
/// Generation is the expensive half and it produces the same bytes every time,
/// so it is cached in the probe directory. Both builds of this test, the plain
/// one and the instrumented one, then measure the identical files.
fn ensure_pyramids(dir: &Path, plan: &PyramidPlan) -> (PathBuf, PathBuf) {
    let archive = dir.join("pyramid.pmtiles");
    let tree = dir.join("tree");
    if archive.is_file() && tree.is_dir() {
        println!("probe: reusing the pyramids already in {}", dir.display());
        return (archive, tree);
    }
    std::fs::create_dir_all(dir).expect("the probe directory is writable");
    let source = gradient(CELL.0, CELL.1);
    if !archive.is_file() {
        let sink = PmTilesSink::builder(&archive)
            .plan(plan.clone())
            .tile_format(TileFormat::Png)
            .build()
            .expect("the archive sink builds");
        EngineBuilder::new(&source, plan.clone(), sink)
            .run()
            .expect("the archive run succeeds");
    }
    if !tree.is_dir() {
        let sink = FsSink::new(&tree, plan.clone()).with_format(TileFormat::Png);
        EngineBuilder::new(&source, plan.clone(), sink)
            .run()
            .expect("the directory run succeeds");
    }
    (archive, tree)
}

/// A second archive over the same coordinates whose leaves are small enough
/// that there are about sixty of them.
///
/// The point is the cache depth and nothing else: `MAX_CACHED_LEAVES` is 64,
/// the real cell has six leaves and therefore never fills more than six slots,
/// and the claim under investigation is about a hit memmoving up to 64. So the
/// archive moves and the constant does not. Payloads are distinct by
/// construction so nothing is deduplicated and the entry count is the tile
/// count, which makes the leaf count predictable.
fn ensure_deep_archive(dir: &Path, coords: &[TileCoord], leaf_entries: usize) -> PathBuf {
    let archive = dir.join(format!("deep-{leaf_entries}.pmtiles"));
    if archive.is_file() {
        return archive;
    }
    let options = WriterOptions::default().with_leaf_entries(leaf_entries);
    let mut writer = Writer::create(&archive, options).expect("the deep writer opens");
    let mut payload = vec![0u8; 1500];
    for (index, coord) in coords.iter().enumerate() {
        let (z, x, y) = tile_coord_to_zxy(*coord).expect("a planned coordinate is addressable");
        for (at, byte) in payload.iter_mut().enumerate() {
            *byte = ((index * 31 + at * 17) % 251) as u8;
        }
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&(index as u64).to_le_bytes());
        writer
            .add_tile(z, x, y, &payload, hash)
            .expect("a tile is added");
    }
    writer.finish().expect("the deep archive finishes");
    archive
}

// ---------------------------------------------------------------------------
// Host conditions
// ---------------------------------------------------------------------------

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|line| {
            line.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|_| "unknown".to_string())
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(fallback)
}

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "a measurement, not an assertion: it generates two 21851-tile pyramids and runs millions of lookups"]
fn the_concurrent_tail_is_attributed() {
    let dir = PathBuf::from(
        std::env::var("LIBVIPRS_PROBE_DIR")
            .expect("LIBVIPRS_PROBE_DIR names the scratch directory"),
    );
    let rounds = env_usize("LIBVIPRS_PROBE_ROUNDS", 20);
    let thread_counts: Vec<usize> = std::env::var("LIBVIPRS_PROBE_THREADS")
        .unwrap_or_else(|_| "1,2,4,8".to_string())
        .split(',')
        .map(|raw| raw.trim().parse().expect("a thread count is a number"))
        .collect();
    let arms: Vec<String> = std::env::var("LIBVIPRS_PROBE_ARMS")
        .unwrap_or_else(|_| "pm-shared,pm-perthread,dir,pm-deep".to_string())
        .split(',')
        .map(|raw| raw.trim().to_string())
        .collect();
    let wants = |arm: &str| arms.iter().any(|name| name == arm);

    let plan = plan_for(CELL.0, CELL.1, CELL.2);
    let all = coordinates(&plan);
    assert_eq!(all.len(), 21_851, "the leaf-bearing cell plans 21851 tiles");
    let samples = READ_SAMPLES.min(all.len());
    let mut rng = Splitmix(RANDOM_SEED);
    let random: Vec<TileCoord> = (0..samples).map(|_| all[rng.below(all.len())]).collect();

    let (archive, tree) = ensure_pyramids(&dir, &plan);

    // Say what shape the thing being measured actually is, because the whole
    // hypothesis is a claim about how many slots the cache holds.
    let shape = Reader::try_open(&archive).expect("the archive opens");
    let root = shape.root_entries();
    let leaf_pointers = root.iter().filter(|entry| entry.is_leaf()).count();
    println!(
        "probe: {} root entries, {leaf_pointers} of them leaf pointers; MAX_CACHED_LEAVES is {}",
        root.len(),
        libviprs::pmtiles::reader::MAX_CACHED_LEAVES,
    );
    drop(shape);
    println!(
        "probe: cfg pmtiles_lock_probe is {}",
        if cfg!(pmtiles_lock_probe) {
            "ON"
        } else {
            "OFF (blocked and held columns will be zero)"
        }
    );
    println!(
        "probe: available_parallelism {:?}, loadavg {}",
        std::thread::available_parallelism(),
        loadavg()
    );

    let mut lines: Vec<String> = Vec::new();

    // The shared reader, timed transport: the harness's configuration.
    let shared = Reader::try_new(TimedSource(
        FileRangeReader::try_open(&archive).expect("the archive opens for ranged reads"),
    ))
    .expect("the shared reader opens");
    let shared_lookup = |reader: &Reader<TimedSource>, coord: TileCoord| {
        let (z, x, y) = tile_coord_to_zxy(coord).expect("a planned coordinate is addressable");
        reader
            .get_tile(z, x, y)
            .expect("a lookup succeeds")
            .map_or(0, |bytes| bytes.len())
    };

    // One reader per thread: the same work with every mutex private.
    let per_thread: Vec<Reader<TimedSource>> = (0..*thread_counts.iter().max().unwrap_or(&8))
        .map(|_| {
            Reader::try_new(TimedSource(
                FileRangeReader::try_open(&archive).expect("the archive opens for ranged reads"),
            ))
            .expect("a per-thread reader opens")
        })
        .collect();

    let directory = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens for reading");
    let dir_lookup = |reader: &DirectoryPyramidReader, coord: TileCoord| {
        reader
            .tile(coord)
            .expect("a lookup succeeds")
            .map_or(0, |bytes| bytes.len())
    };

    let deep = if wants("pm-deep") {
        let leaf_entries = env_usize("LIBVIPRS_PROBE_DEEP_LEAF_ENTRIES", 350);
        let path = ensure_deep_archive(&dir, &all, leaf_entries);
        let shape = Reader::try_open(&path).expect("the deep archive opens");
        let leaves = shape.root_entries().iter().filter(|e| e.is_leaf()).count();
        println!(
            "probe: deep archive at {leaf_entries} entries a leaf holds {} root entries, {leaves} leaf pointers",
            shape.root_entries().len()
        );
        drop(shape);
        let readers = vec![
            Reader::try_new(TimedSource(
                FileRangeReader::try_open(&path).expect("the deep archive opens for ranged reads"),
            ))
            .expect("the deep reader opens"),
        ];
        Some(readers)
    } else {
        None
    };

    // Warm every cache the way the harness does before its concurrent row: the
    // reader it hands to `concurrent_pass` has already served a cold pass, a
    // warm pass, a sequential pass and a random pass.
    let warm = all.clone();
    for _ in 0..2 {
        let _ = timed_pass(std::slice::from_ref(&shared), &warm, 1, &shared_lookup);
        for reader in &per_thread {
            let _ = timed_pass(std::slice::from_ref(reader), &warm, 1, &shared_lookup);
        }
        let _ = timed_pass(std::slice::from_ref(&directory), &random, 1, &dir_lookup);
        if let Some(readers) = &deep {
            let _ = timed_pass(&readers[..1], &warm, 1, &shared_lookup);
        }
    }

    // Rounds on the outside, arms on the inside, so host drift lands on every
    // arm rather than on whichever one ran last.
    for round in 0..rounds {
        for &threads in &thread_counts {
            if wants("pm-shared") {
                let (recs, shape) = timed_pass(
                    std::slice::from_ref(&shared),
                    &random,
                    threads,
                    &shared_lookup,
                );
                lines.push(summarise(
                    "pm-shared",
                    threads,
                    round,
                    &recs,
                    shape,
                    &format!("load {}", loadavg()),
                ));
            }
            if wants("pm-perthread") {
                let (recs, shape) =
                    timed_pass(&per_thread[..threads], &random, threads, &shared_lookup);
                lines.push(summarise("pm-perthread", threads, round, &recs, shape, ""));
            }
            if wants("dir") {
                let (recs, shape) = timed_pass(
                    std::slice::from_ref(&directory),
                    &random,
                    threads,
                    &dir_lookup,
                );
                lines.push(summarise("dir", threads, round, &recs, shape, ""));
            }
            if let Some(readers) = &deep {
                let (recs, shape) = timed_pass(&readers[..1], &random, threads, &shared_lookup);
                lines.push(summarise("pm-deep", threads, round, &recs, shape, ""));
            }
        }
        println!(
            "probe: round {} of {rounds} done, load {}",
            round + 1,
            loadavg()
        );
    }

    #[cfg(pmtiles_lock_probe)]
    {
        let stats = libviprs::pmtiles::reader::lock_probe::snapshot();
        println!(
            "probe: main-thread lock stats after everything: {stats:?} (the pass threads' own \
             totals are folded into the per-lookup columns)"
        );
    }

    for line in &lines {
        println!("PROBE {line}");
    }
    if let Ok(path) = std::env::var("LIBVIPRS_PROBE_OUT") {
        std::fs::write(&path, lines.join("\n") + "\n").expect("the probe output is writable");
        println!("probe: wrote {} lines to {path}", lines.len());
    }
}

#[test]
fn the_percentile_rule_is_the_harness_rule() {
    // Nearest rank over 100 samples: p50 is the 50th, p99 the 99th, p100 the
    // last. A probe whose statistics disagree with the harness's would be
    // comparing two different numbers and calling the difference a finding.
    let sorted: Vec<u64> = (1..=100).map(|n| n * 1000).collect();
    assert_eq!(percentile(&sorted, 0.50), 50.0);
    assert_eq!(percentile(&sorted, 0.99), 99.0);
    assert_eq!(percentile(&sorted, 1.0), 100.0);
    assert_eq!(percentile(&[], 0.99), 0.0);
}
