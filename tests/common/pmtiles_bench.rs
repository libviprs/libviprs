//! Measurement plumbing shared by the PMTiles benchmark and bounded-memory
//! guards (issue #993).
//!
//! Nothing here asserts anything. It is the part of a benchmark that has to be
//! the same across every scenario if the numbers are going to be comparable:
//! one clock, one memory basis, one way of counting what a backend left on the
//! filesystem, and one serialiser.
//!
//! # Two memory bases, never added together
//!
//! `tracked_memory_mb` is what the engine's own [`MemoryTracker`] charged, so
//! it counts raster buffers and nothing else. `peak_rss_mb` is the process
//! high-water mark the kernel reports, so it counts everything including the
//! allocator's slack and the page cache pages the process touched. They answer
//! different questions and libviprs-bench keeps them in separate columns for
//! exactly that reason (its issue #153), so this does too.
//!
//! The kernel number comes from `/proc/self/status`' `VmHWM`, which exists on
//! Linux and nowhere else this crate builds for. A macOS run reports `0.0`,
//! the same way a libvips row in the scalability data reports `0.0` tracked
//! memory: a column that platform cannot fill rather than a number invented to
//! fill it. The published figures come from the Linux container run.
//!
//! # The writer here and the reader that checks it are not the same code
//!
//! `serde_json` is already a dependency of this crate, so the shape guard in
//! `tests/pmtiles_benchmarks.rs` parses what this emits and asserts the field
//! set, the types and the record count. This side stays a hand-rolled
//! serialiser rather than a `Serialize` derive, for two reasons: it fixes the
//! field **order** to the one `scalability_results.json` already uses (a
//! `serde_json::Map` is a `BTreeMap` and would alphabetise them), and a
//! producer checked by an independent parser is worth more than a producer
//! checked by its own round trip. No dependency is added either way.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use libviprs::{PixelFormat, Raster};

/// Where a run's results land when `LIBVIPRS_BENCH_JSON` does not say.
pub const DEFAULT_RESULTS_PATH: &str = "target/pmtiles-benchmarks.json";

/// Which of the two profiles a benchmark run is on.
///
/// The CI profile has to be cheap enough that nobody is tempted to skip it;
/// the large one exists to show the shape holds two orders of magnitude up and
/// is opt-in through `LIBVIPRS_BENCH_PROFILE=large`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Ci,
    Large,
}

impl Profile {
    /// The profile the environment selects, defaulting to the cheap one.
    pub fn from_env() -> Self {
        match std::env::var("LIBVIPRS_BENCH_PROFILE").as_deref() {
            Ok("large") => Self::Large,
            _ => Self::Ci,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ci => "ci",
            Self::Large => "large",
        }
    }

    /// The source canvases a generation sweep walks, in pixels.
    ///
    /// The CI row is one canvas, because the comparison this issue is about is
    /// between two storage backends at one size rather than a scaling curve,
    /// and a second canvas doubles the cost of the cheap profile to say the
    /// same thing twice.
    pub fn canvases(self) -> &'static [(u32, u32)] {
        match self {
            Self::Ci => &[(2048, 2048)],
            Self::Large => &[(2048, 2048), (8192, 8192), (16384, 16384)],
        }
    }

    /// How many lookups a read scenario performs.
    pub fn read_samples(self) -> usize {
        match self {
            Self::Ci => 512,
            Self::Large => 20_000,
        }
    }
}

// ---------------------------------------------------------------------------
// One measured point
// ---------------------------------------------------------------------------

/// One row of the exported benchmark data.
///
/// The first twelve fields are `scalability_results.json`'s record shape,
/// spelled identically so the libviprs.org renderer and the `ScalabilityPoint`
/// deserialiser in libviprs-bench both read this file without a second code
/// path. Everything after `resource_cost` is additive: an unknown key is
/// ignored by `serde_json` and by the site's JavaScript, so adding them costs
/// the existing consumers nothing.
#[derive(Debug, Clone)]
pub struct Measurement {
    // --- the scalability_results.json shape ---
    pub width: u32,
    pub height: u32,
    pub megapixels: f64,
    /// `"pmtiles"` or `"directory"`. Named `engine` because that is the key
    /// the renderer groups on; the storage backend is what varies here.
    pub engine: String,
    pub concurrency: usize,
    pub wall_time_ms: f64,
    pub tracked_memory_mb: f64,
    pub peak_rss_mb: f64,
    pub tiles_produced: u64,
    pub tiles_per_second: f64,
    pub tiles_per_second_per_mb: f64,
    pub resource_cost: f64,

    // --- the PMTiles columns ---
    /// `"generate"`, `"read_cold"`, `"read_warm"`, `"read_random"`,
    /// `"read_sequential"` or `"read_concurrent"`.
    pub scenario: String,
    /// Same value as `engine`, under the name that says what it is.
    pub storage: String,
    /// `"ci"` or `"large"`.
    pub profile: String,
    /// Bytes the pyramid occupies: one archive, or the sum of every file in
    /// the tree.
    pub output_bytes: u64,
    /// Filesystem entries the pyramid occupies, directories included. This is
    /// the namespace-explosion column: one for an archive, one per tile plus
    /// the level and column directories for a tree.
    pub filesystem_entries: u64,
    /// Bytes the reader fetched to answer the scenario's lookups. Zero for a
    /// generation row.
    pub bytes_fetched: u64,
    /// Median and 99th-percentile per-lookup latency. Zero for a generation
    /// row.
    pub p50_latency_us: f64,
    pub p99_latency_us: f64,
}

impl Measurement {
    /// Build a row from the raw measurements, deriving the four ratios the
    /// same way `libviprs-bench`'s `scalability` binary derives them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scenario: &str,
        storage: &str,
        profile: Profile,
        width: u32,
        height: u32,
        concurrency: usize,
        elapsed: Duration,
        tracked_bytes: u64,
        rss_bytes: u64,
        tiles: u64,
    ) -> Self {
        let secs = elapsed.as_secs_f64();
        let tracked_mb = tracked_bytes as f64 / (1024.0 * 1024.0);
        let rss_mb = rss_bytes as f64 / (1024.0 * 1024.0);
        let tps = if secs > 0.0 { tiles as f64 / secs } else { 0.0 };
        // Both ratios use the RSS basis so a row here means what the same
        // column means in scalability_results.json.
        let tps_per_mb = if rss_mb > 0.0 { tps / rss_mb } else { 0.0 };
        let cost = if tiles > 0 {
            (rss_mb * secs) / tiles as f64
        } else {
            0.0
        };

        Self {
            width,
            height,
            megapixels: f64::from(width) * f64::from(height) / 1_000_000.0,
            engine: storage.to_string(),
            concurrency,
            wall_time_ms: secs * 1000.0,
            tracked_memory_mb: tracked_mb,
            peak_rss_mb: rss_mb,
            tiles_produced: tiles,
            tiles_per_second: tps,
            tiles_per_second_per_mb: tps_per_mb,
            resource_cost: cost,
            scenario: scenario.to_string(),
            storage: storage.to_string(),
            profile: profile.label().to_string(),
            output_bytes: 0,
            filesystem_entries: 0,
            bytes_fetched: 0,
            p50_latency_us: 0.0,
            p99_latency_us: 0.0,
        }
    }

    pub fn with_output(mut self, bytes: u64, entries: u64) -> Self {
        self.output_bytes = bytes;
        self.filesystem_entries = entries;
        self
    }

    pub fn with_latencies(mut self, samples: &mut [Duration], fetched: u64) -> Self {
        self.bytes_fetched = fetched;
        self.p50_latency_us = percentile_micros(samples, 0.50);
        self.p99_latency_us = percentile_micros(samples, 0.99);
        self
    }

    fn to_json(&self) -> String {
        let mut out = String::from("  {\n");
        push_u64(&mut out, "width", u64::from(self.width));
        push_u64(&mut out, "height", u64::from(self.height));
        push_f64(&mut out, "megapixels", self.megapixels);
        push_str(&mut out, "engine", &self.engine);
        push_u64(&mut out, "concurrency", self.concurrency as u64);
        push_f64(&mut out, "wall_time_ms", self.wall_time_ms);
        push_f64(&mut out, "tracked_memory_mb", self.tracked_memory_mb);
        push_f64(&mut out, "peak_rss_mb", self.peak_rss_mb);
        push_u64(&mut out, "tiles_produced", self.tiles_produced);
        push_f64(&mut out, "tiles_per_second", self.tiles_per_second);
        push_f64(
            &mut out,
            "tiles_per_second_per_mb",
            self.tiles_per_second_per_mb,
        );
        push_f64(&mut out, "resource_cost", self.resource_cost);
        push_str(&mut out, "scenario", &self.scenario);
        push_str(&mut out, "storage", &self.storage);
        push_str(&mut out, "profile", &self.profile);
        push_u64(&mut out, "output_bytes", self.output_bytes);
        push_u64(&mut out, "filesystem_entries", self.filesystem_entries);
        push_u64(&mut out, "bytes_fetched", self.bytes_fetched);
        push_f64(&mut out, "p50_latency_us", self.p50_latency_us);
        // The last field carries no trailing comma.
        out.push_str(&format!(
            "    \"p99_latency_us\": {}\n",
            json_number(self.p99_latency_us)
        ));
        out.push_str("  }");
        out
    }
}

/// Every field name a row carries, in order.
///
/// The shape guard reads this rather than repeating the list, so a field added
/// to [`Measurement::to_json`] and not here fails that guard instead of
/// quietly shipping.
pub const FIELDS: [&str; 20] = [
    "width",
    "height",
    "megapixels",
    "engine",
    "concurrency",
    "wall_time_ms",
    "tracked_memory_mb",
    "peak_rss_mb",
    "tiles_produced",
    "tiles_per_second",
    "tiles_per_second_per_mb",
    "resource_cost",
    "scenario",
    "storage",
    "profile",
    "output_bytes",
    "filesystem_entries",
    "bytes_fetched",
    "p50_latency_us",
    "p99_latency_us",
];

/// The first twelve, which are the ones an existing consumer already reads.
pub const SCALABILITY_FIELDS: [&str; 12] = [
    "width",
    "height",
    "megapixels",
    "engine",
    "concurrency",
    "wall_time_ms",
    "tracked_memory_mb",
    "peak_rss_mb",
    "tiles_produced",
    "tiles_per_second",
    "tiles_per_second_per_mb",
    "resource_cost",
];

// ---------------------------------------------------------------------------
// Serialising
// ---------------------------------------------------------------------------

fn push_u64(out: &mut String, key: &str, value: u64) {
    out.push_str(&format!("    \"{key}\": {value},\n"));
}

fn push_f64(out: &mut String, key: &str, value: f64) {
    out.push_str(&format!("    \"{key}\": {},\n", json_number(value)));
}

fn push_str(out: &mut String, key: &str, value: &str) {
    out.push_str(&format!("    \"{key}\": \"{}\",\n", escape(value)));
}

/// A finite JSON number, or `0` for one that is not.
///
/// JSON has no spelling for NaN or an infinity, and a benchmark that divided
/// by a zero duration would otherwise emit a document no parser accepts. A
/// zero is the same answer the existing scalability producer gives when its
/// denominator is zero.
fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:?}")
    } else {
        "0".to_string()
    }
}

fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Serialise a run's rows as the top-level array `scalability_results.json`
/// is.
pub fn to_json(rows: &[Measurement]) -> String {
    let body: Vec<String> = rows.iter().map(Measurement::to_json).collect();
    format!("[\n{}\n]\n", body.join(",\n"))
}

/// Where the results of this run go.
pub fn results_path() -> PathBuf {
    match std::env::var("LIBVIPRS_BENCH_JSON") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_RESULTS_PATH),
    }
}

/// Write a run's rows out, creating the parent directory if it is missing.
pub fn write_results(rows: &[Measurement]) -> PathBuf {
    let path = results_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("the results directory can be created");
    }
    std::fs::write(&path, to_json(rows)).expect("the results file can be written");
    path
}

// ---------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------

/// Process peak resident set size in bytes, where the platform reports one.
///
/// Linux only: `VmHWM` in `/proc/self/status`, in kibibytes. Everywhere else
/// this answers `None` and the row's `peak_rss_mb` is `0.0`.
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

/// The same, as the `0` a row carries when the platform has no answer.
pub fn peak_rss_bytes_or_zero() -> u64 {
    peak_rss_bytes().unwrap_or(0)
}

/// Bytes and filesystem entries a pyramid occupies.
///
/// A file counts one entry and so does a directory, because the cost this
/// column exists to show is namespace pressure rather than data volume: a
/// directory is an inode, a dentry and a lookup on every path resolution
/// underneath it. For a single archive the answer is `(len, 1)`.
pub fn occupancy(path: &Path) -> (u64, u64) {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(_) => return (0, 0),
    };
    if !meta.is_dir() {
        return (meta.len(), 1);
    }
    let mut bytes = 0;
    let mut entries = 1; // the directory itself
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(listing) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in listing.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            entries += 1;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                bytes += meta.len();
            }
        }
    }
    (bytes, entries)
}

/// The `p`th percentile of a latency sample, in microseconds.
///
/// Sorts in place (nearest-rank), so the caller hands over a `&mut` and gets a
/// reordered slice back. An empty sample is `0.0`.
pub fn percentile_micros(samples: &mut [Duration], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_unstable();
    let rank = ((samples.len() as f64) * p).ceil() as usize;
    let index = rank.clamp(1, samples.len()) - 1;
    samples[index].as_secs_f64() * 1_000_000.0
}

/// A deterministic RGB gradient, the same one `tests/pmtiles_pyramid_reader.rs`
/// generates.
///
/// Deterministic matters more than realistic here: a benchmark whose source
/// changes between runs cannot be compared against its own history, and the
/// gradient compresses like real imagery rather than like a solid fill, which
/// a flat colour would turn into one deduped payload and no work at all.
pub fn gradient(width: u32, height: u32) -> Raster {
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

/// A cheap deterministic sequence, so "random access" means the same
/// coordinates on every run and on every backend.
///
/// This is `splitmix64`, chosen because it is eight lines and needs no
/// dependency. A benchmark that seeded from the clock would compare two
/// backends over two different coordinate sets.
pub struct Splitmix(u64);

impl Splitmix {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}
