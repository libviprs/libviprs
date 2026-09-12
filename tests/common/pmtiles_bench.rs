//! Measurement plumbing shared by the PMTiles benchmark and the release
//! readiness guards (issue #993).
//!
//! `tests/pmtiles_benchmarks.rs` measures with it and
//! `tests/pmtiles_release_readiness.rs` checks the documented column table
//! against [`FIELDS`]. The bounded-memory guards do **not** use it, whatever
//! an earlier version of this sentence said: they install their own counting
//! allocator and share nothing with this.
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
//! Linux and nowhere else this crate builds for. Everywhere else the column is
//! `null`. The published figures come from the Linux container run.
//!
//! # A column nobody measured is `null`, never `0`
//!
//! Every unmeasured number here emits JSON `null`, and the reason is that a
//! zero is a value on a scale somebody plots. `occupancy` could not stat a
//! path and published `filesystem_entries: 0`, which is *better* than the `1`
//! a real archive costs, on the one column this whole epic exists to move.
//! `peak_rss_mb` was `0.0` off Linux and `resource_cost` divides by it, so a
//! macOS run published `resource_cost: 0`, the best possible score on that
//! column, as a measurement. Every failure mode in the old shape pointed at a
//! flattering number. A `null` is a hole a consumer can see.
//!
//! # The envelope
//!
//! The document is `{"schema": 1, "rows": [...]}` rather than a bare array, so
//! a consumer that does not understand a future shape can say so instead of
//! reading a renamed column as absent. [`SCHEMA_VERSION`] is the number to
//! bump when a field changes meaning.
//!
//! # The writer here and the reader that checks it are not the same code
//!
//! `serde_json` is already a dependency of this crate, so the shape guard in
//! `tests/pmtiles_benchmarks.rs` parses what this emits and asserts the field
//! set, the types and the record count. This side stays a hand-rolled
//! serialiser rather than a `Serialize` derive, for two reasons: it fixes the
//! field **order**, which a `serde_json::Map` would alphabetise because it is
//! a `BTreeMap`, and a producer checked by an independent parser is worth more
//! than a producer checked by its own round trip. No dependency is added
//! either way.

// This module is pulled into two test binaries through `#[path]`, and each
// uses a different subset of it, so anything unused by one is unused code in
// that binary. The allow is for that and not for a grab bag: nothing here is
// kept without a caller, which is why `write_results` went when its only
// would-be consumer turned out to inline the same three lines.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use libviprs::{PixelFormat, Raster};

/// Where a run's results land when `LIBVIPRS_BENCH_JSON` does not say.
///
/// `pmtiles_results.json`, and deliberately not `scalability_results.json`:
/// that file is a generated artefact of libviprs-bench and a hand-written file
/// landing on its name is a number nobody can trace back to a run.
pub const DEFAULT_RESULTS_PATH: &str = "target/pmtiles_results.json";

/// The shape of the exported document.
///
/// Bump it when a field changes meaning or leaves, so a consumer can refuse a
/// document it was not written against instead of quietly reading a renamed
/// column as absent.
pub const SCHEMA_VERSION: u32 = 1;

/// What `engine` carries.
///
/// The engine under test, which is this crate, and the same one on both sides
/// of the comparison. `storage` is what varies. The two used to hold the same
/// string, which made `engine` a duplicate of `storage` under a name that says
/// something else.
pub const ENGINE: &str = "libviprs";

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

    /// The cells a generation sweep walks: source width, source height, tile
    /// size.
    ///
    /// The CI row is one cell, because the comparison this issue is about is
    /// between two storage backends at one size rather than a scaling curve,
    /// and a second cell doubles the cost of the cheap profile to say the same
    /// thing twice.
    ///
    /// The last large cell is 8192 pixels at a **64 pixel** tile, which is the
    /// only one of the four that produces more than `ROOT_ONLY_MAX_ENTRIES`
    /// directory entries and so the only one whose archive has leaf
    /// directories at all. It plans 21851 tiles, which
    /// `the_eight_thousand_pixel_cell_plans_the_tile_count_the_doc_publishes`
    /// pins. Reaching past the cutoff through the tile size rather than
    /// through a bigger canvas is deliberate: that many entries at 256 pixel
    /// tiles needs a 32768 pixel source, which is a 3.2 GB raster, and the
    /// archive's directory shape is what the read path cares about rather than
    /// the pixels behind it.
    pub fn canvases(self) -> &'static [(u32, u32, u32)] {
        match self {
            Self::Ci => &[(2048, 2048, 256)],
            Self::Large => &[
                (2048, 2048, 256),
                (8192, 8192, 256),
                (16384, 16384, 256),
                (8192, 8192, 64),
            ],
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
/// Every field that a row did not measure is `None`, and `None` is written as
/// JSON `null`. A generation row measures the pyramid it wrote and no
/// latencies; a read row measures latencies and not the pyramid, which some
/// other row already did.
#[derive(Debug, Clone)]
pub struct Measurement {
    pub width: u32,
    pub height: u32,
    /// Tile edge in pixels. The archive's directory shape follows from how
    /// many tiles a plan has, so two rows at one canvas size and two tile
    /// sizes are not the same measurement.
    pub tile_size: u32,
    pub megapixels: f64,
    /// The engine under test, which is [`ENGINE`] on every row here. It is
    /// not the storage backend, which is what `storage` is for.
    pub engine: String,
    /// `"pmtiles"` or `"directory"`.
    pub storage: String,
    pub concurrency: usize,
    pub wall_time_ms: f64,
    /// The engine's own [`MemoryTracker`] peak: raster buffers and nothing
    /// else. `None` on a read row, where the tracker charges nothing at all.
    pub tracked_memory_mb: Option<f64>,
    /// Process peak resident set for the phase. `None` where the platform has
    /// no answer, which is everywhere without `/proc`.
    pub peak_rss_mb: Option<f64>,
    pub tiles_produced: u64,
    /// `None` when the row took no measurable time, so there is no rate.
    pub tiles_per_second: Option<f64>,
    /// `None` whenever `peak_rss_mb` is, because it is the denominator.
    pub tiles_per_second_per_mb: Option<f64>,
    /// `None` whenever `peak_rss_mb` is. This is the column a zero flattered
    /// most: lower is better, so an unmeasured RSS used to publish the best
    /// possible score.
    pub resource_cost: Option<f64>,
    /// `"generate"`, `"read_cold"`, `"read_warm"`, `"read_random"`,
    /// `"read_sequential"` or `"read_concurrent"`.
    pub scenario: String,
    /// `"ci"` or `"large"`.
    pub profile: String,
    /// Bytes the pyramid occupies: one archive, or the sum of every file in
    /// the tree. `None` on a read row and `None` when the path could not be
    /// walked.
    pub output_bytes: Option<u64>,
    /// Filesystem entries the pyramid occupies, directories included. This is
    /// the namespace-explosion column: one for an archive, one per tile plus
    /// the level and column directories for a tree. `None` on a read row and
    /// `None` when the path could not be walked, because a zero here reads as
    /// better than the `1` an archive really costs.
    pub filesystem_entries: Option<u64>,
    /// Tile payload bytes the row's lookups returned, summed. `None` on a
    /// generation row.
    ///
    /// Named for what it is. It was `bytes_fetched`, and bytes off the
    /// transport is exactly the quantity the index-only proof is about, so a
    /// reader of libviprs.org would have taken this column as evidence for a
    /// claim it does not measure. Transport bytes are counted in
    /// `tests/pmtiles_index_only_reads.rs`, against a `RangeReader` that can
    /// see them.
    pub tile_bytes_returned: Option<u64>,
    /// Median per-lookup latency in microseconds. `None` on a generation row.
    pub p50_latency_us: Option<f64>,
    /// 99th-percentile per-lookup latency. `None` on a generation row.
    pub p99_latency_us: Option<f64>,
}

impl Measurement {
    /// Build a row from the raw measurements, deriving the three ratios the
    /// same way `libviprs-bench`'s `scalability` binary derives them, except
    /// that a missing denominator gives `None` rather than zero.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scenario: &str,
        storage: &str,
        profile: Profile,
        width: u32,
        height: u32,
        tile_size: u32,
        concurrency: usize,
        elapsed: Duration,
        tracked_bytes: Option<u64>,
        rss_bytes: Option<u64>,
        tiles: u64,
    ) -> Self {
        let secs = elapsed.as_secs_f64();
        let megabytes = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        let rss_mb = rss_bytes.map(megabytes);
        let tps = if secs > 0.0 {
            Some(tiles as f64 / secs)
        } else {
            None
        };
        // Both ratios use the RSS basis so a row here means what the same
        // column means in the scalability data, and both are `None` when that
        // basis is missing rather than zero.
        let tps_per_mb = match (tps, rss_mb) {
            (Some(tps), Some(mb)) if mb > 0.0 => Some(tps / mb),
            _ => None,
        };
        let cost = match rss_mb {
            Some(mb) if tiles > 0 => Some((mb * secs) / tiles as f64),
            _ => None,
        };

        Self {
            width,
            height,
            tile_size,
            megapixels: f64::from(width) * f64::from(height) / 1_000_000.0,
            engine: ENGINE.to_string(),
            storage: storage.to_string(),
            concurrency,
            wall_time_ms: secs * 1000.0,
            tracked_memory_mb: tracked_bytes.map(megabytes),
            peak_rss_mb: rss_mb,
            tiles_produced: tiles,
            tiles_per_second: tps,
            tiles_per_second_per_mb: tps_per_mb,
            resource_cost: cost,
            scenario: scenario.to_string(),
            profile: profile.label().to_string(),
            output_bytes: None,
            filesystem_entries: None,
            tile_bytes_returned: None,
            p50_latency_us: None,
            p99_latency_us: None,
        }
    }

    /// Record what the pyramid occupies, or that it could not be measured.
    pub fn with_output(mut self, occupancy: Option<(u64, u64)>) -> Self {
        self.output_bytes = occupancy.map(|(bytes, _)| bytes);
        self.filesystem_entries = occupancy.map(|(_, entries)| entries);
        self
    }

    pub fn with_latencies(mut self, samples: &mut [Duration], returned: u64) -> Self {
        self.tile_bytes_returned = Some(returned);
        self.p50_latency_us = percentile_micros(samples);
        self.p99_latency_us = percentile_micros_at(samples, 0.99);
        self
    }

    fn to_json(&self) -> String {
        let mut out = String::from("  {\n");
        push_u64(&mut out, "width", u64::from(self.width));
        push_u64(&mut out, "height", u64::from(self.height));
        push_f64(&mut out, "megapixels", self.megapixels);
        push_u64(&mut out, "tile_size", u64::from(self.tile_size));
        push_str(&mut out, "engine", &self.engine);
        push_usize(&mut out, "concurrency", self.concurrency);
        push_f64(&mut out, "wall_time_ms", self.wall_time_ms);
        push_opt_f64(&mut out, "tracked_memory_mb", self.tracked_memory_mb);
        push_opt_f64(&mut out, "peak_rss_mb", self.peak_rss_mb);
        push_u64(&mut out, "tiles_produced", self.tiles_produced);
        push_opt_f64(&mut out, "tiles_per_second", self.tiles_per_second);
        push_opt_f64(
            &mut out,
            "tiles_per_second_per_mb",
            self.tiles_per_second_per_mb,
        );
        push_opt_f64(&mut out, "resource_cost", self.resource_cost);
        push_str(&mut out, "scenario", &self.scenario);
        push_str(&mut out, "storage", &self.storage);
        push_str(&mut out, "profile", &self.profile);
        push_opt_u64(&mut out, "output_bytes", self.output_bytes);
        push_opt_u64(&mut out, "filesystem_entries", self.filesystem_entries);
        push_opt_u64(&mut out, "tile_bytes_returned", self.tile_bytes_returned);
        push_opt_f64(&mut out, "p50_latency_us", self.p50_latency_us);
        // The last field carries no trailing comma.
        out.push_str(&format!(
            "    \"p99_latency_us\": {}\n",
            json_opt_number(self.p99_latency_us)
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
pub const FIELDS: [&str; 21] = [
    "width",
    "height",
    "megapixels",
    "tile_size",
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
    "tile_bytes_returned",
    "p50_latency_us",
    "p99_latency_us",
];

/// The fields that carry a string rather than a number or `null`.
pub const STRING_FIELDS: [&str; 4] = ["engine", "scenario", "storage", "profile"];

/// The fields a row may leave `null` because it did not measure them.
///
/// Everything else has to be a finite number on every row, which is what stops
/// a `null` spreading into a column that always has an answer.
pub const NULLABLE_FIELDS: [&str; 10] = [
    "tracked_memory_mb",
    "peak_rss_mb",
    "tiles_per_second",
    "tiles_per_second_per_mb",
    "resource_cost",
    "output_bytes",
    "filesystem_entries",
    "tile_bytes_returned",
    "p50_latency_us",
    "p99_latency_us",
];

// ---------------------------------------------------------------------------
// Serialising
// ---------------------------------------------------------------------------

fn push_u64(out: &mut String, key: &str, value: u64) {
    out.push_str(&format!("    \"{key}\": {value},\n"));
}

fn push_usize(out: &mut String, key: &str, value: usize) {
    out.push_str(&format!("    \"{key}\": {value},\n"));
}

fn push_opt_u64(out: &mut String, key: &str, value: Option<u64>) {
    match value {
        Some(value) => push_u64(out, key, value),
        None => out.push_str(&format!("    \"{key}\": null,\n")),
    }
}

fn push_f64(out: &mut String, key: &str, value: f64) {
    out.push_str(&format!(
        "    \"{key}\": {},\n",
        json_opt_number(Some(value))
    ));
}

fn push_opt_f64(out: &mut String, key: &str, value: Option<f64>) {
    out.push_str(&format!("    \"{key}\": {},\n", json_opt_number(value)));
}

fn push_str(out: &mut String, key: &str, value: &str) {
    out.push_str(&format!("    \"{key}\": \"{}\",\n", escape(value)));
}

/// A finite JSON number, or `null`.
///
/// JSON has no spelling for NaN or an infinity, and a benchmark that divided
/// by a zero duration would otherwise emit a document no parser accepts. The
/// fallback used to be `0`, which is a value on a scale somebody plots and, on
/// `resource_cost`, the best score on the column. A hole says what happened.
fn json_opt_number(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:?}"),
        _ => "null".to_string(),
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

/// Serialise a run's rows as a bare JSON array.
///
/// This is the wire format between a benchmark child process and its parent,
/// not the published document: a child owns its own rows and the parent owns
/// the envelope around all of them. [`document`] is what gets written out.
pub fn rows_to_json(rows: &[Measurement]) -> String {
    let body: Vec<String> = rows.iter().map(Measurement::to_json).collect();
    format!("[\n{}\n]\n", body.join(",\n"))
}

/// Wrap an array of rows in the schema envelope.
///
/// `rows_array` is the text of a JSON array, which is what [`rows_to_json`]
/// produces and what splicing several children's documents produces. Taking
/// text rather than rows is what lets the parent keep each child's field order
/// and pretty printing, which re-serialising through `serde_json` would not:
/// its map is a `BTreeMap`, so a round trip alphabetises the columns.
pub fn document(rows_array: &str) -> String {
    let indented: String = rows_array
        .trim_end()
        .lines()
        .map(|line| {
            if line.is_empty() {
                line.to_string()
            } else {
                format!("  {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{{\n  \"schema\": {SCHEMA_VERSION},\n  \"rows\": {}\n}}\n",
        indented.trim_start()
    )
}

/// The published document for a run's rows.
pub fn to_json(rows: &[Measurement]) -> String {
    document(&rows_to_json(rows))
}

/// Where the results of this run go.
pub fn results_path() -> PathBuf {
    match std::env::var("LIBVIPRS_BENCH_JSON") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_RESULTS_PATH),
    }
}

/// Write a finished document out, creating the parent directory if it is
/// missing.
///
/// Takes the document text rather than the rows because the benchmark builds
/// it by splicing several child processes' arrays together, and that text is
/// what has to land on disk unchanged.
pub fn write_document(text: &str) -> PathBuf {
    let path = results_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("the results directory can be created");
    }
    std::fs::write(&path, text).expect("the results file can be written");
    path
}

// ---------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------

/// Process peak resident set size in bytes, where the platform reports one.
///
/// Linux only: `VmHWM` in `/proc/self/status`, in kibibytes. Everywhere else
/// this answers `None`, and `None` travels all the way to a `null` in the
/// exported row rather than becoming a zero somewhere in between.
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

/// Bytes and filesystem entries a pyramid occupies, or `None` if the path
/// could not be stat'd at all.
///
/// A file counts one entry and so does a directory, because the cost this
/// column exists to show is namespace pressure rather than data volume: a
/// directory is an inode, a dentry and a lookup on every path resolution
/// underneath it. For a single archive the answer is `(len, 1)`.
///
/// A failure is `None` and not `(0, 0)`. Zero entries is a better number than
/// the one a real archive costs, so the old answer published a win on the
/// single column this epic exists to move, every time the measurement broke.
pub fn occupancy(path: &Path) -> Option<(u64, u64)> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.is_dir() {
        return Some((meta.len(), 1));
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
    Some((bytes, entries))
}

/// The median of a latency sample, in microseconds.
pub fn percentile_micros(samples: &mut [Duration]) -> Option<f64> {
    percentile_micros_at(samples, 0.50)
}

/// The `p`th percentile of a latency sample, in microseconds.
///
/// Sorts in place (nearest-rank), so the caller hands over a `&mut` and gets a
/// reordered slice back. An empty sample is `None`, because no lookup happened
/// and zero microseconds a lookup is a claim rather than a hole.
pub fn percentile_micros_at(samples: &mut [Duration], p: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    let rank = ((samples.len() as f64) * p).ceil() as usize;
    let index = rank.clamp(1, samples.len()) - 1;
    Some(samples[index].as_secs_f64() * 1_000_000.0)
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
