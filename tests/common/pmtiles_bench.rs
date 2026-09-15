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
//! The document is `{"schema": 2, "provenance": {...}, "rows": [...]}` rather
//! than a bare array, so a consumer that does not understand a future shape
//! can say so instead of reading a renamed column as absent.
//! [`SCHEMA_VERSION`] is the number to bump when a field changes meaning.
//!
//! # The envelope says where the numbers came from
//!
//! Schema 2 adds [`Provenance`]: the commit, whether the tree was dirty, the
//! architecture, the OS, the CPU count, the load average and the rustc that
//! ran. Issue #1021 exists because a published ramp was extrapolated from
//! three points, and the same run's own provenance said only "amd64
//! container", which on an Apple Silicon host means Rosetta and nobody could
//! tell from the document. A wall-clock number with no host attached is not a
//! measurement anybody else can check, so the envelope carries one now.
//!
//! It mirrors `libviprs_bench::provenance::Provenance` field for field where
//! the fields overlap, on purpose: that harness solved this already (its issue
//! #159), including the `measurement_condition_warnings()` that flags a run
//! taken while the box was busy. No dependency is added for it, because
//! `libviprs-bench` depends on this crate and not the other way round, so the
//! shape is copied rather than imported.
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
///
/// 2 (issue #1021): the envelope gained [`Provenance`], rows gained
/// `root_entries`, and `scenario` gained the six `read_cold_*` phases and a
/// `read_concurrent` row per thread count rather than one. All of it is
/// additive, so a schema 1 consumer that reads by name still finds every
/// column it knew, but it would silently plot the phase rows as though they
/// were whole cold opens, which is why the number moved.
pub const SCHEMA_VERSION: u32 = 2;

/// What `engine` carries.
///
/// The engine under test, which is this crate, and the same one on both sides
/// of the comparison. `storage` is what varies. The two used to hold the same
/// string, which made `engine` a duplicate of `storage` under a name that says
/// something else.
pub const ENGINE: &str = "libviprs";

/// The phases a cold PMTiles open goes through, in the order
/// `Reader::try_new` runs them, plus the lookup that follows.
///
/// `read_cold` times all six as one number, which is what issue #1021 is
/// complaining about: the fix for a slow gzip inflate, a slow varint loop and
/// a slow `pread` are three different pieces of work and the combined row
/// cannot say which one is the problem. Every one of these is measured
/// alongside the combined row rather than instead of it, so the history stays
/// comparable, and `the_cold_split_accounts_for_the_whole_combined_row` in
/// `tests/pmtiles_benchmarks.rs` checks that they are the same work it does,
/// by the byte ranges each one reads and the values each one decodes rather
/// than by comparing durations.
pub const COLD_PHASES: [&str; 6] = [
    "read_cold_open",
    "read_cold_header",
    "read_cold_root_fetch",
    "read_cold_root_inflate",
    "read_cold_root_decode",
    "read_cold_lookup",
];

/// The cell whose root stops just under the writer's root-only cutoff.
///
/// 4096 by 6256 pixels at a 46 pixel tile plans 16369 tiles, and because a
/// gradient's tiles are all distinct that is also 16369 run-length-encoded
/// directory entries, 14 under the 16383 the writer will still put in a flat
/// root. `the_brink_cell_is_the_largest_root_the_search_space_reaches` re-runs
/// the search that found it, and
/// `the_brink_cells_root_stops_just_under_the_writers_cutoff` opens the
/// archive and asks it, which is the assertion that matters: this triple is
/// arithmetic over a planner, and arithmetic over a planner stops being the
/// brink the day the planner changes.
///
/// The shape looks arbitrary because it is a search result rather than a
/// choice. What it is optimising is the distance to the cutoff: the planner
/// only produces the tile counts it produces, and stepping the canvas by a
/// pixel at this tile size moves the total by about 150, so 14 under is close
/// to the best any cell can do.
pub const BRINK_CANVAS: (u32, u32, u32) = (4096, 6256, 46);

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
    /// The fourth large cell is 8192 pixels at a **64 pixel** tile, which is the
    /// only one of the four that produces more than `ROOT_ONLY_MAX_ENTRIES`
    /// directory entries and so the only one whose archive has leaf
    /// directories at all. It plans 21851 tiles, which
    /// `the_eight_thousand_pixel_cell_plans_the_tile_count_the_doc_publishes`
    /// pins. Reaching past the cutoff through the tile size rather than
    /// through a bigger canvas is deliberate: that many entries at 256 pixel
    /// tiles needs a 32768 pixel source, which is a 3.2 GB raster, and the
    /// archive's directory shape is what the read path cares about rather than
    /// the pixels behind it.
    ///
    /// The fifth is [`BRINK_CANVAS`], and it is there because the other four
    /// sit at 93, 1373, 5469 and 6 root entries, which brackets the worst case
    /// without ever touching it. Issue #1021 fitted a line through three of
    /// those and put the worst open at about 277 us, and an extrapolation from
    /// three points is not a measurement. This cell's root stops one step
    /// under the writer's own cutoff, so the peak of the ramp is measured
    /// rather than predicted.
    pub fn canvases(self) -> &'static [(u32, u32, u32)] {
        match self {
            Self::Ci => &[(2048, 2048, 256)],
            Self::Large => &[
                (2048, 2048, 256),
                (8192, 8192, 256),
                (16384, 16384, 256),
                (8192, 8192, 64),
                BRINK_CANVAS,
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
    /// Entries in the archive's **root** directory, as the archive itself
    /// answers it.
    ///
    /// This is the x axis of issue #1021's ramp: a cold open decodes the whole
    /// root, so what a first lookup costs is a function of this number and not
    /// of the tile count, the canvas or the file size. It is the count of
    /// run-length-encoded entries rather than of tiles, which is what the
    /// writer's own cutoff counts too, and on a gradient source the two are
    /// equal because no two neighbouring tiles share a payload.
    ///
    /// `None` on every directory-backend row, because a tree has no root to
    /// decode, and `None` where the archive was not asked. Never `0`: a root
    /// of no entries is not a directory a reader can open, so a zero here
    /// would be a free open on the one column the ramp is measured against.
    pub root_entries: Option<u64>,
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
            root_entries: None,
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

    /// Record how many entries the archive's root holds, which is what the
    /// cold-open ramp is measured against.
    ///
    /// Takes an `Option` rather than a count so the directory backend's rows
    /// say "there is no root here" in the one spelling this harness has for
    /// that, instead of claiming a root of zero entries.
    pub fn with_root_entries(mut self, entries: Option<u64>) -> Self {
        self.root_entries = entries;
        self
    }

    pub fn with_latencies(mut self, samples: &mut [Duration], returned: u64) -> Self {
        self.tile_bytes_returned = Some(returned);
        self.p50_latency_us = percentile_micros(samples);
        self.p99_latency_us = percentile_micros_at(samples, 0.99);
        self
    }

    /// The latency half of [`Measurement::with_latencies`], for a row that
    /// returned no tile bytes because its phase does not fetch a tile.
    ///
    /// The four index phases of a cold open read the header, the root's bytes,
    /// the inflated root and the decoded entries, and not one of them returns
    /// a tile. `tile_bytes_returned` is `null` on those rows rather than `0`,
    /// which would read as a phase that fetched a tile for free.
    pub fn with_latency_samples(mut self, samples: &mut [Duration]) -> Self {
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
        push_opt_u64(&mut out, "root_entries", self.root_entries);
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
pub const FIELDS: [&str; 22] = [
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
    "root_entries",
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
pub const NULLABLE_FIELDS: [&str; 11] = [
    "tracked_memory_mb",
    "peak_rss_mb",
    "tiles_per_second",
    "tiles_per_second_per_mb",
    "resource_cost",
    "output_bytes",
    "filesystem_entries",
    "root_entries",
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
///
/// The provenance comes in as a value rather than being captured here, because
/// the host has to be sampled once for the whole run by the parent process and
/// not once per child cell: a load average read after four cells have already
/// run is the benchmark measuring itself.
pub fn document(rows_array: &str, provenance: &Provenance) -> String {
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
        "{{\n  \"schema\": {SCHEMA_VERSION},\n  \"provenance\": {},\n  \"rows\": {}\n}}\n",
        provenance.to_json(),
        indented.trim_start()
    )
}

/// The published document for a run's rows.
pub fn to_json(rows: &[Measurement], provenance: &Provenance) -> String {
    document(&rows_to_json(rows), provenance)
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// Where a run's numbers came from.
///
/// Modelled on `libviprs_bench::provenance::Provenance`, which already carries
/// exactly this and learned the hard way why (its issue #159). Copied rather
/// than imported: `libviprs-bench` depends on this crate, so importing it here
/// would be a cycle, and a benchmark harness is the last place to take a
/// dependency for seven fields.
///
/// Two deliberate differences from that shape. An unknown value here is `null`
/// rather than the string `"unknown"`, because the rest of this file already
/// spells "nobody measured this" as a hole and two spellings for it in one
/// document is one too many. And it carries `commit` and `dirty`, which that
/// harness gets from a build script: this crate has none, so both are read
/// from git at run time.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Short commit the tree was at, or `None` when git could not answer.
    pub commit: Option<String>,
    /// Whether the working tree had uncommitted changes to tracked files.
    ///
    /// `Some(true)` is the one that matters: the commit above then does not
    /// describe what was measured, so the numbers cannot be reproduced from
    /// it. `None` when git could not answer at all, which is not the same
    /// claim as a clean tree.
    pub dirty: Option<bool>,
    /// What `rustc --version` says **at run time**.
    ///
    /// This crate has no build script, so there is nowhere to stamp the
    /// compiler that actually built the harness. The harness is built and run
    /// inside one container, so the two agree in the way it is meant to be
    /// used; a binary carried to a box with a different toolchain would
    /// record that box's rustc, and this doc comment is the only thing that
    /// says so.
    pub rustc_version: Option<String>,
    /// `"release"` or `"debug"`, from `cfg!(debug_assertions)`, so it is the
    /// profile the harness was compiled with rather than the one somebody
    /// meant to use. A timing number from a debug build is not a measurement
    /// and [`Provenance::measurement_condition_warnings`] says so.
    pub build_profile: &'static str,
    pub host: HostInfo,
    /// 1/5/15-minute load average sampled before the run.
    pub load_average: Option<LoadAverage>,
}

/// The machine a run happened on.
#[derive(Debug, Clone)]
pub struct HostInfo {
    pub cpu_model: Option<String>,
    pub ncpu: Option<u32>,
    pub arch: &'static str,
    pub os: &'static str,
    /// Best effort. Container CPU quotas move both timing and RSS, and the
    /// published PMTiles numbers were taken in one.
    pub in_container: bool,
}

/// The 1/5/15-minute run-queue averages.
#[derive(Debug, Clone, Copy)]
pub struct LoadAverage {
    pub one_min: f64,
    pub five_min: f64,
    pub fifteen_min: f64,
}

impl Provenance {
    /// Sample the environment.
    ///
    /// Call it once, in the parent, before any cell runs.
    pub fn capture() -> Self {
        let (commit, dirty) = commit_and_dirty(Path::new(env!("CARGO_MANIFEST_DIR")));
        Self {
            commit,
            dirty,
            rustc_version: rustc_version(),
            build_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            host: HostInfo {
                cpu_model: cpu_model(),
                ncpu: std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get() as u32),
                arch: std::env::consts::ARCH,
                os: std::env::consts::OS,
                in_container: detect_container(),
            },
            load_average: load_average(),
        }
    }

    /// Provenance for a document nobody captured one for.
    ///
    /// Every field a hole, which is what a schema 1 document effectively
    /// carried, and it is distinguishable from a real capture precisely
    /// because a real one fills something in.
    pub fn unknown() -> Self {
        Self {
            commit: None,
            dirty: None,
            rustc_version: None,
            build_profile: "unknown",
            host: HostInfo {
                cpu_model: None,
                ncpu: None,
                arch: "unknown",
                os: "unknown",
                in_container: false,
            },
            load_average: None,
        }
    }

    /// Whether the box already had every core queued when the load was
    /// sampled, so the wall-clock numbers are contended.
    ///
    /// `false` when either half is missing: a signal nobody has must not cry
    /// wolf. The threshold is `libviprs-bench`'s, one-minute load at or above
    /// the CPU count.
    pub fn host_looked_contended(&self) -> bool {
        match (self.load_average, self.host.ncpu) {
            (Some(load), Some(ncpu)) if ncpu > 0 => load.one_min >= f64::from(ncpu),
            _ => false,
        }
    }

    /// Everything about this run that makes its numbers less believable, one
    /// string per line, in a stable order.
    ///
    /// The caller prints them to stderr. They are warnings and not failures
    /// because a contended run is still a run somebody may have meant to take,
    /// and a benchmark that refuses to produce a number is worse than one that
    /// produces a number with a label on it.
    pub fn measurement_condition_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.build_profile != "release" {
            warnings.push(format!(
                "WARNING: this harness was built in the {} profile, so its wall-clock numbers \
                 measure an unoptimised build and are not comparable to anything published.",
                self.build_profile
            ));
        }
        if self.host_looked_contended() {
            warnings.push(format!(
                "WARNING: 1-minute host load {} against {} CPUs when it was sampled, so the box \
                 was already busy and these numbers are inflated by scheduling pressure rather \
                 than by the code under test.",
                self.load_average_line(),
                self.host.ncpu.unwrap_or(0),
            ));
        }
        if self.dirty == Some(true) {
            warnings.push(format!(
                "WARNING: the working tree had uncommitted changes, so commit {} does not \
                 describe what was measured and nobody can reproduce these numbers from it.",
                self.commit.as_deref().unwrap_or("unknown"),
            ));
        }
        if self.commit.is_none() {
            warnings.push(
                "WARNING: no commit could be read, so this document does not say which tree \
                 produced it."
                    .to_string(),
            );
        }
        warnings
    }

    /// `"1.23 / 1.05 / 0.98"`, or that there was no sample.
    pub fn load_average_line(&self) -> String {
        match self.load_average {
            Some(load) => format!(
                "{:.2} / {:.2} / {:.2}",
                load.one_min, load.five_min, load.fifteen_min
            ),
            None => "unavailable".to_string(),
        }
    }

    /// The provenance object, indented to sit inside the envelope.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        push_opt_str(&mut out, 4, "commit", self.commit.as_deref());
        push_opt_bool(&mut out, 4, "dirty", self.dirty);
        push_opt_str(&mut out, 4, "rustc_version", self.rustc_version.as_deref());
        push_opt_str(&mut out, 4, "build_profile", Some(self.build_profile));
        out.push_str("    \"host\": {\n");
        push_opt_str(&mut out, 6, "cpu_model", self.host.cpu_model.as_deref());
        push_indented_opt_u64(&mut out, 6, "ncpu", self.host.ncpu.map(u64::from));
        push_opt_str(&mut out, 6, "arch", Some(self.host.arch));
        push_opt_str(&mut out, 6, "os", Some(self.host.os));
        out.push_str(&format!(
            "      \"in_container\": {}\n    }},\n",
            self.host.in_container
        ));
        match self.load_average {
            Some(load) => out.push_str(&format!(
                "    \"load_average\": {{\n      \"one_min\": {},\n      \"five_min\": {},\n      \
                 \"fifteen_min\": {}\n    }}\n",
                json_opt_number(Some(load.one_min)),
                json_opt_number(Some(load.five_min)),
                json_opt_number(Some(load.fifteen_min)),
            )),
            None => out.push_str("    \"load_average\": null\n"),
        }
        out.push_str("  }");
        out
    }
}

/// Every key the provenance object carries, in order, including the nested
/// ones under their parent's name.
///
/// The shape guard reads this rather than repeating the list, the same way it
/// reads [`FIELDS`] for a row.
pub const PROVENANCE_FIELDS: [&str; 6] = [
    "commit",
    "dirty",
    "rustc_version",
    "build_profile",
    "host",
    "load_average",
];

/// Every key inside `provenance.host`, in order.
pub const PROVENANCE_HOST_FIELDS: [&str; 5] = ["cpu_model", "ncpu", "arch", "os", "in_container"];

fn push_opt_str(out: &mut String, indent: usize, key: &str, value: Option<&str>) {
    let pad = " ".repeat(indent);
    match value {
        Some(value) => out.push_str(&format!("{pad}\"{key}\": \"{}\",\n", escape(value))),
        None => out.push_str(&format!("{pad}\"{key}\": null,\n")),
    }
}

fn push_opt_bool(out: &mut String, indent: usize, key: &str, value: Option<bool>) {
    let pad = " ".repeat(indent);
    match value {
        Some(value) => out.push_str(&format!("{pad}\"{key}\": {value},\n")),
        None => out.push_str(&format!("{pad}\"{key}\": null,\n")),
    }
}

fn push_indented_opt_u64(out: &mut String, indent: usize, key: &str, value: Option<u64>) {
    let pad = " ".repeat(indent);
    match value {
        Some(value) => out.push_str(&format!("{pad}\"{key}\": {value},\n")),
        None => out.push_str(&format!("{pad}\"{key}\": null,\n")),
    }
}

/// What commit a tree is at and whether it has been edited since.
///
/// Both `None` when git cannot answer, and there are more ways for that to
/// happen than there look to be. A git-less container and a tarball with no
/// `.git` are the obvious two. The one that actually turned up here is a
/// **linked worktree bind-mounted into a container**: its `.git` is a file
/// pointing at a gitdir under the main checkout, that path is outside the
/// mount, and git answers `not a git repository`. So a run taken from an agent
/// worktree publishes a null commit, honestly, and a run taken from a clone
/// (which is what `tools/local-ci.py` makes inside the container) publishes a
/// real one.
///
/// Takes the directory rather than reading `CARGO_MANIFEST_DIR` itself so
/// `a_repository_with_a_commit_is_read_back` can point it at a repository it
/// built, and prove the reading works without depending on how this checkout
/// happens to be mounted.
pub fn commit_and_dirty(dir: &Path) -> (Option<String>, Option<bool>) {
    (
        git_in(dir, &["rev-parse", "--short", "HEAD"]),
        git_in(dir, &["status", "--porcelain"]).map(|out| !out.is_empty()),
    )
}

/// Run one git command in `dir`, or answer `None`.
pub fn git_in(dir: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Whether `git` can be run at all.
///
/// Used by the control that proves [`commit_and_dirty`] reads a repository,
/// so that the control asserts the null path instead of skipping when there is
/// no git to read one with. A test that skips is the same colour as a test
/// that passed.
pub fn git_is_available() -> bool {
    std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

fn rustc_version() -> Option<String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if line.is_empty() { None } else { Some(line) }
}

/// What the CPU calls itself, where it says.
///
/// `/proc/cpuinfo` on aarch64 Linux carries no `model name` line at all, so
/// this answers `None` in the arm64 container rather than inventing one. The
/// architecture is recorded separately and that is the field that matters for
/// the question this issue asks, which is whether a run was emulated.
fn cpu_model() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        if out.status.success() {
            let model = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !model.is_empty() {
                return Some(model);
            }
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in text.lines() {
            if line.starts_with("model name")
                && let Some((_, rest)) = line.split_once(':')
            {
                return Some(rest.trim().to_string());
            }
        }
        None
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// The 1/5/15-minute load average, where the platform reports one.
///
/// `/proc/loadavg` on Linux. On macOS `getloadavg` needs `libc`, which this
/// crate does not depend on and will not gain one for a benchmark's banner, so
/// it shells out to `sysctl -n vm.loadavg` instead, whose output is
/// `{ 1.83 1.92 1.98 }`.
fn load_average() -> Option<LoadAverage> {
    #[cfg(target_os = "linux")]
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    #[cfg(not(target_os = "linux"))]
    let text = {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    parse_load_average(&text)
}

/// Pull three averages out of either platform's spelling of them.
///
/// Separated from the reading so it can be tested without a host under load:
/// Linux writes `0.52 0.58 0.59 1/523 12345` and macOS writes
/// `{ 1.83 1.92 1.98 }`, and the braces are the part a naive split gets wrong.
pub fn parse_load_average(text: &str) -> Option<LoadAverage> {
    let mut parts = text
        .split(|c: char| c.is_whitespace() || c == '{' || c == '}')
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<f64>().ok());
    let one_min = parts.next()?;
    let five_min = parts.next()?;
    let fifteen_min = parts.next()?;
    Some(LoadAverage {
        one_min,
        five_min,
        fifteen_min,
    })
}

/// Best-effort "is this a container?".
///
/// The same two probes `libviprs-bench` uses. It matters here because the
/// published PMTiles numbers were taken in one and the document did not say
/// so, which is half of why nobody spotted that they were emulated.
fn detect_container() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    match std::fs::read_to_string("/proc/1/cgroup") {
        Ok(text) => {
            text.contains("docker") || text.contains("kubepods") || text.contains("containerd")
        }
        Err(_) => false,
    }
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
