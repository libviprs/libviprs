//! The streaming, bounded-memory PMTiles v3 writer.
//!
//! [`Writer`] takes tiles in **any order**, stores each distinct payload
//! exactly once, and assembles a spec-correct archive at [`Writer::finish`]
//! with a staged temp file, an `fsync` and an atomic rename. Peak memory does
//! not grow with the number of tiles: the per-tile index is spilled to an
//! append-only log on disk and externally sorted at finalize, and the
//! directories are built one leaf at a time.
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
//! The tiebreaker is structural rather than a matter of taste. The root
//! directory has to fit inside the first 16384 bytes of the archive, so its
//! compressed size decides whether the entries spill into leaves, and that is
//! not known until every entry has been sorted. The header, the root, the
//! metadata and the leaf section therefore cannot be sized until the last tile
//! has arrived, which means **the staged payloads get copied at finalize
//! whatever layout is chosen**. Given the copy is happening anyway, doing it
//! in tile id order costs one extra pass over the staged data and buys three
//! things: a deterministic archive, an honest `clustered = true`, and a reader
//! whose sequential scan of a zoom level is a sequential scan of the file.
//!
//! So this writer stages payloads in arrival order, sorts at finalize, and
//! writes the data region in tile id order. `clustered` is `true` and it is
//! true.
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
//! # What is bounded and what is not
//!
//! Bounded, and independent of the tile count: the per-tile index (spilled,
//! sorted in fixed-size runs), the entry list (spilled, streamed into leaves),
//! the leaf section (spilled), and the payload copy (one blob at a time).
//!
//! **Not** bounded by the tile count but bounded by the number of *distinct
//! payloads*: the content hash table, and the map from a staged offset to a
//! final one. That is inherent to content dedupe and it is the same bound
//! [`DedupeIndex`](crate::dedupe::DedupeIndex) already carries. A pyramid of
//! mostly blank tiles has very few distinct payloads, which is the case this
//! exists for.
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
use std::collections::{BinaryHeap, HashMap};
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
const DEFAULT_LEAF_ENTRIES: usize = 4096;

/// How many spill records are sorted in memory before a run is written out.
///
/// 1 Mi records is 20 MB of buffer, and it keeps the number of runs (and so
/// the number of open file handles during the merge) low: a billion tiles is
/// 954 runs, which is a merge a `BinaryHeap` handles without noticing.
const SORT_RUN_RECORDS: usize = 1 << 20;

/// One spilled index record: `tile_id`, staged offset, length.
const SPILL_RECORD_BYTES: usize = 8 + 8 + 4;

/// One spilled directory entry: `tile_id`, final offset, length, run length.
const ENTRY_RECORD_BYTES: usize = 8 + 8 + 4 + 4;

/// Copy buffer for moving staged payloads and leaf bytes into the archive.
const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// Disambiguates the scratch prefix of two writers sharing one directory.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

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

/// One record in the append-only index log: which tile, and which staged
/// payload it points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Spill {
    tile_id: u64,
    staged_offset: u64,
    length: u32,
}

impl Spill {
    fn encode(&self) -> [u8; SPILL_RECORD_BYTES] {
        let mut out = [0u8; SPILL_RECORD_BYTES];
        out[0..8].copy_from_slice(&self.tile_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.staged_offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.length.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8; SPILL_RECORD_BYTES]) -> Self {
        Self {
            tile_id: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
            staged_offset: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
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
    staged: Option<BufWriter<File>>,
    staged_len: u64,
    /// The append-only index log.
    log: Option<BufWriter<File>>,
    log_len: u64,

    /// Distinct payloads: content hash to `(staged offset, length)`.
    payloads: HashMap<[u8; 32], (u64, u32)>,
    /// Records not yet written to a run.
    sort_buffer: Vec<Spill>,
    /// Runs already written to the log.
    runs: Vec<Run>,

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
            .field("distinct_payloads", &self.payloads.len())
            .field("staged_bytes", &self.staged_len)
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
            staged: Some(BufWriter::new(staged)),
            staged_len: 0,
            log: Some(BufWriter::new(log)),
            log_len: 0,
            payloads: HashMap::new(),
            sort_buffer: Vec::new(),
            runs: Vec::new(),
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

        let staged_offset = match self.payloads.get(&content_hash) {
            Some(&(offset, stored)) => {
                if stored != length {
                    return Err(PmTilesError::ContentHashMismatch { length, stored });
                }
                offset
            }
            None => {
                let offset = self.staged_len;
                self.staged
                    .as_mut()
                    .expect("a live writer has its staging file")
                    .write_all(bytes)?;
                self.staged_len = self.staged_len.checked_add(u64::from(length)).ok_or(
                    PmTilesError::Overflow {
                        what: "the staged payload region",
                    },
                )?;
                self.payloads.insert(content_hash, (offset, length));
                offset
            }
        };

        self.push_spill(Spill {
            tile_id,
            staged_offset,
            length,
        })?;

        self.tile_count += 1;
        self.min_zoom = self.min_zoom.min(z);
        self.max_zoom = self.max_zoom.max(z);
        Ok(())
    }

    /// How many tiles have been added so far. Runs are not collapsed until
    /// [`finish`](Writer::finish), so this counts addressed tiles.
    pub fn tile_count(&self) -> u64 {
        self.tile_count
    }

    /// How many distinct payloads are staged so far.
    pub fn distinct_payload_count(&self) -> usize {
        self.payloads.len()
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
            staged.flush()?;
            staged.get_ref().sync_data()?;
        }
        if let Some(log) = self.log.as_mut() {
            log.flush()?;
            log.get_ref().sync_data()?;
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
    fn flush_run(&mut self) -> Result<(), PmTilesError> {
        if self.sort_buffer.is_empty() {
            return Ok(());
        }
        self.sort_buffer.sort_unstable();
        let log = self.log.as_mut().expect("a live writer has its log");
        let start = self.log_len;
        let count = self.sort_buffer.len() as u64;
        for record in &self.sort_buffer {
            log.write_all(&record.encode())?;
        }
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
        // There is deliberately no early "did you add any tiles" check here.
        // A zero-tile archive is refused by `serialize_entries`, which is
        // where the spec's "MUST be greater than 0" lives, and a second
        // refusal in front of it would be a guard no test can distinguish
        // from the real one: removing it changes no observable behaviour,
        // which makes it exactly the kind of code that rots unnoticed.
        self.flush_run()?;

        // Both scratch files are done being written. Flush them through their
        // buffers and close the handles before anything reads them back.
        if let Some(staged) = self.staged.take() {
            staged
                .into_inner()
                .map_err(|e| e.into_error())?
                .sync_data()?;
        }
        if let Some(log) = self.log.take() {
            log.into_inner().map_err(|e| e.into_error())?.sync_data()?;
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
    /// Distinct payloads, in the order they are written into the archive,
    /// as `(staged offset, length)`.
    order: Vec<(u64, u32)>,
    tile_data_length: u64,
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

        let mut final_offsets: HashMap<u64, u64> = HashMap::with_capacity(self.payloads.len());
        let mut order: Vec<(u64, u32)> = Vec::with_capacity(self.payloads.len());
        let mut next_offset: u64 = 0;

        let mut entry_count: u64 = 0;
        let mut addressed: u64 = 0;
        let mut open: Option<Entry> = None;
        let mut previous_id: Option<u64> = None;

        let mut source = SortedSpill::open(&suffixed(&self.base, ".idx"), &self.runs)?;
        while let Some(record) = source.next_record()? {
            if previous_id == Some(record.tile_id) {
                return Err(PmTilesError::DuplicateTile {
                    tile_id: record.tile_id,
                });
            }
            previous_id = Some(record.tile_id);

            let offset = match final_offsets.get(&record.staged_offset) {
                Some(&offset) => offset,
                None => {
                    let offset = next_offset;
                    next_offset = next_offset.checked_add(u64::from(record.length)).ok_or(
                        PmTilesError::Overflow {
                            what: "the tile data section",
                        },
                    )?;
                    final_offsets.insert(record.staged_offset, offset);
                    order.push((record.staged_offset, record.length));
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
        out.into_inner().map_err(|e| e.into_error())?.sync_data()?;

        Ok(Plan {
            entries_path,
            entry_count,
            addressed_tiles: addressed,
            order,
            tile_data_length: next_offset,
        })
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
            leaf_file
                .into_inner()
                .map_err(|e| e.into_error())?
                .sync_data()?;

            let root = self
                .options
                .internal_compression
                .compress(&serialize_entries(&pointers)?)?;
            if root.len() <= ROOT_BUDGET {
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
                return Err(PmTilesError::RootDirectoryTooLarge {
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
            return Err(PmTilesError::RootDirectoryTooLarge {
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
            tile_contents_count: plan.order.len() as u64,
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
        for &(staged_offset, length) in &plan.order {
            staged_data.seek(SeekFrom::Start(staged_offset))?;
            copy_exactly(&mut staged_data, out, u64::from(length), &mut buffer)?;
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
                file.sync_all()?;
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

/// One run's cursor during the k-way merge.
struct RunCursor {
    reader: BufReader<File>,
    left: u64,
    head: Option<Spill>,
}

impl RunCursor {
    fn open(path: &Path, run: Run) -> Result<Self, PmTilesError> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(run.start))?;
        let mut cursor = Self {
            reader: BufReader::new(file),
            left: run.count,
            head: None,
        };
        cursor.advance()?;
        Ok(cursor)
    }

    fn advance(&mut self) -> Result<(), PmTilesError> {
        if self.left == 0 {
            self.head = None;
            return Ok(());
        }
        let mut bytes = [0u8; SPILL_RECORD_BYTES];
        self.reader.read_exact(&mut bytes)?;
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
/// inside a 20 MB buffer.
struct SortedSpill {
    cursors: Vec<RunCursor>,
    queue: BinaryHeap<Reverse<(u64, usize)>>,
}

impl SortedSpill {
    fn open(path: &Path, runs: &[Run]) -> Result<Self, PmTilesError> {
        let mut cursors = Vec::with_capacity(runs.len());
        let mut queue = BinaryHeap::with_capacity(runs.len());
        for (index, run) in runs.iter().enumerate() {
            let cursor = RunCursor::open(path, *run)?;
            if let Some(head) = cursor.head {
                queue.push(Reverse((head.tile_id, index)));
            }
            cursors.push(cursor);
        }
        Ok(Self { cursors, queue })
    }

    fn next_record(&mut self) -> Result<Option<Spill>, PmTilesError> {
        let Some(Reverse((_, index))) = self.queue.pop() else {
            return Ok(None);
        };
        let record = self.cursors[index].head.expect("a queued run has a head");
        self.cursors[index].advance()?;
        if let Some(head) = self.cursors[index].head {
            self.queue.push(Reverse((head.tile_id, index)));
        }
        Ok(Some(record))
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

    #[test]
    fn a_spill_record_round_trips_through_its_twenty_bytes() {
        let record = Spill {
            tile_id: u64::MAX - 3,
            staged_offset: 1 << 40,
            length: u32::MAX,
        };
        assert_eq!(Spill::decode(&record.encode()), record);
        assert_eq!(record.encode().len(), SPILL_RECORD_BYTES);
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
                        staged_offset: *id * 10,
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

    /// The sort buffer spills once it fills, and the merge still produces one
    /// ascending sequence. Driven through the public surface with a lowered
    /// run size would be better, but the constant is not configurable, so this
    /// drives the same machinery directly with three runs' worth of records.
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
