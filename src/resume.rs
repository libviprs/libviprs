//! Resumable pyramid generation — building blocks (Phase 3).
//!
//! This module provides the on-disk checkpoint format, plan-hash computation,
//! and helper types used by `generate_pyramid_resumable` (the end-to-end
//! entry point lives in [`crate::engine`] and is wired up separately).
//!
//! # Checkpoint format
//!
//! Each output directory contains a small JSON header, `.libviprs-job.json`,
//! whose contents deserialise to [`JobMetadata`]. The header is written
//! atomically via a `.tmp` sibling + rename so that a crash mid-write cannot
//! produce a torn or partially-updated checkpoint. The sibling name is made
//! unique per process (pid + counter) so two jobs sharing one output directory
//! cannot stage into, and torn-rename over, the same temp file (issue #126).
//!
//! The list of completed tile coordinates does *not* live inline in that
//! header on the hot path. Re-serialising the whole vector on every periodic
//! flush made per-flush I/O proportional to all completed tiles, so a
//! libvips-scale run re-wrote hundreds of gigabytes cumulatively (issue #127).
//! Instead, [`CheckpointState`](crate::engine) appends only the coordinates
//! completed *since the last flush* to an append-only side log,
//! `.libviprs-job.segments`, keeping each periodic flush bounded. The header
//! carries the small, fixed set of scalar fields (schema, plan hash, completed
//! levels, timestamps, format) plus any coordinates that were already inline
//! at resume time (a hand-written or legacy checkpoint). [`JobCheckpoint::load`]
//! reconstructs the full completed set by merging the header's inline
//! coordinates with the segment log, so callers see one coherent
//! [`JobMetadata`] regardless of how the checkpoint was written. A trailing
//! partially-written segment record (a crash mid-append) is ignored on read;
//! the coordinates before it are still recovered.
//!
//! # Plan hashing
//!
//! A run may resume only if the current [`PyramidPlan`] and its output
//! *content contract* match the ones originally used to produce the
//! checkpoint. [`compute_plan_hash`] serialises the plan's load-bearing
//! geometry together with a [`PlanContract`] (tile format, padding colour,
//! blank-tile strategy, dedupe layout, and an optional source digest) into a
//! canonical byte layout and hashes them with Blake3. Any change to tile
//! size, overlap, layout, level count, per-level dimensions, or the content
//! contract changes the hash — so a mismatched run is detected before a
//! single tile is written.
//!
//! # Intended use
//!
//! ```ignore
//! use libviprs::resume::{JobCheckpoint, JobMetadata, PlanContract, compute_plan_hash};
//!
//! let contract = PlanContract::from_engine(&config, &sink);
//! let hash = compute_plan_hash(&plan, &contract);
//! let meta = JobMetadata {
//!     schema_version: "1".to_string(),
//!     plan_hash: hash,
//!     completed_tiles: Vec::new(),
//!     levels_completed: Vec::new(),
//!     started_at: now_rfc3339(),
//!     last_checkpoint_at: now_rfc3339(),
//!     content_format: contract.format,
//! };
//! JobCheckpoint::save(output_dir, &meta)?;
//! ```

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dedupe::DedupeStrategy;
use crate::engine::{BlankTileStrategy, EngineConfig};
use crate::manifest::ChecksumAlgo;
use crate::planner::{Layout, PyramidPlan, TileCoord};
use crate::sink::{TileFormat, TileSink};

// `generate_pyramid_resumable` is now the exclusive purview of
// `EngineBuilder::with_resume` — no crate-external entry point.

/// Current on-disk schema version for [`JobMetadata`].
///
/// Bumping this value forces older checkpoints to be rejected with
/// [`ResumeError::SchemaMismatch`], preventing a newer binary from
/// misinterpreting a legacy layout.
pub const SCHEMA_VERSION: &str = "1";

/// Well-known filename for the on-disk job checkpoint.
///
/// Always lives directly inside the output directory (the tile sink's base
/// path). Relative path: `<output_dir>/.libviprs-job.json`.
pub const CHECKPOINT_FILENAME: &str = ".libviprs-job.json";

/// Well-known filename for the append-only completed-coordinate log that sits
/// alongside [`CHECKPOINT_FILENAME`].
///
/// Each record is a fixed 12-byte little-endian frame `(level, col, row)`.
/// The engine appends only the coordinates completed since the previous flush,
/// which keeps per-flush I/O bounded instead of proportional to every tile
/// completed so far (issue #127). [`JobCheckpoint::load`] replays this log and
/// merges it with the header's inline coordinates. Relative path:
/// `<output_dir>/.libviprs-job.segments`.
pub const SEGMENTS_FILENAME: &str = ".libviprs-job.segments";

/// On-disk size, in bytes, of one segment record: three little-endian `u32`
/// fields `(level, col, row)`. A trailing run of fewer than this many bytes is
/// a torn final append and is ignored on read.
pub(crate) const SEGMENT_RECORD_LEN: usize = 12;

/// Well-known filename for the advisory run lock guarding a checkpoint root.
///
/// Lives directly inside the checkpoint directory alongside
/// [`CHECKPOINT_FILENAME`]. Relative path: `<dir>/.libviprs-job.lock`. Held for
/// the duration of a run via [`RunLock`] so two jobs cannot share one output
/// directory (issue #126).
pub const LOCK_FILENAME: &str = ".libviprs-job.lock";

/// Behaviour selector for resumable pyramid generation.
///
/// * [`ResumeMode::Overwrite`] — wipe any pre-existing output and start fresh.
///   This is the default and matches the behaviour of the non-resumable entry
///   points.
/// * [`ResumeMode::Resume`] — read the on-disk checkpoint, skip tiles that are
///   already recorded as completed, and write only what remains. Refuses to
///   proceed if the stored `plan_hash` disagrees with the current plan.
/// * [`ResumeMode::Verify`] — do not write anything. Walk the plan and check
///   that every tile is present and internally consistent on disk. Useful for
///   post-hoc validation of a finished job.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResumeMode {
    /// Discard pre-existing output and regenerate every tile.
    #[default]
    Overwrite,
    /// Skip tiles already recorded in the on-disk checkpoint.
    Resume,
    /// Verify on-disk tiles without producing new output.
    Verify,
}

/// Default flush cadence seeded by [`ResumePolicy::resume`]: persist the
/// checkpoint every 1000 completed tiles. Chosen as a balance between crash
/// granularity (at most ~1000 tiles re-rendered after an interruption) and
/// filesystem churn (one small JSON rewrite per 1000 writes). Overwrite and
/// Verify keep the "final flush only" default of `0`.
pub const DEFAULT_RESUME_CHECKPOINT_EVERY: u64 = 1000;

/// Fluent builder bundling a [`ResumeMode`] with its checkpoint-persistence
/// knobs.
///
/// `ResumePolicy` is the user-facing type threaded through `EngineBuilder`.
/// Three mutually-exclusive factories anchor the mode:
///
/// * [`ResumePolicy::overwrite`] — fresh run; any existing checkpoint is
///   wiped. Matches [`ResumeMode::Overwrite`] and is the `Default`.
/// * [`ResumePolicy::resume`] — skip tiles already recorded in the on-disk
///   checkpoint. Matches [`ResumeMode::Resume`].
/// * [`ResumePolicy::verify`] — read-only audit of existing tiles. Matches
///   [`ResumeMode::Verify`].
///
/// After picking a factory, chain `.with_checkpoint_every(n)` to flush the
/// checkpoint file every `n` completed tiles (`0` = final flush only) and
/// `.with_checkpoint_root(path)` to place the checkpoint somewhere other
/// than the sink's base directory. [`ResumePolicy::resume`] seeds a non-zero
/// cadence ([`DEFAULT_RESUME_CHECKPOINT_EVERY`]); Overwrite and Verify start
/// at `0`.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumePolicy {
    mode: ResumeMode,
    checkpoint_every: u64,
    checkpoint_root: Option<PathBuf>,
}

impl Default for ResumePolicy {
    fn default() -> Self {
        Self {
            mode: ResumeMode::Overwrite,
            checkpoint_every: 0,
            checkpoint_root: None,
        }
    }
}

impl ResumePolicy {
    /// Start a fresh run. Equivalent to [`ResumeMode::Overwrite`].
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-overwrite)
    pub fn overwrite() -> Self {
        Self {
            mode: ResumeMode::Overwrite,
            ..Self::default()
        }
    }

    /// Continue an interrupted run from the on-disk checkpoint. Equivalent
    /// to [`ResumeMode::Resume`].
    ///
    /// Unlike [`ResumePolicy::overwrite`] / [`ResumePolicy::verify`], this
    /// factory seeds a non-zero flush cadence
    /// ([`DEFAULT_RESUME_CHECKPOINT_EVERY`]) so that a long run periodically
    /// persists progress even when the caller never sets an explicit cadence
    /// — the whole point of Resume mode is that an interruption does not
    /// throw away completed work. Override it with
    /// [`ResumePolicy::with_checkpoint_every`] (including `0` to defer the
    /// write to the end of the run).
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
    pub fn resume() -> Self {
        Self {
            mode: ResumeMode::Resume,
            checkpoint_every: DEFAULT_RESUME_CHECKPOINT_EVERY,
            ..Self::default()
        }
    }

    /// Audit an existing output directory against the plan without writing
    /// anything. Equivalent to [`ResumeMode::Verify`].
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-verify)
    pub fn verify() -> Self {
        Self {
            mode: ResumeMode::Verify,
            ..Self::default()
        }
    }

    /// Flush the checkpoint file every `n` completed tiles. `0` defers the
    /// checkpoint write until the end of the run. Overrides the cadence
    /// seeded by the factory (notably the non-zero
    /// [`DEFAULT_RESUME_CHECKPOINT_EVERY`] set by [`ResumePolicy::resume`]).
    pub fn with_checkpoint_every(mut self, n: u64) -> Self {
        self.checkpoint_every = n;
        self
    }

    /// Override the checkpoint directory. When unset, the engine falls back
    /// to the sink's `checkpoint_root()` (typically the output base dir).
    ///
    /// Two supported uses, both kept by design when issue #137 shrank the
    /// `TileSink` bookkeeping surface:
    /// * place the checkpoint somewhere other than the output directory;
    /// * drive resume and Verify through an opaque external sink wrapper
    ///   that does not forward `checkpoint_root()`. The crate's own
    ///   wrappers forward it via `TileSink::inner_sink`, but a wrapper in
    ///   user code may only forward the data path, and this override is
    ///   the documented escape hatch for exactly that sink.
    pub fn with_checkpoint_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.checkpoint_root = Some(path.into());
        self
    }

    /// The resume mode this policy lowers to.
    pub fn mode(&self) -> ResumeMode {
        self.mode
    }

    /// The configured checkpoint frequency (`0` means "only at the end").
    pub fn checkpoint_every(&self) -> u64 {
        self.checkpoint_every
    }

    /// The configured checkpoint root, if one was set via
    /// [`ResumePolicy::with_checkpoint_root`].
    pub fn checkpoint_root(&self) -> Option<&Path> {
        self.checkpoint_root.as_deref()
    }
}

/// On-disk checkpoint describing the state of a pyramid generation job.
///
/// Produced and consumed by [`JobCheckpoint::save`] / [`JobCheckpoint::load`].
/// The struct is intentionally simple and flat so that it serialises cleanly
/// as JSON — a debugger or shell user can inspect it with `cat` / `jq`.
///
/// `schema_version` is stored as a [`String`] so we can read back old
/// checkpoints, compare them against [`SCHEMA_VERSION`], and return a
/// structured error if they disagree.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct JobMetadata {
    /// On-disk schema version. Equal to [`SCHEMA_VERSION`] ("1") for
    /// checkpoints produced by this crate version. Stored as a plain
    /// [`String`] so the value read from disk is preserved verbatim and can
    /// be compared against the binary's expected version.
    pub schema_version: String,
    /// Lowercase hex Blake3 digest of the plan's canonical byte
    /// representation (see [`compute_plan_hash`]).
    pub plan_hash: String,
    /// Coordinates of every tile that has been successfully written and
    /// flushed since the job started.
    ///
    /// Uses the `tile_coord_vec_serde` adapter because [`TileCoord`] in
    /// `crate::planner` does not itself implement [`Serialize`] /
    /// [`Deserialize`].
    #[serde(with = "tile_coord_vec_serde")]
    pub completed_tiles: Vec<TileCoord>,
    /// Level indices that have been fully completed (every tile in the level
    /// is present in `completed_tiles`). Populated eagerly so resumption can
    /// skip whole levels without re-checking each tile.
    #[serde(default)]
    pub levels_completed: Vec<u32>,
    /// RFC 3339 timestamp captured when the job first started (Overwrite
    /// mode) or when an existing checkpoint was first resumed.
    #[serde(default)]
    pub started_at: String,
    /// RFC 3339 timestamp of the most recent checkpoint write.
    #[serde(default)]
    pub last_checkpoint_at: String,
    /// Tile encoding the sink committed to when this checkpoint was written,
    /// mirroring [`PlanContract::format`]. Recorded explicitly (rather than
    /// only being folded into `plan_hash`) so the resume gate can reproduce
    /// the write-time hash without depending on the *live* sink to report the
    /// format — a transparent wrapper (recording / retry / tee) may not
    /// forward [`crate::sink::TileSink::content_format`]. Defaults to `None`
    /// for checkpoints written before this field existed and for sinks that
    /// do not pin a single on-disk format.
    #[serde(default)]
    pub content_format: Option<TileFormat>,
}

impl JobMetadata {
    /// Construct a fresh [`JobMetadata`] tagged with the current
    /// [`SCHEMA_VERSION`]. All other fields default to empty / zero values;
    /// callers fill them in as the job progresses.
    pub fn new(plan_hash: String, started_at: String) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            plan_hash,
            completed_tiles: Vec::new(),
            levels_completed: Vec::new(),
            last_checkpoint_at: started_at.clone(),
            started_at,
            content_format: None,
        }
    }
}

/// Serde adapter for `Vec<TileCoord>`.
///
/// [`TileCoord`] lives in the `planner` module and does not implement
/// [`Serialize`] / [`Deserialize`] directly — wiring serde into that module
/// is out of scope for the resume module. Instead we serialise each coord as
/// a small `{ level, col, row }` JSON object via a local shadow struct.
pub(super) mod tile_coord_vec_serde {
    use super::TileCoord;
    use serde::de::{SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    /// Upper bound on the number of elements we will speculatively reserve
    /// from a deserializer's reported `size_hint` before reading any data.
    ///
    /// The hint is advisory and, for a self-describing or length-prefixed
    /// wire format, is attacker-controlled: a corrupt or hostile checkpoint
    /// could otherwise advertise a gigantic sequence length and trigger a
    /// multi-gigabyte speculative allocation before a single element is read.
    /// We cap the reservation so a bogus hint costs at most this many slots;
    /// the `Vec` still grows normally if the sequence genuinely turns out to
    /// be longer, so honest large checkpoints remain correct.
    pub(super) const MAX_SEQ_PREALLOC: usize = 4096;

    /// Bound a deserializer-reported sequence length by [`MAX_SEQ_PREALLOC`].
    ///
    /// A missing hint reserves nothing; any reported hint is clamped to the
    /// cap so the pre-read reservation is fixed regardless of the value the
    /// format claims.
    pub(super) fn capped_prealloc(size_hint: Option<usize>) -> usize {
        size_hint.unwrap_or(0).min(MAX_SEQ_PREALLOC)
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct CoordShadow {
        pub(super) level: u32,
        pub(super) col: u32,
        pub(super) row: u32,
    }

    impl From<&TileCoord> for CoordShadow {
        fn from(c: &TileCoord) -> Self {
            Self {
                level: c.level,
                col: c.col,
                row: c.row,
            }
        }
    }

    impl From<CoordShadow> for TileCoord {
        fn from(s: CoordShadow) -> Self {
            TileCoord {
                level: s.level,
                col: s.col,
                row: s.row,
            }
        }
    }

    pub fn serialize<S>(coords: &[TileCoord], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(coords.len()))?;
        for c in coords {
            seq.serialize_element(&CoordShadow::from(c))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<TileCoord>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<TileCoord>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sequence of {level,col,row} tile coordinates")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut out = Vec::with_capacity(capped_prealloc(seq.size_hint()));
                while let Some(shadow) = seq.next_element::<CoordShadow>()? {
                    out.push(shadow.into());
                }
                Ok(out)
            }
        }
        deserializer.deserialize_seq(V)
    }
}

/// Errors that can occur while reading, writing, or validating a checkpoint.
///
/// `Io(io::Error)` wraps filesystem failures from the underlying
/// [`std::fs`] calls; `SchemaMismatch` surfaces a semantic incompatibility
/// that makes it unsafe to resume.
///
/// A checkpoint whose `plan_hash` disagrees with the current plan is *not*
/// reported here: that divergence is detected at the resume gate
/// (`verify_checkpoint_contract`) and surfaced to callers as
/// [`crate::engine::EngineError::PlanHashMismatch`], which carries the
/// additional format-change context a fixed-field checkpoint error could not.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ResumeError {
    /// The checkpoint's `schema_version` does not match [`SCHEMA_VERSION`].
    #[error("checkpoint schema mismatch: binary speaks version {expected}, file declares {found}")]
    SchemaMismatch {
        /// Schema version this binary knows how to read.
        expected: &'static str,
        /// Schema version declared by the on-disk checkpoint.
        found: String,
    },
    /// The checkpoint file exists but does not deserialise as valid JSON.
    ///
    /// Distinct from [`ResumeError::Io`] so callers can tell "couldn't read
    /// the file" from "read the file but couldn't parse it" — the latter
    /// indicates a corrupt or truncated checkpoint.
    #[error("checkpoint at {path} is corrupt: {source}")]
    Corrupt {
        /// Absolute path of the malformed checkpoint file.
        path: PathBuf,
        /// Underlying serde_json parse error.
        #[source]
        source: serde_json::Error,
    },
    /// Another live job already holds the advisory run lock on a run/output
    /// directory.
    ///
    /// Returned by [`RunLock::acquire`] when a concurrent Resume or Overwrite
    /// run owns the same directory. A run locks the union of the directories it
    /// mutates — the resolved checkpoint root AND the sink's own output dir — so
    /// `path` names whichever of those the colliding job already holds, not just
    /// the checkpoint root. Refusing here — rather than running anyway — is what
    /// stops two jobs from interleaving checkpoint flushes (each overwriting the
    /// other's `completed_tiles`, last-writer-wins) or letting one job's
    /// Overwrite wipe delete another job's in-flight output (issue #126).
    #[error("run directory is locked by another live job: {path}")]
    Locked {
        /// Path of the lock file whose exclusive lock is already held.
        path: PathBuf,
    },
    /// Underlying filesystem error.
    #[error("checkpoint I/O error: {0}")]
    Io(#[from] io::Error),
}

/// Unit struct grouping filesystem operations against a checkpoint directory.
///
/// The on-disk format has no hidden state beyond a single JSON file, so this
/// type is purely a namespace for `load` / `save` / `checkpoint_path` rather
/// than a live handle. Callers that want to hold onto the last-known metadata
/// should keep their own [`JobMetadata`] around.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume)
pub struct JobCheckpoint;

impl JobCheckpoint {
    /// Absolute path of the checkpoint file for the given output directory.
    ///
    /// Returns `<dir>/.libviprs-job.json` without checking whether the file
    /// actually exists.
    pub fn checkpoint_path(dir: &Path) -> PathBuf {
        dir.join(CHECKPOINT_FILENAME)
    }

    /// Absolute path of the append-only completed-coordinate segment log for
    /// the given output directory.
    ///
    /// Returns `<dir>/.libviprs-job.segments` without checking whether the
    /// file actually exists.
    pub fn segments_path(dir: &Path) -> PathBuf {
        dir.join(SEGMENTS_FILENAME)
    }

    /// Load and deserialise the checkpoint from `dir`.
    ///
    /// * `Ok(None)` — the checkpoint file does not exist.
    /// * `Ok(Some(meta))` — the file exists, parses cleanly, and its
    ///   `schema_version` matches [`SCHEMA_VERSION`].
    /// * `Err(ResumeError::Io)` — the file exists but could not be read.
    /// * `Err(ResumeError::Corrupt)` — the file exists but does not parse as
    ///   valid JSON for [`JobMetadata`].
    /// * `Err(ResumeError::SchemaMismatch)` — the file parsed but declares a
    ///   `schema_version` this binary does not understand.
    ///
    /// Corrupt and mismatched checkpoints are surfaced as errors rather than
    /// swallowed as `None` so callers do not silently overwrite a file that
    /// might be recoverable.
    pub fn load(dir: &Path) -> Result<Option<JobMetadata>, ResumeError> {
        let path = Self::checkpoint_path(dir);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(ResumeError::Io(e)),
        };
        let mut meta: JobMetadata = serde_json::from_slice(&bytes)
            .map_err(|source| ResumeError::Corrupt { path, source })?;
        if meta.schema_version != SCHEMA_VERSION {
            return Err(ResumeError::SchemaMismatch {
                expected: SCHEMA_VERSION,
                found: meta.schema_version,
            });
        }

        // Merge the append-only segment log (issue #127). The completed set is
        // reconstructed as `segment coordinates ++ header-inline coordinates`,
        // deduplicated with first-seen order preserved. Segment coordinates
        // come first so their count lines up with the physical frame count in
        // the file, which `CheckpointState` uses to decide which coordinates
        // are already durable versus still inline in the header.
        let segment_coords = read_segment_log(&Self::segments_path(dir))?;
        if !segment_coords.is_empty() {
            let mut seen = std::collections::HashSet::with_capacity(
                segment_coords.len() + meta.completed_tiles.len(),
            );
            let inline = std::mem::take(&mut meta.completed_tiles);
            let mut merged = Vec::with_capacity(segment_coords.len() + inline.len());
            for coord in segment_coords.into_iter().chain(inline) {
                if seen.insert(coord) {
                    merged.push(coord);
                }
            }
            meta.completed_tiles = merged;
        }

        Ok(Some(meta))
    }

    /// Persist `meta` to `<dir>/.libviprs-job.json` atomically.
    ///
    /// The payload is written to a `.tmp` sibling first and then renamed over
    /// the final path. On POSIX filesystems this rename is atomic, so a crash
    /// mid-write cannot leave a torn checkpoint — the old file either remains
    /// intact or is fully replaced by the new one.
    // TODO(windows): `std::fs::rename` is not atomic-replace on Windows; switch to `ReplaceFileW` (dtolnay #9).
    pub fn save(dir: &Path, meta: &JobMetadata) -> io::Result<()> {
        Self::save_with(dir, meta, &RealDurability)
    }

    /// [`JobCheckpoint::save`] with an injectable [`Durability`] backend.
    ///
    /// Production callers go through [`JobCheckpoint::save`], which supplies
    /// [`RealDurability`] (issuing genuine `fsync` calls). Tests inject a
    /// recorder to assert the durability ordering — in particular that the
    /// tile data a checkpoint certifies is fsynced *before* the checkpoint is
    /// renamed into place and its directory is fsynced.
    pub(crate) fn save_with(
        dir: &Path,
        meta: &JobMetadata,
        durability: &dyn Durability,
    ) -> io::Result<()> {
        // Make sure the target directory exists; callers typically create it,
        // but checkpointing should not fail just because the sink has not yet
        // materialised a sub-tree.
        std::fs::create_dir_all(dir)?;

        let final_path = Self::checkpoint_path(dir);
        let tmp_path = tmp_path_for(&final_path);

        let bytes = serde_json::to_vec_pretty(meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Scope the file handle so it's closed before the rename — some
        // filesystems refuse to rename over an open file handle. The
        // checkpoint payload itself is fsynced here so its bytes are durable
        // before the rename publishes it.
        {
            let mut f = std::fs::File::create(&tmp_path)?;
            if let Err(e) = f.write_all(&bytes).and_then(|()| f.sync_all()) {
                // Don't leave a partially-written unique temp file behind.
                let _ = std::fs::remove_file(&tmp_path);
                return Err(e);
            }
        }

        if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        // Fsync the containing directory so the rename itself is durable. A
        // synced checkpoint file whose *directory entry* is still in the page
        // cache can vanish on power loss, taking the resume record with it.
        durability.sync_dir(dir)?;
        Ok(())
    }
}

/// Encode a single tile coordinate as its fixed 12-byte little-endian segment
/// frame `(level, col, row)`.
fn encode_segment_frame(coord: &TileCoord) -> [u8; SEGMENT_RECORD_LEN] {
    let mut buf = [0u8; SEGMENT_RECORD_LEN];
    buf[0..4].copy_from_slice(&coord.level.to_le_bytes());
    buf[4..8].copy_from_slice(&coord.col.to_le_bytes());
    buf[8..12].copy_from_slice(&coord.row.to_le_bytes());
    buf
}

/// Count the number of complete records currently in the segment log without
/// parsing them. A trailing partial frame (a torn append) is not counted.
///
/// Returns `0` when the file does not exist. `CheckpointState` uses this at
/// construction to learn how many completed coordinates are already durable in
/// the log versus still carried inline in the header.
pub(crate) fn count_segment_frames(path: &Path) -> io::Result<usize> {
    match std::fs::metadata(path) {
        Ok(md) => Ok((md.len() as usize) / SEGMENT_RECORD_LEN),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

/// Replay the append-only segment log at `path`, decoding every complete
/// 12-byte frame into a [`TileCoord`].
///
/// A trailing run shorter than one record is a crash mid-append and is
/// dropped: the coordinates written before it are intact and returned. Missing
/// file yields an empty vector.
fn read_segment_log(path: &Path) -> Result<Vec<TileCoord>, ResumeError> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(ResumeError::Io(e)),
    };
    let frames = bytes.len() / SEGMENT_RECORD_LEN;
    let mut coords = Vec::with_capacity(frames);
    // Only the complete frames: `as_chunks` hands back any torn trailing
    // partial record as the `.1` remainder, which is exactly what this
    // replay drops.
    for chunk in bytes.as_chunks::<SEGMENT_RECORD_LEN>().0 {
        let level = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let col = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let row = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        coords.push(TileCoord::new(level, col, row));
    }
    Ok(coords)
}

/// Append completed coordinates to the segment log at `<dir>/.libviprs-job.segments`.
///
/// The write is a single `write_all` of the concatenated frames followed by a
/// data fsync, so the appended coordinates are durable before the caller
/// rewrites the header that summarises them. When the log did not exist yet
/// (the first append of a run), the containing directory is fsynced too so the
/// new file's directory entry survives power loss.
///
/// Callers must serialise their appends: the engine drives this from a single
/// flush critical section so two workers never interleave partial frames.
pub(crate) fn append_segments(
    dir: &Path,
    coords: &[TileCoord],
    durability: &dyn Durability,
) -> io::Result<()> {
    if coords.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    let path = JobCheckpoint::segments_path(dir);
    let is_new = !path.exists();

    let mut buf = Vec::with_capacity(coords.len() * SEGMENT_RECORD_LEN);
    for coord in coords {
        buf.extend_from_slice(&encode_segment_frame(coord));
    }

    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }

    // A freshly created log needs its directory entry flushed as well;
    // subsequent appends only grow an already-linked file.
    if is_new {
        durability.sync_dir(dir)?;
    }
    Ok(())
}

/// Durability backend used by the checkpoint/sink write paths to issue
/// `fsync`-equivalent calls.
///
/// The production implementation ([`RealDurability`]) performs genuine
/// `File::sync_all` on files and directories. The trait exists so tests can
/// inject a recorder and assert the *ordering* of durability operations
/// (tile data must be synced before the checkpoint that certifies it), which
/// is otherwise invisible in on-disk state.
pub(crate) trait Durability {
    /// Flush a file's data and metadata to stable storage.
    fn sync_file(&self, path: &Path) -> io::Result<()>;
    /// Flush a directory entry (e.g. a completed rename) to stable storage.
    fn sync_dir(&self, path: &Path) -> io::Result<()>;
}

/// Real filesystem durability: opens the target and calls `sync_all`.
pub(crate) struct RealDurability;

impl Durability for RealDurability {
    fn sync_file(&self, path: &Path) -> io::Result<()> {
        let f = std::fs::File::open(path)?;
        f.sync_all()
    }

    fn sync_dir(&self, path: &Path) -> io::Result<()> {
        // Fsyncing a directory handle is a POSIX construct. On non-Unix
        // platforms opening a directory as a file fails, and the ordering
        // guarantee is provided differently, so this is a no-op there.
        #[cfg(unix)]
        {
            let f = std::fs::File::open(path)?;
            f.sync_all()
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Ok(())
        }
    }
}

/// Monotonic counter feeding the temp-file suffix in [`tmp_path_for`], so two
/// checkpoint writers in the same process never stage into the same `.tmp`
/// sibling.
static CHECKPOINT_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Build a *unique* temp-file sibling path used by [`JobCheckpoint::save`].
///
/// For `/foo/bar/.libviprs-job.json` this returns something like
/// `/foo/bar/.libviprs-job.json.tmp.<pid>.<counter>`. The suffix is
/// disambiguated by the process id and a process-wide monotonic counter so
/// that two jobs sharing one output directory — whether two processes or two
/// threads within one process — never stage into the same temp file.
///
/// A fixed `.tmp` name (the historical scheme) let two concurrent saves
/// interleave `File::create` / `write_all` on the *same* sibling and then each
/// `rename` a half-written, torn JSON into place, so the next `--resume`
/// refused the checkpoint as [`ResumeError::Corrupt`] (issue #126). Because
/// each writer now owns a distinct temp file, its `rename` publishes a
/// complete payload atomically and cannot be clobbered mid-write by another
/// writer's staging.
fn tmp_path_for(final_path: &Path) -> PathBuf {
    let pid = std::process::id();
    let seq = CHECKPOINT_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut s = final_path.as_os_str().to_owned();
    s.push(format!(".tmp.{pid}.{seq}"));
    PathBuf::from(s)
}

/// Atomically write `bytes` to `path` using the same staged-`.tmp` + `rename`
/// discipline as the checkpoint writer ([`JobCheckpoint::save_with`]).
///
/// The payload is written to a unique sibling temp file, flushed, and then
/// renamed over `path`. On POSIX the rename is atomic, so a concurrent reader
/// (or a crash mid-write) never observes a truncated or half-written `path`:
/// it sees either the previous complete contents or the new complete contents.
/// This is what keeps the two on-disk manifest copies from diverging when a
/// run dies partway through emitting them (issue #124).
///
/// Because the rename replaces the directory entry rather than truncating the
/// existing file, it also succeeds when `path` is a read-only regular file, as
/// long as its parent directory is writable, matching real atomic-replace
/// semantics rather than the truncate-in-place behaviour of `fs::write`.
///
/// Parent directories are created as needed. The unique temp name (pid + a
/// process-wide counter) means two writers targeting the same destination
/// never stage into or clobber each other's temp file.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = tmp_path_for(path);
    // Scope the handle so it is closed before the rename — some filesystems
    // refuse to rename over an open handle. Flush the payload to disk before
    // publishing so the renamed file is not left empty on power loss.
    if let Err(e) = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()
    })() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Advisory, exclusive run lock held for the lifetime of a resumable job.
///
/// A resumable run takes this lock against `<dir>/.libviprs-job.lock`
/// ([`LOCK_FILENAME`]) *before* it touches the output directory and holds it
/// until the guard is dropped at the end of the run. While the lock is held,
/// any second job that calls [`RunLock::acquire`] on the same directory is
/// refused with [`ResumeError::Locked`] instead of being allowed to run
/// concurrently.
///
/// This closes the two concurrency hazards from issue #126 that unique temp
/// filenames alone cannot:
///
/// * two live jobs whose periodic checkpoint flushes would otherwise clobber
///   each other's `completed_tiles` (last-writer-wins, even with per-writer
///   unique temps), and
/// * a concurrent Overwrite wipe deleting another live job's in-flight output.
///
/// Because both Resume and Overwrite take this lock before doing any work, at
/// most one of them runs against a given checkpoint root at a time.
///
/// The lock is **advisory**: it only excludes other holders that also go
/// through `RunLock`. It is an OS-level file lock (`flock`-style, via
/// [`std::fs::File::try_lock`]) tied to the open file handle, so the kernel
/// releases it automatically if the process dies without dropping the guard —
/// a crashed job never strands a stale lock that would wedge every later run.
///
/// Releasing the guard also removes the lock file itself, so a finished (or
/// refused) run leaves no bookkeeping behind in the output directory: a clean
/// run and a crash+resume run produce byte-identical trees. The unlink happens
/// while the exclusive lock is still held, and [`RunLock::acquire`]
/// revalidates after locking that the path still names the inode it locked,
/// so the removal cannot let two jobs hold "the" lock at once (see `Drop`).
#[derive(Debug)]
#[must_use = "the run lock is released as soon as the guard is dropped"]
pub struct RunLock {
    // Field order is load-bearing: struct fields drop in declaration order, so
    // `file` (which owns the OS lock) is still open when `Drop` runs its
    // explicit `unlock`, and is closed only afterwards.
    file: std::fs::File,
    path: PathBuf,
}

impl RunLock {
    /// Path of the advisory lock file for checkpoint root `dir`.
    ///
    /// Returns `<dir>/.libviprs-job.lock` without touching the filesystem.
    pub fn lock_path(dir: &Path) -> PathBuf {
        dir.join(LOCK_FILENAME)
    }

    /// Try to take the exclusive run lock on `dir` without blocking.
    ///
    /// * `Ok(RunLock)` — the lock is now held; keep the returned guard alive
    ///   for the whole run and drop it when finished.
    /// * `Err(ResumeError::Locked)` — another live job already holds the lock;
    ///   the caller must refuse to proceed rather than clobber that run.
    /// * `Err(ResumeError::Io)` — the lock file could not be created or locked.
    ///
    /// Acquisition is non-blocking on purpose: a resumable job would rather
    /// fail fast telling the user another run owns the directory than silently
    /// queue behind it for an unbounded time.
    ///
    /// The lock file is opened (creating it if absent) but never truncated. A
    /// releasing holder unlinks the file while its lock is still held (see
    /// `Drop`), so an acquirer may open a doomed inode just before that unlink
    /// and lock it just after the release. A lock on an orphaned inode
    /// excludes nobody: after every successful `try_lock` the path is
    /// therefore revalidated against the locked handle, and on a mismatch the
    /// stale lock is dropped and the acquisition retried against the fresh
    /// file.
    pub fn acquire(dir: &Path) -> Result<RunLock, ResumeError> {
        // Ensure the directory exists so a first-ever run can place its lock
        // before the sink has materialised any output.
        std::fs::create_dir_all(dir)?;
        let path = Self::lock_path(dir);
        // Every retry requires another job to have completed a full
        // acquire-release cycle inside the microsecond window between our
        // `open` and our `try_lock`, so a handful of attempts is ample. The
        // bound turns pathological churn into a clean refusal, not a livelock.
        const MAX_ACQUIRE_ATTEMPTS: usize = 16;
        for _ in 0..MAX_ACQUIRE_ATTEMPTS {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            // `File::try_lock` is stable since Rust 1.89; the crate's declared MSRV
            // is older, so silence the incompatible-MSRV lint at this single call
            // rather than pulling in a third-party file-locking crate for the same
            // primitive. Promoting the whole crate to 1.89 is left to the maintainer.
            #[allow(clippy::incompatible_msrv)]
            let outcome = file.try_lock();
            match outcome {
                Ok(()) => {
                    if lock_path_names_handle(&file, &path)? {
                        return Ok(RunLock { file, path });
                    }
                    // The file we locked was unlinked by a releasing holder:
                    // drop the stale lock (closing `file`) and race for the
                    // fresh file now at `path`.
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(ResumeError::Locked { path });
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(ResumeError::Io(e)),
            }
        }
        Err(ResumeError::Locked { path })
    }

    /// Path of the lock file this guard currently holds.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// True when `path` still names the same file `locked` has open (same device
/// and inode). [`RunLock::acquire`] calls this after a successful `try_lock`
/// to detect that a releasing holder unlinked the lock file between the
/// acquirer's `open` and its lock.
#[cfg(unix)]
fn lock_path_names_handle(locked: &std::fs::File, path: &Path) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let held = locked.metadata()?;
    match std::fs::metadata(path) {
        Ok(on_disk) => Ok(on_disk.dev() == held.dev() && on_disk.ino() == held.ino()),
        // The path is gone entirely: the locked inode is orphaned.
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Non-Unix fallback: report the handle as current. On these platforms the
/// release path cannot unlink an open, locked file in the first place (the
/// `remove_file` in [`RunLock`]'s `Drop` fails and the file persists), so the
/// stale-inode race this check guards against cannot occur.
#[cfg(not(unix))]
fn lock_path_names_handle(_locked: &std::fs::File, _path: &Path) -> io::Result<bool> {
    Ok(true)
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // Unlink the lock file, then release the lock. The ordering is
        // load-bearing: because the unlink happens while the exclusive lock is
        // still held, the path can never name an inode whose lock has already
        // been released. A concurrent `acquire` that opened this inode before
        // the unlink obtains its lock only after the `unlock` below, then
        // notices the path no longer names its inode and retries against the
        // fresh file (see `RunLock::acquire`). Removing the file is what keeps
        // a finished run from leaking bookkeeping into the caller's output
        // directory, and a refused resume from mutating a directory it
        // promised not to touch.
        //
        // Both calls are best-effort. On platforms where unlinking an open
        // file fails (Windows), the file persists, which simply restores the
        // historical leave-it-behind behaviour there; the kernel also drops
        // the lock when the handle closes immediately after this.
        let _ = std::fs::remove_file(&self.path);
        // `File::unlock` is stable since Rust 1.89 (see `acquire`).
        #[allow(clippy::incompatible_msrv)]
        let _ = self.file.unlock();
    }
}

/// True if `coord` appears in `meta.completed_tiles`.
///
/// Linear scan — O(n) in the number of completed tiles. Fine for a single
/// ad-hoc probe, but a resume that tests *every* planned tile against a large
/// checkpoint this way is O(n²). For repeated membership queries build a
/// [`CompletedTileSet`] once (hashing the coordinates up front) and use its
/// O(1) [`CompletedTileSet::contains`].
pub fn is_tile_completed(meta: &JobMetadata, coord: &TileCoord) -> bool {
    meta.completed_tiles.iter().any(|c| c == coord)
}

/// O(1)-membership view over the tiles a checkpoint records as completed.
///
/// [`is_tile_completed`] scans `completed_tiles` linearly, so probing every
/// planned tile against a large checkpoint during a resume is quadratic. A
/// `CompletedTileSet` hashes the coordinates once; each
/// [`contains`](CompletedTileSet::contains) query is then O(1), which keeps a
/// resume's skip decision linear in the number of tiles rather than
/// quadratic. Duplicate coordinates in the checkpoint collapse to a single
/// entry.
///
/// Build one from a loaded [`JobMetadata`] via
/// [`CompletedTileSet::from_metadata`], or collect it directly from an
/// iterator of [`TileCoord`]s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletedTileSet {
    tiles: std::collections::HashSet<TileCoord>,
}

impl CompletedTileSet {
    /// Build the set from a checkpoint's `completed_tiles`, hashing each
    /// coordinate once so later membership tests are O(1).
    pub fn from_metadata(meta: &JobMetadata) -> Self {
        meta.completed_tiles.iter().copied().collect()
    }

    /// O(1) membership test: `true` iff `coord` was recorded as completed.
    pub fn contains(&self, coord: &TileCoord) -> bool {
        self.tiles.contains(coord)
    }

    /// Number of distinct completed coordinates.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// `true` when no completed tiles are recorded.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

impl FromIterator<TileCoord> for CompletedTileSet {
    fn from_iter<I: IntoIterator<Item = TileCoord>>(iter: I) -> Self {
        Self {
            tiles: iter.into_iter().collect(),
        }
    }
}

/// The output *content contract* of a run: the non-geometry choices that
/// change the bytes a resume would produce, and therefore must invalidate a
/// checkpoint when they change.
///
/// [`compute_plan_hash`] folds the padding colour, blank-tile handling,
/// dedupe layout, and source content into the plan hash alongside the plan
/// geometry so that resuming a checkpoint whose contract differs is rejected
/// with [`crate::engine::EngineError::PlanHashMismatch`] instead of silently
/// mixing two incompatible outputs on disk. The [`format`](Self::format) is
/// *not* hashed — it is compared separately by `verify_checkpoint_contract`
/// so a transparent sink wrapper that cannot report the format does not break
/// a legitimate resume.
///
/// Build one with [`PlanContract::from_engine`] so that the value derived at
/// checkpoint-write time and the value derived at the resume gate agree for
/// the same run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanContract<'a> {
    /// Tile encoding the sink commits to, when it has one. `None` for sinks
    /// that do not pin a single on-disk format. Recorded in the checkpoint
    /// (see [`JobMetadata::content_format`]) and compared by
    /// `verify_checkpoint_contract`, but deliberately excluded from
    /// [`compute_plan_hash`].
    pub format: Option<TileFormat>,
    /// Background RGB used to pad edge tiles.
    pub background_rgb: [u8; 3],
    /// Blank-tile handling strategy.
    pub blank_strategy: BlankTileStrategy,
    /// Content-addressed dedupe strategy.
    pub dedupe: DedupeStrategy,
    /// Optional content digest identifying the source raster. Folded in only
    /// when present, so callers that do not compute one keep the historical
    /// geometry+format behaviour.
    pub source_digest: Option<&'a str>,
}

impl<'a> PlanContract<'a> {
    /// Derive the contract from the engine config and sink that drive a run.
    ///
    /// Both the checkpoint writer and the resume gate call this against the
    /// same `config`/`sink`, guaranteeing they hash identical bytes for a
    /// legitimate resume and divergent bytes when any contract input changed.
    pub fn from_engine(config: &'a EngineConfig, sink: &dyn TileSink) -> Self {
        Self {
            format: sink.content_format(),
            background_rgb: config.background_rgb,
            blank_strategy: config.blank_tile_strategy,
            dedupe: config.dedupe_strategy.unwrap_or_default(),
            source_digest: config.source_content_hash.as_deref(),
        }
    }
}

/// Compute the plan hash that identifies a run's on-disk output on disk.
///
/// Hashes the plan's load-bearing geometry *and* the [`PlanContract`] — the
/// non-geometry content contract — in a fixed canonical byte layout so that
/// the hash is stable across:
///
/// * process restarts,
/// * struct-field reordering in future revisions of [`PyramidPlan`] (as long
///   as the serialisation code here is updated deliberately),
/// * serde representation choices elsewhere in the crate.
///
/// The exact byte layout is: a constant domain-separator prefix, then each
/// field as a fixed-width little-endian integer (or a single tag byte for
/// enums), in the order declared below. The result is the lowercase hex
/// Blake3 digest of those bytes.
///
/// Two runs whose geometry is identical but whose padding colour, blank-tile
/// strategy, dedupe layout, or source digest differ produce different hashes,
/// so the resume gate rejects a changed content contract rather than resuming
/// into a mixed output.
///
/// The tile **format** is deliberately *not* hashed here. The format is a
/// property of the live sink, and a legitimate resume may wrap that sink in a
/// transparent decorator (recording / retry / tee) that does not forward
/// [`crate::sink::TileSink::content_format`]. Baking the format into this
/// hash would then flip a bit purely because a wrapper was added or removed,
/// spuriously rejecting a valid checkpoint. A genuine format change is caught
/// separately by `verify_checkpoint_contract`, which compares the format
/// recorded in the checkpoint against the live sink's.
pub fn compute_plan_hash(plan: &PyramidPlan, contract: &PlanContract<'_>) -> String {
    // Domain separator — ties this hash to a specific canonicalisation so
    // the same bytes cannot accidentally match some other hash contract.
    // Bumped to v3 when the tile format was lifted out of the hash into an
    // explicit, wildcard-tolerant comparison (see the module docs); v1/v2
    // checkpoints no longer match and are treated as a plan mismatch.
    const DOMAIN: &[u8] = b"libviprs/plan/v3";

    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN);

    // Plan-level scalars.
    hasher.update(&plan.image_width.to_le_bytes());
    hasher.update(&plan.image_height.to_le_bytes());
    hasher.update(&plan.tile_size.to_le_bytes());
    hasher.update(&plan.overlap.to_le_bytes());
    hasher.update(&[layout_tag(plan.layout)]);
    hasher.update(&plan.canvas_width.to_le_bytes());
    hasher.update(&plan.canvas_height.to_le_bytes());
    hasher.update(&[u8::from(plan.centre)]);
    hasher.update(&plan.centre_offset_x.to_le_bytes());
    hasher.update(&plan.centre_offset_y.to_le_bytes());

    // Level count, then each level's full shape. Including every level's
    // dimensions means that any change to the pyramid geometry — including
    // ones we might otherwise consider derived — invalidates the hash.
    hasher.update(&(plan.levels.len() as u64).to_le_bytes());
    for lvl in &plan.levels {
        hasher.update(&lvl.level.to_le_bytes());
        hasher.update(&lvl.width.to_le_bytes());
        hasher.update(&lvl.height.to_le_bytes());
        hasher.update(&lvl.cols.to_le_bytes());
        hasher.update(&lvl.rows.to_le_bytes());
    }

    // Content contract. A single tag byte (plus any payload) per field, so
    // that changing the padding colour, blank-tile handling, dedupe layout,
    // or source content changes the digest. The tile format is intentionally
    // excluded — it is compared separately in `verify_checkpoint_contract` so
    // that a transparent sink wrapper that cannot report the format does not
    // invalidate an otherwise-matching checkpoint.
    hasher.update(&contract.background_rgb);
    let (blank_tag, blank_delta) = blank_strategy_tag(contract.blank_strategy);
    hasher.update(&[blank_tag, blank_delta]);
    hasher.update(&[dedupe_tag(contract.dedupe)]);
    match contract.source_digest {
        Some(d) => {
            hasher.update(&[1u8]);
            hasher.update(&(d.len() as u64).to_le_bytes());
            hasher.update(d.as_bytes());
        }
        None => {
            hasher.update(&[0u8]);
        }
    }

    hasher.finalize().to_hex().to_string()
}

/// Decide whether a loaded checkpoint is compatible with the current run.
///
/// Resume must tolerate a caller wrapping their sink in a transparent
/// decorator (recording, retry, tee, crash-injecting, …). Such wrappers
/// legitimately forward `write_tile` and `checkpoint_root` but often do not
/// forward [`crate::sink::TileSink::content_format`], so the format they
/// report is `None` even though the underlying run pins a concrete encoding.
/// Because the format is folded into [`compute_plan_hash`], recomputing the
/// expected hash straight from the live sink would flip the format bits and
/// reject an otherwise valid checkpoint — the resume would fail with
/// [`crate::engine::EngineError::PlanHashMismatch`] purely because a wrapper
/// was added or removed between the writing run and the resuming run.
///
/// To stay robust the geometry + content contract is re-hashed using the
/// format that was **recorded when the checkpoint was written**
/// ([`JobMetadata::content_format`]) rather than whatever the live sink can
/// report now, so a legitimate resume matches regardless of wrapping. A
/// *genuine* output-format change is still caught: when the recorded format
/// and the live sink's format are both concrete and disagree, the checkpoint
/// is rejected so two incompatible encodings are never mixed on disk.
///
/// Returns `Ok(())` when the checkpoint is compatible, or `Err(got)` carrying
/// the freshly computed hash to surface in the resulting mismatch error.
pub(crate) fn verify_checkpoint_contract(
    meta: &JobMetadata,
    plan: &PyramidPlan,
    config: &EngineConfig,
    sink: &dyn TileSink,
) -> Result<(), String> {
    let live = PlanContract::from_engine(config, sink);

    // Geometry + non-format content contract. Stable regardless of how the
    // sink is wrapped, because every field here is derived from the plan and
    // the engine config, not from the live sink.
    let expected = compute_plan_hash(plan, &live);
    if meta.plan_hash != expected {
        return Err(expected);
    }

    // Genuine output-format change: only when both the checkpoint and the
    // live sink pin a concrete — and different — encoding. Either side being
    // `None` means "format unknown", which we treat as compatible rather than
    // rejecting a valid resume behind a transparent wrapper.
    if let (Some(recorded), Some(current)) = (meta.content_format, live.format)
        && recorded != current
    {
        return Err(format!(
            "{expected} (tile format changed from {recorded:?} to {current:?})"
        ));
    }

    Ok(())
}

/// Tag byte (and tolerance payload) for a [`BlankTileStrategy`].
fn blank_strategy_tag(strategy: BlankTileStrategy) -> (u8, u8) {
    match strategy {
        BlankTileStrategy::Emit => (1, 0),
        BlankTileStrategy::Placeholder => (2, 0),
        BlankTileStrategy::PlaceholderWithTolerance { max_channel_delta } => (3, max_channel_delta),
    }
}

/// Tag byte for a [`DedupeStrategy`], including its checksum algorithm.
fn dedupe_tag(dedupe: DedupeStrategy) -> u8 {
    match dedupe {
        DedupeStrategy::None => 0,
        DedupeStrategy::Blanks => 1,
        DedupeStrategy::All {
            algo: ChecksumAlgo::Blake3,
        } => 2,
        DedupeStrategy::All {
            algo: ChecksumAlgo::Sha256,
        } => 3,
    }
}

/// Single-byte discriminator for a [`Layout`] value.
///
/// Kept in one place so that adding a new layout forces an explicit decision
/// about what byte to assign it — rather than letting Rust's auto-assigned
/// enum discriminants silently influence the hash.
fn layout_tag(layout: Layout) -> u8 {
    match layout {
        Layout::DeepZoom => 1,
        Layout::Xyz => 2,
        Layout::Google => 3,
        Layout::Zoomify => 4,
        Layout::Iiif => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PyramidPlanner;
    use crate::sink::{SinkError, Tile};

    fn sample_plan() -> PyramidPlan {
        PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    /// Minimal sink whose only interesting behaviour is the tile format it
    /// reports (or refuses to report). Lets the resume-gate tests drive
    /// [`verify_checkpoint_contract`] without a real filesystem sink.
    struct FormatSink(Option<TileFormat>);

    impl TileSink for FormatSink {
        fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
            Ok(())
        }
        fn finish(&self) -> Result<(), SinkError> {
            Ok(())
        }
        fn content_format(&self) -> Option<TileFormat> {
            self.0
        }
    }

    /// Build a checkpoint as the engine would for a run driven by `sink`:
    /// a geometry+contract hash plus the recorded write-time format.
    fn checkpoint_for(
        plan: &PyramidPlan,
        config: &EngineConfig,
        sink: &dyn TileSink,
    ) -> JobMetadata {
        let contract = PlanContract::from_engine(config, sink);
        let mut meta = JobMetadata::new(
            compute_plan_hash(plan, &contract),
            "1970-01-01T00:00:00Z".into(),
        );
        meta.content_format = contract.format;
        meta
    }

    /// A baseline content contract for the plan-hash tests. Individual tests
    /// mutate one field to assert the hash reacts to that change.
    fn sample_contract() -> PlanContract<'static> {
        PlanContract {
            format: Some(TileFormat::Png),
            background_rgb: [255, 255, 255],
            blank_strategy: BlankTileStrategy::Emit,
            dedupe: DedupeStrategy::None,
            source_digest: None,
        }
    }

    fn sample_meta(hash: &str) -> JobMetadata {
        JobMetadata {
            schema_version: SCHEMA_VERSION.to_string(),
            plan_hash: hash.to_string(),
            completed_tiles: vec![TileCoord::new(0, 0, 0), TileCoord::new(1, 1, 0)],
            levels_completed: vec![0],
            started_at: "1970-01-01T00:00:00Z".into(),
            last_checkpoint_at: "1970-01-01T00:00:00Z".into(),
            content_format: None,
        }
    }

    #[test]
    fn default_mode_is_overwrite() {
        assert_eq!(ResumeMode::default(), ResumeMode::Overwrite);
    }

    #[test]
    fn checkpoint_path_is_well_known_filename() {
        let p = JobCheckpoint::checkpoint_path(Path::new("/tmp/out"));
        assert_eq!(p, PathBuf::from("/tmp/out/.libviprs-job.json"));
    }

    /// Issue #127: coordinates written to the append-only segment log are
    /// merged back into `completed_tiles` by `load`, and combine with any
    /// inline header coordinates without duplication.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_merges_segment_log_with_inline_header() {
        let dir = tempfile::tempdir().unwrap();
        // Header carries one inline coordinate (a legacy / hand-written
        // checkpoint), the segment log carries two more.
        let mut meta = sample_meta("deadbeef");
        meta.completed_tiles = vec![TileCoord::new(9, 9, 9)];
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        append_segments(
            dir.path(),
            &[TileCoord::new(0, 0, 0), TileCoord::new(1, 2, 3)],
            &RealDurability,
        )
        .unwrap();

        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        // Segment coordinates come first, then inline-only coordinates.
        assert_eq!(
            loaded.completed_tiles,
            vec![
                TileCoord::new(0, 0, 0),
                TileCoord::new(1, 2, 3),
                TileCoord::new(9, 9, 9),
            ]
        );
        // Physical frame count matches the coordinates written to the log.
        assert_eq!(
            count_segment_frames(&JobCheckpoint::segments_path(dir.path())).unwrap(),
            2
        );
    }

    /// A coordinate present in *both* the header inline set and the segment
    /// log is reported once (dedup keeps `completed_tiles.len()` honest even
    /// after a crash re-appended an already-inline coordinate).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_dedupes_overlap_between_header_and_segments() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = sample_meta("deadbeef");
        meta.completed_tiles = vec![TileCoord::new(0, 0, 0)];
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        append_segments(dir.path(), &[TileCoord::new(0, 0, 0)], &RealDurability).unwrap();

        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.completed_tiles, vec![TileCoord::new(0, 0, 0)]);
    }

    /// A torn trailing segment frame (a crash mid-append) is dropped on read;
    /// the whole frames before it are still recovered.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_ignores_torn_trailing_segment_frame() {
        let dir = tempfile::tempdir().unwrap();
        let meta = sample_meta("deadbeef");
        // sample_meta populates two inline coordinates; clear them so this
        // test isolates the segment log.
        let mut header = meta;
        header.completed_tiles.clear();
        JobCheckpoint::save(dir.path(), &header).unwrap();
        append_segments(
            dir.path(),
            &[TileCoord::new(2, 0, 0), TileCoord::new(2, 1, 0)],
            &RealDurability,
        )
        .unwrap();
        // Simulate a partial append: 5 stray bytes shorter than one record.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(JobCheckpoint::segments_path(dir.path()))
                .unwrap();
            f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }

        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        assert_eq!(
            loaded.completed_tiles,
            vec![TileCoord::new(2, 0, 0), TileCoord::new(2, 1, 0)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let plan = sample_plan();
        let hash = compute_plan_hash(&plan, &sample_contract());
        let meta = sample_meta(&hash);
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, meta);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_returns_none_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(JobCheckpoint::load(dir.path()).unwrap().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_rejects_corrupt_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = JobCheckpoint::checkpoint_path(dir.path());
        std::fs::write(&path, b"{not valid json").unwrap();
        match JobCheckpoint::load(dir.path()) {
            Err(ResumeError::Corrupt { path: p, .. }) => assert_eq!(p, path),
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn load_rejects_schema_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = JobCheckpoint::checkpoint_path(dir.path());
        std::fs::write(
            &path,
            br#"{
                "schema_version": "999",
                "plan_hash": "deadbeef",
                "completed_tiles": [],
                "levels_completed": [],
                "started_at": "",
                "last_checkpoint_at": ""
            }"#,
        )
        .unwrap();
        match JobCheckpoint::load(dir.path()) {
            Err(ResumeError::SchemaMismatch { expected, found }) => {
                assert_eq!(expected, SCHEMA_VERSION);
                assert_eq!(found, "999");
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_is_atomic_no_tmp_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let plan = sample_plan();
        let meta = sample_meta(&compute_plan_hash(&plan, &sample_contract()));
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        // No staging sibling (`.libviprs-job.json.tmp.*`) may linger after a
        // successful save — the unique temp file must have been renamed away.
        let leftover: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(
            leftover.is_empty(),
            "tmp files should be renamed, not linger: {leftover:?}"
        );
        assert!(JobCheckpoint::checkpoint_path(dir.path()).exists());
    }

    // Issue #126 (acceptance criterion 2): temp files must be unique per
    // process so two concurrent saves on one output directory never stage into
    // the same `.tmp` sibling and torn-rename over each other. The historical
    // fixed name (`.libviprs-job.json.tmp`) made `tmp_path_for` a pure function
    // of the final path, so successive calls collided; this test fails on that
    // code and passes once the name is disambiguated by pid + counter.
    #[test]
    fn tmp_paths_are_unique_per_call() {
        let final_path = JobCheckpoint::checkpoint_path(Path::new("/tmp/out"));
        let a = tmp_path_for(&final_path);
        let b = tmp_path_for(&final_path);
        assert_ne!(
            a, b,
            "concurrent checkpoint saves must not share a tmp filename"
        );
        // Both temps stage next to the real checkpoint and carry the pid so a
        // second *process* on the same directory also picks a distinct name.
        let pid = std::process::id().to_string();
        for tmp in [&a, &b] {
            let s = tmp.to_string_lossy();
            assert!(s.contains(".libviprs-job.json.tmp."), "unexpected tmp: {s}");
            assert!(s.contains(&pid), "tmp name must carry the pid: {s}");
        }
    }

    // Issue #126 (acceptance criterion 1): concurrent jobs must not corrupt
    // each other's checkpoint file. Many threads hammer `save` on one shared
    // directory while the main thread continuously reloads it. With a fixed
    // temp name the interleaved `create`/`write_all`/`rename` published torn
    // JSON that `load` rejected as `Corrupt`; with per-writer unique temps each
    // rename publishes a complete payload, so a concurrent read always sees a
    // valid checkpoint.
    #[test]
    #[cfg_attr(miri, ignore)] // threads + filesystem access blocked by Miri isolation
    fn concurrent_saves_never_publish_a_torn_checkpoint() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let dir_path: Arc<PathBuf> = Arc::new(dir.path().to_path_buf());
        let plan = sample_plan();
        let hash = compute_plan_hash(&plan, &sample_contract());

        // Seed one valid checkpoint so the reader always has something to load.
        JobCheckpoint::save(&dir_path, &sample_meta(&hash)).unwrap();

        const WRITERS: usize = 8;
        const SAVES_PER_WRITER: usize = 40;

        let mut handles = Vec::new();
        for w in 0..WRITERS {
            let dir_path = Arc::clone(&dir_path);
            let hash = hash.clone();
            handles.push(thread::spawn(move || {
                for i in 0..SAVES_PER_WRITER {
                    let mut meta = sample_meta(&hash);
                    // Vary the payload length so a torn rename would surface as
                    // an obviously invalid parse rather than an aligned one.
                    for r in 0..((w * SAVES_PER_WRITER + i) as u32 % 50) {
                        meta.completed_tiles.push(TileCoord::new(0, 0, r));
                    }
                    JobCheckpoint::save(&dir_path, &meta).unwrap();
                }
            }));
        }

        // While the writers race, keep reloading: a torn checkpoint would
        // surface as `ResumeError::Corrupt`.
        for _ in 0..400 {
            match JobCheckpoint::load(&dir_path) {
                Ok(Some(_)) => {}
                Ok(None) => panic!("checkpoint vanished mid-run"),
                Err(e) => panic!("concurrent save corrupted the checkpoint: {e:?}"),
            }
        }

        for h in handles {
            h.join().unwrap();
        }

        // Final state is still a clean, loadable checkpoint with no stray temps.
        assert!(JobCheckpoint::load(&dir_path).unwrap().is_some());
        let leftover: Vec<_> = std::fs::read_dir(&*dir_path)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(leftover.is_empty(), "stray temp files remain: {leftover:?}");
    }

    #[test]
    fn lock_path_is_well_known_filename() {
        let p = RunLock::lock_path(Path::new("/tmp/out"));
        assert_eq!(p, PathBuf::from("/tmp/out/.libviprs-job.lock"));
    }

    // Issue #126 (advisory run-lock remainder): a second job on an output
    // directory already owned by a live run must be refused, not allowed to run
    // concurrently and clobber the first job's checkpoint / output. While the
    // guard is held, `acquire` returns `Locked`; once it drops, the directory
    // is free again. The pre-fix code has no `RunLock`, so this test cannot even
    // compile against it (RED); it passes once the lock lands (GREEN).
    #[test]
    #[cfg_attr(miri, ignore)] // file locking unsupported under Miri isolation
    fn run_lock_refuses_a_second_holder() {
        let dir = tempfile::tempdir().unwrap();

        let held = RunLock::acquire(dir.path()).expect("first acquire succeeds");
        assert_eq!(held.path(), RunLock::lock_path(dir.path()));

        match RunLock::acquire(dir.path()) {
            Err(ResumeError::Locked { path }) => {
                assert_eq!(path, RunLock::lock_path(dir.path()));
            }
            other => panic!("a second job must be refused while the lock is held, got {other:?}"),
        }

        // Releasing the guard frees the directory for the next run.
        drop(held);
        let reacquired = RunLock::acquire(dir.path()).expect("acquire after release succeeds");
        drop(reacquired);
    }

    // A finished (or refused) run must leave the checkpoint root exactly as it
    // found it, so releasing the guard removes the lock file. The unlink runs
    // while the exclusive lock is still held and `acquire` revalidates the
    // path after locking, which keeps the removal safe against a concurrent
    // acquirer; this test pins the on-disk outcome. Unix-only: on platforms
    // where unlinking an open file fails, the file is deliberately left
    // behind.
    #[cfg(unix)]
    #[test]
    #[cfg_attr(miri, ignore)] // file locking unsupported under Miri isolation
    fn run_lock_release_removes_the_lock_file() {
        let dir = tempfile::tempdir().unwrap();

        let held = RunLock::acquire(dir.path()).expect("first acquire succeeds");
        assert!(
            RunLock::lock_path(dir.path()).exists(),
            "the lock file exists while the guard is held"
        );

        drop(held);
        assert!(
            !RunLock::lock_path(dir.path()).exists(),
            "the lock file must be removed when the guard drops"
        );

        // The directory is immediately acquirable again after the removal.
        let reacquired = RunLock::acquire(dir.path()).expect("acquire after release succeeds");
        drop(reacquired);
    }

    // Issue #126 (advisory run-lock remainder): the whole point of the lock is
    // mutual exclusion of live runs on one checkpoint root. Many threads race to
    // acquire the same directory's lock; each holder briefly bumps a shared
    // "currently inside the critical section" counter and asserts it never sees
    // more than one holder at a time. Without the lock (or if it failed to
    // exclude) this counter would exceed 1 and the run's flushes/wipe could
    // clobber each other. Every contender either holds exclusively or is cleanly
    // refused with `Locked`.
    #[test]
    #[cfg_attr(miri, ignore)] // threads + file locking unsupported under Miri isolation
    fn run_lock_serialises_concurrent_jobs() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let dir_path: Arc<PathBuf> = Arc::new(dir.path().to_path_buf());
        // Number of threads holding the lock simultaneously (must never > 1).
        let live = Arc::new(AtomicUsize::new(0));
        // Count of successful exclusive acquisitions across all threads.
        let acquisitions = Arc::new(AtomicUsize::new(0));

        const THREADS: usize = 8;
        const ATTEMPTS: usize = 50;

        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let dir_path = Arc::clone(&dir_path);
            let live = Arc::clone(&live);
            let acquisitions = Arc::clone(&acquisitions);
            handles.push(thread::spawn(move || {
                for _ in 0..ATTEMPTS {
                    match RunLock::acquire(&dir_path) {
                        Ok(guard) => {
                            // Inside the critical section: at most one holder.
                            let inside = live.fetch_add(1, Ordering::SeqCst) + 1;
                            assert_eq!(
                                inside, 1,
                                "the run lock let two jobs into the critical section at once",
                            );
                            acquisitions.fetch_add(1, Ordering::SeqCst);
                            // Hold briefly so contenders genuinely collide.
                            thread::yield_now();
                            live.fetch_sub(1, Ordering::SeqCst);
                            drop(guard);
                        }
                        // A contended attempt is cleanly refused — never a torn
                        // co-tenancy. That is exactly the desired behaviour.
                        Err(ResumeError::Locked { .. }) => {}
                        Err(e) => panic!("unexpected lock error: {e:?}"),
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(live.load(Ordering::SeqCst), 0, "a holder leaked the lock");
        assert!(
            acquisitions.load(Ordering::SeqCst) >= 1,
            "at least one thread must have taken the lock",
        );

        // The directory is free once every guard has dropped.
        let final_guard = RunLock::acquire(&dir_path).expect("lock is free after all jobs finish");
        drop(final_guard);
    }

    #[test]
    fn plan_hash_is_deterministic() {
        let plan = sample_plan();
        assert_eq!(
            compute_plan_hash(&plan, &sample_contract()),
            compute_plan_hash(&plan, &sample_contract())
        );
    }

    #[test]
    fn plan_hash_changes_with_tile_size() {
        let a = PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(128, 128, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert_ne!(
            compute_plan_hash(&a, &sample_contract()),
            compute_plan_hash(&b, &sample_contract())
        );
    }

    #[test]
    fn plan_hash_changes_with_layout() {
        let a = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(256, 256, 64, 0, Layout::Xyz)
            .unwrap()
            .plan();
        assert_ne!(
            compute_plan_hash(&a, &sample_contract()),
            compute_plan_hash(&b, &sample_contract())
        );
    }

    #[test]
    fn plan_hash_changes_with_overlap() {
        let a = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(256, 256, 64, 1, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert_ne!(
            compute_plan_hash(&a, &sample_contract()),
            compute_plan_hash(&b, &sample_contract())
        );
    }

    #[test]
    fn plan_hash_ignores_tile_format() {
        // The format is compared separately (see the resume-gate tests
        // below), not folded into the geometry+contract hash, so two contracts
        // that differ only in format hash identically.
        let plan = sample_plan();
        let png = PlanContract {
            format: Some(TileFormat::Png),
            ..sample_contract()
        };
        let jpeg = PlanContract {
            format: Some(TileFormat::Jpeg { quality: 80 }),
            ..sample_contract()
        };
        assert_eq!(
            compute_plan_hash(&plan, &png),
            compute_plan_hash(&plan, &jpeg),
            "tile format must not affect the plan hash"
        );
    }

    #[test]
    fn resume_rejects_a_changed_tile_format() {
        // A checkpoint written by a PNG sink must not resume against a JPEG
        // sink — that would mix two encodings in the same output tree.
        let plan = sample_plan();
        let config = EngineConfig::default();
        let png = FormatSink(Some(TileFormat::Png));
        let meta = checkpoint_for(&plan, &config, &png);

        assert!(
            verify_checkpoint_contract(&meta, &plan, &config, &png).is_ok(),
            "same-format resume must be accepted"
        );

        let jpeg = FormatSink(Some(TileFormat::Jpeg { quality: 80 }));
        assert!(
            verify_checkpoint_contract(&meta, &plan, &config, &jpeg).is_err(),
            "changing the tile format must invalidate the checkpoint"
        );
    }

    #[test]
    fn plan_hash_divergence_surfaces_as_engine_error_not_a_resume_variant() {
        // A checkpoint whose recorded `plan_hash` does not match the current
        // plan is the "real condition" the removed `ResumeError::PlanHashMismatch`
        // variant was documented to carry. It is NOT reported through
        // `ResumeError`: the resume gate returns the freshly-computed hash and
        // the engine layer maps it to `EngineError::PlanHashMismatch`. This
        // test pins that boundary so nobody re-introduces a redundant
        // checkpoint-error variant for a condition the engine already owns.
        let plan = sample_plan();
        let config = EngineConfig::default();
        let png = FormatSink(Some(TileFormat::Png));

        // A metadata blob whose plan_hash is garbage relative to the live plan.
        let meta = sample_meta("0000000000000000000000000000000000000000000000000000000000000000");

        let got = verify_checkpoint_contract(&meta, &plan, &config, &png)
            .expect_err("a divergent plan_hash must be rejected by the resume gate");
        // The gate hands back the freshly computed hash of the current plan,
        // which the engine embeds as the `expected` field of
        // `EngineError::PlanHashMismatch { expected, actual }` (the `actual`
        // field carries the divergent hash recorded in the checkpoint).
        assert_eq!(
            got,
            compute_plan_hash(&plan, &PlanContract::from_engine(&config, &png)),
            "the resume gate must return the freshly computed hash for the engine error",
        );
        assert_ne!(got, meta.plan_hash, "the two hashes must actually differ");
    }

    #[test]
    fn resume_rejects_a_changed_jpeg_quality() {
        let plan = sample_plan();
        let config = EngineConfig::default();
        let q80 = FormatSink(Some(TileFormat::Jpeg { quality: 80 }));
        let meta = checkpoint_for(&plan, &config, &q80);

        let q40 = FormatSink(Some(TileFormat::Jpeg { quality: 40 }));
        assert!(
            verify_checkpoint_contract(&meta, &plan, &config, &q40).is_err(),
            "changing the JPEG quality must invalidate the checkpoint"
        );
    }

    #[test]
    fn resume_tolerates_a_sink_that_does_not_report_its_format() {
        // Regression: a transparent wrapper sink returns `None` from
        // `content_format()`. Resume must still accept a checkpoint written by
        // the concrete underlying sink instead of failing with a spurious
        // PlanHashMismatch.
        let plan = sample_plan();
        let config = EngineConfig::default();

        // Written by a concrete PNG sink, resumed behind a format-blind wrapper.
        let png = FormatSink(Some(TileFormat::Png));
        let meta = checkpoint_for(&plan, &config, &png);
        let wrapper = FormatSink(None);
        assert!(
            verify_checkpoint_contract(&meta, &plan, &config, &wrapper).is_ok(),
            "a wrapper that does not report its format must not break resume"
        );

        // Symmetric case: written behind a format-blind wrapper, resumed by
        // the concrete sink.
        let meta_blind = checkpoint_for(&plan, &config, &wrapper);
        assert!(
            verify_checkpoint_contract(&meta_blind, &plan, &config, &png).is_ok(),
            "a checkpoint written without a format must resume against a concrete sink"
        );
    }

    #[test]
    fn plan_hash_changes_with_background() {
        let plan = sample_plan();
        let white = sample_contract();
        let black = PlanContract {
            background_rgb: [0, 0, 0],
            ..sample_contract()
        };
        assert_ne!(
            compute_plan_hash(&plan, &white),
            compute_plan_hash(&plan, &black)
        );
    }

    #[test]
    fn plan_hash_changes_with_blank_strategy() {
        let plan = sample_plan();
        let emit = sample_contract();
        let placeholder = PlanContract {
            blank_strategy: BlankTileStrategy::Placeholder,
            ..sample_contract()
        };
        assert_ne!(
            compute_plan_hash(&plan, &emit),
            compute_plan_hash(&plan, &placeholder)
        );
    }

    #[test]
    fn plan_hash_changes_with_dedupe_strategy() {
        let plan = sample_plan();
        let none = sample_contract();
        let blanks = PlanContract {
            dedupe: DedupeStrategy::Blanks,
            ..sample_contract()
        };
        assert_ne!(
            compute_plan_hash(&plan, &none),
            compute_plan_hash(&plan, &blanks)
        );
    }

    #[test]
    fn plan_hash_changes_with_source_digest() {
        // Two runs over same-dimension sources: identical geometry and
        // contract but different source content must not resume one another.
        let plan = sample_plan();
        let src_a = PlanContract {
            source_digest: Some("aaaaaaaa"),
            ..sample_contract()
        };
        let src_b = PlanContract {
            source_digest: Some("bbbbbbbb"),
            ..sample_contract()
        };
        assert_ne!(
            compute_plan_hash(&plan, &src_a),
            compute_plan_hash(&plan, &src_b),
            "a different source digest must invalidate the checkpoint"
        );
        // Absent digest differs from any present digest (opt-in behaviour).
        assert_ne!(
            compute_plan_hash(&plan, &sample_contract()),
            compute_plan_hash(&plan, &src_a)
        );
    }

    #[test]
    fn plan_hash_is_lowercase_hex() {
        let hash = compute_plan_hash(&sample_plan(), &sample_contract());
        assert_eq!(hash.len(), 64, "Blake3 produces a 32-byte / 64-hex digest");
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hash should be lowercase hex: {hash}"
        );
    }

    #[test]
    fn is_tile_completed_reports_membership() {
        let meta = sample_meta("deadbeef");
        assert!(is_tile_completed(&meta, &TileCoord::new(0, 0, 0)));
        assert!(is_tile_completed(&meta, &TileCoord::new(1, 1, 0)));
        assert!(!is_tile_completed(&meta, &TileCoord::new(2, 0, 0)));
    }

    // Issue #127 (acceptance criterion 2): membership lookup against a
    // checkpoint must be better than O(n). `is_tile_completed` is a linear
    // scan; a resume that probes every planned tile against it is O(n^2).
    // `CompletedTileSet` hashes the coordinates once so each `contains`
    // query is O(1). This test fails to compile on the pre-fix code because
    // the type does not exist yet.
    #[test]
    fn completed_tile_set_gives_constant_time_membership() {
        let meta = sample_meta("deadbeef");
        let set = CompletedTileSet::from_metadata(&meta);

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.contains(&TileCoord::new(0, 0, 0)));
        assert!(set.contains(&TileCoord::new(1, 1, 0)));
        assert!(!set.contains(&TileCoord::new(2, 0, 0)));

        // Duplicate coordinates in the checkpoint collapse to one entry.
        let mut dupey = meta.clone();
        dupey.completed_tiles.push(TileCoord::new(0, 0, 0));
        assert_eq!(
            CompletedTileSet::from_metadata(&dupey).len(),
            2,
            "duplicate coords must not inflate the membership set"
        );

        // Built directly from an iterator of coords.
        let from_iter: CompletedTileSet = [TileCoord::new(3, 2, 1), TileCoord::new(3, 2, 1)]
            .into_iter()
            .collect();
        assert_eq!(from_iter.len(), 1);
        assert!(from_iter.contains(&TileCoord::new(3, 2, 1)));

        let empty = CompletedTileSet::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn capped_prealloc_clamps_hostile_hint() {
        use super::tile_coord_vec_serde::{MAX_SEQ_PREALLOC, capped_prealloc};

        // A missing hint reserves nothing.
        assert_eq!(capped_prealloc(None), 0);
        // A modest, honest hint passes through unchanged.
        assert_eq!(capped_prealloc(Some(10)), 10);
        assert_eq!(capped_prealloc(Some(MAX_SEQ_PREALLOC)), MAX_SEQ_PREALLOC);
        // Anything above the cap — including a hostile `usize::MAX` — is bounded.
        assert_eq!(
            capped_prealloc(Some(MAX_SEQ_PREALLOC + 1)),
            MAX_SEQ_PREALLOC
        );
        assert_eq!(capped_prealloc(Some(usize::MAX)), MAX_SEQ_PREALLOC);
    }

    /// A [`SeqAccess`] that advertises a gigantic sequence length but yields no
    /// elements — the shape a corrupt or hostile length-prefixed checkpoint
    /// would take. Driving the real coord deserializer with this must not turn
    /// the reported hint into a speculative allocation.
    #[test]
    fn coord_deserializer_ignores_hostile_size_hint() {
        use super::tile_coord_vec_serde::{MAX_SEQ_PREALLOC, deserialize};
        use serde::Deserializer;
        use serde::de::value::Error as DeError;
        use serde::de::{DeserializeSeed, SeqAccess, Visitor};

        struct HostileSeq;
        impl<'de> SeqAccess<'de> for HostileSeq {
            type Error = DeError;
            fn next_element_seed<T>(&mut self, _seed: T) -> Result<Option<T::Value>, Self::Error>
            where
                T: DeserializeSeed<'de>,
            {
                Ok(None)
            }
            fn size_hint(&self) -> Option<usize> {
                // Without a cap this would drive `Vec::with_capacity(usize::MAX)`,
                // which panics on capacity overflow.
                Some(usize::MAX)
            }
        }

        struct HostileDeserializer;
        impl<'de> Deserializer<'de> for HostileDeserializer {
            type Error = DeError;
            fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                visitor.visit_seq(HostileSeq)
            }
            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str
                string bytes byte_buf option unit unit_struct newtype_struct
                tuple tuple_struct map struct enum identifier ignored_any
            }
            fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                unreachable!("coord deserializer only requests a sequence")
            }
        }

        let coords = deserialize(HostileDeserializer).expect("empty hostile sequence deserializes");
        assert!(coords.is_empty());
        assert!(
            coords.capacity() <= MAX_SEQ_PREALLOC,
            "reservation ({}) must stay within the cap ({MAX_SEQ_PREALLOC})",
            coords.capacity(),
        );
    }
}
