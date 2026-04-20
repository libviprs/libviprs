//! Resumable pyramid generation — building blocks (Phase 3).
//!
//! This module provides the on-disk checkpoint format, plan-hash computation,
//! and helper types used by `generate_pyramid_resumable` (the end-to-end
//! entry point lives in [`crate::engine`] and is wired up separately).
//!
//! # Checkpoint format
//!
//! Each output directory contains a single file, `.libviprs-job.json`, whose
//! contents deserialise to [`JobMetadata`]. The file is written atomically via
//! a `.tmp` sibling + rename so that a crash mid-write cannot produce a torn
//! or partially-updated checkpoint.
//!
//! # Plan hashing
//!
//! A run may resume only if the current [`PyramidPlan`] matches the plan that
//! was originally used to produce the checkpoint. [`compute_plan_hash`]
//! serialises the plan's load-bearing fields into a canonical byte layout and
//! hashes them with Blake3. Any change to tile size, overlap, layout, level
//! count, or per-level dimensions changes the hash — so a mismatched plan is
//! detected before a single tile is written.
//!
//! # Intended use
//!
//! ```ignore
//! use libviprs::resume::{JobCheckpoint, JobMetadata, compute_plan_hash};
//!
//! let hash = compute_plan_hash(&plan);
//! let meta = JobMetadata {
//!     schema_version: "1".to_string(),
//!     plan_hash: hash,
//!     completed_tiles: Vec::new(),
//!     levels_completed: Vec::new(),
//!     started_at: now_rfc3339(),
//!     last_checkpoint_at: now_rfc3339(),
//! };
//! JobCheckpoint::save(output_dir, &meta)?;
//! ```

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::planner::{Layout, PyramidPlan, TileCoord};

// Re-export the engine entry point from `libviprs::resume::*` so tests can
// grab it from the same module that owns the checkpoint types.
pub use crate::engine::generate_pyramid_resumable;

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

/// On-disk checkpoint describing the state of a pyramid generation job.
///
/// Produced and consumed by [`JobCheckpoint::save`] / [`JobCheckpoint::load`].
/// The struct is intentionally simple and flat so that it serialises cleanly
/// as JSON — a debugger or shell user can inspect it with `cat` / `jq`.
///
/// `schema_version` is stored as a [`String`] so we can read back old
/// checkpoints, compare them against [`SCHEMA_VERSION`], and return a
/// structured error if they disagree.
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
    /// Uses the [`tile_coord_vec_serde`] adapter because [`TileCoord`] in
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
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
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
/// [`std::fs`] calls; `PlanHashMismatch` and `SchemaMismatch` surface
/// semantic incompatibilities that make it unsafe to resume.
#[derive(Debug, Error)]
pub enum ResumeError {
    /// The checkpoint's `plan_hash` disagrees with the current plan's hash.
    /// Resuming would produce incoherent output, so the engine refuses.
    #[error("plan hash mismatch: checkpoint records {expected}, current plan hashes to {actual}")]
    PlanHashMismatch {
        /// Hash stored in the checkpoint file.
        expected: String,
        /// Hash freshly computed from the current plan.
        actual: String,
    },
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
pub struct JobCheckpoint;

impl JobCheckpoint {
    /// Absolute path of the checkpoint file for the given output directory.
    ///
    /// Returns `<dir>/.libviprs-job.json` without checking whether the file
    /// actually exists.
    pub fn checkpoint_path(dir: &Path) -> PathBuf {
        dir.join(CHECKPOINT_FILENAME)
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
        let meta: JobMetadata = serde_json::from_slice(&bytes)
            .map_err(|source| ResumeError::Corrupt { path, source })?;
        if meta.schema_version != SCHEMA_VERSION {
            return Err(ResumeError::SchemaMismatch {
                expected: SCHEMA_VERSION,
                found: meta.schema_version,
            });
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
        // Make sure the target directory exists; callers typically create it,
        // but checkpointing should not fail just because the sink has not yet
        // materialised a sub-tree.
        std::fs::create_dir_all(dir)?;

        let final_path = Self::checkpoint_path(dir);
        let tmp_path = tmp_path_for(&final_path);

        let bytes = serde_json::to_vec_pretty(meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Scope the file handle so it's closed before the rename — some
        // filesystems refuse to rename over an open file handle.
        {
            let mut f = std::fs::File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }

        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }
}

/// Build the temp-file sibling path used by [`JobCheckpoint::save`].
///
/// For `/foo/bar/.libviprs-job.json` this returns
/// `/foo/bar/.libviprs-job.json.tmp`. Extracted so the naming scheme is kept
/// in one place and can be adjusted independently of the save logic.
fn tmp_path_for(final_path: &Path) -> PathBuf {
    let mut s = final_path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// True if `coord` appears in `meta.completed_tiles`.
///
/// Linear scan. Callers that need repeated lookups against a large checkpoint
/// should build their own `HashSet<TileCoord>` once from
/// `meta.completed_tiles`; for typical pyramid sizes this straightforward
/// implementation is fast enough.
pub fn is_tile_completed(meta: &JobMetadata, coord: &TileCoord) -> bool {
    meta.completed_tiles.iter().any(|c| c == coord)
}

/// Compute the plan hash that identifies a [`PyramidPlan`] on disk.
///
/// Hashes the plan's load-bearing fields — not any run-time state — in a
/// fixed canonical byte layout so that the hash is stable across:
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
pub fn compute_plan_hash(plan: &PyramidPlan) -> String {
    // Domain separator — ties this hash to a specific canonicalisation so
    // the same bytes cannot accidentally match some other hash contract.
    const DOMAIN: &[u8] = b"libviprs/plan/v1";

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

    hasher.finalize().to_hex().to_string()
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PyramidPlanner;

    fn sample_plan() -> PyramidPlan {
        PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn sample_meta(hash: &str) -> JobMetadata {
        JobMetadata {
            schema_version: SCHEMA_VERSION.to_string(),
            plan_hash: hash.to_string(),
            completed_tiles: vec![TileCoord::new(0, 0, 0), TileCoord::new(1, 1, 0)],
            levels_completed: vec![0],
            started_at: "1970-01-01T00:00:00Z".into(),
            last_checkpoint_at: "1970-01-01T00:00:00Z".into(),
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

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let plan = sample_plan();
        let hash = compute_plan_hash(&plan);
        let meta = sample_meta(&hash);
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        let loaded = JobCheckpoint::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, meta);
    }

    #[test]
    fn load_returns_none_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(JobCheckpoint::load(dir.path()).unwrap().is_none());
    }

    #[test]
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
    fn save_is_atomic_no_tmp_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let plan = sample_plan();
        let meta = sample_meta(&compute_plan_hash(&plan));
        JobCheckpoint::save(dir.path(), &meta).unwrap();
        let tmp = tmp_path_for(&JobCheckpoint::checkpoint_path(dir.path()));
        assert!(!tmp.exists(), "tmp file should be renamed, not linger");
        assert!(JobCheckpoint::checkpoint_path(dir.path()).exists());
    }

    #[test]
    fn plan_hash_is_deterministic() {
        let plan = sample_plan();
        assert_eq!(compute_plan_hash(&plan), compute_plan_hash(&plan));
    }

    #[test]
    fn plan_hash_changes_with_tile_size() {
        let a = PyramidPlanner::new(128, 128, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(128, 128, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert_ne!(compute_plan_hash(&a), compute_plan_hash(&b));
    }

    #[test]
    fn plan_hash_changes_with_layout() {
        let a = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(256, 256, 64, 0, Layout::Xyz)
            .unwrap()
            .plan();
        assert_ne!(compute_plan_hash(&a), compute_plan_hash(&b));
    }

    #[test]
    fn plan_hash_changes_with_overlap() {
        let a = PyramidPlanner::new(256, 256, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let b = PyramidPlanner::new(256, 256, 64, 1, Layout::DeepZoom)
            .unwrap()
            .plan();
        assert_ne!(compute_plan_hash(&a), compute_plan_hash(&b));
    }

    #[test]
    fn plan_hash_is_lowercase_hex() {
        let hash = compute_plan_hash(&sample_plan());
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
}
