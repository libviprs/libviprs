//! Content-addressed deduplication for tile output (Phase 3 hardening).
//!
//! This module implements the in-memory machinery used by [`crate::sink::FsSink`]
//! to collapse tiles whose byte contents are identical onto a single physical
//! file under `_shared/` in the sink's base directory. It is intentionally
//! decoupled from the sink itself so it can be unit-tested independently and
//! reused by alternative sinks.
//!
//! # Storage contract
//!
//! For each distinct content hash, at most one physical file is written at
//! `_shared/<shared_key>.<ext>`. [`DedupeIndex::record`] tells the sink
//! whether it is looking at the *first* occurrence of some content
//! ([`DedupeDecision::WriteNew`]) or a later duplicate
//! ([`DedupeDecision::Reference`]); the sink materialises the layout with a
//! *promote-on-second-hit* scheme (see [`crate::sink::FsSink`]):
//!
//! 1. The first tile with a given content hash is written in full at its own
//!    planned path. Nothing is placed under `_shared/` yet — content seen only
//!    once never bloats `_shared/`.
//! 2. When a *second* tile with identical bytes arrives, the first tile's file
//!    is promoted into `_shared/<shared_key>.<ext>` and the first tile path is
//!    re-pointed at it with a **hardlink** (guaranteeing at least one tile
//!    resolves to the shared inode).
//! 3. That second tile — and every later duplicate — is written as a 1-byte
//!    placeholder at its own path, and the `tile_path -> shared_path`
//!    relationship is recorded in `manifest.json`'s `blank_references` map.
//!    Readers consult the manifest to resolve the pointer.
//!
//! # Deterministic placement (issue #275)
//!
//! [`DedupeIndex::record`] returns `WriteNew` for whichever occurrence of a
//! content hash it sees *first*, and under `tile_concurrency > 0` that is
//! producer-completion order over an MPSC queue — non-deterministic. If the
//! sink kept the arrival-first occurrence as the full-payload holder, the byte
//! layout and the `blank_references` map would depend on thread scheduling,
//! breaking the [`crate::sink::TileSink`] commutative-placement contract.
//!
//! To make placement a pure function of content + coordinates, `FsSink` runs
//! the promote-on-second-hit scheme above *during* the run for its
//! at-least-one-hardlink guarantee, then, at `finish()` (after all writers have
//! joined), reassigns the single full-payload holder of each duplicated content
//! to its **coordinate-minimal** occurrence — sorted by `(level, row, col)`,
//! the canonical order used by [`crate::mapreduce_hot_cache`]. The layout shape
//! is otherwise identical; only *which* occurrence keeps the full bytes changes,
//! and it changes to a value fixed by the input alone. See
//! [`crate::sink::FsSink::canonicalize_dedupe_layout`].
//!
//! The standalone [`materialize_reference`] helper (symlink → hardlink →
//! placeholder fallback, reported via [`LinkResult`]) is provided for
//! *alternative* sinks that prefer link-based references over the manifest
//! scheme above; `FsSink` does not use it.
//!
//! # Shared-key contract
//!
//! Shared-key filenames are deterministic functions of tile content:
//! `blank_<hex-hash>`. This is asserted by
//! `libviprs-tests/tests/phase3_dedupe_blanks.rs::determinism_of_dedupe_hashes`.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::manifest::ChecksumAlgo;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Selects how tiles are content-addressed prior to being written to disk.
///
/// CLI users pick a strategy via
/// [`--dedupe-blanks`](https://libviprs.org/cli/#flag-dedupe-blanks) (blanks-only)
/// or [`--dedupe-all`](https://libviprs.org/cli/#flag-dedupe-all) (every tile).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DedupeStrategy {
    /// No deduplication — every tile is written to its own file. Default.
    #[default]
    None,
    /// Only blank (uniform-colour) tiles are deduplicated. Non-blank tiles
    /// are written individually.
    ///
    /// CLI: [`--dedupe-blanks`](https://libviprs.org/cli/#flag-dedupe-blanks).
    Blanks,
    /// Every tile is content-addressed; identical tiles share a single
    /// physical file regardless of whether they're blank.
    ///
    /// CLI: [`--dedupe-all`](https://libviprs.org/cli/#flag-dedupe-all).
    All {
        /// Hash algorithm used to derive the shared-key filename.
        algo: ChecksumAlgo,
    },
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

/// Outcome of a [`DedupeIndex::record`] call.
///
/// Drives the on-disk layout produced by the CLI's
/// [`--dedupe-blanks`](https://libviprs.org/cli/#flag-dedupe-blanks) flag.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DedupeDecision {
    /// This content hash was not previously seen. The caller must write
    /// `bytes` to `shared_path` and then materialize a reference at the
    /// tile's planned path.
    WriteNew {
        /// Deterministic filename stem (without directory or extension) under
        /// `_shared/`.
        shared_key: String,
        /// Absolute-relative path (relative to the sink base dir) of the
        /// shared file: `_shared/<shared_key>.<ext>`.
        shared_path: PathBuf,
    },
    /// This content hash was already seen; the caller must only materialize
    /// a reference at the tile's planned path pointing at `shared_path`.
    Reference {
        /// Deterministic filename stem of the existing shared file.
        shared_key: String,
        /// Path of the existing shared file (relative to sink base dir).
        shared_path: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Compact, `Ord`-able tag for the hash algorithm component of the `seen`
/// map's key. `ChecksumAlgo` itself doesn't derive `Ord` and we can't touch
/// `manifest.rs` on this branch, so we project it onto a single byte.
type AlgoTag = u8;

const ALGO_TAG_BLAKE3: AlgoTag = 1;
const ALGO_TAG_SHA256: AlgoTag = 2;

fn algo_tag(a: ChecksumAlgo) -> AlgoTag {
    match a {
        ChecksumAlgo::Blake3 => ALGO_TAG_BLAKE3,
        ChecksumAlgo::Sha256 => ALGO_TAG_SHA256,
    }
}

/// Inner shared state for [`DedupeIndex`], wrapped in a single mutex so the
/// `seen` / `refs` maps always observe a coherent invariant (Mara Bos B5:
/// readers that held two independent locks could see `refs` populated before
/// `seen` caught up, briefly violating "refs is a subset of seen's producible
/// keys"). Coalescing into one lock takes the same number of lock operations
/// as the previous two-mutex layout.
///
/// The `seen` map is keyed by `(algo_tag, raw_digest_bytes)` rather than hex
/// strings: Blake3 and SHA-256 both produce 32-byte digests, so a single
/// `[u8; 32]` representation avoids the ~2 GB of per-tile `String`
/// allocations BurntSushi flagged on a 10M-tile run. The `algo_tag`
/// component of the key prevents collisions between two runs where different
/// algorithms happen to produce the same byte pattern (they won't in
/// practice, but bundling the algo is free and makes the invariant
/// explicit). We use a local [`AlgoTag`] (a `u8`) rather than
/// `ChecksumAlgo` directly because the latter doesn't derive `Ord` and this
/// branch is restricted to editing `dedupe.rs`.
///
/// `BTreeMap` is used instead of `HashMap` (dtolnay #5) so
/// `references()`-derived output — which is serialized into
/// `manifest.json`'s `blank_references` field — iterates in a deterministic
/// order regardless of hash-state randomization.
#[derive(Debug, Default)]
struct DedupeState {
    /// `(algo_tag, digest) -> SeenEntry` for first-write / reference
    /// decisions.
    seen: BTreeMap<(AlgoTag, [u8; 32]), SeenEntry>,
    /// `tile_path -> shared_key` for manifest emission.
    refs: BTreeMap<String, String>,
}

/// Value stored per distinct content hash in [`DedupeState::seen`].
///
/// The shared-key stem is a pure function of the content hash, but the
/// physical file's *extension* is inherited from whichever tile was recorded
/// first (that tile's bytes are the ones promoted into `_shared/`). A later
/// same-bytes tile with a different extension must therefore reference the
/// path that was actually written, not one re-derived from its own extension
/// (issue #96). We remember that full path here so `Reference` decisions can
/// return it verbatim.
#[derive(Debug, Clone)]
struct SeenEntry {
    /// Deterministic `blank_<hash>` stem of the shared file.
    shared_key: String,
    /// Full relative path of the shared file as first written, e.g.
    /// `_shared/blank_<hash>.png`. Recorded from the first tile that carried
    /// this content so later duplicates reference the file that actually
    /// exists on disk.
    shared_path: PathBuf,
}

/// In-memory mapping from content-hash to shared-key. One instance per
/// generation run; not persisted across restarts (resume-mode sinks rebuild
/// the index by walking `_shared/`).
///
/// Constructed under the hood when the CLI runs with
/// [`--dedupe-blanks`](https://libviprs.org/cli/#flag-dedupe-blanks).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
#[non_exhaustive]
pub struct DedupeIndex {
    strategy: DedupeStrategy,
    /// Single mutex guarding both the hash->shared-key map and the
    /// path->shared-key map, so callers never observe one updated without
    /// the other.
    state: Mutex<DedupeState>,
}

impl DedupeIndex {
    /// Create a fresh index for the given strategy.
    pub fn new(strategy: DedupeStrategy) -> Self {
        Self {
            strategy,
            state: Mutex::new(DedupeState::default()),
        }
    }

    /// Returns the strategy this index was built for.
    pub fn strategy(&self) -> DedupeStrategy {
        self.strategy
    }

    /// Algorithm used to produce the raw digest for `hash_content_raw`.
    /// [`DedupeStrategy::Blanks`] and [`DedupeStrategy::None`] always hash
    /// with Blake3 (fast, unkeyed); [`DedupeStrategy::All`] honours the
    /// caller-chosen algorithm.
    fn effective_algo(&self) -> ChecksumAlgo {
        match self.strategy {
            DedupeStrategy::None | DedupeStrategy::Blanks => ChecksumAlgo::Blake3,
            DedupeStrategy::All { algo } => algo,
        }
    }

    /// Compute the content hash using the algorithm dictated by `strategy`,
    /// returning the raw 32-byte digest alongside the algorithm that
    /// produced it. Keeping raw bytes avoids the per-tile `String`
    /// allocation that `ChecksumAlgo::hash` would incur.
    fn hash_content_raw(&self, bytes: &[u8]) -> (ChecksumAlgo, [u8; 32]) {
        let algo = self.effective_algo();
        let digest = match algo {
            ChecksumAlgo::Blake3 => {
                let h = blake3::hash(bytes);
                *h.as_bytes()
            }
            ChecksumAlgo::Sha256 => {
                use sha2::Digest;
                let mut hasher = sha2::Sha256::new();
                hasher.update(bytes);
                let out = hasher.finalize();
                let mut buf = [0u8; 32];
                buf.copy_from_slice(&out);
                buf
            }
        };
        (algo, digest)
    }

    /// Record a tile whose byte contents are `bytes` at the planned path
    /// `path` (relative to the sink base dir, e.g. `"5/0_0.png"`).
    ///
    /// Returns a [`DedupeDecision`] describing what the caller must do.
    /// The `path`-to-`shared_key` mapping is also recorded internally so it
    /// can be emitted into the manifest via [`Self::references`].
    pub fn record(&self, path: &str, bytes: &[u8]) -> DedupeDecision {
        // Infer an extension from the tile path so the shared file sits on
        // disk under the same format as the tiles that reference it.
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin");

        let (algo, digest) = self.hash_content_raw(bytes);
        let hash_hex = crate::hex::hex_lower(&digest);
        let shared_key = format!("blank_{hash_hex}");
        let shared_path = Self::shared_path(&shared_key, ext);

        // `DedupeStrategy::None` is a true passthrough: every tile is written
        // to its own file and no content is ever collapsed. We must therefore
        // neither consult nor populate `seen`/`refs`, so two calls with
        // identical bytes can never yield a `Reference` (issue #99). Returning
        // `WriteNew` for every call tells the caller to write the tile verbatim
        // at its planned path. `FsSink` guards this externally and never calls
        // `record` under `None`, but `record` is public API and other consumers
        // must not be silently deduplicated.
        if self.strategy == DedupeStrategy::None {
            return DedupeDecision::WriteNew {
                shared_key,
                shared_path,
            };
        }

        // Single lock: update `refs` and `seen` atomically so readers never
        // observe one populated without the other (Mara B5). Poison is
        // recovered (crate::poison policy): the maps are structurally valid
        // between operations, so a panicked prior holder must not cascade.
        let mut state = crate::poison::recover(&self.state);
        state.refs.insert(path.to_string(), shared_key.clone());

        match state.seen.entry((algo_tag(algo), digest)) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(SeenEntry {
                    shared_key: shared_key.clone(),
                    shared_path: shared_path.clone(),
                });
                DedupeDecision::WriteNew {
                    shared_key,
                    shared_path,
                }
            }
            // A first occurrence already recorded the physical path that its
            // bytes will occupy under `_shared/`. Return the *stored* shared
            // key and path so the reference resolves to the file that actually
            // gets written, rather than one re-derived from this tile's
            // (possibly different) extension (issue #96).
            std::collections::btree_map::Entry::Occupied(e) => {
                let stored = e.get();
                DedupeDecision::Reference {
                    shared_key: stored.shared_key.clone(),
                    shared_path: stored.shared_path.clone(),
                }
            }
        }
    }

    /// Drop a recorded reference (e.g. because the sink successfully created
    /// a symlink / hardlink and no manifest pointer is required). No-op if
    /// the path was never recorded.
    pub fn forget_reference(&self, path: &str) {
        let mut state = crate::poison::recover(&self.state);
        state.refs.remove(path);
    }

    /// Compute the on-disk relative path for a shared blob.
    pub fn shared_path(shared_key: &str, ext: &str) -> PathBuf {
        let mut p = PathBuf::from("_shared");
        p.push(format!("{shared_key}.{ext}"));
        p
    }

    /// Consume-style snapshot of the current tile-path -> shared-key map for
    /// `manifest.json::blank_references` emission.
    ///
    /// Returns a `BTreeMap` so callers that serialize the result get
    /// deterministic key order (dtolnay #5).
    pub fn references(&self) -> BTreeMap<String, String> {
        crate::poison::recover(&self.state).refs.clone()
    }

    /// Total number of distinct content hashes seen so far. Useful for
    /// diagnostics and tests.
    pub fn distinct_count(&self) -> usize {
        crate::poison::recover(&self.state).seen.len()
    }
}

// ---------------------------------------------------------------------------
// Link materialization
// ---------------------------------------------------------------------------

/// Outcome of [`materialize_reference`]: which strategy was ultimately used
/// to point `tile_path` at `shared_path`.
///
/// Reflects the link/placeholder fallback chain triggered by
/// [`--dedupe-blanks`](https://libviprs.org/cli/#flag-dedupe-blanks).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinkResult {
    /// A symbolic link was created at `tile_path` pointing at `shared_path`.
    Symlink,
    /// A hardlink was created; both inodes are identical.
    Hardlink,
    /// Neither symlink nor hardlink was possible on this filesystem. A
    /// 1-byte placeholder was written at `tile_path` and the caller must
    /// record the `tile_path -> shared_key` mapping in `manifest.json`.
    ManifestOnly,
}

/// 1-byte placeholder written at tile paths that could not be linked. Chosen
/// to match [`crate::sink::BLANK_TILE_MARKER`] semantically (a single NUL
/// byte) so existing tooling that recognises blank placeholders continues
/// to work. The exact value is not load-bearing for tests — they detect
/// placeholders by checking `len() == 1`.
const PLACEHOLDER_MARKER: [u8; 1] = [0u8];

#[cfg(unix)]
fn try_symlink(target: &Path, link: &Path) -> io::Result<()> {
    // `target` is interpreted *relative to the directory containing `link`*
    // by readlink()/open() on unix. We therefore need a target path that is
    // valid from `link`'s parent; compute it from the absolute forms.
    let target_for_link = symlink_target_for(target, link);
    std::os::unix::fs::symlink(&target_for_link, link)
}

#[cfg(windows)]
fn try_symlink(target: &Path, link: &Path) -> io::Result<()> {
    // On Windows we need to know whether the target is a file or a
    // directory. Shared blobs are always files.
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(not(any(unix, windows)))]
fn try_symlink(_target: &Path, _link: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "symlinks are not supported on this platform",
    ))
}

#[cfg(any(unix, windows))]
fn try_hardlink(target: &Path, link: &Path) -> io::Result<()> {
    std::fs::hard_link(target, link)
}

#[cfg(not(any(unix, windows)))]
fn try_hardlink(target: &Path, link: &Path) -> io::Result<()> {
    std::fs::hard_link(target, link)
}

/// Compute a filesystem path suitable for passing as the *target* of a
/// symlink whose link file will live at `link`. On unix, symlink targets
/// are resolved relative to the link's parent directory, so we try to
/// produce a path relative to that directory.
///
/// Falls back to the absolute `target` path if we can't compute a relative
/// form (e.g. different prefixes on windows).
#[cfg(unix)]
fn symlink_target_for(target: &Path, link: &Path) -> PathBuf {
    // Both paths are typically already absolute (sink base dir is absolute).
    // If not, just use the target as-is — the caller is in control of cwd.
    let (Some(link_parent), true) = (link.parent(), target.is_absolute()) else {
        return target.to_path_buf();
    };
    if !link_parent.is_absolute() {
        return target.to_path_buf();
    }
    match pathdiff(target, link_parent) {
        Some(rel) => rel,
        None => target.to_path_buf(),
    }
}

/// Pure-function relative-path computation. Returns `None` if the paths
/// live on different roots (no relative form exists).
#[cfg(unix)]
fn pathdiff(to: &Path, from: &Path) -> Option<PathBuf> {
    use std::path::Component;
    let to_components: Vec<Component<'_>> = to.components().collect();
    let from_components: Vec<Component<'_>> = from.components().collect();

    // Find the common prefix.
    let mut i = 0;
    while i < to_components.len() && i < from_components.len() {
        if to_components[i] != from_components[i] {
            break;
        }
        i += 1;
    }

    // If nothing common, there is no meaningful relative path.
    if i == 0 {
        return None;
    }

    let mut rel = PathBuf::new();
    for _ in i..from_components.len() {
        rel.push("..");
    }
    for comp in &to_components[i..] {
        rel.push(comp.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        rel.push(".");
    }
    Some(rel)
}

/// Materialize a reference at `tile_path` that resolves to `shared_path`.
///
/// Per-platform fallback order (BurntSushi honourable mention):
///
/// * **Unix**: symlink → hardlink → 1-byte placeholder + manifest entry.
///   Symlinks work unprivileged on unix; hardlinks can fail across mount
///   points, so they come second.
/// * **Windows**: hardlink → symlink → 1-byte placeholder + manifest entry.
///   Symlink creation requires admin / Developer Mode on Windows, while
///   hardlinks work unprivileged. Since `tile_path` and `shared_path` both
///   live under the sink base dir, hardlinks almost always succeed there.
///
/// The parent directory of `tile_path` must already exist; the caller is
/// responsible for `std::fs::create_dir_all`. Any pre-existing file at
/// `tile_path` is removed first (so this is safe to call in resume mode).
///
/// Returns the [`LinkResult`] describing which strategy succeeded, or an
/// [`io::Error`] if *every* strategy — including the placeholder write —
/// failed. A failure is never reported as success: if the returned value is
/// `Ok`, a resolvable file exists at `tile_path`.
pub fn materialize_reference(tile_path: &Path, shared_path: &Path) -> io::Result<LinkResult> {
    // Absolutize the shared target before linking. A relative `shared_path`
    // (e.g. `_shared/blank_<hash>.png`) passed verbatim as a symlink target
    // is resolved relative to `tile_path`'s parent, which points at a file
    // that does not exist there — a dangling link. Anchoring to an absolute
    // path makes the reference resolve regardless of where `tile_path` lives.
    let shared_target = absolutize(shared_path);

    // Remove any leftover file (or dangling symlink) at the tile path so link
    // creation doesn't fail with EEXIST. Removing unconditionally — and
    // tolerating only `NotFound` — closes the check-then-remove TOCTOU window
    // while still surfacing genuine errors (e.g. EACCES on the parent dir).
    match std::fs::remove_file(tile_path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    #[cfg(windows)]
    {
        // Windows: hardlink first (works without admin), then symlink.
        if try_hardlink(&shared_target, tile_path).is_ok() {
            return Ok(LinkResult::Hardlink);
        }
        if try_symlink(&shared_target, tile_path).is_ok() {
            return Ok(LinkResult::Symlink);
        }
    }

    #[cfg(not(windows))]
    {
        // Unix (and other unix-like targets): symlink first, then hardlink.
        if try_symlink(&shared_target, tile_path).is_ok() {
            return Ok(LinkResult::Symlink);
        }
        if try_hardlink(&shared_target, tile_path).is_ok() {
            return Ok(LinkResult::Hardlink);
        }
    }

    // Final fallback: placeholder file. If even this write fails there is no
    // resolvable file at `tile_path`, so the error must be surfaced rather
    // than masqueraded as a successful `ManifestOnly`.
    std::fs::write(tile_path, PLACEHOLDER_MARKER)?;
    Ok(LinkResult::ManifestOnly)
}

/// Resolve `p` to an absolute path. Already-absolute paths are returned
/// unchanged; relative ones are joined onto the current working directory.
/// Falls back to the input verbatim only if the cwd cannot be read (in which
/// case the caller is presumed to control cwd).
fn absolutize(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(p),
            Err(_) => p.to_path_buf(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_strategy_is_none() {
        assert_eq!(DedupeStrategy::default(), DedupeStrategy::None);
    }

    #[test]
    fn shared_path_is_under_shared_dir() {
        let p = DedupeIndex::shared_path("blank_deadbeef", "png");
        assert_eq!(p, PathBuf::from("_shared").join("blank_deadbeef.png"));
    }

    #[test]
    fn none_strategy_never_deduplicates() {
        // Regression (issue #99): `DedupeStrategy::None` must be a true
        // passthrough. Two tiles with byte-identical contents must each be
        // reported as `WriteNew` — the second must NOT collapse onto the first
        // as a `Reference` — and no internal dedupe state may be populated.
        let idx = DedupeIndex::new(DedupeStrategy::None);
        let bytes = b"identical tile bytes";

        let d1 = idx.record("0/0_0.png", bytes);
        let d2 = idx.record("0/0_1.png", bytes);

        assert!(
            matches!(d1, DedupeDecision::WriteNew { .. }),
            "None must report the first tile as WriteNew; got {d1:?}"
        );
        assert!(
            matches!(d2, DedupeDecision::WriteNew { .. }),
            "None must never emit a dedupe Reference; got {d2:?}"
        );
        assert!(
            idx.references().is_empty(),
            "None must not record any manifest references"
        );
        assert_eq!(
            idx.distinct_count(),
            0,
            "None must not populate the seen index"
        );
    }

    /// Reproducer for #117: a poisoned dedupe-state lock must not cascade a
    /// second panic into every later `record` / `references` / `distinct_count`
    /// call. We poison the state mutex under `catch_unwind`, then keep using the
    /// index. Before the fix (`.lock().expect(...)`) each of these re-panicked
    /// on the poison (RED); after it they recover the guard and preserve the
    /// already-seen entries (GREEN).
    #[test]
    fn poisoned_dedupe_state_recovers_without_cascade() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        let bytes = b"content that gets deduplicated";
        let d1 = idx.record("0/0_0.png", bytes);
        assert!(matches!(d1, DedupeDecision::WriteNew { .. }));

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = idx.state.lock().unwrap();
            panic!("worker panic while holding the dedupe state lock");
        }));
        assert!(poisoned.is_err());
        assert!(idx.state.is_poisoned());

        // Recovered reads see the pre-poison entry, and a duplicate still
        // resolves to a Reference — none of these may panic.
        assert_eq!(idx.distinct_count(), 1);
        assert_eq!(idx.references().len(), 1);
        let d2 = idx.record("0/0_1.png", bytes);
        assert!(
            matches!(d2, DedupeDecision::Reference { .. }),
            "duplicate must still dedupe after poison recovery; got {d2:?}"
        );
    }

    #[test]
    fn record_first_returns_write_new_second_returns_reference() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        let bytes = b"hello world";
        let d1 = idx.record("0/0_0.png", bytes);
        let d2 = idx.record("0/0_1.png", bytes);
        match d1 {
            DedupeDecision::WriteNew { shared_key, .. } => {
                assert!(shared_key.starts_with("blank_"));
            }
            _ => panic!("first record must be WriteNew"),
        }
        match d2 {
            DedupeDecision::Reference { shared_key, .. } => {
                assert!(shared_key.starts_with("blank_"));
            }
            _ => panic!("second record must be Reference"),
        }
    }

    #[test]
    fn distinct_contents_produce_distinct_shared_keys() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        let d1 = idx.record("a.png", b"aaaa");
        let d2 = idx.record("b.png", b"bbbb");
        let k1 = match d1 {
            DedupeDecision::WriteNew { shared_key, .. } => shared_key,
            _ => unreachable!(),
        };
        let k2 = match d2 {
            DedupeDecision::WriteNew { shared_key, .. } => shared_key,
            _ => unreachable!(),
        };
        assert_ne!(k1, k2);
    }

    #[test]
    fn references_map_captures_all_recorded_paths() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        idx.record("a.png", b"xxxx");
        idx.record("b.png", b"xxxx");
        idx.record("c.png", b"yyyy");
        let refs = idx.references();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs["a.png"], refs["b.png"]);
        assert_ne!(refs["a.png"], refs["c.png"]);
    }

    #[test]
    fn forget_reference_removes_manifest_entry() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        idx.record("a.png", b"xxxx");
        idx.forget_reference("a.png");
        assert!(idx.references().is_empty());
    }

    #[test]
    fn all_strategy_honours_custom_algo() {
        let idx_blake = DedupeIndex::new(DedupeStrategy::All {
            algo: ChecksumAlgo::Blake3,
        });
        let idx_sha = DedupeIndex::new(DedupeStrategy::All {
            algo: ChecksumAlgo::Sha256,
        });
        let key_blake = match idx_blake.record("x.png", b"content") {
            DedupeDecision::WriteNew { shared_key, .. } => shared_key,
            _ => unreachable!(),
        };
        let key_sha = match idx_sha.record("x.png", b"content") {
            DedupeDecision::WriteNew { shared_key, .. } => shared_key,
            _ => unreachable!(),
        };
        assert_ne!(
            key_blake, key_sha,
            "different algos must produce different shared keys"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn materialize_reference_creates_a_readable_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let shared_dir = tmp.path().join("_shared");
        std::fs::create_dir_all(&shared_dir).unwrap();
        let shared = shared_dir.join("blank_abc.png");
        std::fs::write(&shared, b"SHAREDCONTENT").unwrap();

        let tile = tmp.path().join("0").join("0_0.png");
        std::fs::create_dir_all(tile.parent().unwrap()).unwrap();

        let result = materialize_reference(&tile, &shared).expect("materialization must succeed");
        // Depending on the host filesystem, any of the three is legal.
        assert!(matches!(
            result,
            LinkResult::Symlink | LinkResult::Hardlink | LinkResult::ManifestOnly
        ));

        // The tile path must exist after materialization.
        assert!(tile.exists() || tile.is_symlink());

        if matches!(result, LinkResult::Symlink | LinkResult::Hardlink) {
            // Following links must yield the shared content.
            let read_back = std::fs::read(&tile).unwrap();
            assert_eq!(read_back, b"SHAREDCONTENT");
        } else {
            // Placeholder is exactly one byte.
            let md = std::fs::metadata(&tile).unwrap();
            assert_eq!(md.len(), 1);
        }
    }

    #[test]
    fn cross_extension_reference_points_at_written_shared_file() {
        // Regression: two same-bytes tiles with *different* extensions must
        // not collide onto a shared file that was never written. The first
        // tile (`.png`) is the one whose bytes actually get promoted into
        // `_shared/`, so a later `.jpg` tile with identical bytes must
        // reference `_shared/<key>.png` — the file that exists on disk — not
        // `_shared/<key>.jpg`, which nothing ever created.
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        let bytes = b"identical tile payload";

        let first = idx.record("0/0_0.png", bytes);
        let written_shared_path = match first {
            DedupeDecision::WriteNew { shared_path, .. } => shared_path,
            _ => panic!("first record must be WriteNew"),
        };

        let second = idx.record("0/0_1.jpg", bytes);
        match second {
            DedupeDecision::Reference { shared_path, .. } => {
                assert_eq!(
                    shared_path, written_shared_path,
                    "reference must point at the actually-written shared file, \
                     not a path derived from the second tile's extension"
                );
            }
            _ => panic!("second record must be Reference"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn materialize_reference_with_relative_target_does_not_dangle() {
        // Regression (issue #98): a *relative* `shared_path` was passed to the
        // symlink call verbatim. On unix a symlink target is resolved relative
        // to the link's own directory, so `_shared/blank_x.png` resolved to
        // `<tile-dir>/_shared/blank_x.png` — a path that does not exist — and
        // the reference dangled. Absolutizing the target fixes it.
        let tmp = tempfile::tempdir().unwrap();
        // Canonicalize the root so the cwd reported by `current_dir` (which the
        // OS may resolve through symlinks, e.g. /var -> /private/var on macOS)
        // matches the paths we build here; otherwise the relative-symlink math
        // compares mismatched prefixes.
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let shared_dir = root.join("_shared");
        std::fs::create_dir_all(&shared_dir).unwrap();
        std::fs::write(shared_dir.join("blank_rel.png"), b"SHAREDPAYLOAD").unwrap();

        let tile = root.join("7").join("1_2.png");
        std::fs::create_dir_all(tile.parent().unwrap()).unwrap();

        // Deliberately relative shared path, mirroring what `record` returns.
        let relative_shared = Path::new("_shared").join("blank_rel.png");

        // Run with cwd at the temp root so the relative target *could* be
        // resolved from there but never from the tile's own directory.
        let cwd_guard = CwdGuard::change_to(&root);
        let result =
            materialize_reference(&tile, &relative_shared).expect("materialization must succeed");
        drop(cwd_guard);

        // Whichever strategy was chosen, the tile must resolve to the shared
        // bytes — i.e. it must not dangle.
        assert!(matches!(result, LinkResult::Symlink | LinkResult::Hardlink));
        let read_back = std::fs::read(&tile).expect("tile reference must resolve, not dangle");
        assert_eq!(read_back, b"SHAREDPAYLOAD");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn materialize_reference_surfaces_total_failure() {
        // Regression (issue #98): when symlink, hardlink, and the placeholder
        // write all fail, the function used to return `Ok(ManifestOnly)` —
        // reporting success on total failure. A tile path whose parent
        // directory does not exist makes every strategy fail, so the call must
        // now surface an error and leave no file behind.
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("_shared").join("blank_x.png");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::write(&shared, b"payload").unwrap();

        let missing_parent = tmp.path().join("does").join("not").join("exist");
        let tile = missing_parent.join("tile.png");

        let result = materialize_reference(&tile, &shared);
        assert!(
            result.is_err(),
            "total failure must surface an error, not report success"
        );
        assert!(!tile.exists(), "no file should be left at the tile path");
    }

    /// Scoped `set_current_dir` helper. Serialises via a process-wide mutex so
    /// concurrent tests don't clobber each other's cwd, and restores the
    /// previous directory on drop.
    struct CwdGuard {
        prev: PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl CwdGuard {
        fn change_to(dir: &Path) -> Self {
            static CWD_LOCK: Mutex<()> = Mutex::new(());
            let lock = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let prev = std::env::current_dir().unwrap();
            std::env::set_current_dir(dir).unwrap();
            CwdGuard { prev, _lock: lock }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.prev);
        }
    }
}
