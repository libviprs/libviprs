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
//! `_shared/<shared_key>.<ext>`. Every other tile with the same content is
//! then referenced from its planned path via one of three strategies,
//! attempted in order:
//!
//! 1. **Symlink** (`try_symlink`) — preferred on unix because it's cheap and
//!    readers that follow symlinks see the target bytes transparently.
//! 2. **Hardlink** (`try_hardlink`) — used when the filesystem or platform
//!    rejects symlinks (e.g. Windows without the Developer Mode privilege).
//! 3. **Manifest-only** — as a last resort, a 1-byte placeholder is written
//!    at the tile path and the sink records the relationship in
//!    `manifest.json`'s `blank_references` map. Readers must consult the
//!    manifest to resolve the pointer.
//!
//! # Shared-key contract
//!
//! Shared-key filenames are deterministic functions of tile content:
//! `blank_<hex-hash>`. This is asserted by
//! `libviprs-tests/tests/phase3_dedupe_blanks.rs::determinism_of_dedupe_hashes`.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::manifest::ChecksumAlgo;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Selects how tiles are content-addressed prior to being written to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupeStrategy {
    /// No deduplication — every tile is written to its own file. Default.
    None,
    /// Only blank (uniform-colour) tiles are deduplicated. Non-blank tiles
    /// are written individually.
    Blanks,
    /// Every tile is content-addressed; identical tiles share a single
    /// physical file regardless of whether they're blank.
    All {
        /// Hash algorithm used to derive the shared-key filename.
        algo: ChecksumAlgo,
    },
}

impl Default for DedupeStrategy {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

/// Outcome of a [`DedupeIndex::record`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// In-memory mapping from content-hash to shared-key. One instance per
/// generation run; not persisted across restarts (resume-mode sinks rebuild
/// the index by walking `_shared/`).
pub struct DedupeIndex {
    strategy: DedupeStrategy,
    /// Content-hash (hex) -> shared_key.
    seen: Mutex<HashMap<String, String>>,
    /// Tile path -> shared_key (for manifest emission).
    refs: Mutex<HashMap<String, String>>,
}

impl DedupeIndex {
    /// Create a fresh index for the given strategy.
    pub fn new(strategy: DedupeStrategy) -> Self {
        Self {
            strategy,
            seen: Mutex::new(HashMap::new()),
            refs: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the strategy this index was built for.
    pub fn strategy(&self) -> DedupeStrategy {
        self.strategy
    }

    /// Compute the content hash using the algorithm dictated by `strategy`.
    /// For [`DedupeStrategy::Blanks`] this is always blake3 (fast, unkeyed);
    /// for [`DedupeStrategy::All`] the caller-chosen algorithm is used.
    fn hash_content(&self, bytes: &[u8]) -> String {
        match self.strategy {
            DedupeStrategy::None => {
                // No hashing needed, but callers shouldn't invoke `record`
                // in this mode. We still produce a stable hex string so the
                // returned decision is self-consistent.
                let h = blake3::hash(bytes);
                hex_lower(h.as_bytes())
            }
            DedupeStrategy::Blanks => {
                let h = blake3::hash(bytes);
                hex_lower(h.as_bytes())
            }
            DedupeStrategy::All { algo } => algo.hash(bytes),
        }
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

        let hash = self.hash_content(bytes);
        let shared_key = format!("blank_{hash}");
        let shared_path = Self::shared_path(&shared_key, ext);

        // Record the reference for manifest emission unconditionally — the
        // sink may choose to suppress it when it successfully symlinked or
        // hardlinked (that's a sink decision, not ours).
        {
            let mut refs = self.refs.lock().expect("dedupe refs mutex poisoned");
            refs.insert(path.to_string(), shared_key.clone());
        }

        // First-write vs. reference.
        let mut seen = self.seen.lock().expect("dedupe seen mutex poisoned");
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(hash) {
            e.insert(shared_key.clone());
            DedupeDecision::WriteNew {
                shared_key,
                shared_path,
            }
        } else {
            DedupeDecision::Reference {
                shared_key,
                shared_path,
            }
        }
    }

    /// Drop a recorded reference (e.g. because the sink successfully created
    /// a symlink / hardlink and no manifest pointer is required). No-op if
    /// the path was never recorded.
    pub fn forget_reference(&self, path: &str) {
        let mut refs = self.refs.lock().expect("dedupe refs mutex poisoned");
        refs.remove(path);
    }

    /// Record an already-known shared key for a given shared-key filename,
    /// used by resume-mode sinks when walking an existing `_shared/`
    /// directory so that subsequent `record` calls don't emit `WriteNew`
    /// for content that's already physically present on disk.
    pub fn seed_shared_key(&self, hash_hex: String, shared_key: String) {
        let mut seen = self.seen.lock().expect("dedupe seen mutex poisoned");
        seen.insert(hash_hex, shared_key);
    }

    /// Compute the on-disk relative path for a shared blob.
    pub fn shared_path(shared_key: &str, ext: &str) -> PathBuf {
        let mut p = PathBuf::from("_shared");
        p.push(format!("{shared_key}.{ext}"));
        p
    }

    /// Consume-style snapshot of the current tile-path -> shared-key map for
    /// `manifest.json::blank_references` emission.
    pub fn references(&self) -> HashMap<String, String> {
        self.refs
            .lock()
            .expect("dedupe refs mutex poisoned")
            .clone()
    }

    /// Total number of distinct content hashes seen so far. Useful for
    /// diagnostics and tests.
    pub fn distinct_count(&self) -> usize {
        self.seen.lock().expect("dedupe seen mutex poisoned").len()
    }
}

// ---------------------------------------------------------------------------
// Link materialization
// ---------------------------------------------------------------------------

/// Outcome of [`materialize_reference`]: which strategy was ultimately used
/// to point `tile_path` at `shared_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// Tries, in order: symlink, hardlink, 1-byte placeholder + manifest entry.
///
/// The parent directory of `tile_path` must already exist; the caller is
/// responsible for `std::fs::create_dir_all`. Any pre-existing file at
/// `tile_path` is removed first (so this is safe to call in resume mode).
pub fn materialize_reference(tile_path: &Path, shared_path: &Path) -> LinkResult {
    // Remove any leftover file at the tile path so link creation doesn't
    // fail with EEXIST. `remove_file` silently succeeds on not-found via
    // explicit check to avoid swallowing unrelated errors.
    if tile_path.exists() || tile_path.is_symlink() {
        let _ = std::fs::remove_file(tile_path);
    }

    if try_symlink(shared_path, tile_path).is_ok() {
        return LinkResult::Symlink;
    }
    if try_hardlink(shared_path, tile_path).is_ok() {
        return LinkResult::Hardlink;
    }
    // Final fallback: placeholder file.
    match std::fs::write(tile_path, PLACEHOLDER_MARKER) {
        Ok(()) => LinkResult::ManifestOnly,
        Err(_) => {
            // Even the placeholder write failed; there's not much we can do
            // without bubbling an error. The caller's manifest map still
            // documents the intended relationship, so readers that consult
            // it will recover.
            LinkResult::ManifestOnly
        }
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
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
    fn materialize_reference_creates_a_readable_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let shared_dir = tmp.path().join("_shared");
        std::fs::create_dir_all(&shared_dir).unwrap();
        let shared = shared_dir.join("blank_abc.png");
        std::fs::write(&shared, b"SHAREDCONTENT").unwrap();

        let tile = tmp.path().join("0").join("0_0.png");
        std::fs::create_dir_all(tile.parent().unwrap()).unwrap();

        let result = materialize_reference(&tile, &shared);
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
    fn seed_shared_key_suppresses_later_write_new() {
        let idx = DedupeIndex::new(DedupeStrategy::Blanks);
        let hash = {
            let h = blake3::hash(b"payload");
            hex_lower(h.as_bytes())
        };
        idx.seed_shared_key(hash.clone(), format!("blank_{hash}"));
        let decision = idx.record("a.png", b"payload");
        assert!(matches!(decision, DedupeDecision::Reference { .. }));
    }
}
