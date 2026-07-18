//! Per-tile checksum emission and verification for libviprs Phase 3.
//!
//! This module supplies three pieces of machinery:
//!
//! 1. [`ChecksumAlgo`] / [`ChecksumMode`] — configuration enums chosen by the
//!    caller when wiring a sink (e.g. `FsSink::with_checksums(...)`).
//! 2. [`hash_tile`] — a small helper that hashes arbitrary tile bytes with the
//!    requested algorithm and returns the lowercase hex digest used in the
//!    manifest.
//! 3. [`verify_output`] — a post-hoc verifier that reads the manifest emitted
//!    alongside a pyramid and re-hashes every tile on disk, returning a
//!    [`VerifyReport`] describing the outcome.
//!
//! The [`ChecksumAlgo`] enum is intentionally defined here (rather than
//! re-exported from a `manifest` module) because the manifest module does not
//! yet exist in `lib.rs`. If/when the manifest module is introduced, the
//! integration agent is expected to dedupe these definitions — the `Serialize`
//! form (lowercase `"blake3"` / `"sha256"`) matches the on-disk shape tested
//! in the phase-3 integration suite.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sha2::Digest;
use thiserror::Error;

// ---------------------------------------------------------------------------
// ChecksumAlgo (re-exported from manifest to avoid duplication)
// ---------------------------------------------------------------------------

pub use crate::manifest::ChecksumAlgo;

// The lowercase manifest-string parser lives on `ChecksumAlgo`
// (`ChecksumAlgo::from_manifest_str`) so every verify path shares one
// definition and treats an unknown algorithm identically. See issue #95.

// ---------------------------------------------------------------------------
// ChecksumMode
// ---------------------------------------------------------------------------

/// How a sink should treat checksums.
///
/// The CLI surfaces this via
/// [`--manifest-emit-checksums`](https://libviprs.org/cli/#flag-manifest-emit-checksums)
/// (and, when verification is requested,
/// [`--verify`](https://libviprs.org/cli/#flag-verify)).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-manifest-emit-checksums)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChecksumMode {
    /// Do not compute or emit any per-tile checksums.
    #[default]
    None,
    /// Compute per-tile checksums and record them in the manifest.
    EmitOnly,
    /// Compute per-tile checksums, record them in the manifest, and verify
    /// the on-disk bytes match the computed hash before reporting success.
    Verify,
}

// ---------------------------------------------------------------------------
// hash_tile
// ---------------------------------------------------------------------------

/// Hash `bytes` with the requested algorithm and return the lowercase hex
/// digest. Both supported algorithms produce a 32-byte (64 hex char) output.
pub fn hash_tile(bytes: &[u8], algo: ChecksumAlgo) -> String {
    match algo {
        ChecksumAlgo::Blake3 => blake3::hash(bytes).to_hex().to_string(),
        ChecksumAlgo::Sha256 => {
            let mut hasher = sha2::Sha256::new();
            hasher.update(bytes);
            let out = hasher.finalize();
            crate::hex::hex_lower(&out)
        }
    }
}

// ---------------------------------------------------------------------------
// Untrusted-path sanitization + streaming hash
// ---------------------------------------------------------------------------

/// Join an attacker-controlled *relative* manifest path onto a trusted `root`,
/// rejecting anything that could escape `root`.
///
/// Manifest `per_tile` keys are attacker-controlled: a hostile bundle can list
/// `/etc/passwd`, a Windows drive prefix, or `../../secret`. `Path::join` with
/// an absolute or prefixed component silently *replaces* the base, and `..`
/// walks upward — both give arbitrary-file read (and a hash/existence oracle,
/// since the digest is echoed back on mismatch).
///
/// This returns `Some(root.join(rel))` only when every component of `rel` is a
/// plain name (or a redundant `.`); it returns `None` for an empty path or any
/// path containing a root, a prefix (e.g. `C:`), or a `..` component. Because
/// only `Normal`/`CurDir` components survive, the join can never escape `root`.
pub(crate) fn safe_manifest_join(root: &Path, rel: &str) -> Option<PathBuf> {
    use std::path::Component;

    let rel_path = Path::new(rel);
    if rel_path.as_os_str().is_empty() {
        return None;
    }
    for comp in rel_path.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => return None,
        }
    }
    Some(root.join(rel_path))
}

/// Hash the contents of the file at `path` by streaming it through the hasher
/// in fixed-size chunks, capping peak memory regardless of file size.
///
/// This replaces `std::fs::read` + [`hash_tile`], which buffered the whole file
/// and was an OOM primitive when pointed at a huge (or infinite, e.g.
/// `/dev/zero`) file. Returns the same lowercase hex digest as [`hash_tile`]
/// would for identical bytes.
pub(crate) fn hash_file(path: &Path, algo: ChecksumAlgo) -> std::io::Result<String> {
    use std::io::Read;

    // 64 KiB is large enough to amortize syscall/read overhead while keeping
    // the transient buffer trivially small.
    let mut buf = [0u8; 64 * 1024];
    let mut file = std::fs::File::open(path)?;

    match algo {
        ChecksumAlgo::Blake3 => {
            let mut hasher = blake3::Hasher::new();
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            Ok(hasher.finalize().to_hex().to_string())
        }
        ChecksumAlgo::Sha256 => {
            let mut hasher = sha2::Sha256::new();
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            let out = hasher.finalize();
            Ok(crate::hex::hex_lower(&out))
        }
    }
}

// ---------------------------------------------------------------------------
// VerifyReport / VerifyError
// ---------------------------------------------------------------------------

/// Summary of what [`verify_output`] found.
///
/// Produced by the CLI's [`--verify`](https://libviprs.org/cli/#flag-verify)
/// post-hoc check.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-verify)
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    /// Total number of tile entries considered (== size of the manifest's
    /// `checksums.per_tile` map).
    pub tiles_checked: u64,
    /// Number of tiles whose on-disk bytes hashed to the recorded digest.
    pub tiles_ok: u64,
    /// Tiles whose on-disk bytes did not match the recorded digest.
    pub tiles_mismatched: Vec<PathBuf>,
    /// Tile entries from the manifest that had no corresponding file on disk.
    pub tiles_missing: Vec<PathBuf>,
}

/// Errors produced by [`verify_output`].
///
/// Surfaced by the CLI's [`--verify`](https://libviprs.org/cli/#flag-verify) flag.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-verify)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerifyError {
    #[error("manifest.json not found (checked {sibling} and {inside})")]
    ManifestNotFound { sibling: PathBuf, inside: PathBuf },

    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse manifest JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("manifest missing required field: {0}")]
    MissingField(&'static str),

    #[error("manifest field {field} has unexpected shape: {reason}")]
    BadField {
        field: &'static str,
        reason: &'static str,
    },

    #[error("unknown checksum algorithm in manifest: {0}")]
    UnknownAlgo(String),

    /// A `per_tile` key was an absolute, prefixed, or `..`-containing path that
    /// would escape the pyramid directory. Rejected before any filesystem
    /// access so it cannot be used for path traversal or as a hash/existence
    /// oracle.
    #[error("manifest tile path escapes pyramid directory: {0}")]
    UnsafePath(String),
    // NOTE: per-tile checksum mismatches are *not* errors — `verify_output`
    // records them in [`VerifyReport::tiles_mismatched`] and still returns
    // `Ok`. `VerifyError` is reserved for structural failures (manifest
    // missing, unparseable, malformed, or an unsafe tile path).
}

// ---------------------------------------------------------------------------
// verify_output
// ---------------------------------------------------------------------------

/// Locate and read the `manifest.json` that sits alongside (or inside) the
/// pyramid directory `dir`.
///
/// Search order:
///   1. `<dir.parent>/<dir.file_name>.manifest.json` (sibling to the DZI/base).
///   2. `<dir>/manifest.json` (inside the pyramid dir).
fn load_manifest(dir: &Path) -> Result<(PathBuf, serde_json::Value), VerifyError> {
    let sibling = match (dir.parent(), dir.file_name()) {
        (Some(parent), Some(stem)) => {
            let mut name = stem.to_os_string();
            name.push(".manifest.json");
            parent.join(name)
        }
        _ => dir.join("__invalid_sibling__.manifest.json"),
    };

    if sibling.is_file() {
        let bytes = std::fs::read(&sibling).map_err(|e| VerifyError::Io {
            path: sibling.clone(),
            source: e,
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        return Ok((sibling, value));
    }

    let inside = dir.join("manifest.json");
    if inside.is_file() {
        let bytes = std::fs::read(&inside).map_err(|e| VerifyError::Io {
            path: inside.clone(),
            source: e,
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        return Ok((inside, value));
    }

    Err(VerifyError::ManifestNotFound { sibling, inside })
}

/// Post-hoc verifier. Reads the manifest for the pyramid at `dir`, re-hashes
/// every tile listed in `checksums.per_tile`, and reports mismatches / missing
/// files.
///
/// Returns `Err(...)` only for structural problems (manifest missing, bad
/// JSON, unknown algo). Individual tile mismatches or missing tiles are
/// reported via the returned [`VerifyReport`] rather than as errors.
pub fn verify_output(dir: &Path) -> Result<VerifyReport, VerifyError> {
    let (_manifest_path, manifest) = load_manifest(dir)?;

    // `checksums` may be absent / null — in that case there is nothing to do
    // and we report an empty, clean report.
    let checksums = match manifest.get("checksums") {
        None | Some(serde_json::Value::Null) => return Ok(VerifyReport::default()),
        Some(v) => v,
    };

    let algo_str = checksums
        .get("algo")
        .and_then(|v| v.as_str())
        .ok_or(VerifyError::MissingField("checksums.algo"))?;
    let algo = ChecksumAlgo::from_manifest_str(algo_str)
        .ok_or_else(|| VerifyError::UnknownAlgo(algo_str.to_string()))?;

    let per_tile = checksums
        .get("per_tile")
        .and_then(|v| v.as_object())
        .ok_or(VerifyError::MissingField("checksums.per_tile"))?;

    // Sort for deterministic report ordering.
    let entries: BTreeMap<&String, &serde_json::Value> = per_tile.iter().collect();

    let mut report = VerifyReport {
        tiles_checked: entries.len() as u64,
        ..VerifyReport::default()
    };

    for (rel, digest) in entries {
        let digest_hex = match digest.as_str() {
            Some(s) => s,
            None => {
                return Err(VerifyError::BadField {
                    field: "checksums.per_tile[value]",
                    reason: "expected string digest",
                });
            }
        };

        let rel_path = PathBuf::from(rel);
        // Reject traversal / absolute / prefixed paths before touching the
        // filesystem — a hostile manifest must not be able to read outside the
        // pyramid directory or probe file existence via the report.
        let abs = match safe_manifest_join(dir, rel) {
            Some(p) => p,
            None => return Err(VerifyError::UnsafePath(rel.clone())),
        };

        let got = match hash_file(&abs, algo) {
            Ok(g) => g,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                report.tiles_missing.push(rel_path);
                continue;
            }
            Err(e) => {
                return Err(VerifyError::Io {
                    path: abs,
                    source: e,
                });
            }
        };

        if got.eq_ignore_ascii_case(digest_hex) {
            report.tiles_ok += 1;
        } else {
            report.tiles_mismatched.push(rel_path);
        }
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_tile_blake3_matches_reference() {
        let data = b"hello, libviprs";
        let got = hash_tile(data, ChecksumAlgo::Blake3);
        let expected = blake3::hash(data).to_hex().to_string();
        assert_eq!(got, expected);
        assert_eq!(got.len(), 64);
        assert!(
            got.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
        );
    }

    #[test]
    fn hash_tile_sha256_has_correct_length_and_casing() {
        let got = hash_tile(b"abc", ChecksumAlgo::Sha256);
        assert_eq!(got.len(), 64);
        assert!(
            got.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
        );
        // Well-known SHA-256 of "abc".
        assert_eq!(
            got,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// Byte-for-byte reproduction of the open-coded lowercase-hex encoder that
    /// `hash_tile` / `hash_file` used before they were routed through
    /// `crate::hex::hex_lower` (the #302 follow-up). Kept local to the test so
    /// the swap can be proven output-preserving.
    fn old_inline_02x(bytes: &[u8]) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes.iter() {
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    #[test]
    fn hex_encoding_swap_is_byte_identical_on_checksum_path() {
        // `hash_tile` / `hash_file` previously hex-encoded the SHA-256 digest
        // with an inline `write!("{:02x}")` loop; both were replaced with
        // `crate::hex::hex_lower`. The hashing is untouched, so proving the two
        // encoders agree byte-for-byte proves the emitted digest strings — and
        // therefore the checksum reference goldens — are unchanged. Cover empty
        // input, each nibble in isolation, a full 0..=255 sweep, and the 32-byte
        // SHA-256 digest width.
        let digest_width: Vec<u8> = (0u8..32).collect();
        let all_bytes: Vec<u8> = (0u8..=255).collect();
        let cases: &[&[u8]] = &[
            &[],           // empty input → ""
            &[0x00],       // zero-padded low nibble
            &[0x0f],       // low nibble only
            &[0xf0],       // high nibble only
            &[0xff],       // both nibbles set
            &digest_width, // full SHA-256 digest width (32 bytes → 64 chars)
            &all_bytes,    // every possible byte value
        ];
        for bytes in cases {
            let via_helper = crate::hex::hex_lower(bytes);
            let via_old = old_inline_02x(bytes);
            assert_eq!(via_helper, via_old, "encoders diverged for {bytes:02x?}");
            assert_eq!(via_helper.len(), bytes.len() * 2, "width not two-per-byte");
        }
        // Explicit empty-input contract shared with the checksum path.
        assert_eq!(crate::hex::hex_lower(&[]), "");

        // End-to-end pin: the real SHA-256 path still yields the known golden,
        // now that it encodes via `hex_lower`.
        assert_eq!(
            hash_tile(b"abc", ChecksumAlgo::Sha256),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
    }

    #[test]
    fn blank_marker_hash_is_stable() {
        // One byte 0x00 is the canonical BLANK_TILE_MARKER; its hash is the
        // value that will appear in the per-tile manifest for placeholders.
        let got = hash_tile(&[0x00u8], ChecksumAlgo::Blake3);
        let expected = blake3::hash(&[0x00u8]).to_hex().to_string();
        assert_eq!(got, expected);
    }

    #[test]
    fn checksum_mode_default_is_none() {
        assert_eq!(ChecksumMode::default(), ChecksumMode::None);
    }

    #[test]
    fn safe_manifest_join_rejects_escaping_paths() {
        let root = Path::new("/tmp/pyramid");
        // Absolute / rooted paths.
        assert!(safe_manifest_join(root, "/etc/passwd").is_none());
        // Parent-dir traversal, at any depth.
        assert!(safe_manifest_join(root, "../secret").is_none());
        assert!(safe_manifest_join(root, "0/../../secret").is_none());
        assert!(safe_manifest_join(root, "a/b/../../../x").is_none());
        // Empty path.
        assert!(safe_manifest_join(root, "").is_none());
    }

    #[test]
    fn safe_manifest_join_accepts_plain_relative_paths() {
        let root = Path::new("/tmp/pyramid");
        assert_eq!(
            safe_manifest_join(root, "0/0_0.raw"),
            Some(PathBuf::from("/tmp/pyramid/0/0_0.raw"))
        );
        // A redundant `.` is harmless and does not escape.
        assert_eq!(
            safe_manifest_join(root, "./0/0_0.raw"),
            Some(PathBuf::from("/tmp/pyramid/./0/0_0.raw"))
        );
    }

    #[test]
    fn hash_file_matches_in_memory_hash_for_both_algos() {
        let dir = tempfile::tempdir().unwrap();
        // Larger than the 64 KiB streaming chunk to exercise multiple reads.
        let bytes = vec![0x5Au8; 200 * 1024 + 3];
        let p = dir.path().join("tile.raw");
        std::fs::write(&p, &bytes).unwrap();

        for algo in [ChecksumAlgo::Blake3, ChecksumAlgo::Sha256] {
            assert_eq!(hash_file(&p, algo).unwrap(), hash_tile(&bytes, algo));
        }
    }

    #[test]
    fn hash_file_reports_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.raw");
        let err = hash_file(&missing, ChecksumAlgo::Blake3).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn verify_output_reports_tile_mismatch_in_report_not_as_error() {
        // The "real condition" the removed `VerifyError::Mismatch` variant was
        // meant to represent: a tile whose on-disk bytes do not hash to the
        // recorded digest. By design that is NOT an error — `verify_output`
        // returns `Ok` and records the offending tile in
        // `VerifyReport::tiles_mismatched`. This pins that contract so no one
        // re-introduces a context-free error variant for a per-tile mismatch.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // A tile whose bytes match its recorded digest, and one whose do not.
        let ok_bytes = b"good tile bytes";
        let bad_bytes = b"actual bytes on disk";
        std::fs::create_dir_all(root.join("0")).unwrap();
        std::fs::write(root.join("0/0_0.raw"), ok_bytes).unwrap();
        std::fs::write(root.join("0/1_0.raw"), bad_bytes).unwrap();

        let ok_digest = hash_tile(ok_bytes, ChecksumAlgo::Blake3);
        // A digest that deliberately disagrees with `bad_bytes`.
        let wrong_digest = "0000000000000000000000000000000000000000000000000000000000000000";
        assert_ne!(hash_tile(bad_bytes, ChecksumAlgo::Blake3), wrong_digest);

        let manifest = serde_json::json!({
            "checksums": {
                "algo": "blake3",
                "per_tile": {
                    "0/0_0.raw": ok_digest,
                    "0/1_0.raw": wrong_digest,
                }
            }
        });
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let report =
            verify_output(root).expect("a per-tile mismatch must not be a structural error");
        assert_eq!(report.tiles_checked, 2);
        assert_eq!(report.tiles_ok, 1);
        assert!(report.tiles_missing.is_empty());
        assert_eq!(report.tiles_mismatched, vec![PathBuf::from("0/1_0.raw")]);
    }

    #[test]
    fn algo_serde_roundtrip() {
        let j = serde_json::to_string(&ChecksumAlgo::Blake3).unwrap();
        assert_eq!(j, "\"blake3\"");
        let j = serde_json::to_string(&ChecksumAlgo::Sha256).unwrap();
        assert_eq!(j, "\"sha256\"");

        assert_eq!(
            ChecksumAlgo::from_manifest_str("blake3"),
            Some(ChecksumAlgo::Blake3)
        );
        assert_eq!(
            ChecksumAlgo::from_manifest_str("sha256"),
            Some(ChecksumAlgo::Sha256)
        );
        assert_eq!(ChecksumAlgo::from_manifest_str("md5"), None);
    }
}
