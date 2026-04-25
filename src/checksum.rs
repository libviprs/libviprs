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

/// Parse the lowercase-string form (`"blake3"` / `"sha256"`) used in the
/// manifest JSON. Returns `None` for unknown names.
fn checksum_algo_from_manifest_str(s: &str) -> Option<ChecksumAlgo> {
    match s {
        "blake3" => Some(ChecksumAlgo::Blake3),
        "sha256" => Some(ChecksumAlgo::Sha256),
        _ => None,
    }
}

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
            let mut s = String::with_capacity(out.len() * 2);
            for b in out.iter() {
                use std::fmt::Write;
                let _ = write!(s, "{:02x}", b);
            }
            s
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

    #[error("checksum mismatch")]
    Mismatch,
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
    let algo = checksum_algo_from_manifest_str(algo_str)
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
        let abs = dir.join(&rel_path);

        let bytes = match std::fs::read(&abs) {
            Ok(b) => b,
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

        let got = hash_tile(&bytes, algo);
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
    fn algo_serde_roundtrip() {
        let j = serde_json::to_string(&ChecksumAlgo::Blake3).unwrap();
        assert_eq!(j, "\"blake3\"");
        let j = serde_json::to_string(&ChecksumAlgo::Sha256).unwrap();
        assert_eq!(j, "\"sha256\"");

        assert_eq!(
            checksum_algo_from_manifest_str("blake3"),
            Some(ChecksumAlgo::Blake3)
        );
        assert_eq!(
            checksum_algo_from_manifest_str("sha256"),
            Some(ChecksumAlgo::Sha256)
        );
        assert_eq!(checksum_algo_from_manifest_str("md5"), None);
    }
}
