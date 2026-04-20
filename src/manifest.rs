//! Versioned manifest for pyramid generation (Phase 3).
//!
//! Emits a `manifest.json` alongside the DZI XML that records generation
//! settings, source metadata, per-level counts, sparse policy, and optional
//! per-tile checksums. The top-level [`Manifest`] is a serde-tagged enum on
//! `schema_version` so future schema bumps reject old-reader parses cleanly
//! instead of silently falling through with default values.
//!
//! # Canonical wire format
//!
//! The upstream enums [`Layout`], [`TileFormat`], [`PixelFormat`], and
//! [`BlankTileStrategy`] do not derive serde, so this module owns thin
//! adapter modules that define the canonical on-disk representation:
//!
//! - `Layout`: `"deep_zoom"`, `"xyz"`, `"google"` (snake_case only).
//! - `TileFormat`: `{"kind": "png"}`, `{"kind": "jpeg", "quality": N}`,
//!   `{"kind": "raw"}`.
//! - `PixelFormat`: `"gray8"`, `"gray16"`, `"rgb8"`, `"rgba8"`, `"rgb16"`,
//!   `"rgba16"`.
//! - `BlankTileStrategy`: `{"kind": "emit"}`, `{"kind": "placeholder"}`,
//!   `{"kind": "placeholder_with_tolerance", "tolerance": N}`.
//!
//! Alternate spellings (PascalCase, etc.) are rejected. A follow-up can
//! delete these adapters once the upstream types derive serde.
//!
//! Forward compatibility is preserved by NOT using `#[serde(deny_unknown_fields)]`.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::BlankTileStrategy;
use crate::pixel::PixelFormat;
use crate::planner::Layout;
use crate::sink::TileFormat;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur while reading or writing a [`Manifest`].
#[derive(Debug, Error)]
pub enum ManifestError {
    /// Underlying I/O failure (file not found, permission denied, etc.).
    #[error("manifest I/O error: {0}")]
    Io(#[from] io::Error),
    /// JSON (de)serialization failure.
    #[error("manifest JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Deterministic map type used for `blank_references`. `BTreeMap` keeps
/// serialization order reproducible across runs.
pub type BlankReferences = BTreeMap<String, String>;

// ---------------------------------------------------------------------------
// ChecksumAlgo
// ---------------------------------------------------------------------------

/// Cryptographic hash algorithm used for per-tile checksums.
///
/// Serializes as a lowercase string (`"blake3"` / `"sha256"`) so the
/// manifest.json is human-readable and stable across releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChecksumAlgo {
    /// BLAKE3 cryptographic hash. Hex digest is 64 characters.
    Blake3,
    /// SHA-256 cryptographic hash. Hex digest is 64 characters.
    Sha256,
}

impl Default for ChecksumAlgo {
    /// Default algorithm is BLAKE3 (faster and modern). Tests that need a
    /// specific algorithm should name it explicitly.
    fn default() -> Self {
        Self::Blake3
    }
}

impl ChecksumAlgo {
    /// Returns the canonical lowercase string name for this algorithm.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Blake3 => "blake3",
            Self::Sha256 => "sha256",
        }
    }

    /// Hashes `bytes` and returns the lowercase hex digest.
    pub fn hash(&self, bytes: &[u8]) -> String {
        match self {
            Self::Blake3 => {
                let h = blake3::hash(bytes);
                hex_lower(h.as_bytes())
            }
            Self::Sha256 => {
                use sha2::Digest;
                let mut hasher = sha2::Sha256::new();
                hasher.update(bytes);
                let out = hasher.finalize();
                hex_lower(&out)
            }
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
// Serde adapters for upstream enums (which do not derive Serialize/Deserialize).
//
// These are kept in place because the upstream types live in other modules
// (`planner.rs`, `sink.rs`, `engine.rs`, `pixel.rs`) and deriving serde on
// them would require edits outside this file. Each adapter accepts ONLY the
// canonical snake_case form on deserialize; alternate spellings are rejected.
// Once the upstream types derive serde, these adapters can be deleted and the
// fields can use direct serde attributes.
// ---------------------------------------------------------------------------

mod layout_serde {
    use super::Layout;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Layout, s: S) -> Result<S::Ok, S::Error> {
        let name = match v {
            Layout::DeepZoom => "deep_zoom",
            Layout::Xyz => "xyz",
            Layout::Google => "google",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Layout, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "deep_zoom" => Ok(Layout::DeepZoom),
            "xyz" => Ok(Layout::Xyz),
            "google" => Ok(Layout::Google),
            other => Err(serde::de::Error::custom(format!("unknown layout: {other}"))),
        }
    }
}

mod tile_format_serde {
    use super::TileFormat;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "lowercase")]
    enum Repr {
        Png,
        Jpeg { quality: u8 },
        Raw,
    }

    pub fn serialize<S: Serializer>(v: &TileFormat, s: S) -> Result<S::Ok, S::Error> {
        let r = match *v {
            TileFormat::Png => Repr::Png,
            TileFormat::Jpeg { quality } => Repr::Jpeg { quality },
            TileFormat::Raw => Repr::Raw,
        };
        r.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TileFormat, D::Error> {
        let r = Repr::deserialize(d)?;
        Ok(match r {
            Repr::Png => TileFormat::Png,
            Repr::Jpeg { quality } => TileFormat::Jpeg { quality },
            Repr::Raw => TileFormat::Raw,
        })
    }
}

mod blank_strategy_serde {
    //! Canonical wire format for [`BlankTileStrategy`]:
    //! - `{"kind": "emit"}`
    //! - `{"kind": "placeholder"}`
    //! - `{"kind": "placeholder_with_tolerance", "tolerance": N}`
    //!
    //! `tolerance` maps to the upstream `max_channel_delta` field. It defaults
    //! to 0 when missing so manifests produced by older writers (or older
    //! `"placeholder"`-as-string values round-tripped through tooling) still
    //! parse cleanly.
    use super::BlankTileStrategy;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Repr {
        Emit,
        Placeholder,
        PlaceholderWithTolerance {
            #[serde(default)]
            tolerance: u8,
        },
    }

    pub fn serialize<S: Serializer>(v: &BlankTileStrategy, s: S) -> Result<S::Ok, S::Error> {
        let r = match *v {
            BlankTileStrategy::Emit => Repr::Emit,
            BlankTileStrategy::Placeholder => Repr::Placeholder,
            BlankTileStrategy::PlaceholderWithTolerance { max_channel_delta } => {
                Repr::PlaceholderWithTolerance {
                    tolerance: max_channel_delta,
                }
            }
        };
        r.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BlankTileStrategy, D::Error> {
        let r = Repr::deserialize(d)?;
        Ok(match r {
            Repr::Emit => BlankTileStrategy::Emit,
            Repr::Placeholder => BlankTileStrategy::Placeholder,
            Repr::PlaceholderWithTolerance { tolerance } => {
                BlankTileStrategy::PlaceholderWithTolerance {
                    max_channel_delta: tolerance,
                }
            }
        })
    }
}

mod pixel_format_serde {
    use super::PixelFormat;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &PixelFormat, s: S) -> Result<S::Ok, S::Error> {
        let name = match v {
            PixelFormat::Gray8 => "gray8",
            PixelFormat::Gray16 => "gray16",
            PixelFormat::Rgb8 => "rgb8",
            PixelFormat::Rgba8 => "rgba8",
            PixelFormat::Rgb16 => "rgb16",
            PixelFormat::Rgba16 => "rgba16",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<PixelFormat, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "gray8" => Ok(PixelFormat::Gray8),
            "gray16" => Ok(PixelFormat::Gray16),
            "rgb8" => Ok(PixelFormat::Rgb8),
            "rgba8" => Ok(PixelFormat::Rgba8),
            "rgb16" => Ok(PixelFormat::Rgb16),
            "rgba16" => Ok(PixelFormat::Rgba16),
            other => Err(serde::de::Error::custom(format!(
                "unknown pixel_format: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationSettings
// ---------------------------------------------------------------------------

/// Snapshot of the engine/pipeline knobs that produced the pyramid.
///
/// Captured at the moment the manifest is emitted so downstream consumers can
/// reproduce or verify the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GenerationSettings {
    /// Width/height of each tile in pixels (square tiles).
    pub tile_size: u32,
    /// Overlap in pixels between adjacent tiles.
    pub overlap: u32,
    /// Target layout (DeepZoom / XYZ / Google).
    #[serde(with = "layout_serde")]
    pub layout: Layout,
    /// Tile encoding format (PNG / JPEG{quality} / Raw).
    #[serde(with = "tile_format_serde")]
    pub format: TileFormat,
    /// Number of worker threads used during extraction.
    pub concurrency: usize,
    /// Background RGB triple used to pad edge tiles.
    pub background_rgb: [u8; 3],
    /// Blank-tile handling strategy (Emit / Placeholder).
    #[serde(with = "blank_strategy_serde")]
    pub blank_strategy: BlankTileStrategy,
}

// ---------------------------------------------------------------------------
// SourceMetadata
// ---------------------------------------------------------------------------

/// Metadata describing the source raster the pyramid was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SourceMetadata {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// Pixel format of the source raster.
    #[serde(with = "pixel_format_serde")]
    pub pixel_format: PixelFormat,
    /// Optional hex-encoded hash of the raw source bytes. Populated only when
    /// the manifest builder was configured with `include_source_hash(true)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_hash: Option<String>,
}

// ---------------------------------------------------------------------------
// LevelMetadata
// ---------------------------------------------------------------------------

/// Per-level counts and dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LevelMetadata {
    /// Zero-based level index (0 == smallest / most zoomed out for DeepZoom).
    pub level_index: u32,
    /// Level width in pixels.
    pub width: u32,
    /// Level height in pixels.
    pub height: u32,
    /// Number of tiles written to disk for this level.
    pub tiles_produced: u64,
    /// Number of tiles replaced by a placeholder / deduped reference.
    pub tiles_skipped: u64,
}

// ---------------------------------------------------------------------------
// SparsePolicy
// ---------------------------------------------------------------------------

/// Sparse-tile handling policy active during generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SparsePolicy {
    /// Per-channel tolerance used when detecting "blank" (uniform) tiles.
    /// 0 means exact-match only.
    pub tolerance: u8,
    /// Whether identical blank tiles are deduplicated into a shared reference.
    pub dedupe: bool,
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

/// Per-tile checksum table.
///
/// Uses `BTreeMap` so the serialized order is deterministic across runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Checksums {
    /// Hash algorithm used to compute each digest.
    pub algo: ChecksumAlgo,
    /// Map from relative tile path to lowercase hex digest.
    pub per_tile: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// BlankReference
// ---------------------------------------------------------------------------

/// Single blank-tile dedupe reference entry (reserved for future richer
/// metadata than the current `BTreeMap<String, String>` wire format).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BlankReference {
    /// Canonical relative path within the pyramid root.
    pub path: String,
    /// Digest (or other dedupe key) identifying the shared payload.
    pub digest: String,
}

// ---------------------------------------------------------------------------
// ManifestV1
// ---------------------------------------------------------------------------

/// Versioned pyramid manifest (schema v1).
///
/// Emitted by [`FsSink`](crate::sink::FsSink) when a [`ManifestBuilder`] is
/// attached via `with_manifest(...)`. Consumers should route through
/// [`Manifest`] (the tagged enum wrapper) so that future schema bumps can be
/// distinguished rather than silently accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ManifestV1 {
    /// Engine/pipeline settings that produced the pyramid.
    pub generation: GenerationSettings,
    /// Source raster metadata.
    pub source: SourceMetadata,
    /// Per-level counts.
    pub levels: Vec<LevelMetadata>,
    /// Sparse-tile handling policy.
    pub sparse_policy: SparsePolicy,
    /// Optional per-tile checksum table. None when the builder did not request
    /// checksums.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksums: Option<Checksums>,
    /// RFC3339 timestamp recorded when the manifest was emitted.
    pub created_at: String,
    /// Optional dedupe interop: map of canonical blank-tile reference paths to
    /// the digest (or equivalent key) of the shared blank payload.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub blank_references: BlankReferences,
}

impl ManifestV1 {
    /// The schema version literal this module emits.
    pub const SCHEMA_VERSION: &'static str = "1";

    /// Construct an empty manifest.
    ///
    /// Every optional field is zero/default; callers are expected to fill in
    /// real values before emitting.
    pub fn new(generation: GenerationSettings, source: SourceMetadata) -> Self {
        Self {
            generation,
            source,
            levels: Vec::new(),
            sparse_policy: SparsePolicy {
                tolerance: 0,
                dedupe: false,
            },
            checksums: None,
            created_at: String::new(),
            blank_references: BTreeMap::new(),
        }
    }

    /// Wrap this `ManifestV1` in the versioned [`Manifest`] enum.
    pub fn into_manifest(self) -> Manifest {
        Manifest::V1(self)
    }

    /// Serialize this manifest (as a v1) to a JSON string.
    ///
    /// Delegates to [`Manifest::to_json_string`] via a clone-and-wrap step for
    /// backward compatibility with call sites that still hold a `ManifestV1`.
    pub fn to_json_string(&self) -> Result<String, ManifestError> {
        self.clone().into_manifest().to_json_string()
    }

    /// Serialize and write this manifest to `path`, creating parent
    /// directories as needed.
    pub fn write_to(&self, path: &Path) -> Result<(), ManifestError> {
        self.clone().into_manifest().write_to(path)
    }

    /// Read and parse a `ManifestV1` from `path`.
    ///
    /// Unknown fields are silently ignored (forward compatibility). Returns an
    /// error if the on-disk schema version is not `"1"`.
    pub fn read_from(path: &Path) -> Result<Self, ManifestError> {
        match Manifest::read_from(path)? {
            Manifest::V1(m) => Ok(m),
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest (versioned wrapper)
// ---------------------------------------------------------------------------

/// Versioned pyramid manifest.
///
/// The `schema_version` JSON field is the serde tag: `"1"` maps to
/// [`Manifest::V1`]. Unknown versions produce a deserialization error, which
/// is the intentional contrast with the forward-compatible "unknown inner
/// fields are ignored" policy on [`ManifestV1`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "schema_version")]
pub enum Manifest {
    /// Schema v1.
    #[serde(rename = "1")]
    V1(ManifestV1),
}

impl Manifest {
    /// Returns the inner v1 payload by reference.
    pub fn as_v1(&self) -> &ManifestV1 {
        match self {
            Manifest::V1(m) => m,
        }
    }

    /// Consumes the wrapper and returns the inner v1 payload.
    pub fn into_v1(self) -> ManifestV1 {
        match self {
            Manifest::V1(m) => m,
        }
    }

    /// Serialize this manifest to a pretty JSON string.
    pub fn to_json_string(&self) -> Result<String, ManifestError> {
        serde_json::to_string_pretty(self).map_err(ManifestError::Json)
    }

    /// Serialize and write this manifest to `path`, creating parent
    /// directories as needed.
    pub fn write_to(&self, path: &Path) -> Result<(), ManifestError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = self.to_json_string()?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Read and parse a `Manifest` from `path`.
    pub fn read_from(path: &Path) -> Result<Self, ManifestError> {
        let bytes = fs::read(path)?;
        Self::from_json_slice(&bytes)
    }

    /// Parse a `Manifest` from a byte slice.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, ManifestError> {
        serde_json::from_slice(bytes).map_err(ManifestError::Json)
    }

    /// Parse a `Manifest` from any `io::Read`.
    pub fn from_json_reader<R: io::Read>(reader: R) -> Result<Self, ManifestError> {
        serde_json::from_reader(reader).map_err(ManifestError::Json)
    }
}

// ---------------------------------------------------------------------------
// ManifestBuilder
// ---------------------------------------------------------------------------

/// Fluent builder used to attach a manifest emitter to a sink.
///
/// The builder is cheap to construct and clone, carries no references, and is
/// consumed by `FsSink::with_manifest(...)` to configure the manifest emitter.
///
/// # Example
///
/// ```ignore
/// use libviprs::manifest::{ManifestBuilder, ChecksumAlgo};
/// let builder = ManifestBuilder::new()
///     .with_checksums(ChecksumAlgo::Blake3)
///     .with_dedupe(true);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestBuilder {
    checksums: Option<ChecksumAlgo>,
    include_source_hash: bool,
    dedupe: Option<bool>,
    tolerance: Option<u8>,
}

impl ManifestBuilder {
    /// Create a new builder with defaults: no checksums, no source hash, and
    /// dedupe/tolerance derived from the engine's active sparse policy.
    pub fn new() -> Self {
        Self {
            checksums: None,
            include_source_hash: false,
            dedupe: None,
            tolerance: None,
        }
    }

    /// Record a per-tile checksum of the on-disk bytes using `algo`.
    pub fn with_checksums(mut self, algo: ChecksumAlgo) -> Self {
        self.checksums = Some(algo);
        self
    }

    /// When `true`, hash the raw source raster bytes and record the digest in
    /// [`SourceMetadata::bytes_hash`].
    pub fn include_source_hash(mut self, enabled: bool) -> Self {
        self.include_source_hash = enabled;
        self
    }

    /// Override the dedupe flag in the emitted sparse policy.
    pub fn with_dedupe(mut self, dedupe: bool) -> Self {
        self.dedupe = Some(dedupe);
        self
    }

    /// Override the blank-tile tolerance in the emitted sparse policy.
    pub fn with_tolerance(mut self, tolerance: u8) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Returns the configured checksum algorithm, if any.
    pub fn checksum_algo(&self) -> Option<ChecksumAlgo> {
        self.checksums
    }

    /// Whether the source raster should be hashed into the manifest.
    pub fn wants_source_hash(&self) -> bool {
        self.include_source_hash
    }

    /// Returns the explicit dedupe override, if any.
    pub fn dedupe_override(&self) -> Option<bool> {
        self.dedupe
    }

    /// Returns the explicit tolerance override, if any.
    pub fn tolerance_override(&self) -> Option<u8> {
        self.tolerance
    }
}

impl Default for ManifestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> ManifestV1 {
        ManifestV1 {
            generation: GenerationSettings {
                tile_size: 256,
                overlap: 0,
                layout: Layout::DeepZoom,
                format: TileFormat::Jpeg { quality: 80 },
                concurrency: 4,
                background_rgb: [255, 255, 255],
                blank_strategy: BlankTileStrategy::Emit,
            },
            source: SourceMetadata {
                width: 1024,
                height: 768,
                pixel_format: PixelFormat::Rgba8,
                bytes_hash: None,
            },
            levels: vec![LevelMetadata {
                level_index: 0,
                width: 1,
                height: 1,
                tiles_produced: 1,
                tiles_skipped: 0,
            }],
            sparse_policy: SparsePolicy {
                tolerance: 0,
                dedupe: false,
            },
            checksums: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            blank_references: BTreeMap::new(),
        }
    }

    #[test]
    fn round_trips_through_json() {
        let m = sample_manifest().into_manifest();
        let s = m.to_json_string().unwrap();
        let parsed: Manifest = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn ignores_unknown_fields() {
        let m = sample_manifest().into_manifest();
        let mut v: serde_json::Value = serde_json::from_str(&m.to_json_string().unwrap()).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("future".into(), serde_json::json!("ignored"));
        let bumped = serde_json::to_string(&v).unwrap();
        let parsed: Manifest = serde_json::from_str(&bumped).unwrap();
        match parsed {
            Manifest::V1(_) => {}
        }
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let m = sample_manifest().into_manifest();
        let mut v: serde_json::Value = serde_json::from_str(&m.to_json_string().unwrap()).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("schema_version".into(), serde_json::json!("99"));
        let bumped = serde_json::to_string(&v).unwrap();
        let parsed: Result<Manifest, _> = serde_json::from_str(&bumped);
        assert!(parsed.is_err(), "unknown schema_version must fail to parse");
    }

    #[test]
    fn checksum_algo_serializes_lowercase() {
        let s = serde_json::to_string(&ChecksumAlgo::Blake3).unwrap();
        assert_eq!(s, "\"blake3\"");
        let s2 = serde_json::to_string(&ChecksumAlgo::Sha256).unwrap();
        assert_eq!(s2, "\"sha256\"");
    }

    #[test]
    fn blake3_hex_digest_is_64_chars() {
        let h = ChecksumAlgo::Blake3.hash(b"hello world");
        assert_eq!(h.len(), 64);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
        );
    }

    #[test]
    fn sha256_hex_digest_is_64_chars() {
        let h = ChecksumAlgo::Sha256.hash(b"hello world");
        assert_eq!(h.len(), 64);
        // Known SHA-256 of "hello world"
        assert_eq!(
            h,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn builder_records_options() {
        let b = ManifestBuilder::new()
            .with_checksums(ChecksumAlgo::Sha256)
            .include_source_hash(true)
            .with_dedupe(true)
            .with_tolerance(4);
        assert_eq!(b.checksum_algo(), Some(ChecksumAlgo::Sha256));
        assert!(b.wants_source_hash());
        assert_eq!(b.dedupe_override(), Some(true));
        assert_eq!(b.tolerance_override(), Some(4));
    }

    #[test]
    fn blank_strategy_with_tolerance_round_trips() {
        let m_in = ManifestV1 {
            generation: GenerationSettings {
                tile_size: 256,
                overlap: 0,
                layout: Layout::Xyz,
                format: TileFormat::Png,
                concurrency: 1,
                background_rgb: [0, 0, 0],
                blank_strategy: BlankTileStrategy::PlaceholderWithTolerance {
                    max_channel_delta: 7,
                },
            },
            source: SourceMetadata {
                width: 10,
                height: 10,
                pixel_format: PixelFormat::Rgb8,
                bytes_hash: None,
            },
            levels: Vec::new(),
            sparse_policy: SparsePolicy {
                tolerance: 7,
                dedupe: true,
            },
            checksums: None,
            created_at: String::new(),
            blank_references: BTreeMap::new(),
        };
        let wire = m_in.to_json_string().unwrap();
        let parsed = Manifest::from_json_slice(wire.as_bytes())
            .unwrap()
            .into_v1();
        assert_eq!(
            parsed.generation.blank_strategy,
            BlankTileStrategy::PlaceholderWithTolerance {
                max_channel_delta: 7,
            }
        );
    }
}
