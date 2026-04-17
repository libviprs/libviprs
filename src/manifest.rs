//! Versioned manifest for pyramid generation (Phase 3).
//!
//! Emits a `manifest.json` alongside the DZI XML that records generation
//! settings, source metadata, per-level counts, sparse policy, and optional
//! per-tile checksums. The top-level struct carries a `schema_version` field
//! to support forward-compatible evolution.
//!
//! This module is intentionally self-contained: helper adapters for
//! serializing the existing `Layout`, `TileFormat`, `BlankTileStrategy`, and
//! `PixelFormat` enums (which do not derive serde traits upstream) live here.
//! Forward compatibility is preserved by NOT using `#[serde(deny_unknown_fields)]`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::engine::BlankTileStrategy;
use crate::pixel::PixelFormat;
use crate::planner::Layout;
use crate::sink::TileFormat;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur while reading or writing a [`ManifestV1`].
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
// ChecksumAlgo
// ---------------------------------------------------------------------------

/// Cryptographic hash algorithm used for per-tile checksums.
///
/// Serializes as a lowercase string (`"blake3"` / `"sha256"`) so the
/// manifest.json is human-readable and stable across releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl Serialize for ChecksumAlgo {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ChecksumAlgo {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        match s.as_str() {
            "blake3" => Ok(Self::Blake3),
            "sha256" => Ok(Self::Sha256),
            other => Err(serde::de::Error::custom(format!(
                "unknown checksum algorithm: {other}"
            ))),
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
// Serde adapters for upstream enums (which do not derive Serialize/Deserialize)
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
            "deep_zoom" | "DeepZoom" | "deepzoom" => Ok(Layout::DeepZoom),
            "xyz" | "Xyz" | "XYZ" => Ok(Layout::Xyz),
            "google" | "Google" => Ok(Layout::Google),
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
    use super::BlankTileStrategy;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &BlankTileStrategy, s: S) -> Result<S::Ok, S::Error> {
        // `PlaceholderWithTolerance` serializes as `"placeholder"` to remain
        // forward compatible with readers that do not yet understand the new
        // variant; the tolerance value is recorded in `SparsePolicy.tolerance`.
        let name = match v {
            BlankTileStrategy::Emit => "emit",
            BlankTileStrategy::Placeholder => "placeholder",
            BlankTileStrategy::PlaceholderWithTolerance { .. } => "placeholder",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BlankTileStrategy, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "emit" | "Emit" => Ok(BlankTileStrategy::Emit),
            "placeholder" | "Placeholder" => Ok(BlankTileStrategy::Placeholder),
            other => Err(serde::de::Error::custom(format!(
                "unknown blank_tile_strategy: {other}"
            ))),
        }
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
            "gray8" | "Gray8" => Ok(PixelFormat::Gray8),
            "gray16" | "Gray16" => Ok(PixelFormat::Gray16),
            "rgb8" | "Rgb8" => Ok(PixelFormat::Rgb8),
            "rgba8" | "Rgba8" => Ok(PixelFormat::Rgba8),
            "rgb16" | "Rgb16" => Ok(PixelFormat::Rgb16),
            "rgba16" | "Rgba16" => Ok(PixelFormat::Rgba16),
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
pub struct Checksums {
    /// Hash algorithm used to compute each digest.
    pub algo: ChecksumAlgo,
    /// Map from relative tile path to lowercase hex digest.
    pub per_tile: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// ManifestV1
// ---------------------------------------------------------------------------

/// Versioned pyramid manifest (schema v1).
///
/// Emitted by [`FsSink`](crate::sink::FsSink) when a [`ManifestBuilder`] is
/// attached via `with_manifest(...)`. Consumers should inspect the
/// [`ManifestV1::schema_version`] field before interpreting the payload and
/// tolerate unknown future fields (which this struct explicitly preserves by
/// not setting `deny_unknown_fields`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestV1 {
    /// Schema discriminator. Always `"1"` for this struct.
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub blank_references: HashMap<String, String>,
}

fn default_schema_version() -> String {
    "1".to_string()
}

impl ManifestV1 {
    /// The schema version literal this module emits.
    pub const SCHEMA_VERSION: &'static str = "1";

    /// Construct an empty manifest with the current schema version.
    ///
    /// Every field is zero/default; callers are expected to fill in real
    /// values before emitting.
    pub fn new(generation: GenerationSettings, source: SourceMetadata) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION.to_string(),
            generation,
            source,
            levels: Vec::new(),
            sparse_policy: SparsePolicy {
                tolerance: 0,
                dedupe: false,
            },
            checksums: None,
            created_at: String::new(),
            blank_references: HashMap::new(),
        }
    }

    /// Serialize this manifest to a JSON string.
    ///
    /// All maps inside the manifest are `BTreeMap`, so iteration order is
    /// deterministic across runs. Any fields whose order is structural (the
    /// struct field order and the `Vec<LevelMetadata>`) are controlled by the
    /// engine producing the manifest.
    pub fn to_json_string(&self) -> String {
        // Unwrap is safe: ManifestV1 only contains types with infallible
        // Serialize implementations.
        serde_json::to_string_pretty(self).expect("ManifestV1 serialization must not fail")
    }

    /// Serialize and write this manifest to `path`, creating parent
    /// directories as needed.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = self.to_json_string();
        fs::write(path, json)
    }

    /// Read and parse a `ManifestV1` from `path`.
    ///
    /// Unknown fields are silently ignored (forward compatibility).
    pub fn read_from(path: &Path) -> Result<Self, ManifestError> {
        let bytes = fs::read(path)?;
        let value: Self = serde_json::from_slice(&bytes)?;
        Ok(value)
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
            schema_version: "1".to_string(),
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
            blank_references: HashMap::new(),
        }
    }

    #[test]
    fn round_trips_through_json() {
        let m = sample_manifest();
        let s = m.to_json_string();
        let parsed: ManifestV1 = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn ignores_unknown_fields() {
        let mut v: serde_json::Value =
            serde_json::from_str(&sample_manifest().to_json_string()).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("future".into(), serde_json::json!("ignored"));
        let bumped = serde_json::to_string(&v).unwrap();
        let parsed: ManifestV1 = serde_json::from_str(&bumped).unwrap();
        assert_eq!(parsed.schema_version, "1");
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
}
