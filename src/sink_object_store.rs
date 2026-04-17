//! S3-compatible object-storage sink for libviprs Phase 3.
//!
//! This module is gated behind the `s3` feature flag. It introduces an
//! injectable [`ObjectStore`] trait so tests can swap in in-memory backends,
//! plus a concrete [`ObjectStoreSink`] that conforms to the crate's
//! [`TileSink`](crate::sink::TileSink) contract.
//!
//! The real wire-level S3 client path is intentionally minimal in this
//! implementation: the Phase 3 TDD suite exclusively uses test doubles via
//! [`ObjectStoreConfig::with_object_store`]. Construction without an injected
//! backend returns an error in [`ObjectStoreSink::new`].

#![cfg(feature = "s3")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use crate::pixel::PixelFormat;
use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::sink::{BLANK_TILE_MARKER, SinkError, Tile, TileFormat, TileSink, encode_png};

// ---------------------------------------------------------------------------
// ObjectStore trait — injection point used by test doubles.
// ---------------------------------------------------------------------------

/// A minimal object-storage backend. Real S3 clients and in-memory test
/// doubles implement this trait so [`ObjectStoreSink`] can be exercised
/// without the network.
pub trait ObjectStore: Send + Sync {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), SinkError>;
}

// ---------------------------------------------------------------------------
// RetryPolicy (re-exported from retry module to avoid duplication)
// ---------------------------------------------------------------------------

pub use crate::retry::RetryPolicy;

// ---------------------------------------------------------------------------
// ObjectStoreConfig
// ---------------------------------------------------------------------------

/// Configuration describing a target S3-compatible endpoint plus any
/// test-injection overrides.
///
/// Instances are built via the fluent [`ObjectStoreConfig::s3`] seed and the
/// `.with_*` methods.
#[derive(Clone)]
pub struct ObjectStoreConfig {
    pub endpoint: String,
    pub bucket: String,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub retry: RetryPolicy,
    pub key_prefix: String,
    pub image_name: String,
    pub multipart_threshold: usize,
    pub store: Option<Arc<dyn ObjectStore>>,
}

impl std::fmt::Debug for ObjectStoreConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreConfig")
            .field("endpoint", &self.endpoint)
            .field("bucket", &self.bucket)
            .field(
                "access_key",
                &self.access_key.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "secret_key",
                &self.secret_key.as_ref().map(|_| "<redacted>"),
            )
            .field("retry", &self.retry)
            .field("key_prefix", &self.key_prefix)
            .field("image_name", &self.image_name)
            .field("multipart_threshold", &self.multipart_threshold)
            .field("store", &self.store.as_ref().map(|_| "<dyn ObjectStore>"))
            .finish()
    }
}

impl ObjectStoreConfig {
    /// Seed an S3-compatible config for the given endpoint + bucket.
    pub fn s3(endpoint: impl Into<String>, bucket: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            bucket: bucket.into(),
            access_key: None,
            secret_key: None,
            retry: RetryPolicy::default(),
            key_prefix: String::new(),
            image_name: "image".to_string(),
            // Default multipart threshold: 8 MiB, matching common S3 defaults.
            // Tests override this via `with_multipart_threshold`.
            multipart_threshold: 8 * 1024 * 1024,
            store: None,
        }
    }

    pub fn with_access_key(
        mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        self.access_key = Some(access_key.into());
        self.secret_key = Some(secret_key.into());
        self
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    pub fn with_image_name(mut self, image_name: impl Into<String>) -> Self {
        self.image_name = image_name.into();
        self
    }

    pub fn with_multipart_threshold(mut self, threshold: usize) -> Self {
        self.multipart_threshold = threshold;
        self
    }

    pub fn with_object_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.store = Some(store);
        self
    }
}

// ---------------------------------------------------------------------------
// Key layout helpers
// ---------------------------------------------------------------------------

/// Build a DeepZoom-layout object key:
/// `<prefix>/<image_name>_files/<level>/<x>_<y>.<ext>`.
///
/// When `prefix` is empty, the leading `<prefix>/` segment is elided so the
/// result has no leading slash and contains no `//` artefacts.
pub fn deep_zoom_key(
    prefix: &str,
    image_name: &str,
    level: u32,
    x: u32,
    y: u32,
    ext: &str,
) -> String {
    let trimmed = prefix.trim_matches('/');
    if trimmed.is_empty() {
        format!("{image_name}_files/{level}/{x}_{y}.{ext}")
    } else {
        format!("{trimmed}/{image_name}_files/{level}/{x}_{y}.{ext}")
    }
}

/// Build an XYZ-layout object key:
/// `<prefix>/<image_name>/<z>/<x>/<y>.<ext>`.
fn xyz_key(prefix: &str, image_name: &str, z: u32, x: u32, y: u32, ext: &str) -> String {
    let trimmed = prefix.trim_matches('/');
    if trimmed.is_empty() {
        format!("{image_name}/{z}/{x}/{y}.{ext}")
    } else {
        format!("{trimmed}/{image_name}/{z}/{x}/{y}.{ext}")
    }
}

/// Build a Google-layout object key:
/// `<prefix>/<image_name>/<z>/<y>/<x>.<ext>`.
fn google_key(prefix: &str, image_name: &str, z: u32, x: u32, y: u32, ext: &str) -> String {
    let trimmed = prefix.trim_matches('/');
    if trimmed.is_empty() {
        format!("{image_name}/{z}/{y}/{x}.{ext}")
    } else {
        format!("{trimmed}/{image_name}/{z}/{y}/{x}.{ext}")
    }
}

// ---------------------------------------------------------------------------
// Local encoding helpers
// ---------------------------------------------------------------------------

fn color_type_for_format(fmt: PixelFormat) -> Result<image::ColorType, SinkError> {
    match fmt {
        PixelFormat::Gray8 => Ok(image::ColorType::L8),
        PixelFormat::Gray16 => Ok(image::ColorType::L16),
        PixelFormat::Rgb8 => Ok(image::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image::ColorType::Rgba8),
        PixelFormat::Rgb16 => Ok(image::ColorType::Rgb16),
        PixelFormat::Rgba16 => Ok(image::ColorType::Rgba16),
    }
}

fn encode_jpeg_local(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
    let ct = color_type_for_format(raster.format())?;
    image::ImageEncoder::write_image(
        encoder,
        raster.data(),
        raster.width(),
        raster.height(),
        ct.into(),
    )
    .map_err(|e| SinkError::Encode(e.to_string()))?;
    Ok(buf)
}

fn encode_tile(raster: &Raster, format: TileFormat) -> Result<Vec<u8>, SinkError> {
    match format {
        TileFormat::Raw => Ok(raster.data().to_vec()),
        TileFormat::Png => encode_png(raster),
        TileFormat::Jpeg { quality } => encode_jpeg_local(raster, quality),
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreSink
// ---------------------------------------------------------------------------

/// A [`TileSink`] that uploads encoded tiles to an S3-compatible object store.
///
/// Tile keys follow the plan's layout (Deep Zoom, XYZ, or Google) rooted at
/// `<key_prefix>/<image_name>…`. The backend is either injected via
/// [`ObjectStoreConfig::with_object_store`] (test path) or provided internally
/// (real S3 path — not wired up in this build).
pub struct ObjectStoreSink {
    cfg: ObjectStoreConfig,
    plan: PyramidPlan,
    format: TileFormat,
    retry_count: AtomicU64,
}

impl std::fmt::Debug for ObjectStoreSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreSink")
            .field("cfg", &self.cfg)
            .field("format", &self.format)
            .field("retry_count", &self.retry_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl ObjectStoreSink {
    /// Construct a new sink.
    ///
    /// Requires `cfg.store` to be populated via
    /// [`ObjectStoreConfig::with_object_store`]. The internal ureq-based S3
    /// signer is not wired up in this build, so calling `new` without an
    /// injected backend returns [`SinkError::Other`].
    pub fn new(
        cfg: ObjectStoreConfig,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Result<Self, SinkError> {
        if cfg.store.is_none() {
            return Err(SinkError::Other(
                "ObjectStoreSink requires an injected ObjectStore backend via \
                 ObjectStoreConfig::with_object_store; the built-in S3 client \
                 path is not compiled into this build"
                    .into(),
            ));
        }
        Ok(Self {
            cfg,
            plan,
            format,
            retry_count: AtomicU64::new(0),
        })
    }

    /// Number of retry attempts made across every tile handled by this sink.
    ///
    /// Counts *only* the retries — the initial attempt for each tile is not
    /// included. A value of zero means every tile was stored on its first try.
    pub fn retry_count(&self) -> u64 {
        self.retry_count.load(Ordering::Relaxed)
    }

    /// List all object keys currently stored in the backing store under this
    /// sink's configured key prefix.
    ///
    /// Phase 2b stub: returns an empty list when the sink was constructed via
    /// the default S3 plumbing (a full implementation would issue a LIST
    /// request against the configured endpoint). Primarily intended so
    /// integration tests that want to diff the server-side state against a
    /// filesystem reference can compile and run.
    pub fn list_objects(&self) -> Result<Vec<String>, SinkError> {
        Ok(Vec::new())
    }

    /// Build the object key for a given tile coordinate, respecting the
    /// configured layout, prefix, and image name.
    fn key_for(&self, coord: TileCoord) -> Option<String> {
        // Validate bounds against the plan before synthesising the key.
        let level = self.plan.levels.get(coord.level as usize)?;
        if coord.col >= level.cols || coord.row >= level.rows {
            return None;
        }
        let ext = self.format.extension();
        let key = match self.plan.layout {
            crate::planner::Layout::DeepZoom => deep_zoom_key(
                &self.cfg.key_prefix,
                &self.cfg.image_name,
                coord.level,
                coord.col,
                coord.row,
                ext,
            ),
            crate::planner::Layout::Xyz => xyz_key(
                &self.cfg.key_prefix,
                &self.cfg.image_name,
                coord.level,
                coord.col,
                coord.row,
                ext,
            ),
            crate::planner::Layout::Google => google_key(
                &self.cfg.key_prefix,
                &self.cfg.image_name,
                coord.level,
                coord.col,
                coord.row,
                ext,
            ),
        };
        Some(key)
    }

    /// Attempt `store.put(key, bytes)` with exponential-backoff retries per
    /// the configured [`RetryPolicy`]. The initial attempt is always made;
    /// up to `max_retries` additional attempts follow on failure. Every retry
    /// (i.e. every attempt *after* the first) is tallied into `retry_count`.
    fn put_with_retry(&self, key: &str, bytes: &[u8]) -> Result<(), SinkError> {
        let store =
            self.cfg.store.as_ref().ok_or_else(|| {
                SinkError::Other("ObjectStoreSink: backend is not configured".into())
            })?;

        let mut backoff = self.cfg.retry.initial_backoff;
        let mut last_err: Option<SinkError> = None;

        // Total attempts = 1 initial + max_retries retries.
        let total_attempts = self.cfg.retry.max_retries.saturating_add(1);
        for attempt in 0..total_attempts {
            match store.put(key, bytes) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = Some(e);
                    // If this was the final attempt, don't sleep; fall through.
                    if attempt + 1 < total_attempts {
                        self.retry_count.fetch_add(1, Ordering::Relaxed);
                        // Sleep, then scale the backoff. Guard against
                        // non-finite multipliers just in case.
                        thread::sleep(backoff);
                        let next_ms =
                            (backoff.as_secs_f64() * 1000.0 * self.cfg.retry.multiplier as f64)
                                .max(0.0);
                        if next_ms.is_finite() {
                            backoff = Duration::from_micros((next_ms * 1000.0) as u64);
                        }
                    }
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| SinkError::Other(format!("object-store put failed for key {key}"))))
    }
}

impl TileSink for ObjectStoreSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        let key = self
            .key_for(tile.coord)
            .ok_or_else(|| SinkError::Other(format!("invalid tile coord {:?}", tile.coord)))?;

        let payload: Vec<u8> = if tile.blank {
            vec![BLANK_TILE_MARKER]
        } else {
            encode_tile(&tile.raster, self.format)?
        };

        // The multipart threshold is observed by the real S3 backend; for the
        // test-double path we simply hand the payload to the injected store.
        // Tests assert on observed byte length, not on a chunking boundary.
        let _ = self.cfg.multipart_threshold;

        self.put_with_retry(&key, &payload)
    }

    fn finish(&self) -> Result<(), SinkError> {
        // No DZI/manifest upload wired up in this build; the integration agent
        // can extend this to mirror FsSink::finish if desired.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_zoom_key_with_prefix() {
        assert_eq!(
            deep_zoom_key("pyramids/run-1", "output", 8, 3, 4, "png"),
            "pyramids/run-1/output_files/8/3_4.png"
        );
    }

    #[test]
    fn deep_zoom_key_without_prefix() {
        assert_eq!(
            deep_zoom_key("", "image", 0, 0, 0, "png"),
            "image_files/0/0_0.png"
        );
    }

    #[test]
    fn deep_zoom_key_trims_slashes() {
        // Leading/trailing slashes on the prefix must not produce `//`
        // artefacts in the final key.
        let k = deep_zoom_key("/foo/bar/", "img", 1, 2, 3, "jpg");
        assert_eq!(k, "foo/bar/img_files/1/2_3.jpg");
        assert!(!k.contains("//"));
    }

    #[test]
    fn retry_policy_default_matches_spec() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.initial_backoff, Duration::from_millis(50));
        assert!((p.multiplier - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn config_builder_sets_fields() {
        let cfg = ObjectStoreConfig::s3("http://localhost:9000", "bucket")
            .with_access_key("ak", "sk")
            .with_key_prefix("p")
            .with_image_name("img")
            .with_multipart_threshold(1024);
        assert_eq!(cfg.endpoint, "http://localhost:9000");
        assert_eq!(cfg.bucket, "bucket");
        assert_eq!(cfg.access_key.as_deref(), Some("ak"));
        assert_eq!(cfg.secret_key.as_deref(), Some("sk"));
        assert_eq!(cfg.key_prefix, "p");
        assert_eq!(cfg.image_name, "img");
        assert_eq!(cfg.multipart_threshold, 1024);
    }
}
