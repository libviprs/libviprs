//! S3-compatible object-storage sink for libviprs Phase 3.
//!
//! This module is gated behind the `object-store-sink` feature flag (the
//! deprecated `s3` alias also enables it). It introduces an
//! injectable [`ObjectStore`] trait so tests can swap in in-memory backends,
//! plus a concrete [`ObjectStoreSink`] that conforms to the crate's
//! [`TileSink`] contract.
//!
//! The real wire-level S3 client path is intentionally minimal in this
//! implementation: the Phase 3 TDD suite exclusively uses test doubles via
//! [`ObjectStoreConfig::with_object_store`]. Construction without an injected
//! backend returns an error in [`ObjectStoreSink::new`].
//!
//! Retry behaviour is **not** built into [`ObjectStoreSink`]. Callers who want
//! automatic retries compose one externally:
//!
//! ```ignore
//! use libviprs::retry::{FailurePolicy, RetryPolicy, RetryingSink};
//! let sink = RetryingSink::new(
//!     ObjectStoreSink::new(cfg, plan, fmt)?,
//!     RetryPolicy::default(),
//! );
//! ```

use std::sync::Arc;

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

    /// Enumerate the object keys stored under `prefix`.
    ///
    /// **Defaulted to a loud refusal.** A listing-capable backend — a real
    /// object-store client, or a test double that tracks its own key space —
    /// overrides this to return the keys it holds. Write-only backends (any
    /// store that only implements `put`) inherit this default, which fails loud
    /// with [`SinkError::Unsupported`] rather than returning a misleading empty
    /// `Ok(vec![])`. A silent empty list is a footgun: a caller diffing
    /// server-side keys against a filesystem reference cannot distinguish
    /// "nothing was uploaded" from "this backend cannot list", so the diff
    /// passes falsely or fails spuriously.
    ///
    /// [`ObjectStoreSink::list_objects`] delegates here, so the sink no longer
    /// hardcodes the refusal — listing capability now lives with the backend,
    /// and no sink edit is needed when a listing-capable transport lands.
    fn list(&self, prefix: &str) -> Result<Vec<String>, SinkError> {
        let _ = prefix;
        Err(SinkError::Unsupported(
            "list_objects: this ObjectStore backend does not implement a LIST \
             operation (ObjectStore::list is defaulted to refuse). A \
             listing-capable backend overrides it; write-only backends inherit \
             this loud failure. Do not treat this as an empty object set."
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreConfig
// ---------------------------------------------------------------------------

/// Configuration describing a target S3-compatible endpoint plus any
/// test-injection overrides.
///
/// Instances are built via the fluent [`ObjectStoreConfig::s3`] seed and the
/// `.with_*` methods.
///
/// Retry behaviour is **not** configured here — wrap the resulting sink in
/// [`crate::retry::RetryingSink`] if you need automatic retries.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-sink)
#[derive(Clone)]
#[non_exhaustive]
pub struct ObjectStoreConfig {
    pub endpoint: String,
    pub bucket: String,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub key_prefix: String,
    pub image_name: String,
    pub multipart_threshold: usize,
    /// Injected object-storage backend. Kept private so the only supported
    /// mutation path is [`ObjectStoreConfig::with_object_store`]; tests and
    /// production code both go through that builder.
    store: Option<Arc<dyn ObjectStore>>,
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

    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    pub fn with_image_name(mut self, image_name: impl Into<String>) -> Self {
        self.image_name = image_name.into();
        self
    }

    /// Set the multipart-upload threshold (in bytes).
    ///
    /// **Currently inert in this build.** The threshold is stored on the config
    /// and observed only by a real multipart-capable object-store transport,
    /// which is not compiled in (see [`ObjectStoreSink::list_objects`] and the
    /// Phase 3 TODO in [`ObjectStoreSink::new`]). The injectable [`ObjectStore`]
    /// trait exposes only a single `put`, so no chunking boundary is applied to
    /// uploads regardless of this value. It is retained so callers can configure
    /// the intended threshold ahead of that transport landing, at which point
    /// this setting will take effect.
    pub fn with_multipart_threshold(mut self, threshold: usize) -> Self {
        self.multipart_threshold = threshold;
        self
    }

    /// Inject an [`ObjectStore`] backend — the only supported way to attach a
    /// backend to the config. The Phase 3 TDD suite uses in-memory test
    /// doubles here; a production integration wires up the real S3 client.
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

/// The `image` colour type a tile is encoded through, or a typed refusal for
/// the carriers that have none.
///
/// The same refusal [`crate::encode`]'s `image_color_type` makes, argued the
/// same way, and its doc carries the measured oracle: three vips routes
/// answer three different things for a `uint` or `float` raster and the
/// interpretation tag moves one of them again, so there is no answer here to
/// be faithful to (issue #952).
fn color_type_for_format(fmt: PixelFormat) -> Result<image::ColorType, SinkError> {
    match fmt {
        PixelFormat::Gray8 => Ok(image::ColorType::L8),
        PixelFormat::Gray16 => Ok(image::ColorType::L16),
        PixelFormat::Rgb8 => Ok(image::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image::ColorType::Rgba8),
        PixelFormat::Rgb16 => Ok(image::ColorType::Rgb16),
        PixelFormat::Rgba16 => Ok(image::ColorType::Rgba16),
        // Multiband intermediates (from the band ops in `crate::bands`) have
        // no image-crate colour type; reduce or extract to 1/3/4 bands first.
        PixelFormat::Multi8(_) | PixelFormat::Multi16(_) => Err(SinkError::EncodeMsg(format!(
            "multiband raster ({} bands) cannot be encoded as an image tile",
            fmt.channels()
        ))),
        // Float compute intermediates have no PNG/JPEG representation;
        // cast to an unsigned 8/16-bit format before encoding tiles.
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => Err(SinkError::EncodeMsg(format!(
            "float raster ({fmt:?}) cannot be encoded as an image tile; \
             cast to an unsigned 8/16-bit format first"
        ))),
        // The `image` crate's widest integer colour type is 16-bit (L8,
        // L16, Rgb8, Rgba8, Rgb16, Rgba16, and two 32-bit *float* ones),
        // so the 32-bit unsigned carrier has no representation there at
        // all. A typed refusal is the implementation here rather than a
        // gap: unlike `resize` or the band ops, where vips supports `uint`
        // and refusing would have been a parity regression, there is
        // nothing to be faithful to (issue #517). Measured, rather than
        // argued from the dependency: `vips pngsave` on a `uint`
        // `[3000000000, 100]` answers `[0, 100]` under a `b-w` tag and
        // `[0, 0]` under a `multiband` one, `vips cast` to `uchar` answers
        // `[0, 100]`, and `vips dzsave` writes 0 in the full-resolution tile
        // and 255 in the overview. No route answers the data (issue #952).
        PixelFormat::Uint32(_) => Err(SinkError::EncodeMsg(format!(
            "32-bit unsigned raster ({fmt:?}) cannot be encoded as an image tile; \
             cast to an unsigned 8/16-bit format first"
        ))),
        // Every `image` colour type is unsigned, so this is not a width
        // question and `Int8` is refused alongside `Int32` (issue #516).
        PixelFormat::Int8(_) | PixelFormat::Int16(_) | PixelFormat::Int32(_) => {
            Err(SinkError::EncodeMsg(format!(
                "signed raster ({fmt:?}) cannot be encoded as an image tile; \
                 the image colour types are all unsigned, so cast first"
            )))
        }
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
    .map_err(|e| SinkError::Encode {
        format: "jpeg".into(),
        source: e,
    })?;
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
/// `<key_prefix>/<image_name>…`. The backend is injected via
/// [`ObjectStoreConfig::with_object_store`]; a built-in object-store transport
/// (the planned `object_store` + `tokio` client) is not wired into this build
/// (see the TODO stub in [`ObjectStoreSink::new`]).
///
/// This sink performs **no** retries of its own — if the underlying store's
/// `put` fails, the error is returned verbatim. Callers wanting automatic
/// retry should wrap this sink in [`crate::retry::RetryingSink`] with the
/// appropriate [`crate::retry::RetryPolicy`] and
/// [`crate::retry::FailurePolicy`].
///
/// On the CLI this sink is selected by passing an `s3://…` URI to `--sink`.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-sink)
#[non_exhaustive]
pub struct ObjectStoreSink {
    cfg: ObjectStoreConfig,
    plan: PyramidPlan,
    format: TileFormat,
}

impl std::fmt::Debug for ObjectStoreSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreSink")
            .field("cfg", &self.cfg)
            .field("format", &self.format)
            .finish()
    }
}

impl ObjectStoreSink {
    /// Construct a new sink.
    ///
    /// Requires a backend to have been attached to `cfg` via
    /// [`ObjectStoreConfig::with_object_store`]. The built-in object-store
    /// transport (the planned `object_store` + `tokio` client) is not wired up
    /// in this build (tracking: Phase 3 TODO — add a native transport so
    /// production callers don't need to inject a backend), so calling `new`
    /// without an injected backend returns [`SinkError::Other`] with a message
    /// pointing at the injection API.
    pub fn new(
        cfg: ObjectStoreConfig,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Result<Self, SinkError> {
        if cfg.store.is_none() {
            // Phase 3 TODO: wire up the built-in object-store transport (the
            // planned `object_store` + `tokio` client) so this branch becomes
            // unreachable. For now, all callers (tests and production) must
            // inject a backend via `.with_object_store(...)`.
            return Err(SinkError::Other(
                "ObjectStoreSink: no backend attached. Call \
                 ObjectStoreConfig::with_object_store(Arc<dyn ObjectStore>) \
                 to inject an ObjectStore implementation."
                    .into(),
            ));
        }
        Ok(Self { cfg, plan, format })
    }

    /// Enumerate the object keys stored under this sink's key prefix.
    ///
    /// Delegates to the injected backend's [`ObjectStore::list`], passing the
    /// sink's configured `key_prefix`. The sink no longer hardcodes a refusal:
    /// a listing-capable backend overrides `ObjectStore::list` and returns real
    /// keys, while a write-only backend inherits the trait's defaulted loud
    /// failure ([`SinkError::Unsupported`]). Rather than a silently-empty
    /// `Ok(vec![])` — which a caller diffing server-side keys against a
    /// filesystem reference cannot distinguish from "nothing was uploaded",
    /// making the diff pass falsely or fail spuriously — the default path fails
    /// loud. Callers should treat [`SinkError::Unsupported`] as "this backend
    /// cannot list" rather than "the store is empty".
    pub fn list_objects(&self) -> Result<Vec<String>, SinkError> {
        let store =
            self.cfg.store.as_ref().ok_or_else(|| {
                SinkError::Other("ObjectStoreSink: backend is not configured".into())
            })?;
        // Forward the *trimmed* prefix so a listing observes the same key space
        // the write path produces: every `*_key` helper (and the Zoomify/IIIF
        // branch) normalises `key_prefix` with `trim_matches('/')` before it
        // builds an object key, so a raw `"/run-1/"` would list under a prefix
        // no written key ever carries. Match that normalisation here.
        store.list(self.cfg.key_prefix.trim_matches('/'))
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
            // Zoomify / IIIF tile paths are plan-dependent (cumulative tile
            // numbering, region maths), so the planner is the single source of
            // truth. Prefix its canonical relative tile path with the key
            // prefix and image name, matching the XYZ / Google key shape.
            crate::planner::Layout::Zoomify | crate::planner::Layout::Iiif => {
                let rel = self.plan.tile_path(coord, ext)?;
                let trimmed = self.cfg.key_prefix.trim_matches('/');
                if trimmed.is_empty() {
                    format!("{}/{rel}", self.cfg.image_name)
                } else {
                    format!("{trimmed}/{}/{rel}", self.cfg.image_name)
                }
            }
        };
        Some(key)
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

        // Retries (if any) are handled by wrapping this sink in
        // `RetryingSink` — see module-level docs.
        let store =
            self.cfg.store.as_ref().ok_or_else(|| {
                SinkError::Other("ObjectStoreSink: backend is not configured".into())
            })?;
        store.put(&key, &payload)
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

    #[test]
    fn new_without_backend_mentions_with_object_store() {
        use crate::planner::{Layout, PyramidPlanner};
        let cfg = ObjectStoreConfig::s3("http://localhost:9000", "bucket");
        let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom)
            .expect("planner params are valid")
            .plan();
        let err = ObjectStoreSink::new(cfg, plan, TileFormat::Png).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("with_object_store"),
            "error should point callers to the injection API: {msg}"
        );
    }

    /// In-crate recording backend: captures every `(key, bytes)` handed to
    /// `put` so a test can prove the sink actually uploaded, then confront that
    /// with what `list_objects` reports.
    #[derive(Default)]
    struct RecordingStore {
        puts: std::sync::Mutex<Vec<(String, Vec<u8>)>>,
    }

    impl RecordingStore {
        fn keys(&self) -> Vec<String> {
            self.puts
                .lock()
                .unwrap()
                .iter()
                .map(|(k, _)| k.clone())
                .collect()
        }
    }

    impl ObjectStore for RecordingStore {
        fn put(&self, key: &str, bytes: &[u8]) -> Result<(), SinkError> {
            self.puts
                .lock()
                .unwrap()
                .push((key.to_string(), bytes.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn list_objects_is_unsupported_even_after_writes() {
        // Regression guard for the silent-empty-list footgun (issues #379,
        // #380, #383, #384): the earlier test asserted `list_objects() == []`
        // over a sink that had never uploaded anything, a property a *correct*
        // LIST implementation would also satisfy — so it locked in nothing.
        //
        // Here we upload a real tile through the sink (the recording backend
        // captures it), then assert the sink cannot enumerate it. Because the
        // `ObjectStore` trait exposes only `put`, `list_objects` fails loud
        // with `SinkError::Unsupported` rather than masquerading the backend's
        // populated state as an empty `Ok(vec![])`.
        use crate::planner::{Layout, PyramidPlanner};
        use crate::sink::TileSink;

        let store = Arc::new(RecordingStore::default());
        let cfg = ObjectStoreConfig::s3("http://localhost:9000", "bucket")
            .with_object_store(store.clone());
        let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom)
            .expect("planner params are valid")
            .plan();
        let sink = ObjectStoreSink::new(cfg, plan, TileFormat::Png)
            .expect("sink constructs once a backend is injected");

        let tile = Tile {
            coord: TileCoord::new(0, 0, 0),
            raster: Raster::zeroed(256, 256, PixelFormat::Rgb8).unwrap(),
            blank: false,
        };
        sink.write_tile(&tile).expect("tile upload succeeds");

        // The backend really recorded the upload...
        assert_eq!(
            store.keys().len(),
            1,
            "backend must have recorded exactly one put"
        );

        // ...yet the sink still cannot enumerate it: the stub fails loud
        // instead of returning a misleading empty list.
        let err = sink
            .list_objects()
            .expect_err("list_objects must not report success while unimplemented");
        assert!(
            matches!(err, SinkError::Unsupported(_)),
            "expected SinkError::Unsupported, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("list_objects") && msg.contains("LIST"),
            "error must name the operation and the missing transport: {msg}"
        );
    }

    /// A listing-capable backend: records every `put` and *overrides*
    /// [`ObjectStore::list`] to return the keys it holds (filtered by prefix).
    /// Proves the sink delegates enumeration to the backend rather than
    /// hardcoding a refusal.
    #[derive(Default)]
    struct ListingStore {
        puts: std::sync::Mutex<Vec<String>>,
        last_prefix: std::sync::Mutex<Option<String>>,
    }

    impl ObjectStore for ListingStore {
        fn put(&self, key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
            self.puts.lock().unwrap().push(key.to_string());
            Ok(())
        }

        fn list(&self, prefix: &str) -> Result<Vec<String>, SinkError> {
            *self.last_prefix.lock().unwrap() = Some(prefix.to_string());
            Ok(self
                .puts
                .lock()
                .unwrap()
                .iter()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect())
        }
    }

    #[test]
    fn list_objects_delegates_to_overriding_backend() {
        // A backend that overrides `ObjectStore::list` makes `list_objects`
        // return real keys — proving the sink delegates to the backend's
        // `list` instead of hardcoding `SinkError::Unsupported`, and that the
        // sink's `key_prefix` is forwarded as the `list` argument.
        use crate::planner::{Layout, PyramidPlanner};
        use crate::sink::TileSink;

        let store = Arc::new(ListingStore::default());
        // Wrap the prefix in slashes: the write path trims them via
        // `trim_matches('/')` before building keys, so `list_objects` must
        // forward the same trimmed prefix ("run-1"), not the raw "/run-1/".
        let cfg = ObjectStoreConfig::s3("http://localhost:9000", "bucket")
            .with_key_prefix("/run-1/")
            .with_image_name("deleg")
            .with_object_store(store.clone());
        let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom)
            .expect("planner params are valid")
            .plan();
        let sink = ObjectStoreSink::new(cfg, plan, TileFormat::Png)
            .expect("sink constructs once a backend is injected");

        let tile = Tile {
            coord: TileCoord::new(0, 0, 0),
            raster: Raster::zeroed(256, 256, PixelFormat::Rgb8).unwrap(),
            blank: false,
        };
        sink.write_tile(&tile).expect("tile upload succeeds");

        let listed = sink
            .list_objects()
            .expect("list_objects delegates to the overriding backend");
        assert_eq!(
            listed,
            vec!["run-1/deleg_files/0/0_0.png".to_string()],
            "list_objects must return the keys the backend holds"
        );

        // The sink forwarded its configured key_prefix to the backend's `list`,
        // *trimmed* to match the write path — the backend sees "run-1", not
        // the raw "/run-1/".
        assert_eq!(
            store.last_prefix.lock().unwrap().as_deref(),
            Some("run-1"),
            "list_objects must delegate with the write-path-trimmed key_prefix"
        );
    }

    /// The trait's *defaulted* `list` refuses loud on a write-only backend that
    /// does not override it — the behaviour the sink used to hardcode now lives
    /// on the trait.
    #[test]
    fn default_trait_list_refuses_loud() {
        let store = RecordingStore::default();
        let err = ObjectStore::list(&store, "any-prefix")
            .expect_err("defaulted ObjectStore::list must refuse");
        assert!(
            matches!(err, SinkError::Unsupported(_)),
            "expected SinkError::Unsupported, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("list_objects") && msg.contains("LIST"),
            "default list error must name the operation and the missing transport: {msg}"
        );
    }
}
