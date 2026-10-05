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
//! ```
//! use std::sync::Arc;
//!
//! use libviprs::planner::{Layout, PyramidPlanner};
//! use libviprs::retry::{RetryPolicy, RetryingSink};
//! use libviprs::sink::{SinkError, TileFormat};
//! use libviprs::sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};
//!
//! /// The injected backend the sink needs. A real client puts bytes on the
//! /// wire; this one drops them, which is all the example needs.
//! struct NullStore;
//! impl ObjectStore for NullStore {
//!     fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
//!         Ok(())
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let plan = PyramidPlanner::new(1024, 768, 256, 0, Layout::DeepZoom)?.plan();
//! let cfg = ObjectStoreConfig::s3("https://example.invalid", "tiles")
//!     .with_object_store(Arc::new(NullStore));
//!
//! let sink = RetryingSink::new(
//!     ObjectStoreSink::new(cfg, plan, TileFormat::Png)?,
//!     RetryPolicy::default(),
//! );
//! # let _ = sink;
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex};

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::sink::{
    BLANK_TILE_MARKER, SinkError, Tile, TileFormat, TileSink, background_from, encode_jpeg,
    encode_png,
};

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

    /// Read exactly `len` bytes of the object at `key`, starting at `offset`.
    ///
    /// This is the read half of the transport seam (issue #1121). A PMTiles
    /// archive is read entirely through ranged fetches, so this is the one
    /// method a backend has to implement to be handed to
    /// [`ObjectStoreRangeReader`](crate::pmtiles::ObjectStoreRangeReader) and
    /// have an archive open over it, with no other change anywhere. Overriding
    /// [`ObjectStore::size`] as well is optional, and what it buys is the
    /// reader's section bounds checks.
    ///
    /// **Defaulted to a loud refusal**, the same shape [`ObjectStore::list`]
    /// uses and for the same reason: a write-only backend should say it cannot
    /// read rather than invent an answer.
    ///
    /// **Returning more or fewer bytes than `len` is a contract violation**,
    /// and the bridge checks it rather than trusting it. Both directions are
    /// real: a server that ignores `Range` answers 200 with the whole object,
    /// and a connection that drops mid-body answers short. Implementations
    /// should still refuse a range that runs past the end of the object rather
    /// than clamping it, because a clamped range is a short read wearing a
    /// success.
    fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        let _ = (key, offset, len);
        Err(SinkError::Unsupported(
            "get_range: this ObjectStore backend does not implement a ranged \
             READ (ObjectStore::get_range is defaulted to refuse). A \
             read-capable backend overrides it; write-only backends inherit \
             this loud failure."
                .into(),
        ))
    }

    /// The total size of the object at `key`, when the backend knows it
    /// cheaply.
    ///
    /// A reader uses it to bounds-check a header's claimed section offsets
    /// against the real object before believing any of them.
    ///
    /// **Defaulted to a refusal, deliberately not to `Ok(None)`.** That is the
    /// one choice in this module worth reading twice. `None` is a *legal*
    /// answer from [`RangeReader::size`](crate::pmtiles::RangeReader::size),
    /// because a streaming transport genuinely may not know, and it disables
    /// the four `SectionOutOfBounds` checks in `Reader::try_new`. So a default
    /// of `Ok(None)` would hand every backend that never thought about `size`
    /// a reader with those checks quietly switched off, and nothing anywhere
    /// would report a problem. A refusal makes a backend say out loud that it
    /// has not implemented the HEAD, and the bridge is what decides that a
    /// refusal (and *only* a refusal, never a transient failure) means
    /// "unknown".
    ///
    /// A backend that can answer returns `Ok(Some(bytes))`. `Ok(None)` is for
    /// a backend that implemented this, asked, and genuinely got no answer.
    fn size(&self, key: &str) -> Result<Option<u64>, SinkError> {
        let _ = key;
        Err(SinkError::Unsupported(
            "size: this ObjectStore backend does not implement a HEAD \
             (ObjectStore::size is defaulted to refuse). A read-capable \
             backend overrides it; write-only backends inherit this loud \
             failure. Do not default this to Ok(None): None is a legal \
             RangeReader answer and it costs the reader its section bounds \
             checks."
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// DirectoryObjectStore
// ---------------------------------------------------------------------------

/// An [`ObjectStore`] that keeps each object as a file under a root
/// directory, so object `a/b.png` is the file `root/a/b.png`.
///
/// It is the transport-free stand-in for a bucket: for tests, for harnesses
/// that want to look at what a run uploaded, and for builds with no network
/// client (libviprs-cli's `--sink s3://` writes through one). It implements
/// the whole trait, so [`ObjectStoreSink`] writes through it,
/// [`ObjectStoreSink::list_objects`] lists it, and
/// [`ObjectStoreRangeReader`](crate::pmtiles::ObjectStoreRangeReader) reads an
/// archive back out of it.
///
/// ```
/// use libviprs::sink_object_store::{DirectoryObjectStore, ObjectStore};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = std::env::temp_dir().join(format!("libviprs-doc-dirstore-{}", std::process::id()));
/// // s3://tiles/run-1/... lands under <dir>/tiles/run-1/...
/// let store = DirectoryObjectStore::for_bucket(&dir, "tiles")?;
/// store.put("run-1/image_files/0/0_0.png", b"...")?;
/// assert_eq!(store.list("run-1/")?, vec!["run-1/image_files/0/0_0.png"]);
/// assert_eq!(store.size("run-1/image_files/0/0_0.png")?, Some(3));
/// # std::fs::remove_dir_all(&dir)?;
/// # Ok(())
/// # }
/// ```
///
/// * **Keys** are `/`-separated relative paths made only of ordinary
///   segments. An empty key, a leading `/`, an empty segment (`a//b`, `a/`),
///   a `.` or `..` segment, a backslash or a NUL is refused before the
///   filesystem is touched, and so is anything the platform's own path
///   parser would read as other than one plain name per segment (a Windows
///   drive prefix, say). No key can name a file outside the root.
/// * **Symlinks** under the root are not followed. A read, size or write
///   whose path crosses one is refused, and listing skips them, so a link
///   planted inside the root is not a way out of it.
/// * **Writes** are atomic. Each object is staged into a uniquely named
///   sibling, flushed, and renamed over its key, so a reader sees the old
///   object or the new one and never a torn one. A write that fails removes
///   its staging file, and one that a dead process left behind is not an
///   object: it is never listed, and no key can name it.
/// * **Listing** walks the root and returns every object key that starts
///   with the prefix, sorted, as plain string prefix matching, the way a
///   bucket listing does. A root that does not exist yet lists as empty.
/// * **A missing object** is an error from [`ObjectStore::get_range`] and
///   [`ObjectStore::size`] alike. `size` never answers `Ok(None)`, because a
///   file's length is always known and `None` would switch off a reader's
///   bounds checks.
///
/// A file and a directory cannot share a name, so `a` and `a/b` cannot both
/// be objects here, where a real bucket would hold both. The second `put`
/// fails rather than replacing the first.
#[derive(Debug, Clone)]
pub struct DirectoryObjectStore {
    root: std::path::PathBuf,
}

/// The suffix every staging file ends with. A key segment may not end with
/// it, which is what keeps a staged half-object from ever being addressed as
/// an object.
const STAGING_SUFFIX: &str = ".libviprs-part";

/// Distinguishes the staging files of concurrent writers in one process.
static STAGING_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `segment` is one plain path segment that names the same thing on
/// every platform: no separators, no NUL, not `.` or `..`, and nothing the
/// platform's path parser reads as a prefix or a root.
fn is_plain_segment(segment: &str) -> bool {
    if segment.is_empty() || segment.contains(['/', '\\', '\0']) {
        return false;
    }
    let mut parts = std::path::Path::new(segment).components();
    matches!(
        (parts.next(), parts.next()),
        (Some(std::path::Component::Normal(name)), None) if name == segment
    )
}

impl DirectoryObjectStore {
    /// A store whose objects live under `root`. Nothing is created until the
    /// first write.
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// A store for `bucket` under `root`, so object `k` is `root/bucket/k`.
    ///
    /// `bucket` has to be one plain name. An empty name, `.`, `..`, anything
    /// with a `/`, a backslash or a NUL in it, or an absolute path is refused
    /// with [`SinkError::Other`], since each would put the bucket beside the
    /// root or several levels into it rather than directly under it.
    pub fn for_bucket(root: impl AsRef<std::path::Path>, bucket: &str) -> Result<Self, SinkError> {
        if !is_plain_segment(bucket) {
            return Err(SinkError::Other(format!(
                "{bucket:?} is not a bucket name: a bucket is one plain name, \
                 such as \"tiles\", and the directory store refuses anything \
                 that would not sit directly under its root"
            )));
        }
        Ok(Self::new(root.as_ref().join(bucket)))
    }

    /// The directory objects are stored under.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// The file that holds `key`, after refusing a key that is not a plain
    /// relative path and a path that crosses a symlink under the root.
    fn path_of(&self, key: &str) -> Result<std::path::PathBuf, SinkError> {
        let plain = !key.is_empty()
            && key
                .split('/')
                .all(|seg| is_plain_segment(seg) && !seg.ends_with(STAGING_SUFFIX));
        if !plain {
            return Err(SinkError::Other(format!(
                "object key {key:?} is not a plain relative path; the directory \
                 store refuses it rather than resolve it outside its root"
            )));
        }
        let mut walked = self.root.clone();
        for seg in key.split('/') {
            walked.push(seg);
            match std::fs::symlink_metadata(&walked) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err(SinkError::Other(format!(
                        "object key {key:?} crosses the symlink {}; the directory \
                         store does not follow links under its root",
                        walked.display()
                    )));
                }
                Ok(_) => {}
                // Nothing further down exists either, so there is no link
                // left to cross.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(e) => return Err(SinkError::Io(e)),
            }
        }
        Ok(self.root.join(key))
    }

    /// The regular file that holds the existing object `key`.
    fn existing_object(&self, key: &str) -> Result<(std::path::PathBuf, u64), SinkError> {
        let path = self.path_of(key)?;
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() {
            return Err(SinkError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{key:?} is a directory under the store, not an object"),
            )));
        }
        Ok((path, meta.len()))
    }

    /// Stage `key`'s new contents through `write`, flush them, and rename the
    /// staging file over the key. Any failure removes the staging file and
    /// leaves whatever object was there before untouched.
    fn put_with(
        &self,
        key: &str,
        write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
    ) -> Result<(), SinkError> {
        let path = self.path_of(key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Creating the directories cannot have planted a link, but a racing
        // process could have; look again now that the path exists.
        let path = self.path_of(key)?;
        let seq = STAGING_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut staging = path.clone().into_os_string();
        staging.push(format!(".{}.{seq}{STAGING_SUFFIX}", std::process::id()));
        let staging = std::path::PathBuf::from(staging);

        let staged = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)?;
            write(&mut file)?;
            file.sync_all()
        })();
        if let Err(e) = staged {
            let _ = std::fs::remove_file(&staging);
            return Err(SinkError::Io(e));
        }
        if let Err(e) = std::fs::rename(&staging, &path) {
            let _ = std::fs::remove_file(&staging);
            return Err(SinkError::Io(e));
        }
        Ok(())
    }

    /// Every object key under `dir`, relative to the root and `/`-separated.
    /// Symlinks and staging files are not objects and are skipped.
    fn walk(&self, dir: &std::path::Path, out: &mut Vec<String>) -> Result<(), SinkError> {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(SinkError::Io(e)),
        };
        for entry in entries {
            let entry = entry?;
            // `DirEntry::file_type` does not follow links, which is what
            // keeps a planted link from pulling outside files into a listing.
            let kind = entry.file_type()?;
            let path = entry.path();
            if kind.is_dir() {
                self.walk(&path, out)?;
            } else if kind.is_file() {
                let Ok(rel) = path.strip_prefix(&self.root) else {
                    continue;
                };
                let mut segments = Vec::new();
                for part in rel.components() {
                    match part.as_os_str().to_str() {
                        Some(seg) => segments.push(seg),
                        // A file whose name is not UTF-8 has no key that
                        // could name it, so it is not an object.
                        None => break,
                    }
                }
                if segments.len() != rel.components().count()
                    || segments.last().is_some_and(|s| s.ends_with(STAGING_SUFFIX))
                {
                    continue;
                }
                out.push(segments.join("/"));
            }
        }
        Ok(())
    }
}

impl ObjectStore for DirectoryObjectStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), SinkError> {
        use std::io::Write;
        self.put_with(key, |file| file.write_all(bytes))
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, SinkError> {
        let mut keys = Vec::new();
        self.walk(&self.root, &mut keys)?;
        keys.retain(|k| k.starts_with(prefix));
        keys.sort();
        Ok(keys)
    }

    fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        use std::io::{Read, Seek, SeekFrom};
        let (path, size) = self.existing_object(key)?;
        let in_bounds = u64::try_from(len)
            .ok()
            .and_then(|len| offset.checked_add(len))
            .is_some_and(|end| end <= size);
        if !in_bounds {
            return Err(SinkError::Other(format!(
                "range {offset}+{len} of {key:?} runs past the end of the \
                 {size}-byte object"
            )));
        }
        let mut file = std::fs::File::open(&path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len).map_err(|_| {
            SinkError::Other(format!("cannot allocate a {len}-byte range of {key:?}"))
        })?;
        file.take(len as u64).read_to_end(&mut buf)?;
        if buf.len() != len {
            // The file shrank between the size check and the read.
            return Err(SinkError::Other(format!(
                "range {offset}+{len} of {key:?} came back {} bytes short",
                len - buf.len()
            )));
        }
        Ok(buf)
    }

    fn size(&self, key: &str) -> Result<Option<u64>, SinkError> {
        let (_, size) = self.existing_object(key)?;
        Ok(Some(size))
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

/// Encode one tile, flattening any alpha onto `background` for the formats
/// that cannot carry it.
///
/// Every arm is `crate::sink`'s own encoder now. The JPEG one used to be a
/// local copy of the same `image`-crate wrapper, kept so this module would not
/// need `sink`'s helpers to be `pub(crate)`; they are, and the copy went out
/// of step the moment `sink`'s grew the alpha flattening of issue #1133.
fn encode_tile(
    raster: &Raster,
    format: TileFormat,
    background: [u8; 3],
) -> Result<Vec<u8>, SinkError> {
    match format {
        TileFormat::Raw => Ok(raster.data().to_vec()),
        TileFormat::Png => encode_png(raster),
        TileFormat::Jpeg { quality } => encode_jpeg(raster, quality, background),
        TileFormat::Webp => crate::sink::encode_webp(raster),
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
    /// Captured by [`TileSink::record_engine_config`] before the tile loop
    /// starts, and read for one field: the background a JPEG tile's
    /// transparent pixels land on (issue #1133). This sink uploads tiles and
    /// keeps no manifest, so it had nowhere to read `background_rgb` from and
    /// a run with `--background` would have honoured it in the padding and
    /// ignored it in the pixels.
    engine_config: Mutex<Option<crate::engine::EngineConfig>>,
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
        Ok(Self {
            cfg,
            plan,
            format,
            engine_config: Mutex::new(None),
        })
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
            encode_tile(
                &tile.raster,
                self.format,
                background_from(&self.engine_config),
            )?
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

    /// Keep the run's configuration for the one thing this sink reads out of
    /// it: the background a JPEG tile's alpha is flattened onto.
    fn record_engine_config(&self, config: &crate::engine::EngineConfig) {
        *crate::poison::recover(&self.engine_config) = Some(config.clone());
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
    // Only the refusal cells below reach the mapping directly now that this
    // module's copy of the JPEG wrapper is gone (issue #1133).
    use crate::pixel::PixelFormat;
    use crate::sink::color_type_for_format;

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

    /// This module had no direct test of the uint/float refusal before
    /// issue #969: `encode.rs` and `sink.rs` each had one
    /// (`the_encoders_refuse_the_uint_carrier_by_name`,
    /// `the_tile_sinks_refuse_the_uint_carrier_by_name`), and this module had
    /// none, so a mutation landing only here had nothing to catch it. Mirrors
    /// those two so all three routes onto
    /// [`crate::pixel::image_color_type`] are directly covered, even though
    /// this module now calls [`crate::sink::color_type_for_format`] directly
    /// rather than keeping its own copy of the wrapper (issue #940).
    #[test]
    fn the_object_store_sink_refuses_the_uint_carrier_by_name() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let u = PixelFormat::Uint32(n(1));
        let msg = color_type_for_format(u)
            .expect_err("a uint raster is not an image tile")
            .to_string();
        assert!(
            msg.contains("32-bit unsigned") && msg.contains("Uint32"),
            "the refusal does not name the carrier: {msg}"
        );
        let f = PixelFormat::FloatF32(n(1));
        let fmsg = color_type_for_format(f)
            .expect_err("a float raster is not an image tile")
            .to_string();
        assert!(fmsg.contains("float"), "{fmsg}");
        assert_ne!(msg, fmsg);
    }

    // -----------------------------------------------------------------------
    // DirectoryObjectStore (issue #1171)
    // -----------------------------------------------------------------------

    /// Every file under `dir`, relative to it, `/`-separated and sorted. The
    /// cells use it to check what actually landed on disk, staging files
    /// included, without going through the store under test.
    fn files_under(dir: &std::path::Path) -> Vec<String> {
        fn walk(base: &std::path::Path, dir: &std::path::Path, out: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(base, &path, out);
                } else {
                    let rel = path.strip_prefix(base).unwrap();
                    let parts: Vec<String> = rel
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect();
                    out.push(parts.join("/"));
                }
            }
        }
        let mut out = Vec::new();
        walk(dir, dir, &mut out);
        out.sort();
        out
    }

    /// Put, list under a prefix, ranged read, size and overwrite all go
    /// through the directory, with object `a/b` landing at `root/a/b`.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_round_trips_objects() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bucket");
        let store = DirectoryObjectStore::new(&root);
        assert_eq!(store.root(), root.as_path());

        store.put("run/a/0_0.png", b"hello world").expect("put");
        store.put("run/b.png", b"xy").expect("put");
        store.put("runner/c.png", b"z").expect("put");
        store
            .put("other/d.png", b"")
            .expect("an empty object is an object");
        assert_eq!(
            std::fs::read(root.join("run/a/0_0.png")).unwrap(),
            b"hello world"
        );

        // A bucket listing is a plain string prefix match, so `run` also
        // matches `runner/`, and the keys come back sorted.
        assert_eq!(
            store.list("run").expect("list"),
            vec!["run/a/0_0.png", "run/b.png", "runner/c.png"]
        );
        assert_eq!(
            store.list("run/").expect("list"),
            vec!["run/a/0_0.png", "run/b.png"]
        );
        assert_eq!(store.list("").expect("list everything").len(), 4);
        assert!(store.list("nothing/").expect("list").is_empty());

        assert_eq!(store.get_range("run/a/0_0.png", 6, 5).unwrap(), b"world");
        assert_eq!(store.get_range("run/a/0_0.png", 11, 0).unwrap(), b"");
        assert_eq!(store.size("run/a/0_0.png").unwrap(), Some(11));
        assert_eq!(store.size("other/d.png").unwrap(), Some(0));

        // A range past the end is refused, not clamped (the trait contract).
        assert!(store.get_range("run/b.png", 1, 5).is_err());
        // A missing object is an error, not an empty object and not "size
        // unknown", which would switch off a reader's bounds checks.
        assert!(store.get_range("run/missing.png", 0, 1).is_err());
        assert!(store.size("run/missing.png").is_err());
        // A directory is not an object.
        assert!(store.size("run/a").is_err());
        assert!(store.get_range("run/a", 0, 0).is_err());

        store.put("run/b.png", b"replaced").expect("overwrite");
        assert_eq!(store.get_range("run/b.png", 0, 8).unwrap(), b"replaced");
        assert_eq!(store.size("run/b.png").unwrap(), Some(8));

        // A root that does not exist yet lists as empty rather than failing,
        // the way an empty bucket does.
        let fresh = DirectoryObjectStore::new(dir.path().join("not-yet"));
        assert!(fresh.list("").expect("list").is_empty());
    }

    /// A key that would leave the root, that names nothing, or that the
    /// filesystem would read differently from a bucket is refused by every
    /// operation before anything touches the disk.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_refuses_keys_that_escape_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bucket");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(dir.path().join("secret"), b"outside").unwrap();
        let store = DirectoryObjectStore::new(&root);
        for key in [
            "",
            "/",
            "..",
            ".",
            "../secret",
            "a/../../secret",
            "a/..",
            "/etc/passwd",
            "a//b",
            "a/",
            "./a",
            "a/./b",
            "a\\..\\..\\secret",
            "..\\secret",
            "a\0b",
        ] {
            assert!(
                store.put(key, b"no").is_err(),
                "put {key:?} must be refused"
            );
            assert!(
                store.get_range(key, 0, 1).is_err(),
                "get_range {key:?} must be refused"
            );
            assert!(store.size(key).is_err(), "size {key:?} must be refused");
        }
        assert_eq!(
            std::fs::read(dir.path().join("secret")).unwrap(),
            b"outside",
            "the file beside the root is untouched"
        );
        assert!(
            files_under(&root).is_empty(),
            "nothing was written inside the root either: {:?}",
            files_under(&root)
        );
    }

    /// A symlink planted under the root does not become a way out of it:
    /// reads, sizes and writes through it are refused and listing skips it.
    #[cfg(unix)]
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_does_not_follow_symlinks_out_of_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bucket");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret"), b"outside").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        std::os::unix::fs::symlink(outside.join("secret"), root.join("file-link")).unwrap();

        let store = DirectoryObjectStore::new(&root);
        assert!(store.get_range("link/secret", 0, 1).is_err());
        assert!(store.size("link/secret").is_err());
        assert!(store.get_range("file-link", 0, 1).is_err());
        assert!(store.size("file-link").is_err());
        assert!(store.put("link/planted", b"no").is_err());
        assert!(!outside.join("planted").exists(), "nothing landed outside");
        assert!(
            store.list("").expect("list").is_empty(),
            "a symlink is not an object"
        );
    }

    /// A bucket is one plain name under the root; anything that would put it
    /// beside the root, or several levels down, is refused.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_bucket_is_one_plain_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = DirectoryObjectStore::for_bucket(dir.path(), "tiles").expect("plain name");
        assert_eq!(store.root(), dir.path().join("tiles").as_path());
        store.put("run/0.png", b"t").unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("tiles/run/0.png")).unwrap(),
            b"t"
        );
        assert!(DirectoryObjectStore::for_bucket(dir.path(), "my.bucket-1").is_ok());

        for bucket in [
            "", ".", "..", "a/b", "/abs", "../up", "a\\b", "x\0y", "tiles/",
        ] {
            assert!(
                DirectoryObjectStore::for_bucket(dir.path(), bucket).is_err(),
                "bucket {bucket:?} must be refused"
            );
        }
    }

    /// A write that dies partway leaves the previous object whole, cleans up
    /// after itself, and never shows the half-written bytes under the key.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_interrupted_write_never_leaves_a_torn_object() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bucket");
        let store = DirectoryObjectStore::new(&root);
        store
            .put("run/0.png", b"the whole old object")
            .expect("put");

        let err = store
            .put_with("run/0.png", |f| {
                f.write_all(b"half of the new")?;
                Err(std::io::Error::other("the writer died here"))
            })
            .expect_err("a failed write is an error");
        assert!(err.to_string().contains("the writer died here"), "{err}");
        assert_eq!(
            store.get_range("run/0.png", 0, 20).unwrap(),
            b"the whole old object"
        );
        assert_eq!(store.size("run/0.png").unwrap(), Some(20));
        assert_eq!(
            files_under(&root),
            vec!["run/0.png"],
            "the staging file went with the failed write"
        );

        // A first write that dies leaves no object at all, not an empty one.
        assert!(
            store
                .put_with("run/1.png", |f| {
                    f.write_all(b"partial")?;
                    Err(std::io::Error::other("killed"))
                })
                .is_err()
        );
        assert!(store.size("run/1.png").is_err());
        assert_eq!(store.list("").unwrap(), vec!["run/0.png"]);
    }

    /// What a killed process leaves behind (a staging file it never renamed)
    /// is not an object: it is not listed, and a key cannot name one, so a
    /// caller can never read a staged half-object back as if it were real.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn directory_store_never_lists_a_dead_writers_staging_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bucket");
        let store = DirectoryObjectStore::new(&root);
        store.put("run/0.png", b"old").expect("put");

        // Stop a write between the staged bytes and the rename by panicking
        // inside it, which is the nearest a test gets to the process dying
        // there, and keep the staging file it leaves.
        let staged = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = store.put_with("run/0.png", |f| {
                f.write_all(b"torn")?;
                panic!("the process died between write and rename");
            });
        }));
        assert!(staged.is_err());
        let on_disk = files_under(&root);
        let leftover: Vec<&String> = on_disk.iter().filter(|p| *p != "run/0.png").collect();
        assert_eq!(
            leftover.len(),
            1,
            "the dead write left its staging file: {on_disk:?}"
        );

        assert_eq!(store.list("").unwrap(), vec!["run/0.png"]);
        assert_eq!(store.get_range("run/0.png", 0, 3).unwrap(), b"old");
        assert!(
            store.size(leftover[0]).is_err() && store.put(leftover[0], b"x").is_err(),
            "a staging name is not a key"
        );
    }

    /// The sink writes a whole pyramid through the directory store, the
    /// store holds exactly the objects the sink put, byte for byte, and the
    /// sink's own listing reads them back.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn the_sink_round_trips_a_pyramid_through_the_directory_store() {
        use crate::planner::{Layout, PyramidPlanner};
        let plan = PyramidPlanner::new(96, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let mut pixels = vec![0u8; 96 * 64 * 3];
        for (i, p) in pixels.iter_mut().enumerate() {
            *p = (i * 7 % 251) as u8;
        }
        let src = Raster::new(96, 64, PixelFormat::Rgb8, pixels).unwrap();

        let run = |store: Arc<dyn ObjectStore>| {
            let cfg = ObjectStoreConfig::s3("file://unused", "tiles")
                .with_key_prefix("runs/1")
                .with_object_store(store);
            let sink = ObjectStoreSink::new(cfg, plan.clone(), TileFormat::Png).unwrap();
            crate::EngineBuilder::new(&src, plan.clone(), &sink)
                .run()
                .unwrap();
            sink
        };

        let reference = Arc::new(RecordingStore::default());
        run(reference.clone());
        let mut expected = reference.puts.lock().unwrap().clone();
        expected.sort();
        assert!(!expected.is_empty());

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(DirectoryObjectStore::for_bucket(dir.path(), "tiles").unwrap());
        let sink = run(store.clone());

        let keys: Vec<String> = expected.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(
            sink.list_objects().expect("the sink lists through it"),
            keys
        );
        for (key, bytes) in &expected {
            let len = usize::try_from(store.size(key).unwrap().unwrap()).unwrap();
            assert_eq!(&store.get_range(key, 0, len).unwrap(), bytes, "{key}");
        }
    }
}
