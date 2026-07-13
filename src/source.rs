use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use image::{GenericImageView, ImageReader, Limits};
use thiserror::Error;

use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// Default entry cap for the process-global load cache: at most this many
/// distinct decoded rasters are retained before the least-recently-used is
/// evicted. Mirrors libvips' operation-cache size default
/// (`vips_cache_set_max`, ~100 live operations).
const DEFAULT_LOAD_CACHE_MAX_ENTRIES: usize = 100;

/// Default byte ceiling for the process-global load cache: once the total
/// decoded-pixel bytes held exceed this, least-recently-used entries are
/// evicted until it fits. Mirrors libvips' bounded max-memory cache ceiling
/// (`vips_cache_set_max_mem`), here ~100 MB.
const DEFAULT_LOAD_CACHE_MAX_BYTES: usize = 100 * 1024 * 1024;

/// A single cached decode: the shared raster plus its byte footprint and a
/// recency stamp used to pick the least-recently-used victim on eviction.
struct CacheEntry {
    /// The decoded raster, shared behind an [`Arc`] so a cache hit clones a
    /// cheap handle under the lock and defers the deep pixel copy until
    /// after the lock is released.
    raster: Arc<Raster>,
    /// The raster's pixel-buffer length, summed into
    /// [`LoadCache::total_bytes`] for the byte-cap.
    bytes: usize,
    /// Monotonic access stamp: the highest value is the most-recently-used
    /// entry, the lowest is the eviction victim.
    last_used: u64,
}

/// A bounded, least-recently-used cache of decoded rasters keyed by
/// canonical path.
///
/// libvips keeps a *bounded*, LRU-evicted operation cache (governed by
/// `vips_cache_set_max` and `vips_cache_set_max_mem`): repeatedly opening
/// the same file hands back the already-decoded image, but the cache never
/// grows without limit. A binding built on that library inherits both the
/// caching and its bound. This mirrors that with a process-global map that
/// evicts the least-recently-used entry whenever an insert pushes it past
/// either the entry-count cap or the total-bytes cap.
///
/// Recency is tracked by a monotonic counter stamped on every insert and
/// every hit; the victim is the entry with the smallest stamp. The entry
/// count is small (100 by default), so scanning for the minimum on the rare
/// eviction is cheaper than maintaining an intrusive ordering.
struct LoadCache {
    /// Canonical-path → cached decode.
    map: HashMap<PathBuf, CacheEntry>,
    /// Running sum of every entry's `bytes`, checked against `max_bytes`.
    total_bytes: usize,
    /// Source of monotonically increasing recency stamps.
    tick: u64,
    /// Maximum number of resident entries before LRU eviction.
    max_entries: usize,
    /// Maximum total pixel bytes before LRU eviction.
    max_bytes: usize,
}

impl LoadCache {
    /// A cache bounded by `max_entries` resident decodes and `max_bytes` of
    /// total pixel data.
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            total_bytes: 0,
            tick: 0,
            max_entries,
            max_bytes,
        }
    }

    /// Look up `key`, bumping its recency on a hit and returning the shared
    /// raster handle (no pixel copy).
    fn get(&mut self, key: &Path) -> Option<Arc<Raster>> {
        self.tick += 1;
        let stamp = self.tick;
        let entry = self.map.get_mut(key)?;
        entry.last_used = stamp;
        Some(Arc::clone(&entry.raster))
    }

    /// Insert `raster` under `key`, replacing any existing entry, then evict
    /// least-recently-used entries until both caps hold. Returns the shared
    /// handle now resident (the one just inserted).
    fn insert(&mut self, key: PathBuf, raster: Arc<Raster>) -> Arc<Raster> {
        let bytes = raster.data().len();
        self.tick += 1;
        let stamp = self.tick;
        let handle = Arc::clone(&raster);
        if let Some(old) = self.map.insert(
            key,
            CacheEntry {
                raster,
                bytes,
                last_used: stamp,
            },
        ) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        self.total_bytes += bytes;
        self.evict_to_caps();
        handle
    }

    /// Insert `raster` under `key` only if the key is absent, returning the
    /// resident handle (the existing entry on a race, otherwise the new one).
    fn get_or_insert(&mut self, key: PathBuf, raster: Arc<Raster>) -> Arc<Raster> {
        if let Some(existing) = self.get(&key) {
            return existing;
        }
        self.insert(key, raster)
    }

    /// Drop the entry stored under `key`, if any.
    fn remove(&mut self, key: &Path) {
        if let Some(old) = self.map.remove(key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
    }

    /// Drop every entry.
    fn clear(&mut self) {
        self.map.clear();
        self.total_bytes = 0;
    }

    /// Set the entry cap, evicting immediately if the cache now exceeds it.
    fn set_max_entries(&mut self, max: usize) {
        self.max_entries = max;
        self.evict_to_caps();
    }

    /// Set the byte cap, evicting immediately if the cache now exceeds it.
    fn set_max_bytes(&mut self, max: usize) {
        self.max_bytes = max;
        self.evict_to_caps();
    }

    /// Evict the least-recently-used entry repeatedly until both the
    /// entry-count and total-bytes caps hold (or the cache is empty).
    fn evict_to_caps(&mut self) {
        while !self.map.is_empty()
            && (self.map.len() > self.max_entries || self.total_bytes > self.max_bytes)
        {
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            match victim {
                Some(key) => {
                    if let Some(old) = self.map.remove(&key) {
                        self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
                    }
                }
                None => break,
            }
        }
    }
}

/// The process-global load cache backing [`decode_file`],
/// [`decode_file_with_options`], and [`Raster::invalidate`]. See
/// [`LoadCache`] for the bounded-LRU contract and its libvips-binding
/// rationale.
static LOAD_CACHE: LazyLock<Mutex<LoadCache>> = LazyLock::new(|| {
    Mutex::new(LoadCache::new(
        DEFAULT_LOAD_CACHE_MAX_ENTRIES,
        DEFAULT_LOAD_CACHE_MAX_BYTES,
    ))
});

/// Lock the global load cache, recovering in place from a poisoned lock.
///
/// The cached rasters are plain data with no broken invariant a panicking
/// thread could have left behind, so recovering the guard is always safe
/// and no path here can panic on lock poisoning.
fn lock_cache() -> MutexGuard<'static, LoadCache> {
    LOAD_CACHE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// The canonical cache key for `path`: the symlink-resolved absolute path,
/// falling back to the path as given when it cannot be canonicalized (for
/// example a not-yet-existing file). Insert, lookup, and invalidate all key
/// off this single identity so differently-spelled but equal paths (a
/// trailing `./`, a relative spelling, a symlink) share one entry.
fn cache_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Set the maximum number of decoded rasters the process-global load cache
/// retains (libvips `vips_cache_set_max`). Lowering it below the current
/// occupancy evicts the least-recently-used entries immediately.
pub fn set_load_cache_max_entries(max: usize) {
    lock_cache().set_max_entries(max);
}

/// Set the total decoded-pixel byte ceiling the process-global load cache
/// holds (libvips `vips_cache_set_max_mem`). Lowering it below the current
/// footprint evicts the least-recently-used entries immediately.
pub fn set_load_cache_max_bytes(max: usize) {
    lock_cache().set_max_bytes(max);
}

/// Drop every entry from the process-global load cache (libvips
/// `vips_cache_drop_all`), so the next plain [`decode_file`] of any path
/// re-reads and re-decodes it from disk.
pub fn clear_load_cache() {
    lock_cache().clear();
}

/// Errors that can occur when decoding an image source.
///
/// Wraps the underlying I/O, image-decoding, and raster-construction
/// errors into a single enum so that callers of [`decode_file`],
/// [`decode_bytes`], and [`generate_test_raster`] can handle all failure
/// modes uniformly.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SourceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image decode error: {0}")]
    Decode(#[from] image::ImageError),
    #[error("unsupported color type: {0:?}")]
    UnsupportedColorType(image::ColorType),
    #[error("raster construction error: {0}")]
    Raster(#[from] crate::raster::RasterError),
    #[error(
        "image dimensions {width}x{height} exceed the configured pixel ceiling ({max_pixels} px)"
    )]
    DimensionLimitExceeded {
        width: u32,
        height: u32,
        max_pixels: u64,
    },
    /// A malformed or unsupported native `.v` file (bad magic, truncated
    /// header or pixel data, unsupported coding/band format, or header
    /// geometry past the [`crate::imageio::get_max_coord`] ceiling).
    #[error("vips .v file error: {0}")]
    VipsFormat(String),
}

/// Resource limits applied to a single image decode.
///
/// These bound the work a decoder may perform before pixel data is
/// materialised into a [`Raster`], guarding the process against
/// decompression bombs and pathologically large inputs. The width,
/// height, and allocation ceilings are pushed down into the underlying
/// decoder via [`image::Limits`] (so an oversized image is rejected
/// *before* it is fully allocated), and the combined `width * height`
/// pixel count is checked explicitly just before raster construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    /// Maximum decoded width, in pixels.
    pub max_width: u32,
    /// Maximum decoded height, in pixels.
    pub max_height: u32,
    /// Maximum total pixel count (`width * height`).
    pub max_pixels: u64,
    /// Maximum number of bytes the decoder may allocate at one time.
    pub max_alloc_bytes: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            // Matches the widest dimension the `image` strict limits can
            // express and covers every format libviprs targets.
            max_width: 65_535,
            max_height: 65_535,
            // ~1 gigapixel; large enough for legitimate scans, small
            // enough to reject a decompression bomb before allocation.
            max_pixels: 1u64 << 30,
            // Mirrors the `image` crate default allocation budget.
            max_alloc_bytes: 512 * 1024 * 1024,
        }
    }
}

impl DecodeLimits {
    /// Translate into the `image` crate's strict/non-strict limit set.
    fn to_image_limits(self) -> Limits {
        let mut limits = Limits::no_limits();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_alloc_bytes);
        limits
    }

    /// Enforce the `width * height` ceiling before a [`Raster`] is built.
    /// Crate-visible so the `.v` decoder in [`crate::imageio`] applies
    /// the same budget to its untrusted header geometry.
    pub(crate) fn check_pixels(self, width: u32, height: u32) -> Result<(), SourceError> {
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if pixels > self.max_pixels {
            return Err(SourceError::DimensionLimitExceeded {
                width,
                height,
                max_pixels: self.max_pixels,
            });
        }
        Ok(())
    }
}

/// Map `image` crate color types to our canonical pixel format.
fn color_type_to_format(ct: image::ColorType) -> Result<PixelFormat, SourceError> {
    match ct {
        image::ColorType::L8 => Ok(PixelFormat::Gray8),
        image::ColorType::L16 => Ok(PixelFormat::Gray16),
        image::ColorType::Rgb8 => Ok(PixelFormat::Rgb8),
        image::ColorType::Rgba8 => Ok(PixelFormat::Rgba8),
        image::ColorType::Rgb16 => Ok(PixelFormat::Rgb16),
        image::ColorType::Rgba16 => Ok(PixelFormat::Rgba16),
        // La8/La16 (gray + alpha) → promote to Rgba
        image::ColorType::La8 => Ok(PixelFormat::Rgba8),
        image::ColorType::La16 => Ok(PixelFormat::Rgba16),
        other => Err(SourceError::UnsupportedColorType(other)),
    }
}

/// Decode an image file into a canonical [`Raster`].
///
/// Reads the file at `path`, auto-detects the format (JPEG, PNG, TIFF),
/// and decodes it into an in-memory [`Raster`] with a canonical
/// [`PixelFormat`]. Palette and gray+alpha images are promoted to
/// RGB/RGBA so that downstream code only needs to handle a small set of
/// uniform formats.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   calls `decode_file` in the `info` command to display image metadata
///   and in the `pyramid` command to load the input raster.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid) (general
/// entry point) and [`viprs info`](https://libviprs.org/cli/#info).
///
/// The decode is served through a process-global, bounded-LRU load cache
/// (see [`LoadCache`]): the first load of a path is decoded from disk and
/// cached, and every later call returns that cached raster even if the
/// file has since changed on disk. Use [`decode_file_with_options`] with
/// `revalidate = true` to force a re-read, or [`Raster::invalidate`] to
/// drop a path's cached entry. The cache is bounded by an entry count and a
/// byte ceiling ([`set_load_cache_max_entries`] / [`set_load_cache_max_bytes`],
/// [`clear_load_cache`]), so it cannot grow without limit.
pub fn decode_file(path: &Path) -> Result<Raster, SourceError> {
    decode_file_with_options(path, false)
}

/// Decode an image file into a [`Raster`], optionally bypassing the
/// process-global load cache (libvips' `revalidate` load option).
///
/// With `revalidate = false` this is [`decode_file`]: a cache-first load
/// that returns the raster first decoded for `path`, even if the file has
/// since changed on disk. With `revalidate = true` the cache lookup is
/// skipped, the file is re-read and decoded fresh, and the cache entry for
/// `path` is refreshed so subsequent plain [`decode_file`] calls see the
/// new image. See [`LoadCache`] for the caching contract and its
/// libvips-binding rationale.
///
/// # Errors
///
/// As [`decode_file`]: I/O, decode, unsupported-format, dimension-limit,
/// and raster-construction errors are all surfaced as [`SourceError`].
pub fn decode_file_with_options(path: &Path, revalidate: bool) -> Result<Raster, SourceError> {
    let key = cache_key(path);
    if !revalidate {
        // Cache-first: on a hit, clone only the cheap Arc handle under the
        // lock and defer the deep pixel copy until the lock is released.
        let hit = lock_cache().get(&key);
        if let Some(raster) = hit {
            return Ok((*raster).clone());
        }
    }
    // Miss, or a forced revalidate: decode from disk outside the lock so a
    // slow decode never serialises other paths' loads.
    let raster = Arc::new(decode_file_with_limits(path, DecodeLimits::default())?);
    // Reconcile under one critical section: a forced revalidate always
    // overwrites the entry, while a plain miss inserts only if the key is
    // still absent (another thread may have populated it while we decoded).
    let resident = {
        let mut cache = lock_cache();
        if revalidate {
            cache.insert(key, raster)
        } else {
            cache.get_or_insert(key, raster)
        }
    };
    // Deep-clone the resident raster only after the lock is released.
    Ok((*resident).clone())
}

impl Raster {
    /// Drop this image's entry from the process-global load cache (libvips
    /// `vips_image_invalidate`), so the next plain [`decode_file`] of its
    /// source path re-reads and re-decodes it from disk.
    ///
    /// The path is the `filename` recorded by [`decode_file`] when the
    /// image was loaded. An image with no recorded filename (one built in
    /// memory) has nothing cached under a path, so this is a no-op for it.
    /// The recorded filename is canonicalized to the same identity the load
    /// keyed off, so invalidation reliably drops the entry even when this
    /// raster's filename spells the path differently. See [`LoadCache`] for
    /// the caching contract.
    pub fn invalidate(&mut self) {
        if let Some(MetadataValue::Str(filename)) = self.fields.get("filename") {
            let key = cache_key(Path::new(filename));
            lock_cache().remove(&key);
        }
    }
}

/// Decode an image file into a [`Raster`] under explicit [`DecodeLimits`].
///
/// Identical to [`decode_file`] but lets the caller supply the
/// dimension/allocation budget instead of using [`DecodeLimits::default`].
/// The limits are configured on the decoder before any pixel data is
/// allocated, and the `width * height` ceiling is checked before the
/// [`Raster`] is constructed.
/// Decode an image file in sequential-access mode (libvips
/// `access = sequential`).
///
/// Sequential access is a memory/IO hint: it lets a loader stream the image
/// top to bottom instead of keeping it randomly addressable. libviprs
/// decodes each supported format fully into a [`Raster`] regardless, so the
/// hint does not change the result: this returns exactly what [`decode_file`]
/// returns for the same file. The distinct entry point pins the ported
/// `access = sequential` call surface and reserves the seam for a future
/// streaming loader.
///
/// # Errors
///
/// As [`decode_file`].
pub fn decode_file_sequential(path: &Path) -> Result<Raster, SourceError> {
    decode_file(path)
}

/// Decode an image file with shrink-on-load by an integer factor (libvips
/// JPEG `shrink` load option, typically 1, 2, 4, or 8).
///
/// The image is decoded and then reduced by `shrink` on each axis with the
/// box shrink, giving output dimensions `round(dim / shrink)`. A `shrink` of
/// 0 or 1 returns the image at full size. A dedicated shrink-on-load decode
/// path (decoding directly at reduced resolution) can replace the body later
/// without changing this signature.
///
/// # Errors
///
/// As [`decode_file`], plus a shrink/resample failure surfaced as
/// [`SourceError::Io`].
pub fn decode_file_with_shrink(path: &Path, shrink: u32) -> Result<Raster, SourceError> {
    let base = decode_file(path)?;
    if shrink <= 1 {
        return Ok(base);
    }
    let factor = f64::from(shrink);
    base.try_shrink(factor, factor)
        .map_err(|e| SourceError::Io(std::io::Error::other(e.to_string())))
}

pub fn decode_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<Raster, SourceError> {
    // Sniff the leading magic: native .v files and JPEGs take the
    // in-memory path (the .v decoder parses the libvips header itself,
    // and the JPEG path scans APP1/APP2 segments for EXIF/ICC metadata
    // after pixel decode). Every other format keeps the original
    // streaming reader so its memory profile is unchanged.
    let mut head = [0u8; 4];
    {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut filled = 0;
        while filled < head.len() {
            let n = file.read(&mut head[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
    }
    let mut raster = if crate::imageio::is_vips_bytes(&head) || head[..3] == [0xFF, 0xD8, 0xFF] {
        decode_bytes_with_limits(&std::fs::read(path)?, limits)?
    } else {
        decode_reader(ImageReader::open(path)?, limits)?
    };
    // Record the source path, like the libvips header's filename slot.
    raster
        .fields
        .set("filename", path.display().to_string().into());
    Ok(raster)
}

/// Decode from an in-memory buffer (format auto-detected).
///
/// Behaves identically to [`decode_file`] but operates on a byte slice
/// that is already in memory. The image format is inferred from magic
/// bytes at the start of the buffer. This is the primary entry point
/// when the input arrives over a pipe or network socket rather than from
/// a filesystem path.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   calls `decode_bytes` when the user passes `"-"` as the input file,
///   reading the image data from stdin.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid) (general
/// entry point) and [`viprs info`](https://libviprs.org/cli/#info).
pub fn decode_bytes(bytes: &[u8]) -> Result<Raster, SourceError> {
    decode_bytes_with_limits(bytes, DecodeLimits::default())
}

/// Decode an in-memory buffer into a [`Raster`] under explicit [`DecodeLimits`].
///
/// Identical to [`decode_bytes`] but lets the caller supply the
/// dimension/allocation budget. The limits are configured on the decoder
/// before any pixel data is allocated, and the `width * height` ceiling
/// is checked before the [`Raster`] is constructed.
pub fn decode_bytes_with_limits(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    if crate::imageio::is_vips_bytes(bytes) {
        return crate::imageio::decode_vips_bytes(bytes, limits);
    }
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let is_jpeg = reader.format() == Some(image::ImageFormat::Jpeg);
    let mut raster = decode_reader(reader, limits)?;
    if is_jpeg {
        // Attach the EXIF/ICC metadata segments the pixel decoder skips,
        // as the libvips JPEG loader populates the image header.
        crate::imageio::attach_jpeg_metadata(&mut raster, bytes);
    }
    Ok(raster)
}

/// Apply the shared decode budget to a configured [`ImageReader`] and
/// finalize its output into a [`Raster`].
///
/// This is the single tail shared by [`decode_file_with_limits`] and
/// [`decode_bytes_with_limits`]: the only thing that differs between the
/// file and in-memory entry points is how the reader is constructed. Both
/// funnel through here so the limit push-down, decode, and
/// [`build_raster`] finalization (color-type mapping + La repacking) live
/// in exactly one place and cannot drift out of parity.
fn decode_reader<R: std::io::BufRead + std::io::Seek>(
    mut reader: ImageReader<R>,
    limits: DecodeLimits,
) -> Result<Raster, SourceError> {
    reader.limits(limits.to_image_limits());
    let img = reader.decode()?;
    build_raster(img, limits)
}

/// Materialise a decoded [`image::DynamicImage`] into a [`Raster`].
///
/// Enforces the pixel ceiling, maps the color type to a canonical
/// [`PixelFormat`], and packs the sample bytes. For gray+alpha inputs
/// the luminance is expanded to RGB in a single streaming pass so no
/// second full-image copy is buffered.
fn build_raster(img: image::DynamicImage, limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (width, height) = img.dimensions();
    // Enforce the explicit ceiling before allocating the packed buffer.
    limits.check_pixels(width, height)?;
    let color = img.color();
    let format = color_type_to_format(color)?;
    let data = pack_bytes(img, color);
    Ok(Raster::new(width, height, format, data)?)
}

/// Pack a decoded image into the canonical native-endian byte layout.
fn pack_bytes(img: image::DynamicImage, color: image::ColorType) -> Vec<u8> {
    match color {
        // 8-bit gray+alpha: a single expanding copy to RGBA8, then a
        // zero-copy unwrap of the backing buffer.
        image::ColorType::La8 => img.to_rgba8().into_raw(),
        // 16-bit gray+alpha: stream luminance → RGB directly into the
        // output byte buffer, borrowing the already-decoded LumaA16
        // samples so no intermediate RGBA16 image is materialised.
        image::ColorType::La16 => {
            let la = img
                .as_luma_alpha16()
                .expect("color type verified as La16 above");
            la16_to_rgba16_bytes(la.as_raw())
        }
        _ => img.into_bytes(),
    }
}

/// Expand interleaved `[luma, alpha]` u16 samples into RGBA16 bytes.
///
/// Writes exactly one output buffer: for each source pixel the single
/// luminance sample is emitted on all three color channels followed by
/// the alpha sample, each in native-endian byte order. This avoids the
/// second whole-image allocation that a `to_rgba16` conversion would
/// require before the byte re-pack.
fn la16_to_rgba16_bytes(samples: &[u16]) -> Vec<u8> {
    // 2 input samples per pixel → 4 output channels × 2 bytes.
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for pair in samples.chunks_exact(2) {
        let luma = pair[0];
        let alpha = pair[1];
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&alpha.to_ne_bytes());
    }
    bytes
}

/// Generate a synthetic test image (RGB8 gradient pattern).
///
/// Creates a `width x height` [`Raster`] in [`PixelFormat::Rgb8`] filled
/// with a deterministic gradient: the red channel increases left-to-right,
/// the green channel increases top-to-bottom, and the blue channel is
/// a diagonal blend. This is useful for verifying the full pipeline
/// without needing an external test fixture on disk.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   exposes this as the `test-image` subcommand, generating a gradient
///   PNG for quick smoke-testing.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#test-image)
pub fn generate_test_raster(width: u32, height: u32) -> Result<Raster, SourceError> {
    // Bound the requested dimensions against the shared decode budget before
    // allocating anything. Without this an oversized request would attempt an
    // unbounded `width * height * bpp` allocation (a debug abort / process
    // OOM) instead of failing cleanly; it also keeps `width + height` and the
    // per-pixel gradient math below within range. Oversized dimensions now
    // yield a typed `DimensionLimitExceeded`.
    let limits = DecodeLimits::default();
    limits.check_pixels(width, height)?;

    let bpp = PixelFormat::Rgb8.bytes_per_pixel();
    let mut data = vec![0u8; width as usize * height as usize * bpp];
    // Gradient math is widened to `u64`: for accepted dimensions a single
    // axis can still be as large as `max_pixels`, so `x * 255` (peaking near
    // 2^30 * 255) overflows a `u32`. The `.max(1)` denominators keep the
    // 1-pixel-wide/tall degenerate cases division-safe.
    let w_denom = u64::from(width).max(1);
    let h_denom = u64::from(height).max(1);
    let wh_denom = (u64::from(width) + u64::from(height)).max(1);
    for y in 0..height {
        for x in 0..width {
            let offset = (y as usize * width as usize + x as usize) * bpp;
            data[offset] = (u64::from(x) * 255 / w_denom) as u8;
            data[offset + 1] = (u64::from(y) * 255 / h_denom) as u8;
            data[offset + 2] = ((u64::from(x) + u64::from(y)) * 255 / wh_denom) as u8;
        }
    }
    Ok(Raster::new(width, height, PixelFormat::Rgb8, data)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(Cursor::new(&mut buf));
            let data = vec![128u8; w as usize * h as usize * 3];
            image::ImageEncoder::write_image(encoder, &data, w, h, image::ColorType::Rgb8.into())
                .unwrap();
        }
        buf
    }

    /// Encode a `w x h` La16 (gray + alpha) PNG in memory, returning the
    /// encoded bytes alongside the `(luma, alpha)` samples that were
    /// written so callers can verify the decoded RGBA16 layout.
    fn create_la16_png(w: u32, h: u32) -> (Vec<u8>, Vec<(u16, u16)>) {
        use image::{DynamicImage, ImageBuffer, ImageFormat, LumaA};

        let mut buf: ImageBuffer<LumaA<u16>, Vec<u16>> = ImageBuffer::new(w, h);
        let mut expected = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let luma = ((x.wrapping_mul(4096).wrapping_add(y.wrapping_mul(7))) & 0xFFFF) as u16;
                let alpha = ((x.wrapping_add(y.wrapping_mul(300))) & 0xFFFF) as u16;
                buf.put_pixel(x, y, LumaA([luma, alpha]));
                expected.push((luma, alpha));
            }
        }
        let dyn_img = DynamicImage::ImageLumaA16(buf);
        let mut out = Vec::new();
        dyn_img
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        (out, expected)
    }

    fn create_test_jpeg(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(Cursor::new(&mut buf), 95);
            let data = vec![128u8; w as usize * h as usize * 3];
            image::ImageEncoder::write_image(encoder, &data, w, h, image::ColorType::Rgb8.into())
                .unwrap();
        }
        buf
    }

    /**
     * Tests that a valid PNG byte buffer can be decoded into a Raster.
     * Works by encoding a known 32x24 RGB image to PNG in-memory, then
     * decoding it back and verifying dimensions, format, and buffer size.
     * Input: 32x24 RGB8 PNG bytes → Output: Raster(32, 24, Rgb8, 2304 bytes).
     */
    #[test]
    fn decode_png_from_memory() {
        let png = create_test_png(32, 24);
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 32);
        assert_eq!(raster.height(), 24);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.data().len(), 32 * 24 * 3);
    }

    /**
     * Tests that a valid JPEG byte buffer can be decoded into a Raster.
     * Works by encoding a 16x16 RGB image to JPEG at quality 95, then
     * decoding it and checking dimensions and format are preserved.
     * Input: 16x16 RGB8 JPEG bytes → Output: Raster(16, 16, Rgb8).
     */
    #[test]
    fn decode_jpeg_from_memory() {
        let jpeg = create_test_jpeg(16, 16);
        let raster = decode_bytes(&jpeg).unwrap();
        assert_eq!(raster.width(), 16);
        assert_eq!(raster.height(), 16);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
    }

    /**
     * Tests that decode_bytes returns an error for invalid image data.
     * Works by passing an arbitrary non-image byte string and asserting Err,
     * confirming the decoder rejects garbage input.
     * Input: b"not an image" → Output: Err.
     */
    #[test]
    fn decode_invalid_bytes_returns_error() {
        let result = decode_bytes(b"not an image");
        assert!(result.is_err());
    }

    /**
     * Tests that decode_bytes returns an error for an empty buffer.
     * Works by passing a zero-length slice, ensuring the decoder does not
     * panic and instead produces a meaningful error.
     * Input: b"" → Output: Err.
     */
    #[test]
    fn decode_empty_bytes_returns_error() {
        let result = decode_bytes(b"");
        assert!(result.is_err());
    }

    /**
     * Tests that generate_test_raster produces a Raster with correct
     * dimensions, pixel format, and buffer size.
     * Works by generating a 100x50 test raster and checking all properties.
     * Input: (100, 50) → Output: Raster(100, 50, Rgb8, 15000 bytes).
     */
    #[test]
    fn generate_test_raster_dimensions() {
        let r = generate_test_raster(100, 50).unwrap();
        assert_eq!(r.width(), 100);
        assert_eq!(r.height(), 50);
        assert_eq!(r.format(), PixelFormat::Rgb8);
        assert_eq!(r.data().len(), 100 * 50 * 3);
    }

    /**
     * Reproducer for the unchecked gradient arithmetic: a single axis large
     * enough to keep the total pixel count under the ceiling still drives the
     * old `x * 255` computation past `u32::MAX` (16_777_215 * 255 already
     * exceeds it). Before widening the math to `u64` this panicked on
     * overflow in debug builds (and silently wrapped in release); it must now
     * complete and paint the far-right column at full red intensity.
     * Input: (20_000_000, 1) → Output: Ok(Raster) with a saturated last pixel.
     */
    #[test]
    fn generate_test_raster_wide_no_overflow() {
        let width = 20_000_000u32;
        let r = generate_test_raster(width, 1).unwrap();
        assert_eq!(r.width(), width);
        assert_eq!(r.height(), 1);
        // Red channel of the last pixel: (width-1) * 255 / width ≈ 254.
        let last = ((width - 1) as usize) * 3;
        assert_eq!(r.data()[last], 254);
    }

    /**
     * Reproducer for the missing allocation cap: dimensions whose pixel count
     * exceeds the shared decode budget must be rejected with a typed
     * `DimensionLimitExceeded` before any buffer is allocated. Before the fix
     * `generate_test_raster` sized its `vec!` straight from the raw
     * dimensions, so this request attempted a multi-gigabyte allocation (a
     * process abort) instead of returning an error.
     * Input: (65_535, 65_535) → Output: Err(DimensionLimitExceeded).
     */
    #[test]
    fn generate_test_raster_rejects_oversized() {
        let (width, height) = (65_535u32, 65_535u32);
        match generate_test_raster(width, height) {
            Err(SourceError::DimensionLimitExceeded {
                width: w,
                height: h,
                max_pixels,
            }) => {
                assert_eq!(w, width);
                assert_eq!(h, height);
                assert_eq!(max_pixels, DecodeLimits::default().max_pixels);
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }
    }

    /**
     * Tests that color_type_to_format correctly maps image crate ColorType
     * variants to PixelFormat, including the La8→Rgba8 promotion.
     * Works by checking each supported mapping individually.
     * Input: e.g. ColorType::L8 → Output: PixelFormat::Gray8.
     */
    #[test]
    fn color_type_mapping() {
        assert_eq!(
            color_type_to_format(image::ColorType::L8).unwrap(),
            PixelFormat::Gray8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgb8).unwrap(),
            PixelFormat::Rgb8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgba8).unwrap(),
            PixelFormat::Rgba8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgb16).unwrap(),
            PixelFormat::Rgb16
        );
        assert_eq!(
            color_type_to_format(image::ColorType::La8).unwrap(),
            PixelFormat::Rgba8
        );
    }

    /**
     * Tests that decode_file can read and decode a PNG from disk.
     * Works by writing a known PNG to a temp file, then decoding it
     * with decode_file and verifying the resulting Raster properties.
     * Input: 8x8 RGB8 PNG on disk → Output: Raster(8, 8, Rgb8).
     *
     * Split for Miri: tempdir/write are blocked under Miri's isolation
     * mode. The first half decodes the PNG bytes in memory via
     * decode_bytes and checks the resulting Raster dimensions and format
     * (runs everywhere). The #[cfg(not(miri))] block writes the PNG to
     * a temp file and decodes it back via decode_file to test the
     * filesystem round-trip (skipped under Miri).
     */
    #[test]
    fn decode_file_from_disk() {
        let png = create_test_png(8, 8);

        // Miri-safe: verify decoding from bytes in memory
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 8);
        assert_eq!(raster.height(), 8);
        assert_eq!(raster.format(), PixelFormat::Rgb8);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.png");
            std::fs::write(&path, &png).unwrap();

            let from_disk = decode_file(&path).unwrap();
            assert_eq!(from_disk.width(), 8);
            assert_eq!(from_disk.height(), 8);
            assert_eq!(from_disk.format(), PixelFormat::Rgb8);
        }
    }

    /**
     * Tests that decode_file returns an error for a nonexistent path.
     * Works by passing a path that does not exist and asserting Err,
     * confirming proper I/O error propagation.
     * Input: Path("/nonexistent/image.png") → Output: Err.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn decode_file_not_found() {
        let result = decode_file(Path::new("/nonexistent/image.png"));
        assert!(result.is_err());
    }

    /**
     * Reproducer for the missing dimension ceiling: a fully decodable
     * image must still be rejected when its pixel count exceeds the
     * configured `max_pixels`. Works by decoding a valid 64x64 PNG under
     * a `DecodeLimits` whose ceiling (100 px) is far below 64*64=4096 and
     * asserting a `DimensionLimitExceeded` error is returned before the
     * raster is built. Before the fix no ceiling existed and this decode
     * succeeded. Input: 64x64 PNG + max_pixels=100 → Output: Err.
     */
    #[test]
    fn decode_bytes_rejects_over_pixel_ceiling() {
        let png = create_test_png(64, 64);
        let limits = DecodeLimits {
            max_pixels: 100,
            ..DecodeLimits::default()
        };
        let result = decode_bytes_with_limits(&png, limits);
        match result {
            Err(SourceError::DimensionLimitExceeded {
                width,
                height,
                max_pixels,
            }) => {
                assert_eq!(width, 64);
                assert_eq!(height, 64);
                assert_eq!(max_pixels, 100);
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }
        // The same bytes decode fine under the default (generous) ceiling.
        assert!(decode_bytes(&png).is_ok());
    }

    /**
     * Confirms the explicit width/height limits are pushed down into the
     * decoder itself (not merely checked after the fact): decoding a
     * 64-wide PNG under `max_width = 10` must fail with an `image`
     * limit/decode error. Input: 64x48 PNG + max_width=10 → Output: Err.
     */
    #[test]
    fn decode_bytes_enforces_decoder_width_limit() {
        let png = create_test_png(64, 48);
        let limits = DecodeLimits {
            max_width: 10,
            ..DecodeLimits::default()
        };
        let result = decode_bytes_with_limits(&png, limits);
        assert!(
            matches!(result, Err(SourceError::Decode(_))),
            "expected a decoder limit error, got {result:?}"
        );
    }

    /**
     * Guards the shared finalize path: `decode_file` and `decode_bytes`
     * must produce byte-identical rasters for the same input, including the
     * La16 -> RGBA16 promotion. Both entry points funnel through the single
     * `decode_reader` -> `build_raster` tail (the issue's proposed
     * `finalize()`), so the color-type mapping and La repacking cannot drift
     * back into copy-pasted divergence. Were the two paths ever re-forked,
     * an La16 (a format with non-trivial repacking) input would expose the
     * mismatch here. Input: 6x4 La16 PNG decoded via both entry points ->
     * Output: equal dimensions, format, and pixel bytes.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn decode_file_and_bytes_share_finalize_path() {
        let (png, _expected) = create_la16_png(6, 4);

        let from_bytes = decode_bytes(&png).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parity.png");
        std::fs::write(&path, &png).unwrap();
        let from_file = decode_file(&path).unwrap();

        assert_eq!(from_file.width(), from_bytes.width());
        assert_eq!(from_file.height(), from_bytes.height());
        assert_eq!(from_file.format(), from_bytes.format());
        assert_eq!(from_file.format(), PixelFormat::Rgba16);
        assert_eq!(from_file.data(), from_bytes.data());
    }

    /**
     * Verifies the streaming La16 conversion produces the exact RGBA16
     * native-endian byte layout: luminance replicated across R, G, B and
     * the alpha sample last, two bytes each. Works by encoding a known
     * La16 PNG, decoding it, and comparing every 8-byte pixel against the
     * expected expansion of the original (luma, alpha) samples.
     * Input: 5x3 La16 PNG → Output: Raster(5, 3, Rgba16) with each pixel
     * bytes == [luma, luma, luma, alpha] in native endian.
     */
    #[test]
    fn decode_la16_streams_to_rgba16_layout() {
        let (png, expected) = create_la16_png(5, 3);
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 5);
        assert_eq!(raster.height(), 3);
        assert_eq!(raster.format(), PixelFormat::Rgba16);

        let data = raster.data();
        assert_eq!(data.len(), expected.len() * 4 * 2);
        for (i, &(luma, alpha)) in expected.iter().enumerate() {
            let base = i * 8;
            let mut want = Vec::with_capacity(8);
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&alpha.to_ne_bytes());
            assert_eq!(&data[base..base + 8], want.as_slice(), "pixel {i} mismatch");
        }
    }

    /**
     * Reproduces the ported `test_revalidate` cache contract. A plain
     * `decode_file` serves the first raster decoded for a path even after
     * the file is overwritten on disk; `decode_file_with_options(_, true)`
     * bypasses the cache, re-reads, and refreshes the entry; and a later
     * plain reload then observes the refreshed image.
     * Input: write 10x10 .v -> load 10; overwrite 20x20 -> plain reload
     * still 10 (stale); revalidate -> 20; plain reload -> 20.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn revalidate_bypasses_and_refreshes_cache() {
        let _serial = cache_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("revalidate.v");

        Raster::black(10, 10).save(&path).unwrap();
        assert_eq!(decode_file(&path).unwrap().width(), 10);

        // Overwrite with a differently sized image on disk.
        Raster::black(20, 20).save(&path).unwrap();

        // Plain reload is served from the cache: still the old width.
        assert_eq!(decode_file(&path).unwrap().width(), 10);

        // Revalidate bypasses the cache and refreshes the entry.
        assert_eq!(decode_file_with_options(&path, true).unwrap().width(), 20);

        // The refreshed entry is what a plain reload now returns.
        assert_eq!(decode_file(&path).unwrap().width(), 20);
    }

    /**
     * Tests that `Raster::invalidate` drops the cached entry for the
     * image's source path, so the next plain `decode_file` re-reads the
     * (changed) file from disk.
     * Input: load 8x8 .v; overwrite 16x16; plain reload still 8 (cached);
     * invalidate; plain reload -> 16.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn invalidate_drops_cache_entry() {
        let _serial = cache_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalidate.v");

        Raster::black(8, 8).save(&path).unwrap();
        let mut im = decode_file(&path).unwrap();
        assert_eq!(im.width(), 8);

        Raster::black(16, 16).save(&path).unwrap();
        // Still served from the cache until invalidated.
        assert_eq!(decode_file(&path).unwrap().width(), 8);

        im.invalidate();
        // Entry dropped: the plain reload re-reads the 16-wide file.
        assert_eq!(decode_file(&path).unwrap().width(), 16);
    }

    /// Serializes the tests that assert on the *global* load cache's
    /// cross-call state, so one test clearing or shrinking the shared cache
    /// can never race another's stale-hit expectation. Poison is recovered
    /// in place (the guarded unit is `()`).
    fn cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());
        CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// A shared raster whose pixel buffer is exactly `bytes` long, for
    /// exercising the byte-cap accounting deterministically. Gray8 is one
    /// byte per pixel, so a `bytes`x1 raster is exactly `bytes` long.
    fn cache_raster(bytes: usize) -> Arc<Raster> {
        Arc::new(Raster::zeroed(bytes as u32, 1, PixelFormat::Gray8).unwrap())
    }

    fn key(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    /**
     * Past the entry-count cap, an insert evicts the least-recently-used
     * entry, not an arbitrary or the newest one. Touching "a" after both
     * "a" and "b" are resident makes "b" the LRU, so the third insert
     * evicts "b" and keeps "a" and "c".
     */
    #[test]
    fn lru_evicts_least_recently_used_past_entry_cap() {
        let mut cache = LoadCache::new(2, usize::MAX);
        cache.insert(key("a"), cache_raster(1));
        cache.insert(key("b"), cache_raster(1));
        // Touch "a": now "b" is the least-recently-used.
        assert!(cache.get(&key("a")).is_some());
        // Third insert past the 2-entry cap evicts the LRU ("b").
        cache.insert(key("c"), cache_raster(1));
        assert_eq!(cache.map.len(), 2);
        assert!(cache.get(&key("a")).is_some());
        assert!(cache.get(&key("c")).is_some());
        assert!(
            cache.get(&key("b")).is_none(),
            "b was least-recently-used and should have been evicted"
        );
    }

    /**
     * Past the total-bytes cap, an insert evicts least-recently-used entries
     * until the footprint fits. A 150-byte cap holds one 100-byte raster;
     * inserting a second pushes the total to 200 and evicts the LRU first
     * entry, leaving the cache at one entry and 100 bytes.
     */
    #[test]
    fn lru_evicts_past_byte_cap() {
        let mut cache = LoadCache::new(usize::MAX, 150);
        cache.insert(key("a"), cache_raster(100));
        assert_eq!(cache.total_bytes, 100);
        cache.insert(key("b"), cache_raster(100));
        assert_eq!(cache.map.len(), 1);
        assert_eq!(cache.total_bytes, 100);
        assert!(
            cache.get(&key("a")).is_none(),
            "a should have been evicted to satisfy the byte cap"
        );
        assert!(cache.get(&key("b")).is_some());
    }

    /**
     * Lowering the entry cap (the `vips_cache_set_max` knob) evicts the
     * least-recently-used entries immediately until the new cap holds.
     */
    #[test]
    fn set_max_entries_evicts_on_shrink() {
        let mut cache = LoadCache::new(4, usize::MAX);
        cache.insert(key("a"), cache_raster(1));
        cache.insert(key("b"), cache_raster(1));
        cache.insert(key("c"), cache_raster(1));
        cache.set_max_entries(1);
        assert_eq!(cache.map.len(), 1);
        assert!(
            cache.get(&key("c")).is_some(),
            "the most-recently-used entry survives the shrink"
        );
    }

    /**
     * Lowering the byte cap (the `vips_cache_set_max_mem` knob) evicts the
     * least-recently-used entries immediately until the footprint fits.
     */
    #[test]
    fn set_max_bytes_evicts_on_shrink() {
        let mut cache = LoadCache::new(usize::MAX, usize::MAX);
        cache.insert(key("a"), cache_raster(100));
        cache.insert(key("b"), cache_raster(100));
        assert_eq!(cache.total_bytes, 200);
        cache.set_max_bytes(100);
        assert_eq!(cache.total_bytes, 100);
        assert!(cache.get(&key("a")).is_none());
        assert!(cache.get(&key("b")).is_some());
    }

    /**
     * Clearing drops every entry and zeroes the byte accounting.
     */
    #[test]
    fn clear_empties_the_cache() {
        let mut cache = LoadCache::new(usize::MAX, usize::MAX);
        cache.insert(key("a"), cache_raster(10));
        cache.insert(key("b"), cache_raster(20));
        assert_eq!(cache.total_bytes, 30);
        cache.clear();
        assert_eq!(cache.map.len(), 0);
        assert_eq!(cache.total_bytes, 0);
        assert!(cache.get(&key("a")).is_none());
    }

    /**
     * A plain miss inserts only if absent (`get_or_insert` keeps the resident
     * entry when a concurrent decode already populated it), while a forced
     * revalidate (`insert`) overwrites unconditionally. Verified by identity:
     * `get_or_insert` returns the original Arc, `insert` a fresh one.
     */
    #[test]
    fn get_or_insert_keeps_existing_while_insert_overwrites() {
        let mut cache = LoadCache::new(usize::MAX, usize::MAX);
        let first = cache.insert(key("a"), cache_raster(4));
        let resident = cache.get_or_insert(key("a"), cache_raster(8));
        assert!(
            Arc::ptr_eq(&first, &resident),
            "get_or_insert must keep the already-resident entry"
        );
        assert_eq!(cache.map.len(), 1);
        assert_eq!(cache.total_bytes, 4);
        let replaced = cache.insert(key("a"), cache_raster(8));
        assert!(
            !Arc::ptr_eq(&first, &replaced),
            "insert must overwrite with the new entry"
        );
        assert_eq!(cache.total_bytes, 8);
    }

    /**
     * Insert, lookup, and invalidate key off one canonical path identity, so
     * differently-spelled but equal paths share a single cache entry. Loading
     * through a `./`-spelled path caches under the canonical key; a later
     * plain load through the canonical spelling hits that same entry (a naive
     * raw-path key would miss and re-read the overwritten file); and
     * invalidating the `./`-spelled raster canonicalizes its recorded
     * filename to the same key and drops the entry.
     * Input: cache 8x8 via `dir/./ident.v`; overwrite 16x16; load via
     * `dir/ident.v` -> still 8 (canonical hit); invalidate -> next load 16.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn path_identity_canonicalizes_lookup_and_invalidate() {
        let _serial = cache_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().join("ident.v");
        // A differently-spelled but equal path (a redundant `.` component).
        let spelled = dir.path().join(".").join("ident.v");

        Raster::black(8, 8).save(&canonical).unwrap();

        // Populate through the differently-spelled path: the key is
        // canonicalized, but the raster records the raw `./` spelling.
        let mut via_spelled = decode_file(&spelled).unwrap();
        assert_eq!(via_spelled.width(), 8);

        // Overwrite on disk; a plain load through the canonical spelling
        // still hits the same entry, proving lookup collapses both spellings
        // to one identity.
        Raster::black(16, 16).save(&canonical).unwrap();
        assert_eq!(decode_file(&canonical).unwrap().width(), 8);

        // Invalidating the `./`-spelled raster canonicalizes its filename to
        // the same key and drops the entry, so the next load re-reads.
        via_spelled.invalidate();
        assert_eq!(decode_file(&canonical).unwrap().width(), 16);
    }

    /**
     * The public cache-control knobs bound and clear the process-global
     * cache. Bounding to a single entry evicts the older path on the next
     * load (so an overwritten, previously-cached path re-reads fresh); the
     * defaults are restored and the cache cleared so the test leaves no
     * global residue. Serialized with the other global-cache tests.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn public_cache_controls_bound_and_clear_global_cache() {
        let _serial = cache_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("ctl_a.v");
        let b = dir.path().join("ctl_b.v");
        Raster::black(4, 4).save(&a).unwrap();
        Raster::black(4, 4).save(&b).unwrap();

        clear_load_cache();
        set_load_cache_max_bytes(usize::MAX);
        set_load_cache_max_entries(1);

        // Two distinct loads under a 1-entry cap: "a" (older) is evicted.
        assert_eq!(decode_file(&a).unwrap().width(), 4);
        assert_eq!(decode_file(&b).unwrap().width(), 4);

        // Overwrite "a": had its entry survived it would report a stale 4;
        // instead it was evicted, so a fresh load sees the new 8.
        Raster::black(8, 8).save(&a).unwrap();
        assert_eq!(
            decode_file(&a).unwrap().width(),
            8,
            "a should have been evicted by the 1-entry bound"
        );

        // Restore defaults and clear so no global residue leaks to other
        // (serialized) global-cache tests.
        set_load_cache_max_entries(DEFAULT_LOAD_CACHE_MAX_ENTRIES);
        set_load_cache_max_bytes(DEFAULT_LOAD_CACHE_MAX_BYTES);
        clear_load_cache();
    }

    #[test]
    fn decode_file_sequential_matches_decode_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seq.v");
        generate_test_raster(48, 32).unwrap().save(&path).unwrap();

        let normal = decode_file(&path).unwrap();
        let sequential = decode_file_sequential(&path).unwrap();
        assert_eq!(normal.width(), sequential.width());
        assert_eq!(normal.height(), sequential.height());
        assert_eq!(normal.format(), sequential.format());
        assert_eq!(normal.data(), sequential.data());
    }

    #[test]
    fn decode_file_with_shrink_reduces_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shrink.v");
        generate_test_raster(100, 100).unwrap().save(&path).unwrap();

        // shrink <= 1 is a no-op returning the full-size image.
        assert_eq!(decode_file_with_shrink(&path, 1).unwrap().width(), 100);

        for factor in [2u32, 4] {
            let shrunk = decode_file_with_shrink(&path, factor).unwrap();
            let expected = (100.0_f64 / f64::from(factor)).round() as i64;
            assert!(
                (i64::from(shrunk.width()) - expected).abs() <= 1,
                "shrink={factor}: width {} not near {expected}",
                shrunk.width()
            );
            assert!(
                (i64::from(shrunk.height()) - expected).abs() <= 1,
                "shrink={factor}: height {} not near {expected}",
                shrunk.height()
            );
        }
    }
}
