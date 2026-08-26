use std::collections::HashMap;
use std::io::{Cursor, Seek};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use image::{GenericImageView, ImageDecoder, ImageReader, Limits};
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
    /// Untrusted header geometry whose single-axis width or height exceeds
    /// the per-decode [`DecodeLimits::max_coord`] ceiling (the libvips
    /// `VIPS_MAX_COORD` limit). Distinct from
    /// [`DimensionLimitExceeded`](SourceError::DimensionLimitExceeded),
    /// which bounds the *total* `width * height` pixel count. Raised by the
    /// native `.v` reader — and, since libviprs#349, the `image`-crate
    /// raster path — before any allocation so callers can match the
    /// over-ceiling case as a typed variant instead of substring-matching
    /// [`VipsFormat`](SourceError::VipsFormat). This variant reports only the
    /// [`max_coord`](DecodeLimits::max_coord) ceiling; a raster dimension
    /// above [`max_width`](DecodeLimits::max_width) /
    /// [`max_height`](DecodeLimits::max_height) is rejected by the `image`
    /// crate instead and surfaces as [`Decode`](SourceError::Decode) wrapping
    /// [`image::ImageError::Limits`], not as this variant.
    #[error(
        "image dimensions {width}x{height} exceed the single-axis coordinate \
         ceiling ({max_coord} px per axis); raise DecodeLimits::max_coord"
    )]
    CoordLimitExceeded {
        width: u32,
        height: u32,
        max_coord: u32,
    },
    /// A malformed or unsupported native `.v` file (bad magic, truncated
    /// header or pixel data, unsupported coding/band format, or a metadata
    /// trailer that opens with `{` and is not valid JSON, which is a
    /// corrupt legacy libviprs trailer rather than a foreign one; see the
    /// [`crate::imageio`] container contract).
    #[error("vips .v file error: {0}")]
    VipsFormat(String),
    /// A malformed Radiance `.hdr` file. libviprs decodes Radiance itself
    /// rather than through the `image` crate (see [`crate::radiance`]), so
    /// its failures arrive as the codec's own typed
    /// [`RadianceError`](crate::radiance::RadianceError) rather than as an
    /// opaque string.
    #[error(transparent)]
    Radiance(#[from] crate::radiance::RadianceError),
    /// A malformed GIF. libviprs decodes GIF through the `gif` crate rather
    /// than the `image` facade (see [`crate::gif`]), because the facade
    /// hard-codes RGBA output and hides the transparent index, so its
    /// failures arrive as the codec's own typed
    /// [`GifError`](crate::gif::GifError) rather than as an opaque string.
    #[error(transparent)]
    Gif(#[from] crate::gif::GifError),
    /// A malformed or unreachable FITS file. libviprs decodes FITS itself
    /// rather than through any crate (see [`crate::fits`]), so its failures
    /// arrive as the codec's own typed
    /// [`FitsError`](crate::fits::FitsError) rather than as an opaque
    /// string. That matters more here than elsewhere, because a FITS file
    /// can be perfectly well formed and still carry a sample type this
    /// build has no pixel format for; the variant says which.
    #[error(transparent)]
    Fits(#[from] crate::fits::FitsError),
    /// An SVG document `usvg` refused to parse, raised by
    /// [`crate::svg::decode_svg`]. Carries the underlying message rather
    /// than the foreign error type so `SourceError` does not leak a
    /// feature-gated dependency into its public shape.
    #[error("SVG parse error: {message}")]
    SvgParse {
        /// The `usvg` parse failure, rendered through its `Display`.
        message: String,
    },
    /// An SVG buffer larger than the input ceiling with
    /// [`crate::svg::SvgOptions::unlimited`] left false. Distinct from the
    /// output-geometry ceilings: this bounds the *document* before it is
    /// parsed, where [`CoordLimitExceeded`](SourceError::CoordLimitExceeded)
    /// and [`DimensionLimitExceeded`](SourceError::DimensionLimitExceeded)
    /// bound the raster it renders to.
    #[error(
        "SVG input is {bytes} bytes, over the {max_bytes}-byte ceiling; \
         set SvgOptions::unlimited to lift it"
    )]
    SvgInputTooLarge {
        /// The length of the buffer that was offered.
        bytes: usize,
        /// The ceiling in force, [`crate::svg::MAX_INPUT_BYTES`].
        max_bytes: usize,
    },
    /// An SVG whose scaled geometry rounded to zero on at least one axis,
    /// mirroring the `zero-sized image` bail-out in libvips
    /// `svgload.c:588`. Reported instead of constructing a zero-dimension
    /// [`Raster`], which [`crate::raster::RasterError::ZeroDimension`]
    /// would refuse anyway with less context about why.
    #[error("SVG renders to a zero-sized image ({width}x{height} after scaling)")]
    SvgZeroSize {
        /// The rounded output width, in pixels.
        width: u32,
        /// The rounded output height, in pixels.
        height: u32,
    },
    /// A single buffer a decoder is about to reserve would be larger than
    /// [`DecodeLimits::max_alloc_bytes`]. Raised before the allocation
    /// happens, from the size the file *declares*, so a decompression bomb
    /// is refused rather than served. Distinct from
    /// [`DimensionLimitExceeded`](SourceError::DimensionLimitExceeded),
    /// which counts pixels and so cannot see the band count or the sample
    /// depth: a 1-gigapixel `max_pixels` still permits a 4 GiB `Rgba8`
    /// frame. Used by the TIFF page readers for both the file body they
    /// read in and the pixel buffer a page decodes into; the equivalent GIF
    /// check has its own
    /// [`GifError::AllocLimitExceeded`](crate::gif::GifError::AllocLimitExceeded).
    #[error(
        "{what} needs {needed_bytes} bytes, over the {max_alloc_bytes}-byte \
         allocation ceiling; raise DecodeLimits::max_alloc_bytes"
    )]
    AllocLimitExceeded {
        /// What the allocation was for, e.g. `"TIFF file body"`.
        what: &'static str,
        /// The number of bytes that single buffer would have taken.
        needed_bytes: u64,
        /// The ceiling in force, [`DecodeLimits::max_alloc_bytes`].
        max_alloc_bytes: u64,
    },
    /// A multi-page file whose page chain runs past
    /// [`DecodeLimits::max_pages`]. The TIFF IFD chain is a linked list with
    /// no count in the header, so the only way to know how long it is, is to
    /// walk it; this variant is what stops that walk turning into unbounded
    /// work on a hostile file.
    #[error("image declares more than {max_pages} pages; raise DecodeLimits::max_pages")]
    PageLimitExceeded {
        /// The ceiling in force, [`DecodeLimits::max_pages`]. The real page
        /// count is deliberately not reported: the walk stops at the ceiling
        /// rather than running to the end of the chain to count it, which is
        /// the whole point of the ceiling.
        max_pages: u32,
    },
}

/// Resource limits applied to a single image decode.
///
/// These bound the work a decoder may perform before pixel data is
/// materialised into a [`Raster`], guarding the process against
/// decompression bombs and pathologically large inputs. The width,
/// height, and allocation ceilings are pushed down into the underlying
/// decoder via [`image::Limits`] (so an oversized image is rejected
/// *before* it is fully allocated), and the combined `width * height`
/// pixel count is checked explicitly on the declared header geometry
/// before the output frame is allocated (and re-verified just before
/// raster construction).
///
/// # Which decoder enforces which field
///
/// The decode paths bound untrusted geometry with different mechanisms, so
/// not every field is consulted by all of them. The `image`-crate raster
/// path covers PNG and JPEG; TIFF reaches it too through
/// [`decode_file`]/[`decode_bytes`], but the page-aware TIFF readers
/// ([`crate::tiff_page_count`], [`crate::decode_tiff_page`] and their
/// `_with_limits` twins) go straight to the `tiff` crate and enforce the
/// ceilings themselves.
///
/// | Field | `image` raster path | native `.v` reader | TIFF page readers |
/// |---|---|---|---|
/// | [`max_coord`](Self::max_coord) | ✅ before allocation | ✅ before allocation | ✅ before allocation |
/// | [`max_pixels`](Self::max_pixels) | ✅ before allocation (in [`decode_reader`], re-verified in `build_raster`) | ✅ before allocation | ✅ before allocation |
/// | [`max_width`](Self::max_width) / [`max_height`](Self::max_height) | ✅ via [`image::Limits`] (see below) | — (bounded instead by `max_coord`) | — (bounded instead by `max_coord`) |
/// | [`max_alloc_bytes`](Self::max_alloc_bytes) | ✅ via [`image::Limits`] | — (`.v` is an uncompressed body sized by its header, gated by `max_coord`/`max_pixels`) | ✅ on the file body, the pixel buffer, and the `tiff` decoder's own buffers |
/// | [`max_pages`](Self::max_pages) | — (single-page entry points) | — (`.v` is single-page) | ✅ bounds the IFD walk |
///
/// The single-axis [`max_coord`](Self::max_coord) and total
/// [`max_pixels`](Self::max_pixels) ceilings are the universally honoured
/// knobs; `max_width`/`max_height` shape only the `image`-crate decoders
/// they are pushed into.
///
/// The format decoders libviprs owns outright take the same
/// [`DecodeLimits`] and apply `max_coord`, `max_pixels` and
/// `max_alloc_bytes` in that order before reserving a frame:
/// [`crate::gif::decode_gif`], [`crate::webp::decode_webp`], and the TIFF
/// page readers. [`max_alloc_bytes`](Self::max_alloc_bytes) is the one that
/// catches a frame `max_pixels` waves through, since a pixel count sees
/// neither the band count nor the sample depth.
///
/// Raising [`max_alloc_bytes`](Self::max_alloc_bytes) above 256 MiB does
/// **not** raise the effective ceiling on the TIFF page readers: the `tiff`
/// crate's own `decoding_buffer_size` default is 256 MiB and libviprs only
/// ever tightens it, never loosens it, so the effective bound there is the
/// smaller of the two.
///
/// Note the [`max_width`](Self::max_width) / [`max_height`](Self::max_height)
/// ceilings and [`max_coord`](Self::max_coord) surface *different* errors on
/// the raster path: a declared dimension above `max_width` / `max_height` is
/// rejected inside the `image` crate via [`image::Limits`], so it arrives as
/// [`SourceError::Decode`] wrapping [`image::ImageError::Limits`] — **not**
/// [`SourceError::CoordLimitExceeded`], which is reserved for the
/// `max_coord` check applied by [`decode_reader`] / the `.v` reader. Because
/// the default `max_width` / `max_height` (65,535) sit far below the default
/// `max_coord` (10,000,000), a raster dimension between those bounds trips
/// the `image::Limits` path first; `CoordLimitExceeded` is what you see once
/// `max_coord` is tightened at or below `max_width` / `max_height`.
///
/// The struct is `#[non_exhaustive]`: new limit fields can be added in a
/// future minor release without it being a breaking change, so external
/// callers cannot construct it with a struct literal (including functional
/// update syntax). Start from [`DecodeLimits::default`] and adjust
/// individual ceilings with the `with_*` builder setters, e.g.
///
/// ```
/// use libviprs::source::DecodeLimits;
/// let limits = DecodeLimits::default().with_max_coord(20_000);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct DecodeLimits {
    /// Maximum decoded width, in pixels.
    pub max_width: u32,
    /// Maximum decoded height, in pixels.
    pub max_height: u32,
    /// Maximum single-axis dimension permitted in untrusted header
    /// geometry, in pixels per axis (the libvips `VIPS_MAX_COORD`
    /// ceiling, default [`crate::imageio::DEFAULT_MAX_COORD`]). Enforced
    /// per decode on the declared header geometry — before any pixel
    /// allocation — by **every** decoder: the native `.v` reader and the
    /// `image`-crate raster path (PNG/JPEG/TIFF) alike, both routing
    /// through [`DecodeLimits::check_coord`] and returning
    /// [`SourceError::CoordLimitExceeded`] on an over-ceiling axis. This is
    /// the sole coordinate-ceiling knob: it replaced an earlier
    /// process-global whose races under concurrent jobs made the ceiling
    /// unreadable from the API.
    pub max_coord: u32,
    /// Maximum total pixel count (`width * height`).
    pub max_pixels: u64,
    /// Maximum number of bytes the decoder may allocate at one time.
    pub max_alloc_bytes: u64,
    /// Maximum number of pages (frames, IFDs) a multi-page file may declare
    /// before it is refused, default `100_000`. A TIFF's IFD chain is a
    /// linked list with no count anywhere in the header, so reporting
    /// `n-pages` means walking it; this bounds that walk. Enforced by the
    /// TIFF page readers ([`crate::tiff_page_count`],
    /// [`crate::decode_tiff_page`]) as
    /// [`SourceError::PageLimitExceeded`], raised as soon as the walk
    /// reaches the ceiling rather than after counting to the end.
    ///
    /// The default is the ceiling libvips puts on both the page index and
    /// the page count on every multi-page loader it has
    /// (`VIPS_ARG_INT(class, "page", 20, ..., 0, 100000, 0)` in
    /// `tiffload.c:195-200` at `fe420cf3a`, and `-1, 100000, 1` for `n`).
    /// Measured against 8.18.4: `vips tiffload x.tif o.v --page 100001` and
    /// `--n 100001` are both refused by GObject before the loader runs, so a
    /// chain longer than this is past anything vips will address in one go.
    pub max_pages: u32,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            // Matches the widest dimension the `image` strict limits can
            // express and covers every format libviprs targets.
            max_width: 65_535,
            max_height: 65_535,
            // The libvips `VIPS_MAX_COORD` single-axis ceiling. Kept equal
            // to the former process-global default so an in-bounds decode
            // is byte-identical to prior releases.
            max_coord: crate::imageio::DEFAULT_MAX_COORD,
            // ~1 gigapixel; large enough for legitimate scans, small
            // enough to reject a decompression bomb before allocation.
            max_pixels: 1u64 << 30,
            // Mirrors the `image` crate default allocation budget.
            max_alloc_bytes: 512 * 1024 * 1024,
            // The libvips `page` / `n` property ceiling on every multi-page
            // loader; see the field doc.
            max_pages: 100_000,
        }
    }
}

impl DecodeLimits {
    /// Set the maximum decoded width, in pixels, returning the updated
    /// limits. Builder setter for [`#[non_exhaustive]`](DecodeLimits)
    /// customisation from [`DecodeLimits::default`].
    #[must_use]
    pub fn with_max_width(mut self, max_width: u32) -> Self {
        self.max_width = max_width;
        self
    }

    /// Set the maximum decoded height, in pixels, returning the updated
    /// limits.
    #[must_use]
    pub fn with_max_height(mut self, max_height: u32) -> Self {
        self.max_height = max_height;
        self
    }

    /// Set the single-axis [`max_coord`](DecodeLimits::max_coord) ceiling,
    /// in pixels per axis, returning the updated limits.
    #[must_use]
    pub fn with_max_coord(mut self, max_coord: u32) -> Self {
        self.max_coord = max_coord;
        self
    }

    /// Set the maximum total pixel count (`width * height`), returning the
    /// updated limits.
    #[must_use]
    pub fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    /// Set the maximum number of bytes the decoder may allocate at one
    /// time, returning the updated limits.
    #[must_use]
    pub fn with_max_alloc_bytes(mut self, max_alloc_bytes: u64) -> Self {
        self.max_alloc_bytes = max_alloc_bytes;
        self
    }

    /// Set the maximum number of pages a multi-page file may declare,
    /// returning the updated limits.
    #[must_use]
    pub fn with_max_pages(mut self, max_pages: u32) -> Self {
        self.max_pages = max_pages;
        self
    }

    /// Translate into the `image` crate's strict/non-strict limit set.
    fn to_image_limits(self) -> Limits {
        let mut limits = Limits::no_limits();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_alloc_bytes);
        limits
    }

    /// Enforce the single-axis [`max_coord`](DecodeLimits::max_coord)
    /// ceiling on untrusted header geometry before a [`Raster`] is built.
    /// Applied by both decode paths — the native `.v` reader in
    /// [`crate::imageio`] and the `image`-crate raster path (via
    /// [`decode_reader`], on the decoder's declared dimensions before the
    /// frame is allocated) — so the ceiling the `.v` decoder once read from
    /// a mutable process-global is now the same per-decode budget every
    /// decoder honours. Returns the typed
    /// [`SourceError::CoordLimitExceeded`] on an over-ceiling dimension
    /// rather than allowing a later wrapping cast or oversized allocation.
    pub(crate) fn check_coord(self, width: u32, height: u32) -> Result<(), SourceError> {
        if width > self.max_coord || height > self.max_coord {
            return Err(SourceError::CoordLimitExceeded {
                width,
                height,
                max_coord: self.max_coord,
            });
        }
        Ok(())
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

    /// Enforce [`max_alloc_bytes`](DecodeLimits::max_alloc_bytes) on a
    /// single buffer a decoder is about to reserve, named by `what` so the
    /// error says which one. `check_pixels` cannot stand in for this: it
    /// counts pixels and so sees neither the band count nor the sample
    /// depth, and the default 1-gigapixel ceiling still permits a 4 GiB
    /// `Rgba8` frame. Crate-visible so the format decoders that do their own
    /// reads (the TIFF page readers) apply the published budget rather than
    /// falling back to [`crate::raster::Raster::new`]'s much looser one.
    pub(crate) fn check_alloc(
        self,
        what: &'static str,
        needed_bytes: u64,
    ) -> Result<(), SourceError> {
        if needed_bytes > self.max_alloc_bytes {
            return Err(SourceError::AllocLimitExceeded {
                what,
                needed_bytes,
                max_alloc_bytes: self.max_alloc_bytes,
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
/// Reads the file at `path`, auto-detects the format, and decodes it into
/// an in-memory [`Raster`] with a canonical [`PixelFormat`]. Palette and
/// gray+alpha images are promoted to RGB/RGBA so that downstream code only
/// needs to handle a small set of uniform formats.
///
/// The format is identified from the file's leading magic bytes and never
/// from its extension, so this returns exactly what [`decode_bytes`] returns
/// for the same bytes, and a misnamed file still decodes correctly. libvips
/// resolves a loader the same way. Native `.v`, JPEG, PNG, TIFF, GIF,
/// WebP, and Radiance are recognised directly; anything else falls back to
/// the `image` crate's own content guess.
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

// ---------------------------------------------------------------------------
// Format sniffing and routing
// ---------------------------------------------------------------------------

/// Number of leading bytes read to identify a container.
///
/// Sized for the whole format wave rather than for the magics libviprs
/// reads today: WebP's `RIFF????WEBP` needs 12 (bytes 4..8 are a chunk
/// length and carry no signature) and Radiance's `#?RADIANCE` needs 10. It
/// is also exactly how many bytes `image`'s own `with_guessed_format` reads,
/// so the fallback in [`reader_for`] never sees more of a file than
/// [`sniff`] did.
const SNIFF_HEAD_LEN: usize = 16;

/// A container libviprs identifies from the leading magic bytes of a file
/// or buffer.
///
/// Only containers this build can actually reach a decoder for are listed;
/// an unrecognised one is `None` from [`sniff`] and falls through to
/// `image`'s own content guess. Growing this list is how a format lane
/// joins the routing: add a variant, its magic in [`sniff`], and its
/// decoder in [`SniffedFormat::image_format`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SniffedFormat {
    /// Native libvips `.v`, either byte order.
    Vips,
    /// JPEG (JFIF/EXIF), `FF D8 FF`.
    Jpeg,
    /// PNG, `89 P N G 0D 0A 1A 0A`.
    Png,
    /// TIFF, little-endian `II*\0` or big-endian `MM\0*`.
    Tiff,
    /// GIF, `GIF87a` or `GIF89a`.
    Gif,
    /// WebP, `RIFF` + a 4-byte length + `WEBP`.
    WebP,
    /// Radiance HDR, the first line `#?RADIANCE`.
    Radiance,
    /// FITS, the first card's `SIMPLE  =` keyword and fixed-format marker.
    Fits,
}

impl SniffedFormat {
    /// Whether [`decode_file_with_limits`] has to read the whole file into
    /// memory rather than streaming it.
    ///
    /// True for the containers libviprs parses itself, and for reasons that
    /// belong to the decoder rather than to the format:
    ///
    /// * `.v`, because [`crate::imageio::decode_vips_bytes`] parses the
    ///   libvips header and the metadata trailer itself and needs the buffer
    ///   addressable end to end.
    /// * JPEG, because the metadata pass rescans the APP1/APP2 segments for
    ///   EXIF and ICC after the pixel decode.
    /// * Radiance, because [`crate::radiance::decode_radiance`] walks the
    ///   header lines and the run-length-encoded body over one addressable
    ///   buffer.
    /// * GIF, because [`crate::gif::decode_gif`] has to scan every frame's
    ///   metadata before it can size the output — the band count depends on
    ///   whether *any* frame declares transparency — and then rewind to
    ///   decode frame 0. vips does exactly the same thing, and with the same
    ///   consequence: `vips_foreign_load_nsgif_header` opens with
    ///   `vips_source_map(gif->source, &size)`, mapping the whole file.
    /// * WebP, because [`crate::webp::decode_webp`] reads the `ICCP`,
    ///   `EXIF` and `XMP ` chunks out of the RIFF directory as well as the
    ///   frame, and the frame is rarely the last chunk in the file.
    /// * FITS, because [`crate::fits::decode_fits`] may walk past one or
    ///   more header units before it finds the one carrying the image, and
    ///   the sample array is band-planar and stored bottom row first, so
    ///   the decode reads it in an order no strip reader would.
    ///
    /// Everything else keeps the streaming reader, so widening the table
    /// above cannot quietly turn a streaming decode into a whole-file read.
    const fn decodes_from_memory(self) -> bool {
        matches!(
            self,
            Self::Vips | Self::Jpeg | Self::Gif | Self::WebP | Self::Radiance | Self::Fits
        )
    }

    /// The `image` decoder for this container, or `None` for `.v`, GIF,
    /// WebP and Radiance, which libviprs decodes itself.
    ///
    /// This is the entire route table, and it is an identity mapping on
    /// purpose: one sniffed container in, one decoder out, no per-format
    /// options. Save-side options live in the per-format modules
    /// ([`crate::webp`], [`crate::gif`]); nothing about how a file is
    /// decoded should ever need to be configured here.
    const fn image_format(self) -> Option<image::ImageFormat> {
        match self {
            Self::Vips => None,
            Self::Jpeg => Some(image::ImageFormat::Jpeg),
            Self::Png => Some(image::ImageFormat::Png),
            Self::Tiff => Some(image::ImageFormat::Tiff),
            // `image`'s GIF route is reachable but not usable for parity:
            // `GifDecoder::color_type()` is hard-coded to `Rgba8`, where
            // `vips gifload` emits three bands unless some frame declares a
            // transparent index, and the facade surfaces none of the fields
            // `gifload` attaches. [`crate::gif`] drives the `gif` crate
            // directly instead (issue #570).
            Self::Gif => None,
            // The `image` facade's WebP decoder reports neither the frame
            // count nor the XMP chunk, and [`crate::webp`] needs both, so
            // that module drives `image-webp` directly (issue #567).
            Self::WebP => None,
            // `image`'s Radiance route is behind its `hdr` feature, which
            // this build deliberately leaves off: the crate decodes RGBE as
            // `mantissa * 2^(e-136)` where vips uses the half-bit-centred
            // `(mantissa + 0.5) * 2^(e-136)`, a 100% error at mantissa 0.
            // [`crate::radiance`] hand-rolls the codec instead.
            Self::Radiance => None,
            // `image` has no FITS route at all, and no FITS crate models
            // the vips-side behaviour libviprs needs (the vertical flip,
            // the `fits-N` records, cfitsio's equivalent-type table), so
            // [`crate::fits`] hand-rolls the codec (issue #505).
            Self::Fits => None,
        }
    }
}

/// Identify a container from its leading bytes.
///
/// This is the one detector both decode entry points consult. libvips does
/// the same thing in `vips_foreign_find_load` (`foreign.c`), asking each
/// loader's `is_a` in priority order and never trusting the filename.
///
/// `head` may be shorter than [`SNIFF_HEAD_LEN`]; a buffer too short for a
/// given magic simply does not match it. The byte patterns are the same
/// ones `image` 0.25 keeps in its `MAGIC_BYTES` table
/// (`io/free_functions.rs`), so this sniff and the fallback guess in
/// [`reader_for`] cannot disagree about the same file.
pub(crate) fn sniff(head: &[u8]) -> Option<SniffedFormat> {
    if crate::imageio::is_vips_bytes(head) {
        return Some(SniffedFormat::Vips);
    }
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(SniffedFormat::Jpeg);
    }
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(SniffedFormat::Png);
    }
    if head.starts_with(b"II*\x00") || head.starts_with(b"MM\x00*") {
        return Some(SniffedFormat::Tiff);
    }
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        return Some(SniffedFormat::Gif);
    }
    // The one masked magic: bytes 4..8 are the RIFF chunk length, which is
    // file-specific, so the signature is split either side of it.
    if head.len() >= 12 && head.starts_with(b"RIFF") && &head[8..12] == b"WEBP" {
        return Some(SniffedFormat::WebP);
    }
    // Radiance's magic is a whole *line*, not a prefix: `vips__rad_israd`
    // (`radiance.c:568-577`) reads the first line and compares it to
    // `#?RADIANCE` in full, so the near-miss `#?RGBE` is not Radiance and
    // neither is `#?RADIANCEX`.
    let magic = crate::radiance::MAGIC;
    if head.len() > magic.len()
        && head.starts_with(magic)
        && matches!(head[magic.len()], b'\n' | b'\r')
    {
        return Some(SniffedFormat::Radiance);
    }
    // FITS has no signature to speak of: the standard fixes the primary
    // header's first card as `SIMPLE` with a logical value, so the keyword
    // field and the fixed-format `= ` in columns 9 and 10 are the only
    // bytes every file shares. vips does not sniff at all here, it hands
    // the file to `fits_open_diskfile` (`fits.c:526-548`).
    if head.starts_with(crate::fits::MAGIC) {
        return Some(SniffedFormat::Fits);
    }
    None
}

/// Read up to [`SNIFF_HEAD_LEN`] leading bytes, returning the buffer and how
/// many bytes were actually filled.
///
/// A source shorter than the head is not an error: [`sniff`] is given only
/// the filled prefix and matches nothing it cannot see.
fn read_head<R: std::io::Read>(mut source: R) -> std::io::Result<([u8; SNIFF_HEAD_LEN], usize)> {
    let mut head = [0u8; SNIFF_HEAD_LEN];
    let mut filled = 0;
    while filled < head.len() {
        let n = source.read(&mut head[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    Ok((head, filled))
}

/// Configure an [`ImageReader`] over an already-opened source for a sniffed
/// container.
///
/// This is the single place either entry point turns bytes into a reader
/// that knows its format. A container [`sniff`] recognised has its format
/// set directly; anything else falls through to `image`'s own content guess
/// over the same leading bytes. Either way the answer comes from the
/// content.
///
/// The path extension is deliberately never consulted. `ImageReader::open`
/// resolves the format from the extension alone and never reads the file,
/// which is exactly why the file and in-memory entry points used to give
/// two different answers for one run of bytes (issue #563). Taking an
/// already-opened reader instead of a path makes that mistake unavailable.
fn reader_for<R: std::io::BufRead + std::io::Seek>(
    inner: R,
    sniffed: Option<SniffedFormat>,
) -> Result<ImageReader<R>, SourceError> {
    let mut reader = ImageReader::new(inner);
    match sniffed.and_then(SniffedFormat::image_format) {
        Some(format) => {
            reader.set_format(format);
            Ok(reader)
        }
        None => Ok(reader.with_guessed_format()?),
    }
}

/// Decode an image file into a [`Raster`] under explicit [`DecodeLimits`].
///
/// Identical to [`decode_file`] but lets the caller supply the
/// dimension/allocation budget instead of using [`DecodeLimits::default`].
/// The limits are configured on the decoder before any pixel data is
/// allocated, and the `width * height` ceiling is checked before the
/// [`Raster`] is constructed.
///
/// # Errors
///
/// As [`decode_file`], plus [`SourceError::DimensionLimitExceeded`] when
/// the decoded `width * height` exceeds the supplied budget.
pub fn decode_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<Raster, SourceError> {
    // Identify the container from its leading magic, never from the path
    // extension: `decode_bytes_with_limits` has no filename to consult, so
    // any filename-derived answer here is one the two entry points cannot
    // both give (issue #563).
    let mut file = std::fs::File::open(path)?;
    let (head, filled) = read_head(&mut file)?;
    let sniffed = sniff(&head[..filled]);
    let mut raster = if sniffed.is_some_and(SniffedFormat::decodes_from_memory) {
        decode_bytes_with_limits(&std::fs::read(path)?, limits)?
    } else {
        // Rewind past the sniff and keep reading from the same handle, so
        // every streaming format's memory profile is unchanged.
        file.seek(std::io::SeekFrom::Start(0))?;
        decode_reader(reader_for(std::io::BufReader::new(file), sniffed)?, limits)?
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
    let sniffed = sniff(bytes);
    if sniffed == Some(SniffedFormat::Vips) {
        return crate::imageio::decode_vips_bytes(bytes, limits);
    }
    if sniffed == Some(SniffedFormat::Radiance) {
        return crate::radiance::decode_radiance(bytes, limits);
    }
    if sniffed == Some(SniffedFormat::Gif) {
        return crate::gif::decode_gif(bytes, limits);
    }
    if sniffed == Some(SniffedFormat::WebP) {
        return crate::webp::decode_webp(bytes, limits);
    }
    if sniffed == Some(SniffedFormat::Fits) {
        return crate::fits::decode_fits(bytes, limits);
    }
    let reader = reader_for(Cursor::new(bytes), sniffed)?;
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
    let image_limits = limits.to_image_limits();
    reader.limits(image_limits.clone());
    // Split what `ImageReader::decode` does in one call (build decoder →
    // reserve the output buffer → materialise pixels) so the single-axis
    // `max_coord` ceiling is enforced on the untrusted header geometry
    // *before* the frame is allocated, matching the native `.v` reader.
    // Without this the `image`-crate decoders (PNG/JPEG/TIFF) honoured only
    // `max_width`/`max_height`/`max_alloc_bytes` and silently ignored
    // `max_coord` (libviprs#349). An over-ceiling declared dimension now
    // returns the same typed `SourceError::CoordLimitExceeded` on every
    // raster decoder instead of decoding anyway.
    let mut decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    limits.check_coord(width, height)?;
    // Also enforce the total-pixel ceiling on the declared header geometry
    // here, before the output buffer is reserved, for true pre-allocation
    // parity with the `.v` reader (`build_raster` re-verifies it after
    // materialisation as a defence-in-depth backstop). Strictly tightens:
    // an over-`max_pixels` image is now rejected ahead of allocation rather
    // than only after the frame is decoded.
    limits.check_pixels(width, height)?;
    // Drift guard: this block hand-rolls the sequence `image` 0.25's
    // `ImageReader::decode` runs internally — build decoder → `reserve` the
    // output-buffer budget → `set_limits` → `from_decoder` — so that we can
    // slot the `max_coord` / `max_pixels` checks in ahead of the allocation.
    // It assumes that ordering (reserve-then-set_limits, with `total_bytes`
    // as the reserve size) and that `from_decoder` honours the limits set on
    // the decoder. If a future `image` release reorders these steps or
    // changes what `decode` reserves, revisit this and the allocation guard
    // below; a plain `reader.decode()` would silently drop the pre-alloc
    // coordinate/pixel enforcement.
    //
    // Preserve `decode`'s allocation guard: refuse to reserve an output
    // buffer larger than the decode budget permits before the pixel copy.
    let mut alloc_limits = image_limits;
    alloc_limits.reserve(decoder.total_bytes())?;
    decoder.set_limits(alloc_limits)?;
    let img = image::DynamicImage::from_decoder(decoder)?;
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
    for &[luma, alpha] in samples.as_chunks::<2>().0 {
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

    /// `DecodeLimits` is `#[non_exhaustive]`, so external callers customise
    /// it through the `with_*` builder setters rather than a struct literal.
    /// Each setter overrides exactly its own field and leaves the rest at
    /// their [`Default`] values, and `check_coord` reports an over-ceiling
    /// dimension through the dedicated typed [`SourceError::CoordLimitExceeded`]
    /// variant (issue #349 / #338 follow-up) instead of an opaque
    /// [`SourceError::VipsFormat`] string.
    #[test]
    fn decode_limits_builder_and_typed_coord_error() {
        let d = DecodeLimits::default();

        // (a) Builder setters override only their own field.
        let tuned = DecodeLimits::default()
            .with_max_width(11)
            .with_max_height(22)
            .with_max_coord(33)
            .with_max_pixels(44)
            .with_max_alloc_bytes(55);
        assert_eq!(tuned.max_width, 11);
        assert_eq!(tuned.max_height, 22);
        assert_eq!(tuned.max_coord, 33);
        assert_eq!(tuned.max_pixels, 44);
        assert_eq!(tuned.max_alloc_bytes, 55);

        // A single setter leaves every other ceiling at the default.
        let only_coord = DecodeLimits::default().with_max_coord(1000);
        assert_eq!(only_coord.max_coord, 1000);
        assert_eq!(only_coord.max_width, d.max_width);
        assert_eq!(only_coord.max_height, d.max_height);
        assert_eq!(only_coord.max_pixels, d.max_pixels);
        assert_eq!(only_coord.max_alloc_bytes, d.max_alloc_bytes);

        // (b) check_coord returns the typed variant carrying the offending
        // dimensions and the ceiling; the over-axis may be width or height.
        assert!(matches!(
            only_coord.check_coord(1001, 10),
            Err(SourceError::CoordLimitExceeded {
                width: 1001,
                height: 10,
                max_coord: 1000,
            })
        ));
        assert!(matches!(
            only_coord.check_coord(10, 5000),
            Err(SourceError::CoordLimitExceeded {
                width: 10,
                height: 5000,
                max_coord: 1000,
            })
        ));

        // (c) At or below the ceiling is accepted.
        assert!(only_coord.check_coord(1000, 1000).is_ok());
        assert!(only_coord.check_coord(0, 0).is_ok());
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

    /// Encode a `w x h` RGB image as GIF, so the sniff table's GIF arm and
    /// the `image` `gif` feature are both exercised on real bytes.
    fn create_test_gif(w: u32, h: u32) -> Vec<u8> {
        encode_via_image(w, h, image::ImageFormat::Gif)
    }

    /// Encode a `w x h` RGB image as WebP. This is the fixture the widened
    /// sniff head exists for: the `RIFF????WEBP` magic is 12 bytes with a
    /// file-specific length in the middle, so the old 4-byte head could not
    /// have identified it.
    fn create_test_webp(w: u32, h: u32) -> Vec<u8> {
        encode_via_image(w, h, image::ImageFormat::WebP)
    }

    /// Shared body of the GIF and WebP fixtures: a deterministic RGB ramp
    /// written out in `format`.
    fn encode_via_image(w: u32, h: u32, format: image::ImageFormat) -> Vec<u8> {
        let mut buf: image::RgbImage = image::ImageBuffer::new(w, h);
        for (x, y, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 20) as u8, (y * 30) as u8, 90]);
        }
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut Cursor::new(&mut out), format)
            .unwrap();
        out
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
     * Guards the shared format sniff: `decode_file_with_limits` and
     * `decode_bytes_with_limits` must identify a container from the same
     * evidence, so a filename can never change what a given run of bytes
     * decodes to. Works by writing six buffers to disk under names that
     * disagree with their content — a PNG called `.jpg`, a JPEG called
     * `.png`, a PNG with no extension at all, and native `.v`, GIF, and
     * WebP bytes all called `.png` — then decoding each through both entry
     * points and
     * comparing every case before reporting, so one broken route does not
     * hide the others. Before the shared sniff the file entry point
     * resolved the format from the path extension (`ImageReader::open`)
     * while the byte entry point resolved it from the content
     * (`with_guessed_format`), so the same bytes decoded two different ways
     * depending on which entry point the caller reached for (issue #563).
     * The WebP case is also the one that needs the widened sniff head: its
     * `RIFF????WEBP` magic is 12 bytes with a file-specific length in the
     * middle, so four bytes could never have identified it.
     * Input: six mislabelled PNG/JPEG/`.v`/GIF/WebP files → Output: both
     * entry points return equal dimensions, pixel format, and pixel bytes.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn content_beats_extension_in_both_entry_points() {
        /// Compact one-line rendering of a decode outcome: the raster's
        /// shape on success, the error message on failure. Keeps the
        /// assertion message readable where a `{:?}` of the raster would
        /// dump the whole pixel buffer.
        fn outcome(result: &Result<Raster, SourceError>) -> String {
            match result {
                Ok(im) => format!("Ok({}x{} {:?})", im.width(), im.height(), im.format()),
                Err(e) => format!("Err({e})"),
            }
        }

        let png = create_test_png(9, 7);
        let jpeg = create_test_jpeg(9, 7);
        let gif = create_test_gif(9, 7);
        let webp = create_test_webp(9, 7);
        let vips = decode_bytes(&png).unwrap().encode_vips().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut disagreements: Vec<String> = Vec::new();
        // Each case is (file name, bytes): the name is picked to disagree
        // with the magic, so extension-based routing cannot get it right.
        for (name, bytes) in [
            ("png_bytes_named.jpg", &png),
            ("jpeg_bytes_named.png", &jpeg),
            ("png_bytes_with_no_extension", &png),
            ("vips_bytes_named.png", &vips),
            ("gif_bytes_named.png", &gif),
            ("webp_bytes_named.png", &webp),
        ] {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();

            let from_bytes = decode_bytes_with_limits(bytes, DecodeLimits::default());
            let from_file = decode_file_with_limits(&path, DecodeLimits::default());

            match (&from_file, &from_bytes) {
                (Ok(f), Ok(b)) => {
                    if (f.width(), f.height(), f.format()) != (b.width(), b.height(), b.format())
                        || f.data() != b.data()
                    {
                        disagreements.push(format!(
                            "{name}: decode_file_with_limits {} vs decode_bytes_with_limits {} \
                             (pixel bytes equal: {})",
                            outcome(&from_file),
                            outcome(&from_bytes),
                            f.data() == b.data()
                        ));
                    }
                }
                _ => disagreements.push(format!(
                    "{name}: decode_file_with_limits {} vs decode_bytes_with_limits {}",
                    outcome(&from_file),
                    outcome(&from_bytes)
                )),
            }
        }

        assert!(
            disagreements.is_empty(),
            "the file and byte entry points disagree on {} of 6 inputs:\n  {}",
            disagreements.len(),
            disagreements.join("\n  ")
        );
    }

    /**
     * Pins the sniff table itself: every magic libviprs routes on maps to
     * exactly one container, and nothing else does. Works by running each
     * known magic (both TIFF byte orders, both GIF versions, both `.v`
     * byte orders) plus a set of near-misses through `sniff` and comparing
     * against the expected variant. The WebP case is the one that needs
     * more than four leading bytes: its signature is split either side of
     * a file-specific chunk length, which is why the head is
     * `SNIFF_HEAD_LEN` and not the four bytes it used to be (issue #563).
     * Input: one buffer per magic plus five non-matches -> Output: the
     * expected `Option<SniffedFormat>` for each.
     */
    #[test]
    fn sniff_maps_each_magic_to_one_container() {
        let cases: [(&str, &[u8], Option<SniffedFormat>); 24] = [
            (
                "vips le",
                &[0xb6, 0xa6, 0xf2, 0x08],
                Some(SniffedFormat::Vips),
            ),
            (
                "vips be",
                &[0x08, 0xf2, 0xa6, 0xb6],
                Some(SniffedFormat::Vips),
            ),
            (
                "jpeg jfif",
                &[0xFF, 0xD8, 0xFF, 0xE0],
                Some(SniffedFormat::Jpeg),
            ),
            (
                "jpeg exif",
                &[0xFF, 0xD8, 0xFF, 0xE1],
                Some(SniffedFormat::Jpeg),
            ),
            ("png", b"\x89PNG\r\n\x1a\n", Some(SniffedFormat::Png)),
            ("tiff le", b"II*\x00", Some(SniffedFormat::Tiff)),
            ("tiff be", b"MM\x00*", Some(SniffedFormat::Tiff)),
            ("gif87a", b"GIF87a", Some(SniffedFormat::Gif)),
            ("gif89a", b"GIF89a", Some(SniffedFormat::Gif)),
            // Bytes 4..8 are the RIFF chunk length and are deliberately
            // arbitrary here: the sniff must ignore them entirely.
            (
                "webp",
                b"RIFF\x2a\x13\x00\x00WEBPVP8 ",
                Some(SniffedFormat::WebP),
            ),
            ("riff but not webp", b"RIFF\x00\x00\x00\x00WAVEfmt ", None),
            // Truncated one byte short of the `WEBP` tag: a 12-byte magic
            // cannot be decided from 11 bytes.
            ("webp truncated", b"RIFF\x00\x00\x00\x00WEB", None),
            // Radiance's magic is a whole line, not a prefix, so the two
            // near-misses below must not match: `vips__rad_israd` compares
            // the first line to `#?RADIANCE` in full.
            (
                "radiance",
                b"#?RADIANCE\nFORMAT=",
                Some(SniffedFormat::Radiance),
            ),
            (
                "radiance dos",
                b"#?RADIANCE\r\nFORMAT=",
                Some(SniffedFormat::Radiance),
            ),
            ("radiance rgbe", b"#?RGBE\nFORMAT=", None),
            ("radiance longer first line", b"#?RADIANCEX\n", None),
            ("radiance with no newline", b"#?RADIANCE", None),
            // FITS is a fixed-width prefix: the keyword field, the `=` in
            // column 9 and the space after it. `SIMPLE=T` is a legal FITS
            // value written in free format, but no conforming file opens
            // that way, and cfitsio refuses it too.
            (
                "fits",
                b"SIMPLE  =                    T",
                Some(SniffedFormat::Fits),
            ),
            ("fits free format", b"SIMPLE=T", None),
            ("fits without the keyword padding", b"SIMPLE = T", None),
            ("fits truncated", b"SIMPLE  ", None),
            ("plain text", b"not an image at all", None),
            ("empty", b"", None),
            ("one byte of png", b"\x89", None),
        ];
        for (name, head, expected) in cases {
            assert_eq!(sniff(head), expected, "sniff disagreed on {name}");
        }
    }

    /**
     * Pins the two contracts the route table has to keep as the format
     * lanes extend it. First, the memory profile: only the containers
     * whose decoders parse the container themselves read a whole in-memory
     * buffer, and every other format keeps the streaming reader. A lane
     * that adds a variant and gets this wrong silently turns a streaming
     * decode into a read-whole-file. Second, the mapping is identity: one
     * container in, one decoder out, no options and no many-to-one
     * collapsing.
     * Input: every `SniffedFormat` variant -> Output: `decodes_from_memory`
     * true for exactly `Vips`, `Jpeg`, `Gif`, `WebP`, `Radiance` and
     * `Fits`, and distinct `image` formats for every container libviprs
     * does not decode itself.
     */
    #[test]
    fn route_table_is_identity_and_only_self_decoded_formats_buffer_whole() {
        let all = [
            SniffedFormat::Vips,
            SniffedFormat::Jpeg,
            SniffedFormat::Png,
            SniffedFormat::Tiff,
            SniffedFormat::Gif,
            SniffedFormat::WebP,
            SniffedFormat::Radiance,
            SniffedFormat::Fits,
        ];
        // These are the containers libviprs decodes itself, so they are
        // the ones the route table maps to no `image` decoder.
        let self_decoded = [
            SniffedFormat::Vips,
            SniffedFormat::Radiance,
            SniffedFormat::Gif,
            SniffedFormat::WebP,
            SniffedFormat::Fits,
        ];

        let buffered: Vec<SniffedFormat> = all
            .iter()
            .copied()
            .filter(|f| f.decodes_from_memory())
            .collect();
        assert_eq!(
            buffered,
            vec![
                SniffedFormat::Vips,
                SniffedFormat::Jpeg,
                SniffedFormat::Gif,
                SniffedFormat::WebP,
                SniffedFormat::Radiance,
                SniffedFormat::Fits
            ],
            "only .v, JPEG, GIF, WebP, Radiance and FITS may read the whole file into memory"
        );

        for format in self_decoded {
            assert_eq!(
                format.image_format(),
                None,
                "{format:?} is decoded by libviprs itself, not by the image facade"
            );
        }
        let mut mapped: Vec<image::ImageFormat> = all
            .iter()
            .copied()
            .filter_map(SniffedFormat::image_format)
            .collect();
        assert_eq!(
            mapped.len(),
            all.len() - self_decoded.len(),
            "every container libviprs does not decode itself maps to an image decoder"
        );
        mapped.sort_by_key(|f| format!("{f:?}"));
        mapped.dedup();
        assert_eq!(
            mapped.len(),
            all.len() - self_decoded.len(),
            "the route table must be an identity mapping, not many-to-one"
        );
    }

    /**
     * Verifies that a FITS file reaches the hand-rolled codec through both
     * public decode entry points, and that neither consults the file name
     * to get there. Works by writing one `.fits` under a misleading `.png`
     * extension and decoding it from the path and from the bytes.
     * Input: a 4x1 BITPIX 8 file named `misnamed.png` -> Output: the same
     * `Gray8` raster from `decode_file` and `decode_bytes`, right way up.
     */
    #[test]
    fn fits_reaches_its_codec_from_both_entry_points() {
        let raster = Raster::new(4, 1, PixelFormat::Gray8, vec![3, 1, 4, 1]).unwrap();
        let file = raster.encode_fits().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("misnamed.png");
        std::fs::write(&path, &file).unwrap();

        let from_path = decode_file_with_limits(&path, DecodeLimits::default()).unwrap();
        let from_bytes = decode_bytes(&file).unwrap();
        for decoded in [&from_path, &from_bytes] {
            assert_eq!((decoded.width(), decoded.height()), (4, 1));
            assert_eq!(decoded.format(), PixelFormat::Gray8);
            assert_eq!(decoded.data(), &[3, 1, 4, 1]);
        }
    }

    /**
     * Verifies that a Radiance file reaches the hand-rolled codec through
     * both public decode entry points, and that neither consults the file
     * name to get there. Works by writing one `.hdr` under a misleading
     * `.png` extension and decoding it from the path and from the bytes.
     * Input: a 6x1 Radiance file named `misnamed.png` -> Output: the same
     * `FloatF32(3)` raster from `decode_file` and `decode_bytes`, with the
     * first pixel at the half-bit value vips prints.
     */
    #[test]
    fn radiance_reaches_its_codec_from_both_entry_points() {
        let mut file = Vec::new();
        file.extend_from_slice(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 6\n");
        for i in 0..6u8 {
            file.extend_from_slice(&[255, 128, 64, 128 + i]);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("misnamed.png");
        std::fs::write(&path, &file).unwrap();

        let from_path = decode_file_with_limits(&path, DecodeLimits::default()).unwrap();
        let from_bytes = decode_bytes(&file).unwrap();
        for raster in [&from_path, &from_bytes] {
            assert_eq!((raster.width(), raster.height()), (6, 1));
            assert_eq!(raster.format().channels(), 3);
            assert!(raster.format().is_float());
        }
        assert_eq!(from_path.data(), from_bytes.data());

        let first = from_bytes.getpoint(0, 0);
        assert!(
            (first[0] - 0.998046875).abs() < 1e-9,
            "the half-bit decode constant, got {}",
            first[0]
        );
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
