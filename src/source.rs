use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};
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
    /// A malformed or unsupported OpenEXR file, raised by
    /// [`crate::exr::decode_exr`]. libviprs decodes EXR through the `exr`
    /// crate rather than through the `image` facade (see [`crate::exr`]),
    /// so the failure arrives as a typed
    /// [`ExrError`](crate::exr::ExrError) rather than as an
    /// [`image::ImageError`].
    #[error(transparent)]
    Exr(#[from] crate::exr::ExrError),
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
    /// A malformed or unsupported JPEG XL file, raised by
    /// [`crate::jxl::decode_jxl`]. libviprs decodes JPEG XL through
    /// `jxl-oxide` rather than through the `image` facade, which has no
    /// JPEG XL variant at all (see [`crate::jxl`]), so its failures arrive
    /// as the codec's own typed [`JxlError`](crate::jxl::JxlError) rather
    /// than as an `image::ImageError` carrying a hand-spelled format hint.
    ///
    /// The variant is declared whether or not the **`jxl`** feature is on,
    /// so `SourceError` has the same shape in both builds. Without the
    /// feature the only [`JxlError`](crate::jxl::JxlError) it can carry is
    /// [`FeatureNotEnabled`](crate::jxl::JxlError::FeatureNotEnabled),
    /// which is how a caller tells "this build has no JPEG XL" from "these
    /// bytes are not JPEG XL" without reading a message (issue #634).
    #[error(transparent)]
    Jxl(#[from] crate::jxl::JxlError),
    /// A malformed or unreachable NIfTI file. libviprs decodes NIfTI
    /// itself rather than through any crate (see [`crate::nifti`]), so its
    /// failures arrive as the codec's own typed
    /// [`NiftiError`](crate::nifti::NiftiError) rather than as an opaque
    /// string. As with FITS, that matters here because a NIfTI file can be
    /// perfectly well formed and still declare a sample type this build has
    /// no pixel format for; the variant says which.
    #[error(transparent)]
    Nifti(#[from] crate::nifti::NiftiError),
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
    /// frame. Raised for the whole-file read every memory-decoded container
    /// needs, with `what = "image file body"` (issue #629), and by the TIFF
    /// page readers for both their own file body and the pixel buffer a
    /// page decodes into.
    ///
    /// This is the shape **every decoder that prices a frame itself** uses.
    /// GIF, FITS, OpenEXR, Radiance and JPEG XL each used to report an
    /// `AllocLimitExceeded` of their own, five variants re-tagging a refusal
    /// computed by the same shared arithmetic; they were collapsed onto this
    /// one in issue #686, and [`geometry`](DeclaredGeometry) is what carries
    /// the width, height and band count they used to carry individually.
    ///
    /// It is still not the only shape the budget can refuse a file in, and
    /// the remainder is a real distinction rather than a leftover. JPEG,
    /// PNG, single-image TIFF and WebP are refused by the `image` crate's own
    /// budget from inside its decoder, and arrive as [`SourceError::Decode`]
    /// carrying an `image` `LimitError`; there is no libviprs price behind
    /// them and no declared geometry to report. JPEG XL can also trip
    /// `jxl-oxide`'s internal allocation tracker, which reports
    /// [`JxlError::DecoderAllocLimitExceeded`](crate::jxl::JxlError::DecoderAllocLimitExceeded)
    /// because it is a different ceiling biting on a buffer whose size the
    /// decoder does not report out.
    ///
    /// Use [`SourceError::is_alloc_limit`] to catch all three in one call
    /// rather than matching them.
    #[error(
        "{what}{} needs {needed_bytes} bytes, over the {max_alloc_bytes}-byte \
         allocation ceiling; raise DecodeLimits::max_alloc_bytes",
        ShowGeometry(*geometry)
    )]
    AllocLimitExceeded {
        /// What the allocation was for, e.g. `"TIFF file body"` or
        /// `"GIF canvas"`.
        ///
        /// A human-readable label for the message, **not** part of the
        /// compatibility promise: the wording may change in any release and
        /// new decoders add new labels. Branch on
        /// [`geometry`](DeclaredGeometry) or on the variant, never on this
        /// string.
        what: &'static str,
        /// The declared geometry the price was computed from, where the
        /// refusal priced an image. `None` where it priced a byte count with
        /// no image behind it, which is the whole-file read: a file's length
        /// on disk says nothing about what geometry it declares inside.
        geometry: Option<DeclaredGeometry>,
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

impl SourceError {
    /// Whether this is the decode allocation budget refusing the file.
    ///
    /// Answers in one call what issue #686 found took seven match arms. The
    /// budget can bite in three places and they are genuinely different
    /// checks, so they stay three variants and this is the predicate over
    /// them:
    ///
    /// * [`SourceError::AllocLimitExceeded`], every decoder that prices a
    ///   buffer against [`DecodeLimits::max_alloc_bytes`] itself;
    /// * [`SourceError::Decode`] carrying an `image` `LimitError` of kind
    ///   `InsufficientMemory`, which is the same ceiling spent inside the
    ///   `image` crate's own decoder for JPEG, PNG, single-image TIFF and
    ///   WebP;
    /// * [`JxlError::DecoderAllocLimitExceeded`](crate::jxl::JxlError::DecoderAllocLimitExceeded),
    ///   `jxl-oxide`'s internal allocation tracker refusing a buffer whose
    ///   size it does not report out.
    ///
    /// Raising `max_alloc_bytes` is the response to all three, which is what
    /// makes one predicate the right shape rather than a convenience over
    /// unrelated things. It is the *whole* test: a shape that is not fixed by
    /// raising that one knob is not this.
    ///
    /// So it does **not** cover [`SourceError::DimensionLimitExceeded`] or
    /// [`SourceError::PageLimitExceeded`], which are different ceilings with
    /// different remedies, and it does not cover
    /// `SourceError::Raster(RasterError::ByteBudgetExceeded)` either. That
    /// last one is worth naming because it *looks* like this: it says "needs N
    /// bytes, exceeding the M-byte allocation budget" and
    /// [`Raster::ppm_load`](crate::raster::Raster::ppm_load),
    /// `csv_load` and `matrix_load` all return it through this same enum. But
    /// the budget it names is
    /// [`DEFAULT_MAX_ALLOC_BYTES`](crate::raster::DEFAULT_MAX_ALLOC_BYTES),
    /// the raster **construction** ceiling, not
    /// [`DecodeLimits::max_alloc_bytes`], and raising the decode limit does
    /// nothing about it. There is a negative control on it.
    ///
    /// # Example
    ///
    /// ```
    /// # use libviprs::SourceError;
    /// fn is_too_big(err: &SourceError) -> bool {
    ///     err.is_alloc_limit()
    /// }
    /// ```
    #[must_use]
    pub fn is_alloc_limit(&self) -> bool {
        match self {
            SourceError::AllocLimitExceeded { .. } => true,
            SourceError::Decode(image::ImageError::Limits(e)) => {
                matches!(e.kind(), image::error::LimitErrorKind::InsufficientMemory)
            }
            // Deliberately NOT `#[cfg(feature = "jxl")]`. `JxlError` and this
            // variant are declared unconditionally so that "a caller's `match`
            // has the same arms in either build" (issue #634), and gating the
            // arm broke that promise for the predicate: the identical value
            // answered `false` without the feature and `true` with it. Features
            // are additive, so one crate in a workspace turning `jxl` on would
            // silently change another crate's error handling.
            SourceError::Jxl(crate::jxl::JxlError::DecoderAllocLimitExceeded { .. }) => true,
            _ => false,
        }
    }
}

/// Writes `" WxHxB"` for a geometry and nothing at all for `None`.
///
/// The point is the "nothing at all". `SourceError::AllocLimitExceeded`'s
/// message is built by `thiserror` in a `Display` impl, and the obvious
/// spelling there is a `format!` producing a `String`. That puts a heap
/// allocation on the path that formats an **allocation-refusal** error, which
/// is exactly where the host is least likely to serve one, and cuts against
/// the abort-freedom work in #627, #672 and #685. This writes straight into
/// the formatter instead.
struct ShowGeometry(Option<DeclaredGeometry>);

impl std::fmt::Display for ShowGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(g) => write!(f, " {}x{}x{}", g.width, g.height, g.bands),
            None => Ok(()),
        }
    }
}

/// The geometry a decoder priced a frame from, reported by
/// [`SourceError::AllocLimitExceeded`].
///
/// Every declared-geometry decoder prices its frame as
/// `width * height * bands * sample_bytes`, and these are the first three of
/// those four. The sample size is not reported separately because the price
/// already carries it and the three fields here are what a caller can compare
/// against what they expected the file to hold.
///
/// The band count is the one the *header declares*, which is not always the
/// band count of the raster a successful decode would have produced. OpenEXR
/// is the case that shows it: the decoder builds a full-resolution buffer for
/// every channel the header declares, so a file declaring sixteen channels and
/// selecting four is priced at sixteen and reports sixteen, while a successful
/// decode of it hands back four bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeclaredGeometry {
    /// Declared width in pixels.
    pub width: u32,
    /// Declared height in pixels.
    pub height: u32,
    /// Declared band count.
    pub bands: u32,
}

impl DeclaredGeometry {
    /// Build a geometry.
    ///
    /// The struct is `#[non_exhaustive]` so that a fourth field can be added
    /// without breaking a caller who reads the three, and that is exactly what
    /// stops a caller building one with a struct literal. Without this they
    /// could construct the degenerate `geometry: None` form of
    /// [`SourceError::AllocLimitExceeded`] and never the interesting one,
    /// which would make the variant untestable from outside the crate.
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
        }
    }
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
/// | [`max_alloc_bytes`](Self::max_alloc_bytes) | ✅ via [`image::Limits`], plus the whole-file read for a memory-decoded container | ✅ on the whole-file read (`.v`'s own uncompressed body is sized by its header, gated by `max_coord`/`max_pixels`) | ✅ on the file body, the pixel buffer, and the `tiff` decoder's own buffers |
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

    /// Answer whether a single buffer of `needed_bytes` is over
    /// [`max_alloc_bytes`](DecodeLimits::max_alloc_bytes), with no error
    /// attached.
    ///
    /// The arithmetic half of [`check_alloc`](Self::check_alloc), split out
    /// because five of the seven callers of the budget throw the
    /// `SourceError` away and report a per-format variant of their own, so
    /// the `what` label they had to pass was built at every call and never
    /// observable by anyone.
    ///
    /// `u64::MAX` is refused whatever the budget says, and that arm is the
    /// point of this function rather than a detail of it.
    /// [`decode_alloc_bytes`](crate::raster::decode_alloc_bytes) saturates
    /// there, so `u64::MAX` is the answer "this product did not fit a
    /// `u64`" rather than a price. Compared with `>` alone it clears the
    /// one budget a caller is most likely to set, since
    /// [`with_max_alloc_bytes(u64::MAX)`](Self::with_max_alloc_bytes) is
    /// the idiomatic spelling of "no limit", and the decoder then goes on
    /// to size a buffer from a number that was never the real one. Refusing
    /// it costs nothing real either way: 16 exbibytes is not an allocation
    /// any target can serve, so a geometry that priced there exactly is
    /// just as unservable as one that saturated.
    pub(crate) fn exceeds_alloc_budget(self, needed_bytes: u64) -> bool {
        needed_bytes > self.max_alloc_bytes || needed_bytes == u64::MAX
    }

    /// Enforce [`max_alloc_bytes`](DecodeLimits::max_alloc_bytes) on a
    /// single buffer a decoder is about to reserve, named by `what`, where
    /// the size is a byte count with no image behind it.
    ///
    /// That is the whole-file read: a file's length on disk says nothing
    /// about the geometry it declares inside, so the refusal carries no
    /// geometry. A decoder pricing a frame from a declared width, height and
    /// band count calls [`check_image_alloc`](Self::check_image_alloc)
    /// instead, which is the same ceiling reported with those three attached.
    ///
    /// `check_pixels` cannot stand in for either: it counts pixels and so
    /// sees neither the band count nor the sample depth, and the default
    /// 1-gigapixel ceiling still permits a 4 GiB `Rgba8` frame.
    /// Crate-visible so the format decoders that do their own reads (the
    /// TIFF page readers) apply the published budget rather than falling
    /// back to [`crate::raster::Raster::new`]'s much looser one.
    pub(crate) fn check_alloc(
        self,
        what: &'static str,
        needed_bytes: u64,
    ) -> Result<(), SourceError> {
        if self.exceeds_alloc_budget(needed_bytes) {
            return Err(SourceError::AllocLimitExceeded {
                what,
                geometry: None,
                needed_bytes,
                max_alloc_bytes: self.max_alloc_bytes,
            });
        }
        Ok(())
    }

    /// Price a frame from the geometry a header declares and enforce
    /// [`max_alloc_bytes`](DecodeLimits::max_alloc_bytes) on it, reporting
    /// the geometry alongside the price.
    ///
    /// One call where every declared-geometry decoder used to spell out
    /// [`decode_alloc_bytes`](crate::raster::decode_alloc_bytes), then
    /// [`exceeds_alloc_budget`](Self::exceeds_alloc_budget), then a variant
    /// of its own. #632 put the price and the comparison behind one
    /// implementation each; this puts the reporting behind one too, which is
    /// what stops the five drifting apart again (issue #686).
    ///
    /// `bands` and `sample_bytes` are separate arguments rather than one
    /// bytes-per-pixel product because the message says the band count, and
    /// because `decode_alloc_bytes` widens each multiplicand to `u64` before
    /// multiplying and saturates rather than wrapping. Handing it a product
    /// a caller already narrowed would give that up.
    ///
    /// They are `u64` for the same reason. Every saturation in this area goes
    /// **up**, to `u64::MAX`, which `exceeds_alloc_budget` treats as a sentinel
    /// refusal, so a caller who lifts every ceiling still gets refused rather
    /// than served a wrapped price. Narrowing a band count on the way in would
    /// saturate the price *down* instead, which is the one direction that
    /// turns a refusal into a decode. Only the geometry the refusal
    /// **reports** narrows to `u32`, and it saturates there because
    /// [`DeclaredGeometry`] holds a `u32`; a count that large understates in
    /// the message and cannot change the verdict, because the price it came
    /// from has already saturated.
    ///
    /// Returns the price on success, since several callers size a buffer
    /// from the same number straight afterwards.
    pub(crate) fn check_image_alloc(
        self,
        what: &'static str,
        width: u32,
        height: u32,
        bands: u64,
        sample_bytes: u64,
    ) -> Result<u64, SourceError> {
        let needed_bytes = crate::raster::decode_alloc_bytes(width, height, bands, sample_bytes);
        if self.exceeds_alloc_budget(needed_bytes) {
            return Err(SourceError::AllocLimitExceeded {
                what,
                geometry: Some(DeclaredGeometry {
                    width,
                    height,
                    bands: u32::try_from(bands).unwrap_or(u32::MAX),
                }),
                needed_bytes,
                max_alloc_bytes: self.max_alloc_bytes,
            });
        }
        Ok(needed_bytes)
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
/// WebP, JPEG XL, Radiance, FITS and OpenEXR are recognised directly; anything
/// else falls
/// back to the `image` crate's own content guess.
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
const SNIFF_HEAD_LEN: usize = 348;

/// The ISOBMFF signature box that opens a boxed JPEG XL file: a 12-byte box
/// whose type is `JXL ` and whose payload is the `\r\n\x87\n` line-ending
/// check (ISO/IEC 18181-2, and the first arm of libjxl's
/// `JxlSignatureCheck`).
const JXL_CONTAINER_MAGIC: &[u8] = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a";

/// The bare JPEG XL codestream's magic and the other arm of
/// `JxlSignatureCheck` (`jxlload.c:213-221`).
///
/// Two bytes is as short as any signature in the table gets, and JPEG XL is
/// the only container in it with two unrelated magics: a table that knew
/// only one of them would silently drop half the format.
const JXL_CODESTREAM_MAGIC: &[u8] = b"\xff\x0a";

/// One signature in the sniff table, as data rather than as a hand-written
/// byte test.
///
/// Three shapes cover every container libviprs routes, and there is more
/// than one because a magic is not always a leading prefix: WebP's is split
/// either side of a file-specific chunk length, and Radiance's is a whole
/// first *line*. Modelling that as data is what lets [`sniff`] be driven
/// from the route table, instead of from a second hand-ordered chain that
/// has to be kept in step with it (issue #633).
#[derive(Clone, Copy, Debug)]
enum Magic {
    /// The head opens with exactly these bytes.
    Prefix(&'static [u8]),
    /// The head opens with `prefix` and carries `tag` at byte `tag_at`; the
    /// bytes between the two are file-specific and are ignored. WebP's
    /// `RIFF????WEBP` is the only one, because bytes 4..8 are the RIFF chunk
    /// length and carry no signature.
    Split {
        /// The bytes at offset 0.
        prefix: &'static [u8],
        /// Where `tag` starts.
        tag_at: usize,
        /// The bytes at `tag_at`.
        tag: &'static [u8],
    },
    /// The head's whole first line is exactly these bytes, CR- or
    /// LF-terminated. Radiance's `#?RADIANCE` is the only one:
    /// `vips__rad_israd` (`radiance.c:568-577`) reads the first line and
    /// compares it in full, so the near-miss `#?RGBE` is not Radiance and
    /// neither is `#?RADIANCEX`.
    Line(&'static [u8]),
}

impl Magic {
    /// Whether `head` carries this signature.
    ///
    /// A head too short to decide the signature is not a match: [`sniff`] is
    /// handed only the bytes a source actually yielded, and a 12-byte magic
    /// answered from 11 bytes would be a guess.
    fn matches(self, head: &[u8]) -> bool {
        match self {
            Self::Prefix(magic) => {
                // `[].starts_with(&[])` is true, so an empty prefix matches
                // every buffer and would shadow every row declared after it.
                debug_assert!(!magic.is_empty(), "an empty Prefix matches every buffer");
                head.starts_with(magic)
            }
            Self::Split {
                prefix,
                tag_at,
                tag,
            } => {
                // Nothing else constrains these two to sit apart, and a row
                // where they overlap is quietly self-consistent rather than
                // rejected: `shortest_head` lays the prefix down and then
                // writes the tag over the end of it, and `matches` accepts
                // the result it just built. The row would be wrong and every
                // test that probes it with its own head would still pass.
                debug_assert!(!tag.is_empty(), "an empty Split tag constrains nothing");
                debug_assert!(
                    prefix.len() <= tag_at,
                    "a Split prefix that runs into its own tag is self-consistent and wrong"
                );
                head.len() >= tag_at + tag.len()
                    && head.starts_with(prefix)
                    && head[tag_at..tag_at + tag.len()] == *tag
            }
            Self::Line(magic) => {
                debug_assert!(
                    !magic.is_empty(),
                    "an empty Line magic matches every terminated head"
                );
                head.len() > magic.len()
                    && head.starts_with(magic)
                    && matches!(head[magic.len()], b'\n' | b'\r')
            }
        }
    }

    /// The shortest head this signature accepts.
    ///
    /// Lives beside [`Self::matches`] so the two cannot drift, and exists so
    /// the route-table tests can build a probe per row. They must not get
    /// one from a second table of sample bytes, because a second table kept
    /// in step by hand is exactly what this shape is retiring.
    #[cfg(test)]
    fn shortest_head(self) -> Vec<u8> {
        match self {
            Self::Prefix(magic) => magic.to_vec(),
            Self::Split {
                prefix,
                tag_at,
                tag,
            } => {
                let mut head = vec![0u8; tag_at + tag.len()];
                head[..prefix.len()].copy_from_slice(prefix);
                head[tag_at..].copy_from_slice(tag);
                head
            }
            Self::Line(magic) => [magic, b"\n"].concat(),
        }
    }
}

/// Where a sniffed container's bytes go to become pixels.
///
/// The difference between the first two arms is a memory profile rather
/// than a decoder: [`decode_file_with_limits`] streams a `Streamed` row past
/// the `image` facade and reads a `Buffered` or a `Native` one whole. That
/// read goes through [`read_file_bounded`], so a container joining either of
/// the latter two gets [`DecodeLimits::max_alloc_bytes`] applied to the file
/// length before a byte of it is read. It used to be a plain `std::fs::read`,
/// which meant every ceiling was checked after the whole file was already
/// resident (issue #629).
#[derive(Clone, Copy, Debug)]
enum Decoder {
    /// The `image` facade, over the streaming reader. Nothing else in the
    /// crate has to know the container exists.
    Streamed(image::ImageFormat),
    /// The `image` facade, but over the whole file in memory, because
    /// libviprs makes a second pass over the same bytes afterwards. JPEG is
    /// the only one: the metadata pass rescans the APP1/APP2 segments for
    /// EXIF and ICC after the pixel decode.
    Buffered(image::ImageFormat),
    /// A libviprs codec, over the whole file in memory. Every one of these
    /// parses the container itself and needs the bytes addressable end to
    /// end; the per-format reason is on the row.
    Native(fn(&[u8], DecodeLimits) -> Result<Raster, SourceError>),
}

/// One row of the route table: everything routing knows about a container.
///
/// This is the single place the routing data for a container lives. Four
/// sites used to carry it between them: its magic in `sniff`, its memory
/// profile in `decodes_from_memory`, its decoder in `image_format`, and its
/// arm in the dispatch chain at the top of [`decode_bytes_with_limits`]. Two
/// of those, the magic and the memory profile, compiled clean and tested
/// clean when they were missed, which is how one wave of three parallel
/// format lanes managed to drop a different one each (issue #633).
///
/// It is not the only edit adding a format takes. The variant itself, one
/// arm in [`SniffedFormat::next`] and the two lengths on
/// [`SniffedFormat::ALL`] are still hand-written. The difference is that
/// `cargo build` now insists on every one of them, where two of the six used
/// to be silent.
#[derive(Clone, Copy, Debug)]
struct Route {
    /// The signatures [`sniff`] accepts for this container; any one of them
    /// matching is a match. More than one wherever a format has more than
    /// one container form: both `.v` byte orders, both TIFF byte orders,
    /// both GIF versions, and JPEG XL's bare codestream beside its ISOBMFF
    /// box.
    magics: &'static [Magic],
    /// Which decoder gets the bytes, and over what.
    decoder: Decoder,
}

/// A container libviprs identifies from the leading magic bytes of a file
/// or buffer.
///
/// Only containers this build can actually reach a decoder for are listed;
/// an unrecognised one is `None` from [`sniff`] and falls through to
/// `image`'s own content guess.
///
/// Declaration order is the order [`sniff`] tries the signatures in, which
/// is how libvips orders its loaders' `is_a` calls in
/// `vips_foreign_find_load` (`foreign.c`). No two signatures in the table
/// overlap today, so the order is not load-bearing; it is pinned anyway by
/// `every_container_is_reachable_from_its_own_magic`, which reports the
/// wrong variant the moment one row starts shadowing another.
///
/// Growing this list takes four edits, and `cargo build` insists on all
/// four: the variant here, its row in [`Self::route`] saying what the
/// container is, one arm of bookkeeping in [`Self::next`], and the two
/// lengths on [`Self::ALL`]. Everything else about the container is read off
/// the row. The count is not the point, the enforcement is: none of the four
/// can be missed quietly, where the magic and the memory profile used to be
/// (issue #633).
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
    /// JPEG XL, in either of its two containers: the bare codestream,
    /// `FF 0A`, or the ISOBMFF signature box.
    Jxl,
    /// Radiance HDR, the first line `#?RADIANCE`.
    Radiance,
    /// FITS, the first card's `SIMPLE  =` keyword and fixed-format marker.
    Fits,
    /// OpenEXR, `76 2F 31 01`.
    OpenExr,
    /// NIfTI, in either version and either byte order: a `sizeof_hdr` of
    /// 348 or 540 at offset 0, plus the version's own magic, at 344 for
    /// NIfTI-1 and at 4 for NIfTI-2.
    Nifti,
}

impl SniffedFormat {
    /// The variant after `self` in declaration order, or `None` at the end
    /// of the enum.
    ///
    /// This exists only to build [`Self::ALL`], and it is written as an
    /// exhaustive `match` on purpose: adding a variant stops the crate
    /// compiling here, which is the link a hand-maintained list of variants
    /// does not have. `Jxl` escaped exactly that way between #628 and #659.
    /// The route-table test asserted "only these read the whole file" over
    /// a list of its own, `Jxl` was missing from that list *and* from the
    /// expected answer, so the arithmetic stayed consistent and two
    /// invariants held for the wrong reason.
    const fn next(self) -> Option<Self> {
        match self {
            Self::Vips => Some(Self::Jpeg),
            Self::Jpeg => Some(Self::Png),
            Self::Png => Some(Self::Tiff),
            Self::Tiff => Some(Self::Gif),
            Self::Gif => Some(Self::WebP),
            Self::WebP => Some(Self::Jxl),
            Self::Jxl => Some(Self::Radiance),
            Self::Radiance => Some(Self::Fits),
            Self::Fits => Some(Self::OpenExr),
            Self::OpenExr => Some(Self::Nifti),
            Self::Nifti => None,
        }
    }

    /// Every variant, in declaration order.
    ///
    /// Walked out of [`Self::next`] rather than written out, so the length
    /// and the contents both come from the enum. A variant added without
    /// growing the length fails this `const` block at compile time, and one
    /// added without touching [`Self::next`] fails that `match` first.
    ///
    /// [`sniff`] walks it, so both of those land on `cargo build` rather
    /// than only on `cargo test`. It used to be test-only, which meant the
    /// library itself compiled happily with a variant nothing could reach.
    pub(crate) const ALL: [Self; 11] = {
        let mut all = [Self::Vips; 11];
        let mut i = 1;
        while i < all.len() {
            all[i] = match all[i - 1].next() {
                Some(format) => format,
                None => panic!("SniffedFormat::ALL is longer than the enum"),
            };
            i += 1;
        }
        assert!(
            all[all.len() - 1].next().is_none(),
            "SniffedFormat::ALL is shorter than the enum"
        );
        all
    };

    /// This container's row in the route table.
    ///
    /// The whole of the routing seam is here, in one exhaustive `match`, so
    /// a variant that reaches a decoder is a variant [`sniff`] can find and
    /// [`decode_bytes_with_limits`] can dispatch, by construction rather
    /// than by three lists agreeing. Adding a container without a row is a
    /// compile error; adding a row with the wrong magic is a test failure.
    /// Neither used to be true (issue #633).
    const fn route(self) -> Route {
        match self {
            // `crate::imageio::decode_vips_bytes` parses the libvips header
            // and the metadata trailer itself and needs the buffer
            // addressable end to end, so `.v` never streams.
            Self::Vips => Route {
                magics: &[
                    Magic::Prefix(&crate::imageio::VIPS_MAGIC_LE),
                    Magic::Prefix(&crate::imageio::VIPS_MAGIC_BE),
                ],
                decoder: Decoder::Native(crate::imageio::decode_vips_bytes),
            },
            // The one facade row that is read whole rather than streamed:
            // the metadata pass rescans the APP1/APP2 segments for EXIF and
            // ICC over the same bytes after the pixel decode.
            Self::Jpeg => Route {
                magics: &[Magic::Prefix(b"\xff\xd8\xff")],
                decoder: Decoder::Buffered(image::ImageFormat::Jpeg),
            },
            Self::Png => Route {
                magics: &[Magic::Prefix(b"\x89PNG\r\n\x1a\n")],
                decoder: Decoder::Streamed(image::ImageFormat::Png),
            },
            Self::Tiff => Route {
                magics: &[Magic::Prefix(b"II*\x00"), Magic::Prefix(b"MM\x00*")],
                decoder: Decoder::Streamed(image::ImageFormat::Tiff),
            },
            // `image`'s GIF route is reachable but not usable for parity:
            // `GifDecoder::color_type()` is hard-coded to `Rgba8`, where
            // `vips gifload` emits three bands unless some frame declares a
            // transparent index, and the facade surfaces none of the fields
            // `gifload` attaches. [`crate::gif`] drives the `gif` crate
            // directly instead (issue #570), and reads the file whole: it
            // has to scan every frame's metadata before it can size the
            // output — the band count depends on whether *any* frame
            // declares transparency — and then rewind to decode frame 0.
            // vips does exactly the same thing and pays exactly the same
            // price, `vips_foreign_load_nsgif_header` opening with
            // `vips_source_map(gif->source, &size)`.
            Self::Gif => Route {
                magics: &[Magic::Prefix(b"GIF87a"), Magic::Prefix(b"GIF89a")],
                decoder: Decoder::Native(crate::gif::decode_gif),
            },
            // The `image` facade's WebP decoder reports neither the frame
            // count nor the XMP chunk, and [`crate::webp`] needs both, so
            // that module drives `image-webp` directly (issue #567). It
            // reads the file whole because it takes the `ICCP`, `EXIF` and
            // `XMP ` chunks out of the RIFF directory as well as the frame,
            // and the frame is rarely the last chunk in the file.
            Self::WebP => Route {
                magics: &[Magic::Split {
                    prefix: b"RIFF",
                    tag_at: 8,
                    tag: b"WEBP",
                }],
                decoder: Decoder::Native(crate::webp::decode_webp),
            },
            // `image` 0.25 has no JPEG XL decoder at all, so this row was
            // never anything but a native one; [`crate::jxl`] drives
            // `jxl-oxide` directly (issue #619). It feeds the decoder in two
            // phases so the declared header geometry can be checked against
            // [`DecodeLimits`] before the frame data is fed in at all, which
            // needs the whole buffer addressable up front.
            //
            // The row stays live without the `jxl` feature, on purpose:
            // `decode_jxl` then reports "this build has no JPEG XL", where
            // falling through to [`reader_for`] would report "these bytes
            // are not an image", which is a different and wrong answer.
            Self::Jxl => Route {
                magics: &[
                    Magic::Prefix(JXL_CODESTREAM_MAGIC),
                    Magic::Prefix(JXL_CONTAINER_MAGIC),
                ],
                decoder: Decoder::Native(crate::jxl::decode_jxl),
            },
            // `image`'s Radiance route is behind its `hdr` feature, which
            // this build deliberately leaves off: the crate decodes RGBE as
            // `mantissa * 2^(e-136)` where vips uses the half-bit-centred
            // `(mantissa + 0.5) * 2^(e-136)`, a 100% error at mantissa 0.
            // [`crate::radiance`] hand-rolls the codec instead, and walks
            // the header lines and the run-length-encoded body over one
            // addressable buffer.
            Self::Radiance => Route {
                magics: &[Magic::Line(crate::radiance::MAGIC)],
                decoder: Decoder::Native(crate::radiance::decode_radiance),
            },
            // `image` has no FITS route at all, and no FITS crate models the
            // vips-side behaviour libviprs needs (the vertical flip, the
            // `fits-N` records, cfitsio's equivalent-type table), so
            // [`crate::fits`] hand-rolls the codec (issue #505). It reads
            // the file whole because it may walk past one or more header
            // units before it finds the one carrying the image, and the
            // sample array is band-planar and stored bottom row first, so
            // the decode reads it in an order no strip reader would.
            //
            // FITS also has no signature to speak of: the standard fixes the
            // primary header's first card as `SIMPLE` with a logical value,
            // so the keyword field and the fixed-format `= ` in columns 9
            // and 10 are the only bytes every file shares. vips does not
            // sniff at all here, it hands the file to `fits_open_diskfile`
            // (`fits.c:526-548`).
            Self::Fits => Route {
                magics: &[Magic::Prefix(crate::fits::MAGIC)],
                decoder: Decoder::Native(crate::fits::decode_fits),
            },
            // `image`'s EXR route is behind its `exr` feature, which is
            // exactly `dep:exr`, so naming the crate directly costs nothing
            // extra (issue #504). The reason to name it is that the facade
            // flattens every file to one of its fixed colour types, where an
            // EXR is an arbitrary set of named channels and [`crate::exr`]
            // needs the names, the per-channel sample types and the data
            // window. It reads the file whole because it parses the header
            // twice, once to price the declared data window against the
            // decode budget and once to decode, and the second pass has to
            // start from the beginning of the same bytes.
            Self::OpenExr => Route {
                magics: &[Magic::Prefix(&crate::exr::MAGIC)],
                decoder: Decoder::Native(crate::exr::decode_exr),
            },
            // `image` has no NIfTI route, and neither has the pinned vips:
            // that build reports `NIfTI load/save with libnifti: false` and
            // registers no `niftiload`, which is why [`crate::nifti`] is
            // measured against `nifti_clib` instead (issue #510). It reads
            // the file whole because the header declares a volume and the
            // voxels start at a `vox_offset` the header names, so the
            // decode seeks inside the same bytes it sniffed.
            //
            // Six signatures, because the format has six spellings of its
            // own front: two versions, two byte orders and, on NIfTI-1, the
            // paired `ni1` form as well as the single-file `n+1`. The
            // paired rows are here on purpose, for the same reason the
            // `Jxl` row stays live without the `jxl` feature: a `.hdr` from
            // a pair reaching `decode_nifti` gets "the voxels are in a
            // sibling .img", where falling through to [`reader_for`] would
            // say "these bytes are not an image", which is a different and
            // wrong answer. The NIfTI-2 pair needs no separate byte-order
            // row because its sentinel and magic are adjacent and the magic
            // is never swapped.
            Self::Nifti => Route {
                magics: &[
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_1_LE,
                        tag_at: crate::nifti::MAGIC_1_AT,
                        tag: crate::nifti::MAGIC_1_SINGLE,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_1_BE,
                        tag_at: crate::nifti::MAGIC_1_AT,
                        tag: crate::nifti::MAGIC_1_SINGLE,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_1_LE,
                        tag_at: crate::nifti::MAGIC_1_AT,
                        tag: crate::nifti::MAGIC_1_PAIR,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_1_BE,
                        tag_at: crate::nifti::MAGIC_1_AT,
                        tag: crate::nifti::MAGIC_1_PAIR,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_2_LE,
                        tag_at: crate::nifti::MAGIC_2_AT,
                        tag: crate::nifti::MAGIC_2_SINGLE,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_2_BE,
                        tag_at: crate::nifti::MAGIC_2_AT,
                        tag: crate::nifti::MAGIC_2_SINGLE,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_2_LE,
                        tag_at: crate::nifti::MAGIC_2_AT,
                        tag: crate::nifti::MAGIC_2_PAIR,
                    },
                    Magic::Split {
                        prefix: crate::nifti::SIZEOF_HDR_2_BE,
                        tag_at: crate::nifti::MAGIC_2_AT,
                        tag: crate::nifti::MAGIC_2_PAIR,
                    },
                ],
                decoder: Decoder::Native(crate::nifti::decode_nifti),
            },
        }
    }

    /// Whether [`decode_file_with_limits`] has to read the whole file into
    /// memory rather than streaming it.
    ///
    /// Read off the row rather than kept as a second list of variants, so
    /// widening the table cannot quietly turn a streaming decode into a
    /// whole-file read, and a container that needs the whole file cannot
    /// quietly be left streaming. The latter is the one that used to be
    /// invisible: it decoded fine through [`decode_bytes_with_limits`] and
    /// failed only from a path (issue #633).
    const fn decodes_from_memory(self) -> bool {
        !matches!(self.route().decoder, Decoder::Streamed(_))
    }

    /// The `image` decoder for this container, or `None` for the ones
    /// libviprs decodes itself.
    ///
    /// The mapping is an identity on purpose: one sniffed container in, one
    /// decoder out, no per-format options. Save-side options live in the
    /// per-format modules ([`crate::webp`], [`crate::gif`], [`crate::jxl`]);
    /// nothing about how a file is decoded should ever need to be configured
    /// here.
    const fn image_format(self) -> Option<image::ImageFormat> {
        match self.route().decoder {
            Decoder::Streamed(format) | Decoder::Buffered(format) => Some(format),
            Decoder::Native(_) => None,
        }
    }
}

/// Identify a container from its leading bytes.
///
/// This is the one detector both decode entry points consult. libvips does
/// the same thing in `vips_foreign_find_load` (`foreign.c`), asking each
/// loader's `is_a` in priority order and never trusting the filename. The
/// order here is [`SniffedFormat::ALL`], the enum's declaration order, and
/// the signatures come off each container's own row in the route table, so
/// a container that can reach a decoder is a container this can find
/// (issue #633).
///
/// `head` may be shorter than [`SNIFF_HEAD_LEN`]; a buffer too short for a
/// given magic simply does not match it. For the containers `image` also
/// knows, the byte patterns are the same ones it keeps in its `MAGIC_BYTES`
/// table (`io/free_functions.rs`), so this sniff and the fallback guess in
/// [`reader_for`] cannot disagree about the same file.
pub(crate) fn sniff(head: &[u8]) -> Option<SniffedFormat> {
    SniffedFormat::ALL.into_iter().find(|format| {
        format
            .route()
            .magics
            .iter()
            .any(|magic| magic.matches(head))
    })
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

/// Read the whole of `path` into memory, refusing a file whose length is
/// past [`DecodeLimits::max_alloc_bytes`].
///
/// This is the crate's one bounded whole-file read. Some decoders genuinely
/// need the bytes addressable end to end rather than streamed (see
/// [`SniffedFormat::decodes_from_memory`], and the TIFF page readers, which
/// patch the multiband photometric tag before the decoder ever sees it), and
/// `std::fs::read` is the wrong way to get them: it sizes the buffer from the
/// file and then grows it infallibly, so every ceiling in [`DecodeLimits`] is
/// consulted after the allocation has already happened, and on a constrained
/// host the failure is an abort rather than a returned error (issue #629).
///
/// The declared length is checked first, so an oversized file costs one
/// `stat` rather than a full read, and the read itself is capped as well so a
/// file that grows between the two cannot slip past. `what` names the buffer
/// in the error, because a caller reading
/// [`SourceError::AllocLimitExceeded`] needs to know whether it was the file
/// or a pixel buffer that blew the budget.
///
/// # Errors
///
/// * [`SourceError::Io`] if the file cannot be opened, stat'd or read.
/// * [`SourceError::AllocLimitExceeded`] if the file is longer than
///   [`DecodeLimits::max_alloc_bytes`], which is the same variant the
///   declared-geometry checks raise, so a caller does not have to tell "too
///   big by header" from "too big by file length".
pub(crate) fn read_file_bounded(
    path: &Path,
    limits: DecodeLimits,
    what: &'static str,
) -> Result<Vec<u8>, SourceError> {
    let file = std::fs::File::open(path)?;
    let declared = file.metadata()?.len();
    limits.check_alloc(what, declared)?;

    let cap = limits.max_alloc_bytes;
    let mut bytes = Vec::with_capacity(usize::try_from(declared).unwrap_or(0));
    let mut reader = std::io::BufReader::new(&file).take(cap.saturating_add(1));
    reader.read_to_end(&mut bytes)?;
    let read = bytes.len() as u64;
    if read > cap {
        return Err(SourceError::AllocLimitExceeded {
            what,
            geometry: None,
            needed_bytes: read,
            max_alloc_bytes: cap,
        });
    }
    Ok(bytes)
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
/// PNG and TIFF stream, and never hold more than the decoder asks for. Every
/// other container libviprs recognises is read into memory whole, through a
/// single bounded read, so [`DecodeLimits::max_alloc_bytes`] bounds the read
/// itself rather than only what the decoder does with the bytes afterwards.
/// That is native `.v`, JPEG, GIF, WebP, JPEG XL, Radiance HDR, FITS,
/// OpenEXR and NIfTI: each one either parses its own container end to end or
/// makes a second pass over the same bytes for metadata.
///
/// A file in a container libviprs does not recognise is streamed and guessed
/// by the `image` facade. The two lists above are checked against the routing
/// table by `every_row_carries_the_decoder_kind_its_container_needs`, so this
/// paragraph cannot drift away from what the code does.
///
/// # Errors
///
/// As [`decode_file`], plus [`SourceError::DimensionLimitExceeded`] when
/// the decoded `width * height` exceeds the supplied budget, and
/// [`SourceError::AllocLimitExceeded`] when a memory-decoded container's
/// file is longer than [`DecodeLimits::max_alloc_bytes`].
pub fn decode_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<Raster, SourceError> {
    // Identify the container from its leading magic, never from the path
    // extension: `decode_bytes_with_limits` has no filename to consult, so
    // any filename-derived answer here is one the two entry points cannot
    // both give (issue #563).
    let mut file = std::fs::File::open(path)?;
    let (head, filled) = read_head(&mut file)?;
    let sniffed = sniff(&head[..filled]);
    let mut raster = if sniffed.is_some_and(SniffedFormat::decodes_from_memory) {
        decode_bytes_with_limits(&read_file_bounded(path, limits, "image file body")?, limits)?
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
    // The containers libviprs decodes itself go straight to their own codec.
    // The arm is read off the route table rather than written out as a chain
    // of `if sniffed == Some(..)` tests, so the edit that declares a
    // container is the edit that dispatches it (issue #633).
    if let Some(Decoder::Native(decode)) = sniffed.map(|format| format.route().decoder) {
        return decode(bytes, limits);
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

    /// Compact one-line rendering of a decode outcome: the raster's shape on
    /// success, the error message on failure. Keeps the assertion message
    /// readable where a `{:?}` of the raster would dump the whole pixel
    /// buffer. Shared by the two tests that compare the file entry point
    /// against the byte entry point.
    fn outcome(result: &Result<Raster, SourceError>) -> String {
        match result {
            Ok(im) => format!("Ok({}x{} {:?})", im.width(), im.height(), im.format()),
            Err(e) => format!("Err({e})"),
        }
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

    /**
     * Tests that the shared decode price's saturation sentinel is refused
     * by every budget including `u64::MAX`, the one a caller sets to mean
     * "no limit" (issue #632).
     * Works by pricing a geometry whose product does not fit a `u64`, so
     * `decode_alloc_bytes` saturates, and offering it to both halves of
     * the budget check under the largest budget that can be expressed at
     * all. Saturating is only a refusal if something refuses the sentinel,
     * and a plain `needed > max` does not.
     * Input: `decode_alloc_bytes(u32::MAX, u32::MAX, u64::MAX, 1)` against
     * `max_alloc_bytes = u64::MAX` -> Output: refused, carrying
     * `needed_bytes = u64::MAX`, while `u64::MAX - 1` at the same budget
     * is accepted.
     */
    #[test]
    fn the_saturated_price_is_refused_even_by_a_u64_max_budget() {
        let no_limit = DecodeLimits::default().with_max_alloc_bytes(u64::MAX);
        let saturated = crate::raster::decode_alloc_bytes(u32::MAX, u32::MAX, u64::MAX, 1);
        assert_eq!(saturated, u64::MAX);

        assert!(no_limit.exceeds_alloc_budget(saturated));
        assert!(matches!(
            no_limit.check_alloc("saturated price", saturated),
            Err(SourceError::AllocLimitExceeded {
                what: "saturated price",
                geometry: None,
                needed_bytes: u64::MAX,
                max_alloc_bytes: u64::MAX,
            })
        ));

        // The arm costs exactly one value, and that value is 16 EiB: one
        // byte below the sentinel is a price like any other and the
        // "no limit" budget still clears it.
        assert!(!no_limit.exceeds_alloc_budget(u64::MAX - 1));
        assert!(
            no_limit
                .check_alloc("one below the sentinel", u64::MAX - 1)
                .is_ok()
        );

        // And the ordinary boundary is unmoved: `needed == budget` is
        // accepted, one byte more is not.
        let tight = DecodeLimits::default().with_max_alloc_bytes(4096);
        assert!(!tight.exceeds_alloc_budget(4096));
        assert!(tight.check_alloc("exactly the budget", 4096).is_ok());
        assert!(tight.exceeds_alloc_budget(4097));
        assert!(tight.check_alloc("one over the budget", 4097).is_err());
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
     * Pins the reachability half of the route table: every container in it
     * can actually be found by `sniff`, from the signatures its own row
     * declares, inside the `SNIFF_HEAD_LEN` bytes a file entry point ever
     * reads. Works by building the shortest head each `Magic` accepts and
     * running that back through `sniff`, for every signature of every
     * variant of `SniffedFormat::ALL`.
     * This is the check the old shape did not have, and it is one of the two
     * silent sites in issue #633: a container whose magic never made it into
     * the sniff chain compiled clean, tested clean and was simply never
     * detected. Three ways of getting a row wrong land here — a row with no
     * signature at all, a signature `sniff` cannot match, and a signature
     * some earlier row shadows, which comes back as the wrong variant rather
     * than as `None`. The probes come off the rows themselves, because a
     * hand-kept table of sample bytes is the thing being retired.
     * Input: every `Magic` of every `SniffedFormat` -> Output: `sniff`
     * returns that variant, from no more than `SNIFF_HEAD_LEN` bytes.
     */
    #[test]
    fn every_container_is_reachable_from_its_own_magic() {
        for format in SniffedFormat::ALL {
            let magics = format.route().magics;
            assert!(
                !magics.is_empty(),
                "{format:?} declares no magic, so nothing can ever sniff it"
            );
            for magic in magics {
                let head = magic.shortest_head();
                assert!(
                    !head.is_empty(),
                    "{format:?} declares {magic:?}, which no buffer can fail to match, \
                     so it would shadow every row declared after it"
                );
                assert!(
                    head.len() <= SNIFF_HEAD_LEN,
                    "{format:?} needs {} bytes to decide {magic:?}, more than the \
                     {SNIFF_HEAD_LEN} a file entry point reads, so it is unreachable from disk",
                    head.len()
                );
                assert_eq!(
                    sniff(&head),
                    Some(format),
                    "{magic:?} does not sniff back to {format:?}"
                );
            }
        }
    }

    /**
     * Pins the memory-profile half of the route table, end to end: for every
     * container in it, the file entry point and the byte entry point give
     * the same answer for the same bytes. The file one branches on
     * `decodes_from_memory` — read the whole file and hand it to
     * `decode_bytes_with_limits`, or stream it past the `image` facade — so
     * a row whose profile is wrong shows up here as two answers for one
     * input. That is the other silent site in issue #633: a container whose
     * decoder parses the container itself but whose row said "stream me"
     * reached its own codec through `decode_bytes` and reached `image`'s
     * "these bytes are not an image" through `decode_file`, and only the
     * second of the two was wrong.
     * The inputs are the magic heads themselves, so every container in the
     * table is covered rather than the five that happen to have an encoder
     * in this crate to build a fixture with. They are all far too short to
     * decode, which is the point: what is being compared is the refusal.
     * Input: the shortest head of every magic of every variant -> Output:
     * `decode_file_with_limits` and `decode_bytes_with_limits` report the
     * same outcome for each.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn both_entry_points_agree_on_every_container_in_the_route_table() {
        let dir = tempfile::tempdir().unwrap();
        let mut disagreements: Vec<String> = Vec::new();
        let mut probes = 0;
        for format in SniffedFormat::ALL {
            for (i, magic) in format.route().magics.iter().enumerate() {
                let head = magic.shortest_head();
                let path = dir.path().join(format!("{format:?}-{i}"));
                std::fs::write(&path, &head).unwrap();
                probes += 1;

                let from_file = decode_file_with_limits(&path, DecodeLimits::default());
                let from_bytes = decode_bytes_with_limits(&head, DecodeLimits::default());
                if outcome(&from_file) != outcome(&from_bytes) {
                    disagreements.push(format!(
                        "{format:?} {magic:?}: decode_file_with_limits {} vs \
                         decode_bytes_with_limits {}",
                        outcome(&from_file),
                        outcome(&from_bytes)
                    ));
                }
            }
        }
        assert!(
            disagreements.is_empty(),
            "the file and byte entry points disagree on {} of {probes} containers:\n  {}",
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
        // NIfTI is the one container whose signature does not fit in a
        // hand-written literal: NIfTI-1 puts its magic at byte 344, so a
        // probe is 348 bytes long. These come out of the oracle capture
        // rather than being spelled out, which also means they are real
        // files rather than my idea of one (issue #510).
        const NIFTI_1_LE: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "dt2_uint8.nii"
        ));
        const NIFTI_1_BE: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "endian_nifti1_int16_be.nii"
        ));
        const NIFTI_2_LE: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "ver_n2_single.nii"
        ));
        const NIFTI_2_BE: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "endian_nifti2_int16_be.nii"
        ));
        const NIFTI_1_PAIR: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "pair_n1.hdr"
        ));
        const NIFTI_2_PAIR: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "pair_n2.hdr"
        ));
        // A 348-byte header with the right sentinel and an all-zero magic.
        // The reference reads it as the Analyze 7.5 dialect and decides the
        // container from the filename, which is not something a content
        // sniff can do, so libviprs does not claim it.
        const NIFTI_ANALYZE_DIALECT: &[u8] = include_bytes!(concat!(
            "../oracle-captures/foreign-nifti/fixtures/",
            "magic_zero_analyze.nii"
        ));
        let cases: [(&str, &[u8], Option<SniffedFormat>); 40] = [
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
            // JPEG XL is the only format here with two unrelated magics.
            // The bare codestream is two bytes, which is as short as any
            // signature in the table gets, and the boxed form is the
            // 12-byte ISOBMFF signature box.
            (
                "jxl codestream",
                b"\xff\x0a\x10\x30\x10\x09\x08\x00",
                Some(SniffedFormat::Jxl),
            ),
            (
                "jxl container",
                b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0aftyp",
                Some(SniffedFormat::Jxl),
            ),
            // A JPEG starts `FF D8`, one byte away from the codestream
            // magic, so the near-miss has to stay a JPEG.
            (
                "jpeg is not jxl",
                b"\xff\xd8\xff\xdb",
                Some(SniffedFormat::Jpeg),
            ),
            // The container magic decided from 11 bytes would be a guess.
            (
                "jxl container truncated",
                b"\x00\x00\x00\x0cJXL \x0d\x0a\x87",
                None,
            ),
            // An ISOBMFF file that is not JPEG XL: same box length, wrong
            // box type.
            (
                "isobmff but not jxl",
                b"\x00\x00\x00\x0cftypisom\x00\x00\x02\x00",
                None,
            ),
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
            // OpenEXR is a plain four-byte prefix, the same four bytes
            // `vips__openexr_isexr` reads (`openexr2vips.c:105-115`). The
            // two near-misses below share three of them.
            (
                "openexr",
                b"\x76\x2f\x31\x01\x02\x00\x00\x00",
                Some(SniffedFormat::OpenExr),
            ),
            ("openexr wrong version byte", b"\x76\x2f\x31\x02", None),
            ("openexr truncated", b"\x76\x2f\x31", None),
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
            // NIfTI has six spellings of its own front: two versions, two
            // byte orders, and on each version a single-file and a paired
            // magic. All six route to the same container, and the paired
            // ones on purpose: `decode_nifti` then says "the voxels are in
            // a sibling .img" rather than leaving the file to report "these
            // bytes are not an image".
            ("nifti-1 single le", NIFTI_1_LE, Some(SniffedFormat::Nifti)),
            ("nifti-1 single be", NIFTI_1_BE, Some(SniffedFormat::Nifti)),
            ("nifti-2 single le", NIFTI_2_LE, Some(SniffedFormat::Nifti)),
            ("nifti-2 single be", NIFTI_2_BE, Some(SniffedFormat::Nifti)),
            (
                "nifti-1 pair header",
                NIFTI_1_PAIR,
                Some(SniffedFormat::Nifti),
            ),
            (
                "nifti-2 pair header",
                NIFTI_2_PAIR,
                Some(SniffedFormat::Nifti),
            ),
            // The sentinel alone is not enough: this file has a valid
            // 348-byte sizeof_hdr and no NIfTI magic at all.
            (
                "nifti sentinel without a magic",
                NIFTI_ANALYZE_DIALECT,
                None,
            ),
            // One byte short of the magic, so a 348-byte signature cannot
            // be decided.
            ("nifti-1 truncated to 347", &NIFTI_1_LE[..347], None),
            ("plain text", b"not an image at all", None),
            ("empty", b"", None),
            ("one byte of png", b"\x89", None),
        ];
        for (name, head, expected) in cases {
            assert_eq!(sniff(head), expected, "sniff disagreed on {name}");
        }
        // The case list above is hand-written, which makes it exactly the
        // kind of list issue #633 is about, so it has to prove it names every
        // container rather than only the ones whoever last touched it
        // remembered. The near-misses are what the route table cannot check
        // for itself and are why the list is still worth keeping.
        for format in SniffedFormat::ALL {
            assert!(
                cases
                    .iter()
                    .any(|(_, _, expected)| *expected == Some(format)),
                "{format:?} has no case in the sniff table above"
            );
        }
    }

    /**
     * Tests that adding a container reddens the allocation-refusal tables in
     * `tests/decode_alloc_refusal_shape.rs`, which are hand-written and carry
     * a universal claim ("this is the shape every decoder that prices a frame
     * itself uses") that nothing otherwise ties to the set of containers.
     *
     * `SniffedFormat::ALL` is built by an exhaustive `match`, so a new variant
     * stops the crate compiling *there* rather than here. What this adds is
     * the link to the other file: without it, Batch D can add JP2K, AVIF,
     * UHDR and the rest and those tables silently stay one short each time.
     * Every new decoder either prices its own frame, which needs a
     * `priced_by_libviprs` row, or wraps a crate that refuses internally the
     * way `jxl-oxide` does, which needs an `is_alloc_limit` arm and nothing
     * else will say so.
     * Input: `SniffedFormat::ALL.len()` -> Output: 10, which is what the two
     * tables plus the one documented exclusion account for.
     */
    #[test]
    fn adding_a_container_reddens_the_alloc_refusal_tables() {
        assert_eq!(
            SniffedFormat::ALL.len(),
            11,
            "a container was added or removed. tests/decode_alloc_refusal_shape.rs \
             enumerates every container the decode allocation budget can refuse, in \
             two hand-written tables. Add a row there, or an is_alloc_limit arm if the \
             wrapped crate refuses internally the way jxl-oxide does, then update this \
             count. Today the 11 are: 7 self-priced (gif, radiance, fits, openexr, jxl, \
             webp which joined them in #686, and nifti which joined them in #510), \
             3 refused inside the image crate (jpeg, png, tiff), and .v, which applies \
             no allocation budget at all and is issue #710"
        );
    }

    /**
     * Pins the two contracts the route table cannot check for itself, both
     * of which are judgement calls rather than derivations. First, the
     * mapping into the `image` facade is an identity: one container in, one
     * decoder out, never two containers collapsed onto one decoder. Second,
     * exactly one facade row is read whole rather than streamed, and it is
     * JPEG, because the metadata pass rescans the APP1/APP2 segments over
     * the same bytes after the pixel decode. Marking another facade row
     * `Buffered` would cost every caller the whole file for a second pass
     * that does not exist, and nothing else in the crate would notice.
     * The set of variants comes from `SniffedFormat::ALL`, which is built
     * from an exhaustive `match`, not from a list kept here. A list kept
     * here is how `Jxl` escaped between #628 and #659: it was missing from
     * the list *and* from the expected answer, so the arithmetic stayed
     * consistent and both invariants held over nine variants of ten. The two
     * lists this test used to keep are gone with it — "which containers
     * libviprs decodes itself" and "which read the whole file" are both read
     * off the rows now (issue #633).
     * Input: every `SniffedFormat` variant -> Output: a distinct `image`
     * format for every facade row, and `Buffered` for exactly `Jpeg`.
     */
    #[test]
    fn route_table_is_identity_and_only_jpeg_rereads_the_facade_bytes() {
        let mut mapped: Vec<image::ImageFormat> = SniffedFormat::ALL
            .into_iter()
            .filter_map(SniffedFormat::image_format)
            .collect();
        let facade_rows = mapped.len();
        assert!(
            facade_rows > 0,
            "the image facade decodes something, or reader_for is dead code"
        );
        mapped.sort_by_key(|format| format!("{format:?}"));
        mapped.dedup();
        assert_eq!(
            mapped.len(),
            facade_rows,
            "the route table must be an identity mapping, not many-to-one"
        );

        let buffered: Vec<SniffedFormat> = SniffedFormat::ALL
            .into_iter()
            .filter(|format| matches!(format.route().decoder, Decoder::Buffered(_)))
            .collect();
        assert_eq!(
            buffered,
            vec![SniffedFormat::Jpeg],
            "only JPEG rereads bytes the image facade has already decoded"
        );
    }

    /**
     * Restates, by hand and per variant, which kind of decoder each
     * container's row has to name. Deliberately redundant with the table: the
     * table is where the answer lives, and this is a second opinion about
     * what the answer ought to be, kept for the same reason the near-miss
     * list in `sniff_maps_each_magic_to_one_container` is kept.
     *
     * Collapsing the old sites into one row makes a MISSING row a build
     * error, but it leaves a WRONG row consistent with itself. Changing
     * WebP's row from the libviprs codec to
     * `Decoder::Streamed(ImageFormat::WebP)` bypasses `crate::webp`, drops
     * the ICCP/EXIF/XMP handling issue #567 exists for, and turns a bounded
     * whole-file read into a stream, and every other test in this module
     * still passes. The full suite does catch it, in `webp::tests`, and it
     * catches the same mutation on every other row too, so nothing merges
     * silently. But the red lands three modules from the edit that caused it,
     * and someone changing the table and running `cargo test source::` sees
     * green. This puts the red beside the table.
     *
     * What it does not cover is worth naming, because the discriminant is
     * only half of a row: pointing FITS at `crate::exr::decode_exr` leaves
     * this green, both being `Native`. That one is caught by
     * `fits_reaches_its_codec_from_both_entry_points`, and every native row
     * has an equivalent somewhere that decodes a real file through both entry
     * points. This is the cheap local guard, not the oracle.
     *
     * The inner `match` is exhaustive, so a new variant fails to compile here
     * rather than quietly going unasserted.
     * Input: every `SniffedFormat` variant -> Output: the decoder kind its
     * row carries, the memory profile derived from it, and the streaming list
     * the public docs promise.
     */
    #[test]
    fn every_row_carries_the_decoder_kind_its_container_needs() {
        #[derive(Debug, PartialEq, Eq, Clone, Copy)]
        enum Kind {
            /// A libviprs codec, over the whole file.
            Native,
            /// The `image` facade, over the whole file.
            Buffered,
            /// The `image` facade, streamed.
            Streamed,
        }

        // Written out rather than read off `route()`; see the doc above.
        fn wanted(format: SniffedFormat) -> Kind {
            match format {
                // libviprs parses the `.v` header and metadata trailer itself.
                SniffedFormat::Vips => Kind::Native,
                // The facade decodes it, then the metadata pass rescans the
                // same bytes for the APP1/APP2 segments.
                SniffedFormat::Jpeg => Kind::Buffered,
                SniffedFormat::Png => Kind::Streamed,
                SniffedFormat::Tiff => Kind::Streamed,
                // `crate::gif` drives the `gif` crate directly (issue #570).
                SniffedFormat::Gif => Kind::Native,
                // `crate::webp` drives `image-webp` directly (issue #567).
                SniffedFormat::WebP => Kind::Native,
                // `image` 0.25 has no JPEG XL decoder at all (issue #619).
                SniffedFormat::Jxl => Kind::Native,
                // `image`'s RGBE maths is not vips' (issue #506).
                SniffedFormat::Radiance => Kind::Native,
                // `image` has no FITS route at all (issue #505).
                SniffedFormat::Fits => Kind::Native,
                // The facade flattens the channel set `crate::exr` needs
                // (issue #504).
                SniffedFormat::OpenExr => Kind::Native,
                // Neither `image` nor the pinned vips has a NIfTI route at
                // all (issue #510).
                SniffedFormat::Nifti => Kind::Native,
            }
        }

        for format in SniffedFormat::ALL {
            let carried = match format.route().decoder {
                Decoder::Native(_) => Kind::Native,
                Decoder::Buffered(_) => Kind::Buffered,
                Decoder::Streamed(_) => Kind::Streamed,
            };
            assert_eq!(
                carried,
                wanted(format),
                "{format:?}'s row names a {carried:?} decoder where the container needs a \
                 {:?} one. That is a behaviour change rather than a refactor: it swaps \
                 which codec sees the bytes, and whether the file is read whole",
                wanted(format)
            );
            assert_eq!(
                format.decodes_from_memory(),
                wanted(format) != Kind::Streamed,
                "{format:?}'s memory profile does not follow from its decoder kind"
            );
        }

        // The concrete list `decode_file_with_limits`' public doc gives a
        // caller sizing `max_alloc_bytes`. It is prose, so nothing else would
        // notice it drifting away from the table.
        let streaming: Vec<SniffedFormat> = SniffedFormat::ALL
            .into_iter()
            .filter(|format| !format.decodes_from_memory())
            .collect();
        assert_eq!(
            streaming,
            vec![SniffedFormat::Png, SniffedFormat::Tiff],
            "decode_file_with_limits' doc tells callers PNG and TIFF are the only \
             containers that stream; move the doc and this list together"
        );

        // The same doc paragraph names the other half by hand, and only the
        // streaming half above had a check behind it. The NIfTI row (#510)
        // went in with that sentence left saying eight containers, the
        // suite stayed green, and the prose was wrong for a whole PR.
        let whole_file: Vec<SniffedFormat> = SniffedFormat::ALL
            .into_iter()
            .filter(|format| format.decodes_from_memory())
            .collect();
        assert_eq!(
            whole_file,
            vec![
                SniffedFormat::Vips,
                SniffedFormat::Jpeg,
                SniffedFormat::Gif,
                SniffedFormat::WebP,
                SniffedFormat::Jxl,
                SniffedFormat::Radiance,
                SniffedFormat::Fits,
                SniffedFormat::OpenExr,
                SniffedFormat::Nifti,
            ],
            "decode_file_with_limits' doc names every container it reads whole, in \
             this order; move the doc and this list together"
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

    /// Write `bytes` to `path` and then grow the file to `apparent` bytes.
    ///
    /// The tail is a hole rather than written zeros, so a file that claims
    /// megabytes costs one block on disk. That asymmetry is the whole shape
    /// of issue #629: the file is cheap to make and expensive to serve, and
    /// nothing in the header hints at how big the read will be.
    fn write_sparse(path: &Path, bytes: &[u8], apparent: u64) {
        std::fs::write(path, bytes).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_len(apparent).unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), apparent);
    }

    /**
     * Verifies that `decode_file_with_limits` bounds the whole-file read it
     * does for the containers that decode from memory, so an oversized file
     * costs one `stat` instead of a full read. Before issue #629 the read
     * ran first and every ceiling in `DecodeLimits` was consulted after it
     * had already finished, so a 3 GiB sparse FITS declaring a 4x3 image
     * decoded successfully at 3 GiB resident.
     * Works by writing one real FITS file and one that is byte-identical
     * except for a sparse tail, then decoding both under a ceiling that sits
     * between the two lengths. The small one has to decode, otherwise the
     * bound is refusing on something other than the file size.
     * Input: a 4x3 Gray8 FITS and the same file grown to 4 MiB, both under
     * max_alloc_bytes = 65536 -> Output: the Gray8 raster from the first and
     * `AllocLimitExceeded { what: "image file body" }` from the second.
     */
    #[test]
    fn decode_file_bounds_the_whole_file_read() {
        let raster = Raster::new(4, 3, PixelFormat::Gray8, vec![7u8; 12]).unwrap();
        let file = raster.encode_fits().unwrap();
        let dir = tempfile::tempdir().unwrap();

        let honest = dir.path().join("honest.fits");
        std::fs::write(&honest, &file).unwrap();
        let real_len = std::fs::metadata(&honest).unwrap().len();

        let apparent = 4 * 1024 * 1024;
        let sparse = dir.path().join("sparse.fits");
        write_sparse(&sparse, &file, apparent);

        let limits = DecodeLimits::default().with_max_alloc_bytes(65_536);
        assert!(
            real_len < 65_536,
            "the honest file must sit under the ceiling, got {real_len}"
        );

        let ok = decode_file_with_limits(&honest, limits).unwrap();
        assert_eq!((ok.width(), ok.height()), (4, 3));
        assert_eq!(ok.data(), &[7u8; 12]);

        match decode_file_with_limits(&sparse, limits) {
            Err(SourceError::AllocLimitExceeded {
                what,
                geometry,
                needed_bytes,
                max_alloc_bytes,
            }) => {
                assert_eq!(what, "image file body");
                assert_eq!(
                    geometry, None,
                    "a file length is not a geometry, so the refusal must not invent one"
                );
                assert_eq!(needed_bytes, apparent);
                assert_eq!(max_alloc_bytes, 65_536);
            }
            other => panic!("expected the file body to be refused, got {other:?}"),
        }
    }

    /**
     * Verifies that the file-body ceiling is inclusive, so a file of exactly
     * `max_alloc_bytes` still decodes and one byte less of budget refuses it.
     * A bound that is safe but off by one is a bound that rejects files the
     * caller explicitly paid for, and this epic has found several.
     * Works by measuring the encoded file rather than assuming its length,
     * then running the decode at exactly that budget and at one byte under.
     * The refusal has to name the file body, otherwise it is the pixel
     * buffer's own check firing and the boundary is untested.
     * Input: a 4x3 Gray8 FITS of `n` bytes at max_alloc_bytes = n, then
     * n - 1 -> Output: the Gray8 raster, then
     * `AllocLimitExceeded { needed_bytes: n }`.
     */
    #[test]
    fn the_file_body_ceiling_is_inclusive() {
        let raster = Raster::new(4, 3, PixelFormat::Gray8, vec![7u8; 12]).unwrap();
        let file = raster.encode_fits().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exact.fits");
        std::fs::write(&path, &file).unwrap();
        let n = std::fs::metadata(&path).unwrap().len();

        let exact = DecodeLimits::default().with_max_alloc_bytes(n);
        let decoded = decode_file_with_limits(&path, exact).unwrap();
        assert_eq!(decoded.data(), &[7u8; 12]);

        let one_short = DecodeLimits::default().with_max_alloc_bytes(n - 1);
        assert!(
            matches!(
                decode_file_with_limits(&path, one_short),
                Err(SourceError::AllocLimitExceeded {
                    what: "image file body",
                    geometry: None,
                    needed_bytes,
                    max_alloc_bytes,
                }) if needed_bytes == n && max_alloc_bytes == n - 1
            ),
            "one byte under the file length must refuse the body"
        );
    }

    /**
     * Verifies that the post-read half of the file-body ceiling refuses a
     * source that yields more bytes than its `stat` declared. The
     * stat-first check is the cheap one and it is the one the other two
     * tests pin, but it is only as good as `metadata().len()`, and there
     * are ordinary sources where that number is a lie: a FIFO reports 0,
     * a `/proc` file reports 0, and a regular file can grow between the
     * stat and the read. Without the post-read check the `take(cap + 1)`
     * still bounds the memory, so nothing aborts, but the caller gets a
     * silently truncated buffer handed to the decoder as though it were a
     * whole file and the refusal comes back as "not a recognisable image"
     * instead of "over the ceiling". That is a worse failure than the one
     * issue #629 set out to fix.
     * Works by reading a FIFO, which stats as 0 bytes so the declared-length
     * check cannot fire, while a writer thread feeds it four times the
     * ceiling. `needed_bytes` has to be exactly `cap + 1`, which is all the
     * capped read ever sees, so this cannot be confused with the stat-first
     * refusal that `decode_file_bounds_the_whole_file_read` pins at the
     * apparent length.
     * Input: a FIFO fed 16384 bytes at max_alloc_bytes = 4096 -> Output:
     * `AllocLimitExceeded { what: "image file body", needed_bytes: 4097,
     * max_alloc_bytes: 4096 }`.
     */
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn the_file_body_ceiling_refuses_a_source_that_outruns_its_stat() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("body.fifo");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo(1) is POSIX and must be on PATH");
        assert!(made.success(), "mkfifo failed for {}", fifo.display());
        assert_eq!(
            std::fs::metadata(&fifo).unwrap().len(),
            0,
            "a FIFO must stat as empty, otherwise the declared-length check \
             fires and this test is pinning the wrong half"
        );

        let cap = 4096u64;
        let fed = usize::try_from(cap).unwrap() * 4;
        let writer = {
            let fifo = fifo.clone();
            std::thread::spawn(move || {
                use std::io::Write;
                let mut sink = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
                // The reader stops at cap + 1 bytes and closes, so the tail
                // of this write is expected to come back as a broken pipe.
                // That is the refusal working, not a failure.
                let _ = sink.write_all(&vec![0x5Au8; fed]);
            })
        };

        let limits = DecodeLimits::default().with_max_alloc_bytes(cap);
        let refused = read_file_bounded(&fifo, limits, "image file body");
        writer.join().unwrap();

        match refused {
            Err(SourceError::AllocLimitExceeded {
                what,
                geometry,
                needed_bytes,
                max_alloc_bytes,
            }) => {
                assert_eq!(what, "image file body");
                assert_eq!(geometry, None, "a file length is not a geometry");
                assert_eq!(
                    needed_bytes,
                    cap + 1,
                    "the capped read never sees more than one byte past the \
                     ceiling, so anything else means the stat-first check fired"
                );
                assert_eq!(max_alloc_bytes, cap);
            }
            other => panic!(
                "a source that outruns its stat must be refused, not truncated \
                 and handed to the decoder, got {other:?}"
            ),
        }
    }

    /**
     * Verifies that the new ceiling reaches only the containers that decode
     * from memory, and that the streaming decoders still never see the whole
     * file. Those paths already read no more than they need, so bounding
     * them by file length would refuse files that cost nothing to decode.
     * Works by growing a PNG with a sparse tail well past the ceiling. PNG
     * stops at `IEND`, so the tail is never read, and a decode that succeeds
     * under a ceiling far below the apparent length is proof the streaming
     * path is untouched.
     * Input: a 16x16 RGB PNG grown to 4 MiB, max_alloc_bytes = 65536 ->
     * Output: the 16x16 raster, decoded.
     */
    #[test]
    fn the_file_body_ceiling_leaves_the_streaming_path_alone() {
        let png = create_test_png(16, 16);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trailing.png");
        write_sparse(&path, &png, 4 * 1024 * 1024);

        let limits = DecodeLimits::default().with_max_alloc_bytes(65_536);
        let decoded = decode_file_with_limits(&path, limits).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (16, 16));
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
