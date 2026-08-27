use crate::conversion::RasterMeta;
use crate::imageio::MetadataFields;
use crate::pixel::PixelFormat;
use thiserror::Error;

/// Errors that can occur when creating or slicing a [`Raster`].
///
/// These guard against programmer mistakes such as mismatched buffer sizes,
/// zero-dimension images, and out-of-bounds region requests. They are checked
/// at construction or access time so that pixel-processing code can work with
/// trusted, bounds-checked data.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterError {
    #[error(
        "dimensions {width}x{height} with format {format:?} require {expected} bytes, got {actual}"
    )]
    BufferSizeMismatch {
        width: u32,
        height: u32,
        format: PixelFormat,
        expected: usize,
        actual: usize,
    },
    #[error("zero dimension: {width}x{height}")]
    ZeroDimension { width: u32, height: u32 },
    #[error("region ({x},{y})+({w},{h}) out of bounds for {raster_w}x{raster_h}")]
    RegionOutOfBounds {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        raster_w: u32,
        raster_h: u32,
    },
    #[error("size overflow: {width}x{height} with {bpp} bytes per pixel exceeds usize::MAX")]
    SizeOverflow { width: u32, height: u32, bpp: usize },
    #[error(
        "dimensions {width}x{height} with format {format:?} need {bytes} bytes, exceeding the {budget}-byte allocation budget"
    )]
    ByteBudgetExceeded {
        width: u32,
        height: u32,
        format: PixelFormat,
        bytes: u64,
        budget: u64,
    },
    #[error("failed to allocate {bytes} bytes for {width}x{height} raster")]
    AllocationFailed {
        width: u32,
        height: u32,
        bytes: usize,
    },
    #[error(
        "upscale not supported: target {dst_w}x{dst_h} exceeds source {src_w}x{src_h} (downscale-only)"
    )]
    UpscaleNotSupported {
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    },
    #[error("{op} does not support float rasters yet; cast to an unsigned 8/16-bit format first")]
    FloatUnsupported { op: &'static str },
    #[error("from_f32_samples requires a float pixel format (RgbaF32 / FloatF32), got {format:?}")]
    NotFloatFormat { format: PixelFormat },
    #[error("unknown memory format {format:?}; expected \"uchar\", \"ushort\", or \"float\"")]
    UnknownMemoryFormat { format: String },
    #[error("invalid band count {bands} for memory format {format:?}")]
    InvalidMemoryBands { bands: u32, format: String },
}

/// Default ceiling, in bytes, on a single raster buffer allocation sized from
/// untrusted dimensions.
///
/// Dimensions flow unclamped from file headers (`/MediaBox`, TIFF/PNG IHDR)
/// into buffer allocations. A crafted `50000 × 50000 × Rgba16` (~20 GB) is
/// below the `usize`-overflow threshold [`buffer_len`] guards against, yet far
/// above host memory: an infallible `vec![0u8; size]` would call
/// `handle_alloc_error` and abort the process (a remote DoS) with no chance to
/// return a [`Result`]. [`Raster::new`] and [`Raster::zeroed`] reject any size
/// past this budget with [`RasterError::ByteBudgetExceeded`] before allocating.
///
/// The ceiling is a backstop against clearly-adversarial sizes, not a tight
/// bound: legitimate large-format rasters must still succeed. `8 GiB` sits well
/// above the `4 GiB` an RGBA8 render at the [`DEFAULT_MAX_RENDER_PIXELS`]
/// (`2^30` px) pixel ceiling produces, while still rejecting the multi-tens-of-
/// gigabyte requests an attacker uses to force an abort. Callers that need a
/// tighter (or looser) bound use the `*_with_budget` constructors with an
/// explicit `max_bytes`.
///
/// [`DEFAULT_MAX_RENDER_PIXELS`]: crate::pdf::DEFAULT_MAX_RENDER_PIXELS
pub const DEFAULT_MAX_ALLOC_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Allocate a zeroed buffer of `width × height × bpp` bytes using fallible
/// allocation, rejecting an over-budget size before touching the allocator and
/// converting an allocation failure into a typed error rather than aborting.
fn alloc_zeroed_checked(
    width: u32,
    height: u32,
    format: PixelFormat,
    max_bytes: u64,
) -> Result<Vec<u8>, RasterError> {
    let size = buffer_len(width, height, format.bytes_per_pixel())?;
    if size as u64 > max_bytes {
        return Err(RasterError::ByteBudgetExceeded {
            width,
            height,
            format,
            bytes: size as u64,
            budget: max_bytes,
        });
    }
    let mut data: Vec<u8> = Vec::new();
    data.try_reserve_exact(size)
        .map_err(|_| RasterError::AllocationFailed {
            width,
            height,
            bytes: size,
        })?;
    data.resize(size, 0);
    Ok(data)
}

/// Allocate a zeroed output buffer for an operation whose size derives from an
/// already-validated input raster, using fallible allocation.
///
/// Unlike [`alloc_zeroed_checked`], this does **not** re-impose the
/// [`DEFAULT_MAX_ALLOC_BYTES`] budget. That budget is a backstop against
/// *untrusted file-header dimensions* flowing straight into an allocation; an
/// op output, by contrast, is derived from an input raster that was already
/// budget-checked when it was constructed. A depth-promoting op (8-bit to
/// `f32`) grows the byte size, and — note — a *band-expanding* op grows it
/// further still: `recomb` produces one output band per matrix row (up to
/// `u16::MAX`) and `complexform` doubles the band count, so an output is **not**
/// bounded by any small multiple of its input. Re-applying the byte budget
/// here would therefore spuriously reject legal large outputs (and, at the
/// panicking op forms, abort through the `.expect` that used to wrap
/// [`Raster::new`]).
///
/// Abort-safety does not depend on that budget: the allocation is fallible via
/// [`Vec::try_reserve_exact`], so however large a band-expanding op makes the
/// output, a request the host cannot satisfy surfaces as
/// [`RasterError::AllocationFailed`] (or [`RasterError::SizeOverflow`] for a
/// length past `usize`) rather than aborting through `handle_alloc_error`.
pub(crate) fn alloc_op_output(
    width: u32,
    height: u32,
    format: PixelFormat,
) -> Result<Vec<u8>, RasterError> {
    let size = buffer_len(width, height, format.bytes_per_pixel())?;
    let mut data: Vec<u8> = Vec::new();
    data.try_reserve_exact(size)
        .map_err(|_| RasterError::AllocationFailed {
            width,
            height,
            bytes: size,
        })?;
    data.resize(size, 0);
    Ok(data)
}

/// Compute `width * height * bpp` as a `usize`, checking for overflow.
///
/// The multiplication is performed in `u64` so it never wraps for `u32`
/// dimensions, then narrowed to `usize`. On 32-bit targets a product that
/// exceeds `usize::MAX` yields [`RasterError::SizeOverflow`] rather than
/// wrapping, so behaviour is identical on 32- and 64-bit targets.
fn buffer_len(width: u32, height: u32, bpp: usize) -> Result<usize, RasterError> {
    let overflow = || RasterError::SizeOverflow { width, height, bpp };
    (width as u64)
        .checked_mul(height as u64)
        .and_then(|wh| wh.checked_mul(bpp as u64))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(overflow)
}

/// Price `width * height * bands * sample_bytes` for a decoder's allocation
/// budget, saturating at `u64::MAX`.
///
/// The same product [`buffer_len`] computes, and deliberately next to it,
/// because the two differ only in what they do when it does not fit.
/// `buffer_len` sizes a buffer that is about to exist, so a product it cannot
/// represent has to be an error; this one is only ever handed to
/// [`DecodeLimits::check_alloc`](crate::source::DecodeLimits::check_alloc) and
/// compared against a ceiling, so saturating to `u64::MAX` *is* the answer,
/// the one value no budget can clear. A wrapping product is the failure to
/// avoid: `2^24 x 2^24 x 2^14` four-byte samples is exactly `2^64`, which
/// wraps to `0`, clears every budget, and then sizes a buffer from a
/// different number.
///
/// Every multiplicand widens to `u64` before it is multiplied, so the price
/// does not depend on the target's pointer width the way a `usize` chain
/// does. That is the same rule `buffer_len` states above, and issue #632 is
/// the five per-format spellings of this product that had each drifted off
/// it in their own direction.
///
/// `bands` and `sample_bytes` are `u64` rather than the narrower types the
/// callers hold, because they are not all the same type: a FITS `NAXIS3` is a
/// `u16`, an OpenEXR channel count is a `usize`, and a TIFF sample depth
/// arrives in bits and is rounded up here by its caller.
pub(crate) fn decode_alloc_bytes(width: u32, height: u32, bands: u64, sample_bytes: u64) -> u64 {
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(bands)
        .saturating_mul(sample_bytes)
}

/// An owned raster image buffer with known dimensions and pixel format.
///
/// `Raster` is the core pixel container in libviprs. It owns a tightly-packed
/// `Vec<u8>` whose length is always exactly `width * height * format.bytes_per_pixel()`.
/// This invariant is enforced at construction time by [`Raster::new`] and
/// [`Raster::zeroed`], so downstream code can index into the buffer without
/// additional bounds arithmetic.
///
/// Use [`Raster::region`] for zero-copy sub-region access or [`Raster::extract`]
/// to copy a sub-rectangle into a new `Raster`.
///
/// # Cloning
///
/// `Raster` derives [`Clone`], but cloning is **not** cheap.
/// Cloning copies the entire owned pixel buffer, which for a full-resolution
/// image can be multiple gigabytes, so a stray `.clone()` duplicates the whole
/// image in memory. Pass `&Raster` and use [`Raster::region`] for zero-copy
/// views wherever possible, and reach for `.clone()` only when you genuinely
/// need a second owned copy.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid)
#[derive(Debug, Clone)]
pub struct Raster {
    width: u32,
    height: u32,
    format: PixelFormat,
    data: Vec<u8>,
    /// Interpretation / resolution / orientation metadata, managed by the
    /// conversion operations (see [`crate::conversion`]). Every constructor
    /// starts from [`RasterMeta::default`]; [`Raster::copy`] and
    /// [`Raster::autorot`] are the mutation surface.
    pub(crate) meta: RasterMeta,
    /// Attached metadata fields (ICC profile, EXIF blob, arbitrary named
    /// values), managed by the IO operations (see [`crate::imageio`]).
    /// Every constructor starts empty; [`Raster::set_field`] and the
    /// decoders are the mutation surface, and the fields travel with
    /// clones and [`Raster::copy`].
    pub(crate) fields: MetadataFields,
}

impl Raster {
    /// Create a new raster from existing pixel data.
    ///
    /// Validates that `data.len()` equals `width * height * format.bytes_per_pixel()`
    /// and that neither dimension is zero. This is the primary constructor used
    /// when pixel data has already been produced by a decoder or renderer.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0, or
    /// [`RasterError::BufferSizeMismatch`] if the buffer length is wrong.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, RasterError> {
        Self::new_with_budget(width, height, format, data, DEFAULT_MAX_ALLOC_BYTES)
    }

    /// Like [`Raster::new`], but rejects a declared size exceeding `max_bytes`
    /// with [`RasterError::ByteBudgetExceeded`].
    ///
    /// The buffer is already allocated by the caller, so this cannot make that
    /// allocation fallible; it enforces the budget on the *declared* dimensions
    /// so an attacker cannot smuggle an over-budget raster in through the
    /// pre-allocated path. The budget is checked before the length comparison
    /// so oversized dimensions are rejected as [`RasterError::ByteBudgetExceeded`]
    /// regardless of the buffer supplied.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0,
    /// [`RasterError::ByteBudgetExceeded`] if the size exceeds `max_bytes`, or
    /// [`RasterError::BufferSizeMismatch`] if the buffer length is wrong.
    pub fn new_with_budget(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        max_bytes: u64,
    ) -> Result<Self, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let expected = buffer_len(width, height, format.bytes_per_pixel())?;
        if expected as u64 > max_bytes {
            return Err(RasterError::ByteBudgetExceeded {
                width,
                height,
                format,
                bytes: expected as u64,
                budget: max_bytes,
            });
        }
        if data.len() != expected {
            return Err(RasterError::BufferSizeMismatch {
                width,
                height,
                format,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            format,
            data,
            meta: RasterMeta::default(),
            fields: MetadataFields::default(),
        })
    }

    /// Construct a raster from an operation's output buffer, validating the
    /// length invariant but **not** the allocation budget.
    ///
    /// The buffer has already been produced (typically via [`alloc_op_output`])
    /// from an input raster that was budget-checked at its own construction, so
    /// re-applying [`DEFAULT_MAX_ALLOC_BYTES`] here — as [`Raster::new`] does —
    /// would reject the legal, larger outputs of depth-promoting operations
    /// (issue #279) and, at the panicking op forms, turn that rejection into a
    /// process-ending `.expect`. This constructor keeps the
    /// [`RasterError::ZeroDimension`] and [`RasterError::BufferSizeMismatch`]
    /// invariants and omits only the budget check; the fallibility that guards
    /// against oversized allocations lives in [`alloc_op_output`].
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0,
    /// [`RasterError::SizeOverflow`] if the dimensions overflow `usize`, or
    /// [`RasterError::BufferSizeMismatch`] if the buffer length is wrong.
    pub(crate) fn from_op_output(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let expected = buffer_len(width, height, format.bytes_per_pixel())?;
        if data.len() != expected {
            return Err(RasterError::BufferSizeMismatch {
                width,
                height,
                format,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            format,
            data,
            meta: RasterMeta::default(),
            fields: MetadataFields::default(),
        })
    }

    /// A fallible [`Clone`], for the operation paths that must not abort.
    ///
    /// `Raster` derives `Clone` and cloning copies the whole pixel buffer,
    /// which on a full-resolution image is the largest single allocation an
    /// operation makes. `Clone::clone` reaches `handle_alloc_error` and
    /// **ends the process** when that allocation fails, so a `try_` operation
    /// that copies its input with `.clone()` is not actually fallible however
    /// its signature reads. This reserves with [`Vec::try_reserve_exact`] and
    /// reports [`RasterError::AllocationFailed`] instead, the same contract
    /// [`alloc_op_output`] and [`Raster::zeroed`] already publish.
    ///
    /// The metadata rides along exactly as `Clone` carries it: interpretation,
    /// resolution, orientation and every attached field. That is the reason a
    /// copy is not spelled as `Raster::new` over a fresh buffer, which would
    /// silently reset all of it.
    ///
    /// No budget is applied. The source raster is already held in memory and
    /// already passed whatever budget built it, so a copy of it is by
    /// definition in budget; the fallibility here is against the allocator,
    /// not against a declared size.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::AllocationFailed`] if the allocator cannot
    /// satisfy a buffer the same size as this one.
    pub(crate) fn try_clone(&self) -> Result<Self, RasterError> {
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(self.data.len())
            .map_err(|_| RasterError::AllocationFailed {
                width: self.width,
                height: self.height,
                bytes: self.data.len(),
            })?;
        data.extend_from_slice(&self.data);
        Ok(Self {
            width: self.width,
            height: self.height,
            format: self.format,
            data,
            meta: self.meta,
            fields: self.fields.clone(),
        })
    }

    /// Create a raster filled with zeros.
    ///
    /// Allocates a buffer of the correct size and fills it with `0u8`. Useful
    /// for creating blank tiles or output buffers that will be written into
    /// later (e.g., compositing or scaling operations).
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0,
    /// [`RasterError::ByteBudgetExceeded`] if the size exceeds
    /// [`DEFAULT_MAX_ALLOC_BYTES`], or [`RasterError::AllocationFailed`] if the
    /// allocator cannot satisfy the (in-budget) request — never an abort.
    pub fn zeroed(width: u32, height: u32, format: PixelFormat) -> Result<Self, RasterError> {
        Self::zeroed_with_budget(width, height, format, DEFAULT_MAX_ALLOC_BYTES)
    }

    /// Like [`Raster::zeroed`], but rejects a size exceeding `max_bytes` with
    /// [`RasterError::ByteBudgetExceeded`] before allocating.
    ///
    /// The allocation itself is fallible ([`Vec::try_reserve_exact`]), so an
    /// in-budget request the host still cannot satisfy surfaces as
    /// [`RasterError::AllocationFailed`] rather than aborting the process.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0,
    /// [`RasterError::ByteBudgetExceeded`] if the size exceeds `max_bytes`, or
    /// [`RasterError::AllocationFailed`] on allocator failure.
    pub fn zeroed_with_budget(
        width: u32,
        height: u32,
        format: PixelFormat,
        max_bytes: u64,
    ) -> Result<Self, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let data = alloc_zeroed_checked(width, height, format, max_bytes)?;
        Ok(Self {
            width,
            height,
            format,
            data,
            meta: RasterMeta::default(),
            fields: MetadataFields::default(),
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The [`PixelFormat`] describing channel count and bit depth.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Immutable reference to the raw pixel data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable reference to the raw pixel data buffer.
    ///
    /// Allows in-place pixel manipulation without re-allocating.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Create a float raster from per-channel `f32` samples.
    ///
    /// `samples` is the flat sample sequence (row-major, channels
    /// interleaved), so it must hold exactly
    /// `width * height * format.channels()` values. Each sample is stored
    /// in native byte order, the same convention [`Raster::f32_samples`]
    /// and the arithmetic on 16-bit samples use.
    ///
    /// ```
    /// # use libviprs::{PixelFormat, Raster};
    /// let fmt = PixelFormat::with_channels(1, 4).unwrap(); // FloatF32(1)
    /// let im = Raster::from_f32_samples(2, 1, fmt, &[0.25, -1.5]).unwrap();
    /// assert_eq!(im.getpoint(0, 0), vec![0.25]);
    /// assert_eq!(im.getpoint(1, 0), vec![-1.5]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::NotFloatFormat`] if `format` is not a float
    /// format, [`RasterError::ZeroDimension`] if width or height is 0,
    /// [`RasterError::BufferSizeMismatch`] if the sample count is wrong
    /// (reported in bytes, matching [`Raster::new`]), or the size and
    /// allocation errors of [`Raster::new`].
    pub fn from_f32_samples(
        width: u32,
        height: u32,
        format: PixelFormat,
        samples: &[f32],
    ) -> Result<Self, RasterError> {
        if !format.is_float() {
            return Err(RasterError::NotFloatFormat { format });
        }
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let expected = buffer_len(width, height, format.bytes_per_pixel())?;
        // A slice of f32 occupies exactly 4 bytes per element, so this
        // multiplication cannot overflow usize.
        let actual = samples.len() * 4;
        if actual != expected {
            return Err(RasterError::BufferSizeMismatch {
                width,
                height,
                format,
                expected,
                actual,
            });
        }
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(expected)
            .map_err(|_| RasterError::AllocationFailed {
                width,
                height,
                bytes: expected,
            })?;
        for s in samples {
            data.extend_from_slice(&s.to_ne_bytes());
        }
        Raster::new(width, height, format, data)
    }

    /// The pixel data as `f32` samples, for float formats.
    ///
    /// Returns the flat sample sequence (row-major, channels interleaved)
    /// decoded from the native-byte-order buffer, or `None` when the
    /// format does not store float samples. The inverse of
    /// [`Raster::from_f32_samples`].
    pub fn f32_samples(&self) -> Option<Vec<f32>> {
        if !self.format.is_float() {
            return None;
        }
        Some(
            self.data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_ne_bytes(c))
                .collect(),
        )
    }

    /// Bytes per row (stride). No padding -- rows are tightly packed.
    ///
    /// Equal to `width * format.bytes_per_pixel()`. Needed when computing
    /// byte offsets into the flat data buffer for a given `(x, y)` position.
    pub fn stride(&self) -> usize {
        self.width as usize * self.format.bytes_per_pixel()
    }

    /// Get an immutable, zero-copy view of a rectangular sub-region.
    ///
    /// The returned [`RegionView`] borrows from this `Raster` and provides
    /// row-by-row or per-pixel access without copying any data. This is the
    /// preferred way to read tile-sized chunks during pyramid generation.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::RegionOutOfBounds`] if the rectangle exceeds the
    /// raster dimensions or has a zero width/height.
    pub fn region(&self, x: u32, y: u32, w: u32, h: u32) -> Result<RegionView<'_>, RasterError> {
        // Widen to u64 so `x + w` / `y + h` cannot overflow u32: an unchecked
        // u32 add panics in debug and wraps in release, which would admit an
        // out-of-bounds rectangle and defeat the RegionOutOfBounds contract.
        if x as u64 + w as u64 > self.width as u64
            || y as u64 + h as u64 > self.height as u64
            || w == 0
            || h == 0
        {
            return Err(RasterError::RegionOutOfBounds {
                x,
                y,
                w,
                h,
                raster_w: self.width,
                raster_h: self.height,
            });
        }
        Ok(RegionView {
            raster: self,
            x,
            y,
            w,
            h,
        })
    }

    /// Extract a sub-region as a new owned `Raster`.
    ///
    /// Copies the pixel data row-by-row into a freshly allocated buffer.
    /// Use this when you need an independent `Raster` (e.g., to encode a tile
    /// to disk) rather than a borrowed view.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::RegionOutOfBounds`] if the rectangle is invalid.
    pub fn extract(&self, x: u32, y: u32, w: u32, h: u32) -> Result<Raster, RasterError> {
        let view = self.region(x, y, w, h)?;
        let bpp = self.format.bytes_per_pixel();
        let size = buffer_len(w, h, bpp)?;
        // The sub-region is bounded by `self`, so it is already within budget;
        // use fallible reservation so an allocator failure surfaces as a typed
        // error instead of aborting.
        let mut out: Vec<u8> = Vec::new();
        out.try_reserve_exact(size)
            .map_err(|_| RasterError::AllocationFailed {
                width: w,
                height: h,
                bytes: size,
            })?;
        for row in view.rows() {
            out.extend_from_slice(row);
        }
        Raster::new(w, h, self.format, out)
    }

    /// Fallible form of [`Raster::new_from_memory`].
    ///
    /// Parses `format` (`"uchar"` / `"ushort"` / `"float"`) into the
    /// per-channel byte depth (1 / 2 / 4), builds the canonical
    /// [`PixelFormat`] for `bands` at that depth, and wraps the raw buffer
    /// with the budget-checked [`Raster::new`].
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::UnknownMemoryFormat`] for a format string
    /// outside the supported set, [`RasterError::InvalidMemoryBands`] when
    /// no format can carry `bands` at that depth (zero or above
    /// `u16::MAX`), or any error from [`Raster::new`] (notably
    /// [`RasterError::BufferSizeMismatch`] when `data.len()` does not equal
    /// `width * height * bands * bytes_per_channel`).
    pub fn try_new_from_memory(
        data: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        format: &str,
    ) -> Result<Raster, RasterError> {
        let bytes_per_channel = match format {
            "uchar" => 1,
            "ushort" => 2,
            "float" => 4,
            other => {
                return Err(RasterError::UnknownMemoryFormat {
                    format: other.to_string(),
                });
            }
        };
        let pixel_format = usize::try_from(bands)
            .ok()
            .and_then(|b| PixelFormat::with_channels(b, bytes_per_channel))
            .ok_or_else(|| RasterError::InvalidMemoryBands {
                bands,
                format: format.to_string(),
            })?;
        Self::new(width, height, pixel_format, data.to_vec())
    }

    /// Create a raster from a raw pixel buffer already in memory (libvips
    /// `vips_image_new_from_memory`).
    ///
    /// `format` names the per-channel sample type (`"uchar"`, `"ushort"`,
    /// or `"float"`) and `bands` the channel count; together with `width`
    /// and `height` they must describe a buffer exactly `data.len()` bytes
    /// long. Panicking form of [`Raster::try_new_from_memory`], matching
    /// the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`RasterError`]; see [`Raster::try_new_from_memory`].
    #[track_caller]
    pub fn new_from_memory(
        data: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        format: &str,
    ) -> Raster {
        match Self::try_new_from_memory(data, width, height, bands, format) {
            Ok(raster) => raster,
            Err(e) => panic!("new_from_memory: {e}"),
        }
    }

    /// Copy the raw pixel bytes into a new owned buffer (libvips
    /// `vips_image_write_to_memory`).
    ///
    /// The bytes are the raster's native-endian, tightly-packed pixel
    /// data, so `Raster::new_from_memory(&im.write_to_memory(), w, h,
    /// bands, fmt)` reconstructs an identical image.
    pub fn write_to_memory(&self) -> Vec<u8> {
        self.data().to_vec()
    }
}

/// An immutable, zero-copy view into a rectangular sub-region of a [`Raster`].
///
/// Borrows the parent `Raster` and exposes only the pixels within the
/// specified rectangle. Row iteration via [`RegionView::rows`] and single-pixel
/// access via [`RegionView::pixel`] translate region-local coordinates to
/// absolute buffer offsets automatically.
///
/// Prefer `RegionView` over [`Raster::extract`] when you only need to read
/// pixels without owning them, as it avoids allocation and copying.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug)]
pub struct RegionView<'a> {
    raster: &'a Raster,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl<'a> RegionView<'a> {
    /// Width of the viewed sub-region in pixels.
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Height of the viewed sub-region in pixels.
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Iterate over rows of pixel data in this region.
    ///
    /// Each item is a byte slice of length `width * format.bytes_per_pixel()`,
    /// representing one scanline of the sub-region. Rows are yielded from top
    /// to bottom.
    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> {
        let bpp = self.raster.format.bytes_per_pixel();
        let stride = self.raster.stride();
        let x_offset = self.x as usize * bpp;
        let row_len = self.w as usize * bpp;
        let data = self.raster.data();
        (self.y..self.y + self.h).map(move |row| {
            let start = row as usize * stride + x_offset;
            &data[start..start + row_len]
        })
    }

    /// Get pixel data at `(px, py)` relative to the region origin.
    ///
    /// Returns a byte slice of length `format.bytes_per_pixel()` for the
    /// requested pixel, or `None` if `(px, py)` is outside the region bounds.
    pub fn pixel(&self, px: u32, py: u32) -> Option<&'a [u8]> {
        if px >= self.w || py >= self.h {
            return None;
        }
        let bpp = self.raster.format.bytes_per_pixel();
        let stride = self.raster.stride();
        // Widen to u64 for defense in depth: for a validly bounded region the
        // absolute coordinates cannot overflow u32, but computing them in a
        // wider type keeps the offset math safe even if that invariant is ever
        // weakened.
        let abs_x = self.x as u64 + px as u64;
        let abs_y = self.y as u64 + py as u64;
        let start = abs_y as usize * stride + abs_x as usize * bpp;
        Some(&self.raster.data()[start..start + bpp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageio::MetadataValue;

    fn make_rgb_raster(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        // Fill with a pattern: pixel (x,y) = (x as u8, y as u8, (x+y) as u8)
        for y in 0..h {
            for x in 0..w {
                let offset = (y as usize * w as usize + x as usize) * bpp;
                data[offset] = x as u8;
                data[offset + 1] = y as u8;
                data[offset + 2] = (x + y) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /**
     * Tests that Raster::new rejects buffers that don't match width*height*bpp.
     * Works by providing a too-small buffer (11 bytes for 2x2 Rgb8=12) and
     * verifying Err, then the exact size and verifying Ok.
     * Input: 2x2 Rgb8 with 11 bytes → Err; with 12 bytes → Ok.
     */
    #[test]
    fn new_validates_buffer_size() {
        let result = Raster::new(2, 2, PixelFormat::Rgb8, vec![0u8; 11]);
        assert!(result.is_err());

        let result = Raster::new(2, 2, PixelFormat::Rgb8, vec![0u8; 12]);
        assert!(result.is_ok());
    }

    /**
     * Tests that zero-dimension rasters are rejected by both new() and zeroed().
     * Works by passing width=0 or height=0 and asserting Err is returned.
     * Input: 0x10 Rgb8 → Err; 10x0 Rgb8 → Err; zeroed(0,5) → Err.
     */
    #[test]
    fn zero_dimension_rejected() {
        assert!(Raster::new(0, 10, PixelFormat::Rgb8, vec![]).is_err());
        assert!(Raster::new(10, 0, PixelFormat::Rgb8, vec![]).is_err());
        assert!(Raster::zeroed(0, 5, PixelFormat::Gray8).is_err());
    }

    /**
     * Tests that stride equals width * bytes_per_pixel.
     * Works by creating a 100x50 Rgba8 raster and checking stride == 400.
     * Input: 100x50 Rgba8 → Output: stride() == 400.
     */
    #[test]
    fn stride_is_width_times_bpp() {
        let r = Raster::zeroed(100, 50, PixelFormat::Rgba8).unwrap();
        assert_eq!(r.stride(), 400);
    }

    /**
     * Tests that region() validates bounds against the raster dimensions.
     * Works by requesting valid regions (Ok) and out-of-bounds or zero-width
     * regions (Err) on a 10x10 raster.
     * Input: region(5,5,6,5) on 10x10 → Err (x+w > width).
     */
    #[test]
    fn region_bounds_checking() {
        let r = Raster::zeroed(10, 10, PixelFormat::Rgb8).unwrap();
        assert!(r.region(0, 0, 10, 10).is_ok());
        assert!(r.region(5, 5, 5, 5).is_ok());
        assert!(r.region(5, 5, 6, 5).is_err()); // x+w > width
        assert!(r.region(0, 0, 0, 5).is_err()); // zero width
    }

    /**
     * Tests that region() rejects rectangles whose `x + w` or `y + h` would
     * overflow a u32, instead of panicking (debug) or wrapping to a passing
     * guard (release). Regression test for the u32-add bounds bypass.
     * Works by requesting coordinates whose u32 sum wraps below the raster
     * dimensions on a small raster and asserting RegionOutOfBounds is returned
     * without panicking.
     * Input: region(3_000_000_000, 0, 2_000_000_000, 1) on a 10x10 raster →
     * Err(RegionOutOfBounds). `x + w` = 5e9 wraps to ~705M as u32.
     */
    #[test]
    fn region_rejects_coordinate_overflow() {
        let r = Raster::zeroed(10, 10, PixelFormat::Rgb8).unwrap();
        // x + w overflows u32 (3e9 + 2e9 = 5e9 > u32::MAX).
        assert!(matches!(
            r.region(3_000_000_000, 0, 2_000_000_000, 1),
            Err(RasterError::RegionOutOfBounds { .. })
        ));
        // y + h overflows u32 the same way.
        assert!(matches!(
            r.region(0, 3_000_000_000, 1, 2_000_000_000),
            Err(RasterError::RegionOutOfBounds { .. })
        ));
        // Exact-boundary saturation just past u32::MAX is also rejected.
        assert!(matches!(
            r.region(u32::MAX, 0, 1, 1),
            Err(RasterError::RegionOutOfBounds { .. })
        ));
    }

    /**
     * Tests that RegionView pixels correspond to the correct source raster pixels.
     * Works by creating a raster with position-dependent values (x, y, x+y per pixel)
     * and verifying region pixel (0,0) maps to source pixel (4,3).
     * Input: region(4,3,8,8).pixel(0,0) → [4, 3, 7].
     */
    #[test]
    fn region_pixel_matches_source() {
        let r = make_rgb_raster(16, 16);
        let view = r.region(4, 3, 8, 8).unwrap();

        // pixel (0,0) in region = (4,3) in raster
        let px = view.pixel(0, 0).unwrap();
        assert_eq!(px, &[4, 3, 7]);

        // pixel (7,7) in region = (11,10) in raster
        let px = view.pixel(7, 7).unwrap();
        assert_eq!(px, &[11, 10, 21]);
    }

    /**
     * Tests that accessing a pixel outside the region returns None.
     * Works by creating a 5x5 region and requesting pixel (5,0) and (0,5),
     * both one past the boundary.
     * Input: 5x5 region, pixel(5,0) → None.
     */
    #[test]
    fn region_pixel_out_of_bounds_returns_none() {
        let r = Raster::zeroed(10, 10, PixelFormat::Rgb8).unwrap();
        let view = r.region(0, 0, 5, 5).unwrap();
        assert!(view.pixel(5, 0).is_none());
        assert!(view.pixel(0, 5).is_none());
    }

    /**
     * Tests that extract() copies the correct sub-rectangle into a new Raster.
     * Works by extracting a 4x5 region from a position-encoded 16x16 raster
     * and verifying the first and last pixels match the expected source coords.
     * Input: extract(2,3,4,5) → Output: 4x5 Raster, first pixel=[2,3,5].
     */
    #[test]
    fn extract_produces_correct_sub_image() {
        let r = make_rgb_raster(16, 16);
        let sub = r.extract(2, 3, 4, 5).unwrap();

        assert_eq!(sub.width(), 4);
        assert_eq!(sub.height(), 5);
        assert_eq!(sub.format(), PixelFormat::Rgb8);
        assert_eq!(sub.data().len(), 4 * 5 * 3);

        // First pixel of extracted region should be (2,3) from original
        let bpp = 3;
        assert_eq!(sub.data()[0], 2); // x
        assert_eq!(sub.data()[1], 3); // y
        assert_eq!(sub.data()[2], 5); // x+y
        // Last pixel: (5,7) in original
        let last = (4 * 5 - 1) * bpp;
        assert_eq!(sub.data()[last], 5);
        assert_eq!(sub.data()[last + 1], 7);
        assert_eq!(sub.data()[last + 2], 12);
    }

    /**
     * Tests that RegionView::rows() yields the correct row slices.
     * Works by iterating rows of a 3x2 region starting at (1,1) and
     * verifying row count and pixel values against the source raster.
     * Input: region(1,1,3,2).rows() → 2 rows, each 9 bytes (3px * 3bpp).
     */
    #[test]
    fn region_rows_iteration() {
        let r = make_rgb_raster(8, 8);
        let view = r.region(1, 1, 3, 2).unwrap();

        let rows: Vec<&[u8]> = view.rows().collect();
        assert_eq!(rows.len(), 2);
        // Row 0 of region = row 1 of raster, pixels 1..4
        assert_eq!(rows[0].len(), 9); // 3 pixels * 3 bpp
        assert_eq!(rows[0][0..3], [1, 1, 2]); // pixel (1,1)
        assert_eq!(rows[0][3..6], [2, 1, 3]); // pixel (2,1)
    }

    /**
     * Tests that a 1x1 raster works correctly for all operations.
     * Works by creating a single Gray8 pixel and verifying dimensions, data,
     * region creation, and pixel access all succeed.
     * Input: 1x1 Gray8 [42] → region(0,0,1,1).pixel(0,0) == [42].
     */
    #[test]
    fn single_pixel_raster() {
        let r = Raster::new(1, 1, PixelFormat::Gray8, vec![42]).unwrap();
        assert_eq!(r.width(), 1);
        assert_eq!(r.height(), 1);
        assert_eq!(r.data(), &[42]);

        let view = r.region(0, 0, 1, 1).unwrap();
        assert_eq!(view.pixel(0, 0), Some([42].as_slice()));
    }

    /**
     * Tests that Raster::zeroed produces a buffer filled entirely with zeros.
     * Works by creating a 5x5 Rgba8 zeroed raster and checking every byte.
     * Input: zeroed(5,5,Rgba8) → Output: all 100 bytes == 0.
     */
    #[test]
    fn zeroed_raster_is_all_zeros() {
        let r = Raster::zeroed(5, 5, PixelFormat::Rgba8).unwrap();
        assert!(r.data().iter().all(|&b| b == 0));
    }

    /**
     * Tests that near-u32::MAX dimensions do not overflow width*height*bpp
     * arithmetic. The product w*h*bpp exceeds usize::MAX, so construction must
     * surface a typed error rather than panicking (debug) or wrapping to a
     * small value (release) and letting an inconsistent Raster escape.
     * Input: zeroed/new at u32::MAX x u32::MAX Rgba16 → Output: typed error.
     */
    #[test]
    fn near_u32_max_dimensions_return_error_not_panic() {
        // zeroed must not panic/wrap while computing the buffer size.
        let result = Raster::zeroed(u32::MAX, u32::MAX, PixelFormat::Rgba16);
        assert!(
            matches!(result, Err(RasterError::SizeOverflow { .. })),
            "expected SizeOverflow from zeroed, got {result:?}"
        );

        // new must reject the dimensions before its length check wraps: a short
        // buffer must not be accepted as matching a wrapped `expected`.
        let result = Raster::new(u32::MAX, u32::MAX, PixelFormat::Rgba16, vec![0u8; 8]);
        assert!(
            matches!(result, Err(RasterError::SizeOverflow { .. })),
            "expected SizeOverflow from new, not an Ok with a short buffer, got {result:?}"
        );
    }

    /**
     * Tests that a multi-gigabyte allocation request driven from untrusted
     * dimensions is rejected with a typed error instead of aborting the process.
     * 50000 x 50000 x Rgba16 is ~20 GB — below the usize-overflow threshold
     * (so `buffer_len` succeeds) but far above the allocation budget, so an
     * infallible `vec![0u8; size]` would call `handle_alloc_error` and SIGABRT.
     * Input: zeroed(50000,50000,Rgba16) → Output: Err(ByteBudgetExceeded).
     */
    #[test]
    fn zeroed_multi_gb_request_returns_budget_error_not_abort() {
        let result = Raster::zeroed(50_000, 50_000, PixelFormat::Rgba16);
        assert!(
            matches!(result, Err(RasterError::ByteBudgetExceeded { .. })),
            "expected ByteBudgetExceeded from zeroed, got {result:?}"
        );
    }

    /**
     * Tests that Raster::new also rejects an over-budget declared size before
     * accepting an already-allocated buffer, so an attacker cannot smuggle a
     * multi-GB raster in through the pre-allocated path either. The buffer is
     * intentionally short (the budget check must fire before the size check).
     * Input: new(50000,50000,Rgba16, short buffer) → Output: Err(ByteBudgetExceeded).
     */
    #[test]
    fn new_over_budget_dimensions_rejected() {
        let result = Raster::new(50_000, 50_000, PixelFormat::Rgba16, vec![0u8; 8]);
        assert!(
            matches!(result, Err(RasterError::ByteBudgetExceeded { .. })),
            "expected ByteBudgetExceeded from new, got {result:?}"
        );
    }

    /**
     * Tests that the op-output constructor (issue #279) skips the allocation
     * budget while keeping the length invariant. An operation output whose
     * size derives from an already-validated input is legal even past
     * DEFAULT_MAX_ALLOC_BYTES (a depth promotion grows it up to 4x), so
     * from_op_output must NOT return ByteBudgetExceeded the way Raster::new
     * does. With a deliberately short buffer the budget-free path instead
     * surfaces BufferSizeMismatch, proving the budget check was skipped.
     * Input: from_op_output(50000,50000,Rgba16, short) → Err(BufferSizeMismatch).
     */
    #[test]
    fn from_op_output_skips_budget_but_keeps_length_invariant() {
        let over_budget = Raster::from_op_output(50_000, 50_000, PixelFormat::Rgba16, vec![0u8; 8]);
        assert!(
            matches!(over_budget, Err(RasterError::BufferSizeMismatch { .. })),
            "expected BufferSizeMismatch (budget skipped), got {over_budget:?}"
        );
        // Contrast: Raster::new rejects the identical shape on the budget.
        assert!(matches!(
            Raster::new(50_000, 50_000, PixelFormat::Rgba16, vec![0u8; 8]),
            Err(RasterError::ByteBudgetExceeded { .. })
        ));
        // A well-formed output constructs successfully.
        let ok = Raster::from_op_output(2, 2, PixelFormat::Gray8, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(ok.data(), &[1, 2, 3, 4]);
        assert_eq!((ok.width(), ok.height()), (2, 2));
        // Overflowing dimensions are rejected, not `.expect`-panicked.
        assert!(matches!(
            Raster::from_op_output(u32::MAX, u32::MAX, PixelFormat::Rgba8, Vec::new()),
            Err(RasterError::SizeOverflow { .. })
        ));
    }

    /**
     * Tests that the op-output allocation (issues #279 / #280) is fallible:
     * an oversized request returns a typed error rather than aborting the
     * process through handle_alloc_error, matching the crate's abort-safety
     * design that the op modules previously bypassed with `vec![0u8; n]`.
     * Input: alloc_op_output at overflowing / over-capacity sizes → Err.
     */
    #[test]
    fn alloc_op_output_is_fallible_not_aborting() {
        // Overflows usize entirely (4 bpp) — rejected before touching the allocator.
        assert!(matches!(
            alloc_op_output(u32::MAX, u32::MAX, PixelFormat::Rgba8),
            Err(RasterError::SizeOverflow { .. })
        ));
        // Fits usize (1 bpp) on a 64-bit target but exceeds the Vec capacity
        // ceiling, so try_reserve returns AllocationFailed instead of a SIGABRT.
        // On a 32-bit-usize target the u32::MAX x u32::MAX product overflows
        // usize, so `buffer_len` narrows it to SizeOverflow before the allocator
        // is reached; either typed error satisfies the abort-safety contract.
        assert!(matches!(
            alloc_op_output(u32::MAX, u32::MAX, PixelFormat::Gray8),
            Err(RasterError::AllocationFailed { .. } | RasterError::SizeOverflow { .. })
        ));
        // A normal request yields a zero-filled buffer of the exact size.
        let buf = alloc_op_output(4, 4, PixelFormat::Gray8).unwrap();
        assert_eq!(buf.len(), 16);
        assert!(buf.iter().all(|&b| b == 0));
    }

    /**
     * Tests that `try_clone` is a faithful stand-in for `Clone::clone`
     * (issue #575): the operation paths reach for it precisely because
     * `Clone` reaches handle_alloc_error and ends the process on an
     * image-sized allocation, so it has to carry everything `Clone` does or
     * it is not a substitute. Reconstructing through `Raster::new` over a
     * fresh buffer would compile and would silently drop the interpretation,
     * the resolution and every attached field, which is exactly the failure
     * this pins.
     * Works by giving a raster non-default metadata on both sides (a header
     * field through `copy()`, an attached field through `set_field`), then
     * comparing the fallible copy against the derived one field for field.
     * Input: a 2x2 Rgb8 with xres 42 and a "hello" field → both copies agree
     * on pixels, geometry, format, xres and the attachment.
     */
    #[test]
    fn try_clone_carries_everything_clone_does() {
        let mut im = Raster::new(2, 2, PixelFormat::Rgb8, (0..12).collect()).unwrap();
        im.set_field("hello", MetadataValue::Int(7));
        let im = im.copy().xres(42.0).build();

        let copy = im.try_clone().unwrap();
        assert_eq!(copy.data(), im.data());
        assert_eq!((copy.width(), copy.height()), (im.width(), im.height()));
        assert_eq!(copy.format(), im.format());
        assert!(
            (copy.xres() - 42.0).abs() < 1e-9,
            "xres must survive the copy, got {}",
            copy.xres()
        );
        assert_eq!(copy.interpretation(), im.interpretation());
        assert_eq!(copy.get_fields(), im.get_fields());
        assert_eq!(copy.get_field("hello"), Some(MetadataValue::Int(7)));
    }

    /**
     * Tests that the budget is configurable: a size that exceeds a caller-set
     * budget is rejected, while the same size succeeds under a budget that
     * admits it. Uses `zeroed_with_budget` with a tiny 100-byte budget against
     * a 10x10 Gray8 (100-byte) raster (admitted at 100, rejected at 99).
     * Input: zeroed_with_budget(10,10,Gray8, 99|100) → Output: Err | Ok.
     */
    #[test]
    fn zeroed_with_budget_is_configurable() {
        assert!(matches!(
            Raster::zeroed_with_budget(10, 10, PixelFormat::Gray8, 99),
            Err(RasterError::ByteBudgetExceeded { .. })
        ));
        assert!(Raster::zeroed_with_budget(10, 10, PixelFormat::Gray8, 100).is_ok());
    }

    /**
     * Tests float raster construction and access: from_f32_samples stores
     * native-endian f32s so data length is w*h*channels*4, f32_samples
     * decodes them back exactly (including negatives and fractions), and
     * the byte buffer round-trips through Raster::new unchanged.
     * Input: 2x1 RgbaF32 with 8 samples → f32_samples returns them exactly.
     */
    #[test]
    fn float_raster_construct_and_access() {
        let samples = [0.0f32, 0.5, -1.25, 255.0, 65536.0, 1e-6, -0.0, 3.5];
        let im = Raster::from_f32_samples(2, 1, PixelFormat::RgbaF32, &samples).unwrap();
        assert_eq!(im.width(), 2);
        assert_eq!(im.height(), 1);
        assert_eq!(im.format(), PixelFormat::RgbaF32);
        assert_eq!(im.data().len(), 2 * 16);
        assert_eq!(im.stride(), 32);
        assert_eq!(im.f32_samples().unwrap(), samples.to_vec());

        // The raw byte buffer constructs an identical raster through new().
        let again = Raster::new(2, 1, PixelFormat::RgbaF32, im.data().to_vec()).unwrap();
        assert_eq!(again.f32_samples().unwrap(), samples.to_vec());

        // A zeroed float raster reads as all-0.0 samples.
        let z = Raster::zeroed(3, 2, PixelFormat::with_channels(1, 4).unwrap()).unwrap();
        assert_eq!(z.f32_samples().unwrap(), vec![0.0f32; 6]);
    }

    /**
     * Tests from_f32_samples typed errors: a non-float format is rejected
     * as NotFloatFormat, a wrong sample count as BufferSizeMismatch (in
     * bytes), and zero dimensions as ZeroDimension.
     * Input: Rgb8 → NotFloatFormat; 3 samples for 2x1 FloatF32(1) →
     * BufferSizeMismatch; 0x1 → ZeroDimension.
     */
    #[test]
    fn from_f32_samples_typed_errors() {
        assert!(matches!(
            Raster::from_f32_samples(1, 1, PixelFormat::Rgb8, &[0.0, 0.0, 0.0]),
            Err(RasterError::NotFloatFormat { .. })
        ));
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        assert!(matches!(
            Raster::from_f32_samples(2, 1, f1, &[0.0, 0.0, 0.0]),
            Err(RasterError::BufferSizeMismatch {
                expected: 8,
                actual: 12,
                ..
            })
        ));
        assert!(matches!(
            Raster::from_f32_samples(0, 1, f1, &[]),
            Err(RasterError::ZeroDimension { .. })
        ));
    }

    /**
     * Tests that f32_samples is None for every unsigned format, so callers
     * cannot misread u8/u16 buffers as floats.
     * Input: Gray8 and Rgba16 rasters → f32_samples() == None.
     */
    #[test]
    fn f32_samples_none_for_unsigned() {
        let g = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        assert!(g.f32_samples().is_none());
        let r = Raster::zeroed(2, 2, PixelFormat::Rgba16).unwrap();
        assert!(r.f32_samples().is_none());
    }

    /**
     * Tests the buffer-size invariant for the float formats, mirroring the
     * unsigned buffer_size_invariant proptest at fixed sizes: data length
     * equals w*h*bytes_per_pixel for RgbaF32 and FloatF32(1/3/7).
     * Input: 5x4 rasters → data().len() == 20 * bpp.
     */
    #[test]
    fn float_buffer_size_invariant() {
        for fmt in [
            PixelFormat::RgbaF32,
            PixelFormat::with_channels(1, 4).unwrap(),
            PixelFormat::with_channels(3, 4).unwrap(),
            PixelFormat::with_channels(7, 4).unwrap(),
        ] {
            let r = Raster::zeroed(5, 4, fmt).unwrap();
            assert_eq!(r.data().len(), 20 * fmt.bytes_per_pixel(), "{fmt:?}");
        }
    }

    /**
     * Tests that `bands()` reports the channel count for a known raster of
     * each canonical band count (1/3/4), matching `format().channels()`.
     * Input: Gray8/Rgb8/Rgba8 zeroed rasters -> bands 1/3/4.
     */
    #[test]
    fn bands_reports_channel_count() {
        assert_eq!(Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap().bands(), 1);
        assert_eq!(Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap().bands(), 3);
        assert_eq!(Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap().bands(), 4);
    }

    /**
     * Tests the memory round-trip: a 200-byte zero buffer builds a 20x10
     * 1-band uchar (`Gray8`) raster with the expected geometry and avg 0,
     * and `write_to_memory` returns byte-identical data.
     * Input: vec![0u8; 200] -> new_from_memory(20,10,1,"uchar") ->
     * write_to_memory() == input.
     */
    #[test]
    fn new_from_memory_write_to_memory_round_trips() {
        let data = vec![0u8; 200];
        let im = Raster::new_from_memory(&data, 20, 10, 1, "uchar");
        assert_eq!(im.width(), 20);
        assert_eq!(im.height(), 10);
        assert_eq!(im.bands(), 1);
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert!((im.avg() - 0.0).abs() < 1e-9);
        assert_eq!(im.write_to_memory(), data);
    }

    /**
     * Tests that the format string selects the sample depth and the band
     * count selects the canonical variant: `ushort`+3 -> Rgb16,
     * `float`+4 -> RgbaF32.
     * Input: 36-byte ushort 3x2x3 -> Rgb16; 16-byte float 1x1x4 -> RgbaF32.
     */
    #[test]
    fn new_from_memory_parses_formats_and_bands() {
        let im = Raster::new_from_memory(&[0u8; 36], 3, 2, 3, "ushort");
        assert_eq!(im.format(), PixelFormat::Rgb16);
        assert_eq!(im.bands(), 3);

        let f = Raster::new_from_memory(&[0u8; 16], 1, 1, 4, "float");
        assert_eq!(f.format(), PixelFormat::RgbaF32);
    }

    /**
     * Tests that an unknown format string, a buffer-length mismatch, and a
     * zero band count each surface as a typed error from the fallible form
     * rather than panicking.
     * Input: "uint32" -> UnknownMemoryFormat; 199 bytes for a 200-byte
     * image -> BufferSizeMismatch; 0 bands -> InvalidMemoryBands.
     */
    #[test]
    fn new_from_memory_typed_errors() {
        assert!(matches!(
            Raster::try_new_from_memory(&[0u8; 4], 2, 2, 1, "uint32"),
            Err(RasterError::UnknownMemoryFormat { .. })
        ));
        assert!(matches!(
            Raster::try_new_from_memory(&[0u8; 199], 20, 10, 1, "uchar"),
            Err(RasterError::BufferSizeMismatch { .. })
        ));
        assert!(matches!(
            Raster::try_new_from_memory(&[], 1, 1, 0, "uchar"),
            Err(RasterError::InvalidMemoryBands { bands: 0, .. })
        ));
    }

    /**
     * Tests that the decode allocation price is exact where it fits and
     * saturates where it does not, which is the one boundary every format
     * decoder now shares (issue #632).
     * Works by pricing three geometries whose exact products are known: a
     * small one, the largest product the two axes alone can reach (which is
     * already past `u32::MAX` and so is the case a `usize` chain gets wrong
     * on a 32-bit target), and the one geometry whose product lands exactly
     * on 2^64, where a wrapping multiply gives `0` and clears every budget.
     * Input: 4x3x3x2, `u32::MAX` square, and 2^24 x 2^24 x 2^14 x 4 ->
     * Output: 72, 18446744065119617025, and `u64::MAX`.
     */
    #[test]
    fn the_decode_price_is_exact_where_it_fits_and_saturates_where_it_does_not() {
        assert_eq!(decode_alloc_bytes(4, 3, 3, 2), 72);

        // Past `u32::MAX` on the axes alone, so a product computed in
        // `usize` and narrowed would already be wrong here on a 32-bit
        // target while this one is not.
        assert_eq!(
            decode_alloc_bytes(u32::MAX, u32::MAX, 1, 1),
            18_446_744_065_119_617_025
        );
        assert!(decode_alloc_bytes(u32::MAX, u32::MAX, 1, 1) > u64::from(u32::MAX));

        // Exactly 2^64: `0` if the multiply wraps, `u64::MAX` if it
        // saturates. No budget can be cleared by `u64::MAX`, and every
        // budget is cleared by `0`.
        assert_eq!(decode_alloc_bytes(1 << 24, 1 << 24, 1 << 14, 4), u64::MAX);
        // And well past it, where each of the four multiplicands is at its
        // own ceiling.
        assert_eq!(
            decode_alloc_bytes(u32::MAX, u32::MAX, u64::from(u16::MAX), 4),
            u64::MAX
        );
    }

    /**
     * Tests that the shared price agrees with [`buffer_len`], the crate's
     * other spelling of the same product, everywhere `buffer_len` can
     * answer at all.
     * Works by sweeping band counts and sample sizes through both and
     * comparing, which is what stops the two drifting apart the way the
     * four per-format spellings did before #632.
     * Input: a sweep of geometries and carriers -> Output: the same byte
     * count from both, and `u64::MAX` from the saturating one wherever
     * `buffer_len` refuses.
     */
    #[test]
    fn the_decode_price_agrees_with_buffer_len_wherever_buffer_len_answers() {
        for (w, h, bands, sample_bytes) in [
            (1u32, 1u32, 1u64, 1u64),
            (4, 3, 3, 2),
            (65_535, 65_535, 4, 1),
            (1024, 1024, 4, 4),
        ] {
            let bpp = usize::try_from(bands * sample_bytes).unwrap();
            assert_eq!(
                decode_alloc_bytes(w, h, bands, sample_bytes),
                buffer_len(w, h, bpp).unwrap() as u64,
                "{w}x{h}x{bands} at {sample_bytes} bytes"
            );
        }
        // Where `buffer_len` refuses, the price is the one value no budget
        // can clear rather than an error the budget check has no variant
        // for.
        assert!(buffer_len(u32::MAX, u32::MAX, usize::MAX).is_err());
        assert_eq!(
            decode_alloc_bytes(u32::MAX, u32::MAX, u64::MAX, 1),
            u64::MAX
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            .. ProptestConfig::default()
        })]
        // Tests that buffer size always equals w*h*bpp for all formats and dimensions.
        // Works by generating random dimensions and checking the invariant across
        // all 6 PixelFormat variants.
        // Input: random w,h in 1..256, all formats → Output: data.len() == w*h*bpp.
        #[test]
        fn buffer_size_invariant(w in 1u32..256, h in 1u32..256) {
            for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8,
                        PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16] {
                let r = Raster::zeroed(w, h, fmt).unwrap();
                prop_assert_eq!(
                    r.data().len(),
                    w as usize * h as usize * fmt.bytes_per_pixel()
                );
            }
        }

        // Tests that extract() and region().pixel() return identical data.
        // Works by generating random sub-rectangles and comparing every pixel
        // between the RegionView and the extracted Raster.
        // Input: random region within random raster → Output: all pixels match.
        #[test]
        fn extract_matches_region_pixels(
            w in 4u32..64, h in 4u32..64,
            rx in 0u32..4, ry in 0u32..4,
            rw in 1u32..4, rh in 1u32..4,
        ) {
            prop_assume!(rx + rw <= w && ry + rh <= h);

            let bpp = PixelFormat::Rgb8.bytes_per_pixel();
            let mut data = vec![0u8; w as usize * h as usize * bpp];
            for y in 0..h {
                for x in 0..w {
                    let offset = (y as usize * w as usize + x as usize) * bpp;
                    data[offset] = (x % 256) as u8;
                    data[offset + 1] = (y % 256) as u8;
                    data[offset + 2] = ((x + y) % 256) as u8;
                }
            }
            let raster = Raster::new(w, h, PixelFormat::Rgb8, data).unwrap();
            let view = raster.region(rx, ry, rw, rh).unwrap();
            let extracted = raster.extract(rx, ry, rw, rh).unwrap();

            for py in 0..rh {
                for px in 0..rw {
                    let view_pixel = view.pixel(px, py).unwrap();
                    let ext_offset = (py as usize * rw as usize + px as usize) * bpp;
                    let ext_pixel = &extracted.data()[ext_offset..ext_offset + bpp];
                    prop_assert_eq!(view_pixel, ext_pixel);
                }
            }
        }
    }
}
