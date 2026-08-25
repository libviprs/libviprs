//! Convolution and correlation operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], [`crate::colour`], [`crate::morphology`],
//! [`crate::mosaicing`], and [`crate::create`]): 2D convolution with a
//! mask at integer or float precision, separable convolution, rotating
//! compass convolution, Gaussian blur, unsharp-mask sharpening, and the
//! two template correlations. Operations that can fail on caller input
//! exist in two forms, following the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, ConvolutionError>`
//!   with typed errors for bad kernels and unsupported shapes; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`conv`, `convsep`, `compass`, `gaussblur`, `sharpen`, `spcor`,
//!   `fastcor`) exactly, delegating to the `try_*` form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::conv`] | `vips_conv` | convolved image |
//! | [`Raster::convsep`] | `vips_convsep` | separably convolved image |
//! | [`Raster::compass`] | `vips_compass` | combined rotating convolutions |
//! | [`Raster::gaussblur`] | `vips_gaussblur` | Gaussian-blurred image |
//! | [`Raster::sharpen`] | `vips_sharpen` | unsharp-masked image |
//! | [`Raster::spcor`] | `vips_spcor` | normalised cross-correlation surface |
//! | [`Raster::fastcor`] | `vips_fastcor` | sum-of-squared-differences surface |
//! | [`Kernel::gaussmat`] | `vips_gaussmat` | Gaussian mask |
//! | [`Kernel::logmat`] | `vips_logmat` | Laplacian-of-Gaussian mask |
//!
//! # Semantics shared with libvips
//!
//! * **Masks.** A [`Kernel`] is the double matrix plus its `scale`
//!   divisor, exactly the pair a libvips matrix image carries. Each output
//!   pixel of [`Raster::conv`] is `sum(mask[i] * pixel[i]) / scale`
//!   (`convolution/conv.c`). The scale must be finite and non-zero.
//!   libvips matrix images also carry an `offset` summand, added once per
//!   output sample after the division; the engine honours it and carries
//!   it alongside the scale on its internal mask, but the ported surface
//!   builds `Kernel` as a two-field struct literal, so the offset is `0`
//!   on every public entry point, the value every mask in the ported
//!   suites has.
//! * **Precision.** [`Precision::Float`] is `vips_convf`: coefficients are
//!   baked as `mask / scale`, accumulation is `f64`, and the result is a
//!   32-bit float image regardless of the input depth (libvips promotes
//!   int inputs to `float`; its `double` case has no libviprs depth).
//!   [`Precision::Integer`] is the `vips_convi` C path: the mask is
//!   converted to integers with `rint()` and the scale adjusted to keep
//!   the input/output brightness ratio (`vips__image_intize`), the sum is
//!   accumulated in `i64`, and `(sum + scale / 2) / scale` is written back
//!   in the input format, clipped to its range. A float input under
//!   integer precision keeps the float path of `vips_convi_gen`: `f64`
//!   accumulation with the integer mask and no clipping.
//! * **Edges.** The input is notionally extended by replicating its edge
//!   pixels (`vips_embed` with `VIPS_EXTEND_COPY`, exactly as
//!   `vips_convi_build` / `vips_convf_build` / `vips_correlation_build`
//!   do), so every output has the same dimensions as the input and the
//!   output pixel at `(x, y)` sees the input window whose top-left corner
//!   is `(x - mask_w / 2, y - mask_h / 2)`.
//! * **Separable convolution.** `convsep` requires a `1xN` or `Nx1` mask
//!   and convolves twice, the second pass with the mask rotated 90
//!   degrees, each pass applying the mask scale (`convolution/convsep.c`).
//! * **Compass.** `compass` convolves `times` times, rotating the mask by
//!   `angle` between rounds with the exact [`Raster::rot45`] ring
//!   permutation, then combines the absolute results: [`Combine::Max`] is
//!   the pixelwise maximum (`vips_bandrank`), [`Combine::Sum`] the
//!   pixelwise sum (`vips_sum`). Summed unsigned results are carried one
//!   depth wider (`Gray8 -> Gray16` and so on, saturating at 16 bits:
//!   libviprs has no 32-bit unsigned depth).
//! * **Correlation.** `fastcor` writes the per-band sum of squared
//!   differences between the template and the input window centred on the
//!   output pixel; `spcor` writes the normalised cross-correlation in
//!   `-1..1` with the libvips "constant reference is uncorrelated" rule.
//!   Both accept any mix of unsigned and float inputs with equal band
//!   counts. libvips stores `fastcor` of int images as 32-bit unsigned
//!   sums (wrapping); libviprs has no u32 depth, so both correlations
//!   return 32-bit float images. The wrap-around of the u32 accumulator is
//!   reproduced before the float store.
//! * **Sharpen.** `vips_sharpen` transforms to LabS, blurs the L channel
//!   with a separable integer Gaussian (`sigma`, min_ampl `0.1`), maps the
//!   difference through the m1/m2 lookup curve (x1 `2.0`, brightening cap
//!   `10`, darkening cap `20` L* units), and converts back to the original
//!   interpretation. With `m1 == 0` and `m2 == 0` the curve is identically
//!   zero and the operation is an exact identity for 8-bit sources: the
//!   [`Raster::colourspace`] LabS round trip is byte-exact for every 8-bit
//!   sRGB and mono value (verified exhaustively over all 256^3 sRGB
//!   triples). 16-bit sources round-trip within LabS quantisation, as in
//!   libvips.
//! * **Mask precision defaults.** `gaussmat` and `logmat` default to
//!   integer precision in libvips (`create/gaussmat.c`, `create/logmat.c`
//!   both init `precision = VIPS_PRECISION_INTEGER`); the ported
//!   convolution cell passes the precision explicitly, so
//!   [`Kernel::gaussmat`] takes it as an argument, while
//!   [`Kernel::logmat`] keeps the libvips default (the ported call sites
//!   pass three arguments) and [`Kernel::logmat_with_precision`] exposes
//!   the float form the libvips originals use.

use crate::colour::ColourError;
use crate::conversion::{Angle45, Interpretation};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use thiserror::Error;

/// Don't allow the gaussmat/logmat mask radius to go over this
/// (`MASK_SANITY` in `create/gaussmat.c`).
const MASK_SANITY: u32 = 5000;

/// The fixed sharpen curve parameters libvips defaults to and the ported
/// `sharpen(sigma, m1, m2)` surface does not expose: flat/jaggy threshold
/// `x1`, maximum brightening `y2`, maximum darkening `y3` (L* units).
const SHARPEN_X1: f64 = 2.0;
const SHARPEN_Y2: f64 = 10.0;
const SHARPEN_Y3: f64 = 20.0;

/// Calculation accuracy for the convolution operations (libvips
/// `VipsPrecision`; the `APPROXIMATE` variant is not ported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// Integer arithmetic: the mask is converted to integers with
    /// `rint()` and the result stays in the input format.
    Integer,
    /// Floating-point arithmetic: the result is a 32-bit float image.
    Float,
}

/// How [`Raster::compass`] combines the absolute convolution results
/// (libvips `VipsCombine`; the `MIN` variant is not ported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Combine {
    /// Take the pixelwise maximum of the results.
    Max,
    /// Take the pixelwise sum of the results.
    Sum,
}

/// A convolution mask: the coefficient matrix and its scale divisor
/// (libvips matrix image plus its `scale` metadata).
///
/// The ported tests build this directly as a struct literal
/// (`Kernel { data, scale }`), so the fields are public and the struct is
/// deliberately not `#[non_exhaustive]`. `data` is row-major:
/// `data[row][column]`. libvips masks also carry an `offset` summand; this
/// surface fixes it at `0` (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Mask coefficients, one inner `Vec` per row.
    pub data: Vec<Vec<f64>>,
    /// The divisor applied to each convolution sum.
    pub scale: f64,
}

/// Typed errors for the convolution operations in [`crate::convolution`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConvolutionError {
    /// The kernel has no rows or an empty row.
    #[error("empty kernel")]
    EmptyKernel,
    /// The kernel rows have differing lengths.
    #[error("ragged kernel: row {row} has {got} elements, expected {expected}")]
    RaggedKernel {
        row: usize,
        got: usize,
        expected: usize,
    },
    /// The kernel scale is zero, which would divide every sum by zero.
    #[error("kernel scale must be non-zero")]
    ZeroScale,
    /// A mask scalar is `NaN` or infinite. Neither survives the integer
    /// path: a non-finite scale rounds to an integer scale of `0` and
    /// divides by it, and a non-finite offset saturates to `i64::MAX` and
    /// overflows the summand add.
    #[error("mask {param} must be finite, got {value}")]
    NonFiniteMaskParameter {
        /// Which mask scalar was rejected, `"scale"` or `"offset"`.
        param: &'static str,
        /// The offending value, as supplied.
        value: f64,
    },
    /// `convsep` needs a `1xN` or `Nx1` kernel (libvips
    /// `vips_check_separable`).
    #[error("separable convolution needs a 1xN or Nx1 kernel, got {width}x{height}")]
    NotSeparable { width: u32, height: u32 },
    /// `compass` rotates its kernel with `rot45`, which is defined on
    /// odd-sided square kernels only.
    #[error("compass needs an odd-sided square kernel, got {width}x{height}")]
    NotOddSquareKernel { width: u32, height: u32 },
    /// `compass` must convolve at least once (libvips bounds `times` at
    /// `1..1000`).
    #[error("compass needs times >= 1")]
    ZeroTimes,
    /// The correlation template must have the same band count as the
    /// image.
    #[error("correlation band count mismatch: image has {image} bands, template {template}")]
    BandCountMismatch { image: usize, template: usize },
    /// A mask generator argument is outside the libvips argument range.
    #[error("{op}: {param} must be a finite value in {min}..={max}, got {value}")]
    InvalidMaskParameter {
        op: &'static str,
        param: &'static str,
        min: f64,
        max: f64,
        value: f64,
    },
    /// The generated mask would exceed the libvips `MASK_SANITY` radius.
    #[error("{op}: mask too large")]
    MaskTooLarge { op: &'static str },
    /// A colourspace conversion inside `sharpen` failed (unsupported
    /// source interpretation or too few bands).
    #[error(transparent)]
    Colour(#[from] ColourError),
    /// Constructing a result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_conv<T>(op: &str, r: Result<T, ConvolutionError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// C `rint()` under the default rounding mode: round half to even.
#[inline]
fn rint(v: f64) -> f64 {
    v.round_ties_even()
}

/// A validated dense view of a [`Kernel`]: dimensions, the row-major
/// coefficients, and the two scalars the convolution arithmetic needs.
///
/// libvips keeps `scale` and `offset` on the mask image itself
/// (`vips_image_get_scale` / `vips_image_get_offset`) rather than at the
/// call site, which is what lets `convolution/convsep.c:89-94` express
/// "second pass, same mask, offset zero" by copying the mask and stamping
/// one field. Carrying them here does the same job: a mask and its
/// scalars cannot be separated on the way into [`conv_raster`], and they
/// survive a rotation instead of being rebuilt by hand on the far side.
struct DenseKernel {
    w: usize,
    h: usize,
    coeff: Vec<f64>,
    /// The divisor applied to each convolution sum, from [`Kernel::scale`].
    scale: f64,
    /// The summand applied once per output sample. [`Kernel`] has no
    /// offset field, so everything on the ported surface leaves this at
    /// `0.0`; [`DenseKernel::with_offset`] is how an internal caller
    /// stamps the `vips_image_set_double(mask, "offset", ...)` its C
    /// original does.
    offset: f64,
}

impl DenseKernel {
    fn new(kernel: &Kernel) -> Result<Self, ConvolutionError> {
        if kernel.data.is_empty() || kernel.data[0].is_empty() {
            return Err(ConvolutionError::EmptyKernel);
        }
        let w = kernel.data[0].len();
        let mut coeff = Vec::with_capacity(w * kernel.data.len());
        for (row, r) in kernel.data.iter().enumerate() {
            if r.len() != w {
                return Err(ConvolutionError::RaggedKernel {
                    row,
                    got: r.len(),
                    expected: w,
                });
            }
            coeff.extend_from_slice(r);
        }
        Ok(DenseKernel {
            w,
            h: kernel.data.len(),
            coeff,
            scale: kernel.scale,
            offset: 0.0,
        })
    }

    /// Stamp the mask offset summand, as `convolution/edge.c` and
    /// `convolution/canny.c` do with
    /// `vips_image_set_double(mask, "offset", 128.0)`.
    fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self
    }
}

impl Kernel {
    /// Mask width in coefficients (libvips matrix `Xsize`).
    pub fn width(&self) -> u32 {
        self.data.first().map_or(0, |r| r.len()) as u32
    }

    /// Mask height in coefficients (libvips matrix `Ysize`).
    pub fn height(&self) -> u32 {
        self.data.len() as u32
    }

    /// Largest coefficient (libvips `max` on the mask image). Negative
    /// infinity for an empty kernel.
    pub fn max(&self) -> f64 {
        self.data
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Fallible form of [`Kernel::gaussmat`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::InvalidMaskParameter`] when `sigma` or
    /// `min_ampl` is outside the libvips argument range
    /// `0.000001..=10000`, or [`ConvolutionError::MaskTooLarge`] when the
    /// mask radius would exceed the libvips sanity bound.
    pub fn try_gaussmat(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Result<Kernel, ConvolutionError> {
        check_mask_param("gaussmat", "sigma", sigma)?;
        check_mask_param("gaussmat", "min_ampl", min_ampl)?;

        let sig2 = 2.0 * sigma * sigma;
        // `int max_x = VIPS_CLIP(0, 8 * sigma, MASK_SANITY)`: the C double
        // to int conversion truncates.
        let max_x = (8.0 * sigma).clamp(0.0, MASK_SANITY as f64) as u32;

        // Find the size of the mask: the first x whose amplitude drops
        // below min_ampl ends the loop (gaussmat.c allows x == 0, a 1x1
        // mask).
        let mut x = 0;
        while x < max_x {
            let v = (-((x * x) as f64) / sig2).exp();
            if v < min_ampl {
                break;
            }
            x += 1;
        }
        if x >= MASK_SANITY {
            return Err(ConvolutionError::MaskTooLarge { op: "gaussmat" });
        }
        let width = 2 * x.saturating_sub(1) as usize + 1;
        let height = if separable { 1 } else { width };

        let mut sum = 0.0;
        let mut data = Vec::with_capacity(height);
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let xo = x as f64 - (width / 2) as f64;
                let yo = y as f64 - (height / 2) as f64;
                let distance = xo * xo + yo * yo;
                let mut v = (-distance / sig2).exp();
                if precision != Precision::Float {
                    v = rint(20.0 * v);
                }
                sum += v;
                row.push(v);
            }
            data.push(row);
        }
        // Make sure we can't make sum == 0: it'd certainly cause /0 later
        // (gaussmat.c).
        if sum == 0.0 {
            sum = 1.0;
        }

        Ok(Kernel { data, scale: sum })
    }

    /// Create a Gaussian mask of standard deviation `sigma`, clipped where
    /// the amplitude falls below `min_ampl` (libvips `vips_gaussmat`).
    ///
    /// The mask has odd size. With [`Precision::Float`] the maximum value
    /// is normalised to `1.0`; otherwise each element is `rint(20 * v)`,
    /// the integer mask `vips_conv` uses. With `separable` only the centre
    /// row is generated (`1xN`), for use with [`Raster::convsep`]. `scale`
    /// is the sum of all elements.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_gaussmat`].
    #[track_caller]
    pub fn gaussmat(sigma: f64, min_ampl: f64, separable: bool, precision: Precision) -> Kernel {
        expect_conv(
            "gaussmat",
            Self::try_gaussmat(sigma, min_ampl, separable, precision),
        )
    }

    /// Fallible form of [`Kernel::logmat_with_precision`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::InvalidMaskParameter`] when `sigma` or
    /// `min_ampl` is outside the libvips argument range
    /// `0.000001..=10000`, or [`ConvolutionError::MaskTooLarge`] when the
    /// mask never flattens out within the libvips sanity bound.
    pub fn try_logmat(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Result<Kernel, ConvolutionError> {
        check_mask_param("logmat", "sigma", sigma)?;
        check_mask_param("logmat", "min_ampl", min_ampl)?;

        let sig2 = sigma * sigma;

        // Find the size of the mask: eval out beyond the minimum to where
        // the curve comes back up towards zero, i.e. stop when the change
        // from the previous point is non-negative and the absolute value
        // is below min_ampl (logmat.c).
        let mut last = 0.0;
        let mut x = 0;
        while x < MASK_SANITY {
            let distance = (x * x) as f64;
            let val = 0.5 * (2.0 - distance / sig2) * (-distance / (2.0 * sig2)).exp();
            if val - last >= 0.0 && val.abs() < min_ampl {
                break;
            }
            last = val;
            x += 1;
        }
        if x == MASK_SANITY {
            return Err(ConvolutionError::MaskTooLarge { op: "logmat" });
        }

        let width = 2 * x as usize + 1;
        let height = if separable { 1 } else { width };

        let mut sum = 0.0;
        let mut data = Vec::with_capacity(height);
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let xo = x as f64 - (width / 2) as f64;
                let yo = y as f64 - (height / 2) as f64;
                let distance = xo * xo + yo * yo;
                let mut v = 0.5 * (2.0 - distance / sig2) * (-distance / (2.0 * sig2)).exp();
                if precision == Precision::Integer {
                    v = rint(20.0 * v);
                }
                sum += v;
                row.push(v);
            }
            data.push(row);
        }
        // Unlike gaussmat, logmat.c stores the raw sum with no zero guard:
        // an integer Laplacian-of-Gaussian mask can legitimately sum (and
        // therefore scale) to exactly zero. Convolving with such a mask is
        // a typed ZeroScale error rather than the C path's division by an
        // unadjusted zero.
        Ok(Kernel { data, scale: sum })
    }

    /// Create a Laplacian-of-Gaussian mask of standard deviation `sigma`
    /// at integer precision, the libvips default (libvips `vips_logmat`;
    /// `create/logmat.c` inits `precision = VIPS_PRECISION_INTEGER`). The
    /// ported call sites pass exactly these three arguments; use
    /// [`Kernel::logmat_with_precision`] for the float form.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_logmat`].
    #[track_caller]
    pub fn logmat(sigma: f64, min_ampl: f64, separable: bool) -> Kernel {
        expect_conv(
            "logmat",
            Self::try_logmat(sigma, min_ampl, separable, Precision::Integer),
        )
    }

    /// [`Kernel::logmat`] with an explicit precision, matching the
    /// `precision=` optional argument of `vips_logmat`. With
    /// [`Precision::Float`] the maximum value is normalised to `1.0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_logmat`].
    #[track_caller]
    pub fn logmat_with_precision(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Kernel {
        expect_conv(
            "logmat",
            Self::try_logmat(sigma, min_ampl, separable, precision),
        )
    }
}

/// Enforce the libvips `0.000001..=10000` argument range shared by the
/// gaussmat/logmat sigma and min_ampl parameters.
fn check_mask_param(
    op: &'static str,
    param: &'static str,
    value: f64,
) -> Result<(), ConvolutionError> {
    const MIN: f64 = 0.000001;
    const MAX: f64 = 10000.0;
    if !value.is_finite() || !(MIN..=MAX).contains(&value) {
        return Err(ConvolutionError::InvalidMaskParameter {
            op,
            param,
            min: MIN,
            max: MAX,
            value,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared pixel plumbing
// ---------------------------------------------------------------------------

/// Read every sample of `r` as `f64`, row-major with bands interleaved.
/// Unsigned samples convert exactly; float samples widen exactly.
fn samples_f64(r: &Raster) -> Vec<f64> {
    let fmt = r.format();
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    let data = r.data();
    match fmt.bytes_per_channel() {
        1 => data.iter().map(|&b| b as f64).collect(),
        2 => (0..n)
            .map(|i| u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f64)
            .collect(),
        _ => (0..n)
            .map(|i| {
                f32::from_ne_bytes([
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ]) as f64
            })
            .collect(),
    }
}

/// The 32-bit float format with `channels` bands (the `vips_convf` output
/// depth).
fn float_format(channels: usize) -> PixelFormat {
    PixelFormat::with_channels(channels, 4).expect("validated channel count has a float format")
}

/// Build a float raster from `f64` samples (stored as `f32`, the crate's
/// float depth).
fn raster_from_f64(
    w: u32,
    h: u32,
    channels: usize,
    samples: &[f64],
) -> Result<Raster, RasterError> {
    let f32s: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
    Raster::from_f32_samples(w, h, float_format(channels), &f32s)
}

/// Build an unsigned raster in `fmt` from already-clipped integer samples.
fn raster_from_i64(
    w: u32,
    h: u32,
    fmt: PixelFormat,
    samples: &[i64],
) -> Result<Raster, RasterError> {
    let bpc = fmt.bytes_per_channel();
    let mut data = Vec::with_capacity(samples.len() * bpc);
    if bpc == 1 {
        data.extend(samples.iter().map(|&v| v as u8));
    } else {
        for &v in samples {
            data.extend_from_slice(&(v as u16).to_ne_bytes());
        }
    }
    Raster::new(w, h, fmt, data)
}

/// Clamp a mask-relative coordinate to the image, replicating edge pixels
/// (`VIPS_EXTEND_COPY`).
#[inline]
fn clamp_coord(v: i64, size: u32) -> usize {
    v.clamp(0, size as i64 - 1) as usize
}

// ---------------------------------------------------------------------------
// conv
// ---------------------------------------------------------------------------

/// The integer form of a mask, for the `vips_convi` path: the rounded
/// coefficients, the brightness-corrected scale, and the rounded offset.
///
/// Named fields rather than a tuple because the scale and the offset are
/// both `i64` and mean entirely different things; transposing them in a
/// destructuring pattern would compile and quietly produce wrong pixels.
struct IntKernel {
    coeff: Vec<i64>,
    scale: i64,
    offset: i64,
}

/// Convert a double mask to its integer form (`vips__image_intize`):
/// `rint()` every element, then nudge the rounded scale so an all-ones
/// input keeps the same input/output brightness ratio as the double mask.
///
/// The offset is rounded, clamped into `int` range, and nothing more.
/// `vips__image_intize` does compute an `out_offset` alongside the nudged
/// scale (`convolution/convi.c:898`), but nothing downstream ever reads
/// it: `vips_convi_build` keeps the intized mask in a local only to
/// harvest the coefficients, and `vips_convi_gen` takes both the scale and
/// the offset off the *original* mask (`convi.c:757`). For the offset that
/// makes no difference, because `rint()` is idempotent and the summand is
/// in output units either way. For the scale it does make a difference,
/// and libviprs diverges from libvips there; that is issue #547, and it
/// moves real output bytes, so it is deliberately not fixed here.
///
/// The clamp keeps the rounded offset inside the `int` that
/// `vips_convi_gen` reads it into, which also keeps the `i64` add on the
/// unsigned arm from overflowing on a large finite offset.
/// [`conv_raster`] has already rejected a non-finite one.
fn intize(dense: &DenseKernel) -> IntKernel {
    let scale = dense.scale;
    let sum_double: f64 = dense.coeff.iter().sum();
    let double_result = sum_double / scale;

    let imask: Vec<i64> = dense.coeff.iter().map(|&v| rint(v) as i64).collect();

    let mut out_scale = rint(scale);
    if out_scale == 0.0 {
        out_scale = 1.0;
    }

    // `int_result /= out_scale` in C: double division truncated back into
    // the int.
    let int_sum: i64 = imask.iter().sum();
    let int_result = (int_sum as f64 / out_scale).trunc();

    let mut out_scale = rint(out_scale + (int_result - double_result));
    if out_scale == 0.0 {
        out_scale = 1.0;
    }

    IntKernel {
        coeff: imask,
        scale: out_scale as i64,
        offset: rint(dense.offset).clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i64,
    }
}

/// The clip ceiling for an unsigned format depth.
#[inline]
fn depth_max(fmt: PixelFormat) -> i64 {
    if fmt.bytes_per_channel() == 1 {
        255
    } else {
        65535
    }
}

/// One full 2D convolution pass (both precisions, all formats), the shared
/// engine behind `conv` and `convsep`.
///
/// `dense` carries the mask, its `scale` divisor and its `offset`
/// summand, the same three things a libvips matrix image carries. The
/// public [`Kernel`] has no offset field, so everything on the ported
/// surface convolves with the zero summand.
///
/// Each arm adds the summand exactly where its C counterpart does: before
/// the clip on the integer/unsigned path (`convi.c:710`), after the
/// division and with no clip on the integer/float-input path
/// (`convi.c:733`), and as the starting value of the accumulator at float
/// precision, where the scale is already baked into the coefficients
/// (`convf.c:172`). Both integer arms use the `rint()`-ed offset
/// [`intize`] hands back, matching the `int offset = rint(...)`
/// `vips_convi_gen` reads off the mask; the float arm keeps it as an
/// unrounded `f64`, as `vips_convf_gen` does. It is applied once per
/// output *sample*, not once per tap, so it is one add per sample however
/// large the mask is, and there is nothing here worth specialising away.
///
/// The edge detectors are what need this: `convolution/edge.c` stamps
/// `offset = 128.0, scale = 2.0` on its mask for the uchar path, and
/// `convolution/canny.c` stamps `offset = 128.0` on its gradient mask, so
/// a signed response lands centred in the unsigned output range instead of
/// clipping away at zero. Two libvips rules come with the summand, and a
/// consumer that ignores either gets wrong pixels:
///
/// * `vips_convsep` applies it on the **first pass only**. The second
///   pass runs against a copy of the mask with the offset stamped back to
///   zero (`convolution/convsep.c:89-94`), because the summand is in
///   output units and a two-pass mask would otherwise add it twice.
/// * `vips_compass` takes the absolute value of every rotation before
///   combining them (`convolution/compass.c`), and an offset makes that
///   meaningless: 128 moves the zero point of the response, so `.abs()`
///   folds it about the wrong value. A compass mask carries offset zero.
fn conv_raster(
    src: &Raster,
    dense: &DenseKernel,
    precision: Precision,
) -> Result<Raster, ConvolutionError> {
    let (scale, offset) = (dense.scale, dense.offset);
    if scale == 0.0 {
        return Err(ConvolutionError::ZeroScale);
    }
    // A `NaN` scale slips straight past the zero test and reaches
    // `intize`, where `rint(NaN) as i64` is `0` and the integer arm then
    // divides by it; a non-finite offset saturates `rint(offset) as i64`
    // to `i64::MAX` and overflows the add, which panics in debug and
    // silently wraps to black in release. Both are rejected here, at the
    // one boundary every caller passes through.
    for (param, value) in [("scale", scale), ("offset", offset)] {
        if !value.is_finite() {
            return Err(ConvolutionError::NonFiniteMaskParameter { param, value });
        }
    }
    let (w, h) = (src.width(), src.height());
    let channels = src.format().channels();
    let samples = samples_f64(src);
    let (kw, kh) = (dense.w, dense.h);
    let (ax, ay) = ((kw / 2) as i64, (kh / 2) as i64);
    let row_stride = w as usize * channels;

    match precision {
        Precision::Float => {
            // vips_convf: bake the scale into the coefficients, seed the
            // accumulator with the offset, accumulate in f64, store
            // 32-bit float.
            let coeff: Vec<f64> = dense.coeff.iter().map(|&v| v / scale).collect();
            let mut out = vec![0.0f64; samples.len()];
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    for b in 0..channels {
                        let mut sum = offset;
                        for j in 0..kh as i64 {
                            let sy = clamp_coord(y + j - ay, h);
                            for i in 0..kw as i64 {
                                let sx = clamp_coord(x + i - ax, w);
                                sum += coeff[j as usize * kw + i as usize]
                                    * samples[sy * row_stride + sx * channels + b];
                            }
                        }
                        out[y as usize * row_stride + x as usize * channels + b] = sum;
                    }
                }
            }
            Ok(raster_from_f64(w, h, channels, &out)?)
        }
        Precision::Integer => {
            let IntKernel {
                coeff: imask,
                scale: iscale,
                offset: ioffset,
            } = intize(dense);
            if src.format().is_float() {
                // vips_convi_gen keeps a double path for float inputs: the
                // integer mask, real division, the rounded offset added
                // after it, no rounding of the result and no clip.
                let mut out = vec![0.0f64; samples.len()];
                for y in 0..h as i64 {
                    for x in 0..w as i64 {
                        for b in 0..channels {
                            let mut sum = 0.0;
                            for j in 0..kh as i64 {
                                let sy = clamp_coord(y + j - ay, h);
                                for i in 0..kw as i64 {
                                    let sx = clamp_coord(x + i - ax, w);
                                    sum += imask[j as usize * kw + i as usize] as f64
                                        * samples[sy * row_stride + sx * channels + b];
                                }
                            }
                            out[y as usize * row_stride + x as usize * channels + b] =
                                sum / iscale as f64 + ioffset as f64;
                        }
                    }
                }
                Ok(raster_from_f64(w, h, channels, &out)?)
            } else {
                // CONV_INT: i64 accumulation, (sum + scale/2) / scale with
                // C truncating division, then the offset, then the clip
                // into the input format. The offset lands before the clip,
                // so it recentres the response rather than shifting an
                // already-saturated one.
                let rounding = iscale / 2;
                let max = depth_max(src.format());
                let mut out = vec![0i64; samples.len()];
                for y in 0..h as i64 {
                    for x in 0..w as i64 {
                        for b in 0..channels {
                            let mut sum = 0i64;
                            for j in 0..kh as i64 {
                                let sy = clamp_coord(y + j - ay, h);
                                for i in 0..kw as i64 {
                                    let sx = clamp_coord(x + i - ax, w);
                                    sum += imask[j as usize * kw + i as usize]
                                        * samples[sy * row_stride + sx * channels + b] as i64;
                                }
                            }
                            let v = (sum + rounding) / iscale + ioffset;
                            out[y as usize * row_stride + x as usize * channels + b] =
                                v.clamp(0, max);
                        }
                    }
                }
                Ok(raster_from_i64(w, h, src.format(), &out)?)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel rotation helpers
// ---------------------------------------------------------------------------

/// Rotate a kernel matrix 90 degrees clockwise (`vips_rot` on the mask,
/// as `vips_convsep` does for its second pass).
fn rot90_kernel(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let in_h = data.len();
    let in_w = data[0].len();
    (0..in_w)
        .map(|j| (0..in_h).map(|i| data[in_h - 1 - i][j]).collect())
        .collect()
}

/// Rotate an odd-square kernel matrix by a multiple of 45 degrees using
/// the exact [`Raster::rot45`] ring permutation. The permutation is
/// computed on an index raster so the `f64` coefficients survive without
/// any float quantisation.
fn rot45_kernel(data: &[Vec<f64>], angle: Angle45) -> Vec<Vec<f64>> {
    let n = data.len();
    let indices: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let idx = Raster::from_f32_samples(n as u32, n as u32, float_format(1), &indices)
        .expect("index raster for an odd-square kernel is well-formed");
    let rotated = idx.rot45(angle);
    let perm = rotated
        .f32_samples()
        .expect("rot45 preserves the float format");
    (0..n)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let s = perm[j * n + i] as usize;
                    data[s / n][s % n]
                })
                .collect()
        })
        .collect()
}

impl DenseKernel {
    /// The coefficients back as rows, for the matrix rotation helpers.
    fn rows(&self) -> Vec<Vec<f64>> {
        self.coeff.chunks(self.w).map(<[f64]>::to_vec).collect()
    }

    /// This mask rotated 90 degrees clockwise, keeping its scale and its
    /// offset. `vips_rot` copies the mask metadata across, which is
    /// exactly why `convsep.c:94` has to stamp the offset back to zero by
    /// hand rather than rely on the rotation dropping it.
    fn rot90(&self) -> Self {
        self.respun(rot90_kernel(&self.rows()))
    }

    /// This mask rotated by a multiple of 45 degrees, keeping its scale
    /// and its offset (`vips_compass` rotates its mask with `vips_rot45`).
    fn rot45(&self, angle: Angle45) -> Self {
        self.respun(rot45_kernel(&self.rows(), angle))
    }

    /// Rebuild from rotated rows, carrying the two scalars across. Going
    /// back through a [`Kernel`] literal here would silently drop the
    /// offset, since `Kernel` has nowhere to put it.
    fn respun(&self, data: Vec<Vec<f64>>) -> Self {
        DenseKernel {
            w: data[0].len(),
            h: data.len(),
            coeff: data.into_iter().flatten().collect(),
            scale: self.scale,
            offset: self.offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Fallible form of [`Raster::conv`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::EmptyKernel`] / [`ConvolutionError::RaggedKernel`]
    /// for a malformed mask, [`ConvolutionError::ZeroScale`] for a zero
    /// scale, [`ConvolutionError::NonFiniteMaskParameter`] for a `NaN` or
    /// infinite one, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_conv(
        &self,
        kernel: &Kernel,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        conv_raster(self, &dense, precision)
    }

    /// Convolve the image with `kernel` (libvips `vips_conv`).
    ///
    /// Each output pixel is `sum(mask[i] * pixel[i]) / scale`, evaluated
    /// over the input window centred on the pixel with edges replicated.
    /// See the [module docs](crate::convolution) for the integer/float
    /// precision semantics and output formats.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_conv`].
    #[track_caller]
    pub fn conv(&self, kernel: &Kernel, precision: Precision) -> Raster {
        expect_conv("conv", self.try_conv(kernel, precision))
    }

    /// Fallible form of [`Raster::convsep`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::NotSeparable`] unless the kernel is `1xN` or
    /// `Nx1`, plus the [`Raster::try_conv`] errors.
    pub fn try_convsep(
        &self,
        kernel: &Kernel,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        if dense.w != 1 && dense.h != 1 {
            return Err(ConvolutionError::NotSeparable {
                width: dense.w as u32,
                height: dense.h as u32,
            });
        }
        // vips_convsep: convolve with the mask, then with the mask
        // rotated 90 degrees. The scale divides in both passes, the offset
        // applies to the first only (`convsep.c:89-94`), which is what the
        // explicit zeroing below says. Both ride on the mask, so neither
        // can drift away from the coefficients it belongs to.
        let first = conv_raster(self, &dense, precision)?;
        let second = dense.rot90().with_offset(0.0);
        conv_raster(&first, &second, precision)
    }

    /// Separable convolution with a 1D kernel (libvips `vips_convsep`):
    /// the image is convolved twice, once with the `1xN` (or `Nx1`) mask
    /// and once with the mask rotated 90 degrees. For a separable mask
    /// such as [`Kernel::gaussmat`] with `separable = true` this matches
    /// the full 2D convolution at a fraction of the cost.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_convsep`].
    #[track_caller]
    pub fn convsep(&self, kernel: &Kernel, precision: Precision) -> Raster {
        expect_conv("convsep", self.try_convsep(kernel, precision))
    }

    /// Fallible form of [`Raster::compass`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::NotOddSquareKernel`] unless the kernel is an
    /// odd-sided square, [`ConvolutionError::ZeroTimes`] for `times == 0`,
    /// plus the [`Raster::try_conv`] errors.
    pub fn try_compass(
        &self,
        kernel: &Kernel,
        times: u32,
        angle: Angle45,
        combine: Combine,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        if dense.w != dense.h || dense.w % 2 == 0 {
            return Err(ConvolutionError::NotOddSquareKernel {
                width: dense.w as u32,
                height: dense.h as u32,
            });
        }
        if times == 0 {
            return Err(ConvolutionError::ZeroTimes);
        }

        // vips_compass: convolve, rotate the mask by `angle`, repeat.
        // The rotation carries the scale and the offset, so the mask stays
        // one object all the way round the loop.
        let mut results = Vec::with_capacity(times as usize);
        let mut mask = dense;
        for _ in 0..times {
            results.push(conv_raster(self, &mask, precision)?);
            mask = mask.rot45(angle);
        }

        // Take the absolute value of every result, then combine
        // (vips_abs + vips_bandrank / vips_sum). All results share one
        // format because they come from the same input and precision.
        let (w, h) = (self.width(), self.height());
        let channels = results[0].format().channels();
        let planes: Vec<Vec<f64>> = results.iter().map(samples_f64).collect();
        let n = planes[0].len();
        let mut combined = vec![0.0f64; n];
        for i in 0..n {
            let mut acc: f64 = planes[0][i].abs();
            for plane in &planes[1..] {
                let v = plane[i].abs();
                acc = match combine {
                    Combine::Max => acc.max(v),
                    Combine::Sum => acc + v,
                };
            }
            combined[i] = acc;
        }

        let fmt = results[0].format();
        if fmt.is_float() {
            Ok(raster_from_f64(w, h, channels, &combined)?)
        } else {
            // Unsigned inputs stay unsigned for Max; Sum promotes one
            // depth (vips_sum promotes uchar sums; libviprs tops out at 16
            // bits and saturates, as the arithmetic batch does).
            let out_fmt = match combine {
                Combine::Max => fmt,
                Combine::Sum => PixelFormat::with_channels(channels, 2)
                    .expect("validated channel count has a 16-bit format"),
            };
            let max = depth_max(out_fmt);
            let vals: Vec<i64> = combined.iter().map(|&v| (v as i64).min(max)).collect();
            Ok(raster_from_i64(w, h, out_fmt, &vals)?)
        }
    }

    /// Compass-direction convolution (libvips `vips_compass`): convolve
    /// `times` times, rotating `kernel` by `angle` between rounds, and
    /// combine the absolute results with `combine`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_compass`].
    #[track_caller]
    pub fn compass(
        &self,
        kernel: &Kernel,
        times: u32,
        angle: Angle45,
        combine: Combine,
        precision: Precision,
    ) -> Raster {
        expect_conv(
            "compass",
            self.try_compass(kernel, times, angle, combine, precision),
        )
    }

    /// Fallible form of [`Raster::gaussblur`].
    ///
    /// # Errors
    ///
    /// The [`Kernel::try_gaussmat`] and [`Raster::try_convsep`] errors.
    pub fn try_gaussblur(
        &self,
        sigma: f64,
        min_ampl: f64,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        // vips_gaussblur: gaussmat would make a 1x1 mask for anything
        // smaller than this, so just copy.
        if sigma < 0.2 {
            return Ok(self.clone());
        }
        let mask = Kernel::try_gaussmat(sigma, min_ampl, true, precision)?;
        self.try_convsep(&mask, precision)
    }

    /// Gaussian blur (libvips `vips_gaussblur`): builds the separable
    /// Gaussian mask for `sigma` / `min_ampl` and runs [`Raster::convsep`]
    /// at the given precision. A `sigma` below `0.2` returns a copy, as in
    /// libvips.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_gaussblur`].
    #[track_caller]
    pub fn gaussblur(&self, sigma: f64, min_ampl: f64, precision: Precision) -> Raster {
        expect_conv("gaussblur", self.try_gaussblur(sigma, min_ampl, precision))
    }

    /// Fallible form of [`Raster::sharpen`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::Colour`] when the image has no LabS colourspace
    /// route (for example a 2-band multiband image), plus the
    /// [`Kernel::try_gaussmat`] errors.
    pub fn try_sharpen(&self, sigma: f64, m1: f64, m2: f64) -> Result<Raster, ConvolutionError> {
        // vips_sharpen: remember the interpretation, work in LabS.
        let old_interpretation = self.interpretation();
        let labs = self.try_colourspace(Interpretation::Labs)?;
        let channels = labs.format().channels();
        let (w, h) = (labs.width() as usize, labs.height() as usize);

        // "We always sharpen a short, so there's no point using a float
        // mask": a separable integer Gaussian at 10% amplitude.
        let mask = Kernel::try_gaussmat(sigma, 0.1, true, Precision::Integer)?;
        let mask1d: Vec<i64> = mask.data[0].iter().map(|&v| rint(v) as i64).collect();
        // gaussmat integer masks carry integer sums, so the intize scale
        // adjustment is exact and the divisor is simply the sum.
        let iscale = rint(mask.scale) as i64;

        // vips_cast_short on the L band: the LabS codes from colourspace
        // are already rounded; clamp into the signed 16-bit range.
        let labs_samples = labs.f32_samples().expect("LabS rasters store f32 codes");
        let l: Vec<i32> = (0..w * h)
            .map(|p| (labs_samples[p * channels] as f64).clamp(-32768.0, 32767.0) as i32)
            .collect();

        // Separable integer blur of L with the short clip range, exactly
        // vips_convsep at integer precision on a short image.
        let blur_h = convsep_short_pass(&l, w, h, &mask1d, iscale, true);
        let blurred = convsep_short_pass(&blur_h, w, h, &mask1d, iscale, false);

        // The vips_sharpen LUT, evaluated directly: index i = diff + 32768
        // rescales to +/- 100 as (i - 32767) / 327.67, runs the m1/m2
        // curve capped at y2/-y3, and rounds back to LabS code units.
        let mut out_samples = labs_samples.clone();
        for p in 0..w * h {
            let v1 = l[p];
            let v2 = blurred[p];
            let diff = (v1 & 0x7fff) - (v2 & 0x7fff);
            let v = (diff + 1) as f64 / 327.67;
            let y = if v < -SHARPEN_X1 {
                (v + SHARPEN_X1) * m2 + -SHARPEN_X1 * m1
            } else if v < SHARPEN_X1 {
                v * m1
            } else {
                (v - SHARPEN_X1) * m2 + SHARPEN_X1 * m1
            };
            let y = y.clamp(-SHARPEN_Y3, SHARPEN_Y2);
            let adjusted = (v1 + rint(y * 327.67) as i32).clamp(0, 32767);
            out_samples[p * channels] = adjusted as f32;
        }

        // Reattach a/b (and any extra bands, untouched in out_samples) and
        // convert back to the original interpretation.
        let mut sharpened = labs.clone();
        for (dst, s) in sharpened
            .data_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(out_samples.iter())
        {
            *dst = s.to_ne_bytes();
        }
        Ok(sharpened.try_colourspace(old_interpretation)?)
    }

    /// Unsharp masking for print (libvips `vips_sharpen`): blur the LabS
    /// L channel with a separable integer Gaussian of the given `sigma`,
    /// pass the difference through the two-slope curve (`m1` in flat
    /// areas, `m2` in jaggy areas, thresholds fixed at the libvips
    /// defaults), and add it back. With `m1 == 0` and `m2 == 0` this is an
    /// exact identity for 8-bit sources (see the module docs).
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_sharpen`].
    #[track_caller]
    pub fn sharpen(&self, sigma: f64, m1: f64, m2: f64) -> Raster {
        expect_conv("sharpen", self.try_sharpen(sigma, m1, m2))
    }

    /// Fallible form of [`Raster::spcor`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::BandCountMismatch`] unless the template has the
    /// image's band count, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_spcor(&self, template: &Raster) -> Result<Raster, ConvolutionError> {
        let channels = check_correlation_bands(self, template)?;
        let (w, h) = (self.width(), self.height());
        let (tw, th) = (template.width() as usize, template.height() as usize);
        let n_pels = (tw * th) as f64;
        let input = samples_f64(self);
        let refs = samples_f64(template);
        let row_stride = w as usize * channels;

        // Pre-generate: per-band template mean and
        // sqrt(sum((ref - mean)^2)) (vips_spcor_pre_generate).
        let mut rmean = vec![0.0f64; channels];
        let mut c1 = vec![0.0f64; channels];
        for b in 0..channels {
            let mut sum = 0.0;
            for p in 0..tw * th {
                sum += refs[p * channels + b];
            }
            rmean[b] = sum / n_pels;
            let mut sum2 = 0.0;
            for p in 0..tw * th {
                let d = refs[p * channels + b] - rmean[b];
                sum2 += d * d;
            }
            c1[b] = sum2.sqrt();
        }

        let (ax, ay) = ((tw / 2) as i64, (th / 2) as i64);
        let mut out = vec![0.0f64; input.len()];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for b in 0..channels {
                    // Mean of the input window under the template.
                    let mut sum1 = 0.0;
                    for j in 0..th as i64 {
                        let sy = clamp_coord(y + j - ay, h);
                        for i in 0..tw as i64 {
                            let sx = clamp_coord(x + i - ax, w);
                            sum1 += input[sy * row_stride + sx * channels + b];
                        }
                    }
                    let imean = sum1 / n_pels;

                    // Sum-of-squares of the window and the
                    // product-of-differences against the template.
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;
                    for j in 0..th as i64 {
                        let sy = clamp_coord(y + j - ay, h);
                        for i in 0..tw as i64 {
                            let sx = clamp_coord(x + i - ax, w);
                            let ip = input[sy * row_stride + sx * channels + b];
                            let rp = refs[(j as usize * tw + i as usize) * channels + b];
                            let t = ip - imean;
                            sum2 += t * t;
                            sum3 += (rp - rmean[b]) * t;
                        }
                    }

                    let c2 = c1[b] * sum2.sqrt();
                    // A constant reference (or window) is regarded as
                    // uncorrelated.
                    let cc = if c2 == 0.0 { 0.0 } else { sum3 / c2 };
                    out[y as usize * row_stride + x as usize * channels + b] = cc;
                }
            }
        }
        Ok(raster_from_f64(w, h, channels, &out)?)
    }

    /// Spatial correlation (libvips `vips_spcor`): each output pixel is
    /// the normalised cross-correlation (`-1..1`) between `template` and
    /// the input window centred on it, per band. The maximum marks the
    /// best match; a perfect match scores exactly `1.0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_spcor`].
    #[track_caller]
    pub fn spcor(&self, template: &Raster) -> Raster {
        expect_conv("spcor", self.try_spcor(template))
    }

    /// Fallible form of [`Raster::fastcor`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::BandCountMismatch`] unless the template has the
    /// image's band count, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_fastcor(&self, template: &Raster) -> Result<Raster, ConvolutionError> {
        let channels = check_correlation_bands(self, template)?;
        let (w, h) = (self.width(), self.height());
        let (tw, th) = (template.width() as usize, template.height() as usize);
        let input = samples_f64(self);
        let refs = samples_f64(template);
        let row_stride = w as usize * channels;
        let (ax, ay) = ((tw / 2) as i64, (th / 2) as i64);

        // vips__formatalike: any float input switches both sides to the
        // float path (f32 accumulation, CORR_FLOAT); two unsigned inputs
        // keep the integer path (u32 accumulation with C wrap-around,
        // CORR_INT).
        let float_path = self.format().is_float() || template.format().is_float();

        let mut out = vec![0.0f64; input.len()];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for b in 0..channels {
                    let o = y as usize * row_stride + x as usize * channels + b;
                    if float_path {
                        let mut sum = 0.0f32;
                        for j in 0..th as i64 {
                            let sy = clamp_coord(y + j - ay, h);
                            for i in 0..tw as i64 {
                                let sx = clamp_coord(x + i - ax, w);
                                let dif = refs[(j as usize * tw + i as usize) * channels + b]
                                    as f32
                                    - input[sy * row_stride + sx * channels + b] as f32;
                                sum += dif * dif;
                            }
                        }
                        out[o] = sum as f64;
                    } else {
                        let mut sum = 0u32;
                        for j in 0..th as i64 {
                            let sy = clamp_coord(y + j - ay, h);
                            for i in 0..tw as i64 {
                                let sx = clamp_coord(x + i - ax, w);
                                let t = refs[(j as usize * tw + i as usize) * channels + b] as i64
                                    - input[sy * row_stride + sx * channels + b] as i64;
                                sum = sum.wrapping_add((t * t) as u32);
                            }
                        }
                        out[o] = sum as f64;
                    }
                }
            }
        }
        Ok(raster_from_f64(w, h, channels, &out)?)
    }

    /// Fast correlation (libvips `vips_fastcor`): each output pixel is the
    /// per-band sum of squared differences between `template` and the
    /// input window centred on it. The minimum marks the best match; a
    /// perfect match scores exactly `0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_fastcor`].
    #[track_caller]
    pub fn fastcor(&self, template: &Raster) -> Raster {
        expect_conv("fastcor", self.try_fastcor(template))
    }
}

/// Shared correlation validation: equal band counts.
fn check_correlation_bands(image: &Raster, template: &Raster) -> Result<usize, ConvolutionError> {
    let channels = image.format().channels();
    if channels != template.format().channels() {
        return Err(ConvolutionError::BandCountMismatch {
            image: channels,
            template: template.format().channels(),
        });
    }
    Ok(channels)
}

/// One 1D integer convolution pass over an L-code plane in the signed
/// 16-bit domain, horizontal or vertical: the `CONV_INT` inner loop of
/// `vips_convi_gen` with `CLIP_SHORT`, which is how `vips_sharpen` blurs
/// the L channel (`vips_convsep` at integer precision on a short image).
fn convsep_short_pass(
    src: &[i32],
    w: usize,
    h: usize,
    mask1d: &[i64],
    iscale: i64,
    horizontal: bool,
) -> Vec<i32> {
    let rounding = iscale / 2;
    let half = (mask1d.len() / 2) as i64;
    let mut out = vec![0i32; src.len()];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut sum = 0i64;
            for (k, &m) in mask1d.iter().enumerate() {
                let off = k as i64 - half;
                let (sx, sy) = if horizontal {
                    (x + off, y)
                } else {
                    (x, y + off)
                };
                let sx = clamp_coord(sx, w as u32);
                let sy = clamp_coord(sy, h as u32);
                sum += m * src[sy * w + sx] as i64;
            }
            let v = (sum + rounding) / iscale;
            out[y as usize * w + x as usize] = v.clamp(-32768, 32767) as i32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random byte stream for synthetic images.
    fn lcg(seed: u32) -> impl FnMut() -> u8 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }
    }

    /// A `w x h` Gray8 raster of deterministic noise.
    fn noise_gray(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let data: Vec<u8> = (0..w as usize * h as usize).map(|_| next()).collect();
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /// A `w x h` Rgb8 raster of deterministic noise.
    fn noise_rgb(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let data: Vec<u8> = (0..w as usize * h as usize * 3).map(|_| next()).collect();
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// Scalar reference convolution at `(x, y)` with edge replication,
    /// unclipped, in `f64`.
    fn ref_conv(im: &Raster, kernel: &Kernel, x: i64, y: i64) -> Vec<f64> {
        let channels = im.format().channels();
        let kh = kernel.data.len() as i64;
        let kw = kernel.data[0].len() as i64;
        let (ax, ay) = (kw / 2, kh / 2);
        let mut sums = vec![0.0; channels];
        for j in 0..kh {
            for i in 0..kw {
                let sx = (x + i - ax).clamp(0, im.width() as i64 - 1) as u32;
                let sy = (y + j - ay).clamp(0, im.height() as i64 - 1) as u32;
                let px = im.getpoint(sx, sy);
                for (s, &p) in sums.iter_mut().zip(px.iter()) {
                    *s += kernel.data[j as usize][i as usize] * p;
                }
            }
        }
        sums.iter().map(|&s| s / kernel.scale).collect()
    }

    /// The four masks of test_convolution.py: sharp, blur, line, sobel.
    fn ported_masks() -> Vec<Kernel> {
        vec![
            Kernel {
                data: vec![
                    vec![-1.0, -1.0, -1.0],
                    vec![-1.0, 16.0, -1.0],
                    vec![-1.0, -1.0, -1.0],
                ],
                scale: 8.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                ],
                scale: 9.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 1.0, 1.0],
                    vec![-2.0, -2.0, -2.0],
                    vec![1.0, 1.0, 1.0],
                ],
                scale: 1.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 2.0, 1.0],
                    vec![0.0, 0.0, 0.0],
                    vec![-1.0, -2.0, -1.0],
                ],
                scale: 1.0,
            },
        ]
    }

    /// gaussmat integer, sigma 1, min_ampl 0.1: the exact 5x5 libvips
    /// matrix (`rint(20 * exp(-d2 / 2))`) with scale 124, the values
    /// test_create.py::test_gaussmat pins (max 20, centre 20).
    #[test]
    fn gaussmat_integer_exact_matrix() {
        let k = Kernel::gaussmat(1.0, 0.1, false, Precision::Integer);
        assert_eq!(k.width(), 5);
        assert_eq!(k.height(), 5);
        let expected = [
            [0.0, 2.0, 3.0, 2.0, 0.0],
            [2.0, 7.0, 12.0, 7.0, 2.0],
            [3.0, 12.0, 20.0, 12.0, 3.0],
            [2.0, 7.0, 12.0, 7.0, 2.0],
            [0.0, 2.0, 3.0, 2.0, 0.0],
        ];
        for (row, exp) in k.data.iter().zip(expected.iter()) {
            assert_eq!(row.as_slice(), exp.as_slice());
        }
        assert_eq!(k.scale, 124.0);
        assert_eq!(k.max(), 20.0);
        assert_eq!(k.data[2][2], 20.0);
    }

    /// gaussmat float: centre normalised to 1.0, elements are the raw
    /// exponentials, scale is their sum.
    #[test]
    fn gaussmat_float_values() {
        let k = Kernel::gaussmat(1.0, 0.1, false, Precision::Float);
        assert_eq!(k.width(), 5);
        assert_eq!(k.height(), 5);
        assert_eq!(k.data[2][2], 1.0);
        assert!((k.max() - 1.0).abs() < 1e-12);
        let mut sum = 0.0;
        for j in 0..5i64 {
            for i in 0..5i64 {
                let d = ((i - 2) * (i - 2) + (j - 2) * (j - 2)) as f64;
                let v = (-d / 2.0).exp();
                assert!((k.data[j as usize][i as usize] - v).abs() < 1e-12);
                sum += v;
            }
        }
        assert!((k.scale - sum).abs() < 1e-12);
    }

    /// Separable gaussmat is the centre row: 1xN with the same width as
    /// the 2D form, and the row sum as scale.
    #[test]
    fn gaussmat_separable_shapes() {
        let k2 = Kernel::gaussmat(2.0, 0.1, false, Precision::Float);
        let k1 = Kernel::gaussmat(2.0, 0.1, true, Precision::Float);
        assert_eq!(k2.width(), 9);
        assert_eq!(k2.height(), 9);
        assert_eq!(k1.width(), 9);
        assert_eq!(k1.height(), 1);
        assert_eq!(k1.data[0], k2.data[4]);
        let row_sum: f64 = k1.data[0].iter().sum();
        assert!((k1.scale - row_sum).abs() < 1e-12);
        assert_eq!(k1.data[0][4], 1.0);
    }

    /// logmat integer, sigma 1, min_ampl 0.1: 7x7 with centre and max 20
    /// (test_create.py::test_logmat), and the faithful scale of exactly
    /// zero that the C code stores for this mask (logmat.c has no zero
    /// guard, unlike gaussmat.c).
    #[test]
    fn logmat_integer_shape_and_zero_scale() {
        let k = Kernel::logmat(1.0, 0.1, false);
        assert_eq!(k.width(), 7);
        assert_eq!(k.height(), 7);
        assert_eq!(k.max(), 20.0);
        assert_eq!(k.data[3][3], 20.0);
        let sum: f64 = k.data.iter().flatten().sum();
        assert_eq!(sum, 0.0);
        assert_eq!(k.scale, 0.0);
    }

    /// logmat float separable: 7x1 with maximum 1.0, the values the
    /// libvips original pins with precision="float".
    #[test]
    fn logmat_float_separable() {
        let k = Kernel::logmat_with_precision(1.0, 0.1, true, Precision::Float);
        assert_eq!(k.width(), 7);
        assert_eq!(k.height(), 1);
        assert!((k.max() - 1.0).abs() < 1e-12);
        assert_eq!(k.data[0][3], 1.0);
        // Known LoG values at distance 1 and 2.
        assert!((k.data[0][2] - 0.5 * (-0.5f64).exp()).abs() < 1e-12);
        assert!((k.data[0][1] - (-(2.0f64).exp().recip())).abs() < 1e-12);
    }

    /// Convolving with a zero-scale mask (the sigma-1 integer logmat) is a
    /// typed error on both precision paths.
    #[test]
    fn conv_zero_scale_is_typed_error() {
        let im = noise_gray(8, 8, 7);
        let k = Kernel::logmat(1.0, 0.1, false);
        assert!(matches!(
            im.try_conv(&k, Precision::Integer),
            Err(ConvolutionError::ZeroScale)
        ));
        assert!(matches!(
            im.try_conv(&k, Precision::Float),
            Err(ConvolutionError::ZeroScale)
        ));
    }

    /// A 1x1 identity kernel copies the image exactly at integer
    /// precision and reproduces the values as float at float precision.
    #[test]
    fn conv_identity_kernel() {
        let id = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        for im in [noise_gray(9, 7, 1), noise_rgb(9, 7, 2)] {
            let out = im.conv(&id, Precision::Integer);
            assert_eq!(out.format(), im.format());
            assert_eq!(out.data(), im.data());

            let outf = im.conv(&id, Precision::Float);
            assert!(outf.format().is_float());
            assert_eq!(outf.format().channels(), im.format().channels());
            for y in 0..im.height() {
                for x in 0..im.width() {
                    assert_eq!(outf.getpoint(x, y), im.getpoint(x, y));
                }
            }
        }
    }

    /// Box blur on a hand-built 3x3 gray image: interior pixel and the
    /// edge-replicated corner match hand-computed values, integer
    /// truncation and float exactness included.
    #[test]
    fn conv_box_blur_hand_values() {
        let im = Raster::new(
            3,
            3,
            PixelFormat::Gray8,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
        .unwrap();
        let blur = Kernel {
            data: vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
            scale: 9.0,
        };

        let out = im.conv(&blur, Precision::Integer);
        // Interior: (450 + 4) / 9 = 50 (exact).
        assert_eq!(out.getpoint(1, 1), vec![50.0]);
        // Corner with edge replication: 4*10 + 2*20 + 2*40 + 50 = 210;
        // (210 + 4) / 9 truncates to 23.
        assert_eq!(out.getpoint(0, 0), vec![23.0]);

        let outf = im.conv(&blur, Precision::Float);
        assert_eq!(outf.getpoint(1, 1), vec![50.0]);
        let corner = outf.getpoint(0, 0)[0];
        assert!((corner - 210.0 / 9.0).abs() < 1e-5);
    }

    /// conv matches the scalar reference for all four ported masks on
    /// noise, mono and colour: float precision to 1e-3 against the exact
    /// reference, integer precision within 1.0 of the clipped reference.
    #[test]
    fn conv_matches_reference_on_noise() {
        for im in [noise_gray(20, 20, 3), noise_rgb(20, 20, 4)] {
            for kernel in ported_masks() {
                let outf = im.conv(&kernel, Precision::Float);
                let outi = im.conv(&kernel, Precision::Integer);
                assert_eq!(outi.format(), im.format());
                for (x, y) in [(5, 5), (10, 7), (18, 19), (0, 0)] {
                    let expected = ref_conv(&im, &kernel, x as i64, y as i64);
                    let gotf = outf.getpoint(x, y);
                    let goti = outi.getpoint(x, y);
                    for c in 0..expected.len() {
                        assert!(
                            (gotf[c] - expected[c]).abs() < 1e-3,
                            "float conv at ({x},{y}) band {c}: got {}, expected {}",
                            gotf[c],
                            expected[c]
                        );
                        let clipped = expected[c].clamp(0.0, 255.0);
                        assert!(
                            (goti[c] - clipped).abs() <= 1.0,
                            "integer conv at ({x},{y}) band {c}: got {}, expected {clipped}",
                            goti[c]
                        );
                    }
                }
            }
        }
    }

    /// The intize scale adjustment: a fractional mask whose rounded scale
    /// would skew brightness gets the libvips nudge. Mask [[0.4, 0.4]],
    /// scale 0.8: double result 1.0; rint mask [0, 0], rint scale 1,
    /// int result 0; adjusted scale rint(1 + (0 - 1)) = 0 -> 1. The offset
    /// rides along rounded but never rescaled, as the mask
    /// `vips_convi_gen` reads leaves it.
    #[test]
    fn intize_matches_vips_image_intize() {
        let fractional = Kernel {
            data: vec![vec![0.4, 0.4]],
            scale: 0.8,
        };
        let int = intize(&DenseKernel::new(&fractional).unwrap());
        assert_eq!(int.coeff, vec![0, 0]);
        assert_eq!(int.scale, 1);
        assert_eq!(int.offset, 0);

        // The scale is nudged from 0.8 to 1 here, and the offset does not
        // follow it: 127.6 rounds to 128 and stays there.
        let int = intize(&DenseKernel::new(&fractional).unwrap().with_offset(127.6));
        assert_eq!(int.scale, 1);
        assert_eq!(int.offset, 128);

        // The ported blur mask stays untouched: ints in, sum matches.
        let int = intize(
            &DenseKernel::new(&ported_masks()[1])
                .unwrap()
                .with_offset(128.0),
        );
        assert_eq!(int.coeff, vec![1; 9]);
        assert_eq!(int.scale, 9);
        assert_eq!(int.offset, 128);

        // rint() is round-half-to-even, like C under the default mode, for
        // the coefficients and for the offset alike.
        let halves = Kernel {
            data: vec![vec![0.5, 1.5, 2.5]],
            scale: 4.5,
        };
        let int = intize(&DenseKernel::new(&halves).unwrap().with_offset(0.5));
        assert_eq!(int.coeff, vec![0, 2, 2]);
        assert_eq!(int.offset, 0);
        let int = intize(&DenseKernel::new(&halves).unwrap().with_offset(1.5));
        assert_eq!(int.offset, 2);

        // A finite offset too large for a C `int` is clamped rather than
        // saturated to `i64::MAX`, so the summand add on the unsigned arm
        // cannot overflow. libvips reads the offset as `int` and gets the
        // same bound.
        let int = DenseKernel::new(&halves).unwrap();
        assert_eq!(intize(&int.with_offset(9.3e18)).offset, i64::from(i32::MAX));
        let int = DenseKernel::new(&halves).unwrap();
        assert_eq!(
            intize(&int.with_offset(-9.3e18)).offset,
            i64::from(i32::MIN)
        );
    }

    /// FNV-1a over a whole buffer, so a full `data()` comparison fits in
    /// one pinned constant.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// A `w x h` single-band float raster of deterministic noise spanning
    /// negatives, for the two arms that never clip.
    fn noise_float(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let samples: Vec<f32> = (0..w as usize * h as usize)
            .map(|_| f32::from(next()) - 128.0)
            .collect();
        Raster::from_f32_samples(w, h, float_format(1), &samples).unwrap()
    }

    /// The regression guard for the mask offset: at `offset` 0
    /// `conv_raster` reproduces the pre-offset output byte for byte. The
    /// digests are FNV-1a over the whole `data()` buffer, captured from
    /// the implementation before the summand existed, for all four
    /// `ported_masks()` at both precisions on the uchar, colour, and float
    /// input arms. The public `conv` has to agree with them as well.
    ///
    /// One case at the bottom deliberately does *not* reproduce the base
    /// bytes; see the comment there.
    #[test]
    fn conv_raster_at_offset_zero_reproduces_the_pre_offset_bytes() {
        // Per input, per mask: [integer-precision digest, float-precision
        // digest].
        let cases: [(Raster, [[u64; 2]; 4]); 3] = [
            (
                noise_gray(20, 20, 3),
                [
                    [0x174d_88ee_bcd8_e438, 0x3bd6_7d6f_0acf_a5e1],
                    [0x9fc6_4404_fea1_81aa, 0x846e_0e05_d2fc_f4a8],
                    [0xe3d0_476d_d341_4c0c, 0x2040_8791_364f_a5fa],
                    [0xc498_5b9d_52c5_d19d, 0x2161_3ea2_a21a_d2a2],
                ],
            ),
            (
                noise_rgb(20, 20, 4),
                [
                    [0xafe4_ffc1_6589_4821, 0x39af_ddc5_5d7b_224a],
                    [0x394a_993d_89dc_9a48, 0x384b_b952_bc40_fbd2],
                    [0xac22_fd7d_f031_f7ee, 0xc48f_d1e2_df84_e123],
                    [0x3fbb_9591_64dc_34f1, 0x9fa3_91a6_b1da_13dc],
                ],
            ),
            (
                noise_float(20, 20, 7),
                [
                    [0x3ac8_42a9_30db_a65a, 0x3ac8_42a9_30db_a65a],
                    [0x8c9f_1585_edd2_37cc, 0x9349_0ad8_c550_c5bd],
                    [0xa98a_afa4_f29e_4792, 0xa98a_afa4_f29e_4792],
                    [0xbed3_519d_d204_b58a, 0xbed3_519d_d204_b58a],
                ],
            ),
        ];

        for (im, per_mask) in &cases {
            for (mask, digests) in ported_masks().iter().zip(per_mask) {
                let dense = DenseKernel::new(mask).unwrap();
                for (precision, expected) in [
                    (Precision::Integer, digests[0]),
                    (Precision::Float, digests[1]),
                ] {
                    let out = conv_raster(im, &dense, precision).unwrap();
                    let shim = im.conv(mask, precision);
                    assert_eq!(
                        fnv1a(out.data()),
                        expected,
                        "offset 0 changed the {precision:?} output for {:?} input, mask {mask:?}",
                        im.format()
                    );
                    assert_eq!(out.format(), shim.format());
                    assert_eq!(
                        out.data(),
                        shim.data(),
                        "conv no longer agrees with the engine it delegates to"
                    );
                }
            }
        }

        // The one deliberate deviation from base, and the reason for the
        // `### Fixed` CHANGELOG entry. On the integer-precision float-input
        // arm the result went from `sum / iscale` to
        // `sum / iscale + ioffset as f64`, which is bit-identical for every
        // f64 except `-0.0`: adding `+0.0` promotes it to `+0.0`, and the
        // sign bit reaches `data()`. It takes a negative integer scale to
        // reach, which no `ported_masks()` entry has, and which the mask
        // below does. vips 8.18.4 writes `+0.0` here, C's
        // `(sum / scale) + offset` promoting the `int 0` the same way, so
        // base libviprs was the one diverging. libviprs' own float-precision
        // arm already wrote `+0.0`, and now both arms agree: the two
        // digests below are the same number.
        //
        // Row 0 is all zeros (the `-0.0` sites), row 1 all 5.0, so the rest
        // of the buffer is pinned alongside the sign bit.
        const BASE_MINUS_ZERO: u64 = 0x5e65_8c80_2e0c_dfa5;
        const VIPS_PLUS_ZERO: u64 = 0x453c_3d6d_5edc_8fa5;
        let samples: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        let im = Raster::from_f32_samples(4, 2, float_format(1), &samples).unwrap();
        let negative = Kernel {
            data: vec![vec![1.0, 1.0, 1.0]],
            scale: -3.0,
        };
        let dense = DenseKernel::new(&negative).unwrap();
        for precision in [Precision::Integer, Precision::Float] {
            let out = conv_raster(&im, &dense, precision).unwrap();
            assert_eq!(
                fnv1a(out.data()),
                VIPS_PLUS_ZERO,
                "negative-scale float input at {precision:?} precision no longer matches vips"
            );
        }
        assert_ne!(
            VIPS_PLUS_ZERO, BASE_MINUS_ZERO,
            "the base digest and the vips-matching one must differ, or this case pins nothing"
        );
    }

    /// The uchar recipe of `convolution/edge.c` (`offset` 128, `scale` 2 on
    /// a sobel mask): the offset lands before the clip, so a flat region
    /// reads as the 128 zero point, a falling edge saturates at 0 and a
    /// rising edge at 255. Without the offset the same input never reaches
    /// the top of the range.
    #[test]
    fn conv_raster_offset_clips_symmetrically_on_the_uchar_arm() {
        // Three columns wide so the horizontal edge replication is a
        // no-op; rows 2..=4 are the bright band.
        let rows: [u8; 7] = [0, 0, 255, 255, 255, 0, 0];
        let data: Vec<u8> = rows.iter().flat_map(|&v| [v, v, v]).collect();
        let im = Raster::new(3, 7, PixelFormat::Gray8, data).unwrap();
        let mask = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 2.0,
        };
        let dense = DenseKernel::new(&mask).unwrap().with_offset(128.0);

        let out = conv_raster(&im, &dense, Precision::Integer).unwrap();
        assert_eq!(out.format(), PixelFormat::Gray8);
        // sum 0 -> (0 + 1) / 2 + 128; sum -1020 -> -509 + 128 -> clipped to
        // 0; sum +1020 -> 510 + 128 -> clipped to 255.
        let got: Vec<f64> = (0..7).map(|y| out.getpoint(1, y)[0]).collect();
        assert_eq!(got, vec![128.0, 0.0, 0.0, 128.0, 255.0, 255.0, 128.0]);

        // The same mask with no offset: the low end still clips at 0, but
        // nothing reaches 255 and the flat region is black.
        let plain =
            conv_raster(&im, &DenseKernel::new(&mask).unwrap(), Precision::Integer).unwrap();
        let got: Vec<f64> = (0..7).map(|y| plain.getpoint(1, y)[0]).collect();
        assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0]);
    }

    /// A float input never clips, on either arm. Under integer precision
    /// the offset is the rounded one `vips_convi_gen` reads
    /// (`rint()`, half to even); under float precision `vips_convf_gen`
    /// keeps the raw double.
    #[test]
    fn conv_raster_offset_is_unclipped_on_the_float_input_arms() {
        let im = Raster::from_f32_samples(2, 2, float_format(1), &[10.0; 4]).unwrap();
        let mask = ported_masks().remove(1); // the 3x3 all-ones blur, scale 9

        // Integer precision, float input: intize gives an all-ones mask and
        // scale 9, so the sum is exactly 10 before the offset.
        for (offset, want) in [
            (1000.0, 1010.0),
            (-1000.0, -990.0),
            (0.5, 10.0),
            (1.5, 12.0),
        ] {
            let dense = DenseKernel::new(&mask).unwrap().with_offset(offset);
            let out = conv_raster(&im, &dense, Precision::Integer).unwrap();
            assert_eq!(out.format(), float_format(1));
            let got = out.getpoint(0, 0)[0];
            assert!(
                (got - want).abs() < 1e-4,
                "integer precision, float input, offset {offset}: got {got}, want {want}"
            );
        }

        // Float precision keeps the offset unrounded and unclipped.
        for (offset, want) in [(1000.0, 1010.0), (-1000.0, -990.0), (0.5, 10.5)] {
            let dense = DenseKernel::new(&mask).unwrap().with_offset(offset);
            let out = conv_raster(&im, &dense, Precision::Float).unwrap();
            let got = out.getpoint(1, 1)[0];
            assert!(
                (got - want).abs() < 1e-4,
                "float precision, offset {offset}: got {got}, want {want}"
            );
        }
    }

    /// A non-finite mask scalar is a typed error, not a panic and not a
    /// silent wrong answer. Before the guard, `f64::INFINITY` as an offset
    /// saturated `rint(offset) as i64` to `i64::MAX` and overflowed the
    /// summand add (a panic in debug, black pixels in release), `NaN` as an
    /// offset was silently dropped to 0 on the integer arms while the float
    /// arm returned `NaN`, a `NaN` scale slipped past the `scale == 0.0`
    /// test into an integer divide by zero, and an infinite scale past it
    /// into an all-zero image.
    #[test]
    fn conv_raster_rejects_a_non_finite_scale_or_offset() {
        let im = noise_gray(4, 4, 11);
        let ones = Kernel {
            data: vec![vec![1.0, 1.0, 1.0]],
            scale: 3.0,
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for precision in [Precision::Integer, Precision::Float] {
                let dense = DenseKernel::new(&ones).unwrap().with_offset(bad);
                assert!(matches!(
                    conv_raster(&im, &dense, precision),
                    Err(ConvolutionError::NonFiniteMaskParameter {
                        param: "offset",
                        value
                    }) if value.is_nan() == bad.is_nan()
                ));

                let scaled = Kernel {
                    data: ones.data.clone(),
                    scale: bad,
                };
                assert!(matches!(
                    im.try_conv(&scaled, precision),
                    Err(ConvolutionError::NonFiniteMaskParameter { param: "scale", .. })
                ));
            }
        }

        // A zero scale keeps its own more specific error.
        let zero = Kernel {
            data: ones.data.clone(),
            scale: 0.0,
        };
        assert!(matches!(
            im.try_conv(&zero, Precision::Integer),
            Err(ConvolutionError::ZeroScale)
        ));
    }

    /// A rotated mask keeps its scale and its offset. Going back through a
    /// `Kernel` literal would drop the offset silently, which is the whole
    /// reason the rotation lives on `DenseKernel`.
    #[test]
    fn rotating_a_dense_kernel_carries_the_scale_and_offset() {
        let sobel = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 2.0,
        };
        let dense = DenseKernel::new(&sobel).unwrap().with_offset(128.0);
        for spun in [dense.rot90(), dense.rot45(Angle45::D45)] {
            assert!(
                (spun.scale - 2.0).abs() < f64::EPSILON,
                "rotation dropped the scale: got {}",
                spun.scale
            );
            assert!(
                (spun.offset - 128.0).abs() < f64::EPSILON,
                "rotation dropped the offset: got {}",
                spun.offset
            );
            assert_eq!((spun.w, spun.h), (3, 3));
        }

        // rot90 of a 1xN is an Nx1, coefficients in order.
        let row = Kernel {
            data: vec![vec![1.0, 2.0, 3.0]],
            scale: 6.0,
        };
        let spun = DenseKernel::new(&row).unwrap().rot90();
        assert_eq!((spun.w, spun.h), (1, 3));
        assert_eq!(spun.coeff, vec![1.0, 2.0, 3.0]);
    }

    /// convsep equals the full 2D convolution with the outer-product mask
    /// at float precision (the Gaussian factorises exactly).
    #[test]
    fn convsep_equals_conv_2d_float() {
        let sep = Kernel::gaussmat(1.5, 0.1, true, Precision::Float);
        let full = Kernel::gaussmat(1.5, 0.1, false, Precision::Float);
        for im in [noise_gray(16, 12, 5), noise_rgb(16, 12, 6)] {
            let a = im.conv(&full, Precision::Float);
            let b = im.convsep(&sep, Precision::Float);
            for y in 0..im.height() {
                for x in 0..im.width() {
                    let pa = a.getpoint(x, y);
                    let pb = b.getpoint(x, y);
                    for c in 0..pa.len() {
                        assert!(
                            (pa[c] - pb[c]).abs() < 1e-2,
                            "sep mismatch at ({x},{y}) band {c}: {} vs {}",
                            pa[c],
                            pb[c]
                        );
                    }
                }
            }
        }
    }

    /// convsep accepts Nx1 masks too, and matches two manual conv passes
    /// (second pass with the 90-degree-rotated mask).
    #[test]
    fn convsep_vertical_mask_matches_manual_passes() {
        // Asymmetric 1D mask: rotation direction matters.
        let row = Kernel {
            data: vec![vec![1.0, 2.0, 4.0]],
            scale: 7.0,
        };
        let col = Kernel {
            data: vec![vec![1.0], vec![2.0], vec![4.0]],
            scale: 7.0,
        };
        let im = noise_gray(10, 10, 9);

        // Horizontal mask: pass 1 with the row, pass 2 with the column in
        // the same order (rot90 clockwise of a 1xN row keeps order).
        let manual = im
            .conv(&row, Precision::Integer)
            .conv(&col, Precision::Integer);
        assert_eq!(im.convsep(&row, Precision::Integer).data(), manual.data());

        // Vertical mask: rot90 clockwise of an Nx1 column reverses it.
        let row_rev = Kernel {
            data: vec![vec![4.0, 2.0, 1.0]],
            scale: 7.0,
        };
        let manual = im
            .conv(&col, Precision::Integer)
            .conv(&row_rev, Precision::Integer);
        assert_eq!(im.convsep(&col, Precision::Integer).data(), manual.data());
    }

    /// The internal 45-degree kernel rotation matches the classic sobel
    /// rotation (one clockwise ring step), and eight steps are an
    /// identity.
    #[test]
    fn rot45_kernel_sobel() {
        let sobel = vec![
            vec![1.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, -2.0, -1.0],
        ];
        let once = rot45_kernel(&sobel, Angle45::D45);
        assert_eq!(
            once,
            vec![
                vec![0.0, 1.0, 2.0],
                vec![-1.0, 0.0, 1.0],
                vec![-2.0, -1.0, 0.0],
            ]
        );
        let full = rot45_kernel(&sobel, Angle45::D0);
        assert_eq!(full, sobel);
        let mut spun = sobel.clone();
        for _ in 0..8 {
            spun = rot45_kernel(&spun, Angle45::D45);
        }
        assert_eq!(spun, sobel);
    }

    /// compass with times 1 is |conv|; with times 2 at D45 it matches the
    /// manual max/sum of the two rotated convolutions.
    #[test]
    fn compass_matches_manual_combination() {
        let sobel = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 1.0,
        };
        let sobel45 = Kernel {
            data: rot45_kernel(&sobel.data, Angle45::D45),
            scale: 1.0,
        };
        let im = noise_gray(12, 12, 11);

        let one = im.compass(&sobel, 1, Angle45::D45, Combine::Max, Precision::Float);
        let conv = im.conv(&sobel, Precision::Float);
        for y in 0..im.height() {
            for x in 0..im.width() {
                assert_eq!(one.getpoint(x, y)[0], conv.getpoint(x, y)[0].abs());
            }
        }

        let two_max = im.compass(&sobel, 2, Angle45::D45, Combine::Max, Precision::Float);
        let two_sum = im.compass(&sobel, 2, Angle45::D45, Combine::Sum, Precision::Float);
        let conv45 = im.conv(&sobel45, Precision::Float);
        for y in 0..im.height() {
            for x in 0..im.width() {
                let a = conv.getpoint(x, y)[0].abs();
                let b = conv45.getpoint(x, y)[0].abs();
                assert_eq!(two_max.getpoint(x, y)[0], a.max(b));
                let sum = two_sum.getpoint(x, y)[0];
                assert!((sum - (a + b)).abs() < 1e-3);
            }
        }
    }

    /// compass at integer precision keeps the unsigned format for Max and
    /// widens for Sum.
    #[test]
    fn compass_integer_formats() {
        let sharp = ported_masks().remove(0);
        let im = noise_rgb(10, 10, 13);
        let max = im.compass(&sharp, 3, Angle45::D45, Combine::Max, Precision::Integer);
        assert_eq!(max.format(), PixelFormat::Rgb8);
        let sum = im.compass(&sharp, 3, Angle45::D45, Combine::Sum, Precision::Integer);
        assert_eq!(sum.format(), PixelFormat::Rgb16);
        assert_eq!(max.width(), im.width());
        assert_eq!(sum.height(), im.height());
    }

    /// gaussblur is exactly gaussmat + convsep at both precisions, and a
    /// sigma below 0.2 returns a byte-identical copy.
    #[test]
    fn gaussblur_is_separable_gaussian() {
        let im = noise_rgb(14, 10, 15);
        for precision in [Precision::Integer, Precision::Float] {
            let mask = Kernel::gaussmat(1.4, 0.2, true, precision);
            let direct = im.convsep(&mask, precision);
            let blurred = im.gaussblur(1.4, 0.2, precision);
            assert_eq!(direct.data(), blurred.data());
        }
        let copy = im.gaussblur(0.1, 0.2, Precision::Integer);
        assert_eq!(copy.data(), im.data());
        assert_eq!(copy.format(), im.format());
    }

    /// fastcor hand values: a 1x1 template subtracts and squares every
    /// pixel; unsigned inputs use the integer path.
    #[test]
    fn fastcor_hand_values() {
        let im = Raster::new(2, 1, PixelFormat::Gray8, vec![0, 100]).unwrap();
        let t = Raster::new(1, 1, PixelFormat::Gray8, vec![200]).unwrap();
        let cor = im.fastcor(&t);
        assert!(cor.format().is_float());
        assert_eq!(cor.getpoint(0, 0), vec![40000.0]);
        assert_eq!(cor.getpoint(1, 0), vec![10000.0]);
    }

    /// fastcor finds an extracted patch: SSD is exactly zero at the patch
    /// centre and positive elsewhere; spcor scores exactly 1.0 there.
    #[test]
    fn correlation_peaks_at_match() {
        for im in [noise_gray(40, 40, 21), noise_rgb(40, 40, 22)] {
            let patch = im.extract(12, 17, 8, 8).unwrap();

            let cor = im.fastcor(&patch);
            assert_eq!(cor.width(), im.width());
            assert_eq!(cor.height(), im.height());
            let (v, x, y) = cor.minpos();
            assert_eq!(v, 0.0);
            assert_eq!((x, y), (16, 21));

            let ncc = im.spcor(&patch);
            let (v, x, y) = ncc.maxpos();
            assert!((v - 1.0).abs() < 1e-6, "NCC at match should be 1, got {v}");
            assert_eq!((x, y), (16, 21));
        }
    }

    /// A constant template is regarded as uncorrelated: spcor returns
    /// zero everywhere (the c2 == 0 arm).
    #[test]
    fn spcor_constant_template_is_uncorrelated() {
        let im = noise_gray(10, 10, 31);
        let t = Raster::new(3, 3, PixelFormat::Gray8, vec![7; 9]).unwrap();
        let ncc = im.spcor(&t);
        assert_eq!(ncc.min(), 0.0);
        assert_eq!(ncc.max(), 0.0);
    }

    /// Correlation templates must have the image's band count.
    #[test]
    fn correlation_band_mismatch_is_typed_error() {
        let im = noise_rgb(8, 8, 33);
        let t = noise_gray(3, 3, 34);
        assert!(matches!(
            im.try_fastcor(&t),
            Err(ConvolutionError::BandCountMismatch {
                image: 3,
                template: 1
            })
        ));
        assert!(matches!(
            im.try_spcor(&t),
            Err(ConvolutionError::BandCountMismatch { .. })
        ));
    }

    /// sharpen with m1 = m2 = 0 is a byte-exact identity for 8-bit mono
    /// and colour sources, across the ported sigma sweep.
    #[test]
    fn sharpen_zero_slopes_is_identity() {
        for im in [noise_gray(24, 18, 41), noise_rgb(24, 18, 42)] {
            for sigma in [0.5, 1.0, 1.5, 2.0] {
                let out = im.sharpen(sigma, 0.0, 0.0);
                assert_eq!(out.format(), im.format());
                assert_eq!(out.data(), im.data(), "sigma {sigma}");
            }
        }
    }

    /// sharpen keeps dimensions and format, changes pixels near an edge
    /// when m2 is positive, and leaves the a/b chroma codes untouched.
    #[test]
    fn sharpen_sharpens_luminance_only() {
        // A hard vertical edge: flat halves at 40 and 220.
        let mut data = vec![0u8; 20 * 10 * 3];
        for y in 0..10 {
            for x in 0..20 {
                let v = if x < 10 { 40 } else { 220 };
                let o = (y * 20 + x) * 3;
                data[o] = v;
                data[o + 1] = v;
                data[o + 2] = v;
            }
        }
        let im = Raster::new(20, 10, PixelFormat::Rgb8, data).unwrap();
        let sharp = im.sharpen(1.0, 1.0, 2.0);
        assert_eq!(sharp.width(), im.width());
        assert_eq!(sharp.height(), im.height());
        assert_eq!(sharp.format(), im.format());
        assert_ne!(sharp.data(), im.data(), "sharpening an edge must change it");

        // The chroma planes survive: a and b codes match before/after.
        let labs_in = im.colourspace(Interpretation::Labs);
        let labs_out = sharp.colourspace(Interpretation::Labs);
        let a = labs_in.f32_samples().unwrap();
        let b = labs_out.f32_samples().unwrap();
        for p in 0..(20 * 10) {
            // Allow the one-code wobble the sRGB re-quantisation can
            // introduce on the chroma of a changed pixel.
            assert!((a[p * 3 + 1] - b[p * 3 + 1]).abs() <= 1.0, "a band at {p}");
            assert!((a[p * 3 + 2] - b[p * 3 + 2]).abs() <= 1.0, "b band at {p}");
        }
    }

    /// sharpen on an image with no LabS route is a typed colour error.
    #[test]
    fn sharpen_unsupported_source_is_typed_error() {
        let two = PixelFormat::with_channels(2, 1).unwrap();
        let im = Raster::zeroed(4, 4, two).unwrap();
        assert!(matches!(
            im.try_sharpen(1.0, 0.0, 0.0),
            Err(ConvolutionError::Colour(_))
        ));
    }

    /// Kernel shape errors are typed: empty, ragged, non-separable for
    /// convsep, non-odd-square and zero times for compass.
    #[test]
    fn kernel_shape_errors() {
        let im = noise_gray(6, 6, 51);
        let empty = Kernel {
            data: vec![],
            scale: 1.0,
        };
        assert!(matches!(
            im.try_conv(&empty, Precision::Float),
            Err(ConvolutionError::EmptyKernel)
        ));
        let ragged = Kernel {
            data: vec![vec![1.0, 2.0], vec![3.0]],
            scale: 1.0,
        };
        assert!(matches!(
            im.try_conv(&ragged, Precision::Float),
            Err(ConvolutionError::RaggedKernel {
                row: 1,
                got: 1,
                expected: 2
            })
        ));
        let square = Kernel {
            data: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            scale: 4.0,
        };
        assert!(matches!(
            im.try_convsep(&square, Precision::Float),
            Err(ConvolutionError::NotSeparable {
                width: 2,
                height: 2
            })
        ));
        assert!(matches!(
            im.try_compass(&square, 1, Angle45::D45, Combine::Max, Precision::Float),
            Err(ConvolutionError::NotOddSquareKernel { .. })
        ));
        let odd = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        assert!(matches!(
            im.try_compass(&odd, 0, Angle45::D45, Combine::Max, Precision::Float),
            Err(ConvolutionError::ZeroTimes)
        ));
    }

    /// gaussmat/logmat argument validation matches the libvips bounds.
    #[test]
    fn mask_parameter_errors() {
        assert!(matches!(
            Kernel::try_gaussmat(0.0, 0.1, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter { param: "sigma", .. })
        ));
        assert!(matches!(
            Kernel::try_gaussmat(1.0, f64::NAN, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter {
                param: "min_ampl",
                ..
            })
        ));
        assert!(matches!(
            Kernel::try_logmat(-1.0, 0.1, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter { param: "sigma", .. })
        ));
    }

    /// conv at integer precision on a float raster keeps the float path:
    /// integer mask, real division, no clipping.
    #[test]
    fn conv_integer_on_float_raster() {
        let fmt = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(2, 1, fmt, &[-10.0, 350.5]).unwrap();
        let id = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        let out = im.conv(&id, Precision::Integer);
        assert!(out.format().is_float());
        assert_eq!(out.getpoint(0, 0), vec![-10.0]);
        assert_eq!(out.getpoint(1, 0), vec![350.5]);
    }

    /// conv at integer precision on 16-bit rasters clips into the 16-bit
    /// range rather than the 8-bit one.
    #[test]
    fn conv_integer_16bit_clip() {
        let hi = 60000u16.to_ne_bytes();
        let im = Raster::new(1, 1, PixelFormat::Gray16, vec![hi[0], hi[1]]).unwrap();
        let double = Kernel {
            data: vec![vec![2.0]],
            scale: 1.0,
        };
        let out = im.conv(&double, Precision::Integer);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![65535.0]);
    }
}
