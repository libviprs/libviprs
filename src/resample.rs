//! Resampling operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], [`crate::colour`], [`crate::morphology`],
//! [`crate::mosaicing`], and [`crate::convolution`]): box shrink, kernel
//! reduce, resize, affine transforms driven by an interpolator, the
//! similarity and rotate convenience forms, and coordinate-image remapping.
//! Operations that can fail on caller input exist in two forms, following
//! the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, ResampleError>` with
//!   typed errors for bad factors, singular matrices, and unknown kernel or
//!   interpolator names; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`shrink`, `reduce`, `resize`, `affine`, `similarity`, `rotate`,
//!   `mapim`) exactly, delegating to the `try_*` form. Where the ported
//!   surface passes libvips nicknames (`"bilinear"`, `"lanczos3"`), the
//!   panicking form takes `&str` and parses it; the `try_*` form takes the
//!   typed [`Interpolator`] / [`ReduceKernel`] enum.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::shrink`] | `vips_shrink` | box-filter downsample |
//! | [`Raster::shrinkh`] / [`Raster::shrinkv`] | `vips_shrinkh` / `vips_shrinkv` | one-axis integer box shrink |
//! | [`Raster::reduce`] | `vips_reduce` | kernel downsample |
//! | [`Raster::reduceh`] / [`Raster::reducev`] | `vips_reduceh` / `vips_reducev` | one-axis kernel downsample |
//! | [`Raster::resize`] | `vips_resize` | scale by a factor |
//! | [`Raster::affine`] | `vips_affine` | 2x2 matrix transform |
//! | [`Raster::similarity`] | `vips_similarity` | rotate + scale |
//! | [`Raster::rotate`] | `vips_rotate` | rotate by an angle in degrees |
//! | [`Raster::mapim`] | `vips_mapim` | remap through a coordinate image |
//! | [`Raster::constant_u8`] | `vips_black` + `linear` | constant one-band image |
//!
//! # Semantics shared with libvips
//!
//! * **Output sizes.** Downsampled dimensions round to nearest, half away
//!   from zero (`VIPS_ROUND_UINT`): `reduce` and fractional `shrink` produce
//!   `round(dim / factor)` and `resize` produces `round(dim * scale)`. The
//!   internal box-shrink passes that `reduce` runs for large factors use
//!   ceiling rounding, exactly as `vips_reduceh` invokes `vips_shrinkh`
//!   with `ceil` set.
//! * **`shrink` composition.** Integer factors run the plain box filter
//!   (`shrinkv` then `shrinkh`, integer mean with round-half-up). Fractional
//!   factors delegate to `reduce` with the default `lanczos3` kernel and a
//!   reducing gap of 1, reproducing `vips_shrink_build`.
//! * **`reduce` kernels.** Each output sample is a 1D convolution of
//!   `vips_reduce_get_points(kernel, shrink)` input samples with the kernel
//!   stretched by the shrink factor and normalised to unit sum, evaluated
//!   at the exact fractional offset. libvips quantises the offset to
//!   `VIPS_TRANSFORM_SCALE` fixed-point buckets as a speed optimisation;
//!   the masks here are computed in `f64` per output position, which is the
//!   same convolution without the quantisation error. Edges extend by
//!   replication (`VIPS_EXTEND_COPY`), so constant images are preserved
//!   exactly. `reduce` itself runs with gap 0 (no box pre-pass); `shrink`
//!   passes gap 1 and `resize` gap 2, as in libvips.
//! * **`resize` composition.** The scale is split per axis: any downscale
//!   runs `reducev` / `reduceh` with the chosen kernel (default `lanczos3`,
//!   gap 2), any residual upscale runs `affine` with the interpolator
//!   mapped from the kernel (`nearest` to nearest, `linear` to bilinear,
//!   everything else to bicubic), input displacement 0.5 for centre
//!   sampling, copy extension, and premultiplication skipped. The `nearest`
//!   kernel subsamples by the integer part first and enlarges integral
//!   factors by pixel replication (`vips_zoom`).
//! * **`affine` geometry.** The matrix `[a, b, c, d]` maps input to output
//!   as `x' = a*x + b*y + odx`, `y' = c*x + d*y + ody`. The default output
//!   area is the bounding box of the transformed input corners, rounded to
//!   nearest, computed from the matrix alone (the `odx` / `ody` / `idx` /
//!   `idy` displacements do not move the default area, matching the
//!   `vips_affine_build` ordering). Each output pixel is inverse-mapped and
//!   interpolated; positions whose floor falls outside `[-1, dim - 1]` are
//!   painted with the background, and interpolation taps outside the image
//!   read the [`Extend`] mode (background 0 by default), reproducing the
//!   one-pixel anti-aliased border of `vips_affine_gen`.
//! * **Premultiplied alpha.** Like `vips_affine`, images with an alpha band
//!   are premultiplied before interpolation and unpremultiplied afterwards
//!   unless [`AffineOptions::premultiplied`] says the input already is. The
//!   alpha ceiling is 255 for 8-bit and float samples and 65535 for 16-bit
//!   samples, the `vips_premultiply` defaults.
//! * **`similarity` / `rotate`.** `similarity(angle, scale)` builds the
//!   matrix `a = scale*cos, b = -scale*sin, c = -b, d = a` and calls
//!   `affine` with the default bilinear interpolator; `rotate(angle)` is
//!   `similarity(angle, 1.0)`. Note that libvips affine rotations sample on
//!   a grid displaced by the bounding-box rounding, so `rotate(90.0)` is
//!   the exact `rot90` permutation shifted one column right with a
//!   background seam in column 0. The unit tests pin this faithfully.
//! * **`mapim`.** The index image must have exactly two bands (band 0 is
//!   the source x, band 1 the source y). Coordinates inside
//!   `[-1, dim + 1)` are interpolated with background-extended taps (edge
//!   antialiasing); everything else, including NaN, paints the background.
//! * **Interpolators.** `nearest`, `bilinear`, and `bicubic` (Catmull-Rom,
//!   the libvips `VipsInterpolateBicubic` coefficients) are implemented.
//!   The libvips names `nohalo` and `lbb` are recognised by the parser but
//!   return [`ResampleError::InterpolatorNotImplemented`]: their
//!   minmod-subdivision resamplers (`nohalo.cpp`, `lbb.cpp`) are a
//!   dedicated later batch, and silently aliasing them to bicubic would
//!   misreport libvips semantics.
//!
//! # Example usage
//!
//! * [ported_resample tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/ported_resample.rs)

use crate::extract::Extend;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use std::f64::consts::PI;
use thiserror::Error;

/// Largest reduce mask supported, the libvips `MAX_POINT` from
/// `resample/presample.h`.
const MAX_POINT: usize = 2000;

/// Largest accepted shrink / reduce factor, the libvips argument ceiling on
/// `vips_shrink` and `vips_reduce`.
const MAX_FACTOR: f64 = 1_000_000.0;

/// Determinant threshold below which an affine matrix is treated as
/// singular, the libvips `TOO_SMALL` from `resample/transform.c`.
const TOO_SMALL: f64 = 2.0 * f64::MIN_POSITIVE;

/// Typed errors for the resampling operations in [`crate::resample`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ResampleError {
    /// The interpolator name is not a libvips interpolator nickname.
    #[error(
        "unknown interpolator {name:?}; expected \"nearest\", \"bilinear\", \"bicubic\", \"nohalo\" or \"lbb\""
    )]
    UnknownInterpolator { name: String },
    /// The interpolator is a recognised libvips nickname whose resampler is
    /// not implemented yet.
    #[error(
        "interpolator {name:?} is not implemented yet; use \"nearest\", \"bilinear\" or \"bicubic\""
    )]
    InterpolatorNotImplemented { name: &'static str },
    /// The kernel name is not a libvips `VipsKernel` nickname.
    #[error(
        "unknown kernel {name:?}; expected \"nearest\", \"linear\", \"cubic\", \"mitchell\", \"lanczos2\" or \"lanczos3\""
    )]
    UnknownKernel { name: String },
    /// A shrink or reduce factor is not a finite number in
    /// `1.0..=1_000_000.0`.
    #[error("{op} factor should be in 1.0..=1000000.0, got {factor}")]
    BadFactor { op: &'static str, factor: f64 },
    /// The reducing gap is below 1.0.
    #[error("reduce gap should be >= 1.0, got {gap}")]
    GapTooSmall { gap: f64 },
    /// The reduce mask would exceed the libvips `MAX_POINT` limit.
    #[error("reduce factor too large: {n_point}-point mask exceeds {max}")]
    FactorTooLarge { n_point: usize, max: usize },
    /// A resize scale is not a finite positive number.
    #[error("resize scale should be a finite positive number, got {scale}")]
    BadScale { scale: f64 },
    /// The output would have a zero dimension.
    #[error("image has shrunk to nothing")]
    ShrunkToNothing,
    /// The affine matrix is singular or near-singular.
    #[error("singular or near-singular matrix")]
    SingularMatrix,
    /// The affine output area is empty or does not fit in `u32` dimensions.
    #[error("bad affine output area {width}x{height}")]
    BadOutputArea { width: i64, height: i64 },
    /// The mapim index image does not have exactly two bands.
    #[error("mapim index image must have 2 bands, got {bands}")]
    IndexBands { bands: usize },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_resample<T>(op: &str, r: Result<T, ResampleError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// A point resampler for [`Raster::affine`] and [`Raster::mapim`] (libvips
/// `VipsInterpolate`).
///
/// The libvips nicknames `"nohalo"` and `"lbb"` parse but are not
/// implemented yet; see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Interpolator {
    /// Nearest neighbour: the sample whose floor position contains the
    /// point.
    Nearest,
    /// Bilinear blend of the surrounding 2x2 samples.
    Bilinear,
    /// Catmull-Rom bicubic over the surrounding 4x4 samples, the libvips
    /// `VipsInterpolateBicubic` coefficients.
    Bicubic,
}

impl Interpolator {
    /// Parse a libvips interpolator nickname.
    ///
    /// # Errors
    ///
    /// [`ResampleError::InterpolatorNotImplemented`] for `"nohalo"` and
    /// `"lbb"` (recognised libvips names whose resampler is a later batch),
    /// [`ResampleError::UnknownInterpolator`] for anything else.
    pub fn from_name(name: &str) -> Result<Self, ResampleError> {
        match name {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            "bicubic" => Ok(Self::Bicubic),
            "nohalo" => Err(ResampleError::InterpolatorNotImplemented { name: "nohalo" }),
            "lbb" => Err(ResampleError::InterpolatorNotImplemented { name: "lbb" }),
            _ => Err(ResampleError::UnknownInterpolator {
                name: name.to_string(),
            }),
        }
    }
}

/// A downsampling kernel for [`Raster::reduce`] and [`Raster::resize`]
/// (libvips `VipsKernel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceKernel {
    /// Point sample.
    Nearest,
    /// Triangle (tent) filter.
    Linear,
    /// Catmull-Rom cubic.
    Cubic,
    /// Mitchell-Netravali cubic (B = C = 1/3).
    Mitchell,
    /// Two-lobe Lanczos windowed sinc.
    Lanczos2,
    /// Three-lobe Lanczos windowed sinc, the libvips default.
    Lanczos3,
}

impl ReduceKernel {
    /// Parse a libvips `VipsKernel` nickname.
    ///
    /// # Errors
    ///
    /// [`ResampleError::UnknownKernel`] when the name is not one of
    /// `"nearest"`, `"linear"`, `"cubic"`, `"mitchell"`, `"lanczos2"`,
    /// `"lanczos3"`.
    pub fn from_name(name: &str) -> Result<Self, ResampleError> {
        match name {
            "nearest" => Ok(Self::Nearest),
            "linear" => Ok(Self::Linear),
            "cubic" => Ok(Self::Cubic),
            "mitchell" => Ok(Self::Mitchell),
            "lanczos2" => Ok(Self::Lanczos2),
            "lanczos3" => Ok(Self::Lanczos3),
            _ => Err(ResampleError::UnknownKernel {
                name: name.to_string(),
            }),
        }
    }

    /// Mask size for a shrink factor (`vips_reduce_get_points`). Always odd.
    fn points(self, shrink: f64) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Linear => 2 * shrink.round_ties_even() as usize + 1,
            Self::Cubic | Self::Mitchell | Self::Lanczos2 => {
                2 * (2.0 * shrink).round_ties_even() as usize + 1
            }
            Self::Lanczos3 => 2 * (3.0 * shrink).round_ties_even() as usize + 1,
        }
    }

    /// The kernel function at distance `x` from the centre
    /// (`resample/templates.h` `filter<K>`).
    fn filter(self, x: f64) -> f64 {
        match self {
            // Nearest masks are built directly in `mask`.
            Self::Nearest => 0.0,
            Self::Linear => {
                let ax = x.abs();
                if ax < 1.0 { 1.0 - ax } else { 0.0 }
            }
            Self::Cubic => cubic_filter(x, 0.0, 0.5),
            Self::Mitchell => cubic_filter(x, 1.0 / 3.0, 1.0 / 3.0),
            Self::Lanczos2 => {
                if (-2.0..=2.0).contains(&x) {
                    sinc_filter(x) * sinc_filter(x / 2.0)
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                if (-3.0..=3.0).contains(&x) {
                    sinc_filter(x) * sinc_filter(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Fill `c` with the mask for sub-pixel offset `x` in `[0, 1]`,
    /// normalised to unit sum (`vips_reduce_make_mask` over
    /// `calculate_coefficients`).
    fn mask(self, c: &mut [f64], shrink: f64, x: f64) {
        if self == Self::Nearest {
            c[0] = 1.0;
            return;
        }
        let n = c.len();
        let half = x + n as f64 / 2.0 - 1.0;
        let mut sum = 0.0;
        for (i, ci) in c.iter_mut().enumerate() {
            let xp = (i as f64 - half) / shrink;
            *ci = self.filter(xp);
            sum += *ci;
        }
        for ci in c.iter_mut() {
            *ci /= sum;
        }
    }

    /// The interpolator `vips_resize` upsizes with for this kernel
    /// (`vips_resize_interpolate`).
    fn upsize_interpolator(self) -> Interpolator {
        match self {
            Self::Nearest => Interpolator::Nearest,
            Self::Linear => Interpolator::Bilinear,
            _ => Interpolator::Bicubic,
        }
    }
}

/// The two-parameter cubic family from `resample/templates.h`
/// (`cubic_filter`): B = 0, C = 0.5 is Catmull-Rom, B = C = 1/3 is
/// Mitchell-Netravali.
fn cubic_filter(x: f64, b: f64, c: f64) -> f64 {
    let ax = x.abs();
    let ax2 = ax * ax;
    let ax3 = ax2 * ax;
    if ax <= 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax3 + (-18.0 + 12.0 * b + 6.0 * c) * ax2 + (6.0 - 2.0 * b))
            / 6.0
    } else if ax <= 2.0 {
        ((-b - 6.0 * c) * ax3
            + (6.0 * b + 30.0 * c) * ax2
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Normalised sinc (`resample/templates.h` `sinc_filter`).
fn sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * PI;
    x.sin() / x
}

/// Options for [`Raster::try_affine_with`], mirroring the optional
/// arguments of `vips_affine`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineOptions {
    /// Horizontal output displacement (`odx`).
    pub odx: f64,
    /// Vertical output displacement (`ody`).
    pub ody: f64,
    /// Horizontal input displacement (`idx`).
    pub idx: f64,
    /// Vertical input displacement (`idy`).
    pub idy: f64,
    /// Output rectangle `[left, top, width, height]` (`oarea`); the
    /// bounding box of the transformed input when `None`.
    pub oarea: Option<[i32; 4]>,
    /// How interpolation taps outside the input read (`extend`); the
    /// libvips default is [`Extend::Background`].
    pub extend: Extend,
    /// Background sample value for [`Extend::Background`] taps and for
    /// output pixels outside the transformed input (`background`,
    /// broadcast to every band).
    pub background: f64,
    /// The input already has premultiplied alpha, so skip the
    /// premultiply / unpremultiply pair (`premultiplied`).
    pub premultiplied: bool,
}

impl Default for AffineOptions {
    fn default() -> Self {
        Self {
            odx: 0.0,
            ody: 0.0,
            idx: 0.0,
            idy: 0.0,
            oarea: None,
            extend: Extend::Background,
            background: 0.0,
            premultiplied: false,
        }
    }
}

/// Options for [`Raster::try_resize_with`], mirroring the optional
/// arguments of `vips_resize`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeOptions {
    /// Vertical scale factor; the horizontal scale when `None`.
    pub vscale: Option<f64>,
    /// Downsampling kernel (libvips default `lanczos3`).
    pub kernel: ReduceKernel,
    /// Reducing gap (libvips default 2.0).
    pub gap: f64,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            vscale: None,
            kernel: ReduceKernel::Lanczos3,
            gap: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Sample plumbing
// ---------------------------------------------------------------------------

/// `VIPS_ROUND_UINT`: round a non-negative quantity to nearest, half up.
fn round_uint(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// `VIPS_ROUND_INT`: round to nearest, half away from zero, with the C
/// truncation-toward-zero cast.
fn round_int(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// Per-format sample layout: bytes per channel and float flag.
#[derive(Clone, Copy)]
struct SampleLayout {
    bpc: usize,
    is_float: bool,
    /// Sample ceiling for rounding and for the premultiply denominator
    /// (255 for 8-bit and float, 65535 for 16-bit, the `vips_premultiply`
    /// defaults).
    max: f64,
}

impl SampleLayout {
    fn of(format: PixelFormat) -> Self {
        let bpc = format.bytes_per_channel();
        let is_float = format.is_float();
        let max = if is_float {
            255.0
        } else if bpc == 2 {
            65535.0
        } else {
            255.0
        };
        Self { bpc, is_float, max }
    }

    /// Read sample `i` (flat sample index, not byte index) as `f64`.
    fn read(self, data: &[u8], i: usize) -> f64 {
        let o = i * self.bpc;
        if self.is_float {
            f64::from(f32::from_ne_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
            ]))
        } else if self.bpc == 2 {
            f64::from(u16::from_ne_bytes([data[o], data[o + 1]]))
        } else {
            f64::from(data[o])
        }
    }

    /// Write sample `i` from `f64`, rounding half up and clamping for the
    /// unsigned formats and storing raw `f32` for the float formats.
    fn write(self, data: &mut [u8], i: usize, v: f64) {
        let o = i * self.bpc;
        if self.is_float {
            data[o..o + 4].copy_from_slice(&(v as f32).to_ne_bytes());
        } else {
            let r = (v + 0.5).floor().clamp(0.0, self.max);
            if self.bpc == 2 {
                data[o..o + 2].copy_from_slice(&(r as u16).to_ne_bytes());
            } else {
                data[o] = r as u8;
            }
        }
    }
}

/// The axis a one-dimensional shrink or reduce runs along.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    Horizontal,
    Vertical,
}

// ---------------------------------------------------------------------------
// Box shrink
// ---------------------------------------------------------------------------

/// One-axis integer box shrink (`vips_shrinkh` / `vips_shrinkv`): each
/// output sample is the mean of `factor` consecutive input samples, with
/// round-half-up integer arithmetic for the unsigned formats. Blocks past
/// the edge replicate the edge sample (`VIPS_EXTEND_COPY`).
fn shrink_axis(src: &Raster, factor: u32, ceil: bool, axis: Axis) -> Result<Raster, ResampleError> {
    let op = match axis {
        Axis::Horizontal => "shrinkh",
        Axis::Vertical => "shrinkv",
    };
    if factor < 1 || f64::from(factor) > MAX_FACTOR {
        return Err(ResampleError::BadFactor {
            op,
            factor: f64::from(factor),
        });
    }
    if factor == 1 {
        return Ok(src.clone());
    }

    let (w, h) = (src.width() as usize, src.height() as usize);
    let dim = match axis {
        Axis::Horizontal => w,
        Axis::Vertical => h,
    };
    let f = factor as usize;
    let out_dim = if ceil {
        dim.div_ceil(f)
    } else {
        usize::try_from(round_uint(dim as f64 / f as f64)).unwrap_or(0)
    };
    if out_dim == 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let (ow, oh) = match axis {
        Axis::Horizontal => (out_dim, h),
        Axis::Vertical => (w, out_dim),
    };

    let format = src.format();
    let layout = SampleLayout::of(format);
    let bands = format.channels();
    let data = src.data();
    let mut out = vec![0u8; ow * oh * format.bytes_per_pixel()];

    for oy in 0..oh {
        for ox in 0..ow {
            for band in 0..bands {
                let sample = |k: usize| -> f64 {
                    let (sx, sy) = match axis {
                        Axis::Horizontal => ((ox * f + k).min(w - 1), oy),
                        Axis::Vertical => (ox, (oy * f + k).min(h - 1)),
                    };
                    layout.read(data, (sy * w + sx) * bands + band)
                };
                let oi = (oy * ow + ox) * bands + band;
                if layout.is_float {
                    let sum: f64 = (0..f).map(sample).sum();
                    layout.write(&mut out, oi, sum / f as f64);
                } else {
                    // Integer mean with round-half-up, the libvips
                    // `(sum + hshrink / 2) / hshrink`.
                    let sum: u64 = (0..f).map(|k| sample(k) as u64).sum();
                    let mean = (sum + f as u64 / 2) / f as u64;
                    layout.write(&mut out, oi, mean as f64);
                }
            }
        }
    }

    Ok(Raster::new(ow as u32, oh as u32, format, out)?)
}

// ---------------------------------------------------------------------------
// Kernel reduce
// ---------------------------------------------------------------------------

/// One-axis kernel reduce (`vips_reduceh` / `vips_reducev`), including the
/// gap-driven integer box pre-shrink.
fn reduce_axis(
    src: &Raster,
    shrink: f64,
    kernel: ReduceKernel,
    gap: f64,
    axis: Axis,
) -> Result<Raster, ResampleError> {
    let op = match axis {
        Axis::Horizontal => "reduceh",
        Axis::Vertical => "reducev",
    };
    if !shrink.is_finite() || !(1.0..=MAX_FACTOR).contains(&shrink) {
        return Err(ResampleError::BadFactor { op, factor: shrink });
    }

    let dim = match axis {
        Axis::Horizontal => src.width() as usize,
        Axis::Vertical => src.height() as usize,
    };
    let out_dim = round_uint(dim as f64 / shrink);
    if out_dim <= 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let out_dim = out_dim as usize;

    // How many samples we invent in the input, negative for discarding
    // (`extra_pixels` in vips_reduceh_build).
    let mut extra = out_dim as f64 * shrink - dim as f64;
    let mut shrink = shrink;

    // Gap-driven integer box shrink first (`vips_shrinkh` with ceil), then
    // reduce the residual.
    let mut boxed: Option<Raster> = None;
    if gap > 0.0 && kernel != ReduceKernel::Nearest {
        if gap < 1.0 {
            return Err(ResampleError::GapTooSmall { gap });
        }
        let int_shrink = (dim as f64 / out_dim as f64 / gap).floor().max(1.0) as u32;
        if int_shrink > 1 {
            boxed = Some(shrink_axis(src, int_shrink, true, axis)?);
            shrink /= f64::from(int_shrink);
            extra /= f64::from(int_shrink);
        }
    }
    let cur = boxed.as_ref().unwrap_or(src);

    if shrink == 1.0 {
        return Ok(cur.clone());
    }

    let n = kernel.points(shrink);
    if n > MAX_POINT {
        return Err(ResampleError::FactorTooLarge {
            n_point: n,
            max: MAX_POINT,
        });
    }
    // The embed margin, `VIPS_CEIL(n_point / 2.0) - 1`; `n` is always odd.
    let margin = (n - 1) / 2;
    // Discard invented pixels equally from both ends
    // (`hoffset` / `voffset` in the vips builds).
    let offset = (1.0 + extra) / 2.0 - 1.0;

    let (w, h) = (cur.width() as usize, cur.height() as usize);
    let cur_dim = match axis {
        Axis::Horizontal => w,
        Axis::Vertical => h,
    };
    let (ow, oh) = match axis {
        Axis::Horizontal => (out_dim, h),
        Axis::Vertical => (w, out_dim),
    };

    // Precompute the mask and first tap for every output position along the
    // axis. libvips quantises the sub-pixel offset into fixed-point tables;
    // computing the mask per position in f64 is the same convolution
    // without the quantisation.
    let mut masks: Vec<(i64, Vec<f64>)> = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let x = (i as f64 + 0.5) * shrink - 0.5 - offset;
        let ix = x.floor();
        let t = x - ix;
        let mut c = vec![0.0f64; n];
        kernel.mask(&mut c, shrink, t);
        masks.push((ix as i64 - margin as i64, c));
    }

    let format = cur.format();
    let layout = SampleLayout::of(format);
    let bands = format.channels();
    let data = cur.data();
    let mut out = vec![0u8; ow * oh * format.bytes_per_pixel()];

    let clamp_dim = |v: i64| -> usize { v.clamp(0, cur_dim as i64 - 1) as usize };
    for oy in 0..oh {
        for ox in 0..ow {
            let (start, c) = match axis {
                Axis::Horizontal => &masks[ox],
                Axis::Vertical => &masks[oy],
            };
            for band in 0..bands {
                let mut acc = 0.0f64;
                for (k, ck) in c.iter().enumerate() {
                    let tap = clamp_dim(start + k as i64);
                    let (sx, sy) = match axis {
                        Axis::Horizontal => (tap, oy),
                        Axis::Vertical => (ox, tap),
                    };
                    acc += ck * layout.read(data, (sy * w + sx) * bands + band);
                }
                layout.write(&mut out, (oy * ow + ox) * bands + band, acc);
            }
        }
    }

    Ok(Raster::new(ow as u32, oh as u32, format, out)?)
}

// ---------------------------------------------------------------------------
// Point sampling for affine / mapim
// ---------------------------------------------------------------------------

/// Bounds-aware sample fetch applying an [`Extend`] rule to taps outside
/// the image, the equivalent of the `vips_embed` borders the libvips
/// resamplers add.
struct TapFetch<'a> {
    data: &'a [u8],
    w: i64,
    h: i64,
    bands: usize,
    layout: SampleLayout,
    extend: Extend,
    background: f64,
}

impl TapFetch<'_> {
    fn new(src: &Raster, extend: Extend, background: f64) -> TapFetch<'_> {
        TapFetch {
            data: src.data(),
            w: i64::from(src.width()),
            h: i64::from(src.height()),
            bands: src.format().channels(),
            layout: SampleLayout::of(src.format()),
            extend,
            background,
        }
    }

    /// Fold a coordinate into `0..dim` per the extend rule; `None` paints
    /// the fill value.
    fn resolve(&self, v: i64, dim: i64) -> Option<i64> {
        if (0..dim).contains(&v) {
            return Some(v);
        }
        match self.extend {
            Extend::Copy => Some(v.clamp(0, dim - 1)),
            Extend::Repeat => Some(v.rem_euclid(dim)),
            Extend::Mirror => {
                // Reflect with the edge sample duplicated, period 2 * dim.
                let m = v.rem_euclid(2 * dim);
                Some(if m < dim { m } else { 2 * dim - 1 - m })
            }
            Extend::Black | Extend::White | Extend::Background => None,
        }
    }

    fn fill_value(&self) -> f64 {
        match self.extend {
            Extend::White => self.layout.max,
            Extend::Background => self.background,
            _ => 0.0,
        }
    }

    /// Fetch the full pixel at `(x, y)` into `px`, applying the extend
    /// rule, and premultiply the colour bands when asked.
    fn fetch(&self, x: i64, y: i64, premultiply: bool, px: &mut [f64]) {
        match (self.resolve(x, self.w), self.resolve(y, self.h)) {
            (Some(x), Some(y)) => {
                let base = (y as usize * self.w as usize + x as usize) * self.bands;
                for (b, v) in px.iter_mut().enumerate() {
                    *v = self.layout.read(self.data, base + b);
                }
            }
            _ => px.fill(self.fill_value()),
        }
        if premultiply {
            let alpha = px[self.bands - 1] / self.layout.max;
            for v in px.iter_mut().take(self.bands - 1) {
                *v *= alpha;
            }
        }
    }
}

/// Interpolate every band at the continuous position `(x, y)`, writing the
/// per-band result (premultiplied when `premultiply` is set) into `out`.
fn interpolate_at(
    fetch: &TapFetch<'_>,
    interp: Interpolator,
    x: f64,
    y: f64,
    premultiply: bool,
    px: &mut [f64],
    out: &mut [f64],
) {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    match interp {
        Interpolator::Nearest => {
            fetch.fetch(x0, y0, premultiply, out);
        }
        Interpolator::Bilinear => {
            let u = x - x0 as f64;
            let v = y - y0 as f64;
            let wx = [1.0 - u, u];
            let wy = [1.0 - v, v];
            out.fill(0.0);
            for (j, wyj) in wy.iter().enumerate() {
                for (i, wxi) in wx.iter().enumerate() {
                    let wgt = wyj * wxi;
                    if wgt == 0.0 {
                        continue;
                    }
                    fetch.fetch(x0 + i as i64, y0 + j as i64, premultiply, px);
                    for (o, p) in out.iter_mut().zip(px.iter()) {
                        *o += wgt * p;
                    }
                }
            }
        }
        Interpolator::Bicubic => {
            let mut cx = [0.0f64; 4];
            let mut cy = [0.0f64; 4];
            catmull_coefficients(&mut cx, x - x0 as f64);
            catmull_coefficients(&mut cy, y - y0 as f64);
            out.fill(0.0);
            for (j, cyj) in cy.iter().enumerate() {
                for (i, cxi) in cx.iter().enumerate() {
                    let wgt = cyj * cxi;
                    if wgt == 0.0 {
                        continue;
                    }
                    fetch.fetch(x0 - 1 + i as i64, y0 - 1 + j as i64, premultiply, px);
                    for (o, p) in out.iter_mut().zip(px.iter()) {
                        *o += wgt * p;
                    }
                }
            }
        }
    }
}

/// Catmull-Rom coefficients for offset `x` in `[0, 1]`
/// (`calculate_coefficients_catmull` in `resample/templates.h`).
fn catmull_coefficients(c: &mut [f64; 4], x: f64) {
    let cr1 = 1.0 - x;
    let cr2 = -0.5 * x;
    let cr3 = cr1 * cr2;
    let cone = cr1 * cr3;
    let cfou = x * cr3;
    let cr4 = cfou - cone;
    c[0] = cone;
    c[1] = cr1 - cone + cr4;
    c[2] = x - cfou - cr4;
    c[3] = cfou;
}

/// Unpremultiply an interpolated pixel in place (`vips_unpremultiply`):
/// colour bands scale by `max / alpha`, zero alpha zeroes them.
fn unpremultiply(px: &mut [f64], max: f64) {
    let bands = px.len();
    let alpha = px[bands - 1];
    if alpha == 0.0 {
        for v in px.iter_mut().take(bands - 1) {
            *v = 0.0;
        }
    } else {
        for v in px.iter_mut().take(bands - 1) {
            *v *= max / alpha;
        }
    }
}

// ---------------------------------------------------------------------------
// Nearest-kernel resize helpers
// ---------------------------------------------------------------------------

/// Point-sample every `xfac`-th / `yfac`-th pixel (`vips_subsample`).
/// Output dimensions truncate, matching the libvips integer division.
fn subsample(src: &Raster, xfac: u32, yfac: u32) -> Result<Raster, ResampleError> {
    let ow = src.width() / xfac;
    let oh = src.height() / yfac;
    if ow == 0 || oh == 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let format = src.format();
    let bpp = format.bytes_per_pixel();
    let w = src.width() as usize;
    let data = src.data();
    let mut out = vec![0u8; ow as usize * oh as usize * bpp];
    for oy in 0..oh as usize {
        for ox in 0..ow as usize {
            let src_off = (oy * yfac as usize * w + ox * xfac as usize) * bpp;
            let dst_off = (oy * ow as usize + ox) * bpp;
            out[dst_off..dst_off + bpp].copy_from_slice(&data[src_off..src_off + bpp]);
        }
    }
    Ok(Raster::new(ow, oh, format, out)?)
}

/// Integral pixel replication (`vips_zoom`).
fn zoom(src: &Raster, xfac: u32, yfac: u32) -> Result<Raster, ResampleError> {
    let ow = u64::from(src.width()) * u64::from(xfac);
    let oh = u64::from(src.height()) * u64::from(yfac);
    let (Ok(ow), Ok(oh)) = (u32::try_from(ow), u32::try_from(oh)) else {
        return Err(ResampleError::BadOutputArea {
            width: (u64::from(src.width()) * u64::from(xfac)) as i64,
            height: (u64::from(src.height()) * u64::from(yfac)) as i64,
        });
    };
    let format = src.format();
    let bpp = format.bytes_per_pixel();
    let w = src.width() as usize;
    let data = src.data();
    let mut out = Raster::zeroed(ow, oh, format)?;
    let ow = ow as usize;
    let buf = out.data_mut();
    for oy in 0..oh as usize {
        let sy = oy / yfac as usize;
        for ox in 0..ow {
            let sx = ox / xfac as usize;
            let src_off = (sy * w + sx) * bpp;
            let dst_off = (oy * ow + ox) * bpp;
            buf[dst_off..dst_off + bpp].copy_from_slice(&data[src_off..src_off + bpp]);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Fallible form of [`Raster::shrink`].
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadFactor`] unless both factors are finite numbers
    /// in `1.0..=1_000_000.0`, [`ResampleError::ShrunkToNothing`] when a
    /// dimension would reach zero, or [`ResampleError::Raster`] on
    /// allocation failure.
    pub fn try_shrink(&self, hshrink: f64, vshrink: f64) -> Result<Raster, ResampleError> {
        for factor in [hshrink, vshrink] {
            if !factor.is_finite() || !(1.0..=MAX_FACTOR).contains(&factor) {
                return Err(ResampleError::BadFactor {
                    op: "shrink",
                    factor,
                });
            }
        }
        if hshrink.fract() != 0.0 || vshrink.fract() != 0.0 {
            // Fractional factors delegate to reduce with the default
            // lanczos3 kernel and gap 1 (`vips_shrink_build`).
            let t = reduce_axis(self, vshrink, ReduceKernel::Lanczos3, 1.0, Axis::Vertical)?;
            reduce_axis(&t, hshrink, ReduceKernel::Lanczos3, 1.0, Axis::Horizontal)
        } else {
            let t = shrink_axis(self, vshrink as u32, false, Axis::Vertical)?;
            shrink_axis(&t, hshrink as u32, false, Axis::Horizontal)
        }
    }

    /// Shrink by a pair of factors with a box filter (libvips
    /// `vips_shrink`). Integer factors run the plain box average; for
    /// fractional factors the residual is reduced with the default
    /// `lanczos3` kernel, exactly as libvips composes it. Output dimensions
    /// are `round(dim / factor)`. Panicking form of [`Raster::try_shrink`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrink`].
    #[track_caller]
    pub fn shrink(&self, hshrink: f64, vshrink: f64) -> Raster {
        expect_resample("shrink", self.try_shrink(hshrink, vshrink))
    }

    /// Fallible form of [`Raster::shrinkh`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_shrink`].
    pub fn try_shrinkh(&self, hshrink: u32) -> Result<Raster, ResampleError> {
        shrink_axis(self, hshrink, false, Axis::Horizontal)
    }

    /// Horizontal integer box shrink (libvips `vips_shrinkh`); the output
    /// width is `round(width / hshrink)`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrinkh`].
    #[track_caller]
    pub fn shrinkh(&self, hshrink: u32) -> Raster {
        expect_resample("shrinkh", self.try_shrinkh(hshrink))
    }

    /// Fallible form of [`Raster::shrinkv`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_shrink`].
    pub fn try_shrinkv(&self, vshrink: u32) -> Result<Raster, ResampleError> {
        shrink_axis(self, vshrink, false, Axis::Vertical)
    }

    /// Vertical integer box shrink (libvips `vips_shrinkv`); the output
    /// height is `round(height / vshrink)`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrinkv`].
    #[track_caller]
    pub fn shrinkv(&self, vshrink: u32) -> Raster {
        expect_resample("shrinkv", self.try_shrinkv(vshrink))
    }

    /// Fallible form of [`Raster::reduce`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadFactor`] unless both factors are finite numbers
    /// in `1.0..=1_000_000.0`, [`ResampleError::FactorTooLarge`] when the
    /// mask would exceed the libvips `MAX_POINT`,
    /// [`ResampleError::ShrunkToNothing`] when a dimension would reach
    /// zero, or [`ResampleError::Raster`] on allocation failure.
    pub fn try_reduce(
        &self,
        hshrink: f64,
        vshrink: f64,
        kernel: ReduceKernel,
    ) -> Result<Raster, ResampleError> {
        let t = reduce_axis(self, vshrink, kernel, 0.0, Axis::Vertical)?;
        reduce_axis(&t, hshrink, kernel, 0.0, Axis::Horizontal)
    }

    /// Downsample with an anti-aliasing kernel (libvips `vips_reduce`):
    /// vertical pass then horizontal pass, no box pre-pass (gap 0), output
    /// dimensions `round(dim / factor)`. The kernel is a libvips nickname:
    /// `"nearest"`, `"linear"`, `"cubic"`, `"mitchell"`, `"lanczos2"`, or
    /// `"lanczos3"`. Panicking form of [`Raster::try_reduce`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reduce`].
    #[track_caller]
    pub fn reduce(&self, hshrink: f64, vshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reduce", ReduceKernel::from_name(kernel));
        expect_resample("reduce", self.try_reduce(hshrink, vshrink, kernel))
    }

    /// Fallible form of [`Raster::reduceh`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_reduce`].
    pub fn try_reduceh(&self, hshrink: f64, kernel: ReduceKernel) -> Result<Raster, ResampleError> {
        reduce_axis(self, hshrink, kernel, 0.0, Axis::Horizontal)
    }

    /// Horizontal kernel reduce (libvips `vips_reduceh`); the kernel is a
    /// libvips nickname as in [`Raster::reduce`].
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reduceh`].
    #[track_caller]
    pub fn reduceh(&self, hshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reduceh", ReduceKernel::from_name(kernel));
        expect_resample("reduceh", self.try_reduceh(hshrink, kernel))
    }

    /// Fallible form of [`Raster::reducev`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_reduce`].
    pub fn try_reducev(&self, vshrink: f64, kernel: ReduceKernel) -> Result<Raster, ResampleError> {
        reduce_axis(self, vshrink, kernel, 0.0, Axis::Vertical)
    }

    /// Vertical kernel reduce (libvips `vips_reducev`); the kernel is a
    /// libvips nickname as in [`Raster::reduce`].
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reducev`].
    #[track_caller]
    pub fn reducev(&self, vshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reducev", ReduceKernel::from_name(kernel));
        expect_resample("reducev", self.try_reducev(vshrink, kernel))
    }

    /// Fallible form of [`Raster::resize`] with explicit options.
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadScale`] unless the scales are finite positive
    /// numbers, plus any error of the underlying reduce or affine pass.
    pub fn try_resize_with(
        &self,
        scale: f64,
        options: ResizeOptions,
    ) -> Result<Raster, ResampleError> {
        let mut hscale = scale;
        let mut vscale = options.vscale.unwrap_or(scale);
        for s in [hscale, vscale] {
            if !s.is_finite() || s <= 0.0 {
                return Err(ResampleError::BadScale { scale: s });
            }
        }

        let mut cur = self.clone();

        // The nearest kernel subsamples the integer part first
        // (`vips_resize_build`).
        if options.kernel == ReduceKernel::Nearest {
            let int_shrink = |dim: u32, s: f64| -> u32 {
                let f = if options.gap < 1.0 {
                    (1.0 / s).floor()
                } else {
                    let target = round_uint(f64::from(dim) * s).max(1) as f64;
                    (f64::from(dim) / target / options.gap).floor()
                };
                f.max(1.0) as u32
            };
            let int_h = int_shrink(cur.width(), hscale);
            let int_v = int_shrink(cur.height(), vscale);
            if int_h > 1 || int_v > 1 {
                cur = subsample(&cur, int_h, int_v)?;
                hscale *= f64::from(int_h);
                vscale *= f64::from(int_v);
            }
        }

        // Don't let either axis drop below one pixel.
        hscale = hscale.max(1.0 / f64::from(cur.width()));
        vscale = vscale.max(1.0 / f64::from(cur.height()));

        // Any residual downsizing, vertical then horizontal.
        if vscale < 1.0 {
            cur = reduce_axis(
                &cur,
                1.0 / vscale,
                options.kernel,
                options.gap,
                Axis::Vertical,
            )?;
        }
        if hscale < 1.0 {
            cur = reduce_axis(
                &cur,
                1.0 / hscale,
                options.kernel,
                options.gap,
                Axis::Horizontal,
            )?;
        }

        // Any upsizing: affine with the interpolator mapped from the
        // kernel, or pixel replication for integral nearest enlargement.
        if hscale > 1.0 || vscale > 1.0 {
            let nearest = options.kernel == ReduceKernel::Nearest;
            if nearest && hscale.fract() == 0.0 && vscale.fract() == 0.0 {
                cur = zoom(&cur, hscale as u32, vscale as u32)?;
            } else {
                let id = if nearest { 0.0 } else { 0.5 };
                let matrix = [hscale.max(1.0), 0.0, 0.0, vscale.max(1.0)];
                cur = cur.try_affine_with(
                    matrix,
                    options.kernel.upsize_interpolator(),
                    AffineOptions {
                        idx: id,
                        idy: id,
                        extend: Extend::Copy,
                        premultiplied: true,
                        ..AffineOptions::default()
                    },
                )?;
            }
        }

        Ok(cur)
    }

    /// Fallible form of [`Raster::resize`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_resize_with`].
    pub fn try_resize(&self, scale: f64) -> Result<Raster, ResampleError> {
        self.try_resize_with(scale, ResizeOptions::default())
    }

    /// Resize by a scale factor (libvips `vips_resize`): reduce with the
    /// default `lanczos3` kernel for downscales, affine with bicubic for
    /// upscales. Output dimensions are `round(dim * scale)`. Panicking form
    /// of [`Raster::try_resize`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_resize`].
    #[track_caller]
    pub fn resize(&self, scale: f64) -> Raster {
        expect_resample("resize", self.try_resize(scale))
    }

    /// Resize with explicit [`ResizeOptions`]. Panicking form of
    /// [`Raster::try_resize_with`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_resize_with`].
    #[track_caller]
    pub fn resize_with(&self, scale: f64, options: ResizeOptions) -> Raster {
        expect_resample("resize", self.try_resize_with(scale, options))
    }

    /// Fallible form of [`Raster::affine`] with explicit options.
    ///
    /// # Errors
    ///
    /// [`ResampleError::SingularMatrix`] when the matrix cannot be
    /// inverted, [`ResampleError::BadOutputArea`] when the output area is
    /// empty or overflows `u32` dimensions, or [`ResampleError::Raster`] on
    /// allocation failure.
    pub fn try_affine_with(
        &self,
        matrix: [f64; 4],
        interpolate: Interpolator,
        options: AffineOptions,
    ) -> Result<Raster, ResampleError> {
        let [a, b, c, d] = matrix;
        let det = a * d - b * c;
        if det.abs() < TOO_SMALL {
            return Err(ResampleError::SingularMatrix);
        }
        let (ia, ib, ic, idd) = (d / det, -b / det, -c / det, a / det);

        let w = i64::from(self.width());
        let h = i64::from(self.height());

        // Default output area: bounding box of the transformed input
        // corners, rounded to nearest. Computed from the matrix alone; the
        // displacement options do not move it (`vips__transform_set_area`
        // runs before they are applied in `vips_affine_build`).
        let default_oarea = {
            let corners = [
                (0.0, 0.0),
                (w as f64, 0.0),
                (0.0, h as f64),
                (w as f64, h as f64),
            ];
            let xs = corners.map(|(x, y)| a * x + b * y);
            let ys = corners.map(|(x, y)| c * x + d * y);
            let fold =
                |v: [f64; 4], f: fn(f64, f64) -> f64| v.into_iter().reduce(f).expect("non-empty");
            let left = fold(xs, f64::min);
            let right = fold(xs, f64::max);
            let top = fold(ys, f64::min);
            let bottom = fold(ys, f64::max);
            [
                round_int(left),
                round_int(top),
                round_int(right - left),
                round_int(bottom - top),
            ]
        };
        let oarea = options
            .oarea
            .map(|o| o.map(i64::from))
            .unwrap_or(default_oarea);
        let (ow, oh) = (oarea[2], oarea[3]);
        if ow < 1 || oh < 1 || u32::try_from(ow).is_err() || u32::try_from(oh).is_err() {
            return Err(ResampleError::BadOutputArea {
                width: ow,
                height: oh,
            });
        }

        // Identity transform writing the full input straight through is a
        // copy (`vips__transform_isidentity` shortcut).
        if matrix == [1.0, 0.0, 0.0, 1.0]
            && options.odx == 0.0
            && options.ody == 0.0
            && options.idx == 0.0
            && options.idy == 0.0
            && oarea == [0, 0, w, h]
        {
            return Ok(self.clone());
        }

        let format = self.format();
        let layout = SampleLayout::of(format);
        let bands = format.channels();
        let premultiply = format.has_alpha() && !options.premultiplied;
        let fetch = TapFetch::new(self, options.extend, options.background);

        let mut out = Raster::zeroed(ow as u32, oh as u32, format)?;
        let buf = out.data_mut();
        let mut px = vec![0.0f64; bands];
        let mut acc = vec![0.0f64; bands];

        for y in 0..oh {
            let oy = (y + oarea[1]) as f64 - options.ody;
            for x in 0..ow {
                let ox = (x + oarea[0]) as f64 - options.odx;
                let ix = ia * ox + ib * oy - options.idx;
                let iy = ic * ox + idd * oy - options.idy;
                let oi = (y * ow + x) as usize * bands;
                let (fx, fy) = (ix.floor(), iy.floor());
                if fx >= -1.0 && fx <= (w - 1) as f64 && fy >= -1.0 && fy <= (h - 1) as f64 {
                    interpolate_at(&fetch, interpolate, ix, iy, premultiply, &mut px, &mut acc);
                    if premultiply {
                        unpremultiply(&mut acc, layout.max);
                    }
                    for (bi, v) in acc.iter().enumerate() {
                        layout.write(buf, oi + bi, *v);
                    }
                } else {
                    for bi in 0..bands {
                        layout.write(buf, oi + bi, options.background);
                    }
                }
            }
        }

        Ok(out)
    }

    /// Fallible form of [`Raster::affine`], taking the typed interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_affine(
        &self,
        matrix: [f64; 4],
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        self.try_affine_with(matrix, interpolate, AffineOptions::default())
    }

    /// Transform by the 2x2 matrix `[a, b, c, d]` (libvips `vips_affine`):
    /// input `(x, y)` maps to output `(a*x + b*y, c*x + d*y)`, the output
    /// is the rounded bounding box of the transformed input, and each
    /// output pixel is inverse-mapped and interpolated. The interpolator is
    /// a libvips nickname: `"nearest"`, `"bilinear"`, or `"bicubic"`
    /// (`"nohalo"` and `"lbb"` are recognised but not implemented yet).
    /// Panicking form of [`Raster::try_affine`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown or unimplemented interpolator name or any
    /// [`ResampleError`]; see [`Raster::try_affine`].
    #[track_caller]
    pub fn affine(&self, matrix: [f64; 4], interpolate: &str) -> Raster {
        let interpolate = expect_resample("affine", Interpolator::from_name(interpolate));
        expect_resample("affine", self.try_affine(matrix, interpolate))
    }

    /// Fallible form of [`Raster::similarity`] with an explicit
    /// interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_similarity_with(
        &self,
        angle: f64,
        scale: f64,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        let rad = angle * PI / 180.0;
        let a = scale * rad.cos();
        let b = scale * -rad.sin();
        self.try_affine([a, b, -b, a], interpolate)
    }

    /// Fallible form of [`Raster::similarity`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_similarity(&self, angle: f64, scale: f64) -> Result<Raster, ResampleError> {
        self.try_similarity_with(angle, scale, Interpolator::Bilinear)
    }

    /// Rotate by `angle` degrees and scale by `scale` (libvips
    /// `vips_similarity`), expanding the canvas to the rotated bounding
    /// box and interpolating bilinearly, the libvips default. Panicking
    /// form of [`Raster::try_similarity`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_similarity`].
    #[track_caller]
    pub fn similarity(&self, angle: f64, scale: f64) -> Raster {
        expect_resample("similarity", self.try_similarity(angle, scale))
    }

    /// Fallible form of [`Raster::rotate`] with an explicit interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_rotate_with(
        &self,
        angle: f64,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        self.try_similarity_with(angle, 1.0, interpolate)
    }

    /// Fallible form of [`Raster::rotate`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_rotate(&self, angle: f64) -> Result<Raster, ResampleError> {
        self.try_similarity(angle, 1.0)
    }

    /// Rotate by an arbitrary angle in degrees (libvips `vips_rotate`,
    /// `vips_similarity` with scale 1), expanding the canvas to the
    /// rotated bounding box. Panicking form of [`Raster::try_rotate`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_rotate`].
    #[track_caller]
    pub fn rotate(&self, angle: f64) -> Raster {
        expect_resample("rotate", self.try_rotate(angle))
    }

    /// Fallible form of [`Raster::mapim`], taking the typed interpolator.
    ///
    /// # Errors
    ///
    /// [`ResampleError::IndexBands`] unless the index image has exactly
    /// two bands, or [`ResampleError::Raster`] on allocation failure.
    pub fn try_mapim(
        &self,
        index: &Raster,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        let index_bands = index.format().channels();
        if index_bands != 2 {
            return Err(ResampleError::IndexBands { bands: index_bands });
        }

        let format = self.format();
        let layout = SampleLayout::of(format);
        let bands = format.channels();
        let (w, h) = (f64::from(self.width()), f64::from(self.height()));
        let background = 0.0f64;
        let fetch = TapFetch::new(self, Extend::Background, background);

        let index_layout = SampleLayout::of(index.format());
        let index_data = index.data();
        let (ow, oh) = (index.width() as usize, index.height() as usize);

        let mut out = Raster::zeroed(index.width(), index.height(), format)?;
        let buf = out.data_mut();
        let mut px = vec![0.0f64; bands];
        let mut acc = vec![0.0f64; bands];

        for y in 0..oh {
            for x in 0..ow {
                let ii = (y * ow + x) * 2;
                let sx = index_layout.read(index_data, ii);
                let sy = index_layout.read(index_data, ii + 1);
                let oi = (y * ow + x) * bands;
                // Coordinates inside [-1, dim + 1) interpolate with
                // background-extended taps (edge antialiasing); everything
                // else, including NaN, paints the background
                // (`vips_mapim_gen` clip against `Xsize - window_size` on
                // the embedded input).
                if sx >= -1.0 && sx < w + 1.0 && sy >= -1.0 && sy < h + 1.0 {
                    interpolate_at(&fetch, interpolate, sx, sy, false, &mut px, &mut acc);
                    for (bi, v) in acc.iter().enumerate() {
                        layout.write(buf, oi + bi, *v);
                    }
                } else {
                    for bi in 0..bands {
                        layout.write(buf, oi + bi, background);
                    }
                }
            }
        }

        Ok(out)
    }

    /// Remap through a two-band coordinate image (libvips `vips_mapim`):
    /// output pixel `(x, y)` samples this image at the position stored in
    /// the index pixel `(x, y)` (band 0 is the source x, band 1 the source
    /// y). The interpolator is a libvips nickname as in
    /// [`Raster::affine`]. Panicking form of [`Raster::try_mapim`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown or unimplemented interpolator name or any
    /// [`ResampleError`]; see [`Raster::try_mapim`].
    #[track_caller]
    pub fn mapim(&self, index: &Raster, interpolate: &str) -> Raster {
        let interpolate = expect_resample("mapim", Interpolator::from_name(interpolate));
        expect_resample("mapim", self.try_mapim(index, interpolate))
    }

    /// Fallible form of [`Raster::constant_u8`].
    ///
    /// # Errors
    ///
    /// [`ResampleError::Raster`] on zero dimensions or allocation failure.
    pub fn try_constant_u8(width: u32, height: u32, value: u8) -> Result<Raster, ResampleError> {
        let mut out = Raster::zeroed(width, height, PixelFormat::Gray8)?;
        out.data_mut().fill(value);
        Ok(out)
    }

    /// Create a one-band 8-bit image with every sample set to `value`
    /// (libvips `vips_black` plus a constant add). The ported resample
    /// cell uses this to pin constant preservation through
    /// [`Raster::reduce`]; it lives in this module to keep the resample
    /// batch file-disjoint from [`crate::create`]. Panicking form of
    /// [`Raster::try_constant_u8`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_constant_u8`].
    #[track_caller]
    pub fn constant_u8(width: u32, height: u32, value: u8) -> Raster {
        expect_resample("constant_u8", Self::try_constant_u8(width, height, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversion::Angle;

    /// A 4x4 Gray8 ramp with distinct values per pixel.
    fn ramp_4x4() -> Raster {
        let data: Vec<u8> = (0..16u8).map(|v| v * 10).collect();
        Raster::new(4, 4, PixelFormat::Gray8, data).unwrap()
    }

    /// shrink by 2 box-averages each 2x2 block with round-half-up
    /// integer arithmetic.
    #[test]
    fn shrink_2x_averages_blocks() {
        let im = ramp_4x4();
        let out = im.shrink(2.0, 2.0);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
        // Blocks: {0,10,40,50} {20,30,60,70} {80,90,120,130} {100,110,140,150},
        // means with (sum + 2) / 4 rounding.
        assert_eq!(out.data(), &[25, 45, 105, 125]);
    }

    /// Fractional shrink sizes round to nearest and preserve a constant
    /// image exactly through the lanczos3 residual reduce.
    #[test]
    fn shrink_fractional_dims_and_constant() {
        let im = Raster::constant_u8(10, 10, 77);
        let out = im.shrink(2.5, 2.5);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert!(out.data().iter().all(|&v| v == 77));
    }

    /// shrink factors below 1 are a typed error, as in libvips.
    #[test]
    fn shrink_bad_factor_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_shrink(0.5, 2.0),
            Err(ResampleError::BadFactor { op: "shrink", .. })
        ));
    }

    /// Every reduce kernel preserves constant images exactly (the masks
    /// are normalised and the edges extend by replication).
    #[test]
    fn reduce_preserves_constants_for_every_kernel() {
        for &val in &[0u8, 1, 2, 254, 255] {
            let im = Raster::constant_u8(10, 10, val);
            for kernel in [
                "nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3",
            ] {
                let out = im.reduce(2.0, 2.0, kernel);
                assert_eq!(out.width(), 5);
                assert_eq!(out.height(), 5);
                assert!(
                    out.data().iter().all(|&v| v == val),
                    "constant {val} not preserved by reduce with {kernel}"
                );
            }
        }
    }

    /// reduce output dimensions round to nearest, and the average of a
    /// smooth ramp stays close through every kernel.
    #[test]
    fn reduce_dims_and_average() {
        let im = crate::source::generate_test_raster(64, 48).unwrap();
        for kernel in [
            "nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3",
        ] {
            for &fac in &[1.0f64, 1.1, 1.5, 1.999] {
                let out = im.reduce(fac, fac, kernel);
                assert_eq!(i64::from(out.width()), round_uint(64.0 / fac));
                assert_eq!(i64::from(out.height()), round_uint(48.0 / fac));
                let d = (out.avg() - im.avg()).abs();
                assert!(d < 2.0, "reduce({fac}, {kernel}) moved the average by {d}");
            }
        }
    }

    /// An unknown kernel nickname is a typed error, and the recognised
    /// but unimplemented interpolators fail loudly rather than aliasing.
    #[test]
    fn kernel_and_interpolator_parsing() {
        assert!(matches!(
            ReduceKernel::from_name("box"),
            Err(ResampleError::UnknownKernel { .. })
        ));
        assert!(matches!(
            Interpolator::from_name("nohalo"),
            Err(ResampleError::InterpolatorNotImplemented { name: "nohalo" })
        ));
        assert!(matches!(
            Interpolator::from_name("lbb"),
            Err(ResampleError::InterpolatorNotImplemented { name: "lbb" })
        ));
        assert!(matches!(
            Interpolator::from_name("vsqbs"),
            Err(ResampleError::UnknownInterpolator { .. })
        ));
        assert_eq!(
            Interpolator::from_name("bilinear").unwrap(),
            Interpolator::Bilinear
        );
        assert_eq!(
            ReduceKernel::from_name("lanczos3").unwrap(),
            ReduceKernel::Lanczos3
        );
    }

    /// resize dimensions follow round(dim * scale), reproducing the
    /// libvips resize sizing rules the ported cell pins.
    #[test]
    fn resize_dims_round_to_nearest() {
        let im = Raster::black(100, 1);
        let out = im.resize(0.5);
        assert_eq!(out.width(), 50);
        assert_eq!(out.height(), 1);

        let im = Raster::black(1600, 1000);
        let out = im.resize(10.0 / 1600.0);
        assert_eq!(out.width(), 10);
        assert_eq!(out.height(), 6);
    }

    /// resize round-trips dimensions: halving then doubling restores the
    /// original size, and the average of a smooth ramp survives a quarter
    /// resize.
    #[test]
    fn resize_round_trip_and_average() {
        let im = crate::source::generate_test_raster(64, 64).unwrap();
        let half = im.resize(0.5);
        assert_eq!(half.width(), 32);
        assert_eq!(half.height(), 32);
        let back = half.resize(2.0);
        assert_eq!(back.width(), 64);
        assert_eq!(back.height(), 64);

        let quarter = im.resize(0.25);
        assert_eq!(quarter.width(), 16);
        assert_eq!(quarter.height(), 16);
        assert!((quarter.avg() - im.avg()).abs() < 1.0);
    }

    /// A nearest-kernel integral upscale replicates pixels exactly
    /// (the vips_zoom path).
    #[test]
    fn nearest_upsample_duplicates_pixels() {
        let im = Raster::new(2, 2, PixelFormat::Gray8, vec![10, 20, 30, 40]).unwrap();
        let out = im.resize_with(
            2.0,
            ResizeOptions {
                kernel: ReduceKernel::Nearest,
                ..ResizeOptions::default()
            },
        );
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        #[rustfmt::skip]
        let expected = [
            10, 10, 20, 20,
            10, 10, 20, 20,
            30, 30, 40, 40,
            30, 30, 40, 40,
        ];
        assert_eq!(out.data(), &expected);
    }

    /// Bilinear interpolation at the midpoint of four samples is their
    /// mean (pinned through mapim with a half-integer coordinate).
    #[test]
    fn bilinear_midpoint_is_mean_of_four() {
        let im = Raster::new(2, 2, PixelFormat::Gray8, vec![10, 20, 30, 40]).unwrap();
        let mut index = Raster::zeroed(
            1,
            1,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(2).unwrap()),
        )
        .unwrap();
        index.data_mut()[0..4].copy_from_slice(&0.5f32.to_ne_bytes());
        index.data_mut()[4..8].copy_from_slice(&0.5f32.to_ne_bytes());
        let out = im.mapim(&index, "bilinear");
        assert_eq!(out.data(), &[25]);
    }

    /// The identity affine transform is a straight copy.
    #[test]
    fn affine_identity_is_copy() {
        let im = crate::source::generate_test_raster(7, 5).unwrap();
        for interp in ["nearest", "bilinear", "bicubic"] {
            let out = im.affine([1.0, 0.0, 0.0, 1.0], interp);
            assert_eq!(out.width(), im.width());
            assert_eq!(out.height(), im.height());
            assert_eq!(out.data(), im.data());
        }
    }

    /// The transpose matrix [0, 1, 1, 0] samples exactly on the input
    /// grid: every interpolator reproduces rot90 + fliphor byte for byte,
    /// and four applications are the identity (the ported test_affine
    /// invariant).
    #[test]
    fn affine_transpose_matches_rot90_fliphor_and_round_trips() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        for interp in ["nearest", "bilinear", "bicubic"] {
            let t = im.affine([0.0, 1.0, 1.0, 0.0], interp);
            assert_eq!(t.width(), im.height());
            assert_eq!(t.height(), im.width());
            assert_eq!(
                t.data(),
                reference.data(),
                "transpose mismatch for {interp}"
            );

            let mut x = im.clone();
            for _ in 0..4 {
                x = x.affine([0.0, 1.0, 1.0, 0.0], interp);
            }
            assert_eq!(
                x.data(),
                im.data(),
                "4x transpose not identity for {interp}"
            );
        }
    }

    /// rotate(90) is the rot90 permutation displaced one column right by
    /// the bounding-box rounding, with a background seam in column 0: the
    /// faithful libvips affine geometry (`vips_affine_gen` samples
    /// `in(y, h - x)` for the [0, -1, 1, 0] matrix).
    #[test]
    fn rotate_90_is_shifted_rot90() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let rotated = im.rotate(90.0);
        let reference = im.rot(Angle::D90);
        assert_eq!(rotated.width(), reference.width());
        assert_eq!(rotated.height(), reference.height());

        let bands = im.format().channels();
        let (w, h) = (rotated.width() as usize, rotated.height() as usize);
        for y in 0..h {
            for x in 0..w {
                let got = &rotated.data()[(y * w + x) * bands..][..bands];
                if x == 0 {
                    assert!(got.iter().all(|&v| v == 0), "column 0 should be background");
                } else {
                    let want = &reference.data()[(y * w + x - 1) * bands..][..bands];
                    assert_eq!(got, want, "mismatch at ({x}, {y})");
                }
            }
        }
    }

    /// similarity(0, 2) equals affine([2, 0, 0, 2]) exactly, and
    /// similarity(90, 1) equals affine([0, -1, 1, 0]) (the ported
    /// test_similarity bodies).
    #[test]
    fn similarity_matches_affine() {
        let im = crate::source::generate_test_raster(8, 6).unwrap();

        let scaled = im.similarity(0.0, 2.0);
        let affined = im.affine([2.0, 0.0, 0.0, 2.0], "bilinear");
        assert_eq!(scaled.data(), affined.data());

        let rotated = im.similarity(90.0, 1.0);
        let affined = im.affine([0.0, -1.0, 1.0, 0.0], "bilinear");
        assert_eq!(rotated.width(), affined.width());
        assert_eq!(rotated.height(), affined.height());
        let max_diff = rotated
            .data()
            .iter()
            .zip(affined.data().iter())
            .map(|(&p, &q)| (i16::from(p) - i16::from(q)).unsigned_abs())
            .max()
            .unwrap();
        assert!(max_diff < 50, "similarity(90) vs affine: {max_diff}");
    }

    /// An identity coordinate image maps every pixel to itself: bicubic
    /// weights at integer offsets are exactly [0, 1, 0, 0], so mapim
    /// reproduces the input byte for byte.
    #[test]
    fn mapim_identity_is_exact() {
        let im = crate::source::generate_test_raster(9, 7).unwrap();
        let index = Raster::xyz(im.width(), im.height());
        let out = im.mapim(&index, "bicubic");
        assert_eq!(out.data(), im.data());
        assert!((out.avg() - im.avg()).abs() < 0.001);
    }

    /// mapim rejects index images without exactly two bands.
    #[test]
    fn mapim_index_bands_is_typed_error() {
        let im = ramp_4x4();
        let bad = Raster::black(4, 4);
        assert!(matches!(
            im.try_mapim(&bad, Interpolator::Bilinear),
            Err(ResampleError::IndexBands { bands: 1 })
        ));
    }

    /// A singular affine matrix is a typed error.
    #[test]
    fn affine_singular_matrix_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_affine([1.0, 2.0, 2.0, 4.0], Interpolator::Bilinear),
            Err(ResampleError::SingularMatrix)
        ));
    }

    /// Alpha images premultiply through affine: a transpose remains exact
    /// for pixels with non-zero alpha.
    #[test]
    fn affine_transpose_premultiplies_alpha_exactly() {
        #[rustfmt::skip]
        let data = vec![
            200, 10, 30, 255,   40, 80, 120, 128,
            10, 20, 30, 64,     90, 60, 30, 1,
        ];
        let im = Raster::new(2, 2, PixelFormat::Rgba8, data).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        let out = im.affine([0.0, 1.0, 1.0, 0.0], "bilinear");
        assert_eq!(out.data(), reference.data());
    }

    /// constant_u8 fills a one-band image with the value.
    #[test]
    fn constant_u8_fills() {
        let im = Raster::constant_u8(3, 2, 254);
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert_eq!(im.data(), &[254; 6]);
    }

    /// shrinkh / shrinkv shrink one axis only.
    #[test]
    fn shrink_axis_forms() {
        let im = ramp_4x4();
        let h = im.shrinkh(2);
        assert_eq!((h.width(), h.height()), (2, 4));
        let v = im.shrinkv(2);
        assert_eq!((v.width(), v.height()), (4, 2));
        // First row of shrinkh: means of (0,10) and (20,30).
        assert_eq!(&h.data()[0..2], &[5, 25]);
    }

    /// reduceh / reducev reduce one axis only, with round-to-nearest
    /// sizing.
    #[test]
    fn reduce_axis_forms() {
        let im = Raster::constant_u8(10, 10, 33);
        let h = im.reduceh(2.5, "cubic");
        assert_eq!((h.width(), h.height()), (4, 10));
        let v = im.reducev(2.5, "cubic");
        assert_eq!((v.width(), v.height()), (10, 4));
        assert!(h.data().iter().all(|&p| p == 33));
        assert!(v.data().iter().all(|&p| p == 33));
    }

    /// resize is honest about invalid scales.
    #[test]
    fn resize_bad_scale_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_resize(0.0),
            Err(ResampleError::BadScale { .. })
        ));
        assert!(matches!(
            im.try_resize(f64::NAN),
            Err(ResampleError::BadScale { .. })
        ));
    }

    /// 16-bit and float formats resample through the same paths.
    #[test]
    fn shrink_gray16_and_float() {
        let mut data = Vec::new();
        for v in [1000u16, 2000, 3000, 4000] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let im = Raster::new(2, 2, PixelFormat::Gray16, data).unwrap();
        let out = im.shrink(2.0, 2.0);
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        assert_eq!(u16::from_ne_bytes([out.data()[0], out.data()[1]]), 2500);

        let mut fdata = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            fdata.extend_from_slice(&v.to_ne_bytes());
        }
        let fim = Raster::new(
            2,
            2,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            fdata,
        )
        .unwrap();
        let fout = fim.shrink(2.0, 2.0);
        let got = f32::from_ne_bytes([
            fout.data()[0],
            fout.data()[1],
            fout.data()[2],
            fout.data()[3],
        ]);
        assert!((got - 2.5).abs() < 1e-6);
    }
}
