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
//!   samples, the `vips_premultiply` defaults. The averaging resamplers —
//!   `reduce` / `reduceh` / `reducev`, `shrink` / `shrinkh` / `shrinkv`, and
//!   `resize` — do the same: an alpha image is premultiplied once into a float
//!   working buffer, the separable box / kernel / affine passes all run in that
//!   premultiplied space, and the result is unpremultiplied once at the end
//!   (the `vips_resize` bracket). This coverage-weights the colour so the
//!   meaningless RGB of transparent pixels cannot bleed into opaque neighbours
//!   (the dark fringe at transparency boundaries). Note this is a deliberate
//!   divergence from the bare `vips_reduce*` / `vips_shrink*` namesakes, which
//!   do *not* premultiply — only `vips_resize` does — but it is the behaviour
//!   the pyramid pipeline needs by default and matches a premultiplied vips
//!   pipeline (`premultiply | reduce/shrink | unpremultiply`). The single-tap
//!   Nearest kernel is exempt: it does no averaging, so it stays an exact pick
//!   with no premultiply round-trip.
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
//! * **Interpolators.** `nearest`, `bilinear`, `bicubic` (Catmull-Rom,
//!   the libvips `VipsInterpolateBicubic` coefficients), `nohalo`, and
//!   `lbb` are all implemented. `nohalo` and `lbb` are faithful ports of
//!   the libvips `nohalo.cpp` and `lbb.cpp` minmod-subdivision resamplers:
//!   `lbb` is locally bounded bicubic (a nonlinear Catmull-Rom variant
//!   whose reconstruction stays within the range of the 16 nearest input
//!   samples, so it never overshoots), and `nohalo` is level-1 co-monotone
//!   subdivision (minmod slopes) finished with `lbb`. Both centre and
//!   reflect their stencils exactly as the C interpolators do (`lbb` at
//!   window offset 1, `nohalo` at window offset 2 with round-to-nearest
//!   centring), so on samples that land exactly on the input grid they
//!   return the input pixel unchanged, which keeps the 4x rotation
//!   round-trip an identity.
//!
//! # Example usage
//!
//! * [ported_resample tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/ported_resample.rs)

use crate::colour::{ColourError, Intent, Pcs};
use crate::conversion::Interpretation;
use crate::extract::{Extend, ExtractError};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::SourceError;
use std::f64::consts::PI;
use std::path::Path;
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

/// Typed errors for the [`Raster::thumbnail`] family (libvips
/// `vips_thumbnail`): decode, resample, colour, and crop failures folded
/// into one surface so the panicking forms report a single cause.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ThumbnailError {
    /// The target width (or height) is zero; a thumbnail box must be at
    /// least one pixel on each side.
    #[error("thumbnail target size must be at least 1 pixel, got {size}")]
    BadSize { size: u32 },
    /// The output profile name is not a recognised built-in
    /// ([`Raster::thumbnail_with_profile`] currently accepts only
    /// `"srgb"`).
    #[error("unknown output profile {name:?}; expected \"srgb\"")]
    UnknownProfile { name: String },
    /// The built-in sRGB profile could not be encoded.
    #[error("could not build the built-in sRGB profile: {0}")]
    Profile(String),
    /// Decoding the source file or buffer failed.
    #[error(transparent)]
    Decode(#[from] SourceError),
    /// A resampling step (reduce / resize / affine) failed.
    #[error(transparent)]
    Resample(#[from] ResampleError),
    /// A colour step (linear import, ICC import / export) failed.
    #[error(transparent)]
    Colour(#[from] ColourError),
    /// The crop-to-box step failed.
    #[error(transparent)]
    Extract(#[from] ExtractError),
}

#[track_caller]
fn expect_thumbnail(r: Result<Raster, ThumbnailError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e) => panic!("thumbnail: {e}"),
    }
}

/// A point resampler for [`Raster::affine`] and [`Raster::mapim`] (libvips
/// `VipsInterpolate`).
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
    /// Nohalo level-1 co-monotone subdivision finished with LBB, the
    /// libvips `VipsInterpolateNohalo` (`nohalo.cpp`). A halo-reducing,
    /// edge-sharpening resampler that stays within the range of nearby
    /// input samples.
    Nohalo,
    /// Locally bounded bicubic, the libvips `VipsInterpolateLbb`
    /// (`lbb.cpp`). A nonlinear Catmull-Rom variant whose reconstruction
    /// is bounded by the 16 nearest input samples, so it produces no
    /// overshoot.
    Lbb,
}

impl Interpolator {
    /// Parse a libvips interpolator nickname.
    ///
    /// # Errors
    ///
    /// [`ResampleError::UnknownInterpolator`] for any name that is not a
    /// recognised libvips interpolator nickname.
    pub fn from_name(name: &str) -> Result<Self, ResampleError> {
        match name {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            "bicubic" => Ok(Self::Bicubic),
            "nohalo" => Ok(Self::Nohalo),
            "lbb" => Ok(Self::Lbb),
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

    // Alpha coverage-weighting (#288/#348) is bracketed once by the caller in
    // [`with_premultiply`], not here: an alpha image arrives already
    // premultiplied in a float working raster, so `reduce_axis` just convolves
    // its input linearly, whatever it is — straight colour for the no-alpha
    // (and Nearest single-tap) paths, premultiplied float for the bracketed
    // alpha paths. Keeping the premultiply outside means the inter-axis
    // intermediate stays premultiplied and full-precision instead of being
    // un-premultiplied and requantised to straight 8/16-bit alpha between the
    // vertical and horizontal passes (the low-alpha colour-banding of a
    // per-axis integer bracket).

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
        // The integer box pre-shrink consumed the whole factor, or this is a
        // pure passthrough: no residual convolution to run.
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
    // Accumulate every band of a destination pixel before writing.
    let mut px = vec![0.0f64; bands];
    for oy in 0..oh {
        for ox in 0..ow {
            let (start, c) = match axis {
                Axis::Horizontal => &masks[ox],
                Axis::Vertical => &masks[oy],
            };
            for (band, p) in px.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (k, ck) in c.iter().enumerate() {
                    let tap = clamp_dim(start + k as i64);
                    let (sx, sy) = match axis {
                        Axis::Horizontal => (tap, oy),
                        Axis::Vertical => (ox, tap),
                    };
                    acc += ck * layout.read(data, (sy * w + sx) * bands + band);
                }
                *p = acc;
            }
            let obase = (oy * ow + ox) * bands;
            for (band, &p) in px.iter().enumerate() {
                layout.write(&mut out, obase + band, p);
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
        Interpolator::Lbb => {
            // LBB samples the 4x4 block at (x0-1..x0+2, y0-1..y0+2), the
            // patch corner at (x0, y0), relative offset in [0, 1]; the
            // same stencil geometry as bicubic (window_offset 1).
            let k = LbbCoeffs::new(x - x0 as f64, y - y0 as f64);
            let offsets = stencil_offsets_4x4();
            let cols = gather_stencil(fetch, x0, y0, &offsets, premultiply, px);
            for (b, o) in out.iter_mut().enumerate() {
                let mut s = [0.0f64; 16];
                for (i, si) in s.iter_mut().enumerate() {
                    *si = cols[i * fetch.bands + b];
                }
                *o = lbbicubic(&k, &s);
            }
        }
        Interpolator::Nohalo => {
            // Nohalo centres on the nearest pixel (window_offset 2, round
            // to nearest), reflects the diamond stencil so the sample sits
            // to the bottom-right of the centre, subdivides to a 4x4 LBB
            // stencil, then finishes with LBB at the reflected offset.
            let ix = (x + 0.5).floor() as i64;
            let iy = (y + 0.5).floor() as i64;
            let rel_x = x - ix as f64;
            let rel_y = y - iy as f64;
            let sx: i64 = if rel_x >= 0.0 { 1 } else { -1 };
            let sy: i64 = if rel_y >= 0.0 { 1 } else { -1 };
            // xp1over2 = 2 * |relative| in [0, 1] after the reflection.
            let k = LbbCoeffs::new(2.0 * rel_x.abs(), 2.0 * rel_y.abs());
            let offsets = nohalo_offsets(sx, sy);
            let cols = gather_stencil(fetch, ix, iy, &offsets, premultiply, px);
            for (b, o) in out.iter_mut().enumerate() {
                let mut diamond = [0.0f64; 21];
                for (i, di) in diamond.iter_mut().enumerate() {
                    *di = cols[i * fetch.bands + b];
                }
                let st = NohaloStencil::from_diamond(&diamond);
                let lbb_stencil = nohalo_subdivision(&st);
                *o = lbbicubic(&k, &lbb_stencil);
            }
        }
    }
}

/// Gather a nonlinear-interpolator stencil into a flat `taps * bands`
/// buffer (tap-major), fetching each `(dx, dy)` offset from `(cx, cy)`
/// through the [`Extend`] rule and premultiplying when asked. `px` is the
/// per-tap scratch. Allocates one small buffer per output pixel, which the
/// nonlinear resamplers need because they cannot be expressed as a fixed
/// weighted sum of taps.
fn gather_stencil(
    fetch: &TapFetch<'_>,
    cx: i64,
    cy: i64,
    offsets: &[(i64, i64)],
    premultiply: bool,
    px: &mut [f64],
) -> Vec<f64> {
    let bands = fetch.bands;
    let mut cols = vec![0.0f64; offsets.len() * bands];
    for (idx, &(dx, dy)) in offsets.iter().enumerate() {
        fetch.fetch(cx + dx, cy + dy, premultiply, px);
        cols[idx * bands..idx * bands + bands].copy_from_slice(px);
    }
    cols
}

/// The 16 `(dx, dy)` offsets of the LBB 4x4 stencil relative to the patch
/// corner `(x0, y0)`, in row-major uno/dos/tre/qua order.
fn stencil_offsets_4x4() -> [(i64, i64); 16] {
    let mut out = [(0i64, 0i64); 16];
    let mut idx = 0;
    let mut dy = -1;
    while dy <= 2 {
        let mut dx = -1;
        while dx <= 2 {
            out[idx] = (dx, dy);
            idx += 1;
            dx += 1;
        }
        dy += 1;
    }
    out
}

/// The 21 `(dx, dy)` offsets of the nohalo diamond stencil relative to the
/// centre pixel, reflected by the sample-position signs `(sx, sy)`, in the
/// order [`NohaloStencil::from_diamond`] expects.
fn nohalo_offsets(sx: i64, sy: i64) -> [(i64, i64); 21] {
    [
        (-sx, -2 * sy), // uno_two
        (0, -2 * sy),   // uno_thr
        (sx, -2 * sy),  // uno_fou
        (-2 * sx, -sy), // dos_one
        (-sx, -sy),     // dos_two
        (0, -sy),       // dos_thr
        (sx, -sy),      // dos_fou
        (2 * sx, -sy),  // dos_fiv
        (-2 * sx, 0),   // tre_one
        (-sx, 0),       // tre_two
        (0, 0),         // tre_thr
        (sx, 0),        // tre_fou
        (2 * sx, 0),    // tre_fiv
        (-2 * sx, sy),  // qua_one
        (-sx, sy),      // qua_two
        (0, sy),        // qua_thr
        (sx, sy),       // qua_fou
        (2 * sx, sy),   // qua_fiv
        (-sx, 2 * sy),  // cin_two
        (0, 2 * sy),    // cin_thr
        (sx, 2 * sy),   // cin_fou
    ]
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

// ---------------------------------------------------------------------------
// Nohalo / LBB (minmod-subdivision resamplers)
// ---------------------------------------------------------------------------
//
// A faithful port of the libvips `nohalo.cpp` and `lbb.cpp` (v8.18) nonlinear
// resamplers by N. Robidoux, C. Racette and J. Cupitt. Nohalo is level-1
// co-monotone subdivision (minmod slopes) producing a 4x4 stencil that feeds
// LBB; LBB is locally bounded bicubic, a nonlinear Hermite variant of
// Catmull-Rom whose reconstruction stays within the range of the 16 nearest
// input samples, so no output clamping is needed to avoid overshoot.

/// The sixteen LBB Hermite coefficients for a sample offset, the
/// `c00 .. c11dxdy` block shared verbatim by `nohalo.cpp` and `lbb.cpp`.
/// `xp1over2` and `yp1over2` are both in `[0, 1]`: for LBB they are the
/// relative offsets directly, for nohalo they are `2 * |relative|` after
/// the stencil reflection.
#[derive(Clone, Copy)]
struct LbbCoeffs {
    c00: f64,
    c10: f64,
    c01: f64,
    c11: f64,
    c00dx: f64,
    c10dx: f64,
    c01dx: f64,
    c11dx: f64,
    c00dy: f64,
    c10dy: f64,
    c01dy: f64,
    c11dy: f64,
    c00dxdy: f64,
    c10dxdy: f64,
    c01dxdy: f64,
    c11dxdy: f64,
}

impl LbbCoeffs {
    fn new(xp1over2: f64, yp1over2: f64) -> Self {
        let xm1over2 = xp1over2 - 1.0;
        let onepx = 0.5 + xp1over2;
        let onemx = 1.5 - xp1over2;
        let xp1over2sq = xp1over2 * xp1over2;

        let ym1over2 = yp1over2 - 1.0;
        let onepy = 0.5 + yp1over2;
        let onemy = 1.5 - yp1over2;
        let yp1over2sq = yp1over2 * yp1over2;

        let xm1over2sq = xm1over2 * xm1over2;
        let ym1over2sq = ym1over2 * ym1over2;

        let twice1px = onepx + onepx;
        let twice1py = onepy + onepy;
        let twice1mx = onemx + onemx;
        let twice1my = onemy + onemy;

        let xm1over2sq_times_ym1over2sq = xm1over2sq * ym1over2sq;
        let xp1over2sq_times_ym1over2sq = xp1over2sq * ym1over2sq;
        let xp1over2sq_times_yp1over2sq = xp1over2sq * yp1over2sq;
        let xm1over2sq_times_yp1over2sq = xm1over2sq * yp1over2sq;

        let four_times_1px_times_1py = twice1px * twice1py;
        let four_times_1mx_times_1py = twice1mx * twice1py;
        let twice_xp1over2_times_1py = xp1over2 * twice1py;
        let twice_xm1over2_times_1py = xm1over2 * twice1py;

        let twice_xm1over2_times_1my = xm1over2 * twice1my;
        let twice_xp1over2_times_1my = xp1over2 * twice1my;
        let four_times_1mx_times_1my = twice1mx * twice1my;
        let four_times_1px_times_1my = twice1px * twice1my;

        let twice_1px_times_ym1over2 = twice1px * ym1over2;
        let twice_1mx_times_ym1over2 = twice1mx * ym1over2;
        let xp1over2_times_ym1over2 = xp1over2 * ym1over2;
        let xm1over2_times_ym1over2 = xm1over2 * ym1over2;

        let xm1over2_times_yp1over2 = xm1over2 * yp1over2;
        let xp1over2_times_yp1over2 = xp1over2 * yp1over2;
        let twice_1mx_times_yp1over2 = twice1mx * yp1over2;
        let twice_1px_times_yp1over2 = twice1px * yp1over2;

        Self {
            c00: four_times_1px_times_1py * xm1over2sq_times_ym1over2sq,
            c00dx: twice_xp1over2_times_1py * xm1over2sq_times_ym1over2sq,
            c00dy: twice_1px_times_yp1over2 * xm1over2sq_times_ym1over2sq,
            c00dxdy: xp1over2_times_yp1over2 * xm1over2sq_times_ym1over2sq,

            c10: four_times_1mx_times_1py * xp1over2sq_times_ym1over2sq,
            c10dx: twice_xm1over2_times_1py * xp1over2sq_times_ym1over2sq,
            c10dy: twice_1mx_times_yp1over2 * xp1over2sq_times_ym1over2sq,
            c10dxdy: xm1over2_times_yp1over2 * xp1over2sq_times_ym1over2sq,

            c01: four_times_1px_times_1my * xm1over2sq_times_yp1over2sq,
            c01dx: twice_xp1over2_times_1my * xm1over2sq_times_yp1over2sq,
            c01dy: twice_1px_times_ym1over2 * xm1over2sq_times_yp1over2sq,
            c01dxdy: xp1over2_times_ym1over2 * xm1over2sq_times_yp1over2sq,

            c11: four_times_1mx_times_1my * xp1over2sq_times_yp1over2sq,
            c11dx: twice_xm1over2_times_1my * xp1over2sq_times_yp1over2sq,
            c11dy: twice_1mx_times_ym1over2 * xp1over2sq_times_yp1over2sq,
            c11dxdy: xm1over2_times_ym1over2 * xp1over2sq_times_yp1over2sq,
        }
    }
}

/// Minmod: the smaller (in absolute value) of two slopes when they share a
/// sign, else zero (`NOHALO_MINMOD`). `aa` is `a * a`, `ab` is `a * b`.
#[inline]
fn nohalo_minmod(a: f64, b: f64, aa: f64, ab: f64) -> f64 {
    if ab >= 0.0 {
        if aa <= ab { a } else { b }
    } else {
        0.0
    }
}

/// Locally bounded bicubic over a 4x4 stencil (`lbbicubic`, the "soft"
/// 3x3-block limiter version, the libvips default). `s` holds the sixteen
/// stencil values in row-major order (uno/dos/tre/qua rows, one/two/thr/fou
/// columns).
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn lbbicubic(k: &LbbCoeffs, s: &[f64; 16]) -> f64 {
    let (uno_one, uno_two, uno_thr, uno_fou) = (s[0], s[1], s[2], s[3]);
    let (dos_one, dos_two, dos_thr, dos_fou) = (s[4], s[5], s[6], s[7]);
    let (tre_one, tre_two, tre_thr, tre_fou) = (s[8], s[9], s[10], s[11]);
    let (qua_one, qua_two, qua_thr, qua_fou) = (s[12], s[13], s[14], s[15]);

    // Four min and four max over 3x3 sub-blocks of the 4x4 stencil.
    let m1 = dos_two.min(dos_thr);
    let big_m1 = dos_two.max(dos_thr);
    let m2 = tre_two.min(tre_thr);
    let big_m2 = tre_two.max(tre_thr);
    let m6 = dos_one.min(tre_one);
    let big_m6 = dos_one.max(tre_one);
    let m7 = dos_fou.min(tre_fou);
    let big_m7 = dos_fou.max(tre_fou);
    let m3 = uno_two.min(uno_thr);
    let big_m3 = uno_two.max(uno_thr);
    let m4 = qua_two.min(qua_thr);
    let big_m4 = qua_two.max(qua_thr);
    let m5 = m1.min(m2);
    let big_m5 = big_m1.max(big_m2);
    let m10 = m6.min(uno_one);
    let big_m10 = big_m6.max(uno_one);
    let m11 = m6.min(qua_one);
    let big_m11 = big_m6.max(qua_one);
    let m12 = m7.min(uno_fou);
    let big_m12 = big_m7.max(uno_fou);
    let m13 = m7.min(qua_fou);
    let big_m13 = big_m7.max(qua_fou);
    let m8 = m5.min(m3);
    let big_m8 = big_m5.max(big_m3);
    let m9 = m5.min(m4);
    let big_m9 = big_m5.max(big_m4);
    let min00 = m8.min(m10);
    let max00 = big_m8.max(big_m10);
    let min10 = m8.min(m12);
    let max10 = big_m8.max(big_m12);
    let min01 = m9.min(m11);
    let max01 = big_m9.max(big_m11);
    let min11 = m9.min(m13);
    let max11 = big_m9.max(big_m13);

    // Distances to the local min and max.
    let u00 = dos_two - min00;
    let v00 = max00 - dos_two;
    let u10 = dos_thr - min10;
    let v10 = max10 - dos_thr;
    let u01 = tre_two - min01;
    let v01 = max01 - tre_two;
    let u11 = tre_thr - min11;
    let v11 = max11 - tre_thr;

    // Centred-difference first derivatives (factors of 1/2 folded in later).
    let dble_dzdx00i = dos_thr - dos_one;
    let dble_dzdy11i = qua_thr - dos_thr;
    let dble_dzdx10i = dos_fou - dos_two;
    let dble_dzdy01i = qua_two - dos_two;
    let dble_dzdx01i = tre_thr - tre_one;
    let dble_dzdy10i = tre_thr - uno_thr;
    let dble_dzdx11i = tre_fou - tre_two;
    let dble_dzdy00i = tre_two - uno_two;

    let sign_dzdx00 = if dble_dzdx00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx10 = if dble_dzdx10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx01 = if dble_dzdx01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx11 = if dble_dzdx11i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy00 = if dble_dzdy00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy10 = if dble_dzdy10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy01 = if dble_dzdy01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy11 = if dble_dzdy11i >= 0.0 { 1.0 } else { -1.0 };

    // Centred-difference cross derivatives (factors of 1/4 folded in later).
    let quad_d2zdxdy00i = uno_one - uno_thr + dble_dzdx01i;
    let quad_d2zdxdy10i = uno_two - uno_fou + dble_dzdx11i;
    let quad_d2zdxdy01i = qua_thr - qua_one - dble_dzdx00i;
    let quad_d2zdxdy11i = qua_fou - qua_two - dble_dzdx10i;

    // Slope limiters (key multiplier 3, folded with a factor of 2).
    let dble_slopelimit_00 = 6.0 * u00.min(v00);
    let dble_slopelimit_10 = 6.0 * u10.min(v10);
    let dble_slopelimit_01 = 6.0 * u01.min(v01);
    let dble_slopelimit_11 = 6.0 * u11.min(v11);

    let clamp_slope = |sign: f64, deriv: f64, limit: f64| -> f64 {
        if sign * deriv <= limit {
            deriv
        } else {
            sign * limit
        }
    };
    let dble_dzdx00 = clamp_slope(sign_dzdx00, dble_dzdx00i, dble_slopelimit_00);
    let dble_dzdy00 = clamp_slope(sign_dzdy00, dble_dzdy00i, dble_slopelimit_00);
    let dble_dzdx10 = clamp_slope(sign_dzdx10, dble_dzdx10i, dble_slopelimit_10);
    let dble_dzdy10 = clamp_slope(sign_dzdy10, dble_dzdy10i, dble_slopelimit_10);
    let dble_dzdx01 = clamp_slope(sign_dzdx01, dble_dzdx01i, dble_slopelimit_01);
    let dble_dzdy01 = clamp_slope(sign_dzdy01, dble_dzdy01i, dble_slopelimit_01);
    let dble_dzdx11 = clamp_slope(sign_dzdx11, dble_dzdx11i, dble_slopelimit_11);
    let dble_dzdy11 = clamp_slope(sign_dzdy11, dble_dzdy11i, dble_slopelimit_11);

    // Sums and differences of first derivatives.
    let twelve_sum00 = 6.0 * (dble_dzdx00 + dble_dzdy00);
    let twelve_dif00 = 6.0 * (dble_dzdx00 - dble_dzdy00);
    let twelve_sum10 = 6.0 * (dble_dzdx10 + dble_dzdy10);
    let twelve_dif10 = 6.0 * (dble_dzdx10 - dble_dzdy10);
    let twelve_sum01 = 6.0 * (dble_dzdx01 + dble_dzdy01);
    let twelve_dif01 = 6.0 * (dble_dzdx01 - dble_dzdy01);
    let twelve_sum11 = 6.0 * (dble_dzdx11 + dble_dzdy11);
    let twelve_dif11 = 6.0 * (dble_dzdx11 - dble_dzdy11);

    let twelve_abs_sum00 = twelve_sum00.abs();
    let twelve_abs_sum10 = twelve_sum10.abs();
    let twelve_abs_sum01 = twelve_sum01.abs();
    let twelve_abs_sum11 = twelve_sum11.abs();

    let u00_times_36 = 36.0 * u00;
    let u10_times_36 = 36.0 * u10;
    let u01_times_36 = 36.0 * u01;
    let u11_times_36 = 36.0 * u11;

    let first_limit00 = twelve_abs_sum00 - u00_times_36;
    let first_limit10 = twelve_abs_sum10 - u10_times_36;
    let first_limit01 = twelve_abs_sum01 - u01_times_36;
    let first_limit11 = twelve_abs_sum11 - u11_times_36;

    let quad_d2zdxdy00ii = quad_d2zdxdy00i.max(first_limit00);
    let quad_d2zdxdy10ii = quad_d2zdxdy10i.max(first_limit10);
    let quad_d2zdxdy01ii = quad_d2zdxdy01i.max(first_limit01);
    let quad_d2zdxdy11ii = quad_d2zdxdy11i.max(first_limit11);

    let v00_times_36 = 36.0 * v00;
    let v10_times_36 = 36.0 * v10;
    let v01_times_36 = 36.0 * v01;
    let v11_times_36 = 36.0 * v11;

    let second_limit00 = v00_times_36 - twelve_abs_sum00;
    let second_limit10 = v10_times_36 - twelve_abs_sum10;
    let second_limit01 = v01_times_36 - twelve_abs_sum01;
    let second_limit11 = v11_times_36 - twelve_abs_sum11;

    let quad_d2zdxdy00iii = quad_d2zdxdy00ii.min(second_limit00);
    let quad_d2zdxdy10iii = quad_d2zdxdy10ii.min(second_limit10);
    let quad_d2zdxdy01iii = quad_d2zdxdy01ii.min(second_limit01);
    let quad_d2zdxdy11iii = quad_d2zdxdy11ii.min(second_limit11);

    let twelve_abs_dif00 = twelve_dif00.abs();
    let twelve_abs_dif10 = twelve_dif10.abs();
    let twelve_abs_dif01 = twelve_dif01.abs();
    let twelve_abs_dif11 = twelve_dif11.abs();

    let third_limit00 = twelve_abs_dif00 - v00_times_36;
    let third_limit10 = twelve_abs_dif10 - v10_times_36;
    let third_limit01 = twelve_abs_dif01 - v01_times_36;
    let third_limit11 = twelve_abs_dif11 - v11_times_36;

    let quad_d2zdxdy00iiii = quad_d2zdxdy00iii.max(third_limit00);
    let quad_d2zdxdy10iiii = quad_d2zdxdy10iii.max(third_limit10);
    let quad_d2zdxdy01iiii = quad_d2zdxdy01iii.max(third_limit01);
    let quad_d2zdxdy11iiii = quad_d2zdxdy11iii.max(third_limit11);

    let fourth_limit00 = u00_times_36 - twelve_abs_dif00;
    let fourth_limit10 = u10_times_36 - twelve_abs_dif10;
    let fourth_limit01 = u01_times_36 - twelve_abs_dif01;
    let fourth_limit11 = u11_times_36 - twelve_abs_dif11;

    let quad_d2zdxdy00 = quad_d2zdxdy00iiii.min(fourth_limit00);
    let quad_d2zdxdy10 = quad_d2zdxdy10iiii.min(fourth_limit10);
    let quad_d2zdxdy01 = quad_d2zdxdy01iiii.min(fourth_limit01);
    let quad_d2zdxdy11 = quad_d2zdxdy11iiii.min(fourth_limit11);

    let newval1 = k.c00 * dos_two + k.c10 * dos_thr + k.c01 * tre_two + k.c11 * tre_thr;
    let newval2 = k.c00dx * dble_dzdx00
        + k.c10dx * dble_dzdx10
        + k.c01dx * dble_dzdx01
        + k.c11dx * dble_dzdx11
        + k.c00dy * dble_dzdy00
        + k.c10dy * dble_dzdy10
        + k.c01dy * dble_dzdy01
        + k.c11dy * dble_dzdy11;
    let newval3 = k.c00dxdy * quad_d2zdxdy00
        + k.c10dxdy * quad_d2zdxdy10
        + k.c01dxdy * quad_d2zdxdy01
        + k.c11dxdy * quad_d2zdxdy11;

    // `dble_dzdy11i` participates only in the reference implementation's
    // symmetry; it is unused in the final combination, kept above for a
    // line-by-line correspondence with the C source.
    let _ = dble_dzdy11i;

    newval1 + 0.5 * newval2 + 0.25 * newval3
}

/// Nohalo level-1 subdivision (`nohalo_subdivision`): from the 21-point
/// diamond stencil, compute the twelve new half-density values and return
/// the sixteen LBB stencil values in row-major order. `st` holds the input
/// stencil already reflected so the sample sits to the bottom-right of the
/// centre (`tre_thr`); see [`gather_nohalo_stencil`].
#[allow(clippy::many_single_char_names)]
fn nohalo_subdivision(st: &NohaloStencil) -> [f64; 16] {
    let NohaloStencil {
        uno_two,
        uno_thr,
        uno_fou,
        dos_one,
        dos_two,
        dos_thr,
        dos_fou,
        dos_fiv,
        tre_one,
        tre_two,
        tre_thr,
        tre_fou,
        tre_fiv,
        qua_one,
        qua_two,
        qua_thr,
        qua_fou,
        qua_fiv,
        cin_two,
        cin_thr,
        cin_fou,
    } = *st;

    // Vertical simple differences.
    let d_unodos_two = dos_two - uno_two;
    let d_dostre_two = tre_two - dos_two;
    let d_trequa_two = qua_two - tre_two;
    let d_quacin_two = cin_two - qua_two;
    let d_unodos_thr = dos_thr - uno_thr;
    let d_dostre_thr = tre_thr - dos_thr;
    let d_trequa_thr = qua_thr - tre_thr;
    let d_quacin_thr = cin_thr - qua_thr;
    let d_unodos_fou = dos_fou - uno_fou;
    let d_dostre_fou = tre_fou - dos_fou;
    let d_trequa_fou = qua_fou - tre_fou;
    let d_quacin_fou = cin_fou - qua_fou;
    // Horizontal simple differences.
    let d_dos_onetwo = dos_two - dos_one;
    let d_dos_twothr = dos_thr - dos_two;
    let d_dos_thrfou = dos_fou - dos_thr;
    let d_dos_foufiv = dos_fiv - dos_fou;
    let d_tre_onetwo = tre_two - tre_one;
    let d_tre_twothr = tre_thr - tre_two;
    let d_tre_thrfou = tre_fou - tre_thr;
    let d_tre_foufiv = tre_fiv - tre_fou;
    let d_qua_onetwo = qua_two - qua_one;
    let d_qua_twothr = qua_thr - qua_two;
    let d_qua_thrfou = qua_fou - qua_thr;
    let d_qua_foufiv = qua_fiv - qua_fou;

    // Recyclable vertical products and squares.
    let d_unodos_times_dostre_two = d_unodos_two * d_dostre_two;
    let d_dostre_two_sq = d_dostre_two * d_dostre_two;
    let d_dostre_times_trequa_two = d_dostre_two * d_trequa_two;
    let d_trequa_times_quacin_two = d_quacin_two * d_trequa_two;
    let d_quacin_two_sq = d_quacin_two * d_quacin_two;

    let d_unodos_times_dostre_thr = d_unodos_thr * d_dostre_thr;
    let d_dostre_thr_sq = d_dostre_thr * d_dostre_thr;
    let d_dostre_times_trequa_thr = d_trequa_thr * d_dostre_thr;
    let d_trequa_times_quacin_thr = d_trequa_thr * d_quacin_thr;
    let d_quacin_thr_sq = d_quacin_thr * d_quacin_thr;

    let d_unodos_times_dostre_fou = d_unodos_fou * d_dostre_fou;
    let d_dostre_fou_sq = d_dostre_fou * d_dostre_fou;
    let d_dostre_times_trequa_fou = d_trequa_fou * d_dostre_fou;
    let d_trequa_times_quacin_fou = d_trequa_fou * d_quacin_fou;
    let d_quacin_fou_sq = d_quacin_fou * d_quacin_fou;
    // Recyclable horizontal products and squares.
    let d_dos_onetwo_times_twothr = d_dos_onetwo * d_dos_twothr;
    let d_dos_twothr_sq = d_dos_twothr * d_dos_twothr;
    let d_dos_twothr_times_thrfou = d_dos_twothr * d_dos_thrfou;
    let d_dos_thrfou_times_foufiv = d_dos_thrfou * d_dos_foufiv;
    let d_dos_foufiv_sq = d_dos_foufiv * d_dos_foufiv;

    let d_tre_onetwo_times_twothr = d_tre_onetwo * d_tre_twothr;
    let d_tre_twothr_sq = d_tre_twothr * d_tre_twothr;
    let d_tre_twothr_times_thrfou = d_tre_thrfou * d_tre_twothr;
    let d_tre_thrfou_times_foufiv = d_tre_thrfou * d_tre_foufiv;
    let d_tre_foufiv_sq = d_tre_foufiv * d_tre_foufiv;

    let d_qua_onetwo_times_twothr = d_qua_onetwo * d_qua_twothr;
    let d_qua_twothr_sq = d_qua_twothr * d_qua_twothr;
    let d_qua_twothr_times_thrfou = d_qua_thrfou * d_qua_twothr;
    let d_qua_thrfou_times_foufiv = d_qua_thrfou * d_qua_foufiv;
    let d_qua_foufiv_sq = d_qua_foufiv * d_qua_foufiv;

    // Minmod slopes and first-level pixel values.
    let dos_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_unodos_thr,
        d_dostre_thr_sq,
        d_unodos_times_dostre_thr,
    );
    let tre_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_trequa_thr,
        d_dostre_thr_sq,
        d_dostre_times_trequa_thr,
    );
    let newval_uno_two = 0.5 * (dos_thr + tre_thr) + 0.25 * (dos_thr_y - tre_thr_y);

    let qua_thr_y = nohalo_minmod(
        d_quacin_thr,
        d_trequa_thr,
        d_quacin_thr_sq,
        d_trequa_times_quacin_thr,
    );
    let newval_tre_two = 0.5 * (tre_thr + qua_thr) + 0.25 * (tre_thr_y - qua_thr_y);

    let tre_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_trequa_fou,
        d_dostre_fou_sq,
        d_dostre_times_trequa_fou,
    );
    let qua_fou_y = nohalo_minmod(
        d_quacin_fou,
        d_trequa_fou,
        d_quacin_fou_sq,
        d_trequa_times_quacin_fou,
    );
    let newval_tre_fou = 0.5 * (tre_fou + qua_fou) + 0.25 * (tre_fou_y - qua_fou_y);

    let dos_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_unodos_fou,
        d_dostre_fou_sq,
        d_unodos_times_dostre_fou,
    );
    let newval_uno_fou = 0.5 * (dos_fou + tre_fou) + 0.25 * (dos_fou_y - tre_fou_y);

    let tre_two_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_onetwo,
        d_tre_twothr_sq,
        d_tre_onetwo_times_twothr,
    );
    let tre_thr_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_thrfou,
        d_tre_twothr_sq,
        d_tre_twothr_times_thrfou,
    );
    let newval_dos_one = 0.5 * (tre_two + tre_thr) + 0.25 * (tre_two_x - tre_thr_x);

    let tre_fou_x = nohalo_minmod(
        d_tre_foufiv,
        d_tre_thrfou,
        d_tre_foufiv_sq,
        d_tre_thrfou_times_foufiv,
    );
    let tre_thr_x_minus_tre_fou_x = tre_thr_x - tre_fou_x;
    let newval_dos_thr = 0.5 * (tre_thr + tre_fou) + 0.25 * tre_thr_x_minus_tre_fou_x;

    let qua_thr_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_thrfou,
        d_qua_twothr_sq,
        d_qua_twothr_times_thrfou,
    );
    let qua_fou_x = nohalo_minmod(
        d_qua_foufiv,
        d_qua_thrfou,
        d_qua_foufiv_sq,
        d_qua_thrfou_times_foufiv,
    );
    let qua_thr_x_minus_qua_fou_x = qua_thr_x - qua_fou_x;
    let newval_qua_thr = 0.5 * (qua_thr + qua_fou) + 0.25 * qua_thr_x_minus_qua_fou_x;

    let qua_two_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_onetwo,
        d_qua_twothr_sq,
        d_qua_onetwo_times_twothr,
    );
    let newval_qua_one = 0.5 * (qua_two + qua_thr) + 0.25 * (qua_two_x - qua_thr_x);

    let newval_tre_thr = 0.125 * (tre_thr_x_minus_tre_fou_x + qua_thr_x_minus_qua_fou_x)
        + 0.5 * (newval_tre_two + newval_tre_fou);

    let dos_thr_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_thrfou,
        d_dos_twothr_sq,
        d_dos_twothr_times_thrfou,
    );
    let dos_fou_x = nohalo_minmod(
        d_dos_foufiv,
        d_dos_thrfou,
        d_dos_foufiv_sq,
        d_dos_thrfou_times_foufiv,
    );
    let newval_uno_thr = 0.25 * (dos_fou - tre_thr)
        + 0.125 * (dos_fou_y - tre_fou_y + dos_thr_x - dos_fou_x)
        + 0.5 * (newval_uno_two + newval_dos_thr);

    let tre_two_y = nohalo_minmod(
        d_dostre_two,
        d_trequa_two,
        d_dostre_two_sq,
        d_dostre_times_trequa_two,
    );
    let qua_two_y = nohalo_minmod(
        d_quacin_two,
        d_trequa_two,
        d_quacin_two_sq,
        d_trequa_times_quacin_two,
    );
    let newval_tre_one = 0.25 * (qua_two - tre_thr)
        + 0.125 * (qua_two_x - qua_thr_x + tre_two_y - qua_two_y)
        + 0.5 * (newval_dos_one + newval_tre_two);

    let dos_two_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_onetwo,
        d_dos_twothr_sq,
        d_dos_onetwo_times_twothr,
    );
    let dos_two_y = nohalo_minmod(
        d_dostre_two,
        d_unodos_two,
        d_dostre_two_sq,
        d_unodos_times_dostre_two,
    );
    let newval_uno_one = 0.25 * (dos_two + dos_thr + tre_two + tre_thr)
        + 0.125
            * (dos_two_x - dos_thr_x + tre_two_x - tre_thr_x + dos_two_y + dos_thr_y
                - tre_two_y
                - tre_thr_y);

    [
        newval_uno_one,
        newval_uno_two,
        newval_uno_thr,
        newval_uno_fou,
        newval_dos_one,
        tre_thr,
        newval_dos_thr,
        tre_fou,
        newval_tre_one,
        newval_tre_two,
        newval_tre_thr,
        newval_tre_fou,
        newval_qua_one,
        qua_thr,
        newval_qua_thr,
        qua_fou,
    ]
}

/// The 21-point nohalo input stencil, already reflected so the sampling
/// point lies to the bottom-right of `tre_thr`.
#[derive(Clone, Copy)]
struct NohaloStencil {
    uno_two: f64,
    uno_thr: f64,
    uno_fou: f64,
    dos_one: f64,
    dos_two: f64,
    dos_thr: f64,
    dos_fou: f64,
    dos_fiv: f64,
    tre_one: f64,
    tre_two: f64,
    tre_thr: f64,
    tre_fou: f64,
    tre_fiv: f64,
    qua_one: f64,
    qua_two: f64,
    qua_thr: f64,
    qua_fou: f64,
    qua_fiv: f64,
    cin_two: f64,
    cin_thr: f64,
    cin_fou: f64,
}

impl NohaloStencil {
    /// Build the stencil from the 21 diamond taps in [`nohalo_offsets`]
    /// order.
    fn from_diamond(d: &[f64; 21]) -> Self {
        Self {
            uno_two: d[0],
            uno_thr: d[1],
            uno_fou: d[2],
            dos_one: d[3],
            dos_two: d[4],
            dos_thr: d[5],
            dos_fou: d[6],
            dos_fiv: d[7],
            tre_one: d[8],
            tre_two: d[9],
            tre_thr: d[10],
            tre_fou: d[11],
            tre_fiv: d[12],
            qua_one: d[13],
            qua_two: d[14],
            qua_thr: d[15],
            qua_fou: d[16],
            qua_fiv: d[17],
            cin_two: d[18],
            cin_thr: d[19],
            cin_fou: d[20],
        }
    }
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

/// Premultiply the colour bands of an alpha raster by `alpha / max` into a
/// four-band **float** working raster (`RgbaF32`); the alpha band is copied
/// unchanged. `max` is the source sample ceiling.
///
/// Premultiplying into float — the way `vips_resize` premultiplies once into a
/// float buffer — is what lets [`with_premultiply`] bracket the whole separable
/// pipeline (see there): colour is averaged weighted by coverage, the same
/// thing the affine path does per tap via [`TapFetch::fetch`], so the
/// meaningless RGB of transparent pixels cannot bleed into opaque neighbours
/// (dark fringes at transparency boundaries). Unlike a same-bit-depth integer
/// intermediate, the float buffer does not requantise `round(c * a / max)` to a
/// couple of bits for near-transparent pixels — quantisation that
/// un-premultiply would then amplify by `max / a` into visible colour banding.
fn premultiply_to_float(src: &Raster, max: f64) -> Result<Raster, ResampleError> {
    let in_layout = SampleLayout::of(src.format());
    let out_fmt = PixelFormat::RgbaF32;
    let out_layout = SampleLayout::of(out_fmt);
    let bands = src.format().channels();
    let data = src.data();
    let count = src.width() as usize * src.height() as usize;
    let mut out = vec![0u8; count * out_fmt.bytes_per_pixel()];
    for p in 0..count {
        let base = p * bands;
        let alpha = in_layout.read(data, base + bands - 1);
        let a = alpha / max;
        for b in 0..bands - 1 {
            out_layout.write(&mut out, base + b, in_layout.read(data, base + b) * a);
        }
        out_layout.write(&mut out, base + bands - 1, alpha);
    }
    Ok(Raster::new(src.width(), src.height(), out_fmt, out)?)
}

/// Un-premultiply the float working raster produced by [`premultiply_to_float`]
/// back into `dst_fmt` (the original source format), dividing each colour band
/// by the alpha and requantising exactly once. `max` is the source sample
/// ceiling used for the premultiply, so the round-trip cancels.
fn unpremultiply_from_float(
    src: &Raster,
    dst_fmt: PixelFormat,
    max: f64,
) -> Result<Raster, ResampleError> {
    let in_layout = SampleLayout::of(src.format());
    let out_layout = SampleLayout::of(dst_fmt);
    let bands = dst_fmt.channels();
    let data = src.data();
    let count = src.width() as usize * src.height() as usize;
    let mut out = vec![0u8; count * dst_fmt.bytes_per_pixel()];
    let mut px = vec![0.0f64; bands];
    for p in 0..count {
        let base = p * bands;
        for (b, v) in px.iter_mut().enumerate() {
            *v = in_layout.read(data, base + b);
        }
        unpremultiply(&mut px, max);
        for (b, &v) in px.iter().enumerate() {
            out_layout.write(&mut out, base + b, v);
        }
    }
    Ok(Raster::new(src.width(), src.height(), dst_fmt, out)?)
}

/// Bracket the alpha premultiply exactly once around a separable resample
/// `pipeline`, mirroring how `vips_resize` premultiplies before its reduce and
/// affine passes and un-premultiplies after.
///
/// When `bracket` is set and the source carries an alpha band, the source is
/// premultiplied into a float working raster, `pipeline` runs entirely in
/// premultiplied float space — so the intermediate between the vertical and
/// horizontal passes stays full-precision, with no per-axis requantisation of
/// low-alpha colour and no straight/premultiplied round-trip between axes — and
/// the result is un-premultiplied back to the source format once. Otherwise
/// `pipeline` runs directly on the source: no-alpha images need no coverage
/// weighting, and the Nearest single-tap kernel must stay an exact pick, since
/// bracketing would only requantise its semi-transparent RGB (#287/#288).
///
/// Callers pass `bracket = false` for the pure-passthrough cases (an identity
/// factor that does no work) so a no-op never pays for a premultiply /
/// un-premultiply round-trip.
fn with_premultiply<F>(src: &Raster, bracket: bool, pipeline: F) -> Result<Raster, ResampleError>
where
    F: FnOnce(&Raster) -> Result<Raster, ResampleError>,
{
    if !bracket || !src.format().has_alpha() {
        return pipeline(src);
    }
    let max = SampleLayout::of(src.format()).max;
    let work = premultiply_to_float(src, max)?;
    let reduced = pipeline(&work)?;
    unpremultiply_from_float(&reduced, src.format(), max)
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
        // Alpha is coverage-weighted once around both axes (#348): premultiply
        // into float, run the separable box / kernel passes, un-premultiply.
        if hshrink.fract() != 0.0 || vshrink.fract() != 0.0 {
            // Fractional factors delegate to reduce with the default
            // lanczos3 kernel and gap 1 (`vips_shrink_build`).
            with_premultiply(self, true, |w| {
                let t = reduce_axis(w, vshrink, ReduceKernel::Lanczos3, 1.0, Axis::Vertical)?;
                reduce_axis(&t, hshrink, ReduceKernel::Lanczos3, 1.0, Axis::Horizontal)
            })
        } else {
            // Integer factors run the plain box average on both axes; bracket
            // them the same way so integer and fractional shrink handle alpha
            // consistently (no factor-dependent bleed).
            with_premultiply(self, hshrink > 1.0 || vshrink > 1.0, |w| {
                let t = shrink_axis(w, vshrink as u32, false, Axis::Vertical)?;
                shrink_axis(&t, hshrink as u32, false, Axis::Horizontal)
            })
        }
    }

    /// Shrink by a pair of factors with a box filter (libvips
    /// `vips_shrink`). Integer factors run the plain box average; for
    /// fractional factors the residual is reduced with the default
    /// `lanczos3` kernel, exactly as libvips composes it. Output dimensions
    /// are `round(dim / factor)`. Panicking form of [`Raster::try_shrink`],
    /// matching the ported-test call surface.
    ///
    /// Alpha images are premultiplied around the whole shrink (both the integer
    /// and fractional paths) so transparent colour cannot bleed into opaque
    /// neighbours; see the module-level *Premultiplied alpha* note. This
    /// diverges from bare `vips_shrink`, which does not premultiply.
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
        with_premultiply(self, hshrink > 1, |w| {
            shrink_axis(w, hshrink, false, Axis::Horizontal)
        })
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
        with_premultiply(self, vshrink > 1, |w| {
            shrink_axis(w, vshrink, false, Axis::Vertical)
        })
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
        // Bracket the alpha premultiply once around both axes (#348); Nearest
        // is a single-tap pick that must not premultiply, and an identity
        // factor does no work so it needs no bracket either.
        let bracket = kernel != ReduceKernel::Nearest && (hshrink > 1.0 || vshrink > 1.0);
        with_premultiply(self, bracket, |w| {
            let t = reduce_axis(w, vshrink, kernel, 0.0, Axis::Vertical)?;
            reduce_axis(&t, hshrink, kernel, 0.0, Axis::Horizontal)
        })
    }

    /// Downsample with an anti-aliasing kernel (libvips `vips_reduce`):
    /// vertical pass then horizontal pass, no box pre-pass (gap 0), output
    /// dimensions `round(dim / factor)`. The kernel is a libvips nickname:
    /// `"nearest"`, `"linear"`, `"cubic"`, `"mitchell"`, `"lanczos2"`, or
    /// `"lanczos3"`. Panicking form of [`Raster::try_reduce`], matching the
    /// ported-test call surface.
    ///
    /// Alpha images (every kernel except Nearest) are premultiplied once around
    /// both axes so transparent colour cannot bleed into opaque neighbours; see
    /// the module-level *Premultiplied alpha* note. This diverges from bare
    /// `vips_reduce`, which does not premultiply.
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
        let bracket = kernel != ReduceKernel::Nearest && hshrink > 1.0;
        with_premultiply(self, bracket, |w| {
            reduce_axis(w, hshrink, kernel, 0.0, Axis::Horizontal)
        })
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
        let bracket = kernel != ReduceKernel::Nearest && vshrink > 1.0;
        with_premultiply(self, bracket, |w| {
            reduce_axis(w, vshrink, kernel, 0.0, Axis::Vertical)
        })
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

        let nearest = options.kernel == ReduceKernel::Nearest;
        let mut start = self.clone();

        // The nearest kernel subsamples the integer part first
        // (`vips_resize_build`).
        if nearest {
            let int_shrink = |dim: u32, s: f64| -> u32 {
                let f = if options.gap < 1.0 {
                    (1.0 / s).floor()
                } else {
                    let target = round_uint(f64::from(dim) * s).max(1) as f64;
                    (f64::from(dim) / target / options.gap).floor()
                };
                f.max(1.0) as u32
            };
            let int_h = int_shrink(start.width(), hscale);
            let int_v = int_shrink(start.height(), vscale);
            if int_h > 1 || int_v > 1 {
                start = subsample(&start, int_h, int_v)?;
                hscale *= f64::from(int_h);
                vscale *= f64::from(int_v);
            }
        }

        // Don't let either axis drop below one pixel.
        hscale = hscale.max(1.0 / f64::from(start.width()));
        vscale = vscale.max(1.0 / f64::from(start.height()));

        // Premultiply once around the *whole* resize — the residual reduce
        // passes and the affine enlargement together (#348/#406) — so every
        // separable pass is coverage-weighted and internally consistent. This
        // fixes the mixed downscale-one-axis / upscale-other-axis case (the
        // reduce emits premultiplied colour and the affine, running with
        // `premultiplied: true`, keeps it premultiplied rather than
        // interpolating straight-alpha colour across transparency boundaries)
        // and premultiplies a pure upscale too. The single un-premultiply
        // happens once after, in `with_premultiply`. Nearest never averages, so
        // it is not bracketed; nor is a no-op resize that neither reduces nor
        // enlarges.
        let will_reduce = vscale < 1.0 || hscale < 1.0;
        let will_upscale = hscale > 1.0 || vscale > 1.0;
        let bracket = !nearest && (will_reduce || will_upscale);
        with_premultiply(&start, bracket, |w| {
            let mut cur = w.clone();

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
        })
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
    /// As in `vips_resize`, an alpha image is premultiplied once around the
    /// whole operation — the reduce passes and the affine enlargement together
    /// — and unpremultiplied once at the end, so every axis is coverage-weighted
    /// consistently; see the module-level *Premultiplied alpha* note.
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
    /// a libvips nickname: `"nearest"`, `"bilinear"`, `"bicubic"`,
    /// `"nohalo"`, or `"lbb"`. Panicking form of [`Raster::try_affine`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown interpolator name or any
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
    /// Panics on an unknown interpolator name or any
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

// ---------------------------------------------------------------------------
// Thumbnail (vips_thumbnail / vips_thumbnail_image)
// ---------------------------------------------------------------------------

/// The colour space a thumbnail resamples in, mirroring the `vips_thumbnail`
/// `linear` and `export-profile` options.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThumbSpace {
    /// Resample directly in the decoded device space (the default).
    Device,
    /// Import to linear-light scRGB, resample, re-encode to sRGB
    /// (`--linear`).
    Linear,
    /// Import through the embedded ICC profile, resample in the PCS, then
    /// export to the built-in sRGB profile (`--export-profile srgb`).
    IccSrgb,
}

/// The built-in sRGB ICC profile bytes (moxcms `ColorProfile::new_srgb`),
/// the target for the [`ThumbSpace::IccSrgb`] export.
fn builtin_srgb_profile() -> Result<Vec<u8>, ThumbnailError> {
    moxcms::ColorProfile::new_srgb()
        .encode()
        .map_err(|e| ThumbnailError::Profile(format!("{e:?}")))
}

/// Resize `src` by `scale`, short-circuiting the exact-unit case so a
/// no-op thumbnail (target == source) keeps the pixels and metadata
/// untouched rather than round-tripping through the resampler.
fn resize_if_needed(src: &Raster, scale: f64) -> Result<Raster, ResampleError> {
    if (scale - 1.0).abs() <= f64::EPSILON {
        Ok(src.clone())
    } else {
        src.try_resize(scale)
    }
}

/// Fit an in-memory raster into a `width` x `height` box, the core of the
/// whole thumbnail family (libvips `vips_thumbnail_image`).
///
/// The shrink factor fits the image inside the bounding box preserving
/// aspect (the larger of the per-axis shrinks) or, when `crop` is set,
/// fills it (the smaller), exactly as `vips_thumbnail_calculate_shrink`.
/// The resample runs in the space `space` selects, and a crop centre-crops
/// the filled image down to the box.
fn thumbnail_fit(
    src: &Raster,
    width: u32,
    height: Option<u32>,
    crop: bool,
    space: ThumbSpace,
) -> Result<Raster, ThumbnailError> {
    if width == 0 {
        return Err(ThumbnailError::BadSize { size: 0 });
    }
    let box_w = width;
    let box_h = match height {
        Some(0) => return Err(ThumbnailError::BadSize { size: 0 }),
        Some(h) => h,
        // The bare-width forms fit a square box, matching the ported
        // `thumbnail(width)` call surface and the vips CLI where a single
        // size bounds both axes.
        None => width,
    };

    let horizontal = f64::from(src.width()) / f64::from(box_w);
    let vertical = f64::from(src.height()) / f64::from(box_h);
    let shrink = if crop {
        horizontal.min(vertical)
    } else {
        horizontal.max(vertical)
    };
    let scale = 1.0 / shrink;

    let fitted = match space {
        ThumbSpace::Device => resize_if_needed(src, scale)?,
        ThumbSpace::Linear => {
            let linear = src.try_colourspace(Interpretation::ScRgb)?;
            let small = resize_if_needed(&linear, scale)?;
            // The resampler resets the interpretation tag to the format
            // default, so restore scRGB before re-encoding to sRGB.
            let small = small.copy().interpretation(Interpretation::ScRgb).build();
            small.try_colourspace(Interpretation::Srgb)?
        }
        ThumbSpace::IccSrgb => {
            let lab = src.try_icc_import_with(Intent::Perceptual, None, Some(Pcs::Lab))?;
            let small = resize_if_needed(&lab, scale)?;
            // Restore the Lab tag (dropped by the resampler) so the export
            // takes the direct PCS path, and point the export at the
            // built-in sRGB profile.
            let mut small = small.copy().interpretation(Interpretation::Lab).build();
            small.set_icc_profile(&builtin_srgb_profile()?);
            small.try_icc_export_with(8, Intent::Perceptual, None)?
        }
    };

    if crop {
        let (ow, oh) = (fitted.width(), fitted.height());
        let cw = box_w.min(ow);
        let ch = box_h.min(oh);
        let left = (ow - cw) / 2;
        let top = (oh - ch) / 2;
        Ok(fitted.try_extract_area(left, top, cw, ch)?)
    } else {
        Ok(fitted)
    }
}

impl Raster {
    /// Fallible form of [`Raster::thumbnail`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::Decode`] when the file cannot be read or decoded,
    /// [`ThumbnailError::BadSize`] for a zero target, or the resample /
    /// crop errors from the fit.
    pub fn try_thumbnail(
        path: &Path,
        width: u32,
        height: Option<u32>,
        crop: bool,
    ) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_file(path)?;
        thumbnail_fit(&src, width, height, crop, ThumbSpace::Device)
    }

    /// Make a thumbnail from an image file (libvips `vips_thumbnail`).
    ///
    /// The image is loaded and shrunk to fit inside the `width` x `height`
    /// bounding box preserving aspect ratio; a bare width (`height` is
    /// `None`) fits a square `width` x `width` box, so the bound axis lands
    /// exactly on `width`. With `crop` the image fills the box and is
    /// centre-cropped to it. The heavy shrink runs through
    /// [`Raster::resize`], whose gap-driven box pre-shrink keeps the reduce
    /// mask bounded even for large downscales, matching the shrink-on-load
    /// then residual-reduce shape of `vips_thumbnail`. Panicking form of
    /// [`Raster::try_thumbnail`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail`].
    #[track_caller]
    pub fn thumbnail(path: &Path, width: u32, height: Option<u32>, crop: bool) -> Raster {
        expect_thumbnail(Self::try_thumbnail(path, width, height, crop))
    }

    /// Fallible form of [`Raster::thumbnail_buffer`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_thumbnail`]; decodes from memory instead of a file.
    pub fn try_thumbnail_buffer(data: &[u8], width: u32) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_bytes(data)?;
        thumbnail_fit(&src, width, None, false, ThumbSpace::Device)
    }

    /// Make a thumbnail from an in-memory encoded image buffer (libvips
    /// `vips_thumbnail_buffer`), fitting a square `width` x `width` box.
    /// Panicking form of [`Raster::try_thumbnail_buffer`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail_buffer`].
    #[track_caller]
    pub fn thumbnail_buffer(data: &[u8], width: u32) -> Raster {
        expect_thumbnail(Self::try_thumbnail_buffer(data, width))
    }

    /// Fallible form of [`Raster::thumbnail_with_options`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_thumbnail`], plus [`ThumbnailError::Colour`] from
    /// the linear-light import / export.
    pub fn try_thumbnail_with_options(
        path: &Path,
        width: u32,
        linear: bool,
    ) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_file(path)?;
        let space = if linear {
            ThumbSpace::Linear
        } else {
            ThumbSpace::Device
        };
        thumbnail_fit(&src, width, None, false, space)
    }

    /// Make a thumbnail with the `vips_thumbnail` `linear` option: when
    /// `linear` is set the reduce runs in linear-light scRGB and the result
    /// is re-encoded to sRGB, which avoids the darkening a naive gamma-space
    /// average produces. Fits a square `width` x `width` box. Panicking form
    /// of [`Raster::try_thumbnail_with_options`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`].
    #[track_caller]
    pub fn thumbnail_with_options(path: &Path, width: u32, linear: bool) -> Raster {
        expect_thumbnail(Self::try_thumbnail_with_options(path, width, linear))
    }

    /// Fallible form of [`Raster::thumbnail_with_profile`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::UnknownProfile`] for an output profile other than
    /// `"srgb"`, plus the decode / colour / resample errors.
    pub fn try_thumbnail_with_profile(
        path: &Path,
        width: u32,
        output_profile: &str,
    ) -> Result<Raster, ThumbnailError> {
        let space = match output_profile {
            "srgb" | "sRGB" => ThumbSpace::IccSrgb,
            other => {
                return Err(ThumbnailError::UnknownProfile {
                    name: other.to_string(),
                });
            }
        };
        let src = crate::source::decode_file(path)?;
        thumbnail_fit(&src, width, None, false, space)
    }

    /// Make a thumbnail through the embedded ICC profile (libvips
    /// `vips_thumbnail` with `export-profile`): the image is imported from
    /// its embedded profile to the Lab PCS, reduced there, and exported to
    /// `output_profile` (only the built-in `"srgb"` today). Fits a square
    /// `width` x `width` box. Panicking form of
    /// [`Raster::try_thumbnail_with_profile`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`].
    #[track_caller]
    pub fn thumbnail_with_profile(path: &Path, width: u32, output_profile: &str) -> Raster {
        expect_thumbnail(Self::try_thumbnail_with_profile(
            path,
            width,
            output_profile,
        ))
    }

    /// Fallible form of [`Raster::thumbnail_image`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::BadSize`] for a zero target, or the resample
    /// errors from the fit.
    pub fn try_thumbnail_image(&self, width: u32) -> Result<Raster, ThumbnailError> {
        thumbnail_fit(self, width, None, false, ThumbSpace::Device)
    }

    /// Make a thumbnail from this already-loaded raster (libvips
    /// `vips_thumbnail_image`), fitting a square `width` x `width` box
    /// preserving aspect ratio. This is the in-memory counterpart to the
    /// file-loading [`Raster::thumbnail`]; the sequential-access ported
    /// cell drives it after a decode. Panicking form of
    /// [`Raster::try_thumbnail_image`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail_image`].
    #[track_caller]
    pub fn thumbnail_image(&self, width: u32) -> Raster {
        expect_thumbnail(self.try_thumbnail_image(width))
    }
}

/// Map a [`ThumbnailError`] onto the shared decode error, preserving the
/// decode cause and folding the resample/colour/crop steps into an I/O error.
fn thumbnail_to_decode(err: ThumbnailError) -> crate::codec::DecodeError {
    match err {
        ThumbnailError::Decode(source) => source,
        other => crate::source::SourceError::Io(std::io::Error::other(other.to_string())),
    }
}

/// Make a thumbnail from an image file, bounded by `width` (libvips
/// `vips_thumbnail` bare-width form).
///
/// A convenience free function over [`Raster::try_thumbnail`] returning the
/// shared [`crate::codec::DecodeError`], matching the ported foreign cell's
/// `thumbnail(path, width)` surface. The image is loaded and shrunk to fit
/// inside a `width` x `width` box, preserving aspect ratio.
///
/// # Errors
///
/// A [`crate::codec::DecodeError`] when the file cannot be read or decoded,
/// or when the resample step fails.
pub fn thumbnail(path: &Path, width: u32) -> Result<Raster, crate::codec::DecodeError> {
    Raster::try_thumbnail(path, width, None, false).map_err(thumbnail_to_decode)
}

/// Make a thumbnail from an image file into a `width` x `height` box with a
/// crop mode (libvips `vips_thumbnail` with `crop`).
///
/// `crop` selects the fit: `"none"` (or an empty string) fits inside the box
/// preserving aspect ratio, and any other value (for example `"centre"`)
/// fills the box and centre-crops to it.
///
/// # Errors
///
/// As [`thumbnail`], plus the crop step.
pub fn thumbnail_crop(
    path: &Path,
    width: u32,
    height: u32,
    crop: &str,
) -> Result<Raster, crate::codec::DecodeError> {
    let do_crop = !matches!(crop, "" | "none");
    Raster::try_thumbnail(path, width, Some(height), do_crop).map_err(thumbnail_to_decode)
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

    /// An unknown kernel or interpolator nickname is a typed error, and
    /// every recognised interpolator nickname parses, including the
    /// nohalo and lbb minmod resamplers.
    #[test]
    fn kernel_and_interpolator_parsing() {
        assert!(matches!(
            ReduceKernel::from_name("box"),
            Err(ResampleError::UnknownKernel { .. })
        ));
        assert_eq!(
            Interpolator::from_name("nohalo").unwrap(),
            Interpolator::Nohalo
        );
        assert_eq!(Interpolator::from_name("lbb").unwrap(), Interpolator::Lbb);
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

    /// Regression for #288: the reduce (Lanczos) path must premultiply alpha
    /// so the RGB of transparent pixels cannot bleed across a transparency
    /// boundary. A 24x2 RGBA raster is opaque red on the left (cols 0..11) and
    /// *transparent green* on the right (cols 12..23); a Lanczos3 reduceh has
    /// taps that cross the seam. Under straight-alpha convolution (the pre-fix
    /// behaviour) the transparent green leaks into the output (G > 0) and the
    /// opaque red darkens at the seam. With the premultiply bracket the green,
    /// carrying zero coverage, contributes nothing: G stays 0 everywhere and
    /// the opaque colour stays saturated wherever any coverage survives.
    #[test]
    fn reduce_premultiplies_alpha_no_colour_bleed() {
        let w = 24u32;
        let h = 2u32;
        let mut data = Vec::with_capacity((w * h) as usize * 4);
        for _ in 0..h {
            for x in 0..w {
                if x < 12 {
                    data.extend_from_slice(&[255, 0, 0, 255]); // opaque red
                } else {
                    data.extend_from_slice(&[0, 255, 0, 0]); // transparent green
                }
            }
        }
        let im = Raster::new(w, h, PixelFormat::Rgba8, data).unwrap();
        let out = im.reduceh(2.0, "lanczos3");
        assert_eq!((out.width(), out.height()), (12, 2));

        let ow = out.width() as usize;
        let mut saw_opaque = false;
        let mut saw_transparent = false;
        for (i, chunk) in out.data().chunks(4).enumerate() {
            let (r, g, b, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            let col = i % ow;
            // The transparent green must never bleed into any output pixel.
            assert!(
                g <= 1,
                "G bled at col {col}: {g} (transparent green leaked)"
            );
            assert!(b <= 1, "B bled at col {col}: {b}");
            // Wherever coverage survives, the opaque red is preserved intact
            // (no dark fringe): un-premultiply restores the saturated colour.
            if a > 0 {
                assert!(r >= 254, "R darkened at col {col}: {r} (dark fringe)");
            }
            // Deep opaque columns stay fully opaque saturated red...
            if col == 0 {
                assert_eq!((r, a), (255, 255), "deep opaque column must survive");
                saw_opaque = true;
            }
            // ...and deep transparent columns stay fully transparent (colour
            // zeroed, no leaked green).
            if col == ow - 1 {
                assert_eq!(a, 0, "deep transparent column must stay transparent");
                assert_eq!((r, g, b), (0, 0, 0), "transparent colour must be zeroed");
                saw_transparent = true;
            }
        }
        assert!(
            saw_opaque && saw_transparent,
            "fixture must span both regions"
        );
    }

    /// Follow-up to #287/#288: the reduce path must NOT premultiply for the
    /// Nearest kernel. Nearest is a single-tap pick with no averaging, so a
    /// premultiply -> un-premultiply round-trip through the same-bit-depth
    /// integer raster would only requantise — and thus corrupt — the straight-
    /// alpha RGB of semi-transparent pixels (e.g. `(200,100,50,10)` round-trips
    /// to `(204,102,51,10)`). A single-tap nearest pick must return each
    /// selected source pixel byte-identically.
    #[test]
    fn reduce_nearest_preserves_exact_alpha_pixels() {
        // Four semi-transparent colours whose RGB does not survive the
        // premultiply/un-premultiply integer round-trip. Tiled on a 2x2 lattice
        // so every source pixel is one of the four; a single-tap pick can only
        // ever return one of them exactly.
        const PALETTE: [[u8; 4]; 4] = [
            [200, 100, 50, 10],
            [30, 220, 140, 3],
            [170, 90, 240, 7],
            [90, 200, 60, 5],
        ];
        let (w, h) = (4u32, 4u32);
        let mut data = Vec::with_capacity((w * h) as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let idx = (x % 2) as usize + 2 * (y % 2) as usize;
                data.extend_from_slice(&PALETTE[idx]);
            }
        }
        let im = Raster::new(w, h, PixelFormat::Rgba8, data).unwrap();

        for (label, out) in [
            ("reduceh", im.reduceh(2.0, "nearest")),
            ("reducev", im.reducev(2.0, "nearest")),
            ("reduce", im.reduce(2.0, 2.0, "nearest")),
            ("reduce-fractional", im.reduce(1.5, 1.5, "nearest")),
        ] {
            assert_eq!(out.format(), PixelFormat::Rgba8);
            for px in out.data().as_chunks::<4>().0 {
                assert!(
                    PALETTE.contains(px),
                    "{label} nearest corrupted a semi-transparent pixel: {px:?} \
                     is not an exact source sample (premultiply round-trip)"
                );
            }
        }
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

    // -----------------------------------------------------------------
    // Thumbnail
    // -----------------------------------------------------------------

    /// A 290x442 portrait RGB raster, the aspect of the libvips `sample.jpg`
    /// fixture, so the thumbnail dimension checks reuse the vips oracle.
    fn portrait_290x442() -> Raster {
        let (w, h) = (290u32, 442u32);
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push((x % 256) as u8);
                data.push((y % 256) as u8);
                data.push(((x + y) % 256) as u8);
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// A one-pixel black/white checkerboard: every 2x downscale sees equal
    /// black and white, so linear and gamma-space averages diverge sharply.
    fn checker(side: u32) -> Raster {
        let mut data = Vec::with_capacity((side * side * 3) as usize);
        for y in 0..side {
            for x in 0..side {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        Raster::new(side, side, PixelFormat::Rgb8, data).unwrap()
    }

    /// The fit dimensions match the real `vips thumbnail` / `vipsthumbnail`
    /// oracle for the 290x442 aspect: a bare width fits a square box (the
    /// bound axis lands exactly), a width x height box fits inside it, and
    /// crop fills it.
    #[test]
    fn thumbnail_fit_matches_vips_oracle_dims() {
        let im = portrait_290x442();
        let dims = |w, h, crop| {
            let t = thumbnail_fit(&im, w, h, crop, ThumbSpace::Device).unwrap();
            (t.width(), t.height())
        };
        // Square boxes: the bound (larger) axis lands exactly on the size.
        assert_eq!(dims(100, None, false), (66, 100));
        assert_eq!(dims(128, None, false), (84, 128));
        assert_eq!(dims(442, None, false), (290, 442));
        // Rectangular boxes fit inside, preserving aspect.
        assert_eq!(dims(100, Some(300), false), (100, 152));
        assert_eq!(dims(300, Some(100), false), (66, 100));
        // Crop fills the box exactly.
        assert_eq!(dims(100, Some(300), true), (100, 300));
    }

    /// The bare-width thumbnail always lands the target on the bound axis,
    /// the libvips `for height in range(440, 1, -13)` invariant.
    #[test]
    fn thumbnail_height_series_is_exact() {
        let im = portrait_290x442();
        let mut h = 440u32;
        while h >= 2 {
            let t = thumbnail_fit(&im, h, None, false, ThumbSpace::Device).unwrap();
            assert_eq!(t.height(), h, "bound axis must land exactly for {h}");
            h = h.saturating_sub(13);
        }
    }

    /// A plain shrink preserves the mean, the `|avg_orig - avg_thumb| < 1`
    /// invariant of the ported thumbnail cell.
    #[test]
    fn thumbnail_preserves_average() {
        let im = portrait_290x442();
        let t = thumbnail_fit(&im, 100, None, false, ThumbSpace::Device).unwrap();
        assert!(
            (im.avg() - t.avg()).abs() < 1.0,
            "mean drifted: {}",
            t.avg()
        );
    }

    /// Reducing in linear light lifts the mean of a black/white pattern well
    /// above the gamma-space average, so the linear path is demonstrably not
    /// a plain reduce.
    #[test]
    fn thumbnail_linear_differs_from_naive() {
        let im = checker(64);
        let naive = thumbnail_fit(&im, 8, None, false, ThumbSpace::Device).unwrap();
        let linear = thumbnail_fit(&im, 8, None, false, ThumbSpace::Linear).unwrap();
        assert_eq!((linear.width(), linear.height()), (8, 8));
        assert_eq!(linear.format().channels(), 3);
        // Gamma-space average of 0 and 255 is ~127; linear-light average of
        // the same pair re-encodes to ~188.
        assert!(naive.avg() < 140.0, "naive avg {}", naive.avg());
        assert!(
            linear.avg() > naive.avg() + 30.0,
            "linear {} should exceed naive {} by a wide margin",
            linear.avg(),
            naive.avg()
        );
    }

    /// The file and buffer entry points decode the same bytes and agree, and
    /// the associated `thumbnail` resolves to this inherent method.
    #[test]
    fn thumbnail_file_and_buffer_agree() {
        let im = portrait_290x442();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.png");
        im.save(&path).unwrap();

        let by_file = Raster::thumbnail(&path, 100, None, false);
        assert_eq!((by_file.width(), by_file.height()), (66, 100));
        assert_eq!(by_file.format().channels(), 3);

        let buf = std::fs::read(&path).unwrap();
        let by_buf = Raster::thumbnail_buffer(&buf, 100);
        assert_eq!((by_buf.width(), by_buf.height()), (66, 100));
        assert!((by_file.avg() - by_buf.avg()).abs() < 1.0);

        // The in-memory instance form fits the same square box.
        let by_image = im.thumbnail_image(100);
        assert_eq!((by_image.width(), by_image.height()), (66, 100));
    }

    /// The ICC export path imports through the attached profile and exports
    /// to the built-in sRGB profile; a source already in sRGB round-trips
    /// close to identity, exercising the whole import/reduce/export machine.
    #[test]
    fn thumbnail_icc_srgb_roundtrips() {
        let mut im = portrait_290x442();
        im.set_icc_profile(&builtin_srgb_profile().unwrap());
        let t = thumbnail_fit(&im, 442, None, false, ThumbSpace::IccSrgb).unwrap();
        assert_eq!((t.width(), t.height()), (290, 442));
        assert_eq!(t.format().channels(), 3);
        // sRGB -> Lab(D50) -> sRGB is near identity at 8-bit.
        assert!(
            (im.avg() - t.avg()).abs() < 4.0,
            "sRGB ICC round-trip drifted: {} vs {}",
            im.avg(),
            t.avg()
        );
    }

    /// A zero target is a typed error, not a panic in the fit math.
    #[test]
    fn thumbnail_zero_size_is_typed_error() {
        let im = portrait_290x442();
        assert!(matches!(
            im.try_thumbnail_image(0),
            Err(ThumbnailError::BadSize { size: 0 })
        ));
    }

    /// An unknown output profile is reported, not silently treated as sRGB.
    #[test]
    fn thumbnail_unknown_profile_is_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.png");
        portrait_290x442().save(&path).unwrap();
        assert!(matches!(
            Raster::try_thumbnail_with_profile(&path, 100, "adobe-rgb"),
            Err(ThumbnailError::UnknownProfile { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Nohalo / LBB interpolators, pinned to a real libvips 8.18.3 oracle
    // -----------------------------------------------------------------

    /// The 16x16 single-band fixture the libvips oracle affines: sharp
    /// 4x4 block-parity edges so the nohalo minmod slopes and the LBB
    /// range limiters both activate. The generator formula is
    /// `base = (x*17 + y*29) % 256; v = 255 - base` on the block-parity
    /// squares, `base` elsewhere.
    fn oracle_16x16() -> Raster {
        let mut data = vec![0u8; 16 * 16];
        for y in 0..16usize {
            for x in 0..16usize {
                let base = ((x * 17 + y * 29) % 256) as u8;
                let bx = (x / 4) % 2;
                let by = (y / 4) % 2;
                data[y * 16 + x] = if bx ^ by != 0 { 255 - base } else { base };
            }
        }
        Raster::new(16, 16, PixelFormat::Gray8, data).unwrap()
    }

    /// The interior 12x12 crop `[6, 6, 12, 12]` of the 28x28 affine of the
    /// oracle fixture by `[1.5, 0.25, -0.25, 1.5]` matches real libvips
    /// 8.18.3 byte for byte, for every interpolator. Pinning nearest,
    /// bilinear, and bicubic confirms the affine geometry and rounding
    /// agree with libvips; pinning nohalo and lbb confirms the two
    /// minmod-subdivision ports are faithful. The interior crop keeps the
    /// stencils off the image edge so the comparison is pure kernel math.
    #[test]
    fn affine_interpolators_match_libvips_oracle() {
        // Captured with: vips affine in.pgm out.v "1.5 0.25 -0.25 1.5"
        //   --interpolate INTERP  (libvips 8.18.3), interior crop [6,6,12,12].
        #[rustfmt::skip]
        let oracle: [(&str, [u8; 144]); 5] = [
            ("nearest", [
                80, 129, 129, 112, 95, 95, 78, 194, 194, 211, 1, 1,
                109, 129, 129, 112, 66, 66, 49, 223, 223, 240, 1, 1,
                138, 100, 100, 83, 66, 66, 49, 223, 3, 242, 242, 225,
                138, 100, 184, 201, 201, 218, 235, 235, 3, 242, 242, 196,
                88, 88, 184, 201, 201, 247, 8, 8, 230, 213, 213, 196,
                59, 59, 213, 230, 230, 247, 8, 8, 201, 201, 184, 167,
                59, 59, 242, 242, 3, 20, 20, 37, 201, 201, 184, 138,
                47, 30, 242, 242, 3, 49, 49, 66, 172, 172, 155, 138,
                18, 1, 15, 15, 32, 49, 49, 66, 112, 112, 129, 129,
                18, 1, 211, 211, 194, 194, 177, 160, 160, 112, 129, 129,
                10, 27, 27, 211, 194, 194, 148, 131, 131, 141, 158, 158,
                39, 56, 56, 182, 165, 165, 148, 131, 131, 170, 170, 187,
            ]),
            ("bilinear", [
                123, 122, 108, 94, 79, 65, 146, 218, 232, 126, 10, 41,
                118, 105, 91, 77, 67, 76, 129, 156, 212, 195, 152, 170,
                123, 129, 138, 151, 169, 192, 149, 3, 165, 231, 217, 202,
                122, 184, 198, 212, 226, 175, 116, 143, 199, 214, 200, 185,
                103, 188, 215, 229, 243, 122, 42, 177, 211, 197, 183, 168,
                76, 185, 187, 115, 101, 62, 37, 148, 194, 180, 166, 151,
                40, 173, 157, 13, 21, 36, 50, 128, 177, 163, 149, 136,
                17, 66, 70, 31, 38, 55, 76, 120, 150, 143, 141, 144,
                7, 33, 84, 106, 123, 135, 143, 139, 112, 126, 140, 154,
                19, 82, 211, 197, 183, 169, 154, 140, 132, 143, 157, 171,
                33, 80, 176, 180, 166, 152, 137, 128, 144, 160, 174, 188,
                50, 77, 150, 163, 149, 135, 120, 108, 147, 177, 191, 205,
            ]),
            ("bicubic", [
                126, 124, 107, 89, 67, 46, 150, 249, 248, 117, 0, 9,
                118, 95, 82, 71, 63, 60, 124, 158, 228, 209, 152, 180,
                124, 122, 133, 148, 172, 225, 155, 3, 162, 253, 235, 228,
                117, 184, 206, 226, 247, 196, 105, 145, 211, 216, 200, 189,
                96, 193, 234, 248, 255, 111, 10, 198, 239, 197, 182, 169,
                69, 202, 210, 109, 96, 46, 33, 155, 207, 180, 165, 149,
                38, 201, 182, 0, 0, 14, 41, 131, 188, 164, 148, 135,
                10, 61, 54, 16, 33, 54, 69, 120, 152, 141, 138, 141,
                2, 17, 64, 105, 129, 143, 151, 141, 112, 121, 137, 153,
                0, 79, 211, 214, 195, 180, 165, 143, 128, 140, 157, 171,
                13, 78, 191, 194, 168, 152, 136, 127, 144, 161, 174, 188,
                44, 73, 156, 171, 149, 134, 117, 107, 147, 181, 193, 208,
            ]),
            ("nohalo", [
                127, 125, 106, 89, 70, 54, 144, 232, 239, 120, 2, 21,
                121, 97, 83, 72, 64, 61, 125, 160, 221, 200, 156, 181,
                126, 121, 130, 144, 166, 205, 159, 3, 168, 240, 226, 211,
                118, 184, 202, 220, 237, 181, 111, 150, 208, 217, 200, 185,
                97, 191, 222, 238, 247, 122, 23, 196, 220, 196, 183, 169,
                69, 195, 193, 111, 99, 56, 33, 155, 200, 180, 163, 144,
                38, 184, 169, 4, 11, 29, 46, 132, 184, 164, 148, 134,
                6, 60, 64, 21, 34, 54, 73, 122, 147, 137, 137, 142,
                2, 22, 81, 112, 131, 143, 151, 138, 112, 119, 136, 150,
                9, 85, 211, 206, 191, 177, 160, 143, 126, 137, 156, 171,
                26, 79, 188, 188, 167, 152, 133, 126, 143, 159, 174, 188,
                50, 75, 156, 167, 148, 135, 117, 108, 145, 179, 194, 208,
            ]),
            ("lbb", [
                126, 124, 106, 89, 69, 47, 154, 233, 238, 116, 2, 19,
                117, 96, 81, 71, 63, 57, 131, 161, 226, 203, 161, 188,
                125, 122, 134, 148, 172, 215, 164, 3, 176, 238, 228, 216,
                118, 184, 206, 222, 238, 184, 104, 154, 214, 215, 200, 189,
                95, 193, 233, 238, 246, 131, 19, 201, 224, 196, 182, 170,
                68, 198, 197, 107, 93, 53, 33, 157, 208, 180, 165, 149,
                38, 196, 181, 4, 9, 17, 40, 132, 188, 164, 148, 135,
                10, 65, 60, 19, 34, 53, 69, 119, 152, 141, 137, 141,
                2, 20, 70, 108, 130, 143, 152, 139, 112, 121, 137, 153,
                8, 85, 211, 203, 194, 180, 165, 145, 128, 140, 157, 171,
                17, 77, 187, 190, 167, 152, 135, 126, 144, 161, 174, 188,
                44, 73, 156, 172, 149, 135, 117, 107, 148, 181, 194, 208,
            ]),
        ];

        let im = oracle_16x16();
        for (name, expected) in oracle {
            let out = im.affine([1.5, 0.25, -0.25, 1.5], name);
            assert_eq!(out.width(), 28, "{name} width");
            assert_eq!(out.height(), 28, "{name} height");
            let interior = out.extract_area(6, 6, 12, 12);
            let got = interior.data();
            let (mismatches, worst) =
                got.iter()
                    .zip(expected.iter())
                    .fold((0usize, 0u8), |(n, worst), (&a, &b)| {
                        let d = a.abs_diff(b);
                        (n + usize::from(d != 0), worst.max(d))
                    });
            // nohalo and lbb reproduce libvips byte for byte (0 of 144),
            // which is the exact-parity gate for this work: both compute
            // their Hermite coefficients directly, just like `nohalo.cpp`
            // and `lbb.cpp`. The other three kernels only bound the affine
            // geometry: bilinear differs at a single `.5` rounding tie
            // (delta 1); nearest at 2 equidistant-neighbour ties (a
            // whole-pixel swap, so a large delta but the adjacent sample);
            // and bicubic within delta 3 because libvips bicubic reads
            // precomputed coefficient tables quantised to a fixed-point
            // sub-pixel grid (VIPS_TRANSFORM_SCALE) while this direct
            // Catmull-Rom does not. None of those conventions touch the
            // on-grid ported test_affine round-trip.
            let (allowed_count, allowed_delta) = match name {
                "nohalo" | "lbb" => (0, 0),
                "bilinear" => (1, 1),
                "nearest" => (2, u8::MAX),
                "bicubic" => (60, 3),
                _ => unreachable!(),
            };
            assert!(
                mismatches <= allowed_count && worst <= allowed_delta,
                "{name} differs from the libvips oracle in {mismatches} bytes \
                 (worst delta {worst}); expected at most {allowed_count} bytes, delta {allowed_delta}"
            );
        }
    }

    /// The transpose matrix `[0, 1, 1, 0]` samples exactly on the input
    /// grid, so every interpolator (including nohalo and lbb) reproduces
    /// the transpose byte for byte, and four applications are the
    /// identity: the ported test_affine round-trip invariant.
    #[test]
    fn nohalo_lbb_transpose_round_trip_is_identity() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        for interp in ["nohalo", "lbb"] {
            let t = im.affine([0.0, 1.0, 1.0, 0.0], interp);
            assert_eq!(t.width(), im.height(), "{interp} transpose width");
            assert_eq!(t.height(), im.width(), "{interp} transpose height");
            assert_eq!(t.data(), reference.data(), "{interp} transpose bytes");

            let mut x = im.clone();
            for _ in 0..4 {
                x = x.affine([0.0, 1.0, 1.0, 0.0], interp);
            }
            assert_eq!(x.data(), im.data(), "{interp} 4x transpose not identity");
        }
    }

    /// LBB stays locally bounded: an upscale never overshoots the range of
    /// the input samples, the defining property of the resampler (no
    /// output clamping needed). Nohalo, being co-monotone, likewise keeps
    /// a monotone ramp within its endpoints.
    #[test]
    fn nohalo_lbb_stay_within_input_range() {
        // A ramp with a sharp central step, the classic overshoot probe:
        // a plain bicubic rings above 255 / below 0 at the step, lbb and
        // nohalo may not.
        let mut data = vec![0u8; 12 * 12];
        for y in 0..12usize {
            for x in 0..12usize {
                data[y * 12 + x] = if x < 6 { 30 } else { 220 };
            }
        }
        let im = Raster::new(12, 12, PixelFormat::Gray8, data).unwrap();
        for interp in ["nohalo", "lbb"] {
            // A 2.5x upscale around the step, the regime where cubic
            // resamplers overshoot.
            let up = im.affine([2.5, 0.0, 0.0, 2.5], interp);
            // Sample the interior to avoid the background-extended border.
            let inner = up.extract_area(4, 4, up.width() - 8, up.height() - 8);
            let (lo, hi) = inner
                .data()
                .iter()
                .fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            assert!(
                lo >= 30 && hi <= 220,
                "{interp} overshot the [30, 220] input range: got [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn thumbnail_free_fn_fits_the_width_box() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thumb_in.png");
        Raster::new(100, 60, PixelFormat::Rgb8, vec![120u8; 100 * 60 * 3])
            .unwrap()
            .save(&path)
            .unwrap();

        // Bare-width fit into a 50x50 box: shrink = max(100/50, 60/50) = 2.
        let thumb = super::thumbnail(&path, 50).unwrap();
        assert_eq!(thumb.width(), 50);
        assert!(
            (i64::from(thumb.height()) - 30).abs() <= 1,
            "height {} not near 30",
            thumb.height()
        );
    }

    #[test]
    fn thumbnail_crop_free_fn_fills_and_crops_the_box() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thumb_crop_in.png");
        Raster::new(100, 60, PixelFormat::Rgb8, vec![90u8; 100 * 60 * 3])
            .unwrap()
            .save(&path)
            .unwrap();

        // crop="centre" fills a 40x40 box and centre-crops to it.
        let thumb = super::thumbnail_crop(&path, 40, 40, "centre").unwrap();
        assert_eq!(thumb.width(), 40);
        assert_eq!(thumb.height(), 40);

        // crop="none" fits inside the box, preserving aspect ratio.
        let fit = super::thumbnail_crop(&path, 40, 40, "none").unwrap();
        assert!(fit.width() <= 40 && fit.height() <= 40);
        assert!(fit.width() == 40 || fit.height() == 40);
    }
}
