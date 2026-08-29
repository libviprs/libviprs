//! Arithmetic and statistics operations ported from libvips.
//!
//! This module is the second batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`]): whole-image
//! reductions, per-sample arithmetic against constants and images,
//! comparisons, bitwise operations, and the statistical enhancement ops.
//! Operations that can fail on caller input exist in two forms, following
//! the [`crate::bands`] convention:
//!
//! * a fallible `try_*` method returning `Result<_, ArithmeticError>` with
//!   typed errors for mismatched dimensions, band counts, and malformed
//!   arguments; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`sub`, `add_vec`, `recomb`, ...) exactly, delegating to the `try_*`
//!   form and `expect`ing the result.
//!
//! The integer constant family (`add_const`, `sub_const`, `mul_const`,
//! `floordiv_const`, `pow_const`, `rem_const`) follows this convention too:
//! it rounds and saturates into an unsigned output and so rejects float
//! input, a data-dependent failure surfaced by the `try_*_const` forms
//! (libviprs#281). Operations that genuinely cannot fail on caller input —
//! the whole-image reductions and the float-output family that accepts every
//! depth ([`Raster::div_const`], [`Raster::linear`]) — have only the single
//! infallible form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::avg`] | `vips_avg` | mean of every sample, `f64` |
//! | [`Raster::deviate`] | `vips_deviate` | sample standard deviation, `f64` |
//! | [`Raster::min`] / [`Raster::max`] | `vips_min` / `vips_max` | extremum sample, `f64` |
//! | [`Raster::minpos`] / [`Raster::maxpos`] | `vips_min` / `vips_max` with position | `(value, x, y)` |
//! | [`Raster::stats`] | `vips_stats` | per-band and overall statistics matrix |
//! | [`Raster::measure`] | `vips_measure` | patch-grid mean matrix |
//! | [`Raster::find_trim`] | `vips_find_trim` | content bounding box |
//! | [`Raster::profile`] | `vips_profile` | first non-zero positions |
//! | [`Raster::project`] | `vips_project` | column and row sums |
//! | [`Raster::add_const`], [`Raster::sub_const`], ... | `vips_linear1` family | per-sample constant arithmetic |
//! | [`Raster::add_vec`], [`Raster::sub_vec`], ... | `vips_linear` family | per-band constant arithmetic |
//! | [`Raster::sub`] | `vips_subtract` | float raster (signed differences survive) |
//! | [`Raster::mul`] | `vips_multiply` | image-image arithmetic |
//! | [`Raster::div`], [`Raster::div_const`], [`Raster::div_vec`] | `vips_divide` | float raster |
//! | [`Raster::remainder`] | `vips_remainder` | samplewise `a % b`, integer raster |
//! | [`Raster::linear`] / [`Raster::linear_uchar`] | `vips_linear1` (default / `uchar` option) | `a * x + b`, float / uchar raster |
//! | [`Raster::sum`] | `vips_sum` | pixelwise sum of an image list |
//! | [`Raster::minpair`] / [`Raster::maxpair`] | `vips_minpair` / `vips_maxpair` | pixelwise extremum of two images |
//! | [`Raster::more_than`] family | `vips_relational` | `0` / `255` uchar mask |
//! | [`Raster::bitand`] family, [`Raster::lshift`], [`Raster::rshift`] | `vips_boolean` | bitwise arithmetic |
//! | [`Raster::scaleimage`] | `vips_scale` | values scaled to `0..=255` |
//! | [`Raster::stdif`] | `vips_stdif` | statistical differencing |
//! | [`Raster::recomb`] | `vips_recomb` | band recombination matrix multiply |
//! | [`Raster::premultiply`] / [`Raster::unpremultiply`] | `vips_premultiply` / `vips_unpremultiply` | alpha (un)premultiplication |
//! | [`Raster::sin`] .. [`Raster::atanh`], [`Raster::log`], [`Raster::log10`], [`Raster::exp`], [`Raster::exp10`] | `vips_math` | float raster |
//! | [`Raster::atan2`], [`Raster::pow`], [`Raster::wop`] | `vips_math2` | float raster |
//! | [`Raster::neg`] | pyvips `-image` | float raster |
//! | [`Raster::complexform`], [`Raster::polar`], [`Raster::rect`], [`Raster::conj`], [`Raster::real`], [`Raster::imag`] | `vips_complexform` / `vips_complex` / `vips_complexget` | complex (re/im pair) float raster |
//! | [`Raster::hough_line`] (vips-exact binning) / [`Raster::hough_circle`] (golden-only, see its docs) | `vips_hough_line` / `vips_hough_circle` | vote accumulator |
//!
//! # Semantics shared by the integer operations
//!
//! * **Value domain.** Samples are unsigned integers, `0..=255` (8-bit) or
//!   `0..=65535` (16-bit). Arithmetic is computed in `f64` and the result is
//!   rounded to nearest and saturated into the output depth. This integer
//!   round-and-saturate contract is kept exactly where libvips keeps
//!   integer output: `vips_add` / `vips_multiply` map integer input to
//!   integer output, so `add` / `mul` and their constant forms stay integer
//!   here. The divide family and `linear` promote to float output instead
//!   (see below), as does the transcendental family, and — matching the
//!   `vips_subtract` promotion to signed `short` — so does image-image
//!   `sub` (issue #282; the constant/per-band `sub_const` / `sub_vec`
//!   stay integer and saturate, see their own docs).
//! * **Depth promotion.** Operations whose exact result can exceed the
//!   input depth (`add_const`, `mul`, `pow_const`, `sum`, ...)
//!   promote 8-bit input to 16-bit output, matching the promotion
//!   [`Raster::add`] already performs. 16-bit input has no wider format and
//!   saturates at `65535`. Operations whose result stays within the input
//!   depth (`clamp`, ...) keep it.
//! * **Comparisons.** The relational family returns an 8-bit image with the
//!   input's band count holding `255` where the relation holds and `0`
//!   where it does not, matching libvips.
//! * **Division by zero.** `x / 0` produces `0`, matching libvips
//!   `vips_divide`.
//! * **Remainder by zero.** `x % 0` produces `0`, where libvips writes
//!   `-1`. That `-1` is not the integer branch's quirk alone:
//!   `remainder.c:101` writes it in `IREMAINDER` and `remainder.c:116`
//!   writes it again in `FREMAINDER`, so a float carrier would not change
//!   it. It is simply not representable on an unsigned carrier, so the
//!   crate keeps `0`. Both remainder forms follow this:
//!   [`Raster::remainder`] and [`Raster::rem_const`].
//! * **NaN.** A NaN result (e.g. `0.0.powf(f64::NAN)`) writes `0`.
//!
//! # Float-output operations
//!
//! The divide family ([`Raster::div`], [`Raster::div_const`],
//! [`Raster::div_vec`]), [`Raster::linear`], and image-image
//! [`Raster::sub`] produce a float raster, matching the libvips promotion
//! tables: `vips_divide` maps every integer input format to float,
//! `vips_linear` computes in float and only casts down when the caller
//! asks for it (the `uchar` option, [`Raster::linear_uchar`] here), and
//! `vips_subtract` promotes `uchar` to signed `short` — carried here as
//! float so negative differences survive (issue #282). A quotient such as
//! `128 / 255` therefore stays `0.502` instead of rounding to `1`, which
//! keeps `atanh` and the other domain-limited maths finite on scaled
//! input, and `10 - 200` stays `-190` instead of saturating to `0`.
//! Division by zero still produces `0`, matching `vips_divide`.
//!
//! The transcendental family (`vips_math`: `sin` through `atanh`, `log`,
//! `log10`, `exp`, `exp10`; `vips_math2`: `atan2`, `pow`, `wop`; `neg`) and
//! the complex family accept every sample depth, including float input,
//! and produce a float raster ([`PixelFormat::RgbaF32`] or
//! [`PixelFormat::FloatF32`]) so fractional and negative results survive,
//! matching the libvips float promotion. Following libvips `vips_math`,
//! `sin` / `cos` / `tan` take their input in degrees and `asin` / `acos` /
//! `atan` / `atan2` produce degrees. Out-of-domain inputs keep IEEE
//! semantics (`log(0)` is `-inf`, `acosh(0.5)` is NaN) rather than
//! saturating.
//!
//! A complex image is a float raster with an even band count holding
//! `(re, im)` pairs: [`Raster::complexform`] interleaves two real images,
//! [`Raster::real`] / [`Raster::imag`] extract the halves, and
//! [`Raster::polar`] / [`Raster::rect`] / [`Raster::conj`] map pairs
//! (angles in degrees, matching `vips_complex`).
//!
//! # The alpha pair on a float carrier
//!
//! [`Raster::premultiply`] and [`Raster::unpremultiply`] accept float rasters
//! as well as the unsigned ones (issue #631), which matters because
//! [`crate::exr`] and [`crate::fits`] hand back float pixel data straight from
//! a file. They keep the input format either way, so a float raster stays
//! float and is stored raw rather than rounded and saturated.
//!
//! Two things about the float arm are worth knowing before reading the code.
//! Its `max_alpha` comes from the raster's [`Interpretation`] and not from the
//! sample depth, exactly as `vips_interpretation_max_alpha` supplies it to
//! `vips_premultiply`, so an scRGB raster divides by `1.0` where an untagged
//! one divides by `255`. And its arithmetic runs in `f32`, not `f64`: the C
//! macros land the multiplier in a `float` before the colour multiply, so the
//! result rounds twice, and an `f64` expression rounded once at the store
//! differs from vips by around an ulp on ordinary values.
//!
//! `floor`, `ceil`, and `rint` round float rasters samplewise and are exact
//! identities on the integer formats, which is also what libvips produces
//! for integer input. `abs` and `sign` likewise gained float branches when
//! `neg` made negative samples possible.
//!
//! The `hist_find*` family lives in [`crate::histogram`]; the creation /
//! conversion helpers the ported statistics tests use for setup (`grey`,
//! `insert`) live in their own batches.

use crate::conversion::Interpretation;
use crate::pixel::{PixelFormat, SampleKind};
use crate::raster::{Raster, RasterError, alloc_op_output, try_plane_len_filled};
use thiserror::Error;

/// Typed errors for the arithmetic operations in [`crate::arithmetic`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ArithmeticError {
    /// Two rasters that must share pixel dimensions do not.
    #[error("dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        got_w: u32,
        got_h: u32,
    },
    /// Two rasters that must share a band count do not.
    #[error("band-count mismatch: expected {expected} bands, got {got}")]
    BandCountMismatch { expected: usize, got: usize },
    /// A per-band constant vector's length does not equal the band count.
    #[error("constant count mismatch: expected {expected} constants, got {got}")]
    ConstCountMismatch { expected: usize, got: usize },
    /// `recomb` was given an empty matrix.
    #[error("recomb requires at least one matrix row")]
    EmptyMatrix,
    /// A `recomb` matrix row's length does not equal the input band count.
    #[error("recomb matrix row {row} has {got} coefficients, expected {expected}")]
    MatrixRowMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
    /// `sum` was given an empty image list.
    #[error("sum requires at least one image")]
    EmptyImageList,
    /// `premultiply` / `unpremultiply` need at least two bands (the last
    /// band is the alpha band).
    #[error("alpha operation requires at least 2 bands, image has {bands}")]
    NoAlphaBand { bands: usize },
    /// A `stdif` window dimension is zero.
    #[error("stdif window dimensions must be greater than zero")]
    ZeroWindow,
    /// A `stdif` window dimension exceeds the image (vips rejects this as
    /// `stdif: window too large`).
    #[error("stdif window {win_w}x{win_h} is larger than image {width}x{height}")]
    WindowTooLarge {
        win_w: u32,
        win_h: u32,
        width: u32,
        height: u32,
    },
    /// A `measure` patch grid dimension is zero.
    #[error("measure patch grid dimensions must be greater than zero")]
    ZeroPatches,
    /// A `measure` patch grid has more patches than pixels along an axis.
    #[error("measure grid {across}x{down} does not fit image {width}x{height}")]
    PatchGridTooFine {
        across: u32,
        down: u32,
        width: u32,
        height: u32,
    },
    /// The result would have more bands than [`PixelFormat`] can carry.
    #[error("result band count {bands} exceeds the supported maximum of 65535")]
    TooManyBands { bands: usize },
    /// A complex operation (`polar`, `rect`, `conj`, `real`, `imag`) was
    /// given an image whose band count is odd, so it cannot hold
    /// `(re, im)` pairs.
    #[error("complex operation requires an even band count of (re, im) pairs, image has {bands}")]
    NotComplex { bands: usize },
    /// `hough_circle` was given an empty radius range.
    #[error("hough_circle radius range is empty: min {min} exceeds max {max}")]
    EmptyRadiusRange { min: u32, max: u32 },
    /// An integer-only arithmetic operation (image-image `add` / `mul`, the
    /// constant / per-band `*_const` / `*_vec` forms, the bitwise `bitand` /
    /// `bitor` / `bitxor` pair ops, `recomb` and `stdif`) was given a float
    /// raster. These operations round-and-saturate into an unsigned integer
    /// output, so a float input has no representable result; cast to an
    /// unsigned 8/16-bit format first, or use the float-output family
    /// (image-image `sub`, `div`, `linear`, the transcendental ops, and the
    /// alpha pair `premultiply` / `unpremultiply`), which reads every input
    /// depth. Image-image `sub` floats its output (libviprs#282) so it is a
    /// float-output op and accepts float input — unlike `sub_const` / `sub_vec`,
    /// which stay integer and saturate. Mirrors [`RasterError::FloatUnsupported`]
    /// so the `try_*` forms return a typed error instead of panicking.
    #[error("{op} does not support float rasters yet; cast to an unsigned 8/16-bit format first")]
    FloatUnsupported { op: &'static str },
    /// Constructing the result raster failed (allocation, size overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

// ---------------------------------------------------------------------------
// Sample-level helpers
// ---------------------------------------------------------------------------

/// Read the flat `i`-th sample as its stored bit pattern, zero-extended
/// into a `u32` (native byte order for the multi-byte kinds, matching
/// [`crate::raster_ops`]). Integer kinds only: the [`SampleKind::F32`] arm
/// panics rather than misreading float bytes as `u16` pairs, which is what
/// the arithmetic ops did before the float formats existed.
///
/// This is the *storage* read, and it is deliberately not the numeric one.
/// Every caller wants the bits: the bitwise family operates on them (as
/// libvips `boolean` does), and `profile`'s scans only ask whether a sample
/// is non-zero, which the two's-complement pattern answers correctly for
/// the signed kinds. Use [`read_f64`] where the *value* is wanted, since
/// that one sign-extends.
///
/// The match is over the kind and has no wildcard, so a carrier added to
/// [`SampleKind`] is a compile error here instead of a silent misread
/// (issue #607).
#[inline]
fn read_u32(data: &[u8], kind: SampleKind, i: usize) -> u32 {
    match kind {
        SampleKind::U8 | SampleKind::I8 => data[i] as u32,
        SampleKind::U16 | SampleKind::I16 => {
            u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as u32
        }
        SampleKind::U32 | SampleKind::I32 => u32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ]),
        SampleKind::F32 => panic!(
            "the arithmetic operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Read the flat `i`-th sample as `f64`. Unlike [`read_u32`], this reads
/// every kind including [`SampleKind::F32`] (native byte order, matching
/// [`crate::raster_ops`]), so the read-only reductions (`avg`, `deviate`,
/// `min`/`max`, `minpos`/`maxpos`) and the relational ops work on the
/// float rasters the create generators emit. The integer-writing ops
/// still go through [`write_u32`] / [`depth_max`] and keep rejecting
/// float input loudly; the float-output linear / divide family reads
/// every kind.
#[inline]
fn read_f64(data: &[u8], kind: SampleKind, i: usize) -> f64 {
    match kind {
        SampleKind::U8 => f64::from(data[i]),
        SampleKind::I8 => f64::from(data[i] as i8),
        SampleKind::U16 => f64::from(u16::from_ne_bytes([data[2 * i], data[2 * i + 1]])),
        SampleKind::I16 => f64::from(i16::from_ne_bytes([data[2 * i], data[2 * i + 1]])),
        SampleKind::U32 => f64::from(u32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ])),
        SampleKind::I32 => f64::from(i32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ])),
        SampleKind::F32 => f64::from(f32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ])),
    }
}

/// Write the flat `i`-th sample from its stored bit pattern, truncated to
/// the kind's width. `v` must already fit the kind. Integer kinds only; see
/// [`read_u32`], which is the read this inverts, including on why the match
/// has no wildcard arm.
#[inline]
fn write_u32(data: &mut [u8], kind: SampleKind, i: usize, v: u32) {
    match kind {
        SampleKind::U8 | SampleKind::I8 => data[i] = v as u8,
        SampleKind::U16 | SampleKind::I16 => {
            let b = (v as u16).to_ne_bytes();
            data[2 * i] = b[0];
            data[2 * i + 1] = b[1];
        }
        SampleKind::U32 | SampleKind::I32 => {
            let b = v.to_ne_bytes();
            data[4 * i..4 * i + 4].copy_from_slice(&b);
        }
        SampleKind::F32 => panic!(
            "the arithmetic operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Round `v` to nearest, saturate into the kind's range capped at `max`,
/// and write it as the flat `i`-th sample. NaN writes `0`.
///
/// The floor comes from [`SampleKind::range`] rather than being a literal
/// `0.0`, because zero is only the right floor for three of the six integer
/// kinds. On the unsigned carriers the crate has today the two spellings
/// are the same number, so this changes nothing; on a signed carrier the
/// old spelling would have clipped every negative result to zero, which is
/// the "samples are non-negative" assumption issue #607 names as the
/// expensive half of #516.
///
/// `max` stays a parameter because callers cap below the kind's ceiling
/// (`profile` and the hough accumulators pass `65535.0` regardless of what
/// the kind could hold).
#[inline]
fn write_f64(data: &mut [u8], kind: SampleKind, i: usize, v: f64, max: f64) {
    // A float kind has no range; `write_u32` below refuses it anyway, so
    // the floor it never reads is immaterial.
    let min = kind.range().map_or(0.0, |(lo, _)| lo as f64);
    let v = if v.is_nan() {
        0.0
    } else {
        v.round().clamp(min, max)
    };
    // Through `i64` first: `(-5.0f64) as u32` saturates to `0` in Rust,
    // where the two's-complement pattern `write_u32` wants is `0xFFFF_FFFB`.
    // Non-negative values are unaffected.
    write_u32(data, kind, i, v as i64 as u32);
}

/// Dead zone around zero alpha for the un-premultiply factor, the `0.01` of
/// `factor = fabs(alpha) < 0.01 ? 0 : max_alpha / alpha` in the `FUNPRE_*`
/// macros of `libvips/conversion/unpremultiply.c` (issue #604).
///
/// The literal is **absolute, in whatever units the alpha band carries**, and
/// is deliberately not scaled by `max`: libvips applies the same `0.01`
/// whatever `max_alpha` is. Measured on the 8.18.4 binary with a 1x1 float
/// pixel `(100, 100, 100, alpha)`, `alpha = 0.02` unpremultiplies to `5000`
/// under scRGB (`max_alpha` 1.0), `1275000` under the 255 default and
/// `327675008` under RGB16 (`max_alpha` 65535), while `alpha = 0.005` gives
/// `0` in all three. So the same absolute literal covers every libviprs
/// carrier: `0.01 / 255` of full scale on the 8-bit and float rasters and
/// `0.01 / 65535` on the 16-bit ones.
///
/// It only ever bites on a float carrier. libvips itself splits the two:
/// `UNPRE_*` tests `alpha == 0` for the integer formats, `FUNPRE_*` tests the
/// dead zone for float and double, and on an integer carrier the two agree
/// because the smallest non-zero magnitude is `1`. The guard exists because a
/// lanczos resample undershoots: an alpha that dips to `0.003`, or through
/// zero to a small negative, is ordinary at a hard transparency edge, and
/// dividing by it amplifies the colour by ~333 or flips its sign.
///
/// [`crate::resample`]'s premultiply bracket was the only caller that could
/// reach the float branch until #631; [`Raster::try_unpremultiply`] now takes
/// it too, on any float raster the caller hands it, including one loaded
/// straight out of an OpenEXR or FITS file.
pub(crate) const UNPREMULTIPLY_DEAD_ZONE: f64 = 0.01;

/// The libvips un-premultiply factor for `alpha` against a `max` sample
/// ceiling: `0` inside [`UNPREMULTIPLY_DEAD_ZONE`], `max / alpha` outside it.
///
/// The raw alpha is used, never a clipped one. That is the other half of the
/// contract and libvips is explicit about it: "Don't use clip_alpha to
/// calculate factor: we want over and undershoots on alpha and RGB to cancel"
/// (`libvips/conversion/unpremultiply.c`). Only the alpha that is *stored*
/// is clipped, to `0..=max`, and callers do that separately.
#[inline]
pub(crate) fn unpremultiply_factor(alpha: f64, max: f64) -> f64 {
    if alpha.abs() < UNPREMULTIPLY_DEAD_ZONE {
        0.0
    } else {
        max / alpha
    }
}

/// Which of the two alpha operations [`Raster::alpha_map`] is running.
///
/// libvips keeps these in two files with two sets of macros rather than one
/// parameterised kernel, and the split matters: the factor and the stored
/// alpha swap which of them sees the clipped value, and only un-premultiply
/// has a dead zone.
#[derive(Clone, Copy)]
#[repr(u8)]
enum AlphaOp {
    /// `libvips/conversion/premultiply.c`.
    Premultiply = 0,
    /// `libvips/conversion/unpremultiply.c`.
    Unpremultiply = 1,
}

impl AlphaOp {
    /// [`AlphaOp::Premultiply`] as the const-generic parameter of
    /// [`Raster::alpha_map_unsigned`] and [`Raster::alpha_map_float`].
    ///
    /// The discriminant travels as a `u8` and not as this enum because const
    /// generics on stable Rust are limited to the integral types; the two
    /// consts exist so no call site or comparison spells a bare `0` / `1`.
    const PREMULTIPLY: u8 = Self::Premultiply as u8;
    /// [`AlphaOp::Unpremultiply`] likewise; see [`AlphaOp::PREMULTIPLY`].
    const UNPREMULTIPLY: u8 = Self::Unpremultiply as u8;
}

/// The alpha ceiling an interpretation implies, transcribed from
/// `vips_interpretation_max_alpha` (`libvips/iofuncs/header.c:195`): 65535
/// for RGB16 and GREY16, 1.0 for scRGB, 255 for everything else.
///
/// This is where `vips_premultiply` and `vips_unpremultiply` get their
/// default `max_alpha` (`premultiply.c:227`, `unpremultiply.c:284`), and
/// through them `vips_affine` (`affine.c:553`) and `vips_thumbnail`
/// (`thumbnail.c:835`), which is every place libvips brackets a resample in
/// a premultiply. So it is a property of the *interpretation*, never of the
/// sample depth.
///
/// On the unsigned carriers the two agree, which is why a depth-derived
/// ceiling has served: an untagged `Gray8` / `Rgb8` / `Rgba8` resolves to
/// `Bw` / `Srgb` (255) and an untagged `Gray16` / `Rgb16` / `Rgba16` to
/// `Grey16` / `Rgb16` (65535). A float carrier has no depth-implied ceiling
/// at all, so the tag is the only thing that can say what "fully opaque"
/// means, and getting it wrong is not a rounding difference:
/// [`crate::colour`] hands back an `RgbaF32` tagged
/// [`Interpretation::ScRgb`] from `colourspace(ScRgb)` and [`crate::exr`]
/// tags an RGB OpenEXR load the same way, and those samples are
/// scene-linear around 0..1. Measured on vips 8.18.6, a `(100, 100, 100,
/// 0.5)` float pixel premultiplies to `50` under scRGB and `0.19607845`
/// under the 255 default.
///
/// Callers keep the unsigned carriers on their depth ceiling deliberately
/// (see `bracket_max_alpha` in [`crate::resample`]): routing those through
/// the tag as well would let a mis-tagged raster premultiply against a
/// ceiling its bytes cannot reach, and an untagged `Multi16` resolves to
/// `Multiband`, which would move it from 65535 to 255.
#[inline]
pub(crate) fn interpretation_max_alpha(interpretation: Interpretation) -> f64 {
    match interpretation {
        Interpretation::Rgb16 | Interpretation::Grey16 => 65535.0,
        Interpretation::ScRgb => 1.0,
        _ => 255.0,
    }
}

/// Largest sample value an unsigned sample of this kind can hold, as
/// `f64`. Unsigned kinds only; see [`depth_max_u32`].
#[inline]
fn depth_max(kind: SampleKind) -> f64 {
    f64::from(depth_max_u32(kind))
}

/// Largest sample value an unsigned sample of this kind can hold, as
/// `u32`.
///
/// Delegates to [`SampleKind::max_value`], which is the crate's one
/// exhaustive answer, and turns its `None` into the same "no float
/// rasters yet" panic [`read_u32`] raises. A carrier added to
/// [`SampleKind`] gets its ceiling there, once, rather than here.
#[inline]
fn depth_max_u32(kind: SampleKind) -> u32 {
    kind.max_value().unwrap_or_else(|| {
        panic!(
            "the arithmetic operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        )
    })
}

/// The output format for a band count and sample kind; the band count is
/// bounded by the caller except for `recomb`, which maps `None` to
/// `TooManyBands`.
fn format_for(bands: usize, kind: SampleKind) -> Result<PixelFormat, ArithmeticError> {
    PixelFormat::with_kind(bands, kind).ok_or(ArithmeticError::TooManyBands { bands })
}

/// `base ** exp` with libvips `math2` POW semantics.
///
/// Matches vips 8.18.4, whose `math2` POW guards the whole `base == 0 &&
/// exp <= 0` range to `0` rather than the IEEE / C values [`f64::powf`]
/// returns there (`0 ** 0 = 1`, `0 ** -1 = +inf`, `0 ** -0.5 = +inf`).
/// Verified with the oracle: `vips math2_const zero out pow c` yields `0`
/// for `c` in `{0, -1, -2, -0.5}`, while a positive base is untouched
/// (`2 ** -1 = 0.5`). Every other operand pair is left to [`f64::powf`], so
/// only the `base == 0, exp <= 0` quadrant changes. `wop` reuses this with
/// the operands swapped, since it is the same `math2` operation.
#[inline]
fn pow_vips(base: f64, exp: f64) -> f64 {
    if base == 0.0 && exp <= 0.0 {
        0.0
    } else {
        base.powf(exp)
    }
}

/// `a` mod `b` with libvips `remainder` semantics, shared by the image-image
/// [`Raster::try_remainder`] and the constant [`Raster::try_rem_const`] so
/// the two forms cannot drift apart.
///
/// The body is C's *truncating* `%`, with a zero divisor short-circuited to
/// `0` before the division so `0 / 0` never forms (`b == 0.0` catches `-0.0`
/// too). Rust's `%` on `f64` truncates exactly as C's does on integers, and
/// the operands here are always whole numbers read out of an integer carrier.
///
/// libvips does not pick one definition, it dispatches on format
/// (`remainder.c:101,116`): `IREMAINDER` uses C's truncating `%` for `CHAR`,
/// `UCHAR`, `SHORT`, `USHORT`, `INT` and `UINT`, while `FREMAINDER` uses
/// `a - b * floor(a / b)` for `FLOAT` and `DOUBLE`. Every carrier the crate
/// has today is an unsigned integer one, so `IREMAINDER` is the only branch
/// reachable and this kernel implements it. Both remainder forms therefore
/// match vips on the carrier they actually run on, including the negative
/// divisor [`Raster::try_rem_const`] can be handed: measured against vips
/// 8.18.4, `remainder_const` on a uchar `[7,20,30]` with `c = -3` gives
/// `[1,2,0]`, which is what this returns.
///
/// The two definitions are indistinguishable on non-negative operands, so the
/// image-image form cannot tell them apart at all (verified exhaustively over
/// all 4,294,836,225 pairs with `a` in `0..=65535` and `b` in `1..=65535`,
/// zero disagreements). Only a negative divisor separates them, and only
/// [`Raster::try_rem_const`] can supply one.
///
/// **A float carrier needs the floored branch added here**: keep this body
/// for the integer formats and add `a - b * (a / b).floor()` for the float
/// ones, exactly as `remainder.c` does. That is measured rather than guessed:
/// the same uchar `[7,20,30]` cast to `float` first gives `[-2,-1,0]`.
///
/// The zero case is the one deliberate divergence: libvips writes `-1` in
/// both branches, which an unsigned carrier cannot hold, so the crate-wide
/// `x % 0 == 0` convention wins.
#[inline]
fn remainder_vips(a: f64, b: f64) -> f64 {
    if b == 0.0 { 0.0 } else { a % b }
}

/// Error unless `a` and `b` share pixel dimensions and band count.
fn ensure_compatible(a: &Raster, b: &Raster) -> Result<(), ArithmeticError> {
    if (a.width(), a.height()) != (b.width(), b.height()) {
        return Err(ArithmeticError::DimensionMismatch {
            expected_w: a.width(),
            expected_h: a.height(),
            got_w: b.width(),
            got_h: b.height(),
        });
    }
    if a.format().channels() != b.format().channels() {
        return Err(ArithmeticError::BandCountMismatch {
            expected: a.format().channels(),
            got: b.format().channels(),
        });
    }
    Ok(())
}

/// Unwrap an arithmetic result for the panicking ported-test surface.
///
/// Most [`ArithmeticError`] variants do not name the failing op, so the panic
/// prefixes `"<op>: "` for context. [`ArithmeticError::FloatUnsupported`] is
/// the exception: it embeds the op in its own `Display` (mirroring
/// [`RasterError::FloatUnsupported`]), so prefixing it here as well would
/// double the name ("sub: sub does not support float rasters yet ...", #339).
/// That one variant is emitted verbatim; every other variant keeps the prefix.
#[inline]
#[track_caller]
fn expect_arith<T>(op: &str, r: Result<T, ArithmeticError>) -> T {
    match r {
        Ok(v) => v,
        Err(e @ ArithmeticError::FloatUnsupported { .. }) => panic!("{e}"),
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Reject float inputs on the mutating (integer-writing) `try_*` paths,
/// returning a typed error, now that [`read_f64`] itself reads floats for the
/// reductions. Without this, an op that forces an integer output depth
/// (`add_const` promotes to 16-bit) would silently round-trip a float raster
/// through `u16` instead of failing.
///
/// The fallible helpers ([`vec_map`], [`binary_map`]) propagate the returned
/// [`ArithmeticError::FloatUnsupported`] with `?`, so their public `try_*`
/// forms surface it to the caller instead of panicking (issue #271). The
/// infallible panicking helpers ([`unary_map`]) keep asserting directly.
fn reject_float_input(op: &'static str, r: &Raster) -> Result<(), ArithmeticError> {
    if r.format().is_float() {
        return Err(ArithmeticError::FloatUnsupported { op });
    }
    Ok(())
}

/// Allocate an op-output buffer for the infallible (panicking) op forms.
///
/// The fallible `try_*` forms call [`alloc_op_output`] directly and return
/// [`RasterError::AllocationFailed`] / [`RasterError::SizeOverflow`]; the
/// panicking forms have no error channel, so an output the allocator cannot
/// satisfy surfaces here as a panic — never a process abort through
/// `handle_alloc_error` (issue #280).
#[track_caller]
fn op_output_or_panic(width: u32, height: u32, format: PixelFormat) -> Vec<u8> {
    alloc_op_output(width, height, format)
        .unwrap_or_else(|e| panic!("arithmetic output allocation failed: {e}"))
}

/// The site labels this module's plane reservations carry.
///
/// A label rather than an ordinal, which is the whole point of the shared
/// probe: `raster::with_plane_cap_at` starves the one buffer a check names and
/// leaves every other reservation on the path alone. The three private
/// ceilings this module's used to be one of refused the *Nth* over-ceiling
/// request instead, so a check that read as though it named a buffer was
/// really naming a position, and told two same-sized buffers apart only by
/// accident (issue #696).
///
/// `PROJECT_COL_SUMS` and `PROJECT_ROW_SUMS` are the pair #532 has to widen.
/// They accumulate in `f64` and [`Raster::project`] then saturates the result
/// into 16-bit samples where vips emits `VIPS_FORMAT_UINT`, so when the `uint`
/// carrier lands the buffers keep these names and only what they hold moves.
mod plane {
    /// [`super::Raster::project`]'s column accumulator, one `f64` per column
    /// per band, so a wide short raster makes it up to ~8x the input.
    pub(super) const PROJECT_COL_SUMS: &str = "arithmetic.project.col_sums";
    /// [`super::Raster::project`]'s row accumulator, the same for rows.
    pub(super) const PROJECT_ROW_SUMS: &str = "arithmetic.project.row_sums";
    /// [`super::Raster::try_stdif`]'s integral image of the padded input.
    pub(super) const STDIF_INTEGRAL: &str = "arithmetic.stdif.integral";
    /// [`super::Raster::try_stdif`]'s integral image of the padded input's
    /// squares, the same size again and allocated straight after it.
    pub(super) const STDIF_INTEGRAL_SQUARES: &str = "arithmetic.stdif.integral_squares";
    /// [`super::Raster::try_hough_circle`]'s vote accumulator, `w * h * radii`
    /// `u32`s, sized by the caller's radius range rather than by the image.
    pub(super) const HOUGH_CIRCLE_ACCUMULATOR: &str = "arithmetic.hough_circle.accumulator";
}

/// Allocate a zero-filled scratch plane for an infallible (panicking) op form,
/// fallibly.
///
/// Several ops here build intermediate buffers far larger than their output:
/// the [`Raster::try_stdif`] integral images (two `f64` buffers, each ~8x a
/// Gray8 input), the [`Raster::try_hough_circle`] vote accumulator (`w * h *
/// radii` `u32`s, sized by the caller-controlled radius range) and
/// [`Raster::project`]'s two input-scaled accumulators. PR #339 made only the
/// *output* allocation fallible ([`alloc_op_output`], issue #280) and left
/// these as infallible `vec![..]`, so an over-capacity size still reached
/// `handle_alloc_error` and aborted the process (SIGABRT) before the fallible
/// output path ever ran, which is the exact remote-DoS abort #280 set out to
/// remove (issues #433 / #434 / #435).
///
/// The reservation itself is [`try_plane_len_filled`], the crate's one plane
/// funnel. This function is only the panic mapping: a `try_*` form propagates
/// [`RasterError::AllocationFailed`] and calls the funnel directly, while an
/// infallible form (e.g. [`Raster::project`], whose `(Raster, Raster)`
/// signature has no error channel) surfaces an unsatisfiable scratch as a
/// panic here, never a process abort. That mirrors how [`op_output_or_panic`]
/// guards the *output* allocation of the same forms.
///
/// This used to be `try_scratch`, a private helper with its own
/// `try_reserve_exact`, its own `SCRATCH_ALLOC_CAP` thread-local and its own
/// `with_scratch_alloc_cap` hook, one of three such copies in three modules
/// with three signatures and three different test-ceiling stories. There is
/// one now, and a check addresses it by [site label](plane) (issue #696).
#[track_caller]
fn scratch_or_panic<T: Clone>(
    site: &'static str,
    width: u32,
    height: u32,
    len: usize,
    fill: T,
) -> Vec<T> {
    try_plane_len_filled(site, width, height, len, fill)
        .unwrap_or_else(|e| panic!("arithmetic scratch allocation failed: {e}"))
}

/// Write `v` as the flat `i`-th native-endian `f32` sample.
#[inline]
fn write_f32(data: &mut [u8], i: usize, v: f64) {
    data[4 * i..4 * i + 4].copy_from_slice(&(v as f32).to_ne_bytes());
}

/// Stamp an integer arithmetic output with the *resolved* interpretation of
/// its `source` input, mirroring libvips copying the input header onto the
/// operation result.
///
/// The constant ops that widen depth — `add_const` / `mul_const` / `pow_const`
/// / `add_vec` (`unary_map` / `vec_map` with `out_kind == SampleKind::U16`),
/// and the widening binary path (`binary_map` with `widen`) — promote an 8-bit input into a
/// 16-bit container while keeping the samples numerically on the 0..255 scale
/// (the crate's promoted-container idiom). If that `Rgb16` / `Grey16`-shaped
/// buffer were left untagged, [`Raster::interpretation`] (like libvips'
/// `vips_image_guess_interpretation`) would *resolve* it to the genuine 16-bit
/// space, and a downstream [`Raster::composite2`] — which keys its 0..65535 vs
/// 0..255 scale on that resolved interpretation — would read the promoted
/// buffer on the 65535 scale, collapsing a fully-opaque promoted overlay to
/// ~0.4% of its value (silent data loss). Stamping the source interpretation
/// (`Srgb` / `Bw` / `Multiband` for an 8-bit input) keeps the promoted buffer
/// resolving to a *non*-genuine-16 space, while a genuinely 16-bit input
/// (already resolving `Rgb16` / `Grey16`) stays honoured. It is a no-op for a
/// same-depth output whose resolved interpretation already matches.
fn stamp_source_interpretation(mut out: Raster, source: &Raster) -> Raster {
    out.meta.interpretation = Some(source.interpretation());
    out
}

/// Apply `f` to every sample, producing a float raster of the same shape.
/// Accepts every input depth including float; results keep IEEE semantics
/// (no rounding, clamping, or NaN rewriting).
fn unary_map_float(r: &Raster, f: impl Fn(f64) -> f64) -> Raster {
    let fmt = r.format();
    let bands = fmt.channels();
    let in_kind = fmt.kind();
    let out_fmt = PixelFormat::with_kind(bands, SampleKind::F32)
        .expect("band count unchanged, so the float output format exists");
    let n = r.width() as usize * r.height() as usize * bands;
    let mut out = op_output_or_panic(r.width(), r.height(), out_fmt);
    let data = r.data();
    for i in 0..n {
        write_f32(&mut out, i, f(read_f64(data, in_kind, i)));
    }
    Raster::from_op_output(r.width(), r.height(), out_fmt, out)
        .expect("arithmetic output is well-formed")
}

/// Apply per-band `f(sample, band_constant)` to every sample, producing
/// a float raster. Accepts every input depth including float; see
/// [`unary_map_float`].
fn vec_map_float(
    r: &Raster,
    v: &[f64],
    f: impl Fn(f64, f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    let fmt = r.format();
    let bands = fmt.channels();
    if v.len() != bands {
        return Err(ArithmeticError::ConstCountMismatch {
            expected: bands,
            got: v.len(),
        });
    }
    let in_kind = fmt.kind();
    let out_fmt = PixelFormat::with_kind(bands, SampleKind::F32)
        .expect("band count unchanged, so the float output format exists");
    let n = r.width() as usize * r.height() as usize * bands;
    let mut out = alloc_op_output(r.width(), r.height(), out_fmt)?;
    let data = r.data();
    for i in 0..n {
        write_f32(&mut out, i, f(read_f64(data, in_kind, i), v[i % bands]));
    }
    Ok(Raster::from_op_output(r.width(), r.height(), out_fmt, out)?)
}

/// Apply `f` samplewise across two compatible images, producing a float
/// raster. Accepts every input depth including float; see
/// [`unary_map_float`].
fn binary_map_float(
    a: &Raster,
    b: &Raster,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    ensure_compatible(a, b)?;
    let (a_kind, b_kind) = (a.format().kind(), b.format().kind());
    let bands = a.format().channels();
    let out_fmt = PixelFormat::with_kind(bands, SampleKind::F32)
        .expect("band count unchanged, so the float output format exists");
    let n = a.width() as usize * a.height() as usize * bands;
    let mut out = alloc_op_output(a.width(), a.height(), out_fmt)?;
    let (a_data, b_data) = (a.data(), b.data());
    for i in 0..n {
        write_f32(
            &mut out,
            i,
            f(read_f64(a_data, a_kind, i), read_f64(b_data, b_kind, i)),
        );
    }
    Ok(Raster::from_op_output(a.width(), a.height(), out_fmt, out)?)
}

/// The pair count of a complex image, or `NotComplex` for an odd band
/// count. A complex image is a raster with an even band count holding
/// `(re, im)` pairs; see the module docs.
fn ensure_complex(r: &Raster) -> Result<usize, ArithmeticError> {
    let bands = r.format().channels();
    if !bands.is_multiple_of(2) {
        return Err(ArithmeticError::NotComplex { bands });
    }
    Ok(bands / 2)
}

/// Apply `f(re, im) -> (re', im')` to every complex pair, producing a float
/// raster with the input band count.
fn complex_map(r: &Raster, f: impl Fn(f64, f64) -> (f64, f64)) -> Result<Raster, ArithmeticError> {
    ensure_complex(r)?;
    let fmt = r.format();
    let (bands, kind) = (fmt.channels(), fmt.kind());
    let out_fmt = PixelFormat::with_kind(bands, SampleKind::F32)
        .expect("band count unchanged, so the float output format exists");
    let n = r.width() as usize * r.height() as usize * bands;
    let mut out = alloc_op_output(r.width(), r.height(), out_fmt)?;
    let data = r.data();
    for i in (0..n).step_by(2) {
        let (re, im) = f(read_f64(data, kind, i), read_f64(data, kind, i + 1));
        write_f32(&mut out, i, re);
        write_f32(&mut out, i + 1, im);
    }
    Ok(Raster::from_op_output(r.width(), r.height(), out_fmt, out)?)
}

/// Extract one half of every complex pair (`part` is 0 for real, 1 for
/// imaginary), producing a float raster with half the band count.
fn complex_get(r: &Raster, part: usize) -> Result<Raster, ArithmeticError> {
    let pairs = ensure_complex(r)?;
    let fmt = r.format();
    let (bands, kind) = (fmt.channels(), fmt.kind());
    let out_fmt = PixelFormat::with_kind(pairs, SampleKind::F32)
        .expect("pair count is at least 1 and at most half the input band count");
    let pixels = r.width() as usize * r.height() as usize;
    let mut out = alloc_op_output(r.width(), r.height(), out_fmt)?;
    let data = r.data();
    for p in 0..pixels {
        for pair in 0..pairs {
            let v = read_f64(data, kind, p * bands + 2 * pair + part);
            write_f32(&mut out, p * pairs + pair, v);
        }
    }
    Ok(Raster::from_op_output(r.width(), r.height(), out_fmt, out)?)
}

/// Apply `f` to every sample, writing a result of the same shape at
/// `out_kind` sample kind (rounded and saturated).
#[track_caller]
fn unary_map(r: &Raster, out_kind: SampleKind, f: impl Fn(f64) -> f64) -> Raster {
    assert!(
        !r.format().is_float(),
        "the arithmetic operations do not support float rasters yet; \
         cast to an unsigned 8/16-bit format first"
    );
    let fmt = r.format();
    let bands = fmt.channels();
    let in_kind = fmt.kind();
    let out_fmt = PixelFormat::with_kind(bands, out_kind)
        .expect("band count unchanged, so the output format exists");
    let n = r.width() as usize * r.height() as usize * bands;
    let max = depth_max(out_kind);
    let mut out = op_output_or_panic(r.width(), r.height(), out_fmt);
    let data = r.data();
    for i in 0..n {
        write_f64(&mut out, out_kind, i, f(read_f64(data, in_kind, i)), max);
    }
    let out = Raster::from_op_output(r.width(), r.height(), out_fmt, out)
        .expect("arithmetic output is well-formed");
    stamp_source_interpretation(out, r)
}

/// Fallible twin of [`unary_map`]: apply `f` samplewise at the `out_kind`
/// sample kind,
/// returning a typed error instead of panicking. `op` names the caller for the
/// [`ArithmeticError::FloatUnsupported`] error.
///
/// This is the shared body of the constant-arithmetic `try_*` forms
/// (`try_add_const`, `try_rem_const`, ...); their panicking twins are
/// [`unary_map`] callers via [`expect_arith`]. It rejects float input up front
/// (the integer ops round-and-saturate into an unsigned output, which a float
/// raster has no representable result for) and routes allocation through the
/// fallible [`alloc_op_output`], so an over-capacity output returns
/// [`RasterError::AllocationFailed`] rather than aborting.
fn try_unary_map(
    op: &'static str,
    r: &Raster,
    out_kind: SampleKind,
    f: impl Fn(f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    reject_float_input(op, r)?;
    let fmt = r.format();
    let bands = fmt.channels();
    let in_kind = fmt.kind();
    let out_fmt = format_for(bands, out_kind)?;
    let n = r.width() as usize * r.height() as usize * bands;
    let max = depth_max(out_kind);
    let mut out = alloc_op_output(r.width(), r.height(), out_fmt)?;
    let data = r.data();
    for i in 0..n {
        write_f64(&mut out, out_kind, i, f(read_f64(data, in_kind, i)), max);
    }
    let out = Raster::from_op_output(r.width(), r.height(), out_fmt, out)?;
    Ok(stamp_source_interpretation(out, r))
}

/// Apply integer `f` to every sample, keeping the input depth. `f` results
/// are masked into the depth by the caller-provided closure contract.
fn unary_map_u32(r: &Raster, f: impl Fn(u32) -> u32) -> Raster {
    let fmt = r.format();
    let kind = fmt.kind();
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    let mut out = op_output_or_panic(r.width(), r.height(), fmt);
    let data = r.data();
    for i in 0..n {
        write_u32(&mut out, kind, i, f(read_u32(data, kind, i)));
    }
    let out = Raster::from_op_output(r.width(), r.height(), fmt, out)
        .expect("arithmetic output is well-formed");
    stamp_source_interpretation(out, r)
}

/// Apply per-band `f(sample, band_constant)` to every sample. `op` names the
/// caller for the [`ArithmeticError::FloatUnsupported`] error.
fn vec_map(
    op: &'static str,
    r: &Raster,
    v: &[f64],
    out_kind: SampleKind,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    reject_float_input(op, r)?;
    let fmt = r.format();
    let bands = fmt.channels();
    if v.len() != bands {
        return Err(ArithmeticError::ConstCountMismatch {
            expected: bands,
            got: v.len(),
        });
    }
    let in_kind = fmt.kind();
    let out_fmt = format_for(bands, out_kind)?;
    let n = r.width() as usize * r.height() as usize * bands;
    let max = depth_max(out_kind);
    let mut out = alloc_op_output(r.width(), r.height(), out_fmt)?;
    let data = r.data();
    for i in 0..n {
        write_f64(
            &mut out,
            out_kind,
            i,
            f(read_f64(data, in_kind, i), v[i % bands]),
            max,
        );
    }
    let out = Raster::from_op_output(r.width(), r.height(), out_fmt, out)?;
    Ok(stamp_source_interpretation(out, r))
}

/// Apply `f` samplewise across two compatible images. Output depth is the
/// wider input depth, widened to 16-bit when `widen` is set. `op` names the
/// caller for the [`ArithmeticError::FloatUnsupported`] error.
fn binary_map(
    op: &'static str,
    a: &Raster,
    b: &Raster,
    widen: bool,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    reject_float_input(op, a)?;
    reject_float_input(op, b)?;
    ensure_compatible(a, b)?;
    let (a_kind, b_kind) = (a.format().kind(), b.format().kind());
    let out_kind = if widen {
        SampleKind::U16
    } else {
        a_kind.promote(b_kind)
    };
    let out_fmt = format_for(a.format().channels(), out_kind)?;
    let n = a.width() as usize * a.height() as usize * a.format().channels();
    let max = depth_max(out_kind);
    let mut out = alloc_op_output(a.width(), a.height(), out_fmt)?;
    let (a_data, b_data) = (a.data(), b.data());
    for i in 0..n {
        write_f64(
            &mut out,
            out_kind,
            i,
            f(read_f64(a_data, a_kind, i), read_f64(b_data, b_kind, i)),
            max,
        );
    }
    let out = Raster::from_op_output(a.width(), a.height(), out_fmt, out)?;
    Ok(stamp_source_interpretation(out, a))
}

/// Apply integer `f` samplewise across two compatible images, masking into
/// the wider input depth. `op` names the caller for the
/// [`ArithmeticError::FloatUnsupported`] error.
///
/// Float input is refused rather than computed, and that is the vips-faithful
/// answer rather than a shortcut: `vips_boolean` casts a float operand to
/// `VIPS_FORMAT_INT` before the bitwise op instead of operating on it.
/// Measured on 8.18.6, `vips boolean f.v f.v out.v and` over a 4-band float
/// raster gives an **`int`** output of `100 100 100 0` for input
/// `(100.5, 100.5, 100.5, 0.5)` and `-3 -3 -3 0` for
/// `(-3.75, -3.75, -3.75, -0.5)`, so the float samples are truncated toward
/// zero and the bits ANDed are the integers'. This op keeps the input depth,
/// which a float carrier has no unsigned spelling of, so it says so instead
/// of picking a cast for the caller (issue #631).
fn binary_map_u32(
    op: &'static str,
    a: &Raster,
    b: &Raster,
    f: impl Fn(u32, u32) -> u32,
) -> Result<Raster, ArithmeticError> {
    reject_float_input(op, a)?;
    reject_float_input(op, b)?;
    ensure_compatible(a, b)?;
    let (a_kind, b_kind) = (a.format().kind(), b.format().kind());
    let out_kind = a_kind.promote(b_kind);
    let mask = depth_max_u32(out_kind);
    let out_fmt = format_for(a.format().channels(), out_kind)?;
    let n = a.width() as usize * a.height() as usize * a.format().channels();
    let mut out = alloc_op_output(a.width(), a.height(), out_fmt)?;
    let (a_data, b_data) = (a.data(), b.data());
    for i in 0..n {
        let v = f(read_u32(a_data, a_kind, i), read_u32(b_data, b_kind, i)) & mask;
        write_u32(&mut out, out_kind, i, v);
    }
    let out = Raster::from_op_output(a.width(), a.height(), out_fmt, out)?;
    Ok(stamp_source_interpretation(out, a))
}

/// Samplewise relational op across two compatible images: 8-bit output with
/// `255` where the relation holds.
fn compare_map(
    a: &Raster,
    b: &Raster,
    f: impl Fn(f64, f64) -> bool,
) -> Result<Raster, ArithmeticError> {
    ensure_compatible(a, b)?;
    let (a_kind, b_kind) = (a.format().kind(), b.format().kind());
    let out_fmt = format_for(a.format().channels(), SampleKind::U8)?;
    let mut out = alloc_op_output(a.width(), a.height(), out_fmt)?;
    let (a_data, b_data) = (a.data(), b.data());
    for (i, o) in out.iter_mut().enumerate() {
        *o = if f(read_f64(a_data, a_kind, i), read_f64(b_data, b_kind, i)) {
            255
        } else {
            0
        };
    }
    Ok(Raster::from_op_output(a.width(), a.height(), out_fmt, out)?)
}

/// Samplewise relational op against a constant: 8-bit output with `255`
/// where the relation holds.
fn compare_const_map(r: &Raster, c: f64, f: impl Fn(f64, f64) -> bool) -> Raster {
    let fmt = r.format();
    let kind = fmt.kind();
    let out_fmt = PixelFormat::with_kind(fmt.channels(), SampleKind::U8)
        .expect("band count unchanged, so the output format exists");
    let mut out = op_output_or_panic(r.width(), r.height(), out_fmt);
    let data = r.data();
    for (i, o) in out.iter_mut().enumerate() {
        *o = if f(read_f64(data, kind, i), c) {
            255
        } else {
            0
        };
    }
    Raster::from_op_output(r.width(), r.height(), out_fmt, out)
        .expect("arithmetic output is well-formed")
}

mod comparand_sealed {
    pub trait Sealed {}
    impl Sealed for &super::Raster {}
    impl Sealed for f64 {}
}

/// Right-hand operand for the samplewise comparison methods
/// ([`Raster::more_than`], [`Raster::less_than`], and the rest of the
/// family). It is implemented for `&Raster` (compare against another image,
/// samplewise) and for `f64` (compare every sample against a constant), so a
/// single call surface serves both `x.less_than(&other)` and
/// `x.less_than(128.0)`. The dedicated `*_const` methods remain for callers
/// that want an unambiguous constant form.
///
/// The trait is sealed: only this crate implements it.
pub trait Comparand: comparand_sealed::Sealed {
    #[doc(hidden)]
    #[track_caller]
    fn compare_against(
        self,
        lhs: &Raster,
        label: &'static str,
        pred: fn(f64, f64) -> bool,
    ) -> Raster;
}

impl Comparand for &Raster {
    #[track_caller]
    fn compare_against(
        self,
        lhs: &Raster,
        label: &'static str,
        pred: fn(f64, f64) -> bool,
    ) -> Raster {
        expect_arith(label, compare_map(lhs, self, pred))
    }
}

impl Comparand for f64 {
    #[track_caller]
    fn compare_against(
        self,
        lhs: &Raster,
        _label: &'static str,
        pred: fn(f64, f64) -> bool,
    ) -> Raster {
        compare_const_map(lhs, self, pred)
    }
}

/// Statistical-differencing constants, the libvips `vips_stdif` defaults:
/// blend factor `a`, target mean `m0`, deviation blend `b`, target
/// deviation `s0`.
const STDIF_A: f64 = 0.5;
const STDIF_M0: f64 = 128.0;
const STDIF_B: f64 = 0.5;
const STDIF_S0: f64 = 50.0;

/// The `find_trim` difference threshold, the libvips default.
const FIND_TRIM_THRESHOLD: f64 = 10.0;

/// The `scaleimage` log-mode exponent, the libvips `vips_scale` default.
const SCALE_LOG_EXP: f64 = 0.25;

/// Hough-line accumulator dimensions, the libvips `hough_line` defaults:
/// angle bins across the width, distance bins down the height.
const HOUGH_LINE_WIDTH: u32 = 256;
const HOUGH_LINE_HEIGHT: u32 = 256;

impl Raster {
    // -----------------------------------------------------------------
    // Reductions
    // -----------------------------------------------------------------

    /// Mean of every sample in every band (libvips `avg`).
    pub fn avg(&self) -> f64 {
        let kind = self.format().kind();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        let sum: f64 = (0..n).map(|i| read_f64(data, kind, i)).sum();
        sum / n as f64
    }

    /// Sample standard deviation of every sample in every band (libvips
    /// `deviate`, using the `n - 1` denominator). A single-sample image has
    /// deviation `0`.
    pub fn deviate(&self) -> f64 {
        let kind = self.format().kind();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        if n < 2 {
            return 0.0;
        }
        let data = self.data();
        let (mut sum, mut sum2) = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = read_f64(data, kind, i);
            sum += v;
            sum2 += v * v;
        }
        (((sum2 - sum * sum / n as f64) / (n as f64 - 1.0)).max(0.0)).sqrt()
    }

    /// Smallest sample across every band (libvips `min`).
    pub fn min(&self) -> f64 {
        let kind = self.format().kind();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        (0..n)
            .map(|i| read_f64(data, kind, i))
            .fold(f64::MAX, f64::min)
    }

    /// Largest sample across every band (libvips `max`).
    pub fn max(&self) -> f64 {
        let kind = self.format().kind();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        (0..n)
            .map(|i| read_f64(data, kind, i))
            .fold(f64::MIN, f64::max)
    }

    /// Smallest sample and the position of its first occurrence in
    /// row-major scan order, as `(value, x, y)` (libvips `min` with
    /// position). For multi-band images the value is the smallest sample of
    /// any band at that pixel.
    pub fn minpos(&self) -> (f64, u32, u32) {
        self.extremum_pos(|v, best| v < best)
    }

    /// Largest sample and the position of its first occurrence in row-major
    /// scan order, as `(value, x, y)` (libvips `max` with position).
    pub fn maxpos(&self) -> (f64, u32, u32) {
        self.extremum_pos(|v, best| v > best)
    }

    /// Shared scan for [`Raster::minpos`] / [`Raster::maxpos`]: `better`
    /// decides whether a strictly better sample replaces the current best,
    /// so ties keep the first occurrence.
    fn extremum_pos(&self, better: impl Fn(f64, f64) -> bool) -> (f64, u32, u32) {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let data = self.data();
        let mut best = read_f64(data, kind, 0);
        let (mut bx, mut by) = (0u32, 0u32);
        for y in 0..h {
            for x in 0..w {
                for c in 0..bands {
                    let v = read_f64(data, kind, (y * w + x) * bands + c);
                    if better(v, best) {
                        best = v;
                        bx = x as u32;
                        by = y as u32;
                    }
                }
            }
        }
        (best, bx, by)
    }

    /// Image statistics matrix (libvips `stats`).
    ///
    /// Returns `bands + 1` rows of `[min, max, sum, sum_of_squares, mean,
    /// deviation]`: row `0` covers the whole image, row `b + 1` covers band
    /// `b`. The deviation uses the same `n - 1` denominator as
    /// [`Raster::deviate`].
    pub fn stats(&self) -> Vec<Vec<f64>> {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let pixels = self.width() as usize * self.height() as usize;
        let data = self.data();

        // Per-band accumulators: min, max, sum, sum2.
        let mut acc = vec![(f64::MAX, f64::MIN, 0.0f64, 0.0f64); bands];
        for p in 0..pixels {
            for (c, a) in acc.iter_mut().enumerate() {
                let v = read_f64(data, kind, p * bands + c);
                a.0 = a.0.min(v);
                a.1 = a.1.max(v);
                a.2 += v;
                a.3 += v * v;
            }
        }
        let row = |min: f64, max: f64, sum: f64, sum2: f64, n: f64| {
            let mean = sum / n;
            let sd = if n < 2.0 {
                0.0
            } else {
                (((sum2 - sum * sum / n) / (n - 1.0)).max(0.0)).sqrt()
            };
            vec![min, max, sum, sum2, mean, sd]
        };
        let overall = acc.iter().fold(
            (f64::MAX, f64::MIN, 0.0f64, 0.0f64),
            |(mn, mx, s, s2), a| (mn.min(a.0), mx.max(a.1), s + a.2, s2 + a.3),
        );
        let mut result = Vec::with_capacity(bands + 1);
        result.push(row(
            overall.0,
            overall.1,
            overall.2,
            overall.3,
            (pixels * bands) as f64,
        ));
        for a in &acc {
            result.push(row(a.0, a.1, a.2, a.3, pixels as f64));
        }
        result
    }

    /// Mean of each patch in a grid of `h` patches across and `v` patches
    /// down (libvips `measure`).
    ///
    /// The image is divided into `h * v` equal patches; each patch is
    /// sampled over its central 50% to avoid edge effects, matching
    /// libvips. Returns one row per patch, left-to-right then top-to-bottom,
    /// each row holding the per-band means.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ZeroPatches`] if `h` or `v` is zero, or
    /// [`ArithmeticError::PatchGridTooFine`] if the grid has more patches
    /// than pixels along either axis.
    pub fn try_measure(&self, h: u32, v: u32) -> Result<Vec<Vec<f64>>, ArithmeticError> {
        if h == 0 || v == 0 {
            return Err(ArithmeticError::ZeroPatches);
        }
        if h > self.width() || v > self.height() {
            return Err(ArithmeticError::PatchGridTooFine {
                across: h,
                down: v,
                width: self.width(),
                height: self.height(),
            });
        }
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let w = self.width() as usize;
        let (pw, ph) = ((self.width() / h) as usize, (self.height() / v) as usize);
        let (sw, sh) = ((pw / 2).max(1), (ph / 2).max(1));
        let data = self.data();
        let mut result = Vec::with_capacity((h * v) as usize);
        for j in 0..v as usize {
            for i in 0..h as usize {
                let x0 = i * pw + pw / 4;
                let y0 = j * ph + ph / 4;
                let mut sums = vec![0.0f64; bands];
                for y in y0..y0 + sh {
                    for x in x0..x0 + sw {
                        for (c, s) in sums.iter_mut().enumerate() {
                            *s += read_f64(data, kind, (y * w + x) * bands + c);
                        }
                    }
                }
                let n = (sw * sh) as f64;
                result.push(sums.into_iter().map(|s| s / n).collect());
            }
        }
        Ok(result)
    }

    /// Panicking form of [`Raster::try_measure`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_measure`].
    #[track_caller]
    pub fn measure(&self, h: u32, v: u32) -> Vec<Vec<f64>> {
        expect_arith("measure", self.try_measure(h, v))
    }

    /// Bounding box of non-background content as `(left, top, width,
    /// height)` (libvips `find_trim`).
    ///
    /// A pixel is content when any band differs from the background by more
    /// than the libvips default threshold of `10`. `background` defaults to
    /// `255` (white) in every band, the libvips default; a single-element
    /// slice broadcasts across bands. An all-background image returns
    /// `(0, 0, 0, 0)`. libvips median-filters the image before
    /// thresholding; this implementation thresholds directly, which is
    /// equivalent for noise-free rasters.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `background` has
    /// neither one element nor one per band.
    pub fn try_find_trim(
        &self,
        background: Option<&[f64]>,
    ) -> Result<(u32, u32, u32, u32), ArithmeticError> {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let bg: Vec<f64> = match background {
            None => vec![255.0; bands],
            Some(v) if v.len() == 1 => vec![v[0]; bands],
            Some(v) if v.len() == bands => v.to_vec(),
            Some(v) => {
                return Err(ArithmeticError::ConstCountMismatch {
                    expected: bands,
                    got: v.len(),
                });
            }
        };
        let (w, h) = (self.width() as usize, self.height() as usize);
        let data = self.data();
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        let mut found = false;
        for y in 0..h {
            for x in 0..w {
                let content = (0..bands).any(|c| {
                    (read_f64(data, kind, (y * w + x) * bands + c) - bg[c]).abs()
                        > FIND_TRIM_THRESHOLD
                });
                if content {
                    found = true;
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        if !found {
            return Ok((0, 0, 0, 0));
        }
        Ok((
            x0 as u32,
            y0 as u32,
            (x1 - x0 + 1) as u32,
            (y1 - y0 + 1) as u32,
        ))
    }

    /// Panicking form of [`Raster::try_find_trim`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_find_trim`].
    #[track_caller]
    pub fn find_trim(&self, background: Option<&[f64]>) -> (u32, u32, u32, u32) {
        expect_arith("find_trim", self.try_find_trim(background))
    }

    /// First non-zero sample positions (libvips `profile`).
    ///
    /// Returns `(columns, rows)`: `columns` is a `width x 1` image whose
    /// value at `x` is the row index of the first non-zero sample in column
    /// `x` (the image height when the column is all zero); `rows` is a
    /// `1 x height` image whose value at `y` is the column index of the
    /// first non-zero sample in row `y` (the image width when all zero).
    /// Both outputs are 16-bit with the input band count, positions
    /// saturating at `65535`.
    ///
    /// **That ceiling is a deviation, and the saturation is visible on any
    /// image longer than 65535 along the axis being profiled.** libvips
    /// emits `VIPS_FORMAT_INT` here, measured on 8.18.6 for every one of
    /// the eight input formats, so its positions are exact up to
    /// `i32::MAX`: on a 1x65537 all-zero image `vips profile` reports
    /// `65537` where this reports `65535`. Note the signedness. `INT` is
    /// the *signed* 32-bit carrier, so closing this gap is a payoff of the
    /// signed carriers (issue #516) and not of the uint one (issue #517),
    /// which is the opposite of what issue #532 assumes about the counter
    /// family. Values are unaffected below the ceiling.
    pub fn profile(&self) -> (Raster, Raster) {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let out_fmt = PixelFormat::with_kind(bands, SampleKind::U16)
            .expect("band count unchanged, so the output format exists");
        let data = self.data();

        let mut cols = op_output_or_panic(self.width(), 1, out_fmt);
        for x in 0..w {
            for c in 0..bands {
                let first = (0..h)
                    .find(|&y| read_u32(data, kind, (y * w + x) * bands + c) != 0)
                    .unwrap_or(h);
                write_u32(
                    &mut cols,
                    SampleKind::U16,
                    x * bands + c,
                    first.min(0xFFFF) as u32,
                );
            }
        }
        let mut rows = op_output_or_panic(1, self.height(), out_fmt);
        for y in 0..h {
            for c in 0..bands {
                let first = (0..w)
                    .find(|&x| read_u32(data, kind, (y * w + x) * bands + c) != 0)
                    .unwrap_or(w);
                write_u32(
                    &mut rows,
                    SampleKind::U16,
                    y * bands + c,
                    first.min(0xFFFF) as u32,
                );
            }
        }
        (
            Raster::from_op_output(self.width(), 1, out_fmt, cols)
                .expect("profile output is well-formed"),
            Raster::from_op_output(1, self.height(), out_fmt, rows)
                .expect("profile output is well-formed"),
        )
    }

    /// Column and row sums (libvips `project`).
    ///
    /// Returns `(columns, rows)`: `columns` is a `width x 1` image holding
    /// the per-band sum of each column; `rows` is a `1 x height` image
    /// holding the per-band sum of each row. Outputs are 16-bit and sums
    /// saturate at `65535`.
    ///
    /// **The ceiling is a deviation and it is reached by any image with
    /// more than 257 rows of full-scale 8-bit samples.** libvips promotes
    /// to a 32-bit carrier this crate does not have, and *which* one
    /// depends on the input, measured on 8.18.6: `UINT` for `uchar`,
    /// `ushort` and `uint`, `INT` for `char`, `short` and `int`, and
    /// `DOUBLE` for `float` and `double`. So matching vips here needs both
    /// carrier families and not just the uint one (issues #517 and #516).
    /// On a 1x65537 all-255 image `vips project` reports `16711935` where
    /// this reports `65535`.
    pub fn project(&self) -> (Raster, Raster) {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let out_fmt = PixelFormat::with_kind(bands, SampleKind::U16)
            .expect("band count unchanged, so the output format exists");
        let data = self.data();

        // `col_sums` / `row_sums` are input-scaled — a wide, short raster
        // makes `col_sums` up to ~8x the input (`f64` per sample), so a legal
        // large input could drive an infallible `vec![..]` into
        // `handle_alloc_error` and abort. Route them through the fallible
        // scratch path so an unsatisfiable size panics (project has no error
        // channel) rather than aborting (#460).
        let mut col_sums = scratch_or_panic(
            plane::PROJECT_COL_SUMS,
            self.width(),
            self.height(),
            w * bands,
            0.0f64,
        );
        let mut row_sums = scratch_or_panic(
            plane::PROJECT_ROW_SUMS,
            self.width(),
            self.height(),
            h * bands,
            0.0f64,
        );
        for y in 0..h {
            for x in 0..w {
                for c in 0..bands {
                    let v = read_f64(data, kind, (y * w + x) * bands + c);
                    col_sums[x * bands + c] += v;
                    row_sums[y * bands + c] += v;
                }
            }
        }
        let mut cols = op_output_or_panic(self.width(), 1, out_fmt);
        for (i, &s) in col_sums.iter().enumerate() {
            write_f64(&mut cols, SampleKind::U16, i, s, 65535.0);
        }
        let mut rows = op_output_or_panic(1, self.height(), out_fmt);
        for (i, &s) in row_sums.iter().enumerate() {
            write_f64(&mut rows, SampleKind::U16, i, s, 65535.0);
        }
        (
            Raster::from_op_output(self.width(), 1, out_fmt, cols)
                .expect("project output is well-formed"),
            Raster::from_op_output(1, self.height(), out_fmt, rows)
                .expect("project output is well-formed"),
        )
    }

    // -----------------------------------------------------------------
    // Constant arithmetic
    // -----------------------------------------------------------------

    /// Add a constant to every sample (libvips `linear` with `a = 1`).
    /// 8-bit input promotes to 16-bit so sums above 255 survive.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_add_const(&self, c: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("add_const", self, SampleKind::U16, move |v| v + c)
    }

    /// Panicking form of [`Raster::try_add_const`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_add_const`].
    #[track_caller]
    pub fn add_const(&self, c: f64) -> Raster {
        expect_arith("add_const", self.try_add_const(c))
    }

    /// Subtract a constant from every sample, saturating at `0`.
    ///
    /// This constant form keeps the integer round-and-saturate contract,
    /// matching `vips_linear`'s requested- (integer-) format output. It
    /// differs from image-image [`Raster::sub`], which floats its output and
    /// preserves negative differences (libviprs#282, matching `vips_subtract`'s
    /// promotion to signed `short`).
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_sub_const(&self, c: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("sub_const", self, self.format().kind(), move |v| v - c)
    }

    /// Panicking form of [`Raster::try_sub_const`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_sub_const`].
    #[track_caller]
    pub fn sub_const(&self, c: f64) -> Raster {
        expect_arith("sub_const", self.try_sub_const(c))
    }

    /// Multiply every sample by a constant. 8-bit input promotes to 16-bit
    /// so products above 255 survive.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_mul_const(&self, c: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("mul_const", self, SampleKind::U16, move |v| v * c)
    }

    /// Panicking form of [`Raster::try_mul_const`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_mul_const`].
    #[track_caller]
    pub fn mul_const(&self, c: f64) -> Raster {
        expect_arith("mul_const", self.try_mul_const(c))
    }

    /// Divide every sample by a constant. The output is a float raster,
    /// matching the libvips float promotion for division (`vips_divide`
    /// maps every integer format to float, and pyvips lowers `image / c`
    /// to `vips_linear`, which also floats), so quotients keep their
    /// fractional part: `128 / 255` stays `~0.502` instead of rounding
    /// to `1`. Division by zero produces `0`, matching `vips_divide`.
    /// Accepts every input depth including float.
    pub fn div_const(&self, c: f64) -> Raster {
        unary_map_float(self, move |v| if c == 0.0 { 0.0 } else { v / c })
    }

    /// Floor-divide every sample by a constant (Python `//`); division by
    /// zero produces `0`. This op keeps the integer output contract: the
    /// floored quotient of an unsigned integer sample is exactly
    /// representable at the input depth, so the sample values match the
    /// pyvips float result exactly and the integer format serves the
    /// index-building callers ([`crate::histogram`]).
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_floordiv_const(&self, c: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("floordiv_const", self, self.format().kind(), move |v| {
            if c == 0.0 { 0.0 } else { (v / c).floor() }
        })
    }

    /// Panicking form of [`Raster::try_floordiv_const`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_floordiv_const`].
    #[track_caller]
    pub fn floordiv_const(&self, c: f64) -> Raster {
        expect_arith("floordiv_const", self.try_floordiv_const(c))
    }

    /// Raise every sample to a power. 8-bit input promotes to 16-bit;
    /// results saturate at the output depth.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_pow_const(&self, exp: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("pow_const", self, SampleKind::U16, move |v| {
            pow_vips(v, exp)
        })
    }

    /// Panicking form of [`Raster::try_pow_const`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_pow_const`].
    #[track_caller]
    pub fn pow_const(&self, exp: f64) -> Raster {
        expect_arith("pow_const", self.try_pow_const(exp))
    }

    /// Remainder of every sample divided by a constant (libvips
    /// `remainder_const`); a zero divisor produces `0`.
    ///
    /// The kernel is the shared `remainder_vips`, the same truncating `%`
    /// [`Raster::try_remainder`] uses, so the constant and image-image forms
    /// cannot disagree for identical operands. `c` is an unconstrained `f64`,
    /// so unlike the image-image form this one can be handed a negative
    /// divisor, and truncating is what libvips does there for an integer
    /// input: measured against vips 8.18.4, `remainder_const` on a uchar
    /// `[7,20,30]` with `c = -3` gives `[1,2,0]`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::FloatUnsupported`] if the input is a float
    /// raster (this integer op rounds and saturates into an unsigned output).
    pub fn try_rem_const(&self, c: f64) -> Result<Raster, ArithmeticError> {
        try_unary_map("rem_const", self, self.format().kind(), move |v| {
            remainder_vips(v, c)
        })
    }

    /// Panicking form of [`Raster::try_rem_const`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_rem_const`].
    #[track_caller]
    pub fn rem_const(&self, c: f64) -> Raster {
        expect_arith("rem_const", self.try_rem_const(c))
    }

    /// `a * sample + b` for every sample (libvips `linear`). The output
    /// is a float raster, matching `vips_linear`, which computes in
    /// float and never rounds into an integer container unless the
    /// caller asks; [`Raster::linear_uchar`] is the asking form.
    /// Fractional and negative results survive. Accepts every input
    /// depth including float.
    pub fn linear(&self, a: f64, b: f64) -> Raster {
        unary_map_float(self, move |v| a * v + b)
    }

    /// `a * sample + b` clipped into `0..=255` and truncated into an
    /// 8-bit raster (libvips `linear` with the `uchar` option): the
    /// caller-requests-integer form of [`Raster::linear`]. The
    /// truncation is the C float-to-uchar cast `vips_linear1` performs,
    /// not a rounding, matching the `mask_*` uchar path in
    /// [`crate::create`]. Accepts every input depth including float, so
    /// it also casts a floated linear / divide result back to uchar.
    pub fn linear_uchar(&self, a: f64, b: f64) -> Raster {
        let fmt = self.format();
        let (bands, in_kind) = (fmt.channels(), fmt.kind());
        let out_fmt = PixelFormat::with_kind(bands, SampleKind::U8)
            .expect("band count unchanged, so the 8-bit output format exists");
        let mut out = op_output_or_panic(self.width(), self.height(), out_fmt);
        let data = self.data();
        for (i, o) in out.iter_mut().enumerate() {
            // C-style cast: clip, then truncate toward zero (NaN casts
            // to 0 in Rust, which VIPS_FCLIP's comparisons also yield).
            *o = (a * read_f64(data, in_kind, i) + b).clamp(0.0, 255.0) as u8;
        }
        Raster::from_op_output(self.width(), self.height(), out_fmt, out)
            .expect("arithmetic output is well-formed")
    }

    /// Per-band constant addition (libvips `add` with a vector constant);
    /// 8-bit input promotes to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band, or [`ArithmeticError::FloatUnsupported`] if the
    /// input is a float raster (this integer op rounds and saturates into an
    /// unsigned output).
    pub fn try_add_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map("add_vec", self, v, SampleKind::U16, |s, c| s + c)
    }

    /// Panicking form of [`Raster::try_add_vec`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_add_vec`].
    #[track_caller]
    pub fn add_vec(&self, v: &[f64]) -> Raster {
        expect_arith("add_vec", self.try_add_vec(v))
    }

    /// Per-band constant subtraction, saturating at `0`.
    ///
    /// Like [`Raster::sub_const`], this per-band form keeps the integer
    /// round-and-saturate contract (matching `vips_linear`'s requested-format
    /// output) and so differs from image-image [`Raster::sub`], which floats
    /// its output to preserve negative differences (libviprs#282).
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band, or [`ArithmeticError::FloatUnsupported`] if the
    /// input is a float raster (this integer op rounds and saturates into an
    /// unsigned output).
    pub fn try_sub_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map("sub_vec", self, v, self.format().kind(), |s, c| s - c)
    }

    /// Panicking form of [`Raster::try_sub_vec`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_sub_vec`].
    #[track_caller]
    pub fn sub_vec(&self, v: &[f64]) -> Raster {
        expect_arith("sub_vec", self.try_sub_vec(v))
    }

    /// Per-band constant multiplication; 8-bit input promotes to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band, or [`ArithmeticError::FloatUnsupported`] if the
    /// input is a float raster (this integer op rounds and saturates into an
    /// unsigned output).
    pub fn try_mul_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map("mul_vec", self, v, SampleKind::U16, |s, c| s * c)
    }

    /// Panicking form of [`Raster::try_mul_vec`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_mul_vec`].
    #[track_caller]
    pub fn mul_vec(&self, v: &[f64]) -> Raster {
        expect_arith("mul_vec", self.try_mul_vec(v))
    }

    /// Per-band constant division. Float output, matching the libvips
    /// float promotion for division (see [`Raster::div_const`]);
    /// division by zero produces `0`. Accepts every input depth
    /// including float.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band.
    pub fn try_div_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map_float(self, v, |s, c| if c == 0.0 { 0.0 } else { s / c })
    }

    /// Panicking form of [`Raster::try_div_vec`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_div_vec`].
    #[track_caller]
    pub fn div_vec(&self, v: &[f64]) -> Raster {
        expect_arith("div_vec", self.try_div_vec(v))
    }

    // -----------------------------------------------------------------
    // Unary shape / rounding ops
    // -----------------------------------------------------------------

    /// Unary plus: an identity copy (libvips and pyvips `+image`).
    pub fn pos(&self) -> Raster {
        self.clone()
    }

    /// Unary negation of every sample (libvips and pyvips `-image`). The
    /// output is a float raster so negative results survive; `abs`
    /// round-trips it back.
    pub fn neg(&self) -> Raster {
        unary_map_float(self, |v| -v)
    }

    /// Absolute value of every sample (libvips `abs`). Unsigned integer
    /// samples cannot be negative, so those formats are an identity copy;
    /// float rasters (for example a [`Raster::neg`] result) map `|v|`
    /// samplewise and stay float.
    pub fn abs(&self) -> Raster {
        if self.format().is_float() {
            unary_map_float(self, f64::abs)
        } else {
            self.clone()
        }
    }

    /// Sign of every sample (libvips `sign`): `1` for positive samples,
    /// `0` for zero, `-1` for negative samples. Unsigned integer input
    /// keeps its depth and cannot produce `-1`; float input produces a
    /// float raster (NaN maps to `0`).
    pub fn sign(&self) -> Raster {
        if self.format().is_float() {
            unary_map_float(self, |v| {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
        } else {
            unary_map_u32(self, |v| u32::from(v > 0))
        }
    }

    /// Clamp every sample into `[min, max]`; the bounds default to the
    /// libvips `clamp` defaults `0` and `1`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    #[track_caller]
    pub fn clamp(&self, min: Option<f64>, max: Option<f64>) -> Raster {
        let lo = min.unwrap_or(0.0);
        let hi = max.unwrap_or(1.0);
        assert!(lo <= hi, "clamp: min bound {lo} exceeds max bound {hi}");
        unary_map(self, self.format().kind(), move |v| v.clamp(lo, hi))
    }

    /// Round every sample down (libvips `floor`): float rasters map
    /// `v.floor()` samplewise and stay float; the integer formats are an
    /// exact identity, matching libvips for integer input.
    pub fn floor(&self) -> Raster {
        if self.format().is_float() {
            unary_map_float(self, f64::floor)
        } else {
            self.clone()
        }
    }

    /// Round every sample up (libvips `ceil`): float rasters map
    /// `v.ceil()` samplewise and stay float; the integer formats are an
    /// exact identity.
    pub fn ceil(&self) -> Raster {
        if self.format().is_float() {
            unary_map_float(self, f64::ceil)
        } else {
            self.clone()
        }
    }

    /// Round every sample to the nearest integer (libvips `rint`, which
    /// rounds halves to the nearest **even** integer — banker's rounding,
    /// matching C99 `rint` under the default rounding mode that vips 8.18.4
    /// uses): float rasters map `v.round_ties_even()` samplewise and stay
    /// float; the integer formats are an exact identity. Verified against
    /// `vips round in out rint`: `0.5 -> 0`, `1.5 -> 2`, `2.5 -> 2`,
    /// `3.5 -> 4`, `-2.5 -> -2`.
    pub fn rint(&self) -> Raster {
        if self.format().is_float() {
            unary_map_float(self, f64::round_ties_even)
        } else {
            self.clone()
        }
    }

    // -----------------------------------------------------------------
    // Image-image arithmetic
    // -----------------------------------------------------------------

    /// Subtract `other` from `self` samplewise, producing a float raster
    /// (libvips `subtract`).
    ///
    /// The output is a float raster, matching the `vips_subtract` promotion
    /// table: subtracting two `uchar` images promotes to signed `short` in
    /// libvips so negative differences survive, and a caller consumes that
    /// signed result exactly as this crate's float output. Promoting to
    /// float (rather than adding a signed integer carrier) is the
    /// proportionate fix for issue #282 — the pre-#282 integer path routed
    /// through the round-and-saturate writer and collapsed every negative
    /// difference to `0` (silent data loss). Because the output floats, this
    /// op also accepts float input, so a cast-then-subtract chain works.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_sub(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_float(self, other, |a, b| a - b)
    }

    /// Panicking form of [`Raster::try_sub`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_sub`].
    #[track_caller]
    pub fn sub(&self, other: &Raster) -> Raster {
        expect_arith("sub", self.try_sub(other))
    }

    // -----------------------------------------------------------------
    // Difference reductions
    // -----------------------------------------------------------------

    /// The maximum absolute per-sample difference between `self` and `other`
    /// (libvips `max(abs(a - b))`), as `f64`.
    ///
    /// Reads every sample of both rasters at full `f64` precision (all
    /// depths, including float), so a lossless round-trip reports exactly
    /// `0.0`. The ported foreign cells assert on this directly, for example
    /// `im.max_diff(&expected) == 0.0`. Both rasters must share pixel
    /// dimensions and band count.
    ///
    /// NaN caveat: NaN samples are unsupported. A NaN difference propagates,
    /// so any NaN-containing input yields a NaN result, matching
    /// [`Raster::try_avg_diff`].
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the rasters disagree.
    pub fn try_max_diff(&self, other: &Raster) -> Result<f64, ArithmeticError> {
        ensure_compatible(self, other)?;
        // Propagate NaN instead of dropping it: `f64::max` would silently
        // return the finite operand, disagreeing with the NaN-propagating
        // sum in `try_avg_diff` and letting a `max_diff == 0.0` assertion
        // pass over unsupported NaN input. Fold to NaN once either side is
        // NaN so the whole reduction surfaces it.
        Ok(self.diff_fold(other, 0.0, |acc, d| {
            if acc.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                acc.max(d)
            }
        }))
    }

    /// Panicking form of [`Raster::try_max_diff`], matching the ported-test
    /// surface (`im.max_diff(&other)`).
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_max_diff`].
    #[track_caller]
    pub fn max_diff(&self, other: &Raster) -> f64 {
        expect_arith("max_diff", self.try_max_diff(other))
    }

    /// The mean absolute per-sample difference between `self` and `other`
    /// (libvips `avg(abs(a - b))`), as `f64`.
    ///
    /// Reads every sample of both rasters at full `f64` precision, then
    /// divides the summed absolute differences by the sample count. The
    /// ported foreign cells assert on this for lossy round-trips, for
    /// example `im.colourspace("scrgb").avg_diff(...) < 0.02`. Both rasters
    /// must share pixel dimensions and band count.
    ///
    /// NaN caveat: NaN samples are unsupported. A NaN difference propagates
    /// through the sum, so any NaN-containing input yields a NaN result,
    /// matching [`Raster::try_max_diff`].
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the rasters disagree.
    pub fn try_avg_diff(&self, other: &Raster) -> Result<f64, ArithmeticError> {
        ensure_compatible(self, other)?;
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        if n == 0 {
            return Ok(0.0);
        }
        Ok(self.diff_fold(other, 0.0, |acc, d| acc + d) / n as f64)
    }

    /// Panicking form of [`Raster::try_avg_diff`], matching the ported-test
    /// surface (`im.avg_diff(&other)`).
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_avg_diff`].
    #[track_caller]
    pub fn avg_diff(&self, other: &Raster) -> f64 {
        expect_arith("avg_diff", self.try_avg_diff(other))
    }

    /// Fold `f` over the absolute per-sample differences of two
    /// dimension-compatible rasters, starting from `init`. Each raster is
    /// read at its own depth via [`read_f64`], so a mixed-depth pair
    /// compares numerically. The caller is responsible for having checked
    /// compatibility (`ensure_compatible`) first.
    fn diff_fold(&self, other: &Raster, init: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
        let a_kind = self.format().kind();
        let b_kind = other.format().kind();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let (a_data, b_data) = (self.data(), other.data());
        (0..n)
            .map(|i| (read_f64(a_data, a_kind, i) - read_f64(b_data, b_kind, i)).abs())
            .fold(init, f)
    }

    /// Multiply two images samplewise (libvips `multiply`); 8-bit inputs
    /// promote to 16-bit and results saturate at the output depth.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float raster
    /// (this integer op rounds and saturates into an unsigned output).
    pub fn try_mul(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map("mul", self, other, true, |a, b| a * b)
    }

    /// Panicking form of [`Raster::try_mul`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_mul`].
    #[track_caller]
    pub fn mul(&self, other: &Raster) -> Raster {
        expect_arith("mul", self.try_mul(other))
    }

    /// Divide `self` by `other` samplewise (libvips `divide`). Float
    /// output: the `vips_divide` promotion table maps every integer
    /// input format to float, so fractional quotients survive. Division
    /// by zero produces `0`, matching libvips. Accepts every input
    /// depth including float.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_div(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_float(self, other, |a, b| if b == 0.0 { 0.0 } else { a / b })
    }

    /// Panicking form of [`Raster::try_div`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_div`].
    #[track_caller]
    pub fn div(&self, other: &Raster) -> Raster {
        expect_arith("div", self.try_div(other))
    }
    /// Remainder of `self` divided by `other` samplewise (libvips
    /// `remainder`), the image-image companion to [`Raster::rem_const`].
    ///
    /// The output depth is the wider of the two input depths. libvips gets
    /// there via the identity promotion table (`vips_remainder_format_table`,
    /// `remainder.c:173-178`) applied after `vips__formatalike` has already
    /// cast both inputs to the smallest common format, which on this crate's
    /// unsigned carriers is exactly `a_kind.promote(b_kind)`.
    ///
    /// The kernel is the shared `remainder_vips`, C's truncating `%`, which
    /// [`Raster::rem_const`] runs too so the two forms cannot disagree for
    /// identical operands.
    ///
    /// libvips does not pick one definition, it dispatches on format
    /// (`remainder.c:101,116`): truncating `%` in `IREMAINDER` for the
    /// integer formats, floored `a - b * floor(a / b)` in `FREMAINDER` for
    /// `FLOAT` and `DOUBLE`. Every carrier the crate has today is an
    /// unsigned integer one, so `IREMAINDER` is the branch this operation
    /// runs and it matches vips exactly. The choice is in any case invisible
    /// here: truncated and floored agree on every non-negative operand pair,
    /// verified exhaustively over all 4,294,836,225 pairs with `a` in
    /// `0..=65535` and `b` in `1..=65535`, zero disagreements. Only a
    /// negative divisor separates them, which this form cannot produce and
    /// [`Raster::rem_const`] can. A float carrier will need the floored
    /// branch added to the kernel, and the kernel says so where it is
    /// defined.
    ///
    /// # Divergences from libvips
    ///
    /// Three, all deliberate:
    ///
    /// * **A zero divisor produces `0`, where libvips produces `-1`.**
    ///   `remainder.c:101` writes `q[x] = p2[x] ? p1[x] % p2[x] : -1;` in
    ///   `IREMAINDER`, and `remainder.c:116` writes the same `-1` in
    ///   `FREMAINDER`, so this is not an integer-branch quirk that a float
    ///   carrier would resolve. On a uchar carrier that `-1` reads back as
    ///   `255` (measured against vips 8.18.4: divisor `[[0,7,0],[7,0,7]]`
    ///   gives `[255,6,255,5,255,4]`). libviprs has no signed carrier, so
    ///   `-1` is not representable here at all, and `x % 0 == 0` is the
    ///   crate-wide convention the module header already states under
    ///   "Remainder by zero" and [`Raster::rem_const`] already follows.
    /// * **No band broadcast and no size alignment.** libvips runs
    ///   `bandalike` (a 1-band operand repeated across an n-band one) and
    ///   then `sizealike` (the smaller image zero-padded up to the larger)
    ///   before the kernel sees anything. Here the two rasters must agree
    ///   exactly on width, height, and band count, which is how every other
    ///   image-image operation in this module behaves. Matching the family
    ///   matters more than matching libvips on this point. ([`Raster::add`]
    ///   is the crate's one broadcasting operation and is not the model for
    ///   this family.)
    /// * **Unsigned carriers only.** A float raster on either side is
    ///   rejected with [`ArithmeticError::FloatUnsupported`], so libvips's
    ///   integer/float kernel split collapses to a single branch here. The
    ///   reason is the output, not the input: this is one of the integer
    ///   round-and-saturate operations, so the result is an unsigned integer
    ///   raster with no representable place for a fractional or negative
    ///   sample, and rejecting float up front through `reject_float_input`
    ///   is what that whole family does. [`Raster::rem_const`], the sibling
    ///   this operation is the companion to, carries the same restriction
    ///   for the same reason. Cast to an unsigned 8- or 16-bit format first.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float raster
    /// (this integer op rounds and saturates into an unsigned output).
    pub fn try_remainder(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map("remainder", self, other, false, remainder_vips)
    }

    /// Panicking form of [`Raster::try_remainder`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_remainder`].
    #[track_caller]
    pub fn remainder(&self, other: &Raster) -> Raster {
        expect_arith("remainder", self.try_remainder(other))
    }

    /// Samplewise minimum of two images (libvips `minpair`); mixed depths
    /// promote numerically to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float raster
    /// (this integer op rounds and saturates into an unsigned output).
    pub fn try_minpair(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map("minpair", self, other, false, f64::min)
    }

    /// Panicking form of [`Raster::try_minpair`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_minpair`].
    #[track_caller]
    pub fn minpair(&self, other: &Raster) -> Raster {
        expect_arith("minpair", self.try_minpair(other))
    }

    /// Samplewise maximum of two images (libvips `maxpair`); mixed depths
    /// promote numerically to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float raster
    /// (this integer op rounds and saturates into an unsigned output).
    pub fn try_maxpair(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map("maxpair", self, other, false, f64::max)
    }

    /// Panicking form of [`Raster::try_maxpair`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_maxpair`].
    #[track_caller]
    pub fn maxpair(&self, other: &Raster) -> Raster {
        expect_arith("maxpair", self.try_maxpair(other))
    }

    /// Sum a list of images samplewise (libvips `sum`). The output is
    /// 16-bit so 8-bit sums survive; totals saturate at `65535`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::EmptyImageList`] for an empty slice, or
    /// [`ArithmeticError::DimensionMismatch`] /
    /// [`ArithmeticError::BandCountMismatch`] if any image disagrees with
    /// the first.
    pub fn try_sum(images: &[&Raster]) -> Result<Raster, ArithmeticError> {
        let first = *images.first().ok_or(ArithmeticError::EmptyImageList)?;
        for r in &images[1..] {
            ensure_compatible(first, r)?;
        }
        let bands = first.format().channels();
        let out_fmt = format_for(bands, SampleKind::U16)?;
        let n = first.width() as usize * first.height() as usize * bands;
        let mut out = alloc_op_output(first.width(), first.height(), out_fmt)?;
        for i in 0..n {
            let total: f64 = images
                .iter()
                .map(|r| read_f64(r.data(), r.format().kind(), i))
                .sum();
            write_f64(&mut out, SampleKind::U16, i, total, 65535.0);
        }
        Ok(Raster::from_op_output(
            first.width(),
            first.height(),
            out_fmt,
            out,
        )?)
    }

    /// Panicking form of [`Raster::try_sum`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_sum`].
    #[track_caller]
    pub fn sum(images: &[&Raster]) -> Raster {
        expect_arith("sum", Self::try_sum(images))
    }

    // -----------------------------------------------------------------
    // Comparisons (0 / 255 uchar masks)
    // -----------------------------------------------------------------

    /// Samplewise `self > other` as a `0` / `255` 8-bit mask (libvips
    /// `relational` MORE).
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_more_than(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a > b)
    }

    /// Samplewise `self > other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` (compared samplewise) or an `f64` constant
    /// (compared against every sample); see [`Comparand`].
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_more_than`]. A constant operand never fails.
    #[track_caller]
    pub fn more_than(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "more_than", |a, b| a > b)
    }

    /// Samplewise `self > c` as a `0` / `255` 8-bit mask.
    pub fn more_than_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a > b)
    }

    /// Samplewise `self >= other` as a `0` / `255` 8-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_more_eq(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a >= b)
    }

    /// Samplewise `self >= other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` or an `f64` constant; see [`Comparand`].
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_more_eq`]. A constant operand never fails.
    #[track_caller]
    pub fn more_eq(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "more_eq", |a, b| a >= b)
    }

    /// Samplewise `self >= c` as a `0` / `255` 8-bit mask.
    pub fn more_eq_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a >= b)
    }

    /// Samplewise `self < other` as a `0` / `255` 8-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_less_than(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a < b)
    }

    /// Samplewise `self < other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` or an `f64` constant; see [`Comparand`].
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_less_than`]. A constant operand never fails.
    #[track_caller]
    pub fn less_than(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "less_than", |a, b| a < b)
    }

    /// Samplewise `self < c` as a `0` / `255` 8-bit mask.
    pub fn less_than_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a < b)
    }

    /// Samplewise `self <= other` as a `0` / `255` 8-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_less_eq(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a <= b)
    }

    /// Samplewise `self <= other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` or an `f64` constant; see [`Comparand`].
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_less_eq`]. A constant operand never fails.
    #[track_caller]
    pub fn less_eq(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "less_eq", |a, b| a <= b)
    }

    /// Samplewise `self <= c` as a `0` / `255` 8-bit mask.
    pub fn less_eq_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a <= b)
    }

    /// Samplewise `self == other` as a `0` / `255` 8-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_equal(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a == b)
    }

    /// Samplewise `self == other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` or an `f64` constant; see [`Comparand`]. The
    /// comparison is exact, so a fractional constant matches no integer sample.
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_equal`]. A constant operand never fails.
    #[track_caller]
    pub fn equal(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "equal", |a, b| a == b)
    }

    /// Samplewise `self == c` as a `0` / `255` 8-bit mask. The comparison
    /// is exact, so a fractional constant matches no integer sample.
    pub fn equal_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a == b)
    }

    /// Samplewise `self != other` as a `0` / `255` 8-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_noteq(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        compare_map(self, other, |a, b| a != b)
    }

    /// Samplewise `self != other` as a `0` / `255` 8-bit mask. The operand is
    /// either another `&Raster` or an `f64` constant; see [`Comparand`].
    ///
    /// # Panics
    ///
    /// With a `&Raster` operand, panics on any [`ArithmeticError`]; see
    /// [`Raster::try_noteq`]. A constant operand never fails.
    #[track_caller]
    pub fn noteq(&self, other: impl Comparand) -> Raster {
        other.compare_against(self, "noteq", |a, b| a != b)
    }

    /// Samplewise `self != c` as a `0` / `255` 8-bit mask.
    pub fn noteq_const(&self, c: f64) -> Raster {
        compare_const_map(self, c, |a, b| a != b)
    }

    // -----------------------------------------------------------------
    // Bitwise operations
    // -----------------------------------------------------------------

    /// Samplewise bitwise AND of two images (libvips `boolean` AND); mixed
    /// depths promote to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float
    /// raster (issue #631): `vips_boolean` casts float to `int` rather than
    /// operating on it, so there is no float answer to give and this keeps
    /// the input depth. Cast to an unsigned 8/16-bit format first.
    pub fn try_bitand(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32("bitand", self, other, |a, b| a & b)
    }

    /// Panicking form of [`Raster::try_bitand`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_bitand`].
    #[track_caller]
    pub fn bitand(&self, other: &Raster) -> Raster {
        expect_arith("bitand", self.try_bitand(other))
    }

    /// Samplewise boolean AND of two images, an alias for [`Raster::bitand`]
    /// under the libvips `image & image` spelling. On `0` / `255` masks it
    /// composes the relational ops into a range test, e.g.
    /// `x.more_eq(64.0).band_and(&x.less_than(128.0))`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_bitand`].
    #[track_caller]
    pub fn band_and(&self, other: &Raster) -> Raster {
        expect_arith("band_and", self.try_bitand(other))
    }

    /// Bitwise AND of every sample with a constant. The constant is masked
    /// into the sample depth (two's complement, so `-1` is all ones).
    pub fn bitand_const(&self, c: i64) -> Raster {
        let mask = (c as u64 & depth_max_u32(self.format().kind()) as u64) as u32;
        unary_map_u32(self, move |v| v & mask)
    }

    /// Samplewise bitwise OR of two images.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float
    /// raster; see [`Raster::try_bitand`] (issue #631).
    pub fn try_bitor(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32("bitor", self, other, |a, b| a | b)
    }

    /// Panicking form of [`Raster::try_bitor`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_bitor`].
    #[track_caller]
    pub fn bitor(&self, other: &Raster) -> Raster {
        expect_arith("bitor", self.try_bitor(other))
    }

    /// Bitwise OR of every sample with a constant, masked into the sample
    /// depth.
    pub fn bitor_const(&self, c: i64) -> Raster {
        let mask = (c as u64 & depth_max_u32(self.format().kind()) as u64) as u32;
        unary_map_u32(self, move |v| v | mask)
    }

    /// Samplewise bitwise XOR of two images.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::FloatUnsupported`] if either input is a float
    /// raster; see [`Raster::try_bitand`] (issue #631).
    pub fn try_bitxor(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32("bitxor", self, other, |a, b| a ^ b)
    }

    /// Panicking form of [`Raster::try_bitxor`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_bitxor`].
    #[track_caller]
    pub fn bitxor(&self, other: &Raster) -> Raster {
        expect_arith("bitxor", self.try_bitxor(other))
    }

    /// Bitwise XOR of every sample with a constant, masked into the sample
    /// depth.
    pub fn bitxor_const(&self, c: i64) -> Raster {
        let mask = (c as u64 & depth_max_u32(self.format().kind()) as u64) as u32;
        unary_map_u32(self, move |v| v ^ mask)
    }

    /// Bitwise NOT of every sample within its depth (libvips `invert` for
    /// integer formats): `!v & 0xFF` for 8-bit, `!v & 0xFFFF` for 16-bit.
    pub fn bitnot(&self) -> Raster {
        let mask = depth_max_u32(self.format().kind());
        unary_map_u32(self, move |v| !v & mask)
    }

    /// Shift every sample left by `n` bits, truncating into the sample
    /// depth (the same wrap-in-format behavior as the libvips integer
    /// path). Shifts of the full sample width or more produce `0`.
    pub fn lshift(&self, n: u32) -> Raster {
        let mask = depth_max_u32(self.format().kind());
        unary_map_u32(self, move |v| v.checked_shl(n).unwrap_or(0) & mask)
    }

    /// Shift every sample right by `n` bits. Shifts of the full sample
    /// width or more produce `0`.
    pub fn rshift(&self, n: u32) -> Raster {
        unary_map_u32(self, move |v| v.checked_shr(n).unwrap_or(0))
    }

    // -----------------------------------------------------------------
    // Enhancement / recombination
    // -----------------------------------------------------------------

    /// Scale samples to fill `0..=255` (libvips `scale`). The output is
    /// 8-bit with the input band count.
    ///
    /// The default maps the global minimum to `0` and the global maximum
    /// to `255` linearly; a constant image maps to all zeros.
    /// `log = Some(true)` uses the libvips log-scaling curve
    /// `255 / log10(1 + max^0.25) * log10(1 + v^0.25)` instead.
    pub fn scaleimage(&self, log: Option<bool>) -> Raster {
        if log == Some(true) {
            let mx = self.max();
            let denom = (1.0 + mx.powf(SCALE_LOG_EXP)).log10();
            let f = if denom > 0.0 { 255.0 / denom } else { 0.0 };
            unary_map(self, SampleKind::U8, move |v| {
                f * (1.0 + v.powf(SCALE_LOG_EXP)).log10()
            })
        } else {
            let (mn, mx) = (self.min(), self.max());
            let range = mx - mn;
            if range == 0.0 {
                unary_map(self, SampleKind::U8, |_| 0.0)
            } else {
                unary_map(self, SampleKind::U8, move |v| (v - mn) * 255.0 / range)
            }
        }
    }

    /// Statistical differencing over a `width x height` window (libvips
    /// `stdif` with its default parameters).
    ///
    /// Each sample is adjusted toward a target mean of `128` and a target
    /// deviation of `50` relative to the statistics of the window centred
    /// on it: `out = a*m0 + (1-a)*mean + (in - mean) * b*s0 / (b*dev + s0)`
    /// with `a = 0.5, m0 = 128, b = 0.5, s0 = 50`. Bands are processed
    /// independently; at the edges the window is edge-replicated
    /// (`VIPS_EXTEND_COPY`, extending the border pixel) to match vips exactly,
    /// with no remaining border divergence. The output keeps the input format.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ZeroWindow`] if either window dimension
    /// is zero, or [`ArithmeticError::WindowTooLarge`] if a window dimension
    /// exceeds the corresponding image dimension (vips: `stdif: window too
    /// large`), or [`ArithmeticError::FloatUnsupported`] on a float raster
    /// (issue #631). The float refusal is stricter than this op's own
    /// carriers and looser than vips': measured on 8.18.6, `vips stdif`
    /// answers `stdif: image must be VIPS_FORMAT_UCHAR` for **both** a float
    /// and a `ushort` input, so it takes `uchar` alone where this takes
    /// `uchar` and `ushort` (a pre-existing divergence, unchanged here) and
    /// refuses float the way vips does.
    pub fn try_stdif(&self, width: u32, height: u32) -> Result<Raster, ArithmeticError> {
        // Before the window guards, so the diagnostic names the carrier
        // rather than the geometry: a float raster used to reach
        // `depth_max`'s panic from inside this fallible form (#631).
        reject_float_input("stdif", self)?;
        if width == 0 || height == 0 {
            return Err(ArithmeticError::ZeroWindow);
        }
        // vips rejects a window larger than the image ("stdif: window too
        // large"): with a 5-wide image, window 5 is accepted but 6+ errors
        // (verified with the oracle). Mirror that instead of silently
        // computing a result the differential harness could never obtain
        // from vips.
        if width > self.width() || height > self.height() {
            return Err(ArithmeticError::WindowTooLarge {
                win_w: width,
                win_h: height,
                width: self.width(),
                height: self.height(),
            });
        }
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let max = depth_max(kind);
        let data = self.data();
        let mut out = alloc_op_output(self.width(), self.height(), fmt)?;

        // Integral images per band over an *edge-replicated* padding of the
        // plane. vips embeds the input with a `window/2` border before taking
        // window statistics, and that border extends the edge pixel (verified
        // with the oracle: `vips stdif` on `[3,200,17,...]` with a 5-wide
        // window matches replicate, not mirror/reflect — see #490). Clipping
        // the window at the image edge (the old behaviour) shrank `npel` and
        // diverged from vips in a `window/2` border; replicating instead keeps
        // a full `width*height` window everywhere and matches vips exactly.
        //
        // The padded plane is `(w + width - 1) x (h + height - 1)`: `width/2`
        // extra columns on the left plus `width - 1 - width/2` on the right
        // (and likewise for rows), so every output window fits without
        // clipping. `s[y][x]` holds the sum over the padded rectangle
        // `[0, x) x [0, y)`, so any window sum is four lookups. These two f64
        // buffers dwarf the output, so they allocate fallibly — an
        // over-capacity size returns a typed error rather than aborting
        // through `handle_alloc_error` (#435).
        let (hw, hh) = (width as usize / 2, height as usize / 2);
        let pw = w + width as usize - 1;
        let ph = h + height as usize - 1;
        let stride = pw + 1;
        let scratch_len = stride
            .checked_mul(ph + 1)
            .ok_or(RasterError::SizeOverflow {
                width: self.width(),
                height: self.height(),
                bpp: 8,
            })?;
        let mut s = try_plane_len_filled(
            plane::STDIF_INTEGRAL,
            self.width(),
            self.height(),
            scratch_len,
            0.0f64,
        )?;
        let mut s2 = try_plane_len_filled(
            plane::STDIF_INTEGRAL_SQUARES,
            self.width(),
            self.height(),
            scratch_len,
            0.0f64,
        )?;
        // Map a (possibly out-of-range) coordinate onto the nearest edge
        // pixel — vips `EXTEND_COPY` / replicate border semantics.
        let clamp_edge = |i: i64, n: usize| -> usize {
            if i < 0 {
                0
            } else if i as usize >= n {
                n - 1
            } else {
                i as usize
            }
        };
        for band in 0..bands {
            for py in 0..ph {
                let sy = clamp_edge(py as i64 - hh as i64, h);
                for px in 0..pw {
                    let sx = clamp_edge(px as i64 - hw as i64, w);
                    let v = read_f64(data, kind, (sy * w + sx) * bands + band);
                    let i = (py + 1) * stride + (px + 1);
                    s[i] = v + s[i - 1] + s[i - stride] - s[i - stride - 1];
                    s2[i] = v * v + s2[i - 1] + s2[i - stride] - s2[i - stride - 1];
                }
            }
            let npel = (width as usize * height as usize) as f64;
            for y in 0..h {
                // The window for output row `y` spans padded rows [y, y+height).
                let (y0, y1) = (y, y + height as usize);
                for x in 0..w {
                    // ...and padded columns [x, x+width): a full, unclipped window.
                    let (x0, x1) = (x, x + width as usize);
                    let win = |t: &[f64]| {
                        t[y1 * stride + x1] - t[y0 * stride + x1] - t[y1 * stride + x0]
                            + t[y0 * stride + x0]
                    };
                    let mean = win(&s) / npel;
                    let var = (win(&s2) / npel - mean * mean).max(0.0);
                    let dev = var.sqrt();
                    let v = read_f64(data, kind, (y * w + x) * bands + band);
                    let res = STDIF_A * STDIF_M0
                        + (1.0 - STDIF_A) * mean
                        + (v - mean) * (STDIF_B * STDIF_S0) / (STDIF_B * dev + STDIF_S0);
                    write_f64(&mut out, kind, (y * w + x) * bands + band, res, max);
                }
            }
        }
        Ok(Raster::from_op_output(
            self.width(),
            self.height(),
            fmt,
            out,
        )?)
    }

    /// Panicking form of [`Raster::try_stdif`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_stdif`].
    #[track_caller]
    pub fn stdif(&self, width: u32, height: u32) -> Raster {
        expect_arith("stdif", self.try_stdif(width, height))
    }

    /// Recombine bands with a matrix (libvips `recomb`).
    ///
    /// `matrix` has one row per output band and one coefficient per input
    /// band: output band `r` is `sum(matrix[r][b] * in[b])`, rounded and
    /// saturated into the input depth.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::EmptyMatrix`] for an empty matrix,
    /// [`ArithmeticError::MatrixRowMismatch`] if any row does not have one
    /// coefficient per input band, [`ArithmeticError::TooManyBands`] if
    /// the output band count exceeds `u16::MAX`, or
    /// [`ArithmeticError::FloatUnsupported`] on a float raster (issue #631).
    ///
    /// # Divergence from stock libvips
    ///
    /// vips does compute `recomb` on a float image and keeps it float:
    /// measured on 8.18.6, an identity matrix over a 4-band float raster
    /// returns `100.5 100.5 100.5 0.5` in `float`. This port writes through
    /// `write_u32` into the *input* depth, which a float carrier has no
    /// unsigned spelling of, so it returns the typed refusal rather than
    /// inventing an output format. That is a narrower surface than vips',
    /// deliberately, and it replaces a panic from inside a `try_` form.
    pub fn try_recomb(&self, matrix: &[&[f64]]) -> Result<Raster, ArithmeticError> {
        reject_float_input("recomb", self)?;
        if matrix.is_empty() {
            return Err(ArithmeticError::EmptyMatrix);
        }
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        for (row, r) in matrix.iter().enumerate() {
            if r.len() != bands {
                return Err(ArithmeticError::MatrixRowMismatch {
                    row,
                    expected: bands,
                    got: r.len(),
                });
            }
        }
        let out_bands = matrix.len();
        let out_fmt = format_for(out_bands, kind)?;
        let pixels = self.width() as usize * self.height() as usize;
        let max = depth_max(kind);
        let data = self.data();
        let mut out = alloc_op_output(self.width(), self.height(), out_fmt)?;
        for p in 0..pixels {
            for (r, coeffs) in matrix.iter().enumerate() {
                let acc: f64 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(b, &m)| m * read_f64(data, kind, p * bands + b))
                    .sum();
                // vips recomb accumulates in double but STORES the result as
                // float32 (VIPS_FORMAT_FLOAT); the differential comparison then
                // casts that float32 into the input depth, and vips's
                // float->integer cast truncates toward zero (verified with the
                // oracle: `vips cast` of 2.6 -> 2, 3.5 -> 3, 0.5 -> 0). We must
                // round the f64 accumulator to f32 *before* truncating, because
                // an accumulator just below an integer (e.g. 6.99999988 for
                // input [10,20,30] with coeff row [0,0,0.23333332933333335])
                // rounds up to 7.0 in f32 storage, and `vips recomb` + `vips
                // cast`->uchar yields 7 there, not the 6 a direct f64 truncation
                // would give. Rounding to nearest here diverged by up to 1 LSB,
                // so truncate the f32-clamped value instead. NaN -> 0.
                let f = acc as f32;
                let v = if f.is_nan() {
                    0.0
                } else {
                    f.clamp(0.0, max as f32)
                };
                write_u32(&mut out, kind, p * out_bands + r, v as u32);
            }
        }
        Ok(Raster::from_op_output(
            self.width(),
            self.height(),
            out_fmt,
            out,
        )?)
    }

    /// Panicking form of [`Raster::try_recomb`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_recomb`].
    #[track_caller]
    pub fn recomb(&self, matrix: &[&[f64]]) -> Raster {
        expect_arith("recomb", self.try_recomb(matrix))
    }

    /// Premultiply the colour bands by the alpha band (libvips
    /// `premultiply`).
    ///
    /// The last band is the alpha band; every other band becomes
    /// `v * clip(alpha) / max_alpha`. The alpha band and the format are
    /// unchanged. On the unsigned carriers the result is rounded to nearest
    /// and saturated into the input depth; on the float carriers it is
    /// stored raw, which is what libvips writes there.
    ///
    /// `max_alpha` comes from the raster's [`Interpretation`], the way
    /// `vips_interpretation_max_alpha` supplies it: `65535` for
    /// [`Interpretation::Rgb16`] / [`Interpretation::Grey16`], `1.0` for
    /// [`Interpretation::ScRgb`], `255` otherwise. For the unsigned carriers
    /// that is the depth ceiling and nothing changes. For a float carrier it
    /// is the only thing that can say what "fully opaque" means, so an
    /// OpenEXR RGB load (tagged [`Interpretation::ScRgb`] by [`crate::exr`])
    /// divides by `1.0` and not by `255`. See `interpretation_max_alpha`
    /// in this module.
    ///
    /// There is no dead zone here to match [`Raster::try_unpremultiply`]'s,
    /// and that asymmetry is libvips' rather than an omission (issue #604):
    /// premultiply *multiplies* by the alpha, so a near-zero alpha damps the
    /// colour instead of amplifying it, and there is no division to produce an
    /// infinity. `libvips/conversion/premultiply.c` has one macro for every
    /// band format, with no `fabs(alpha) < 0.01` float variant at all.
    ///
    /// What it does have is the mirror image of un-premultiply's clip: the
    /// *factor* is built from `VIPS_CLIP(0, alpha, max_alpha)` while the alpha
    /// that is *stored* is the raw one, exactly the other way round from
    /// un-premultiply, so the pair cancels on a round trip. On the unsigned
    /// carriers that clip can never fire; on a float carrier it does, and both
    /// halves of it are observable (issue #631).
    ///
    /// # Divergence from stock libvips
    ///
    /// `vips_premultiply` always writes `FLOAT` output, even for `uchar`
    /// input. Here the input format survives, so the unsigned carriers stay
    /// unsigned and round. That is the crate's standing integer contract (see
    /// the module docs) and it is unchanged by #631; only the float carriers,
    /// where the two agree, are new.
    ///
    /// # The output carries the input's interpretation
    ///
    /// On **every** carrier, not only the float ones, the result is stamped
    /// with the input's resolved [`Interpretation`], because that is what
    /// `vips_premultiply` and `vips_unpremultiply` do when they copy the
    /// input header. Measured on vips 8.18.6, a `1x1 ushort, 4 bands, srgb`
    /// input comes back `srgb` from both ops and a `multiband` one comes back
    /// `multiband`.
    ///
    /// It is a visible change on the unsigned path (issue #631): an `Rgba16`
    /// tagged [`Interpretation::Srgb`] used to come back resolving to
    /// [`Interpretation::Rgb16`], since an untagged 4-band 16-bit buffer
    /// resolves to the genuine 16-bit space. It matters downstream, because
    /// [`Raster::composite2`] keys its 0..255 against 0..65535 scale on the
    /// resolved interpretation.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NoAlphaBand`] if the image has fewer
    /// than two bands. Float rasters are **not** an error: they used to reach
    /// `depth_max`'s panic arm through this method, which is exactly what a
    /// `try_` form must not do, and they now take the float arm instead
    /// (issue #631).
    pub fn try_premultiply(&self) -> Result<Raster, ArithmeticError> {
        self.alpha_map(AlphaOp::Premultiply)
    }

    /// Panicking form of [`Raster::try_premultiply`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_premultiply`].
    /// In practice that is [`ArithmeticError::NoAlphaBand`] and the
    /// allocation failures behind [`ArithmeticError::Raster`]. A **float
    /// raster is not one of them** any more: it used to take the whole
    /// process down from inside the fallible form, which is what #631
    /// fixed, and both forms now compute it.
    #[track_caller]
    pub fn premultiply(&self) -> Raster {
        expect_arith("premultiply", self.try_premultiply())
    }

    /// Undo alpha premultiplication (libvips `unpremultiply`).
    ///
    /// The last band is the alpha band; every other band is multiplied by
    /// the shared un-premultiply factor (`unpremultiply_factor`), so it
    /// becomes `v * max_alpha / alpha` outside the `0.01` dead zone around
    /// zero alpha and `0` inside it, while the stored alpha clips to
    /// `0..=max_alpha`. The format is unchanged: the unsigned carriers round
    /// and saturate into their depth, the float carriers store the raw
    /// result. `max_alpha` comes from the [`Interpretation`], as it does for
    /// [`Raster::try_premultiply`].
    ///
    /// Which carrier this runs on decides whether either guard does anything,
    /// and libvips splits the macros for the same reason. On an unsigned 8- or
    /// 16-bit raster the smallest non-zero alpha magnitude is `1`, so the dead
    /// zone collapses to `alpha == 0` (libvips takes the `UNPRE_*` integer
    /// macros there), and the stored alpha is inside `0..=max` by
    /// construction, so `VIPS_CLIP(0, alpha, max_alpha)` has nothing to do.
    /// On a float raster both are live and observable, through the `FUNPRE_*`
    /// macros: a lanczos undershoot at a hard transparency edge routinely
    /// lands alpha in `(0, 0.01)` or just below zero, and an OpenEXR or FITS
    /// file can hand back an alpha of any magnitude at all, NaN and the
    /// infinities included. That is the case [`crate::resample`]'s premultiply
    /// bracket has always hit; since #631 it is reachable through this method
    /// too, instead of panicking.
    ///
    /// # Divergence from stock libvips
    ///
    /// `vips_unpremultiply` always writes `FLOAT` output and never saturates,
    /// so an unsigned raster whose colour exceeds its alpha comes back from
    /// vips as a number above the depth ceiling and from here clamped to it.
    /// The float carriers do not saturate and agree with vips exactly.
    ///
    /// # The output carries the input's interpretation
    ///
    /// On **every** carrier, not only the float ones, the result is stamped
    /// with the input's resolved [`Interpretation`], because that is what
    /// `vips_premultiply` and `vips_unpremultiply` do when they copy the
    /// input header. Measured on vips 8.18.6, a `1x1 ushort, 4 bands, srgb`
    /// input comes back `srgb` from both ops and a `multiband` one comes back
    /// `multiband`.
    ///
    /// It is a visible change on the unsigned path (issue #631): an `Rgba16`
    /// tagged [`Interpretation::Srgb`] used to come back resolving to
    /// [`Interpretation::Rgb16`], since an untagged 4-band 16-bit buffer
    /// resolves to the genuine 16-bit space. It matters downstream, because
    /// [`Raster::composite2`] keys its 0..255 against 0..65535 scale on the
    /// resolved interpretation.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NoAlphaBand`] if the image has fewer
    /// than two bands. Float rasters are **not** an error; see
    /// [`Raster::try_premultiply`] (issue #631).
    pub fn try_unpremultiply(&self) -> Result<Raster, ArithmeticError> {
        self.alpha_map(AlphaOp::Unpremultiply)
    }

    /// Panicking form of [`Raster::try_unpremultiply`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see
    /// [`Raster::try_unpremultiply`]. As with [`Raster::premultiply`], that
    /// is [`ArithmeticError::NoAlphaBand`] and allocation failure, and no
    /// longer a float carrier (#631).
    #[track_caller]
    pub fn unpremultiply(&self) -> Raster {
        expect_arith("unpremultiply", self.try_unpremultiply())
    }

    /// Entry point for the alpha ops: pick the kernel for `op` and the
    /// carrier, and run it.
    ///
    /// The two carriers are genuinely different code in libvips as well as
    /// here: `premultiply.c` and `unpremultiply.c` each switch on
    /// `im->BandFmt`, and the float arms differ from the integer ones in the
    /// dead zone (`FUNPRE_*` against `UNPRE_*`), in the output depth, and in
    /// the rounding. So there are four kernels behind this, one per
    /// (op, carrier) pair, and this function is the only thing that branches
    /// on either.
    ///
    /// [`Raster::alpha_map_unsigned`] is `main`'s kernel unchanged: compute in
    /// `f64`, round to nearest and saturate into the input depth, `max` from
    /// [`depth_max`]. [`Raster::alpha_map_float`] stores raw `f32` with no
    /// rounding or clamping, which is what `VIPS_FORMAT_FLOAT` input produces
    /// (both ops write `FLOAT` output), and takes `max` from the
    /// interpretation via [`interpretation_max_alpha`].
    ///
    /// # Why the op is a const-generic parameter and not a value
    ///
    /// This is a dispatcher onto two kernels, each taking the discriminant as
    /// `const OP: u8` so the op folds away inside the pixel loop. That shape
    /// is load-bearing rather than stylistic. The first cut of #631 hoisted
    /// the op out of the loop into a `fn(f64, f64, f64) -> f64` pointer, which
    /// reads like the faster spelling and is not: a `fn` pointer is opaque to
    /// the inliner, so the loop kept an indirect call. The evidence is in the
    /// binary. `nm` on that build shows two
    /// `alpha_map::{closure} as FnOnce<(f64, f64, f64)>::call_once` shims that
    /// exist neither on `main` nor here, and the loop body reads
    /// `blr x22` / `fcmp` / `frinta`, an indirect call landing immediately
    /// before the rounding. End-to-end at 2048x1536 that cost 43 to 70 percent
    /// on the unsigned carriers against the generic form `main` shipped
    /// (`Rgba16` unpremultiply 19.71 ms against 11.72 ms). Both shims and both
    /// `blr`s are gone in this form.
    ///
    /// The float arm keeps its `match` *inside* the pixel loop and pays
    /// nothing for it, because with `OP` a constant there is no match left to
    /// pay for. The lesson generalises: hoisting a branch out of a hot loop
    /// only helps if what replaces it is still transparent to the optimiser.
    ///
    /// The dispatch is four-way rather than two, because the carrier chooses
    /// the *function* as well; see [`Raster::alpha_map_float`] for the
    /// measurement behind that.
    fn alpha_map(&self, op: AlphaOp) -> Result<Raster, ArithmeticError> {
        match (op, self.format().is_float()) {
            (AlphaOp::Premultiply, false) => self.alpha_map_unsigned::<{ AlphaOp::PREMULTIPLY }>(),
            (AlphaOp::Unpremultiply, false) => {
                self.alpha_map_unsigned::<{ AlphaOp::UNPREMULTIPLY }>()
            }
            (AlphaOp::Premultiply, true) => self.alpha_map_float::<{ AlphaOp::PREMULTIPLY }>(),
            (AlphaOp::Unpremultiply, true) => self.alpha_map_float::<{ AlphaOp::UNPREMULTIPLY }>(),
        }
    }

    /// The unsigned arm of [`Raster::alpha_map`]: compute in `f64`, round to
    /// nearest and saturate into the input depth, `max` from [`depth_max`].
    /// `OP` is [`AlphaOp::PREMULTIPLY`] or [`AlphaOp::UNPREMULTIPLY`].
    ///
    /// This is `main`'s kernel unchanged apart from where the op comes from,
    /// and it is a separate function from [`Raster::alpha_map_float`] for a
    /// measured reason rather than a tidiness one. See
    /// [`Raster::alpha_map_float`].
    fn alpha_map_unsigned<const OP: u8>(&self) -> Result<Raster, ArithmeticError> {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        if bands < 2 {
            return Err(ArithmeticError::NoAlphaBand { bands });
        }
        let pixels = self.width() as usize * self.height() as usize;
        let data = self.data();
        let mut out = alloc_op_output(self.width(), self.height(), fmt)?;
        let max = depth_max(kind);
        for p in 0..pixels {
            let alpha = read_f64(data, kind, p * bands + bands - 1);
            for c in 0..bands - 1 {
                let v = read_f64(data, kind, p * bands + c);
                // `OP` is a constant, so only one of these survives into the
                // loop body.
                let mapped = if OP == AlphaOp::PREMULTIPLY {
                    v * alpha.clamp(0.0, max) / max
                } else {
                    v * unpremultiply_factor(alpha, max)
                };
                write_f64(&mut out, kind, p * bands + c, mapped, max);
            }
            write_f64(&mut out, kind, p * bands + bands - 1, alpha, max);
        }
        Ok(stamp_source_interpretation(
            Raster::from_op_output(self.width(), self.height(), fmt, out)?,
            self,
        ))
    }

    /// The float arm of [`Raster::alpha_map`]: store raw `f32` with no
    /// rounding or clamping, which is what `VIPS_FORMAT_FLOAT` input produces
    /// (both ops write `FLOAT` output), with `max` from the interpretation via
    /// [`interpretation_max_alpha`]. `OP` as for
    /// [`Raster::alpha_map_unsigned`].
    ///
    /// # Why the two carriers are two functions
    ///
    /// Because sharing one was expensive, measured. The first cut of #631 put
    /// both arms in a single function behind an `if fmt.is_float()`, and that
    /// alone cost the *unsigned* carriers 18 to 51 percent against `main`, on
    /// top of the `fn`-pointer regression and independent of it. Round-robin
    /// at 2048x1536, `opt-level=3`, min over 5 runs of 7 to 9 reps:
    ///
    /// | | main | one function | two functions |
    /// |---|---|---|---|
    /// | `Rgba8` premultiply | 11.36 | 13.35 | 11.01 |
    /// | `Rgba8` unpremultiply | 10.48 | 13.33 | 10.36 |
    /// | `Rgba16` premultiply | 12.16 | 17.93 | 12.07 |
    /// | `Rgba16` unpremultiply | 11.91 | 17.94 | 11.87 |
    ///
    /// The `main` and one-function columns share a round-robin; the
    /// two-function column comes from a second one, whose `main` re-measured
    /// within one percent at 11.27, 10.31, 11.96 and 11.72.
    ///
    /// Three candidate explanations are ruled out rather than argued. The
    /// interpretation stamp is free (removing it moved nothing, 17.88 against
    /// 17.89). The op dispatch is free: the const-generic unsigned kernel with
    /// the float arm deleted outright runs at 12.31 against `main`'s 12.16. And
    /// the float arm's *body* is not the cost either, since moving it behind an
    /// `#[inline(never)]` call while leaving the `is_float` test in place
    /// recovered under a third of the gap (16.85).
    ///
    /// What is left is that the optimiser stops treating the unsigned pixel
    /// loop the way it does on `main` as soon as a second carrier arm shares
    /// the function with it. I have not pinned down which transform gives up.
    /// `read_f64` and `write_f64` both match on the loop-invariant `kind`, so
    /// unswitching that match is the obvious suspect, but I could not confirm
    /// it from the disassembly and am not going to assert it.
    ///
    /// The shape of the lesson is the same as the `fn` pointer this replaced,
    /// one level up: the code that reads as though it costs nothing is what
    /// stops the optimiser seeing a loop it could specialise, and the only way
    /// to know is to measure both.
    fn alpha_map_float<const OP: u8>(&self) -> Result<Raster, ArithmeticError> {
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        if bands < 2 {
            return Err(ArithmeticError::NoAlphaBand { bands });
        }
        let pixels = self.width() as usize * self.height() as usize;
        let data = self.data();
        let mut out = alloc_op_output(self.width(), self.height(), fmt)?;
        // `read_f64` widens the stored `f32` losslessly, so narrowing it
        // straight back is exact and keeps the arithmetic in `f32` from
        // here on, which is the point (see below).
        let max = interpretation_max_alpha(self.interpretation()) as f32;
        for p in 0..pixels {
            let alpha = read_f64(data, kind, p * bands + bands - 1) as f32;
            // Both C macros land the multiplier in a `float` *before*
            // the colour multiply (`OUT nalpha` / `OUT factor`, with
            // `OUT` = float for FLOAT input), so the result rounds twice
            // and an `f64` expression rounded once at the store is not
            // the same number: 100 * 0.5 / 255 is 0.19607845 through the
            // float intermediate and 0.19607843 without it.
            let (factor, stored) = if OP == AlphaOp::PREMULTIPLY {
                // `premultiply.c`: the factor takes the *clipped* alpha
                // and the stored alpha stays raw.
                (
                    (f64::from(alpha.clamp(0.0, max)) / f64::from(max)) as f32,
                    alpha,
                )
            } else {
                // `unpremultiply.c`: the mirror image, factor from the
                // raw alpha (so over- and undershoots cancel) and the
                // stored alpha clipped. This is the carrier the `0.01`
                // dead zone was written for (#604).
                (
                    unpremultiply_factor(f64::from(alpha), f64::from(max)) as f32,
                    alpha.clamp(0.0, max),
                )
            };
            for c in 0..bands - 1 {
                let v = read_f64(data, kind, p * bands + c) as f32;
                write_f32(&mut out, p * bands + c, f64::from(v * factor));
            }
            write_f32(&mut out, p * bands + bands - 1, f64::from(stored));
        }
        Ok(stamp_source_interpretation(
            Raster::from_op_output(self.width(), self.height(), fmt, out)?,
            self,
        ))
    }

    // -----------------------------------------------------------------
    // Transcendental maths (libvips math / math2): float output
    // -----------------------------------------------------------------

    /// Sine of every sample, input in degrees (libvips `math` SIN). Float
    /// output; accepts every input depth including float.
    pub fn sin(&self) -> Raster {
        unary_map_float(self, |v| v.to_radians().sin())
    }

    /// Cosine of every sample, input in degrees (libvips `math` COS).
    /// Float output.
    pub fn cos(&self) -> Raster {
        unary_map_float(self, |v| v.to_radians().cos())
    }

    /// Tangent of every sample, input in degrees (libvips `math` TAN).
    /// Float output.
    pub fn tan(&self) -> Raster {
        unary_map_float(self, |v| v.to_radians().tan())
    }

    /// Arc sine of every sample, output in degrees (libvips `math` ASIN).
    /// Float output; inputs outside `[-1, 1]` produce NaN.
    pub fn asin(&self) -> Raster {
        unary_map_float(self, |v| v.asin().to_degrees())
    }

    /// Arc cosine of every sample, output in degrees (libvips `math`
    /// ACOS). Float output; inputs outside `[-1, 1]` produce NaN.
    pub fn acos(&self) -> Raster {
        unary_map_float(self, |v| v.acos().to_degrees())
    }

    /// Arc tangent of every sample, output in degrees (libvips `math`
    /// ATAN). Float output.
    pub fn atan(&self) -> Raster {
        unary_map_float(self, |v| v.atan().to_degrees())
    }

    /// Hyperbolic sine of every sample (libvips `math` SINH). Float
    /// output.
    pub fn sinh(&self) -> Raster {
        unary_map_float(self, f64::sinh)
    }

    /// Hyperbolic cosine of every sample (libvips `math` COSH). Float
    /// output.
    pub fn cosh(&self) -> Raster {
        unary_map_float(self, f64::cosh)
    }

    /// Hyperbolic tangent of every sample (libvips `math` TANH). Float
    /// output.
    pub fn tanh(&self) -> Raster {
        unary_map_float(self, f64::tanh)
    }

    /// Inverse hyperbolic sine of every sample (libvips `math` ASINH).
    /// Float output.
    pub fn asinh(&self) -> Raster {
        unary_map_float(self, f64::asinh)
    }

    /// Inverse hyperbolic cosine of every sample (libvips `math` ACOSH).
    /// Float output; inputs below `1` produce NaN.
    pub fn acosh(&self) -> Raster {
        unary_map_float(self, f64::acosh)
    }

    /// Inverse hyperbolic tangent of every sample (libvips `math` ATANH).
    /// Float output; `atanh(1)` is `+inf` and inputs outside `[-1, 1]`
    /// produce NaN.
    pub fn atanh(&self) -> Raster {
        unary_map_float(self, f64::atanh)
    }

    /// Natural logarithm of every sample (libvips `math` LOG). Float
    /// output; `log(0)` is `-inf` and negative inputs produce NaN.
    pub fn log(&self) -> Raster {
        unary_map_float(self, f64::ln)
    }

    /// Base-10 logarithm of every sample (libvips `math` LOG10). Float
    /// output; see [`Raster::log`] for the domain edges.
    pub fn log10(&self) -> Raster {
        unary_map_float(self, f64::log10)
    }

    /// `e` raised to every sample (libvips `math` EXP). Float output;
    /// large inputs overflow to `+inf` in `f32`.
    pub fn exp(&self) -> Raster {
        unary_map_float(self, f64::exp)
    }

    /// `10` raised to every sample (libvips `math` EXP10). Float output;
    /// see [`Raster::exp`].
    pub fn exp10(&self) -> Raster {
        unary_map_float(self, |v| 10.0f64.powf(v))
    }

    /// Samplewise `atan2(self, other)` in degrees (libvips `math2` ATAN2:
    /// `self` is the ordinate, `other` the abscissa). Float output.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_atan2(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_float(self, other, |a, b| a.atan2(b).to_degrees())
    }

    /// Panicking form of [`Raster::try_atan2`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_atan2`].
    #[track_caller]
    pub fn atan2(&self, other: &Raster) -> Raster {
        expect_arith("atan2", self.try_atan2(other))
    }

    /// Samplewise `self ** other` (libvips `math2` POW). Float output, so
    /// fractional exponents and large results survive; contrast
    /// [`Raster::pow_const`], which keeps the older integer contract.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_pow(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_float(self, other, pow_vips)
    }

    /// Panicking form of [`Raster::try_pow`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_pow`].
    #[track_caller]
    pub fn pow(&self, other: &Raster) -> Raster {
        expect_arith("pow", self.try_pow(other))
    }

    /// Samplewise `other ** self`, power with the operands reversed
    /// (libvips `math2` WOP). Float output.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_wop(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_float(self, other, |a, b| pow_vips(b, a))
    }

    /// Panicking form of [`Raster::try_wop`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_wop`].
    #[track_caller]
    pub fn wop(&self, other: &Raster) -> Raster {
        expect_arith("wop", self.try_wop(other))
    }

    // -----------------------------------------------------------------
    // Complex operations
    // -----------------------------------------------------------------

    /// Build a complex image from a real and an imaginary image (libvips
    /// `complexform`).
    ///
    /// The output is a float raster with twice the input band count,
    /// holding `(re, im)` pairs: band `2b` is band `b` of `real`, band
    /// `2b + 1` is band `b` of `imag`. Both inputs may be any depth.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree, or
    /// [`ArithmeticError::TooManyBands`] if the doubled band count
    /// exceeds `u16::MAX`.
    pub fn try_complexform(real: &Raster, imag: &Raster) -> Result<Raster, ArithmeticError> {
        ensure_compatible(real, imag)?;
        let bands = real.format().channels();
        let out_fmt = format_for(bands * 2, SampleKind::F32)?;
        let (re_kind, im_kind) = (real.format().kind(), imag.format().kind());
        let pixels = real.width() as usize * real.height() as usize;
        let mut out = alloc_op_output(real.width(), real.height(), out_fmt)?;
        let (re_data, im_data) = (real.data(), imag.data());
        for p in 0..pixels {
            for b in 0..bands {
                let i = p * bands + b;
                write_f32(&mut out, 2 * i, read_f64(re_data, re_kind, i));
                write_f32(&mut out, 2 * i + 1, read_f64(im_data, im_kind, i));
            }
        }
        Ok(Raster::from_op_output(
            real.width(),
            real.height(),
            out_fmt,
            out,
        )?)
    }

    /// Panicking form of [`Raster::try_complexform`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_complexform`].
    #[track_caller]
    pub fn complexform(real: &Raster, imag: &Raster) -> Raster {
        expect_arith("complexform", Self::try_complexform(real, imag))
    }

    /// Convert every complex pair from rectangular to polar form (libvips
    /// `complex` POLAR): `(re, im)` becomes `(magnitude, angle)` with the
    /// angle in degrees. Float output.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NotComplex`] if the band count is odd.
    pub fn try_polar(&self) -> Result<Raster, ArithmeticError> {
        complex_map(self, |re, im| (re.hypot(im), im.atan2(re).to_degrees()))
    }

    /// Panicking form of [`Raster::try_polar`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_polar`].
    #[track_caller]
    pub fn polar(&self) -> Raster {
        expect_arith("polar", self.try_polar())
    }

    /// Convert every complex pair from polar to rectangular form (libvips
    /// `complex` RECT): `(magnitude, angle_degrees)` becomes `(re, im)`.
    /// Float output.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NotComplex`] if the band count is odd.
    pub fn try_rect(&self) -> Result<Raster, ArithmeticError> {
        complex_map(self, |mag, angle| {
            let (s, c) = angle.to_radians().sin_cos();
            (mag * c, mag * s)
        })
    }

    /// Panicking form of [`Raster::try_rect`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_rect`].
    #[track_caller]
    pub fn rect(&self) -> Raster {
        expect_arith("rect", self.try_rect())
    }

    /// Complex conjugate of every pair (libvips `complex` CONJ):
    /// `(re, im)` becomes `(re, -im)`. Float output.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NotComplex`] if the band count is odd.
    pub fn try_conj(&self) -> Result<Raster, ArithmeticError> {
        complex_map(self, |re, im| (re, -im))
    }

    /// Panicking form of [`Raster::try_conj`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_conj`].
    #[track_caller]
    pub fn conj(&self) -> Raster {
        expect_arith("conj", self.try_conj())
    }

    /// Real part of every complex pair (libvips `complexget` REAL): a
    /// float raster with half the band count.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NotComplex`] if the band count is odd.
    pub fn try_real(&self) -> Result<Raster, ArithmeticError> {
        complex_get(self, 0)
    }

    /// Panicking form of [`Raster::try_real`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_real`].
    #[track_caller]
    pub fn real(&self) -> Raster {
        expect_arith("real", self.try_real())
    }

    /// Imaginary part of every complex pair (libvips `complexget` IMAG):
    /// a float raster with half the band count.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NotComplex`] if the band count is odd.
    pub fn try_imag(&self) -> Result<Raster, ArithmeticError> {
        complex_get(self, 1)
    }

    /// Panicking form of [`Raster::try_imag`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_imag`].
    #[track_caller]
    pub fn imag(&self) -> Raster {
        expect_arith("imag", self.try_imag())
    }

    // -----------------------------------------------------------------
    // Hough transforms
    // -----------------------------------------------------------------

    /// Hough line transform (libvips `hough_line` with its default
    /// 256 x 256 accumulator).
    ///
    /// Every non-zero sample votes for the lines through its pixel, and the
    /// **binning** of those votes is vips-exact: the accumulator cell each
    /// vote lands in matches vips 8.18.4 `hough_line` cell-for-cell (see the
    /// oracle-pinned tests). The output is a single-band 16-bit accumulator:
    /// column `i` is the angle bin `theta = 180deg * i / 256`. The vote
    /// *counts* carried in those cells are ushort rather than vips's uint and
    /// saturate at `65535` — an intentional format deviation, documented
    /// below, that leaves the "vips-exact" claim scoped to the binning only.
    ///
    /// vips normalizes the pixel coordinates by the image **diagonal**
    /// `d = sqrt(w^2 + h^2)` — not by width/height separately — so
    /// `xd = x / d`, `yd = y / d`, and the signed line distance
    /// `r = xd*cos(theta) + yd*sin(theta)` always lies in the open interval
    /// `(-1, 1)`. That distance is mapped linearly onto the accumulator rows
    /// with `ri = (r + 1) * (height / 2)` (so `r = -1` -> row 0, `r = +1`
    /// -> row `height`), centring `r = 0` on the middle row. Because `r` is
    /// strictly inside `(-1, 1)` no vote ever falls outside the accumulator,
    /// so — unlike a raw `height * r` binning — no votes are discarded.
    ///
    /// A peak at `(i, ri)` therefore decodes as angle `180 * i / width` and
    /// signed pixel distance `(2 * ri / height - 1) * sqrt(w^2 + h^2)`.
    ///
    /// # Intentional accumulator-format deviation from vips (#495)
    ///
    /// vips 8.18.4 emits the `hough_line` accumulator as a **32-bit `uint`**
    /// image whose cells hold the full collinear-vote count with **no
    /// saturation** (confirmed: `vipsheader` reports `uint`, and a 70000-wide
    /// lit line peaks at exactly `70000`). This crate has no unsigned 32-bit
    /// pixel format, so the accumulator is carried as `Gray16` (ushort) and
    /// every cell is clamped with `v.min(0xFFFF)`. The two therefore diverge
    /// only when a single accumulator cell would exceed `65535` — i.e. when
    /// more than 65535 collinear lit pixels concentrate into one bin, which
    /// requires an image dimension above 65535 (vips accepts such images).
    /// On that case vips reports the true count while this op reports `65535`.
    /// The *binning* (which cell each vote lands in) is unaffected and stays
    /// vips-exact; only the cell's carrier width and the >65535 saturation
    /// differ. This mirrors the format deviation disclosed for
    /// [`Raster::hough_circle`] and is tracked by issue #495.
    ///
    /// The angle terms are read from a `sin` lookup table indexed by
    /// `i` and `i + width/2` (vips uses `sin[i + width/2]` for the cosine
    /// term); for the default even 256-wide accumulator `width/2` is an
    /// exact quarter turn, but the table indexing is preserved so odd or
    /// non-default widths bin identically to vips.
    pub fn hough_line(&self) -> Raster {
        let (aw, ah) = (HOUGH_LINE_WIDTH as usize, HOUGH_LINE_HEIGHT as usize);
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let data = self.data();

        // vips normalizes coordinates by the image diagonal, so a pixel's
        // signed distance to a line through the origin stays in (-1, 1).
        let diag = ((w * w + h * h) as f64).sqrt();

        // sin table: s[k] = sin(PI * k / width). vips reads the cosine term
        // as s[i + width/2] rather than cos(theta), so replicate the table
        // (indices run up to (aw - 1) + aw/2).
        let sin_t: Vec<f64> = (0..aw + aw / 2)
            .map(|k| (std::f64::consts::PI * k as f64 / aw as f64).sin())
            .collect();

        let mut acc = vec![0u32; aw * ah];
        for y in 0..h {
            let yd = y as f64 / diag;
            for x in 0..w {
                let voters = (0..bands)
                    .filter(|&c| read_f64(data, kind, (y * w + x) * bands + c) != 0.0)
                    .count() as u32;
                if voters == 0 {
                    continue;
                }
                let xd = x as f64 / diag;
                for i in 0..aw {
                    // r in (-1, 1); vips: r = xd*sin[i + width/2] + yd*sin[i].
                    let r = xd * sin_t[i + aw / 2] + yd * sin_t[i];
                    // vips: int ri = (r + 1) * (height / 2.0). r < 1 keeps
                    // ri < height; the guard is defensive against a boundary
                    // rounding to exactly height and never discards a vote.
                    let ri = ((r + 1.0) * (ah as f64 / 2.0)) as usize;
                    if ri < ah {
                        acc[ri * aw + i] += voters;
                    }
                }
            }
        }

        // vips carries the accumulator as uint; this crate has no unsigned
        // 32-bit format, so it is emitted as Gray16 and cells are clamped at
        // 65535 (an intentional, documented deviation — see the rustdoc and
        // issue #495). The u32 accumulator above keeps the binning exact; only
        // cells with >65535 votes (a >65535-pixel collinear line) saturate.
        let out_fmt = PixelFormat::with_kind(1, SampleKind::U16).expect("Gray16 exists");
        let mut out = op_output_or_panic(HOUGH_LINE_WIDTH, HOUGH_LINE_HEIGHT, out_fmt);
        for (i, &v) in acc.iter().enumerate() {
            write_u32(&mut out, SampleKind::U16, i, v.min(0xFFFF));
        }
        Raster::from_op_output(HOUGH_LINE_WIDTH, HOUGH_LINE_HEIGHT, out_fmt, out)
            .expect("hough accumulator is well-formed")
    }

    /// Hough circle transform (libvips `hough_circle` at scale 1).
    ///
    /// Every non-zero sample votes, for each candidate radius, along the
    /// midpoint circle of that radius centred on its pixel: exactly the
    /// point set [`Raster::draw_circle`] plots, so a drawn circle's pixels
    /// all vote for the drawn centre at the drawn radius. The output has
    /// the input dimensions and one 16-bit band per candidate radius
    /// (`min_radius..=max_radius`); the peak's pixel is the detected
    /// centre and its strongest band index plus `min_radius` is the
    /// detected radius. Counts saturate at `65535`.
    ///
    /// # Intentional accumulator-model deviation from vips (#495)
    ///
    /// This op does **not** reproduce vips 8.18.4's exact per-cell vote
    /// counts, and is deliberately kept as a golden-only op (no vips
    /// cross-oracle in the differential suite). vips votes by running its
    /// Bresenham circle walker (`vips__draw_circle_direct`) and, for every
    /// scanline it emits, incrementing **both** span endpoints
    /// *unconditionally* — with no deduplication at the cardinal (`x == 0`)
    /// and diagonal (`x == y`) points where octant reflections coincide,
    /// and re-emitting the final `x == y` scanlines. The result is a
    /// radius-dependent vote multiplicity: a single lit pixel yields
    /// accumulator cells as high as `4` for small radii (e.g. vips totals
    /// `64/36/40/48/48` votes for radii `4..=8` of one point), where the
    /// multiplier collapses back to `1` only once the radius is large
    /// enough that no two octant points share a cell.
    ///
    /// This crate instead casts **one** vote per *distinct* circle point
    /// (the deduplicated octant walk shared with [`Raster::draw_circle`]),
    /// so a single pixel produces a clean max of `1` per cell. Peak
    /// **location** and the detected radius agree with vips; only the raw
    /// vote magnitudes differ. Matching vips's counts exactly would require
    /// porting its non-deduplicating scanline-endpoint accumulation, whose
    /// small-radius multiplicity could not be reproduced faithfully even
    /// from the 8.18.4 source; the deduplicated model is retained as an
    /// intentional, internally-consistent deviation rather than risk a
    /// subtly-wrong rewrite of a correctly-peaking transform.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::EmptyRadiusRange`] if
    /// `min_radius > max_radius`, or [`ArithmeticError::TooManyBands`] if
    /// the radius count exceeds `u16::MAX`.
    pub fn try_hough_circle(
        &self,
        min_radius: u32,
        max_radius: u32,
    ) -> Result<Raster, ArithmeticError> {
        if min_radius > max_radius {
            return Err(ArithmeticError::EmptyRadiusRange {
                min: min_radius,
                max: max_radius,
            });
        }
        let radii = (max_radius - min_radius + 1) as usize;
        let out_fmt = format_for(radii, SampleKind::U16)?;
        let fmt = self.format();
        let (bands, kind) = (fmt.channels(), fmt.kind());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let data = self.data();

        // The vote accumulator is `radii` bands deep — twice the output and
        // sized by the caller-controlled radius range — so it dominates the
        // op's memory. Allocate it fallibly: an over-capacity range returns a
        // typed error rather than aborting through `handle_alloc_error`, which
        // making only the output fallible left open (#433 / #434).
        let acc_len = w
            .checked_mul(h)
            .and_then(|wh| wh.checked_mul(radii))
            .ok_or(RasterError::SizeOverflow {
                width: self.width(),
                height: self.height(),
                bpp: radii.saturating_mul(4),
            })?;
        let mut acc = try_plane_len_filled(
            plane::HOUGH_CIRCLE_ACCUMULATOR,
            self.width(),
            self.height(),
            acc_len,
            0u32,
        )?;
        {
            let mut vote = |cx: i32, cy: i32, band: usize, votes: u32| {
                if cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h {
                    acc[(cy as usize * w + cx as usize) * radii + band] += votes;
                }
            };
            for y in 0..h {
                for x in 0..w {
                    let voters = (0..bands)
                        .filter(|&c| read_f64(data, kind, (y * w + x) * bands + c) != 0.0)
                        .count() as u32;
                    if voters == 0 {
                        continue;
                    }
                    let (px, py) = (x as i32, y as i32);
                    for r in min_radius..=max_radius {
                        let band = (r - min_radius) as usize;
                        if r == 0 {
                            vote(px, py, band, voters);
                            continue;
                        }
                        crate::draw::for_each_octant_step(r as i32, |ox, oy| {
                            // The eight octant reflections, deduplicated at
                            // oy == 0 and ox == oy so votes are one per
                            // distinct circle point.
                            vote(px + ox, py + oy, band, voters);
                            vote(px - ox, py + oy, band, voters);
                            if oy != 0 {
                                vote(px + ox, py - oy, band, voters);
                                vote(px - ox, py - oy, band, voters);
                            }
                            if ox != oy {
                                vote(px + oy, py + ox, band, voters);
                                vote(px + oy, py - ox, band, voters);
                                if oy != 0 {
                                    vote(px - oy, py + ox, band, voters);
                                    vote(px - oy, py - ox, band, voters);
                                }
                            }
                        });
                    }
                }
            }
        }

        let mut out = alloc_op_output(self.width(), self.height(), out_fmt)?;
        for (i, &v) in acc.iter().enumerate() {
            write_u32(&mut out, SampleKind::U16, i, v.min(0xFFFF));
        }
        Ok(Raster::from_op_output(
            self.width(),
            self.height(),
            out_fmt,
            out,
        )?)
    }

    /// Panicking form of [`Raster::try_hough_circle`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_hough_circle`].
    #[track_caller]
    pub fn hough_circle(&self, min_radius: u32, max_radius: u32) -> Raster {
        expect_arith(
            "hough_circle",
            self.try_hough_circle(min_radius, max_radius),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{counting_planes, with_plane_cap_at};

    /**
     * Tests that this module dispatches on sample kind and never on byte
     * width, by asserting that neither the byte-width accessor on
     * [`PixelFormat`] nor its width-keyed constructor survives in
     * `src/arithmetic.rs`.
     * Works by scanning the module's own source, compiled in with
     * `include_str!`, for the accessor's name; the needle is spelled in two
     * halves so this assertion is not itself a hit. A byte width is not a
     * sample kind: four bytes is `f32` today and would be `u32` under issue
     * #517, so a `match` keyed on the width silently takes a wrong arm for
     * any carrier added later instead of failing to compile (issue #607).
     * Input: `src/arithmetic.rs` -> Output: zero occurrences.
     */
    #[test]
    fn arithmetic_does_not_dispatch_on_byte_width() {
        const SRC: &str = include_str!("arithmetic.rs");
        // Both spellings of a byte-width dispatch: reading the width off a
        // format, and handing one back to the width-keyed constructor.
        let needles = [
            concat!("bytes_per_", "channel"),
            concat!("with_", "channels"),
        ];
        // Positive control: the same scan over the same string finds a token
        // that is present, so the zero below is a real zero and not the
        // vacuous pass an empty read would give.
        assert!(
            SRC.contains(concat!("fn read_", "u32")),
            "positive control failed: the scan cannot see this module's source"
        );
        for needle in needles {
            assert_eq!(
                SRC.matches(needle).count(),
                0,
                "{needle} is back in src/arithmetic.rs; dispatch on PixelFormat::kind() \
                 and PixelFormat::with_kind() instead"
            );
        }
    }

    /**
     * Tests that the module's sample helpers read and write every
     * [`SampleKind`], including the signed and 32-bit kinds no
     * `PixelFormat` carries yet (issues #516, #517), so those arms are a
     * real path rather than a hole waiting for the carrier.
     * Works by writing a sample through `write_u32` and reading it back
     * with both `read_u32` (the storage bit pattern, which is what the
     * bitwise family and the non-zero scans want) and `read_f64` (the
     * numeric value, sign-extended), on a buffer sized for the kind.
     * Input: -1 stored as I16 -> `read_u32` 0xFFFF, `read_f64` -1.0;
     * 4294967295 stored as U32 -> `read_f64` 4294967295.0.
     */
    #[test]
    fn sample_helpers_read_and_write_every_integer_kind() {
        // (kind, stored bit pattern, numeric value)
        let cases: [(SampleKind, u32, f64); 12] = [
            (SampleKind::U8, 0, 0.0),
            (SampleKind::U8, 255, 255.0),
            (SampleKind::I8, 0xFF, -1.0),
            (SampleKind::I8, 0x80, -128.0),
            (SampleKind::I8, 127, 127.0),
            (SampleKind::U16, 65535, 65535.0),
            (SampleKind::I16, 0xFFFF, -1.0),
            (SampleKind::I16, 0x8000, -32768.0),
            (SampleKind::I16, 32767, 32767.0),
            (SampleKind::U32, 0xFFFF_FFFF, 4_294_967_295.0),
            (SampleKind::I32, 0xFFFF_FFFF, -1.0),
            (SampleKind::I32, 0x7FFF_FFFF, 2_147_483_647.0),
        ];
        for (kind, bits, value) in cases {
            // Two samples wide, and the write goes to index 1, so an arm
            // that ignores the stride writes over index 0 and is caught.
            let mut buf = vec![0u8; kind.bytes() * 2];
            write_u32(&mut buf, kind, 1, bits);
            assert_eq!(
                read_u32(&buf, kind, 1),
                bits,
                "{kind:?} did not round-trip the bit pattern {bits:#x}"
            );
            assert_eq!(
                read_f64(&buf, kind, 1),
                value,
                "{kind:?} read {bits:#x} as the wrong number"
            );
            assert!(
                buf[..kind.bytes()].iter().all(|&b| b == 0),
                "{kind:?} wrote outside sample 1"
            );
        }
    }

    /**
     * Tests that the rounding, saturating write clamps into the sample
     * kind's whole range rather than into `0..=max`, which is the floor
     * issue #607 names as the assumption the signed carriers of #516 break.
     * Works by writing a value under the kind's floor and one over its
     * ceiling and reading both back numerically, with the unsigned kinds
     * asserted alongside so the change cannot have moved them.
     * Input: -300.0 into I16 -> -32768; -5.0 into U8 -> 0; 400.0 into I8
     * -> 127.
     */
    #[test]
    fn write_f64_clamps_into_the_kind_range() {
        // (kind, written, read back)
        let cases: [(SampleKind, f64, f64); 10] = [
            (SampleKind::U8, -5.0, 0.0),
            (SampleKind::U8, 400.0, 255.0),
            (SampleKind::U16, -5.0, 0.0),
            (SampleKind::U16, 70000.0, 65535.0),
            (SampleKind::I8, -300.0, -128.0),
            (SampleKind::I8, 400.0, 127.0),
            (SampleKind::I16, -40000.0, -32768.0),
            (SampleKind::I16, 40000.0, 32767.0),
            (SampleKind::U32, -1.0, 0.0),
            (SampleKind::I32, -3.0e9, f64::from(i32::MIN)),
        ];
        for (kind, wrote, want) in cases {
            let mut buf = vec![0u8; kind.bytes()];
            let max = f64::from(kind.max_value().expect("an integer kind has a ceiling"));
            write_f64(&mut buf, kind, 0, wrote, max);
            assert_eq!(
                read_f64(&buf, kind, 0),
                want,
                "{kind:?} clamped {wrote} to the wrong value"
            );
        }
        // Positive control: an in-range value survives untouched, so the
        // clamp cannot be passing as a constant.
        let mut buf = vec![0u8; 2];
        write_f64(&mut buf, SampleKind::I16, 0, -1234.0, 32767.0);
        assert_eq!(read_f64(&buf, SampleKind::I16, 0), -1234.0);
    }

    /// A width x height Gray8 raster from a byte vector.
    fn gray(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /// A 1-band 16-bit raster from sample values.
    fn gray16(w: u32, h: u32, vals: &[u16]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Gray16, data).unwrap()
    }

    /// A ceiling below every scratch plane the ops under test reserve for a
    /// [`SCRATCH_PROBE_DIM`]² input, so the fallible path trips
    /// deterministically at a tiny, instantly-constructible raster instead of
    /// the multi-TiB one the real overflow would need (#460).
    ///
    /// It applies to **one named site** at a time now rather than to every
    /// scratch allocation on the thread, which is what lets the checks below
    /// starve `project`'s second accumulator and `try_stdif`'s second integral
    /// image at all: under the old per-module ceiling both pairs are the same
    /// size, so no ceiling existed that admitted the first and refused the
    /// second (issue #696).
    const SCRATCH_TEST_CAP_BYTES: u64 = 64;

    /// A modest Gray8 raster (`64² = 4 KiB`, trivially within the construction
    /// budget). Its `project` accumulators (`64 · 8 = 512` bytes), its `stdif`
    /// integral images (`75² · 8 = 45000` bytes) and its `hough_circle` vote
    /// accumulator (`64² · 3 · 4 = 49152` bytes) all dwarf
    /// [`SCRATCH_TEST_CAP_BYTES`], so the capped site's reservation is refused
    /// in each.
    const SCRATCH_PROBE_DIM: u32 = 64;

    /// The probe raster the scratch checks below drive.
    fn scratch_probe() -> Raster {
        Raster::zeroed(SCRATCH_PROBE_DIM, SCRATCH_PROBE_DIM, PixelFormat::Gray8)
            .expect("probe Gray8 raster is within the construction budget")
    }

    /**
     * Tests that **each** of `project`'s two input-scaled accumulators is
     * reserved through the crate's one fallible plane funnel, by starving them
     * one at a time by name (issue #696).
     *
     * `project` returns `(Raster, Raster)` and so has no error channel, which
     * makes a panic its only honest answer to a scratch it cannot have: a
     * panic unwinds and is catchable, where `handle_alloc_error` takes the
     * process down with it. That is what [`scratch_or_panic`] is for.
     *
     * The reason this is two cells and not one is the point of the labels.
     * `col_sums` and `row_sums` are the same size on a square raster and are
     * reserved one after the other, so a ceiling that refuses the Nth
     * over-ceiling request on the thread cannot express "let the first
     * through, starve the second": it would refuse `col_sums` either way and
     * `row_sums` would never be exercised. Naming the site does express it.
     *
     * Input: a 64² Gray8 raster with one accumulator's site capped at 64 bytes
     * -> a caught panic; the same raster uncapped -> two rasters.
     */
    #[test]
    fn project_starves_each_accumulator_by_name_rather_than_aborting() {
        for site in [plane::PROJECT_COL_SUMS, plane::PROJECT_ROW_SUMS] {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let caught = std::panic::catch_unwind(|| {
                with_plane_cap_at(site, SCRATCH_TEST_CAP_BYTES, || scratch_probe().project())
            });
            std::panic::set_hook(prev);
            assert!(
                caught.is_err(),
                "an unservable {site} must panic (unwindable), not abort"
            );
        }

        // Positive control: with nothing capped the same call completes, so
        // the two panics above are the ceiling and not the op.
        let (cols, rows) = scratch_probe().project();
        assert_eq!(
            (cols.width(), cols.height(), rows.width(), rows.height()),
            (SCRATCH_PROBE_DIM, 1, 1, SCRATCH_PROBE_DIM)
        );
    }

    /**
     * Tests that **each** of `try_stdif`'s two `f64` integral images is
     * reserved fallibly: an unservable one comes back as a typed `Err` routed
     * through [`RasterError::AllocationFailed`], never a process abort
     * (issues #459, #460, #696).
     *
     * Same shape as `project` above and the same reason for being two cells:
     * the sum and the sum-of-squares integral images are byte-for-byte the
     * same size and are reserved back to back, so only a per-site ceiling can
     * reach the second one.
     *
     * Input: a 64² Gray8 raster, `try_stdif(11, 11)`, one site capped at 64
     * bytes -> `Err`; uncapped -> a raster.
     */
    #[test]
    fn stdif_starves_each_integral_image_by_name_rather_than_aborting() {
        for site in [plane::STDIF_INTEGRAL, plane::STDIF_INTEGRAL_SQUARES] {
            let result = with_plane_cap_at(site, SCRATCH_TEST_CAP_BYTES, || {
                scratch_probe().try_stdif(11, 11)
            });
            assert!(
                result.is_err(),
                "an unservable {site} must return Err, not a raster"
            );
        }
        assert!(scratch_probe().try_stdif(11, 11).is_ok());
    }

    /**
     * Tests that `try_hough_circle`'s vote accumulator is reserved fallibly
     * (issues #433, #434, #696).
     *
     * This one had **no test ceiling at all** before the shared funnel. Its
     * buffer is the largest thing the op holds and it is sized by the caller's
     * radius range rather than by the image, so it is the one an untrusted
     * caller can steer, and the `try_` in the name promised something nothing
     * checked. It got a hook the moment it joined the funnel, which is the
     * argument for one funnel rather than a helper per module: the ceiling
     * arrives with the reservation instead of having to be written again.
     *
     * Input: a 64² Gray8 raster, radii 3..=5, the accumulator site capped at
     * 64 bytes -> `Err`; uncapped -> a raster.
     */
    #[test]
    fn hough_circle_accumulator_is_reserved_fallibly_rather_than_aborting() {
        let result = with_plane_cap_at(
            plane::HOUGH_CIRCLE_ACCUMULATOR,
            SCRATCH_TEST_CAP_BYTES,
            || scratch_probe().try_hough_circle(3, 5),
        );
        assert!(
            result.is_err(),
            "an unservable vote accumulator must return Err, not a raster"
        );
        assert!(scratch_probe().try_hough_circle(3, 5).is_ok());
    }

    /**
     * The arithmetic half of `raster.rs`'s funnel table: every plane these
     * entry points reserve goes through `raster::try_plane_len`, and the count
     * of them is what says so (issue #696).
     *
     * It lives here rather than in the table over in `raster.rs` for the same
     * reason the ICC entry points live in `colour.rs`: these rows need this
     * module's own inputs (a radius range, a window size) and dragging the
     * fixtures across buys nothing. The table there gained an `arithmetic.`
     * column all the same, at zero, so a convolution or colour path that
     * started reaching into this module would be caught there.
     *
     * Exact equality, and the total asserted to be the sum of the parts, so a
     * site added to one of these ops is red rather than absorbed. Measured,
     * then written down.
     */
    #[test]
    fn every_plane_these_arithmetic_paths_reserve_goes_through_the_one_funnel() {
        struct Row {
            op: &'static str,
            run: fn(&Raster),
            /// Reservations at an `arithmetic.` site.
            scratch: usize,
            /// Reservations at `raster.op_output`.
            outputs: usize,
        }
        const ROWS: &[Row] = &[
            Row {
                op: "project",
                run: |src| drop(src.project()),
                scratch: 2,
                outputs: 2,
            },
            Row {
                op: "try_stdif",
                run: |src| drop(src.try_stdif(11, 11)),
                scratch: 2,
                outputs: 1,
            },
            Row {
                op: "try_hough_circle over 3 radii",
                run: |src| drop(src.try_hough_circle(3, 5)),
                scratch: 1,
                outputs: 1,
            },
        ];

        let src = scratch_probe();
        for row in ROWS {
            // Warm-up outside every window, so a one-time lazily built table
            // cannot be charged to the first row that happens to touch it.
            (row.run)(&src);

            let count = |prefix| counting_planes(prefix, || (row.run)(&src)).1;
            let parts = [
                ("arithmetic planes", count("arithmetic."), row.scratch),
                (
                    "op outputs",
                    count(crate::raster::PLANE_OP_OUTPUT),
                    row.outputs,
                ),
            ];
            for (what, got, want) in parts {
                assert_eq!(
                    got, want,
                    "{} reserved {got} {what} against {want}: a site was added, removed, or \
                     routed around `raster::try_plane` (issue #696)",
                    row.op
                );
            }
            let total = count("");
            let sum: usize = parts.iter().map(|(_, got, _)| got).sum();
            assert_eq!(
                total, sum,
                "{} made {total} reservations and only {sum} of them are under one of the \
                 prefixes above: another module joined the funnel and no row here names it",
                row.op
            );
        }
    }

    /// Issue #271: the fallible per-band `try_*` ops return the typed
    /// [`ArithmeticError::FloatUnsupported`] error on float input instead of
    /// panicking through the old `assert!` in `vec_map`. The error carries the
    /// specific op name so callers can act on it, and no panic escapes.
    #[test]
    fn try_vec_ops_on_float_return_float_unsupported() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let im = Raster::zeroed(2, 2, f1).unwrap();

        assert!(matches!(
            im.try_add_vec(&[1.0]),
            Err(ArithmeticError::FloatUnsupported { op: "add_vec" })
        ));
        assert!(matches!(
            im.try_sub_vec(&[1.0]),
            Err(ArithmeticError::FloatUnsupported { op: "sub_vec" })
        ));
        assert!(matches!(
            im.try_mul_vec(&[1.0]),
            Err(ArithmeticError::FloatUnsupported { op: "mul_vec" })
        ));
    }

    /// Issue #271: the integer-output image-image `try_*` ops return
    /// [`ArithmeticError::FloatUnsupported`] on float input rather than
    /// panicking through the old `assert!` in `binary_map`. Image-image
    /// `sub` no longer belongs here — it floats its output (issue #282) and
    /// so accepts float input; `mul` keeps the integer contract and its
    /// float-input guard (see [`sub_accepts_float_input_and_stays_float`]).
    #[test]
    fn try_binary_ops_on_float_return_float_unsupported() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let a = Raster::zeroed(2, 2, f1).unwrap();
        let b = Raster::zeroed(2, 2, f1).unwrap();

        assert!(matches!(
            a.try_mul(&b),
            Err(ArithmeticError::FloatUnsupported { op: "mul" })
        ));
    }

    /// Issue #282: because image-image `sub` promotes to a float raster, it
    /// accepts float input too (the float-output family reads every depth),
    /// so a cast-then-subtract chain over float intermediates works and
    /// stays float instead of returning [`ArithmeticError::FloatUnsupported`].
    #[test]
    fn sub_accepts_float_input_and_stays_float() {
        let a = grayf(2, 1, &[0.5, 10.0]);
        let b = grayf(2, 1, &[2.0, 3.0]);
        let out = a.sub(&b);
        assert!(out.format().is_float());
        assert_eq!(float_samples(&out), vec![-1.5, 7.0]);
        assert!(a.try_sub(&b).is_ok());
    }

    /// Follow-up to #339 (tracked by #350): `FloatUnsupported` embeds the op
    /// name in its `Display`, so the panicking op forms must not re-prefix it
    /// via `expect_arith` — otherwise the diagnostic reads "mul: mul does not
    /// support float rasters yet ...". Both the fallible `try_*` error Display
    /// and the panicking form's panic message must name the op exactly once.
    /// Uses `mul`, which keeps the integer contract and its float-input guard
    /// (image-image `sub` now floats and accepts float input, issue #282).
    #[test]
    fn float_unsupported_names_op_exactly_once() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let a = Raster::zeroed(2, 2, f1).unwrap();
        let b = Raster::zeroed(2, 2, f1).unwrap();

        // Fallible form: the typed error's Display already names "mul" once.
        let display = a.try_mul(&b).unwrap_err().to_string();
        assert_eq!(
            display.matches("mul").count(),
            1,
            "try_mul error Display must name the op once, got: {display:?}"
        );
        assert!(display.starts_with("mul does not support float rasters yet"));

        // Panicking form: the caught panic payload must name "mul" once too
        // (was doubled as "mul: mul does not support float rasters yet ...").
        // Silence the default hook so the intentional panic stays off stderr.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.mul(&b)))
            .expect_err("mul() on a float raster must panic");
        std::panic::set_hook(prev);
        let panic_msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .expect("panic payload is a string");
        assert_eq!(
            panic_msg.matches("mul").count(),
            1,
            "mul() panic message must name the op once, got: {panic_msg:?}"
        );
    }

    /// Issues #279 / #280: an arithmetic op's output is built through the
    /// fallible, budget-free op-output path ([`alloc_op_output`] +
    /// [`Raster::from_op_output`]) that every op now funnels its result
    /// through, so an oversized declared output returns a typed `Err` instead
    /// of the old `Raster::new(...).expect(...)` panic, and an out-of-memory
    /// request returns `AllocationFailed` instead of aborting the process.
    ///
    /// The oversize contract is asserted at that shared boundary: a genuine
    /// multi-gigabyte op *input* cannot be allocated in a unit test (an op
    /// output is at most 4x its already-allocated input), so exercising the
    /// constructor the ops delegate to is the deterministic, cheap proof.
    #[test]
    fn op_output_path_rejects_oversize_without_panicking() {
        // 4 bytes-per-pixel at u32::MAX x u32::MAX overflows usize, so the
        // op-output constructor returns SizeOverflow rather than `.expect`.
        let rgba8 = PixelFormat::Rgba8;
        let built = Raster::from_op_output(u32::MAX, u32::MAX, rgba8, Vec::new());
        assert!(
            matches!(built, Err(RasterError::SizeOverflow { .. })),
            "expected SizeOverflow, got {built:?}"
        );

        // 1 byte-per-pixel fits usize on a 64-bit target but exceeds the Vec
        // capacity ceiling, so the fallible allocation returns AllocationFailed
        // (never SIGABRT). On a 32-bit-usize target the same u32::MAX x u32::MAX
        // product overflows usize, so `buffer_len` narrows it to SizeOverflow
        // first; either typed error proves the abort-safety contract holds.
        let alloc = alloc_op_output(u32::MAX, u32::MAX, PixelFormat::Gray8);
        assert!(
            matches!(
                alloc,
                Err(RasterError::AllocationFailed { .. } | RasterError::SizeOverflow { .. })
            ),
            "expected AllocationFailed or SizeOverflow, got {alloc:?}"
        );
    }

    /// Issue #279: a depth-promoting op still produces correct values through
    /// the budget-free op-output path (the panic fix must not perturb output).
    #[test]
    fn op_output_path_preserves_values() {
        let a = gray(1, 1, vec![200]);
        // add_const promotes 8-bit to 16-bit; the sum survives past 255.
        let out = a.add_const(100.0);
        assert_eq!(out.format().kind(), SampleKind::U16);
        assert_eq!(read_u32(out.data(), SampleKind::U16, 0), 300);
    }

    /// A 1-band float raster from `f32` sample values.
    fn grayf(w: u32, h: u32, vals: &[f32]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let fmt = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        Raster::new(w, h, fmt, data).unwrap()
    }

    #[test]
    fn max_diff_and_avg_diff_on_known_uchar_samples() {
        // |10-12| = 2, |20-25| = 5  ->  max 5, mean (2+5)/2 = 3.5.
        let a = gray(2, 1, vec![10, 20]);
        let b = gray(2, 1, vec![12, 25]);
        assert_eq!(a.max_diff(&b), 5.0);
        assert!((a.avg_diff(&b) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn max_diff_and_avg_diff_of_identical_rasters_are_zero() {
        // Pins the lossless-roundtrip assertion the foreign cells make.
        let a = gray(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(a.max_diff(&a.clone()), 0.0);
        assert_eq!(a.avg_diff(&a.clone()), 0.0);
    }

    #[test]
    fn avg_diff_reads_float_rasters() {
        // |0.5-0.25| = 0.25, |1.0-1.5| = 0.5  ->  mean 0.375.
        let a = grayf(2, 1, &[0.5, 1.0]);
        let b = grayf(2, 1, &[0.25, 1.5]);
        assert!((a.avg_diff(&b) - 0.375).abs() < 1e-6);
        assert!((a.max_diff(&b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn max_diff_and_avg_diff_propagate_a_nan_sample() {
        // NaN samples are unsupported input; both reductions agree by
        // propagating NaN, so neither silently reports a finite difference
        // (in particular max_diff must not drop the NaN and pass `== 0.0`).
        let a = grayf(2, 1, &[f32::NAN, 1.0]);
        let b = grayf(2, 1, &[0.0, 1.0]);
        assert!(a.max_diff(&b).is_nan());
        assert!(a.avg_diff(&b).is_nan());
    }

    #[test]
    fn diff_mismatched_dimensions_is_a_typed_error() {
        let a = gray(2, 1, vec![0, 0]);
        let b = gray(3, 1, vec![0, 0, 0]);
        assert!(matches!(
            a.try_max_diff(&b),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            a.try_avg_diff(&b),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn diff_mismatched_bands_is_a_typed_error() {
        let a = gray(2, 1, vec![0, 0]);
        let rgb = Raster::new(2, 1, PixelFormat::Rgb8, vec![0; 6]).unwrap();
        assert!(matches!(
            a.try_max_diff(&rgb),
            Err(ArithmeticError::BandCountMismatch { .. })
        ));
    }

    /// The flat samples of a float raster as `f64`, for assertions.
    fn float_samples(r: &Raster) -> Vec<f64> {
        assert!(r.format().is_float(), "expected a float raster");
        let n = r.width() as usize * r.height() as usize * r.format().channels();
        (0..n)
            .map(|i| read_f64(r.data(), SampleKind::F32, i))
            .collect()
    }

    /// A 100x100 Gray8 image whose left half is 0 and right half is `v`.
    fn half_half(v: u8) -> Raster {
        let mut data = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                data[y * 100 + x] = v;
            }
        }
        gray(100, 100, data)
    }

    // ---- avg / deviate ----

    /// avg is the mean of every sample: half-0 half-100 averages 50, and a
    /// 3-band pixel averages across bands.
    #[test]
    fn avg_all_samples() {
        assert!((half_half(100).avg() - 50.0).abs() < 1e-9);
        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        assert!((rgb.avg() - 20.0).abs() < 1e-9);
    }

    /// deviate uses the sample (n-1) formula: the half-0 half-100 image
    /// deviates by ~50.0025, and a constant image by 0.
    #[test]
    fn deviate_sample_stddev() {
        let d = half_half(100).deviate();
        assert!((d - 50.0).abs() < 0.01, "deviate should be ~50, got {d}");
        assert_eq!(gray(2, 2, vec![9; 4]).deviate(), 0.0);
    }

    /// deviate of a single-sample image is 0, not a division by zero.
    #[test]
    fn deviate_single_sample_is_zero() {
        assert_eq!(gray(1, 1, vec![42]).deviate(), 0.0);
    }

    // ---- min / max / minpos / maxpos ----

    /// min and max scan every band of every pixel.
    #[test]
    fn min_max_values() {
        let im = Raster::new(1, 2, PixelFormat::Rgb8, vec![5, 200, 30, 7, 2, 9]).unwrap();
        assert_eq!(im.min(), 2.0);
        assert_eq!(im.max(), 200.0);
    }

    /// maxpos returns the value and pixel position; ties keep the first
    /// occurrence in row-major order.
    #[test]
    fn maxpos_position_and_ties() {
        let mut data = vec![0u8; 100 * 100];
        data[50 * 100 + 40] = 100;
        let im = gray(100, 100, data);
        assert_eq!(im.maxpos(), (100.0, 40, 50));

        let tie = gray(3, 1, vec![7, 7, 3]);
        assert_eq!(tie.maxpos(), (7.0, 0, 0));
    }

    /// minpos finds the lone zero in a bright image.
    #[test]
    fn minpos_position() {
        let mut data = vec![100u8; 100 * 100];
        data[50 * 100 + 40] = 0;
        let im = gray(100, 100, data);
        assert_eq!(im.minpos(), (0.0, 40, 50));
    }

    /// 16-bit reductions read full-depth samples.
    #[test]
    fn reductions_16bit() {
        let im = gray16(2, 1, &[4096, 300]);
        assert_eq!(im.max(), 4096.0);
        assert_eq!(im.min(), 300.0);
        assert!((im.avg() - 2198.0).abs() < 1e-9);
    }

    // ---- stats ----

    /// stats row 0 holds overall [min, max, sum, sum2, mean, sd] and
    /// matches avg / deviate; band rows follow.
    #[test]
    fn stats_overall_row() {
        let mut data = vec![0u8; 100 * 50];
        for y in 0..50 {
            for x in 50..100 {
                data[y * 100 + x] = 10;
            }
        }
        let im = gray(100, 50, data);
        let stats = im.stats();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0][0], 0.0);
        assert_eq!(stats[0][1], 10.0);
        assert_eq!(stats[0][2], 25_000.0);
        assert_eq!(stats[0][3], 250_000.0);
        assert!((stats[0][4] - im.avg()).abs() < 1e-9);
        assert!((stats[0][5] - im.deviate()).abs() < 1e-9);
    }

    /// stats has one row per band after the overall row, each with
    /// per-band values.
    #[test]
    fn stats_per_band_rows() {
        let im = Raster::new(2, 1, PixelFormat::Rgb8, vec![1, 10, 100, 3, 30, 200]).unwrap();
        let stats = im.stats();
        assert_eq!(stats.len(), 4);
        assert_eq!(stats[1][0], 1.0); // band 0 min
        assert_eq!(stats[1][1], 3.0); // band 0 max
        assert_eq!(stats[2][2], 40.0); // band 1 sum
        assert_eq!(stats[3][4], 150.0); // band 2 mean
    }

    // ---- measure ----

    /// measure(2, 1) on a left-0 right-10 image returns [[0], [10]]: the
    /// patch order is left-to-right.
    #[test]
    fn measure_two_across() {
        let mut data = vec![0u8; 100 * 50];
        for y in 0..50 {
            for x in 50..100 {
                data[y * 100 + x] = 10;
            }
        }
        let im = gray(100, 50, data);
        let matrix = im.measure(2, 1);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![0.0]);
        assert_eq!(matrix[1], vec![10.0]);
    }

    /// measure reports per-band means and orders patches row-major
    /// (across, then down).
    #[test]
    fn measure_grid_and_bands() {
        // 2x2 grid over a 4x4 RGB image; each quadrant is constant.
        let mut data = vec![0u8; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..4 {
                let q = (u8::from(y >= 2) * 2 + u8::from(x >= 2)) * 10;
                let off = (y * 4 + x) * 3;
                data[off] = q;
                data[off + 1] = q + 1;
                data[off + 2] = q + 2;
            }
        }
        let im = Raster::new(4, 4, PixelFormat::Rgb8, data).unwrap();
        let m = im.measure(2, 2);
        assert_eq!(m.len(), 4);
        assert_eq!(m[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(m[1], vec![10.0, 11.0, 12.0]);
        assert_eq!(m[2], vec![20.0, 21.0, 22.0]);
        assert_eq!(m[3], vec![30.0, 31.0, 32.0]);
    }

    /// measure rejects zero and too-fine patch grids with typed errors.
    #[test]
    fn measure_typed_errors() {
        let im = gray(4, 4, vec![0; 16]);
        assert!(matches!(
            im.try_measure(0, 1),
            Err(ArithmeticError::ZeroPatches)
        ));
        assert!(matches!(
            im.try_measure(5, 1),
            Err(ArithmeticError::PatchGridTooFine { .. })
        ));
    }

    // ---- find_trim ----

    /// find_trim locates content against the default white background.
    #[test]
    fn find_trim_default_background() {
        let mut data = vec![255u8; 200 * 300];
        for y in 20..80 {
            for x in 10..60 {
                data[y * 200 + x] = 100;
            }
        }
        let im = gray(200, 300, data);
        assert_eq!(im.find_trim(None), (10, 20, 50, 60));
    }

    /// find_trim accepts an explicit background and returns a zero box
    /// when everything is background.
    #[test]
    fn find_trim_explicit_background_and_empty() {
        let mut data = vec![0u8; 50 * 50];
        data[25 * 50 + 30] = 200;
        let im = gray(50, 50, data);
        assert_eq!(im.find_trim(Some(&[0.0])), (30, 25, 1, 1));

        let blank = gray(10, 10, vec![255; 100]);
        assert_eq!(blank.find_trim(None), (0, 0, 0, 0));
    }

    /// Differences at or below the threshold of 10 are background.
    #[test]
    fn find_trim_threshold() {
        let mut data = vec![255u8; 10 * 10];
        data[5 * 10 + 5] = 246; // |246 - 255| = 9, below threshold
        let im = gray(10, 10, data);
        assert_eq!(im.find_trim(None), (0, 0, 0, 0));
    }

    /// A background vector of the wrong length is a typed error.
    #[test]
    fn find_trim_background_mismatch_is_typed_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_find_trim(Some(&[1.0, 2.0])),
            Err(ArithmeticError::ConstCountMismatch {
                expected: 1,
                got: 2
            })
        ));
    }

    // ---- profile / project ----

    /// profile returns first-non-zero positions: columns as a 1-row image,
    /// rows as a 1-column image, with the dimension as the all-zero value.
    #[test]
    fn profile_positions() {
        let mut data = vec![0u8; 100 * 100];
        data[50 * 100 + 40] = 100;
        let im = gray(100, 100, data);
        let (columns, rows) = im.profile();
        assert_eq!((columns.width(), columns.height()), (100, 1));
        assert_eq!((rows.width(), rows.height()), (1, 100));
        assert_eq!(columns.getpoint(40, 0), vec![50.0]);
        assert_eq!(columns.getpoint(0, 0), vec![100.0]); // all-zero column
        assert_eq!(rows.getpoint(0, 50), vec![40.0]);
        assert_eq!(rows.getpoint(0, 0), vec![100.0]); // all-zero row
        assert_eq!(columns.minpos(), (50.0, 40, 0));
        assert_eq!(rows.minpos(), (40.0, 0, 50));
    }

    /// project returns column and row sums per band.
    #[test]
    fn project_sums() {
        let mut data = vec![0u8; 100 * 50];
        for y in 0..50 {
            for x in 50..100 {
                data[y * 100 + x] = 10;
            }
        }
        let im = gray(100, 50, data);
        let (columns, rows) = im.project();
        assert_eq!(columns.getpoint(10, 0), vec![0.0]);
        assert_eq!(columns.getpoint(70, 0), vec![500.0]);
        assert_eq!(rows.getpoint(0, 10), vec![500.0]);
    }

    /**
     * Tests the carrier `profile` and `project` actually write, so the
     * doc's claims about them have a check behind them and the day a wider
     * carrier lands this goes red at the two places that must change.
     * Works by asserting the output `PixelFormat` of both ops for a 1-band
     * and a 3-band input, since the band count and the sample kind are
     * chosen separately.
     * Measured on vips 8.18.6: `profile` emits `VIPS_FORMAT_INT` for every
     * one of the eight input formats and `project` emits `UINT` for the
     * unsigned inputs, `INT` for the signed ones and `DOUBLE` for the float
     * ones, so both of these are deviations and neither is the `ushort` the
     * doc used to claim (issue #759).
     * Input: Gray8 and Rgb8 -> Output: Gray16 / Rgb16 from both ops.
     */
    #[test]
    fn profile_and_project_carry_16_bit_samples() {
        let one = gray(4, 3, vec![0, 0, 5, 0, 0, 7, 0, 0, 0, 0, 0, 0]);
        let (pc, pr) = one.profile();
        assert_eq!(pc.format(), PixelFormat::Gray16);
        assert_eq!(pr.format(), PixelFormat::Gray16);
        let (jc, jr) = one.project();
        assert_eq!(jc.format(), PixelFormat::Gray16);
        assert_eq!(jr.format(), PixelFormat::Gray16);

        let three = Raster::new(2, 2, PixelFormat::Rgb8, vec![1u8; 12]).unwrap();
        let (pc3, _) = three.profile();
        assert_eq!(pc3.format(), PixelFormat::Rgb16);
        let (jc3, _) = three.project();
        assert_eq!(jc3.format(), PixelFormat::Rgb16);
    }

    /**
     * Tests that `profile` saturates a position past 65535 rather than
     * wrapping, the deviation the 16-bit carrier forces.
     * Works by profiling a 1x65537 all-zero image, whose columns entry is
     * the height because the column never goes non-zero. The image is 64
     * KiB, so reaching the ceiling costs nothing.
     * Measured on vips 8.18.6, whose `INT` output holds the true value:
     * `vips profile` on this image reports `65537`. libviprs reports the
     * `65535` asserted here (issue #759).
     * Input: 1x65537 of zeros -> Output: columns(0,0) == 65535, not 65537.
     */
    #[test]
    fn profile_saturates_a_position_past_the_16_bit_ceiling() {
        let tall = gray(1, 65_537, vec![0u8; 65_537]);
        let (columns, _) = tall.profile();
        assert_eq!(columns.getpoint(0, 0), vec![65535.0]);
    }

    /// project saturates sums past 65535 rather than wrapping.
    #[test]
    fn project_saturates() {
        let im = gray(1, 300, vec![255; 300]); // column sum 76500 > 65535
        let (columns, _) = im.project();
        assert_eq!(columns.getpoint(0, 0), vec![65535.0]);
    }

    // ---- constant arithmetic ----

    /// add_const promotes 8-bit input to 16-bit so sums survive, and
    /// saturates negative results at zero.
    #[test]
    fn add_const_promotes() {
        let im = gray(2, 1, vec![200, 10]);
        let r = im.add_const(100.0);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![300.0]);
        assert_eq!(im.add_const(-20.0).getpoint(1, 0), vec![0.0]);
    }

    /// sub_const keeps the depth and saturates at zero.
    #[test]
    fn sub_const_saturates_at_zero() {
        let im = gray(2, 1, vec![5, 100]);
        let r = im.sub_const(10.0);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.getpoint(0, 0), vec![0.0]);
        assert_eq!(r.getpoint(1, 0), vec![90.0]);
    }

    /// mul_const promotes and rounds to nearest.
    #[test]
    fn mul_const_promotes_and_rounds() {
        let im = gray(2, 1, vec![200, 3]);
        let r = im.mul_const(2.0);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![400.0]);
        assert_eq!(im.mul_const(1.5).getpoint(1, 0), vec![5.0]); // 4.5 -> 5 (round half up)
    }

    /// div_const halves values into a float raster; a zero divisor
    /// produces zero, matching libvips.
    #[test]
    fn div_const_and_zero_divisor() {
        let im = gray(1, 1, vec![100]);
        let r = im.div_const(2.0);
        assert!(r.format().is_float());
        assert_eq!(r.getpoint(0, 0), vec![50.0]);
        assert_eq!(im.div_const(0.0).getpoint(0, 0), vec![0.0]);
    }

    /// The ported test_atanh scenario (libviprs-tests issue #77):
    /// div_const(255.0) on a 128 image keeps the float quotient
    /// (libvips promotes division to float), so atanh of the mid-range
    /// value is finite and correct instead of atanh(1) = inf.
    #[test]
    fn div_const_float_enables_atanh() {
        let im = gray(1, 1, vec![128]);
        let q = im.div_const(255.0);
        assert!(q.format().is_float());
        let v = float_samples(&q)[0];
        assert!((v - 128.0 / 255.0).abs() < 1e-6, "128/255 ~ 0.502, got {v}");
        let a = float_samples(&q.atanh())[0];
        let expected = (128.0f64 / 255.0).atanh(); // ~0.551924
        assert!(a.is_finite(), "atanh(0.502) is finite, got {a}");
        assert!((a - expected).abs() < 1e-6, "expected {expected}, got {a}");
    }

    /// floordiv_const floors the quotient into the integer container;
    /// div_const keeps the exact float quotient.
    #[test]
    fn floordiv_const_floors() {
        let im = gray(1, 1, vec![100]);
        assert_eq!(im.floordiv_const(3.0).getpoint(0, 0), vec![33.0]);
        let q = im.div_const(3.0).getpoint(0, 0)[0];
        assert!((q - 100.0 / 3.0).abs() < 1e-4, "100/3 stays 33.33, got {q}");
        let im9 = gray(1, 1, vec![9]);
        assert_eq!(im9.floordiv_const(5.0).getpoint(0, 0), vec![1.0]); // 1.8 floors to 1
        let q = im9.div_const(5.0).getpoint(0, 0)[0];
        assert!((q - 1.8).abs() < 1e-6, "9/5 stays 1.8, got {q}");
    }

    /// pow_const promotes so squares survive, and saturates at 65535.
    #[test]
    fn pow_const_promotes_and_saturates() {
        let im = gray(2, 1, vec![200, 3]);
        let r = im.pow_const(2.0);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![40_000.0]);
        assert_eq!(r.getpoint(1, 0), vec![9.0]);
        assert_eq!(im.pow_const(3.0).getpoint(0, 0), vec![65_535.0]);
    }

    /// rem_const computes the remainder; a zero divisor produces zero.
    #[test]
    fn rem_const_values() {
        let im = gray(3, 1, vec![7, 8, 9]);
        let r = im.rem_const(2.0);
        assert_eq!(r.getpoint(0, 0), vec![1.0]);
        assert_eq!(r.getpoint(1, 0), vec![0.0]);
        assert_eq!(im.rem_const(0.0).getpoint(2, 0), vec![0.0]);
    }

    /// linear computes a*x + b into a float raster; negative and
    /// fractional results survive instead of saturating or rounding.
    #[test]
    fn linear_values() {
        let im = gray(2, 1, vec![0, 100]);
        let r = im.linear(1.0, 10.0);
        assert!(r.format().is_float());
        assert_eq!(r.getpoint(0, 0), vec![10.0]);
        assert_eq!(r.getpoint(1, 0), vec![110.0]);
        assert!((r.avg() - 60.0).abs() < 1e-9);
        assert_eq!(im.linear(3.0, 5.0).getpoint(1, 0), vec![305.0]);
        // The float contract: no zero saturation, no rounding.
        assert_eq!(float_samples(&im.linear(1.0, -20.0)), vec![-20.0, 80.0]);
        assert_eq!(float_samples(&im.linear(1.0, 0.5)), vec![0.5, 100.5]);
    }

    /// linear_uchar is the caller-requests-integer form (the vips_linear
    /// uchar option): clipped into 0..=255, then truncated by the C
    /// float-to-uchar cast, accepting float input, so it casts a floated
    /// raster back to uchar.
    #[test]
    fn linear_uchar_truncates_and_saturates() {
        let im = gray(3, 1, vec![10, 200, 0]);
        let r = im.linear_uchar(2.0, -20.0);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.data(), &[0, 255, 0]); // 0 exact, 380 clips, -20 clips

        // Truncation, not rounding: 9 * 0.2 = 1.8 casts to 1.
        assert_eq!(gray(1, 1, vec![9]).linear_uchar(0.2, 0.0).data(), &[1]);

        // Round-trip a floated quotient back to uchar.
        let back = gray(1, 1, vec![128])
            .div_const(255.0)
            .linear_uchar(255.0, 0.0);
        assert_eq!(back.data(), &[128]);
    }

    /// add_vec applies one constant per band; a wrong-length vector is a
    /// typed error.
    #[test]
    fn add_vec_per_band_and_error() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        let r = im.add_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(r.getpoint(0, 0), vec![11.0, 22.0, 33.0]);
        assert!(matches!(
            im.try_add_vec(&[1.0]),
            Err(ArithmeticError::ConstCountMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    /// sub_vec and mul_vec apply per-band constants with the documented
    /// integer saturation / promotion; div_vec floats per band, with the
    /// zero-divisor rule.
    #[test]
    fn other_vec_ops() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        assert_eq!(
            im.sub_vec(&[5.0, 25.0, 10.0]).getpoint(0, 0),
            vec![5.0, 0.0, 20.0]
        );
        let r = im.mul_vec(&[1.0, 2.0, 30.0]);
        assert_eq!(r.format(), PixelFormat::Rgb16);
        assert_eq!(r.getpoint(0, 0), vec![10.0, 40.0, 900.0]);
        let d = im.div_vec(&[2.0, 0.0, 3.0]);
        assert!(d.format().is_float());
        let d = d.getpoint(0, 0);
        assert_eq!(&d[..2], &[5.0, 0.0]);
        assert!((d[2] - 10.0).abs() < 1e-6);
        // Wrong-length vectors stay typed errors on the float path.
        assert!(matches!(
            im.try_div_vec(&[1.0]),
            Err(ArithmeticError::ConstCountMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    /// The libvips promotion table for the linear / divide family:
    /// divide always outputs float (the `vips_divide` table maps every
    /// integer format to float); linear outputs float (`vips_linear`)
    /// unless the caller requests uchar; image-image `sub` outputs float
    /// (`vips_subtract` promotes `uchar` to signed `short`, carried here
    /// as float, issue #282); add / multiply and the constant `sub_const`
    /// / `sub_vec` keep the integer contract.
    #[test]
    fn linear_divide_promotion_table() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let im8 = gray(1, 1, vec![128]);
        let im16 = gray16(1, 1, &[1000]);

        // Divide family: always float.
        assert_eq!(im8.div_const(2.0).format(), f1);
        assert_eq!(im16.div_const(2.0).format(), f1);
        assert_eq!(im8.div(&im8).format(), f1);
        assert_eq!(im8.div_vec(&[2.0]).format(), f1);
        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
        assert_eq!(
            rgb.div_const(2.0).format(),
            PixelFormat::with_kind(3, SampleKind::F32).unwrap()
        );

        // linear: float unless uchar is requested.
        assert_eq!(im8.linear(1.0, 0.0).format(), f1);
        assert_eq!(im16.linear(1.0, 0.0).format(), f1);
        assert_eq!(im8.linear_uchar(1.0, 0.0).format(), PixelFormat::Gray8);

        // Image-image sub: float (signed short in libvips, issue #282).
        assert_eq!(im8.sub(&im8).format(), f1);
        assert_eq!(im16.sub(&im16).format(), f1);

        // Integer contract unchanged everywhere libvips keeps integer.
        assert_eq!(im8.add_const(1.0).format(), PixelFormat::Gray16);
        assert_eq!(im8.sub_const(1.0).format(), PixelFormat::Gray8);
        assert_eq!(im8.mul_const(2.0).format(), PixelFormat::Gray16);
        assert_eq!(im8.add_vec(&[1.0]).format(), PixelFormat::Gray16);
        assert_eq!(im8.sub_vec(&[1.0]).format(), PixelFormat::Gray8);
        assert_eq!(im8.mul_vec(&[2.0]).format(), PixelFormat::Gray16);
        assert_eq!(im8.mul(&im8).format(), PixelFormat::Gray16);
        assert_eq!(im8.floordiv_const(2.0).format(), PixelFormat::Gray8);
        assert_eq!(im8.rem_const(2.0).format(), PixelFormat::Gray8);
        assert_eq!(im8.pow_const(2.0).format(), PixelFormat::Gray16);
    }

    /// The linear / divide family accepts float input now that it writes
    /// float output, so chains like div_const(...).linear(...) work.
    #[test]
    fn linear_divide_accept_float_input() {
        let im = grayf(1, 1, &[0.5]);
        assert_eq!(float_samples(&im.div_const(2.0)), vec![0.25]);
        assert_eq!(float_samples(&im.linear(2.0, 1.0)), vec![2.0]);
        assert_eq!(float_samples(&im.div(&im)), vec![1.0]);
        assert_eq!(float_samples(&im.div_vec(&[4.0])), vec![0.125]);
    }

    // ---- unary shape ops ----

    /// pos, abs, floor, ceil, and rint are identities on the unsigned
    /// integer formats.
    #[test]
    fn identity_ops() {
        let im = gray(2, 1, vec![7, 200]);
        for r in [im.pos(), im.abs(), im.floor(), im.ceil(), im.rint()] {
            assert_eq!(r.format(), im.format());
            assert_eq!(r.data(), im.data());
        }
    }

    /// sign maps zero to 0 and positive samples to 1, keeping the depth.
    #[test]
    fn sign_values() {
        let im = gray(3, 1, vec![0, 1, 200]);
        let r = im.sign();
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.data(), &[0, 1, 1]);
        let im16 = gray16(2, 1, &[0, 4096]);
        assert_eq!(im16.sign().getpoint(1, 0), vec![1.0]);
    }

    /// clamp defaults to [0, 1] and accepts explicit bounds.
    #[test]
    fn clamp_bounds() {
        let im = gray(3, 1, vec![0, 1, 200]);
        let r = im.clamp(None, None);
        assert!(r.max() <= 1.0);
        assert!(r.min() >= 0.0);
        let r = im.clamp(Some(14.0), Some(45.0));
        assert_eq!(r.data(), &[14, 14, 45]);
    }

    /// clamp panics when the bounds are inverted.
    #[test]
    #[should_panic(expected = "clamp: min bound")]
    fn clamp_inverted_bounds_panics() {
        let _ = gray(1, 1, vec![0]).clamp(Some(2.0), Some(1.0));
    }

    // ---- image-image arithmetic ----

    /// sub of an image from itself is all zeros; negative differences
    /// survive as signed float values instead of saturating to `0`
    /// (issue #282, matching the `vips_subtract` promotion to signed short).
    #[test]
    fn sub_zeros_and_signed_difference() {
        let a = gray(2, 1, vec![100, 10]);
        let z = a.sub(&a);
        assert!(z.format().is_float());
        assert_eq!(z.avg(), 0.0);
        let b = gray(2, 1, vec![50, 200]);
        assert_eq!(a.sub(&b).getpoint(0, 0), vec![50.0]);
        // 10 - 200 = -190 survives instead of the pre-#282 saturated `0`.
        assert_eq!(a.sub(&b).getpoint(1, 0), vec![-190.0]);
    }

    /// mul promotes 8-bit products to 16-bit.
    #[test]
    fn mul_promotes() {
        let a = gray(1, 1, vec![200]);
        let r = a.mul(&a);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![40_000.0]);
    }

    /// div of an image by itself is 1 for non-zero samples and 0 where the
    /// divisor is zero.
    #[test]
    fn div_self_and_zero() {
        let a = gray(2, 1, vec![100, 0]);
        let r = a.div(&a);
        assert_eq!(r.getpoint(0, 0), vec![1.0]);
        assert_eq!(r.getpoint(1, 0), vec![0.0]);
    }

    /// The shared `remainder_vips` kernel is TRUNCATED, matching libvips
    /// `IREMAINDER`, which is the only branch the crate's unsigned integer
    /// carriers can reach.
    ///
    /// Nothing else in the suite can tell truncated from floored: the two
    /// agree on every non-negative operand pair, so the image-image form is
    /// blind to the choice and swapping the body leaves its tests green.
    /// This pins the definition directly, on the operands where the two
    /// split. Rust's `%` on `f64` truncates exactly as C's does on integers,
    /// which is what `IREMAINDER` runs.
    #[test]
    fn remainder_vips_is_truncated_matching_iremainder() {
        // Negative divisor: truncated keeps the sign of the dividend. This
        // is the one split reachable today, through `rem_const`. Floored
        // would give -2, -1, 0 here; vips 8.18.4 measures 1, 2, 0 on a uchar
        // carrier, which is what `IREMAINDER` and this kernel produce.
        assert_eq!(remainder_vips(7.0, -3.0), 1.0);
        assert_eq!(remainder_vips(20.0, -3.0), 2.0);
        assert_eq!(remainder_vips(30.0, -3.0), 0.0);
        // Negative dividend: unreachable on an unsigned carrier, pinned so
        // the definition is unambiguous either way. Floored would give 2.
        assert_eq!(remainder_vips(-7.0, 3.0), -1.0);
        // Non-negative operands, i.e. everything the image-image form can
        // see: truncated and floored agree, so these hold under both.
        assert_eq!(remainder_vips(7.0, 3.0), 1.0);
        assert_eq!(remainder_vips(3.0, 7.0), 3.0);
        assert_eq!(remainder_vips(0.0, 7.0), 0.0);
        // A zero divisor short-circuits before the division, so `0 / 0`
        // never forms, and `-0.0 == 0.0` catches the negative zero too.
        assert_eq!(remainder_vips(7.0, 0.0), 0.0);
        assert_eq!(remainder_vips(0.0, 0.0), 0.0);
        assert_eq!(remainder_vips(7.0, -0.0), 0.0);
    }

    /// rem_const shares the truncating `remainder_vips` kernel with the
    /// image-image form, and truncating is what libvips runs for an integer
    /// input, so a negative constant matches vips rather than diverging.
    ///
    /// Measured against vips 8.18.4 on uchar `[7,20,30]`:
    /// `vips remainder_const a out -- -3` gives `[1,2,0]`. The same input
    /// cast to `float` first gives `[-2,-1,0]`, because `remainder_const`
    /// dispatches on format and the float path floors. That is the branch a
    /// future float carrier will need, and it is not the branch this is.
    ///
    /// `c = 3` happens to give the same `[1,2,0]`, which is a coincidence of
    /// these operands, not the point: the negative case is the one where the
    /// two definitions could have disagreed.
    #[test]
    fn rem_const_negative_divisor_matches_the_vips_integer_branch() {
        let a = gray(3, 1, vec![7, 20, 30]);
        assert_eq!(a.rem_const(-3.0).data().to_vec(), vec![1, 2, 0]);
        assert_eq!(a.rem_const(3.0).data().to_vec(), vec![1, 2, 0]);
    }

    /// The measured vips 8.18.4 oracle for the 2-image remainder:
    /// `a = uchar [[10,20,30],[40,50,60]]`, `b = uchar` all 7, and
    /// `vips remainder a b` gives a uchar raster holding `[3,6,2,5,1,4]`.
    /// The output format is the identity promotion (`remainder.c:173-178`)
    /// applied after formatalike, i.e. the wider of the two input depths.
    #[test]
    fn remainder_matches_vips_oracle() {
        let a = gray(3, 2, vec![10, 20, 30, 40, 50, 60]);
        let b = gray(3, 2, vec![7; 6]);
        let r = a.remainder(&b);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.data().to_vec(), vec![3, 6, 2, 5, 1, 4]);
    }

    /// A zero divisor gives `0`, the crate-wide `x % 0 == 0` convention the
    /// module header states and `rem_const` already follows.
    ///
    /// This is a DELIBERATE divergence from vips, which writes `-1`
    /// (`remainder.c:101`: `q[x] = p2[x] ? p1[x] % p2[x] : -1;`, and again at
    /// `remainder.c:116` in the float branch). Measured
    /// with divisor `[[0,7,0],[7,0,7]]`, vips 8.18.4 gives
    /// `[255,6,255,5,255,4]` — the `-1` read back through the uchar carrier.
    /// libviprs has no signed carrier, so `-1` is unrepresentable here.
    #[test]
    fn remainder_by_zero_is_zero_not_the_vips_minus_one() {
        let a = gray(3, 2, vec![10, 20, 30, 40, 50, 60]);
        let b = gray(3, 2, vec![0, 7, 0, 7, 0, 7]);
        assert_eq!(a.remainder(&b).data().to_vec(), vec![0, 6, 0, 5, 0, 4]);
    }

    /// Format promotion is the identity table applied to the formatalike
    /// result, so the output depth is the wider input depth: `uchar %
    /// uchar` stays 8-bit and `uchar % ushort` promotes to 16-bit.
    #[test]
    fn remainder_promotes_to_the_wider_depth() {
        let a = gray(2, 1, vec![10, 200]);
        let narrow = a.remainder(&gray(2, 1, vec![7, 7]));
        assert_eq!(narrow.format(), PixelFormat::Gray8);
        assert_eq!(narrow.getpoint(0, 0), vec![3.0]);

        let wide = a.remainder(&gray16(2, 1, &[7, 300]));
        assert_eq!(wide.format(), PixelFormat::Gray16);
        assert_eq!(wide.getpoint(0, 0), vec![3.0]);
        // 200 % 300 == 200: the dividend survives a larger divisor.
        assert_eq!(wide.getpoint(1, 0), vec![200.0]);
    }

    /// remainder requires exact dimension and band-count equality (no
    /// bandalike, no sizealike — see [`Raster::try_remainder`]) and rejects
    /// float operands on either side, all as typed errors.
    #[test]
    fn remainder_typed_errors() {
        let a = gray(2, 1, vec![10, 20]);
        assert!(matches!(
            a.try_remainder(&gray(1, 1, vec![7])),
            Err(ArithmeticError::DimensionMismatch {
                expected_w: 2,
                expected_h: 1,
                got_w: 1,
                got_h: 1
            })
        ));
        let rgb = Raster::new(2, 1, PixelFormat::Rgb8, vec![7; 6]).unwrap();
        assert!(matches!(
            a.try_remainder(&rgb),
            Err(ArithmeticError::BandCountMismatch {
                expected: 1,
                got: 3
            })
        ));

        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let f = Raster::zeroed(2, 1, f1).unwrap();
        assert!(matches!(
            f.try_remainder(&a),
            Err(ArithmeticError::FloatUnsupported { op: "remainder" })
        ));
        assert!(matches!(
            a.try_remainder(&f),
            Err(ArithmeticError::FloatUnsupported { op: "remainder" })
        ));
    }

    /// Binary ops reject mismatched dimensions and band counts with typed
    /// errors, and the panicking surface reports the op name.
    #[test]
    fn binary_typed_errors() {
        let a = gray(2, 1, vec![0, 0]);
        let b = gray(1, 1, vec![0]);
        assert!(matches!(
            a.try_sub(&b),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
        let rgb = Raster::new(2, 1, PixelFormat::Rgb8, vec![0; 6]).unwrap();
        assert!(matches!(
            a.try_mul(&rgb),
            Err(ArithmeticError::BandCountMismatch {
                expected: 1,
                got: 3
            })
        ));
    }

    /// The panicking surface reports the op name and the typed message.
    #[test]
    #[should_panic(expected = "sub: dimension mismatch")]
    fn sub_panicking_surface_message() {
        let a = gray(2, 1, vec![0, 0]);
        let b = gray(1, 1, vec![0]);
        let _ = a.sub(&b);
    }

    /// minpair / maxpair take the samplewise extremum; mixed depths
    /// promote numerically to 16-bit.
    #[test]
    fn minpair_maxpair() {
        let a = gray(2, 1, vec![100, 5]);
        let b = gray(2, 1, vec![50, 60]);
        assert_eq!(a.minpair(&b).getpoint(0, 0), vec![50.0]);
        assert_eq!(a.minpair(&b).getpoint(1, 0), vec![5.0]);
        assert_eq!(a.maxpair(&b).getpoint(0, 0), vec![100.0]);
        assert_eq!(a.maxpair(&b).getpoint(1, 0), vec![60.0]);

        let wide = gray16(2, 1, &[4096, 2]);
        let r = a.maxpair(&wide);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![4096.0]);
        assert_eq!(r.getpoint(1, 0), vec![5.0]);
    }

    /// sum adds a list of images with 16-bit promotion; empty lists and
    /// mismatched images are typed errors.
    #[test]
    fn sum_list() {
        let images: Vec<Raster> = (0..10u8).map(|x| gray(2, 2, vec![x * 10; 4])).collect();
        let refs: Vec<&Raster> = images.iter().collect();
        let result = Raster::sum(&refs);
        assert_eq!(result.format(), PixelFormat::Gray16);
        assert_eq!(result.max(), 450.0);

        assert!(matches!(
            Raster::try_sum(&[]),
            Err(ArithmeticError::EmptyImageList)
        ));
        let odd = gray(1, 1, vec![0]);
        assert!(matches!(
            Raster::try_sum(&[&images[0], &odd]),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
    }

    /// sum saturates at 65535.
    #[test]
    fn sum_saturates() {
        let a = gray16(1, 1, &[60_000]);
        let b = gray16(1, 1, &[10_000]);
        assert_eq!(Raster::sum(&[&a, &b]).getpoint(0, 0), vec![65_535.0]);
    }

    // ---- comparisons ----

    /// The relational family produces 8-bit 0/255 masks with the expected
    /// truth tables.
    #[test]
    fn comparison_masks() {
        let a = gray(2, 1, vec![10, 200]);
        let b = gray(2, 1, vec![10, 100]);
        assert_eq!(a.more_than(&b).data(), &[0, 255]);
        assert_eq!(a.more_eq(&b).data(), &[255, 255]);
        assert_eq!(a.less_than(&b).data(), &[0, 0]);
        assert_eq!(a.less_eq(&b).data(), &[255, 0]);
        assert_eq!(a.equal(&b).data(), &[255, 0]);
        assert_eq!(a.noteq(&b).data(), &[0, 255]);
    }

    /// Constant comparisons produce the same masks against a scalar, and
    /// self-comparisons degenerate correctly.
    #[test]
    fn comparison_const_masks() {
        let im = gray(3, 1, vec![50, 100, 150]);
        assert_eq!(im.more_than_const(100.0).data(), &[0, 0, 255]);
        assert_eq!(im.more_eq_const(100.0).data(), &[0, 255, 255]);
        assert_eq!(im.less_than_const(100.0).data(), &[255, 0, 0]);
        assert_eq!(im.less_eq_const(100.0).data(), &[255, 255, 0]);
        assert_eq!(im.equal_const(100.0).data(), &[0, 255, 0]);
        assert_eq!(im.noteq_const(100.0).data(), &[255, 0, 255]);

        assert_eq!(im.more_than(&im).avg(), 0.0);
        assert_eq!(im.more_eq(&im).avg(), 255.0);
        assert_eq!(im.less_than(&im).avg(), 0.0);
        assert_eq!(im.less_eq(&im).avg(), 255.0);
        assert_eq!(im.equal(&im).avg(), 255.0);
        assert_eq!(im.noteq(&im).avg(), 0.0);
    }

    /// equal_const is exact: out-of-range and fractional constants match
    /// nothing.
    #[test]
    fn equal_const_exactness() {
        let ramp: Vec<u8> = (0..=255).collect();
        let im = gray(256, 1, ramp);
        assert!(im.equal_const(1000.0).max() < 1.0);
        assert_eq!(im.equal_const(12.0).max(), 255.0);
        assert!(im.equal_const(12.5).max() < 1.0);
    }

    /// Comparisons on 16-bit input still produce 8-bit masks.
    #[test]
    fn comparison_16bit_input_masks_are_8bit() {
        let im = gray16(2, 1, &[4096, 100]);
        let r = im.more_than_const(1000.0);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.data(), &[255, 0]);
    }

    // ---- bitwise ----

    /// Image and constant bitwise ops match the sample-wise integer ops.
    #[test]
    fn bitwise_values() {
        let a = gray(1, 1, vec![0xCC]);
        let b = gray(1, 1, vec![0xAA]);
        assert_eq!(a.bitand(&b).data(), &[0x88]);
        assert_eq!(a.bitor(&b).data(), &[0xEE]);
        assert_eq!(a.bitxor(&b).data(), &[0x66]);
        assert_eq!(a.bitand_const(0x0F).data(), &[0x0C]);
        assert_eq!(a.bitor_const(0x0F).data(), &[0xCF]);
        assert_eq!(a.bitxor_const(0xFF).data(), &[0x33]);
        assert_eq!(a.bitnot().data(), &[0x33]);
    }

    /// AND with itself is identity, AND with 0 is zero, OR with 0xFF is
    /// 255 (the ported truth checks).
    #[test]
    fn bitwise_ported_identities() {
        let im = gray(2, 1, vec![0xF0, 0x0F]);
        assert_eq!(im.bitand(&im).data(), im.data());
        assert_eq!(im.bitand_const(0).avg(), 0.0);
        assert_eq!(im.bitor_const(0xFF).avg(), 255.0);
        assert_eq!(im.bitxor(&im).avg(), 0.0);
    }

    /// A negative constant masks to all ones in the sample depth.
    #[test]
    fn bitwise_negative_constant_is_all_ones() {
        let im = gray(1, 1, vec![0xA5]);
        assert_eq!(im.bitand_const(-1).data(), &[0xA5]);
        let im16 = gray16(1, 1, &[0x0F0F]);
        assert_eq!(im16.bitand_const(-1).getpoint(0, 0), vec![0x0F0F as f64]);
    }

    /// Shifts truncate into the depth; oversized shift counts produce 0.
    #[test]
    fn shift_values() {
        let im = gray(2, 1, vec![3, 0xF0]);
        assert_eq!(im.lshift(2).data(), &[12, 0xC0]); // 0xF0 << 2 truncates to 0xC0
        assert_eq!(im.rshift(2).data(), &[0, 0x3C]);
        assert_eq!(im.lshift(40).data(), &[0, 0]);
        assert_eq!(im.rshift(40).data(), &[0, 0]);
        let im16 = gray16(1, 1, &[0x0100]);
        assert_eq!(im16.lshift(4).getpoint(0, 0), vec![0x1000 as f64]);
    }

    /// bitnot on 16-bit input inverts within 16 bits.
    #[test]
    fn bitnot_16bit() {
        let im = gray16(1, 1, &[0x0F0F]);
        assert_eq!(im.bitnot().getpoint(0, 0), vec![0xF0F0 as f64]);
    }

    // ---- scaleimage ----

    /// scaleimage maps the global min to 0 and max to 255 and always
    /// outputs 8-bit.
    #[test]
    fn scaleimage_linear() {
        let im = gray16(3, 1, &[100, 300, 500]);
        let r = im.scaleimage(None);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.getpoint(0, 0), vec![0.0]);
        assert_eq!(r.getpoint(1, 0), vec![128.0]); // 127.5 rounds up
        assert_eq!(r.getpoint(2, 0), vec![255.0]);
    }

    /// scaleimage log mode still peaks at 255 and is monotone.
    #[test]
    fn scaleimage_log() {
        let im = gray(3, 1, vec![0, 10, 200]);
        let r = im.scaleimage(Some(true));
        assert_eq!(r.getpoint(2, 0), vec![255.0]);
        let v0 = r.getpoint(0, 0)[0];
        let v1 = r.getpoint(1, 0)[0];
        assert!(v0 < v1 && v1 < 255.0);
    }

    /// A constant image scales to all zeros instead of dividing by zero.
    #[test]
    fn scaleimage_constant_input() {
        let im = gray(2, 2, vec![7; 4]);
        assert_eq!(im.scaleimage(None).max(), 0.0);
        let black = gray(2, 2, vec![0; 4]);
        assert_eq!(black.scaleimage(Some(true)).max(), 0.0);
    }

    /// scaleimage keeps the band count.
    #[test]
    fn scaleimage_multiband() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![0, 100, 200]).unwrap();
        let r = im.scaleimage(None);
        assert_eq!(r.format(), PixelFormat::Rgb8);
        assert_eq!(r.getpoint(0, 0), vec![0.0, 128.0, 255.0]);
    }

    // ---- stdif ----

    /// stdif keeps dimensions and format and pulls the mean toward the
    /// target mean of 128.
    #[test]
    fn stdif_shifts_mean_toward_target() {
        // A gradient image with mean far below 128.
        let mut data = vec![0u8; 60 * 40];
        for (i, d) in data.iter_mut().enumerate() {
            *d = (i % 60) as u8;
        }
        let im = gray(60, 40, data);
        let r = im.stdif(10, 10);
        assert_eq!((r.width(), r.height()), (im.width(), im.height()));
        assert_eq!(r.format(), im.format());
        let orig_dist = (im.avg() - 128.0).abs();
        let new_dist = (r.avg() - 128.0).abs();
        assert!(
            new_dist < orig_dist,
            "stdif should shift mean toward 128: orig {orig_dist}, new {new_dist}"
        );
    }

    /// On a constant image the deviation term vanishes and every pixel
    /// becomes a*m0 + (1-a)*mean.
    #[test]
    fn stdif_constant_image() {
        let im = gray(20, 20, vec![40; 400]);
        let r = im.stdif(5, 5);
        // 0.5 * 128 + 0.5 * 40 = 84.
        assert_eq!(r.getpoint(10, 10), vec![84.0]);
    }

    /// stdif processes bands independently. A window equal to the image is
    /// the largest vips accepts; on a constant image the deviation term
    /// vanishes so every band is `0.5*128 + 0.5*band`.
    #[test]
    fn stdif_multiband() {
        let im = Raster::new(2, 2, PixelFormat::Rgb8, [10u8, 40, 70].repeat(4)).unwrap();
        let r = im.stdif(2, 2);
        assert_eq!(r.format(), PixelFormat::Rgb8);
        // Constant bands: 0.5*128 + 0.5*band.
        assert_eq!(r.getpoint(0, 0), vec![69.0, 84.0, 99.0]);
    }

    /// A window larger than the image is a typed error, matching vips, which
    /// rejects it as `stdif: window too large` (verified with the oracle: on a
    /// 5-wide image, window 5 is accepted but 6+ errors). The core previously
    /// computed a silent result for any window; #490 aligns it with vips.
    #[test]
    fn stdif_window_larger_than_image_is_typed_error() {
        let im = Raster::new(2, 2, PixelFormat::Rgb8, [10u8, 40, 70].repeat(4)).unwrap();
        assert!(matches!(
            im.try_stdif(3, 2),
            Err(ArithmeticError::WindowTooLarge { .. })
        ));
        assert!(matches!(
            im.try_stdif(2, 3),
            Err(ArithmeticError::WindowTooLarge { .. })
        ));
        // A window exactly the image size is still accepted.
        assert!(im.try_stdif(2, 2).is_ok());
    }

    /// Edge pixels use a replicated (edge-extended) window, matching vips.
    /// vips embeds the input with a `window/2` replicated border before taking
    /// window statistics; the old code clipped the window at the edge, which
    /// diverged in a `window/2`-wide border. Pinned against the oracle:
    /// `vips stdif` of `[0,40,80,120,200]` (uchar) with a `3x1` window yields
    /// `[65,84,104,126,160]`, and with a `5x1` window on
    /// `[3,200,17,250,99,5,128,64,33,210]` yields
    /// `[75,137,96,156,114,90,118,101,101,155]`.
    #[test]
    fn stdif_border_replicates_like_vips() {
        let im = gray(5, 1, vec![0, 40, 80, 120, 200]);
        let r = im.stdif(3, 1);
        let got: Vec<f64> = (0..5).map(|x| r.getpoint(x, 0)[0]).collect();
        assert_eq!(got, vec![65.0, 84.0, 104.0, 126.0, 160.0]);

        let im = gray(10, 1, vec![3, 200, 17, 250, 99, 5, 128, 64, 33, 210]);
        let r = im.stdif(5, 1);
        let got: Vec<f64> = (0..10).map(|x| r.getpoint(x, 0)[0]).collect();
        assert_eq!(
            got,
            vec![
                75.0, 137.0, 96.0, 156.0, 114.0, 90.0, 118.0, 101.0, 101.0, 155.0
            ]
        );
    }

    /// The 2D window replicates on all four edges. Pinned against the oracle:
    /// `vips stdif` of the 3x3 `[0,40,80; 120,160,200; 30,60,90]` (uchar) with
    /// a `3x3` window yields `[74,92,109; 114,130,148; 86,100,113]`.
    #[test]
    fn stdif_border_replicates_2d() {
        let im = gray(3, 3, vec![0, 40, 80, 120, 160, 200, 30, 60, 90]);
        let r = im.stdif(3, 3);
        let got: Vec<f64> = (0..3)
            .flat_map(|y| (0..3).map(move |x| (x, y)))
            .map(|(x, y)| r.getpoint(x, y)[0])
            .collect();
        assert_eq!(
            got,
            vec![74.0, 92.0, 109.0, 114.0, 130.0, 148.0, 86.0, 100.0, 113.0]
        );
    }

    /// A zero window dimension is a typed error.
    #[test]
    fn stdif_zero_window_is_typed_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_stdif(0, 5),
            Err(ArithmeticError::ZeroWindow)
        ));
        assert!(matches!(
            im.try_stdif(5, 0),
            Err(ArithmeticError::ZeroWindow)
        ));
    }

    // ---- recomb ----

    /// recomb with one row produces the weighted band sum as a single-band
    /// image.
    #[test]
    fn recomb_single_row() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![100, 50, 10]).unwrap();
        let matrix: &[&[f64]] = &[&[0.2, 0.5, 0.3]];
        let r = im.recomb(matrix);
        assert_eq!(r.format(), PixelFormat::Gray8);
        assert_eq!(r.getpoint(0, 0), vec![48.0]); // 20 + 25 + 3
    }

    /// recomb with the identity matrix reproduces the image; a swap matrix
    /// reorders bands, and row count sets the output band count.
    #[test]
    fn recomb_identity_and_swap() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        let identity: &[&[f64]] = &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
        assert_eq!(im.recomb(identity).data(), im.data());

        let swap: &[&[f64]] = &[&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]];
        let r = im.recomb(swap);
        assert_eq!(r.format().channels(), 2);
        assert_eq!(r.getpoint(0, 0), vec![30.0, 10.0]);
    }

    /// recomb saturates results into the input depth.
    #[test]
    fn recomb_saturates() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![200, 200, 200]).unwrap();
        let matrix: &[&[f64]] = &[&[1.0, 1.0, 1.0]];
        assert_eq!(im.recomb(matrix).getpoint(0, 0), vec![255.0]);
    }

    /// recomb truncates the float accumulator toward zero, matching vips's
    /// float-then-cast rather than round-to-nearest. vips `recomb` produces a
    /// float image and the differential cast to the input depth truncates
    /// (verified with the oracle: `vips cast` of `6.5 -> 6`, `6.6 -> 6`). A
    /// fractional result of `6.5` therefore becomes `6`, where the old
    /// round-to-nearest gave `7` — a 1-LSB divergence.
    #[test]
    fn recomb_truncates_like_vips() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        // 0.1*10 + 0.1*20 + 0.1166667*30 = 6.5 -> truncates to 6.
        let m65: &[&[f64]] = &[&[0.1, 0.1, 0.1166667]];
        assert_eq!(im.recomb(m65).getpoint(0, 0), vec![6.0]);
        // 0.1*10 + 0.1*20 + 0.12*30 = 6.6 -> truncates to 6.
        let m66: &[&[f64]] = &[&[0.1, 0.1, 0.12]];
        assert_eq!(im.recomb(m66).getpoint(0, 0), vec![6.0]);
        // A negative accumulator clamps to 0 (not wrapping).
        let mneg: &[&[f64]] = &[&[-1.0, 0.0, 0.0]];
        assert_eq!(im.recomb(mneg).getpoint(0, 0), vec![0.0]);
    }

    /// vips stores the recomb accumulator as float32 before the cast truncates,
    /// so an f64 accumulator that lands just below an integer inside the f32
    /// round-up band rounds *up* first. Pinned against the oracle: input
    /// `[10,20,30]` with coeff row `[0,0,0.23333332933333335]` gives an f64
    /// accumulator of `6.99999988`, which `vips recomb` stores as f32 `7.0` and
    /// `vips cast`->uchar yields `7` (not the `6` a direct f64 truncation would
    /// give). Regression for #491.
    #[test]
    fn recomb_rounds_to_f32_before_truncating_like_vips() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        // f64 acc = 6.99999988; f32 storage -> 7.0; cast -> 7.
        let m: &[&[f64]] = &[&[0.0, 0.0, 0.233_333_329_333_333_35]];
        assert_eq!(im.recomb(m).getpoint(0, 0), vec![7.0]);
    }

    /// Malformed recomb matrices are typed errors.
    #[test]
    fn recomb_typed_errors() {
        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![0, 0, 0]).unwrap();
        assert!(matches!(
            im.try_recomb(&[]),
            Err(ArithmeticError::EmptyMatrix)
        ));
        let bad: &[&[f64]] = &[&[1.0, 2.0]];
        assert!(matches!(
            im.try_recomb(bad),
            Err(ArithmeticError::MatrixRowMismatch {
                row: 0,
                expected: 3,
                got: 2
            })
        ));
    }

    // ---- premultiply / unpremultiply ----

    /// premultiply scales colour bands by alpha/max and keeps alpha.
    #[test]
    fn premultiply_values() {
        let im = Raster::new(1, 1, PixelFormat::Rgba8, vec![100, 200, 50, 127]).unwrap();
        let r = im.premultiply();
        assert_eq!(r.format(), PixelFormat::Rgba8);
        // v * 127 / 255, rounded.
        assert_eq!(r.getpoint(0, 0), vec![50.0, 100.0, 25.0, 127.0]);
    }

    /// unpremultiply inverts premultiply within rounding error and maps
    /// zero alpha to zero.
    #[test]
    fn unpremultiply_round_trip_and_zero_alpha() {
        let im = Raster::new(
            2,
            1,
            PixelFormat::Rgba8,
            vec![100, 200, 50, 127, 90, 90, 90, 0],
        )
        .unwrap();
        let round = im.premultiply().unpremultiply();
        let orig = im.getpoint(0, 0);
        let got = round.getpoint(0, 0);
        for (o, g) in orig.iter().zip(got.iter()) {
            assert!((o - g).abs() < 2.0, "expected {o}, got {g}");
        }
        // Zero alpha: premultiplied colour is 0 and stays 0.
        assert_eq!(round.getpoint(1, 0), vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// The alpha ops treat the last band as alpha for any band count >= 2
    /// and work at 16-bit depth.
    #[test]
    fn premultiply_two_band_and_16bit() {
        let im = Raster::new(
            1,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            vec![100, 128],
        )
        .unwrap();
        let r = im.premultiply();
        assert_eq!(r.getpoint(0, 0), vec![50.0, 128.0]); // 100*128/255 = 50.2

        let vals = [40_000u16, 32_768u16];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let im16 = Raster::new(
            1,
            1,
            PixelFormat::with_kind(2, SampleKind::U16).unwrap(),
            data,
        )
        .unwrap();
        let r16 = im16.premultiply();
        assert_eq!(r16.getpoint(0, 0), vec![20_000.0, 32_768.0]); // 40000*32768/65535 ~ 20000.3
    }

    /// The un-premultiply dead zone (#604) is an absolute `0.01` in sample
    /// units, so on an unsigned carrier it can only ever catch `alpha == 0`.
    /// Scaling it by `max` would be the natural-looking mistake and would
    /// wrongly damp the smallest real alphas; this pins that it does not.
    /// Measured on vips 8.18.4, a 3x1 uchar RGBA of `(1,1,1,alpha)`:
    ///
    /// ```text
    /// vips unpremultiply tiny.png t.v ; vips getpoint t.v <x> 0
    ///   alpha = 1  ->  255 255 255 1
    ///   alpha = 3  ->   85  85  85 3
    ///   alpha = 0  ->    0   0   0 0
    /// ```
    #[test]
    fn unpremultiply_dead_zone_never_catches_a_real_integer_alpha() {
        let im = Raster::new(
            3,
            1,
            PixelFormat::Rgba8,
            vec![1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 0],
        )
        .unwrap();
        let out = im.unpremultiply();
        assert_eq!(out.getpoint(0, 0), vec![255.0, 255.0, 255.0, 1.0]);
        assert_eq!(out.getpoint(1, 0), vec![85.0, 85.0, 85.0, 3.0]);
        assert_eq!(out.getpoint(2, 0), vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// A single-band image has no alpha band: typed error.
    #[test]
    fn premultiply_single_band_is_typed_error() {
        let im = gray(1, 1, vec![0]);
        assert!(matches!(
            im.try_premultiply(),
            Err(ArithmeticError::NoAlphaBand { bands: 1 })
        ));
        assert!(matches!(
            im.try_unpremultiply(),
            Err(ArithmeticError::NoAlphaBand { bands: 1 })
        ));
    }

    /// unpremultiply saturates invalid premultiplied data instead of
    /// wrapping.
    #[test]
    fn unpremultiply_saturates() {
        // Colour 200 with alpha 100: 200 * 255 / 100 = 510 saturates.
        let im = Raster::new(
            1,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            vec![200, 100],
        )
        .unwrap();
        assert_eq!(im.unpremultiply().getpoint(0, 0), vec![255.0, 100.0]);
    }

    // ---- premultiply / unpremultiply on the float carriers (#631) ----

    /// The eight alphas the float sweeps below are pinned on. They cover the
    /// dead zone (`0.005`), its first value outside (`0.02`), the ordinary
    /// range, an overshoot above every `max_alpha` (`1.5`, `300`), and an
    /// undershoot below zero.
    const FLOAT_ALPHAS: [f32; 8] = [0.0, 0.005, 0.02, 0.5, 1.0, 1.5, -0.5, 300.0];

    /// A 4-band `RgbaF32` raster of `(100, 100, 100, alpha)` pixels, one
    /// column per alpha: the same probe pixel the #604 resample tests use.
    fn float_rgba(alphas: &[f32]) -> Raster {
        let mut data = Vec::with_capacity(alphas.len() * 16);
        for &a in alphas {
            for v in [100.0f32, 100.0, 100.0, a] {
                data.extend_from_slice(&v.to_ne_bytes());
            }
        }
        Raster::new(alphas.len() as u32, 1, PixelFormat::RgbaF32, data).unwrap()
    }

    /// Assert a float sample came out bit-identical to the one vips wrote.
    /// The comparison is on the `f32` bit pattern rather than a tolerance
    /// because the point of these pins is the exact rounding, and a
    /// tolerance loose enough to be comfortable would not see it.
    fn assert_vips_f32(got: f64, want: f32, what: &str) {
        let got = got as f32;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{what}: got {got:?}, want vips' {want:?}"
        );
    }

    /// Run one of the two alpha ops over [`FLOAT_ALPHAS`] and check every
    /// colour band and the stored alpha against the vips capture.
    fn check_float_sweep(
        src: &Raster,
        out: &Raster,
        colours: [f32; 8],
        alphas: [f32; 8],
        label: &str,
    ) {
        assert_eq!(out.format(), src.format(), "{label}: format must survive");
        for (i, ((&want_c, &want_a), &alpha_in)) in colours
            .iter()
            .zip(alphas.iter())
            .zip(FLOAT_ALPHAS.iter())
            .enumerate()
        {
            let px = out.getpoint(i as u32, 0);
            for (b, &got) in px[..3].iter().enumerate() {
                assert_vips_f32(got, want_c, &format!("{label} alpha {alpha_in} band {b}"));
            }
            assert_vips_f32(
                px[3],
                want_a,
                &format!("{label} alpha {alpha_in} stored alpha"),
            );
        }
    }

    /// `try_premultiply` on a float carrier is arithmetic, not a panic
    /// (#631). Pinned bit-exactly on vips 8.18.6 with an untagged float
    /// raster, whose interpretation resolves to sRGB and so takes the
    /// default `max_alpha` of 255:
    ///
    /// ```text
    /// vips rawload sw.raw sw.v 8 1 4 --format float
    /// vips premultiply sw.v swp.v ; vips rawsave swp.v swp.raw
    ///   alpha  0      ->  0            0
    ///   alpha  0.005  ->  0.0019607844 0.005
    ///   alpha  0.02   ->  0.007843138  0.02
    ///   alpha  0.5    ->  0.19607845   0.5
    ///   alpha  1      ->  0.3921569    1
    ///   alpha  1.5    ->  0.5882353    1.5
    ///   alpha -0.5    ->  0           -0.5
    ///   alpha  300    ->  100          300
    /// ```
    ///
    /// The last two are the mirror-image guard of #604: the *factor* takes
    /// the clipped alpha (so `300` normalises to `1` and `-0.5` to `0`)
    /// while the alpha that is *stored* stays raw.
    #[test]
    fn premultiply_float_matches_vips() {
        let im = float_rgba(&FLOAT_ALPHAS);
        let out = im
            .try_premultiply()
            .expect("float premultiply must succeed");
        check_float_sweep(
            &im,
            &out,
            [
                0.0,
                0.001_960_784_4,
                0.007_843_138,
                0.196_078_45,
                0.392_156_9,
                0.588_235_3,
                0.0,
                100.0,
            ],
            FLOAT_ALPHAS,
            "premultiply f32 max_alpha 255",
        );
    }

    /// `try_unpremultiply` on a float carrier is arithmetic too (#631), and
    /// this is where both #604 guards are live at once. Pinned on vips
    /// 8.18.6 over the same raster:
    ///
    /// ```text
    /// vips unpremultiply sw.v swu.v ; vips rawsave swu.v swu.raw
    ///   alpha  0      ->        0    0      (dead zone)
    ///   alpha  0.005  ->        0    0.005  (dead zone)
    ///   alpha  0.02   ->  1275000    0.02
    ///   alpha  0.5    ->    51000    0.5
    ///   alpha  1      ->    25500    1
    ///   alpha  1.5    ->    17000    1.5
    ///   alpha -0.5    ->   -51000    0      (raw factor, clipped alpha)
    ///   alpha  300    ->       85    255
    /// ```
    #[test]
    fn unpremultiply_float_matches_vips() {
        let im = float_rgba(&FLOAT_ALPHAS);
        let out = im
            .try_unpremultiply()
            .expect("float unpremultiply must succeed");
        check_float_sweep(
            &im,
            &out,
            [
                0.0,
                0.0,
                1_275_000.0,
                51_000.0,
                25_500.0,
                17_000.0,
                -51_000.0,
                85.0,
            ],
            [0.0, 0.005, 0.02, 0.5, 1.0, 1.5, 0.0, 255.0],
            "unpremultiply f32 max_alpha 255",
        );
    }

    /// The float `max_alpha` follows the raster's interpretation, exactly as
    /// `vips_interpretation_max_alpha` does, so an scRGB raster (which is
    /// what [`crate::exr`] tags an RGB OpenEXR load) divides by `1.0` and not
    /// by `255`. Without this an EXR's 0..1 samples premultiply to
    /// approximately black. Pinned on vips 8.18.6:
    ///
    /// ```text
    /// vips copy sw.v sws.v --interpretation scrgb
    /// vips premultiply sws.v swps.v
    ///   alpha 0.005 -> 0.5    ;  alpha 0.5 -> 50  ;  alpha 1.5 -> 100
    /// vips unpremultiply sws.v swus.v
    ///   alpha 0.02  -> 5000   ;  alpha 0.5 -> 200 ;  alpha 1.5 -> 66.66667
    /// ```
    #[test]
    fn float_alpha_ops_read_max_alpha_from_the_interpretation() {
        let im = float_rgba(&FLOAT_ALPHAS)
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build();
        let pre = im
            .try_premultiply()
            .expect("scRGB premultiply must succeed");
        check_float_sweep(
            &im,
            &pre,
            [0.0, 0.5, 2.0, 50.0, 100.0, 100.0, 0.0, 100.0],
            FLOAT_ALPHAS,
            "premultiply f32 scRGB",
        );
        let unp = im
            .try_unpremultiply()
            .expect("scRGB unpremultiply must succeed");
        check_float_sweep(
            &im,
            &unp,
            [
                0.0,
                0.0,
                5000.0,
                200.0,
                100.0,
                66.666_67,
                -200.0,
                0.333_333_34,
            ],
            [0.0, 0.005, 0.02, 0.5, 1.0, 1.0, 0.0, 1.0],
            "unpremultiply f32 scRGB",
        );
        assert_eq!(
            pre.interpretation(),
            Interpretation::ScRgb,
            "the output must carry the tag its max_alpha was read from, or a \
             premultiply / unpremultiply round trip divides by 1 and \
             multiplies by 255"
        );
    }

    /// The 16-bit interpretations put the float carrier on the 65535 ceiling,
    /// completing the `vips_interpretation_max_alpha` table. Pinned on vips
    /// 8.18.6 for the `alpha = 0.5` column of the same raster:
    ///
    /// ```text
    /// vips copy sw.v sw16.v --interpretation rgb16
    /// vips premultiply   sw16.v ... -> 0.00076295109465718269
    /// vips unpremultiply sw16.v ... -> 13107000
    /// ```
    #[test]
    fn float_alpha_ops_take_the_65535_ceiling_from_rgb16() {
        let im = float_rgba(&[0.5])
            .copy()
            .interpretation(Interpretation::Rgb16)
            .build();
        assert_vips_f32(
            im.try_premultiply().unwrap().getpoint(0, 0)[0],
            0.000_762_951_1,
            "premultiply f32 Rgb16",
        );
        assert_vips_f32(
            im.try_unpremultiply().unwrap().getpoint(0, 0)[0],
            13_107_000.0,
            "unpremultiply f32 Rgb16",
        );
    }

    /// The float path rounds the way the C does: `nalpha` and `factor` land
    /// in `float` before the colour multiply, so there are **two** roundings
    /// and not one. Computing the whole expression in `f64` and rounding at
    /// the store gives `0.19607843` for this pixel where vips gives
    /// `0.19607845`, a difference of about 1.2 ulp, so this pins the
    /// intermediate rather than only the shape of the formula (runbook
    /// section 14, same class as the fused multiply-add).
    #[test]
    fn float_alpha_ops_round_through_f32_like_the_c_macros() {
        let im = float_rgba(&[0.5]);
        let got = im.try_premultiply().unwrap().getpoint(0, 0)[0] as f32;
        let single_rounded = (100.0f64 * 0.5f64 / 255.0f64) as f32;
        assert_eq!(
            got.to_bits(),
            0.196_078_45f32.to_bits(),
            "want vips' 0.19607845, got {got:?}"
        );
        assert_ne!(
            got.to_bits(),
            single_rounded.to_bits(),
            "an f64 intermediate would have given {single_rounded:?}; the C \
             stores nalpha in a float before multiplying"
        );

        // The un-premultiply half has the same shape: 1.5 against max_alpha
        // 1.0 is 66.66667 through a float factor, 66.666664 through f64.
        let im = float_rgba(&[1.5])
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build();
        let got = im.try_unpremultiply().unwrap().getpoint(0, 0)[0] as f32;
        let single_rounded = (100.0f64 * 1.0f64 / 1.5f64) as f32;
        assert_eq!(
            got.to_bits(),
            66.666_67f32.to_bits(),
            "want vips' 66.66667, got {got:?}"
        );
        assert_ne!(
            got.to_bits(),
            single_rounded.to_bits(),
            "an f64 intermediate would have given {single_rounded:?}"
        );
    }

    /// A float file can hand back NaN and infinities, and OpenEXR routinely
    /// does, so both ops are pinned on them rather than left to chance.
    /// `VIPS_CLIP` is `MAX(0, MIN(max, alpha))` with plain `<` / `>`
    /// ternaries, so `MIN(max, NaN)` returns the NaN and it propagates.
    /// Pinned on vips 8.18.6 over `(100, 100, 100, alpha)`:
    ///
    /// ```text
    /// premultiply    NaN -> nan nan nan nan   +inf -> 100 100 100 inf
    ///                                         -inf ->   0   0   0 -inf
    /// unpremultiply  NaN -> nan nan nan nan   +inf ->   0   0   0 255
    ///                                         -inf ->  -0  -0  -0 0
    /// ```
    #[test]
    fn float_alpha_ops_propagate_nan_and_infinity_like_vips() {
        let im = float_rgba(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

        let pre = im.try_premultiply().expect("NaN must not panic");
        let nan = pre.getpoint(0, 0);
        assert!(
            nan.iter().all(|v| v.is_nan()),
            "a NaN alpha must propagate through premultiply, got {nan:?}"
        );
        assert_vips_f32(pre.getpoint(1, 0)[0], 100.0, "premultiply +inf colour");
        assert_vips_f32(pre.getpoint(2, 0)[0], 0.0, "premultiply -inf colour");
        assert!(
            pre.getpoint(1, 0)[3].is_infinite() && pre.getpoint(2, 0)[3].is_infinite(),
            "premultiply stores the raw alpha, infinities included"
        );

        let unp = im.try_unpremultiply().expect("NaN must not panic");
        let nan = unp.getpoint(0, 0);
        assert!(
            nan.iter().all(|v| v.is_nan()),
            "a NaN alpha must propagate through unpremultiply, got {nan:?}"
        );
        assert_vips_f32(unp.getpoint(1, 0)[0], 0.0, "unpremultiply +inf colour");
        assert_vips_f32(unp.getpoint(1, 0)[3], 255.0, "unpremultiply +inf alpha");
        assert_vips_f32(unp.getpoint(2, 0)[0], -0.0, "unpremultiply -inf colour");
        assert_vips_f32(unp.getpoint(2, 0)[3], 0.0, "unpremultiply -inf alpha");
    }

    /// The float path is band-agnostic, like the `*_MANY` C macros: the last
    /// band is the alpha whatever the band count, and a two-band
    /// `FloatF32(2)` raster gives the same numbers as the RGBA sweep.
    /// Pinned on vips 8.18.6 with a 2x1 two-band float raster of
    /// `(100, alpha)`:
    ///
    /// ```text
    /// vips premultiply   tb.v ... -> 0.19607844948768616 / 0.58823531866073608
    /// vips unpremultiply tb.v ... -> 51000 / 17000
    /// ```
    #[test]
    fn float_alpha_ops_work_on_a_two_band_carrier() {
        let mut data = Vec::new();
        for v in [100.0f32, 0.5, 100.0, 1.5] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let im = Raster::new(
            2,
            1,
            PixelFormat::with_kind(2, SampleKind::F32).unwrap(),
            data,
        )
        .unwrap();
        let pre = im.try_premultiply().expect("two-band float premultiply");
        assert_vips_f32(
            pre.getpoint(0, 0)[0],
            0.196_078_45,
            "2-band premultiply 0.5",
        );
        assert_vips_f32(pre.getpoint(1, 0)[0], 0.588_235_3, "2-band premultiply 1.5");
        let unp = im
            .try_unpremultiply()
            .expect("two-band float unpremultiply");
        assert_vips_f32(unp.getpoint(0, 0)[0], 51_000.0, "2-band unpremultiply 0.5");
        assert_vips_f32(unp.getpoint(1, 0)[0], 17_000.0, "2-band unpremultiply 1.5");
    }

    /// The panicking twins keep their contract on the float carrier: they
    /// panic only on the errors their `try_` form returns, and a float
    /// raster is no longer one of them (#631).
    #[test]
    fn panicking_alpha_twins_no_longer_panic_on_float() {
        let im = float_rgba(&[0.5]);
        assert_vips_f32(
            im.premultiply().getpoint(0, 0)[0],
            0.196_078_45,
            "premultiply",
        );
        assert_vips_f32(
            im.unpremultiply().getpoint(0, 0)[0],
            51_000.0,
            "unpremultiply",
        );
    }

    /// The alpha pair stamps the source interpretation on the **unsigned**
    /// carriers too, and that is new (issue #631). It is easy to read the
    /// float work as leaving the unsigned arm alone, and this is the one way
    /// it does not: `alpha_map`'s output goes through
    /// [`stamp_source_interpretation`] whatever the carrier, so an `Rgba16`
    /// explicitly tagged [`Interpretation::Srgb`] keeps that tag where it
    /// used to come back resolving to [`Interpretation::Rgb16`].
    ///
    /// The new behaviour is the correct one. Measured on vips 8.18.6, both
    /// ops copy the input header: a `1x1 ushort, 4 bands, srgb` input gives a
    /// `srgb` output from `vips premultiply` and from `vips unpremultiply`,
    /// and the same input left `multiband` gives `multiband` back.
    ///
    /// The untagged assertion is what gives the tagged one its meaning: with
    /// no stamp, [`Raster::interpretation`] resolves a 4-band 16-bit buffer to
    /// `Rgb16`, so `Srgb` surviving is the stamp and nothing else.
    #[test]
    fn the_alpha_pair_stamps_the_interpretation_on_the_unsigned_carriers() {
        let untagged = Raster::zeroed(1, 1, PixelFormat::Rgba16).unwrap();
        assert_eq!(
            untagged.interpretation(),
            Interpretation::Rgb16,
            "an untagged 4-band 16-bit raster resolves to Rgb16, so Srgb \
             below can only come from the stamp"
        );
        let tagged = untagged.copy().interpretation(Interpretation::Srgb).build();
        assert_eq!(
            tagged.try_premultiply().unwrap().interpretation(),
            Interpretation::Srgb,
            "premultiply must copy the input interpretation, as vips does"
        );
        assert_eq!(
            tagged.try_unpremultiply().unwrap().interpretation(),
            Interpretation::Srgb,
            "unpremultiply must copy the input interpretation, as vips does"
        );
    }
    /// A single-band float raster still has no alpha band, and that stays a
    /// typed error rather than becoming a float panic by another route.
    #[test]
    fn float_single_band_is_still_a_typed_error() {
        let im = Raster::zeroed(1, 1, PixelFormat::with_kind(1, SampleKind::F32).unwrap()).unwrap();
        assert!(matches!(
            im.try_premultiply(),
            Err(ArithmeticError::NoAlphaBand { bands: 1 })
        ));
        assert!(matches!(
            im.try_unpremultiply(),
            Err(ArithmeticError::NoAlphaBand { bands: 1 })
        ));
    }

    /// The five `try_` methods that were still reaching [`depth_max`]'s panic
    /// on a float raster return the typed refusal instead (issue #631).
    ///
    /// The alpha pair got a real float implementation because vips has one to
    /// copy. These five do not, and the reason is different for each half:
    ///
    /// - `vips_boolean` never operates on float, it **casts to `int` first**.
    ///   Measured on 8.18.6, `vips boolean f.v f.v out.v and` over the 4-band
    ///   float pixel `(100.5, 100.5, 100.5, 0.5)` gives an `int` image of
    ///   `100 100 100 0`, and `(-3.75, -3.75, -3.75, -0.5)` gives
    ///   `-3 -3 -3 0`, so the samples truncate toward zero before the bits
    ///   meet. Refusing and naming the cast is the faithful answer for an op
    ///   that keeps the input depth.
    /// - `vips stdif` refuses outright: `stdif: image must be
    ///   VIPS_FORMAT_UCHAR`, measured for a `float` **and** a `ushort` input.
    /// - `vips recomb` does compute on float and keeps it float (an identity
    ///   matrix over the same raster returns `100.5 100.5 100.5 0.5`), so
    ///   this one is a genuine narrowing, taken because the port writes
    ///   through [`write_u32`] into the input depth. It is written up under
    ///   [`Raster::try_recomb`]'s divergence heading rather than hidden.
    ///
    /// Whichever it is, a `try_` form may refuse but may not unwind, which is
    /// what `proptests::no_try_method_panics_on_a_float_raster` pins for the
    /// whole module.
    #[test]
    fn the_integer_only_ops_refuse_float_instead_of_panicking() {
        let im = Raster::zeroed(4, 4, PixelFormat::RgbaF32).unwrap();
        assert!(matches!(
            im.try_bitand(&im),
            Err(ArithmeticError::FloatUnsupported { op: "bitand" })
        ));
        assert!(matches!(
            im.try_bitor(&im),
            Err(ArithmeticError::FloatUnsupported { op: "bitor" })
        ));
        assert!(matches!(
            im.try_bitxor(&im),
            Err(ArithmeticError::FloatUnsupported { op: "bitxor" })
        ));
        // A 3x3 window inside a 4x4 raster, so this is past the size guard
        // and into the kernel that used to panic.
        assert!(matches!(
            im.try_stdif(3, 3),
            Err(ArithmeticError::FloatUnsupported { op: "stdif" })
        ));
        let row: &[f64] = &[0.25, 0.25, 0.25, 0.25];
        assert!(matches!(
            im.try_recomb(&[row, row, row, row]),
            Err(ArithmeticError::FloatUnsupported { op: "recomb" })
        ));
    }

    /// An unsigned raster still reaches the bitwise, `stdif` and `recomb`
    /// kernels: the refusal above is a refusal of float, not of everything,
    /// and a guard placed one condition too wide would take the whole op out
    /// without any of the tests above noticing.
    #[test]
    fn the_float_refusal_does_not_catch_unsigned_rasters() {
        let im = Raster::new(2, 1, PixelFormat::Rgba8, vec![9, 9, 9, 255, 3, 3, 3, 255]).unwrap();
        assert_eq!(
            im.try_bitand(&im).unwrap().getpoint(0, 0),
            vec![9.0, 9.0, 9.0, 255.0]
        );
        assert_eq!(
            im.try_bitor(&im).unwrap().getpoint(1, 0),
            vec![3.0, 3.0, 3.0, 255.0]
        );
        assert_eq!(
            im.try_bitxor(&im).unwrap().getpoint(0, 0),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(im.try_stdif(2, 1).unwrap().width(), 2);
        let row: &[f64] = &[1.0, 0.0, 0.0, 0.0];
        assert_eq!(im.try_recomb(&[row]).unwrap().getpoint(0, 0), vec![9.0]);
    }

    /// The integer-writing arithmetic ops still reject float rasters
    /// loudly instead of writing garbage (the linear / divide family
    /// floats; cast to an unsigned format first for the rest). The
    /// read-only reductions accept floats since the create batch. The
    /// panicking constant forms now delegate to their `try_*_const` twin,
    /// so the diagnostic is the typed [`ArithmeticError::FloatUnsupported`]
    /// message, which names the op (libviprs#281).
    #[test]
    #[should_panic(expected = "does not support float rasters")]
    fn arithmetic_float_write_panics() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let im = Raster::zeroed(2, 2, f1).unwrap();
        let _ = im.add_const(1.0);
    }

    /// The read-only reductions read `f32` samples: the float generators
    /// from the create batch (`gaussnoise`, the masks) call `avg`,
    /// `deviate`, `min`/`max`, and the position scans directly.
    #[test]
    fn reductions_read_float_rasters() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let mut im = Raster::zeroed(2, 2, f1).unwrap();
        let samples: [f32; 4] = [0.5, -1.5, 2.0, 1.0];
        for (i, v) in samples.iter().enumerate() {
            im.data_mut()[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
        }
        assert_eq!(im.avg(), 0.5);
        assert_eq!(im.min(), -1.5);
        assert_eq!(im.max(), 2.0);
        assert_eq!(im.minpos(), (-1.5, 1, 0));
        assert_eq!(im.maxpos(), (2.0, 0, 1));
        // sum 2.0, sum of squares 7.5: variance (7.5 - 2^2/4) / 3.
        let dev = im.deviate();
        assert!((dev - (6.5f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    // ---- transcendental maths (float output) ----

    /// The trig ops take degrees and produce a float raster:
    /// sin(90) = 1, cos(0) = 1, tan(45) = 1.
    #[test]
    fn trig_degrees_float_output() {
        let im = gray(3, 1, vec![90, 0, 45]);
        let s = im.sin();
        assert_eq!(
            s.format(),
            PixelFormat::with_kind(1, SampleKind::F32).unwrap()
        );
        let sv = float_samples(&s);
        assert!((sv[0] - 1.0).abs() < 1e-6, "sin(90deg) = 1, got {}", sv[0]);
        assert!(sv[1].abs() < 1e-6, "sin(0deg) = 0");

        let cv = float_samples(&im.cos());
        assert!((cv[1] - 1.0).abs() < 1e-6, "cos(0deg) = 1, got {}", cv[1]);

        let tv = float_samples(&im.tan());
        assert!((tv[2] - 1.0).abs() < 1e-6, "tan(45deg) = 1, got {}", tv[2]);
    }

    /// The inverse trig ops accept float input and produce degrees:
    /// asin(1) = 90, asin(0.5) = 30, acos(1) = 0, atan(1) = 45.
    #[test]
    fn inverse_trig_degrees_from_float_input() {
        let im = grayf(2, 1, &[1.0, 0.5]);
        let a = float_samples(&im.asin());
        assert!((a[0] - 90.0).abs() < 1e-4);
        assert!((a[1] - 30.0).abs() < 1e-4);
        let a = float_samples(&im.acos());
        assert!(a[0].abs() < 1e-4);
        assert!((a[1] - 60.0).abs() < 1e-4);
        let a = float_samples(&im.atan());
        assert!((a[0] - 45.0).abs() < 1e-4);
    }

    /// atan2 covers all four quadrants in degrees, atan2(y, x) with self
    /// as the ordinate.
    #[test]
    fn atan2_quadrants() {
        let y = grayf(4, 1, &[1.0, 1.0, -1.0, -1.0]);
        let x = grayf(4, 1, &[1.0, -1.0, -1.0, 1.0]);
        let a = float_samples(&y.atan2(&x));
        assert!((a[0] - 45.0).abs() < 1e-4, "quadrant I, got {}", a[0]);
        assert!((a[1] - 135.0).abs() < 1e-4, "quadrant II, got {}", a[1]);
        assert!((a[2] + 135.0).abs() < 1e-4, "quadrant III, got {}", a[2]);
        assert!((a[3] + 45.0).abs() < 1e-4, "quadrant IV, got {}", a[3]);
    }

    /// sinh / cosh overflow the f32 op output above asinh / acosh of
    /// f32::MAX (~89.4): real libvips `vips_math` on uchar input also
    /// outputs float (f32) and yields inf there, so inf IS the correct
    /// op result for large fixture values (libviprs-tests issue #77:
    /// sinh(226) ~ 7.07e97 >> f32::MAX ~ 3.4e38). Small probes stay
    /// finite and exact.
    #[test]
    fn sinh_cosh_f32_overflow_matches_libvips() {
        let big = gray(1, 1, vec![226]);
        assert_eq!(float_samples(&big.sinh()), vec![f64::INFINITY]);
        assert_eq!(float_samples(&big.cosh()), vec![f64::INFINITY]);

        let small = gray(2, 1, vec![3, 5]);
        let s = float_samples(&small.sinh());
        assert!((s[0] - 3.0f64.sinh()).abs() < 1e-3, "sinh(3), got {}", s[0]);
        let c = float_samples(&small.cosh());
        assert!((c[1] - 5.0f64.cosh()).abs() < 1e-2, "cosh(5), got {}", c[1]);
    }

    /// The hyperbolic family round-trips through its inverses on float
    /// rasters.
    #[test]
    fn hyperbolic_round_trips() {
        let im = grayf(1, 1, &[0.5]);
        let v = float_samples(&im.sinh().asinh())[0];
        assert!((v - 0.5).abs() < 1e-6, "asinh(sinh(0.5)), got {v}");
        let v = float_samples(&im.tanh().atanh())[0];
        assert!((v - 0.5).abs() < 1e-6, "atanh(tanh(0.5)), got {v}");
        let two = grayf(1, 1, &[2.0]);
        let v = float_samples(&two.cosh().acosh())[0];
        assert!((v - 2.0).abs() < 1e-6, "acosh(cosh(2)), got {v}");
        let v = float_samples(&im.tanh())[0];
        assert!((v - 0.5f64.tanh()).abs() < 1e-7);
    }

    /// log is the natural log, log10 base 10; exp and exp10 invert them:
    /// log(e) = 1, log10(100) = 2, exp(0) = 1, exp10(2) = 100.
    #[test]
    fn log_exp_families() {
        let im = grayf(2, 1, &[std::f64::consts::E as f32, 100.0]);
        let l = float_samples(&im.log());
        assert!((l[0] - 1.0).abs() < 1e-6, "log(e) = 1, got {}", l[0]);
        let l = float_samples(&im.log10());
        assert!((l[1] - 2.0).abs() < 1e-6, "log10(100) = 2, got {}", l[1]);

        let im = gray(2, 1, vec![0, 2]);
        let e = float_samples(&im.exp());
        assert!((e[0] - 1.0).abs() < 1e-6, "exp(0) = 1, got {}", e[0]);
        assert!((e[1] - 2.0f64.exp()).abs() < 1e-4);
        let e = float_samples(&im.exp10());
        assert!((e[0] - 1.0).abs() < 1e-6, "exp10(0) = 1");
        assert!((e[1] - 100.0).abs() < 1e-3, "exp10(2) = 100, got {}", e[1]);
    }

    /// pow is self ** other and wop is other ** self, both float:
    /// pow(2, 10) = 1024, wop(2, 10) = 100.
    #[test]
    fn pow_and_wop() {
        let base = gray(1, 1, vec![2]);
        let exp = gray(1, 1, vec![10]);
        let v = float_samples(&base.pow(&exp))[0];
        assert!((v - 1024.0).abs() < 1e-3, "2^10 = 1024, got {v}");
        let v = float_samples(&base.wop(&exp))[0];
        assert!((v - 100.0).abs() < 1e-3, "10^2 = 100, got {v}");
    }

    /// vips `math2` POW guards the whole `base == 0 && exp <= 0` quadrant to
    /// `0`, not the IEEE / `f64::powf` values (`0 ** 0 = 1`, `0 ** -1 = +inf`,
    /// `0 ** -0.5 = +inf`). Verified with the oracle: `vips math2_const zero
    /// out pow c` yields `0` for `c` in `{0, -1, -2, -0.5}`, while a positive
    /// base is untouched (`2 ** -1 = 0.5`). This holds for `pow`, the reversed
    /// `wop`, and the depth-preserving `pow_const`; every other operand pair is
    /// unchanged (`0 ** 5 = 0`, `5 ** 0 = 1`).
    #[test]
    fn pow_zero_zero_matches_vips() {
        let zero = gray(1, 1, vec![0]);
        let five = gray(1, 1, vec![5]);
        assert_eq!(float_samples(&zero.pow(&zero))[0], 0.0, "pow(0,0) = 0");
        assert_eq!(float_samples(&zero.wop(&zero))[0], 0.0, "wop(0,0) = 0");
        assert_eq!(
            zero.pow_const(0.0).getpoint(0, 0),
            vec![0.0],
            "pow_const(0,0)"
        );
        // Neighbouring cases are untouched.
        assert_eq!(float_samples(&zero.pow(&five))[0], 0.0, "0^5 = 0");
        assert_eq!(float_samples(&five.pow(&zero))[0], 1.0, "5^0 = 1");
        assert_eq!(
            five.pow_const(0.0).getpoint(0, 0),
            vec![1.0],
            "pow_const(5,0)=1"
        );
    }

    /// vips POW keeps `0 ** exp = 0` for every `exp <= 0`, not just `exp == 0`:
    /// `f64::powf(0, negative)` is `+inf`, but vips returns `0` (verified with
    /// the oracle: `vips math2_const zero out pow c` yields `0` for `c` in
    /// `{-1, -2, -0.5}`). A positive base still uses the normal power
    /// (`2 ** -1 = 0.5`). Pinned across `pow`, `wop`, and `pow_const`.
    #[test]
    fn pow_zero_negative_exponent_matches_vips() {
        let zero = gray(1, 1, vec![0]);
        // pow_const: 0 ** -1 and 0 ** -0.5 are 0, not +inf.
        assert_eq!(zero.pow_const(-1.0).getpoint(0, 0), vec![0.0], "0 ** -1");
        assert_eq!(zero.pow_const(-2.0).getpoint(0, 0), vec![0.0], "0 ** -2");
        assert_eq!(zero.pow_const(-0.5).getpoint(0, 0), vec![0.0], "0 ** -0.5");

        // Image-image pow / wop with a zero base and a negative exponent.
        let neg1 = grayf(1, 1, &[-1.0]);
        assert_eq!(float_samples(&zero.pow(&neg1))[0], 0.0, "pow: 0 ** -1");
        // wop(a, b) = b ** a, so wop(neg1, zero) = zero ** neg1 = 0.
        assert_eq!(float_samples(&neg1.wop(&zero))[0], 0.0, "wop: 0 ** -1");

        let neghalf = grayf(1, 1, &[-0.5]);
        assert_eq!(float_samples(&zero.pow(&neghalf))[0], 0.0, "pow: 0 ** -0.5");

        // A positive base keeps the ordinary power.
        let two = grayf(1, 1, &[2.0]);
        assert_eq!(float_samples(&two.pow(&neg1))[0], 0.5, "2 ** -1 = 0.5");
    }

    /// The math2 ops validate image compatibility like the other binary
    /// ops.
    #[test]
    fn math2_rejects_mismatch() {
        let a = gray(2, 1, vec![1, 2]);
        let b = gray(1, 1, vec![3]);
        assert!(matches!(
            a.try_atan2(&b),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            a.try_pow(&b),
            Err(ArithmeticError::DimensionMismatch { .. })
        ));
    }

    /// neg produces a float raster with negated samples; abs and sign have
    /// float branches, so abs(neg(x)) round-trips and sign sees negatives.
    #[test]
    fn neg_abs_sign_float() {
        let im = gray(2, 1, vec![5, 0]);
        let n = im.neg();
        assert!(n.format().is_float());
        assert_eq!(float_samples(&n), vec![-5.0, 0.0]);
        assert_eq!(float_samples(&n.abs()), vec![5.0, 0.0]);

        let s = float_samples(&grayf(3, 1, &[-3.5, 0.0, 2.0]).sign());
        assert_eq!(s, vec![-1.0, 0.0, 1.0]);
    }

    /// floor, ceil, and rint round float rasters samplewise (rint rounds
    /// halves to the nearest even integer — banker's rounding, the libvips
    /// VIPS_RINT / C99 `rint` behavior) and stay identities on integer input.
    #[test]
    fn rounding_on_float_rasters() {
        let im = grayf(2, 1, &[1.5, -1.5]);
        assert_eq!(float_samples(&im.floor()), vec![1.0, -2.0]);
        assert_eq!(float_samples(&im.ceil()), vec![2.0, -1.0]);
        assert_eq!(float_samples(&im.rint()), vec![2.0, -2.0]);

        let ints = gray(2, 1, vec![3, 200]);
        assert_eq!(ints.floor().data(), ints.data());
        assert_eq!(ints.ceil().format(), PixelFormat::Gray8);
    }

    /// rint rounds exact halves to the nearest even integer (banker's
    /// rounding), matching vips 8.18.4 `vips round in out rint`, which was
    /// verified against the oracle to map the half-integers as pinned below.
    /// `f64::round` (half away from zero) would give `1, 2, 3, 4, -1, -2, -3`
    /// for these inputs — the old, divergent behavior this replaces.
    #[test]
    fn rint_rounds_half_to_even() {
        let im = grayf(8, 1, &[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]);
        assert_eq!(
            float_samples(&im.rint()),
            vec![0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0, -4.0]
        );
        // Non-half values still round to nearest as usual.
        let f = grayf(4, 1, &[0.4, 0.6, -0.4, 2.9]);
        assert_eq!(float_samples(&f.rint()), vec![0.0, 1.0, 0.0, 3.0]);
    }

    /// The unary maths accept float input (a chained result) without
    /// panicking, unlike the integer-writing ops.
    #[test]
    fn math_accepts_float_input() {
        let im = gray(1, 1, vec![90]);
        let chained = im.sin().asin();
        assert!((float_samples(&chained)[0] - 90.0).abs() < 1e-4);
    }

    // ---- complex operations ----

    /// complexform interleaves (re, im) pairs; real / imag extract them
    /// and conj negates the imaginary half.
    #[test]
    fn complexform_real_imag_conj() {
        let re = gray(1, 1, vec![3]);
        let im = gray(1, 1, vec![4]);
        let z = Raster::complexform(&re, &im);
        assert_eq!(
            z.format(),
            PixelFormat::with_kind(2, SampleKind::F32).unwrap()
        );
        assert_eq!(float_samples(&z), vec![3.0, 4.0]);
        assert_eq!(float_samples(&z.real()), vec![3.0]);
        assert_eq!(float_samples(&z.imag()), vec![4.0]);
        assert_eq!(float_samples(&z.conj()), vec![3.0, -4.0]);
    }

    /// Multi-band complexform interleaves per band: bands [1, 2] and
    /// [3, 4] become [1, 3, 2, 4], and real / imag recover the halves.
    #[test]
    fn complexform_multiband_interleaving() {
        let two = PixelFormat::with_kind(2, SampleKind::U8).unwrap();
        let re = Raster::new(1, 1, two, vec![1, 2]).unwrap();
        let im = Raster::new(1, 1, two, vec![3, 4]).unwrap();
        let z = Raster::complexform(&re, &im);
        assert_eq!(float_samples(&z), vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(float_samples(&z.real()), vec![1.0, 2.0]);
        assert_eq!(float_samples(&z.imag()), vec![3.0, 4.0]);
    }

    /// polar converts (3, 4) to magnitude 5 and angle atan2(4, 3) in
    /// degrees; rect converts back.
    #[test]
    fn polar_rect_round_trip() {
        let z = Raster::complexform(&gray(1, 1, vec![3]), &gray(1, 1, vec![4]));
        let p = float_samples(&z.polar());
        assert!((p[0] - 5.0).abs() < 1e-5, "|3+4i| = 5, got {}", p[0]);
        let want = 4.0f64.atan2(3.0).to_degrees();
        assert!((p[1] - want).abs() < 1e-4, "arg(3+4i), got {}", p[1]);

        let back = float_samples(&z.polar().rect());
        assert!((back[0] - 3.0).abs() < 1e-4);
        assert!((back[1] - 4.0).abs() < 1e-4);
    }

    /// The ported polar expectation: (100 + 100i) has magnitude
    /// 100 * sqrt(2) and angle 45 degrees.
    #[test]
    fn polar_ported_values() {
        let re = gray(1, 1, vec![100]);
        let z = Raster::complexform(&re, &re);
        let p = float_samples(&z.polar());
        assert!((p[0] - 100.0 * 2.0f64.sqrt()).abs() < 1e-3);
        assert!((p[1] - 45.0).abs() < 1e-4);
    }

    /// The complex ops reject odd band counts with a typed error.
    #[test]
    fn complex_rejects_odd_bands() {
        let odd = gray(1, 1, vec![7]);
        assert!(matches!(
            odd.try_polar(),
            Err(ArithmeticError::NotComplex { bands: 1 })
        ));
        assert!(matches!(
            odd.try_real(),
            Err(ArithmeticError::NotComplex { bands: 1 })
        ));
    }

    // ---- hough transforms ----

    /// Decode a hough_line accumulator row back to a signed pixel distance,
    /// inverting vips's `ri = (r + 1) * (height / 2)` mapping over the
    /// image-diagonal normalization.
    fn hough_distance(row: u32, acc_height: u32, w: u32, h: u32) -> f64 {
        let diag = ((w * w + h * h) as f64).sqrt();
        (2.0 * row as f64 / acc_height as f64 - 1.0) * diag
    }

    /// A horizontal line at y = 30 peaks at the 90-degree angle bin and
    /// the distance bin decoding to ~30. Verified against vips 8.18.4
    /// `hough_line`: the peak lands at accumulator cell (x=128, y=155) with
    /// exactly 100 votes (`vips csvsave` of the accumulator), matching this
    /// crate's output cell-for-cell.
    #[test]
    fn hough_line_horizontal_peak() {
        let mut data = vec![0u8; 100 * 100];
        for x in 0..100 {
            data[30 * 100 + x] = 255;
        }
        let im = gray(100, 100, data);
        let acc = im.hough_line();
        assert_eq!((acc.width(), acc.height()), (256, 256));
        let (votes, x, y) = acc.maxpos();
        assert_eq!(votes, 100.0, "all 100 line pixels vote in one bin");
        // vips places the peak at exactly (128, 155).
        assert_eq!((x, y), (128, 155), "vips peak cell (angle, distance)");
        let angle = 180.0 * x as f64 / acc.width() as f64;
        let distance = hough_distance(y, acc.height(), 100, 100);
        assert!(
            (angle - 90.0).abs() < 2.0,
            "angle should be ~90, got {angle}"
        );
        assert!(
            (distance - 30.0).abs() < 2.0,
            "distance should be ~30, got {distance}"
        );
    }

    /// The ported diagonal: a line along x + y = 100 peaks at 45 degrees
    /// and signed distance ~ 100 / sqrt(2), decoded through vips's
    /// diagonal-normalized `(r + 1) * height / 2` binning.
    #[test]
    fn hough_line_diagonal_peak() {
        let mut data = vec![0u8; 100 * 100];
        for x in 10..=90 {
            data[(100 - x) * 100 + x] = 255;
        }
        let im = gray(100, 100, data);
        let acc = im.hough_line();
        let (_votes, x, y) = acc.maxpos();
        let angle = 180.0 * x as f64 / acc.width() as f64;
        let distance = hough_distance(y, acc.height(), 100, 100);
        assert!(
            (angle - 45.0).abs() < 5.0,
            "angle should be ~45, got {angle}"
        );
        assert!(
            (distance - 100.0 / 2.0f64.sqrt()).abs() < 5.0,
            "distance should be ~70.7, got {distance}"
        );
    }

    /// Pins the exact vips 8.18.4 binning for a single lit pixel against the
    /// oracle. A 16x16 image with one pixel at (3, 4) produces, at the
    /// theta = 0 column (i = 0, so `r = xd = x / sqrt(w^2+h^2)`), a vote in
    /// row `(r + 1) * 128`. vips `hough_line` (default 256x256) casts every
    /// column's vote — 256 votes total for one pixel, none discarded — and
    /// the theta=0 vote lands in row 144 (`vips hough_line` + `csvsave`
    /// confirm exactly `[144]` in column 0), matching this computation.
    #[test]
    fn hough_line_single_pixel_matches_vips_binning() {
        let mut data = vec![0u8; 16 * 16];
        data[4 * 16 + 3] = 255;
        let acc = gray(16, 16, data).hough_line();
        // One lit pixel votes in every one of the 256 angle columns, and
        // none fall outside the accumulator (diagonal normalization).
        let total: f64 = {
            let (w, h) = (acc.width(), acc.height());
            (0..w)
                .flat_map(|x| (0..h).map(move |y| (x, y)))
                .map(|(x, y)| acc.getpoint(x, y)[0])
                .sum()
        };
        assert_eq!(total, 256.0, "one pixel votes once per angle column");
        // theta = 0 column: r = 3 / sqrt(16^2+16^2), ri = (r + 1) * 128.
        let diag = ((16 * 16 + 16 * 16) as f64).sqrt();
        let r = 3.0 / diag;
        let ri = ((r + 1.0) * 128.0) as u32;
        assert_eq!(ri, 144, "vips theta=0 row for pixel x=3");
        assert_eq!(acc.getpoint(0, ri)[0], 1.0, "vote sits in the vips row");
    }

    /// Pins the intentional accumulator-format deviation from vips (#495): a
    /// line with more than 65535 collinear lit pixels overflows a single
    /// accumulator cell. vips carries the accumulator as 32-bit `uint` and
    /// reports the true count; this crate carries `Gray16` (ushort) and clamps
    /// at 65535. A 70000x1 all-lit row concentrates every vote into one cell at
    /// the theta bin where the x-term vanishes (column 128, row 128), so vips
    /// reports 70000 there (verified: `vips hough_line` + `vips max` on a
    /// 70000x1 uchar line = 70000) while this op saturates to 65535. Peak
    /// location still agrees; only the >65535 count is clamped.
    #[test]
    fn hough_line_saturates_above_65535_votes_deviation() {
        let w = 70000usize;
        let acc = gray(w as u32, 1, vec![255u8; w]).hough_line();
        let (peak, x, y) = acc.maxpos();
        // vips reports the uncapped 70000 here; core clamps to u16::MAX.
        assert_eq!(peak, 65535.0, "ushort accumulator saturates (vips: 70000)");
        assert_eq!(
            (x, y),
            (128, 128),
            "all votes concentrate in the vanishing-x-term bin, as in vips"
        );
    }

    /// A drawn circle of radius 40 at (50, 50) peaks at its centre with
    /// the strongest band at radius 40: every circle pixel votes for the
    /// exact centre because voting reuses the drawing octant walk.
    #[test]
    fn hough_circle_centre_and_radius() {
        let mut im = Raster::zeroed(100, 100, PixelFormat::Gray8).unwrap();
        im.draw_circle(&[100], 50, 50, 40);

        let acc = im.hough_circle(35, 45);
        assert_eq!((acc.width(), acc.height()), (100, 100));
        assert_eq!(acc.format().channels(), 11);

        let (_votes, x, y) = acc.maxpos();
        assert!((x as f64 - 50.0).abs() < 2.0, "centre x ~50, got {x}");
        assert!((y as f64 - 50.0).abs() < 2.0, "centre y ~50, got {y}");
        let bands = acc.getpoint(x, y);
        let r = bands
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32 + 35)
            .unwrap();
        assert!((r as f64 - 40.0).abs() < 2.0, "radius ~40, got {r}");
    }

    /// hough_circle validates the radius range.
    #[test]
    fn hough_circle_rejects_empty_range() {
        let im = gray(4, 4, vec![0; 16]);
        assert!(matches!(
            im.try_hough_circle(45, 35),
            Err(ArithmeticError::EmptyRadiusRange { min: 45, max: 35 })
        ));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// A 4x4 `RgbaF32` raster from 64 `f32` samples.
    fn float_raster(samples: &[f32]) -> Raster {
        let mut data = Vec::with_capacity(samples.len() * 4);
        for &v in samples {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        Raster::new(4, 4, PixelFormat::RgbaF32, data).expect("4x4 RgbaF32 from 64 samples")
    }

    /// Call every `try_*` method this module exposes, on `a` (with `b` as the
    /// second operand where one is needed), discarding every result. Arguments
    /// are chosen to get *past* each method's own validation and into its
    /// kernel: a 3x3 `stdif` window fits inside the 4x4 raster, the `recomb`
    /// matrix has one row of four coefficients per band, and the vectors have
    /// one element per band. A method that panics takes the calling test with
    /// it, which is the whole point.
    ///
    /// Returns the names it called, so
    /// [`every_try_method_in_the_module_is_in_the_sweep`] can prove the list
    /// is complete against the source rather than trusting it.
    fn call_every_try_method(a: &Raster, b: &Raster) -> Vec<&'static str> {
        let mut called = Vec::new();
        macro_rules! sweep {
            ($name:literal, $call:expr) => {{
                let _ = $call;
                called.push($name);
            }};
        }
        let vec4 = [1.0f64, 2.0, 3.0, 4.0];
        let row: &[f64] = &[0.25, 0.25, 0.25, 0.25];
        let matrix: &[&[f64]] = &[row, row, row, row];

        sweep!("try_measure", a.try_measure(2, 2));
        sweep!("try_find_trim", a.try_find_trim(None));
        sweep!("try_add_const", a.try_add_const(1.5));
        sweep!("try_sub_const", a.try_sub_const(1.5));
        sweep!("try_mul_const", a.try_mul_const(1.5));
        sweep!("try_floordiv_const", a.try_floordiv_const(1.5));
        sweep!("try_pow_const", a.try_pow_const(2.0));
        sweep!("try_rem_const", a.try_rem_const(1.5));
        sweep!("try_add_vec", a.try_add_vec(&vec4));
        sweep!("try_sub_vec", a.try_sub_vec(&vec4));
        sweep!("try_mul_vec", a.try_mul_vec(&vec4));
        sweep!("try_div_vec", a.try_div_vec(&vec4));
        sweep!("try_sub", a.try_sub(b));
        sweep!("try_max_diff", a.try_max_diff(b));
        sweep!("try_avg_diff", a.try_avg_diff(b));
        sweep!("try_mul", a.try_mul(b));
        sweep!("try_div", a.try_div(b));
        sweep!("try_remainder", a.try_remainder(b));
        sweep!("try_minpair", a.try_minpair(b));
        sweep!("try_maxpair", a.try_maxpair(b));
        sweep!("try_sum", Raster::try_sum(&[a, b]));
        sweep!("try_more_than", a.try_more_than(b));
        sweep!("try_more_eq", a.try_more_eq(b));
        sweep!("try_less_than", a.try_less_than(b));
        sweep!("try_less_eq", a.try_less_eq(b));
        sweep!("try_equal", a.try_equal(b));
        sweep!("try_noteq", a.try_noteq(b));
        sweep!("try_bitand", a.try_bitand(b));
        sweep!("try_bitor", a.try_bitor(b));
        sweep!("try_bitxor", a.try_bitxor(b));
        sweep!("try_stdif", a.try_stdif(3, 3));
        sweep!("try_recomb", a.try_recomb(matrix));
        sweep!("try_premultiply", a.try_premultiply());
        sweep!("try_unpremultiply", a.try_unpremultiply());
        sweep!("try_atan2", a.try_atan2(b));
        sweep!("try_pow", a.try_pow(b));
        sweep!("try_wop", a.try_wop(b));
        sweep!("try_complexform", Raster::try_complexform(a, b));
        sweep!("try_polar", a.try_polar());
        sweep!("try_rect", a.try_rect());
        sweep!("try_conj", a.try_conj());
        sweep!("try_real", a.try_real());
        sweep!("try_imag", a.try_imag());
        sweep!("try_hough_circle", a.try_hough_circle(1, 2));
        called
    }

    /// The sweep above covers **every** `pub fn try_*` in this module, checked
    /// against the module's own source rather than against a list somebody
    /// remembered to update.
    ///
    /// This is what turns [`no_try_method_panics_on_a_float_raster`] from a
    /// test of five methods into a test of the class: a new fallible op added
    /// to `arithmetic.rs` fails here until it is swept, so it cannot arrive
    /// with an unnoticed `depth_max` panic behind it the way `try_recomb`,
    /// `try_stdif` and the three bitwise ops did (issue #631).
    #[test]
    fn every_try_method_in_the_module_is_in_the_sweep() {
        let source = include_str!("arithmetic.rs");
        let mut declared: Vec<&str> = source
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub fn try_"))
            .map(|rest| {
                let end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                &rest[..end]
            })
            .collect();
        declared.sort_unstable();
        declared.dedup();
        // A floor on the parser, not on the module. The set comparison below
        // is what enforces coverage; this only catches the scan silently
        // matching nothing, which is how a guard ends up passing everything.
        assert!(
            declared.len() >= 20,
            "the `pub fn try_` scan found only {} declarations in \
             arithmetic.rs, so the scan itself is broken",
            declared.len()
        );

        let im = float_raster(&[0.5f32; 64]);
        let mut swept: Vec<&str> = call_every_try_method(&im, &im)
            .into_iter()
            .map(|n| n.strip_prefix("try_").expect("sweep names start try_"))
            .collect();
        swept.sort_unstable();
        swept.dedup();

        let missing: Vec<&&str> = declared.iter().filter(|d| !swept.contains(d)).collect();
        let stale: Vec<&&str> = swept.iter().filter(|s| !declared.contains(s)).collect();
        assert!(
            missing.is_empty() && stale.is_empty(),
            "the float sweep is out of step with the module: not swept {missing:?}, \
             swept but not declared {stale:?}"
        );
    }

    /// One `f32` sample: mostly ordinary magnitudes, with the values a float
    /// file actually delivers and the guards actually key on mixed in. `0.005`
    /// and `0.02` sit either side of the un-premultiply dead zone, `f32::MAX`
    /// overflows every intermediate, and OpenEXR hands back NaN and both
    /// infinities as a matter of course.
    fn float_sample() -> impl Strategy<Value = f32> {
        prop_oneof![
            8 => -1.0e6f32..1.0e6f32,
            1 => Just(0.0f32),
            1 => Just(-0.0f32),
            1 => Just(0.005f32),
            1 => Just(0.02f32),
            1 => Just(f32::NAN),
            1 => Just(f32::INFINITY),
            1 => Just(f32::NEG_INFINITY),
            1 => Just(f32::MAX),
            1 => Just(f32::MIN_POSITIVE),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        /// No `try_*` method in this module unwinds on an `RgbaF32` raster,
        /// whatever the samples and whatever the interpretation tag.
        ///
        /// A `try_` form that panics has no way for the caller to recover, and
        /// this class of bug shipped five times: `try_recomb`, `try_stdif`,
        /// `try_bitand`, `try_bitor` and `try_bitxor` all reached
        /// [`depth_max`]'s "does not support float rasters yet" panic on a
        /// raster [`crate::exr`] and [`crate::fits`] produce from an ordinary
        /// file (issue #631). Each of them may still *refuse* float input;
        /// what it may not do is unwind. Returning an
        /// [`ArithmeticError::FloatUnsupported`] and computing a real answer
        /// both pass, which is deliberate: this pins the contract, not the
        /// policy.
        ///
        /// The interpretation is swept because the alpha pair reads its
        /// `max_alpha` from it, so `ScRgb` and the default are different code
        /// paths through [`interpretation_max_alpha`].
        #[test]
        fn no_try_method_panics_on_a_float_raster(
            samples in prop::collection::vec(float_sample(), 64),
            scrgb in any::<bool>(),
        ) {
            let a = float_raster(&samples);
            let a = if scrgb {
                a.copy().interpretation(Interpretation::ScRgb).build()
            } else {
                a
            };
            let mut reversed = samples.clone();
            reversed.reverse();
            let b = float_raster(&reversed);
            // A panic in any of them fails this test where it happens, which
            // is the assertion. Reaching here at all is the pass condition;
            // the count is left to `every_try_method_in_the_module_is_in_the_sweep`
            // so there is only one place that knows how many there are.
            let called = call_every_try_method(&a, &b);
            prop_assert!(
                !called.is_empty(),
                "the sweep called nothing, so this test proves nothing"
            );
        }
    }
}
