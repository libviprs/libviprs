//! Conversion, orientation, and colour-adjacent operations ported from
//! libvips.
//!
//! This module is the fourth batch of the libvips operation surface required
//! by the ported integration tests: it covers sample-format casting, the
//! metadata copy family, right-angle geometry, tone curves, and the small
//! generator constructors the ported suites use as fixtures. Each fallible
//! operation exists in two forms:
//!
//! * a fallible `try_*` method returning `Result<_, ConversionError>`, the
//!   primary implementation with typed errors for band-count changes,
//!   invalid exponents, and mismatched condition images; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`cast`, `rot`, `arrayjoin`, ...) exactly. These delegate to the
//!   `try_*` form and panic with the typed error's message, mirroring the
//!   "known-good input" contract of [`Raster::add`] and
//!   [`Raster::getpoint`].
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::cast`] | `vips_cast` | samples clip-cast to a new bit depth |
//! | [`Raster::copy`] | `vips_copy` | pixel-identical copy with modified metadata |
//! | [`Raster::autorot`] | `vips_autorot` | orientation tag applied, then cleared |
//! | [`Raster::fliphor`] | `vips_flip` (horizontal) | left-right mirror |
//! | [`Raster::flipver`] | `vips_flip` (vertical) | top-bottom mirror |
//! | [`Raster::rot`] | `vips_rot` | rotation by a right-angle [`Angle`] |
//! | [`Raster::wrap`] | `vips_wrap` | toroidal shift moving the centre to the origin |
//! | [`Raster::gamma`] | `vips_gamma` | gamma curve, maximum value preserved |
//! | [`Raster::falsecolour`] | `vips_falsecolour` | band 0 mapped through the PET false-colour LUT |
//! | [`Raster::addalpha`] | `vips_addalpha` | one opaque alpha band appended |
//! | [`Raster::arrayjoin`] | `vips_arrayjoin` | images tiled into a grid |
//! | [`Raster::join`] | `vips_join` | two images joined left-right or top-bottom |
//! | [`Raster::grey`] | `vips_grey` | horizontal grey ramp |
//! | [`Raster::identity`] | `vips_identity` | 256x1 identity LUT |
//! | [`Raster::identity_ushort`] | `vips_identity` (`ushort: true`) | 65536x1 identity LUT |
//! | [`Raster::switch`] | `vips_switch` | index image of the first true condition |
//!
//! # Join
//!
//! [`Raster::join`] is a literal transcription of `conversion/join.c`,
//! which has no pixel loop of its own: it is [`Raster::insert`] plus an
//! optional crop.
//!
//! * The placement ladder is `join.c:109-156`. A horizontal join puts the
//!   second image at `x = in1.width + shim`; a vertical one at
//!   `y = in1.height + shim`. On the other axis [`Align::Low`] gives `0`,
//!   [`Align::High`] gives `in1 - in2`, and [`Align::Centre`] gives
//!   `in1 / 2 - in2 / 2` — **two separate truncating integer divisions**,
//!   exactly as the C writes it. That is not the same as
//!   `(in1 - in2) / 2`: for `in1 = 4, in2 = 3` the C form gives `1` and
//!   the combined form gives `0`. Under `High` and `Centre` the offset
//!   goes negative when the second image is the larger one.
//! * The insert at `join.c:158-162` always runs with `expand` **true**,
//!   whatever the caller asked for, so no input pixel is ever lost at
//!   this stage and `background` fills whatever neither image covers.
//! * The caller's `expand` only selects the crop at `join.c:164-207`. A
//!   horizontal join crops to `(0, max(0, y) - y, t.width,
//!   min(in1.height, in2.height))` and a vertical one to
//!   `(max(0, x) - x, 0, min(in1.width, in2.width), t.height)`; the
//!   `max(0, ...)` origin is what recovers the rows or columns the
//!   negative offset pushed off the top or the left. The crop is skipped
//!   when the rectangle is already the whole image, as in the C.
//!
//! Band and format unification comes free: `join.c` does none itself and
//! leans on `vips_insert`'s `formatalike` + `bandalike`, so composing on
//! [`Raster::try_insert`] inherits the crate's equivalent (widest depth,
//! largest band count, a one-band input replicated).
//!
//! # Metadata
//!
//! This batch introduces the raster metadata block (`RasterMeta`): colour
//! interpretation, x/y resolution, x/y offset, and the EXIF-style
//! orientation tag. Every [`Raster`] constructor starts from the defaults
//! (no explicit interpretation, resolution `1.0`, offsets `0`, orientation
//! `1`). [`Raster::copy`] is the mutation surface, the getters
//! ([`Raster::interpretation`], [`Raster::xres`], [`Raster::yres`],
//! [`Raster::xoffset`], [`Raster::yoffset`], [`Raster::orientation`]) are
//! the read surface, and [`Raster::interpretation`] infers a value from the
//! [`PixelFormat`] when none has been set explicitly (`Gray8` reads as
//! [`Interpretation::Bw`], `Rgb8`/`Rgba8` as [`Interpretation::Srgb`], and
//! so on).
//!
//! The operations in this module carry the metadata of their (first) input
//! through to the result, with two exceptions: [`Raster::autorot`] resets
//! the orientation tag to `1` after applying it, and
//! [`Raster::falsecolour`] stamps its RGB result as
//! [`Interpretation::Srgb`]. Operations in the earlier batches
//! ([`crate::bands`], [`crate::arithmetic`], [`crate::extract`]) predate the
//! metadata block and return default metadata.
//!
//! The decode paths do not read EXIF yet, so a freshly decoded JPEG always
//! carries orientation `1` and [`Raster::autorot`] is the identity for it,
//! which is exactly what the ported `test_autorot` asserts for
//! `sample.jpg`. Orientation enters programmatically through
//! `copy().orientation(n).build()` until EXIF parsing lands.
//!
//! # Deferred surface
//!
//! * **Signed and 64-bit sample formats.** [`PixelFormat`] carries
//!   unsigned 8/16-bit and 32-bit float samples; the signed
//!   (`char`/`short`/`int`) and `double`/complex targets of `vips_cast`
//!   remain unrepresentable. No ported test names them yet.
//! * **Float inputs to the earlier op batches.** `cast` converts to and
//!   from the float formats and `grey(w, h, false)` produces the float
//!   0.0..1.0 ramp, but the arithmetic, histogram, band, extract, draw,
//!   and compositing operations still assume unsigned samples; they
//!   reject float rasters loudly (a panic with a clear message from their
//!   panicking ported-test surface) rather than misreading the bytes.
//!   Float support for those ops lands with their own batches (trig/log
//!   maths, float compositing, colour spaces).
//! * **`composite` / `CompositeMode`.** Only the ported composite tests
//!   reference `CompositeMode`; no conversion-batch test needs it, so the
//!   enum ships with the Porter-Duff compositing batch instead.
//! * **`rot45` / `Angle45`.** The 45-degree family has odd-square diagonal
//!   semantics of its own and ships with a later geometry batch.

use crate::bands::BandError;
use crate::extract::ExtractError;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use core::num::NonZeroU16;
use thiserror::Error;

/// Typed errors for the conversion operations in [`crate::conversion`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConversionError {
    /// `cast` was asked to change the band count; it only changes the bit
    /// depth. Use the band operations in [`crate::bands`] to change the
    /// band count.
    #[error(
        "cast cannot change the band count: {from:?} has {from_bands} bands, {to:?} has {to_bands}"
    )]
    CastBandCountChange {
        from: PixelFormat,
        from_bands: usize,
        to: PixelFormat,
        to_bands: usize,
    },
    /// A gamma exponent outside the usable range was supplied.
    #[error("gamma exponent must be a finite value greater than zero, got {exponent}")]
    InvalidGammaExponent { exponent: f64 },
    /// An operation over a list of images received an empty list.
    #[error("{op} requires at least one input image")]
    EmptyInput { op: &'static str },
    /// `switch` writes indices into an 8-bit image, so at most 255
    /// conditions are addressable (index 255 doubles as the no-match
    /// value for exactly 255 conditions).
    #[error("switch supports at most 255 condition images, got {count}")]
    TooManyConditions { count: usize },
    /// Two rasters that must share dimensions do not.
    #[error("dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        got_w: u32,
        got_h: u32,
    },
    /// `arrayjoin` inputs whose band counts differ and where the smaller
    /// count is not 1 (libvips `bandalike` rejects the same shapes).
    #[error(
        "band-count mismatch: images have {expected} and {got} bands; counts must match unless one is 1"
    )]
    BandCountMismatch { expected: usize, got: usize },
    /// A `switch` condition image has more than one band.
    #[error("switch conditions must be single-band images, got {bands} bands")]
    ConditionNotMono { bands: usize },
    /// `rot45` was given an image that is not an odd-sided square. Like
    /// libvips, the 45-degree rotation is a ring permutation defined only on
    /// odd squares.
    #[error("rot45 requires an odd-sided square image, got {width}x{height}")]
    NotOddSquare { width: u32, height: u32 },
    /// An align name handed to `Align::from_str` is not one of the
    /// libvips `VipsAlign` nicknames.
    #[error("unknown align {name:?}, expected low, centre, or high")]
    UnknownAlign {
        /// The name that was not recognised.
        name: String,
    },
    /// The result dimensions would overflow `u32`. `arrayjoin` reports the
    /// grid it was asked to build. [`Raster::join`] does not use this
    /// variant: its canvas is sized inside the delegated
    /// [`Raster::try_insert`], so an oversized canvas arrives as
    /// [`ConversionError::Extract`] wrapping
    /// [`crate::extract::ExtractError::SizeOverflow`], and an offset that
    /// will not fit the placement is
    /// [`ConversionError::PlacementOffsetOverflow`]. A caller that wants
    /// every "this is too big" outcome from `join` has to match both.
    #[error("result size {width}x{height} exceeds u32::MAX")]
    SizeOverflow { width: u64, height: u64 },
    /// A `shim` above the maximum libvips accepts, from either
    /// [`Raster::join`] or [`Raster::arrayjoin`]. Both declare the property
    /// as `VIPS_ARG_INT(class, "shim", 5, ..., 0, 1000000, 0)`, so GObject
    /// refuses `--shim 1000001` before either operation runs at all.
    /// Widening the argument to `u32` here carried the lower bound into the
    /// type but not the upper one, so the upper bound is checked instead:
    /// without it a single argument asks for a canvas of several gigabytes
    /// out of two tiny inputs.
    #[error("shim must be at most {max}, got {shim}")]
    ShimTooLarge {
        /// The `shim` that was asked for.
        shim: u32,
        /// The largest `shim` libvips accepts, `1000000`.
        max: u32,
    },
    /// An explicit `across` outside the range libvips declares on
    /// [`Raster::arrayjoin`]'s property,
    /// `VIPS_ARG_INT(class, "across", 4, ..., 1, 1000000, 1)`. GObject
    /// refuses `--across 0` and `--across 1000001` before the operation is
    /// built at all, so neither reaches a grid layout in vips and neither
    /// does here. `across` used to be clamped into `1..=n` instead, which
    /// silently accepted both. The default (`None`, meaning one row of every
    /// image) is not checked against this range: vips assigns it directly to
    /// the struct field and bypasses its own property check the same way.
    #[error("across must be between {min} and {max}, got {across}")]
    AcrossOutOfRange {
        /// The `across` that was asked for.
        across: u32,
        /// The smallest `across` libvips accepts, `1`.
        min: u32,
        /// The largest `across` libvips accepts, `1000000`.
        max: u32,
    },
    /// The offset [`Raster::join`] computed for the second image does not
    /// fit the signed 32-bit range images are placed at (`join.c` does
    /// this arithmetic in `int`, and [`Raster::try_insert`] takes `i32`
    /// coordinates). With `shim` bounded by
    /// [`ConversionError::ShimTooLarge`] only a large input *dimension*
    /// reaches this, either directly (`w1 + shim` past `i32::MAX`) or
    /// through a negative offset under [`Align::Centre`] / [`Align::High`],
    /// where the second image reaching left of or above the first makes
    /// the offset negative. Some offsets rejected here name a canvas that
    /// would have fitted `u32` and the allocation budget; lifting that
    /// needs the placement widened to `i64` end to end, which is its own
    /// change.
    #[error("join placement offset ({x}, {y}) does not fit i32")]
    PlacementOffsetOverflow {
        /// Horizontal offset of the second image from the first image's
        /// origin, in pixels. Negative when the second image reaches left
        /// of it.
        x: i64,
        /// Vertical offset of the second image from the first image's
        /// origin, in pixels. Negative when the second image reaches above
        /// it.
        y: i64,
    },
    /// The operation was handed a float raster it cannot read yet.
    /// [`Raster::try_join`] and [`Raster::try_arrayjoin`] return this
    /// instead of running into the unsigned-only sample copy, which reads
    /// samples as `u8` or `u16` and panics on 4-byte ones. Float input is
    /// ordinary rather than exotic here: every `colourspace` result for
    /// Lab, Lch, OkLab, OkLCh, XYZ, scRGB and Yxy is a float raster, so
    /// `im.colourspace(Lab)` is already one. Real vips handles float on
    /// both operations, so this is a libviprs limitation rather than
    /// parity; it is a typed error so a fallible method reports it instead
    /// of panicking. Mirrors
    /// [`crate::arithmetic::ArithmeticError::FloatUnsupported`].
    #[error("{op} does not support float rasters yet; cast to an unsigned 8/16-bit format first")]
    FloatFormatUnsupported {
        /// The operation that was asked for.
        op: &'static str,
    },
    /// A delegated band operation failed.
    #[error(transparent)]
    Band(#[from] BandError),
    /// A delegated extract or placement operation failed. [`Raster::join`]
    /// is `insert` plus an optional `extract_area`, so its band-count,
    /// background, and size errors arrive through here.
    #[error(transparent)]
    Extract(#[from] ExtractError),
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Unwrap a conversion result for the panicking ported-test surface.
#[inline]
#[track_caller]
fn expect_conv<T>(op: &str, r: Result<T, ConversionError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Read the flat `i`-th unsigned sample of a buffer with the given
/// bytes-per-channel (native byte order for 16-bit, matching
/// [`crate::raster_ops`]). Unsigned depths only: float callers use
/// [`read_f32_flat`], and the panic arm keeps unsigned-only operations
/// from misreading float bytes as `u16` pairs.
#[inline]
fn read_flat(data: &[u8], bpc: usize, i: usize) -> u32 {
    match bpc {
        1 => data[i] as u32,
        2 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as u32,
        _ => panic!(
            "this operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Write the flat `i`-th unsigned sample. `v` must already fit the depth.
/// Unsigned depths only; see [`read_flat`].
#[inline]
fn write_flat(data: &mut [u8], bpc: usize, i: usize, v: u32) {
    match bpc {
        1 => data[i] = v as u8,
        2 => {
            let b = (v as u16).to_ne_bytes();
            data[2 * i] = b[0];
            data[2 * i + 1] = b[1];
        }
        _ => panic!(
            "this operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Read the flat `i`-th sample of a float buffer (native byte order).
#[inline]
fn read_f32_flat(data: &[u8], i: usize) -> f32 {
    let b = 4 * i;
    f32::from_ne_bytes([data[b], data[b + 1], data[b + 2], data[b + 3]])
}

/// One `vips_cast` sample from a float carrier down to an unsigned one.
///
/// `vips_cast` semantics (`conversion/cast.c:566-568`): clip to the
/// target range, then TRUNCATE. C clips in the source type via
/// `VIPS_CLIP_UCHAR` / `VIPS_CLIP_USHORT` and then assigns to the narrow
/// integer, and that implicit conversion truncates rather than rounds.
/// The C doc comment spells it out: "Floats are truncated (not rounded).
/// Out of range values are clipped."
///
/// `trunc`, NOT `floor`. The two are indistinguishable today, because
/// every carrier here is unsigned and a negative sample clips to 0 before
/// the rounding mode can show. C's `(int)` cast truncates toward zero, so
/// `trunc` is the one that stays right when a signed carrier lands
/// (#516). Do not simplify it to `floor`.
///
/// `NaN` pins to 0 explicitly: `trunc` leaves `NaN` alone and `clamp`
/// passes it straight through, so without this branch it would reach the
/// cast and land on 0 by accident rather than by rule.
///
/// It lives out here as a scalar rather than inline in
/// [`Raster::try_cast`] because the fused edge detectors narrow their
/// magnitude to uchar one sample at a time instead of building a float
/// raster to hand to `vips_cast` (issue #562). Sharing the one spelling
/// is what stops the two from drifting apart.
#[inline]
pub(crate) fn cast_float_sample(v: f64, out_bpc: usize) -> u32 {
    let max = if out_bpc == 1 { 255.0 } else { 65535.0 };
    if v.is_nan() {
        0
    } else {
        v.clamp(0.0, max).trunc() as u32
    }
}

/// Write the flat `i`-th sample of a float buffer (native byte order).
#[inline]
fn write_f32_flat(data: &mut [u8], i: usize, v: f32) {
    let b = 4 * i;
    data[b..b + 4].copy_from_slice(&v.to_ne_bytes());
}

/// The libvips colour interpretation tags (`VipsInterpretation`).
///
/// An interpretation records how the numbers in a raster should be read
/// (sRGB, CIE Lab, a histogram, a plain matrix, ...). It is advisory
/// metadata: it does not change the stored samples, and the pipeline does
/// not validate that the band count matches the tag, exactly as in libvips
/// where `copy` accepts any interpretation.
///
/// The perceptual `OkLab` and `OkLch` tags are libvips' own
/// `VIPS_INTERPRETATION_OKLAB` and `VIPS_INTERPRETATION_OKLCH`, assigned by
/// libvips 8.18 (`libvips/include/vips/image.h:115-116`), so a raster
/// tagged with either writes the same `.v` header code real vips writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Interpretation {
    /// A many-band image with no colour meaning.
    Multiband,
    /// Single-band luminance, 0 to 255.
    Bw,
    /// A set of histogram bins.
    Histogram,
    /// CIE XYZ.
    Xyz,
    /// CIE Lab, D65-relative.
    Lab,
    /// CMYK ink values.
    Cmyk,
    /// Lab packed into libvips' 10:11:11 coding.
    Labq,
    /// Generic RGB with no calibrated space.
    Rgb,
    /// CMC(l:c) uniform colour space.
    Cmc,
    /// CIE LCh, the cylindrical form of Lab.
    Lch,
    /// Lab held in signed 16-bit samples.
    Labs,
    /// sRGB, 0 to 255.
    Srgb,
    /// CIE Yxy.
    Yxy,
    /// The result of a Fourier transform.
    Fourier,
    /// RGB held in 16-bit samples, 0 to 65535.
    Rgb16,
    /// Single-band luminance held in 16-bit samples, 0 to 65535.
    Grey16,
    /// A plain array of numbers, not an image.
    Matrix,
    /// Linear-light scRGB.
    ScRgb,
    /// Hue, saturation, value.
    Hsv,
    /// OkLab perceptual space.
    OkLab,
    /// OkLCh, the cylindrical form of OkLab.
    OkLch,
}

impl Interpretation {
    /// The interpretation libvips would assign a raster of this
    /// [`PixelFormat`] absent any explicit tag: 8-bit mono reads as
    /// [`Interpretation::Bw`], 16-bit mono as [`Interpretation::Grey16`],
    /// 8-bit colour as [`Interpretation::Srgb`], 16-bit colour as
    /// [`Interpretation::Rgb16`], four-band float as
    /// [`Interpretation::Srgb`] (libvips' guess for non-ushort colour),
    /// and the multiband and float intermediates as
    /// [`Interpretation::Multiband`].
    ///
    /// Read off [`PixelFormat::canonical`], so both spellings of a layout
    /// get the same answer: `FloatF32(4)` reads as sRGB exactly as
    /// `RgbaF32` does (issue #531).
    pub fn for_format(format: PixelFormat) -> Self {
        match format.canonical() {
            PixelFormat::Gray8 => Self::Bw,
            PixelFormat::Gray16 => Self::Grey16,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::RgbaF32 => Self::Srgb,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => Self::Rgb16,
            PixelFormat::Multi8(_) | PixelFormat::Multi16(_) | PixelFormat::FloatF32(_) => {
                Self::Multiband
            }
        }
    }
}

/// A right-angle rotation for [`Raster::rot`] (libvips `VipsAngle`).
///
/// `D90` is a quarter turn clockwise, matching libvips: the left column of
/// the input becomes the top row of the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Angle {
    /// No rotation.
    D0,
    /// 90 degrees clockwise.
    D90,
    /// 180 degrees.
    D180,
    /// 270 degrees clockwise (a quarter turn anticlockwise).
    D270,
}

/// A multiple-of-45-degree rotation for [`Raster::rot45`] (libvips
/// `VipsAngle45`).
///
/// Like libvips, `rot45` is defined only on odd-sided square images: each
/// concentric square ring of the image is rotated clockwise by an eighth of
/// a turn per 45-degree step, so the transform is an exact pixel permutation
/// and every angle has an exact inverse (`D45`/`D315`, `D90`/`D270`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Angle45 {
    /// No rotation.
    D0,
    /// 45 degrees clockwise.
    D45,
    /// 90 degrees clockwise.
    D90,
    /// 135 degrees clockwise.
    D135,
    /// 180 degrees.
    D180,
    /// 225 degrees clockwise.
    D225,
    /// 270 degrees clockwise.
    D270,
    /// 315 degrees clockwise.
    D315,
}

impl Angle45 {
    /// The number of 45-degree clockwise steps this angle represents, `0..8`.
    fn steps(self) -> u32 {
        match self {
            Self::D0 => 0,
            Self::D45 => 1,
            Self::D90 => 2,
            Self::D135 => 3,
            Self::D180 => 4,
            Self::D225 => 5,
            Self::D270 => 6,
            Self::D315 => 7,
        }
    }
}

/// The axis [`Raster::join`] joins a pair of images along (libvips
/// `VipsDirection`).
///
/// This is deliberately not the crate-root `Direction` of
/// [`crate::morphology`], which names an erosion/dilation axis, nor
/// [`crate::mosaicing::MergeDirection`], which names a feathered
/// tie-point merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JoinDirection {
    /// Left-right: the second image goes to the right of the first.
    Horizontal,
    /// Top-bottom: the second image goes below the first.
    Vertical,
}

/// Which edge the two images of a [`Raster::join`] line up on (libvips
/// `VipsAlign`).
///
/// [`str::parse`] accepts the libvips nicknames (`"low"`, `"centre"`,
/// `"high"`), plus the `"center"` spelling, mirroring the `&str` surface
/// [`crate::extract::CompassDirection`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Align {
    /// Line up on the low coordinate edge: the top for a horizontal
    /// join, the left for a vertical one. The libvips default.
    Low,
    /// Line up on the centre of the shared axis.
    Centre,
    /// Line up on the high coordinate edge: the bottom for a horizontal
    /// join, the right for a vertical one.
    High,
}

impl std::str::FromStr for Align {
    type Err = ConversionError;

    /// Parse a libvips `VipsAlign` nickname: `"low"`, `"centre"`, or
    /// `"high"`. `"center"` is accepted as a spelling of `"centre"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "low" => Self::Low,
            "centre" | "center" => Self::Centre,
            "high" => Self::High,
            other => {
                return Err(ConversionError::UnknownAlign {
                    name: other.to_string(),
                });
            }
        })
    }
}

/// The metadata block carried by every [`Raster`]: colour interpretation,
/// resolution, offset, and the EXIF-style orientation tag.
///
/// Crate-private: [`Raster::copy`] is the public mutation surface and the
/// [`Raster`] getters are the public read surface. `interpretation` is
/// `None` until set explicitly, so the getter can infer a value from the
/// pixel format the same way libvips assigns an interpretation at image
/// creation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RasterMeta {
    pub(crate) interpretation: Option<Interpretation>,
    pub(crate) xres: f64,
    pub(crate) yres: f64,
    pub(crate) xoffset: i32,
    pub(crate) yoffset: i32,
    pub(crate) orientation: u8,
}

impl Default for RasterMeta {
    fn default() -> Self {
        Self {
            interpretation: None,
            xres: 1.0,
            yres: 1.0,
            xoffset: 0,
            yoffset: 0,
            orientation: 1,
        }
    }
}

/// Builder returned by [`Raster::copy`]: a pixel-identical copy with
/// selectively modified metadata (libvips `vips_copy`).
///
/// Setters not called keep the source raster's values. `build` clones the
/// pixel data, so the source is untouched.
#[derive(Debug)]
#[must_use = "the copy is only produced by calling .build()"]
pub struct RasterCopyBuilder<'a> {
    src: &'a Raster,
    meta: RasterMeta,
}

impl RasterCopyBuilder<'_> {
    /// Set the colour [`Interpretation`] tag.
    pub fn interpretation(mut self, interpretation: Interpretation) -> Self {
        self.meta.interpretation = Some(interpretation);
        self
    }

    /// Set the horizontal resolution in pixels per millimetre.
    pub fn xres(mut self, xres: f64) -> Self {
        self.meta.xres = xres;
        self
    }

    /// Set the vertical resolution in pixels per millimetre.
    pub fn yres(mut self, yres: f64) -> Self {
        self.meta.yres = yres;
        self
    }

    /// Set the horizontal offset of the image origin.
    pub fn xoffset(mut self, xoffset: i32) -> Self {
        self.meta.xoffset = xoffset;
        self
    }

    /// Set the vertical offset of the image origin.
    pub fn yoffset(mut self, yoffset: i32) -> Self {
        self.meta.yoffset = yoffset;
        self
    }

    /// Set the EXIF-style orientation tag (1 to 8; see
    /// [`Raster::autorot`]). Values outside that range are kept verbatim
    /// and treated as "already upright" by `autorot`, matching libvips'
    /// tolerance of malformed tags.
    pub fn orientation(mut self, orientation: u8) -> Self {
        self.meta.orientation = orientation;
        self
    }

    /// Produce the copy: same dimensions, format, and pixels as the
    /// source, with this builder's metadata.
    pub fn build(self) -> Raster {
        let mut out = self.src.clone();
        out.meta = self.meta;
        out
    }
}

/// Build a raster by mapping every output pixel to an input pixel of the
/// same format. `map` receives output coordinates and returns the source
/// coordinates whose whole pixel (all bands) is copied. Metadata is
/// carried over from `src`.
/// Read a flat `bpc`-byte sample as `u32` (native byte order), for the
/// 1-, 2-, and 4-byte sample depths.
fn read_sample_u32(bytes: &[u8], bpc: usize) -> u32 {
    match bpc {
        1 => bytes[0] as u32,
        2 => u16::from_ne_bytes([bytes[0], bytes[1]]) as u32,
        _ => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

/// Write `v` as a flat `bpc`-byte sample (native byte order), for the 1- and
/// 2-byte integer depths; wider depths take the low bytes.
fn write_sample_u32(bytes: &mut [u8], bpc: usize, v: u32) {
    match bpc {
        1 => bytes[0] = v as u8,
        2 => bytes[..2].copy_from_slice(&(v as u16).to_ne_bytes()),
        _ => bytes[..4].copy_from_slice(&v.to_ne_bytes()),
    }
}

/// Position of the cell `(u, v)` (coordinates relative to the ring centre)
/// clockwise along its Chebyshev ring of radius `r`, in `0..8r`. The walk
/// starts at the top-left corner `(-r, -r)` and proceeds right, down, left,
/// then up. `r` must be positive and `(u, v)` must lie on the ring.
fn ring_index(u: i64, v: i64, r: i64) -> i64 {
    if v == -r && u < r {
        // top edge, moving right (includes the top-left corner)
        u + r
    } else if u == r && v < r {
        // right edge, moving down (includes the top-right corner)
        2 * r + (v + r)
    } else if v == r && u > -r {
        // bottom edge, moving left (includes the bottom-right corner)
        4 * r + (r - u)
    } else {
        // left edge, moving up (includes the bottom-left corner)
        6 * r + (r - v)
    }
}

/// Inverse of [`ring_index`]: the cell `(u, v)` at clockwise position `p` in
/// `0..8r` along the Chebyshev ring of radius `r`.
fn ring_cell(p: i64, r: i64) -> (i64, i64) {
    let seg = p / (2 * r);
    let off = p % (2 * r);
    match seg {
        0 => (-r + off, -r),
        1 => (r, -r + off),
        2 => (r - off, r),
        _ => (-r, r - off),
    }
}

fn remap(
    src: &Raster,
    out_w: u32,
    out_h: u32,
    map: impl Fn(u32, u32) -> (u32, u32),
) -> Result<Raster, ConversionError> {
    let bpp = src.format().bytes_per_pixel();
    let mut out = Raster::zeroed(out_w, out_h, src.format())?;
    let sdata = src.data().to_vec();
    let sstride = src.stride();
    let ostride = out.stride();
    let odata = out.data_mut();
    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = map(x, y);
            let so = sy as usize * sstride + sx as usize * bpp;
            let oo = y as usize * ostride + x as usize * bpp;
            odata[oo..oo + bpp].copy_from_slice(&sdata[so..so + bpp]);
        }
    }
    out.meta = src.meta;
    Ok(out)
}

/// The largest `shim` [`Raster::join`] and [`Raster::arrayjoin`] accept,
/// matching the libvips property bound both operations declare:
/// `VIPS_ARG_INT(class, "shim", 5, ..., 0, 1000000, 0)` in `join.c` and the
/// same range in `arrayjoin.c`. GObject refuses anything above it before
/// either operation is even built, and the binary agrees:
/// `vips join --shim 1000001` and `vips arrayjoin --shim 1000001` both fail
/// with the same CRITICAL.
const SHIM_MAX: u32 = 1_000_000;

/// The range [`Raster::arrayjoin`] accepts for an explicit `across`,
/// matching the libvips property bound `arrayjoin.c:400-406` declares at
/// `fe420cf3a`: `VIPS_ARG_INT(class, "across", 4, ..., 1, 1000000, 1)`.
/// GObject refuses anything outside it before the operation is built, so
/// `vips arrayjoin "a.v b.v" z.v --across 0` and `--across 1000001` both
/// fail with a CRITICAL rather than producing a grid. The default (all the
/// images in one row) is not range checked, because vips assigns it
/// straight to the struct field in `vips_arrayjoin_build`
/// (`arrayjoin.c:250-251`) and so bypasses the property check the same way.
const ACROSS_MIN: u32 = 1;
/// Upper end of the [`ACROSS_MIN`] range.
const ACROSS_MAX: u32 = 1_000_000;

/// Where the second image sits relative to the first, in pixels from the
/// first image's origin (`join.c:109-156`).
///
/// Every input is widened to `i64` first, so nothing here can wrap: the
/// dimensions come from `u32` and `shim` is bounded by [`SHIM_MAX`],
/// which puts every intermediate under 2^33. The result is then range
/// checked against the `i32` placement [`Raster::try_insert`] takes and
/// libvips itself computes in.
///
/// [`Align::Centre`] is two separate truncating integer divisions,
/// `in1 / 2 - in2 / 2`, NOT `(in1 - in2) / 2`: for 4 and 3 the C form gives
/// 1 where the combined form gives 0. Under `Centre` and [`Align::High`]
/// the offset goes negative when the second image is the larger one.
///
/// Split out of [`Raster::try_join`] so the range check is testable without
/// building the multi-gigabyte input the real call would need to reach it.
///
/// # Errors
///
/// [`ConversionError::PlacementOffsetOverflow`] if the offset falls
/// outside `i32`.
fn join_placement(
    (w1, h1): (i64, i64),
    (w2, h2): (i64, i64),
    direction: JoinDirection,
    align: Align,
    shim: i64,
) -> Result<(i32, i32), ConversionError> {
    let (x, y) = match direction {
        JoinDirection::Horizontal => {
            let y = match align {
                Align::Low => 0,
                Align::Centre => h1 / 2 - h2 / 2,
                Align::High => h1 - h2,
            };
            (w1 + shim, y)
        }
        JoinDirection::Vertical => {
            let x = match align {
                Align::Low => 0,
                Align::Centre => w1 / 2 - w2 / 2,
                Align::High => w1 - w2,
            };
            (x, h1 + shim)
        }
    };
    let (Ok(ix), Ok(iy)) = (i32::try_from(x), i32::try_from(y)) else {
        return Err(ConversionError::PlacementOffsetOverflow { x, y });
    };
    Ok((ix, iy))
}

impl Raster {
    // ------------------------------------------------------------------
    // Metadata read surface
    // ------------------------------------------------------------------

    /// Number of colour bands (channels) in this raster (libvips `bands`).
    ///
    /// Equal to `self.format().channels()`: `1` for the grayscale formats,
    /// `3` for RGB, `4` for RGBA, and `n` for the multiband / float
    /// carriers. Unlike libvips, where the band count is stored
    /// independently of the sample format, libviprs derives it from the
    /// [`PixelFormat`], so it always agrees with the buffer geometry.
    pub fn bands(&self) -> u32 {
        self.format().channels() as u32
    }

    /// The colour [`Interpretation`] of this raster: the value set by
    /// [`Raster::copy`], or one inferred from the [`PixelFormat`] via
    /// [`Interpretation::for_format`] when none has been set.
    pub fn interpretation(&self) -> Interpretation {
        self.meta
            .interpretation
            .unwrap_or_else(|| Interpretation::for_format(self.format()))
    }

    /// Horizontal resolution in pixels per millimetre (default `1.0`).
    pub fn xres(&self) -> f64 {
        self.meta.xres
    }

    /// Vertical resolution in pixels per millimetre (default `1.0`).
    pub fn yres(&self) -> f64 {
        self.meta.yres
    }

    /// Horizontal offset of the image origin (default `0`).
    pub fn xoffset(&self) -> i32 {
        self.meta.xoffset
    }

    /// Vertical offset of the image origin (default `0`).
    pub fn yoffset(&self) -> i32 {
        self.meta.yoffset
    }

    /// The EXIF-style orientation tag (default `1`, upright). Applied and
    /// cleared by [`Raster::autorot`].
    pub fn orientation(&self) -> u8 {
        self.meta.orientation
    }

    /// Start a metadata-modifying copy (libvips `vips_copy`): same pixels,
    /// selectively replaced metadata.
    ///
    /// ```
    /// # use libviprs::{PixelFormat, Raster};
    /// let im = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
    /// let copy = im.copy().xres(42.0).build();
    /// assert_eq!(copy.xres(), 42.0);
    /// ```
    pub fn copy(&self) -> RasterCopyBuilder<'_> {
        RasterCopyBuilder {
            src: self,
            meta: self.meta,
        }
    }

    // ------------------------------------------------------------------
    // cast
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::cast`].
    ///
    /// Changes the sample format without changing the band count.
    /// Widening (8 to 16 bit) preserves sample values numerically (a `200`
    /// stays `200`); narrowing (16 to 8 bit) clips values above `255`.
    /// Casting to a float format stores the exact sample value as an
    /// `f32` (a `200` becomes `200.0`, never rescaled); casting a float
    /// raster to an unsigned format clips to the target range (`0..=255`
    /// or `0..=65535`) and then **truncates toward zero**, so `1.7`
    /// becomes `1` and `2.5` becomes `2`. `NaN` pins to `0`. Casting to
    /// the current depth retags the format and copies the pixels.
    /// Metadata is carried over.
    ///
    /// Truncating is what `vips_cast` does: "Floats are truncated (not
    /// rounded). Out of range values are clipped" (`cast.c:566-567`).
    /// libviprs rounded to nearest before this, so a float-to-integer
    /// cast could land one above the value vips gives (issue #561).
    ///
    /// Parity with `vips_cast` reaches only as far as [`PixelFormat`]
    /// does. The signed (`char`/`short`/`int`), `double`, and complex
    /// targets have no representation here, so a cast to one of those is
    /// not expressible rather than wrong; and the `shift` option, which
    /// rescales instead of clipping when the depth changes, is not
    /// implemented, so this is always the non-shifting behaviour.
    ///
    /// # Errors
    ///
    /// [`ConversionError::CastBandCountChange`] if `format` has a
    /// different band count, or [`ConversionError::Raster`] on allocation
    /// failure.
    pub fn try_cast(&self, format: PixelFormat) -> Result<Raster, ConversionError> {
        let from = self.format();
        if from.channels() != format.channels() {
            return Err(ConversionError::CastBandCountChange {
                from,
                from_bands: from.channels(),
                to: format,
                to_bands: format.channels(),
            });
        }
        let in_bpc = from.bytes_per_channel();
        let out_bpc = format.bytes_per_channel();
        let mut out = Raster::zeroed(self.width(), self.height(), format)?;
        let samples = self.width() as usize * self.height() as usize * from.channels();
        let sdata = self.data();
        let odata = out.data_mut();
        if !from.is_float() && !format.is_float() {
            // Unsigned to unsigned: the integer path, byte-identical to the
            // pre-float behaviour.
            for i in 0..samples {
                let v = read_flat(sdata, in_bpc, i);
                let v = if out_bpc == 1 { v.min(255) } else { v };
                write_flat(odata, out_bpc, i, v);
            }
        } else {
            // A float endpoint: go through f64, which holds every u8/u16
            // and f32 sample exactly.
            for i in 0..samples {
                let v: f64 = if from.is_float() {
                    read_f32_flat(sdata, i) as f64
                } else {
                    read_flat(sdata, in_bpc, i) as f64
                };
                if format.is_float() {
                    write_f32_flat(odata, i, v as f32);
                } else {
                    write_flat(odata, out_bpc, i, cast_float_sample(v, out_bpc));
                }
            }
        }
        out.meta = self.meta;
        Ok(out)
    }

    /// Cast the samples to a new bit depth, clipping on overflow (libvips
    /// `vips_cast`). Panicking form of [`Raster::try_cast`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_cast`].
    #[track_caller]
    pub fn cast(&self, format: PixelFormat) -> Raster {
        expect_conv("cast", self.try_cast(format))
    }

    // ------------------------------------------------------------------
    // flips
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::fliphor`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_fliphor(&self) -> Result<Raster, ConversionError> {
        let w = self.width();
        remap(self, w, self.height(), |x, y| (w - 1 - x, y))
    }

    /// Mirror left-right (libvips `vips_flip` with
    /// `VIPS_DIRECTION_HORIZONTAL`). Panicking form of
    /// [`Raster::try_fliphor`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_fliphor`].
    #[track_caller]
    pub fn fliphor(&self) -> Raster {
        expect_conv("fliphor", self.try_fliphor())
    }

    /// Fallible form of [`Raster::flipver`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_flipver(&self) -> Result<Raster, ConversionError> {
        let h = self.height();
        remap(self, self.width(), h, |x, y| (x, h - 1 - y))
    }

    /// Mirror top-bottom (libvips `vips_flip` with
    /// `VIPS_DIRECTION_VERTICAL`). Panicking form of
    /// [`Raster::try_flipver`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_flipver`].
    #[track_caller]
    pub fn flipver(&self) -> Raster {
        expect_conv("flipver", self.try_flipver())
    }

    // ------------------------------------------------------------------
    // rot / autorot
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::rot`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_rot(&self, angle: Angle) -> Result<Raster, ConversionError> {
        let w = self.width();
        let h = self.height();
        match angle {
            Angle::D0 => Ok(self.clone()),
            Angle::D90 => remap(self, h, w, |x, y| (y, h - 1 - x)),
            Angle::D180 => remap(self, w, h, |x, y| (w - 1 - x, h - 1 - y)),
            Angle::D270 => remap(self, h, w, |x, y| (w - 1 - y, x)),
        }
    }

    /// Rotate by a right-angle [`Angle`] (libvips `vips_rot`). `D90` is a
    /// quarter turn clockwise; `D90` and `D270` swap the dimensions.
    /// Panicking form of [`Raster::try_rot`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_rot`].
    #[track_caller]
    pub fn rot(&self, angle: Angle) -> Raster {
        expect_conv("rot", self.try_rot(angle))
    }

    /// Fallible form of [`Raster::rot45`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::NotOddSquare`] unless the image is an odd-sided
    /// square, or [`ConversionError::Raster`] on allocation failure.
    pub fn try_rot45(&self, angle: Angle45) -> Result<Raster, ConversionError> {
        let size = self.width();
        if self.height() != size || size.is_multiple_of(2) {
            return Err(ConversionError::NotOddSquare {
                width: self.width(),
                height: self.height(),
            });
        }
        let k = angle.steps();
        if k == 0 {
            return Ok(self.clone());
        }
        let centre = (size / 2) as i64;
        // Each concentric square ring (Chebyshev radius `r`, perimeter `8r`)
        // rotates clockwise by `k * r` cells, i.e. `k` eighths of a turn.
        // `remap` reads the source cell for every output cell, so the output
        // ring position `p` pulls from input position `p - k * r`.
        remap(self, size, size, |x, y| {
            let u = x as i64 - centre;
            let v = y as i64 - centre;
            let r = u.abs().max(v.abs());
            if r == 0 {
                return (x, y);
            }
            let perim = 8 * r;
            let p_out = ring_index(u, v, r);
            let shift = (k as i64 * r) % perim;
            let p_src = (p_out - shift).rem_euclid(perim);
            let (su, sv) = ring_cell(p_src, r);
            ((su + centre) as u32, (sv + centre) as u32)
        })
    }

    /// Rotate an odd-sided square image by a multiple of 45 degrees
    /// clockwise (libvips `vips_rot45`). Each concentric square ring is
    /// rotated by an eighth of a turn per step, so the transform is an exact
    /// pixel permutation with an exact inverse. Panicking form of
    /// [`Raster::try_rot45`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_rot45`].
    #[track_caller]
    pub fn rot45(&self, angle: Angle45) -> Raster {
        expect_conv("rot45", self.try_rot45(angle))
    }

    /// Swap the byte order within every sample (libvips `vips_byteswap`):
    /// for a 16-bit sample the two bytes are exchanged, for a 32-bit float
    /// the four bytes are reversed, and for an 8-bit sample it is a no-op.
    /// Applying it twice is always an identity.
    pub fn byteswap(&self) -> Raster {
        let bpc = self.format().bytes_per_channel();
        let mut out = self.clone();
        if bpc > 1 {
            for sample in out.data_mut().chunks_exact_mut(bpc) {
                sample.reverse();
            }
        }
        out
    }

    /// Take the most significant byte of every sample (libvips `vips_msb`),
    /// producing an 8-bit image. With `band = Some(b)` only band `b` is kept,
    /// giving a single-band result; with `None` every band is converted.
    ///
    /// # Panics
    ///
    /// Panics if `band` is out of range for the image.
    #[track_caller]
    pub fn msb(&self, band: Option<u32>) -> Raster {
        expect_conv("msb", self.try_msb(band))
    }

    /// Fallible form of [`Raster::msb`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Band`] if `band` is out of range, or
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_msb(&self, band: Option<u32>) -> Result<Raster, ConversionError> {
        let fmt = self.format();
        let bpc = fmt.bytes_per_channel();
        let bands = fmt.channels();
        let shift = ((bpc - 1) * 8) as u32;
        let (src_bands, out_bands): (Vec<usize>, usize) = match band {
            None => ((0..bands).collect(), bands),
            Some(b) => {
                let b = b as usize;
                if b >= bands {
                    return Err(ConversionError::Band(BandError::BandOutOfRange {
                        band: b as i64,
                        bands,
                    }));
                }
                (vec![b], 1)
            }
        };
        let out_fmt = PixelFormat::with_channels(out_bands, 1)
            .expect("an 8-bit format exists for a band count already carried by this raster");
        let w = self.width();
        let h = self.height();
        let src = self.data();
        let sstride = self.stride();
        let mut out = Raster::zeroed(w, h, out_fmt)?;
        let ostride = out.stride();
        let odata = out.data_mut();
        for y in 0..h as usize {
            for x in 0..w as usize {
                for (oi, &sb) in src_bands.iter().enumerate() {
                    let so = y * sstride + (x * bands + sb) * bpc;
                    let sample = read_sample_u32(&src[so..so + bpc], bpc);
                    let oo = y * ostride + x * out_bands + oi;
                    odata[oo] = (sample >> shift) as u8;
                }
            }
        }
        Ok(out)
    }

    /// Reshape a tall stack of `across * down` tiles, each `tile_height` high
    /// and the full image width, into an `across`-by-`down` grid of tiles
    /// (libvips `vips_grid`). Tile `i` (counted from the top of the input)
    /// lands at grid column `i % across`, row `i / across`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_grid`].
    #[track_caller]
    pub fn grid(&self, tile_height: u32, across: u32, down: u32) -> Raster {
        expect_conv("grid", self.try_grid(tile_height, across, down))
    }

    /// Fallible form of [`Raster::grid`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::DimensionMismatch`] if the input height is not
    /// `tile_height * across * down`, or [`ConversionError::Raster`] on
    /// allocation failure.
    pub fn try_grid(
        &self,
        tile_height: u32,
        across: u32,
        down: u32,
    ) -> Result<Raster, ConversionError> {
        let tile_w = self.width();
        if tile_height == 0
            || across == 0
            || down == 0
            || self.height() != tile_height * across * down
        {
            return Err(ConversionError::DimensionMismatch {
                expected_w: tile_w,
                expected_h: tile_height.saturating_mul(across).saturating_mul(down),
                got_w: tile_w,
                got_h: self.height(),
            });
        }
        let out_w = tile_w * across;
        let out_h = tile_height * down;
        remap(self, out_w, out_h, move |x, y| {
            let col = x / tile_w;
            let row = y / tile_height;
            let tile = row * across + col;
            let sx = x % tile_w;
            let sy = tile * tile_height + (y % tile_height);
            (sx, sy)
        })
    }

    /// Flatten an image with an alpha channel against a background (libvips
    /// `vips_flatten`): the last band is treated as alpha and removed, and
    /// each remaining band `b` becomes
    /// `src[b] * alpha/max + background[b] * (max - alpha)/max`, where `max`
    /// is the sample maximum for the depth. `background` defaults to zero
    /// (black); a single value is used for every band, otherwise one value
    /// per output band is expected.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_flatten`].
    #[track_caller]
    pub fn flatten(&self, background: Option<&[f64]>) -> Raster {
        expect_conv("flatten", self.try_flatten(background))
    }

    /// Fallible form of [`Raster::flatten`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::BandCountMismatch`] if the image has no alpha band
    /// to remove (fewer than two bands) or `background` has the wrong length,
    /// or [`ConversionError::Raster`] on allocation failure.
    pub fn try_flatten(&self, background: Option<&[f64]>) -> Result<Raster, ConversionError> {
        let fmt = self.format();
        let bpc = fmt.bytes_per_channel();
        let in_bands = fmt.channels();
        if in_bands < 2 {
            return Err(ConversionError::BandCountMismatch {
                expected: 2,
                got: in_bands,
            });
        }
        let out_bands = in_bands - 1;
        let bg: Vec<f64> = match background {
            None => vec![0.0; out_bands],
            Some(v) if v.len() == 1 => vec![v[0]; out_bands],
            Some(v) if v.len() == out_bands => v.to_vec(),
            Some(v) => {
                return Err(ConversionError::BandCountMismatch {
                    expected: out_bands,
                    got: v.len(),
                });
            }
        };
        let out_fmt = PixelFormat::with_channels(out_bands, bpc)
            .expect("a format exists for a band count already carried by this raster");
        let max = ((1u64 << (bpc * 8)) - 1) as f64;
        let w = self.width();
        let h = self.height();
        let src = self.data();
        let sstride = self.stride();
        let mut out = Raster::zeroed(w, h, out_fmt)?;
        let ostride = out.stride();
        let odata = out.data_mut();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let apos = y * sstride + (x * in_bands + (in_bands - 1)) * bpc;
                let alpha = read_sample_u32(&src[apos..apos + bpc], bpc) as f64;
                for (b, &bgb) in bg.iter().enumerate() {
                    let so = y * sstride + (x * in_bands + b) * bpc;
                    let s = read_sample_u32(&src[so..so + bpc], bpc) as f64;
                    let val = s * alpha / max + bgb * (max - alpha) / max;
                    let v = val.round().clamp(0.0, max) as u64;
                    let oo = y * ostride + (x * out_bands + b) * bpc;
                    write_sample_u32(&mut odata[oo..oo + bpc], bpc, v as u32);
                }
            }
        }
        Ok(out)
    }

    /// Blend two images under a boolean/mask condition (libvips
    /// `vips_ifthenelse`, no blend): wherever a `self` sample is non-zero the
    /// `then` pixel is taken, otherwise the `otherwise` pixel. A single-band
    /// condition selects for every band of the operands. Mixed branch depths
    /// promote to a common format first (libvips `vips__formatalike`): an
    /// `Rgb8` branch paired with an `Rgb16` branch yields `Rgb16`, and any
    /// float branch switches both to float.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_ifthenelse`].
    #[track_caller]
    pub fn ifthenelse(&self, then: &Raster, otherwise: &Raster) -> Raster {
        expect_conv("ifthenelse", self.try_ifthenelse(then, otherwise))
    }

    /// Fallible form of [`Raster::ifthenelse`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::DimensionMismatch`] if the three images disagree on
    /// size, [`ConversionError::BandCountMismatch`] if `then` and `otherwise`
    /// disagree on band count or the condition is neither single-band nor a
    /// band match, or [`ConversionError::Raster`] on allocation failure.
    pub fn try_ifthenelse(
        &self,
        then: &Raster,
        otherwise: &Raster,
    ) -> Result<Raster, ConversionError> {
        let (w, h) = (self.width(), self.height());
        for other in [then, otherwise] {
            if (other.width(), other.height()) != (w, h) {
                return Err(ConversionError::DimensionMismatch {
                    expected_w: w,
                    expected_h: h,
                    got_w: other.width(),
                    got_h: other.height(),
                });
            }
        }
        let bands = then.format().channels();
        if otherwise.format().channels() != bands {
            return Err(ConversionError::BandCountMismatch {
                expected: bands,
                got: otherwise.format().channels(),
            });
        }
        let cond_bands = self.format().channels();
        if cond_bands != 1 && cond_bands != bands {
            return Err(ConversionError::BandCountMismatch {
                expected: bands,
                got: cond_bands,
            });
        }
        // vips__formatalike: bring both branches to one numeric format
        // before the per-pixel select, so a narrower branch is never read
        // at the wider branch's stride. The common depth is the widest
        // input depth (any float side switches both to float), the same
        // promotion the band and arrayjoin paths use. Equal-format inputs
        // keep their format tag untouched.
        let out_fmt = if then.format() == otherwise.format() {
            then.format()
        } else {
            let bpc = then
                .format()
                .bytes_per_channel()
                .max(otherwise.format().bytes_per_channel());
            PixelFormat::with_channels(bands, bpc)
                .expect("band count comes from a valid input format")
        };
        let then_cast;
        let then = if then.format() == out_fmt {
            then
        } else {
            then_cast = then.try_cast(out_fmt)?;
            &then_cast
        };
        let otherwise_cast;
        let otherwise = if otherwise.format() == out_fmt {
            otherwise
        } else {
            otherwise_cast = otherwise.try_cast(out_fmt)?;
            &otherwise_cast
        };
        let cbpc = self.format().bytes_per_channel();
        let cdata = self.data();
        let cstride = self.stride();
        let mut out = Raster::zeroed(w, h, out_fmt)?;
        let bpp = out_fmt.bytes_per_pixel();
        let bpc = out_fmt.bytes_per_channel();
        let (tstride, ostride) = (then.stride(), out.stride());
        let ostride_o = otherwise.stride();
        let tdata = then.data();
        let odata_o = otherwise.data();
        let out_data = out.data_mut();
        for y in 0..h as usize {
            for x in 0..w as usize {
                for b in 0..bands {
                    let cb = if cond_bands == 1 { 0 } else { b };
                    let co = y * cstride + (x * cond_bands + cb) * cbpc;
                    let take_then = read_sample_u32(&cdata[co..co + cbpc], cbpc) != 0;
                    let sample_stride = if take_then { tstride } else { ostride_o };
                    let sample_data = if take_then { tdata } else { odata_o };
                    let so = y * sample_stride + (x * bands + b) * bpc;
                    let oo = y * ostride + x * bpp + b * bpc;
                    out_data[oo..oo + bpc].copy_from_slice(&sample_data[so..so + bpc]);
                }
            }
        }
        Ok(out)
    }

    /// Flip along the top-left to bottom-right diagonal (EXIF
    /// orientation 5).
    fn transpose(&self) -> Result<Raster, ConversionError> {
        let w = self.width();
        let h = self.height();
        remap(self, h, w, |x, y| (y, x))
    }

    /// Flip along the bottom-left to top-right diagonal (EXIF
    /// orientation 7).
    fn transverse(&self) -> Result<Raster, ConversionError> {
        let w = self.width();
        let h = self.height();
        remap(self, h, w, |x, y| (w - 1 - y, h - 1 - x))
    }

    /// Fallible form of [`Raster::autorot`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_autorot(&self) -> Result<Raster, ConversionError> {
        let mut out = match self.meta.orientation {
            2 => self.try_fliphor()?,
            3 => self.try_rot(Angle::D180)?,
            4 => self.try_flipver()?,
            5 => self.transpose()?,
            6 => self.try_rot(Angle::D90)?,
            7 => self.transverse()?,
            8 => self.try_rot(Angle::D270)?,
            // 1 is upright; 0 and 9.. are malformed tags libvips ignores.
            _ => self.clone(),
        };
        out.meta.orientation = 1;
        Ok(out)
    }

    /// Apply the rotation or mirror implied by the orientation tag, then
    /// clear the tag to `1` (libvips `vips_autorot`). All eight EXIF
    /// orientations are handled; a missing or malformed tag leaves the
    /// pixels untouched. Panicking form of [`Raster::try_autorot`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_autorot`].
    #[track_caller]
    pub fn autorot(&self) -> Raster {
        expect_conv("autorot", self.try_autorot())
    }

    // ------------------------------------------------------------------
    // wrap
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::wrap`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_wrap(&self) -> Result<Raster, ConversionError> {
        let w = self.width();
        let h = self.height();
        let dx = w / 2;
        let dy = h / 2;
        remap(self, w, h, |x, y| ((x + dx) % w, (y + dy) % h))
    }

    /// Toroidally shift the image so the pixel at the centre
    /// `(width / 2, height / 2)` moves to the origin, swapping the four
    /// quadrants (libvips `vips_wrap` with default offsets). Useful for
    /// re-centring the DC component of a Fourier transform. Panicking form
    /// of [`Raster::try_wrap`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_wrap`].
    #[track_caller]
    pub fn wrap(&self) -> Raster {
        expect_conv("wrap", self.try_wrap())
    }

    // ------------------------------------------------------------------
    // gamma
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::gamma`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::InvalidGammaExponent`] if `exponent` is not a
    /// finite value greater than zero, or [`ConversionError::Raster`] on
    /// allocation failure.
    pub fn try_gamma(&self, exponent: Option<f64>) -> Result<Raster, ConversionError> {
        let exponent = exponent.unwrap_or(1.0 / 2.4);
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(ConversionError::InvalidGammaExponent { exponent });
        }
        let power = 1.0 / exponent;
        let bpc = self.format().bytes_per_channel();
        let mx = if bpc == 1 { 255u32 } else { 65535u32 };
        let mxf = mx as f64;
        // Match vips_gamma bit-for-bit (gamma.c): the curve is built as an
        // identity LUT put through `vips_pow_const1(power)` then
        // `vips_linear1(mx / pow(mx, power), 0)`, then cast back to the
        // integer format. `vips_pow_const1` promotes uchar/ushort to a
        // single-precision float image. For a float image `vips_linear1`
        // takes its FLOOP path, which pre-rounds the scale coefficient to
        // f32 (`linear->a_float[]`) and computes `a_float * powered_f32`
        // entirely in f32 (both operands single-precision), storing the f32
        // result back to the float image. The final `vips_cast` to the
        // integer format then TRUNCATES toward zero (it does not round to
        // nearest). Reproduce that ordering exactly so parity holds by
        // construction: coefficient rounded to f32, pow in f64 truncated to
        // f32, product in f32, then clamp and truncate. NB. keeping the
        // coefficient in f64 (multiplying `(double)coeff * (double)powered`)
        // does NOT match vips and diverges by 1 LSB on truncation edges
        // (e.g. ushort 23843 -> 5788 vs vips 5789, uchar 255@power2.1 -> 255
        // vs vips 254). Verified against vips 8.18.4 over the full 0..=255
        // and 0..=65535 default domains and at custom powers 2.1/2.3
        // (0 divergences).
        let scale = (mxf / mxf.powf(power)) as f32;
        let mxf32 = mxf as f32;
        let lut: Vec<u32> = (0..=mx)
            .map(|i| {
                let powered = (i as f64).powf(power) as f32;
                (scale * powered).clamp(0.0, mxf32) as u32
            })
            .collect();
        let mut out = Raster::zeroed(self.width(), self.height(), self.format())?;
        let samples = self.width() as usize * self.height() as usize * self.format().channels();
        let sdata = self.data();
        let odata = out.data_mut();
        for i in 0..samples {
            write_flat(odata, bpc, i, lut[read_flat(sdata, bpc, i) as usize]);
        }
        out.meta = self.meta;
        Ok(out)
    }

    /// Apply a gamma curve to every band (libvips `vips_gamma`): each
    /// sample becomes `v.powf(1.0 / exponent)`, rescaled so the format
    /// maximum maps to itself. The default exponent is `1.0 / 2.4`, so
    /// `gamma(None)` raises samples to the power `2.4`, exactly the curve
    /// the ported `test_gamma` predicts. Panicking form of
    /// [`Raster::try_gamma`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_gamma`].
    #[track_caller]
    pub fn gamma(&self, exponent: Option<f64>) -> Raster {
        expect_conv("gamma", self.try_gamma(exponent))
    }

    // ------------------------------------------------------------------
    // falsecolour
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::falsecolour`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] on allocation failure.
    pub fn try_falsecolour(&self) -> Result<Raster, ConversionError> {
        let channels = self.format().channels();
        let bpc = self.format().bytes_per_channel();
        let mut out = Raster::zeroed(self.width(), self.height(), PixelFormat::Rgb8)?;
        let pixels = self.width() as usize * self.height() as usize;
        let sdata = self.data();
        let odata = out.data_mut();
        for p in 0..pixels {
            let v = read_flat(sdata, bpc, p * channels).min(255) as usize;
            odata[p * 3..p * 3 + 3].copy_from_slice(&FALSECOLOUR_PET[v]);
        }
        out.meta = self.meta;
        out.meta.interpretation = Some(Interpretation::Srgb);
        Ok(out)
    }

    /// Map the image through the libvips PET false-colour scale
    /// (`vips_falsecolour`): band 0 is extracted, clip-cast to 8-bit, and
    /// looked up in the 256-entry LUT, producing a 3-band sRGB image of
    /// the same dimensions. Matching `falsecolour.c`, no colourspace
    /// conversion is performed first, so a multi-band input is reduced to
    /// its first band and a 16-bit input clips at 255. Panicking form of
    /// [`Raster::try_falsecolour`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_falsecolour`].
    #[track_caller]
    pub fn falsecolour(&self) -> Raster {
        expect_conv("falsecolour", self.try_falsecolour())
    }

    // ------------------------------------------------------------------
    // addalpha
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::addalpha`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Band`] if the result would exceed the supported
    /// band count, or [`ConversionError::Raster`] on allocation failure.
    pub fn try_addalpha(&self) -> Result<Raster, ConversionError> {
        let max = if self.format().bytes_per_channel() == 1 {
            255.0
        } else {
            65535.0
        };
        let mut out = self.try_bandjoin_const(max)?;
        out.meta = self.meta;
        Ok(out)
    }

    /// Append one fully-opaque alpha band: `255` for 8-bit formats,
    /// `65535` for 16-bit (libvips `vips_addalpha`). `Rgb8` becomes
    /// `Rgba8`; a mono input gains a second band and becomes a `Multi8` /
    /// `Multi16` intermediate. Panicking form of
    /// [`Raster::try_addalpha`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_addalpha`].
    #[track_caller]
    pub fn addalpha(&self) -> Raster {
        expect_conv("addalpha", self.try_addalpha())
    }

    // ------------------------------------------------------------------
    // arrayjoin
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::arrayjoin`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::EmptyInput`] for an empty list,
    /// [`ConversionError::FloatFormatUnsupported`] if any input is a float
    /// raster (the sample copy is unsigned-only),
    /// [`ConversionError::ShimTooLarge`] for a `shim` above `1000000`, the
    /// bound libvips declares on the property,
    /// [`ConversionError::AcrossOutOfRange`] for an explicit `across`
    /// outside the `1..=1000000` bound libvips declares on its property,
    /// [`ConversionError::BandCountMismatch`] if band counts differ and
    /// the smaller is not 1, [`ConversionError::SizeOverflow`] if the
    /// grid exceeds `u32` dimensions, or [`ConversionError::Raster`] on
    /// allocation failure.
    pub fn try_arrayjoin(
        images: &[&Raster],
        across: Option<u32>,
        shim: Option<u32>,
    ) -> Result<Raster, ConversionError> {
        if images.is_empty() {
            return Err(ConversionError::EmptyInput { op: "arrayjoin" });
        }
        // The sample copy below is `read_flat` / `write_flat`, which are
        // unsigned-only and panic on 4-byte samples, so a float input has to
        // be refused here rather than panicking out of a fallible method.
        // Same reason as `try_join`: `space_depth` maps Lab, Lch, OkLab,
        // OkLCh, XYZ, scRGB and Yxy all to F32, so a `colourspace` result is
        // already float.
        if images.iter().any(|i| i.format().is_float()) {
            return Err(ConversionError::FloatFormatUnsupported { op: "arrayjoin" });
        }
        let n = u32::try_from(images.len()).unwrap_or(u32::MAX);
        // vips does not clamp `across` to the image count: it lays out a grid
        // that many cells wide and leaves the trailing cells background
        // (`arrayjoin.c:255-268`). It refuses an out-of-range value at the
        // GObject property boundary instead; see `ACROSS_MIN`.
        if let Some(across) = across
            && !(ACROSS_MIN..=ACROSS_MAX).contains(&across)
        {
            return Err(ConversionError::AcrossOutOfRange {
                across,
                min: ACROSS_MIN,
                max: ACROSS_MAX,
            });
        }
        let across = across.unwrap_or(n);
        let down = n.div_ceil(across);
        let shim = shim.unwrap_or(0);
        // `arrayjoin` declares the same 0..=1000000 property range `join`
        // does, so the same bound applies; see `SHIM_MAX`.
        if shim > SHIM_MAX {
            return Err(ConversionError::ShimTooLarge {
                shim,
                max: SHIM_MAX,
            });
        }

        let bands = images
            .iter()
            .map(|i| i.format().channels())
            .max()
            .expect("images is non-empty");
        let bpc = images
            .iter()
            .map(|i| i.format().bytes_per_channel())
            .max()
            .expect("images is non-empty");
        for img in images {
            let c = img.format().channels();
            if c != bands && c != 1 {
                return Err(ConversionError::BandCountMismatch {
                    expected: bands,
                    got: c,
                });
            }
        }
        let cell_w = images
            .iter()
            .map(|i| i.width())
            .max()
            .expect("images is non-empty");
        let cell_h = images
            .iter()
            .map(|i| i.height())
            .max()
            .expect("images is non-empty");

        let out_w64 = across as u64 * cell_w as u64 + (across as u64 - 1) * shim as u64;
        let out_h64 = down as u64 * cell_h as u64 + (down as u64 - 1) * shim as u64;
        let (Ok(out_w), Ok(out_h)) = (u32::try_from(out_w64), u32::try_from(out_h64)) else {
            return Err(ConversionError::SizeOverflow {
                width: out_w64,
                height: out_h64,
            });
        };

        let fmt = PixelFormat::with_channels(bands, bpc)
            .expect("band count comes from valid input formats");
        let mut out = Raster::zeroed(out_w, out_h, fmt)?;
        let odata = out.data_mut();
        for (k, img) in images.iter().enumerate() {
            let col = k as u32 % across;
            let row = k as u32 / across;
            let ox = col as usize * (cell_w as usize + shim as usize);
            let oy = row as usize * (cell_h as usize + shim as usize);
            let ichannels = img.format().channels();
            let ibpc = img.format().bytes_per_channel();
            let idata = img.data();
            let iw = img.width() as usize;
            for y in 0..img.height() as usize {
                for x in 0..iw {
                    let si = (y * iw + x) * ichannels;
                    let oi = ((oy + y) * out_w as usize + ox + x) * bands;
                    for b in 0..bands {
                        let sb = if ichannels == 1 { 0 } else { b };
                        write_flat(odata, bpc, oi + b, read_flat(idata, ibpc, si + sb));
                    }
                }
            }
        }
        out.meta = images[0].meta;
        // Same re-stamp as `join`: `bandalike` can give the grid more bands
        // than images[0] has, and images[0]'s interpretation then describes
        // an image that no longer exists. `space_bands(Bw) == 1`, so a
        // grid left tagged `b-w` reads downstream as grey plus passthrough
        // extras. Drop the tag and let the getter infer from the format. A
        // depth-only promotion keeps the tag, matching vips.
        if out.bands() != images[0].bands() {
            out.meta.interpretation = None;
        }
        Ok(out)
    }

    /// Tile a list of images into a grid (libvips `vips_arrayjoin`).
    ///
    /// `across` is the number of images per row (default: all of them in
    /// one row); `shim` is the gap in pixels between cells (default 0,
    /// maximum `1000000` as in libvips). `across` is **not** clamped to the
    /// number of images: a value larger than the list lays out that many
    /// cells wide and leaves the trailing ones background, so two images
    /// with `across = 5` give a 5-cell row, not a 2-cell one. An explicit
    /// `across` outside `1..=1000000` is refused
    /// ([`ConversionError::AcrossOutOfRange`]), which is the range libvips
    /// declares on the property. Every cell is the size of the
    /// largest input; smaller images sit at the top-left of their cell and
    /// the remainder (and any trailing empty cells and shim gaps) is
    /// filled with black, libvips' default background. Band counts are
    /// aligned like libvips `bandalike` (a one-band image is replicated up
    /// to the widest count) and depths promote numerically to the widest
    /// input. The result carries the metadata of the first image, except
    /// that a band promotion drops the interpretation so it is inferred
    /// from the result format instead of describing the first image's
    /// narrower band count. Float rasters are not supported yet on any
    /// input. Panicking form of [`Raster::try_arrayjoin`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_arrayjoin`].
    #[track_caller]
    pub fn arrayjoin(images: &[&Raster], across: Option<u32>, shim: Option<u32>) -> Raster {
        expect_conv("arrayjoin", Self::try_arrayjoin(images, across, shim))
    }

    // ------------------------------------------------------------------
    // join
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::join`].
    ///
    /// # Errors
    ///
    /// Checked here, before anything is placed or allocated:
    ///
    /// * [`ConversionError::FloatFormatUnsupported`] if either input is a
    ///   float raster. The placement path underneath reads samples as `u8`
    ///   or `u16` and panics on 4-byte ones, so a float input is rejected
    ///   up front rather than delegated into a panic.
    /// * [`ConversionError::ShimTooLarge`] for a `shim` above `1000000`,
    ///   the bound libvips declares on the property.
    /// * [`ConversionError::PlacementOffsetOverflow`] if the offset the
    ///   second image would sit at falls outside `i32`, the range libvips
    ///   places images in.
    ///
    /// Delegated to [`Raster::try_insert`], arriving as
    /// [`ConversionError::Extract`]:
    ///
    /// * [`crate::extract::ExtractError::BandCountMismatch`] when the band
    ///   counts differ and neither is 1.
    /// * [`crate::extract::ExtractError::BackgroundLengthMismatch`] for a
    ///   background vector whose length is neither 1 nor the result band
    ///   count.
    /// * [`crate::extract::ExtractError::SizeOverflow`] if the joined
    ///   canvas would not fit `u32` dimensions.
    /// * [`crate::extract::ExtractError::Raster`] on allocation failure,
    ///   including [`crate::raster::RasterError::ByteBudgetExceeded`] for a
    ///   canvas that fits `u32` but not the allocation budget. Note this is
    ///   nested inside [`ConversionError::Extract`]; `try_join` never
    ///   constructs [`ConversionError::Raster`] itself.
    ///
    /// Delegated to [`Raster::try_extract_area`] for the `expand`-false
    /// crop, also arriving as [`ConversionError::Extract`]:
    ///
    /// * [`crate::extract::ExtractError::EmptyArea`] if the crop would be
    ///   zero-sized, and
    ///   [`crate::extract::ExtractError::AreaOutOfBounds`] if it would
    ///   leave the canvas. Neither is reachable through `join` as the
    ///   placement is computed today; they are listed because the crop can
    ///   raise them and a `#[non_exhaustive]` match should expect them.
    pub fn try_join(
        &self,
        other: &Raster,
        direction: JoinDirection,
        expand: bool,
        shim: Option<u32>,
        background: Option<&[f64]>,
        align: Option<Align>,
    ) -> Result<Raster, ConversionError> {
        // The placement path (`insert` -> `blit`) is unsigned-only and
        // panics on 4-byte samples. Float input is not exotic: `space_depth`
        // maps every one of Lab, Lch, OkLab, OkLCh, XYZ, scRGB and Yxy to
        // F32, so `im.colourspace(Lab).join(..)` arrives here as float.
        // Reject it as a typed error rather than delegating into a panic
        // from a fallible method.
        if self.format().is_float() || other.format().is_float() {
            return Err(ConversionError::FloatFormatUnsupported { op: "join" });
        }
        let align = align.unwrap_or(Align::Low);
        let shim = shim.unwrap_or(0);
        // libvips carries this bound in the property declaration, so
        // `vips join --shim 1000001` never reaches the operation. Widening
        // to `u32` kept the lower bound and dropped the upper one, and
        // without it one argument buys a multi-gigabyte canvas out of two
        // 3x2 inputs, each raster staying under the per-raster budget.
        if shim > SHIM_MAX {
            return Err(ConversionError::ShimTooLarge {
                shim,
                max: SHIM_MAX,
            });
        }
        let shim = i64::from(shim);
        let (w1, h1) = (i64::from(self.width()), i64::from(self.height()));
        let (w2, h2) = (i64::from(other.width()), i64::from(other.height()));

        let (ix, iy) = join_placement((w1, h1), (w2, h2), direction, align, shim)?;
        let (x, y) = (i64::from(ix), i64::from(iy));

        // join.c:158-162: the insert ALWAYS expands, whatever the caller
        // asked for, so nothing is lost before the crop below decides.
        let joined = self.try_insert(other, ix, iy, true, background)?;

        // join.c:164-207: the caller's `expand` only selects this crop.
        let mut out = if expand {
            joined
        } else {
            let (jw, jh) = (i64::from(joined.width()), i64::from(joined.height()));
            let (left, top, width, height) = match direction {
                JoinDirection::Horizontal => (0, y.max(0) - y, jw, h1.min(h2)),
                JoinDirection::Vertical => (x.max(0) - x, 0, w1.min(w2), jh),
            };
            if left != 0 || top != 0 || width != jw || height != jh {
                joined.try_extract_area(left as u32, top as u32, width as u32, height as u32)?
            } else {
                joined
            }
        };
        out.meta = self.meta;
        // libvips `bandalike` promotes a one-band input up to the other's
        // band count, so the result can have more bands than in1 has. in1's
        // interpretation then describes an image that no longer exists:
        // `vips join` of 1-band `b-w` with 3-band `srgb` reports `3 bands,
        // srgb`, and keeping `b-w` here is not cosmetic, since
        // `space_bands(Bw) == 1` makes a later `colourspace` read two of the
        // three bands as passthrough extras and hand back 5 bands of
        // garbage. Drop the tag so the getter infers one from the format
        // instead, the same re-stamp `composite2` does for the same reason.
        // A depth-only promotion is left alone on purpose: vips keeps `b-w`
        // when joining `b-w` uchar with `grey16`, so the trigger is
        // specifically the band count changing.
        if out.bands() != self.bands() {
            out.meta.interpretation = None;
        }
        Ok(out)
    }

    /// Join this image and `other` left-right or top-bottom (libvips
    /// `vips_join`).
    ///
    /// `shim` is the gap in pixels between the two (default 0, maximum
    /// `1000000` as in libvips) and `align` says which edge they line up on
    /// (default [`Align::Low`]). With `expand` false the result is cropped
    /// back to the smaller of the two along the shared axis, which is
    /// libvips' default; with `expand` true it is the bounding box of both,
    /// and `background` (black when `None`) fills whatever neither image
    /// covers, including the shim gap. Band counts and depths unify exactly
    /// as [`Raster::insert`] does. The result carries this image's
    /// metadata, except that a band promotion drops the interpretation so
    /// it is inferred from the result format rather than describing this
    /// image's narrower band count. Float rasters are not supported yet on
    /// either side.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_join`].
    #[track_caller]
    pub fn join(
        &self,
        other: &Raster,
        direction: JoinDirection,
        expand: bool,
        shim: Option<u32>,
        background: Option<&[f64]>,
        align: Option<Align>,
    ) -> Raster {
        expect_conv(
            "join",
            self.try_join(other, direction, expand, shim, background, align),
        )
    }

    // ------------------------------------------------------------------
    // generators: grey, identity, switch
    // ------------------------------------------------------------------

    /// Fallible form of [`Raster::grey`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::Raster`] for zero dimensions or allocation
    /// failure.
    pub fn try_grey(width: u32, height: u32, uchar: bool) -> Result<Raster, ConversionError> {
        if !uchar {
            // libvips' default grey: a single-band float ramp running 0.0
            // at the left edge to 1.0 at the right edge.
            let fmt = PixelFormat::FloatF32(NonZeroU16::new(1).expect("1 is non-zero"));
            let mut out = Raster::zeroed(width, height, fmt)?;
            let w = width as usize;
            let row = w * 4;
            let odata = out.data_mut();
            // Fill row 0, then replicate it: every row of the ramp is equal.
            for x in 0..w {
                let v = if w == 1 {
                    0.0f32
                } else {
                    (x as f64 / (w as f64 - 1.0)) as f32
                };
                odata[x * 4..x * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
            for y in 1..height as usize {
                odata.copy_within(0..row, y * row);
            }
            return Ok(out);
        }
        let mut out = Raster::zeroed(width, height, PixelFormat::Gray8)?;
        let w = width as usize;
        let odata = out.data_mut();
        // Fill row 0, then replicate it: every row of the ramp is equal.
        for (x, px) in odata[..w].iter_mut().enumerate() {
            *px = if w == 1 {
                0
            } else {
                ((255.0 * x as f64) / (w as f64 - 1.0)).round() as u8
            };
        }
        for y in 1..height as usize {
            odata.copy_within(0..w, y * w);
        }
        Ok(out)
    }

    /// Create a horizontal grey ramp (libvips `vips_grey`). With
    /// `uchar: true`, `Gray8` pixels running 0 at the left edge to 255 at
    /// the right edge, every row identical; a 256-wide ramp has
    /// `pixel(x) == x` exactly, the fixture shape the ported `switch` and
    /// relational tests rely on. With `uchar: false`, libvips' default
    /// single-band float ramp running 0.0 to 1.0 in
    /// [`PixelFormat::FloatF32`]. Panicking form of [`Raster::try_grey`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_grey`].
    #[track_caller]
    pub fn grey(width: u32, height: u32, uchar: bool) -> Raster {
        expect_conv("grey", Self::try_grey(width, height, uchar))
    }

    /// Create the 8-bit identity look-up table (libvips `vips_identity`):
    /// a 256x1 `Gray8` image with `pixel(x, 0) == x`.
    pub fn identity() -> Raster {
        let data: Vec<u8> = (0..=255u8).collect();
        Raster::new(256, 1, PixelFormat::Gray8, data).expect("identity LUT dimensions are valid")
    }

    /// Create the 16-bit identity look-up table (libvips `vips_identity`
    /// with `ushort: true`): a 65536x1 `Gray16` image with
    /// `pixel(x, 0) == x`.
    pub fn identity_ushort() -> Raster {
        let mut data = Vec::with_capacity(65536 * 2);
        for i in 0..=65535u16 {
            data.extend_from_slice(&i.to_ne_bytes());
        }
        Raster::new(65536, 1, PixelFormat::Gray16, data).expect("identity LUT dimensions are valid")
    }

    /// Fallible form of [`Raster::switch`].
    ///
    /// # Errors
    ///
    /// [`ConversionError::EmptyInput`] for an empty list,
    /// [`ConversionError::TooManyConditions`] for more than 255,
    /// [`ConversionError::DimensionMismatch`] /
    /// [`ConversionError::ConditionNotMono`] for mis-shaped conditions,
    /// or [`ConversionError::Raster`] on allocation failure.
    pub fn try_switch(conditions: &[&Raster]) -> Result<Raster, ConversionError> {
        let Some(first) = conditions.first() else {
            return Err(ConversionError::EmptyInput { op: "switch" });
        };
        if conditions.len() > 255 {
            return Err(ConversionError::TooManyConditions {
                count: conditions.len(),
            });
        }
        for c in conditions {
            if c.width() != first.width() || c.height() != first.height() {
                return Err(ConversionError::DimensionMismatch {
                    expected_w: first.width(),
                    expected_h: first.height(),
                    got_w: c.width(),
                    got_h: c.height(),
                });
            }
            if c.format().channels() != 1 {
                return Err(ConversionError::ConditionNotMono {
                    bands: c.format().channels(),
                });
            }
        }
        let mut out = Raster::zeroed(first.width(), first.height(), PixelFormat::Gray8)?;
        let odata = out.data_mut();
        let no_match = conditions.len() as u8;
        for (p, px) in odata.iter_mut().enumerate() {
            let mut v = no_match;
            for (i, c) in conditions.iter().enumerate() {
                if read_flat(c.data(), c.format().bytes_per_channel(), p) != 0 {
                    v = i as u8;
                    break;
                }
            }
            *px = v;
        }
        Ok(out)
    }

    /// Build an index image from a list of single-band condition images
    /// (libvips `vips_switch`): each output pixel is the index of the
    /// first condition whose sample is non-zero there, or the number of
    /// conditions when none matches. The output is `Gray8`. Panicking
    /// form of [`Raster::try_switch`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConversionError`]; see [`Raster::try_switch`].
    #[track_caller]
    pub fn switch(conditions: &[&Raster]) -> Raster {
        expect_conv("switch", Self::try_switch(conditions))
    }
}

/// The 256-entry false-colour scale from libvips `falsecolour.c`
/// (`vips_falsecolour_pet`), taken from a PET scanner. Index with the
/// clip-cast 8-bit value of band 0; each entry is `[r, g, b]`.
const FALSECOLOUR_PET: [[u8; 3]; 256] = [
    [12, 0, 25],
    [17, 0, 34],
    [20, 0, 41],
    [22, 0, 45],
    [23, 0, 47],
    [27, 0, 55],
    [12, 0, 25],
    [5, 0, 11],
    [5, 0, 11],
    [5, 0, 11],
    [1, 0, 4],
    [1, 0, 4],
    [6, 0, 13],
    [15, 0, 30],
    [19, 0, 40],
    [23, 0, 48],
    [28, 0, 57],
    [36, 0, 74],
    [42, 0, 84],
    [46, 0, 93],
    [51, 0, 102],
    [59, 0, 118],
    [65, 0, 130],
    [69, 0, 138],
    [72, 0, 146],
    [81, 0, 163],
    [47, 0, 95],
    [12, 0, 28],
    [64, 0, 144],
    [61, 0, 146],
    [55, 0, 140],
    [52, 0, 137],
    [47, 0, 132],
    [43, 0, 128],
    [38, 0, 123],
    [30, 0, 115],
    [26, 0, 111],
    [23, 0, 108],
    [17, 0, 102],
    [9, 0, 94],
    [6, 0, 91],
    [2, 0, 87],
    [0, 0, 88],
    [0, 0, 100],
    [0, 0, 104],
    [0, 0, 108],
    [0, 0, 113],
    [0, 0, 121],
    [0, 0, 125],
    [0, 0, 129],
    [0, 0, 133],
    [0, 0, 141],
    [0, 0, 146],
    [0, 0, 150],
    [0, 0, 155],
    [0, 0, 162],
    [0, 0, 167],
    [0, 0, 173],
    [0, 0, 180],
    [0, 0, 188],
    [0, 0, 193],
    [0, 0, 197],
    [0, 0, 201],
    [0, 0, 209],
    [0, 0, 214],
    [0, 0, 218],
    [0, 0, 222],
    [0, 0, 230],
    [0, 0, 235],
    [0, 0, 239],
    [0, 0, 243],
    [0, 0, 247],
    [0, 4, 251],
    [0, 10, 255],
    [0, 14, 255],
    [0, 18, 255],
    [0, 24, 255],
    [0, 31, 255],
    [0, 36, 255],
    [0, 39, 255],
    [0, 45, 255],
    [0, 53, 255],
    [0, 56, 255],
    [0, 60, 255],
    [0, 66, 255],
    [0, 74, 255],
    [0, 77, 255],
    [0, 81, 255],
    [0, 88, 251],
    [0, 99, 239],
    [0, 104, 234],
    [0, 108, 230],
    [0, 113, 225],
    [0, 120, 218],
    [0, 125, 213],
    [0, 128, 210],
    [0, 133, 205],
    [0, 141, 197],
    [0, 145, 193],
    [0, 150, 188],
    [0, 154, 184],
    [0, 162, 176],
    [0, 167, 172],
    [0, 172, 170],
    [0, 180, 170],
    [0, 188, 170],
    [0, 193, 170],
    [0, 197, 170],
    [0, 201, 170],
    [0, 205, 170],
    [0, 211, 170],
    [0, 218, 170],
    [0, 222, 170],
    [0, 226, 170],
    [0, 232, 170],
    [0, 239, 170],
    [0, 243, 170],
    [0, 247, 170],
    [0, 251, 161],
    [0, 255, 147],
    [0, 255, 139],
    [0, 255, 131],
    [0, 255, 120],
    [0, 255, 105],
    [0, 255, 97],
    [0, 255, 89],
    [0, 255, 78],
    [0, 255, 63],
    [0, 255, 55],
    [0, 255, 47],
    [0, 255, 37],
    [0, 255, 21],
    [0, 255, 13],
    [0, 255, 5],
    [2, 255, 2],
    [13, 255, 13],
    [18, 255, 18],
    [23, 255, 23],
    [27, 255, 27],
    [35, 255, 35],
    [40, 255, 40],
    [43, 255, 43],
    [48, 255, 48],
    [55, 255, 55],
    [60, 255, 60],
    [64, 255, 64],
    [69, 255, 69],
    [72, 255, 72],
    [79, 255, 79],
    [90, 255, 82],
    [106, 255, 74],
    [113, 255, 70],
    [126, 255, 63],
    [140, 255, 56],
    [147, 255, 53],
    [155, 255, 48],
    [168, 255, 42],
    [181, 255, 36],
    [189, 255, 31],
    [197, 255, 27],
    [209, 255, 21],
    [224, 255, 14],
    [231, 255, 10],
    [239, 255, 7],
    [247, 251, 3],
    [255, 243, 0],
    [255, 239, 0],
    [255, 235, 0],
    [255, 230, 0],
    [255, 222, 0],
    [255, 218, 0],
    [255, 214, 0],
    [255, 209, 0],
    [255, 201, 0],
    [255, 197, 0],
    [255, 193, 0],
    [255, 188, 0],
    [255, 180, 0],
    [255, 176, 0],
    [255, 172, 0],
    [255, 167, 0],
    [255, 156, 0],
    [255, 150, 0],
    [255, 146, 0],
    [255, 142, 0],
    [255, 138, 0],
    [255, 131, 0],
    [255, 125, 0],
    [255, 121, 0],
    [255, 117, 0],
    [255, 110, 0],
    [255, 104, 0],
    [255, 100, 0],
    [255, 96, 0],
    [255, 90, 0],
    [255, 83, 0],
    [255, 78, 0],
    [255, 75, 0],
    [255, 71, 0],
    [255, 67, 0],
    [255, 65, 0],
    [255, 63, 0],
    [255, 59, 0],
    [255, 54, 0],
    [255, 52, 0],
    [255, 50, 0],
    [255, 46, 0],
    [255, 41, 0],
    [255, 39, 0],
    [255, 36, 0],
    [255, 32, 0],
    [255, 25, 0],
    [255, 22, 0],
    [255, 20, 0],
    [255, 17, 0],
    [255, 13, 0],
    [255, 10, 0],
    [255, 7, 0],
    [255, 4, 0],
    [255, 0, 0],
    [252, 0, 0],
    [251, 0, 0],
    [249, 0, 0],
    [248, 0, 0],
    [244, 0, 0],
    [242, 0, 0],
    [240, 0, 0],
    [237, 0, 0],
    [234, 0, 0],
    [231, 0, 0],
    [229, 0, 0],
    [228, 0, 0],
    [225, 0, 0],
    [222, 0, 0],
    [221, 0, 0],
    [219, 0, 0],
    [216, 0, 0],
    [213, 0, 0],
    [212, 0, 0],
    [210, 0, 0],
    [207, 0, 0],
    [204, 0, 0],
    [201, 0, 0],
    [199, 0, 0],
    [196, 0, 0],
    [193, 0, 0],
    [192, 0, 0],
    [190, 0, 0],
    [188, 0, 0],
    [184, 0, 0],
    [183, 0, 0],
    [181, 0, 0],
    [179, 0, 0],
    [175, 0, 0],
    [174, 0, 0],
    [174, 0, 0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn gray8(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    fn gray16(w: u32, h: u32, vals: &[u16]) -> Raster {
        let mut data = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        Raster::new(w, h, PixelFormat::Gray16, data).unwrap()
    }

    fn rgb8(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    fn rgb16(w: u32, h: u32, vals: &[u16]) -> Raster {
        let mut data = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        Raster::new(w, h, PixelFormat::Rgb16, data).unwrap()
    }

    // ------------------------------------------------------------------
    // cast
    // ------------------------------------------------------------------

    /**
     * Tests that widening casts preserve sample values numerically.
     * Works by casting a Gray8 image to Gray16 and comparing every
     * getpoint value; the format tag must change, the numbers must not.
     * Input: 2x1 Gray8 [0, 200] -> Gray16 [0, 200].
     */
    #[test]
    fn cast_widens_preserving_values() {
        let im = gray8(2, 1, vec![0, 200]);
        let out = im.cast(PixelFormat::Gray16);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![0.0]);
        assert_eq!(out.getpoint(1, 0), vec![200.0]);
    }

    /**
     * Tests that narrowing casts clip at 255 instead of wrapping,
     * matching the default (non-shifting) vips_cast.
     * Input: 3x1 Gray16 [100, 300, 65535] -> Gray8 [100, 255, 255].
     */
    #[test]
    fn cast_narrows_with_clip() {
        let im = gray16(3, 1, &[100, 300, 65535]);
        let out = im.cast(PixelFormat::Gray8);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.data(), &[100, 255, 255]);
    }

    /**
     * Tests that a same-depth cast is a pixel-identical copy.
     * Input: Gray8 -> Gray8; data buffers equal.
     */
    #[test]
    fn cast_same_format_is_identity() {
        let im = gray8(2, 2, vec![1, 2, 3, 4]);
        let out = im.cast(PixelFormat::Gray8);
        assert_eq!(out.data(), im.data());
        assert_eq!(out.format(), PixelFormat::Gray8);
    }

    /**
     * Tests that multiband intermediates cast across depths.
     * Input: 1x1 Multi8(2) [7, 9] -> Multi16(2) with the same values.
     */
    #[test]
    fn cast_multiband_depth_change() {
        let fmt2x8 = PixelFormat::with_channels(2, 1).unwrap();
        let fmt2x16 = PixelFormat::with_channels(2, 2).unwrap();
        let im = Raster::new(1, 1, fmt2x8, vec![7, 9]).unwrap();
        let out = im.cast(fmt2x16);
        assert_eq!(out.format(), fmt2x16);
        assert_eq!(out.getpoint(0, 0), vec![7.0, 9.0]);
    }

    /**
     * Tests that cast rejects a band-count change with a typed error.
     * Input: Gray8 -> Rgb8 => CastBandCountChange.
     */
    #[test]
    fn try_cast_rejects_band_count_change() {
        let im = gray8(1, 1, vec![0]);
        assert!(matches!(
            im.try_cast(PixelFormat::Rgb8),
            Err(ConversionError::CastBandCountChange { .. })
        ));
    }

    /**
     * Tests that cast carries the metadata block through.
     * Works by setting xres via copy(), casting, and reading it back.
     * Input: xres 42 survives Gray8 -> Gray16.
     */
    #[test]
    fn cast_preserves_metadata() {
        let im = gray8(1, 1, vec![0]).copy().xres(42.0).build();
        let out = im.cast(PixelFormat::Gray16);
        assert_eq!(out.xres(), 42.0);
    }

    /**
     * Tests that casting u8 samples to float stores the exact values,
     * never rescaled: a 200 becomes 200.0, matching the value-preserving
     * widening the unsigned casts already do.
     * Input: 4x1 Gray8 [0, 1, 128, 255] -> FloatF32(1) [0.0, 1.0, 128.0, 255.0].
     */
    #[test]
    fn cast_u8_to_float_is_exact() {
        let im = gray8(4, 1, vec![0, 1, 128, 255]);
        let out = im.cast(PixelFormat::with_channels(1, 4).unwrap());
        assert!(out.format().is_float());
        assert_eq!(out.f32_samples().unwrap(), vec![0.0, 1.0, 128.0, 255.0]);
        assert_eq!(out.getpoint(2, 0), vec![128.0]);
    }

    /**
     * Tests that casting u16 samples to float stores the exact values.
     * Every u16 is exactly representable in f32 (needs 16 bits of
     * mantissa; f32 has 24), so no value may change.
     * Input: 3x1 Gray16 [0, 4096, 65535] -> [0.0, 4096.0, 65535.0].
     */
    #[test]
    fn cast_u16_to_float_is_exact() {
        let im = gray16(3, 1, &[0, 4096, 65535]);
        let out = im.cast(PixelFormat::with_channels(1, 4).unwrap());
        assert_eq!(out.f32_samples().unwrap(), vec![0.0, 4096.0, 65535.0]);
    }

    /**
     * Tests float -> u8 casting: truncate toward zero, clip to 0..=255,
     * and NaN pins to 0 (not to a clip bound). The `0.5` sample is the
     * one that moved when the rounding mode was corrected to match
     * `vips_cast` (issue #561): it used to answer 1.
     * Input: [-1.5, 0.4, 0.5, 254.6, 300.0, NaN] -> [0, 0, 0, 254, 255, 0].
     */
    #[test]
    fn cast_float_to_u8_truncates_and_clips() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im =
            Raster::from_f32_samples(6, 1, f1, &[-1.5, 0.4, 0.5, 254.6, 300.0, f32::NAN]).unwrap();
        let out = im.cast(PixelFormat::Gray8);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.data(), &[0, 0, 0, 254, 255, 0]);
    }

    /**
     * Tests float -> u16 casting: truncate toward zero and clip to
     * 0..=65535. The `0.5` and `65534.6` samples are the ones that moved
     * with the rounding-mode correction (issue #561); they used to answer
     * 1 and 65535.
     * Input: [-3.0, 0.5, 65534.6, 70000.0] -> [0, 0, 65534, 65535].
     */
    #[test]
    fn cast_float_to_u16_truncates_and_clips() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(4, 1, f1, &[-3.0, 0.5, 65534.6, 70000.0]).unwrap();
        let out = im.cast(PixelFormat::Gray16);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(
            out.getpoint(0, 0)
                .into_iter()
                .chain(out.getpoint(1, 0))
                .chain(out.getpoint(2, 0))
                .chain(out.getpoint(3, 0))
                .collect::<Vec<_>>(),
            vec![0.0, 0.0, 65534.0, 65535.0]
        );
    }

    /**
     * Tests the vips 8.18.4 oracle rows for float -> uchar: `vips cast`
     * TRUNCATES the fraction away, it does not round to nearest
     * (`cast.c:566-567`, "Floats are truncated (not rounded)").
     * Works by pinning the four values measured against the binary with
     * `vips csvload p.csv p.v && vips cast p.v pu.v uchar`, none of which
     * sit near a clip bound, so the rounding mode is the only thing they
     * can disagree on (issue #561).
     * Input: [1.7, 2.5, 3.999, 254.6] -> [1, 2, 3, 254], never
     * [2, 3, 4, 255].
     */
    #[test]
    fn cast_float_to_u8_truncates_vips_oracle_rows() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(4, 1, f1, &[1.7, 2.5, 3.999, 254.6]).unwrap();
        let out = im.cast(PixelFormat::Gray8);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(
            out.data(),
            &[1, 2, 3, 254],
            "vips cast truncates float samples; rounding would give [2, 3, 4, 255]"
        );
    }

    /**
     * Tests that the same truncation holds on the wider unsigned target,
     * where a value can be fractional without being anywhere near the
     * clip. Works by casting to Gray16 the `300.9` row measured as `300`
     * by `vips cast p.v ps.v ushort` — the same input that clips to 255 on
     * uchar, so this separates the rounding mode from the clip (issue
     * #561).
     * Input: [300.9, 65534.6, 1.5] -> [300, 65534, 1].
     */
    #[test]
    fn cast_float_to_u16_truncates_in_range() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(3, 1, f1, &[300.9, 65534.6, 1.5]).unwrap();
        let out = im.cast(PixelFormat::Gray16);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(
            (0..3).flat_map(|x| out.getpoint(x, 0)).collect::<Vec<_>>(),
            vec![300.0, 65534.0, 1.0],
            "vips cast truncates on ushort too; rounding would give [301, 65535, 2]"
        );
    }

    /**
     * Tests that the clip bounds did NOT move when the rounding mode did
     * (issue #561): clipping was already correct against vips, so the
     * truncation fix must leave it alone.
     * Works by pinning the below-range and above-range rows measured on
     * the binary for both unsigned targets.
     * Input: uchar [-0.5, -3.7, 300.9] -> [0, 0, 255];
     * ushort [-3.0, 65535.5, 70000.0] -> [0, 65535, 65535].
     */
    #[test]
    fn cast_float_clipping_unchanged_by_truncation() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let u8_in = Raster::from_f32_samples(3, 1, f1, &[-0.5, -3.7, 300.9]).unwrap();
        assert_eq!(
            u8_in.cast(PixelFormat::Gray8).data(),
            &[0, 0, 255],
            "out-of-range uchar samples still clip to the format bounds"
        );

        let u16_in = Raster::from_f32_samples(3, 1, f1, &[-3.0, 65535.5, 70000.0]).unwrap();
        let out = u16_in.cast(PixelFormat::Gray16);
        assert_eq!(
            (0..3).flat_map(|x| out.getpoint(x, 0)).collect::<Vec<_>>(),
            vec![0.0, 65535.0, 65535.0],
            "out-of-range ushort samples still clip to the format bounds"
        );
    }

    /**
     * Tests that the NaN -> 0 pin survived the rounding-mode change
     * (issue #561). NaN already matched vips, and it needs its own branch
     * either way: `f64::trunc` of NaN is NaN and `f64::clamp` passes NaN
     * straight through to the cast, so nothing in the arithmetic puts it
     * at 0 on its own.
     * Works by casting a NaN sample to both unsigned targets. Measured on
     * the binary as `vips math a.v nan.v asin` over `2` (csvload cannot
     * parse the literal `nan`), then `vips cast nan.v out.v uchar` -> 0.
     * Input: [NaN, 1.7] -> uchar [0, 1] and ushort [0, 1].
     */
    #[test]
    fn cast_float_nan_still_pins_to_zero() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(2, 1, f1, &[f32::NAN, 1.7]).unwrap();
        assert_eq!(
            im.cast(PixelFormat::Gray8).data(),
            &[0, 1],
            "NaN pins to 0 rather than falling out of the clamp"
        );
        let wide = im.cast(PixelFormat::Gray16);
        assert_eq!(
            (0..2).flat_map(|x| wide.getpoint(x, 0)).collect::<Vec<_>>(),
            vec![0.0, 1.0],
            "NaN pins to 0 on the wide target too"
        );
    }

    /**
     * Tests the u8 -> f32 -> u8 and u16 -> f32 -> u16 round trips are
     * identities: every in-range integer survives both directions.
     * Input: Gray8 ramp [0..=255 sampled] and Gray16 values round-trip.
     */
    #[test]
    fn cast_float_round_trips_unsigned() {
        let vals8: Vec<u8> = vec![0, 1, 2, 63, 127, 128, 200, 254, 255];
        let im = gray8(vals8.len() as u32, 1, vals8.clone());
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let back = im.cast(f1).cast(PixelFormat::Gray8);
        assert_eq!(back.data(), im.data(), "u8 -> f32 -> u8 must be identity");

        let vals16: [u16; 6] = [0, 1, 255, 256, 32768, 65535];
        let im = gray16(vals16.len() as u32, 1, &vals16);
        let back = im.cast(f1).cast(PixelFormat::Gray16);
        assert_eq!(back.data(), im.data(), "u16 -> f32 -> u16 must be identity");
    }

    /**
     * Tests the exact ported call shape of test_composite_non_separable:
     * an Rgb8 image through add_const().bandjoin_const().cast(RgbaF32)
     * produces a 4-band float raster with the expected sample values.
     * Input: 1x1 Rgb8 [10, 20, 30] +100, join 255 -> RgbaF32
     * [110.0, 120.0, 130.0, 255.0].
     */
    #[test]
    fn cast_rgbaf32_ported_call_site() {
        let colour = rgb8(1, 1, vec![10, 20, 30]);
        let base = colour
            .add_const(100.0)
            .bandjoin_const(255.0)
            .cast(PixelFormat::RgbaF32);
        assert_eq!(base.format(), PixelFormat::RgbaF32);
        assert_eq!(base.format().channels(), 4);
        assert!(base.format().has_alpha());
        assert_eq!(
            base.f32_samples().unwrap(),
            vec![110.0, 120.0, 130.0, 255.0]
        );
        assert_eq!(base.getpoint(0, 0), vec![110.0, 120.0, 130.0, 255.0]);
    }

    /**
     * Tests float -> float casting retags and copies: a manually built
     * FloatF32(4) casts to the canonical RgbaF32 with identical samples.
     * Input: 1x1 FloatF32(4) [0.5, -2.0, 1e6, 0.0] -> RgbaF32, same values.
     */
    #[test]
    fn cast_float_to_float_retags() {
        let f4 = PixelFormat::FloatF32(core::num::NonZeroU16::new(4).unwrap());
        let im = Raster::from_f32_samples(1, 1, f4, &[0.5, -2.0, 1e6, 0.0]).unwrap();
        let out = im.cast(PixelFormat::RgbaF32);
        assert_eq!(out.format(), PixelFormat::RgbaF32);
        assert_eq!(out.f32_samples().unwrap(), vec![0.5, -2.0, 1e6, 0.0]);
    }

    /**
     * Tests that cast to a float format still rejects band-count changes
     * with the same typed error as the unsigned targets.
     * Input: Gray8 -> RgbaF32 => CastBandCountChange.
     */
    #[test]
    fn try_cast_to_float_rejects_band_count_change() {
        let im = gray8(1, 1, vec![0]);
        assert!(matches!(
            im.try_cast(PixelFormat::RgbaF32),
            Err(ConversionError::CastBandCountChange { .. })
        ));
    }

    /**
     * Tests that cast to and from float carries the metadata block.
     * Input: xres 42 survives Gray8 -> FloatF32(1) -> Gray8.
     */
    #[test]
    fn cast_float_preserves_metadata() {
        let im = gray8(1, 1, vec![7]).copy().xres(42.0).build();
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let out = im.cast(f1);
        assert_eq!(out.xres(), 42.0);
        let back = out.cast(PixelFormat::Gray8);
        assert_eq!(back.xres(), 42.0);
        assert_eq!(back.data(), &[7]);
    }

    // ------------------------------------------------------------------
    // copy / metadata
    // ------------------------------------------------------------------

    /**
     * Tests that the copy builder sets every metadata field and leaves
     * the pixels and format untouched.
     * Input: all six setters -> all six getters read back.
     */
    #[test]
    fn copy_builder_sets_all_fields() {
        let im = rgb8(1, 1, vec![10, 20, 30]);
        let out = im
            .copy()
            .interpretation(Interpretation::Lab)
            .xres(42.0)
            .yres(43.0)
            .xoffset(-5)
            .yoffset(7)
            .orientation(6)
            .build();
        assert_eq!(out.interpretation(), Interpretation::Lab);
        assert_eq!(out.xres(), 42.0);
        assert_eq!(out.yres(), 43.0);
        assert_eq!(out.xoffset(), -5);
        assert_eq!(out.yoffset(), 7);
        assert_eq!(out.orientation(), 6);
        assert_eq!(out.data(), im.data());
        assert_eq!(out.format(), im.format());
    }

    /**
     * Tests the interpretation inferred from each PixelFormat when no
     * explicit tag is set, plus the other metadata defaults.
     * Input: fresh rasters of each format -> for_format values.
     */
    #[test]
    fn default_metadata_inferred_from_format() {
        let cases = [
            (PixelFormat::Gray8, Interpretation::Bw),
            (PixelFormat::Gray16, Interpretation::Grey16),
            (PixelFormat::Rgb8, Interpretation::Srgb),
            (PixelFormat::Rgba8, Interpretation::Srgb),
            (PixelFormat::Rgb16, Interpretation::Rgb16),
            (PixelFormat::Rgba16, Interpretation::Rgb16),
            (
                PixelFormat::with_channels(2, 1).unwrap(),
                Interpretation::Multiband,
            ),
            (
                PixelFormat::with_channels(5, 2).unwrap(),
                Interpretation::Multiband,
            ),
        ];
        for (fmt, interp) in cases {
            let im = Raster::zeroed(1, 1, fmt).unwrap();
            assert_eq!(im.interpretation(), interp, "format {fmt:?}");
        }
        let im = Raster::zeroed(1, 1, PixelFormat::Gray8).unwrap();
        assert_eq!(im.xres(), 1.0);
        assert_eq!(im.yres(), 1.0);
        assert_eq!(im.xoffset(), 0);
        assert_eq!(im.yoffset(), 0);
        assert_eq!(im.orientation(), 1);
    }

    /**
     * Tests that copy starts from the source's existing metadata, so a
     * second copy only overrides what it sets.
     * Input: xres 42 then yres 9 -> both survive on the final image.
     */
    #[test]
    fn copy_starts_from_existing_metadata() {
        let im = gray8(1, 1, vec![0]).copy().xres(42.0).build();
        let out = im.copy().yres(9.0).build();
        assert_eq!(out.xres(), 42.0);
        assert_eq!(out.yres(), 9.0);
    }

    /**
     * Tests that an explicit interpretation survives even when it does
     * not match the format's natural reading.
     * Input: Gray8 tagged Lab reads back Lab, not Bw.
     */
    #[test]
    fn explicit_interpretation_overrides_inference() {
        let im = gray8(1, 1, vec![0])
            .copy()
            .interpretation(Interpretation::Lab)
            .build();
        assert_eq!(im.interpretation(), Interpretation::Lab);
    }

    // ------------------------------------------------------------------
    // ifthenelse
    // ------------------------------------------------------------------

    /**
     * Tests that mixed branch depths promote to the common format before
     * the per-pixel select (vips__formatalike), the exact shape the ported
     * test_ifthenelse hits after add_const promotes Rgb8 to Rgb16. The old
     * code read the Rgb8 else branch at the Rgb16 stride, so an else pixel
     * of [2,3,4] came back as [770, 516, 1027] (the little-endian byte
     * pairs [2,3], [4,2], [3,4] of the repeating 8-bit data).
     * Input: cond Gray8 [0, 255], then Rgb16 [1000,2000,3000 / 100,200,300],
     * else Rgb8 [2,3,4 / 9,9,9] -> Rgb16 [2,3,4 / 100,200,300].
     */
    #[test]
    fn ifthenelse_mixed_depth_promotes_else_branch() {
        let cond = gray8(2, 1, vec![0, 255]);
        let then = rgb16(2, 1, &[1000, 2000, 3000, 100, 200, 300]);
        let otherwise = rgb8(2, 1, vec![2, 3, 4, 9, 9, 9]);
        let out = cond.ifthenelse(&then, &otherwise);
        assert_eq!(out.format(), PixelFormat::Rgb16);
        assert_eq!(out.getpoint(0, 0), vec![2.0, 3.0, 4.0]);
        assert_ne!(out.getpoint(0, 0), vec![770.0, 516.0, 1027.0]);
        assert_eq!(out.getpoint(1, 0), vec![100.0, 200.0, 300.0]);
    }

    /**
     * Tests the symmetric promotion: an Rgb8 then branch against an Rgb16
     * else branch. The 8-bit then values must survive numerically into the
     * promoted 16-bit output.
     * Input: cond Gray8 [255, 0], then Rgb8 [2,3,4 / 9,9,9],
     * else Rgb16 [1000,2000,3000 / 100,200,300] -> Rgb16 [2,3,4 / 100,200,300].
     */
    #[test]
    fn ifthenelse_mixed_depth_promotes_then_branch() {
        let cond = gray8(2, 1, vec![255, 0]);
        let then = rgb8(2, 1, vec![2, 3, 4, 9, 9, 9]);
        let otherwise = rgb16(2, 1, &[1000, 2000, 3000, 100, 200, 300]);
        let out = cond.ifthenelse(&then, &otherwise);
        assert_eq!(out.format(), PixelFormat::Rgb16);
        assert_eq!(out.getpoint(0, 0), vec![2.0, 3.0, 4.0]);
        assert_eq!(out.getpoint(1, 0), vec![100.0, 200.0, 300.0]);
    }

    /**
     * Tests that equal-format branches keep their format and exact bytes,
     * the pre-fix behaviour that must not regress.
     * Input: cond Gray8 [0, 255], then Rgb8 [10,20,30 / 40,50,60],
     * else Rgb8 [2,3,4 / 9,9,9] -> Rgb8 [2,3,4 / 40,50,60].
     */
    #[test]
    fn ifthenelse_same_format_unchanged() {
        let cond = gray8(2, 1, vec![0, 255]);
        let then = rgb8(2, 1, vec![10, 20, 30, 40, 50, 60]);
        let otherwise = rgb8(2, 1, vec![2, 3, 4, 9, 9, 9]);
        let out = cond.ifthenelse(&then, &otherwise);
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.data(), &[2, 3, 4, 40, 50, 60]);
    }

    // ------------------------------------------------------------------
    // flips
    // ------------------------------------------------------------------

    /**
     * Tests that fliphor reverses each row.
     * Input: 3x2 Gray8 [1,2,3 / 4,5,6] -> [3,2,1 / 6,5,4].
     */
    #[test]
    fn fliphor_mirrors_columns() {
        let im = gray8(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let out = im.fliphor();
        assert_eq!(out.data(), &[3, 2, 1, 6, 5, 4]);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
    }

    /**
     * Tests that flipver reverses the row order.
     * Input: 3x2 Gray8 [1,2,3 / 4,5,6] -> [4,5,6 / 1,2,3].
     */
    #[test]
    fn flipver_mirrors_rows() {
        let im = gray8(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let out = im.flipver();
        assert_eq!(out.data(), &[4, 5, 6, 1, 2, 3]);
    }

    /**
     * Tests the ported test_flip chain: fliphor.flipver.fliphor.flipver
     * is the identity.
     * Input: 3x2 asymmetric Gray8 -> identical buffer.
     */
    #[test]
    fn double_flip_is_identity() {
        let im = gray8(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let out = im.fliphor().flipver().fliphor().flipver();
        assert_eq!(out.data(), im.data());
    }

    /**
     * Tests that flips move whole 16-bit samples, not bytes.
     * Input: 2x1 Gray16 [300, 40000] -> fliphor -> [40000, 300].
     */
    #[test]
    fn flip_16bit_samples() {
        let im = gray16(2, 1, &[300, 40000]);
        let out = im.fliphor();
        assert_eq!(out.getpoint(0, 0), vec![40000.0]);
        assert_eq!(out.getpoint(1, 0), vec![300.0]);
    }

    /**
     * Tests that flips carry metadata (an explicit interpretation).
     * Input: Gray8 tagged Lab -> fliphor/flipver keep the tag.
     */
    #[test]
    fn flips_preserve_metadata() {
        let im = gray8(2, 1, vec![1, 2])
            .copy()
            .interpretation(Interpretation::Lab)
            .build();
        assert_eq!(im.fliphor().interpretation(), Interpretation::Lab);
        assert_eq!(im.flipver().interpretation(), Interpretation::Lab);
    }

    // ------------------------------------------------------------------
    // rot
    // ------------------------------------------------------------------

    /**
     * Tests that rot(D0) is a pixel-identical copy.
     */
    #[test]
    fn rot_d0_is_identity() {
        let im = gray8(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let out = im.rot(Angle::D0);
        assert_eq!(out.data(), im.data());
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 3);
    }

    /**
     * Tests the clockwise quarter turn against a hand-computed buffer:
     * the left column of the input becomes the top row.
     * Input: 2x3 [1,2 / 3,4 / 5,6] -> 3x2 [5,3,1 / 6,4,2].
     */
    #[test]
    fn rot_d90_maps_pixels() {
        let im = gray8(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let out = im.rot(Angle::D90);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        assert_eq!(out.data(), &[5, 3, 1, 6, 4, 2]);
    }

    /**
     * Tests that D180 equals the two flips composed.
     */
    #[test]
    fn rot_d180_equals_double_flip() {
        let im = gray8(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(im.rot(Angle::D180).data(), im.fliphor().flipver().data());
    }

    /**
     * Tests the ported test_rot roundtrip: rot(D90).rot(D270) is the
     * identity, and the corner pixel maps as the ported test asserts
     * (input (w-1, h-1) lands at (0, h-1) after D90).
     */
    #[test]
    fn rot_d90_d270_roundtrip_and_corner() {
        let mut data = vec![0u8; 25];
        for (i, px) in data.iter_mut().enumerate() {
            *px = i as u8;
        }
        let im = gray8(5, 5, data);
        let round = im.rot(Angle::D90).rot(Angle::D270);
        assert_eq!(round.data(), im.data());

        let turned = im.rot(Angle::D90);
        assert_eq!(turned.getpoint(0, 4), im.getpoint(4, 4));
    }

    /**
     * Tests that quarter turns swap the dimensions on non-square input.
     */
    #[test]
    fn rot_swaps_dimensions_for_quarter_turns() {
        let im = gray8(4, 2, vec![0; 8]);
        for angle in [Angle::D90, Angle::D270] {
            let out = im.rot(angle);
            assert_eq!((out.width(), out.height()), (2, 4), "{angle:?}");
        }
        let out = im.rot(Angle::D180);
        assert_eq!((out.width(), out.height()), (4, 2));
    }

    /**
     * Tests that rot carries metadata through.
     */
    #[test]
    fn rot_preserves_metadata() {
        let im = gray8(2, 1, vec![1, 2]).copy().xres(3.5).build();
        assert_eq!(im.rot(Angle::D90).xres(), 3.5);
    }

    // ------------------------------------------------------------------
    // autorot
    // ------------------------------------------------------------------

    /**
     * Tests that autorot with the default (upright) tag copies the
     * pixels and keeps orientation 1.
     */
    #[test]
    fn autorot_upright_is_noop() {
        let im = gray8(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let out = im.autorot();
        assert_eq!(out.data(), im.data());
        assert_eq!((out.width(), out.height()), (2, 3));
        assert_eq!(out.orientation(), 1);
    }

    /**
     * Tests every EXIF orientation against its equivalent op chain and
     * that the tag is cleared afterwards. Orientation 5 (transpose) is
     * flipver-then-D90 and 7 (transverse) is fliphor-then-D90.
     * Input: 2x3 asymmetric Gray8, orientations 2..=8.
     */
    #[test]
    fn autorot_matches_equivalent_ops() {
        let im = gray8(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let cases: [(u8, Raster); 7] = [
            (2, im.fliphor()),
            (3, im.rot(Angle::D180)),
            (4, im.flipver()),
            (5, im.flipver().rot(Angle::D90)),
            (6, im.rot(Angle::D90)),
            (7, im.fliphor().rot(Angle::D90)),
            (8, im.rot(Angle::D270)),
        ];
        for (tag, expected) in cases {
            let out = im.copy().orientation(tag).build().autorot();
            assert_eq!(out.data(), expected.data(), "orientation {tag}");
            assert_eq!(
                (out.width(), out.height()),
                (expected.width(), expected.height()),
                "orientation {tag}"
            );
            assert_eq!(out.orientation(), 1, "orientation {tag}");
        }
    }

    /**
     * Tests that a malformed tag (0 or 9) is treated as upright and
     * still normalised to 1.
     */
    #[test]
    fn autorot_malformed_tag_treated_as_upright() {
        let im = gray8(2, 1, vec![9, 4]);
        for tag in [0u8, 9, 255] {
            let out = im.copy().orientation(tag).build().autorot();
            assert_eq!(out.data(), im.data(), "tag {tag}");
            assert_eq!(out.orientation(), 1, "tag {tag}");
        }
    }

    /**
     * Tests that autorot keeps the rest of the metadata block.
     */
    #[test]
    fn autorot_preserves_other_metadata() {
        let im = gray8(2, 1, vec![1, 2])
            .copy()
            .xres(7.0)
            .orientation(6)
            .build();
        let out = im.autorot();
        assert_eq!(out.xres(), 7.0);
        assert_eq!(out.orientation(), 1);
    }

    // ------------------------------------------------------------------
    // wrap
    // ------------------------------------------------------------------

    /**
     * Tests the even-dimension quadrant swap: the centre moves to the
     * origin and the origin to the centre.
     * Input: 4x4 with value y*4+x; out(0,0)=in(2,2)=10, out(2,2)=in(0,0)=0.
     */
    #[test]
    fn wrap_swaps_quadrants_even() {
        let mut data = vec![0u8; 16];
        for (i, px) in data.iter_mut().enumerate() {
            *px = i as u8;
        }
        let im = gray8(4, 4, data);
        let out = im.wrap();
        assert_eq!((out.width(), out.height()), (4, 4));
        assert_eq!(out.getpoint(0, 0), im.getpoint(2, 2));
        assert_eq!(out.getpoint(2, 2), im.getpoint(0, 0));
        assert_eq!(out.getpoint(1, 3), im.getpoint(3, 1));
    }

    /**
     * Tests the odd-dimension convention from vips_wrap: the pixel at
     * (w/2, h/2) moves to the origin.
     * Input: 3x3, out(0,0) = in(1,1).
     */
    #[test]
    fn wrap_centres_origin_odd() {
        let mut data = vec![0u8; 9];
        for (i, px) in data.iter_mut().enumerate() {
            *px = i as u8;
        }
        let im = gray8(3, 3, data);
        let out = im.wrap();
        assert_eq!(out.getpoint(0, 0), im.getpoint(1, 1));
    }

    /**
     * Tests that wrapping twice on even dimensions is the identity.
     */
    #[test]
    fn wrap_double_is_identity_even() {
        let mut data = vec![0u8; 24];
        for (i, px) in data.iter_mut().enumerate() {
            *px = i as u8;
        }
        let im = gray8(6, 4, data);
        assert_eq!(im.wrap().wrap().data(), im.data());
    }

    /**
     * Tests that wrap moves whole multi-band pixels.
     * Input: 2x2 Rgb8; out(0,0) = in(1,1) as an rgb triple.
     */
    #[test]
    fn wrap_moves_whole_pixels() {
        let im = rgb8(2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let out = im.wrap();
        assert_eq!(out.getpoint(0, 0), vec![10.0, 11.0, 12.0]);
        assert_eq!(out.getpoint(1, 1), vec![1.0, 2.0, 3.0]);
    }

    // ------------------------------------------------------------------
    // gamma
    // ------------------------------------------------------------------

    /**
     * Tests the default curve against exact vips 8.18.4 `vips_gamma` output.
     * vips TRUNCATES the scaled power curve (does not round to nearest), so
     * e.g. input 100 -> 26 even though the continuous value is 26.97. Pinned
     * values captured from the oracle over an identity ramp:
     *   `vips identity id.v; vips gamma id.v g.v; vips getpoint g.v <i> 0`.
     */
    #[test]
    fn gamma_default_matches_libvips_formula() {
        // (input, exact vips 8.18.4 uchar output)
        let cases = [
            (0u8, 0u8),
            (1, 0),
            (16, 0),
            (19, 0), // continuous ~0.500, vips truncates to 0
            (50, 5),
            (100, 26), // continuous 26.97 -> truncated 26 (round would give 27)
            (115, 37),
            (127, 47),
            (128, 48),
            (200, 142),
            (254, 252),
            (255, 255),
        ];
        let vals: Vec<u8> = cases.iter().map(|&(i, _)| i).collect();
        let im = gray8(vals.len() as u32, 1, vals.clone());
        let out = im.gamma(None);
        for (x, &(v, expected)) in cases.iter().enumerate() {
            let got = out.getpoint(x as u32, 0)[0];
            assert_eq!(got, expected as f64, "input v={v}");
        }
    }

    /**
     * Tests a custom exponent: exponent 0.5 applies power 2.
     * Input: 16 -> round(16^2 / (255^2/255)) = round(256/255) = 1.
     */
    #[test]
    fn gamma_custom_exponent() {
        let im = gray8(2, 1, vec![16, 255]);
        let out = im.gamma(Some(0.5));
        assert_eq!(out.data(), &[1, 255]);
    }

    /**
     * Pins custom-exponent parity at the upper-domain endpoints on both
     * depths, at powers 2.1 and 2.3. These are the samples where the f32
     * multiply ordering decides the last LSB (the format maximum does not
     * always map back to itself: at power 2.1 uchar 255 -> 254, at power 2.3
     * ushort 65535 -> 65534). Values captured from vips 8.18.4:
     *   `vips gamma id.v g.v --exponent <1/power>; vips getpoint g.v <i> 0`
     * so any future drift in the pow/scale arithmetic is caught, not left to
     * rest on empirical f32/f64 coincidence.
     */
    #[test]
    fn gamma_custom_exponent_endpoints_match_libvips() {
        // exponent = 1/power, so power 2.1 -> exp 1/2.1, power 2.3 -> 1/2.3.
        let e21 = 1.0 / 2.1;
        let e23 = 1.0 / 2.3;

        // uchar endpoints. (input, exact vips uchar output)
        let u21 = [(253u8, 250u8), (254, 252), (255, 254)];
        let u23 = [(253u8, 250u8), (254, 252), (255, 255)];
        for (exp, cases) in [(e21, u21), (e23, u23)] {
            let vals: Vec<u8> = cases.iter().map(|&(i, _)| i).collect();
            let im = gray8(vals.len() as u32, 1, vals.clone());
            let out = im.gamma(Some(exp));
            for (x, &(v, expected)) in cases.iter().enumerate() {
                assert_eq!(
                    out.getpoint(x as u32, 0)[0],
                    expected as f64,
                    "uchar exp={exp} v={v}"
                );
            }
        }

        // ushort endpoints. (input, exact vips ushort output)
        let s21 = [(65533u16, 65530.0f64), (65534, 65532.0), (65535, 65535.0)];
        let s23 = [(65533u16, 65530.0f64), (65534, 65532.0), (65535, 65534.0)];
        for (exp, cases) in [(e21, s21), (e23, s23)] {
            let vals: Vec<u16> = cases.iter().map(|&(i, _)| i).collect();
            let im = gray16(vals.len() as u32, 1, &vals);
            let out = im.gamma(Some(exp));
            for (x, &(v, expected)) in cases.iter().enumerate() {
                assert_eq!(
                    out.getpoint(x as u32, 0)[0],
                    expected,
                    "ushort exp={exp} v={v}"
                );
            }
        }
    }

    /**
     * Tests that the endpoints are fixed on both depths: 0 maps to 0
     * and the format maximum maps to itself.
     */
    #[test]
    fn gamma_endpoints_fixed() {
        let im8 = gray8(2, 1, vec![0, 255]);
        assert_eq!(im8.gamma(None).data(), &[0, 255]);
        let im16 = gray16(2, 1, &[0, 65535]);
        let out = im16.gamma(None);
        assert_eq!(out.getpoint(0, 0), vec![0.0]);
        assert_eq!(out.getpoint(1, 0), vec![65535.0]);
    }

    /**
     * Tests the 16-bit curve against exact vips 8.18.4 output. vips
     * truncates the scaled power curve, and the intermediate is computed in
     * single-precision float, so parity is exact only when both are matched.
     * Values captured from the oracle over a ushort identity ramp:
     *   `vips identity id16.v --ushort; vips gamma id16.v g16.v`.
     * 30000 -> 10046; 21755 -> 4646; 23843 -> 5789 (each a truncation edge
     * the old round-to-nearest path missed by 1 LSB).
     */
    #[test]
    fn gamma_16bit_path() {
        let cases = [(30000u16, 10046.0f64), (21755, 4646.0), (23843, 5789.0)];
        let vals: Vec<u16> = cases.iter().map(|&(i, _)| i).collect();
        let im = gray16(vals.len() as u32, 1, &vals);
        let out = im.gamma(None);
        for (x, &(v, expected)) in cases.iter().enumerate() {
            assert_eq!(out.getpoint(x as u32, 0)[0], expected, "input v={v}");
        }
    }

    /**
     * Tests that gamma applies per channel on colour images.
     */
    #[test]
    fn gamma_multiband_per_channel() {
        let im = rgb8(1, 1, vec![10, 100, 255]);
        let out = im.gamma(None);
        // Exact vips 8.18.4 uchar outputs: 10 -> 0, 100 -> 26, 255 -> 255.
        let px = out.getpoint(0, 0);
        assert_eq!(px, vec![0.0, 26.0, 255.0]);
    }

    /**
     * Tests that non-finite and non-positive exponents are rejected.
     */
    #[test]
    fn try_gamma_rejects_bad_exponent() {
        let im = gray8(1, 1, vec![0]);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    im.try_gamma(Some(bad)),
                    Err(ConversionError::InvalidGammaExponent { .. })
                ),
                "exponent {bad}"
            );
        }
    }

    // ------------------------------------------------------------------
    // falsecolour
    // ------------------------------------------------------------------

    /**
     * Tests known entries of the PET scale on a Gray8 input.
     * Input: [0, 3, 255] -> LUT rows [12,0,25], [22,0,45], [174,0,0].
     */
    #[test]
    fn falsecolour_maps_gray_through_pet_lut() {
        let im = gray8(3, 1, vec![0, 3, 255]);
        let out = im.falsecolour();
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.getpoint(0, 0), vec![12.0, 0.0, 25.0]);
        assert_eq!(out.getpoint(1, 0), vec![22.0, 0.0, 45.0]);
        assert_eq!(out.getpoint(2, 0), vec![174.0, 0.0, 0.0]);
    }

    /**
     * Tests that only band 0 drives the mapping on colour input,
     * matching falsecolour.c's extract_band(0).
     * Input: rgb pixel [3, 200, 100] -> LUT[3] = [22,0,45].
     */
    #[test]
    fn falsecolour_uses_band_zero_only() {
        let im = rgb8(1, 1, vec![3, 200, 100]);
        let out = im.falsecolour();
        assert_eq!(out.getpoint(0, 0), vec![22.0, 0.0, 45.0]);
    }

    /**
     * Tests the clip-cast from 16-bit: values above 255 index the last
     * LUT row, exactly like vips_cast to uchar before maplut.
     * Input: Gray16 [300, 65535] -> both map to LUT[255] = [174,0,0].
     */
    #[test]
    fn falsecolour_clips_16bit_to_uchar() {
        let im = gray16(2, 1, &[300, 65535]);
        let out = im.falsecolour();
        assert_eq!(out.getpoint(0, 0), vec![174.0, 0.0, 0.0]);
        assert_eq!(out.getpoint(1, 0), vec![174.0, 0.0, 0.0]);
    }

    /**
     * Tests dimensions, band count, and the sRGB interpretation stamp.
     */
    #[test]
    fn falsecolour_output_shape_and_interpretation() {
        let im = gray8(4, 3, vec![0; 12]);
        let out = im.falsecolour();
        assert_eq!((out.width(), out.height()), (4, 3));
        assert_eq!(out.format().channels(), 3);
        assert_eq!(out.interpretation(), Interpretation::Srgb);
    }

    // ------------------------------------------------------------------
    // addalpha
    // ------------------------------------------------------------------

    /**
     * Tests the ported test_addalpha shape: Rgb8 gains an opaque 255
     * alpha band and becomes Rgba8, colour untouched.
     */
    #[test]
    fn addalpha_rgb8() {
        let im = rgb8(1, 1, vec![10, 20, 30]);
        let out = im.addalpha();
        assert_eq!(out.format(), PixelFormat::Rgba8);
        assert_eq!(out.getpoint(0, 0), vec![10.0, 20.0, 30.0, 255.0]);
    }

    /**
     * Tests the 16-bit maximum: Rgb16 gains a 65535 alpha band.
     */
    #[test]
    fn addalpha_rgb16() {
        let mut data = Vec::new();
        for v in [1000u16, 2000, 3000] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let im = Raster::new(1, 1, PixelFormat::Rgb16, data).unwrap();
        let out = im.addalpha();
        assert_eq!(out.format(), PixelFormat::Rgba16);
        assert_eq!(out.getpoint(0, 0), vec![1000.0, 2000.0, 3000.0, 65535.0]);
    }

    /**
     * Tests that a mono input becomes a two-band multiband intermediate.
     */
    #[test]
    fn addalpha_gray8_gives_two_bands() {
        let im = gray8(1, 1, vec![9]);
        let out = im.addalpha();
        assert_eq!(out.format(), PixelFormat::with_channels(2, 1).unwrap());
        assert_eq!(out.getpoint(0, 0), vec![9.0, 255.0]);
    }

    /**
     * Tests that addalpha carries metadata through.
     */
    #[test]
    fn addalpha_preserves_metadata() {
        let im = rgb8(1, 1, vec![0, 0, 0]).copy().xres(5.0).build();
        assert_eq!(im.addalpha().xres(), 5.0);
    }

    // ------------------------------------------------------------------
    // arrayjoin
    // ------------------------------------------------------------------

    /**
     * Tests the default layout: one row, cells in call order.
     * Input: two 2x1 images -> 4x1; pixels verify placement.
     */
    #[test]
    fn arrayjoin_default_single_row() {
        let a = gray8(2, 1, vec![1, 2]);
        let b = gray8(2, 1, vec![3, 4]);
        let out = Raster::arrayjoin(&[&a, &b], None, None);
        assert_eq!((out.width(), out.height()), (4, 1));
        assert_eq!(out.data(), &[1, 2, 3, 4]);
    }

    /**
     * Tests across=1: a vertical stack.
     */
    #[test]
    fn arrayjoin_across_one_stacks_vertically() {
        let a = gray8(2, 1, vec![1, 2]);
        let b = gray8(2, 1, vec![3, 4]);
        let out = Raster::arrayjoin(&[&a, &b], Some(1), None);
        assert_eq!((out.width(), out.height()), (2, 2));
        assert_eq!(out.data(), &[1, 2, 3, 4]);
    }

    /**
     * Tests a ragged grid: three images across 2 leaves the fourth cell
     * black.
     * Input: three 1x1 images [5],[6],[7] -> 2x2 [5,6 / 7,0].
     */
    #[test]
    fn arrayjoin_grid_with_padding_cells() {
        let a = gray8(1, 1, vec![5]);
        let b = gray8(1, 1, vec![6]);
        let c = gray8(1, 1, vec![7]);
        let out = Raster::arrayjoin(&[&a, &b, &c], Some(2), None);
        assert_eq!((out.width(), out.height()), (2, 2));
        assert_eq!(out.data(), &[5, 6, 7, 0]);
    }

    /**
     * Tests shim spacing: gaps between cells are black and the output
     * dimensions include (n-1) shims per axis.
     * Input: two 1x1 images, shim 2 -> 4x1 [1,0,0,2].
     */
    #[test]
    fn arrayjoin_shim_spacing() {
        let a = gray8(1, 1, vec![1]);
        let b = gray8(1, 1, vec![2]);
        let out = Raster::arrayjoin(&[&a, &b], None, Some(2));
        assert_eq!((out.width(), out.height()), (4, 1));
        assert_eq!(out.data(), &[1, 0, 0, 2]);
    }

    /**
     * Tests band and depth alignment: a Gray8 image joined with an
     * Rgb16 image produces Rgb16, the mono band replicated and values
     * kept numerically.
     */
    #[test]
    fn arrayjoin_band_and_depth_promotion() {
        let mono = gray8(1, 1, vec![10]);
        let mut data = Vec::new();
        for v in [1000u16, 2000, 3000] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let colour = Raster::new(1, 1, PixelFormat::Rgb16, data).unwrap();
        let out = Raster::arrayjoin(&[&mono, &colour], None, None);
        assert_eq!(out.format(), PixelFormat::Rgb16);
        assert_eq!(out.getpoint(0, 0), vec![10.0, 10.0, 10.0]);
        assert_eq!(out.getpoint(1, 0), vec![1000.0, 2000.0, 3000.0]);
    }

    /**
     * Tests mixed sizes: cells take the largest input's size, smaller
     * images sit top-left, and the remainder is black.
     * Input: 1x1 [9] and 2x2 [1,2,3,4] -> 4x2.
     */
    #[test]
    fn arrayjoin_mixed_sizes_top_left_aligned() {
        let small = gray8(1, 1, vec![9]);
        let big = gray8(2, 2, vec![1, 2, 3, 4]);
        let out = Raster::arrayjoin(&[&small, &big], None, None);
        assert_eq!((out.width(), out.height()), (4, 2));
        assert_eq!(out.data(), &[9, 0, 1, 2, 0, 0, 3, 4]);
    }

    /**
     * Tests the typed errors: an empty list and a 2-vs-3 band mismatch
     * (neither is 1, so bandalike rejects it).
     */
    #[test]
    fn try_arrayjoin_errors() {
        assert!(matches!(
            Raster::try_arrayjoin(&[], None, None),
            Err(ConversionError::EmptyInput { .. })
        ));
        let two = Raster::zeroed(1, 1, PixelFormat::with_channels(2, 1).unwrap()).unwrap();
        let three = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            Raster::try_arrayjoin(&[&two, &three], None, None),
            Err(ConversionError::BandCountMismatch { .. })
        ));
    }

    /**
     * Tests that `across` is NOT clamped to the image count: a value larger
     * than the list lays out that many cells wide and leaves the trailing
     * ones background, which is what `arrayjoin.c:255-268` does at
     * `fe420cf3a` (`down` is `ROUND_UP(n, across) / across` and the output
     * width is `hspacing * across + shim * (across - 1)`, neither of them
     * capped at `n`). libviprs used to clamp into `1..=n`, so every
     * `across > n` collapsed to one full row.
     * Works by sweeping the whole `across` range vips was measured over on
     * the two inputs whose sizes differ, so the cell-size rule (the cell is
     * the largest input on each axis, here 3x3 out of a 3x2 and a 2x3) is
     * exercised at the same time as the layout.
     * Input: 3x2 and 2x3 gray8. Measured with vips 8.18.4 as
     * `vips black a.v 3 2; vips black b.v 2 3;
     * vips arrayjoin "a.v b.v" o.v --across N`:
     * 1 -> 3x6, 2 -> 6x3, 3 -> 9x3, 4 -> 12x3, 5 -> 15x3, 7 -> 21x3,
     * 10 -> 30x3.
     */
    #[test]
    fn arrayjoin_across_is_not_clamped_to_the_image_count() {
        let a = gray8(3, 2, vec![0; 6]);
        let b = gray8(2, 3, vec![0; 6]);
        for (across, want) in [
            (1u32, (3u32, 6u32)),
            (2, (6, 3)),
            (3, (9, 3)),
            (4, (12, 3)),
            (5, (15, 3)),
            (7, (21, 3)),
            (10, (30, 3)),
        ] {
            let out = Raster::arrayjoin(&[&a, &b], Some(across), None);
            assert_eq!(
                (out.width(), out.height()),
                want,
                "across = {across} should give {want:?}"
            );
        }
    }

    /**
     * Tests that the trailing cells an over-wide `across` opens up are
     * background rather than a wrapped repeat of the inputs, the pixel
     * complement of the geometry sweep above.
     * Works by giving the two 1x1 inputs distinct values so a wrap would be
     * visible, then reading every cell of the row back.
     * Input: 1x1 [5] and 1x1 [9], across 3. Measured as
     * `vips arrayjoin "p1.v p2.v" t3.v --across 3` -> 3x1 reading 5, 9, 0.
     */
    #[test]
    fn arrayjoin_across_past_the_end_leaves_background_cells() {
        let a = gray8(1, 1, vec![5]);
        let b = gray8(1, 1, vec![9]);
        let out = Raster::arrayjoin(&[&a, &b], Some(3), None);
        assert_eq!((out.width(), out.height()), (3, 1));
        assert_eq!(out.data(), &[5, 9, 0]);
    }

    /**
     * Tests that `shim` still spaces the empty trailing cells, so the shim
     * count follows `across` and not the image count. `arrayjoin.c:259-260`
     * sizes the output as `hspacing * across + shim * (across - 1)`, which
     * is 3 * 4 + 2 * 3 = 18 here; the old clamp would have made it 6 + 2.
     * Input: the 3x2 / 2x3 pair, across 4, shim 2. Measured as
     * `vips arrayjoin "a.v b.v" s4.v --across 4 --shim 2` -> 18x3.
     */
    #[test]
    fn arrayjoin_shim_spaces_the_trailing_cells_too() {
        let a = gray8(3, 2, vec![0; 6]);
        let b = gray8(2, 3, vec![0; 6]);
        let out = Raster::arrayjoin(&[&a, &b], Some(4), Some(2));
        assert_eq!((out.width(), out.height()), (18, 3));
    }

    /**
     * Tests the libvips bound on `arrayjoin`'s `across`, the
     * `VIPS_ARG_INT(class, "across", 4, ..., 1, 1000000, 1)` range
     * `arrayjoin.c:400-406` declares at `fe420cf3a`. Both ends are refused
     * by GObject before the operation is built, measured on the 3x2 / 2x3
     * pair: `--across 0` and `--across 1000001` each fail with
     * `value "N" of type 'gint' is invalid or out of range for property
     * 'across'`. libviprs used to clamp both silently, 0 into a vertical
     * stack and anything large into one row.
     * Works by asserting the typed variant with its field values, then
     * checking the default is still accepted for a list longer than the
     * range would allow if it were checked, the way vips assigns
     * `join->across = n` past its own property check.
     * Input: two 1x1 images with across 0, 1000001, and u32::MAX.
     */
    #[test]
    fn try_arrayjoin_across_outside_the_vips_property_range_is_rejected() {
        let a = gray8(1, 1, vec![1]);
        let b = gray8(1, 1, vec![2]);
        assert!(matches!(
            Raster::try_arrayjoin(&[&a, &b], Some(0), None),
            Err(ConversionError::AcrossOutOfRange {
                across: 0,
                min: 1,
                max: 1_000_000
            })
        ));
        assert!(matches!(
            Raster::try_arrayjoin(&[&a, &b], Some(1_000_001), None),
            Err(ConversionError::AcrossOutOfRange {
                across: 1_000_001,
                ..
            })
        ));
        assert!(matches!(
            Raster::try_arrayjoin(&[&a, &b], Some(u32::MAX), None),
            Err(ConversionError::AcrossOutOfRange { .. })
        ));
        // The bound applies to an explicit value only; the default is still
        // whatever the list length is.
        let row = Raster::arrayjoin(&[&a, &b], None, None);
        assert_eq!((row.width(), row.height()), (2, 1));
    }

    /**
     * Tests that the grid carries the first image's metadata.
     */
    #[test]
    fn arrayjoin_meta_from_first() {
        let a = gray8(1, 1, vec![1]).copy().xres(11.0).build();
        let b = gray8(1, 1, vec![2]);
        assert_eq!(Raster::arrayjoin(&[&a, &b], None, None).xres(), 11.0);
    }

    /**
     * Tests that a band-promoting grid does not keep the first image's
     * interpretation, which no longer describes the band count.
     * `vips arrayjoin "gbw.v csrgb.v" out.v` reports `6x3 uchar, 3 bands,
     * srgb`; keeping `b-w` here matters because `space_bands(Bw) == 1`, so
     * a later colourspace conversion reads two of the three bands as
     * passthrough extras.
     * Works by tagging the mono input explicitly, the way a decoder tags
     * one, then reading the tag back off the grid.
     * Input: Gray8 1x1 tagged `Bw` + Rgb8 1x1 -> Rgb8 grid tagged `Srgb`.
     */
    #[test]
    fn arrayjoin_band_promotion_drops_the_first_images_tag() {
        let mono = gray8(1, 1, vec![1])
            .copy()
            .interpretation(Interpretation::Bw)
            .build();
        let colour = rgb8(1, 1, vec![10, 20, 30]);
        let out = Raster::arrayjoin(&[&mono, &colour], None, None);
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.interpretation(), Interpretation::Srgb);
    }

    /**
     * Tests that a float input is a typed error rather than a panic out of
     * a fallible method. `arrayjoin`'s sample copy is `read_flat` /
     * `write_flat`, which are unsigned-only, and a float raster is ordinary
     * input: every `colourspace` result for Lab, Lch, OkLab, OkLCh, XYZ,
     * scRGB and Yxy is float. Real vips handles it (`vips arrayjoin
     * "af.v a.v" out.v` gives `6x2 float`), so this is a libviprs
     * limitation reported honestly rather than parity.
     * Works by putting the float `grey` ramp in each list position.
     * Input: FloatF32 2x2 ramp beside a Gray8 1x1.
     */
    #[test]
    fn try_arrayjoin_float_input_is_a_typed_error_not_a_panic() {
        let ramp = Raster::grey(2, 2, false);
        assert!(ramp.format().is_float());
        let plain = gray8(1, 1, vec![1]);
        for list in [[&ramp, &plain], [&plain, &ramp]] {
            assert!(matches!(
                Raster::try_arrayjoin(&list, None, None),
                Err(ConversionError::FloatFormatUnsupported { op: "arrayjoin" })
            ));
        }
    }

    /**
     * Tests the libvips bound on `arrayjoin`'s `shim`, the same
     * `VIPS_ARG_INT(class, "shim", 5, ..., 0, 1000000, 0)` range `join`
     * declares: `vips arrayjoin "a.v c.v" out.v --shim 1000001` fails with
     * the same GObject CRITICAL. Both sides of the bound are pinned, the
     * accepting one against the measured `vips arrayjoin "t1.v t1.v" out.v
     * --shim 1000000` result of `1000002x1`.
     * Input: two 1x1 images, shim 1000001 then 1000000.
     */
    #[test]
    fn try_arrayjoin_shim_above_the_vips_maximum_is_rejected() {
        let a = gray8(1, 1, vec![1]);
        let b = gray8(1, 1, vec![2]);
        assert!(matches!(
            Raster::try_arrayjoin(&[&a, &b], None, Some(1_000_001)),
            Err(ConversionError::ShimTooLarge {
                shim: 1_000_001,
                max: 1_000_000
            })
        ));
        assert!(matches!(
            Raster::try_arrayjoin(&[&a, &b], None, Some(u32::MAX)),
            Err(ConversionError::ShimTooLarge { .. })
        ));
        let out = Raster::arrayjoin(&[&a, &b], None, Some(1_000_000));
        assert_eq!((out.width(), out.height()), (1_000_002, 1));
    }

    /**
     * Tests that a depth-only promotion keeps the tag, the complement of
     * the band-promotion case. `vips arrayjoin "gbw.v g16.v" out.v` reports
     * `ushort, 1 band, b-w`, so the re-stamp must key off the band count
     * and nothing else.
     * Input: Gray8 1x1 tagged `Bw` + Gray16 1x1 -> Gray16 grid still `Bw`.
     */
    #[test]
    fn arrayjoin_depth_promotion_keeps_the_first_images_tag() {
        let mono = gray8(1, 1, vec![1])
            .copy()
            .interpretation(Interpretation::Bw)
            .build();
        let wide = gray16(1, 1, &[300]);
        let out = Raster::arrayjoin(&[&mono, &wide], None, None);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.interpretation(), Interpretation::Bw);
    }
    // ------------------------------------------------------------------
    // join
    // ------------------------------------------------------------------

    /// The oracle fixture pair: `a` is 3x2, `c` is 2x3, both Gray8.
    fn join_a() -> Raster {
        gray8(3, 2, vec![1, 2, 3, 4, 5, 6])
    }

    fn join_c() -> Raster {
        gray8(2, 3, vec![10, 20, 30, 40, 50, 60])
    }

    /**
     * Tests the default horizontal join: the second image sits to the
     * right of the first and the result is cropped back to the shorter
     * height, exactly as `vips join a c out horizontal`.
     * Works by joining the 3x2 / 2x3 oracle pair with align low and
     * expand off, then comparing the whole buffer.
     * Input: a(3x2) + c(2x3) -> 5x2, height = min(2, 3).
     */
    #[test]
    fn join_horizontal_low_crops_to_shorter() {
        let out = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            None,
            None,
            None,
        );
        assert_eq!((out.width(), out.height()), (5, 2));
        assert_eq!(out.data(), &[1, 2, 3, 10, 20, 4, 5, 6, 30, 40]);
    }

    /**
     * Tests that `expand` keeps every input pixel and fills the gap with
     * the background.
     * Input: a(3x2) + c(2x3), horizontal, expand -> 5x3 with a black
     * bottom-left corner.
     */
    #[test]
    fn join_horizontal_low_expand_keeps_all_pixels() {
        let out = join_a().join(&join_c(), JoinDirection::Horizontal, true, None, None, None);
        assert_eq!((out.width(), out.height()), (5, 3));
        assert_eq!(
            out.data(),
            &[1, 2, 3, 10, 20, 4, 5, 6, 30, 40, 0, 0, 0, 50, 60]
        );
    }

    /**
     * Tests the HIGH alignment on a taller second image, which puts the
     * insert origin at a negative y. The expanded form grows upward and
     * the cropped form starts at `max(0, y) - y`, i.e. row 1.
     * Input: a(3x2) + c(2x3), horizontal, align high.
     */
    #[test]
    fn join_horizontal_high_negative_y() {
        let expanded = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            true,
            None,
            None,
            Some(Align::High),
        );
        assert_eq!((expanded.width(), expanded.height()), (5, 3));
        assert_eq!(
            expanded.data(),
            &[0, 0, 0, 10, 20, 1, 2, 3, 30, 40, 4, 5, 6, 50, 60]
        );

        let cropped = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            None,
            None,
            Some(Align::High),
        );
        assert_eq!((cropped.width(), cropped.height()), (5, 2));
        assert_eq!(cropped.data(), &[1, 2, 3, 30, 40, 4, 5, 6, 50, 60]);
    }

    /**
     * Tests the vertical direction, both expand settings: the second
     * image goes below the first and the crop takes the narrower width.
     * Input: a(3x2) + c(2x3) vertical -> 2x5 cropped, 3x5 expanded.
     */
    #[test]
    fn join_vertical_low() {
        let cropped = join_a().join(&join_c(), JoinDirection::Vertical, false, None, None, None);
        assert_eq!((cropped.width(), cropped.height()), (2, 5));
        assert_eq!(cropped.data(), &[1, 2, 4, 5, 10, 20, 30, 40, 50, 60]);

        let expanded = join_a().join(&join_c(), JoinDirection::Vertical, true, None, None, None);
        assert_eq!((expanded.width(), expanded.height()), (3, 5));
        assert_eq!(
            expanded.data(),
            &[1, 2, 3, 4, 5, 6, 10, 20, 0, 30, 40, 0, 50, 60, 0]
        );
    }

    /**
     * Tests the vertical HIGH alignment with a wider second image, the
     * negative-x twin of the horizontal case: the expanded canvas grows
     * leftward and the crop starts at column `max(0, x) - x` = 1.
     * Input: 4x2 + 5x2 vertical, align high.
     */
    #[test]
    fn join_vertical_high_negative_x() {
        let f = gray8(4, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let g = gray8(5, 2, vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);

        let expanded = f.join(
            &g,
            JoinDirection::Vertical,
            true,
            None,
            None,
            Some(Align::High),
        );
        assert_eq!((expanded.width(), expanded.height()), (5, 4));
        assert_eq!(
            expanded.data(),
            &[
                0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
            ]
        );

        let cropped = f.join(
            &g,
            JoinDirection::Vertical,
            false,
            None,
            None,
            Some(Align::High),
        );
        assert_eq!((cropped.width(), cropped.height()), (4, 4));
        assert_eq!(
            cropped.data(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20]
        );
    }

    /**
     * Tests the CENTRE ladder on the vertical axis, where the two
     * truncating divisions happen to agree with the combined form:
     * 4/2 - 5/2 = 2 - 2 = 0.
     * Input: 4x2 + 5x2 vertical, align centre.
     */
    #[test]
    fn join_vertical_centre() {
        let f = gray8(4, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let g = gray8(5, 2, vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);

        let expanded = f.join(
            &g,
            JoinDirection::Vertical,
            true,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((expanded.width(), expanded.height()), (5, 4));
        assert_eq!(
            expanded.data(),
            &[
                1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
            ]
        );

        let cropped = f.join(
            &g,
            JoinDirection::Vertical,
            false,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((cropped.width(), cropped.height()), (4, 4));
        assert_eq!(
            cropped.data(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19]
        );
    }

    /**
     * Tests that CENTRE is TWO separate truncating integer divisions
     * (`in1.h / 2 - in2.h / 2`), not the combined `(in1.h - in2.h) / 2`.
     * For 4 and 3 the C form gives `2 - 1 = 1` where the combined form
     * gives `0`, so the second image starts one row down and the top
     * right corner stays background.
     * Input: 2x4 + 2x3 horizontal, align centre -> 4x4 expanded.
     */
    #[test]
    fn join_centre_uses_two_truncating_divisions() {
        let d = gray8(2, 4, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let out = d.join(
            &join_c(),
            JoinDirection::Horizontal,
            true,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((out.width(), out.height()), (4, 4));
        // Row 0 is `1 2 | background background`: with the combined
        // `(4 - 3) / 2 = 0` form it would read `1 2 10 20`.
        assert_eq!(
            out.data(),
            &[1, 2, 0, 0, 3, 4, 10, 20, 5, 6, 30, 40, 7, 8, 50, 60]
        );

        let cropped = d.join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((cropped.width(), cropped.height()), (4, 3));
        assert_eq!(cropped.data(), &[1, 2, 0, 0, 3, 4, 10, 20, 5, 6, 30, 40]);
    }

    /**
     * Tests CENTRE with a taller second image, which drives y negative
     * (2/2 - 5/2 = 1 - 2 = -1) and exercises the `max(0, y) - y` crop
     * origin on the centre arm.
     * Input: 3x2 + 2x5 horizontal, align centre.
     */
    #[test]
    fn join_centre_negative_y() {
        let e = gray8(2, 5, vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);

        let expanded = join_a().join(
            &e,
            JoinDirection::Horizontal,
            true,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((expanded.width(), expanded.height()), (5, 5));
        assert_eq!(
            expanded.data(),
            &[
                0, 0, 0, 11, 12, 1, 2, 3, 13, 14, 4, 5, 6, 15, 16, 0, 0, 0, 17, 18, 0, 0, 0, 19, 20
            ]
        );

        let cropped = join_a().join(
            &e,
            JoinDirection::Horizontal,
            false,
            None,
            None,
            Some(Align::Centre),
        );
        assert_eq!((cropped.width(), cropped.height()), (5, 2));
        assert_eq!(cropped.data(), &[1, 2, 3, 13, 14, 4, 5, 6, 15, 16]);
    }

    /**
     * Tests that `shim` opens a gap of that many pixels between the two
     * images and that the gap takes the background colour, not black,
     * when one is given.
     * Input: a(3x2) + c(2x3) horizontal, shim 2 -> 7 wide; expanded with
     * background 128 and cropped with the default black.
     */
    #[test]
    fn join_shim_gap_takes_background() {
        let bg = [128.0];
        let expanded = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            true,
            Some(2),
            Some(&bg),
            None,
        );
        assert_eq!((expanded.width(), expanded.height()), (7, 3));
        assert_eq!(
            expanded.data(),
            &[
                1, 2, 3, 128, 128, 10, 20, 4, 5, 6, 128, 128, 30, 40, 128, 128, 128, 128, 128, 50,
                60
            ]
        );

        let cropped = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            Some(2),
            None,
            None,
        );
        assert_eq!((cropped.width(), cropped.height()), (7, 2));
        assert_eq!(
            cropped.data(),
            &[1, 2, 3, 0, 0, 10, 20, 4, 5, 6, 0, 0, 30, 40]
        );
    }

    /**
     * Tests a non-black background on the vertical arm: the shim row and
     * the column the narrower second image does not cover both take it.
     * Input: a(3x2) + c(2x3) vertical, shim 1, background 200 -> 3x6.
     */
    #[test]
    fn join_vertical_background_fills_uncovered() {
        let bg = [200.0];
        let out = join_a().join(
            &join_c(),
            JoinDirection::Vertical,
            true,
            Some(1),
            Some(&bg),
            None,
        );
        assert_eq!((out.width(), out.height()), (3, 6));
        assert_eq!(
            out.data(),
            &[
                1, 2, 3, 4, 5, 6, 200, 200, 200, 10, 20, 200, 30, 40, 200, 50, 60, 200
            ]
        );
    }

    /**
     * Tests that band alignment comes through from `insert`: a one-band
     * image joined with a three-band one gives three bands, the mono
     * samples replicated, and a three-value background is used per band.
     * Also tests that the promotion does not carry the first image's
     * interpretation onto a band count it no longer describes: `vips join
     * gbw.v csrgb.v out.v horizontal` reports `5x2 uchar, 3 bands, srgb`,
     * not `b-w`. The first image is tagged explicitly, the way a decoder
     * tags one, since an untagged Gray8 already infers `Bw` and would not
     * discriminate.
     * Input: Gray8 3x2 tagged `Bw` + Rgb8 2x3 horizontal, expand,
     * background [1,2,3].
     */
    #[test]
    fn join_band_promotion_mono_and_colour() {
        let colour = rgb8(
            2,
            3,
            vec![
                10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
            ],
        );
        let bg = [1.0, 2.0, 3.0];
        let mono = join_a().copy().interpretation(Interpretation::Bw).build();
        assert_eq!(mono.interpretation(), Interpretation::Bw);
        let out = mono.join(
            &colour,
            JoinDirection::Horizontal,
            true,
            None,
            Some(&bg),
            None,
        );
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!((out.width(), out.height()), (5, 3));
        assert_eq!(out.getpoint(0, 0), vec![1.0, 1.0, 1.0]);
        assert_eq!(out.getpoint(3, 0), vec![10.0, 11.0, 12.0]);
        assert_eq!(out.getpoint(0, 2), vec![1.0, 2.0, 3.0]);
        assert_eq!(out.getpoint(4, 2), vec![60.0, 61.0, 62.0]);
        assert_eq!(out.interpretation(), Interpretation::Srgb);
    }

    /**
     * Tests depth promotion: an 8-bit image joined with a 16-bit one
     * gives 16-bit samples with the numbers unchanged, and keeps the first
     * image's interpretation. `vips join` of a `b-w` uchar with a `grey16`
     * reports `b-w`, so a depth-only promotion must NOT trigger the
     * re-stamp that a band promotion does: the trigger is the band count
     * changing, nothing else.
     * Input: Gray8 3x2 tagged `Bw` + Gray16 2x3 horizontal, expand ->
     * Gray16 5x3 still tagged `Bw`.
     */
    #[test]
    fn join_depth_promotion_8_and_16_bit() {
        let wide = gray16(2, 3, &[10, 20, 30, 40, 50, 60]);
        let base = join_a().copy().interpretation(Interpretation::Bw).build();
        let out = base.join(&wide, JoinDirection::Horizontal, true, None, None, None);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!((out.width(), out.height()), (5, 3));
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(4, 2), vec![60.0]);
        assert_eq!(out.getpoint(0, 2), vec![0.0]);
        assert_eq!(out.interpretation(), Interpretation::Bw);
    }

    /**
     * Tests that the delegated `insert` band-count error surfaces as the
     * typed `ConversionError::Extract` wrapper: 2 bands against 3, with
     * neither being 1, is what libvips `bandalike` rejects too.
     */
    #[test]
    fn try_join_band_count_mismatch() {
        let two = Raster::zeroed(1, 1, PixelFormat::with_channels(2, 1).unwrap()).unwrap();
        let three = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            two.try_join(&three, JoinDirection::Horizontal, false, None, None, None),
            Err(ConversionError::Extract(ExtractError::BandCountMismatch {
                main: 2,
                sub: 3
            }))
        ));
    }

    /**
     * Tests that the join carries the first image's metadata, matching
     * the arrayjoin convention and libvips, which copies in1's fields.
     */
    #[test]
    fn join_meta_from_first() {
        let a = join_a().copy().xres(11.0).build();
        let out = a.join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            None,
            None,
            None,
        );
        assert_eq!(out.xres(), 11.0);
    }

    /**
     * Tests that a float input comes back as a typed error rather than
     * panicking out of a fallible method. The placement path underneath
     * reads samples as u8 or u16 and panics on 4-byte ones, and float input
     * is ordinary here: `space_depth` maps Lab, Lch, OkLab, OkLCh, XYZ,
     * scRGB and Yxy all to F32, so `im.colourspace(Lab).join(..)` is a
     * float join. Works by joining against the float `grey` ramp in each
     * operand position.
     * Input: FloatF32 2x2 ramp on either side of a Gray8 3x2.
     */
    #[test]
    fn try_join_float_input_is_a_typed_error_not_a_panic() {
        let ramp = Raster::grey(2, 2, false);
        assert!(ramp.format().is_float());
        for (a, b) in [(&ramp, &join_a()), (&join_a(), &ramp)] {
            assert!(matches!(
                a.try_join(b, JoinDirection::Horizontal, false, None, None, None),
                Err(ConversionError::FloatFormatUnsupported { op: "join" })
            ));
        }
    }

    /**
     * Tests the libvips bound on `shim`. `join.c` declares the property as
     * `VIPS_ARG_INT(class, "shim", 5, ..., 0, 1000000, 0)`, and the binary
     * refuses `vips join a.v c.v out.v horizontal --shim 1000001` at the
     * property boundary. Without the check one argument buys a
     * multi-gigabyte canvas out of two tiny inputs, each raster still under
     * the per-raster allocation budget so the budget never fires.
     * Works by asserting the typed error one past the bound, which costs no
     * allocation at all.
     * Input: 3x2 + 2x3, shim 1000001.
     */
    #[test]
    fn try_join_shim_above_the_vips_maximum_is_rejected() {
        assert!(matches!(
            join_a().try_join(
                &join_c(),
                JoinDirection::Horizontal,
                false,
                Some(1_000_001),
                None,
                None,
            ),
            Err(ConversionError::ShimTooLarge {
                shim: 1_000_001,
                max: 1_000_000
            })
        ));
        // And well past it, where the old code allocated 6.44 GB.
        assert!(matches!(
            join_a().try_join(
                &join_c(),
                JoinDirection::Horizontal,
                false,
                Some(2_147_483_644),
                None,
                None,
            ),
            Err(ConversionError::ShimTooLarge { .. })
        ));
    }

    /**
     * Tests that the bound is inclusive, so the check is off by nothing:
     * `vips join a.v c.v out.v horizontal --shim 1000000` succeeds and
     * reports `1000005x2 uchar`.
     * Input: 3x2 + 2x3, shim 1000000 -> 1000005x2.
     */
    #[test]
    fn join_shim_at_the_vips_maximum_is_accepted() {
        let out = join_a().join(
            &join_c(),
            JoinDirection::Horizontal,
            false,
            Some(1_000_000),
            None,
            None,
        );
        assert_eq!((out.width(), out.height()), (1_000_005, 2));
    }

    /**
     * Tests the placement range check and what it reports. `join.c` does
     * its offset arithmetic in `int` and `insert` takes `i32`, so an offset
     * outside that range cannot be placed. The error names the offset that
     * did not fit, not a result size: the old message reported
     * `result size 2147483650x3 exceeds u32::MAX` for a value comfortably
     * under `u32::MAX`, which was simply false.
     * Tested on the placement helper rather than through `try_join`,
     * because reaching it through the real call needs a 3 GB input now that
     * `shim` is bounded; the helper is what `try_join` calls, unchanged.
     * Input: 1x1 joined vertically with 3000000000x1 on align high, where
     * `x = 1 - 3e9` falls below `i32::MIN` even though the canvas would fit
     * `u32` and the allocation budget.
     */
    #[test]
    fn join_placement_offset_outside_i32_is_typed_and_honest() {
        assert!(matches!(
            join_placement(
                (1, 1),
                (3_000_000_000, 1),
                JoinDirection::Vertical,
                Align::High,
                0,
            ),
            Err(ConversionError::PlacementOffsetOverflow {
                x: -2_999_999_999,
                y: 1
            })
        ));
        // Above i32::MAX as well, from a wide first image plus a shim.
        assert!(matches!(
            join_placement(
                (4_000_000_000, 1),
                (1, 1),
                JoinDirection::Horizontal,
                Align::Low,
                1_000_000,
            ),
            Err(ConversionError::PlacementOffsetOverflow {
                x: 4_001_000_000,
                y: 0
            })
        ));
        // The message names the offset, and nothing else.
        let e = join_placement(
            (1, 1),
            (3_000_000_000, 1),
            JoinDirection::Vertical,
            Align::High,
            0,
        )
        .unwrap_err();
        assert_eq!(
            e.to_string(),
            "join placement offset (-2999999999, 1) does not fit i32"
        );
    }

    /**
     * Tests the libvips align nicknames and the typed error for an
     * unknown one.
     */
    #[test]
    fn align_from_str_nicknames() {
        assert_eq!("low".parse::<Align>().unwrap(), Align::Low);
        assert_eq!("centre".parse::<Align>().unwrap(), Align::Centre);
        assert_eq!("center".parse::<Align>().unwrap(), Align::Centre);
        assert_eq!("high".parse::<Align>().unwrap(), Align::High);
        assert!(matches!(
            "middle".parse::<Align>(),
            Err(ConversionError::UnknownAlign { .. })
        ));
    }

    // ------------------------------------------------------------------
    // grey
    // ------------------------------------------------------------------

    /**
     * Tests the exact-ramp property the switch fixtures rely on: at 256
     * wide, pixel(x, y) == x on every row.
     */
    #[test]
    fn grey_uchar_ramp_256_is_exact() {
        let im = Raster::grey(256, 2, true);
        assert_eq!(im.format(), PixelFormat::Gray8);
        for x in [0u32, 1, 128, 254, 255] {
            assert_eq!(im.getpoint(x, 0), vec![x as f64]);
            assert_eq!(im.getpoint(x, 1), vec![x as f64]);
        }
    }

    /**
     * Tests the ported test_grey uchar endpoints at width 100.
     */
    #[test]
    fn grey_uchar_endpoints() {
        let im = Raster::grey(100, 90, true);
        assert_eq!((im.width(), im.height()), (100, 90));
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(99, 0), vec![255.0]);
    }

    /**
     * Tests the width-1 degenerate ramp (no division by zero).
     */
    #[test]
    fn grey_width_one_is_zero() {
        let im = Raster::grey(1, 2, true);
        assert_eq!(im.data(), &[0, 0]);
    }

    /**
     * Tests the ported test_grey float endpoints: grey(100, 90, false) is
     * a single-band FloatF32 ramp with pixel(0,0) == 0.0 and
     * pixel(99,0) == 1.0, every row identical. This is the exact call
     * shape of ported_create.rs::test_grey's float half.
     */
    #[test]
    fn grey_float_ramp_endpoints() {
        let im = Raster::grey(100, 90, false);
        assert_eq!((im.width(), im.height()), (100, 90));
        assert_eq!(im.format(), PixelFormat::with_channels(1, 4).unwrap());
        assert!(im.format().is_float());
        let p = im.getpoint(0, 0);
        assert!((p[0] - 0.0).abs() < 0.001, "left edge should be 0.0");
        let p = im.getpoint(99, 0);
        assert!((p[0] - 1.0).abs() < 0.001, "right edge should be 1.0");
        // Every row is identical.
        assert_eq!(im.getpoint(42, 0), im.getpoint(42, 89));
    }

    /**
     * Tests interior float ramp values: pixel(x) == x / (w - 1) exactly
     * at f32 precision for a 256-wide ramp.
     */
    #[test]
    fn grey_float_ramp_is_linear() {
        let im = Raster::grey(256, 1, false);
        for x in [0u32, 1, 128, 254, 255] {
            let expected = (x as f64 / 255.0) as f32 as f64;
            assert_eq!(im.getpoint(x, 0), vec![expected]);
        }
    }

    /**
     * Tests the width-1 degenerate float ramp (no division by zero).
     */
    #[test]
    fn grey_float_width_one_is_zero() {
        let im = Raster::grey(1, 2, false);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(0, 1), vec![0.0]);
    }

    // ------------------------------------------------------------------
    // identity
    // ------------------------------------------------------------------

    /**
     * Tests the ported test_identity 8-bit assertions exactly.
     */
    #[test]
    fn identity_lut_exact() {
        let im = Raster::identity();
        assert_eq!((im.width(), im.height()), (256, 1));
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(128, 0), vec![128.0]);
        assert_eq!(im.getpoint(255, 0), vec![255.0]);
    }

    /**
     * Tests the ported test_identity 16-bit assertions exactly.
     */
    #[test]
    fn identity_ushort_exact() {
        let im = Raster::identity_ushort();
        assert_eq!((im.width(), im.height()), (65536, 1));
        assert_eq!(im.format(), PixelFormat::Gray16);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(99, 0), vec![99.0]);
        assert_eq!(im.getpoint(65535, 0), vec![65535.0]);
    }

    // ------------------------------------------------------------------
    // switch
    // ------------------------------------------------------------------

    /**
     * Tests first-true indexing over explicit masks.
     * Input: conds ([255,0,0], [0,255,0]) per pixel -> [0, 1, 2].
     */
    #[test]
    fn switch_selects_first_true_index() {
        let c0 = gray8(3, 1, vec![255, 0, 0]);
        let c1 = gray8(3, 1, vec![0, 255, 0]);
        let out = Raster::switch(&[&c0, &c1]);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.data(), &[0, 1, 2]);
    }

    /**
     * Tests the no-match value: all-false conditions give N everywhere,
     * the ported "avg = 2" case.
     */
    #[test]
    fn switch_no_match_yields_condition_count() {
        let c0 = gray8(2, 1, vec![0, 0]);
        let c1 = gray8(2, 1, vec![0, 0]);
        let out = Raster::switch(&[&c0, &c1]);
        assert_eq!(out.data(), &[2, 2]);
    }

    /**
     * Tests that overlapping true conditions resolve to the first.
     */
    #[test]
    fn switch_first_true_wins_on_overlap() {
        let c0 = gray8(1, 1, vec![7]);
        let c1 = gray8(1, 1, vec![255]);
        let out = Raster::switch(&[&c0, &c1]);
        assert_eq!(out.data(), &[0]);
    }

    /**
     * Tests that 16-bit conditions use a non-zero sample test.
     */
    #[test]
    fn switch_16bit_conditions() {
        let c0 = gray16(2, 1, &[0, 300]);
        let out = Raster::switch(&[&c0]);
        assert_eq!(out.data(), &[1, 0]);
    }

    /**
     * Tests the ported test_switch shape: the two ramp masks split a
     * 256-wide grey ramp into an index image with avg 0.5.
     */
    #[test]
    fn switch_ramp_masks_average_half() {
        let x = Raster::grey(256, 256, true);
        let cond_lo = x.less_than_const(128.0);
        let cond_hi = x.more_eq_const(128.0);
        let index = Raster::switch(&[&cond_lo, &cond_hi]);
        assert!((index.avg() - 0.5).abs() < 0.01);
    }

    /**
     * Tests the typed errors: empty list, dimension mismatch, a
     * multi-band condition, and more than 255 conditions.
     */
    #[test]
    fn try_switch_errors() {
        assert!(matches!(
            Raster::try_switch(&[]),
            Err(ConversionError::EmptyInput { .. })
        ));

        let a = gray8(2, 1, vec![0, 0]);
        let b = gray8(1, 1, vec![0]);
        assert!(matches!(
            Raster::try_switch(&[&a, &b]),
            Err(ConversionError::DimensionMismatch { .. })
        ));

        let colour = Raster::zeroed(2, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            Raster::try_switch(&[&a, &colour]),
            Err(ConversionError::ConditionNotMono { .. })
        ));

        let one = gray8(1, 1, vec![0]);
        let many: Vec<&Raster> = std::iter::repeat_n(&one, 256).collect();
        assert!(matches!(
            Raster::try_switch(&many),
            Err(ConversionError::TooManyConditions { .. })
        ));
    }

    /**
     * Tests that Interpretation::for_format reads the pixel layout and not
     * the spelling of it. PixelFormat's tuple variants are public, so
     * FloatF32(4) and RgbaF32 name one layout; for_format used to call the
     * first Multiband and the second Srgb, which is the documented answer
     * for four-band float (issue #531).
     * Works by walking every layout that has both spellings and asserting
     * the two land on the same interpretation.
     * Input: FloatF32(4) vs RgbaF32 -> Srgb for both; Multi8(1) vs Gray8 ->
     * Bw for both; Multi16(4) vs Rgba16 -> Rgb16 for both.
     */
    #[test]
    fn for_format_reads_the_layout_not_the_spelling() {
        let nz = |n: u16| core::num::NonZeroU16::new(n).expect("the table holds no zeroes");
        for (alias, named) in [
            (PixelFormat::Multi8(nz(1)), PixelFormat::Gray8),
            (PixelFormat::Multi8(nz(3)), PixelFormat::Rgb8),
            (PixelFormat::Multi8(nz(4)), PixelFormat::Rgba8),
            (PixelFormat::Multi16(nz(1)), PixelFormat::Gray16),
            (PixelFormat::Multi16(nz(3)), PixelFormat::Rgb16),
            (PixelFormat::Multi16(nz(4)), PixelFormat::Rgba16),
            (PixelFormat::FloatF32(nz(4)), PixelFormat::RgbaF32),
        ] {
            assert_eq!(
                Interpretation::for_format(alias),
                Interpretation::for_format(named),
                "for_format({alias:?}) must match for_format({named:?})"
            );
        }
    }
}
