//! Colour-space and ICC operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`], and
//! [`crate::composite`]): colour-space conversion between the libvips
//! interpretations, the CIE colour-difference metrics, and ICC profile
//! import/export/transform. Operations that can fail on caller input exist
//! in two forms, following the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, ColourError>` with
//!   typed errors for unsupported spaces, missing bands, mismatched
//!   dimensions, and ICC profile problems; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`colourspace`, `de00`, `icc_import`, ...) exactly, delegating to the
//!   `try_*` form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::constant`] | `black + c` idiom | constant-colour float image |
//! | [`Raster::colourspace`] | `vips_colourspace` | image in the target colour space |
//! | [`Raster::de76`] | `vips_dE76` | CIE76 colour difference |
//! | [`Raster::de00`] | `vips_dE00` | CIEDE2000 colour difference |
//! | [`Raster::de_cmc`] | `vips_dECMC` | CMC colour difference |
//! | [`Raster::icc_import`] | `vips_icc_import` | device image imported to PCS |
//! | [`Raster::icc_export`] | `vips_icc_export` | PCS image exported to device |
//! | [`Raster::icc_transform`] | `vips_icc_transform` | device image re-profiled |
//!
//! # Colour-space model
//!
//! Conversion is modelled on a hub: every supported space converts to and
//! from CIE XYZ (D65-relative, `Y` white = 100), and a conversion from
//! space `A` to space `B` runs `A -> XYZ -> B`.
//!
//! That is a summary of the libvips route table
//! (`colour/colourspace.c:223-497`), not a transcription of it. libvips
//! stores an explicit pipeline per ordered pair, and plenty of those
//! pipelines never reach XYZ. This port takes ten of them directly, the
//! same-family cartesian/polar pairs `Lab <-> Lch` (`:244`, `:276`) and
//! `OkLab <-> OkLCh` (`:478`, `:494`), and the three pairs that reach the
//! signed-16-bit coding, `Lab <-> Labs` (`:246`, `:310`),
//! `Lch <-> Labs` (`:280`, `:312`) and `Cmc <-> Labs` (`:297`, `:313`),
//! because on those the hub inserts a round trip that changes the answer
//! rather than only costing time (the `direct_edge` source notes carry
//! the measured damage). Everything else goes through the hub here,
//! including these hub-free edges of libvips', which is where the next
//! direct route will be wanted:
//!
//! * the rest of the Lab family reaches `Lch` and `Cmc` from each other
//!   with no XYZ step: `{ LAB, CMC }` (`:245`), `{ LCH, CMC }` (`:279`),
//!   `{ CMC, LAB }` (`:293`), `{ CMC, LCH }` (`:295`);
//! * the 8/16-bit RGB block (`srgb`, `scrgb`, `hsv`, `b-w`, `rgb16`,
//!   `grey16`) converts among itself with no XYZ step (`:352-441`).
//!
//! All intermediate maths is `f64`; quantisation happens only where libvips
//! quantises, i.e. when a space is stored at 8 or 16 bits (`srgb`, `hsv`,
//! `cmyk`, `b-w` at 8 bits; `rgb16`, `grey16` at 16 bits) and inside the
//! XYZ -> HSV step, which passes through 8-bit sRGB exactly as the libvips
//! route does. The one deliberate exception is the linear -> sRGB store,
//! which is `f32` throughout because libvips' is (see below). The
//! individual conversions use the same published formulas and constants
//! as libvips:
//!
//! * sRGB gamma per IEC 61966-2-1 (linear below 0.04045 / 0.0031308), but
//!   only the sRGB -> linear direction is EVALUATED. Going the other way,
//!   libvips reads a precomputed integer table: `calcul_tables`
//!   (`colour/LabQ2sRGB.c:126-146`) rounds `range` samples of the curve
//!   to integer codes in `float`, and `vips_col_scRGB2sRGB` (`:282-353`)
//!   interpolates linearly between two of those rounded entries and
//!   finishes with `rintf`, which rounds halves to EVEN. That is three
//!   quantisations, and evaluating the curve analytically instead missed
//!   vips by a whole count on 16.6% of codes (issue #581), so the table
//!   is ported rather than the formula. `b-w`, `grey16`, `srgb`, `rgb16`
//!   and the sRGB step of `hsv` all read it;
//! * the sRGB primaries matrix for scRGB <-> XYZ (4-decimal forward,
//!   6-decimal inverse, both scaled to `Y` white = 100);
//! * CIE Lab with the 7.787 shadow slope and D65 white
//!   (95.047, 100, 108.8827);
//! * LCh as the cylindrical form of Lab;
//! * the CMC (Colour Measurement Committee) uniform-space functions for
//!   the `cmc` space, inverted numerically (libvips inverts them through
//!   interpolation tables; the bisection used here is at least as
//!   accurate);
//! * `labs` as Lab scaled to the signed-16-bit code range
//!   (`L * 32767/100`, `a`,`b * 256`), clipped and then **truncated
//!   toward zero**, because `colour/Lab2LabS.c:66-68` stores the clipped
//!   double into a `signed short`. The Lab value is rounded to `f32`
//!   first, because `Lab2LabS.c:59` reads a `float` image and every
//!   libvips route into LabS hands it one. There is no signed 16-bit
//!   [`PixelFormat`], so the samples are carried in a float raster whose
//!   values match the libvips LabS codes;
//! * Oklab / OkLCh per Ottosson's published matrices (the same constants
//!   libvips uses);
//! * Yxy chromaticity, the HSV hue circle mapped to 0..255, and the
//!   libvips no-lcms CMYK approximation (naive ink model over
//!   D65-normalised XYZ);
//! * mono (`b-w`, `grey16`) as CIE linear luminance
//!   (0.2126 R + 0.7152 G + 0.0722 B) taken through the same table, and
//!   grey sources replicated to RGB exactly like the libvips `BW2sRGB`
//!   route;
//! * HSV over 8-bit sRGB with both the hue and the saturation code
//!   TRUNCATED on the store, because `sRGB2HSV.c:113-117` writes them
//!   into an `unsigned char`.
//!
//! Extra bands beyond the colour bands of the source space are carried
//! through unchanged and plain-cast to the output depth (clip, no
//! rescale), mirroring `vips__colourspace_process_n`. Source images tagged
//! [`Interpretation::Rgb`] are treated as sRGB and
//! [`Interpretation::Matrix`] as mono, mirroring `vips_colourspace`.
//!
//! [`Raster::colourspace`] accepts either an [`Interpretation`] value or a
//! libvips space nickname (`"srgb"`, `"scrgb"`, `"lab"`, ...): the ported
//! foreign tests use the string shape, the ported colour tests the enum
//! shape. The nickname parse is exposed as `FromStr` on
//! [`Interpretation`].
//!
//! # ICC
//!
//! The ICC operations are real profile transforms implemented on the
//! pure-Rust [moxcms](https://crates.io/crates/moxcms) CMS (BSD-3-Clause
//! OR Apache-2.0). Profiles are parsed from the raster's attached
//! `icc-profile-data` field (see [`Raster::icc_profile`]) or from a
//! profile file. Two engine paths cover the profile classes:
//!
//! * **Matrix-shaper profiles** (sRGB, Display P3, Adobe RGB, camera RGB
//!   profiles, plus grey profiles with a `grayTRC`): evaluated exactly,
//!   per channel, from the parsed TRC curves and colorant matrix. Device
//!   values convert to D50 PCS XYZ and then to Lab with the ICC D50 white
//!   point (0.9642, 1.0, 0.8249); export inverts the same pipeline. The
//!   round trip is exact to quantisation.
//! * **LUT profiles** (CMYK and other table-based profiles): run through
//!   a moxcms float transform against its generic Lab profile, whose LUT
//!   pipeline delivers ICC-encoded PCS XYZ (white = 1/2 of full scale);
//!   the codes are decoded and finished with the same exact XYZ -> Lab
//!   maths. LUT pipelines interpolate, so round trips through this path
//!   carry the usual CMS grid-interpolation error.
//!
//! Rendering [`Intent`] selects the LUT set on LUT profiles.
//! Matrix-shaper profiles have a single colorimetric tag set, so intent
//! does not change their result (the same is true under lcms). The PCS
//! Lab produced by ICC import is D50-relative per the ICC specification,
//! while [`Raster::colourspace`] Lab is the D65-relative libvips Lab; the
//! ported suite's dE76-under-6 thresholds absorb exactly this difference,
//! as they do under libvips + lcms.
//!
//! [`Raster::icc_import`] keeps the source profile attached, so a
//! following [`Raster::icc_export`] with no explicit output profile
//! round-trips through the same profile, and export attaches the output
//! profile it used, both mirroring libvips.
//!
//! # Deferred
//!
//! * `max_value` (used by the ported ICC test alongside these ops) is
//!   part of the create/arithmetic surface, not this batch; the
//!   colour-difference rasters are float, so until the float arithmetic
//!   batch lands they read through [`Raster::getpoint`] /
//!   [`Raster::f32_samples`].
//! * The `histogram` / `fourier` / `multiband` pseudo-interpretations have
//!   no colourspace route, exactly as in libvips: the route table carries
//!   no `from` row for any of them (`colour/colourspace.c:223-497`), and
//!   `vips_colourspace_issupported` (`:511-535`) calls a space unsupported
//!   precisely when scanning those `from` fields finds nothing. They yield
//!   [`ColourError::UnsupportedColourspace`].
//! * The packed `labq` coding yields the same error, but that one is a
//!   **libviprs limitation rather than libvips parity**: libvips routes
//!   LabQ both ways against every space (`{ LAB, LABQ,
//!   { vips_Lab2LabQ } }` at `colour/colourspace.c:243` and the whole
//!   `LABQ` block at `:258-273`). LabQ is both an interpretation and a
//!   coding in libvips (`VIPS_INTERPRETATION_LABQ = 16` and
//!   `VIPS_CODING_LABQ = 2`, `include/vips/image.h:102,138`), and it is
//!   the coding half libviprs has no home for: four `u8` carrying three
//!   logical channels at 10:11:11 with a shared low-bits byte. There is no
//!   coding concept here for that carrier to live in, so there is nothing
//!   for a route to produce (issue #552 records the gap).
//! * libvips built with lcms converts `cmyk` through an embedded generic
//!   CMYK profile; the ported CMYK tests target the no-lcms approximation,
//!   which is what [`Raster::colourspace`] implements. Profiled CMYK is
//!   available through [`Raster::icc_import`] with a CMYK profile.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use moxcms::{
    ColorProfile, DataColorSpace, Layout, RenderingIntent, ToneCurveEvaluator, TransformOptions,
    Vector3d,
};
use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// Typed errors for the colour operations in [`crate::colour`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ColourError {
    /// A colour-space nickname failed to parse.
    #[error("unknown colour space name {name:?}")]
    UnknownColourspace { name: String },
    /// The interpretation has no colourspace route (`labq`, `histogram`,
    /// `fourier`, `multiband`, `matrix` as a target, ...).
    #[error("no colourspace route for {interpretation:?}")]
    UnsupportedColourspace { interpretation: Interpretation },
    /// The image has fewer bands than its colour space needs.
    #[error("too few bands for {interpretation:?}: needs {needed}, image has {got}")]
    TooFewBands {
        interpretation: Interpretation,
        needed: usize,
        got: usize,
    },
    /// Two rasters that must share pixel dimensions do not.
    #[error("dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        got_w: u32,
        got_h: u32,
    },
    /// An ICC operation needs a profile but the image has no
    /// `icc-profile-data` field and no profile path was supplied.
    #[error("no ICC profile: the image has no icc-profile-data and no profile was supplied")]
    NoProfile,
    /// An ICC profile file could not be read.
    #[error("failed to read ICC profile {path:?}")]
    ProfileRead {
        path: PathBuf,
        source: std::io::Error,
    },
    /// ICC profile bytes failed to parse.
    #[error("invalid ICC profile: {detail}")]
    InvalidProfile { detail: String },
    /// The profile's device colour space is not Gray, RGB, or CMYK.
    #[error("unsupported ICC device colour space {space}")]
    UnsupportedDeviceSpace { space: String },
    /// An ICC export depth other than 8 or 16 was requested.
    #[error("unsupported ICC export depth {depth}: must be 8 or 16")]
    UnsupportedDepth { depth: u32 },
    /// The CMS could not build or run a transform for this profile pair.
    #[error("ICC transform failed: {detail}")]
    IccTransform { detail: String },
}

// ---------------------------------------------------------------------------
// Interpretation nickname parsing
// ---------------------------------------------------------------------------

impl FromStr for Interpretation {
    type Err = ColourError;

    /// Parse a libvips colour-space nickname (case-insensitive):
    /// `"multiband"`, `"b-w"` (or `"bw"`), `"histogram"`, `"xyz"`,
    /// `"lab"`, `"cmyk"`, `"labq"`, `"rgb"`, `"cmc"`, `"lch"`, `"labs"`,
    /// `"srgb"`, `"yxy"`, `"fourier"`, `"rgb16"`, `"grey16"`, `"matrix"`,
    /// `"scrgb"`, `"hsv"`, `"oklab"`, `"oklch"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "multiband" => Self::Multiband,
            "b-w" | "bw" => Self::Bw,
            "histogram" => Self::Histogram,
            "xyz" => Self::Xyz,
            "lab" => Self::Lab,
            "cmyk" => Self::Cmyk,
            "labq" => Self::Labq,
            "rgb" => Self::Rgb,
            "cmc" => Self::Cmc,
            "lch" => Self::Lch,
            "labs" => Self::Labs,
            "srgb" => Self::Srgb,
            "yxy" => Self::Yxy,
            "fourier" => Self::Fourier,
            "rgb16" => Self::Rgb16,
            "grey16" => Self::Grey16,
            "matrix" => Self::Matrix,
            "scrgb" => Self::ScRgb,
            "hsv" => Self::Hsv,
            "oklab" => Self::OkLab,
            "oklch" => Self::OkLch,
            _ => {
                return Err(ColourError::UnknownColourspace {
                    name: s.to_string(),
                });
            }
        })
    }
}

/// The `&str` call shape of [`Raster::colourspace`]
/// (`im.colourspace("scrgb")`, used by the ported foreign tests).
///
/// # Panics
///
/// Panics if the name is not a libvips colour-space nickname, mirroring
/// the "known-good input" contract of the panicking convenience methods.
/// Use [`Interpretation::from_str`] for a fallible parse.
impl From<&str> for Interpretation {
    fn from(s: &str) -> Self {
        match s.parse() {
            Ok(i) => i,
            Err(e) => panic!("colourspace: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared colour maths (libvips constants throughout)
// ---------------------------------------------------------------------------

/// D65 white point, `Y` white = 100 (libvips `VIPS_D65_*0`).
const D65: [f64; 3] = [95.0470, 100.0, 108.8827];

/// ICC D50 PCS illuminant, `Y` white = 1 (ICC.1 nCIEXYZ).
const ICC_D50: [f64; 3] = [0.9642, 1.0, 0.8249];

/// The ICC `u1Fixed15` PCS-XYZ code scale: a stored 1.0 encodes
/// 65535/32768 XYZ units, so PCS white `Y = 1.0` is stored as 0.5 of full
/// scale. moxcms LUT pipelines deliver PCS XYZ in this encoding.
const PCS_XYZ_SCALE: f64 = 65535.0 / 32768.0;

fn deg_to_rad(d: f64) -> f64 {
    d.to_radians()
}

/// CIE Lab `f`: cube root above the 0.008856 shadow threshold, the 7.787
/// linear segment below it (the constants libvips tables encode).
fn lab_f(t: f64) -> f64 {
    if t < 0.008856 {
        7.787 * t + 16.0 / 116.0
    } else {
        t.cbrt()
    }
}

/// XYZ -> Lab against the white point `w` (same shadow constants as
/// libvips `XYZ2Lab`).
fn xyz_to_lab(xyz: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / w[0]);
    let fy = lab_f(xyz[1] / w[1]);
    let fz = lab_f(xyz[2] / w[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Lab -> XYZ against the white point `w` (same branch constants as
/// libvips `Lab2XYZ`: 903.3 shadow lightness, 0.2069/0.13793/7.787
/// segment).
fn lab_to_xyz(lab: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    let (y, cby) = if lab[0] < 8.0 {
        let y = lab[0] * w[1] / 903.3;
        (y, 7.787 * (y / w[1]) + 16.0 / 116.0)
    } else {
        let cby = (lab[0] + 16.0) / 116.0;
        (w[1] * cby * cby * cby, cby)
    };

    let tmp = lab[1] / 500.0 + cby;
    let x = if tmp < 0.2069 {
        w[0] * (tmp - 0.13793) / 7.787
    } else {
        w[0] * tmp * tmp * tmp
    };

    let tmp = cby - lab[2] / 200.0;
    let z = if tmp < 0.2069 {
        w[2] * (tmp - 0.13793) / 7.787
    } else {
        w[2] * tmp * tmp * tmp
    };

    [x, y, z]
}

/// sRGB electro-optical transfer: encoded 0..1 to linear 0..1
/// (IEC 61966-2-1).
fn srgb_decode(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Number of codes in the 8-bit sRGB carrier, the `range` libvips hands
/// `calcul_tables` from `calcul_tables_8` (`colour/LabQ2sRGB.c:153`).
const SRGB_RANGE: usize = 256;

/// Number of codes in the 16-bit `rgb16` / `grey16` carrier, the `range`
/// from `calcul_tables_16` (`colour/LabQ2sRGB.c:174`).
const RGB16_RANGE: usize = 65536;

/// Build one libvips `Y2v` table: `range` samples of the sRGB
/// opto-electrical transfer (IEC 61966-2-1, linear below 0.0031308) taken
/// at `i / (range - 1)`, scaled to `range - 1` and rounded to an integer,
/// plus a duplicated final element.
///
/// This is `calcul_tables` (`colour/LabQ2sRGB.c:126-146`) with its `v2Y`
/// half left out. Nothing needs the reverse table: the sRGB -> linear
/// direction stays analytic in `f64` ([`srgb_decode`]), and the only
/// difference that makes is an `f32` rounding of a linear value, far
/// under a code. The forward direction is a different story, which is
/// what [`scrgb_to_code`] is about.
///
/// Everything is deliberately `f32`, the transfer function and the final
/// `round_ties_even` (C `rintf`) alike, because the C is.
///
/// The `mul_add` is the one detail the C source does not show. The arm64
/// Homebrew build of 8.18.4 contracts
/// `(1.0F + 0.055F) * powf(f, 1.0F / 2.4F) - 0.055F` into a single
/// `fmadd`, which `otool -tvV -p _calcul_tables` on `libvips.42.dylib`
/// prints as `fmadd s0, s0, s9, s13`. Evaluating it unfused moves 45 of
/// the 65536 16-bit entries by a count, so the fusion is pinned here
/// rather than left to the optimiser, which is not allowed to introduce
/// it on its own. The 256-entry table is identical either way.
///
/// This site keeps the `f32::mul_add` and [`scrgb_to_code`] does not,
/// which is not an oversight. `f32::mul_add` becomes a libm `fmaf` call
/// wherever `fma` is missing from the baseline ISA, x86-64 included, so
/// it is worth routing around per channel per pixel and not worth it
/// here, where it runs `range` times behind a `OnceLock`. What is
/// written here is the C expression fused exactly as the shipped dylib
/// fuses it, and it should keep reading that way. Do not propagate the
/// other site's `f64` spelling back into this one.
///
/// The trailing duplicate is the C's: "Copy the final element. This is
/// used in the piecewise linear interpolator below." (`:141-144`).
fn calcul_tables(range: usize) -> Box<[i32]> {
    let maxval = (range - 1) as f32;
    let mut y2v: Vec<i32> = Vec::with_capacity(range + 1);
    for i in 0..range {
        let f = i as f32 / maxval;
        // The C compares the promoted `float` against a `double`
        // literal, so the branch point is not an `f32` constant.
        let v = if f64::from(f) <= 0.0031308 {
            12.92_f32 * f
        } else {
            f.powf(1.0_f32 / 2.4_f32).mul_add(1.055, -0.055)
        };
        y2v.push((maxval * v).round_ties_even() as i32);
    }
    y2v.push(y2v[range - 1]);
    y2v.into_boxed_slice()
}

/// The `Y2v` table for `range`, built once, standing in for the libvips
/// `VIPS_ONCE` pair (`colour/LabQ2sRGB.c:150-170`).
fn y2v_table(range: usize) -> &'static [i32] {
    static Y2V_8: OnceLock<Box<[i32]>> = OnceLock::new();
    static Y2V_16: OnceLock<Box<[i32]>> = OnceLock::new();
    debug_assert!(range == SRGB_RANGE || range == RGB16_RANGE);
    if range == SRGB_RANGE {
        Y2V_8.get_or_init(|| calcul_tables(SRGB_RANGE))
    } else {
        Y2V_16.get_or_init(|| calcul_tables(RGB16_RANGE))
    }
}

/// Linear scRGB (0..1) -> the integer sRGB code, through the
/// interpolated `Y2v` lookup that `vips_col_scRGB2sRGB`
/// (`colour/LabQ2sRGB.c:282-353`) and `vips_col_scRGB2BW` (`:385-428`)
/// share.
///
/// vips never evaluates the transfer function per pixel, and the
/// difference is not academic: three quantisations stack here, all of
/// them the C's. [`calcul_tables`] samples the curve at `range` points
/// and rounds each one to an integer; this lookup interpolates linearly
/// between two of those already-rounded points; and the chord is
/// finished with `rintf`, which rounds halves to EVEN rather than away
/// from zero. Evaluating the curve analytically in `f64` and rounding
/// once instead moved 5434 of the 32768 neutral LabS L codes by a count
/// (issue #581).
///
/// The result is ALREADY QUANTISED, so this is the code that reaches the
/// output buffer; `write_sample` rounds an integer and changes nothing.
///
/// The chord is fused, the same way [`calcul_tables`] fuses its
/// multiply-add, but it is deliberately NOT spelled as an
/// `f32::mul_add`. `fma` is not in the x86-64 baseline, so rustc lowers
/// `f32::mul_add` to a libm `fmaf` CALL there, and this runs once per
/// channel per pixel on the route every `srgb`, `rgb16`, `b-w`,
/// `grey16` and `hsv` conversion takes. Measured on rustc 1.98 at
/// `-C opt-level=3`: for `x86_64-unknown-linux-musl` the `mul_add`
/// spelling is `jmpq *fmaf@GOTPCREL(%rip)` and the `f64` one is
/// `cvtss2sd` / `mulsd` / `addsd` / `cvtsd2ss`; add
/// `-C target-feature=+fma` and the `mul_add` collapses to a single
/// `vfmadd213ss`, which is exactly the point: the baseline does not
/// have it. On aarch64 both stay call-free (`fmadd` against `fmul` and
/// `fadd`), so the `f64` spelling costs an instruction or two there and
/// saves a whole libm call on any target without a baseline `fma`.
///
/// The two are bit-identical rather than merely close, because the
/// exact product-sum fits in an `f64` for every reachable input, so the
/// one `as f32` IS the one rounding `fmaf` would do. The
/// `f64_chord_matches_mul_add_*` tests check that rather than taking
/// the argument on trust: two of them sample both LUTs at every
/// structurally interesting point on each `cargo test`, and two more,
/// `#[ignore]`d because they walk over a billion `f32` patterns apiece,
/// sweep the whole reachable domain under
/// `cargo test --release --lib -- --ignored f64_chord_matches_mul_add`.
///
/// [`calcul_tables`] keeps its `f32::mul_add`, which is why the two
/// sites are spelled differently: it runs `range` times behind a
/// `OnceLock`, so a libm call there costs nothing, and it is pinned
/// against the shipped dylib's `fmadd` rather than against anything of
/// ours. Do not unify them.
fn scrgb_to_code(range: usize, value: f64) -> f64 {
    // "RGB can be NaN. Throw those values out, they will break our
    // clipping." (`LabQ2sRGB.c:301-310`, `:404-409`.) vips answers 0
    // rather than clipping. A NaN only reaches here from a NaN XYZ, and
    // the scRGB matrix spreads that to all three channels, so answering
    // per channel lands where the C's per-pixel test does.
    if value.is_nan() {
        return 0.0;
    }
    let lut = y2v_table(range);
    let maxval = (range - 1) as f32;
    let yf = (value as f32 * maxval).clamp(0.0, maxval);
    let yi = yf as usize;
    let lo = lut[yi];
    // The `+ 1` is in bounds because `calcul_tables` duplicates the last
    // entry for exactly this read.
    let delta = (lut[yi + 1] - lo) as f32;
    let t = yf - yi as f32;
    // Fused, like the `fmadd s0, s5, s0, s4` the build compiles
    // `lut[Yi] + (lut[Yi + 1] - lut[Yi]) * (Yf - Yi)` into, but reached
    // through `f64` instead of `f32::mul_add` so that x86-64 does not
    // pay a libm `fmaf` call per channel per pixel. Bit-identical; see
    // the doc above.
    let fused = (f64::from(delta) * f64::from(t) + f64::from(lo)) as f32;
    f64::from(fused.round_ties_even())
}

/// Linear scRGB (0..1) -> D65 XYZ (`Y` white = 100), the sRGB primaries
/// matrix at the 4-decimal precision libvips uses.
fn scrgb_to_xyz(rgb: [f64; 3]) -> [f64; 3] {
    let r = rgb[0] * 100.0;
    let g = rgb[1] * 100.0;
    let b = rgb[2] * 100.0;
    [
        0.4124 * r + 0.3576 * g + 0.1805 * b,
        0.2126 * r + 0.7152 * g + 0.0722 * b,
        0.0193 * r + 0.1192 * g + 0.9505 * b,
    ]
}

/// D65 XYZ (`Y` white = 100) -> linear scRGB, the 6-decimal inverse
/// matrix libvips uses.
fn xyz_to_scrgb(xyz: [f64; 3]) -> [f64; 3] {
    let x = xyz[0] / 100.0;
    let y = xyz[1] / 100.0;
    let z = xyz[2] / 100.0;
    [
        3.240625 * x - 1.537208 * y - 0.498629 * z,
        -0.968931 * x + 1.875756 * y + 0.041518 * z,
        0.055710 * x - 0.204021 * y + 1.056996 * z,
    ]
}

/// Hue angle of `(a, b)` in degrees, wrapped to `[0, 360]` (libvips
/// `vips_col_ab2h`, `colour/Lab2LCh.c:61-89`).
///
/// The explicit `a == 0.0` arm is the C ladder's own, not a shortcut:
/// `a == 0` is true for `-0.0` in C, so vips answers 0 / 90 / 270 on the
/// whole `a` axis, while `atan2(±0.0, -0.0)` is `±PI` and would answer
/// 180. Measured on the binary, `oklab [0.5, -0.0, 0.0] -> oklch` is
/// `0.5 0 0`, and `[0.5, -0.0, ±0.1]` gives 90 / 270. Off that axis the
/// `atan2` form and the C's `atan(b / a)` plus a quadrant offset agree,
/// which `hue_matches_vips_col_ab2h_ladder` pins rather than assumes.
///
/// The upper bound really is closed. For a positive `a` and a `b` small
/// enough that `deg(atan2(b, a))` is a tiny negative, `h + 360.0` rounds
/// to exactly `360.0`: `oklab [0.5, 0.1, -1e-30] -> oklch` gives a hue of
/// `360` here and `360` in vips 8.18.4, which lands on the same edge
/// through `VIPS_DEG(t + VIPS_PI * 2.0)`. Clamping to `[0, 360)` would
/// buy a tidier range by diverging from the C, so the range is documented
/// instead.
///
/// One divergence from the C is deliberate: a non-finite `(a, b)` follows
/// IEEE `atan2` here, so `(inf, inf)` is 45 degrees, where the C's
/// `b / a` is NaN and propagates through `atan` to the output. See
/// [`lab_to_lch`] for the matching chroma divergence.
fn ab_to_h(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        // Matches `if (a == 0)` in the C, which `-0.0` also enters.
        if b < 0.0 {
            270.0
        } else if b == 0.0 {
            0.0
        } else {
            90.0
        }
    } else {
        let h = b.atan2(a).to_degrees();
        if h < 0.0 { h + 360.0 } else { h }
    }
}

/// Lab-like cartesian to polar (libvips `vips_Lab2LCh_line`,
/// `colour/Lab2LCh.c:114`, and `vips_Oklab2Oklch_line`,
/// `colour/Oklab2Oklch.c:64`, which are the same two lines of arithmetic
/// over `float`).
///
/// The chroma uses [`f64::hypot`] where the C squares and adds,
/// `sqrtf(a * a + b * b)`. That is a deliberate divergence in this
/// crate's favour: `a * a` overflows an `f32` to infinity once `a` passes
/// about 1.8e19, the square root of `f32::MAX`, and `hypot` has no such
/// intermediate to overflow.
fn lab_to_lch(lab: [f64; 3]) -> [f64; 3] {
    [lab[0], lab[1].hypot(lab[2]), ab_to_h(lab[1], lab[2])]
}

fn lch_to_lab(lch: [f64; 3]) -> [f64; 3] {
    let r = deg_to_rad(lch[2]);
    [lch[0], lch[1] * r.cos(), lch[1] * r.sin()]
}

// --- CMC uniform colour space (Colour Measurement Committee) ---

/// CMC lightness from CIE `L` (published CMC uniform-space function).
fn l_to_lcmc(l: f64) -> f64 {
    if l < 16.0 {
        1.744 * l
    } else {
        21.75 * l.ln() + 0.3838 * l - 38.54
    }
}

/// CMC chroma from CIE `C`.
fn c_to_ccmc(c: f64) -> f64 {
    (0.162 * c + 10.92 * (0.638 + 0.07216 * c).ln() + 4.907).max(0.0)
}

/// CMC hue from CIE `C` and `h` (degrees). The piecewise `k` constants
/// are the published CMC hue-correction table.
fn ch_to_hcmc(c: f64, h: f64) -> f64 {
    let (k4, k5, k6, k7, k8) = if h < 49.1 {
        (133.87, -134.5, -0.924, 1.727, 340.0)
    } else if h < 110.1 {
        (11.78, -12.7, -0.218, 2.12, 333.0)
    } else if h < 269.6 {
        (13.87, 10.93, 0.14, 1.0, -83.0)
    } else {
        (0.14, 5.23, 0.17, 1.61, 233.0)
    };

    let p = deg_to_rad(k7 * h + k8).cos();
    let d = k4 + k5 * p * p.abs().powf(k6);
    let g = c * c * c * c;
    let f = (g / (g + 1900.0)).sqrt();
    h + d * f
}

/// Invert a monotonically increasing function by bisection over
/// `[lo, hi]`. Targets outside the range clamp to its ends. 64 iterations
/// take the answer to the limit of f64 on these scales.
fn invert_increasing(f: impl Fn(f64) -> f64, lo: f64, hi: f64, target: f64) -> f64 {
    if target <= f(lo) {
        return lo;
    }
    if target >= f(hi) {
        return hi;
    }
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn lcmc_to_l(lcmc: f64) -> f64 {
    invert_increasing(l_to_lcmc, 0.0, 200.0, lcmc)
}

fn ccmc_to_c(ccmc: f64) -> f64 {
    invert_increasing(c_to_ccmc, 0.0, 500.0, ccmc)
}

/// Invert the CMC hue correction for a known `C`: find `h` with
/// `ch_to_hcmc(c, h) = hcmc`. The forward function is `h` plus a bounded
/// wobble, so each whole-degree segment is scanned for a bracket (the
/// same segment structure the libvips inverse tables assume) and the
/// bracket is bisected. Wrap-around targets are tried at `-360`, `0`, and
/// `+360`.
fn hcmc_to_h(c: f64, hcmc: f64) -> f64 {
    for offset in [0.0, 360.0, -360.0] {
        let target = hcmc + offset;
        for k in 0..360 {
            let a = ch_to_hcmc(c, k as f64);
            let b = ch_to_hcmc(c, (k + 1) as f64);
            let (seg_lo, seg_hi) = if a <= b { (a, b) } else { (b, a) };
            if target >= seg_lo && target <= seg_hi {
                let h = if a <= b {
                    invert_increasing(|h| ch_to_hcmc(c, h), k as f64, (k + 1) as f64, target)
                } else {
                    // Locally decreasing segment: invert the negated
                    // function.
                    invert_increasing(|h| -ch_to_hcmc(c, h), k as f64, (k + 1) as f64, -target)
                };
                return h.rem_euclid(360.0);
            }
        }
    }
    // No bracket (numerically impossible for reachable inputs): fall back
    // to the identity, which is exact for neutral colours.
    hcmc.rem_euclid(360.0)
}

/// LCh -> the CMC uniform space, the whole of `vips_LCh2CMC_line`
/// (`colour/LCh2UCS.c:200-216`).
///
/// This is the only place the CMC encode lives, so the XYZ hub arm of
/// [`from_xyz_into`] and the `{ LABS, CMC }` direct edge cannot drift
/// apart.
fn lch_to_cmc(lch: [f64; 3]) -> [f64; 3] {
    [
        l_to_lcmc(lch[0]),
        c_to_ccmc(lch[1]),
        ch_to_hcmc(lch[1], lch[2]),
    ]
}

/// The CMC uniform space -> LCh, the whole of `vips_CMC2LCh_line`
/// (`colour/UCS2LCh.c:238-254`), and the only place the CMC decode
/// lives.
///
/// libvips inverts the three CMC functions through interpolation tables
/// sampled every 0.1 (`UCS2LCh.c:66-135`); this module bisects the
/// forward function instead, which is the more accurate of the two and
/// the one divergence from the binary that survives the direct edges.
fn cmc_to_lch(cmc: [f64; 3]) -> [f64; 3] {
    let c = ccmc_to_c(cmc[1]);
    [lcmc_to_l(cmc[0]), c, hcmc_to_h(c, cmc[2])]
}

// --- Oklab (Ottosson's published matrices, as used by libvips) ---

fn xyz_to_oklab(xyz: [f64; 3]) -> [f64; 3] {
    let x = xyz[0] / 100.0;
    let y = xyz[1] / 100.0;
    let z = xyz[2] / 100.0;

    let l = 0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z;
    let m = 0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z;
    let s = 0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z;

    let lp = l.cbrt();
    let mp = m.cbrt();
    let sp = s.cbrt();

    [
        0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
        1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
        0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp,
    ]
}

fn oklab_to_xyz(lab: [f64; 3]) -> [f64; 3] {
    let (l0, a, b) = (lab[0], lab[1], lab[2]);

    let lp = l0 + 0.39633779 * a + 0.21580376 * b;
    let mp = 1.00000001 * l0 - 0.10556134 * a - 0.06385417 * b;
    let sp = 1.00000005 * l0 - 0.08948418 * a - 1.29148554 * b;

    let l = lp * lp * lp;
    let m = mp * mp * mp;
    let s = sp * sp * sp;

    [
        (1.22701385 * l - 0.55779998 * m + 0.28125615 * s) * 100.0,
        (-0.04058018 * l + 1.11225687 * m - 0.07167668 * s) * 100.0,
        (-0.07638128 * l - 0.42148198 * m + 1.58616322 * s) * 100.0,
    ]
}

// --- HSV over 8-bit sRGB (libvips codes hue as 0..255, 42.5 per sextant) ---

/// 8-bit sRGB -> HSV, hue in libvips' 0..255 coding. Inputs are the
/// integral 0..255 sample values; outputs are unrounded and quantise on
/// write.
fn srgb8_to_hsv(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    let (c_max, c_min, secondary_diff, wrap_around_hue) = if g < b {
        if b < r {
            // Centre red, at the top of the hue circle.
            (r, g, g - b, 255.0)
        } else {
            // Centre blue.
            (b, r.min(g), r - g, 170.0)
        }
    } else if g < r {
        // Centre red, at the bottom of the hue circle.
        (r, b, g - b, 0.0)
    } else {
        // Centre green.
        (g, r.min(b), b - r, 85.0)
    };

    if c_max == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let delta = c_max - c_min;
    let h = if delta == 0.0 {
        0.0
    } else {
        // `secondary_diff` is a `float` and `delta` an `unsigned char`
        // cast to one, so the RATIO is an f32 division; only the
        // `42.5 *` promotes back to double (`sRGB2HSV.c:113-114`).
        42.5 * f64::from(secondary_diff as f32 / delta as f32) + wrap_around_hue
    };
    // `q` is `unsigned char`, so the C TRUNCATES both codes on the store
    // (`sRGB2HSV.c:113-117`); it does not round them. Both are
    // non-negative here -- the hue arms pair a negative `secondary_diff`
    // with a `wrap_around_hue` that more than covers it -- so truncating
    // toward zero and flooring agree.
    [h.trunc(), (delta * 255.0 / c_max).trunc(), c_max]
}

/// HSV (libvips 0..255 hue coding) -> 8-bit sRGB. Inputs are the integral
/// 0..255 sample values.
fn hsv_to_srgb8(hsv: [f64; 3]) -> [f64; 3] {
    const SIXTH: f64 = 42.5;
    let [h, s, v] = hsv;
    let c = v * s / 255.0;
    let x = c * (1.0 - ((h / SIXTH) % 2.0 - 1.0).abs());
    let m = v - c;

    if h < SIXTH.floor() {
        [c + m, x + m, m]
    } else if h < (2.0 * SIXTH).floor() {
        [x + m, c + m, m]
    } else if h < (3.0 * SIXTH).floor() {
        [m, c + m, x + m]
    } else if h < (4.0 * SIXTH).floor() {
        [m, x + m, c + m]
    } else if h < (5.0 * SIXTH).floor() {
        [x + m, m, c + m]
    } else {
        [c + m, m, x + m]
    }
}

// --- CMYK (the libvips no-lcms approximation) ---

/// XYZ -> CMYK ink values 0..255 (the libvips no-lcms naive ink model:
/// D65-normalised XYZ treated as RGB reflectance).
fn xyz_to_cmyk(xyz: [f64; 3]) -> [f64; 4] {
    const EPSILON: f64 = 0.00001;
    let r = xyz[0] / D65[0];
    let g = xyz[1] / D65[1];
    let b = xyz[2] / D65[2];
    let c = 1.0 - r;
    let m = 1.0 - g;
    let y = 1.0 - b;
    let k = c.min(m).min(y);
    let ik = 1.0 - k;

    if ik < EPSILON {
        [255.0, 255.0, 255.0, 255.0]
    } else {
        [
            (255.0 * (c - k) / ik).clamp(0.0, 255.0),
            (255.0 * (m - k) / ik).clamp(0.0, 255.0),
            (255.0 * (y - k) / ik).clamp(0.0, 255.0),
            (255.0 * k).clamp(0.0, 255.0),
        ]
    }
}

/// CMYK ink values 0..255 -> XYZ (inverse of the naive ink model).
fn cmyk_to_xyz(cmyk: [f64; 4]) -> [f64; 3] {
    let c = cmyk[0] / 255.0;
    let m = cmyk[1] / 255.0;
    let y = cmyk[2] / 255.0;
    let k = cmyk[3] / 255.0;
    let r = 1.0 - (c * (1.0 - k) + k);
    let g = 1.0 - (m * (1.0 - k) + k);
    let b = 1.0 - (y * (1.0 - k) + k);
    [D65[0] * r, D65[1] * g, D65[2] * b]
}

// --- Yxy ---

fn xyz_to_yxy(xyz: [f64; 3]) -> [f64; 3] {
    let total = xyz[0] + xyz[1] + xyz[2];
    if total == 0.0 {
        [xyz[1], 0.0, 0.0]
    } else {
        [xyz[1], xyz[0] / total, xyz[1] / total]
    }
}

fn yxy_to_xyz(yxy: [f64; 3]) -> [f64; 3] {
    let [y_lum, x, y] = yxy;
    if x == 0.0 || y == 0.0 {
        [0.0, y_lum, 0.0]
    } else {
        let total = y_lum / y;
        let big_x = x * total;
        let big_z = (big_x - x * big_x - x * y_lum) / x;
        [big_x, y_lum, big_z]
    }
}

// --- LabS code scaling ---

const LABS_L_SCALE: f64 = 32767.0 / 100.0;
const LABS_AB_SCALE: f64 = 32768.0 / 128.0;

/// Lab -> the LabS code triple, the whole of `vips_Lab2LabS_line`
/// (`colour/Lab2LabS.c:64-68`) including the store.
///
/// The C clips in `double` with `VIPS_CLIP(0, .., SHRT_MAX)` on `L` and
/// `VIPS_CLIP(SHRT_MIN, .., SHRT_MAX)` on `a`/`b`, then assigns the
/// result into a `signed short`, which drops the fraction **toward
/// zero**. That last step is the whole quantiser, so it lives here
/// rather than at the call sites: `Lab [50, 0, 0]` scales to exactly
/// `16383.5`, and vips 8.18.4 answers `16383`, not `16384`.
///
/// Truncating is not flooring. LabS is the only signed carrier this
/// module quantises into, so the two differ on negative `a`/`b`, and the
/// binary picks truncation: `a = +/-0.501953125` scales to `+/-128.5`
/// and comes back `+/-128`, where flooring would give `128 / -129` and
/// rounding `129 / -129`.
///
/// The input is rounded to `f32` first, which is the other half of that
/// same line: `Lab2LabS.c:59` declares `float *restrict p`, and every
/// libvips route ending in LabS hands it a float Lab image, so the
/// quantiser never sees more than single precision. That is invisible
/// under rounding and decides whole counts under truncation.
/// `LCh [0, 1, 30]` is the case that shows it: `sin(30 deg)` is
/// 0.49999999999999994 in `f64` and exactly 0.5 as `f32`, so `b * 256`
/// is 127.99999999999999 or 128.0, and vips answers 128. Feeding this a
/// value that was already an `f32` sample, which is what the `Lab` and
/// `Labs` rasters carry, leaves it unchanged.
fn lab_to_labs(lab: [f64; 3]) -> [f64; 3] {
    let lab = lab.map(|v| v as f32 as f64);
    [
        (lab[0] * LABS_L_SCALE).clamp(0.0, 32767.0).trunc(),
        (lab[1] * LABS_AB_SCALE).clamp(-32768.0, 32767.0).trunc(),
        (lab[2] * LABS_AB_SCALE).clamp(-32768.0, 32767.0).trunc(),
    ]
}

/// LabS -> Lab, the plain division `vips_LabS2Lab_line` does
/// (`colour/LabS2Lab.c:57-59`). No quantiser on this side: the target is
/// float.
fn labs_to_lab(labs: [f64; 3]) -> [f64; 3] {
    [
        labs[0] / LABS_L_SCALE,
        labs[1] / LABS_AB_SCALE,
        labs[2] / LABS_AB_SCALE,
    ]
}

/// LCh -> the LabS codes, `{ LCH, LABS, { vips_LCh2Lab, vips_Lab2LabS } }`
/// (`colourspace.c:280`).
fn lch_to_labs(lch: [f64; 3]) -> [f64; 3] {
    lab_to_labs(lch_to_lab(lch))
}

/// LabS -> LCh, `{ LABS, LCH, { vips_LabS2Lab, vips_Lab2LCh } }`
/// (`colourspace.c:312`).
fn labs_to_lch(labs: [f64; 3]) -> [f64; 3] {
    lab_to_lch(labs_to_lab(labs))
}

/// CMC -> the LabS codes, `{ CMC, LABS, { vips_CMC2LCh, vips_LCh2Lab,
/// vips_Lab2LabS } }` (`colourspace.c:297`).
fn cmc_to_labs(cmc: [f64; 3]) -> [f64; 3] {
    lab_to_labs(lch_to_lab(cmc_to_lch(cmc)))
}

/// LabS -> CMC, `{ LABS, CMC, { vips_LabS2Lab, vips_Lab2LCh,
/// vips_LCh2CMC } }` (`colourspace.c:313`).
fn labs_to_cmc(labs: [f64; 3]) -> [f64; 3] {
    lch_to_cmc(lab_to_lch(labs_to_lab(labs)))
}

// --- Mono (CIE linear luminance, libvips scRGB2BW) ---

/// scRGB -> the CIE linear luminance `vips_col_scRGB2BW` takes before
/// the sRGB encode (`colour/LabQ2sRGB.c:400`).
///
/// Nothing is clamped here on purpose: the C clips the SCALED index
/// inside the lookup, not the luminance, so the clip belongs to
/// [`scrgb_to_code`].
fn scrgb_luminance(rgb: [f64; 3]) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

// ---------------------------------------------------------------------------
// Colourspace routing
// ---------------------------------------------------------------------------

/// Storage depth of a colour space's canonical raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpaceDepth {
    U8,
    U16,
    F32,
}

impl SpaceDepth {
    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    fn max_value(self) -> f64 {
        match self {
            Self::U8 => 255.0,
            Self::U16 => 65535.0,
            Self::F32 => f64::INFINITY,
        }
    }
}

/// Number of colour bands a space occupies; bands past these are extra
/// bands and carried through unchanged.
fn space_bands(space: Interpretation) -> usize {
    match space {
        Interpretation::Cmyk => 4,
        Interpretation::Bw | Interpretation::Grey16 => 1,
        _ => 3,
    }
}

/// Canonical storage depth of a space.
fn space_depth(space: Interpretation) -> SpaceDepth {
    match space {
        Interpretation::Srgb | Interpretation::Hsv | Interpretation::Cmyk | Interpretation::Bw => {
            SpaceDepth::U8
        }
        Interpretation::Rgb16 | Interpretation::Grey16 => SpaceDepth::U16,
        _ => SpaceDepth::F32,
    }
}

/// Whether a space has a colourspace route.
fn space_supported(space: Interpretation) -> bool {
    matches!(
        space,
        Interpretation::Xyz
            | Interpretation::Lab
            | Interpretation::Lch
            | Interpretation::Cmc
            | Interpretation::Labs
            | Interpretation::ScRgb
            | Interpretation::Hsv
            | Interpretation::Srgb
            | Interpretation::Yxy
            | Interpretation::OkLab
            | Interpretation::OkLch
            | Interpretation::Bw
            | Interpretation::Grey16
            | Interpretation::Rgb16
            | Interpretation::Cmyk
    )
}

/// Source-side aliases, mirroring `vips_colourspace`: plain RGB is
/// treated as sRGB, matrices as mono.
fn alias_source(space: Interpretation) -> Interpretation {
    match space {
        Interpretation::Rgb => Interpretation::Srgb,
        Interpretation::Matrix => Interpretation::Bw,
        other => other,
    }
}

/// The direct edge, if any, for a same-family pair libvips joins with a
/// single transform.
///
/// Every other pair in the route table meets at the XYZ hub, but libvips
/// joins each Lab-like space to its polar form with one transform and
/// nothing else in the pipeline: `{ LAB, LCH, { vips_Lab2LCh } }`
/// (`colour/colourspace.c:244`), `{ LCH, LAB, { vips_LCh2Lab } }`
/// (:276), `{ OKLAB, OKLCH, { vips_Oklab2Oklch } }` (:478) and
/// `{ OKLCH, OKLAB, { vips_Oklch2Oklab } }` (:494). It joins Lab to its
/// signed-16-bit coding the same way: `{ LAB, LABS, { vips_Lab2LabS } }`
/// (:246) and `{ LABS, LAB, { vips_LabS2Lab } }` (:310). And it reaches
/// that coding from the two other Lab-family spaces without an XYZ step
/// either, through Lab: `{ LCH, LABS, { vips_LCh2Lab, vips_Lab2LabS } }`
/// (:280), `{ LABS, LCH, { vips_LabS2Lab, vips_Lab2LCh } }` (:312),
/// `{ CMC, LABS, { vips_CMC2LCh, vips_LCh2Lab, vips_Lab2LabS } }` (:297)
/// and `{ LABS, CMC, { vips_LabS2Lab, vips_Lab2LCh, vips_LCh2CMC } }`
/// (:313). A multi-stage pipeline is still a direct edge here as long as
/// XYZ is not one of the stages: what costs accuracy is the round trip,
/// not the number of steps.
///
/// Sending any of those through the hub inserts a round trip libvips
/// never runs, and on every one of them the two halves of that round
/// trip fail to invert each other, so this is accuracy and not only time.
///
/// For Oklab the culprit is the matrix: the published inverse is an
/// 8-decimal approximation (the `1.00000001` / `1.00000005` quirk
/// digits), so `Oklab -> XYZ -> Oklab` pushes a neutral colour's `a` and
/// `b` off zero by about 2e-9, and the hue read off them comes out of
/// nowhere: 94.489 degrees for OkLab `[0.5, 0, 0]`, where vips 8.18.4
/// returns 0.
///
/// For Lab the culprit is the shadow branch: `lab_f` switches at
/// `t < 0.008856` while `lab_to_xyz` switches at `L < 8.0`, and those
/// rounded decimal constants are not mutual inverses. Under `L = 8` the
/// same neutral-hue garbage appears about 3e5 times larger in raw units:
/// `Lab [5, 0, 0]` comes back from the hub as
/// `(4.99996, 5.172e-4, -2.069e-4)`, i.e. LCh
/// `(4.99996, C = 5.571e-4, h = 338.199)`, where vips returns `5 0 0`.
/// The residue only reaches exactly zero somewhere above `L = 10` (which
/// still yields a 1.4e-14 chroma carrying the same 338.199 hue), so the
/// `L = 50` neutrals a test naturally reaches for hide it completely.
///
/// For LabS the same residue is what a truncating quantiser cannot
/// survive. `lab_to_labs` drops the fraction toward zero the way
/// `Lab2LabS.c:66` does, so a hub residue of `-1e-6` on a code that
/// should land on a whole number costs a whole count. On a grid of Lab
/// values with `a`/`b` at multiples of `1/256` the hub misses the direct
/// answer on 3420 channels, always by exactly one and always at an
/// integer code: `Lab [0, -128, 1]` is `[0, -32768, 256]` in vips and
/// `[0, -32767, 255]` through the hub. Rounding used to absorb that,
/// which is why the routing looked cosmetic until the quantiser was
/// right. Coming back, `LabS [983, 256, -256]` is `[2.999969482421875,
/// 1, -1]` in vips and `[2.99994, 1.00052, -1.00021]` through the hub.
///
/// The `Lch` and `Cmc` edges inherit both defects at once, because their
/// pipelines end in the same truncating store and start from the same
/// shadow branch. On a 700-pixel LCh sweep (`L` in
/// {0, 1, 3, 5, 8, 10, 20, 50, 80, 100}, `C` in
/// {0, 1, 25, 50, 100, 127, 128}, `h` in {0, 30, 45, 90, 135, 180, 225,
/// 270, 315, 359}) the hub missed vips on 181 of the 2100 `LCh -> LabS`
/// channels, and on a 700-pixel LabS sweep it missed on 748 of the 2100
/// `LabS -> LCh` channels and 681 of the `LabS -> CMC` ones. All three
/// come back exact on the direct edges. The `LabS -> LCh` and
/// `LabS -> CMC` numbers are large because a neutral LabS code is
/// *exactly* neutral, so vips answers `C = 0, h = 0` and the hub reads a
/// hue off its own noise: 338.199 degrees, at every `L`.
///
/// Taking the edge keeps the polar pairs a pure polar swap and the LabS
/// pair a pure scale.
///
/// Every entry owns its **complete** target-side production, quantiser
/// included, because this table never reaches `from_xyz_into`. That is
/// why the `signed short` truncation lives inside `lab_to_labs` rather
/// than at the `Labs` arm there, and why the CMC encode lives inside
/// `lch_to_cmc` rather than at the `Cmc` arm: a route added here cannot
/// lose either of them.
///
/// Cross-family polar pairs are deliberately absent: libvips routes
/// `{ OKLCH, LCH }` (:483) through XYZ like everything else, and so does
/// this table by returning `None` for it.
///
/// What is here is still not everything libvips joins directly.
/// `{ LAB, CMC }` (:245), `{ CMC, LAB }` (:293), `{ LCH, CMC }` (:279)
/// and `{ CMC, LCH }` (:295) are hub-free there as well, and all four
/// spaces are supported here, so those still pay the hub round trip; the
/// module docs list the rest of the hub-free edges this port has not
/// taken.
fn direct_edge(src: Interpretation, target: Interpretation) -> Option<fn([f64; 3]) -> [f64; 3]> {
    use Interpretation::{Cmc, Lab, Labs, Lch, OkLab, OkLch};
    match (src, target) {
        (Lab, Lch) | (OkLab, OkLch) => Some(lab_to_lch),
        (Lch, Lab) | (OkLch, OkLab) => Some(lch_to_lab),
        (Lab, Labs) => Some(lab_to_labs),
        (Labs, Lab) => Some(labs_to_lab),
        (Lch, Labs) => Some(lch_to_labs),
        (Labs, Lch) => Some(labs_to_lch),
        (Cmc, Labs) => Some(cmc_to_labs),
        (Labs, Cmc) => Some(labs_to_cmc),
        _ => None,
    }
}

/// Convert one pixel's colour bands from `space` to D65 XYZ. `v` holds
/// `space_bands(space)` samples in the space's numeric convention.
fn to_xyz(space: Interpretation, v: &[f64]) -> [f64; 3] {
    match space {
        Interpretation::Xyz => [v[0], v[1], v[2]],
        Interpretation::Lab => lab_to_xyz([v[0], v[1], v[2]], D65),
        Interpretation::Lch => lab_to_xyz(lch_to_lab([v[0], v[1], v[2]]), D65),
        Interpretation::Cmc => lab_to_xyz(lch_to_lab(cmc_to_lch([v[0], v[1], v[2]])), D65),
        Interpretation::Labs => lab_to_xyz(labs_to_lab([v[0], v[1], v[2]]), D65),
        Interpretation::ScRgb => scrgb_to_xyz([v[0], v[1], v[2]]),
        Interpretation::Hsv => {
            let rgb = hsv_to_srgb8([v[0], v[1], v[2]]);
            // The libvips HSV decode goes through 8-bit sRGB.
            let rgb = rgb.map(|c| srgb_decode(c.round().clamp(0.0, 255.0) / 255.0));
            scrgb_to_xyz(rgb)
        }
        Interpretation::Srgb => scrgb_to_xyz([v[0], v[1], v[2]].map(|c| srgb_decode(c / 255.0))),
        Interpretation::Rgb16 => scrgb_to_xyz([v[0], v[1], v[2]].map(|c| srgb_decode(c / 65535.0))),
        Interpretation::Yxy => yxy_to_xyz([v[0], v[1], v[2]]),
        Interpretation::OkLab => oklab_to_xyz([v[0], v[1], v[2]]),
        Interpretation::OkLch => oklab_to_xyz(lch_to_lab([v[0], v[1], v[2]])),
        Interpretation::Bw => {
            let g = srgb_decode(v[0] / 255.0);
            scrgb_to_xyz([g, g, g])
        }
        Interpretation::Grey16 => {
            let g = srgb_decode(v[0] / 65535.0);
            scrgb_to_xyz([g, g, g])
        }
        Interpretation::Cmyk => cmyk_to_xyz([v[0], v[1], v[2], v[3]]),
        // Callers check `space_supported` first.
        other => unreachable!("no colourspace route for {other:?}"),
    }
}

/// Convert one pixel from D65 XYZ to `space`, writing the
/// `space_bands(space)` output samples into the front of `out`, in the
/// space's numeric convention.
///
/// Most arms leave the sample unrounded and let the writer quantise, but
/// the arms whose C counterpart quantises INSIDE the transform do it
/// here instead, so the two cannot drift: `labs` truncates into the
/// `signed short` (`Lab2LabS.c:66-68`), `hsv` truncates into the
/// `unsigned char` (`sRGB2HSV.c:113-117`), and `srgb`, `rgb16`, `b-w`,
/// `grey16` and the sRGB step of `hsv` come back already rounded out of
/// the `Y2v` lookup ([`scrgb_to_code`]). Re-rounding an integer on write
/// changes nothing, so those arms pass through it untouched.
///
/// This is the allocation-free form the per-pixel conversion loop drives:
/// the caller supplies one scratch array (`[f64; 4]` covers every space,
/// CMYK being the widest at four bands) and reuses it across the whole
/// image, so the hot path allocates nothing per pixel. `out` must hold at
/// least `space_bands(space)` slots; only that prefix is written.
fn from_xyz_into(space: Interpretation, xyz: [f64; 3], out: &mut [f64]) {
    match space {
        Interpretation::Xyz => out[..3].copy_from_slice(&xyz),
        Interpretation::Lab => out[..3].copy_from_slice(&xyz_to_lab(xyz, D65)),
        Interpretation::Lch => out[..3].copy_from_slice(&lab_to_lch(xyz_to_lab(xyz, D65))),
        // `lch_to_cmc` carries the CMC encode, so this arm and the
        // `{ LABS, CMC }` direct edge quantise through one function.
        Interpretation::Cmc => {
            out[..3].copy_from_slice(&lch_to_cmc(lab_to_lch(xyz_to_lab(xyz, D65))));
        }
        // `lab_to_labs` carries the `signed short` truncation, so this
        // arm and the direct edge quantise through one function.
        Interpretation::Labs => out[..3].copy_from_slice(&lab_to_labs(xyz_to_lab(xyz, D65))),
        Interpretation::ScRgb => out[..3].copy_from_slice(&xyz_to_scrgb(xyz)),
        Interpretation::Hsv => {
            // The libvips HSV encode goes through 8-bit sRGB
            // (`colourspace.c:336` and the rest of the `{ *, HSV }`
            // block), so it sees the LUT's codes, not an analytic
            // encode rounded afterwards.
            let rgb = xyz_to_scrgb(xyz).map(|c| scrgb_to_code(SRGB_RANGE, c));
            out[..3].copy_from_slice(&srgb8_to_hsv(rgb));
        }
        Interpretation::Srgb => {
            out[..3].copy_from_slice(&xyz_to_scrgb(xyz).map(|c| scrgb_to_code(SRGB_RANGE, c)));
        }
        Interpretation::Rgb16 => {
            out[..3].copy_from_slice(&xyz_to_scrgb(xyz).map(|c| scrgb_to_code(RGB16_RANGE, c)));
        }
        Interpretation::Yxy => out[..3].copy_from_slice(&xyz_to_yxy(xyz)),
        Interpretation::OkLab => out[..3].copy_from_slice(&xyz_to_oklab(xyz)),
        Interpretation::OkLch => out[..3].copy_from_slice(&lab_to_lch(xyz_to_oklab(xyz))),
        Interpretation::Bw => {
            out[0] = scrgb_to_code(SRGB_RANGE, scrgb_luminance(xyz_to_scrgb(xyz)));
        }
        Interpretation::Grey16 => {
            out[0] = scrgb_to_code(RGB16_RANGE, scrgb_luminance(xyz_to_scrgb(xyz)));
        }
        Interpretation::Cmyk => out[..4].copy_from_slice(&xyz_to_cmyk(xyz)),
        other => unreachable!("no colourspace route for {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Raw sample plumbing
// ---------------------------------------------------------------------------

/// Read the flat `i`-th channel sample of `raster` as `f64` (native byte
/// order, matching the crate convention).
fn read_sample_f64(raster: &Raster, i: usize) -> f64 {
    let data = raster.data();
    match raster.format().bytes_per_channel() {
        1 => data[i] as f64,
        2 => u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f64,
        _ => f32::from_ne_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]) as f64,
    }
}

/// Write the flat `i`-th channel sample into `buf` at `depth`, rounding
/// and clipping integer depths.
fn write_sample(buf: &mut [u8], depth: SpaceDepth, i: usize, v: f64) {
    match depth {
        SpaceDepth::U8 => buf[i] = v.round().clamp(0.0, 255.0) as u8,
        SpaceDepth::U16 => {
            let bytes = (v.round().clamp(0.0, 65535.0) as u16).to_ne_bytes();
            buf[i * 2] = bytes[0];
            buf[i * 2 + 1] = bytes[1];
        }
        SpaceDepth::F32 => {
            buf[i * 4..i * 4 + 4].copy_from_slice(&(v as f32).to_ne_bytes());
        }
    }
}

/// Bring a stored sample into a space's numeric convention: integer
/// spaces shift between 8- and 16-bit storage (the libvips `shift` cast)
/// and round-clip float storage (the libvips plain cast); float spaces
/// take the stored value as is.
fn normalise_sample(v: f64, storage_bpc: usize, depth: SpaceDepth) -> f64 {
    match (depth, storage_bpc) {
        (SpaceDepth::U8, 1) | (SpaceDepth::U16, 2) | (SpaceDepth::F32, _) => v,
        (SpaceDepth::U8, 2) => (v / 256.0).floor(),
        (SpaceDepth::U8, _) => v.round().clamp(0.0, 255.0),
        (SpaceDepth::U16, 1) => v * 256.0,
        (SpaceDepth::U16, _) => v.round().clamp(0.0, 65535.0),
    }
}

/// The canonical [`PixelFormat`] for `channels` at `depth`.
fn format_for(channels: usize, depth: SpaceDepth) -> PixelFormat {
    PixelFormat::with_channels(channels, depth.bytes())
        .expect("colour op output has a valid channel count")
}

/// Wrap an already-quantised sample byte buffer into a raster, carrying
/// `like`'s metadata block and attached fields, tagged `tag`. The hot
/// [`Raster::try_colourspace`] loop writes its output samples straight
/// into `buf` (via [`write_sample`]) and finishes through here, so no
/// intermediate full-image `Vec<f64>` staging is materialised.
fn raster_from_bytes(
    width: u32,
    height: u32,
    channels: usize,
    depth: SpaceDepth,
    buf: Vec<u8>,
    like: &Raster,
    tag: Interpretation,
) -> Raster {
    let mut out = Raster::new(width, height, format_for(channels, depth), buf)
        .expect("colour op output is well-formed");
    out.meta = like.meta;
    out.meta.interpretation = Some(tag);
    out.fields = like.fields.clone();
    out
}

/// Build a raster from per-channel `f64` samples at `depth`, carrying
/// `like`'s metadata block and attached fields, tagged `tag`.
fn build_raster(
    width: u32,
    height: u32,
    channels: usize,
    depth: SpaceDepth,
    samples: &[f64],
    like: &Raster,
    tag: Interpretation,
) -> Raster {
    let mut buf = vec![0u8; samples.len() * depth.bytes()];
    for (i, &v) in samples.iter().enumerate() {
        write_sample(&mut buf, depth, i, v);
    }
    raster_from_bytes(width, height, channels, depth, buf, like, tag)
}

// ---------------------------------------------------------------------------
// Colour-difference formulas
// ---------------------------------------------------------------------------

/// CIE76 colour difference: Euclidean distance in Lab.
fn de76(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Which hue-wrap convention a [`de00_impl`] evaluation follows. Every
/// other term of the CIEDE2000 kernel is identical between the two
/// colour-difference entry points, so the hue rule is the *only*
/// parameter — factoring it out keeps [`de00`] and [`de00_sharma`] on one
/// arithmetic path (libviprs#370).
#[derive(Clone, Copy)]
enum HueRule {
    /// libvips `vips_col_dE00` parity (vips-8.18.x, byte-for-byte): the
    /// `.abs()` mean-hue reflection `(h1' + h2' - 360).abs() / 2` and the
    /// `360 - (h1' - h2')` delta-hue wrap, both entered at the
    /// `|Δh'| < 180` cutoff. Pins the `vips dE00` oracle.
    VipsParity,
    /// Published Sharma 2005: the signed `(h1' + h2' ± 360) / 2` mean hue
    /// and the sign-consistent `(h1' - h2') ∓ 360` delta hue — matching
    /// the `C1' - C2'` (1->2) chroma order so the `RT · ΔC' · Δh'` cross
    /// term keeps its sign — both entered at the `|Δh'| <= 180` cutoff.
    Sharma,
}

impl HueRule {
    /// The mean hue `h̄'` and the *degrees* delta hue `Δh'` under this
    /// rule, from the two G-adjusted hues `h1'`, `h2'`. These are the only
    /// two quantities in the CIEDE2000 kernel that depend on the rule.
    fn mean_and_delta_hue(self, h1d: f64, h2d: f64) -> (f64, f64) {
        match self {
            HueRule::VipsParity => {
                // libvips mirrors the `.abs()` reflection in the mean and a
                // `360 - Δh'` wrap in the delta; see libviprs#332 for why
                // the delta sign differs from Sharma.
                let hdb = if (h1d - h2d).abs() < 180.0 {
                    (h1d + h2d) / 2.0
                } else {
                    (h1d + h2d - 360.0).abs() / 2.0
                };
                let dhd = if (h1d - h2d).abs() < 180.0 {
                    h1d - h2d
                } else {
                    360.0 - (h1d - h2d)
                };
                (hdb, dhd)
            }
            HueRule::Sharma => {
                // Published Sharma: sign-correct ±360 wraps in both the
                // mean and the delta, so the `RT · ΔC' · Δh'` cross term
                // keeps its sign on asymmetric hue-wrap pairs (e.g.
                // [50,2.5,0]/[56,-27,-3]: 31.9030, not the sign-flipped
                // 21.12 the parity delta would give here). See libviprs#332.
                let hdb = if (h1d - h2d).abs() <= 180.0 {
                    (h1d + h2d) / 2.0
                } else if h1d + h2d < 360.0 {
                    (h1d + h2d + 360.0) / 2.0
                } else {
                    (h1d + h2d - 360.0) / 2.0
                };
                let d = h1d - h2d;
                let dhd = if d.abs() <= 180.0 {
                    d
                } else if d > 180.0 {
                    d - 360.0
                } else {
                    d + 360.0
                };
                (hdb, dhd)
            }
        }
    }
}

/// The shared CIEDE2000 kernel behind [`de00`] and [`de00_sharma`].
///
/// Every term — G, C'/h', dθ, RC, RT, the four-term T polynomial,
/// SL/SC/SH, and the ΔL'/ΔC'/Δh' assembly — is identical between the two
/// entry points; they differ *only* in the hue-wrap `rule` (see
/// [`HueRule`]). Keeping one kernel means a correctness fix to any shared
/// line lands on both paths at once, which a former copy-paste pair could
/// not guarantee (libviprs#370).
fn de00_impl(lab1: [f64; 3], lab2: [f64; 3], rule: HueRule) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    // Chroma and mean chroma.
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let cb = (c1 + c2) / 2.0;

    // G.
    let cb7 = cb.powi(7);
    let g = 0.5 * (1.0 - (cb7 / (cb7 + 25.0_f64.powi(7))).sqrt());

    // L', a', b', C', h'.
    let l1d = l1;
    let a1d = (1.0 + g) * a1;
    let b1d = b1;
    let c1d = a1d.hypot(b1d);
    let h1d = ab_to_h(a1d, b1d);

    let l2d = l2;
    let a2d = (1.0 + g) * a2;
    let b2d = b2;
    let c2d = a2d.hypot(b2d);
    let h2d = ab_to_h(a2d, b2d);

    // Mean L', C', and the rule-dependent mean and (degrees) delta hue —
    // the sole point where `de00` and `de00_sharma` differ.
    let ldb = (l1d + l2d) / 2.0;
    let cdb = (c1d + c2d) / 2.0;
    let (hdb, dhd_deg) = rule.mean_and_delta_hue(h1d, h2d);

    // dtheta, RC.
    let hdbd = (hdb - 275.0) / 25.0;
    let dtheta = 30.0 * (-(hdbd * hdbd)).exp();
    let cdb7 = cdb.powi(7);
    let rc = 2.0 * (cdb7 / (cdb7 + 25.0_f64.powi(7))).sqrt();

    // RT, T.
    let rt = -deg_to_rad(2.0 * dtheta).sin() * rc;
    let t = 1.0 - 0.17 * deg_to_rad(hdb - 30.0).cos()
        + 0.24 * deg_to_rad(2.0 * hdb).cos()
        + 0.32 * deg_to_rad(3.0 * hdb + 6.0).cos()
        - 0.20 * deg_to_rad(4.0 * hdb - 63.0).cos();

    // SL, SC, SH.
    let ldb50 = ldb - 50.0;
    let sl = 1.0 + (0.015 * ldb50 * ldb50) / (20.0 + ldb50 * ldb50).sqrt();
    let sc = 1.0 + 0.045 * cdb;
    let sh = 1.0 + 0.015 * cdb * t;

    let dld = l1d - l2d;
    let dcd = c1d - c2d;
    let dhd = 2.0 * (c1d * c2d).sqrt() * deg_to_rad(dhd_deg / 2.0).sin();

    // Parametric factors are 1 for reference viewing conditions.
    let nl = dld / sl;
    let nc = dcd / sc;
    let nh = dhd / sh;

    (nl * nl + nc * nc + nh * nh + rt * nc * nh).sqrt()
}

/// CIEDE2000 colour difference, following the libvips `vips_col_dE00`
/// arrangement of the published formula (which the ported reference
/// values pin). Thin wrapper over [`de00_impl`] with the
/// [`HueRule::VipsParity`] hue rule; [`de00_sharma`] is the same kernel
/// with [`HueRule::Sharma`].
///
/// # Parity ceiling: the hue-wrap arms
///
/// Two hue branches deviate from the published Sharma 2005 formulation to
/// stay byte-for-byte with `vips_col_dE00` (vips-8.18.x):
///
/// - the **mean** hue uses the `.abs()` reflection `(h1' + h2' - 360).abs()
///   / 2` where Sharma uses the signed `(h1' + h2' ± 360) / 2`;
/// - the **delta** hue uses `360 - (h1' - h2')` in the wrap branch where
///   Sharma uses the signed `(h1' - h2') ∓ 360`;
/// - and the two rules enter their wrap branches at different cutoffs:
///   `|Δh'| < 180` here versus `|Δh'| <= 180` for Sharma.
///
/// So `de00` and [`de00_sharma`] coincide **exactly** on every
/// non-wrapping pair (`|Δh'| < 180`) and diverge only on hue-wrap pairs,
/// of which there are two kinds:
///
/// - **Asymmetric wrap** (`|Δh'| > 180`, so `h1' + h2' != 360`) — colours
///   straddling the 0/360 red boundary. This is where the deviation is
///   largest: across the published Sharma 2005 dataset (34 pairs) `de00`
///   departs from the reference dE00 by at most ~4.67 units (~1.17×), on
///   Lab `[50,2.5,0]`/`[56,-27,-3]` (hues ~0°/186°): 27.23 here vs 31.90
///   for Sharma.
/// - **The antipodal boundary** (`|Δh'| == 180` exactly) — a *symmetric*
///   `h1' + h2' == 360` pair whose colours are diametrically opposite in
///   hue. Here the `< 180` / `<= 180` cutoff split *alone* routes `de00`
///   into the mean-hue reflection while Sharma keeps the plain mean, so
///   only the mean-hue term differs (`ΔC'` and the delta-hue term are
///   identical between the arms). Example Lab `[50,0,2.49]`/`[50,0,-2.49]`,
///   whose hues sit on the yellow/blue b* axis (90°/270°, *not* the red
///   boundary): 4.746 here (and in `vips dE00` on vips-8.18.4) vs 4.804
///   for Sharma.
///
/// This is intentional libvips parity, not a bug: a Sharma "fix" here
/// would regress the pinned `vips dE00` oracle. [`de00_sharma`] applies
/// both signed Sharma arms off this parity path for callers who want the
/// published standard.
fn de00(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    de00_impl(lab1, lab2, HueRule::VipsParity)
}

/// CIEDE2000 colour difference computing the **published Sharma 2005**
/// value, off the libvips parity path used by [`de00`]. Thin wrapper over
/// [`de00_impl`] with the [`HueRule::Sharma`] hue rule.
///
/// This differs from [`de00`] in *both* hue-wrap arms (mean and delta) and
/// in the `|Δh'| <= 180` versus `|Δh'| < 180` branch cutoff. On
/// non-wrapping pairs the two agree exactly; they diverge on hue-wrap
/// pairs — most visibly on asymmetric `ΔC' != 0` pairs such as Lab
/// `[50,2.5,0]`/`[56,-27,-3]`, where this returns the published 31.9030
/// while [`de00`] returns 27.23, and also on the antipodal `|Δh'| == 180`
/// boundary where only the mean-hue term parts. See [`de00`] for the full
/// geometry.
///
/// Verified against the full published Sharma / Wu / Dalal 2005 CIEDE2000
/// test dataset (Table 1, 34 pairs): every pair reproduces the reference
/// dE00 within ~5e-5. Provided for callers who want the textbook value
/// rather than libvips parity; it is **not** wired into the pinned
/// `vips dE00` oracle (see [`de00`]).
fn de00_sharma(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    de00_impl(lab1, lab2, HueRule::Sharma)
}

/// CMC colour difference: the published CMC(l:c) formula at l = c = 1
/// (BS 6923), which the ported reference value pins. libvips itself
/// approximates dECMC as Euclidean distance in its CMC uniform space,
/// which lands ~0.7 lower on the ported reference pair; upstream's own
/// test only asserts `< 6` for exactly that reason, while the ported
/// suite asserts the published value (4.97 for the reference pair).
fn de_cmc(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let h1 = ab_to_h(a1, b1);

    let dl = l1 - l2;
    let dc = c1 - c2;
    let da = a1 - a2;
    let db = b1 - b2;
    // dH^2 = da^2 + db^2 - dC^2, clamped against rounding.
    let dh2 = (da * da + db * db - dc * dc).max(0.0);

    let sl = if l1 < 16.0 {
        0.511
    } else {
        0.040975 * l1 / (1.0 + 0.01765 * l1)
    };
    let sc = 0.0638 * c1 / (1.0 + 0.0131 * c1) + 0.638;
    let f = (c1.powi(4) / (c1.powi(4) + 1900.0)).sqrt();
    let t = if (164.0..345.0).contains(&h1) {
        0.56 + (0.2 * deg_to_rad(h1 + 168.0).cos()).abs()
    } else {
        0.36 + (0.4 * deg_to_rad(h1 + 35.0).cos()).abs()
    };
    let sh = sc * (f * t + 1.0 - f);

    let nl = dl / sl;
    let nc = dc / sc;
    (nl * nl + nc * nc + dh2 / (sh * sh)).sqrt()
}

// ---------------------------------------------------------------------------
// ICC engine
// ---------------------------------------------------------------------------

/// Rendering intent for the ICC operations (libvips `VipsIntent`).
///
/// Only LUT-based profiles carry per-intent tables; matrix-shaper
/// profiles (sRGB and friends) have a single colorimetric tag set, so
/// intent does not change their result, matching lcms behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Intent {
    /// Perceptual rendering (the libvips default).
    Perceptual,
    /// Media-relative colorimetric rendering.
    Relative,
    /// Saturation rendering.
    Saturation,
    /// ICC-absolute colorimetric rendering.
    Absolute,
}

/// Profile connection space for [`Raster::icc_import_with`] (libvips
/// `VipsPCS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pcs {
    /// CIE Lab, the default PCS.
    Lab,
    /// CIE XYZ.
    Xyz,
}

fn moxcms_intent(intent: Intent) -> RenderingIntent {
    match intent {
        Intent::Perceptual => RenderingIntent::Perceptual,
        Intent::Relative => RenderingIntent::RelativeColorimetric,
        Intent::Saturation => RenderingIntent::Saturation,
        Intent::Absolute => RenderingIntent::AbsoluteColorimetric,
    }
}

fn parse_profile(bytes: &[u8]) -> Result<ColorProfile, ColourError> {
    ColorProfile::new_from_slice(bytes).map_err(|e| ColourError::InvalidProfile {
        detail: format!("{e:?}"),
    })
}

/// Device channel count of a profile, or an error for device spaces the
/// raster model cannot carry.
fn device_channels(profile: &ColorProfile) -> Result<usize, ColourError> {
    match profile.color_space {
        DataColorSpace::Gray => Ok(1),
        DataColorSpace::Rgb => Ok(3),
        DataColorSpace::Cmyk => Ok(4),
        other => Err(ColourError::UnsupportedDeviceSpace {
            space: format!("{other:?}"),
        }),
    }
}

fn moxcms_layout(channels: usize) -> Layout {
    match channels {
        1 => Layout::Gray,
        3 => Layout::Rgb,
        // moxcms carries 4-channel CMYK ink data in the 4-channel Rgba
        // layout.
        _ => Layout::Rgba,
    }
}

fn transform_options(intent: Intent) -> TransformOptions {
    TransformOptions {
        rendering_intent: moxcms_intent(intent),
        ..Default::default()
    }
}

/// Evaluators for the exact matrix-shaper path.
struct ShaperCurves {
    curves: Vec<Box<dyn ToneCurveEvaluator + Send + Sync>>,
}

impl ShaperCurves {
    fn linear_rgb(profile: &ColorProfile) -> Option<Self> {
        Some(Self {
            curves: vec![
                profile.red_trc.as_ref()?.make_linear_evaluator().ok()?,
                profile.green_trc.as_ref()?.make_linear_evaluator().ok()?,
                profile.blue_trc.as_ref()?.make_linear_evaluator().ok()?,
            ],
        })
    }

    fn gamma_rgb(profile: &ColorProfile) -> Option<Self> {
        Some(Self {
            curves: vec![
                profile.red_trc.as_ref()?.make_gamma_evaluator().ok()?,
                profile.green_trc.as_ref()?.make_gamma_evaluator().ok()?,
                profile.blue_trc.as_ref()?.make_gamma_evaluator().ok()?,
            ],
        })
    }

    fn eval(&self, i: usize, v: f64) -> f64 {
        self.curves[i].evaluate_value(v as f32) as f64
    }
}

/// Convert normalised (0..1) device pixels to D50 PCS Lab triples.
///
/// Matrix-shaper RGB and grey-TRC profiles evaluate exactly; everything
/// else runs a moxcms float transform to its generic Lab profile and
/// decodes the ICC-encoded PCS XYZ it delivers.
fn icc_device_to_lab(
    profile: &ColorProfile,
    device: &[f32],
    channels: usize,
    intent: Intent,
) -> Result<Vec<[f64; 3]>, ColourError> {
    let pixels = device.len() / channels;

    if profile.color_space == DataColorSpace::Rgb
        && profile.is_matrix_shaper()
        && let Some(lin) = ShaperCurves::linear_rgb(profile)
    {
        let m = profile.colorant_matrix();
        let mut out = Vec::with_capacity(pixels);
        for px in device.as_chunks::<3>().0 {
            let v = m.mul_vector(Vector3d {
                v: [
                    lin.eval(0, px[0] as f64),
                    lin.eval(1, px[1] as f64),
                    lin.eval(2, px[2] as f64),
                ],
            });
            out.push(xyz_to_lab(v.v, ICC_D50));
        }
        return Ok(out);
    }

    if profile.color_space == DataColorSpace::Gray
        && let Some(trc) = profile.gray_trc.as_ref()
    {
        let lin = trc
            .make_linear_evaluator()
            .map_err(|e| ColourError::IccTransform {
                detail: format!("{e:?}"),
            })?;
        let mut out = Vec::with_capacity(pixels);
        for &px in device {
            let v = lin.evaluate_value(px) as f64;
            out.push(xyz_to_lab([ICC_D50[0] * v, v, ICC_D50[2] * v], ICC_D50));
        }
        return Ok(out);
    }

    icc_device_to_lab_fallback(profile, device, channels, intent)
}

/// The LUT-profile import path: moxcms transform to the generic Lab
/// profile, PCS XYZ decoded from the ICC `u1Fixed15` code scale.
fn icc_device_to_lab_fallback(
    profile: &ColorProfile,
    device: &[f32],
    channels: usize,
    intent: Intent,
) -> Result<Vec<[f64; 3]>, ColourError> {
    let pixels = device.len() / channels;
    let lab_profile = ColorProfile::new_lab();
    let xf = profile
        .create_transform_f32(
            moxcms_layout(channels),
            &lab_profile,
            Layout::Rgb,
            transform_options(intent),
        )
        .map_err(|e| ColourError::IccTransform {
            detail: format!("{e:?}"),
        })?;
    let mut pcs = vec![0.0f32; pixels * 3];
    xf.transform(device, &mut pcs)
        .map_err(|e| ColourError::IccTransform {
            detail: format!("{e:?}"),
        })?;
    Ok(pcs
        .as_chunks::<3>()
        .0
        .iter()
        .map(|px| {
            xyz_to_lab(
                [
                    px[0] as f64 * PCS_XYZ_SCALE,
                    px[1] as f64 * PCS_XYZ_SCALE,
                    px[2] as f64 * PCS_XYZ_SCALE,
                ],
                ICC_D50,
            )
        })
        .collect())
}

/// Convert D50 PCS Lab triples to normalised (0..1) device pixels.
fn icc_lab_to_device(
    profile: &ColorProfile,
    labs: &[[f64; 3]],
    intent: Intent,
) -> Result<Vec<f32>, ColourError> {
    if profile.color_space == DataColorSpace::Rgb
        && profile.is_matrix_shaper()
        && let Some(gamma) = ShaperCurves::gamma_rgb(profile)
    {
        let inv = profile.colorant_matrix().inverse();
        let mut out = Vec::with_capacity(labs.len() * 3);
        for lab in labs {
            let xyz = lab_to_xyz(*lab, ICC_D50);
            let v = inv.mul_vector(Vector3d { v: xyz });
            for (i, lin) in v.v.iter().enumerate() {
                out.push(gamma.eval(i, lin.clamp(0.0, 1.0)).clamp(0.0, 1.0) as f32);
            }
        }
        return Ok(out);
    }

    if profile.color_space == DataColorSpace::Gray
        && let Some(trc) = profile.gray_trc.as_ref()
    {
        let gamma = trc
            .make_gamma_evaluator()
            .map_err(|e| ColourError::IccTransform {
                detail: format!("{e:?}"),
            })?;
        let mut out = Vec::with_capacity(labs.len());
        for lab in labs {
            let y = lab_to_xyz(*lab, ICC_D50)[1];
            out.push((gamma.evaluate_value(y.clamp(0.0, 1.0) as f32)).clamp(0.0, 1.0));
        }
        return Ok(out);
    }

    icc_lab_to_device_fallback(profile, labs, intent)
}

/// The LUT-profile export path: PCS XYZ re-encoded to the ICC code scale
/// and run through a moxcms transform from the generic Lab profile.
fn icc_lab_to_device_fallback(
    profile: &ColorProfile,
    labs: &[[f64; 3]],
    intent: Intent,
) -> Result<Vec<f32>, ColourError> {
    let channels = device_channels(profile)?;
    let lab_profile = ColorProfile::new_lab();
    let xf = lab_profile
        .create_transform_f32(
            Layout::Rgb,
            profile,
            moxcms_layout(channels),
            transform_options(intent),
        )
        .map_err(|e| ColourError::IccTransform {
            detail: format!("{e:?}"),
        })?;
    let mut pcs = Vec::with_capacity(labs.len() * 3);
    for lab in labs {
        let xyz = lab_to_xyz(*lab, ICC_D50);
        pcs.push((xyz[0] / PCS_XYZ_SCALE).clamp(0.0, 1.0) as f32);
        pcs.push((xyz[1] / PCS_XYZ_SCALE).clamp(0.0, 1.0) as f32);
        pcs.push((xyz[2] / PCS_XYZ_SCALE).clamp(0.0, 1.0) as f32);
    }
    let mut device = vec![0.0f32; labs.len() * channels];
    xf.transform(&pcs, &mut device)
        .map_err(|e| ColourError::IccTransform {
            detail: format!("{e:?}"),
        })?;
    Ok(device)
}

/// Interpretation tag for a device raster of `channels` at `depth`.
fn device_tag(channels: usize, depth: SpaceDepth) -> Interpretation {
    match (channels, depth) {
        (1, SpaceDepth::U16) => Interpretation::Grey16,
        (1, _) => Interpretation::Bw,
        (4, _) => Interpretation::Cmyk,
        (_, SpaceDepth::U16) => Interpretation::Rgb16,
        _ => Interpretation::Srgb,
    }
}

/// Read a raster's first `channels` bands per pixel as normalised (0..1)
/// device samples, mirroring the libvips input casts: 8-bit / 255,
/// 16-bit / 65535, float clipped to the `0..255` device convention.
///
/// The float arm keeps the crate's `0..255` float scaling but divides
/// straight into `0..1` **without** rounding to 8 bits first, so
/// sub-8-bit precision survives into the f32-native CMS transform
/// (moxcms `create_transform_f32`, driven by `icc_device_to_lab`).
/// libvips itself
/// casts float ICC input to 8-bit before the transform; this deliberately
/// deviates from that cast to preserve precision on genuine float device
/// rasters (issue #301). The historical `v.round()` behaviour rounded the
/// `0..255` device sample to the nearest integer before dividing by 255,
/// quantising genuine float input to the 256 8-bit levels and discarding
/// the sub-integer precision callers assume survives. The 8-bit integer
/// (`/255`) and 16-bit (`/65535`) arms are unchanged and remain
/// byte-parity with libvips.
fn read_device_normalised(raster: &Raster, channels: usize) -> Vec<f32> {
    let total = raster.width() as usize * raster.height() as usize;
    let all = raster.format().channels();
    let bpc = raster.format().bytes_per_channel();
    let mut out = Vec::with_capacity(total * channels);
    for p in 0..total {
        for c in 0..channels {
            let v = read_sample_f64(raster, p * all + c);
            out.push(match bpc {
                1 => (v / 255.0) as f32,
                2 => (v / 65535.0) as f32,
                // Float device raster: preserve sub-8-bit precision for the
                // f32 CMS transform (issue #301). Do NOT re-add `.round()`
                // here — it collapses float input to 8-bit before the
                // transform and regresses precision for float callers.
                _ => (v.clamp(0.0, 255.0) / 255.0) as f32,
            });
        }
    }
    out
}

/// The profile bytes an ICC op should use: the explicit path if given,
/// else the raster's attached profile.
fn profile_bytes(raster: &Raster, path: Option<&Path>) -> Result<Vec<u8>, ColourError> {
    match path {
        Some(p) => std::fs::read(p).map_err(|source| ColourError::ProfileRead {
            path: p.to_path_buf(),
            source,
        }),
        None => raster
            .icc_profile()
            .map(<[u8]>::to_vec)
            .ok_or(ColourError::NoProfile),
    }
}

// ---------------------------------------------------------------------------
// Raster surface
// ---------------------------------------------------------------------------

impl Raster {
    /// Create a `w`x`h` image where every pixel holds `values`, tagged
    /// with `interpretation` (the libvips `black + c` fixture idiom the
    /// ported colour tests build test images with).
    ///
    /// The result is a float raster with one band per value
    /// ([`PixelFormat::RgbaF32`] for four values, `FloatF32(n)`
    /// otherwise), since the colour fixtures hold exact Lab/XYZ values.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty, if a dimension is zero, or if the
    /// band count exceeds `u16::MAX`, mirroring the "known-good input"
    /// contract of the ported-test surface.
    pub fn constant(w: u32, h: u32, values: &[f64], interpretation: Interpretation) -> Raster {
        assert!(!values.is_empty(), "constant: values must not be empty");
        let format =
            PixelFormat::with_channels(values.len(), 4).expect("constant: band count must fit u16");
        let mut raster = Raster::zeroed(w, h, format).expect("constant: valid dimensions");
        let stride = values.len();
        let count = w as usize * h as usize * stride;
        {
            let data = raster.data_mut();
            for i in 0..count {
                let bytes = (values[i % stride] as f32).to_ne_bytes();
                data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
        }
        raster.meta.interpretation = Some(interpretation);
        raster
    }

    /// Convert this image to the target colour space (libvips
    /// `vips_colourspace`), routing through D65 XYZ exactly like the
    /// libvips route table. See the [module docs](crate::colour) for the
    /// space model, quantisation points, and extra-band handling.
    ///
    /// The source space is [`Raster::interpretation`];
    /// [`Interpretation::Rgb`] sources are treated as sRGB and
    /// [`Interpretation::Matrix`] as mono, mirroring libvips.
    ///
    /// `Lab <-> Lch`, `OkLab <-> OkLCh` and `Lab <-> Labs` take the
    /// direct in-place edge libvips gives them instead of the XYZ hub;
    /// see the [module docs](crate::colour#colour-space-model).
    ///
    /// # Precision ceiling: routes through HSV are 8-bit
    ///
    /// Any conversion that passes through [`Interpretation::Hsv`] routes
    /// via **8-bit sRGB regardless of the source depth**, because both
    /// the sRGB->HSV and HSV->sRGB arms round through an 8-bit sRGB code
    /// (`hsv_to_srgb8`/`srgb8_to_hsv`), matching libvips' HSV-through-sRGB
    /// route. So a round trip such as `Rgb16 -> Hsv -> Rgb16` is **lossy**
    /// — it quantises to 8-bit precision. Avoid HSV as an intermediate
    /// space when > 8-bit precision must survive. (The sRGB<->XYZ matrix
    /// pair itself is mutually inverse to ~5e-7, i.e. ~0.06 counts at
    /// 16-bit, so the XYZ hub is effectively lossless by comparison.)
    ///
    /// # Errors
    ///
    /// [`ColourError::UnsupportedColourspace`] when the source or target
    /// has no route, and [`ColourError::TooFewBands`] when the image has
    /// fewer bands than its space needs.
    pub fn try_colourspace(&self, target: Interpretation) -> Result<Raster, ColourError> {
        let src = alias_source(self.interpretation());
        if !space_supported(src) {
            return Err(ColourError::UnsupportedColourspace {
                interpretation: src,
            });
        }
        if !space_supported(target) {
            return Err(ColourError::UnsupportedColourspace {
                interpretation: target,
            });
        }

        let src_bands = space_bands(src);
        let channels = self.format().channels();
        if channels < src_bands {
            return Err(ColourError::TooFewBands {
                interpretation: src,
                needed: src_bands,
                got: channels,
            });
        }
        let extras = channels - src_bands;

        let tgt_bands = space_bands(target);
        let out_channels = tgt_bands + extras;
        let src_depth = space_depth(src);
        let tgt_depth = space_depth(target);
        let bpc = self.format().bytes_per_channel();

        let total = self.width() as usize * self.height() as usize;
        // Stream row by row straight into the output byte buffer: no
        // per-pixel heap allocation and no full-image f64 staging vector
        // (libviprs#284). `src_px` is reused across pixels for the source
        // colour bands, and `tgt_px` is a reused stack scratch wide enough
        // for every space (CMYK is the widest at four bands) that
        // `from_xyz_into` writes into.
        let mut buf = vec![0u8; total * out_channels * tgt_depth.bytes()];
        let mut src_px = vec![0.0f64; src_bands];
        let mut tgt_px = [0.0f64; 4];
        let identity = src == target;
        // The same-family pairs libvips joins with a single transform
        // rather than routing through the XYZ hub; see `direct_edge`.
        let shortcut = direct_edge(src, target);

        for p in 0..total {
            let in_base = p * channels;
            for (c, slot) in src_px.iter_mut().enumerate() {
                *slot = normalise_sample(read_sample_f64(self, in_base + c), bpc, src_depth);
            }
            let out_base = p * out_channels;
            if identity {
                for (c, &v) in src_px.iter().enumerate() {
                    write_sample(&mut buf, tgt_depth, out_base + c, v);
                }
            } else {
                match shortcut {
                    Some(edge) => {
                        tgt_px[..3].copy_from_slice(&edge([src_px[0], src_px[1], src_px[2]]));
                    }
                    None => from_xyz_into(target, to_xyz(src, &src_px), &mut tgt_px),
                }
                for (c, &v) in tgt_px.iter().take(tgt_bands).enumerate() {
                    write_sample(&mut buf, tgt_depth, out_base + c, v);
                }
            }
            for c in src_bands..channels {
                // Extra bands: plain cast (clip, no rescale), mirroring
                // vips__colourspace_process_n. Clipping happens on write.
                let v = read_sample_f64(self, in_base + c).min(tgt_depth.max_value());
                write_sample(
                    &mut buf,
                    tgt_depth,
                    out_base + tgt_bands + (c - src_bands),
                    v,
                );
            }
        }

        Ok(raster_from_bytes(
            self.width(),
            self.height(),
            out_channels,
            tgt_depth,
            buf,
            self,
            target,
        ))
    }

    /// Convert this image to the target colour space (libvips
    /// `vips_colourspace`). The target is an [`Interpretation`] or a
    /// libvips space nickname (`im.colourspace(Interpretation::Lab)`,
    /// `im.colourspace("scrgb")`), the two shapes the ported suites call.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`] and on unknown space names; see
    /// [`Raster::try_colourspace`].
    pub fn colourspace(&self, target: impl Into<Interpretation>) -> Raster {
        match self.try_colourspace(target.into()) {
            Ok(out) => out,
            Err(e) => panic!("colourspace: {e}"),
        }
    }

    /// Shared front end for the colour-difference operations: converts
    /// both images to Lab, applies `f` per pixel, and carries this
    /// image's extra bands, mirroring `VipsColourDifference`.
    fn colour_difference(
        &self,
        other: &Raster,
        f: fn([f64; 3], [f64; 3]) -> f64,
    ) -> Result<Raster, ColourError> {
        if self.width() != other.width() || self.height() != other.height() {
            return Err(ColourError::DimensionMismatch {
                expected_w: self.width(),
                expected_h: self.height(),
                got_w: other.width(),
                got_h: other.height(),
            });
        }
        let left = self.try_colourspace(Interpretation::Lab)?;
        let right = other.try_colourspace(Interpretation::Lab)?;

        let l_ch = left.format().channels();
        let r_ch = right.format().channels();
        let extras = l_ch - 3;
        let out_channels = 1 + extras;
        let total = left.width() as usize * left.height() as usize;

        let mut samples = Vec::with_capacity(total * out_channels);
        for p in 0..total {
            let a = [
                read_sample_f64(&left, p * l_ch),
                read_sample_f64(&left, p * l_ch + 1),
                read_sample_f64(&left, p * l_ch + 2),
            ];
            let b = [
                read_sample_f64(&right, p * r_ch),
                read_sample_f64(&right, p * r_ch + 1),
                read_sample_f64(&right, p * r_ch + 2),
            ];
            samples.push(f(a, b));
            for c in 3..l_ch {
                samples.push(read_sample_f64(&left, p * l_ch + c));
            }
        }

        Ok(build_raster(
            left.width(),
            left.height(),
            out_channels,
            SpaceDepth::F32,
            &samples,
            &left,
            Interpretation::Bw,
        ))
    }

    /// CIE76 colour difference between two images (libvips `vips_dE76`):
    /// the Euclidean distance in Lab. Both images convert to Lab first;
    /// the result is a one-band float image plus this image's extra
    /// bands.
    ///
    /// # Errors
    ///
    /// The [`Raster::try_colourspace`] errors, plus
    /// [`ColourError::DimensionMismatch`] when the images differ in size.
    pub fn try_de76(&self, other: &Raster) -> Result<Raster, ColourError> {
        self.colour_difference(other, de76)
    }

    /// Panicking form of [`Raster::try_de76`], matching the ported call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn de76(&self, other: &Raster) -> Raster {
        match self.try_de76(other) {
            Ok(out) => out,
            Err(e) => panic!("de76: {e}"),
        }
    }

    /// CIEDE2000 colour difference between two images (libvips
    /// `vips_dE00`). Both images convert to Lab first; the result is a
    /// one-band float image plus this image's extra bands.
    ///
    /// This is a faithful port of libvips `vips_col_dE00`, whose hue-wrap
    /// arms deviate from published Sharma 2005 CIEDE2000 on hue-wrap pairs
    /// (asymmetric wrap and the antipodal `|Δh'| == 180` boundary) — by at
    /// most ~4.67 units (~1.17×) across the Sharma dataset. This is an
    /// intentional libvips parity ceiling, not a bug. See [`de00`] for the
    /// exact geometry and numeric detail. Use [`Raster::try_de00_sharma`]
    /// for the textbook value instead of libvips parity.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_de76`].
    pub fn try_de00(&self, other: &Raster) -> Result<Raster, ColourError> {
        self.colour_difference(other, de00)
    }

    /// Panicking form of [`Raster::try_de00`], matching the ported call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn de00(&self, other: &Raster) -> Raster {
        match self.try_de00(other) {
            Ok(out) => out,
            Err(e) => panic!("de00: {e}"),
        }
    }

    /// CIEDE2000 colour difference computing the **published Sharma 2005**
    /// value (both signed hue-wrap arms) rather than the libvips parity
    /// arms of [`Raster::try_de00`]. Reproduces the full published Sharma
    /// 2005 test dataset (34 pairs) within ~5e-5; see [`de00_sharma`] for
    /// the exact deviation from parity (asymmetric hue-wrap pairs only;
    /// identical elsewhere).
    ///
    /// This does not match the pinned `vips dE00` oracle and is offered
    /// only for callers who want the textbook standard.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_de76`].
    pub fn try_de00_sharma(&self, other: &Raster) -> Result<Raster, ColourError> {
        self.colour_difference(other, de00_sharma)
    }

    /// Panicking form of [`Raster::try_de00_sharma`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn de00_sharma(&self, other: &Raster) -> Raster {
        match self.try_de00_sharma(other) {
            Ok(out) => out,
            Err(e) => panic!("de00_sharma: {e}"),
        }
    }

    /// CMC colour difference between two images (libvips `vips_dECMC`):
    /// the Euclidean distance in the CMC uniform space. Both images
    /// convert to Lab first; the result is a one-band float image plus
    /// this image's extra bands.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_de76`].
    pub fn try_de_cmc(&self, other: &Raster) -> Result<Raster, ColourError> {
        self.colour_difference(other, de_cmc)
    }

    /// Panicking form of [`Raster::try_de_cmc`], matching the ported call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn de_cmc(&self, other: &Raster) -> Raster {
        match self.try_de_cmc(other) {
            Ok(out) => out,
            Err(e) => panic!("de_cmc: {e}"),
        }
    }

    /// Import this device-space image to the profile connection space
    /// using a real ICC transform (libvips `vips_icc_import`). See the
    /// [module docs](crate::colour) for the engine paths.
    ///
    /// The profile comes from `input_profile` when given, else from the
    /// image's attached `icc-profile-data`. The result is a float raster
    /// in D50-relative Lab (or XYZ scaled to `Y` white = 100 for
    /// [`Pcs::Xyz`]), with extra device bands carried unchanged and the
    /// source profile still attached so a following export can find it.
    ///
    /// # Errors
    ///
    /// [`ColourError::NoProfile`] with neither an attached nor an
    /// explicit profile, [`ColourError::ProfileRead`] /
    /// [`ColourError::InvalidProfile`] for unreadable profiles,
    /// [`ColourError::UnsupportedDeviceSpace`] for non-Gray/RGB/CMYK
    /// profiles, [`ColourError::TooFewBands`] when the image has fewer
    /// bands than the profile's device space, and
    /// [`ColourError::IccTransform`] when the CMS cannot build a
    /// transform for a LUT profile.
    pub fn try_icc_import_with(
        &self,
        intent: Intent,
        input_profile: Option<&Path>,
        pcs: Option<Pcs>,
    ) -> Result<Raster, ColourError> {
        let bytes = profile_bytes(self, input_profile)?;
        let profile = parse_profile(&bytes)?;
        let dev_ch = device_channels(&profile)?;

        let channels = self.format().channels();
        if channels < dev_ch {
            return Err(ColourError::TooFewBands {
                interpretation: self.interpretation(),
                needed: dev_ch,
                got: channels,
            });
        }
        let extras = channels - dev_ch;

        let device = read_device_normalised(self, dev_ch);
        let labs = icc_device_to_lab(&profile, &device, dev_ch, intent)?;

        let pcs = pcs.unwrap_or(Pcs::Lab);
        let tag = match pcs {
            Pcs::Lab => Interpretation::Lab,
            Pcs::Xyz => Interpretation::Xyz,
        };

        let total = self.width() as usize * self.height() as usize;
        let out_channels = 3 + extras;
        let mut samples = Vec::with_capacity(total * out_channels);
        for (p, lab) in labs.iter().enumerate() {
            match pcs {
                Pcs::Lab => samples.extend_from_slice(lab),
                Pcs::Xyz => {
                    let xyz = lab_to_xyz(*lab, ICC_D50);
                    samples.extend_from_slice(&[xyz[0] * 100.0, xyz[1] * 100.0, xyz[2] * 100.0]);
                }
            }
            for c in dev_ch..channels {
                samples.push(read_sample_f64(self, p * channels + c));
            }
        }

        Ok(build_raster(
            self.width(),
            self.height(),
            out_channels,
            SpaceDepth::F32,
            &samples,
            self,
            tag,
        ))
    }

    /// Panicking form of [`Raster::try_icc_import_with`], matching the
    /// ported call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn icc_import_with(
        &self,
        intent: Intent,
        input_profile: Option<&Path>,
        pcs: Option<Pcs>,
    ) -> Raster {
        match self.try_icc_import_with(intent, input_profile, pcs) {
            Ok(out) => out,
            Err(e) => panic!("icc_import: {e}"),
        }
    }

    /// [`Raster::icc_import_with`] at the libvips defaults: perceptual
    /// intent, the embedded profile, Lab PCS.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn icc_import(&self) -> Raster {
        self.icc_import_with(Intent::Perceptual, None, None)
    }

    /// Export this PCS image to a device colour space using a real ICC
    /// transform (libvips `vips_icc_export`). Non-Lab inputs convert to
    /// Lab first via [`Raster::colourspace`].
    ///
    /// The output profile comes from `output_profile` when given, else
    /// from the image's attached `icc-profile-data` (which
    /// [`Raster::icc_import`] leaves in place, so import-then-export
    /// round-trips through the same profile). `depth` selects 8- or
    /// 16-bit device output. The profile used is attached to the result,
    /// mirroring libvips.
    ///
    /// # Errors
    ///
    /// [`ColourError::UnsupportedDepth`] for depths other than 8 and 16,
    /// plus the [`Raster::try_icc_import_with`] profile errors.
    pub fn try_icc_export_with(
        &self,
        depth: u32,
        intent: Intent,
        output_profile: Option<&Path>,
    ) -> Result<Raster, ColourError> {
        if depth != 8 && depth != 16 {
            return Err(ColourError::UnsupportedDepth { depth });
        }
        let out_depth = if depth == 16 {
            SpaceDepth::U16
        } else {
            SpaceDepth::U8
        };

        let source = if self.interpretation() == Interpretation::Lab {
            self.clone()
        } else {
            self.try_colourspace(Interpretation::Lab)?
        };

        let bytes = profile_bytes(&source, output_profile)?;
        let profile = parse_profile(&bytes)?;
        let dev_ch = device_channels(&profile)?;

        let channels = source.format().channels();
        if channels < 3 {
            return Err(ColourError::TooFewBands {
                interpretation: source.interpretation(),
                needed: 3,
                got: channels,
            });
        }
        let extras = channels - 3;
        let total = source.width() as usize * source.height() as usize;

        let mut labs = Vec::with_capacity(total);
        for p in 0..total {
            labs.push([
                read_sample_f64(&source, p * channels),
                read_sample_f64(&source, p * channels + 1),
                read_sample_f64(&source, p * channels + 2),
            ]);
        }
        let device = icc_lab_to_device(&profile, &labs, intent)?;

        let scale = out_depth.max_value();
        let out_channels = dev_ch + extras;
        let mut samples = Vec::with_capacity(total * out_channels);
        for p in 0..total {
            for c in 0..dev_ch {
                samples.push(device[p * dev_ch + c] as f64 * scale);
            }
            for c in 3..channels {
                samples.push(read_sample_f64(&source, p * channels + c));
            }
        }

        let mut out = build_raster(
            source.width(),
            source.height(),
            out_channels,
            out_depth,
            &samples,
            &source,
            device_tag(dev_ch, out_depth),
        );
        out.set_icc_profile(&bytes);
        Ok(out)
    }

    /// Panicking form of [`Raster::try_icc_export_with`], matching the
    /// ported call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn icc_export_with(
        &self,
        depth: u32,
        intent: Intent,
        output_profile: Option<&Path>,
    ) -> Raster {
        match self.try_icc_export_with(depth, intent, output_profile) {
            Ok(out) => out,
            Err(e) => panic!("icc_export: {e}"),
        }
    }

    /// [`Raster::icc_export_with`] at the libvips defaults: 8-bit depth,
    /// perceptual intent, the attached profile.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn icc_export(&self) -> Raster {
        self.icc_export_with(8, Intent::Perceptual, None)
    }

    /// Transform this device-space image to another device profile in
    /// one step (libvips `vips_icc_transform`): import through the
    /// embedded profile, export through `output_profile`, at the image's
    /// own bit depth with perceptual intent. The output profile is
    /// attached to the result.
    ///
    /// # Errors
    ///
    /// The [`Raster::try_icc_import_with`] and
    /// [`Raster::try_icc_export_with`] errors.
    pub fn try_icc_transform(&self, output_profile: &Path) -> Result<Raster, ColourError> {
        let depth = if self.format().bytes_per_channel() == 2 {
            16
        } else {
            8
        };
        self.try_icc_import_with(Intent::Perceptual, None, None)?
            .try_icc_export_with(depth, Intent::Perceptual, Some(output_profile))
    }

    /// Panicking form of [`Raster::try_icc_transform`], matching the
    /// ported call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ColourError`].
    pub fn icc_transform(&self, output_profile: &Path) -> Raster {
        match self.try_icc_transform(output_profile) {
            Ok(out) => out,
            Err(e) => panic!("icc_transform: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small constant Lab fixture with one extra band, the shape the
    /// ported colour tests build.
    fn lab_fixture() -> Raster {
        Raster::constant(8, 8, &[50.0, 0.0, 0.0, 42.0], Interpretation::Lab)
    }

    /// An sRGB-profiled 8-bit colour fixture with a gradient of colours.
    fn srgb_profiled_fixture() -> Raster {
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8u32 {
            for x in 0..8u32 {
                data.push((x * 30 + 15) as u8);
                data.push((y * 30 + 10) as u8);
                data.push(((x + y) * 15 + 40) as u8);
            }
        }
        let mut im = Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());
        im
    }

    fn srgb_profile_bytes() -> Vec<u8> {
        ColorProfile::new_srgb().encode().unwrap()
    }

    /// Largest sample of a float raster (the arithmetic `max` does not
    /// take float rasters until the float arithmetic batch lands).
    fn float_max(r: &Raster) -> f64 {
        r.f32_samples()
            .expect("float raster")
            .iter()
            .fold(f64::MIN, |m, &v| m.max(v as f64))
    }

    /**
     * Tests that constant() builds a float raster of one band per value,
     * tagged with the given interpretation, every pixel equal.
     * Works by building the ported 4-band Lab fixture and reading pixels.
     * Input: constant(8,8,[50,0,0,42],Lab) -> RgbaF32, tag Lab, getpoint
     * equals the values at the corners.
     */
    #[test]
    fn constant_builds_float_fixture() {
        let im = lab_fixture();
        assert_eq!(im.format(), PixelFormat::RgbaF32);
        assert_eq!(im.interpretation(), Interpretation::Lab);
        assert_eq!(im.getpoint(0, 0), vec![50.0, 0.0, 0.0, 42.0]);
        assert_eq!(im.getpoint(7, 7), vec![50.0, 0.0, 0.0, 42.0]);

        let three = Raster::constant(2, 2, &[1.5, -2.0, 3.0], Interpretation::Xyz);
        assert_eq!(three.format(), PixelFormat::with_channels(3, 4).unwrap());
        assert_eq!(three.getpoint(1, 1), vec![1.5, -2.0, 3.0]);
    }

    /**
     * Tests Lab->XYZ against the Lindbloom reference the ported test
     * pins: mid-grey Lab [50,0,0] is XYZ [17.5064, 18.4187, 20.0547]
     * under D65, and the extra band is carried.
     * Input: constant Lab [50,0,0,42] -> colourspace(Xyz).
     */
    #[test]
    fn lab_xyz_lindbloom_reference() {
        let xyz = lab_fixture().colourspace(Interpretation::Xyz);
        assert_eq!(xyz.interpretation(), Interpretation::Xyz);
        let px = xyz.getpoint(3, 3);
        let expected = [17.5064, 18.4187, 20.0547, 42.0];
        for (got, exp) in px.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 0.01,
                "Lab->XYZ Lindbloom mismatch: got={got}, expected={exp}"
            );
        }
    }

    /**
     * Tests the full ported colour-space loop: Lab through Xyz, Lch, Cmc,
     * Labs, ScRgb, Hsv, Srgb, Yxy, OkLab, OkLch and back to Lab, checking
     * the interpretation tag after every step and the round-trip pixel
     * within the ported 0.1 threshold (including the extra band).
     */
    #[test]
    fn colourspace_roundtrip_loop() {
        let test = lab_fixture();
        let colour_spaces = [
            Interpretation::Xyz,
            Interpretation::Lch,
            Interpretation::Cmc,
            Interpretation::Labs,
            Interpretation::ScRgb,
            Interpretation::Hsv,
            Interpretation::Srgb,
            Interpretation::Yxy,
            Interpretation::OkLab,
            Interpretation::OkLch,
            Interpretation::Lab,
        ];

        let mut im = test.clone();
        for &cs in &colour_spaces {
            im = im.colourspace(cs);
            assert_eq!(im.interpretation(), cs);
        }

        let before = test.getpoint(3, 3);
        let after = im.getpoint(3, 3);
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 0.1,
                "round-trip mismatch: before={b}, after={a}"
            );
        }
    }

    /**
     * Tests round trips between every pair of three-band colour spaces:
     * Lab -> A -> B -> Lab stays within the ported 0.1 threshold for a
     * near-neutral colour. This exercises every to/from pair, including
     * the numerically inverted CMC space.
     */
    #[test]
    fn colourspace_pairwise_roundtrips() {
        let spaces = [
            Interpretation::Xyz,
            Interpretation::Lab,
            Interpretation::Lch,
            Interpretation::Cmc,
            Interpretation::Labs,
            Interpretation::ScRgb,
            Interpretation::Hsv,
            Interpretation::Srgb,
            Interpretation::Yxy,
            Interpretation::OkLab,
            Interpretation::OkLch,
        ];
        let test = Raster::constant(2, 2, &[50.0, 0.0, 0.0], Interpretation::Lab);
        for &a in &spaces {
            for &b in &spaces {
                let out = test
                    .colourspace(a)
                    .colourspace(b)
                    .colourspace(Interpretation::Lab);
                let px = out.getpoint(0, 0);
                for (got, exp) in px.iter().zip([50.0, 0.0, 0.0].iter()) {
                    assert!(
                        (got - exp).abs() < 0.1,
                        "Lab->{a:?}->{b:?}->Lab drifted: got={got}, expected={exp}"
                    );
                }
            }
        }
    }

    /// The `vips_col_ab2h` quadrant ladder, transcribed line for line
    /// from libvips `colour/Lab2LCh.c:61-89`. Used as the reference the
    /// crate's [`ab_to_h`] is pinned against.
    fn vips_col_ab2h(a: f64, b: f64) -> f64 {
        if a == 0.0 {
            if b < 0.0 {
                270.0
            } else if b == 0.0 {
                0.0
            } else {
                90.0
            }
        } else {
            let t = (b / a).atan();
            if a > 0.0 {
                if b < 0.0 {
                    (t + std::f64::consts::PI * 2.0).to_degrees()
                } else {
                    t.to_degrees()
                }
            } else {
                (t + std::f64::consts::PI).to_degrees()
            }
        }
    }

    /**
     * Tests sRGB white and mid-grey Lab against ABSOLUTE Oklab values
     * captured from vips 8.18.4, not just against a round trip: a
     * systematically wrong M1/M2 pair that still inverts cleanly would
     * pass every `Lab -> A -> B -> Lab` test but not these.
     * Works by converting the two fixtures and comparing every band to
     * the capture. vips runs the conversion through XYZ D65 with Y white
     * = 100 (colour/XYZ2Oklab.c:53-79), not through linear sRGB, so the
     * near-zero a/b residues below are the signature of that route --
     * they are not zero, and a linear-sRGB route would not reproduce
     * them.
     * Input (`vips colourspace in.v out.v oklab` + `vips getpoint`):
     *   sRGB [255,255,255] -> [1.0000017881393433, 2.1827961518283701e-06,
     *                          -1.1364420788595453e-04]
     *   Lab  [50,0,0]      -> [0.56896543502807617, -5.7465244935883675e-06,
     *                          -4.8703699576435611e-05]
     * Tolerance is 1e-6: vips carries the whole chain in f32, libviprs in
     * f64, and the two agree to ~1e-7 on these values.
     */
    #[test]
    fn oklab_absolute_vips_pins() {
        // Captured with:
        //   vips black b3.v 1 1 --bands 3
        //   vips linear b3.v w.v 0 255 --uchar
        //   vips copy w.v wsrgb.v --interpretation srgb
        //   vips colourspace wsrgb.v wok.v oklab && vips getpoint wok.v 0 0
        let white = Raster::new(1, 1, PixelFormat::Rgb8, vec![255, 255, 255]).unwrap();
        let got = white.colourspace(Interpretation::OkLab).getpoint(0, 0);
        let expected = [
            1.000_001_788_139_343_3,
            2.182_796_151_828_37e-6,
            -1.136_442_078_859_545_3e-4,
        ];
        for (i, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sRGB white -> Oklab band {i}: got={got}, vips={exp}"
            );
        }

        // Captured with:
        //   vips linear b3.v lab.v "0 0 0" "50 0 0"
        //   vips copy lab.v labi.v --interpretation lab
        //   vips colourspace labi.v labok.v oklab && vips getpoint labok.v 0 0
        let grey = Raster::constant(1, 1, &[50.0, 0.0, 0.0], Interpretation::Lab);
        let got = grey.colourspace(Interpretation::OkLab).getpoint(0, 0);
        let expected = [
            0.568_965_435_028_076_2,
            -5.746_524_493_588_367_5e-6,
            -4.870_369_957_643_561e-5,
        ];
        for (i, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "Lab [50,0,0] -> Oklab band {i}: got={got}, vips={exp}"
            );
        }
    }

    /**
     * Tests the three saturated sRGB primaries against ABSOLUTE OkLCh
     * values captured from vips 8.18.4, pinning the hue in DEGREES and
     * inside the [0, 360) range. Blue is the load-bearing case: its hue
     * is 264.07 degrees, which a raw `atan2` would report as -95.93, so
     * this pins the wrap as well as the unit. (The wrap is onto
     * [0, 360] with a closed top, not [0, 360); these three land well
     * inside it, so they are asserted against the open form.)
     * Works by tagging 1x1 sRGB primaries and converting to OkLCh.
     * Input (`vips colourspace in.v out.v oklch` + `vips getpoint`):
     *   [255,0,0] -> [0.62792587280273438, 0.2576846182346344,  29.223178863525391]
     *   [0,255,0] -> [0.86645191907882690, 0.29480746388435364, 142.51116943359375]
     *   [0,0,255] -> [0.45203295350074768, 0.31329533457756042, 264.07290649414062]
     */
    #[test]
    fn oklch_saturated_primaries_absolute_vips_pins() {
        let cases: [([u8; 3], [f64; 3]); 3] = [
            (
                [255, 0, 0],
                [
                    0.627_925_872_802_734_4,
                    0.257_684_618_234_634_4,
                    29.223_178_863_525_39,
                ],
            ),
            (
                [0, 255, 0],
                [
                    0.866_451_919_078_826_9,
                    0.294_807_463_884_353_64,
                    142.511_169_433_593_75,
                ],
            ),
            (
                [0, 0, 255],
                [
                    0.452_032_953_500_747_7,
                    0.313_295_334_577_560_4,
                    264.072_906_494_140_6,
                ],
            ),
        ];

        for (rgb, expected) in cases {
            let im = Raster::new(1, 1, PixelFormat::Rgb8, rgb.to_vec()).unwrap();
            let lch = im.colourspace(Interpretation::OkLch);
            assert_eq!(lch.interpretation(), Interpretation::OkLch);
            let got = lch.getpoint(0, 0);

            assert!(
                (got[0] - expected[0]).abs() < 1e-6,
                "sRGB {rgb:?} -> OkLCh L: got={}, vips={}",
                got[0],
                expected[0]
            );
            assert!(
                (got[1] - expected[1]).abs() < 1e-6,
                "sRGB {rgb:?} -> OkLCh C: got={}, vips={}",
                got[1],
                expected[1]
            );
            // Hue is in degrees, not radians, and wrapped rather than
            // signed. These three are nowhere near the 360 boundary.
            assert!(
                (got[2] - expected[2]).abs() < 1e-4,
                "sRGB {rgb:?} -> OkLCh h (degrees): got={}, vips={}",
                got[2],
                expected[2]
            );
            assert!(
                (0.0..360.0).contains(&got[2]),
                "sRGB {rgb:?} -> OkLCh h out of [0,360): {}",
                got[2]
            );
        }
    }

    /**
     * Tests that the crate's hue equals the `vips_col_ab2h` quadrant
     * ladder (colour/Lab2LCh.c:61-89) rather than merely being assumed
     * equivalent to it. The ladder has an explicit `a == 0` branch giving
     * 270 / 0 / 90, so those cases are asserted as exact equalities, and
     * the rest of the plane is swept against a line-for-line
     * transcription of the C.
     * `-0.0` is the case where the two forms genuinely disagree and so
     * the one the sweep must carry: `a == 0` is true for `-0.0` in C, so
     * vips takes the explicit branch, while `atan2(±0.0, -0.0)` is `±PI`
     * and a plain atan2 answers 180. vips 8.18.4 on the binary:
     *   oklab [0.5, -0.0,  0.0] -> oklch  0.5  0  0
     *   oklab [0.5, -0.0,  0.1] -> oklch  0.5  0.1  90
     *   oklab [0.5, -0.0, -0.1] -> oklch  0.5  0.1  270
     * The upper bound of the range is closed, not open: a positive `a`
     * with a small enough negative `b` wraps onto exactly 360.0 in both
     * implementations.
     * Works by comparing ab_to_h to vips_col_ab2h over a grid that
     * covers all four quadrants, both axes, and both signed zeros.
     */
    #[test]
    fn hue_matches_vips_col_ab2h_ladder() {
        // The explicit `a == 0` branch of the C ladder, exactly.
        assert_eq!(ab_to_h(0.0, 0.0), 0.0, "a == 0, b == 0 must be 0 degrees");
        assert_eq!(ab_to_h(0.0, 1.0), 90.0, "a == 0, b > 0 must be 90 degrees");
        assert_eq!(
            ab_to_h(0.0, -1.0),
            270.0,
            "a == 0, b < 0 must be 270 degrees"
        );
        assert_eq!(ab_to_h(0.0, 128.0), 90.0);
        assert_eq!(ab_to_h(0.0, -0.001), 270.0);
        // `a == 0` is true for `-0.0` in C too, so the same branch runs
        // and the answer is 0 / 90 / 270, NOT the 180 that
        // `atan2(±0.0, -0.0) == ±PI` would give.
        assert_eq!(
            ab_to_h(-0.0, 0.0),
            0.0,
            "a == -0.0, b == 0 must be 0 degrees, not 180"
        );
        assert_eq!(
            ab_to_h(-0.0, -0.0),
            0.0,
            "a == -0.0, b == -0.0 must be 0 degrees, not 180"
        );
        assert_eq!(ab_to_h(-0.0, 0.1), 90.0, "a == -0.0, b > 0 must be 90");
        assert_eq!(ab_to_h(-0.0, -0.1), 270.0, "a == -0.0, b < 0 must be 270");
        // And the b == 0 axis, which the ladder reaches through atan(0).
        assert_eq!(ab_to_h(1.0, 0.0), 0.0);
        assert_eq!(ab_to_h(-1.0, 0.0), 180.0);
        // The wrap lands on exactly 360.0 here and in the C, so the
        // documented range is [0, 360] and not [0, 360).
        assert_eq!(
            ab_to_h(0.1, -1e-30),
            360.0,
            "a > 0 with a tiny negative b wraps onto exactly 360"
        );
        assert_eq!(vips_col_ab2h(0.1, -1e-30), 360.0);

        let samples = [
            -128.0, -60.0, -25.0, -1.0, -0.1, -1e-9, -1e-30, -0.0, 0.0, 1e-30, 1e-9, 0.1, 1.0,
            25.0, 60.0, 128.0,
        ];
        for &a in &samples {
            for &b in &samples {
                let got = ab_to_h(a, b);
                let want = vips_col_ab2h(a, b);
                assert!(
                    (got - want).abs() < 1e-9,
                    "ab_to_h({a}, {b}) = {got}, vips_col_ab2h = {want}"
                );
                assert!(
                    (0.0..=360.0).contains(&got),
                    "ab_to_h({a}, {b}) = {got} is outside [0, 360]"
                );
            }
        }
    }

    /**
     * Tests the direct same-family polar route. libvips joins OkLab and
     * OkLCh with a single edge, `{ OKLAB, OKLCH, { vips_Oklab2Oklch } }`
     * (colour/colourspace.c:478) and `{ OKLCH, OKLAB }` (:494), so the
     * conversion is a pure polar/cartesian swap with nothing else in the
     * pipeline. Routing it through the XYZ hub instead adds an
     * Oklab2XYZ/XYZ2Oklab cube-root round trip that libvips never runs,
     * which perturbs a neutral colour's a and b away from zero and so
     * scrambles the hue that is read off them.
     * Works by converting OkLab constants captured from vips 8.18.4 and
     * comparing to the capture, then round-tripping back to OkLab.
     * The -0.0 rows are the ones a bare atan2 gets wrong: `a == 0` is
     * true for `-0.0` in C, so vips takes the explicit ladder branch,
     * while `atan2(±0.0, -0.0)` is `±PI` and answers 180.
     * Input (`vips colourspace in.v out.v oklch` + `vips getpoint`, the
     * signed-zero rows written as raw f32 and read with `vips rawload
     * ... --format float --interpretation oklab`):
     *   [0.5, 0,  0   ] -> [0.5, 0,          0        ]
     *   [0.5, 0,  0.1 ] -> [0.5, 0.1,        90       ]
     *   [0.5, 0, -0.1 ] -> [0.5, 0.1,        270      ]
     *   [0.7, 0.1,-0.05] -> [0.7, 0.11180340, 333.43494]
     *   [0.5, -0.0,  0.0] -> [0.5, 0,   0  ]
     *   [0.5, -0.0, -0.0] -> [0.5, 0,   0  ]
     *   [0.5, -0.0,  0.1] -> [0.5, 0.1, 90 ]
     *   [0.5, -0.0, -0.1] -> [0.5, 0.1, 270]
     */
    #[test]
    fn oklab_oklch_direct_route_matches_vips() {
        let cases: [([f64; 3], [f64; 3]); 8] = [
            ([0.5, 0.0, 0.0], [0.5, 0.0, 0.0]),
            ([0.5, 0.0, 0.1], [0.5, 0.100_000_001_490_116_12, 90.0]),
            ([0.5, 0.0, -0.1], [0.5, 0.100_000_001_490_116_12, 270.0]),
            ([0.5, -0.0, 0.0], [0.5, 0.0, 0.0]),
            ([0.5, -0.0, -0.0], [0.5, 0.0, 0.0]),
            ([0.5, -0.0, 0.1], [0.5, 0.100_000_001_490_116_12, 90.0]),
            ([0.5, -0.0, -0.1], [0.5, 0.100_000_001_490_116_12, 270.0]),
            (
                [0.7, 0.1, -0.05],
                [
                    0.699_999_988_079_071,
                    0.111_803_397_536_277_77,
                    333.434_936_523_437_5,
                ],
            ),
        ];

        for (oklab, expected) in cases {
            let src = Raster::constant(1, 1, &oklab, Interpretation::OkLab);
            let lch = src.colourspace(Interpretation::OkLch);
            assert_eq!(lch.interpretation(), Interpretation::OkLch);
            let got = lch.getpoint(0, 0);
            for (i, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
                let tol = if i == 2 { 1e-4 } else { 1e-7 };
                assert!(
                    (got - exp).abs() < tol,
                    "OkLab {oklab:?} -> OkLCh band {i}: got={got}, vips={exp}"
                );
            }

            // The direct edge makes the loop a polar/cartesian swap, so
            // it comes back to the f32 storage value, not to whatever a
            // cube-root round trip through XYZ leaves behind.
            let back = lch.colourspace(Interpretation::OkLab).getpoint(0, 0);
            for (i, (got, exp)) in back.iter().zip(oklab.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-7,
                    "OkLab {oklab:?} -> OkLCh -> OkLab band {i}: got={got}, expected={exp}"
                );
            }
        }
    }

    /**
     * Tests that the same direct polar route also covers Lab <-> LCh,
     * which libvips joins with `{ LAB, LCH, { vips_Lab2LCh } }`
     * (colour/colourspace.c:244) and `{ LCH, LAB }` (:276).
     * The dark neutrals are the load-bearing cases: Lab2XYZ and XYZ2Lab
     * do NOT invert each other below L = 8, because lab_f switches at
     * t < 0.008856 and lab_to_xyz at L < 8.0 and those rounded decimals
     * are not mutual inverses. Routed through the hub, Lab [5, 0, 0]
     * comes out as LCh (4.99996, C = 5.571e-4, h = 338.199) -- the same
     * neutral-garbage-hue defect as the Oklab pair, about 3e5 times
     * larger in raw units -- where vips returns 5 0 0. At L = 50 the hub
     * residual is ~1e-14 and rounds away, which is exactly why an
     * all-L=50 fixture set says nothing about this.
     * The -0.0 case pins the `a == 0` branch of the quadrant ladder,
     * which `-0.0` enters in C and which a bare atan2 would miss.
     * Works by converting Lab constants captured from vips 8.18.4,
     * including the two `a == 0` axis cases the quadrant ladder pins at
     * exactly 90 and 270 degrees.
     * Input (`vips colourspace in.v out.v lch` + `vips getpoint`, the
     * input written as raw f32 and read with `vips rawload ...
     * --format float --interpretation lab` so the signed zero survives):
     *   [50, 0, 0]     -> [50, 0,  0        ]
     *   [50, 0, 25]    -> [50, 25, 90       ]
     *   [50, 0, -25]   -> [50, 25, 270      ]
     *   [50, -30, -40] -> [50, 50, 233.13010]
     *   [50, -0.0, 0]  -> [50, 0,  0        ]
     *   [5,  0, 0]     -> [5,  0,  0        ]
     *   [3,  0, 0]     -> [3,  0,  0        ]
     */
    #[test]
    fn lab_lch_direct_route_matches_vips() {
        let cases: [([f64; 3], [f64; 3]); 7] = [
            ([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]),
            ([50.0, 0.0, 25.0], [50.0, 25.0, 90.0]),
            ([50.0, 0.0, -25.0], [50.0, 25.0, 270.0]),
            ([50.0, -30.0, -40.0], [50.0, 50.0, 233.130_096_435_546_88]),
            ([50.0, -0.0, 0.0], [50.0, 0.0, 0.0]),
            // Under L = 8 the hub route would answer
            // (4.99996, 5.571e-4, 338.199) and (2.99997, 5.571e-4, 338.199).
            ([5.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
            ([3.0, 0.0, 0.0], [3.0, 0.0, 0.0]),
        ];

        for (lab, expected) in cases {
            let src = Raster::constant(1, 1, &lab, Interpretation::Lab);
            let got = src.colourspace(Interpretation::Lch).getpoint(0, 0);
            for (i, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
                let tol = if i == 2 { 1e-4 } else { 1e-5 };
                assert!(
                    (got - exp).abs() < tol,
                    "Lab {lab:?} -> LCh band {i}: got={got}, vips={exp}"
                );
            }
        }
    }

    /**
     * Tests the ported mono round trips: Lab -> mono -> colour spaces ->
     * mono preserves the grey value within the ported thresholds (1 for
     * b-w, 30 of 65535 for grey16) and the extra band within 1.
     */
    #[test]
    fn colourspace_mono_roundtrip() {
        let test = lab_fixture();
        for &mono_fmt in &[Interpretation::Bw, Interpretation::Grey16] {
            let test_grey = test.colourspace(mono_fmt);
            assert_eq!(test_grey.interpretation(), mono_fmt);
            assert_eq!(test_grey.format().channels(), 2);

            let mut im = test_grey.clone();
            for &cs in &[
                Interpretation::Xyz,
                Interpretation::Lab,
                Interpretation::Srgb,
                mono_fmt,
            ] {
                im = im.colourspace(cs);
                assert_eq!(im.interpretation(), cs);
            }

            let before = test_grey.getpoint(3, 3);
            let after = im.getpoint(3, 3);

            let alpha_diff = (after.last().unwrap() - before.last().unwrap()).abs();
            assert!(alpha_diff < 1.0, "alpha not preserved: diff={alpha_diff}");

            let grey_threshold = if mono_fmt == Interpretation::Grey16 {
                30.0
            } else {
                1.0
            };
            let grey_diff = (after[0] - before[0]).abs();
            assert!(
                grey_diff < grey_threshold,
                "{mono_fmt:?} grey mismatch: before={}, after={}",
                before[0],
                after[0]
            );
        }
    }

    /**
     * Tests the ported CMYK round trips: CMYK -> colour space -> CMYK
     * matches within the ported threshold of 10 for Xyz, Lab, Lch, and
     * Srgb intermediates.
     */
    #[test]
    fn colourspace_cmyk_roundtrip() {
        let test = lab_fixture();
        let cmyk = test.colourspace(Interpretation::Cmyk);
        assert_eq!(cmyk.interpretation(), Interpretation::Cmyk);
        assert_eq!(cmyk.format().channels(), 5);

        for &cs in &[
            Interpretation::Xyz,
            Interpretation::Lab,
            Interpretation::Lch,
            Interpretation::Srgb,
        ] {
            let im = cmyk.colourspace(cs);
            let im2 = im.colourspace(Interpretation::Cmyk);

            let before = cmyk.getpoint(3, 3);
            let after = im2.getpoint(3, 3);
            for (b, a) in before.iter().zip(after.iter()) {
                assert!(
                    (b - a).abs() < 10.0,
                    "CMYK round trip via {cs:?}: before={b}, after={a}"
                );
            }
        }
    }

    /**
     * Tests the ported sRGB -> CMYK -> sRGB approximation on a colour
     * image: every channel returns within the ported threshold of 10.
     */
    #[test]
    fn cmyk_srgb_roundtrip() {
        let test = srgb_profiled_fixture();
        let cmyk = test.colourspace(Interpretation::Cmyk);
        let srgb = cmyk.colourspace(Interpretation::Srgb);

        for (x, y) in [(0, 0), (3, 4), (7, 7)] {
            let before = test.getpoint(x, y);
            let after = srgb.getpoint(x, y);
            for (b, a) in before.iter().zip(after.iter()) {
                assert!(
                    (b - a).abs() < 10.0,
                    "CMYK->sRGB mismatch at ({x},{y}): before={b}, after={a}"
                );
            }
        }
    }

    /**
     * Tests dE76 against the ported reference: Lab [50,10,20,42] vs
     * [40,-20,10] is sqrt(1100) = 33.166, and the extra band rides along.
     */
    #[test]
    fn de76_reference_pair() {
        let reference = Raster::constant(8, 8, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
        let sample = Raster::constant(8, 8, &[40.0, -20.0, 10.0], Interpretation::Lab);

        let difference = reference.de76(&sample);
        let px = difference.getpoint(3, 3);
        assert!(
            (px[0] - 33.166).abs() < 0.01,
            "dE76 should be ~33.166, got {}",
            px[0]
        );
        assert!((px[1] - 42.0).abs() < 0.01, "extra band should be 42");
    }

    /**
     * Tests dE00 against the ported reference: Lab [50,10,20] vs
     * [40,-20,10] is 30.238 under the libvips CIEDE2000 arrangement,
     * identical inputs give 0, and the extra band rides along.
     */
    #[test]
    fn de00_reference_pair() {
        let reference = Raster::constant(8, 8, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
        let sample = Raster::constant(8, 8, &[40.0, -20.0, 10.0], Interpretation::Lab);

        let difference = reference.de00(&sample);
        let px = difference.getpoint(3, 3);
        assert!(
            (px[0] - 30.238).abs() < 0.01,
            "dE00 should be ~30.238, got {}",
            px[0]
        );
        assert!((px[1] - 42.0).abs() < 0.01, "extra band should be 42");

        let same = reference.de00(&reference);
        assert!(same.getpoint(0, 0)[0].abs() < 1e-9);
    }

    /**
     * Tests the documented dE00 parity ceiling (issue #274): on the
     * antipodal boundary pair Lab [50,0,2.49] / [50,0,-2.49] — a symmetric
     * (h1'+h2'==360), |Δh'|==180 pair on the yellow/blue b* axis — the
     * default `de00` matches libvips `vips_col_dE00` (~4.746) while
     * `de00_sharma` returns the published Sharma value (~4.804); on the
     * non-wrapping reference pair the two are identical. Locks the
     * intentional deviation so neither arm can silently change.
     *
     * Also pins a `dC' != 0` asymmetric-wrap pair from the published
     * Sharma dataset (Lab [50,2.5,0] / [56,-27,-3], dE00 = 31.9030). The
     * [50,0,2.49] pair above has `dC' = 0`, so the `RT * dC' * dh'` cross
     * term vanishes and a sign error in the delta-hue wrap arm is
     * invisible; this second pair exercises it and locks libviprs#332.
     */
    #[test]
    fn de00_sharma_documents_wrap_deviation() {
        // Free-function level: exact numeric contract.
        let p1 = [50.0, 0.0, 2.49];
        let p2 = [50.0, 0.0, -2.49];
        let libvips = de00(p1, p2);
        let sharma = de00_sharma(p1, p2);
        assert!(
            (libvips - 4.7460).abs() < 5e-3,
            "libvips-parity de00 should be ~4.746, got {libvips}"
        );
        assert!(
            (sharma - 4.8045).abs() < 5e-3,
            "published Sharma de00 should be ~4.804, got {sharma}"
        );
        assert!(
            (sharma - libvips).abs() > 0.05,
            "wrap arms must diverge on asymmetric pairs: {sharma} vs {libvips}"
        );

        // Asymmetric-wrap pair with dC' != 0 (libviprs#332): the delta-hue
        // wrap arm's sign now matters. Published Sharma dE00 is 31.9030;
        // the pre-fix arm (copied from `de00`) negated the cross term and
        // returned 21.12.
        let w1 = [50.0, 2.5, 0.0];
        let w2 = [56.0, -27.0, -3.0];
        let sharma_wrap = de00_sharma(w1, w2);
        assert!(
            (sharma_wrap - 31.9030).abs() < 1e-3,
            "published Sharma de00 for [50,2.5,0]/[56,-27,-3] should be 31.9030, got {sharma_wrap}"
        );
        // Raster surface parity for the same pair.
        let rw1 = Raster::constant(4, 4, &w1, Interpretation::Lab);
        let rw2 = Raster::constant(4, 4, &w2, Interpretation::Lab);
        assert!((rw1.de00_sharma(&rw2).getpoint(0, 0)[0] - sharma_wrap).abs() < 1e-6);

        // Non-wrapping pair: both arms must agree exactly.
        let n1 = [50.0, 10.0, 20.0];
        let n2 = [40.0, -20.0, 10.0];
        assert!(
            (de00(n1, n2) - de00_sharma(n1, n2)).abs() < 1e-9,
            "non-wrap pair must be arm-independent"
        );

        // Raster surface parity with the free functions.
        let r1 = Raster::constant(4, 4, &[50.0, 0.0, 2.49], Interpretation::Lab);
        let r2 = Raster::constant(4, 4, &[50.0, 0.0, -2.49], Interpretation::Lab);
        assert!((r1.de00(&r2).getpoint(0, 0)[0] - libvips).abs() < 1e-6);
        assert!((r1.de00_sharma(&r2).getpoint(0, 0)[0] - sharma).abs() < 1e-6);
    }

    /**
     * Tests dECMC against the ported reference: Lab [50,10,20] vs
     * [55,11,23] is ~4.97 within the ported 0.5 tolerance, and the extra
     * band rides along.
     */
    #[test]
    fn de_cmc_reference_pair() {
        let reference = Raster::constant(8, 8, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
        let sample = Raster::constant(8, 8, &[55.0, 11.0, 23.0], Interpretation::Lab);

        let difference = reference.de_cmc(&sample);
        let px = difference.getpoint(3, 3);
        assert!(
            (px[0] - 4.97).abs() < 0.5,
            "dECMC should be ~4.97, got {}",
            px[0]
        );
        assert!((px[1] - 42.0).abs() < 0.01, "extra band should be 42");
    }

    /**
     * Tests that the colour-difference ops convert non-Lab inputs to Lab
     * first (the ported resample suite compares decoded sRGB JPEGs):
     * identical sRGB images difference to 0, and mismatched dimensions
     * are a typed error.
     */
    #[test]
    fn de_ops_convert_and_validate() {
        let im = srgb_profiled_fixture();
        let de = im.de00(&im);
        assert!(
            float_max(&de) < 1e-6,
            "identical images should difference to 0"
        );

        let other = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            im.try_de76(&other),
            Err(ColourError::DimensionMismatch { .. })
        ));
    }

    /**
     * Tests the libvips nickname parse on Interpretation: every nickname
     * maps to its variant, parsing is case-insensitive, unknown names are
     * a typed error, and the &str call shape of colourspace works
     * ("scrgb" white decodes to linear 1.0).
     */
    #[test]
    fn interpretation_from_str() {
        for (name, interp) in [
            ("multiband", Interpretation::Multiband),
            ("b-w", Interpretation::Bw),
            ("bw", Interpretation::Bw),
            ("histogram", Interpretation::Histogram),
            ("xyz", Interpretation::Xyz),
            ("lab", Interpretation::Lab),
            ("cmyk", Interpretation::Cmyk),
            ("labq", Interpretation::Labq),
            ("rgb", Interpretation::Rgb),
            ("cmc", Interpretation::Cmc),
            ("lch", Interpretation::Lch),
            ("labs", Interpretation::Labs),
            ("srgb", Interpretation::Srgb),
            ("yxy", Interpretation::Yxy),
            ("fourier", Interpretation::Fourier),
            ("rgb16", Interpretation::Rgb16),
            ("grey16", Interpretation::Grey16),
            ("matrix", Interpretation::Matrix),
            ("scrgb", Interpretation::ScRgb),
            ("hsv", Interpretation::Hsv),
            ("oklab", Interpretation::OkLab),
            ("oklch", Interpretation::OkLch),
            ("sRGB", Interpretation::Srgb),
        ] {
            assert_eq!(name.parse::<Interpretation>().unwrap(), interp, "{name}");
        }
        assert!(matches!(
            "notaspace".parse::<Interpretation>(),
            Err(ColourError::UnknownColourspace { .. })
        ));

        // The &str call shape used by the ported foreign tests.
        let white = Raster::new(1, 1, PixelFormat::Rgb8, vec![255, 255, 255]).unwrap();
        let scrgb = white.colourspace("scrgb");
        assert_eq!(scrgb.interpretation(), Interpretation::ScRgb);
        for v in scrgb.getpoint(0, 0) {
            assert!((v - 1.0).abs() < 1e-4, "sRGB white should be linear 1.0");
        }
    }

    /**
     * Tests that an unknown space name panics in the &str call shape,
     * matching the panicking convenience contract.
     */
    #[test]
    #[should_panic(expected = "unknown colour space")]
    fn colourspace_unknown_name_panics() {
        let im = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        let _ = im.colourspace("notaspace");
    }

    /**
     * Tests known sRGB gamma values: encoded 188 decodes to ~0.5029
     * linear, and the 16-bit spaces store at 16 bits.
     */
    #[test]
    fn gamma_and_depth_conventions() {
        let g = Raster::new(1, 1, PixelFormat::Rgb8, vec![188, 188, 188]).unwrap();
        let lin = g.colourspace(Interpretation::ScRgb);
        for v in lin.getpoint(0, 0) {
            assert!(
                (v - 0.5029).abs() < 0.002,
                "sRGB 188 should be ~0.5029 linear"
            );
        }

        let rgb16 = g.colourspace(Interpretation::Rgb16);
        assert_eq!(rgb16.format().bytes_per_channel(), 2);
        assert_eq!(rgb16.interpretation(), Interpretation::Rgb16);
        // 188 at 8 bits is 48316 (of 65535) at 16 bits with the shared
        // curve: 65535 * encode(decode(188/255)).
        let v = rgb16.getpoint(0, 0)[0];
        assert!((v - 48316.0).abs() < 3.0, "got {v}");

        let grey16 = g.colourspace(Interpretation::Grey16);
        assert_eq!(grey16.format(), PixelFormat::Gray16);
        assert_eq!(grey16.interpretation(), Interpretation::Grey16);
    }

    /**
     * Tests the LabS code scaling: Lab [50,0,0] stores as L 16383 (of
     * 32767) with a,b at 0, matching the libvips signed-16-bit codes.
     * 50 * 32767/100 is 16383.5, and `Lab2LabS.c:66` lands that in a
     * `signed short`, so the half goes away rather than rounding up.
     * Measured: `vips colourspace <lab> <out> labs` prints 16383.
     */
    #[test]
    fn labs_code_scaling() {
        let labs = lab_fixture().colourspace(Interpretation::Labs);
        assert_eq!(labs.interpretation(), Interpretation::Labs);
        let px = labs.getpoint(0, 0);
        assert!(
            (px[0] - 16383.0).abs() < 1e-6,
            "L code should be 16383, got {}",
            px[0]
        );
        assert!(px[1].abs() < 1e-6 && px[2].abs() < 1e-6);
        assert!((px[3] - 42.0).abs() < 1e-6, "extra band untouched");
    }

    /// One Lab pixel as a float raster, so the input words are the exact
    /// `f32` samples `vips rawload --format float --interpretation lab`
    /// hands `Lab2LabS.c`.
    fn lab_px(lab: [f64; 3]) -> Raster {
        Raster::constant(1, 1, &lab, Interpretation::Lab)
    }

    /// One LabS pixel. libviprs carries LabS in the float raster (it has
    /// no signed-16-bit format), so the code values are exact here too.
    fn labs_px(labs: [f64; 3]) -> Raster {
        Raster::constant(1, 1, &labs, Interpretation::Labs)
    }

    /**
     * Tests that Lab -> LabS truncates the scaled code toward zero
     * rather than rounding it, which is what `Lab2LabS.c:66-68` does by
     * assigning the clipped double into a `signed short`.
     *
     * Every expectation is a measurement from vips 8.18.4, taken with
     * `vips rawload px.raw in.v 1 1 3 --format float --interpretation lab`
     * so the input words are exact, then `vips colourspace in.v out.v
     * labs` and `vips getpoint out.v 0 0`.
     *
     * The `+/-0.501953125` pair is the discriminator that matters most:
     * it scales to exactly +/-128.5, so truncate-toward-zero gives
     * +/-128, round-half-away gives +/-129, and floor gives 128 / -129.
     * vips answers +/-128, so LabS truncates toward zero and does not
     * floor -- LabS is the one signed carrier in this module, so the two
     * are genuinely distinguishable here.
     */
    #[test]
    fn labs_encode_truncates_toward_zero() {
        let cases: [([f64; 3], [f64; 3]); 18] = [
            // L: 50 * 327.67 = 16383.5 -> 16383, not 16384.
            ([50.0, 0.0, 0.0], [16383.0, 0.0, 0.0]),
            // 0.1 * 327.67 = 32.767 -> 32, not 33.
            ([0.1, 0.0, 0.0], [32.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [327.0, 0.0, 0.0]),
            ([5.0, 0.0, 0.0], [1638.0, 0.0, 0.0]),
            ([8.0, 0.0, 0.0], [2621.0, 0.0, 0.0]),
            ([100.0, 0.0, 0.0], [32767.0, 0.0, 0.0]),
            // VIPS_CLIP(0, ..., SHRT_MAX) on L: no negatives, no overflow.
            ([200.0, 0.0, 0.0], [32767.0, 0.0, 0.0]),
            ([-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            // a/b scale by 256. 0.1 -> +/-25.6, so +/-25 either side of
            // zero: truncation is symmetric, floor would give -26.
            ([50.0, 0.1, -0.1], [16383.0, 25.0, -25.0]),
            // Exact halves: +/-128.5 -> +/-128.
            ([50.0, 0.501953125, -0.501953125], [16383.0, 128.0, -128.0]),
            // Exact halves again, small: +/-1.5 -> +/-1.
            ([50.0, 0.005859375, -0.005859375], [16383.0, 1.0, -1.0]),
            ([50.0, 1.0, -1.0], [16383.0, 256.0, -256.0]),
            ([50.0, 50.5, -50.5], [16383.0, 12928.0, -12928.0]),
            // VIPS_CLIP(SHRT_MIN, ..., SHRT_MAX) on a/b.
            ([50.0, 200.0, -200.0], [16383.0, 32767.0, -32768.0]),
            ([50.0, 127.99609375, -128.0], [16383.0, 32767.0, -32768.0]),
            // Shadow-branch L, where the XYZ hub is worst.
            ([3.0, 1.0, -1.0], [983.0, 256.0, -256.0]),
            ([5.0, 0.501953125, -0.501953125], [1638.0, 128.0, -128.0]),
            ([0.0, -128.0, 1.0], [0.0, -32768.0, 256.0]),
        ];

        for (lab, want) in cases {
            let px = lab_px(lab).colourspace(Interpretation::Labs).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "lab {lab:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests that LabS -> Lab is the plain division `LabS2Lab.c:57-59`
     * does, with no quantisation of its own. Values measured from vips
     * 8.18.4 with `vips rawload px.raw in.v 1 1 3 --format short
     * --interpretation labs` then `colourspace ... lab`.
     *
     * The last four cases sit under L = 8, where the XYZ hub's
     * `lab_f`/`lab_to_xyz` branch constants stop being mutual inverses,
     * so they only land if the direct edge is taken.
     */
    #[test]
    fn labs_decode_is_the_plain_division() {
        let cases: [([f64; 3], [f64; 3]); 10] = [
            ([16383.0, 0.0, 0.0], [49.99847412109375, 0.0, 0.0]),
            ([16384.0, 0.0, 0.0], [50.00152587890625, 0.0, 0.0]),
            ([32767.0, 0.0, 0.0], [100.0, 0.0, 0.0]),
            ([0.0, 128.0, -128.0], [0.0, 0.5, -0.5]),
            ([0.0, -1.0, 1.0], [0.0, -0.00390625, 0.00390625]),
            ([0.0, 32767.0, -32768.0], [0.0, 127.99609375, -128.0]),
            (
                [1638.0, 25.0, -25.0],
                [4.998931884765625, 0.09765625, -0.09765625],
            ),
            // vips prints this one as 0.99795526266098022, which is the
            // same f32 with a digit more than f64 needs.
            (
                [327.0, 1.0, -1.0],
                [0.9979552626609802, 0.00390625, -0.00390625],
            ),
            ([983.0, 256.0, -256.0], [2.999969482421875, 1.0, -1.0]),
            ([2621.0, 0.0, 0.0], [7.9989013671875, 0.0, 0.0]),
        ];

        for (labs, want) in cases {
            let px = labs_px(labs)
                .colourspace(Interpretation::Lab)
                .getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "labs {labs:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests that the Lab <-> LabS edge really does skip the XYZ hub, by
     * pinning pixels where the two routes cannot agree.
     *
     * Truncation and the hub do not mix. `Lab -> XYZ -> Lab` leaves a
     * residue of a few parts in 1e6 (the `lab_f` / `lab_to_xyz` branch
     * constants are not mutual inverses), which rounding used to absorb
     * and truncation cannot: whenever the exact code is a whole number,
     * a residue of -1e-6 drops it a whole count. So the direct edge is
     * not a performance nicety once the quantiser is right, it is the
     * only route that reproduces vips.
     *
     * Works by computing the hub answer here, through the same
     * `to_xyz` / `from_xyz_into` pair the generic route uses, and
     * asserting it misses the measured vips value that
     * `try_colourspace` hits.
     */
    #[test]
    fn labs_direct_edge_beats_the_xyz_hub() {
        // vips: lab [0, -128, 1] -> labs [0, -32768, 256].
        let lab = [0.0, -128.0, 1.0];
        let direct = lab_px(lab).colourspace(Interpretation::Labs).getpoint(0, 0);
        assert!(
            (direct[1] + 32768.0).abs() < 1e-6 && (direct[2] - 256.0).abs() < 1e-6,
            "direct edge should give vips's [-32768, 256], got {direct:?}"
        );

        let mut hub = [0.0f64; 4];
        from_xyz_into(
            Interpretation::Labs,
            to_xyz(Interpretation::Lab, &lab),
            &mut hub,
        );
        assert!(
            (hub[1] - direct[1]).abs() > 0.5 || (hub[2] - direct[2]).abs() > 0.5,
            "the hub is supposed to miss by a count here, but gave {hub:?}"
        );

        // The same in reverse: vips says labs [983, 256, -256] decodes to
        // lab [2.999969482421875, 1, -1] exactly.
        let labs = [983.0, 256.0, -256.0];
        let back = labs_px(labs)
            .colourspace(Interpretation::Lab)
            .getpoint(0, 0);
        assert!(
            (back[1] - 1.0).abs() < 1e-6 && (back[2] + 1.0).abs() < 1e-6,
            "direct edge should give vips's [1, -1], got {back:?}"
        );
        let hub_back = xyz_to_lab(to_xyz(Interpretation::Labs, &labs), D65);
        assert!(
            (hub_back[1] - 1.0).abs() > 1e-4,
            "the hub is supposed to drift here, but gave {hub_back:?}"
        );
    }

    /**
     * Tests that the truncation is not confined to the direct edge:
     * every other route into LabS ends in `vips_Lab2LabS` too
     * (`colourspace.c:229` onward), so the hub arm of `from_xyz_into`
     * has to truncate as well. Values measured from vips 8.18.4 with
     * `vips rawload px.raw in.v 1 1 3 --format uchar --interpretation
     * srgb` then `colourspace in.v out.v labs`.
     */
    #[test]
    fn labs_hub_routes_truncate_too() {
        let cases: [([u8; 3], [f64; 3]); 7] = [
            ([255, 255, 255], [32767.0, 1.0, -2.0]),
            ([0, 0, 0], [0.0, 0.0, 0.0]),
            ([255, 0, 0], [17442.0, 20507.0, 17208.0]),
            ([0, 255, 0], [28748.0, -22063.0, 21294.0]),
            ([0, 0, 255], [10584.0, 20274.0, -27613.0]),
            ([128, 128, 128], [17558.0, 0.0, -1.0]),
            ([10, 20, 30], [1949.0, -170.0, -2083.0]),
        ];

        for (rgb, want) in cases {
            let im = Raster::new(1, 1, PixelFormat::Rgb8, rgb.to_vec()).unwrap();
            let px = im.colourspace(Interpretation::Labs).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "srgb {rgb:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests HSV primaries round-trip exactly through the 8-bit hue
     * coding: pure red and blue survive sRGB -> HSV -> sRGB unchanged.
     */
    #[test]
    fn hsv_primaries_roundtrip() {
        for rgb in [[255u8, 0, 0], [0, 0, 255], [0, 255, 0], [30, 200, 100]] {
            let im = Raster::new(1, 1, PixelFormat::Rgb8, rgb.to_vec()).unwrap();
            let hsv = im.colourspace(Interpretation::Hsv);
            assert_eq!(hsv.interpretation(), Interpretation::Hsv);
            let back = hsv.colourspace(Interpretation::Srgb);
            let px = back.getpoint(0, 0);
            for (got, exp) in px.iter().zip(rgb.iter()) {
                // The hue circle is coded in 8 bits (42.5 per sextant),
                // so saturated non-primaries reconstruct within a few
                // counts, exactly as under libvips.
                assert!(
                    (got - *exp as f64).abs() <= 3.0,
                    "HSV round trip {rgb:?}: got={got}, expected={exp}"
                );
            }
        }
    }

    /**
     * Tests the typed colourspace errors: too few bands for the source
     * space, unsupported source interpretations (multiband float,
     * histogram), and unsupported targets (labq, matrix).
     */
    #[test]
    fn colourspace_typed_errors() {
        let two_band = Raster::zeroed(2, 2, PixelFormat::with_channels(2, 4).unwrap()).unwrap();
        let tagged = two_band.copy().interpretation(Interpretation::Lab).build();
        assert!(matches!(
            tagged.try_colourspace(Interpretation::Xyz),
            Err(ColourError::TooFewBands {
                needed: 3,
                got: 2,
                ..
            })
        ));

        // Untagged multiband float infers Multiband: no route.
        let multi = Raster::zeroed(2, 2, PixelFormat::with_channels(5, 4).unwrap()).unwrap();
        assert!(matches!(
            multi.try_colourspace(Interpretation::Lab),
            Err(ColourError::UnsupportedColourspace { .. })
        ));

        let hist = Raster::zeroed(2, 2, PixelFormat::Rgb8)
            .unwrap()
            .copy()
            .interpretation(Interpretation::Histogram)
            .build();
        assert!(matches!(
            hist.try_colourspace(Interpretation::Lab),
            Err(ColourError::UnsupportedColourspace { .. })
        ));

        let srgb = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            srgb.try_colourspace(Interpretation::Labq),
            Err(ColourError::UnsupportedColourspace { .. })
        ));
        assert!(matches!(
            srgb.try_colourspace(Interpretation::Matrix),
            Err(ColourError::UnsupportedColourspace { .. })
        ));
    }

    /**
     * Tests the source aliases: plain RGB converts as sRGB and Matrix as
     * mono, mirroring vips_colourspace.
     */
    #[test]
    fn colourspace_source_aliases() {
        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![119, 119, 119])
            .unwrap()
            .copy()
            .interpretation(Interpretation::Rgb)
            .build();
        let lab = rgb.colourspace(Interpretation::Lab);
        assert!((lab.getpoint(0, 0)[0] - 50.0).abs() < 0.2);

        let matrix = Raster::new(1, 1, PixelFormat::Gray8, vec![119])
            .unwrap()
            .copy()
            .interpretation(Interpretation::Matrix)
            .build();
        let lab = matrix.colourspace(Interpretation::Lab);
        assert!((lab.getpoint(0, 0)[0] - 50.0).abs() < 0.2);
    }

    /**
     * Tests extra-band handling across depths: floats survive float
     * targets untouched and plain-cast (clip, no rescale) into 8-bit
     * targets, mirroring vips__colourspace_process_n.
     */
    #[test]
    fn extra_band_plain_cast() {
        let im = Raster::constant(2, 2, &[50.0, 0.0, 0.0, 300.7], Interpretation::Lab);
        let xyz = im.colourspace(Interpretation::Xyz);
        assert!(
            (xyz.getpoint(0, 0)[3] - 300.7).abs() < 0.01,
            "float target keeps value"
        );

        let srgb = im.colourspace(Interpretation::Srgb);
        assert_eq!(srgb.getpoint(0, 0)[3], 255.0, "u8 target clips");
    }

    // -- ICC --

    /**
     * Tests ICC import through a real sRGB profile: mid-grey decodes to
     * Lab L~50 with near-zero chroma, the result is tagged Lab, and the
     * source profile stays attached for a later export.
     */
    #[test]
    fn icc_import_srgb_grey() {
        let mut im = Raster::new(1, 1, PixelFormat::Rgb8, vec![119, 119, 119]).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());

        let imported = im.icc_import();
        assert_eq!(imported.interpretation(), Interpretation::Lab);
        assert!(imported.format().is_float());
        assert!(
            imported.icc_profile().is_some(),
            "profile carried through import"
        );

        let px = imported.getpoint(0, 0);
        assert!(
            (px[0] - 50.03).abs() < 0.2,
            "L should be ~50, got {}",
            px[0]
        );
        assert!(
            px[1].abs() < 0.2 && px[2].abs() < 0.2,
            "grey should be neutral"
        );
    }

    /**
     * Tests the ported import-then-export round trip: through the
     * matrix-shaper sRGB engine every 8-bit channel returns within 1
     * count, and dE76 against the original is far under the ported
     * threshold of 6.
     */
    #[test]
    fn icc_import_export_roundtrip() {
        let im = srgb_profiled_fixture();
        let imported = im.icc_import();
        let exported = imported.icc_export();

        assert_eq!(exported.format(), PixelFormat::Rgb8);
        assert_eq!(exported.interpretation(), Interpretation::Srgb);
        for (a, b) in im.data().iter().zip(exported.data().iter()) {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "round trip drifted: {a} vs {b}"
            );
        }

        let de = exported.de76(&im);
        let max_de = float_max(&de);
        assert!(max_de < 6.0, "dE76 should be < 6, got {max_de}");
    }

    /**
     * Tests that ICC import of a *float* device raster preserves
     * sub-8-bit precision (issue #301). Two pixels whose device values
     * both round to the same 8-bit code (118.6 and 119.4 -> 119) must
     * still import to *distinct* Lab values: the f32 CMS transform sees
     * the un-rounded 118.6/255 and 119.4/255. Before the fix,
     * `read_device_normalised` rounded float input to 8 bits first, so
     * both pixels collapsed to an identical Lab and this assertion
     * (difference > 1e-3) failed.
     */
    #[test]
    fn icc_import_float_preserves_subcount_precision() {
        let fmt = PixelFormat::with_channels(3, 4).unwrap();
        assert!(fmt.is_float(), "3-band float device raster");
        // Two grey pixels that both round to the 8-bit code 119.
        let samples = [118.6f32, 118.6, 118.6, 119.4, 119.4, 119.4];
        let mut im = Raster::from_f32_samples(2, 1, fmt, &samples).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());

        let lab = im.icc_import();
        assert_eq!(lab.interpretation(), Interpretation::Lab);
        let l0 = lab.getpoint(0, 0)[0];
        let l1 = lab.getpoint(1, 0)[0];
        assert!(
            (l1 - l0).abs() > 1e-3,
            "sub-8-bit float precision must survive ICC import: \
             L0={l0}, L1={l1} (would be identical if rounded to 8-bit)"
        );
        // Higher device value maps to higher L (sanity on direction).
        assert!(l1 > l0, "monotonic: L1={l1} should exceed L0={l0}");
    }

    /**
     * Tests 16-bit ICC export: depth 16 produces 16-bit samples tagged
     * Rgb16, and unsupported depths are a typed error.
     */
    #[test]
    fn icc_export_depths() {
        let imported = srgb_profiled_fixture().icc_import();

        let exported_16 = imported.icc_export_with(16, Intent::Perceptual, None);
        assert_eq!(exported_16.format().bytes_per_channel(), 2);
        assert_eq!(exported_16.interpretation(), Interpretation::Rgb16);

        assert!(matches!(
            imported.try_icc_export_with(12, Intent::Perceptual, None),
            Err(ColourError::UnsupportedDepth { depth: 12 })
        ));
    }

    /**
     * Tests export attaches the profile it used: the exported raster
     * carries the sRGB profile bytes.
     */
    #[test]
    fn icc_export_attaches_profile() {
        let exported = srgb_profiled_fixture().icc_import().icc_export();
        assert_eq!(
            exported.icc_profile(),
            Some(srgb_profile_bytes().as_slice())
        );
    }

    /**
     * Tests ICC import to the XYZ PCS: the result is tagged Xyz and
     * mid-grey lands at Y ~18.4 on the Y-white-100 scale.
     */
    #[test]
    fn icc_import_pcs_xyz() {
        let mut im = Raster::new(1, 1, PixelFormat::Rgb8, vec![119, 119, 119]).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());

        let xyz = im.icc_import_with(Intent::Perceptual, None, Some(Pcs::Xyz));
        assert_eq!(xyz.interpretation(), Interpretation::Xyz);
        let px = xyz.getpoint(0, 0);
        assert!(
            (px[1] - 18.42).abs() < 0.5,
            "Y should be ~18.4, got {}",
            px[1]
        );

        // Default PCS stays Lab.
        let lab = im.icc_import();
        assert_eq!(lab.interpretation(), Interpretation::Lab);
    }

    /**
     * Tests icc_transform between real profiles: sRGB red re-profiled to
     * Display P3 desaturates (the P3 red channel drops, green rises), the
     * result is tagged sRGB-device, and the output profile is attached.
     */
    #[test]
    fn icc_transform_to_display_p3() {
        let dir = tempfile::tempdir().unwrap();
        let p3_bytes = ColorProfile::new_display_p3().encode().unwrap();
        let p3_path = dir.path().join("p3.icc");
        std::fs::write(&p3_path, &p3_bytes).unwrap();

        let mut im = Raster::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());

        let out = im.icc_transform(&p3_path);
        assert_eq!(out.interpretation(), Interpretation::Srgb);
        assert_eq!(out.icc_profile(), Some(p3_bytes.as_slice()));
        let px = out.getpoint(0, 0);
        assert!(
            px[0] < 250.0 && px[0] > 200.0,
            "P3 red should drop, got {}",
            px[0]
        );
        assert!(px[1] > 20.0, "P3 green should rise, got {}", px[1]);
    }

    /**
     * Tests grey-profile ICC support: a Gray8 image with a gamma-2.2 grey
     * profile imports through the exact grayTRC path (L ~54 for code 128)
     * and export round-trips within 1 count.
     */
    #[test]
    fn icc_gray_profile_roundtrip() {
        let gray_bytes = ColorProfile::new_gray_with_gamma(2.2).encode().unwrap();
        let mut im = Raster::new(1, 1, PixelFormat::Gray8, vec![128]).unwrap();
        im.set_icc_profile(&gray_bytes);

        let imported = im.icc_import();
        assert_eq!(imported.interpretation(), Interpretation::Lab);
        let px = imported.getpoint(0, 0);
        assert!(
            (px[0] - 53.8).abs() < 1.0,
            "L should be ~53.8, got {}",
            px[0]
        );
        assert!(px[1].abs() < 0.5 && px[2].abs() < 0.5);

        let exported = imported.icc_export();
        assert_eq!(exported.interpretation(), Interpretation::Bw);
        let diff = (exported.getpoint(0, 0)[0] - 128.0).abs();
        assert!(diff <= 1.0, "grey round trip drifted by {diff}");
    }

    /**
     * Tests that extra bands (alpha) ride through ICC import unchanged.
     */
    #[test]
    fn icc_import_carries_extra_bands() {
        let mut im = Raster::new(1, 1, PixelFormat::Rgba8, vec![119, 119, 119, 42]).unwrap();
        im.set_icc_profile(&srgb_profile_bytes());
        let imported = im.icc_import();
        assert_eq!(imported.format().channels(), 4);
        assert_eq!(imported.getpoint(0, 0)[3], 42.0);
    }

    /**
     * Tests the typed ICC errors: no profile anywhere, invalid profile
     * bytes, and an unreadable profile path.
     */
    #[test]
    fn icc_typed_errors() {
        let bare = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            bare.try_icc_import_with(Intent::Perceptual, None, None),
            Err(ColourError::NoProfile)
        ));

        let mut garbage = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        garbage.set_icc_profile(b"not a profile");
        assert!(matches!(
            garbage.try_icc_import_with(Intent::Perceptual, None, None),
            Err(ColourError::InvalidProfile { .. })
        ));

        assert!(matches!(
            bare.try_icc_import_with(
                Intent::Perceptual,
                Some(Path::new("/nonexistent/profile.icc")),
                None
            ),
            Err(ColourError::ProfileRead { .. })
        ));
    }

    /**
     * Pins the LUT-fallback engine's PCS encoding: running the sRGB
     * profile through the moxcms generic-Lab transform (the path LUT
     * profiles take) must agree with the exact matrix-shaper path within
     * CMS grid-interpolation error for low-chroma colours. A moxcms
     * upgrade that changes the Lab-profile PCS encoding fails here.
     */
    #[test]
    fn icc_fallback_encoding_pinned() {
        let profile = parse_profile(&srgb_profile_bytes()).unwrap();
        let device: Vec<f32> = vec![
            119.0 / 255.0,
            119.0 / 255.0,
            119.0 / 255.0,
            200.0 / 255.0,
            180.0 / 255.0,
            160.0 / 255.0,
        ];
        let exact = icc_device_to_lab(&profile, &device, 3, Intent::Perceptual).unwrap();
        let fallback =
            icc_device_to_lab_fallback(&profile, &device, 3, Intent::Perceptual).unwrap();
        for (e, f) in exact.iter().zip(fallback.iter()) {
            assert!(
                de76(*e, *f) < 2.0,
                "fallback drifted from exact path: {e:?} vs {f:?}"
            );
        }
    }

    /**
     * Tests that rendering intents map and run: relative and perceptual
     * give the same result on a matrix-shaper profile (single tag set),
     * matching lcms behaviour.
     */
    #[test]
    fn icc_intents_on_matrix_shaper() {
        let im = srgb_profiled_fixture();
        let a = im.icc_import_with(Intent::Perceptual, None, None);
        let b = im.icc_import_with(Intent::Relative, None, None);
        assert_eq!(a.data(), b.data());
    }

    /// One LCh pixel as a float raster, so the input words are the exact
    /// `f32` samples `vips rawload --format float --interpretation lch`
    /// hands `LCh2Lab.c`.
    fn lch_px(lch: [f64; 3]) -> Raster {
        Raster::constant(1, 1, &lch, Interpretation::Lch)
    }

    /// One CMC pixel as a float raster, the same shape
    /// `vips rawload --format float --interpretation cmc` produces.
    fn cmc_px(cmc: [f64; 3]) -> Raster {
        Raster::constant(1, 1, &cmc, Interpretation::Cmc)
    }

    /**
     * Tests that the LabS quantiser reads a **float** Lab rather than
     * the f64 one the rest of this module carries.
     *
     * `Lab2LabS.c:59` declares `float *restrict p`, and every libvips
     * route that ends in LabS hands it a float image, so the Lab value
     * is rounded to `f32` before the scale-and-truncate. That rounding
     * is not cosmetic once the quantiser truncates: it decides whole
     * counts.
     *
     * `LCh [0, 1, 30]` is the case that shows it. `sin(30 deg)` is
     * 0.49999999999999994 in f64 and exactly 0.5 once stored as `f32`,
     * so `b * 256` is either 127.99999999999999 (truncating to 127) or
     * 128.0 (truncating to 128). vips 8.18.4 prints 128.
     *
     * Works by pinning the whole triple for LCh inputs whose `a`/`b`
     * land on an f32 boundary, measured with `vips rawload px.raw in.v
     * 1 1 3 --format float --interpretation lch`, `vips colourspace
     * in.v out.v labs`, `vips rawsave out.v out.raw`.
     */
    #[test]
    fn labs_quantiser_reads_a_float_lab() {
        let cases: [([f64; 3], [f64; 3]); 4] = [
            ([0.0, 1.0, 30.0], [0.0, 221.0, 128.0]),
            ([50.0, 1.0, 30.0], [16383.0, 221.0, 128.0]),
            ([50.0, 25.0, 30.0], [16383.0, 5542.0, 3200.0]),
            ([20.0, 100.0, 30.0], [6553.0, 22170.0, 12800.0]),
        ];

        for (lch, want) in cases {
            let px = lch_px(lch).colourspace(Interpretation::Labs).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "lch {lch:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests the direct `LCh -> LabS` edge, `{ LCH, LABS,
     * { vips_LCh2Lab, vips_Lab2LabS } }` (`colourspace.c:280`), which
     * never touches XYZ.
     *
     * Every expectation is a measurement from vips 8.18.4 taken through
     * `rawload` / `colourspace` / `rawsave`, so the input words and the
     * output codes are both exact.
     *
     * The low-`L` rows are the ones the XYZ hub cannot reach: under
     * `L = 8` the `lab_f` / `lab_to_xyz` branch constants stop being
     * mutual inverses, and a residue of a few parts in 1e6 costs a whole
     * count once the code lands on an integer.
     */
    #[test]
    fn lch_to_labs_takes_the_direct_edge() {
        let cases: [([f64; 3], [f64; 3]); 16] = [
            ([50.0, 0.0, 0.0], [16383.0, 0.0, 0.0]),
            ([50.0, 1.0, 0.0], [16383.0, 256.0, 0.0]),
            ([50.0, 1.0, 180.0], [16383.0, -256.0, 0.0]),
            ([50.0, 25.0, 90.0], [16383.0, 0.0, 6400.0]),
            ([50.0, 25.0, 270.0], [16383.0, 0.0, -6400.0]),
            ([100.0, 128.0, 0.0], [32767.0, 32767.0, 0.0]),
            ([0.0, 128.0, 180.0], [0.0, -32768.0, 0.0]),
            ([10.0, 25.0, 270.0], [3276.0, 0.0, -6400.0]),
            ([20.0, 50.5, 180.0], [6553.0, -12928.0, 0.0]),
            ([80.0, 1.0, 0.0], [26213.0, 256.0, 0.0]),
            // Under L = 8, where the hub residue is worst.
            ([8.0, 1.0, 180.0], [2621.0, -256.0, 0.0]),
            ([5.0, 0.5019531, 0.0], [1638.0, 128.0, 0.0]),
            ([4.0, 0.5, 180.0], [1310.0, -128.0, 0.0]),
            ([3.0, 1.0, 0.0], [983.0, 256.0, 0.0]),
            ([3.0, 1.0, 180.0], [983.0, -256.0, 0.0]),
            ([2.0, 2.0, 180.0], [655.0, -512.0, 0.0]),
        ];

        for (lch, want) in cases {
            let px = lch_px(lch).colourspace(Interpretation::Labs).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "lch {lch:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests the direct `LabS -> LCh` edge, `{ LABS, LCH,
     * { vips_LabS2Lab, vips_Lab2LCh } }` (`colourspace.c:312`).
     *
     * The neutral rows are the loud ones. A LabS code with `a = b = 0`
     * is exactly neutral, so vips answers `C = 0, h = 0`, but the hub's
     * `Lab -> XYZ -> Lab` round trip pushes `a` and `b` off zero and the
     * hue read off them is garbage: 338.199 degrees, at every `L`.
     *
     * The tolerance is 1e-4 rather than the 1e-6 used for code pins,
     * because vips prints these through `float`: `hypot(127.99609375,
     * 128)` is 181.0165738689659 in f64 and 181.01657104492188 once
     * rounded to f32.
     */
    #[test]
    fn labs_to_lch_takes_the_direct_edge() {
        let cases: [([f64; 3], [f64; 3]); 15] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([327.0, 0.0, 0.0], [0.9979552626609802, 0.0, 0.0]),
            ([983.0, 0.0, 0.0], [2.999969482421875, 0.0, 0.0]),
            ([1638.0, 0.0, 0.0], [4.998931884765625, 0.0, 0.0]),
            ([2621.0, 0.0, 0.0], [7.9989013671875, 0.0, 0.0]),
            ([3276.0, 0.0, 0.0], [9.99786376953125, 0.0, 0.0]),
            ([6553.0, 0.0, 0.0], [19.998779296875, 0.0, 0.0]),
            ([16383.0, 0.0, 0.0], [49.99847412109375, 0.0, 0.0]),
            ([32767.0, 0.0, 0.0], [100.0, 0.0, 0.0]),
            // The `vips_col_ab2h` quadrant ladder, one count off zero.
            ([0.0, 1.0, 0.0], [0.0, 0.00390625, 0.0]),
            ([0.0, -1.0, 0.0], [0.0, 0.00390625, 180.0]),
            ([0.0, 0.0, 1.0], [0.0, 0.00390625, 90.0]),
            ([0.0, 0.0, -1.0], [0.0, 0.00390625, 270.0]),
            (
                [983.0, 256.0, -256.0],
                [2.999969482421875, 1.4142135381698608, 315.0],
            ),
            (
                [32767.0, 32767.0, -32768.0],
                [100.0, 181.01657104492188, 314.9991149902344],
            ),
        ];

        for (labs, want) in cases {
            let px = labs_px(labs)
                .colourspace(Interpretation::Lch)
                .getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-4,
                    "labs {labs:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests the direct `LabS -> CMC` edge, `{ LABS, CMC,
     * { vips_LabS2Lab, vips_Lab2LCh, vips_LCh2CMC } }`
     * (`colourspace.c:313`).
     *
     * Same neutral-hue story as `LabS -> LCh`: vips answers `Ccmc = 0,
     * hcmc = 0` on the neutral axis and the hub answers 338.199 degrees.
     *
     * `LabS [0, 256, 0]` is the row that shows the CMC hue correction is
     * really being applied and not skipped: `h = 0` with `C = 1` comes
     * back as 9.931086e-05, not 0, because `ch_to_hcmc`'s `d * f` term
     * is small but not zero there.
     */
    #[test]
    fn labs_to_cmc_takes_the_direct_edge() {
        let cases: [([f64; 3], [f64; 3]); 13] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([327.0, 0.0, 0.0], [1.740433931350708, 0.0, 0.0]),
            ([983.0, 0.0, 0.0], [5.2319464683532715, 0.0, 0.0]),
            ([1638.0, 0.0, 0.0], [8.71813678741455, 0.0, 0.0]),
            ([2621.0, 0.0, 0.0], [13.95008373260498, 0.0, 0.0]),
            ([3276.0, 0.0, 0.0], [17.4362735748291, 0.0, 0.0]),
            ([6553.0, 0.0, 0.0], [34.2913818359375, 0.0, 0.0]),
            ([16383.0, 0.0, 0.0], [65.7352523803711, 0.0, 0.0]),
            ([32767.0, 0.0, 0.0], [100.00244903564453, 0.0, 0.0]),
            (
                [0.0, 256.0, 0.0],
                [0.0, 1.3314666748046875, 9.931086242431775e-05],
            ),
            ([0.0, -1.0, 0.0], [0.0, 0.004822731018066406, 180.0]),
            (
                [0.0, -32768.0, 0.0],
                [0.0, 50.649295806884766, 192.8778839111328],
            ),
            (
                [16383.0, 0.0, 6400.0],
                [65.7352523803711, 18.706565856933594, 114.02556610107422],
            ),
        ];

        for (labs, want) in cases {
            let px = labs_px(labs)
                .colourspace(Interpretation::Cmc)
                .getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-4,
                    "labs {labs:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests the direct `CMC -> LabS` edge, `{ CMC, LABS,
     * { vips_CMC2LCh, vips_LCh2Lab, vips_Lab2LabS } }`
     * (`colourspace.c:297`).
     *
     * The CMC inverse is the one place this module and libvips really do
     * compute different numbers: libvips inverts `Lcmc`, `Ccmc` and
     * `hcmc` through interpolation tables sampled every 0.1
     * (`UCS2LCh.c:66-135`) and this module bisects the forward function.
     * The tables are the coarser of the two, by about 6e-8 in `L`, so a
     * CMC value whose LabS code sits a hair above a whole number cannot
     * be matched from either side: `Lcmc = 3.4861903190612793` scales to
     * 655.0000104 in libvips and 654.99999 here.
     *
     * So the pins are chosen at codes with real slack, generated by
     * running `vips colourspace <lch> <out> cmc` on `LCh [L, 0, 0]` for
     * an `L` whose code is not near an integer, plus two chromatic
     * values where the tables and the bisection agree.
     */
    #[test]
    fn cmc_to_labs_takes_the_direct_edge() {
        let cases: [([f64; 3], [f64; 3]); 19] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.871999979019165, 0.0, 0.0], [163.0, 0.0, 0.0]),
            ([2.615999937057495, 0.0, 0.0], [491.0, 0.0, 0.0]),
            ([4.359999656677246, 0.0, 0.0], [819.0, 0.0, 0.0]),
            ([6.97599983215332, 0.0, 0.0], [1310.0, 0.0, 0.0]),
            ([10.46399974822998, 0.0, 0.0], [1966.0, 0.0, 0.0]),
            ([13.079999923706055, 0.0, 0.0], [2457.0, 0.0, 0.0]),
            ([15.695999145507812, 0.0, 0.0], [2949.0, 0.0, 0.0]),
            ([20.92799949645996, 0.0, 0.0], [3932.0, 0.0, 0.0]),
            ([26.15999984741211, 0.0, 0.0], [4915.0, 0.0, 0.0]),
            ([34.293174743652344, 0.0, 0.0], [6553.0, 0.0, 0.0]),
            ([46.950042724609375, 0.0, 0.0], [9830.0, 0.0, 0.0]),
            ([65.73650360107422, 0.0, 0.0], [16383.0, 0.0, 0.0]),
            ([80.73076629638672, 0.0, 0.0], [22936.0, 0.0, 0.0]),
            ([93.87285614013672, 0.0, 0.0], [29490.0, 0.0, 0.0]),
            (
                [87.4730759, 22.4464149, 16.5072842],
                [26213.0, 8152.0, 2492.0],
            ),
            (
                [61.5259094, 16.0966263, 316.1648865],
                [14745.0, 3342.0, -3844.0],
            ),
            // The two the hub cannot reach: `a` lands on a whole code,
            // so the XYZ round trip's residue costs a count.
            (
                [0.0, 50.649295806884766, 192.8778839111328],
                [0.0, -32768.0, 0.0],
            ),
            (
                [0.0, 1.3314666748046875, 9.931086242431775e-05],
                [0.0, 256.0, 0.0],
            ),
        ];

        for (cmc, want) in cases {
            let px = cmc_px(cmc).colourspace(Interpretation::Labs).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-6,
                    "cmc {cmc:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests that adding these edges did not fork the arithmetic: where
     * the XYZ round trip leaves no residue, the direct edge and the hub
     * agree, and both match vips. Where it does, only the direct edge
     * matches.
     *
     * This is the check #556 had to invent. `direct_edge` never reaches
     * `from_xyz_into`, so an entry that reproduced the target-side
     * production instead of sharing it would drift silently. The LabS
     * truncation lives in `lab_to_labs` and the CMC encode in
     * `lch_to_cmc` for exactly that reason, and this test is what says
     * so out loud.
     *
     * Works by driving `to_xyz` / `from_xyz_into` directly for the hub
     * answer and `try_colourspace` for the routed one, then comparing
     * both against measured vips values.
     */
    #[test]
    fn lab_family_direct_edges_agree_with_the_hub_off_the_integer_codes() {
        // Off the integer codes and above the shadow branch, the hub's
        // residue is far too small to move a count, so all three agree.
        let agree: [([f64; 3], [f64; 3]); 5] = [
            ([50.0, 1.0, 37.0], [16383.0, 204.0, 154.0]),
            ([80.0, 33.3, 17.0], [26213.0, 8152.0, 2492.0]),
            ([60.0, 12.7, 203.0], [19660.0, -2992.0, -1270.0]),
            ([45.0, 19.9, 311.0], [14745.0, 3342.0, -3844.0]),
            ([90.0, 3.3, 61.0], [29490.0, 409.0, 738.0]),
        ];
        for (lch, want) in agree {
            let direct = lch_px(lch).colourspace(Interpretation::Labs).getpoint(0, 0);
            let mut hub = [0.0f64; 4];
            from_xyz_into(
                Interpretation::Labs,
                to_xyz(Interpretation::Lch, &lch),
                &mut hub,
            );
            for c in 0..3 {
                assert!(
                    (direct[c] - want[c]).abs() < 1e-6,
                    "lch {lch:?} band {c}: vips says {}, direct gave {}",
                    want[c],
                    direct[c]
                );
                assert!(
                    (hub[c] - want[c]).abs() < 1e-6,
                    "lch {lch:?} band {c}: the hub should agree here, \
                     vips says {}, hub gave {}",
                    want[c],
                    hub[c]
                );
            }
        }

        // On an integer code under the shadow branch, only the direct
        // edge can reach vips's answer.
        let lch = [3.0, 1.0, 180.0];
        let direct = lch_px(lch).colourspace(Interpretation::Labs).getpoint(0, 0);
        assert!(
            (direct[1] + 256.0).abs() < 1e-6,
            "direct edge should give vips's -256, got {direct:?}"
        );
        let mut hub = [0.0f64; 4];
        from_xyz_into(
            Interpretation::Labs,
            to_xyz(Interpretation::Lch, &lch),
            &mut hub,
        );
        assert!(
            (hub[1] - direct[1]).abs() > 0.5,
            "the hub is supposed to miss by a count here, but gave {hub:?}"
        );

        // The same for CMC, whose encoder the hub arm shares: a neutral
        // LabS is neutral in CMC too, and the hub invents a hue.
        let labs = [1638.0, 0.0, 0.0];
        let direct = labs_px(labs)
            .colourspace(Interpretation::Cmc)
            .getpoint(0, 0);
        assert!(
            direct[1].abs() < 1e-4 && direct[2].abs() < 1e-4,
            "direct edge should give vips's neutral [0, 0], got {direct:?}"
        );
        let mut hub = [0.0f64; 4];
        from_xyz_into(
            Interpretation::Cmc,
            to_xyz(Interpretation::Labs, &labs),
            &mut hub,
        );
        assert!(
            hub[2] > 1.0,
            "the hub is supposed to invent a hue here, but gave {hub:?}"
        );
    }

    /**
     * Tests that the hub arm still owns the same CMC encoder after the
     * direct edges were added: `srgb -> cmc` is a genuine XYZ route
     * (`colourspace.c:395`) and its answers are unchanged.
     *
     * Values measured from vips 8.18.4 with `vips rawload px.raw in.v
     * 1 1 3 --format uchar --interpretation srgb` then `vips colourspace
     * in.v out.v cmc`. The tolerance is 1e-4 because vips stages this
     * route through `float` images and prints `float`.
     *
     * Only saturated colours are pinned. White and mid-grey come out of
     * the sRGB primaries matrix with a chroma of about 0.01, where the
     * hue is numerically meaningless and this crate and libvips already
     * disagree by 0.1 degrees for reasons that predate these edges.
     */
    #[test]
    fn cmc_hub_route_shares_the_encoder() {
        let cases: [([u8; 3], [f64; 3]); 4] = [
            (
                [255, 0, 0],
                [68.3399887084961, 44.80427551269531, 43.62641525268555],
            ),
            (
                [0, 255, 0],
                [92.4504623413086, 48.64079666137695, 156.00975036621094],
            ),
            (
                [0, 0, 255],
                [49.44218444824219, 52.046016693115234, 311.6222839355469],
            ),
            (
                [10, 20, 30],
                [10.37486743927002, 8.466974258422852, 267.78424072265625],
            ),
        ];

        for (rgb, want) in cases {
            let im = Raster::new(1, 1, PixelFormat::Rgb8, rgb.to_vec()).unwrap();
            let px = im.colourspace(Interpretation::Cmc).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-4,
                    "srgb {rgb:?} band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // scRGB -> sRGB goes through the interpolated libvips LUT (#581)
    // -----------------------------------------------------------------

    /// One LabS pixel converted to `target`, as integer codes.
    fn labs_to(labs_l: f64, target: Interpretation) -> Vec<f64> {
        labs_px([labs_l, 0.0, 0.0])
            .colourspace(target)
            .getpoint(0, 0)
    }

    /**
     * Tests that the linear -> sRGB encode is the libvips 256-entry
     * integer LUT read with a piecewise-linear interpolation, not the
     * analytic IEC 61966-2-1 curve.
     *
     * vips never evaluates the transfer function per pixel.
     * `LabQ2sRGB.c:126-146` builds `Y2v[i] = rintf(255 * encode(i/255))`
     * once, in `float`, and `vips_col_scRGB2sRGB` (`:282-353`) then
     * interpolates between two ALREADY-ROUNDED integer entries and
     * `rintf`s the chord. That stacks three quantisations the analytic
     * form has none of, and it moves the answer by a whole count on
     * 5434 of the 32768 neutral LabS L codes.
     *
     * Works by driving the measured `Labs -> b-w` and `Labs -> sRGB`
     * codes at L values where the two disagree, plus controls where they
     * agree so the test cannot pass by shifting everything.
     *
     * Input: `vips rawload labs.raw in.v 32768 1 3 --format short
     * --interpretation labs`, then `vips colourspace in.v out.v b-w`
     * (and `srgb`) and `vips rawsave out.v out.raw`, on 8.18.4.
     */
    #[test]
    fn scrgb_to_srgb_reads_the_interpolated_vips_lut() {
        // (LabS L, vips b-w, vips sRGB). The first five are codes where
        // the analytic curve answers one LESS than vips; the last three
        // are controls the two already agreed on.
        let cases: [(f64, f64, f64); 8] = [
            (134.0, 2.0, 2.0),
            (224.0, 3.0, 3.0),
            (313.0, 4.0, 4.0),
            (402.0, 5.0, 5.0),
            (20000.0, 148.0, 148.0),
            (0.0, 0.0, 0.0),
            (1000.0, 11.0, 11.0),
            (32767.0, 255.0, 255.0),
        ];

        for (l, want_bw, want_srgb) in cases {
            let bw = labs_to(l, Interpretation::Bw);
            assert!(
                (bw[0] - want_bw).abs() < 1e-9,
                "labs [{l}, 0, 0] -> b-w: vips says {want_bw}, got {}",
                bw[0]
            );
            let srgb = labs_to(l, Interpretation::Srgb);
            for (c, got) in srgb.iter().enumerate().take(3) {
                assert!(
                    (got - want_srgb).abs() < 1e-9,
                    "labs [{l}, 0, 0] -> srgb band {c}: vips says \
                     {want_srgb}, got {got}"
                );
            }
        }
    }

    /**
     * Tests that the chord is finished with `rintf`, which is round half
     * to EVEN, and not with a half-away-from-zero round.
     *
     * `LabQ2sRGB.c:337` / `:422` end the interpolation with `rintf(v)`,
     * and the Homebrew arm64 8.18.4 build compiles that to `frintx`,
     * i.e. the default round-to-nearest-ties-to-even mode. The 16-bit
     * table is where that is observable: the chord lands on an exact
     * `.5` for 70 of the 32768 neutral LabS L codes, and the two rules
     * disagree on half of them.
     *
     * Works by pinning one tie that resolves DOWN and one that resolves
     * UP, so neither `floor` nor `ceil` nor half-away can pass both.
     * L = 4746 puts the chord at exactly 9404.5 and vips answers 9404
     * (half-away would say 9405); L = 5505 puts it at exactly 10651.5
     * and vips answers 10652.
     *
     * Input: as above, then `vips colourspace in.v out.v grey16`.
     */
    #[test]
    fn scrgb_to_srgb_rounds_ties_to_even() {
        for (l, want) in [(4746.0, 9404.0), (5505.0, 10652.0)] {
            let grey = labs_to(l, Interpretation::Grey16);
            assert!(
                (grey[0] - want).abs() < 1e-9,
                "labs [{l}, 0, 0] -> grey16: vips says {want}, got {}",
                grey[0]
            );
        }
    }

    /**
     * Tests that the 16-bit spaces take their own 65536-entry table
     * rather than scaling the 8-bit one, and that the table really is
     * sampled at 65536 points.
     *
     * `calcul_tables_16` (`LabQ2sRGB.c:174`) builds `vips_Y2v_16` at the
     * full 16-bit range, so `rgb16` and `grey16` resolve detail the
     * 8-bit table cannot: L = 491 and L = 4746 both quantise to a flat
     * neutral in sRGB but come out with a per-channel spread at 16 bits.
     *
     * Input: as above, with `vips colourspace in.v out.v rgb16`.
     */
    #[test]
    fn rgb16_takes_the_65536_entry_table() {
        let cases: [(f64, [f64; 3]); 4] = [
            (491.0, [1405.0, 1404.0, 1404.0]),
            (4746.0, [9404.0, 9405.0, 9404.0]),
            (6923.0, [13040.0, 13041.0, 13039.0]),
            (20000.0, [37845.0, 37846.0, 37843.0]),
        ];

        for (l, want) in cases {
            let px = labs_to(l, Interpretation::Rgb16);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-9,
                    "labs [{l}, 0, 0] -> rgb16 band {c}: vips says {exp}, \
                     got {}",
                    px[c]
                );
            }
        }
    }

    /**
     * Tests that the HSV arm quantises through the SAME LUT, because
     * `{ *, HSV }` reaches HSV via `vips_scRGB2sRGB` then
     * `vips_sRGB2HSV` (`colourspace.c:336`, `:355` onward) and so sees
     * the LUT's 8-bit codes, not an analytic encode rounded afterwards.
     *
     * Works by picking L = 491, the one neutral LabS code in this set
     * where the LUT breaks the grey: it gives sRGB [6, 5, 5], so HSV
     * reports a real saturation of 42, while the analytic encode gives a
     * flat [5, 5, 5] and therefore saturation 0. That makes the case
     * discriminating on the HSV arm specifically rather than on the sRGB
     * codes it is built from.
     *
     * Input: as above, with `vips colourspace in.v out.v hsv`.
     */
    #[test]
    fn hsv_quantises_through_the_lut_sampled_srgb() {
        let px = labs_to(491.0, Interpretation::Hsv);
        let want = [0.0, 42.0, 6.0];
        for (c, &exp) in want.iter().enumerate() {
            assert!(
                (px[c] - exp).abs() < 1e-9,
                "labs [491, 0, 0] -> hsv band {c}: vips says {exp}, got {}",
                px[c]
            );
        }
    }

    /**
     * Tests that [`calcul_tables`] reproduces the libvips `Y2v` tables
     * entry for entry, at both ranges.
     *
     * The table is the thing everything else in this mechanism is built
     * on, so it is pinned directly rather than only through the codes it
     * produces. Seven of the 16-bit entries here are ones the FUSED
     * multiply-add decides: evaluate `1.055 * powf(f, 1/2.4) - 0.055`
     * unfused and they each drop by a count, which is 45 of the 65536
     * entries in total. The 256-entry table is the same either way, so
     * the 8-bit rows pin the shape and the 16-bit rows pin the fusion.
     *
     * Input: the tables were read back out of vips 8.18.4 rather than
     * recomputed. Feeding scRGB knots `i / (range - 1)` through
     * `vips rawload knots.raw k.v <range> 1 3 --format float
     * --interpretation scrgb` then `vips colourspace k.v out.v srgb`
     * (and `rgb16`) makes the interpolation land on entry `i`, so the
     * output code IS `Y2v[i]`.
     */
    #[test]
    fn calcul_tables_matches_the_vips_y2v_tables() {
        let y2v_8 = calcul_tables(SRGB_RANGE);
        assert_eq!(y2v_8.len(), SRGB_RANGE + 1);
        assert_eq!(&y2v_8[..10], &[0, 13, 22, 28, 34, 38, 42, 46, 50, 53]);
        assert_eq!(&y2v_8[248..256], &[252, 252, 253, 253, 254, 254, 255, 255]);
        // "Copy the final element" (`LabQ2sRGB.c:141-144`).
        assert_eq!(y2v_8[SRGB_RANGE], y2v_8[SRGB_RANGE - 1]);

        let y2v_16 = calcul_tables(RGB16_RANGE);
        assert_eq!(y2v_16.len(), RGB16_RANGE + 1);
        // (index, vips entry). Everything from 3696 to 25615 is an entry
        // the unfused form gets one count too low.
        let cases: [(usize, i32); 15] = [
            (0, 0),
            (1, 13),
            (2, 26),
            (3, 39),
            (255, 3244),
            (3696, 17261),
            (3857, 17635),
            (5925, 21795),
            (8993, 26618),
            (8998, 26625),
            (9393, 27171),
            (25615, 43141),
            (64674, 65155),
            (65534, 65535),
            (65535, 65535),
        ];
        for (i, want) in cases {
            assert_eq!(y2v_16[i], want, "Y2v_16[{i}]");
        }
        assert_eq!(y2v_16[RGB16_RANGE], y2v_16[RGB16_RANGE - 1]);
    }

    /**
     * One `f32` bit pattern's worth of the [`scrgb_to_code`] chord,
     * evaluated both ways and compared as raw bits.
     *
     * [`scrgb_to_code`] finishes its chord with `f64` arithmetic rather
     * than `f32::mul_add`, because `fma` is not in the x86-64 baseline
     * and rustc lowers `f32::mul_add` to a libm `fmaf` call there, once
     * per channel per pixel. The two spellings agree bit for bit
     * because the exact product-sum is representable in an `f64` for
     * every reachable input, so the single `as f32` is the single
     * rounding `fmaf` performs:
     *
     * - `lo = lut[yi]` is an integer in `0..=range - 1`, exact in both
     *   `f32` and `f64`.
     * - `delta = lut[yi + 1] - lut[yi]` is an integer with
     *   `|delta| <= 65535`, exact in both.
     * - `t = yf - yi as f32` is exact: `yf < 2^24`, so subtracting its
     *   truncated integer part cannot lose a bit, and `t` lands in
     *   `[0, 1)`.
     * - For `yi >= 1`, `yf >= 1` so `ulp(yf) >= 2^-23` and `t` is a
     *   multiple of `2^-23`. Then `delta * t` is a multiple of `2^-23`
     *   under `2^16`, and adding the integer `lo` keeps it a multiple
     *   of `2^-23` under `2^17`: at most 40 significand bits, inside
     *   `f64`'s 53.
     * - For `yi == 0`, `lo == lut[0] == 0` (the linear arm gives `v = 0`
     *   at `i = 0`), so the sum is a bare product of two `f32`s, exact
     *   in `f64` at 24 + 24 = 48 bits.
     *
     * Works by taking `bits` as a raw `f32` pattern standing in for the
     * clamped `yf`, so the caller owns the coverage and this owns only
     * the comparison.
     *
     * Input: none. Nothing here is a parity claim against vips, it is a
     * claim about two Rust expressions, so there is no oracle to read.
     */
    fn check_f64_chord_at(lut: &[i32], range: usize, bits: u32) {
        let yf = f32::from_bits(bits);
        let yi = yf as usize;
        let lo = lut[yi];
        let delta = (lut[yi + 1] - lo) as f32;
        let t = yf - yi as f32;
        let want = delta.mul_add(t, lo as f32);
        let got = (f64::from(delta) * f64::from(t) + f64::from(lo)) as f32;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "range {range}, yf bits {bits:#010x} ({yf:e}), yi {yi}, \
             delta {delta}, lo {lo}: mul_add gives {want:e}, the f64 \
             chord gives {got:e}"
        );
    }

    /**
     * The structural subset of the chord equivalence that runs on every
     * `cargo test`, as against the exhaustive sweep in
     * [`sweep_f64_chord_against_mul_add`], which does not.
     *
     * Works by checking, for one `range`:
     *
     * - `-0.0`, which survives the `clamp(0.0, maxval)` in
     *   [`scrgb_to_code`] (it is neither below the low bound nor above
     *   the high one) and so reaches the lookup;
     * - every `f32` in `[1.0, 2.0)`, 8388608 patterns, which is where
     *   the arithmetic is under the most pressure: `ulp(yf)` is at its
     *   smallest of any `yi >= 1`, so `t` carries a full 23 fractional
     *   bits, and `delta` is near its largest because the transfer
     *   curve is steepest at the bottom of the table;
     * - both ends of EVERY LUT cell, 64 patterns deep each way, so
     *   `t == 0` and the largest `t` below 1 are covered at every `yi`,
     *   including `yi == 0`, both knees of the piecewise curve, and the
     *   `yi == range - 1` cell where the duplicated final entry makes
     *   `delta` zero and the clamp allows only `t == 0`;
     * - a stride of 4517 over the whole `+0.0..=maxval` bit range, so
     *   every `f32` exponent is represented, denormals included. They
     *   all land in `yi == 0`, which the per-cell walk only samples at
     *   its two ends.
     *
     * That is a few tens of millions of patterns rather than the
     * billion-odd the full sweep walks, and it runs in well under a
     * second per range in a debug build.
     *
     * Input: none, see [`check_f64_chord_at`].
     */
    fn spot_check_f64_chord(range: usize) {
        let lut = y2v_table(range);
        let maxval = (range - 1) as f32;
        let top = maxval.to_bits();

        check_f64_chord_at(lut, range, 0x8000_0000);

        for bits in 1.0_f32.to_bits()..2.0_f32.to_bits() {
            check_f64_chord_at(lut, range, bits);
        }

        const BAND: u32 = 64;
        for yi in 0..range {
            let cell_lo = (yi as f32).to_bits();
            let cell_hi = ((yi + 1) as f32).to_bits();
            for bits in cell_lo..=(cell_lo + BAND).min(top) {
                check_f64_chord_at(lut, range, bits);
            }
            for bits in cell_hi.saturating_sub(BAND)..cell_hi.min(top + 1) {
                check_f64_chord_at(lut, range, bits);
            }
        }

        const STRIDE: u32 = 4517;
        let mut bits = 0;
        while bits < top {
            check_f64_chord_at(lut, range, bits);
            bits += STRIDE;
        }
        check_f64_chord_at(lut, range, top);
    }

    /**
     * The exhaustive sweep behind the two `#[ignore]`d
     * `f64_chord_matches_mul_add_*` tests.
     *
     * Works by walking `yf` over EVERY `f32` bit pattern from `+0.0` to
     * `maxval`, plus `-0.0`, which is a superset of what the clamp in
     * [`scrgb_to_code`] can hand the lookup. That is 1132396546
     * patterns at range 256 and 1199570690 at range 65536, so it is a
     * real sweep of the index space rather than a sample of it, and it
     * is why the two tests that call this are `#[ignore]`d rather than
     * run on every `cargo test`. [`spot_check_f64_chord`] is what runs
     * by default; it covers the structure but not the whole space.
     *
     * Input: none, see [`check_f64_chord_at`], whose doc carries the
     * argument for why the two spellings must agree.
     */
    fn sweep_f64_chord_against_mul_add(range: usize) {
        let lut = y2v_table(range);
        let maxval = (range - 1) as f32;

        let mut bits = 0x8000_0000_u32;
        let top = maxval.to_bits();
        loop {
            check_f64_chord_at(lut, range, bits);
            if bits == 0x8000_0000 {
                bits = 0;
            } else if bits == top {
                break;
            } else {
                bits += 1;
            }
        }
    }

    /**
     * Tests that the `f64` chord in [`scrgb_to_code`] is bit-identical
     * to the `f32::mul_add` it replaced at every structurally
     * interesting point of the 256-entry LUT.
     *
     * Works by [`spot_check_f64_chord`], whose doc lists exactly what
     * that covers. The point of running it rather than only stating the
     * argument is that the equivalence is what licenses dropping a
     * per-pixel libm `fmaf` call on x86-64 without moving a single
     * output code.
     *
     * Input: none, see the helper.
     */
    #[test]
    fn f64_chord_matches_mul_add_over_the_srgb_lut_sample() {
        spot_check_f64_chord(SRGB_RANGE);
    }

    /**
     * Tests the same structural sample across the 65536-entry LUT.
     *
     * Input: none, see [`spot_check_f64_chord`].
     */
    #[test]
    fn f64_chord_matches_mul_add_over_the_rgb16_lut_sample() {
        spot_check_f64_chord(RGB16_RANGE);
    }

    /**
     * Tests the same equivalence over EVERY `f32` the 256-entry lookup
     * can be handed, not just the structural sample.
     *
     * `#[ignore]`d because it walks 1132396546 bit patterns. Measured
     * on an M-series mac that is 10.64s in a debug build and about a
     * second in release, and it will be worse on a target without `fma`
     * in its baseline, where the `f32::mul_add` arm of the comparison
     * is itself a libm call. Nothing in the default `cargo test` run
     * covers the whole space; this is what does, and it is worth
     * running whenever the LUT or the chord changes:
     *
     * ```text
     * cargo test --release --lib -- --ignored f64_chord_matches_mul_add
     * ```
     *
     * Input: none, see [`sweep_f64_chord_against_mul_add`].
     */
    #[test]
    #[ignore = "walks every f32 bit pattern up to maxval; run with --ignored"]
    fn f64_chord_matches_mul_add_over_the_whole_srgb_domain() {
        sweep_f64_chord_against_mul_add(SRGB_RANGE);
    }

    /**
     * Tests the same exhaustive equivalence across the whole
     * 65536-entry LUT, 1199570690 bit patterns.
     *
     * Split from the 8-bit sweep so the two run on separate test
     * threads; they have no state in common. `#[ignore]`d for the same
     * reason, 10.30s in a debug build, and run by the same invocation
     * as [`f64_chord_matches_mul_add_over_the_whole_srgb_domain`].
     *
     * Input: none, see [`sweep_f64_chord_against_mul_add`].
     */
    #[test]
    #[ignore = "walks every f32 bit pattern up to maxval; run with --ignored"]
    fn f64_chord_matches_mul_add_over_the_whole_rgb16_domain() {
        sweep_f64_chord_against_mul_add(RGB16_RANGE);
    }

    /**
     * Tests that sRGB -> HSV TRUNCATES the hue and saturation codes on
     * the store, and that the hue's ratio is an `f32` division.
     *
     * `sRGB2HSV.c:113-117` writes both into an `unsigned char`, which
     * drops the fraction; libviprs used to hand them out unrounded and
     * let the writer round, which missed vips on about a third of the
     * two bands. It stayed invisible until #581, because the analytic
     * sRGB encode produced flat greys where the LUT produces a real
     * spread, and a flat grey has `delta == 0` and therefore no hue or
     * saturation to get wrong.
     *
     * Works by pinning one case where both codes have a fraction over a
     * half, so rounding and truncating disagree on BOTH, and one where
     * the hue is exactly on the boundary between the two precisions:
     * sRGB [5, 7, 22] puts `42.5 * (-2 / 17) + 170` a hair under 165 in
     * `f32` and a hair over it in `f64`, and vips answers 164.
     *
     * Input: `vips rawload px.raw in.v N 1 3 --format uchar
     * --interpretation srgb`, then `vips colourspace in.v out.v hsv`.
     */
    #[test]
    fn srgb_to_hsv_truncates_hue_and_saturation() {
        let cases: [([u8; 3], [f64; 3]); 3] = [
            ([5, 7, 66], [168.0, 235.0, 66.0]),
            ([5, 7, 88], [168.0, 240.0, 88.0]),
            ([5, 7, 22], [164.0, 197.0, 22.0]),
        ];

        for (rgb, want) in cases {
            let im = Raster::new(1, 1, PixelFormat::Rgb8, rgb.to_vec()).unwrap();
            let px = im.colourspace(Interpretation::Hsv).getpoint(0, 0);
            for (c, &exp) in want.iter().enumerate() {
                assert!(
                    (px[c] - exp).abs() < 1e-9,
                    "srgb {rgb:?} -> hsv band {c}: vips says {exp}, got {}",
                    px[c]
                );
            }
        }
    }
}
