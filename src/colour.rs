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
//! Conversion mirrors the libvips route table: every supported space
//! converts to and from CIE XYZ (D65-relative, `Y` white = 100), and a
//! conversion from space `A` to space `B` runs `A -> XYZ -> B`. All
//! intermediate maths is `f64`; quantisation happens only where libvips
//! quantises, i.e. when a space is stored at 8 or 16 bits (`srgb`, `hsv`,
//! `cmyk`, `b-w` at 8 bits; `rgb16`, `grey16` at 16 bits) and inside the
//! XYZ -> HSV step, which passes through 8-bit sRGB exactly as the libvips
//! route does. The individual conversions use the same published formulas
//! and constants as libvips:
//!
//! * sRGB gamma per IEC 61966-2-1 (linear below 0.04045 / 0.0031308);
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
//!   (`L * 32767/100`, `a`,`b * 256`). There is no signed 16-bit
//!   [`PixelFormat`], so the samples are carried in a float raster whose
//!   values match the libvips LabS codes;
//! * Oklab / OkLCh per Ottosson's published matrices (the same constants
//!   libvips uses);
//! * Yxy chromaticity, HSV over 8-bit sRGB (hue circle mapped to 0..255),
//!   and the libvips no-lcms CMYK approximation (naive ink model over
//!   D65-normalised XYZ);
//! * mono (`b-w`, `grey16`) as gamma-encoded CIE linear luminance
//!   (0.2126 R + 0.7152 G + 0.0722 B), and grey sources replicated to RGB
//!   exactly like the libvips `BW2sRGB` route.
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
//! * The packed `labq` coding and the `histogram` / `fourier` /
//!   `multiband` pseudo-interpretations have no colourspace route, exactly
//!   as in libvips: they yield [`ColourError::UnsupportedColourspace`].
//! * libvips built with lcms converts `cmyk` through an embedded generic
//!   CMYK profile; the ported CMYK tests target the no-lcms approximation,
//!   which is what [`Raster::colourspace`] implements. Profiled CMYK is
//!   available through [`Raster::icc_import`] with a CMYK profile.

use std::path::{Path, PathBuf};
use std::str::FromStr;

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

/// sRGB opto-electrical transfer: linear 0..1 to encoded 0..1.
fn srgb_encode(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
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

/// Hue angle of `(a, b)` in degrees, wrapped to `[0, 360)` (libvips
/// `vips_col_ab2h`).
fn ab_to_h(a: f64, b: f64) -> f64 {
    let h = b.atan2(a).to_degrees();
    if h < 0.0 { h + 360.0 } else { h }
}

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
        42.5 * (secondary_diff / delta) + wrap_around_hue
    };
    [h, delta * 255.0 / c_max, c_max]
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

fn lab_to_labs(lab: [f64; 3]) -> [f64; 3] {
    [
        (lab[0] * LABS_L_SCALE).clamp(0.0, 32767.0),
        (lab[1] * LABS_AB_SCALE).clamp(-32768.0, 32767.0),
        (lab[2] * LABS_AB_SCALE).clamp(-32768.0, 32767.0),
    ]
}

fn labs_to_lab(labs: [f64; 3]) -> [f64; 3] {
    [
        labs[0] / LABS_L_SCALE,
        labs[1] / LABS_AB_SCALE,
        labs[2] / LABS_AB_SCALE,
    ]
}

// --- Mono (gamma-encoded CIE linear luminance, libvips scRGB2BW) ---

/// scRGB -> the normalised (0..1) gamma-encoded grey value.
fn scrgb_to_bw(rgb: [f64; 3]) -> f64 {
    let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    srgb_encode(y.clamp(0.0, 1.0))
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

/// Convert one pixel's colour bands from `space` to D65 XYZ. `v` holds
/// `space_bands(space)` samples in the space's numeric convention.
fn to_xyz(space: Interpretation, v: &[f64]) -> [f64; 3] {
    match space {
        Interpretation::Xyz => [v[0], v[1], v[2]],
        Interpretation::Lab => lab_to_xyz([v[0], v[1], v[2]], D65),
        Interpretation::Lch => lab_to_xyz(lch_to_lab([v[0], v[1], v[2]]), D65),
        Interpretation::Cmc => {
            let c = ccmc_to_c(v[1]);
            let lch = [lcmc_to_l(v[0]), c, hcmc_to_h(c, v[2])];
            lab_to_xyz(lch_to_lab(lch), D65)
        }
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
/// `space_bands(space)` output samples into the front of `out` (in the
/// space's numeric convention, unrounded; integer spaces quantise on
/// write).
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
        Interpretation::Cmc => {
            let lch = lab_to_lch(xyz_to_lab(xyz, D65));
            out[0] = l_to_lcmc(lch[0]);
            out[1] = c_to_ccmc(lch[1]);
            out[2] = ch_to_hcmc(lch[1], lch[2]);
        }
        Interpretation::Labs => {
            out[..3].copy_from_slice(&lab_to_labs(xyz_to_lab(xyz, D65)).map(f64::round));
        }
        Interpretation::ScRgb => out[..3].copy_from_slice(&xyz_to_scrgb(xyz)),
        Interpretation::Hsv => {
            // The libvips HSV encode goes through 8-bit sRGB.
            let rgb = xyz_to_scrgb(xyz).map(|c| (255.0 * srgb_encode(c.clamp(0.0, 1.0))).round());
            out[..3].copy_from_slice(&srgb8_to_hsv(rgb));
        }
        Interpretation::Srgb => {
            out[..3].copy_from_slice(
                &xyz_to_scrgb(xyz).map(|c| 255.0 * srgb_encode(c.clamp(0.0, 1.0))),
            );
        }
        Interpretation::Rgb16 => {
            out[..3].copy_from_slice(
                &xyz_to_scrgb(xyz).map(|c| 65535.0 * srgb_encode(c.clamp(0.0, 1.0))),
            );
        }
        Interpretation::Yxy => out[..3].copy_from_slice(&xyz_to_yxy(xyz)),
        Interpretation::OkLab => out[..3].copy_from_slice(&xyz_to_oklab(xyz)),
        Interpretation::OkLch => out[..3].copy_from_slice(&lab_to_lch(xyz_to_oklab(xyz))),
        Interpretation::Bw => out[0] = 255.0 * scrgb_to_bw(xyz_to_scrgb(xyz)),
        Interpretation::Grey16 => out[0] = 65535.0 * scrgb_to_bw(xyz_to_scrgb(xyz)),
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
                from_xyz_into(target, to_xyz(src, &src_px), &mut tgt_px);
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
     * Tests the LabS code scaling: Lab [50,0,0] stores as L ~16384 (of
     * 32767) with a,b at 0, matching the libvips signed-16-bit codes.
     */
    #[test]
    fn labs_code_scaling() {
        let labs = lab_fixture().colourspace(Interpretation::Labs);
        assert_eq!(labs.interpretation(), Interpretation::Labs);
        let px = labs.getpoint(0, 0);
        assert!(
            (px[0] - 16384.0).abs() < 1.0,
            "L code should be ~16384, got {}",
            px[0]
        );
        assert!(px[1].abs() < 1.0 && px[2].abs() < 1.0);
        assert!((px[3] - 42.0).abs() < 1e-6, "extra band untouched");
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
}
