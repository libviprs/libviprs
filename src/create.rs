//! Image generators ported from libvips (the `create` operation family).
//!
//! This module is the create/generator batch of the libvips operation
//! surface required by the ported integration tests (libviprs-tests issue
//! #55, `tests/ported_create.rs`): synthetic test patterns, seeded
//! procedural noise, look-up-table builders, frequency-domain filter
//! masks, signed-distance fields, and text rendering. Everything here is
//! an associated constructor on [`Raster`]; the small generators that
//! earlier batches already shipped ([`Raster::grey`], [`Raster::identity`],
//! [`Raster::identity_ushort`] in [`crate::conversion`]) are reused, not
//! redefined.
//!
//! Each fallible operation exists in two forms, following the convention
//! set by [`crate::conversion`]:
//!
//! * a fallible `try_*` associated function returning
//!   `Result<Raster, CreateError>`, the primary implementation with typed
//!   errors for bad control-point matrices, unknown SDF shapes, and text
//!   with nothing to render; and
//! * a panicking convenience form matching the ported-test call surface
//!   (`black`, `perlin`, `mask_ideal`, ...) exactly.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::black`] / [`Raster::black_bands`] | `vips_black` | all-zero uchar image |
//! | [`Raster::xyz`] | `vips_xyz` | 2-band coordinate ramp, `pixel(x,y) == [x, y]` |
//! | [`Raster::eye`] | `vips_eye` | eye/frequency test pattern, -1..1 (or uchar 0..255) |
//! | [`Raster::zone`] | `vips_zone` | zone plate, -1..1 |
//! | [`Raster::sines`] | `vips_sines` | 2D sine grating, -1..1 |
//! | [`Raster::gaussnoise`] | `vips_gaussnoise` | seeded Gaussian noise |
//! | [`Raster::perlin`] | `vips_perlin` | seeded Perlin gradient noise |
//! | [`Raster::worley`] | `vips_worley` | seeded Worley cellular noise |
//! | [`Raster::fractsurf`] | `vips_fractsurf` | fractal surface (noise x fractal mask via DFT) |
//! | [`Raster::buildlut`] | `vips_buildlut` | piece-wise linear LUT from control points |
//! | [`Raster::tonelut`] | `vips_tonelut` | print tone curve, 32768x1 ushort |
//! | [`Raster::from_matrix`] | `vips_image_new_matrix` | matrix image from rows of `f64` |
//! | [`Raster::mask_ideal`] (+`_ring`, `_band`) | `vips_mask_ideal*` | ideal highpass/ring/band mask |
//! | [`Raster::mask_gaussian`] (+`_ring`, `_band`) | `vips_mask_gaussian*` | Gaussian masks |
//! | [`Raster::mask_butterworth`] (+`_ring`, `_band`) | `vips_mask_butterworth*` | Butterworth masks |
//! | [`Raster::mask_fractal`] | `vips_mask_fractal` | fractal power-spectrum mask |
//! | [`Raster::sdf`] | `vips_sdf` | signed distance field (circle/box/rounded-box/line) |
//! | [`Raster::text`] | `vips_text` | rendered text, Gray8 coverage |
//!
//! # Parity notes
//!
//! * **Sample carriers.** libvips emits `float` for the point/mask/noise
//!   generators, `double` for `buildlut` and matrices, `uint` for `xyz`,
//!   and `ushort` for `tonelut`. libviprs carries every float and double
//!   surface in [`PixelFormat::FloatF32`] (values are computed in `f64`
//!   and stored as `f32`, exactly the precision libvips' float outputs
//!   have) and `xyz` coordinates in `FloatF32` too, where they are exact
//!   for any dimension below 2^24. `tonelut` is `Gray16`, matching
//!   libvips' ushort.
//! * **The mask DC rule.** In libvips (`create/mask.c`) the DC component
//!   of every frequency mask is forced to `1.0` *unless* `nodc` is set;
//!   with `nodc` the DC value comes from the filter formula itself (0 for
//!   a Butterworth or Gaussian highpass, a small positive value for the
//!   ring masks). `nodc` does not blindly write a `0`.
//! * **Mask layout.** Masks are generated in FFT layout (DC at the
//!   origin) by default; `optical` skips the quadrant rotation so DC sits
//!   at the image centre.
//! * **uchar scaling.** Generators with a `uchar` flag mirror libvips
//!   `vips_linear1(..., "uchar", TRUE)`: `v * 255/(max-min) - min*255/(max-min)`
//!   computed in `f32`, clipped to `0..=255`, then *truncated* (C's
//!   float-to-uchar cast), not rounded.
//! * **Noise seeding.** The noise generators reproduce libvips'
//!   FNV-based per-pixel hash (`vips__random` / `vips__random_add` in
//!   `iofuncs/util.c`) bit for bit, so a given seed produces the same
//!   image libvips produces for that seed. libvips defaults the seed to a
//!   random per-process value; libviprs defaults to `0` so unseeded calls
//!   are deterministic. Use the `*_seeded` forms to control the seed.
//! * **`fractsurf`.** Mirrors the libvips pipeline `vips_gaussnoise` ->
//!   `vips_freqmult` with a `vips_mask_fractal` mask: forward DFT scaled
//!   by `1/(w*h)` (libvips `fwfft`), multiply by the mask, unscaled
//!   inverse DFT, real part. The DFT is an internal row-column transform;
//!   the public `fwfft` operation belongs to a later batch.
//! * **`text`.** libvips renders through Pango/fontconfig with whatever
//!   host fonts exist; glyph-exact parity is out of scope (same rationale
//!   as ICC and codec parity). libviprs renders real text deterministically
//!   with a pure-Rust rasteriser ([`ab_glyph`]) and the bundled Bitstream
//!   Vera Sans face (`fonts/Vera.ttf`, licence in
//!   `fonts/VERA-COPYRIGHT.TXT`) at libvips' default 12pt, honouring
//!   `dpi`, wrapping (`word`, `char`, `word-char`, `none`) at `width`,
//!   and the width x height auto-fit search. Every assertion the ported
//!   suite makes (dimensions, coverage range, auto-fit width) holds.
//!
//! # Metadata
//!
//! Generators stamp the interpretation libvips stamps: float masks are
//! [`Interpretation::Fourier`], LUTs ([`Raster::buildlut`],
//! [`Raster::tonelut`]) are [`Interpretation::Histogram`],
//! [`Raster::from_matrix`] is [`Interpretation::Matrix`], and
//! [`Raster::sdf`] is [`Interpretation::Bw`]. The remaining generators
//! keep the default metadata, like the earlier constructor batches.

use std::f64::consts::PI;
use std::num::NonZeroU16;

use ab_glyph::{Font, FontRef, Glyph, PxScale, ScaleFont, point as ab_point};
use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};

/// Typed error for the create/generator surface.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CreateError {
    /// The underlying raster allocation failed (zero dimension, byte
    /// budget, or allocator failure).
    #[error(transparent)]
    Raster(#[from] RasterError),
    /// A band count outside the supported range was requested.
    #[error("{op}: unsupported band count {bands}")]
    InvalidBandCount { op: &'static str, bands: u32 },
    /// A control-point or matrix argument had no rows.
    #[error("{op}: the matrix must have at least one row")]
    EmptyMatrix { op: &'static str },
    /// A control-point or matrix argument had rows of unequal length.
    #[error(
        "{op}: matrix rows must all have the same length (row {row} has {got} values, expected {expected})"
    )]
    RaggedMatrix {
        op: &'static str,
        row: usize,
        got: usize,
        expected: usize,
    },
    /// `buildlut` needs an input column plus at least one output column.
    #[error(
        "buildlut: control points need an input column and at least one output column (got {columns})"
    )]
    TooFewColumns { columns: usize },
    /// `buildlut` input (column 0) values must be integers, like libvips.
    #[error("buildlut: x value in row {row} is not an integer: {value}")]
    NonIntegerInput { row: usize, value: f64 },
    /// An SDF shape name outside the supported set.
    #[error(
        "sdf: unknown shape {shape:?} (expected \"circle\", \"box\", \"rounded-box\", or \"line\")"
    )]
    UnknownSdfShape { shape: String },
    /// An SDF shape was missing a required parameter.
    #[error("sdf: shape \"{shape}\" needs parameter `{param}`")]
    MissingSdfParam {
        shape: &'static str,
        param: &'static str,
    },
    /// A text wrap mode outside the supported set.
    #[error(
        "text: unknown wrap mode {mode:?} (expected \"word\", \"char\", \"word-char\", or \"none\")"
    )]
    UnknownWrapMode { mode: String },
    /// The text argument rendered no ink (empty or whitespace-only).
    #[error("text: no text to render")]
    EmptyText,
}

/// Unwrap helper for the panicking op forms, mirroring the ported-test
/// "known-good input" contract.
#[track_caller]
fn expect_create<T>(op: &str, r: Result<T, CreateError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// The single-band `f32` carrier used by every float generator.
fn float1() -> PixelFormat {
    PixelFormat::FloatF32(NonZeroU16::new(1).expect("1 is non-zero"))
}

/// Append `v` as one native-endian sample of `bpc` bytes: a C-style cast
/// into the 8- or 16-bit unsigned range, or an `f32` for the 4-byte float
/// depth. These are exactly the depths [`PixelFormat::with_channels`]
/// accepts, so `bpc` is always 1, 2, or 4. Rust's saturating float-to-int
/// cast reproduces libvips' clip-then-truncate `vips_cast` behaviour.
fn push_constant_sample(out: &mut Vec<u8>, bpc: usize, v: f64) {
    match bpc {
        1 => out.push(v as u8),
        2 => out.extend_from_slice(&(v as u16).to_ne_bytes()),
        _ => out.extend_from_slice(&(v as f32).to_ne_bytes()),
    }
}

/// Build a single-band float raster by evaluating `f` at every pixel.
fn float1_from_fn(
    width: u32,
    height: u32,
    f: impl Fn(u32, u32) -> f32,
) -> Result<Raster, CreateError> {
    let mut out = Raster::zeroed(width, height, float1())?;
    let w = width as usize;
    let data = out.data_mut();
    for y in 0..height {
        for x in 0..width {
            let off = (y as usize * w + x as usize) * 4;
            data[off..off + 4].copy_from_slice(&f(x, y).to_ne_bytes());
        }
    }
    Ok(out)
}

/// libvips' point-generator base (`create/point.c`): evaluate `point` at
/// every pixel as `f32`; with `uchar`, apply the `vips_linear1(...,
/// "uchar")` mapping from the class range `[min, max]` to `0..=255`
/// (computed in `f32`, clipped, then truncated exactly like the C cast).
fn point_image(
    width: u32,
    height: u32,
    uchar: bool,
    class_min: f32,
    class_max: f32,
    point: impl Fn(u32, u32) -> f64,
) -> Result<Raster, CreateError> {
    if !uchar {
        return float1_from_fn(width, height, |x, y| point(x, y) as f32);
    }
    let range = class_max - class_min;
    let a = 255.0f32 / range;
    let b = -class_min * 255.0 / range;
    let mut out = Raster::zeroed(width, height, PixelFormat::Gray8)?;
    let w = width as usize;
    let data = out.data_mut();
    for y in 0..height {
        for x in 0..width {
            let v = point(x, y) as f32;
            let t = (a * v + b).clamp(0.0, 255.0);
            // C float-to-uchar conversion truncates toward zero.
            data[y as usize * w + x as usize] = t as u8;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// libvips' pixel-hash RNG (iofuncs/util.c), reproduced bit for bit.
// ---------------------------------------------------------------------

/// `vips__random_add`: fold the four bytes of `value` into an FNV hash.
fn random_add(mut hash: u32, value: u32) -> u32 {
    for shift in [0u32, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

/// `vips__random`: rehash a seed through the FNV basis.
fn vips_random(seed: u32) -> u32 {
    random_add(2_166_136_261, seed)
}

impl Raster {
    // -----------------------------------------------------------------
    // Basic generators
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::black_bands`].
    ///
    /// # Errors
    ///
    /// [`CreateError::InvalidBandCount`] for a band count of zero or
    /// above `u16::MAX`, or [`CreateError::Raster`] on allocation
    /// failure.
    pub fn try_black_bands(width: u32, height: u32, bands: u32) -> Result<Raster, CreateError> {
        let format = usize::try_from(bands)
            .ok()
            .and_then(|b| PixelFormat::with_channels(b, 1))
            .ok_or(CreateError::InvalidBandCount { op: "black", bands })?;
        Ok(Raster::zeroed(width, height, format)?)
    }

    /// Create an all-zero uchar image with `bands` bands (libvips
    /// `vips_black` with `bands` set). One band is `Gray8`, three
    /// `Rgb8`, four `Rgba8`, anything else the 8-bit multiband carrier.
    /// Panicking form of [`Raster::try_black_bands`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_black_bands`].
    #[track_caller]
    pub fn black_bands(width: u32, height: u32, bands: u32) -> Raster {
        expect_create("black", Self::try_black_bands(width, height, bands))
    }

    /// Create a single-band all-zero `Gray8` image (libvips `vips_black`
    /// with the default `bands: 1`).
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_black_bands`].
    #[track_caller]
    pub fn black(width: u32, height: u32) -> Raster {
        Self::black_bands(width, height, 1)
    }

    /// Fallible form of [`Raster::new_from_image`].
    ///
    /// # Errors
    ///
    /// [`CreateError::InvalidBandCount`] when `value` is empty or longer
    /// than `u16::MAX` (no [`PixelFormat`] carries that band count), or
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_new_from_image(&self, value: &[f64]) -> Result<Raster, CreateError> {
        let bands = value.len();
        // libvips `vips_image_new_from_image` keeps the source `BandFmt`
        // (sample depth) and only changes the band count to `value.len()`.
        let bpc = self.format().bytes_per_channel();
        let format = PixelFormat::with_channels(bands, bpc).ok_or_else(|| {
            CreateError::InvalidBandCount {
                op: "new_from_image",
                bands: u32::try_from(bands).unwrap_or(u32::MAX),
            }
        })?;
        // One pixel's native-endian sample bytes, then replicated across
        // every pixel so `avg()` equals the mean of `value`.
        let mut pixel = Vec::with_capacity(bands * bpc);
        for &v in value {
            push_constant_sample(&mut pixel, bpc, v);
        }
        let mut out = Raster::zeroed(self.width(), self.height(), format)?;
        for chunk in out.data_mut().chunks_exact_mut(pixel.len()) {
            chunk.copy_from_slice(&pixel);
        }
        // Carry the source geometry/metadata (libvips copies the header),
        // pinning the resolved interpretation so it survives the band-count
        // change even when the source left it inferred from the format.
        out.meta = self.meta;
        out.meta.interpretation = Some(self.interpretation());
        Ok(out)
    }

    /// Create a constant image the size of `self`, filled with `value`
    /// (libvips `vips_image_new_from_image`).
    ///
    /// The result keeps `self`'s width, height, sample depth,
    /// interpretation, resolution, and offsets; only the band count
    /// changes, to `value.len()`. Every pixel is set to `value`
    /// (per channel), so a scalar produces a 1-band image with
    /// `avg() == value[0]`, and an N-element vector produces an N-band
    /// image with `avg()` equal to the mean of `value`. Panicking form of
    /// [`Raster::try_new_from_image`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_new_from_image`].
    #[track_caller]
    pub fn new_from_image(&self, value: &[f64]) -> Raster {
        expect_create("new_from_image", self.try_new_from_image(value))
    }

    /// Fallible form of [`Raster::xyz`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_xyz(width: u32, height: u32) -> Result<Raster, CreateError> {
        let format = PixelFormat::FloatF32(NonZeroU16::new(2).expect("2 is non-zero"));
        let mut out = Raster::zeroed(width, height, format)?;
        let w = width as usize;
        let data = out.data_mut();
        for y in 0..height {
            for x in 0..width {
                let off = (y as usize * w + x as usize) * 8;
                data[off..off + 4].copy_from_slice(&(x as f32).to_ne_bytes());
                data[off + 4..off + 8].copy_from_slice(&(y as f32).to_ne_bytes());
            }
        }
        Ok(out)
    }

    /// Create a two-band coordinate image where `pixel(x, y) == [x, y]`
    /// (libvips `vips_xyz` with the default two dimensions). libvips
    /// emits `uint` samples; libviprs carries the coordinates in
    /// [`PixelFormat::FloatF32`], where they are exact below 2^24.
    /// Panicking form of [`Raster::try_xyz`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_xyz`].
    #[track_caller]
    pub fn xyz(width: u32, height: u32) -> Raster {
        expect_create("xyz", Self::try_xyz(width, height))
    }

    /// Fallible form of [`Raster::eye`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_eye(width: u32, height: u32, uchar: bool) -> Result<Raster, CreateError> {
        // libvips create/eye.c with the default factor 0.5:
        // point(x, y) = y^2 * cos(c * x^2) / max_y^2,
        // c = factor * pi / (2 * max_x).
        let factor = 0.5f64;
        let max_x = f64::from(width.saturating_sub(1).max(1));
        let max_y = f64::from(height.saturating_sub(1).max(1));
        let c = factor * PI / (2.0 * max_x);
        let h = max_y * max_y;
        point_image(width, height, uchar, -1.0, 1.0, |x, y| {
            let (x, y) = (f64::from(x), f64::from(y));
            y * y * (c * x * x).cos() / h
        })
    }

    /// Create an "eye" (frequency response test) pattern (libvips
    /// `vips_eye`, default `factor: 0.5`). Float output spans -1.0 to
    /// 1.0; with `uchar`, mapped to `0..=255`. Panicking form of
    /// [`Raster::try_eye`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_eye`].
    #[track_caller]
    pub fn eye(width: u32, height: u32, uchar: bool) -> Raster {
        expect_create("eye", Self::try_eye(width, height, uchar))
    }

    /// Fallible form of [`Raster::zone`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_zone(width: u32, height: u32) -> Result<Raster, CreateError> {
        // libvips create/zone.c: cos(pi/width * ((x - w/2)^2 + (y - h/2)^2)).
        let hw = i64::from(width / 2);
        let hh = i64::from(height / 2);
        let c = PI / f64::from(width);
        point_image(width, height, false, -1.0, 1.0, |x, y| {
            let h2 = (i64::from(x) - hw) * (i64::from(x) - hw);
            let v2 = (i64::from(y) - hh) * (i64::from(y) - hh);
            (c * (v2 + h2) as f64).cos()
        })
    }

    /// Create a zone-plate test pattern (libvips `vips_zone`), a
    /// single-band float image spanning -1.0 to 1.0. Panicking form of
    /// [`Raster::try_zone`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_zone`].
    #[track_caller]
    pub fn zone(width: u32, height: u32) -> Raster {
        expect_create("zone", Self::try_zone(width, height))
    }

    /// Fallible form of [`Raster::sines`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_sines(width: u32, height: u32) -> Result<Raster, CreateError> {
        // libvips create/sines.c with the default hfreq = vfreq = 0.5:
        // theta = atan(vfreq/hfreq), c = |freq| * 2pi / width,
        // point(x, y) = cos(c * (x cos(theta) - y sin(theta))).
        let (hfreq, vfreq) = (0.5f64, 0.5f64);
        let theta = if hfreq == 0.0 {
            PI / 2.0
        } else {
            (vfreq / hfreq).atan()
        };
        let factor = (hfreq * hfreq + vfreq * vfreq).sqrt();
        let (costheta, sintheta) = (theta.cos(), theta.sin());
        let c = factor * PI * 2.0 / f64::from(width);
        point_image(width, height, false, -1.0, 1.0, |x, y| {
            (c * (f64::from(x) * costheta - f64::from(y) * sintheta)).cos()
        })
    }

    /// Create a 2D sine grating (libvips `vips_sines`, default
    /// horizontal and vertical frequencies of 0.5 cycles per image), a
    /// single-band float image spanning -1.0 to 1.0. Panicking form of
    /// [`Raster::try_sines`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_sines`].
    #[track_caller]
    pub fn sines(width: u32, height: u32) -> Raster {
        expect_create("sines", Self::try_sines(width, height))
    }

    // -----------------------------------------------------------------
    // Noise generators
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::gaussnoise_seeded`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_gaussnoise_seeded(
        width: u32,
        height: u32,
        mean: f64,
        sigma: f64,
        seed: u32,
    ) -> Result<Raster, CreateError> {
        // libvips create/gaussnoise.c: each pixel hashes (seed, x, y)
        // then sums 12 uniform samples, giving an Irwin-Hall
        // approximation of a unit Gaussian.
        float1_from_fn(width, height, |x, y| {
            let mut s = random_add(seed, x);
            s = random_add(s, y);
            let mut sum = 0.0f64;
            for _ in 0..12 {
                s = vips_random(s);
                sum += f64::from(s) / f64::from(u32::MAX);
            }
            ((sum - 6.0) * sigma + mean) as f32
        })
    }

    /// [`Raster::gaussnoise`] with an explicit seed; the same seed
    /// reproduces the same image libvips produces for that seed value.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_gaussnoise_seeded`].
    #[track_caller]
    pub fn gaussnoise_seeded(width: u32, height: u32, mean: f64, sigma: f64, seed: u32) -> Raster {
        expect_create(
            "gaussnoise",
            Self::try_gaussnoise_seeded(width, height, mean, sigma, seed),
        )
    }

    /// Create a single-band float image of Gaussian noise with the given
    /// mean and standard deviation (libvips `vips_gaussnoise`). libvips
    /// defaults the seed to a random per-process value; libviprs pins it
    /// to `0` so unseeded output is deterministic. See
    /// [`Raster::gaussnoise_seeded`].
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_gaussnoise_seeded`].
    #[track_caller]
    pub fn gaussnoise(width: u32, height: u32, mean: f64, sigma: f64) -> Raster {
        Self::gaussnoise_seeded(width, height, mean, sigma, 0)
    }

    /// Fallible form of [`Raster::perlin_seeded`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_perlin_seeded(width: u32, height: u32, seed: u32) -> Result<Raster, CreateError> {
        // libvips create/perlin.c with the default cell_size 256.
        const CELL: i64 = 256;
        let mut cos_table = [0.0f32; 256];
        let mut sin_table = [0.0f32; 256];
        for (i, (c, s)) in cos_table.iter_mut().zip(sin_table.iter_mut()).enumerate() {
            let angle = 2.0 * PI * i as f64 / 256.0;
            *c = angle.cos() as f32;
            *s = angle.sin() as f32;
        }
        let cells_across = (i64::from(width) + CELL - 1) / CELL;
        let cells_down = (i64::from(height) + CELL - 1) / CELL;

        // The 2x2 grid of corner gradients for the cell at (cell_x, cell_y).
        let gradients = |cell_x: i64, cell_y: i64| -> ([f32; 4], [f32; 4]) {
            let mut gx = [0.0f32; 4];
            let mut gy = [0.0f32; 4];
            for cy_off in 0..2i64 {
                for cx_off in 0..2i64 {
                    let ci = (cx_off + cy_off * 2) as usize;
                    let mut cx = cell_x + cx_off;
                    let mut cy = cell_y + cy_off;
                    // Wrap the trailing edge so tessellating sizes tile,
                    // exactly as perlin.c does (y is hashed first).
                    if cy >= cells_down {
                        cy = 0;
                    }
                    let mut s = random_add(seed, cy as u32);
                    if cx >= cells_across {
                        cx = 0;
                    }
                    s = random_add(s, cx as u32);
                    let angle = ((s ^ (s >> 8) ^ (s >> 16)) & 0xff) as usize;
                    gx[ci] = cos_table[angle];
                    gy[ci] = sin_table[angle];
                }
            }
            (gx, gy)
        };

        fn smootherstep(x: f32) -> f32 {
            x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
        }

        // One-entry cell cache, like perlin.c's per-sequence cache.
        let mut cache_key = (-1i64, -1i64);
        let (mut gx, mut gy) = ([0.0f32; 4], [0.0f32; 4]);
        let mut out = Raster::zeroed(width, height, float1())?;
        let w = width as usize;
        let data = out.data_mut();
        for y in 0..height {
            for x in 0..width {
                let cell_x = i64::from(x) / CELL;
                let cell_y = i64::from(y) / CELL;
                let dx = (i64::from(x) - cell_x * CELL) as f32 / CELL as f32;
                let dy = (i64::from(y) - cell_y * CELL) as f32 / CELL as f32;
                let sx = smootherstep(dx);
                let sy = smootherstep(dy);
                if cache_key != (cell_x, cell_y) {
                    (gx, gy) = gradients(cell_x, cell_y);
                    cache_key = (cell_x, cell_y);
                }

                let n0 = -dx * gx[0] + -dy * gy[0];
                let n1 = (1.0 - dx) * gx[1] + -dy * gy[1];
                let ix0 = n0 + sx * (n1 - n0);
                let n0 = -dx * gx[2] + (1.0 - dy) * gy[2];
                let n1 = (1.0 - dx) * gx[3] + (1.0 - dy) * gy[3];
                let ix1 = n0 + sx * (n1 - n0);
                let p = ix0 + sy * (ix1 - ix0);

                let off = (y as usize * w + x as usize) * 4;
                data[off..off + 4].copy_from_slice(&p.to_ne_bytes());
            }
        }
        Ok(out)
    }

    /// [`Raster::perlin`] with an explicit seed; the same seed
    /// reproduces the same image libvips produces for that seed value.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_perlin_seeded`].
    #[track_caller]
    pub fn perlin_seeded(width: u32, height: u32, seed: u32) -> Raster {
        expect_create("perlin", Self::try_perlin_seeded(width, height, seed))
    }

    /// Create single-band float Perlin (gradient) noise with libvips'
    /// default 256-pixel cells (libvips `vips_perlin`). Values span
    /// roughly -1.0 to 1.0 and are exactly `0.0` on cell corners.
    /// libvips defaults the seed to a random per-process value; libviprs
    /// pins it to `0`. See [`Raster::perlin_seeded`].
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_perlin_seeded`].
    #[track_caller]
    pub fn perlin(width: u32, height: u32) -> Raster {
        Self::perlin_seeded(width, height, 0)
    }

    /// Fallible form of [`Raster::worley_seeded`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_worley_seeded(width: u32, height: u32, seed: u32) -> Result<Raster, CreateError> {
        // libvips create/worley.c with the default cell_size 256.
        const CELL: i64 = 256;
        const MAX_FEATURES: usize = 10;
        let cells_across = (i64::from(width) + CELL - 1) / CELL;
        let cells_down = (i64::from(height) + CELL - 1) / CELL;

        struct Cell {
            n_features: usize,
            fx: [i64; MAX_FEATURES],
            fy: [i64; MAX_FEATURES],
        }

        let make_cell = |cell_x: i64, cell_y: i64| -> Cell {
            // Wrap out-of-range neighbours to the opposite edge, exactly
            // as worley.c does (x is hashed first).
            let wrap_x = if cell_x >= cells_across {
                0
            } else if cell_x < 0 {
                cells_across - 1
            } else {
                cell_x
            };
            let mut s = random_add(seed, wrap_x as u32);
            let wrap_y = if cell_y >= cells_down {
                0
            } else if cell_y < 0 {
                cells_down - 1
            } else {
                cell_y
            };
            s = random_add(s, wrap_y as u32);
            let n_features = (s % (MAX_FEATURES as u32 - 1)) as usize + 1;
            let mut cell = Cell {
                n_features,
                fx: [0; MAX_FEATURES],
                fy: [0; MAX_FEATURES],
            };
            for j in 0..n_features {
                s = vips_random(s);
                cell.fx[j] = cell_x * CELL + i64::from(s % CELL as u32);
                s = vips_random(s);
                cell.fy[j] = cell_y * CELL + i64::from(s % CELL as u32);
            }
            cell
        };

        let mut cache_key = (-1i64, -1i64);
        let mut cells: Vec<Cell> = Vec::new();
        let mut out = Raster::zeroed(width, height, float1())?;
        let w = width as usize;
        let data = out.data_mut();
        for y in 0..height {
            for x in 0..width {
                let cell_x = i64::from(x) / CELL;
                let cell_y = i64::from(y) / CELL;
                if cache_key != (cell_x, cell_y) {
                    cells.clear();
                    for cy in 0..3i64 {
                        for cx in 0..3i64 {
                            cells.push(make_cell(cell_x + cx - 1, cell_y + cy - 1));
                        }
                    }
                    cache_key = (cell_x, cell_y);
                }
                let mut distance = (CELL as f32) * 1.5;
                for cell in &cells {
                    for j in 0..cell.n_features {
                        let ddx = i64::from(x) - cell.fx[j];
                        let ddy = i64::from(y) - cell.fy[j];
                        let d = ((ddx * ddx + ddy * ddy) as f64).sqrt() as f32;
                        distance = distance.min(d);
                    }
                }
                let off = (y as usize * w + x as usize) * 4;
                data[off..off + 4].copy_from_slice(&distance.to_ne_bytes());
            }
        }
        Ok(out)
    }

    /// [`Raster::worley`] with an explicit seed; the same seed
    /// reproduces the same image libvips produces for that seed value.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_worley_seeded`].
    #[track_caller]
    pub fn worley_seeded(width: u32, height: u32, seed: u32) -> Raster {
        expect_create("worley", Self::try_worley_seeded(width, height, seed))
    }

    /// Create single-band float Worley (cellular) noise with libvips'
    /// default 256-pixel cells (libvips `vips_worley`): each sample is
    /// the distance to the nearest feature point of the surrounding
    /// 3x3 cells. libvips defaults the seed to a random per-process
    /// value; libviprs pins it to `0`. See [`Raster::worley_seeded`].
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_worley_seeded`].
    #[track_caller]
    pub fn worley(width: u32, height: u32) -> Raster {
        Self::worley_seeded(width, height, 0)
    }

    /// Fallible form of [`Raster::fractsurf`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_fractsurf(
        width: u32,
        height: u32,
        fractal_dimension: f64,
    ) -> Result<Raster, CreateError> {
        // libvips create/fractsurf.c: gaussnoise(mean 0, sigma 1)
        // frequency-multiplied by mask_fractal (freqfilt/freqmult.c):
        // fwfft (scaled 1/(w*h)) -> multiply -> unscaled invfft, real part.
        let noise = Raster::try_gaussnoise_seeded(width, height, 0.0, 1.0, 0)?;
        let mask = Raster::try_mask_fractal(width, height, fractal_dimension)?;
        let (w, h) = (width as usize, height as usize);
        let noise_samples = noise.f32_samples().expect("gaussnoise is float");
        let mask_samples = mask.f32_samples().expect("mask_fractal is float");

        let mut buf: Vec<(f64, f64)> = noise_samples.iter().map(|&v| (f64::from(v), 0.0)).collect();
        dft_2d(&mut buf, w, h, false);
        let size = (w * h) as f64;
        for (c, &m) in buf.iter_mut().zip(mask_samples.iter()) {
            c.0 = c.0 / size * f64::from(m);
            c.1 = c.1 / size * f64::from(m);
        }
        dft_2d(&mut buf, w, h, true);
        float1_from_fn(width, height, |x, y| {
            buf[y as usize * w + x as usize].0 as f32
        })
    }

    /// Create a fractal surface with the given fractal dimension
    /// (libvips `vips_fractsurf`, dimension between 2 and 3): Gaussian
    /// noise shaped by a [`Raster::mask_fractal`] power spectrum.
    /// Deterministic with the pinned default seed. Panicking form of
    /// [`Raster::try_fractsurf`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_fractsurf`].
    #[track_caller]
    pub fn fractsurf(width: u32, height: u32, fractal_dimension: f64) -> Raster {
        expect_create(
            "fractsurf",
            Self::try_fractsurf(width, height, fractal_dimension),
        )
    }
}

/// Naive O(n^2) 1D DFT used by the internal 2D transform. Sizes here are
/// the create-generator sizes (hundreds), where the row-column transform
/// costs w*h*(w+h) flops, comfortably below a decoder's budget.
fn dft_1d(inp: &[(f64, f64)], out: &mut [(f64, f64)], twiddle: &[(f64, f64)]) {
    let n = inp.len();
    for (k, o) in out.iter_mut().enumerate() {
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for (j, &(r, i)) in inp.iter().enumerate() {
            let (tr, ti) = twiddle[(k * j) % n];
            sr += r * tr - i * ti;
            si += r * ti + i * tr;
        }
        *o = (sr, si);
    }
}

/// Twiddle factors `e^(-+2*pi*i*k/n)` for [`dft_1d`].
fn dft_twiddles(n: usize, inverse: bool) -> Vec<(f64, f64)> {
    let sign = if inverse { 2.0 } else { -2.0 };
    (0..n)
        .map(|k| {
            let a = sign * PI * k as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect()
}

/// In-place row-column 2D DFT over `buf` (row-major, `h` rows of `w`).
/// Unnormalised in both directions; [`Raster::try_fractsurf`] applies the
/// libvips `fwfft` 1/(w*h) scale itself.
fn dft_2d(buf: &mut [(f64, f64)], w: usize, h: usize, inverse: bool) {
    let tw = dft_twiddles(w, inverse);
    let mut row_out = vec![(0.0, 0.0); w];
    for y in 0..h {
        dft_1d(&buf[y * w..(y + 1) * w], &mut row_out, &tw);
        buf[y * w..(y + 1) * w].copy_from_slice(&row_out);
    }
    let th = dft_twiddles(h, inverse);
    let mut col_in = vec![(0.0, 0.0); h];
    let mut col_out = vec![(0.0, 0.0); h];
    for x in 0..w {
        for y in 0..h {
            col_in[y] = buf[y * w + x];
        }
        dft_1d(&col_in, &mut col_out, &th);
        for y in 0..h {
            buf[y * w + x] = col_out[y];
        }
    }
}

/// Validate a rows-of-`f64` matrix argument: at least one row, all rows
/// the same length. Returns the column count.
fn check_matrix(op: &'static str, rows: &[Vec<f64>]) -> Result<usize, CreateError> {
    let Some(first) = rows.first() else {
        return Err(CreateError::EmptyMatrix { op });
    };
    let expected = first.len();
    for (row, r) in rows.iter().enumerate() {
        if r.len() != expected {
            return Err(CreateError::RaggedMatrix {
                op,
                row,
                got: r.len(),
                expected,
            });
        }
    }
    Ok(expected)
}

impl Raster {
    // -----------------------------------------------------------------
    // LUT and matrix constructors
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::buildlut`].
    ///
    /// # Errors
    ///
    /// [`CreateError::EmptyMatrix`] / [`CreateError::RaggedMatrix`] for a
    /// mis-shaped control-point matrix, [`CreateError::TooFewColumns`]
    /// when there is no output column, [`CreateError::NonIntegerInput`]
    /// when a column-0 value is not an integer (within libvips' 0.001
    /// tolerance), or [`CreateError::Raster`] on allocation failure.
    pub fn try_buildlut(control_points: &[Vec<f64>]) -> Result<Raster, CreateError> {
        let columns = check_matrix("buildlut", control_points)?;
        if columns < 2 {
            return Err(CreateError::TooFewColumns { columns });
        }
        // Column 0 must hold integers (libvips checks |v - rint(v)| > 0.001).
        for (row, r) in control_points.iter().enumerate() {
            let v = r[0];
            if !v.is_finite() || (v - v.round_ties_even()).abs() > 0.001 {
                return Err(CreateError::NonIntegerInput { row, value: v });
            }
        }
        // Sort by the input column, like buildlut.c.
        let mut rows: Vec<&Vec<f64>> = control_points.iter().collect();
        rows.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).expect("inputs are finite"));

        let xs: Vec<i64> = rows.iter().map(|r| r[0].round_ties_even() as i64).collect();
        let xlow = *xs.iter().min().expect("at least one row");
        let xhigh = *xs.iter().max().expect("at least one row");
        let lut_size = (xhigh - xlow + 1) as usize;
        let bands = columns - 1;

        let mut buf = vec![0.0f64; lut_size * bands];
        for b in 0..bands {
            for i in 0..rows.len().saturating_sub(1) {
                let x1 = xs[i];
                let x2 = xs[i + 1];
                let dx = x2 - x1;
                let y1 = rows[i][b + 1];
                let y2 = rows[i + 1][b + 1];
                let dy = y2 - y1;
                for x in 0..dx {
                    buf[b + ((x + x1 - xlow) as usize) * bands] = y1 + x as f64 * dy / dx as f64;
                }
            }
            // Inclusive: pop the final value in by hand, like buildlut.c.
            let xlast = xs[rows.len() - 1];
            buf[b + ((xlast - xlow) as usize) * bands] = rows[rows.len() - 1][b + 1];
        }

        let width = u32::try_from(lut_size).map_err(|_| {
            CreateError::Raster(RasterError::ZeroDimension {
                width: u32::MAX,
                height: 1,
            })
        })?;
        let format = PixelFormat::with_channels(bands, 4).ok_or(CreateError::InvalidBandCount {
            op: "buildlut",
            bands: bands as u32,
        })?;
        let mut data = Vec::with_capacity(buf.len() * 4);
        for v in &buf {
            data.extend_from_slice(&(*v as f32).to_ne_bytes());
        }
        let mut out = Raster::new(width, 1, format, data)?;
        out.meta.interpretation = Some(Interpretation::Histogram);
        Ok(out)
    }

    /// Build a piece-wise linear look-up table from a matrix of control
    /// points (libvips `vips_buildlut`). Column 0 holds integer input
    /// values, columns 1.. hold the output value of each band; rows may
    /// arrive unsorted. The result is a `(xmax - xmin + 1)` x 1 image
    /// with one band per output column, linearly interpolated between
    /// control points. libvips emits `double` samples; libviprs carries
    /// them in [`PixelFormat::FloatF32`]. Panicking form of
    /// [`Raster::try_buildlut`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_buildlut`].
    #[track_caller]
    pub fn buildlut(control_points: &[Vec<f64>]) -> Raster {
        expect_create("buildlut", Self::try_buildlut(control_points))
    }

    /// Fallible form of [`Raster::tonelut`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_tonelut() -> Result<Raster, CreateError> {
        // libvips create/tonelut.c with every default: in_max/out_max
        // 32767, Lb 0, Lw 100, Ps 0.2, Pm 0.5, Ph 0.8, S = M = H = 0.
        // The smoothstep blends are kept literal so future parameter
        // exposure only widens the signature.
        const IN_MAX: usize = 32767;
        const OUT_MAX: f64 = 32767.0;
        const LB: f64 = 0.0;
        const LW: f64 = 100.0;
        const PS: f64 = 0.2;
        const PM: f64 = 0.5;
        const PH: f64 = 0.8;
        const S: f64 = 0.0;
        const M: f64 = 0.0;
        const H: f64 = 0.0;
        let ls = LB + PS * (LW - LB);
        let lm = LB + PM * (LW - LB);
        let lh = LB + PH * (LW - LB);

        // The rising/falling smoothstep pair each of the three adjusters
        // uses (tonelut.c shad()/mid()/high()).
        let bump = |x: f64, lo: f64, mid: f64, hi: f64| -> f64 {
            let x1 = (x - lo) / (mid - lo);
            let x2 = (x - mid) / (hi - mid);
            if x < lo {
                0.0
            } else if x < mid {
                3.0 * x1 * x1 - 2.0 * x1 * x1 * x1
            } else if x < hi {
                1.0 - 3.0 * x2 * x2 + 2.0 * x2 * x2 * x2
            } else {
                0.0
            }
        };

        let mut data = Vec::with_capacity((IN_MAX + 1) * 2);
        for i in 0..=IN_MAX {
            let x = 100.0 * i as f64 / IN_MAX as f64;
            let tone =
                x + S * bump(x, LB, ls, lm) + M * bump(x, ls, lm, lh) + H * bump(x, lm, lh, LW);
            // C: int v = (out_max / 100.0) * tone; truncates toward zero.
            let v = ((OUT_MAX / 100.0) * tone) as i64;
            let v = v.clamp(0, OUT_MAX as i64) as u16;
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let mut out = Raster::new((IN_MAX + 1) as u32, 1, PixelFormat::Gray16, data)?;
        out.meta.interpretation = Some(Interpretation::Histogram);
        Ok(out)
    }

    /// Build the print tone-mapping curve (libvips `vips_tonelut` with
    /// every default): a monotonic 32768x1 `Gray16` LUT. Panicking form
    /// of [`Raster::try_tonelut`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_tonelut`].
    #[track_caller]
    pub fn tonelut() -> Raster {
        expect_create("tonelut", Self::try_tonelut())
    }

    /// Fallible form of [`Raster::from_matrix`].
    ///
    /// # Errors
    ///
    /// [`CreateError::EmptyMatrix`] / [`CreateError::RaggedMatrix`] for a
    /// mis-shaped matrix (including zero-length rows), or
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_from_matrix(rows: &[Vec<f64>]) -> Result<Raster, CreateError> {
        let columns = check_matrix("from_matrix", rows)?;
        if columns == 0 {
            return Err(CreateError::EmptyMatrix { op: "from_matrix" });
        }
        let width = u32::try_from(columns).map_err(|_| CreateError::InvalidBandCount {
            op: "from_matrix",
            bands: u32::MAX,
        })?;
        let height = rows.len() as u32;
        let mut data = Vec::with_capacity(rows.len() * columns * 4);
        for r in rows {
            for &v in r {
                data.extend_from_slice(&(v as f32).to_ne_bytes());
            }
        }
        let mut out = Raster::new(width, height, float1(), data)?;
        out.meta.interpretation = Some(Interpretation::Matrix);
        Ok(out)
    }

    /// Create a single-band matrix image from rows of `f64`, one image
    /// row per entry (libvips `vips_image_new_matrix` /
    /// `new_from_array`), stamped [`Interpretation::Matrix`]. libvips
    /// stores matrices as `double`; libviprs carries them in
    /// [`PixelFormat::FloatF32`]. Panicking form of
    /// [`Raster::try_from_matrix`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_from_matrix`].
    #[track_caller]
    pub fn from_matrix(rows: &[Vec<f64>]) -> Raster {
        expect_create("from_matrix", Self::try_from_matrix(rows))
    }
}

/// libvips' frequency-mask base (`create/mask.c`): map each pixel to
/// normalised frequency coordinates `(dx, dy)` in `-1..1`, apply the
/// filter formula, force DC to `1.0` unless `nodc`, and lay the result
/// out in FFT order (DC at the origin) unless `optical`. With `uchar`,
/// scale `0..1` to `0..=255` through the point-class linear map.
fn mask_image(
    width: u32,
    height: u32,
    optical: bool,
    nodc: bool,
    uchar: bool,
    point: impl Fn(f64, f64) -> f64,
) -> Result<Raster, CreateError> {
    let half_w = i64::from((width / 2).max(1));
    let half_h = i64::from((height / 2).max(1));
    let mut out = point_image(width, height, uchar, 0.0, 1.0, |x, y| {
        let (mut fx, mut fy) = (i64::from(x), i64::from(y));
        if !optical {
            fx = (fx + half_w) % i64::from(width);
            fy = (fy + half_h) % i64::from(height);
        }
        fx -= half_w;
        fy -= half_h;
        if !nodc && fx == 0 && fy == 0 {
            // The DC component is always 1 unless nodc is set.
            1.0
        } else {
            point(fx as f64 / half_w as f64, fy as f64 / half_h as f64)
        }
    })?;
    if !uchar {
        out.meta.interpretation = Some(Interpretation::Fourier);
    }
    Ok(out)
}

impl Raster {
    // -----------------------------------------------------------------
    // Frequency-domain filter masks
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::mask_ideal`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_ideal(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let fc2 = frequency_cutoff * frequency_cutoff;
        mask_image(width, height, false, nodc, false, |dx, dy| {
            if dx * dx + dy * dy <= fc2 { 0.0 } else { 1.0 }
        })
    }

    /// Create an ideal highpass mask (libvips `vips_mask_ideal`): 0
    /// inside the cutoff radius, 1 outside. Panicking form of
    /// [`Raster::try_mask_ideal`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_mask_ideal`].
    #[track_caller]
    pub fn mask_ideal(width: u32, height: u32, frequency_cutoff: f64, nodc: bool) -> Raster {
        expect_create(
            "mask_ideal",
            Self::try_mask_ideal(width, height, frequency_cutoff, nodc),
        )
    }

    /// Fallible form of [`Raster::mask_ideal_ring`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_ideal_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let df = ringwidth / 2.0;
        let fc2_1 = (frequency_cutoff - df) * (frequency_cutoff - df);
        let fc2_2 = (frequency_cutoff + df) * (frequency_cutoff + df);
        mask_image(width, height, false, nodc, false, |dx, dy| {
            let dist2 = dx * dx + dy * dy;
            if dist2 > fc2_1 && dist2 < fc2_2 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Create an ideal ring-pass mask (libvips `vips_mask_ideal_ring`):
    /// 1 inside the ring `frequency_cutoff +- ringwidth/2`, 0 elsewhere.
    /// Panicking form of [`Raster::try_mask_ideal_ring`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_ideal_ring`].
    #[track_caller]
    pub fn mask_ideal_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Raster {
        expect_create(
            "mask_ideal_ring",
            Self::try_mask_ideal_ring(width, height, frequency_cutoff, ringwidth, nodc),
        )
    }

    /// Fallible form of [`Raster::mask_ideal_band`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_ideal_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
    ) -> Result<Raster, CreateError> {
        let r2 = radius * radius;
        mask_image(width, height, false, false, false, |dx, dy| {
            let d1 = (dx - frequency_cutoff_x) * (dx - frequency_cutoff_x)
                + (dy - frequency_cutoff_y) * (dy - frequency_cutoff_y);
            let d2 = (dx + frequency_cutoff_x) * (dx + frequency_cutoff_x)
                + (dy + frequency_cutoff_y) * (dy + frequency_cutoff_y);
            if d1 < r2 || d2 < r2 { 1.0 } else { 0.0 }
        })
    }

    /// Create an ideal band-pass mask (libvips `vips_mask_ideal_band`):
    /// 1 within `radius` of `(+-fx, +-fy)`, 0 elsewhere. Panicking form
    /// of [`Raster::try_mask_ideal_band`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_ideal_band`].
    #[track_caller]
    pub fn mask_ideal_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
    ) -> Raster {
        expect_create(
            "mask_ideal_band",
            Self::try_mask_ideal_band(
                width,
                height,
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_gaussian`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_gaussian(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let cnst = amplitude_cutoff.ln();
        let fc2 = frequency_cutoff * frequency_cutoff;
        mask_image(width, height, false, nodc, false, |dx, dy| {
            1.0 - (cnst * (dx * dx + dy * dy) / fc2).exp()
        })
    }

    /// Create a Gaussian highpass mask (libvips `vips_mask_gaussian`):
    /// `1 - exp(ln(ac) * d^2 / fc^2)`. Panicking form of
    /// [`Raster::try_mask_gaussian`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_mask_gaussian`].
    #[track_caller]
    pub fn mask_gaussian(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        nodc: bool,
    ) -> Raster {
        expect_create(
            "mask_gaussian",
            Self::try_mask_gaussian(width, height, frequency_cutoff, amplitude_cutoff, nodc),
        )
    }

    /// Fallible form of [`Raster::mask_gaussian_ring`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_gaussian_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let df = ringwidth / 2.0;
        let df2 = df * df;
        let cnst = amplitude_cutoff.ln();
        mask_image(width, height, false, nodc, false, |dx, dy| {
            let dist = (dx * dx + dy * dy).sqrt();
            (cnst * (dist - frequency_cutoff) * (dist - frequency_cutoff) / df2).exp()
        })
    }

    /// Create a Gaussian ring-pass mask (libvips
    /// `vips_mask_gaussian_ring`): a Gaussian bump centred on the ring
    /// at `frequency_cutoff` with width `ringwidth`. Panicking form of
    /// [`Raster::try_mask_gaussian_ring`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_gaussian_ring`].
    #[track_caller]
    pub fn mask_gaussian_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Raster {
        expect_create(
            "mask_gaussian_ring",
            Self::try_mask_gaussian_ring(
                width,
                height,
                frequency_cutoff,
                amplitude_cutoff,
                ringwidth,
                nodc,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_gaussian_band`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_gaussian_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
    ) -> Result<Raster, CreateError> {
        let r2 = radius * radius;
        let cnst = amplitude_cutoff.ln();
        // Normalise the amplitude at (fcx, fcy) to 1.0, like
        // mask_gaussian_band.c.
        let cnsta = 1.0
            / (1.0
                + (cnst
                    * 4.0
                    * (frequency_cutoff_x * frequency_cutoff_x
                        + frequency_cutoff_y * frequency_cutoff_y)
                    / r2)
                    .exp());
        mask_image(width, height, false, false, false, |dx, dy| {
            let d1 = (dx - frequency_cutoff_x) * (dx - frequency_cutoff_x)
                + (dy - frequency_cutoff_y) * (dy - frequency_cutoff_y);
            let d2 = (dx + frequency_cutoff_x) * (dx + frequency_cutoff_x)
                + (dy + frequency_cutoff_y) * (dy + frequency_cutoff_y);
            cnsta * ((cnst * d1 / r2).exp() + (cnst * d2 / r2).exp())
        })
    }

    /// Create a Gaussian band-pass mask (libvips
    /// `vips_mask_gaussian_band`): Gaussian bumps at `(+-fx, +-fy)`
    /// normalised to peak at 1.0. Panicking form of
    /// [`Raster::try_mask_gaussian_band`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_gaussian_band`].
    #[track_caller]
    pub fn mask_gaussian_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
    ) -> Raster {
        expect_create(
            "mask_gaussian_band",
            Self::try_mask_gaussian_band(
                width,
                height,
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
                amplitude_cutoff,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_butterworth`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    pub fn try_mask_butterworth(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        nodc: bool,
        optical: bool,
        uchar: bool,
    ) -> Result<Raster, CreateError> {
        let cnst = (1.0 / amplitude_cutoff) - 1.0;
        let fc2 = frequency_cutoff * frequency_cutoff;
        mask_image(width, height, optical, nodc, uchar, |dx, dy| {
            let d = dx * dx + dy * dy;
            if d == 0.0 {
                0.0
            } else {
                1.0 / (1.0 + cnst * (fc2 / d).powf(order))
            }
        })
    }

    /// Create a Butterworth highpass mask (libvips
    /// `vips_mask_butterworth`): `1 / (1 + ((1/ac) - 1) * (fc^2/d^2)^order)`,
    /// 0 at DC. Panicking form of [`Raster::try_mask_butterworth`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_butterworth`].
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn mask_butterworth(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        nodc: bool,
        optical: bool,
        uchar: bool,
    ) -> Raster {
        expect_create(
            "mask_butterworth",
            Self::try_mask_butterworth(
                width,
                height,
                order,
                frequency_cutoff,
                amplitude_cutoff,
                nodc,
                optical,
                uchar,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_butterworth_ring`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    pub fn try_mask_butterworth_ring(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let df = ringwidth / 2.0;
        let df2 = df * df;
        let cnst = (1.0 / amplitude_cutoff) - 1.0;
        mask_image(width, height, false, nodc, false, |dx, dy| {
            let dist = (dx * dx + dy * dy).sqrt();
            1.0 / (1.0
                + cnst * ((dist - frequency_cutoff) * (dist - frequency_cutoff) / df2).powf(order))
        })
    }

    /// Create a Butterworth ring-pass mask (libvips
    /// `vips_mask_butterworth_ring`): a Butterworth bump centred on the
    /// ring at `frequency_cutoff` with width `ringwidth`. Panicking form
    /// of [`Raster::try_mask_butterworth_ring`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_butterworth_ring`].
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn mask_butterworth_ring(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
        nodc: bool,
    ) -> Raster {
        expect_create(
            "mask_butterworth_ring",
            Self::try_mask_butterworth_ring(
                width,
                height,
                order,
                frequency_cutoff,
                amplitude_cutoff,
                ringwidth,
                nodc,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_butterworth_band`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    pub fn try_mask_butterworth_band(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
        uchar: bool,
        optical: bool,
        nodc: bool,
    ) -> Result<Raster, CreateError> {
        let r2 = radius * radius;
        let cnst = (1.0 / amplitude_cutoff) - 1.0;
        // Normalise the amplitude at (fcx, fcy) to 1.0, like
        // mask_butterworth_band.c.
        let cnsta = 1.0
            / (1.0
                + 1.0
                    / (1.0
                        + cnst
                            * (4.0
                                * (frequency_cutoff_x * frequency_cutoff_x
                                    + frequency_cutoff_y * frequency_cutoff_y)
                                / r2)
                                .powf(order)));
        mask_image(width, height, optical, nodc, uchar, |dx, dy| {
            let d1 = (dx - frequency_cutoff_x) * (dx - frequency_cutoff_x)
                + (dy - frequency_cutoff_y) * (dy - frequency_cutoff_y);
            let d2 = (dx + frequency_cutoff_x) * (dx + frequency_cutoff_x)
                + (dy + frequency_cutoff_y) * (dy + frequency_cutoff_y);
            cnsta
                * (1.0 / (1.0 + cnst * (d1 / r2).powf(order))
                    + 1.0 / (1.0 + cnst * (d2 / r2).powf(order)))
        })
    }

    /// Create a Butterworth band-pass mask (libvips
    /// `vips_mask_butterworth_band`): Butterworth bumps at `(+-fx, +-fy)`
    /// normalised to peak at 1.0. Panicking form of
    /// [`Raster::try_mask_butterworth_band`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see
    /// [`Raster::try_mask_butterworth_band`].
    // The signature is pinned by the ported-test call surface.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn mask_butterworth_band(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
        uchar: bool,
        optical: bool,
        nodc: bool,
    ) -> Raster {
        expect_create(
            "mask_butterworth_band",
            Self::try_mask_butterworth_band(
                width,
                height,
                order,
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
                amplitude_cutoff,
                uchar,
                optical,
                nodc,
            ),
        )
    }

    /// Fallible form of [`Raster::mask_fractal`].
    ///
    /// # Errors
    ///
    /// [`CreateError::Raster`] on allocation failure.
    pub fn try_mask_fractal(
        width: u32,
        height: u32,
        fractal_dimension: f64,
    ) -> Result<Raster, CreateError> {
        let fd = (fractal_dimension - 4.0) / 2.0;
        mask_image(width, height, false, false, false, |dx, dy| {
            (dx * dx + dy * dy).powf(fd)
        })
    }

    /// Create a fractal power-spectrum mask (libvips
    /// `vips_mask_fractal`, dimension between 2 and 3):
    /// `(d^2)^((fd - 4) / 2)` with the DC component forced to 1.
    /// Panicking form of [`Raster::try_mask_fractal`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_mask_fractal`].
    #[track_caller]
    pub fn mask_fractal(width: u32, height: u32, fractal_dimension: f64) -> Raster {
        expect_create(
            "mask_fractal",
            Self::try_mask_fractal(width, height, fractal_dimension),
        )
    }
}

/// Shape parameters for [`Raster::sdf`], mirroring the libvips `vips_sdf`
/// optional arguments. `a` is required by every shape; `b` by `box`,
/// `rounded-box`, and `line`; `r` by `circle`; `corners` optionally
/// rounds the `rounded-box` (clockwise from top-right, defaulting to
/// four square corners, exactly like libvips).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SdfParams {
    /// First point: the circle centre, the box corner opposite `b`, or
    /// the line start.
    pub a: [i64; 2],
    /// Second point: the box corner opposite `a`, or the line end.
    pub b: Option<[i64; 2]>,
    /// Circle radius.
    pub r: Option<i64>,
    /// `rounded-box` corner radii, clockwise from top-right.
    pub corners: Option<[i64; 4]>,
}

impl Raster {
    // -----------------------------------------------------------------
    // Signed distance fields
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::sdf`].
    ///
    /// # Errors
    ///
    /// [`CreateError::UnknownSdfShape`] for a shape outside
    /// `"circle"` / `"box"` / `"rounded-box"` / `"line"`,
    /// [`CreateError::MissingSdfParam`] when the shape's required
    /// parameters are absent, or [`CreateError::Raster`] on allocation
    /// failure.
    pub fn try_sdf(
        width: u32,
        height: u32,
        shape: &str,
        params: &SdfParams,
    ) -> Result<Raster, CreateError> {
        // libvips create/sdf.c, evaluated in f32 exactly as the C code
        // is (hypotf / fabsf / the f32 clip).
        let a = [params.a[0] as f32, params.a[1] as f32];
        let need_b = |shape: &'static str| {
            params
                .b
                .map(|b| [b[0] as f32, b[1] as f32])
                .ok_or(CreateError::MissingSdfParam { shape, param: "b" })
        };

        enum Shape {
            Circle { r: f32 },
            Box { corners: Option<[f32; 4]> },
            Line,
        }

        let (shape_fn, b) = match shape {
            "circle" => {
                let r = params.r.ok_or(CreateError::MissingSdfParam {
                    shape: "circle",
                    param: "r",
                })? as f32;
                (Shape::Circle { r }, None)
            }
            "box" => (Shape::Box { corners: None }, Some(need_b("box")?)),
            "rounded-box" => {
                let corners = params.corners.unwrap_or([0; 4]).map(|c| c as f32);
                (
                    Shape::Box {
                        corners: Some(corners),
                    },
                    Some(need_b("rounded-box")?),
                )
            }
            "line" => (Shape::Line, Some(need_b("line")?)),
            other => {
                return Err(CreateError::UnknownSdfShape {
                    shape: other.to_owned(),
                });
            }
        };

        // Centre, difference, and half-size, like vips_sdf_build.
        let b_pt = b.unwrap_or([0.0, 0.0]);
        let cx = (a[0] + b_pt[0]) / 2.0;
        let cy = (a[1] + b_pt[1]) / 2.0;
        let dx = b_pt[0] - a[0];
        let dy = b_pt[1] - a[1];
        let sx = dx / 2.0;
        let sy = dy / 2.0;

        let mut out = float1_from_fn(width, height, |x, y| {
            let (x, y) = (x as f32, y as f32);
            match &shape_fn {
                Shape::Circle { r } => (x - a[0]).hypot(y - a[1]) - r,
                Shape::Box { corners: None } => {
                    let px = x - cx;
                    let py = y - cy;
                    let qx = px.abs() - sx;
                    let qy = py.abs() - sy;
                    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0)
                }
                Shape::Box {
                    corners: Some(corners),
                } => {
                    let px = x - cx;
                    let py = y - cy;
                    // Radius of the nearest corner, clockwise from
                    // top-right, like vips_sdf_rounded_box.
                    let r_top = if px > 0.0 { corners[0] } else { corners[2] };
                    let r_bottom = if px > 0.0 { corners[1] } else { corners[3] };
                    let r = if py > 0.0 { r_top } else { r_bottom };
                    let qx = px.abs() - sx + r;
                    let qy = py.abs() - sy + r;
                    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
                }
                Shape::Line => {
                    let pax = x - a[0];
                    let pay = y - a[1];
                    let dot_paba = pax * dx + pay * dy;
                    let dot_baba = dx * dx + dy * dy;
                    let h = (dot_paba / dot_baba).clamp(0.0, 1.0);
                    (pax - h * dx).hypot(pay - h * dy)
                }
            }
        })?;
        out.meta.interpretation = Some(Interpretation::Bw);
        Ok(out)
    }

    /// Create a signed distance field for a shape (libvips `vips_sdf`):
    /// `"circle"` (needs `a`, `r`), `"box"` / `"rounded-box"` (need `a`,
    /// `b`; `rounded-box` also reads `corners`), or `"line"` (needs `a`,
    /// `b`). Single-band float; negative inside the shape, positive
    /// outside. Panicking form of [`Raster::try_sdf`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_sdf`].
    #[track_caller]
    pub fn sdf(width: u32, height: u32, shape: &str, params: &SdfParams) -> Raster {
        expect_create("sdf", Self::try_sdf(width, height, shape, params))
    }

    // -----------------------------------------------------------------
    // Value-range aliases
    // -----------------------------------------------------------------

    /// Largest sample across every band; the spelling the ported suites
    /// use for [`Raster::max`] (pyvips `im.max()` reads back as
    /// `max_value` in the ported Rust cells).
    pub fn max_value(&self) -> f64 {
        self.max()
    }

    /// Smallest sample across every band; the spelling the ported
    /// suites use for [`Raster::min`].
    pub fn min_value(&self) -> f64 {
        self.min()
    }
}

// ---------------------------------------------------------------------
// Text rendering
// ---------------------------------------------------------------------

/// The bundled default face for [`Raster::text`]: Bitstream Vera Sans
/// (licence in `fonts/VERA-COPYRIGHT.TXT`; free redistribution bundled
/// with software).
static VERA_TTF: &[u8] = include_bytes!("../fonts/Vera.ttf");

/// libvips' default text size in points (`vips_text` default font
/// "sans 12").
const TEXT_POINTS: f32 = 12.0;

/// How [`Raster::text`] breaks lines at `width`, mirroring
/// `VipsTextWrap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextWrap {
    Word,
    Char,
    WordChar,
    None,
}

fn parse_wrap(wrap: Option<&str>) -> Result<TextWrap, CreateError> {
    match wrap {
        // libvips defaults to word wrapping.
        None | Some("word") => Ok(TextWrap::Word),
        Some("char") => Ok(TextWrap::Char),
        Some("word-char") => Ok(TextWrap::WordChar),
        Some("none") => Ok(TextWrap::None),
        Some(other) => Err(CreateError::UnknownWrapMode {
            mode: other.to_owned(),
        }),
    }
}

/// Sum of scaled advances (with kerning) for a run of characters.
fn advance_width(font: &FontRef<'_>, px: f32, s: &str) -> f32 {
    let scaled = font.as_scaled(PxScale::from(px));
    let mut w = 0.0f32;
    let mut prev = None;
    for ch in s.chars() {
        let id = font.glyph_id(ch);
        if let Some(p) = prev {
            w += scaled.kern(p, id);
        }
        w += scaled.h_advance(id);
        prev = Some(id);
    }
    w
}

/// Break `text` into rendered lines: hard-break on `'\n'`, then wrap at
/// `max_width` pixels per `wrap`. Greedy, like Pango: a word that alone
/// exceeds the limit overflows in `Word` mode and is split in `Char` /
/// `WordChar` modes.
fn break_lines(
    font: &FontRef<'_>,
    px: f32,
    text: &str,
    max_width: Option<f32>,
    wrap: TextWrap,
) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim_end_matches('\r');
        let Some(limit) = max_width else {
            lines.push(paragraph.to_owned());
            continue;
        };
        if wrap == TextWrap::None {
            lines.push(paragraph.to_owned());
            continue;
        }
        let mut line = String::new();
        let push_chars = |line: &mut String, unit: &str, lines: &mut Vec<String>| {
            for ch in unit.chars() {
                let mut candidate = line.clone();
                candidate.push(ch);
                if !line.is_empty() && advance_width(font, px, &candidate) > limit {
                    lines.push(std::mem::take(line));
                    if ch != ' ' {
                        line.push(ch);
                    }
                } else {
                    *line = candidate;
                }
            }
        };
        match wrap {
            TextWrap::Char => push_chars(&mut line, paragraph, &mut lines),
            TextWrap::Word | TextWrap::WordChar => {
                for word in paragraph.split_inclusive(' ') {
                    let mut candidate = line.clone();
                    candidate.push_str(word);
                    if !line.is_empty()
                        && advance_width(font, px, candidate.trim_end_matches(' ')) > limit
                    {
                        lines.push(std::mem::take(&mut line).trim_end_matches(' ').to_owned());
                        candidate = word.to_owned();
                    }
                    let word_alone = candidate.trim_end_matches(' ');
                    if wrap == TextWrap::WordChar && advance_width(font, px, word_alone) > limit {
                        // A single over-long word: fall back to
                        // character wrapping for it.
                        let overflow = std::mem::take(&mut candidate);
                        push_chars(&mut line, &overflow, &mut lines);
                    } else {
                        line = candidate;
                    }
                }
            }
            TextWrap::None => unreachable!("handled above"),
        }
        lines.push(line.trim_end_matches(' ').to_owned());
    }
    lines
}

/// Position every glyph of `lines` at scale `px`, returning the glyphs
/// and nothing else; ink extents come from the rasteriser.
fn position_glyphs(font: &FontRef<'_>, px: f32, lines: &[String]) -> Vec<Glyph> {
    let scaled = font.as_scaled(PxScale::from(px));
    let line_height = scaled.ascent() - scaled.descent() + scaled.line_gap();
    let mut glyphs = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let baseline = i as f32 * line_height + scaled.ascent();
        let mut pen = 0.0f32;
        let mut prev = None;
        for ch in line.chars() {
            let id = font.glyph_id(ch);
            if let Some(p) = prev {
                pen += scaled.kern(p, id);
            }
            glyphs.push(id.with_scale_and_position(PxScale::from(px), ab_point(pen, baseline)));
            pen += scaled.h_advance(id);
            prev = Some(id);
        }
    }
    glyphs
}

/// Ink bounding box of positioned glyphs from their outline bounds, as
/// `(min_x, min_y, max_x, max_y)`; `None` when nothing has an outline.
fn ink_bounds(font: &FontRef<'_>, glyphs: &[Glyph]) -> Option<(i64, i64, i64, i64)> {
    let mut bounds: Option<(i64, i64, i64, i64)> = None;
    for g in glyphs {
        if let Some(og) = font.outline_glyph(g.clone()) {
            let r = og.px_bounds();
            let (x0, y0) = (r.min.x.floor() as i64, r.min.y.floor() as i64);
            let (x1, y1) = (r.max.x.ceil() as i64, r.max.y.ceil() as i64);
            bounds = Some(match bounds {
                None => (x0, y0, x1, y1),
                Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
            });
        }
    }
    bounds
}

/// Rasterise positioned glyphs into a tightly trimmed Gray8 coverage
/// image.
fn render_glyphs(font: &FontRef<'_>, glyphs: &[Glyph]) -> Result<Raster, CreateError> {
    let (min_x, min_y, max_x, max_y) = ink_bounds(font, glyphs).ok_or(CreateError::EmptyText)?;
    let w = (max_x - min_x).max(1) as usize;
    let h = (max_y - min_y).max(1) as usize;
    let mut canvas = vec![0.0f32; w * h];
    for g in glyphs {
        if let Some(og) = font.outline_glyph(g.clone()) {
            let r = og.px_bounds();
            let gx = r.min.x.floor() as i64 - min_x;
            let gy = r.min.y.floor() as i64 - min_y;
            og.draw(|x, y, c| {
                let cx = gx + i64::from(x);
                let cy = gy + i64::from(y);
                if cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h {
                    let idx = cy as usize * w + cx as usize;
                    canvas[idx] = canvas[idx].max(c);
                }
            });
        }
    }
    // Trim to the true ink rectangle (outline bounds are conservative).
    let mut lo_x = w;
    let mut lo_y = h;
    let (mut hi_x, mut hi_y) = (0usize, 0usize);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if canvas[y * w + x] > 0.0 {
                any = true;
                lo_x = lo_x.min(x);
                lo_y = lo_y.min(y);
                hi_x = hi_x.max(x);
                hi_y = hi_y.max(y);
            }
        }
    }
    if !any {
        return Err(CreateError::EmptyText);
    }
    let out_w = hi_x - lo_x + 1;
    let out_h = hi_y - lo_y + 1;
    let mut data = vec![0u8; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let c = canvas[(y + lo_y) * w + (x + lo_x)].clamp(0.0, 1.0);
            data[y * out_w + x] = (c * 255.0).round() as u8;
        }
    }
    Ok(Raster::new(
        out_w as u32,
        out_h as u32,
        PixelFormat::Gray8,
        data,
    )?)
}

/// Rendered ink extents at scale `px` (width, height), without filling.
fn measure_text(
    font: &FontRef<'_>,
    px: f32,
    text: &str,
    max_width: Option<f32>,
    wrap: TextWrap,
) -> (i64, i64) {
    let lines = break_lines(font, px, text, max_width, wrap);
    match ink_bounds(font, &position_glyphs(font, px, &lines)) {
        Some((x0, y0, x1, y1)) => (x1 - x0, y1 - y0),
        None => (0, 0),
    }
}

impl Raster {
    /// Fallible form of [`Raster::text`].
    ///
    /// # Errors
    ///
    /// [`CreateError::UnknownWrapMode`] for a wrap mode outside
    /// `"word"` / `"char"` / `"word-char"` / `"none"`,
    /// [`CreateError::EmptyText`] when nothing renders (empty or
    /// whitespace-only text), or [`CreateError::Raster`] on allocation
    /// failure.
    pub fn try_text(
        text: &str,
        dpi: Option<u32>,
        width: Option<u32>,
        height: Option<u32>,
        wrap: Option<&str>,
    ) -> Result<Raster, CreateError> {
        let wrap = parse_wrap(wrap)?;
        if text.is_empty() {
            return Err(CreateError::EmptyText);
        }
        let font = FontRef::try_from_slice(VERA_TTF).expect("the bundled Vera.ttf parses");
        let max_width = width.map(|w| w as f32);
        let px_for = |dpi: u32| TEXT_POINTS * dpi as f32 / 72.0;

        let dpi = match (dpi, width, height) {
            // The libvips auto-fit: with a fit box and no explicit dpi,
            // find the largest dpi whose ink extents fit width x height.
            (None, Some(fit_w), Some(fit_h)) => {
                let fits = |dpi: u32| {
                    let (w, h) = measure_text(&font, px_for(dpi), text, max_width, wrap);
                    w <= i64::from(fit_w) && h <= i64::from(fit_h)
                };
                let mut lo = 1u32;
                if fits(lo) {
                    let mut hi = 72u32;
                    while hi < 65536 && fits(hi) {
                        lo = hi;
                        hi *= 2;
                    }
                    // Largest fitting dpi in [lo, hi).
                    while lo + 1 < hi {
                        let mid = lo + (hi - lo) / 2;
                        if fits(mid) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                }
                lo
            }
            (dpi, _, _) => dpi.unwrap_or(72),
        };

        let lines = break_lines(&font, px_for(dpi), text, max_width, wrap);
        let glyphs = position_glyphs(&font, px_for(dpi), &lines);
        render_glyphs(&font, &glyphs)
    }

    /// Render text to a single-band `Gray8` coverage image (libvips
    /// `vips_text`): 0 background, up to 255 ink, trimmed to the ink
    /// extents.
    ///
    /// * `dpi` sets the rendering resolution (default 72); the face is
    ///   the bundled Bitstream Vera Sans at libvips' default 12pt.
    /// * `width` wraps lines at that many pixels using `wrap` (`"word"`,
    ///   default; `"char"`; `"word-char"`; `"none"`).
    /// * `width` and `height` together, with no explicit `dpi`,
    ///   auto-fit: the largest dpi whose rendered extents fit the box is
    ///   used, like the libvips `autofit_dpi` search.
    ///
    /// Rendering is deterministic; glyph-exact parity with Pango is out
    /// of scope (see the module docs). Panicking form of
    /// [`Raster::try_text`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CreateError`]; see [`Raster::try_text`].
    #[track_caller]
    pub fn text(
        text: &str,
        dpi: Option<u32>,
        width: Option<u32>,
        height: Option<u32>,
        wrap: Option<&str>,
    ) -> Raster {
        expect_create("text", Self::try_text(text, dpi, width, height, wrap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- black ----

    /// black is a Gray8 all-zero image; black_bands picks the canonical
    /// format for the band count.
    #[test]
    fn black_formats_and_zero_fill() {
        let im = Raster::black(100, 100);
        assert_eq!((im.width(), im.height()), (100, 100));
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert!(im.data().iter().all(|&b| b == 0));

        assert_eq!(Raster::black_bands(4, 4, 3).format(), PixelFormat::Rgb8);
        assert_eq!(Raster::black_bands(4, 4, 4).format(), PixelFormat::Rgba8);
        let multi = Raster::black_bands(4, 4, 2);
        assert_eq!(multi.format().channels(), 2);
        assert_eq!(multi.getpoint(3, 3), vec![0.0, 0.0]);
    }

    /// A zero band count is a typed error, not a panic in the format
    /// table.
    #[test]
    fn black_bands_zero_errors() {
        assert!(matches!(
            Raster::try_black_bands(4, 4, 0),
            Err(CreateError::InvalidBandCount { .. })
        ));
    }

    // ---- xyz ----

    /// Every pixel of xyz is its own coordinates.
    #[test]
    fn xyz_pixel_is_its_coordinates() {
        let im = Raster::xyz(128, 128);
        assert_eq!(im.format().channels(), 2);
        assert_eq!(im.getpoint(45, 35), vec![45.0, 35.0]);
        assert_eq!(im.getpoint(0, 0), vec![0.0, 0.0]);
        assert_eq!(im.getpoint(127, 127), vec![127.0, 127.0]);
    }

    // ---- eye ----

    /// The float eye pattern spans exactly -1..1 on the 100x90 grid the
    /// ported test uses: +1 at (0, 89) (cos(0) with the full-amplitude
    /// row), -1 at (66, 89) (c*66^2 is exactly 11*pi with the default
    /// factor 0.5).
    #[test]
    fn eye_float_hits_exact_extremes() {
        let im = Raster::eye(100, 90, false);
        assert_eq!((im.width(), im.height()), (100, 90));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.getpoint(0, 89), vec![1.0]);
        assert!((im.getpoint(66, 89)[0] + 1.0).abs() < 1e-6);
        assert!((im.max_value() - 1.0).abs() < 0.001);
        assert!((im.min_value() + 1.0).abs() < 0.001);
    }

    /// The uchar eye maps -1..1 to the full 0..=255 range.
    #[test]
    fn eye_uchar_full_range() {
        let im = Raster::eye(100, 90, true);
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert_eq!(im.max_value(), 255.0);
        assert_eq!(im.min_value(), 0.0);
    }

    // ---- zone / sines ----

    /// The zone plate is 1.0 at its centre and bounded by -1..1.
    #[test]
    fn zone_centre_and_range() {
        let im = Raster::zone(128, 128);
        assert_eq!((im.width(), im.height()), (128, 128));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.getpoint(64, 64), vec![1.0]);
        assert!(im.max_value() <= 1.0 && im.min_value() >= -1.0);
    }

    /// The sine grating is 1.0 at the origin and bounded by -1..1.
    #[test]
    fn sines_origin_and_range() {
        let im = Raster::sines(128, 128);
        assert_eq!((im.width(), im.height()), (128, 128));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.getpoint(0, 0), vec![1.0]);
        assert!(im.max_value() <= 1.0 && im.min_value() >= -1.0);
    }

    // ---- gaussnoise ----

    /// The noise statistics track the requested mean and sigma (the
    /// ported tolerances), and the pinned default seed is deterministic.
    #[test]
    fn gaussnoise_stats_and_determinism() {
        let im = Raster::gaussnoise(100, 90, 100.0, 10.0);
        assert_eq!(im.format().channels(), 1);
        assert!((im.avg() - 100.0).abs() < 0.4, "avg {}", im.avg());
        assert!((im.deviate() - 10.0).abs() < 0.4, "dev {}", im.deviate());

        let again = Raster::gaussnoise(100, 90, 100.0, 10.0);
        assert_eq!(im.data(), again.data());
        let reseeded = Raster::gaussnoise_seeded(100, 90, 100.0, 10.0, 42);
        assert_ne!(im.data(), reseeded.data());
    }

    // ---- perlin ----

    /// Perlin noise is exactly zero on cell corners, bounded, and
    /// deterministic per seed.
    #[test]
    fn perlin_corners_bounds_seeds() {
        let im = Raster::perlin(300, 300);
        assert_eq!((im.width(), im.height()), (300, 300));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(256, 256), vec![0.0]);
        let samples = im.f32_samples().unwrap();
        assert!(samples.iter().all(|v| v.abs() <= 1.5));
        // Non-degenerate: the field actually varies.
        assert!(im.deviate() > 0.01);

        assert_eq!(im.data(), Raster::perlin(300, 300).data());
        assert_ne!(im.data(), Raster::perlin_seeded(300, 300, 7).data());
    }

    // ---- worley ----

    /// Worley noise is a distance field: non-negative, clamped at 1.5
    /// cells, deterministic per seed.
    #[test]
    fn worley_bounds_and_seeds() {
        let im = Raster::worley(300, 200);
        assert_eq!((im.width(), im.height()), (300, 200));
        assert_eq!(im.format().channels(), 1);
        let samples = im.f32_samples().unwrap();
        assert!(samples.iter().all(|&v| (0.0..=384.0).contains(&v)));
        assert!(im.deviate() > 0.01);

        assert_eq!(im.data(), Raster::worley(300, 200).data());
        assert_ne!(im.data(), Raster::worley_seeded(300, 200, 7).data());
    }

    // ---- fractsurf ----

    /// The fractal surface has the requested shape, varies, and is
    /// deterministic with the pinned seed.
    #[test]
    fn fractsurf_shape_and_determinism() {
        let im = Raster::fractsurf(64, 48, 2.5);
        assert_eq!((im.width(), im.height()), (64, 48));
        assert_eq!(im.format().channels(), 1);
        assert!(im.deviate() > 0.0);
        assert!(im.f32_samples().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(im.data(), Raster::fractsurf(64, 48, 2.5).data());
    }

    // ---- internal DFT ----

    /// An impulse transforms to a flat spectrum; forward-then-inverse
    /// scaled by 1/(w*h) reproduces the input.
    #[test]
    fn dft_impulse_and_roundtrip() {
        let (w, h) = (8usize, 4usize);
        let mut buf = vec![(0.0, 0.0); w * h];
        buf[0] = (1.0, 0.0);
        dft_2d(&mut buf, w, h, false);
        for &(re, im) in &buf {
            assert!((re - 1.0).abs() < 1e-12 && im.abs() < 1e-12);
        }

        let orig: Vec<(f64, f64)> = (0..w * h).map(|i| ((i % 7) as f64 - 3.0, 0.0)).collect();
        let mut buf = orig.clone();
        dft_2d(&mut buf, w, h, false);
        dft_2d(&mut buf, w, h, true);
        let n = (w * h) as f64;
        for (got, want) in buf.iter().zip(orig.iter()) {
            assert!((got.0 / n - want.0).abs() < 1e-9);
            assert!((got.1 / n - want.1).abs() < 1e-9);
        }
    }

    // ---- buildlut ----

    /// The two-point LUT from the ported test: exact endpoints, linear
    /// interior.
    #[test]
    fn buildlut_two_point_matches_libvips() {
        let lut = Raster::buildlut(&[vec![0.0, 0.0], vec![255.0, 100.0]]);
        assert_eq!((lut.width(), lut.height()), (256, 1));
        assert_eq!(lut.format().channels(), 1);
        assert_eq!(lut.interpretation(), Interpretation::Histogram);
        assert_eq!(lut.getpoint(0, 0), vec![0.0]);
        assert_eq!(lut.getpoint(255, 0), vec![100.0]);
        assert!((lut.getpoint(10, 0)[0] - 100.0 * 10.0 / 255.0).abs() < 1e-4);
    }

    /// Control points may arrive unsorted; multi-band outputs
    /// interpolate each band; the final row is written inclusively.
    #[test]
    fn buildlut_multiband_unsorted_inclusive() {
        let lut = Raster::buildlut(&[
            vec![128.0, 10.0, 90.0],
            vec![0.0, 0.0, 100.0],
            vec![255.0, 100.0, 0.0],
        ]);
        assert_eq!(lut.width(), 256);
        assert_eq!(lut.format().channels(), 2);
        let p = lut.getpoint(64, 0);
        assert!((p[0] - 5.0).abs() < 1e-4 && (p[1] - 95.0).abs() < 1e-4);
        assert_eq!(lut.getpoint(255, 0), vec![100.0, 0.0]);
        // An offset x range produces an offset-width LUT.
        let lut = Raster::buildlut(&[vec![10.0, 1.0], vec![19.0, 10.0]]);
        assert_eq!(lut.width(), 10);
        assert_eq!(lut.getpoint(9, 0), vec![10.0]);
    }

    /// Mis-shaped control points are typed errors.
    #[test]
    fn buildlut_errors() {
        assert!(matches!(
            Raster::try_buildlut(&[]),
            Err(CreateError::EmptyMatrix { .. })
        ));
        assert!(matches!(
            Raster::try_buildlut(&[vec![0.0, 0.0], vec![1.0]]),
            Err(CreateError::RaggedMatrix { row: 1, .. })
        ));
        assert!(matches!(
            Raster::try_buildlut(&[vec![0.0]]),
            Err(CreateError::TooFewColumns { columns: 1 })
        ));
        assert!(matches!(
            Raster::try_buildlut(&[vec![0.5, 0.0]]),
            Err(CreateError::NonIntegerInput { row: 0, .. })
        ));
        assert!(matches!(
            Raster::try_buildlut(&[vec![f64::NAN, 0.0]]),
            Err(CreateError::NonIntegerInput { row: 0, .. })
        ));
    }

    // ---- tonelut ----

    /// With every parameter at the libvips default the tone curve is the
    /// (truncated) identity: monotonic, 0 at 0, in_max at in_max.
    #[test]
    fn tonelut_identity_curve() {
        let im = Raster::tonelut();
        assert_eq!((im.width(), im.height()), (32768, 1));
        assert_eq!(im.format(), PixelFormat::Gray16);
        assert_eq!(im.interpretation(), Interpretation::Histogram);
        assert!(im.hist_ismonotonic());
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.getpoint(32767, 0), vec![32767.0]);
        // Truncation matches the C int cast at an interior point.
        let x = 100.0 * 12345.0 / 32767.0;
        let expected = ((32767.0 / 100.0) * x) as i64 as f64;
        assert_eq!(im.getpoint(12345, 0), vec![expected]);
    }

    // ---- from_matrix ----

    /// A matrix image keeps its values, one row per image row, stamped
    /// with the Matrix interpretation.
    #[test]
    fn from_matrix_values_and_stamp() {
        let m = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!((m.width(), m.height()), (3, 2));
        assert_eq!(m.format().channels(), 1);
        assert_eq!(m.interpretation(), Interpretation::Matrix);
        assert_eq!(m.getpoint(2, 0), vec![3.0]);
        assert_eq!(m.getpoint(0, 1), vec![4.0]);
    }

    /// Mis-shaped matrices are typed errors.
    #[test]
    fn from_matrix_errors() {
        assert!(matches!(
            Raster::try_from_matrix(&[]),
            Err(CreateError::EmptyMatrix { .. })
        ));
        assert!(matches!(
            Raster::try_from_matrix(&[vec![1.0], vec![1.0, 2.0]]),
            Err(CreateError::RaggedMatrix { row: 1, .. })
        ));
        assert!(matches!(
            Raster::try_from_matrix(&[vec![]]),
            Err(CreateError::EmptyMatrix { .. })
        ));
    }

    // ---- frequency masks ----

    /// The ideal highpass mask in FFT layout: DC at the origin (0 with
    /// nodc, forced to 1 without), 0 inside the cutoff, 1 outside,
    /// stamped Fourier.
    #[test]
    fn mask_ideal_highpass_layout_and_dc() {
        let im = Raster::mask_ideal(128, 128, 0.7, true);
        assert_eq!((im.width(), im.height()), (128, 128));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.interpretation(), Interpretation::Fourier);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert_eq!(im.min_value(), 0.0);
        // Just inside the cutoff along the x axis: dx = 44/64 < 0.7.
        assert_eq!(im.getpoint(44, 0), vec![0.0]);
        // Just outside: dx = 45/64 > 0.7.
        assert_eq!(im.getpoint(45, 0), vec![1.0]);
        // The DC component is forced to 1.0 without nodc (mask.c).
        let im = Raster::mask_ideal(128, 128, 0.7, false);
        assert_eq!(im.getpoint(0, 0), vec![1.0]);
    }

    /// The ideal ring passes the ring only; the ideal band passes discs
    /// at (+-fx, +-fy).
    #[test]
    fn mask_ideal_ring_and_band() {
        let im = Raster::mask_ideal_ring(128, 128, 0.7, 0.5, true);
        assert_eq!(im.getpoint(45, 0), vec![1.0]);
        // Inside the inner edge: 28/64 = 0.4375 < 0.45.
        assert_eq!(im.getpoint(28, 0), vec![0.0]);

        let im = Raster::mask_ideal_band(128, 128, 0.5, 0.5, 0.7);
        assert!((im.max_value() - 1.0).abs() < 0.01);
        assert_eq!(im.getpoint(32, 32), vec![1.0]);
    }

    /// The Gaussian highpass, ring, and band masks match the mask_*.c
    /// formulas at their pinned points.
    #[test]
    fn mask_gaussian_family() {
        let im = Raster::mask_gaussian(128, 128, 0.7, 0.1, true);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        assert!(im.min_value().abs() < 0.01);

        let im = Raster::mask_gaussian_ring(128, 128, 0.7, 0.1, 0.5, true);
        assert!((im.getpoint(45, 0)[0] - 1.0).abs() < 0.001);

        let im = Raster::mask_gaussian_band(128, 128, 0.5, 0.5, 0.7, 0.1);
        assert!((im.max_value() - 1.0).abs() < 0.01);
        assert!((im.getpoint(32, 32)[0] - 1.0).abs() < 0.01);
    }

    /// The Butterworth highpass float mask: DC 0 with nodc, maximum at
    /// the Nyquist corner, with the mask_butterworth.c value there.
    #[test]
    fn mask_butterworth_float_shape() {
        let im = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, true, false, false);
        assert_eq!(im.getpoint(0, 0), vec![0.0]);
        let (v, mx, my) = im.maxpos();
        assert_eq!((mx, my), (64, 64));
        // 1 / (1 + 9 * (0.49 / 2)^2) at the (+-1, +-1) corner.
        assert!((v - 1.0 / (1.0 + 9.0 * (0.49f64 / 2.0).powi(2))).abs() < 1e-6);
    }

    /// The optical uchar Butterworth variant. The ported cell passes
    /// `nodc: true` here and expects 255; that is a mis-port. Proof: the
    /// libvips original (test_create.py::test_mask_butterworth, second
    /// block) builds this mask with `optical=True, uchar=True` and NO
    /// nodc, and mask.c forces the DC component to 1.0 only when nodc
    /// is off ("DC component is always 1"); with optical layout DC sits
    /// at the centre, so (64, 64) reads 255 without nodc. With
    /// `nodc: true` the DC value comes from the highpass formula, which
    /// is 0 at d == 0 (mask_butterworth.c returns 0 for d == 0), so the
    /// same pixel reads 0, never 255.
    #[test]
    fn mask_butterworth_optical_dc() {
        let im = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, false, true, true);
        assert_eq!(im.getpoint(64, 64), vec![255.0]);
        let im = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, true, true, true);
        assert_eq!(im.getpoint(64, 64), vec![0.0]);
    }

    /// The Butterworth ring: ~1.0 on the ring, and with nodc the DC
    /// pixel keeps its (small) formula value, so the global minimum sits
    /// at the Nyquist corner, exactly as the libvips original asserts.
    #[test]
    fn mask_butterworth_ring_minpos() {
        let im = Raster::mask_butterworth_ring(128, 128, 2.0, 0.7, 0.1, 0.5, true);
        assert!((im.getpoint(45, 0)[0] - 1.0).abs() < 0.001);
        // nodc computes the formula at DC rather than writing 0:
        // 1 / (1 + 9 * (0.49 / 0.0625)^2).
        let dc = im.getpoint(0, 0)[0];
        assert!((dc - 1.0 / (1.0 + 9.0 * (0.49f64 / 0.0625).powi(2))).abs() < 1e-6);
        let (v, mx, my) = im.minpos();
        assert_eq!((mx, my), (64, 64));
        assert!(v < dc);
    }

    /// The three ported Butterworth band variants (all consistent with
    /// mask_butterworth_band.c).
    #[test]
    fn mask_butterworth_band_variants() {
        let im =
            Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, false, false, false);
        assert!((im.max_value() - 1.0).abs() < 0.01);
        assert!((im.getpoint(32, 32)[0] - 1.0).abs() < 0.01);

        let im =
            Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, true, true, false);
        assert_eq!(im.max_value(), 255.0);
        assert_eq!(im.getpoint(32, 32), vec![255.0]);
        // Without nodc the optical centre is the forced-to-1 DC.
        assert_eq!(im.getpoint(64, 64), vec![255.0]);

        let im = Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, true, true, true);
        assert_ne!(im.getpoint(64, 64)[0], 255.0);
    }

    /// The fractal mask: DC forced to 1 (no nodc parameter on this op),
    /// power-law falloff elsewhere.
    #[test]
    fn mask_fractal_dc_and_falloff() {
        let im = Raster::mask_fractal(128, 128, 2.3);
        assert_eq!((im.width(), im.height()), (128, 128));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.getpoint(0, 0), vec![1.0]);
        // (d^2)^((2.3 - 4) / 2) at dx = 1/64.
        let expected = ((1.0f64 / 64.0) * (1.0 / 64.0)).powf((2.3 - 4.0) / 2.0);
        assert!((im.getpoint(1, 0)[0] - f64::from(expected as f32)).abs() < expected * 1e-5);
        // Falloff: further from DC is smaller.
        assert!(im.getpoint(10, 0)[0] < im.getpoint(1, 0)[0]);
    }

    /// The uchar mask path truncates 255*v like the C float-to-uchar
    /// cast in vips_linear1, rather than rounding.
    #[test]
    fn mask_uchar_truncates_like_linear() {
        let float = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, true, false, false);
        let uchar = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, true, false, true);
        for x in 0..128 {
            let v = float.getpoint(x, 17)[0] as f32;
            let expected = f64::from((255.0f32 * v).clamp(0.0, 255.0) as u8);
            assert_eq!(uchar.getpoint(x, 17), vec![expected], "column {x}");
        }
    }

    // ---- sdf ----

    /// The circle, box, and line SDFs at the ported probe point, plus
    /// the rounded box (radius 50 on the corner nearest the probe).
    #[test]
    fn sdf_shapes_match_libvips() {
        let im = Raster::sdf(
            128,
            128,
            "circle",
            &SdfParams {
                a: [64, 64],
                r: Some(32),
                ..Default::default()
            },
        );
        assert_eq!((im.width(), im.height()), (128, 128));
        assert_eq!(im.format().channels(), 1);
        assert_eq!(im.interpretation(), Interpretation::Bw);
        // hypot(19, 29) - 32.
        assert!((im.getpoint(45, 35)[0] - 2.6702).abs() < 0.01);

        let im = Raster::sdf(
            128,
            128,
            "box",
            &SdfParams {
                a: [10, 10],
                b: Some([50, 40]),
                ..Default::default()
            },
        );
        assert!((im.getpoint(45, 35)[0] + 5.0).abs() < 0.001);

        let im = Raster::sdf(
            128,
            128,
            "rounded-box",
            &SdfParams {
                a: [10, 10],
                b: Some([50, 40]),
                corners: Some([50, 0, 0, 0]),
                ..Default::default()
            },
        );
        // qx = qy = 45, hypot - 50 = 45*sqrt(2) - 50.
        assert!((im.getpoint(45, 35)[0] - (45.0 * 2.0f64.sqrt() - 50.0)).abs() < 0.001);
        // Zero corner radii degrade to the plain box.
        let plain = Raster::sdf(
            128,
            128,
            "rounded-box",
            &SdfParams {
                a: [10, 10],
                b: Some([50, 40]),
                ..Default::default()
            },
        );
        assert!((plain.getpoint(45, 35)[0] + 5.0).abs() < 0.001);

        let im = Raster::sdf(
            128,
            128,
            "line",
            &SdfParams {
                a: [10, 10],
                b: Some([50, 40]),
                ..Default::default()
            },
        );
        assert!((im.getpoint(45, 35)[0] - 1.0).abs() < 0.001);
    }

    /// Unknown shapes and missing parameters are typed errors.
    #[test]
    fn sdf_errors() {
        assert!(matches!(
            Raster::try_sdf(8, 8, "triangle", &SdfParams::default()),
            Err(CreateError::UnknownSdfShape { .. })
        ));
        assert!(matches!(
            Raster::try_sdf(8, 8, "circle", &SdfParams::default()),
            Err(CreateError::MissingSdfParam { param: "r", .. })
        ));
        assert!(matches!(
            Raster::try_sdf(8, 8, "box", &SdfParams::default()),
            Err(CreateError::MissingSdfParam { param: "b", .. })
        ));
        assert!(matches!(
            Raster::try_sdf(8, 8, "line", &SdfParams::default()),
            Err(CreateError::MissingSdfParam { param: "b", .. })
        ));
    }

    // ---- text ----

    /// Text at 300 dpi renders real ink: taller and wider than 10px,
    /// saturated coverage, zero background, deterministic.
    #[test]
    fn text_renders_coverage() {
        let im = Raster::text("Hello, world!", Some(300), None, None, None);
        assert!(im.width() > 10);
        assert!(im.height() > 10);
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert!(im.max_value() > 240.0);
        assert_eq!(im.min_value(), 0.0);
        let again = Raster::text("Hello, world!", Some(300), None, None, None);
        assert_eq!(im.data(), again.data());
    }

    /// The width x height auto-fit lands within the ported +-50px of the
    /// requested box without overflowing it by more than the 1px
    /// bounds-vs-trim slack.
    #[test]
    fn text_autofit_width() {
        let im = Raster::text("Hello, world!", None, Some(500), Some(500), None);
        assert!(
            (i64::from(im.width()) - 500).abs() < 50,
            "autofit width {}",
            im.width()
        );
        assert!(im.width() <= 501 && im.height() <= 501);
    }

    /// Character wrapping at a narrow width produces a narrower, taller
    /// image than the unwrappable single word.
    #[test]
    fn text_wrap_modes() {
        let word = Raster::text("helloworld", Some(500), Some(100), None, None);
        let ch = Raster::text("helloworld", Some(500), Some(100), None, Some("char"));
        assert!(ch.width() < word.width());
        assert!(ch.height() > word.height());
        assert!(ch.width() <= 100);
        // word-char falls back to char wrapping for over-long words.
        let wc = Raster::text("helloworld", Some(500), Some(100), None, Some("word-char"));
        assert!(wc.width() <= 100);
        // Multi-word text breaks between words in word mode.
        let two = Raster::text("hello world", Some(500), Some(100), None, None);
        assert!(two.width() <= 100 || two.height() > word.height());
    }

    /// Empty text and unknown wrap modes are typed errors.
    #[test]
    fn text_errors() {
        assert!(matches!(
            Raster::try_text("", Some(72), None, None, None),
            Err(CreateError::EmptyText)
        ));
        assert!(matches!(
            Raster::try_text("   ", Some(72), None, None, None),
            Err(CreateError::EmptyText)
        ));
        assert!(matches!(
            Raster::try_text("hi", Some(72), None, None, Some("banana")),
            Err(CreateError::UnknownWrapMode { .. })
        ));
    }

    // ---- value aliases ----

    /// max_value / min_value read the same reductions as max / min.
    #[test]
    fn value_aliases_match_min_max() {
        let im = Raster::gaussnoise(16, 16, 5.0, 2.0);
        assert_eq!(im.max_value(), im.max());
        assert_eq!(im.min_value(), im.min());
    }

    // ---- new_from_image ----

    /// `new_from_image` copies the source geometry and metadata, gives the
    /// result `value.len()` bands, and fills every pixel with `value` so
    /// `avg()` equals the mean of `value`.
    #[test]
    fn new_from_image_bands_avg_and_metadata() {
        // A source carrying distinctive, non-default metadata to copy.
        let src = Raster::zeroed(7, 5, PixelFormat::Rgb8)
            .unwrap()
            .copy()
            .interpretation(Interpretation::Srgb)
            .xres(3.5)
            .yres(4.25)
            .xoffset(11)
            .yoffset(-9)
            .build();

        // Scalar -> 1 band, constant 12; metadata carried over.
        let one = src.new_from_image(&[12.0]);
        assert_eq!(one.width(), src.width());
        assert_eq!(one.height(), src.height());
        assert_eq!(one.bands(), 1);
        assert!((one.avg() - 12.0).abs() < 1e-9);
        assert_eq!(one.interpretation(), src.interpretation());
        assert!((one.xres() - src.xres()).abs() < 1e-9);
        assert!((one.yres() - src.yres()).abs() < 1e-9);
        assert_eq!(one.xoffset(), src.xoffset());
        assert_eq!(one.yoffset(), src.yoffset());

        // 3-element vector -> 3 bands, avg = (1+2+3)/3 = 2.
        let three = src.new_from_image(&[1.0, 2.0, 3.0]);
        assert_eq!(three.bands(), 3);
        assert!((three.avg() - 2.0).abs() < 1e-9);

        // 4-element vector -> 4 bands.
        let four = src.new_from_image(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(four.bands(), 4);
        assert!((four.avg() - 0.0).abs() < 1e-9);
    }

    /// An empty value vector has no band count any format can carry, so
    /// the fallible form reports `InvalidBandCount` instead of panicking.
    #[test]
    fn new_from_image_rejects_empty_value() {
        let src = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        assert!(matches!(
            src.try_new_from_image(&[]),
            Err(CreateError::InvalidBandCount {
                op: "new_from_image",
                bands: 0
            })
        ));
    }
}
