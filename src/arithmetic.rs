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
//!   form.
//!
//! Operations with no argument validation (reductions, constant ops) have
//! only the direct form.
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
//! | [`Raster::sub`], [`Raster::mul`], [`Raster::div`] | `vips_subtract` / `multiply` / `divide` | image-image arithmetic |
//! | [`Raster::linear`] | `vips_linear1` | `a * x + b` |
//! | [`Raster::sum`] | `vips_sum` | pixelwise sum of an image list |
//! | [`Raster::minpair`] / [`Raster::maxpair`] | `vips_minpair` / `vips_maxpair` | pixelwise extremum of two images |
//! | [`Raster::more_than`] family | `vips_relational` | `0` / `255` uchar mask |
//! | [`Raster::bitand`] family, [`Raster::lshift`], [`Raster::rshift`] | `vips_boolean` | bitwise arithmetic |
//! | [`Raster::scaleimage`] | `vips_scale` | values scaled to `0..=255` |
//! | [`Raster::stdif`] | `vips_stdif` | statistical differencing |
//! | [`Raster::recomb`] | `vips_recomb` | band recombination matrix multiply |
//! | [`Raster::premultiply`] / [`Raster::unpremultiply`] | `vips_premultiply` / `vips_unpremultiply` | alpha (un)premultiplication |
//!
//! # Semantics shared by every operation
//!
//! * **Value domain.** Samples are unsigned integers, `0..=255` (8-bit) or
//!   `0..=65535` (16-bit). Arithmetic is computed in `f64` and the result is
//!   rounded to nearest and saturated into the output depth. libvips
//!   promotes many of these ops to float output; this crate has no float
//!   pixel format, so round-and-saturate is the documented contract.
//! * **Depth promotion.** Operations whose exact result can exceed the
//!   input depth (`add_const`, `mul`, `pow_const`, `linear`, `sum`, ...)
//!   promote 8-bit input to 16-bit output, matching the promotion
//!   [`Raster::add`] already performs. 16-bit input has no wider format and
//!   saturates at `65535`. Operations that cannot exceed the input depth
//!   (`sub`, `div`, `clamp`, ...) keep it; subtraction saturates at `0`.
//! * **Comparisons.** The relational family returns an 8-bit image with the
//!   input's band count holding `255` where the relation holds and `0`
//!   where it does not, matching libvips.
//! * **Division by zero.** `x / 0` and `x % 0` produce `0`, matching
//!   libvips `vips_divide`.
//! * **NaN.** A NaN result (e.g. `0.0.powf(f64::NAN)`) writes `0`.
//!
//! # Deferred operations
//!
//! The ported arithmetic suite also calls operations that cannot be
//! represented with the unsigned integer formats in [`PixelFormat`] and are
//! deferred until a float sample depth exists: `neg` (negative samples),
//! the trigonometric / logarithmic / exponential family (`sin`, `cos`,
//! `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
//! `acosh`, `atanh`, `log`, `log10`, `exp`, `exp10`, fractional results),
//! and the complex-number family (`complexform`, `polar`, `rect`, `conj`,
//! `real`, `imag`). The histogram family (`hist_find*`, `hough_*`) and the
//! creation / conversion helpers the ported statistics tests use for setup
//! (`grey`, `insert`) belong to later batches. `floor`, `ceil`, and `rint`
//! are implemented here: on integer formats they are exact identities,
//! which is also what libvips produces for integer input.

use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
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
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

// ---------------------------------------------------------------------------
// Sample-level helpers
// ---------------------------------------------------------------------------

/// Read the flat `i`-th sample as `u32` (native byte order for 16-bit,
/// matching [`crate::raster_ops`]).
#[inline]
fn read_u32(data: &[u8], bpc: usize, i: usize) -> u32 {
    if bpc == 1 {
        data[i] as u32
    } else {
        u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as u32
    }
}

/// Read the flat `i`-th sample as `f64`.
#[inline]
fn read_f64(data: &[u8], bpc: usize, i: usize) -> f64 {
    read_u32(data, bpc, i) as f64
}

/// Write the flat `i`-th sample. `v` must already fit the depth.
#[inline]
fn write_u32(data: &mut [u8], bpc: usize, i: usize, v: u32) {
    if bpc == 1 {
        data[i] = v as u8;
    } else {
        let b = (v as u16).to_ne_bytes();
        data[2 * i] = b[0];
        data[2 * i + 1] = b[1];
    }
}

/// Round `v` to nearest, saturate into `0..=max`, and write it as the flat
/// `i`-th sample. NaN writes `0`.
#[inline]
fn write_f64(data: &mut [u8], bpc: usize, i: usize, v: f64, max: f64) {
    let v = if v.is_nan() {
        0.0
    } else {
        v.round().clamp(0.0, max)
    };
    write_u32(data, bpc, i, v as u32);
}

/// Largest sample value representable at a depth, as `f64`.
#[inline]
fn depth_max(bpc: usize) -> f64 {
    if bpc == 1 { 255.0 } else { 65535.0 }
}

/// Largest sample value representable at a depth, as `u32`.
#[inline]
fn depth_max_u32(bpc: usize) -> u32 {
    if bpc == 1 { 0xFF } else { 0xFFFF }
}

/// The output format for a band count and depth; the band count is bounded
/// by the caller except for `recomb`, which maps `None` to `TooManyBands`.
fn format_for(bands: usize, bpc: usize) -> Result<PixelFormat, ArithmeticError> {
    PixelFormat::with_channels(bands, bpc).ok_or(ArithmeticError::TooManyBands { bands })
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
#[inline]
#[track_caller]
fn expect_arith<T>(op: &str, r: Result<T, ArithmeticError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Apply `f` to every sample, writing a result of the same shape at
/// `out_bpc` depth (rounded and saturated).
fn unary_map(r: &Raster, out_bpc: usize, f: impl Fn(f64) -> f64) -> Raster {
    let fmt = r.format();
    let bands = fmt.channels();
    let in_bpc = fmt.bytes_per_channel();
    let out_fmt = PixelFormat::with_channels(bands, out_bpc)
        .expect("band count unchanged, so the output format exists");
    let n = r.width() as usize * r.height() as usize * bands;
    let max = depth_max(out_bpc);
    let mut out = vec![0u8; n * out_bpc];
    let data = r.data();
    for i in 0..n {
        write_f64(&mut out, out_bpc, i, f(read_f64(data, in_bpc, i)), max);
    }
    Raster::new(r.width(), r.height(), out_fmt, out).expect("arithmetic output is well-formed")
}

/// Apply integer `f` to every sample, keeping the input depth. `f` results
/// are masked into the depth by the caller-provided closure contract.
fn unary_map_u32(r: &Raster, f: impl Fn(u32) -> u32) -> Raster {
    let fmt = r.format();
    let bpc = fmt.bytes_per_channel();
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    let mut out = vec![0u8; n * bpc];
    let data = r.data();
    for i in 0..n {
        write_u32(&mut out, bpc, i, f(read_u32(data, bpc, i)));
    }
    Raster::new(r.width(), r.height(), fmt, out).expect("arithmetic output is well-formed")
}

/// Apply per-band `f(sample, band_constant)` to every sample.
fn vec_map(
    r: &Raster,
    v: &[f64],
    out_bpc: usize,
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
    let in_bpc = fmt.bytes_per_channel();
    let out_fmt = format_for(bands, out_bpc)?;
    let n = r.width() as usize * r.height() as usize * bands;
    let max = depth_max(out_bpc);
    let mut out = vec![0u8; n * out_bpc];
    let data = r.data();
    for i in 0..n {
        write_f64(
            &mut out,
            out_bpc,
            i,
            f(read_f64(data, in_bpc, i), v[i % bands]),
            max,
        );
    }
    Ok(Raster::new(r.width(), r.height(), out_fmt, out)?)
}

/// Apply `f` samplewise across two compatible images. Output depth is the
/// wider input depth, widened to 16-bit when `widen` is set.
fn binary_map(
    a: &Raster,
    b: &Raster,
    widen: bool,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Raster, ArithmeticError> {
    ensure_compatible(a, b)?;
    let (a_bpc, b_bpc) = (
        a.format().bytes_per_channel(),
        b.format().bytes_per_channel(),
    );
    let out_bpc = if widen { 2 } else { a_bpc.max(b_bpc) };
    let out_fmt = format_for(a.format().channels(), out_bpc)?;
    let n = a.width() as usize * a.height() as usize * a.format().channels();
    let max = depth_max(out_bpc);
    let mut out = vec![0u8; n * out_bpc];
    let (a_data, b_data) = (a.data(), b.data());
    for i in 0..n {
        write_f64(
            &mut out,
            out_bpc,
            i,
            f(read_f64(a_data, a_bpc, i), read_f64(b_data, b_bpc, i)),
            max,
        );
    }
    Ok(Raster::new(a.width(), a.height(), out_fmt, out)?)
}

/// Apply integer `f` samplewise across two compatible images, masking into
/// the wider input depth.
fn binary_map_u32(
    a: &Raster,
    b: &Raster,
    f: impl Fn(u32, u32) -> u32,
) -> Result<Raster, ArithmeticError> {
    ensure_compatible(a, b)?;
    let (a_bpc, b_bpc) = (
        a.format().bytes_per_channel(),
        b.format().bytes_per_channel(),
    );
    let out_bpc = a_bpc.max(b_bpc);
    let mask = depth_max_u32(out_bpc);
    let out_fmt = format_for(a.format().channels(), out_bpc)?;
    let n = a.width() as usize * a.height() as usize * a.format().channels();
    let mut out = vec![0u8; n * out_bpc];
    let (a_data, b_data) = (a.data(), b.data());
    for i in 0..n {
        let v = f(read_u32(a_data, a_bpc, i), read_u32(b_data, b_bpc, i)) & mask;
        write_u32(&mut out, out_bpc, i, v);
    }
    Ok(Raster::new(a.width(), a.height(), out_fmt, out)?)
}

/// Samplewise relational op across two compatible images: 8-bit output with
/// `255` where the relation holds.
fn compare_map(
    a: &Raster,
    b: &Raster,
    f: impl Fn(f64, f64) -> bool,
) -> Result<Raster, ArithmeticError> {
    ensure_compatible(a, b)?;
    let (a_bpc, b_bpc) = (
        a.format().bytes_per_channel(),
        b.format().bytes_per_channel(),
    );
    let out_fmt = format_for(a.format().channels(), 1)?;
    let n = a.width() as usize * a.height() as usize * a.format().channels();
    let mut out = vec![0u8; n];
    let (a_data, b_data) = (a.data(), b.data());
    for (i, o) in out.iter_mut().enumerate() {
        *o = if f(read_f64(a_data, a_bpc, i), read_f64(b_data, b_bpc, i)) {
            255
        } else {
            0
        };
    }
    Ok(Raster::new(a.width(), a.height(), out_fmt, out)?)
}

/// Samplewise relational op against a constant: 8-bit output with `255`
/// where the relation holds.
fn compare_const_map(r: &Raster, c: f64, f: impl Fn(f64, f64) -> bool) -> Raster {
    let fmt = r.format();
    let bpc = fmt.bytes_per_channel();
    let out_fmt = PixelFormat::with_channels(fmt.channels(), 1)
        .expect("band count unchanged, so the output format exists");
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    let mut out = vec![0u8; n];
    let data = r.data();
    for (i, o) in out.iter_mut().enumerate() {
        *o = if f(read_f64(data, bpc, i), c) { 255 } else { 0 };
    }
    Raster::new(r.width(), r.height(), out_fmt, out).expect("arithmetic output is well-formed")
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

impl Raster {
    // -----------------------------------------------------------------
    // Reductions
    // -----------------------------------------------------------------

    /// Mean of every sample in every band (libvips `avg`).
    pub fn avg(&self) -> f64 {
        let bpc = self.format().bytes_per_channel();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        let sum: f64 = (0..n).map(|i| read_f64(data, bpc, i)).sum();
        sum / n as f64
    }

    /// Sample standard deviation of every sample in every band (libvips
    /// `deviate`, using the `n - 1` denominator). A single-sample image has
    /// deviation `0`.
    pub fn deviate(&self) -> f64 {
        let bpc = self.format().bytes_per_channel();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        if n < 2 {
            return 0.0;
        }
        let data = self.data();
        let (mut sum, mut sum2) = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = read_f64(data, bpc, i);
            sum += v;
            sum2 += v * v;
        }
        (((sum2 - sum * sum / n as f64) / (n as f64 - 1.0)).max(0.0)).sqrt()
    }

    /// Smallest sample across every band (libvips `min`).
    pub fn min(&self) -> f64 {
        let bpc = self.format().bytes_per_channel();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        (0..n)
            .map(|i| read_f64(data, bpc, i))
            .fold(f64::MAX, f64::min)
    }

    /// Largest sample across every band (libvips `max`).
    pub fn max(&self) -> f64 {
        let bpc = self.format().bytes_per_channel();
        let n = self.width() as usize * self.height() as usize * self.format().channels();
        let data = self.data();
        (0..n)
            .map(|i| read_f64(data, bpc, i))
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
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let data = self.data();
        let mut best = read_f64(data, bpc, 0);
        let (mut bx, mut by) = (0u32, 0u32);
        for y in 0..h {
            for x in 0..w {
                for c in 0..bands {
                    let v = read_f64(data, bpc, (y * w + x) * bands + c);
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
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        let pixels = self.width() as usize * self.height() as usize;
        let data = self.data();

        // Per-band accumulators: min, max, sum, sum2.
        let mut acc = vec![(f64::MAX, f64::MIN, 0.0f64, 0.0f64); bands];
        for p in 0..pixels {
            for (c, a) in acc.iter_mut().enumerate() {
                let v = read_f64(data, bpc, p * bands + c);
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
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
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
                            *s += read_f64(data, bpc, (y * w + x) * bands + c);
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
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
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
                    (read_f64(data, bpc, (y * w + x) * bands + c) - bg[c]).abs()
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
    /// saturating at `65535`, matching the libvips `ushort` output.
    pub fn profile(&self) -> (Raster, Raster) {
        let fmt = self.format();
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let out_fmt = PixelFormat::with_channels(bands, 2)
            .expect("band count unchanged, so the output format exists");
        let data = self.data();

        let mut cols = vec![0u8; w * bands * 2];
        for x in 0..w {
            for c in 0..bands {
                let first = (0..h)
                    .find(|&y| read_u32(data, bpc, (y * w + x) * bands + c) != 0)
                    .unwrap_or(h);
                write_u32(&mut cols, 2, x * bands + c, first.min(0xFFFF) as u32);
            }
        }
        let mut rows = vec![0u8; h * bands * 2];
        for y in 0..h {
            for c in 0..bands {
                let first = (0..w)
                    .find(|&x| read_u32(data, bpc, (y * w + x) * bands + c) != 0)
                    .unwrap_or(w);
                write_u32(&mut rows, 2, y * bands + c, first.min(0xFFFF) as u32);
            }
        }
        (
            Raster::new(self.width(), 1, out_fmt, cols).expect("profile output is well-formed"),
            Raster::new(1, self.height(), out_fmt, rows).expect("profile output is well-formed"),
        )
    }

    /// Column and row sums (libvips `project`).
    ///
    /// Returns `(columns, rows)`: `columns` is a `width x 1` image holding
    /// the per-band sum of each column; `rows` is a `1 x height` image
    /// holding the per-band sum of each row. Outputs are 16-bit and sums
    /// saturate at `65535` (libvips promotes to a 32-bit format this crate
    /// does not have).
    pub fn project(&self) -> (Raster, Raster) {
        let fmt = self.format();
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let out_fmt = PixelFormat::with_channels(bands, 2)
            .expect("band count unchanged, so the output format exists");
        let data = self.data();

        let mut col_sums = vec![0.0f64; w * bands];
        let mut row_sums = vec![0.0f64; h * bands];
        for y in 0..h {
            for x in 0..w {
                for c in 0..bands {
                    let v = read_f64(data, bpc, (y * w + x) * bands + c);
                    col_sums[x * bands + c] += v;
                    row_sums[y * bands + c] += v;
                }
            }
        }
        let mut cols = vec![0u8; w * bands * 2];
        for (i, &s) in col_sums.iter().enumerate() {
            write_f64(&mut cols, 2, i, s, 65535.0);
        }
        let mut rows = vec![0u8; h * bands * 2];
        for (i, &s) in row_sums.iter().enumerate() {
            write_f64(&mut rows, 2, i, s, 65535.0);
        }
        (
            Raster::new(self.width(), 1, out_fmt, cols).expect("project output is well-formed"),
            Raster::new(1, self.height(), out_fmt, rows).expect("project output is well-formed"),
        )
    }

    // -----------------------------------------------------------------
    // Constant arithmetic
    // -----------------------------------------------------------------

    /// Add a constant to every sample (libvips `linear` with `a = 1`).
    /// 8-bit input promotes to 16-bit so sums above 255 survive.
    pub fn add_const(&self, c: f64) -> Raster {
        unary_map(self, 2, |v| v + c)
    }

    /// Subtract a constant from every sample, saturating at `0`.
    pub fn sub_const(&self, c: f64) -> Raster {
        unary_map(self, self.format().bytes_per_channel(), |v| v - c)
    }

    /// Multiply every sample by a constant. 8-bit input promotes to 16-bit
    /// so products above 255 survive.
    pub fn mul_const(&self, c: f64) -> Raster {
        unary_map(self, 2, |v| v * c)
    }

    /// Divide every sample by a constant; division by zero produces `0`,
    /// matching libvips.
    pub fn div_const(&self, c: f64) -> Raster {
        unary_map(self, self.format().bytes_per_channel(), move |v| {
            if c == 0.0 { 0.0 } else { v / c }
        })
    }

    /// Floor-divide every sample by a constant (Python `//`); division by
    /// zero produces `0`.
    pub fn floordiv_const(&self, c: f64) -> Raster {
        unary_map(self, self.format().bytes_per_channel(), move |v| {
            if c == 0.0 { 0.0 } else { (v / c).floor() }
        })
    }

    /// Raise every sample to a power. 8-bit input promotes to 16-bit;
    /// results saturate at the output depth.
    pub fn pow_const(&self, exp: f64) -> Raster {
        unary_map(self, 2, move |v| v.powf(exp))
    }

    /// Remainder of every sample divided by a constant (libvips
    /// `remainder_const`); a zero divisor produces `0`.
    pub fn rem_const(&self, c: f64) -> Raster {
        unary_map(self, self.format().bytes_per_channel(), move |v| {
            if c == 0.0 { 0.0 } else { v % c }
        })
    }

    /// `a * sample + b` for every sample (libvips `linear`). 8-bit input
    /// promotes to 16-bit; results saturate at the output depth and at `0`.
    pub fn linear(&self, a: f64, b: f64) -> Raster {
        unary_map(self, 2, move |v| a * v + b)
    }

    /// Per-band constant addition (libvips `add` with a vector constant);
    /// 8-bit input promotes to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band.
    pub fn try_add_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map(self, v, 2, |s, c| s + c)
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
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band.
    pub fn try_sub_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map(self, v, self.format().bytes_per_channel(), |s, c| s - c)
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
    /// one element per band.
    pub fn try_mul_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map(self, v, 2, |s, c| s * c)
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

    /// Per-band constant division; division by zero produces `0`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ConstCountMismatch`] if `v` does not have
    /// one element per band.
    pub fn try_div_vec(&self, v: &[f64]) -> Result<Raster, ArithmeticError> {
        vec_map(self, v, self.format().bytes_per_channel(), |s, c| {
            if c == 0.0 { 0.0 } else { s / c }
        })
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

    /// Absolute value of every sample (libvips `abs`). Samples are
    /// unsigned, so this is an identity copy; it exists so the ported
    /// call surface composes.
    pub fn abs(&self) -> Raster {
        self.clone()
    }

    /// Sign of every sample (libvips `sign`): `1` for positive samples,
    /// `0` for zero. Samples are unsigned, so `-1` cannot occur.
    pub fn sign(&self) -> Raster {
        unary_map_u32(self, |v| u32::from(v > 0))
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
        unary_map(self, self.format().bytes_per_channel(), move |v| {
            v.clamp(lo, hi)
        })
    }

    /// Round every sample down (libvips `floor`): an exact identity on the
    /// integer formats this crate stores.
    pub fn floor(&self) -> Raster {
        self.clone()
    }

    /// Round every sample up (libvips `ceil`): an exact identity on the
    /// integer formats this crate stores.
    pub fn ceil(&self) -> Raster {
        self.clone()
    }

    /// Round every sample to the nearest integer (libvips `rint`): an
    /// exact identity on the integer formats this crate stores.
    pub fn rint(&self) -> Raster {
        self.clone()
    }

    // -----------------------------------------------------------------
    // Image-image arithmetic
    // -----------------------------------------------------------------

    /// Subtract `other` from `self` samplewise, saturating at `0` (libvips
    /// `subtract`; unsigned formats cannot carry negative differences).
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_sub(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map(self, other, false, |a, b| a - b)
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

    /// Multiply two images samplewise (libvips `multiply`); 8-bit inputs
    /// promote to 16-bit and results saturate at the output depth.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_mul(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map(self, other, true, |a, b| a * b)
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

    /// Divide `self` by `other` samplewise (libvips `divide`); division by
    /// zero produces `0`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_div(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map(
            self,
            other,
            false,
            |a, b| if b == 0.0 { 0.0 } else { a / b },
        )
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

    /// Samplewise minimum of two images (libvips `minpair`); mixed depths
    /// promote numerically to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_minpair(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map(self, other, false, f64::min)
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
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_maxpair(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map(self, other, false, f64::max)
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
        let out_fmt = format_for(bands, 2)?;
        let n = first.width() as usize * first.height() as usize * bands;
        let mut out = vec![0u8; n * 2];
        for i in 0..n {
            let total: f64 = images
                .iter()
                .map(|r| read_f64(r.data(), r.format().bytes_per_channel(), i))
                .sum();
            write_f64(&mut out, 2, i, total, 65535.0);
        }
        Ok(Raster::new(first.width(), first.height(), out_fmt, out)?)
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

    /// Panicking form of [`Raster::try_more_than`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_more_than`].
    #[track_caller]
    pub fn more_than(&self, other: &Raster) -> Raster {
        expect_arith("more_than", self.try_more_than(other))
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

    /// Panicking form of [`Raster::try_more_eq`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_more_eq`].
    #[track_caller]
    pub fn more_eq(&self, other: &Raster) -> Raster {
        expect_arith("more_eq", self.try_more_eq(other))
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

    /// Panicking form of [`Raster::try_less_than`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_less_than`].
    #[track_caller]
    pub fn less_than(&self, other: &Raster) -> Raster {
        expect_arith("less_than", self.try_less_than(other))
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

    /// Panicking form of [`Raster::try_less_eq`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_less_eq`].
    #[track_caller]
    pub fn less_eq(&self, other: &Raster) -> Raster {
        expect_arith("less_eq", self.try_less_eq(other))
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

    /// Panicking form of [`Raster::try_equal`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_equal`].
    #[track_caller]
    pub fn equal(&self, other: &Raster) -> Raster {
        expect_arith("equal", self.try_equal(other))
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

    /// Panicking form of [`Raster::try_noteq`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_noteq`].
    #[track_caller]
    pub fn noteq(&self, other: &Raster) -> Raster {
        expect_arith("noteq", self.try_noteq(other))
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
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_bitand(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32(self, other, |a, b| a & b)
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

    /// Bitwise AND of every sample with a constant. The constant is masked
    /// into the sample depth (two's complement, so `-1` is all ones).
    pub fn bitand_const(&self, c: i64) -> Raster {
        let mask = (c as u64 & depth_max_u32(self.format().bytes_per_channel()) as u64) as u32;
        unary_map_u32(self, move |v| v & mask)
    }

    /// Samplewise bitwise OR of two images.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_bitor(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32(self, other, |a, b| a | b)
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
        let mask = (c as u64 & depth_max_u32(self.format().bytes_per_channel()) as u64) as u32;
        unary_map_u32(self, move |v| v | mask)
    }

    /// Samplewise bitwise XOR of two images.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::DimensionMismatch`] or
    /// [`ArithmeticError::BandCountMismatch`] if the images disagree.
    pub fn try_bitxor(&self, other: &Raster) -> Result<Raster, ArithmeticError> {
        binary_map_u32(self, other, |a, b| a ^ b)
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
        let mask = (c as u64 & depth_max_u32(self.format().bytes_per_channel()) as u64) as u32;
        unary_map_u32(self, move |v| v ^ mask)
    }

    /// Bitwise NOT of every sample within its depth (libvips `invert` for
    /// integer formats): `!v & 0xFF` for 8-bit, `!v & 0xFFFF` for 16-bit.
    pub fn bitnot(&self) -> Raster {
        let mask = depth_max_u32(self.format().bytes_per_channel());
        unary_map_u32(self, move |v| !v & mask)
    }

    /// Shift every sample left by `n` bits, truncating into the sample
    /// depth (the same wrap-in-format behavior as the libvips integer
    /// path). Shifts of the full sample width or more produce `0`.
    pub fn lshift(&self, n: u32) -> Raster {
        let mask = depth_max_u32(self.format().bytes_per_channel());
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
            unary_map(self, 1, move |v| f * (1.0 + v.powf(SCALE_LOG_EXP)).log10())
        } else {
            let (mn, mx) = (self.min(), self.max());
            let range = mx - mn;
            if range == 0.0 {
                unary_map(self, 1, |_| 0.0)
            } else {
                unary_map(self, 1, move |v| (v - mn) * 255.0 / range)
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
    /// independently; at the edges the window is clipped to the image
    /// (libvips mirrors instead, which differs only in a border of
    /// `window/2` pixels). The output keeps the input format.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::ZeroWindow`] if either window dimension
    /// is zero.
    pub fn try_stdif(&self, width: u32, height: u32) -> Result<Raster, ArithmeticError> {
        if width == 0 || height == 0 {
            return Err(ArithmeticError::ZeroWindow);
        }
        let fmt = self.format();
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        let (w, h) = (self.width() as usize, self.height() as usize);
        let max = depth_max(bpc);
        let data = self.data();
        let mut out = vec![0u8; w * h * bands * bpc];

        // Integral images per band: s[y][x] holds the sum over the
        // rectangle [0, x) x [0, y), so any window sum is four lookups.
        let stride = w + 1;
        let mut s = vec![0.0f64; stride * (h + 1)];
        let mut s2 = vec![0.0f64; stride * (h + 1)];
        for band in 0..bands {
            for y in 0..h {
                for x in 0..w {
                    let v = read_f64(data, bpc, (y * w + x) * bands + band);
                    let i = (y + 1) * stride + (x + 1);
                    s[i] = v + s[i - 1] + s[i - stride] - s[i - stride - 1];
                    s2[i] = v * v + s2[i - 1] + s2[i - stride] - s2[i - stride - 1];
                }
            }
            for y in 0..h {
                let y0 = (y as i64 - height as i64 / 2).max(0) as usize;
                let y1 = ((y as i64 - height as i64 / 2) + height as i64).min(h as i64) as usize;
                for x in 0..w {
                    let x0 = (x as i64 - width as i64 / 2).max(0) as usize;
                    let x1 = ((x as i64 - width as i64 / 2) + width as i64).min(w as i64) as usize;
                    let npel = ((x1 - x0) * (y1 - y0)) as f64;
                    let win = |t: &[f64]| {
                        t[y1 * stride + x1] - t[y0 * stride + x1] - t[y1 * stride + x0]
                            + t[y0 * stride + x0]
                    };
                    let mean = win(&s) / npel;
                    let var = (win(&s2) / npel - mean * mean).max(0.0);
                    let dev = var.sqrt();
                    let v = read_f64(data, bpc, (y * w + x) * bands + band);
                    let res = STDIF_A * STDIF_M0
                        + (1.0 - STDIF_A) * mean
                        + (v - mean) * (STDIF_B * STDIF_S0) / (STDIF_B * dev + STDIF_S0);
                    write_f64(&mut out, bpc, (y * w + x) * bands + band, res, max);
                }
            }
        }
        Ok(Raster::new(self.width(), self.height(), fmt, out)?)
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
    /// coefficient per input band, or [`ArithmeticError::TooManyBands`] if
    /// the output band count exceeds `u16::MAX`.
    pub fn try_recomb(&self, matrix: &[&[f64]]) -> Result<Raster, ArithmeticError> {
        if matrix.is_empty() {
            return Err(ArithmeticError::EmptyMatrix);
        }
        let fmt = self.format();
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
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
        let out_fmt = format_for(out_bands, bpc)?;
        let pixels = self.width() as usize * self.height() as usize;
        let max = depth_max(bpc);
        let data = self.data();
        let mut out = vec![0u8; pixels * out_bands * bpc];
        for p in 0..pixels {
            for (r, coeffs) in matrix.iter().enumerate() {
                let acc: f64 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(b, &m)| m * read_f64(data, bpc, p * bands + b))
                    .sum();
                write_f64(&mut out, bpc, p * out_bands + r, acc, max);
            }
        }
        Ok(Raster::new(self.width(), self.height(), out_fmt, out)?)
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
    /// `v * alpha / max` (`max` is `255` or `65535` by depth), rounded to
    /// nearest. The alpha band and the format are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NoAlphaBand`] if the image has fewer
    /// than two bands.
    pub fn try_premultiply(&self) -> Result<Raster, ArithmeticError> {
        self.alpha_map(|v, a, max| v * a / max)
    }

    /// Panicking form of [`Raster::try_premultiply`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see [`Raster::try_premultiply`].
    #[track_caller]
    pub fn premultiply(&self) -> Raster {
        expect_arith("premultiply", self.try_premultiply())
    }

    /// Undo alpha premultiplication (libvips `unpremultiply`).
    ///
    /// The last band is the alpha band; every other band becomes
    /// `v * max / alpha` (saturated), or `0` where alpha is zero, matching
    /// libvips. The alpha band and the format are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ArithmeticError::NoAlphaBand`] if the image has fewer
    /// than two bands.
    pub fn try_unpremultiply(&self) -> Result<Raster, ArithmeticError> {
        self.alpha_map(|v, a, max| if a == 0.0 { 0.0 } else { v * max / a })
    }

    /// Panicking form of [`Raster::try_unpremultiply`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ArithmeticError`]; see
    /// [`Raster::try_unpremultiply`].
    #[track_caller]
    pub fn unpremultiply(&self) -> Raster {
        expect_arith("unpremultiply", self.try_unpremultiply())
    }

    /// Shared kernel for the alpha ops: apply `f(sample, alpha, max)` to
    /// every non-alpha band and copy the alpha band through.
    fn alpha_map(&self, f: impl Fn(f64, f64, f64) -> f64) -> Result<Raster, ArithmeticError> {
        let fmt = self.format();
        let (bands, bpc) = (fmt.channels(), fmt.bytes_per_channel());
        if bands < 2 {
            return Err(ArithmeticError::NoAlphaBand { bands });
        }
        let pixels = self.width() as usize * self.height() as usize;
        let max = depth_max(bpc);
        let data = self.data();
        let mut out = vec![0u8; pixels * bands * bpc];
        for p in 0..pixels {
            let alpha = read_f64(data, bpc, p * bands + bands - 1);
            for c in 0..bands - 1 {
                let v = read_f64(data, bpc, p * bands + c);
                write_f64(&mut out, bpc, p * bands + c, f(v, alpha, max), max);
            }
            write_f64(&mut out, bpc, p * bands + bands - 1, alpha, max);
        }
        Ok(Raster::new(self.width(), self.height(), fmt, out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A width x height Gray8 raster from a byte vector.
    fn gray(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /// A 1-band 16-bit raster from sample values.
    fn gray16(w: u32, h: u32, vals: &[u16]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Gray16, data).unwrap()
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

    /// div_const halves values; a zero divisor produces zero, matching
    /// libvips.
    #[test]
    fn div_const_and_zero_divisor() {
        let im = gray(1, 1, vec![100]);
        assert_eq!(im.div_const(2.0).getpoint(0, 0), vec![50.0]);
        assert_eq!(im.div_const(0.0).getpoint(0, 0), vec![0.0]);
    }

    /// floordiv_const floors the quotient instead of rounding.
    #[test]
    fn floordiv_const_floors() {
        let im = gray(1, 1, vec![100]);
        assert_eq!(im.floordiv_const(3.0).getpoint(0, 0), vec![33.0]);
        assert_eq!(im.div_const(3.0).getpoint(0, 0), vec![33.0]); // 33.33 rounds to 33
        let im9 = gray(1, 1, vec![9]);
        assert_eq!(im9.floordiv_const(5.0).getpoint(0, 0), vec![1.0]); // 1.8 floors to 1
        assert_eq!(im9.div_const(5.0).getpoint(0, 0), vec![2.0]); // 1.8 rounds to 2
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

    /// linear computes a*x + b with promotion.
    #[test]
    fn linear_values() {
        let im = gray(2, 1, vec![0, 100]);
        let r = im.linear(1.0, 10.0);
        assert_eq!(r.format(), PixelFormat::Gray16);
        assert_eq!(r.getpoint(0, 0), vec![10.0]);
        assert_eq!(r.getpoint(1, 0), vec![110.0]);
        assert!((r.avg() - 60.0).abs() < 1e-9);
        assert_eq!(im.linear(3.0, 5.0).getpoint(1, 0), vec![305.0]);
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

    /// sub_vec, mul_vec, and div_vec apply per-band constants with the
    /// documented saturation / promotion.
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
        assert_eq!(
            im.div_vec(&[2.0, 0.0, 3.0]).getpoint(0, 0),
            vec![5.0, 0.0, 10.0]
        );
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

    /// sub of an image from itself is all zeros; differences saturate at
    /// zero instead of wrapping.
    #[test]
    fn sub_zeros_and_saturation() {
        let a = gray(2, 1, vec![100, 10]);
        assert_eq!(a.sub(&a).avg(), 0.0);
        let b = gray(2, 1, vec![50, 200]);
        assert_eq!(a.sub(&b).getpoint(0, 0), vec![50.0]);
        assert_eq!(a.sub(&b).getpoint(1, 0), vec![0.0]);
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

    /// stdif processes bands independently and accepts windows larger than
    /// the image.
    #[test]
    fn stdif_multiband_and_large_window() {
        let im = Raster::new(2, 2, PixelFormat::Rgb8, [10u8, 40, 70].repeat(4)).unwrap();
        let r = im.stdif(10, 10);
        assert_eq!(r.format(), PixelFormat::Rgb8);
        // Constant bands: 0.5*128 + 0.5*band.
        assert_eq!(r.getpoint(0, 0), vec![69.0, 84.0, 99.0]);
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
            PixelFormat::with_channels(2, 1).unwrap(),
            vec![100, 128],
        )
        .unwrap();
        let r = im.premultiply();
        assert_eq!(r.getpoint(0, 0), vec![50.0, 128.0]); // 100*128/255 = 50.2

        let vals = [40_000u16, 32_768u16];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let im16 = Raster::new(1, 1, PixelFormat::with_channels(2, 2).unwrap(), data).unwrap();
        let r16 = im16.premultiply();
        assert_eq!(r16.getpoint(0, 0), vec![20_000.0, 32_768.0]); // 40000*32768/65535 ~ 20000.3
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
            PixelFormat::with_channels(2, 1).unwrap(),
            vec![200, 100],
        )
        .unwrap();
        assert_eq!(im.unpremultiply().getpoint(0, 0), vec![255.0, 100.0]);
    }
}
