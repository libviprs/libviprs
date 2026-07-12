//! Frequency-domain filter operations ported from libvips `freqfilt`.
//!
//! This module carries the FFT family: the forward and inverse discrete
//! Fourier transforms plus the three compound operations libvips builds
//! on them. Every operation is a faithful port of the corresponding
//! libvips C source (`libvips/freqfilt/*.c` plus the `cross_phase` macro
//! in `libvips/arithmetic/complex.c`), swapping FFTW for the pure-Rust
//! `rustfft` planner.
//!
//! | libviprs | libvips | notes |
//! |---|---|---|
//! | [`Raster::fwfft`] | `vips_fwfft` | forward transform, scaled `1/(w*h)` |
//! | [`Raster::invfft`] | `vips_invfft` | inverse transform, unscaled, complex output |
//! | [`Raster::invfft_real`] | `vips_invfft` with `real: TRUE` | inverse transform, real output |
//! | [`Raster::freqmult`] | `vips_freqmult` | multiply in frequency space |
//! | [`Raster::spectrum`] | `vips_spectrum` | displayable log-scaled power spectrum |
//! | [`Raster::phasecor`] | `vips_phasecor` | phase correlation of two images |
//!
//! # Conventions
//!
//! * **Complex images.** libvips has native complex band formats;
//!   libviprs does not. Following the convention established by
//!   [`crate::arithmetic`]'s complex family, a complex image here is a
//!   float raster with an even band count holding `(re, im)` pairs. To
//!   distinguish a Fourier-domain image from an ordinary even-band float
//!   image (libvips branches on `vips_band_format_iscomplex`), the
//!   compound operations treat an input as already-complex only when it
//!   has an even band count *and* is stamped
//!   [`Interpretation::Fourier`], which is exactly what [`Raster::fwfft`]
//!   produces. Any other input takes the real path, mirroring the
//!   band-splitting `vips__fftproc` driver: each band is transformed as
//!   an independent real (or, for `invfft`, zero-imaginary) signal.
//! * **Normalisation.** `fwfft` divides by `w*h` (libvips `fwfft.c`
//!   divides every output sample by `VIPS_IMAGE_N_PELS`); `invfft` is
//!   unnormalised (libvips `invfft.c` runs FFTW backward with no scale).
//!   `invfft(fwfft(x))` therefore round-trips to `x`.
//! * **Layout.** `fwfft` output is in FFT layout with the DC component at
//!   the origin `(0, 0)`; libvips does not re-centre it. [`Raster::spectrum`]
//!   is the displayable form: it applies the log-scale map and then
//!   [`Raster::wrap`] to move DC to the centre.
//! * **`invfft_real`.** libvips `rinvfft1` feeds only the left
//!   `w/2 + 1` columns to FFTW's complex-to-real transform, which
//!   reconstructs the rest by Hermitian symmetry. This port does the
//!   same: the right half of the input spectrum is ignored and rebuilt
//!   as the conjugate mirror of the left half, so results match libvips
//!   even for spectra that are not exactly Hermitian.
//! * **Precision.** Transforms run in `f64` and results are stored in
//!   [`PixelFormat::FloatF32`] rasters (libvips stores `dpcomplex` /
//!   `double`; libviprs carries float rasters as `f32`, the same
//!   depth trade documented by [`crate::create`] for its generators).
//! * **Sizes.** `rustfft`'s mixed-radix planner (Bluestein for large
//!   primes) handles any width and height, matching FFTW; images do not
//!   need power-of-two dimensions.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use thiserror::Error;

use crate::conversion::{ConversionError, Interpretation};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};

/// The `vips_scale` log-mode exponent (its libvips default), shared
/// with [`Raster::scaleimage`].
const SCALE_LOG_EXP: f64 = 0.25;

/// Typed errors for the frequency-domain operations in
/// [`crate::freqfilt`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FreqfiltError {
    /// Two rasters that must share pixel dimensions do not.
    #[error("{op}: dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        /// The operation that failed.
        op: &'static str,
        /// Width of the first image.
        expected_w: u32,
        /// Height of the first image.
        expected_h: u32,
        /// Width of the second image.
        got_w: u32,
        /// Height of the second image.
        got_h: u32,
    },
    /// The mask or second image's band count is incompatible: it must
    /// match the image's complex pair count, or be a single band (a
    /// single pair for a complex mask) to broadcast.
    #[error("{op}: band count mismatch: image has {image_pairs} complex pair(s), got {got}")]
    BandMismatch {
        /// The operation that failed.
        op: &'static str,
        /// Complex pair count of the transformed image.
        image_pairs: usize,
        /// Band (or pair) count of the incompatible input.
        got: usize,
    },
    /// The output band count would exceed the [`PixelFormat`] limit of
    /// `u16::MAX` bands.
    #[error("{op}: output band count {bands} exceeds the format limit")]
    TooManyBands {
        /// The operation that failed.
        op: &'static str,
        /// The band count that does not fit.
        bands: usize,
    },
    /// An underlying raster allocation failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
    /// A conversion step (`cast` in `freqmult`, `wrap` in `spectrum`)
    /// failed.
    #[error(transparent)]
    Conversion(#[from] ConversionError),
}

/// Panic with the standard message shape for the panicking wrappers.
#[track_caller]
fn expect_freq<T>(op: &str, r: Result<T, FreqfiltError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Whether this raster is a Fourier-domain complex image: an even band
/// count of `(re, im)` pairs stamped [`Interpretation::Fourier`]. This is
/// the libviprs analogue of libvips' `vips_band_format_iscomplex` branch
/// (see the module docs).
fn is_fourier_complex(r: &Raster) -> bool {
    r.format().channels() % 2 == 0 && r.interpretation() == Interpretation::Fourier
}

/// Read every sample as `f64` in raster order (row-major, bands
/// interleaved), whatever the depth.
fn samples_f64(r: &Raster) -> Vec<f64> {
    let fmt = r.format();
    let bpc = fmt.bytes_per_channel();
    let data = r.data();
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    (0..n)
        .map(|i| match bpc {
            1 => f64::from(data[i]),
            2 => f64::from(u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]])),
            _ => f64::from(f32::from_ne_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ])),
        })
        .collect()
}

/// The canonical float format for `bands` bands, or `TooManyBands`.
fn float_format(op: &'static str, bands: usize) -> Result<PixelFormat, FreqfiltError> {
    PixelFormat::with_channels(bands, 4).ok_or(FreqfiltError::TooManyBands { op, bands })
}

/// Build a float raster from `f64` samples in raster order.
fn float_raster(
    width: u32,
    height: u32,
    format: PixelFormat,
    samples: &[f64],
) -> Result<Raster, FreqfiltError> {
    let mut data = Vec::with_capacity(samples.len() * 4);
    for &v in samples {
        data.extend_from_slice(&(v as f32).to_ne_bytes());
    }
    Ok(Raster::new(width, height, format, data)?)
}

/// In-place 2D DFT over `buf` (row-major, `h` rows of `w`): row
/// transforms then column transforms, both unnormalised, forward
/// (`e^-i`) or inverse (`e^+i`) like FFTW's `FFTW_FORWARD` /
/// `FFTW_BACKWARD`. Callers apply the libvips `fwfft` `1/(w*h)` scale
/// themselves.
fn fft_2d(buf: &mut [Complex<f64>], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::new();
    let row_fft = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    // `process` splits the buffer into `h` chunks of length `w` and
    // transforms each: all rows in one call.
    row_fft.process(buf);

    let col_fft = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex::new(0.0, 0.0); h];
    for x in 0..w {
        for (y, c) in col.iter_mut().enumerate() {
            *c = buf[y * w + x];
        }
        col_fft.process(&mut col);
        for (y, c) in col.iter().enumerate() {
            buf[y * w + x] = *c;
        }
    }
}

/// The complex pairs of one Fourier-domain band pair `p` of `r`, or one
/// real band `p` (imaginary parts zero) when `complex_input` is false.
fn load_plane(samples: &[f64], bands: usize, p: usize, complex_input: bool) -> Vec<Complex<f64>> {
    let pixels = samples.len() / bands;
    (0..pixels)
        .map(|i| {
            if complex_input {
                Complex::new(samples[i * bands + 2 * p], samples[i * bands + 2 * p + 1])
            } else {
                Complex::new(samples[i * bands + p], 0.0)
            }
        })
        .collect()
}

/// Rebuild the full spectrum FFTW's complex-to-real path implies: keep
/// the left `w/2 + 1` columns and replace the right half with the
/// conjugate mirror `F[y][x] = conj(F[(h - y) % h][w - x])`, exactly the
/// half-complex slice libvips `rinvfft1` feeds to `fftw_plan_dft_c2r_2d`.
fn hermitian_from_left_half(buf: &mut [Complex<f64>], w: usize, h: usize) {
    let half_w = w / 2 + 1;
    for y in 0..h {
        for x in half_w..w {
            let src_y = (h - y) % h;
            let src_x = w - x;
            buf[y * w + x] = buf[src_y * w + src_x].conj();
        }
    }
}

impl Raster {
    // -----------------------------------------------------------------
    // Forward transform
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::fwfft`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::TooManyBands`] if doubling the band count
    /// overflows the format limit, or [`FreqfiltError::Raster`] on
    /// allocation failure.
    pub fn try_fwfft(&self) -> Result<Raster, FreqfiltError> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let bands = self.format().channels();
        let samples = samples_f64(self);
        let scale = 1.0 / (w as f64 * h as f64);

        let complex_input = is_fourier_complex(self);
        let (planes, out_bands) = if complex_input {
            (bands / 2, bands)
        } else {
            (bands, bands * 2)
        };
        let format = float_format("fwfft", out_bands)?;

        let mut out = vec![0.0f64; w * h * out_bands];
        for p in 0..planes {
            let mut buf = load_plane(&samples, bands, p, complex_input);
            fft_2d(&mut buf, w, h, false);
            for (i, c) in buf.iter().enumerate() {
                out[i * out_bands + 2 * p] = c.re * scale;
                out[i * out_bands + 2 * p + 1] = c.im * scale;
            }
        }

        let mut raster = float_raster(self.width(), self.height(), format, &out)?;
        raster.meta = self.meta;
        raster.meta.interpretation = Some(Interpretation::Fourier);
        Ok(raster)
    }

    /// Transform to Fourier space (libvips `vips_fwfft`): an
    /// unnormalised forward 2D DFT scaled by `1/(w*h)`, with the DC
    /// component at the origin. Each real band becomes a `(re, im)`
    /// float pair (libvips band-splits through `vips__fftproc`); a
    /// Fourier-domain complex input (see the module docs) is transformed
    /// pair-wise instead. The output is stamped
    /// [`Interpretation::Fourier`]. Panicking form of
    /// [`Raster::try_fwfft`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_fwfft`].
    #[track_caller]
    pub fn fwfft(&self) -> Raster {
        expect_freq("fwfft", self.try_fwfft())
    }

    // -----------------------------------------------------------------
    // Inverse transform
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::invfft`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::TooManyBands`] if doubling a real input's band
    /// count overflows the format limit, or [`FreqfiltError::Raster`] on
    /// allocation failure.
    pub fn try_invfft(&self) -> Result<Raster, FreqfiltError> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let bands = self.format().channels();
        let samples = samples_f64(self);

        // libvips `cinvfft1` casts any input to complex first: a real
        // band becomes a zero-imaginary pair, mirroring `vips_cast` to
        // `dpcomplex`.
        let complex_input = is_fourier_complex(self);
        let (planes, out_bands) = if complex_input {
            (bands / 2, bands)
        } else {
            (bands, bands * 2)
        };
        let format = float_format("invfft", out_bands)?;

        let mut out = vec![0.0f64; w * h * out_bands];
        for p in 0..planes {
            let mut buf = load_plane(&samples, bands, p, complex_input);
            fft_2d(&mut buf, w, h, true);
            for (i, c) in buf.iter().enumerate() {
                out[i * out_bands + 2 * p] = c.re;
                out[i * out_bands + 2 * p + 1] = c.im;
            }
        }

        let mut raster = float_raster(self.width(), self.height(), format, &out)?;
        raster.meta = self.meta;
        // Back in the spatial domain: drop the Fourier stamp (libvips
        // `invfft.c` retags the output B_W).
        raster.meta.interpretation = None;
        Ok(raster)
    }

    /// Transform from Fourier space (libvips `vips_invfft` with the
    /// default `real: FALSE`): an unnormalised inverse 2D DFT, so
    /// `invfft(fwfft(x))` round-trips. The output is complex, one
    /// `(re, im)` float pair per input pair; a non-Fourier input is cast
    /// to complex first (each band gains a zero imaginary half), like
    /// libvips. Panicking form of [`Raster::try_invfft`].
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_invfft`].
    #[track_caller]
    pub fn invfft(&self) -> Raster {
        expect_freq("invfft", self.try_invfft())
    }

    /// Fallible form of [`Raster::invfft_real`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::Raster`] on allocation failure.
    pub fn try_invfft_real(&self) -> Result<Raster, FreqfiltError> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let bands = self.format().channels();
        let samples = samples_f64(self);

        let complex_input = is_fourier_complex(self);
        let planes = if complex_input { bands / 2 } else { bands };
        let format = float_format("invfft", planes)?;

        let mut out = vec![0.0f64; w * h * planes];
        for p in 0..planes {
            let mut buf = load_plane(&samples, bands, p, complex_input);
            // libvips `rinvfft1` hands FFTW only the left half of the
            // spectrum; mirror that before the full inverse transform.
            hermitian_from_left_half(&mut buf, w, h);
            fft_2d(&mut buf, w, h, true);
            for (i, c) in buf.iter().enumerate() {
                out[i * planes + p] = c.re;
            }
        }

        let mut raster = float_raster(self.width(), self.height(), format, &out)?;
        raster.meta = self.meta;
        raster.meta.interpretation = None;
        Ok(raster)
    }

    /// Transform from Fourier space to a real image (libvips
    /// `vips_invfft` with `real: TRUE`): an unnormalised inverse 2D DFT
    /// keeping the real component, one float band per input pair. Like
    /// libvips' complex-to-real FFTW path, only the left `w/2 + 1`
    /// columns of the spectrum are read; the rest is reconstructed by
    /// Hermitian symmetry. Panicking form of [`Raster::try_invfft_real`].
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_invfft_real`].
    #[track_caller]
    pub fn invfft_real(&self) -> Raster {
        expect_freq("invfft_real", self.try_invfft_real())
    }

    // -----------------------------------------------------------------
    // Compound operations
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::freqmult`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::DimensionMismatch`] if `mask` is not the same
    /// size as the image, [`FreqfiltError::BandMismatch`] if the mask's
    /// band count neither matches the image's complex pair count nor is
    /// broadcastable from one band, or [`FreqfiltError::Raster`] /
    /// [`FreqfiltError::Conversion`] on allocation or cast failure.
    pub fn try_freqmult(&self, mask: &Raster) -> Result<Raster, FreqfiltError> {
        if self.width() != mask.width() || self.height() != mask.height() {
            return Err(FreqfiltError::DimensionMismatch {
                op: "freqmult",
                expected_w: self.width(),
                expected_h: self.height(),
                got_w: mask.width(),
                got_h: mask.height(),
            });
        }

        // libvips `freqmult.c`: a complex input skips the forward
        // transform; a real input is transformed, multiplied, inverted,
        // and cast back to its original format.
        if is_fourier_complex(self) {
            let product = fourier_multiply("freqmult", self, mask)?;
            product.try_invfft_real()
        } else {
            let fourier = self.try_fwfft()?;
            let product = fourier_multiply("freqmult", &fourier, mask)?;
            let real = product.try_invfft_real()?;
            Ok(real.try_cast(
                PixelFormat::with_channels(
                    self.format().channels(),
                    self.format().bytes_per_channel(),
                )
                .expect("input format is valid, so its channel/depth pair is too"),
            )?)
        }
    }

    /// Multiply the image by `mask` in Fourier space (libvips
    /// `vips_freqmult`): transform forward if not already
    /// Fourier-domain, multiply by the mask (a real mask, such as the
    /// [`Raster::mask_ideal`] family, scales both halves of each pair; a
    /// Fourier-domain complex mask multiplies pair-wise), transform back
    /// keeping the real component, and cast a real input back to its
    /// original format. Panicking form of [`Raster::try_freqmult`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_freqmult`].
    #[track_caller]
    pub fn freqmult(&self, mask: &Raster) -> Raster {
        expect_freq("freqmult", self.try_freqmult(mask))
    }

    /// Fallible form of [`Raster::spectrum`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::TooManyBands`], [`FreqfiltError::Raster`], or
    /// [`FreqfiltError::Conversion`]; see [`Raster::try_fwfft`].
    pub fn try_spectrum(&self) -> Result<Raster, FreqfiltError> {
        // libvips `spectrum.c`: fwfft unless complex, then
        // abs -> scale log -> wrap.
        let fourier = if is_fourier_complex(self) {
            self.clone()
        } else {
            self.try_fwfft()?
        };

        let (w, h) = (fourier.width() as usize, fourier.height() as usize);
        let bands = fourier.format().channels();
        let pairs = bands / 2;
        let samples = samples_f64(&fourier);

        // Complex magnitude, one band per pair (`vips_abs` on a complex
        // image).
        let mut mag = vec![0.0f64; w * h * pairs];
        for i in 0..w * h {
            for p in 0..pairs {
                let re = samples[i * bands + 2 * p];
                let im = samples[i * bands + 2 * p + 1];
                mag[i * pairs + p] = re.hypot(im);
            }
        }

        // `vips_scale` in log mode: the same curve as
        // [`Raster::scaleimage`] with `log = Some(true)`
        // (`255 / log10(1 + max^0.25) * log10(1 + v^0.25)`), applied
        // here directly because the magnitudes are float and the
        // integer-raster `scaleimage` path does not accept float input.
        // Rounding matches the arithmetic module's uchar write: round to
        // nearest, clamp, NaN to zero.
        let mx = mag.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom = (1.0 + mx.powf(SCALE_LOG_EXP)).log10();
        let f = if denom > 0.0 { 255.0 / denom } else { 0.0 };
        let format = PixelFormat::with_channels(pairs, 1).ok_or(FreqfiltError::TooManyBands {
            op: "spectrum",
            bands: pairs,
        })?;
        let data: Vec<u8> = mag
            .iter()
            .map(|&v| {
                let scaled = f * (1.0 + v.powf(SCALE_LOG_EXP)).log10();
                if scaled.is_nan() {
                    0
                } else {
                    scaled.round().clamp(0.0, 255.0) as u8
                }
            })
            .collect();
        let scaled = Raster::new(fourier.width(), fourier.height(), format, data)?;

        // `vips_wrap` moves DC to the centre.
        Ok(scaled.try_wrap()?)
    }

    /// Make a displayable power spectrum (libvips `vips_spectrum`): the
    /// image is transformed to Fourier space if it is not already, the
    /// complex magnitude is log-scaled to 8-bit through
    /// [`Raster::scaleimage`], and the quadrants are swapped with
    /// [`Raster::wrap`] so the DC component sits at the centre.
    /// Panicking form of [`Raster::try_spectrum`].
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_spectrum`].
    #[track_caller]
    pub fn spectrum(&self) -> Raster {
        expect_freq("spectrum", self.try_spectrum())
    }

    /// Fallible form of [`Raster::phasecor`].
    ///
    /// # Errors
    ///
    /// [`FreqfiltError::DimensionMismatch`] if the images differ in
    /// size, [`FreqfiltError::BandMismatch`] if their complex pair
    /// counts differ, or [`FreqfiltError::TooManyBands`] /
    /// [`FreqfiltError::Raster`] from the transforms.
    pub fn try_phasecor(&self, other: &Raster) -> Result<Raster, FreqfiltError> {
        if self.width() != other.width() || self.height() != other.height() {
            return Err(FreqfiltError::DimensionMismatch {
                op: "phasecor",
                expected_w: self.width(),
                expected_h: self.height(),
                got_w: other.width(),
                got_h: other.height(),
            });
        }

        // libvips `phasecor.c`: fwfft either input unless complex.
        let a = if is_fourier_complex(self) {
            self.clone()
        } else {
            self.try_fwfft()?
        };
        let b = if is_fourier_complex(other) {
            other.clone()
        } else {
            other.try_fwfft()?
        };

        let bands = a.format().channels();
        let pairs = bands / 2;
        if b.format().channels() / 2 != pairs {
            return Err(FreqfiltError::BandMismatch {
                op: "phasecor",
                image_pairs: pairs,
                got: b.format().channels() / 2,
            });
        }

        let (w, h) = (a.width(), a.height());
        let sa = samples_f64(&a);
        let sb = samples_f64(&b);

        // `vips_cross_phase`, ported exactly from the CROSS macro in
        // libvips `arithmetic/complex.c`.
        let mut cross = vec![0.0f64; sa.len()];
        for i in 0..sa.len() / 2 {
            let (re, im) = cross_phase(sa[2 * i], sa[2 * i + 1], sb[2 * i], sb[2 * i + 1]);
            cross[2 * i] = re;
            cross[2 * i + 1] = im;
        }
        let mut cross = float_raster(w, h, float_format("phasecor", bands)?, &cross)?;
        cross.meta.interpretation = Some(Interpretation::Fourier);

        cross.try_invfft_real()
    }

    /// Phase correlation of two images (libvips `vips_phasecor`): both
    /// are transformed to Fourier space if not already, the phase of
    /// their cross power spectrum is taken, and the result is
    /// transformed back keeping the real component. The peak of the
    /// output sits at the translation relating the two images.
    /// Panicking form of [`Raster::try_phasecor`].
    ///
    /// # Panics
    ///
    /// Panics on any [`FreqfiltError`]; see [`Raster::try_phasecor`].
    #[track_caller]
    pub fn phasecor(&self, other: &Raster) -> Raster {
        expect_freq("phasecor", self.try_phasecor(other))
    }
}

/// Multiply a Fourier-domain complex image by a mask, pair-wise
/// (libvips reaches this through `vips_multiply`, which handles the
/// complex/real and band-broadcast cases). A real mask scales both
/// halves of every pair; a Fourier-domain complex mask (even bands plus
/// the Fourier stamp) multiplies complex-wise. A one-band (or one-pair)
/// mask broadcasts across pairs.
fn fourier_multiply(
    op: &'static str,
    fourier: &Raster,
    mask: &Raster,
) -> Result<Raster, FreqfiltError> {
    let (w, h) = (fourier.width() as usize, fourier.height() as usize);
    let bands = fourier.format().channels();
    let pairs = bands / 2;
    let samples = samples_f64(fourier);
    let mask_samples = samples_f64(mask);
    let mask_bands = mask.format().channels();
    let mask_complex = is_fourier_complex(mask);
    let mask_planes = if mask_complex {
        mask_bands / 2
    } else {
        mask_bands
    };
    if mask_planes != pairs && mask_planes != 1 {
        return Err(FreqfiltError::BandMismatch {
            op,
            image_pairs: pairs,
            got: mask_planes,
        });
    }

    let mut out = vec![0.0f64; samples.len()];
    for i in 0..w * h {
        for p in 0..pairs {
            let mp = if mask_planes == 1 { 0 } else { p };
            let re = samples[i * bands + 2 * p];
            let im = samples[i * bands + 2 * p + 1];
            let (ore, oim) = if mask_complex {
                let mre = mask_samples[i * mask_bands + 2 * mp];
                let mim = mask_samples[i * mask_bands + 2 * mp + 1];
                (re * mre - im * mim, re * mim + im * mre)
            } else {
                let m = mask_samples[i * mask_bands + mp];
                (re * m, im * m)
            };
            out[i * bands + 2 * p] = ore;
            out[i * bands + 2 * p + 1] = oim;
        }
    }

    let format = float_format(op, bands)?;
    let mut raster = float_raster(fourier.width(), fourier.height(), format, &out)?;
    raster.meta = fourier.meta;
    raster.meta.interpretation = Some(Interpretation::Fourier);
    Ok(raster)
}

/// One sample of `vips_cross_phase`: the phase of the cross power
/// spectrum of `(x1, y1)` and `(x2, y2)`, normalised to unit modulus.
/// Ported branch-for-branch from the CROSS macro in libvips
/// `arithmetic/complex.c`, including its zero cases (either input zero,
/// or both imaginary halves zero, give `(0, 0)`).
fn cross_phase(x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64) {
    if (x1 == 0.0 && y1 == 0.0) || (x2 == 0.0 && y2 == 0.0) || (y1 == 0.0 && y2 == 0.0) {
        (0.0, 0.0)
    } else if y1.abs() > y2.abs() {
        let a = y2 / y1;
        let b = y1 + y2 * a;
        let re = (x1 + x2 * a) / b;
        let im = (x2 - x1 * a) / b;
        let modulus = re.hypot(im);
        (re / modulus, im / modulus)
    } else {
        let a = y1 / y2;
        let b = y2 + y1 * a;
        let re = (x1 * a + x2) / b;
        let im = (x2 * a - x1) / b;
        let modulus = re.hypot(im);
        (re / modulus, im / modulus)
    }
}

/// A deterministic single-band float test image with no symmetry, so
/// spectra exercise every bin.
#[cfg(test)]
fn test_image(width: u32, height: u32) -> Raster {
    let n = width as usize * height as usize;
    let samples: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.7).sin() * 100.0 + (i as f64 * 0.13).cos() * 40.0)
        .collect();
    float_raster(
        width,
        height,
        PixelFormat::with_channels(1, 4).expect("one float band"),
        &samples,
    )
    .expect("test image allocates")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fwfft of a constant image puts all energy in the DC bin at the
    /// origin (libvips does not re-centre), scaled by `1/(w*h)` so the
    /// DC value equals the constant.
    #[test]
    fn fwfft_constant_is_dc_only() {
        let im = Raster::new(8, 4, PixelFormat::Gray8, vec![200u8; 8 * 4]).expect("raster");
        let f = im.fwfft();
        assert_eq!(f.width(), 8);
        assert_eq!(f.height(), 4);
        assert_eq!(f.format().channels(), 2);
        assert_eq!(f.interpretation(), Interpretation::Fourier);
        let dc = f.getpoint(0, 0);
        assert!((dc[0] - 200.0).abs() < 1e-3, "DC re = {}", dc[0]);
        assert!(dc[1].abs() < 1e-3, "DC im = {}", dc[1]);
        for y in 0..4 {
            for x in 0..8 {
                if x == 0 && y == 0 {
                    continue;
                }
                let p = f.getpoint(x, y);
                assert!(
                    p[0].abs() < 1e-3 && p[1].abs() < 1e-3,
                    "bin ({x},{y}) = {p:?}"
                );
            }
        }
    }

    /// The ported `test_fwfft_small_image` body: a 2x1 black image
    /// transforms without panicking.
    #[test]
    fn fwfft_small_image() {
        let im = Raster::black(2, 1);
        let fft = im.fwfft();
        assert_eq!(fft.width(), 2);
        assert_eq!(fft.height(), 1);
        assert_eq!(fft.format().channels(), 2);
    }

    /// invfft(fwfft(x)) round-trips a non-power-of-two image: the real
    /// half matches the input and the imaginary half is zero. The real
    /// output variant matches too.
    #[test]
    fn fwfft_invfft_round_trip() {
        let im = test_image(5, 3);
        let expected = samples_f64(&im);

        let complex = im.fwfft().invfft();
        assert_eq!(complex.format().channels(), 2);
        let got = samples_f64(&complex);
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (got[2 * i] - e).abs() < 1e-3,
                "re[{i}]: {} vs {e}",
                got[2 * i]
            );
            assert!(got[2 * i + 1].abs() < 1e-3, "im[{i}]: {}", got[2 * i + 1]);
        }

        let real = im.fwfft().invfft_real();
        assert_eq!(real.format().channels(), 1);
        let got = samples_f64(&real);
        for (i, &e) in expected.iter().enumerate() {
            assert!((got[i] - e).abs() < 1e-3, "[{i}]: {} vs {e}", got[i]);
        }
    }

    /// fwfft of an n-band image transforms each band independently
    /// (libvips `vips__fftproc`): the pairs equal the per-band
    /// transforms.
    #[test]
    fn fwfft_multiband_matches_per_band() {
        // A 2-band float image assembled from two different planes.
        let a = test_image(4, 4);
        let b = Raster::new(4, 4, PixelFormat::Gray8, (0..16u8).map(|v| v * 3).collect())
            .expect("raster");
        let sa = samples_f64(&a);
        let sb = samples_f64(&b);
        let joined: Vec<f64> = (0..16).flat_map(|i| [sa[i], sb[i]]).collect();
        let im = float_raster(
            4,
            4,
            PixelFormat::with_channels(2, 4).expect("two float bands"),
            &joined,
        )
        .expect("raster");

        let f = im.fwfft();
        assert_eq!(f.format().channels(), 4);
        let fa = samples_f64(&a.fwfft());
        let fb = samples_f64(&b.fwfft());
        let fj = samples_f64(&f);
        for i in 0..16 {
            assert!((fj[4 * i] - fa[2 * i]).abs() < 1e-6);
            assert!((fj[4 * i + 1] - fa[2 * i + 1]).abs() < 1e-6);
            assert!((fj[4 * i + 2] - fb[2 * i]).abs() < 1e-6);
            assert!((fj[4 * i + 3] - fb[2 * i + 1]).abs() < 1e-6);
        }
    }

    /// fwfft of a Fourier-domain image takes the complex path: for a
    /// delta at `x = 1` in a 4x1 image, `fwfft(fwfft(x))` is the
    /// index-reversed input scaled by `1/(w*h)`, i.e. a `0.25` delta at
    /// `x = 3`.
    #[test]
    fn fwfft_complex_input_double_transform() {
        let im = float_raster(
            4,
            1,
            PixelFormat::with_channels(1, 4).expect("one float band"),
            &[0.0, 1.0, 0.0, 0.0],
        )
        .expect("raster");
        let ff = im.fwfft().fwfft();
        assert_eq!(ff.format().channels(), 2);
        let got = samples_f64(&ff);
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!((got[i] - e).abs() < 1e-9, "[{i}]: {} vs {e}", got[i]);
        }
    }

    /// freqmult by an all-ones mask is the identity, and the output is
    /// cast back to the input format (libvips `freqmult.c` casts to
    /// `in->BandFmt`).
    #[test]
    fn freqmult_ones_mask_is_identity() {
        let data: Vec<u8> = (0..24u8).map(|v| v.wrapping_mul(11)).collect();
        let im = Raster::new(6, 4, PixelFormat::Gray8, data.clone()).expect("raster");
        let mask = float_raster(
            6,
            4,
            PixelFormat::with_channels(1, 4).expect("one float band"),
            &[1.0; 24],
        )
        .expect("mask");
        let out = im.freqmult(&mask);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.data(), im.data());
    }

    /// freqmult validates geometry and mask bands with typed errors.
    #[test]
    fn freqmult_typed_errors() {
        let im = test_image(4, 4);
        let small = test_image(2, 2);
        assert!(matches!(
            im.try_freqmult(&small),
            Err(FreqfiltError::DimensionMismatch { op: "freqmult", .. })
        ));

        let mask3 = float_raster(
            4,
            4,
            PixelFormat::with_channels(3, 4).expect("three float bands"),
            &[1.0; 48],
        )
        .expect("mask");
        assert!(matches!(
            im.try_freqmult(&mask3),
            Err(FreqfiltError::BandMismatch { op: "freqmult", .. })
        ));
    }

    /// spectrum is 8-bit, and for a constant image the only non-zero
    /// bin is DC, re-centred by wrap: the maximum sits at
    /// `(w/2, h/2)` with value 255.
    #[test]
    fn spectrum_constant_peaks_at_centre() {
        let im = Raster::new(16, 16, PixelFormat::Gray8, vec![37u8; 256]).expect("raster");
        let s = im.spectrum();
        assert_eq!(s.format(), PixelFormat::Gray8);
        let (value, x, y) = s.maxpos();
        assert_eq!((x, y), (8, 8));
        assert!((value - 255.0).abs() < f64::EPSILON);
    }

    /// phasecor of an image with a toroidally shifted copy peaks at the
    /// shift: with `B(x, y) = A((x+3) % w, (y+2) % h)`, the peak of
    /// `phasecor(A, B)` is at `(3, 2)` (verified against the libvips
    /// CROSS macro semantics numerically).
    #[test]
    fn phasecor_peaks_at_shift() {
        let (w, h) = (8u32, 8u32);
        let a = test_image(w, h);
        let sa = samples_f64(&a);
        let (dx, dy) = (3u32, 2u32);
        let shifted: Vec<f64> = (0..h)
            .flat_map(|y| {
                let sa = &sa;
                (0..w).map(move |x| {
                    let sx = (x + dx) % w;
                    let sy = (y + dy) % h;
                    sa[(sy * w + sx) as usize]
                })
            })
            .collect();
        let b = float_raster(
            w,
            h,
            PixelFormat::with_channels(1, 4).expect("one float band"),
            &shifted,
        )
        .expect("raster");

        let c = a.phasecor(&b);
        assert_eq!(c.format().channels(), 1);
        let (_, x, y) = c.maxpos();
        assert_eq!((x, y), (dx, dy));
    }

    /// phasecor validates geometry with a typed error.
    #[test]
    fn phasecor_dimension_mismatch() {
        let a = test_image(4, 4);
        let b = test_image(5, 4);
        assert!(matches!(
            a.try_phasecor(&b),
            Err(FreqfiltError::DimensionMismatch { op: "phasecor", .. })
        ));
    }

    /// The cross-phase zero cases from the libvips CROSS macro: either
    /// input zero, or both imaginary halves zero, give `(0, 0)`; a unit
    /// rotation comes back normalised.
    #[test]
    fn cross_phase_matches_libvips_macro() {
        assert_eq!(cross_phase(0.0, 0.0, 1.0, 2.0), (0.0, 0.0));
        assert_eq!(cross_phase(1.0, 2.0, 0.0, 0.0), (0.0, 0.0));
        assert_eq!(cross_phase(3.0, 0.0, 4.0, 0.0), (0.0, 0.0));
        // Same phase in both inputs: the cross phase is purely real.
        let (re, im) = cross_phase(1.0, 1.0, 2.0, 2.0);
        assert!((re - 1.0).abs() < 1e-12);
        assert!(im.abs() < 1e-12);
        // Unit modulus in the general case.
        let (re, im) = cross_phase(0.3, -1.7, 2.2, 0.9);
        assert!((re.hypot(im) - 1.0).abs() < 1e-12);
    }

    /// invfft on a real (non-Fourier) input casts to complex first,
    /// like libvips: each band gains a zero imaginary half before the
    /// unnormalised inverse transform.
    #[test]
    fn invfft_real_input_casts_to_complex() {
        let im = test_image(4, 3);
        let out = im.invfft();
        assert_eq!(out.format().channels(), 2);
        // The DC bin of an unnormalised inverse transform is the sum.
        let sum: f64 = samples_f64(&im).iter().sum();
        let dc = out.getpoint(0, 0);
        assert!((dc[0] - sum).abs() < 0.35, "{} vs {sum}", dc[0]);
    }
}
