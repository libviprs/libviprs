//! Band (channel) operations ported from libvips.
//!
//! This module is the first batch of the libvips operation surface required
//! by the ported integration tests: it covers the band axis only. Each
//! operation exists in two forms:
//!
//! * a fallible `try_*` method returning `Result<Raster, BandError>`, the
//!   primary implementation with typed errors for mismatched dimensions,
//!   band counts, and out-of-range indices; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`bandjoin`, `extract_band`, ...) exactly. These delegate to the
//!   `try_*` form and panic with the typed error's message, mirroring the
//!   "known-good input" contract of [`Raster::add`] and [`Raster::getpoint`].
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::bandjoin`] | `vips_bandjoin2` | bands of both images concatenated |
//! | [`Raster::bandjoin_const`] | `vips_bandjoin_const1` | one constant band appended |
//! | [`Raster::bandjoin_vec`] | `vips_bandjoin_const` | one constant band per element appended |
//! | [`Raster::bandfold`] | `vips_bandfold` | width folded into bands |
//! | [`Raster::bandunfold`] | `vips_bandunfold` | bands unfolded into width |
//! | [`Raster::bandmean`] | `vips_bandmean` | per-pixel mean across bands (1 band) |
//! | [`Raster::bandrank`] | `vips_bandrank` | per-sample rank statistic across images |
//! | [`Raster::bandand`] | `vips_bandbool` (AND) | bitwise AND across bands (1 band) |
//! | [`Raster::bandor`] | `vips_bandbool` (OR) | bitwise OR across bands (1 band) |
//! | [`Raster::bandeor`] | `vips_bandbool` (EOR) | bitwise XOR across bands (1 band) |
//! | [`Raster::extract_band`] | `vips_extract_band` (n=1) | one band |
//! | [`Raster::extract_bands`] | `vips_extract_band` | a consecutive band range |
//!
//! # Semantics shared by every operation
//!
//! * **Band counts.** Results whose band count is 1, 3, or 4 use the named
//!   [`PixelFormat`] variants; any other count uses
//!   [`PixelFormat::Multi8`] / [`PixelFormat::Multi16`]
//!   (via [`PixelFormat::with_channels`]), so a 3-band result always
//!   compares equal to `Rgb8`/`Rgb16`.
//! * **Depth promotion.** Operations over multiple images promote to the
//!   widest input depth. Promotion is numeric (a `200` stays `200`), the
//!   same rule libvips uses when casting `uchar` to `ushort` for a join.
//! * **Constants.** `f64` constants are clamped to the format range
//!   (`0..=255` or `0..=65535`) and rounded to nearest. libvips does not
//!   exercise non-integral constants in its band tests, so the rounding mode
//!   is our choice; round-to-nearest is documented here as the contract.
//! * **Negative band indices.** `extract_band(-1)` addresses the last band.
//!   libvips itself has no negative indexing; the ported tests fold pyvips's
//!   Python `im[-1]` sugar into the Rust call, so the resolution happens
//!   here.
//! * **Metadata.** Every operation carries its input's interpretation,
//!   resolution, offsets, orientation and attached fields onto its result
//!   (issue #727). Nothing here stamps the origin offset, unlike `flip`,
//!   `rot`, `wrap` and the convolving ops, and `bandfold` / `bandunfold`
//!   reshape the pixel grid without rescaling the resolution. `bandjoin` and
//!   `bandrank` take more than one input, so they follow the multi-input rule
//!   (issue #718): the header block comes from `self` alone and the attached
//!   fields are the union of every input, `self` winning a name they share.
//!
//! Multiband (`Multi8`/`Multi16`) rasters are compute intermediates: the
//! tile-encoding sinks reject them, so reduce or extract back to 1/3/4 bands
//! before handing a raster to the pyramid pipeline.

use crate::pixel::{PixelFormat, SampleKind};
use crate::raster::{Raster, RasterError};
use thiserror::Error;

/// Typed errors for the band operations in [`crate::bands`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BandError {
    /// The raster carries a sample kind the band operations have no
    /// implementation for.
    ///
    /// The band ops read samples as unsigned integers, so they carry `U8`,
    /// `U16` and the 32-bit unsigned carrier of issue #517, and refuse
    /// float and the signed carriers of issue #516. It replaces a panic
    /// whose message said "float rasters" over a raster that was not
    /// float. Mirrors
    /// [`crate::mosaicing::MosaicError::UnsupportedSampleKind`].
    #[error("{op} does not support {kind:?} samples yet")]
    UnsupportedSampleKind {
        /// The operation that refused.
        op: &'static str,
        /// The sample kind it cannot read.
        kind: SampleKind,
    },
    /// Two rasters that must share dimensions do not.
    #[error("dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        got_w: u32,
        got_h: u32,
    },
    /// Two rasters that must share a band count do not (e.g. `bandrank`
    /// inputs).
    #[error("band-count mismatch: expected {expected} bands, got {got}")]
    BandCountMismatch { expected: usize, got: usize },
    /// A band index (after resolving negative indexing) is outside
    /// `0..bands`.
    #[error("band {band} out of range for {bands}-band image")]
    BandOutOfRange { band: i64, bands: usize },
    /// A band range `start..start+n` (after resolving negative indexing)
    /// does not fit in `0..bands`.
    #[error("band range {start}..{start}+{n} out of range for {bands}-band image")]
    BandRangeOutOfRange { start: i64, n: u32, bands: usize },
    /// `extract_bands` was asked for zero bands.
    #[error("cannot extract an empty band range")]
    EmptyBandRange,
    /// `bandjoin_vec` was given an empty constant vector.
    #[error("bandjoin_vec requires at least one constant")]
    EmptyConstants,
    /// A fold/unfold factor of zero was supplied.
    #[error("band fold factor must be greater than zero")]
    ZeroFactor,
    /// `bandfold` factor does not divide the image width.
    #[error("bandfold factor {factor} does not divide width {width}")]
    FoldNotDivisible { width: u32, factor: u32 },
    /// `bandunfold` factor does not divide the band count.
    #[error("bandunfold factor {factor} does not divide band count {bands}")]
    UnfoldNotDivisible { bands: usize, factor: u32 },
    /// The result would have more bands than [`PixelFormat`] can carry
    /// (`u16::MAX`).
    #[error("result band count {bands} exceeds the supported maximum of 65535")]
    TooManyBands { bands: usize },
    /// The result width would overflow `u32` (only reachable via
    /// `bandunfold` on an extremely wide input).
    #[error("result width {width} exceeds u32::MAX")]
    WidthOverflow { width: u64 },
    /// A `bandrank` index is outside the number of input images.
    #[error("bandrank index {index} out of range for {count} input images")]
    RankIndexOutOfRange { index: u32, count: usize },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Read the flat `i`-th sample of a buffer as a signed integer, honouring
/// the sample kind (native byte order, matching [`crate::raster_ops`]).
///
/// `i64` and total over [`SampleKind`], which is the whole of issue #909 on
/// this side: the `u32` this returned could not hold a negative, so the
/// three signed carriers of issue #516 had to be refused by
/// [`reject_unreadable_kind`] even though vips runs every band op on them.
/// Measured on `/opt/homebrew/bin/vips` 8.18.6, a `char` raster survives
/// `bandjoin`, `bandmean`, `bandrank` and the `bandbool` family and comes
/// back CHAR. The shape is [`crate::convolution`]'s `put_sample`, which
/// #748 made total first.
///
/// The signed kinds sign-extend, so this is the numeric read and not the
/// storage one. `F32` truncates toward zero, the `vips_cast` rounding, and
/// is reachable only from a direct call today because the band ops still
/// refuse a float raster for the reason they always have; `read_flat_reads_
/// every_kind_at_its_own_stride` drives it anyway, so the arm is held by a
/// test rather than by nothing.
#[inline]
fn read_flat(data: &[u8], kind: SampleKind, i: usize) -> i64 {
    match kind {
        SampleKind::U8 => i64::from(data[i]),
        SampleKind::I8 => i64::from(data[i] as i8),
        SampleKind::U16 => i64::from(u16::from_ne_bytes([data[2 * i], data[2 * i + 1]])),
        SampleKind::I16 => i64::from(i16::from_ne_bytes([data[2 * i], data[2 * i + 1]])),
        SampleKind::U32 => i64::from(u32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ])),
        SampleKind::I32 => i64::from(i32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ])),
        SampleKind::F32 => f32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ]) as i64,
    }
}

/// Store the flat `i`-th sample. `v` must already fit the kind.
///
/// A store and not a cast, the same contract [`crate::convolution`]'s
/// `put_sample` carries: every caller here either copies a sample it just
/// read at the same kind or has already divided a sum of such samples, so
/// the value is in range by construction and narrowing cannot clip.
/// Total over [`SampleKind`]; see [`read_flat`].
#[inline]
fn write_flat(data: &mut [u8], kind: SampleKind, i: usize, v: i64) {
    match kind {
        SampleKind::U8 => data[i] = v as u8,
        SampleKind::I8 => data[i] = v as i8 as u8,
        SampleKind::U16 => {
            let b = (v as u16).to_ne_bytes();
            data[2 * i..2 * i + 2].copy_from_slice(&b);
        }
        SampleKind::I16 => {
            let b = (v as i16).to_ne_bytes();
            data[2 * i..2 * i + 2].copy_from_slice(&b);
        }
        SampleKind::U32 => data[4 * i..4 * i + 4].copy_from_slice(&(v as u32).to_ne_bytes()),
        SampleKind::I32 => data[4 * i..4 * i + 4].copy_from_slice(&(v as i32).to_ne_bytes()),
        SampleKind::F32 => data[4 * i..4 * i + 4].copy_from_slice(&(v as f32).to_ne_bytes()),
    }
}

/// Clamp-and-round an `f64` constant into the sample range of an integer
/// kind.
///
/// The floor comes off [`SampleKind::range`] rather than being a literal
/// `0.0`, which is the third hazard class issue #909 names: a
/// `clamp(0.0, max)` is right for the three unsigned carriers and wrong for
/// all three signed ones. The visible case is `addalpha`, which inks
/// `interpretation_max_alpha` into the carrier: measured on
/// `/opt/homebrew/bin/vips` 8.18.6, a `b-w` tagged `char` raster gets alpha
/// **127** where `short` and `int` get 255, because 255 does not fit a
/// `char` and vips clips it.
///
/// `F32` has no range to clip into, so the constant is carried as it is and
/// [`write_flat`] narrows it; that is what `vips_cast` does on a float
/// target ("narrows without clipping"). No band op reaches it today,
/// because they all still refuse a float raster.
#[inline]
fn const_to_sample(c: f64, kind: SampleKind) -> i64 {
    match kind.range() {
        Some((lo, hi)) => c.clamp(lo as f64, hi as f64).round() as i64,
        None => c.round() as i64,
    }
}

/// Canonical output format for a band count and sample kind.
///
/// Through [`PixelFormat::with_kind`], because a byte width does not name
/// a carrier: `with_channels(bands, 4)` answers the float carrier, so
/// bandjoining two `uint` rasters used to tag the result float (issue
/// #517).
fn format_for(bands: usize, kind: SampleKind) -> Result<PixelFormat, BandError> {
    PixelFormat::with_kind(bands, kind).ok_or(BandError::TooManyBands { bands })
}

/// Refuse a sample kind the integer-only band ops cannot read, as a typed
/// error rather than a panic out of a `Result`-returning method (the shape
/// issue #694 landed).
///
/// Float is the only refusal left. The three signed carriers went through
/// here until issue #909 widened [`read_flat`] and [`write_flat`] to `i64`,
/// and that refusal was a parity regression rather than an implementation:
/// vips runs every one of these ops on a `char` raster and answers CHAR.
fn reject_unreadable_kind(op: &'static str, r: &Raster) -> Result<(), BandError> {
    let kind = r.format().kind();
    match kind {
        SampleKind::U8
        | SampleKind::U16
        | SampleKind::U32
        | SampleKind::I8
        | SampleKind::I16
        | SampleKind::I32 => Ok(()),
        SampleKind::F32 => Err(BandError::UnsupportedSampleKind { op, kind }),
    }
}

/// Error unless `a` and `b` share pixel dimensions.
fn ensure_same_dims(a: &Raster, b: &Raster) -> Result<(), BandError> {
    if (a.width(), a.height()) != (b.width(), b.height()) {
        return Err(BandError::DimensionMismatch {
            expected_w: a.width(),
            expected_h: a.height(),
            got_w: b.width(),
            got_h: b.height(),
        });
    }
    Ok(())
}

/// Unwrap a band-op result for the panicking ported-test surface.
#[inline]
#[track_caller]
fn expect_band(op: &str, r: Result<Raster, BandError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

impl Raster {
    /// Join the bands of `self` and `other` into one image
    /// (libvips `bandjoin`).
    ///
    /// The result has `self.bands + other.bands` bands: `self`'s bands come
    /// first, then `other`'s. Mixed 8/16-bit inputs promote numerically to
    /// 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::DimensionMismatch`] if the images differ in
    /// width or height, or [`BandError::TooManyBands`] if the combined band
    /// count exceeds `u16::MAX`.
    pub fn try_bandjoin(&self, other: &Raster) -> Result<Raster, BandError> {
        ensure_same_dims(self, other)?;
        reject_unreadable_kind("bandjoin", self)?;
        reject_unreadable_kind("bandjoin", other)?;
        let (a_fmt, b_fmt) = (self.format(), other.format());
        let (a_bands, b_bands) = (a_fmt.channels(), b_fmt.channels());
        let (a_kind, b_kind) = (a_fmt.kind(), b_fmt.kind());
        let out_bands = a_bands + b_bands;
        // `SampleKind::promote`, the libvips `vips__formatalike` order,
        // rather than the wider byte width, which cannot order the carriers
        // (issues #517, #607).
        let out_kind = a_kind.promote(b_kind);
        let out_bpc = out_kind.bytes();
        let out_fmt = format_for(out_bands, out_kind)?;

        let pixels = self.width() as usize * self.height() as usize;
        let mut out = vec![0u8; pixels * out_bands * out_bpc];
        let (a_data, b_data) = (self.data(), other.data());
        for p in 0..pixels {
            let base = p * out_bands;
            for c in 0..a_bands {
                let v = read_flat(a_data, a_kind, p * a_bands + c);
                write_flat(&mut out, out_kind, base + c, v);
            }
            for c in 0..b_bands {
                let v = read_flat(b_data, b_kind, p * b_bands + c);
                write_flat(&mut out, out_kind, base + a_bands + c, v);
            }
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        // Two-input, so the union rule (#718): the header block is `self`'s
        // alone and the fields are both inputs' with `self` winning a shared
        // name. Measured both ways round on vips 8.18.6, and reversing the
        // arguments flips the header block and the shared name while the ICC
        // profile still crosses from whichever input has one.
        result.merge_fields_from(other);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandjoin`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandjoin`].
    #[track_caller]
    pub fn bandjoin(&self, other: &Raster) -> Raster {
        expect_band("bandjoin", self.try_bandjoin(other))
    }

    /// Append one constant band (libvips `bandjoin_const` with a single
    /// element).
    ///
    /// The constant is clamped to the format range and rounded to nearest;
    /// the result keeps `self`'s depth and gains one band.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::TooManyBands`] if the resulting band count
    /// exceeds `u16::MAX`, or a wrapped [`RasterError`] if the result cannot
    /// be allocated.
    pub fn try_bandjoin_const(&self, c: f64) -> Result<Raster, BandError> {
        self.try_bandjoin_vec(&[c])
    }

    /// Panicking form of [`Raster::try_bandjoin_const`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandjoin_const`].
    #[track_caller]
    pub fn bandjoin_const(&self, c: f64) -> Raster {
        expect_band("bandjoin_const", self.try_bandjoin_const(c))
    }

    /// Append one constant band per element of `v` (libvips
    /// `bandjoin_const`).
    ///
    /// Constants are clamped to the format range and rounded to nearest; the
    /// result keeps `self`'s depth and gains `v.len()` bands.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::EmptyConstants`] if `v` is empty, or
    /// [`BandError::TooManyBands`] if the resulting band count exceeds
    /// `u16::MAX`.
    pub fn try_bandjoin_vec(&self, v: &[f64]) -> Result<Raster, BandError> {
        if v.is_empty() {
            return Err(BandError::EmptyConstants);
        }
        reject_unreadable_kind("bandjoin_const", self)?;
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bpc = kind.bytes();
        let out_bands = bands + v.len();
        let out_fmt = format_for(out_bands, kind)?;
        let consts: Vec<i64> = v.iter().map(|&c| const_to_sample(c, kind)).collect();

        let pixels = self.width() as usize * self.height() as usize;
        let mut out = vec![0u8; pixels * out_bands * bpc];
        let data = self.data();
        for p in 0..pixels {
            let base = p * out_bands;
            for c in 0..bands {
                write_flat(
                    &mut out,
                    kind,
                    base + c,
                    read_flat(data, kind, p * bands + c),
                );
            }
            for (k, &cv) in consts.iter().enumerate() {
                write_flat(&mut out, kind, base + bands + k, cv);
            }
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandjoin_vec`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandjoin_vec`].
    #[track_caller]
    pub fn bandjoin_vec(&self, v: &[f64]) -> Raster {
        expect_band("bandjoin_vec", self.try_bandjoin_vec(v))
    }

    /// Fold image width into bands (libvips `bandfold`).
    ///
    /// Groups of `factor` horizontally adjacent pixels become single pixels
    /// with `factor * bands` bands. `None` folds the whole width, producing
    /// a 1-pixel-wide image with `width * bands` bands. Because rows are
    /// stored interleaved and tightly packed, the pixel data is preserved
    /// byte-for-byte; only the shape changes.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::ZeroFactor`] for `Some(0)`,
    /// [`BandError::FoldNotDivisible`] if `factor` does not divide the
    /// width, or [`BandError::TooManyBands`] if the folded band count
    /// exceeds `u16::MAX`.
    pub fn try_bandfold(&self, factor: Option<u32>) -> Result<Raster, BandError> {
        let width = self.width();
        let factor = match factor {
            Some(0) => return Err(BandError::ZeroFactor),
            Some(f) => f,
            None => width,
        };
        if !width.is_multiple_of(factor) {
            return Err(BandError::FoldNotDivisible { width, factor });
        }
        let fmt = self.format();
        let out_bands = fmt.channels() * factor as usize;
        let out_fmt = format_for(out_bands, fmt.kind())?;
        let mut result = Raster::new(width / factor, self.height(), out_fmt, self.data().to_vec())?;
        // Reshapes the pixel grid and does *not* rescale the resolution:
        // measured on vips 8.18.6, an 8x8 3-band raster folds to 1x8 24-band
        // and still reports `xres 5 yres 7`. Same shape `zoom` and `subsample`
        // had in #690, so it is measured rather than assumed.
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandfold`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandfold`].
    #[track_caller]
    pub fn bandfold(&self, factor: Option<u32>) -> Raster {
        expect_band("bandfold", self.try_bandfold(factor))
    }

    /// Unfold bands into image width (libvips `bandunfold`); the inverse of
    /// [`Raster::bandfold`].
    ///
    /// Each pixel becomes `factor` horizontally adjacent pixels with
    /// `bands / factor` bands. `None` unfolds all bands, producing a 1-band
    /// image `width * bands` wide. Pixel data is preserved byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::ZeroFactor`] for `Some(0)`,
    /// [`BandError::UnfoldNotDivisible`] if `factor` does not divide the
    /// band count, or [`BandError::WidthOverflow`] if the unfolded width
    /// exceeds `u32::MAX`.
    pub fn try_bandunfold(&self, factor: Option<u32>) -> Result<Raster, BandError> {
        let fmt = self.format();
        let bands = fmt.channels();
        let factor = match factor {
            Some(0) => return Err(BandError::ZeroFactor),
            Some(f) => f,
            None => u32::try_from(bands).map_err(|_| BandError::TooManyBands { bands })?,
        };
        if !bands.is_multiple_of(factor as usize) {
            return Err(BandError::UnfoldNotDivisible { bands, factor });
        }
        let out_width = self.width() as u64 * factor as u64;
        let out_width =
            u32::try_from(out_width).map_err(|_| BandError::WidthOverflow { width: out_width })?;
        let out_fmt = format_for(bands / factor as usize, fmt.kind())?;
        let mut result = Raster::new(out_width, self.height(), out_fmt, self.data().to_vec())?;
        // Same as `bandfold`: 8x8 3-band unfolds to 24x8 1-band and reports
        // the same `xres 5 yres 7`.
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandunfold`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandunfold`].
    #[track_caller]
    pub fn bandunfold(&self, factor: Option<u32>) -> Raster {
        expect_band("bandunfold", self.try_bandunfold(factor))
    }

    /// Per-pixel mean across bands, producing a single-band image (libvips
    /// `bandmean`).
    ///
    /// The mean rounds **half away from zero**, which on the unsigned
    /// carriers is the round-half-up the libvips integer path
    /// (`bandmean.c`: `q[i] = (sum + bands / 2) / bands`) has always given.
    /// A negative mean is where the two part, and the answer is measured
    /// rather than derived: on `/opt/homebrew/bin/vips` 8.18.6 a two-band
    /// `char` raster gives `(-100, -101) -> -101`, `(-100, -99) -> -100`
    /// and `(-1, -2) -> -2`, every one of them further from zero, against
    /// `(100, 101) -> 101` on the unsigned side. Confirmed at three and
    /// four bands, so the `bands / 2` term is the one that moves.
    ///
    /// It disagrees with [`Raster::try_shrink`], which truncates toward
    /// zero on the same numbers. Two ops, two roundings, both matched.
    ///
    /// The result keeps `self`'s depth.
    ///
    /// # Errors
    ///
    /// Returns a wrapped [`RasterError`] if the result cannot be allocated.
    pub fn try_bandmean(&self) -> Result<Raster, BandError> {
        reject_unreadable_kind("bandmean", self)?;
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bpc = kind.bytes();
        let out_fmt = format_for(1, kind)?;
        let pixels = self.width() as usize * self.height() as usize;
        let mut out = vec![0u8; pixels * bpc];
        let data = self.data();
        for p in 0..pixels {
            let sum: i64 = (0..bands)
                .map(|c| read_flat(data, kind, p * bands + c))
                .sum();
            let bands64 = bands as i64;
            // Half away from zero. Rust's `/` truncates toward zero like
            // C's, so the rounding term has to follow the sign of the sum
            // or a negative mean lands one step nearer zero than vips puts
            // it. `(-201, 2)` is `-101` and not `-100`.
            let half = bands64 / 2;
            let rounded = if sum >= 0 {
                (sum + half) / bands64
            } else {
                (sum - half) / bands64
            };
            write_flat(&mut out, kind, p, rounded);
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandmean`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandmean`].
    #[track_caller]
    pub fn bandmean(&self) -> Raster {
        expect_band("bandmean", self.try_bandmean())
    }

    /// Per-sample rank statistic across `self` and `others` (libvips
    /// `bandrank`).
    ///
    /// For every pixel and band, the sample values from all
    /// `others.len() + 1` images are sorted ascending and the value at
    /// `index` is selected. `None` picks the median position `count / 2`
    /// (the upper median for an even count), matching libvips's default
    /// `index = -1`. Mixed 8/16-bit inputs promote numerically to 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::RankIndexOutOfRange`] if `index` is not below
    /// the image count, [`BandError::DimensionMismatch`] or
    /// [`BandError::BandCountMismatch`] if any input disagrees with `self`.
    pub fn try_bandrank(
        &self,
        others: &[&Raster],
        index: Option<u32>,
    ) -> Result<Raster, BandError> {
        let count = others.len() + 1;
        let idx = match index {
            Some(i) => {
                if (i as usize) >= count {
                    return Err(BandError::RankIndexOutOfRange { index: i, count });
                }
                i as usize
            }
            None => count / 2,
        };
        let bands = self.format().channels();
        reject_unreadable_kind("bandrank", self)?;
        // Promoted over the whole input set, so the answer does not depend
        // on the order they were listed in (issues #517, #607).
        let mut out_kind = self.format().kind();
        for r in others {
            ensure_same_dims(self, r)?;
            reject_unreadable_kind("bandrank", r)?;
            if r.format().channels() != bands {
                return Err(BandError::BandCountMismatch {
                    expected: bands,
                    got: r.format().channels(),
                });
            }
            out_kind = out_kind.promote(r.format().kind());
        }
        let out_bpc = out_kind.bytes();
        let out_fmt = format_for(bands, out_kind)?;

        let samples = self.width() as usize * self.height() as usize * bands;
        let mut out = vec![0u8; samples * out_bpc];
        let mut vals = vec![0i64; count];
        for i in 0..samples {
            vals[0] = read_flat(self.data(), self.format().kind(), i);
            for (k, r) in others.iter().enumerate() {
                vals[k + 1] = read_flat(r.data(), r.format().kind(), i);
            }
            vals.sort_unstable();
            write_flat(&mut out, out_kind, i, vals[idx]);
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        // n-input, same union as `bandjoin`: measured on vips 8.18.6 over three
        // sources, the output carries each one's own fields and `self` wins the
        // name all three share.
        for r in others {
            result.merge_fields_from(r);
        }
        Ok(result)
    }

    /// Panicking form of [`Raster::try_bandrank`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandrank`].
    #[track_caller]
    pub fn bandrank(&self, others: &[&Raster], index: Option<u32>) -> Raster {
        expect_band("bandrank", self.try_bandrank(others, index))
    }

    /// Bitwise AND across all bands, producing a single-band image (libvips
    /// `bandbool` with AND).
    ///
    /// # Errors
    ///
    /// Returns a wrapped [`RasterError`] if the result cannot be allocated.
    pub fn try_bandand(&self) -> Result<Raster, BandError> {
        self.bandbool(|acc, v| acc & v)
    }

    /// Panicking form of [`Raster::try_bandand`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandand`].
    #[track_caller]
    pub fn bandand(&self) -> Raster {
        expect_band("bandand", self.try_bandand())
    }

    /// Bitwise OR across all bands, producing a single-band image (libvips
    /// `bandbool` with OR).
    ///
    /// # Errors
    ///
    /// Returns a wrapped [`RasterError`] if the result cannot be allocated.
    pub fn try_bandor(&self) -> Result<Raster, BandError> {
        self.bandbool(|acc, v| acc | v)
    }

    /// Panicking form of [`Raster::try_bandor`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandor`].
    #[track_caller]
    pub fn bandor(&self) -> Raster {
        expect_band("bandor", self.try_bandor())
    }

    /// Bitwise XOR across all bands, producing a single-band image (libvips
    /// `bandbool` with EOR).
    ///
    /// # Errors
    ///
    /// Returns a wrapped [`RasterError`] if the result cannot be allocated.
    pub fn try_bandeor(&self) -> Result<Raster, BandError> {
        self.bandbool(|acc, v| acc ^ v)
    }

    /// Panicking form of [`Raster::try_bandeor`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_bandeor`].
    #[track_caller]
    pub fn bandeor(&self) -> Raster {
        expect_band("bandeor", self.try_bandeor())
    }

    /// Shared reduction for the `bandbool` family: fold every pixel's bands
    /// with `op`, keeping the input depth. A single-band input reduces to a
    /// copy of itself, matching libvips.
    fn bandbool(&self, op: impl Fn(i64, i64) -> i64) -> Result<Raster, BandError> {
        reject_unreadable_kind("bandbool", self)?;
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bpc = kind.bytes();
        let out_fmt = format_for(1, kind)?;
        let pixels = self.width() as usize * self.height() as usize;
        let mut out = vec![0u8; pixels * bpc];
        let data = self.data();
        for p in 0..pixels {
            let mut acc = read_flat(data, kind, p * bands);
            for c in 1..bands {
                acc = op(acc, read_flat(data, kind, p * bands + c));
            }
            write_flat(&mut out, kind, p, acc);
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Extract a single band as a one-band image (libvips `extract_band`
    /// with `n = 1`).
    ///
    /// A negative index addresses bands from the end, so `-1` is the last
    /// band (the ported tests fold pyvips's `im[-1]` sugar into this call).
    ///
    /// # Errors
    ///
    /// Returns [`BandError::BandOutOfRange`] if the resolved index is
    /// outside `0..bands`.
    pub fn try_extract_band(&self, band: i64) -> Result<Raster, BandError> {
        let bands = self.format().channels();
        let resolved = if band < 0 { band + bands as i64 } else { band };
        if resolved < 0 || resolved >= bands as i64 {
            return Err(BandError::BandOutOfRange { band, bands });
        }
        self.try_extract_bands(resolved, 1)
    }

    /// Panicking form of [`Raster::try_extract_band`], matching the
    /// ported-test surface. Accepts any integer index type (`i32`, `u32`,
    /// ...); negative values index from the end.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_extract_band`].
    #[track_caller]
    pub fn extract_band<B: Into<i64>>(&self, band: B) -> Raster {
        expect_band("extract_band", self.try_extract_band(band.into()))
    }

    /// Extract `n` consecutive bands starting at `start` (libvips
    /// `extract_band`).
    ///
    /// A negative `start` addresses bands from the end. The result keeps
    /// `self`'s depth and has `n` bands.
    ///
    /// # Errors
    ///
    /// Returns [`BandError::EmptyBandRange`] if `n` is zero, or
    /// [`BandError::BandRangeOutOfRange`] if the resolved range does not fit
    /// in `0..bands`.
    pub fn try_extract_bands(&self, start: i64, n: u32) -> Result<Raster, BandError> {
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bpc = kind.bytes();
        if n == 0 {
            return Err(BandError::EmptyBandRange);
        }
        let resolved = if start < 0 {
            start + bands as i64
        } else {
            start
        };
        if resolved < 0 || resolved + n as i64 > bands as i64 {
            return Err(BandError::BandRangeOutOfRange { start, n, bands });
        }
        let first = resolved as usize;
        let n = n as usize;
        // A pure byte copy, so every carrier passes through: the format
        // has to come off the kind or a `Uint32` band extract would be
        // tagged float (issue #517).
        let out_fmt = format_for(n, kind)?;

        let pixels = self.width() as usize * self.height() as usize;
        let mut out = vec![0u8; pixels * n * bpc];
        let data = self.data();
        for p in 0..pixels {
            let src = (p * bands + first) * bpc;
            let dst = p * n * bpc;
            out[dst..dst + n * bpc].copy_from_slice(&data[src..src + n * bpc]);
        }
        let mut result = Raster::new(self.width(), self.height(), out_fmt, out)?;
        result.carry_meta_from(self);
        Ok(result)
    }

    /// Panicking form of [`Raster::try_extract_bands`], matching the
    /// ported-test surface. Accepts any integer start type (`i32`, `u32`,
    /// ...); negative values index from the end.
    ///
    /// # Panics
    ///
    /// Panics on any [`BandError`]; see [`Raster::try_extract_bands`].
    #[track_caller]
    pub fn extract_bands<B: Into<i64>>(&self, start: B, n: u32) -> Raster {
        expect_band("extract_bands", self.try_extract_bands(start.into(), n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::ALL_KINDS;

    /// A one-band `Int8` raster from signed sample values.
    fn int8(w: u32, h: u32, bands: u16, vals: &[i8]) -> Raster {
        let data: Vec<u8> = vals.iter().map(|v| *v as u8).collect();
        let fmt = PixelFormat::Int8(core::num::NonZeroU16::new(bands).unwrap());
        Raster::new(w, h, fmt, data).unwrap()
    }

    /// Every sample of an `Int8` raster, read back signed.
    fn i8s(r: &Raster) -> Vec<i8> {
        r.data().iter().map(|b| *b as i8).collect()
    }

    /**
     * Tests that the band ops' sample reader and writer round-trip every
     * sample kind at its own stride and its own signedness, including the
     * `F32` arm no band op can reach because they all still refuse a float
     * raster.
     * Works by sweeping [`ALL_KINDS`] rather than a hand-written list, and
     * by writing each kind's range endpoints and one interior value into a
     * two-sample buffer at index 1, so a helper that used the wrong stride
     * would write over index 0 and be caught by the untouched-neighbour
     * assertion rather than only by the value.
     * Input: each kind's `range()` endpoints and 0 -> Output: the same
     * numbers back, with index 0 still zero.
     */
    #[test]
    fn read_flat_and_write_flat_round_trip_every_kind_at_its_own_stride() {
        for kind in ALL_KINDS {
            let bytes = kind.bytes();
            // The values a kind can hold. `F32` has no integer range, so it
            // is swept over the same three numbers the narrowest kind uses,
            // which it represents exactly.
            let cases: [i64; 3] = match kind.range() {
                Some((lo, hi)) => [lo, 0, hi],
                None => [-128, 0, 127],
            };
            for v in cases {
                let mut buf = vec![0u8; bytes * 2];
                write_flat(&mut buf, kind, 1, v);
                assert_eq!(
                    read_flat(&buf, kind, 1),
                    v,
                    "{kind:?} did not round-trip {v}"
                );
                assert!(
                    buf[..bytes].iter().all(|&b| b == 0),
                    "{kind:?} wrote outside sample 1, so its stride is wrong"
                );
            }
        }
        // The collisions a width-keyed helper would exploit, stated
        // directly: the two one-byte kinds, the two two-byte ones, and the
        // three-way tie at four bytes. `-1` is the value that separates
        // them, because it is the same bit pattern in both members of each
        // pair and a different number.
        let mut b8 = vec![0u8; 1];
        write_flat(&mut b8, SampleKind::I8, 0, -1);
        assert_eq!(b8[0], 0xFF);
        assert_eq!(read_flat(&b8, SampleKind::U8, 0), 255);
        assert_eq!(read_flat(&b8, SampleKind::I8, 0), -1);
        let mut b32 = vec![0u8; 4];
        write_flat(&mut b32, SampleKind::I32, 0, -1);
        assert_eq!(read_flat(&b32, SampleKind::U32, 0), 4_294_967_295);
        assert_eq!(read_flat(&b32, SampleKind::I32, 0), -1);
        assert_eq!(
            read_flat(&b32, SampleKind::F32, 0),
            f32::from_ne_bytes([0xFF; 4]) as i64,
            "the four-byte tie has three members, not two"
        );
    }

    /**
     * Tests that a constant is clipped into the carrier's own range at
     * **both** ends rather than into `0..=max`, which is what makes
     * `addalpha` ink the right sample on a signed carrier.
     * Works by pushing 255 and -300 at every integer kind and reading the
     * clip, with `Gray8` as the control that the ceiling still applies
     * where it always did.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: `vips addalpha` on a
     * `b-w` tagged `char` raster inks **127**, and on `short` and `int`
     * 255, because the interpretation's max alpha is 255 for all three and
     * only `char` cannot hold it.
     * Input: 255 into I8 -> 127; -300 into I8 -> -128; 255 into I16 -> 255.
     */
    #[test]
    fn const_to_sample_clips_into_the_carrier_range_at_both_ends() {
        for kind in ALL_KINDS {
            let Some((lo, hi)) = kind.range() else {
                continue;
            };
            assert_eq!(const_to_sample(255.0, kind), 255i64.min(hi), "{kind:?}");
            assert_eq!(const_to_sample(-300.0, kind), (-300i64).max(lo), "{kind:?}");
        }
        // The two cells the measurement names, spelled out so the loop
        // above cannot pass by computing the same wrong thing twice.
        assert_eq!(const_to_sample(255.0, SampleKind::I8), 127);
        assert_eq!(const_to_sample(255.0, SampleKind::I16), 255);
        assert_eq!(const_to_sample(255.0, SampleKind::U8), 255);
        // And the floor really is per kind: an unsigned carrier still
        // clips a negative constant to zero.
        assert_eq!(const_to_sample(-300.0, SampleKind::U8), 0);
        assert_eq!(const_to_sample(-300.0, SampleKind::I8), -128);
    }

    /**
     * Tests that the band ops carry the three signed carriers with the
     * values vips produces, replacing the interim refusal issue #909 was
     * filed for.
     * The refusal it retires was correct as an interim and wrong as an
     * implementation: measured on `/opt/homebrew/bin/vips` 8.18.6, a
     * `char` raster survives `bandjoin`, `bandjoin_const`, `bandmean`,
     * `bandrank` and the whole `bandbool` family and comes back CHAR.
     * Works by driving each op on `Int8` samples and asserting the sample
     * values, with `Uint32` alongside as the control that the guard still
     * discriminates by kind, and a float raster as the control that the
     * one remaining refusal is intact.
     * Input: `Int8` `[-100, -1, 0, 100]` -> Output: the measured samples.
     */
    #[test]
    fn the_band_ops_carry_the_signed_carriers() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let one = int8(2, 2, 1, &[-100, -1, 0, 100]);

        // bandjoin: samples copied, band count doubled, carrier kept.
        let joined = one.try_bandjoin(&one).unwrap();
        assert_eq!(joined.format(), PixelFormat::Int8(n(2)));
        assert_eq!(i8s(&joined), vec![-100, -100, -1, -1, 0, 0, 100, 100]);

        // bandjoin_const, which is what `addalpha` is: the ink clips into
        // the carrier, so 255 becomes 127 on `char`.
        let inked = one.try_bandjoin_const(255.0).unwrap();
        assert_eq!(i8s(&inked), vec![-100, 127, -1, 127, 0, 127, 100, 127]);
        let alpha = one.try_addalpha().unwrap();
        assert_eq!(alpha.format(), PixelFormat::Int8(n(2)));
        assert_eq!(i8s(&alpha), vec![-100, 127, -1, 127, 0, 127, 100, 127]);

        // bandrank sorts by **value**, not by bit pattern: read as
        // unsigned bytes, -101 (0x9B) would sort above -100 (0x9C) is
        // false, but 100 (0x64) would sort below both, which it must not.
        let two = int8(1, 2, 1, &[-100, 100]);
        let other = int8(1, 2, 1, &[-101, 101]);
        let lo = two.try_bandrank(&[&other], Some(0)).unwrap();
        let hi = two.try_bandrank(&[&other], Some(1)).unwrap();
        assert_eq!(i8s(&lo), vec![-101, 100]);
        assert_eq!(i8s(&hi), vec![-100, 101]);

        // bandbool works on the two's complement bit patterns and keeps
        // the carrier. Measured on 8.18.6 over the four pairs below:
        // and -> [-104, -102, -104, 100], or -> [-97, -101, -101, 101],
        // eor -> [7, 1, 3, 1]. The `or` row is worth its place: the second
        // and third pairs answer the *same* number from different inputs,
        // which a table written from expectation gets wrong (this one was,
        // and the test caught it).
        let pairs = int8(1, 4, 2, &[-100, -101, -101, -102, -102, -103, 100, 101]);
        assert_eq!(
            i8s(&pairs.try_bandand().unwrap()),
            vec![-104, -102, -104, 100]
        );
        assert_eq!(
            i8s(&pairs.try_bandor().unwrap()),
            vec![-97, -101, -101, 101]
        );
        assert_eq!(i8s(&pairs.try_bandeor().unwrap()), vec![7, 1, 3, 1]);

        // Control: the unsigned 32-bit carrier still goes through, so
        // these are not a blanket "accept anything".
        let u = Raster::new(
            1,
            1,
            PixelFormat::Uint32(n(1)),
            90_000u32.to_ne_bytes().to_vec(),
        )
        .unwrap();
        assert!(u.try_bandmean().is_ok());
        // Control: the one refusal left is float, and it is still there.
        let f = Raster::new(
            1,
            1,
            PixelFormat::FloatF32(n(1)),
            1.5f32.to_ne_bytes().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            f.try_bandmean(),
            Err(BandError::UnsupportedSampleKind {
                op: "bandmean",
                kind: SampleKind::F32
            })
        ));
    }

    /**
     * Tests that `bandmean` rounds **half away from zero**, which is the
     * round-half-up it has always done on the unsigned carriers and a
     * different answer on a negative mean.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6 over a two-band `char`
     * raster: `(-100, -101) -> -101`, `(-100, -99) -> -100`,
     * `(-99, -98) -> -99`, `(-1, -2) -> -2`, against `(0, 1) -> 1`,
     * `(1, 2) -> 2`, `(99, 100) -> 100` and `(100, 101) -> 101`. Every
     * negative answer is **further** from zero than truncation gives, so
     * `(sum + n/2) / n` with C's truncating divide answers one step nearer
     * zero on four of those eight rows. Confirmed at three and four bands,
     * which is what pins the `n / 2` term rather than a hard-coded half.
     * Works by asserting all eight two-band pairs plus the three- and
     * four-band rows, and by including `(-100, -100)`, an exact mean,
     * where every candidate rule agrees, as the row that stops the test
     * passing because the fixture is degenerate.
     * Input: the pairs above -> Output: the measured means.
     */
    #[test]
    fn bandmean_rounds_half_away_from_zero() {
        // (band 0, band 1, the mean vips answers)
        let pairs: [(i8, i8, i8); 9] = [
            (-100, -100, -100),
            (-100, -101, -101),
            (-100, -99, -100),
            (-99, -98, -99),
            (-1, -2, -2),
            (0, 1, 1),
            (1, 2, 2),
            (99, 100, 100),
            (100, 101, 101),
        ];
        let data: Vec<i8> = pairs.iter().flat_map(|&(a, b, _)| [a, b]).collect();
        let want: Vec<i8> = pairs.iter().map(|&(_, _, m)| m).collect();
        let r = int8(1, pairs.len() as u32, 2, &data);
        assert_eq!(i8s(&r.try_bandmean().unwrap()), want);

        // Three bands: sums -4, -5, -3 answer -1, -2, -1, and 4, 5, 2
        // answer 1, 2, 1. Measured the same way.
        let three = int8(
            1,
            6,
            3,
            &[
                -2, -1, -1, -2, -2, -1, -1, -1, -1, 2, 1, 1, 2, 2, 1, 1, 1, 0,
            ],
        );
        assert_eq!(
            i8s(&three.try_bandmean().unwrap()),
            vec![-1, -2, -1, 1, 2, 1]
        );

        // Four bands: sums -2, -6, 2, 6, -10, 10 answer -1, -2, 1, 2, -3, 3.
        let four = int8(
            1,
            6,
            4,
            &[
                -1, -1, 0, 0, -2, -2, -1, -1, 1, 1, 0, 0, 2, 2, 1, 1, -3, -3, -2, -2, 3, 3, 2, 2,
            ],
        );
        assert_eq!(
            i8s(&four.try_bandmean().unwrap()),
            vec![-1, -2, 1, 2, -3, 3]
        );
    }

    /// A 2x1 Rgb8 raster with distinct, bit-pattern-friendly samples.
    fn rgb2x1() -> Raster {
        Raster::new(2, 1, PixelFormat::Rgb8, vec![0xF0, 0x0F, 0xFF, 10, 20, 30]).unwrap()
    }

    /// A 2x1 Gray8 raster.
    fn gray2x1() -> Raster {
        Raster::new(2, 1, PixelFormat::Gray8, vec![7, 9]).unwrap()
    }

    /// A 1x1 Gray16 raster holding `v`.
    fn gray16_1x1(v: u16) -> Raster {
        let b = v.to_ne_bytes();
        Raster::new(1, 1, PixelFormat::Gray16, vec![b[0], b[1]]).unwrap()
    }

    // ---- bandjoin ----

    /// bandjoin appends the other image's bands after self's, per pixel.
    /// Rgb8 + Gray8 canonicalizes to Rgba8 (4 bands, 8-bit).
    #[test]
    fn bandjoin_rgb_plus_gray_is_rgba() {
        let joined = rgb2x1().bandjoin(&gray2x1());
        assert_eq!(joined.format(), PixelFormat::Rgba8);
        assert_eq!(joined.getpoint(0, 0), vec![240.0, 15.0, 255.0, 7.0]);
        assert_eq!(joined.getpoint(1, 0), vec![10.0, 20.0, 30.0, 9.0]);
    }

    /// Gray8 + Gray8 has 2 bands, which no named format carries, so the
    /// result is the canonical Multi8(2).
    #[test]
    fn bandjoin_gray_plus_gray_is_two_band_multi() {
        let joined = gray2x1().bandjoin(&gray2x1());
        assert_eq!(joined.format().channels(), 2);
        assert_eq!(joined.format().bytes_per_channel(), 1);
        assert_eq!(joined.getpoint(0, 0), vec![7.0, 7.0]);
    }

    /// Mixed 8/16-bit inputs promote numerically to 16-bit: an 8-bit 240
    /// stays 240 in the wider result.
    #[test]
    fn bandjoin_mixed_depth_promotes_to_16bit() {
        let a = Raster::new(1, 1, PixelFormat::Gray8, vec![240]).unwrap();
        let b = gray16_1x1(4096);
        let joined = a.bandjoin(&b);
        assert_eq!(joined.format().bytes_per_channel(), 2);
        assert_eq!(joined.format().channels(), 2);
        assert_eq!(joined.getpoint(0, 0), vec![240.0, 4096.0]);
    }

    /// bandjoin rejects images with different pixel dimensions.
    #[test]
    fn bandjoin_dimension_mismatch_is_typed_error() {
        let a = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let b = Raster::zeroed(3, 2, PixelFormat::Gray8).unwrap();
        assert!(matches!(
            a.try_bandjoin(&b),
            Err(BandError::DimensionMismatch { .. })
        ));
    }

    /// The panicking surface reports the op name and the typed message.
    #[test]
    #[should_panic(expected = "bandjoin: dimension mismatch")]
    fn bandjoin_panicking_surface_message() {
        let a = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let b = Raster::zeroed(3, 2, PixelFormat::Gray8).unwrap();
        let _ = a.bandjoin(&b);
    }

    // ---- bandjoin_const / bandjoin_vec ----

    /// bandjoin_const appends one constant band and keeps depth: 3 -> 4
    /// bands with every new sample equal to the constant.
    #[test]
    fn bandjoin_const_appends_constant_band() {
        let joined = rgb2x1().bandjoin_const(1.0);
        assert_eq!(joined.format().channels(), 4);
        assert_eq!(joined.getpoint(0, 0)[3], 1.0);
        assert_eq!(joined.getpoint(1, 0)[3], 1.0);
    }

    /// Constants clamp to the 8-bit range (negative -> 0, above 255 -> 255)
    /// and round to nearest.
    #[test]
    fn bandjoin_const_clamps_and_rounds_8bit() {
        let g = gray2x1();
        assert_eq!(g.bandjoin_const(-5.0).getpoint(0, 0)[1], 0.0);
        assert_eq!(g.bandjoin_const(300.0).getpoint(0, 0)[1], 255.0);
        assert_eq!(g.bandjoin_const(1.6).getpoint(0, 0)[1], 2.0);
        assert_eq!(g.bandjoin_const(1.4).getpoint(0, 0)[1], 1.0);
    }

    /// Constants clamp to the 16-bit range on 16-bit inputs.
    #[test]
    fn bandjoin_const_clamps_16bit() {
        let g = gray16_1x1(1);
        let joined = g.bandjoin_const(70000.0);
        assert_eq!(joined.format().bytes_per_channel(), 2);
        assert_eq!(joined.getpoint(0, 0), vec![1.0, 65535.0]);
    }

    /// bandjoin_vec appends one band per constant: 3 + 2 = 5 bands, with
    /// band 3 = 1 and band 4 = 2 everywhere (the ported-test scenario).
    #[test]
    fn bandjoin_vec_appends_multiple_bands() {
        let joined = rgb2x1().bandjoin_vec(&[1.0, 2.0]);
        assert_eq!(joined.format().channels(), 5);
        for x in 0..2 {
            let px = joined.getpoint(x, 0);
            assert_eq!(px[3], 1.0);
            assert_eq!(px[4], 2.0);
        }
    }

    /// An empty constant vector is a typed error, not a silent no-op.
    #[test]
    fn bandjoin_vec_empty_is_typed_error() {
        assert!(matches!(
            gray2x1().try_bandjoin_vec(&[]),
            Err(BandError::EmptyConstants)
        ));
    }

    // ---- bandfold / bandunfold ----

    /// bandfold(None) folds the whole width: 4x1 Gray8 becomes 1x1 with 4
    /// bands, and the sample order is preserved byte-for-byte.
    #[test]
    fn bandfold_none_folds_whole_width() {
        let im = Raster::new(4, 1, PixelFormat::Gray8, vec![1, 2, 3, 4]).unwrap();
        let folded = im.bandfold(None);
        assert_eq!(folded.width(), 1);
        assert_eq!(folded.height(), 1);
        assert_eq!(folded.format().channels(), 4);
        assert_eq!(folded.getpoint(0, 0), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// bandfold(Some(2)) halves the width and doubles the bands, grouping
    /// horizontally adjacent pixels.
    #[test]
    fn bandfold_factor_two() {
        let im = Raster::new(4, 1, PixelFormat::Gray8, vec![1, 2, 3, 4]).unwrap();
        let folded = im.bandfold(Some(2));
        assert_eq!(folded.width(), 2);
        assert_eq!(folded.format().channels(), 2);
        assert_eq!(folded.getpoint(0, 0), vec![1.0, 2.0]);
        assert_eq!(folded.getpoint(1, 0), vec![3.0, 4.0]);
    }

    /// bandfold on a multi-band input multiplies the existing band count.
    #[test]
    fn bandfold_multiband_input() {
        let im = rgb2x1();
        let folded = im.bandfold(Some(2));
        assert_eq!(folded.width(), 1);
        assert_eq!(folded.format().channels(), 6);
        assert_eq!(
            folded.getpoint(0, 0),
            vec![240.0, 15.0, 255.0, 10.0, 20.0, 30.0]
        );
    }

    /// A factor that does not divide the width is a typed error, as is a
    /// zero factor.
    #[test]
    fn bandfold_bad_factor_is_typed_error() {
        let im = Raster::new(4, 1, PixelFormat::Gray8, vec![1, 2, 3, 4]).unwrap();
        assert!(matches!(
            im.try_bandfold(Some(3)),
            Err(BandError::FoldNotDivisible {
                width: 4,
                factor: 3
            })
        ));
        assert!(matches!(
            im.try_bandfold(Some(0)),
            Err(BandError::ZeroFactor)
        ));
    }

    /// bandunfold(None) is the inverse of bandfold(None): the round trip
    /// restores shape, format, and every byte.
    #[test]
    fn bandfold_bandunfold_round_trip() {
        let im = Raster::new(6, 2, PixelFormat::Gray8, (0u8..12).collect::<Vec<_>>()).unwrap();
        let round = im.bandfold(None).bandunfold(None);
        assert_eq!(round.width(), im.width());
        assert_eq!(round.height(), im.height());
        assert_eq!(round.format(), im.format());
        assert_eq!(round.data(), im.data());
    }

    /// bandunfold with an explicit factor splits bands into width.
    #[test]
    fn bandunfold_factor_two() {
        let im = Raster::new(1, 1, PixelFormat::Rgba8, vec![1, 2, 3, 4]).unwrap();
        let unfolded = im.bandunfold(Some(2));
        assert_eq!(unfolded.width(), 2);
        assert_eq!(unfolded.format().channels(), 2);
        assert_eq!(unfolded.getpoint(0, 0), vec![1.0, 2.0]);
        assert_eq!(unfolded.getpoint(1, 0), vec![3.0, 4.0]);
    }

    /// A factor that does not divide the band count is a typed error.
    #[test]
    fn bandunfold_bad_factor_is_typed_error() {
        let im = rgb2x1();
        assert!(matches!(
            im.try_bandunfold(Some(2)),
            Err(BandError::UnfoldNotDivisible {
                bands: 3,
                factor: 2
            })
        ));
    }

    /// 16-bit rasters fold and unfold with samples intact.
    #[test]
    fn bandfold_16bit_round_trip_values() {
        let a = 4096u16.to_ne_bytes();
        let b = 300u16.to_ne_bytes();
        let im = Raster::new(2, 1, PixelFormat::Gray16, vec![a[0], a[1], b[0], b[1]]).unwrap();
        let folded = im.bandfold(None);
        assert_eq!(folded.format().bytes_per_channel(), 2);
        assert_eq!(folded.getpoint(0, 0), vec![4096.0, 300.0]);
        assert_eq!(folded.bandunfold(None).getpoint(1, 0), vec![300.0]);
    }

    // ---- bandmean ----

    /// bandmean rounds the per-pixel mean to nearest (halves up), matching
    /// the libvips integer path `(sum + bands / 2) / bands`. Verified against
    /// vips 8.18.4 `vips_bandmean`: sum 5 / 3 bands -> 2 (not floor 1),
    /// sum 61 / 3 -> 20, sum 6 / 3 -> 2.
    #[test]
    fn bandmean_rounds_to_nearest() {
        // Pixel 0: [1,2,2] sum 5 -> round(5/3) = 2 (truncation would give 1).
        // Pixel 1: [10,20,31] sum 61 -> round(61/3) = 20.
        let im = Raster::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 2, 10, 20, 31]).unwrap();
        let mean = im.bandmean();
        assert_eq!(mean.format(), PixelFormat::Gray8);
        assert_eq!(mean.getpoint(0, 0), vec![2.0]);
        assert_eq!(mean.getpoint(1, 0), vec![20.0]);
    }

    /// bandmean rounds exact halves up like vips on an even band count.
    /// Verified against vips 8.18.4 `vips_bandmean`: 4-band [1,1,0,0]
    /// sum 2 -> (2 + 2) / 4 = 1 (truncation would give 0).
    #[test]
    fn bandmean_rounds_halves_up() {
        let im = Raster::new(1, 1, PixelFormat::Rgba8, vec![1, 1, 0, 0]).unwrap();
        let mean = im.bandmean();
        assert_eq!(mean.format(), PixelFormat::Gray8);
        assert_eq!(mean.getpoint(0, 0), vec![1.0]);
    }

    /// bandmean keeps 16-bit depth and handles sums beyond u16 range.
    #[test]
    fn bandmean_16bit() {
        let hi = 60000u16.to_ne_bytes();
        let lo = 30000u16.to_ne_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&hi);
        data.extend_from_slice(&hi);
        data.extend_from_slice(&lo);
        let im = Raster::new(1, 1, PixelFormat::Rgb16, data).unwrap();
        let mean = im.bandmean();
        assert_eq!(mean.format(), PixelFormat::Gray16);
        assert_eq!(mean.getpoint(0, 0), vec![50000.0]);
    }

    /// bandmean of a single-band image is the image itself.
    #[test]
    fn bandmean_single_band_is_identity() {
        let g = gray2x1();
        let mean = g.bandmean();
        assert_eq!(mean.format(), PixelFormat::Gray8);
        assert_eq!(mean.data(), g.data());
    }

    /// bandmean works on multiband intermediates produced by bandjoin_vec.
    #[test]
    fn bandmean_on_multiband_intermediate() {
        let im = gray2x1().bandjoin_vec(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(im.format().channels(), 5);
        // Pixel 0: (7+1+2+3+4)/5 = 3.4 -> 3.
        assert_eq!(im.bandmean().getpoint(0, 0), vec![3.0]);
    }

    // ---- bandrank ----

    /// bandrank with two identical inputs returns the input (any rank of
    /// identical values is that value) — the ported-test scenario.
    #[test]
    fn bandrank_identical_images_is_identity() {
        let im = rgb2x1();
        let ranked = im.bandrank(&[&im], None);
        assert_eq!(ranked.format(), im.format());
        assert_eq!(ranked.data(), im.data());
    }

    /// bandrank(None) across three images picks the per-sample median.
    #[test]
    fn bandrank_median_of_three() {
        let a = Raster::new(1, 1, PixelFormat::Gray8, vec![10]).unwrap();
        let b = Raster::new(1, 1, PixelFormat::Gray8, vec![200]).unwrap();
        let c = Raster::new(1, 1, PixelFormat::Gray8, vec![90]).unwrap();
        let ranked = a.bandrank(&[&b, &c], None);
        assert_eq!(ranked.getpoint(0, 0), vec![90.0]);
    }

    /// Explicit indices select the minimum (0) and maximum (count-1).
    #[test]
    fn bandrank_explicit_index_min_max() {
        let a = Raster::new(1, 1, PixelFormat::Gray8, vec![10]).unwrap();
        let b = Raster::new(1, 1, PixelFormat::Gray8, vec![200]).unwrap();
        let c = Raster::new(1, 1, PixelFormat::Gray8, vec![90]).unwrap();
        assert_eq!(a.bandrank(&[&b, &c], Some(0)).getpoint(0, 0), vec![10.0]);
        assert_eq!(a.bandrank(&[&b, &c], Some(2)).getpoint(0, 0), vec![200.0]);
    }

    /// The median is selected independently per band.
    #[test]
    fn bandrank_per_band_median() {
        let a = Raster::new(1, 1, PixelFormat::Rgb8, vec![1, 200, 50]).unwrap();
        let b = Raster::new(1, 1, PixelFormat::Rgb8, vec![2, 100, 60]).unwrap();
        let c = Raster::new(1, 1, PixelFormat::Rgb8, vec![3, 150, 40]).unwrap();
        let ranked = a.bandrank(&[&b, &c], None);
        assert_eq!(ranked.getpoint(0, 0), vec![2.0, 150.0, 50.0]);
    }

    /// bandrank rejects an index >= image count, mismatched dimensions, and
    /// mismatched band counts with typed errors.
    #[test]
    fn bandrank_typed_errors() {
        let im = rgb2x1();
        assert!(matches!(
            im.try_bandrank(&[&im], Some(2)),
            Err(BandError::RankIndexOutOfRange { index: 2, count: 2 })
        ));
        let small = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            im.try_bandrank(&[&small], None),
            Err(BandError::DimensionMismatch { .. })
        ));
        let gray = gray2x1();
        assert!(matches!(
            im.try_bandrank(&[&gray], None),
            Err(BandError::BandCountMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    /// Mixed 8/16-bit bandrank inputs promote numerically to 16-bit.
    #[test]
    fn bandrank_mixed_depth_promotes() {
        let a = Raster::new(1, 1, PixelFormat::Gray8, vec![100]).unwrap();
        let b = gray16_1x1(300);
        let c = gray16_1x1(200);
        let ranked = a.bandrank(&[&b, &c], None);
        assert_eq!(ranked.format(), PixelFormat::Gray16);
        assert_eq!(ranked.getpoint(0, 0), vec![200.0]);
    }

    // ---- bandand / bandor / bandeor ----

    /// The bandbool family reduces bands with the bitwise op: for
    /// (0xF0, 0x0F, 0xFF) AND = 0x00, OR = 0xFF, XOR = 0x00; and for
    /// (0xCC, 0xAA, 0xF0) all three reductions are non-degenerate:
    /// AND = 0x80, OR = 0xFE, XOR = 0x96.
    #[test]
    fn bandbool_family_reduces_bands() {
        let im = rgb2x1();
        assert_eq!(im.bandand().getpoint(0, 0), vec![0.0]);
        assert_eq!(im.bandor().getpoint(0, 0), vec![255.0]);
        assert_eq!(im.bandeor().getpoint(0, 0), vec![0.0]);

        let im = Raster::new(1, 1, PixelFormat::Rgb8, vec![0xCC, 0xAA, 0xF0]).unwrap();
        assert_eq!(im.bandand().getpoint(0, 0), vec![0x80 as f64]);
        assert_eq!(im.bandor().getpoint(0, 0), vec![0xFE as f64]);
        assert_eq!(im.bandeor().getpoint(0, 0), vec![0x96 as f64]);
    }

    /// The reductions produce a single-band image of the input depth.
    #[test]
    fn bandbool_output_format() {
        let im = rgb2x1();
        assert_eq!(im.bandand().format(), PixelFormat::Gray8);
        let a = 0x0F0Fu16.to_ne_bytes();
        let b = 0x00FFu16.to_ne_bytes();
        let c = 0x0F00u16.to_ne_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&a);
        data.extend_from_slice(&b);
        data.extend_from_slice(&c);
        let im16 = Raster::new(1, 1, PixelFormat::Rgb16, data).unwrap();
        let anded = im16.bandand();
        assert_eq!(anded.format(), PixelFormat::Gray16);
        assert_eq!(
            anded.getpoint(0, 0),
            vec![(0x0F0F & 0x00FF & 0x0F00) as f64]
        );
        assert_eq!(
            im16.bandeor().getpoint(0, 0),
            vec![(0x0F0F ^ 0x00FF ^ 0x0F00) as f64]
        );
    }

    /// bandbool on a single-band image is an identity copy, matching
    /// libvips.
    #[test]
    fn bandbool_single_band_identity() {
        let g = gray2x1();
        assert_eq!(g.bandand().data(), g.data());
        assert_eq!(g.bandor().data(), g.data());
        assert_eq!(g.bandeor().data(), g.data());
    }

    // ---- extract_band / extract_bands ----

    /// extract_band pulls one band; index 0, a middle index, and -1 (last)
    /// all resolve correctly.
    #[test]
    fn extract_band_by_index_and_negative() {
        let im = rgb2x1();
        assert_eq!(im.extract_band(0).getpoint(1, 0), vec![10.0]);
        assert_eq!(im.extract_band(1).getpoint(1, 0), vec![20.0]);
        assert_eq!(im.extract_band(-1).getpoint(1, 0), vec![30.0]);
        // u32 indices work too (the ported tests pass `as u32` casts).
        assert_eq!(im.extract_band(2u32).getpoint(1, 0), vec![30.0]);
        assert_eq!(im.extract_band(0).format(), PixelFormat::Gray8);
    }

    /// extract_band keeps 16-bit depth.
    #[test]
    fn extract_band_16bit() {
        let a = 4096u16.to_ne_bytes();
        let b = 300u16.to_ne_bytes();
        let c = 7u16.to_ne_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&a);
        data.extend_from_slice(&b);
        data.extend_from_slice(&c);
        let im = Raster::new(1, 1, PixelFormat::Rgb16, data).unwrap();
        let band = im.extract_band(1);
        assert_eq!(band.format(), PixelFormat::Gray16);
        assert_eq!(band.getpoint(0, 0), vec![300.0]);
    }

    /// Out-of-range indices (positive and too-negative) are typed errors.
    #[test]
    fn extract_band_out_of_range_is_typed_error() {
        let im = rgb2x1();
        assert!(matches!(
            im.try_extract_band(3),
            Err(BandError::BandOutOfRange { band: 3, bands: 3 })
        ));
        assert!(matches!(
            im.try_extract_band(-4),
            Err(BandError::BandOutOfRange { band: -4, bands: 3 })
        ));
    }

    /// extract_bands copies a consecutive band range per pixel; a 2-band
    /// slice of an Rgb8 image is the canonical Multi8(2).
    #[test]
    fn extract_bands_middle_slice() {
        let im = rgb2x1();
        let s = im.extract_bands(1, 2);
        assert_eq!(s.format().channels(), 2);
        assert_eq!(s.getpoint(0, 0), vec![15.0, 255.0]);
        assert_eq!(s.getpoint(1, 0), vec![20.0, 30.0]);
    }

    /// Extracting every band reproduces the original raster and format.
    #[test]
    fn extract_bands_full_range_is_identity() {
        let im = rgb2x1();
        let s = im.extract_bands(0, 3);
        assert_eq!(s.format(), PixelFormat::Rgb8);
        assert_eq!(s.data(), im.data());
    }

    /// A negative start selects a suffix of the bands.
    #[test]
    fn extract_bands_negative_start() {
        let im = rgb2x1();
        let s = im.extract_bands(-2, 2);
        assert_eq!(s.getpoint(0, 0), vec![15.0, 255.0]);
    }

    /// Zero-length and out-of-range band ranges are typed errors.
    #[test]
    fn extract_bands_bad_ranges_are_typed_errors() {
        let im = rgb2x1();
        assert!(matches!(
            im.try_extract_bands(0, 0),
            Err(BandError::EmptyBandRange)
        ));
        assert!(matches!(
            im.try_extract_bands(2, 2),
            Err(BandError::BandRangeOutOfRange {
                start: 2,
                n: 2,
                bands: 3
            })
        ));
        assert!(matches!(
            im.try_extract_bands(-5, 1),
            Err(BandError::BandRangeOutOfRange { .. })
        ));
    }

    // ---- interactions across ops ----

    /// The ported test_bandjoin flow end-to-end: 3-band + 1-band = 4 bands;
    /// bandjoin_const(1) band 3 averages 1; bandjoin_vec([1,2]) gives 5
    /// bands with band 3 avg 1 and band 4 avg 2.
    #[test]
    fn ported_bandjoin_flow() {
        let colour = rgb2x1();
        let mono = gray2x1();
        let joined = colour.bandjoin(&mono);
        assert_eq!(joined.format().channels(), 4);

        let joined = colour.bandjoin_const(1.0);
        assert_eq!(joined.format().channels(), 4);
        let band3 = joined.extract_band(3);
        assert_eq!(band3.getpoint(0, 0), vec![1.0]);
        assert_eq!(band3.getpoint(1, 0), vec![1.0]);

        let joined = colour.bandjoin_vec(&[1.0, 2.0]);
        assert_eq!(joined.format().channels(), 5);
        assert_eq!(joined.extract_band(3).getpoint(0, 0), vec![1.0]);
        assert_eq!(joined.extract_band(4).getpoint(1, 0), vec![2.0]);
    }

    /// The ported test_bandfold flow: fold a wide mono image to width 1,
    /// unfold back to the original width with the same data.
    #[test]
    fn ported_bandfold_flow() {
        let w = 100u32;
        let data: Vec<u8> = (0..w as usize).map(|i| (i % 251) as u8).collect();
        let mono = Raster::new(w, 1, PixelFormat::Gray8, data).unwrap();

        let folded = mono.bandfold(None);
        assert_eq!(folded.width(), 1);
        assert_eq!(folded.format().channels(), 100);

        let unfolded = folded.bandunfold(None);
        assert_eq!(unfolded.width(), mono.width());
        assert_eq!(unfolded.data(), mono.data());

        let folded2 = mono.bandfold(Some(2));
        assert_eq!(folded2.width(), mono.width() / 2);
    }

    /// The band ops reject a sample kind they cannot read instead of
    /// misreading its bytes as `u16` pairs (float band maths lands with a
    /// later batch; cast to an unsigned format first). The panicking
    /// surface still panics, and the message now names the kind it was
    /// handed: it used to say "float rasters" for anything four bytes
    /// wide, which is the wrong carrier as soon as the unsigned 32-bit one
    /// exists (issue #517).
    #[test]
    #[should_panic(expected = "does not support F32 samples")]
    fn bands_float_panics() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::zeroed(2, 2, f1).unwrap();
        let _ = im.bandjoin_const(255.0);
    }

    // ------------------------------------------------------------------
    // the unsigned 32-bit carrier (issue #517)
    // ------------------------------------------------------------------

    /// A one-band `Uint32` raster from sample values.
    fn uint32(w: u32, h: u32, vals: &[u32]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let fmt = PixelFormat::Uint32(core::num::NonZeroU16::new(1).unwrap());
        Raster::new(w, h, fmt, data).unwrap()
    }

    fn u32_at(r: &Raster, i: usize) -> u32 {
        let d = r.data();
        u32::from_ne_bytes([d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]])
    }

    /**
     * Tests that the band ops carry the unsigned 32-bit carrier and keep a
     * value no 16-bit carrier holds, and that the output format comes off
     * the kind rather than the byte width.
     * Works by joining two `uint` rasters and taking their band mean,
     * against `/opt/homebrew/bin/vips` 8.18.6: `vips bandmean` on a
     * two-band `uint` image holding (90000, 128) answers **45064**, which
     * is `(90000 + 128) / 2`, and the output stays UINT. The mixed
     * 8-with-uint join is the case the byte width gets wrong: four bytes
     * names the float carrier.
     * Input: bandjoin(uint 90000, uint 128) -> Uint32(2); bandmean of it
     * -> 45064; bandjoin(u8, uint) -> Uint32(2).
     */
    #[test]
    fn band_ops_carry_the_uint_carrier() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let a = uint32(1, 1, &[90_000]);
        let b = uint32(1, 1, &[128]);
        let joined = a.try_bandjoin(&b).unwrap();
        assert_eq!(joined.format(), PixelFormat::Uint32(n(2)));
        assert_eq!(u32_at(&joined, 0), 90_000);
        assert_eq!(u32_at(&joined, 1), 128);

        let mean = joined.try_bandmean().unwrap();
        assert_eq!(mean.format(), PixelFormat::Uint32(n(1)));
        assert_eq!(u32_at(&mean, 0), 45_064);

        // The promotion the byte width cannot express.
        let mixed = gray2x1()
            .try_bandjoin(&uint32(2, 1, &[90_000, 90_000]))
            .unwrap();
        assert_eq!(mixed.format(), PixelFormat::Uint32(n(2)));
        assert_eq!(u32_at(&mixed, 0), 7);
        assert_eq!(u32_at(&mixed, 1), 90_000);

        // A band extract is a byte copy, so it has to keep the carrier too.
        let one = joined.try_extract_band(1).unwrap();
        assert_eq!(one.format(), PixelFormat::Uint32(n(1)));
        assert_eq!(u32_at(&one, 0), 128);

        // Control: the 8-with-16 promotion that existed before.
        let old = gray2x1().try_bandjoin(&gray2x1()).unwrap();
        assert_eq!(old.format().bytes_per_channel(), 1);
    }

    /**
     * Tests that a band op handed a sample kind it cannot read answers a
     * typed error rather than panicking out of a `Result`, the shape issue
     * #694 landed, and that the message names the kind it was given rather
     * than claiming the raster is float.
     * Works by handing a float raster to three band ops, with the uint
     * carrier as the positive control that the guard is about the kind.
     * Input: float -> Err(UnsupportedSampleKind { kind: F32 }); uint -> Ok.
     */
    #[test]
    fn band_ops_refuse_a_kind_they_cannot_read() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let f = Raster::new(
            1,
            1,
            PixelFormat::FloatF32(n(1)),
            1.5f32.to_ne_bytes().to_vec(),
        )
        .unwrap();
        for got in [f.try_bandmean(), f.try_bandand(), f.try_bandjoin(&f)] {
            assert!(
                matches!(
                    got,
                    Err(BandError::UnsupportedSampleKind {
                        kind: SampleKind::F32,
                        ..
                    })
                ),
                "a float raster gave {got:?}"
            );
        }
        assert_eq!(
            f.try_bandmean().unwrap_err().to_string(),
            "bandmean does not support F32 samples yet"
        );
        // Positive control: the uint carrier passes all three.
        let u = uint32(1, 1, &[90_000]);
        assert!(u.try_bandmean().is_ok());
        assert!(u.try_bandand().is_ok());
        assert!(u.try_bandjoin(&u).is_ok());
    }
}
