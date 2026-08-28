//! Histogram operations ported from libvips.
//!
//! This module is the sixth batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`], and
//! [`crate::draw`]): computing histograms, transforming them (cumulative,
//! normalised, matched), applying them back to images as look-up tables, and
//! the equalisation ops built from those pieces. Operations that can fail on
//! caller input exist in two forms, following the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, HistogramError>` with
//!   typed errors for malformed histograms, mismatched dimensions and band
//!   counts, and out-of-range arguments; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`hist_cum`, `maplut`, `percent`, ...) exactly, delegating to the
//!   `try_*` form.
//!
//! [`Raster::hist_equal`] validates nothing (any raster equalises) and has
//! only the direct form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::hist_find`] | `vips_hist_find` | per-band value-count histogram |
//! | [`Raster::hist_find_band`] | `vips_hist_find` with `band` | one band's histogram |
//! | [`Raster::hist_find_indexed`] | `vips_hist_find_indexed` | per-index sample sums |
//! | [`Raster::hist_find_ndim`] | `vips_hist_find_ndim` | up to 3-dimensional histogram |
//! | [`Raster::hist_cum`] | `vips_hist_cum` | running sum along the histogram |
//! | [`Raster::hist_norm`] | `vips_hist_norm` | band maxima scaled to the max index |
//! | [`Raster::hist_equal`] | `vips_hist_equal` | global histogram equalisation |
//! | [`Raster::hist_local`] | `vips_hist_local` | local (CLAHE) equalisation |
//! | [`Raster::hist_match`] | `vips_hist_match` | histogram-specification LUT |
//! | [`Raster::hist_plot`] | `vips_hist_plot` | bar-graph image of a histogram |
//! | [`Raster::hist_entropy`] | `vips_hist_entropy` | Shannon entropy, `f64` bits |
//! | [`Raster::hist_ismonotonic`] | `vips_hist_ismonotonic` | `bool` |
//! | [`Raster::maplut`] | `vips_maplut` | image mapped through a LUT |
//! | [`Raster::case`] | `vips_case` | index image mapped to scalar cases |
//! | [`Raster::percent`] | `vips_percent` | percentile threshold, `f64` |
//!
//! # Semantics shared by every operation
//!
//! * **Histogram shape.** A histogram is itself a [`Raster`]: a `N`x1 or
//!   1x`N` image whose sample at element `i` (band `b`) is the count or
//!   table entry for value `i`. Operations that consume a histogram
//!   ([`Raster::hist_cum`], [`Raster::maplut`], ...) require this shape and
//!   return [`HistogramError::NotAHistogram`] otherwise. Element order is
//!   identical for both orientations (row-major data with interleaved
//!   bands), and outputs preserve the input's orientation.
//! * **Count depth.** [`PixelFormat`] has no unsigned depth wider than 16
//!   bits, so every op that produces counts or sums writes 16-bit samples
//!   and saturates at `65535`. This is the documented contract until a
//!   wider sample kind lands. Operations that consume pixel-value
//!   distributions internally ([`Raster::hist_equal`],
//!   [`Raster::hist_local`], [`Raster::percent`]) compute full-precision
//!   `u64` histograms directly from the image and are exact regardless of
//!   image size.
//!
//!   **The ceiling is a deviation and any image over 256x256 reaches it**,
//!   because these are pixel counters. What libvips emits instead is not
//!   one format, measured on 8.18.6 (issue #759):
//!
//!   | op | vips output format |
//!   |---|---|
//!   | `hist_find`, `hist_find_ndim` | `UINT`, whatever the input |
//!   | `hist_find_indexed` | `DOUBLE`, whatever the input and either `combine` |
//!   | `hist_cum` | `UINT` / `INT` / `FLOAT` / `DOUBLE`, following the input |
//!
//!   So closing this needs more than the uint carrier of issue #517: the
//!   signed carriers of #516 for `hist_cum` on a signed input, and a
//!   double one, which is why #518 being closed matters here. Reading the
//!   libvips source will mislead you on `hist_find`: for a
//!   `VipsStatisticClass` the per-op `format_table` is an **input cast**
//!   table (`statistic.c`), not the output format, which is set separately
//!   to `UINT` in `hist_find.c`.
//! * **Bins.** 8-bit images histogram into 256 bins, 16-bit images into
//!   65536 bins, indexed by the raw sample value.
//! * **Bands.** `hist_find`, `hist_cum`, `hist_norm`, `hist_equal`,
//!   `hist_local`, and `hist_match` treat bands independently, matching the
//!   libvips defaults. `hist_entropy` and `percent` pool every band into
//!   one distribution.
//!
//! # Deferred operations
//!
//! The ported histogram suite also exercises `stdif` (already implemented
//! in [`crate::arithmetic`]) and the LUT constructors `identity`, `grey`,
//! and `switch` (already implemented in [`crate::conversion`]). The libvips
//! histogram family members not called by the ported tests (`hist_find`
//! variants aside, `invertlut`, `buildlut`, `tonelut`, and the Hough
//! transforms) belong to later batches.

use crate::pixel::{PixelFormat, SampleKind};
use crate::raster::{Raster, RasterError};
use thiserror::Error;

/// Typed errors for the histogram operations in [`crate::histogram`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HistogramError {
    /// An operation that consumes a histogram was given an image that is
    /// neither `N`x1 nor 1x`N`.
    #[error("not a histogram: expected an Nx1 or 1xN image, got {width}x{height}")]
    NotAHistogram { width: u32, height: u32 },
    /// An operation requires a one-band image.
    #[error("{op} requires a one-band image, got {bands} bands")]
    OneBandOnly { op: &'static str, bands: usize },
    /// Two rasters that must share pixel dimensions do not.
    #[error("dimension mismatch: {expected_w}x{expected_h} vs {got_w}x{got_h}")]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        got_w: u32,
        got_h: u32,
    },
    /// Two histograms that must share a band count do not.
    #[error("band-count mismatch: expected {expected} bands, got {got}")]
    BandCountMismatch { expected: usize, got: usize },
    /// `maplut` band counts are incompatible: either side may have one
    /// band, or both must match.
    #[error("maplut band counts are incompatible: image has {image} bands, LUT has {lut}")]
    LutBandMismatch { image: usize, lut: usize },
    /// A band index is out of range.
    #[error("band {band} out of range for a {bands}-band image")]
    InvalidBand { band: u32, bands: usize },
    /// `hist_find_ndim` input has more than three bands.
    #[error("hist_find_ndim supports at most 3 bands, image has {bands}")]
    TooManyDimensions { bands: usize },
    /// A `hist_find_ndim` bin count is zero or exceeds the value range.
    #[error("invalid bin count {bins}: must be between 1 and {max}")]
    InvalidBins { bins: u32, max: u32 },
    /// A `hist_local` window dimension is zero.
    #[error("hist_local window dimensions must be greater than zero")]
    ZeroWindow,
    /// A `hist_local` contrast limit is negative or not finite.
    #[error("invalid max_slope {max_slope}: must be finite and non-negative")]
    InvalidMaxSlope { max_slope: f64 },
    /// `case` was given an empty case list.
    #[error("case requires at least one case value")]
    EmptyCases,
    /// A `percent` argument is outside `0..=100` or not finite.
    #[error("invalid percent {percent}: must be within 0..=100")]
    InvalidPercent { percent: f64 },
    /// The result would have more bands than [`PixelFormat`] can carry.
    #[error("result band count {bands} exceeds the supported maximum of 65535")]
    TooManyBands { bands: usize },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_hist<T>(op: &str, r: Result<T, HistogramError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Sample-level helpers
// ---------------------------------------------------------------------------

/// Read the flat `i`-th sample as a `u32` bin index (native byte order for
/// the multi-byte kinds, matching [`crate::raster_ops`]). Integer kinds
/// only: the [`SampleKind::F32`] arm panics rather than misreading float
/// bytes as `u16` pairs, which is what the histogram ops did before the
/// float formats existed.
///
/// This is the read a `VipsStatisticClass` op performs *after* its input
/// cast, which is why the signed and 32-bit arms fold rather than widen: a
/// negative sample indexes bin zero and a sample past 65535 indexes the
/// last bin of the 16-bit table. Both are measured, not read out of the C,
/// and the measurements are on [`SampleKind::hist_bins`].
///
/// It doubles as the read for a histogram's own counts, which is exact for
/// every carrier the crate has because a count is non-negative and
/// saturates at 65535 anyway (issue #532). A wider count carrier needs the
/// two reads separated.
///
/// The match is over the kind and has no wildcard, so a carrier added to
/// [`SampleKind`] is a compile error here instead of a silent misread
/// (issue #607).
#[inline]
fn read_flat(data: &[u8], kind: SampleKind, i: usize) -> u32 {
    match kind {
        SampleKind::U8 => data[i] as u32,
        SampleKind::U16 => u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as u32,
        // A signed sample is folded through the unsigned kind of the same
        // width, saturating, so every negative sample indexes bin zero.
        // That is what libvips does and it is observable rather than read
        // out of the C: on 8.18.6 a `char` image of `[-128, -1, 0, 127]`
        // histograms to `bin 0 = 3`, `bin 127 = 1`. It is the
        // `VipsStatisticClass` input cast, not a signed bin table.
        SampleKind::I8 => (data[i] as i8).max(0) as u32,
        SampleKind::I16 => i16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]).max(0) as u32,
        // The 32-bit kinds have no value-indexed table of their own
        // (`SampleKind::hist_bins` is `None` for both) and libvips casts
        // them into `ushort` before counting, measured: a `uint` image
        // whose largest sample is 70000 gives a 65536-wide histogram.
        SampleKind::U32 => u32::from_ne_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ])
        .min(u32::from(u16::MAX)),
        SampleKind::I32 => i32::from_ne_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ])
        .clamp(0, i32::from(i16::MAX) * 2 + 1) as u32,
        SampleKind::F32 => panic!(
            "the histogram operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Write the flat `i`-th sample, saturating into the kind's ceiling.
/// Unsigned kinds only; see [`read_flat`], including on why the match has
/// no wildcard arm.
#[inline]
fn write_flat(data: &mut [u8], kind: SampleKind, i: usize, v: u32) {
    match kind {
        SampleKind::U8 => data[i] = v.min(255) as u8,
        SampleKind::U16 => {
            let b = (v.min(65535) as u16).to_ne_bytes();
            data[i * 2] = b[0];
            data[i * 2 + 1] = b[1];
        }
        // Counts are non-negative, so only the ceiling can bind and the
        // signed kinds saturate at their positive end. Sourced from
        // `SampleKind::max_value` rather than re-spelling 127 / 32767 /
        // 2147483647 here.
        SampleKind::I8 => data[i] = v.min(0x7F) as u8,
        SampleKind::I16 => {
            let b = (v.min(0x7FFF) as u16).to_ne_bytes();
            data[i * 2] = b[0];
            data[i * 2 + 1] = b[1];
        }
        SampleKind::U32 => data[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes()),
        SampleKind::I32 => {
            data[i * 4..i * 4 + 4].copy_from_slice(&v.min(0x7FFF_FFFF).to_ne_bytes());
        }
        SampleKind::F32 => panic!(
            "the histogram operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Number of histogram bins for an integer sample kind: 256 or 65536.
///
/// Reads [`SampleKind::hist_bins`] where the kind has a table of its own,
/// so a carrier added to [`SampleKind`] gets its bin count there rather
/// than here. The 32-bit kinds have no table of their own and libvips does
/// not build one: it casts the input into `ushort` first, measured on
/// 8.18.6, so the table stays 65536 wide and [`read_flat`] performs the
/// matching saturation. `F32` keeps the "no float rasters yet" panic.
#[inline]
fn bins_for(kind: SampleKind) -> usize {
    match kind {
        SampleKind::U8 | SampleKind::I8 | SampleKind::U16 | SampleKind::I16 => kind
            .hist_bins()
            .expect("the 8- and 16-bit kinds have a bin table"),
        SampleKind::U32 | SampleKind::I32 => {
            SampleKind::U16.hist_bins().expect("U16 has a bin table")
        }
        SampleKind::F32 => panic!(
            "the histogram operations do not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// The canonical format for a band count and sample kind, or a typed error.
fn format_for(bands: usize, kind: SampleKind) -> Result<PixelFormat, HistogramError> {
    PixelFormat::with_kind(bands, kind).ok_or(HistogramError::TooManyBands { bands })
}

/// The element count of a histogram-shaped raster (`N`x1 or 1x`N`), or a
/// typed error for any other shape.
fn hist_len(r: &Raster) -> Result<usize, HistogramError> {
    if r.width() == 1 || r.height() == 1 {
        Ok(r.width() as usize * r.height() as usize)
    } else {
        Err(HistogramError::NotAHistogram {
            width: r.width(),
            height: r.height(),
        })
    }
}

/// Full-precision per-band value histograms of an image.
fn per_band_hist(r: &Raster) -> Vec<Vec<u64>> {
    let fmt = r.format();
    let bands = fmt.channels();
    let kind = fmt.kind();
    let bins = bins_for(kind);
    let n = r.width() as usize * r.height() as usize;
    let data = r.data();
    let mut hists = vec![vec![0u64; bins]; bands];
    for i in 0..n {
        for (b, hist) in hists.iter_mut().enumerate() {
            hist[read_flat(data, kind, i * bands + b) as usize] += 1;
        }
    }
    hists
}

/// Row count for a [`Raster::hist_plot`] bar graph of `values`.
///
/// `uchar` is the only format libvips gives a fixed plot height, and that
/// is measured rather than inferred from the width: on 8.18.6 a histogram
/// of `[0, 5]` plots 256 rows high as `uchar` and 5 rows high as `char`,
/// `ushort`, `short`, `uint` or `int`. So the signed one-byte kind belongs
/// with the data-driven group even though it shares `U8`'s width, which is
/// exactly the distinction [`SampleKind`] exists to carry.
///
/// A separate function rather than a `match` inside `try_hist_plot` so the
/// grouping can be asserted for the kinds no `PixelFormat` produces, since
/// there is no raster to hand the op.
///
/// # The height rule, measured
///
/// `vips hist_plot` on 8.18.6, `ushort` input: `[0, 1]` plots 1 row,
/// `[1, 1]` plots 1, `[0, 0, 0]` plots 1, `[0, 5]` plots 5, `[3, 9]` plots
/// 9, `[100, 200]` plots 200 and `[65535, 0]` plots 65535. So the height is
/// the largest count, floored at one, and **not** `max + 1`, which is what
/// this returned before issue #802.
///
/// The `[3, 9]` row is the one that shows the floor is a literal zero
/// rather than the smallest count: 9 rows, not 6. The full libvips rule is
/// `max - min(0, min)`, and the second term only fires on a histogram
/// holding a negative count (a `char` `[-5, -1]` plots 4 rows). Every
/// carrier this crate has is unsigned and [`read_flat`] folds a negative
/// sample into bin zero besides, so `values` cannot contain one and the
/// term is left out rather than written as dead arithmetic. The signed
/// carriers of issue #516 are where it starts to matter.
#[inline]
fn plot_height(kind: SampleKind, values: &[u32]) -> usize {
    match kind {
        SampleKind::U8 => 256,
        SampleKind::U16
        | SampleKind::I8
        | SampleKind::I16
        | SampleKind::U32
        | SampleKind::I32
        | SampleKind::F32 => values.iter().copied().max().unwrap_or(0).max(1) as usize,
    }
}

/// Saturate a `u64` count into a 16-bit sample value.
#[inline]
fn sat16(v: u64) -> u32 {
    v.min(u16::MAX as u64) as u32
}

impl Raster {
    // -----------------------------------------------------------------------
    // hist_find family
    // -----------------------------------------------------------------------

    /// Compute the per-band value histogram, like `vips_hist_find` with the
    /// default `band` of -1.
    ///
    /// The result is a 256x1 (8-bit input) or 65536x1 (16-bit input) image
    /// with the input's band count: sample `(v, 0)` of band `b` is the
    /// number of band-`b` samples equal to `v`. Counts are written as
    /// 16-bit samples and saturate at `65535` (see the module notes on
    /// count depth).
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::Raster`] if the histogram would exceed the
    /// allocation budget (only possible for extreme band counts).
    pub fn try_hist_find(&self) -> Result<Raster, HistogramError> {
        let bands = self.format().channels();
        let bins = bins_for(self.format().kind());
        let hists = per_band_hist(self);
        let out_fmt = format_for(bands, SampleKind::U16)?;
        let mut out = Raster::zeroed(bins as u32, 1, out_fmt)?;
        let buf = out.data_mut();
        for (b, hist) in hists.iter().enumerate() {
            for (v, &n) in hist.iter().enumerate() {
                write_flat(buf, SampleKind::U16, v * bands + b, sat16(n));
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_find`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_find`].
    #[track_caller]
    pub fn hist_find(&self) -> Raster {
        expect_hist("hist_find", self.try_hist_find())
    }

    /// Compute the value histogram of one band, like `vips_hist_find` with
    /// an explicit `band`.
    ///
    /// The result is a one-band 256x1 or 65536x1 image; see
    /// [`Raster::try_hist_find`] for the count semantics.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::InvalidBand`] if `band` is out of range.
    pub fn try_hist_find_band(&self, band: u32) -> Result<Raster, HistogramError> {
        let bands = self.format().channels();
        if band as usize >= bands {
            return Err(HistogramError::InvalidBand { band, bands });
        }
        let kind = self.format().kind();
        let bins = bins_for(kind);
        let n = self.width() as usize * self.height() as usize;
        let data = self.data();
        let mut hist = vec![0u64; bins];
        for i in 0..n {
            hist[read_flat(data, kind, i * bands + band as usize) as usize] += 1;
        }
        let mut out = Raster::zeroed(bins as u32, 1, PixelFormat::Gray16)?;
        let buf = out.data_mut();
        for (v, &count) in hist.iter().enumerate() {
            write_flat(buf, SampleKind::U16, v, sat16(count));
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_find_band`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_find_band`].
    #[track_caller]
    pub fn hist_find_band(&self, band: u32) -> Raster {
        expect_hist("hist_find_band", self.try_hist_find_band(band))
    }

    /// Compute a histogram indexed by a second image, like
    /// `vips_hist_find_indexed` with the default `sum` combine.
    ///
    /// For every pixel, the sample values of `self` are added to the output
    /// element selected by the corresponding `index` pixel. The result has
    /// one element per possible index value (256x1 for an 8-bit index,
    /// 65536x1 for 16-bit) and the input's band count. Sums are written as
    /// 16-bit samples and saturate at `65535`.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::DimensionMismatch`] if `index` differs in
    /// size, or [`HistogramError::OneBandOnly`] if `index` has more than
    /// one band.
    pub fn try_hist_find_indexed(&self, index: &Raster) -> Result<Raster, HistogramError> {
        if (self.width(), self.height()) != (index.width(), index.height()) {
            return Err(HistogramError::DimensionMismatch {
                expected_w: self.width(),
                expected_h: self.height(),
                got_w: index.width(),
                got_h: index.height(),
            });
        }
        let index_bands = index.format().channels();
        if index_bands != 1 {
            return Err(HistogramError::OneBandOnly {
                op: "hist_find_indexed index",
                bands: index_bands,
            });
        }
        let bands = self.format().channels();
        let kind = self.format().kind();
        let idx_kind = index.format().kind();
        let bins = bins_for(idx_kind);
        let n = self.width() as usize * self.height() as usize;
        let data = self.data();
        let idx_data = index.data();
        let mut sums = vec![0u64; bins * bands];
        for i in 0..n {
            let slot = read_flat(idx_data, idx_kind, i) as usize;
            for b in 0..bands {
                sums[slot * bands + b] += read_flat(data, kind, i * bands + b) as u64;
            }
        }
        let out_fmt = format_for(bands, SampleKind::U16)?;
        let mut out = Raster::zeroed(bins as u32, 1, out_fmt)?;
        let buf = out.data_mut();
        for (i, &s) in sums.iter().enumerate() {
            write_flat(buf, SampleKind::U16, i, sat16(s));
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_find_indexed`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see
    /// [`Raster::try_hist_find_indexed`].
    #[track_caller]
    pub fn hist_find_indexed(&self, index: &Raster) -> Raster {
        expect_hist("hist_find_indexed", self.try_hist_find_indexed(index))
    }

    /// Compute an up to three-dimensional histogram, like
    /// `vips_hist_find_ndim`.
    ///
    /// Each pixel's bands select one cell of a `bins`-per-side grid: band 0
    /// selects the column, band 1 (when present) the row, and band 2 (when
    /// present) the output band. A sample value `v` falls in bin
    /// `v * bins / range` where `range` is 256 or 65536 by depth. `bins`
    /// defaults to 10, matching libvips. Counts are written as 16-bit
    /// samples and saturate at `65535`.
    ///
    /// The output is `bins` wide, `bins` high when the input has two or
    /// more bands (1 otherwise), and has `bins` bands when the input has
    /// three bands (1 otherwise).
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::TooManyDimensions`] for more than three
    /// bands, [`HistogramError::InvalidBins`] if `bins` is zero or exceeds
    /// the value range, [`HistogramError::TooManyBands`] if a 3-band
    /// histogram would need more than 65535 output bands, or
    /// [`HistogramError::Raster`] if the result exceeds the allocation
    /// budget.
    pub fn try_hist_find_ndim(&self, bins: Option<u32>) -> Result<Raster, HistogramError> {
        let bands = self.format().channels();
        if bands > 3 {
            return Err(HistogramError::TooManyDimensions { bands });
        }
        let kind = self.format().kind();
        let range = bins_for(kind) as u64;
        let bins = bins.unwrap_or(10);
        if bins == 0 || bins as u64 > range {
            return Err(HistogramError::InvalidBins {
                bins,
                max: range as u32,
            });
        }
        let out_w = bins;
        let out_h = if bands >= 2 { bins } else { 1 };
        let out_bands = if bands >= 3 { bins as usize } else { 1 };
        let out_fmt = format_for(out_bands, SampleKind::U16)?;
        let mut out = Raster::zeroed(out_w, out_h, out_fmt)?;
        let buf = out.data_mut();

        let n = self.width() as usize * self.height() as usize;
        let data = self.data();
        let bin_of = |v: u32| -> usize { (v as u64 * bins as u64 / range) as usize };
        for i in 0..n {
            let bx = bin_of(read_flat(data, kind, i * bands));
            let by = if bands >= 2 {
                bin_of(read_flat(data, kind, i * bands + 1))
            } else {
                0
            };
            let bz = if bands >= 3 {
                bin_of(read_flat(data, kind, i * bands + 2))
            } else {
                0
            };
            let cell = (by * out_w as usize + bx) * out_bands + bz;
            let cur = read_flat(buf, SampleKind::U16, cell);
            if cur < u16::MAX as u32 {
                write_flat(buf, SampleKind::U16, cell, cur + 1);
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_find_ndim`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_find_ndim`].
    #[track_caller]
    pub fn hist_find_ndim(&self, bins: Option<u32>) -> Raster {
        expect_hist("hist_find_ndim", self.try_hist_find_ndim(bins))
    }

    // -----------------------------------------------------------------------
    // Histogram transforms
    // -----------------------------------------------------------------------

    /// Compute the cumulative histogram, like `vips_hist_cum`.
    ///
    /// Each band is replaced by its running sum along the histogram.
    /// 8-bit input is promoted to 16-bit output (libvips promotes to
    /// 32-bit); sums saturate at `65535`.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if the image is neither
    /// `N`x1 nor 1x`N`.
    pub fn try_hist_cum(&self) -> Result<Raster, HistogramError> {
        let n = hist_len(self)?;
        let bands = self.format().channels();
        let kind = self.format().kind();
        let out_fmt = format_for(bands, SampleKind::U16)?;
        let mut out = Raster::zeroed(self.width(), self.height(), out_fmt)?;
        let buf = out.data_mut();
        let data = self.data();
        for b in 0..bands {
            let mut sum = 0u64;
            for i in 0..n {
                sum += read_flat(data, kind, i * bands + b) as u64;
                write_flat(buf, SampleKind::U16, i * bands + b, sat16(sum));
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_cum`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_cum`].
    #[track_caller]
    pub fn hist_cum(&self) -> Raster {
        expect_hist("hist_cum", self.try_hist_cum())
    }

    /// Normalise a histogram, like `vips_hist_norm`.
    ///
    /// Each band is scaled so its maximum equals the maximum index
    /// (`N - 1` for an `N`-element histogram), which makes a cumulative
    /// histogram usable as an equalisation LUT. All-zero bands stay zero.
    /// The output depth is 8-bit when `N <= 256` and 16-bit otherwise, so
    /// normalising a 256-element histogram yields a `maplut`-ready 8-bit
    /// LUT.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if the image is neither
    /// `N`x1 nor 1x`N`.
    pub fn try_hist_norm(&self) -> Result<Raster, HistogramError> {
        let n = hist_len(self)?;
        let bands = self.format().channels();
        let kind = self.format().kind();
        let out_kind = if n <= 256 {
            SampleKind::U8
        } else {
            SampleKind::U16
        };
        let out_fmt = format_for(bands, out_kind)?;
        let mut out = Raster::zeroed(self.width(), self.height(), out_fmt)?;
        let buf = out.data_mut();
        let data = self.data();
        for b in 0..bands {
            let mut max = 0u32;
            for i in 0..n {
                max = max.max(read_flat(data, kind, i * bands + b));
            }
            if max == 0 {
                continue;
            }
            let scale = (n - 1) as f64 / max as f64;
            for i in 0..n {
                let v = read_flat(data, kind, i * bands + b) as f64;
                write_flat(buf, out_kind, i * bands + b, (v * scale).round() as u32);
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_norm`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_norm`].
    #[track_caller]
    pub fn hist_norm(&self) -> Raster {
        expect_hist("hist_norm", self.try_hist_norm())
    }

    /// Build a LUT that matches this histogram to a reference histogram,
    /// like `vips_hist_match`.
    ///
    /// Both images must be histogram-shaped with equal band counts. For
    /// each band the cumulative distributions are compared: entry `i` of
    /// the result is the smallest reference index whose cumulative fraction
    /// reaches the input's cumulative fraction at `i`. Matching a histogram
    /// to itself therefore yields the identity LUT. Bands whose total is
    /// zero (in either image) map to zero. The output has this histogram's
    /// shape; its depth is 8-bit when the reference has at most 256
    /// elements and 16-bit otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if either image is not
    /// histogram-shaped, or [`HistogramError::BandCountMismatch`] if the
    /// band counts differ.
    pub fn try_hist_match(&self, reference: &Raster) -> Result<Raster, HistogramError> {
        let n_in = hist_len(self)?;
        let n_ref = hist_len(reference)?;
        let bands = self.format().channels();
        let ref_bands = reference.format().channels();
        if bands != ref_bands {
            return Err(HistogramError::BandCountMismatch {
                expected: bands,
                got: ref_bands,
            });
        }
        let kind = self.format().kind();
        let ref_bpc = reference.format().kind();
        let out_kind = if n_ref <= 256 {
            SampleKind::U8
        } else {
            SampleKind::U16
        };
        let out_fmt = format_for(bands, out_kind)?;
        let mut out = Raster::zeroed(self.width(), self.height(), out_fmt)?;
        let buf = out.data_mut();
        let data = self.data();
        let ref_data = reference.data();

        for b in 0..bands {
            let mut in_cum = vec![0f64; n_in];
            let mut sum = 0u64;
            for (i, c) in in_cum.iter_mut().enumerate() {
                sum += read_flat(data, kind, i * bands + b) as u64;
                *c = sum as f64;
            }
            let in_total = sum;
            let mut ref_cum = vec![0f64; n_ref];
            let mut sum = 0u64;
            for (i, c) in ref_cum.iter_mut().enumerate() {
                sum += read_flat(ref_data, ref_bpc, i * bands + b) as u64;
                *c = sum as f64;
            }
            let ref_total = sum;
            if in_total == 0 || ref_total == 0 {
                continue;
            }
            let mut j = 0usize;
            for (i, &c) in in_cum.iter().enumerate() {
                let target = c / in_total as f64;
                while j < n_ref - 1 && ref_cum[j] / (ref_total as f64) < target {
                    j += 1;
                }
                write_flat(buf, out_kind, i * bands + b, j as u32);
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_match`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_match`].
    #[track_caller]
    pub fn hist_match(&self, reference: &Raster) -> Raster {
        expect_hist("hist_match", self.try_hist_match(reference))
    }

    /// Plot a one-band histogram as a bar graph, like `vips_hist_plot`.
    ///
    /// The output is `N` columns wide for an `N`-element histogram. 8-bit
    /// histograms plot 256 rows high whatever the counts are; every other
    /// depth plots as many rows as the largest count, floored at one. Both
    /// halves are measured against libvips 8.18.6 rather than asserted;
    /// see [`plot_height`], and note that the second half used to say
    /// `max + 1` and used to say that matched libvips (issue #802).
    ///
    /// The result is a `Gray8` image: in column `x`, the bottom `hist[x]`
    /// pixels are `255` and the rest `0`, so the graph reads with its
    /// origin at the bottom left.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if the image is not
    /// histogram-shaped, [`HistogramError::OneBandOnly`] for a multiband
    /// histogram, or [`HistogramError::Raster`] if the plot would exceed
    /// the allocation budget.
    pub fn try_hist_plot(&self) -> Result<Raster, HistogramError> {
        let n = hist_len(self)?;
        let bands = self.format().channels();
        if bands != 1 {
            return Err(HistogramError::OneBandOnly {
                op: "hist_plot",
                bands,
            });
        }
        let kind = self.format().kind();
        let data = self.data();
        let values: Vec<u32> = (0..n).map(|i| read_flat(data, kind, i)).collect();
        // Total over the kind rather than "8-bit or not", so a carrier
        // added to `SampleKind` has to state its plot height here instead
        // of inheriting the 16-bit branch (issue #607). `F32` is
        // unreachable in practice: `read_flat` above panics on it for any
        // non-empty histogram, and `hist_len` rejects the empty shape.
        let height = plot_height(kind, &values);
        let mut out = Raster::zeroed(n as u32, height as u32, PixelFormat::Gray8)?;
        let buf = out.data_mut();
        for (x, &v) in values.iter().enumerate() {
            for y in (height - v as usize)..height {
                buf[y * n + x] = 255;
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_hist_plot`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_plot`].
    #[track_caller]
    pub fn hist_plot(&self) -> Raster {
        expect_hist("hist_plot", self.try_hist_plot())
    }

    /// Shannon entropy of a histogram in bits, like `vips_hist_entropy`.
    ///
    /// Every band is pooled into one distribution: with `p_i` the fraction
    /// of the total count in cell `i`, the result is `-sum(p_i *
    /// log2(p_i))` over the non-zero cells. An all-zero histogram has
    /// entropy `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if the image is not
    /// histogram-shaped.
    pub fn try_hist_entropy(&self) -> Result<f64, HistogramError> {
        let n = hist_len(self)?;
        let bands = self.format().channels();
        let kind = self.format().kind();
        let data = self.data();
        let count = n * bands;
        let mut total = 0u64;
        for i in 0..count {
            total += read_flat(data, kind, i) as u64;
        }
        if total == 0 {
            return Ok(0.0);
        }
        let mut entropy = 0.0;
        for i in 0..count {
            let v = read_flat(data, kind, i);
            if v > 0 {
                let p = v as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }
        Ok(entropy)
    }

    /// Panicking form of [`Raster::try_hist_entropy`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_entropy`].
    #[track_caller]
    pub fn hist_entropy(&self) -> f64 {
        expect_hist("hist_entropy", self.try_hist_entropy())
    }

    /// Whether a histogram is monotonically non-decreasing, like
    /// `vips_hist_ismonotonic`.
    ///
    /// Returns `true` only if every band is non-decreasing along the
    /// histogram. A LUT must be monotonic to preserve value ordering when
    /// applied with [`Raster::maplut`].
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if the image is not
    /// histogram-shaped.
    pub fn try_hist_ismonotonic(&self) -> Result<bool, HistogramError> {
        let n = hist_len(self)?;
        let bands = self.format().channels();
        let kind = self.format().kind();
        let data = self.data();
        for b in 0..bands {
            let mut prev = 0u32;
            for i in 0..n {
                let v = read_flat(data, kind, i * bands + b);
                if v < prev {
                    return Ok(false);
                }
                prev = v;
            }
        }
        Ok(true)
    }

    /// Panicking form of [`Raster::try_hist_ismonotonic`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see
    /// [`Raster::try_hist_ismonotonic`].
    #[track_caller]
    pub fn hist_ismonotonic(&self) -> bool {
        expect_hist("hist_ismonotonic", self.try_hist_ismonotonic())
    }

    // -----------------------------------------------------------------------
    // Equalisation
    // -----------------------------------------------------------------------

    /// Histogram-equalise the image, like `vips_hist_equal` with the
    /// default of equalising each band independently.
    ///
    /// For each band the LUT is `round(cdf(v) * (bins - 1) / n_pixels)`,
    /// the composition `hist_find`, `hist_cum`, `hist_norm`, `maplut`
    /// performs, computed here with full-precision `u64` counts so image
    /// size never saturates. The output keeps the input's size and format.
    /// A constant band maps to the depth maximum (its cumulative
    /// distribution jumps straight to 1).
    pub fn hist_equal(&self) -> Raster {
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bins = bins_for(kind);
        let n = self.width() as usize * self.height() as usize;
        let data = self.data();
        let mut out = vec![0u8; data.len()];
        // The scale is computed once per band, exactly as hist_norm does,
        // so this equals the hist_find / hist_cum / hist_norm / maplut
        // pipeline bit for bit (dividing per entry instead can differ by
        // one ulp at rounding boundaries).
        let scale = (bins - 1) as f64 / n as f64;
        for b in 0..bands {
            let mut hist = vec![0u64; bins];
            for i in 0..n {
                hist[read_flat(data, kind, i * bands + b) as usize] += 1;
            }
            let mut lut = vec![0u32; bins];
            let mut cum = 0u64;
            for (l, &h) in lut.iter_mut().zip(hist.iter()) {
                cum += h;
                *l = (cum as f64 * scale).round() as u32;
            }
            for i in 0..n {
                let v = read_flat(data, kind, i * bands + b) as usize;
                write_flat(&mut out, kind, i * bands + b, lut[v]);
            }
        }
        Raster::new(self.width(), self.height(), fmt, out)
            .expect("hist_equal output is well-formed")
    }

    /// Local histogram equalisation over a sliding window, like
    /// `vips_hist_local` (CLAHE when `max_slope` is set).
    ///
    /// Every output sample is the equalisation of its input sample within
    /// the `width` x `height` window centred on it:
    /// `round(cdf(v) * (bins - 1) / area)` where `cdf(v)` counts the window
    /// samples less than or equal to `v`. Windows extend past the edges by
    /// replicating the border samples (libvips `extend: copy`), so the
    /// window area is constant. Bands are processed independently and the
    /// output keeps the input's size and format.
    ///
    /// `max_slope` is the CLAHE contrast limit: the maximum acceptable
    /// slope of the window's cumulative histogram. `None` or `Some(0.0)`
    /// leaves the contrast unlimited. With a limit `s`, each window bin is
    /// clipped at `s * area / bins` and the clipped excess is
    /// redistributed evenly across all bins before the cumulative sum, so
    /// larger values allow more contrast. A constant window maps to the
    /// depth maximum, as with [`Raster::hist_equal`].
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::ZeroWindow`] if either window dimension is
    /// zero, or [`HistogramError::InvalidMaxSlope`] if `max_slope` is
    /// negative or not finite.
    pub fn try_hist_local(
        &self,
        width: u32,
        height: u32,
        max_slope: Option<f64>,
    ) -> Result<Raster, HistogramError> {
        if width == 0 || height == 0 {
            return Err(HistogramError::ZeroWindow);
        }
        let slope = max_slope.unwrap_or(0.0);
        if !slope.is_finite() || slope < 0.0 {
            return Err(HistogramError::InvalidMaxSlope { max_slope: slope });
        }
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bins = bins_for(kind);
        let iw = self.width() as i64;
        let ih = self.height() as i64;
        let ww = width as i64;
        let wh = height as i64;
        let area = (ww * wh) as f64;
        let range = (bins - 1) as f64;
        let data = self.data();
        let mut out = vec![0u8; data.len()];

        // Sample of band `b` at border-replicated (clamped) coordinates.
        let sample = |x: i64, y: i64, b: usize| -> usize {
            let cx = x.clamp(0, iw - 1) as usize;
            let cy = y.clamp(0, ih - 1) as usize;
            read_flat(data, kind, (cy * iw as usize + cx) * bands + b) as usize
        };
        let limit = slope * area / bins as f64;

        for b in 0..bands {
            let mut hist = vec![0u32; bins];
            for y in 0..ih {
                // Build the window histogram at x = 0, then slide it right
                // one column at a time.
                hist.fill(0);
                for dx in 0..ww {
                    for dy in 0..wh {
                        hist[sample(dx - ww / 2, y + dy - wh / 2, b)] += 1;
                    }
                }
                for x in 0..iw {
                    if x > 0 {
                        for dy in 0..wh {
                            let yy = y + dy - wh / 2;
                            hist[sample(x - 1 - ww / 2, yy, b)] -= 1;
                            hist[sample(x + ww - 1 - ww / 2, yy, b)] += 1;
                        }
                    }
                    let v = sample(x, y, b);
                    let cdf = if slope > 0.0 {
                        let mut below = 0.0;
                        let mut clipped_total = 0.0;
                        for (k, &c) in hist.iter().enumerate() {
                            let clipped = (c as f64).min(limit);
                            clipped_total += clipped;
                            if k <= v {
                                below += clipped;
                            }
                        }
                        let excess = area - clipped_total;
                        below + excess * (v as f64 + 1.0) / bins as f64
                    } else {
                        hist.iter().take(v + 1).map(|&c| c as f64).sum()
                    };
                    let mapped = (cdf * range / area).round().clamp(0.0, range) as u32;
                    write_flat(
                        &mut out,
                        kind,
                        (y as usize * iw as usize + x as usize) * bands + b,
                        mapped,
                    );
                }
            }
        }
        Ok(Raster::new(self.width(), self.height(), fmt, out)?)
    }

    /// Panicking form of [`Raster::try_hist_local`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_hist_local`].
    #[track_caller]
    pub fn hist_local(&self, width: u32, height: u32, max_slope: Option<f64>) -> Raster {
        expect_hist("hist_local", self.try_hist_local(width, height, max_slope))
    }

    // -----------------------------------------------------------------------
    // LUT application
    // -----------------------------------------------------------------------

    /// Map every sample through a look-up table, like `vips_maplut`.
    ///
    /// `lut` must be histogram-shaped (`N`x1 or 1x`N`); entry `i` of a LUT
    /// band is the output for input value `i`, and input values past the
    /// end of the LUT read the last entry, matching libvips. Band counts
    /// combine as in libvips: a one-band image fans out to the LUT's band
    /// count, a one-band LUT applies to every image band, and equal counts
    /// map band to band. The output has the image's size, the combined
    /// band count, and the LUT's depth.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::NotAHistogram`] if `lut` is not
    /// histogram-shaped, [`HistogramError::LutBandMismatch`] if neither
    /// band-combination rule applies, or [`HistogramError::Raster`] if the
    /// result exceeds the allocation budget.
    pub fn try_maplut(&self, lut: &Raster) -> Result<Raster, HistogramError> {
        let n_lut = hist_len(lut)?;
        let image_bands = self.format().channels();
        let lut_bands = lut.format().channels();
        let out_bands = if image_bands == 1 {
            lut_bands
        } else if lut_bands == 1 || lut_bands == image_bands {
            image_bands
        } else {
            return Err(HistogramError::LutBandMismatch {
                image: image_bands,
                lut: lut_bands,
            });
        };
        let kind = self.format().kind();
        let lut_bpc = lut.format().kind();
        let out_fmt = format_for(out_bands, lut_bpc)?;
        let mut out = Raster::zeroed(self.width(), self.height(), out_fmt)?;
        let buf = out.data_mut();
        let data = self.data();
        let lut_data = lut.data();
        let n = self.width() as usize * self.height() as usize;
        for i in 0..n {
            for c in 0..out_bands {
                let src_band = if image_bands == 1 { 0 } else { c };
                let lut_band = if lut_bands == 1 { 0 } else { c };
                let v = read_flat(data, kind, i * image_bands + src_band) as usize;
                let entry = read_flat(lut_data, lut_bpc, v.min(n_lut - 1) * lut_bands + lut_band);
                write_flat(buf, lut_bpc, i * out_bands + c, entry);
            }
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_maplut`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_maplut`].
    #[track_caller]
    pub fn maplut(&self, lut: &Raster) -> Raster {
        expect_hist("maplut", self.try_maplut(lut))
    }

    /// Map an index image to scalar case values, like `vips_case` with
    /// constant cases.
    ///
    /// Every sample selects `cases[v]`, with indices past the end reading
    /// the last case, matching libvips. Case values are rounded to nearest
    /// and clamped into the unsigned sample range (negative cases write
    /// `0`; there is no signed sample depth). The output keeps the input's
    /// size and band count; its depth is 8-bit when every case fits in
    /// `0..=255` and 16-bit otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::EmptyCases`] if `cases` is empty.
    pub fn try_case(&self, cases: &[f64]) -> Result<Raster, HistogramError> {
        if cases.is_empty() {
            return Err(HistogramError::EmptyCases);
        }
        let values: Vec<u32> = cases
            .iter()
            .map(|&c| {
                if c.is_nan() {
                    0
                } else {
                    c.round().clamp(0.0, 65535.0) as u32
                }
            })
            .collect();
        let out_kind = if values.iter().all(|&v| v <= 255) {
            SampleKind::U8
        } else {
            SampleKind::U16
        };
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let out_fmt = format_for(bands, out_kind)?;
        let mut out = Raster::zeroed(self.width(), self.height(), out_fmt)?;
        let buf = out.data_mut();
        let data = self.data();
        let count = self.width() as usize * self.height() as usize * bands;
        for i in 0..count {
            let idx = (read_flat(data, kind, i) as usize).min(values.len() - 1);
            write_flat(buf, out_kind, i, values[idx]);
        }
        Ok(out)
    }

    /// Panicking form of [`Raster::try_case`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_case`].
    #[track_caller]
    pub fn case(&self, cases: &[f64]) -> Raster {
        expect_hist("case", self.try_case(cases))
    }

    // -----------------------------------------------------------------------
    // Percentile
    // -----------------------------------------------------------------------

    /// The threshold at or below which `percent` per cent of the samples
    /// lie, like `vips_percent`.
    ///
    /// All bands are pooled into one full-precision distribution; the
    /// result is the smallest sample value whose cumulative count reaches
    /// `percent / 100` of the total sample count, returned as `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`HistogramError::InvalidPercent`] if `percent` is outside
    /// `0..=100` or not finite.
    pub fn try_percent(&self, percent: f64) -> Result<f64, HistogramError> {
        if !percent.is_finite() || !(0.0..=100.0).contains(&percent) {
            return Err(HistogramError::InvalidPercent { percent });
        }
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bins = bins_for(kind);
        let count = self.width() as usize * self.height() as usize * bands;
        let data = self.data();
        let mut hist = vec![0u64; bins];
        for i in 0..count {
            hist[read_flat(data, kind, i) as usize] += 1;
        }
        let target = percent / 100.0 * count as f64;
        let mut cum = 0u64;
        for (v, &h) in hist.iter().enumerate() {
            cum += h;
            if cum as f64 >= target {
                return Ok(v as f64);
            }
        }
        Ok((bins - 1) as f64)
    }

    /// Panicking form of [`Raster::try_percent`], matching the ported-test
    /// call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`HistogramError`]; see [`Raster::try_percent`].
    #[track_caller]
    pub fn percent(&self, percent: f64) -> f64 {
        expect_hist("percent", self.try_percent(percent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * Tests that this module dispatches on sample kind and never on byte
     * width, by asserting that neither the byte-width accessor on
     * [`PixelFormat`] nor its width-keyed constructor survives in
     * `src/histogram.rs`.
     * Works by scanning the module's own source, compiled in with
     * `include_str!`, for the accessor's name; the needle is spelled in two
     * halves so this assertion is not itself a hit. A byte width is not a
     * sample kind: four bytes is `f32` today and would be `u32` under issue
     * #517, so a `match` keyed on the width silently takes a wrong arm for
     * any carrier added later instead of failing to compile (issue #607).
     * Input: `src/histogram.rs` -> Output: zero occurrences.
     */
    #[test]
    fn histogram_does_not_dispatch_on_byte_width() {
        const SRC: &str = include_str!("histogram.rs");
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
            SRC.contains(concat!("fn read_", "flat")),
            "positive control failed: the scan cannot see this module's source"
        );
        for needle in needles {
            assert_eq!(
                SRC.matches(needle).count(),
                0,
                "{needle} is back in src/histogram.rs; dispatch on PixelFormat::kind() \
                 and PixelFormat::with_kind() instead"
            );
        }
    }

    fn gray(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /**
     * Tests the carrier every counting histogram op writes, so the module
     * doc's claims about count depth have a check behind them and a wider
     * carrier lands as a red test at each op that must change.
     * Works by asserting the output `PixelFormat` of `hist_find`,
     * `hist_find_band`, `hist_find_indexed`, `hist_find_ndim` and
     * `hist_cum` on an 8-bit input, covering both the per-band and the
     * pooled shapes.
     * Measured on vips 8.18.6: `hist_find`, `hist_find_ndim` and
     * `hist_cum` (on an unsigned input) emit `VIPS_FORMAT_UINT`, while
     * `hist_find_indexed` emits `DOUBLE` for every input format and either
     * `combine` mode, which the module doc used to sweep into "32-bit
     * unsigned" (issue #759).
     * Input: a 4x4 Gray8 image -> Output: Gray16 from all five.
     */
    #[test]
    fn counting_ops_carry_16_bit_samples() {
        let im = gray(4, 4, (0u8..16).collect());
        assert_eq!(im.hist_find().format(), PixelFormat::Gray16);
        assert_eq!(im.hist_find_band(0).format(), PixelFormat::Gray16);
        assert_eq!(im.hist_find_indexed(&im).format(), PixelFormat::Gray16);
        assert_eq!(im.hist_find_ndim(Some(4)).format(), PixelFormat::Gray16);
        assert_eq!(im.hist_find().hist_cum().format(), PixelFormat::Gray16);
    }

    /**
     * Tests that `write_flat` saturates into the kind rather than
     * truncating, which is the contract its callers that do not pre-clamp
     * rely on.
     * Works by writing an over-ceiling value at each unsigned kind and
     * reading it back through `read_flat`. Mutation found this one too:
     * every op-level caller reaching `write_flat` today either goes
     * through `sat16` or is bounded by an index, so dropping this clamp
     * left all 60 histogram tests green even though a truncating write
     * would turn 65536 into 0.
     * Input: 300 at U8 and 70000 at U16 -> Output: 255 and 65535.
     */
    #[test]
    fn write_flat_saturates_into_the_kind() {
        let mut one = [0u8; 1];
        write_flat(&mut one, SampleKind::U8, 0, 300);
        assert_eq!(read_flat(&one, SampleKind::U8, 0), 255);

        let mut two = [0u8; 2];
        write_flat(&mut two, SampleKind::U16, 0, 70_000);
        assert_eq!(read_flat(&two, SampleKind::U16, 0), 65_535);

        // A value inside the kind is written through unchanged, so the
        // saturations above are a clamp and not a constant.
        write_flat(&mut one, SampleKind::U8, 0, 7);
        assert_eq!(read_flat(&one, SampleKind::U8, 0), 7);
        write_flat(&mut two, SampleKind::U16, 0, 4_242);
        assert_eq!(read_flat(&two, SampleKind::U16, 0), 4_242);
    }

    /**
     * Tests that the bin-index read folds a negative sample the way libvips
     * does, which is what the signed carriers of issue #516 will meet the
     * moment `hist_find` sees one.
     * Works by reading each stored bit pattern back as a bin index and
     * comparing against the measured libvips answer, with an in-range
     * positive value alongside so a clamp that answered zero for
     * everything cannot pass.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: a one-band `char` image
     * holding `[-128, -1, 0, 127]` histograms to a 256-wide result with
     * `bin 0 = 3` and `bin 127 = 1`, so every negative sample lands in bin
     * zero. That is the `VipsStatisticClass` input cast at work (the
     * per-op `format_table` casts `CHAR` to `UCHAR`, saturating), not a
     * signed bin table.
     * Input: I8 bit patterns 0x80, 0xFF, 0x00, 0x7F -> bins 0, 0, 0, 127.
     */
    #[test]
    fn read_flat_folds_a_negative_sample_into_bin_zero() {
        for (kind, bits, bin) in [
            (SampleKind::I8, 0x80u32, 0u32),
            (SampleKind::I8, 0xFF, 0),
            (SampleKind::I8, 0x00, 0),
            (SampleKind::I8, 0x7F, 127),
            (SampleKind::I16, 0x8000, 0),
            (SampleKind::I16, 0xFFFF, 0),
            (SampleKind::I16, 0x7FFF, 32_767),
        ] {
            let mut buf = vec![0u8; kind.bytes()];
            match kind.bytes() {
                1 => buf[0] = bits as u8,
                _ => buf.copy_from_slice(&(bits as u16).to_ne_bytes()),
            }
            assert_eq!(
                read_flat(&buf, kind, 0),
                bin,
                "{kind:?} read {bits:#x} into the wrong bin"
            );
        }
    }

    /**
     * Tests that the count write saturates into every sample kind's
     * ceiling, including the four kinds no `PixelFormat` carries yet.
     * Works by writing one value over the ceiling and one inside it per
     * kind and reading both back, so the clamp cannot pass as a constant.
     * Input: 300 into I8 -> 127; 5e9 into U32 -> 4294967295; 5e9 into I32
     * -> 2147483647.
     */
    #[test]
    fn write_flat_saturates_into_every_integer_kind() {
        // The stored bit pattern, read back without going through
        // `read_flat`, which folds a 32-bit sample into the 16-bit bin
        // table and so cannot see a count above 65535.
        fn stored(buf: &[u8]) -> u32 {
            match buf.len() {
                1 => u32::from(buf[0]),
                2 => u32::from(u16::from_ne_bytes([buf[0], buf[1]])),
                _ => u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]),
            }
        }
        for kind in [
            SampleKind::U8,
            SampleKind::I8,
            SampleKind::U16,
            SampleKind::I16,
            SampleKind::U32,
            SampleKind::I32,
        ] {
            let ceiling = kind.max_value().expect("an integer kind has a ceiling");
            let mut buf = vec![0u8; kind.bytes()];
            write_flat(&mut buf, kind, 0, u32::MAX);
            assert_eq!(
                stored(&buf),
                ceiling,
                "{kind:?} did not saturate at its ceiling"
            );
            write_flat(&mut buf, kind, 0, 5);
            assert_eq!(
                stored(&buf),
                5,
                "{kind:?} did not write an in-range count through"
            );
        }
        // The control on `stored` itself: on the two carried unsigned
        // kinds it agrees with `read_flat`, so it is reading the same
        // bytes the module does.
        let mut two = vec![0u8; 2];
        write_flat(&mut two, SampleKind::U16, 0, 4_242);
        assert_eq!(stored(&two), read_flat(&two, SampleKind::U16, 0));
    }

    /**
     * Tests that the bin-index read performs the same input cast a
     * `VipsStatisticClass` op performs, folding a 32-bit sample into the
     * 16-bit bin table rather than indexing past its end.
     * Works by storing samples either side of 65535 and asserting the
     * index, with an in-range value alongside so the fold cannot pass as a
     * constant, and by asserting the index never reaches `bins_for`.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: a `uint` image whose
     * largest sample is 70000 gives a **65536**-wide histogram, so the
     * sample was saturated into `ushort` before it was counted.
     * Input: U32 70000 -> 65535; U32 1000 -> 1000; I32 -7 -> 0.
     */
    /**
     * Tests that the fixed 256-row plot height belongs to `U8` alone and
     * not to every one-byte kind, which is the one claim in this change
     * that a raster cannot reach, since no `PixelFormat` carries `I8`.
     * Works by calling the height rule directly for every kind on the same
     * values, with a control that the data-driven group does answer from
     * the data.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: a histogram of
     * `[0, 5]` plots 256 rows high as `uchar` and 5 rows high as `char`,
     * `ushort`, `short`, `uint` and `int`.
     * Input: ([0, 5], U8) -> 256; ([0, 5], I8) -> 6 (libviprs's own
     * `max + 1`, see #802).
     */
    #[test]
    fn only_the_unsigned_byte_kind_plots_a_fixed_height() {
        let values = [0u32, 5];
        assert_eq!(plot_height(SampleKind::U8, &values), 256);
        for kind in [
            SampleKind::I8,
            SampleKind::U16,
            SampleKind::I16,
            SampleKind::U32,
            SampleKind::I32,
            SampleKind::F32,
        ] {
            assert_eq!(
                plot_height(kind, &values),
                5,
                "{kind:?} did not take its plot height from the data"
            );
        }
        // Control: the data-driven answer really does follow the data, so
        // the 5 above is not a second constant. An all-zero histogram
        // still gets one row, measured: a `ushort` `[0, 0, 0]` plots 1.
        assert_eq!(plot_height(SampleKind::I8, &[0, 40]), 40);
        assert_eq!(plot_height(SampleKind::I8, &[0, 0, 0]), 1);
        assert_eq!(plot_height(SampleKind::I8, &[]), 1);
    }

    #[test]
    fn read_flat_folds_a_32_bit_sample_into_the_16_bit_bin_table() {
        for (kind, stored, index) in [
            (SampleKind::U32, 70_000i64, 65_535u32),
            (SampleKind::U32, 1_000, 1_000),
            (SampleKind::U32, i64::from(u32::MAX), 65_535),
            (SampleKind::I32, -7, 0),
            (SampleKind::I32, 1_000, 1_000),
            (SampleKind::I32, i64::from(i32::MAX), 65_535),
        ] {
            let mut buf = vec![0u8; 4];
            buf.copy_from_slice(&(stored as i32).to_ne_bytes());
            let got = read_flat(&buf, kind, 0);
            assert_eq!(got, index, "{kind:?} folded {stored} into the wrong bin");
            assert!(
                (got as usize) < bins_for(kind),
                "{kind:?} indexed past the {} bins it declares",
                bins_for(kind)
            );
        }
    }

    /**
     * Tests `sat16` at the narrowing it owns, which the whole-op saturation
     * test above cannot reach.
     * Works by calling it directly with counts either side of the 16-bit
     * ceiling and at the top of `u64`. Mutation showed why this is needed:
     * two independent clamps produce the op's observable 65535, `sat16`
     * here and `write_flat`'s `v.min(65535)`, so breaking either one alone
     * leaves `hist_find_saturates_a_count_past_the_16_bit_ceiling` green.
     * They are not redundant, they cover different ranges: `sat16` guards
     * the `u64` to `u32` narrowing, which only a count above 4.29e9 can
     * cross and which needs a four-billion-pixel image to reach through the
     * op. Calling it directly costs nothing.
     * Input: 0, 65535, 65536, u64::MAX -> Output: 0, 65535, 65535, 65535.
     */
    #[test]
    fn sat16_clamps_at_the_ceiling_and_across_the_u32_narrowing() {
        assert_eq!(sat16(0), 0);
        assert_eq!(sat16(65_535), 65_535);
        assert_eq!(sat16(65_536), 65_535);
        // Above `u32::MAX`, where a bare `as u32` would wrap to 4294967295
        // and `write_flat`'s clamp would then have nothing to catch.
        assert_eq!(sat16(u64::MAX), 65_535);
        assert_eq!(sat16(u64::from(u32::MAX) + 1), 65_535);
    }

    /**
     * Tests that `hist_find` saturates a bin count past 65535 rather than
     * wrapping, the deviation the 16-bit carrier forces.
     * Works by histogramming a 256x256 single-valued image, whose one
     * populated bin holds 65536 samples, one past the ceiling. That is the
     * smallest square image that overflows it, and it is 64 KiB.
     * Measured on vips 8.18.6, whose `UINT` output holds the true value: a
     * 300x300 single-valued image gives `vips max` of `90000` on the
     * histogram. libviprs reports the `65535` asserted here (issue #759).
     * Input: 256x256 all-7 -> Output: bin 7 is 65535, not 65536.
     */
    #[test]
    fn hist_find_saturates_a_count_past_the_16_bit_ceiling() {
        let im = gray(256, 256, vec![7u8; 256 * 256]);
        let h = im.hist_find();
        assert_eq!(h.getpoint(7, 0), vec![65535.0]);
        // The bins either side stay at zero, so the saturation above is a
        // real count and not every bin reading full.
        assert_eq!(h.getpoint(6, 0), vec![0.0]);
        assert_eq!(h.getpoint(8, 0), vec![0.0]);
    }

    fn gray16(w: u32, h: u32, vals: &[u16]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Gray16, data).unwrap()
    }

    /// A 100x100 image whose left half is 0 and right half is 10, the
    /// fixture the ported `test_histfind` builds with `zeroed` + `insert`.
    fn half_zero_half_ten() -> Raster {
        let mut data = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                data[y * 100 + x] = 10;
            }
        }
        gray(100, 100, data)
    }

    /// A dark, low-contrast, textured 100x100 image: values 20..=49 in a
    /// deterministic pattern. Global and local equalisation both raise its
    /// mean and deviation.
    fn dark_textured() -> Raster {
        let mut data = vec![0u8; 100 * 100];
        for y in 0..100usize {
            for x in 0..100usize {
                data[y * 100 + x] = (20 + (x * 7 + y * 13) % 30) as u8;
            }
        }
        gray(100, 100, data)
    }

    // ---- hist_find ----

    /// hist_find counts values into a 256x1 16-bit histogram.
    /// Input: half-0 half-10 image. Output: 5000 at 0 and 10, 0 at 5.
    #[test]
    fn hist_find_counts_values() {
        let im = half_zero_half_ten();
        let hist = im.hist_find();
        assert_eq!(hist.width(), 256);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.format(), PixelFormat::Gray16);
        assert_eq!(hist.getpoint(0, 0), vec![5000.0]);
        assert_eq!(hist.getpoint(10, 0), vec![5000.0]);
        assert_eq!(hist.getpoint(5, 0), vec![0.0]);
    }

    /// hist_find histograms each band independently: a constant [1, 2, 3]
    /// RGB image counts all pixels at a different value per band.
    #[test]
    fn hist_find_multiband_per_band() {
        let data: Vec<u8> = std::iter::repeat_n([1u8, 2, 3], 50 * 40)
            .flatten()
            .collect();
        let im = Raster::new(50, 40, PixelFormat::Rgb8, data).unwrap();
        let hist = im.hist_find();
        assert_eq!(hist.width(), 256);
        assert_eq!(hist.format(), PixelFormat::Rgb16);
        assert_eq!(hist.getpoint(1, 0), vec![2000.0, 0.0, 0.0]);
        assert_eq!(hist.getpoint(2, 0), vec![0.0, 2000.0, 0.0]);
        assert_eq!(hist.getpoint(3, 0), vec![0.0, 0.0, 2000.0]);
    }

    /// 16-bit input histograms into 65536 bins, indexed by raw value.
    #[test]
    fn hist_find_16bit_width() {
        let im = gray16(3, 1, &[4096, 4096, 9]);
        let hist = im.hist_find();
        assert_eq!(hist.width(), 65536);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.getpoint(4096, 0), vec![2.0]);
        assert_eq!(hist.getpoint(9, 0), vec![1.0]);
        assert_eq!(hist.getpoint(0, 0), vec![0.0]);
    }

    /// Counts saturate at 65535 rather than wrapping: 90000 zero-valued
    /// pixels report 65535.
    #[test]
    fn hist_find_saturates_at_u16_max() {
        let im = Raster::zeroed(300, 300, PixelFormat::Gray8).unwrap();
        let hist = im.hist_find();
        assert_eq!(hist.getpoint(0, 0), vec![65535.0]);
    }

    // ---- hist_find_band ----

    /// hist_find_band histograms only the selected band, and rejects an
    /// out-of-range band with a typed error.
    #[test]
    fn hist_find_band_selects_band() {
        let data: Vec<u8> = std::iter::repeat_n([1u8, 2, 3], 100).flatten().collect();
        let im = Raster::new(10, 10, PixelFormat::Rgb8, data).unwrap();
        let hist = im.hist_find_band(1);
        assert_eq!(hist.format(), PixelFormat::Gray16);
        assert_eq!(hist.getpoint(2, 0), vec![100.0]);
        assert_eq!(hist.getpoint(1, 0), vec![0.0]);

        assert!(matches!(
            im.try_hist_find_band(3),
            Err(HistogramError::InvalidBand { band: 3, bands: 3 })
        ));
    }

    // ---- hist_find_indexed ----

    /// hist_find_indexed sums the input values per index bin: the ported
    /// fixture (half 0 / half 10, index = value // 10) sums to 0 and 50000.
    #[test]
    fn hist_find_indexed_sums() {
        let im = half_zero_half_ten();
        let index = im.floordiv_const(10.0);
        let hist = im.hist_find_indexed(&index);
        assert_eq!(hist.width(), 256);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.getpoint(0, 0), vec![0.0]);
        assert_eq!(hist.getpoint(1, 0), vec![50000.0]);
        assert_eq!(hist.getpoint(2, 0), vec![0.0]);
    }

    /// A 16-bit index image widens the output to 65536 elements, and a
    /// multiband input keeps one sum per band.
    #[test]
    fn hist_find_indexed_16bit_index_and_bands() {
        let data: Vec<u8> = std::iter::repeat_n([4u8, 6], 6).flatten().collect();
        let im = Raster::new(
            3,
            2,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        let index = gray16(3, 2, &[300, 300, 300, 0, 0, 0]);
        let hist = im.hist_find_indexed(&index);
        assert_eq!(hist.width(), 65536);
        assert_eq!(hist.format().channels(), 2);
        assert_eq!(hist.getpoint(300, 0), vec![12.0, 18.0]);
        assert_eq!(hist.getpoint(0, 0), vec![12.0, 18.0]);
    }

    /// hist_find_indexed rejects a size mismatch and a multiband index.
    #[test]
    fn hist_find_indexed_errors() {
        let im = gray(4, 4, vec![0; 16]);
        let small = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_hist_find_indexed(&small),
            Err(HistogramError::DimensionMismatch { .. })
        ));

        let rgb = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            im.try_hist_find_indexed(&rgb),
            Err(HistogramError::OneBandOnly { bands: 3, .. })
        ));
    }

    // ---- hist_find_ndim ----

    /// The ported test_histfind_ndim shape: a constant [1, 2, 3] RGB image
    /// puts all 10000 pixels in cell (0, 0) band 0 with the default 10
    /// bins, and in the single cell with bins = 1.
    #[test]
    fn hist_find_ndim_default_and_single_bin() {
        let data: Vec<u8> = std::iter::repeat_n([1u8, 2, 3], 100 * 100)
            .flatten()
            .collect();
        let im = Raster::new(100, 100, PixelFormat::Rgb8, data).unwrap();

        let hist = im.hist_find_ndim(None);
        assert_eq!(hist.width(), 10);
        assert_eq!(hist.height(), 10);
        assert_eq!(hist.format().channels(), 10);
        let px = hist.getpoint(0, 0);
        assert_eq!(px[0], 10000.0);
        assert_eq!(px[1], 0.0);
        assert_eq!(hist.getpoint(1, 0)[0], 0.0);

        let hist = im.hist_find_ndim(Some(1));
        assert_eq!(hist.width(), 1);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.format(), PixelFormat::Gray16);
        assert_eq!(hist.getpoint(0, 0), vec![10000.0]);
    }

    /// One- and two-band inputs collapse the missing dimensions, and bin
    /// placement follows v * bins / 256 (255 lands in the last bin).
    #[test]
    fn hist_find_ndim_lower_dimensions_and_bin_edges() {
        let im = gray(2, 1, vec![255, 0]);
        let hist = im.hist_find_ndim(None);
        assert_eq!((hist.width(), hist.height()), (10, 1));
        assert_eq!(hist.format(), PixelFormat::Gray16);
        assert_eq!(hist.getpoint(9, 0), vec![1.0]);
        assert_eq!(hist.getpoint(0, 0), vec![1.0]);

        let data = vec![0u8, 255, 255, 0];
        let im = Raster::new(
            2,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        let hist = im.hist_find_ndim(None);
        assert_eq!((hist.width(), hist.height()), (10, 10));
        assert_eq!(hist.format(), PixelFormat::Gray16);
        // Pixel [0, 255] -> column 0, row 9; pixel [255, 0] -> column 9, row 0.
        assert_eq!(hist.getpoint(0, 9), vec![1.0]);
        assert_eq!(hist.getpoint(9, 0), vec![1.0]);
        assert_eq!(hist.getpoint(0, 0), vec![0.0]);
    }

    /// hist_find_ndim rejects zero bins, bins past the value range, and
    /// inputs with more than three bands.
    #[test]
    fn hist_find_ndim_errors() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_hist_find_ndim(Some(0)),
            Err(HistogramError::InvalidBins { bins: 0, max: 256 })
        ));
        assert!(matches!(
            im.try_hist_find_ndim(Some(257)),
            Err(HistogramError::InvalidBins {
                bins: 257,
                max: 256
            })
        ));

        let im = Raster::zeroed(2, 2, PixelFormat::with_kind(5, SampleKind::U8).unwrap()).unwrap();
        assert!(matches!(
            im.try_hist_find_ndim(None),
            Err(HistogramError::TooManyDimensions { bands: 5 })
        ));
    }

    // ---- hist_cum ----

    /// The ported test_hist_cum body: the cumulative identity LUT ends at
    /// avg * 256, and the output is promoted to 16-bit.
    #[test]
    fn hist_cum_identity_total() {
        let im = Raster::identity();
        let total = im.avg() * 256.0;
        let cum = im.hist_cum();
        assert_eq!(cum.format(), PixelFormat::Gray16);
        assert_eq!((cum.width(), cum.height()), (256, 1));
        let px = cum.getpoint(255, 0);
        assert!(
            (px[0] - total).abs() < 0.001,
            "got {}, expected {total}",
            px[0]
        );
    }

    /// hist_cum runs per band and preserves a vertical (1xN) orientation.
    #[test]
    fn hist_cum_per_band_and_vertical() {
        let data = vec![1u8, 10, 2, 20, 3, 30];
        let im = Raster::new(
            1,
            3,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        let cum = im.hist_cum();
        assert_eq!((cum.width(), cum.height()), (1, 3));
        assert_eq!(cum.getpoint(0, 0), vec![1.0, 10.0]);
        assert_eq!(cum.getpoint(0, 1), vec![3.0, 30.0]);
        assert_eq!(cum.getpoint(0, 2), vec![6.0, 60.0]);
    }

    /// 16-bit cumulative sums saturate at 65535 rather than wrapping.
    #[test]
    fn hist_cum_saturates() {
        let cum = gray16(3, 1, &[60000, 60000, 60000]).hist_cum();
        assert_eq!(cum.getpoint(0, 0), vec![60000.0]);
        assert_eq!(cum.getpoint(1, 0), vec![65535.0]);
        assert_eq!(cum.getpoint(2, 0), vec![65535.0]);
    }

    /// hist_cum rejects a 2D image with a typed error.
    #[test]
    fn hist_cum_not_histogram_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_hist_cum(),
            Err(HistogramError::NotAHistogram {
                width: 2,
                height: 2
            })
        ));
    }

    // ---- hist_norm ----

    /// The ported test_hist_norm body: the identity LUT is already
    /// normalised, byte for byte.
    #[test]
    fn hist_norm_identity_is_identity() {
        let im = Raster::identity();
        let im2 = im.hist_norm();
        assert_eq!(im2.format(), PixelFormat::Gray8);
        assert_eq!(im.data(), im2.data());
    }

    /// hist_norm scales each band's maximum to the maximum index and
    /// leaves all-zero bands untouched.
    #[test]
    fn hist_norm_scales_bands_independently() {
        let data = vec![0u8, 0, 5, 0, 10, 0];
        let im = Raster::new(
            3,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        let normed = im.hist_norm();
        assert_eq!(normed.getpoint(0, 0), vec![0.0, 0.0]);
        assert_eq!(normed.getpoint(1, 0), vec![1.0, 0.0]);
        assert_eq!(normed.getpoint(2, 0), vec![2.0, 0.0]);
    }

    /// A histogram wider than 256 elements normalises into 16-bit output
    /// with its maximum at N - 1.
    #[test]
    fn hist_norm_wide_histogram_16bit() {
        let mut vals = vec![0u16; 65536];
        vals[100] = 40;
        let normed = gray16(65536, 1, &vals).hist_norm();
        assert_eq!(normed.format(), PixelFormat::Gray16);
        assert_eq!(normed.getpoint(100, 0), vec![65535.0]);
        assert_eq!(normed.getpoint(0, 0), vec![0.0]);
    }

    /// hist_norm rejects a 2D image with a typed error.
    #[test]
    fn hist_norm_not_histogram_error() {
        let im = gray(3, 2, vec![0; 6]);
        assert!(matches!(
            im.try_hist_norm(),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- hist_equal ----

    /// Equalising a dark, low-contrast image raises both the mean and the
    /// deviation while preserving size and format.
    #[test]
    fn hist_equal_raises_avg_and_deviate() {
        let im = dark_textured();
        let im2 = im.hist_equal();
        assert_eq!((im2.width(), im2.height()), (im.width(), im.height()));
        assert_eq!(im2.format(), im.format());
        assert!(im.avg() < im2.avg(), "avg {} -> {}", im.avg(), im2.avg());
        assert!(
            im.deviate() < im2.deviate(),
            "deviate {} -> {}",
            im.deviate(),
            im2.deviate()
        );
    }

    /// hist_equal maps each band with its own LUT: a band that is already
    /// full-range keeps its extremes while a narrow band spreads.
    #[test]
    fn hist_equal_per_band() {
        let mut data = Vec::with_capacity(256 * 2);
        for v in 0..=255u8 {
            data.push(v); // band 0: full ramp
            data.push(100 + (v % 10)); // band 1: narrow
        }
        let im = Raster::new(
            256,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        let eq = im.hist_equal();
        let band1 = eq.extract_band(1);
        assert!(
            band1.max() > 200.0,
            "narrow band should spread, max {}",
            band1.max()
        );
        let band0 = eq.extract_band(0);
        assert_eq!(band0.max(), 255.0);
    }

    /// A constant image equalises to the depth maximum: its cumulative
    /// distribution reaches 1 in the first occupied bin.
    #[test]
    fn hist_equal_constant_image() {
        let im = gray(4, 4, vec![77; 16]);
        let eq = im.hist_equal();
        assert_eq!(eq.min(), 255.0);
        assert_eq!(eq.max(), 255.0);
    }

    // ---- hist_ismonotonic ----

    /// The identity LUT is monotonic; a decreasing LUT is not; a constant
    /// LUT (non-decreasing) is; orientation does not matter.
    #[test]
    fn hist_ismonotonic_cases() {
        assert!(Raster::identity().hist_ismonotonic());
        assert!(!gray(3, 1, vec![5, 4, 6]).hist_ismonotonic());
        assert!(gray(3, 1, vec![7, 7, 7]).hist_ismonotonic());
        assert!(gray(1, 3, vec![1, 2, 3]).hist_ismonotonic());
    }

    /// Every band must be monotonic: one decreasing band fails the whole
    /// histogram.
    #[test]
    fn hist_ismonotonic_per_band() {
        let data = vec![0u8, 9, 1, 8, 2, 7];
        let im = Raster::new(
            3,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        assert!(!im.hist_ismonotonic());
    }

    /// hist_ismonotonic rejects a 2D image with a typed error.
    #[test]
    fn hist_ismonotonic_not_histogram_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_hist_ismonotonic(),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- hist_local ----

    /// Local equalisation raises the mean and deviation of a dark textured
    /// image and preserves size and format.
    #[test]
    fn hist_local_increases_contrast() {
        let im = dark_textured();
        let im2 = im.hist_local(10, 10, None);
        assert_eq!((im2.width(), im2.height()), (im.width(), im.height()));
        assert_eq!(im2.format(), im.format());
        assert!(im.avg() < im2.avg(), "avg {} -> {}", im.avg(), im2.avg());
        assert!(
            im.deviate() < im2.deviate(),
            "deviate {} -> {}",
            im.deviate(),
            im2.deviate()
        );
    }

    /// The CLAHE contrast limit reduces the deviation relative to
    /// unlimited local equalisation.
    #[test]
    fn hist_local_max_slope_limits_contrast() {
        let im = dark_textured();
        let unlimited = im.hist_local(10, 10, None);
        let clamped = im.hist_local(10, 10, Some(3.0));
        assert!(
            clamped.deviate() < unlimited.deviate(),
            "clamped {} vs unlimited {}",
            clamped.deviate(),
            unlimited.deviate()
        );
    }

    /// A constant image maps to the depth maximum, like hist_equal.
    #[test]
    fn hist_local_constant_image() {
        let im = gray(8, 8, vec![42; 64]);
        let out = im.hist_local(3, 3, None);
        assert_eq!(out.min(), 255.0);
    }

    /// The sliding window matches a brute-force per-pixel histogram,
    /// including the border-replicated edges.
    #[test]
    fn hist_local_matches_brute_force() {
        let im = dark_textured();
        let fast = im.hist_local(5, 3, None);
        let (w, h) = (im.width() as i64, im.height() as i64);
        let data = im.data();
        for &(px, py) in &[(0i64, 0i64), (3, 7), (99, 99), (50, 0), (0, 42)] {
            let v = data[(py * w + px) as usize];
            let mut cdf = 0u32;
            for dy in 0..3i64 {
                for dx in 0..5i64 {
                    let sx = (px + dx - 2).clamp(0, w - 1);
                    let sy = (py + dy - 1).clamp(0, h - 1);
                    if data[(sy * w + sx) as usize] <= v {
                        cdf += 1;
                    }
                }
            }
            let expected = (cdf as f64 * 255.0 / 15.0).round();
            assert_eq!(
                fast.getpoint(px as u32, py as u32),
                vec![expected],
                "mismatch at ({px}, {py})"
            );
        }
    }

    /// 16-bit input equalises over 65536 bins and keeps its format.
    #[test]
    fn hist_local_16bit_smoke() {
        let im = gray16(4, 2, &[100, 200, 300, 400, 500, 600, 700, 800]);
        let out = im.hist_local(3, 3, None);
        assert_eq!(out.format(), PixelFormat::Gray16);
        // The largest value in its window always maps to the maximum.
        assert_eq!(out.getpoint(3, 1), vec![65535.0]);
    }

    /// hist_local rejects a zero window and a negative or non-finite
    /// max_slope with typed errors.
    #[test]
    fn hist_local_errors() {
        let im = gray(4, 4, vec![0; 16]);
        assert!(matches!(
            im.try_hist_local(0, 3, None),
            Err(HistogramError::ZeroWindow)
        ));
        assert!(matches!(
            im.try_hist_local(3, 0, None),
            Err(HistogramError::ZeroWindow)
        ));
        assert!(matches!(
            im.try_hist_local(3, 3, Some(-1.0)),
            Err(HistogramError::InvalidMaxSlope { .. })
        ));
        assert!(matches!(
            im.try_hist_local(3, 3, Some(f64::NAN)),
            Err(HistogramError::InvalidMaxSlope { .. })
        ));
    }

    // ---- hist_match ----

    /// The ported test_hist_match body: matching a histogram to itself is
    /// the identity, byte for byte.
    #[test]
    fn hist_match_self_is_identity() {
        let im = Raster::identity();
        let im2 = Raster::identity();
        let matched = im.hist_match(&im2);
        assert_eq!(matched.format(), PixelFormat::Gray8);
        assert_eq!(im.data(), matched.data());
    }

    /// Matching a low spike to a high spike maps the spike's index to the
    /// reference position.
    #[test]
    fn hist_match_shifts_toward_reference() {
        let mut in_hist = vec![0u8; 256];
        in_hist[10] = 100;
        let mut ref_hist = vec![0u8; 256];
        ref_hist[200] = 100;
        let lut = gray(256, 1, in_hist).hist_match(&gray(256, 1, ref_hist));
        assert_eq!(lut.getpoint(10, 0), vec![200.0]);
        assert_eq!(lut.getpoint(255, 0), vec![200.0]);
    }

    /// A reference wider than 256 elements produces a 16-bit LUT spanning
    /// the reference range.
    #[test]
    fn hist_match_wide_reference_16bit() {
        let mut ref_vals = vec![0u16; 65536];
        ref_vals[60000] = 5;
        let mut in_hist = vec![0u8; 256];
        in_hist[10] = 100;
        let lut = gray(256, 1, in_hist).hist_match(&gray16(65536, 1, &ref_vals));
        assert_eq!(lut.format(), PixelFormat::Gray16);
        assert_eq!((lut.width(), lut.height()), (256, 1));
        assert_eq!(lut.getpoint(10, 0), vec![60000.0]);
    }

    /// hist_match rejects mismatched band counts and non-histogram shapes.
    #[test]
    fn hist_match_errors() {
        let mono = gray(256, 1, vec![1; 256]);
        let rgb = Raster::zeroed(256, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            mono.try_hist_match(&rgb),
            Err(HistogramError::BandCountMismatch {
                expected: 1,
                got: 3
            })
        ));

        let square = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            square.try_hist_match(&mono),
            Err(HistogramError::NotAHistogram { .. })
        ));
        assert!(matches!(
            mono.try_hist_match(&square),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- hist_plot ----

    /// The ported test_hist_plot body: the identity LUT plots as a 256x256
    /// Gray8 image.
    #[test]
    fn hist_plot_identity_shape() {
        let plot = Raster::identity().hist_plot();
        assert_eq!(plot.width(), 256);
        assert_eq!(plot.height(), 256);
        assert_eq!(plot.format(), PixelFormat::Gray8);
    }

    /**
     * Tests the bar geometry and the plot height of a 16-bit histogram
     * against libvips rather than against libviprs's own previous answer
     * (issue #802).
     * Works by plotting the same three counts libvips was measured on and
     * asserting every pixel of the result, so a height that is right by
     * accident still fails on where the bars start.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: `vips hist_plot` of a
     * `ushort` `[2, 0, 3]` gives a 3x**3** image whose rows are
     * `[0, 0, 255]`, `[255, 0, 255]`, `[255, 0, 255]`. This test asserted
     * 3x4 before, which was libviprs's `max + 1`.
     * Input: `[2, 0, 3]` -> 3x3, bars growing from the bottom.
     */
    #[test]
    fn hist_plot_bar_geometry() {
        let plot = gray16(3, 1, &[2, 0, 3]).hist_plot();
        assert_eq!((plot.width(), plot.height()), (3, 3));
        assert_eq!(plot.format(), PixelFormat::Gray8);
        // Column 0: two white pixels at the bottom of three rows.
        assert_eq!(plot.getpoint(0, 0), vec![0.0]);
        assert_eq!(plot.getpoint(0, 1), vec![255.0]);
        assert_eq!(plot.getpoint(0, 2), vec![255.0]);
        // Column 1: empty.
        for y in 0..3 {
            assert_eq!(plot.getpoint(1, y), vec![0.0]);
        }
        // Column 2: the full height.
        for y in 0..3 {
            assert_eq!(plot.getpoint(2, y), vec![255.0]);
        }
    }

    /**
     * Tests the plot height of a non-8-bit histogram against the libvips
     * sweep, which is `max` and not `max + 1` (issue #802).
     * Works by plotting each measured case through the op and comparing
     * the height, with the 8-bit fixed height asserted alongside so the
     * change cannot have collapsed the two rules into one.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6, `ushort` input:
     * `[0, 1]` -> 1, `[1, 1]` -> 1, `[0, 0, 0]` -> 1, `[0, 5]` -> 5,
     * `[3, 9]` -> 9, `[100, 200]` -> 200, `[65535, 0]` -> 65535. A `uchar`
     * `[0, 5]` and a `uchar` `[255, 0]` both give 256.
     */
    #[test]
    fn hist_plot_height_is_the_largest_count() {
        for (counts, height) in [
            (vec![0u16, 1], 1u32),
            (vec![1, 1], 1),
            (vec![0, 0, 0], 1),
            (vec![0, 5], 5),
            (vec![3, 9], 9),
            (vec![100, 200], 200),
            (vec![65535, 0], 65535),
        ] {
            let n = u32::try_from(counts.len()).unwrap();
            let plot = gray16(n, 1, &counts).hist_plot();
            assert_eq!(
                plot.height(),
                height,
                "16-bit histogram {counts:?} plotted the wrong height"
            );
            assert_eq!(plot.width(), n);
        }
        // The 8-bit height stays fixed at 256 whatever the counts are,
        // which is the half of the old claim that did hold.
        for counts in [vec![0u8, 5], vec![255, 0]] {
            let n = u32::try_from(counts.len()).unwrap();
            assert_eq!(gray(n, 1, counts).hist_plot().height(), 256);
        }
    }

    /// hist_plot rejects multiband histograms and non-histogram shapes.
    #[test]
    fn hist_plot_errors() {
        let rgb = Raster::zeroed(256, 1, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            rgb.try_hist_plot(),
            Err(HistogramError::OneBandOnly { bands: 3, .. })
        ));
        let square = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            square.try_hist_plot(),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- hist_entropy ----

    /// A uniform 256-cell histogram has exactly 8 bits of entropy; a
    /// single spike has 0; an all-zero histogram reports 0.
    #[test]
    fn hist_entropy_uniform_spike_zero() {
        assert!((gray(256, 1, vec![1; 256]).hist_entropy() - 8.0).abs() < 1e-12);

        let mut spike = vec![0u8; 256];
        spike[7] = 200;
        assert_eq!(gray(256, 1, spike).hist_entropy(), 0.0);

        assert_eq!(gray(256, 1, vec![0; 256]).hist_entropy(), 0.0);
    }

    /// Bands pool into one distribution: two bands with one count each
    /// give 1 bit.
    #[test]
    fn hist_entropy_pools_bands() {
        let data = vec![1u8, 0, 0, 1];
        let im = Raster::new(
            2,
            1,
            PixelFormat::with_kind(2, SampleKind::U8).unwrap(),
            data,
        )
        .unwrap();
        assert!((im.hist_entropy() - 1.0).abs() < 1e-12);
    }

    /// hist_find composes with hist_entropy: the half-0 half-10 image has
    /// exactly two equally likely values, so 1 bit.
    #[test]
    fn hist_entropy_of_hist_find() {
        let ent = half_zero_half_ten().hist_find().hist_entropy();
        assert!((ent - 1.0).abs() < 1e-12, "got {ent}");
    }

    /// hist_entropy rejects a 2D image with a typed error.
    #[test]
    fn hist_entropy_not_histogram_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_hist_entropy(),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- maplut ----

    /// The ported test_hist_map body: mapping the identity LUT through
    /// itself is the identity, byte for byte.
    #[test]
    fn maplut_identity_roundtrip() {
        let im = Raster::identity();
        let im2 = im.maplut(&im);
        assert_eq!(im2.format(), PixelFormat::Gray8);
        assert_eq!(im.data(), im2.data());
    }

    /// An inverting LUT reverses the ramp.
    #[test]
    fn maplut_invert() {
        let invert: Vec<u8> = (0..=255u8).rev().collect();
        let lut = gray(256, 1, invert);
        let out = gray(3, 1, vec![0, 100, 255]).maplut(&lut);
        assert_eq!(out.getpoint(0, 0), vec![255.0]);
        assert_eq!(out.getpoint(1, 0), vec![155.0]);
        assert_eq!(out.getpoint(2, 0), vec![0.0]);
    }

    /// Band combination rules: a one-band image fans out through a
    /// three-band LUT, a one-band LUT applies to every image band, equal
    /// counts map band to band, and anything else is a typed error.
    #[test]
    fn maplut_band_combinations() {
        let lut3_data: Vec<u8> = (0..256u32)
            .flat_map(|v| [v as u8, v.saturating_mul(2).min(255) as u8, 255 - v as u8])
            .collect();
        let lut3 = Raster::new(256, 1, PixelFormat::Rgb8, lut3_data).unwrap();

        let mono = gray(2, 1, vec![10, 20]);
        let fanned = mono.maplut(&lut3);
        assert_eq!(fanned.format().channels(), 3);
        assert_eq!(fanned.getpoint(0, 0), vec![10.0, 20.0, 245.0]);

        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
        let lut1 = Raster::identity();
        assert_eq!(rgb.maplut(&lut1).getpoint(0, 0), vec![1.0, 2.0, 3.0]);
        assert_eq!(rgb.maplut(&lut3).getpoint(0, 0), vec![1.0, 4.0, 252.0]);

        let two = Raster::zeroed(1, 1, PixelFormat::with_kind(2, SampleKind::U8).unwrap()).unwrap();
        assert!(matches!(
            two.try_maplut(&lut3),
            Err(HistogramError::LutBandMismatch { image: 2, lut: 3 })
        ));
    }

    /// Input values past the end of the LUT read the last entry, and a
    /// 16-bit LUT sets the output depth.
    #[test]
    fn maplut_clamps_and_takes_lut_depth() {
        let lut = gray(4, 1, vec![9, 8, 7, 6]);
        let im = gray16(3, 1, &[0, 3, 500]);
        let out = im.maplut(&lut);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.getpoint(0, 0), vec![9.0]);
        assert_eq!(out.getpoint(1, 0), vec![6.0]);
        assert_eq!(out.getpoint(2, 0), vec![6.0]);

        let lut16 = gray16(256, 1, &{
            let mut v = [0u16; 256];
            v[5] = 3000;
            v
        });
        let out = gray(1, 1, vec![5]).maplut(&lut16);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![3000.0]);
    }

    /// maplut rejects a LUT that is not histogram-shaped.
    #[test]
    fn maplut_not_histogram_error() {
        let im = gray(2, 1, vec![0, 1]);
        let square = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_maplut(&square),
            Err(HistogramError::NotAHistogram { .. })
        ));
    }

    // ---- case ----

    /// The ported test_case body: switch classes map through scalar cases
    /// with exact averages, and indices past the end use the last case.
    #[test]
    fn case_switch_classes() {
        let x = Raster::grey(256, 256, true);

        let cond_lo = x.less_than_const(128.0);
        let cond_hi = x.more_eq_const(128.0);
        let index = Raster::switch(&[&cond_lo, &cond_hi]);
        let y = index.case(&[10.0, 20.0]);
        assert!((y.avg() - 15.0).abs() < 0.001, "got {}", y.avg());

        let c0 = x.less_than_const(64.0);
        let c1 = x.more_eq_const(64.0).bitand(&x.less_than_const(128.0));
        let c2 = x.more_eq_const(128.0).bitand(&x.less_than_const(192.0));
        let c3 = x.more_eq_const(192.0);
        let index = Raster::switch(&[&c0, &c1, &c2, &c3]);
        let y = index.case(&[10.0, 20.0, 30.0, 40.0]);
        assert!((y.avg() - 25.0).abs() < 0.001, "got {}", y.avg());

        let y = index.case(&[10.0, 20.0, 30.0]);
        assert!((y.avg() - 22.5).abs() < 0.001, "got {}", y.avg());
    }

    /// Case values above 255 promote the output to 16-bit; negative and
    /// NaN cases clamp to zero; rounding is to nearest.
    #[test]
    fn case_promotion_and_clamping() {
        let index = gray(3, 1, vec![0, 1, 2]);
        let out = index.case(&[300.0, -5.0, 19.6]);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![300.0]);
        assert_eq!(out.getpoint(1, 0), vec![0.0]);
        assert_eq!(out.getpoint(2, 0), vec![20.0]);

        let out = index.case(&[1.0, f64::NAN, 2.4]);
        assert_eq!(out.format(), PixelFormat::Gray8);
        assert_eq!(out.getpoint(1, 0), vec![0.0]);
        assert_eq!(out.getpoint(2, 0), vec![2.0]);
    }

    /// case keeps the input band count and rejects an empty case list.
    #[test]
    fn case_bands_and_empty_error() {
        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![0, 1, 5]).unwrap();
        let out = rgb.case(&[10.0, 20.0]);
        assert_eq!(out.format().channels(), 3);
        assert_eq!(out.getpoint(0, 0), vec![10.0, 20.0, 20.0]);

        assert!(matches!(rgb.try_case(&[]), Err(HistogramError::EmptyCases)));
    }

    // ---- percent ----

    /// On a uniform ramp the percentile thresholds are exact: the median
    /// of a 256-wide grey ramp is 127 and the 90th percentile is 230.
    #[test]
    fn percent_uniform_ramp() {
        let im = Raster::grey(256, 256, true);
        assert_eq!(im.percent(50.0), 127.0);
        assert_eq!(im.percent(90.0), 230.0);
        assert_eq!(im.percent(100.0), 255.0);
        assert_eq!(im.percent(0.0), 0.0);
    }

    /// The returned threshold captures the requested fraction: counting
    /// samples at or below it recovers the percentage.
    #[test]
    fn percent_threshold_captures_fraction() {
        let im = dark_textured();
        let pc = im.percent(90.0);
        let total = im.width() as f64 * im.height() as f64;
        let below = im.data().iter().filter(|&&b| (b as f64) <= pc).count() as f64;
        let captured = 100.0 * below / total;
        assert!((captured - 90.0).abs() < 4.0, "captured {captured}%");
    }

    /// percent rejects out-of-range and non-finite arguments.
    #[test]
    fn percent_invalid_error() {
        let im = gray(2, 2, vec![0; 4]);
        assert!(matches!(
            im.try_percent(-1.0),
            Err(HistogramError::InvalidPercent { .. })
        ));
        assert!(matches!(
            im.try_percent(100.5),
            Err(HistogramError::InvalidPercent { .. })
        ));
        assert!(matches!(
            im.try_percent(f64::NAN),
            Err(HistogramError::InvalidPercent { .. })
        ));
    }

    // ---- composition ----

    /// The classic pipeline hist_find -> hist_cum -> hist_norm -> maplut
    /// equals hist_equal for an 8-bit image.
    #[test]
    fn manual_pipeline_matches_hist_equal() {
        let im = dark_textured();
        let lut = im.hist_find().hist_cum().hist_norm();
        let manual = im.maplut(&lut);
        let direct = im.hist_equal();
        assert_eq!(manual.data(), direct.data());
    }

    /// An equalisation LUT is monotonic.
    #[test]
    fn equalisation_lut_is_monotonic() {
        let lut = dark_textured().hist_find().hist_cum().hist_norm();
        assert!(lut.hist_ismonotonic());
    }

    /// The histogram ops reject float rasters loudly instead of
    /// misreading their bytes as u16 pairs (histograms are defined over
    /// the unsigned sample ranges; cast to an unsigned format first).
    #[test]
    #[should_panic(expected = "do not support float rasters")]
    fn histogram_float_panics() {
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let im = Raster::zeroed(2, 2, f1).unwrap();
        let _ = im.hist_find();
    }
}
