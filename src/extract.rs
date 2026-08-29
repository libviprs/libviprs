//! Extract, crop, and geometry-placement operations ported from libvips.
//!
//! This module is the third batch of the libvips operation surface required
//! by the ported integration tests: it covers rectangle extraction and the
//! canvas-placement family. Each operation exists in two forms:
//!
//! * a fallible `try_*` method returning `Result<_, ExtractError>`, the
//!   primary implementation with typed errors for out-of-bounds rectangles,
//!   zero factors, and mismatched band counts; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`extract_area`, `crop`, `embed`, ...) exactly. These delegate to the
//!   `try_*` form and panic with the typed error's message, mirroring the
//!   "known-good input" contract of [`Raster::add`] and
//!   [`Raster::getpoint`].
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::extract_area`] | `vips_extract_area` | a rectangle copied out |
//! | [`Raster::crop`] | `vips_crop` | alias of `extract_area` |
//! | [`Raster::embed`] | `vips_embed` | image placed in a larger canvas |
//! | [`Raster::gravity`] | `vips_gravity` | embed at a compass position |
//! | [`Raster::replicate`] | `vips_replicate` | image tiled across a grid |
//! | [`Raster::insert`] | `vips_insert` | one image composited onto another |
//! | [`Raster::zoom`] | `vips_zoom` | integer pixel-replication upscale |
//! | [`Raster::subsample`] | `vips_subsample` | integer point-sample downscale |
//! | [`Raster::smartcrop`] | `vips_smartcrop` | crop to the most interesting area |
//!
//! # Semantics shared by every operation
//!
//! * **Formats.** All operations except `insert` preserve the input
//!   [`PixelFormat`] exactly, including the `Multi8` / `Multi16`
//!   intermediates. `insert` promotes like the band operations: the result
//!   takes the wider depth and the larger band count of its two inputs, a
//!   one-band input is replicated across the result bands, and promotion is
//!   numeric (a `200` stays `200`), matching [`crate::bands`].
//! * **Backgrounds.** `embed`, `gravity`, and the expanded area of `insert`
//!   fill new pixels with black (all-zero samples) unless an extend mode or
//!   background vector says otherwise. Background constants are truncated
//!   toward zero (matching libvips' `double`->integer cast) and clamped to
//!   the sample depth (`0..=255` or `0..=65535`). [`Extend::White`] is the
//!   one fill that does **not** come from the depth: its ink is a property of
//!   the [`Interpretation`] and of the mechanism vips paints it with, so it
//!   reads the tag rather than the depth ceiling (issue #667), and the variant
//!   doc on [`Extend::White`] carries the measured table. That table's float
//!   column belongs to the resamplers: `embed` and `gravity` refuse a float
//!   raster outright with [`ExtractError::FloatUnsupported`], so no `Extend`
//!   mode ever inks one here (issue #694).
//!   A background vector must have one entry (replicated across bands) or
//!   exactly one entry per band.
//! * **Clipping.** `embed` and `insert` accept placements partly or wholly
//!   outside the canvas and clip, exactly as libvips does.
//! * **Metadata.** Every operation carries its input's interpretation,
//!   resolution, orientation and attached fields onto its result (issue #690).
//!   The origin offset is the one field they disagree on: `extract_area` and
//!   `crop` stamp it to `(-left, -top)` and discard the source's, matching
//!   `vips_extract_area`, where the placement and tiling ops leave the
//!   source's alone. `insert` is a two-input op and takes two rules (issue
//!   #718): the header block comes from `main` alone, and the attached fields
//!   are the union of `main`'s and `sub`'s with `main` winning a name they
//!   share.
//!
//! # Smartcrop
//!
//! [`Raster::smartcrop`] follows libvips `smartcrop.c`. The `Low`, `Centre`,
//! `High`, and `All` strategies are pure geometry. `Entropy` repeatedly
//! slices the lower-entropy edge strip off the longer axis (aiming for eight
//! steps) where a strip's score is the Shannon entropy of its pooled
//! all-band histogram, matching `vips_smartcrop_entropy` +
//! `vips_hist_entropy`. When the image has an alpha band it is premultiplied
//! before analysis (unless the caller says it already is), and the final
//! crop is always taken from the original image, as in libvips.
//!
//! The `Attention` strategy is a faithful port of
//! `vips_smartcrop_attention`: shrink to 32x32 with the default `lanczos3`
//! [`Raster::resize`], Laplacian edge detection on the XYZ luminance band
//! scaled by 5, a skin-tone distance score and the LAB `a*` band masked to
//! bright pixels (`Y > 5`), the three maps summed, blurred by a Gaussian
//! sized from the crop target, and the maximum position mapped back to
//! input coordinates. Using the same lanczos3 shrink libvips uses (rather
//! than a box filter) lands the energy maximum on the libvips pixel, so the
//! attention coordinates match the real fixtures (`sample.jpg`: 199, 234).
//! The strategy is deterministic.

use crate::arithmetic::interpretation_max_alpha;
use crate::conversion::Interpretation;
use crate::pixel::{PixelFormat, SampleKind};
use crate::raster::{Raster, RasterError};
use crate::resample::{ReduceKernel, ResizeOptions};
use thiserror::Error;

/// Typed errors for the extract and placement operations in
/// [`crate::extract`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExtractError {
    /// A requested rectangle does not fit inside the image.
    #[error("area ({left},{top})+({width}x{height}) out of bounds for {image_w}x{image_h}")]
    AreaOutOfBounds {
        left: u32,
        top: u32,
        width: u32,
        height: u32,
        image_w: u32,
        image_h: u32,
    },
    /// A requested width or height is zero.
    #[error("area width and height must be greater than zero")]
    EmptyArea,
    /// An extract operation that reads and writes individual samples was
    /// given a float raster.
    ///
    /// `embed`, `gravity`, `insert` and `smartcrop`'s `Entropy` and
    /// `Attention` strategies copy samples through an unsigned 8/16-bit
    /// path, so a float carrier has nowhere to land; cast to an unsigned
    /// format first. They used to **panic** out of a `Result` signature
    /// instead (issue #694), which is the shape [`ArithmeticError`] already
    /// fixed for `recomb` and `stdif` in #631, and this mirrors it.
    ///
    /// The rest of this module takes a float raster unchanged, because it
    /// copies whole pixels byte-wise rather than reading samples:
    /// `extract_area`, `crop`, `replicate`, `zoom`, `subsample`, and
    /// `smartcrop`'s four pure-geometry strategies (`Centre`, `Low`, `High`,
    /// `All`).
    ///
    /// [`ArithmeticError`]: crate::arithmetic::ArithmeticError::FloatUnsupported
    #[error("{op} does not support float rasters yet; cast to an unsigned 8/16-bit format first")]
    FloatUnsupported {
        /// The operation that refused, e.g. `"embed"`.
        op: &'static str,
    },
    /// The raster carries a sample kind this operation has no
    /// implementation for.
    ///
    /// The sibling of [`ExtractError::FloatUnsupported`] for the carriers
    /// that are not float. `embed`, `gravity` and `insert` carry the
    /// unsigned 32-bit one of issue #517, so this reaches them only for
    /// the signed carriers of issue #516; `smartcrop`'s entropy and
    /// attention strategies also refuse `Uint32`, because both build a
    /// value-indexed table and 2^32 bins is not a table. Mirrors
    /// [`crate::mosaicing::MosaicError::UnsupportedSampleKind`].
    #[error("{op} does not support {kind:?} samples yet")]
    UnsupportedSampleKind {
        /// The operation that refused.
        op: &'static str,
        /// The sample kind it cannot read.
        kind: SampleKind,
    },
    /// A zoom, subsample, or replicate factor is zero.
    #[error("factor must be greater than zero")]
    ZeroFactor,
    /// A subsample factor exceeds the image size along its axis, so the
    /// result would be empty.
    #[error("subsample factor {xfac}x{yfac} exceeds image size {width}x{height}")]
    FactorExceedsImage {
        xfac: u32,
        yfac: u32,
        width: u32,
        height: u32,
    },
    /// The result dimensions would overflow `u32`.
    #[error("result size {width}x{height} exceeds u32::MAX")]
    SizeOverflow { width: u64, height: u64 },
    /// `insert` inputs whose band counts differ and where neither input has
    /// a single band (libvips `bandalike` rejects the same shapes).
    #[error(
        "band-count mismatch: images have {main} and {sub} bands; counts must match unless one is 1"
    )]
    BandCountMismatch { main: usize, sub: usize },
    /// A background vector length is neither 1 nor the band count.
    #[error("background has {got} values, expected 1 or {expected}")]
    BackgroundLengthMismatch { expected: usize, got: usize },
    /// A compass-direction name is not a libvips nickname.
    #[error("unknown compass direction {got:?}")]
    UnknownDirection { got: String },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// How to fill the new pixels created by [`Raster::embed`] and
/// [`Raster::gravity`] (libvips `VipsExtend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Extend {
    /// Fill with black (all-zero samples).
    Black,
    /// Replicate the nearest edge pixel outward.
    Copy,
    /// Tile the image.
    Repeat,
    /// Reflect the image at its edges (the edge pixel is duplicated, so a
    /// row `0 1 2` extends as `... 1 0 | 0 1 2 | 2 1 0 ...`).
    Mirror,
    /// Fill with white, which libvips takes from the image's
    /// [`Interpretation`] and never from its depth. `vips_embed` inks the
    /// border with `(int) vips_interpretation_max_alpha(in->Type)`
    /// (`libvips/conversion/embed.c:280`): 65535 for
    /// [`Interpretation::Rgb16`] / [`Interpretation::Grey16`], 1.0 for
    /// [`Interpretation::ScRgb`], 255 for everything else.
    ///
    /// What reaches the pixels then depends on how `vips_region_paint`
    /// (`libvips/iofuncs/region.c:909`) writes that `int`. A float carrier
    /// gets it per band as a float (`FILL_LINE(float, ...)`, `region.c:936`),
    /// so an scRGB float border is `1.0` and an RGB16 one `65535.0`. An
    /// integer carrier gets `memset((char *) q, value, wd)` (`region.c:922`),
    /// which keeps only the **low byte** of the ink and repeats that byte
    /// across every byte of the sample. On the ordinary tags that is
    /// invisible, since `0xff` memset over a `u16` is 65535 again, which is
    /// why a depth-derived ceiling served this long. On scRGB it is very
    /// visible: the ink is 1, so a `u8` raster tagged scRGB fills with 1 and a
    /// `u16` one with `0x0101` = **257**. That is the paint mechanism showing
    /// through rather than any kind of white, and it is ported as it stands,
    /// because 257 is what a comparison against the oracle has to expect and
    /// the other reading of the intent (clamp the ink into the carrier's
    /// range, giving 1) is not whiter, it is black.
    ///
    /// Measured on vips 8.18.6, `vips embed in.v out.v 1 1 10 10 --extend
    /// white`, reading the corner:
    ///
    /// ```text
    /// carrier  multiband  srgb   rgb16  grey16  scrgb
    /// uchar    255        255    255    255     1
    /// ushort   65535      65535  65535  65535   257
    /// float    255        255    65535  65535   1
    /// ```
    ///
    /// # Which operations that table describes
    ///
    /// [`Extend`] is shared, and the ink does not land the same way at both
    /// ends of it.
    ///
    /// [`Raster::embed`] and [`Raster::gravity`] paint it straight into the
    /// output, so the table above is exactly what they give. They do not carry
    /// the float row: both refuse a float carrier rather than paint it
    /// wrongly, with [`ExtractError::FloatUnsupported`] out of the `try_`
    /// forms. That used to be a **panic** out of a `Result` signature, which
    /// this doc's float row made easy to walk into (issue #694).
    ///
    /// The resamplers that read this mode for taps landing outside the input
    /// ([`Raster::affine`] and the interpolating forms in [`crate::resample`])
    /// match the table only on a raster **without** an alpha band. Once alpha
    /// is present `vips_affine` premultiplies into a **float** image before it
    /// paints the border, so vips runs `FILL_LINE(float, ...)`, the byte
    /// `memset` never happens, and its border comes out at the plain
    /// interpretation maximum instead (255 for sRGB, 1 for scRGB). libviprs
    /// paints the ink first and premultiplies after, so on an alpha raster it
    /// keeps the memset values; issue #692 tracks that reordering.
    White,
    /// Fill with the background colour passed alongside the extend mode
    /// (black when the background is `None`).
    Background,
}

/// A compass position for [`Raster::gravity`] (libvips
/// `VipsCompassDirection`).
///
/// The panicking [`Raster::gravity`] surface also accepts the libvips
/// nicknames as `&str` (`"centre"`, `"north"`, `"south-east"`, ...) via
/// `From<&str>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompassDirection {
    /// Centre of the canvas.
    Centre,
    /// Top edge, horizontally centred.
    North,
    /// Right edge, vertically centred.
    East,
    /// Bottom edge, horizontally centred.
    South,
    /// Left edge, vertically centred.
    West,
    /// Top-right corner.
    NorthEast,
    /// Bottom-right corner.
    SouthEast,
    /// Bottom-left corner.
    SouthWest,
    /// Top-left corner.
    NorthWest,
}

impl std::str::FromStr for CompassDirection {
    type Err = ExtractError;

    /// Parse a libvips compass nickname (`"centre"`, `"north"`,
    /// `"north-east"`, ...). `"center"` and the dash-less spellings are
    /// accepted too.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "centre" | "center" => Self::Centre,
            "north" => Self::North,
            "east" => Self::East,
            "south" => Self::South,
            "west" => Self::West,
            "north-east" | "northeast" => Self::NorthEast,
            "south-east" | "southeast" => Self::SouthEast,
            "south-west" | "southwest" => Self::SouthWest,
            "north-west" | "northwest" => Self::NorthWest,
            other => {
                return Err(ExtractError::UnknownDirection {
                    got: other.to_string(),
                });
            }
        })
    }
}

impl From<&str> for CompassDirection {
    /// Convert a libvips compass nickname, panicking on an unknown name.
    ///
    /// This exists for the panicking ported-test surface, which passes
    /// string literals (`im.gravity("centre", 3, 3)`); use
    /// [`str::parse`] for a fallible conversion.
    #[track_caller]
    fn from(s: &str) -> Self {
        match s.parse() {
            Ok(d) => d,
            Err(e) => panic!("gravity: {e}"),
        }
    }
}

/// The area-of-interest strategy for [`Raster::smartcrop`] (libvips
/// `VipsInteresting`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SmartcropInteresting {
    /// Crop the exact centre.
    Centre,
    /// Keep the strips with the highest pooled-histogram Shannon entropy.
    Entropy,
    /// Centre the crop on the saliency maximum (edges, skin tones,
    /// saturation); see the module docs for the fidelity note.
    Attention,
    /// Crop at the low coordinate corner (0, 0).
    Low,
    /// Crop at the high coordinate corner (bottom right).
    High,
    /// Ignore the requested size and return the whole image.
    All,
}

// ---------------------------------------------------------------------------
// Sample-level helpers
// ---------------------------------------------------------------------------

/// Read the flat `i`-th sample as `i64` (native byte order, matching
/// [`crate::raster_ops`]).
///
/// `i64` and total over [`SampleKind`], which is issue #909 on this side:
/// the `u32` this used to return could not hold a negative, so `embed`,
/// `gravity` and `insert` refused the three signed carriers of issue #516
/// even though `vips embed --extend white` and `vips insert` both accept a
/// `char` raster and answer CHAR, measured on `/opt/homebrew/bin/vips`
/// 8.18.6. The shape is [`crate::convolution`]'s `put_sample`, total
/// since #748.
///
/// Keyed on the kind rather than on a byte width, so the three four-byte
/// kinds stay three different reads (issues #517, #607). The signed kinds
/// sign-extend, so this is the numeric read; `F32` truncates toward zero
/// the way `vips_cast` does, and is reachable only from a direct call
/// because these ops still refuse a float raster. (The pure byte-copy
/// paths like `extract_area` are depth-agnostic and handle every carrier
/// fine.)
#[inline]
fn read_s(data: &[u8], kind: SampleKind, i: usize) -> i64 {
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
/// A store and not a cast, the contract [`crate::convolution`]'s
/// `put_sample` carries: every caller copies a sample read at the same
/// kind or writes an ink [`resolve_ink`] has already clipped into the
/// carrier's range. The one deliberate narrow is [`Extend::White`], whose
/// ink is a **byte pattern** rather than a number: `white_ink` answers 255
/// for a one-byte carrier whatever its signedness, and `255 as i8` is the
/// `-1` vips fills a `char` border with.
/// Total over [`SampleKind`]; see [`read_s`].
#[inline]
fn write_s(data: &mut [u8], kind: SampleKind, i: usize, v: i64) {
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

/// The sample [`Extend::White`] paints: `vips_embed`'s interpretation-derived
/// ink, laid down by whichever paint mechanism the carrier selects.
///
/// `vips_embed` inks a white border with
/// `(int) vips_interpretation_max_alpha(in->Type)`
/// (`libvips/conversion/embed.c:280`), which is 65535 for RGB16 / GREY16, 1.0
/// for scRGB and 255 for everything else (`libvips/iofuncs/header.c:195`). So
/// the **interpretation** picks the ink and the depth never does... but what
/// reaches the pixels also depends on how `vips_region_paint`
/// (`libvips/iofuncs/region.c:909`) writes that `int`:
///
/// * a float carrier gets it per band as a float (`FILL_LINE(float, ...)`,
///   `region.c:936`), so an scRGB float border is `1.0` and an RGB16 one
///   `65535.0`; while
/// * an integer carrier gets `memset((char *) q, value, wd)` (`region.c:922`),
///   which keeps only the **low byte** of the ink and repeats that byte across
///   every byte of the sample.
///
/// The memset is invisible on the ordinary tags, which is why a depth-derived
/// ceiling has served this long: 255 is `0xff`, and `0xff` memset over a `u16`
/// is 65535, the depth maximum again. It is very visible on scRGB, where the
/// ink is 1 and a `u16` sample comes back `0x0101` = **257**. That is not
/// white in any sense, it is the paint mechanism showing through, and it is
/// ported rather than rounded off: 257 is what a comparison against the oracle
/// has to expect, and the alternative reading of the intent (clamp the ink into
/// the carrier's range, giving 1) is not any whiter, it is black.
///
/// Measured on vips 8.18.6, `vips embed in.v out.v 1 1 10 10 --extend white`
/// on a 4-band raster, reading the corner:
///
/// ```text
/// carrier  multiband  srgb   rgb16  grey16  scrgb
/// uchar    255        255    255    255     1
/// ushort   65535      65535  65535  65535   257
/// float    255        255    65535  65535   1
/// ```
///
/// `vips affine --extend white` gives the same values **on a raster without an
/// alpha band**, because it builds its resampling border with `vips_embed`
/// (`affine.c:534`); that is [`crate::resample`]'s side of the same ink. It
/// cannot once the raster carries alpha, because `vips_image_hasalpha()` sends
/// `vips_affine` through a premultiply into a **float** image before it paints
/// that border: `FILL_LINE(float, ...)` runs, the memset above never happens,
/// and the border lands on the plain interpretation maximum (255 for sRGB, 1
/// for scRGB) instead. libviprs paints the ink first and premultiplies after,
/// so it keeps the memset ink there; issue #692 tracks the reordering and
/// [`crate::resample`] pins the divergence.
//
// This is `pub(crate)`, so nothing public may link it: rustdoc renders a
// `[white_ink]` from a public doc as literal brackets with no anchor. The two
// public docs that used to do that, the module doc above and `Extend::White`,
// inline what a caller needs instead. Nothing in CI stops that coming back yet.
// `rustdoc::private_intra_doc_links` is warn-by-default and the doc gate denies
// only `broken_intra_doc_links`, and denying the other one is not a one-line
// change, because 33 sites across 13 files elsewhere in the tree trip it too.
// Issue #697 carries the gate and the sweep together; it is deliberately not
// here, since a doc-only conflict across those 13 files is the worst kind to
// resolve while the lanes holding them are still open.
#[inline]
pub(crate) fn white_ink(format: PixelFormat, interpretation: Interpretation) -> f64 {
    let ink = interpretation_max_alpha(interpretation);
    // `memset` takes the ink as an `int` and converts it to `unsigned char`,
    // so only the low byte survives, and it lands in every byte of the sample.
    let byte = u32::from(ink as i32 as u8);
    // Matched on the carrier rather than counted out over `bytes_per_channel()`,
    // so that adding a format is a compile error here rather than a silent ink
    // nobody checked against the oracle (the lever issue #633 landed). The
    // numeric fan-out this replaces answered for every depth, including ones
    // that do not exist: right by luck at 4 bytes, since vips measures `int` +
    // scRGB as `0x01010101`, and wrong at 8, where the `u32` shift drops the
    // high half. Neither would have failed to build.
    match format {
        // `FILL_LINE(float, ...)` writes the ink as a number, so a float
        // carrier keeps it whole.
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => ink,
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Multi8(_) => {
            f64::from(byte)
        }
        PixelFormat::Gray16
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Multi16(_) => f64::from((byte << 8) | byte),
        // `memset` fills every byte of the sample, so a four-byte integer
        // carrier gets the ink byte replicated four times. Measured:
        // `vips embed --extend white` on a one-band `uint` raster fills
        // 4294967295 (`0xFFFFFFFF`) and on an `int` one fills -1, the same
        // bytes read signed. The comment above about `int` + scRGB
        // measuring `0x01010101` is the low-ink end of the same rule.
        PixelFormat::Uint32(_) | PixelFormat::Int32(_) => {
            f64::from((byte << 24) | (byte << 16) | (byte << 8) | byte)
        }
        // The signed carriers replicate the same byte their unsigned twins
        // of the same width do, and the sign appears at the **store**
        // rather than here, because `memset` fills bytes and does not know
        // the type. Measured: `vips embed --extend white` fills -1 on
        // `char`, `short` and `int` alike, which is 0xFF, 0xFFFF and
        // 0xFFFFFFFF read signed, the same three patterns `uchar`,
        // `ushort` and `uint` fill as 255, 65535 and 4294967295
        // (issue #516).
        PixelFormat::Int8(_) => f64::from(byte),
        PixelFormat::Int16(_) => f64::from((byte << 8) | byte),
    }
}

/// Truncate (toward zero) and clamp an `f64` background constant into the
/// carrier's own range, matching the C `double`->integer cast libvips
/// performs when it casts a background colour to the image format.
///
/// The floor is the range's, not a literal `0.0`: that is the third hazard
/// class issue #909 names, and it is observable. Measured on
/// `/opt/homebrew/bin/vips` 8.18.6, `vips embed --extend background` on a
/// `char` raster fills **-50** for `--background -50`, **-128** for -200
/// and **127** for 200, so it clips at both ends and a `clamp(0.0, max)`
/// would have turned every negative background into black.
#[inline]
fn ink_value(v: f64, lo: i64, hi: i64) -> i64 {
    if v.is_nan() {
        0
    } else {
        v.trunc().clamp(lo as f64, hi as f64) as i64
    }
}

/// Resolve a background option into one clamped sample per band.
///
/// `None` is black, one value is replicated across the bands, and a
/// full-length vector is used per band; any other length is a typed error.
fn resolve_ink(
    bands: usize,
    range: (i64, i64),
    background: Option<&[f64]>,
) -> Result<Vec<i64>, ExtractError> {
    let (lo, hi) = range;
    match background {
        None => Ok(vec![0; bands]),
        Some(bg) if bg.len() == 1 => Ok(vec![ink_value(bg[0], lo, hi); bands]),
        Some(bg) if bg.len() == bands => Ok(bg.iter().map(|&v| ink_value(v, lo, hi)).collect()),
        Some(bg) => Err(ExtractError::BackgroundLengthMismatch {
            expected: bands,
            got: bg.len(),
        }),
    }
}

/// Reflect coordinate `i` into `0..n` with the edge pixel duplicated
/// (period `2n`).
#[inline]
fn reflect(i: i64, n: i64) -> i64 {
    let m = i.rem_euclid(2 * n);
    if m < n { m } else { 2 * n - 1 - m }
}

/// Copy `src` into `dst` at `(dx, dy)`, clipping to the destination and
/// replicating a one-band source across the destination bands. Depth
/// promotion is numeric.
fn blit(dst: &mut Raster, src: &Raster, dx: i64, dy: i64) {
    let dbands = dst.format().channels();
    let dkind = dst.format().kind();
    let (dw, dh) = (dst.width() as i64, dst.height() as i64);
    let dstride = dst.width() as usize;
    let sbands = src.format().channels();
    let skind = src.format().kind();
    let sstride = src.width() as usize;
    let sdata = src.data();
    let ddata = dst.data_mut();
    for sy in 0..src.height() as i64 {
        let oy = sy + dy;
        if oy < 0 || oy >= dh {
            continue;
        }
        for sx in 0..src.width() as i64 {
            let ox = sx + dx;
            if ox < 0 || ox >= dw {
                continue;
            }
            let si = (sy as usize * sstride + sx as usize) * sbands;
            let di = (oy as usize * dstride + ox as usize) * dbands;
            for c in 0..dbands {
                let sc = if sbands == 1 { 0 } else { c };
                write_s(ddata, dkind, di + c, read_s(sdata, skind, si + sc));
            }
        }
    }
}

/// Fill every pixel of `dst` with the per-band `ink` samples. Used to lay
/// down the `insert` background before the inputs are blitted on top.
fn fill_ink(dst: &mut Raster, ink: &[i64]) {
    let bands = dst.format().channels();
    let kind = dst.format().kind();
    let count = dst.width() as usize * dst.height() as usize;
    let data = dst.data_mut();
    for p in 0..count {
        let di = p * bands;
        for (c, &v) in ink.iter().enumerate() {
            write_s(data, kind, di + c, v);
        }
    }
}

/// `-v` as an `i32` for the crop-origin stamp, exact wherever `-v` fits an
/// `i32` and saturating at `i32::MIN` beyond it.
///
/// vips holds every dimension and both offsets in an `int`, so the question
/// does not arise there; here a `left` above `i32::MAX` is representable and
/// a bare `-(left as i32)` would wrap it back to a *positive* offset. A
/// raster that wide fits the 8 GiB construction budget at one byte per pixel,
/// so the branch is reachable rather than theoretical.
///
/// The obvious spelling, `-(v.min(i32::MAX as u32) as i32)`, is wrong at
/// exactly one input. It saturates at `i32::MIN + 1`, so `left = 2147483648`
/// comes back as `-2147483647` when `-2147483648` is both the right answer
/// and representable. Negating through `i64` and narrowing is exact
/// everywhere it can be, and the unit test below sweeps the four inputs
/// around the boundary. Proving `extract_area` *reaches* this needs a 2 GiB
/// raster; proving the arithmetic needs nothing at all, which is why the
/// first commit's "asserted only by reasoning" was the wrong call.
#[inline]
fn negated_origin(v: u32) -> i32 {
    i32::try_from(-i64::from(v)).unwrap_or(i32::MIN)
}

/// Refuse a float raster for an operation that reads samples through
/// [`read_s`] / [`write_s`] (issue #694).
///
/// **Where this sits relative to the other checks**, because the four entry
/// points do not agree and the difference is observable. `try_embed` and
/// `try_gravity` reject float *before* the zero-canvas check, so
/// `try_embed(.., 0, 0, ..)` on a float raster reports `FloatUnsupported`
/// where it used to report `EmptyArea`. `try_insert` rejects after the
/// band-count check and `try_smartcrop` after both geometry checks.
///
/// That is deliberate rather than an accident of where the line went. The
/// carrier is a property of the input the caller already holds, and the
/// geometry is a property of the arguments they just passed, so reporting the
/// carrier first tells them the thing they cannot fix by changing an argument.
/// Where an op has a *cheaper* structural check that also names an input
/// (`insert`'s band count), that one goes first, because it is the same class
/// of answer and it is already there.
///
/// The predicate is [`PixelFormat::is_float`], which covers **both** spellings
/// of a float layout, `RgbaF32` and `FloatF32(n)`. This crate has two on
/// purpose (#531), and a guard that is right for one and wrong for the other
/// is invisible to a suite that only builds one of them; the tests build both.
///
/// The split in this module is not about the operation, it is about how the
/// operation moves pixels. `extract_area`, `crop`, `replicate`, `zoom` and
/// `subsample` copy whole pixels byte-wise, so the sample depth never comes
/// up and a float carrier travels through untouched. `embed`, `gravity`,
/// `insert` and `smartcrop`'s two analysing strategies read and write
/// individual samples through a `u32`, and there is no float on that path.
///
/// So the guard sits at each of those four entry points rather than inside
/// `read_s`, which would cost a branch on every sample of every op to say
/// something that is already decided before the first one.
///
/// This mirrors [`reject_float_input`](crate::arithmetic) in `arithmetic.rs`,
/// which #631 added for `recomb` and `stdif`. Same problem, same shape.
#[inline]
fn reject_unreadable_kind(op: &'static str, r: &Raster) -> Result<(), ExtractError> {
    let kind = r.format().kind();
    match kind {
        SampleKind::U8
        | SampleKind::U16
        | SampleKind::U32
        | SampleKind::I8
        | SampleKind::I16
        | SampleKind::I32 => Ok(()),
        SampleKind::F32 => Err(ExtractError::FloatUnsupported { op }),
    }
}

/// The stricter guard, for the two smartcrop strategies that build a
/// value-indexed table.
///
/// `region_entropy` allocates one bin per sample value and `rgb_planes`
/// divides by a fixed 8- or 16-bit scale, so those two need a kind whose
/// values a table can be indexed by. That question is
/// [`SampleKind::hist_bins`], and it answers `None` for the 32-bit kinds
/// for the same reason it answers `None` for float: 2^32 bins is not a
/// table. Without this the `uint` carrier would index a 65536-entry
/// histogram with a sample of 90000 and panic out of a `Result`.
///
/// The signed one- and two-byte carriers pass, because
/// [`SampleKind::hist_bins`] answers by width and both scorers clip a
/// negative sample the way vips does rather than indexing with it. That is
/// measured rather than assumed: on `/opt/homebrew/bin/vips` 8.18.6,
/// `vips smartcrop` picks the **same** 16x16 crop from a `char` raster as
/// from its clipped `uchar` twin, on a fixture whose only texture once
/// negatives fold to zero sits in the positive half, under both
/// `--interesting entropy` and `--interesting attention`. Scored on the
/// raw signed values the noisy negative half would have won.
fn reject_untabulated_kind(op: &'static str, r: &Raster) -> Result<(), ExtractError> {
    reject_unreadable_kind(op, r)?;
    let kind = r.format().kind();
    if kind.hist_bins().is_none() {
        return Err(ExtractError::UnsupportedSampleKind { op, kind });
    }
    Ok(())
}

/// Unwrap an extract-op result for the panicking ported-test surface.
///
/// Most [`ExtractError`] variants do not name the failing op, so the panic
/// prefixes `"<op>: "` for context. [`ExtractError::FloatUnsupported`] is the
/// exception: it embeds the op in its own `Display`, so prefixing it here as
/// well doubles the name (`"embed: embed does not support float rasters yet
/// ..."`). That is issue #339's defect, which `expect_arith` in
/// `arithmetic.rs` already fixes for the same variant on that side; #694
/// mirrored the error shape and not this wrapper, so it arrived here too.
/// That one variant is emitted verbatim; every other variant keeps the prefix.
#[inline]
#[track_caller]
fn expect_extract(op: &str, r: Result<Raster, ExtractError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e @ ExtractError::FloatUnsupported { .. }) => panic!("{e}"),
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Unwrap a smartcrop result for the panicking ported-test surface.
///
/// Same doubling rule as [`expect_extract`].
#[inline]
#[track_caller]
fn expect_smartcrop(r: Result<(Raster, i32, i32), ExtractError>) -> (Raster, i32, i32) {
    match r {
        Ok(v) => v,
        Err(e @ ExtractError::FloatUnsupported { .. }) => panic!("{e}"),
        Err(e) => panic!("smartcrop: {e}"),
    }
}

impl Raster {
    /// Extract a rectangular region (libvips `extract_area`).
    ///
    /// The rectangle must lie entirely inside the image; libvips
    /// `extract_area` rejects out-of-bounds rectangles the same way.
    /// The format is preserved exactly.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::EmptyArea`] if `width` or `height` is zero,
    /// or [`ExtractError::AreaOutOfBounds`] if the rectangle extends past
    /// the image.
    pub fn try_extract_area(
        &self,
        left: u32,
        top: u32,
        width: u32,
        height: u32,
    ) -> Result<Raster, ExtractError> {
        if width == 0 || height == 0 {
            return Err(ExtractError::EmptyArea);
        }
        if left as u64 + width as u64 > self.width() as u64
            || top as u64 + height as u64 > self.height() as u64
        {
            return Err(ExtractError::AreaOutOfBounds {
                left,
                top,
                width,
                height,
                image_w: self.width(),
                image_h: self.height(),
            });
        }
        // The carry is `Raster::extract`'s now (#740), so this no longer does
        // it: one physical crop, one carry. Measured on vips 8.18.6:
        // `extract_area`, `crop`, `embed`, `gravity`, `replicate`, `zoom`,
        // `subsample` and `smartcrop` all hand the header block and the
        // attachments straight on, including through the ops that rescale the
        // pixel grid (`zoom` by 2x3 on `xres=5 yres=7` reports 5 and 7 back,
        // not 10 and 21). Issue #690.
        let mut out = self.extract(left, top, width, height)?;
        // `vips_extract_area` writes `Xoffset = -left` / `Yoffset = -top` and
        // throws the source's away (`conversion.c`, `vips_extract_area_build`),
        // where the placement and tiling ops leave the source's alone. It is
        // the only field of the header block that is not a verbatim carry, and
        // `smartcrop` inherits the rule by going through here.
        out.meta.xoffset = negated_origin(left);
        out.meta.yoffset = negated_origin(top);
        Ok(out)
    }

    /// Panicking form of [`Raster::try_extract_area`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_extract_area`].
    #[track_caller]
    pub fn extract_area(&self, left: u32, top: u32, width: u32, height: u32) -> Raster {
        expect_extract(
            "extract_area",
            self.try_extract_area(left, top, width, height),
        )
    }

    /// Crop a rectangle out of the image (libvips `crop`, an alias of
    /// `extract_area`).
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_extract_area`].
    #[track_caller]
    pub fn crop(&self, left: u32, top: u32, width: u32, height: u32) -> Raster {
        expect_extract("crop", self.try_extract_area(left, top, width, height))
    }

    /// Embed the image in a `width` x `height` canvas with its top-left
    /// corner at `(x, y)` (libvips `embed`).
    ///
    /// New pixels are filled per `extend`; `background` is only read by
    /// [`Extend::Background`] and must then hold one value (replicated) or
    /// one value per band, rounded and clamped to the depth. The placement
    /// may put the image partly or wholly outside the canvas; it is
    /// clipped. The format is preserved exactly.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::EmptyArea`] if `width` or `height` is zero,
    /// [`ExtractError::BackgroundLengthMismatch`] for a bad background
    /// vector, or [`ExtractError::FloatUnsupported`] on a float raster: this
    /// copies samples through an unsigned 8/16-bit path, so cast first
    /// (issue #694).
    pub fn try_embed(
        &self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        extend: Extend,
        background: Option<&[f64]>,
    ) -> Result<Raster, ExtractError> {
        reject_unreadable_kind("embed", self)?;
        self.embed_impl(x as i64, y as i64, width, height, extend, background)
    }

    /// Panicking form of [`Raster::try_embed`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_embed`].
    #[track_caller]
    pub fn embed(
        &self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        extend: Extend,
        background: Option<&[f64]>,
    ) -> Raster {
        expect_extract(
            "embed",
            self.try_embed(x, y, width, height, extend, background),
        )
    }

    /// Shared embed kernel with an `i64` origin so `gravity` cannot
    /// overflow the public `i32` surface on extreme canvas sizes.
    ///
    /// Both callers reject a float raster before they get here, because the
    /// refusal has to name the public operation and this kernel cannot see
    /// which one called (issue #694). The `debug_assert` below is what holds
    /// that rather than this sentence: the suite runs in debug, so a third
    /// caller added without a guard trips it in the first test that reaches it.
    fn embed_impl(
        &self,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
        extend: Extend,
        background: Option<&[f64]>,
    ) -> Result<Raster, ExtractError> {
        if width == 0 || height == 0 {
            return Err(ExtractError::EmptyArea);
        }
        debug_assert!(
            !self.format().is_float(),
            "embed_impl's callers must reject a float raster first; \
             see reject_unreadable_kind (issue #694)"
        );
        let fmt = self.format();
        let bands = fmt.channels();
        let kind = fmt.kind();
        let bpc = kind.bytes();
        let range = kind
            .range()
            .expect("an integer kind has a range; float is refused before here");
        let ink: Vec<i64> = match extend {
            // The white ink comes from the interpretation, never from the
            // range; see [`white_ink`]. It is a byte pattern rather than a
            // number, so it is *not* clipped: `white_ink` answers 255 for a
            // one-byte carrier of either signedness and `write_s` narrows
            // that to the `-1` vips fills a `char` border with.
            Extend::White => vec![white_ink(fmt, self.interpretation()) as i64; bands],
            Extend::Background => resolve_ink(bands, range, background)?,
            _ => vec![0; bands],
        };
        let (w, h) = (self.width() as i64, self.height() as i64);
        let sstride = self.width() as usize;
        let data = self.data();
        let mut out = vec![0u8; width as usize * height as usize * bands * bpc];
        for oy in 0..height as i64 {
            for ox in 0..width as i64 {
                let (sx, sy) = (ox - x, oy - y);
                let src = if sx >= 0 && sx < w && sy >= 0 && sy < h {
                    Some((sx, sy))
                } else {
                    match extend {
                        Extend::Copy => Some((sx.clamp(0, w - 1), sy.clamp(0, h - 1))),
                        Extend::Repeat => Some((sx.rem_euclid(w), sy.rem_euclid(h))),
                        Extend::Mirror => Some((reflect(sx, w), reflect(sy, h))),
                        _ => None,
                    }
                };
                let di = (oy as usize * width as usize + ox as usize) * bands;
                match src {
                    Some((sx, sy)) => {
                        let si = (sy as usize * sstride + sx as usize) * bands;
                        for c in 0..bands {
                            write_s(&mut out, kind, di + c, read_s(data, kind, si + c));
                        }
                    }
                    None => {
                        for (c, &v) in ink.iter().enumerate() {
                            write_s(&mut out, kind, di + c, v);
                        }
                    }
                }
            }
        }
        let mut out = Raster::new(width, height, fmt, out)?;
        out.carry_meta_from(self);
        Ok(out)
    }

    /// Place the image at a compass position inside a `width` x `height`
    /// canvas (libvips `gravity`).
    ///
    /// The placement offsets use libvips' truncating integer halving, and
    /// a canvas smaller than the image crops it. New pixels are filled per
    /// `extend` and `background`, exactly as in [`Raster::try_embed`].
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::EmptyArea`] if `width` or `height` is zero,
    /// [`ExtractError::BackgroundLengthMismatch`] for a bad background
    /// vector, or [`ExtractError::FloatUnsupported`] on a float raster: this
    /// copies samples through an unsigned 8/16-bit path, so cast first
    /// (issue #694).
    pub fn try_gravity(
        &self,
        direction: CompassDirection,
        width: u32,
        height: u32,
        extend: Extend,
        background: Option<&[f64]>,
    ) -> Result<Raster, ExtractError> {
        use CompassDirection as D;
        let cx = (width as i64 - self.width() as i64) / 2;
        let rx = width as i64 - self.width() as i64;
        let cy = (height as i64 - self.height() as i64) / 2;
        let by = height as i64 - self.height() as i64;
        let (x, y) = match direction {
            D::Centre => (cx, cy),
            D::North => (cx, 0),
            D::East => (rx, cy),
            D::South => (cx, by),
            D::West => (0, cy),
            D::NorthEast => (rx, 0),
            D::SouthEast => (rx, by),
            D::SouthWest => (0, by),
            D::NorthWest => (0, 0),
        };
        reject_unreadable_kind("gravity", self)?;
        self.embed_impl(x, y, width, height, extend, background)
    }

    /// Panicking form of [`Raster::try_gravity`] with libvips' defaults
    /// (black extend), matching the ported-test surface. Accepts either a
    /// [`CompassDirection`] or a libvips nickname string (`"centre"`,
    /// `"north-east"`, ...).
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`] and on an unknown direction name;
    /// see [`Raster::try_gravity`].
    #[track_caller]
    pub fn gravity<D: Into<CompassDirection>>(
        &self,
        direction: D,
        width: u32,
        height: u32,
    ) -> Raster {
        expect_extract(
            "gravity",
            self.try_gravity(direction.into(), width, height, Extend::Black, None),
        )
    }

    /// Tile the image `across` times horizontally and `down` times
    /// vertically (libvips `replicate`).
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::ZeroFactor`] if either factor is zero, or
    /// [`ExtractError::SizeOverflow`] if the result would not fit `u32`
    /// dimensions.
    pub fn try_replicate(&self, across: u32, down: u32) -> Result<Raster, ExtractError> {
        if across == 0 || down == 0 {
            return Err(ExtractError::ZeroFactor);
        }
        let ow = self.width() as u64 * across as u64;
        let oh = self.height() as u64 * down as u64;
        self.map_pixels(ow, oh, |x, y| {
            (
                (x % self.width() as u64) as usize,
                (y % self.height() as u64) as usize,
            )
        })
    }

    /// Panicking form of [`Raster::try_replicate`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_replicate`].
    #[track_caller]
    pub fn replicate(&self, across: u32, down: u32) -> Raster {
        expect_extract("replicate", self.try_replicate(across, down))
    }

    /// Zoom in by integer factors, replicating each pixel `xfac` times
    /// horizontally and `yfac` times vertically (libvips `zoom`).
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::ZeroFactor`] if either factor is zero, or
    /// [`ExtractError::SizeOverflow`] if the result would not fit `u32`
    /// dimensions.
    pub fn try_zoom(&self, xfac: u32, yfac: u32) -> Result<Raster, ExtractError> {
        if xfac == 0 || yfac == 0 {
            return Err(ExtractError::ZeroFactor);
        }
        let ow = self.width() as u64 * xfac as u64;
        let oh = self.height() as u64 * yfac as u64;
        self.map_pixels(ow, oh, |x, y| {
            ((x / xfac as u64) as usize, (y / yfac as u64) as usize)
        })
    }

    /// Panicking form of [`Raster::try_zoom`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_zoom`].
    #[track_caller]
    pub fn zoom(&self, xfac: u32, yfac: u32) -> Raster {
        expect_extract("zoom", self.try_zoom(xfac, yfac))
    }

    /// Shrink by taking the top-left pixel of every `xfac` x `yfac` cell
    /// (libvips `subsample`). The result is `width / xfac` by
    /// `height / yfac` with integer division, so trailing pixels that do
    /// not fill a cell are dropped.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::ZeroFactor`] if either factor is zero, or
    /// [`ExtractError::FactorExceedsImage`] if a factor is larger than the
    /// image along its axis (the result would be empty).
    pub fn try_subsample(&self, xfac: u32, yfac: u32) -> Result<Raster, ExtractError> {
        if xfac == 0 || yfac == 0 {
            return Err(ExtractError::ZeroFactor);
        }
        let ow = (self.width() / xfac) as u64;
        let oh = (self.height() / yfac) as u64;
        if ow == 0 || oh == 0 {
            return Err(ExtractError::FactorExceedsImage {
                xfac,
                yfac,
                width: self.width(),
                height: self.height(),
            });
        }
        self.map_pixels(ow, oh, |x, y| {
            ((x * xfac as u64) as usize, (y * yfac as u64) as usize)
        })
    }

    /// Panicking form of [`Raster::try_subsample`], matching the
    /// ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_subsample`].
    #[track_caller]
    pub fn subsample(&self, xfac: u32, yfac: u32) -> Raster {
        expect_extract("subsample", self.try_subsample(xfac, yfac))
    }

    /// Shared kernel for the pixel-mapping ops (`replicate`, `zoom`,
    /// `subsample`): build an `ow` x `oh` raster whose pixel `(x, y)` is
    /// the source pixel `map(x, y)`, copying whole pixels byte-wise.
    fn map_pixels(
        &self,
        ow: u64,
        oh: u64,
        map: impl Fn(u64, u64) -> (usize, usize),
    ) -> Result<Raster, ExtractError> {
        if ow > u32::MAX as u64 || oh > u32::MAX as u64 {
            return Err(ExtractError::SizeOverflow {
                width: ow,
                height: oh,
            });
        }
        let bpp = self.format().bytes_per_pixel();
        let sstride = self.width() as usize;
        let data = self.data();
        let size = (ow as usize)
            .checked_mul(oh as usize)
            .and_then(|n| n.checked_mul(bpp))
            .ok_or(ExtractError::SizeOverflow {
                width: ow,
                height: oh,
            })?;
        let mut out: Vec<u8> = Vec::new();
        out.try_reserve_exact(size)
            .map_err(|_| RasterError::AllocationFailed {
                width: ow as u32,
                height: oh as u32,
                bytes: size,
            })?;
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = map(x, y);
                let si = (sy * sstride + sx) * bpp;
                out.extend_from_slice(&data[si..si + bpp]);
            }
        }
        let mut out = Raster::new(ow as u32, oh as u32, self.format(), out)?;
        out.carry_meta_from(self);
        Ok(out)
    }

    /// Insert `sub` over `self` with its top-left corner at `(x, y)`
    /// (libvips `insert`).
    ///
    /// With `expand` false the result keeps `self`'s size and `sub` is
    /// clipped; with `expand` true the result is the bounding box of both
    /// rectangles and uncovered pixels take `background` (black when it is
    /// `None`, matching libvips `insert --background`). `sub` overwrites
    /// `self` where they overlap (no alpha blending), as in libvips. The
    /// result takes the wider depth and the larger band count; a one-band
    /// input is replicated across the result bands, and band counts that
    /// differ with neither being 1 are an error, matching libvips
    /// `bandalike`.
    ///
    /// `background` is only read where neither input covers a pixel (the
    /// gaps that appear when `expand` is true); a single value is
    /// replicated across the result bands and a full-length vector is used
    /// per band, exactly as [`Raster::try_embed`].
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::BandCountMismatch`] for incompatible band
    /// counts, [`ExtractError::BackgroundLengthMismatch`] for a background
    /// vector whose length is neither 1 nor the band count,
    /// [`ExtractError::SizeOverflow`] if the expanded canvas would not fit
    /// `u32` dimensions, or [`ExtractError::FloatUnsupported`] if **either**
    /// input is a float raster: the result takes the wider of the two depths,
    /// so a float `sub` reaches the sample copy exactly as a float `self` does
    /// (issue #694).
    pub fn try_insert(
        &self,
        sub: &Raster,
        x: i32,
        y: i32,
        expand: bool,
        background: Option<&[f64]>,
    ) -> Result<Raster, ExtractError> {
        let mb = self.format().channels();
        let sb = sub.format().channels();
        if mb != sb && mb != 1 && sb != 1 {
            return Err(ExtractError::BandCountMismatch { main: mb, sub: sb });
        }
        // Both inputs, because the result's depth is the wider of the two, so
        // a float `sub` reaches the sample copy just as a float `self` does.
        reject_unreadable_kind("insert", self)?;
        reject_unreadable_kind("insert", sub)?;
        let bands = mb.max(sb);
        // Through `SampleKind::promote`, not the wider byte width: a width
        // cannot order the carriers, and four bytes answers float for a
        // `uint` input (issues #517, #607).
        let kind = self.format().kind().promote(sub.format().kind());
        let range = kind
            .range()
            .expect("an integer kind has a range; float is refused above");
        // Resolve the fill up front so a bad background vector errors even
        // when `expand` leaves no visible gap.
        let ink = resolve_ink(bands, range, background)?;
        let fmt = PixelFormat::with_kind(bands, kind)
            .expect("band count is bounded by the two input formats, and the kind is carried");
        let (ox, oy, ow, oh) = if expand {
            let left = 0i64.min(x as i64);
            let top = 0i64.min(y as i64);
            let right = (self.width() as i64).max(x as i64 + sub.width() as i64);
            let bottom = (self.height() as i64).max(y as i64 + sub.height() as i64);
            (left, top, (right - left) as u64, (bottom - top) as u64)
        } else {
            (0i64, 0i64, self.width() as u64, self.height() as u64)
        };
        if ow > u32::MAX as u64 || oh > u32::MAX as u64 {
            return Err(ExtractError::SizeOverflow {
                width: ow,
                height: oh,
            });
        }
        let mut out = Raster::zeroed(ow as u32, oh as u32, fmt)?;
        // Pre-fill with the background so uncovered pixels keep it; `blit`
        // then overwrites the pixels the two inputs actually cover. A black
        // (all-zero) ink is already the zeroed state, so skip the fill.
        if ink.iter().any(|&v| v != 0) {
            fill_ink(&mut out, &ink);
        }
        blit(&mut out, self, -ox, -oy);
        blit(&mut out, sub, x as i64 - ox, y as i64 - oy);
        // Two rules, both measured on vips 8.18.6 (issue #718). The header
        // block comes from `main` alone: an scRGB `sub` under an sRGB `main`
        // reports sRGB, and the resolution, the offsets and the orientation
        // are all `main`'s. The attached fields are the union of both, with
        // `main` winning a name they share, so a profile only `sub` carries
        // still reaches the output. I ran it both ways round rather than
        // reading one cell.
        out.carry_meta_from(self);
        out.merge_fields_from(sub);
        Ok(out)
    }

    /// Panicking form of [`Raster::try_insert`], matching the ported-test
    /// surface. New pixels are filled black; use [`Raster::try_insert`] for
    /// a background colour.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_insert`].
    #[track_caller]
    pub fn insert(&self, sub: &Raster, x: i32, y: i32, expand: bool) -> Raster {
        expect_extract("insert", self.try_insert(sub, x, y, expand, None))
    }

    /// Crop to `width` x `height` around the most interesting part of the
    /// image (libvips `smartcrop`), also returning the attention centre.
    ///
    /// The returned coordinates are the saliency maximum in input
    /// coordinates for [`SmartcropInteresting::Attention`] and `(0, 0)`
    /// for every other strategy, matching the libvips `attention-x` /
    /// `attention-y` metadata. When the image has an alpha band and
    /// `premultiplied` is false, the analysis runs on a premultiplied
    /// copy so fully transparent content cannot attract the crop; the
    /// crop itself is always taken from `self`.
    /// [`SmartcropInteresting::All`] ignores the requested size and
    /// returns the whole image.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::EmptyArea`] if `width` or `height` is zero,
    /// or [`ExtractError::AreaOutOfBounds`] if the crop is larger than the
    /// image (libvips "bad extract area").
    ///
    /// [`SmartcropInteresting::Entropy`] and
    /// [`SmartcropInteresting::Attention`] also return
    /// [`ExtractError::FloatUnsupported`] on a float raster, because they read
    /// samples; the four pure-geometry strategies take one unchanged (issue
    /// #694).
    pub fn try_smartcrop(
        &self,
        width: u32,
        height: u32,
        interesting: SmartcropInteresting,
        premultiplied: bool,
    ) -> Result<(Raster, i32, i32), ExtractError> {
        if width == 0 || height == 0 {
            return Err(ExtractError::EmptyArea);
        }
        if width > self.width() || height > self.height() {
            return Err(ExtractError::AreaOutOfBounds {
                left: 0,
                top: 0,
                width,
                height,
                image_w: self.width(),
                image_h: self.height(),
            });
        }
        let (width, height) = match interesting {
            SmartcropInteresting::All => (self.width(), self.height()),
            _ => (width, height),
        };
        // Only the two strategies that read samples. The other four are pure
        // geometry and take a float raster unchanged, so a guard at this entry
        // point would break four working strategies to fix two (issue #694).
        // Ahead of the premultiply below, so a refused call does not pay for a
        // whole-image copy first.
        if matches!(
            interesting,
            SmartcropInteresting::Entropy | SmartcropInteresting::Attention
        ) {
            reject_untabulated_kind("smartcrop", self)?;
        }
        // libvips premultiplies before the strategy switch whenever an
        // alpha band is present; `has_alpha` guarantees the two bands
        // `premultiply` needs, so the panicking form cannot fire.
        let analysis_storage;
        let analysis: &Raster = if self.format().has_alpha() && !premultiplied {
            analysis_storage = self.premultiply();
            &analysis_storage
        } else {
            self
        };
        let (left, top, ax, ay) = match interesting {
            SmartcropInteresting::Low | SmartcropInteresting::All => (0, 0, 0, 0),
            SmartcropInteresting::Centre => (
                (self.width() - width) / 2,
                (self.height() - height) / 2,
                0,
                0,
            ),
            SmartcropInteresting::High => (self.width() - width, self.height() - height, 0, 0),
            SmartcropInteresting::Entropy => {
                let (l, t) = entropy_crop(analysis, width, height);
                (l, t, 0, 0)
            }
            SmartcropInteresting::Attention => attention_crop(analysis, width, height),
        };
        let out = self.try_extract_area(left, top, width, height)?;
        Ok((out, ax, ay))
    }

    /// Panicking smartcrop, matching the ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_smartcrop`].
    #[track_caller]
    pub fn smartcrop(&self, width: u32, height: u32, interesting: SmartcropInteresting) -> Raster {
        expect_smartcrop(self.try_smartcrop(width, height, interesting, false)).0
    }

    /// Panicking smartcrop returning `(crop, attention_x, attention_y)`,
    /// matching the ported-test surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_smartcrop`].
    #[track_caller]
    pub fn smartcrop_with_coords(
        &self,
        width: u32,
        height: u32,
        interesting: SmartcropInteresting,
    ) -> (Raster, i32, i32) {
        expect_smartcrop(self.try_smartcrop(width, height, interesting, false))
    }

    /// [`Raster::smartcrop_with_coords`] with an explicit `premultiplied`
    /// flag: pass true when the image's alpha is already premultiplied so
    /// the analysis skips its internal premultiply, matching the libvips
    /// option.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_smartcrop`].
    #[track_caller]
    pub fn smartcrop_with_coords_premultiplied(
        &self,
        width: u32,
        height: u32,
        interesting: SmartcropInteresting,
        premultiplied: bool,
    ) -> (Raster, i32, i32) {
        expect_smartcrop(self.try_smartcrop(width, height, interesting, premultiplied))
    }
}

// ---------------------------------------------------------------------------
// Smartcrop: entropy strategy
// ---------------------------------------------------------------------------

/// Shannon entropy of the pooled all-band histogram of a region, the score
/// `vips_smartcrop_score` computes via `hist_find` + `hist_entropy`.
fn region_entropy(im: &Raster, x: i64, y: i64, w: i64, h: i64) -> f64 {
    let bands = im.format().channels();
    let kind = im.format().kind();
    let bins = kind
        .hist_bins()
        .expect("reject_untabulated_kind refuses a kind with no bin count");
    let mut hist = vec![0u64; bins];
    let stride = im.width() as usize;
    let data = im.data();
    for yy in y..y + h {
        for xx in x..x + w {
            let base = (yy as usize * stride + xx as usize) * bands;
            for c in 0..bands {
                // `vips_hist_find` clips a sample into the bin table rather
                // than indexing with it, so every negative lands in bin
                // zero. Measured on 8.18.6: a `char` image holding
                // `[-100, -1, 0, 100]` histograms to `bin 0 = 3` and
                // `bin 100 = 1` (issue #909).
                let bin = read_s(data, kind, base + c).clamp(0, bins as i64 - 1);
                hist[bin as usize] += 1;
            }
        }
    }
    let total = (w * h) as f64 * bands as f64;
    let mut entropy = 0.0;
    for &n in &hist {
        if n > 0 {
            let p = n as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// The `vips_smartcrop_entropy` slicing loop: repeatedly drop the
/// lower-entropy edge strip until the target size is reached.
fn entropy_crop(im: &Raster, cw: u32, ch: u32) -> (u32, u32) {
    let mut left = 0i64;
    let mut top = 0i64;
    let mut width = im.width() as i64;
    let mut height = im.height() as i64;

    // Aim for eight steps along the axis that needs trimming most.
    let max_slice = (((width - cw as i64) as f64 / 8.0).ceil() as i64)
        .max(((height - ch as i64) as f64 / 8.0).ceil() as i64);

    while width > cw as i64 || height > ch as i64 {
        let slice_width = (width - cw as i64).min(max_slice);
        let slice_height = (height - ch as i64).min(max_slice);

        if slice_width > 0 {
            let left_score = region_entropy(im, left, top, slice_width, height);
            let right_score =
                region_entropy(im, left + width - slice_width, top, slice_width, height);
            width -= slice_width;
            if left_score < right_score {
                left += slice_width;
            }
        }

        if slice_height > 0 {
            let top_score = region_entropy(im, left, top, width, slice_height);
            let bottom_score =
                region_entropy(im, left, top + height - slice_height, width, slice_height);
            height -= slice_height;
            if top_score < bottom_score {
                top += slice_height;
            }
        }
    }

    (left as u32, top as u32)
}

// ---------------------------------------------------------------------------
// Smartcrop: attention strategy
// ---------------------------------------------------------------------------

/// The attention working size; libvips shrinks to 32 pixels per axis and
/// that shrink sets the precision of the crop placement.
const ATTENTION_SIZE: usize = 32;

/// Transpose a `w` x `h` plane into an `h` x `w` one.
fn transpose(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    out
}

/// Read the (r, g, b) planes of an already-shrunk analysis image on the
/// 8-bit `0..=255` scale (16-bit samples divide by 257). One- and two-band
/// images replicate band 0; three or more bands use the first three, the
/// analysis input being already premultiplied.
fn rgb_planes(im: &Raster) -> [Vec<f64>; 3] {
    let bands = im.format().channels();
    let kind = im.format().kind();
    let scale = if kind.bytes() == 1 { 1.0 } else { 257.0 };
    let (w, h) = (im.width() as usize, im.height() as usize);
    let data = im.data();
    let mut planes: [Vec<f64>; 3] = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
    for i in 0..w * h {
        for (c, plane) in planes.iter_mut().enumerate() {
            let sc = if bands >= 3 { c } else { 0 };
            // Clipped at zero, the bottom half of the `vips_cast` to
            // uchar the attention path scores through. The unsigned
            // carriers cannot reach the clamp, so this is the signed
            // carriers' arm and nothing else (issue #909).
            plane[i] = read_s(data, kind, i * bands + sc).max(0) as f64 / scale;
        }
    }
    planes
}

/// sRGB samples on the `0..=255` scale to XYZ with libvips' PCS scaling
/// (D65 white at Y = 100).
fn srgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let lin = |v: f64| {
        let c = v / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    (
        (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) * 100.0,
        (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) * 100.0,
        (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) * 100.0,
    )
}

/// The LAB `a*` component of an XYZ pixel (D65 reference white), the
/// "saturation" band the libvips attention pipeline scores.
fn lab_a_star(x: f64, y: f64) -> f64 {
    const XN: f64 = 95.047;
    const YN: f64 = 100.0;
    let f = |t: f64| {
        const D: f64 = 6.0 / 29.0;
        if t > D * D * D {
            t.cbrt()
        } else {
            t / (3.0 * D * D) + 4.0 / 29.0
        }
    };
    500.0 * (f(x / XN) - f(y / YN))
}

/// `|5 * laplacian|` of a `w` x `h` plane with copy-edge extension, the
/// libvips attention edge detector.
fn edge_map(y_plane: &[f64], w: usize, h: usize) -> Vec<f64> {
    let at = |x: i64, yy: i64| {
        let x = x.clamp(0, w as i64 - 1) as usize;
        let yy = yy.clamp(0, h as i64 - 1) as usize;
        y_plane[yy * w + x]
    };
    let mut out = vec![0.0; w * h];
    for yy in 0..h as i64 {
        for x in 0..w as i64 {
            let lap =
                4.0 * at(x, yy) - at(x, yy - 1) - at(x, yy + 1) - at(x - 1, yy) - at(x + 1, yy);
            out[yy as usize * w + x as usize] = (5.0 * lap).abs();
        }
    }
    out
}

/// Separable Gaussian blur of a `w` x `h` plane with copy-edge extension.
/// The mask half-width follows libvips `gaussblur` with its default minimum
/// amplitude of 0.2.
fn gaussblur_plane(plane: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let half = ((sigma * (-2.0 * 0.2f64.ln()).sqrt()).ceil() as i64).max(1);
    let weights: Vec<f64> = (-half..=half)
        .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = weights.iter().sum();
    // One horizontal pass over a `pw` x `ph` plane.
    let pass = |src: &[f64], pw: usize, ph: usize| {
        let mut out = vec![0.0; pw * ph];
        for y in 0..ph {
            for x in 0..pw as i64 {
                let mut acc = 0.0;
                for (k, &wt) in weights.iter().enumerate() {
                    let sx = (x + k as i64 - half).clamp(0, pw as i64 - 1) as usize;
                    acc += src[y * pw + sx] * wt;
                }
                out[y * pw + x as usize] = acc / norm;
            }
        }
        out
    };
    // Horizontal pass, then the same pass over the transposed plane.
    let horiz = pass(plane, w, h);
    let vert = pass(&transpose(&horiz, w, h), h, w);
    transpose(&vert, h, w)
}

/// The `vips_smartcrop_attention` pipeline; returns
/// `(left, top, attention_x, attention_y)` in input coordinates.
fn attention_crop(im: &Raster, cw: u32, ch: u32) -> (u32, u32, i32, i32) {
    // From smartcrop.js via libvips.
    const SKIN: [f64; 3] = [0.78, 0.57, 0.44];
    let n = ATTENTION_SIZE;
    let (w, h) = (im.width(), im.height());
    let hscale = n as f64 / w as f64;
    let vscale = n as f64 / h as f64;
    let sigma =
        (((cw as f64 * hscale).powi(2) + (ch as f64 * vscale).powi(2)).sqrt() / 10.0).max(1.0);

    // Drop the alpha band before the shrink, because libviprs' `resize`
    // premultiplies where `vips_resize` does not (issue #603).
    //
    // `vips_smartcrop_build` premultiplies once into float and hands the
    // result to `vips_resize`, which explicitly does NOT premultiply ("This
    // operation does not premultiply alpha. If your image has an alpha
    // channel, you should use premultiply on it first", `resize.c`). So in
    // vips the analysis image is still premultiplied when the argmax is taken,
    // and every transparent pixel is still at colour 0. libviprs' `resize`
    // brackets its own premultiply / un-premultiply pair around the resample
    // instead (a deliberate divergence, core #458), so handing it the
    // already-premultiplied analysis image un-premultiplies it on the way out
    // and the colour hiding behind transparent pixels comes back as bright
    // garbage, which then dominates the edge and skin scores.
    //
    // vips drops the alpha band immediately after the resize anyway
    // (`vips_colourspace` to XYZ, then `extract_band(0, "n", 3)`), and a
    // resample that does not premultiply is per-band independent, so dropping
    // it *before* the resize is exactly equivalent to what vips computes and
    // leaves libviprs' bracket with nothing to do. `has_alpha` is only ever
    // the four-band formats, so the band range always fits and the panicking
    // form cannot fire.
    let analysis_storage;
    let analysis: &Raster = if im.format().has_alpha() {
        analysis_storage = im.extract_bands(0, im.format().channels() as u32 - 1);
        &analysis_storage
    } else {
        im
    };

    // Shrink to the attention working size with the default lanczos3
    // `vips_resize`, matching libvips exactly. A box filter here shifts the
    // energy argmax, and thus the crop, off the libvips position.
    let small = analysis.resize_with(
        hscale,
        ResizeOptions {
            vscale: Some(vscale),
            kernel: ReduceKernel::Lanczos3,
            gap: 2.0,
        },
    );
    let (sw, sh) = (small.width() as usize, small.height() as usize);

    let [rp, gp, bp] = rgb_planes(&small);
    let mut total = vec![0.0; sw * sh];
    let mut y_plane = vec![0.0; sw * sh];
    let mut xyz = vec![(0.0, 0.0, 0.0); sw * sh];
    for i in 0..sw * sh {
        let (x, y, z) = srgb_to_xyz(rp[i], gp[i], bp[i]);
        y_plane[i] = y;
        xyz[i] = (x, y, z);
    }
    let edges = edge_map(&y_plane, sw, sh);
    for (i, t) in total.iter_mut().enumerate() {
        let (x, y, z) = xyz[i];
        // Skin score: distance of the normalised XYZ vector from the skin
        // point, rescaled to a 100..0 score.
        let mag = (x * x + y * y + z * z).sqrt();
        let (nx, ny, nz) = if mag == 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (x / mag, y / mag, z / mag)
        };
        let dist =
            ((nx - SKIN[0]).powi(2) + (ny - SKIN[1]).powi(2) + (nz - SKIN[2]).powi(2)).sqrt();
        let skin = 100.0 - 100.0 * dist;
        // Both colour scores are masked to bright pixels (Y > 5).
        let masked = if y > 5.0 {
            skin + lab_a_star(x, y)
        } else {
            0.0
        };
        *t = edges[i] + masked;
    }
    let blurred = gaussblur_plane(&total, sw, sh, sigma);

    // `vips_max` reports the first maximum in raster order; the strict
    // comparison keeps the same top-left-most winner.
    let mut best = f64::NEG_INFINITY;
    let (mut mx, mut my) = (0usize, 0usize);
    for (i, &v) in blurred.iter().enumerate() {
        if v > best {
            best = v;
            mx = i % sw;
            my = i / sw;
        }
    }

    // Transform back into image coordinates and centre the crop on the
    // maximum, clamped inside the image (libvips truncates both).
    let ax = (mx as f64 / hscale) as i64;
    let ay = (my as f64 / vscale) as i64;
    let left = (ax - (cw / 2) as i64).clamp(0, (w - cw) as i64);
    let top = (ay - (ch / 2) as i64).clamp(0, (h - ch) as i64);
    (left as u32, top as u32, ax as i32, ay as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageio::MetadataValue;
    use crate::pixel::ALL_KINDS;

    /// A one-band `Int8` raster from signed sample values.
    fn int8(w: u32, h: u32, vals: &[i8]) -> Raster {
        let data: Vec<u8> = vals.iter().map(|v| *v as u8).collect();
        let fmt = PixelFormat::Int8(core::num::NonZeroU16::new(1).unwrap());
        Raster::new(w, h, fmt, data).unwrap()
    }

    /// Every sample of an `Int8` raster, read back signed.
    fn i8s(r: &Raster) -> Vec<i8> {
        r.data().iter().map(|b| *b as i8).collect()
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

    /// A width x height Rgb8 raster from a byte vector.
    fn rgb(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// A width x height Rgba8 raster from a byte vector.
    fn rgba(w: u32, h: u32, data: Vec<u8>) -> Raster {
        Raster::new(w, h, PixelFormat::Rgba8, data).unwrap()
    }

    /// A 4x4 Gray8 ramp 0..16, row-major.
    fn ramp4() -> Raster {
        gray(4, 4, (0..16).collect())
    }

    /// `im` carrying an explicit interpretation, or left untagged so
    /// [`Interpretation::for_format`] answers for it.
    fn tagged(im: Raster, tag: Option<Interpretation>) -> Raster {
        match tag {
            Some(t) => im.copy().interpretation(t).build(),
            None => im,
        }
    }

    // -- extract_area / crop ------------------------------------------------

    #[test]
    fn extract_area_copies_the_rectangle() {
        let im = ramp4();
        let sub = im.extract_area(1, 1, 2, 2);
        assert_eq!(sub.width(), 2);
        assert_eq!(sub.height(), 2);
        assert_eq!(sub.format(), PixelFormat::Gray8);
        assert_eq!(sub.data(), &[5, 6, 9, 10]);
    }

    #[test]
    fn extract_area_rgb_preserves_bands() {
        #[rustfmt::skip]
        let im = rgb(2, 2, vec![
            1, 2, 3,    4, 5, 6,
            7, 8, 9,    10, 11, 12,
        ]);
        let sub = im.extract_area(1, 0, 1, 2);
        assert_eq!(sub.data(), &[4, 5, 6, 10, 11, 12]);
    }

    #[test]
    fn extract_area_16bit() {
        let im = gray16(2, 2, &[1000, 2000, 3000, 4000]);
        let sub = im.extract_area(1, 1, 1, 1);
        assert_eq!(sub.getpoint(0, 0), vec![4000.0]);
    }

    #[test]
    fn try_extract_area_out_of_bounds_is_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_extract_area(2, 0, 3, 2),
            Err(ExtractError::AreaOutOfBounds {
                left: 2,
                width: 3,
                ..
            })
        ));
        assert!(matches!(
            im.try_extract_area(0, 4, 1, 1),
            Err(ExtractError::AreaOutOfBounds { .. })
        ));
    }

    #[test]
    fn try_extract_area_zero_size_is_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_extract_area(0, 0, 0, 2),
            Err(ExtractError::EmptyArea)
        ));
    }

    #[test]
    fn crop_is_an_alias_of_extract_area() {
        let im = ramp4();
        assert_eq!(
            im.crop(1, 0, 2, 3).data(),
            im.extract_area(1, 0, 2, 3).data()
        );
    }

    // -- embed ---------------------------------------------------------------

    #[test]
    fn embed_black_places_and_fills() {
        let im = rgb(2, 2, vec![9; 12]);
        let out = im.embed(2, 2, 6, 6, Extend::Black, None);
        assert_eq!(out.width(), 6);
        assert_eq!(out.height(), 6);
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.getpoint(0, 0), vec![0.0, 0.0, 0.0]);
        assert_eq!(out.getpoint(5, 5), vec![0.0, 0.0, 0.0]);
        assert_eq!(out.getpoint(2, 2), vec![9.0, 9.0, 9.0]);
        assert_eq!(out.getpoint(3, 3), vec![9.0, 9.0, 9.0]);
    }

    /// Issue #667, the whole measured table in one place, including the float
    /// column [`Raster::embed`] itself cannot reach (`read_s` panics on a
    /// float carrier, so float embed is unimplemented rather than wrong) and
    /// the `uchar` + RGB16 cell, whose `255` is the `memset` keeping the low
    /// byte of `65535`. That cell is only visible here: the sample writers
    /// downstream truncate (`write_s`) or clamp (`SampleLayout::write`, over
    /// in [`crate::resample`]) a 65535 into an 8-bit sample and land on 255 by
    /// their own route, so an ink that skipped the truncation would paint the
    /// same pixel anyway.
    ///
    /// The `grey16` column is here for the same reason. Its float cell is
    /// 65535 where the depth rule this PR replaces gave 255, and it is the
    /// other cell that moved without anybody having to tag a raster `Rgb16`,
    /// so leaving it unasserted would let the whole `Grey16` arm regress
    /// silently.
    ///
    /// Measured on vips 8.18.6, `vips embed in.v out.v 1 1 10 10 --extend
    /// white`, reading the corner.
    #[test]
    fn white_ink_reproduces_the_measured_embed_table() {
        use Interpretation as I;
        let float3 = PixelFormat::FloatF32(core::num::NonZeroU16::new(3).unwrap());
        // (carrier, multiband, srgb, rgb16, grey16, scrgb)
        #[rustfmt::skip]
        let cases = [
            (PixelFormat::Rgb8,    255.0,   255.0,   255.0,   255.0,     1.0),
            (PixelFormat::Rgb16, 65535.0, 65535.0, 65535.0, 65535.0,   257.0),
            (float3,               255.0,   255.0, 65535.0, 65535.0,     1.0),
        ];
        for (fmt, multiband, srgb, rgb16, grey16, scrgb) in cases {
            for (tag, want) in [
                (I::Multiband, multiband),
                (I::Srgb, srgb),
                (I::Rgb16, rgb16),
                (I::Grey16, grey16),
                (I::ScRgb, scrgb),
            ] {
                assert_eq!(white_ink(fmt, tag), want, "{fmt:?} tagged {tag:?}");
            }
        }
    }

    /// Issue #667. `Extend::White` inks from the **interpretation**, and the
    /// depth only ever shows through the `memset` that paints an integer
    /// carrier; see [`white_ink`].
    ///
    /// Measured on vips 8.18.6, `vips embed in.v out.v 1 1 10 10 --extend
    /// white` reading the corner. The 8- and 16-bit columns are the two this
    /// module can carry; the float column is [`crate::resample`]'s, since
    /// [`read_s`] still refuses a float raster here:
    ///
    /// ```text
    /// carrier  multiband  srgb   rgb16  grey16  scrgb
    /// uchar    255        255    255    255     1
    /// ushort   65535      65535  65535  65535   257
    /// float    255        255    65535  65535   1
    /// ```
    ///
    /// The untagged rows are the regression pins: `Gray8` resolves to
    /// [`Interpretation::Bw`] and `Gray16` to [`Interpretation::Grey16`], and
    /// both land on the depth maximum the old ink happened to give.
    #[test]
    fn embed_white_inks_the_interpretation_through_the_paint_memset() {
        use Interpretation as I;
        // (tag, uchar corner, ushort corner)
        let cases = [
            (None, 255.0, 65535.0),
            (Some(I::Multiband), 255.0, 65535.0),
            (Some(I::Srgb), 255.0, 65535.0),
            (Some(I::Rgb16), 255.0, 65535.0),
            (Some(I::Grey16), 255.0, 65535.0),
            (Some(I::ScRgb), 1.0, 257.0),
        ];
        for (tag, want8, want16) in cases {
            let out = tagged(gray(1, 1, vec![7]), tag).embed(1, 1, 3, 3, Extend::White, None);
            assert_eq!(
                out.getpoint(0, 0),
                vec![want8],
                "8-bit carrier, tag {tag:?}"
            );
            assert_eq!(out.getpoint(1, 1), vec![7.0], "the image itself is copied");

            let out16 = tagged(gray16(1, 1, &[7]), tag).embed(1, 1, 3, 3, Extend::White, None);
            assert_eq!(
                out16.getpoint(0, 0),
                vec![want16],
                "16-bit carrier, tag {tag:?}"
            );
        }
    }

    #[test]
    fn embed_background_vector_per_band_and_replicated() {
        let im = rgb(1, 1, vec![1, 2, 3]);
        let out = im.embed(1, 1, 3, 3, Extend::Background, Some(&[7.0, 8.0, 9.0]));
        assert_eq!(out.getpoint(0, 0), vec![7.0, 8.0, 9.0]);
        assert_eq!(out.getpoint(1, 1), vec![1.0, 2.0, 3.0]);

        let one = im.embed(1, 1, 3, 3, Extend::Background, Some(&[42.0]));
        assert_eq!(one.getpoint(2, 2), vec![42.0, 42.0, 42.0]);

        // None means black, and constants clamp to the depth.
        let none = im.embed(1, 1, 3, 3, Extend::Background, None);
        assert_eq!(none.getpoint(0, 0), vec![0.0, 0.0, 0.0]);
        let clamped = im.embed(1, 1, 3, 3, Extend::Background, Some(&[300.0, -5.0, 1.4]));
        assert_eq!(clamped.getpoint(0, 0), vec![255.0, 0.0, 1.0]);
    }

    #[test]
    fn try_embed_background_length_mismatch_is_typed() {
        let im = rgb(1, 1, vec![1, 2, 3]);
        assert!(matches!(
            im.try_embed(0, 0, 2, 2, Extend::Background, Some(&[1.0, 2.0])),
            Err(ExtractError::BackgroundLengthMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn embed_copy_replicates_edges() {
        #[rustfmt::skip]
        let im = gray(2, 2, vec![
            1, 2,
            3, 4,
        ]);
        let out = im.embed(1, 1, 4, 4, Extend::Copy, None);
        // Corners take the nearest corner pixel; edges clamp per axis.
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(3, 0), vec![2.0]);
        assert_eq!(out.getpoint(0, 3), vec![3.0]);
        assert_eq!(out.getpoint(3, 3), vec![4.0]);
        assert_eq!(out.getpoint(2, 0), vec![2.0]);
        assert_eq!(out.getpoint(0, 2), vec![3.0]);
    }

    #[test]
    fn embed_repeat_tiles_the_image() {
        #[rustfmt::skip]
        let im = gray(2, 2, vec![
            1, 2,
            3, 4,
        ]);
        let out = im.embed(2, 2, 6, 6, Extend::Repeat, None);
        // The tile grid is aligned to the placement, so (0,0) is pixel (0,0).
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(1, 0), vec![2.0]);
        assert_eq!(out.getpoint(4, 4), vec![1.0]);
        assert_eq!(out.getpoint(5, 5), vec![4.0]);
    }

    #[test]
    fn embed_mirror_reflects_with_edge_duplication() {
        let im = gray(3, 1, vec![10, 20, 30]);
        let out = im.embed(3, 0, 9, 1, Extend::Mirror, None);
        // Row: 30 20 10 | 10 20 30 | 30 20 10.
        assert_eq!(out.data(), &[30, 20, 10, 10, 20, 30, 30, 20, 10]);
    }

    #[test]
    fn embed_clips_a_negative_origin() {
        #[rustfmt::skip]
        let im = gray(2, 2, vec![
            1, 2,
            3, 4,
        ]);
        let out = im.embed(-1, 0, 2, 2, Extend::Black, None);
        // Column 0 of the input is dropped; column 1 lands at x = 0.
        assert_eq!(out.getpoint(0, 0), vec![2.0]);
        assert_eq!(out.getpoint(1, 0), vec![0.0]);
        assert_eq!(out.getpoint(0, 1), vec![4.0]);
    }

    #[test]
    fn try_embed_zero_canvas_is_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_embed(0, 0, 0, 3, Extend::Black, None),
            Err(ExtractError::EmptyArea)
        ));
    }

    // -- gravity --------------------------------------------------------------

    #[test]
    fn gravity_places_at_all_nine_positions() {
        let im = gray(1, 1, vec![255]);
        let positions: &[(&str, u32, u32)] = &[
            ("centre", 1, 1),
            ("north", 1, 0),
            ("south", 1, 2),
            ("east", 2, 1),
            ("west", 0, 1),
            ("north-east", 2, 0),
            ("south-east", 2, 2),
            ("south-west", 0, 2),
            ("north-west", 0, 0),
        ];
        for &(direction, x, y) in positions {
            let out = im.gravity(direction, 3, 3);
            assert_eq!(out.width(), 3);
            assert_eq!(out.height(), 3);
            assert_eq!(
                out.getpoint(x, y),
                vec![255.0],
                "gravity({direction}) should place the pixel at ({x},{y})"
            );
            assert!(
                (out.avg() - 255.0 / 9.0).abs() < 1e-9,
                "gravity({direction}) fills the rest with black"
            );
        }
    }

    #[test]
    fn gravity_accepts_the_enum() {
        let im = gray(1, 1, vec![9]);
        let out = im.gravity(CompassDirection::SouthEast, 2, 2);
        assert_eq!(out.getpoint(1, 1), vec![9.0]);
        assert_eq!(out.getpoint(0, 0), vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "unknown compass direction")]
    fn gravity_unknown_direction_panics() {
        let im = gray(1, 1, vec![9]);
        let _ = im.gravity("upwards", 3, 3);
    }

    #[test]
    fn try_gravity_supports_extend_and_background() {
        let im = gray(1, 1, vec![9]);
        let out = im
            .try_gravity(
                CompassDirection::Centre,
                3,
                3,
                Extend::Background,
                Some(&[5.0]),
            )
            .unwrap();
        assert_eq!(out.getpoint(0, 0), vec![5.0]);
        assert_eq!(out.getpoint(1, 1), vec![9.0]);
    }

    #[test]
    fn gravity_crops_when_the_canvas_is_smaller() {
        let im = ramp4();
        let out = im.gravity(CompassDirection::Centre, 2, 2);
        // (4 - 2) / 2 = 1: the centre 2x2 block.
        assert_eq!(out.data(), im.extract_area(1, 1, 2, 2).data());
    }

    // -- replicate -------------------------------------------------------------

    #[test]
    fn replicate_tiles_across_and_down() {
        #[rustfmt::skip]
        let im = gray(2, 2, vec![
            1, 2,
            3, 4,
        ]);
        let out = im.replicate(3, 2);
        assert_eq!(out.width(), 6);
        assert_eq!(out.height(), 4);
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(4, 2), vec![1.0]);
        assert_eq!(out.getpoint(5, 3), vec![4.0]);
        assert_eq!(out.getpoint(3, 1), vec![4.0]);
    }

    #[test]
    fn try_replicate_zero_factor_is_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_replicate(0, 2),
            Err(ExtractError::ZeroFactor)
        ));
    }

    // -- insert -----------------------------------------------------------------

    #[test]
    fn insert_without_expand_keeps_the_main_size() {
        let main = gray(4, 4, vec![1; 16]);
        let sub = gray(2, 2, vec![9; 4]);
        let out = main.insert(&sub, 1, 1, false);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(1, 1), vec![9.0]);
        assert_eq!(out.getpoint(2, 2), vec![9.0]);
        assert_eq!(out.getpoint(3, 3), vec![1.0]);
    }

    #[test]
    fn insert_with_expand_grows_the_canvas() {
        let main = gray(4, 4, vec![1; 16]);
        let sub = gray(2, 2, vec![9; 4]);
        let out = main.insert(&sub, 3, 3, true);
        assert_eq!(out.width(), 5);
        assert_eq!(out.height(), 5);
        assert_eq!(out.getpoint(3, 3), vec![9.0]);
        assert_eq!(out.getpoint(4, 4), vec![9.0]);
        // The expanded strip outside both rectangles is black.
        assert_eq!(out.getpoint(4, 0), vec![0.0]);
        assert_eq!(out.getpoint(0, 4), vec![0.0]);
    }

    #[test]
    fn insert_expand_fills_gaps_with_the_background() {
        // Pinned against vips 8.18.4:
        //   vips insert main.v sub.v out.v 3 3 --expand --background 50
        // where main is 4x4 value 100 and sub is 2x2 value 200. The 5x5
        // result fills every pixel neither input covers with 50 and lets
        // sub win the (3,3) overlap.
        let main = gray(4, 4, vec![100; 16]);
        let sub = gray(2, 2, vec![200; 4]);
        let out = main.try_insert(&sub, 3, 3, true, Some(&[50.0])).unwrap();
        assert_eq!((out.width(), out.height()), (5, 5));
        #[rustfmt::skip]
        let expected = [
            100.0, 100.0, 100.0, 100.0,  50.0,
            100.0, 100.0, 100.0, 100.0,  50.0,
            100.0, 100.0, 100.0, 100.0,  50.0,
            100.0, 100.0, 100.0, 200.0, 200.0,
             50.0,  50.0,  50.0, 200.0, 200.0,
        ];
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(
                    out.getpoint(x, y),
                    vec![expected[(y * 5 + x) as usize]],
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn insert_fractional_background_truncates_toward_zero() {
        // Pinned against vips 8.18.4: a fractional --background is cast to the
        // integer image format by truncation toward zero, NOT rounded to
        // nearest.
        //   vips insert main.v sub.v out.v 3 3 --expand --background <bg>
        // with main/sub 1x1 gives gap pixels of 0 (bg=0.9), 1 (bg=1.5) and
        // 199 (bg=199.5) for uchar; ushort bg=1000.5 gives 1000.
        let main = gray(1, 1, vec![0]);
        let sub = gray(1, 1, vec![0]);
        for (bg, want) in [(0.9_f64, 0.0), (1.5, 1.0), (3.5, 3.0), (199.5, 199.0)] {
            let out = main.try_insert(&sub, 3, 3, true, Some(&[bg])).unwrap();
            assert_eq!(out.getpoint(1, 1), vec![want], "uchar bg={bg}");
        }

        let main16 = gray16(1, 1, &[0]);
        let sub16 = gray16(1, 1, &[0]);
        let out16 = main16
            .try_insert(&sub16, 3, 3, true, Some(&[1000.5]))
            .unwrap();
        assert_eq!(out16.getpoint(1, 1), vec![1000.0], "ushort bg=1000.5");
    }

    #[test]
    fn embed_fractional_background_truncates_toward_zero() {
        // The ink helper is shared with embed; vips embed also truncates
        // (bg=0.9->0, 1.5->1, 199.5->199), so pin it here too.
        let im = gray(1, 1, vec![9]);
        for (bg, want) in [(0.9_f64, 0.0), (1.5, 1.0), (199.5, 199.0)] {
            let out = im.embed(1, 1, 3, 3, Extend::Background, Some(&[bg]));
            assert_eq!(out.getpoint(0, 0), vec![want], "embed bg={bg}");
        }
    }

    #[test]
    fn insert_default_background_is_black() {
        // No background => black gaps, matching `vips insert ... --expand`
        // with no --background flag (unchanged legacy behaviour).
        let main = gray(4, 4, vec![100; 16]);
        let sub = gray(2, 2, vec![200; 4]);
        let out = main.try_insert(&sub, 3, 3, true, None).unwrap();
        assert_eq!(out.getpoint(4, 0), vec![0.0]);
        assert_eq!(out.getpoint(0, 4), vec![0.0]);
        // And the panicking wrapper stays black-only.
        let same = main.insert(&sub, 3, 3, true);
        assert_eq!(same.getpoint(4, 0), vec![0.0]);
    }

    #[test]
    fn insert_background_replicates_single_value_across_bands() {
        // vips replicates a scalar --background across every result band;
        // a mono main banded up by an rgb sub gets a 3-band background.
        let main = gray(4, 4, vec![100; 16]);
        let sub = rgb(
            2,
            2,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        );
        let out = main.try_insert(&sub, 3, 3, true, Some(&[77.0])).unwrap();
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.getpoint(4, 0), vec![77.0, 77.0, 77.0]);
        assert_eq!(out.getpoint(0, 4), vec![77.0, 77.0, 77.0]);
    }

    #[test]
    fn insert_background_full_vector_is_per_band() {
        // A full-length --background "10 20 30" is applied per band.
        let main = rgb(4, 4, vec![100; 48]);
        let sub = rgb(2, 2, vec![200; 12]);
        let out = main
            .try_insert(&sub, 3, 3, true, Some(&[10.0, 20.0, 30.0]))
            .unwrap();
        assert_eq!(out.getpoint(4, 0), vec![10.0, 20.0, 30.0]);
        assert_eq!(out.getpoint(0, 4), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn insert_background_length_mismatch_is_typed() {
        let main = rgb(2, 2, vec![1; 12]);
        let sub = rgb(1, 1, vec![9, 9, 9]);
        assert!(matches!(
            main.try_insert(&sub, 3, 3, true, Some(&[1.0, 2.0])),
            Err(ExtractError::BackgroundLengthMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn insert_with_negative_position_and_expand_shifts_the_origin() {
        let main = gray(2, 2, vec![1; 4]);
        let sub = gray(2, 2, vec![9; 4]);
        let out = main.insert(&sub, -1, -1, true);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);
        // Sub occupies the top-left 2x2, main the bottom-right 2x2, and sub
        // wins the overlap.
        assert_eq!(out.getpoint(0, 0), vec![9.0]);
        assert_eq!(out.getpoint(1, 1), vec![9.0]);
        assert_eq!(out.getpoint(2, 2), vec![1.0]);
        assert_eq!(out.getpoint(2, 0), vec![0.0]);
    }

    #[test]
    fn insert_clips_the_sub_image_without_expand() {
        let main = gray(3, 3, vec![1; 9]);
        let sub = gray(2, 2, vec![9; 4]);
        let out = main.insert(&sub, 2, 2, false);
        assert_eq!(out.width(), 3);
        assert_eq!(out.getpoint(2, 2), vec![9.0]);
        assert_eq!(out.getpoint(1, 1), vec![1.0]);
    }

    #[test]
    fn insert_bands_up_a_mono_image() {
        let mono = gray(2, 2, vec![5; 4]);
        let colour = rgb(1, 1, vec![1, 2, 3]);
        let out = mono.insert(&colour, 0, 0, false);
        assert_eq!(out.format(), PixelFormat::Rgb8);
        assert_eq!(out.getpoint(0, 0), vec![1.0, 2.0, 3.0]);
        // The mono main is replicated across the three bands.
        assert_eq!(out.getpoint(1, 1), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn insert_promotes_depth_numerically() {
        let main = gray(2, 1, vec![200, 201]);
        let sub = gray16(1, 1, &[60000]);
        let out = main.insert(&sub, 1, 0, false);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![200.0]);
        assert_eq!(out.getpoint(1, 0), vec![60000.0]);
    }

    #[test]
    fn try_insert_band_count_mismatch_is_typed() {
        let main = rgb(1, 1, vec![1, 2, 3]);
        let sub = rgba(1, 1, vec![1, 2, 3, 4]);
        assert!(matches!(
            main.try_insert(&sub, 0, 0, false, None),
            Err(ExtractError::BandCountMismatch { main: 3, sub: 4 })
        ));
    }

    // -- zoom ---------------------------------------------------------------------

    #[test]
    fn zoom_replicates_pixels() {
        #[rustfmt::skip]
        let im = gray(2, 2, vec![
            1, 2,
            3, 4,
        ]);
        let out = im.zoom(3, 2);
        assert_eq!(out.width(), 6);
        assert_eq!(out.height(), 4);
        assert_eq!(out.getpoint(0, 0), vec![1.0]);
        assert_eq!(out.getpoint(2, 1), vec![1.0]);
        assert_eq!(out.getpoint(3, 0), vec![2.0]);
        assert_eq!(out.getpoint(5, 3), vec![4.0]);
        assert_eq!(out.getpoint(1, 2), vec![3.0]);
    }

    #[test]
    fn zoom_identity_and_zero_factor() {
        let im = ramp4();
        assert_eq!(im.zoom(1, 1).data(), im.data());
        assert!(matches!(im.try_zoom(0, 1), Err(ExtractError::ZeroFactor)));
    }

    // -- subsample -------------------------------------------------------------------

    #[test]
    fn subsample_takes_the_top_left_of_each_cell() {
        let im = gray(6, 6, (0..36).collect());
        let out = im.subsample(3, 3);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
        assert_eq!(out.data(), &[0, 3, 18, 21]);
    }

    #[test]
    fn subsample_truncates_a_ragged_edge() {
        let im = gray(7, 5, vec![1; 35]);
        let out = im.subsample(3, 2);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
    }

    #[test]
    fn subsample_16bit_preserves_samples() {
        let im = gray16(2, 2, &[1000, 2000, 3000, 4000]);
        let out = im.subsample(2, 2);
        assert_eq!(out.getpoint(0, 0), vec![1000.0]);
    }

    #[test]
    fn try_subsample_bad_factors_are_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_subsample(0, 1),
            Err(ExtractError::ZeroFactor)
        ));
        assert!(matches!(
            im.try_subsample(5, 1),
            Err(ExtractError::FactorExceedsImage { xfac: 5, .. })
        ));
    }

    // -- smartcrop ----------------------------------------------------------------------

    #[test]
    fn smartcrop_centre_low_high_are_pure_geometry() {
        let im = gray(10, 10, (0..100).collect());
        let (centre, ax, ay) = im.smartcrop_with_coords(4, 4, SmartcropInteresting::Centre);
        assert_eq!((ax, ay), (0, 0));
        assert_eq!(centre.data(), im.extract_area(3, 3, 4, 4).data());

        let low = im.smartcrop(4, 4, SmartcropInteresting::Low);
        assert_eq!(low.data(), im.extract_area(0, 0, 4, 4).data());

        let high = im.smartcrop(4, 4, SmartcropInteresting::High);
        assert_eq!(high.data(), im.extract_area(6, 6, 4, 4).data());
    }

    #[test]
    fn smartcrop_all_returns_the_whole_image() {
        let im = ramp4();
        let out = im.smartcrop(2, 2, SmartcropInteresting::All);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert_eq!(out.data(), im.data());
    }

    #[test]
    fn smartcrop_entropy_keeps_the_busy_side() {
        // Left half constant, right half a checkerboard: every slice
        // comparison prefers the checkerboard, so the crop is the exact
        // right half.
        let w = 80u32;
        let h = 40u32;
        let mut data = vec![100u8; (w * h) as usize];
        for y in 0..h {
            for x in w / 2..w {
                data[(y * w + x) as usize] = if (x + y) % 2 == 0 { 0 } else { 255 };
            }
        }
        let im = gray(w, h, data);
        let out = im.smartcrop(40, 40, SmartcropInteresting::Entropy);
        assert_eq!(out.data(), im.extract_area(40, 0, 40, 40).data());
    }

    #[test]
    fn smartcrop_entropy_on_a_uniform_image_stays_at_the_origin() {
        let im = gray(20, 10, vec![7; 200]);
        let (out, ax, ay) = im.smartcrop_with_coords(5, 5, SmartcropInteresting::Entropy);
        assert_eq!((ax, ay), (0, 0));
        assert_eq!(out.data(), im.extract_area(0, 0, 5, 5).data());
    }

    #[test]
    fn smartcrop_attention_centres_on_the_feature() {
        // A white block on black: edges, brightness, and the masked colour
        // scores all point at the block.
        let w = 64u32;
        let h = 64u32;
        let mut data = vec![0u8; (w * h) as usize];
        for y in 12..20 {
            for x in 44..52 {
                data[(y * w as usize) + x] = 255;
            }
        }
        let im = gray(w, h, data);
        let (out, ax, ay) = im.smartcrop_with_coords(16, 16, SmartcropInteresting::Attention);
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        assert!(ax > 32, "attention_x {ax} should be in the right half");
        assert!(ay < 32, "attention_y {ay} should be in the top half");
        assert!(
            out.data().contains(&255),
            "the crop should contain the white block"
        );
    }

    #[test]
    fn smartcrop_attention_argmax_matches_libvips_on_a_synthetic_block() {
        // A 120x90 sRGB frame, dark gray (32, 32, 32) everywhere except a
        // bright saturated red block at rows 18..30, cols 84..100. The block
        // drives the edge, skin, and a* terms together. libvips 8.18
        // `vips_smartcrop` with the attention strategy and a 40x30 crop
        // reports attention (90, 22) on exactly these pixels, verified
        // against the real `vips` CLI. The lanczos3 shrink is load-bearing
        // here: a box-filter shrink lands the argmax elsewhere, so this pins
        // the resample fidelity in-repo, independent of the sample.jpg
        // fixture the ported suite asserts.
        let (w, h) = (120u32, 90u32);
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let (r, g, b) = if (18..30).contains(&y) && (84..100).contains(&x) {
                    (250u8, 40, 40)
                } else {
                    (32, 32, 32)
                };
                let off = (y * w as usize + x) * 3;
                data[off] = r;
                data[off + 1] = g;
                data[off + 2] = b;
            }
        }
        let im = rgb(w, h, data);
        let (_out, ax, ay) = im.smartcrop_with_coords(40, 30, SmartcropInteresting::Attention);
        assert_eq!(
            (ax, ay),
            (90, 22),
            "attention argmax must match libvips 8.18 on the synthetic block"
        );
    }

    #[test]
    fn smartcrop_attention_ignores_transparent_content() {
        // Two white blocks; the right one is fully transparent. The
        // premultiply step zeroes it, so attention goes left.
        let w = 64usize;
        let h = 64usize;
        let mut data = vec![0u8; w * h * 4];
        for i in 0..w * h {
            data[i * 4 + 3] = 255;
        }
        for y in 28..36 {
            for x in 12..20 {
                let i = y * w + x;
                data[i * 4] = 255;
                data[i * 4 + 1] = 255;
                data[i * 4 + 2] = 255;
            }
            for x in 44..52 {
                let i = y * w + x;
                data[i * 4] = 255;
                data[i * 4 + 1] = 255;
                data[i * 4 + 2] = 255;
                data[i * 4 + 3] = 0;
            }
        }
        let im = rgba(w as u32, h as u32, data);
        let (_, ax, _) = im.smartcrop_with_coords(16, 16, SmartcropInteresting::Attention);
        assert!(ax < 32, "attention_x {ax} should point at the opaque block");
    }

    /// Regression for #603: the attention analysis must stay in premultiplied
    /// space all the way to the argmax, the way it does in vips.
    ///
    /// A 128x128 image split on the diagonal: a saturated skin-tone triangle
    /// that is fully opaque, and a bright grey remainder that is fully
    /// transparent. vips premultiplies once and then resizes *without*
    /// un-premultiplying, so the transparent side is colour 0 by the time it is
    /// scored, its luma fails the `Y > 5` mask, and the argmax has to land on
    /// an opaque pixel. libviprs' `resize` premultiplies on its own, so before
    /// the fix its bracket un-premultiplied the analysis image on the way out;
    /// the lanczos ringing along the boundary came back divided by a near-zero
    /// alpha, lit up a wide band on the transparent side, and the argmax landed
    /// there.
    ///
    /// The assertion is the mechanism rather than the exact coordinate. This
    /// fixture is symmetric about the diagonal, so the two ends of the ridge
    /// score almost equally and the winner between them is a near tie — pinning
    /// which end wins would pin a coincidence. What is not a tie is which
    /// *side* of the transparency boundary wins, and that flips cleanly with
    /// the bug. For the record, the oracle agrees with the fixed code on the
    /// coordinate as well:
    ///
    /// ```text
    /// vips smartcrop diag.png o.png 32 32 --interesting attention \
    ///   --attention-x --attention-y   ->  100 then 0
    /// ```
    ///
    /// and it stays 100/0 through `--premultiplied` on a pre-premultiplied
    /// copy. Every threshold from 100 to 120 puts vips on the opaque side and
    /// the pre-fix code on the transparent side.
    #[test]
    fn smartcrop_attention_cannot_land_on_a_transparent_pixel() {
        const SPLIT: usize = 110;
        let (w, h) = (128usize, 128usize);
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let px: [u8; 4] = if x + y < SPLIT {
                    [215, 150, 120, 255]
                } else {
                    [181, 184, 193, 0]
                };
                data[i..i + 4].copy_from_slice(&px);
            }
        }
        let im = rgba(w as u32, h as u32, data);

        let (_, ax, ay) = im.smartcrop_with_coords(32, 32, SmartcropInteresting::Attention);
        assert!(
            (ax as usize) + (ay as usize) < SPLIT,
            "attention ({ax}, {ay}) landed on a fully transparent pixel: the \
             analysis image was un-premultiplied on the way out of resize (#603)"
        );

        // Same through the `premultiplied = true` door, which skips the
        // smartcrop-level premultiply because the caller already did it. The
        // resize must not undo it there either.
        let (_, pax, pay) = im.premultiply().smartcrop_with_coords_premultiplied(
            32,
            32,
            SmartcropInteresting::Attention,
            true,
        );
        assert!(
            (pax as usize) + (pay as usize) < SPLIT,
            "attention ({pax}, {pay}) landed on a fully transparent pixel \
             through the premultiplied door (#603)"
        );
    }

    #[test]
    fn smartcrop_premultiplied_flag_skips_the_internal_premultiply() {
        // Fully opaque image: premultiplying is the identity, so both
        // paths agree, which pins the flag's plumbing.
        let mut data = vec![10u8; 40 * 30 * 4];
        for px in data.chunks_mut(4) {
            px[3] = 255;
        }
        let im = rgba(40, 30, data);
        let (a, ax1, ay1) =
            im.smartcrop_with_coords_premultiplied(20, 10, SmartcropInteresting::Attention, true);
        let (b, ax2, ay2) = im.smartcrop_with_coords(20, 10, SmartcropInteresting::Attention);
        assert_eq!(a.data(), b.data());
        assert_eq!((ax1, ay1), (ax2, ay2));
    }

    #[test]
    fn smartcrop_16bit_centre() {
        let im = gray16(4, 4, &(0..16).map(|v| v * 1000).collect::<Vec<u16>>());
        let out = im.smartcrop(2, 2, SmartcropInteresting::Centre);
        assert_eq!(out.getpoint(0, 0), vec![5000.0]);
    }

    #[test]
    fn try_smartcrop_bad_sizes_are_typed() {
        let im = ramp4();
        assert!(matches!(
            im.try_smartcrop(5, 4, SmartcropInteresting::Centre, false),
            Err(ExtractError::AreaOutOfBounds { .. })
        ));
        assert!(matches!(
            im.try_smartcrop(0, 4, SmartcropInteresting::Centre, false),
            Err(ExtractError::EmptyArea)
        ));
    }

    #[test]
    fn smartcrop_entropy_slicing_matches_the_libvips_loop_shape() {
        // 100 -> 33 wide: max_slice = ceil(67 / 8) = 9, and the loop must
        // terminate exactly at the target width.
        let im = gray(100, 33, vec![1; 3300]);
        let out = im.smartcrop(33, 33, SmartcropInteresting::Entropy);
        assert_eq!(out.width(), 33);
        assert_eq!(out.height(), 33);
    }

    // -- direction parsing ---------------------------------------------------------------

    #[test]
    fn compass_direction_parses_all_nicknames() {
        let cases: &[(&str, CompassDirection)] = &[
            ("centre", CompassDirection::Centre),
            ("center", CompassDirection::Centre),
            ("north", CompassDirection::North),
            ("east", CompassDirection::East),
            ("south", CompassDirection::South),
            ("west", CompassDirection::West),
            ("north-east", CompassDirection::NorthEast),
            ("south-east", CompassDirection::SouthEast),
            ("south-west", CompassDirection::SouthWest),
            ("north-west", CompassDirection::NorthWest),
        ];
        for &(name, expected) in cases {
            assert_eq!(name.parse::<CompassDirection>().unwrap(), expected);
        }
        assert!(matches!(
            "sideways".parse::<CompassDirection>(),
            Err(ExtractError::UnknownDirection { .. })
        ));
    }
    // -- crop-origin arithmetic (issue #690) ----------------------------------------------

    /// The offset stamp is exact wherever `-v` fits an `i32`, and saturates
    /// only past that.
    ///
    /// `negated_origin` is a private function of a `u32`, so this costs
    /// nothing, where proving `extract_area` reaches the far end of the range
    /// needs a 2 GiB raster. #690 shipped the saturation "asserted only by
    /// reasoning" on the strength of that second cost, and the reasoning was
    /// wrong by one: `-(v.min(i32::MAX as u32) as i32)` saturates at
    /// `i32::MIN + 1`, so `2147483648` came back as `-2147483647` when
    /// `-2147483648` is representable and correct.
    #[test]
    fn the_crop_origin_stamp_is_exact_until_it_cannot_be() {
        assert_eq!(negated_origin(0), 0);
        assert_eq!(negated_origin(1), -1);
        assert_eq!(negated_origin(i32::MAX as u32), -2_147_483_647);
        // The cell the old spelling got wrong. `-2147483648` is `i32::MIN`,
        // it fits, and it is the true answer.
        assert_eq!(negated_origin(2_147_483_648), i32::MIN);
        // And the first input whose negation genuinely does not fit.
        assert_eq!(negated_origin(2_147_483_649), i32::MIN);
        assert_eq!(negated_origin(u32::MAX), i32::MIN);
    }

    // -- float refusal (issue #694) -------------------------------------------------------

    /// A float raster of `bands` bands filled with a ramp, the carrier an EXR,
    /// FITS or `.v` decode hands back, spelled `FloatF32(n)`.
    fn floatf(bands: u16, w: u32, h: u32) -> Raster {
        let n = (w * h) as usize * bands as usize;
        let data: Vec<u8> = (0..n).flat_map(|v| (v as f32).to_ne_bytes()).collect();
        let fmt = PixelFormat::FloatF32(std::num::NonZeroU16::new(bands).expect("bands"));
        Raster::new(w, h, fmt, data).expect("float fixture")
    }

    /// The same four-band float layout spelled `RgbaF32`, which is what
    /// [`PixelFormat::canonical`] produces and what a decoded RGBA EXR hands
    /// back.
    ///
    /// This crate has two spellings of every float layout on purpose (#531),
    /// so a predicate that is right for one and wrong for the other is the
    /// interesting way to get `reject_float` wrong, and it is invisible to a
    /// suite that only ever builds `FloatF32(n)`. Narrowing the guard to
    /// `matches!(fmt, PixelFormat::FloatF32(_))` leaves the whole library
    /// suite green and puts `RgbaF32` back to panicking out of a `Result`.
    fn rgbaf32(w: u32, h: u32) -> Raster {
        let n = (w * h) as usize * 4;
        let data: Vec<u8> = (0..n).flat_map(|v| (v as f32).to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::RgbaF32, data).expect("rgbaf32 fixture")
    }

    /// Issue #694. The four entry points that read samples through
    /// [`read_s`] / [`write_s`] return a typed refusal on a float raster
    /// instead of panicking out of a `Result` signature.
    ///
    /// The issue names `try_embed` and `try_gravity`. I probed all of
    /// `src/extract.rs` and it is four, plus two of `smartcrop`'s six
    /// strategies, which is the next test.
    ///
    /// Every `Extend` mode is here rather than just `White`, because the panic
    /// is in the sample copy and not in the ink: `Black`, `Copy`, `Repeat`,
    /// `Mirror` and `Background` reach it exactly as `White` does, so a fix
    /// that only guarded the inking path would leave five of six still
    /// panicking.
    #[test]
    fn the_sample_reading_ops_refuse_a_float_raster_instead_of_panicking() {
        let im = floatf(3, 8, 8);
        let sub = floatf(3, 2, 2);
        let bg = [1.0f64, 2.0, 3.0];
        let cases: Vec<(&str, Result<Raster, ExtractError>)> = vec![
            (
                "embed black",
                im.try_embed(1, 1, 12, 12, Extend::Black, None),
            ),
            (
                "embed white",
                im.try_embed(1, 1, 12, 12, Extend::White, None),
            ),
            ("embed copy", im.try_embed(1, 1, 12, 12, Extend::Copy, None)),
            (
                "embed repeat",
                im.try_embed(1, 1, 12, 12, Extend::Repeat, None),
            ),
            (
                "embed mirror",
                im.try_embed(1, 1, 12, 12, Extend::Mirror, None),
            ),
            (
                "embed background",
                im.try_embed(1, 1, 12, 12, Extend::Background, Some(&bg)),
            ),
            (
                "gravity",
                im.try_gravity(CompassDirection::Centre, 12, 12, Extend::Black, None),
            ),
            ("insert", im.try_insert(&sub, 1, 1, false, None)),
            ("insert expand", im.try_insert(&sub, 1, 1, true, None)),
            // Either input is enough on its own. The result takes the wider of
            // the two depths, so a float `sub` under an unsigned `main`
            // reaches the same sample copy, and a guard that only looked at
            // `self` would leave it panicking. I found this by mutating the
            // `sub` guard away and watching this test stay green, so it is
            // here rather than in the "nice to have" pile.
            (
                "insert float sub",
                rgb(8, 8, vec![1u8; 8 * 8 * 3]).try_insert(&sub, 1, 1, false, None),
            ),
            (
                "insert float sub expanding",
                rgb(8, 8, vec![1u8; 8 * 8 * 3]).try_insert(&sub, -1, -1, true, None),
            ),
            (
                "insert float main",
                im.try_insert(&rgb(2, 2, vec![1u8; 12]), 1, 1, false, None),
            ),
            // The other spelling of the same layout. See `rgbaf32`.
            (
                "embed rgbaf32",
                rgbaf32(8, 8).try_embed(1, 1, 12, 12, Extend::White, None),
            ),
            (
                "gravity rgbaf32",
                rgbaf32(8, 8).try_gravity(CompassDirection::Centre, 12, 12, Extend::Black, None),
            ),
            (
                "insert rgbaf32",
                rgbaf32(8, 8).try_insert(&rgbaf32(2, 2), 1, 1, false, None),
            ),
        ];
        for (name, got) in cases {
            assert!(
                matches!(got, Err(ExtractError::FloatUnsupported { .. })),
                "{name} must refuse a float raster rather than panic, got {got:?}"
            );
        }
    }

    /// Issue #694. `smartcrop` splits on the strategy, and a caller cannot see
    /// that from the signature.
    ///
    /// `Centre`, `Low`, `High` and `All` are pure geometry, so they take a
    /// float raster today and must keep taking it. `Entropy` pools a histogram
    /// and `Attention` builds saliency maps, and both read samples, so both
    /// panic. Measured, not assumed: I ran all six.
    ///
    /// The four that work are asserted as well as the two that do not, because
    /// a fix that rejected float at the `try_smartcrop` entry point would make
    /// this test's first half green by breaking four working strategies, and
    /// nothing else in the suite would notice.
    #[test]
    fn smartcrop_refuses_float_only_on_the_strategies_that_read_samples() {
        let im = floatf(3, 8, 8);
        for interesting in [
            SmartcropInteresting::Entropy,
            SmartcropInteresting::Attention,
        ] {
            let got = im.try_smartcrop(4, 4, interesting, false);
            assert!(
                matches!(got, Err(ExtractError::FloatUnsupported { .. })),
                "smartcrop {interesting:?} reads samples, so it must refuse a float raster"
            );
        }
        for interesting in [
            SmartcropInteresting::Centre,
            SmartcropInteresting::Low,
            SmartcropInteresting::High,
            SmartcropInteresting::All,
        ] {
            let (out, _, _) = im
                .try_smartcrop(4, 4, interesting, false)
                .unwrap_or_else(|e| panic!("smartcrop {interesting:?} is pure geometry: {e}"));
            assert_eq!(out.format(), im.format(), "smartcrop {interesting:?}");
        }

        // And on the alpha carrier, which is the only way to reach the
        // `premultiply()` branch above the strategy switch. A three-band
        // fixture never takes it, so "the four pure-geometry strategies take a
        // float raster unchanged" was untested for the case that actually has
        // a premultiply in front of it.
        let alpha = rgbaf32(8, 8);
        for premultiplied in [false, true] {
            let (out, _, _) = alpha
                .try_smartcrop(4, 4, SmartcropInteresting::Centre, premultiplied)
                .unwrap_or_else(|e| panic!("smartcrop centre on RgbaF32 ({premultiplied}): {e}"));
            assert_eq!(out.format(), PixelFormat::RgbaF32);
        }
        for interesting in [
            SmartcropInteresting::Entropy,
            SmartcropInteresting::Attention,
        ] {
            assert!(
                matches!(
                    alpha.try_smartcrop(4, 4, interesting, false),
                    Err(ExtractError::FloatUnsupported { .. })
                ),
                "smartcrop {interesting:?} on RgbaF32"
            );
        }
    }

    /// Issue #694. The kernel's `debug_assert` fires, so the sentence saying
    /// it is what holds the guard rather than a comment is itself held.
    ///
    /// `embed_impl` takes the guard on trust because the refusal has to name
    /// the public operation and the kernel cannot see which of its two callers
    /// came in. Deleting the assert leaves the suite green, since there is no
    /// third caller today, so without this the claim was exactly the shape
    /// #700 was filed about. The release net is still `write_s`'s own panic,
    /// so nothing is weaker than before either way.
    #[test]
    #[should_panic(expected = "callers must reject a float raster first")]
    fn embed_impl_asserts_its_callers_rejected_float_first() {
        let _ = floatf(3, 8, 8).embed_impl(1, 1, 12, 12, Extend::Black, None);
    }

    /// Issue #694 against #339. The panicking forms do not double the op name.
    ///
    /// `expect_extract` prefixes `"<op>: "` because most `ExtractError`
    /// variants do not say which operation failed. `FloatUnsupported` does, so
    /// prefixing it too gave `"embed: embed does not support float rasters
    /// yet"`. `arithmetic.rs` already fixed exactly this for its own
    /// `FloatUnsupported` and named #339 while doing it; #694 mirrored the
    /// error shape and not the wrapper, so the defect arrived here with it.
    #[test]
    fn the_panicking_forms_do_not_say_the_op_twice() {
        let im = floatf(3, 8, 8);
        for (op, call) in [
            (
                "embed",
                Box::new(move || {
                    floatf(3, 8, 8).embed(1, 1, 12, 12, Extend::Black, None);
                }) as Box<dyn FnOnce()>,
            ),
            (
                "gravity",
                Box::new(move || {
                    floatf(3, 8, 8).gravity(CompassDirection::Centre, 12, 12);
                }),
            ),
            (
                "insert",
                Box::new(move || {
                    floatf(3, 8, 8).insert(&floatf(3, 2, 2), 1, 1, false);
                }),
            ),
            (
                "smartcrop",
                Box::new(move || {
                    floatf(3, 8, 8).smartcrop(4, 4, SmartcropInteresting::Entropy);
                }),
            ),
        ] {
            let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call))
                .expect_err("must panic");
            let text = msg
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| msg.downcast_ref::<&str>().map(ToString::to_string))
                .expect("panic payload is a string");
            assert_eq!(
                text.matches(&format!("{op} does not support float"))
                    .count()
                    + text.matches(&format!("{op}: ")).count(),
                1,
                "the op name must appear once, got {text:?}"
            );
        }
        drop(im);
    }

    /// Issue #694. The refusal names the operation, so a caller reading the
    /// message knows which call to change.
    ///
    /// The old panic said "this extract operation", which is the whole problem
    /// in miniature: it reached the caller as a process-visible panic out of a
    /// `Result` signature, and it did not even say which operation.
    #[test]
    fn the_float_refusal_names_the_operation() {
        let im = floatf(3, 8, 8);
        for (want, got) in [
            ("embed", im.try_embed(1, 1, 12, 12, Extend::Black, None)),
            (
                "gravity",
                im.try_gravity(CompassDirection::Centre, 12, 12, Extend::Black, None),
            ),
            ("insert", im.try_insert(&floatf(3, 2, 2), 1, 1, false, None)),
        ] {
            let e = got.expect_err("must refuse");
            assert!(
                matches!(&e, ExtractError::FloatUnsupported { op } if *op == want),
                "expected the refusal to name {want}, got {e:?}"
            );
            assert!(
                e.to_string().contains(want),
                "the message a caller prints must name the op: {e}"
            );
        }
    }

    /// Issue #694. The unsigned carriers are untouched, which is the control
    /// that stops the refusal being written too wide.
    ///
    /// A guard that rejected by anything other than "is this float" would take
    /// these with it, and every one of them is an op the rest of this module's
    /// tests already exercise on 8-bit.
    #[test]
    fn the_unsigned_carriers_still_go_through_every_op() {
        let im16 =
            Raster::new(8, 8, PixelFormat::Rgb16, vec![3u8; 8 * 8 * 3 * 2]).expect("rgb16 fixture");
        let sub16 =
            Raster::new(2, 2, PixelFormat::Rgb16, vec![9u8; 2 * 2 * 3 * 2]).expect("sub fixture");
        assert!(im16.try_embed(1, 1, 12, 12, Extend::White, None).is_ok());
        assert!(
            im16.try_gravity(CompassDirection::Centre, 12, 12, Extend::Black, None)
                .is_ok()
        );
        assert!(im16.try_insert(&sub16, 1, 1, false, None).is_ok());
        assert!(
            im16.try_smartcrop(4, 4, SmartcropInteresting::Entropy, false)
                .is_ok()
        );
        assert!(
            im16.try_smartcrop(4, 4, SmartcropInteresting::Attention, false)
                .is_ok()
        );
    }

    // -- metadata (issue #690) -----------------------------------------------------------

    /// An 8x8 `Rgb8` ramp carrying every field [`crate::conversion::RasterMeta`]
    /// holds, plus an attached one, matching the raster the oracle tables in
    /// this section were measured on.
    fn tagged_source() -> Raster {
        let mut im = rgb(8, 8, (0..8u32 * 8 * 3).map(|v| v as u8).collect())
            .copy()
            .interpretation(Interpretation::ScRgb)
            .xres(5.0)
            .yres(7.0)
            .xoffset(11)
            .yoffset(13)
            .orientation(6)
            .build();
        im.set_field("lane-690", MetadataValue::Str("carried".to_string()));
        im
    }

    /// The seven operations issue #690 names, run on `im` with the arguments
    /// the oracle ran, plus `smartcrop` to pin that it inherits whatever
    /// `extract_area` decides.
    ///
    /// [`Raster::try_insert`] is deliberately absent. It drops the metadata
    /// the same way, but its rule is a two-input one and a different shape:
    /// measured on vips 8.18.6, the header block comes from `main` alone
    /// (`vips insert` of an scRGB main with a Lab sub reports scRGB, and the
    /// main's resolution, offset and orientation), while the attached fields
    /// are the union of both with `main` winning a name they share. Carrying
    /// that union needs a merge on `MetadataFields`, which lives in
    /// `imageio`, so it is a separate change rather than a line folded in
    /// here.
    fn extract_op_results(im: &Raster) -> Vec<(&'static str, Raster)> {
        vec![
            ("extract_area", im.extract_area(1, 1, 4, 4)),
            ("crop", im.crop(1, 1, 4, 4)),
            ("embed", im.embed(1, 1, 12, 12, Extend::White, None)),
            ("gravity", im.gravity(CompassDirection::Centre, 12, 12)),
            ("replicate", im.replicate(2, 3)),
            ("zoom", im.zoom(2, 3)),
            ("subsample", im.subsample(2, 4)),
            (
                "smartcrop",
                im.smartcrop(4, 4, SmartcropInteresting::Centre),
            ),
        ]
    }

    /// Issue #690. The interpretation survives every extract operation:
    /// none of them changes what a sample means.
    ///
    /// `ScRgb` on an `Rgb8` carrier is the tag that can actually fail here.
    /// An untagged `Rgb8` infers [`Interpretation::Srgb`], so a result that
    /// dropped the tag reads back as `Srgb` rather than as anything obviously
    /// empty, and the first assertion pins that the two differ so the loop
    /// below cannot pass by agreeing with the inference.
    ///
    /// Measured on vips 8.18.6 against an 8x8 uchar 3-band
    /// `--interpretation scrgb`: all eight report
    /// `VIPS_INTERPRETATION_scRGB`. The tag is not an scRGB special case
    /// either, `grey16`, `b-w`, `lab`, `cmyk`, `rgb16`, `hsv` and `oklab`
    /// all come back through `extract_area` and `embed` unchanged.
    #[test]
    fn every_extract_op_carries_the_interpretation() {
        let im = tagged_source();
        assert_eq!(
            Interpretation::for_format(im.format()),
            Interpretation::Srgb,
            "the tag under test has to differ from the inferred one"
        );
        for (name, out) in extract_op_results(&im) {
            assert_eq!(out.interpretation(), Interpretation::ScRgb, "{name}");
        }
    }

    /// Issue #690. Resolution and orientation ride along too.
    ///
    /// `zoom` and `subsample` are the two the issue asked to measure rather
    /// than assume, since they rescale the pixel grid: vips does **not**
    /// rescale the resolution with it. `vips zoom` by 2x3 on `xres=5 yres=7`
    /// reports 5 and 7 back, not 10 and 21, and `vips subsample` by 2x4
    /// reports the same 5 and 7 rather than 2.5 and 1.75. So the carry is
    /// verbatim on all seven and no per-op scaling rule is needed.
    ///
    /// A `.v` container stores the resolution as `float`, so `vipsheader -f
    /// yres` prints `6.9999606...` for the 7 that went in. The seven ops
    /// each report exactly what their input reported, which is the part
    /// under test; the rounding is the container's, not theirs.
    ///
    /// The defaults are `1.0`, `1.0` and `1`, so every value asserted here
    /// differs from what a freshly built raster would report.
    #[test]
    fn every_extract_op_carries_the_resolution_and_orientation() {
        let im = tagged_source();
        for (name, out) in extract_op_results(&im) {
            assert_eq!(out.xres(), 5.0, "{name} xres");
            assert_eq!(out.yres(), 7.0, "{name} yres");
            assert_eq!(out.orientation(), 6, "{name} orientation");
        }
    }

    /// Issue #690. The attached fields survive as well, which is the half a
    /// bare `out.meta = self.meta` leaves behind.
    ///
    /// Measured on vips 8.18.6 with a `VipsRefString` field written into the
    /// source's extension block (`vipsedit --setext`): every one of the
    /// eight reports it back.
    ///
    /// The ICC blob is here because it is the attachment a caller notices
    /// losing, and because it exercises the other `MetadataValue` arm. That
    /// half was measured on a real profile rather than a hand-written one:
    /// `vips icc_transform in.v out.v "sRGB Profile.icc"` attaches 3144 bytes
    /// of `icc-profile-data`, and all eight hand the same 3144 bytes on.
    ///
    /// This used to say a hand-written `VipsBlob` in an extension block does
    /// not survive being written back out. **That does not reproduce on
    /// 8.18.6**: a 48-byte blob written with `vipsedit --setext` comes back
    /// through `copy`, `gamma`, `fwfft` and the rest unchanged (#717 uses one
    /// as a control precisely because it does). A real profile is still the
    /// better carrier here, because it is the attachment the issue is about
    /// and the only one `icc_transform` will produce; it was never the only
    /// one that works.
    #[test]
    fn every_extract_op_carries_the_attached_fields() {
        let mut im = tagged_source();
        im.set_field("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3]));
        for (name, out) in extract_op_results(&im) {
            assert_eq!(
                out.get_field("lane-690"),
                Some(MetadataValue::Str("carried".to_string())),
                "{name} attached string"
            );
            assert_eq!(
                out.get_field("icc-profile-data"),
                Some(MetadataValue::Blob(vec![1, 2, 3])),
                "{name} attached blob"
            );
        }
    }

    /// Issue #690. The origin offset is the one field the seven do not agree
    /// on, so it is the one a wholesale carry would get wrong.
    ///
    /// `vips_extract_area` sets `Xoffset = -left` and `Yoffset = -top` and
    /// discards the source's, while the placement and tiling ops leave the
    /// source's alone. Measured on vips 8.18.6 from a source at
    /// `xoffset=11 yoffset=13`, sweeping `left` over 0/1/3/4 against `top`
    /// over 0/2/5: `extract_area` reports `-left` / `-top` in all twelve
    /// cells. `vips embed` over `x` in 0/2/-2 against `y` in 0/3/-3 reports
    /// 11 / 13 in all nine, and `replicate 2 3`, `zoom 2 3` and
    /// `subsample 2 4` report 11 / 13 as well.
    ///
    /// `smartcrop` inherits the rule because it is `extract_area`
    /// underneath: `--interesting centre` on 8x8 to 4x4 gives -2 / -2,
    /// `low` gives 0 / 0 and `high` gives -4 / -4, which is `-left` / `-top`
    /// for the three crops those strategies pick.
    #[test]
    fn crop_stamps_the_offset_where_the_others_carry_it() {
        let im = tagged_source();
        assert_eq!((im.xoffset(), im.yoffset()), (11, 13), "source offset");
        for left in [0u32, 1, 3, 4] {
            for top in [0u32, 2, 5] {
                let want = (-(left as i32), -(top as i32));
                let area = im.extract_area(left, top, 4, 3);
                assert_eq!(
                    (area.xoffset(), area.yoffset()),
                    want,
                    "extract_area {left},{top}"
                );
                let cropped = im.crop(left, top, 4, 3);
                assert_eq!(
                    (cropped.xoffset(), cropped.yoffset()),
                    want,
                    "crop {left},{top}"
                );
            }
        }
        for (name, want, out) in [
            (
                "smartcrop centre",
                (-2, -2),
                im.smartcrop(4, 4, SmartcropInteresting::Centre),
            ),
            (
                "smartcrop low",
                (0, 0),
                im.smartcrop(4, 4, SmartcropInteresting::Low),
            ),
            (
                "smartcrop high",
                (-4, -4),
                im.smartcrop(4, 4, SmartcropInteresting::High),
            ),
        ] {
            assert_eq!((out.xoffset(), out.yoffset()), want, "{name}");
        }
        for (name, out) in [
            ("embed", im.embed(1, 1, 12, 12, Extend::White, None)),
            (
                "embed negative",
                im.embed(-2, -3, 12, 12, Extend::Black, None),
            ),
            ("gravity", im.gravity(CompassDirection::Centre, 12, 12)),
            ("replicate", im.replicate(2, 3)),
            ("zoom", im.zoom(2, 3)),
            ("subsample", im.subsample(2, 4)),
        ] {
            assert_eq!((out.xoffset(), out.yoffset()), (11, 13), "{name}");
        }
    }

    /// Issue #690 against #667. [`Extend::White`] inks from the
    /// interpretation, so an embed that hands back an untagged result inks
    /// *differently the second time round*: the tag that chose the ink is
    /// gone, `Rgb8` infers [`Interpretation::Srgb`], and the border lands on
    /// 255 instead of 1.
    ///
    /// This is the pixel-level consequence of the tag carry, so it is
    /// asserted separately from the header tests above rather than folded
    /// into them.
    ///
    /// Measured on vips 8.18.6:
    ///
    /// ```text
    /// vips copy sevens.v sc.v --interpretation scrgb
    /// vips embed sc.v e1.v 1 1 8 8 --extend white     -> corner 1 1 1, scrgb
    /// vips embed e1.v e2.v 1 1 12 12 --extend white   -> corner 1 1 1, scrgb
    /// vips crop sc.v c1.v 1 1 2 2                     -> scrgb
    /// vips embed c1.v c2.v 1 1 6 6 --extend white     -> corner 1 1 1
    /// ```
    ///
    /// The sRGB control is the other half of it: the same source tagged
    /// `--interpretation srgb` paints 255 on both passes, so the assertion
    /// below is reading the tag rather than a constant either way.
    #[test]
    fn embed_white_inks_the_same_on_a_second_pass() {
        let scrgb = rgb(4, 4, vec![7; 48])
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build();
        let once = scrgb.embed(1, 1, 8, 8, Extend::White, None);
        assert_eq!(once.getpoint(0, 0), vec![1.0, 1.0, 1.0], "first embed");
        let twice = once.embed(1, 1, 12, 12, Extend::White, None);
        assert_eq!(twice.getpoint(0, 0), vec![1.0, 1.0, 1.0], "second embed");
        let after_crop = scrgb
            .crop(1, 1, 2, 2)
            .embed(1, 1, 6, 6, Extend::White, None);
        assert_eq!(after_crop.getpoint(0, 0), vec![1.0, 1.0, 1.0], "after crop");

        let srgb = rgb(4, 4, vec![7; 48])
            .copy()
            .interpretation(Interpretation::Srgb)
            .build();
        let white = srgb.embed(1, 1, 8, 8, Extend::White, None);
        assert_eq!(white.getpoint(0, 0), vec![255.0, 255.0, 255.0], "srgb once");
        let white2 = white.embed(1, 1, 12, 12, Extend::White, None);
        assert_eq!(
            white2.getpoint(0, 0),
            vec![255.0, 255.0, 255.0],
            "srgb twice"
        );
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
     * Tests the `Extend::White` ink for every carrier, including the three
     * signed ones, which no op-level test can reach while `embed` refuses
     * them (issue #909).
     * Works by calling [`white_ink`] directly under the `b-w` tag, whose
     * `max_alpha` is 255, so the ink byte is 0xFF and the answer is that
     * byte replicated across the sample width. Measured on
     * `/opt/homebrew/bin/vips` 8.18.6: `vips embed --extend white` fills
     * 255 on `uchar`, 65535 on `ushort`, 4294967295 on `uint`, and **-1**
     * on `char`, `short` and `int` alike, which are the same three
     * patterns read signed. The signedness appears at the store and not
     * here, because `memset` fills bytes and does not know the type.
     * Input: each carrier under `b-w` -> Output: 255, 65535, 4294967295 by
     * width, whatever the signedness.
     */
    #[test]
    fn white_ink_replicates_the_ink_byte_across_every_carrier() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let bw = Interpretation::Bw;
        // (format, the value the bytes hold, read unsigned)
        let cases = [
            (PixelFormat::Gray8, 255.0),
            (PixelFormat::Int8(n(1)), 255.0),
            (PixelFormat::Gray16, 65535.0),
            (PixelFormat::Int16(n(1)), 65535.0),
            (PixelFormat::Uint32(n(1)), 4_294_967_295.0),
            (PixelFormat::Int32(n(1)), 4_294_967_295.0),
        ];
        for (fmt, want) in cases {
            assert_eq!(
                white_ink(fmt, bw),
                want,
                "{fmt:?} inks the wrong pattern; vips fills the same bytes for a \
                 signed carrier as for its unsigned twin of the same width"
            );
        }
        // The pairs that share a width and must agree, which is what a
        // per-width arm gets right by luck and a per-carrier arm has to
        // state: one byte, two bytes, four bytes.
        assert_eq!(
            white_ink(PixelFormat::Int8(n(1)), bw),
            white_ink(PixelFormat::Gray8, bw)
        );
        assert_eq!(
            white_ink(PixelFormat::Int16(n(1)), bw),
            white_ink(PixelFormat::Gray16, bw)
        );
        assert_eq!(
            white_ink(PixelFormat::Int32(n(1)), bw),
            white_ink(PixelFormat::Uint32(n(1)), bw)
        );
        // And the control that the widths really differ, so the three
        // equalities above are not all comparing the same number.
        assert_ne!(
            white_ink(PixelFormat::Gray8, bw),
            white_ink(PixelFormat::Gray16, bw)
        );
    }

    /**
     * Tests that the sample-level extract ops carry the three signed
     * carriers, with the samples vips produces.
     * This replaces `embed_refuses_the_signed_carriers_for_now_issue_909`,
     * which asserted the typed refusal #516 shipped and said in its own
     * doc that it should be **replaced by value assertions rather than
     * deleted as though it had been wrong**. It was not wrong: the
     * refusal was the right interim while `read_s` returned a `u32` that
     * could not hold a negative, and it was a parity regression the whole
     * time, which is what issue #909 closes.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6 on a 2x2 `char` raster
     * holding `[-100, -1, 0, 100]`, embedded at (1, 1) in a 4x4 canvas:
     * `--extend white` fills **-1** (the all-bits-set byte a `memset`
     * lays down, read signed), `--extend black` fills 0, and
     * `--extend background` clips the constant into the carrier at both
     * ends, filling -50 for `--background -50`, **-128** for -200 and
     * **127** for 200. `vips insert` copies the sub-image's samples
     * unchanged.
     * Works by asserting each of those fills and the copied samples, with
     * `Uint32` as the control that the guard still discriminates by kind
     * and a float raster as the control that the one refusal left is
     * intact.
     * Input: `Int8` `[-100, -1, 0, 100]` -> Output: the measured canvases.
     */
    #[test]
    fn the_extract_ops_carry_the_signed_carriers() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let one = int8(2, 2, &[-100, -1, 0, 100]);

        let white = one.try_embed(1, 1, 4, 4, Extend::White, None).unwrap();
        assert_eq!(white.format(), PixelFormat::Int8(n(1)));
        #[rustfmt::skip]
        assert_eq!(i8s(&white), vec![
            -1,   -1, -1, -1,
            -1, -100, -1, -1,
            -1,    0, 100, -1,
            -1,   -1, -1, -1,
        ]);

        let black = one.try_embed(1, 1, 4, 4, Extend::Black, None).unwrap();
        #[rustfmt::skip]
        assert_eq!(i8s(&black), vec![
            0,    0,   0, 0,
            0, -100,  -1, 0,
            0,    0, 100, 0,
            0,    0,   0, 0,
        ]);

        // The background clips at **both** ends, which a `clamp(0, max)`
        // floor gets wrong for every negative constant.
        for (bg, want) in [(-50.0, -50i8), (-200.0, -128), (200.0, 127)] {
            let r = one
                .try_embed(1, 1, 4, 4, Extend::Background, Some(&[bg]))
                .unwrap();
            assert_eq!(
                i8s(&r)[0],
                want,
                "background {bg} must clip into the carrier, not into 0..=max"
            );
        }

        // insert copies the sub-image's samples, negatives included.
        let sub = int8(2, 2, &[-101, 2, -1, 101]);
        let inserted = one.try_insert(&sub, 0, 0, false, None).unwrap();
        assert_eq!(i8s(&inserted), vec![-101, 2, -1, 101]);

        // gravity is embed under another name, so the same ink rule holds.
        let grav = one
            .try_gravity(CompassDirection::Centre, 4, 4, Extend::White, None)
            .unwrap();
        assert_eq!(i8s(&grav)[0], -1);

        // Control: the unsigned 32-bit carrier of issue #517 still goes
        // through, so this is a refusal of a kind and not of a stride.
        assert!(
            Raster::zeroed(1, 1, PixelFormat::Uint32(n(1)))
                .unwrap()
                .try_embed(1, 1, 3, 3, Extend::White, None)
                .is_ok()
        );
        // Control: float is the one refusal left, and it still names the op.
        let f = Raster::new(
            1,
            1,
            PixelFormat::FloatF32(n(1)),
            1.5f32.to_ne_bytes().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            f.try_embed(1, 1, 3, 3, Extend::White, None),
            Err(ExtractError::FloatUnsupported { op: "embed" })
        ));
    }

    /// A one-band `FloatF32` raster from `f32` sample values.
    fn float1(w: u32, h: u32, vals: &[f32]) -> Raster {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let fmt = PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap());
        Raster::new(w, h, fmt, data).unwrap()
    }

    /// Every sample of a float raster, read back as `f32`.
    fn f32s(r: &Raster) -> Vec<f32> {
        r.data()
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /**
     * Tests that `embed`, `gravity` and `insert` carry a float raster and
     * answer with the samples vips produces, rather than refusing it.
     * The refusal was posture 1, a parity regression, and the same shape
     * issue #909 closed one carrier family earlier: `read_s` could not
     * hold the value, so the op refused, and the refusal outlived the
     * reason for it.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6 over a 3x1 `float`
     * raster holding `[1.5, -0.25, 3.75]`, cast from a `csvload` double:
     * `vips embed a.v out.v 1 0 5 1` answers **FLOAT** and fills the
     * border with 0 for `--extend black`, **255** for `--extend white`
     * and **-0.5** for `--extend background --background -0.5`, that last
     * one *unrounded*, which is what separates a float ink from the
     * integer one. `vips gravity a.v out.v centre 5 1` gives the same
     * canvas as the centred embed. `vips insert a.v b.v out.v 1 0` over
     * `b = [10.5, -2.75, 0.125]` answers `[1.5, 10.5, -2.75]`, and
     * `--expand --background -0.5` at x=4 answers the seven samples
     * below.
     * Works by asserting each of those canvases sample by sample, with
     * the fractional background as the cell a truncating store cannot
     * pass and the `Int8` white ink beside it as the control that the
     * integer dialect is untouched.
     * Input: `FloatF32(1)` `[1.5, -0.25, 3.75]` -> Output: the measured
     * canvases, at `FloatF32(1)`.
     */
    #[test]
    fn the_extract_ops_carry_a_float_raster_issue_945() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let a = float1(3, 1, &[1.5, -0.25, 3.75]);
        let b = float1(3, 1, &[10.5, -2.75, 0.125]);

        let black = a.try_embed(1, 0, 5, 1, Extend::Black, None).unwrap();
        assert_eq!(black.format(), PixelFormat::FloatF32(n(1)));
        assert_eq!(f32s(&black), vec![0.0, 1.5, -0.25, 3.75, 0.0]);

        let white = a.try_embed(1, 0, 5, 1, Extend::White, None).unwrap();
        assert_eq!(f32s(&white), vec![255.0, 1.5, -0.25, 3.75, 255.0]);

        // The cell a truncating store cannot pass: vips fills -0.5, not 0.
        let bg = a
            .try_embed(1, 0, 5, 1, Extend::Background, Some(&[-0.5]))
            .unwrap();
        assert_eq!(f32s(&bg), vec![-0.5, 1.5, -0.25, 3.75, -0.5]);

        // gravity is embed under another name, and vips centres the 3-wide
        // image in a 5-wide canvas at the same offset.
        let grav = a
            .try_gravity(CompassDirection::Centre, 5, 1, Extend::Black, None)
            .unwrap();
        assert_eq!(f32s(&grav), vec![0.0, 1.5, -0.25, 3.75, 0.0]);

        let ins = a.try_insert(&b, 1, 0, false, None).unwrap();
        assert_eq!(ins.format(), PixelFormat::FloatF32(n(1)));
        assert_eq!(f32s(&ins), vec![1.5, 10.5, -2.75]);

        let expanded = a.try_insert(&b, 4, 0, true, Some(&[-0.5])).unwrap();
        assert_eq!(expanded.width(), 7);
        assert_eq!(
            f32s(&expanded),
            vec![1.5, -0.25, 3.75, -0.5, 10.5, -2.75, 0.125]
        );

        // Control: the integer dialect is untouched, so the `char` white
        // ink is still the -1 a `memset` lays down rather than a clipped
        // 127.
        let one = int8(2, 2, &[-100, -1, 0, 100]);
        let iwhite = one.try_embed(1, 1, 4, 4, Extend::White, None).unwrap();
        assert_eq!(i8s(&iwhite)[0], -1);
        // Control: an integer background is still truncated toward zero,
        // so the float pass-through above is a property of the carrier and
        // not a dropped rounding step.
        let ibg = one
            .try_embed(1, 1, 4, 4, Extend::Background, Some(&[-0.5]))
            .unwrap();
        assert_eq!(i8s(&ibg)[0], 0);
    }

    /**
     * Tests that this module's sample reader and writer round-trip every
     * sample kind at its own stride and its own signedness.
     * It exists because **no op-level test can catch a read-side
     * signedness bug here**: `embed` and `insert` read a sample and write
     * it back at the same kind, so reading `char` -100 as 156 and storing
     * `156 as i8` gives -100 again and the canvas is identical. Mutating
     * `read_s`'s `I8` arm to an unsigned read left all 74 tests in this
     * module green, a real NO TEST REDDENS, and the round trip is what
     * cancelled it. This cell does not cancel: `write_s` stores the two's
     * complement and the read has to give the number back, so an unsigned
     * read answers 255 where -1 was written.
     * Works by sweeping [`ALL_KINDS`] rather than a hand-written list, and
     * by writing at sample index 1 of a two-sample buffer so a wrong
     * stride overwrites index 0 and is caught by the neighbour assertion.
     * Input: each kind's `range()` endpoints and 0 -> Output: the same
     * numbers back, index 0 still zero.
     */
    #[test]
    fn read_s_and_write_s_round_trip_every_kind_at_its_own_stride() {
        for kind in ALL_KINDS {
            let bytes = kind.bytes();
            let cases: [i64; 3] = match kind.range() {
                Some((lo, hi)) => [lo, 0, hi],
                None => [-128, 0, 127],
            };
            for v in cases {
                let mut buf = vec![0u8; bytes * 2];
                write_s(&mut buf, kind, 1, v);
                assert_eq!(read_s(&buf, kind, 1), v, "{kind:?} did not round-trip {v}");
                assert!(
                    buf[..bytes].iter().all(|&b| b == 0),
                    "{kind:?} wrote outside sample 1, so its stride is wrong"
                );
            }
        }
        // The width collisions, stated directly. `-1` is the same byte in
        // both one-byte kinds and a different number, which is the exact
        // substitution the round trip through an op cannot see.
        let mut b8 = vec![0u8; 1];
        write_s(&mut b8, SampleKind::I8, 0, -1);
        assert_eq!(b8[0], 0xFF);
        assert_eq!(read_s(&b8, SampleKind::U8, 0), 255);
        assert_eq!(read_s(&b8, SampleKind::I8, 0), -1);
        let mut b32 = vec![0u8; 4];
        write_s(&mut b32, SampleKind::I32, 0, -1);
        assert_eq!(read_s(&b32, SampleKind::U32, 0), 4_294_967_295);
        assert_eq!(read_s(&b32, SampleKind::I32, 0), -1);
    }

    /**
     * Tests that the background ink clips into each carrier's own range at
     * both ends, driven directly rather than through an op, so the arms no
     * op can reach are held by something.
     * Works by sweeping [`ALL_KINDS`] and pushing one constant past each
     * end of every integer kind's range, with `NaN` beside them because
     * `clamp` passes `NaN` through and the explicit zero arm is what stops
     * it landing on a carrier value by accident.
     * Input: -300 and 1e12 at every integer kind -> Output: that kind's
     * `range()` endpoints; `NaN` -> 0.
     */
    #[test]
    fn ink_value_clips_into_every_carrier_at_both_ends() {
        for kind in ALL_KINDS {
            let Some((lo, hi)) = kind.range() else {
                continue;
            };
            assert_eq!(
                ink_value(-300.0, lo, hi),
                (-300i64).max(lo),
                "{kind:?} floor"
            );
            assert_eq!(
                ink_value(1e12, lo, hi),
                1_000_000_000_000i64.min(hi),
                "{kind:?} ceiling"
            );
            assert_eq!(ink_value(f64::NAN, lo, hi), 0, "{kind:?} NaN");
            // Truncation toward zero, not flooring: the two differ only on
            // a negative fraction, which is exactly what a signed carrier
            // adds. `vips_cast` truncates.
            assert_eq!(
                ink_value(-1.9, lo, hi),
                if lo < 0 { -1 } else { 0 },
                "{kind:?}"
            );
        }
        // The pair a per-width rule would collide, stated directly.
        assert_eq!(ink_value(-50.0, -128, 127), -50);
        assert_eq!(ink_value(-50.0, 0, 255), 0);
    }

    /**
     * Tests that `embed` carries the unsigned 32-bit carrier: it copies
     * the source samples at the right stride and inks the border with the
     * carrier's own white.
     * Works by embedding a 1x1 `uint` raster in the middle of a 3x3 canvas
     * with `Extend::White` and reading both a source sample and a border
     * one, so a read at half stride moves the first and a wrong ink moves
     * the second. Both pinned to `/opt/homebrew/bin/vips` 8.18.6, where
     * `vips embed --extend white` on a `uint` raster fills **4294967295**
     * and an `int` one fills -1, the same bytes read signed.
     * Input: uint 90000 embedded at (1, 1) in 3x3 -> centre 90000, border
     * 4294967295.
     */
    #[test]
    fn embed_carries_the_uint_carrier_and_its_white_ink() {
        let src = uint32(1, 1, &[90_000]);
        let out = src
            .try_embed(1, 1, 3, 3, Extend::White, None)
            .expect("embed carries the uint carrier");
        assert_eq!(out.format(), src.format());
        assert_eq!(
            u32_at(&out, 4),
            90_000,
            "the source sample moved or was misread"
        );
        assert_eq!(
            u32_at(&out, 0),
            4_294_967_295,
            "the white ink is not the carrier's"
        );
        // Controls: the same call on the carriers that already worked.
        let g8 = gray(1, 1, vec![200]);
        let o8 = g8.try_embed(1, 1, 3, 3, Extend::White, None).unwrap();
        assert_eq!(o8.data()[4], 200);
        assert_eq!(o8.data()[0], 255);
        let g16 = gray16(1, 1, &[40000]);
        let o16 = g16.try_embed(1, 1, 3, 3, Extend::White, None).unwrap();
        assert_eq!(u16::from_ne_bytes([o16.data()[8], o16.data()[9]]), 40000);
    }

    /**
     * Tests that `insert` picks the output carrier through
     * `SampleKind::promote` rather than through the wider byte width,
     * which answers the float carrier at four bytes (issues #517, #607).
     * Works by inserting an 8-bit raster into a `uint` one and asserting
     * both the format and a sample, with the 8-into-16 pair as the control
     * that the existing promotion did not move.
     * Input: insert(uint, u8) -> Uint32 carrying 90000; insert(u16, u8) ->
     * Gray16.
     */
    #[test]
    fn insert_promotes_through_the_kind() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let main = uint32(2, 1, &[90_000, 90_000]);
        let sub = gray(1, 1, vec![7]);
        let out = main.try_insert(&sub, 0, 0, false, None).unwrap();
        assert_eq!(out.format(), PixelFormat::Uint32(n(1)));
        assert_eq!(u32_at(&out, 1), 90_000);
        assert_eq!(u32_at(&out, 0), 7);
        // Control: the promotion that existed before is untouched.
        let out16 = gray16(2, 1, &[40000, 40000])
            .try_insert(&sub, 0, 0, false, None)
            .unwrap();
        assert_eq!(out16.format(), PixelFormat::Gray16);
    }

    /**
     * Tests that the two smartcrop strategies which build a value-indexed
     * table refuse the 32-bit carrier as a typed error rather than
     * panicking on an out-of-range histogram index.
     * Works by asking for both strategies on a `uint` raster whose samples
     * are far above 65536, with the pure-geometry strategies as the
     * control that smartcrop itself still works on that carrier.
     * Input: Entropy / Attention on uint -> Err(UnsupportedSampleKind);
     * Centre on uint -> Ok.
     */
    #[test]
    fn smartcrop_refuses_the_uint_carrier_where_it_needs_a_table() {
        let r = uint32(4, 4, &[90_000; 16]);
        for interesting in [
            SmartcropInteresting::Entropy,
            SmartcropInteresting::Attention,
        ] {
            let got = r.try_smartcrop(2, 2, interesting, false);
            assert!(
                matches!(
                    got,
                    Err(ExtractError::UnsupportedSampleKind {
                        op: "smartcrop",
                        kind: SampleKind::U32
                    })
                ),
                "{interesting:?} on a uint raster gave {got:?}"
            );
        }
        // Control: the geometry strategies are depth-agnostic and still
        // carry the same raster, so the refusal is about the table.
        assert!(
            r.try_smartcrop(2, 2, SmartcropInteresting::Centre, false)
                .is_ok()
        );
    }
}
