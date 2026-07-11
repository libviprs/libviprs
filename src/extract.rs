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
//!   background vector says otherwise. Background constants are rounded to
//!   nearest and clamped to the sample depth (`0..=255` or `0..=65535`).
//!   A background vector must have one entry (replicated across bands) or
//!   exactly one entry per band.
//! * **Clipping.** `embed` and `insert` accept placements partly or wholly
//!   outside the canvas and clip, exactly as libvips does.
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
//! The `Attention` strategy is a structural port of
//! `vips_smartcrop_attention`: shrink to 32x32, Laplacian edge detection on
//! the XYZ luminance band scaled by 5, a skin-tone distance score and the
//! LAB `a*` band masked to bright pixels (`Y > 5`), the three maps summed,
//! blurred by a Gaussian sized from the crop target, and the maximum
//! position mapped back to input coordinates. It is not yet bit-identical
//! to libvips: the shrink is a box filter rather than `vips_resize`'s
//! Lanczos3 pipeline and the convolution runs in floating point, so the
//! exact attention coordinates the ported suite asserts for the real
//! fixtures (`sample.jpg`: 199, 234; `rgba.png`: 20, 124) are deferred
//! until the resample and colourspace primitives match libvips bit for
//! bit. The strategy is otherwise complete and deterministic.

use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
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
    /// Fill with white (the depth maximum in every band).
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

/// Read the flat `i`-th sample as `u32` (native byte order for 16-bit,
/// matching [`crate::raster_ops`]). Unsigned depths only: the panic arm
/// keeps the sample-level extract ops, which predate the float formats,
/// from misreading float bytes as `u16` pairs. (The pure byte-copy paths
/// like `extract_area` are depth-agnostic and handle float fine.)
#[inline]
fn read_s(data: &[u8], bpc: usize, i: usize) -> u32 {
    match bpc {
        1 => data[i] as u32,
        2 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as u32,
        _ => panic!(
            "this extract operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Write the flat `i`-th sample. `v` must already fit the depth.
/// Unsigned depths only; see [`read_s`].
#[inline]
fn write_s(data: &mut [u8], bpc: usize, i: usize, v: u32) {
    match bpc {
        1 => data[i] = v as u8,
        2 => {
            let b = (v as u16).to_ne_bytes();
            data[2 * i] = b[0];
            data[2 * i + 1] = b[1];
        }
        _ => panic!(
            "this extract operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Round and clamp an `f64` background constant into `0..=max`.
#[inline]
fn ink_value(v: f64, max: u32) -> u32 {
    if v.is_nan() {
        0
    } else {
        v.round().clamp(0.0, max as f64) as u32
    }
}

/// Resolve a background option into one clamped sample per band.
///
/// `None` is black, one value is replicated across the bands, and a
/// full-length vector is used per band; any other length is a typed error.
fn resolve_ink(
    bands: usize,
    max: u32,
    background: Option<&[f64]>,
) -> Result<Vec<u32>, ExtractError> {
    match background {
        None => Ok(vec![0; bands]),
        Some(bg) if bg.len() == 1 => Ok(vec![ink_value(bg[0], max); bands]),
        Some(bg) if bg.len() == bands => Ok(bg.iter().map(|&v| ink_value(v, max)).collect()),
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
    let dbpc = dst.format().bytes_per_channel();
    let (dw, dh) = (dst.width() as i64, dst.height() as i64);
    let dstride = dst.width() as usize;
    let sbands = src.format().channels();
    let sbpc = src.format().bytes_per_channel();
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
                write_s(ddata, dbpc, di + c, read_s(sdata, sbpc, si + sc));
            }
        }
    }
}

/// Unwrap an extract-op result for the panicking ported-test surface.
#[inline]
#[track_caller]
fn expect_extract(op: &str, r: Result<Raster, ExtractError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Unwrap a smartcrop result for the panicking ported-test surface.
#[inline]
#[track_caller]
fn expect_smartcrop(r: Result<(Raster, i32, i32), ExtractError>) -> (Raster, i32, i32) {
    match r {
        Ok(v) => v,
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
        Ok(self.extract(left, top, width, height)?)
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
    /// or [`ExtractError::BackgroundLengthMismatch`] for a bad background
    /// vector.
    pub fn try_embed(
        &self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        extend: Extend,
        background: Option<&[f64]>,
    ) -> Result<Raster, ExtractError> {
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
        let fmt = self.format();
        let bands = fmt.channels();
        let bpc = fmt.bytes_per_channel();
        let max = if bpc == 1 { 255u32 } else { 65535u32 };
        let ink: Vec<u32> = match extend {
            Extend::White => vec![max; bands],
            Extend::Background => resolve_ink(bands, max, background)?,
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
                            write_s(&mut out, bpc, di + c, read_s(data, bpc, si + c));
                        }
                    }
                    None => {
                        for (c, &v) in ink.iter().enumerate() {
                            write_s(&mut out, bpc, di + c, v);
                        }
                    }
                }
            }
        }
        Ok(Raster::new(width, height, fmt, out)?)
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
    /// or [`ExtractError::BackgroundLengthMismatch`] for a bad background
    /// vector.
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
        Ok(Raster::new(ow as u32, oh as u32, self.format(), out)?)
    }

    /// Insert `sub` over `self` with its top-left corner at `(x, y)`
    /// (libvips `insert`).
    ///
    /// With `expand` false the result keeps `self`'s size and `sub` is
    /// clipped; with `expand` true the result is the bounding box of both
    /// rectangles and uncovered pixels are black. `sub` overwrites `self`
    /// where they overlap (no alpha blending), as in libvips. The result
    /// takes the wider depth and the larger band count; a one-band input
    /// is replicated across the result bands, and band counts that differ
    /// with neither being 1 are an error, matching libvips `bandalike`.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractError::BandCountMismatch`] for incompatible band
    /// counts, or [`ExtractError::SizeOverflow`] if the expanded canvas
    /// would not fit `u32` dimensions.
    pub fn try_insert(
        &self,
        sub: &Raster,
        x: i32,
        y: i32,
        expand: bool,
    ) -> Result<Raster, ExtractError> {
        let mb = self.format().channels();
        let sb = sub.format().channels();
        if mb != sb && mb != 1 && sb != 1 {
            return Err(ExtractError::BandCountMismatch { main: mb, sub: sb });
        }
        let bands = mb.max(sb);
        let bpc = self
            .format()
            .bytes_per_channel()
            .max(sub.format().bytes_per_channel());
        let fmt = PixelFormat::with_channels(bands, bpc)
            .expect("band count is bounded by the two input formats");
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
        blit(&mut out, self, -ox, -oy);
        blit(&mut out, sub, x as i64 - ox, y as i64 - oy);
        Ok(out)
    }

    /// Panicking form of [`Raster::try_insert`], matching the ported-test
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ExtractError`]; see [`Raster::try_insert`].
    #[track_caller]
    pub fn insert(&self, sub: &Raster, x: i32, y: i32, expand: bool) -> Raster {
        expect_extract("insert", self.try_insert(sub, x, y, expand))
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
    let bpc = im.format().bytes_per_channel();
    let bins = if bpc == 1 { 256 } else { 65536 };
    let mut hist = vec![0u64; bins];
    let stride = im.width() as usize;
    let data = im.data();
    for yy in y..y + h {
        for xx in x..x + w {
            let base = (yy as usize * stride + xx as usize) * bands;
            for c in 0..bands {
                hist[read_s(data, bpc, base + c) as usize] += 1;
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

/// One box-filtered 1D resample pass over the rows of a `sw` x `sh` plane,
/// producing `dw` samples per row (weights are cell coverage, so the pass
/// is exact area averaging for downscale and blends the covered pixel for
/// upscale).
fn resample_rows(src: &[f64], sw: usize, sh: usize, dw: usize) -> Vec<f64> {
    let ratio = sw as f64 / dw as f64;
    let mut out = vec![0.0; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for dx in 0..dw {
            let start = dx as f64 * ratio;
            let end = start + ratio;
            let i0 = start.floor() as usize;
            let i1 = (end.ceil() as usize).min(sw);
            let mut acc = 0.0;
            for (i, &v) in row.iter().enumerate().take(i1).skip(i0) {
                let overlap = end.min((i + 1) as f64) - start.max(i as f64);
                acc += v * overlap;
            }
            out[y * dw + dx] = acc / ratio;
        }
    }
    out
}

/// Transpose a `w` x `h` plane.
fn transpose(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    out
}

/// Box-resample a plane to `dw` x `dh` via two 1D passes.
fn resample_plane(src: &[f64], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f64> {
    let rows = resample_rows(src, sw, sh, dw);
    let cols = resample_rows(&transpose(&rows, dw, sh), sh, dw, dh);
    transpose(&cols, dh, dw)
}

/// Extract the (r, g, b) planes of the analysis image at the attention
/// working size, on the 8-bit `0..=255` scale (16-bit samples divide by
/// 257). One- and two-band images replicate band 0; four or more bands use
/// the first three, the analysis input being already premultiplied.
fn rgb_planes(im: &Raster, n: usize) -> [Vec<f64>; 3] {
    let bands = im.format().channels();
    let bpc = im.format().bytes_per_channel();
    let scale = if bpc == 1 { 1.0 } else { 257.0 };
    let (w, h) = (im.width() as usize, im.height() as usize);
    let data = im.data();
    let mut planes: [Vec<f64>; 3] = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
    for i in 0..w * h {
        for (c, plane) in planes.iter_mut().enumerate() {
            let sc = if bands >= 3 { c } else { 0 };
            plane[i] = read_s(data, bpc, i * bands + sc) as f64 / scale;
        }
    }
    planes.map(|p| resample_plane(&p, w, h, n, n))
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

/// `|5 * laplacian|` of a plane with copy-edge extension, the libvips
/// attention edge detector.
fn edge_map(y_plane: &[f64], n: usize) -> Vec<f64> {
    let at = |x: i64, yy: i64| {
        let x = x.clamp(0, n as i64 - 1) as usize;
        let yy = yy.clamp(0, n as i64 - 1) as usize;
        y_plane[yy * n + x]
    };
    let mut out = vec![0.0; n * n];
    for yy in 0..n as i64 {
        for x in 0..n as i64 {
            let lap =
                4.0 * at(x, yy) - at(x, yy - 1) - at(x, yy + 1) - at(x - 1, yy) - at(x + 1, yy);
            out[yy as usize * n + x as usize] = (5.0 * lap).abs();
        }
    }
    out
}

/// Separable Gaussian blur with copy-edge extension. The mask half-width
/// follows libvips `gaussblur` with its default minimum amplitude of 0.2.
fn gaussblur_plane(plane: &[f64], n: usize, sigma: f64) -> Vec<f64> {
    let half = ((sigma * (-2.0 * 0.2f64.ln()).sqrt()).ceil() as i64).max(1);
    let weights: Vec<f64> = (-half..=half)
        .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = weights.iter().sum();
    let pass = |src: &[f64]| {
        let mut out = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n as i64 {
                let mut acc = 0.0;
                for (k, &wt) in weights.iter().enumerate() {
                    let sx = (x + k as i64 - half).clamp(0, n as i64 - 1) as usize;
                    acc += src[y * n + sx] * wt;
                }
                out[y * n + x as usize] = acc / norm;
            }
        }
        out
    };
    // Horizontal pass, then the same pass over the transposed plane.
    transpose(&pass(&transpose(&pass(plane), n, n)), n, n)
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

    let [rp, gp, bp] = rgb_planes(im, n);
    let mut total = vec![0.0; n * n];
    let mut y_plane = vec![0.0; n * n];
    let mut xyz = vec![(0.0, 0.0, 0.0); n * n];
    for i in 0..n * n {
        let (x, y, z) = srgb_to_xyz(rp[i], gp[i], bp[i]);
        y_plane[i] = y;
        xyz[i] = (x, y, z);
    }
    let edges = edge_map(&y_plane, n);
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
    let blurred = gaussblur_plane(&total, n, sigma);

    let mut best = f64::NEG_INFINITY;
    let (mut mx, mut my) = (0usize, 0usize);
    for (i, &v) in blurred.iter().enumerate() {
        if v > best {
            best = v;
            mx = i % n;
            my = i / n;
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

    #[test]
    fn embed_white_fills_with_depth_max() {
        let im = gray(1, 1, vec![7]);
        let out = im.embed(1, 1, 3, 3, Extend::White, None);
        assert_eq!(out.getpoint(0, 0), vec![255.0]);
        assert_eq!(out.getpoint(1, 1), vec![7.0]);

        let im16 = gray16(1, 1, &[7]);
        let out16 = im16.embed(1, 1, 3, 3, Extend::White, None);
        assert_eq!(out16.getpoint(0, 0), vec![65535.0]);
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
            main.try_insert(&sub, 0, 0, false),
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
}
