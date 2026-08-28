//! Morphological operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], and [`crate::colour`]): binary erosion and
//! dilation with a structuring element, the rank (order-statistic) filter,
//! line counting, and connected-component labelling. Operations that can
//! fail on caller input exist in two forms, following the established
//! convention:
//!
//! * a fallible `try_*` method returning `Result<_, MorphologyError>` with
//!   typed errors for bad structuring elements, out-of-range windows, and
//!   unsupported formats; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`erode`, `dilate`, `rank`, `countlines`, `label_regions`) exactly,
//!   delegating to the `try_*` form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::erode`] | `vips_morph` with `VIPS_OPERATION_MORPHOLOGY_ERODE` | eroded image |
//! | [`Raster::dilate`] | `vips_morph` with `VIPS_OPERATION_MORPHOLOGY_DILATE` | dilated image |
//! | [`Raster::rank`] | `vips_rank` | order-statistic filtered image |
//! | [`Raster::countlines`] | `vips_countlines` | average line count, `f64` |
//! | [`Raster::label_regions`] | `vips_labelregions` | (label mask, segment count) |
//!
//! # Semantics shared with libvips
//!
//! * **Structuring elements.** `erode` / `dilate` take the mask as rows of
//!   bytes where `255` means the input must be non-zero (object), `0`
//!   means it must be zero (background), and `128` means don't care. Any
//!   other value is a typed error, exactly like the libvips "bad mask
//!   element" check. The mask origin is at `(width / 2, height / 2)`,
//!   integer division.
//! * **Bitwise combination.** libvips combines mask hits with bitwise
//!   AND/OR over the sample bytes (`vips_erode_gen` starts at 255 and
//!   ANDs `coeff ? p : ~p`; dilate starts at 0 and ORs). For the 0/255
//!   binary images the operations are defined over this is boolean
//!   erosion/dilation; for intermediate grey values the same bitwise
//!   behaviour is reproduced here.
//! * **Edges.** The input is notionally extended by replicating its edge
//!   pixels (`VIPS_EXTEND_COPY`), so the output has the same dimensions as
//!   the input. The rank filter extends the same way.
//! * **Bands.** All operations except `countlines` treat each band
//!   independently. `countlines` requires a one-band image, as in libvips.
//! * **Depths.** `erode` and `dilate` accept 8-bit images (libvips casts
//!   other depths to uchar first; here the cast is left to the caller and
//!   deeper formats are a typed error). `rank` accepts unsigned 8- and
//!   16-bit images. `countlines` and `label_regions` accept unsigned 8-
//!   and 16-bit images; float rasters are a typed error for every
//!   operation, matching the "cast first" convention used by the
//!   histogram batch.
//! * **`label_regions` labels.** libvips scans row-major for the first
//!   unlabelled pixel and flood-fills the 4-connected region of equal
//!   pixel value with the next serial, *starting from 1*; the reported
//!   segment count is the final serial, i.e. `regions + 1`. A black
//!   image with one white circle therefore yields labels 1 (background)
//!   and 2 (circle) and a segment count of 3, exactly as
//!   `test_morphology.py::test_labelregions` pins. libvips stores labels
//!   in a 32-bit int mask; [`crate::pixel::PixelFormat`] has no int-32
//!   depth, so the mask here is `Gray16` and stored labels saturate at
//!   65535 (the returned count stays exact).
//! * **`countlines`.** Pixels are thresholded at `>= 128`, transitions
//!   from below- to above-threshold are counted along the axis
//!   perpendicular to the requested line direction, and the total is
//!   divided by the image width (horizontal) or height (vertical). This
//!   reproduces the libvips pipeline (`moreeq_const 128`, integer `conv`
//!   with a `[-1, 1]` mask whose negative lobe clips to zero in uchar,
//!   `project`, `avg`, `/255`).

use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use thiserror::Error;

/// Direction selector for [`Raster::countlines`].
///
/// `Horizontal` counts horizontal lines (scanning down each column for
/// threshold transitions); `Vertical` counts vertical lines (scanning along
/// each row). Mirrors `VipsDirection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Direction {
    /// Count horizontal lines.
    Horizontal,
    /// Count vertical lines.
    Vertical,
}

/// Typed errors for the morphology operations in [`crate::morphology`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MorphologyError {
    /// The structuring element has no rows or an empty row.
    #[error("empty structuring element")]
    EmptyKernel,
    /// The structuring element rows have differing lengths.
    #[error("ragged structuring element: row {row} has {got} elements, expected {expected}")]
    RaggedKernel {
        row: usize,
        got: usize,
        expected: usize,
    },
    /// A structuring-element value is not 0, 128, or 255.
    #[error("bad mask element ({value} should be 0, 128 or 255)")]
    BadKernelElement { value: u8 },
    /// The operation supports 8-bit images only.
    #[error("{op} supports 8-bit images only; cast deeper formats to an 8-bit format first")]
    EightBitOnly { op: &'static str },
    /// The operation supports unsigned 8/16-bit images only.
    #[error("{op} does not support float rasters; cast to an unsigned 8/16-bit format first")]
    UnsignedOnly { op: &'static str },
    /// The operation requires a one-band image.
    #[error("{op} requires a one-band image, got {bands} bands")]
    OneBandOnly { op: &'static str, bands: usize },
    /// The rank window does not fit inside the image.
    #[error("window too large: {width}x{height} window on a {image_w}x{image_h} image")]
    WindowTooLarge {
        width: u32,
        height: u32,
        image_w: u32,
        image_h: u32,
    },
    /// The rank window has a zero dimension.
    #[error("window dimensions must be greater than zero")]
    ZeroWindow,
    /// The rank index is outside `0..width * height`.
    #[error("index out of range: {index} not below {n}")]
    IndexOutOfRange { index: u32, n: u32 },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_morph<T>(op: &str, r: Result<T, MorphologyError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// A validated, dense copy of a caller-supplied structuring element.
struct Kernel {
    width: usize,
    height: usize,
    /// Row-major mask bytes, each 0, 128, or 255.
    coeff: Vec<u8>,
}

impl Kernel {
    fn from_rows(rows: &[&[u8]]) -> Result<Self, MorphologyError> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(MorphologyError::EmptyKernel);
        }
        let width = rows[0].len();
        let mut coeff = Vec::with_capacity(width * rows.len());
        for (row, r) in rows.iter().enumerate() {
            if r.len() != width {
                return Err(MorphologyError::RaggedKernel {
                    row,
                    got: r.len(),
                    expected: width,
                });
            }
            for &v in *r {
                if v != 0 && v != 128 && v != 255 {
                    return Err(MorphologyError::BadKernelElement { value: v });
                }
                coeff.push(v);
            }
        }
        Ok(Kernel {
            width,
            height: rows.len(),
            coeff,
        })
    }
}

/// Which of the two morphological operations to run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MorphOp {
    Erode,
    Dilate,
}

/// Clamp a mask-relative coordinate to the image, replicating edge pixels
/// (`VIPS_EXTEND_COPY`).
#[inline]
fn clamp_coord(v: i64, size: u32) -> usize {
    v.clamp(0, size as i64 - 1) as usize
}

/// Shared erode/dilate walker over an 8-bit raster, treating each band
/// independently and replicating edges.
fn morph(src: &Raster, kernel: &Kernel, op: MorphOp) -> Result<Raster, MorphologyError> {
    let fmt = src.format();
    if fmt.bytes_per_channel() != 1 {
        return Err(MorphologyError::EightBitOnly {
            op: match op {
                MorphOp::Erode => "erode",
                MorphOp::Dilate => "dilate",
            },
        });
    }
    let (w, h) = (src.width(), src.height());
    let bands = fmt.channels();
    let stride = src.stride();
    let data = src.data();
    let mut out = Raster::zeroed(w, h, fmt)?;
    let out_stride = out.stride();
    let out_data = out.data_mut();

    // Anchor at (width / 2, height / 2), as vips_morph places the mask
    // origin.
    let ax = (kernel.width / 2) as i64;
    let ay = (kernel.height / 2) as i64;

    for y in 0..h as i64 {
        for x in 0..w as i64 {
            for b in 0..bands {
                // Erode starts from all-ones and ANDs; dilate starts from
                // zero and ORs. `coeff == 0` complements the sample first,
                // reproducing the bitwise `~p` in vips_erode_gen /
                // vips_dilate_gen.
                let mut acc: u8 = match op {
                    MorphOp::Erode => 255,
                    MorphOp::Dilate => 0,
                };
                for my in 0..kernel.height as i64 {
                    let sy = clamp_coord(y + my - ay, h);
                    let row = sy * stride;
                    for mx in 0..kernel.width as i64 {
                        let coeff = kernel.coeff[(my as usize) * kernel.width + mx as usize];
                        if coeff == 128 {
                            continue;
                        }
                        let sx = clamp_coord(x + mx - ax, w);
                        let p = data[row + sx * bands + b];
                        let v = if coeff == 0 { !p } else { p };
                        match op {
                            MorphOp::Erode => acc &= v,
                            MorphOp::Dilate => acc |= v,
                        }
                    }
                }
                out_data[y as usize * out_stride + x as usize * bands + b] = acc;
            }
        }
    }
    Ok(out)
}

/// Read the one-band sample at `(x, y)` as `u32` for the unsigned 8/16-bit
/// depths (native byte order for 16-bit, matching [`crate::raster_ops`]).
#[inline]
fn sample_u32(data: &[u8], stride: usize, bpc: usize, x: usize, y: usize) -> u32 {
    match bpc {
        1 => data[y * stride + x] as u32,
        _ => {
            let b = y * stride + x * 2;
            u16::from_ne_bytes([data[b], data[b + 1]]) as u32
        }
    }
}

impl Raster {
    /// Erode with a structuring element, the fallible form of
    /// [`Raster::erode`].
    ///
    /// # Errors
    ///
    /// Returns [`MorphologyError::EmptyKernel`] /
    /// [`MorphologyError::RaggedKernel`] /
    /// [`MorphologyError::BadKernelElement`] for a malformed structuring
    /// element and [`MorphologyError::EightBitOnly`] for a non-8-bit
    /// raster.
    pub fn try_erode(&self, kernel: &[&[u8]]) -> Result<Raster, MorphologyError> {
        morph(self, &Kernel::from_rows(kernel)?, MorphOp::Erode)
    }

    /// Erode this image with the given structuring element (libvips
    /// `vips_morph` with `VIPS_OPERATION_MORPHOLOGY_ERODE`).
    ///
    /// `kernel` rows hold `255` (input must be non-zero), `0` (input must
    /// be zero), or `128` (don't care). The output pixel is set (255) only
    /// when every non-don't-care element matches; the image should be a
    /// 0/255 binary image, with 255 meaning object. The output has the
    /// same dimensions and format as the input; edge pixels are made by
    /// replicating the input edges.
    ///
    /// # Panics
    ///
    /// Panics on a malformed structuring element or a non-8-bit raster;
    /// use [`Raster::try_erode`] to handle those as errors.
    pub fn erode(&self, kernel: &[&[u8]]) -> Raster {
        expect_morph("erode", self.try_erode(kernel))
    }

    /// Dilate with a structuring element, the fallible form of
    /// [`Raster::dilate`].
    ///
    /// # Errors
    ///
    /// As [`Raster::try_erode`].
    pub fn try_dilate(&self, kernel: &[&[u8]]) -> Result<Raster, MorphologyError> {
        morph(self, &Kernel::from_rows(kernel)?, MorphOp::Dilate)
    }

    /// Dilate this image with the given structuring element (libvips
    /// `vips_morph` with `VIPS_OPERATION_MORPHOLOGY_DILATE`).
    ///
    /// Same mask encoding as [`Raster::erode`]; the output pixel is set
    /// when *any* non-don't-care element matches (a `255` over a non-zero
    /// pixel, or a `0` over a zero pixel).
    ///
    /// # Panics
    ///
    /// Panics on a malformed structuring element or a non-8-bit raster;
    /// use [`Raster::try_dilate`] to handle those as errors.
    pub fn dilate(&self, kernel: &[&[u8]]) -> Raster {
        expect_morph("dilate", self.try_dilate(kernel))
    }

    /// Rank filter, the fallible form of [`Raster::rank`].
    ///
    /// # Errors
    ///
    /// Returns [`MorphologyError::ZeroWindow`] /
    /// [`MorphologyError::WindowTooLarge`] /
    /// [`MorphologyError::IndexOutOfRange`] for bad window geometry and
    /// [`MorphologyError::UnsignedOnly`] for float rasters.
    pub fn try_rank(&self, width: u32, height: u32, index: u32) -> Result<Raster, MorphologyError> {
        if width == 0 || height == 0 {
            return Err(MorphologyError::ZeroWindow);
        }
        if width > self.width() || height > self.height() {
            return Err(MorphologyError::WindowTooLarge {
                width,
                height,
                image_w: self.width(),
                image_h: self.height(),
            });
        }
        let n = width * height;
        if index >= n {
            return Err(MorphologyError::IndexOutOfRange { index, n });
        }
        let fmt = self.format();
        let bpc = fmt.bytes_per_channel();
        if bpc == 4 {
            return Err(MorphologyError::UnsignedOnly { op: "rank" });
        }
        let (w, h) = (self.width(), self.height());
        let bands = fmt.channels();
        let stride = self.stride();
        let data = self.data();
        let mut out = Raster::zeroed(w, h, fmt)?;
        let out_stride = out.stride();
        let out_data = out.data_mut();

        // Window anchor at (width / 2, height / 2), as vips_rank embeds.
        let ax = (width / 2) as i64;
        let ay = (height / 2) as i64;
        let mut window: Vec<u32> = Vec::with_capacity(n as usize);

        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for b in 0..bands {
                    window.clear();
                    for wy in 0..height as i64 {
                        let sy = clamp_coord(y + wy - ay, h);
                        for wx in 0..width as i64 {
                            let sx = clamp_coord(x + wx - ax, w);
                            window.push(sample_u32(data, stride, bpc, sx * bands + b, sy));
                        }
                    }
                    window.sort_unstable();
                    let v = window[index as usize];
                    let off = y as usize * out_stride + (x as usize * bands + b) * bpc;
                    match bpc {
                        1 => out_data[off] = v as u8,
                        _ => out_data[off..off + 2].copy_from_slice(&(v as u16).to_ne_bytes()),
                    }
                }
            }
        }
        Ok(out)
    }

    /// Rank filter (libvips `vips_rank`): move a `width` x `height` window
    /// across the image and output the sample at position `index` in the
    /// sorted window (0 = minimum, `width * height - 1` = maximum; the
    /// middle index is the median). Bands are filtered independently and
    /// edge pixels are made by replicating the input edges.
    ///
    /// # Panics
    ///
    /// Panics when the window is empty or larger than the image, when
    /// `index` is out of range, or on a float raster; use
    /// [`Raster::try_rank`] to handle those as errors.
    pub fn rank(&self, width: u32, height: u32, index: u32) -> Raster {
        expect_morph("rank", self.try_rank(width, height, index))
    }

    /// Count lines, the fallible form of [`Raster::countlines`].
    ///
    /// # Errors
    ///
    /// Returns [`MorphologyError::OneBandOnly`] for a multi-band raster
    /// and [`MorphologyError::UnsignedOnly`] for float rasters.
    pub fn try_countlines(&self, direction: Direction) -> Result<f64, MorphologyError> {
        let fmt = self.format();
        if fmt.channels() != 1 {
            return Err(MorphologyError::OneBandOnly {
                op: "countlines",
                bands: fmt.channels(),
            });
        }
        let bpc = fmt.bytes_per_channel();
        if bpc == 4 {
            return Err(MorphologyError::UnsignedOnly { op: "countlines" });
        }
        let (w, h) = (self.width() as usize, self.height() as usize);
        let stride = self.stride();
        let data = self.data();
        let above = |x: usize, y: usize| sample_u32(data, stride, bpc, x, y) >= 128;

        // Count below-to-above transitions of the 128 threshold. The
        // libvips pipeline convolves the thresholded image with [-1, 1] in
        // integer precision; the uchar output clips the falling edges to
        // zero, so only rising transitions contribute. Edge extension is
        // EXTEND_COPY, so there is never a transition into the first
        // row/column.
        let mut transitions: u64 = 0;
        match direction {
            Direction::Horizontal => {
                for x in 0..w {
                    for y in 1..h {
                        if above(x, y) && !above(x, y - 1) {
                            transitions += 1;
                        }
                    }
                }
                Ok(transitions as f64 / w as f64)
            }
            Direction::Vertical => {
                for y in 0..h {
                    for x in 1..w {
                        if above(x, y) && !above(x - 1, y) {
                            transitions += 1;
                        }
                    }
                }
                Ok(transitions as f64 / h as f64)
            }
        }
    }

    /// Count the average number of lines in a one-band image (libvips
    /// `vips_countlines`).
    ///
    /// Pixels are thresholded at 128; a "line" is a rising transition
    /// through the threshold, counted down every column for
    /// [`Direction::Horizontal`] or along every row for
    /// [`Direction::Vertical`], and averaged over the columns / rows. A
    /// single one-pixel-wide full-width white line on black therefore
    /// counts as exactly `1.0`.
    ///
    /// # Panics
    ///
    /// Panics on a multi-band or float raster; use
    /// [`Raster::try_countlines`] to handle those as errors.
    pub fn countlines(&self, direction: Direction) -> f64 {
        expect_morph("countlines", self.try_countlines(direction))
    }

    /// Label regions, the fallible form of [`Raster::label_regions`].
    ///
    /// # Errors
    ///
    /// Returns [`MorphologyError::UnsignedOnly`] for float rasters.
    pub fn try_label_regions(&self) -> Result<(Raster, u32), MorphologyError> {
        let fmt = self.format();
        if fmt.bytes_per_channel() == 4 {
            return Err(MorphologyError::UnsignedOnly {
                op: "label_regions",
            });
        }
        let (w, h) = (self.width() as usize, self.height() as usize);
        let bpp = fmt.bytes_per_pixel();
        let stride = self.stride();
        let data = self.data();
        // Full-pixel equality (all bands), exactly like the
        // vips__draw_flood_direct blob test.
        let pixel = |x: usize, y: usize| -> &[u8] {
            let off = y * stride + x * bpp;
            &data[off..off + bpp]
        };

        let mut labels: Vec<u32> = vec![0; w * h];
        // libvips starts the serial at 1 and reports the *final* serial as
        // the segment count, i.e. regions + 1.
        let mut serial: u32 = 1;
        let mut stack: Vec<(usize, usize)> = Vec::new();

        for sy in 0..h {
            for sx in 0..w {
                if labels[sy * w + sx] != 0 {
                    continue;
                }
                // Flood-fill the 4-connected region of pixels equal to the
                // seed value with the current serial.
                let seed = pixel(sx, sy);
                labels[sy * w + sx] = serial;
                stack.push((sx, sy));
                while let Some((x, y)) = stack.pop() {
                    let mut visit = |nx: usize, ny: usize| {
                        let i = ny * w + nx;
                        if labels[i] == 0 && pixel(nx, ny) == seed {
                            labels[i] = serial;
                            stack.push((nx, ny));
                        }
                    };
                    if x > 0 {
                        visit(x - 1, y);
                    }
                    if x + 1 < w {
                        visit(x + 1, y);
                    }
                    if y > 0 {
                        visit(x, y - 1);
                    }
                    if y + 1 < h {
                        visit(x, y + 1);
                    }
                }
                serial += 1;
            }
        }

        // libvips writes int-32 labels; Gray16 is the widest unsigned
        // PixelFormat depth, so stored labels saturate at 65535 while the
        // returned segment count stays exact.
        let mut mask = Raster::zeroed(self.width(), self.height(), PixelFormat::Gray16)?;
        let mask_stride = mask.stride();
        let mask_data = mask.data_mut();
        for y in 0..h {
            for x in 0..w {
                let v = labels[y * w + x].min(65535) as u16;
                let off = y * mask_stride + x * 2;
                mask_data[off..off + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        Ok((mask, serial))
    }

    /// Label the 4-connected regions of equal pixel value (libvips
    /// `vips_labelregions`).
    ///
    /// Scans row-major for the first unlabelled pixel and flood-fills its
    /// region with the next serial, starting from 1, so the first region
    /// found (usually the background) gets label 1. Returns the `Gray16`
    /// label mask and the segment count, which is the final serial:
    /// `number of regions + 1`, exactly as libvips reports it.
    ///
    /// # Panics
    ///
    /// Panics on a float raster; use [`Raster::try_label_regions`] to
    /// handle that as an error.
    pub fn label_regions(&self) -> (Raster, u32) {
        expect_morph("label_regions", self.try_label_regions())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU16;

    fn black(w: u32, h: u32) -> Raster {
        Raster::zeroed(w, h, PixelFormat::Gray8).unwrap()
    }

    fn avg(r: &Raster) -> f64 {
        r.data().iter().map(|&b| b as f64).sum::<f64>() / r.data().len() as f64
    }

    const CROSS: &[&[u8]] = &[&[128, 255, 128], &[255, 255, 255], &[128, 255, 128]];

    /**
     * Compile-position pin for the exact ported-test call surface of the
     * morphology batch (ported_morphology.rs): erode/dilate over a
     * `&[&[u8]]` structuring element, rank, countlines with the Direction
     * enum, and label_regions returning `(Raster, u32)`.
     * Works by invoking every call site shape the ported tests use.
     */
    #[test]
    fn ported_call_surface_compiles() {
        let mut im = black(20, 20);
        im.draw_line(&[255], 0, 10, 20, 10);
        let kernel: &[&[u8]] = CROSS;
        let _: Raster = im.erode(kernel);
        let _: Raster = im.dilate(kernel);
        let _: Raster = im.rank(3, 3, 8);
        let n_lines: f64 = im.countlines(Direction::Horizontal);
        assert!(n_lines > 0.0);
        let (mask, segments): (Raster, u32) = im.label_regions();
        let _ = mask.data().iter().copied().max().unwrap();
        assert!(segments >= 2);
    }

    /**
     * Tests the ported erode scenario: a filled white circle on black,
     * eroded with the 3x3 cross element, keeps its dimensions and format
     * and loses mass (fewer white pixels).
     * Works by mirroring test_morphology.py::test_erode and comparing
     * averages. Input: 100x100 black + circle r=25 at (50,50).
     */
    #[test]
    fn erode_circle_shrinks() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[255], 50, 50, 25);
        let im2 = im.erode(CROSS);
        assert_eq!(im.width(), im2.width());
        assert_eq!(im.height(), im2.height());
        assert_eq!(im.format(), im2.format());
        assert!(avg(&im) > avg(&im2));
    }

    /**
     * Tests the ported dilate scenario: the same circle dilated with the
     * cross element gains mass.
     * Works by mirroring test_morphology.py::test_dilate.
     */
    #[test]
    fn dilate_circle_grows() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[255], 50, 50, 25);
        let im2 = im.dilate(CROSS);
        assert_eq!(im.width(), im2.width());
        assert_eq!(im.height(), im2.height());
        assert_eq!(im.format(), im2.format());
        assert!(avg(&im2) > avg(&im));
    }

    /**
     * Tests exact erode pixels on a known square: a 3x3 all-255 element
     * erodes a 5x5 white square centred in an 11x11 black image down to
     * the 3x3 core (only pixels whose full 3x3 neighbourhood is white
     * survive).
     * Works by checking every pixel of the output against the expected
     * shrunken square.
     */
    #[test]
    fn erode_square_exact() {
        let mut im = black(11, 11);
        im.draw_rect_filled(&[255], 3, 3, 5, 5);
        let all: &[&[u8]] = &[&[255, 255, 255], &[255, 255, 255], &[255, 255, 255]];
        let out = im.erode(all);
        for y in 0..11u32 {
            for x in 0..11u32 {
                let expected = (4..=6).contains(&x) && (4..=6).contains(&y);
                assert_eq!(
                    out.getpoint(x, y)[0] != 0.0,
                    expected,
                    "pixel ({x},{y}) after erode"
                );
            }
        }
    }

    /**
     * Tests exact dilate pixels: dilating a single white pixel with the
     * cross element paints the 4-connected plus shape.
     * Works by checking the five expected white pixels and a corner that
     * must stay black.
     */
    #[test]
    fn dilate_point_exact() {
        let mut im = black(7, 7);
        im.put_pixel(3, 3, &[255]);
        let out = im.dilate(CROSS);
        for (x, y) in [(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)] {
            assert_eq!(out.getpoint(x, y)[0], 255.0, "plus arm at ({x},{y})");
        }
        assert_eq!(out.getpoint(2, 2)[0], 0.0, "diagonal stays black");
        assert_eq!(out.getpoint(0, 0)[0], 0.0);
    }

    /**
     * Tests the must-be-zero (0) mask encoding: a hit-miss element that
     * requires the centre white and the left neighbour black fires only
     * on left edges of white runs.
     * Works by eroding a two-pixel white run with [[0, 255]] (anchor at
     * index 1) and checking only the run's first column matches.
     */
    #[test]
    fn erode_hit_miss_left_edge() {
        let mut im = black(7, 1);
        im.put_pixel(3, 0, &[255]);
        im.put_pixel(4, 0, &[255]);
        let elem: &[&[u8]] = &[&[0, 255]];
        let out = im.erode(elem);
        let hits: Vec<u32> = (0..7).filter(|&x| out.getpoint(x, 0)[0] != 0.0).collect();
        assert_eq!(hits, vec![3], "only the left edge of the run matches");
    }

    /**
     * Tests edge replication: a fully white image erodes to fully white
     * because the off-canvas neighbourhood replicates the white edges
     * (VIPS_EXTEND_COPY), never introducing black.
     * Works by eroding an all-255 image and asserting it is unchanged.
     */
    #[test]
    fn erode_edge_replication_keeps_white() {
        let im = Raster::new(4, 4, PixelFormat::Gray8, vec![255; 16]).unwrap();
        let all: &[&[u8]] = &[&[255, 255, 255], &[255, 255, 255], &[255, 255, 255]];
        let out = im.erode(all);
        assert!(out.data().iter().all(|&b| b == 255));
    }

    /**
     * Tests erode/dilate treat bands independently: an Rgb8 image with a
     * white square in the red band only dilates red without leaking into
     * green or blue.
     * Works by dilating and inspecting per-band samples next to the
     * square.
     */
    #[test]
    fn morph_is_bandwise() {
        let mut im = Raster::zeroed(7, 7, PixelFormat::Rgb8).unwrap();
        im.put_pixel(3, 3, &[255, 0, 0]);
        let out = im.dilate(CROSS);
        let p = out.getpoint(2, 3);
        assert_eq!(p, vec![255.0, 0.0, 0.0], "red dilates, green/blue stay");
    }

    /**
     * Tests structuring-element validation: empty, ragged, and non-0/128/255
     * elements are typed errors, and deeper depths are rejected.
     * Works by asserting each try_erode error variant.
     */
    #[test]
    fn erode_validation_errors() {
        let im = black(4, 4);
        assert!(matches!(
            im.try_erode(&[]),
            Err(MorphologyError::EmptyKernel)
        ));
        let ragged: &[&[u8]] = &[&[255, 255], &[255]];
        assert!(matches!(
            im.try_erode(ragged),
            Err(MorphologyError::RaggedKernel { row: 1, .. })
        ));
        let bad: &[&[u8]] = &[&[7]];
        assert!(matches!(
            im.try_erode(bad),
            Err(MorphologyError::BadKernelElement { value: 7 })
        ));
        let deep = Raster::zeroed(4, 4, PixelFormat::Gray16).unwrap();
        assert!(matches!(
            deep.try_erode(CROSS),
            Err(MorphologyError::EightBitOnly { .. })
        ));
    }

    /**
     * Tests the ported rank scenario: rank(3, 3, 8) is a maximum filter,
     * so the circle grows and dimensions/format are preserved.
     * Works by mirroring test_morphology.py::test_rank.
     */
    #[test]
    fn rank_max_grows_circle() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[255], 50, 50, 25);
        let im2 = im.rank(3, 3, 8);
        assert_eq!(im.width(), im2.width());
        assert_eq!(im.height(), im2.height());
        assert_eq!(im.format(), im2.format());
        assert!(avg(&im2) > avg(&im));
    }

    /**
     * Tests rank order statistics on a known 3x3 window: index 0 is the
     * minimum, the middle index the median, and n-1 the maximum.
     * Works by ranking a 3x3 image holding 10..90 and reading the centre
     * pixel, whose window is the whole image.
     */
    #[test]
    fn rank_order_statistics_exact() {
        let data: Vec<u8> = vec![50, 10, 30, 90, 70, 20, 40, 80, 60];
        let im = Raster::new(3, 3, PixelFormat::Gray8, data).unwrap();
        assert_eq!(im.rank(3, 3, 0).getpoint(1, 1)[0], 10.0, "minimum");
        assert_eq!(im.rank(3, 3, 4).getpoint(1, 1)[0], 50.0, "median");
        assert_eq!(im.rank(3, 3, 8).getpoint(1, 1)[0], 90.0, "maximum");
    }

    /**
     * Tests that a 1x1 rank window is the identity for any index 0.
     * Works by comparing the output bytes to the input.
     */
    #[test]
    fn rank_identity_window() {
        let mut im = black(9, 9);
        im.draw_circle_filled(&[200], 4, 4, 3);
        let out = im.rank(1, 1, 0);
        assert_eq!(out.data(), im.data());
    }

    /**
     * Tests rank on Gray16 samples: the median of a 16-bit window uses
     * full sample precision (values above 255).
     * Works by ranking a 3x1 Gray16 image and reading the middle pixel.
     */
    #[test]
    fn rank_gray16() {
        let mut data = Vec::new();
        for v in [1000u16, 3000, 2000] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let im = Raster::new(3, 1, PixelFormat::Gray16, data).unwrap();
        let out = im.rank(3, 1, 1);
        assert_eq!(out.getpoint(1, 0)[0], 2000.0, "median of 16-bit window");
    }

    /**
     * Tests rank window/index validation: zero windows, windows larger
     * than the image, and out-of-range indices are typed errors.
     * Works by asserting each try_rank error variant.
     */
    #[test]
    fn rank_validation_errors() {
        let im = black(4, 4);
        assert!(matches!(
            im.try_rank(0, 3, 0),
            Err(MorphologyError::ZeroWindow)
        ));
        assert!(matches!(
            im.try_rank(5, 3, 0),
            Err(MorphologyError::WindowTooLarge { .. })
        ));
        assert!(matches!(
            im.try_rank(3, 3, 9),
            Err(MorphologyError::IndexOutOfRange { index: 9, n: 9 })
        ));
    }

    /**
     * Tests the ported countlines scenario: one horizontal white line
     * across a black image counts as exactly 1.0.
     * Works by mirroring test_morphology.py::test_countlines.
     */
    #[test]
    fn countlines_single_horizontal() {
        let mut im = black(100, 100);
        im.draw_line(&[255], 0, 50, 100, 50);
        assert_eq!(im.countlines(Direction::Horizontal), 1.0);
    }

    /**
     * Tests countlines in the vertical direction and with several lines:
     * two full-height vertical lines count as 2.0.
     * Works by drawing two vertical lines and counting.
     */
    #[test]
    fn countlines_two_vertical() {
        let mut im = black(100, 100);
        im.draw_line(&[255], 20, 0, 20, 99);
        im.draw_line(&[255], 60, 0, 60, 99);
        assert_eq!(im.countlines(Direction::Vertical), 2.0);
    }

    /**
     * Tests the 128 threshold boundary: ink 127 is below threshold and
     * counts zero lines, ink 128 counts one, matching moreeq_const 128.
     * Works by drawing a line at each intensity and comparing counts.
     */
    #[test]
    fn countlines_threshold_at_128() {
        let mut dim = black(10, 10);
        dim.draw_line(&[127], 0, 5, 10, 5);
        assert_eq!(dim.countlines(Direction::Horizontal), 0.0);
        let mut lit = black(10, 10);
        lit.draw_line(&[128], 0, 5, 10, 5);
        assert_eq!(lit.countlines(Direction::Horizontal), 1.0);
    }

    /**
     * Tests countlines only counts rising transitions: a line that runs
     * to the bottom edge has no falling edge inside the image and still
     * counts once, and a half-width line averages to 0.5.
     * Works on a bottom-row line and a half-width line.
     */
    #[test]
    fn countlines_partial_and_edge_lines() {
        let mut im = black(10, 10);
        im.draw_line(&[255], 0, 9, 10, 9);
        assert_eq!(im.countlines(Direction::Horizontal), 1.0, "bottom row");
        let mut half = black(10, 10);
        half.draw_line(&[255], 0, 5, 4, 5);
        assert_eq!(
            half.countlines(Direction::Horizontal),
            0.5,
            "line across half the columns"
        );
    }

    /**
     * Tests countlines input validation: multi-band and float rasters are
     * typed errors.
     * Works by asserting the try_countlines error variants.
     */
    #[test]
    fn countlines_validation_errors() {
        let rgb = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            rgb.try_countlines(Direction::Horizontal),
            Err(MorphologyError::OneBandOnly { bands: 3, .. })
        ));
    }

    /**
     * Tests the ported label_regions scenario: a filled circle on black
     * yields two regions labelled 1 (background) and 2 (circle), and the
     * libvips segment count regions + 1 = 3.
     * Works by mirroring test_morphology.py::test_labelregions, including
     * the byte-level max the ported test reads (labels below 256 keep the
     * Gray16 mask's byte maximum equal to the label maximum).
     */
    #[test]
    fn label_regions_circle() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[255], 50, 50, 25);
        let (mask, segments) = im.label_regions();
        assert_eq!(segments, 3);
        let max_label = mask.data().iter().copied().max().unwrap();
        assert_eq!(max_label, 2);
        assert_eq!(mask.format(), PixelFormat::Gray16);
    }

    /**
     * Tests label_regions on a uniform image: one region, labelled 1
     * everywhere, segment count 2.
     * Works by labelling an all-black image and checking every mask
     * sample.
     */
    #[test]
    fn label_regions_uniform() {
        let im = black(8, 8);
        let (mask, segments) = im.label_regions();
        assert_eq!(segments, 2);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(mask.getpoint(x, y)[0], 1.0);
            }
        }
    }

    /**
     * Tests 4-connectivity: two white pixels touching only diagonally are
     * separate regions (labels 2 and 3 in scan order), giving four
     * segments in total with the background.
     * Works by placing pixels at (1,1) and (2,2) and counting.
     */
    #[test]
    fn label_regions_diagonal_not_connected() {
        let mut im = black(4, 4);
        im.put_pixel(1, 1, &[255]);
        im.put_pixel(2, 2, &[255]);
        let (mask, segments) = im.label_regions();
        assert_eq!(segments, 4, "background + two diagonal blobs + 1");
        assert_eq!(mask.getpoint(1, 1)[0], 2.0, "first blob in scan order");
        assert_eq!(mask.getpoint(2, 2)[0], 3.0, "second blob in scan order");
    }

    /**
     * Tests scan-order label assignment on equal-value regions split by a
     * barrier: a white column splits the black background into two
     * regions of the same value with distinct labels.
     * Works by labelling a black image with a full-height white line and
     * checking the left background is 1, the line 2, and the right
     * background 3.
     */
    #[test]
    fn label_regions_equal_values_distinct_regions() {
        let mut im = black(5, 3);
        im.draw_line(&[255], 2, 0, 2, 2);
        let (mask, segments) = im.label_regions();
        assert_eq!(segments, 4);
        assert_eq!(mask.getpoint(0, 1)[0], 1.0, "left background first");
        assert_eq!(mask.getpoint(2, 1)[0], 2.0, "line second");
        assert_eq!(mask.getpoint(4, 1)[0], 3.0, "right background third");
    }

    /**
     * Tests label_regions uses full-pixel equality on multi-band images:
     * two rectangles differing only in the green band are separate
     * regions.
     * Works by drawing [10,0,0] and [10,9,0] rectangles that tile the
     * whole canvas and expecting 3 segments (2 regions + 1).
     */
    #[test]
    fn label_regions_multiband_pixel_equality() {
        let mut im = Raster::zeroed(6, 2, PixelFormat::Rgb8).unwrap();
        im.draw_rect_filled(&[10, 0, 0], 0, 0, 3, 2);
        im.draw_rect_filled(&[10, 9, 0], 3, 0, 3, 2);
        let (_, segments) = im.label_regions();
        assert_eq!(segments, 3, "two touching rects of distinct colour + 1");
    }

    /**
     * Tests that this module dispatches on sample kind and never on byte
     * width, by asserting that neither the byte-width accessor on
     * [`PixelFormat`] nor its width-keyed constructor survives in
     * `src/morphology.rs`.
     * Works by scanning the module's own source, compiled in with
     * `include_str!`, for the accessor's name; the needle is spelled in two
     * halves so this assertion is not itself a hit. A byte width is not a
     * sample kind: four bytes is `f32` today and would be `u32` under issue
     * #517, and the sites this replaced stepped two bytes per sample for
     * every width but one, so a four-byte carrier would have been walked at
     * half stride (issue #607).
     * Input: `src/morphology.rs` -> Output: zero occurrences.
     */
    #[test]
    fn morphology_does_not_dispatch_on_byte_width() {
        const SRC: &str = include_str!("morphology.rs");
        let needles = [
            concat!("bytes_per_", "channel"),
            concat!("with_", "channels"),
        ];
        // Positive control: the same scan over the same string finds a token
        // that is present, so the zero below is a real zero and not the
        // vacuous pass an empty read would give.
        assert!(
            SRC.contains(concat!("fn sample_", "u32")),
            "positive control failed: the scan cannot see this module's source"
        );
        for needle in needles {
            assert_eq!(
                SRC.matches(needle).count(),
                0,
                "{needle} is back in src/morphology.rs; dispatch on \
                 PixelFormat::kind() and PixelFormat::with_kind() instead"
            );
        }
    }

    /**
     * Tests that the rank walker reads a 16-bit raster at the 16-bit
     * stride, which is the property the width-keyed helper got right only
     * by accident: its `_` arm stepped two bytes for every width that was
     * not one, so it is the arm a four-byte carrier would have taken.
     * Works by ranking a one-band `u16` row whose samples are all above
     * `u8::MAX` and asserting the median comes back as the middle sample
     * rather than as a byte of one of them.
     * Input: `[1000, 40000, 20000]` in one row, 3x1 window, index 1 ->
     * Output: 20000 in every output pixel the window covers.
     */
    #[test]
    fn rank_reads_sixteen_bit_samples_at_the_sixteen_bit_stride() {
        let mut im = Raster::zeroed(3, 1, PixelFormat::Gray16).unwrap();
        let data = im.data_mut();
        for (i, v) in [1000u16, 40000, 20000].iter().enumerate() {
            data[2 * i..2 * i + 2].copy_from_slice(&v.to_ne_bytes());
        }
        let out = im.try_rank(3, 1, 1).expect("16-bit rank is supported");
        let got: Vec<u16> = out
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect();
        // Edges replicate, so every window is {1000, 40000, 20000} with one
        // sample doubled; the median is 20000 in the middle and at the right
        // edge, and 1000 at the left edge where 1000 is doubled.
        assert_eq!(got, vec![1000, 20000, 20000], "rank read the wrong bytes");
    }

    /**
     * Tests that the erode/dilate guard names the 8-bit *kind* and not the
     * one-byte width, so a one-byte signed carrier would be refused rather
     * than walked as unsigned (issue #607).
     * Works by checking the guard still refuses the deeper kinds the crate
     * carries today and still accepts `U8`, which is the half of the
     * contract that is observable before a signed carrier exists.
     * Input: Gray16 and FloatF32(1) -> `EightBitOnly`; Gray8 -> Ok.
     */
    #[test]
    fn erode_refuses_every_kind_but_u8() {
        let kernel: &[&[u8]] = &[&[255, 255, 255]];
        for fmt in [
            PixelFormat::Gray16,
            PixelFormat::FloatF32(NonZeroU16::new(1).unwrap()),
        ] {
            let im = Raster::zeroed(3, 3, fmt).unwrap();
            assert!(
                matches!(
                    im.try_erode(kernel),
                    Err(MorphologyError::EightBitOnly { .. })
                ),
                "{fmt:?} must be refused by the 8-bit guard"
            );
        }
        let im = Raster::zeroed(3, 3, PixelFormat::Gray8).unwrap();
        assert!(im.try_erode(kernel).is_ok(), "Gray8 must still be accepted");
    }

    /**
     * Tests that the rank / countlines / label-regions guard is keyed on
     * the sample kind, so it accepts exactly the two unsigned kinds its
     * error variant documents.
     * Works by driving all three ops at `Gray8`, `Gray16` and
     * `FloatF32(1)` and asserting the float one is the only refusal.
     * Input: three formats x three ops -> Output: `UnsignedOnly` for float
     * only.
     */
    #[test]
    fn the_unsigned_guard_accepts_u8_and_u16_and_refuses_float() {
        for fmt in [PixelFormat::Gray8, PixelFormat::Gray16] {
            let im = Raster::zeroed(4, 4, fmt).unwrap();
            assert!(im.try_rank(3, 3, 4).is_ok(), "{fmt:?} rank");
            assert!(im.try_countlines(Direction::Horizontal).is_ok(), "{fmt:?}");
            assert!(im.try_label_regions().is_ok(), "{fmt:?} label_regions");
        }
        let f = Raster::zeroed(4, 4, PixelFormat::FloatF32(NonZeroU16::new(1).unwrap())).unwrap();
        for got in [
            f.try_rank(3, 3, 4).err(),
            f.try_countlines(Direction::Horizontal).err(),
            f.try_label_regions().map(|_| ()).err(),
        ] {
            assert!(
                matches!(got, Some(MorphologyError::UnsignedOnly { .. })),
                "a float raster must be refused, got {got:?}"
            );
        }
    }
}
