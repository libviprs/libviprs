//! Matrix-image and LUT-inversion operations ported from libvips.
//!
//! A matrix image is a small one-band image holding dense numeric data,
//! produced by [`Raster::from_matrix`] and stamped
//! [`Interpretation::Matrix`]. This module carries the operations
//! libviprs offers on them:
//!
//! | libviprs | libvips | notes |
//! |---|---|---|
//! | [`Raster::matrixinvert`] | `vips_matrixinvert` | dense inverse of a square matrix image |
//! | [`Raster::matrixmultiply`] | `vips_matrixmultiply` | dense product of two matrix images |
//! | [`Raster::invertlut`] | `vips_invertlut` | inverse LUT from measured `(x, f(x))` rows |
//!
//! [`Raster::buildlut`] (the forward companion of `invertlut`) already
//! lives in [`crate::create`] with the other generators.
//!
//! # Semantics
//!
//! * **`matrixinvert`** is a faithful port of libvips
//!   `mosaicing/matrixinvert.c`: matrices below 4x4 invert through the
//!   direct cofactor formulas, larger ones through the Numerical
//!   Recipes PLU decomposition with implicit (scaled) partial pivoting,
//!   solving one unit vector per column. A zero row, or a pivot smaller
//!   than `2 * DBL_MIN`, is a typed [`MatrixError::Singular`] error,
//!   matching the C thresholds exactly.
//! * **`matrixmultiply`** is a faithful port of libvips
//!   `mosaicing/matrixmultiply.c:105-127`: both operands go through the
//!   `vips_check_matrix` gate, the left matrix's width has to equal the
//!   right matrix's height (`matrixmultiply.c:90` raises `"bad sizes"`
//!   otherwise, here [`MatrixError::ShapeMismatch`]), and the output is
//!   a `right.width` x `left.height` matrix image whose every element
//!   is the plain `f64` dot product of a left row with a right column,
//!   accumulated in the C loop's `i`-`j`-`k` order so the per-cell
//!   summation order matches. There is no scale and no offset:
//!   `vips_matrixmultiply`'s own docs say the scale and offset members
//!   of both inputs are ignored, and libviprs matrices carry neither.
//!   The output's two dimensions come from two *independent* operands,
//!   so it can be far larger than either input; it is sized on the
//!   budget-checked fallible path before any of it is committed.
//! * **`invertlut`** is a faithful port of libvips `create/invertlut.c`:
//!   rows of the input matrix are measured points, column 0 the input
//!   level and each further column one band's measured response, all in
//!   `0..=1`. Rows are sorted by column 0; each output band linearly
//!   interpolates between the bracketing measured rows, extrapolating
//!   the head to `(0, 0)` and the tail to `(1, 1)`. The output is a
//!   `size` x 1 image (default 256) with one band per measured column,
//!   stamped [`Interpretation::Histogram`].
//! * **Precision.** All three operations compute in `f64` and store
//!   results in [`PixelFormat::FloatF32`] rasters (libvips stores
//!   `double`; libviprs carries float data as `f32`, the same trade
//!   documented by [`crate::create`] for `from_matrix` itself).

use std::num::NonZeroU16;

use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::raster_ops::sample_f64;

/// The `vips_check_matrix` size limit: matrix images wider or taller
/// than this are rejected.
const MAX_MATRIX_SIZE: u32 = 100_000;

/// libvips `matrixinvert.c` TOO_SMALL: twice the smallest normalised
/// double. Pivots below this magnitude count as singular.
const TOO_SMALL: f64 = 2.0 * f64::MIN_POSITIVE;

/// Typed errors for the matrix operations in [`crate::matrix`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MatrixError {
    /// The input is not a matrix image: matrices are one-band.
    #[error("{op} requires a one-band matrix image, got {bands} bands")]
    NotOneBand {
        /// The operation that failed.
        op: &'static str,
        /// The offending band count.
        bands: usize,
    },
    /// The input exceeds the libvips `vips_check_matrix` size limit.
    #[error("{op}: matrix is too large: {width}x{height} (limit {MAX_MATRIX_SIZE})")]
    TooLarge {
        /// The operation that failed.
        op: &'static str,
        /// Matrix width.
        width: u32,
        /// Matrix height.
        height: u32,
    },
    /// `matrixinvert` was given a non-square matrix.
    #[error("matrixinvert: non-square matrix ({width}x{height})")]
    NotSquare {
        /// Matrix width.
        width: u32,
        /// Matrix height.
        height: u32,
    },
    /// The matrix is singular or near-singular: a row of zeros, or a
    /// pivot below the libvips `TOO_SMALL` threshold.
    #[error("matrixinvert: singular or near-singular matrix")]
    Singular,
    /// `matrixmultiply` was given operands whose shapes do not chain:
    /// the left matrix's width has to equal the right matrix's height.
    #[error(
        "matrixmultiply: shape mismatch: left is {left_w}x{left_h}, \
         right is {right_w}x{right_h}; left width must equal right height"
    )]
    ShapeMismatch {
        /// Width of the left matrix.
        left_w: u32,
        /// Height of the left matrix.
        left_h: u32,
        /// Width of the right matrix.
        right_w: u32,
        /// Height of the right matrix.
        right_h: u32,
    },
    /// `invertlut` needs an input column plus at least one measured
    /// column.
    #[error("invertlut: bad input matrix: need at least two columns, got {columns}")]
    TooFewColumns {
        /// The offending column count.
        columns: u32,
    },
    /// An `invertlut` matrix element is outside `0..=1` (NaN included;
    /// libvips checks the same range).
    #[error("invertlut: element ({column}, {row}) is {value}, outside range [0,1]")]
    OutOfRange {
        /// Zero-based column of the offending element.
        column: u32,
        /// Zero-based row of the offending element.
        row: u32,
        /// The offending value.
        value: f64,
    },
    /// The requested `invertlut` LUT size is outside the libvips range
    /// `1..=65536`.
    #[error("invertlut: bad size {size} (expected 1..=65536)")]
    BadSize {
        /// The offending size.
        size: u32,
    },
    /// An underlying raster allocation failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Panic with the standard message shape for the panicking wrappers.
///
/// Every [`MatrixError`] variant except the transparent
/// [`MatrixError::Raster`] tail already opens with the failing
/// operation, either through its own `op` field
/// ([`MatrixError::NotOneBand`], [`MatrixError::TooLarge`]) or
/// hardcoded in its `Display` ([`MatrixError::ShapeMismatch`] and the
/// rest), so prefixing here as well doubles the name
/// ("matrixmultiply: matrixmultiply: shape mismatch ..."). That is the
/// #339 class `expect_arith` in [`crate::arithmetic`] already fixed.
/// Only the `Raster` tail, whose `Display` is the [`RasterError`]
/// message verbatim and names no operation, takes the prefix.
#[track_caller]
fn expect_matrix<T>(op: &str, r: Result<T, MatrixError>) -> T {
    match r {
        Ok(v) => v,
        Err(e @ MatrixError::Raster(_)) => panic!("{op}: {e}"),
        Err(e) => panic!("{e}"),
    }
}

/// The `vips_check_matrix` gate: one band, within the size limit.
/// Returns the matrix as rows of `f64`.
fn check_matrix(op: &'static str, r: &Raster) -> Result<Vec<Vec<f64>>, MatrixError> {
    let bands = r.format().channels();
    if bands != 1 {
        return Err(MatrixError::NotOneBand { op, bands });
    }
    if r.width() > MAX_MATRIX_SIZE || r.height() > MAX_MATRIX_SIZE {
        return Err(MatrixError::TooLarge {
            op,
            width: r.width(),
            height: r.height(),
        });
    }
    Ok((0..r.height())
        .map(|y| (0..r.width()).map(|x| r.getpoint(x, y)[0]).collect())
        .collect())
}

/// The `vips_check_matrix` gate again, returning the matrix as one flat
/// row-major `Vec<f64>` instead of a `Vec<Vec<f64>>`.
///
/// Same gate, same order, same errors as [`check_matrix`]; only the
/// shape of the result differs. `matrixmultiply` reads through this one
/// because its kernel indexes both operands linearly: a row-of-rows
/// costs one heap allocation per matrix row plus a pointer chase per
/// element, and [`Raster::getpoint`] allocates a whole `Vec<f64>` per
/// element only to index `[0]`. Reading the samples flat through
/// [`sample_f64`] is bit-for-bit the same values (`getpoint` decodes
/// through `sample_f64` too) and an order of magnitude faster
/// end-to-end.
///
/// The two readers share no gate on purpose for now: hoisting one is
/// the deferred sample-reader cleanup, and `matrixinvert` / `invertlut`
/// index by row, so they keep the nested form.
fn check_matrix_flat(op: &'static str, r: &Raster) -> Result<Vec<f64>, MatrixError> {
    let fmt = r.format();
    let bands = fmt.channels();
    if bands != 1 {
        return Err(MatrixError::NotOneBand { op, bands });
    }
    if r.width() > MAX_MATRIX_SIZE || r.height() > MAX_MATRIX_SIZE {
        return Err(MatrixError::TooLarge {
            op,
            width: r.width(),
            height: r.height(),
        });
    }
    // The buffer is tightly packed (`stride == width * bytes_per_pixel`)
    // and one-band, so element `i` starts at `i * bpp` and the element
    // count is exactly `data.len() / bpp`, with no `width * height`
    // product to overflow.
    let bpp = fmt.bytes_per_pixel();
    let data = r.data();
    Ok((0..data.len() / bpp)
        .map(|i| sample_f64(data, i * bpp, 0, fmt))
        .collect())
}

/// Build the one-band float matrix image for `rows`, stamped with
/// `interpretation`.
fn matrix_raster(
    rows: &[Vec<f64>],
    width: u32,
    height: u32,
    bands: u16,
    interpretation: Interpretation,
) -> Result<Raster, MatrixError> {
    let mut data = Vec::with_capacity(width as usize * height as usize * bands as usize * 4);
    for row in rows {
        for &v in row {
            data.extend_from_slice(&(v as f32).to_ne_bytes());
        }
    }
    // Canonical spelling of the layout: `invertlut` on a four-column matrix
    // produces four bands, and that layout is `RgbaF32`, not `FloatF32(4)`
    // (issue #531).
    let format = PixelFormat::with_channels(usize::from(bands), 4)
        .expect("bands is non-zero and 4 bytes per channel is a known depth");
    let mut out = Raster::new(width, height, format, data)?;
    out.meta.interpretation = Some(interpretation);
    Ok(out)
}

/// The direct cofactor inverses for 1x1, 2x2, and 3x3 matrices, ported
/// from `vips_matrixinvert_direct`.
fn invert_direct(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MatrixError> {
    match m.len() {
        1 => {
            let det = m[0][0];
            if det.abs() < TOO_SMALL {
                return Err(MatrixError::Singular);
            }
            Ok(vec![vec![1.0 / det]])
        }
        2 => {
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            if det.abs() < TOO_SMALL {
                return Err(MatrixError::Singular);
            }
            let t = 1.0 / det;
            Ok(vec![
                vec![t * m[1][1], -t * m[0][1]],
                vec![-t * m[1][0], t * m[0][0]],
            ])
        }
        _ => {
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            if det.abs() < TOO_SMALL {
                return Err(MatrixError::Singular);
            }
            let t = 1.0 / det;
            Ok(vec![
                vec![
                    t * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
                    t * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
                    t * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
                ],
                vec![
                    t * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
                    t * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
                    t * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
                ],
                vec![
                    t * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
                    t * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
                    t * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
                ],
            ])
        }
    }
}

/// The PLU decomposition from `lu_decomp` (Numerical Recipes `ludcmp`
/// with implicit pivot scaling): returns the packed LU matrix and the
/// row permutation, or `Singular`.
fn lu_decomp(m: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<usize>), MatrixError> {
    let n = m.len();
    let mut lu: Vec<Vec<f64>> = m.to_vec();
    let mut perm = vec![0usize; n];

    // Scaling factors: the largest magnitude in each row.
    let mut row_scale = vec![0.0f64; n];
    for (i, row) in lu.iter().enumerate() {
        for &v in row {
            row_scale[i] = row_scale[i].max(v.abs());
        }
        if row_scale[i] == 0.0 {
            return Err(MatrixError::Singular);
        }
        row_scale[i] = 1.0 / row_scale[i];
    }

    for j in 0..n {
        // Upper half, except the diagonal.
        for i in 0..j {
            for k in 0..i {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }

        // Diagonal and lower half, tracking the best scaled pivot.
        let mut max = -1.0f64;
        let mut i_of_max = 0usize;
        for i in j..n {
            for k in 0..j {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
            let abs_val = row_scale[i] * lu[i][j].abs();
            if abs_val > max {
                max = abs_val;
                i_of_max = i;
            }
        }

        if lu[i_of_max][j].abs() < TOO_SMALL {
            return Err(MatrixError::Singular);
        }

        if i_of_max != j {
            lu.swap(j, i_of_max);
            row_scale[i_of_max] = row_scale[j];
        }
        perm[j] = i_of_max;

        for i in j + 1..n {
            lu[i][j] /= lu[j][j];
        }
    }

    Ok((lu, perm))
}

/// Solve `A x = vec` in place given the PLU decomposition, ported from
/// `lu_solve`.
fn lu_solve(lu: &[Vec<f64>], perm: &[usize], vec: &mut [f64]) {
    let n = lu.len();
    for i in 0..n {
        let i_perm = perm[i];
        if i_perm != i {
            vec.swap(i, i_perm);
        }
        for j in 0..i {
            vec[i] -= lu[i][j] * vec[j];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            vec[i] -= lu[i][j] * vec[j];
        }
        vec[i] /= lu[i][i];
    }
}

impl Raster {
    // -----------------------------------------------------------------
    // matrixinvert
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::matrixinvert`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotOneBand`] / [`MatrixError::TooLarge`] for an
    /// input `vips_check_matrix` would reject, [`MatrixError::NotSquare`]
    /// for a non-square matrix, [`MatrixError::Singular`] for a singular
    /// or near-singular one, or [`MatrixError::Raster`] on allocation
    /// failure.
    pub fn try_matrixinvert(&self) -> Result<Raster, MatrixError> {
        let m = check_matrix("matrixinvert", self)?;
        if self.width() != self.height() {
            return Err(MatrixError::NotSquare {
                width: self.width(),
                height: self.height(),
            });
        }
        let n = m.len();

        // libvips: direct path below 4x4, PLU above.
        let inv = if n < 4 {
            invert_direct(&m)?
        } else {
            let (lu, perm) = lu_decomp(&m)?;
            let mut inv = vec![vec![0.0f64; n]; n];
            let mut vec_buf = vec![0.0f64; n];
            for j in 0..n {
                vec_buf.fill(0.0);
                vec_buf[j] = 1.0;
                lu_solve(&lu, &perm, &mut vec_buf);
                for i in 0..n {
                    inv[i][j] = vec_buf[i];
                }
            }
            inv
        };

        matrix_raster(&inv, self.width(), self.height(), 1, Interpretation::Matrix)
    }

    /// Invert a square matrix image (libvips `vips_matrixinvert`):
    /// direct cofactor formulas below 4x4, PLU decomposition with
    /// scaled partial pivoting above. The output is a matrix image the
    /// same size as the input, stamped [`Interpretation::Matrix`].
    /// Panicking form of [`Raster::try_matrixinvert`], matching the
    /// ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`MatrixError`]; see [`Raster::try_matrixinvert`].
    #[track_caller]
    pub fn matrixinvert(&self) -> Raster {
        expect_matrix("matrixinvert", self.try_matrixinvert())
    }

    // -----------------------------------------------------------------
    // matrixmultiply
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::matrixmultiply`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotOneBand`] / [`MatrixError::TooLarge`] for
    /// either operand `vips_check_matrix` would reject,
    /// [`MatrixError::ShapeMismatch`] when `self.width()` is not
    /// `right.height()`, or [`MatrixError::Raster`] when the
    /// `right.width()` x `self.height()` output exceeds the allocation
    /// budget ([`RasterError::ByteBudgetExceeded`]) or overflows
    /// `usize` ([`RasterError::SizeOverflow`]). That last one is not a
    /// remote possibility: the output's width and height come from two
    /// *independent* operands, each capped only at `MAX_MATRIX_SIZE`,
    /// so a pair of 400 KB matrices shaped `1 x 100000` and
    /// `100000 x 1` asks for a 40 GB product. The size is checked on
    /// the `u64` path before a byte of it is committed.
    pub fn try_matrixmultiply(&self, right: &Raster) -> Result<Raster, MatrixError> {
        // `vips_matrixmultiply_build` gates both operands first and
        // only then tests the shapes (the "bad sizes" raise is
        // `matrixmultiply.c:90`); keep that precedence, so a
        // doubly-invalid pair reports what vips reports.
        let left_data = check_matrix_flat("matrixmultiply", self)?;
        let right_data = check_matrix_flat("matrixmultiply", right)?;
        if self.width() != right.height() {
            return Err(MatrixError::ShapeMismatch {
                left_w: self.width(),
                left_h: self.height(),
                right_w: right.width(),
                right_h: right.height(),
            });
        }

        // Size the output before building anything. Two operands whose
        // dimensions are independent make the product quadratically
        // larger than either input, so an intermediate built first
        // would commit tens of gigabytes on legal 400 KB inputs and
        // reach `handle_alloc_error`, the uncatchable abort issues
        // #280 and #433 were spent removing. `Raster::zeroed` rejects
        // an over-budget size on the checked `u64` path *before* it
        // touches the allocator, and allocates fallibly after that.
        let out_w = right.width();
        let out_h = self.height();
        let mut out = Raster::zeroed(out_w, out_h, PixelFormat::FloatF32(NonZeroU16::MIN))?;

        // `vips_matrixmultiply_gen` (`matrixmultiply.c:105-127`): one
        // f64 accumulator per output cell, sweeping a left row against
        // a right column, and each cell narrowed to `f32` straight into
        // the output buffer, with no `Vec<Vec<f64>>` in between.
        // The `i`-`j`-`k` nesting is load-bearing: it fixes the order
        // the f64 terms are summed in per cell, so swapping to the
        // cache-friendlier `i`-`k`-`j` would change the last bits of a
        // result libvips computes this way.
        let cols = out_w as usize;
        let inner = self.width() as usize;
        let buf = out.data_mut();
        for (i, left_row) in left_data.chunks(inner).enumerate() {
            for j in 0..cols {
                let mut sum = 0.0f64;
                for (k, &a) in left_row.iter().enumerate() {
                    sum += a * right_data[k * cols + j];
                }
                let at = (i * cols + j) * 4;
                buf[at..at + 4].copy_from_slice(&(sum as f32).to_ne_bytes());
            }
        }

        out.meta.interpretation = Some(Interpretation::Matrix);
        Ok(out)
    }

    /// Multiply two matrix images (libvips `vips_matrixmultiply`): the
    /// dense product of `self` on the left with `right` on the right,
    /// which needs `self.width() == right.height()` and gives a
    /// `right.width()` x `self.height()` matrix image stamped
    /// [`Interpretation::Matrix`]. Elements accumulate in `f64`, with no
    /// scale and no offset. Panicking form of
    /// [`Raster::try_matrixmultiply`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`MatrixError`]; see
    /// [`Raster::try_matrixmultiply`].
    #[track_caller]
    pub fn matrixmultiply(&self, right: &Raster) -> Raster {
        expect_matrix("matrixmultiply", self.try_matrixmultiply(right))
    }

    // -----------------------------------------------------------------
    // invertlut
    // -----------------------------------------------------------------

    /// Fallible form of [`Raster::invertlut`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_invertlut_size`]; the size is the libvips
    /// default of 256.
    pub fn try_invertlut(&self) -> Result<Raster, MatrixError> {
        self.try_invertlut_size(256)
    }

    /// Build an inverse look-up table (libvips `vips_invertlut` with the
    /// default `size: 256`): see [`Raster::try_invertlut_size`].
    /// Panicking form, matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`MatrixError`]; see [`Raster::try_invertlut_size`].
    #[track_caller]
    pub fn invertlut(&self) -> Raster {
        expect_matrix("invertlut", self.try_invertlut())
    }

    /// Fallible form of [`Raster::invertlut_size`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotOneBand`] / [`MatrixError::TooLarge`] for an
    /// input `vips_check_matrix` would reject,
    /// [`MatrixError::TooFewColumns`] without a measured column,
    /// [`MatrixError::OutOfRange`] for an element outside `0..=1`,
    /// [`MatrixError::BadSize`] for a size outside `1..=65536`, or
    /// [`MatrixError::Raster`] on allocation failure.
    pub fn try_invertlut_size(&self, size: u32) -> Result<Raster, MatrixError> {
        let mut data = check_matrix("invertlut", self)?;
        let width = self.width();
        let height = data.len();
        if width < 2 {
            return Err(MatrixError::TooFewColumns { columns: width });
        }
        if !(1..=65536).contains(&size) {
            return Err(MatrixError::BadSize { size });
        }
        let size = size as usize;

        // Range-check every element like `vips_invertlut_build_init`.
        // The `!(0.0..=1.0).contains(...)` shape also rejects NaN, which
        // the C comparison chain lets through into undefined territory.
        for (y, row) in data.iter().enumerate() {
            for (x, &v) in row.iter().enumerate() {
                if !(0.0..=1.0).contains(&v) {
                    return Err(MatrixError::OutOfRange {
                        column: x as u32,
                        row: y as u32,
                        value: v,
                    });
                }
            }
        }

        // Sort rows by the input column, like the C qsort on column 0.
        data.sort_by(|a, b| a[0].partial_cmp(&b[0]).expect("range check rejects NaN"));

        let bands = width as usize - 1;
        let mut buf = vec![0.0f64; size * bands];

        // `vips_invertlut_build_create`, band by band.
        for b in 0..bands {
            // The first and last LUT positions with known real values.
            // C truncates the double to int.
            let first = (data[0][b + 1] * (size - 1) as f64) as usize;
            let last = (data[height - 1][b + 1] * (size - 1) as f64) as usize;

            // Extrapolate the head towards (0, 0) ...
            for k in 0..first {
                // Divide inside the loop, like the C source, so
                // first == 0 never divides by zero.
                let fac = data[0][0] / first as f64;
                buf[b + k * bands] = k as f64 * fac;
            }

            // ... and the tail towards (1, 1).
            for k in last..size {
                let fac = (1.0 - data[height - 1][0]) / ((size - 1) - last) as f64;
                buf[b + k * bands] = data[height - 1][0] + (k - last) as f64 * fac;
            }

            // Interpolate the measured section (inclusive of `last`,
            // overwriting the tail's first slot, like the C loop).
            for k in first..=last {
                let ki = k as f64 / (size - 1) as f64;

                // Search down for the lowest row strictly below ki;
                // default to row 0 when there is none.
                let j = (0..height)
                    .rev()
                    .find(|&j| data[j][b + 1] < ki)
                    .unwrap_or(0);

                if height > 1 {
                    let irange = data[j + 1][b + 1] - data[j][b + 1];
                    let orange = data[j + 1][0] - data[j][0];
                    buf[b + k * bands] = data[j][0] + orange * ((ki - data[j][b + 1]) / irange);
                } else {
                    buf[b + k * bands] = data[j][0];
                }
            }
        }

        let rows: Vec<Vec<f64>> = vec![buf];
        let bands_u16 = u16::try_from(bands).map_err(|_| MatrixError::TooLarge {
            op: "invertlut",
            width,
            height: self.height(),
        })?;
        matrix_raster(&rows, size as u32, 1, bands_u16, Interpretation::Histogram)
    }

    /// Build an inverse look-up table with an explicit output size
    /// (libvips `vips_invertlut` with `size` set). Given a matrix image
    /// of measured points, column 0 the input level and each further
    /// column one band's measured response (all in `0..=1`), produce a
    /// `size` x 1 float image with one band per measured column mapping
    /// each target level back to the input that produces it: linear
    /// interpolation between the bracketing rows, extrapolated to
    /// `(0, 0)` and `(1, 1)` at the ends. Handy for linearising
    /// printers. The output is stamped [`Interpretation::Histogram`].
    /// Panicking form of [`Raster::try_invertlut_size`].
    ///
    /// # Panics
    ///
    /// Panics on any [`MatrixError`]; see [`Raster::try_invertlut_size`].
    #[track_caller]
    pub fn invertlut_size(&self, size: u32) -> Raster {
        expect_matrix("invertlut", self.try_invertlut_size(size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a matrix image back as rows of `f64`.
    fn rows(r: &Raster) -> Vec<Vec<f64>> {
        (0..r.height())
            .map(|y| (0..r.width()).map(|x| r.getpoint(x, y)[0]).collect())
            .collect()
    }

    /// A 2x2 inverse matches the analytic cofactor inverse (the libvips
    /// direct path).
    #[test]
    fn matrixinvert_2x2_analytic() {
        let m = Raster::from_matrix(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let inv = m.matrixinvert();
        assert_eq!(inv.width(), 2);
        assert_eq!(inv.height(), 2);
        assert_eq!(inv.interpretation(), Interpretation::Matrix);
        let got = rows(&inv);
        let expected = [[-2.0, 1.0], [1.5, -0.5]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((got[i][j] - expected[i][j]).abs() < 1e-6, "({i},{j})");
            }
        }
    }

    /// A 3x3 inverse matches the analytic inverse (the libvips direct
    /// path).
    #[test]
    fn matrixinvert_3x3_analytic() {
        let m = Raster::from_matrix(&[
            vec![1.0, 0.0, 1.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ]);
        let inv = m.matrixinvert();
        let got = rows(&inv);
        let expected = [[1.0, 0.0, -0.25], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((got[i][j] - expected[i][j]).abs() < 1e-6, "({i},{j})");
            }
        }
    }

    /// The ported 4x4 matrix (the libvips PLU path) matches its full
    /// analytic inverse, not just the two entries the ported cell
    /// checks.
    #[test]
    fn matrixinvert_4x4_ported_matrix() {
        let m = Raster::from_matrix(&[
            vec![4.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![0.0, 1.0, 2.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        ]);
        let inv = m.matrixinvert();
        assert_eq!(inv.width(), 4);
        assert_eq!(inv.height(), 4);
        let got = rows(&inv);
        let expected = [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [-0.25, 0.0, 0.0, 1.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (got[i][j] - expected[i][j]).abs() < 1e-6,
                    "({i},{j}): {} vs {}",
                    got[i][j],
                    expected[i][j]
                );
            }
        }
    }

    /// Inverting twice returns the original matrix (PLU path).
    #[test]
    fn matrixinvert_round_trip() {
        let original = [
            vec![2.0, 1.0, 0.5, 0.0],
            vec![1.0, 3.0, 0.0, 1.0],
            vec![0.0, 1.0, 4.0, 2.0],
            vec![1.0, 0.0, 2.0, 5.0],
        ];
        let m = Raster::from_matrix(&original);
        let back = m.matrixinvert().matrixinvert();
        let got = rows(&back);
        for i in 0..4 {
            for j in 0..4 {
                assert!((got[i][j] - original[i][j]).abs() < 1e-4, "({i},{j})");
            }
        }
    }

    /// Singular matrices are a typed error on both paths, matching the
    /// libvips TOO_SMALL threshold.
    #[test]
    fn matrixinvert_singular() {
        // Direct path (2x2, linearly dependent rows).
        let m = Raster::from_matrix(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(matches!(m.try_matrixinvert(), Err(MatrixError::Singular)));

        // PLU path (4x4 with a zero row).
        let m = Raster::from_matrix(&[
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
        assert!(matches!(m.try_matrixinvert(), Err(MatrixError::Singular)));

        // PLU path, singular without a zero row (rank 3).
        let m = Raster::from_matrix(&[
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ]);
        assert!(matches!(m.try_matrixinvert(), Err(MatrixError::Singular)));
    }

    /// Shape errors are typed: non-square and multi-band inputs.
    #[test]
    fn matrixinvert_shape_errors() {
        let m = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(matches!(
            m.try_matrixinvert(),
            Err(MatrixError::NotSquare {
                width: 3,
                height: 2
            })
        ));

        let rgb = Raster::black_bands(2, 2, 3);
        assert!(matches!(
            rgb.try_matrixinvert(),
            Err(MatrixError::NotOneBand {
                op: "matrixinvert",
                bands: 3
            })
        ));
    }

    /// The ported `test_invertlut` table, checked against hand-computed
    /// values from the libvips algorithm: zero at 0, one at 255, and
    /// the measured levels mapped back to their inputs.
    #[test]
    fn invertlut_ported_table() {
        let lut = Raster::from_matrix(&[
            vec![0.1, 0.2, 0.3, 0.1],
            vec![0.2, 0.4, 0.4, 0.2],
            vec![0.7, 0.5, 0.6, 0.3],
        ]);
        let im = lut.invertlut();

        assert_eq!(im.width(), 256);
        assert_eq!(im.height(), 1);
        assert_eq!(im.format().channels(), 3);
        assert_eq!(im.interpretation(), Interpretation::Histogram);

        // Head extrapolation: everything starts at 0.
        for &v in &im.getpoint(0, 0) {
            assert!(v.abs() < 0.001);
        }
        // Tail extrapolation: everything ends at 1.
        for &v in &im.getpoint(255, 0) {
            assert!((v - 1.0).abs() < 0.001);
        }
        // Measured points map back to their inputs: band 0 measured 0.2
        // at input 0.1, so LUT[trunc(0.2 * 255)][0] = 0.1. Bands 1 and 2
        // likewise.
        let p = im.getpoint(51, 0);
        assert!((p[0] - 0.1).abs() < 0.001, "band 0 at 51: {}", p[0]);
        let p = im.getpoint(76, 0);
        assert!((p[1] - 0.1).abs() < 0.01, "band 1 at 76: {}", p[1]);
        let p = im.getpoint(25, 0);
        assert!((p[2] - 0.1).abs() < 0.01, "band 2 at 25: {}", p[2]);

        // An interior interpolated value, hand-computed: band 0 at
        // k = 90, ki = 90/255, bracketed by rows (0.1, 0.2) and
        // (0.2, 0.4): 0.1 + 0.1 * (90/255 - 0.2) / 0.2.
        let ki = 90.0 / 255.0;
        let expected = 0.1 + 0.1 * ((ki - 0.2) / 0.2);
        let p = im.getpoint(90, 0);
        assert!((p[0] - expected).abs() < 1e-4, "{} vs {expected}", p[0]);
    }

    /// A one-row table: the measured section collapses to the row's
    /// input value and both extrapolations still anchor at (0,0) and
    /// (1,1).
    #[test]
    fn invertlut_single_row() {
        let lut = Raster::from_matrix(&[vec![0.5, 0.5]]);
        let im = lut.invertlut();
        assert_eq!(im.width(), 256);
        let p = im.getpoint(0, 0);
        assert!(p[0].abs() < 1e-6);
        // trunc(0.5 * 255) = 127: the single measured point.
        let p = im.getpoint(127, 0);
        assert!((p[0] - 0.5).abs() < 1e-6);
        let p = im.getpoint(255, 0);
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    /// An explicit size produces that many entries.
    #[test]
    fn invertlut_explicit_size() {
        let lut = Raster::from_matrix(&[vec![0.2, 0.5], vec![0.7, 0.9]]);
        let im = lut.invertlut_size(1024);
        assert_eq!(im.width(), 1024);
        assert_eq!(im.height(), 1);
        let p = im.getpoint(1023, 0);
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    /// invertlut input validation is typed: range, shape, band count,
    /// and size errors.
    #[test]
    fn invertlut_typed_errors() {
        let out_of_range = Raster::from_matrix(&[vec![0.1, 1.5], vec![0.2, 0.4]]);
        assert!(matches!(
            out_of_range.try_invertlut(),
            Err(MatrixError::OutOfRange {
                column: 1,
                row: 0,
                ..
            })
        ));

        let nan = Raster::from_matrix(&[vec![0.1, f64::NAN], vec![0.2, 0.4]]);
        assert!(matches!(
            nan.try_invertlut(),
            Err(MatrixError::OutOfRange { .. })
        ));

        let narrow = Raster::from_matrix(&[vec![0.1], vec![0.2]]);
        assert!(matches!(
            narrow.try_invertlut(),
            Err(MatrixError::TooFewColumns { columns: 1 })
        ));

        let rgb = Raster::black_bands(4, 2, 3);
        assert!(matches!(
            rgb.try_invertlut(),
            Err(MatrixError::NotOneBand {
                op: "invertlut",
                bands: 3
            })
        ));

        let lut = Raster::from_matrix(&[vec![0.2, 0.5], vec![0.7, 0.9]]);
        assert!(matches!(
            lut.try_invertlut_size(0),
            Err(MatrixError::BadSize { size: 0 })
        ));
        assert!(matches!(
            lut.try_invertlut_size(65537),
            Err(MatrixError::BadSize { size: 65537 })
        ));
    }

    /// The measured vips 8.18.4 case: a 3x2 left times a 2x3 right is
    /// the 2x2 product, exactly, stamped as a matrix image.
    #[test]
    fn matrixmultiply_measured_case() {
        let left = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let right = Raster::from_matrix(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
        let out = left.matrixmultiply(&right);

        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
        assert_eq!(out.format().channels(), 1);
        assert_eq!(out.interpretation(), Interpretation::Matrix);

        let got = rows(&out);
        let expected = [[58.0, 64.0], [139.0, 154.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (got[i][j] - expected[i][j]).abs() < 1e-6,
                    "({i},{j}): {} vs {}",
                    got[i][j],
                    expected[i][j]
                );
            }
        }
    }

    /// Multiplying by the identity gives the original matrix back, with
    /// the non-square shape preserved.
    #[test]
    fn matrixmultiply_identity() {
        let original = [vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let left = Raster::from_matrix(&original);
        let identity = Raster::from_matrix(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let out = left.matrixmultiply(&identity);

        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        let got = rows(&out);
        for i in 0..2 {
            for j in 0..3 {
                assert!((got[i][j] - original[i][j]).abs() < 1e-6, "({i},{j})");
            }
        }
    }

    /// Matrix multiplication does not commute: both orders of the same
    /// pair of 2x2 matrices are legal and give different products, both
    /// measured against vips 8.18.4.
    #[test]
    fn matrixmultiply_not_commutative() {
        let a = Raster::from_matrix(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Raster::from_matrix(&[vec![0.0, 1.0], vec![1.0, 0.0]]);

        let ab = rows(&a.matrixmultiply(&b));
        let ba = rows(&b.matrixmultiply(&a));
        let ab_expected = [[2.0, 1.0], [4.0, 3.0]];
        let ba_expected = [[3.0, 4.0], [1.0, 2.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((ab[i][j] - ab_expected[i][j]).abs() < 1e-6, "ab ({i},{j})");
                assert!((ba[i][j] - ba_expected[i][j]).abs() < 1e-6, "ba ({i},{j})");
            }
        }
    }

    /// A matrix times its own inverse is the identity, cross-checking
    /// `matrixmultiply` against `matrixinvert` on the PLU path.
    #[test]
    fn matrixmultiply_by_inverse_is_identity() {
        let m = Raster::from_matrix(&[
            vec![2.0, 1.0, 0.5, 0.0],
            vec![1.0, 3.0, 0.0, 1.0],
            vec![0.0, 1.0, 4.0, 2.0],
            vec![1.0, 0.0, 2.0, 5.0],
        ]);
        let out = m.matrixmultiply(&m.matrixinvert());

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        let got = rows(&out);
        for (i, row) in got.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                // f32 storage of an f64 accumulation, so a loose bound.
                assert!((v - expected).abs() < 1e-5, "({i},{j}): {v} vs {expected}");
            }
        }
    }

    /// Incompatible sizes are the typed `ShapeMismatch` error, carrying
    /// both shapes, exactly where vips 8.18.4 says "matrixmultiply: bad
    /// sizes". The message names the rule that was violated, not just
    /// the two shapes.
    #[test]
    fn matrixmultiply_shape_mismatch() {
        let a = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let Err(err) = a.try_matrixmultiply(&a) else {
            panic!("3x2 times 3x2 must be rejected, not multiplied");
        };
        assert!(
            matches!(
                err,
                MatrixError::ShapeMismatch {
                    left_w: 3,
                    left_h: 2,
                    right_w: 3,
                    right_h: 2
                }
            ),
            "unexpected variant: {err}"
        );
        assert_eq!(
            err.to_string(),
            "matrixmultiply: shape mismatch: left is 3x2, right is 3x2; \
             left width must equal right height"
        );
    }

    /// The panicking twin names the op exactly once. `ShapeMismatch`
    /// hardcodes "matrixmultiply" in its own `Display`, so
    /// `expect_matrix` must not prefix it again (the #339 stutter).
    #[test]
    fn matrixmultiply_panic_names_the_op_once() {
        let a = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = a.matrixmultiply(&a);
        }));
        std::panic::set_hook(previous);

        let payload = caught.expect_err("mismatched shapes must panic");
        let msg = payload
            .downcast_ref::<String>()
            .map_or("<not a String>", String::as_str);
        assert_eq!(
            msg,
            "matrixmultiply: shape mismatch: left is 3x2, right is 3x2; \
             left width must equal right height"
        );
    }

    /// The output is `right.width()` x `self.height()`, a product of
    /// two *independent* operand dimensions, so 800 KB of legal input
    /// can ask for a 40 GB result. That request has to come back as a
    /// typed error from the checked size path, not as tens of gigabytes
    /// committed first and rejected afterwards (issues #280, #433).
    #[test]
    fn matrixmultiply_oversized_output_is_refused_before_allocating() {
        let column: Vec<Vec<f64>> = (0..MAX_MATRIX_SIZE).map(|_| vec![1.0]).collect();
        let left = Raster::from_matrix(&column);
        let right = Raster::from_matrix(&[vec![1.0; MAX_MATRIX_SIZE as usize]]);
        assert_eq!((left.width(), left.height()), (1, MAX_MATRIX_SIZE));
        assert_eq!((right.width(), right.height()), (MAX_MATRIX_SIZE, 1));

        match left.try_matrixmultiply(&right) {
            Err(MatrixError::Raster(RasterError::ByteBudgetExceeded { bytes, .. })) => {
                assert_eq!(
                    bytes,
                    u64::from(MAX_MATRIX_SIZE) * u64::from(MAX_MATRIX_SIZE) * 4
                );
            }
            Err(other) => panic!("expected a budget rejection, got: {other}"),
            Ok(_) => panic!("expected an error; the 40 GB output was allocated instead"),
        }
    }

    /// Both operands go through the `vips_check_matrix` gate, so a
    /// multi-band input on either side is a typed error.
    #[test]
    fn matrixmultiply_not_one_band() {
        let ok = Raster::from_matrix(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let rgb = Raster::black_bands(2, 2, 3);

        assert!(matches!(
            rgb.try_matrixmultiply(&ok),
            Err(MatrixError::NotOneBand {
                op: "matrixmultiply",
                bands: 3
            })
        ));
        assert!(matches!(
            ok.try_matrixmultiply(&rgb),
            Err(MatrixError::NotOneBand {
                op: "matrixmultiply",
                bands: 3
            })
        ));
    }
}
