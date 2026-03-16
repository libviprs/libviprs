use crate::pixel::PixelFormat;
use thiserror::Error;

/// Errors that can occur when creating or slicing a [`Raster`].
///
/// These guard against programmer mistakes such as mismatched buffer sizes,
/// zero-dimension images, and out-of-bounds region requests. They are checked
/// at construction or access time so that pixel-processing code can work with
/// trusted, bounds-checked data.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug, Error)]
pub enum RasterError {
    #[error(
        "dimensions {width}x{height} with format {format:?} require {expected} bytes, got {actual}"
    )]
    BufferSizeMismatch {
        width: u32,
        height: u32,
        format: PixelFormat,
        expected: usize,
        actual: usize,
    },
    #[error("zero dimension: {width}x{height}")]
    ZeroDimension { width: u32, height: u32 },
    #[error("region ({x},{y})+({w},{h}) out of bounds for {raster_w}x{raster_h}")]
    RegionOutOfBounds {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        raster_w: u32,
        raster_h: u32,
    },
}

/// An owned raster image buffer with known dimensions and pixel format.
///
/// `Raster` is the core pixel container in libviprs. It owns a tightly-packed
/// `Vec<u8>` whose length is always exactly `width * height * format.bytes_per_pixel()`.
/// This invariant is enforced at construction time by [`Raster::new`] and
/// [`Raster::zeroed`], so downstream code can index into the buffer without
/// additional bounds arithmetic.
///
/// Use [`Raster::region`] for zero-copy sub-region access or [`Raster::extract`]
/// to copy a sub-rectangle into a new `Raster`.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
#[derive(Debug, Clone)]
pub struct Raster {
    width: u32,
    height: u32,
    format: PixelFormat,
    data: Vec<u8>,
}

impl Raster {
    /// Create a new raster from existing pixel data.
    ///
    /// Validates that `data.len()` equals `width * height * format.bytes_per_pixel()`
    /// and that neither dimension is zero. This is the primary constructor used
    /// when pixel data has already been produced by a decoder or renderer.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0, or
    /// [`RasterError::BufferSizeMismatch`] if the buffer length is wrong.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let expected = width as usize * height as usize * format.bytes_per_pixel();
        if data.len() != expected {
            return Err(RasterError::BufferSizeMismatch {
                width,
                height,
                format,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            format,
            data,
        })
    }

    /// Create a raster filled with zeros.
    ///
    /// Allocates a buffer of the correct size and fills it with `0u8`. Useful
    /// for creating blank tiles or output buffers that will be written into
    /// later (e.g., compositing or scaling operations).
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::ZeroDimension`] if width or height is 0.
    pub fn zeroed(width: u32, height: u32, format: PixelFormat) -> Result<Self, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::ZeroDimension { width, height });
        }
        let size = width as usize * height as usize * format.bytes_per_pixel();
        Self::new(width, height, format, vec![0u8; size])
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The [`PixelFormat`] describing channel count and bit depth.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Immutable reference to the raw pixel data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable reference to the raw pixel data buffer.
    ///
    /// Allows in-place pixel manipulation without re-allocating.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Bytes per row (stride). No padding -- rows are tightly packed.
    ///
    /// Equal to `width * format.bytes_per_pixel()`. Needed when computing
    /// byte offsets into the flat data buffer for a given `(x, y)` position.
    pub fn stride(&self) -> usize {
        self.width as usize * self.format.bytes_per_pixel()
    }

    /// Get an immutable, zero-copy view of a rectangular sub-region.
    ///
    /// The returned [`RegionView`] borrows from this `Raster` and provides
    /// row-by-row or per-pixel access without copying any data. This is the
    /// preferred way to read tile-sized chunks during pyramid generation.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::RegionOutOfBounds`] if the rectangle exceeds the
    /// raster dimensions or has a zero width/height.
    pub fn region(&self, x: u32, y: u32, w: u32, h: u32) -> Result<RegionView<'_>, RasterError> {
        if x + w > self.width || y + h > self.height || w == 0 || h == 0 {
            return Err(RasterError::RegionOutOfBounds {
                x,
                y,
                w,
                h,
                raster_w: self.width,
                raster_h: self.height,
            });
        }
        Ok(RegionView {
            raster: self,
            x,
            y,
            w,
            h,
        })
    }

    /// Extract a sub-region as a new owned `Raster`.
    ///
    /// Copies the pixel data row-by-row into a freshly allocated buffer.
    /// Use this when you need an independent `Raster` (e.g., to encode a tile
    /// to disk) rather than a borrowed view.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::RegionOutOfBounds`] if the rectangle is invalid.
    pub fn extract(&self, x: u32, y: u32, w: u32, h: u32) -> Result<Raster, RasterError> {
        let view = self.region(x, y, w, h)?;
        let bpp = self.format.bytes_per_pixel();
        let mut out = Vec::with_capacity(w as usize * h as usize * bpp);
        for row in view.rows() {
            out.extend_from_slice(row);
        }
        Raster::new(w, h, self.format, out)
    }
}

/// An immutable, zero-copy view into a rectangular sub-region of a [`Raster`].
///
/// Borrows the parent `Raster` and exposes only the pixels within the
/// specified rectangle. Row iteration via [`RegionView::rows`] and single-pixel
/// access via [`RegionView::pixel`] translate region-local coordinates to
/// absolute buffer offsets automatically.
///
/// Prefer `RegionView` over [`Raster::extract`] when you only need to read
/// pixels without owning them, as it avoids allocation and copying.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug)]
pub struct RegionView<'a> {
    raster: &'a Raster,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl<'a> RegionView<'a> {
    /// Width of the viewed sub-region in pixels.
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Height of the viewed sub-region in pixels.
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Iterate over rows of pixel data in this region.
    ///
    /// Each item is a byte slice of length `width * format.bytes_per_pixel()`,
    /// representing one scanline of the sub-region. Rows are yielded from top
    /// to bottom.
    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> {
        let bpp = self.raster.format.bytes_per_pixel();
        let stride = self.raster.stride();
        let x_offset = self.x as usize * bpp;
        let row_len = self.w as usize * bpp;
        let data = self.raster.data();
        (self.y..self.y + self.h).map(move |row| {
            let start = row as usize * stride + x_offset;
            &data[start..start + row_len]
        })
    }

    /// Get pixel data at `(px, py)` relative to the region origin.
    ///
    /// Returns a byte slice of length `format.bytes_per_pixel()` for the
    /// requested pixel, or `None` if `(px, py)` is outside the region bounds.
    pub fn pixel(&self, px: u32, py: u32) -> Option<&'a [u8]> {
        if px >= self.w || py >= self.h {
            return None;
        }
        let bpp = self.raster.format.bytes_per_pixel();
        let stride = self.raster.stride();
        let abs_x = self.x + px;
        let abs_y = self.y + py;
        let start = abs_y as usize * stride + abs_x as usize * bpp;
        Some(&self.raster.data()[start..start + bpp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb_raster(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        // Fill with a pattern: pixel (x,y) = (x as u8, y as u8, (x+y) as u8)
        for y in 0..h {
            for x in 0..w {
                let offset = (y as usize * w as usize + x as usize) * bpp;
                data[offset] = x as u8;
                data[offset + 1] = y as u8;
                data[offset + 2] = (x + y) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /**
     * Tests that Raster::new rejects buffers that don't match width*height*bpp.
     * Works by providing a too-small buffer (11 bytes for 2x2 Rgb8=12) and
     * verifying Err, then the exact size and verifying Ok.
     * Input: 2x2 Rgb8 with 11 bytes → Err; with 12 bytes → Ok.
     */
    #[test]
    fn new_validates_buffer_size() {
        let result = Raster::new(2, 2, PixelFormat::Rgb8, vec![0u8; 11]);
        assert!(result.is_err());

        let result = Raster::new(2, 2, PixelFormat::Rgb8, vec![0u8; 12]);
        assert!(result.is_ok());
    }

    /**
     * Tests that zero-dimension rasters are rejected by both new() and zeroed().
     * Works by passing width=0 or height=0 and asserting Err is returned.
     * Input: 0x10 Rgb8 → Err; 10x0 Rgb8 → Err; zeroed(0,5) → Err.
     */
    #[test]
    fn zero_dimension_rejected() {
        assert!(Raster::new(0, 10, PixelFormat::Rgb8, vec![]).is_err());
        assert!(Raster::new(10, 0, PixelFormat::Rgb8, vec![]).is_err());
        assert!(Raster::zeroed(0, 5, PixelFormat::Gray8).is_err());
    }

    /**
     * Tests that stride equals width * bytes_per_pixel.
     * Works by creating a 100x50 Rgba8 raster and checking stride == 400.
     * Input: 100x50 Rgba8 → Output: stride() == 400.
     */
    #[test]
    fn stride_is_width_times_bpp() {
        let r = Raster::zeroed(100, 50, PixelFormat::Rgba8).unwrap();
        assert_eq!(r.stride(), 400);
    }

    /**
     * Tests that region() validates bounds against the raster dimensions.
     * Works by requesting valid regions (Ok) and out-of-bounds or zero-width
     * regions (Err) on a 10x10 raster.
     * Input: region(5,5,6,5) on 10x10 → Err (x+w > width).
     */
    #[test]
    fn region_bounds_checking() {
        let r = Raster::zeroed(10, 10, PixelFormat::Rgb8).unwrap();
        assert!(r.region(0, 0, 10, 10).is_ok());
        assert!(r.region(5, 5, 5, 5).is_ok());
        assert!(r.region(5, 5, 6, 5).is_err()); // x+w > width
        assert!(r.region(0, 0, 0, 5).is_err()); // zero width
    }

    /**
     * Tests that RegionView pixels correspond to the correct source raster pixels.
     * Works by creating a raster with position-dependent values (x, y, x+y per pixel)
     * and verifying region pixel (0,0) maps to source pixel (4,3).
     * Input: region(4,3,8,8).pixel(0,0) → [4, 3, 7].
     */
    #[test]
    fn region_pixel_matches_source() {
        let r = make_rgb_raster(16, 16);
        let view = r.region(4, 3, 8, 8).unwrap();

        // pixel (0,0) in region = (4,3) in raster
        let px = view.pixel(0, 0).unwrap();
        assert_eq!(px, &[4, 3, 7]);

        // pixel (7,7) in region = (11,10) in raster
        let px = view.pixel(7, 7).unwrap();
        assert_eq!(px, &[11, 10, 21]);
    }

    /**
     * Tests that accessing a pixel outside the region returns None.
     * Works by creating a 5x5 region and requesting pixel (5,0) and (0,5),
     * both one past the boundary.
     * Input: 5x5 region, pixel(5,0) → None.
     */
    #[test]
    fn region_pixel_out_of_bounds_returns_none() {
        let r = Raster::zeroed(10, 10, PixelFormat::Rgb8).unwrap();
        let view = r.region(0, 0, 5, 5).unwrap();
        assert!(view.pixel(5, 0).is_none());
        assert!(view.pixel(0, 5).is_none());
    }

    /**
     * Tests that extract() copies the correct sub-rectangle into a new Raster.
     * Works by extracting a 4x5 region from a position-encoded 16x16 raster
     * and verifying the first and last pixels match the expected source coords.
     * Input: extract(2,3,4,5) → Output: 4x5 Raster, first pixel=[2,3,5].
     */
    #[test]
    fn extract_produces_correct_sub_image() {
        let r = make_rgb_raster(16, 16);
        let sub = r.extract(2, 3, 4, 5).unwrap();

        assert_eq!(sub.width(), 4);
        assert_eq!(sub.height(), 5);
        assert_eq!(sub.format(), PixelFormat::Rgb8);
        assert_eq!(sub.data().len(), 4 * 5 * 3);

        // First pixel of extracted region should be (2,3) from original
        let bpp = 3;
        assert_eq!(sub.data()[0], 2); // x
        assert_eq!(sub.data()[1], 3); // y
        assert_eq!(sub.data()[2], 5); // x+y
        // Last pixel: (5,7) in original
        let last = (4 * 5 - 1) * bpp;
        assert_eq!(sub.data()[last], 5);
        assert_eq!(sub.data()[last + 1], 7);
        assert_eq!(sub.data()[last + 2], 12);
    }

    /**
     * Tests that RegionView::rows() yields the correct row slices.
     * Works by iterating rows of a 3x2 region starting at (1,1) and
     * verifying row count and pixel values against the source raster.
     * Input: region(1,1,3,2).rows() → 2 rows, each 9 bytes (3px * 3bpp).
     */
    #[test]
    fn region_rows_iteration() {
        let r = make_rgb_raster(8, 8);
        let view = r.region(1, 1, 3, 2).unwrap();

        let rows: Vec<&[u8]> = view.rows().collect();
        assert_eq!(rows.len(), 2);
        // Row 0 of region = row 1 of raster, pixels 1..4
        assert_eq!(rows[0].len(), 9); // 3 pixels * 3 bpp
        assert_eq!(rows[0][0..3], [1, 1, 2]); // pixel (1,1)
        assert_eq!(rows[0][3..6], [2, 1, 3]); // pixel (2,1)
    }

    /**
     * Tests that a 1x1 raster works correctly for all operations.
     * Works by creating a single Gray8 pixel and verifying dimensions, data,
     * region creation, and pixel access all succeed.
     * Input: 1x1 Gray8 [42] → region(0,0,1,1).pixel(0,0) == [42].
     */
    #[test]
    fn single_pixel_raster() {
        let r = Raster::new(1, 1, PixelFormat::Gray8, vec![42]).unwrap();
        assert_eq!(r.width(), 1);
        assert_eq!(r.height(), 1);
        assert_eq!(r.data(), &[42]);

        let view = r.region(0, 0, 1, 1).unwrap();
        assert_eq!(view.pixel(0, 0), Some([42].as_slice()));
    }

    /**
     * Tests that Raster::zeroed produces a buffer filled entirely with zeros.
     * Works by creating a 5x5 Rgba8 zeroed raster and checking every byte.
     * Input: zeroed(5,5,Rgba8) → Output: all 100 bytes == 0.
     */
    #[test]
    fn zeroed_raster_is_all_zeros() {
        let r = Raster::zeroed(5, 5, PixelFormat::Rgba8).unwrap();
        assert!(r.data().iter().all(|&b| b == 0));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            .. ProptestConfig::default()
        })]
        // Tests that buffer size always equals w*h*bpp for all formats and dimensions.
        // Works by generating random dimensions and checking the invariant across
        // all 6 PixelFormat variants.
        // Input: random w,h in 1..256, all formats → Output: data.len() == w*h*bpp.
        #[test]
        fn buffer_size_invariant(w in 1u32..256, h in 1u32..256) {
            for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8,
                        PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16] {
                let r = Raster::zeroed(w, h, fmt).unwrap();
                prop_assert_eq!(
                    r.data().len(),
                    w as usize * h as usize * fmt.bytes_per_pixel()
                );
            }
        }

        // Tests that extract() and region().pixel() return identical data.
        // Works by generating random sub-rectangles and comparing every pixel
        // between the RegionView and the extracted Raster.
        // Input: random region within random raster → Output: all pixels match.
        #[test]
        fn extract_matches_region_pixels(
            w in 4u32..64, h in 4u32..64,
            rx in 0u32..4, ry in 0u32..4,
            rw in 1u32..4, rh in 1u32..4,
        ) {
            prop_assume!(rx + rw <= w && ry + rh <= h);

            let bpp = PixelFormat::Rgb8.bytes_per_pixel();
            let mut data = vec![0u8; w as usize * h as usize * bpp];
            for y in 0..h {
                for x in 0..w {
                    let offset = (y as usize * w as usize + x as usize) * bpp;
                    data[offset] = (x % 256) as u8;
                    data[offset + 1] = (y % 256) as u8;
                    data[offset + 2] = ((x + y) % 256) as u8;
                }
            }
            let raster = Raster::new(w, h, PixelFormat::Rgb8, data).unwrap();
            let view = raster.region(rx, ry, rw, rh).unwrap();
            let extracted = raster.extract(rx, ry, rw, rh).unwrap();

            for py in 0..rh {
                for px in 0..rw {
                    let view_pixel = view.pixel(px, py).unwrap();
                    let ext_offset = (py as usize * rw as usize + px as usize) * bpp;
                    let ext_pixel = &extracted.data()[ext_offset..ext_offset + bpp];
                    prop_assert_eq!(view_pixel, ext_pixel);
                }
            }
        }
    }
}
