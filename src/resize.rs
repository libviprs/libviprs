use crate::raster::{Raster, RasterError};

/// Downscale a raster by 2x using a box filter (area averaging).
///
/// Each 2x2 block in the source maps to one pixel in the output.
/// For odd dimensions, the last row/column is averaged with fewer samples.
/// This is the workhorse of the pyramid builder: each pyramid level is
/// produced by applying `downscale_half` to the level above it.
///
/// # Example usage
///
/// - [test_resize_quarter](https://github.com/libviprs/libviprs-tests/blob/main/tests/ported_resample.rs)
///   chains two `downscale_half` calls to produce a quarter-size image and
///   verifies the resulting dimensions.
pub fn downscale_half(src: &Raster) -> Result<Raster, RasterError> {
    let dst_w = src.width().div_ceil(2);
    let dst_h = src.height().div_ceil(2);
    let fmt = src.format();
    let bpp = fmt.bytes_per_pixel();
    let bpc = fmt.bytes_per_channel();
    let channels = fmt.channels();
    let src_stride = src.stride();
    let src_data = src.data();

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * bpp];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx * 2;
            let sy = dy * 2;

            // Determine how many source pixels contribute (1-4)
            let x_count = if sx + 1 < src.width() { 2u32 } else { 1 };
            let y_count = if sy + 1 < src.height() { 2u32 } else { 1 };

            let dst_offset = (dy as usize * dst_w as usize + dx as usize) * bpp;

            for c in 0..channels {
                let mut sum: u32 = 0;
                let count = x_count * y_count;

                for oy in 0..y_count {
                    for ox in 0..x_count {
                        let src_offset =
                            (sy + oy) as usize * src_stride + (sx + ox) as usize * bpp + c * bpc;

                        if bpc == 1 {
                            sum += src_data[src_offset] as u32;
                        } else {
                            // 16-bit: native endian
                            let val = u16::from_ne_bytes([
                                src_data[src_offset],
                                src_data[src_offset + 1],
                            ]);
                            sum += val as u32;
                        }
                    }
                }

                let avg = (sum + count / 2) / count; // Rounded average

                if bpc == 1 {
                    dst[dst_offset + c] = avg as u8;
                } else {
                    let bytes = (avg as u16).to_ne_bytes();
                    dst[dst_offset + c * 2] = bytes[0];
                    dst[dst_offset + c * 2 + 1] = bytes[1];
                }
            }
        }
    }

    Raster::new(dst_w, dst_h, fmt, dst)
}

/// Downscale a raster to arbitrary dimensions using simple bilinear-ish area averaging.
///
/// Maps each destination pixel to the corresponding rectangular region in the
/// source and averages all source samples within that region. This handles
/// non-power-of-two scale factors, unlike [`downscale_half`] which only
/// supports exact 2x reduction.
///
/// For pyramid generation, prefer `downscale_half` iteratively -- it is faster
/// and matches the level-halving semantics exactly.
///
/// # Example usage
///
/// - [test_resize_rounding](https://github.com/libviprs/libviprs-tests/blob/main/tests/ported_resample.rs)
///   exercises arbitrary-ratio downscaling and checks that output dimensions
///   are correctly rounded.
pub fn downscale_to(src: &Raster, dst_w: u32, dst_h: u32) -> Result<Raster, RasterError> {
    if dst_w == 0 || dst_h == 0 {
        return Err(RasterError::ZeroDimension {
            width: dst_w,
            height: dst_h,
        });
    }

    let fmt = src.format();
    let bpp = fmt.bytes_per_pixel();
    let bpc = fmt.bytes_per_channel();
    let channels = fmt.channels();
    let src_stride = src.stride();
    let src_data = src.data();
    let src_w = src.width();
    let src_h = src.height();

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * bpp];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel to source region
            let sx0 = (dx as u64 * src_w as u64 / dst_w as u64) as u32;
            let sy0 = (dy as u64 * src_h as u64 / dst_h as u64) as u32;
            let sx1 = (((dx + 1) as u64 * src_w as u64).div_ceil(dst_w as u64)) as u32;
            let sy1 = (((dy + 1) as u64 * src_h as u64).div_ceil(dst_h as u64)) as u32;
            let sx1 = sx1.min(src_w);
            let sy1 = sy1.min(src_h);

            let dst_offset = (dy as usize * dst_w as usize + dx as usize) * bpp;
            let count = (sx1 - sx0) * (sy1 - sy0);

            if count == 0 {
                continue;
            }

            for c in 0..channels {
                let mut sum: u64 = 0;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let src_offset = sy as usize * src_stride + sx as usize * bpp + c * bpc;
                        if bpc == 1 {
                            sum += src_data[src_offset] as u64;
                        } else {
                            let val = u16::from_ne_bytes([
                                src_data[src_offset],
                                src_data[src_offset + 1],
                            ]);
                            sum += val as u64;
                        }
                    }
                }
                let avg = (sum + count as u64 / 2) / count as u64;
                if bpc == 1 {
                    dst[dst_offset + c] = avg as u8;
                } else {
                    let bytes = (avg as u16).to_ne_bytes();
                    dst[dst_offset + c * 2] = bytes[0];
                    dst[dst_offset + c * 2 + 1] = bytes[1];
                }
            }
        }
    }

    Raster::new(dst_w, dst_h, fmt, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    fn solid_raster(w: u32, h: u32, pixel: &[u8], fmt: PixelFormat) -> Raster {
        let bpp = fmt.bytes_per_pixel();
        assert_eq!(pixel.len(), bpp);
        let mut data = Vec::with_capacity(w as usize * h as usize * bpp);
        for _ in 0..(w * h) {
            data.extend_from_slice(pixel);
        }
        Raster::new(w, h, fmt, data).unwrap()
    }

    /**
     * Tests that halving a raster with even dimensions produces exact half sizes.
     * Works by creating a 4x4 solid gray raster and verifying the output is 2x2
     * with all pixel values preserved.
     * Input: 4x4 Gray8 solid(200) → Output: 2x2 Gray8, all pixels == 200.
     */
    #[test]
    fn half_even_dimensions() {
        // 4x4 solid gray → 2x2 solid gray
        let src = solid_raster(4, 4, &[200], PixelFormat::Gray8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);
        assert!(dst.data().iter().all(|&b| b == 200));
    }

    /**
     * Tests that halving a raster with odd dimensions rounds up correctly.
     * Works by halving a 5x5 solid raster and verifying ceil(5/2)=3 for both axes.
     * Input: 5x5 Gray8 solid(100) → Output: 3x3 Gray8, all pixels == 100.
     */
    #[test]
    fn half_odd_dimensions() {
        // 5x5 → 3x3
        let src = solid_raster(5, 5, &[100], PixelFormat::Gray8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 3);
        assert!(dst.data().iter().all(|&b| b == 100));
    }

    /**
     * Tests that halving a 1x1 raster returns a 1x1 raster unchanged.
     * Works by verifying the minimum size boundary — cannot shrink below 1x1.
     * Input: 1x1 Gray8 [42] → Output: 1x1 Gray8 [42].
     */
    #[test]
    fn half_1x1_stays_1x1() {
        let src = solid_raster(1, 1, &[42], PixelFormat::Gray8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 1);
        assert_eq!(dst.height(), 1);
        assert_eq!(dst.data(), &[42]);
    }

    /**
     * Tests that downscale_half correctly averages pixel values.
     * Works by using a 2x2 image with known distinct values and checking
     * the single output pixel equals their arithmetic mean.
     * Input: 2x2 Gray8 [10,20,30,40] → Output: 1x1 Gray8 [25].
     */
    #[test]
    fn half_averaging_works() {
        // 2x2 with known pixel values → 1x1 with average
        let data = vec![10, 20, 30, 40]; // Four Gray8 pixels
        let src = Raster::new(2, 2, PixelFormat::Gray8, data).unwrap();
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 1);
        assert_eq!(dst.height(), 1);
        // Average of 10,20,30,40 = 25
        assert_eq!(dst.data()[0], 25);
    }

    /**
     * Tests that downscale_half works correctly with RGB8 (3-channel) images.
     * Works by halving a 2x2 solid red image and verifying the 1x1 result
     * preserves the exact RGB values.
     * Input: 2x2 Rgb8 solid(255,0,0) → Output: 1x1 Rgb8 [255,0,0].
     */
    #[test]
    fn half_rgb8() {
        // 2x2 solid red → 1x1 solid red
        let src = solid_raster(2, 2, &[255, 0, 0], PixelFormat::Rgb8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 1);
        assert_eq!(dst.height(), 1);
        assert_eq!(dst.data(), &[255, 0, 0]);
    }

    /**
     * Tests that downscale_half works correctly with RGBA8 (4-channel) images.
     * Works by halving a 4x4 solid RGBA image and verifying all 2x2 output
     * pixels preserve the exact channel values including alpha.
     * Input: 4x4 Rgba8 solid(100,150,200,255) → Output: 2x2 Rgba8, same values.
     */
    #[test]
    fn half_rgba8() {
        let src = solid_raster(4, 4, &[100, 150, 200, 255], PixelFormat::Rgba8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);
        // All pixels should be the same solid color
        for chunk in dst.data().chunks(4) {
            assert_eq!(chunk, &[100, 150, 200, 255]);
        }
    }

    /**
     * Tests that downscale_half preserves the PixelFormat of the source.
     * Works by halving images in Gray8, Rgb8, and Rgba8 and asserting the
     * output format matches the input format.
     * Input: 8x8 in each format → Output: 4x4 with same format.
     */
    #[test]
    fn half_preserves_format() {
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let bpp = fmt.bytes_per_pixel();
            let pixel: Vec<u8> = (0..bpp).map(|i| (i * 50) as u8).collect();
            let src = solid_raster(8, 8, &pixel, fmt);
            let dst = downscale_half(&src).unwrap();
            assert_eq!(dst.format(), fmt);
        }
    }

    /**
     * Tests that repeatedly halving converges to a 1x1 image without error.
     * Works by iteratively halving a 256x256 solid raster until 1x1 and
     * verifying the final pixel value is preserved (no drift from rounding).
     * Input: 256x256 Gray8 solid(128) → Output: 1x1 Gray8 [128].
     */
    #[test]
    fn half_iterative_to_1x1() {
        let mut r = solid_raster(256, 256, &[128], PixelFormat::Gray8);
        while r.width() > 1 || r.height() > 1 {
            r = downscale_half(&r).unwrap();
        }
        assert_eq!(r.width(), 1);
        assert_eq!(r.height(), 1);
        assert_eq!(r.data()[0], 128);
    }

    /**
     * Tests that downscaling to the same dimensions is a no-op.
     * Works by calling downscale_to with identical width/height and
     * verifying pixel values are unchanged.
     * Input: 10x10 Gray8 solid(77) → Output: 10x10 Gray8, all pixels == 77.
     */
    #[test]
    fn downscale_to_same_size() {
        let src = solid_raster(10, 10, &[77], PixelFormat::Gray8);
        let dst = downscale_to(&src, 10, 10).unwrap();
        assert_eq!(dst.width(), 10);
        assert_eq!(dst.height(), 10);
        assert!(dst.data().iter().all(|&b| b == 77));
    }

    /**
     * Tests that downscale_to rejects zero target dimensions.
     * Works by passing width=0 or height=0 and asserting an Err is returned.
     * Input: downscale_to(10x10, 0, 5) → Output: Err.
     */
    #[test]
    fn downscale_to_zero_rejected() {
        let src = solid_raster(10, 10, &[1], PixelFormat::Gray8);
        assert!(downscale_to(&src, 0, 5).is_err());
        assert!(downscale_to(&src, 5, 0).is_err());
    }

    /**
     * Tests that downscaling a solid-color image preserves the color exactly.
     * Works by area-averaging a uniform RGB image to an arbitrary smaller size
     * and verifying every output pixel matches the original color.
     * Input: 100x100 Rgb8 solid(200,100,50) → Output: 33x25 Rgb8, same color.
     */
    #[test]
    fn downscale_to_solid_preserved() {
        let src = solid_raster(100, 100, &[200, 100, 50], PixelFormat::Rgb8);
        let dst = downscale_to(&src, 33, 25).unwrap();
        assert_eq!(dst.width(), 33);
        assert_eq!(dst.height(), 25);
        for chunk in dst.data().chunks(3) {
            assert_eq!(chunk, &[200, 100, 50]);
        }
    }
}
