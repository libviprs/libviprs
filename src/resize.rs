use crate::pixel::SampleKind;
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
    let fmt = src.format();

    // The integer box-filter kernels assume unsigned 8/16-bit samples;
    // float pyramid levels are not part of the pipeline yet.
    if fmt.is_float() {
        return Err(RasterError::FloatUnsupported {
            op: "downscale_half",
        });
    }

    if fmt.has_alpha() {
        downscale_half_alpha(src)
    } else {
        downscale_half_noalpha(src)
    }
}

/// Read one sample at byte offset `off` as an `i64`, honouring the sample
/// kind.
///
/// Keyed on the kind rather than on `bytes_per_channel() == 1`, whose
/// `else` branch reads two bytes whatever the carrier actually is. That
/// branch used to take a `uint` raster at half stride and average the
/// halves: a uniform 90000 image downscaled to 24464 instead of 90000
/// (issues #517, #607).
///
/// `i64` and total over [`SampleKind`], the shape
/// [`crate::convolution`]'s `put_sample` has carried since #748. Before
/// issue #909 this returned `u32` and panicked on the three signed
/// carriers of #516, which a `Result`-returning entry point could reach:
/// [`downscale_half`] only refuses a float raster, so a `char` pyramid
/// level panicked out of a fallible call. `F32` still cannot arrive,
/// because both entry points refuse it, and the arm answers the
/// `vips_cast` truncation rather than a panic so that stays a fact about
/// the callers rather than a landmine inside the kernel.
#[inline]
fn sample_at(data: &[u8], kind: SampleKind, off: usize) -> i64 {
    match kind {
        SampleKind::U8 => i64::from(data[off]),
        SampleKind::I8 => i64::from(data[off] as i8),
        SampleKind::U16 => i64::from(u16::from_ne_bytes([data[off], data[off + 1]])),
        SampleKind::I16 => i64::from(i16::from_ne_bytes([data[off], data[off + 1]])),
        SampleKind::U32 => i64::from(u32::from_ne_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ])),
        SampleKind::I32 => i64::from(i32::from_ne_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ])),
        SampleKind::F32 => {
            f32::from_ne_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as i64
        }
    }
}

/// Write `v` as one sample at byte offset `off`; the counterpart of
/// [`sample_at`].
///
/// A store and not a cast: every caller has already divided a sum of
/// samples of this kind by their count, so the result is inside the
/// carrier's range and narrowing cannot clip.
#[inline]
fn put_sample(data: &mut [u8], kind: SampleKind, off: usize, v: i64) {
    match kind {
        SampleKind::U8 => data[off] = v as u8,
        SampleKind::I8 => data[off] = v as i8 as u8,
        SampleKind::U16 => data[off..off + 2].copy_from_slice(&(v as u16).to_ne_bytes()),
        SampleKind::I16 => data[off..off + 2].copy_from_slice(&(v as i16).to_ne_bytes()),
        SampleKind::U32 => data[off..off + 4].copy_from_slice(&(v as u32).to_ne_bytes()),
        SampleKind::I32 => data[off..off + 4].copy_from_slice(&(v as i32).to_ne_bytes()),
        SampleKind::F32 => data[off..off + 4].copy_from_slice(&(v as f32).to_ne_bytes()),
    }
}

/// The rounding both kernels close every average with: add half the divisor
/// and **floor**.
///
/// On a non-negative sum this is the round-half-up these kernels have always
/// done, and `div_euclid` and `/` agree there. They part on a negative one,
/// and the floor is the measured answer rather than the convenient one:
/// libvips spells `SHRINK_TYPE_MEAN_INT` as `(tot + 2) >> 2`, an arithmetic
/// shift on a signed `int`, so it floors. Measured on
/// `/opt/homebrew/bin/vips` 8.18.6 by building a `--pyramid` TIFF from a
/// 512x512 `char` raster of repeating 2x2 blocks and reading level 1:
///
/// | block | sum | exact | vips |
/// |---|---|---|---|
/// | `-100, -101, -100, -101` | -402 | -100.5 | **-100** |
/// | `-1, -2, -1, -1` | -5 | -1.25 | **-1** |
/// | `-1, -1, -2, -2` | -6 | -1.5 | **-1** |
/// | `100, 101, 100, 101` | 402 | 100.5 | **101** |
///
/// The second row is the one that decides it: truncating toward zero
/// answers **0** there and vips answers -1 (issue #909).
#[inline]
fn mean_round(sum: i64, count: i64) -> i64 {
    (sum + count / 2).div_euclid(count)
}

/// Downscale without alpha — all channels averaged uniformly.
/// Matches libvips `SHRINK_TYPE_MEAN_INT`: `(sum + 2) >> 2` for 4 pixels.
fn downscale_half_noalpha(src: &Raster) -> Result<Raster, RasterError> {
    let dst_w = src.width().div_ceil(2);
    let dst_h = src.height().div_ceil(2);
    let fmt = src.format();
    let bpp = fmt.bytes_per_pixel();
    let kind = fmt.kind();
    let bpc = kind.bytes();
    let channels = fmt.channels();
    let src_stride = src.stride();
    let src_data = src.data();

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * bpp];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx * 2;
            let sy = dy * 2;

            let x_count = if sx + 1 < src.width() { 2u32 } else { 1 };
            let y_count = if sy + 1 < src.height() { 2u32 } else { 1 };

            let dst_offset = (dy as usize * dst_w as usize + dx as usize) * bpp;

            for c in 0..channels {
                // `i64`, because four four-byte samples do not fit one.
                let mut sum: i64 = 0;
                let count = i64::from(x_count * y_count);

                for oy in 0..y_count {
                    for ox in 0..x_count {
                        let src_offset =
                            (sy + oy) as usize * src_stride + (sx + ox) as usize * bpp + c * bpc;
                        sum += sample_at(src_data, kind, src_offset);
                    }
                }

                put_sample(&mut dst, kind, dst_offset + c * bpc, mean_round(sum, count));
            }
        }
    }

    let mut out = Raster::new(dst_w, dst_h, fmt, dst)?;
    // vips carries the whole block through a shrink, including the resolution,
    // which it does *not* rescale with the pixel grid: `vips shrink in.v out.v
    // 2 2` on an `xres 5 yres 7` source reports 5 and 7 back, and hands on the
    // orientation, the attached fields and the ICC profile, with the origin
    // offsets carried rather than stamped. `reduce` and `resize` agree.
    // Measured on 8.18.6 (#740).
    out.carry_meta_from(src);
    Ok(out)
}

/// Downscale with alpha-weighted averaging for color channels.
///
/// Takes its shape from libvips `SHRINK_ALPHA_TYPE` in
/// `libvips/iofuncs/region.c`:
/// - Alpha channel (last band) is averaged normally: `(a1+a2+a3+a4) / 4`
/// - Color channels are weighted by their pixel's alpha:
///   `(a1*c1 + a2*c2 + a3*c3 + a4*c4) / (a1 + a2 + a3 + a4)`
/// - If the summed alpha is zero, all channels are set to zero
///
/// This prevents transparent pixels from darkening opaque neighbors
/// when averaged together.
///
/// The **rounding deliberately diverges** from `SHRINK_ALPHA_TYPE`, which is
/// worth stating plainly because the shape above matches so closely.
/// `SHRINK_ALPHA_TYPE` accumulates in `double` and stores through a C cast to
/// the sample type, so it **truncates toward zero** — it carries a systematic
/// -0.5 LSB bias, and it does not agree with its own no-alpha sibling
/// `SHRINK_TYPE_MEAN_INT`, which rounds half up with `(tot + 2) >> 2`. Core
/// #458 fixed that: here every sum is accumulated in `u64` and the final
/// divides round half up (colour `(w + alpha_sum/2)/alpha_sum`, alpha
/// `(alpha_sum + count/2)/count`), matching [`downscale_to`] and
/// [`downscale_half_noalpha`] so a fully-opaque RGBA image downscales
/// bit-identically to its RGB twin.
fn downscale_half_alpha(src: &Raster) -> Result<Raster, RasterError> {
    let dst_w = src.width().div_ceil(2);
    let dst_h = src.height().div_ceil(2);
    let fmt = src.format();
    let bpp = fmt.bytes_per_pixel();
    let kind = fmt.kind();
    let bpc = kind.bytes();
    let channels = fmt.channels();
    let alpha_idx = channels - 1;
    let src_stride = src.stride();
    let src_data = src.data();

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * bpp];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx * 2;
            let sy = dy * 2;

            let x_count = if sx + 1 < src.width() { 2u32 } else { 1 };
            let y_count = if sy + 1 < src.height() { 2u32 } else { 1 };
            let count = x_count * y_count;

            let dst_offset = (dy as usize * dst_w as usize + dx as usize) * bpp;

            // Read a single sample, honouring the kind.
            let read = |off: usize| -> i64 { sample_at(src_data, kind, off) };

            // Accumulate the alpha-weighted colour sums and the total alpha over
            // the (up-to) 2x2 source block in `u64`, mirroring `downscale_to`:
            // colour is premultiplied by each pixel's alpha before averaging and
            // un-premultiplied after, so the meaningless RGB of fully-transparent
            // pixels cannot bleed into opaque neighbours. All sums stay integer so
            // this path and `downscale_to` produce bit-identical output for the
            // same block.
            let mut alpha_sum: i64 = 0;
            // `i128` for the products: four four-byte alphas times four
            // four-byte colours overflow an `i64` sum, and the whole point
            // of this kernel is that it stays exact in integers.
            let mut weighted = [0i128; 4];
            for oy in 0..y_count {
                for ox in 0..x_count {
                    let off = (sy + oy) as usize * src_stride + (sx + ox) as usize * bpp;
                    let a = read(off + alpha_idx * bpc);
                    alpha_sum += a;
                    for (c, w) in weighted[..alpha_idx].iter_mut().enumerate() {
                        *w += i128::from(a) * i128::from(read(off + c * bpc));
                    }
                }
            }

            if alpha_sum == 0 {
                // Fully transparent block — leave every band zero (`dst` is
                // pre-zeroed), matching `downscale_to`.
                continue;
            }

            // Alpha-weighted colour bands, `weighted / alpha_sum` round-half-up
            // (matches the `downscale_to` fix, not a truncating C double→int cast),
            // so a fully-opaque RGBA image downscales bit-identically to its RGB
            // twin instead of carrying a systematic -0.5 LSB bias.
            let alpha_sum128 = i128::from(alpha_sum);
            for (c, &w) in weighted[..alpha_idx].iter().enumerate() {
                let result = (w + alpha_sum128 / 2).div_euclid(alpha_sum128);
                put_sample(&mut dst, kind, dst_offset + c * bpc, result as i64);
            }

            // Alpha band: simple average, round-half-up like the no-alpha branch.
            let avg_alpha = mean_round(alpha_sum, i64::from(count));
            put_sample(&mut dst, kind, dst_offset + alpha_idx * bpc, avg_alpha);
        }
    }

    let mut out = Raster::new(dst_w, dst_h, fmt, dst)?;
    // vips carries the whole block through a shrink, including the resolution,
    // which it does *not* rescale with the pixel grid: `vips shrink in.v out.v
    // 2 2` on an `xres 5 yres 7` source reports 5 and 7 back, and hands on the
    // orientation, the attached fields and the ICC profile, with the origin
    // offsets carried rather than stamped. `reduce` and `resize` agree.
    // Measured on 8.18.6 (#740).
    out.carry_meta_from(src);
    Ok(out)
}

/// Number of source samples covered by one destination pixel's source region.
///
/// This is the divisor used to average a destination pixel, so it must be exact
/// for every downscale ratio. The half-open span `[sx0, sx1) x [sy0, sy1)` can
/// be as large as the entire source raster (for a 1x1 destination), and a source
/// may hold more than `u32::MAX` samples, so the product is computed in `u64`.
#[inline]
fn source_region_area(sx0: u32, sx1: u32, sy0: u32, sy1: u32) -> u64 {
    (sx1 - sx0) as u64 * (sy1 - sy0) as u64
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
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-match-page-size)
pub fn downscale_to(src: &Raster, dst_w: u32, dst_h: u32) -> Result<Raster, RasterError> {
    if dst_w == 0 || dst_h == 0 {
        return Err(RasterError::ZeroDimension {
            width: dst_w,
            height: dst_h,
        });
    }

    // The integer area-averaging kernel assumes unsigned 8/16-bit samples;
    // float pyramid levels are not part of the pipeline yet.
    if src.format().is_float() {
        return Err(RasterError::FloatUnsupported { op: "downscale_to" });
    }

    let fmt = src.format();
    let bpp = fmt.bytes_per_pixel();
    let kind = fmt.kind();
    let bpc = kind.bytes();
    let channels = fmt.channels();
    let has_alpha = fmt.has_alpha();
    let src_stride = src.stride();
    let src_data = src.data();
    let src_w = src.width();
    let src_h = src.height();

    // `downscale_to` is downscale-only: reject any target larger than the
    // source in either axis. This bounds the output buffer to the source's
    // already-allocated size, so an adversarial `u32` target (e.g. a crafted
    // PDF page at a large `--dpi` saturating to `u32::MAX`) cannot drive a
    // multi-gigapixel allocation or overflow the size arithmetic below.
    if dst_w > src_w || dst_h > src_h {
        return Err(RasterError::UpscaleNotSupported {
            src_w,
            src_h,
            dst_w,
            dst_h,
        });
    }

    // Fallible allocation: compute the length with checked arithmetic (defence
    // in depth — the bound above already keeps it within the source size) and
    // reserve via `try_reserve` so an allocation failure surfaces as a typed
    // error instead of aborting the process.
    let len = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|px| px.checked_mul(bpp))
        .ok_or(RasterError::SizeOverflow {
            width: dst_w,
            height: dst_h,
            bpp,
        })?;
    let mut dst: Vec<u8> = Vec::new();
    dst.try_reserve(len)
        .map_err(|_| RasterError::AllocationFailed {
            width: dst_w,
            height: dst_h,
            bytes: len,
        })?;
    dst.resize(len, 0);

    // Reusable per-output-pixel alpha-weighted colour accumulator (one entry per
    // colour band), allocated once so the alpha branch does not allocate inside
    // the loop. Empty for the no-alpha path.
    let alpha_idx = if has_alpha { channels - 1 } else { 0 };
    // `u128`, for the reason `downscale_half_alpha` uses it: an alpha
    // times a colour on the 32-bit carrier does not fit a `u64` sum.
    let mut weighted = vec![0i128; alpha_idx];

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
            let count = source_region_area(sx0, sx1, sy0, sy1);

            if count == 0 {
                continue;
            }

            if has_alpha {
                // Alpha-weighted area averaging, mirroring `downscale_half_alpha`
                // for the arbitrary-ratio region `[sx0,sx1) x [sy0,sy1)`. Colour
                // bands are weighted by each source pixel's alpha (premultiply
                // before averaging, un-premultiply after) so the meaningless RGB
                // of fully-transparent pixels cannot bleed into opaque neighbours
                // and darken edges; the alpha band is a simple average.
                //
                // The region is scanned exactly once, reading each source pixel's
                // alpha a single time while accumulating every colour band's
                // alpha-weighted sum alongside the total alpha (#414). All sums
                // stay in `u64` — exact where the colour accumulator would drop
                // low bits above 2^53 for an extreme 16-bit region (#418), just
                // like the no-alpha branch — and the final divides round half-up
                // rather than truncating (#416/#417), so a fully-opaque RGBA image
                // downscales bit-identically to its RGB twin instead of carrying a
                // systematic -0.5 LSB bias.
                let read = |off: usize| -> i64 { sample_at(src_data, kind, off) };

                weighted.iter_mut().for_each(|w| *w = 0);
                let mut alpha_sum: i64 = 0;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let px = sy as usize * src_stride + sx as usize * bpp;
                        let a = read(px + alpha_idx * bpc);
                        alpha_sum += a;
                        for (c, w) in weighted.iter_mut().enumerate() {
                            *w += i128::from(a) * i128::from(read(px + c * bpc));
                        }
                    }
                }

                if alpha_sum == 0 {
                    // Fully transparent region: every band stays zero (the
                    // destination buffer is pre-zeroed), matching the sibling.
                    continue;
                }

                // Alpha-weighted colour bands, `weighted / alpha_sum` round-half-up.
                let alpha_sum128 = i128::from(alpha_sum);
                for (c, &w) in weighted.iter().enumerate() {
                    let result = (w + alpha_sum128 / 2).div_euclid(alpha_sum128);
                    put_sample(&mut dst, kind, dst_offset + c * bpc, result as i64);
                }

                // Alpha band: simple average, round-half-up like the no-alpha
                // branch.
                let avg_alpha = mean_round(alpha_sum, count as i64);
                put_sample(&mut dst, kind, dst_offset + alpha_idx * bpc, avg_alpha);
            } else {
                for c in 0..channels {
                    let mut sum: i64 = 0;
                    for sy in sy0..sy1 {
                        for sx in sx0..sx1 {
                            let src_offset = sy as usize * src_stride + sx as usize * bpp + c * bpc;
                            sum += sample_at(src_data, kind, src_offset);
                        }
                    }
                    put_sample(
                        &mut dst,
                        kind,
                        dst_offset + c * bpc,
                        mean_round(sum, count as i64),
                    );
                }
            }
        }
    }

    let mut out = Raster::new(dst_w, dst_h, fmt, dst)?;
    // vips carries the whole block through a shrink, including the resolution,
    // which it does *not* rescale with the pixel grid: `vips shrink in.v out.v
    // 2 2` on an `xres 5 yres 7` source reports 5 and 7 back, and hands on the
    // orientation, the attached fields and the ICC profile, with the origin
    // offsets carried rather than stamped. `reduce` and `resize` agree.
    // Measured on 8.18.6 (#740).
    out.carry_meta_from(src);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{ALL_KINDS, PixelFormat};

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

    /**
     * Tests that the box-filter kernels' sample reader and writer
     * round-trip every sample kind at its own stride and its own
     * signedness, including the `F32` arm both entry points refuse.
     * Works by sweeping [`ALL_KINDS`] rather than a hand-written list, and
     * by writing at the second sample of a two-sample buffer so a wrong
     * stride overwrites the first and is caught by the neighbour
     * assertion as well as by the value.
     * Input: each kind's `range()` endpoints and 0 -> Output: the same
     * numbers back, byte 0 still zero.
     */
    #[test]
    fn sample_at_and_put_sample_round_trip_every_kind_at_its_own_stride() {
        for kind in ALL_KINDS {
            let bytes = kind.bytes();
            let cases: [i64; 3] = match kind.range() {
                Some((lo, hi)) => [lo, 0, hi],
                None => [-128, 0, 127],
            };
            for v in cases {
                let mut buf = vec![0u8; bytes * 2];
                put_sample(&mut buf, kind, bytes, v);
                assert_eq!(
                    sample_at(&buf, kind, bytes),
                    v,
                    "{kind:?} did not round-trip {v}"
                );
                assert!(
                    buf[..bytes].iter().all(|&b| b == 0),
                    "{kind:?} wrote outside the second sample"
                );
            }
        }
    }

    /**
     * Tests the rounding both box kernels close every average with: add
     * half the divisor and **floor**.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6 by writing a `--pyramid`
     * TIFF from a 512x512 `char` raster of repeating 2x2 blocks and
     * reading level 1, which is what runs `vips_region_shrink`'s
     * `SHRINK_TYPE_MEAN_INT`:
     *
     * | block | sum | exact | vips |
     * |---|---|---|---|
     * | `-100, -101, -100, -101` | -402 | -100.5 | -100 |
     * | `-1, -2, -1, -1` | -5 | -1.25 | **-1** |
     * | `-1, -1, -2, -2` | -6 | -1.5 | -1 |
     * | `100, 101, 100, 101` | 402 | 100.5 | 101 |
     *
     * Works by driving [`mean_round`] on those four sums directly and then
     * driving [`downscale_half`] over the same four blocks laid out as an
     * 8x2 `Int8` raster, so the helper and the kernel are both held. The
     * second row is the one that decides the rule: truncating toward zero
     * answers **0** there, and it is the only row where truncation and
     * flooring disagree, so a table without it passes under either.
     * Input: the four blocks -> Output: `[-100, -1, -1, 101]`.
     */
    #[test]
    fn the_box_mean_floors_after_adding_half_the_divisor() {
        assert_eq!(mean_round(-402, 4), -100);
        assert_eq!(mean_round(-5, 4), -1);
        assert_eq!(mean_round(-6, 4), -1);
        assert_eq!(mean_round(402, 4), 101);
        // The unsigned behaviour is unchanged, which is what makes this a
        // widening rather than a rounding change: on a non-negative sum
        // `div_euclid` and `/` are the same function.
        assert_eq!(mean_round(0, 4), 0);
        assert_eq!(mean_round(1, 4), 0);
        assert_eq!(mean_round(2, 4), 1);
        assert_eq!(mean_round(1020, 4), 255);

        #[rustfmt::skip]
        let src = int8(8, 2, &[
            -100, -101,  -1, -2,  -1, -1,  100, 101,
            -100, -101,  -1, -1,  -2, -2,  100, 101,
        ]);
        let half = downscale_half(&src).unwrap();
        assert_eq!(half.width(), 4);
        assert_eq!(half.height(), 1);
        assert_eq!(i8s(&half), vec![-100, -1, -1, 101]);

        // `downscale_to` shares the rule, over the same blocks.
        let to = downscale_to(&src, 4, 1).unwrap();
        assert_eq!(i8s(&to), vec![-100, -1, -1, 101]);
    }

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
     * Tests that a saturated-dimension target is rejected with a typed error
     * rather than overflowing the output-buffer size arithmetic.
     *
     * `downscale_to` derives its output allocation from the free `u32` target
     * dimensions. A crafted request such as `u32::MAX x u32::MAX` drives
     * `dst_w * dst_h * bpp` past `usize::MAX`: in debug the multiplication
     * panics, and in release it wraps to a small value, under-allocates, and
     * then slice-panics on the first write. Either way the process is taken
     * down by untrusted input. The checked path must return an `Err` before
     * allocating.
     *
     * Input: downscale_to(4x4 Rgba8, u32::MAX, u32::MAX) -> Err (no panic/abort).
     */
    #[test]
    fn downscale_to_saturated_dimensions_rejected() {
        let src = solid_raster(4, 4, &[10, 20, 30, 40], PixelFormat::Rgba8);
        let result = downscale_to(&src, u32::MAX, u32::MAX);
        assert!(result.is_err(), "expected Err for saturated target, got Ok");
        assert!(
            matches!(result, Err(RasterError::UpscaleNotSupported { .. })),
            "expected UpscaleNotSupported, got {result:?}"
        );
    }

    /**
     * Tests that downscale_to enforces its downscale-only contract: a target
     * larger than the source in either axis is rejected with a typed error
     * rather than allocating an unbounded buffer, while an equal-size target
     * (a no-op downscale) is still accepted.
     *
     * Input: downscale_to(8x8, 9, 8) -> Err(UpscaleNotSupported);
     *        downscale_to(8x8, 8, 9) -> Err(UpscaleNotSupported);
     *        downscale_to(8x8, 8, 8) -> Ok.
     */
    #[test]
    fn downscale_to_upscale_rejected() {
        let src = solid_raster(8, 8, &[42], PixelFormat::Gray8);
        assert!(matches!(
            downscale_to(&src, 9, 8),
            Err(RasterError::UpscaleNotSupported { .. })
        ));
        assert!(matches!(
            downscale_to(&src, 8, 9),
            Err(RasterError::UpscaleNotSupported { .. })
        ));
        // Equal size is a valid (no-op) downscale.
        assert!(downscale_to(&src, 8, 8).is_ok());
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

    /**
     * Regression for #287: `downscale_to` must alpha-weight colour channels so
     * the RGB of fully-transparent pixels cannot bleed into opaque neighbours.
     *
     * A 4x1 RGBA raster alternates opaque red and *transparent green* so every
     * 2-pixel destination region mixes an opaque and a transparent pixel across
     * the boundary. Uniform averaging (the pre-fix behaviour) would pull the
     * transparent green into the result (G ~= 128) and darken the red; the
     * alpha-weighted average discards the zero-alpha contribution, so the
     * surviving colour stays pure opaque red and the transparent green never
     * appears.
     *
     * Input: 4x1 Rgba8 [red/A255, green/A0, green/A0, red/A255] -> downscale_to
     * 2x1 -> both pixels (255, 0, 0, 128): red preserved, no green bleed.
     */
    #[test]
    fn downscale_to_rgba_no_colour_bleed() {
        #[rustfmt::skip]
        let data = vec![
            255, 0, 0, 255, // opaque red
            0, 255, 0, 0,   // transparent green (must not bleed)
            0, 255, 0, 0,   // transparent green (must not bleed)
            255, 0, 0, 255, // opaque red
        ];
        let src = Raster::new(4, 1, PixelFormat::Rgba8, data).unwrap();
        let dst = downscale_to(&src, 2, 1).unwrap();
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 1);
        // Each output pixel averages one opaque-red and one transparent-green
        // source pixel. Alpha-weighted: R = (255*255 + 0*0)/255 = 255,
        // G = (255*0 + 0*255)/255 = 0, A = (255 + 0)/2 = 127.5 -> 128
        // (round-half-up, like the no-alpha branch).
        for chunk in dst.data().chunks(4) {
            assert_eq!(chunk[0], 255, "R: opaque red must be preserved");
            assert_eq!(chunk[1], 0, "G: transparent green must NOT bleed in");
            assert_eq!(chunk[2], 0, "B stays 0");
            assert_eq!(chunk[3], 128, "A: simple average, round-half-up");
        }
    }

    /**
     * Regression for #416/#417: a fully-opaque RGBA image and its RGB twin must
     * downscale to identical colour values. The alpha branch previously
     * truncated the alpha-weighted colour while the no-alpha branch rounded
     * half-up, so opaque RGBA output carried a systematic -0.5 LSB bias versus
     * the same image without an alpha band. With both branches rounding half-up
     * the colour channels now agree exactly.
     *
     * Input: 3x1 Rgb8 and its opaque Rgba8 twin with values that average to a
     * .5 boundary -> downscale_to 1x1 -> identical RGB, alpha 255.
     */
    #[test]
    fn downscale_to_opaque_rgba_matches_rgb_twin() {
        // Two pixels whose per-channel means land exactly on .5, where
        // round-half-up and truncation diverge (10.5 -> 11 rounded, 10 truncated).
        #[rustfmt::skip]
        let rgb = vec![
            10, 20, 30,
            11, 21, 31,
        ]; // means: 10.5, 20.5, 30.5 -> round-half-up 11, 21, 31
        let rgba = vec![
            10, 20, 30, 255, //
            11, 21, 31, 255, //
        ];
        let rgb_src = Raster::new(2, 1, PixelFormat::Rgb8, rgb).unwrap();
        let rgba_src = Raster::new(2, 1, PixelFormat::Rgba8, rgba).unwrap();
        let rgb_out = downscale_to(&rgb_src, 1, 1).unwrap();
        let rgba_out = downscale_to(&rgba_src, 1, 1).unwrap();
        let a = rgb_out.data();
        let b = rgba_out.data();
        assert_eq!(a, &[11, 21, 31], "RGB twin rounds half-up");
        assert_eq!(
            &b[..3],
            &[11, 21, 31],
            "opaque RGBA colour must match the RGB twin exactly (no truncation bias)"
        );
        assert_eq!(b[3], 255, "opaque alpha preserved");
    }

    /**
     * Regression for #418: the alpha-weighted colour accumulator must stay
     * integer-exact for 16-bit input, where an f64 accumulator would drop the
     * low bits of a large weighted sum. A uniform opaque 16-bit image downscaled
     * to 1x1 must reproduce its exact channel values (the alpha-weighted mean of
     * a constant is that constant), with no rounding drift.
     *
     * Input: 8x8 Rgba16 uniform (40000, 20000, 8000, 65535) -> 1x1 -> same.
     */
    #[test]
    fn downscale_to_rgba16_opaque_exact() {
        let (w, h) = (8u32, 8u32);
        let color = [40000u16, 20000, 8000, 65535];
        let mut data = Vec::with_capacity((w * h) as usize * 8);
        for _ in 0..w * h {
            for &c in &color {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        let src = Raster::new(w, h, PixelFormat::Rgba16, data).unwrap();
        let dst = downscale_to(&src, 1, 1).unwrap();
        let px = dst.data();
        let s = |i: usize| u16::from_ne_bytes([px[2 * i], px[2 * i + 1]]);
        assert_eq!(
            [s(0), s(1), s(2), s(3)],
            color,
            "16-bit opaque mean is exact"
        );
    }

    // -- Alpha-weighted averaging tests --

    /// Alpha-weighted: solid RGBA with full alpha preserves color exactly.
    #[test]
    fn half_rgba_solid_opaque() {
        let src = solid_raster(4, 4, &[100, 150, 200, 255], PixelFormat::Rgba8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);
        for chunk in dst.data().chunks(4) {
            assert_eq!(chunk, &[100, 150, 200, 255]);
        }
    }

    /// Alpha-weighted: fully transparent pixels produce all-zero output.
    #[test]
    fn half_rgba_fully_transparent() {
        let src = solid_raster(2, 2, &[100, 200, 50, 0], PixelFormat::Rgba8);
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.data(), &[0, 0, 0, 0]);
    }

    /// Alpha-weighted: when alpha varies, color channels are weighted by alpha.
    /// Two opaque red pixels and two transparent green pixels should produce
    /// pure red (not a red/green average), since the green pixels have zero
    /// weight.
    #[test]
    fn half_rgba_alpha_weights_color() {
        // Top-left: opaque red, top-right: opaque red
        // Bottom-left: transparent green, bottom-right: transparent green
        let data = vec![
            255, 0, 0, 255, // red, alpha=255
            255, 0, 0, 255, // red, alpha=255
            0, 255, 0, 0, // green, alpha=0
            0, 255, 0, 0, // green, alpha=0
        ];
        let src = Raster::new(2, 2, PixelFormat::Rgba8, data).unwrap();
        let dst = downscale_half(&src).unwrap();

        // Alpha average = (255+255+0+0)/4 = 127.5 → 128 (round half up)
        // R = (255*255 + 255*255 + 0*0 + 0*0) / (255+255) = 130050/510 = 255
        // G = (255*0 + 255*0 + 0*255 + 0*255) / 510 = 0
        // B = 0
        assert_eq!(dst.data()[0], 255, "R should be 255 (alpha-weighted)");
        assert_eq!(
            dst.data()[1],
            0,
            "G should be 0 (transparent pixels ignored)"
        );
        assert_eq!(dst.data()[2], 0, "B should be 0");
        assert_eq!(
            dst.data()[3],
            128,
            "A should be 128 (127.5 rounded half up)"
        );
    }

    /// Alpha-weighted: partial alpha correctly weights the contribution.
    #[test]
    fn half_rgba_partial_alpha() {
        // One pixel at alpha=200 with value 100, one pixel at alpha=50 with value 200
        // Others transparent
        let data = vec![
            100, 0, 0, 200, // pixel 0: R=100, A=200
            200, 0, 0, 50, // pixel 1: R=200, A=50
            0, 0, 0, 0, // pixel 2: transparent
            0, 0, 0, 0, // pixel 3: transparent
        ];
        let src = Raster::new(2, 2, PixelFormat::Rgba8, data).unwrap();
        let dst = downscale_half(&src).unwrap();

        // Alpha_sum = 200+50+0+0 = 250; count = 4.
        // Alpha = (250 + 4/2) / 4 = 252/4 = 63 (62.5 rounded half up).
        // R = (200*100 + 50*200 + alpha_sum/2) / alpha_sum
        //   = (30000 + 125) / 250 = 120 (round half up).
        assert_eq!(dst.data()[0], 120, "R alpha-weighted (round half up)");
        assert_eq!(dst.data()[3], 63, "A averaged (62.5 rounded half up)");
    }

    /// Alpha-weighted averaging with odd dimensions handles edge pixels.
    #[test]
    fn half_rgba_odd_dimensions() {
        // 3x1 RGBA: only 2 pixels contribute to first output, 1 to second
        let data = vec![
            255, 0, 0, 255, // opaque red
            0, 255, 0, 255, // opaque green
            0, 0, 255, 128, // semi-transparent blue
        ];
        let src = Raster::new(3, 1, PixelFormat::Rgba8, data).unwrap();
        let dst = downscale_half(&src).unwrap();
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 1);
        // First pixel: average of red+green (both alpha=255) → (128,128,0,255)
        // (with alpha-weighted: same as uniform since alpha is equal)
        assert_eq!(dst.data()[0], 128); // R: (255*255 + 510/2)/510 = 65280/510 = 128
        assert_eq!(dst.data()[1], 128); // G: (255*255 + 510/2)/510 = 128 (round half up)
        assert_eq!(dst.data()[3], 255); // A: (255+255)/2 = 255
    }

    /**
     * Tests that the per-destination-pixel source area is computed in u64 so an
     * extreme downscale ratio cannot overflow the divisor.
     * Works by asking for the area of a whole 65536x65536 (~4.3 gigapixel)
     * source mapped onto a single destination pixel: 65536*65536 == 2^32, which
     * overflows a u32 product (debug panic / release wrap to 0 → a black or
     * misdivided output pixel). Computed in u64 it is exact and still usable as
     * a rounding divisor.
     * Input: source region [0,65536) x [0,65536) → area 2^32, avg of all-200 == 200.
     */
    #[test]
    fn area_count_widens_past_u32() {
        let area = source_region_area(0, 65536, 0, 65536);
        assert_eq!(area, 1u64 << 32, "gigapixel area must not overflow u32");

        // Exercise the same rounding-divide the averaging loop performs, with a
        // sum that only fits in u64, to confirm the divisor stays exact.
        let sum: u64 = area * 200; // every source sample == 200
        let avg = (sum + area / 2) / area;
        assert_eq!(avg, 200, "u64 divisor must average correctly");
    }

    /**
     * Tests that the downscale entry points reject float rasters with the
     * typed FloatUnsupported error instead of misreading their bytes with
     * the integer box-filter kernels. Covers both the named RgbaF32 (which
     * has alpha and would otherwise take the alpha kernel) and FloatF32.
     * Input: 4x4 float rasters → Err(FloatUnsupported) from both entries.
     */
    #[test]
    fn downscale_rejects_float_with_typed_error() {
        use crate::pixel::PixelFormat;

        let rgba = Raster::zeroed(4, 4, PixelFormat::RgbaF32).unwrap();
        assert!(matches!(
            downscale_half(&rgba),
            Err(RasterError::FloatUnsupported {
                op: "downscale_half"
            })
        ));
        assert!(matches!(
            downscale_to(&rgba, 2, 2),
            Err(RasterError::FloatUnsupported { op: "downscale_to" })
        ));

        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let gray = Raster::zeroed(4, 4, f1).unwrap();
        assert!(matches!(
            downscale_half(&gray),
            Err(RasterError::FloatUnsupported { .. })
        ));
        assert!(matches!(
            downscale_to(&gray, 2, 2),
            Err(RasterError::FloatUnsupported { .. })
        ));
    }

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
     * Tests that the box-filter kernels average the unsigned 32-bit
     * carrier at its own stride, which is the site where a width-keyed
     * `else` branch silently halved the stride and averaged the halves.
     * Works against `/opt/homebrew/bin/vips` 8.18.6: `vips shrink in 2 2`
     * on a 4x4 `uint` ramp of 100000, 101000, ... answers **102500** and
     * **104500** in row 0 and stays UINT. The uniform case is the sharper
     * one, because a stride bug on a constant image still returns a
     * constant: 90000 came back as **24464** before this, and that is the
     * number to break the fix against.
     * Input: 4x4 uint ramp -> 102500, 104500; 2x2 uint all 90000 -> 90000.
     */
    #[test]
    fn downscale_half_carries_the_uint_carrier() {
        let vals: Vec<u32> = (0..16).map(|i| 100_000 + i * 1000).collect();
        let out = downscale_half(&uint32(4, 4, &vals)).unwrap();
        assert_eq!(
            out.format(),
            PixelFormat::Uint32(core::num::NonZeroU16::new(1).unwrap())
        );
        assert_eq!((u32_at(&out, 0), u32_at(&out, 1)), (102_500, 104_500));

        // The uniform case: any stride error shows up as a value that is
        // not the constant, and 24464 is what the `u16` read gave.
        let flat = downscale_half(&uint32(2, 2, &[90_000; 4])).unwrap();
        assert_eq!(u32_at(&flat, 0), 90_000);

        // Control: the same shape on the carriers that already worked, so
        // this cannot pass by the kernel having stopped averaging.
        let g8 = Raster::new(2, 2, PixelFormat::Gray8, vec![10, 20, 30, 40]).unwrap();
        assert_eq!(downscale_half(&g8).unwrap().data()[0], 25);
    }

    /**
     * Tests that the alpha-weighted kernel carries the 32-bit carrier
     * without overflowing its accumulator, since an alpha times a colour
     * on that carrier does not fit a `u64` sum.
     * Works by downscaling a 2x2 four-band `uint` raster whose alpha is
     * `u32::MAX` and whose colour is `u32::MAX`, which is the largest
     * product the accumulator can be asked for: a `u64` sum wraps there
     * and answers something below the constant.
     * Input: 2x2 Uint32(4) all `u32::MAX` -> 1x1 all `u32::MAX`.
     */
    #[test]
    fn the_alpha_kernel_does_not_overflow_on_the_uint_carrier() {
        let n = core::num::NonZeroU16::new(4).unwrap();
        let fmt = PixelFormat::Uint32(n);
        assert!(
            fmt.has_alpha(),
            "the four-band uint carrier must take the alpha path"
        );
        let data: Vec<u8> = std::iter::repeat_n(u32::MAX, 16)
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let r = Raster::new(2, 2, fmt, data).unwrap();
        let out = downscale_half(&r).unwrap();
        assert_eq!(out.format(), fmt);
        for b in 0..4 {
            assert_eq!(u32_at(&out, b), u32::MAX, "band {b} wrapped");
        }
    }
}
