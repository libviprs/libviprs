//! Pixel-access and combining utilities for [`Raster`].
//!
//! These are the small, format-aware helpers the arithmetic and creation ops
//! build on: [`Raster::getpoint`] reads one pixel as per-band `f64` samples,
//! and [`Raster::add`] combines two rasters pixel-by-pixel. They live beside
//! the drawing framework (see [`crate::draw`]) as additive `impl Raster`
//! blocks, so the core [`Raster`] type in [`crate::raster`] is untouched.
//!
//! 16-bit samples are read and written in native byte order, matching the
//! convention the resize and decode paths already use; float samples are
//! likewise native-order `f32`.

use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// Read the `n`-th channel sample at byte offset `off` as `f64`, honouring the
/// format's sample type (native byte order for 16-bit and float).
fn sample_f64(data: &[u8], off: usize, channel: usize, fmt: PixelFormat) -> f64 {
    match fmt.bytes_per_channel() {
        1 => data[off + channel] as f64,
        2 => {
            let b = off + channel * 2;
            u16::from_ne_bytes([data[b], data[b + 1]]) as f64
        }
        _ => {
            let b = off + channel * 4;
            f32::from_ne_bytes([data[b], data[b + 1], data[b + 2], data[b + 3]]) as f64
        }
    }
}

impl Raster {
    /// Read the pixel at `(x, y)` as one `f64` per channel.
    ///
    /// The returned vector has [`PixelFormat::channels`] entries: `[gray]` for
    /// grayscale, `[r, g, b]` for RGB, `[r, g, b, a]` for RGBA. Samples are the
    /// raw stored values (`0..=255` for 8-bit formats, `0..=65535` for 16-bit,
    /// the stored `f32` value for float formats), not normalised. This mirrors
    /// libvips `getpoint`, giving arithmetic and LUT code a depth-independent
    /// view of a pixel.
    ///
    /// # Performance
    ///
    /// Allocates a fresh `Vec<f64>` (one entry per channel) on **every** call,
    /// so reading a pixel in a hot per-pixel loop allocates once per pixel.
    /// It is intended for occasional point reads (LUT/arithmetic setup, tests);
    /// for scanning many pixels, walk [`Raster::data`] with [`Raster::region`]
    /// and decode samples inline instead.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is outside the raster, matching the "known-good
    /// index" contract of the ported callers; use [`Raster::region`] first when
    /// the coordinate might be out of range.
    pub fn getpoint(&self, x: u32, y: u32) -> Vec<f64> {
        assert!(
            x < self.width() && y < self.height(),
            "getpoint({x},{y}) out of bounds for {}x{}",
            self.width(),
            self.height()
        );
        let fmt = self.format();
        let channels = fmt.channels();
        let bpp = fmt.bytes_per_pixel();
        let off = y as usize * self.stride() + x as usize * bpp;
        let data = self.data();
        (0..channels)
            .map(|c| sample_f64(data, off, c, fmt))
            .collect()
    }

    /// Add two rasters pixel-by-pixel, returning a new raster.
    ///
    /// Both inputs must have identical dimensions and channel counts. To avoid
    /// wrap-around, an 8-bit result is promoted to the matching 16-bit format
    /// (`Gray8 -> Gray16`, `Rgb8 -> Rgb16`, `Rgba8 -> Rgba16`), so summing two
    /// `Gray8` images whose values exceed 255 keeps the true sum. 16-bit inputs
    /// have no wider format available, so their sums saturate at `65535`.
    ///
    /// # Panics
    ///
    /// Panics if the two rasters differ in width, height, or channel count,
    /// or if either input is a float raster (float addition lands with a
    /// later arithmetic batch; cast to an unsigned format first). The
    /// libvips-derived callers always add conformable images; the panic turns a
    /// caller bug into an immediate, clear failure rather than silent garbage.
    pub fn add(&self, other: &Raster) -> Raster {
        assert_eq!(
            (self.width(), self.height()),
            (other.width(), other.height()),
            "add: dimension mismatch"
        );
        assert_eq!(
            self.format().channels(),
            other.format().channels(),
            "add: channel-count mismatch"
        );
        assert!(
            !self.format().is_float() && !other.format().is_float(),
            "add: float rasters are not supported yet; cast to an unsigned 8/16-bit format first"
        );

        let width = self.width();
        let height = self.height();
        let channels = self.format().channels();
        let out_fmt = widen(self.format());
        let out_bpc = out_fmt.bytes_per_channel();
        let count = width as usize * height as usize * channels;

        let mut out = vec![0u8; count * out_bpc];
        for i in 0..count {
            let a = read_sample(self, i);
            let b = read_sample(other, i);
            let sum = a + b;
            if out_bpc == 1 {
                out[i] = sum.min(255) as u8;
            } else {
                let v = sum.min(u16::MAX as u32) as u16;
                let bytes = v.to_ne_bytes();
                out[i * 2] = bytes[0];
                out[i * 2 + 1] = bytes[1];
            }
        }
        // The output size is exactly `count * out_bpc`, computed from validated
        // input dimensions, so construction cannot fail.
        Raster::new(width, height, out_fmt, out).expect("add output is well-formed")
    }
}

/// Read the `i`-th channel sample of `raster` (flat channel index) as `u32`.
///
/// Unsigned samples only: `add` guards float inputs out before calling this,
/// and the panic arm keeps a future caller from misreading float bytes as
/// `u16` pairs.
fn read_sample(raster: &Raster, i: usize) -> u32 {
    let data = raster.data();
    match raster.format().bytes_per_channel() {
        1 => data[i] as u32,
        2 => u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as u32,
        _ => panic!("read_sample: float rasters have no u32 sample view"),
    }
}

/// The format one bit-depth wider, used so 8-bit sums do not wrap. 16-bit
/// formats have no wider form here and are returned unchanged (sums saturate).
fn widen(fmt: PixelFormat) -> PixelFormat {
    match fmt {
        PixelFormat::Gray8 => PixelFormat::Gray16,
        PixelFormat::Rgb8 => PixelFormat::Rgb16,
        PixelFormat::Rgba8 => PixelFormat::Rgba16,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// getpoint returns one sample per band for 8-bit grayscale and RGB.
    #[test]
    fn getpoint_reads_bands_8bit() {
        let im = Raster::new(2, 1, PixelFormat::Rgb8, vec![10, 20, 30, 40, 50, 60]).unwrap();
        assert_eq!(im.getpoint(0, 0), vec![10.0, 20.0, 30.0]);
        assert_eq!(im.getpoint(1, 0), vec![40.0, 50.0, 60.0]);

        let g = Raster::new(1, 1, PixelFormat::Gray8, vec![7]).unwrap();
        assert_eq!(g.getpoint(0, 0), vec![7.0]);
    }

    /// getpoint decodes 16-bit samples in native byte order.
    #[test]
    fn getpoint_reads_16bit() {
        let v = 4096u16.to_ne_bytes();
        let im = Raster::new(1, 1, PixelFormat::Gray16, vec![v[0], v[1]]).unwrap();
        assert_eq!(im.getpoint(0, 0), vec![4096.0]);
    }

    /// getpoint round-trips a pixel written with put_pixel.
    #[test]
    fn getpoint_put_pixel_round_trip() {
        let mut im = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        im.put_pixel(2, 3, &[111, 122, 133]);
        assert_eq!(im.getpoint(2, 3), vec![111.0, 122.0, 133.0]);
    }

    /// add combines two grayscale rasters and promotes 8-bit to 16-bit so the
    /// sum does not wrap past 255.
    #[test]
    fn add_promotes_and_sums() {
        let a = Raster::new(2, 1, PixelFormat::Gray8, vec![200, 10]).unwrap();
        let sum = a.add(&a);
        assert_eq!(sum.format(), PixelFormat::Gray16);
        // 200 + 200 = 400 survives because the result is 16-bit.
        assert_eq!(sum.getpoint(0, 0), vec![400.0]);
        assert_eq!(sum.getpoint(1, 0), vec![20.0]);
    }

    /// add works per-band for colour images.
    #[test]
    fn add_colour_per_band() {
        let a = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        let b = Raster::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
        assert_eq!(a.add(&b).getpoint(0, 0), vec![11.0, 22.0, 33.0]);
    }

    /// 16-bit sums saturate at u16::MAX rather than wrapping.
    #[test]
    fn add_16bit_saturates() {
        let hi = 60000u16.to_ne_bytes();
        let a = Raster::new(1, 1, PixelFormat::Gray16, vec![hi[0], hi[1]]).unwrap();
        let sum = a.add(&a);
        assert_eq!(sum.format(), PixelFormat::Gray16);
        assert_eq!(sum.getpoint(0, 0), vec![65535.0]);
    }

    /// add rejects mismatched dimensions.
    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn add_dimension_mismatch_panics() {
        let a = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let b = Raster::zeroed(3, 2, PixelFormat::Gray8).unwrap();
        let _ = a.add(&b);
    }

    /// getpoint reads float samples as their exact stored values,
    /// including negatives and values outside the unsigned ranges.
    #[test]
    fn getpoint_reads_float() {
        let im = Raster::from_f32_samples(
            2,
            1,
            PixelFormat::RgbaF32,
            &[0.5, -1.25, 300.75, 0.0, 65536.5, 1.0, -0.0, 2.5],
        )
        .unwrap();
        assert_eq!(im.getpoint(0, 0), vec![0.5, -1.25, 300.75, 0.0]);
        assert_eq!(im.getpoint(1, 0), vec![65536.5, 1.0, 0.0, 2.5]);
    }

    /// add rejects float rasters loudly instead of misreading their bytes
    /// (float addition lands with a later arithmetic batch).
    #[test]
    #[should_panic(expected = "float rasters are not supported")]
    fn add_float_panics() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let a = Raster::zeroed(2, 2, f1).unwrap();
        let _ = a.add(&a);
    }
}
