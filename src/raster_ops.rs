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

use crate::pixel::{PixelFormat, SampleKind, read_sample_f64};
use crate::raster::Raster;

/// Read the `n`-th channel sample at byte offset `off` as `f64`, honouring the
/// format's sample kind (native byte order for every multi-byte kind).
///
/// Goes through [`read_sample_f64`], the crate's one width-independent
/// sample read. The width-keyed spelling this replaces read four bytes as
/// an `f32` whatever they were, so a 32-bit integer carrier's `1` would
/// have come out of `getpoint` as `1.4e-45` (issue #607).
pub(crate) fn sample_f64(data: &[u8], off: usize, channel: usize, fmt: PixelFormat) -> f64 {
    let kind = fmt.kind();
    read_sample_f64(data, kind, off + channel * kind.bytes())
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
    /// The inputs must share dimensions. Band counts may differ only when one
    /// operand is single-band: it is then **broadcast** across the other's
    /// bands (vips band-broadcast semantics, so `RGB + gray` adds the gray
    /// value to each colour band). The result has `max(bands)` bands.
    ///
    /// The result format follows the libvips `add` promotion table so sums
    /// never wrap:
    ///
    /// * `uchar + uchar` promotes to the matching 16-bit format
    ///   (`Gray8 -> Gray16`, `Rgb8 -> Rgb16`, `Rgba8 -> Rgba16`); a 8-bit sum
    ///   never exceeds `510`, so it is kept exactly.
    /// * if **either** operand is 16-bit, vips promotes to `uint`. This crate
    ///   has no unsigned-32 format, so the promoted result is carried as `f32`
    ///   (a matching `FloatF32` / `RgbaF32`). Every `u16 + u16` sum (at most
    ///   `131070`) is an integer below `2^24` and so is representable in `f32`
    ///   exactly — the true sum is kept, not saturated at `65535`.
    ///
    /// # Panics
    ///
    /// Panics if the two rasters differ in width or height; if their band
    /// counts differ and *neither* is single-band (so broadcasting is
    /// ambiguous); or if either input is a float raster (float addition lands
    /// with a later arithmetic batch; cast to an unsigned format first). The
    /// libvips-derived callers always add conformable images; the panic turns a
    /// caller bug into an immediate, clear failure rather than silent garbage.
    pub fn add(&self, other: &Raster) -> Raster {
        assert_eq!(
            (self.width(), self.height()),
            (other.width(), other.height()),
            "add: dimension mismatch"
        );
        assert!(
            !self.format().is_float() && !other.format().is_float(),
            "add: float rasters are not supported yet; cast to an unsigned 8/16-bit format first"
        );
        let a_bands = self.format().channels();
        let b_bands = other.format().channels();
        assert!(
            a_bands == b_bands || a_bands == 1 || b_bands == 1,
            "add: channel-count mismatch: {a_bands} vs {b_bands} \
             (broadcasting requires one operand to be single-band)"
        );

        let width = self.width();
        let height = self.height();
        let pixels = width as usize * height as usize;
        let out_bands = a_bands.max(b_bands);
        // Flat sample index into an operand for output band `c` at pixel `p`,
        // folding the single-band broadcast: a 1-band operand always reads
        // band 0.
        let idx = |bands: usize, p: usize, c: usize| p * bands + if bands == 1 { 0 } else { c };

        // vips add promotion: uchar+uchar -> ushort; if either operand is
        // 16-bit -> uint (carried here as f32; see the doc comment). Keyed
        // on the kind, so a two-byte *signed* carrier would not be mistaken
        // for the `ushort` this rule is about (issue #607).
        let promote_float =
            self.format().kind() == SampleKind::U16 || other.format().kind() == SampleKind::U16;

        if promote_float {
            let out_fmt = PixelFormat::with_kind(out_bands, SampleKind::F32)
                .expect("out_bands is 1..=max(input bands), so the float format exists");
            let mut samples = Vec::with_capacity(pixels * out_bands);
            for p in 0..pixels {
                for c in 0..out_bands {
                    let a = read_sample(self, idx(a_bands, p, c));
                    let b = read_sample(other, idx(b_bands, p, c));
                    samples.push((a + b) as f32);
                }
            }
            Raster::from_f32_samples(width, height, out_fmt, &samples)
                .expect("add output is well-formed")
        } else {
            // Both operands are 8-bit: promote to 16-bit (vips uchar+uchar ->
            // ushort). The sum is at most 510, so `.min` never actually clips.
            let out_fmt = PixelFormat::with_kind(out_bands, SampleKind::U16)
                .expect("out_bands is 1..=max(input bands), so the 16-bit format exists");
            let mut out = vec![0u8; pixels * out_bands * 2];
            for p in 0..pixels {
                for c in 0..out_bands {
                    let a = read_sample(self, idx(a_bands, p, c));
                    let b = read_sample(other, idx(b_bands, p, c));
                    let v = (a + b).min(u16::MAX as u32) as u16;
                    let bytes = v.to_ne_bytes();
                    let o = (p * out_bands + c) * 2;
                    out[o] = bytes[0];
                    out[o + 1] = bytes[1];
                }
            }
            // The output size is exactly `pixels * out_bands * 2`, computed
            // from validated input dimensions, so construction cannot fail.
            Raster::new(width, height, out_fmt, out).expect("add output is well-formed")
        }
    }
}

/// Read the `i`-th channel sample of `raster` (flat channel index) as `u32`.
///
/// Unsigned samples only: `add` guards float inputs out before calling this,
/// and the panic arm keeps a future caller from misreading float bytes as
/// `u16` pairs.
fn read_sample(raster: &Raster, i: usize) -> u32 {
    let data = raster.data();
    let kind = raster.format().kind();
    match kind {
        SampleKind::U8 => data[i] as u32,
        SampleKind::U16 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as u32,
        SampleKind::I8 | SampleKind::I16 | SampleKind::U32 | SampleKind::I32 | SampleKind::F32 => {
            panic!("read_sample: {kind:?} has no unsigned 8/16-bit sample view")
        }
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

    /// 16-bit sums promote to float and keep the true sum instead of
    /// saturating at `u16::MAX`. vips `add ushort ushort` promotes to `uint`
    /// (verified with the oracle: `60000 + 10000 = 70000`, format `uint`);
    /// this crate has no unsigned-32 format, so the promoted result is carried
    /// as `f32`, which represents the sum exactly.
    #[test]
    fn add_16bit_promotes_to_float_true_sum() {
        let hi = 60000u16.to_ne_bytes();
        let lo = 10000u16.to_ne_bytes();
        let a = Raster::new(1, 1, PixelFormat::Gray16, vec![hi[0], hi[1]]).unwrap();
        let b = Raster::new(1, 1, PixelFormat::Gray16, vec![lo[0], lo[1]]).unwrap();
        let sum = a.add(&b);
        assert!(sum.format().is_float());
        assert_eq!(sum.format().channels(), 1);
        // 70000 > u16::MAX (65535); the old behaviour clamped to 65535.
        assert_eq!(sum.getpoint(0, 0), vec![70000.0]);
        // The largest possible u16 + u16 sum is representable exactly in f32.
        let m = u16::MAX.to_ne_bytes();
        let big = Raster::new(1, 1, PixelFormat::Gray16, vec![m[0], m[1]]).unwrap();
        assert_eq!(big.add(&big).getpoint(0, 0), vec![131070.0]);
    }

    /// A single-band operand broadcasts across the other operand's bands
    /// (vips band-broadcast). `RGB + gray` adds the gray value to each colour
    /// band (verified with the oracle: `[10,20,30] + 5 = [15,25,35]`).
    #[test]
    fn add_broadcasts_single_band() {
        let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
        let gray = Raster::new(1, 1, PixelFormat::Gray8, vec![5]).unwrap();
        // 8-bit + 8-bit promotes to 16-bit; the gray band is added to each.
        let out = rgb.add(&gray);
        assert_eq!(out.format().channels(), 3);
        assert_eq!(out.format().kind(), SampleKind::U16);
        assert_eq!(out.getpoint(0, 0), vec![15.0, 25.0, 35.0]);
        // Broadcast is symmetric.
        assert_eq!(gray.add(&rgb).getpoint(0, 0), vec![15.0, 25.0, 35.0]);
    }

    /// add rejects mismatched dimensions.
    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn add_dimension_mismatch_panics() {
        let a = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let b = Raster::zeroed(3, 2, PixelFormat::Gray8).unwrap();
        let _ = a.add(&b);
    }

    /// add rejects a band-count mismatch when neither operand is single-band
    /// (broadcasting is only defined against a single band).
    #[test]
    #[should_panic(expected = "channel-count mismatch")]
    fn add_incompatible_band_counts_panic() {
        let a = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let b = Raster::zeroed(2, 2, PixelFormat::Rgba8).unwrap();
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
        let f1 = PixelFormat::with_kind(1, SampleKind::F32).unwrap();
        let a = Raster::zeroed(2, 2, f1).unwrap();
        let _ = a.add(&a);
    }

    /**
     * Tests that this module dispatches on sample kind and never on byte
     * width, by asserting that neither the byte-width accessor on
     * [`PixelFormat`] nor its width-keyed constructor survives in
     * `src/raster_ops.rs`.
     * Works by scanning the module's own source, compiled in with
     * `include_str!`, for the accessor's name; the needle is spelled in two
     * halves so this assertion is not itself a hit. A byte width is not a
     * sample kind: four bytes is `f32` today and would be `u32` under issue
     * #517, so the sites this replaced would hand `getpoint` a 32-bit integer sample as `1.4e-45` (issue #607).
     * Input: `src/raster_ops.rs` -> Output: zero occurrences.
     */
    #[test]
    fn raster_ops_does_not_dispatch_on_byte_width() {
        const SRC: &str = include_str!("raster_ops.rs");
        let needles = [
            concat!("bytes_per_", "channel"),
            concat!("with_", "channels"),
        ];
        // Positive control: the same scan over the same string finds a token
        // that is present, so the zero below is a real zero and not the
        // vacuous pass an empty read would give.
        assert!(
            SRC.contains(concat!("fn ", "sample_f64")),
            "positive control failed: the scan cannot see this module's source"
        );
        for needle in needles {
            assert_eq!(
                SRC.matches(needle).count(),
                0,
                "{needle} is back in src/raster_ops.rs; dispatch on \
                 PixelFormat::kind() and PixelFormat::with_kind() instead"
            );
        }
    }
}
