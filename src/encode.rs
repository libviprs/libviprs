//! Raster encoders: the `save_*`, `encode_*`, and `jpegsave_buffer` family the
//! ported foreign cells call on [`Raster`].
//!
//! ## What is real and what is a typed stub
//!
//! Every method here reports failures through the shared [`EncodeError`] /
//! [`SaveError`] spine in [`crate::codec`] and [`crate::imageio`]. The pixel
//! encoders split into two groups by what the pure-Rust build can actually
//! emit:
//!
//! * **Real, backed by `image` 0.25:** [`Raster::encode_jpeg`],
//!   [`Raster::encode_jpeg_options`], [`Raster::save_jpeg`],
//!   [`Raster::jpegsave_buffer`], [`Raster::encode_png`], [`Raster::save_png`].
//! * **Real, hand-rolled here on `flate2`** because `image`'s PNG encoder
//!   exposes neither knob: [`Raster::encode_png_interlaced`] (Adam7) and
//!   [`Raster::encode_png_palette`] (median-cut quantized indexed PNG).
//! * **Typed [`EncodeError::Unsupported`] stubs**, because this build has no
//!   accessible encoder for them: [`Raster::jpegsave_buffer_restart`]
//!   (`"jpeg-restart"`: `image`'s JPEG encoder writes no restart markers).
//!
//! WebP and GIF used to keep their stubs here too. They now live in
//! [`crate::webp`] and [`crate::gif`], one file per format, so the four
//! format lanes each own exactly one file instead of all four rewriting this
//! header and the adjacent stub bodies (issue #563).
//!
//! ## Subsampling caveat
//!
//! `image` 0.25's JPEG encoder fixes its chroma sampling factors and offers no
//! public subsample control, so [`Raster::encode_jpeg_options`] accepts the
//! [`JpegSubsample`] argument for signature and contract compatibility but
//! currently ignores it: the quality is applied for real, while every subsample
//! mode selects the same encoder configuration. The argument keeps the call
//! site and the libvips `subsample_mode` mapping resolving today; when a
//! subsample-capable encoder lands the mode will drive the sampling factors
//! without a signature change.

use crate::codec::{EncodeError, JpegSubsample};
use crate::imageio::SaveError;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;
use std::path::Path;

// ---------------------------------------------------------------------------
// JPEG
// ---------------------------------------------------------------------------

impl Raster {
    /// Encode the raster as JPEG bytes at the given quality (1-100).
    ///
    /// Equivalent to [`Raster::encode_jpeg_options`] with
    /// [`JpegSubsample::Auto`].
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] if the `image` encoder rejects the raster (for
    /// example a multiband or float [`PixelFormat`] with no JPEG colour type).
    pub fn encode_jpeg(&self, quality: u8) -> Result<Vec<u8>, EncodeError> {
        self.encode_jpeg_options(quality, JpegSubsample::Auto)
    }

    /// Encode the raster as JPEG bytes at the given quality, with a requested
    /// chroma [`JpegSubsample`] mode.
    ///
    /// The quality is applied for real; the subsample mode is accepted but
    /// currently ignored (see the [module docs](crate::encode)).
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] if the `image` encoder rejects the raster.
    pub fn encode_jpeg_options(
        &self,
        quality: u8,
        subsample: JpegSubsample,
    ) -> Result<Vec<u8>, EncodeError> {
        // Accepted for the libvips `subsample_mode` contract; `image` 0.25 has
        // no public knob to vary the sampling factors, so the mode is ignored
        // and every mode selects the same encoder configuration.
        let _ = subsample;
        let mut buf = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            std::io::Cursor::new(&mut buf),
            quality.clamp(1, 100),
        );
        let ct = image_color_type(self.format())?;
        image::ImageEncoder::write_image(
            encoder,
            self.data(),
            self.width(),
            self.height(),
            ct.into(),
        )
        .map_err(EncodeError::encode)?;
        Ok(buf)
    }

    /// Encode JPEG bytes at `quality`, mapping a libvips `subsample_mode`
    /// string to a [`JpegSubsample`]: `Some("off"|"4:4:4")` -> `Off`,
    /// `Some("on"|"4:2:0")` -> `On`, and `None` / `Some("auto")` / anything
    /// unrecognized -> `Auto`.
    ///
    /// # Errors
    ///
    /// As [`Raster::encode_jpeg_options`].
    pub fn jpegsave_buffer(
        &self,
        quality: u8,
        subsample_mode: Option<&str>,
    ) -> Result<Vec<u8>, EncodeError> {
        self.encode_jpeg_options(quality, parse_subsample_mode(subsample_mode))
    }

    /// libvips `jpegsave_buffer` with a `restart_interval` option.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`] with format `"jpeg-restart"`:
    /// `image` 0.25's JPEG encoder writes no restart markers and exposes no
    /// restart-interval knob, so the option cannot be honoured in this build.
    pub fn jpegsave_buffer_restart(&self, restart_interval: u32) -> Result<Vec<u8>, EncodeError> {
        let _ = restart_interval;
        Err(EncodeError::unsupported("jpeg-restart"))
    }

    /// Save the raster to `path` as JPEG at the given quality (1-100).
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] if the encoder rejects the raster, or
    /// [`SaveError::Io`] if the file write fails.
    pub fn save_jpeg(&self, path: &Path, quality: u8) -> Result<(), SaveError> {
        let bytes = self.encode_jpeg(quality).map_err(save_error_from_encode)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

impl Raster {
    /// Encode the raster as PNG bytes at the given DEFLATE compression level
    /// (0 = store, 9 = maximum; values above 9 are clamped).
    ///
    /// Lossless: decoding the result reproduces the source pixels exactly.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] if the `image` encoder rejects the raster (a
    /// multiband or float [`PixelFormat`] with no PNG colour type).
    pub fn encode_png(&self, compression: u8) -> Result<Vec<u8>, EncodeError> {
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        let mut buf = Vec::new();
        let encoder = PngEncoder::new_with_quality(
            std::io::Cursor::new(&mut buf),
            CompressionType::Level(compression.min(9)),
            FilterType::Adaptive,
        );
        let ct = image_color_type(self.format())?;
        image::ImageEncoder::write_image(
            encoder,
            self.data(),
            self.width(),
            self.height(),
            ct.into(),
        )
        .map_err(EncodeError::encode)?;
        Ok(buf)
    }

    /// Save the raster to `path` as PNG at the given compression level.
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] if the encoder rejects the raster, or
    /// [`SaveError::Io`] if the file write fails.
    pub fn save_png(&self, path: &Path, compression: u8) -> Result<(), SaveError> {
        let bytes = self
            .encode_png(compression)
            .map_err(save_error_from_encode)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Encode the raster as an Adam7-interlaced PNG.
    ///
    /// `image` 0.25's `PngEncoder` cannot emit interlaced output, so this path
    /// is hand-rolled: it writes the seven Adam7 reduced images with a
    /// `None` per-scanline filter and DEFLATEs them through `flate2`. The
    /// result is lossless and carries `interlace_method = 1` in its IHDR.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] for a multiband or float [`PixelFormat`] that
    /// has no PNG colour type, or [`EncodeError::Io`] if DEFLATE fails.
    pub fn encode_png_interlaced(&self) -> Result<Vec<u8>, EncodeError> {
        let (color_type, bit_depth, channels) = png_colour_params(self.format())?;
        let width = self.width();
        let data = self.data();
        let raw = build_scanlines(width, self.height(), true, |px, py, out| {
            let sample = (py as usize * width as usize + px as usize) * channels;
            if bit_depth == 8 {
                out.extend_from_slice(&data[sample..sample + channels]);
            } else {
                for c in 0..channels {
                    let o = (sample + c) * 2;
                    let v = u16::from_ne_bytes([data[o], data[o + 1]]);
                    out.extend_from_slice(&v.to_be_bytes());
                }
            }
        });
        let idat = zlib_compress(&raw, 6)?;
        let mut out = Vec::with_capacity(idat.len() + 64);
        out.extend_from_slice(&PNG_SIGNATURE);
        write_ihdr(&mut out, width, self.height(), bit_depth, color_type, 1);
        write_chunk(&mut out, b"IDAT", &idat);
        write_chunk(&mut out, b"IEND", &[]);
        Ok(out)
    }

    /// Encode the raster as an indexed (palette) PNG with at most `max_colours`
    /// palette entries (clamped to 1..=256).
    ///
    /// Colours are reduced with a self-contained median-cut quantizer: when the
    /// image already fits within `max_colours` distinct colours the palette is
    /// exact and the round-trip is lossless; otherwise each pixel maps to the
    /// nearest representative. Requires an 8-bit raster
    /// (`Gray8` / `Rgb8` / `Rgba8`).
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] for a non-8-bit raster, or [`EncodeError::Io`]
    /// if DEFLATE fails.
    pub fn encode_png_palette(&self, max_colours: u32) -> Result<Vec<u8>, EncodeError> {
        let max = max_colours.clamp(1, 256) as usize;
        let channels = match self.format() {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            other => {
                return Err(EncodeError::encode(format!(
                    "palette PNG encode requires an 8-bit raster (Gray8/Rgb8/Rgba8), got {other:?}"
                )));
            }
        };
        let width = self.width();
        let height = self.height();
        let data = self.data();
        let npix = width as usize * height as usize;
        let mut pixels: Vec<[u8; 4]> = Vec::with_capacity(npix);
        for p in 0..npix {
            let base = p * channels;
            pixels.push(match channels {
                1 => {
                    let g = data[base];
                    [g, g, g, 255]
                }
                3 => [data[base], data[base + 1], data[base + 2], 255],
                _ => [data[base], data[base + 1], data[base + 2], data[base + 3]],
            });
        }

        let (palette, indices) = quantize(&pixels, max);
        let mut plte = Vec::with_capacity(palette.len() * 3);
        let mut trns = Vec::with_capacity(palette.len());
        let mut needs_trns = false;
        for c in &palette {
            plte.extend_from_slice(&c[0..3]);
            trns.push(c[3]);
            needs_trns |= c[3] != 255;
        }

        let raw = build_scanlines(width, height, false, |px, py, out| {
            out.push(indices[py as usize * width as usize + px as usize]);
        });
        let idat = zlib_compress(&raw, 9)?;

        let mut out = Vec::with_capacity(idat.len() + plte.len() + 64);
        out.extend_from_slice(&PNG_SIGNATURE);
        write_ihdr(&mut out, width, height, 8, 3, 0);
        write_chunk(&mut out, b"PLTE", &plte);
        if needs_trns {
            write_chunk(&mut out, b"tRNS", &trns);
        }
        write_chunk(&mut out, b"IDAT", &idat);
        write_chunk(&mut out, b"IEND", &[]);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The 8-byte PNG file signature.
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Adam7 passes as `(start_col, start_row, col_step, row_step)`.
const ADAM7: [(u32, u32, u32, u32); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// Map a libvips `subsample_mode` string to a [`JpegSubsample`].
fn parse_subsample_mode(mode: Option<&str>) -> JpegSubsample {
    match mode.map(|m| m.trim().to_ascii_lowercase()) {
        Some(m) if m == "off" || m == "4:4:4" || m == "444" => JpegSubsample::Off,
        Some(m) if m == "on" || m == "4:2:0" || m == "420" => JpegSubsample::On,
        _ => JpegSubsample::Auto,
    }
}

/// The `image` colour type for a [`PixelFormat`], or an [`EncodeError`] for the
/// compute-intermediate formats (multiband / float) that have none.
fn image_color_type(fmt: PixelFormat) -> Result<image::ColorType, EncodeError> {
    Ok(match fmt {
        PixelFormat::Gray8 => image::ColorType::L8,
        PixelFormat::Gray16 => image::ColorType::L16,
        PixelFormat::Rgb8 => image::ColorType::Rgb8,
        PixelFormat::Rgba8 => image::ColorType::Rgba8,
        PixelFormat::Rgb16 => image::ColorType::Rgb16,
        PixelFormat::Rgba16 => image::ColorType::Rgba16,
        PixelFormat::Multi8(_) | PixelFormat::Multi16(_) => {
            return Err(EncodeError::encode(format!(
                "multiband raster ({} bands) has no image colour type; reduce to 1/3/4 bands first",
                fmt.channels()
            )));
        }
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => {
            return Err(EncodeError::encode(format!(
                "float raster ({fmt:?}) has no 8/16-bit image colour type; cast first"
            )));
        }
        // The `image` crate's widest integer colour type is 16-bit, so a
        // `uint` raster is refused for the same reason a float one is
        // rather than being narrowed behind the caller's back (issue #517).
        PixelFormat::Uint32(_) => {
            return Err(EncodeError::encode(format!(
                "32-bit unsigned raster ({fmt:?}) has no 8/16-bit image colour type; cast first"
            )));
        }
    })
}

/// PNG `(colour_type, bit_depth, channels)` for a [`PixelFormat`], for the
/// hand-rolled interlaced writer.
fn png_colour_params(fmt: PixelFormat) -> Result<(u8, u8, usize), EncodeError> {
    Ok(match fmt {
        PixelFormat::Gray8 => (0, 8, 1),
        PixelFormat::Gray16 => (0, 16, 1),
        PixelFormat::Rgb8 => (2, 8, 3),
        PixelFormat::Rgb16 => (2, 16, 3),
        PixelFormat::Rgba8 => (6, 8, 4),
        PixelFormat::Rgba16 => (6, 16, 4),
        other => {
            return Err(EncodeError::encode(format!(
                "interlaced PNG encode requires a gray/RGB/RGBA raster, got {other:?}"
            )));
        }
    })
}

/// Build the filtered PNG scanline stream (each row prefixed with a `None`
/// filter byte) for either a single progressive pass or the seven Adam7 passes.
fn build_scanlines<F: FnMut(u32, u32, &mut Vec<u8>)>(
    width: u32,
    height: u32,
    interlaced: bool,
    mut emit: F,
) -> Vec<u8> {
    let progressive = [(0u32, 0u32, 1u32, 1u32)];
    let passes: &[(u32, u32, u32, u32)] = if interlaced { &ADAM7 } else { &progressive };
    let mut raw = Vec::new();
    for &(sc, sr, cs, rs) in passes {
        if sc >= width || sr >= height {
            continue;
        }
        let pass_w = (width - sc).div_ceil(cs);
        let pass_h = (height - sr).div_ceil(rs);
        for j in 0..pass_h {
            let py = sr + j * rs;
            raw.push(0); // filter type: None
            for i in 0..pass_w {
                emit(sc + i * cs, py, &mut raw);
            }
        }
    }
    raw
}

/// DEFLATE `raw` into a zlib stream (PNG IDAT payload) at the given level.
fn zlib_compress(raw: &[u8], level: u32) -> Result<Vec<u8>, EncodeError> {
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(raw)?;
    Ok(enc.finish()?)
}

/// Append an IHDR chunk.
fn write_ihdr(
    out: &mut Vec<u8>,
    width: u32,
    height: u32,
    bit_depth: u8,
    colour_type: u8,
    interlace: u8,
) {
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(bit_depth);
    ihdr.push(colour_type);
    ihdr.push(0); // compression method: DEFLATE
    ihdr.push(0); // filter method: adaptive filtering with five basic types
    ihdr.push(interlace);
    write_chunk(out, b"IHDR", &ihdr);
}

/// Append a length-prefixed, CRC-suffixed PNG chunk.
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = png_crc32(&out[crc_start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// The PNG chunk CRC (CRC-32/ISO-HDLC over the chunk type and data).
fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Median-cut colour quantization over RGBA pixels.
///
/// Returns the palette (at most `max` entries) and one palette index per input
/// pixel. When the distinct-colour count already fits in `max` the palette is
/// exact, so the mapping is lossless.
fn quantize(pixels: &[[u8; 4]], max: usize) -> (Vec<[u8; 4]>, Vec<u8>) {
    use std::collections::HashMap;

    let palette = quantize_palette(pixels, max);
    let mut cache: HashMap<[u8; 4], u8> = HashMap::new();
    let indices = pixels
        .iter()
        .map(|&p| *cache.entry(p).or_insert_with(|| nearest(&palette, p)))
        .collect();
    (palette, indices)
}

/// The palette half of [`quantize`]: median-cut over the distinct colours in
/// `pixels`, weighted by how often each occurs, capped at `max` entries.
///
/// Split out so [`crate::gif`] can take the palette without also paying for a
/// nearest-colour index pass it is about to redo with error diffusion.
///
/// The distinct colours are **sorted** before the split loop. They arrive from
/// a `HashMap`, whose iteration order the default `RandomState` reseeds per
/// process, so without the sort both the box seeding and the fits-in-`max`
/// fast path produce a palette in a different order on every run. That made
/// [`Raster::encode_png_palette`] emit different bytes for identical input,
/// which is the one thing a content-addressed store (see
/// [`crate::dedupe`]) cannot tolerate.
pub(crate) fn quantize_palette(pixels: &[[u8; 4]], max: usize) -> Vec<[u8; 4]> {
    use std::collections::HashMap;

    let mut counts: HashMap<[u8; 4], u32> = HashMap::new();
    for &p in pixels {
        *counts.entry(p).or_insert(0) += 1;
    }
    let mut uniques: Vec<([u8; 4], u32)> = counts.into_iter().collect();
    uniques.sort_unstable_by_key(|&(c, _)| c);

    if uniques.len() <= max {
        return uniques.iter().map(|&(c, _)| c).collect();
    }

    let mut boxes: Vec<Vec<([u8; 4], u32)>> = vec![uniques];
    while boxes.len() < max {
        // Pick the splittable box whose widest channel spans the most.
        let mut best: Option<(usize, usize, i32)> = None;
        for (bi, b) in boxes.iter().enumerate() {
            if b.len() < 2 {
                continue;
            }
            for ch in 0..4 {
                let mut lo = 255i32;
                let mut hi = 0i32;
                for &(c, _) in b {
                    let v = c[ch] as i32;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                let range = hi - lo;
                let better = match best {
                    None => true,
                    Some((_, _, r)) => range > r,
                };
                if better {
                    best = Some((bi, ch, range));
                }
            }
        }
        let Some((bi, ch, _)) = best else {
            break; // no box can be split further
        };
        let mut b = boxes.swap_remove(bi);
        b.sort_by_key(|&(c, _)| c[ch]);
        let right = b.split_off(b.len() / 2);
        boxes.push(b);
        boxes.push(right);
    }

    boxes.iter().map(|b| box_average(b)).collect()
}

/// The count-weighted average colour of a median-cut box.
fn box_average(b: &[([u8; 4], u32)]) -> [u8; 4] {
    let mut acc = [0u64; 4];
    let mut total = 0u64;
    for &(c, n) in b {
        let n = n as u64;
        for (a, &cv) in acc.iter_mut().zip(c.iter()) {
            *a += cv as u64 * n;
        }
        total += n;
    }
    let total = total.max(1);
    let mut out = [0u8; 4];
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = (a / total) as u8;
    }
    out
}

/// The index of the palette entry nearest `c` in squared-Euclidean RGBA space.
pub(crate) fn nearest(palette: &[[u8; 4]], c: [u8; 4]) -> u8 {
    let mut best_i = 0usize;
    let mut best_d = i64::MAX;
    for (i, &q) in palette.iter().enumerate() {
        let d: i64 = c
            .iter()
            .zip(q.iter())
            .map(|(&a, &b)| {
                let dv = a as i64 - b as i64;
                dv * dv
            })
            .sum();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i as u8
}

/// Map an [`EncodeError`] from the byte-producing PNG path onto the
/// file-oriented [`SaveError`] returned by [`Raster::save_png`].
fn save_error_from_encode(e: EncodeError) -> SaveError {
    match e {
        EncodeError::Io(io) => SaveError::Io(io),
        other => SaveError::Encode(SinkError::EncodeMsg(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::decode_bytes;
    use std::collections::HashSet;

    fn rgb8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Raster {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    fn rgba8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Raster {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        Raster::new(w, h, PixelFormat::Rgba8, data).unwrap()
    }

    /// Build an `Rgb16` raster whose samples are laid out native-endian, the
    /// same byte order [`decode_bytes`] produces, so an exact round-trip must
    /// reproduce the source buffer verbatim.
    fn rgb16(w: u32, h: u32, f: impl Fn(u32, u32) -> [u16; 3]) -> Raster {
        let mut data = Vec::with_capacity((w * h * 6) as usize);
        for y in 0..h {
            for x in 0..w {
                for sample in f(x, y) {
                    data.extend_from_slice(&sample.to_ne_bytes());
                }
            }
        }
        Raster::new(w, h, PixelFormat::Rgb16, data).unwrap()
    }

    /// Walk the length-prefixed PNG chunk stream and report whether a chunk of
    /// `kind` is present. Robust against a coincidental byte match inside an
    /// IDAT payload, unlike a raw byte scan.
    fn png_has_chunk(buf: &[u8], kind: &[u8; 4]) -> bool {
        let mut pos = PNG_SIGNATURE.len();
        while pos + 8 <= buf.len() {
            let len =
                u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            if &buf[pos + 4..pos + 8] == kind {
                return true;
            }
            // length(4) + type(4) + data(len) + crc(4)
            pos += 12 + len;
        }
        false
    }

    fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sum: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
            .sum();
        sum as f64 / a.len() as f64
    }

    #[test]
    fn png_encode_is_lossless_roundtrip() {
        let im = rgb8(32, 24, |x, y| {
            [(x * 7) as u8, (y * 9) as u8, ((x + y) * 3) as u8]
        });
        let fast = im.encode_png(0).unwrap();
        let best = im.encode_png(9).unwrap();
        assert!(best.len() <= fast.len());
        assert_eq!(decode_bytes(&fast).unwrap().data(), im.data());
        assert_eq!(decode_bytes(&best).unwrap().data(), im.data());
    }

    #[test]
    fn png_interlaced_roundtrip_is_lossless() {
        let im = rgb8(9, 7, |x, y| [(x * 20) as u8, (y * 30) as u8, 40]);
        let buf = im.encode_png_interlaced().unwrap();
        assert_eq!(&buf[..4], &[0x89, b'P', b'N', b'G']);
        // IHDR interlace byte (Adam7 == 1): sig(8) + len(4) + "IHDR"(4) + data[12].
        assert_eq!(buf[8 + 4 + 4 + 12], 1);
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        assert_eq!(im2.data(), im.data());
    }

    #[test]
    fn png_interlaced_16bit_roundtrip_is_lossless() {
        // Native-endian u16 samples whose high and low bytes differ on every
        // channel, so any slip in the Adam7 emit closure's native-to-big-endian
        // conversion would corrupt the round-trip. Odd 9x7 dimensions force
        // several Adam7 passes to fire.
        let im = rgb16(9, 7, |x, y| {
            [
                x as u16 * 4096 + 1,
                y as u16 * 8192 + 2,
                (x + y) as u16 * 2048 + 3,
            ]
        });
        let buf = im.encode_png_interlaced().unwrap();
        assert_eq!(&buf[..4], &[0x89, b'P', b'N', b'G']);
        // IHDR bit-depth byte (16): sig(8) + len(4) + "IHDR"(4) + data[8].
        assert_eq!(buf[8 + 4 + 4 + 8], 16);
        // IHDR interlace byte (Adam7 == 1): data[12].
        assert_eq!(buf[8 + 4 + 4 + 12], 1);
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        assert_eq!(im2.format(), PixelFormat::Rgb16);
        assert_eq!(im2.data(), im.data());
    }

    #[test]
    fn png_palette_exact_is_lossless_and_indexed() {
        let palette = [[10u8, 20, 30], [200, 10, 10], [10, 200, 10], [10, 10, 200]];
        let im = rgb8(16, 16, |x, y| palette[((x / 8) + 2 * (y / 8)) as usize]);
        let buf = im.encode_png_palette(256).unwrap();
        // IHDR colour-type byte (indexed == 3): sig(8) + len(4) + "IHDR"(4) + data[9].
        assert_eq!(buf[8 + 4 + 4 + 9], 3);
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        assert_eq!(im2.data(), im.data());
    }

    #[test]
    fn png_palette_quantizes_many_colours() {
        let im = rgb8(48, 48, |x, y| {
            [(x * 5) as u8, (y * 5) as u8, ((x * y) % 256) as u8]
        });
        let buf = im.encode_png_palette(16).unwrap();
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        let colours: HashSet<[u8; 3]> = im2.data().as_chunks::<3>().0.iter().copied().collect();
        assert!(
            colours.len() <= 16,
            "quantized palette had {} colours",
            colours.len()
        );
    }

    #[test]
    fn png_palette_rgba8_preserves_alpha_and_writes_trns() {
        // Four exact colours, two of them non-opaque, so the palette is lossless
        // and a tRNS chunk must be emitted to carry the per-entry alpha through
        // the round-trip. Exercises the 4-channel palette input arm.
        let colours = [
            [10u8, 20, 30, 255],
            [200, 10, 10, 128],
            [10, 200, 10, 255],
            [10, 10, 200, 0],
        ];
        let im = rgba8(16, 16, |x, y| colours[((x / 8) + 2 * (y / 8)) as usize]);
        let buf = im.encode_png_palette(256).unwrap();
        // IHDR colour-type byte (indexed == 3): sig(8) + len(4) + "IHDR"(4) + data[9].
        assert_eq!(buf[8 + 4 + 4 + 9], 3);
        assert!(
            png_has_chunk(&buf, b"tRNS"),
            "an indexed PNG with non-opaque pixels must carry a tRNS chunk"
        );
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        assert_eq!(im2.format(), PixelFormat::Rgba8);
        assert_eq!(im2.data(), im.data());
    }

    #[test]
    fn png_palette_gray8_covers_single_channel_input() {
        // Sixteen distinct gray levels stay well within the palette budget, so
        // the encode is lossless. This exercises the 1-channel palette input
        // arm, which expands each gray sample to an opaque RGB triple; with no
        // non-opaque pixel the indexed PNG carries no tRNS and decodes to Rgb8.
        let im = Raster::new(4, 4, PixelFormat::Gray8, (0..16u8).collect()).unwrap();
        let buf = im.encode_png_palette(256).unwrap();
        assert_eq!(buf[8 + 4 + 4 + 9], 3); // indexed colour type
        assert!(
            !png_has_chunk(&buf, b"tRNS"),
            "a fully opaque palette must not emit a tRNS chunk"
        );
        let im2 = decode_bytes(&buf).unwrap();
        assert_eq!(im2.width(), im.width());
        assert_eq!(im2.height(), im.height());
        assert_eq!(im2.format(), PixelFormat::Rgb8);
        let expected: Vec<u8> = (0..16u8).flat_map(|g| [g, g, g]).collect();
        assert_eq!(im2.data(), expected.as_slice());
    }

    #[test]
    fn jpeg_encode_decodes_within_tolerance() {
        let im = rgb8(64, 64, |x, y| [(x * 4) as u8, (y * 4) as u8, 128]);
        let q10 = im.encode_jpeg(10).unwrap();
        let q90 = im.encode_jpeg(90).unwrap();
        assert!(q10.len() < q90.len(), "q10={} q90={}", q10.len(), q90.len());
        let back = decode_bytes(&q90).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        let diff = mean_abs_diff(back.data(), im.data());
        assert!(diff < 12.0, "jpeg q90 mean abs diff {diff} too high");
    }

    #[test]
    fn jpeg_subsample_mode_mapping() {
        assert_eq!(parse_subsample_mode(None), JpegSubsample::Auto);
        assert_eq!(parse_subsample_mode(Some("auto")), JpegSubsample::Auto);
        assert_eq!(parse_subsample_mode(Some("on")), JpegSubsample::On);
        assert_eq!(parse_subsample_mode(Some("4:2:0")), JpegSubsample::On);
        assert_eq!(parse_subsample_mode(Some("off")), JpegSubsample::Off);
        assert_eq!(parse_subsample_mode(Some("4:4:4")), JpegSubsample::Off);
        assert_eq!(parse_subsample_mode(Some("nonsense")), JpegSubsample::Auto);
    }

    #[test]
    fn jpegsave_buffer_matches_encode_jpeg_default() {
        let im = rgb8(32, 32, |x, y| [(x * 8) as u8, (y * 8) as u8, 64]);
        let via_buffer = im.jpegsave_buffer(80, None).unwrap();
        let via_encode = im.encode_jpeg(80).unwrap();
        assert_eq!(via_buffer, via_encode);
        let back = decode_bytes(&via_buffer).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
    }

    #[test]
    fn jpegsave_buffer_restart_is_unsupported() {
        let im = rgb8(8, 8, |_, _| [1, 2, 3]);
        match im.jpegsave_buffer_restart(10) {
            Err(EncodeError::Unsupported { format }) => assert_eq!(format, "jpeg-restart"),
            other => panic!("expected Unsupported(jpeg-restart), got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_jpeg_and_png_write_decodable_files() {
        let im = rgb8(20, 12, |x, y| [(x * 10) as u8, (y * 12) as u8, 77]);
        let dir = tempfile::tempdir().unwrap();

        let jpg = dir.path().join("out.jpg");
        im.save_jpeg(&jpg, 85).unwrap();
        let from_jpg = crate::source::decode_file(&jpg).unwrap();
        assert_eq!(from_jpg.width(), im.width());
        assert_eq!(from_jpg.height(), im.height());

        let png = dir.path().join("out.png");
        im.save_png(&png, 6).unwrap();
        let from_png = crate::source::decode_file(&png).unwrap();
        assert_eq!(from_png.data(), im.data());
    }

    /**
     * Tests that the 8/16-bit container encoders refuse the unsigned
     * 32-bit carrier with a typed error naming it, rather than narrowing
     * it behind the caller's back or reading it as float.
     * Works by encoding a `Uint32` raster to PNG and asserting the error
     * text names the carrier, with the float refusal beside it as the
     * control that the two carriers get distinct messages even though they
     * share a byte width.
     * Input: Uint32(1) -> Err naming "32-bit unsigned"; FloatF32(1) -> Err
     * naming "float".
     */
    #[test]
    fn the_encoders_refuse_the_uint_carrier_by_name() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let u = Raster::zeroed(2, 2, PixelFormat::Uint32(n(1))).unwrap();
        let err = u
            .encode_png(6)
            .expect_err("a uint raster has no PNG colour type");
        let msg = err.to_string();
        assert!(
            msg.contains("32-bit unsigned") && msg.contains("Uint32"),
            "the refusal does not name the carrier: {msg}"
        );
        // Control: the float carrier of the same width gets its own
        // message, so a caller can tell which one they handed over.
        let f = Raster::zeroed(2, 2, PixelFormat::FloatF32(n(1))).unwrap();
        let fmsg = f
            .encode_png(6)
            .expect_err("a float raster has no PNG colour type")
            .to_string();
        assert!(fmsg.contains("float"), "{fmsg}");
        assert_ne!(msg, fmsg);
    }
}
