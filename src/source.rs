use std::io::Cursor;
use std::path::Path;

use image::{GenericImageView, ImageReader, Limits};
use thiserror::Error;

use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// Errors that can occur when decoding an image source.
///
/// Wraps the underlying I/O, image-decoding, and raster-construction
/// errors into a single enum so that callers of [`decode_file`],
/// [`decode_bytes`], and [`generate_test_raster`] can handle all failure
/// modes uniformly.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SourceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image decode error: {0}")]
    Decode(#[from] image::ImageError),
    #[error("unsupported color type: {0:?}")]
    UnsupportedColorType(image::ColorType),
    #[error("raster construction error: {0}")]
    Raster(#[from] crate::raster::RasterError),
    #[error(
        "image dimensions {width}x{height} exceed the configured pixel ceiling ({max_pixels} px)"
    )]
    DimensionLimitExceeded {
        width: u32,
        height: u32,
        max_pixels: u64,
    },
}

/// Resource limits applied to a single image decode.
///
/// These bound the work a decoder may perform before pixel data is
/// materialised into a [`Raster`], guarding the process against
/// decompression bombs and pathologically large inputs. The width,
/// height, and allocation ceilings are pushed down into the underlying
/// decoder via [`image::Limits`] (so an oversized image is rejected
/// *before* it is fully allocated), and the combined `width * height`
/// pixel count is checked explicitly just before raster construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    /// Maximum decoded width, in pixels.
    pub max_width: u32,
    /// Maximum decoded height, in pixels.
    pub max_height: u32,
    /// Maximum total pixel count (`width * height`).
    pub max_pixels: u64,
    /// Maximum number of bytes the decoder may allocate at one time.
    pub max_alloc_bytes: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            // Matches the widest dimension the `image` strict limits can
            // express and covers every format libviprs targets.
            max_width: 65_535,
            max_height: 65_535,
            // ~1 gigapixel; large enough for legitimate scans, small
            // enough to reject a decompression bomb before allocation.
            max_pixels: 1u64 << 30,
            // Mirrors the `image` crate default allocation budget.
            max_alloc_bytes: 512 * 1024 * 1024,
        }
    }
}

impl DecodeLimits {
    /// Translate into the `image` crate's strict/non-strict limit set.
    fn to_image_limits(self) -> Limits {
        let mut limits = Limits::no_limits();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_alloc_bytes);
        limits
    }

    /// Enforce the `width * height` ceiling before a [`Raster`] is built.
    fn check_pixels(self, width: u32, height: u32) -> Result<(), SourceError> {
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if pixels > self.max_pixels {
            return Err(SourceError::DimensionLimitExceeded {
                width,
                height,
                max_pixels: self.max_pixels,
            });
        }
        Ok(())
    }
}

/// Map `image` crate color types to our canonical pixel format.
fn color_type_to_format(ct: image::ColorType) -> Result<PixelFormat, SourceError> {
    match ct {
        image::ColorType::L8 => Ok(PixelFormat::Gray8),
        image::ColorType::L16 => Ok(PixelFormat::Gray16),
        image::ColorType::Rgb8 => Ok(PixelFormat::Rgb8),
        image::ColorType::Rgba8 => Ok(PixelFormat::Rgba8),
        image::ColorType::Rgb16 => Ok(PixelFormat::Rgb16),
        image::ColorType::Rgba16 => Ok(PixelFormat::Rgba16),
        // La8/La16 (gray + alpha) → promote to Rgba
        image::ColorType::La8 => Ok(PixelFormat::Rgba8),
        image::ColorType::La16 => Ok(PixelFormat::Rgba16),
        other => Err(SourceError::UnsupportedColorType(other)),
    }
}

/// Decode an image file into a canonical [`Raster`].
///
/// Reads the file at `path`, auto-detects the format (JPEG, PNG, TIFF),
/// and decodes it into an in-memory [`Raster`] with a canonical
/// [`PixelFormat`]. Palette and gray+alpha images are promoted to
/// RGB/RGBA so that downstream code only needs to handle a small set of
/// uniform formats.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   calls `decode_file` in the `info` command to display image metadata
///   and in the `pyramid` command to load the input raster.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid) (general
/// entry point) and [`viprs info`](https://libviprs.org/cli/#info).
pub fn decode_file(path: &Path) -> Result<Raster, SourceError> {
    decode_file_with_limits(path, DecodeLimits::default())
}

/// Decode an image file into a [`Raster`] under explicit [`DecodeLimits`].
///
/// Identical to [`decode_file`] but lets the caller supply the
/// dimension/allocation budget instead of using [`DecodeLimits::default`].
/// The limits are configured on the decoder before any pixel data is
/// allocated, and the `width * height` ceiling is checked before the
/// [`Raster`] is constructed.
pub fn decode_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<Raster, SourceError> {
    decode_reader(ImageReader::open(path)?, limits)
}

/// Decode from an in-memory buffer (format auto-detected).
///
/// Behaves identically to [`decode_file`] but operates on a byte slice
/// that is already in memory. The image format is inferred from magic
/// bytes at the start of the buffer. This is the primary entry point
/// when the input arrives over a pipe or network socket rather than from
/// a filesystem path.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   calls `decode_bytes` when the user passes `"-"` as the input file,
///   reading the image data from stdin.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid) (general
/// entry point) and [`viprs info`](https://libviprs.org/cli/#info).
pub fn decode_bytes(bytes: &[u8]) -> Result<Raster, SourceError> {
    decode_bytes_with_limits(bytes, DecodeLimits::default())
}

/// Decode an in-memory buffer into a [`Raster`] under explicit [`DecodeLimits`].
///
/// Identical to [`decode_bytes`] but lets the caller supply the
/// dimension/allocation budget. The limits are configured on the decoder
/// before any pixel data is allocated, and the `width * height` ceiling
/// is checked before the [`Raster`] is constructed.
pub fn decode_bytes_with_limits(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    decode_reader(
        ImageReader::new(Cursor::new(bytes)).with_guessed_format()?,
        limits,
    )
}

/// Apply the shared decode budget to a configured [`ImageReader`] and
/// finalize its output into a [`Raster`].
///
/// This is the single tail shared by [`decode_file_with_limits`] and
/// [`decode_bytes_with_limits`]: the only thing that differs between the
/// file and in-memory entry points is how the reader is constructed. Both
/// funnel through here so the limit push-down, decode, and
/// [`build_raster`] finalization (color-type mapping + La repacking) live
/// in exactly one place and cannot drift out of parity.
fn decode_reader<R: std::io::BufRead + std::io::Seek>(
    mut reader: ImageReader<R>,
    limits: DecodeLimits,
) -> Result<Raster, SourceError> {
    reader.limits(limits.to_image_limits());
    let img = reader.decode()?;
    build_raster(img, limits)
}

/// Materialise a decoded [`image::DynamicImage`] into a [`Raster`].
///
/// Enforces the pixel ceiling, maps the color type to a canonical
/// [`PixelFormat`], and packs the sample bytes. For gray+alpha inputs
/// the luminance is expanded to RGB in a single streaming pass so no
/// second full-image copy is buffered.
fn build_raster(img: image::DynamicImage, limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (width, height) = img.dimensions();
    // Enforce the explicit ceiling before allocating the packed buffer.
    limits.check_pixels(width, height)?;
    let color = img.color();
    let format = color_type_to_format(color)?;
    let data = pack_bytes(img, color);
    Ok(Raster::new(width, height, format, data)?)
}

/// Pack a decoded image into the canonical native-endian byte layout.
fn pack_bytes(img: image::DynamicImage, color: image::ColorType) -> Vec<u8> {
    match color {
        // 8-bit gray+alpha: a single expanding copy to RGBA8, then a
        // zero-copy unwrap of the backing buffer.
        image::ColorType::La8 => img.to_rgba8().into_raw(),
        // 16-bit gray+alpha: stream luminance → RGB directly into the
        // output byte buffer, borrowing the already-decoded LumaA16
        // samples so no intermediate RGBA16 image is materialised.
        image::ColorType::La16 => {
            let la = img
                .as_luma_alpha16()
                .expect("color type verified as La16 above");
            la16_to_rgba16_bytes(la.as_raw())
        }
        _ => img.into_bytes(),
    }
}

/// Expand interleaved `[luma, alpha]` u16 samples into RGBA16 bytes.
///
/// Writes exactly one output buffer: for each source pixel the single
/// luminance sample is emitted on all three color channels followed by
/// the alpha sample, each in native-endian byte order. This avoids the
/// second whole-image allocation that a `to_rgba16` conversion would
/// require before the byte re-pack.
fn la16_to_rgba16_bytes(samples: &[u16]) -> Vec<u8> {
    // 2 input samples per pixel → 4 output channels × 2 bytes.
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for pair in samples.chunks_exact(2) {
        let luma = pair[0];
        let alpha = pair[1];
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&luma.to_ne_bytes());
        bytes.extend_from_slice(&alpha.to_ne_bytes());
    }
    bytes
}

/// Generate a synthetic test image (RGB8 gradient pattern).
///
/// Creates a `width x height` [`Raster`] in [`PixelFormat::Rgb8`] filled
/// with a deterministic gradient: the red channel increases left-to-right,
/// the green channel increases top-to-bottom, and the blue channel is
/// a diagonal blend. This is useful for verifying the full pipeline
/// without needing an external test fixture on disk.
///
/// # Example usage
///
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   exposes this as the `test-image` subcommand, generating a gradient
///   PNG for quick smoke-testing.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#test-image)
pub fn generate_test_raster(width: u32, height: u32) -> Result<Raster, SourceError> {
    // Bound the requested dimensions against the shared decode budget before
    // allocating anything. Without this an oversized request would attempt an
    // unbounded `width * height * bpp` allocation (a debug abort / process
    // OOM) instead of failing cleanly; it also keeps `width + height` and the
    // per-pixel gradient math below within range. Oversized dimensions now
    // yield a typed `DimensionLimitExceeded`.
    let limits = DecodeLimits::default();
    limits.check_pixels(width, height)?;

    let bpp = PixelFormat::Rgb8.bytes_per_pixel();
    let mut data = vec![0u8; width as usize * height as usize * bpp];
    // Gradient math is widened to `u64`: for accepted dimensions a single
    // axis can still be as large as `max_pixels`, so `x * 255` (peaking near
    // 2^30 * 255) overflows a `u32`. The `.max(1)` denominators keep the
    // 1-pixel-wide/tall degenerate cases division-safe.
    let w_denom = u64::from(width).max(1);
    let h_denom = u64::from(height).max(1);
    let wh_denom = (u64::from(width) + u64::from(height)).max(1);
    for y in 0..height {
        for x in 0..width {
            let offset = (y as usize * width as usize + x as usize) * bpp;
            data[offset] = (u64::from(x) * 255 / w_denom) as u8;
            data[offset + 1] = (u64::from(y) * 255 / h_denom) as u8;
            data[offset + 2] = ((u64::from(x) + u64::from(y)) * 255 / wh_denom) as u8;
        }
    }
    Ok(Raster::new(width, height, PixelFormat::Rgb8, data)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(Cursor::new(&mut buf));
            let data = vec![128u8; w as usize * h as usize * 3];
            image::ImageEncoder::write_image(encoder, &data, w, h, image::ColorType::Rgb8.into())
                .unwrap();
        }
        buf
    }

    /// Encode a `w x h` La16 (gray + alpha) PNG in memory, returning the
    /// encoded bytes alongside the `(luma, alpha)` samples that were
    /// written so callers can verify the decoded RGBA16 layout.
    fn create_la16_png(w: u32, h: u32) -> (Vec<u8>, Vec<(u16, u16)>) {
        use image::{DynamicImage, ImageBuffer, ImageFormat, LumaA};

        let mut buf: ImageBuffer<LumaA<u16>, Vec<u16>> = ImageBuffer::new(w, h);
        let mut expected = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let luma = ((x.wrapping_mul(4096).wrapping_add(y.wrapping_mul(7))) & 0xFFFF) as u16;
                let alpha = ((x.wrapping_add(y.wrapping_mul(300))) & 0xFFFF) as u16;
                buf.put_pixel(x, y, LumaA([luma, alpha]));
                expected.push((luma, alpha));
            }
        }
        let dyn_img = DynamicImage::ImageLumaA16(buf);
        let mut out = Vec::new();
        dyn_img
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        (out, expected)
    }

    fn create_test_jpeg(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(Cursor::new(&mut buf), 95);
            let data = vec![128u8; w as usize * h as usize * 3];
            image::ImageEncoder::write_image(encoder, &data, w, h, image::ColorType::Rgb8.into())
                .unwrap();
        }
        buf
    }

    /**
     * Tests that a valid PNG byte buffer can be decoded into a Raster.
     * Works by encoding a known 32x24 RGB image to PNG in-memory, then
     * decoding it back and verifying dimensions, format, and buffer size.
     * Input: 32x24 RGB8 PNG bytes → Output: Raster(32, 24, Rgb8, 2304 bytes).
     */
    #[test]
    fn decode_png_from_memory() {
        let png = create_test_png(32, 24);
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 32);
        assert_eq!(raster.height(), 24);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.data().len(), 32 * 24 * 3);
    }

    /**
     * Tests that a valid JPEG byte buffer can be decoded into a Raster.
     * Works by encoding a 16x16 RGB image to JPEG at quality 95, then
     * decoding it and checking dimensions and format are preserved.
     * Input: 16x16 RGB8 JPEG bytes → Output: Raster(16, 16, Rgb8).
     */
    #[test]
    fn decode_jpeg_from_memory() {
        let jpeg = create_test_jpeg(16, 16);
        let raster = decode_bytes(&jpeg).unwrap();
        assert_eq!(raster.width(), 16);
        assert_eq!(raster.height(), 16);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
    }

    /**
     * Tests that decode_bytes returns an error for invalid image data.
     * Works by passing an arbitrary non-image byte string and asserting Err,
     * confirming the decoder rejects garbage input.
     * Input: b"not an image" → Output: Err.
     */
    #[test]
    fn decode_invalid_bytes_returns_error() {
        let result = decode_bytes(b"not an image");
        assert!(result.is_err());
    }

    /**
     * Tests that decode_bytes returns an error for an empty buffer.
     * Works by passing a zero-length slice, ensuring the decoder does not
     * panic and instead produces a meaningful error.
     * Input: b"" → Output: Err.
     */
    #[test]
    fn decode_empty_bytes_returns_error() {
        let result = decode_bytes(b"");
        assert!(result.is_err());
    }

    /**
     * Tests that generate_test_raster produces a Raster with correct
     * dimensions, pixel format, and buffer size.
     * Works by generating a 100x50 test raster and checking all properties.
     * Input: (100, 50) → Output: Raster(100, 50, Rgb8, 15000 bytes).
     */
    #[test]
    fn generate_test_raster_dimensions() {
        let r = generate_test_raster(100, 50).unwrap();
        assert_eq!(r.width(), 100);
        assert_eq!(r.height(), 50);
        assert_eq!(r.format(), PixelFormat::Rgb8);
        assert_eq!(r.data().len(), 100 * 50 * 3);
    }

    /**
     * Reproducer for the unchecked gradient arithmetic: a single axis large
     * enough to keep the total pixel count under the ceiling still drives the
     * old `x * 255` computation past `u32::MAX` (16_777_215 * 255 already
     * exceeds it). Before widening the math to `u64` this panicked on
     * overflow in debug builds (and silently wrapped in release); it must now
     * complete and paint the far-right column at full red intensity.
     * Input: (20_000_000, 1) → Output: Ok(Raster) with a saturated last pixel.
     */
    #[test]
    fn generate_test_raster_wide_no_overflow() {
        let width = 20_000_000u32;
        let r = generate_test_raster(width, 1).unwrap();
        assert_eq!(r.width(), width);
        assert_eq!(r.height(), 1);
        // Red channel of the last pixel: (width-1) * 255 / width ≈ 254.
        let last = ((width - 1) as usize) * 3;
        assert_eq!(r.data()[last], 254);
    }

    /**
     * Reproducer for the missing allocation cap: dimensions whose pixel count
     * exceeds the shared decode budget must be rejected with a typed
     * `DimensionLimitExceeded` before any buffer is allocated. Before the fix
     * `generate_test_raster` sized its `vec!` straight from the raw
     * dimensions, so this request attempted a multi-gigabyte allocation (a
     * process abort) instead of returning an error.
     * Input: (65_535, 65_535) → Output: Err(DimensionLimitExceeded).
     */
    #[test]
    fn generate_test_raster_rejects_oversized() {
        let (width, height) = (65_535u32, 65_535u32);
        match generate_test_raster(width, height) {
            Err(SourceError::DimensionLimitExceeded {
                width: w,
                height: h,
                max_pixels,
            }) => {
                assert_eq!(w, width);
                assert_eq!(h, height);
                assert_eq!(max_pixels, DecodeLimits::default().max_pixels);
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }
    }

    /**
     * Tests that color_type_to_format correctly maps image crate ColorType
     * variants to PixelFormat, including the La8→Rgba8 promotion.
     * Works by checking each supported mapping individually.
     * Input: e.g. ColorType::L8 → Output: PixelFormat::Gray8.
     */
    #[test]
    fn color_type_mapping() {
        assert_eq!(
            color_type_to_format(image::ColorType::L8).unwrap(),
            PixelFormat::Gray8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgb8).unwrap(),
            PixelFormat::Rgb8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgba8).unwrap(),
            PixelFormat::Rgba8
        );
        assert_eq!(
            color_type_to_format(image::ColorType::Rgb16).unwrap(),
            PixelFormat::Rgb16
        );
        assert_eq!(
            color_type_to_format(image::ColorType::La8).unwrap(),
            PixelFormat::Rgba8
        );
    }

    /**
     * Tests that decode_file can read and decode a PNG from disk.
     * Works by writing a known PNG to a temp file, then decoding it
     * with decode_file and verifying the resulting Raster properties.
     * Input: 8x8 RGB8 PNG on disk → Output: Raster(8, 8, Rgb8).
     *
     * Split for Miri: tempdir/write are blocked under Miri's isolation
     * mode. The first half decodes the PNG bytes in memory via
     * decode_bytes and checks the resulting Raster dimensions and format
     * (runs everywhere). The #[cfg(not(miri))] block writes the PNG to
     * a temp file and decodes it back via decode_file to test the
     * filesystem round-trip (skipped under Miri).
     */
    #[test]
    fn decode_file_from_disk() {
        let png = create_test_png(8, 8);

        // Miri-safe: verify decoding from bytes in memory
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 8);
        assert_eq!(raster.height(), 8);
        assert_eq!(raster.format(), PixelFormat::Rgb8);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.png");
            std::fs::write(&path, &png).unwrap();

            let from_disk = decode_file(&path).unwrap();
            assert_eq!(from_disk.width(), 8);
            assert_eq!(from_disk.height(), 8);
            assert_eq!(from_disk.format(), PixelFormat::Rgb8);
        }
    }

    /**
     * Tests that decode_file returns an error for a nonexistent path.
     * Works by passing a path that does not exist and asserting Err,
     * confirming proper I/O error propagation.
     * Input: Path("/nonexistent/image.png") → Output: Err.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn decode_file_not_found() {
        let result = decode_file(Path::new("/nonexistent/image.png"));
        assert!(result.is_err());
    }

    /**
     * Reproducer for the missing dimension ceiling: a fully decodable
     * image must still be rejected when its pixel count exceeds the
     * configured `max_pixels`. Works by decoding a valid 64x64 PNG under
     * a `DecodeLimits` whose ceiling (100 px) is far below 64*64=4096 and
     * asserting a `DimensionLimitExceeded` error is returned before the
     * raster is built. Before the fix no ceiling existed and this decode
     * succeeded. Input: 64x64 PNG + max_pixels=100 → Output: Err.
     */
    #[test]
    fn decode_bytes_rejects_over_pixel_ceiling() {
        let png = create_test_png(64, 64);
        let limits = DecodeLimits {
            max_pixels: 100,
            ..DecodeLimits::default()
        };
        let result = decode_bytes_with_limits(&png, limits);
        match result {
            Err(SourceError::DimensionLimitExceeded {
                width,
                height,
                max_pixels,
            }) => {
                assert_eq!(width, 64);
                assert_eq!(height, 64);
                assert_eq!(max_pixels, 100);
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }
        // The same bytes decode fine under the default (generous) ceiling.
        assert!(decode_bytes(&png).is_ok());
    }

    /**
     * Confirms the explicit width/height limits are pushed down into the
     * decoder itself (not merely checked after the fact): decoding a
     * 64-wide PNG under `max_width = 10` must fail with an `image`
     * limit/decode error. Input: 64x48 PNG + max_width=10 → Output: Err.
     */
    #[test]
    fn decode_bytes_enforces_decoder_width_limit() {
        let png = create_test_png(64, 48);
        let limits = DecodeLimits {
            max_width: 10,
            ..DecodeLimits::default()
        };
        let result = decode_bytes_with_limits(&png, limits);
        assert!(
            matches!(result, Err(SourceError::Decode(_))),
            "expected a decoder limit error, got {result:?}"
        );
    }

    /**
     * Guards the shared finalize path: `decode_file` and `decode_bytes`
     * must produce byte-identical rasters for the same input, including the
     * La16 -> RGBA16 promotion. Both entry points funnel through the single
     * `decode_reader` -> `build_raster` tail (the issue's proposed
     * `finalize()`), so the color-type mapping and La repacking cannot drift
     * back into copy-pasted divergence. Were the two paths ever re-forked,
     * an La16 (a format with non-trivial repacking) input would expose the
     * mismatch here. Input: 6x4 La16 PNG decoded via both entry points ->
     * Output: equal dimensions, format, and pixel bytes.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn decode_file_and_bytes_share_finalize_path() {
        let (png, _expected) = create_la16_png(6, 4);

        let from_bytes = decode_bytes(&png).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parity.png");
        std::fs::write(&path, &png).unwrap();
        let from_file = decode_file(&path).unwrap();

        assert_eq!(from_file.width(), from_bytes.width());
        assert_eq!(from_file.height(), from_bytes.height());
        assert_eq!(from_file.format(), from_bytes.format());
        assert_eq!(from_file.format(), PixelFormat::Rgba16);
        assert_eq!(from_file.data(), from_bytes.data());
    }

    /**
     * Verifies the streaming La16 conversion produces the exact RGBA16
     * native-endian byte layout: luminance replicated across R, G, B and
     * the alpha sample last, two bytes each. Works by encoding a known
     * La16 PNG, decoding it, and comparing every 8-byte pixel against the
     * expected expansion of the original (luma, alpha) samples.
     * Input: 5x3 La16 PNG → Output: Raster(5, 3, Rgba16) with each pixel
     * bytes == [luma, luma, luma, alpha] in native endian.
     */
    #[test]
    fn decode_la16_streams_to_rgba16_layout() {
        let (png, expected) = create_la16_png(5, 3);
        let raster = decode_bytes(&png).unwrap();
        assert_eq!(raster.width(), 5);
        assert_eq!(raster.height(), 3);
        assert_eq!(raster.format(), PixelFormat::Rgba16);

        let data = raster.data();
        assert_eq!(data.len(), expected.len() * 4 * 2);
        for (i, &(luma, alpha)) in expected.iter().enumerate() {
            let base = i * 8;
            let mut want = Vec::with_capacity(8);
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&luma.to_ne_bytes());
            want.extend_from_slice(&alpha.to_ne_bytes());
            assert_eq!(&data[base..base + 8], want.as_slice(), "pixel {i} mismatch");
        }
    }
}
