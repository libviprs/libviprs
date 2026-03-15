use std::path::Path;

use image::GenericImageView;
use thiserror::Error;

use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// Errors that can occur when decoding an image source.
///
/// Wraps the underlying I/O, image-decoding, and raster-construction
/// errors into a single enum so that callers of [`decode_file`],
/// [`decode_bytes`], and [`generate_test_raster`] can handle all failure
/// modes uniformly.
#[derive(Debug, Error)]
pub enum SourceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image decode error: {0}")]
    Decode(#[from] image::ImageError),
    #[error("unsupported color type: {0:?}")]
    UnsupportedColorType(image::ColorType),
    #[error("raster construction error: {0}")]
    Raster(#[from] crate::raster::RasterError),
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
pub fn decode_file(path: &Path) -> Result<Raster, SourceError> {
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    let color = img.color();
    let format = color_type_to_format(color)?;

    // For La8/La16, we need to convert to Rgba to get the right byte layout
    let data = match color {
        image::ColorType::La8 => img.to_rgba8().into_raw(),
        image::ColorType::La16 => {
            let rgba16 = img.to_rgba16();
            let pixels = rgba16.as_raw();
            // Convert &[u16] → Vec<u8> in native endian
            let mut bytes = Vec::with_capacity(pixels.len() * 2);
            for &sample in pixels {
                bytes.extend_from_slice(&sample.to_ne_bytes());
            }
            bytes
        }
        _ => img.into_bytes(),
    };

    Ok(Raster::new(width, height, format, data)?)
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
pub fn decode_bytes(bytes: &[u8]) -> Result<Raster, SourceError> {
    let img = image::load_from_memory(bytes)?;
    let (width, height) = img.dimensions();
    let color = img.color();
    let format = color_type_to_format(color)?;

    let data = match color {
        image::ColorType::La8 => img.to_rgba8().into_raw(),
        image::ColorType::La16 => {
            let rgba16 = img.to_rgba16();
            let pixels = rgba16.as_raw();
            let mut bytes = Vec::with_capacity(pixels.len() * 2);
            for &sample in pixels {
                bytes.extend_from_slice(&sample.to_ne_bytes());
            }
            bytes
        }
        _ => img.into_bytes(),
    };

    Ok(Raster::new(width, height, format, data)?)
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
pub fn generate_test_raster(width: u32, height: u32) -> Result<Raster, SourceError> {
    let bpp = PixelFormat::Rgb8.bytes_per_pixel();
    let mut data = vec![0u8; width as usize * height as usize * bpp];
    for y in 0..height {
        for x in 0..width {
            let offset = (y as usize * width as usize + x as usize) * bpp;
            data[offset] = (x * 255 / width.max(1)) as u8;
            data[offset + 1] = (y * 255 / height.max(1)) as u8;
            data[offset + 2] = ((x + y) * 255 / (width + height).max(1)) as u8;
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
     */
    #[test]
    fn decode_file_from_disk() {
        // Write a temp PNG and decode it
        let png = create_test_png(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        std::fs::write(&path, &png).unwrap();

        let raster = decode_file(&path).unwrap();
        assert_eq!(raster.width(), 8);
        assert_eq!(raster.height(), 8);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
    }

    /**
     * Tests that decode_file returns an error for a nonexistent path.
     * Works by passing a path that does not exist and asserting Err,
     * confirming proper I/O error propagation.
     * Input: Path("/nonexistent/image.png") → Output: Err.
     */
    #[test]
    fn decode_file_not_found() {
        let result = decode_file(Path::new("/nonexistent/image.png"));
        assert!(result.is_err());
    }
}
