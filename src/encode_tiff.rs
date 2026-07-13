//! TIFF encode/decode lane.
//!
//! This module carries the TIFF surface the ported foreign and connection
//! cells reference: the `save_tiff` family of file writers on
//! [`Raster`], the in-memory `tiff_save` / `tiff_load` round-trip, and the
//! free [`tiff_page_count`] / [`decode_tiff_page`] readers for multi-page
//! files. It builds directly on the pure-Rust [`tiff`] crate so the encode
//! side can pick a compression per image and the decode side can walk the
//! IFD chain, neither of which the higher-level `image` facade exposes.
//!
//! ## Real versus deferred
//!
//! Everything the `tiff` crate can emit and read with a pure-Rust codec is
//! real:
//!
//! * [`Raster::save_tiff`] with [`TiffCompression::None`],
//!   [`TiffCompression::Lzw`], and [`TiffCompression::Deflate`].
//! * [`Raster::tiff_save`] / [`Raster::tiff_load`] (the streamed connection
//!   round-trip), which use Deflate so the trip stays lossless.
//! * [`tiff_page_count`] and [`decode_tiff_page`] over multi-page files.
//!
//! The remaining variants have no pure-Rust encoder in this build and return
//! a typed [`SaveError`] rather than panicking, so the call sites compile and
//! pin the error path:
//!
//! * [`Raster::save_tiff`] with [`TiffCompression::Jpeg`],
//!   [`TiffCompression::Ccitt`], or [`TiffCompression::Jp2k`] (JPEG-in-TIFF,
//!   CCITT G4 fax, and JPEG 2000 all need an external codec).
//! * [`Raster::save_bigtiff`] (64-bit-offset container, deferred).
//! * [`Raster::save_tiff_tiled`] (tiled layout plus pyramid/subifd, deferred).
//!
//! ## Pixel formats
//!
//! The encoder handles the 8- and 16-bit gray, RGB, and RGBA formats. The
//! float and multiband [`PixelFormat`] variants are compute intermediates
//! with no TIFF representation here, so they surface the same typed error.
//! Sixteen-bit samples are read from and written back to the raster buffer in
//! native byte order, matching the convention the rest of the crate uses.

use std::io::{Cursor, Read, Seek};
use std::path::Path;

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{Compression, DeflateLevel, Predictor, TiffEncoder, colortype};

use crate::codec::{DecodeError, TiffCompression};
use crate::imageio::SaveError;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;

// ---------------------------------------------------------------------------
// Error helpers
//
// `save_tiff` and friends return the crate-wide `SaveError`, which the shared
// spine owns in `imageio.rs`. That type has no dedicated "unsupported
// compression" variant and this lane must not widen it, so the deferred paths
// reuse `SaveError::Encode(SinkError::Other(_))` with an accurate message
// (the alternative, `SaveError::UnsupportedExtension`, carries a fixed tail
// that would misreport what the pure-Rust build can save). The decode helpers
// return `DecodeError` (an alias of `SourceError`) and wrap `tiff` crate
// failures as an image decoding error tagged with the TIFF format hint.
// ---------------------------------------------------------------------------

/// A deferred TIFF compression mode reported as a typed, matchable error.
fn unsupported_compression(mode: &str) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "TIFF {mode} compression is not supported by the pure-Rust encoder"
    )))
}

/// A pixel format with no TIFF representation in this build.
fn unsupported_pixels(format: PixelFormat) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "TIFF encoding of {format:?} pixels is not supported \
         (only 8- and 16-bit gray, RGB, and RGBA)"
    )))
}

/// A deferred TIFF container feature (BigTIFF, tiled layout) reported typed.
fn unsupported_feature(feature: &str) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "{feature} is not yet implemented in the pure-Rust TIFF lane"
    )))
}

/// Map a `tiff` crate encode failure onto the crate-wide save error.
fn tiff_encode_err(err: tiff::TiffError) -> SaveError {
    SaveError::Encode(SinkError::Other(format!("TIFF encode failed: {err}")))
}

/// Build a TIFF-tagged decode error from a free-form message.
fn decode_err(message: impl Into<String>) -> DecodeError {
    use image::error::{DecodingError, ImageFormatHint};
    DecodeError::Decode(image::ImageError::Decoding(DecodingError::new(
        ImageFormatHint::Exact(image::ImageFormat::Tiff),
        message.into(),
    )))
}

/// Map a `tiff` crate decode failure onto the crate-wide decode error.
fn tiff_decode_err(err: tiff::TiffError) -> DecodeError {
    decode_err(err.to_string())
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Reinterpret a native-byte-order 16-bit sample buffer as `u16` values.
fn as_u16_samples(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

/// Encode this raster to in-memory TIFF bytes with the given compression.
///
/// LZW and Deflate apply the horizontal predictor, which is lossless and is
/// what makes those modes actually shrink natural images (and keeps this in
/// line with what libvips writes). `None` writes raw strips.
fn encode_to_vec(raster: &Raster, compression: TiffCompression) -> Result<Vec<u8>, SaveError> {
    let (comp, predictor) = match compression {
        TiffCompression::None => (Compression::Uncompressed, Predictor::None),
        TiffCompression::Lzw => (Compression::Lzw, Predictor::Horizontal),
        TiffCompression::Deflate => (
            Compression::Deflate(DeflateLevel::default()),
            Predictor::Horizontal,
        ),
        TiffCompression::Jpeg => return Err(unsupported_compression("JPEG")),
        TiffCompression::Ccitt => return Err(unsupported_compression("CCITT G4")),
        TiffCompression::Jp2k => return Err(unsupported_compression("JPEG 2000")),
    };

    let width = raster.width();
    let height = raster.height();
    let mut buf = Vec::new();
    {
        let mut encoder = TiffEncoder::new(Cursor::new(&mut buf))
            .map_err(tiff_encode_err)?
            .with_compression(comp)
            .with_predictor(predictor);

        let write_result = match raster.format() {
            PixelFormat::Gray8 => {
                encoder.write_image::<colortype::Gray8>(width, height, raster.data())
            }
            PixelFormat::Rgb8 => {
                encoder.write_image::<colortype::RGB8>(width, height, raster.data())
            }
            PixelFormat::Rgba8 => {
                encoder.write_image::<colortype::RGBA8>(width, height, raster.data())
            }
            PixelFormat::Gray16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::Gray16>(width, height, &samples)
            }
            PixelFormat::Rgb16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::RGB16>(width, height, &samples)
            }
            PixelFormat::Rgba16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::RGBA16>(width, height, &samples)
            }
            other => return Err(unsupported_pixels(other)),
        };
        write_result.map_err(tiff_encode_err)?;
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Turn the `tiff` decoder's color type and sample buffer into a canonical
/// [`PixelFormat`] and native-byte-order pixel buffer.
fn interpret(
    color: ColorType,
    result: DecodingResult,
) -> Result<(PixelFormat, Vec<u8>), DecodeError> {
    let channels = match color {
        ColorType::Gray(_) => 1usize,
        ColorType::RGB(_) => 3,
        ColorType::RGBA(_) => 4,
        other => {
            return Err(decode_err(format!(
                "unsupported TIFF color type {other:?} \
                 (only gray, RGB, and RGBA are decoded here)"
            )));
        }
    };
    match result {
        DecodingResult::U8(values) => {
            let format = PixelFormat::with_channels(channels, 1)
                .ok_or_else(|| decode_err("unsupported TIFF channel count"))?;
            Ok((format, values))
        }
        DecodingResult::U16(values) => {
            let format = PixelFormat::with_channels(channels, 2)
                .ok_or_else(|| decode_err("unsupported TIFF channel count"))?;
            let bytes = values.into_iter().flat_map(u16::to_ne_bytes).collect();
            Ok((format, bytes))
        }
        _ => Err(decode_err(
            "unsupported TIFF sample format \
             (only 8- and 16-bit unsigned samples are decoded here)",
        )),
    }
}

/// Read the decoder's currently-selected image into a [`Raster`].
fn decode_current_image<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Raster, DecodeError> {
    let (width, height) = decoder.dimensions().map_err(tiff_decode_err)?;
    let color = decoder.colortype().map_err(tiff_decode_err)?;
    let result = decoder.read_image().map_err(tiff_decode_err)?;
    let (format, data) = interpret(color, result)?;
    Ok(Raster::new(width, height, format, data)?)
}

/// Count the pages (IFDs) in a multi-page TIFF file.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened or the TIFF header or
/// its IFD chain is malformed.
pub fn tiff_page_count(path: &Path) -> Result<u32, DecodeError> {
    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file)).map_err(tiff_decode_err)?;
    let mut count = 1u32;
    while decoder.more_images() {
        decoder.next_image().map_err(tiff_decode_err)?;
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// Decode a single 1-indexed page from a multi-page TIFF file.
///
/// Page 1 is the first image. Passing 0, or a page past the end of the file,
/// is an error.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened, the page index is
/// out of range, the TIFF is malformed, or the page uses a color or sample
/// type this lane does not decode.
pub fn decode_tiff_page(path: &Path, page: u32) -> Result<Raster, DecodeError> {
    if page == 0 {
        return Err(decode_err("TIFF pages are 1-indexed; page 0 is invalid"));
    }
    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file)).map_err(tiff_decode_err)?;
    decoder
        .seek_to_image((page - 1) as usize)
        .map_err(tiff_decode_err)?;
    decode_current_image(&mut decoder)
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Save as a single-page TIFF, choosing the strip compression.
    ///
    /// [`TiffCompression::None`], [`TiffCompression::Lzw`], and
    /// [`TiffCompression::Deflate`] are written by the pure-Rust encoder (LZW
    /// and Deflate are lossless and use the horizontal predictor). The
    /// external-codec modes ([`TiffCompression::Jpeg`],
    /// [`TiffCompression::Ccitt`], [`TiffCompression::Jp2k`]) are not
    /// available in this build.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Encode`] if the compression mode or the raster's
    /// pixel format has no pure-Rust TIFF encoder, if the `tiff` encoder
    /// rejects the data, or [`SaveError::Io`] if writing the file fails.
    pub fn save_tiff(&self, path: &Path, compression: TiffCompression) -> Result<(), SaveError> {
        let bytes = encode_to_vec(self, compression)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Save as a BigTIFF (64-bit offsets, for files larger than 4 GB).
    ///
    /// # Errors
    ///
    /// BigTIFF encoding is deferred, so this always returns a typed
    /// [`SaveError::Encode`] naming the unimplemented feature.
    pub fn save_bigtiff(&self, path: &Path, compression: TiffCompression) -> Result<(), SaveError> {
        let _ = (path, compression);
        Err(unsupported_feature("BigTIFF encoding"))
    }

    /// Save as a tiled TIFF, optionally with a pyramid and SubIFDs.
    ///
    /// # Errors
    ///
    /// Tiled TIFF encoding (and the pyramid/subifd layout it feeds) is
    /// deferred, so this always returns a typed [`SaveError::Encode`] naming
    /// the unimplemented feature.
    pub fn save_tiff_tiled(
        &self,
        path: &Path,
        compression: TiffCompression,
        tile_width: u32,
        tile_height: u32,
        pyramid: bool,
        subifd: bool,
    ) -> Result<(), SaveError> {
        let _ = (path, compression, tile_width, tile_height, pyramid, subifd);
        Err(unsupported_feature("Tiled TIFF encoding"))
    }

    /// Encode this raster as in-memory TIFF bytes (the streamed connection
    /// save). Uses Deflate so the round-trip through [`Raster::tiff_load`]
    /// stays lossless.
    ///
    /// The contract is infallible: the 8- and 16-bit gray/RGB/RGBA formats
    /// always encode, and encoding into memory does no fallible I/O. A float
    /// or multiband raster (a compute intermediate with no TIFF form) yields
    /// an empty buffer rather than a spurious error.
    pub fn tiff_save(&self) -> Vec<u8> {
        encode_to_vec(self, TiffCompression::Deflate).unwrap_or_default()
    }

    /// Decode a raster from in-memory TIFF bytes (the inverse of
    /// [`Raster::tiff_save`]); reads the first page.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are not a valid TIFF, or the
    /// first page uses a color or sample type this lane does not decode.
    pub fn tiff_load(data: &[u8]) -> Result<Raster, DecodeError> {
        let mut decoder = Decoder::new(Cursor::new(data)).map_err(tiff_decode_err)?;
        decode_current_image(&mut decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::decode_file;
    use tiff::encoder::{TiffEncoder, colortype};

    fn ramp_gray8(w: u32, h: u32) -> Raster {
        let data: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    fn ramp_gray16(w: u32, h: u32) -> Raster {
        let samples: Vec<u16> = (0..w * h)
            .map(|i| ((i % w) as u16).saturating_mul(200))
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Gray16, bytes).unwrap()
    }

    fn multipage_gray8_fixture(pages: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = TiffEncoder::new(Cursor::new(&mut buf)).unwrap();
            for p in 0..pages {
                let data = vec![p.saturating_mul(10); 16];
                encoder
                    .write_image::<colortype::Gray8>(4, 4, &data)
                    .unwrap();
            }
        }
        buf
    }

    #[test]
    fn save_tiff_none_round_trips_bit_exact() {
        let im = ramp_gray8(64, 64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("none.tif");
        im.save_tiff(&out, TiffCompression::None).unwrap();
        let back = decode_file(&out).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data());
    }

    #[test]
    fn save_tiff_deflate_round_trips_bit_exact() {
        let im = ramp_gray8(64, 64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("deflate.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();
        let back = decode_file(&out).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data(), "Deflate TIFF must be lossless");
    }

    #[test]
    fn save_tiff_lzw_smaller_than_none() {
        let im = ramp_gray8(128, 128);
        let dir = tempfile::tempdir().unwrap();
        let out_lzw = dir.path().join("lzw.tif");
        let out_none = dir.path().join("none.tif");
        im.save_tiff(&out_lzw, TiffCompression::Lzw).unwrap();
        im.save_tiff(&out_none, TiffCompression::None).unwrap();

        let lzw_size = std::fs::metadata(&out_lzw).unwrap().len();
        let none_size = std::fs::metadata(&out_none).unwrap().len();
        assert!(
            lzw_size < none_size,
            "LZW ({lzw_size}) should be smaller than none ({none_size})"
        );

        let back = decode_file(&out_lzw).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data(), "LZW TIFF must be lossless");
    }

    #[test]
    fn save_tiff_gray16_deflate_round_trips_via_decode_tiff_page() {
        let im = ramp_gray16(48, 32);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("gray16.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();

        let back = decode_tiff_page(&out, 1).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.format(), PixelFormat::Gray16);
        assert_eq!(
            back.data(),
            im.data(),
            "16-bit Deflate TIFF must be lossless"
        );
    }

    #[test]
    fn tiff_save_tiff_load_round_trips_bit_exact() {
        let im = ramp_gray8(40, 40);
        let bytes = im.tiff_save();
        assert!(!bytes.is_empty(), "tiff_save must emit TIFF bytes");
        let back = Raster::tiff_load(&bytes).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data());
    }

    #[test]
    fn tiff_page_count_and_decode_tiff_page_on_multipage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tif");
        std::fs::write(&path, multipage_gray8_fixture(3)).unwrap();

        assert_eq!(tiff_page_count(&path).unwrap(), 3);

        for p in 1..=3u32 {
            let page = decode_tiff_page(&path, p).unwrap();
            assert_eq!(page.width(), 4);
            assert_eq!(page.height(), 4);
        }
        // Page 2 was written with the constant value 10.
        assert_eq!(decode_tiff_page(&path, 2).unwrap().data(), vec![10u8; 16]);
    }

    #[test]
    fn decode_tiff_page_rejects_page_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tif");
        std::fs::write(&path, multipage_gray8_fixture(2)).unwrap();
        assert!(decode_tiff_page(&path, 0).is_err());
    }

    #[test]
    fn save_tiff_jpeg_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("jpeg.tif");
        let err = im.save_tiff(&out, TiffCompression::Jpeg).unwrap_err();
        assert!(matches!(err, SaveError::Encode(SinkError::Other(_))));
        assert!(err.to_string().to_lowercase().contains("not supported"));
    }

    #[test]
    fn save_tiff_ccitt_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("ccitt.tif");
        assert!(im.save_tiff(&out, TiffCompression::Ccitt).is_err());
    }

    #[test]
    fn save_tiff_jp2k_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("jp2k.tif");
        assert!(im.save_tiff(&out, TiffCompression::Jp2k).is_err());
    }

    #[test]
    fn save_bigtiff_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("big.tif");
        let err = im.save_bigtiff(&out, TiffCompression::None).unwrap_err();
        assert!(matches!(err, SaveError::Encode(SinkError::Other(_))));
    }

    #[test]
    fn save_tiff_tiled_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("tiled.tif");
        assert!(
            im.save_tiff_tiled(&out, TiffCompression::Jp2k, 128, 128, true, true)
                .is_err()
        );
    }
}
