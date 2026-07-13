//! Deferred foreign-format surface: typed stubs for genuinely-external
//! codecs, plus the in-memory DeepZoom zip writer.
//!
//! The ported foreign and connection cells reference a set of encoders and
//! decoders for formats that have no mature pure-Rust implementation yet
//! (HEIF/AVIF, JPEG 2000, JPEG XL, Ultra HDR, the ImageMagick delegate, SVG
//! rasterisation, OpenSlide, and the libvips `fail_on` strictness knob). This
//! module supplies those symbols so the cells compile and pin the typed error
//! path:
//!
//! * The deferred **encoders** on [`Raster`] return
//!   [`EncodeError::Unsupported`] naming the format, so a call site asserts on
//!   the typed error rather than a panic.
//! * The deferred **decoders** are free functions that return
//!   [`DecodeError`] (an alias of [`crate::source::SourceError`]) naming the
//!   capability that is not available in this build.
//! * [`Raster::dzsave_buffer`] is real: it packs a DeepZoom manifest and the
//!   image tile into a valid, self-contained zip in memory. The zip container
//!   is written by hand (STORE method, CRC-32 via `flate2`) because the
//!   `zip`-crate path lives behind the optional `packfile` feature.
//!
//! When a pure-Rust or delegate backend for one of these formats lands, the
//! stub body is replaced with the real codec while the signature stays put.

use std::path::Path;

use crate::codec::{DecodeError, EncodeError};
use crate::raster::Raster;

/// Options for the ImageMagick/GraphicsMagick delegate loader
/// ([`magickload_with`]), mirroring the libvips `magickload` load options the
/// ported foreign cell exercises.
///
/// Every field is optional and defaults to `None`, so callers set only the
/// options they need with struct-update syntax
/// (`MagickLoadOptions { n: Some(-1), ..Default::default() }`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MagickLoadOptions {
    /// Rendering density passed to the delegate (libvips `density`), for
    /// vector inputs such as SVG. A higher density rasterises at a larger
    /// pixel size. Expressed as the delegate's density string (for example
    /// `"100"` or `"200"`).
    pub density: Option<&'static str>,
    /// First page/frame to load (libvips `page`), zero-based.
    pub page: Option<i32>,
    /// Number of pages/frames to load (libvips `n`); `-1` loads all of them.
    pub n: Option<i32>,
}

/// Build a [`DecodeError`] that names a decode capability the pure-Rust build
/// does not provide yet. The message carries `what` so the ported cells and
/// the in-crate tests can assert on the format name.
fn decode_unavailable(what: impl std::fmt::Display) -> DecodeError {
    DecodeError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{what} is not available in this build"),
    ))
}

// ---------------------------------------------------------------------------
// Deferred encoders (typed Unsupported)
// ---------------------------------------------------------------------------

impl Raster {
    /// Encode as HEIF/AVIF with the given quality and compression codec
    /// (libvips `heifsave`, `compression` = `"av1"`, `"hevc"`, ...).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]: HEIF/AVIF encoding needs an
    /// external `libheif`/`libaom` path that the pure-Rust build does not
    /// provide.
    pub fn encode_heif(&self, quality: u8, compression: &str) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, compression);
        Err(EncodeError::unsupported("heif"))
    }

    /// Encode as lossless HEIF/AVIF (libvips `heifsave` with `lossless`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_heif`].
    pub fn encode_heif_lossless(&self, compression: &str) -> Result<Vec<u8>, EncodeError> {
        let _ = compression;
        Err(EncodeError::unsupported("heif"))
    }

    /// Encode as HEIF/AVIF with explicit chroma sub-sampling control.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_heif`].
    pub fn encode_heif_chroma(
        &self,
        quality: u8,
        compression: &str,
        subsample: bool,
    ) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, compression, subsample);
        Err(EncodeError::unsupported("heif"))
    }

    /// Encode as lossless HEIF/AVIF at a specific stored bit depth (libvips
    /// `heifsave` with `lossless` and `bitdepth`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_heif`].
    pub fn encode_heif_lossless_bitdepth(
        &self,
        compression: &str,
        bitdepth: u32,
    ) -> Result<Vec<u8>, EncodeError> {
        let _ = (compression, bitdepth);
        Err(EncodeError::unsupported("heif"))
    }

    /// Encode as HEIF/AVIF with an encoder tuning parameter (libvips
    /// `heifsave` with `encoder`/`tune`, for example `"ssim"`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_heif`].
    pub fn encode_heif_tune(
        &self,
        quality: u8,
        compression: &str,
        tune: &str,
    ) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, compression, tune);
        Err(EncodeError::unsupported("heif"))
    }

    /// Encode as JPEG 2000 (libvips `jp2ksave`), lossy or lossless.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]: JPEG 2000 encoding needs an
    /// external `libopenjp2` path.
    pub fn encode_jp2k(&self, quality: u8, lossless: bool) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, lossless);
        Err(EncodeError::unsupported("jp2k"))
    }

    /// Encode as JPEG 2000 with explicit chroma sub-sampling control.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_jp2k`].
    pub fn encode_jp2k_chroma(
        &self,
        quality: u8,
        lossless: bool,
        subsample: bool,
    ) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, lossless, subsample);
        Err(EncodeError::unsupported("jp2k"))
    }

    /// Encode as JPEG XL (libvips `jxlsave`), lossy or lossless.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]: JPEG XL encoding needs an
    /// external `libjxl` path.
    pub fn encode_jxl(&self, lossless: bool) -> Result<Vec<u8>, EncodeError> {
        let _ = lossless;
        Err(EncodeError::unsupported("jxl"))
    }

    /// Encode as Ultra HDR (gain-map JPEG; libvips `uhdrsave`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]: Ultra HDR encoding needs an
    /// external `libultrahdr` path.
    pub fn encode_uhdr(&self, quality: u8) -> Result<Vec<u8>, EncodeError> {
        let _ = quality;
        Err(EncodeError::unsupported("uhdr"))
    }

    /// Encode as Ultra HDR with an explicit gain-map scale factor.
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`]; see [`Raster::encode_uhdr`].
    pub fn encode_uhdr_gainmap_scale(
        &self,
        quality: u8,
        scale_factor: u32,
    ) -> Result<Vec<u8>, EncodeError> {
        let _ = (quality, scale_factor);
        Err(EncodeError::unsupported("uhdr"))
    }

    /// Encode via the ImageMagick/GraphicsMagick delegate to a buffer in the
    /// given format (libvips `magicksave_buffer`, `format` = `".png"`,
    /// `".gif"`, ...).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`], naming the requested `format`:
    /// the magick delegate is an external dependency the pure-Rust build does
    /// not link.
    pub fn magicksave_buffer(&self, format: &str) -> Result<Vec<u8>, EncodeError> {
        Err(EncodeError::unsupported(format))
    }
}

// ---------------------------------------------------------------------------
// Deferred decoders (typed decode error)
// ---------------------------------------------------------------------------

/// Load an image through the ImageMagick/GraphicsMagick delegate (libvips
/// `magickload`).
///
/// # Errors
///
/// Always a [`DecodeError`] naming the magick delegate: it is an external
/// dependency the pure-Rust build does not link.
pub fn magickload(path: &Path) -> Result<Raster, DecodeError> {
    Err(decode_unavailable(format!(
        "magick load of {}",
        path.display()
    )))
}

/// Load an image through the magick delegate with [`MagickLoadOptions`].
///
/// # Errors
///
/// Always a [`DecodeError`]; see [`magickload`].
pub fn magickload_with(path: &Path, opts: MagickLoadOptions) -> Result<Raster, DecodeError> {
    let _ = opts;
    Err(decode_unavailable(format!(
        "magick load of {}",
        path.display()
    )))
}

/// Rasterise an SVG document from bytes at an optional DPI (libvips
/// `svgload_buffer`).
///
/// # Errors
///
/// Always a [`DecodeError`]: SVG rasterisation needs an external `librsvg`
/// path.
pub fn decode_svg(data: &[u8], dpi: Option<f64>) -> Result<Raster, DecodeError> {
    let _ = (data, dpi);
    Err(decode_unavailable("SVG rasterisation"))
}

/// Open a whole-slide image at the given pyramid level through OpenSlide
/// (libvips `openslideload`).
///
/// # Errors
///
/// Always a [`DecodeError`]: OpenSlide is an external dependency the
/// pure-Rust build does not link.
pub fn decode_openslide(path: &Path, level: u32) -> Result<Raster, DecodeError> {
    Err(decode_unavailable(format!(
        "OpenSlide load of {} at level {level}",
        path.display()
    )))
}

/// Decode a file with a libvips `fail_on` strictness level (`"none"`,
/// `"truncated"`, `"warning"`, `"error"`).
///
/// # Errors
///
/// Always a [`DecodeError`]: the `fail_on` strictness knob needs decoder
/// warning/truncation reporting that the pure-Rust decode path does not
/// surface yet.
pub fn decode_file_fail_on(path: &Path, fail_on: &str) -> Result<Raster, DecodeError> {
    Err(decode_unavailable(format!(
        "fail_on={fail_on:?} strict decode of {}",
        path.display()
    )))
}

/// Decode a memory buffer with a libvips `fail_on` strictness level.
///
/// # Errors
///
/// Always a [`DecodeError`]; see [`decode_file_fail_on`].
pub fn decode_bytes_fail_on(data: &[u8], fail_on: &str) -> Result<Raster, DecodeError> {
    let _ = data;
    Err(decode_unavailable(format!(
        "fail_on={fail_on:?} strict decode"
    )))
}

// ---------------------------------------------------------------------------
// DeepZoom to an in-memory zip (real)
// ---------------------------------------------------------------------------

impl Raster {
    /// Save the raster as a DeepZoom image set packed into an in-memory zip
    /// blob (libvips `dzsave_buffer`).
    ///
    /// The blob is a valid, self-contained zip carrying the DeepZoom `.dzi`
    /// manifest for the raster's dimensions plus the image tile. It is
    /// written with the STORE method (no compression) and a hand-computed
    /// CRC-32, so it depends only on the always-present `flate2` crate rather
    /// than the optional `packfile`/`zip` path.
    pub fn dzsave_buffer(&self) -> Vec<u8> {
        const STEM: &str = "image";
        const TILE_SIZE: u32 = 256;

        let dzi = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <Image xmlns=\"http://schemas.microsoft.com/deepzoom/2008\" \
             Format=\"raw\" Overlap=\"0\" TileSize=\"{tile}\">\n  \
             <Size Width=\"{w}\" Height=\"{h}\"/>\n</Image>\n",
            tile = TILE_SIZE,
            w = self.width(),
            h = self.height(),
        );

        let entries = [
            ZipEntry::new(format!("{STEM}.dzi"), dzi.into_bytes()),
            ZipEntry::new(format!("{STEM}_files/0/0_0.raw"), self.data().to_vec()),
        ];
        write_store_zip(&entries)
    }
}

/// One STORE-method zip entry, with its CRC-32 pre-computed.
struct ZipEntry {
    name: String,
    data: Vec<u8>,
    crc: u32,
}

impl ZipEntry {
    fn new(name: String, data: Vec<u8>) -> Self {
        let mut crc = flate2::Crc::new();
        crc.update(&data);
        let crc = crc.sum();
        Self { name, data, crc }
    }
}

/// Append a little-endian `u16`.
fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append a little-endian `u32`.
fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Write the given entries into a minimal but valid STORE-method zip archive.
///
/// The layout is the standard sequence of local file headers and stored data,
/// followed by the central directory and the end-of-central-directory record,
/// all little-endian (APPNOTE 6.3, no compression, no Zip64).
fn write_store_zip(entries: &[ZipEntry]) -> Vec<u8> {
    // 1980-01-01 as the fixed DOS timestamp, so the output is reproducible.
    const DOS_TIME: u16 = 0;
    const DOS_DATE: u16 = 0x0021;

    let mut out = Vec::new();
    let mut offsets = Vec::with_capacity(entries.len());

    for entry in entries {
        offsets.push(out.len() as u32);
        let size = entry.data.len() as u32;
        push_u32(&mut out, 0x0403_4b50); // local file header signature
        push_u16(&mut out, 20); // version needed to extract
        push_u16(&mut out, 0); // general purpose flags
        push_u16(&mut out, 0); // compression method: store
        push_u16(&mut out, DOS_TIME);
        push_u16(&mut out, DOS_DATE);
        push_u32(&mut out, entry.crc);
        push_u32(&mut out, size); // compressed size
        push_u32(&mut out, size); // uncompressed size
        push_u16(&mut out, entry.name.len() as u16);
        push_u16(&mut out, 0); // extra field length
        out.extend_from_slice(entry.name.as_bytes());
        out.extend_from_slice(&entry.data);
    }

    let central_offset = out.len() as u32;
    let mut central = Vec::new();
    for (entry, &offset) in entries.iter().zip(offsets.iter()) {
        let size = entry.data.len() as u32;
        push_u32(&mut central, 0x0201_4b50); // central directory header signature
        push_u16(&mut central, 20); // version made by
        push_u16(&mut central, 20); // version needed to extract
        push_u16(&mut central, 0); // general purpose flags
        push_u16(&mut central, 0); // compression method: store
        push_u16(&mut central, DOS_TIME);
        push_u16(&mut central, DOS_DATE);
        push_u32(&mut central, entry.crc);
        push_u32(&mut central, size); // compressed size
        push_u32(&mut central, size); // uncompressed size
        push_u16(&mut central, entry.name.len() as u16);
        push_u16(&mut central, 0); // extra field length
        push_u16(&mut central, 0); // file comment length
        push_u16(&mut central, 0); // disk number start
        push_u16(&mut central, 0); // internal file attributes
        push_u32(&mut central, 0); // external file attributes
        push_u32(&mut central, offset); // relative offset of local header
        central.extend_from_slice(entry.name.as_bytes());
    }

    let central_size = central.len() as u32;
    out.extend_from_slice(&central);

    push_u32(&mut out, 0x0605_4b50); // end of central directory signature
    push_u16(&mut out, 0); // number of this disk
    push_u16(&mut out, 0); // disk with the central directory
    push_u16(&mut out, entries.len() as u16); // entries on this disk
    push_u16(&mut out, entries.len() as u16); // total entries
    push_u32(&mut out, central_size);
    push_u32(&mut out, central_offset);
    push_u16(&mut out, 0); // zip comment length

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    fn rgb_raster(w: u32, h: u32) -> Raster {
        let data = vec![128u8; w as usize * h as usize * 3];
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    #[test]
    fn encode_heif_reports_unsupported_naming_the_format() {
        let im = rgb_raster(4, 4);
        let err = im.encode_heif(50, "av1").unwrap_err();
        assert!(matches!(err, EncodeError::Unsupported { .. }));
        assert_eq!(err.to_string(), "unsupported encode format: heif");
    }

    #[test]
    fn deferred_encoders_all_report_their_format() {
        let im = rgb_raster(4, 4);
        for (err, format) in [
            (im.encode_heif_lossless("av1").unwrap_err(), "heif"),
            (im.encode_heif_chroma(50, "av1", true).unwrap_err(), "heif"),
            (
                im.encode_heif_lossless_bitdepth("hevc", 8).unwrap_err(),
                "heif",
            ),
            (im.encode_heif_tune(50, "av1", "ssim").unwrap_err(), "heif"),
            (im.encode_jp2k(50, false).unwrap_err(), "jp2k"),
            (im.encode_jp2k_chroma(50, false, true).unwrap_err(), "jp2k"),
            (im.encode_jxl(true).unwrap_err(), "jxl"),
            (im.encode_uhdr(75).unwrap_err(), "uhdr"),
            (im.encode_uhdr_gainmap_scale(75, 4).unwrap_err(), "uhdr"),
        ] {
            match err {
                EncodeError::Unsupported { format: got } => assert_eq!(got, format),
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn magicksave_buffer_names_the_requested_format() {
        let im = rgb_raster(4, 4);
        let err = im.magicksave_buffer(".png").unwrap_err();
        assert_eq!(err.to_string(), "unsupported encode format: .png");
    }

    #[test]
    fn deferred_decoders_report_a_typed_error() {
        let path = Path::new("/nonexistent/slide.svs");
        assert!(magickload(path).is_err());
        assert!(magickload_with(path, MagickLoadOptions::default()).is_err());
        assert!(decode_openslide(path, 0).is_err());
        assert!(decode_file_fail_on(path, "warning").is_err());

        let svg = b"<svg/>";
        let err = decode_svg(svg, None).unwrap_err();
        assert!(err.to_string().contains("SVG"));

        let err = decode_bytes_fail_on(b"1,2,3", "truncated").unwrap_err();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn magick_load_options_supports_struct_update() {
        // The ported cell builds these three shapes; keep them compiling.
        let a = MagickLoadOptions {
            density: Some("100"),
            ..Default::default()
        };
        let b = MagickLoadOptions {
            n: Some(-1),
            ..Default::default()
        };
        let c = MagickLoadOptions {
            page: Some(1),
            n: Some(2),
            ..Default::default()
        };
        assert_eq!(a.density, Some("100"));
        assert_eq!(b.n, Some(-1));
        assert_eq!(c.page, Some(1));
        assert_eq!(c.n, Some(2));
    }

    #[test]
    fn dzsave_buffer_is_a_valid_nontrivial_zip() {
        let im = rgb_raster(64, 64);
        let blob = im.dzsave_buffer();

        // Local file header magic and the end-of-central-directory magic.
        assert_eq!(&blob[..4], &[0x50, 0x4b, 0x03, 0x04]);
        assert!(blob.len() > 1000, "blob was {} bytes", blob.len());
        assert!(
            blob.windows(4).any(|w| w == [0x50, 0x4b, 0x05, 0x06]),
            "missing end-of-central-directory record"
        );
        // The DZI manifest names the raster's dimensions.
        assert!(
            blob.windows(b"Width=\"64\"".len())
                .any(|w| w == b"Width=\"64\""),
            "manifest should carry the raster width"
        );
    }
}
