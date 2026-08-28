//! Deferred foreign-format surface: typed stubs for genuinely-external
//! codecs, plus the in-memory DeepZoom zip writer.
//!
//! The ported foreign and connection cells reference a set of encoders and
//! decoders for formats that have no mature pure-Rust implementation yet
//! (HEIF/AVIF, the ImageMagick delegate, OpenSlide, and the libvips
//! `fail_on` strictness knob). Two formats have left this list: JPEG 2000 in
//! issue #501, which replaced its two stubs with [`crate::jp2k`], and Ultra
//! HDR in issue #757, whose two stubs now run [`crate::uhdr`]. This module
//! supplies the rest so the cells compile and pin the typed error path:
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

impl MagickLoadOptions {
    /// Set the delegate rendering density, returning the updated options.
    #[must_use]
    pub fn with_density(mut self, density: Option<&'static str>) -> Self {
        self.density = density;
        self
    }

    /// Set the first page to load, returning the updated options.
    #[must_use]
    pub fn with_page(mut self, page: Option<i32>) -> Self {
        self.page = page;
        self
    }

    /// Set how many pages to load, returning the updated options.
    #[must_use]
    pub fn with_n(mut self, n: Option<i32>) -> Self {
        self.n = n;
        self
    }
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

    /// Encode as Ultra HDR (gain-map JPEG; libvips `uhdrsave`), with the
    /// libvips default gain-map scale factor of 2.
    ///
    /// This is **not** a stub. It runs [`crate::uhdr::encode_uhdr`], which
    /// #508 landed with no new dependency: an Ultra HDR container is two
    /// ordinary JPEGs plus MPF and ISO 21496-1 marker segments, so the
    /// already-required JPEG codec writes both halves.
    ///
    /// `self` must be a **3-band `f32`** raster holding linear-light scRGB,
    /// which is what a gain map is computed from. `quality` is clamped to
    /// 1..=100 by [`crate::uhdr::encode_uhdr`], the way
    /// [`Raster::encode_jpeg`] clamps its own.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidParameter`] if the raster is not 3-band `f32`,
    /// or [`EncodeError::Encode`] if either JPEG half fails to encode.
    pub fn encode_uhdr(&self, quality: u8) -> Result<Vec<u8>, EncodeError> {
        self.encode_uhdr_gainmap_scale(quality, crate::uhdr::SaveOptions::default().gain_map_shrink)
    }

    /// Encode as Ultra HDR with an explicit gain-map scale factor (libvips
    /// `gainmap-scale-factor`): how much smaller than the base image the
    /// gain map is, per axis. 1 keeps it full size, 2 is what libuhdr writes
    /// and what [`Raster::encode_uhdr`] uses.
    ///
    /// # The range, and why an out-of-range factor is refused here
    ///
    /// libvips declares the property as 1..=128 and then, measured on
    /// 8.18.6, silently substitutes the default for anything outside it:
    /// `vips uhdrsave in.v out.jpg --gainmap-scale-factor 0` and
    /// `--gainmap-scale-factor 200` both exit 0 and write the same 2630
    /// bytes as the plain call, with `gainmap-scale-factor: 2` in the
    /// header. A caller cannot act on that, so this refuses instead, the
    /// same call #508 made about the silent `vips_image_get_gainmap`
    /// failure.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidParameter`] if `scale_factor` is outside
    /// 1..=128 or the raster is not 3-band `f32`, or
    /// [`EncodeError::Encode`] if either JPEG half fails to encode.
    pub fn encode_uhdr_gainmap_scale(
        &self,
        quality: u8,
        scale_factor: u32,
    ) -> Result<Vec<u8>, EncodeError> {
        if !(1..=128).contains(&scale_factor) {
            return Err(EncodeError::InvalidParameter(format!(
                "uhdr gain-map scale factor must be 1..=128, got {scale_factor}"
            )));
        }
        crate::uhdr::encode_uhdr(
            self,
            &crate::uhdr::SaveOptions {
                quality,
                gain_map_shrink: scale_factor,
            },
        )
        .map_err(|e| match e {
            crate::uhdr::UhdrError::BadInput { reason } => EncodeError::InvalidParameter(reason),
            other => EncodeError::encode(other),
        })
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

    /// A 3-band `f32` linear-light ramp reaching past the SDR ceiling, so the
    /// gain map it produces is not degenerate.
    fn scrgb_ramp(w: u32, h: u32) -> Raster {
        let mut px: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let t = f64::from(x) / f64::from(w - 1);
                let s = f64::from(y) / f64::from(h - 1);
                px.push((0.02 + 6.0 * t * s) as f32);
                px.push((0.5 * (1.0 - t) + 3.0 * s) as f32);
                px.push((1.5 * t + 0.25) as f32);
            }
        }
        Raster::new(
            w,
            h,
            PixelFormat::FloatF32(std::num::NonZeroU16::new(3).unwrap()),
            px.into_iter().flat_map(f32::to_ne_bytes).collect(),
        )
        .unwrap()
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
        ] {
            match err {
                EncodeError::Unsupported { format: got } => assert_eq!(got, format),
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    /// Issue #757. `crate::uhdr` has written a container libvips reads back
    /// since #508, so this entry point stops refusing and encodes. The bytes
    /// go through the crate's own two-stage detection gate and are expanded
    /// again, which is what says they are a real Ultra HDR file rather than
    /// any two JPEGs stuck together.
    #[test]
    fn encode_uhdr_writes_a_container_that_reads_back() {
        let src = scrgb_ramp(16, 16);
        let bytes = src.encode_uhdr(75).expect("a 3-band float raster encodes");
        assert!(
            crate::uhdr::is_uhdr(&bytes),
            "the bytes must satisfy the two-stage Ultra HDR gate"
        );
        let back = crate::uhdr::from_container(&bytes, crate::source::DecodeLimits::default())
            .expect("the container expands again");
        assert_eq!((back.width(), back.height()), (16, 16));
        assert_eq!(back.meta.interpretation, Some(crate::Interpretation::ScRgb));
    }

    /// Issue #757. The gain-map scale factor is the libvips
    /// `gainmap-scale-factor`, so it has to reach the writer rather than be
    /// swallowed. A full-size gain map costs more bytes than a half-size one
    /// over the same pixels, which is the cheapest observable saying the
    /// argument is used at all.
    #[test]
    fn encode_uhdr_gainmap_scale_reaches_the_writer() {
        let src = scrgb_ramp(32, 32);
        let half = src.encode_uhdr_gainmap_scale(90, 2).expect("shrink 2");
        let full = src.encode_uhdr_gainmap_scale(90, 1).expect("shrink 1");
        assert!(
            full.len() > half.len(),
            "a full-size gain map should cost more than a half-size one, got {} against {}",
            full.len(),
            half.len()
        );
        // And the plain form is the shrink-2 default rather than some third
        // thing, which is what `SaveOptions::default` and `uhdrsave` both say.
        assert_eq!(src.encode_uhdr(90).expect("default"), half);
    }

    /// Issue #757. The one real decision in that issue is the input contract.
    /// `uhdr::encode_uhdr` computes a gain map from linear-light scRGB, so a
    /// raster that is not 3-band `f32` is refused, and the refusal is
    /// [`EncodeError::InvalidParameter`] rather than
    /// [`EncodeError::Unsupported`]: this build *can* write Ultra HDR and
    /// this raster is the wrong shape for it, which is a different answer.
    #[test]
    fn encode_uhdr_refuses_a_raster_that_is_not_three_band_float() {
        let rgb = rgb_raster(8, 8);
        let err = rgb.encode_uhdr(75).unwrap_err();
        assert!(
            matches!(err, EncodeError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            err.to_string().contains("Rgb8"),
            "the refusal should name what it got, given {err}"
        );
        // A one-band float raster is refused for the same reason, so the gate
        // is on the whole format and not only on the sample type.
        let mono = Raster::new(
            4,
            4,
            PixelFormat::FloatF32(std::num::NonZeroU16::new(1).unwrap()),
            vec![0u8; 4 * 4 * 4],
        )
        .unwrap();
        assert!(matches!(
            mono.encode_uhdr(75).unwrap_err(),
            EncodeError::InvalidParameter(_)
        ));
    }

    /// Issue #757. `quality` is a `u8` and libvips' `Q` is 1..=100, and the
    /// doc says the out-of-range ends clamp rather than refuse, which is what
    /// [`Raster::encode_jpeg`] does. Without this the clamp is a claim with
    /// nothing behind it, and it lives one call away in
    /// [`crate::uhdr::encode_uhdr`] rather than here.
    #[test]
    fn encode_uhdr_clamps_the_quality_to_the_libvips_range() {
        let src = scrgb_ramp(8, 8);
        assert_eq!(
            src.encode_uhdr(0).expect("0 clamps"),
            src.encode_uhdr(1).expect("1")
        );
        assert_eq!(
            src.encode_uhdr(200).expect("200 clamps"),
            src.encode_uhdr(100).expect("100")
        );
        // Positive control: the two ends are not the same bytes, so the
        // assertions above are not comparing everything against everything.
        assert_ne!(
            src.encode_uhdr(1).expect("1"),
            src.encode_uhdr(100).expect("100")
        );
    }

    /// Issue #757. libvips declares `gainmap-scale-factor` as 1..=128 and
    /// then, measured on 8.18.6, **silently substitutes the default** for
    /// anything outside it: `vips uhdrsave in.v out.jpg
    /// --gainmap-scale-factor 0` and `--gainmap-scale-factor 200` both exit 0
    /// and write the same 2630 bytes as the plain call, with
    /// `gainmap-scale-factor: 2` in the header. #508 already declined to
    /// reproduce one silent libvips failure in this area, so an out-of-range
    /// factor is a typed refusal here.
    #[test]
    fn encode_uhdr_refuses_a_scale_factor_outside_the_libvips_range() {
        let src = scrgb_ramp(8, 8);
        for bad in [0u32, 129, u32::MAX] {
            let err = src.encode_uhdr_gainmap_scale(75, bad).unwrap_err();
            assert!(
                matches!(err, EncodeError::InvalidParameter(_)),
                "scale factor {bad} should be refused, got {err:?}"
            );
        }
        // Both ends of the declared range are accepted, so this is a range
        // check and not a blanket one.
        for good in [1u32, 128] {
            assert!(
                src.encode_uhdr_gainmap_scale(75, good).is_ok(),
                "scale factor {good} is inside the libvips range"
            );
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
