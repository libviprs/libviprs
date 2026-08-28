//! Shared codec spine: the error taxonomy and format-option enums that the
//! encoder, connection, and TIFF lanes build on.
//!
//! This module deliberately stays dependency-light. It carries the *types*
//! the codec surface agrees on (so every lane returns and accepts the same
//! [`EncodeError`], [`JpegSubsample`], and [`TiffCompression`]) but no codec
//! logic: the JPEG/PNG/TIFF/text encoders and the `Source`/`Target`
//! connection abstraction live in their own modules and share this spine.
//!
//! ## Error types
//!
//! * [`EncodeError`] — every raster encoder (`encode_jpeg`, `encode_png`,
//!   `encode_to_target`, the `save_*` family, ...) reports failures through
//!   this type. Genuinely-external formats that the pure-Rust build cannot
//!   emit yet return [`EncodeError::Unsupported`] so the call site compiles
//!   and asserts the typed error path.
//! * [`DecodeError`] — a re-export alias of [`crate::source::SourceError`],
//!   the single error the decode surface (`decode_file`, `decode_bytes`,
//!   `decode_source`, ...) already returns. The ported cells name the decode
//!   error `DecodeError` in their API contracts; aliasing keeps that name
//!   available without introducing a second, redundant error type.

use thiserror::Error;

/// The decode error type shared across the codec surface.
///
/// This is an alias of [`crate::source::SourceError`], the error that
/// [`crate::decode_file`], [`crate::decode_bytes`], and the connection
/// lane's `decode_source` already return. The ported foreign, connection,
/// and iofuncs cells refer to the decode error as `DecodeError` in their API
/// contracts; the alias lets those contracts resolve to the one real type
/// rather than a duplicate.
pub type DecodeError = crate::source::SourceError;

/// Errors returned by the raster encoders.
///
/// Every encoder on [`crate::Raster`] that returns bytes (`encode_jpeg`,
/// `encode_png`, `encode_to_buffer`, ...) and the connection lane's
/// `encode_to_target` report failures through this type. Encoders backed by
/// the pure-Rust `image` / `png` crates surface their crate errors through
/// [`EncodeError::Encode`]; writes to a caller-supplied target surface
/// through [`EncodeError::Io`]; and formats that require an external C
/// library (HEIF/AVIF, JP2K, JPEG-XL, UHDR, magick, ...) return
/// [`EncodeError::Unsupported`] so the ported cells compile and pin the typed
/// error path.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EncodeError {
    /// Writing encoded bytes to a caller-supplied target failed.
    #[error("I/O error while encoding: {0}")]
    Io(#[from] std::io::Error),
    /// The underlying encoder (for example the `image` or `png` crate)
    /// rejected the raster or failed mid-stream. The message is the
    /// encoder's own error rendered to a string, keeping this spine free of
    /// a dependency on any specific codec crate's error type.
    #[error("image encode error: {0}")]
    Encode(String),
    /// A caller argument was outside the range the encoder accepts (for
    /// example a quality or compression level the format cannot express).
    #[error("invalid encode parameter: {0}")]
    InvalidParameter(String),
    /// The requested output format is not available in this build.
    ///
    /// The deferred lanes return this for genuinely-external formats that
    /// have no mature pure-Rust encoder (HEIF/AVIF, JP2K, JPEG-XL, UHDR,
    /// FITS, magick, TIFF JPEG/CCITT, ...). The call sites compile and
    /// assert on the typed error rather than a panic.
    ///
    /// [`crate::webp`] and [`crate::gif`] also return it, but for a
    /// different reason: a pure-Rust codec is reachable for both, and the
    /// stubs are waiting on their own lanes rather than on a dependency.
    #[error("unsupported encode format: {format}")]
    Unsupported {
        /// The format name the caller asked for (for example `"heif"`).
        format: String,
    },
}

impl EncodeError {
    /// Construct an [`EncodeError::Unsupported`] for the named format.
    ///
    /// A convenience for the deferred-format stubs so a lane can write
    /// `EncodeError::unsupported("heif")` rather than spelling out the
    /// struct variant and the `.to_string()` conversion.
    pub fn unsupported(format: impl Into<String>) -> Self {
        Self::Unsupported {
            format: format.into(),
        }
    }

    /// Construct an [`EncodeError::Encode`] from any displayable error, so a
    /// lane can write `.map_err(EncodeError::encode)` over an `image` or
    /// `png` crate error without this spine depending on those crates.
    pub fn encode(err: impl std::fmt::Display) -> Self {
        Self::Encode(err.to_string())
    }
}

/// JPEG chroma-subsampling mode (libvips `subsample_mode`).
///
/// Selects how the JPEG encoder subsamples the chroma planes:
///
/// * [`JpegSubsample::Auto`] — let the encoder decide from the quality
///   (libvips `VIPS_FOREIGN_SUBSAMPLE_AUTO`): 4:2:0 below quality 90, 4:4:4
///   at or above.
/// * [`JpegSubsample::Off`] — never subsample (4:4:4, full chroma).
/// * [`JpegSubsample::On`] — always subsample (4:2:0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JpegSubsample {
    /// Pick 4:4:4 or 4:2:0 automatically from the quality setting.
    Auto,
    /// Full chroma resolution (4:4:4).
    Off,
    /// Subsampled chroma (4:2:0).
    On,
}

/// TIFF compression scheme (libvips `compression`).
///
/// The variants the ported foreign cells reference for `save_tiff`,
/// `save_bigtiff`, and `save_tiff_tiled`. [`TiffCompression::Jp2k`] is the
/// extended mode the tiled-TIFF cell adds. Which of these a given build can
/// actually emit is the TIFF lane's concern; the spine only fixes the names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TiffCompression {
    /// No compression (raw strips/tiles).
    None,
    /// Lempel-Ziv-Welch.
    Lzw,
    /// Baseline JPEG in-TIFF (lossy; requires an external JPEG-in-TIFF path).
    Jpeg,
    /// Zlib/Deflate.
    Deflate,
    /// CCITT Group 4 fax (bilevel images only).
    Ccitt,
    /// JPEG 2000 in-TIFF (requires an external JP2K path).
    Jp2k,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::Raster;

    /// Issue #758. This module's two prose lists name the formats whose
    /// encoders answer [`EncodeError::Unsupported`], and the second one is the
    /// documentation for the variant itself, so a name on it that this build
    /// can actually encode makes the variant's contract wrong rather than
    /// merely stale.
    ///
    /// The "can encode" half is **measured here** by calling the entry point,
    /// not declared in a table, so the guard tracks the tree rather than
    /// someone's memory of it.
    #[test]
    fn the_unsupported_doc_lists_name_no_format_this_build_encodes() {
        let src = include_str!("codec.rs");
        let list_after = |anchor: &str| -> Vec<String> {
            let at = src
                .find(anchor)
                .unwrap_or_else(|| panic!("the doc anchor {anchor:?} moved"));
            let rest = &src[at + anchor.len()..];
            let end = rest.find(')').expect("the list is parenthesised");
            rest[..end]
                .replace("///", " ")
                .replace("//!", " ")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "...")
                .collect()
        };
        let named: Vec<String> = [
            list_after("formats that require an external C\n/// library ("),
            list_after("have no mature pure-Rust encoder ("),
        ]
        .concat();
        // The extraction is the thing most likely to rot into a vacuous pass,
        // so pin that it found both lists and that they still name the two
        // formats nothing in this tree encodes.
        assert!(
            named.len() >= 6,
            "only extracted {named:?} from the two doc lists"
        );
        for must in ["HEIF/AVIF", "magick"] {
            assert!(
                named.iter().any(|n| n == must),
                "expected {must} on a list, got {named:?}"
            );
        }

        let rgb = Raster::new(4, 4, crate::pixel::PixelFormat::Rgb8, vec![128u8; 48]).unwrap();
        let scrgb = Raster::new(
            4,
            4,
            crate::pixel::PixelFormat::FloatF32(std::num::NonZeroU16::new(3).unwrap()),
            (0..48)
                .flat_map(|i| (i as f32 / 48.0).to_ne_bytes())
                .collect(),
        )
        .unwrap();
        // Each probe runs the encoder, so `encodes` is a measurement.
        let probes: [(&str, bool); 5] = [
            ("FITS", rgb.encode_fits().is_ok()),
            (
                "UHDR",
                crate::uhdr::encode_uhdr(&scrgb, &crate::uhdr::SaveOptions::default()).is_ok(),
            ),
            ("HEIF/AVIF", rgb.encode_heif(50, "av1").is_ok()),
            (
                "JP2K",
                rgb.encode_jp2k(crate::jp2k::SaveOptions::default()).is_ok(),
            ),
            ("magick", rgb.magicksave_buffer(".png").is_ok()),
        ];
        // Both directions have to be represented or a probe list that always
        // answered the same way would pass vacuously.
        assert!(
            probes.iter().any(|p| p.1) && probes.iter().any(|p| !p.1),
            "the probes must cover both a format this build encodes and one it \
             does not, got {probes:?}"
        );
        let wrong: Vec<&str> = probes
            .iter()
            .filter(|(name, encodes)| *encodes && named.iter().any(|n| n == name))
            .map(|(name, _)| *name)
            .collect();
        assert!(
            wrong.is_empty(),
            "{wrong:?} are named in this module's Unsupported doc lists, and \
             this build encodes every one of them"
        );
    }

    #[test]
    fn jpeg_subsample_has_the_three_cell_variants() {
        // The ported foreign cell names `JpegSubsample { Auto, Off, On }`.
        let all = [JpegSubsample::Auto, JpegSubsample::Off, JpegSubsample::On];
        assert_eq!(all.len(), 3);
        assert_ne!(JpegSubsample::Off, JpegSubsample::On);
    }

    #[test]
    fn tiff_compression_has_the_six_cell_variants() {
        // The ported foreign cell names the extended set
        // `TiffCompression { None, Lzw, Jpeg, Deflate, Ccitt, Jp2k }`.
        let all = [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Jpeg,
            TiffCompression::Deflate,
            TiffCompression::Ccitt,
            TiffCompression::Jp2k,
        ];
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn encode_error_unsupported_names_the_format() {
        let e = EncodeError::unsupported("heif");
        assert!(matches!(e, EncodeError::Unsupported { .. }));
        assert_eq!(e.to_string(), "unsupported encode format: heif");
    }

    #[test]
    fn encode_error_wraps_a_displayable_cause() {
        let e = EncodeError::encode("boom");
        assert_eq!(e.to_string(), "image encode error: boom");
    }

    #[test]
    fn encode_error_carries_io() {
        let io = std::io::Error::new(std::io::ErrorKind::WriteZero, "short write");
        let e = EncodeError::from(io);
        assert!(matches!(e, EncodeError::Io(_)));
    }

    #[test]
    fn decode_error_is_an_alias_of_source_error() {
        // The fn-item coercion below only type-checks when `DecodeError` and
        // `SourceError` are the same type, pinning the alias.
        fn takes_decode(_: DecodeError) {}
        let _f: fn(crate::source::SourceError) = takes_decode;
    }
}
