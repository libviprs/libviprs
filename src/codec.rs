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
