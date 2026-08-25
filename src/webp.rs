//! WebP lane: the save-side option types and the `webpsave` surface on
//! [`Raster`].
//!
//! This module is the single file the WebP lane (issue #497) owns. It exists
//! ahead of that lane so the format wave does not have four PRs rewriting one
//! paragraph of [`crate::encode`]'s header and one adjacent pair of stub
//! bodies (issue #563). The encode side is still a typed stub:
//! [`Raster::encode_webp`] reports [`EncodeError::Unsupported`], because
//! wiring up a codec belongs to the WebP lane and not to this one. A
//! lossless encoder *is* now reachable, since enabling the `image` crate's
//! `webp` feature brings `image::codecs::webp::WebPEncoder` with it, so the
//! lane's encode side is a small job and this stub is a scheduling decision
//! rather than a missing dependency. Decoding is not
//! handled here at all; a WebP file is recognised by its magic in the
//! shared sniff table in [`crate::source`] and decoded through the `image`
//! facade like every other streaming format.
//!
//! # Operations
//!
//! | libviprs method        | libvips equivalent | result                                     |
//! |------------------------|--------------------|--------------------------------------------|
//! | [`Raster::encode_webp`] | `webpsave_buffer`  | [`EncodeError::Unsupported`] (`"webp"`)    |
//!
//! # Semantics
//!
//! * vips `webpsave` (`foreign/webpsave.c`) takes `Q` (default 75) *and*
//!   `lossless` (default `false`), so quality only means anything on the
//!   lossy path. The pure-Rust encoder libviprs can reach is lossless-only
//!   and has no quality knob at all.
//! * That is why [`SaveOptions`] carries a [`Compression`] rather than a
//!   `quality: u8`. A quality argument that the encoder throws away inverts
//!   the contract (ask for quality 10, get a lossless file possibly larger
//!   than the PNG you started from) and is a semver time bomb: the day a
//!   lossy encoder lands, every existing `encode_webp(10)` would silently
//!   start emitting small lossy files in a patch release. Making quality
//!   unrepresentable turns that into a compile error today instead.
//! * [`Compression`] is `#[non_exhaustive]` precisely so
//!   `Compression::Lossy { .. }` can be added as a minor bump when there is
//!   an encoder behind it.
//! * [`SaveOptions`] is deliberately *not* `#[non_exhaustive]`: that would
//!   block the struct literal downstream and kill `..Default::default()`,
//!   which is the whole point of an options struct.
//!
//! Every entry point here is fallible and returns [`EncodeError`]; there is
//! no panicking twin, matching the rest of the encode surface in
//! [`crate::encode`].

use crate::codec::EncodeError;
use crate::raster::Raster;

/// How the WebP encoder compresses pixels (libvips `webpsave`'s `lossless`
/// flag plus its `Q` factor, folded into one axis).
///
/// Lossless is the only representable mode because it is the only mode this
/// build can encode. `Lossy { quality }` joins the enum when there is a
/// lossy encoder to back it, which is a minor bump rather than a breaking
/// one thanks to `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// Lossless compression (libvips `webpsave --lossless`). The pixels
    /// round-trip exactly and there is no quality factor to set.
    #[default]
    Lossless,
}

/// Options for [`Raster::encode_webp`] (libvips `webpsave` / `webpsave_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `webp::SaveOptions { compression, ..Default::default() }` and later
/// fields can be added without a breaking change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveOptions {
    /// How to compress. Defaults to [`Compression::Lossless`], the only
    /// mode with an encoder behind it.
    pub compression: Compression,
}

impl Raster {
    /// Encode as WebP bytes (libvips `webpsave_buffer`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`] with format `"webp"`: no codec is
    /// wired up here yet, so the call site compiles and the error path is
    /// pinned, but no bytes are produced.
    pub fn encode_webp(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let _ = options;
        Err(EncodeError::unsupported("webp"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    /**
     * Tests that the WebP encoder is still a typed stub rather than a
     * panic or a silent empty buffer, and that it reports the format tag
     * downstream code matches on. Works by encoding a small RGB raster at
     * the default options and at an explicitly spelled-out lossless
     * setting, and checking both report `Unsupported { format: "webp" }`.
     * Input: 8x8 Rgb8 raster -> Output: `EncodeError::Unsupported` with
     * format `"webp"` from both calls.
     */
    #[test]
    fn encode_webp_is_a_typed_unsupported_stub() {
        let im = Raster::new(8, 8, PixelFormat::Rgb8, vec![7u8; 8 * 8 * 3]).unwrap();
        for options in [
            SaveOptions::default(),
            SaveOptions {
                compression: Compression::Lossless,
            },
        ] {
            match im.encode_webp(options) {
                Err(EncodeError::Unsupported { format }) => assert_eq!(format, "webp"),
                other => panic!("expected Unsupported(webp), got {other:?}"),
            }
        }
    }

    /**
     * Pins the shape of the options struct the WebP lane inherits: the
     * default is lossless, and the struct is open enough to build with
     * `..Default::default()` from outside its own module. Works by
     * comparing `SaveOptions::default()` against an explicit literal and
     * against a functional-update literal.
     * Input: none -> Output: all three spellings compare equal, with
     * compression `Lossless`.
     */
    #[test]
    fn save_options_default_is_lossless_and_updatable() {
        let explicit = SaveOptions {
            compression: Compression::Lossless,
        };
        let updated = SaveOptions {
            ..Default::default()
        };
        assert_eq!(SaveOptions::default(), explicit);
        assert_eq!(updated, explicit);
        assert_eq!(Compression::default(), Compression::Lossless);
    }
}
