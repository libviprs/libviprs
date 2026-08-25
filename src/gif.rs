//! GIF lane: the save-side option types and the `gifsave` surface on
//! [`Raster`].
//!
//! This module is the single file the GIF lane (issue #499) owns. Like
//! [`crate::webp`] it exists ahead of that lane so four format PRs do not
//! collide inside [`crate::encode`] (issue #563). The encode side is still a
//! typed stub: GIF save needs the multi-page animation model and the palette
//! quantisation work the lane carries, so [`Raster::encode_gif`] reports
//! [`EncodeError::Unsupported`]. As with [`crate::webp`] that is a
//! scheduling decision and not a missing dependency: enabling the `image`
//! crate's `gif` feature brings `image::codecs::gif::GifEncoder` with it.
//! Decoding is not handled here; a GIF is
//! recognised by its magic in the shared sniff table in [`crate::source`]
//! and decoded through the `image` facade like every other streaming format.
//!
//! # Operations
//!
//! | libviprs method       | libvips equivalent | result                                  |
//! |-----------------------|--------------------|-----------------------------------------|
//! | [`Raster::encode_gif`] | `gifsave_buffer`   | [`EncodeError::Unsupported`] (`"gif"`)  |
//!
//! # Semantics
//!
//! * The option defaults are the ones vips 8.18.4 reports for `gifsave`
//!   (`foreign/cgifsave.c`), measured from the binary rather than guessed:
//!   `dither` defaults to 1 over the range 0 to 1, and `interlace` defaults
//!   to `false`.
//! * The three separate stubs this replaces (`encode_gif`,
//!   `encode_gif_interlaced`, `encode_gif_dither`) were the start of the
//!   arity explosion the format roadmap exists to avoid: vips `gifsave` has
//!   twelve options, and one method per combination does not scale. One
//!   method taking [`SaveOptions`] does.
//! * [`SaveOptions`] is deliberately *not* `#[non_exhaustive]`: that would
//!   block the struct literal downstream and kill `..Default::default()`,
//!   which is the whole point of an options struct. The lane adds
//!   `effort`, `bitdepth`, and the interframe knobs as plain fields.
//!
//! Every entry point here is fallible and returns [`EncodeError`]; there is
//! no panicking twin, matching the rest of the encode surface in
//! [`crate::encode`].

use crate::codec::EncodeError;
use crate::raster::Raster;

/// libvips `gifsave`'s `dither` default, measured on vips 8.18.4
/// (`dither`, `default: 1`, `min: 0`, `max: 1`).
const DEFAULT_DITHER: f64 = 1.0;

/// Options for [`Raster::encode_gif`] (libvips `gifsave` / `gifsave_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `gif::SaveOptions { dither: 0.0, ..Default::default() }` and later fields
/// can be added without a breaking change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaveOptions {
    /// Write the frames interlaced (libvips `gifsave` `interlace`).
    /// Defaults to `false`, as vips does.
    pub interlaced: bool,
    /// Amount of dithering applied during palette quantisation, 0.0 to 1.0
    /// (libvips `gifsave` `dither`). Defaults to 1.0, as vips does.
    pub dither: f64,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            interlaced: false,
            dither: DEFAULT_DITHER,
        }
    }
}

impl Raster {
    /// Encode as GIF bytes (libvips `gifsave_buffer`).
    ///
    /// # Errors
    ///
    /// Always [`EncodeError::Unsupported`] with format `"gif"`: GIF save
    /// needs the multi-page animation model and the palette quantisation the
    /// GIF lane carries, so no codec is wired up here yet. The call site
    /// compiles and the error path is pinned, but no bytes are produced.
    pub fn encode_gif(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let _ = options;
        Err(EncodeError::unsupported("gif"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    /**
     * Tests that the GIF encoder is still a typed stub rather than a panic
     * or a silent empty buffer, and that it reports the format tag
     * downstream code matches on, whichever options it is handed. Works by
     * encoding a small RGB raster at the default options, interlaced, and
     * with dithering off, and checking all three report
     * `Unsupported { format: "gif" }`.
     * Input: 8x8 Rgb8 raster -> Output: `EncodeError::Unsupported` with
     * format `"gif"` from every call.
     */
    #[test]
    fn encode_gif_is_a_typed_unsupported_stub() {
        let im = Raster::new(8, 8, PixelFormat::Rgb8, vec![7u8; 8 * 8 * 3]).unwrap();
        for options in [
            SaveOptions::default(),
            SaveOptions {
                interlaced: true,
                ..Default::default()
            },
            SaveOptions {
                dither: 0.0,
                ..Default::default()
            },
        ] {
            match im.encode_gif(options) {
                Err(EncodeError::Unsupported { format }) => assert_eq!(format, "gif"),
                other => panic!("expected Unsupported(gif), got {other:?}"),
            }
        }
    }

    /**
     * Pins the option defaults against the vips 8.18.4 `gifsave` operation
     * description, so the GIF lane inherits the right starting point rather
     * than rediscovering it. Works by reading the defaults straight off
     * `SaveOptions::default()`.
     * Input: none -> Output: `interlaced == false` and `dither == 1.0`,
     * matching `vips gifsave`'s reported defaults.
     */
    #[test]
    fn save_options_defaults_match_vips() {
        let d = SaveOptions::default();
        assert!(!d.interlaced, "vips gifsave interlace defaults to false");
        assert!(
            (d.dither - 1.0).abs() < f64::EPSILON,
            "vips gifsave dither defaults to 1, got {}",
            d.dither
        );
    }
}
