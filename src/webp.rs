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
    use crate::imageio::MetadataValue;
    use crate::pixel::PixelFormat;
    use crate::source::{DecodeLimits, decode_bytes_with_limits};

    // -----------------------------------------------------------------
    // Oracle fixtures. Every byte below came out of vips 8.18.4; the
    // commands are in `oracle-captures/foreign-webp/commands.sh` and the
    // expected pixels in its `oracle.json`.
    // -----------------------------------------------------------------

    /// `vips webpsave --lossless --keep none` on the 4x3 sRGB raster in
    /// [`ramp_rgb`], captured verbatim. One `VP8L` chunk, 100 bytes.
    const LOSSLESS_RGB: [u8; 100] = [
        0x52, 0x49, 0x46, 0x46, 0x5c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x50, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x60, 0x98, 0x8d, 0xa4,
        0x7d, 0x0e, 0xcd, 0x7a, 0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0, 0x18, 0x64, 0x1b, 0xe9, 0x04,
        0x4e, 0xe0, 0x90, 0x06, 0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83, 0x6c, 0x23, 0x1d, 0xc2, 0x44,
        0x4e, 0xe8, 0x01, 0x4f, 0xe8, 0x45, 0x86, 0x30, 0xff, 0x91, 0xa4, 0x48, 0xf4, 0x77, 0x44,
        0xfb, 0xea, 0x06, 0x81, 0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08, 0x20, 0xf1, 0x48, 0x28, 0x80,
        0x0d, 0x26, 0xd9, 0x63, 0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f,
    ];

    /// The same 4x3 raster with an alpha ramp, `vips webpsave --lossless
    /// --keep none`. One `VP8L` chunk, 116 bytes.
    const LOSSLESS_RGBA: [u8; 116] = [
        0x52, 0x49, 0x46, 0x46, 0x6c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x60, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x10, 0x5f, 0xa0, 0xa4, 0x8d, 0x24,
        0x68, 0xe1, 0x3d, 0x2a, 0x0f, 0x8e, 0xfb, 0x83, 0x23, 0xc5, 0x20, 0xdb, 0x48, 0x27, 0x76,
        0x02, 0xd3, 0x78, 0xa5, 0x67, 0x39, 0xb1, 0x09, 0x18, 0x66, 0x23, 0x69, 0x11, 0x56, 0x60,
        0x11, 0x86, 0xe5, 0xfc, 0x7f, 0x87, 0x30, 0x08, 0x6a, 0xda, 0x36, 0x82, 0x4a, 0xf1, 0x3e,
        0x38, 0x99, 0x4a, 0xb2, 0x58, 0x6e, 0x8a, 0x24, 0x12, 0x11, 0x41, 0x52, 0xab, 0xff, 0xff,
        0x91, 0xd4, 0xea, 0xeb, 0x6e, 0xef, 0x0a, 0x89, 0x08, 0x08, 0x02, 0x48, 0x3c, 0xa2, 0x89,
        0x61, 0x85, 0x61, 0x66, 0xda, 0x34, 0xa2, 0xff, 0xe1, 0xe8, 0x03,
    ];

    /// `vips webpsave` at the default `Q` on the same 4x3 raster: a `VP8`
    /// chunk, the lossy bitstream libviprs decodes but cannot write.
    const LOSSY_RGB: [u8; 96] = [
        0x52, 0x49, 0x46, 0x46, 0x58, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x20, 0x4c, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x04, 0x00, 0x03, 0x00,
        0x02, 0x00, 0x34, 0x25, 0xb0, 0x02, 0x74, 0x01, 0x0e, 0xfe, 0x03, 0xc8, 0x00, 0x00, 0xfc,
        0x3c, 0x7e, 0x73, 0xd3, 0xe4, 0x80, 0x52, 0xee, 0x82, 0x37, 0xda, 0xf7, 0x4f, 0xea, 0xd3,
        0xe3, 0xd3, 0xf7, 0xff, 0x5b, 0x8b, 0x76, 0x19, 0xcc, 0xfa, 0x2d, 0xf7, 0xdf, 0xee, 0x72,
        0x65, 0x9b, 0xfe, 0x35, 0x44, 0xe9, 0x04, 0x77, 0xca, 0xd5, 0x96, 0xb8, 0xf9, 0xc9, 0xe2,
        0x39, 0xfa, 0xd7, 0xa8, 0x80, 0x00,
    ];

    /// The `LOSSLESS_RGB` bitstream rewrapped in an extended container
    /// with an `ICCP`, an `EXIF` and an `XMP ` chunk, all three flagged in
    /// `VP8X`. vips reports them as 24, 10 and 37 bytes of binary data.
    const META_RGB: [u8; 214] = [
        0x52, 0x49, 0x46, 0x46, 0xce, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x49, 0x43, 0x43, 0x50, 0x18, 0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
        0x26, 0x27, 0x56, 0x50, 0x38, 0x4c, 0x50, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00,
        0x5f, 0x60, 0x98, 0x8d, 0xa4, 0x7d, 0x0e, 0xcd, 0x7a, 0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0,
        0x18, 0x64, 0x1b, 0xe9, 0x04, 0x4e, 0xe0, 0x90, 0x06, 0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83,
        0x6c, 0x23, 0x1d, 0xc2, 0x44, 0x4e, 0xe8, 0x01, 0x4f, 0xe8, 0x45, 0x86, 0x30, 0xff, 0x91,
        0xa4, 0x48, 0xf4, 0x77, 0x44, 0xfb, 0xea, 0x06, 0x81, 0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08,
        0x20, 0xf1, 0x48, 0x28, 0x80, 0x0d, 0x26, 0xd9, 0x63, 0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f,
        0x45, 0x58, 0x49, 0x46, 0x0a, 0x00, 0x00, 0x00, 0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x58, 0x4d, 0x50, 0x20, 0x25, 0x00, 0x00, 0x00, 0x3c, 0x78, 0x3a, 0x78,
        0x6d, 0x70, 0x6d, 0x65, 0x74, 0x61, 0x20, 0x78, 0x6d, 0x6c, 0x6e, 0x73, 0x3a, 0x78, 0x3d,
        0x22, 0x61, 0x64, 0x6f, 0x62, 0x65, 0x3a, 0x6e, 0x73, 0x3a, 0x6d, 0x65, 0x74, 0x61, 0x2f,
        0x22, 0x2f, 0x3e, 0x00,
    ];

    /// `vips webpsave --lossless --page-height 3` on a 4x9 toilet-roll: an
    /// animation of three 4x3 frames, whose frame 0 is the `LOSSLESS_RGB`
    /// image. vips reports `n-pages: 3` and loads 4x3 by default.
    const ANIM3: [u8; 374] = [
        0x52, 0x49, 0x46, 0x46, 0x6e, 0x01, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x50, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x60, 0x98, 0x8d, 0xa4, 0x7d, 0x0e, 0xcd, 0x7a,
        0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0, 0x18, 0x64, 0x1b, 0xe9, 0x04, 0x4e, 0xe0, 0x90, 0x06,
        0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83, 0x6c, 0x23, 0x1d, 0xc2, 0x44, 0x4e, 0xe8, 0x01, 0x4f,
        0xe8, 0x45, 0x86, 0x30, 0xff, 0x91, 0xa4, 0x48, 0xf4, 0x77, 0x44, 0xfb, 0xea, 0x06, 0x81,
        0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08, 0x20, 0xf1, 0x48, 0x28, 0x80, 0x0d, 0x26, 0xd9, 0x63,
        0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f, 0x41, 0x4e, 0x4d, 0x46, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
        0x56, 0x50, 0x38, 0x4c, 0x4b, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x40,
        0x16, 0x60, 0xf2, 0x8e, 0x26, 0xdd, 0xbc, 0xa2, 0x08, 0x61, 0x28, 0x64, 0x01, 0x26, 0x04,
        0x09, 0x84, 0x34, 0xe0, 0xe1, 0x74, 0xc6, 0xa1, 0x90, 0x8d, 0x24, 0x48, 0x61, 0xf8, 0x16,
        0x70, 0x79, 0xee, 0x11, 0x19, 0x84, 0xf9, 0x8f, 0x6f, 0xa6, 0x85, 0x40, 0xa4, 0x85, 0x68,
        0xde, 0x1e, 0x22, 0xd1, 0x1e, 0x22, 0x90, 0x05, 0x98, 0xfc, 0x33, 0x18, 0x49, 0x06, 0x99,
        0x25, 0x14, 0x6d, 0x44, 0xff, 0x23, 0xec, 0x15, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x66, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64,
        0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x4d, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00,
        0x00, 0x5f, 0x60, 0x90, 0x6d, 0x24, 0x80, 0x23, 0x3d, 0xba, 0x77, 0x39, 0x85, 0x97, 0x28,
        0x8a, 0x41, 0xb6, 0x91, 0x90, 0xce, 0xa3, 0x1a, 0xcf, 0xfc, 0x0a, 0x27, 0x70, 0x02, 0x66,
        0x02, 0x26, 0x01, 0xcd, 0x43, 0x16, 0x7f, 0xfa, 0x07, 0x58, 0x03, 0xe7, 0x3f, 0xfe, 0x3f,
        0xb8, 0x2c, 0x00, 0xd6, 0x02, 0x5d, 0xe4, 0x4e, 0x17, 0x60, 0x2d, 0x90, 0x05, 0x98, 0x48,
        0x4c, 0x66, 0x36, 0x49, 0xe4, 0xd4, 0x1d, 0x6d, 0x44, 0xff, 0xe3, 0x35, 0x03, 0x00,
    ];

    /// The 4x3 sRGB ramp every fixture above was written from.
    fn ramp_rgb() -> Raster {
        let mut data = Vec::with_capacity(4 * 3 * 3);
        for y in 0..3u32 {
            for x in 0..4u32 {
                data.push(((x * 61 + y * 13) % 256) as u8);
                data.push(((x * 97 + y * 151) % 256) as u8);
                data.push(((x * 29 + y * 211) % 256) as u8);
            }
        }
        Raster::new(4, 3, PixelFormat::Rgb8, data).unwrap()
    }

    /// The same ramp with a fourth, independent alpha channel.
    fn ramp_rgba() -> Raster {
        let mut data = Vec::with_capacity(4 * 3 * 4);
        for y in 0..3u32 {
            for x in 0..4u32 {
                data.push(((x * 61 + y * 13) % 256) as u8);
                data.push(((x * 97 + y * 151) % 256) as u8);
                data.push(((x * 29 + y * 211) % 256) as u8);
                data.push(((x * 85 + y * 40) % 256) as u8);
            }
        }
        Raster::new(4, 3, PixelFormat::Rgba8, data).unwrap()
    }

    /// The twelve RGB triples `vips getpoint` prints for every fixture
    /// that carries the lossless ramp.
    const RAMP_PIXELS: [[u8; 3]; 12] = [
        [0, 0, 0],
        [61, 97, 29],
        [122, 194, 58],
        [183, 35, 87],
        [13, 151, 211],
        [74, 248, 240],
        [135, 89, 13],
        [196, 186, 42],
        [26, 46, 166],
        [87, 143, 195],
        [148, 240, 224],
        [209, 81, 253],
    ];

    /// Read every pixel of `raster` in raster order as bytes.
    fn pixels(raster: &Raster) -> Vec<Vec<u8>> {
        (0..raster.height())
            .flat_map(|y| (0..raster.width()).map(move |x| (x, y)))
            .map(|(x, y)| {
                raster
                    .getpoint(x, y)
                    .iter()
                    .map(|s| *s as u8)
                    .collect::<Vec<u8>>()
            })
            .collect()
    }

    /**
     * Tests that a lossless WebP written by vips decodes to exactly the
     * pixels vips reads back out of it, so the VP8L path is pinned to the
     * reference decoder rather than to itself. Works by decoding the
     * 100-byte `--lossless --keep none` capture and comparing every pixel
     * to the `vips getpoint` output recorded beside it.
     * Input: `LOSSLESS_RGB` -> Output: 4x3 `Rgb8`, pixels equal to
     * `RAMP_PIXELS`.
     */
    #[test]
    fn lossless_decode_matches_vips_getpoint() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGB, DecodeLimits::default())
            .expect("the vips lossless capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
    }

    /**
     * Tests that the lossy VP8 path is bit-exact against libwebp, not
     * merely close: VP8 reconstruction is integer-specified and
     * `image-webp` defaults to the same fancy (bilinear) chroma
     * upsampling libwebp does, so the two agree byte for byte. Works by
     * decoding the default-`Q` capture and comparing to the twelve
     * triples vips printed for the same file.
     * Input: `LOSSY_RGB` -> Output: 4x3 `Rgb8`, pixels exactly the vips
     * values, which differ from the original ramp because the encode was
     * lossy.
     */
    #[test]
    fn lossy_decode_is_bit_exact_against_libwebp() {
        let raster = decode_bytes_with_limits(&LOSSY_RGB, DecodeLimits::default())
            .expect("the vips lossy capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        let expected: [[u8; 3]; 12] = [
            [0, 14, 20],
            [56, 92, 82],
            [160, 153, 112],
            [112, 85, 28],
            [73, 125, 155],
            [166, 197, 214],
            [100, 90, 80],
            [201, 171, 147],
            [26, 66, 144],
            [115, 136, 204],
            [219, 204, 253],
            [160, 126, 165],
        ];
        assert_eq!(pixels(&raster), expected.map(Vec::from).to_vec());
        assert_ne!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
    }

    /**
     * Tests that an alpha channel survives the lossless decode as a
     * fourth band rather than being flattened, matching the `4 bands`
     * vips reports for the same file. Works by decoding the RGBA capture
     * and checking the alpha ramp.
     * Input: `LOSSLESS_RGBA` -> Output: 4x3 `Rgba8` whose bytes equal the
     * source raster's.
     */
    #[test]
    fn lossless_alpha_decodes_as_four_bands() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGBA, DecodeLimits::default())
            .expect("the vips lossless RGBA capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(raster.data(), ramp_rgba().data());
    }

    /**
     * Tests that the three metadata RIFF chunks are lifted onto the
     * raster under the same field names the JPEG loader uses, so a
     * caller reads `icc-profile-data` regardless of which container the
     * profile arrived in. Works by decoding the hand-built extended
     * container and comparing each blob to the exact chunk payload vips
     * reports the size of.
     * Input: `META_RGB` -> Output: `icc-profile-data` = 24 bytes
     * `0x10..0x27`, `exif-data` = the 10-byte little-endian TIFF header,
     * `xmp-data` = the 37-byte packet, and the pixels unchanged.
     */
    #[test]
    fn decode_attaches_icc_exif_and_xmp_from_the_riff_chunks() {
        let raster = decode_bytes_with_limits(&META_RGB, DecodeLimits::default())
            .expect("the extended container decodes");
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        let blob = |name: &str| match raster.get_field(name) {
            Some(MetadataValue::Blob(b)) => b.clone(),
            other => panic!("{name} should be a blob, got {other:?}"),
        };
        assert_eq!(
            blob("icc-profile-data"),
            (0x10u8..=0x27).collect::<Vec<u8>>()
        );
        assert_eq!(
            blob("exif-data"),
            b"II*\x00\x08\x00\x00\x00\x00\x00".to_vec()
        );
        assert_eq!(
            blob("xmp-data"),
            b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec()
        );
    }

    /**
     * Tests the multi-frame verdict: an animated WebP loads its first
     * frame and says how many there were, which is exactly what a
     * default `vips webpload` does (`n` defaults to 1). The toilet-roll
     * load lives in issue #569 behind the page model, and refusing the
     * file outright would be a regression, since frame 0 already decoded
     * before this lane. Works by decoding a three-frame capture and
     * checking both the geometry and `n-pages`.
     * Input: `ANIM3` -> Output: 4x3 (not 4x9), pixels equal to frame 0,
     * `get_n_pages() == 3`.
     */
    #[test]
    fn animated_webp_loads_frame_zero_and_reports_the_page_count() {
        let raster = decode_bytes_with_limits(&ANIM3, DecodeLimits::default())
            .expect("the animation decodes");
        assert_eq!(
            (raster.width(), raster.height()),
            (4, 3),
            "one frame, not the 4x9 toilet roll"
        );
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        assert_eq!(raster.get_n_pages(), 3);
    }

    /**
     * Tests the property the lossless-only encoder buys: because there
     * is no quantisation step anywhere in the pipeline, decoding what
     * `encode_webp` wrote returns the input bytes exactly, for both the
     * opaque and the alpha carrier. Works by encoding two rasters at the
     * default options and decoding the result back.
     * Input: the 4x3 `Rgb8` and `Rgba8` ramps -> Output: identical
     * dimensions, identical pixel format, byte-identical data.
     */
    #[test]
    fn lossless_encode_decode_is_the_identity() {
        for original in [ramp_rgb(), ramp_rgba()] {
            let bytes = original
                .encode_webp(SaveOptions::default())
                .expect("the lossless encoder accepts an 8-bit raster");
            let back = decode_bytes_with_limits(&bytes, DecodeLimits::default())
                .expect("our own bytes decode");
            assert_eq!((back.width(), back.height()), (4, 3));
            assert_eq!(back.format(), original.format());
            assert_eq!(back.data(), original.data(), "lossless is exact");
        }
    }

    /**
     * Tests that a 16-bit raster is refused with a message naming the
     * remedy rather than silently narrowed. vips narrows instead: a
     * `ushort` image saved through `webpsave` comes back right-shifted
     * by 8 (measured: 255 -> 0, 256 -> 1, 511 -> 1, 512 -> 2, 65535 ->
     * 255), which throws away the low byte without telling anyone.
     * libviprs makes the caller pick the narrowing. Works by encoding an
     * `Rgb16` raster and matching the typed error.
     * Input: 4x3 `Rgb16` -> Output: `EncodeError::Encode` whose message
     * names the format and says to cast.
     */
    #[test]
    fn sixteen_bit_is_refused_rather_than_narrowed() {
        let wide = Raster::zeroed(4, 3, PixelFormat::Rgb16).unwrap();
        let err = wide
            .encode_webp(SaveOptions::default())
            .expect_err("WebP has no 16-bit sample spelling");
        let msg = err.to_string();
        assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
        assert!(msg.contains("Rgb16"), "{msg}");
        assert!(msg.contains("cast"), "{msg}");
    }

    /**
     * Tests that a one-band raster is promoted to three bands on the
     * round trip, because WebP stores no greyscale: `vips webpsave` on a
     * `b-w` uchar image also reports `3 bands, srgb` when it is loaded
     * back. Works by encoding a `Gray8` ramp and checking the decoded
     * bands repeat the luminance.
     * Input: 4x3 `Gray8` -> Output: 4x3 `Rgb8` whose three bands each
     * equal the source luminance.
     */
    #[test]
    fn grey_promotes_to_rgb_on_the_round_trip_as_vips_does() {
        let data: Vec<u8> = RAMP_PIXELS.iter().map(|p| p[0]).collect();
        let grey = Raster::new(4, 3, PixelFormat::Gray8, data.clone()).unwrap();
        let bytes = grey
            .encode_webp(SaveOptions::default())
            .expect("a one-band raster encodes");
        let back = decode_bytes_with_limits(&bytes, DecodeLimits::default()).expect("it decodes");
        assert_eq!(back.format(), PixelFormat::Rgb8);
        let expected: Vec<Vec<u8>> = data.iter().map(|v| vec![*v, *v, *v]).collect();
        assert_eq!(pixels(&back), expected);
    }

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
