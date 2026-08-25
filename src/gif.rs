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
    use crate::source::decode_bytes;

    /// One frame of a hand-assembled GIF fixture.
    struct Frame {
        /// Frame origin inside the logical screen.
        left: u16,
        /// Frame origin inside the logical screen.
        top: u16,
        /// Frame extent.
        width: u16,
        /// Frame extent.
        height: u16,
        /// Palette indices, row-major, `width * height` of them, in
        /// progressive order even when `interlaced` is set.
        indices: Vec<u8>,
        /// The graphic control extension's transparent index, if any.
        transparent: Option<u8>,
        /// Whether the frame is stored in the four-pass interlaced order.
        interlaced: bool,
        /// Frame delay, in centiseconds, as it goes on the wire.
        delay_cs: u16,
    }

    impl Frame {
        /// A full-canvas opaque frame.
        fn full(width: u16, height: u16, indices: Vec<u8>) -> Self {
            Self {
                left: 0,
                top: 0,
                width,
                height,
                indices,
                transparent: None,
                interlaced: false,
                delay_cs: 0,
            }
        }
    }

    /// Assemble a GIF89a from a global palette and a frame list.
    ///
    /// Everything the decode tests turn on -- the transparent index, the
    /// interlace bit, the frame rectangle, the NETSCAPE loop count -- is
    /// written here explicitly, so the fixtures do not depend on
    /// [`Raster::encode_gif`] and cannot drift with it.
    fn fixture(
        screen: (u16, u16),
        palette: &[[u8; 3]],
        background: u8,
        loop_ext: Option<u16>,
        frames: &[Frame],
    ) -> Vec<u8> {
        let table_size = flag_size(palette.len());
        let entries = 2usize << table_size;
        let mut out = Vec::from(*b"GIF89a");
        out.extend_from_slice(&screen.0.to_le_bytes());
        out.extend_from_slice(&screen.1.to_le_bytes());
        out.push(0x80 | table_size);
        out.push(background);
        out.push(0);
        for i in 0..entries {
            out.extend_from_slice(palette.get(i).unwrap_or(&[0, 0, 0]));
        }
        if let Some(count) = loop_ext {
            out.extend_from_slice(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01");
            out.extend_from_slice(&count.to_le_bytes());
            out.push(0);
        }
        for frame in frames {
            let mut flags = 0u8;
            if frame.transparent.is_some() {
                flags |= 1;
            }
            flags |= 1 << 2; // disposal method "keep", as cgif writes
            out.extend_from_slice(&[0x21, 0xF9, 4, flags]);
            out.extend_from_slice(&frame.delay_cs.to_le_bytes());
            out.push(frame.transparent.unwrap_or(0));
            out.push(0);
            out.push(0x2C);
            out.extend_from_slice(&frame.left.to_le_bytes());
            out.extend_from_slice(&frame.top.to_le_bytes());
            out.extend_from_slice(&frame.width.to_le_bytes());
            out.extend_from_slice(&frame.height.to_le_bytes());
            out.push(if frame.interlaced { 0x40 } else { 0 });
            let min_code_size = (table_size + 1).max(2);
            out.push(min_code_size);
            let stored = if frame.interlaced {
                let mut rows = Vec::with_capacity(frame.indices.len());
                for row in interlace_rows(u32::from(frame.height)) {
                    let start = row as usize * frame.width as usize;
                    rows.extend_from_slice(&frame.indices[start..start + frame.width as usize]);
                }
                rows
            } else {
                frame.indices.clone()
            };
            for chunk in lzw_literal(&stored, min_code_size).chunks(255) {
                out.push(chunk.len() as u8);
                out.extend_from_slice(chunk);
            }
            out.push(0);
        }
        out.push(0x3B);
        out
    }

    /// The GIF colour-table size flag for `n` entries: the table on the wire
    /// holds `2 << flag` colours.
    fn flag_size(n: usize) -> u8 {
        (n.clamp(2, 256).next_power_of_two().trailing_zeros() - 1) as u8
    }

    /// LZW-encode `indices` as literals only: a clear code, one code per
    /// pixel, then end-of-information.
    ///
    /// No dictionary matches are emitted, but the code width still grows on
    /// the schedule a decoder tracks, so the stream is ordinary GIF LZW and
    /// any conformant decoder reads it.
    fn lzw_literal(indices: &[u8], min_code_size: u8) -> Vec<u8> {
        let clear = 1u16 << min_code_size;
        let mut width = u32::from(min_code_size) + 1;
        let mut next_code = clear + 2;
        let mut out = Vec::new();
        let mut acc = 0u32;
        let mut bits = 0u32;
        let mut emit = |code: u16, width: u32, out: &mut Vec<u8>| {
            acc |= u32::from(code) << bits;
            bits += width;
            while bits >= 8 {
                out.push((acc & 0xFF) as u8);
                acc >>= 8;
                bits -= 8;
            }
        };
        emit(clear, width, &mut out);
        for &index in indices {
            emit(u16::from(index), width, &mut out);
            next_code += 1;
            if u32::from(next_code) > (1u32 << width) && width < 12 {
                width += 1;
            }
        }
        emit(clear + 1, width, &mut out);
        if bits > 0 {
            out.push((acc & 0xFF) as u8);
        }
        out
    }

    /// The rows of a `height`-tall frame in GIF's four-pass interlaced
    /// order: rows 0, 8, 16, ...; then 4, 12, ...; then 2, 6, ...; then
    /// every odd row.
    fn interlace_rows(height: u32) -> Vec<u32> {
        let mut rows = Vec::with_capacity(height as usize);
        for (start, step) in [(0u32, 8u32), (4, 8), (2, 4), (1, 2)] {
            let mut row = start;
            while row < height {
                rows.push(row);
                row += step;
            }
        }
        rows
    }

    /// A four-colour palette and a 4x4 frame that uses all of it.
    fn opaque_fixture() -> Vec<u8> {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let indices: Vec<u8> = (0..16u8).map(|i| i % 4).collect();
        fixture((4, 4), &palette, 0, None, &[Frame::full(4, 4, indices)])
    }

    /**
     * Tests that a GIF with no transparent index anywhere loads as a
     * three-band raster, as `vips gifload` does. Works by decoding a
     * hand-assembled four-colour GIF through the shared sniff route and
     * reading the pixel format back. `nsgifload.c:271` sizes the image
     * `has_transparency ? 4 : 3`, and `:522-534` drops the alpha byte per
     * pixel when it is 3; measured on the reference `cramps.gif`, which
     * `vipsheader` reports as `3 bands`.
     * Input: a 4x4 GIF whose graphic control extension clears the
     * transparency flag -> Output: `PixelFormat::Rgb8` and the four palette
     * colours, byte for byte.
     */
    #[test]
    fn an_opaque_gif_loads_as_three_bands() {
        let raster = decode_bytes(&opaque_fixture()).expect("the fixture is a valid GIF");
        assert_eq!(raster.width(), 4);
        assert_eq!(raster.height(), 4);
        assert_eq!(
            raster.format(),
            PixelFormat::Rgb8,
            "vips gifload emits 3 bands when no frame declares transparency"
        );
        assert_eq!(
            &raster.data()[..12],
            &[0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0],
            "the palette must be expanded exactly; GIF's LZW is lossless"
        );
    }

    /**
     * Tests that a declared transparent index keeps the fourth band and
     * expands to transparent black, so a transparent GIF is not silently
     * flattened. Works by decoding a fixture whose graphic control
     * extension names index 0 as transparent and checking the first pixel.
     * `nsgifload.c:431-432` sets `has_transparency` from any frame's
     * transparency flag.
     * Input: a 4x4 GIF with transparent index 0 -> Output:
     * `PixelFormat::Rgba8`, pixel 0 fully transparent, pixel 1 opaque red.
     */
    #[test]
    fn a_transparent_index_keeps_four_bands() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let indices: Vec<u8> = (0..16u8).map(|i| i % 4).collect();
        let mut frame = Frame::full(4, 4, indices);
        frame.transparent = Some(0);
        let bytes = fixture((4, 4), &palette, 0, None, &[frame]);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(&raster.data()[..8], &[0, 0, 0, 0, 255, 0, 0, 255]);
    }

    /**
     * Tests that the canvas around a frame smaller than the logical screen
     * comes back transparent black rather than the background colour, which
     * is what libnsgif renders and what `vips getpoint` reports. Works by
     * building an 8x8 screen with the background index pointing at blue and
     * a 4x4 red frame inset at (2,2), then reading a corner pixel.
     * Measured: `vips gifload` on exactly this file prints `0 0 0` at (0,0)
     * while `vipsheader` reports `background: 0 0 255`.
     * Input: an 8x8 GIF with an inset opaque frame -> Output: corner pixels
     * black, the inset red.
     */
    #[test]
    fn the_canvas_outside_frame_zero_is_transparent_black_not_the_background() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let frame = Frame {
            left: 2,
            top: 2,
            width: 4,
            height: 4,
            indices: vec![1; 16],
            transparent: None,
            interlaced: false,
            delay_cs: 0,
        };
        let bytes = fixture((8, 8), &palette, 2, None, &[frame]);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(
            &raster.data()[..3],
            &[0, 0, 0],
            "the corner is not the background colour"
        );
        let inset = (2 * 8 + 2) * 3;
        assert_eq!(&raster.data()[inset..inset + 3], &[255, 0, 0]);
    }

    /**
     * Tests that an interlaced GIF is de-interlaced on load, so the row
     * order on the wire is invisible to callers. Works by writing the same
     * eight rows twice, once progressive and once in GIF's four-pass order
     * with the interlace bit set, and requiring the two decodes to be
     * byte-identical.
     * Input: two 8x8 GIFs differing only in row order and the interlace bit
     * -> Output: identical rasters.
     */
    #[test]
    fn an_interlaced_gif_decodes_to_the_same_rows_as_a_progressive_one() {
        let palette: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 30, 255 - i * 30, i * 17]).collect();
        let indices: Vec<u8> = (0..64u8).map(|i| i / 8).collect();
        let progressive = fixture(
            (8, 8),
            &palette,
            0,
            None,
            &[Frame::full(8, 8, indices.clone())],
        );
        let mut frame = Frame::full(8, 8, indices);
        frame.interlaced = true;
        let interlaced = fixture((8, 8), &palette, 0, None, &[frame]);
        let a = decode_bytes(&progressive).expect("progressive fixture decodes");
        let b = decode_bytes(&interlaced).expect("interlaced fixture decodes");
        assert_eq!(a.data(), b.data(), "interlaced rows must be reassembled");
    }

    /**
     * Tests that an animated GIF loads its first frame and still reports how
     * many pages the file holds, so a caller can tell an animation from a
     * still without a page model. `vips gifload` defaults to `page=0, n=1`
     * and sets `n-pages` from the whole file
     * (`nsgifload.c:284-285`). Works by assembling a three-frame GIF and
     * reading the first frame's pixels and the attached field.
     * Input: a 4x4 GIF with three frames -> Output: frame zero's pixels and
     * `n-pages == 3`.
     */
    #[test]
    fn an_animation_loads_frame_zero_and_records_the_page_count() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let frames: Vec<Frame> = (0..3u8)
            .map(|f| Frame::full(4, 4, vec![f + 1; 16]))
            .collect();
        let bytes = fixture((4, 4), &palette, 0, Some(0), &frames);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(&raster.data()[..3], &[255, 0, 0], "frame zero is index 1");
        assert_eq!(
            raster.get_int("n-pages"),
            Some(3),
            "n-pages counts every frame in the file, not the ones loaded"
        );
    }

    /**
     * Tests that the GIF encoder produces bytes the decoder reads back
     * unchanged when the raster already fits in the palette. GIF's LZW is
     * exactly lossless, so this is a byte-for-byte equality and not a
     * tolerance band; the `lossy_decoder: true` tag on `cogs.gif` in the
     * existing oracle capture is wrong and would let a real palette or
     * expansion bug through. Measured against vips: `gifsave` of a
     * sixteen-colour source reloads with `max_abs_diff 0`.
     * Input: a 16x16 Rgb8 raster with four distinct colours -> Output: a GIF
     * whose decode equals the input pixel for pixel.
     */
    #[test]
    fn encode_then_decode_is_exactly_lossless_within_the_palette() {
        let colours = [[0u8, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let mut data = Vec::with_capacity(16 * 16 * 3);
        for i in 0..256usize {
            data.extend_from_slice(&colours[i % 4]);
        }
        let im = Raster::new(16, 16, PixelFormat::Rgb8, data.clone()).unwrap();
        let bytes = im
            .encode_gif(SaveOptions::default())
            .expect("a four-colour raster is well inside a GIF palette");
        let back = decode_bytes(&bytes).expect("libviprs must read back what it writes");
        assert_eq!(back.width(), 16);
        assert_eq!(back.height(), 16);
        let rgb: Vec<u8> = match back.format() {
            PixelFormat::Rgb8 => back.data().to_vec(),
            PixelFormat::Rgba8 => back
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| p[..3].to_vec())
                .collect(),
            other => panic!("gifload emits 3 or 4 bands, got {other:?}"),
        };
        assert_eq!(
            rgb, data,
            "LZW is lossless; a palette that fits must round-trip"
        );
    }

    /**
     * Tests that a raster whose alpha straddles 128 is thresholded the way
     * `cgifsave.c:538-548` does: `p[3] >= 128` becomes fully opaque with its
     * colour intact, anything below becomes fully transparent black.
     * Measured on vips 8.18.4 with a 32-wide alpha ramp: columns 0..15
     * (alpha 0..120) reload as `0 0 0 0` and columns 16..31 (alpha 128..248)
     * as `200 100 50 255`.
     * Input: a 32x1 Rgba8 ramp of alpha `x * 8` over a constant colour ->
     * Output: a hard cut at alpha 128 with no partial values anywhere.
     */
    #[test]
    fn alpha_is_thresholded_at_128_on_save() {
        let mut data = Vec::with_capacity(32 * 4);
        for x in 0..32u32 {
            data.extend_from_slice(&[200, 100, 50, (x * 8).min(255) as u8]);
        }
        let im = Raster::new(32, 1, PixelFormat::Rgba8, data).unwrap();
        let bytes = im.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        assert_eq!(back.format(), PixelFormat::Rgba8);
        for (x, pixel) in back.data().as_chunks::<4>().0.iter().enumerate() {
            let expected: [u8; 4] = if x >= 16 {
                [200, 100, 50, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(*pixel, expected, "alpha {} at x={x}", x * 8);
        }
    }

    /**
     * Tests that the interlace option reaches the wire, so a caller asking
     * for a progressive GIF gets one. Works by encoding the same raster
     * twice and requiring the image descriptor's interlace bit to differ
     * while both decode to the same pixels. Measured: `vips gifsave
     * --interlace` sets bit 6 of the image descriptor's packed byte and
     * reorders the stored rows, and vips reloads both files identically.
     * Input: an 8x8 Rgb8 raster -> Output: two GIFs whose interlace bit
     * differs and whose decodes agree.
     */
    #[test]
    fn the_interlace_option_reaches_the_image_descriptor() {
        let data: Vec<u8> = (0..64u32)
            .flat_map(|i| [(i * 4) as u8, 255 - (i * 3) as u8, (i * 7) as u8])
            .collect();
        let im = Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap();
        let plain = im.encode_gif(SaveOptions::default()).expect("encodes");
        let woven = im
            .encode_gif(SaveOptions {
                interlaced: true,
                ..Default::default()
            })
            .expect("encodes");
        assert!(!image_descriptor_flags(&plain).is_some_and(|f| f & 0x40 != 0));
        assert!(
            image_descriptor_flags(&woven).is_some_and(|f| f & 0x40 != 0),
            "interlace must set bit 6 of the image descriptor"
        );
        let a = decode_bytes(&plain).expect("decodes");
        let b = decode_bytes(&woven).expect("decodes");
        assert_eq!(
            a.data(),
            b.data(),
            "interlacing changes storage, not pixels"
        );
    }

    /// The packed byte of the first image descriptor in `bytes`, skipping
    /// the header, the global colour table, and any extension blocks.
    fn image_descriptor_flags(bytes: &[u8]) -> Option<u8> {
        let mut p = 13usize;
        if bytes.get(10)? & 0x80 != 0 {
            p += 3 * (2usize << (bytes[10] & 7));
        }
        loop {
            match *bytes.get(p)? {
                0x21 => {
                    p += 2;
                    while *bytes.get(p)? != 0 {
                        p += 1 + bytes[p] as usize;
                    }
                    p += 1;
                }
                0x2C => return bytes.get(p + 9).copied(),
                _ => return None,
            }
        }
    }

    /**
     * Tests that the GIF encoder is reachable at every option combination
     * and reports its errors through the shared spine rather than panicking.
     * Works by encoding a small RGB raster at the default options,
     * interlaced, and with dithering off.
     * Input: 8x8 Rgb8 raster -> Output: a non-empty GIF89a buffer from every
     * call.
     */
    #[test]
    fn encode_gif_answers_every_option_combination() {
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
            let bytes = im
                .encode_gif(options)
                .expect("every option combination encodes");
            assert_eq!(&bytes[..6], b"GIF89a", "{options:?} did not produce a GIF");
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
