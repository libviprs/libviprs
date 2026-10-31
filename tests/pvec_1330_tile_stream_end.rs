//! Issue #1330: where a tile ends, for every variant libviprs writes. PVec's sniffers refuse a
//! tile whose stream does not end at its last byte (`IEND` last, `EOI` last, RIFF size equal),
//! a colour chunk PNG decoders disagree about, and a JPEG frame header past 64 KiB.

#[path = "common/tile_headers.rs"]
mod tile_headers;

use libviprs::{PixelFormat, Raster};

fn noise(w: u32, h: u32, format: PixelFormat) -> Raster {
    let bpp = format.bytes_per_pixel();
    let mut s = 0x1234_5678u32;
    let data = (0..w as usize * h as usize * bpp)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect();
    Raster::new(w, h, format, data).unwrap()
}

const COLOUR_CHUNKS: [&[u8; 4]; 4] = [b"gAMA", b"cHRM", b"sRGB", b"iCCP"];

fn assert_clean_png(bytes: &[u8], what: &str) {
    let h = tile_headers::png(bytes);
    assert!(h.ends_at_iend, "{what}: IEND is the last chunk and the last byte");
    assert_eq!(h.chunks.last(), Some(b"IEND"), "{what}");
    for c in COLOUR_CHUNKS {
        assert!(!h.has(c), "{what}: {}", String::from_utf8_lossy(c));
    }
    assert_eq!(h.interlace, 0, "{what}");
}

#[test]
fn png_ends_at_iend_with_no_colour_chunk_on_both_writers() {
    for pixel in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let r = noise(256, 256, pixel);
        // The `image` path.
        assert_clean_png(&r.encode_png(6).unwrap(), &format!("encode_png {pixel:?}"));
        // The sink path the pyramid uses.
        assert_clean_png(&libviprs::sink::encode_png(&r).unwrap(), &format!("sink {pixel:?}"));
    }
    // The hand-rolled writer.
    let g = noise(256, 256, PixelFormat::Gray8);
    let one = g.encode_png_bilevel(128).unwrap();
    assert_clean_png(&one, "bilevel");
    let h = tile_headers::png(&one);
    assert_eq!((h.bit_depth, h.colour_type), (1, 0));
}

#[test]
fn jpeg_frame_header_is_near_the_start_and_eoi_is_last() {
    for (pixel, nf) in [(PixelFormat::Gray8, 1), (PixelFormat::Rgb8, 3)] {
        for quality in [50u8, 85, 95] {
            let bytes = noise(256, 256, pixel).encode_jpeg(quality).unwrap();
            let h = tile_headers::jpeg(&bytes);
            assert_eq!(h.sof_marker, 0xC0, "baseline");
            assert_eq!((h.width, h.height, h.precision, h.components), (256, 256, 8, nf));
            assert!(h.sof_offset < 64 * 1024, "SOF at {}", h.sof_offset);
            assert!(h.ends_in_eoi, "{pixel:?} q{quality}");
        }
    }
}

#[test]
fn the_jpeg_sof_offset_is_pinned() {
    // APP0, then the quantisation tables, then SOF: a different writer layout moves this and the
    // test says so, which matters because the sniffer's walk is bounded.
    let gray = noise(256, 256, PixelFormat::Gray8).encode_jpeg(85).unwrap();
    let rgb = noise(256, 256, PixelFormat::Rgb8).encode_jpeg(85).unwrap();
    let (g, c) = (tile_headers::jpeg(&gray).sof_offset, tile_headers::jpeg(&rgb).sof_offset);
    assert!(g > 0 && c > g, "gray {g}, rgb {c}");
    assert_eq!((g, c), (PINNED_GRAY_SOF, PINNED_RGB_SOF));
}

// Measured on the writer as it stands (the issue guessed near byte 600; it is 89 and 158).
const PINNED_GRAY_SOF: usize = 89;
const PINNED_RGB_SOF: usize = 158;

#[test]
fn libviprs_writes_no_progressive_jpeg() {
    // The issue's seed list says "baseline and progressive", but the hand-rolled writer has no
    // progressive mode (its module docs say so), so SOF2 is a case the sniffer accepts and this
    // crate never produces. Pin that, so the day a progressive writer lands this fails and the
    // seed corpus gets its variant.
    for pixel in [PixelFormat::Gray8, PixelFormat::Rgb8] {
        let bytes = noise(256, 256, pixel).encode_jpeg(85).unwrap();
        assert_ne!(tile_headers::jpeg(&bytes).sof_marker, 0xC2);
    }
}

#[test]
fn webp_riff_size_equals_the_length_and_the_chunk_is_vp8l() {
    for pixel in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let bytes = noise(256, 256, pixel)
            .encode_webp(libviprs::webp::SaveOptions::default())
            .unwrap();
        let h = tile_headers::webp(&bytes);
        assert!(h.riff_size_matches, "{pixel:?}");
        assert_eq!(&h.first_chunk, b"VP8L");
        assert_eq!((h.width, h.height), (256, 256));
    }
}
