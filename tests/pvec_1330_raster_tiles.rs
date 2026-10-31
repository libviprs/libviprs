//! Issue #1330 (RL1.1): the tiles libviprs hands PVec pass its sniffers by construction.
//!
//! The legal table, the 1-bit writer, and the refusals by name. The header reader is in
//! `common/tile_headers.rs`, not in the crate.

#[path = "common/tile_headers.rs"]
mod tile_headers;

use std::error::Error;

use libviprs::raster_pvec::pyramid::{RasterChannels, RasterCodec};
use libviprs::raster_pvec::tiles::{RasterTileError, render_markdown, tile_tags};
use libviprs::{PixelFormat, Raster, TileFormat};

const DOC: &str = include_str!("../docs/pvec-raster-tiles.md");

fn gradient(w: u32, h: u32, format: PixelFormat) -> Raster {
    let bpp = format.bytes_per_pixel();
    let mut data = Vec::with_capacity(w as usize * h as usize * bpp);
    for y in 0..h {
        for x in 0..w {
            for c in 0..bpp {
                data.push(((x * 7 + y * 13 + c as u32 * 29) % 251) as u8);
            }
        }
    }
    Raster::new(w, h, format, data).unwrap()
}

/// One row of the legal table, encoded for real.
fn encode(format: TileFormat, raster: &Raster) -> Vec<u8> {
    match format {
        TileFormat::Png => raster.encode_png(6).unwrap(),
        TileFormat::Jpeg { quality } => raster.encode_jpeg(quality).unwrap(),
        TileFormat::Webp => raster
            .encode_webp(libviprs::webp::SaveOptions::default())
            .unwrap(),
        _ => unreachable!(),
    }
}

#[test]
fn every_legal_row_round_trips_and_the_header_says_what_the_table_claims() {
    let rows = [
        (TileFormat::Png, PixelFormat::Gray8, RasterCodec::Png, RasterChannels::Gray),
        (TileFormat::Png, PixelFormat::Rgb8, RasterCodec::Png, RasterChannels::Rgb),
        (TileFormat::Png, PixelFormat::Rgba8, RasterCodec::Png, RasterChannels::Rgba),
        (TileFormat::Jpeg { quality: 85 }, PixelFormat::Gray8, RasterCodec::Jpeg, RasterChannels::Gray),
        (TileFormat::Jpeg { quality: 85 }, PixelFormat::Rgb8, RasterCodec::Jpeg, RasterChannels::Rgb),
        (TileFormat::Webp, PixelFormat::Rgb8, RasterCodec::Webp, RasterChannels::Rgb),
        (TileFormat::Webp, PixelFormat::Rgba8, RasterCodec::Webp, RasterChannels::Rgba),
    ];
    for (format, pixel, codec, channels) in rows {
        assert_eq!(
            tile_tags(format, pixel, 8).unwrap(),
            (codec, channels),
            "{format:?} {pixel:?}"
        );
        let bytes = encode(format, &gradient(256, 256, pixel));
        let n = match channels {
            RasterChannels::Gray => 1,
            RasterChannels::Rgb => 3,
            _ => 4,
        };
        match codec {
            RasterCodec::Png => {
                let h = tile_headers::png(&bytes);
                assert_eq!((h.width, h.height, h.bit_depth, h.channels()), (256, 256, 8, n));
            }
            RasterCodec::Jpeg => {
                let h = tile_headers::jpeg(&bytes);
                assert_eq!((h.width, h.height, h.precision, h.components), (256, 256, 8, n));
            }
            _ => {
                let h = tile_headers::webp(&bytes);
                assert_eq!((h.width, h.height), (256, 256));
            }
        }
    }
}

#[test]
fn one_bit_gray_png_is_a_row_of_the_table() {
    assert_eq!(
        tile_tags(TileFormat::Png, PixelFormat::Gray8, 1).unwrap(),
        (RasterCodec::Png, RasterChannels::Gray)
    );
}

#[test]
fn the_one_bit_writer_packs_a_known_pattern() {
    // 8 x 8: row y has pixel x set when x >= y (a staircase), so row y packs to 0xFF >> y.
    let mut data = Vec::new();
    for y in 0..8u8 {
        for x in 0..8u8 {
            data.push(if x >= y { 255 } else { 0 });
        }
    }
    let raster = Raster::new(8, 8, PixelFormat::Gray8, data).unwrap();
    let bytes = raster.encode_png_bilevel(128).unwrap();
    let h = tile_headers::png(&bytes);
    assert_eq!((h.width, h.height, h.bit_depth, h.colour_type, h.interlace), (8, 8, 1, 0, 0));
    assert_eq!(h.chunks, vec![*b"IHDR", *b"IDAT", *b"IEND"]);
    assert!(h.ends_at_iend);

    // Inflate the IDAT by hand: filter byte 0, then one packed byte a row.
    let idat = {
        let at = 8 + 25; // signature, IHDR chunk (12 + 13)
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        &bytes[at + 8..at + 8 + len]
    };
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(idat), &mut raw).unwrap();
    let expected: Vec<u8> = (0..8u8).flat_map(|y| [0u8, 0xFFu8 >> y]).collect();
    assert_eq!(raw, expected);

    let back = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .unwrap()
        .to_luma8();
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(back.get_pixel(x, y).0[0], if x >= y { 255 } else { 0 }, "({x},{y})");
        }
    }
}

#[test]
fn the_threshold_is_inclusive_and_a_non_gray_source_is_refused() {
    let raster = Raster::new(8, 1, PixelFormat::Gray8, vec![0, 99, 100, 101, 127, 128, 254, 255]).unwrap();
    let bytes = raster.encode_png_bilevel(100).unwrap();
    let back = image::load_from_memory(&bytes).unwrap().to_luma8();
    let got: Vec<u8> = back.pixels().map(|p| p.0[0]).collect();
    assert_eq!(got, vec![0, 0, 255, 255, 255, 255, 255, 255]);

    let rgb = gradient(8, 8, PixelFormat::Rgb8);
    assert!(rgb.encode_png_bilevel(128).is_err(), "only Gray8 has a bilevel form");
}

#[test]
fn a_width_that_is_not_a_multiple_of_eight_pads_each_row() {
    let raster = Raster::new(10, 3, PixelFormat::Gray8, vec![255; 30]).unwrap();
    let bytes = raster.encode_png_bilevel(1).unwrap();
    let back = image::load_from_memory(&bytes).unwrap().to_luma8();
    assert_eq!(back.dimensions(), (10, 3));
    assert!(back.pixels().all(|p| p.0[0] == 255));
}

#[test]
fn every_refusal_is_named_and_keeps_its_source_chain() {
    let jpeg = TileFormat::Jpeg { quality: 85 };
    assert!(matches!(tile_tags(TileFormat::Raw, PixelFormat::Gray8, 8), Err(RasterTileError::RawTiles)));
    assert!(matches!(tile_tags(jpeg, PixelFormat::Rgba8, 8), Err(RasterTileError::JpegAlpha)));
    assert!(matches!(tile_tags(jpeg, PixelFormat::Gray8, 1), Err(RasterTileError::BitDepth { .. })));
    assert!(matches!(tile_tags(TileFormat::Webp, PixelFormat::Gray8, 1), Err(RasterTileError::BitDepth { .. })));
    assert!(matches!(tile_tags(TileFormat::Png, PixelFormat::Rgb8, 1), Err(RasterTileError::BitDepth { .. })));
    assert!(matches!(tile_tags(TileFormat::Png, PixelFormat::Gray8, 4), Err(RasterTileError::BitDepth { .. })));
    for sixteen in [PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16] {
        match tile_tags(TileFormat::Png, sixteen, 8) {
            Err(RasterTileError::SixteenBit(f)) => assert_eq!(f, sixteen),
            other => panic!("{sixteen:?}: {other:?}"),
        }
    }
    assert!(matches!(
        tile_tags(TileFormat::Png, PixelFormat::RgbaF32, 8),
        Err(RasterTileError::PixelFormat(PixelFormat::RgbaF32))
    ));
    // Each is a typed error with a message, and a leaf (no hidden cause to lose).
    let e = tile_tags(TileFormat::Raw, PixelFormat::Gray8, 8).unwrap_err();
    assert!(!e.to_string().is_empty());
    assert!(e.source().is_none());
}

#[test]
fn a_tile_edge_outside_the_three_is_refused() {
    use libviprs::raster_pvec::tiles::check_tile_edge;
    for ok in [256, 512, 1024] {
        assert!(check_tile_edge(ok).is_ok());
    }
    for bad in [0, 128, 255, 257, 2048] {
        assert!(matches!(check_tile_edge(bad), Err(RasterTileError::TileEdge(e)) if e == bad));
    }
}

#[test]
fn interlace_and_palette_never_come_out_of_the_pyramid_path() {
    use libviprs::raster_pvec::pyramid::{RasterPlacement, RasterPyramidOptions, produce_raster_pyramid};
    use libviprs::{Layout, PyramidPlanner};

    // The control: the two encoders exist, and the reader sees what they do.
    let raster = gradient(256, 256, PixelFormat::Rgb8);
    assert_eq!(tile_headers::png(&raster.encode_png_interlaced().unwrap()).interlace, 1);
    assert_eq!(tile_headers::png(&raster.encode_png_palette(16).unwrap()).colour_type, 3);

    for pixel in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let r = gradient(600, 300, pixel);
        let plan = PyramidPlanner::new(600, 300, 256, 0, Layout::Google).unwrap().plan();
        let opts = RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 300.0]));
        let p = produce_raster_pyramid(&r, &plan, &opts).unwrap();
        assert!(p.tiles().count() > 0);
        for (l, c, row, bytes) in p.tiles() {
            let h = tile_headers::png(bytes);
            assert_eq!(h.interlace, 0, "{pixel:?} {l}/{c}/{row}");
            assert_ne!(h.colour_type, 3, "{pixel:?} {l}/{c}/{row}");
            assert_eq!((h.width, h.height, h.bit_depth), (256, 256, 8));
        }
    }
}

#[test]
fn the_doc_is_the_rendered_table_byte_for_byte() {
    if std::env::var_os("PVEC_RASTER_TILES_BLESS").is_some() {
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/docs/pvec-raster-tiles.md"),
            render_markdown(),
        )
        .unwrap();
        return;
    }
    assert_eq!(
        DOC,
        render_markdown(),
        "docs/pvec-raster-tiles.md is generated: regenerate it with \
         `PVEC_RASTER_TILES_BLESS=1 cargo test --test pvec_1330_raster_tiles`"
    );
}

#[test]
fn a_hand_edit_of_the_doc_is_caught() {
    assert_ne!(DOC.replacen("PNG", "PNQ", 1), render_markdown());
}
