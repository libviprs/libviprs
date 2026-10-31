//! Issue #1328 (R2.4): a raster pyramid cut for a PVec raster stream.
//!
//! This is the producer half that needs nothing from PVec: the plan rules, the tiles and the
//! placement fields. The container round trip (every tile digest-equal through pvec's reader) is
//! `pvec_1328_raster_container.rs`, behind the feature that pulls pvec's writer in.

use std::error::Error;

use libviprs::raster_pvec::placement::{PlacementError, SourceCalibration};
use libviprs::raster_pvec::pyramid::{
    RasterChannels, RasterCodec, RasterPlacement, RasterPyramidError, RasterPyramidOptions,
    produce_raster_pyramid,
};
use libviprs::{
    BlankTileStrategy, GeoTransform, Layout, PixelFormat, PixelOrigin, PyramidPlan, PyramidPlanner,
    Raster, TileFormat,
};
use sha2::{Digest, Sha256};

const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0];

fn options() -> RasterPyramidOptions {
    RasterPyramidOptions::new(RasterPlacement::Explicit(IDENTITY))
}

/// A gradient whose every pixel differs from its neighbours, so a shifted tile shows.
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

fn plan(w: u32, h: u32, edge: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, edge, 0, Layout::Google)
        .unwrap()
        .plan()
}

fn decode_png(bytes: &[u8]) -> image::DynamicImage {
    image::load_from_memory_with_format(bytes, image::ImageFormat::Png).expect("a PNG")
}

#[test]
fn the_grid_follows_raster_md_and_padding_tiles_are_dropped() {
    // 300 x 100 at edge 256: two levels (2 * 256 >= 300). Level 1 is 300 x 100, so 2 x 1 tiles
    // against a 2 x 2 Google grid, and level 0 is 150 x 50, one tile.
    let raster = gradient(300, 100, PixelFormat::Gray8);
    let p = produce_raster_pyramid(&raster, &plan(300, 100, 256), &options()).unwrap();
    assert_eq!(p.levels, 2);
    assert_eq!(p.grid(1), Some((2, 1)), "ceil(300/256) by ceil(100/256)");
    assert_eq!(p.grid(0), Some((1, 1)), "150 x 50 is one tile");
    let stored: Vec<_> = p.tiles().map(|(l, c, r, _)| (l, c, r)).collect();
    assert_eq!(stored, vec![(0, 0, 0), (1, 0, 0), (1, 1, 0)]);
    // The Google grid at level 1 is 2 x 2: the two row-1 tiles are padding and were dropped.
    assert_eq!(p.report.dropped_padding_tiles, 2);
    assert!(p.tile(1, 0, 1).is_none() && p.tile(1, 1, 1).is_none());
}

#[test]
fn every_tile_is_the_planners_tile_and_a_lossless_tile_decodes_to_its_pixels() {
    let (w, h) = (700u32, 530u32);
    let raster = gradient(w, h, PixelFormat::Rgb8);
    let p = produce_raster_pyramid(&raster, &plan(w, h, 256), &options()).unwrap();
    assert_eq!(
        (p.tile_codec, p.channels, p.bit_depth),
        (RasterCodec::Png, RasterChannels::Rgb, 8)
    );

    let top = p.levels as u32 - 1;
    let (cols, rows) = p.grid(top).unwrap();
    assert_eq!((cols, rows), (3, 3));
    for row in 0..rows {
        for col in 0..cols {
            let bytes = p
                .tile(top, col, row)
                .expect("a full resolution tile is stored");
            let tile = decode_png(bytes).to_rgb8();
            assert_eq!(tile.dimensions(), (256, 256), "every tile is a full square");
            for ty in 0..256u32 {
                for tx in 0..256u32 {
                    let (x, y) = (col * 256 + tx, row * 256 + ty);
                    let got = tile.get_pixel(tx, ty).0;
                    let want = if x < w && y < h {
                        let off = (y as usize * w as usize + x as usize) * 3;
                        [
                            raster.data()[off],
                            raster.data()[off + 1],
                            raster.data()[off + 2],
                        ]
                    } else {
                        [255, 255, 255] // the padded margin is the background
                    };
                    assert_eq!(got, want, "tile ({col},{row}) pixel ({tx},{ty})");
                }
            }
        }
    }
}

#[test]
fn skip_blanks_leaves_a_tile_not_stored() {
    // The left 256 columns are mid-gray, the rest white: with a white background the right half
    // is blank at every level. A solid tile is "blank" whatever its colour, so make the left
    // half non-uniform.
    let (w, h) = (512u32, 256u32);
    let mut data = vec![255u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..256 {
            data[(y * w + x) as usize] = ((x + y) % 200) as u8;
        }
    }
    let raster = Raster::new(w, h, PixelFormat::Gray8, data).unwrap();

    let mut o = options();
    let all = produce_raster_pyramid(&raster, &plan(w, h, 256), &o).unwrap();
    o.skip_blanks = true;
    let skipped = produce_raster_pyramid(&raster, &plan(w, h, 256), &o).unwrap();

    assert!(all.tile(1, 1, 0).is_some(), "stored when blanks are kept");
    assert!(
        skipped.tile(1, 1, 0).is_none(),
        "the blank tile is not stored"
    );
    assert!(skipped.tile(1, 0, 0).is_some(), "a tile with content is");
    assert_eq!(skipped.report.not_stored_tiles, 1);
    assert_eq!(all.report.not_stored_tiles, 0);
}

#[test]
fn the_refusals_are_by_name() {
    let raster = gradient(300, 100, PixelFormat::Gray8);
    let p256 = plan(300, 100, 256);

    let mut o = options();
    o.blank_strategy = BlankTileStrategy::Placeholder;
    assert!(matches!(
        produce_raster_pyramid(&raster, &p256, &o),
        Err(RasterPyramidError::PlaceholderBlanks)
    ));
    o.blank_strategy = BlankTileStrategy::PlaceholderWithTolerance {
        max_channel_delta: 3,
    };
    assert!(matches!(
        produce_raster_pyramid(&raster, &p256, &o),
        Err(RasterPyramidError::PlaceholderBlanks)
    ));

    let mut o = options();
    o.tile_format = TileFormat::Raw;
    assert!(matches!(
        produce_raster_pyramid(&raster, &p256, &o),
        Err(RasterPyramidError::RawTiles)
    ));

    for bad in [128, 255, 300, 2048] {
        let big = PyramidPlanner::new(300, 100, bad, 0, Layout::Google)
            .unwrap()
            .plan();
        assert!(
            matches!(
                produce_raster_pyramid(&raster, &big, &options()),
                Err(RasterPyramidError::TileSize(n)) if n == bad
            ),
            "tile size {bad}"
        );
    }

    let float = Raster::zeroed(300, 100, PixelFormat::RgbaF32).unwrap();
    assert!(matches!(
        produce_raster_pyramid(&float, &p256, &options()),
        Err(RasterPyramidError::PixelFormat(PixelFormat::RgbaF32))
    ));

    for text in ["a\nb", "x\u{202E}y", &"é".repeat(128)] {
        let mut o = options();
        o.attribution = Some(text.to_string());
        assert!(
            matches!(
                produce_raster_pyramid(&raster, &p256, &o),
                Err(RasterPyramidError::Attribution(_))
            ),
            "{text:?}"
        );
    }
    let mut o = options();
    o.attribution = Some("Scan, City Archive 1974".to_string());
    let ok = produce_raster_pyramid(&raster, &p256, &o).unwrap();
    assert_eq!(ok.attribution.as_deref(), Some("Scan, City Archive 1974"));
}

#[test]
fn centring_and_overlap_are_forced_off_and_reported() {
    let raster = gradient(300, 100, PixelFormat::Gray8);
    let wild = PyramidPlanner::new(300, 100, 256, 4, Layout::DeepZoom)
        .unwrap()
        .with_centre(true)
        .plan();
    let p = produce_raster_pyramid(&raster, &wild, &options()).unwrap();
    assert!(!p.plan.centre);
    assert_eq!(p.plan.overlap, 0);
    assert_eq!(p.plan.layout, Layout::Google);
    assert_eq!(p.report.forced.len(), 3, "{:?}", p.report.forced);

    let clean = produce_raster_pyramid(&raster, &plan(300, 100, 256), &options()).unwrap();
    assert!(clean.report.forced.is_empty());
}

#[test]
fn a_sixteen_bit_source_is_converted_and_the_conversion_is_named() {
    let (w, h) = (64u32, 64u32);
    let mut data = Vec::new();
    for i in 0..w * h {
        let v = (i * 16) as u16; // spans 0..65520
        data.extend_from_slice(&v.to_ne_bytes());
    }
    let raster = Raster::new(w, h, PixelFormat::Gray16, data).unwrap();
    let p = produce_raster_pyramid(&raster, &plan(w, h, 256), &options()).unwrap();
    assert_eq!(p.channels, RasterChannels::Gray);
    assert_eq!(p.bit_depth, 8);
    assert_eq!(
        p.report.conversions,
        vec!["converted-bit-depth: Gray16 -> Gray8".to_string()]
    );

    // Rounded scaling, not the clipping cast: 65520 * 255 / 65535 rounds to 255, 32768 to 128.
    let tile = decode_png(p.tile(0, 0, 0).unwrap()).to_luma8();
    assert_eq!(tile.get_pixel(0, 0).0[0], 0);
    let mid = (32_768u32 / 16) as usize; // pixel index whose 16-bit value is 32768
    assert_eq!(
        tile.get_pixel((mid % 64) as u32, (mid / 64) as u32).0[0],
        128
    );
    assert_eq!(
        tile.get_pixel(63, 63).0[0],
        ((65_520u32 * 255 + 32_767) / 65_535) as u8
    );

    // The digest is of the samples as received, before the conversion.
    let want: [u8; 32] = Sha256::digest(raster.data()).into();
    assert_eq!(p.source_digest, want);
}

#[test]
fn a_jpeg_tile_of_an_rgba_source_says_it_is_rgb() {
    let raster = gradient(300, 100, PixelFormat::Rgba8);
    let mut o = options();
    o.tile_format = TileFormat::Jpeg { quality: 90 };
    let p = produce_raster_pyramid(&raster, &plan(300, 100, 256), &o).unwrap();
    assert_eq!(
        (p.tile_codec, p.channels),
        (RasterCodec::Jpeg, RasterChannels::Rgb)
    );
    assert_eq!(p.report.conversions.len(), 1);
    let bytes = p.tile(1, 0, 0).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "a JPEG");
    assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9], "ends at EOI");
}

#[test]
fn the_placement_is_explicit_or_composed_and_normalised() {
    let raster = gradient(300, 100, PixelFormat::Gray8);
    let mut o =
        RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, -0.0, 0.0, -1.0, 5.0, 7.0]));
    let p = produce_raster_pyramid(&raster, &plan(300, 100, 256), &o).unwrap();
    assert_eq!(
        p.pixel_to_source.map(f64::to_bits),
        [1.0, 0.0, 0.0, -1.0, 5.0, 7.0].map(f64::to_bits),
        "-0.0 leaves as +0.0"
    );
    assert_eq!(p.visible_box, [0, 0, 300, 100]);

    o.placement = RasterPlacement::Explicit([f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(matches!(
        produce_raster_pyramid(&raster, &plan(300, 100, 256), &o),
        Err(RasterPyramidError::NonFinitePlacement)
    ));

    // The vector of docs/RASTER.md section 5, through the composition.
    let composed = |crs: &str| {
        RasterPyramidOptions::new(RasterPlacement::Composed {
            pixel_to_native: GeoTransform::new(0.5, 0.25, 100.0, 0.125, -0.5, 50.0),
            pixel_origin: PixelOrigin::Corner,
            raster_crs: crs.to_string(),
            calibration: SourceCalibration::Affine {
                native_crs: "EPSG:2056".to_string(),
                metres_to_native: GeoTransform::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                metres_per_unit: 1.0,
            },
        })
    };
    let p = produce_raster_pyramid(&raster, &plan(300, 100, 256), &composed("EPSG:2056")).unwrap();
    assert_eq!(
        p.pixel_to_source.map(f64::to_bits),
        [0.5, 0.25, 0.125, -0.5, 100.0, 50.0].map(f64::to_bits)
    );

    // Another CRS is refused through source(), with the typed value intact.
    let err = produce_raster_pyramid(&raster, &plan(300, 100, 256), &composed("EPSG:4326"))
        .expect_err("a raster in another CRS is never resampled into place");
    assert!(matches!(err, RasterPyramidError::Placement(_)));
    let source = Error::source(&err).expect("the placement error is the source");
    assert!(matches!(
        source.downcast_ref::<PlacementError>(),
        Some(PlacementError::CrsMismatch { .. })
    ));
}
