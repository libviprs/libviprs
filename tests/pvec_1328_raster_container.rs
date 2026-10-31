//! Issue #1328 (R2.4): a raster pyramid through PVec's writer and back out of PVec's reader.
//!
//! The ADR's claims on the producer's side: every stored tile comes back **digest-equal** to the
//! tile the producer cut (PVec never touches the bytes), a not-stored tile comes back as nothing,
//! the placement comes back with the **same bits** as the composed one, and a lossless tile
//! decodes to exactly the pixels the producer decoded. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

use libviprs::raster_pvec::container::{RasterWriteError, write_rasters};
use libviprs::raster_pvec::placement::SourceCalibration;
use libviprs::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramid, RasterPyramidOptions, produce_raster_pyramid,
};
use libviprs::{
    GeoTransform, Layout, PixelFormat, PixelOrigin, PyramidPlanner, Raster, TileFormat,
};
use pvec_facade::unstable::{PixelFormat as PvecChannels, TileCodec};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};
use sha2::{Digest, Sha256};

fn manifest() -> Manifest {
    Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [4000.0, 4000.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1328".into(),
        producer_version: "1".into(),
    }
}

fn scan(w: u32, h: u32, format: PixelFormat) -> Raster {
    let bpp = format.bytes_per_pixel();
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..bpp {
                // The right third is flat so skip_blanks has something to leave out.
                let v = if x >= 2 * w / 3 {
                    255
                } else {
                    ((x * 5 + y * 11 + c as u32 * 31) % 250) as u8
                };
                data.push(v);
            }
        }
    }
    Raster::new(w, h, format, data).unwrap()
}

fn plan(w: u32, h: u32, edge: u32) -> libviprs::PyramidPlan {
    PyramidPlanner::new(w, h, edge, 0, Layout::Google)
        .unwrap()
        .plan()
}

/// The vector of docs/RASTER.md, composed from a GeoTransform and an identity calibration.
fn composed() -> RasterPlacement {
    RasterPlacement::Composed {
        pixel_to_native: GeoTransform::new(0.5, 0.25, 100.0, 0.125, -0.5, 50.0),
        pixel_origin: PixelOrigin::Corner,
        raster_crs: "EPSG:2056".to_string(),
        calibration: SourceCalibration::Affine {
            native_crs: "EPSG:2056".to_string(),
            metres_to_native: GeoTransform::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            metres_per_unit: 1.0,
        },
    }
}

/// An artifact needs at least one entity (WRT-01; scan-only sheets are out of v1, ADR 0018), so
/// the sheet carries its footprint outline.
fn outline(w: &mut PvecWriter) {
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 400.0, y: 0.0 },
        Point2 { x: 400.0, y: 300.0 },
        Point2 { x: 0.0, y: 300.0 },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    w.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
}

fn through_pvec(pyramids: &[RasterPyramid]) -> PvecReader {
    let mut w = PvecWriter::new(manifest()).unwrap();
    outline(&mut w);
    write_rasters(&mut w, pyramids).unwrap();
    PvecReader::open(w.finish().unwrap()).unwrap()
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[test]
fn every_tile_comes_back_digest_equal_and_the_placement_bit_equal() {
    let (w, h) = (700u32, 530u32);
    let mut o = RasterPyramidOptions::new(composed());
    o.attribution = Some("Scan, City Archive".to_string());
    let p = produce_raster_pyramid(&scan(w, h, PixelFormat::Rgb8), &plan(w, h, 256), &o).unwrap();
    let reader = through_pvec(std::slice::from_ref(&p));

    let placements = reader.raster_placements().unwrap();
    assert_eq!(placements.len(), 1);
    let pl = &placements[0];
    assert_eq!(
        pl.pixel_to_source.map(f64::to_bits),
        [0.5, 0.25, 0.125, -0.5, 100.0, 50.0].map(f64::to_bits),
        "bit-equal to the composed placement"
    );
    assert_eq!((pl.width, pl.height), (w, h));
    assert_eq!(pl.visible_box, [0, 0, w, h]);
    assert_eq!(
        (pl.tile_edge, u32::from(pl.levels)),
        (256, u32::from(p.levels))
    );
    assert_eq!(pl.tile_codec, TileCodec::Png);
    assert_eq!(pl.pixel_format, PvecChannels::Rgb);
    assert_eq!(pl.bit_depth, 8);
    assert_eq!(pl.source_digest, Some(p.source_digest));
    assert_eq!(pl.attribution, "Scan, City Archive");

    let id = pvec_facade::unstable::RasterId::new(0).unwrap();
    let mut seen = 0;
    for (level, col, row, bytes) in p.tiles() {
        let got = reader
            .raster_tile(id, level, col, row)
            .unwrap()
            .expect("a stored tile");
        assert_eq!(
            digest(&got),
            digest(bytes),
            "tile ({level},{col},{row}) is digest-equal"
        );
        assert_eq!(got, bytes);
        seen += 1;
    }
    assert!(seen > 6, "{seen} tiles");

    // A coordinate the producer dropped as padding is not stored.
    let top = u32::from(p.levels) - 1;
    assert_eq!(reader.raster_tile(id, top, 5, 5).ok().flatten(), None);
}

#[test]
fn a_not_stored_tile_comes_back_as_nothing_and_a_lossless_tile_decodes_to_the_cut_pixels() {
    let (w, h) = (768u32, 256u32);
    let mut o =
        RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 256.0]));
    o.skip_blanks = true;
    let p = produce_raster_pyramid(&scan(w, h, PixelFormat::Gray8), &plan(w, h, 256), &o).unwrap();
    let reader = through_pvec(std::slice::from_ref(&p));
    let id = pvec_facade::unstable::RasterId::new(0).unwrap();
    let top = u32::from(p.levels) - 1;

    assert!(
        p.tile(top, 2, 0).is_none(),
        "the flat right third was left out"
    );
    assert_eq!(
        reader.raster_tile(id, top, 2, 0).unwrap(),
        None,
        "a not-stored tile returns no bytes"
    );

    let stored = reader.raster_tile(id, top, 0, 0).unwrap().expect("stored");
    let ours = image::load_from_memory(p.tile(top, 0, 0).unwrap())
        .unwrap()
        .to_luma8();
    let theirs = image::load_from_memory(&stored).unwrap().to_luma8();
    assert_eq!(
        ours.as_raw(),
        theirs.as_raw(),
        "the decoded pixels are the producer's"
    );
}

#[test]
fn several_rasters_get_dense_ids_in_the_order_given() {
    let a = produce_raster_pyramid(
        &scan(300, 200, PixelFormat::Gray8),
        &plan(300, 200, 256),
        &RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 200.0])),
    )
    .unwrap();
    let mut o = RasterPyramidOptions::new(RasterPlacement::Explicit([
        2.0, 0.0, 0.0, -2.0, 50.0, 400.0,
    ]));
    o.tile_format = TileFormat::Jpeg { quality: 80 };
    let b = produce_raster_pyramid(&scan(300, 200, PixelFormat::Rgb8), &plan(300, 200, 256), &o)
        .unwrap();
    let reader = through_pvec(&[a, b]);
    let placements = reader.raster_placements().unwrap();
    assert_eq!(placements.len(), 2);
    assert_eq!(placements[0].tile_codec, TileCodec::Png);
    assert_eq!(placements[1].tile_codec, TileCodec::Jpeg);
    assert_eq!(placements[1].pixel_to_source[0], 2.0);
}

#[test]
fn a_placement_pvec_refuses_is_a_typed_error_with_the_source_intact() {
    // A singular map (rows [1 2 0] and [2 4 0], determinant 0): RASTER-05 refuses it at write
    // time, so the producer learns there and then.
    let p = produce_raster_pyramid(
        &scan(300, 200, PixelFormat::Gray8),
        &plan(300, 200, 256),
        &RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 2.0, 2.0, 4.0, 0.0, 0.0])),
    )
    .unwrap();
    let mut w = PvecWriter::new(manifest()).unwrap();
    let err = write_rasters(&mut w, &[p]).expect_err("a singular placement is refused");
    assert!(matches!(err, RasterWriteError::Pvec(_)), "{err:?}");
    assert!(
        std::error::Error::source(&err).is_some(),
        "the PVec error is the source"
    );
}

#[test]
fn every_tile_variant_libviprs_writes_goes_through_the_writer() {
    // The writer sniffs each tile's header (RASTER-03/04): PNG in gray, RGB and RGBA, baseline
    // JPEG in gray and RGB, and lossless WebP in RGB and RGBA, edge tiles padded to full squares.
    let cases = [
        (
            TileFormat::Png,
            PixelFormat::Gray8,
            PvecChannels::Gray,
            TileCodec::Png,
        ),
        (
            TileFormat::Png,
            PixelFormat::Rgb8,
            PvecChannels::Rgb,
            TileCodec::Png,
        ),
        (
            TileFormat::Png,
            PixelFormat::Rgba8,
            PvecChannels::Rgba,
            TileCodec::Png,
        ),
        (
            TileFormat::Jpeg { quality: 85 },
            PixelFormat::Gray8,
            PvecChannels::Gray,
            TileCodec::Jpeg,
        ),
        (
            TileFormat::Jpeg { quality: 85 },
            PixelFormat::Rgb8,
            PvecChannels::Rgb,
            TileCodec::Jpeg,
        ),
        // An RGBA source under JPEG is flattened, so the placement says RGB.
        (
            TileFormat::Jpeg { quality: 85 },
            PixelFormat::Rgba8,
            PvecChannels::Rgb,
            TileCodec::Jpeg,
        ),
        (
            TileFormat::Webp,
            PixelFormat::Rgb8,
            PvecChannels::Rgb,
            TileCodec::Webp,
        ),
        (
            TileFormat::Webp,
            PixelFormat::Rgba8,
            PvecChannels::Rgba,
            TileCodec::Webp,
        ),
    ];
    for (tile_format, source, channels, codec) in cases {
        let mut o =
            RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 300.0]));
        o.tile_format = tile_format;
        let p = produce_raster_pyramid(&scan(400, 300, source), &plan(400, 300, 256), &o).unwrap();
        let reader = through_pvec(std::slice::from_ref(&p));
        let pl = &reader.raster_placements().unwrap()[0];
        assert_eq!(
            (pl.tile_codec, pl.pixel_format),
            (codec, channels),
            "{tile_format:?} {source:?}"
        );
        let id = pvec_facade::unstable::RasterId::new(0).unwrap();
        let top = u32::from(p.levels) - 1;
        let got = reader
            .raster_tile(id, top, 1, 1)
            .unwrap()
            .expect("an edge tile is stored");
        assert_eq!(
            Some(got.as_slice()),
            p.tile(top, 1, 1),
            "{tile_format:?} {source:?}"
        );
    }
}

#[test]
fn the_pinned_writer_sniffs_tiles_before_it_stores_them() {
    // The one-byte marker `BlankTileStrategy::Placeholder` writes is not a tile, and neither is a
    // PNG that says it is the wrong size. The pinned PVec revision has to refuse both at write
    // time (RASTER-03), which is how this test tells a pin without the sniffers from one with.
    use pvec_facade::unstable::{Band, Placement, RasterId, TileIn};
    let placement = Placement {
        pixel_to_source: [1.0, 0.0, 0.0, -1.0, 0.0, 256.0],
        width: 256,
        height: 256,
        visible_box: [0, 0, 256, 256],
        tile_edge: 256,
        levels: 1,
        tile_codec: TileCodec::Png,
        pixel_format: PvecChannels::Gray,
        bit_depth: 8,
        opacity: 255,
        band: Band::Underlay,
        layer_index: None,
        source_digest: None,
        attribution: String::new(),
    };
    for bytes in [vec![0u8], b"\x89PNG\r\n\x1a\n not a png".to_vec()] {
        let mut w = PvecWriter::new(manifest()).unwrap();
        outline(&mut w);
        w.rasters(std::slice::from_ref(&placement)).unwrap();
        let refused = w
            .raster_tiles(
                RasterId::new(0).unwrap(),
                vec![TileIn {
                    level: 0,
                    col: 0,
                    row: 0,
                    bytes,
                }],
            )
            .and_then(|()| w.finish().map(|_| ()));
        assert!(
            refused.is_err(),
            "a tile that is not an image of the right size is refused"
        );
    }
}
