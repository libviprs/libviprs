//! Issue #1332: a gray raster can be cut into 1-bit PNG tiles (`encode_png_bilevel`, issue
//! #1330), through the collecting path and through the PVec sink, and the pyramid says so.
//! Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

use libviprs::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramidError, RasterPyramidOptions, prepare_raster,
    produce_raster_pyramid,
};
use libviprs::sink_pvec::PvecSink;
use libviprs::{
    EngineBuilder, EngineKind, Layout, PixelFormat, PyramidPlan, PyramidPlanner, Raster, TileFormat,
};
use pvec_facade::unstable::RasterId;
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};

fn scan() -> Raster {
    let data = (0..300u32 * 300)
        .map(|i| {
            if (i % 300 / 7 + i / 300 / 5) % 2 == 0 {
                0
            } else {
                255
            }
        })
        .collect();
    Raster::new(300, 300, PixelFormat::Gray8, data).unwrap()
}

fn plan() -> PyramidPlan {
    PyramidPlanner::new(300, 300, 256, 0, Layout::Google)
        .unwrap()
        .plan()
}

fn options(threshold: Option<u8>) -> RasterPyramidOptions {
    let mut o =
        RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 300.0]));
    o.bilevel_threshold = threshold;
    o
}

fn depth(png: &[u8]) -> u8 {
    assert_eq!(&png[1..4], b"PNG");
    png[24]
}

#[test]
fn a_bilevel_threshold_cuts_every_tile_of_every_level_at_one_bit() {
    let p = produce_raster_pyramid(&scan(), &plan(), &options(Some(128))).unwrap();
    assert_eq!(p.bit_depth, 1);
    assert!(p.tiles().count() > 1);
    for (l, c, r, bytes) in p.tiles() {
        assert_eq!(depth(bytes), 1, "tile {l}/{c}/{r}");
    }
    // Level 0 of the full resolution is exact: the pixels read back equal the source.
    let top = u32::from(p.levels) - 1;
    let img = image::load_from_memory(p.tile(top, 0, 0).unwrap())
        .unwrap()
        .to_luma8();
    let src = scan();
    for y in 0..256usize {
        for x in 0..256usize {
            assert_eq!(
                img.get_pixel(x as u32, y as u32).0[0],
                src.data()[y * 300 + x]
            );
        }
    }
}

#[test]
fn without_a_threshold_the_tiles_stay_eight_bit() {
    let p = produce_raster_pyramid(&scan(), &plan(), &options(None)).unwrap();
    assert_eq!(p.bit_depth, 8);
    assert!(p.tiles().all(|(_, _, _, b)| depth(b) == 8));
}

#[test]
fn a_threshold_needs_gray_eight_bit_png() {
    let rgb = Raster::new(256, 256, PixelFormat::Rgb8, vec![0; 256 * 256 * 3]).unwrap();
    let plan = PyramidPlanner::new(256, 256, 256, 0, Layout::Google)
        .unwrap()
        .plan();
    let e = produce_raster_pyramid(&rgb, &plan, &options(Some(128))).unwrap_err();
    assert!(matches!(e, RasterPyramidError::Tile(_)), "{e:?}");
    let mut o = options(Some(128));
    o.tile_format = TileFormat::Jpeg { quality: 90 };
    let e = produce_raster_pyramid(&scan(), &self::plan(), &o).unwrap_err();
    assert!(matches!(e, RasterPyramidError::Tile(_)), "{e:?}");
}

#[test]
fn the_sink_cuts_one_bit_tiles_the_same_as_the_collecting_path() {
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [300.0, 300.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1332".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 300.0, y: 0.0 },
        Point2 { x: 300.0, y: 300.0 },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    w.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
    let o = options(Some(128));
    let raster = scan();
    let job = prepare_raster(&raster, &plan(), &o).unwrap();
    assert_eq!(job.pyramid.bit_depth, 1);
    let sink = PvecSink::new(w, &job.pyramid, &o).unwrap();
    EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink)
        .with_engine(EngineKind::Monolithic)
        .run()
        .unwrap();
    let reader = PvecReader::open(sink.into_writer().finish().unwrap()).unwrap();
    assert_eq!(reader.raster_placements().unwrap()[0].bit_depth, 1);
    let want = produce_raster_pyramid(&raster, &plan(), &o).unwrap();
    for (l, c, r, bytes) in want.tiles() {
        let got = reader
            .raster_tile(RasterId::new(0).unwrap(), l, c, r)
            .unwrap()
            .unwrap();
        assert_eq!(got, bytes, "tile {l}/{c}/{r}");
        assert_eq!(depth(&got), 1);
    }
}
