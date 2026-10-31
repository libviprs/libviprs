//! Issue #1347 (RL1.4a): a raster placement carries the opacity and the layer index the caller
//! asked for, through `write_rasters` and through the sink. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

use libviprs::raster_pvec::container::write_rasters;
use libviprs::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramidOptions, prepare_raster, produce_raster_pyramid,
};
use libviprs::sink_pvec::PvecSink;
use libviprs::{EngineBuilder, Layout, PixelFormat, PyramidPlanner, Raster};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};

fn writer() -> PvecWriter {
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [4000.0, 4000.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1347".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 400.0, y: 0.0 },
        Point2 { x: 400.0, y: 300.0 },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    w.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
    w
}

fn scan() -> Raster {
    let data = (0..300u32 * 200).map(|i| (i % 251) as u8).collect();
    Raster::new(300, 200, PixelFormat::Gray8, data).unwrap()
}

fn options(opacity: u8, layer: Option<u32>) -> RasterPyramidOptions {
    let mut o =
        RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 200.0]));
    o.opacity = opacity;
    o.layer_index = layer;
    o
}

fn plan() -> libviprs::PyramidPlan {
    PyramidPlanner::new(300, 200, 256, 0, Layout::Google)
        .unwrap()
        .plan()
}

#[test]
fn the_defaults_stay_opaque_and_off_every_layer() {
    let o = RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]));
    assert_eq!(o.opacity, 255);
    assert_eq!(o.layer_index, None);
}

#[test]
fn write_rasters_stores_the_opacity_and_layer() {
    let pyramid = produce_raster_pyramid(&scan(), &plan(), &options(128, Some(0))).unwrap();
    assert_eq!(pyramid.opacity, 128);
    assert_eq!(pyramid.layer_index, Some(0));
    let mut w = writer();
    write_rasters(&mut w, [&pyramid]).unwrap();
    let reader = PvecReader::open(w.finish().unwrap()).unwrap();
    let placements = reader.raster_placements().unwrap();
    assert_eq!(placements[0].opacity, 128);
    assert_eq!(placements[0].layer_index, Some(0));
}

#[test]
fn the_sink_stores_the_opacity_and_layer() {
    let raster = scan();
    let opts = options(7, Some(2));
    let job = prepare_raster(&raster, &plan(), &opts).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &opts).unwrap();
    EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink)
        .run()
        .unwrap();
    let reader = PvecReader::open(sink.into_writer().finish().unwrap()).unwrap();
    let placements = reader.raster_placements().unwrap();
    assert_eq!(placements[0].opacity, 7);
    assert_eq!(placements[0].layer_index, Some(2));
}
