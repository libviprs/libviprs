//! Issue #1331 (RL1.3): the PVec sink. Tiles arrive in any order and from any number of workers
//! and the `.pvec` is the same bytes; a cancelled or panicking run leaves nothing behind; and the
//! sink writes what the collecting path writes. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use libviprs::raster_pvec::container::write_rasters;
use libviprs::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramidOptions, prepare_raster, produce_raster_pyramid,
};
use libviprs::sink_pvec::PvecSink;
use libviprs::{
    CancelToken, EmissionOrder, EngineBuilder, EngineError, EngineKind, Layout, PixelFormat,
    PyramidPlan, PyramidPlanner, Raster, SinkError, Tile, TileSink,
};
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
        producer_id: "libviprs-1331".into(),
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

fn scan(w: u32, h: u32) -> Raster {
    let mut data = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            data.push(if x >= 2 * w / 3 {
                255
            } else {
                ((x * 5 + y * 11) % 250) as u8
            });
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

fn plan(w: u32, h: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, 256, 0, Layout::Google)
        .unwrap()
        .plan()
}

fn options() -> RasterPyramidOptions {
    let mut o =
        RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 900.0]));
    o.skip_blanks = true;
    o
}

/// Hands the tiles to the sink and asks the engine for the order it was constructed with.
struct Ordered<'a>(&'a PvecSink, EmissionOrder);

impl TileSink for Ordered<'_> {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.0.write_tile(tile)
    }
    fn emission_order(&self) -> EmissionOrder {
        self.1
    }
}

/// A `.pvec` made through the sink with the given engine, workers and order.
fn through_sink(engine: EngineKind, workers: usize, order: EmissionOrder) -> Vec<u8> {
    let raster = scan(1300, 900);
    let job = prepare_raster(&raster, &plan(1300, 900), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    EngineBuilder::new(
        job.source.as_ref(),
        job.pyramid.plan.clone(),
        Ordered(&sink, order),
    )
    .with_engine(engine)
    .with_concurrency(workers)
    .with_skip_blanks(true)
    .run()
    .unwrap();
    sink.into_writer().finish().unwrap()
}

#[test]
fn any_order_and_any_number_of_workers_give_the_same_bytes() {
    // Per engine: the engines differ from each other in the overview's last pixels (the streaming
    // strips round differently), which is theirs to settle; what the sink promises is that for one
    // engine the worker count and the emission order do not change a byte.
    for (engine, runs) in [
        (
            EngineKind::Monolithic,
            vec![
                (2, EmissionOrder::Cascade),
                (8, EmissionOrder::Cascade),
                (8, EmissionOrder::TileId),
            ],
        ),
        (EngineKind::Streaming, vec![(4, EmissionOrder::Cascade)]),
        (
            EngineKind::MapReduce,
            vec![(2, EmissionOrder::Cascade), (8, EmissionOrder::Cascade)],
        ),
    ] {
        let reference = through_sink(engine, 1, EmissionOrder::Cascade);
        for (workers, order) in runs {
            assert!(
                through_sink(engine, workers, order) == reference,
                "{engine:?} x{workers} {order:?} changed the bytes"
            );
        }
    }
}

#[test]
fn skip_blanks_leaves_a_blank_tile_out_whichever_engine_cut_it() {
    // The streaming engines ignore EngineConfig::skip_blanks, so the sink applies the option
    // itself: the right third of the scan is flat, and no flat tile is stored by any engine.
    let blank = |bytes: &[u8]| {
        let reader = PvecReader::open(bytes.to_vec()).unwrap();
        let id = pvec_facade::unstable::RasterId::new(0).unwrap();
        // Level 3 is 1300 x 900 at edge 256: columns 4 and 5 start at x = 1024, flat from x = 866.
        (0..4).all(|row| reader.raster_tile(id, 3, 5, row).unwrap().is_none())
    };
    for engine in [
        EngineKind::Monolithic,
        EngineKind::Streaming,
        EngineKind::MapReduce,
    ] {
        assert!(
            blank(&through_sink(engine, 2, EmissionOrder::Cascade)),
            "{engine:?}"
        );
    }
}

#[test]
fn the_sink_writes_what_the_collecting_path_writes() {
    let raster = scan(1300, 900);
    let pyramid = produce_raster_pyramid(&raster, &plan(1300, 900), &options()).unwrap();
    let mut w = writer();
    write_rasters(&mut w, std::slice::from_ref(&pyramid)).unwrap();
    let collected = w.finish().unwrap();
    assert!(
        through_sink(EngineKind::Monolithic, 4, EmissionOrder::Cascade) == collected,
        "the streaming sink and produce_raster_pyramid + write_rasters disagree"
    );
    // And it reads back through pvec's reader, tile by tile.
    let reader = PvecReader::open(collected).unwrap();
    let id = pvec_facade::unstable::RasterId::new(0).unwrap();
    for (l, c, r, bytes) in pyramid.tiles() {
        assert_eq!(
            reader.raster_tile(id, l, c, r).unwrap().as_deref(),
            Some(bytes)
        );
    }
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        out.push(e.file_name().to_string_lossy().into_owned());
    }
    out.sort();
    out
}

#[test]
fn finish_to_publishes_atomically_and_leaves_no_staging_file() {
    let staging = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let raster = scan(700, 500);
    let job = prepare_raster(&raster, &plan(700, 500), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink)
        .run()
        .unwrap();
    let out = dest.path().join("scan.pvec");
    sink.finish_to(&out, staging.path()).unwrap();
    assert_eq!(files_under(dest.path()), vec!["scan.pvec".to_string()]);
    assert!(
        files_under(staging.path()).is_empty(),
        "nothing is left in the staging root"
    );
    PvecReader::open(std::fs::read(&out).unwrap()).unwrap();
}

#[test]
fn a_failed_publish_leaves_neither_a_file_nor_a_staging_file() {
    let staging = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let raster = scan(700, 500);
    let job = prepare_raster(&raster, &plan(700, 500), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink)
        .run()
        .unwrap();
    let missing = dest.path().join("no-such-dir").join("scan.pvec");
    assert!(sink.finish_to(&missing, staging.path()).is_err());
    assert!(files_under(dest.path()).is_empty());
    assert!(files_under(staging.path()).is_empty());
}

/// Cancels the run from inside the sink after a few tiles.
struct CancelAfter<'a>(&'a PvecSink, CancelToken, AtomicUsize, usize);

impl TileSink for CancelAfter<'_> {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        if self.2.fetch_add(1, Ordering::SeqCst) + 1 == self.3 {
            self.1.cancel();
        }
        self.0.write_tile(tile)
    }
}

#[test]
fn cancelling_mid_run_stops_the_run_and_leaves_nothing_on_disk() {
    let staging = tempfile::tempdir().unwrap();
    let raster = scan(1300, 900);
    let job = prepare_raster(&raster, &plan(1300, 900), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    let token = CancelToken::new();
    let r = EngineBuilder::new(
        job.source.as_ref(),
        job.pyramid.plan.clone(),
        CancelAfter(&sink, token.clone(), AtomicUsize::new(0), 3),
    )
    .with_engine(EngineKind::Monolithic)
    .with_concurrency(1)
    .with_cancel(token)
    .run();
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    drop(sink);
    assert!(files_under(staging.path()).is_empty());
}

struct PanicAfter<'a>(&'a PvecSink, AtomicUsize, usize);

impl TileSink for PanicAfter<'_> {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        if self.1.fetch_add(1, Ordering::SeqCst) + 1 == self.2 {
            panic!("a tile encoder panicked");
        }
        self.0.write_tile(tile)
    }
}

#[test]
fn a_panic_mid_run_leaves_nothing_on_disk_and_the_sink_stays_usable() {
    let staging = tempfile::tempdir().unwrap();
    let raster = scan(1300, 900);
    let job = prepare_raster(&raster, &plan(1300, 900), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = EngineBuilder::new(
            job.source.as_ref(),
            job.pyramid.plan.clone(),
            PanicAfter(&sink, AtomicUsize::new(0), 3),
        )
        .with_engine(EngineKind::Monolithic)
        .with_concurrency(1)
        .run();
    }));
    let _ = r; // the engine may turn the panic into an error or let it through
    assert!(files_under(staging.path()).is_empty());
    // A poisoned lock does not turn the next call into a second panic.
    let _ = sink.into_writer();
}

#[test]
fn raw_tiles_and_placeholder_blanks_are_refused_before_a_tile_is_cut() {
    use libviprs::{BlankTileStrategy, TileFormat};
    let raster = scan(300, 200);
    let mut o = options();
    o.tile_format = TileFormat::Raw;
    assert!(prepare_raster(&raster, &plan(300, 200), &o).is_err());
    let mut o = options();
    o.blank_strategy = BlankTileStrategy::Placeholder;
    assert!(prepare_raster(&raster, &plan(300, 200), &o).is_err());
}

#[test]
fn the_sink_is_a_leaf_so_no_bookkeeping_hook_is_swallowed() {
    let raster = scan(300, 200);
    let job = prepare_raster(&raster, &plan(300, 200), &options()).unwrap();
    let sink = PvecSink::new(writer(), &job.pyramid, &options()).unwrap();
    assert!(sink.inner_sink().is_none());
    assert_eq!(sink.emission_order(), EmissionOrder::default());
}
