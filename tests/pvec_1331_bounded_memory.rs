//! Issue #1331: live heap of the PVec sink against the PMTiles sink on the same scan, measured.
//! Ignored by default like `pmtiles_bounded_memory.rs`; the gate runs it with `--release
//! --ignored`. Needs `pvec-raster`.
//!
//! The basis is a counting `#[global_allocator]` (live bytes, high-water mark), not process RSS,
//! which never falls and so cannot tell a run's peak from the source the test itself holds. The
//! source raster is allocated first and each run's peak is taken above that baseline.
//!
//! What can be promised is limited by the writer the sink hands tiles to. PVec's writer keeps the
//! encoded tiles of the artifact until `finish` and returns the artifact as one `Vec<u8>`, so the
//! sink's peak is the engine's working set, which is what the PMTiles sink also pays, plus the
//! encoded pyramid held by the writer, a copy of it at `finish`, and the index (24 bytes a tile).
//! The test pins exactly that: the PVec peak is within the PMTiles peak plus three times the
//! artifact plus the index allowance, and is at least the artifact (so the bound cannot pass on a
//! sink that stored nothing). The numbers print with `--nocapture`.
//!
//! `LIBVIPRS_1331_SCAN` picks the canvas: `a0` (18,720 x 13,240, the default and the gate's) or
//! `a2` (a quarter of it, for a laptop).

#![cfg(feature = "pvec-raster")]

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use libviprs::raster_pvec::pyramid::{RasterPlacement, RasterPyramidOptions, prepare_raster};
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::sink_pvec::PvecSink;
use libviprs::{EngineBuilder, EngineKind, Layout, PixelFormat, PyramidPlanner, Raster};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecWriter};
use pvec_geometry::{Point2, Polyline};

struct Counting;

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

// SAFETY: every method forwards to `System` and only adds to atomic counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size() as u64, Ordering::Relaxed) + l.size() as u64;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as u64, Ordering::Relaxed);
    }
    unsafe fn alloc_zeroed(&self, l: AllocLayout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size() as u64, Ordering::Relaxed) + l.size() as u64;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: AllocLayout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            let now = if new >= l.size() {
                LIVE.fetch_add((new - l.size()) as u64, Ordering::Relaxed) + (new - l.size()) as u64
            } else {
                LIVE.fetch_sub((l.size() - new) as u64, Ordering::Relaxed) - (l.size() - new) as u64
            };
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// A scanned sheet: white paper, a grid of dark lines and a few diagonals, and a left third of
/// photographic texture (a hash quantised to 16 levels) standing for an aerial underlay, so the
/// encoded pyramid is a real size and not a few kilobytes of lines.
fn scan(w: u32, h: u32) -> Raster {
    let mut data = vec![255u8; w as usize * h as usize];
    for y in 0..h {
        let row = &mut data[y as usize * w as usize..][..w as usize];
        if y % 512 < 3 {
            row.fill(20);
        } else {
            for x in (0..w).step_by(512) {
                for dx in 0..3 {
                    if let Some(p) = row.get_mut((x + dx) as usize) {
                        *p = 20;
                    }
                }
            }
            let d = (y as usize * 7 / 5) % w as usize;
            row[d] = 30;
        }
        for (x, p) in row.iter_mut().enumerate().take(w as usize / 3) {
            let hash = (x as u32).wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503);
            *p = ((hash >> 11) & 0xF0) as u8 | 0x08;
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

fn writer(w: u32, h: u32) -> PvecWriter {
    let mut pw = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [f64::from(w), f64::from(h)],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1331".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 {
            x: f64::from(w),
            y: 0.0,
        },
        Point2 {
            x: f64::from(w),
            y: f64::from(h),
        },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    pw.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
    pw
}

/// The high-water mark of live heap above where it stood when `f` began.
fn peak_above_baseline<T>(f: impl FnOnce() -> T) -> (u64, T) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let out = f();
    (PEAK.load(Ordering::Relaxed).saturating_sub(base), out)
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem
#[ignore = "memory gate: run with --release --ignored on the gate host"]
fn the_pvec_sink_peaks_within_the_pmtiles_sink_plus_the_encoded_pyramid() {
    let (w, h) = match std::env::var("LIBVIPRS_1331_SCAN").as_deref() {
        Ok("a2") => (9_360, 6_620),
        _ => (18_720, 13_240),
    };
    let raster = scan(w, h);
    let plan = PyramidPlanner::new(w, h, 512, 0, Layout::Google)
        .unwrap()
        .plan();
    let budget: u64 = 512 * 1024 * 1024;
    let dir = tempfile::tempdir().unwrap();

    let started = std::time::Instant::now();
    let (pm_peak, pm_bytes) = peak_above_baseline(|| {
        let out = dir.path().join("scan.pmtiles");
        let sink = PmTilesSink::builder(&out)
            .plan(plan.clone())
            .build()
            .unwrap();
        EngineBuilder::new(&raster, plan.clone(), sink)
            .with_engine(EngineKind::Streaming)
            .with_memory_budget(budget)
            .run()
            .unwrap();
        std::fs::metadata(&out).unwrap().len()
    });

    let pm_wall = started.elapsed();

    let options = RasterPyramidOptions::new(RasterPlacement::Explicit([
        1.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        f64::from(h),
    ]));
    let started = std::time::Instant::now();
    let (pv_peak, (artifact, tiles)) = peak_above_baseline(|| {
        let job = prepare_raster(&raster, &plan, &options).unwrap();
        let sink = PvecSink::new(writer(w, h), &job.pyramid, &options).unwrap();
        EngineBuilder::new(job.source.as_ref(), job.pyramid.plan.clone(), &sink)
            .with_engine(EngineKind::Streaming)
            .with_memory_budget(budget)
            .run()
            .unwrap();
        let tiles = sink.stored_tiles();
        let out = dir.path().join("scan.pvec");
        sink.finish_to(&out, dir.path()).unwrap();
        (std::fs::metadata(&out).unwrap().len(), tiles)
    });

    let pv_wall = started.elapsed();
    eprintln!(
        "scan {w}x{h}, budget {} MiB, {}: pmtiles peak {} MiB (archive {} MiB, {:.1} s), \
         pvec peak {} MiB (artifact {} MiB, {tiles} tiles, {:.1} s, {:.0} tiles/s)",
        budget >> 20,
        std::env::consts::ARCH,
        pm_peak >> 20,
        pm_bytes >> 20,
        pm_wall.as_secs_f64(),
        pv_peak >> 20,
        artifact >> 20,
        pv_wall.as_secs_f64(),
        tiles as f64 / pv_wall.as_secs_f64(),
    );
    if let Some(path) = std::env::var_os("LIBVIPRS_1331_DUMP_PGM") {
        // The same pixels for the GDAL comparison (outside this test): a binary PGM.
        let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
        pgm.extend_from_slice(raster.data());
        std::fs::write(path, pgm).unwrap();
    }
    assert!(
        tiles > 0 && artifact > 0,
        "a run that stored nothing proves nothing"
    );
    let index = 24 * tiles;
    assert!(
        pv_peak >= artifact,
        "the writer holds the encoded pyramid, so the peak cannot be below the artifact"
    );
    assert!(
        pv_peak <= pm_peak + 3 * artifact + index + (8 << 20),
        "pvec peak {pv_peak} against pmtiles {pm_peak} + 3 x {artifact} + {index}"
    );
}
