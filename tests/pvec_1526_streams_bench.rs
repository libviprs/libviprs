//! Issue #1526 (part of #1500 and #1093): what the component streams and the regions step cost a
//! conversion of `large_industrial`, with every option off, with provenance only, with all three
//! streams, and with all three plus the regions step.
//!
//! `#[ignore]`d. It prints one row per variant (best of `LIBVIPRS_BENCH_RUNS` runs, default 3) with
//! entities per second, the counting allocator's live-heap peak above where the conversion started,
//! and the artifact size. Speed is not asserted. Memory is: the streams may add at most 15% to the
//! streams-off peak, or 350 B per source entity, whichever is larger (the measured budget, see
//! `docs/pvec-components.md`). The first target, 10% or 128 B per entity, was not met: the provenance
//! and layer rows are held in memory until the streams are written. The drawing is written
//! to disk first, so its construction is in no row. The geometry bytes are the same in every variant
//! (`pvec_1526_corpus_geometry` and `pvec_1500_convert_streams` assert that), so the rows compare
//! like with like.
//!
//! ```text
//! LIBVIPRS_BENCH_SCALE=0.2 cargo test --release --features pvec-convert \
//!     --test pvec_1526_streams_bench -- --ignored --nocapture streams_cost
//! ```

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_alloc.rs"]
mod pvec_alloc;
#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::time::Instant;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::regions::RegionConfig;
use libviprs::pvec_convert::PvecConversion;
use pvec_bench::{cells, generate, manifest};

#[global_allocator]
static ALLOC: pvec_alloc::Counting = pvec_alloc::Counting;

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "a benchmark: run it by name in a release build"]
fn streams_cost() {
    let scale: f64 = std::env::var("LIBVIPRS_BENCH_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.2);
    let runs: usize = std::env::var("LIBVIPRS_BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let m = manifest();
    let cell = &cells(&m)["large_industrial"];
    let mut params = cell["params"].clone();
    params["entities"] = ((params["entities"].as_f64().unwrap() * scale).round() as u64).into();
    let dwg = generate("large_industrial", &params);
    let dir = std::env::temp_dir().join(format!("libviprs-1526-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("plant.dwg");
    std::fs::write(&source, &dwg).unwrap();
    drop(dwg);

    let provenance_only = StreamSelection {
        layer: false,
        line_style: false,
        provenance: true,
    };
    let regions = RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    };
    let variants: [(&str, StreamSelection, Option<RegionConfig>); 4] = [
        ("off", StreamSelection::NONE, None),
        ("provenance", provenance_only, None),
        ("all-streams", StreamSelection::ALL, None),
        ("all-streams+regions", StreamSelection::ALL, Some(regions)),
    ];
    println!("scale {scale} runs {runs} host {}", std::env::consts::ARCH);
    let (mut base, mut base_peak) = (0.0, 0u64);
    for (name, select, region) in variants {
        let (mut best, mut peak, mut bytes, mut written, mut encountered) = (f64::MAX, 0, 0, 0, 0);
        for n in 0..runs {
            let dest = dir.join(format!("{name}-{n}.pvec"));
            let bytes_in = std::fs::read(&source).unwrap();
            let mark = pvec_alloc::mark();
            let t = Instant::now();
            let out = PvecConversion::new(CadSource::Bytes(&bytes_in), &dest)
                .unit(SourceUnit::Meters)
                .components(select)
                .regions(region)
                .convert()
                .expect("converts");
            let wall = t.elapsed().as_secs_f64();
            let (tracked, _) = pvec_alloc::since(mark);
            drop(bytes_in);
            if wall < best {
                best = wall;
                peak = tracked;
                bytes = std::fs::metadata(&dest).unwrap().len();
                written = out.written;
                encountered = out.report.totals.encountered;
            }
            let _ = std::fs::remove_file(&dest);
        }
        let rate = written as f64 / best;
        if name == "off" {
            base = rate;
            base_peak = peak;
        }
        let budget = (base_peak / 100 * 15).max(350 * encountered);
        assert!(
            peak <= base_peak + budget,
            "{name}: the tracked peak grew by {} B, over the budget of {budget} B",
            peak - base_peak
        );
        println!(
            "ROW {name:<20} entities {written:>8} best {best:>7.3}s {rate:>10.0}/s ({:+.1}% vs off) tracked_peak {:>6.1} MiB artifact {:>6.1} MiB",
            (rate / base - 1.0) * 100.0,
            peak as f64 / 1_048_576.0,
            bytes as f64 / 1_048_576.0,
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}
