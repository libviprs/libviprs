//! N8.6 (#1443, slice of #1064): the one tunable the conversion exposes and the benchmarks can
//! measure, `PvecConversion::queue_capacity`, swept on `large_industrial`.
//!
//! `#[ignore]`d. Every capacity is run three times; the wall time is the median with min and max,
//! `peak_queue_depth` is what the conversion reports, and the artifact is hashed: the artifact must
//! be byte for byte the same at every capacity, and a difference fails the sweep whatever its speed.
//! A capacity replaces the default only if it is faster than the default by more than the spread of
//! both, and the run prints which way it came out.
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
//!     --test pvec_queue_sweep -- --ignored --nocapture queue_sweep
//! ```

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::time::{Duration, Instant};

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec_convert::PvecConversion;
use pvec_bench::{
    Stat, cells, generate, host_cpus, host_load, manifest, platform, require_quiet, sha256_hex,
};
use serde_json::json;

const CAPACITIES: [usize; 11] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096];

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "takes numbers: needs a quiet host, run with --ignored --nocapture"]
fn queue_sweep() {
    let platform = platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let before = require_quiet(30, Duration::from_secs(20))
        .unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let m = manifest();
    let c = &cells(&m)["large_industrial"];
    let bytes = generate(c["generator"].as_str().unwrap(), &c["params"]);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let mut reference: Option<String> = None;
    let mut rows = Vec::new();
    for cap in CAPACITIES {
        let mut walls = Vec::new();
        let mut depth = 0;
        for _ in 0..3 {
            let t = Instant::now();
            let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
                .unit(SourceUnit::Meters)
                .queue_capacity(cap)
                .convert()
                .expect("converts");
            walls.push(t.elapsed().as_secs_f64());
            depth = depth.max(out.peak_queue_depth);
            assert!(
                out.peak_queue_depth as usize <= cap,
                "the queue held more than its capacity"
            );
            let hash = sha256_hex(&std::fs::read(&dest).unwrap());
            match &reference {
                None => reference = Some(hash),
                Some(r) => assert_eq!(
                    r, &hash,
                    "capacity {cap} changed the artifact: the sweep is rejected"
                ),
            }
        }
        let s = Stat::of(&walls);
        let row = json!({"bench": "queue", "capacity": cap, "wall_secs": {"median": s.median, "min": s.min, "max": s.max}, "peak_queue_depth": depth, "unstable": s.unstable(), "platform": platform});
        println!("ROW {row}");
        rows.push((cap, s));
    }
    let default = rows.iter().find(|(c, _)| *c == 64).unwrap().1;
    for (cap, s) in &rows {
        let faster = default.median - s.median;
        let beyond_noise = faster > (default.max - default.min) + (s.max - s.min);
        println!(
            "VERDICT capacity {cap}: {:+.1}% against 64 ({})",
            (s.median / default.median - 1.0) * 100.0,
            if *cap == 64 {
                "the default"
            } else if beyond_noise {
                "faster than the default beyond the noise of both"
            } else {
                "not distinguishable from the default"
            }
        );
    }
    let after = host_load().unwrap_or(f64::MAX);
    assert!(
        after < host_cpus() as f64,
        "discarding the run: the one-minute load rose to {after} (it was {before})"
    );
}
