//! N8.2 (#1433, slice of #1064): what the DWG provider costs, per cell of the corpus.
//!
//! `#[ignore]`d and run with `--ignored --nocapture`, in the crate's convention: no `criterion`, no
//! `benches/`, a counting allocator for the tracked figures and `VmHWM` for the process peak. The
//! provider's own cost is open plus block resolution (`BlockResolver::flatten`) with no provenance
//! sink and nothing after it: normalization is not fed, so the number is the provider's alone.
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
//!     --test pvec_provider_bench -- --ignored --nocapture provider_bench
//! ```
//!
//! The run waits for a quiet host (one-minute load under the cpu count for three polls) and
//! refuses, with the loads it saw, when there is none; it checks again afterwards and discards the
//! run when the host got busy. Each figure is the median of three runs with min and max, and a
//! spread over 20% marks the row unstable. Set `LIBVIPRS_BENCH_BASELINE_OUT=path` to write the rows
//! in the baseline's shape.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_alloc.rs"]
mod pvec_alloc;
#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::time::{Duration, Instant};

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind, RawGeometry};
use pvec_bench::{
    BASELINE, Stat, baseline_rows, blank_columns, cells, generate, host_cpus, host_load, manifest,
    peak_rss, platform, provider_regressions, require_quiet, reset_peak_rss,
};
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: pvec_alloc::Counting = pvec_alloc::Counting;

/// What one extraction of a drawing came to.
struct Run {
    secs: f64,
    leaves: u64,
    splines: u64,
    instances: u64,
    tracked_peak: u64,
    allocated: u64,
}

/// Opens `bytes` and flattens model space. With `open` false only the flatten is timed, the
/// provider having been opened beforehand (the recursion sweep wants the walk alone).
fn extract(bytes: &[u8]) -> Run {
    let m = pvec_alloc::mark();
    let t = Instant::now();
    let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
    let layout = provider
        .layouts()
        .expect("layouts")
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model space");
    let flat = BlockResolver::new(&provider, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(layout.id),
            "bench",
            &layout.name,
            &mut NoProvenance,
            &mut LineageStore::new(),
        )
        .expect("flattens");
    let secs = t.elapsed().as_secs_f64();
    let (tracked_peak, allocated) = pvec_alloc::since(m);
    Run {
        secs,
        leaves: flat.entities.len() as u64,
        splines: flat
            .entities
            .iter()
            .filter(|f| matches!(f.entity.geometry, RawGeometry::Spline { .. }))
            .count() as u64,
        instances: flat.instances,
        tracked_peak,
        allocated,
    }
}

fn stat_json(s: Stat) -> Value {
    json!({"median": s.median, "min": s.min, "max": s.max})
}

fn measure(cell: &str, bytes: &[u8], platform: &str) -> Value {
    let rss_reset = reset_peak_rss();
    // A cell that takes milliseconds is run in a batch, so a sample is at least a quarter second
    // and the clock's granularity and the first-touch cost are not the figure.
    let reps = (0.25 / extract(bytes).secs).ceil().max(1.0) as usize;
    let runs: Vec<Run> = (0..3)
        .map(|_| {
            let batch: Vec<Run> = (0..reps).map(|_| extract(bytes)).collect();
            Run {
                secs: batch.iter().map(|r| r.secs).sum::<f64>() / reps as f64,
                tracked_peak: batch.iter().map(|r| r.tracked_peak).max().unwrap_or(0),
                ..batch.into_iter().next().expect("one run")
            }
        })
        .collect();
    let rss = peak_rss();
    let leaves = runs[0].leaves;
    assert!(
        runs.iter().all(|r| r.leaves == leaves),
        "{cell}: the leaf count moved between runs"
    );
    let eps = Stat::of(
        &runs
            .iter()
            .map(|r| r.leaves as f64 / r.secs)
            .collect::<Vec<_>>(),
    );
    let secs = Stat::of(&runs.iter().map(|r| r.secs).collect::<Vec<_>>());
    let row = json!({
        "bench": "provider",
        "cell": cell,
        "entities": leaves,
        "wall_secs": stat_json(secs),
        "entities_per_s": stat_json(eps),
        "allocated_bytes": runs[0].allocated,
        "tracked_peak": runs.iter().map(|r| r.tracked_peak).max(),
        "peak_rss_bytes": rss,
        "rss_is_per_cell": rss_reset,
        "block_frames": runs[0].instances,
        "advanced_curve_share": runs[0].splines as f64 / leaves as f64,
        "unstable": eps.unstable(),
        "platform": platform,
    });
    let blank = blank_columns(
        &row,
        &[
            "cell",
            "entities",
            "wall_secs",
            "entities_per_s",
            "allocated_bytes",
            "tracked_peak",
            "peak_rss_bytes",
            "block_frames",
            "advanced_curve_share",
            "platform",
        ],
    );
    assert!(
        blank.is_empty(),
        "{cell}: the row has blank columns {blank:?}, so it is not published"
    );
    row
}

/// `block_heavy` at nesting depth `d`, with about `2^16` instances in all whatever the depth, so a
/// time per instance compares across depths. Only the walk is timed.
fn sweep_point(d: u64) -> (f64, u64) {
    let per_root = (1u64 << d) - 1;
    let roots = (65_535u64).div_ceil(per_root);
    let bytes = generate(
        "block_heavy",
        &json!({"depth": d, "fanout": 2, "roots": roots}),
    );
    let provider = AcadrustProvider::open(CadSource::Bytes(&bytes)).expect("opens");
    let layout = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let mut per_instance = Vec::new();
    let mut instances = 0;
    for _ in 0..3 {
        let t = Instant::now();
        let flat = BlockResolver::new(&provider, FlattenBounds::default())
            .flatten(
                &EntityScope::Layout(layout.id),
                "bench",
                &layout.name,
                &mut NoProvenance,
                &mut LineageStore::new(),
            )
            .expect("flattens");
        let secs = t.elapsed().as_secs_f64();
        instances = flat.instances;
        per_instance.push(secs / flat.instances as f64 * 1e9);
    }
    (Stat::of(&per_instance).median, instances)
}

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "takes numbers: needs a quiet host, run with --ignored --nocapture"]
fn provider_bench() {
    let platform = platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let before = require_quiet(30, Duration::from_secs(20))
        .unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let m = manifest();
    let all = cells(&m);
    let mut rows = Vec::new();
    for cell in ["simple", "block_heavy", "spline_heavy", "large_industrial"] {
        let c = &all[cell];
        let bytes = generate(c["generator"].as_str().unwrap(), &c["params"]);
        let row = measure(cell, &bytes, &platform);
        println!("ROW {row}");
        rows.push(row);
    }
    let mut sweep = Vec::new();
    for d in 1..=16u64 {
        let (ns, instances) = sweep_point(d);
        println!("SWEEP depth={d} instances={instances} ns_per_instance={ns:.1}");
        sweep.push((d, ns));
    }
    let ns = |cell: &str| {
        let r = rows.iter().find(|r| r["cell"] == cell).unwrap();
        r["wall_secs"]["median"].as_f64().unwrap() / r["entities"].as_f64().unwrap() * 1e9
    };
    println!(
        "RATIO advanced_curve_cost_vs_simple={:.3} (spline_heavy {:.0} ns per entity, simple {:.0})",
        ns("spline_heavy") / ns("simple"),
        ns("spline_heavy"),
        ns("simple"),
    );
    let (lo, hi) = sweep
        .iter()
        .fold((f64::MAX, 0f64), |(l, h), (_, n)| (l.min(*n), h.max(*n)));
    println!(
        "SWEEP spread={:.1}% (min {lo:.1}, max {hi:.1} ns per instance)",
        (hi - lo) / lo * 100.0
    );

    let after = host_load().unwrap_or(f64::MAX);
    assert!(
        after < host_cpus() as f64,
        "discarding the run: the one-minute load rose to {after} (it was {before}) against {} cpus",
        host_cpus()
    );
    if let Ok(out) = std::env::var("LIBVIPRS_BENCH_BASELINE_OUT") {
        let map: serde_json::Map<String, Value> = rows
            .iter()
            .map(|r| {
                (
                    format!("provider/{}/{}", r["cell"].as_str().unwrap(), platform),
                    r.clone(),
                )
            })
            .collect();
        std::fs::write(
            &out,
            serde_json::to_string_pretty(&json!({"schema": 1, "rows": map})).unwrap(),
        )
        .unwrap();
    }
    let base = baseline_rows();
    let mut regressions = Vec::new();
    for r in &rows {
        let key = format!("provider/{}/{}", r["cell"].as_str().unwrap(), platform);
        match base.get(&key) {
            Some(b) => regressions.extend(
                provider_regressions(b, r)
                    .into_iter()
                    .map(|p| format!("{key}: {p}")),
            ),
            None => println!("NO BASELINE for {key} in {BASELINE}: this is a first run"),
        }
    }
    assert!(regressions.is_empty(), "{}", regressions.join("\n"));
}
