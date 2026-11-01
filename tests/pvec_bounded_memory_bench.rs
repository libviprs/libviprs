//! N8.4 (#1440, slice of #1064): what a conversion holds in memory, and how that moves with the
//! drawing.
//!
//! `#[ignore]`d. The parent test runs every variant in a child process of this same binary, so one
//! run's RSS cannot leak into the next, and a separate child writes the drawing to disk first, so
//! the drawing's own construction is not in the conversion's peak. Two bases are reported because
//! they answer different questions: the counting allocator's live-heap peak above where the
//! conversion started (the drawing's bytes are already held and are excluded), and the kernel's
//! `VmHWM` for the whole child.
//!
//! The spool and the streaming sink are not built (pvec#235), so the one strategy there is to
//! measure is the direct in-memory one. The two-pass and external-sort rows are printed as "not
//! built". Bytes spilled, bytes re-read and spill count are therefore zero by construction for the
//! strategy that exists, and the ceilings and the boundedness check are reported against the
//! criterion, not asserted: today they do not hold, and the document says so.
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
//!     --test pvec_bounded_memory_bench -- --ignored --nocapture bounded_memory_bench
//! ```

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_alloc.rs"]
mod pvec_alloc;
#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec_convert::PvecConversion;
use pvec_bench::{
    Stat, baseline_rows, blank_columns, cells, generate, host_cpus, host_load, manifest,
    memory_regressions, peak_rss, platform, require_quiet, reset_peak_rss,
};
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: pvec_alloc::Counting = pvec_alloc::Counting;

/// The ceilings of the criterion: 1.5 GiB process peak RSS and 1.0 GiB tracked allocation peak.
const RSS_CEILING: u64 = 3 << 29;
const TRACKED_CEILING: u64 = 1 << 30;

/// The generator parameters of `cell` scaled by `scale` (a count of 1 is the manifest's).
fn scaled(cell: &str, scale: f64) -> Value {
    let m = manifest();
    let c = &cells(&m)[cell];
    let mut p = c["params"].clone();
    let key = match cell {
        "large_industrial" => "entities",
        "spline_heavy" => "splines",
        "block_heavy" => "roots",
        other => panic!("no scaling rule for {other}"),
    };
    p[key] = json!((p[key].as_f64().unwrap() * scale).round() as u64);
    p
}

fn file(dir: &Path, cell: &str, scale: f64) -> PathBuf {
    dir.join(format!("{cell}-{scale}.dwg"))
}

fn child(test: &str, envs: &[(&str, String)]) -> String {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            test,
            "--nocapture",
            "--test-threads=1",
        ])
        .envs(envs.iter().map(|(k, v)| (*k, v.as_str())))
        .output()
        .expect("spawns itself");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{test} failed: {text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    text
}

fn env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} is not set: this is a child of the benchmark"))
}

/// Child: write one drawing to disk and exit.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "a child of bounded_memory_bench"]
fn bounded_memory_gen() {
    let (cell, scale) = (
        env("LIBVIPRS_BM_CELL"),
        env("LIBVIPRS_BM_SCALE").parse::<f64>().unwrap(),
    );
    let m = manifest();
    let g = cells(&m)[&cell]["generator"].as_str().unwrap().to_string();
    std::fs::write(env("LIBVIPRS_BM_FILE"), generate(&g, &scaled(&cell, scale))).unwrap();
}

/// Child: convert one drawing and print the figures.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "a child of bounded_memory_bench"]
fn bounded_memory_child() {
    let bytes = std::fs::read(env("LIBVIPRS_BM_FILE")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let rss_reset = reset_peak_rss();
    let m = pvec_alloc::mark();
    let t = Instant::now();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("converts");
    let wall = t.elapsed().as_secs_f64();
    let (tracked_peak, allocated) = pvec_alloc::since(m);
    println!(
        "CHILD {}",
        json!({
            "entities": out.report.totals.encountered,
            "written": out.written,
            "artifact_bytes": std::fs::metadata(&dest).unwrap().len(),
            "wall_secs": wall,
            "tracked_peak": tracked_peak,
            "allocated_bytes": allocated,
            "peak_rss_bytes": peak_rss(),
            "rss_is_per_run": rss_reset,
        })
    );
}

fn variant(dir: &Path, cell: &str, scale: f64, platform: &str) -> Value {
    let f = file(dir, cell, scale);
    child(
        "bounded_memory_gen",
        &[
            ("LIBVIPRS_BM_CELL", cell.to_string()),
            ("LIBVIPRS_BM_SCALE", scale.to_string()),
            ("LIBVIPRS_BM_FILE", f.display().to_string()),
        ],
    );
    let runs: Vec<Value> = (0..3)
        .map(|_| {
            let text = child(
                "bounded_memory_child",
                &[("LIBVIPRS_BM_FILE", f.display().to_string())],
            );
            // libtest prints "test name ... " before the child's own output, on the same line.
            let line = text
                .lines()
                .find_map(|l| l.split_once("CHILD ").map(|x| x.1))
                .unwrap_or_else(|| panic!("no CHILD line in the output of the child: {text}"));
            serde_json::from_str(line).unwrap()
        })
        .collect();
    let all = |k: &str| {
        runs.iter()
            .map(|r| r[k].as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let wall = Stat::of(&all("wall_secs"));
    let tracked = runs
        .iter()
        .map(|r| r["tracked_peak"].as_u64().unwrap())
        .max()
        .unwrap();
    let rss = runs
        .iter()
        .filter_map(|r| r["peak_rss_bytes"].as_u64())
        .max();
    let entities = runs[0]["entities"].as_u64().unwrap();
    std::fs::rename(&f, f.with_extension("dwg.done")).ok();
    json!({
        "bench": "bounded_memory",
        "strategy": "direct",
        "cell": cell,
        "scale": scale,
        "entities": entities,
        "artifact_bytes": runs[0]["artifact_bytes"],
        "wall_secs": {"median": wall.median, "min": wall.min, "max": wall.max},
        "tracked_peak": tracked,
        "peak_rss_bytes": rss,
        "allocated_bytes": runs[0]["allocated_bytes"],
        "rss_is_per_run": runs[0]["rss_is_per_run"],
        "bytes_spilled": 0,
        "bytes_reread": 0,
        "spill_count": 0,
        "tracked_bytes_per_entity": tracked as f64 / entities as f64,
        "within_rss_ceiling": rss.is_some_and(|r| r <= RSS_CEILING),
        "within_tracked_ceiling": tracked <= TRACKED_CEILING,
        "unstable": wall.unstable(),
        "platform": platform,
    })
}

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "takes numbers: needs a quiet host, run with --ignored --nocapture"]
fn bounded_memory_bench() {
    let platform = platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let before = require_quiet(30, Duration::from_secs(20))
        .unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let dir = tempfile::tempdir().unwrap();
    let mut rows = Vec::new();
    for cell in ["large_industrial", "spline_heavy", "block_heavy"] {
        for scale in [0.5, 1.0, 2.0] {
            let row = variant(dir.path(), cell, scale, &platform);
            println!("ROW {row}");
            rows.push(row);
        }
    }
    for s in ["two_pass", "external_sort_spool"] {
        println!(
            "ROW {}",
            json!({"bench": "bounded_memory", "strategy": s, "status": "not built (pvec#235)", "platform": platform})
        );
    }
    // Boundedness: doubling the entity count at a fixed configuration should move the tracked peak
    // by under 10%. Reported as the measured growth per cell.
    for cell in ["large_industrial", "spline_heavy", "block_heavy"] {
        let at = |s: f64| {
            rows.iter()
                .find(|r| r["cell"] == cell && r["scale"] == s)
                .unwrap()["tracked_peak"]
                .as_f64()
                .unwrap()
        };
        println!(
            "BOUNDEDNESS {cell}: doubling the entities moved the tracked peak by {:.1}% (criterion: under 10%)",
            (at(2.0) / at(1.0) - 1.0) * 100.0
        );
    }
    for r in &rows {
        let blank = blank_columns(
            r,
            &[
                "bench",
                "strategy",
                "cell",
                "entities",
                "wall_secs",
                "tracked_peak",
                "peak_rss_bytes",
                "bytes_spilled",
                "bytes_reread",
                "spill_count",
                "platform",
            ],
        );
        assert!(blank.is_empty(), "{blank:?} in {r}");
    }
    let after = host_load().unwrap_or(f64::MAX);
    assert!(
        after < host_cpus() as f64,
        "discarding the run: the one-minute load rose to {after} (it was {before})"
    );
    let key = |r: &Value| {
        format!(
            "bounded_memory/{}/{}/x{}/{platform}",
            r["strategy"].as_str().unwrap(),
            r["cell"].as_str().unwrap(),
            r["scale"]
        )
    };
    if let Ok(out) = std::env::var("LIBVIPRS_BENCH_BASELINE_OUT") {
        let map: serde_json::Map<String, Value> =
            rows.iter().map(|r| (key(r), r.clone())).collect();
        std::fs::write(
            &out,
            serde_json::to_string_pretty(&json!({"schema": 1, "rows": map})).unwrap(),
        )
        .unwrap();
    }
    let base = baseline_rows();
    let mut regressions = Vec::new();
    for r in &rows {
        match base.get(key(r)) {
            Some(b) => regressions.extend(
                memory_regressions(b, r)
                    .into_iter()
                    .map(|p| format!("{}: {p}", key(r))),
            ),
            None => println!("NO BASELINE for {}: this is a first run", key(r)),
        }
    }
    assert!(regressions.is_empty(), "{}", regressions.join("\n"));
}
