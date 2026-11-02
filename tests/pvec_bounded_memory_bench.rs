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
//! Two strategies are measured: the direct in-memory one and the spill (`PvecConversion::spill`). The
//! two-pass strategy has no code and prints a row saying so. The spill bytes are the peak size of
//! the files in the spill root, sampled while the conversion runs, since the facade does not report
//! them. The ceilings and the boundedness check are reported against the criterion, not asserted:
//! today they do not hold, and the document says so.
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
use libviprs::pvec::writer::{LeafConfig, SpillConfig};
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
    let spill_root = dir.path().join("spill");
    std::fs::create_dir(&spill_root).unwrap();
    let budget = std::env::var("LIBVIPRS_BM_BUDGET")
        .ok()
        .map(|b| b.parse::<u64>().unwrap());
    // The spill is only bounded for an indexed write, so the spill strategy sets leaves.
    let mut conversion =
        PvecConversion::new(CadSource::Bytes(&bytes), &dest).unit(SourceUnit::Meters);
    if let Some(budget) = budget {
        conversion = conversion
            .leaves(LeafConfig {
                cell_capacity: 256,
                tier0_max_error: None,
            })
            .spill(SpillConfig {
                root: spill_root.clone(),
                budget,
            });
    }
    // The most the spill files held at once, sampled while the conversion runs.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let (stop, root) = (stop.clone(), spill_root.clone());
        std::thread::spawn(move || {
            let (mut peak, mut files) = (0u64, 0u64);
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let sizes: Vec<u64> = walk(&root);
                peak = peak.max(sizes.iter().sum());
                files = files.max(sizes.len() as u64);
                std::thread::sleep(Duration::from_millis(5));
            }
            (peak, files)
        })
    };
    let rss_reset = reset_peak_rss();
    let m = pvec_alloc::mark();
    let t = Instant::now();
    let out = conversion.convert().expect("converts");
    let wall = t.elapsed().as_secs_f64();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (spilled_peak, spill_files) = sampler.join().unwrap();
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
            "spilled_peak_on_disk": spilled_peak,
            "spill_files_peak": spill_files,
        })
    );
}

/// The sizes of the regular files under `dir`.
fn walk(dir: &Path) -> Vec<u64> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            match e.metadata() {
                Ok(m) if m.is_dir() => out.extend(walk(&e.path())),
                Ok(m) => out.push(m.len()),
                Err(_) => {}
            }
        }
    }
    out
}

fn variant(dir: &Path, cell: &str, scale: f64, platform: &str, budget: Option<u64>) -> Value {
    let f = file(dir, cell, scale);
    if !f.exists() {
        child(
            "bounded_memory_gen",
            &[
                ("LIBVIPRS_BM_CELL", cell.to_string()),
                ("LIBVIPRS_BM_SCALE", scale.to_string()),
                ("LIBVIPRS_BM_FILE", f.display().to_string()),
            ],
        );
    }
    let mut envs = vec![("LIBVIPRS_BM_FILE", f.display().to_string())];
    if let Some(b) = budget {
        envs.push(("LIBVIPRS_BM_BUDGET", b.to_string()));
    }
    let runs: Vec<Value> = (0..3)
        .map(|_| {
            let text = child("bounded_memory_child", &envs);
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
    json!({
        "bench": "bounded_memory",
        "strategy": if budget.is_some() { "spill" } else { "direct" },
        "spill_budget": budget,
        "cell": cell,
        "scale": scale,
        "entities": entities,
        "artifact_bytes": runs[0]["artifact_bytes"],
        "wall_secs": {"median": wall.median, "min": wall.min, "max": wall.max},
        "tracked_peak": tracked,
        "peak_rss_bytes": rss,
        "allocated_bytes": runs[0]["allocated_bytes"],
        "rss_is_per_run": runs[0]["rss_is_per_run"],
        // Peak bytes on disk in the spill root, sampled every 5 ms: the facade does not report what it
        // spilled, so this is a lower bound on the bytes written and the files are the peak open at once.
        "bytes_spilled": runs.iter().filter_map(|r| r["spilled_peak_on_disk"].as_u64()).max().unwrap_or(0),
        "bytes_reread": if budget.is_some() { json!("not exposed by the facade") } else { json!(0) },
        "spill_count": runs.iter().filter_map(|r| r["spill_files_peak"].as_u64()).max().unwrap_or(0),
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
    const MIB: u64 = 1 << 20;
    for budget in [None, Some(64 * MIB)] {
        for cell in ["large_industrial", "spline_heavy", "block_heavy"] {
            for scale in [0.5, 1.0, 2.0] {
                let row = variant(dir.path(), cell, scale, &platform, budget);
                println!("ROW {row}");
                rows.push(row);
            }
        }
    }
    // The knob is the budget: the same drawing at a doubled and a quadrupled budget.
    for budget in [128 * MIB, 256 * MIB] {
        let mut row = variant(dir.path(), "large_industrial", 1.0, &platform, Some(budget));
        row["strategy"] = json!("spill_budget");
        println!("ROW {row}");
        rows.push(row);
    }
    let mut at64 = rows
        .iter()
        .find(|r| r["strategy"] == "spill" && r["cell"] == "large_industrial" && r["scale"] == 1.0)
        .unwrap()
        .clone();
    at64["strategy"] = json!("spill_budget");
    rows.push(at64);
    // The two-pass strategy has no code: the spill replaced the need for a second pass over the input.
    println!(
        "ROW {}",
        json!({"bench": "bounded_memory", "strategy": "two_pass", "status": "no code", "platform": platform})
    );
    // Boundedness: doubling the entity count at a fixed configuration should move the tracked peak
    // by under 10%. Reported as the measured growth per cell.
    for (strategy, cell) in ["direct", "spill"]
        .into_iter()
        .flat_map(|st| ["large_industrial", "spline_heavy", "block_heavy"].map(move |c| (st, c)))
    {
        let at = |s: f64| {
            rows.iter()
                .find(|r| r["strategy"] == strategy && r["cell"] == cell && r["scale"] == s)
                .unwrap()["tracked_peak"]
                .as_f64()
                .unwrap()
        };
        println!(
            "BOUNDEDNESS {strategy} {cell}: doubling the entities moved the tracked peak by {:.1}% (criterion: under 10%)",
            (at(2.0) / at(1.0) - 1.0) * 100.0
        );
    }
    for r in &rows {
        if r.get("status").is_some() {
            continue;
        }
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
    let key = |r: &Value| match r["strategy"].as_str().unwrap() {
        "spill_budget" => format!(
            "bounded_memory/spill_budget/{}/b{}MiB/{platform}",
            r["cell"].as_str().unwrap(),
            r["spill_budget"].as_u64().unwrap() / MIB
        ),
        strategy => format!(
            "bounded_memory/{strategy}/{}/x{}/{platform}",
            r["cell"].as_str().unwrap(),
            r["scale"]
        ),
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
