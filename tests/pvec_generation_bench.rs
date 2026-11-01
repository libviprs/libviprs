//! N8.5 (#1442, slice of #1064): what producing the artifact costs once the entities are
//! normalized, and how big it is.
//!
//! `#[ignore]`d. The input to every measurement is the same: the kept leaves of a cell, taken
//! through the view profile, ordered, and turned into records. Only `write_geometry`, the one
//! funnel, is timed. It is written three ways (plain, with PVec's spatial directory, and with the
//! directory and Tier 0 blocks), and the size split is the difference between them, because the
//! facade does not expose its directory: geometry is the plain artifact, the index is what the
//! directory adds, Tier 0 is what the blocks add. That sums to the artifact by construction, and
//! it is not a stream-by-stream measurement of the file. No component or provenance stream is
//! written yet (#1093), so those columns are zero.
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
//!     --test pvec_generation_bench -- --ignored --nocapture generation_bench
//! ```

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::time::{Duration, Instant};

use libviprs::CancelToken;
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::writer::{LeafConfig, WriteConfig, write_geometry};
use libviprs::pvec::{
    DwgProvider, EntityId, Normalized2DGeometry, OrderingPolicy, SourceUnit, ViewProfile,
    dwg_basic, dwg_spline, order_entities,
};
use pvec_bench::{
    Stat, baseline_rows, blank_columns, cells, generate, generation_regressions, host_cpus,
    host_load, manifest, platform, require_quiet,
};
use serde_json::{Value, json};

/// The ordered records of a cell: what the writer is handed.
fn records(bytes: &[u8]) -> Vec<(EntityId, Normalized2DGeometry)> {
    let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
    let view = ViewProfile::default_profile()
        .apply(&provider)
        .expect("view");
    let raws = view.included.into_iter().map(|v| v.entity).collect();
    let ordered = order_entities(provider.name(), "", &OrderingPolicy::new(), raws);
    ordered
        .entities
        .into_iter()
        .filter_map(|o| {
            let r = if let Ok(b) = dwg_basic::extract(&o.entity) {
                Normalized2DGeometry::from_curve(b.curve, b.classification.class()).ok()?
            } else {
                Normalized2DGeometry::from_spline(&dwg_spline::extract(&o.entity).ok()?).ok()?
            };
            Some((o.id, r))
        })
        .collect()
}

fn write(
    items: &[(EntityId, Normalized2DGeometry)],
    leaves: Option<LeafConfig>,
) -> Result<Vec<u8>, String> {
    write_geometry(
        &WriteConfig {
            unit: SourceUnit::Meters,
            leaves,
        },
        items.iter().map(|(i, r)| (*i, r)),
        Vec::new(),
        &CancelToken::new(),
    )
    .map(|a| a.bytes)
    .map_err(|e| {
        format!(
            "{e}: {:?}",
            std::error::Error::source(&e).map(ToString::to_string)
        )
    })
}

fn timed(
    items: &[(EntityId, Normalized2DGeometry)],
    leaves: Option<LeafConfig>,
) -> Result<(Vec<u8>, Stat), String> {
    let first = Instant::now();
    let bytes = write(items, leaves)?;
    let reps = (0.25 / first.elapsed().as_secs_f64()).ceil().max(1.0) as usize;
    let samples: Vec<f64> = (0..3)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                write(items, leaves).expect("writes");
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    Ok((bytes, Stat::of(&samples)))
}

fn stat_json(s: Stat) -> Value {
    json!({"median": s.median, "min": s.min, "max": s.max})
}

fn row(cell: &str, bytes: &[u8], platform: &str) -> Value {
    let items = records(bytes);
    let entities = items.len() as u64;
    let (plain, plain_t) = timed(&items, None).expect("a plain write");
    let indexed_cfg = LeafConfig {
        cell_capacity: 256,
        tier0_max_error: None,
    };
    let (indexed, indexed_t) = timed(&items, Some(indexed_cfg)).expect("an indexed write");
    // Tier 0 asks PVec for previews bounded by a fraction of the drawing; two percent of the
    // largest coordinate span is a bound a viewer would use. PVec may refuse it, and then the row
    // says so.
    let span = 600.0;
    let tier_cfg = LeafConfig {
        cell_capacity: 256,
        tier0_max_error: Some(span * 0.02),
    };
    let tier = timed(&items, Some(tier_cfg));
    let again = write(&items, None).expect("a second plain write");
    assert_eq!(
        plain, again,
        "{cell}: two writes of the same cell are not byte identical"
    );
    let (tier_bytes, tier_t, tier_note) = match &tier {
        Ok((b, t)) => (Some(b.len() as u64), Some(*t), Value::Null),
        Err(e) => (None, None, json!(e)),
    };
    let artifact = tier_bytes.unwrap_or(indexed.len() as u64);
    let geometry = plain.len() as u64;
    let index = indexed.len() as u64 - geometry;
    let tier0 = tier_bytes.map(|t| t - indexed.len() as u64);
    assert_eq!(
        geometry + index + tier0.unwrap_or(0),
        artifact,
        "{cell}: the split does not sum to the artifact"
    );
    let source = bytes.len() as f64;
    let wall = tier_t.unwrap_or(indexed_t);
    json!({
        "bench": "generation",
        "cell": cell,
        "entities": entities,
        "wall_secs": stat_json(wall),
        "plain_wall_secs": stat_json(plain_t),
        "indexed_wall_secs": stat_json(indexed_t),
        "tier0_wall_secs": tier_t.map(stat_json),
        "tier0_refused": tier_note,
        "artifact_bytes": artifact,
        "split": {"geometry": geometry, "spatial_index": index, "tier0": tier0, "components": 0, "provenance": 0},
        "tier0_share": tier0.map(|t| t as f64 / artifact as f64),
        "bytes_per_entity": artifact as f64 / entities as f64,
        "artifact_pct_of_source": artifact as f64 / source * 100.0,
        "deterministic": true,
        "unstable": wall.unstable(),
        "platform": platform,
    })
}

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "takes numbers: needs a quiet host, run with --ignored --nocapture"]
fn generation_bench() {
    let platform = platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let before = require_quiet(30, Duration::from_secs(20))
        .unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let m = manifest();
    let all = cells(&m);
    let mut rows = Vec::new();
    for cell in [
        "simple",
        "block_heavy",
        "spline_heavy",
        "paper_space",
        "large_industrial",
    ] {
        let c = &all[cell];
        let bytes = generate(c["generator"].as_str().unwrap(), &c["params"]);
        let r = row(cell, &bytes, &platform);
        println!("ROW {r}");
        rows.push(r);
    }
    for r in &rows {
        let blank = blank_columns(
            r,
            &[
                "bench",
                "cell",
                "entities",
                "wall_secs",
                "artifact_bytes",
                "split",
                "bytes_per_entity",
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
    let key = |r: &Value| format!("generation/{}/{platform}", r["cell"].as_str().unwrap());
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
                generation_regressions(b, r)
                    .into_iter()
                    .map(|p| format!("{}: {p}", key(r))),
            ),
            None => println!("NO BASELINE for {}: this is a first run", key(r)),
        }
    }
    assert!(regressions.is_empty(), "{}", regressions.join("\n"));
}
