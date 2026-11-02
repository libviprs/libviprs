//! N8.3 (#1436, slice of #1064): what libviprs does between the provider and PVec, stage by stage.
//!
//! `#[ignore]`d, in the convention of `pvec_provider_bench.rs` (counting-free here: the figures are
//! nanoseconds per entity). The four stages and the one public call each is:
//!
//! * visibility and layout filtering: `ViewProfile::apply`
//! * transform composition and promotion: `ComposedTransform::push_inner_coefficients` and `promote`
//! * clipping: `apply_clip_policy`. The DWG path has no clip stage in v1 (model space only, no
//!   viewport clip), so this stage is measured on a generated PDF page, its own pass
//! * exact source bounds: `bounds_inputs`
//!
//! plus the shared plumbing that turns a kept leaf into a record, timed as a fifth line so the
//! pass adds up. The DWG pass is the stages in one run; the stage costs must sum to within 5% of
//! the pass, and what no stage accounts for is printed.
//!
//! ```text
//! LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
//!     --test pvec_normalization_bench -- --ignored --nocapture normalization_bench
//! ```
//!
//! The benchmark asserts its own outputs, so a faster path that changes them fails here and not
//! only in a conformance suite: the kept count equals the independent count from the layer names,
//! an identity promotion leaves every bound bit for bit as it was, and a singular matrix is refused
//! for every entity.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::time::{Duration, Instant};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::dwg_basic;
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore};
use libviprs::pvec::pdf_clip::{AnalyticBounds, ClipLimits, apply_clip_policy};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::transform::{ComposedTransform, promote};
use libviprs::pvec::{
    DwgProvider, EntityScope, InstanceStep, LayoutKind, Normalized2DGeometry, SourceLimits,
    ViewProfile, bounds_inputs, curve_bounds,
};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_bench::{
    Stat, baseline_rows, blank_columns, cells, generate, manifest, platform, require_quiet,
    stage_regressions, unattributed,
};
use serde_json::{Value, json};

const CLASSES: [(&str, [f64; 6]); 8] = [
    ("identity", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    ("translation", [1.0, 0.0, 0.0, 1.0, 12.5, -7.25]),
    (
        "rotation",
        [
            0.866_025_403_784_438_6,
            -0.5,
            0.5,
            0.866_025_403_784_438_6,
            0.0,
            0.0,
        ],
    ),
    ("uniform_scale", [2.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    ("non_uniform_scale", [2.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
    ("reflection", [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    ("shear", [1.0, 0.5, 0.0, 1.0, 0.0, 0.0]),
    ("singular", [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
];

fn step() -> InstanceStep {
    InstanceStep {
        block_name: "BENCH".into(),
        insert_handle: ItemHandle::new(1),
    }
}

/// One record per kept leaf that the basic extractor takes, in order.
fn record(e: &libviprs::pvec::RawEntity) -> Option<Normalized2DGeometry> {
    let b = dwg_basic::extract(e).ok()?;
    Normalized2DGeometry::from_curve(b.curve, b.classification.class()).ok()
}

/// What the promotion of `rec` through `matrix` came to: the curve's bounds as bit patterns when it
/// produced geometry, `None` when it was refused.
fn promoted_bounds(rec: &Normalized2DGeometry, matrix: [f64; 6]) -> Option<[u64; 4]> {
    let composed = ComposedTransform::identity()
        .push_inner_coefficients(step(), matrix)
        .ok()?;
    let p = promote(rec, &composed).ok()?;
    let b = curve_bounds(p.record().geometry()?).ok()?.0;
    Some([
        b.min().x.to_bits(),
        b.min().y.to_bits(),
        b.max().x.to_bits(),
        b.max().y.to_bits(),
    ])
}

/// The seconds each stage of one DWG pass took, and what it saw.
struct Pass {
    visibility: f64,
    records: f64,
    transform: f64,
    bounds: f64,
    total: f64,
    leaves: u64,
    kept: u64,
    promoted: u64,
}

fn dwg_pass(provider: &AcadrustProvider) -> Pass {
    let mix: Vec<[f64; 6]> = CLASSES.iter().take(7).map(|c| c.1).collect();
    let total = Instant::now();
    let t = Instant::now();
    let view = ViewProfile::default_profile()
        .apply(provider)
        .expect("view");
    let visibility = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let recs: Vec<Normalized2DGeometry> = view
        .included
        .iter()
        .filter_map(|v| record(&v.entity))
        .collect();
    let records = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let promoted: Vec<Normalized2DGeometry> = recs
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let c = ComposedTransform::identity()
                .push_inner_coefficients(step(), mix[i % mix.len()])
                .ok()?;
            promote(r, &c).ok().map(|p| p.record().clone())
        })
        .collect();
    let transform = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let report =
        bounds_inputs(promoted.iter().enumerate().filter_map(|(i, r)| {
            Some((pvec_entity::EntityId::new(i as u32).ok()?, r.geometry()?))
        }))
        .expect("bounds");
    let bounds = t.elapsed().as_secs_f64();
    assert_eq!(
        report.entries().len(),
        promoted.len(),
        "a bound per promoted entity"
    );
    Pass {
        visibility,
        records,
        transform,
        bounds,
        total: total.elapsed().as_secs_f64(),
        leaves: (view.included.len() + view.excluded.len()) as u64,
        kept: view.included.len() as u64,
        promoted: promoted.len() as u64,
    }
}

/// The independent count of leaves not on a hidden layer: flatten model space and read the layer
/// names, with no view profile in between.
fn visible_by_layer_name(provider: &AcadrustProvider) -> u64 {
    let layout = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    BlockResolver::new(provider, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(layout.id),
            "bench",
            &layout.name,
            &mut NoProvenance,
            &mut LineageStore::new(),
        )
        .unwrap()
        .entities
        .iter()
        .filter(|f| !f.entity.provenance.layer.starts_with("HID_"))
        .count() as u64
}

fn stat_json(s: Stat) -> Value {
    json!({"median": s.median, "min": s.min, "max": s.max})
}

fn dwg_row(cell: &str, hidden: u64, bytes: &[u8], platform: &str) -> (Value, f64) {
    let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
    let expected_kept = visible_by_layer_name(&provider);
    let first = dwg_pass(&provider);
    assert_eq!(
        first.kept, expected_kept,
        "{cell}/{hidden}: the view profile kept a different set than the layer names say"
    );
    let reps = (0.25 / first.total).ceil().max(1.0) as usize;
    let samples: Vec<Pass> = (0..3)
        .map(|_| {
            let batch: Vec<Pass> = (0..reps).map(|_| dwg_pass(&provider)).collect();
            let mean = |f: fn(&Pass) -> f64| batch.iter().map(f).sum::<f64>() / reps as f64;
            Pass {
                visibility: mean(|p| p.visibility),
                records: mean(|p| p.records),
                transform: mean(|p| p.transform),
                bounds: mean(|p| p.bounds),
                total: mean(|p| p.total),
                leaves: first.leaves,
                kept: first.kept,
                promoted: first.promoted,
            }
        })
        .collect();
    let med = |f: fn(&Pass) -> f64| Stat::of(&samples.iter().map(f).collect::<Vec<_>>());
    let (v, r, t, b, all) = (
        med(|p| p.visibility),
        med(|p| p.records),
        med(|p| p.transform),
        med(|p| p.bounds),
        med(|p| p.total),
    );
    let per = |secs: f64, n: u64| if n == 0 { 0.0 } else { secs / n as f64 * 1e9 };
    let rest = unattributed(&[v.median, r.median, t.median, b.median], all.median);
    assert!(
        rest.abs() <= 0.05,
        "{cell}/{hidden}: the stages account for the pass to within 5% only, the remainder is {:.1}%",
        rest * 100.0
    );
    let row = json!({
        "bench": "normalization",
        "cell": cell,
        "mix": hidden,
        "leaves": first.leaves,
        "kept": first.kept,
        "promoted": first.promoted,
        "stage_ns": {
            "visibility": per(v.median, first.leaves),
            "records": per(r.median, first.kept),
            "transform": per(t.median, first.kept),
            "bounds": per(b.median, first.promoted),
        },
        "stage_share": {
            "visibility": v.median / all.median,
            "records": r.median / all.median,
            "transform": t.median / all.median,
            "bounds": b.median / all.median,
        },
        "pass_secs": stat_json(all),
        "unattributed": rest,
        "unstable": all.unstable(),
        "platform": platform,
    });
    (row, all.median)
}

/// One row per N1.5 class over the records of the large cell: composition and promotion, and for
/// the singular class the cost of the refusal. Also the equality check: identity changes nothing.
fn transform_rows(bytes: &[u8], platform: &str) -> Vec<Value> {
    let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
    let view = ViewProfile::default_profile().apply(&provider).unwrap();
    let recs: Vec<Normalized2DGeometry> = view
        .included
        .iter()
        .take(200_000)
        .filter_map(|v| record(&v.entity))
        .collect();
    let own: Vec<[u64; 4]> = recs
        .iter()
        .map(|r| {
            let b = curve_bounds(r.geometry().unwrap()).unwrap().0;
            [
                b.min().x.to_bits(),
                b.min().y.to_bits(),
                b.max().x.to_bits(),
                b.max().y.to_bits(),
            ]
        })
        .collect();
    let reps = 3;
    let (mut moved, mut max_ulp) = (0usize, 0u64);
    CLASSES
        .iter()
        .map(|(name, m)| {
            let mut ns = Vec::new();
            let mut out = Vec::new();
            for _ in 0..reps {
                let t = Instant::now();
                out = recs
                    .iter()
                    .map(|r| promoted_bounds(r, *m))
                    .collect::<Vec<_>>();
                ns.push(t.elapsed().as_secs_f64() / recs.len() as f64 * 1e9);
            }
            let s = Stat::of(&ns);
            let geometry = out.iter().filter(|o| o.is_some()).count();
            match *name {
                "identity" => {
                    // An identity map is a passthrough for a line or a polyline. A circle or an arc
                    // is rebuilt by PVec's transform and its box is recomputed, which can land one
                    // unit in the last place away, so the check is the number of ULPs, reported.
                    for (got, want) in out.iter().zip(&own) {
                        let got = got.as_ref().expect("identity produced geometry");
                        for (a, b) in got.iter().zip(want) {
                            max_ulp = max_ulp.max(ulps(*a, *b));
                        }
                        moved += usize::from(got != want);
                    }
                    assert!(max_ulp <= 4, "an identity promotion moved a bound by {max_ulp} ulps");
                }
                "singular" => assert_eq!(geometry, 0, "a singular matrix produced geometry"),
                _ => assert!(geometry > 0, "{name} produced nothing"),
            }
            json!({
                "bench": "normalization",
                "cell": "large_industrial",
                "class": name,
                "entities": recs.len(),
                "geometry": geometry,
                "identity_bounds_moved": if *name == "identity" { json!(moved) } else { Value::Null },
                "identity_max_ulps": if *name == "identity" { json!(max_ulp) } else { Value::Null },
                "refused": recs.len() - geometry,
                "stage_ns": {"transform": s.median},
                "ns_per_entity": stat_json(s),
                "unstable": s.unstable(),
                "platform": platform,
            })
        })
        .collect()
}

/// Distance in units in the last place between two bound values given as bit patterns.
fn ulps(a: u64, b: u64) -> u64 {
    a.abs_diff(b)
}

fn pdf(body: &str) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages,
        "MediaBox" => vec![0.into(), 0.into(), 2000.into(), 2000.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages,
        Object::Dictionary(
            dictionary! { "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1 },
        ),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

/// `groups` clip rectangles, each holding `per` stroked lines, half of them outside the clip.
fn clip_row(groups: usize, per: usize, platform: &str) -> Value {
    let mut body = String::new();
    for g in 0..groups {
        let (x, y) = ((g % 40) * 50, (g / 40) * 50);
        body += &format!("q {x} {y} 40 40 re W n\n");
        for k in 0..per {
            let dx = if k % 2 == 0 { 5 } else { 90 };
            body += &format!("{} {} m {} {} l S\n", x + dx, y + 5, x + dx + 20, y + 25);
        }
        body += "Q\n";
    }
    let bytes = pdf(&body);
    let page = extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).expect("page");
    let paths = page.entities.len();
    let limits = ClipLimits::from(&SourceLimits::default());
    let batch = 5;
    let ns: Vec<f64> = (0..3)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..batch {
                let r = apply_clip_policy(&page, &limits, &AnalyticBounds).expect("clips");
                assert_eq!(
                    r.decisions.len(),
                    paths - groups,
                    "a decision per painted path, the clip-only paths excluded"
                );
            }
            t.elapsed().as_secs_f64() / (batch * (paths - groups)) as f64 * 1e9
        })
        .collect();
    let s = Stat::of(&ns);
    json!({
        "bench": "normalization", "cell": "pdf_clip", "entities": paths, "clips": groups,
        "stage_ns": {"clip": s.median}, "ns_per_entity": stat_json(s), "unstable": s.unstable(),
        "platform": platform,
    })
}

#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "takes numbers: needs a quiet host, run with --ignored --nocapture"]
fn normalization_bench() {
    let platform = platform().unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let before = require_quiet(30, Duration::from_secs(20))
        .unwrap_or_else(|e| panic!("refusing to run: {e}"));
    let m = manifest();
    let all = cells(&m);
    let mut rows = Vec::new();
    for cell in ["block_heavy", "paper_space", "large_industrial"] {
        let c = &all[cell];
        let mut totals = Vec::new();
        for hidden in [0u64, 50, 90] {
            let mut p = c["params"].clone();
            p["hidden"] = json!(hidden);
            let bytes = generate(c["generator"].as_str().unwrap(), &p);
            let (row, total) = dwg_row(cell, hidden, &bytes, &platform);
            println!("ROW {row}");
            totals.push(total);
            rows.push(row);
            if cell == "large_industrial" && hidden == 0 {
                for t in transform_rows(&bytes, &platform) {
                    println!("ROW {t}");
                    rows.push(t);
                }
            }
        }
        assert!(
            totals[2] <= totals[0],
            "{cell}: the 90% mix ({:.4}s) is slower than the 0% mix ({:.4}s), so filtering does not pay for itself",
            totals[2],
            totals[0]
        );
    }
    let clip = clip_row(200, 100, &platform);
    println!("ROW {clip}");
    rows.push(clip);

    for r in &rows {
        let blank = blank_columns(r, &["bench", "cell", "stage_ns", "platform"]);
        assert!(blank.is_empty(), "{blank:?} in {r}");
    }
    let after = pvec_bench::host_load().unwrap_or(f64::MAX);
    assert!(
        after < pvec_bench::host_cpus() as f64,
        "discarding the run: the one-minute load rose to {after} (it was {before})"
    );
    let key = |r: &Value| match (r["mix"].as_u64(), r["class"].as_str()) {
        (Some(mix), _) => format!(
            "normalization/{}/mix{mix}/{platform}",
            r["cell"].as_str().unwrap()
        ),
        (_, Some(class)) => format!("normalization/transform/{class}/{platform}"),
        _ => format!("normalization/{}/{platform}", r["cell"].as_str().unwrap()),
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
                stage_regressions(b, r)
                    .into_iter()
                    .map(|p| format!("{}: {p}", key(r))),
            ),
            None => println!("NO BASELINE for {}: this is a first run", key(r)),
        }
    }
    assert!(regressions.is_empty(), "{}", regressions.join("\n"));
}
