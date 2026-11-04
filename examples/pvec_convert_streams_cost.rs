//! What the component streams cost a whole conversion (issues #1093 and #1526).
//!
//! `cargo run --release --features pvec-convert --example pvec_convert_streams_cost -- 500000`
//!
//! Generates the `large_industrial` synthetic drawing (the benchmark corpus cell, `n` entities in
//! model space) and converts it with no stream, with the provenance stream only and with every
//! stream (layer, line style and provenance, plus the region step), best of two runs each. Prints
//! entities per second, the peak of the live heap above the drawing (the counting allocator, not
//! RSS) and the artifact size, and the extra heap per source entity against the run with no stream.
//! The drawing is large (about 3 GiB resident at a million entities), so run it on a build host and
//! not on a laptop.

#[path = "../tests/common/pvec_alloc.rs"]
mod alloc;
#[path = "../tests/common/pvec_bench.rs"]
mod bench;

use std::time::Instant;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::regions::RegionConfig;
use libviprs::pvec_convert::PvecConversion;

#[global_allocator]
static ALLOC: alloc::Counting = alloc::Counting;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(500_000);
    let params = serde_json::json!({ "entities": n, "parts": 8 });
    let bytes = bench::generate("large_industrial", &params);
    let dir = std::env::temp_dir().join(format!("libviprs-streams-cost-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("o.pvec");
    let regions = RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    };
    let provenance_only = StreamSelection {
        layer: false,
        line_style: false,
        provenance: true,
    };
    let cases = [
        ("off", StreamSelection::NONE, None),
        ("provenance", provenance_only, None),
        ("all", StreamSelection::ALL, None),
        ("all+regions", StreamSelection::ALL, Some(regions)),
    ];
    let mut base_peak = 0u64;
    for (name, select, regions) in cases {
        let mut best: Option<(f64, u64, u64, u64)> = None;
        for _ in 0..2 {
            let m = alloc::mark();
            let t = Instant::now();
            let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
                .unit(SourceUnit::Meters)
                .components(select)
                .regions(regions)
                .convert()
                .expect("the drawing converts");
            let secs = t.elapsed().as_secs_f64();
            let (peak, _) = alloc::since(m);
            let size = std::fs::metadata(&dest).map_or(0, |m| m.len());
            let rate = out.report.totals.encountered as f64 / secs;
            if best.is_none_or(|b| rate > b.0) {
                best = Some((rate, peak, size, out.report.totals.encountered));
            }
        }
        let (rate, peak, size, met) = best.unwrap();
        if name == "off" {
            base_peak = peak;
        }
        let extra = peak.saturating_sub(base_peak);
        println!(
            "{name:12} {rate:>10.0} ent/s  peak {:>6} MiB  +{:>5.1}% over off  {:>6.1} B/entity  artifact {} B  ({met} entities)",
            peak >> 20,
            100.0 * extra as f64 / base_peak.max(1) as f64,
            extra as f64 / met.max(1) as f64,
            size,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
