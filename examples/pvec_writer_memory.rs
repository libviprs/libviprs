//! Memory of the PVec writer funnel against the entity count (issue #1314, N6.3).
//!
//! `cargo run --release --features pvec-write --example pvec_writer_memory -- 1000000 [indexed]`
//!
//! Builds `n` two-point polylines (and the ids and geometry for them) first, then resets the
//! kernel's peak-RSS mark and runs `write_geometry`, so the figure printed is what the funnel and
//! PVec's writer add on top of the inputs: the resident set before the call, the peak during it,
//! and the size of the artifact. Linux only (it reads `/proc/self`).

use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::writer::{LeafConfig, WriteConfig, write_geometry};
use libviprs::pvec::{
    EntityStyle, FidelityClass, InstancePath, Normalized2DGeometry, OrderingPolicy, ProvenanceStub,
    RawEntity, RawGeometry, SourceUnit, order_entities,
};

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let indexed = args.next().as_deref() == Some("indexed");

    // The ids come from the ordering pass, which is the only way to make one, from the leanest
    // entity there is. Its entities are dropped as soon as the ids are taken.
    let ids: Vec<_> = order_entities(
        "probe",
        "1",
        &OrderingPolicy::new(),
        (0..n)
            .map(|i| RawEntity {
                style: EntityStyle::absent(),
                geometry: RawGeometry::Line {
                    start: [0.0; 3],
                    end: [1.0, 0.0, 0.0],
                },
                provenance: ProvenanceStub {
                    handle: Some(ItemHandle::new(i as u64 + 1)),
                    class_name: String::new(),
                    block_path: InstancePath::root(),
                    layer: String::new(),
                },
            })
            .collect(),
    )
    .entities
    .into_iter()
    .map(|o| o.id)
    .collect();
    let geoms: Vec<_> = (0..n)
        .map(|i| {
            let (x, y) = ((i % 1000) as f64, (i / 1000) as f64);
            Normalized2DGeometry::polyline(
                &[[x, y], [x + 0.5, y + 0.25]],
                FidelityClass::SupportedExact,
            )
            .unwrap()
        })
        .collect();

    // "5" resets the peak resident-set mark (VmHWM) to the current one.
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
    let before = status_kb("VmRSS:");
    let config = WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: indexed.then_some(LeafConfig {
            cell_capacity: 1024,
            tier0_max_error: Some(0.05),
        }),
    };
    let art = write_geometry(
        &config,
        ids.into_iter().zip(&geoms),
        Vec::new(),
        &CancelToken::new(),
    )
    .unwrap();
    let peak = status_kb("VmHWM:");
    println!(
        "entities={n} indexed={indexed} rss_before_kb={before} peak_kb={peak} added_mb={:.1} artifact_bytes={} bytes_per_entity_added={:.0}",
        (peak - before) as f64 / 1024.0,
        art.bytes.len(),
        (peak - before) as f64 * 1024.0 / n as f64,
    );
}
