//! What the component streams cost the geometry write (issue #1093).
//!
//! `cargo run --release --features pvec-write --example pvec_components_cost -- 500000`
//!
//! Writes `n` two-point polylines twice, geometry only and then with the layer, line style and
//! provenance streams, and prints entities per second for the write alone (the streams are built
//! outside it, off the geometry path) and for build plus write, plus the artifact sizes. Best of
//! three runs each.

use std::time::Instant;

use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::components::{ComponentSource, StreamSelection, build_streams};
use libviprs::pvec::style::{DashPattern, LineStyleComponent};
use libviprs::pvec::writer::{WriteConfig, write_geometry};
use libviprs::pvec::{
    EntityStyle, FidelityClass, InstancePath, Normalized2DGeometry, OrderingPolicy, ProvenanceStub,
    RawEntity, RawGeometry, SourceUnit, order_entities,
};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(500_000);
    let raw = (0..n)
        .map(|i| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [i as f64, 0.0, 0.0],
                end: [i as f64 + 1.0, 1.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(i as u64 + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: format!("L{}", i % 64),
            },
        })
        .collect();
    let ordered = order_entities("bench", "1", &OrderingPolicy::new(), raw).entities;
    let ids: Vec<_> = ordered.iter().map(|o| o.id).collect();
    let stubs: Vec<_> = ordered
        .iter()
        .map(|o| o.entity.provenance.clone())
        .collect();
    let geoms: Vec<_> = (0..n)
        .map(|i| {
            Normalized2DGeometry::polyline(
                &[[i as f64, 0.0], [i as f64 + 1.0, 1.0]],
                FidelityClass::SupportedExact,
            )
            .unwrap()
        })
        .collect();
    let dash = DashPattern::new(vec![3.0, 1.0], 0.0).unwrap();
    let styles: Vec<_> = ids
        .iter()
        .map(|id| LineStyleComponent::new(*id, Some(0.25), Some(dash.clone()), true).unwrap())
        .collect();
    let rows: Vec<_> = ids
        .iter()
        .zip(&stubs)
        .zip(&styles)
        .map(|((id, stub), style)| ComponentSource {
            id: *id,
            stub,
            style: Some(style),
        })
        .collect();
    let config = WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: None,
    };
    let run = |select: StreamSelection| {
        let mut best = (f64::MAX, f64::MAX, 0usize);
        for _ in 0..3 {
            let t = Instant::now();
            let streams = build_streams(select, &ids, &[], &rows).unwrap();
            let built = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let art = write_geometry(
                &config,
                ids.iter().copied().zip(&geoms),
                streams,
                &CancelToken::new(),
            )
            .unwrap();
            let wrote = t.elapsed().as_secs_f64();
            if built + wrote < best.0 + best.1 {
                best = (built, wrote, art.bytes.len());
            }
        }
        best
    };
    for (name, select) in [
        ("geometry only", StreamSelection::NONE),
        (
            "provenance",
            StreamSelection {
                provenance: true,
                ..StreamSelection::NONE
            },
        ),
        ("all streams", StreamSelection::ALL),
    ] {
        let (built, wrote, bytes) = run(select);
        println!(
            "{name:14} build {built:.3}s write {wrote:.3}s ({:.0} ent/s) total {:.0} ent/s artifact {bytes} bytes",
            n as f64 / wrote,
            n as f64 / (built + wrote)
        );
    }
}
