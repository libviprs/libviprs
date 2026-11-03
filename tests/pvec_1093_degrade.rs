//! Issue #1093 review: a row or region PVec cannot hold is dropped, it does not fail the stream,
//! and the region derivation refuses an input past a named cap. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::components::{ComponentSource, StreamSelection, build_streams};
use libviprs::pvec::regions::{RegionConfig, WallEdge, derive_regions, semantic_stream};
use libviprs::pvec::style::{DashPattern, LineStyleComponent};
use libviprs::pvec::{
    EntityId, EntityStyle, InstancePath, LayerState, OrderingPolicy, ProvenanceStub, RawEntity,
    RawGeometry, order_entities,
};
use pvec_facade::EntityId as PvecId;
use pvec_facade::components::{
    EntityGeometry, LayerStream, LineStyleStream, SemanticClassStream, Stream,
};

/// Minted ids and stubs for lines on the given layers, in the order given.
fn minted(layer_of: impl Fn(usize) -> String, n: usize) -> (Vec<EntityId>, Vec<ProvenanceStub>) {
    let raw: Vec<RawEntity> = (0..n)
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
                layer: layer_of(i),
            },
        })
        .collect();
    let ordered = order_entities("t", "1", &OrderingPolicy::new(), raw).entities;
    (
        ordered.iter().map(|o| o.id).collect(),
        ordered
            .iter()
            .map(|o| o.entity.provenance.clone())
            .collect(),
    )
}

fn sources<'a>(
    ids: &'a [EntityId],
    stubs: &'a [ProvenanceStub],
    styles: &'a [LineStyleComponent],
) -> Vec<ComponentSource<'a>> {
    ids.iter()
        .zip(stubs)
        .enumerate()
        .map(|(i, (id, stub))| ComponentSource {
            id: *id,
            stub,
            style: styles.get(i),
        })
        .collect()
}

fn layer_stream(streams: &[Stream]) -> LayerStream {
    let s = streams
        .iter()
        .find(|s| LayerStream::decode(&s.payload).is_ok())
        .expect("a layer stream");
    LayerStream::decode(&s.payload).unwrap()
}

#[test]
fn issue_1093_a_layer_name_past_the_bound_drops_its_row_and_not_the_streams() {
    let long = "L".repeat(256);
    let (ids, stubs) = minted(
        |i| {
            if i == 1 {
                long.clone()
            } else {
                "A-WALL".into()
            }
        },
        3,
    );
    let layers = [LayerState {
        name: "A-WALL".into(),
        handle: None,
        off: false,
        frozen: false,
        locked: false,
    }];
    let out = build_streams(
        StreamSelection::ALL,
        &ids,
        &layers,
        &sources(&ids, &stubs, &[]),
    )
    .expect("one bad name does not refuse the streams");
    let layer = layer_stream(&out);
    assert_eq!(layer.assignments.len(), 2, "the two good rows stay");
    assert!(layer.names.iter().all(|n| n.len() <= 255));
}

#[test]
fn issue_1093_a_dash_array_past_the_bound_drops_that_style_and_not_the_streams() {
    let (ids, stubs) = minted(|_| "0".into(), 2);
    let styles: Vec<LineStyleComponent> = vec![
        LineStyleComponent::new(
            ids[0],
            None,
            Some(DashPattern::new(vec![1.0; 17], 0.0).unwrap()),
            true,
        )
        .unwrap(),
        LineStyleComponent::new(
            ids[1],
            None,
            Some(DashPattern::new(vec![2.0, 1.0], 0.0).unwrap()),
            true,
        )
        .unwrap(),
    ];
    let out = build_streams(
        StreamSelection::ALL,
        &ids,
        &[],
        &sources(&ids, &stubs, &styles),
    )
    .expect("one bad dash array does not refuse the streams");
    let line = out
        .iter()
        .find_map(|s| LineStyleStream::decode(&s.payload).ok())
        .expect("a line style stream");
    assert_eq!(
        line.assignments.len(),
        1,
        "only the entity with a holdable style"
    );
    assert_eq!(line.styles.len(), 1);
}

#[test]
fn issue_1093_more_layer_names_than_a_table_holds_keep_the_first_ones() {
    let (ids, stubs) = minted(|i| format!("L{i}"), 65_537);
    let out = build_streams(StreamSelection::ALL, &ids, &[], &sources(&ids, &stubs, &[]))
        .expect("an over-full table is cut, not refused");
    let layer = layer_stream(&out);
    assert_eq!(layer.names.len(), 65_536);
    assert_eq!(layer.assignments.len(), 65_536);
}

#[test]
fn issue_1093_more_entities_than_a_stream_holds_keep_the_first_ones() {
    let n = (1usize << 20) + 1;
    let (ids, stubs) = minted(|_| "0".into(), n);
    let out = build_streams(StreamSelection::ALL, &ids, &[], &sources(&ids, &stubs, &[]))
        .expect("an over-long stream is cut, not refused");
    assert_eq!(layer_stream(&out).assignments.len(), 1 << 20);
}

fn config() -> RegionConfig {
    RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    }
}

fn walls(segs: &[([f64; 2], [f64; 2])]) -> Vec<WallEdge> {
    let raw = segs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [a[0], a[1], 0.0],
                end: [b[0], b[1], 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(i as u64 + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: "A-WALL".into(),
            },
        })
        .collect();
    order_entities("t", "1", &OrderingPolicy::new(), raw)
        .entities
        .iter()
        .zip(segs)
        .map(|(o, (a, b))| WallEdge {
            entity: o.id,
            start: *a,
            end: *b,
        })
        .collect()
}

struct Ends(Vec<([f64; 2], [f64; 2])>);

impl EntityGeometry for Ends {
    fn endpoints(&self, entity: PvecId) -> Option<([f64; 2], [f64; 2])> {
        self.0.get(entity.index() as usize).copied()
    }
}

#[test]
fn issue_1093_a_room_with_more_walls_than_a_ring_holds_is_left_out_not_a_failed_stream() {
    let n = 5000usize;
    let pts: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / n as f64;
            [1000.0 * t.cos(), 1000.0 * t.sin()]
        })
        .collect();
    let segs: Vec<_> = (0..n).map(|i| (pts[i], pts[(i + 1) % n])).collect();
    let w = walls(&segs);
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    let ids: Vec<_> = w.iter().map(|e| e.entity).collect();
    assert!(
        semantic_stream(&out, &ids).is_ok(),
        "the region PVec cannot hold is left out as the docs say"
    );
}

#[test]
fn issue_1093_a_region_on_a_derived_segment_never_points_at_an_entity_that_was_not_written() {
    // A 5 x 5 room whose left wall has a 0.2 door gap, so one boundary edge is derived.
    let segs = vec![
        ([0.0, 0.0], [5.0, 0.0]),
        ([5.0, 0.0], [5.0, 5.0]),
        ([5.0, 5.0], [0.0, 5.0]),
        ([0.0, 5.0], [0.0, 2.2]),
        ([0.0, 2.0], [0.0, 0.0]),
    ];
    let w = walls(&segs);
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    assert!(
        !out.derived.is_empty(),
        "the gap is closed by a derived segment"
    );
    let ids: Vec<_> = w.iter().map(|e| e.entity).collect();
    let stream = semantic_stream(&out, &ids).unwrap();
    let s = SemanticClassStream::decode(&stream.payload).unwrap();
    s.validate(&Ends(segs), 1e-9)
        .expect("every entity the stream names exists in the geometry written");
}

#[test]
fn issue_1093_the_derivation_refuses_an_input_past_the_node_cap_with_a_named_reason() {
    let segs: Vec<_> = (0..12_000)
        .map(|i| ([i as f64 * 10.0, 0.0], [i as f64 * 10.0 + 1.0, 0.0]))
        .collect();
    let w = walls(&segs);
    let err = derive_regions(&w, &[], &[], &config()).unwrap_err();
    assert_eq!(err.reason_code(), "regions.node-limit");
}
