//! Issue #1606: the provenance stream carries one row per entity, so a pick on any member of a
//! block reference finds its handle, not only the first curve of the INSERT.
//! Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::components::{ComponentSource, StreamSelection, build_streams};
use libviprs::pvec::{
    EntityStyle, InstancePath, OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry,
    order_entities,
};
use pvec_facade::components::ProvenanceStream;

fn provenance(inserts: &[u64]) -> (Vec<libviprs::pvec::EntityId>, ProvenanceStream) {
    // One root line (handle 1) and, per INSERT, three block lines (handles 11 to 13).
    let raw = {
        let line = |handle: u64, path: InstancePath, x: f64| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [x, 0.0, 0.0],
                end: [x + 1.0, 1.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(handle)),
                class_name: "LINE".into(),
                block_path: path,
                layer: "0".into(),
            },
        };
        let mut raw = vec![line(1, InstancePath::root(), 0.0)];
        for (n, insert) in inserts.iter().enumerate() {
            for k in 0..3u64 {
                let path = InstancePath::root().child("BLK", ItemHandle::new(*insert));
                raw.push(line(11 + k, path, 10.0 * (n as f64 + 1.0) + k as f64 * 3.0));
            }
        }
        raw
    };
    let ordered = order_entities("t", "1", &OrderingPolicy::new(), raw).entities;
    let ids: Vec<_> = ordered.iter().map(|o| o.id).collect();
    let rows: Vec<_> = ordered
        .iter()
        .map(|o| ComponentSource {
            id: o.id,
            stub: &o.entity.provenance,
            style: None,
            viewport: None,
            paint: None,
        })
        .collect();
    let built = build_streams(
        StreamSelection {
            provenance: true,
            ..StreamSelection::NONE
        },
        &ids,
        &[],
        &rows,
    )
    .unwrap()
    .streams;
    let stream = ProvenanceStream::decode(&built[0].payload).unwrap();
    (ids, stream)
}

#[test]
fn issue_1606_every_member_of_an_insert_has_a_provenance_row() {
    let (ids, p) = provenance(&[100]);
    // One root line and three block lines: four entities, four rows.
    assert_eq!(p.records.len(), ids.len(), "{:?}", p.records);
    let mut rows_per_entity: Vec<usize> = (0..ids.len())
        .map(|n| {
            p.records
                .iter()
                .filter(|r| r.entity.index() as usize == n)
                .count()
        })
        .collect();
    rows_per_entity.dedup();
    assert_eq!(rows_per_entity, [1], "each entity has exactly one row");
}

#[test]
fn issue_1606_member_rows_carry_the_insert_handle_and_root_rows_their_own() {
    let (_, p) = provenance(&[100]);
    let mut handles: Vec<Vec<u8>> = p.records.iter().map(|r| r.handle.clone()).collect();
    handles.sort();
    let mut want: Vec<Vec<u8>> = [1u64, 100, 100, 100]
        .iter()
        .map(|h| h.to_be_bytes().to_vec())
        .collect();
    want.sort();
    assert_eq!(handles, want);
    assert!(p.records.iter().all(|r| r.source_kind == 1));
}

#[test]
fn issue_1606_two_inserts_of_one_block_keep_their_own_handles_per_member() {
    let (ids, p) = provenance(&[100, 200]);
    assert_eq!(p.records.len(), ids.len(), "{:?}", p.records);
    for h in [100u64, 200] {
        let n = p
            .records
            .iter()
            .filter(|r| r.handle == h.to_be_bytes().to_vec())
            .count();
        assert_eq!(n, 3, "insert {h}: one row per member line");
    }
}
