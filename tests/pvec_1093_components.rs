//! Issue #1093 (N6.6): the layer, line style and provenance component streams.
//! Needs `--features pvec-write`.
//!
//! The named fixture is `flattened_blocks`: two root lines, then two instances (INSERT 100 and
//! INSERT 200) of a block of three lines. Eight curves come out of four source entities.

#![cfg(feature = "pvec-write")]

use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::components::{ComponentSource, StreamSelection, build_streams};
use libviprs::pvec::style::{DashPattern, LineStyleComponent};
use libviprs::pvec::writer::{GeometryArtifact, WriteConfig, write_geometry};
use libviprs::pvec::{
    EntityId, EntityStyle, FidelityClass, InstancePath, LayerState, Normalized2DGeometry,
    OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry, SourceUnit, order_entities,
};
use pvec_facade::PvecReader;
use pvec_facade::components::{
    LAYER_FROZEN, LAYER_OFF, LayerStream, LineStyleStream, ProvenanceStream, Stream, StrokeWidth,
};

struct Fixture {
    ids: Vec<EntityId>,
    stubs: Vec<ProvenanceStub>,
    geoms: Vec<Normalized2DGeometry>,
}

fn flattened_blocks() -> Fixture {
    let mut raw = Vec::new();
    let mut push = |handle: u64, layer: &str, path: InstancePath, x: f64| {
        raw.push(RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [x, 0.0, 0.0],
                end: [x + 1.0, 1.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(handle)),
                class_name: "LINE".into(),
                block_path: path,
                layer: layer.into(),
            },
        });
    };
    push(1, "A-WALL", InstancePath::root(), 0.0);
    push(2, "0", InstancePath::root(), 10.0);
    for (insert, x) in [(100u64, 20.0), (200, 40.0)] {
        for k in 0..3u64 {
            let path = InstancePath::root().child("BLK", ItemHandle::new(insert));
            push(11 + k, "A-DOOR", path, x + k as f64 * 3.0);
        }
    }
    let ordered = order_entities("t", "1", &OrderingPolicy::new(), raw).entities;
    Fixture {
        ids: ordered.iter().map(|o| o.id).collect(),
        stubs: ordered
            .iter()
            .map(|o| o.entity.provenance.clone())
            .collect(),
        geoms: ordered
            .iter()
            .map(|o| match &o.entity.geometry {
                RawGeometry::Line { start, end } => Normalized2DGeometry::polyline(
                    &[[start[0], start[1]], [end[0], end[1]]],
                    FidelityClass::SupportedExact,
                )
                .unwrap(),
                _ => unreachable!(),
            })
            .collect(),
    }
}

fn layers() -> Vec<LayerState> {
    let l = |name: &str, off: bool, frozen: bool| LayerState {
        name: name.into(),
        handle: None,
        off,
        frozen,
        locked: false,
    };
    vec![
        l("0", false, false),
        l("A-WALL", true, true),
        l("A-DOOR", false, false),
    ]
}

fn rows<'a>(f: &'a Fixture, styles: &'a [LineStyleComponent]) -> Vec<ComponentSource<'a>> {
    f.ids
        .iter()
        .zip(&f.stubs)
        .enumerate()
        .map(|(i, (id, stub))| ComponentSource {
            id: *id,
            stub,
            style: styles.get(i),
        })
        .collect()
}

fn styles_for(f: &Fixture) -> Vec<LineStyleComponent> {
    f.ids
        .iter()
        .map(|id| {
            let dash = DashPattern::new(vec![3.0, 1.0], 0.25).unwrap();
            LineStyleComponent::new(*id, Some(0.5), Some(dash), false).unwrap()
        })
        .collect()
}

fn write(f: &Fixture, streams: Vec<Stream>) -> GeometryArtifact {
    let config = WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: None,
    };
    write_geometry(
        &config,
        f.ids.iter().copied().zip(&f.geoms),
        streams,
        &CancelToken::new(),
    )
    .unwrap()
}

fn of<'a>(streams: &'a [Stream], cap: pvec_facade::CapabilityId) -> Option<&'a Stream> {
    streams.iter().find(|s| s.capability == cap)
}

#[test]
fn issue_1093_all_streams_off_writes_the_same_geometry_as_all_streams_on() {
    let f = flattened_blocks();
    let styles = styles_for(&f);
    let with = build_streams(StreamSelection::ALL, &f.ids, &layers(), &rows(&f, &styles)).unwrap();
    let without =
        build_streams(StreamSelection::NONE, &f.ids, &layers(), &rows(&f, &styles)).unwrap();
    assert!(without.is_empty());
    assert_eq!(with.len(), 3);
    let (a, b) = (write(&f, with), write(&f, without));
    assert_ne!(a.bytes, b.bytes, "the streams must actually be in the file");
    let (ra, rb) = (
        PvecReader::open(a.bytes).unwrap(),
        PvecReader::open(b.bytes).unwrap(),
    );
    assert_eq!(ra.entities().unwrap(), rb.entities().unwrap());
    assert_eq!(ra.manifest().bounds, rb.manifest().bounds);
}

#[test]
fn issue_1093_a_component_free_artifact_reads_as_plain_geometry() {
    let f = flattened_blocks();
    let art = write(&f, Vec::new());
    let r = PvecReader::open(art.bytes).unwrap();
    assert_eq!(r.entity_count().unwrap(), 8);
}

#[test]
fn issue_1093_every_subset_of_streams_builds_writes_and_decodes() {
    let f = flattened_blocks();
    let styles = styles_for(&f);
    for mask in 0..8u8 {
        let select = StreamSelection {
            layer: mask & 1 != 0,
            line_style: mask & 2 != 0,
            provenance: mask & 4 != 0,
        };
        let built = build_streams(select, &f.ids, &layers(), &rows(&f, &styles)).unwrap();
        assert_eq!(built.len(), mask.count_ones() as usize, "mask {mask}");
        for s in &built {
            if s.capability == LayerStream::CAPABILITY {
                LayerStream::decode(&s.payload).unwrap();
            } else if s.capability == LineStyleStream::CAPABILITY {
                LineStyleStream::decode(&s.payload).unwrap();
            } else {
                ProvenanceStream::decode(&s.payload).unwrap();
            }
        }
        let r = PvecReader::open(write(&f, built).bytes).unwrap();
        assert_eq!(r.entity_count().unwrap(), 8, "mask {mask}");
    }
}

#[test]
fn issue_1093_provenance_rows_equal_the_source_entities_not_the_curves() {
    let f = flattened_blocks();
    let built = build_streams(
        StreamSelection {
            provenance: true,
            ..StreamSelection::NONE
        },
        &f.ids,
        &layers(),
        &rows(&f, &[]),
    )
    .unwrap();
    let p = ProvenanceStream::decode(&of(&built, ProvenanceStream::CAPABILITY).unwrap().payload)
        .unwrap();
    // Eight curves, four source entities: two root lines and the two INSERTs.
    assert_eq!(p.records.len(), 4);
    let mut handles: Vec<Vec<u8>> = p.records.iter().map(|r| r.handle.clone()).collect();
    handles.sort();
    let want: Vec<Vec<u8>> = [1u64, 2, 100, 200]
        .iter()
        .map(|h| h.to_be_bytes().to_vec())
        .collect();
    assert_eq!(handles, want);
    // Each row points at an entity that exists in the artifact.
    let r = PvecReader::open(write(&f, built).bytes).unwrap();
    let have: Vec<_> = r.entities().unwrap().iter().map(|(id, _)| *id).collect();
    assert!(p.records.iter().all(|rec| have.contains(&rec.entity)));
}

#[test]
fn issue_1093_width_phase_and_visibility_round_trip_through_the_line_style_stream() {
    let f = flattened_blocks();
    let styles = styles_for(&f);
    let built = build_streams(
        StreamSelection {
            line_style: true,
            ..StreamSelection::NONE
        },
        &f.ids,
        &layers(),
        &rows(&f, &styles),
    )
    .unwrap();
    let s =
        LineStyleStream::decode(&of(&built, LineStyleStream::CAPABILITY).unwrap().payload).unwrap();
    assert_eq!(s.assignments.len(), 8);
    let def = &s.styles[s.assignments[0].1 as usize];
    assert_eq!(def.width, Some(StrokeWidth::Width(0.5)));
    assert_eq!((def.phase, def.visible), (0.25, false));
    assert_eq!(def.dashes, [3.0, 1.0]);
    // The reader hands the assignments back.
    let r = PvecReader::open(write(&f, built).bytes).unwrap();
    assert_eq!(r.styles().unwrap().len(), 8);
}

#[test]
fn issue_1093_the_layer_stream_names_layers_flags_them_and_maps_through_the_written_ids() {
    let f = flattened_blocks();
    // Entity 1 carries no geometry, so it is not written and its row has nowhere to point.
    let mut written = f.ids.clone();
    written.remove(1);
    let built = build_streams(
        StreamSelection {
            layer: true,
            ..StreamSelection::NONE
        },
        &written,
        &layers(),
        &rows(&f, &[]),
    )
    .unwrap();
    let l = LayerStream::decode(&of(&built, LayerStream::CAPABILITY).unwrap().payload).unwrap();
    let wall = l.names.iter().position(|n| n == "A-WALL").unwrap();
    assert_eq!(l.flags[wall], LAYER_OFF | LAYER_FROZEN);
    assert_eq!(l.assignments.len(), 7, "the unwritten entity has no row");
    assert!(l.assignments.iter().all(|(e, _)| e.index() < 7));
}

#[test]
fn issue_1093_an_empty_run_builds_no_stream() {
    let built = build_streams(StreamSelection::ALL, &[], &[], &[]).unwrap();
    assert!(built.is_empty());
}
