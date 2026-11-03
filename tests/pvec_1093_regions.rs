//! Issue #1093 (N6.9): rooms and corridors derived from the wall linework. Needs
//! `--features pvec-write`.
//!
//! The known-truth fixture `plan` is two 5 x 5 rooms side by side under a 10 x 2 corridor, drawn as
//! eleven wall segments that meet at their ends.

#![cfg(feature = "pvec-write")]

use libviprs::cad::ItemHandle;
use libviprs::pvec::regions::{
    BoundaryRef, ExplicitRoom, RegionConfig, RegionsError, RoomLabel, WallEdge, derive_regions,
    semantic_stream,
};
use libviprs::pvec::{
    EntityId, EntityStyle, InstancePath, OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry,
    order_entities,
};
use pvec_facade::EntityId as PvecId;
use pvec_facade::components::{EntityGeometry, RegionKind, SemanticClassStream, SourceTag};

fn config() -> RegionConfig {
    RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    }
}

/// Minted ids for `segs`, in the order given.
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

fn plan_segments() -> Vec<([f64; 2], [f64; 2])> {
    vec![
        ([0.0, 0.0], [5.0, 0.0]),
        ([5.0, 0.0], [10.0, 0.0]),
        ([0.0, 5.0], [5.0, 5.0]),
        ([5.0, 5.0], [10.0, 5.0]),
        ([0.0, 7.0], [10.0, 7.0]),
        ([0.0, 0.0], [0.0, 5.0]),
        ([0.0, 5.0], [0.0, 7.0]),
        ([10.0, 0.0], [10.0, 5.0]),
        ([10.0, 5.0], [10.0, 7.0]),
        ([5.0, 0.0], [5.0, 5.0]),
    ]
}

fn sources(r: &libviprs::pvec::regions::Region) -> Vec<EntityId> {
    let mut v: Vec<_> = r
        .boundary
        .iter()
        .map(|b| match b {
            BoundaryRef::Source(e) => *e,
            BoundaryRef::Derived(_) => panic!("unexpected derived edge"),
        })
        .collect();
    v.sort();
    v
}

#[test]
fn issue_1093_the_known_truth_plan_gives_two_rooms_and_a_corridor_bounded_by_their_walls() {
    let w = walls(&plan_segments());
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    assert_eq!((out.report.rooms, out.report.corridors), (2, 1));
    assert_eq!(out.regions.len(), 3);
    assert!(out.derived.is_empty());
    let kinds = |k| out.regions.iter().filter(|r| r.kind == Some(k)).count();
    assert_eq!(
        (kinds(RegionKind::Room), kinds(RegionKind::Corridor)),
        (2, 1)
    );
    // Room 1 is bounded by the bottom-left, the left, the top-left and the shared wall.
    let want: Vec<_> = [0usize, 5, 2, 9].iter().map(|i| w[*i].entity).collect();
    let mut want = want;
    want.sort();
    assert!(out.regions.iter().any(|r| sources(r) == want));
    for r in &out.regions {
        assert_eq!(r.origin.source, SourceTag::Inferred);
        let c = r
            .origin
            .confidence
            .expect("an inferred label carries a confidence");
        assert!((0.0..=1.0).contains(&c));
    }
}

fn room_with_door(gap: f64) -> Vec<([f64; 2], [f64; 2])> {
    // A 4 x 4 room whose bottom wall has a door gap of `gap` centred at x = 2.
    let (a, b) = (2.0 - gap / 2.0, 2.0 + gap / 2.0);
    vec![
        ([0.0, 0.0], [a, 0.0]),
        ([b, 0.0], [4.0, 0.0]),
        ([4.0, 0.0], [4.0, 4.0]),
        ([4.0, 4.0], [0.0, 4.0]),
        ([0.0, 4.0], [0.0, 0.0]),
    ]
}

#[test]
fn issue_1093_a_door_gap_under_the_tolerance_is_closed_and_the_closing_segment_is_derived() {
    let w = walls(&room_with_door(0.4));
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    assert_eq!(out.report.gaps_closed, 1);
    assert_eq!(out.derived.len(), 1);
    assert_eq!(out.regions.len(), 1);
    let derived_refs = out.regions[0]
        .boundary
        .iter()
        .filter(|b| matches!(b, BoundaryRef::Derived(0)))
        .count();
    assert_eq!(derived_refs, 1);
    assert_eq!(out.regions[0].boundary.len(), 6);
}

#[test]
fn issue_1093_a_door_gap_over_the_tolerance_stays_open_and_encloses_nothing() {
    let w = walls(&room_with_door(0.9));
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    assert_eq!(out.report.gaps_closed, 0);
    assert!(out.derived.is_empty());
    assert!(out.regions.is_empty());
}

#[test]
fn issue_1093_a_drawing_with_no_enclosed_space_has_no_regions_and_no_error() {
    let w = walls(&[([0.0, 0.0], [5.0, 0.0]), ([5.0, 0.0], [5.0, 5.0])]);
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    assert!(out.regions.is_empty());
    assert_eq!(out.report, Default::default());
    assert!(
        derive_regions(&[], &[], &[], &config())
            .unwrap()
            .regions
            .is_empty()
    );
}

#[test]
fn issue_1093_a_room_with_an_explicit_boundary_is_explicit_not_inferred() {
    let w = walls(&plan_segments());
    let explicit = ExplicitRoom {
        polygon: vec![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
        boundary: vec![w[0].entity, w[9].entity, w[2].entity, w[5].entity],
        name: Some("Office".into()),
    };
    let out = derive_regions(&w, &[], &[explicit], &config()).unwrap();
    let explicit_rooms: Vec<_> = out
        .regions
        .iter()
        .filter(|r| r.origin.source == SourceTag::Explicit)
        .collect();
    assert_eq!(explicit_rooms.len(), 1);
    assert_eq!(explicit_rooms[0].kind, Some(RegionKind::Room));
    assert_eq!(explicit_rooms[0].label.as_deref(), Some("Office"));
    assert_eq!(explicit_rooms[0].origin.confidence, None);
    // The derived face it replaces is not reported twice.
    assert_eq!(out.regions.len(), 3);
    assert_eq!(out.report.rooms, 2);
}

#[test]
fn issue_1093_a_text_inside_a_room_names_it() {
    let w = walls(&plan_segments());
    let label = RoomLabel {
        at: [7.5, 2.5],
        text: "Kitchen".into(),
    };
    let out = derive_regions(&w, &[label], &[], &config()).unwrap();
    let named: Vec<_> = out.regions.iter().filter(|r| r.label.is_some()).collect();
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].label.as_deref(), Some("Kitchen"));
    assert_eq!(named[0].kind, Some(RegionKind::Room));
}

#[test]
fn issue_1093_the_face_limit_refuses_with_a_reason_code() {
    let w = walls(&plan_segments());
    let cfg = RegionConfig {
        max_faces: 2,
        ..config()
    };
    let err = derive_regions(&w, &[], &[], &cfg).unwrap_err();
    assert_eq!(err, RegionsError::TooManyFaces { limit: 2 });
    assert_eq!(err.reason_code(), "regions.face-limit");
}

#[test]
fn issue_1093_a_bad_tolerance_is_refused() {
    let w = walls(&plan_segments());
    for gap in [f64::NAN, -1.0, f64::INFINITY] {
        let cfg = RegionConfig {
            gap_tolerance: gap,
            ..config()
        };
        assert_eq!(
            derive_regions(&w, &[], &[], &cfg)
                .unwrap_err()
                .reason_code(),
            "regions.bad-config"
        );
    }
}

#[test]
fn issue_1093_the_output_is_the_same_every_run() {
    let w = walls(&plan_segments());
    let a = derive_regions(&w, &[], &[], &config()).unwrap();
    let b = derive_regions(&w, &[], &[], &config()).unwrap();
    assert_eq!(a, b);
}

struct Ends(Vec<([f64; 2], [f64; 2])>);

impl EntityGeometry for Ends {
    fn endpoints(&self, entity: PvecId) -> Option<([f64; 2], [f64; 2])> {
        self.0.get(entity.index() as usize).copied()
    }
}

#[test]
fn issue_1093_the_semantic_stream_validates_against_the_geometry_it_points_at() {
    let segs = plan_segments();
    let w = walls(&segs);
    let out = derive_regions(&w, &[], &[], &config()).unwrap();
    let ids: Vec<_> = w.iter().map(|e| e.entity).collect();
    let stream = semantic_stream(&out, &ids).unwrap().stream.unwrap();
    let s = SemanticClassStream::decode(&stream.payload).unwrap();
    assert_eq!(s.regions_of(RegionKind::Room).len(), 2);
    assert_eq!(s.regions_of(RegionKind::Corridor).len(), 1);
    s.validate(&Ends(segs), 1e-9).unwrap();
}
