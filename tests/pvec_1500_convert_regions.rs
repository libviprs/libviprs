//! Issue #1526 (part of #1500 and #1093, N6.9): `PvecConversion` derives rooms and corridors from
//! the wall linework, writes them as PVec's semantic class stream and reports the counts. Needs
//! `--features pvec-convert`.
//!
//! The fixture is `plan` in `common/pvec_1500_plan.rs`: two 5 x 5 rooms under a 10 x 2 corridor,
//! with a 0.3 door gap in the bottom wall of the first room. Per the panel decision on #1500 the
//! step never writes the segment that closes a gap as geometry, so the room on the gap is counted
//! and left out of the stream (`semantic-class.derived-boundary`).

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::regions::RegionConfig;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::writer::SpillConfig;
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput};
use plan::{GEOMETRY, Mock, scratch};
use pvec_facade::EntityId as PvecId;
use pvec_facade::components::{
    EntityGeometry, LayerStream, LineStyleStream, ProvenanceStream, RegionKind,
    SemanticClassStream, SourceTag, Stream,
};
use pvec_geometry::Curve2d;

fn rooms() -> RegionConfig {
    RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    }
}

fn convert(
    tag: &str,
    bytes: &'static [u8],
    select: StreamSelection,
    regions: Option<RegionConfig>,
) -> (PvecConversionOutput, PathBuf, PathBuf) {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(select)
        .regions(regions)
        .convert_with::<Mock>()
        .unwrap();
    (out, dest, dir)
}

fn streams_of(dest: &Path) -> Vec<Stream> {
    let (streams, skipped) = open_published(dest).unwrap().streams().unwrap();
    assert!(skipped.is_empty(), "{skipped:?}");
    streams
}

fn semantic(dest: &Path) -> Option<SemanticClassStream> {
    streams_of(dest)
        .iter()
        .find(|s| s.capability == SemanticClassStream::CAPABILITY)
        .map(|s| SemanticClassStream::decode(&s.payload).unwrap())
}

struct Ends(Vec<([f64; 2], [f64; 2])>);

impl EntityGeometry for Ends {
    fn endpoints(&self, entity: PvecId) -> Option<([f64; 2], [f64; 2])> {
        self.0.get(entity.index() as usize).copied()
    }
}

fn ends_of(dest: &Path) -> Ends {
    Ends(
        open_published(dest)
            .unwrap()
            .entities()
            .unwrap()
            .iter()
            .map(|(_, c)| {
                let d = c.domain();
                let at = |u: f64| c.eval(u).map_or([f64::NAN; 2], |p| [p.x, p.y]);
                (at(d.lo()), at(d.hi()))
            })
            .collect(),
    )
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_rooms_and_the_corridor_are_counted_and_the_stream_reads_back() {
    let (out, dest, dir) = convert("regions", b"plan", StreamSelection::NONE, Some(rooms()));
    let sem = semantic(&dest).expect("a semantic class stream");
    sem.validate(&ends_of(&dest), 1e-9).unwrap();
    for r in &sem.regions {
        assert_eq!(r.origin.source, SourceTag::Inferred);
    }
    let summary = out.report.components.unwrap().regions.unwrap();
    assert_eq!(
        (
            summary.rooms,
            summary.corridors,
            summary.gaps_closed,
            summary.unlabelled
        ),
        (2, 1, 1, 0)
    );
    assert_eq!(summary.refused, None);
    // The room on the closed gap is left out, the other room and the corridor are in.
    assert_eq!(sem.regions_of(RegionKind::Room).len(), 1);
    assert_eq!(sem.regions_of(RegionKind::Corridor).len(), 1);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_room_on_a_door_gap_is_dropped_and_counted_not_written_as_geometry() {
    let (out, dest, dir) = convert("gap", b"plan", StreamSelection::NONE, Some(rooms()));
    // Only the source curves: the closing segment is not an entity.
    assert_eq!(out.written, GEOMETRY);
    assert_eq!(
        open_published(&dest).unwrap().entity_count().unwrap(),
        GEOMETRY
    );
    let c = out.report.components.unwrap();
    assert_eq!(c.dropped.get("semantic-class.derived-boundary"), Some(&1));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_geometry_is_identical_with_the_region_step_on_and_off() {
    let (_, off, d1) = convert("reg-off", b"plan", StreamSelection::ALL, None);
    let (_, on, d2) = convert("reg-on", b"plan", StreamSelection::ALL, Some(rooms()));
    let a = open_published(&off).unwrap();
    let b = open_published(&on).unwrap();
    assert_ne!(
        fs::read(&off).unwrap(),
        fs::read(&on).unwrap(),
        "the semantic stream must actually be in the file"
    );
    assert_eq!(a.entities().unwrap(), b.entities().unwrap());
    assert_eq!(a.manifest().bounds, b.manifest().bounds);
    // The other streams say the same thing either way.
    let (sa, sb) = (streams_of(&off), streams_of(&on));
    for cap in [
        LayerStream::CAPABILITY,
        LineStyleStream::CAPABILITY,
        ProvenanceStream::CAPABILITY,
    ] {
        let find = |s: &[Stream]| {
            s.iter()
                .find(|x| x.capability == cap)
                .unwrap()
                .payload
                .clone()
        };
        assert_eq!(find(&sa), find(&sb));
    }
    for d in [d1, d2] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_drawing_with_no_walls_gives_no_regions_and_no_error() {
    let (out, dest, dir) = convert("nowalls", b"nowalls", StreamSelection::NONE, Some(rooms()));
    let summary = out.report.components.unwrap().regions.unwrap();
    assert_eq!(
        (summary.rooms, summary.corridors, summary.walls_skipped),
        (0, 0, 0)
    );
    assert_eq!(summary.refused, None);
    assert!(semantic(&dest).is_none());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_plan_past_the_node_cap_converts_and_the_report_names_the_refusal() {
    let (out, dest, dir) = convert("huge", b"huge", StreamSelection::NONE, Some(rooms()));
    assert_eq!(out.written, 21_000);
    let summary = out.report.components.unwrap().regions.unwrap();
    assert_eq!(summary.refused.as_deref(), Some("regions.node-limit"));
    assert_eq!((summary.rooms, summary.corridors), (0, 0));
    assert!(semantic(&dest).is_none());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_walls_that_are_not_one_straight_edge_are_counted_not_silently_lost() {
    let (plain, _, d1) = convert("walls", b"plan", StreamSelection::NONE, Some(rooms()));
    let none = plain.report.components.unwrap().regions.unwrap();
    assert_eq!(none.walls_skipped, 0);
    // A three-vertex wall polyline is one entity with two edges, which a ring cannot name twice.
    let (out, _, d2) = convert(
        "polywall",
        b"polywall",
        StreamSelection::NONE,
        Some(rooms()),
    );
    let r = out.report.components.unwrap().regions.unwrap();
    assert_eq!(r.walls_skipped, 1);
    assert_eq!((r.rooms, r.corridors), (2, 1));
    for d in [d1, d2] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_streams_and_the_regions_are_the_same_bytes_spilled_or_in_memory() {
    let (_, mem, d1) = convert("mem", b"plan", StreamSelection::ALL, Some(rooms()));
    let dir = scratch("spill");
    let root = scratch("spill-root");
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .regions(Some(rooms()))
        .spill(SpillConfig {
            root: root.clone(),
            budget: 4096,
        })
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(fs::read(&mem).unwrap(), fs::read(&dest).unwrap());
    for d in [d1, dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_region_counts_survive_the_report_json() {
    let (out, _, dir) = convert("json", b"plan", StreamSelection::NONE, Some(rooms()));
    let json = out.report.to_json_string().unwrap();
    assert!(json.contains("\"regions\""), "{json}");
    let back = ConversionReport::from_json_slice(json.as_bytes()).unwrap();
    assert_eq!(back.components, out.report.components);
    let _ = fs::remove_dir_all(dir);
}
