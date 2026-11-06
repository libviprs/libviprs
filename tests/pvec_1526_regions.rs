//! Issue #1526 (part of #1500 and #1093, N6.9): `PvecConversion::regions` derives rooms and
//! corridors from the wall linework, writes PVec's semantic class stream and reports what it found,
//! and the geometry is the same bytes with the step on and off. Needs `--features pvec-convert`.
//!
//! The named fixture is `plan` in `common/pvec_1500_plan.rs`: twelve wall lines forming two 5 x 5
//! rooms under a 10 x 2 corridor, with a 0.3 door gap in the bottom wall of the first room.

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
use plan::{Mock, scratch};
use pvec_facade::components::{RegionKind, SemanticClassStream, SourceTag, Stream};

fn config() -> RegionConfig {
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

fn classes_of(dest: &Path) -> Option<SemanticClassStream> {
    streams_of(dest)
        .iter()
        .find(|s| s.capability == SemanticClassStream::CAPABILITY)
        .map(|s| SemanticClassStream::decode(&s.payload).unwrap())
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_conversion_with_no_regions_option_writes_no_class_stream_and_no_section() {
    let (out, dest, dir) = convert("off", b"plan", StreamSelection::ALL, None);
    assert!(out.report.regions.is_none());
    assert!(classes_of(&dest).is_none());
    let json = out.report.to_json_string().unwrap();
    assert!(!json.contains("\"regions\""), "{json}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_report_counts_the_rooms_the_corridor_and_the_closed_gap() {
    let (out, _, dir) = convert("counts", b"plan", StreamSelection::NONE, Some(config()));
    let r = out.report.regions.expect("a regions section");
    assert_eq!(
        (r.rooms, r.corridors, r.gaps_closed, r.unlabelled),
        (2, 1, 1, 0)
    );
    assert_eq!(r.refused, None);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_class_stream_reads_back_and_the_room_on_the_door_gap_is_dropped_and_counted() {
    let (out, dest, dir) = convert("stream", b"plan", StreamSelection::NONE, Some(config()));
    let r = out.report.regions.unwrap();
    // The first room runs along the segment that closes its door gap. That segment is not geometry,
    // so the room has no entity to point at: it is left out and the report says why.
    assert_eq!(r.written, 2);
    assert_eq!(r.dropped.get("derived-boundary"), Some(&1));
    assert_eq!(r.dropped.len(), 1, "{:?}", r.dropped);

    let classes = classes_of(&dest).expect("the semantic class stream");
    assert_eq!(classes.regions.len(), 2);
    let kinds: Vec<_> = classes.regions.iter().map(|x| x.kind).collect();
    assert!(kinds.contains(&RegionKind::Room) && kinds.contains(&RegionKind::Corridor));
    for region in &classes.regions {
        assert_eq!(region.origin.source, SourceTag::Inferred);
        assert!(region.origin.confidence.is_some());
        assert!(!region.outer.is_empty());
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_class_stream_is_the_only_stream_without_components_and_the_fourth_with_them() {
    let (_, only, d1) = convert("only", b"plan", StreamSelection::NONE, Some(config()));
    assert_eq!(streams_of(&only).len(), 1);
    let (out, all, d2) = convert("all", b"plan", StreamSelection::ALL, Some(config()));
    assert_eq!(streams_of(&all).len(), 4);
    assert!(classes_of(&all).is_some());
    assert!(out.report.components.is_some() && out.report.regions.is_some());
    for d in [d1, d2] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_geometry_is_identical_with_the_regions_step_on_and_off() {
    let (_, off, d1) = convert("geo-off", b"plan", StreamSelection::NONE, None);
    let (_, on, d2) = convert("geo-on", b"plan", StreamSelection::NONE, Some(config()));
    let (_, all, d3) = convert("geo-all", b"plan", StreamSelection::ALL, Some(config()));
    assert_ne!(
        fs::read(&off).unwrap(),
        fs::read(&on).unwrap(),
        "the class stream must actually be in the file"
    );
    let a = open_published(&off).unwrap();
    for other in [&on, &all] {
        let b = open_published(other).unwrap();
        assert_eq!(a.entities().unwrap(), b.entities().unwrap());
        assert_eq!(a.manifest().bounds, b.manifest().bounds);
    }
    for d in [d1, d2, d3] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_text_inside_a_room_names_it_and_the_report_counts_the_named_ones() {
    let (out, _, d1) = convert("named", b"labelled", StreamSelection::NONE, Some(config()));
    assert_eq!(out.report.regions.unwrap().labelled, 2);
    let (out, _, d2) = convert("unnamed", b"plan", StreamSelection::NONE, Some(config()));
    assert_eq!(out.report.regions.unwrap().labelled, 0);
    for d in [d1, d2] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_plan_past_the_node_limit_still_converts_and_the_report_says_why_no_regions() {
    let (out, dest, dir) = convert("huge", b"huge", StreamSelection::NONE, Some(config()));
    let r = out.report.regions.expect("a regions section");
    assert_eq!(r.refused.as_deref(), Some("regions.node-limit"));
    assert_eq!((r.rooms, r.corridors, r.written), (0, 0, 0));
    assert!(classes_of(&dest).is_none());
    assert_eq!(out.written, 21_000);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_class_stream_is_the_same_bytes_spilled_or_in_memory() {
    let (_, mem, d1) = convert("mem", b"plan", StreamSelection::ALL, Some(config()));
    let dir = scratch("spill");
    let root = scratch("spill-root");
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .regions(Some(config()))
        .spill(SpillConfig {
            root: root.clone(),
            budget: 4096,
        })
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(fs::read(&mem).unwrap(), fs::read(&dest).unwrap());
    assert!(classes_of(&dest).is_some());
    for d in [d1, dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_regions_summary_is_json_round_trippable() {
    let (out, _, dir) = convert("json", b"plan", StreamSelection::NONE, Some(config()));
    let json = out.report.to_json_string().unwrap();
    assert!(json.contains("\"regions\""), "{json}");
    let back = ConversionReport::from_json_slice(json.as_bytes()).unwrap();
    assert_eq!(back.regions, out.report.regions);
    assert_eq!(back.regions.unwrap().rooms, 2);
    let _ = fs::remove_dir_all(dir);
}

/// A closed 10 x 6 room of four A-WALL lines with a TEXT inside, written as a DWG in memory.
fn kitchen_drawing() -> Vec<u8> {
    use acadrust::entities::{Line, Text};
    use acadrust::tables::Layer;
    use acadrust::types::{Color, Vector3};
    use acadrust::{CadDocument, DwgWriter, EntityType};

    let mut doc = CadDocument::new();
    doc.layers
        .add(Layer::with_color("A-WALL", Color::Index(5)))
        .unwrap();
    let corners = [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]];
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        let mut l = Line::new();
        l.start = Vector3 {
            x: a[0],
            y: a[1],
            z: 0.0,
        };
        l.end = Vector3 {
            x: b[0],
            y: b[1],
            z: 0.0,
        };
        l.common.layer = "A-WALL".into();
        doc.add_entity(EntityType::Line(l)).unwrap();
    }
    let t = Text::with_value(
        "KITCHEN",
        Vector3 {
            x: 4.0,
            y: 3.0,
            z: 0.0,
        },
    )
    .with_height(0.3);
    doc.add_entity(EntityType::Text(t)).unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_a_dwg_with_walls_and_a_text_gives_one_named_room() {
    let bytes = kitchen_drawing();
    let dir = scratch("dwg");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .regions(Some(config()))
        .convert()
        .unwrap();
    let r = out.report.regions.unwrap();
    assert_eq!((r.rooms, r.corridors, r.labelled, r.written), (1, 0, 1, 1));
    let classes = classes_of(&dest).unwrap();
    assert_eq!(classes.regions[0].kind, RegionKind::Room);
    assert_eq!(classes.regions[0].outer.len(), 4);
    let _ = fs::remove_dir_all(dir);
}
