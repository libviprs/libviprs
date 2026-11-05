//! Issue #1525 (part of #1500 and #1093, N6.6): `PvecConversion` writes the layer, line style and
//! provenance streams beside the geometry when asked, and every one reads back through PVec's
//! reader. Needs `--features pvec-convert`.
//!
//! The named fixture is `plan` in `common/pvec_1500_plan.rs`: sixteen source entities that flatten
//! to twenty curves (two INSERTs of a three-line block).

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::cad::ItemHandle;
use libviprs::pvec::components::{ComponentSource, StreamSelection, build_streams};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::style::{DwgStyleInput, dwg_line_style};
use libviprs::pvec::writer::SpillConfig;
use libviprs::pvec::{
    EntityId, EntityStyle, InstancePath, OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry,
    SourceUnit, order_entities,
};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput};
use plan::{GEOMETRY, Mock, SOURCE_ENTITIES, scratch};
use pvec_facade::components::{
    LAYER_LOCKED, LayerStream, LineStyleStream, ProvenanceStream, Stream,
};

fn only(layer: bool, line_style: bool, provenance: bool) -> StreamSelection {
    StreamSelection {
        layer,
        line_style,
        provenance,
    }
}

fn convert(
    tag: &str,
    bytes: &'static [u8],
    select: StreamSelection,
) -> (PvecConversionOutput, PathBuf, PathBuf) {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(select)
        .convert_with::<Mock>()
        .unwrap();
    (out, dest, dir)
}

fn streams_of(dest: &Path) -> Vec<Stream> {
    let (streams, skipped) = open_published(dest).unwrap().streams().unwrap();
    assert!(skipped.is_empty(), "{skipped:?}");
    streams
}

fn of(streams: &[Stream], cap: pvec_facade::CapabilityId) -> Option<&Stream> {
    streams.iter().find(|s| s.capability == cap)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_a_conversion_with_no_option_writes_geometry_only() {
    let dir = scratch("default");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.written, GEOMETRY);
    assert!(streams_of(&dest).is_empty());
    assert!(out.report.components.is_none());
    let json = out.report.to_json_string().unwrap();
    assert!(!json.contains("components"), "{json}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_the_three_streams_are_written_and_read_back_through_pvecs_reader() {
    let (out, dest, dir) = convert("all", b"plan", StreamSelection::ALL);
    assert_eq!(out.written, GEOMETRY);
    let streams = streams_of(&dest);
    assert_eq!(streams.len(), 3);

    let layer =
        LayerStream::decode(&of(&streams, LayerStream::CAPABILITY).unwrap().payload).unwrap();
    let mut names = layer.names.clone();
    names.sort();
    assert_eq!(names, ["0", "A-DOOR", "A-WALL"]);
    let zero = layer.names.iter().position(|n| n == "0").unwrap();
    assert_eq!(layer.flags[zero] & LAYER_LOCKED, LAYER_LOCKED);
    assert_eq!(layer.assignments.len() as u64, GEOMETRY);

    let style =
        LineStyleStream::decode(&of(&streams, LineStyleStream::CAPABILITY).unwrap().payload)
            .unwrap();
    assert!(style.styles.iter().any(|s| s.name == "DASHED"), "{style:?}");

    let prov =
        ProvenanceStream::decode(&of(&streams, ProvenanceStream::CAPABILITY).unwrap().payload)
            .unwrap();
    // Twenty curves, sixteen source entities: the six block lines belong to their two INSERTs.
    assert_eq!(prov.records.len(), SOURCE_ENTITIES);
    let mut handles: Vec<Vec<u8>> = prov.records.iter().map(|r| r.handle.clone()).collect();
    handles.sort();
    handles.dedup();
    assert_eq!(handles.len(), SOURCE_ENTITIES);
    for h in [100u64, 200] {
        assert!(handles.contains(&h.to_be_bytes().to_vec()), "insert {h}");
    }
    for h in [11u64, 12, 13] {
        assert!(
            !handles.contains(&h.to_be_bytes().to_vec()),
            "block line {h}"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_geometry_is_identical_with_every_stream_on_and_off() {
    let (_, off, d1) = convert("geo-off", b"plan", StreamSelection::NONE);
    let (_, on, d2) = convert("geo-on", b"plan", StreamSelection::ALL);
    let (a, b) = (open_published(&off).unwrap(), open_published(&on).unwrap());
    assert_ne!(
        fs::read(&off).unwrap(),
        fs::read(&on).unwrap(),
        "the streams must actually be in the file"
    );
    assert_eq!(a.entities().unwrap(), b.entities().unwrap());
    assert_eq!(a.manifest().bounds, b.manifest().bounds);
    for d in [d1, d2] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_each_stream_can_be_dropped_on_its_own_and_the_rest_still_read() {
    let cases = [
        only(true, true, false),
        only(true, false, true),
        only(false, true, true),
        only(true, false, false),
        only(false, true, false),
        only(false, false, true),
    ];
    let (_, plain, dp) = convert("drop-base", b"plan", StreamSelection::NONE);
    let want = open_published(&plain).unwrap().entities().unwrap();
    for (n, select) in cases.into_iter().enumerate() {
        let (out, dest, dir) = convert(&format!("drop-{n}"), b"plan", select);
        let streams = streams_of(&dest);
        let wanted = [select.layer, select.line_style, select.provenance]
            .iter()
            .filter(|b| **b)
            .count();
        assert_eq!(streams.len(), wanted, "case {n}");
        for (flag, cap) in [
            (select.layer, LayerStream::CAPABILITY),
            (select.line_style, LineStyleStream::CAPABILITY),
            (select.provenance, ProvenanceStream::CAPABILITY),
        ] {
            assert_eq!(of(&streams, cap).is_some(), flag, "case {n}");
        }
        assert_eq!(open_published(&dest).unwrap().entities().unwrap(), want);
        assert_eq!(out.report.components.unwrap().streams.len(), wanted);
        let _ = fs::remove_dir_all(dir);
    }
    let _ = fs::remove_dir_all(dp);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_the_streams_are_the_same_bytes_spilled_or_in_memory() {
    let (_, mem, d1) = convert("mem", b"plan", StreamSelection::ALL);
    let dir = scratch("spill");
    let root = scratch("spill-root");
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
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
fn issue_1525_a_row_pvec_cannot_hold_is_counted_in_the_report_and_costs_only_its_row() {
    let (out, dest, dir) = convert("long", b"longlayer", StreamSelection::ALL);
    let c = out.report.components.unwrap();
    assert_eq!(
        c.dropped.get("layer.name-too-long"),
        Some(&1),
        "{:?}",
        c.dropped
    );
    assert_eq!(streams_of(&dest).len(), 3);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_a_linetype_name_pvec_cannot_hold_costs_only_its_style_row() {
    let (out, dest, dir) = convert("long-linetype", b"longlinetype", StreamSelection::ALL);
    let c = out.report.components.unwrap();
    assert_eq!(
        c.dropped.get("line-style.name-too-long"),
        Some(&1),
        "{:?}",
        c.dropped
    );
    let streams = streams_of(&dest);
    let style =
        LineStyleStream::decode(&of(&streams, LineStyleStream::CAPABILITY).unwrap().payload)
            .unwrap();
    // Every other entity keeps its row, and nothing in the table carries the long name.
    assert_eq!(style.assignments.len() as u64, GEOMETRY - 1);
    assert!(style.styles.iter().all(|s| s.name.len() <= 255));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_a_style_the_dwg_facts_cannot_make_is_counted_as_bad_style() {
    let (out, dest, dir) = convert("bad-style", b"badstyle", StreamSelection::ALL);
    let c = out.report.components.unwrap();
    assert_eq!(
        c.dropped.get("line-style.bad-style"),
        Some(&1),
        "{:?}",
        c.dropped
    );
    // Costs its own row: the other streams and the geometry are whole.
    assert_eq!(streams_of(&dest).len(), 3);
    assert_eq!(out.written, GEOMETRY);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_a_curve_the_writer_refuses_moves_no_stream_row_onto_a_stranger() {
    let bytes = pvec_gold::Corpus::open().get("mixed-entity-001").dwg();
    let dir = scratch("refused");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .convert()
        .unwrap();
    let refused = out
        .report
        .entities
        .iter()
        .filter(|e| e.reason.as_deref() == Some("pvec-cannot-store"))
        .count();
    assert!(refused >= 1, "the fixture has an entity PVec refuses");
    let streams = streams_of(&dest);
    let layer =
        LayerStream::decode(&of(&streams, LayerStream::CAPABILITY).unwrap().payload).unwrap();
    // One layer row for each written entity, each naming an entity that is in the file, in order.
    assert_eq!(layer.assignments.len() as u64, out.written);
    for (n, (id, _)) in layer.assignments.iter().enumerate() {
        assert_eq!(id.index() as usize, n);
    }
    assert_eq!(
        open_published(&dest).unwrap().entity_count().unwrap(),
        out.written
    );
    let _ = fs::remove_dir_all(dir);
}

/// Two root lines on A-WALL with a lineweight, the DASHED linetype and colour 3, and two INSERTs
/// of a three-line block, written as a DWG in memory.
fn dashed_drawing() -> Vec<u8> {
    use acadrust::entities::Line;
    use acadrust::tables::{Layer, LineType, LineTypeElement};
    use acadrust::types::{Color, LineWeight, Vector3};
    use acadrust::{CadDocument, DwgWriter, EntityType};

    let mut doc = CadDocument::new();
    let mut dashed = LineType::new("DASHED");
    dashed.elements = vec![LineTypeElement::dash(0.5), LineTypeElement::space(0.25)];
    doc.line_types.add(dashed).unwrap();
    doc.layers
        .add(Layer::with_color("A-WALL", Color::Index(5)))
        .unwrap();
    for x in [1.0, 2.0] {
        let mut l = Line::new();
        l.start = Vector3 { x, y: 0.0, z: 0.0 };
        l.end = Vector3 { x, y: 1.0, z: 0.0 };
        l.common.layer = "A-WALL".into();
        l.common.color = Color::Index(3);
        l.common.line_weight = LineWeight::Value(25);
        l.common.linetype = "DASHED".into();
        doc.add_entity(EntityType::Line(l)).unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_the_linetype_name_and_the_colour_count_come_from_a_dwg_conversion() {
    let bytes = dashed_drawing();
    let dir = scratch("dwg");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .convert()
        .unwrap();
    let streams = streams_of(&dest);
    let style =
        LineStyleStream::decode(&of(&streams, LineStyleStream::CAPABILITY).unwrap().payload)
            .unwrap();
    assert_eq!(style.styles.len(), 1, "{style:?}");
    assert_eq!(style.styles[0].name, "DASHED");
    // Two lines, each with an explicit colour index the monochrome profile drops.
    assert_eq!(out.report.components.unwrap().colours_ignored, 2);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn issue_1525_two_linetypes_with_the_same_dashes_keep_their_own_names_in_the_table() {
    let raw = |handle: u64| RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 1.0, 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    };
    let ordered = order_entities("t", "1", &OrderingPolicy::new(), vec![raw(1), raw(2)]).entities;
    let ids: Vec<EntityId> = ordered.iter().map(|o| o.id).collect();
    let input = |name: &str| DwgStyleInput {
        linetype_elements: Some(vec![0.5, -0.25]),
        linetype_name: Some(name.into()),
        ..DwgStyleInput::default()
    };
    let styles = [
        dwg_line_style(ids[0], &input("DASHED")).unwrap(),
        dwg_line_style(ids[1], &input("HIDDEN")).unwrap(),
    ];
    let rows: Vec<_> = ordered
        .iter()
        .zip(&styles)
        .map(|(o, s)| ComponentSource {
            id: o.id,
            stub: &o.entity.provenance,
            style: Some(s),
        })
        .collect();
    let built = build_streams(only(false, true, false), &ids, &[], &rows).unwrap();
    let style = LineStyleStream::decode(&built.streams[0].payload).unwrap();
    let names: Vec<_> = style.styles.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["DASHED", "HIDDEN"]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1525_the_component_summary_is_json_round_trippable() {
    let dir = scratch("json");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(b"plan"), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .convert_with::<Mock>()
        .unwrap();
    let json = out.report.to_json_string().unwrap();
    assert!(json.contains("\"components\""), "{json}");
    let back = ConversionReport::from_json_slice(json.as_bytes()).unwrap();
    assert_eq!(back.components, out.report.components);
    let _ = fs::remove_dir_all(dir);
}
