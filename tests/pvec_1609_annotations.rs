//! Issue #1609: TEXT, MTEXT, ATTDEF, HATCH and SOLID reach PVec's text and hatch streams, through
//! INSERTs too, and the conversion report says how many of each were mapped. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::fs;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{
    Annotation, ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope,
    EntityStyle, Facet, FacetSource, LayerState, LayoutId, LayoutKind, LayoutState, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, RawVertex, SourceTransform, SourceUnit,
};
use libviprs::pvec_convert::PvecConversion;
use plan::{scratch, stub};
use pvec_facade::components::{HatchStream, LineStyleStream, Stream, TextStream};

struct Notes {
    model: Vec<RawEntity>,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn unmapped(handle: u64, class: &str) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: "no mapping".into(),
        },
        provenance: stub(handle, class, "0"),
    }
}

fn square(side: f64) -> Vec<RawVertex> {
    [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]]
        .map(|[x, y]| RawVertex { x, y, bulge: 0.0 })
        .to_vec()
}

fn text(position: [f64; 3], height: f64, rotation: f64, content: &str) -> Annotation {
    Annotation::Text {
        position,
        height,
        rotation,
        content: content.into(),
        normal: [0.0, 0.0, 1.0],
    }
}

fn fill(rings: Vec<Vec<RawVertex>>, pattern: Option<(String, f64, f64)>, rgba: u32) -> Annotation {
    Annotation::Fill {
        rings,
        elevation: 0.0,
        normal: [0.0, 0.0, 1.0],
        pattern,
        rgba,
        unread_loops: 0,
    }
}

impl DwgProvider for Notes {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        let insert = RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Insert(BlockRef {
                block_name: "BLK".into(),
                // Turned a quarter, doubled, and moved to (100, 0), with a normal that is the Z axis
                // up to rounding.
                transform: SourceTransform::new([0.0, -2.0, 2.0, 0.0, 100.0, 0.0]).unwrap(),
                normal: [1e-17, 0.0, 1.000_000_000_000_000_2],
                grid: ArrayGrid::SINGLE,
            }),
            provenance: stub(50, "INSERT", "0"),
        };
        let mut model: Vec<RawEntity> = [
            (10, "TEXT"),
            (11, "MTEXT"),
            (12, "ATTDEF"),
            (13, "HATCH"),
            (14, "HATCH"),
            (15, "SOLID"),
            (16, "ATTDEF"),
            (17, "HATCH"),
        ]
        .into_iter()
        .map(|(h, c)| unmapped(h, c))
        .collect();
        model.push(insert);
        // The one curve of the drawing, so there is geometry whatever else is carried.
        model.push(plan::line(
            1,
            "0",
            [0.0, 0.0],
            [9.0, 0.0],
            EntityStyle::absent(),
        ));
        Ok(Notes {
            model,
            caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            }),
        })
    }

    fn name(&self) -> &'static str {
        "notes"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn annotation(&self, handle: ItemHandle) -> Option<Annotation> {
        match handle.get() {
            10 => Some(text([1.0, 2.0, 0.0], 2.5, 0.3, "Room")),
            11 => Some(text([5.0, 6.0, 0.0], 1.0, 0.0, "Hall")),
            12 => Some(text([7.0, 8.0, 0.0], 0.5, 0.0, "TAG")),
            13 => Some(fill(vec![square(4.0)], None, 0xFF00_00FF)),
            14 => Some(fill(
                vec![square(8.0), square(2.0)],
                Some(("ANSI31".into(), 0.5, 1.5)),
                0x0000_FFFF,
            )),
            15 => Some(fill(vec![square(1.0)], None, 0x00FF_00FF)),
            // A loop the provider could not read: the region is not whole, so it is not mapped.
            17 => Some(Annotation::Fill {
                rings: vec![square(3.0)],
                elevation: 0.0,
                normal: [0.0, 0.0, 1.0],
                pattern: None,
                rgba: 0,
                unread_loops: 1,
            }),
            20 => Some(text([1.0, 0.0, 0.0], 1.0, 0.0, "In block")),
            // 16 is an invisible ATTDEF: nothing is drawn.
            _ => None,
        }
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }])
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            handle: None,
            off: false,
            frozen: false,
            locked: false,
            plottable: true,
        }])
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let items = match scope {
            EntityScope::Layout(_) => self.model.clone(),
            EntityScope::BlockDefinition { name, reached_via } if name == "BLK" => {
                let mut e = unmapped(20, "TEXT");
                e.provenance.block_path = reached_via.clone();
                vec![e]
            }
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }
}

struct Run {
    streams: Vec<Stream>,
    written: u64,
    report: libviprs::pvec::report::ConversionReport,
}

fn convert(tag: &str, annotations: bool, select: StreamSelection) -> Run {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(b"notes"), &dest)
        .unit(SourceUnit::Meters)
        .components(select)
        .annotations(annotations)
        .convert_with::<Notes>()
        .unwrap();
    let (streams, skipped) = open_published(&dest).unwrap().streams().unwrap();
    assert!(skipped.is_empty(), "{skipped:?}");
    let _ = fs::remove_dir_all(dir);
    Run {
        streams,
        written: out.written,
        report: out.report,
    }
}

fn texts(run: &Run) -> Option<TextStream> {
    let s = run
        .streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)?;
    Some(TextStream::decode(&s.payload).unwrap())
}

fn hatches(run: &Run) -> Option<HatchStream> {
    let s = run
        .streams
        .iter()
        .find(|s| s.capability == HatchStream::CAPABILITY)?;
    Some(HatchStream::decode(&s.payload).unwrap())
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_without_the_option_nothing_is_carried() {
    let run = convert("1609-off", false, StreamSelection::ALL);
    assert_eq!(run.written, 1);
    assert!(texts(&run).is_none() && hatches(&run).is_none());
    assert!(
        run.report
            .components
            .is_none_or(|c| c.annotations.is_empty())
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_text_mtext_and_attdef_are_text_records() {
    let run = convert("1609-text", true, StreamSelection::ALL);
    let t = texts(&run).expect("a text stream");
    let mut got: Vec<_> = t.records.iter().map(|r| r.text.as_str()).collect();
    got.sort_unstable();
    assert_eq!(got, ["Hall", "In block", "Room", "TAG"]);
    let room = t.records.iter().find(|r| r.text == "Room").unwrap();
    assert_eq!(
        (room.x, room.y, room.height, room.rotation),
        (1.0, 2.0, 2.5, 0.3)
    );
    // Through the INSERT: (1, 0) lands on (100, 2), the height doubles and the quarter turn shows.
    let nested = t.records.iter().find(|r| r.text == "In block").unwrap();
    assert!(
        (nested.x - 100.0).abs() < 1e-9 && (nested.y - 2.0).abs() < 1e-9,
        "{nested:?}"
    );
    assert!((nested.height - 2.0).abs() < 1e-9, "{nested:?}");
    assert!(
        (nested.rotation - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "{nested:?}"
    );
    // Every record hangs on an entity that was written.
    assert!(
        t.records
            .iter()
            .all(|r| u64::from(r.entity.index()) < run.written)
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_hatch_and_solid_are_hatch_records_over_boundary_entities() {
    let run = convert("1609-hatch", true, StreamSelection::ALL);
    let h = hatches(&run).expect("a hatch stream");
    assert_eq!(h.records.len(), 3);
    let pattern = h
        .records
        .iter()
        .find(|r| r.boundary.len() == 2)
        .expect("the two loop hatch");
    match &pattern.fill {
        pvec_facade::components::Fill::Pattern {
            name,
            angle,
            scale,
            rgba,
        } => {
            assert_eq!(
                (name.as_str(), *angle, *scale, *rgba),
                ("ANSI31", 0.5, 1.5, 0x0000_FFFF)
            );
        }
        other => panic!("{other:?}"),
    }
    let solids = h
        .records
        .iter()
        .filter(|r| matches!(r.fill, pvec_facade::components::Fill::Solid { .. }))
        .count();
    assert_eq!(solids, 2);
    assert!(h.records.iter().all(|r| {
        r.boundary
            .iter()
            .all(|e| u64::from(e.index()) < run.written)
    }));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_anchors_and_boundaries_are_hidden_by_the_line_style_stream() {
    let run = convert("1609-hidden", true, StreamSelection::ALL);
    let style = run
        .streams
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
        .expect("a line style stream");
    let style = LineStyleStream::decode(&style.payload).unwrap();
    // Everything but the one line is an anchor or a boundary, so none of it may draw.
    assert_eq!(style.assignments.len() as u64, run.written);
    let hidden = style
        .assignments
        .iter()
        .filter(|(_, n)| !style.styles[*n as usize].visible)
        .count() as u64;
    assert_eq!(hidden, run.written - 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_the_report_counts_each_kind() {
    let run = convert("1609-report", true, StreamSelection::ALL);
    let counts = run.report.components.unwrap().annotations;
    let n = |k: &str| counts.get(k).copied().unwrap_or(0);
    assert_eq!(n("TEXT.mapped"), 2);
    assert_eq!(n("MTEXT.mapped"), 1);
    assert_eq!(n("ATTDEF.mapped"), 1);
    assert_eq!(n("ATTDEF.unmapped"), 1);
    assert_eq!(n("HATCH.mapped"), 2);
    assert_eq!(n("HATCH.unmapped"), 1);
    assert_eq!(n("SOLID.mapped"), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_without_the_line_style_stream_nothing_is_mapped() {
    let select = StreamSelection {
        line_style: false,
        ..StreamSelection::ALL
    };
    let run = convert("1609-nostyle", true, select);
    assert_eq!(run.written, 1);
    assert!(texts(&run).is_none() && hatches(&run).is_none());
}

/// A real drawing: the block `ATT` holds a plain ATTDEF (ROOMNO), a constant one (NOTE) and one
/// nobody fills in (EMPTY), and its INSERT carries a value for ROOMNO. A bare ATTDEF in model
/// space (TOP) is the one place a tag is drawn.
fn attribute_drawing() -> Vec<u8> {
    use acadrust::entities::{AttributeDefinition, AttributeEntity, Insert, Line};
    use acadrust::tables::BlockRecord;
    use acadrust::types::Vector3;
    use acadrust::{CadDocument, DwgWriter, EntityType};

    let mut doc = CadDocument::new();
    let mut record = BlockRecord::new("ATT");
    record.handle = doc.allocate_handle();
    let owner = record.handle;
    doc.block_records.add(record).unwrap();
    let mut def = |tag: &str, value: &str, constant: bool, owner| {
        let mut d = AttributeDefinition::new(tag.into(), String::new(), value.into());
        d.height = 1.0;
        d.flags.constant = constant;
        let mut e = EntityType::AttributeDefinition(d);
        if let Some(owner) = owner {
            e.common_mut().owner_handle = owner;
        }
        doc.add_entity(e).unwrap();
    };
    def("ROOMNO", "000", false, Some(owner));
    def("NOTE", "Fixed note", true, Some(owner));
    def("EMPTY", "unset", false, Some(owner));
    def("TOP", "top default", false, None);
    let mut insert = Insert::new("ATT", Vector3::new(100.0, 0.0, 0.0));
    let mut value = AttributeEntity::new("ROOMNO".into(), "101".into());
    value.height = 1.0;
    insert.attributes.push(value);
    doc.add_entity(EntityType::Insert(insert)).unwrap();
    doc.add_entity(EntityType::Line(Line::from_coords(
        0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    )))
    .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_an_attdef_in_an_insert_draws_the_inserts_value_or_its_constant_never_its_tag() {
    let dir = scratch("1609-attdef");
    let dest = dir.join("o.pvec");
    let bytes = attribute_drawing();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection::ALL)
        .annotations(true)
        .convert()
        .unwrap();
    let (streams, _) = open_published(&dest).unwrap().streams().unwrap();
    let _ = fs::remove_dir_all(dir);
    let t = streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)
        .map(|s| TextStream::decode(&s.payload).unwrap())
        .expect("a text stream");
    let mut got: Vec<_> = t.records.iter().map(|r| r.text.as_str()).collect();
    got.sort_unstable();
    assert_eq!(got, ["101", "Fixed note", "TOP"]);
    let counts = out.report.components.unwrap().annotations;
    let n = |k: &str| counts.get(k).copied().unwrap_or(0);
    assert_eq!((n("ATTDEF.mapped"), n("ATTDEF.unmapped")), (3, 1));
}
