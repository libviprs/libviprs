//! The fixture the #1500 conversion tests share: a provider that serves a small plan from memory.
//!
//! The named fixture is `plan`: two root lines, two INSERTs (100 and 200) of a block of three
//! lines, and twelve wall lines forming two 5 x 5 rooms under a 10 x 2 corridor, with a 0.3 door
//! gap in the bottom wall of the first room. Sixteen source entities, twenty curves. The other
//! byte strings select a variant (`huge`, `polywall`, `nowalls`, `longlayer`, `longlinetype`, `badstyle`).

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceColour, SourceFact,
    SourceLinetype, SourceLineweight, SourceTransform,
};

pub const GEOMETRY: u64 = 20;
pub const SOURCE_ENTITIES: usize = 16;

pub struct Mock {
    model: Vec<RawEntity>,
    caps: ProviderCaps,
}

pub struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

pub fn stub(handle: u64, class: &str, layer: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(ItemHandle::new(handle)),
        class_name: class.into(),
        block_path: InstancePath::root(),
        layer: layer.into(),
    }
}

pub fn line(handle: u64, layer: &str, a: [f64; 2], b: [f64; 2], style: EntityStyle) -> RawEntity {
    RawEntity {
        style,
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
        provenance: stub(handle, "LINE", layer),
    }
}

pub fn insert(handle: u64, at: [f64; 2]) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: "BLK".into(),
            transform: SourceTransform::new([1.0, 0.0, 0.0, 1.0, at[0], at[1]]).unwrap(),
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
        }),
        provenance: stub(handle, "INSERT", "A-DOOR"),
    }
}

pub fn linetype(name: String, scale: f64) -> EntityStyle {
    EntityStyle {
        linetype: SourceFact::Present(SourceLinetype::Pattern {
            name,
            elements: vec![0.5, -0.25],
            scale,
        }),
        ..EntityStyle::absent()
    }
}

pub fn dashed_style() -> EntityStyle {
    EntityStyle {
        colour: SourceFact::Present(SourceColour::Aci(3)),
        lineweight: SourceFact::Present(SourceLineweight::Hundredths(25)),
        linetype: SourceFact::Present(SourceLinetype::Pattern {
            name: "DASHED".into(),
            elements: vec![0.5, -0.25],
            scale: 1.0,
        }),
    }
}

/// The wall segments of the plan, the first room's bottom wall split around a 0.3 door gap.
pub fn plan_walls() -> Vec<([f64; 2], [f64; 2])> {
    vec![
        ([0.0, 0.0], [2.0, 0.0]),
        ([2.3, 0.0], [5.0, 0.0]),
        ([5.0, 0.0], [10.0, 0.0]),
        ([0.0, 5.0], [5.0, 5.0]),
        ([5.0, 5.0], [10.0, 5.0]),
        ([0.0, 7.0], [10.0, 7.0]),
        ([0.0, 0.0], [0.0, 5.0]),
        ([0.0, 5.0], [0.0, 7.0]),
        ([10.0, 0.0], [10.0, 5.0]),
        ([10.0, 5.0], [10.0, 7.0]),
        ([5.0, 0.0], [5.0, 5.0]),
        ([5.0, 5.0], [5.0, 7.0]),
    ]
}

pub fn plan() -> Vec<RawEntity> {
    let mut v = vec![
        line(1, "0", [0.0, -10.0], [1.0, -9.0], dashed_style()),
        line(2, "0", [10.0, -10.0], [11.0, -9.0], EntityStyle::absent()),
        insert(100, [20.0, -20.0]),
        insert(200, [40.0, -20.0]),
    ];
    for (i, (a, b)) in plan_walls().into_iter().enumerate() {
        v.push(line(1000 + i as u64, "A-WALL", a, b, EntityStyle::absent()));
    }
    v
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let model = match source {
            CadSource::Bytes(b"huge") => (0..21_000u64)
                .map(|n| {
                    let x = 10.0 * n as f64;
                    line(
                        n + 1,
                        "A-WALL",
                        [x, 0.0],
                        [x + 1.0, 0.0],
                        EntityStyle::absent(),
                    )
                })
                .collect(),
            CadSource::Bytes(b"polywall") => {
                let mut v = plan();
                let vertex = |x: f64, y: f64| RawVertex { x, y, bulge: 0.0 };
                v.push(RawEntity {
                    style: EntityStyle::absent(),
                    geometry: RawGeometry::Polyline {
                        vertices: vec![vertex(0.0, 100.0), vertex(4.0, 100.0), vertex(4.0, 104.0)],
                        closed: false,
                        elevation: 0.0,
                        normal: [0.0, 0.0, 1.0],
                    },
                    provenance: stub(2000, "LWPOLYLINE", "A-WALL"),
                });
                v
            }
            CadSource::Bytes(b"nowalls") => plan().into_iter().take(4).collect(),
            CadSource::Bytes(b"longlayer") => {
                let mut v = plan();
                v[1].provenance.layer = "L".repeat(300);
                v
            }
            CadSource::Bytes(b"longlinetype") => {
                let mut v = plan();
                v[0].style = linetype("L".repeat(300), 1.0);
                v
            }
            // A linetype scale of zero is not one a dash pattern can be scaled by.
            CadSource::Bytes(b"badstyle") => {
                let mut v = plan();
                v[0].style = linetype("DASHED".into(), 0.0);
                v
            }
            _ => plan(),
        };
        Ok(Mock {
            model,
            caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            }),
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
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
        let l = |name: &str, locked: bool| LayerState {
            name: name.into(),
            handle: None,
            off: false,
            frozen: false,
            locked,
        };
        Ok(vec![l("0", true), l("A-WALL", false), l("A-DOOR", false)])
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let items: Vec<RawEntity> = match scope {
            EntityScope::Layout(_) => self.model.clone(),
            EntityScope::BlockDefinition { name, reached_via } if name == "BLK" => (0..3u64)
                .map(|k| {
                    let mut e = line(
                        11 + k,
                        "A-DOOR",
                        [3.0 * k as f64, 0.0],
                        [3.0 * k as f64 + 1.0, 1.0],
                        EntityStyle::absent(),
                    );
                    e.provenance.block_path = reached_via.clone();
                    e
                })
                .collect(),
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }
}

pub fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1500-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}
