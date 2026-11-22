//! A hand-built drawing for #1633, picked by the bytes handed to `open`: model space with a root
//! line and an INSERT whose blocks hold lines on layer "0" and on named layers. `invisible` lists
//! the handles whose own visibility flag is off, and `entity_facts` reports it.

#![allow(dead_code)]

use std::collections::BTreeMap;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityFacts, EntityScope,
    EntityStyle, Facet, FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact,
    SourceTransform,
};

pub struct Mock {
    pub model: Vec<RawEntity>,
    pub blocks: Vec<(&'static str, Vec<RawEntity>)>,
    pub invisible: Vec<u64>,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn stub(handle: u64, class: &str, layer: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(ItemHandle::new(handle)),
        class_name: class.into(),
        block_path: InstancePath::root(),
        layer: layer.into(),
    }
}

pub fn line(handle: u64, layer: &str, x: f64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [x, 0.0, 0.0],
            end: [x + 1.0, 1.0, 0.0],
        },
        provenance: stub(handle, "LINE", layer),
    }
}

pub fn insert(handle: u64, layer: &str, block: &str, x: f64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::new([1.0, 0.0, 0.0, 1.0, x, 0.0]).unwrap(),
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
            elevation: 0.0,
            z_scale: 1.0,
        }),
        provenance: stub(handle, "INSERT", layer),
    }
}

impl Mock {
    fn scenario(name: &[u8]) -> Self {
        let (model, blocks, invisible) = match name {
            b"layer-one" => (
                vec![line(1, "0", 0.0), insert(100, "A", "BLK", 10.0)],
                vec![("BLK", vec![line(11, "0", 0.0), line(12, "C", 3.0)])],
                vec![],
            ),
            // OUTER is placed on A, and holds a line on "0" and an INSERT of INNER on "0".
            b"layer-nested" => (
                vec![line(1, "0", 0.0), insert(100, "A", "OUTER", 10.0)],
                vec![
                    (
                        "OUTER",
                        vec![line(11, "0", 0.0), insert(12, "0", "INNER", 3.0)],
                    ),
                    ("INNER", vec![line(21, "0", 0.0), line(22, "C", 3.0)]),
                ],
                vec![],
            ),
            // Three root lines, one block line, nothing invisible: the fixture check.
            b"visible" => (
                vec![
                    line(1, "A", 0.0),
                    line(2, "A", 2.0),
                    insert(100, "A", "BLK", 10.0),
                ],
                vec![("BLK", vec![line(11, "A", 0.0)])],
                vec![],
            ),
            // The same drawing with root line 2, the INSERT's line 11 and INSERT 101 off.
            b"invisible" => (
                vec![
                    line(1, "A", 0.0),
                    line(2, "A", 2.0),
                    insert(100, "A", "BLK", 10.0),
                    insert(101, "A", "BLK", 20.0),
                ],
                vec![("BLK", vec![line(11, "A", 0.0), line(12, "A", 3.0)])],
                vec![2, 11, 101],
            ),
            other => panic!("no scenario {other:?}"),
        };
        Mock {
            model,
            blocks,
            invisible,
            caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            }),
        }
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        match source {
            CadSource::Bytes(b) => Ok(Mock::scenario(b)),
            _ => Err(ProviderError::Open {
                provider: "mock",
                detail: "built by hand".into(),
            }),
        }
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
        Ok(["0", "A", "C"]
            .map(|name| LayerState {
                name: name.into(),
                handle: None,
                off: false,
                frozen: false,
                locked: false,
                plottable: true,
                ..LayerState::default()
            })
            .to_vec())
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let items = match scope {
            EntityScope::Layout(_) => self.model.clone(),
            EntityScope::BlockDefinition { name, reached_via } => self
                .blocks
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| ProviderError::NoSuchScope(name.clone()))?
                .1
                .iter()
                .cloned()
                .map(|mut e| {
                    e.provenance.block_path = reached_via.clone();
                    e
                })
                .collect(),
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }

    fn entity_facts(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        let handles: Vec<ItemHandle> = match scope {
            EntityScope::Layout(_) => self.model.iter(),
            EntityScope::BlockDefinition { name, .. } => self
                .blocks
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| ProviderError::NoSuchScope(name.clone()))?
                .1
                .iter(),
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        }
        .filter_map(|e| e.provenance.handle)
        .collect();
        Ok(handles
            .into_iter()
            .map(|h| {
                let visible = !self.invisible.contains(&h.get());
                (
                    h,
                    EntityFacts {
                        visible: SourceFact::Present(visible),
                        layer_override: SourceFact::Absent,
                        provider_visible: SourceFact::Present(visible),
                    },
                )
            })
            .collect())
    }
}
