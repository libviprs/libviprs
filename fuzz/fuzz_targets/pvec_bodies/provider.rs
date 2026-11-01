//! The body of `fuzz_pvec_provider` (N7.6, #1437), shared with `tests/pvec_1437_fuzz_seeds.rs` so
//! the seed corpus runs through exactly this code in the ordinary suite.
//!
//! An arbitrary drawing held in memory sits behind the `DwgProvider` trait: layers, layouts, block
//! definitions that nest, recurse and refer to blocks that do not exist, entities of every kind
//! including splines with absent or mismatched facts and non-finite numbers, capabilities that
//! vary, and a cursor that fails part-way through a scope. It is driven through the view profile,
//! the block flattener and the extractors under tight limits. Whatever comes back must be a value:
//! a panic, an abort or an unbounded allocation is the finding.

use std::collections::BTreeMap;

use arbitrary::{Result as AResult, Unstructured};
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::dwg_basic;
use libviprs::pvec::dwg_spline;
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, Normalized2DGeometry,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact,
    SourceLimits, SourceTransform, ViewProfile,
};

pub struct Mock {
    caps: ProviderCaps,
    layouts: Vec<(LayoutState, Vec<RawEntity>)>,
    blocks: BTreeMap<String, Vec<RawEntity>>,
    layers: Vec<LayerState>,
    /// A cursor yields this many entities and then fails, when set.
    fail_after: Option<usize>,
}

struct Cursor {
    items: std::vec::IntoIter<RawEntity>,
    left: Option<usize>,
}

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        if let Some(n) = &mut self.left {
            if *n == 0 {
                return Err(ProviderError::CannotReport(Facet::LayerState));
            }
            *n -= 1;
        }
        Ok(self.items.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Err(ProviderError::Unrecognised {
            provider: "fuzz",
            detail: "built from arbitrary data".into(),
        })
    }
    fn name(&self) -> &'static str {
        "fuzz"
    }
    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(self.layouts.iter().map(|(l, _)| l.clone()).collect())
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        if self.caps.availability(Facet::LayerState) == Availability::CannotReport {
            return Err(ProviderError::CannotReport(Facet::LayerState));
        }
        Ok(self.layers.clone())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let (items, path) = match scope {
            EntityScope::Layout(id) => {
                let (_, e) = self
                    .layouts
                    .get(id.0 as usize)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {}", id.0)))?;
                (e.clone(), InstancePath::root())
            }
            EntityScope::BlockDefinition { name, reached_via } => (
                self.blocks
                    .get(name)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("block {name}")))?
                    .clone(),
                reached_via.clone(),
            ),
            _ => return Err(ProviderError::NoSuchScope("unknown scope kind".into())),
        };
        let stamped: Vec<RawEntity> = items
            .into_iter()
            .map(|mut e| {
                e.provenance.block_path = path.clone();
                e
            })
            .collect();
        Ok(Box::new(Cursor {
            items: stamped.into_iter(),
            left: self.fail_after,
        }))
    }
}

const LAYERS: [&str; 4] = ["0", "A", "B", "HIDDEN"];

fn point(u: &mut Unstructured<'_>) -> AResult<[f64; 3]> {
    Ok([u.arbitrary()?, u.arbitrary()?, u.arbitrary()?])
}

fn fact<T>(
    u: &mut Unstructured<'_>,
    value: impl FnOnce(&mut Unstructured<'_>) -> AResult<T>,
) -> AResult<SourceFact<T>> {
    Ok(match u.int_in_range(0..=2)? {
        0 => SourceFact::Present(value(u)?),
        1 => SourceFact::Absent,
        _ => SourceFact::NotReportable,
    })
}

fn geometry(u: &mut Unstructured<'_>, blocks: u8) -> AResult<(RawGeometry, &'static str)> {
    Ok(match u.int_in_range(0..=7)? {
        0 => (
            RawGeometry::Line {
                start: point(u)?,
                end: point(u)?,
            },
            "LINE",
        ),
        1 => {
            let n = u.arbitrary_len::<[f64; 3]>()?.min(16);
            let mut vertices = Vec::new();
            for _ in 0..n {
                vertices.push(RawVertex {
                    x: u.arbitrary()?,
                    y: u.arbitrary()?,
                    bulge: u.arbitrary()?,
                });
            }
            (
                RawGeometry::Polyline {
                    vertices,
                    closed: u.arbitrary()?,
                    elevation: u.arbitrary()?,
                    normal: point(u)?,
                },
                "LWPOLYLINE",
            )
        }
        2 => (
            RawGeometry::Arc {
                center: point(u)?,
                radius: u.arbitrary()?,
                start_angle: u.arbitrary()?,
                end_angle: u.arbitrary()?,
                normal: point(u)?,
            },
            "ARC",
        ),
        3 => (
            RawGeometry::Circle {
                center: point(u)?,
                radius: u.arbitrary()?,
                normal: point(u)?,
            },
            "CIRCLE",
        ),
        4 => (
            RawGeometry::Ellipse {
                center: point(u)?,
                major_axis: point(u)?,
                minor_ratio: u.arbitrary()?,
                start_parameter: u.arbitrary()?,
                end_parameter: u.arbitrary()?,
                normal: point(u)?,
            },
            "ELLIPSE",
        ),
        5 => (spline(u)?, "SPLINE"),
        6 => {
            // An INSERT, usually of a block that exists, sometimes of one that does not.
            let k: u8 = u.int_in_range(0..=blocks)?;
            let c: [f64; 6] = [
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
            ];
            let transform = SourceTransform::new(c).unwrap_or(SourceTransform::IDENTITY);
            let grid = if u.arbitrary()? {
                ArrayGrid {
                    rows: u.arbitrary()?,
                    columns: u.arbitrary()?,
                    row_spacing: u.arbitrary()?,
                    column_spacing: u.arbitrary()?,
                }
            } else {
                ArrayGrid::SINGLE
            };
            (
                RawGeometry::Insert(BlockRef {
                    block_name: format!("B{k}"),
                    transform,
                    normal: if u.arbitrary()? {
                        [0.0, 0.0, 1.0]
                    } else {
                        point(u)?
                    },
                    grid,
                }),
                "INSERT",
            )
        }
        _ => (
            RawGeometry::Unmapped {
                reason: "fuzz".into(),
            },
            "PROXY",
        ),
    })
}

fn spline(u: &mut Unstructured<'_>) -> AResult<RawGeometry> {
    let n = u.arbitrary_len::<[f64; 3]>()?.min(24);
    let mut control_points = Vec::new();
    for _ in 0..n {
        control_points.push(point(u)?);
    }
    let floats = |u: &mut Unstructured<'_>| -> AResult<Vec<f64>> {
        let n = u.arbitrary_len::<f64>()?.min(48);
        (0..n).map(|_| u.arbitrary()).collect()
    };
    Ok(RawGeometry::Spline {
        degree: u.arbitrary()?,
        control_points,
        knots: fact(u, floats)?,
        weights: fact(u, floats)?,
        periodic: fact(u, |u| u.arbitrary())?,
        closed: u.arbitrary()?,
    })
}

fn entity(u: &mut Unstructured<'_>, blocks: u8, handle: &mut u64) -> AResult<RawEntity> {
    let (geometry, class) = geometry(u, blocks)?;
    *handle += 1;
    Ok(RawEntity {
        geometry,
        style: EntityStyle::absent(),
        provenance: ProvenanceStub {
            handle: if u.arbitrary()? {
                Some(ItemHandle::new(*handle))
            } else {
                None
            },
            class_name: class.to_string(),
            block_path: InstancePath::root(),
            layer: LAYERS[u.int_in_range(0..=3usize)?].to_string(),
        },
    })
}

fn build(u: &mut Unstructured<'_>) -> AResult<Mock> {
    let blocks: u8 = u.int_in_range(0..=5)?;
    let mut handle = 0;
    let mut defs = BTreeMap::new();
    for k in 0..blocks {
        let n = u.int_in_range(0..=8)?;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(entity(u, blocks, &mut handle)?);
        }
        defs.insert(format!("B{k}"), v);
    }
    let n = u.int_in_range(0..=24)?;
    let mut model = Vec::new();
    for _ in 0..n {
        model.push(entity(u, blocks, &mut handle)?);
    }
    let mut layouts = vec![(
        LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        },
        model,
    )];
    if u.arbitrary()? {
        layouts.push((
            LayoutState {
                id: LayoutId(1),
                name: "Layout1".into(),
                kind: LayoutKind::Paper,
                block_name: "*Paper_Space".into(),
            },
            Vec::new(),
        ));
    }
    let mask: u16 = u.arbitrary()?;
    let caps = Facet::ALL
        .iter()
        .enumerate()
        .fold(ProviderCaps::new(), |c, (i, f)| {
            let a = if mask >> i & 1 == 1 {
                Availability::Reported
            } else {
                Availability::CannotReport
            };
            c.with(*f, a, FacetSource::DecoderAdapter)
        });
    let mut layers = Vec::new();
    for name in LAYERS {
        if u.arbitrary()? {
            layers.push(LayerState {
                name: name.to_string(),
                handle: None,
                off: u.arbitrary()?,
                frozen: u.arbitrary()?,
                locked: u.arbitrary()?,
            });
        }
    }
    let fail_after = if u.arbitrary()? {
        Some(u.int_in_range(0..=12usize)?)
    } else {
        None
    };
    Ok(Mock {
        caps,
        layouts,
        blocks: defs,
        layers,
        fail_after,
    })
}

/// The tight limits a run works under, so a refusal is cheap and an amplification shows up as a
/// timeout or an out-of-memory rather than a slow pass.
fn limits() -> SourceLimits {
    let mut l = SourceLimits::default();
    l.max_block_depth = 12;
    l.max_entities = 4_000;
    l.max_polyline_vertices = 1 << 10;
    l.max_spline_control_points = 1 << 10;
    l.max_spline_knots = (1 << 10) + 64;
    l.max_insert_grid = 64;
    l.max_expanded_instances = 20_000;
    l.max_allocation_bytes = 64 << 20;
    l.max_name_bytes = 64;
    l
}

pub fn run(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let Ok(p) = build(&mut u) else { return };
    let _ = run_mock(&p);
}

/// Drives `p` through the view profile, the flattener and the extractors, and gives back what the
/// flattener said so a direct test can look at it.
pub fn run_mock(p: &Mock) -> Option<Result<(), String>> {
    let limits = limits();
    let _ = ViewProfile::default_profile().with_limits(limits).apply(p);
    let layouts = p.layouts().ok()?;
    let model = layouts.iter().find(|l| l.kind == LayoutKind::Model)?;
    let flat = BlockResolver::new(p, FlattenBounds::new(limits, 1_000)).flatten(
        &EntityScope::Layout(model.id),
        "fuzz",
        &model.name,
        &mut NoProvenance,
        &mut LineageStore::new(),
    );
    let flat = match flat {
        Ok(f) => f,
        Err(e) => return Some(Err(e.to_string())),
    };
    for f in &flat.entities {
        let _ = dwg_basic::extract(&f.entity);
        if let Ok(s) = dwg_spline::extract(&f.entity) {
            let _ = Normalized2DGeometry::from_spline(&s);
        }
    }
    Some(Ok(()))
}

/// A recursion bomb: `levels` blocks, each inserting the next, and the last inserting the first,
/// instanced once from model space.
pub fn recursion_bomb(levels: u8) -> Mock {
    let mut handle = 0;
    let insert = |to: u8, handle: &mut u64| {
        *handle += 1;
        RawEntity {
            geometry: RawGeometry::Insert(BlockRef {
                block_name: format!("B{to}"),
                transform: SourceTransform::IDENTITY,
                normal: [0.0, 0.0, 1.0],
                grid: ArrayGrid::SINGLE,
            }),
            style: EntityStyle::absent(),
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(*handle)),
                class_name: "INSERT".into(),
                block_path: InstancePath::root(),
                layer: "0".into(),
            },
        }
    };
    let mut blocks = BTreeMap::new();
    for k in 0..levels {
        blocks.insert(format!("B{k}"), vec![insert((k + 1) % levels, &mut handle)]);
    }
    let model = vec![insert(0, &mut handle)];
    Mock {
        caps: ProviderCaps::new(),
        layouts: vec![(
            LayoutState {
                id: LayoutId(0),
                name: "Model".into(),
                kind: LayoutKind::Model,
                block_name: "*Model_Space".into(),
            },
            model,
        )],
        blocks,
        layers: Vec::new(),
        fail_after: None,
    }
}
