//! Issue #1075 (N4.4): nested block and INSERT flattening with bounded recursion and lineage.
//!
//! The provider is an in-memory drawing, so nothing here needs a CAD crate. The nested-blocks
//! numbers are those of `nested-blocks-001` in libviprs/pvec-gold-fixtures (a translation, a 90
//! degree rotation and a non-uniform scale that do not commute), copied as exact binary64 values
//! because that repository is not a dependency of this one. The `O1.6` amplification probe has no
//! drawing yet, so the amplification fixtures here are generated with a known depth and fan-out
//! and their instance counts are asserted as numbers.
#![cfg(feature = "pvec-normalized")]

use std::cell::Cell;
use std::collections::BTreeMap;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::flatten::{
    BlockResolver, FlatEntity, FlattenBounds, FlattenError, Flattened, LineageStore, SkipReason,
};
use libviprs::pvec::provenance::{NoProvenance, ProvenanceStore};
use libviprs::pvec::transform::ComposedTransform;
use libviprs::pvec::{
    ArrayGrid, BlockRef, DwgProvider, EntityCursor, EntityScope, FidelityClass, InstancePath,
    InstanceStep, LayerState, LayoutId, LayoutState, Limit, Normalized2DGeometry, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceLimits, SourceTransform,
};

fn h(raw: u64) -> ItemHandle {
    ItemHandle::new(raw)
}

fn stub(handle: u64, class: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(h(handle)),
        class_name: class.to_string(),
        block_path: InstancePath::root(),
        layer: "0".to_string(),
    }
}

fn line(handle: u64, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
        provenance: stub(handle, "LINE"),
    }
}

fn insert_with(
    handle: Option<u64>,
    block: &str,
    m: [f64; 6],
    normal: [f64; 3],
    grid: ArrayGrid,
) -> RawEntity {
    RawEntity {
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.to_string(),
            transform: SourceTransform::new(m).unwrap(),
            normal,
            grid,
        }),
        provenance: ProvenanceStub {
            handle: handle.map(h),
            ..stub(0, "INSERT")
        },
    }
}

fn insert(handle: u64, block: &str, m: [f64; 6]) -> RawEntity {
    insert_with(Some(handle), block, m, [0.0, 0.0, 1.0], ArrayGrid::SINGLE)
}

const ID: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

struct Mock {
    model: Vec<RawEntity>,
    blocks: BTreeMap<String, Vec<RawEntity>>,
    caps: ProviderCaps,
    opened: Cell<u64>,
}

impl Mock {
    fn new(model: Vec<RawEntity>, blocks: Vec<(&str, Vec<RawEntity>)>) -> Self {
        Self {
            model,
            blocks: blocks
                .into_iter()
                .map(|(n, e)| (n.to_string(), e))
                .collect(),
            caps: ProviderCaps::new(),
            opened: Cell::new(0),
        }
    }
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Err(ProviderError::Unrecognised {
            provider: "mock",
            detail: String::new(),
        })
    }
    fn name(&self) -> &'static str {
        "mock"
    }
    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(Vec::new())
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(Vec::new())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        self.opened.set(self.opened.get() + 1);
        let (items, path) = match scope {
            EntityScope::Layout(_) => (self.model.clone(), InstancePath::root()),
            EntityScope::BlockDefinition { name, reached_via } => (
                self.blocks
                    .get(name)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("block {name}")))?
                    .clone(),
                reached_via.clone(),
            ),
            _ => return Err(ProviderError::NoSuchScope("scope".into())),
        };
        let stamped: Vec<RawEntity> = items
            .into_iter()
            .map(|mut e| {
                e.provenance.block_path = path.clone();
                e
            })
            .collect();
        Ok(Box::new(Cursor(stamped.into_iter())))
    }
}

fn run(
    mock: &Mock,
    bounds: FlattenBounds,
) -> Result<(Flattened, ProvenanceStore, LineageStore), FlattenError> {
    let mut store = ProvenanceStore::new();
    let mut lineage = LineageStore::new();
    let out = BlockResolver::new(mock, bounds).flatten(
        &EntityScope::Layout(LayoutId(0)),
        "fixture.dwg",
        "Model",
        &mut store,
        &mut lineage,
    )?;
    Ok((out, store, lineage))
}

fn bits(p: [f64; 2]) -> [u64; 2] {
    [p[0].to_bits(), p[1].to_bits()]
}

// nested-blocks-001: outer T(4,1), middle R90, inner S(2,1), line (0.5,0.25)-(3.75,2.5).
fn nested_blocks_001() -> Mock {
    let t = [1.0, 0.0, 0.0, 1.0, 4.0, 1.0];
    let r = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0];
    let s = [2.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    Mock::new(
        vec![insert(0x10, "OUTER", t)],
        vec![
            ("OUTER", vec![insert(0x20, "MIDDLE", r)]),
            ("MIDDLE", vec![insert(0x30, "INNER", s)]),
            ("INNER", vec![line(0x40, [0.5, 0.25], [3.75, 2.5])]),
        ],
    )
}

#[test]
fn nested_blocks_001_lands_where_the_fixture_says() {
    let mock = nested_blocks_001();
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.entities.len(), 1);
    let leaf = &out.entities[0];
    let RawGeometry::Line { start, end } = &leaf.entity.geometry else {
        panic!("not a line");
    };
    // 0x400e000000000000 is 3.75, 0x4000000000000000 is 2.0, 0x3ff8... is 1.5, 0x4021 is 8.5.
    assert_eq!(
        bits(leaf.map_point(*start)),
        [3.75f64.to_bits(), 2.0f64.to_bits()]
    );
    assert_eq!(
        bits(leaf.map_point(*end)),
        [1.5f64.to_bits(), 8.5f64.to_bits()]
    );
}

#[test]
fn the_wrong_multiplication_order_is_not_what_we_produce() {
    let mock = nested_blocks_001();
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    let leaf = &out.entities[0];
    let got = leaf.map_point([0.5, 0.25, 0.0]);
    assert_ne!(bits(got), [(-2.5f64).to_bits(), 4.5f64.to_bits()]);
}

#[test]
fn leaf_is_the_composed_transform_applied_once() {
    // A chain chosen so that mapping level by level rounds differently from mapping once.
    let rot = |a: f64| [a.cos(), -a.sin(), a.sin(), a.cos(), 0.1, 0.3];
    let levels = [
        [0.1, 0.0, 0.0, 0.3, 0.7, 0.1],
        rot(0.7),
        [3.0, 0.0, 0.0, 1.1, 0.0, 0.2],
        rot(-0.3),
        [0.9, 0.0, 0.0, 0.7, 5.5, 1.0e-3],
    ];
    let mut blocks = Vec::new();
    let names = ["L0", "L1", "L2", "L3", "L4"];
    for i in 0..4 {
        blocks.push((
            names[i],
            vec![insert(0x20 + i as u64, names[i + 1], levels[i + 1])],
        ));
    }
    blocks.push(("L4", vec![line(0x99, [0.3, 0.7], [11.1, -4.9])]));
    let mock = Mock::new(vec![insert(0x10, "L0", levels[0])], blocks);
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    let leaf = &out.entities[0];

    let steps: Vec<(InstanceStep, SourceTransform)> = levels
        .iter()
        .enumerate()
        .map(|(i, m)| {
            (
                InstanceStep {
                    block_name: names[i].to_string(),
                    insert_handle: h(if i == 0 { 0x10 } else { 0x20 + i as u64 - 1 }),
                },
                SourceTransform::new(*m).unwrap(),
            )
        })
        .collect();
    let composed = ComposedTransform::compose(steps).unwrap();
    assert_eq!(
        leaf.matrix().map(f64::to_bits),
        composed.matrix().map(f64::to_bits)
    );

    let [m00, m01, m10, m11, tx, ty] = composed.matrix();
    let p = [0.3, 0.7];
    let once = [m00 * p[0] + m01 * p[1] + tx, m10 * p[0] + m11 * p[1] + ty];
    assert_eq!(bits(leaf.map_point([p[0], p[1], 0.0])), bits(once));

    // The two-step route (map through every level, innermost first) rounds again at each level,
    // and on this chain it lands on different bits, which is what makes the assertion above bite.
    let mut q = p;
    for m in levels.iter().rev() {
        q = [
            m[0] * q[0] + m[1] * q[1] + m[4],
            m[2] * q[0] + m[3] * q[1] + m[5],
        ];
    }
    assert_ne!(bits(q), bits(once), "the chain must be rounding-sensitive");
}

fn self_reference() -> Mock {
    Mock::new(
        vec![insert(0x10, "LOOP", ID)],
        vec![(
            "LOOP",
            vec![insert(0x11, "LOOP", ID), line(0x12, [0.0, 0.0], [1.0, 1.0])],
        )],
    )
}

#[test]
fn self_referencing_block_terminates_with_block_recursion_limit() {
    let mock = self_reference();
    let err = run(&mock, FlattenBounds::default()).unwrap_err();
    let FlattenError::BlockRecursionLimit { depth, block_path } = err else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!(depth, 2);
    assert_eq!(block_path.to_string(), "LOOP@10/LOOP@11");
    // The cycle is refused when met: the layout and one block cursor were opened, nothing more.
    assert_eq!(mock.opened.get(), 2);
}

#[test]
fn self_reference_stays_inside_a_tiny_allocation_budget() {
    let mock = self_reference();
    let mut limits = SourceLimits::default();
    limits.max_allocation_bytes = 4096;
    let err = run(&mock, FlattenBounds::new(limits, 1_000)).unwrap_err();
    assert!(
        matches!(err, FlattenError::BlockRecursionLimit { .. }),
        "the cycle must be refused before the budget runs out, got {err:?}"
    );
}

#[test]
fn mutual_cycle_is_refused_by_name_path() {
    let mock = Mock::new(
        vec![insert(0x10, "A", ID)],
        vec![
            ("A", vec![insert(0x20, "B", ID)]),
            ("B", vec![insert(0x30, "a", ID)]),
        ],
    );
    let err = run(&mock, FlattenBounds::default()).unwrap_err();
    let FlattenError::BlockRecursionLimit { depth, block_path } = err else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!(depth, 3);
    assert_eq!(block_path.to_string(), "A@10/B@20/a@30");
}

// Depth `depth`, each block holding `fan` INSERTs of the next, one line at the bottom.
fn amplifier(depth: usize, fan: u64) -> Mock {
    let names: Vec<String> = (0..depth).map(|i| format!("AMP{i}")).collect();
    let mut next_handle = 0x100u64;
    let mut handle = || {
        next_handle += 1;
        next_handle
    };
    let step = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let model: Vec<RawEntity> = (0..fan)
        .map(|_| insert(handle(), &names[0], step))
        .collect();
    let mut blocks = Vec::new();
    for i in 0..depth {
        let body = if i + 1 < depth {
            (0..fan)
                .map(|_| insert(handle(), &names[i + 1], step))
                .collect()
        } else {
            vec![line(handle(), [0.0, 0.0], [1.0, 0.0])]
        };
        blocks.push((names[i].as_str().to_owned(), body));
    }
    let mut m = Mock::new(model, Vec::new());
    m.blocks = blocks.into_iter().collect();
    m
}

#[test]
fn amplification_fixture_expands_to_exactly_the_expected_instance_count() {
    // fan 3, depth 5: 3 + 9 + 27 + 81 + 243 instances, 243 leaves.
    let mock = amplifier(5, 3);
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.instances, 363);
    assert_eq!(out.entities.len(), 243);
    // Every leaf sits five translations of +1 in x from the origin.
    for leaf in &out.entities {
        assert_eq!(leaf.map_point([0.0, 0.0, 0.0]), [5.0, 0.0]);
    }
}

#[test]
fn expanded_instance_ceiling_is_a_typed_error() {
    let mock = amplifier(5, 3);
    let mut limits = SourceLimits::default();
    limits.max_expanded_instances = 100;
    let err = run(&mock, FlattenBounds::new(limits, 1_000)).unwrap_err();
    let FlattenError::Limit {
        limit,
        observed,
        ceiling,
        block_path,
    } = err
    else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!(limit, Limit::ExpandedInstances);
    assert_eq!(ceiling, 100);
    assert!(observed > 100);
    assert!(block_path.depth() >= 1);
}

#[test]
fn depth_ceiling_is_a_typed_error_naming_the_path() {
    let mock = amplifier(5, 1);
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 3;
    let err = run(&mock, FlattenBounds::new(limits, 1_000)).unwrap_err();
    let FlattenError::Limit {
        limit,
        observed,
        ceiling,
        block_path,
    } = err
    else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!((limit, observed, ceiling), (Limit::BlockDepth, 4, 3));
    assert_eq!(block_path.depth(), 4);
}

#[test]
fn total_entity_ceiling_is_a_typed_error() {
    let mock = amplifier(4, 3);
    let mut limits = SourceLimits::default();
    limits.max_entities = 50;
    let err = run(&mock, FlattenBounds::new(limits, 1_000)).unwrap_err();
    let FlattenError::Limit { limit, ceiling, .. } = err else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!((limit, ceiling), (Limit::Entities, 50));
}

#[test]
fn entities_per_block_ceiling_is_a_typed_error() {
    let mock = Mock::new(
        vec![insert(0x10, "BIG", ID)],
        vec![(
            "BIG",
            (0..10)
                .map(|i| line(0x20 + i, [0.0, 0.0], [1.0, 1.0]))
                .collect(),
        )],
    );
    let err = run(&mock, FlattenBounds::new(SourceLimits::default(), 4)).unwrap_err();
    let FlattenError::EntitiesPerBlock {
        block,
        observed,
        ceiling,
        block_path,
    } = err
    else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!((block.as_str(), observed, ceiling), ("BIG", 5, 4));
    assert_eq!(block_path.to_string(), "BIG@10");
}

#[test]
fn minsert_grid_expands_rows_times_columns_in_the_inserts_frame() {
    let grid = ArrayGrid {
        rows: 2,
        columns: 3,
        row_spacing: 10.0,
        column_spacing: 5.0,
    };
    // Rotated 90 degrees: the insert's x axis points along +y.
    let m = [0.0, -1.0, 1.0, 0.0, 100.0, 200.0];
    let mock = Mock::new(
        vec![insert_with(Some(0x10), "CELL", m, [0.0, 0.0, 1.0], grid)],
        vec![("CELL", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.instances, 6);
    let origins: Vec<[f64; 2]> = out
        .entities
        .iter()
        .map(|e| e.map_point([0.0, 0.0, 0.0]))
        .collect();
    // Row-major: row 0 columns 0..3, then row 1. Columns step along the rotated x (+y), rows along
    // the rotated y (-x).
    assert_eq!(
        origins,
        vec![
            [100.0, 200.0],
            [100.0, 205.0],
            [100.0, 210.0],
            [90.0, 200.0],
            [90.0, 205.0],
            [90.0, 210.0],
        ]
    );
}

#[test]
fn minsert_grid_counts_toward_the_instance_ceiling() {
    let grid = ArrayGrid {
        rows: 100,
        columns: 100,
        row_spacing: 1.0,
        column_spacing: 1.0,
    };
    let mock = Mock::new(
        vec![insert_with(Some(0x10), "CELL", ID, [0.0, 0.0, 1.0], grid)],
        vec![("CELL", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let mut limits = SourceLimits::default();
    limits.max_expanded_instances = 9_999;
    let err = run(&mock, FlattenBounds::new(limits, 1_000)).unwrap_err();
    assert!(matches!(
        err,
        FlattenError::Limit {
            limit: Limit::ExpandedInstances,
            observed: 10_000,
            ..
        }
    ));
}

#[test]
fn negative_z_normal_mirrors_x_outside_the_inserts_own_map() {
    let m = [1.0, 0.0, 0.0, 1.0, 10.0, 0.0];
    let mock = Mock::new(
        vec![insert_with(
            Some(0x10),
            "B",
            m,
            [0.0, 0.0, -1.0],
            ArrayGrid::SINGLE,
        )],
        vec![("B", vec![line(0x20, [2.0, 3.0], [4.0, 5.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    // OCS to WCS for N = -Z is x -> -x, applied after the insertion translation.
    assert_eq!(out.entities[0].map_point([2.0, 3.0, 0.0]), [-12.0, 3.0]);
}

#[test]
fn tilted_normal_is_skipped_and_classified_not_dropped() {
    let mock = Mock::new(
        vec![insert_with(
            Some(0x10),
            "B",
            ID,
            [1.0, 0.0, 0.0],
            ArrayGrid::SINGLE,
        )],
        vec![("B", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert!(out.entities.is_empty());
    assert_eq!(out.skipped.len(), 1);
    assert_eq!(out.skipped[0].reason, SkipReason::NormalNotZ);
    assert_eq!(out.skipped[0].class, FidelityClass::RequiresExtension);
}

#[test]
fn anonymous_block_without_content_is_requires_extension() {
    let mock = Mock::new(
        vec![insert(0x10, "*U17", ID), insert(0x11, "GONE", ID)],
        vec![],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.skipped.len(), 2);
    assert_eq!(out.skipped[0].reason, SkipReason::AnonymousBlockUnresolved);
    assert_eq!(out.skipped[0].class, FidelityClass::RequiresExtension);
    assert_eq!(out.skipped[1].reason, SkipReason::BlockNotFound);
    assert_eq!(out.skipped[1].class, FidelityClass::Unsupported);
}

#[test]
fn anonymous_block_with_resolved_content_is_flattened() {
    let mock = Mock::new(
        vec![insert(0x10, "*U17", [1.0, 0.0, 0.0, 1.0, 5.0, 0.0])],
        vec![("*U17", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert!(out.skipped.is_empty());
    assert_eq!(out.entities[0].map_point([0.0, 0.0, 0.0]), [5.0, 0.0]);
}

#[test]
fn insert_without_a_handle_is_skipped_not_dropped() {
    let mock = Mock::new(
        vec![insert_with(
            None,
            "B",
            ID,
            [0.0, 0.0, 1.0],
            ArrayGrid::SINGLE,
        )],
        vec![("B", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.skipped[0].reason, SkipReason::MissingInsertHandle);
}

#[test]
fn singular_insert_map_is_a_compose_error() {
    let mock = Mock::new(
        vec![insert(0x10, "B", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])],
        vec![("B", vec![line(0x20, [0.0, 0.0], [1.0, 0.0])])],
    );
    let err = run(&mock, FlattenBounds::default()).unwrap_err();
    assert!(matches!(err, FlattenError::Compose(_)), "{err:?}");
}

#[test]
fn the_walk_is_deterministic_and_depth_first() {
    let mock = amplifier(3, 2);
    let a = run(&mock, FlattenBounds::default()).unwrap().0;
    let b = run(&mock, FlattenBounds::default()).unwrap().0;
    let key = |e: &FlatEntity| {
        (
            e.entity.provenance.handle,
            e.entity.provenance.block_path.to_string(),
            e.matrix().map(f64::to_bits),
        )
    };
    assert_eq!(
        a.entities.iter().map(key).collect::<Vec<_>>(),
        b.entities.iter().map(key).collect::<Vec<_>>()
    );
    // Layout entities that are plain leaves keep their place between the expanded blocks.
    let mock = Mock::new(
        vec![
            line(1, [0.0, 0.0], [1.0, 0.0]),
            insert(2, "B", ID),
            line(3, [0.0, 0.0], [1.0, 0.0]),
        ],
        vec![("B", vec![line(4, [0.0, 0.0], [1.0, 0.0])])],
    );
    let (out, _, _) = run(&mock, FlattenBounds::default()).unwrap();
    let order: Vec<u64> = out
        .entities
        .iter()
        .map(|e| e.entity.provenance.handle.unwrap().get())
        .collect();
    assert_eq!(order, vec![1, 4, 3]);
}

// A chain of unique handles so a path can be checked against the drawing.
fn resolves_to_source(mock: &Mock, path: &InstancePath) -> bool {
    let mut container: &[RawEntity] = &mock.model;
    for step in path.steps() {
        let found = container.iter().any(|e| {
            matches!(&e.geometry, RawGeometry::Insert(r) if r.block_name == step.block_name)
                && e.provenance.handle == Some(step.insert_handle)
        });
        if !found {
            return false;
        }
        match mock.blocks.get(&step.block_name) {
            Some(b) => container = b,
            None => return false,
        }
    }
    true
}

#[test]
fn every_emitted_entity_has_a_lineage_record_that_resolves_to_the_source() {
    let mock = amplifier(4, 2);
    let (out, store, lineage) = run(&mock, FlattenBounds::default()).unwrap();
    assert_eq!(out.entities.len(), 16);
    assert_eq!(lineage.len(), out.entities.len());
    for leaf in &out.entities {
        let handle = leaf.lineage_handle().expect("a kept provenance handle");
        let rec = lineage.get(handle).expect("a lineage record");
        assert_eq!(rec.depth(), 4, "the innermost level must be there");
        assert_eq!(rec.block_path(), leaf.entity.provenance.block_path);
        assert!(resolves_to_source(&mock, &rec.block_path()));
        // The provenance record on the same handle carries the same path and the composed map.
        let p = store.get(handle).unwrap();
        assert_eq!(p.block_path(), &rec.block_path());
        assert_eq!(
            p.transform().coefficients().map(f64::to_bits),
            leaf.matrix().map(f64::to_bits)
        );
        // Each level's recorded transform is the INSERT's own.
        for level in rec.levels() {
            assert_eq!(level.transform, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        }
    }
}

#[test]
fn a_leaf_a_layout_owns_has_an_empty_lineage() {
    let mock = Mock::new(vec![line(1, [0.0, 0.0], [1.0, 0.0])], vec![]);
    let (out, _, lineage) = run(&mock, FlattenBounds::default()).unwrap();
    let rec = lineage
        .get(out.entities[0].lineage_handle().unwrap())
        .unwrap();
    assert_eq!(rec.depth(), 0);
    assert!(rec.block_path().is_root());
}

#[test]
fn a_sink_that_keeps_nothing_records_no_lineage() {
    let mock = amplifier(3, 2);
    let mut lineage = LineageStore::new();
    let out = BlockResolver::new(&mock, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(LayoutId(0)),
            "f.dwg",
            "Model",
            &mut NoProvenance,
            &mut lineage,
        )
        .unwrap();
    assert_eq!(out.entities.len(), 8);
    assert!(lineage.is_empty());
    assert!(out.entities.iter().all(|e| e.lineage_handle().is_none()));
}

#[test]
fn lineage_recording_does_not_change_the_geometry_record_size() {
    // Lineage lives in `LineageStore`, keyed by the provenance handle, so the record is the same
    // size whether lineage is on or off. The number is pinned for 64-bit targets, so a field added
    // to `Normalized2DGeometry` for lineage changes it and fails here.
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        std::mem::size_of::<Normalized2DGeometry>(),
        PINNED_GEOMETRY_SIZE
    );
    let mock = amplifier(2, 2);
    let _ = run(&mock, FlattenBounds::default()).unwrap();
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        std::mem::size_of::<Normalized2DGeometry>(),
        PINNED_GEOMETRY_SIZE
    );
}

#[cfg(target_pointer_width = "64")]
const PINNED_GEOMETRY_SIZE: usize = 176;

// ---- #1277 (N3.6a): SourceLimits carried through the ViewProfile options path ---------------

use libviprs::pvec::ViewProfile;

#[test]
fn n3_6a_1277_default_profile_carries_the_default_limits() {
    assert_eq!(
        ViewProfile::default_profile().limits,
        SourceLimits::default()
    );
}

#[test]
fn n3_6a_1277_with_limits_sets_the_field_and_touches_nothing_else() {
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 3;
    let p = ViewProfile::default_profile().with_limits(limits);
    assert_eq!(p.limits, limits);
    let mut expected = ViewProfile::default_profile();
    expected.limits = limits;
    assert_eq!(p, expected);
}

#[test]
fn n3_6a_1277_limits_are_not_a_view_knob() {
    let mut limits = SourceLimits::default();
    limits.max_entities = 7;
    let a = ViewProfile::default_profile().knobs();
    let b = ViewProfile::default_profile().with_limits(limits).knobs();
    assert_eq!(a, b);
}

#[test]
fn n3_6a_1277_bounds_from_a_profile_hold_its_limits() {
    let mut limits = SourceLimits::default();
    limits.max_expanded_instances = 99;
    let p = ViewProfile::default_profile().with_limits(limits);
    assert_eq!(FlattenBounds::from_profile(&p).limits, limits);
    assert_eq!(
        FlattenBounds::from_profile(&ViewProfile::default_profile()),
        FlattenBounds::default()
    );
}

#[test]
fn n3_6a_1277_depth_override_set_on_the_profile_reaches_the_guard() {
    let mock = amplifier(5, 1);
    // The default profile walks this fixture fine.
    let ok = FlattenBounds::from_profile(&ViewProfile::default_profile());
    run(&mock, ok).unwrap();
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 3;
    let p = ViewProfile::default_profile().with_limits(limits);
    let err = run(&mock, FlattenBounds::from_profile(&p)).unwrap_err();
    let FlattenError::Limit {
        limit,
        observed,
        ceiling,
        ..
    } = err
    else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!((limit, observed, ceiling), (Limit::BlockDepth, 4, 3));
}

#[test]
fn n3_6a_1277_entity_override_set_on_the_profile_reaches_the_guard() {
    let mock = amplifier(5, 1);
    let mut limits = SourceLimits::default();
    limits.max_entities = 2;
    let p = ViewProfile::default_profile().with_limits(limits);
    let err = run(&mock, FlattenBounds::from_profile(&p)).unwrap_err();
    let FlattenError::Limit { limit, ceiling, .. } = err else {
        panic!("wrong error: {err:?}");
    };
    assert_eq!((limit, ceiling), (Limit::Entities, 2));
}
