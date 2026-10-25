//! The contract of the DWG provider traits (N4.1, #1072), driven by a provider
//! that has no CAD dependency behind it.
//!
//! Everything here compiles and runs with default features. `MockDwgProvider` is a
//! drawing held in memory, and `extract` is a stand-in for the N4.2 extraction
//! walk: it descends INSERTs through the traits and classifies what it meets, and
//! it uses nothing but the module under test. If this file needed a CAD crate, the
//! boundary would be leaking.
//!
//! Comparisons of floats are done on bits, field by field, because `RawEntity` has
//! no `PartialEq` on purpose (see `src/pvec/provider.rs`).

use std::collections::BTreeMap;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact, SourceTransform,
};

fn h(raw: u64) -> ItemHandle {
    ItemHandle::new(raw)
}

fn prov(handle: u64, class: &str, layer: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(h(handle)),
        class_name: class.to_string(),
        block_path: InstancePath::root(),
        layer: layer.to_string(),
    }
}

fn line(handle: u64, layer: &str, a: [f64; 3], b: [f64; 3]) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Line { start: a, end: b },
        provenance: prov(handle, "LINE", layer),
    }
}

fn insert(handle: u64, layer: &str, block: &str, at: [f64; 2], rot: f64, scale: f64) -> RawEntity {
    let (s, c) = rot.sin_cos();
    let transform =
        SourceTransform::new([c * scale, -s * scale, s * scale, c * scale, at[0], at[1]])
            .expect("finite");
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.to_string(),
            transform,
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
        }),
        provenance: prov(handle, "INSERT", layer),
    }
}

fn spline(
    handle: u64,
    knots: SourceFact<Vec<f64>>,
    weights: SourceFact<Vec<f64>>,
    periodic: SourceFact<bool>,
) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Spline {
            degree: 3,
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            knots,
            weights,
            periodic,
            closed: false,
        },
        provenance: prov(handle, "SPLINE", "VC_ON"),
    }
}

/// A drawing in memory. Layouts and blocks are plain vectors of entities.
struct MockDwgProvider {
    caps: ProviderCaps,
    layouts: Vec<(LayoutState, Vec<RawEntity>)>,
    blocks: BTreeMap<String, Vec<RawEntity>>,
    layers: Vec<LayerState>,
}

struct MockCursor {
    items: std::vec::IntoIter<RawEntity>,
}

impl EntityCursor for MockCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.items.next())
    }
}

impl DwgProvider for MockDwgProvider {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK")))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        if !Self::probe(source)? {
            return Err(ProviderError::Unrecognised {
                provider: "mock",
                detail: "no MOCK stamp".into(),
            });
        }
        Ok(MockDwgProvider::nested_drawing())
    }

    fn name(&self) -> &'static str {
        "mock"
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
            EntityScope::BlockDefinition { name, reached_via } => {
                let e = self
                    .blocks
                    .get(name)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("block {name}")))?;
                (e.clone(), reached_via.clone())
            }
            _ => return Err(ProviderError::NoSuchScope("unknown scope kind".into())),
        };
        let stamped: Vec<RawEntity> = items
            .into_iter()
            .map(|mut e| {
                e.provenance.block_path = path.clone();
                e
            })
            .collect();
        Ok(Box::new(MockCursor {
            items: stamped.into_iter(),
        }))
    }
}

impl MockDwgProvider {
    fn full_caps() -> ProviderCaps {
        Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
            c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
        })
    }

    /// The shape of the vc_nested worked example: model space holds two INSERTs of
    /// the inner block and one of the outer, and the outer reaches the inner through
    /// the middle block.
    fn nested_drawing() -> Self {
        let model = LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        };
        let paper = LayoutState {
            id: LayoutId(1),
            name: "Layout1".into(),
            kind: LayoutKind::Paper,
            block_name: "*Paper_Space".into(),
        };
        let mut blocks = BTreeMap::new();
        blocks.insert(
            "VC_INNER".to_string(),
            vec![line(0x4C, "VC_ON", [0.0; 3], [1.0, 0.0, 0.0])],
        );
        blocks.insert(
            "VC_MID".to_string(),
            vec![
                line(0x50, "VC_ON", [0.0; 3], [0.0, 1.0, 0.0]),
                insert(
                    0x51,
                    "VC_ON",
                    "VC_INNER",
                    [1.0, 0.0],
                    std::f64::consts::FRAC_PI_2,
                    2.0,
                ),
            ],
        );
        blocks.insert(
            "VC_OUTER".to_string(),
            vec![insert(
                0x55,
                "VC_ON",
                "VC_MID",
                [3.0, 0.0],
                std::f64::consts::FRAC_PI_2,
                1.0,
            )],
        );
        MockDwgProvider {
            caps: Self::full_caps(),
            layouts: vec![
                (
                    model,
                    vec![
                        insert(0x56, "VC_ON", "VC_INNER", [10.0, 0.0], 0.0, 1.0),
                        insert(0x57, "VC_ON", "VC_INNER", [20.0, 0.0], 0.0, 3.0),
                        insert(0x58, "VC_ON", "VC_OUTER", [100.0, 100.0], 0.0, 2.0),
                    ],
                ),
                (paper, vec![]),
            ],
            blocks,
            layers: vec![
                LayerState {
                    name: "0".into(),
                    handle: Some(h(0x10)),
                    off: false,
                    frozen: false,
                    locked: false,
                },
                LayerState {
                    name: "VC_ON".into(),
                    handle: Some(h(0x11)),
                    off: false,
                    frozen: false,
                    locked: true,
                },
            ],
        }
    }

    fn with_entities(model: Vec<RawEntity>, caps: ProviderCaps) -> Self {
        let mut p = Self::nested_drawing();
        p.layouts[0].1 = model;
        p.caps = caps;
        p
    }
}

// ---- the N4.2 stand-in -----------------------------------------------------------

/// A local mirror of the outcomes N3.5 will define. It is not the real enum and does
/// not pretend to be: it only lets this file say what a consumer does with a fact.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Geometry,
    RequiresExtension(Vec<Facet>),
    Unmapped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Leaf {
    path: String,
    class: String,
    outcome: Outcome,
}

fn classify(e: &RawEntity) -> Outcome {
    match &e.geometry {
        RawGeometry::Unmapped { .. } => Outcome::Unmapped,
        _ => {
            let missing = e.not_reportable();
            if missing.is_empty() {
                Outcome::Geometry
            } else {
                // Never rebuild the curve from points: report what is missing.
                Outcome::RequiresExtension(missing)
            }
        }
    }
}

fn walk(
    p: &dyn DwgProvider,
    scope: &EntityScope,
    depth: usize,
    out: &mut Vec<Leaf>,
) -> Result<(), ProviderError> {
    let mut cursor = p.entities(scope)?;
    while let Some(e) = cursor.next_entity()? {
        if let RawGeometry::Insert(r) = &e.geometry {
            if depth < 8 {
                let via = e.child_path().expect("insert with a handle");
                walk(
                    p,
                    &EntityScope::BlockDefinition {
                        name: r.block_name.clone(),
                        reached_via: via,
                    },
                    depth + 1,
                    out,
                )?;
            }
        } else {
            out.push(Leaf {
                path: e.provenance.block_path.to_string(),
                class: e.provenance.class_name.clone(),
                outcome: classify(&e),
            });
        }
    }
    Ok(())
}

fn extract(p: &dyn DwgProvider) -> Vec<Leaf> {
    let mut out = Vec::new();
    for layout in p.layouts().expect("layouts") {
        if layout.kind == LayoutKind::Model {
            walk(p, &EntityScope::Layout(layout.id), 0, &mut out).expect("walk");
        }
    }
    out
}

// ---- tests -----------------------------------------------------------------------

#[test]
fn mock_provider_drives_extraction_with_no_cad_dependency() {
    // Opened through the trait's own constructors, the way a generic caller would.
    assert!(MockDwgProvider::probe(CadSource::Bytes(b"MOCK1")).unwrap());
    assert!(!MockDwgProvider::probe(CadSource::Bytes(b"AC1032")).unwrap());
    assert!(matches!(
        MockDwgProvider::open(CadSource::Bytes(b"AC1032")),
        Err(ProviderError::Unrecognised { .. })
    ));
    let p = MockDwgProvider::open(CadSource::Bytes(b"MOCK1")).unwrap();

    let leaves = extract(&p);
    let seen: Vec<(String, String)> = leaves
        .iter()
        .map(|l| (l.path.clone(), l.class.clone()))
        .collect();
    assert_eq!(
        seen,
        vec![
            ("VC_INNER@56".to_string(), "LINE".to_string()),
            ("VC_INNER@57".to_string(), "LINE".to_string()),
            ("VC_OUTER@58/VC_MID@55".to_string(), "LINE".to_string()),
            (
                "VC_OUTER@58/VC_MID@55/VC_INNER@51".to_string(),
                "LINE".to_string()
            ),
        ],
        "four leaves, each with the chain of INSERTs it was reached through"
    );
    assert!(leaves.iter().all(|l| l.outcome == Outcome::Geometry));

    // The same handle repeats across instances, which is why the path exists.
    let mut handles = Vec::new();
    for insert_handle in [0x56, 0x57] {
        let scope = EntityScope::BlockDefinition {
            name: "VC_INNER".into(),
            reached_via: InstancePath::root().child("VC_INNER", h(insert_handle)),
        };
        let e = p.entities(&scope).unwrap().collect_all().unwrap();
        handles.push(e[0].provenance.handle.unwrap().get());
    }
    assert_eq!(handles, vec![0x4C, 0x4C]);
}

#[test]
#[cfg_attr(miri, ignore)] // reads its own source file, which Miri isolation blocks
fn this_file_names_no_cad_crate() {
    let src = std::fs::read_to_string(file!()).or_else(|_| {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/pvec_provider_contract.rs"
        ))
    });
    let src = src.expect("read own source");
    let foreign = ["acad", "sharp"].concat();
    let foreign2 = ["acad", "rust"].concat();
    for line in src.lines().filter(|l| l.trim_start().starts_with("use ")) {
        assert!(!line.contains(&foreign), "foreign import: {line}");
        assert!(!line.contains(&foreign2), "foreign import: {line}");
        assert!(!line.contains("lopdf"), "foreign import: {line}");
    }
}

#[test]
fn instance_methods_are_object_safe() {
    let p = MockDwgProvider::nested_drawing();
    let d: &dyn DwgProvider = &p;
    assert_eq!(d.name(), "mock");
    assert_eq!(d.layouts().unwrap().len(), 2);
    assert_eq!(d.layers().unwrap().len(), 2);
    assert!(d.caps().get(Facet::KnotMultiplicity).is_some());
    let mut c = d.entities(&EntityScope::Layout(LayoutId(0))).unwrap();
    assert!(c.next_entity().unwrap().is_some());
}

fn assert_entity_identical(i: usize, a: &RawEntity, b: &RawEntity) {
    assert_eq!(a.provenance, b.provenance, "entity {i}: provenance");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    match (&a.geometry, &b.geometry) {
        (RawGeometry::Line { start: s1, end: e1 }, RawGeometry::Line { start: s2, end: e2 }) => {
            assert_eq!(bits(s1), bits(s2), "entity {i}: start");
            assert_eq!(bits(e1), bits(e2), "entity {i}: end");
        }
        (RawGeometry::Insert(r1), RawGeometry::Insert(r2)) => {
            assert_eq!(r1.block_name, r2.block_name, "entity {i}: block name");
            assert_eq!(
                bits(&r1.transform.coefficients()),
                bits(&r2.transform.coefficients()),
                "entity {i}: transform"
            );
            assert_eq!(bits(&r1.normal), bits(&r2.normal), "entity {i}: normal");
            assert_eq!(r1.grid.rows, r2.grid.rows, "entity {i}: rows");
            assert_eq!(r1.grid.columns, r2.grid.columns, "entity {i}: columns");
            assert_eq!(r1.grid.row_spacing.to_bits(), r2.grid.row_spacing.to_bits());
            assert_eq!(
                r1.grid.column_spacing.to_bits(),
                r2.grid.column_spacing.to_bits()
            );
        }
        (
            RawGeometry::Polyline {
                vertices: v1,
                closed: c1,
                elevation: z1,
                normal: n1,
            },
            RawGeometry::Polyline {
                vertices: v2,
                closed: c2,
                elevation: z2,
                normal: n2,
            },
        ) => {
            assert_eq!(c1, c2, "entity {i}: closed");
            assert_eq!(z1.to_bits(), z2.to_bits(), "entity {i}: elevation");
            assert_eq!(bits(n1), bits(n2), "entity {i}: normal");
            assert_eq!(v1.len(), v2.len(), "entity {i}: vertex count");
            for (k, (p, q)) in v1.iter().zip(v2).enumerate() {
                assert_eq!(p.x.to_bits(), q.x.to_bits(), "entity {i} vertex {k}: x");
                assert_eq!(p.y.to_bits(), q.y.to_bits(), "entity {i} vertex {k}: y");
                assert_eq!(
                    p.bulge.to_bits(),
                    q.bulge.to_bits(),
                    "entity {i} vertex {k}: bulge"
                );
            }
        }
        (
            RawGeometry::Spline {
                degree: d1,
                control_points: c1,
                knots: k1,
                weights: w1,
                periodic: p1,
                closed: x1,
            },
            RawGeometry::Spline {
                degree: d2,
                control_points: c2,
                knots: k2,
                weights: w2,
                periodic: p2,
                closed: x2,
            },
        ) => {
            assert_eq!(d1, d2, "entity {i}: degree");
            assert_eq!(x1, x2, "entity {i}: closed");
            assert_eq!(c1.len(), c2.len(), "entity {i}: control point count");
            for (k, (p, q)) in c1.iter().zip(c2).enumerate() {
                assert_eq!(bits(p), bits(q), "entity {i} control point {k}");
            }
            let fact_bits = |f: &SourceFact<Vec<f64>>| match f {
                SourceFact::Present(v) => Some(Some(bits(v))),
                SourceFact::Absent => Some(None),
                _ => None,
            };
            assert_eq!(fact_bits(k1), fact_bits(k2), "entity {i}: knots");
            assert_eq!(fact_bits(w1), fact_bits(w2), "entity {i}: weights");
            assert_eq!(p1.present(), p2.present(), "entity {i}: periodic");
            assert_eq!(
                p1.is_not_reportable(),
                p2.is_not_reportable(),
                "entity {i}: periodic"
            );
        }
        (x, y) => panic!(
            "entity {i}: different kinds, {:?} against {:?}",
            std::mem::discriminant(x),
            std::mem::discriminant(y)
        ),
    }
}

#[test]
fn two_traversals_of_the_same_drawing_are_identical_field_by_field() {
    let mut p = MockDwgProvider::nested_drawing();
    p.layouts[0].1.push(spline(
        0x70,
        SourceFact::Present(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        SourceFact::Absent,
        SourceFact::Present(false),
    ));
    p.layouts[0].1.push(RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Polyline {
            vertices: vec![
                RawVertex {
                    x: 0.0,
                    y: 0.0,
                    bulge: 0.0,
                },
                RawVertex {
                    x: 4.0,
                    y: 0.0,
                    bulge: 0.41421356237309503,
                },
                RawVertex {
                    x: 4.0,
                    y: 4.0,
                    bulge: -0.0,
                },
            ],
            closed: true,
            elevation: 0.0,
            normal: [0.0, 0.0, 1.0],
        },
        provenance: prov(0x71, "LWPOLYLINE", "VC_ON"),
    });

    let scope = EntityScope::Layout(LayoutId(0));
    let first = p.entities(&scope).unwrap().collect_all().unwrap();
    let second = p.entities(&scope).unwrap().collect_all().unwrap();
    assert_eq!(first.len(), 5);
    assert_eq!(first.len(), second.len());
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_entity_identical(i, a, b);
    }
    // And through a nested scope, where the path is stamped on.
    let via = InstancePath::root().child("VC_OUTER", h(0x58));
    let scope = EntityScope::BlockDefinition {
        name: "VC_MID".into(),
        reached_via: via,
    };
    let a = p.entities(&scope).unwrap().collect_all().unwrap();
    let b = p.entities(&scope).unwrap().collect_all().unwrap();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_entity_identical(i, x, y);
    }
}

#[test]
fn a_cursor_is_fused_after_its_end() {
    let p = MockDwgProvider::nested_drawing();
    let mut c = p.entities(&EntityScope::Layout(LayoutId(1))).unwrap();
    assert!(c.next_entity().unwrap().is_none());
    assert!(c.next_entity().unwrap().is_none());
    assert!(c.collect_all().unwrap().is_empty());
}

#[test]
fn an_unknown_scope_is_an_error_not_an_empty_pass() {
    let p = MockDwgProvider::nested_drawing();
    assert!(matches!(
        p.entities(&EntityScope::Layout(LayoutId(9))),
        Err(ProviderError::NoSuchScope(_))
    ));
    assert!(matches!(
        p.entities(&EntityScope::BlockDefinition {
            name: "NOPE".into(),
            reached_via: InstancePath::root()
        }),
        Err(ProviderError::NoSuchScope(_))
    ));
}

#[test]
fn caps_tell_absent_from_cannot_report() {
    // A provider that reads weights: a plain spline has none, and that is Absent.
    let reads = MockDwgProvider::full_caps();
    let plain = spline(
        1,
        SourceFact::Present(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        SourceFact::Absent,
        SourceFact::Present(false),
    );
    assert!(reads.violations(&plain).is_empty());
    assert!(plain.not_reportable().is_empty());
    assert_eq!(
        plain.reported(),
        vec![
            Facet::KnotMultiplicity,
            Facet::Weights,
            Facet::Periodicity,
            // The hand-built entity carries no style, which is Absent and so reported.
            Facet::Colour,
            Facet::Lineweight,
        ]
    );

    // A provider that cannot read knots: the drawing may have them, we cannot say.
    let blind = MockDwgProvider::full_caps().with(
        Facet::KnotMultiplicity,
        Availability::CannotReport,
        FacetSource::DecoderAdapter,
    );
    let unknown = spline(
        2,
        SourceFact::NotReportable,
        SourceFact::Absent,
        SourceFact::Present(false),
    );
    assert_eq!(unknown.not_reportable(), vec![Facet::KnotMultiplicity]);
    assert!(blind.violations(&unknown).is_empty());
    assert_eq!(
        blind.availability(Facet::KnotMultiplicity),
        Availability::CannotReport
    );
    assert_eq!(
        reads.availability(Facet::KnotMultiplicity),
        Availability::Reported
    );

    // Mixing them up is a contract violation either way.
    assert_eq!(reads.violations(&unknown), vec![Facet::KnotMultiplicity]);
    assert_eq!(blind.violations(&plain), vec![Facet::KnotMultiplicity]);
}

#[test]
fn a_facet_with_no_row_cannot_be_reported_and_the_source_is_recorded() {
    let caps = ProviderCaps::new();
    for f in Facet::ALL {
        assert_eq!(caps.availability(f), Availability::CannotReport, "{f:?}");
        assert!(caps.get(f).is_none());
    }
    let caps = caps
        .with(
            Facet::Weights,
            Availability::Reported,
            FacetSource::DecoderDirect,
        )
        .with(
            Facet::KnotMultiplicity,
            Availability::Reported,
            FacetSource::DecoderAdapter,
        )
        .with(
            Facet::Weights,
            Availability::Reported,
            FacetSource::DecoderAdapter,
        );
    assert_eq!(
        caps.rows().len(),
        2,
        "with() replaces a row, it does not stack one"
    );
    assert_eq!(
        caps.get(Facet::Weights).unwrap().source,
        FacetSource::DecoderAdapter
    );
    assert_eq!(
        caps.get(Facet::KnotMultiplicity).unwrap().source,
        FacetSource::DecoderAdapter
    );
}

#[test]
fn a_provider_that_cannot_report_knots_yields_requires_extension_never_a_curve() {
    let caps = MockDwgProvider::full_caps().with(
        Facet::KnotMultiplicity,
        Availability::CannotReport,
        FacetSource::DecoderAdapter,
    );
    let s = spline(
        0x70,
        SourceFact::NotReportable,
        SourceFact::Absent,
        SourceFact::Present(false),
    );
    let p = MockDwgProvider::with_entities(vec![s], caps);

    let leaves = extract(&p);
    assert_eq!(leaves.len(), 1);
    assert_eq!(
        leaves[0].outcome,
        Outcome::RequiresExtension(vec![Facet::KnotMultiplicity])
    );

    // What the boundary hands over for it is the control points and nothing sampled:
    // there is no variant that carries a polyline built from a spline, and the knots
    // stay NotReportable instead of being invented.
    let got = p
        .entities(&EntityScope::Layout(LayoutId(0)))
        .unwrap()
        .collect_all()
        .unwrap();
    match &got[0].geometry {
        RawGeometry::Spline {
            control_points,
            knots,
            ..
        } => {
            assert_eq!(control_points.len(), 4);
            assert!(knots.is_not_reportable());
            assert!(knots.present().is_none());
        }
        other => panic!("expected the spline back, got {other:?}"),
    }
}

#[test]
fn a_provider_without_layer_caps_says_so_instead_of_returning_nothing() {
    let p = MockDwgProvider::with_entities(
        vec![],
        MockDwgProvider::full_caps().with(
            Facet::LayerState,
            Availability::CannotReport,
            FacetSource::DecoderAdapter,
        ),
    );
    assert_eq!(
        p.layers(),
        Err(ProviderError::CannotReport(Facet::LayerState))
    );
}

#[test]
fn unmapped_entities_are_counted_not_dropped() {
    let hatch = RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: "no mapping for HATCH".into(),
        },
        provenance: prov(0x80, "HATCH", "VC_ON"),
    };
    let p = MockDwgProvider::with_entities(
        vec![hatch, line(0x81, "VC_ON", [0.0; 3], [1.0; 3])],
        MockDwgProvider::full_caps(),
    );
    let leaves = extract(&p);
    assert_eq!(leaves.len(), 2);
    assert_eq!(leaves[0].outcome, Outcome::Unmapped);
    assert_eq!(leaves[0].class, "HATCH");
}

#[test]
fn a_source_transform_is_finite_checked_in_every_coefficient() {
    let names = ["m00", "m01", "m10", "m11", "tx", "ty"];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for i in 0..6 {
            let mut c = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            c[i] = bad;
            match SourceTransform::new(c) {
                Err(ProviderError::NonFinite { field }) => assert_eq!(field, names[i]),
                other => panic!("coefficient {i} = {bad}: {other:?}"),
            }
        }
    }
    // The first offender is the one named.
    let err = SourceTransform::new([1.0, f64::NAN, 0.0, 1.0, f64::INFINITY, 0.0]).unwrap_err();
    assert_eq!(err, ProviderError::NonFinite { field: "m01" });
}

#[test]
fn a_source_transform_keeps_f64_bits_and_the_documented_order() {
    let c = [1.3, 0.4, -0.7, 0.9, 2.0, -1.0];
    let t = SourceTransform::new(c).unwrap();
    for (a, b) in t.coefficients().iter().zip(c) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    // Extreme but finite values are not this layer's to refuse (NP-19 is N3.4's).
    let wide = SourceTransform::new([f64::MAX, 0.0, 0.0, f64::MIN_POSITIVE, -0.0, 5e-324]).unwrap();
    assert_eq!(wide.coefficients()[0], f64::MAX);
    assert_eq!(wide.coefficients()[4].to_bits(), (-0.0f64).to_bits());
    // A singular map is accepted here, and refused later.
    assert!(SourceTransform::new([1.0, 2.0, 2.0, 4.0, 0.0, 0.0]).is_ok());
    let id = SourceTransform::IDENTITY.coefficients();
    assert_eq!(id, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn an_instance_path_is_an_ordered_block_and_handle_chain() {
    let root = InstancePath::root();
    assert!(root.is_root());
    assert_eq!(root.depth(), 0);
    assert_eq!(root.to_string(), "/");

    let a = root.child("VC_OUTER", h(0x58));
    let b = a.child("VC_MID", h(0x55));
    let c = b.child("VC_INNER", h(0x51));

    // Children do not disturb their parents.
    assert!(root.is_root());
    assert_eq!(a.depth(), 1);
    assert_eq!(b.depth(), 2);
    assert_eq!(c.depth(), 3);

    let steps: Vec<(&str, u64)> = c
        .steps()
        .iter()
        .map(|s| (s.block_name.as_str(), s.insert_handle.get()))
        .collect();
    assert_eq!(
        steps,
        vec![("VC_OUTER", 0x58), ("VC_MID", 0x55), ("VC_INNER", 0x51)]
    );
    assert_eq!(c.to_string(), "VC_OUTER@58/VC_MID@55/VC_INNER@51");

    // Order is part of identity, and so is the handle.
    let swapped = root.child("VC_MID", h(0x55)).child("VC_OUTER", h(0x58));
    assert_ne!(b, swapped);
    assert_ne!(a, root.child("VC_OUTER", h(0x59)));
    assert_eq!(a, root.child("VC_OUTER", h(0x58)));
}

#[test]
fn an_insert_entity_extends_its_own_path_by_one_step() {
    let mut e = insert(0x55, "VC_ON", "VC_MID", [3.0, 0.0], 0.0, 1.0);
    e.provenance.block_path = InstancePath::root().child("VC_OUTER", h(0x58));
    let next = e.child_path().expect("an insert with a handle");
    assert_eq!(next.to_string(), "VC_OUTER@58/VC_MID@55");

    // A leaf has nowhere to descend, and an insert without a handle cannot be named.
    assert!(line(1, "0", [0.0; 3], [1.0; 3]).child_path().is_none());
    e.provenance.handle = None;
    assert!(e.child_path().is_none());
}

#[test]
fn a_block_scope_stamps_the_path_it_was_given_and_does_not_invent_one() {
    let p = MockDwgProvider::nested_drawing();
    let via = InstancePath::root()
        .child("VC_OUTER", h(0x58))
        .child("VC_MID", h(0x55));
    let scope = EntityScope::BlockDefinition {
        name: "VC_INNER".into(),
        reached_via: via.clone(),
    };
    let got = p.entities(&scope).unwrap().collect_all().unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].provenance.block_path, via);
    assert_eq!(got[0].provenance.layer, "VC_ON");
    assert_eq!(got[0].provenance.class_name, "LINE");
    assert_eq!(got[0].provenance.handle, Some(h(0x4C)));

    // A layout traversal starts from the empty path.
    let top = p
        .entities(&EntityScope::Layout(LayoutId(0)))
        .unwrap()
        .collect_all()
        .unwrap();
    assert!(top.iter().all(|e| e.provenance.block_path.is_root()));
}

#[test]
fn layer_and_layout_state_carry_the_flags_and_the_kind() {
    let p = MockDwgProvider::nested_drawing();
    let layers = p.layers().unwrap();
    assert_eq!(layers[1].name, "VC_ON");
    assert!(layers[1].locked && !layers[1].off && !layers[1].frozen);
    assert!(!layers[0].locked);

    let layouts = p.layouts().unwrap();
    assert_eq!(layouts[0].kind, LayoutKind::Model);
    assert_eq!(layouts[0].block_name, "*Model_Space");
    assert_eq!(layouts[1].kind, LayoutKind::Paper);
    assert_ne!(layouts[0].kind, layouts[1].kind);
    assert_eq!(layouts[1].id, LayoutId(1));
}
