//! Issue #1631 (C3.4 of iasbuilt/webapp #1434): xref resolution. The caller supplies the drawings
//! an xref block stands for with `PvecConversion::xref(block_name, source)`; the walk descends into
//! the xref INSERT with a second provider opened on that source, and its layers come in as
//! `xref|layer`. An xref nobody supplied stays `ExternalReference`, as today.
//!
//! The acadrust provider reads DWG only, so the DXF fixtures the issue names cannot be opened yet.
//! The drawings here are served from memory: `Mock::open` looks the drawing up by the bytes of the
//! source, so the host and each xref are separate providers, as they are in a real run. Layer
//! overrides (#1627) join later; the layer test uses the layer table's own off flag, which resolves
//! by the same formula. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::flatten::FlattenError;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, Limit,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceLimits,
    SourceTransform, SourceUnit, XrefAttachment,
};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput, PvecConvertError};
use pvec_facade::Curve;

#[derive(Clone, Default)]
struct Drawing {
    model: Vec<RawEntity>,
    layers: Vec<LayerState>,
    xrefs: Vec<XrefAttachment>,
}

thread_local! {
    /// The drawings of this test, by the bytes of the source that names them.
    static DRAWINGS: RefCell<HashMap<Vec<u8>, Drawing>> = RefCell::new(HashMap::new());
    /// Every source `Mock::open` was asked for, in order.
    static OPENED: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

struct Mock(Drawing);

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let CadSource::Bytes(b) = source else {
            panic!("the mock is only handed bytes: a stored path must never be opened");
        };
        OPENED.with(|o| o.borrow_mut().push(b.to_vec()));
        DRAWINGS
            .with(|d| d.borrow().get(b).cloned())
            .map(Mock)
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1631",
                detail: format!("no drawing for {:?}", String::from_utf8_lossy(b)),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1631"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: std::sync::OnceLock<ProviderCaps> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            })
        })
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
        Ok(self.0.layers.clone())
    }
    fn xref_blocks(&self) -> Result<Vec<XrefAttachment>, ProviderError> {
        Ok(self.0.xrefs.clone())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        match scope {
            EntityScope::Layout(_) => Ok(Box::new(Cursor(self.0.model.clone().into_iter()))),
            other => Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        }
    }
}

fn layer(name: &str, off: bool) -> LayerState {
    LayerState {
        name: name.into(),
        handle: None,
        off,
        frozen: false,
        locked: false,
        plottable: true,
    }
}

fn line(handle: u64, layer: &str, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

/// An INSERT of `block` scaled by `scale` and moved to `at`.
fn insert(handle: u64, block: &str, scale: f64, at: [f64; 2]) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::new([scale, 0.0, 0.0, scale, at[0], at[1]]).unwrap(),
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
            elevation: 0.0,
            z_scale: 1.0,
        }),
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "INSERT".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

fn attach(block: &str) -> XrefAttachment {
    XrefAttachment {
        block_name: block.into(),
        // Not on any disk: the conversion must never look.
        path: format!("/nonexistent-1631/{block}.dwg"),
        overlay: false,
        unloaded: false,
    }
}

fn register(bytes: &[u8], d: Drawing) {
    DRAWINGS.with(|m| m.borrow_mut().insert(bytes.to_vec(), d));
}

/// The host: one line of its own, and an INSERT of the xref block `WING` at (10, 20) scaled 0.001
/// (the xref is in millimetres, the host in metres: the unit mismatch the issue asks for first).
fn host(block_spelling: &str, wing_layer_off: bool) -> Drawing {
    Drawing {
        model: vec![
            line(1, "0", [1.0, 1.0], [2.0, 1.0]),
            insert(2, block_spelling, 0.001, [10.0, 20.0]),
        ],
        layers: vec![layer("0", false), layer("WING|A-WALL", wing_layer_off)],
        xrefs: vec![attach("WING")],
    }
}

/// The xref drawing: two lines on `A-WALL`, 1000 mm and 500 mm long.
fn wing() -> Drawing {
    Drawing {
        model: vec![
            line(100, "A-WALL", [0.0, 0.0], [1000.0, 0.0]),
            line(101, "A-WALL", [0.0, 0.0], [0.0, 500.0]),
        ],
        layers: vec![layer("0", false), layer("A-WALL", false)],
        xrefs: vec![],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1631-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn run(
    tag: &str,
    configure: impl for<'a> FnOnce(PvecConversion<'a>) -> PvecConversion<'a>,
) -> (Result<PvecConversionOutput, PvecConvertError>, PathBuf) {
    OPENED.with(|o| o.borrow_mut().clear());
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    let c = PvecConversion::new(CadSource::Bytes(b"host"), &dest).unit(SourceUnit::Meters);
    (configure(c).convert_with::<Mock>(), dest)
}

fn segments(dest: &std::path::Path) -> Vec<([f64; 2], [f64; 2], f64)> {
    open_published(dest)
        .unwrap()
        .entities()
        .unwrap()
        .into_iter()
        .map(|(_, c)| {
            let Curve::Trimmed(t) = c else {
                panic!("a trimmed line")
            };
            let Curve::Line(l) = t.basis() else {
                panic!("a trimmed line")
            };
            let (o, d) = (l.origin(), l.direction());
            ([o.x, o.y], [d.x, d.y], t.trim().span)
        })
        .collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// What every test relies on, asserted so a red result is about xrefs and not the setup: the host
/// converts with no xref supplied, draws its own line, and leaves the WING INSERT unresolved.
fn assert_baseline(tag: &str) -> ConversionReport {
    register(b"host", host("WING", false));
    let (out, dest) = run(tag, |c| c);
    let out = out.expect("the host converts without an xref");
    assert_eq!(
        out.written, 1,
        "only the host's own line draws: {:#?}",
        out.report.entities
    );
    assert_eq!(segments(&dest).len(), 1);
    assert_eq!(
        out.report.xrefs.unresolved.len(),
        1,
        "{:#?}",
        out.report.xrefs
    );
    assert!(out.report.xrefs.resolved.is_empty());
    out.report
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_xref_content_draws_when_supplied() {
    assert_baseline("draws-base");
    register(b"host", host("WING", false));
    register(b"wing-file", wing());
    let (out, _) = run("draws", |c| c.xref("WING", CadSource::Bytes(b"wing-file")));
    let out = out.unwrap();
    assert_eq!(
        out.written, 3,
        "the host line and both xref lines: {:#?}",
        out.report.entities
    );
    assert_eq!(out.report.xrefs.resolved, vec!["WING".to_string()]);
    assert!(
        out.report.xrefs.unresolved.is_empty(),
        "{:#?}",
        out.report.xrefs
    );
    assert!(
        out.report
            .entities
            .iter()
            .any(|r| r.source_handle == Some(100) && r.semantic_type.is_some())
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_the_host_insert_transform_composes_with_the_xref_unit_mismatch() {
    assert_baseline("unit-base");
    register(b"host", host("WING", false));
    register(b"wing-file", wing());
    let (out, dest) = run("unit", |c| c.xref("WING", CadSource::Bytes(b"wing-file")));
    out.unwrap();
    let segs = segments(&dest);
    assert_eq!(segs.len(), 3, "{segs:?}");
    // 1000 mm at scale 0.001 from (10, 20): one metre east. 500 mm north: half a metre.
    assert!(
        segs.iter().any(|(o, d, len)| near(o[0], 10.0)
            && near(o[1], 20.0)
            && near(d[0], 1.0)
            && near(*len, 1.0)),
        "{segs:?}"
    );
    assert!(
        segs.iter().any(|(o, d, len)| near(o[0], 10.0)
            && near(o[1], 20.0)
            && near(d[1], 1.0)
            && near(*len, 0.5)),
        "{segs:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_the_block_name_matches_ignoring_ascii_case() {
    assert_baseline("case-base");
    // The host spells the INSERT `Wing`, the record says `WING`, the caller says `wing`.
    register(b"host", host("Wing", false));
    register(b"wing-file", wing());
    let (out, _) = run("case", |c| c.xref("wing", CadSource::Bytes(b"wing-file")));
    let out = out.unwrap();
    assert_eq!(out.written, 3, "{:#?}", out.report.entities);
    assert_eq!(out.report.xrefs.resolved, vec!["WING".to_string()]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_xref_layer_override_applies() {
    assert_baseline("layer-base");
    register(b"wing-file", wing());
    // On: the xref's lines draw, and the layer is named `WING|A-WALL`.
    register(b"host", host("WING", false));
    let (on, _) = run("layer-on", |c| {
        c.xref("WING", CadSource::Bytes(b"wing-file"))
    });
    let on = on.unwrap();
    assert_eq!(on.written, 3);
    assert!(
        on.report.view.excluded_by_layer.is_empty(),
        "{:#?}",
        on.report.view
    );
    // Off in the host's layer table: the same content is held back by the same rule as any layer.
    register(b"host", host("WING", true));
    let (off, _) = run("layer-off", |c| {
        c.xref("WING", CadSource::Bytes(b"wing-file"))
    });
    let off = off.unwrap();
    assert_eq!(off.written, 1, "{:#?}", off.report.entities);
    assert_eq!(
        off.report.view.excluded_by_layer.get("WING|A-WALL"),
        Some(&2)
    );
    assert_eq!(
        off.report.view.excluded_by_field.get("layers.off"),
        Some(&2)
    );
    let hidden: Vec<_> = off
        .report
        .entities
        .iter()
        .filter(|r| matches!(r.source_handle, Some(100 | 101)))
        .collect();
    assert_eq!(hidden.len(), 2, "{:#?}", off.report.entities);
    assert!(hidden.iter().all(|r| r.fidelity == "ignored-by-policy"));
    assert!(
        hidden
            .iter()
            .all(|r| r.policy_field.as_deref() == Some("layers.off"))
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_xref_missing_stays_skipped() {
    let report = assert_baseline("missing");
    assert_eq!(report.xrefs.unresolved[0].name, "WING");
    // Another xref is supplied, WING is not: WING stays skipped and is still counted.
    register(b"other-file", wing());
    let (out, _) = run("missing-other", |c| {
        c.xref("ELSEWHERE", CadSource::Bytes(b"other-file"))
    });
    let out = out.unwrap();
    assert_eq!(out.written, 1);
    assert_eq!(out.report.xrefs.unresolved.len(), 1);
    assert_eq!(out.report.xrefs.unresolved[0].name, "WING");
    assert!(out.report.xrefs.resolved.is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_a_supplied_xref_the_host_does_not_have_is_reported() {
    assert_baseline("unknown-base");
    register(b"wing-file", wing());
    let (out, _) = run("unknown", |c| {
        c.xref("NOPE", CadSource::Bytes(b"wing-file"))
    });
    let out = out.unwrap();
    assert!(
        out.report
            .diagnostics
            .iter()
            .any(|d| d.message.contains("NOPE")),
        "a diagnostic names the supplied xref nothing attaches: {:#?}",
        out.report.diagnostics
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_the_stored_path_is_never_opened() {
    assert_baseline("path-base");
    register(b"host", host("WING", false));
    register(b"wing-file", wing());
    assert!(!std::path::Path::new(&attach("WING").path).exists());
    let (out, _) = run("path", |c| c.xref("WING", CadSource::Bytes(b"wing-file")));
    let out = out.unwrap();
    assert_eq!(
        out.written, 3,
        "the supplied source is used, not the stored path"
    );
    let opened = OPENED.with(|o| o.borrow().clone());
    assert_eq!(
        opened,
        vec![b"host".to_vec(), b"wing-file".to_vec()],
        "opened in this order only"
    );
}

/// WING holds an INSERT of `DEEP`, itself an xref of WING, one level further down.
fn nested() {
    register(b"host", host("WING", false));
    let mut w = wing();
    w.model.push(insert(102, "DEEP", 1.0, [0.0, 0.0]));
    w.xrefs = vec![attach("DEEP")];
    register(b"wing-file", w);
    register(
        b"deep-file",
        Drawing {
            model: vec![line(200, "A-DEEP", [0.0, 0.0], [10.0, 0.0])],
            layers: vec![layer("A-DEEP", false)],
            xrefs: vec![],
        },
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_xref_nested_depth_capped() {
    assert_baseline("nested-base");
    nested();
    fn both<'a>(c: PvecConversion<'a>) -> PvecConversion<'a> {
        c.xref("WING", CadSource::Bytes(b"wing-file"))
            .xref("DEEP", CadSource::Bytes(b"deep-file"))
    }
    // Room for both levels: the nested content draws too.
    let (ok, _) = run("nested-ok", both);
    let ok = ok.unwrap();
    assert_eq!(ok.written, 4, "{:#?}", ok.report.entities);
    // A depth of one admits WING and refuses DEEP, the way a block nested too deep is refused.
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 1;
    let (capped, _) = run("nested-capped", |c| both(c).limits(limits));
    match capped {
        Err(PvecConvertError::Flatten(FlattenError::Limit {
            limit: Limit::BlockDepth,
            ..
        })) => {}
        other => panic!("the nesting is refused at the depth cap: {other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1631_a_fan_out_is_refused_by_the_shared_entity_guard() {
    assert_baseline("fan-base");
    register(b"host", host("WING", false));
    let mut big = wing();
    big.model = (0..50u64)
        .map(|k| line(1000 + k, "A-WALL", [0.0, k as f64], [1.0, k as f64]))
        .collect();
    register(b"wing-file", big);
    // The host has 2 entities; the cap leaves too little for the xref's 50.
    let mut limits = SourceLimits::default();
    limits.max_entities = 10;
    let (out, _) = run("fan", |c| {
        c.xref("WING", CadSource::Bytes(b"wing-file"))
            .limits(limits)
    });
    match out {
        Err(PvecConvertError::Flatten(FlattenError::Limit {
            limit: Limit::Entities,
            ..
        })) => {}
        other => panic!("the xref's entities count against the host's guard: {other:?}"),
    }
}
