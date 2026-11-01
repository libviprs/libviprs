//! Issue #1416 (slice of #1048, N7.5): structured conversion diagnostics. The closed code table is
//! read from `docs/pvec-diagnostics.md` and checked against what the fixtures emit in both
//! directions, diagnostics round-trip through JSON with stable field names, a failure carries its
//! entity's provenance in the error value, and a clean drawing makes none and allocates nothing for
//! them. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::observe::NoopObserver;
use libviprs::pvec::diagnostics::{Diagnostic, DiagnosticCode, Severity};
use libviprs::pvec::report::ReportError;
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceLimits, SourceTransform, SourceUnit,
    SpaceSelection, ViewProfile,
};
use libviprs::pvec_convert::{CapabilityPolicy, PvecConversion, PvecConvertError};
use libviprs::{CancelToken, CollectingObserver, EngineEvent};
use pvec_facade::CapabilityId;

// ---- a counting allocator, per thread, so the other tests in this binary do not show ----

struct Counting;

thread_local! {
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.with(|b| b.set(b.get() + layout.size() as u64));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocated<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = BYTES.with(Cell::get);
    let out = f();
    (out, BYTES.with(Cell::get) - before)
}

// ---- a mock provider whose drawing is picked by the bytes it is opened with ----

struct Mock {
    scene: &'static str,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn prov(handle: Option<u64>, class: &str, layer: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: handle.map(ItemHandle::new),
        class_name: class.into(),
        block_path: InstancePath::root(),
        layer: layer.into(),
    }
}

fn circle(handle: u64, layer: &str) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Circle {
            center: [5.0 * handle as f64, 5.0, 0.0],
            radius: 2.0,
            normal: [0.0, 0.0, 1.0],
        },
        provenance: prov(Some(handle), "CIRCLE", layer),
    }
}

fn line(handle: u64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [0.0, 0.0, 0.0],
            end: [10.0, 0.0, 0.0],
        },
        provenance: prov(Some(handle), "LINE", "0"),
    }
}

fn insert_without_handle() -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Insert(BlockRef {
            block_name: "B".into(),
            transform: SourceTransform::new([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid {
                rows: 1,
                columns: 1,
                row_spacing: 0.0,
                column_spacing: 0.0,
            },
        }),
        provenance: prov(None, "INSERT", "0"),
    }
}

impl DwgProvider for Mock {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK")))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let CadSource::Bytes(b) = source else {
            return Err(ProviderError::Unrecognised {
                provider: "mock",
                detail: "bytes only".into(),
            });
        };
        let scene = match b {
            b"MOCK-unmapped" => "unmapped",
            b"MOCK-hidden" => "hidden",
            b"MOCK-insert" => "insert",
            _ => "plain",
        };
        let caps = Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
            c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
        });
        Ok(Mock { scene, caps })
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
        let layer = |name: &str, off| LayerState {
            name: name.into(),
            handle: None,
            off,
            frozen: false,
            locked: false,
        };
        Ok(vec![layer("0", false), layer("HIDE", true)])
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let items = match (scope, self.scene) {
            (EntityScope::Layout(_), "unmapped") => vec![
                circle(1, "0"),
                RawEntity {
                    style: EntityStyle::absent(),
                    geometry: RawGeometry::Unmapped {
                        reason: "hatch".into(),
                    },
                    provenance: prov(Some(9), "HATCH", "0"),
                },
            ],
            (EntityScope::Layout(_), "hidden") => vec![circle(1, "0"), circle(2, "HIDE")],
            (EntityScope::Layout(_), "insert") => vec![circle(1, "0"), insert_without_handle()],
            (EntityScope::BlockDefinition { name, reached_via }, "insert") if name == "B" => {
                let mut c = circle(3, "0");
                c.provenance.block_path = reached_via.clone();
                vec![c]
            }
            (EntityScope::Layout(_), _) => vec![circle(1, "0"), circle(2, "0"), line(5)],
            _ => return Err(ProviderError::NoSuchScope("no such block".into())),
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }
}

fn scratch(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("libviprs-1416-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn mock<'a>(scene: &'a [u8], dest: &'a Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Bytes(scene), dest).unit(SourceUnit::Meters)
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider")
        .join(name)
}

/// Every fixture: the name `docs/pvec-diagnostics.md` gives it, and the diagnostics it makes.
fn fixtures() -> Vec<(&'static str, Vec<Diagnostic>)> {
    let dir = scratch("fixtures");
    let dest = dir.join("o.pvec");
    let without_circle: Vec<CapabilityId> = ConversionProfileSet::without_circle();
    let mut out: Vec<(&'static str, Vec<Diagnostic>)> = Vec::new();

    let report_of = |scene: &[u8], conv: PvecConversion<'_>| {
        let _ = scene;
        conv.convert_with::<Mock>().unwrap().report.diagnostics
    };
    out.push((
        "an_entity_outside_the_allowed_set",
        report_of(
            b"MOCK-plain",
            mock(b"MOCK-plain", &dest)
                .allowed_capabilities(without_circle.clone())
                .capability_policy(CapabilityPolicy::RejectEntity),
        ),
    ));
    out.push((
        "an_entity_the_provider_does_not_map",
        report_of(b"", mock(b"MOCK-unmapped", &dest)),
    ));
    out.push((
        "a_leaf_on_a_hidden_layer",
        report_of(b"", mock(b"MOCK-hidden", &dest)),
    ));
    out.push((
        "an_insert_without_a_handle",
        report_of(b"", mock(b"MOCK-insert", &dest)),
    ));

    let fail = |e: PvecConvertError| vec![e.diagnostic()];
    let missing = dir.join("missing.dwg");
    out.push((
        "a_missing_source",
        fail(
            PvecConversion::new(CadSource::Path(&missing), &dest)
                .convert()
                .unwrap_err(),
        ),
    ));
    let mut paper = ViewProfile::default_profile();
    paper.space = SpaceSelection::PaperLayout("Layout1".into());
    out.push((
        "a_paper_layout_in_v1",
        fail(
            mock(b"MOCK-plain", &dest)
                .view_profile(paper)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    let nested = fixture("vc_nested.dwg");
    let mut shallow = SourceLimits::default();
    shallow.max_block_depth = 1;
    out.push((
        "a_block_depth_limit",
        fail(
            PvecConversion::new(CadSource::Path(&nested), &dest)
                .unit(SourceUnit::Meters)
                .limits(shallow)
                .convert()
                .unwrap_err(),
        ),
    ));
    out.push((
        "a_generic_provider_without_a_unit",
        fail(
            PvecConversion::new(CadSource::Bytes(b"MOCK-plain"), &dest)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    out.push((
        "reject_input",
        fail(
            mock(b"MOCK-plain", &dest)
                .allowed_capabilities(without_circle)
                .capability_policy(CapabilityPolicy::RejectInput)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    out.push((
        "a_unit_pvec_cannot_declare",
        fail(
            PvecConversion::new(CadSource::Bytes(b"MOCK-plain"), &dest)
                .unit(SourceUnit::Unitless)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    out.push((
        "a_report_that_does_not_add_up",
        fail(PvecConvertError::Report(ReportError::DuplicateId(1))),
    ));
    // The destination is a directory, so the final rename cannot succeed.
    let as_dir = dir.join("is-a-dir");
    fs::create_dir_all(&as_dir).unwrap();
    out.push((
        "a_destination_that_is_a_directory",
        fail(
            mock(b"MOCK-plain", &as_dir)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    out.push((
        "a_control_point_that_is_not_finite",
        fail(
            mock(b"MOCK-plain", &dest)
                .control_points([[f64::INFINITY, 0.0]])
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    let token = CancelToken::new();
    token.cancel();
    out.push((
        "a_precancelled_run",
        fail(
            mock(b"MOCK-plain", &dest)
                .cancel(token)
                .convert_with::<Mock>()
                .unwrap_err(),
        ),
    ));
    let _ = fs::remove_dir_all(&dir);
    out
}

struct ConversionProfileSet;

impl ConversionProfileSet {
    /// Every capability the default profile allows but the circle's (6).
    fn without_circle() -> Vec<CapabilityId> {
        libviprs::pvec_convert::ConversionProfile::Default
            .capabilities()
            .into_iter()
            .filter(|c| c.get() != 6)
            .collect()
    }
}

/// The table in the document: `(code, fixture)` per row.
fn documented() -> Vec<(String, String)> {
    let doc = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/pvec-diagnostics.md"
    ))
    .unwrap();
    let table = doc.split("## The codes").nth(1).unwrap();
    let table = table.split("## Fields").next().unwrap();
    table
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            let cells: Vec<&str> = l.split('|').map(str::trim).collect();
            let code = cells[1].trim_matches('`').to_owned();
            let fixture = cells[cells.len() - 2].trim_matches('`').to_owned();
            (code, fixture)
        })
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1416_every_code_in_the_table_has_a_fixture_and_every_emitted_code_is_in_the_table() {
    let ran = fixtures();
    let emitted: BTreeSet<&str> = ran
        .iter()
        .flat_map(|(_, ds)| ds.iter().map(|d| d.code.name()))
        .collect();
    let all: BTreeSet<&str> = DiagnosticCode::ALL.iter().map(|c| c.name()).collect();
    let table = documented();
    let in_doc: BTreeSet<&str> = table.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(
        emitted, all,
        "a code with no fixture, or an emitted code outside ALL"
    );
    assert_eq!(in_doc, all, "the document and DiagnosticCode::ALL disagree");
    // The fixture the document names for a code is one that emits it.
    for (code, fixture) in &table {
        let (_, ds) = ran
            .iter()
            .find(|(n, _)| n == fixture)
            .unwrap_or_else(|| panic!("the document names the unknown fixture {fixture}"));
        assert!(
            ds.iter().any(|d| d.code.name() == code),
            "{fixture} does not emit {code}"
        );
    }
}

#[test]
fn issue_1416_severity_is_a_property_of_the_code_and_not_of_the_wording() {
    for code in DiagnosticCode::ALL {
        let a = Diagnostic::new(code, "id", "one wording");
        let b = Diagnostic::new(code, "id", "a completely different wording");
        assert_eq!(
            (a.severity, a.stage, &a.message_id),
            (b.severity, b.stage, &b.message_id)
        );
        assert_eq!(a.severity, code.severity());
    }
    let sev = |c: DiagnosticCode| format!("{:?}", c.severity());
    assert_eq!(sev(DiagnosticCode::EntityIgnoredByPolicy), "Info");
    assert_eq!(sev(DiagnosticCode::EntityUnsupported), "Warning");
    assert_eq!(sev(DiagnosticCode::CapabilityRejected), "Error");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1416_diagnostics_round_trip_through_json_with_stable_field_names() {
    let mut all = fixtures();
    let (_, ds) = all
        .iter_mut()
        .find(|(n, _)| *n == "an_entity_outside_the_allowed_set")
        .unwrap();
    let d = ds.remove(0);
    let json = serde_json::to_string(&d).unwrap();
    assert_eq!(serde_json::from_str::<Diagnostic>(&json).unwrap(), d);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "block_path",
            "class",
            "code",
            "handle",
            "layer",
            "layout",
            "message",
            "message_id",
            "severity",
            "source",
            "stage"
        ]
    );
    assert_eq!(v["code"], "ENTITY_REQUIRES_CAPABILITY");
    assert_eq!(v["message_id"], "capability-not-allowed");
    assert_eq!(v["class"], "CIRCLE");
    assert_eq!(v["layout"], "Model");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1416_a_failure_carries_the_entitys_handle_layer_and_block_path_in_the_error() {
    let dir = scratch("context");
    let dest = dir.join("o.pvec");
    let err = mock(b"MOCK-plain", &dest)
        .allowed_capabilities(ConversionProfileSet::without_circle())
        .capability_policy(CapabilityPolicy::RejectInput)
        .convert_with::<Mock>()
        .unwrap_err();
    let PvecConvertError::CapabilityNotAllowed {
        handle,
        layer,
        block_path,
        source_class,
        ..
    } = &err
    else {
        panic!("{err:?}")
    };
    assert_eq!(
        (
            *handle,
            layer.as_str(),
            block_path.as_str(),
            source_class.as_str()
        ),
        (Some(1), "0", "/", "CIRCLE")
    );
    let d = err.diagnostic();
    assert_eq!(d.code, DiagnosticCode::CapabilityRejected);
    assert_eq!((d.handle, d.layer.as_deref()), (Some(1), Some("0")));

    // A block failure names the INSERT chain it happened in.
    let nested = fixture("vc_nested.dwg");
    let mut shallow = SourceLimits::default();
    shallow.max_block_depth = 1;
    let err = PvecConversion::new(CadSource::Path(&nested), &dest)
        .unit(SourceUnit::Meters)
        .limits(shallow)
        .convert()
        .unwrap_err();
    let d = err.diagnostic();
    assert_eq!(d.code, DiagnosticCode::BlocksFailed);
    assert!(
        d.block_path.as_deref().is_some_and(|p| p.contains('@')),
        "{d:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1416_the_observer_hears_each_diagnostic_and_the_failure() {
    let dir = scratch("observer");
    let dest = dir.join("o.pvec");
    let seen = Arc::new(CollectingObserver::new());
    let out = mock(b"MOCK-unmapped", &dest)
        .observer(seen.clone())
        .convert_with::<Mock>()
        .unwrap();
    let events: Vec<EngineEvent> = seen
        .events()
        .into_iter()
        .filter(|e| matches!(e, EngineEvent::ConversionDiagnostic { .. }))
        .collect();
    assert_eq!(events.len(), out.report.diagnostics.len());
    assert_eq!(events.len(), 1);
    let EngineEvent::ConversionDiagnostic {
        code,
        severity,
        message_id,
        handle,
        ..
    } = &events[0]
    else {
        unreachable!()
    };
    assert_eq!(
        (
            code.as_str(),
            severity.as_str(),
            message_id.as_str(),
            *handle
        ),
        (
            "ENTITY_UNSUPPORTED",
            "warning",
            "unmapped-source-class",
            Some(9)
        )
    );
    assert_eq!(events[0].name(), "conversion_diagnostic");

    let failing = Arc::new(CollectingObserver::new());
    let token = CancelToken::new();
    token.cancel();
    mock(b"MOCK-plain", &dest)
        .observer(failing.clone())
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(
        failing.events().as_slice(),
        [EngineEvent::ConversionDiagnostic { code, .. }] if code == "CANCELLED"
    ));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1416_a_clean_drawing_makes_no_diagnostic_and_allocates_nothing_for_them() {
    let dir = scratch("alloc");
    let dest = dir.join("o.pvec");
    let (bare, bytes_bare) = allocated(|| mock(b"MOCK-plain", &dest).convert_with::<Mock>());
    let bare = bare.unwrap();
    assert!(bare.report.diagnostics.is_empty());
    let (observed, bytes_observed) = allocated(|| {
        mock(b"MOCK-plain", &dest)
            .observer(Arc::new(NoopObserver))
            .convert_with::<Mock>()
    });
    assert!(observed.unwrap().report.diagnostics.is_empty());
    // The observer is an `Arc` clone and a pointer, and a supported entity makes no diagnostic, so
    // the two runs allocate the same number of bytes but for the `Arc` itself.
    let arc = std::mem::size_of::<usize>() as u64 * 4;
    assert!(
        bytes_observed <= bytes_bare + arc,
        "bare {bytes_bare}, with an observer {bytes_observed}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn issue_1416_the_diagnostic_enums_are_non_exhaustive() {
    #[deny(unreachable_patterns)]
    #[allow(clippy::match_like_matches_macro)]
    fn code(c: DiagnosticCode) -> bool {
        match c {
            DiagnosticCode::Cancelled => true,
            _ => false,
        }
    }
    #[deny(unreachable_patterns)]
    #[allow(clippy::match_like_matches_macro)]
    fn sev(s: Severity) -> bool {
        match s {
            Severity::Info => true,
            _ => false,
        }
    }
    assert!(code(DiagnosticCode::Cancelled) && sev(Severity::Info));
}
