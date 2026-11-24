//! Issue #1674: conversion errors have a stable code, sheets report progress, and nothing runs past a
//! deadline. The drawings are hand-built (a mock provider), so every outcome is pinned exactly.
//!
//! - codes: `code()` on the conversion, layout set, sheet skip, PC3 and limit errors, frozen by a table.
//! - progress: `ConversionSheetStarted` and `ConversionSheetFinished` from `PvecLayoutSet` and `PdfPlot`,
//!   and a cancel that stops the next sheet from starting.
//! - deadline: `deadline(Duration)` on `PvecConversion`, `PvecLayoutSet` and `PdfPlot`, its own error,
//!   distinct from a cancel, with nothing published.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use libviprs::CancelToken;
use libviprs::cad::pc3::Pc3Error;
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::observe::{CollectingObserver, EngineEvent, EngineObserver};
use libviprs::pvec::flatten::FlattenError;
use libviprs::pvec::publish::PublishError;
use libviprs::pvec::view_profile::ViewProfileError;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, Limit, LimitError,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact, SourceLimits,
    SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{
    LayoutSetError, PdfPlot, PvecConversion, PvecConvertError, PvecLayoutSet, SheetSkip,
};

// ---- a hand-built provider ------------------------------------------------------------

/// One paper layout: its name, how many lines it draws and how long reading each one takes.
#[derive(Clone)]
struct Sheet {
    name: &'static str,
    lines: u64,
    delay_ms: u64,
}

#[derive(Clone)]
struct Mock {
    sheets: Vec<Sheet>,
}

thread_local! {
    static NEXT: RefCell<Option<Mock>> = const { RefCell::new(None) };
}

/// Hands out its lines, sleeping before each, the way a slow reader does.
struct SlowCursor {
    items: std::vec::IntoIter<RawEntity>,
    delay: Duration,
}

impl EntityCursor for SlowCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        let next = self.items.next();
        if next.is_some() && !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(next)
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1674",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1674"
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
        let mut v = vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }];
        for (i, s) in self.sheets.iter().enumerate() {
            v.push(LayoutState {
                id: LayoutId(i as u32 + 1),
                name: s.name.into(),
                kind: LayoutKind::Paper,
                block_name: format!("*Paper_Space{i}"),
            });
        }
        Ok(v)
    }
    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        match self.sheets.get((layout.0 as usize).wrapping_sub(1)) {
            Some(_) => Ok(SourceFact::Present(plot())),
            None => Err(ProviderError::NoSuchScope(format!("layout {}", layout.0))),
        }
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            ..LayerState::default()
        }])
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let (items, delay) = match scope {
            EntityScope::Layout(LayoutId(0)) => (Vec::new(), 0),
            EntityScope::Layout(LayoutId(n)) => {
                let s = self
                    .sheets
                    .get(*n as usize - 1)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {n}")))?;
                let lines = (0..s.lines).map(|i| line(u64::from(*n) * 1000 + i + 1, i as f64));
                (lines.collect(), s.delay_ms)
            }
            other => return Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        };
        Ok(Box::new(SlowCursor {
            items: items.into_iter(),
            delay: Duration::from_millis(delay),
        }))
    }
}

fn line(handle: u64, y: f64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [10.0, y, 0.0],
            end: [100.0, y + 1.0, 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

/// A4 landscape in millimetres, nothing else off its default.
fn plot() -> LayoutPlot {
    LayoutPlot {
        paper_size: "TEST_SHEET".into(),
        paper_width: 297.0,
        paper_height: 210.0,
        paper_units: 1,
        plot_type: 5,
        window: [0.0; 4],
        scale_numerator: 1.0,
        scale_denominator: 1.0,
        rotation: 0,
        margins: [0.0; 4],
        origin: [0.0; 2],
        style_sheet: String::new(),
        print_lineweights: true,
        scale_lineweights: false,
        plot_plot_styles: true,
        plot_viewport_borders: true,
        draw_viewports_first: false,
    }
}

fn sheet(name: &'static str, lines: u64) -> Sheet {
    Sheet {
        name,
        lines,
        delay_ms: 0,
    }
}

fn slow(name: &'static str, lines: u64, delay_ms: u64) -> Sheet {
    Sheet {
        name,
        lines,
        delay_ms,
    }
}

/// Hands `sheets` to the next `open` on this thread.
fn drawing(sheets: Vec<Sheet>) {
    NEXT.with(|n| *n.borrow_mut() = Some(Mock { sheets }));
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1674-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn bytes() -> CadSource<'static> {
    CadSource::Bytes(b"mock")
}

/// The sheet events an observer saw, one string each.
fn progress(o: &CollectingObserver) -> Vec<String> {
    o.events()
        .into_iter()
        .filter_map(|e| match e {
            EngineEvent::ConversionSheetStarted {
                index,
                total,
                layout,
            } => Some(format!("start {index}/{total} {layout}")),
            EngineEvent::ConversionSheetFinished {
                index,
                total,
                layout,
                status,
                code,
            } => Some(format!(
                "done {index}/{total} {layout} {status} {}",
                code.as_deref().unwrap_or("-")
            )),
            _ => None,
        })
        .collect()
}

/// Cancels `token` when sheet 1 finishes, and records what it sees.
struct CancelAfterFirst {
    token: CancelToken,
    seen: CollectingObserver,
}

impl EngineObserver for CancelAfterFirst {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::ConversionSheetFinished { index: 1, .. }) {
            self.token.cancel();
        }
        self.seen.on_event(event);
    }
}

fn io_err() -> io::Error {
    io::Error::other("disk")
}

fn limit_error(limit: Limit) -> LimitError {
    LimitError::Exceeded {
        limit,
        observed: 2,
        ceiling: 1,
    }
}

// ---- codes ------------------------------------------------------------------------------

/// Each error's code against the code it must keep: `(got, want)`.
fn code_table() -> Vec<(&'static str, &'static str)> {
    let not_found = || ViewProfileError::LayoutNotFound {
        requested: "X".into(),
        available: vec![],
    };
    let scope = || ProviderError::NoSuchScope("x".into());
    let mut t = vec![
        // PvecConvertError, and the lower errors that carry a more precise code than their variant.
        (PvecConvertError::UnitRequired.code(), "UNIT_REQUIRED"),
        (PvecConvertError::Cancelled.code(), "CANCELLED"),
        (PvecConvertError::Deadline.code(), "DEADLINE"),
        (PvecConvertError::Open(scope()).code(), "OPEN_FAILED"),
        (
            PvecConvertError::View(not_found()).code(),
            "LAYOUT_NOT_FOUND",
        ),
        (
            PvecConvertError::View(ViewProfileError::NoModelSpace { available: vec![] }).code(),
            "NO_MODEL_SPACE",
        ),
        (PvecConvertError::View(scope().into()).code(), "VIEW_FAILED"),
        (
            PvecConvertError::View(limit_error(Limit::Entities).into()).code(),
            "LIMIT_ENTITIES",
        ),
        (
            PvecConvertError::Flatten(FlattenError::Limit {
                limit: Limit::ExpandedInstances,
                observed: 2,
                ceiling: 1,
                block_path: InstancePath::root(),
            })
            .code(),
            "LIMIT_EXPANDED_INSTANCES",
        ),
        (
            PvecConvertError::Flatten(FlattenError::BlockRecursionLimit {
                depth: 3,
                block_path: InstancePath::root(),
            })
            .code(),
            "BLOCK_CYCLE",
        ),
        (
            PvecConvertError::Publish(PublishError::Io(io_err())).code(),
            "PUBLISH_FAILED",
        ),
        // LayoutSetError and SheetSkip.
        (
            LayoutSetError::NotADirectory(PathBuf::from("x")).code(),
            "NOT_A_DIRECTORY",
        ),
        (LayoutSetError::StemRequired.code(), "STEM_REQUIRED"),
        (LayoutSetError::Open(scope()).code(), "OPEN_FAILED"),
        (LayoutSetError::Layouts(scope()).code(), "LAYOUTS_FAILED"),
        (
            LayoutSetError::LayoutNotFound {
                requested: "X".into(),
                available: vec![],
            }
            .code(),
            "LAYOUT_NOT_FOUND",
        ),
        (LayoutSetError::Index(io_err()).code(), "INDEX_FAILED"),
        (LayoutSetError::Cancelled.code(), "CANCELLED"),
        (LayoutSetError::Deadline.code(), "DEADLINE"),
        (SheetSkip::Empty.code(), "EMPTY"),
        (SheetSkip::EmptyAfterView.code(), "EMPTY_AFTER_VIEW"),
        (
            SheetSkip::Failed(PvecConvertError::Deadline).code(),
            "DEADLINE",
        ),
        // Pc3Error.
        (Pc3Error::TooLarge(1).code(), "PC3_TOO_LARGE"),
        (Pc3Error::TooDeep.code(), "PC3_TOO_DEEP"),
        (Pc3Error::Unrecognised.code(), "PC3_UNRECOGNISED"),
        (Pc3Error::Syntax(0, "x").code(), "PC3_SYNTAX"),
        (
            Pc3Error::BinaryNotEditable("x").code(),
            "PC3_BINARY_NOT_EDITABLE",
        ),
        (Pc3Error::Missing("x").code(), "PC3_MISSING"),
        (Pc3Error::NotFinite("x").code(), "PC3_NOT_FINITE"),
        // LimitError.
        (
            LimitError::BlockCycle {
                block: "B".into(),
                chain: vec!["B".into(), "B".into()],
            }
            .code(),
            "BLOCK_CYCLE",
        ),
    ];
    for (limit, want) in [
        (Limit::BlockDepth, "LIMIT_BLOCK_DEPTH"),
        (Limit::Entities, "LIMIT_ENTITIES"),
        (Limit::PolylineVertices, "LIMIT_POLYLINE_VERTICES"),
        (Limit::SplineControlPoints, "LIMIT_SPLINE_CONTROL_POINTS"),
        (Limit::SplineKnots, "LIMIT_SPLINE_KNOTS"),
        (Limit::SplineWeights, "LIMIT_SPLINE_WEIGHTS"),
        (Limit::InsertGrid, "LIMIT_INSERT_GRID"),
        (Limit::ExpandedInstances, "LIMIT_EXPANDED_INSTANCES"),
        (Limit::AllocationBytes, "LIMIT_ALLOCATION_BYTES"),
        (Limit::NameBytes, "LIMIT_NAME_BYTES"),
    ] {
        t.push((limit_error(limit).code(), want));
    }
    t
}

/// The codes are a contract: a caller matches on them, so a rename is a break.
#[test]
fn issue_1674_every_error_keeps_its_code() {
    for (got, want) in code_table() {
        assert_eq!(got, want);
    }
}

/// A code is upper snake: a letter first, then capitals, digits and underscores.
#[test]
fn issue_1674_a_code_is_upper_snake() {
    for (got, want) in code_table() {
        for c in [got, want] {
            assert!(
                c.starts_with(|ch: char| ch.is_ascii_uppercase())
                    && c.chars()
                        .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_'),
                "{c:?} is not upper snake"
            );
        }
    }
}

/// A limit's code names the limit, so a caller tells an entity ceiling from a block depth one.
#[test]
fn issue_1674_the_limit_codes_are_all_different() {
    let mut codes: Vec<&str> = code_table()
        .into_iter()
        .map(|(got, _)| got)
        .filter(|c| c.starts_with("LIMIT_"))
        .collect();
    let n = codes.len();
    assert!(n >= 10, "{n} limit codes");
    codes.sort_unstable();
    codes.dedup();
    assert!(codes.len() >= 10, "{codes:?}");
}

/// A conversion that fails for a reason a caller can act on says which, by code.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_failed_conversion_carries_its_code() {
    let dir = scratch("codes-real");
    let dest = dir.join("a.pvec");
    let convert = |sheets: Vec<Sheet>,
                   configure: &dyn for<'a> Fn(PvecConversion<'a>) -> PvecConversion<'a>|
     -> PvecConvertError {
        drawing(sheets);
        let c = PvecConversion::new(bytes(), &dest).view_profile(ViewProfile::print_profile("A"));
        configure(c).convert_with::<Mock>().unwrap_err()
    };
    // No unit.
    let e = convert(vec![sheet("A", 2)], &|c| c);
    assert_eq!(e.code(), "UNIT_REQUIRED", "{e}");
    // A layout that is not there.
    let e = convert(vec![sheet("B", 2)], &|c| c.unit(SourceUnit::Millimeters));
    assert_eq!(e.code(), "LAYOUT_NOT_FOUND", "{e}");
    // A token that was cancelled before the run.
    let token = CancelToken::new();
    token.cancel();
    let e = convert(vec![sheet("A", 2)], &|c| {
        c.unit(SourceUnit::Millimeters).cancel(token.clone())
    });
    assert_eq!(e.code(), "CANCELLED", "{e}");
    // An entity ceiling, named by the limit.
    let mut limits = SourceLimits::default();
    limits.max_entities = 2;
    let e = convert(vec![sheet("A", 5)], &|c| {
        let mut p = ViewProfile::print_profile("A");
        p.limits = limits;
        c.view_profile(p).unit(SourceUnit::Millimeters)
    });
    assert_eq!(e.code(), "LIMIT_ENTITIES", "{e}");
    assert!(!dest.exists());
    let _ = fs::remove_dir_all(dir);
}

/// A set that cannot start says why, by code.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_failed_set_carries_its_code() {
    let dir = scratch("codes-set");
    drawing(vec![sheet("A", 2)]);
    let e = PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .layouts(["Nope"])
        .convert_with::<Mock>()
        .unwrap_err();
    assert_eq!(e.code(), "LAYOUT_NOT_FOUND", "{e}");
    drawing(vec![sheet("A", 2)]);
    let e = PvecLayoutSet::new(bytes(), &dir.join("missing"))
        .stem("t")
        .convert_with::<Mock>()
        .unwrap_err();
    assert_eq!(e.code(), "NOT_A_DIRECTORY", "{e}");
    let _ = fs::remove_dir_all(dir);
}

// ---- sheet progress ---------------------------------------------------------------------

/// Sheet 3 of 4 breaks (an entity ceiling); the observer is told of each sheet's start and end, in order,
/// with the status, and the code of the one that failed.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_set_reports_each_sheet_as_it_starts_and_ends() {
    let dir = scratch("progress-set");
    drawing(vec![
        sheet("A", 2),
        sheet("Blank", 0),
        sheet("B", 6),
        sheet("C", 1),
    ]);
    let observer = Arc::new(CollectingObserver::new());
    let seen = observer.clone();
    let out = PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .sheets(move |c| {
            let mut p = ViewProfile::print_profile("A");
            p.limits.max_entities = 3;
            c.view_profile(p)
                .unit(SourceUnit::Millimeters)
                .observer(observer.clone())
        })
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.sheets.len(), 4);
    assert_eq!(
        progress(&seen),
        [
            "start 1/4 A",
            "done 1/4 A written -",
            "start 2/4 Blank",
            "done 2/4 Blank empty -",
            "start 3/4 B",
            "done 3/4 B failed LIMIT_ENTITIES",
            "start 4/4 C",
            "done 4/4 C written -",
        ]
    );
    let _ = fs::remove_dir_all(dir);
}

/// The total is the sheets asked for, not every layout the drawing has.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_the_total_counts_the_sheets_asked_for() {
    let dir = scratch("progress-subset");
    drawing(vec![sheet("A", 1), sheet("B", 1), sheet("C", 1)]);
    let observer = Arc::new(CollectingObserver::new());
    let seen = observer.clone();
    PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .layouts(["C", "A"])
        .sheets(move |c| c.unit(SourceUnit::Millimeters).observer(observer.clone()))
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(
        progress(&seen),
        [
            "start 1/2 A",
            "done 1/2 A written -",
            "start 2/2 C",
            "done 2/2 C written -",
        ]
    );
    let _ = fs::remove_dir_all(dir);
}

/// A PDF plot tells of its pages the same way.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_pdf_plot_reports_each_page() {
    let dir = scratch("progress-pdf");
    drawing(vec![sheet("A", 2), sheet("B", 1), sheet("C", 1)]);
    let observer = Arc::new(CollectingObserver::new());
    let out = PdfPlot::new(bytes(), &dir.join("p.pdf"))
        .unit(SourceUnit::Millimeters)
        .layouts(["A", "B"])
        .observer(observer.clone())
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.pages, 2);
    assert_eq!(
        progress(&observer),
        [
            "start 1/2 A",
            "done 1/2 A written -",
            "start 2/2 B",
            "done 2/2 B written -",
        ]
    );
    let _ = fs::remove_dir_all(dir);
}

/// A PDF plot whose page fails says so before it stops.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_pdf_plot_reports_the_page_that_failed() {
    let dir = scratch("progress-pdf-fail");
    drawing(vec![sheet("A", 1), sheet("B", 6)]);
    let observer = Arc::new(CollectingObserver::new());
    let dest = dir.join("p.pdf");
    let mut p = ViewProfile::print_profile("A");
    p.limits.max_entities = 3;
    let e = PdfPlot::new(bytes(), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(p)
        .layouts(["A", "B"])
        .observer(observer.clone())
        .convert_with::<Mock>()
        .unwrap_err();
    assert_eq!(e.code(), "LIMIT_ENTITIES", "{e}");
    assert_eq!(
        progress(&observer),
        [
            "start 1/2 A",
            "done 1/2 A written -",
            "start 2/2 B",
            "done 2/2 B failed LIMIT_ENTITIES",
        ]
    );
    assert!(!dest.exists());
    let _ = fs::remove_dir_all(dir);
}

// ---- cancel -----------------------------------------------------------------------------

/// A cancel raised when sheet 1 ends stops the set before sheet 2 starts, and no index says it finished.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_cancel_after_a_sheet_stops_the_next_one_starting() {
    let dir = scratch("cancel-set");
    drawing(vec![sheet("A", 1), sheet("B", 1), sheet("C", 1)]);
    let token = CancelToken::new();
    let observer = Arc::new(CancelAfterFirst {
        token: token.clone(),
        seen: CollectingObserver::new(),
    });
    let o = observer.clone();
    let e = PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .sheets(move |c| {
            c.unit(SourceUnit::Millimeters)
                .cancel(token.clone())
                .observer(o.clone())
        })
        .convert_with::<Mock>()
        .unwrap_err();
    assert_eq!(e.code(), "CANCELLED", "{e}");
    assert_eq!(
        progress(&observer.seen),
        ["start 1/3 A", "done 1/3 A written -"]
    );
    assert!(dir.join("t.01-A.pvec").exists(), "sheet 1 was published");
    assert!(!dir.join("t.sheets.json").exists());
    let _ = fs::remove_dir_all(dir);
}

/// A PDF plot takes a cancel token, and one raised after page 1 stops it with no PDF written.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_pdf_plot_stops_on_cancel() {
    let dir = scratch("cancel-pdf");
    drawing(vec![sheet("A", 1), sheet("B", 1)]);
    let token = CancelToken::new();
    let observer = Arc::new(CancelAfterFirst {
        token: token.clone(),
        seen: CollectingObserver::new(),
    });
    let dest = dir.join("p.pdf");
    let e = PdfPlot::new(bytes(), &dest)
        .unit(SourceUnit::Millimeters)
        .layouts(["A", "B"])
        .cancel(token)
        .observer(observer.clone())
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(e, PvecConvertError::Cancelled), "{e}");
    assert_eq!(
        progress(&observer.seen),
        ["start 1/2 A", "done 1/2 A written -"]
    );
    assert!(!dest.exists());
    assert!(!dest.with_extension("plot-work").exists());
    let _ = fs::remove_dir_all(dir);
}

// ---- deadline ---------------------------------------------------------------------------

/// A deadline that has passed is its own error, not a cancel, and nothing is published.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_passed_deadline_is_not_a_cancel() {
    let dir = scratch("deadline-zero");
    let dest = dir.join("a.pvec");
    drawing(vec![sheet("A", 3)]);
    let token = CancelToken::new();
    let e = PvecConversion::new(bytes(), &dest)
        .view_profile(ViewProfile::print_profile("A"))
        .unit(SourceUnit::Millimeters)
        .cancel(token.clone())
        .deadline(Duration::ZERO)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(e, PvecConvertError::Deadline), "{e}");
    assert_eq!(e.code(), "DEADLINE");
    assert!(
        !token.is_cancelled(),
        "a deadline does not cancel the caller's token"
    );
    assert!(!dest.exists());
    let _ = fs::remove_dir_all(dir);
}

/// A slow read that runs past the deadline ends in the deadline error, with nothing published.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_slow_conversion_stops_at_its_deadline() {
    let dir = scratch("deadline-slow");
    let dest = dir.join("a.pvec");
    drawing(vec![slow("A", 100, 5)]);
    let e = PvecConversion::new(bytes(), &dest)
        .view_profile(ViewProfile::print_profile("A"))
        .unit(SourceUnit::Millimeters)
        .deadline(Duration::from_millis(50))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(e, PvecConvertError::Deadline), "{e}");
    assert!(!dest.exists());
    let _ = fs::remove_dir_all(dir);
}

/// A deadline a conversion meets changes nothing.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_deadline_that_is_met_changes_nothing() {
    let dir = scratch("deadline-met");
    let dest = dir.join("a.pvec");
    drawing(vec![sheet("A", 3)]);
    let out = PvecConversion::new(bytes(), &dest)
        .view_profile(ViewProfile::print_profile("A"))
        .unit(SourceUnit::Millimeters)
        .deadline(Duration::from_secs(3600))
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(out.written, 3);
    assert!(dest.exists());
    let _ = fs::remove_dir_all(dir);
}

/// The deadline on a set is for the whole set: the sheets before it stay, the rest never start, and
/// no index says it finished.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_set_stops_at_its_deadline() {
    let dir = scratch("deadline-set");
    drawing(vec![
        slow("A", 30, 10),
        slow("B", 30, 10),
        slow("C", 30, 10),
        slow("D", 30, 10),
    ]);
    let start = Instant::now();
    let e = PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .deadline(Duration::from_millis(400))
        .sheets(|c| c.unit(SourceUnit::Millimeters))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(e, LayoutSetError::Deadline), "{e}");
    assert_eq!(e.code(), "DEADLINE");
    assert!(
        start.elapsed() < Duration::from_millis(1100),
        "it ran {:?}, the four sheets take 1.2 s",
        start.elapsed()
    );
    assert!(!dir.join("t.sheets.json").exists());
    assert!(!dir.join("t.03-C.pvec").exists());
    assert!(!dir.join("t.04-D.pvec").exists());
    let _ = fs::remove_dir_all(dir);
}

/// A deadline a sheet's own options set bounds that sheet: it fails with the code, and the set goes on.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_deadline_on_one_sheet_fails_only_that_sheet() {
    let dir = scratch("deadline-sheet");
    drawing(vec![sheet("A", 2), sheet("B", 2)]);
    let out = PvecLayoutSet::new(bytes(), &dir)
        .stem("t")
        .sheets(|c| c.unit(SourceUnit::Millimeters).deadline(Duration::ZERO))
        .convert_with::<Mock>()
        .unwrap();
    assert!(out.index.is_some());
    let codes: Vec<&str> = out
        .sheets
        .iter()
        .map(|s| s.outcome.as_ref().err().map_or("-", SheetSkip::code))
        .collect();
    assert_eq!(codes, ["DEADLINE", "DEADLINE"]);
    let _ = fs::remove_dir_all(dir);
}

/// A PDF plot takes a deadline, and one that has passed writes no PDF and leaves no scratch directory.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1674_a_pdf_plot_stops_at_its_deadline() {
    let dir = scratch("deadline-pdf");
    drawing(vec![
        slow("A", 30, 10),
        slow("B", 30, 10),
        slow("C", 30, 10),
    ]);
    let dest = dir.join("p.pdf");
    let start = Instant::now();
    let e = PdfPlot::new(bytes(), &dest)
        .unit(SourceUnit::Millimeters)
        .layouts(["A", "B", "C"])
        .deadline(Duration::from_millis(350))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(e, PvecConvertError::Deadline), "{e}");
    assert!(
        start.elapsed() < Duration::from_millis(850),
        "it ran {:?}, the three pages take 0.9 s",
        start.elapsed()
    );
    assert!(!dest.exists());
    assert!(!dest.with_extension("plot-work").exists());
    let _ = fs::remove_dir_all(dir);
}
