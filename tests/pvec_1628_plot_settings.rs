//! Plot settings honoured on a paper sheet (#1628, spec section 6): plot area Extents, plot offset
//! and centre, a paper size override and fit to paper, over the layout's own settings. The sheet
//! is 100 by 50 mm unless a test says otherwise, with no margins, no origin and no rotation.
//!
//! The drawings are hand-built, so every frame is pinned exactly. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{
    PlotArea, PlotSettings, PvecConversion, PvecConversionOutput, PvecConvertError,
};
use pvec_facade::Curve;

// ---- a hand-built provider ------------------------------------------------------------

/// A drawing served from memory: model space (layout 0) and one paper layout, `Sheet` (layout 1).
/// `plot` is what `layout_plot` answers; `None` is a provider that cannot report plot settings.
#[derive(Clone)]
struct Mock {
    plot: Option<SourceFact<LayoutPlot>>,
    paper: Vec<RawEntity>,
    model: Vec<RawEntity>,
}

thread_local! {
    /// The drawing `Mock::open` hands back, set by the test on its own thread.
    static NEXT: RefCell<Option<Mock>> = const { RefCell::new(None) };
}

struct VecCursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for VecCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn caps() -> ProviderCaps {
    Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
        c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
    })
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1562",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1562"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: std::sync::OnceLock<ProviderCaps> = std::sync::OnceLock::new();
        CAPS.get_or_init(caps)
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(vec![
            LayoutState {
                id: LayoutId(0),
                name: "Model".into(),
                kind: LayoutKind::Model,
                block_name: "*Model_Space".into(),
            },
            LayoutState {
                id: LayoutId(1),
                name: "Sheet".into(),
                kind: LayoutKind::Paper,
                block_name: "*Paper_Space".into(),
            },
        ])
    }
    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        if layout.0 > 1 {
            return Err(ProviderError::NoSuchScope(format!("layout {}", layout.0)));
        }
        self.plot
            .clone()
            .ok_or(ProviderError::CannotReport(Facet::LayoutPlot))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        let layer = |name: &str, plottable: bool| LayerState {
            name: name.into(),
            handle: None,
            off: false,
            frozen: false,
            locked: false,
            plottable,
        };
        Ok(vec![
            layer("0", true),
            layer("Defpoints", false),
            layer("VP", true),
        ])
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => self.model.clone(),
            EntityScope::Layout(LayoutId(1)) => self.paper.clone(),
            other => return Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        };
        Ok(Box::new(VecCursor(list.into_iter())))
    }
}

fn ent(handle: u64, class: &str, layer: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry,
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: class.into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

fn line(handle: u64, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    ent(
        handle,
        "LINE",
        "0",
        RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
    )
}

/// Plot settings: `width` by `height` millimetres of paper in `units`, nothing else off its
/// default (no margins, no origin, no rotation, the layout plotted 1:1, lineweights printed).
fn plot(units: i16, width: f64, height: f64) -> LayoutPlot {
    LayoutPlot {
        paper_size: "TEST_SHEET".into(),
        paper_width: width,
        paper_height: height,
        paper_units: units,
        plot_type: 5,
        window: [0.0; 4],
        scale_numerator: 1.0,
        scale_denominator: 1.0,
        rotation: 0,
        margins: [0.0; 4],
        origin: [0.0; 2],
        style_sheet: "monochrome.ctb".into(),
        print_lineweights: true,
        scale_lineweights: false,
        plot_plot_styles: true,
        plot_viewport_borders: true,
        draw_viewports_first: false,
    }
}

/// A sheet of `plot` holding `paper`, with one model-space line far from everything.
fn sheet(plot: LayoutPlot, paper: Vec<RawEntity>) -> Mock {
    Mock {
        plot: Some(SourceFact::Present(plot)),
        paper,
        model: vec![line(0x99, [1000.0, 1000.0], [1001.0, 1000.0])],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1628-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts `mock` under the print profile of `Sheet`, in metres unless the plot says otherwise,
/// after `configure` has had its say. The artifact is `<dir>/sheet.pvec`.
fn convert(
    tag: &str,
    mock: Mock,
    configure: impl for<'a> FnOnce(PvecConversion<'a>) -> PvecConversion<'a>,
) -> (Result<PvecConversionOutput, PvecConvertError>, PathBuf) {
    let dir = scratch(tag);
    let dest = dir.join("sheet.pvec");
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let conversion = PvecConversion::new(CadSource::Bytes(b"mock"), &dest)
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Sheet"));
    let out = configure(conversion).convert_with::<Mock>();
    (out, dir)
}

// ---- the view profile ----------------------------------------------------------------
fn stored_scale(dir: &std::path::Path) -> Option<pvec_facade::unstable::ScaleClaim> {
    open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
}

fn rows_of(report: &ConversionReport, handle: u64) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle))
        .collect()
}

fn row_of(report: &ConversionReport, handle: u64) -> &EntityRow {
    let rows = rows_of(report, handle);
    assert_eq!(rows.len(), 1, "one row for {handle:#X}: {rows:#?}");
    rows[0]
}

/// The written curves with the handle of the row each came from: the artifact's entities are
/// the rows that carry geometry, in id order.
fn written(report: &ConversionReport, dir: &std::path::Path) -> Vec<(u64, Curve)> {
    let curves = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .entities()
        .unwrap();
    let handles: Vec<u64> = report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .map(|r| r.source_handle.unwrap())
        .collect();
    assert_eq!(handles.len(), curves.len(), "{:#?}", report.entities);
    handles
        .into_iter()
        .zip(curves.into_iter().map(|(_, c)| c))
        .collect()
}

/// A trimmed line as (origin, unit direction, length).
fn segment(c: &Curve) -> ([f64; 2], [f64; 2], f64) {
    let Curve::Trimmed(t) = c else {
        panic!("a trimmed line, not {c:?}");
    };
    let Curve::Line(l) = t.basis() else {
        panic!("a trimmed line, not {c:?}");
    };
    let (o, d) = (l.origin(), l.direction());
    ([o.x, o.y], [d.x, d.y], t.trim().span)
}

// ---- the cases -----------------------------------------------------------------------------

/// A sheet of 100 by 50 mm whose one line runs from `a` to `b`.
fn one_line(a: [f64; 2], b: [f64; 2]) -> Mock {
    sheet(plot(1, 100.0, 50.0), vec![line(1, a, b)])
}

/// Converts with `settings` and returns the report's sheet box as numbers, with the sheet.
fn framed(
    tag: &str,
    mock: Mock,
    settings: PlotSettings,
) -> ([f64; 4], libviprs::pvec::report::SheetSummary, PathBuf) {
    let (out, dir) = convert(tag, mock, |c| {
        c.unit(SourceUnit::Millimeters).plot_settings(settings)
    });
    let s = out.unwrap().report.sheet.unwrap();
    let b = s
        .sheet_box
        .clone()
        .unwrap_or_else(|| panic!("a paper frame: {s:#?}"))
        .map(|v| v.parse::<f64>().unwrap());
    (b, s, dir)
}

fn extents() -> PlotSettings {
    PlotSettings {
        area: PlotArea::Extents,
        ..PlotSettings::default()
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_default_settings_leave_the_layout_frame_as_it_was() {
    let (b, s, dir) = framed(
        "default",
        one_line([10.0, 10.0], [40.0, 20.0]),
        PlotSettings::default(),
    );
    assert_eq!(b, [0.0, 0.0, 100.0, 50.0]);
    assert!(
        s.notes.iter().all(|n| !n.starts_with("plot-")),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_extents_place_the_paper_at_the_extents_lower_left() {
    let (b, s, dir) = framed(
        "extents",
        one_line([200.0, 100.0], [240.0, 120.0]),
        extents(),
    );
    assert_eq!(b, [200.0, 100.0, 300.0, 150.0]);
    assert_eq!(s.frame, "paper");
    assert!(
        s.notes.contains(&"plot-area-extents".to_string()),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_extents_keep_what_the_layout_frame_would_cut() {
    // Far from the origin the layout frame holds none of it; the extents frame holds all of it.
    let (out, dir) = convert(
        "extents-kept",
        one_line([200.0, 100.0], [240.0, 120.0]),
        |c| c.unit(SourceUnit::Millimeters).plot_settings(extents()),
    );
    let report = out.unwrap().report;
    assert_eq!(row_of(&report, 1).fidelity, "SupportedEquivalent");
    assert_eq!(report.sheet.unwrap().clip["outside"], 0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_extents_wider_than_the_paper_are_cut_at_its_edge() {
    let (out, dir) = convert(
        "extents-cut",
        one_line([200.0, 100.0], [400.0, 100.0]),
        |c| c.unit(SourceUnit::Millimeters).plot_settings(extents()),
    );
    let report = out.unwrap().report;
    let s = report.sheet.clone().unwrap();
    assert_eq!(s.clip["cut"], 1, "{:?}", s.clip);
    let (_, curve) = written(&report, &dir).remove(0);
    let (origin, _, length) = segment(&curve);
    assert_eq!((origin, length), ([200.0, 100.0], 100.0));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_the_offset_moves_the_paper_against_the_drawing() {
    let layout = PlotSettings {
        offset_mm: [10.0, 5.0],
        ..PlotSettings::default()
    };
    let mut p = plot(1, 100.0, 50.0);
    p.origin = [2.0, 3.0];
    let (b, s, dir) = framed(
        "offset-layout",
        sheet(p, vec![line(1, [0.0; 2], [10.0; 2])]),
        layout,
    );
    assert_eq!(
        b,
        [-12.0, -8.0, 88.0, 42.0],
        "added to the layout's own origin"
    );
    assert!(
        s.notes.contains(&"plot-offset".to_string()),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);

    let both = PlotSettings {
        offset_mm: [10.0, 5.0],
        ..extents()
    };
    let (b, _, dir) = framed(
        "offset-extents",
        one_line([200.0, 100.0], [240.0, 120.0]),
        both,
    );
    assert_eq!(b, [190.0, 95.0, 290.0, 145.0]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_centre_puts_the_extents_in_the_middle_of_the_paper() {
    let centred = PlotSettings {
        center: true,
        // Ignored once centred.
        offset_mm: [30.0, 30.0],
        ..extents()
    };
    let (b, s, dir) = framed("centre", one_line([200.0, 100.0], [240.0, 120.0]), centred);
    assert_eq!(b, [170.0, 85.0, 270.0, 135.0]);
    assert!(
        s.notes.contains(&"plot-centered".to_string()),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);

    // Centring is about the printable area, so an unprintable left margin shifts it.
    let mut p = plot(1, 100.0, 50.0);
    p.margins = [10.0, 0.0, 0.0, 0.0];
    let (b, _, dir) = framed(
        "centre-margin",
        sheet(p, vec![line(1, [200.0, 100.0], [240.0, 120.0])]),
        PlotSettings {
            center: true,
            ..PlotSettings::default()
        },
    );
    assert_eq!(b, [165.0, 85.0, 265.0, 135.0]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_a_paper_size_override_replaces_the_paper_and_its_margins() {
    let mut p = plot(1, 100.0, 50.0);
    p.margins = [5.0, 4.0, 3.0, 2.0];
    let (b, s, dir) = framed(
        "paper",
        sheet(p, vec![line(1, [0.0; 2], [10.0; 2])]),
        PlotSettings {
            paper_mm: Some([210.0, 297.0]),
            ..PlotSettings::default()
        },
    );
    assert_eq!(b, [0.0, 0.0, 210.0, 297.0]);
    assert!(
        s.notes.contains(&"plot-paper-overridden".to_string()),
        "{:?}",
        s.notes
    );
    // The report still says what the layout itself holds.
    assert_eq!(s.plot.unwrap().paper_width, "100");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_a_paper_override_follows_the_layouts_rotation() {
    let mut p = plot(1, 100.0, 50.0);
    p.rotation = 1;
    let (b, _, dir) = framed(
        "paper-rotated",
        sheet(p, vec![line(1, [0.0; 2], [10.0; 2])]),
        PlotSettings {
            paper_mm: Some([210.0, 297.0]),
            ..PlotSettings::default()
        },
    );
    assert_eq!(b, [0.0, 0.0, 297.0, 210.0]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_fit_to_paper_scales_the_extents_into_the_printable_area() {
    let fit = PlotSettings {
        fit_to_paper: true,
        ..PlotSettings::default()
    };
    // 200 by 100 on 100 by 50: half size, so the paper shows a 200 by 100 window onto the drawing.
    let (out, dir) = convert("fit", one_line([0.0, 0.0], [200.0, 100.0]), |c| {
        c.unit(SourceUnit::Millimeters).plot_settings(fit)
    });
    let report = out.unwrap().report;
    let s = report.sheet.clone().unwrap();
    assert_eq!(
        s.sheet_box,
        Some(["0".into(), "0".into(), "200".into(), "100".into()])
    );
    assert_eq!(s.calibration, "plot-scale");
    assert_eq!(s.scale.as_deref(), Some("1:2"));
    assert!(
        s.notes.contains(&"plot-fit-to-paper".to_string()),
        "{:?}",
        s.notes
    );
    assert_eq!(s.clip["outside"] + s.clip["cut"] + s.clip["crossing"], 0);
    assert_eq!(row_of(&report, 1).fidelity, "SupportedEquivalent");
    let claim = stored_scale(&dir).expect("the fit scale is the calibration record");
    assert_eq!(claim.metres_per_source_unit, 0.001 * 2.0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_fit_with_centre_uses_the_tighter_side_and_centres_the_other() {
    // 200 by 50 on 100 by 50: the width binds (1:2), the window is 200 by 100 and 25 either side.
    let both = PlotSettings {
        fit_to_paper: true,
        center: true,
        ..PlotSettings::default()
    };
    let (b, s, dir) = framed("fit-centre", one_line([0.0, 0.0], [200.0, 50.0]), both);
    assert_eq!(b, [0.0, -25.0, 200.0, 75.0]);
    assert_eq!(s.scale.as_deref(), Some("1:2"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_fit_leaves_the_margins_unprintable() {
    // 10 mm off the left and the right: 80 of 100 printable, so 200 wide is 2.5 drawing units a mm.
    let mut p = plot(1, 100.0, 50.0);
    p.margins = [10.0, 0.0, 10.0, 0.0];
    let (b, s, dir) = framed(
        "fit-margin",
        sheet(p, vec![line(1, [0.0, 0.0], [200.0, 0.0])]),
        PlotSettings {
            fit_to_paper: true,
            ..PlotSettings::default()
        },
    );
    assert_eq!(b, [-25.0, 0.0, 225.0, 125.0]);
    assert_eq!(s.scale.as_deref(), Some("2:5"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_without_plot_settings_nothing_is_applied_and_it_says_so() {
    let mut mock = one_line([0.0; 2], [10.0; 2]);
    mock.plot = Some(SourceFact::Absent);
    let (out, dir) = convert("absent", mock, |c| {
        c.unit(SourceUnit::Millimeters).plot_settings(extents())
    });
    let s = out.unwrap().report.sheet.unwrap();
    assert_eq!(s.frame, "extents");
    assert_eq!(s.frame_reason.as_deref(), Some("plot-settings-absent"));
    assert!(
        s.notes.contains(&"plot-settings-not-applied".to_string()),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1628_a_paper_that_is_not_a_size_is_refused_not_framed() {
    for bad in [[0.0, 297.0], [210.0, -1.0], [f64::NAN, 10.0]] {
        let (out, dir) = convert("bad-paper", one_line([0.0; 2], [10.0; 2]), |c| {
            c.unit(SourceUnit::Millimeters).plot_settings(PlotSettings {
                paper_mm: Some(bad),
                ..PlotSettings::default()
            })
        });
        let s = out.unwrap().report.sheet.unwrap();
        assert_eq!(s.frame, "extents", "{bad:?}");
        assert_eq!(
            s.frame_reason.as_deref(),
            Some("paper-size-zero"),
            "{bad:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
