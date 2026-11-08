//! Layouts slice 3 (#1562): a paper layout converts to one pvec of its sheet under the print
//! profile (`docs/adr/pvec-print-profile.md`, P-1, P-3, P-6 to P-9).
//!
//! The view profile walks the layout's own block record, no-plot layers are excluded and counted
//! per layer, and each VIEWPORT on the sheet is reported `Unsupported` with `view-not-enabled`
//! until model space through viewports lands. The sheet is the paper, placed from the plot origin
//! and the margins in the plot paper unit; lines and polylines are cut exactly at its edge, a curve
//! that crosses it is refused, and the plot scale is the artifact's calibration. Without plot
//! settings the frame is the kept geometry's extent and nothing is cut.
//!
//! Most of it runs on a hand-built provider, so the frame maths, the clipper and the scale
//! reduction are pinned exactly; the last cells convert a real DWG written with acadrust, the
//! vendored multi-viewport gold probe and `sample_AC1018`. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::diagnostics::DiagnosticCode;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Exclusion, Facet,
    FacetSource, InstancePath, LayerState, LayerStateRule, LayoutId, LayoutKind, LayoutPlot,
    LayoutState, ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex,
    SourceFact, SourceLineweight, SourceUnit, SpaceSelection, ViewProfile, ViewProfileError,
    ViewportClipping, ViewportFreezeRule,
};
use libviprs::pvec_convert::{
    Calibration, PvecConversion, PvecConversionOutput, PvecConvertError, ScaleClaim,
    ScaleProvenance,
};
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, Stream};
use pvec_facade::unstable::Provenance;

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

fn polyline(handle: u64, points: &[[f64; 2]], closed: bool, bulges: &[(usize, f64)]) -> RawEntity {
    let mut vertices: Vec<RawVertex> = points
        .iter()
        .map(|p| RawVertex {
            x: p[0],
            y: p[1],
            bulge: 0.0,
        })
        .collect();
    for &(i, b) in bulges {
        vertices[i].bulge = b;
    }
    ent(
        handle,
        "LWPOLYLINE",
        "0",
        RawGeometry::Polyline {
            vertices,
            closed,
            elevation: 0.0,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

fn circle(handle: u64, centre: [f64; 2], radius: f64) -> RawEntity {
    ent(
        handle,
        "CIRCLE",
        "0",
        RawGeometry::Circle {
            center: [centre[0], centre[1], 0.0],
            radius,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

fn viewport(handle: u64) -> RawEntity {
    ent(
        handle,
        "VIEWPORT",
        "VP",
        RawGeometry::Unmapped {
            reason: "a viewport is not geometry".into(),
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

/// The clipping sheet: 100 by 50 millimetres at the origin, and one entity of each case.
const CUT_LINE: u64 = 0x10;
const INSIDE_LINE: u64 = 0x11;
const OUTSIDE_LINE: u64 = 0x12;
const ZIGZAG: u64 = 0x13;
const RING: u64 = 0x14;
const INSIDE_CIRCLE: u64 = 0x15;
const CROSSING_CIRCLE: u64 = 0x16;
const OUTSIDE_CIRCLE: u64 = 0x17;
const VIEWPORT: u64 = 0x18;
const NO_PLOT_LINE: u64 = 0x19;
const BULGED: u64 = 0x1A;

fn clipping_sheet() -> Mock {
    let mut no_plot = line(NO_PLOT_LINE, [30.0, 30.0], [40.0, 30.0]);
    no_plot.provenance.layer = "Defpoints".into();
    sheet(
        plot(1, 100.0, 50.0),
        vec![
            line(CUT_LINE, [-50.0, 25.0], [50.0, 25.0]),
            line(INSIDE_LINE, [10.0, 10.0], [20.0, 20.0]),
            line(OUTSIDE_LINE, [200.0, 10.0], [300.0, 10.0]),
            polyline(
                ZIGZAG,
                &[[10.0, 10.0], [10.0, 70.0], [20.0, 70.0], [20.0, 10.0]],
                false,
                &[],
            ),
            polyline(
                RING,
                &[[90.0, 10.0], [110.0, 10.0], [110.0, 40.0], [90.0, 40.0]],
                true,
                &[],
            ),
            circle(INSIDE_CIRCLE, [50.0, 25.0], 10.0),
            circle(CROSSING_CIRCLE, [100.0, 25.0], 10.0),
            circle(OUTSIDE_CIRCLE, [300.0, 25.0], 10.0),
            viewport(VIEWPORT),
            no_plot,
            polyline(
                BULGED,
                &[[80.0, 45.0], [90.0, 45.0], [90.0, 60.0]],
                false,
                &[(1, 0.5)],
            ),
        ],
    )
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1562-{tag}-{}", std::process::id()));
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

fn vertices(c: &Curve) -> Vec<[f64; 2]> {
    match c {
        Curve::Polyline(p) => p.vertices().iter().map(|v| [v.x, v.y]).collect(),
        other => panic!("a polyline, not {other:?}"),
    }
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

fn manifest_box(dir: &std::path::Path) -> ([f64; 2], [f64; 2], f64) {
    let r = open_published(&dir.join("sheet.pvec")).unwrap();
    let m = r.manifest();
    (m.bounds.min, m.bounds.max, m.metres_per_unit)
}

// ---- the view profile ----------------------------------------------------------------

#[test]
fn issue_1562_the_print_profile_walks_the_layouts_own_block_record() {
    let mock = clipping_sheet();
    let out = ViewProfile::print_profile("sheet")
        .apply(&mock)
        .expect("a paper layout is walked under the print profile");
    assert_eq!(out.report.space_name, "Sheet");
    let kept: Vec<u64> = out
        .included
        .iter()
        .map(|v| v.entity.provenance.handle.unwrap().get())
        .collect();
    assert_eq!(
        kept,
        [
            CUT_LINE,
            INSIDE_LINE,
            OUTSIDE_LINE,
            ZIGZAG,
            RING,
            INSIDE_CIRCLE,
            CROSSING_CIRCLE,
            OUTSIDE_CIRCLE,
            VIEWPORT,
            BULGED
        ],
        "the layout's own entities, not model space, and not the no-plot line"
    );
}

#[test]
fn issue_1562_paper_space_under_the_default_rules_stays_refused() {
    let mut p = ViewProfile::default_profile();
    p.space = SpaceSelection::PaperLayout("Sheet".into());
    assert_eq!(
        p.apply(&clipping_sheet()).unwrap_err(),
        ViewProfileError::PaperSpaceUnsupported {
            layout: "Sheet".into()
        }
    );
}

#[test]
fn issue_1562_a_no_plot_layer_is_excluded_and_counted_per_layer() {
    let out = ViewProfile::print_profile("Sheet")
        .apply(&clipping_sheet())
        .unwrap();
    assert_eq!(out.excluded.len(), 1, "{:?}", out.excluded);
    let x = &out.excluded[0];
    assert_eq!(x.handle.map(ItemHandle::get), Some(NO_PLOT_LINE));
    assert_eq!(x.reason, Exclusion::LayerNoPlot);
    assert_eq!(x.layer, "Defpoints");
    assert_eq!(x.reason.field(), "layers.no_plot");
    assert_eq!(x.reason.code(), "LAYER_HIDDEN");
    assert_eq!(out.report.excluded_by_field.get("layers.no_plot"), Some(&1));
    assert_eq!(out.report.excluded_by_layer.get("Defpoints"), Some(&1));

    // Model space with the no-plot rule switched on applies it too, now that the provider says
    // which layers plot.
    let mut model = clipping_sheet();
    model.model[0].provenance.layer = "Defpoints".into();
    let mut p = ViewProfile::default_profile();
    p.layers.no_plot = LayerStateRule::Exclude;
    let out = p.apply(&model).expect("applied, not refused");
    assert_eq!(out.included.len(), 0);
    assert_eq!(out.excluded[0].reason, Exclusion::LayerNoPlot);
}

#[test]
fn issue_1562_the_print_rules_are_exclusion_fields_only_where_they_exclude() {
    let print = ViewProfile::print_profile("Sheet").exclusion_fields();
    assert!(print.contains(&"layers.no_plot"), "{print:?}");
    assert!(print.contains(&"viewport_clipping"), "{print:?}");
    let default = ViewProfile::default_profile().exclusion_fields();
    assert!(!default.contains(&"layers.no_plot"), "{default:?}");
    assert!(!default.contains(&"viewport_clipping"), "{default:?}");
    assert!(Exclusion::ALL.contains(&Exclusion::LayerNoPlot));
}

#[test]
fn issue_1562_exact_clipping_and_applied_freezes_still_mean_nothing_on_model_space() {
    let mut p = ViewProfile::default_profile();
    p.viewport_clipping = ViewportClipping::Exact;
    assert_eq!(
        p.apply(&clipping_sheet()).unwrap_err(),
        ViewProfileError::RuleNotEnabled {
            field: "viewport_clipping"
        }
    );
    let mut p = ViewProfile::default_profile();
    p.layers.frozen_in_viewport = ViewportFreezeRule::Apply;
    assert_eq!(
        p.apply(&clipping_sheet()).unwrap_err(),
        ViewProfileError::RuleNotEnabled {
            field: "layers.frozen_in_viewport"
        }
    );
}

// ---- the sheet clip --------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_lines_and_polylines_are_cut_exactly_at_the_sheet_edge() {
    let (out, dir) = convert("cut", clipping_sheet(), |c| c);
    let out = out.expect("the sheet converts");
    let report = &out.report;

    // One row per piece, ids dense, and the counts close over the rows.
    let ids: Vec<u64> = report.entities.iter().map(|r| r.id).collect();
    assert_eq!(ids, (0..ids.len() as u64).collect::<Vec<_>>());
    assert_eq!(report.totals.encountered, report.entities.len() as u64);
    assert_eq!(out.fidelity.encountered(), report.totals.encountered);
    assert_eq!(report.entities.len(), 12, "{:#?}", report.entities);
    assert_eq!(
        rows_of(report, ZIGZAG).len(),
        2,
        "the zigzag leaves and comes back"
    );
    for r in rows_of(report, ZIGZAG) {
        assert_eq!(r.fidelity, "SupportedExact");
    }

    let curves = written(report, &dir);
    assert_eq!(out.written, 6);
    let of = |h: u64| -> Vec<&Curve> {
        curves
            .iter()
            .filter(|(k, _)| *k == h)
            .map(|(_, c)| c)
            .collect()
    };
    // The line from x = -50 starts on the left edge now.
    assert_eq!(
        segment(of(CUT_LINE)[0]),
        ([0.0, 25.0], [1.0, 0.0], 50.0),
        "cut at x = 0"
    );
    // A line wholly inside is untouched.
    assert_eq!(segment(of(INSIDE_LINE)[0]).0, [10.0, 10.0]);
    // The zigzag goes over the top edge and back: two pieces, cut at y = 50.
    let zig: Vec<Vec<[f64; 2]>> = of(ZIGZAG).into_iter().map(vertices).collect();
    assert_eq!(
        zig,
        vec![
            vec![[10.0, 10.0], [10.0, 50.0]],
            vec![[20.0, 50.0], [20.0, 10.0]]
        ]
    );
    // The closed ring over the right edge is one piece, joined where it was closed.
    let ring: Vec<Vec<[f64; 2]>> = of(RING).into_iter().map(vertices).collect();
    assert_eq!(
        ring,
        vec![vec![
            [100.0, 40.0],
            [90.0, 40.0],
            [90.0, 10.0],
            [100.0, 10.0]
        ]]
    );

    let sheet = report.sheet.as_ref().expect("a sheet section");
    let clip: Vec<(&str, u64)> = sheet.clip.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(
        clip,
        [("crossing", 2), ("cut", 3), ("outside", 2), ("pieces", 4)]
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_curves_inside_are_kept_outside_dropped_and_crossing_refused_with_508() {
    let (out, dir) = convert("curves", clipping_sheet(), |c| c);
    let report = out.unwrap().report;
    assert_eq!(row_of(&report, INSIDE_CIRCLE).fidelity, "SupportedExact");
    for h in [OUTSIDE_LINE, OUTSIDE_CIRCLE] {
        let r = row_of(&report, h);
        assert_eq!(r.fidelity, "IgnoredByPolicy", "{r:?}");
        assert_eq!(r.policy_field.as_deref(), Some("viewport_clipping"));
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("OUTSIDE_VIEWPORT") && w.contains("sheet")),
            "{r:?}"
        );
    }
    for h in [CROSSING_CIRCLE, BULGED] {
        let r = row_of(&report, h);
        assert_eq!(r.fidelity, "Unsupported", "{r:?}");
        assert_eq!(r.reason.as_deref(), Some("view-not-enabled"));
        assert!(r.warnings.iter().any(|w| w.contains("sheet edge")), "{r:?}");
        let d = report
            .diagnostics
            .iter()
            .find(|d| d.handle == Some(h))
            .expect("a diagnostic");
        assert_eq!(d.code, DiagnosticCode::EntityUnsupported);
        assert_eq!(d.message_id, "view-not-enabled");
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_viewport_and_a_no_plot_line_never_vanish() {
    let (out, dir) = convert("viewport", clipping_sheet(), |c| c);
    let report = out.unwrap().report;
    let r = row_of(&report, VIEWPORT);
    assert_eq!(r.fidelity, "Unsupported");
    assert_eq!(r.reason.as_deref(), Some("view-not-enabled"));
    assert_eq!(report.sheet.as_ref().unwrap().viewports, 1);
    let r = row_of(&report, NO_PLOT_LINE);
    assert_eq!(r.fidelity, "IgnoredByPolicy");
    assert_eq!(r.policy_field.as_deref(), Some("layers.no_plot"));
    assert_eq!(report.view.excluded_by_layer.get("Defpoints"), Some(&1));
    let _ = fs::remove_dir_all(dir);
}

// ---- the frame ---------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_the_sheet_frame_is_the_paper_placed_from_the_margins_and_origin() {
    let mut p = plot(1, 420.0, 297.0);
    p.margins = [5.0, 6.0, 7.0, 8.0];
    p.origin = [9.0, 10.0];
    let (out, dir) = convert(
        "a3",
        sheet(p, vec![line(1, [0.0, 0.0], [10.0, 10.0])]),
        |c| c,
    );
    let report = out.unwrap().report;
    assert_eq!(
        manifest_box(&dir),
        ([-14.0, -16.0], [406.0, 281.0], 0.001),
        "the paper, in millimetres, lower left at -(margin + origin)"
    );
    let s = report.sheet.unwrap();
    assert_eq!(s.layout, "Sheet");
    assert_eq!(s.frame, "paper");
    assert_eq!(s.frame_reason, None);
    assert_eq!(s.position.as_deref(), Some("unmeasured"));
    assert_eq!(s.unit, "Millimeters");
    assert_eq!(s.unit_source, "plot-paper-units");
    assert_eq!(
        s.sheet_box,
        Some(["-14".into(), "-16".into(), "406".into(), "281".into()])
    );
    let plot = s.plot.unwrap();
    assert_eq!(plot.paper_size, "TEST_SHEET");
    assert_eq!(
        (plot.paper_width.as_str(), plot.paper_height.as_str()),
        ("420", "297")
    );
    assert_eq!(plot.style_sheet, "monochrome.ctb");
    assert!(plot.plot_plot_styles && !plot.scale_lineweights);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_inches_and_a_quarter_turn_swap_the_sheet() {
    let mut p = plot(0, 215.9, 279.4);
    p.rotation = 1;
    p.origin = [25.4, 50.8];
    let (out, dir) = convert(
        "letter",
        sheet(p, vec![line(1, [0.0, 0.0], [1.0, 1.0])]),
        |c| c,
    );
    let report = out.unwrap().report;
    let (lo, hi, unit) = manifest_box(&dir);
    assert_eq!(unit, 0.0254);
    assert_eq!(lo, [-1.0, -2.0]);
    assert_eq!(
        hi,
        [-1.0 + 279.4 / 25.4, -2.0 + 215.9 / 25.4],
        "width and height swap"
    );
    let s = report.sheet.unwrap();
    assert_eq!(s.unit, "Inches");
    assert!(
        s.notes.iter().any(|n| n == "rotation-not-applied"),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_unknown_paper_units_fall_back_to_the_drawing_unit() {
    let (out, dir) = convert(
        "units",
        sheet(plot(2, 420.0, 297.0), vec![line(1, [0.0, 0.0], [0.1, 0.1])]),
        |c| c,
    );
    let report = out.unwrap().report;
    let (lo, hi, unit) = manifest_box(&dir);
    assert_eq!(unit, 1.0);
    assert_eq!((lo, hi), ([0.0, 0.0], [420.0 * 0.001, 297.0 * 0.001]));
    let s = report.sheet.unwrap();
    assert_eq!(s.unit, "Meters");
    assert_eq!(s.unit_source, "drawing");
    assert!(
        s.notes.iter().any(|n| n == "paper-units-fallback"),
        "{:?}",
        s.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_without_plot_settings_the_frame_is_the_extents_and_nothing_is_cut() {
    let cases: [(Option<SourceFact<LayoutPlot>>, &str); 3] = [
        (Some(SourceFact::Absent), "plot-settings-absent"),
        (None, "plot-settings-not-reportable"),
        (
            Some(SourceFact::Present(plot(1, 0.0, 297.0))),
            "paper-size-zero",
        ),
    ];
    for (n, (fact, why)) in cases.into_iter().enumerate() {
        let mut mock = sheet(
            plot(1, 1.0, 1.0),
            vec![line(1, [-50.0, 25.0], [50.0, 25.0])],
        );
        mock.plot = fact;
        let (out, dir) = convert(&format!("extents{n}"), mock, |c| c);
        let report = out.unwrap().report;
        assert_eq!(row_of(&report, 1).fidelity, "SupportedEquivalent", "{why}");
        assert_eq!(
            manifest_box(&dir),
            ([-50.0, 25.0], [50.0, 25.0], 1.0),
            "{why}: the box is the geometry's, in the drawing unit"
        );
        let s = report.sheet.unwrap();
        assert_eq!(s.frame, "extents", "{why}");
        assert_eq!(s.frame_reason.as_deref(), Some(why));
        assert_eq!(s.sheet_box, None);
        assert_eq!(s.calibration, "none", "{why}");
        assert!(s.clip.values().all(|v| *v == 0), "{why}: {:?}", s.clip);
        assert!(
            open_published(&dir.join("sheet.pvec"))
                .unwrap()
                .calibration()
                .unwrap()
                .is_none()
        );
        let _ = fs::remove_dir_all(dir);
    }
}

// ---- lineweights ---------------------------------------------------------------------------

fn style_widths(dir: &std::path::Path) -> Vec<bool> {
    let (streams, _) = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .streams()
        .unwrap();
    let stream: &Stream = streams
        .iter()
        .find(|s| s.capability == LineStyleStream::CAPABILITY)
        .expect("a line-style stream");
    LineStyleStream::decode(&stream.payload)
        .unwrap()
        .styles
        .iter()
        .map(|s| s.width.is_some())
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_print_lineweights_false_drops_widths_from_the_line_style_stream() {
    for print in [true, false] {
        let mut p = plot(1, 100.0, 100.0);
        p.print_lineweights = print;
        let mut l = line(1, [0.0, 0.0], [10.0, 10.0]);
        l.style.lineweight = SourceFact::Present(SourceLineweight::Hundredths(50));
        let (out, dir) = convert(&format!("lw{print}"), sheet(p, vec![l]), |c| {
            c.components(StreamSelection {
                layer: false,
                line_style: true,
                provenance: false,
            })
        });
        let report = out.unwrap().report;
        let widths = style_widths(&dir);
        assert_eq!(widths, vec![print], "print_lineweights = {print}");
        assert_eq!(report.sheet.unwrap().widths_dropped, !print);
        let _ = fs::remove_dir_all(dir);
    }
}

// ---- calibration -----------------------------------------------------------------------

fn stored_scale(dir: &std::path::Path) -> Option<pvec_facade::unstable::ScaleClaim> {
    open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
}

fn scaled(numerator: f64, denominator: f64) -> Mock {
    let mut p = plot(1, 100.0, 100.0);
    p.scale_numerator = numerator;
    p.scale_denominator = denominator;
    sheet(p, vec![line(1, [0.0, 0.0], [10.0, 10.0])])
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_the_plot_scale_is_the_calibration_record() {
    let (out, dir) = convert("scale", scaled(1.0, 100.0), |c| c);
    let s = out.unwrap().report.sheet.unwrap();
    assert_eq!(s.calibration, "plot-scale");
    assert_eq!(s.scale.as_deref(), Some("1:100"));
    let claim = stored_scale(&dir).expect("a scale record");
    assert_eq!(
        claim.provenance,
        Provenance::PlanScale {
            numerator: 1,
            denominator: 100,
            paper_metres_per_unit: 0.001,
        }
    );
    assert_eq!(claim.metres_per_source_unit, 0.001 * 100.0 / 1.0);
    assert_eq!(claim.uncertainty, None);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_decimal_plot_scale_is_reduced_to_coprime_integers() {
    let (out, dir) = convert("decimal", scaled(0.5, 1.25), |c| c);
    assert_eq!(
        out.unwrap().report.sheet.unwrap().scale.as_deref(),
        Some("2:5")
    );
    let claim = stored_scale(&dir).unwrap();
    assert_eq!(
        claim.provenance,
        Provenance::PlanScale {
            numerator: 2,
            denominator: 5,
            paper_metres_per_unit: 0.001,
        }
    );
    assert_eq!(claim.metres_per_source_unit, 0.001 * 5.0 / 2.0);
    assert_eq!(claim.uncertainty, None);
    let _ = fs::remove_dir_all(dir);
}

/// Decimal ratios that are not exact binary fractions (2.01, 4.02, 0.0079) still reduce: a side
/// is integral at a power of ten when rounding it there and dividing back gives the stored value.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_decimal_ratio_that_is_not_a_binary_fraction_still_reduces() {
    for (n, d, num, den) in [
        (1.0, 2.01, 100, 201),
        (1.0, 4.02, 50, 201),
        (0.0079, 1.0, 79, 10_000),
    ] {
        let (out, dir) = convert(&format!("dec{num}-{den}"), scaled(n, d), |c| c);
        let s = out.unwrap().report.sheet.unwrap();
        assert_eq!(s.calibration, "plot-scale", "{n}:{d}");
        assert_eq!(s.scale, Some(format!("{num}:{den}")), "{n}:{d}");
        assert!(
            !s.notes.iter().any(|x| x == "scale-not-reduced"),
            "{n}:{d}: {:?}",
            s.notes
        );
        let claim = stored_scale(&dir).unwrap();
        assert_eq!(
            claim.provenance,
            Provenance::PlanScale {
                numerator: num,
                denominator: den,
                paper_metres_per_unit: 0.001,
            },
            "{n}:{d}"
        );
        assert_eq!(claim.uncertainty, None, "{n}:{d}");
        let _ = fs::remove_dir_all(dir);
    }
}

/// A probe frames the sheet the way the conversion does. With plot paper units the drawing's unit
/// has to stand in for and no unit given, `convert_with` refuses with `UnitRequired`, so
/// `probe_with` does too instead of falling back to the extents.
#[test]
fn issue_1562_probe_with_needs_the_unit_where_the_sheet_frame_does() {
    NEXT.with(|n| {
        *n.borrow_mut() = Some(sheet(
            plot(2, 100.0, 50.0),
            vec![polyline(
                1,
                &[[10.0, 10.0], [10.0, 70.0], [20.0, 70.0], [20.0, 10.0]],
                false,
                &[],
            )],
        ))
    });
    let err = PvecConversion::new(CadSource::Bytes(b"mock"), std::path::Path::new("x.pvec"))
        .view_profile(ViewProfile::print_profile("Sheet"))
        .probe_with::<Mock>()
        .expect_err("the frame needs the drawing's unit");
    assert!(matches!(err, PvecConvertError::UnitRequired), "{err:?}");

    // With the unit given, the probe cuts the zigzag into the two pieces the conversion makes.
    NEXT.with(|n| {
        *n.borrow_mut() = Some(sheet(
            plot(2, 100.0, 50.0),
            vec![polyline(
                1,
                &[[10.0, 10.0], [10.0, 70.0], [20.0, 70.0], [20.0, 10.0]],
                false,
                &[],
            )],
        ))
    });
    let probed = PvecConversion::new(CadSource::Bytes(b"mock"), std::path::Path::new("x.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .probe_with::<Mock>()
        .unwrap();
    assert_eq!(probed.encountered(), 2, "one row per piece");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_plot_scale_that_does_not_reduce_is_written_with_its_uncertainty() {
    // The model layout of sample_AC1018 stores this one.
    let (out, dir) = convert("raw", scaled(1.0, 2.584_895_464_708_373_5), |c| c);
    let s = out.unwrap().report.sheet.unwrap();
    assert_eq!(s.calibration, "plot-scale-uncertain");
    assert!(
        s.notes.iter().any(|n| n == "scale-not-reduced"),
        "{:?}",
        s.notes
    );
    let claim = stored_scale(&dir).unwrap();
    let u = claim.uncertainty.expect("an uncertainty");
    assert!((0.0..1e-6).contains(&u), "{u}");
    assert!(claim.note.contains("2.5848954647083735"), "{}", claim.note);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_zero_plot_scale_writes_no_record() {
    let (out, dir) = convert("zero", scaled(0.0, 1.0), |c| c);
    let s = out.unwrap().report.sheet.unwrap();
    assert_eq!(s.calibration, "none");
    assert_eq!(s.scale, None);
    assert!(s.notes.iter().any(|n| n == "scale-zero"), "{:?}", s.notes);
    assert!(stored_scale(&dir).is_none());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_a_caller_calibration_wins_over_the_plot_scale() {
    let mine = Calibration::new(
        Some(ScaleClaim {
            metres_per_source_unit: 0.5,
            provenance: ScaleProvenance::Manual,
            uncertainty: None,
            note: "by hand".into(),
        }),
        None,
        None,
    )
    .unwrap();
    let (out, dir) = convert("caller", scaled(1.0, 100.0), move |c| c.calibration(mine));
    assert_eq!(out.unwrap().report.sheet.unwrap().calibration, "caller");
    let claim = stored_scale(&dir).unwrap();
    assert_eq!(claim.provenance, Provenance::Manual);
    assert_eq!(claim.metres_per_source_unit, 0.5);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1562_model_space_has_no_sheet_section() {
    let (out, dir) = convert("model", clipping_sheet(), |c| {
        c.view_profile(ViewProfile::default_profile())
    });
    let report = out.unwrap().report;
    assert!(report.sheet.is_none());
    assert!(!report.to_json_string().unwrap().contains("\"sheet\""));
    let _ = fs::remove_dir_all(dir);
}

// ---- real drawings -------------------------------------------------------------------------

#[cfg(feature = "dwg-acadrust")]
mod with_a_drawing {
    use super::*;
    use acadrust::entities::{Line, Viewport};
    use acadrust::objects::ObjectType;
    use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
    use libviprs::pvec::FidelityClass;
    use libviprs::pvec::ReasonCode;

    /// An A3 sheet in millimetres plotted 1:1, holding a line over its left edge, a line on a
    /// no-plot layer and a viewport. Returns the bytes and the three handles.
    fn a3() -> (Vec<u8>, [u64; 3]) {
        let mut doc = CadDocument::new();
        let mut defpoints = acadrust::tables::Layer::new("Defpoints");
        defpoints.is_plottable = false;
        defpoints.handle = doc.allocate_handle();
        doc.layers.add(defpoints).expect("layer");
        let layout = doc.add_layout("Sheet A").expect("layout");
        let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
            panic!("the layout object");
        };
        l.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
        l.paper_width = 420.0;
        l.paper_height = 297.0;
        l.plot_paper_units = 1;
        l.plot_type = 5;
        l.plot_scale_numerator = 1.0;
        l.plot_scale_denominator = 1.0;
        l.plot_rotation = 0;
        l.plot_margin_left = 0.0;
        l.plot_margin_bottom = 0.0;
        l.plot_origin_x = 0.0;
        l.plot_origin_y = 0.0;
        l.plot_flags.print_lineweights = true;
        let mut add = |e: EntityType| {
            doc.add_entity_to_layout(e, "Sheet A")
                .expect("entity")
                .value()
        };
        let crossing = add(EntityType::Line(Line::from_points(
            Vector3::new(-10.0, 100.0, 0.0),
            Vector3::new(100.0, 100.0, 0.0),
        )));
        let mut hidden =
            Line::from_points(Vector3::new(10.0, 10.0, 0.0), Vector3::new(20.0, 10.0, 0.0));
        hidden.common.layer = "Defpoints".into();
        let hidden = add(EntityType::Line(hidden));
        let vp = add(EntityType::Viewport(Box::new(Viewport::with_size(
            Vector3::new(200.0, 150.0, 0.0),
            100.0,
            80.0,
        ))));
        let mut bytes = Vec::new();
        DwgWriter::write_to_writer(std::io::Cursor::new(&mut bytes), &doc).expect("write");
        (bytes, [crossing, hidden, vp])
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1562_a_real_layout_converts_to_its_sheet() {
        let (bytes, [crossing, hidden, vp]) = a3();
        let dir = scratch("real");
        let dest = dir.join("sheet.pvec");
        let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
            .view_profile(ViewProfile::print_profile("Sheet A"))
            .convert()
            .expect("the sheet converts");
        let report = &out.report;
        let s = report.sheet.as_ref().expect("a sheet section");
        assert_eq!(
            (s.frame.as_str(), s.unit.as_str()),
            ("paper", "Millimeters")
        );
        assert_eq!(manifest_box(&dir), ([0.0, 0.0], [420.0, 297.0], 0.001));
        assert_eq!(
            row_of(report, hidden).policy_field.as_deref(),
            Some("layers.no_plot")
        );
        assert_eq!(
            row_of(report, vp).reason.as_deref(),
            Some("view-not-enabled")
        );
        assert_eq!(row_of(report, crossing).fidelity, "SupportedEquivalent");
        let curves = written(report, &dir);
        assert_eq!(curves.len(), 1);
        assert_eq!(segment(&curves[0].1), ([0.0, 100.0], [1.0, 0.0], 100.0));
        let claim = stored_scale(&dir).expect("the plot scale");
        assert_eq!(
            claim.provenance,
            Provenance::PlanScale {
                numerator: 1,
                denominator: 1,
                paper_metres_per_unit: 0.001,
            }
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1562_every_viewport_of_the_gold_probe_is_reported_not_dropped() {
        let bytes =
            include_bytes!("fixtures/pvec-gold/fixtures/paper-space-multi-viewport-001/source.dwg");
        let probed = PvecConversion::new(CadSource::Bytes(bytes), std::path::Path::new("x.pvec"))
            .view_profile(ViewProfile::print_profile("Layout1"))
            .probe()
            .expect("Layout1 is walked");
        assert_eq!(probed.encountered(), 20);
        assert_eq!(probed.count(FidelityClass::Unsupported), 20);
        let reasons: Vec<_> = probed.reasons().collect();
        assert_eq!(reasons, [(ReasonCode::ViewNotEnabled, "VIEWPORT", 20)]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1562_the_sample_drawings_paper_layouts_are_walked_not_refused() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/cad/sample_AC1018.dwg");
        for layout in ["Layout1", "Layout2", "MyLayout"] {
            let probed =
                PvecConversion::new(CadSource::Path(&path), std::path::Path::new("x.pvec"))
                    .view_profile(ViewProfile::print_profile(layout))
                    .probe()
                    .unwrap_or_else(|e| panic!("{layout}: {e:?}"));
            let reasons: Vec<_> = probed.reasons().collect();
            assert_eq!(
                reasons,
                [(ReasonCode::ViewNotEnabled, "VIEWPORT", 2)],
                "{layout}"
            );
        }
    }
}
