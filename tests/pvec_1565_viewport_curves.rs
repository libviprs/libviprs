//! Layouts slice 6 (#1565): a curve crossing a viewport or the sheet is cut exactly, and a
//! viewport clipped to a polygon draws what it shows inside the polygon
//! (`docs/adr/pvec-print-profile.md`, P-7).
//!
//! Slices 3 and 4 cut lines and straight polylines and refused any other crossing curve with
//! 508. Here a circle, arc, ellipse, spline or bulged polyline is split by
//! `pvec::curve_clip` (#1567): conics as arcs of their own carrier (`SupportedExact`), splines as
//! sub-ranges of themselves (`SupportedEquivalent`, the parameter tolerance in the row). A
//! straight-edged polygon clip is drawn; a bulged one, a circle, ellipse or spline boundary and a
//! twisted viewport stay refused. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow, SheetViewport};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact, SourceUnit,
    ViewProfile, ViewportClip, ViewportEntity,
};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput, PvecConvertError};
use pvec_facade::Curve;
use pvec_geometry::Curve2d;

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), with its viewports.
#[derive(Clone)]
struct Mock {
    paper: Vec<RawEntity>,
    model: Vec<RawEntity>,
    viewports: Vec<ViewportEntity>,
}

thread_local! {
    static NEXT: RefCell<Option<Mock>> = const { RefCell::new(None) };
}

struct VecCursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for VecCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1565",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1565"
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
        // An A3 sheet in millimetres plotted 1:1 at the paper-space origin.
        Ok(SourceFact::Present(LayoutPlot {
            paper_size: "ISO_A3_(420.00_x_297.00_MM)".into(),
            paper_width: 420.0,
            paper_height: 297.0,
            paper_units: 1,
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
        }))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            handle: None,
            off: false,
            frozen: false,
            locked: false,
            plottable: true,
        }])
    }
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        Ok(self.viewports.clone())
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

fn ent(handle: u64, class: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry,
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: class.into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

fn line(handle: u64, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    ent(
        handle,
        "LINE",
        RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
    )
}

fn circle(handle: u64, centre: [f64; 2], radius: f64) -> RawEntity {
    ent(
        handle,
        "CIRCLE",
        RawGeometry::Circle {
            center: [centre[0], centre[1], 0.0],
            radius,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

fn spline(handle: u64, weights: SourceFact<Vec<f64>>) -> RawEntity {
    ent(
        handle,
        "SPLINE",
        RawGeometry::Spline {
            degree: 3,
            // Two knot spans, so it stays a B-spline (one span would be a Bezier).
            control_points: vec![
                [6.0, 4.0, 0.0],
                [8.0, 7.0, 0.0],
                [11.0, 7.0, 0.0],
                [12.0, 6.0, 0.0],
                [13.0, 3.0, 0.0],
            ],
            knots: SourceFact::Present(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            weights,
            periodic: SourceFact::Present(false),
            closed: false,
        },
    )
}

fn viewport_entity(handle: u64) -> RawEntity {
    ent(
        handle,
        "VIEWPORT",
        RawGeometry::Unmapped {
            reason: "a viewport is not geometry".into(),
        },
    )
}

/// A plan-view, untwisted, rectangular viewport that is on and prints.
fn vp(
    handle: u64,
    id: i16,
    center: [f64; 2],
    size: [f64; 2],
    view: [f64; 2],
    height: f64,
) -> ViewportEntity {
    let (hw, hh) = (size[0] / 2.0, size[1] / 2.0);
    ViewportEntity {
        handle: ItemHandle::new(handle),
        layout: LayoutId(1),
        id,
        visible: true,
        clip: ViewportClip::Rectangle([
            [center[0] - hw, center[1] - hh],
            [center[0] + hw, center[1] - hh],
            [center[0] + hw, center[1] + hh],
            [center[0] - hw, center[1] + hh],
        ]),
        frozen_layers: Vec::new(),
        layer: "0".into(),
        center,
        width: size[0],
        height: size[1],
        view_center: view,
        view_height: height,
        view_target: [0.0, 0.0, 0.0],
        view_direction: [0.0, 0.0, 1.0],
        twist: 0.0,
        perspective: false,
        hide_plot: false,
        non_rectangular_clipping: false,
    }
}

fn polygon(corners: &[[f64; 2]], bulge: f64) -> ViewportClip {
    ViewportClip::Polygon(
        corners
            .iter()
            .map(|p| RawVertex {
                x: p[0],
                y: p[1],
                bulge,
            })
            .collect(),
    )
}

// A: 10 paper units per model unit, centred at (100, 100) on model (5, 5), showing model x 0..10,
// y 1..9 in a 100 x 80 rectangle. T: the same view, clipped to a triangle. B and R: a bulged
// polygon and a circle boundary, both still refused, and W: a twisted viewport.
const VP_A: u64 = 0x31;
const VP_T: u64 = 0x32;
const VP_B: u64 = 0x33;
const VP_R: u64 = 0x34;
const VP_W: u64 = 0x35;

const EDGE_RING: u64 = 0x40;
const SPLINE: u64 = 0x41;
const RATIONAL: u64 = 0x42;
const ACROSS: u64 = 0x43;
const RING: u64 = 0x44;
const PAPER_RING: u64 = 0x45;
const RATIONAL_IN: u64 = 0x46;
const CORNER_DOT: u64 = 0x47;
const RATIONAL_AROUND: u64 = 0x48;

/// The triangle T is clipped to, in paper units: its apex is above A's centre.
const TRIANGLE: [[f64; 2]; 3] = [[260.0, 60.0], [340.0, 60.0], [300.0, 140.0]];

fn drawing() -> Mock {
    let elsewhere = |h: u64, id: i16| vp(h, id, [210.0, 250.0], [40.0, 30.0], [0.0, 0.0], 3.0);
    let mut t = vp(VP_T, 3, [300.0, 100.0], [100.0, 80.0], [5.0, 5.0], 8.0);
    t.clip = polygon(&TRIANGLE, 0.0);
    t.non_rectangular_clipping = true;
    let mut b = elsewhere(VP_B, 4);
    b.clip = polygon(&[[190.0, 235.0], [230.0, 235.0], [210.0, 265.0]], 0.3);
    b.non_rectangular_clipping = true;
    let mut r = elsewhere(VP_R, 5);
    r.clip = ViewportClip::Circle(ItemHandle::new(0x99));
    r.non_rectangular_clipping = true;
    let mut w = elsewhere(VP_W, 6);
    w.twist = 0.5;
    let viewports = vec![
        vp(VP_A, 2, [100.0, 100.0], [100.0, 80.0], [5.0, 5.0], 8.0),
        t,
        b,
        r,
        w,
    ];
    let mut paper: Vec<RawEntity> = viewports
        .iter()
        .map(|v| viewport_entity(v.handle.get()))
        .collect();
    // A paper-space circle over the sheet's left edge.
    paper.push(circle(PAPER_RING, [0.0, 150.0], 20.0));
    Mock {
        paper,
        model: vec![
            // Model x 8..12 crosses A's right edge at x 10.
            circle(EDGE_RING, [10.0, 5.0], 2.0),
            spline(SPLINE, SourceFact::Absent),
            spline(RATIONAL, SourceFact::Present(vec![1.0, 0.5, 2.0, 0.5, 1.0])),
            line(ACROSS, [-20.0, 5.0], [20.0, 5.0]),
            circle(RING, [5.0, 5.0], 3.0),
            // Paper (295..305, 90..100) through T, well inside the triangle.
            ent(
                RATIONAL_IN,
                "SPLINE",
                RawGeometry::Spline {
                    degree: 2,
                    control_points: vec![
                        [4.5, 4.0, 0.0],
                        [5.0, 5.0, 0.0],
                        [5.5, 4.0, 0.0],
                        [5.5, 4.5, 0.0],
                    ],
                    knots: SourceFact::Present(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]),
                    weights: SourceFact::Present(vec![1.0, 0.5, 1.0, 1.0]),
                    periodic: SourceFact::Present(false),
                    closed: false,
                },
            ),
            // Paper (265, 130) r 2 through T: inside the triangle's box, outside the triangle.
            circle(CORNER_DOT, [1.5, 8.0], 0.2),
            // Paper (250..350, 50..150) through T: its box holds the whole triangle, and the curve
            // runs through it.
            ent(
                RATIONAL_AROUND,
                "SPLINE",
                RawGeometry::Spline {
                    degree: 2,
                    control_points: vec![
                        [0.0, 0.0, 0.0],
                        [10.0, 2.0, 0.0],
                        [10.0, 8.0, 0.0],
                        [0.0, 10.0, 0.0],
                    ],
                    knots: SourceFact::Present(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]),
                    weights: SourceFact::Present(vec![1.0, 0.5, 0.5, 1.0]),
                    periodic: SourceFact::Present(false),
                    closed: false,
                },
            ),
        ],
        viewports,
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1565-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn convert(tag: &str) -> (Result<PvecConversionOutput, PvecConvertError>, PathBuf) {
    let dir = scratch(tag);
    let dest = dir.join("sheet.pvec");
    NEXT.with(|n| *n.borrow_mut() = Some(drawing()));
    let out = PvecConversion::new(CadSource::Bytes(b"mock"), &dest)
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .convert_with::<Mock>();
    (out, dir)
}

fn rows(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle) && r.viewport == viewport)
        .collect()
}

/// The written curves of `handle` through `viewport`, with their rows: the artifact's entities
/// are the rows that carry geometry, in id order.
fn pieces(
    report: &ConversionReport,
    dir: &Path,
    handle: u64,
    viewport: Option<u64>,
) -> Vec<(EntityRow, Curve)> {
    let curves = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .entities()
        .unwrap();
    let kept: Vec<EntityRow> = report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .cloned()
        .collect();
    assert_eq!(kept.len(), curves.len(), "{:#?}", report.entities);
    kept.into_iter()
        .zip(curves.into_iter().map(|(_, c)| c))
        .filter(|(r, _)| r.source_handle == Some(handle) && r.viewport == viewport)
        .collect()
}

fn ends(c: &Curve) -> [[f64; 2]; 2] {
    let d = c.domain();
    let (a, z) = (c.eval(d.lo()).unwrap(), c.eval(d.hi()).unwrap());
    [[a.x, a.y], [z.x, z.y]]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

/// The distance from `q` to the nearest edge of the closed ring.
fn off_boundary(ring: &[[f64; 2]], q: [f64; 2]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let t =
                (((q[0] - a[0]) * ex + (q[1] - a[1]) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
            ((q[0] - a[0] - t * ex).powi(2) + (q[1] - a[1] - t * ey).powi(2)).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn sheet_row(report: &ConversionReport, handle: u64) -> &SheetViewport {
    let s = report.sheet.as_ref().expect("a sheet section");
    s.viewports
        .iter()
        .find(|v| v.handle == handle)
        .unwrap_or_else(|| panic!("no sheet row for {handle:#X}: {:#?}", s.viewports))
}

// ---- curves through a rectangle ---------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_circle_over_a_viewport_edge_is_cut_into_an_arc_of_itself() {
    let (out, dir) = convert("ring");
    let report = out.expect("the sheet converts").report;
    let cut = pieces(&report, &dir, EDGE_RING, Some(VP_A));
    assert_eq!(cut.len(), 1, "{cut:#?}");
    let (row, curve) = &cut[0];
    assert_eq!(row.fidelity, "SupportedExact");
    let Curve::Trimmed(t) = curve else {
        panic!("an arc, not {curve:?}");
    };
    let Curve::Circle(c) = t.basis() else {
        panic!("an arc of a circle, not {curve:?}");
    };
    // Model (10, 5) r 2 is paper (150, 100) r 20, and A's right edge is x 150: the left half.
    let o = c.placement().origin();
    assert_eq!(([o.x, o.y], c.radius()), ([150.0, 100.0], 20.0));
    assert!(close(t.trim().span, std::f64::consts::PI), "{:?}", t.trim());
    for q in ends(curve) {
        assert!(close(q[0], 150.0), "{q:?}");
    }
    let sheet = report.sheet.as_ref().unwrap();
    assert!(
        sheet.clip.get("cut").is_some_and(|n| *n >= 1),
        "{:?}",
        sheet.clip
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_spline_over_a_viewport_edge_is_cut_as_a_sub_range_and_says_its_tolerance() {
    let (out, dir) = convert("spline");
    let report = out.expect("the sheet converts").report;
    let cut = pieces(&report, &dir, SPLINE, Some(VP_A));
    assert_eq!(cut.len(), 1, "{:#?}", rows(&report, SPLINE, Some(VP_A)));
    let (row, curve) = &cut[0];
    assert_eq!(row.fidelity, "SupportedEquivalent");
    assert!(matches!(curve, Curve::BSpline(_)), "{curve:?}");
    assert!(
        row.warnings
            .iter()
            .any(|w| w.contains("parameter tolerance")),
        "{row:?}"
    );
    // It starts inside at model (6, 4), paper (110, 90), and ends on A's right edge.
    let [a, z] = ends(curve);
    assert!(close(a[0], 110.0) && close(a[1], 90.0), "{a:?}");
    assert!((z[0] - 150.0).abs() <= 1e-6, "{z:?}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_rational_spline_over_an_edge_is_still_refused_with_508() {
    let (out, dir) = convert("rational");
    let report = out.expect("the sheet converts").report;
    let r = rows(&report, RATIONAL, Some(VP_A));
    assert_eq!(r.len(), 1, "{r:#?}");
    assert_eq!(r[0].fidelity, "Unsupported");
    assert_eq!(r[0].reason.as_deref(), Some("view-not-enabled"));
    assert!(
        r[0].warnings.iter().any(|w| w.contains("rational")),
        "{:?}",
        r[0]
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_circle_over_the_sheet_edge_is_cut_too() {
    let (out, dir) = convert("paper");
    let report = out.expect("the sheet converts").report;
    let cut = pieces(&report, &dir, PAPER_RING, None);
    assert_eq!(cut.len(), 1, "{cut:#?}");
    for q in ends(&cut[0].1) {
        assert!(close(q[0], 0.0), "{q:?}");
    }
    let _ = fs::remove_dir_all(dir);
}

// ---- polygon clips ----------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_viewport_clipped_to_a_polygon_draws_inside_the_polygon() {
    let (out, dir) = convert("polygon");
    let report = out.expect("the sheet converts").report;
    let t = sheet_row(&report, VP_T);
    assert_eq!((t.status.as_str(), t.reason.as_deref()), ("drawn", None));
    // The line y = 5 is paper y 100 through T: the triangle is 280..320 wide there.
    let line = pieces(&report, &dir, ACROSS, Some(VP_T));
    assert_eq!(line.len(), 1, "{line:#?}");
    let [a, z] = ends(&line[0].1);
    assert!(close(a[0], 280.0) && close(a[1], 100.0), "{a:?}");
    assert!(close(z[0], 320.0) && close(z[1], 100.0), "{z:?}");
    // The circle r 3 about the centre is paper r 30 about (300, 100). It clears the base, and
    // each slanted side cuts off a cap: two arcs are left, the bottom one and the top one through
    // the apex, each starting and ending on the triangle.
    let ring = pieces(&report, &dir, RING, Some(VP_T));
    assert_eq!(ring.len(), 2, "{ring:#?}");
    for (row, c) in &ring {
        assert_eq!(row.fidelity, "SupportedExact");
        for q in ends(c) {
            assert!(off_boundary(&TRIANGLE, q) <= 1e-9, "{q:?}");
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_bulged_and_curved_clips_and_a_twist_stay_refused() {
    let (out, dir) = convert("refused");
    let report = out.expect("the sheet converts").report;
    let status = |h: u64| {
        let r = sheet_row(&report, h);
        (r.status.clone(), r.reason.clone())
    };
    assert_eq!(status(VP_B), ("refused".into(), Some("clip-bulge".into())));
    assert_eq!(status(VP_R), ("refused".into(), Some("clip-circle".into())));
    assert_eq!(status(VP_W), ("refused".into(), Some("twist".into())));
    for v in [VP_B, VP_R, VP_W] {
        assert!(rows(&report, ACROSS, Some(v)).is_empty());
    }
    let _ = fs::remove_dir_all(dir);
}

// ---- the gold probe ---------------------------------------------------------------------

#[cfg(feature = "dwg-acadrust")]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_viewport_clipped_001_draws_its_line_cut_at_the_clip_polygon() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-gold/fixtures/viewport-clipped-001/source.dwg");
    let dir = scratch("gold");
    // The probe declares no unit, so the caller says one, as for any unitless drawing.
    let out = PvecConversion::new(CadSource::Path(&path), &dir.join("sheet.pvec"))
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .convert()
        .expect("the probe converts through the print profile");
    let report = out.report;
    let sheet = report.sheet.as_ref().expect("a sheet section");
    let drawn: Vec<&SheetViewport> = sheet
        .viewports
        .iter()
        .filter(|v| v.status == "drawn")
        .collect();
    assert_eq!(drawn.len(), 1, "{:#?}", sheet.viewports);
    let vp = drawn[0].handle;
    let line: Vec<(EntityRow, Curve)> = {
        let curves = open_published(&dir.join("sheet.pvec"))
            .unwrap()
            .entities()
            .unwrap();
        let kept: Vec<EntityRow> = report
            .entities
            .iter()
            .filter(|r| r.semantic_type.is_some())
            .cloned()
            .collect();
        kept.into_iter()
            .zip(curves.into_iter().map(|(_, c)| c))
            .filter(|(r, _)| r.viewport == Some(vp))
            .collect()
    };
    assert_eq!(line.len(), 1, "{:#?}", report.entities);
    // Model (1, 1)..(9, 7) at 6 paper units per model unit about (100, 75) on (5, 4) is paper
    // (76, 57)..(124, 93). The clip (70, 55), (130, 60), (120, 95), (85, 100) cuts it on the edge
    // from (130, 60) to (120, 95), at 15/17 of the way: (2060/17, 1545/17).
    let [a, z] = ends(&line[0].1);
    assert!(close(a[0], 76.0) && close(a[1], 57.0), "{a:?}");
    assert!(
        close(z[0], 2060.0 / 17.0) && close(z[1], 1545.0 / 17.0),
        "{z:?}"
    );
    // The probe's line is SupportedEquivalent in model space already
    // (tests/fixtures/pvec-gate/classification.baseline), and a straight cut keeps its class.
    assert_eq!(line[0].0.fidelity, "SupportedEquivalent");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_polygon_keeps_what_is_inside_it_and_drops_what_is_only_inside_its_box() {
    let (out, dir) = convert("polygon-whole");
    let report = out.expect("the sheet converts").report;
    // A rational spline wholly inside is kept whole, even though it could not be cut.
    let inside = rows(&report, RATIONAL_IN, Some(VP_T));
    assert_eq!(inside.len(), 1, "{inside:#?}");
    assert_ne!(inside[0].fidelity, "Unsupported", "{:?}", inside[0]);
    assert!(inside[0].semantic_type.is_some(), "{:?}", inside[0]);
    // A circle inside the triangle's box but outside the triangle is dropped as outside.
    let corner = rows(&report, CORNER_DOT, Some(VP_T));
    assert_eq!(corner.len(), 1, "{corner:#?}");
    assert_eq!(corner[0].fidelity, "IgnoredByPolicy", "{:?}", corner[0]);
    assert_eq!(corner[0].policy_field.as_deref(), Some("viewport_clipping"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1565_a_rational_spline_whose_box_holds_the_polygon_is_refused_not_dropped() {
    let (out, dir) = convert("polygon-around");
    let report = out.expect("the sheet converts").report;
    let r = rows(&report, RATIONAL_AROUND, Some(VP_T));
    assert_eq!(r.len(), 1, "{r:#?}");
    assert_eq!(r[0].fidelity, "Unsupported", "{:?}", r[0]);
    assert_eq!(r[0].reason.as_deref(), Some("view-not-enabled"));
    let _ = fs::remove_dir_all(dir);
}
