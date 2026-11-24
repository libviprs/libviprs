//! A DWG print layout converts with its linework in the unit its bounds and descriptor say (#1665).
//!
//! A layout's plot scale says how many paper units one paper-space drawing unit prints as. The
//! drawing this came from stores its Tabloid sheets in millimetres at 1 : 0.04, so one drawing
//! unit is 25 mm of paper, and its "Letter - P" sheet says the same in inches (1 in : 1.016
//! units). The converter frames the sheet in paper millimetres (or inches) but writes the
//! linework as if one drawing unit were one paper unit, so a 431.8 by 279.4 mm sheet carries
//! linework about 17 units across and draws as a tiny copy in its corner.
//!
//! Every cell measures in metres from the frame's lower-left corner, through the manifest's
//! metres per unit, so it holds whichever way the fix goes (scale the linework into the paper
//! unit, or carry the drawing unit as the descriptor's unit). Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::SheetSummary;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact, SourceLineweight,
    SourceUnit, ViewProfile, ViewportClip, ViewportEntity,
};
use libviprs::pvec_convert::{PlotArea, PlotScale, PlotSettings, PvecConversion};
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, Stream};
use pvec_facade::unstable::Provenance;

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), whose plot settings are
/// `plot` and whose own entities are `paper`.
#[derive(Clone)]
struct Mock {
    plot: LayoutPlot,
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
                provider: "mock-1665",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1665"
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
        Ok(SourceFact::Present(self.plot.clone()))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            handle: None,
            off: false,
            frozen: false,
            locked: false,
            plottable: true,
            ..LayerState::default()
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

/// A closed rectangle on layer "0", in paper-space drawing units.
fn rectangle(handle: u64, lo: [f64; 2], hi: [f64; 2]) -> RawEntity {
    let corner = |x: f64, y: f64| RawVertex { x, y, bulge: 0.0 };
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Polyline {
            vertices: vec![
                corner(lo[0], lo[1]),
                corner(hi[0], lo[1]),
                corner(hi[0], hi[1]),
                corner(lo[0], hi[1]),
            ],
            closed: true,
            elevation: 0.0,
            normal: [0.0, 0.0, 1.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LWPOLYLINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

/// Plot settings as the drawing stores them: the paper always in millimetres, `units` saying
/// which unit the plot dialog shows, and `n` paper units to `d` drawing units.
fn plot(units: i16, paper: [f64; 2], rotation: i16, [n, d]: [f64; 2]) -> LayoutPlot {
    LayoutPlot {
        paper_size: "TEST_SHEET".into(),
        paper_width: paper[0],
        paper_height: paper[1],
        paper_units: units,
        plot_type: 5,
        window: [0.0; 4],
        scale_numerator: n,
        scale_denominator: d,
        rotation,
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

/// The Tabloid sheet of the report: 279.4 by 431.8 mm turned a quarter, 1 mm to 0.04 units, so
/// one drawing unit is 25 mm and the sheet is 17.272 by 11.176 units. The rectangle sits one
/// unit (25 mm) in from every edge.
fn tabloid() -> Mock {
    Mock {
        plot: plot(1, [279.4, 431.8], 1, [1.0, 0.04]),
        paper: vec![rectangle(0x10, [1.0, 1.0], [16.272, 10.176])],
        model: vec![],
        viewports: vec![],
    }
}

/// The "Letter - P" sheet: 215.9 by 279.4 mm in inches, 1 in to 1.016 units, so one drawing
/// unit is 25 mm here too and the sheet is 8.636 by 11.176 units. The rectangle sits one unit
/// in from every edge.
fn letter() -> Mock {
    Mock {
        plot: plot(0, [215.9, 279.4], 0, [1.0, 1.016]),
        paper: vec![rectangle(0x10, [1.0, 1.0], [7.636, 10.176])],
        model: vec![],
        viewports: vec![],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1665-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts `mock` under the print profile of `Sheet`, the caller saying `unit` and `settings`.
fn convert(
    tag: &str,
    mock: Mock,
    unit: SourceUnit,
    settings: PlotSettings,
) -> (PathBuf, SheetSummary) {
    let dir = scratch(tag);
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let out = PvecConversion::new(CadSource::Bytes(b"mock"), &dir.join("sheet.pvec"))
        .unit(unit)
        .plot_settings(settings)
        .components(StreamSelection {
            layer: false,
            line_style: true,
            provenance: false,
            stroke_paint: false,
        })
        .view_profile(ViewProfile::print_profile("Sheet"))
        .convert_with::<Mock>()
        .expect("the sheet converts");
    (dir, out.report.sheet.expect("a paper sheet"))
}

/// The frame's size and the drawn linework's extent, both in metres, the extent measured from
/// the frame's lower-left corner.
fn measured(dir: &Path) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let path = dir.join("sheet.pvec");
    let (lo, hi, mpu) = {
        let r = open_published(&path).unwrap();
        let m = r.manifest();
        (m.bounds.min, m.bounds.max, m.metres_per_unit)
    };
    let curves = open_published(&path).unwrap().entities().unwrap();
    assert!(!curves.is_empty(), "the sheet draws something");
    let mut points = Vec::new();
    for (_, c) in &curves {
        match c {
            Curve::Polyline(p) => points.extend(p.vertices().iter().map(|v| [v.x, v.y])),
            Curve::Trimmed(t) => {
                let Curve::Line(l) = t.basis() else {
                    panic!("a trimmed line, not {c:?}");
                };
                let (o, d, tr) = (l.origin(), l.direction(), t.trim());
                for s in [tr.start, tr.start + tr.span] {
                    points.push([o.x + d.x * s, o.y + d.y * s]);
                }
            }
            other => panic!("a polyline or a line, not {other:?}"),
        }
    }
    let metres = |p: [f64; 2]| [(p[0] - lo[0]) * mpu, (p[1] - lo[1]) * mpu];
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for p in points.into_iter().map(metres) {
        min = [min[0].min(p[0]), min[1].min(p[1])];
        max = [max[0].max(p[0]), max[1].max(p[1])];
    }
    let frame = [(hi[0] - lo[0]) * mpu, (hi[1] - lo[1]) * mpu];
    (frame, min, max)
}

fn assert_close(what: &str, got: [f64; 2], want: [f64; 2]) {
    let off = (got[0] - want[0]).abs().max((got[1] - want[1]).abs());
    assert!(off < 1e-9, "{what}: {got:?} m, expected {want:?} m");
}

// ---- the cells -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_millimetre_sheet_at_1_to_0_04_draws_at_full_paper_size() {
    let (dir, _) = convert(
        "tabloid",
        tabloid(),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    // One drawing unit is 25 mm of paper: the rectangle runs from 25 mm in to 25 mm short of
    // the far edges, not 17 mm across in the corner.
    assert_close("the rectangle's lower left", min, [0.025, 0.025]);
    assert_close("the rectangle's upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_an_inch_sheet_at_1_to_1_016_draws_at_full_paper_size() {
    let (dir, _) = convert(
        "letter",
        letter(),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.2159, 0.2794]);
    // 1 in to 1.016 units is 25 mm a unit, not 25.4: the rectangle ends 25 mm short of the
    // paper's edges, not 1.6% further out.
    assert_close("the rectangle's lower left", min, [0.025, 0.025]);
    assert_close("the rectangle's upper right", max, [0.1909, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_the_callers_unit_does_not_change_a_plotted_sheet() {
    for (tag, unit) in [
        ("m", SourceUnit::Meters),
        ("mm", SourceUnit::Millimeters),
        ("in", SourceUnit::Inches),
    ] {
        let (dir, _) = convert(
            &format!("caller-{tag}"),
            tabloid(),
            unit,
            PlotSettings::default(),
        );
        let (frame, min, max) = measured(&dir);
        assert_close(&format!("{tag}: the frame"), frame, [0.4318, 0.2794]);
        assert_close(&format!("{tag}: the lower left"), min, [0.025, 0.025]);
        assert_close(&format!("{tag}: the upper right"), max, [0.4068, 0.2544]);
        let _ = fs::remove_dir_all(dir);
    }
}

/// Guard: a millimetre sheet plotted 1 : 1 already has one drawing unit per paper millimetre,
/// so its linework is where it always was, whichever way the fix goes.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_millimetre_sheet_at_1_to_1_is_unchanged() {
    let mock = Mock {
        plot: plot(1, [279.4, 431.8], 1, [1.0, 1.0]),
        paper: vec![rectangle(0x10, [25.0, 25.0], [406.8, 254.4])],
        model: vec![],
        viewports: vec![],
    };
    let (dir, _) = convert(
        "mm-1to1",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the rectangle's lower left", min, [0.025, 0.025]);
    assert_close("the rectangle's upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

// ---- what else the fix has to get right ------------------------------------------------

/// The vertices of the first polyline as written, in the file's own unit.
fn first_polyline(dir: &Path) -> Vec<[f64; 2]> {
    let curves = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .entities()
        .unwrap();
    let Some((_, Curve::Polyline(p))) = curves.first() else {
        panic!("a polyline, not {curves:?}");
    };
    p.vertices().iter().map(|v| [v.x, v.y]).collect()
}

/// A viewport on the sheet showing model space 1 : 1 about its own centre.
fn window(handle: u64, centre: [f64; 2], size: [f64; 2]) -> ViewportEntity {
    let (hw, hh) = (size[0] / 2.0, size[1] / 2.0);
    ViewportEntity {
        handle: ItemHandle::new(handle),
        layout: LayoutId(1),
        id: 2,
        visible: true,
        clip: ViewportClip::Rectangle([
            [centre[0] - hw, centre[1] - hh],
            [centre[0] + hw, centre[1] - hh],
            [centre[0] + hw, centre[1] + hh],
            [centre[0] - hw, centre[1] + hh],
        ]),
        frozen_layers: Vec::new(),
        layer: "0".into(),
        center: centre,
        width: size[0],
        height: size[1],
        view_center: [0.0, 0.0],
        view_height: size[1],
        view_target: [0.0, 0.0, 0.0],
        view_direction: [0.0, 0.0, 1.0],
        twist: 0.0,
        perspective: false,
        hide_plot: false,
        non_rectangular_clipping: false,
    }
}

fn entity_line(handle: u64, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    RawEntity {
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
        style: EntityStyle::absent(),
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_viewport_and_what_it_shows_scale_together() {
    // The window is 10 by 6 units about (8, 5), so 250 by 150 mm about (200, 125) mm. The model
    // line is far longer than the window and must still be cut at its rectangle.
    let vp = RawEntity {
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(0x31)),
            class_name: "VIEWPORT".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
        style: EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: "a viewport is not geometry".into(),
        },
    };
    let mock = Mock {
        paper: vec![vp],
        model: vec![entity_line(0x40, [-100.0, 0.0], [100.0, 0.0])],
        viewports: vec![window(0x31, [8.0, 5.0], [10.0, 6.0])],
        ..tabloid()
    };
    let (dir, _) = convert(
        "viewport",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the line's left end, at the rectangle", min, [0.075, 0.125]);
    assert_close(
        "the line's right end, at the rectangle",
        max,
        [0.325, 0.125],
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_lineweight_is_not_scaled() {
    // A lineweight is a paper width (P-8): the stream carries the same width at 1 : 1 and at
    // 1 : 0.04.
    let widths = |tag: &str, scale: [f64; 2]| {
        let mut mock = tabloid();
        mock.plot.scale_numerator = scale[0];
        mock.plot.scale_denominator = scale[1];
        mock.paper = vec![{
            let mut l = entity_line(0x10, [1.0, 1.0], [5.0, 5.0]);
            l.style.lineweight = SourceFact::Present(SourceLineweight::Hundredths(50));
            l
        }];
        let (dir, _) = convert(tag, mock, SourceUnit::Millimeters, PlotSettings::default());
        let (streams, _) = open_published(&dir.join("sheet.pvec"))
            .unwrap()
            .streams()
            .unwrap();
        let stream: &Stream = streams
            .iter()
            .find(|s| s.capability == LineStyleStream::CAPABILITY)
            .expect("a line-style stream");
        let styles = LineStyleStream::decode(&stream.payload).unwrap().styles;
        let widths: Vec<String> = styles.iter().map(|s| format!("{:?}", s.width)).collect();
        let _ = fs::remove_dir_all(dir);
        widths
    };
    let plotted = widths("lw-scaled", [1.0, 0.04]);
    assert_eq!(plotted, widths("lw-unit", [1.0, 1.0]));
    assert!(plotted.iter().all(|w| w != "None"), "{plotted:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_one_to_one_sheet_is_bit_identical() {
    // Awkward coordinates, written at 1 : 1 in millimetres: not one bit moves.
    let (lo, hi) = ([0.1 + 0.2, 1.0 / 3.0], [123.456_789_012_345, 200.0 / 7.0]);
    let mock = Mock {
        plot: plot(1, [279.4, 431.8], 0, [1.0, 1.0]),
        paper: vec![rectangle(0x10, lo, hi)],
        model: vec![],
        viewports: vec![],
    };
    let (dir, _) = convert(
        "bits",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let got = first_polyline(&dir);
    let want = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
    for (g, w) in got.iter().zip(want) {
        assert_eq!(
            [g[0].to_bits(), g[1].to_bits()],
            [w[0].to_bits(), w[1].to_bits()],
            "{g:?} against {w:?}"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_an_upside_down_sheet_turns_about_the_scaled_frame() {
    // Rotation 2 keeps the sheet 279.4 by 431.8 mm. The rectangle runs (50, 75) to (125, 150) mm
    // scaled, so turned 180 degrees about the frame's centre it runs (154.4, 281.8) to
    // (229.4, 356.8).
    let mock = Mock {
        plot: plot(1, [279.4, 431.8], 2, [1.0, 0.04]),
        paper: vec![rectangle(0x10, [2.0, 3.0], [5.0, 6.0])],
        model: vec![],
        viewports: vec![],
    };
    let (dir, _) = convert(
        "flip",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.2794, 0.4318]);
    assert_close("the lower left", min, [0.1544, 0.2818]);
    assert_close("the upper right", max, [0.2294, 0.3568]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_plot_window_is_scaled_with_the_sheet() {
    // Plot type 4 prints the window (2, 3) to (5, 6) units: a 75 mm square, whose lower left is
    // (50, 75) mm on the scaled sheet. The rectangle (3, 4) to (4, 5) sits 25 mm in from it.
    let mut p = plot(1, [279.4, 431.8], 0, [1.0, 0.04]);
    p.plot_type = 4;
    p.window = [2.0, 3.0, 5.0, 6.0];
    let mock = Mock {
        plot: p,
        paper: vec![rectangle(0x10, [3.0, 4.0], [4.0, 5.0])],
        model: vec![],
        viewports: vec![],
    };
    let (dir, _) = convert(
        "window",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.075, 0.075]);
    assert_close("the lower left", min, [0.025, 0.025]);
    assert_close("the upper right", max, [0.05, 0.05]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_the_record_still_states_the_plot_scale() {
    // 1 mm of paper to 0.04 units is 25 : 1, and with the linework in paper millimetres the
    // record is true as it stands: a paper millimetre is 0.04 mm of the drawing.
    let (dir, sheet) = convert(
        "record",
        tabloid(),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    assert_eq!(sheet.scale.as_deref(), Some("25:1"));
    let claim = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
        .expect("a scale record");
    assert_eq!(
        claim.provenance,
        Provenance::PlanScale {
            numerator: 25,
            denominator: 1,
            paper_metres_per_unit: 0.001,
        }
    );
    assert!(
        sheet
            .notes
            .iter()
            .any(|n| n.starts_with("plot-scale-applied")),
        "{:?}",
        sheet.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_paper_units_that_are_not_a_length_are_left_alone() {
    // Paper units 2 are pixels: the scale says nothing about millimetres, so the linework stays
    // as it was and the sheet says it did not apply the scale.
    let mock = Mock {
        plot: plot(2, [279.4, 431.8], 0, [1.0, 0.04]),
        paper: vec![rectangle(0x10, [1.0, 1.0], [5.0, 5.0])],
        model: vec![],
        viewports: vec![],
    };
    let (dir, sheet) = convert(
        "pixels",
        mock,
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (_, min, max) = measured(&dir);
    assert_close("the lower left", min, [0.001, 0.001]);
    assert_close("the upper right", max, [0.005, 0.005]);
    assert!(
        sheet.notes.iter().any(|n| n == "plot-scale-not-applied"),
        "{:?}",
        sheet.notes
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_callers_plot_scale_is_applied_like_the_stored_one() {
    // The layout stores 1 : 1; the caller plots 1 mm to 0.04 units, or 1 in to 1.016 units, which
    // is 25 mm a unit as well.
    for (tag, scale) in [
        (
            "mm",
            PlotScale {
                paper: 1.0,
                paper_unit: SourceUnit::Millimeters,
                drawing: 0.04,
            },
        ),
        (
            "in",
            PlotScale {
                paper: 1.0,
                paper_unit: SourceUnit::Inches,
                drawing: 1.016,
            },
        ),
    ] {
        let mock = Mock {
            plot: plot(1, [279.4, 431.8], 1, [1.0, 1.0]),
            ..tabloid()
        };
        let settings = PlotSettings {
            scale: Some(scale),
            ..PlotSettings::default()
        };
        let (dir, _) = convert(
            &format!("caller-scale-{tag}"),
            mock,
            SourceUnit::Millimeters,
            settings,
        );
        let (frame, min, max) = measured(&dir);
        assert_close(&format!("{tag}: the frame"), frame, [0.4318, 0.2794]);
        assert_close(&format!("{tag}: the lower left"), min, [0.025, 0.025]);
        assert_close(&format!("{tag}: the upper right"), max, [0.4068, 0.2544]);
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_fitted_sheet_does_not_take_the_stored_scale_twice() {
    // Fit to paper works out its own ratio over the plot area, so the stored 1 : 0.04 must not
    // also move the linework: a fit that would be 1 : 1 stays 1 : 1.
    let mock = Mock {
        plot: plot(1, [279.4, 431.8], 0, [1.0, 0.04]),
        paper: vec![rectangle(0x10, [0.0, 0.0], [279.4, 431.8])],
        model: vec![],
        viewports: vec![],
    };
    let settings = PlotSettings {
        area: PlotArea::Extents,
        fit_to_paper: true,
        ..PlotSettings::default()
    };
    let (dir, _) = convert("fit", mock, SourceUnit::Millimeters, settings);
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.2794, 0.4318]);
    assert_close("the lower left", min, [0.0, 0.0]);
    assert_close("the upper right", max, [0.2794, 0.4318]);
    let _ = fs::remove_dir_all(dir);
}

/// The same Tabloid sheet written as a real DWG with acadrust and read back by the default
/// provider, so the plot scale makes it through the decoder as well as the converter.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_real_tabloid_layout_draws_at_full_paper_size() {
    use acadrust::entities::Line;
    use acadrust::objects::ObjectType;
    use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};

    let mut doc = CadDocument::new();
    let layout = doc.add_layout("2.0 - L0").expect("layout");
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ANSI_B_(11.00_x_17.00_Inches)".into();
    l.paper_width = 279.4;
    l.paper_height = 431.8;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = 1.0;
    l.plot_scale_denominator = 0.04;
    l.plot_rotation = 1;
    l.plot_margin_left = 0.0;
    l.plot_margin_bottom = 0.0;
    l.plot_origin_x = 0.0;
    l.plot_origin_y = 0.0;
    l.plot_flags.print_lineweights = true;
    doc.add_entity_to_layout(
        EntityType::Line(Line::from_points(
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(16.272, 10.176, 0.0),
        )),
        "2.0 - L0",
    )
    .expect("entity");
    let mut bytes = Vec::new();
    DwgWriter::write_to_writer(std::io::Cursor::new(&mut bytes), &doc).expect("write");

    let dir = scratch("real");
    PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("sheet.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("2.0 - L0"))
        .convert()
        .expect("the sheet converts");
    let (frame, min, max) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the line's lower left", min, [0.025, 0.025]);
    assert_close("the line's upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}
