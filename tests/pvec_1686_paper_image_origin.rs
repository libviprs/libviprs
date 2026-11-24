//! A layout's paper image origin moves its sheet on the plot (#1686).
//!
//! A layout stores a paper image origin (DXF groups 148 and 149, in paper-space drawing units)
//! and limits. A sheet converted at 1 : 0.04, 25 mm of paper to the unit, with an image origin of
//! (0, -0.0266) sat 0.665 mm higher on its frame than the plotted sheet: the converter placed the
//! paper's lower left at -(margin + plot origin) and left the image origin out. Every cell here
//! measures in metres from the frame's lower-left corner, as the 1665 cells do, on small
//! synthetic sheets. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::open_published;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact, SourceUnit,
    ViewProfile,
};
use libviprs::pvec_convert::{PlotArea, PlotScale, PlotSettings, PvecConversion};
use pvec_facade::Curve;

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), whose plot settings are
/// `plot` and whose own entities are `paper`.
#[derive(Clone)]
struct Mock {
    plot: LayoutPlot,
    paper: Vec<RawEntity>,
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
                provider: "mock-1686",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1686"
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
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => vec![],
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
        paper_image_origin: [0.0; 2],
        limits: [0.0; 4],
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
    }
}

/// The "Letter - P" sheet: 215.9 by 279.4 mm in inches, 1 in to 1.016 units, so one drawing
/// unit is 25 mm here too and the sheet is 8.636 by 11.176 units. The rectangle sits one unit
/// in from every edge.
fn letter() -> Mock {
    Mock {
        plot: plot(0, [215.9, 279.4], 0, [1.0, 1.016]),
        paper: vec![rectangle(0x10, [1.0, 1.0], [7.636, 10.176])],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1686-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts `mock` under the print profile of `Sheet`, the caller saying `unit` and asking for
/// `settings`.
fn convert(tag: &str, mock: Mock, unit: SourceUnit, settings: PlotSettings) -> PathBuf {
    converted(tag, mock, unit, settings).0
}

/// As [`convert`], with the sheet's report row as well.
fn converted(
    tag: &str,
    mock: Mock,
    unit: SourceUnit,
    settings: PlotSettings,
) -> (PathBuf, libviprs::pvec::report::SheetSummary) {
    let dir = scratch(tag);
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let out = PvecConversion::new(CadSource::Bytes(b"mock"), &dir.join("sheet.pvec"))
        .unit(unit)
        .plot_settings(settings)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .convert_with::<Mock>()
        .expect("the sheet converts");
    let sheet = out.report.sheet.expect("a sheet row");
    (dir, sheet)
}

/// The frame's size and the drawn linework's extent, both in metres, the extent measured from
/// the frame's lower-left corner, and the manifest's metres per unit.
fn measured(dir: &Path) -> ([f64; 2], [f64; 2], [f64; 2], f64) {
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
    (frame, min, max, mpu)
}

/// The plot scale as the artifact's calibration record.
fn stored_scale(dir: &Path) -> pvec_facade::unstable::ScaleClaim {
    open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
        .expect("the plot scale is the calibration record")
}

fn assert_close(what: &str, got: [f64; 2], want: [f64; 2]) {
    let off = (got[0] - want[0]).abs().max((got[1] - want[1]).abs());
    assert!(off < 1e-9, "{what}: {got:?} m, expected {want:?} m");
}

// ---- the cells -------------------------------------------------------------------------

/// 0.0266 drawing units at 25 mm a unit.
const SHIFT: f64 = 0.000665;

fn with_origin(mut mock: Mock, origin: [f64; 2]) -> Mock {
    mock.plot.paper_image_origin = origin;
    mock
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_an_image_origin_at_1_to_0_04_moves_the_sheet_by_its_paper_length() {
    let dir = convert(
        "y",
        with_origin(tabloid(), [0.0, -0.0266]),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    // The frame moves up 0.665 mm, so the linework sits that much lower on it, x untouched.
    assert_close("the lower left", min, [0.025, 0.025 - SHIFT]);
    assert_close("the upper right", max, [0.4068, 0.2544 - SHIFT]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_the_x_of_the_image_origin_moves_the_sheet_the_same_way() {
    let dir = convert(
        "x",
        with_origin(tabloid(), [0.0266, 0.0]),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (_, min, max, _) = measured(&dir);
    assert_close("the lower left", min, [0.025 + SHIFT, 0.025]);
    assert_close("the upper right", max, [0.4068 + SHIFT, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_a_layout_with_no_image_origin_does_not_move() {
    let dir = convert(
        "zero",
        with_origin(tabloid(), [0.0, 0.0]),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (_, min, max, _) = measured(&dir);
    assert_close("the lower left", min, [0.025, 0.025]);
    assert_close("the upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

/// The limits are read, and the image origin is what places the sheet: limits alone move nothing.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_limits_without_an_image_origin_do_not_move_the_sheet() {
    let mut mock = tabloid();
    mock.plot.limits = [0.0, 0.0266, 17.272, 11.2026];
    let dir = convert("limits", mock, SourceUnit::Millimeters, PlotSettings::default());
    let (_, min, max, _) = measured(&dir);
    assert_close("the lower left", min, [0.025, 0.025]);
    assert_close("the upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

/// An inch sheet at 1 in : 1.016 units is 25 mm a unit as well, so the same 0.665 mm.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_an_inch_sheet_moves_by_the_same_paper_length() {
    let dir = convert(
        "inch",
        with_origin(letter(), [0.0, -0.0266]),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.2159, 0.2794]);
    assert_close("the lower left", min, [0.025, 0.025 - SHIFT]);
    assert_close("the upper right", max, [0.1909, 0.2544 - SHIFT]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_the_callers_unit_does_not_change_the_shift() {
    for (tag, unit) in [
        ("m", SourceUnit::Meters),
        ("mm", SourceUnit::Millimeters),
        ("in", SourceUnit::Inches),
    ] {
        let dir = convert(
            &format!("caller-{tag}"),
            with_origin(tabloid(), [0.0, -0.0266]),
            unit,
            PlotSettings::default(),
        );
        let (_, min, max, _) = measured(&dir);
        assert_close(&format!("{tag}: lower left"), min, [0.025, 0.025 - SHIFT]);
        assert_close(&format!("{tag}: upper right"), max, [0.4068, 0.2544 - SHIFT]);
        let _ = fs::remove_dir_all(dir);
    }
}

/// The caller's plot scale is the scale the origin is carried at, like the linework.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_a_callers_plot_scale_carries_the_origin_too() {
    let mut mock = with_origin(tabloid(), [0.0, -0.0266]);
    mock.plot.scale_denominator = 1.0;
    let dir = convert(
        "caller-scale",
        mock,
        SourceUnit::Millimeters,
        PlotSettings {
            scale: Some(PlotScale {
                paper: 1.0,
                paper_unit: SourceUnit::Millimeters,
                drawing: 0.04,
            }),
            ..PlotSettings::default()
        },
    );
    let (_, min, _, _) = measured(&dir);
    assert_close("the lower left", min, [0.025, 0.025 - SHIFT]);
    let _ = fs::remove_dir_all(dir);
}

/// A sheet with no margins and plot origin is the measured case, the report says so, and it says
/// the image origin was applied.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_the_position_is_reported_measured_for_the_measured_case() {
    let (dir, row) = converted(
        "report",
        with_origin(tabloid(), [0.0, -0.0266]),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    assert_eq!(row.frame, "paper");
    assert_eq!(row.position.as_deref(), Some("measured"));
    assert!(
        row.notes.iter().any(|n| n == "paper-image-origin"),
        "{:?}",
        row.notes
    );
    let _ = fs::remove_dir_all(dir);
}

/// With a margin or a plot origin the placement is still the formula no plot has measured.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_a_margin_keeps_the_position_unmeasured() {
    let mut mock = tabloid();
    mock.plot.margins = [5.0, 6.0, 7.0, 8.0];
    let (dir, row) = converted("margin", mock, SourceUnit::Millimeters, PlotSettings::default());
    assert_eq!(row.position.as_deref(), Some("unmeasured"));
    let _ = fs::remove_dir_all(dir);
}

/// A layout with no image origin has no note about one.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_no_image_origin_no_note() {
    let (dir, row) = converted("nonote", tabloid(), SourceUnit::Millimeters, PlotSettings::default());
    assert!(!row.notes.iter().any(|n| n == "paper-image-origin"));
    let _ = fs::remove_dir_all(dir);
}

/// The same sheet written as a real DWG with acadrust: the provider reads groups 148 and 149 and
/// the limits, and the converter places the sheet by them.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1686_a_real_layout_with_an_image_origin_sits_on_its_paper() {
    use acadrust::entities::Line;
    use acadrust::objects::ObjectType;
    use acadrust::{CadDocument, DwgWriter, EntityType, Vector3};
    use libviprs::pvec::SourceFact;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;

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
    l.paper_image_origin_x = 0.0;
    l.paper_image_origin_y = -0.0266;
    l.min_limits = (0.0, 0.0266);
    l.max_limits = (17.272, 11.2026);
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

    let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).expect("opens");
    let sheet = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.name == "2.0 - L0")
        .expect("the layout");
    let SourceFact::Present(plot) = p.layout_plot(sheet.id).expect("reported") else {
        panic!("plot settings");
    };
    assert_eq!(plot.paper_image_origin, [0.0, -0.0266]);
    assert_eq!(plot.limits, [0.0, 0.0266, 17.272, 11.2026]);

    let dir = scratch("real");
    PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("sheet.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("2.0 - L0"))
        .convert()
        .expect("the sheet converts");
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the line's lower left", min, [0.025, 0.025 - SHIFT]);
    assert_close("the line's upper right", max, [0.4068, 0.2544 - SHIFT]);
    let _ = fs::remove_dir_all(dir);
}
