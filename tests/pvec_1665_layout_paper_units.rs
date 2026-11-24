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
    let d = std::env::temp_dir().join(format!("libviprs-1665-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts `mock` under the print profile of `Sheet`, the caller saying `unit` and asking for
/// `settings`.
fn convert(tag: &str, mock: Mock, unit: SourceUnit, settings: PlotSettings) -> PathBuf {
    let dir = scratch(tag);
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    PvecConversion::new(CadSource::Bytes(b"mock"), &dir.join("sheet.pvec"))
        .unit(unit)
        .plot_settings(settings)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .convert_with::<Mock>()
        .expect("the sheet converts");
    dir
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

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_millimetre_sheet_at_1_to_0_04_draws_at_full_paper_size() {
    let dir = convert(
        "tabloid",
        tabloid(),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max, mpu) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    // One drawing unit is 25 mm of paper: the rectangle runs from 25 mm in to 25 mm short of
    // the far edges, not 17 mm across in the corner.
    assert_close("the rectangle's lower left", min, [0.025, 0.025]);
    assert_close("the rectangle's upper right", max, [0.4068, 0.2544]);
    // The plot scale stays the calibration record, and a length measured through it is what it
    // was: the rectangle is 15.272 by 9.176 drawing units, millimetres here, whatever unit its
    // coordinates are now in.
    let per_unit = stored_scale(&dir).metres_per_source_unit;
    let across = [(max[0] - min[0]) / mpu, (max[1] - min[1]) / mpu];
    assert_close(
        "the rectangle measured through the calibration",
        [across[0] * per_unit, across[1] * per_unit],
        [0.015272, 0.009176],
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_an_inch_sheet_at_1_to_1_016_draws_at_full_paper_size() {
    let dir = convert(
        "letter",
        letter(),
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max, _) = measured(&dir);
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
        let dir = convert(
            &format!("caller-{tag}"),
            tabloid(),
            unit,
            PlotSettings::default(),
        );
        let (frame, min, max, _) = measured(&dir);
        assert_close(&format!("{tag}: the frame"), frame, [0.4318, 0.2794]);
        assert_close(&format!("{tag}: the lower left"), min, [0.025, 0.025]);
        assert_close(&format!("{tag}: the upper right"), max, [0.4068, 0.2544]);
        let _ = fs::remove_dir_all(dir);
    }
}

/// A caller's plot scale is applied the way the layout's own is: the Tabloid sheet stored at 1:1
/// and plotted at 1 mm to 0.04 units draws the same as the one stored at 1 : 0.04.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_callers_plot_scale_is_applied_like_the_stored_one() {
    let mut mock = tabloid();
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
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the rectangle's lower left", min, [0.025, 0.025]);
    assert_close("the rectangle's upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}

/// A plot window is picked in paper-space drawing units, so at 1 : 0.04 it frames 25 mm of paper
/// per unit, the linework with it: a window half a unit outside the rectangle all round is a frame
/// 406.8 by 254.4 mm with the rectangle 12.5 mm in from its edges.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_a_plot_window_is_scaled_with_the_linework() {
    let dir = convert(
        "window",
        tabloid(),
        SourceUnit::Millimeters,
        PlotSettings {
            area: PlotArea::Window([0.5, 0.5, 16.772, 10.676]),
            ..PlotSettings::default()
        },
    );
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4068, 0.2544]);
    assert_close("the rectangle's lower left", min, [0.0125, 0.0125]);
    assert_close("the rectangle's upper right", max, [0.3943, 0.2419]);
    let _ = fs::remove_dir_all(dir);
}

/// Plotting upside-down at 1 : 0.04: the linework is drawn at 25 mm a unit, then turned 180
/// degrees about the frame's centre. A rectangle 25 mm in from the lower left, 225 by 100 mm,
/// lands 25 mm in from the upper right. Written after the fix, as a guard on the order of the
/// scale and the turn.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1665_an_upside_down_sheet_at_1_to_0_04_is_scaled_then_turned() {
    let dir = convert(
        "flip",
        Mock {
            plot: plot(1, [279.4, 431.8], 3, [1.0, 0.04]),
            paper: vec![rectangle(0x10, [1.0, 1.0], [10.0, 5.0])],
        },
        SourceUnit::Millimeters,
        PlotSettings::default(),
    );
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the rectangle's lower left", min, [0.1818, 0.1544]);
    assert_close("the rectangle's upper right", max, [0.4068, 0.2544]);
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
    let (frame, min, max, _) = measured(&dir);
    assert_close("the frame", frame, [0.4318, 0.2794]);
    assert_close("the line's lower left", min, [0.025, 0.025]);
    assert_close("the line's upper right", max, [0.4068, 0.2544]);
    let _ = fs::remove_dir_all(dir);
}
