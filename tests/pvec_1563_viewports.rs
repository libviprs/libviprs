//! Layouts slice 4 (#1563): model space shows through each viewport of a sheet as it would print
//! (`docs/adr/pvec-print-profile.md`, P-1, P-3, P-4, P-7, P-9).
//!
//! A plan-view, untwisted viewport maps model space onto the sheet by a similarity, `paper =
//! centre + s * (model - target - view centre)` with `s = height / view height`, composed onto
//! each leaf's flatten map. What it shows is cut to its rectangle by the sheet's clipper, the
//! layers it freezes are dropped inside it, and a model entity seen through two viewports is two
//! entities with provenance kind 2 (source handle then viewport handle). A viewport that is off,
//! hidden when plotting or on a no-plot layer contributes nothing; a non-plan, perspective,
//! zero-height, twisted or non-rectangular one is refused with 508. The paper view itself (id 1,
//! showing paper space 1:1 centred on itself, as AutoCAD writes it) is not a window onto the model.
//!
//! Most of it runs on a hand-built provider so the maps and the cuts are pinned exactly; the rest
//! converts the vendored multi-viewport gold probe and `sample_AC1018`. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow, SheetViewport};
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact,
    SourceTransform, SourceUnit, ViewProfile, ViewportClip, ViewportEntity,
};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput, PvecConvertError};
use pvec_facade::Curve;
use pvec_facade::components::{ProvenanceStream, Stream};

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), with its viewports.
#[derive(Clone)]
struct Mock {
    plot: LayoutPlot,
    paper: Vec<RawEntity>,
    model: Vec<RawEntity>,
    viewports: Vec<ViewportEntity>,
    /// The one block definition, `BLK`.
    block: Vec<RawEntity>,
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
                provider: "mock-1563",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1563"
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
            layer("COLD", true),
            layer("WALLS", true),
        ])
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
            EntityScope::BlockDefinition { name, reached_via } if name == "BLK" => self
                .block
                .iter()
                .cloned()
                .map(|mut e| {
                    e.provenance.block_path = reached_via.clone();
                    e
                })
                .collect(),
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

fn line(handle: u64, layer: &str, a: [f64; 2], b: [f64; 2]) -> RawEntity {
    ent(
        handle,
        "LINE",
        layer,
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
        "0",
        RawGeometry::Circle {
            center: [centre[0], centre[1], 0.0],
            radius,
            normal: [0.0, 0.0, 1.0],
        },
    )
}

fn viewport_entity(handle: u64, layer: &str) -> RawEntity {
    ent(
        handle,
        "VIEWPORT",
        layer,
        RawGeometry::Unmapped {
            reason: "a viewport is not geometry".into(),
        },
    )
}

/// An A3 sheet in millimetres plotted 1:1 at the paper-space origin.
fn a3(draw_viewports_first: bool) -> LayoutPlot {
    LayoutPlot {
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
        draw_viewports_first,
    }
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
        layer: "VP".into(),
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

// The paper view, and two viewports onto the model: A at 10 paper units per model unit showing
// model x 0..10, y 1..9; B at 5 showing x -5..15, y -3..13.
const PAPER_VIEW: u64 = 0x30;
const VP_A: u64 = 0x31;
const VP_B: u64 = 0x32;
const HIDDEN: u64 = 0x33;
const OFF: u64 = 0x34;
const TWISTED: u64 = 0x35;
const PERSPECTIVE: u64 = 0x36;
const SIDE_VIEW: u64 = 0x37;
const FLAT: u64 = 0x38;
const POLYGON: u64 = 0x39;
const ROUND: u64 = 0x3A;
const NO_PLOT_VP: u64 = 0x3B;
const TARGETED: u64 = 0x3C;

const TITLE: u64 = 0x20;
const SHORT: u64 = 0x40;
const LONG: u64 = 0x41;
const COLD: u64 = 0x42;
const SMALL_RING: u64 = 0x43;
const EDGE_RING: u64 = 0x44;
const NO_PLOT_LINE: u64 = 0x45;
const FAR: u64 = 0x46;
const COLD_INSERT: u64 = 0x47;
const IN_BLOCK: u64 = 0x48;

fn viewports() -> Vec<ViewportEntity> {
    let elsewhere = |h: u64, id: i16| vp(h, id, [210.0, 250.0], [40.0, 30.0], [0.0, 0.0], 3.0);
    let mut a = vp(VP_A, 2, [100.0, 100.0], [100.0, 80.0], [5.0, 5.0], 8.0);
    a.frozen_layers = vec!["cold".into()];
    let mut hidden = elsewhere(HIDDEN, 4);
    hidden.hide_plot = true;
    let mut off = elsewhere(OFF, 5);
    off.visible = false;
    let mut twisted = elsewhere(TWISTED, 6);
    twisted.twist = 0.5;
    let mut perspective = elsewhere(PERSPECTIVE, 7);
    perspective.perspective = true;
    let mut side = elsewhere(SIDE_VIEW, 8);
    side.view_direction = [1.0, 0.0, 0.0];
    let mut flat = elsewhere(FLAT, 9);
    flat.view_height = 0.0;
    let mut polygon = elsewhere(POLYGON, 10);
    polygon.clip = ViewportClip::Polygon(
        [[190.0, 235.0], [230.0, 235.0], [210.0, 265.0]]
            .iter()
            .map(|p| RawVertex {
                x: p[0],
                y: p[1],
                bulge: 0.0,
            })
            .collect(),
    );
    polygon.non_rectangular_clipping = true;
    let mut round = elsewhere(ROUND, 11);
    round.clip = ViewportClip::Circle(ItemHandle::new(0x99));
    round.non_rectangular_clipping = true;
    let mut no_plot = elsewhere(NO_PLOT_VP, 12);
    no_plot.layer = "Defpoints".into();
    let mut targeted = elsewhere(TARGETED, 13);
    targeted.view_target = [1.0, 1.0, 0.0];
    vec![
        vp(
            PAPER_VIEW,
            1,
            [210.0, 148.5],
            [420.0, 297.0],
            [210.0, 148.5],
            297.0,
        ),
        a,
        vp(VP_B, 3, [300.0, 100.0], [100.0, 80.0], [5.0, 5.0], 16.0),
        hidden,
        off,
        twisted,
        perspective,
        side,
        flat,
        polygon,
        round,
        no_plot,
        targeted,
    ]
}

fn drawing(draw_viewports_first: bool) -> Mock {
    let mut paper = vec![line(TITLE, "0", [10.0, 10.0], [410.0, 10.0])];
    for v in viewports() {
        paper.push(viewport_entity(v.handle.get(), &v.layer));
    }
    Mock {
        plot: a3(draw_viewports_first),
        paper,
        model: vec![
            line(SHORT, "0", [2.0, 5.0], [8.0, 5.0]),
            line(LONG, "0", [-20.0, 7.0], [20.0, 7.0]),
            line(COLD, "COLD", [3.0, 3.0], [4.0, 3.0]),
            circle(SMALL_RING, [5.0, 5.0], 1.0),
            circle(EDGE_RING, [10.0, 5.0], 2.0),
            line(NO_PLOT_LINE, "Defpoints", [1.0, 1.0], [2.0, 2.0]),
            line(FAR, "0", [100.0, 100.0], [101.0, 100.0]),
            ent(
                COLD_INSERT,
                "INSERT",
                "COLD",
                RawGeometry::Insert(BlockRef {
                    block_name: "BLK".into(),
                    transform: SourceTransform::IDENTITY,
                    normal: [0.0, 0.0, 1.0],
                    grid: ArrayGrid::SINGLE,
                }),
            ),
        ],
        viewports: viewports(),
        block: vec![line(IN_BLOCK, "WALLS", [3.0, 6.0], [4.0, 6.0])],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1563-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

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

/// The rows of `handle` seen through `viewport` (`None`: the sheet's own entities).
fn rows(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle) && r.viewport == viewport)
        .collect()
}

fn row(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> &EntityRow {
    let r = rows(report, handle, viewport);
    assert_eq!(
        r.len(),
        1,
        "one row for {handle:#X} through {viewport:X?}: {r:#?}"
    );
    r[0]
}

/// The written curves, each with the row it came from: the artifact's entities are the rows that
/// carry geometry, in id order.
fn written(report: &ConversionReport, dir: &Path) -> Vec<(EntityRow, Curve)> {
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
        .collect()
}

fn curve_of(report: &ConversionReport, dir: &Path, handle: u64, viewport: u64) -> Curve {
    let mut found: Vec<Curve> = written(report, dir)
        .into_iter()
        .filter(|(r, _)| r.source_handle == Some(handle) && r.viewport == Some(viewport))
        .map(|(_, c)| c)
        .collect();
    assert_eq!(found.len(), 1, "{handle:#X} through {viewport:#X}");
    found.remove(0)
}

/// A trimmed line as its two ends.
fn ends(c: &Curve) -> [[f64; 2]; 2] {
    let Curve::Trimmed(t) = c else {
        panic!("a trimmed line, not {c:?}");
    };
    let Curve::Line(l) = t.basis() else {
        panic!("a trimmed line, not {c:?}");
    };
    let (o, d, s) = (l.origin(), l.direction(), t.trim().span);
    [[o.x, o.y], [o.x + d.x * s, o.y + d.y * s]]
}

fn near(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| (x - y).abs() <= 1e-9 * (1.0 + y.abs()))
}

fn sheet_row(report: &ConversionReport, handle: u64) -> &SheetViewport {
    let s = report.sheet.as_ref().expect("a sheet section");
    s.viewports
        .iter()
        .find(|v| v.handle == handle)
        .unwrap_or_else(|| panic!("no sheet row for {handle:#X}: {:#?}", s.viewports))
}

// ---- placement --------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_a_model_line_lands_on_the_sheet_through_each_viewport_that_shows_it() {
    let (out, dir) = convert("place", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    // A: 100 + 10 * (x - 5), 100 + 10 * (y - 5). B: 300 + 5 * (x - 5), 100 + 5 * (y - 5).
    assert_eq!(
        ends(&curve_of(&report, &dir, SHORT, VP_A)),
        [[70.0, 100.0], [130.0, 100.0]]
    );
    assert_eq!(
        ends(&curve_of(&report, &dir, SHORT, VP_B)),
        [[285.0, 100.0], [315.0, 100.0]]
    );
    assert_eq!(
        row(&report, SHORT, Some(VP_A)).fidelity,
        row(&report, TITLE, None).fidelity,
        "a line through a viewport classifies as a line on the sheet"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_what_a_viewport_shows_is_cut_exactly_at_its_rectangle() {
    let (out, dir) = convert("cut", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    assert!(near(
        ends(&curve_of(&report, &dir, LONG, VP_A)),
        [[50.0, 120.0], [150.0, 120.0]]
    ));
    assert!(near(
        ends(&curve_of(&report, &dir, LONG, VP_B)),
        [[250.0, 110.0], [350.0, 110.0]]
    ));
    // Wholly outside both windows: dropped in each, never drawn.
    for v in [VP_A, VP_B] {
        let r = row(&report, FAR, Some(v));
        assert_eq!(r.fidelity, "IgnoredByPolicy");
        assert_eq!(r.policy_field.as_deref(), Some("viewport_clipping"));
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_a_circle_is_scaled_exactly_through_each_viewport() {
    let (out, dir) = convert("ring", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    let Curve::Circle(a) = curve_of(&report, &dir, SMALL_RING, VP_A) else {
        panic!("a circle through A");
    };
    let o = a.placement().origin();
    assert_eq!(([o.x, o.y], a.radius()), ([100.0, 100.0], 10.0));
    let Curve::Circle(b) = curve_of(&report, &dir, SMALL_RING, VP_B) else {
        panic!("a circle through B");
    };
    let o = b.placement().origin();
    assert_eq!(([o.x, o.y], b.radius()), ([300.0, 100.0], 5.0));
    // Model x 8..12 crosses A's right edge at x 10, which is cut since #1565; B shows all of it.
    assert_eq!(
        row(&report, EDGE_RING, Some(VP_B)).fidelity,
        "SupportedExact"
    );
    let _ = fs::remove_dir_all(dir);
}

// ---- visibility -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_a_layer_frozen_in_a_viewport_is_dropped_there_and_drawn_elsewhere() {
    let (out, dir) = convert("freeze", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    let a = row(&report, COLD, Some(VP_A));
    assert_eq!(a.fidelity, "IgnoredByPolicy");
    assert_eq!(a.policy_field.as_deref(), Some("layers.frozen_in_viewport"));
    assert_eq!(
        ends(&curve_of(&report, &dir, COLD, VP_B)),
        [[290.0, 90.0], [295.0, 90.0]]
    );
    // A no-plot layer prints nowhere, and is counted once, not once per viewport.
    let no_plot: Vec<&EntityRow> = report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(NO_PLOT_LINE))
        .collect();
    assert_eq!(no_plot.len(), 1, "{no_plot:#?}");
    assert_eq!(no_plot[0].policy_field.as_deref(), Some("layers.no_plot"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_an_insert_on_a_layer_frozen_in_a_viewport_takes_its_whole_block_with_it() {
    let (out, dir) = convert("insert-freeze", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    // The line inside is on WALLS, but its INSERT is on COLD, which A freezes.
    let a = row(&report, IN_BLOCK, Some(VP_A));
    assert_eq!(a.fidelity, "IgnoredByPolicy", "{a:?}");
    assert_eq!(a.policy_field.as_deref(), Some("layers.frozen_in_viewport"));
    assert_eq!(
        ends(&curve_of(&report, &dir, IN_BLOCK, VP_B)),
        [[290.0, 105.0], [295.0, 105.0]]
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_each_viewport_says_what_it_did_and_the_refused_ones_say_why() {
    let (out, dir) = convert("rows", drawing(false), |c| c);
    let report = out.expect("the sheet converts").report;
    let status = |h: u64| {
        let r = sheet_row(&report, h);
        (r.status.as_str(), r.reason.as_deref())
    };
    assert_eq!(status(PAPER_VIEW), ("paper-view", None));
    assert_eq!(status(VP_A), ("drawn", None));
    assert_eq!(status(VP_B), ("drawn", None));
    assert_eq!(status(HIDDEN), ("hide-plot", None));
    assert_eq!(status(OFF), ("off", None));
    assert_eq!(status(NO_PLOT_VP), ("no-plot-layer", None));
    assert_eq!(status(TWISTED), ("refused", Some("twist")));
    assert_eq!(status(PERSPECTIVE), ("refused", Some("perspective")));
    assert_eq!(status(SIDE_VIEW), ("refused", Some("non-plan")));
    assert_eq!(status(FLAT), ("refused", Some("view-height")));
    // A straight-edged polygon clip is drawn since #1565.
    assert_eq!(status(POLYGON), ("drawn", None));
    assert_eq!(status(ROUND), ("refused", Some("clip-circle")));
    // A target off the origin moves the view in a way no reference print has measured.
    assert_eq!(status(TARGETED), ("refused", Some("target")));
    // The scale is one paper unit to so many model units.
    assert_eq!(sheet_row(&report, VP_A).scale.as_deref(), Some("1:0.1"));
    assert_eq!(sheet_row(&report, VP_B).scale.as_deref(), Some("1:0.2"));
    // A refused viewport's own entity is refused with 508; nothing it shows is drawn.
    let twisted = row(&report, TWISTED, None);
    assert_eq!(twisted.reason.as_deref(), Some("view-not-enabled"));
    assert!(rows(&report, SHORT, Some(TWISTED)).is_empty());
    assert!(rows(&report, SHORT, Some(HIDDEN)).is_empty());
    assert!(rows(&report, SHORT, Some(PAPER_VIEW)).is_empty());
    // The paper view is kept as a row under its own code, never dropped silently.
    let paper = row(&report, PAPER_VIEW, None);
    assert!(
        paper.warnings.iter().any(|w| w.starts_with("PAPER_VIEW")),
        "{paper:?}"
    );
    // A viewport that is drawn is not refused any more.
    assert_ne!(row(&report, VP_A, None).fidelity, "Unsupported");
    let _ = fs::remove_dir_all(dir);
}

// ---- provenance and order ---------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_one_model_entity_through_two_viewports_has_two_provenance_rows_of_kind_2() {
    let (out, dir) = convert("prov", drawing(false), |c| {
        c.components(StreamSelection {
            provenance: true,
            ..StreamSelection::NONE
        })
    });
    out.expect("the sheet converts");
    let (streams, _) = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .streams()
        .unwrap();
    let stream: &Stream = streams
        .iter()
        .find(|s| s.capability == ProvenanceStream::CAPABILITY)
        .expect("a provenance stream");
    let prov = ProvenanceStream::decode(&stream.payload).unwrap();
    let through = |viewport: u64| {
        let mut h = SHORT.to_be_bytes().to_vec();
        h.extend_from_slice(&viewport.to_be_bytes());
        h
    };
    for v in [VP_A, VP_B] {
        assert!(
            prov.records
                .iter()
                .any(|r| r.source_kind == 2 && r.handle == through(v)),
            "{v:#X}: {:?}",
            prov.records
        );
    }
    // The sheet's own line stays a kind 1 row.
    assert!(
        prov.records
            .iter()
            .any(|r| r.source_kind == 1 && r.handle == TITLE.to_be_bytes().to_vec())
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_draw_viewports_first_puts_what_they_show_under_the_sheet() {
    for first in [false, true] {
        let (out, dir) = convert("order", drawing(first), |c| c);
        let report = out.expect("the sheet converts").report;
        let title = row(&report, TITLE, None).id;
        let model = row(&report, SHORT, Some(VP_A)).id;
        assert_eq!(model < title, first, "draw_viewports_first = {first}");
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1563_one_viewport_onto_the_model_puts_its_scale_in_the_calibration_note() {
    let mut mock = drawing(false);
    // B, and the polygon viewport, which is drawn since #1565.
    mock.viewports
        .retain(|v| ![VP_B, POLYGON].contains(&v.handle.get()));
    mock.paper.retain(|e| {
        ![VP_B, POLYGON]
            .iter()
            .any(|h| e.provenance.handle == Some(ItemHandle::new(*h)))
    });
    let (out, dir) = convert("note", mock, |c| c);
    out.expect("the sheet converts");
    let claim = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
        .expect("the plot scale");
    assert_eq!(claim.note, "plot scale 1:1; viewport 1:0.1");
    let _ = fs::remove_dir_all(dir);
}

// ---- real drawings ----------------------------------------------------------------------

#[cfg(feature = "dwg-acadrust")]
mod with_a_drawing {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1563_autocads_paper_view_is_told_apart_from_the_viewport_onto_the_model() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/sample_AC1018.dwg");
        for layout in ["Layout1", "Layout2", "MyLayout"] {
            let dir = scratch(&format!("sample-{layout}"));
            let out = PvecConversion::new(CadSource::Path(&path), &dir.join("sheet.pvec"))
                .view_profile(ViewProfile::print_profile(layout))
                .convert()
                .unwrap_or_else(|e| panic!("{layout}: {e:?}"));
            let sheet = out.report.sheet.as_ref().expect("a sheet section");
            let mut statuses: Vec<(i16, &str)> = sheet
                .viewports
                .iter()
                .map(|v| (v.id, v.status.as_str()))
                .collect();
            statuses.sort_unstable();
            assert_eq!(statuses, [(1, "paper-view"), (2, "drawn")], "{layout}");
            let _ = fs::remove_dir_all(dir);
        }
    }
}
