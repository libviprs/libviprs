//! Issue #1627 (C1.2 of iasbuilt/webapp #1434): layer overrides are evaluated before the file's own
//! flags (spec section 2.4, tests 9.7 and 9.8, and Rom's answers 2 and 3 of section 12).
//!
//! A `Visible` override draws the layer whatever the file says (off, frozen, no-plot, Defpoints)
//! and thaws it in every viewport; a `Hidden` override leaves it out of the sheet's own entities,
//! of model space seen through every viewport, and of an INSERT's expansion. A per-viewport
//! override edits that viewport's frozen list only. A per-row `Hidden` silences the layer's text
//! too, where the file-level text switches would have shown it. `apply_choices` is where
//! `choices.text`, `choices.layers` and `choices.viewports` become profile knobs.
//!
//! Runs on a hand-built provider so each layer state is there on purpose, and every test first
//! checks the unchanged profile behaves as the file says, so an override never "passes" on an
//! entity that was never there. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::report::{ConversionReport, EntityRow};
use libviprs::pvec::{
    Annotation, ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope,
    EntityStyle, Facet, FacetSource, InstancePath, LayerOverride, LayerState, LayerStateRule,
    LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub, ProviderCaps, ProviderError,
    RawEntity, RawGeometry, SourceFact, SourceTransform, SourceUnit, TextAnchorV, ViewProfile,
    ViewportClip, ViewportEntity,
};
use libviprs::pvec_convert::{
    CadChoices, CadManifest, ManifestLayer, ManifestLayerState, ManifestMask, PvecConversion,
    StateLayer, apply_choices,
};

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), with its viewports and two
/// blocks, `BLK_FRZ` and `BLK_WALLS`, each one line on layer "0".
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

/// One layer per state the rules act on.
fn layer_table() -> Vec<LayerState> {
    let layer = |name: &str, off: bool, frozen: bool, plottable: bool| LayerState {
        name: name.into(),
        handle: None,
        off,
        frozen,
        locked: false,
        plottable,
        ..LayerState::default()
    };
    vec![
        layer("0", false, false, true),
        layer("WALLS", false, false, true),
        layer("FRZ", false, true, true),
        layer("OFFL", true, false, true),
        layer("NOPLOT", false, false, false),
        layer("Defpoints", false, false, false),
        layer("VP", false, false, true),
        // Thawed in the layer table, frozen in viewport A only.
        layer("VPCOLD", false, false, true),
        // Thawed in the layer table, frozen in viewports A and B.
        layer("BOTHCOLD", false, false, true),
    ]
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1627",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1627"
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
        Ok(SourceFact::Present(a3()))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(layer_table())
    }
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        Ok(self.viewports.clone())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let block = |reached_via: &InstancePath, handle: u64| {
            vec![line(handle, "0", [3.0, 6.0], [4.0, 6.0]).with_path(reached_via)]
        };
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => self.model.clone(),
            EntityScope::Layout(LayoutId(1)) => self.paper.clone(),
            EntityScope::BlockDefinition { name, reached_via } if name == "BLK_FRZ" => {
                block(reached_via, BLOCK_FRZ_LINE)
            }
            EntityScope::BlockDefinition { name, reached_via } if name == "BLK_WALLS" => {
                block(reached_via, BLOCK_WALLS_LINE)
            }
            other => return Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        };
        Ok(Box::new(VecCursor(list.into_iter())))
    }
    fn annotation(&self, handle: ItemHandle) -> Option<Annotation> {
        let h = handle.get();
        [T_FRZ, T_WALLS].contains(&h).then(|| Annotation::Text {
            position: [5.0, 5.0, 0.0],
            height: 0.2,
            rotation: 0.0,
            content: format!("NOTE {h:X}"),
            normal: [0.0, 0.0, 1.0],
            width_factor: 1.0,
            anchor: (0.0, TextAnchorV::Baseline),
            fit: None,
            block_width: 0.0,
            line_spacing: 1.0,
        })
    }
}

// Paper space of `Sheet`.
const P_WALLS: u64 = 0x21;
const P_FRZ: u64 = 0x22;
const P_NOPLOT: u64 = 0x23;
const P_OFF: u64 = 0x24;
const P_DEFPOINTS: u64 = 0x25;
/// A paper line on a layer that viewport A freezes: a viewport freeze never touches paper entities.
const P_VPCOLD: u64 = 0x26;
/// A paper line on layer 0, which no test overrides, so a conversion never ends up empty.
const P_ZERO: u64 = 0x27;

// Viewports: the paper view, A (freezes VPCOLD and BOTHCOLD) and B (freezes BOTHCOLD).
const VP_A: u64 = 0x31;
const VP_B: u64 = 0x32;

// Model space: one line per layer, two notes, two INSERTs.
const L_VPCOLD: u64 = 0x40;
const L_BOTHCOLD: u64 = 0x41;
const L_FRZ: u64 = 0x42;
const L_OFF: u64 = 0x43;
const L_NOPLOT: u64 = 0x44;
const L_DEFPOINTS: u64 = 0x45;
const L_WALLS: u64 = 0x46;
const T_FRZ: u64 = 0x47;
const T_WALLS: u64 = 0x48;
/// An INSERT of `BLK_FRZ` on the frozen layer.
const INS_FRZ: u64 = 0x49;
/// An INSERT of `BLK_WALLS` on WALLS.
const INS_WALLS: u64 = 0x4A;
// The blocks' lines, on layer "0", so each takes its INSERT's layer.
const BLOCK_FRZ_LINE: u64 = 0x60;
const BLOCK_WALLS_LINE: u64 = 0x61;

trait WithPath {
    fn with_path(self, via: &InstancePath) -> Self;
}

impl WithPath for RawEntity {
    fn with_path(mut self, via: &InstancePath) -> Self {
        self.provenance.block_path = via.clone();
        self
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

fn unmapped(handle: u64, class: &str, layer: &str) -> RawEntity {
    ent(
        handle,
        class,
        layer,
        RawGeometry::Unmapped {
            reason: "drawn through the annotation streams".into(),
        },
    )
}

fn insert(handle: u64, layer: &str, block: &str) -> RawEntity {
    ent(
        handle,
        "INSERT",
        layer,
        RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::IDENTITY,
            normal: [0.0, 0.0, 1.0],
            grid: ArrayGrid::SINGLE,
            elevation: 0.0,
            z_scale: 1.0,
        }),
    )
}

/// An A3 sheet in millimetres plotted 1:1 at the paper-space origin.
fn a3() -> LayoutPlot {
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

/// A shows model x 0..10, y 1..9; B shows x -5..15, y -3..13. Everything model-side sits near
/// (5, 5), so each viewport frames every entity.
fn viewports() -> Vec<ViewportEntity> {
    let mut a = vp(VP_A, 2, [100.0, 100.0], [100.0, 80.0], [5.0, 5.0], 8.0);
    a.frozen_layers = vec!["vpcold".into(), "bothcold".into()];
    let mut b = vp(VP_B, 3, [300.0, 100.0], [100.0, 80.0], [5.0, 5.0], 16.0);
    b.frozen_layers = vec!["BOTHCOLD".into()];
    vec![
        vp(
            0x30,
            1,
            [210.0, 148.5],
            [420.0, 297.0],
            [210.0, 148.5],
            297.0,
        ),
        a,
        b,
    ]
}

/// The drawing, checked first: every layer state a test asserts about has an entity on it.
fn drawing() -> Mock {
    let mut paper = vec![
        line(P_WALLS, "WALLS", [10.0, 10.0], [410.0, 10.0]),
        line(P_FRZ, "FRZ", [10.0, 20.0], [410.0, 20.0]),
        line(P_NOPLOT, "NOPLOT", [10.0, 30.0], [410.0, 30.0]),
        line(P_OFF, "OFFL", [10.0, 40.0], [410.0, 40.0]),
        line(P_DEFPOINTS, "Defpoints", [10.0, 50.0], [410.0, 50.0]),
        line(P_VPCOLD, "VPCOLD", [10.0, 60.0], [410.0, 60.0]),
        line(P_ZERO, "0", [10.0, 70.0], [410.0, 70.0]),
    ];
    for v in viewports() {
        paper.push(unmapped(v.handle.get(), "VIEWPORT", &v.layer));
    }
    let model = vec![
        line(L_VPCOLD, "VPCOLD", [4.0, 4.0], [6.0, 4.0]),
        line(L_BOTHCOLD, "BOTHCOLD", [4.0, 4.5], [6.0, 4.5]),
        line(L_FRZ, "FRZ", [4.0, 5.0], [6.0, 5.0]),
        line(L_OFF, "OFFL", [4.0, 5.5], [6.0, 5.5]),
        line(L_NOPLOT, "NOPLOT", [4.0, 6.0], [6.0, 6.0]),
        line(L_DEFPOINTS, "Defpoints", [4.0, 6.5], [6.0, 6.5]),
        line(L_WALLS, "WALLS", [4.0, 7.0], [6.0, 7.0]),
        unmapped(T_FRZ, "TEXT", "FRZ"),
        unmapped(T_WALLS, "TEXT", "WALLS"),
        insert(INS_FRZ, "FRZ", "BLK_FRZ"),
        insert(INS_WALLS, "WALLS", "BLK_WALLS"),
    ];
    let mock = Mock {
        paper,
        model,
        viewports: viewports(),
    };
    let layers = layer_table();
    let state = |n: &str| layers.iter().find(|l| l.name == n).unwrap().clone();
    assert!(state("FRZ").frozen && state("OFFL").off && !state("NOPLOT").plottable);
    assert!(!state("Defpoints").plottable && !state("VPCOLD").frozen);
    let a = &mock.viewports[1];
    let b = &mock.viewports[2];
    assert!(a.frozen_layers.iter().any(|l| l == "vpcold") && a.frozen_layers.len() == 2);
    assert!(b.frozen_layers == ["BOTHCOLD"]);
    mock
}

// ---- running it -------------------------------------------------------------------------

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1627-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts the drawing under `profile` with the text stream on and hands back the report.
fn convert(tag: &str, profile: ViewProfile) -> ConversionReport {
    let dir = scratch(tag);
    let dest = dir.join("out.pvec");
    NEXT.with(|n| *n.borrow_mut() = Some(drawing()));
    let out = PvecConversion::new(CadSource::Bytes(b"mock"), &dest)
        .unit(SourceUnit::Meters)
        .view_profile(profile)
        .annotations(true)
        .convert_with::<Mock>()
        .expect("the drawing converts");
    let _ = fs::remove_dir_all(dir);
    out.report
}

fn sheet() -> ViewProfile {
    ViewProfile::print_profile("Sheet")
}

fn model() -> ViewProfile {
    ViewProfile::default_profile()
}

fn with_overrides(mut p: ViewProfile, o: &[(&str, LayerOverride)]) -> ViewProfile {
    p.layers.overrides = o.iter().map(|(n, w)| ((*n).to_string(), *w)).collect();
    p
}

fn with_viewport_overrides(mut p: ViewProfile, o: &[(&str, &str, LayerOverride)]) -> ViewProfile {
    p.layers.viewport_overrides = o
        .iter()
        .map(|(k, n, w)| ((*k).to_string(), (*n).to_string(), *w))
        .collect();
    p
}

fn rows(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle) && r.viewport == viewport)
        .collect()
}

/// Drawn there: it has a row, and no policy dropped any of them.
fn shown(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> bool {
    let r = rows(report, handle, viewport);
    !r.is_empty() && r.iter().all(|r| r.fidelity != "IgnoredByPolicy")
}

/// Drawn on the sheet or in model space (`None`) or through each of the viewports.
fn shown_everywhere(report: &ConversionReport, handle: u64) -> bool {
    [VP_A, VP_B].iter().all(|v| shown(report, handle, Some(*v)))
}

/// Drawn through neither viewport.
fn shown_nowhere(report: &ConversionReport, handle: u64) -> bool {
    [VP_A, VP_B]
        .iter()
        .all(|v| !shown(report, handle, Some(*v)))
}

// ---- 9.7: a Visible override thaws everywhere --------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_visible_override_draws_a_viewport_frozen_layer_through_every_viewport() {
    let before = convert("thaw-before", sheet());
    assert!(
        !shown(&before, L_VPCOLD, Some(VP_A)) && shown(&before, L_VPCOLD, Some(VP_B)),
        "VPCOLD is frozen in A only"
    );
    assert!(
        shown_nowhere(&before, L_BOTHCOLD),
        "BOTHCOLD is frozen in A and B"
    );
    let report = convert(
        "thaw",
        with_overrides(
            sheet(),
            &[
                ("VPCOLD", LayerOverride::Visible),
                ("BOTHCOLD", LayerOverride::Visible),
            ],
        ),
    );
    assert!(
        shown_everywhere(&report, L_VPCOLD),
        "{:#?}",
        report.entities
    );
    assert!(
        shown_everywhere(&report, L_BOTHCOLD),
        "{:#?}",
        report.entities
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_visible_override_beats_off_frozen_no_plot_and_defpoints_on_a_sheet() {
    let before = convert("beats-before", sheet());
    for h in [L_FRZ, L_OFF, L_NOPLOT, L_DEFPOINTS] {
        assert!(shown_nowhere(&before, h), "{h:#X} is left out by the file");
    }
    for h in [P_FRZ, P_OFF, P_NOPLOT, P_DEFPOINTS] {
        assert!(!shown(&before, h, None), "{h:#X} is left out by the file");
    }
    let report = convert(
        "beats",
        with_overrides(
            sheet(),
            &[
                ("FRZ", LayerOverride::Visible),
                ("OFFL", LayerOverride::Visible),
                ("NOPLOT", LayerOverride::Visible),
                ("DEFPOINTS", LayerOverride::Visible),
            ],
        ),
    );
    for h in [L_FRZ, L_OFF, L_NOPLOT, L_DEFPOINTS] {
        assert!(
            shown_everywhere(&report, h),
            "{h:#X} through both viewports"
        );
    }
    for h in [P_FRZ, P_OFF, P_NOPLOT, P_DEFPOINTS] {
        assert!(shown(&report, h, None), "{h:#X} on the sheet");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_visible_override_beats_off_and_frozen_in_model_space() {
    let before = convert("model-before", model());
    assert!(!shown(&before, L_FRZ, None) && !shown(&before, L_OFF, None));
    let report = convert(
        "model",
        with_overrides(
            model(),
            &[
                ("FRZ", LayerOverride::Visible),
                ("OFFL", LayerOverride::Visible),
            ],
        ),
    );
    assert!(shown(&report, L_FRZ, None), "{:#?}", report.entities);
    assert!(shown(&report, L_OFF, None), "{:#?}", report.entities);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_an_override_on_an_inserts_layer_carries_its_expansion() {
    // The block's line is on layer "0", so it takes its INSERT's layer.
    let before = convert("insert-before", sheet());
    assert!(shown_nowhere(&before, BLOCK_FRZ_LINE));
    assert!(shown_everywhere(&before, BLOCK_WALLS_LINE));
    let report = convert(
        "insert",
        with_overrides(
            sheet(),
            &[
                ("FRZ", LayerOverride::Visible),
                ("WALLS", LayerOverride::Hidden),
            ],
        ),
    );
    assert!(
        shown_everywhere(&report, BLOCK_FRZ_LINE),
        "{:#?}",
        report.entities
    );
    assert!(
        shown_nowhere(&report, BLOCK_WALLS_LINE),
        "{:#?}",
        report.entities
    );
    let model_report = convert(
        "insert-model",
        with_overrides(
            model(),
            &[
                ("FRZ", LayerOverride::Visible),
                ("WALLS", LayerOverride::Hidden),
            ],
        ),
    );
    assert!(shown(&model_report, BLOCK_FRZ_LINE, None));
    assert!(!shown(&model_report, BLOCK_WALLS_LINE, None));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_override_names_match_ascii_case_insensitively() {
    let report = convert(
        "case",
        with_overrides(
            sheet(),
            &[
                ("vpCold", LayerOverride::Visible),
                ("walls", LayerOverride::Hidden),
            ],
        ),
    );
    assert!(shown_everywhere(&report, L_VPCOLD));
    assert!(shown_nowhere(&report, L_WALLS));
    assert!(!shown(&report, P_WALLS, None));
}

// ---- 9.7: a Hidden override drops paper and viewport entities ----------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_hidden_override_drops_a_layers_sheet_and_viewport_entities() {
    let before = convert("hidden-before", sheet());
    assert!(shown(&before, P_WALLS, None) && shown_everywhere(&before, L_WALLS));
    assert!(
        shown(&before, P_VPCOLD, None),
        "paper entities ignore a viewport freeze"
    );
    let report = convert(
        "hidden",
        with_overrides(
            sheet(),
            &[
                ("WALLS", LayerOverride::Hidden),
                ("VPCOLD", LayerOverride::Hidden),
            ],
        ),
    );
    assert!(!shown(&report, P_WALLS, None), "{:#?}", report.entities);
    assert!(!shown(&report, P_VPCOLD, None), "{:#?}", report.entities);
    assert!(shown_nowhere(&report, L_WALLS), "{:#?}", report.entities);
    assert!(shown_nowhere(&report, L_VPCOLD), "{:#?}", report.entities);
    // The layers nobody overrode are untouched.
    assert!(shown_everywhere(&report, L_BOTHCOLD) == shown_everywhere(&before, L_BOTHCOLD));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_hidden_override_drops_a_layer_in_model_space() {
    let before = convert("model-hidden-before", model());
    assert!(shown(&before, L_WALLS, None) && shown(&before, T_WALLS, None));
    let report = convert(
        "model-hidden",
        with_overrides(model(), &[("WALLS", LayerOverride::Hidden)]),
    );
    assert!(!shown(&report, L_WALLS, None), "{:#?}", report.entities);
    assert!(!shown(&report, T_WALLS, None), "{:#?}", report.entities);
}

// ---- the text exceptions ------------------------------------------------------------------

// Rom's answer 2 (section 12): the text switches show notes on frozen layers, and a row the user
// set to Hidden still silences them.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_per_row_hidden_wins_over_the_frozen_text_exception() {
    for (tag, profile) in [
        ("frozen-text-model", model()),
        ("frozen-text-sheet", sheet()),
    ] {
        let at = |h: u64, r: &ConversionReport| {
            if tag.ends_with("model") {
                shown(r, h, None)
            } else {
                shown_everywhere(r, h)
            }
        };
        let before = convert(tag, profile.clone());
        assert!(
            at(T_FRZ, &before),
            "{tag}: a note on a frozen layer shows by default"
        );
        assert!(!at(L_FRZ, &before), "{tag}: its line does not");
        let report = convert(
            &format!("{tag}-hidden"),
            with_overrides(profile.clone(), &[("FRZ", LayerOverride::Hidden)]),
        );
        assert!(!at(T_FRZ, &report), "{tag}: Hidden silences the note");
        assert!(!at(L_FRZ, &report), "{tag}");
        let report = convert(
            &format!("{tag}-visible"),
            with_overrides(profile, &[("FRZ", LayerOverride::Visible)]),
        );
        assert!(at(T_FRZ, &report), "{tag}: Visible draws the note");
        assert!(at(L_FRZ, &report), "{tag}: and the line");
    }
}

// ---- per-viewport overrides ---------------------------------------------------------------

// Rom's answer 3 (section 12): the converter accepts `choices.viewports` and edits each viewport's
// frozen list before drawing.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_per_viewport_override_thaws_a_layer_in_that_viewport_only() {
    let report = convert(
        "vp-visible",
        with_viewport_overrides(
            sheet(),
            &[
                ("Sheet/31", "BOTHCOLD", LayerOverride::Visible),
                ("Sheet/31", "vpcold", LayerOverride::Visible),
            ],
        ),
    );
    assert!(shown(&report, L_BOTHCOLD, Some(VP_A)), "thawed in A");
    assert!(!shown(&report, L_BOTHCOLD, Some(VP_B)), "still frozen in B");
    assert!(shown(&report, L_VPCOLD, Some(VP_A)), "thawed in A");
    assert!(shown(&report, L_VPCOLD, Some(VP_B)), "B never froze it");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_per_viewport_override_hides_a_layer_in_that_viewport_only() {
    let before = convert("vp-hidden-before", sheet());
    assert!(shown_everywhere(&before, L_WALLS) && shown_everywhere(&before, BLOCK_WALLS_LINE));
    let report = convert(
        "vp-hidden",
        with_viewport_overrides(sheet(), &[("Sheet/32", "walls", LayerOverride::Hidden)]),
    );
    assert!(shown(&report, L_WALLS, Some(VP_A)), "drawn in A");
    assert!(!shown(&report, L_WALLS, Some(VP_B)), "hidden in B");
    assert!(
        shown(&report, BLOCK_WALLS_LINE, Some(VP_A)),
        "A keeps the expansion"
    );
    assert!(!shown(&report, BLOCK_WALLS_LINE, Some(VP_B)), "B drops it");
    assert!(
        shown(&report, P_WALLS, None),
        "a viewport override never touches the layout's own entities"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_layer_can_be_visible_in_one_viewport_and_hidden_in_another() {
    let report = convert(
        "vp-split",
        with_viewport_overrides(
            sheet(),
            &[
                ("Sheet/31", "BOTHCOLD", LayerOverride::Visible),
                ("Sheet/32", "BOTHCOLD", LayerOverride::Hidden),
                ("Sheet/31", "WALLS", LayerOverride::Hidden),
                ("Sheet/32", "WALLS", LayerOverride::Visible),
            ],
        ),
    );
    assert!(shown(&report, L_BOTHCOLD, Some(VP_A)));
    assert!(!shown(&report, L_BOTHCOLD, Some(VP_B)));
    assert!(!shown(&report, L_WALLS, Some(VP_A)));
    assert!(shown(&report, L_WALLS, Some(VP_B)));
}

// ---- 9.8: the choices become profile knobs -------------------------------------------------

fn choices(json: &str) -> CadChoices {
    serde_json::from_str(json).expect("the choices parse")
}

fn apply(profile: ViewProfile, json: &str) -> ViewProfile {
    apply_choices(profile, &CadManifest::default(), &choices(json))
}

// 9.8 text-switches-map-to-knobs
#[test]
fn issue_1627_the_text_switches_map_to_the_two_text_knobs() {
    let hide_frozen = r#"{"schema":1,"mode":"layouts",
        "text":{"frozenLayers":"hide","noPlotLayers":"hide"}}"#;
    for base in [model(), sheet()] {
        let p = apply(base, hide_frozen);
        assert_eq!(p.layers.frozen_text, LayerStateRule::Exclude);
        assert_eq!(p.layers.no_plot_text, LayerStateRule::Exclude);
    }
    let show_both = r#"{"schema":1,"mode":"layouts",
        "text":{"frozenLayers":"show","noPlotLayers":"show"}}"#;
    for base in [model(), sheet()] {
        let p = apply(base, show_both);
        assert_eq!(p.layers.frozen_text, LayerStateRule::Ignore);
        assert_eq!(p.layers.no_plot_text, LayerStateRule::Ignore);
    }
    // The switches act on text only: the line rules stay as the profile had them.
    let p = apply(sheet(), hide_frozen);
    assert_eq!(p.layers.frozen, sheet().layers.frozen);
    assert_eq!(p.layers.no_plot, sheet().layers.no_plot);
}

// 9.8 again: the D4 tests stay green at the defaults, and a user who changed nothing gets the
// quick path's profile.
#[test]
fn issue_1627_no_choices_leave_the_profile_as_it_was() {
    for base in [model(), sheet()] {
        let untouched = base.clone();
        let p = apply(base, r#"{"schema":1,"mode":"layouts"}"#);
        assert_eq!(p, untouched);
    }
    // Section 12, answer 1: the no-plot text default is show in model mode, hide on a sheet.
    let p = apply(model(), r#"{"schema":1,"mode":"model"}"#);
    assert_eq!(p.layers.no_plot_text, LayerStateRule::Ignore);
    let p = apply(sheet(), r#"{"schema":1,"mode":"layouts"}"#);
    assert_eq!(p.layers.no_plot_text, LayerStateRule::Exclude);
}

#[test]
fn issue_1627_choices_layers_become_overrides() {
    let p = apply(
        sheet(),
        r#"{"schema":1,"mode":"layouts",
            "layers":{"A-DIMS":"visible","a-anno-demo":"hidden"}}"#,
    );
    let find = |name: &str| {
        p.layers
            .overrides
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, o)| *o)
    };
    assert_eq!(p.layers.overrides.len(), 2, "{:?}", p.layers.overrides);
    assert_eq!(find("A-DIMS"), Some(LayerOverride::Visible));
    assert_eq!(find("A-ANNO-DEMO"), Some(LayerOverride::Hidden));
}

#[test]
fn issue_1627_choices_viewports_become_per_viewport_overrides() {
    let p = apply(
        sheet(),
        r#"{"schema":1,"mode":"layouts",
            "viewports":{"Sheet/31":{"A-DIMS":"visible"},"Sheet/32":{"WALLS":"hidden"}}}"#,
    );
    let mut got = p.layers.viewport_overrides.clone();
    got.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    assert_eq!(
        got,
        vec![
            (
                "Sheet/31".to_string(),
                "A-DIMS".to_string(),
                LayerOverride::Visible
            ),
            (
                "Sheet/32".to_string(),
                "WALLS".to_string(),
                LayerOverride::Hidden
            ),
        ]
    );
    assert!(
        p.layers.overrides.is_empty(),
        "a viewport override is not a layer override"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_conversion_under_the_choices_follows_them() {
    let p = apply(
        sheet(),
        r#"{"schema":1,"mode":"layouts",
            "layers":{"BOTHCOLD":"visible","frz":"hidden"},
            "viewports":{"Sheet/32":{"WALLS":"hidden"}},
            "text":{"frozenLayers":"show","noPlotLayers":"hide"}}"#,
    );
    let report = convert("choices", p);
    assert!(shown_everywhere(&report, L_BOTHCOLD));
    assert!(
        !shown(&report, T_FRZ, Some(VP_A)),
        "the row's Hidden wins over the text switch"
    );
    assert!(shown(&report, L_WALLS, Some(VP_A)) && !shown(&report, L_WALLS, Some(VP_B)));
}

// ---- the reason an override gives (panel decision Q1) ---------------------------------------

fn fields(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> Vec<Option<String>> {
    rows(report, handle, viewport)
        .iter()
        .map(|r| r.policy_field.clone())
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_hidden_override_is_reported_under_layers_overrides() {
    let report = convert(
        "reason-sheet",
        with_overrides(sheet(), &[("WALLS", LayerOverride::Hidden)]),
    );
    let on_sheet = fields(&report, P_WALLS, None);
    assert_eq!(
        on_sheet,
        [Some("layers.overrides".to_string())],
        "{on_sheet:?}"
    );
    let report = convert(
        "reason-model",
        with_overrides(model(), &[("WALLS", LayerOverride::Hidden)]),
    );
    for h in [L_WALLS, T_WALLS] {
        let f = fields(&report, h, None);
        assert_eq!(f, [Some("layers.overrides".to_string())], "{h:#X}: {f:?}");
    }
    // An off layer is still `layers.off`: the override is the user's word, not the file's.
    let report = convert("reason-off", model());
    assert_eq!(
        fields(&report, L_OFF, None),
        [Some("layers.off".to_string())]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_per_viewport_hidden_reports_the_same_reason() {
    let report = convert(
        "reason-vp",
        with_viewport_overrides(sheet(), &[("Sheet/32", "WALLS", LayerOverride::Hidden)]),
    );
    let f = fields(&report, L_WALLS, Some(VP_B));
    assert_eq!(f, [Some("layers.overrides".to_string())], "{f:?}");
}

// ---- the most specific override wins (panel decision Q3) -------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_viewport_visible_beats_a_global_hidden_for_that_viewport_only() {
    let mut p = with_overrides(sheet(), &[("WALLS", LayerOverride::Hidden)]);
    p = with_viewport_overrides(p, &[("Sheet/31", "WALLS", LayerOverride::Visible)]);
    let report = convert("specific-visible", p);
    assert!(
        shown(&report, L_WALLS, Some(VP_A)),
        "{:#?}",
        report.entities
    );
    assert!(
        !shown(&report, L_WALLS, Some(VP_B)),
        "the other viewport follows the row"
    );
    assert!(
        shown(&report, BLOCK_WALLS_LINE, Some(VP_A)),
        "and so does the expansion"
    );
    assert!(!shown(&report, BLOCK_WALLS_LINE, Some(VP_B)));
    assert!(
        !shown(&report, P_WALLS, None),
        "paper space follows the global override, a viewport entry never reaches it"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_viewport_hidden_beats_a_global_visible_for_that_viewport_only() {
    let mut p = with_overrides(sheet(), &[("BOTHCOLD", LayerOverride::Visible)]);
    p = with_viewport_overrides(p, &[("Sheet/32", "BOTHCOLD", LayerOverride::Hidden)]);
    let report = convert("specific-hidden", p);
    assert!(shown(&report, L_BOTHCOLD, Some(VP_A)), "thawed by the row");
    assert!(
        !shown(&report, L_BOTHCOLD, Some(VP_B)),
        "hidden by its own entry"
    );
}

// ---- a chosen layer state folds into the overrides (panel decision Q2) ------------------------

fn layer(name: &str, on: bool, frozen: bool, plot: bool) -> ManifestLayer {
    ManifestLayer {
        name: name.into(),
        on,
        frozen,
        plot,
        ..ManifestLayer::default()
    }
}

fn recorded(name: &str, on: bool, frozen: bool, plot: bool) -> StateLayer {
    StateLayer {
        name: name.into(),
        on,
        frozen,
        locked: false,
        plot,
        vp_frozen: false,
    }
}

fn state_manifest(
    layers: Vec<ManifestLayer>,
    mask: ManifestMask,
    recorded: Vec<StateLayer>,
) -> CadManifest {
    CadManifest {
        schema: 1,
        layers,
        layer_states: vec![ManifestLayerState {
            name: "S".into(),
            mask,
            layers: recorded,
            ..ManifestLayerState::default()
        }],
        ..CadManifest::default()
    }
}

fn folded(m: &CadManifest, profile: ViewProfile, json: &str) -> ViewProfile {
    apply_choices(profile, m, &choices(json))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_state_that_freezes_a_layer_hides_it_in_model_mode() {
    let m = state_manifest(
        vec![layer("WALLS", true, false, true)],
        ManifestMask {
            frozen: true,
            ..ManifestMask::default()
        },
        vec![recorded("WALLS", true, true, true)],
    );
    let p = folded(
        &m,
        model(),
        r#"{"schema":1,"mode":"model","layerState":"S"}"#,
    );
    assert!(
        p.layers.state_overrides.is_empty(),
        "a freeze is a flag, not an override"
    );
    assert_eq!(p.layers.state_frozen, ["WALLS"]);
    assert!(
        p.layers.overrides.is_empty(),
        "nothing the user set by hand"
    );
    let report = convert("state-freeze", p);
    assert_eq!(
        fields(&report, L_WALLS, None),
        [Some("layers.frozen".to_string())]
    );
}

#[test]
fn issue_1627_a_state_that_does_not_set_plot_leaves_a_no_plot_layer_alone() {
    // The state records plot = true for the layer, but its mask does not carry Plot, so the
    // answer does not change and no override is made.
    let m = state_manifest(
        vec![layer("NOPLOT", true, false, false)],
        ManifestMask {
            plot: false,
            on: true,
            ..ManifestMask::default()
        },
        vec![recorded("NOPLOT", true, false, true)],
    );
    let p = folded(
        &m,
        sheet(),
        r#"{"schema":1,"mode":"layouts","layerState":"S"}"#,
    );
    assert!(
        p.layers.state_overrides.is_empty(),
        "{:?}",
        p.layers.state_overrides
    );
    // And with Plot in the mask the state does change it, to visible.
    let m = state_manifest(
        vec![layer("NOPLOT", true, false, false)],
        ManifestMask {
            plot: true,
            ..ManifestMask::default()
        },
        vec![recorded("NOPLOT", true, false, true)],
    );
    let p = folded(
        &m,
        sheet(),
        r#"{"schema":1,"mode":"layouts","layerState":"S"}"#,
    );
    assert_eq!(
        p.layers.state_overrides,
        [("NOPLOT".to_string(), LayerOverride::Visible)]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_state_visible_never_thaws_a_viewport() {
    let m = state_manifest(
        vec![layer("BOTHCOLD", false, false, true)],
        ManifestMask {
            on: true,
            ..ManifestMask::default()
        },
        vec![recorded("BOTHCOLD", true, false, true)],
    );
    let p = folded(
        &m,
        sheet(),
        r#"{"schema":1,"mode":"layouts","layerState":"S"}"#,
    );
    assert_eq!(
        p.layers.state_overrides,
        [("BOTHCOLD".to_string(), LayerOverride::Visible)]
    );
    assert!(
        p.layers.overrides.is_empty(),
        "only a user's Visible thaws viewports"
    );
    let report = convert("state-visible", p);
    assert!(
        shown_nowhere(&report, L_BOTHCOLD),
        "A and B still freeze it"
    );
}

#[test]
fn issue_1627_turn_off_layers_not_in_state_hides_an_unlisted_layer() {
    let m = state_manifest(
        vec![
            layer("WALLS", true, false, true),
            layer("FRZ2", true, false, true),
        ],
        ManifestMask::default(),
        vec![recorded("FRZ2", true, false, true)],
    );
    let p = folded(
        &m,
        model(),
        r#"{"schema":1,"mode":"model","layerState":"S","turnOffLayersNotInState":true}"#,
    );
    assert_eq!(
        p.layers.state_overrides,
        [("WALLS".to_string(), LayerOverride::Hidden)]
    );
}

#[test]
fn issue_1627_a_users_layer_entry_beats_the_state_for_the_same_name() {
    let m = state_manifest(
        vec![layer("WALLS", true, false, true)],
        ManifestMask {
            frozen: true,
            ..ManifestMask::default()
        },
        vec![recorded("WALLS", true, true, true)],
    );
    let p = folded(
        &m,
        model(),
        r#"{"schema":1,"mode":"model","layerState":"S","layers":{"walls":"visible"}}"#,
    );
    assert_eq!(
        p.layers.overrides,
        [("walls".to_string(), LayerOverride::Visible)]
    );
    assert!(
        p.layers.state_overrides.is_empty(),
        "the state's entry for the same name is replaced: {:?}",
        p.layers.state_overrides
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1627_a_state_freeze_keeps_the_notes_under_frozen_layers_show() {
    let m = state_manifest(
        vec![layer("WALLS", true, false, true)],
        ManifestMask {
            frozen: true,
            ..ManifestMask::default()
        },
        vec![recorded("WALLS", true, true, true)],
    );
    for (tag, text, note) in [
        (
            "show",
            r#""frozenLayers":"show","noPlotLayers":"show""#,
            true,
        ),
        (
            "hide",
            r#""frozenLayers":"hide","noPlotLayers":"show""#,
            false,
        ),
    ] {
        let json = format!(r#"{{"schema":1,"mode":"model","layerState":"S","text":{{{text}}}}}"#);
        let p = folded(&m, model(), &json);
        assert_eq!(p.layers.state_frozen, ["WALLS"], "{tag}");
        let report = convert(&format!("state-freeze-text-{tag}"), p);
        assert!(
            !shown(&report, L_WALLS, None),
            "{tag}: the line is frozen out"
        );
        assert_eq!(shown(&report, T_WALLS, None), note, "{tag}: the note");
    }
}
