//! Issue #1621, layer rules for DWG text (panel decision D4, Rom's call of 2026-11-19): a TEXT,
//! MTEXT or visible attribute on a frozen layer is drawn and one on a no-plot layer is not, in
//! model mode (`default_profile`) and on a sheet (`print_profile`) alike. Everything else stays as
//! V1-3 and P-3 say: an off layer hides text, a viewport freeze drops it inside that viewport only,
//! an INSERT on a frozen layer takes its whole expansion (attributes included) with it, DIMENSION
//! and MULTILEADER follow their layer whole, and a viewport on a no-plot layer still shows what it
//! frames. The layer is the effective one after flattening. Two knobs carry it,
//! `layers.frozen_text` and `layers.no_plot_text`, and the one layer filter applies them, so the
//! annotation mapper never sees what the filter dropped.
//!
//! Most of it runs on a hand-built provider so each layer state is there on purpose (Rom's drawing
//! may not have every case); every test first checks that the fixture really has text on the layer
//! state it is about. The attribute cases also run on a drawing written with acadrust, so the
//! provider's own ATTRIB handling is in the loop. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow};
use libviprs::pvec::{
    Annotation, ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope,
    EntityStyle, Exclusion, Facet, FacetSource, InstancePath, LayerState, LayoutId, LayoutKind,
    LayoutPlot, LayoutState, ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry,
    SourceFact, SourceTransform, SourceUnit, ViewProfile, ViewportClip, ViewportEntity,
};
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput};
use pvec_facade::components::TextStream;

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0) and one paper layout, `Sheet` (layout 1), with its viewports, one block
/// `BLK`, and the text each TEXT, MTEXT and ATTDEF draws.
#[derive(Clone)]
struct Mock {
    paper: Vec<RawEntity>,
    model: Vec<RawEntity>,
    block: Vec<RawEntity>,
    viewports: Vec<ViewportEntity>,
}

thread_local! {
    static NEXT: RefCell<Option<Mock>> = const { RefCell::new(None) };
    /// Every handle the conversion asked the provider to draw as an annotation, on this thread.
    static ASKED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

struct VecCursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for VecCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

/// The layer table: one layer per state the rules act on.
fn layer_table() -> Vec<LayerState> {
    let layer = |name: &str, off: bool, frozen: bool, plottable: bool| LayerState {
        name: name.into(),
        handle: None,
        off,
        frozen,
        locked: false,
        plottable,
    };
    vec![
        layer("0", false, false, true),
        layer("NOTES", false, false, true),
        layer("FRZ", false, true, true),
        layer("NOPLOT", false, false, false),
        layer("OFF", true, false, true),
        layer("VP", false, false, true),
        // Thawed in the layer table, frozen in viewport A only.
        layer("VPCOLD", false, false, true),
    ]
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1621",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1621"
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
    fn annotation(&self, handle: ItemHandle) -> Option<Annotation> {
        ASKED.with(|a| a.borrow_mut().push(handle.get()));
        drawn(handle.get())
    }
    fn inserted_attribute(&self, attdef: ItemHandle, insert: ItemHandle) -> Option<Annotation> {
        let _ = insert;
        ASKED.with(|a| a.borrow_mut().push(attdef.get()));
        drawn(attdef.get())
    }
}

// Paper space of `Sheet`.
const SHEET_FROZEN_TEXT: u64 = 0x21;
const SHEET_FROZEN_LINE: u64 = 0x22;
const SHEET_NO_PLOT_TEXT: u64 = 0x23;
const SHEET_NO_PLOT_LINE: u64 = 0x24;
const SHEET_OFF_TEXT: u64 = 0x25;

// Viewports: the paper view, A (freezes VPCOLD), B, and one on the no-plot layer.
const VP_A: u64 = 0x31;
const VP_B: u64 = 0x32;
const VP_NO_PLOT: u64 = 0x33;

// Model space.
const NOTE: u64 = 0x40;
const FROZEN_TEXT: u64 = 0x41;
const FROZEN_MTEXT: u64 = 0x42;
const FROZEN_ATTDEF: u64 = 0x43;
const FROZEN_LINE: u64 = 0x44;
const NO_PLOT_TEXT: u64 = 0x45;
const NO_PLOT_LINE: u64 = 0x46;
const OFF_TEXT: u64 = 0x47;
const VP_COLD_TEXT: u64 = 0x48;
const FROZEN_DIMENSION: u64 = 0x49;
const FROZEN_MULTILEADER: u64 = 0x4A;
/// An INSERT of `BLK` on the frozen layer.
const COLD_INSERT: u64 = 0x4B;

// The block `BLK`.
/// An attribute definition on layer "0", so it takes its INSERT's layer.
const BLOCK_ATTDEF: u64 = 0x60;
/// A TEXT on its own frozen layer.
const BLOCK_FROZEN_TEXT: u64 = 0x61;
/// A TEXT on layer "0".
const BLOCK_ZERO_TEXT: u64 = 0x62;

/// What a text handle draws: its string, at model (5, 5), height 0.2.
fn content(handle: u64) -> Option<&'static str> {
    Some(match handle {
        SHEET_FROZEN_TEXT => "SHEET FROZEN NOTE",
        SHEET_NO_PLOT_TEXT => "SHEET NO PLOT NOTE",
        SHEET_OFF_TEXT => "SHEET OFF NOTE",
        NOTE => "NOTE",
        FROZEN_TEXT => "FROZEN NOTE",
        FROZEN_MTEXT => "FROZEN MTEXT",
        FROZEN_ATTDEF => "FROZEN TAG",
        NO_PLOT_TEXT => "NO PLOT NOTE",
        OFF_TEXT => "OFF NOTE",
        VP_COLD_TEXT => "VP COLD NOTE",
        BLOCK_ATTDEF => "ROOM 101",
        BLOCK_FROZEN_TEXT => "BLOCK FROZEN NOTE",
        BLOCK_ZERO_TEXT => "BLOCK ZERO NOTE",
        _ => return None,
    })
}

fn drawn(handle: u64) -> Option<Annotation> {
    content(handle).map(|s| Annotation::Text {
        position: [5.0, 5.0, 0.0],
        height: 0.2,
        rotation: 0.0,
        content: s.into(),
        normal: [0.0, 0.0, 1.0],
    })
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

/// A text-like or annotation entity: the cursor hands it out unmapped, as the providers do.
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

fn insert(handle: u64, layer: &str) -> RawEntity {
    ent(
        handle,
        "INSERT",
        layer,
        RawGeometry::Insert(BlockRef {
            block_name: "BLK".into(),
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

/// A shows model x 0..10, y 1..9 and freezes VPCOLD; B shows x -5..15, y -3..13; the one on
/// NOPLOT shows x 3..7, y 3.5..6.5. Each one frames model (5, 5), where every text sits.
fn viewports() -> Vec<ViewportEntity> {
    let mut a = vp(VP_A, 2, [100.0, 100.0], [100.0, 80.0], [5.0, 5.0], 8.0);
    a.frozen_layers = vec!["vpcold".into()];
    let mut no_plot = vp(VP_NO_PLOT, 4, [210.0, 250.0], [40.0, 30.0], [5.0, 5.0], 3.0);
    no_plot.layer = "NOPLOT".into();
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
        vp(VP_B, 3, [300.0, 100.0], [100.0, 80.0], [5.0, 5.0], 16.0),
        no_plot,
    ]
}

/// The drawing, checked first: it has a text on every layer state a test asserts about (TI's
/// rule), so an absent text never passes because it was never there.
fn drawing() -> Mock {
    let mut paper = vec![
        line(0x20, "0", [10.0, 10.0], [410.0, 10.0]),
        unmapped(SHEET_FROZEN_TEXT, "TEXT", "FRZ"),
        line(SHEET_FROZEN_LINE, "FRZ", [10.0, 20.0], [410.0, 20.0]),
        unmapped(SHEET_NO_PLOT_TEXT, "TEXT", "NOPLOT"),
        line(SHEET_NO_PLOT_LINE, "NOPLOT", [10.0, 30.0], [410.0, 30.0]),
        unmapped(SHEET_OFF_TEXT, "TEXT", "OFF"),
    ];
    for v in viewports() {
        paper.push(unmapped(v.handle.get(), "VIEWPORT", &v.layer));
    }
    let mock = Mock {
        paper,
        model: vec![
            unmapped(NOTE, "TEXT", "NOTES"),
            unmapped(FROZEN_TEXT, "TEXT", "FRZ"),
            unmapped(FROZEN_MTEXT, "MTEXT", "FRZ"),
            unmapped(FROZEN_ATTDEF, "ATTDEF", "FRZ"),
            line(FROZEN_LINE, "FRZ", [4.0, 4.0], [6.0, 4.0]),
            unmapped(NO_PLOT_TEXT, "TEXT", "NOPLOT"),
            line(NO_PLOT_LINE, "NOPLOT", [4.0, 6.0], [6.0, 6.0]),
            unmapped(OFF_TEXT, "TEXT", "OFF"),
            unmapped(VP_COLD_TEXT, "TEXT", "VPCOLD"),
            unmapped(FROZEN_DIMENSION, "DIMENSION", "FRZ"),
            unmapped(FROZEN_MULTILEADER, "MULTILEADER", "FRZ"),
            insert(COLD_INSERT, "FRZ"),
            insert(0x4C, "NOTES"),
        ],
        block: vec![
            unmapped(BLOCK_ATTDEF, "ATTDEF", "0"),
            unmapped(BLOCK_FROZEN_TEXT, "TEXT", "FRZ"),
            unmapped(BLOCK_ZERO_TEXT, "TEXT", "0"),
        ],
        viewports: viewports(),
    };
    check_fixture(&mock);
    mock
}

fn check_fixture(mock: &Mock) {
    let layers = layer_table();
    let state = |name: &str| {
        layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer {name} in the table"))
            .clone()
    };
    assert!(state("FRZ").frozen && !state("FRZ").off && state("FRZ").plottable);
    assert!(!state("NOPLOT").plottable && !state("NOPLOT").frozen && !state("NOPLOT").off);
    assert!(state("OFF").off && !state("OFF").frozen);
    assert!(!state("VPCOLD").frozen && !state("VPCOLD").off);
    assert!(!state("NOTES").frozen && !state("NOTES").off && state("NOTES").plottable);
    let a = mock
        .viewports
        .iter()
        .find(|v| v.handle.get() == VP_A)
        .unwrap();
    assert!(
        a.frozen_layers
            .iter()
            .any(|l| l.eq_ignore_ascii_case("VPCOLD"))
    );
    let b = mock
        .viewports
        .iter()
        .find(|v| v.handle.get() == VP_B)
        .unwrap();
    assert!(b.frozen_layers.is_empty());
    let np = mock
        .viewports
        .iter()
        .find(|v| v.handle.get() == VP_NO_PLOT)
        .unwrap();
    assert!(
        !state(&np.layer).plottable,
        "the no-plot viewport sits on a no-plot layer"
    );
    let has = |list: &[RawEntity], class: &str, layer: &str| {
        list.iter().any(|e| {
            e.provenance.class_name == class
                && e.provenance.layer == layer
                && e.provenance
                    .handle
                    .is_some_and(|h| content(h.get()).is_some())
        })
    };
    for (class, layer) in [
        ("TEXT", "NOTES"),
        ("TEXT", "FRZ"),
        ("MTEXT", "FRZ"),
        ("ATTDEF", "FRZ"),
        ("TEXT", "NOPLOT"),
        ("TEXT", "OFF"),
        ("TEXT", "VPCOLD"),
    ] {
        assert!(has(&mock.model, class, layer), "a model {class} on {layer}");
    }
    for layer in ["FRZ", "NOPLOT", "OFF"] {
        assert!(has(&mock.paper, "TEXT", layer), "a sheet TEXT on {layer}");
    }
    assert!(has(&mock.block, "TEXT", "FRZ") && has(&mock.block, "ATTDEF", "0"));
    for class in ["LINE", "DIMENSION", "MULTILEADER", "INSERT"] {
        assert!(
            mock.model
                .iter()
                .any(|e| e.provenance.class_name == class && e.provenance.layer == "FRZ"),
            "a model {class} on the frozen layer"
        );
    }
}

// ---- running it -------------------------------------------------------------------------

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1621-layers-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The text records' strings of the artifact at `path`, sorted.
fn stream_texts(path: &Path) -> Vec<String> {
    let reader = open_published(path).unwrap();
    let (streams, _) = reader.streams().unwrap();
    let mut out: Vec<String> = streams
        .iter()
        .find(|s| s.capability == TextStream::CAPABILITY)
        .map(|s| {
            TextStream::decode(&s.payload)
                .unwrap()
                .records
                .into_iter()
                .map(|r| r.text)
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

struct Run {
    report: ConversionReport,
    texts: Vec<String>,
    asked: BTreeSet<u64>,
}

/// Converts the drawing under `profile` with the text stream on.
fn convert(tag: &str, mock: Mock, profile: ViewProfile) -> Run {
    let dir = scratch(tag);
    let dest = dir.join("out.pvec");
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    ASKED.with(|a| a.borrow_mut().clear());
    let out: PvecConversionOutput = PvecConversion::new(CadSource::Bytes(b"mock"), &dest)
        .unit(SourceUnit::Meters)
        .view_profile(profile)
        .components(StreamSelection {
            line_style: true,
            ..StreamSelection::NONE
        })
        .annotations(true)
        .convert_with::<Mock>()
        .expect("the drawing converts");
    let texts = stream_texts(&dest);
    let _ = fs::remove_dir_all(dir);
    Run {
        report: out.report,
        texts,
        asked: ASKED.with(|a| a.borrow().iter().copied().collect()),
    }
}

fn model_run(tag: &str) -> Run {
    convert(tag, drawing(), ViewProfile::default_profile())
}

fn sheet_run(tag: &str) -> Run {
    convert(tag, drawing(), ViewProfile::print_profile("Sheet"))
}

/// The rows of `handle` seen through `viewport` (`None`: model space, or the sheet's own).
fn rows(report: &ConversionReport, handle: u64, viewport: Option<u64>) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle) && r.viewport == viewport)
        .collect()
}

/// Drawn: it has a row there (a text may be carried on more than one), and no policy dropped any.
fn assert_drawn(report: &ConversionReport, handle: u64, viewport: Option<u64>) {
    let r = rows(report, handle, viewport);
    assert!(!r.is_empty(), "{handle:#X} through {viewport:X?} has a row");
    assert!(
        r.iter().all(|r| r.fidelity != "IgnoredByPolicy"),
        "{handle:#X} through {viewport:X?} is drawn: {r:#?}"
    );
}

/// Dropped by `field`: every row of the handle (there is one, counted once, not per viewport).
fn assert_dropped(report: &ConversionReport, handle: u64, field: &str) {
    let r: Vec<&EntityRow> = report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle))
        .collect();
    assert_eq!(r.len(), 1, "one row for {handle:#X}: {r:#?}");
    assert_eq!(r[0].fidelity, "IgnoredByPolicy", "{:?}", r[0]);
    assert_eq!(r[0].policy_field.as_deref(), Some(field), "{:?}", r[0]);
}

fn has(texts: &[String], s: &str) -> bool {
    texts.iter().any(|t| t == s)
}

const DRAWN_VIEWPORTS: [u64; 3] = [VP_A, VP_B, VP_NO_PLOT];

// ---- the knobs ----------------------------------------------------------------------------

// Read through the knob table and the printed profile rather than the fields, so this file
// compiles before the knobs exist and is red, not broken.
#[test]
fn issue_1621_both_profiles_draw_frozen_text_and_drop_no_plot_text() {
    for p in [
        ViewProfile::default_profile(),
        ViewProfile::print_profile("Sheet"),
    ] {
        let text = p.to_string();
        assert!(text.contains("layers.frozen_text = ignore"), "{text}");
        assert!(text.contains("layers.no_plot_text = exclude"), "{text}");
        // Geometry keeps its rule: frozen is excluded in both profiles.
        assert!(text.contains("layers.frozen = exclude"), "{text}");
        let knobs = p.knobs();
        let row = |f: &str| knobs.iter().find(|k| k.field == f).map(|k| k.row);
        assert_eq!(row("layers.frozen_text"), Some("layer-frozen-text"));
        assert_eq!(row("layers.no_plot_text"), Some("layer-no-plot-text"));
        let fields = p.exclusion_fields();
        assert!(fields.contains(&"layers.no_plot_text"), "{fields:?}");
        assert!(!fields.contains(&"layers.frozen_text"), "{fields:?}");
    }
    let no_plot_text = Exclusion::ALL
        .into_iter()
        .find(|e| e.field() == "layers.no_plot_text")
        .expect("an exclusion for text on a no-plot layer");
    assert_eq!(no_plot_text.code(), "LAYER_HIDDEN");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_the_adrs_record_the_text_exception_as_roms_call() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let read = |p: &str| fs::read_to_string(root.join(p)).unwrap();
    let view = read("docs/adr/pvec-view-profile.md");
    let print = read("docs/adr/pvec-print-profile.md");
    for (name, doc) in [("V1-3", &view), ("P-3", &print)] {
        assert!(
            doc.contains("Rom's call, 2026-11-19") && doc.contains("#1621"),
            "the {name} amendment names Rom's call and #1621"
        );
    }
    assert!(view.contains("except their\n  text, which is excluded in every profile"));
    assert!(view.contains("frozen-layer text is drawn"));
    for row in ["| `layer-frozen-text` |", "| `layer-no-plot-text` |"] {
        assert!(print.contains(row), "the print ADR's table has {row}");
    }
}

// ---- model mode (default_profile) ---------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_model_mode_draws_text_on_a_frozen_layer_but_not_its_line() {
    let run = model_run("model-frozen");
    for s in ["FROZEN NOTE", "FROZEN MTEXT", "FROZEN TAG"] {
        assert!(
            has(&run.texts, s),
            "{s:?} in the text stream {:?}",
            run.texts
        );
    }
    for h in [FROZEN_TEXT, FROZEN_MTEXT, FROZEN_ATTDEF] {
        assert_drawn(&run.report, h, None);
    }
    assert_dropped(&run.report, FROZEN_LINE, "layers.frozen");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_model_mode_drops_text_on_a_no_plot_layer_and_keeps_its_line() {
    let run = model_run("model-no-plot");
    assert!(!has(&run.texts, "NO PLOT NOTE"), "{:?}", run.texts);
    assert_dropped(&run.report, NO_PLOT_TEXT, "layers.no_plot_text");
    // V1-3 still draws the no-plot layer's geometry on screen.
    assert_drawn(&run.report, NO_PLOT_LINE, None);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_model_mode_an_off_layer_still_hides_its_text() {
    let run = model_run("model-off");
    assert!(!has(&run.texts, "OFF NOTE"), "{:?}", run.texts);
    assert_dropped(&run.report, OFF_TEXT, "layers.off");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_dimension_and_multileader_on_a_frozen_layer_stay_whole_and_excluded() {
    for run in [model_run("dim-model"), sheet_run("dim-sheet")] {
        assert_dropped(&run.report, FROZEN_DIMENSION, "layers.frozen");
        assert_dropped(&run.report, FROZEN_MULTILEADER, "layers.frozen");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_text_on_its_own_frozen_layer_inside_a_visible_insert_is_drawn() {
    let run = model_run("model-block");
    // Through the INSERT on NOTES, the block's own-frozen TEXT is drawn (the layer is the effective
    // one after flattening), and so are its layer-0 TEXT and attribute, which take NOTES.
    for s in ["BLOCK FROZEN NOTE", "BLOCK ZERO NOTE", "ROOM 101"] {
        let n = run.texts.iter().filter(|t| *t == s).count();
        assert_eq!(
            n, 1,
            "{s:?} once, from the INSERT on NOTES only: {:?}",
            run.texts
        );
    }
    for h in [BLOCK_FROZEN_TEXT, BLOCK_ZERO_TEXT, BLOCK_ATTDEF] {
        assert_drawn(&run.report, h, None);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_an_insert_on_a_frozen_layer_takes_its_attributes_and_text_with_it() {
    let outcome = ViewProfile::default_profile()
        .apply(&drawing())
        .expect("the profile applies");
    let cold: Vec<_> = outcome
        .excluded
        .iter()
        .filter(|x| x.handle.map(|h| h.get()) == Some(COLD_INSERT))
        .collect();
    assert_eq!(cold.len(), 1, "{cold:?}");
    assert_eq!(cold[0].reason, Exclusion::LayerFrozen);
    // Its expansion never reached the filter: each block leaf is kept once, from the INSERT on
    // NOTES, and none is excluded on its own.
    for h in [BLOCK_ATTDEF, BLOCK_FROZEN_TEXT, BLOCK_ZERO_TEXT] {
        let kept = outcome
            .included
            .iter()
            .filter(|v| v.entity.provenance.handle.map(|x| x.get()) == Some(h))
            .count();
        assert_eq!(kept, 1, "{h:#X} kept once, through the INSERT on NOTES");
        assert!(
            !outcome
                .excluded
                .iter()
                .any(|x| x.handle.map(|x| x.get()) == Some(h)),
            "{h:#X} is not excluded on its own"
        );
    }
    let run = model_run("model-cold-insert");
    assert_dropped(&run.report, COLD_INSERT, "layers.frozen");
    // The attribute and the texts of the cold INSERT are not in the stream: each string is there
    // once, from the INSERT on NOTES.
    for s in ["ROOM 101", "BLOCK FROZEN NOTE", "BLOCK ZERO NOTE"] {
        assert_eq!(
            run.texts.iter().filter(|t| *t == s).count(),
            1,
            "{:?}",
            run.texts
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_the_annotation_mapper_only_sees_what_the_layer_filter_kept() {
    let mock = drawing();
    let outcome = ViewProfile::default_profile()
        .apply(&mock)
        .expect("the profile applies");
    let kept: BTreeSet<u64> = outcome
        .included
        .iter()
        .filter_map(|v| v.entity.provenance.handle.map(|h| h.get()))
        .collect();
    let dropped: BTreeSet<u64> = outcome
        .excluded
        .iter()
        .filter_map(|x| x.handle.map(|h| h.get()))
        .collect();
    let run = model_run("mapper");
    // Every annotation the conversion asked the provider for was one the filter kept, and every
    // text the filter kept was asked for and carried.
    let stray: Vec<_> = run.asked.difference(&kept).collect();
    assert!(
        stray.is_empty(),
        "the mapper asked for {stray:X?}, which the filter did not keep"
    );
    for h in [OFF_TEXT, NO_PLOT_TEXT] {
        assert!(dropped.contains(&h), "the filter drops {h:#X}");
        assert!(!run.asked.contains(&h), "the mapper never sees {h:#X}");
    }
    let want_kept: BTreeSet<u64> = [
        NOTE,
        FROZEN_TEXT,
        FROZEN_MTEXT,
        FROZEN_ATTDEF,
        VP_COLD_TEXT,
        BLOCK_ATTDEF,
        BLOCK_FROZEN_TEXT,
        BLOCK_ZERO_TEXT,
    ]
    .into();
    assert!(want_kept.is_subset(&kept), "kept {kept:X?}");
    assert!(want_kept.is_subset(&run.asked), "asked {:X?}", run.asked);
    let mut want: Vec<String> = want_kept
        .iter()
        .map(|h| content(*h).unwrap().to_owned())
        .collect();
    want.sort();
    assert_eq!(run.texts, want);
}

// ---- on a sheet (print_profile) -----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_sheet_draws_its_own_frozen_text_but_not_the_frozen_line() {
    let run = sheet_run("sheet-frozen");
    assert_drawn(&run.report, SHEET_FROZEN_TEXT, None);
    assert_dropped(&run.report, SHEET_FROZEN_LINE, "layers.frozen");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_sheet_drops_no_plot_text_and_no_plot_lines() {
    let run = sheet_run("sheet-no-plot");
    assert_dropped(&run.report, SHEET_NO_PLOT_TEXT, "layers.no_plot_text");
    assert_dropped(&run.report, SHEET_NO_PLOT_LINE, "layers.no_plot");
    // Model space seen through the viewports: counted once, not once per viewport.
    assert_dropped(&run.report, NO_PLOT_TEXT, "layers.no_plot_text");
    assert_dropped(&run.report, NO_PLOT_LINE, "layers.no_plot");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_sheet_still_hides_text_on_an_off_layer() {
    let run = sheet_run("sheet-off");
    assert_dropped(&run.report, SHEET_OFF_TEXT, "layers.off");
    assert_dropped(&run.report, OFF_TEXT, "layers.off");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_frozen_model_text_shows_through_every_viewport() {
    let run = sheet_run("sheet-through");
    for v in DRAWN_VIEWPORTS {
        for h in [FROZEN_TEXT, FROZEN_MTEXT, FROZEN_ATTDEF, NOTE] {
            assert_drawn(&run.report, h, Some(v));
        }
    }
    for v in DRAWN_VIEWPORTS {
        let r = rows(&run.report, FROZEN_LINE, Some(v));
        assert!(
            r.is_empty(),
            "the frozen line is not drawn through {v:#X}: {r:#?}"
        );
    }
    assert_dropped(&run.report, FROZEN_LINE, "layers.frozen");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_viewport_freeze_still_drops_text_in_that_viewport_only() {
    let run = sheet_run("sheet-vp-freeze");
    let a = rows(&run.report, VP_COLD_TEXT, Some(VP_A));
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].fidelity, "IgnoredByPolicy", "{:?}", a[0]);
    assert_eq!(
        a[0].policy_field.as_deref(),
        Some("layers.frozen_in_viewport"),
        "{:?}",
        a[0]
    );
    assert_drawn(&run.report, VP_COLD_TEXT, Some(VP_B));
    assert_drawn(&run.report, VP_COLD_TEXT, Some(VP_NO_PLOT));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_a_viewport_on_a_no_plot_layer_still_shows_its_text() {
    let run = sheet_run("sheet-no-plot-viewport");
    for h in [NOTE, FROZEN_TEXT, BLOCK_ZERO_TEXT] {
        assert_drawn(&run.report, h, Some(VP_NO_PLOT));
    }
    // Its own VIEWPORT entity is what the no-plot layer drops.
    assert_dropped(&run.report, VP_NO_PLOT, "layers.no_plot");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_on_a_sheet_text_in_a_visible_insert_follows_its_own_layer() {
    let run = sheet_run("sheet-block");
    for v in DRAWN_VIEWPORTS {
        for h in [BLOCK_FROZEN_TEXT, BLOCK_ZERO_TEXT, BLOCK_ATTDEF] {
            assert_drawn(&run.report, h, Some(v));
        }
    }
    assert_dropped(&run.report, COLD_INSERT, "layers.frozen");
}

// ---- the same rules on a drawing acadrust wrote ---------------------------------------------

/// Model space holding: a TEXT on a frozen layer and a LINE beside it, a TEXT on Defpoints (no
/// plot), a TEXT on an off layer, and two INSERTs of block `TAG`. `TAG` holds two attribute
/// definitions on layer 0 (VIS, HID) and a TEXT on its own frozen layer. The INSERT on layer 0
/// fills VIS with "SHOWN" and HID with "HIDDEN", which is invisible; the INSERT on the frozen
/// layer fills VIS with "COLDVAL".
fn written_drawing() -> Vec<u8> {
    use acadrust::entities::{AttributeDefinition, AttributeEntity, Insert, Line, Text};
    use acadrust::tables::{BlockRecord, Layer};
    use acadrust::types::Vector3;
    use acadrust::{CadDocument, DwgWriter, EntityType};

    let mut doc = CadDocument::new();
    let mut frozen = Layer::new("FRZ");
    frozen.freeze();
    frozen.handle = doc.allocate_handle();
    doc.layers.add(frozen).unwrap();
    let mut off = Layer::new("OFF");
    off.turn_off();
    off.handle = doc.allocate_handle();
    doc.layers.add(off).unwrap();
    let mut defpoints = Layer::new("Defpoints");
    defpoints.is_plottable = false;
    defpoints.handle = doc.allocate_handle();
    doc.layers.add(defpoints).unwrap();

    let on = |mut e: EntityType, layer: &str| {
        e.common_mut().layer = layer.into();
        e
    };
    let text = |s: &str, y: f64| {
        EntityType::Text(Text::with_value(s, Vector3::new(0.0, y, 0.0)).with_height(1.0))
    };

    let mut record = BlockRecord::new("TAG");
    record.handle = doc.allocate_handle();
    let owner = record.handle;
    doc.block_records.add(record).unwrap();
    for tag in ["VIS", "HID"] {
        let mut d = AttributeDefinition::new(tag.into(), String::new(), "default".into());
        d.height = 1.0;
        let mut e = EntityType::AttributeDefinition(d);
        e.common_mut().owner_handle = owner;
        doc.add_entity(e).unwrap();
    }
    let mut block_note = on(text("BLOCK FROZEN NOTE", 2.0), "FRZ");
    block_note.common_mut().owner_handle = owner;
    doc.add_entity(block_note).unwrap();

    let mut warm = Insert::new("TAG", Vector3::new(100.0, 0.0, 0.0));
    let mut shown = AttributeEntity::new("VIS".into(), "SHOWN".into());
    shown.height = 1.0;
    warm.attributes.push(shown);
    let mut hidden = AttributeEntity::new("HID".into(), "HIDDEN".into());
    hidden.height = 1.0;
    hidden.flags.invisible = true;
    warm.attributes.push(hidden);
    doc.add_entity(EntityType::Insert(warm)).unwrap();

    let mut cold = Insert::new("TAG", Vector3::new(200.0, 0.0, 0.0));
    let mut coldval = AttributeEntity::new("VIS".into(), "COLDVAL".into());
    coldval.height = 1.0;
    cold.attributes.push(coldval);
    doc.add_entity(on(EntityType::Insert(cold), "FRZ")).unwrap();

    doc.add_entity(on(text("FROZEN NOTE", 10.0), "FRZ"))
        .unwrap();
    doc.add_entity(on(
        EntityType::Line(Line::from_coords(0.0, 12.0, 0.0, 9.0, 12.0, 0.0)),
        "FRZ",
    ))
    .unwrap();
    doc.add_entity(on(text("CONSTRUCTION LABEL", 20.0), "Defpoints"))
        .unwrap();
    doc.add_entity(on(text("OFF NOTE", 30.0), "OFF")).unwrap();
    doc.add_entity(text("PLAIN NOTE", 40.0)).unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1621_written_drawing_text_follows_the_layer_rules_in_model_mode() {
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;

    let bytes = written_drawing();
    // TI's rule: the drawing read back has the layer states and the texts the asserts are about.
    let provider = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
    let layers = provider.layers().unwrap();
    let state = |n: &str| {
        layers
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(n))
            .unwrap_or_else(|| panic!("layer {n} read back"))
            .clone()
    };
    assert!(state("FRZ").frozen && state("OFF").off && !state("Defpoints").plottable);
    let model = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let classes: Vec<(String, String)> = provider
        .entities(&EntityScope::Layout(model.id))
        .unwrap()
        .collect_all()
        .unwrap()
        .into_iter()
        .map(|e| (e.provenance.class_name, e.provenance.layer))
        .collect();
    for (class, layer) in [
        ("TEXT", "FRZ"),
        ("TEXT", "Defpoints"),
        ("TEXT", "OFF"),
        ("INSERT", "FRZ"),
        ("LINE", "FRZ"),
    ] {
        assert!(
            classes
                .iter()
                .any(|(c, l)| c == class && l.eq_ignore_ascii_case(layer)),
            "a {class} on {layer} in {classes:?}"
        );
    }
    drop(provider);

    let dir = scratch("written");
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection {
            line_style: true,
            ..StreamSelection::NONE
        })
        .annotations(true)
        .convert()
        .expect("the drawing converts");
    let texts = stream_texts(&dest);
    let _ = fs::remove_dir_all(dir);
    let count = |s: &str| texts.iter().filter(|t| *t == s).count();
    assert_eq!(count("PLAIN NOTE"), 1, "{texts:?}");
    assert_eq!(
        count("FROZEN NOTE"),
        1,
        "text on a frozen layer is drawn: {texts:?}"
    );
    assert_eq!(count("SHOWN"), 1, "{texts:?}");
    assert_eq!(
        count("BLOCK FROZEN NOTE"),
        1,
        "the block's own-frozen TEXT, from the visible INSERT only: {texts:?}"
    );
    for gone in ["CONSTRUCTION LABEL", "OFF NOTE", "HIDDEN", "COLDVAL"] {
        assert_eq!(count(gone), 0, "{gone:?} is not drawn: {texts:?}");
    }
    let frozen_line_drawn = out.report.entities.iter().any(|r| {
        r.source_class == "LINE" && r.fidelity != "IgnoredByPolicy" && r.semantic_type.is_some()
    });
    assert!(
        !frozen_line_drawn,
        "the only LINE is on the frozen layer and is not drawn"
    );
}
