//! A raster of a sheet for the server (#1629, EPIC C): the conversion that writes a PVec also
//! hands back a PNG, sized by the caller, one per print layout and one for model space. Strokes
//! only, black on white. Nothing here compares pixels for equality: it counts dark ones and looks
//! at where they are.
//!
//! The drawings are hand-built. Needs `--features pvec-raster-thumbnail`.

#![cfg(feature = "pvec-raster-thumbnail")]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::thumbnail::thumbnail;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{
    PlotArea, PlotSettings, PvecConversion, PvecConversionOutput, PvecConvertError, PvecLayoutSet,
};

// ---- a hand-built provider ------------------------------------------------------------

/// Model space (layout 0), sheet `A` (layout 1) and sheet `B` (layout 2).
#[derive(Clone)]
struct Mock {
    plots: [LayoutPlot; 2],
    model: Vec<RawEntity>,
    paper: [Vec<RawEntity>; 2],
}

/// The drawing each source key stands for, so a set that opens the drawing once per sheet and
/// tests that run side by side never share one.
fn drawings() -> &'static Mutex<HashMap<Vec<u8>, Mock>> {
    static D: OnceLock<Mutex<HashMap<Vec<u8>, Mock>>> = OnceLock::new();
    D.get_or_init(|| Mutex::new(HashMap::new()))
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
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let CadSource::Bytes(key) = source else {
            panic!("the tests hand over bytes");
        };
        drawings()
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1629",
                detail: "no drawing for this key".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1629"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: OnceLock<ProviderCaps> = OnceLock::new();
        CAPS.get_or_init(|| {
            Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            })
        })
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        let paper = |id: u32, name: &str| LayoutState {
            id: LayoutId(id),
            name: name.into(),
            kind: LayoutKind::Paper,
            block_name: "*Paper_Space".into(),
        };
        Ok(vec![
            LayoutState {
                id: LayoutId(0),
                name: "Model".into(),
                kind: LayoutKind::Model,
                block_name: "*Model_Space".into(),
            },
            paper(1, "A"),
            paper(2, "B"),
        ])
    }
    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        match layout.0 {
            1 | 2 => Ok(SourceFact::Present(
                self.plots[layout.0 as usize - 1].clone(),
            )),
            n => Err(ProviderError::NoSuchScope(format!("layout {n}"))),
        }
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
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => self.model.clone(),
            EntityScope::Layout(LayoutId(1)) => self.paper[0].clone(),
            EntityScope::Layout(LayoutId(2)) => self.paper[1].clone(),
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

fn text(handle: u64) -> RawEntity {
    ent(
        handle,
        "TEXT",
        RawGeometry::Unmapped {
            reason: "text".into(),
        },
    )
}

/// Millimetres of paper, no margins, no origin, no rotation, the layout plotted 1:1.
fn plot(width: f64, height: f64, rotation: i16) -> LayoutPlot {
    LayoutPlot {
        paper_size: "TEST_SHEET".into(),
        paper_width: width,
        paper_height: height,
        paper_units: 1,
        plot_type: 5,
        window: [0.0; 4],
        scale_numerator: 1.0,
        scale_denominator: 1.0,
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

/// Sheet `A` is 100 by 50 mm holding `paper`; sheet `B` is the same paper turned a quarter.
/// Model space holds one line from (0, 0) to (100, 300), three times as tall as it is wide.
fn drawing(key: &str, paper: Vec<RawEntity>) -> Vec<u8> {
    let mock = Mock {
        plots: [plot(100.0, 50.0, 0), plot(100.0, 50.0, 1)],
        model: vec![line(0x90, [0.0, 0.0], [100.0, 300.0])],
        paper: [paper.clone(), paper],
    };
    drawings().lock().unwrap().insert(key.into(), mock);
    key.as_bytes().to_vec()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1629-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Sheet `A` of `paper`, converted under its print profile with a thumbnail `width` wide.
fn sheet_a(
    tag: &str,
    paper: Vec<RawEntity>,
    width: u32,
    settings: PlotSettings,
) -> (Result<PvecConversionOutput, PvecConvertError>, PathBuf) {
    let key = drawing(tag, paper);
    let dir = scratch(tag);
    let dest = dir.join("sheet.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&key), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A"))
        .plot_settings(settings)
        .thumbnail(width)
        .convert_with::<Mock>();
    (out, dir)
}

fn diagonal() -> Vec<RawEntity> {
    vec![line(1, [5.0, 5.0], [95.0, 45.0])]
}

/// The PNG's luma, as (width, height, pixels row by row from the top).
fn decode(png: &[u8]) -> (u32, u32, Vec<u8>) {
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "a PNG");
    let img = image::load_from_memory(png).expect("it decodes").to_luma8();
    (img.width(), img.height(), img.into_raw())
}

fn png_of(out: Result<PvecConversionOutput, PvecConvertError>) -> Vec<u8> {
    out.expect("the conversion works")
        .thumbnail
        .expect("a thumbnail was asked for")
}

fn dark(px: &[u8]) -> usize {
    px.iter().filter(|&&p| p < 128).count()
}

/// Dark pixels in the box `[x0, x1) x [y0, y1)` of a `w` wide image.
fn dark_in(px: &[u8], w: u32, x: (u32, u32), y: (u32, u32)) -> usize {
    (y.0..y.1)
        .flat_map(|r| (x.0..x.1).map(move |c| px[(r * w + c) as usize]))
        .filter(|&p| p < 128)
        .count()
}

fn ratio(w: u32, h: u32) -> f64 {
    f64::from(h) / f64::from(w)
}

// ---- the cases -----------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_200_px_thumbnail_is_a_png_of_the_paper_with_strokes_on_it() {
    let (out, _) = sheet_a("t200", diagonal(), 200, PlotSettings::default());
    let (w, h, px) = decode(&png_of(out));
    assert_eq!(w, 200);
    assert_eq!(h, 100, "the paper is twice as wide as it is tall");
    let n = dark(&px);
    assert!(n > 0, "the line is drawn");
    assert!(
        n * 2 < px.len(),
        "strokes on white, not a black block: {n} of {}",
        px.len()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_the_caller_picks_the_width() {
    for width in [64u32, 200, 512] {
        let (out, _) = sheet_a(
            &format!("t-w{width}"),
            diagonal(),
            width,
            PlotSettings::default(),
        );
        let (w, h, _) = decode(&png_of(out));
        assert_eq!(w, width);
        assert!(
            h.abs_diff(width / 2) <= 1,
            "{width} wide gives {h} high, not {}",
            width / 2
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_width_of_zero_is_refused() {
    let (out, _) = sheet_a("t-zero", diagonal(), 0, PlotSettings::default());
    assert!(out.is_err(), "a zero-wide raster is not a thumbnail");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_the_layouts_rotation_turns_the_aspect() {
    let key = drawing("t-rot", diagonal());
    let dir = scratch("t-rot");
    let out = PvecConversion::new(CadSource::Bytes(&key), &dir.join("b.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("B"))
        .thumbnail(200)
        .convert_with::<Mock>();
    let (w, h, _) = decode(&png_of(out));
    assert_eq!((w, h), (200, 400), "sheet B is the paper turned a quarter");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_paper_override_and_the_extents_area_set_the_aspect() {
    let settings = PlotSettings {
        area: PlotArea::Extents,
        paper_mm: Some([40.0, 40.0]),
        ..PlotSettings::default()
    };
    let (out, _) = sheet_a(
        "t-over",
        vec![line(1, [10.0, 10.0], [30.0, 20.0])],
        120,
        settings,
    );
    let (w, h, px) = decode(&png_of(out));
    assert_eq!(
        (w, h),
        (120, 120),
        "the thumbnail is what gets plotted: a square paper"
    );
    assert!(dark(&px) > 0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_y_is_up_so_a_rising_line_runs_bottom_left_to_top_right() {
    let (out, _) = sheet_a("t-up", diagonal(), 200, PlotSettings::default());
    let (w, h, px) = decode(&png_of(out));
    let (mx, my) = (w / 2, h / 2);
    let top_left = dark_in(&px, w, (0, mx), (0, my));
    let bottom_left = dark_in(&px, w, (0, mx), (my, h));
    let top_right = dark_in(&px, w, (mx, w), (0, my));
    let bottom_right = dark_in(&px, w, (mx, w), (my, h));
    assert!(
        bottom_left > 0 && top_right > 0,
        "the line crosses those two quarters"
    );
    assert_eq!(top_left, 0, "nothing in the top left");
    assert_eq!(bottom_right, 0, "nothing in the bottom right");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_text_only_sheet_is_a_valid_blank_png() {
    let (out, _) = sheet_a(
        "t-text",
        vec![text(1), text(2)],
        200,
        PlotSettings::default(),
    );
    let (w, h, px) = decode(&png_of(out));
    assert_eq!((w, h), (200, 100));
    assert_eq!(dark(&px), 0, "strokes only: no text is drawn");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_hairline_is_at_least_one_pixel_wide_at_any_width() {
    // 80 mm long at y = 25.25 mm, which is the middle of a pixel row at 200 px wide.
    for (width, columns) in [(200u32, 160usize), (64, 48)] {
        let (out, _) = sheet_a(
            &format!("t-hair{width}"),
            vec![line(1, [10.0, 25.25], [90.0, 25.25])],
            width,
            PlotSettings::default(),
        );
        let (_, _, px) = decode(&png_of(out));
        assert!(
            dark(&px) >= columns,
            "{width} px: {} dark pixels for a line {columns} columns long",
            dark(&px)
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_every_print_layout_and_model_space_get_a_thumbnail_of_their_own_shape() {
    let key = drawing("t-set", diagonal());
    let dir = scratch("t-set");
    let set = PvecLayoutSet::new(CadSource::Bytes(&key), &dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters).thumbnail(200))
        .convert_with::<Mock>()
        .expect("the set converts");
    let mut shapes = Vec::new();
    for s in &set.sheets {
        let out = s.outcome.as_ref().expect("the sheet converts");
        let (w, h, px) = decode(out.thumbnail.as_ref().expect("a thumbnail per layout"));
        assert!(dark(&px) > 0, "{} is not blank", s.layout);
        shapes.push((s.layout.clone(), w, h));
    }
    assert_eq!(
        shapes,
        vec![("A".into(), 200, 100), ("B".into(), 200, 400)],
        "one per print layout, each the shape of its paper"
    );

    let model = PvecConversion::new(CadSource::Bytes(&key), &dir.join("model.pvec"))
        .unit(SourceUnit::Millimeters)
        .thumbnail(200)
        .convert_with::<Mock>();
    let (w, h, px) = decode(&png_of(model));
    assert_eq!(w, 200);
    // The extents are 100 by 300, with 2 percent of the longer side (6) added on every side.
    assert!(
        (ratio(w, h) - 312.0 / 112.0).abs() < 0.05,
        "model space is the shape of its extents plus the margin, 112 by 312: {w} x {h}"
    );
    assert!(dark(&px) > 0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_thumbnail_of_a_published_artifact_agrees_with_the_conversions() {
    let (out, _) = sheet_a("t-art", diagonal(), 200, PlotSettings::default());
    let out = out.expect("the conversion works");
    let (w, h, px) = decode(out.thumbnail.as_ref().expect("a thumbnail"));
    let (aw, ah, apx) = decode(&thumbnail(&out.artifact, 200).expect("the artifact draws"));
    assert_eq!((aw, ah), (w, h));
    assert!(dark(&apx) > 0);
    assert!(
        dark(&apx).abs_diff(dark(&px)) * 10 <= dark(&px),
        "the same drawing"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_thumbnail_of_an_artifact_refuses_a_zero_width() {
    let key = drawing("t-art0", diagonal());
    let dir = scratch("t-art0");
    let out = PvecConversion::new(CadSource::Bytes(&key), &dir.join("s.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A"))
        .convert_with::<Mock>()
        .expect("the conversion works");
    assert!(thumbnail(&out.artifact, 0).is_err());
}

/// The model-space drawing's thumbnail, 200 px wide.
fn model_png(tag: &str) -> Vec<u8> {
    let key = drawing(tag, diagonal());
    let dir = scratch(tag);
    let out = PvecConversion::new(CadSource::Bytes(&key), &dir.join("model.pvec"))
        .unit(SourceUnit::Millimeters)
        .thumbnail(200)
        .convert_with::<Mock>();
    png_of(out)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_the_png_is_8_bit_rgb_on_an_opaque_white_background() {
    let (out, _) = sheet_a("t-rgb", vec![text(1)], 200, PlotSettings::default());
    let png = png_of(out);
    let img = image::load_from_memory(&png).expect("it decodes");
    assert_eq!(img.color(), image::ColorType::Rgb8, "no alpha channel");
    assert!(
        img.to_rgb8().as_raw().iter().all(|&c| c == 255),
        "a sheet with nothing to stroke is white all over"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_model_space_keeps_a_white_margin_all_round() {
    let (w, h, px) = decode(&model_png("t-margin"));
    assert!(dark(&px) > 0);
    assert_eq!(dark_in(&px, w, (0, 1), (0, h)), 0, "left column is white");
    assert_eq!(
        dark_in(&px, w, (w - 1, w), (0, h)),
        0,
        "right column is white"
    );
    assert_eq!(dark_in(&px, w, (0, w), (0, 1)), 0, "top row is white");
    assert_eq!(
        dark_in(&px, w, (0, w), (h - 1, h)),
        0,
        "bottom row is white"
    );
    // The line runs from the drawing's corner, which sits 2 percent of 300 mm in from the edge.
    assert!(
        dark_in(&px, w, (0, 20), (h - 20, h)) > 0,
        "and it still starts near the bottom left"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1629_a_paper_layout_has_no_margin_so_its_corner_is_the_paper_edge() {
    let (out, _) = sheet_a(
        "t-edge",
        vec![line(1, [0.0, 0.0], [100.0, 50.0])],
        200,
        PlotSettings::default(),
    );
    let (w, h, px) = decode(&png_of(out));
    assert!(
        dark_in(&px, w, (0, 3), (h - 3, h)) > 0,
        "the paper's bottom left corner is drawn"
    );
    assert!(
        dark_in(&px, w, (w - 3, w), (0, 3)) > 0,
        "the paper's top right corner is drawn"
    );
}
