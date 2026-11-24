//! PMTiles per sheet from a plot (#1676): the plotted page as a raster pyramid, one archive per
//! sheet. The drawings are hand-built, as in #1636, so every frame is pinned exactly. Needs
//! `--features pvec-convert,pdfium` (libpdfium at runtime).

#![cfg(all(feature = "pvec-convert", feature = "pdfium"))]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pmtiles::Reader;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceColour, SourceFact,
    SourceLineweight, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{PmTilesPlot, PmTilesPlotError, PmTilesPlotOutput, RasterLimits};

// ---- a hand-built provider ------------------------------------------------------------

/// A drawing served from memory: model space (layout 0) and paper layouts 1.. in order.
#[derive(Clone)]
struct Mock {
    /// One per paper layout: its name, its plot settings and what it draws.
    sheets: Vec<(String, LayoutPlot, Vec<RawEntity>)>,
}

thread_local! {
    /// The drawing every `open` on this thread hands out (a plot opens the drawing twice).
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
        NEXT.with(|n| n.borrow().clone())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1636",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1636"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: std::sync::OnceLock<ProviderCaps> = std::sync::OnceLock::new();
        CAPS.get_or_init(caps)
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        let mut v = vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }];
        for (i, (name, _, _)) in self.sheets.iter().enumerate() {
            v.push(LayoutState {
                id: LayoutId(i as u32 + 1),
                name: name.clone(),
                kind: LayoutKind::Paper,
                block_name: format!("*Paper_Space{i}"),
            });
        }
        Ok(v)
    }
    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        match self.sheets.get((layout.0 as usize).wrapping_sub(1)) {
            Some((_, p, _)) => Ok(SourceFact::Present(p.clone())),
            None => Err(ProviderError::NoSuchScope(format!("layout {}", layout.0))),
        }
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            ..LayerState::default()
        }])
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(0)) => vec![],
            EntityScope::Layout(LayoutId(n)) => self
                .sheets
                .get(*n as usize - 1)
                .map(|s| s.2.clone())
                .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {n}")))?,
            other => return Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        };
        Ok(Box::new(VecCursor(list.into_iter())))
    }
}

/// A line of `weight` hundredths of a millimetre in AutoCAD colour `aci`.
fn line(handle: u64, a: [f64; 2], b: [f64; 2], aci: u8, weight: u16) -> RawEntity {
    RawEntity {
        style: EntityStyle {
            colour: SourceFact::Present(SourceColour::Aci(aci)),
            lineweight: SourceFact::Present(SourceLineweight::Hundredths(weight)),
            ..EntityStyle::absent()
        },
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

/// Paper of `width` by `height` millimetres, nothing else off its default.
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

fn one_sheet(p: LayoutPlot, paper: Vec<RawEntity>) -> Mock {
    Mock {
        sheets: vec![("Sheet".into(), p, paper)],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1636-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn horizontal() -> Mock {
    one_sheet(
        plot(100.0, 50.0, 0),
        vec![line(1, [10.0, 25.0], [90.0, 25.0], 7, 100)],
    )
}

/// 200 x 100 mm: at 254 dpi that is 2000 x 1000 px, four tiles across at 256.
fn wide() -> Mock {
    one_sheet(
        plot(200.0, 100.0, 0),
        vec![line(1, [20.0, 50.0], [180.0, 50.0], 7, 100)],
    )
}

fn plot_tiles(
    tag: &str,
    mock: Mock,
    configure: impl for<'a> FnOnce(PmTilesPlot<'a>) -> PmTilesPlot<'a>,
) -> (Result<PmTilesPlotOutput, PmTilesPlotError>, PathBuf) {
    let dir = scratch(tag);
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let plotter = PmTilesPlot::new(CadSource::Bytes(b"mock"), &dir)
        .stem("drawing")
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"));
    let out = configure(plotter).convert_with::<Mock>();
    (out, dir)
}

fn files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// The RGB of pixel (x, y) of a tile's PNG.
fn tile_pixel(archive: &Path, z: u8, x: u32, y: u32, px: u32, py: u32) -> [u8; 3] {
    let reader = Reader::try_open(archive).unwrap();
    let png = reader
        .get_tile(z, x, y)
        .unwrap()
        .expect("the tile is there");
    let img = image::load_from_memory(&png).unwrap().to_rgb8();
    img.get_pixel(px, py).0
}

fn is_dark(p: [u8; 3]) -> bool {
    p.iter().all(|&c| c < 160)
}

fn is_white(p: [u8; 3]) -> bool {
    p.iter().all(|&c| c > 200)
}

// ---- the archive ----------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_one_archive_per_sheet_named_by_its_position_and_layout() {
    let l = |h| vec![line(h, [10.0, 25.0], [90.0, 25.0], 7, 100)];
    let mock = Mock {
        sheets: vec![
            ("A Plan".into(), plot(100.0, 50.0, 0), l(1)),
            ("B".into(), plot(100.0, 50.0, 0), l(2)),
        ],
    };
    let (out, dir) = plot_tiles("two", mock, |p| p.layouts(["B", "a plan"]).dpi(127));
    let out = out.unwrap();
    assert_eq!(out.sheets.len(), 2);
    assert_eq!(out.sheets[0].layout, "A Plan");
    assert_eq!(out.sheets[1].layout, "B");
    assert_eq!(
        files(&dir),
        ["drawing.01-a-plan.pmtiles", "drawing.02-b.pmtiles"],
        "one archive per sheet and no scratch left behind"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_the_header_is_a_png_pyramid_of_the_sheet() {
    let (out, dir) = plot_tiles("header", wide(), |p| p.dpi(254));
    let sheet = out.unwrap().sheets.remove(0);
    let reader = Reader::try_open(&sheet.file).unwrap();
    let h = reader.header();
    assert_eq!(h.tile_type, libviprs::pmtiles::TileType::Png);
    assert_eq!(h.addressed_tiles_count, sheet.tiles);
    assert_eq!((sheet.width_px, sheet.height_px), (2000, 1000));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_the_zoom_range_runs_from_one_tile_to_the_full_resolution() {
    let (out, dir) = plot_tiles("zoom", wide(), |p| p.dpi(254));
    let sheet = out.unwrap().sheets.remove(0);
    // The pyramid halves down to one pixel: 2000 px sits on a 2048 canvas, so zoom 0 (one pixel,
    // one tile) to zoom 11 (eight tiles across).
    assert_eq!((sheet.min_zoom, sheet.max_zoom), (0, 11));
    let reader = Reader::try_open(&sheet.file).unwrap();
    assert_eq!((reader.min_zoom(), reader.max_zoom()), (0, 11));
    assert!(reader.tile_exists(0, 0, 0).unwrap());
    assert!(reader.tile_exists(11, 7, 3).unwrap());
    assert!(!reader.tile_exists(12, 0, 0).unwrap());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_sample_tile_has_the_line_where_the_sheet_puts_it() {
    // 100 x 50 mm at 127 dpi is 500 x 250 px on a 512 canvas, so two tiles across at zoom 9. The line is at
    // 25 mm from the bottom, which is 125 px from the top, from x = 50 px to x = 450 px.
    let (out, dir) = plot_tiles("pixels", horizontal(), |p| p.dpi(127));
    let sheet = out.unwrap().sheets.remove(0);
    assert_eq!(sheet.max_zoom, 9);
    // A 0.25 mm pen is 1.25 px wide at 127 dpi, so the line is anti-aliased over rows 124 and 125.
    let on_line = |y| tile_pixel(&sheet.file, 9, 0, 0, 100, y);
    assert!(is_dark(on_line(124)) || is_dark(on_line(125)));
    assert!(is_white(tile_pixel(&sheet.file, 9, 0, 0, 100, 60)));
    assert!(is_white(tile_pixel(&sheet.file, 9, 0, 0, 20, 125)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_stem_that_is_not_a_plain_file_name_is_refused_and_touches_nothing() {
    for stem in ["../x", "", "a/b", "a\\b", ".", "..", "a\0b", "/abs"] {
        let base = scratch("stem");
        let out = base.join("out");
        // What a "../x" stem would reach: the scratch directory's name beside the destination.
        let bystander = base.join("x.plot-work");
        fs::create_dir_all(&bystander).unwrap();
        fs::write(bystander.join("keep"), b"keep").unwrap();
        NEXT.with(|n| *n.borrow_mut() = Some(horizontal()));
        let err = PmTilesPlot::new(CadSource::Bytes(b"mock"), &out)
            .stem(stem)
            .unit(SourceUnit::Millimeters)
            .view_profile(ViewProfile::print_profile("Sheet"))
            .convert_with::<Mock>()
            .unwrap_err();
        assert!(
            matches!(err, PmTilesPlotError::BadStem(_)),
            "{stem:?}: {err:?}"
        );
        assert_eq!(err.code(), "PLOT_STEM");
        assert_eq!(
            fs::read(bystander.join("keep")).unwrap(),
            b"keep",
            "{stem:?}"
        );
        assert_eq!(files(&base), ["x.plot-work"], "{stem:?}");
        assert!(
            !out.exists(),
            "{stem:?}: the refusal came before anything was made"
        );
        let _ = fs::remove_dir_all(base);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_plot_that_fails_leaves_an_archive_that_was_already_there() {
    let l = |h| vec![line(h, [10.0, 25.0], [90.0, 25.0], 7, 100)];
    let mock = Mock {
        sheets: vec![
            ("A".into(), plot(100.0, 50.0, 0), l(1)),
            ("B".into(), plot(100.0, 50.0, 0), l(2)),
        ],
    };
    let dir = scratch("keep-old");
    fs::write(dir.join("drawing.01-a.pmtiles"), b"keep").unwrap();
    // The second sheet's archive cannot be written: a directory has its name.
    fs::create_dir(dir.join("drawing.02-b.pmtiles")).unwrap();
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let err = PmTilesPlot::new(CadSource::Bytes(b"mock"), &dir)
        .stem("drawing")
        .unit(SourceUnit::Millimeters)
        .layouts(["A", "B"])
        .dpi(127)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PmTilesPlotError::Io(_)), "{err:?}");
    assert_eq!(fs::read(dir.join("drawing.01-a.pmtiles")).unwrap(), b"keep");
    assert_eq!(
        files(&dir),
        ["drawing.01-a.pmtiles", "drawing.02-b.pmtiles"]
    );
    let _ = fs::remove_dir_all(dir);
}

// ---- hostile input --------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_page_longer_than_the_limit_is_refused_before_anything_is_drawn() {
    let huge = one_sheet(plot(10_000.0, 100.0, 0), vec![]);
    let (out, dir) = plot_tiles("page", huge, |p| p.dpi(72));
    let err = out.unwrap_err();
    assert!(
        matches!(err, PmTilesPlotError::PageTooLarge { .. }),
        "{err:?}"
    );
    assert_eq!(err.code(), "PLOT_PAGE_TOO_LARGE");
    assert!(files(&dir).is_empty(), "a refusal leaves nothing behind");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_page_that_renders_to_too_many_pixels_is_refused_before_it_is_rendered() {
    // 100 x 50 mm at 600 dpi is 2362 x 1181 px, about 2.8 million pixels.
    let (out, dir) = plot_tiles("pixels-limit", horizontal(), |p| {
        p.dpi(600).limits(RasterLimits {
            max_pixels: 1_000_000,
            ..RasterLimits::default()
        })
    });
    let err = out.unwrap_err();
    assert!(
        matches!(err, PmTilesPlotError::TooManyPixels { max: 1_000_000, .. }),
        "{err:?}"
    );
    assert_eq!(err.code(), "PLOT_PIXELS");
    assert!(files(&dir).is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_dpi_outside_the_range_is_refused() {
    for dpi in [0, 35, 601, u32::MAX] {
        let (out, dir) = plot_tiles("dpi", horizontal(), |p| p.dpi(dpi));
        let err = out.unwrap_err();
        assert!(
            matches!(err, PmTilesPlotError::DpiOutOfRange(d) if d == dpi),
            "{err:?}"
        );
        assert_eq!(err.code(), "PLOT_DPI_OUT_OF_RANGE");
        assert!(files(&dir).is_empty());
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_more_zoom_levels_than_the_limit_is_refused() {
    let (out, dir) = plot_tiles("levels", wide(), |p| {
        p.dpi(254).limits(RasterLimits {
            max_zoom_levels: 3,
            ..RasterLimits::default()
        })
    });
    let err = out.unwrap_err();
    assert!(
        matches!(
            err,
            PmTilesPlotError::TooManyZoomLevels {
                levels: 12,
                max: 3,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(err.code(), "PLOT_ZOOM_LEVELS");
    assert!(files(&dir).is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_more_tiles_than_the_limit_is_refused() {
    let (out, dir) = plot_tiles("tiles", wide(), |p| {
        p.dpi(254).limits(RasterLimits {
            max_tiles: 10,
            ..RasterLimits::default()
        })
    });
    let err = out.unwrap_err();
    assert!(
        matches!(err, PmTilesPlotError::TooManyTiles { max: 10, .. }),
        "{err:?}"
    );
    assert_eq!(err.code(), "PLOT_TILE_COUNT");
    assert!(files(&dir).is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_an_unknown_layout_is_the_plots_own_refusal() {
    let (out, dir) = plot_tiles("unknown", horizontal(), |p| p.layouts(["Nope"]));
    let err = out.unwrap_err();
    assert!(matches!(err, PmTilesPlotError::Plot(_)), "{err:?}");
    assert_eq!(err.code(), "LAYOUT_NOT_FOUND");
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_cancelled_plot_writes_nothing() {
    let token = libviprs::CancelToken::new();
    token.cancel();
    let (out, dir) = plot_tiles("cancel", horizontal(), |p| p.cancel(token));
    let err = out.unwrap_err();
    assert!(matches!(err, PmTilesPlotError::Cancelled), "{err:?}");
    assert_eq!(err.code(), "CANCELLED");
    assert!(files(&dir).is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1676_a_spent_deadline_writes_nothing() {
    let (out, dir) = plot_tiles("deadline", horizontal(), |p| {
        p.deadline(std::time::Duration::ZERO)
    });
    let err = out.unwrap_err();
    assert!(matches!(err, PmTilesPlotError::Deadline), "{err:?}");
    assert_eq!(err.code(), "DEADLINE");
    assert!(files(&dir).is_empty());
    let _ = fs::remove_dir_all(dir);
}
