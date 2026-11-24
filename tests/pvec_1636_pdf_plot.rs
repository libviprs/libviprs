//! PDF Printer (#1636): one print layout written as one PDF page, paper, plot area, offset and
//! rotation applied. The drawings are hand-built (the issue names `cad-import-plot.dxf`, which does
//! not exist, and the acadrust provider reads DWG only), so every frame is pinned exactly.
//!
//! Page sizes are read back with `pdf_info`. Pixels are sampled from a pdfium render, in the
//! `rendered` module behind the `pdfium` feature (libpdfium at runtime). Needs `--features
//! pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceColour, SourceFact,
    SourceLineweight, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{PdfPlot, PdfPlotOutput, PlotSettings, PvecConvertError};

// ---- a hand-built provider ------------------------------------------------------------

/// A drawing served from memory: model space (layout 0) and paper layouts 1.. in order.
#[derive(Clone)]
struct Mock {
    /// One per paper layout: its name, its plot settings and what it draws.
    sheets: Vec<(String, LayoutPlot, Vec<RawEntity>)>,
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

/// Plots `mock` to `<dir>/sheet.pdf` after `configure`; the profile names layout `Sheet`.
fn plot_pdf(
    tag: &str,
    mock: Mock,
    configure: impl for<'a> FnOnce(PdfPlot<'a>) -> PdfPlot<'a>,
) -> (Result<PdfPlotOutput, PvecConvertError>, PathBuf) {
    let dir = scratch(tag);
    let dest = dir.join("sheet.pdf");
    NEXT.with(|n| *n.borrow_mut() = Some(mock));
    let plotter = PdfPlot::new(CadSource::Bytes(b"mock"), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"));
    let out = configure(plotter).convert_with::<Mock>();
    (out, dir)
}

const PT_PER_MM: f64 = 72.0 / 25.4;

fn page_sizes(pdf: &Path) -> Vec<(f64, f64)> {
    libviprs::pdf_info(pdf)
        .unwrap()
        .pages
        .iter()
        .map(|p| (p.width_pts, p.height_pts))
        .collect()
}

fn near(got: (f64, f64), w_mm: f64, h_mm: f64) {
    assert!(
        (got.0 - w_mm * PT_PER_MM).abs() < 0.1 && (got.1 - h_mm * PT_PER_MM).abs() < 0.1,
        "{got:?} pt is not {w_mm} x {h_mm} mm"
    );
}

fn horizontal() -> Mock {
    one_sheet(
        plot(100.0, 50.0, 0),
        vec![line(1, [10.0, 25.0], [90.0, 25.0], 7, 100)],
    )
}

// ---- page size, rotation, layouts, producer, text -------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_pdf_plot_page_size_is_the_paper() {
    let (out, dir) = plot_pdf("size", horizontal(), |p| p);
    let out = out.unwrap();
    assert_eq!(out.pages, 1);
    let sizes = page_sizes(&out.file);
    assert_eq!(sizes.len(), 1);
    near(sizes[0], 100.0, 50.0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_a_paper_override_sets_the_page_size() {
    let (out, dir) = plot_pdf("override", horizontal(), |p| {
        p.plot_settings(PlotSettings {
            paper_mm: Some([210.0, 297.0]),
            ..PlotSettings::default()
        })
    });
    near(page_sizes(&out.unwrap().file)[0], 210.0, 297.0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_rotation_one_swaps_the_page_width_and_height() {
    let mock = one_sheet(
        plot(100.0, 50.0, 1),
        vec![line(1, [10.0, 25.0], [90.0, 25.0], 7, 100)],
    );
    let (out, dir) = plot_pdf("rotated", mock, |p| p);
    near(page_sizes(&out.unwrap().file)[0], 50.0, 100.0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_one_page_per_layout_in_tab_order() {
    let l = |h| vec![line(h, [10.0, 25.0], [90.0, 25.0], 7, 100)];
    let mock = Mock {
        sheets: vec![
            ("A".into(), plot(100.0, 50.0, 0), l(1)),
            ("B".into(), plot(200.0, 100.0, 0), l(2)),
        ],
    };
    let (out, dir) = plot_pdf("set", mock, |p| p.layouts(["A", "B"]));
    let out = out.unwrap();
    assert_eq!(out.pages, 2);
    let sizes = page_sizes(&out.file);
    assert_eq!(sizes.len(), 2);
    near(sizes[0], 100.0, 50.0);
    near(sizes[1], 200.0, 100.0);
    let _ = fs::remove_dir_all(dir);
}

/// A paper override is said in the output's notes, in the sheet report's words.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_notes_carry_what_was_replaced() {
    let (out, dir) = plot_pdf("notes", horizontal(), |p| {
        p.plot_settings(PlotSettings {
            paper_mm: Some([210.0, 297.0]),
            ..PlotSettings::default()
        })
    });
    let notes = out.unwrap().notes;
    assert!(
        notes.contains(&"plot-paper-overridden".to_string()),
        "{notes:?}"
    );
    let _ = fs::remove_dir_all(dir);
}

/// The document information names libviprs as the producer.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_the_producer_names_libviprs() {
    let (out, dir) = plot_pdf("producer", horizontal(), |p| p);
    let doc = lopdf::Document::load(out.unwrap().file).unwrap();
    let info = doc
        .trailer
        .get(b"Info")
        .and_then(|o| doc.dereference(o))
        .map(|(_, o)| o.clone())
        .expect("an /Info entry");
    let producer = info
        .as_dict()
        .unwrap()
        .get(b"Producer")
        .and_then(|o| o.as_str())
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .expect("a /Producer string");
    assert!(producer.contains("libviprs"), "{producer}");
    let _ = fs::remove_dir_all(dir);
}

/// Text stays out until #1621 settles widths: no text operator, no font on the page.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1636_no_text_is_written() {
    let (out, dir) = plot_pdf("notext", horizontal(), |p| p);
    let doc = lopdf::Document::load(out.unwrap().file).unwrap();
    let (_, page) = doc.get_pages().into_iter().next().expect("one page");
    let content = doc.get_page_content(page).unwrap();
    let text = String::from_utf8_lossy(&content);
    assert!(!text.split_whitespace().any(|t| t == "BT"), "{text}");
    assert!(doc.get_page_fonts(page).unwrap_or_default().is_empty());
    // And the page does draw something: a stroke operator.
    assert!(text.split_whitespace().any(|t| t == "S"), "{text}");
    let _ = fs::remove_dir_all(dir);
}

// ---- pixels ---------------------------------------------------------------------------------

#[cfg(feature = "pdfium")]
mod rendered {
    use super::*;
    use acadrust::plot_style::PlotStyleTable;
    use libviprs::pvec_convert::PlotArea;

    const DPI: u32 = 144;
    const PX_PER_MM: f64 = DPI as f64 / 25.4;

    /// The RGB of the pixel at (`x_mm`, `y_mm`) from the page's lower left corner.
    fn pixel(pdf: &Path, x_mm: f64, y_mm: f64) -> [u8; 3] {
        let r = libviprs::render_page_pdfium(pdf, 1, DPI).unwrap();
        let bpp = r.data().len() / (r.width() as usize * r.height() as usize);
        let x = ((x_mm * PX_PER_MM) as usize).min(r.width() as usize - 1);
        let y = (r.height() as f64 - y_mm * PX_PER_MM) as usize;
        let at = (y.min(r.height() as usize - 1) * r.width() as usize + x) * bpp;
        [r.data()[at], r.data()[at + 1], r.data()[at + 2]]
    }

    fn is_dark(p: [u8; 3]) -> bool {
        p.iter().all(|&c| c < 80)
    }

    fn is_white(p: [u8; 3]) -> bool {
        p.iter().all(|&c| c > 200)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1636_the_line_is_where_the_sheet_puts_it() {
        let (out, dir) = plot_pdf("where", horizontal(), |p| p);
        let pdf = out.unwrap().file;
        assert!(is_dark(pixel(&pdf, 50.0, 25.0)));
        assert!(is_white(pixel(&pdf, 50.0, 10.0)));
        assert!(is_white(pixel(&pdf, 5.0, 25.0)));
        let _ = fs::remove_dir_all(dir);
    }

    /// A plot style table that maps colour 1 to blue and a 1 mm pen, though the entity asks for
    /// red and 0.1 mm.
    fn blue_one_mm() -> PlotStyleTable {
        let mut table = PlotStyleTable::identity("blue.ctb");
        let pen = table
            .lineweights
            .iter()
            .position(|w| (*w - 1.0).abs() < 1e-6)
            .expect("a 1.00 mm pen") as u8
            + 1; // a style's lineweight is one more than the list's index (#1692)
        table.aci_entries[1].color = Some([0, 0, 255]);
        table.aci_entries[1].lineweight = pen;
        table
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1636_pdf_plot_applies_ctb_width_and_colour() {
        let mock = one_sheet(
            plot(100.0, 50.0, 0),
            vec![line(1, [10.0, 25.0], [90.0, 25.0], 1, 10)],
        );
        let (out, dir) = plot_pdf("ctb", mock, |p| p.plot_style_table(blue_one_mm()));
        let pdf = out.unwrap().file;
        let on = pixel(&pdf, 50.0, 25.0);
        assert!(
            on[0] < 60 && on[1] < 60 && on[2] > 200,
            "centre is blue: {on:?}"
        );
        let edge = pixel(&pdf, 50.0, 25.35);
        assert!(
            edge[2] > 200 && edge[0] < 60,
            "0.35 mm off is inside 1 mm: {edge:?}"
        );
        assert!(
            is_white(pixel(&pdf, 50.0, 26.5)),
            "1.5 mm off is outside it"
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// Screening 50 per cent lightens the stroke to a mid grey.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1636_pdf_plot_applies_ctb_screening() {
        let mock = one_sheet(
            plot(100.0, 50.0, 0),
            vec![line(1, [10.0, 25.0], [90.0, 25.0], 7, 100)],
        );
        let mut table = PlotStyleTable::identity("half.ctb");
        table.aci_entries[7].color = Some([0, 0, 0]);
        table.aci_entries[7].screening = 50;
        let (out, dir) = plot_pdf("screen", mock, |p| p.plot_style_table(table));
        let on = pixel(&out.unwrap().file, 50.0, 25.0);
        assert!(
            on.iter().all(|&c| (100..200).contains(&c)),
            "mid grey: {on:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1636_the_offset_moves_the_geometry_on_the_page() {
        let mock = one_sheet(
            plot(100.0, 50.0, 0),
            vec![line(1, [10.0, 25.0], [30.0, 25.0], 7, 100)],
        );
        let (out, dir) = plot_pdf("offset", mock, |p| {
            p.plot_settings(PlotSettings {
                offset_mm: Some([20.0, 0.0]),
                ..PlotSettings::default()
            })
        });
        let pdf = out.unwrap().file;
        assert!(
            is_dark(pixel(&pdf, 40.0, 25.0)),
            "the line now covers 30 to 50"
        );
        assert!(is_white(pixel(&pdf, 20.0, 25.0)), "where it was is empty");
        let _ = fs::remove_dir_all(dir);
    }

    /// Geometry 500 mm from the origin is off the layout's frame, and on the page when the plot
    /// area is the extents: the paper is placed at the extents' lower left.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1636_pdf_plot_extents_area_frames_geometry() {
        let far = vec![
            line(1, [500.0, 500.0], [500.0, 501.0], 7, 100),
            line(2, [505.0, 505.0], [515.0, 505.0], 7, 100),
        ];
        let (out, dir) = plot_pdf(
            "extents",
            one_sheet(plot(100.0, 50.0, 0), far.clone()),
            |p| {
                p.plot_settings(PlotSettings {
                    area: PlotArea::Extents,
                    ..PlotSettings::default()
                })
            },
        );
        let pdf = out.unwrap().file;
        near(page_sizes(&pdf)[0], 100.0, 50.0);
        assert!(
            is_dark(pixel(&pdf, 10.0, 5.0)),
            "the far line is 5 mm up and 5 to 15 across"
        );
        let _ = fs::remove_dir_all(dir);
        let (out, dir) = plot_pdf("layout-area", one_sheet(plot(100.0, 50.0, 0), far), |p| p);
        assert!(
            is_white(pixel(&out.unwrap().file, 10.0, 5.0)),
            "the layout frame misses it"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
