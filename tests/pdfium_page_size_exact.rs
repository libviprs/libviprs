//! Issue #1199: a PDF page rendered at a DPI comes out at the size libvips
//! gives, `rint(points * (dpi / 72.0))` per axis, whichever route renders it.
//!
//! The pages are blank lopdf-built PDFs with a black ring around the edge, so
//! the same file drives the size checks and the edge checks. The expected size
//! is written out here from the libvips rule and does not go through
//! `PageSizing`, so a mistake in that type cannot hide behind itself.
//!
//! The renders need a libpdfium at runtime (`PDFIUM_PATH` or the system
//! library), the same as the rest of the pdfium suites.
#![cfg(feature = "pdfium")]

use std::path::{Path, PathBuf};

use libviprs::{
    PdfiumStripSource, Raster, StripSource, render_page_pdfium, render_page_pdfium_budgeted,
};

/// Page sizes in points: name, width, height.
const SIZES: &[(&str, f64, f64)] = &[
    ("Letter", 612.0, 792.0),
    ("Tabloid", 792.0, 1224.0),
    ("Tabloid landscape", 1224.0, 792.0),
    ("ARCH A", 648.0, 864.0),
    ("ARCH D", 1728.0, 2592.0),
    ("ARCH E", 2592.0, 3456.0),
    ("ANSI D", 1584.0, 2448.0),
    ("A4", 595.276, 841.89),
    ("A3", 841.89, 1190.551),
    ("A2", 1190.551, 1683.78),
    ("A1", 1683.78, 2383.937),
    ("A0", 2383.937, 3370.394),
];

const DPIS: [u32; 5] = [72, 96, 150, 300, 600];

/// Largest raster a test renders in full. Bigger pages are still checked
/// through the streaming source, which reports its size without rendering.
const FULL_RENDER_PIXEL_CAP: u64 = 25_000_000;

/// One-page PDF: a black page with a white inset, so a 12 pt black ring runs
/// round the edge. `rotate` is the page's `/Rotate`, `origin` the MediaBox's
/// lower-left corner.
fn write_pdf(dir: &Path, name: &str, w: f64, h: f64, rotate: i64, origin: (f64, f64)) -> PathBuf {
    use lopdf::{Document, Object, Stream, dictionary};

    let (ox, oy) = origin;
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let content = format!(
        "0 g {ox} {oy} {w} {h} re f 1 g {} {} {} {} re f",
        ox + 12.0,
        oy + 12.0,
        w - 24.0,
        h - 24.0
    );
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![
            Object::Real(ox as f32),
            Object::Real(oy as f32),
            Object::Real((ox + w) as f32),
            Object::Real((oy + h) as f32),
        ],
        "Contents" => content_id,
        "Resources" => dictionary! {},
    };
    if rotate != 0 {
        page.set("Rotate", Object::Integer(rotate));
    }
    let page_id = doc.add_object(page);
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1i64,
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let path = dir.join(name);
    doc.save(&path).unwrap();
    path
}

/// The libvips rule on the page size pdfium reports, which is the MediaBox
/// narrowed to `f32`.
fn libvips_dims(w_pt: f64, h_pt: f64, dpi: u32) -> (u32, u32) {
    let scale = dpi as f64 / 72.0;
    let (w, h) = (w_pt as f32 as f64, h_pt as f32 as f64);
    (
        (w * scale).round_ties_even() as u32,
        (h * scale).round_ties_even() as u32,
    )
}

fn px(r: &Raster, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * r.width() + x) * 4) as usize;
    r.data()[i..i + 4].try_into().unwrap()
}

fn is_dark(p: [u8; 4]) -> bool {
    p[0] < 128 && p[1] < 128 && p[2] < 128
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_render_route_reports_the_libvips_size() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong: Vec<String> = Vec::new();
    for &(name, w, h) in SIZES {
        let pdf = write_pdf(dir.path(), "page.pdf", w, h, 0, (0.0, 0.0));
        for dpi in DPIS {
            let want = libvips_dims(w, h, dpi);
            let ctx = format!("{name} at {dpi} dpi, want {want:?}");

            let streaming = PdfiumStripSource::new_streaming(&pdf, 1, dpi).unwrap();
            let got = (streaming.width(), streaming.height());
            if got != want {
                wrong.push(format!("{ctx}: new_streaming reports {got:?}"));
            }

            if u64::from(want.0) * u64::from(want.1) > FULL_RENDER_PIXEL_CAP {
                continue;
            }

            let cached = render_page_pdfium(&pdf, 1, dpi).unwrap();
            let got = (cached.width(), cached.height());
            if got != want {
                wrong.push(format!("{ctx}: render_page_pdfium renders {got:?}"));
            }

            let budgeted = render_page_pdfium_budgeted(&pdf, 1, dpi, u64::MAX / 4).unwrap();
            let got = (budgeted.raster.width(), budgeted.raster.height());
            if got != want || budgeted.dpi_used != dpi || budgeted.capped {
                wrong.push(format!(
                    "{ctx}: render_page_pdfium_budgeted renders {got:?} at {} dpi",
                    budgeted.dpi_used
                ));
            }

            let source = PdfiumStripSource::new(&pdf, 1, dpi).unwrap();
            let got = (source.width(), source.height());
            if got != want {
                wrong.push(format!("{ctx}: PdfiumStripSource::new reports {got:?}"));
            }
            let strip = source.render_strip(0, 8).unwrap();
            if strip.width() != want.0 {
                wrong.push(format!("{ctx}: cached strip is {} px wide", strip.width()));
            }
            let strip = streaming.render_strip(0, 8).unwrap();
            if strip.width() != want.0 || strip.height() != 8 {
                wrong.push(format!(
                    "{ctx}: streaming strip is {}x{}",
                    strip.width(),
                    strip.height()
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} mismatches:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A page whose size is not a whole number of points is stretched into the
/// bitmap edge to edge: the black ring is on the first and last row and
/// column of the cached render, and a streaming strip of the last rows has the
/// same ring and the same pixels as those rows of the cached render. A white
/// band on the right or bottom edge is what `FPDF_RenderPageBitmapWithMatrix`
/// does when its scale ignores that the page box is truncated to whole points.
#[test]
#[cfg_attr(miri, ignore)]
fn a_fractional_page_fills_the_bitmap_to_the_edge_in_every_route() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write_pdf(dir.path(), "a3.pdf", 841.89, 1190.551, 0, (0.0, 0.0));
    let dpi = 150;
    let (w, h) = libvips_dims(841.89, 1190.551, dpi);
    assert_eq!((w, h), (1754, 2480));

    let cached = render_page_pdfium(&pdf, 1, dpi).unwrap();
    assert_eq!((cached.width(), cached.height()), (w, h));
    for x in [0, w / 2, w - 1] {
        assert!(is_dark(px(&cached, x, 0)), "cached first row at x={x}");
        assert!(is_dark(px(&cached, x, h - 1)), "cached last row at x={x}");
    }
    for y in [0, h / 2, h - 1] {
        assert!(is_dark(px(&cached, 0, y)), "cached first column at y={y}");
        assert!(
            is_dark(px(&cached, w - 1, y)),
            "cached last column at y={y}"
        );
    }

    let streaming = PdfiumStripSource::new_streaming(&pdf, 1, dpi).unwrap();
    assert_eq!((streaming.width(), streaming.height()), (w, h));
    for (y0, rows) in [(h - 16, 16), (h / 2, 32), (0, 16)] {
        let strip = streaming.render_strip(y0, rows).unwrap();
        assert_eq!((strip.width(), strip.height()), (w, rows));
        let want = cached.extract(0, y0, w, rows).unwrap();
        assert_eq!(
            strip.data(),
            want.data(),
            "streaming rows {y0}..{} differ from the cached render",
            y0 + rows
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn rotation_swaps_the_axes_before_the_size_is_rounded() {
    let dir = tempfile::tempdir().unwrap();
    for rotate in [90, 270] {
        let pdf = write_pdf(dir.path(), "r.pdf", 612.0, 792.0, rotate, (0.0, 0.0));
        let want = libvips_dims(792.0, 612.0, 300);
        assert_eq!(want, (3300, 2550));
        let cached = render_page_pdfium(&pdf, 1, 300).unwrap();
        assert_eq!((cached.width(), cached.height()), want, "rotate {rotate}");
        let streaming = PdfiumStripSource::new_streaming(&pdf, 1, 300).unwrap();
        assert_eq!(
            (streaming.width(), streaming.height()),
            want,
            "rotate {rotate}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_media_box_away_from_the_origin_has_the_same_size() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write_pdf(dir.path(), "o.pdf", 612.0, 792.0, 0, (100.0, 50.0));
    let cached = render_page_pdfium(&pdf, 1, 300).unwrap();
    assert_eq!((cached.width(), cached.height()), (2550, 3300));
    assert!(is_dark(px(&cached, 0, 0)));
    assert!(is_dark(px(&cached, 2549, 3299)));
    let streaming = PdfiumStripSource::new_streaming(&pdf, 1, 300).unwrap();
    assert_eq!((streaming.width(), streaming.height()), (2550, 3300));
}
