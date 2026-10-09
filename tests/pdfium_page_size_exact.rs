//! Issue #1199: a PDF page rendered at a DPI comes out at the size libvips
//! gives, `rint(points * (dpi / 72.0))` per axis, whichever route renders it.
//!
//! The pages are lopdf-built PDFs with real content (border, grid, shapes, text,
//! a triangle in the top-left corner only), so the same file drives the size
//! checks and the edge and landmark checks. The size checks never look at the
//! ink. The expected size
//! is written out here from the libvips rule and does not go through
//! `PageSizing`, so a mistake in that type cannot hide behind itself.
//!
//! The renders need a libpdfium at runtime (`PDFIUM_PATH` or the system
//! library), the same as the rest of the pdfium suites.
#![cfg(feature = "pdfium")]

use std::path::{Path, PathBuf};

use libviprs::{
    PageSizing, PdfiumStripSource, Raster, StripSource, render_page_pdfium,
    render_page_pdfium_budgeted,
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

/// Side of the solid triangle in the top-left corner, in points: the one
/// asymmetric landmark, so a shift or a flip puts the wrong pixel in the ink.
fn tri_pt(w: f64, h: f64) -> f64 {
    (0.1 * w.min(h)).max(30.0).min(w.min(h) / 3.0)
}

/// Drawing for a `w` x `h` page with its lower-left corner at the origin:
/// a 2 pt border flush with the page box, a grid, a few dozen vector shapes
/// (rects, bezier circles, hatching, dashed lines), a title block with base-14
/// text at several sizes, and the top-left triangle. Deterministic, and sized
/// from the page so the drawing is the same shape on every sheet.
fn rich_content(w: f64, h: f64) -> String {
    use std::fmt::Write as _;
    let short = w.min(h);
    let mut c = String::new();

    // Grid every 36 pt (or 5% of the short side), heavier every fifth line.
    let step = (0.05 * short).max(36.0);
    let (mut minor, mut major) = (String::new(), String::new());
    let mut i = 0u32;
    while i as f64 * step <= w.max(h) {
        let d = i as f64 * step;
        let dst = if i.is_multiple_of(5) {
            &mut major
        } else {
            &mut minor
        };
        let _ = write!(
            dst,
            "{d:.2} 0 m {d:.2} {h:.2} l 0 {d:.2} m {w:.2} {d:.2} l "
        );
        i += 1;
    }
    let _ = write!(c, "q 0.25 w 0.88 G {minor} S 0.6 w 0.55 G {major} S Q ");

    // Thirty shapes from a fixed-seed generator: rects, circles as four
    // beziers, dashed lines.
    let mut seed: u64 = 0x5EED_1199;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64)
    };
    const K: f64 = 0.552_284_749_8;
    for n in 0..30 {
        let (x, y) = (0.2 * w + next() * 0.6 * w, 0.2 * h + next() * 0.5 * h);
        let r = (0.008 + next() * 0.02) * short;
        let _ = write!(
            c,
            "q {:.2} {:.2} {:.2} rg 0 G 0.4 w ",
            next(),
            next(),
            next()
        );
        match n % 3 {
            0 => {
                let _ = write!(c, "{x:.2} {y:.2} {:.2} {r:.2} re B ", 2.0 * r);
            }
            1 => {
                let _ = write!(
                    c,
                    "{:.2} {y:.2} m {:.2} {:.2} {:.2} {:.2} {x:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {y:.2} c \
                     {:.2} {:.2} {:.2} {:.2} {x:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {y:.2} c B ",
                    x + r,
                    x + r,
                    y + K * r,
                    x + K * r,
                    y + r,
                    y + r,
                    x - K * r,
                    y + r,
                    x - r,
                    y + K * r,
                    x - r,
                    x - r,
                    y - K * r,
                    x - K * r,
                    y - r,
                    y - r,
                    x + K * r,
                    y - r,
                    x + r,
                    y - K * r,
                    x + r,
                );
            }
            _ => {
                let _ = write!(
                    c,
                    "[{:.1} {:.1}] 0 d {x:.2} {y:.2} m {:.2} {:.2} l S ",
                    r * 0.4,
                    r * 0.3,
                    x + 4.0 * r,
                    y + r
                );
            }
        }
        c.push_str("Q ");
    }

    // Diagonal hatching clipped to a box.
    let (hx, hy, hw, hh) = (0.46 * w, 0.36 * h, 0.14 * w, 0.12 * h);
    let _ = write!(c, "q {hx:.2} {hy:.2} {hw:.2} {hh:.2} re W n 0.3 w 0 G ");
    let mut d = -hh;
    while d < hw {
        let _ = write!(
            c,
            "{:.2} {hy:.2} m {:.2} {:.2} l ",
            hx + d,
            hx + d + hh,
            hy + hh
        );
        d += (0.012 * short).max(4.0);
    }
    c.push_str("S Q ");

    // Title block and text at several sizes, in three base-14 fonts.
    let (bw, bh) = (0.32 * w, 0.14 * h);
    let (bx, by) = (w - 14.0 - bw, 14.0);
    let _ = write!(
        c,
        "q 0 G 0.8 w {bx:.2} {by:.2} {bw:.2} {bh:.2} re S {:.2} {by:.2} m {:.2} {:.2} l S Q ",
        bx + bw / 2.0,
        bx + bw / 2.0,
        by + bh
    );
    let size = |frac: f64, lo: f64, hi: f64| (frac * short).clamp(lo, hi);
    for (font, sz, x, y, text) in [
        (2, size(0.06, 10.0, 48.0), 0.12 * w, 0.92 * h, "SITE PLAN"),
        (
            3,
            size(0.025, 8.0, 24.0),
            0.12 * w,
            0.88 * h,
            "LOREM IPSUM DOLOR SIT AMET",
        ),
        (
            1,
            size(0.012, 6.0, 12.0),
            0.12 * w,
            0.855 * h,
            "CONSECTETUR ADIPISCING ELIT",
        ),
        (
            1,
            size(0.014, 5.0, 14.0),
            bx + 4.0,
            by + bh * 0.6,
            "PLACEHOLDER SITE",
        ),
        (
            2,
            size(0.02, 6.0, 20.0),
            bx + bw / 2.0 + 4.0,
            by + bh * 0.3,
            "A-101",
        ),
    ] {
        let _ = write!(c, "BT /F{font} {sz:.2} Tf {x:.2} {y:.2} Td ({text}) Tj ET ");
    }

    // The landmark, then the border over everything: 2 pt of ink on the first
    // and last row and column.
    let t = tri_pt(w, h);
    let _ = write!(c, "q 0 g 0 {h:.2} m {t:.2} {h:.2} l 0 {:.2} l f Q ", h - t);
    let _ = write!(c, "q 0 G 2 w 1 1 {:.2} {:.2} re S Q ", w - 2.0, h - 2.0);
    c
}

/// One-page PDF carrying [`rich_content`]. `rotate` is the page's `/Rotate`,
/// `origin` the MediaBox's lower-left corner.
fn write_pdf(dir: &Path, name: &str, w: f64, h: f64, rotate: i64, origin: (f64, f64)) -> PathBuf {
    use lopdf::{Document, Object, Stream, dictionary};

    let (ox, oy) = origin;
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let content = format!("q 1 0 0 1 {ox} {oy} cm {} Q", rich_content(w, h));
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
    let font =
        |name: &str| dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => name };
    let (f1, f2, f3) = (
        doc.add_object(font("Helvetica")),
        doc.add_object(font("Times-Roman")),
        doc.add_object(font("Courier")),
    );
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
        "Resources" => dictionary! { "Font" => dictionary! { "F1" => f1, "F2" => f2, "F3" => f3 } },
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

/// Pixels whose colour differs by more than 96 in some channel, and the total.
fn strong_diffs(a: &Raster, b: &Raster) -> (usize, usize) {
    let n = a
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.data().as_chunks::<4>().0)
        .filter(|(p, q)| p.iter().zip(q.iter()).any(|(x, y)| x.abs_diff(*y) > 96))
        .count();
    (n, a.data().len() / 4)
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
/// bitmap edge to edge: the border is on the first and last row and column of
/// the cached render, a triangle sits in the top-left corner only, and the
/// streaming source gives the same pixels as the cached one for every row of
/// the page, border and shape edges included. A blank band on the right or
/// bottom is what a strip scale that ignores the fraction would leave.
#[test]
#[cfg_attr(miri, ignore)]
fn a_fractional_page_fills_the_bitmap_to_the_edge_in_every_route() {
    let dir = tempfile::tempdir().unwrap();
    for (name, pw, ph, dpi, want) in [
        ("A3", 841.89, 1190.551, 150, (1754, 2480)),
        ("A2", 1190.551, 1683.78, 96, (1587, 2245)),
        ("A1", 1683.78, 2383.937, 72, (1684, 2384)),
    ] {
        let pdf = write_pdf(dir.path(), "frac.pdf", pw, ph, 0, (0.0, 0.0));
        let (w, h) = libvips_dims(pw, ph, dpi);
        assert_eq!((w, h), want, "{name}");

        let cached = render_page_pdfium(&pdf, 1, dpi).unwrap();
        assert_eq!((cached.width(), cached.height()), (w, h), "{name}");
        for x in [0, w / 2, w - 1] {
            assert!(is_dark(px(&cached, x, 0)), "{name} first row at x={x}");
            assert!(is_dark(px(&cached, x, h - 1)), "{name} last row at x={x}");
        }
        for y in [0, h / 2, h - 1] {
            assert!(is_dark(px(&cached, 0, y)), "{name} first column at y={y}");
            assert!(
                is_dark(px(&cached, w - 1, y)),
                "{name} last column at y={y}"
            );
        }
        let t = (tri_pt(pw, ph) * dpi as f64 / 72.0 / 3.0) as u32;
        assert!(is_dark(px(&cached, t, t)), "{name} triangle top-left");
        assert!(
            !is_dark(px(&cached, w - 1 - t, t)),
            "{name} top-right is clear"
        );

        let streaming = PdfiumStripSource::new_streaming(&pdf, 1, dpi).unwrap();
        assert_eq!((streaming.width(), streaming.height()), (w, h), "{name}");
        let mut y0 = 0;
        while y0 < h {
            let rows = 128.min(h - y0);
            let strip = streaming.render_strip(y0, rows).unwrap();
            assert_eq!((strip.width(), strip.height()), (w, rows), "{name}");
            let want = cached.extract(0, y0, w, rows).unwrap();
            let (strong, total) = strong_diffs(&strip, &want);
            // The matrix and stretch paths antialias alike, so no pixel should
            // differ by much. A strip scale that ignores the fraction of a
            // point puts the border a pixel or two off, one strong pixel per
            // row, which is what this catches.
            assert_eq!(
                strong,
                0,
                "{name}: streaming rows {y0}..{} differ from the cached render in {strong} of {total} pixels",
                y0 + rows
            );
            y0 += rows;
        }
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
        // The triangle sits at the page's top-left, so /Rotate moves it: to
        // the top-right for 90 and the bottom-left for 270.
        let t = (tri_pt(612.0, 792.0) * 300.0 / 72.0 / 3.0) as u32;
        let (tx, ty) = if rotate == 90 {
            (3299 - t, t)
        } else {
            (t, 2549 - t)
        };
        assert!(is_dark(px(&cached, tx, ty)), "rotate {rotate}: triangle");
        assert!(
            !is_dark(px(&cached, t, t)),
            "rotate {rotate}: old corner clear"
        );
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
    let t = (tri_pt(612.0, 792.0) * 300.0 / 72.0 / 3.0) as u32;
    assert!(
        is_dark(px(&cached, t, t)),
        "triangle follows the MediaBox origin"
    );
    let streaming = PdfiumStripSource::new_streaming(&pdf, 1, 300).unwrap();
    assert_eq!((streaming.width(), streaming.height()), (2550, 3300));
}

/// `LegacyTruncated` is the old pipeline end to end: the cached render comes
/// out at the sizes 0.5.x rendered (quoted from the issue), `pixel_dims`
/// agrees with it, and a streaming source reports the truncated arithmetic
/// it always did.
#[test]
#[cfg_attr(miri, ignore)]
fn legacy_truncated_renders_the_sizes_0_5_x_rendered() {
    let dir = tempfile::tempdir().unwrap();
    for (name, w, h, want) in [
        ("Letter", 612.0, 792.0, (2549, 3299)),
        ("Tabloid", 792.0, 1224.0, (3299, 5098)),
        ("A3", 841.89, 1190.551, (3507, 4959)),
    ] {
        let pdf = write_pdf(dir.path(), "legacy.pdf", w, h, 0, (0.0, 0.0));
        let cached =
            libviprs::render_page_pdfium_with(&pdf, 1, 300, PageSizing::LegacyTruncated).unwrap();
        assert_eq!((cached.width(), cached.height()), want, "{name}");
        let (pw, ph) = (w as f32 as f64, h as f32 as f64);
        assert_eq!(
            PageSizing::LegacyTruncated.pixel_dims(pw, ph, 300),
            want,
            "{name}: pixel_dims is the rendered size"
        );
        let source = PdfiumStripSource::builder(&pdf, 1, 300)
            .sizing(PageSizing::LegacyTruncated)
            .build()
            .unwrap();
        assert_eq!((source.width(), source.height()), want, "{name} cached");
        assert_eq!(source.sizing(), PageSizing::LegacyTruncated);
    }
}
