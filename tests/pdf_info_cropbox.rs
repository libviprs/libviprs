//! Issue #1209: `pdf_info` reports the size of the box a page is rendered into,
//! so `info`, `plan` and the planner's canvas agree with `render_page_pdfium`
//! and with libvips `pdfload`.
//!
//! pdfium (and libvips on top of it or poppler) sizes a page to its CropBox
//! intersected with its MediaBox, both inherited through the page tree, and
//! then applies `/Rotate`. A box that is missing, not four numbers, or empty
//! after the intersection falls back to the MediaBox. `/UserUnit` is ignored
//! by every one of them, and by `pdf_info`, which is pinned here too.
//!
//! The first block needs no pdfium: it checks the points `pdf_info` reports
//! against numbers written out by hand. The second block (feature `pdfium`,
//! libpdfium at runtime) renders the same files and checks that the pixel
//! size of the render is `PageSizing::Exact.pixel_dims` of what `pdf_info`
//! said, at 72, 150 and 300 dpi.

use std::path::{Path, PathBuf};

use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// Where a page-tree attribute lives.
#[derive(Clone, Copy, PartialEq)]
enum At {
    /// Not written at all.
    Nowhere,
    /// On the page dictionary.
    Page,
    /// On the parent `/Pages` node, to be inherited.
    Pages,
}

/// How the CropBox array is stored.
#[derive(Clone, Copy, PartialEq)]
enum Store {
    Direct,
    /// An indirect object holding the array.
    Indirect,
}

struct Spec {
    media: Vec<f64>,
    media_at: At,
    crop: Option<Vec<Object>>,
    crop_at: At,
    crop_store: Store,
    rotate: Option<(i64, At)>,
    user_unit: Option<f64>,
    /// A second, larger CropBox on the pages node, which the page's own must beat.
    shadowed_crop: Option<[f64; 4]>,
}

impl Spec {
    fn new(media: [f64; 4]) -> Self {
        Spec {
            media: media.to_vec(),
            media_at: At::Page,
            crop: None,
            crop_at: At::Nowhere,
            crop_store: Store::Direct,
            rotate: None,
            user_unit: None,
            shadowed_crop: None,
        }
    }
    fn crop(mut self, b: [f64; 4]) -> Self {
        self.crop = Some(reals(&b));
        self.crop_at = At::Page;
        self
    }
    fn crop_objects(mut self, b: Vec<Object>) -> Self {
        self.crop = Some(b);
        self.crop_at = At::Page;
        self
    }
    fn crop_on(mut self, at: At) -> Self {
        self.crop_at = at;
        self
    }
    fn indirect(mut self) -> Self {
        self.crop_store = Store::Indirect;
        self
    }
    fn rotate(mut self, deg: i64, at: At) -> Self {
        self.rotate = Some((deg, at));
        self
    }
    fn media_on(mut self, at: At) -> Self {
        self.media_at = at;
        self
    }
    fn shadowed_by_pages_crop(mut self, b: [f64; 4]) -> Self {
        self.shadowed_crop = Some(b);
        self
    }
    fn user_unit(mut self, u: f64) -> Self {
        self.user_unit = Some(u);
        self
    }
}

fn reals(v: &[f64]) -> Vec<Object> {
    v.iter().map(|&x| Object::Real(x as f32)).collect()
}

fn write_pdf(dir: &Path, name: &str, spec: &Spec) -> PathBuf {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let content_id = doc.add_object(Stream::new(dictionary! {}, b"0 0 m 10 10 l S".to_vec()));
    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => dictionary! {},
    };
    let mut pages = dictionary! { "Type" => "Pages", "Count" => 1i64 };

    let set = |at: At, key: &str, value: Object, page: &mut Dictionary, pages: &mut Dictionary| {
        match at {
            At::Page => page.set(key, value),
            At::Pages => pages.set(key, value),
            At::Nowhere => {}
        }
    };

    set(
        spec.media_at,
        "MediaBox",
        Object::Array(reals(&spec.media)),
        &mut page,
        &mut pages,
    );
    if let Some(crop) = &spec.crop {
        let value = match spec.crop_store {
            Store::Direct => Object::Array(crop.clone()),
            Store::Indirect => Object::Reference(doc.add_object(Object::Array(crop.clone()))),
        };
        set(spec.crop_at, "CropBox", value, &mut page, &mut pages);
    }
    if let Some(b) = spec.shadowed_crop {
        pages.set("CropBox", Object::Array(reals(&b)));
    }
    if let Some((deg, at)) = spec.rotate {
        set(at, "Rotate", Object::Integer(deg), &mut page, &mut pages);
    }
    if let Some(u) = spec.user_unit {
        page.set("UserUnit", Object::Real(u as f32));
    }

    let page_id = doc.add_object(page);
    pages.set("Kids", vec![page_id.into()]);
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let path = dir.join(name);
    doc.save(&path).unwrap();
    path
}

/// One named page and the size, in points, a viewer shows it at.
fn cases() -> Vec<(&'static str, Spec, (f64, f64))> {
    let letter = [0.0, 0.0, 612.0, 792.0];
    vec![
        ("media box only", Spec::new(letter), (612.0, 792.0)),
        (
            "smaller crop box",
            Spec::new(letter).crop([50.0, 40.0, 550.0, 700.0]),
            (500.0, 660.0),
        ),
        (
            "crop box larger than the media box on every side",
            Spec::new(letter).crop([-50.0, -50.0, 700.0, 900.0]),
            (612.0, 792.0),
        ),
        (
            "crop box overlapping one corner of the media box",
            Spec::new(letter).crop([100.0, 100.0, 700.0, 900.0]),
            (512.0, 692.0),
        ),
        (
            "crop box inherited from the pages node",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .crop_on(At::Pages),
            (500.0, 660.0),
        ),
        (
            "crop and media box both inherited",
            Spec::new(letter)
                .media_on(At::Pages)
                .crop([50.0, 40.0, 550.0, 700.0])
                .crop_on(At::Pages),
            (500.0, 660.0),
        ),
        (
            "page crop box wins over the inherited one",
            Spec::new(letter)
                .crop([0.0, 0.0, 300.0, 400.0])
                .shadowed_by_pages_crop([10.0, 10.0, 600.0, 780.0]),
            (300.0, 400.0),
        ),
        (
            "crop box with rotate 90",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .rotate(90, At::Page),
            (660.0, 500.0),
        ),
        (
            "crop box with rotate 270",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .rotate(270, At::Page),
            (660.0, 500.0),
        ),
        (
            "crop box with rotate 180",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .rotate(180, At::Page),
            (500.0, 660.0),
        ),
        (
            "inherited crop box with inherited rotate 90",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .crop_on(At::Pages)
                .rotate(90, At::Pages),
            (660.0, 500.0),
        ),
        (
            "crop box with a non-zero origin",
            Spec::new([100.0, 50.0, 712.0, 842.0]).crop([150.0, 100.0, 650.0, 700.0]),
            (500.0, 600.0),
        ),
        (
            "crop box given back to front",
            Spec::new(letter).crop([550.0, 700.0, 50.0, 40.0]),
            (500.0, 660.0),
        ),
        (
            "indirect crop box",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .indirect(),
            (500.0, 660.0),
        ),
        (
            "indirect crop box inherited from the pages node",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .crop_on(At::Pages)
                .indirect(),
            (500.0, 660.0),
        ),
        (
            "integer crop box entries",
            Spec::new(letter).crop_objects(vec![
                Object::Integer(50),
                Object::Integer(40),
                Object::Integer(550),
                Object::Integer(700),
            ]),
            (500.0, 660.0),
        ),
        (
            "fractional crop box",
            Spec::new([0.0, 0.0, 841.89, 1190.551]).crop([10.5, 20.25, 800.125, 1100.75]),
            (789.625, 1080.5),
        ),
        (
            "crop box with only three entries falls back to the media box",
            Spec::new(letter).crop_objects(reals(&[50.0, 40.0, 550.0])),
            (612.0, 792.0),
        ),
        (
            "crop box with five entries falls back to the media box",
            Spec::new(letter).crop_objects(reals(&[50.0, 40.0, 550.0, 700.0, 9.0])),
            (612.0, 792.0),
        ),
        (
            "zero-area crop box falls back to the media box",
            Spec::new(letter).crop([100.0, 100.0, 100.0, 700.0]),
            (612.0, 792.0),
        ),
        (
            "zero-height crop box falls back to the media box",
            Spec::new(letter).crop([100.0, 100.0, 500.0, 100.0]),
            (612.0, 792.0),
        ),
        (
            "all-zero crop box falls back to the media box",
            Spec::new(letter).crop([0.0, 0.0, 0.0, 0.0]),
            (612.0, 792.0),
        ),
        (
            "crop box of the wrong type falls back to the media box",
            Spec::new(letter).crop_objects(vec![Object::Name(b"x".to_vec())]),
            (612.0, 792.0),
        ),
        (
            "user unit is ignored, as in the render",
            Spec::new(letter)
                .crop([50.0, 40.0, 550.0, 700.0])
                .user_unit(2.0),
            (500.0, 660.0),
        ),
    ]
}

#[test]
#[cfg_attr(miri, ignore)]
fn pdf_info_reports_the_box_the_page_is_rendered_into() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (name, spec, want) in cases() {
        let pdf = write_pdf(dir.path(), "page.pdf", &spec);
        let info = libviprs::pdf_info(&pdf).unwrap();
        let page = &info.pages[0];
        // The expected numbers are exact in f32, so compare exactly.
        let got = (page.width_pts, page.height_pts);
        if got != want {
            wrong.push(format!("{name}: want {want:?}, pdf_info says {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A crop box disjoint from the media box leaves nothing to render. pdfium
/// treats that as an empty box and uses the MediaBox, and so does `pdf_info`.
#[test]
#[cfg_attr(miri, ignore)]
fn a_crop_box_outside_the_media_box_falls_back_to_the_media_box() {
    let dir = tempfile::tempdir().unwrap();
    let spec = Spec::new([0.0, 0.0, 612.0, 792.0]).crop([700.0, 800.0, 900.0, 1000.0]);
    let pdf = write_pdf(dir.path(), "outside.pdf", &spec);
    let info = libviprs::pdf_info(&pdf).unwrap();
    assert_eq!(
        (info.pages[0].width_pts, info.pages[0].height_pts),
        (612.0, 792.0)
    );
}

#[cfg(feature = "pdfium")]
mod against_the_render {
    use super::*;
    use libviprs::{PageSizing, PdfiumStripSource, StripSource, render_page_pdfium};

    const DPIS: [u32; 3] = [72, 150, 300];

    fn check(name: &str, path: &Path, wrong: &mut Vec<String>) {
        let info = libviprs::pdf_info(path).unwrap();
        let (w, h) = (info.pages[0].width_pts, info.pages[0].height_pts);
        for dpi in DPIS {
            let want = PageSizing::Exact.pixel_dims(w, h, dpi);
            let rendered = render_page_pdfium(path, 1, dpi).unwrap();
            let got = (rendered.width(), rendered.height());
            if got != want {
                wrong.push(format!(
                    "{name} at {dpi} dpi: pdf_info ({w} x {h} pt) gives {want:?}, render is {got:?}"
                ));
            }
            let streaming = PdfiumStripSource::new_streaming(path, 1, dpi).unwrap();
            let got = (streaming.width(), streaming.height());
            if got != want {
                wrong.push(format!(
                    "{name} at {dpi} dpi: pdf_info gives {want:?}, streaming source is {got:?}"
                ));
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pdf_info_sizes_every_case_like_the_pdfium_render() {
        let dir = tempfile::tempdir().unwrap();
        let mut wrong = Vec::new();
        for (name, spec, _) in cases() {
            let pdf = write_pdf(dir.path(), "page.pdf", &spec);
            check(name, &pdf, &mut wrong);
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// pdfium makes a 0 x 0 page of a CropBox that misses the MediaBox and
    /// refuses to render it, so there is no render to agree with. `pdf_info`
    /// reports the MediaBox (pinned above) rather than a zero-size page the
    /// planner would divide by.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_crop_box_outside_the_media_box_is_a_page_pdfium_will_not_render() {
        let dir = tempfile::tempdir().unwrap();
        let spec = Spec::new([0.0, 0.0, 612.0, 792.0]).crop([700.0, 800.0, 900.0, 1000.0]);
        let pdf = write_pdf(dir.path(), "outside.pdf", &spec);
        assert!(render_page_pdfium(&pdf, 1, 72).is_err());
    }

    /// Fractional origins and sizes are where an f64 subtraction of the
    /// widened f32 corners and pdfium's f32 subtraction can round a pixel
    /// apart. The sweep is wide on purpose.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn pdf_info_sizes_fractional_boxes_like_the_pdfium_render() {
        let dir = tempfile::tempdir().unwrap();
        let mut wrong = Vec::new();
        let mut seed: u64 = 0x1209_C0B0;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for n in 0..24 {
            let (ox, oy) = (next() * 300.0, next() * 300.0);
            let media = [
                ox,
                oy,
                ox + 300.0 + next() * 900.0,
                oy + 300.0 + next() * 900.0,
            ];
            let crop = [
                media[0] + next() * 100.0,
                media[1] + next() * 100.0,
                media[2] - next() * 100.0,
                media[3] - next() * 100.0,
            ];
            let spec = Spec::new(media).crop(crop);
            let pdf = write_pdf(dir.path(), "frac.pdf", &spec);
            check(&format!("fractional box {n}"), &pdf, &mut wrong);
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
