//! Issue #1578 (L6 and L7), on small in-memory pages: the text parameters come back on `Q`, the
//! text anchors keep a line style of their own next to the clip rings, and a conversion bakes no
//! images unless it is asked to. `pvec_1578_e_text_images.rs` has the `blueprint-mix.pdf` checks.
//! Needs `--features pvec-write`.

#![cfg(all(feature = "pvec-write", feature = "pvec-raster"))]

use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use libviprs::pvec::{pdf_page_to_pvec, pdf_page_to_pvec_with};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{ClipStream, LineStyleStream, TextStream};

/// A 612 by 792 page with Helvetica as `/F1` (no `/Widths`, so every glyph is 500 units wide) and
/// a 32 by 32 grey image as `/Im1`.
fn pdf(content: &str) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
    let font = doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
    });
    let image = doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 32, "Height" => 32,
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
        },
        vec![128u8; 32 * 32],
    ));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => content,
        "Resources" => dictionary! {
            "Font" => dictionary! { "F1" => font },
            "XObject" => dictionary! { "Im1" => image },
        },
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn page(content: &str) -> PdfVectorPage {
    extract_page_paths_from_bytes(&pdf(content), 1, &PdfVectorLimits::default()).unwrap()
}

const RULE: &str = "10 10 m 100 10 l S ";

#[test]
fn q_and_big_q_bring_back_tc_tw_tz_tl_tf_and_ts() {
    let page = page(&format!(
        "{RULE}BT /F1 10 Tf 0 50 Td (A) Tj ET \
         q BT /F1 20 Tf 3 Tc 7 Tw 200 Tz 14 TL 5 Ts 0 100 Td (A) Tj ET Q \
         BT 0 200 Td T* (A A) Tj ET"
    ));
    assert_eq!(page.texts.len(), 3, "{:?}", page.texts);
    let inside = &page.texts[1];
    assert_eq!(inside.start, [0.0, 105.0], "Ts applies inside the q");
    // 1 glyph of 500 units at 20 points, 3 points of Tc, all at 200 percent.
    assert_eq!(inside.end[0] - inside.start[0], (10.0 + 3.0) * 2.0);
    let after = &page.texts[2];
    // Tl is back to 0 (no move from T*), Ts is back to 0, and the size is 10 again.
    assert_eq!(after.start, [0.0, 200.0]);
    // Three glyphs of 5 points, no Tc, no Tw, 100 percent.
    assert_eq!(after.end[0] - after.start[0], 15.0);
    assert_eq!(after.cap_height, 7.0);
}

struct Written {
    texts: TextStream,
    clips: ClipStream,
    styles: LineStyleStream,
}

#[test]
fn a_text_anchor_and_a_clip_ring_do_not_share_a_line_style() {
    let page = page(&format!(
        "q 40 40 m 280 40 l 160 190 l h W n {RULE}Q BT /F1 10 Tf 0 50 Td (A) Tj ET"
    ));
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let payload = |cap| {
        &streams
            .iter()
            .find(|s| s.capability == cap)
            .unwrap_or_else(|| panic!("no stream {cap:?}"))
            .payload
    };
    let written = Written {
        texts: TextStream::decode(payload(TextStream::CAPABILITY)).unwrap(),
        clips: ClipStream::decode(payload(ClipStream::CAPABILITY)).unwrap(),
        styles: LineStyleStream::decode(payload(LineStyleStream::CAPABILITY)).unwrap(),
    };
    let name_of = |entity| {
        let (_, style) = written
            .styles
            .assignments
            .iter()
            .find(|(id, _)| *id == entity)
            .expect("the entity has a line style");
        written.styles.styles[*style as usize].name.clone()
    };
    assert_eq!(name_of(written.texts.records[0].entity), "pdf-text-anchor");
    assert_eq!(name_of(written.clips.clips[0].boundary[0]), "pdf-clip-ring");
}

#[test]
fn a_conversion_bakes_no_image_unless_it_is_asked_to() {
    let page = page(&format!("{RULE}q 100 0 0 100 50 50 cm /Im1 Do Q"));
    assert_eq!(page.images.len(), 1);
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let placements = PvecReader::open(bytes)
        .unwrap()
        .raster_placements()
        .unwrap();
    assert!(
        placements.is_empty(),
        "{} rasters were baked",
        placements.len()
    );
}

#[test]
fn asked_to_bake_the_image_is_a_raster_placed_where_the_pdf_paints_it() {
    let page = page(&format!("{RULE}q 100 0 0 50 50 200 cm /Im1 Do Q"));
    let bytes = pdf_page_to_pvec_with(&page, "test", "0", true).unwrap();
    let placements = PvecReader::open(bytes)
        .unwrap()
        .raster_placements()
        .unwrap();
    assert_eq!(placements.len(), 1);
    // Pixel (0, 0) is the top left of the image, (0, 1) in its unit square, and the far corner is
    // (1, 0). Source space is inches with the origin at the top left of the page.
    let [a, b, c, d, e, f] = page.images[0].ctm;
    let placement = &placements[0];
    let (w, h) = (f64::from(placement.width), f64::from(placement.height));
    for (px, py, u, v) in [(0.0, 0.0, 0.0, 1.0), (w, h, 1.0, 0.0)] {
        let [pa, pb, pc, pd, ptx, pty] = placement.pixel_to_source;
        let got = [pa * px + pb * py + ptx, pc * px + pd * py + pty];
        let want = [
            (a * u + c * v + e) / 72.0,
            (page.height - (b * u + d * v + f)) / 72.0,
        ];
        assert!((got[0] - want[0]).abs() < 1e-6, "{got:?} against {want:?}");
        assert!((got[1] - want[1]).abs() < 1e-6, "{got:?} against {want:?}");
    }
}
