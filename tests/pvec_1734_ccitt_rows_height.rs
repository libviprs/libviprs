//! Issue #1734: a CCITT image whose `/Rows` is not its `/Height` is refused, so the tables of an
//! AutoCAD plot never reach the pvec.
//!
//! AutoCAD (`pdfplot16.hdi`) writes `/DecodeParms << /K -1 /Columns W /Rows W >>` on a 1-bit
//! Indexed image of height H, with `/Rows` equal to the width. `/Height` is the number of scan
//! lines and `/Rows` is only a hint (PDF 32000-1, table 11), so these images have to bake.
//!
//! Every PDF here is built in memory from the libtiff-encoded G4 stream the #1363 tests use
//! (`G4_SMALL`, 40 by 8); no customer file is involved. These tests are red at 417e967c: each
//! one is refused with `Parms("/Rows N is not /Height 8")`. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

#[path = "common/ccitt_fixtures.rs"]
mod ccitt_fixtures;
#[path = "common/pdf_images.rs"]
mod fixture;

use ccitt_fixtures::G4_SMALL;
use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageOptions, bake_page_images};
use libviprs::pvec::pdf_page_to_pvec_report;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Object, Stream, dictionary};

/// Must match `ink_small` in `tests/common/ccitt_gen.py`.
fn ink_small(x: i64, y: i64) -> bool {
    (x + 2 * y) % 11 < 4 || (x >= 30 && y % 2 == 0)
}

/// The 40 by 8 G4 picture, black ink as gray 0 on gray 255.
fn picture() -> Vec<u8> {
    (0..8)
        .flat_map(|y| (0..40).map(move |x| (x, y)))
        .map(|(x, y)| if ink_small(x, y) { 0 } else { 255 })
        .collect()
}

/// A 40 by 8 CCITT G4 image with the given `/Rows` and colour space, painted on a one page PDF.
fn g4_pdf(rows: i64, colour_space: Object) -> Vec<u8> {
    let parms = dictionary! { "K" => -1, "Columns" => 40, "Rows" => rows };
    let dict = dictionary! {
        "ColorSpace" => colour_space,
        "BitsPerComponent" => 1,
        "Filter" => "CCITTFaxDecode",
        "DecodeParms" => parms,
    };
    one(
        image(40, 8, dict, G4_SMALL.to_vec()),
        "q 20 0 0 20 100 200 cm /Im1 Do Q",
    )
}

fn indexed_black_white() -> Object {
    Object::Array(vec![
        name("Indexed"),
        name("DeviceRGB"),
        1.into(),
        literal(&[0, 0, 0, 255, 255, 255]),
    ])
}

/// The one image of `bytes` baked, and its full resolution samples read back from the tile.
fn baked_samples(bytes: &[u8]) -> Vec<u8> {
    let bake: PdfImageBake = bake_page_images(&page(bytes), &PdfImageOptions::default());
    assert!(
        bake.refused.is_empty(),
        "the image was refused: {:?}",
        bake.refused.iter().map(|r| &r.error).collect::<Vec<_>>()
    );
    assert_eq!(bake.written.len(), 1);
    let pyramid = &bake.written[0].pyramid;
    assert_eq!((pyramid.width, pyramid.height), (40, 8));
    let tile = pyramid
        .tile(u32::from(pyramid.levels) - 1, 0, 0)
        .expect("a stored tile");
    let luma = image::load_from_memory(tile).unwrap().to_luma8();
    (0..8u32)
        .flat_map(|y| (0..40u32).map(move |x| (x, y)))
        .map(|(x, y)| luma.get_pixel(x, y).0[0])
        .collect()
}

#[test]
fn issue_1734_rows_equal_to_columns_decodes_at_the_image_height() {
    // What AutoCAD writes: /Rows is the width (40), /Height is 8.
    let samples = baked_samples(&g4_pdf(40, name("DeviceGray")));
    assert_eq!(samples, picture());
}

#[test]
fn issue_1734_rows_larger_than_height_decodes_at_the_image_height() {
    // The stream holds the 8 rows /Height names; /Rows says 9.
    let samples = baked_samples(&g4_pdf(9, name("DeviceGray")));
    assert_eq!(samples, picture());
}

#[test]
fn issue_1734_rows_smaller_than_height_decodes_at_the_image_height() {
    // /Rows 4 is a hint and does not cut the image to four lines.
    let samples = baked_samples(&g4_pdf(4, name("DeviceGray")));
    assert_eq!(samples, picture());
}

#[test]
fn issue_1734_one_bit_indexed_black_white_with_rows_columns_decodes() {
    // The colour space of the plot's tables: [/Indexed /DeviceRGB 1 <000000ffffff>].
    let samples = baked_samples(&g4_pdf(40, indexed_black_white()));
    assert_eq!(samples, picture());
}

/// A page shaped like the AutoCAD plot: `/Rotate 270`, the `0.06` scale in a content stream of
/// its own that never restores it, then the images under a clip, each in its own `q ... Q`, and
/// one stroked frame so the page has a path. Two images are CCITT with `/Rows` equal to `/Columns`
/// (the tables), one is a plain RGB image (stands in for the logo).
fn autocad_style_page() -> Vec<u8> {
    let mut doc = doc();
    let pages_id = doc.new_object_id();
    let ccitt = |doc: &mut lopdf::Document, rows: i64| {
        let dict = dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 40, "Height" => 8,
            "ColorSpace" => indexed_black_white(), "BitsPerComponent" => 1,
            "Filter" => "CCITTFaxDecode",
            "DecodeParms" => dictionary! { "K" => -1, "Columns" => 40, "Rows" => rows },
        };
        doc.add_object(Stream::new(dict, G4_SMALL.to_vec()))
    };
    let table_a = ccitt(&mut doc, 40);
    let table_b = ccitt(&mut doc, 40);
    let logo = doc.add_object(rgb8(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]));
    let scale = doc.add_object(Stream::new(
        Dictionary::new(),
        b"0.06 0 0 0.06 0 0 cm".to_vec(),
    ));
    let clip = "0 0 m 10200 0 l 10200 13200 l 0 13200 l 0 0 l W n";
    let body = format!(
        "{clip}\n1000 1000 m 9000 1000 l 9000 12000 l 1000 12000 l 1000 1000 l S\n\
         q 4000 0 0 800 1000 5000 cm /TableA Do Q\n\
         q 3000 0 0 600 1000 8000 cm /TableB Do Q\n\
         q 900 0 0 900 6000 9000 cm /Logo Do Q"
    );
    let content = doc.add_object(Stream::new(Dictionary::new(), body.into_bytes()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Rotate" => 270,
        "Contents" => vec![Object::from(scale), Object::from(content)],
        "Resources" => dictionary! { "XObject" => dictionary! {
            "TableA" => table_a, "TableB" => table_b, "Logo" => logo,
        } },
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

#[test]
fn issue_1734_a_plot_page_converts_with_every_image_baked() {
    let bytes = autocad_style_page();
    let page = extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).unwrap();
    assert_eq!(page.images.len(), 3, "the page paints three images");
    let (_, report) = pdf_page_to_pvec_report(&page, "test-1734", "1", true).unwrap();
    assert_eq!(
        report.refused_total(),
        0,
        "images were refused: {:?}",
        report.refused
    );
    assert_eq!(report.written, 3);
    assert!(
        report.baked(),
        "a viewer would still need the page pyramid: {report:?}"
    );
}
