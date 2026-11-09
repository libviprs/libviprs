//! Issue #1578 (L6 and L7, the second half): text and baked rasters are named in the paint order
//! (stream 23) where the PDF painted them, and every kind of item (stroke, fill, text, raster)
//! under a non-rectangular clip is assigned to its clip record (stream 26). The content stream is
//! the oracle: an item painted between two paths sits between them in the order. Small in-memory
//! pages; `blueprint-mix.pdf` stays the regression in `pvec_1578_e_text_images.rs` and the pinned
//! bytes. Needs `--features pvec-write,pvec-raster`.

#![cfg(all(feature = "pvec-write", feature = "pvec-raster"))]

use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use libviprs::pvec::{pdf_page_to_pvec, pdf_page_to_pvec_with};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{
    ClipStream, ItemKind, PaintOrderStream, Stream as PvecStream, TextStream,
};

const TRIANGLE: &str = "40 40 m 280 40 l 160 190 l h ";
const TEXT: &str = "BT /F1 12 Tf 20 50 Td (Hi) Tj ET ";
const IMAGE: &str = "q 50 0 0 50 100 100 cm /Im1 Do Q ";
const RULE_A: &str = "10 10 m 100 10 l S ";
const RULE_B: &str = "10 20 m 100 20 l S ";

/// A 612 by 792 page with Helvetica as `/F1` and a 32 by 32 grey image as `/Im1`.
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

fn streams(content: &str, bake: bool) -> Vec<PvecStream> {
    let page =
        extract_page_paths_from_bytes(&pdf(content), 1, &PdfVectorLimits::default()).unwrap();
    let bytes = if bake {
        pdf_page_to_pvec_with(&page, "test", "0", true).unwrap()
    } else {
        pdf_page_to_pvec(&page, "test", "0").unwrap()
    };
    PvecReader::open(bytes).unwrap().streams().unwrap().0
}

/// The paint order, one (kind, number) per item. Entity ids start at 1, so the first stroke is 1.
fn order(streams: &[PvecStream]) -> Vec<(ItemKind, u32)> {
    let stream = streams
        .iter()
        .find(|s| s.capability == PaintOrderStream::CAPABILITY)
        .expect("stream 23");
    PaintOrderStream::decode(&stream.payload)
        .unwrap()
        .runs
        .iter()
        .flat_map(|r| (r.first..r.first + r.count).map(move |n| (r.kind, n)))
        .collect()
}

fn clips(streams: &[PvecStream]) -> ClipStream {
    let stream = streams
        .iter()
        .find(|s| s.capability == ClipStream::CAPABILITY)
        .expect("stream 26");
    ClipStream::decode(&stream.payload).unwrap()
}

/// How many items of `kind` the runs of the clip stream assign to clip `clip`.
fn under(stream: &ClipStream, kind: ItemKind, clip: u32) -> u32 {
    stream
        .runs
        .iter()
        .filter(|r| r.kind == kind && r.clip == clip)
        .map(|r| r.count)
        .sum()
}

#[test]
fn text_shown_between_two_strokes_is_between_them_in_the_paint_order() {
    let streams = streams(&format!("{RULE_A}{TEXT}{RULE_B}"), false);
    let texts = TextStream::decode(
        &streams
            .iter()
            .find(|s| s.capability == TextStream::CAPABILITY)
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(texts.records.len(), 1);
    let kinds: Vec<_> = order(&streams).into_iter().collect();
    assert_eq!(
        kinds.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [ItemKind::Stroke, ItemKind::Text, ItemKind::Stroke]
    );
    assert_eq!(kinds[1].1, 0, "the first text record");
}

#[test]
fn two_text_runs_are_numbered_by_their_records_in_order() {
    let streams = streams(&format!("{TEXT}{RULE_A}{TEXT}"), false);
    let items = order(&streams);
    assert_eq!(
        items,
        [
            (ItemKind::Text, 0),
            (ItemKind::Stroke, 1),
            (ItemKind::Text, 1)
        ]
    );
}

#[test]
fn a_raster_between_two_strokes_is_between_them_in_the_paint_order() {
    let items = order(&streams(&format!("{RULE_A}{IMAGE}{RULE_B}"), true));
    assert_eq!(
        items.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [ItemKind::Stroke, ItemKind::Raster, ItemKind::Stroke]
    );
    assert_eq!(items[1].1, 0, "the first raster");
}

#[test]
fn a_raster_between_two_fills_is_between_them_and_text_after_the_second_is_last() {
    let content = format!("0 g 10 10 80 80 re f {IMAGE}1 0 0 rg 200 200 80 80 re f {TEXT}");
    let items = order(&streams(&content, true));
    assert_eq!(
        items,
        [
            (ItemKind::Fill, 0),
            (ItemKind::Raster, 0),
            (ItemKind::Fill, 1),
            (ItemKind::Text, 0)
        ]
    );
}

#[test]
fn a_raster_painted_first_is_first_and_one_painted_last_is_last() {
    let items = order(&streams(&format!("{IMAGE}{RULE_A}{IMAGE}"), true));
    assert_eq!(
        items,
        [
            (ItemKind::Raster, 0),
            (ItemKind::Stroke, 1),
            (ItemKind::Raster, 1)
        ]
    );
}

#[test]
fn no_raster_is_named_when_the_images_are_not_baked() {
    let items = order(&streams(&format!("{RULE_A}{IMAGE}{RULE_B}"), false));
    assert!(items.iter().all(|(k, _)| *k != ItemKind::Raster));
}

#[test]
fn text_under_a_triangle_clip_is_assigned_to_its_record_and_text_after_the_restore_is_not() {
    let streams = streams(&format!("{RULE_A}q {TRIANGLE}W n {TEXT}Q {TEXT}"), false);
    let clips = clips(&streams);
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(under(&clips, ItemKind::Text, 0), 1);
    let texts: u32 = clips
        .runs
        .iter()
        .filter(|r| r.kind == ItemKind::Text)
        .map(|r| r.count)
        .sum();
    assert_eq!(texts, 1, "the text after Q is not clipped");
}

#[test]
fn a_raster_under_a_triangle_clip_is_assigned_to_its_record() {
    let streams = streams(&format!("{RULE_A}q {TRIANGLE}W n {IMAGE}Q {IMAGE}"), true);
    let clips = clips(&streams);
    assert_eq!(under(&clips, ItemKind::Raster, 0), 1);
    let rasters: u32 = clips
        .runs
        .iter()
        .filter(|r| r.kind == ItemKind::Raster)
        .map(|r| r.count)
        .sum();
    assert_eq!(rasters, 1);
}

#[test]
fn strokes_fills_text_and_rasters_under_one_clip_share_its_record() {
    let content = format!("q {TRIANGLE}W n {RULE_A}0 g 50 50 60 60 re f {TEXT}{IMAGE}Q");
    let clips = clips(&streams(&content, true));
    assert_eq!(clips.clips.len(), 1);
    for kind in [
        ItemKind::Stroke,
        ItemKind::Fill,
        ItemKind::Text,
        ItemKind::Raster,
    ] {
        assert_eq!(under(&clips, kind, 0), 1, "{kind:?}");
    }
}

#[test]
fn clipped_items_name_records_that_exist() {
    // The reader refuses a run past the end of a stream (COMP-22), so opening is the check.
    let content = format!("q {TRIANGLE}W n {TEXT}{IMAGE}{RULE_A}Q {RULE_B}");
    let streams = streams(&content, true);
    assert!(clips(&streams).runs.len() >= 3);
    assert_eq!(order(&streams).len(), 4);
}
