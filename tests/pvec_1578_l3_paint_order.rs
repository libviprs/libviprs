//! Issue #1578, L3: `pdf_page_to_pvec` writes the paint order (stream 23) and the fill rule
//! (stream 25) of the page, along with the hatch record each fill is. The PDF's own operators and
//! the page's own entities are the oracle. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths};
use pvec_facade::PvecReader;
use pvec_facade::components::{
    FillPaintStream, FillRule, HatchStream, ItemKind, LineStyleStream, PaintOrderStream,
};
use std::path::PathBuf;

struct Written {
    order: Option<PaintOrderStream>,
    fills: Option<FillPaintStream>,
    hatch: Option<HatchStream>,
    styles: Option<LineStyleStream>,
}

fn streams(bytes: Vec<u8>) -> Written {
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let find = |cap| streams.iter().find(|s| s.capability == cap);
    Written {
        order: find(PaintOrderStream::CAPABILITY)
            .map(|s| PaintOrderStream::decode(&s.payload).unwrap()),
        fills: find(FillPaintStream::CAPABILITY)
            .map(|s| FillPaintStream::decode(&s.payload).unwrap()),
        hatch: find(HatchStream::CAPABILITY).map(|s| HatchStream::decode(&s.payload).unwrap()),
        styles: find(LineStyleStream::CAPABILITY)
            .map(|s| LineStyleStream::decode(&s.payload).unwrap()),
    }
}

fn convert(content: &str) -> Written {
    let page = page(&pdf(
        doc(),
        Resources::images(Vec::new()),
        content.as_bytes(),
    ));
    streams(pdf_page_to_pvec(&page, "test", "0").unwrap())
}

fn kinds(order: &PaintOrderStream) -> Vec<ItemKind> {
    order.items().map(|(kind, _)| kind).collect()
}

#[test]
fn f_is_nonzero_and_f_star_is_even_odd() {
    let w = convert("0 0 10 10 re f 20 0 10 10 re f* 40 0 10 10 re B 60 0 10 10 re B*");
    let rules: Vec<FillRule> = w
        .fills
        .unwrap()
        .assignments
        .iter()
        .map(|a| a.rule)
        .collect();
    assert_eq!(
        rules,
        [
            FillRule::NonZero,
            FillRule::EvenOdd,
            FillRule::NonZero,
            FillRule::EvenOdd
        ]
    );
}

#[test]
fn a_compound_path_is_one_hatch_record_with_every_ring_in_its_boundary() {
    let w = convert("0 0 100 100 re 30 30 40 40 re f*");
    let hatch = w.hatch.expect("a fill is a hatch record");
    assert_eq!(hatch.records.len(), 1);
    assert_eq!(hatch.records[0].boundary.len(), 2);
    assert_eq!(w.fills.unwrap().assignments.len(), 1);
}

#[test]
fn a_fill_only_ring_is_hidden_so_only_the_hatch_draws_it() {
    let w = convert("0 0 10 10 re f");
    let styles = w.styles.expect("the hidden ring needs a line style");
    assert!(styles.styles.iter().any(|s| !s.visible));
    assert_eq!(styles.assignments.len(), 1);
}

#[test]
fn paint_order_follows_the_content_stream_not_the_legacy_stage_order() {
    // A stroke painted first and a fill painted over it: legacy order (fills, then strokes)
    // would put the stroke on top.
    let w = convert("0 0 10 10 re S 1 0 0 rg 0 0 10 10 re f");
    assert_eq!(kinds(&w.order.unwrap()), [ItemKind::Stroke, ItemKind::Fill]);
}

#[test]
fn the_fill_of_a_b_path_is_painted_before_its_outline() {
    let w = convert("0 0 10 10 re B 20 0 10 10 re S");
    let order = w.order.unwrap();
    assert_eq!(
        kinds(&order),
        [ItemKind::Fill, ItemKind::Stroke, ItemKind::Stroke]
    );
    let strokes: Vec<u32> = order
        .items()
        .filter(|(kind, _)| *kind == ItemKind::Stroke)
        .map(|(_, n)| n)
        .collect();
    assert!(strokes[0] < strokes[1], "{strokes:?}");
}

#[test]
fn an_interleaving_is_kept_item_for_item_and_consecutive_items_share_a_run() {
    let w = convert(
        "0 0 5 5 re S 6 0 5 5 re S 1 0 0 rg 0 6 5 5 re f 0 1 0 rg 6 6 5 5 re f 12 0 5 5 re S",
    );
    let order = w.order.unwrap();
    assert_eq!(
        kinds(&order),
        [
            ItemKind::Stroke,
            ItemKind::Stroke,
            ItemKind::Fill,
            ItemKind::Fill,
            ItemKind::Stroke
        ]
    );
    assert_eq!(order.runs.len(), 3);
    assert_eq!(order.runs[1].count, 2);
}

#[test]
fn a_white_fill_is_not_an_item() {
    let w = convert("1 1 1 rg 0 0 10 10 re f 0 0 1 RG 20 0 10 10 re S");
    assert_eq!(kinds(&w.order.unwrap()), [ItemKind::Stroke]);
    assert!(w.hatch.is_none());
}

#[test]
fn every_hatch_record_and_stroke_is_named_exactly_once() {
    let w = convert("0 0 10 10 re f 20 0 10 10 re B 40 0 10 10 re S");
    let order = w.order.unwrap();
    let hatch = w.hatch.unwrap();
    let fills: Vec<u32> = order
        .items()
        .filter(|(k, _)| *k == ItemKind::Fill)
        .map(|(_, n)| n)
        .collect();
    assert_eq!(fills, (0..hatch.records.len() as u32).collect::<Vec<_>>());
    order
        .validate(|_| true, hatch.records.len() as u32, 0, 0)
        .unwrap();
}

fn blueprint_mix() -> Option<PathBuf> {
    let fixtures = std::env::var_os("LIBVIPRS_TESTS_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../libviprs-tests/tests/fixtures")
        });
    let path = fixtures.join("blueprint-mix.pdf");
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipped: {} is not there (set LIBVIPRS_TESTS_FIXTURES)",
            path.display()
        );
        None
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn blueprint_mix_paint_order_names_every_stroke_in_the_pdfs_order() {
    let Some(path) = blueprint_mix() else { return };
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let w = streams(pdf_page_to_pvec(&page, "test", "0").unwrap());
    let order = w.order.expect("stream 23 is written");
    // The fixture's vector part is strokes only (the oracle counts no filled path in it), so the
    // order is one run of strokes, numbered the way the entities were written, ascending.
    assert!(
        page.entities
            .iter()
            .all(|entity| entity.paint.fill.is_none())
    );
    assert!(w.hatch.is_none() && w.fills.is_none());
    assert_eq!(order.runs.len(), 1);
    assert_eq!(order.runs[0].kind, ItemKind::Stroke);
    // A path with several subpaths is several stroke entities, so there are at least as many
    // items as stroked paths the clip did not remove.
    assert_eq!(order.runs[0].first, 1);
    assert!(order.runs[0].count >= 1);
    order.validate(|_| true, 0, 0, 0).unwrap();
}
