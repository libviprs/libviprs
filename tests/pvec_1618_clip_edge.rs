//! Issue #1618: a stroke that lies on the edge of a rectangular clip is kept (the clip is a closed
//! region, with a small tolerance), a stroke strictly outside is dropped, and one crossing the edge
//! is still cut. The content stream is the oracle. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_page_to_pvec;
use pvec_facade::PvecReader;
use pvec_facade::components::{ItemKind, PaintOrderStream};

/// How many strokes the PVec of a page drawing `content` holds. A marker stroke outside any clip
/// follows it (a page with no path at all does not convert), and is not counted.
fn strokes(content: &str) -> u32 {
    let content = format!("{content} 1 1 m 9 1 l S");
    let page = page(&pdf(
        doc(),
        Resources::images(Vec::new()),
        content.as_bytes(),
    ));
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    streams
        .iter()
        .find(|s| s.capability == PaintOrderStream::CAPABILITY)
        .map(|s| PaintOrderStream::decode(&s.payload).unwrap())
        .map(|order| {
            order
                .runs
                .iter()
                .filter(|r| r.kind == ItemKind::Stroke)
                .map(|r| r.count)
                .sum()
        })
        .unwrap_or(0)
        - 1
}

#[test]
fn a_stroke_on_the_clip_edge_is_kept() {
    assert_eq!(
        strokes("q 100 100 200 200 re W n 100 120 m 100 280 l S Q"),
        1
    );
    assert_eq!(
        strokes("q 100 100 200 200 re W n 120 300 m 280 300 l S Q"),
        1
    );
}

#[test]
fn a_stroke_on_an_edge_that_is_not_exactly_representable_is_kept() {
    // 61.88 + 100 is not exactly 161.88 in binary, the edge the stroke is drawn on.
    assert_eq!(
        strokes("q 61.88 50 100 100 re W n 161.88 60 m 161.88 140 l S Q"),
        1
    );
}

#[test]
fn a_stroke_strictly_outside_the_clip_is_dropped() {
    assert_eq!(strokes("q 100 100 200 200 re W n 99 120 m 99 280 l S Q"), 0);
    assert_eq!(
        strokes("q 100 100 200 200 re W n 120 301 m 280 301 l S Q"),
        0
    );
}

#[test]
fn a_stroke_crossing_the_edge_is_still_cut() {
    assert_eq!(
        strokes("q 100 100 200 200 re W n 50 200 m 200 200 l S Q"),
        1
    );
    // The run that starts outside, crosses in and leaves again stays one stroke per piece.
    assert_eq!(
        strokes("q 100 100 200 200 re W n 50 200 m 350 200 l S Q"),
        1
    );
}
