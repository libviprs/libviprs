//! Issue #1578, L4 (non-rectangular clips): a `W` or `W*` clip that is not an axis-aligned
//! rectangle is written as capability 26. Its path becomes hidden closed ring entities (a hidden
//! style in the line style stream), the clip record carries the rule and the parent (nesting is
//! the parent chain, 8 deep at most), and the strokes and fills drawn under it are named by runs
//! (a fill by its hatch record). A
//! rectangular clip is still cut into the geometry and writes no record. The PDF's own operators
//! are the oracle; `blueprint-mix.pdf` has rectangles only, so its bytes stay the pinned ones in
//! `pvec_1578_pdf_page_to_pvec.rs`. Needs `--features pvec-write`.
//!
//! Left for L5: the fill rings themselves are not cut to the clip yet. The circle, even-odd and
//! triangle regions of `fid-clips` need that on top of this.

#![cfg(feature = "pvec-write")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_page_to_pvec;
use pvec_facade::PvecReader;
use pvec_facade::components::{
    ClipRecord, ClipStream, FillRule, ItemKind, LineStyleStream, Stream,
};

const TRIANGLE: &str = "40 40 m 280 40 l 160 190 l h ";

struct Out {
    clips: Option<ClipStream>,
    styles: Option<LineStyleStream>,
}

fn convert(content: &str) -> Out {
    let page = page(&pdf(
        doc(),
        Resources::images(Vec::new()),
        content.as_bytes(),
    ));
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let find = |cap| streams.iter().find(|s: &&Stream| s.capability == cap);
    Out {
        clips: find(ClipStream::CAPABILITY).map(|s| ClipStream::decode(&s.payload).unwrap()),
        styles: find(LineStyleStream::CAPABILITY)
            .map(|s| LineStyleStream::decode(&s.payload).unwrap()),
    }
}

fn hatch(first: i32, lines: i32) -> String {
    (first..first + lines)
        .map(|i| format!("{} 30 m {} 230 l S ", i * 10, i * 10 + 200))
        .collect()
}

fn circle(cx: f64, cy: f64, r: f64) -> String {
    let k = 0.5522847498 * r;
    format!(
        "{} {} m {} {} {} {} {} {} c {} {} {} {} {} {} c {} {} {} {} {} {} c {} {} {} {} {} {} c h ",
        cx + r,
        cy,
        cx + r,
        cy + k,
        cx + k,
        cy + r,
        cx,
        cy + r,
        cx - k,
        cy + r,
        cx - r,
        cy + k,
        cx - r,
        cy,
        cx - r,
        cy - k,
        cx - k,
        cy - r,
        cx,
        cy - r,
        cx + k,
        cy - r,
        cx + r,
        cy - k,
        cx + r,
        cy,
    )
}

/// The strokes the runs assign to clip `clip`.
fn items(stream: &ClipStream, clip: u32) -> u32 {
    kind_items(stream, ItemKind::Stroke, clip)
}

fn kind_items(stream: &ClipStream, kind: ItemKind, clip: u32) -> u32 {
    stream
        .runs
        .iter()
        .filter(|r| r.kind == kind && r.clip == clip)
        .map(|r| r.count)
        .sum()
}

fn is_hidden(styles: &LineStyleStream, record: &ClipRecord) -> bool {
    record.boundary.iter().all(|ring| {
        styles
            .assignments
            .iter()
            .find(|(id, _)| id == ring)
            .is_some_and(|&(_, style)| !styles.styles[style as usize].visible)
    })
}

#[test]
fn a_triangle_clip_is_a_record_with_one_hidden_ring_and_a_run_over_the_strokes_under_it() {
    let out = convert(&format!("q {TRIANGLE}W n {}Q", hatch(0, 10)));
    let clips = out.clips.expect("stream 26 is written for a triangle clip");
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(clips.clips[0].rule, FillRule::NonZero);
    assert_eq!(clips.clips[0].parent, None);
    assert_eq!(clips.clips[0].boundary.len(), 1);
    assert_eq!(items(&clips, 0), 10);
    let styles = out.styles.expect("the ring is hidden by a line style");
    assert!(is_hidden(&styles, &clips.clips[0]));
}

#[test]
fn a_stroke_after_the_restore_is_not_clipped() {
    let out = convert(&format!("q {TRIANGLE}W n {}Q {}", hatch(0, 3), hatch(5, 2)));
    assert_eq!(items(&out.clips.unwrap(), 0), 3);
}

#[test]
fn an_even_odd_clip_of_two_circles_is_one_record_with_two_rings_and_the_even_odd_rule() {
    let content = format!(
        "q {}{}W* n {}Q",
        circle(450.0, 330.0, 90.0),
        circle(450.0, 330.0, 45.0),
        hatch(0, 4)
    );
    let out = convert(&content);
    let clips = out.clips.unwrap();
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(clips.clips[0].rule, FillRule::EvenOdd);
    assert_eq!(clips.clips[0].boundary.len(), 2);
    assert_eq!(items(&clips, 0), 4);
    assert!(is_hidden(&out.styles.unwrap(), &clips.clips[0]));
}

#[test]
fn a_nested_clip_has_the_outer_one_as_its_parent_and_each_stroke_is_in_the_innermost() {
    let inner = "100 60 m 200 60 l 150 120 l h ";
    let content = format!(
        "q {TRIANGLE}W n {}q {inner}W n {}Q {}Q",
        hatch(0, 2),
        hatch(3, 3),
        hatch(7, 1)
    );
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 2);
    assert_eq!(clips.clips[1].parent, Some(0));
    assert_eq!(items(&clips, 0), 3);
    assert_eq!(items(&clips, 1), 3);
}

#[test]
fn a_chain_deeper_than_eight_keeps_the_first_eight_and_still_converts() {
    let mut content = String::new();
    for i in 0..10 {
        content += &format!("q {} 40 m 280 40 l 160 190 l h W n ", 40 - i);
    }
    content += &hatch(0, 2);
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 8);
    assert_eq!(clips.chain_of(7).len(), 8);
    assert_eq!(items(&clips, 7), 2);
}

#[test]
fn a_rectangular_clip_is_cut_into_the_geometry_and_writes_no_record() {
    let out = convert(&format!("q 0 0 150 100 re W n {}Q", hatch(0, 3)));
    assert!(out.clips.is_none());
}

#[test]
fn a_rectangle_inside_a_triangle_cuts_the_geometry_and_only_the_triangle_is_a_record() {
    let content = format!("q {TRIANGLE}W n q 0 0 150 100 re W n {}Q Q", hatch(0, 3));
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(items(&clips, 0), 3);
}

#[test]
fn two_siblings_are_two_roots_and_share_no_item() {
    let other = "300 60 m 400 60 l 350 120 l h ";
    let content = format!(
        "q {TRIANGLE}W n {}Q q {other}W n {}Q",
        hatch(0, 2),
        hatch(4, 2)
    );
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 2);
    assert_eq!(clips.clips[1].parent, None);
    assert_eq!((items(&clips, 0), items(&clips, 1)), (2, 2));
}

#[test]
fn the_same_clip_used_twice_in_a_row_is_one_record() {
    let content = format!("q {TRIANGLE}W n {}{}Q", hatch(0, 2), hatch(5, 2));
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(items(&clips, 0), 4);
}

#[test]
fn fills_under_a_general_clip_are_runs_over_their_hatch_records() {
    // Two filled squares under a triangle, one after the restore, and a stroke between them.
    let content = format!(
        "q {TRIANGLE}W n 1 0 0 rg 50 50 40 40 re f {}0 0 1 rg 100 50 40 40 re f Q 0 1 0 rg 200 50 40 40 re f",
        hatch(0, 1)
    );
    let clips = convert(&content).clips.unwrap();
    assert_eq!(clips.clips.len(), 1);
    assert_eq!(kind_items(&clips, ItemKind::Fill, 0), 2);
    assert_eq!(items(&clips, 0), 1);
    let fills: Vec<_> = clips
        .runs
        .iter()
        .filter(|r| r.kind == ItemKind::Fill)
        .map(|r| (r.first, r.count))
        .collect();
    assert_eq!(fills, vec![(0, 2)]);
}
