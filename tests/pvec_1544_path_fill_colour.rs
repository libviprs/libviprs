//! Issue #1544: a painted path reports the fill colour in force, so a converter can tell a white
//! fill from a grey one.

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;

fn fills(content: &str) -> Vec<[u8; 3]> {
    let bytes = pdf(doc(), Resources::images(Vec::new()), content.as_bytes());
    page(&bytes)
        .entities
        .iter()
        .filter(|e| e.paint.fill.is_some())
        .map(|e| e.fill_colour)
        .collect()
}

#[test]
fn a_white_and_a_grey_fill_report_their_own_colours() {
    let got = fills("1 1 1 rg 0 0 10 10 re f 0.5 g 20 0 10 10 re f");
    assert_eq!(got, vec![[255, 255, 255], [128, 128, 128]]);
}

#[test]
fn a_fill_with_no_colour_set_is_black() {
    assert_eq!(fills("0 0 10 10 re f"), vec![[0, 0, 0]]);
}

#[test]
fn q_and_big_q_restore_the_colour_for_the_next_path() {
    let got = fills("q 1 0 0 rg 0 0 10 10 re f Q 20 0 10 10 re f");
    assert_eq!(got, vec![[255, 0, 0], [0, 0, 0]]);
}

#[test]
fn a_cmyk_fill_is_converted_to_rgb() {
    assert_eq!(fills("0 0 0 1 k 0 0 10 10 re f"), vec![[0, 0, 0]]);
    assert_eq!(fills("0 0 0 0 k 0 0 10 10 re f"), vec![[255, 255, 255]]);
}
