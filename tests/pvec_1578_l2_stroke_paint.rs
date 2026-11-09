//! Issue #1578, L2: `pdf_page_to_pvec` writes the stroke colour and alpha (stream 24) and the
//! stroke width and dashes (the line style stream) of every stroke, from what the PDF says. The
//! PDF's own operators and the page's own entities are the oracle. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths};
use lopdf::{Object, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{LineStyleStream, StrokePaintStream, StrokeWidth};
use std::path::PathBuf;

fn streams(bytes: Vec<u8>) -> (Option<StrokePaintStream>, Option<LineStyleStream>) {
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let find = |cap| streams.iter().find(|s| s.capability == cap);
    (
        find(StrokePaintStream::CAPABILITY).map(|s| StrokePaintStream::decode(&s.payload).unwrap()),
        find(LineStyleStream::CAPABILITY).map(|s| LineStyleStream::decode(&s.payload).unwrap()),
    )
}

fn convert(content: &str) -> (Option<StrokePaintStream>, Option<LineStyleStream>) {
    let page = page(&pdf(
        doc(),
        Resources::images(Vec::new()),
        content.as_bytes(),
    ));
    streams(pdf_page_to_pvec(&page, "test", "0").unwrap())
}

fn colour_of(paint: &StrokePaintStream, n: usize) -> u32 {
    paint.colours[paint.assignments[n].1 as usize]
}

#[test]
fn a_stroke_colour_is_written_as_an_rgba_word() {
    let (paint, _) = convert("1 0 0 RG 0 0 10 10 re S");
    let paint = paint.expect("stream 24 is written for a coloured stroke");
    assert_eq!(paint.assignments.len(), 1);
    assert_eq!(colour_of(&paint, 0), 0xFF0000FF);
}

#[test]
fn an_uncoloured_stroke_is_black_and_opaque() {
    let (paint, _) = convert("0 0 10 10 re S");
    assert_eq!(colour_of(&paint.unwrap(), 0), 0x000000FF);
}

#[test]
fn the_stroke_alpha_is_the_alpha_byte_not_the_fill_alpha() {
    let gs = Object::Dictionary(dictionary! {"ca" => 0.25, "CA" => 0.5});
    let resources = Resources {
        xobjects: Vec::new(),
        gstates: vec![("GS1", gs)],
    };
    let page = page(&pdf(doc(), resources, b"/GS1 gs 0 0 1 RG 0 0 10 10 re B"));
    let (paint, _) = streams(pdf_page_to_pvec(&page, "test", "0").unwrap());
    assert_eq!(colour_of(&paint.unwrap(), 0), 0x0000FF80);
}

#[test]
fn equal_colours_share_a_table_entry_and_different_ones_do_not() {
    let (paint, _) = convert("1 0 0 RG 0 0 5 5 re S 0 1 0 RG 6 0 5 5 re S 1 0 0 RG 12 0 5 5 re S");
    let paint = paint.unwrap();
    assert_eq!(paint.colours.len(), 2);
    assert_eq!(paint.assignments.len(), 3);
    assert_eq!(colour_of(&paint, 0), colour_of(&paint, 2));
    assert_ne!(colour_of(&paint, 0), colour_of(&paint, 1));
}

#[test]
fn a_width_is_the_pdf_width_times_the_ctm_scale_in_inches() {
    // 3 user units under a scale of 2 is 6 points, which is 6/72 inch.
    let (_, styles) = convert("3 w 2 0 0 2 0 0 cm 0 0 10 10 re S");
    let styles = styles.expect("the line style stream carries the width");
    let StrokeWidth::Width(w) = styles.styles[0].width.unwrap() else {
        panic!("explicit width expected");
    };
    assert!((w - 6.0 / 72.0).abs() < 1e-12, "{w}");
    assert!(styles.styles[0].visible);
}

#[test]
fn no_w_operator_is_the_pdf_default_of_one_unit_and_w_zero_is_the_thinnest_line() {
    let (_, styles) = convert("0 0 10 10 re S");
    let StrokeWidth::Width(w) = styles.unwrap().styles[0].width.unwrap() else {
        panic!("explicit width expected");
    };
    assert!((w - 1.0 / 72.0).abs() < 1e-12, "{w}");
    let (_, styles) = convert("0 w 0 0 10 10 re S");
    assert_eq!(styles.unwrap().styles[0].width, Some(StrokeWidth::Thinnest));
}

#[test]
fn dashes_and_their_phase_are_scaled_like_the_width() {
    let (_, styles) = convert("[4 2] 1 d 0 0 10 10 re S");
    let style = &styles.unwrap().styles[0];
    assert_eq!(style.dashes, vec![4.0 / 72.0, 2.0 / 72.0]);
    assert!((style.phase - 1.0 / 72.0).abs() < 1e-12);
    let (_, styles) = convert("[] 0 d 0 0 10 10 re S");
    assert!(styles.unwrap().styles[0].dashes.is_empty());
}

#[test]
fn a_fill_only_page_writes_no_stroke_colour_and_only_the_hidden_ring_style() {
    let (paint, styles) = convert("1 0 0 rg 0 0 10 10 re f");
    assert!(paint.is_none());
    let styles = styles.unwrap().styles;
    assert_eq!(styles.len(), 1);
    assert!(!styles[0].visible);
}

#[test]
fn a_filled_and_stroked_path_colours_its_outline_only() {
    let (paint, _) = convert("0 0 1 rg 1 0 0 RG 0 0 10 10 re B");
    let paint = paint.unwrap();
    assert_eq!(paint.assignments.len(), 1);
    assert_eq!(colour_of(&paint, 0), 0xFF0000FF);
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
fn blueprint_mix_every_stroke_carries_the_colour_and_width_the_pdf_painted_it_with() {
    let Some(path) = blueprint_mix() else { return };
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let (paint, styles) = streams(pdf_page_to_pvec(&page, "test", "0").unwrap());
    let (paint, styles) = (paint.unwrap(), styles.unwrap());
    // The oracle: the distinct (colour, alpha) of every stroked path of the page, and the distinct
    // widths of the same, read straight from the PDF's entities.
    let mut wanted: Vec<u32> = page
        .entities
        .iter()
        .filter(|e| e.paint.stroke && !e.segments.is_empty())
        .map(|e| {
            u32::from_be_bytes([
                e.stroke_colour[0],
                e.stroke_colour[1],
                e.stroke_colour[2],
                e.stroke_alpha,
            ])
        })
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    let mut written = paint.colours.clone();
    written.sort_unstable();
    assert_eq!(written, wanted);
    assert!(
        wanted.len() > 1,
        "the fixture strokes in more than one colour, so one colour for all would be wrong"
    );
    assert_eq!(paint.assignments.len(), styles.assignments.len());
    assert!(styles.styles.iter().all(|s| s.width.is_some()));
}
