//! Pins the morphology call surface required by the libviprs-tests ported
//! suite (libviprs-tests issue #55, `tests/ported_morphology.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: method names, argument types (including the
//! `&[&[u8]]` structuring-element shape and the `Direction` enum), and
//! return types. Behaviour depth is covered by the unit tests in
//! `src/morphology.rs`; this file is the API contract.
//!
//! Every ported morphology test builds its image synthetically (as the
//! libvips originals do), so the bodies here are kept literal, including
//! the `test_labelregions` segment-count assertion: libvips assigns
//! region serials from 1 and reports `regions + 1` as the segment count,
//! so one circle on black is labels 1 and 2 with a count of 3.

use libviprs::{Direction, PixelFormat, Raster};

/// The ported suite's `black_image` helper.
fn black_image(w: u32, h: u32) -> Raster {
    Raster::zeroed(w, h, PixelFormat::Gray8).unwrap()
}

/// The ported `test_countlines` body.
#[test]
fn ported_countlines_call_site() {
    let mut im = black_image(100, 100);

    // Draw a horizontal line: ink=255 from (0,50) to (100,50)
    im.draw_line(&[255], 0, 50, 100, 50);

    let n_lines = im.countlines(Direction::Horizontal);
    assert_eq!(n_lines, 1.0);
}

/// The ported `test_labelregions` body.
#[test]
fn ported_labelregions_call_site() {
    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[255], 50, 50, 25);

    let (mask, segments) = im.label_regions();
    assert_eq!(segments, 3);

    let max_label = mask.data().iter().copied().max().unwrap();
    assert_eq!(max_label, 2);
}

/// The ported `test_erode` body.
#[test]
fn ported_erode_call_site() {
    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[255], 50, 50, 25);

    let kernel: &[&[u8]] = &[&[128, 255, 128], &[255, 255, 255], &[128, 255, 128]];
    let im2 = im.erode(kernel);

    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());
    assert_eq!(im.format(), im2.format());

    let avg_before: f64 = im.data().iter().map(|&b| b as f64).sum::<f64>() / im.data().len() as f64;
    let avg_after: f64 =
        im2.data().iter().map(|&b| b as f64).sum::<f64>() / im2.data().len() as f64;
    assert!(
        avg_before > avg_after,
        "Erosion should reduce the average pixel value: before={avg_before}, after={avg_after}"
    );
}

/// The ported `test_dilate` body.
#[test]
fn ported_dilate_call_site() {
    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[255], 50, 50, 25);

    let kernel: &[&[u8]] = &[&[128, 255, 128], &[255, 255, 255], &[128, 255, 128]];
    let im2 = im.dilate(kernel);

    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());
    assert_eq!(im.format(), im2.format());

    let avg_before: f64 = im.data().iter().map(|&b| b as f64).sum::<f64>() / im.data().len() as f64;
    let avg_after: f64 =
        im2.data().iter().map(|&b| b as f64).sum::<f64>() / im2.data().len() as f64;
    assert!(
        avg_after > avg_before,
        "Dilation should increase the average pixel value: before={avg_before}, after={avg_after}"
    );
}

/// The ported `test_rank` body.
#[test]
fn ported_rank_call_site() {
    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[255], 50, 50, 25);

    let im2 = im.rank(3, 3, 8);

    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());
    assert_eq!(im.format(), im2.format());

    let avg_before: f64 = im.data().iter().map(|&b| b as f64).sum::<f64>() / im.data().len() as f64;
    let avg_after: f64 =
        im2.data().iter().map(|&b| b as f64).sum::<f64>() / im2.data().len() as f64;
    assert!(
        avg_after > avg_before,
        "Max rank filter should increase average: before={avg_before}, after={avg_after}"
    );
}
