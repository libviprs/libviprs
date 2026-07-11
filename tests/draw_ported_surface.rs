//! Pins the drawing call surface required by the libviprs-tests ported suite
//! (libviprs-tests issue #55, `tests/ported_draw.rs`, plus the `draw_line`
//! call site in `ported_morphology.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types (`ink: &[u8]`, `i32`
//! coordinates, `&Raster` mask/overlay operands), and `draw_flood`'s
//! `Result<(), DrawError>`. Each test body reproduces its ported counterpart
//! literally, down to the asserted pixel results; the one liberty taken is
//! `.unwrap()` on `draw_flood`, whose `Result` the ported happy path drops on
//! the floor. Behavior depth beyond the ported assertions is covered by the
//! unit tests in `src/draw.rs`.
//!
//! The ported doc block for `draw_flood` paraphrases the fill as "pixels that
//! match the value at (x, y)"; the referenced libvips `test_draw.py` actually
//! calls the default (bounded) variant, which fills up to an ink-coloured
//! edge. Both variants agree on the outlined-circle scenario asserted here;
//! the blob variant ships as `draw_flood_blob`, matching libvips
//! `draw_flood` with `equal` set.

use libviprs::draw::DrawOp;
use libviprs::{PixelFormat, Raster};

/// The ported `black_image`: a black (all-zero) single-band u8 image.
fn black_image(w: u32, h: u32) -> Raster {
    Raster::zeroed(w, h, PixelFormat::Gray8).unwrap()
}

/// The ported `pixel_at`: pixel value at (x, y) of a single-band image.
fn pixel_at(im: &Raster, x: u32, y: u32) -> u8 {
    let view = im.region(x, y, 1, 1).unwrap();
    view.pixel(0, 0).unwrap()[0]
}

/// The ported `abs_max_diff`: max absolute difference between two same-sized
/// single-band images.
fn abs_max_diff(a: &Raster, b: &Raster) -> u8 {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    a.data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// ported_draw.rs::test_draw_circle
#[test]
fn test_draw_circle() {
    let mut im = black_image(100, 100);
    im.draw_circle(&[100], 50, 50, 25);
    assert_eq!(pixel_at(&im, 25, 50), 100);
    assert_eq!(pixel_at(&im, 26, 50), 0);

    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[100], 50, 50, 25);
    assert_eq!(pixel_at(&im, 25, 50), 100);
    assert_eq!(pixel_at(&im, 26, 50), 100);
    assert_eq!(pixel_at(&im, 24, 50), 0);
}

/// ported_draw.rs::test_draw_flood
#[test]
fn test_draw_flood() {
    let mut im = black_image(100, 100);
    im.draw_circle(&[100], 50, 50, 25);
    im.draw_flood(&[100], 50, 50).unwrap();

    let mut im2 = black_image(100, 100);
    im2.draw_circle_filled(&[100], 50, 50, 25);

    assert_eq!(
        abs_max_diff(&im, &im2),
        0,
        "Flood-filled outline should match filled circle"
    );
}

/// ported_draw.rs::test_draw_flood_out_of_bounds
#[test]
fn test_draw_flood_out_of_bounds() {
    let mut im = black_image(100, 100);

    assert!(im.draw_flood(&[100], 200, 50).is_err());
    assert!(im.draw_flood(&[100], 50, 200).is_err());
    assert!(im.draw_flood(&[100], -1, 50).is_err());
    assert!(im.draw_flood(&[100], 50, -1).is_err());
}

/// ported_draw.rs::test_draw_image
#[test]
fn test_draw_image() {
    let mut small = black_image(51, 51);
    small.draw_circle_filled(&[100], 25, 25, 25);

    let mut im2 = black_image(100, 100);
    im2.draw_image(&small, 25, 25);

    let mut im3 = black_image(100, 100);
    im3.draw_circle_filled(&[100], 50, 50, 25);

    assert_eq!(
        abs_max_diff(&im2, &im3),
        0,
        "draw_image should match direct filled circle"
    );
}

/// ported_draw.rs::test_draw_line, plus the `ported_morphology.rs`
/// `test_countlines` call shape (`draw_line(&[255], 0, 50, 100, 50)`).
#[test]
fn test_draw_line() {
    let mut im = black_image(100, 100);
    im.draw_line(&[100], 0, 0, 100, 0);

    assert_eq!(pixel_at(&im, 0, 0), 100);
    assert_eq!(pixel_at(&im, 0, 1), 0);

    let mut im = black_image(100, 100);
    im.draw_line(&[255], 0, 50, 100, 50);
    assert_eq!(pixel_at(&im, 0, 50), 255);
    assert_eq!(pixel_at(&im, 99, 50), 255);
}

/// ported_draw.rs::test_draw_mask
#[test]
fn test_draw_mask() {
    let mut mask = black_image(51, 51);
    mask.draw_circle_filled(&[128], 25, 25, 25);

    let mut im = black_image(100, 100);
    im.draw_mask(&[200], &mask, 25, 25);

    let mut im2 = black_image(100, 100);
    im2.draw_circle_filled(&[100], 50, 50, 25);

    assert_eq!(abs_max_diff(&im, &im2), 0, "Mask-drawn image should match");
}

/// ported_draw.rs::test_draw_rect
#[test]
fn test_draw_rect() {
    let mut im = black_image(100, 100);
    im.draw_rect_filled(&[100], 25, 25, 50, 50);

    let mut im2 = black_image(100, 100);
    for y in 25..75 {
        im2.draw_line(&[100], 25, y, 74, y);
    }

    assert_eq!(
        abs_max_diff(&im, &im2),
        0,
        "Filled rect should match line-drawn region"
    );
}

/// ported_draw.rs::test_draw_smudge
#[test]
fn test_draw_smudge() {
    let mut im = black_image(100, 100);
    im.draw_circle_filled(&[100], 50, 50, 25);

    let mut im2 = im.clone();
    im2.draw_smudge(10, 10, 50, 50);

    let patch = im.extract(10, 10, 50, 50).unwrap();

    let mut im4 = im2.clone();
    im4.draw_image(&patch, 10, 10);

    assert_eq!(
        abs_max_diff(&im4, &im),
        0,
        "Restoring the original region after smudge should recover the original image"
    );
}

/// The extension seam holds from an external crate: `DrawOp` is implementable
/// out of tree with nothing but the public surface, and such an op drives the
/// same generic entry point as the built-ins, composing with them.
#[test]
fn external_custom_draw_op_composes_with_builtins() {
    /// A checkerboard stipple, an op libviprs does not ship.
    struct Stipple<'a> {
        ink: &'a [u8],
    }

    impl DrawOp for Stipple<'_> {
        fn apply(&self, raster: &mut Raster) {
            for y in 0..raster.height() as i32 {
                for x in 0..raster.width() as i32 {
                    if (x + y) % 2 == 0 {
                        raster.put_pixel(x, y, self.ink);
                    }
                }
            }
        }
    }

    let mut im = black_image(8, 8);
    im.draw(&Stipple { ink: &[10] });
    // A built-in op composes on top of the custom one.
    im.draw_line(&[200], 0, 3, 7, 3);
    assert_eq!(pixel_at(&im, 0, 0), 10);
    assert_eq!(pixel_at(&im, 1, 0), 0);
    assert_eq!(pixel_at(&im, 5, 3), 200, "line overwrites the stipple");

    // And the custom op erases to a trait object like any built-in.
    let boxed: Box<dyn DrawOp> = Box::new(Stipple { ink: &[10] });
    let mut im2 = black_image(4, 4);
    im2.draw(boxed.as_ref());
    assert_eq!(pixel_at(&im2, 2, 2), 10);
}
