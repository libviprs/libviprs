//! Issue #1332 (RL2.1): the clips in force at an image become `visible_box` and alpha.
//!
//! An axis-aligned rectangular clip that lands on pixel edges sets the box and leaves the pixels
//! alone; any other clip is rasterised at image resolution into alpha (nonzero or even-odd, as the
//! clip says) and the box is the clip's bounding box inside the image. A clipped image is PNG or
//! WebP, never JPEG (RASTER-04).

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::TileFormat;
use libviprs::pvec::pdf_image::{
    PdfBakedImage, PdfImageBake, PdfImageError, PdfImageNote, PdfImageOptions, bake_page_images,
};
use libviprs::raster_pvec::pyramid::{RasterChannels, RasterPyramid};

/// An 8 by 8 RGB image, pixel `(x, y)` is `(x * 30, y * 30, 7)`, painted one point per pixel at
/// the page origin, so page `(x, y)` is pixel `(x, 8 - y)`.
fn eight(content: &str) -> Vec<u8> {
    let mut data = Vec::new();
    for y in 0..8u8 {
        for x in 0..8u8 {
            data.extend_from_slice(&[x * 30, y * 30, 7]);
        }
    }
    one(rgb8(8, 8, &data), content)
}

fn bake(bytes: &[u8]) -> PdfImageBake {
    bake_page_images(&page(bytes), &PdfImageOptions::default())
}

fn only(b: &PdfImageBake) -> &PdfBakedImage {
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
    &b.written[0]
}

/// The decoded full resolution tile as RGBA.
fn rgba(p: &RasterPyramid) -> image::RgbaImage {
    let bytes = p
        .tile(u32::from(p.levels) - 1, 0, 0)
        .expect("a stored tile");
    image::load_from_memory(bytes).unwrap().to_rgba8()
}

#[test]
fn an_axis_aligned_rectangular_clip_sets_visible_box_and_leaves_alpha_opaque() {
    // Page (2, 1) to (5, 4): pixels x 2..5, y 8-4..8-1 = 4..7.
    let b = bake(&eight("q 2 1 3 3 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q"));
    let w = only(&b);
    assert_eq!(w.pyramid.visible_box, [2, 4, 5, 7]);
    assert_eq!(
        w.pyramid.channels,
        RasterChannels::Rgb,
        "no alpha was baked"
    );
    assert!(w.notes.contains(&PdfImageNote::Clipped));
    let t = rgba(&w.pyramid);
    assert_eq!(
        t.get_pixel(3, 5).0,
        [90, 150, 7, 255],
        "the pixels are the source's"
    );
    assert_eq!(
        t.get_pixel(0, 0).0,
        [0, 0, 7, 255],
        "outside the box is not blanked"
    );
}

#[test]
fn a_clip_that_is_not_on_pixel_edges_is_baked_into_alpha() {
    // The rectangle edge at x = 2.5 cuts through pixel 2.
    let b = bake(&eight("q 2.5 1 3 3 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q"));
    let w = only(&b);
    assert_eq!(w.pyramid.channels, RasterChannels::Rgba);
    // The clip spans x 2.5 to 5.5, so the box is the outward rounded bounds.
    assert_eq!(w.pyramid.visible_box, [2, 4, 6, 7]);
    let t = rgba(&w.pyramid);
    // A pixel is kept when its centre is in [2.5, 5.5): pixel 2 (centre 2.5) is in, pixel 5
    // (centre 5.5) is out.
    assert_eq!(t.get_pixel(1, 5).0[3], 0);
    assert_eq!(t.get_pixel(2, 5).0[3], 255);
    assert_eq!(t.get_pixel(4, 5).0[3], 255);
    assert_eq!(t.get_pixel(5, 5).0[3], 0);
}

#[test]
fn a_triangular_clip_bakes_alpha_and_sets_the_box() {
    // Page (2, 2), (6.25, 2), (2, 6.25): the hypotenuse is x + y = 8.25.
    let b = bake(&eight(
        "q 2 2 m 6.25 2 l 2 6.25 l h W n q 8 0 0 8 0 0 cm /Im1 Do Q Q",
    ));
    let w = only(&b);
    assert_eq!(w.pyramid.channels, RasterChannels::Rgba);
    // x 2..6.25 is pixels 2..7, page y 2..6.25 is pixel rows 8 - 6.25 = 1.75 to 6, so 1..6.
    assert_eq!(w.pyramid.visible_box, [2, 1, 7, 6]);
    let t = rgba(&w.pyramid);
    for py in 0..8u32 {
        for pxl in 0..8u32 {
            // The centre is (px + 0.5, 7.5 - py) in page space.
            let (x, y) = (f64::from(pxl) + 0.5, 7.5 - f64::from(py));
            let inside = x >= 2.0 && y >= 2.0 && x + y < 8.25;
            let px = t.get_pixel(pxl, py).0;
            assert_eq!(px[3], if inside { 255 } else { 0 }, "pixel ({pxl}, {py})");
            if inside {
                assert_eq!(&px[..3], &[pxl as u8 * 30, py as u8 * 30, 7]);
            }
        }
    }
}

#[test]
fn an_even_odd_ring_clip_punches_the_hole_and_nonzero_does_not() {
    // Two rectangles in one path: the outer page (0,0)-(8,8) and the inner (2,2)-(6,6).
    let ring = |op: &str| {
        eight(&format!(
            "q 0 0 8 8 re 2 2 4 4 re {op} n q 8 0 0 8 0 0 cm /Im1 Do Q Q"
        ))
    };
    let b = bake(&ring("W*"));
    let w = only(&b);
    assert_eq!(w.pyramid.visible_box, [0, 0, 8, 8]);
    let t = rgba(&w.pyramid);
    for py in 0..8u32 {
        for px in 0..8u32 {
            let hole = (2..6).contains(&px) && (2..6).contains(&py);
            assert_eq!(
                t.get_pixel(px, py).0[3],
                if hole { 0 } else { 255 },
                "({px}, {py})"
            );
        }
    }
    // Both rectangles wind the same way, so under nonzero the inner one is inside too.
    let b = bake(&ring("W"));
    let t = rgba(&only(&b).pyramid);
    assert!((0..8).all(|y| (0..8).all(|x| t.get_pixel(x, y).0[3] == 255)));
}

#[test]
fn two_clips_are_intersected() {
    let b = bake(&eight(
        "q 0 0 5 8 re W n q 3 0 5 8 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q Q",
    ));
    let w = only(&b);
    assert_eq!(w.pyramid.visible_box, [3, 0, 5, 8]);
    assert_eq!(w.pyramid.channels, RasterChannels::Rgb);
}

#[test]
fn a_clip_that_leaves_nothing_refuses_the_image_by_name() {
    let b = bake(&eight("q 20 20 5 5 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q"));
    assert!(b.written.is_empty());
    assert!(matches!(b.refused[0].error, PdfImageError::FullyClipped));
    assert_eq!(b.report.refused.get("fully-clipped"), Some(&1));
}

#[test]
fn a_clipped_image_under_jpeg_is_refused_by_raster_04_at_the_producer() {
    let mut o = PdfImageOptions::default();
    o.tile_format = TileFormat::Jpeg { quality: 80 };
    let triangle = eight("q 2 2 m 6.25 2 l 2 6.25 l h W n q 8 0 0 8 0 0 cm /Im1 Do Q Q");
    let b = bake_page_images(&page(&triangle), &o);
    assert!(b.written.is_empty());
    assert!(matches!(
        b.refused[0].error,
        PdfImageError::AlphaNeedsLosslessTiles
    ));
    assert!(b.refused[0].error.to_string().contains("RASTER-04"));

    // A rectangle on pixel edges bakes no alpha, so JPEG stays legal.
    let rect = eight("q 2 1 3 3 re W n q 8 0 0 8 0 0 cm /Im1 Do Q Q");
    let b = bake_page_images(&page(&rect), &o);
    assert_eq!(b.written.len(), 1, "{:?}", b.refused);
    assert_eq!(b.written[0].pyramid.visible_box, [2, 4, 5, 7]);
}
