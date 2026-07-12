//! Pins the mosaicing call surface required by the libviprs-tests ported
//! suite (libviprs-tests issue #55, `tests/ported_mosaicing.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: method names, argument types (the
//! `MergeDirection` enum and `i32` offsets/tie-points), and return types.
//! Behaviour depth is covered by the unit tests in `src/mosaicing.rs`;
//! this file is the API contract.
//!
//! The ported tests decode the libvips `cd*.jpg` mosaic fixtures; the
//! setups here reproduce that character with synthetic one-band crops of
//! a shared deterministic textured scene, so the tie-point search has
//! real structure to lock onto, and the merge / mosaic / global-balance
//! expressions are kept literal. The exact fixture dimensions the ported
//! tests assert (1014x379 and friends) are covered by the ported suite
//! itself once its reference images are fetched; here the same
//! assertions run against the synthetic scene's known true geometry.

use libviprs::{MergeDirection, PixelFormat, Raster};

/// A crop of the shared textured scene, standing in for one `cd*.jpg`
/// mosaic fixture (one band, no zero pixels, high local contrast).
fn scene_crop(w: u32, h: u32, ox: u32, oy: u32) -> Raster {
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut v = (x + ox).wrapping_mul(0x9E37_79B9) ^ (y + oy).wrapping_mul(0x85EB_CA6B);
            v ^= v >> 13;
            v = v.wrapping_mul(0xC2B2_AE35);
            v ^= v >> 16;
            data.push(30 + (v % 196) as u8);
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

/// The ported `test_lrmerge` body, on synthetic fixtures.
#[test]
fn ported_lrmerge_call_site() {
    let left = scene_crop(240, 180, 0, 0);
    let right = scene_crop(240, 170, 230, 6);

    let dx = 10 - left.width() as i32;
    let join = left.merge(&right, MergeDirection::Horizontal, dx, 0);

    assert_eq!(join.width(), left.width() + right.width() - 10);
    assert_eq!(join.height(), left.height().max(right.height()));
    assert_eq!(join.format().channels(), 1);
}

/// The ported `test_tbmerge` body, on synthetic fixtures.
#[test]
fn ported_tbmerge_call_site() {
    let top = scene_crop(180, 240, 0, 0);
    let bottom = scene_crop(170, 240, 6, 230);

    let dy = 10 - top.height() as i32;
    let join = top.merge(&bottom, MergeDirection::Vertical, 0, dy);

    assert_eq!(join.width(), top.width().max(bottom.width()));
    assert_eq!(join.height(), top.height() + bottom.height() - 10);
    assert_eq!(join.format().channels(), 1);
}

/// The ported `test_lrmosaic` body: the tie-point call shape, with the
/// dimension assertions against the synthetic scene's known true geometry
/// (`right` really sits at scene (240, 4), so the joined image spans
/// 540x240).
#[test]
fn ported_lrmosaic_call_site() {
    let left = scene_crop(300, 240, 0, 0);
    let right = scene_crop(300, 234, 240, 4);

    // Tie-point: scene (270, 120) is left (270, 120) and right (30, 116).
    let ref_x = left.width() as i32 - 30;
    let join = left.mosaic(&right, MergeDirection::Horizontal, ref_x, 120, 30, 116);

    assert_eq!(join.width(), 540);
    assert_eq!(join.height(), 240);
}

/// The ported `test_tbmosaic` body, transposed geometry.
#[test]
fn ported_tbmosaic_call_site() {
    let top = scene_crop(240, 300, 0, 0);
    let bottom = scene_crop(234, 300, 4, 240);

    // Tie-point: scene (120, 270) is top (120, 270) and bottom (116, 30).
    let ref_y = top.height() as i32 - 30;
    let join = top.mosaic(&bottom, MergeDirection::Vertical, 120, ref_y, 116, 30);

    assert_eq!(join.width(), 240);
    assert_eq!(join.height(), 540);
}

/// The ported `test_mosaic` loop shape: horizontal pairs joined
/// vertically through the same iterative `Option` fold, on a 2x2 grid of
/// crops of one shared scene.
#[test]
fn ported_mosaic_call_site() {
    // Two horizontal pairs: the top row at scene y 0, the bottom row at
    // scene y 190, each pair overlapping 60 columns around scene x 270.
    let files = [
        scene_crop(300, 240, 0, 0),
        scene_crop(300, 234, 240, 4),
        scene_crop(300, 240, 2, 190),
        scene_crop(300, 230, 242, 194),
    ];
    // Tie-point marks per pair, like MOSAIC_MARKS: scene (270, 120) for
    // the first pair, scene (272, 310) for the second.
    let marks = [(270, 120), (30, 116), (270, 120), (30, 116)];
    // Vertical marks, like MOSAIC_VERTICAL_MARKS: scene (150, 210) seen
    // by the finished strips (strip 2 sits at (2, 190) under strip 1).
    let vertical_marks = [(148, 20), (150, 210)];

    let mut mosaiced: Option<Raster> = None;

    for i in (0..files.len()).step_by(2) {
        let im = &files[i];
        let sec_im = &files[i + 1];

        let (ref_x, ref_y) = marks[i];
        let (sec_x, sec_y) = marks[i + 1];

        let horizontal_part = im.mosaic(
            sec_im,
            MergeDirection::Horizontal,
            ref_x,
            ref_y,
            sec_x,
            sec_y,
        );

        mosaiced = Some(match mosaiced {
            None => horizontal_part,
            Some(prev) => {
                let vi = i - 2;
                let (vref_x, vref_y) = vertical_marks[vi + 1];
                let (vsec_x, vsec_y) = vertical_marks[vi];
                prev.mosaic(
                    &horizontal_part,
                    MergeDirection::Vertical,
                    vref_x,
                    vref_y,
                    vsec_x,
                    vsec_y,
                )
            }
        });
    }

    let result = mosaiced.unwrap();
    // True geometry: both 540-wide strips, the second offset (2, 190).
    assert_eq!(result.width(), 542);
    assert_eq!(result.height(), 430);
    // Mosaic images are grayscale
    assert_eq!(result.format().channels(), 1);
}

/// The ported `test_globalbalance` body: balance the mosaic built by the
/// mosaic surface and keep its geometry, one band, float format.
#[test]
fn ported_globalbalance_call_site() {
    let left = scene_crop(300, 240, 0, 0);
    let right = scene_crop(300, 234, 240, 4);

    let ref_x = left.width() as i32 - 30;
    let mosaiced = left.mosaic(&right, MergeDirection::Horizontal, ref_x, 120, 30, 116);

    let balanced = mosaiced.global_balance();

    assert_eq!(balanced.width(), 540);
    assert_eq!(balanced.height(), 240);
    assert_eq!(balanced.format().channels(), 1);
    // Global balance produces float output.
    assert!(balanced.format().is_float());
}
