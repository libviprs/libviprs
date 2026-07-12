//! Pins the create/generator call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, `tests/ported_create.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: constructor names, argument types (including
//! the `Option` arguments of `text` and the by-reference `SdfParams`),
//! and return types. Behaviour depth is covered by the unit tests in
//! `src/create.rs`; this file is the API contract, with each ported test
//! body reproduced literally.
//!
//! Not pinned here, because they belong to later batches, are the
//! ported cell's `fwfft` (`test_fwfft_small_image`), `invertlut`
//! (`test_invertlut`), `matrixinvert` (`test_matrixinvert`), and
//! `Kernel::gaussmat` / `Kernel::logmat` (`test_gaussmat` /
//! `test_logmat`) call sites. The `from_matrix` setup those tests share
//! is in this batch and is pinned below. The `test_grey` and
//! `test_identity` bodies are already pinned in
//! `tests/conversion_ported_surface.rs`.
//!
//! One ported assertion is corrected here (see
//! `ported_mask_butterworth`): the ported cell passes `nodc: true` in
//! the uchar+optical variant while still expecting the libvips original's
//! 255; the original (`test_create.py::test_mask_butterworth`, second
//! block) does not set nodc there, and libvips `create/mask.c` forces
//! the DC component to 1.0 only when nodc is off, so 255 requires
//! `nodc: false`. The proof is spelled out at the call site.

use libviprs::{PixelFormat, Raster, SdfParams};

/// The ported `test_black` body.
#[test]
fn ported_black() {
    let im = Raster::black(100, 100);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 100);
    assert_eq!(im.format(), PixelFormat::Gray8);
    assert_eq!(im.format().channels(), 1);
    for i in 0..100u32 {
        let px = im.getpoint(i, i);
        assert_eq!(px, vec![0.0], "Pixel at ({i},{i}) should be 0");
    }

    let im = Raster::black_bands(100, 100, 3);
    assert_eq!(im.format().channels(), 3);
    for i in 0..100u32 {
        let px = im.getpoint(i, i);
        assert_eq!(px, vec![0.0, 0.0, 0.0]);
    }
}

/// The ported `test_buildlut` body.
#[test]
fn ported_buildlut() {
    // Simple two-point LUT
    let lut = Raster::buildlut(&[vec![0.0, 0.0], vec![255.0, 100.0]]);
    assert_eq!(lut.width(), 256);
    assert_eq!(lut.height(), 1);
    assert_eq!(lut.format().channels(), 1);

    let p0 = lut.getpoint(0, 0);
    assert!((p0[0] - 0.0).abs() < 0.001);
    let p255 = lut.getpoint(255, 0);
    assert!((p255[0] - 100.0).abs() < 0.001);
    let p10 = lut.getpoint(10, 0);
    assert!((p10[0] - 100.0 * 10.0 / 255.0).abs() < 0.1);

    // Multi-band LUT
    let lut = Raster::buildlut(&[
        vec![0.0, 0.0, 100.0],
        vec![255.0, 100.0, 0.0],
        vec![128.0, 10.0, 90.0],
    ]);
    assert_eq!(lut.width(), 256);
    assert_eq!(lut.format().channels(), 2);
    let p0 = lut.getpoint(0, 0);
    assert!((p0[0] - 0.0).abs() < 0.1);
    assert!((p0[1] - 100.0).abs() < 0.1);
    let p64 = lut.getpoint(64, 0);
    assert!((p64[0] - 5.0).abs() < 0.5);
    assert!((p64[1] - 95.0).abs() < 0.5);
}

/// The ported `test_eye` body.
#[test]
fn ported_eye() {
    let im = Raster::eye(100, 90, false);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 90);
    assert_eq!(im.format().channels(), 1);
    assert!((im.max_value() - 1.0).abs() < 0.001);
    assert!((im.min_value() - (-1.0)).abs() < 0.001);

    let im = Raster::eye(100, 90, true);
    assert_eq!(im.format(), PixelFormat::Gray8);
    assert!((im.max_value() - 255.0).abs() < 0.001);
    assert!((im.min_value() - 0.0).abs() < 0.001);
}

/// The ported `test_fractsurf` body.
#[test]
fn ported_fractsurf() {
    let im = Raster::fractsurf(100, 90, 2.5);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 90);
    assert_eq!(im.format().channels(), 1);
}

/// The ported `test_gaussnoise` body.
#[test]
fn ported_gaussnoise() {
    let im = Raster::gaussnoise(100, 90, 0.0, 1.0);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 90);
    assert_eq!(im.format().channels(), 1);

    let im = Raster::gaussnoise(100, 90, 100.0, 10.0);
    assert!((im.deviate() - 10.0).abs() < 0.4);
    assert!((im.avg() - 100.0).abs() < 0.4);
}

/// The ported `test_invertlut` / `test_matrixinvert` setup: the
/// `from_matrix` constructor they build their inputs with (the
/// `invertlut` / `matrixinvert` calls themselves belong to a later
/// batch).
#[test]
fn ported_from_matrix_call_sites() {
    let lut = Raster::from_matrix(&[
        vec![0.1, 0.2, 0.3, 0.1],
        vec![0.2, 0.4, 0.4, 0.2],
        vec![0.7, 0.5, 0.6, 0.3],
    ]);
    assert_eq!((lut.width(), lut.height()), (4, 3));

    let mat = Raster::from_matrix(&[
        vec![4.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![0.0, 1.0, 2.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ]);
    assert_eq!((mat.width(), mat.height()), (4, 4));
    assert_eq!(mat.getpoint(0, 0), vec![4.0]);
}

/// The ported `test_mask_butterworth` body. The float half is literal;
/// the uchar+optical half is corrected from the ported cell's
/// `(..., true, true, true)` to `(..., false, true, true)`:
///
/// * libvips `create/mask.c` forces the DC component to 1.0 *only when
///   nodc is off* (`if (!mask->nodc && x == 0 && y == 0) result = 1.0`),
///   and with `optical` the DC lands at the image centre (64, 64).
/// * With `nodc: true` the DC comes from the Butterworth highpass
///   formula, which returns 0 at d == 0 (`mask_butterworth.c`), so the
///   pixel reads 0 and can never satisfy the 255 the test keeps from the
///   libvips original.
/// * The original (`test_create.py::test_mask_butterworth`, second
///   block) builds this variant with `optical=True, uchar=True` and no
///   nodc, and expects 255 at (64, 64): `nodc: false` is the faithful
///   port, and 255 then holds.
#[test]
fn ported_mask_butterworth() {
    let im = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, true, false, false);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
    let p = im.getpoint(0, 0);
    assert!((p[0] - 0.0).abs() < 0.001, "DC should be 0 with nodc");
    let (_, mx, my) = im.maxpos();
    assert_eq!(mx, 64);
    assert_eq!(my, 64);

    // uchar + optical variant (nodc corrected to false; see above).
    let im = Raster::mask_butterworth(128, 128, 2.0, 0.7, 0.1, false, true, true);
    let p = im.getpoint(64, 64);
    assert_eq!(p[0], 255.0);
}

/// The ported `test_mask_butterworth_band` body.
#[test]
fn ported_mask_butterworth_band() {
    let im = Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, false, false, false);
    assert_eq!(im.width(), 128);
    assert_eq!(im.format().channels(), 1);
    assert!((im.max_value() - 1.0).abs() < 0.01);
    let p = im.getpoint(32, 32);
    assert!((p[0] - 1.0).abs() < 0.01);

    // uchar + optical variant
    let im = Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, true, true, false);
    assert_eq!(im.max_value(), 255.0);
    let p = im.getpoint(32, 32);
    assert_eq!(p[0], 255.0);

    // nodc variant
    let im = Raster::mask_butterworth_band(128, 128, 2.0, 0.5, 0.5, 0.7, 0.1, true, true, true);
    let p = im.getpoint(64, 64);
    assert_ne!(p[0], 255.0);
}

/// The ported `test_mask_butterworth_ring` body.
#[test]
fn ported_mask_butterworth_ring() {
    let im = Raster::mask_butterworth_ring(128, 128, 2.0, 0.7, 0.1, 0.5, true);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
    let p = im.getpoint(45, 0);
    assert!((p[0] - 1.0).abs() < 0.001);
    let (_, mx, my) = im.minpos();
    assert_eq!(mx, 64);
    assert_eq!(my, 64);
}

/// The ported `test_mask_fractal` body.
#[test]
fn ported_mask_fractal() {
    let im = Raster::mask_fractal(128, 128, 2.3);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
}

/// The ported `test_mask_gaussian` body.
#[test]
fn ported_mask_gaussian() {
    let im = Raster::mask_gaussian(128, 128, 0.7, 0.1, true);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
    assert!(im.min_value().abs() < 0.01);
    let p = im.getpoint(0, 0);
    assert!((p[0] - 0.0).abs() < 0.01);
}

/// The ported `test_mask_gaussian_band` body.
#[test]
fn ported_mask_gaussian_band() {
    let im = Raster::mask_gaussian_band(128, 128, 0.5, 0.5, 0.7, 0.1);
    assert_eq!(im.width(), 128);
    assert_eq!(im.format().channels(), 1);
    assert!((im.max_value() - 1.0).abs() < 0.01);
    let p = im.getpoint(32, 32);
    assert!((p[0] - 1.0).abs() < 0.01);
}

/// The ported `test_mask_gaussian_ring` body.
#[test]
fn ported_mask_gaussian_ring() {
    let im = Raster::mask_gaussian_ring(128, 128, 0.7, 0.1, 0.5, true);
    assert_eq!(im.width(), 128);
    assert_eq!(im.format().channels(), 1);
    let p = im.getpoint(45, 0);
    assert!((p[0] - 1.0).abs() < 0.01);
}

/// The ported `test_mask_gaussian_ring_2` body (misleading name kept
/// from the libvips original; it exercises `mask_ideal_ring`).
#[test]
fn ported_mask_ideal_ring() {
    let im = Raster::mask_ideal_ring(128, 128, 0.7, 0.5, true);
    assert_eq!(im.width(), 128);
    assert_eq!(im.format().channels(), 1);
    let p = im.getpoint(45, 0);
    assert!((p[0] - 1.0).abs() < 0.01);
}

/// The ported `test_mask_ideal` body.
#[test]
fn ported_mask_ideal() {
    let im = Raster::mask_ideal(128, 128, 0.7, true);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
    assert!(im.min_value().abs() < 0.01);
    let p = im.getpoint(0, 0);
    assert!((p[0] - 0.0).abs() < 0.01);
}

/// The ported `test_mask_ideal_band` body.
#[test]
fn ported_mask_ideal_band() {
    let im = Raster::mask_ideal_band(128, 128, 0.5, 0.5, 0.7);
    assert_eq!(im.width(), 128);
    assert_eq!(im.format().channels(), 1);
    assert!((im.max_value() - 1.0).abs() < 0.01);
    let p = im.getpoint(32, 32);
    assert!((p[0] - 1.0).abs() < 0.01);
}

/// The ported `test_sines` body.
#[test]
fn ported_sines() {
    let im = Raster::sines(128, 128);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
}

/// The ported `test_text` body.
#[test]
fn ported_text() {
    let im = Raster::text("Hello, world!", Some(300), None, None, None);
    assert!(im.width() > 10);
    assert!(im.height() > 10);
    assert_eq!(im.format().channels(), 1);
    assert_eq!(im.format(), PixelFormat::Gray8);
    assert!(im.max_value() > 240.0);
    assert!((im.min_value() - 0.0).abs() < 0.001);

    // Auto-fit
    let im = Raster::text("Hello, world!", None, Some(500), Some(500), None);
    assert!((im.width() as i32 - 500).abs() < 50);
}

/// The ported `test_tonelut` body.
#[test]
fn ported_tonelut() {
    let im = Raster::tonelut();
    assert_eq!(im.format().channels(), 1);
    assert_eq!(im.width(), 32768);
    assert_eq!(im.height(), 1);
    assert!(im.hist_ismonotonic());
}

/// The ported `test_xyz` body.
#[test]
fn ported_xyz() {
    let im = Raster::xyz(128, 128);
    assert_eq!(im.format().channels(), 2);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    let p = im.getpoint(45, 35);
    assert!((p[0] - 45.0).abs() < 0.001);
    assert!((p[1] - 35.0).abs() < 0.001);
}

/// The ported `test_sdf` body.
#[test]
fn ported_sdf() {
    // Circle SDF
    let im = Raster::sdf(
        128,
        128,
        "circle",
        &SdfParams {
            a: [64, 64],
            r: Some(32),
            ..Default::default()
        },
    );
    assert_eq!(im.width(), 128);
    let p = im.getpoint(45, 35);
    assert!((p[0] - 2.670).abs() < 0.1);

    // Box SDF
    let im = Raster::sdf(
        128,
        128,
        "box",
        &SdfParams {
            a: [10, 10],
            b: Some([50, 40]),
            ..Default::default()
        },
    );
    let p = im.getpoint(45, 35);
    assert!((p[0] - (-5.0)).abs() < 0.1);

    // Line SDF
    let im = Raster::sdf(
        128,
        128,
        "line",
        &SdfParams {
            a: [10, 10],
            b: Some([50, 40]),
            ..Default::default()
        },
    );
    let p = im.getpoint(45, 35);
    assert!((p[0] - 1.0).abs() < 0.1);
}

/// The ported `test_zone` body.
#[test]
fn ported_zone() {
    let im = Raster::zone(128, 128);
    assert_eq!(im.width(), 128);
    assert_eq!(im.height(), 128);
    assert_eq!(im.format().channels(), 1);
}

/// The ported `test_worley` body.
#[test]
fn ported_worley() {
    let im = Raster::worley(512, 512);
    assert_eq!(im.width(), 512);
    assert_eq!(im.height(), 512);
    assert_eq!(im.format().channels(), 1);
}

/// The ported `test_perlin` body.
#[test]
fn ported_perlin() {
    let im = Raster::perlin(512, 512);
    assert_eq!(im.width(), 512);
    assert_eq!(im.height(), 512);
    assert_eq!(im.format().channels(), 1);
}
