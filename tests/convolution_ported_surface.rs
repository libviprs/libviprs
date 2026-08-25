//! Pins the convolution call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, `tests/ported_convolution.rs`,
//! plus the `Kernel::gaussmat` / `Kernel::logmat` call sites of
//! `tests/ported_create.rs` deferred by `tests/create_ported_surface.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: the `Kernel { data, scale }` struct literal, the
//! `conv` / `convsep` / `compass` / `gaussblur` / `sharpen` / `spcor` /
//! `fastcor` methods, and the `Precision` / `Combine` / `Angle45`
//! arguments, with each ported test body reproduced literally. Behaviour
//! depth is covered by the unit tests in `src/convolution.rs`.
//!
//! The ported cell decodes `sample.jpg` from the fetched reference suite;
//! this file substitutes deterministic synthetic stand-ins (the libvips
//! originals themselves run on synthetic `mask_ideal` images, not on
//! photographs): linear colour planes for the convolution comparisons,
//! where the ported probe positions provably avoid unsigned clipping for
//! all four masks, and seeded noise for the correlation and sharpen
//! bodies, where a patch match must be unique.
//!
//! Two documented adaptations, neither an assertion change:
//!
//! * The ported `pixel_f64` helper decodes 1- and 2-byte channels and
//!   returns an empty vector otherwise, which would skip every
//!   `Precision::Float` comparison (float results are 4-byte channels)
//!   by zipping against nothing. The helper here decodes the f32 case
//!   too, so those assertions actually run; the asserted values and
//!   thresholds are untouched.
//! * The ported `Kernel::gaussmat(1.0, 0.1, true)` / separable
//!   `Kernel::logmat(1.0, 0.1, true)` calls drop the `precision="float"`
//!   argument that the libvips originals (`test_create.py::test_gaussmat`
//!   / `test_logmat`) pass explicitly, while keeping the float
//!   expectation `max == 1.0`. libvips defaults both generators to
//!   integer precision (`create/gaussmat.c` / `create/logmat.c` both init
//!   `precision = VIPS_PRECISION_INTEGER`), which makes the separable
//!   maximum `rint(20 * 1.0) = 20`, so the three-argument ported calls
//!   are a mis-port: the assertions can only hold with the original's
//!   explicit float precision, restored below via the `precision`
//!   argument of `gaussmat` and via `logmat_with_precision`.

use libviprs::conversion::Angle45;
use libviprs::{Combine, Kernel, PixelFormat, Precision, Raster};

/// Read pixel values at (x, y) as f64 slice (the ported helper, with the
/// float arm decoding f32 samples instead of returning an empty vector).
fn pixel_f64(im: &Raster, x: u32, y: u32) -> Vec<f64> {
    let view = im.region(x, y, 1, 1).unwrap();
    let raw = view.pixel(0, 0).unwrap();
    match im.format().bytes_per_channel() {
        1 => raw.iter().map(|&b| b as f64).collect(),
        2 => raw
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]) as f64)
            .collect(),
        _ => raw
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
    }
}

/// Perform a point convolution on `image` at position `(px, py)` with the
/// given `kernel` (2D f64 matrix) and `scale` divisor (the ported
/// reference implementation, verbatim).
fn point_conv(image: &Raster, kernel: &[Vec<f64>], scale: f64, px: u32, py: u32) -> Vec<f64> {
    let kh = kernel.len();
    let kw = kernel[0].len();
    let channels = image.format().channels();
    let mut sums = vec![0.0_f64; channels];

    for (ky, row) in kernel.iter().enumerate().take(kh) {
        for (kx, &m) in row.iter().enumerate().take(kw) {
            let ix = px + kx as u32;
            let iy = py + ky as u32;
            let pix = pixel_f64(image, ix, iy);
            for (s, &p) in sums.iter_mut().zip(pix.iter()) {
                *s += m * p;
            }
        }
    }

    sums.iter().map(|&s| s / scale).collect()
}

/// The colour stand-in for `sample.jpg`: 100x100 Rgb8 with linear planes
/// (each band `a + b*x + c*(99 - y)`, integer coefficients). Every plane
/// is linear and non-increasing in y, so at the ported probe positions
/// the sharp and blur masks reproduce the plane value, the line mask is
/// exactly zero, and the sobel mask is a non-negative constant: no
/// unsigned clipping anywhere, which the unclipped `point_conv` reference
/// requires.
fn sample_colour() -> Raster {
    let mut data = vec![0u8; 100 * 100 * 3];
    for y in 0..100u32 {
        for x in 0..100u32 {
            let o = ((y * 100 + x) * 3) as usize;
            data[o] = (10 + x + (99 - y)) as u8;
            data[o + 1] = (20 + (99 - y)) as u8;
            data[o + 2] = (5 + x) as u8;
        }
    }
    Raster::new(100, 100, PixelFormat::Rgb8, data).unwrap()
}

/// The noise stand-in for the correlation and sharpen bodies: a seeded
/// LCG makes the 10x10 patch at (20, 45) unique.
fn sample_noise() -> Raster {
    let mut state = 0xB5297A4Du32;
    let mut data = vec![0u8; 100 * 100 * 3];
    for b in data.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    Raster::new(100, 100, PixelFormat::Rgb8, data).unwrap()
}

/// The ported `test_conv` body (kernels from the Python setup: sharp,
/// blur, line, sobel).
#[test]
fn ported_conv() {
    let colour = sample_colour();
    // Extract band 1 as mono (green channel)
    let mono = colour.extract_band(1);

    let kernels = vec![
        (
            vec![
                vec![-1.0, -1.0, -1.0],
                vec![-1.0, 16.0, -1.0],
                vec![-1.0, -1.0, -1.0],
            ],
            8.0,
        ),
        (
            vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
            9.0,
        ),
        (
            vec![
                vec![1.0, 1.0, 1.0],
                vec![-2.0, -2.0, -2.0],
                vec![1.0, 1.0, 1.0],
            ],
            1.0,
        ),
        (
            vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            1.0,
        ),
    ];

    for im in [&mono, &colour] {
        for (kernel_data, scale) in &kernels {
            for precision in [Precision::Integer, Precision::Float] {
                let kernel = Kernel {
                    data: kernel_data.clone(),
                    scale: *scale,
                };
                let convolved = im.conv(&kernel, precision);

                let result = pixel_f64(&convolved, 25, 50);
                let expected = point_conv(im, kernel_data, *scale, 24, 49);
                for (r, e) in result.iter().zip(expected.iter()) {
                    assert!(
                        (r - e).abs() < 1.0,
                        "Conv mismatch at (25,50): got {r}, expected {e}"
                    );
                }

                let result = pixel_f64(&convolved, 50, 50);
                let expected = point_conv(im, kernel_data, *scale, 49, 49);
                for (r, e) in result.iter().zip(expected.iter()) {
                    assert!(
                        (r - e).abs() < 1.0,
                        "Conv mismatch at (50,50): got {r}, expected {e}"
                    );
                }
            }
        }
    }
}

/// The ported `test_compass` body.
#[test]
fn ported_compass() {
    let colour = sample_colour();
    let mono = colour.extract_band(1);

    let sharp = Kernel {
        data: vec![
            vec![-1.0, -1.0, -1.0],
            vec![-1.0, 16.0, -1.0],
            vec![-1.0, -1.0, -1.0],
        ],
        scale: 8.0,
    };

    for im in [&mono, &colour] {
        for precision in [Precision::Integer, Precision::Float] {
            for times in 1..4u32 {
                // Test MAX combine
                let convolved = im.compass(&sharp, times, Angle45::D45, Combine::Max, precision);
                assert_eq!(convolved.width(), im.width());
                assert_eq!(convolved.height(), im.height());

                // Test SUM combine
                let convolved = im.compass(&sharp, times, Angle45::D45, Combine::Sum, precision);
                assert_eq!(convolved.width(), im.width());
                assert_eq!(convolved.height(), im.height());
            }
        }
    }
}

/// The ported `test_convsep` body.
#[test]
fn ported_convsep() {
    let colour = sample_colour();
    let mono = colour.extract_band(1);

    for im in [&mono, &colour] {
        for precision in [Precision::Integer, Precision::Float] {
            let gmask = Kernel::gaussmat(2.0, 0.1, false, precision);
            let gmask_sep = Kernel::gaussmat(2.0, 0.1, true, precision);

            // 2D kernel should be square
            assert_eq!(gmask.width(), gmask.height());
            // Separable kernel: same width, height=1
            assert_eq!(gmask_sep.width(), gmask.width());
            assert_eq!(gmask_sep.height(), 1);

            let a = im.conv(&gmask, precision);
            let b = im.convsep(&gmask_sep, precision);

            let a_px = pixel_f64(&a, 25, 50);
            let b_px = pixel_f64(&b, 25, 50);
            for (av, bv) in a_px.iter().zip(b_px.iter()) {
                assert!(
                    (av - bv).abs() < 1.0,
                    "convsep mismatch: conv={av}, convsep={bv}"
                );
            }
        }
    }
}

/// The ported `test_fastcor` body.
#[test]
fn ported_fastcor() {
    let colour = sample_noise();
    let mono = colour.extract_band(1);

    for im in [&mono, &colour] {
        let small = im.extract(20, 45, 10, 10).unwrap();
        let cor = im.fastcor(&small);
        let (v, x, y) = cor.minpos();

        assert_eq!(v, 0.0, "Perfect match should have SSD=0");
        assert_eq!(x, 25, "Match x position");
        assert_eq!(y, 50, "Match y position");
    }
}

/// The ported `test_spcor` body.
#[test]
fn ported_spcor() {
    let colour = sample_noise();
    let mono = colour.extract_band(1);

    for im in [&mono, &colour] {
        let small = im.extract(20, 45, 10, 10).unwrap();
        let cor = im.spcor(&small);
        let (v, x, y) = cor.maxpos();

        assert!(
            (v - 1.0).abs() < 0.001,
            "NCC perfect match should be 1.0, got {v}"
        );
        assert_eq!(x, 25, "Match x position");
        assert_eq!(y, 50, "Match y position");
    }
}

/// The ported `test_gaussblur` body.
#[test]
fn ported_gaussblur() {
    let colour = sample_colour();
    let mono = colour.extract_band(1);

    for im in [&mono, &colour] {
        for precision in [Precision::Integer, Precision::Float] {
            for i in 5..10 {
                let sigma = i as f64 / 5.0;
                let gmask = Kernel::gaussmat(sigma, 0.2, false, precision);

                let a = im.conv(&gmask, precision);
                let b = im.gaussblur(sigma, 0.2, precision);

                let a_px = pixel_f64(&a, 25, 50);
                let b_px = pixel_f64(&b, 25, 50);
                for (av, bv) in a_px.iter().zip(b_px.iter()) {
                    assert!(
                        (av - bv).abs() < 1.0,
                        "gaussblur mismatch at sigma={sigma}: conv={av}, gaussblur={bv}"
                    );
                }
            }
        }
    }
}

/// The ported `test_sharpen` body.
#[test]
fn ported_sharpen() {
    let colour = sample_noise();
    let mono = colour.extract_band(1);

    for im in [&mono, &colour] {
        for &sigma in &[0.5, 1.0, 1.5, 2.0] {
            let sharp = im.sharpen(sigma, 1.0, 2.0);
            assert_eq!(im.width(), sharp.width());
            assert_eq!(im.height(), sharp.height());

            // With m1=0 and m2=0, sharpen should be identity
            let noop = im.sharpen(sigma, 0.0, 0.0);
            let max_diff: u8 = im
                .data()
                .iter()
                .zip(noop.data().iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(max_diff, 0, "sharpen with m1=0, m2=0 should be identity");
        }
    }
}

/// The ported `test_gaussmat` body (from `ported_create.rs`; the
/// separable call restores the original's explicit float precision, see
/// the module docs for the mis-port proof).
#[test]
fn ported_gaussmat() {
    let k = Kernel::gaussmat(1.0, 0.1, false, Precision::Integer);
    assert_eq!(k.width(), 5);
    assert_eq!(k.height(), 5);
    assert!((k.max() - 20.0).abs() < 0.001);
    let center = k.data[2][2];
    assert!((center - 20.0).abs() < 0.001);
    // total == scale: the sum of all mask elements is the stored scale.
    let total: f64 = k.data.iter().flatten().sum();
    assert!((total - k.scale).abs() < 0.001);

    let ks = Kernel::gaussmat(1.0, 0.1, true, Precision::Float);
    assert_eq!(ks.width(), 5);
    assert_eq!(ks.height(), 1);
    assert!((ks.max() - 1.0).abs() < 0.001);
    let center = ks.data[0][2];
    assert!((center - 1.0).abs() < 0.001);
}

/// The ported `test_logmat` body (from `ported_create.rs`; the separable
/// call restores the original's explicit float precision through
/// `logmat_with_precision`, see the module docs for the mis-port proof).
#[test]
fn ported_logmat() {
    let k = Kernel::logmat(1.0, 0.1, false);
    assert_eq!(k.width(), 7);
    assert_eq!(k.height(), 7);
    assert!((k.max() - 20.0).abs() < 0.001);
    assert!((k.data[3][3] - 20.0).abs() < 0.001);

    let ks = Kernel::logmat_with_precision(1.0, 0.1, true, Precision::Float);
    assert_eq!(ks.width(), 7);
    assert_eq!(ks.height(), 1);
    assert!((ks.max() - 1.0).abs() < 0.001);
}

/// The edge-detector surface, which the ported cell does not reach: the
/// three no-argument detectors and their `try_*` twins, called from an
/// external crate. `vips sobel` / `scharr` / `prewitt` take no options at
/// all, so the whole call surface is the receiver.
///
/// The expected values are the vips 8.18.4 measurements on a 7x7 uchar
/// image with a background of 10 stepping to 20: a pure vertical step is
/// a pure Gx, so the answer is `|Gx|` on the two columns straddling the
/// step. Behaviour depth lives in the unit tests in `src/convolution.rs`.
#[test]
fn edge_detector_surface() {
    let mut data = Vec::with_capacity(7 * 7);
    for _ in 0..7 {
        for x in 0..7u32 {
            data.push(if x >= 4 { 20u8 } else { 10 });
        }
    }
    let im = Raster::new(7, 7, PixelFormat::Gray8, data).unwrap();

    let cases: [(&str, Raster, Raster, f64); 3] = [
        ("sobel", im.sobel(), im.try_sobel().unwrap(), 40.0),
        ("scharr", im.scharr(), im.try_scharr().unwrap(), 160.0),
        ("prewitt", im.prewitt(), im.try_prewitt().unwrap(), 30.0),
    ];
    for (name, out, fallible, step) in cases {
        assert_eq!(out.format(), PixelFormat::Gray8, "{name} is always uchar");
        assert_eq!(out.width(), 7, "{name} width");
        assert_eq!(out.height(), 7, "{name} height");
        assert_eq!(out.data(), fallible.data(), "{name} try_ form agrees");
        assert_eq!(pixel_f64(&out, 3, 3), vec![step], "{name} on the step");
        assert_eq!(pixel_f64(&out, 0, 3), vec![0.0], "{name} off the step");
    }
}

/// Every input format answers with an 8-bit raster of the same band
/// count: `edge.c` dispatches on the format but both arms end in uchar.
#[test]
fn edge_detector_output_is_always_uchar() {
    for (src, want) in [
        (PixelFormat::Gray16, PixelFormat::Gray8),
        (PixelFormat::Rgb16, PixelFormat::Rgb8),
        (PixelFormat::RgbaF32, PixelFormat::Rgba8),
    ] {
        let im = Raster::zeroed(5, 4, src).unwrap();
        assert_eq!(im.sobel().format(), want, "sobel of {src:?}");
        assert_eq!(im.scharr().format(), want, "scharr of {src:?}");
        assert_eq!(im.prewitt().format(), want, "prewitt of {src:?}");
    }
}
