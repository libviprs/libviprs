//! Pins the arithmetic / statistics call surface required by the
//! libviprs-tests ported suite (libviprs-tests issue #55,
//! `tests/ported_arithmetic.rs`, plus the arithmetic call sites in
//! `ported_conversion.rs`, `ported_histogram.rs`, and `ported_iofuncs.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types, `Option` parameters, tuple
//! and `f64` return types. Behavior depth is covered by the unit tests in
//! `src/arithmetic.rs`; this file is the API contract.
//!
//! Where a ported test's setup uses an operation from a later batch
//! (`insert`, `grey`, integer-ink draw calls, `decode_file` fixtures), the
//! setup is reproduced with direct `Raster` construction and the arithmetic
//! expressions are kept literal.

use libviprs::{PixelFormat, Raster};

/// The ported `make_test_mono`: a 100x100 Gray8 band-reject ring image.
fn make_test_mono() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let mut data = vec![0u8; (w * h) as usize];
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 - cx) / cx;
            let dy = (y as f64 - cy) / cy;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r > 0.5 {
                (r * 200.0).min(255.0) as u8
            } else {
                0
            };
            data[(y * w + x) as usize] = v;
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

/// The ported `make_test_colour`: `mono * [1, 2, 3] + [2, 3, 4]` as Rgb8.
fn make_test_colour() -> Raster {
    let mono = make_test_mono();
    let w = mono.width();
    let h = mono.height();
    let md = mono.data();
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        let v = md[i] as u16;
        data[i * 3] = (v + 2).min(255) as u8;
        data[i * 3 + 1] = (v * 2 + 3).min(255) as u8;
        data[i * 3 + 2] = (v * 3 + 4).min(255) as u8;
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// A 100x100 Gray8 image whose left half is 0 and right half is `v`
/// (the ported tests build this with `insert`, a conversion-batch op).
fn half_half_100(v: u8) -> Raster {
    let mut data = vec![0u8; 100 * 100];
    for y in 0..100 {
        for x in 50..100 {
            data[y * 100 + x] = v;
        }
    }
    Raster::new(100, 100, PixelFormat::Gray8, data).unwrap()
}

/// The ported `test_add` call sites: image + image, + scalar, + vector.
#[test]
fn ported_surface_add() {
    let colour = make_test_colour();
    let mono = make_test_mono();

    let result = colour.add(&colour);
    let px_a = colour.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    for (a, r) in px_a.iter().zip(px_r.iter()) {
        assert!((r - (a + a)).abs() < 1.0);
    }

    let result = mono.add_const(42.0);
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    assert!((px_r[0] - (px_m[0] + 42.0)).abs() < 1.0);

    let result = colour.add_vec(&[1.0, 2.0, 3.0]);
    let px_c = colour.getpoint(10, 10);
    let px_r = result.getpoint(10, 10);
    for (i, (c, r)) in px_c.iter().zip(px_r.iter()).enumerate() {
        assert!((r - (c + (i as f64 + 1.0))).abs() < 1.0);
    }
}

/// The ported `test_sub` call sites: image - image is zero, - scalar.
#[test]
fn ported_surface_sub() {
    let colour = make_test_colour();

    let result = colour.sub(&colour);
    assert!(result.avg().abs() < 0.001);

    let result = colour.sub_const(1.0);
    let px_c = colour.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - (c - 1.0)).abs() < 1.0);
    }
}

/// The ported `test_mul` / `test_div` / `test_floordiv` / `test_pow` /
/// `test_mod` constant call sites.
#[test]
fn ported_surface_mul_div_pow_mod() {
    let colour = make_test_colour();
    let mono = make_test_mono();

    let result = colour.mul_const(2.0);
    let px_c = colour.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - c * 2.0).abs() < 1.0);
    }

    let result = colour.div_const(2.0);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - c / 2.0).abs() < 1.0);
    }

    let result = colour.floordiv_const(3.0);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - (c / 3.0).floor()).abs() < 1.0);
    }

    let result = mono.pow_const(2.0);
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    assert!((px_r[0] - px_m[0].powf(2.0)).abs() < 1.0);

    let result = mono.rem_const(2.0);
    let px_r = result.getpoint(50, 50);
    assert!((px_r[0] - (px_m[0] as i64 % 2) as f64).abs() < 1.0);
}

/// The ported `test_pos` / `test_abs` / `test_sign` / `test_clamp` and
/// `test_floor` / `test_ceil` / `test_rint` call sites. `abs` is exercised
/// directly (the ported test derives its input from `neg`, which needs a
/// signed depth and is deferred).
#[test]
fn ported_surface_unary_shape() {
    let mono = make_test_mono();
    let colour = make_test_colour();

    let result = mono.pos();
    assert!((result.getpoint(50, 50)[0] - mono.getpoint(50, 50)[0]).abs() < 0.001);

    let result = colour.abs();
    let px_c = colour.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - c).abs() < 1.0);
    }

    let result = mono.sign();
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    let expected = if px_m[0] > 0.0 { 1.0 } else { 0.0 };
    assert!((px_r[0] - expected).abs() < 0.001);

    let result = colour.clamp(None, None);
    assert!(result.max() <= 1.0);
    assert!(result.min() >= 0.0);
    let result = colour.clamp(Some(14.0), Some(45.0));
    assert!(result.max() <= 45.0);
    assert!(result.min() >= 14.0);

    for result in [mono.floor(), mono.ceil(), mono.rint()] {
        assert!((result.getpoint(50, 50)[0] - px_m[0]).abs() < 0.001);
    }
}

/// The ported bitwise call sites: `test_and`, `test_or`, `test_xor`,
/// `test_invert`, `test_lshift`, `test_rshift`.
#[test]
fn ported_surface_bitwise() {
    let mono = make_test_mono();

    let result = mono.bitand(&mono);
    let px_m = mono.getpoint(50, 50);
    assert!((result.getpoint(50, 50)[0] - px_m[0]).abs() < 0.001);
    let result = mono.bitand_const(0);
    assert!(result.avg().abs() < 0.001);

    let result = mono.bitor(&mono);
    assert!((result.getpoint(50, 50)[0] - px_m[0]).abs() < 0.001);
    let result = mono.bitor_const(0xFF);
    assert!((result.avg() - 255.0).abs() < 0.001);

    let result = mono.bitxor(&mono);
    assert!(result.avg().abs() < 0.001);
    let result = mono.bitxor_const(0);
    assert!((result.getpoint(50, 50)[0] - px_m[0]).abs() < 0.001);

    let result = mono.bitnot();
    let expected = (!(px_m[0] as u8)) as f64;
    assert!((result.getpoint(50, 50)[0] - expected).abs() < 1.0);

    let result = mono.lshift(2);
    assert!((result.getpoint(50, 50)[0] - ((px_m[0] as i64) << 2) as f64).abs() < 1.0);
    let result = mono.rshift(2);
    assert!((result.getpoint(50, 50)[0] - ((px_m[0] as i64) >> 2) as f64).abs() < 1.0);
}

/// The ported comparison call sites: `test_more`, `test_moreeq`,
/// `test_less`, `test_lesseq`, `test_equal`, `test_noteq`. The grey-ramp
/// input for `equal_const` is built directly (`Raster::grey` belongs to
/// the creation batch).
#[test]
fn ported_surface_comparisons() {
    let mono = make_test_mono();

    let result = mono.more_than(&mono);
    assert!(result.avg().abs() < 0.001);
    let result = mono.more_than_const(100.0);
    let px = result.getpoint(50, 50);
    let m = mono.getpoint(50, 50);
    let expected = if m[0] > 100.0 { 255.0 } else { 0.0 };
    assert!((px[0] - expected).abs() < 1.0);

    let result = mono.more_eq(&mono);
    assert!((result.avg() - 255.0).abs() < 0.001);
    let result = mono.more_eq_const(0.0);
    assert!((result.avg() - 255.0).abs() < 0.001);

    let result = mono.less_than(&mono);
    assert!(result.avg().abs() < 0.001);
    let result = mono.less_than_const(0.0);
    assert!(result.avg().abs() < 0.001);

    let result = mono.less_eq(&mono);
    assert!((result.avg() - 255.0).abs() < 0.001);
    let result = mono.less_eq_const(255.0);
    assert!((result.avg() - 255.0).abs() < 0.001);

    let result = mono.equal(&mono);
    assert!((result.avg() - 255.0).abs() < 0.001);
    let result = mono.noteq(&mono);
    assert!(result.avg().abs() < 0.001);
    let result = mono.noteq_const(-1.0);
    assert!((result.avg() - 255.0).abs() < 0.001);

    // Grey ramp: 256x256, value = x.
    let mut data = vec![0u8; 256 * 256];
    for y in 0..256 {
        for x in 0..256 {
            data[y * 256 + x] = x as u8;
        }
    }
    let x = Raster::new(256, 256, PixelFormat::Gray8, data).unwrap();

    let cmp = x.equal_const(1000.0);
    assert!(cmp.max() < 1.0, "No uchar pixel can equal 1000");
    let cmp = x.equal_const(12.0);
    assert!((cmp.max() - 255.0).abs() < 1.0, "x==12 should find matches");
    let cmp = x.equal_const(12.5);
    assert!(cmp.max() < 1.0, "No integer pixel can equal 12.5");
}

/// The ported `test_avg` / `test_deviate` call sites (setup built without
/// `insert`).
#[test]
fn ported_surface_avg_deviate() {
    let combined = half_half_100(100);

    assert!(
        (combined.avg() - 50.0).abs() < 1.0,
        "Average of half-black, half-100 image should be ~50, got {}",
        combined.avg()
    );
    assert!(
        (combined.deviate() - 50.0).abs() < 1.0,
        "Deviate should be ~50, got {}",
        combined.deviate()
    );
}

/// The ported `test_max` / `test_min` call sites (single pixel written
/// with the existing draw surface instead of an integer-ink helper).
#[test]
fn ported_surface_min_max_pos() {
    let mut im = Raster::zeroed(100, 100, PixelFormat::Gray8).unwrap();
    im.put_pixel(40, 50, &[100]);

    assert!((im.max() - 100.0).abs() < 1.0);
    let (v, x, y) = im.maxpos();
    assert!((v - 100.0).abs() < 1.0);
    assert_eq!(x, 40);
    assert_eq!(y, 50);

    let data = vec![100u8; 100 * 100];
    let mut im = Raster::new(100, 100, PixelFormat::Gray8, data).unwrap();
    im.put_pixel(40, 50, &[0]);

    assert!(im.min().abs() < 1.0);
    let (v, x, y) = im.minpos();
    assert!(v.abs() < 1.0);
    assert_eq!(x, 40);
    assert_eq!(y, 50);
}

/// The ported `test_stats` call site: [min, max, sum, sum2, mean, sd]
/// rows, row 0 overall.
#[test]
fn ported_surface_stats() {
    let mut data = vec![0u8; 100 * 50];
    for y in 0..50 {
        for x in 50..100 {
            data[y * 100 + x] = 10;
        }
    }
    let im = Raster::new(100, 50, PixelFormat::Gray8, data).unwrap();

    let stats = im.stats();
    assert!((stats[0][0] - 0.0).abs() < 0.001, "min should be 0");
    assert!((stats[0][1] - 10.0).abs() < 0.001, "max should be 10");
    assert!((stats[0][2] - 25000.0).abs() < 1.0, "sum should be 25000");
    assert!((stats[0][4] - im.avg()).abs() < 0.01);
    assert!((stats[0][5] - im.deviate()).abs() < 0.01);
}

/// The ported `test_measure` call site: measure(2, 1) over a left-0
/// right-10 image gives [[0], [10]].
#[test]
fn ported_surface_measure() {
    let mut data = vec![0u8; 100 * 50];
    for y in 0..50 {
        for x in 50..100 {
            data[y * 100 + x] = 10;
        }
    }
    let im = Raster::new(100, 50, PixelFormat::Gray8, data).unwrap();

    let matrix = im.measure(2, 1);
    assert!((matrix[0][0] - 0.0).abs() < 1.0);
    assert!((matrix[1][0] - 10.0).abs() < 1.0);
}

/// The ported `test_find_trim` call site: a 50x60 patch of 100 at (10,20)
/// on a white 200x300 canvas.
#[test]
fn ported_surface_find_trim() {
    let mut data = vec![255u8; 200 * 300];
    for y in 20..80 {
        for x in 10..60 {
            data[y * 200 + x] = 100;
        }
    }
    let im = Raster::new(200, 300, PixelFormat::Gray8, data).unwrap();

    let (left, top, width, height) = im.find_trim(None);
    assert_eq!(left, 10);
    assert_eq!(top, 20);
    assert_eq!(width, 50);
    assert_eq!(height, 60);
}

/// The ported `test_profile` call site: single bright pixel at (40,50).
#[test]
fn ported_surface_profile() {
    let mut im = Raster::zeroed(100, 100, PixelFormat::Gray8).unwrap();
    im.put_pixel(40, 50, &[100]);

    let (columns, rows) = im.profile();

    let (v, x, y) = columns.minpos();
    assert!((v - 50.0).abs() < 1.0);
    assert_eq!(x, 40);
    assert_eq!(y, 0);

    let (v, x, y) = rows.minpos();
    assert!((v - 40.0).abs() < 1.0);
    assert_eq!(x, 0);
    assert_eq!(y, 50);
}

/// The ported `test_project` call site: column and row sums of a left-0
/// right-10 image.
#[test]
fn ported_surface_project() {
    let mut data = vec![0u8; 100 * 50];
    for y in 0..50 {
        for x in 50..100 {
            data[y * 100 + x] = 10;
        }
    }
    let im = Raster::new(100, 50, PixelFormat::Gray8, data).unwrap();

    let (columns, rows) = im.project();
    let col_10 = columns.getpoint(10, 0);
    assert!((col_10[0] - 0.0).abs() < 1.0);
    let col_70 = columns.getpoint(70, 0);
    assert!((col_70[0] - 500.0).abs() < 1.0);
    let row_10 = rows.getpoint(0, 10);
    assert!((row_10[0] - 500.0).abs() < 1.0);
}

/// The ported `test_sum` call site: `Raster::sum(&refs)` over ten constant
/// images.
#[test]
fn ported_surface_sum() {
    let images: Vec<Raster> = (0..10)
        .map(|x| {
            let data = vec![(x * 10) as u8; 50 * 50];
            Raster::new(50, 50, PixelFormat::Gray8, data).unwrap()
        })
        .collect();
    let refs: Vec<&Raster> = images.iter().collect();
    let result = Raster::sum(&refs);
    let expected_max: f64 = (0..10).map(|x| (x * 10) as f64).sum();
    assert!(
        (result.max() - expected_max).abs() < 1.0,
        "Sum max should be {expected_max}, got {}",
        result.max()
    );
}

/// The ported `test_minpair` / `test_maxpair` call sites.
#[test]
fn ported_surface_minpair_maxpair() {
    let a_data = vec![100u8; 50 * 50];
    let a = Raster::new(50, 50, PixelFormat::Gray8, a_data).unwrap();
    let b_data = vec![50u8; 50 * 50];
    let b = Raster::new(50, 50, PixelFormat::Gray8, b_data).unwrap();

    let result = a.minpair(&b);
    assert!(
        (result.avg() - 50.0).abs() < 1.0,
        "min(100,50) should be 50"
    );
    let result = a.maxpair(&b);
    assert!(
        (result.avg() - 100.0).abs() < 1.0,
        "max(100,50) should be 100"
    );
}

/// The ported `test_new_from_memory` linear call site
/// (`ported_iofuncs.rs`): add 10 to a zeroed image via linear(1, 10).
#[test]
fn ported_surface_linear() {
    let im = Raster::zeroed(20, 10, PixelFormat::Gray8).unwrap();
    assert!((im.avg() - 0.0).abs() < 0.001);

    let im2 = im.linear(1.0, 10.0);
    assert!((im2.avg() - 10.0).abs() < 0.001);
}

/// The ported `test_scaleimage` call site (`ported_conversion.rs`).
#[test]
fn ported_surface_scaleimage() {
    let colour = make_test_colour();
    let result = colour.scaleimage(None);
    assert!((result.max() - 255.0).abs() < 1.0);
    assert!(result.min().abs() < 1.0);

    let result_log = colour.scaleimage(Some(true));
    assert!((result_log.max() - 255.0).abs() < 1.0);
}

/// The ported `test_recomb` call site (`ported_conversion.rs`).
#[test]
fn ported_surface_recomb() {
    let colour = make_test_colour();
    let matrix: &[&[f64]] = &[&[0.2, 0.5, 0.3]];
    let result = colour.recomb(matrix);

    let px = colour.getpoint(50, 50);
    let rpx = result.getpoint(50, 50);
    let expected = 0.2 * px[0] + 0.5 * px[1] + 0.3 * px[2];
    assert!((rpx[0] - expected).abs() < 1.0);
}

/// The ported `test_premultiply` / `test_unpremultiply` call sites
/// (`ported_conversion.rs`), literal including the batch-1
/// `bandjoin_const` setup.
#[test]
fn ported_surface_premultiply() {
    let colour = make_test_colour();
    let alpha = 127.0;
    let rgba = colour.bandjoin_const(alpha);

    let pre = rgba.premultiply();
    assert_eq!(pre.format().channels(), 4);
    let px_src = rgba.getpoint(30, 30);
    let px_pre = pre.getpoint(30, 30);
    for i in 0..3 {
        let expected = px_src[i] * alpha / 255.0;
        assert!((px_pre[i] - expected).abs() < 2.0);
    }
    assert!((px_pre[3] - alpha).abs() < 1.0);

    let unpre = pre.unpremultiply();
    assert_eq!(unpre.format().channels(), 4);
    let px_unpre = unpre.getpoint(30, 30);
    for i in 0..3 {
        assert!(
            (px_unpre[i] - px_src[i]).abs() < 2.0,
            "unpremultiply channel {i}: expected {}, got {}",
            px_src[i],
            px_unpre[i]
        );
    }
    assert!((px_unpre[3] - alpha).abs() < 1.0);
}

/// The ported `test_stdif` call site (`ported_histogram.rs`): stdif(10, 10)
/// keeps dimensions and moves the mean toward 128 (synthetic input instead
/// of the sample.jpg fixture).
#[test]
fn ported_surface_stdif() {
    let mut data = vec![0u8; 120 * 80];
    for (i, d) in data.iter_mut().enumerate() {
        *d = (i % 61) as u8;
    }
    let im = Raster::new(120, 80, PixelFormat::Gray8, data).unwrap();
    let im2 = im.stdif(10, 10);

    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());

    let orig_dist = (im.avg() - 128.0).abs();
    let new_dist = (im2.avg() - 128.0).abs();
    assert!(
        new_dist < orig_dist,
        "stdif should shift mean closer to 128: orig_dist={orig_dist}, new_dist={new_dist}"
    );
}

/// The ported `test_sin` / `test_cos` / `test_tan` bodies
/// (`ported_arithmetic.rs` math_functions): degree input, float output,
/// expected values computed from the same `getpoint` reads the ported
/// cell uses.
#[test]
fn ported_surface_math_trig() {
    let mono = make_test_mono();

    let result = mono.sin();
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    let expected = (px_m[0].to_radians()).sin();
    assert!(
        (px_r[0] - expected).abs() < 0.001,
        "sin({}) should be {expected}, got {}",
        px_m[0],
        px_r[0]
    );

    let result = mono.cos();
    let px_r = result.getpoint(50, 50);
    let expected = (px_m[0].to_radians()).cos();
    assert!((px_r[0] - expected).abs() < 0.001);

    let result = mono.tan();
    let px_m = mono.getpoint(10, 10);
    let px_r = result.getpoint(10, 10);
    let expected = (px_m[0].to_radians()).tan();
    assert!((px_r[0] - expected).abs() < 0.01);
}

/// The ported `test_asin` / `test_acos` / `test_atan` and `test_atanh`
/// bodies, literal: `div_const` outputs a float raster (the libvips
/// divide / linear float promotion), so the `div_const(255.0)` setup
/// keeps 128/255 as ~0.502 and the expected values read the
/// post-division image. The float contract is what makes the literal
/// `test_atanh` body pass: atanh(0.502) ~ 0.5519 is finite, where the
/// old integer round-to-1 contract degenerated it to atanh(1) = inf.
#[test]
fn ported_surface_math_inverse_trig() {
    let data = vec![128u8; 100 * 100];
    let im = Raster::new(100, 100, PixelFormat::Gray8, data).unwrap();
    let im = im.div_const(255.0);
    assert!(im.format().is_float(), "div_const promotes to float");

    let result = im.asin();
    let px_i = im.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    let expected = px_i[0].asin().to_degrees();
    assert!((px_r[0] - expected).abs() < 0.1);

    let result = im.acos();
    let px_r = result.getpoint(50, 50);
    let expected = px_i[0].acos().to_degrees();
    assert!((px_r[0] - expected).abs() < 0.1);

    let result = im.atan();
    let px_r = result.getpoint(50, 50);
    let expected = px_i[0].atan().to_degrees();
    assert!((px_r[0] - expected).abs() < 0.1);

    // The literal test_atanh body (ported_arithmetic.rs::test_atanh).
    let result = im.atanh();
    let px_r = result.getpoint(50, 50);
    let expected = px_i[0].atanh();
    assert!(
        expected.is_finite() && px_r[0].is_finite(),
        "atanh(128/255) is finite under the float div contract"
    );
    assert!((px_r[0] - expected).abs() < 0.01);
}

/// The ported `test_atan2` body (`ported_arithmetic.rs`).
#[test]
fn ported_surface_atan2() {
    let data_a = vec![128u8; 100 * 100];
    let data_b = vec![64u8; 100 * 100];
    let a = Raster::new(100, 100, PixelFormat::Gray8, data_a).unwrap();
    let b = Raster::new(100, 100, PixelFormat::Gray8, data_b).unwrap();

    let result = a.atan2(&b);
    let px_r = result.getpoint(50, 50);
    let expected = (128.0_f64).atan2(64.0).to_degrees();
    assert!((px_r[0] - expected).abs() < 0.1);
}

/// The ported `test_sinh` / `test_cosh` / `test_tanh` and
/// `test_asinh` / `test_acosh` bodies; the `atanh` op additionally
/// pinned from a hand-built float raster holding 128/255, which now
/// matches what `div_const(255.0)` itself produces (see
/// `ported_surface_math_inverse_trig` for the literal setup).
///
/// The ported sinh / cosh bodies probe (10, 10), where the mono image
/// holds 226; sinh(226) is about 7.07e97, far beyond `f32::MAX`
/// (3.4e38), so an `f32` sample can only hold `inf` there. That is
/// libvips behavior too: `vips_math` on uchar input produces a `float`
/// (f32) image, so the same probe overflows upstream. The pin therefore
/// probes sinh / cosh on values whose results are f32-representable
/// (0 at (50, 50), and a small constant image) and keeps tanh at
/// (10, 10), where tanh(226) = 1 is exact.
#[test]
fn ported_surface_math_hyperbolic() {
    let mono = make_test_mono();

    let result = mono.sinh();
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    let expected = px_m[0].sinh();
    assert!((px_r[0] - expected).abs() / expected.abs().max(1.0) < 0.01);

    let result = mono.cosh();
    let px_r = result.getpoint(50, 50);
    let expected = px_m[0].cosh();
    assert!((px_r[0] - expected).abs() / expected.abs().max(1.0) < 0.01);

    let small = Raster::new(100, 100, PixelFormat::Gray8, vec![3u8; 100 * 100]).unwrap();
    let px_r = small.sinh().getpoint(10, 10);
    assert!((px_r[0] - 3.0_f64.sinh()).abs() / 3.0_f64.sinh() < 0.01);
    let px_r = small.cosh().getpoint(10, 10);
    assert!((px_r[0] - 3.0_f64.cosh()).abs() / 3.0_f64.cosh() < 0.01);

    let result = mono.tanh();
    let px_m = mono.getpoint(10, 10);
    let px_r = result.getpoint(10, 10);
    let expected = px_m[0].tanh();
    assert!((px_r[0] - expected).abs() < 0.001);

    let data = vec![150u8; 100 * 100];
    let im = Raster::new(100, 100, PixelFormat::Gray8, data).unwrap();
    let px_r = im.asinh().getpoint(50, 50);
    assert!((px_r[0] - 150.0_f64.asinh()).abs() < 0.01);
    let px_r = im.acosh().getpoint(50, 50);
    assert!((px_r[0] - 150.0_f64.acosh()).abs() < 0.01);

    let v = (128.0f32) / 255.0;
    let fmt = PixelFormat::with_channels(1, 4).unwrap();
    let data: Vec<u8> = std::iter::repeat_n(v, 100 * 100)
        .flat_map(|s| s.to_ne_bytes())
        .collect();
    let im = Raster::new(100, 100, fmt, data).unwrap();
    let px_i = im.getpoint(50, 50);
    let px_r = im.atanh().getpoint(50, 50);
    let expected = px_i[0].atanh();
    assert!((px_r[0] - expected).abs() < 0.01);
}

/// The ported `test_log` / `test_log10` / `test_exp` / `test_exp10`
/// bodies.
#[test]
fn ported_surface_log_exp() {
    let mono = make_test_mono();

    let result = mono.log();
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    if px_m[0] > 0.0 {
        assert!((px_r[0] - px_m[0].ln()).abs() < 0.01);
    }
    let px_m = mono.getpoint(10, 10);
    let px_r = result.getpoint(10, 10);
    assert!(px_m[0] > 0.0, "the ring pixel is non-zero");
    assert!((px_r[0] - px_m[0].ln()).abs() < 0.01);

    let result = mono.log10();
    let px_r = result.getpoint(10, 10);
    assert!((px_r[0] - px_m[0].log10()).abs() < 0.01);

    let data = vec![2u8; 100 * 100];
    let im = Raster::new(100, 100, PixelFormat::Gray8, data).unwrap();
    let px_r = im.exp().getpoint(50, 50);
    assert!((px_r[0] - 2.0_f64.exp()).abs() < 0.01);
    let px_r = im.exp10().getpoint(50, 50);
    assert!((px_r[0] - 10.0_f64.powf(2.0)).abs() < 0.01);
}

/// The ported `test_neg` and `test_abs` bodies: neg produces a float
/// raster, and abs recovers the original values.
#[test]
fn ported_surface_neg_abs() {
    let mono = make_test_mono();
    let result = mono.neg();
    let px_m = mono.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    assert!((px_r[0] - (-px_m[0])).abs() < 1.0);

    let colour = make_test_colour();
    let negated = colour.neg();
    let result = negated.abs();
    let px_c = colour.getpoint(50, 50);
    let px_r = result.getpoint(50, 50);
    for (c, r) in px_c.iter().zip(px_r.iter()) {
        assert!((r - c).abs() < 1.0);
    }
}

/// The ported `test_pow` Required-API image form, `pow(&self, other)`,
/// plus the `wop` reversed spelling libvips pairs with it.
#[test]
fn ported_surface_pow_image() {
    let mono = make_test_mono();
    let exp = Raster::new(100, 100, PixelFormat::Gray8, vec![2u8; 100 * 100]).unwrap();
    let result = mono.pow(&exp);
    let px_m = mono.getpoint(10, 10);
    let px_r = result.getpoint(10, 10);
    assert!((px_r[0] - px_m[0].powf(2.0)).abs() < 1.0);

    let result = exp.wop(&mono);
    let px_r = result.getpoint(10, 10);
    assert!((px_r[0] - px_m[0].powf(2.0)).abs() < 1.0);
}

/// The ported `test_polar` / `test_rect` / `test_conjugate` bodies
/// (`ported_arithmetic.rs` complex_histogram), literal.
#[test]
fn ported_surface_complex() {
    let data = vec![100u8; 100 * 100];
    let re = Raster::new(100, 100, PixelFormat::Gray8, data.clone()).unwrap();
    let im_part = Raster::new(100, 100, PixelFormat::Gray8, data.clone()).unwrap();

    let complex = Raster::complexform(&re, &im_part);
    let polar = complex.polar();
    let magnitude_avg = polar.real().avg();
    assert!(
        (magnitude_avg - 100.0 * 2.0_f64.sqrt()).abs() < 1.0,
        "Magnitude avg should be ~141.42, got {magnitude_avg}"
    );
    let angle_avg = polar.imag().avg();
    assert!(
        (angle_avg - 45.0).abs() < 1.0,
        "Angle avg should be ~45 degrees, got {angle_avg}"
    );

    let mag = 100.0 * 2.0_f64.sqrt();
    let mag_data = vec![mag as u8; 100 * 100];
    let angle_data = vec![45u8; 100 * 100];
    let re = Raster::new(100, 100, PixelFormat::Gray8, mag_data).unwrap();
    let im_part = Raster::new(100, 100, PixelFormat::Gray8, angle_data).unwrap();
    let complex = Raster::complexform(&re, &im_part);
    let rect = complex.rect();
    assert!((rect.real().avg() - 100.0).abs() < 2.0, "Real part ~100");
    assert!(
        (rect.imag().avg() - 100.0).abs() < 2.0,
        "Imaginary part ~100"
    );

    let data = vec![100u8; 100 * 100];
    let re = Raster::new(100, 100, PixelFormat::Gray8, data.clone()).unwrap();
    let im_part = Raster::new(100, 100, PixelFormat::Gray8, data).unwrap();
    let complex = Raster::complexform(&re, &im_part);
    let conj = complex.conj();
    assert!((conj.real().avg() - 100.0).abs() < 1.0);
    assert!((conj.imag().avg() - (-100.0)).abs() < 1.0);
}

/// The ported `test_hough_line` body with the draw call in the pinned
/// `&[ink]` form (the ported cell passes integer scalar ink, a mis-port
/// against the surface `ported_draw.rs` pins).
#[test]
fn ported_surface_hough_line() {
    let mut im = Raster::zeroed(100, 100, PixelFormat::Gray8).unwrap();
    im.draw_line(&[100], 10, 90, 90, 10);

    let hough = im.hough_line();
    let (_v, x, y) = hough.maxpos();

    let angle = 180.0 * x as f64 / hough.width() as f64;
    let distance = 100.0 * y as f64 / hough.height() as f64;

    assert!(
        (angle - 45.0).abs() < 5.0,
        "Angle should be ~45 degrees, got {angle}"
    );
    assert!(
        (distance - 75.0).abs() < 10.0,
        "Distance should be ~75, got {distance}"
    );
}

/// The ported `test_hough_circle` body with the draw call in the pinned
/// `&[ink]` outline form (the ported cell passes scalar ink plus a
/// `false` fill flag; the surface `ported_draw.rs` pins spells the
/// outline as plain `draw_circle` and the fill as `draw_circle_filled`).
#[test]
fn ported_surface_hough_circle() {
    let mut im = Raster::zeroed(100, 100, PixelFormat::Gray8).unwrap();
    im.draw_circle(&[100], 50, 50, 40);

    let hough = im.hough_circle(35, 45);
    let (_v, x, y) = hough.maxpos();
    let vec = hough.getpoint(x, y);
    let r = vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32 + 35)
        .unwrap();

    assert!(
        (x as f64 - 50.0).abs() < 2.0,
        "Centre x should be ~50, got {x}"
    );
    assert!(
        (y as f64 - 50.0).abs() < 2.0,
        "Centre y should be ~50, got {y}"
    );
    assert!(
        (r as f64 - 40.0).abs() < 2.0,
        "Radius should be ~40, got {r}"
    );
}
