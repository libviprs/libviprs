//! Pins the histogram call surface required by the libviprs-tests ported
//! suite (libviprs-tests issue #55, `tests/ported_histogram.rs`, plus the
//! `hist_find` / `hist_find_indexed` / `hist_find_ndim` call sites in
//! `ported_arithmetic.rs` and the `hist_ismonotonic` call site in
//! `ported_create.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types (including the `Option`
//! arguments), and return types. Behaviour depth is covered by the unit
//! tests in `src/histogram.rs`; this file is the API contract.
//!
//! Where a ported test's setup uses a fixture decode (`sample.jpg`), the
//! setup is reproduced with a synthetic raster of the same character (dark,
//! low contrast, textured) and the histogram expressions are kept literal.
//! `stdif` is pinned in `tests/arithmetic_ported_surface.rs` and the
//! `identity` / `grey` / `switch` constructors in
//! `tests/conversion_ported_surface.rs`.

use libviprs::{PixelFormat, Raster};

/// Stand-in for the `sample.jpg` decode: a dark, low-contrast, textured
/// image whose global and local equalisations both raise the mean and the
/// deviation, the properties the ported tests assert.
fn sample_like() -> Raster {
    let mut data = vec![0u8; 100 * 100];
    for y in 0..100usize {
        for x in 0..100usize {
            data[y * 100 + x] = (20 + (x * 7 + y * 13) % 30) as u8;
        }
    }
    Raster::new(100, 100, PixelFormat::Gray8, data).unwrap()
}

/// Stand-in for the ported `sample.jpg` + `extract_band(1)` percentile and
/// entropy setup: an Rgb8 image whose band 1 cycles through every value,
/// so the pooled distribution is close to uniform.
fn sample_like_colour() -> Raster {
    let mut data = vec![0u8; 100 * 100 * 3];
    for i in 0..(100 * 100) {
        data[i * 3] = 40;
        data[i * 3 + 1] = (i % 256) as u8;
        data[i * 3 + 2] = 80;
    }
    Raster::new(100, 100, PixelFormat::Rgb8, data).unwrap()
}

/// The ported `test_hist_cum` body.
#[test]
fn ported_hist_cum_call_site() {
    let im = Raster::identity();
    let total = im.avg() * 256.0;

    let cum = im.hist_cum();
    let px = cum.getpoint(255, 0);
    assert!(
        (px[0] - total).abs() < 0.001,
        "Cumulative histogram at 255 should equal total sum: got {}, expected {total}",
        px[0]
    );
}

/// The ported `test_hist_equal` body, on the synthetic fixture.
#[test]
fn ported_hist_equal_call_site() {
    let im = sample_like();
    let im2 = im.hist_equal();

    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());
    assert!(im.avg() < im2.avg(), "Equalized avg should be higher");
    assert!(
        im.deviate() < im2.deviate(),
        "Equalized deviate should be higher"
    );
}

/// The ported `test_hist_ismonotonic` body.
#[test]
fn ported_hist_ismonotonic_call_site() {
    let im = Raster::identity();
    assert!(im.hist_ismonotonic(), "Identity LUT should be monotonic");
}

/// The ported `test_hist_local` body, on the synthetic fixture.
#[test]
fn ported_hist_local_call_site() {
    let im = sample_like();

    let im2 = im.hist_local(10, 10, None);
    assert_eq!(im.width(), im2.width());
    assert_eq!(im.height(), im2.height());
    assert!(im.avg() < im2.avg());
    assert!(im.deviate() < im2.deviate());

    let im3 = im.hist_local(10, 10, Some(3.0));
    assert_eq!(im.width(), im3.width());
    assert_eq!(im.height(), im3.height());
    assert!(
        im3.deviate() < im2.deviate(),
        "Clamped CLAHE should have less contrast than unlimited"
    );
}

/// The ported `test_hist_match` body.
#[test]
fn ported_hist_match_call_site() {
    let im = Raster::identity();
    let im2 = Raster::identity();

    let matched = im.hist_match(&im2);

    let max_diff: f64 = im
        .data()
        .iter()
        .zip(matched.data().iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 0.001,
        "hist_match of identical histograms should be identity, got max diff {max_diff}"
    );
}

/// The ported `test_hist_norm` body.
#[test]
fn ported_hist_norm_call_site() {
    let im = Raster::identity();
    let im2 = im.hist_norm();

    let max_diff: f64 = im
        .data()
        .iter()
        .zip(im2.data().iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.001, "hist_norm of identity should be identity");
}

/// The ported `test_hist_plot` body.
#[test]
fn ported_hist_plot_call_site() {
    let im = Raster::identity();
    let im2 = im.hist_plot();

    assert_eq!(im2.width(), 256);
    assert_eq!(im2.height(), 256);
    assert_eq!(im2.format(), PixelFormat::Gray8);
}

/// The ported `test_hist_map` body.
#[test]
fn ported_hist_map_call_site() {
    let im = Raster::identity();
    let im2 = im.maplut(&im);

    let max_diff: f64 = im
        .data()
        .iter()
        .zip(im2.data().iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.001, "maplut with identity should be identity");
}

/// The ported `test_percent` body, on the synthetic colour fixture.
#[test]
fn ported_percent_call_site() {
    let im = sample_like_colour();
    let band1 = im.extract_band(1);

    let pc = band1.percent(90.0);

    let total_pixels = band1.width() as f64 * band1.height() as f64;
    let n_below: f64 = band1.data().iter().filter(|&&b| (b as f64) <= pc).count() as f64;
    let pc_set = 100.0 * n_below / total_pixels;

    assert!(
        (pc_set - 90.0).abs() < 1.0,
        "90th percentile should capture ~90% of pixels, got {pc_set}%"
    );
}

/// The ported `test_hist_entropy` shape: `hist_find` then `hist_entropy`.
/// The near-uniform synthetic band has close to 8 bits of entropy where
/// `sample.jpg` band 1 has 6.67; the call chain is what is pinned.
#[test]
fn ported_hist_entropy_call_site() {
    let im = sample_like_colour();
    let band1 = im.extract_band(1);

    let hist = band1.hist_find();
    let ent = hist.hist_entropy();

    assert!(
        (ent - 8.0).abs() < 0.01,
        "Entropy should be ~8.0, got {ent}"
    );
}

/// The ported `test_histfind` body (`ported_arithmetic.rs`): left half 0,
/// right half 10, built with `zeroed` + `insert`.
#[test]
fn ported_histfind_call_site() {
    let left = Raster::zeroed(50, 100, PixelFormat::Gray8).unwrap();
    let right = Raster::new(50, 100, PixelFormat::Gray8, vec![10u8; 50 * 100]).unwrap();
    let im = left.insert(&right, 50, 0, true);

    let hist = im.hist_find();
    let count_0 = hist.getpoint(0, 0);
    let count_10 = hist.getpoint(10, 0);
    let count_5 = hist.getpoint(5, 0);

    assert!((count_0[0] - 5000.0).abs() < 1.0, "5000 pixels at value 0");
    assert!(
        (count_10[0] - 5000.0).abs() < 1.0,
        "5000 pixels at value 10"
    );
    assert!((count_5[0] - 0.0).abs() < 1.0, "0 pixels at value 5");
}

/// The ported `test_histfind_indexed` body (`ported_arithmetic.rs`).
#[test]
fn ported_histfind_indexed_call_site() {
    let left = Raster::zeroed(50, 100, PixelFormat::Gray8).unwrap();
    let right = Raster::new(50, 100, PixelFormat::Gray8, vec![10u8; 50 * 100]).unwrap();
    let im = left.insert(&right, 50, 0, true);
    let index = im.floordiv_const(10.0);

    let hist = im.hist_find_indexed(&index);
    let h0 = hist.getpoint(0, 0);
    let h1 = hist.getpoint(1, 0);
    assert!((h0[0] - 0.0).abs() < 1.0);
    assert!((h1[0] - 50000.0).abs() < 1.0);
}

/// The ported `test_histfind_ndim` body (`ported_arithmetic.rs`).
#[test]
fn ported_histfind_ndim_call_site() {
    let mut data = vec![0u8; 100 * 100 * 3];
    for i in 0..(100 * 100) {
        data[i * 3] = 1;
        data[i * 3 + 1] = 2;
        data[i * 3 + 2] = 3;
    }
    let im = Raster::new(100, 100, PixelFormat::Rgb8, data).unwrap();

    let hist = im.hist_find_ndim(None);
    let px = hist.getpoint(0, 0);
    assert!((px[0] - 10000.0).abs() < 1.0);

    let hist = im.hist_find_ndim(Some(1));
    assert_eq!(hist.width(), 1);
    assert_eq!(hist.height(), 1);
    let px = hist.getpoint(0, 0);
    assert!((px[0] - 10000.0).abs() < 1.0);
}

/// The ported `test_case` body (`ported_histogram.rs`): switch classes on
/// a grey ramp mapped through scalar cases, with the documented
/// overflow-to-last behaviour.
#[test]
fn ported_case_call_site() {
    let x = Raster::grey(256, 256, true);

    let cond_lo = x.less_than_const(128.0);
    let cond_hi = x.more_eq_const(128.0);
    let index = Raster::switch(&[&cond_lo, &cond_hi]);
    let y = index.case(&[10.0, 20.0]);
    assert!(
        (y.avg() - 15.0).abs() < 0.001,
        "Two-class case avg should be 15, got {}",
        y.avg()
    );

    let c0 = x.less_than_const(64.0);
    let c1 = x.more_eq_const(64.0).bitand(&x.less_than_const(128.0));
    let c2 = x.more_eq_const(128.0).bitand(&x.less_than_const(192.0));
    let c3 = x.more_eq_const(192.0);
    let index = Raster::switch(&[&c0, &c1, &c2, &c3]);
    let y = index.case(&[10.0, 20.0, 30.0, 40.0]);
    assert!(
        (y.avg() - 25.0).abs() < 0.001,
        "Four-class case avg should be 25, got {}",
        y.avg()
    );

    let y = index.case(&[10.0, 20.0, 30.0]);
    assert!(
        (y.avg() - 22.5).abs() < 0.001,
        "Overflow-to-last case avg should be 22.5, got {}",
        y.avg()
    );
}
