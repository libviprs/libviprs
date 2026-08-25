//! Pins the conversion / orientation / colour-adjacent call surface
//! required by the libviprs-tests ported suite (libviprs-tests issue #55,
//! `tests/ported_conversion.rs` and `tests/ported_create.rs`, plus the
//! `grey` / `switch` call sites in `ported_arithmetic.rs` and
//! `ported_histogram.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types (including the `Option`
//! arguments and associated constructors), enum variants, and builder
//! chains. Behaviour depth is covered by the unit tests in
//! `src/conversion.rs`; this file is the API contract.
//!
//! Where a ported test's setup uses a fixture decode (`sample.jpg`) or an
//! operation from a later batch, the setup is reproduced with direct
//! `Raster` construction and the conversion expressions are kept literal.

use libviprs::{Align, Angle, Interpretation, JoinDirection, PixelFormat, Raster};

/// The ported `make_test_mono`: a 100x100 Gray8 band-reject ring image.
fn make_test_mono() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let mut data = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 - cx) / cx;
            let dy = (y as f64 - cy) / cy;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r > 0.5 { (r * 200.0).min(255.0) } else { 0.0 };
            data[(y * w + x) as usize] = ((v * 2.0 + 3.0) as u16).min(255) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

/// The ported `make_test_colour`: the same ring as `mono * [1, 2, 3] +
/// [2, 3, 4]`, as Rgb8.
fn make_test_colour() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 - cx) / cx;
            let dy = (y as f64 - cy) / cy;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r > 0.5 { (r * 200.0).min(255.0) } else { 0.0 };
            let off = ((y * w + x) * 3) as usize;
            data[off] = ((v * 1.0 + 2.0) as u16).min(255) as u8;
            data[off + 1] = ((v * 2.0 + 3.0) as u16).min(255) as u8;
            data[off + 2] = ((v * 3.0 + 4.0) as u16).min(255) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// The faithful `mask_ideal(reject, optical)` analogue behind the pyvips
/// fixture: a 0/1 ring mask scaled into `[1, 2, 3] + [2, 3, 4]`. The
/// falsecolour expectation `[20, 0, 41]` at (30, 30) comes from this
/// shape: band 0 there is `1 * 1 + 2 = 3` and PET LUT row 3 is
/// `[22, 0, 45]`, inside the ported tolerance of 5.
fn make_ideal_ring_colour() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 - cx) / cx;
            let dy = (y as f64 - cy) / cy;
            let r = (dx * dx + dy * dy).sqrt();
            let m = if r > 0.5 { 1u8 } else { 0 };
            let off = ((y * w + x) * 3) as usize;
            data[off] = m + 2;
            data[off + 1] = m * 2 + 3;
            data[off + 2] = m * 3 + 4;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// `Angle` stays non-exhaustive: the trailing arm must be required.
#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_angle_non_exhaustive(v: &Angle) {
    match v {
        Angle::D0 => {}
        Angle::D90 => {}
        Angle::D180 => {}
        Angle::D270 => {}
        _ => {}
    }
}

/// `Interpretation` stays non-exhaustive and carries every variant the
/// ported colour suite references.
#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_interpretation_non_exhaustive(v: &Interpretation) {
    match v {
        Interpretation::Multiband => {}
        Interpretation::Bw => {}
        Interpretation::Histogram => {}
        Interpretation::Xyz => {}
        Interpretation::Lab => {}
        Interpretation::Cmyk => {}
        Interpretation::Labq => {}
        Interpretation::Rgb => {}
        Interpretation::Cmc => {}
        Interpretation::Lch => {}
        Interpretation::Labs => {}
        Interpretation::Srgb => {}
        Interpretation::Yxy => {}
        Interpretation::Fourier => {}
        Interpretation::Rgb16 => {}
        Interpretation::Grey16 => {}
        Interpretation::Matrix => {}
        Interpretation::ScRgb => {}
        Interpretation::Hsv => {}
        Interpretation::OkLab => {}
        Interpretation::OkLch => {}
        _ => {}
    }
}

/// The `cast` call sites from the ported `test_byteswap` and `test_msb`
/// setups (widening casts keep values, and the format changes) and the
/// float casts from the ported `test_composite_non_separable` setup.
#[test]
fn ported_cast_call_sites() {
    let mono = make_test_mono();
    let im16 = mono.cast(PixelFormat::Gray16);
    assert_eq!(im16.format(), PixelFormat::Gray16);
    assert!((im16.avg() - mono.avg()).abs() < 0.001);

    let colour = make_test_colour();
    let im16 = colour.add_const(32.0).cast(PixelFormat::Rgb16);
    assert_eq!(im16.format(), PixelFormat::Rgb16);

    // The exact float cast expressions of test_composite_non_separable.
    let base = colour
        .add_const(100.0)
        .bandjoin_const(255.0)
        .cast(PixelFormat::RgbaF32);
    assert_eq!(base.format(), PixelFormat::RgbaF32);
    assert_eq!(base.format().channels(), 4);
    let overlay = colour.bandjoin_const(128.0).cast(PixelFormat::RgbaF32);
    assert_eq!(overlay.format(), PixelFormat::RgbaF32);
    // The alpha band cast to float reads back as the joined constant.
    assert_eq!(overlay.getpoint(50, 50)[3], 128.0);
}

/// The ported `test_copy` call sites: the builder chain with `xres` and
/// an interpretation change.
#[test]
fn ported_copy_call_sites() {
    let colour = make_test_colour();
    let copy = colour.copy().xres(42.0).build();
    assert!((copy.xres() - 42.0).abs() < 0.001);

    let copy = colour.copy().interpretation(Interpretation::Lab).build();
    assert_eq!(copy.interpretation(), Interpretation::Lab);
}

/// The ported `test_addalpha` body.
#[test]
fn ported_addalpha_call_site() {
    let colour = make_test_colour();
    let result = colour.addalpha();
    assert_eq!(result.format().channels(), 4);
    let alpha = result.extract_band(3);
    assert!((alpha.avg() - 255.0).abs() < 0.001);
}

/// The ported `test_falsecolour` body, on the faithful ideal-ring
/// fixture (see `make_ideal_ring_colour`).
#[test]
fn ported_falsecolour_call_site() {
    let colour = make_ideal_ring_colour();
    let result = colour.falsecolour();
    assert_eq!(result.width(), colour.width());
    assert_eq!(result.height(), colour.height());
    assert_eq!(result.format().channels(), 3);

    let px = result.getpoint(30, 30);
    assert!((px[0] - 20.0).abs() < 5.0);
    assert!((px[1] - 0.0).abs() < 5.0);
    assert!((px[2] - 41.0).abs() < 5.0);
}

/// The ported `test_flip` body: the four-flip chain is the identity.
#[test]
fn ported_flip_call_site() {
    let colour = make_test_colour();
    let result = colour.fliphor().flipver().fliphor().flipver();
    let max_diff: f64 = colour
        .data()
        .iter()
        .zip(result.data().iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1.0,
        "Double flip should be identity, max diff = {max_diff}"
    );
}

/// The ported `test_gamma` body: the default curve matches
/// `v^2.4 / (255^2.4 / 255)` within 1% of the range.
#[test]
fn ported_gamma_call_site() {
    let colour = make_test_colour();
    let exponent = 2.4;

    let result = colour.gamma(None);
    let before = colour.getpoint(30, 30);
    let after = result.getpoint(30, 30);
    let norm = 255.0_f64.powf(exponent) / 255.0;
    for (b, a) in before.iter().zip(after.iter()) {
        let expected = b.powf(exponent) / norm;
        assert!((a - expected).abs() < 255.0 / 100.0);
    }
}

/// The ported `test_rot` body: the D90 corner mapping on a 51x51 crop
/// and the D90/D270 roundtrip.
#[test]
fn ported_rot_call_site() {
    let colour = make_test_colour();
    let test = colour.crop(0, 0, 51, 51);

    let im2 = test.rot(Angle::D90);
    let before = test.getpoint(50, 50);
    let after = im2.getpoint(0, 50);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }

    let round = test.rot(Angle::D90).rot(Angle::D270);
    let max_diff: f64 = test
        .data()
        .iter()
        .zip(round.data().iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0);
}

/// The ported `test_autorot` shape: an upright image passes through
/// unchanged (the ported test decodes `sample.jpg`, which carries
/// orientation 1; the decode is reproduced with direct construction).
/// A tagged image is rotated and the tag cleared.
#[test]
fn ported_autorot_call_site() {
    let im = make_test_colour();
    let rotated = im.autorot();
    assert_eq!(rotated.width(), im.width());
    assert_eq!(rotated.height(), im.height());
    assert_eq!(rotated.orientation(), 1);

    let tagged = im.crop(0, 0, 40, 20).copy().orientation(6).build();
    let upright = tagged.autorot();
    assert_eq!((upright.width(), upright.height()), (20, 40));
    assert_eq!(upright.orientation(), 1);
}

/// The ported `test_wrap` body: quadrants swap and dimensions hold.
#[test]
fn ported_wrap_call_site() {
    let colour = make_test_colour();
    let result = colour.wrap();
    assert_eq!(result.width(), colour.width());
    assert_eq!(result.height(), colour.height());

    let before = colour.getpoint(0, 0);
    let after = result.getpoint(50, 50);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }

    let before = colour.getpoint(50, 50);
    let after = result.getpoint(0, 0);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }
}

/// The ported `test_arrayjoin` body: default single-row layout and the
/// `across = 1` stack, with band alignment between mono and colour.
#[test]
fn ported_arrayjoin_call_sites() {
    let mono = make_test_mono();
    let colour = make_test_colour();

    let im = Raster::arrayjoin(&[&mono, &colour], None, None);
    assert_eq!(im.width(), mono.width() * 2);
    assert_eq!(im.height(), mono.height());

    let im = Raster::arrayjoin(&[&mono, &colour], Some(1), None);
    assert_eq!(im.width(), colour.width());
    assert_eq!(im.height(), mono.height() + colour.height());
}

/// The ported `test_grey` body (`ported_create.rs`), both halves: the
/// uchar 0..255 ramp and the float 0.0..1.0 ramp the float PixelFormat
/// batch enabled.
#[test]
fn ported_grey_call_sites() {
    let im = Raster::grey(100, 90, true);
    assert_eq!(im.format(), PixelFormat::Gray8);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 90);
    let p = im.getpoint(0, 0);
    assert_eq!(p[0], 0.0);
    let p = im.getpoint(99, 0);
    assert_eq!(p[0], 255.0);

    // The float half, exactly as ported_create.rs::test_grey asserts it.
    let im = Raster::grey(100, 90, false);
    assert_eq!(im.width(), 100);
    assert_eq!(im.height(), 90);
    assert!(im.format().is_float());
    let p = im.getpoint(0, 0);
    assert!((p[0] - 0.0).abs() < 0.001);
    let p = im.getpoint(99, 0);
    assert!((p[0] - 1.0).abs() < 0.001);
}

/// The ported `test_identity` body (`ported_create.rs`).
#[test]
fn ported_identity_call_sites() {
    let im = Raster::identity();
    assert_eq!(im.width(), 256);
    assert_eq!(im.height(), 1);
    assert_eq!(im.format(), PixelFormat::Gray8);
    assert_eq!(im.getpoint(0, 0), vec![0.0]);
    assert_eq!(im.getpoint(255, 0), vec![255.0]);
    assert_eq!(im.getpoint(128, 0), vec![128.0]);

    let im = Raster::identity_ushort();
    assert_eq!(im.width(), 65536);
    assert_eq!(im.height(), 1);
    assert_eq!(im.getpoint(0, 0), vec![0.0]);
    assert_eq!(im.getpoint(65535, 0), vec![65535.0]);
}

/// The ported `test_switch` body (`ported_conversion.rs` /
/// `ported_histogram.rs`): ramp masks give an index image with avg 0.5,
/// and no-match pixels take the condition count.
#[test]
fn ported_switch_call_sites() {
    let x = Raster::grey(256, 256, true);

    let cond_lo = x.less_than_const(128.0);
    let cond_hi = x.more_eq_const(128.0);
    let index = Raster::switch(&[&cond_lo, &cond_hi]);
    assert!((index.avg() - 0.5).abs() < 0.01);

    let never_a = x.equal_const(1000.0);
    let never_b = x.equal_const(2000.0);
    let index = Raster::switch(&[&never_a, &never_b]);
    assert!((index.avg() - 2.0).abs() < 0.001);
}

/// The `join` call surface: both directions, every [`Align`] variant, the
/// `expand` flag, the `Option` shim / background / align arguments, the
/// nickname `FromStr`, and the fallible twin. The measured vips 8.18.4
/// sizes for the 3x2 / 2x3 oracle pair are pinned here too.
#[test]
fn ported_join_call_sites() {
    let a = Raster::new(3, 2, PixelFormat::Gray8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let c = Raster::new(2, 3, PixelFormat::Gray8, vec![10, 20, 30, 40, 50, 60]).unwrap();

    let im = a.join(&c, JoinDirection::Horizontal, false, None, None, None);
    assert_eq!((im.width(), im.height()), (5, 2));

    let im = a.join(&c, JoinDirection::Horizontal, true, None, None, None);
    assert_eq!((im.width(), im.height()), (5, 3));

    let im = a.join(
        &c,
        JoinDirection::Horizontal,
        true,
        Some(2),
        None,
        Some(Align::Centre),
    );
    assert_eq!((im.width(), im.height()), (7, 3));

    let background = [255.0];
    let im = a.join(
        &c,
        JoinDirection::Vertical,
        true,
        Some(1),
        Some(&background),
        Some(Align::High),
    );
    assert_eq!((im.width(), im.height()), (3, 6));

    let align: Align = "centre".parse().unwrap();
    assert_eq!(align, Align::Centre);
    assert!("sideways".parse::<Align>().is_err());

    let im = a
        .try_join(
            &c,
            JoinDirection::Vertical,
            false,
            None,
            None,
            Some(Align::Low),
        )
        .unwrap();
    assert_eq!((im.width(), im.height()), (2, 5));
}
