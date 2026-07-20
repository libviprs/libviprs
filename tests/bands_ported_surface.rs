//! Pins the band-operation call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, `tests/ported_conversion.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types (including the `-1` literal and
//! `as u32` band indices the ported tests mix), `Option` parameters, and
//! `format().channels()` on results. Behavior depth is covered by the unit
//! tests in `src/bands.rs`; this file is the API contract.

use libviprs::{PixelFormat, Raster};

/// 3-band colour test image, same shape as the ported `make_test_colour`.
fn make_test_colour() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let off = ((y * w + x) * 3) as usize;
            data[off] = (x % 251) as u8;
            data[off + 1] = (y % 249) as u8;
            data[off + 2] = ((x + y) % 247) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// 1-band mono test image, same shape as the ported `make_test_mono`.
fn make_test_mono() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let data: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 253) as u8).collect();
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

/// The ported `test_bandjoin` call sites: image + image, const, vec.
#[test]
fn ported_surface_bandjoin() {
    let colour = make_test_colour();
    let mono = make_test_mono();

    let joined = colour.bandjoin(&mono);
    assert_eq!(joined.format().channels(), 4);

    let joined = colour.bandjoin_const(1.0);
    assert_eq!(joined.format().channels(), 4);
    let band3 = joined.extract_band(3);
    assert_eq!(band3.getpoint(50, 50), vec![1.0]);

    let joined = colour.bandjoin_vec(&[1.0, 2.0]);
    assert_eq!(joined.format().channels(), 5);
    let band3 = joined.extract_band(3);
    assert_eq!(band3.getpoint(50, 50), vec![1.0]);
    let band4 = joined.extract_band(4);
    assert_eq!(band4.getpoint(50, 50), vec![2.0]);
}

/// The ported `test_band_and` / `test_band_or` / `test_band_eor` call sites.
#[test]
fn ported_surface_bandbool() {
    let colour = make_test_colour();
    let px = colour.getpoint(50, 50);

    let result = colour.bandand();
    let expected = px[0] as u8 & px[1] as u8 & px[2] as u8;
    assert_eq!(result.getpoint(50, 50), vec![expected as f64]);

    let result = colour.bandor();
    let expected = px[0] as u8 | px[1] as u8 | px[2] as u8;
    assert_eq!(result.getpoint(50, 50), vec![expected as f64]);

    let result = colour.bandeor();
    let expected = px[0] as u8 ^ px[1] as u8 ^ px[2] as u8;
    assert_eq!(result.getpoint(50, 50), vec![expected as f64]);
}

/// The ported `test_bandmean` call site: round-to-nearest of the per-pixel
/// mean, matching vips `(sum + bands / 2) / bands` (not truncation).
///
/// The expected value is pinned as a hard literal captured from the vips
/// 8.18.4 oracle rather than recomputed from the production formula, so this
/// stays an independent regression guard instead of a change-detector that
/// would move in lockstep with a formula regression. For `make_test_colour`
/// pixel (50,50) the sample is [50, 50, 100] (sum 200), and vips bandmean
/// yields `(200 + 3/2)/3 = 67`:
///   `vips rawload colour.raw c.v 100 100 3 --format uchar`
///   `vips bandmean c.v bm.v; vips getpoint bm.v 50 50` -> `67`.
#[test]
fn ported_surface_bandmean() {
    let colour = make_test_colour();
    let result = colour.bandmean();
    // Pixel (50,50) is [50, 50, 100]; the vips oracle bandmean there is 67.
    assert_eq!(colour.getpoint(50, 50), vec![50.0, 50.0, 100.0]);
    assert_eq!(result.getpoint(50, 50), vec![67.0]);
}

/// The ported `test_bandrank` call site: `bandrank(&[&colour], None)`.
#[test]
fn ported_surface_bandrank() {
    let colour = make_test_colour();
    let result = colour.bandrank(&[&colour], None);
    assert_eq!(result.getpoint(50, 50), colour.getpoint(50, 50));
}

/// The ported `test_bandfold` call sites: fold all, unfold all, fold by 2.
#[test]
fn ported_surface_bandfold() {
    let mono = make_test_mono();

    let folded = mono.bandfold(None);
    assert_eq!(folded.width(), 1);

    let unfolded = folded.bandunfold(None);
    assert_eq!(unfolded.width(), mono.width());
    assert_eq!(unfolded.data(), mono.data());

    let folded2 = mono.bandfold(Some(2));
    assert_eq!(folded2.width(), mono.width() / 2);
}

/// The ported `test_slice` call sites: extract_band with plain literals,
/// a negative literal, and a `u32` cast; extract_bands with mixed types.
#[test]
fn ported_surface_extract_band_index_types() {
    let test = make_test_colour();
    let n_bands = test.format().channels() as i32;

    let b0 = test.extract_band(0);
    assert_eq!(b0.format().channels(), 1);

    let blast = test.extract_band(-1);
    assert_eq!(blast.format().channels(), 1);
    let orig_last = test.extract_band((n_bands - 1) as u32);
    assert_eq!(blast.data(), orig_last.data());

    let s = test.extract_bands(1, 2);
    assert_eq!(s.format().channels(), 2);

    let s = test.extract_bands(1, (n_bands - 2) as u32);
    assert_eq!(s.format().channels(), 1);

    let s = test.extract_bands(0, 2);
    assert_eq!(s.format().channels(), 2);

    let s = test.extract_bands(1, (n_bands - 1) as u32);
    assert_eq!(s.format().channels(), 2);
}

/// The ported `test_extract` band slice: bands 1..3 of the colour image.
#[test]
fn ported_surface_extract_bands_values() {
    let colour = make_test_colour();
    let sub = colour.extract_bands(1, 2);
    let px = colour.getpoint(30, 30);
    assert_eq!(sub.getpoint(30, 30), vec![px[1], px[2]]);
}
