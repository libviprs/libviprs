//! Pins the colour-space and ICC call surface required by the
//! libviprs-tests ported suite (libviprs-tests issue #55,
//! `tests/ported_colour.rs`, plus the `colourspace("scrgb")` call sites in
//! `ported_foreign.rs` and the `de00` call site in `ported_resample.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: method names, argument types (including the
//! `Option` arguments and both `colourspace` call shapes), and return
//! types. Behaviour depth is covered by the unit tests in
//! `src/colour.rs`; this file is the API contract.
//!
//! Where a ported test's setup decodes a fixture (`sample.jpg`), the
//! setup is reproduced with a synthetic raster of the same character (an
//! 8-bit sRGB colour image; for the ICC tests, one with a real sRGB
//! profile attached) and the colour expressions are kept literal. The
//! ported ICC test reads `de.max_value()`; `max_value` belongs to the
//! create/arithmetic batch, so the same reads go through `f32_samples`
//! here until that batch lands.

use libviprs::{Intent, Interpretation, Pcs, PixelFormat, Raster};
use moxcms::ColorProfile;

/// Stand-in for the `sample.jpg` decode: an 8-bit sRGB colour gradient.
fn sample_like() -> Raster {
    let mut data = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            data.push((x * 4 + 2) as u8);
            data.push((y * 3 + 20) as u8);
            data.push(((x + y) * 2 + 20) as u8);
        }
    }
    Raster::new(64, 64, PixelFormat::Rgb8, data).unwrap()
}

/// The `sample.jpg` stand-in with a real sRGB profile attached, the
/// shape the ported ICC test decodes.
fn sample_like_profiled() -> Raster {
    let mut im = sample_like();
    im.set_icc_profile(&ColorProfile::new_srgb().encode().unwrap());
    im
}

/// Largest sample of a float raster; the ported tests spell this
/// `max_value()`, which lands with the create/arithmetic batch.
fn max_value(im: &Raster) -> f64 {
    im.f32_samples()
        .expect("float raster")
        .iter()
        .fold(f64::MIN, |m, &v| m.max(v as f64))
}

/// The ported `test_colourspace_roundtrip` body.
#[test]
fn ported_colourspace_roundtrip_call_site() {
    // Create constant Lab image [50, 0, 0, 42]
    let test = Raster::constant(100, 100, &[50.0, 0.0, 0.0, 42.0], Interpretation::Lab);

    let colour_spaces = [
        Interpretation::Xyz,
        Interpretation::Lch,
        Interpretation::Cmc,
        Interpretation::Labs,
        Interpretation::ScRgb,
        Interpretation::Hsv,
        Interpretation::Srgb,
        Interpretation::Yxy,
        Interpretation::OkLab,
        Interpretation::OkLch,
        Interpretation::Lab,
    ];

    let mut im = test.clone();
    for &cs in &colour_spaces {
        im = im.colourspace(cs);
        assert_eq!(im.interpretation(), cs);
    }

    // Round-trip should come back close to the original
    let before = test.getpoint(10, 10);
    let after = im.getpoint(10, 10);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            (b - a).abs() < 0.1,
            "Round-trip mismatch: before={b}, after={a}"
        );
    }

    // Test Lab→XYZ against Lindbloom reference for mid-grey
    let xyz = test.colourspace(Interpretation::Xyz);
    let px = xyz.getpoint(10, 10);
    let expected = [17.5064, 18.4187, 20.0547, 42.0];
    for (got, exp) in px.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 0.01,
            "Lab→XYZ Lindbloom mismatch: got={got}, expected={exp}"
        );
    }
}

/// The ported `test_colourspace_mono` body.
#[test]
fn ported_colourspace_mono_call_site() {
    let test = Raster::constant(100, 100, &[50.0, 0.0, 0.0, 42.0], Interpretation::Lab);

    for &mono_fmt in &[Interpretation::Bw, Interpretation::Grey16] {
        let test_grey = test.colourspace(mono_fmt);
        let mut im = test_grey.clone();

        let colour_spaces = [
            Interpretation::Xyz,
            Interpretation::Lab,
            Interpretation::Srgb,
            mono_fmt,
        ];
        for &cs in &colour_spaces {
            im = im.colourspace(cs);
            assert_eq!(im.interpretation(), cs);
        }

        let before = test_grey.getpoint(10, 10);
        let after = im.getpoint(10, 10);

        // Alpha should be preserved
        let alpha_diff = (after.last().unwrap() - before.last().unwrap()).abs();
        assert!(alpha_diff < 1.0, "Alpha not preserved in grey round-trip");

        // Grey value tolerance depends on bit depth
        let grey_threshold = if mono_fmt == Interpretation::Grey16 {
            30.0
        } else {
            1.0
        };
        let grey_diff = (after[0] - before[0]).abs();
        assert!(
            grey_diff < grey_threshold,
            "Grey value mismatch: before={}, after={}, diff={grey_diff}",
            before[0],
            after[0]
        );
    }
}

/// The ported `test_colourspace_cmyk` body.
#[test]
fn ported_colourspace_cmyk_call_site() {
    let test = Raster::constant(100, 100, &[50.0, 0.0, 0.0, 42.0], Interpretation::Lab);
    let cmyk = test.colourspace(Interpretation::Cmyk);

    let colour_spaces = [
        Interpretation::Xyz,
        Interpretation::Lab,
        Interpretation::Lch,
        Interpretation::Srgb,
    ];

    for &cs in &colour_spaces {
        let im = cmyk.colourspace(cs);
        let im2 = im.colourspace(Interpretation::Cmyk);

        let before = cmyk.getpoint(10, 10);
        let after = im2.getpoint(10, 10);
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 10.0,
                "CMYK round-trip mismatch via {cs:?}: before={b}, after={a}"
            );
        }
    }
}

/// The ported `test_lab_xyz_reference` body.
#[test]
fn ported_lab_xyz_reference_call_site() {
    let test = Raster::constant(100, 100, &[50.0, 0.0, 0.0], Interpretation::Lab);
    let xyz = test.colourspace(Interpretation::Xyz);
    let px = xyz.getpoint(10, 10);

    let expected = [17.5064, 18.4187, 20.0547];
    for (i, (&got, &exp)) in px.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "Lab→XYZ channel {i}: got={got}, expected={exp}"
        );
    }
}

/// The ported `test_de00` body.
#[test]
fn ported_de00_call_site() {
    let reference = Raster::constant(100, 100, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
    let sample = Raster::constant(100, 100, &[40.0, -20.0, 10.0], Interpretation::Lab);

    let difference = reference.de00(&sample);
    let px = difference.getpoint(10, 10);

    assert!(
        (px[0] - 30.238).abs() < 0.01,
        "dE00 should be ~30.238, got {}",
        px[0]
    );
    assert!(
        (px[1] - 42.0).abs() < 0.01,
        "Extra band (alpha) should be 42, got {}",
        px[1]
    );
}

/// The ported `test_de76` body.
#[test]
fn ported_de76_call_site() {
    let reference = Raster::constant(100, 100, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
    let sample = Raster::constant(100, 100, &[40.0, -20.0, 10.0], Interpretation::Lab);

    let difference = reference.de76(&sample);
    let px = difference.getpoint(10, 10);

    assert!(
        (px[0] - 33.166).abs() < 0.01,
        "dE76 should be ~33.166, got {}",
        px[0]
    );
    assert!(
        (px[1] - 42.0).abs() < 0.01,
        "Extra band should be 42, got {}",
        px[1]
    );
}

/// The ported `test_decmc` body.
#[test]
fn ported_de_cmc_call_site() {
    let reference = Raster::constant(100, 100, &[50.0, 10.0, 20.0, 42.0], Interpretation::Lab);
    let sample = Raster::constant(100, 100, &[55.0, 11.0, 23.0], Interpretation::Lab);

    let difference = reference.de_cmc(&sample);
    let px = difference.getpoint(10, 10);

    assert!(
        (px[0] - 4.97).abs() < 0.5,
        "dECMC should be ~4.97, got {}",
        px[0]
    );
    assert!(
        (px[1] - 42.0).abs() < 0.01,
        "Extra band should be 42, got {}",
        px[1]
    );
}

/// The ported `test_icc` body, on the profiled synthetic fixture, with a
/// real sRGB profile file standing in for `sRGB.icm`.
#[test]
fn ported_icc_call_site() {
    let dir = tempfile::tempdir().unwrap();
    let srgb_profile: std::path::PathBuf = dir.path().join("sRGB.icm");
    std::fs::write(&srgb_profile, ColorProfile::new_srgb().encode().unwrap()).unwrap();

    let test = sample_like_profiled();

    // Import then export should round-trip
    let imported = test.icc_import();
    let exported = imported.icc_export();
    let de = exported.de76(&test);
    let max_de: f64 = max_value(&de);
    assert!(
        max_de < 6.0,
        "ICC import+export dE76 should be < 6, got {max_de}"
    );

    // Export at 16-bit depth
    let exported_16 = imported.icc_export_with(16, Intent::Perceptual, None);
    assert_eq!(
        exported_16.format().bytes_per_channel(),
        2,
        "16-bit export should be 16bpc"
    );

    // With output_profile = sRGB
    let exported_srgb = imported.icc_export_with(8, Intent::Perceptual, Some(&srgb_profile));
    let srgb_conv = imported.colourspace(Interpretation::Srgb);
    let de = exported_srgb.de76(&srgb_conv);
    assert!(max_value(&de) < 6.0);

    // ICC transform
    let transformed = test.icc_transform(&srgb_profile);
    let srgb_conv = test.icc_import().colourspace(Interpretation::Srgb);
    let de = transformed.de76(&srgb_conv);
    assert!(max_value(&de) < 6.0);

    // Import with XYZ PCS
    let xyz_import = test.icc_import_with(Intent::Perceptual, None, Some(Pcs::Xyz));
    assert_eq!(xyz_import.interpretation(), Interpretation::Xyz);

    // Default import should be Lab
    let lab_import = test.icc_import();
    assert_eq!(lab_import.interpretation(), Interpretation::Lab);
}

/// The ported `test_cmyk` body, on the synthetic fixture.
#[test]
fn ported_cmyk_call_site() {
    let test = sample_like();

    let cmyk = test.colourspace(Interpretation::Cmyk);
    let srgb = cmyk.colourspace(Interpretation::Srgb);

    let before = test.getpoint(15, 21);
    let after = srgb.getpoint(15, 21);

    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            (b - a).abs() < 10.0,
            "CMYK→sRGB round-trip pixel mismatch: before={b}, after={a}"
        );
    }
}

/// The `colourspace("scrgb")` / `colourspace("srgb")` string call shape
/// used by the ported foreign tests (`test_uhdrsave_roundtrip`,
/// `thumbnail_with_profile`).
#[test]
fn ported_colourspace_str_call_site() {
    let im = sample_like();
    let scrgb = im.colourspace("scrgb");
    assert_eq!(scrgb.interpretation(), Interpretation::ScRgb);
    assert!(scrgb.format().is_float());

    let back = scrgb.colourspace("srgb");
    assert_eq!(back.interpretation(), Interpretation::Srgb);
    for (a, b) in im.data().iter().zip(back.data().iter()) {
        assert!(
            (*a as i16 - *b as i16).abs() <= 1,
            "sRGB->scRGB->sRGB drifted: {a} vs {b}"
        );
    }
}

/// The `de00` call shape of the ported resample suite
/// (`test_thumbnail_icc`): a colour difference between two decoded sRGB
/// images, read through `max_value`.
#[test]
fn ported_resample_de00_call_site() {
    let im_orig = sample_like();
    let im = sample_like();
    let de = im_orig.de00(&im);
    assert!(max_value(&de) < 10.0, "identical images should be ~0");
}

/// `Interpretation` also parses from the libvips nicknames, the shape
/// the string call sites rely on.
#[test]
fn interpretation_parse_call_site() {
    let space: Interpretation = "srgb".parse().unwrap();
    assert_eq!(space, Interpretation::Srgb);
    assert!("not-a-space".parse::<Interpretation>().is_err());
}
