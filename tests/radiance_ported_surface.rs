//! Pins the Radiance call surface from outside the crate (issue #506).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_radiance`, the
//! `encode_radiance` / `save_radiance` pair on `Raster`, the
//! `radiance::SaveOptions` struct literal with `..Default::default()`, and
//! the typed `RadianceError` reachable through `SourceError`. Behaviour
//! depth lives in the unit tests in `src/radiance.rs`.
//!
//! There is no `radsave` / `radload` cell in the ported suite to reproduce
//! literally: libvips models Radiance as a *coding* rather than as an
//! operation pair, so what a caller writes against is this crate's own
//! codec surface.

use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{PixelFormat, RadianceError, Raster, decode_radiance, radiance};

/// A minimal 6x1 Radiance file: below `MINELEN`, so its payload is flat
/// RGBE and every byte in it is visible here.
fn sample() -> Vec<u8> {
    let mut file = Vec::from(*b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 6\n");
    for i in 0..6u8 {
        file.extend_from_slice(&[255, 128, 64, 128 + i]);
    }
    file
}

/// The free decode entry point resolves from outside the crate and returns
/// the three-band float carrier tagged as scRGB.
#[test]
fn decode_radiance_is_public_and_returns_float_rgb() {
    let raster = decode_radiance(&sample(), DecodeLimits::default()).unwrap();
    assert_eq!(raster.width(), 6);
    assert_eq!(raster.height(), 1);
    assert_eq!(raster.format().channels(), 3);
    assert!(raster.format().is_float());
    assert_eq!(raster.interpretation(), libviprs::Interpretation::ScRgb);
}

/// The options struct is a `#[non_exhaustive]`, `Default`, module-scoped type
/// a caller outside the crate builds from `default()` through the `with_*`
/// setters. It used to be a struct literal here, which is what issue #630 took
/// away: this test compiles as an external crate, so it was itself the
/// downstream caller the old "later fields can be added without a breaking
/// change" promise would have broken.
#[test]
fn save_options_are_constructible_downstream() {
    let explicit = radiance::SaveOptions::default()
        .with_exposure(Some(2.0))
        .with_aspect(Some(1.5));
    let partial = radiance::SaveOptions::default().with_exposure(Some(2.0));
    assert_eq!(explicit.exposure, partial.exposure);
    assert_eq!(explicit.aspect, Some(1.5));
    assert_eq!(partial.aspect, None);

    let d = radiance::SaveOptions::default();
    assert_eq!((d.exposure, d.aspect), (None, None));
}

/// `encode_radiance` and `save_radiance` are both reachable on `Raster`,
/// and a decode of what they write reproduces the pixels.
#[test]
fn encode_and_save_round_trip_from_outside_the_crate() {
    let raster = decode_radiance(&sample(), DecodeLimits::default()).unwrap();
    let encoded = raster
        .encode_radiance(radiance::SaveOptions::default())
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.hdr");
    raster
        .save_radiance(&path, radiance::SaveOptions::default())
        .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), encoded);

    let back = decode_radiance(&encoded, DecodeLimits::default()).unwrap();
    assert_eq!(back.data(), raster.data());
}

/// Malformed bytes surface as the codec's own typed variant through
/// `SourceError`, so a caller can match the failure rather than parse a
/// message. This is the 81-byte reproducer from #539.
#[test]
fn malformed_bytes_surface_a_typed_radiance_error() {
    let mut file = Vec::from(*b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n");
    file.extend_from_slice(&[0, 0, 0, 0]);
    for _ in 0..8 {
        file.extend_from_slice(&[1, 1, 1, 0]);
    }
    match decode_radiance(&file, DecodeLimits::default()) {
        Err(SourceError::Radiance(RadianceError::RunawayRepeat { row })) => assert_eq!(row, 0),
        other => panic!("expected a typed RunawayRepeat, got {other:?}"),
    }
}

/// The encoder takes three-band float and says so plainly for anything
/// else, rather than inventing a colourspace policy.
#[test]
fn encode_rejects_a_non_float_raster() {
    let rgb8 = Raster::new(4, 1, PixelFormat::Rgb8, vec![9u8; 12]).unwrap();
    assert!(
        rgb8.encode_radiance(radiance::SaveOptions::default())
            .is_err()
    );
}
