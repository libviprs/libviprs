//! Pins the GIF call surface from outside the crate (issues #570, #571).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_gif`, the `encode_gif` /
//! `save_gif` pair on `Raster`, the `gif::SaveOptions` struct literal with
//! `..Default::default()`, and the typed `GifError` reachable through
//! `SourceError`. Behaviour depth lives in the unit tests in `src/gif.rs`.
//!
//! There is no `gifload` / `gifsave` cell in the ported suite to reproduce
//! literally, so what a caller writes against is this crate's own codec
//! surface plus the extension dispatch every format shares.

use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{GifError, PixelFormat, Raster, decode_gif, gif};

/// A 12x9 raster of four repeating colours, well inside a GIF palette.
fn sample() -> Raster {
    let colours = [[0u8, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
    let mut data = Vec::with_capacity(12 * 9 * 3);
    for i in 0..12 * 9usize {
        data.extend_from_slice(&colours[i % 4]);
    }
    Raster::new(12, 9, PixelFormat::Rgb8, data).unwrap()
}

/// The free decode entry point resolves from outside the crate and expands
/// the palette to the band count `vips gifload` would pick.
#[test]
fn decode_gif_is_public_and_expands_the_palette() {
    let source = sample();
    let bytes = source.encode_gif(gif::SaveOptions::default()).unwrap();
    let raster = decode_gif(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(raster.width(), 12);
    assert_eq!(raster.height(), 9);
    assert_eq!(raster.interpretation(), libviprs::Interpretation::Srgb);
    assert_eq!(raster.get_int("n-pages"), Some(1));
    assert_eq!(raster.get_int("palette"), Some(1));
}

/// The options struct is a `#[non_exhaustive]`, `Default`, module-scoped type
/// a caller outside the crate builds from `default()` through the `with_*`
/// setters. It used to be a struct literal here, which is what issue #630 took
/// away: this test compiles as an external crate, so it was itself the
/// downstream caller the old "later fields can be added without a breaking
/// change" promise would have broken.
#[test]
fn save_options_are_constructible_downstream() {
    let explicit = gif::SaveOptions::default()
        .with_interlaced(true)
        .with_dither(0.5)
        .with_bitdepth(4);
    let partial = gif::SaveOptions::default().with_interlaced(true);
    assert_eq!(explicit.interlaced, partial.interlaced);
    assert_eq!(explicit.bitdepth, 4);
    assert_eq!(partial.bitdepth, 8);

    let d = gif::SaveOptions::default();
    assert!(!d.interlaced);
    assert!((d.dither - 1.0).abs() < f64::EPSILON);
    assert_eq!(d.bitdepth, 8);
}

/// `encode_gif` and `save_gif` are both reachable on `Raster`, and a decode
/// of what they write reproduces the pixels, because GIF's LZW is exactly
/// lossless once the palette fits.
#[test]
fn encode_and_save_round_trip_from_outside_the_crate() {
    let source = sample();
    let encoded = source.encode_gif(gif::SaveOptions::default()).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.gif");
    source.save_gif(&path, gif::SaveOptions::default()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), encoded);

    let back = decode_gif(&encoded, DecodeLimits::default()).unwrap();
    let rgb: Vec<u8> = match back.format() {
        PixelFormat::Rgb8 => back.data().to_vec(),
        PixelFormat::Rgba8 => back
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[..3].to_vec())
            .collect(),
        other => panic!("gifload emits 3 or 4 bands, got {other:?}"),
    };
    assert_eq!(rgb, source.data());
}

/// Malformed bytes surface as the codec's own typed variant through
/// `SourceError`, so a caller can match the failure rather than parse a
/// message.
#[test]
fn malformed_bytes_surface_a_typed_gif_error() {
    match decode_gif(b"GIF89a", DecodeLimits::default()) {
        Err(SourceError::Gif(GifError::Decode { .. } | GifError::NoFrames)) => {}
        other => panic!("expected a typed GifError, got {other:?}"),
    }
}

/// The encoder takes 8-bit rasters and says so plainly for anything else,
/// rather than inventing a colour model GIF does not have.
#[test]
fn encode_rejects_a_deep_raster() {
    let deep = Raster::new(4, 1, PixelFormat::Rgb16, vec![9u8; 4 * 3 * 2]).unwrap();
    assert!(deep.encode_gif(gif::SaveOptions::default()).is_err());
}
