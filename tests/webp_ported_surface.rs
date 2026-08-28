//! Pins the WebP call surface from outside the crate (issues #567, #568).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_webp`, the `encode_webp` /
//! `save_webp` pair on `Raster`, the `webp::SaveOptions` struct literal
//! with `..Default::default()`, and the `.webp` rows in the two shared
//! dispatchers (`Raster::save`'s extension route and
//! `Raster::encode_to_buffer`'s format route). Behaviour depth lives in
//! the unit tests in `src/webp.rs`.
//!
//! The ported foreign cell for WebP is `webpsave` with a `Q`, which this
//! crate deliberately cannot spell: the only encoder reachable in pure
//! Rust is lossless and has no quality knob at all. So what a caller
//! writes against is the `Compression` axis, and the point of pinning it
//! from outside is that `Compression::Lossy { .. }` can be added later
//! without breaking any of the code below.

use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{PixelFormat, Raster, decode_bytes, decode_webp, decode_webp_with, webp};

/// A 4x3 sRGB ramp, the same one `oracle-captures/foreign-webp` is built
/// from, so a failure here and a failure in the unit tests point at the
/// same pixels.
fn ramp() -> Raster {
    let mut data = Vec::with_capacity(4 * 3 * 3);
    for y in 0..3u32 {
        for x in 0..4u32 {
            data.push(((x * 61 + y * 13) % 256) as u8);
            data.push(((x * 97 + y * 151) % 256) as u8);
            data.push(((x * 29 + y * 211) % 256) as u8);
        }
    }
    Raster::new(4, 3, PixelFormat::Rgb8, data).unwrap()
}

/// The free decode entry point resolves from the crate root and returns
/// the 8-bit carrier, which is the only one WebP has.
#[test]
fn decode_webp_is_public_and_returns_an_eight_bit_raster() {
    let bytes = ramp().encode_webp(webp::SaveOptions::default()).unwrap();
    let raster = decode_webp(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!((raster.width(), raster.height()), (4, 3));
    assert_eq!(raster.format(), PixelFormat::Rgb8);
    assert_eq!(raster.interpretation(), libviprs::Interpretation::Srgb);
}

/// The options struct is a `#[non_exhaustive]`, `Default`, module-scoped type
/// a caller outside the crate builds from `default()` through the `with_*`
/// setters, and quality still has no spelling in it. It used to be a struct
/// literal here, which is what issue #630 took away: this test compiles as an
/// external crate, so it was itself the downstream caller the old "later
/// fields can be added without a breaking change" promise would have broken.
#[test]
fn save_options_are_constructible_downstream() {
    let explicit = webp::SaveOptions::default()
        .with_compression(webp::Compression::Lossless)
        .with_keep(webp::Keep::None);
    let partial = webp::SaveOptions::default().with_keep(webp::Keep::None);
    assert_eq!(explicit, partial);

    let d = webp::SaveOptions::default();
    assert_eq!(d.compression, webp::Compression::Lossless);
    assert_eq!(d.keep, webp::Keep::All);
}

/// The lossless encoder is a true identity from outside the crate too,
/// through both the buffer entry point and the file one.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn encode_and_save_round_trip_exactly() {
    let original = ramp();
    let bytes = original.encode_webp(webp::SaveOptions::default()).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    assert_eq!(decode_bytes(&bytes).unwrap().data(), original.data());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.webp");
    original
        .save_webp(&path, webp::SaveOptions::default())
        .unwrap();
    assert_eq!(
        libviprs::decode_file(&path).unwrap().data(),
        original.data()
    );
}

/// `.webp` is a live row in both shared dispatchers, and the content
/// sniffer routes the bytes back without help from the filename.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_shared_dispatchers_carry_webp() {
    let original = ramp();
    let dir = tempfile::tempdir().unwrap();

    let by_extension = dir.path().join("via_save.webp");
    original.save(&by_extension).unwrap();
    assert_eq!(
        libviprs::decode_file(&by_extension).unwrap().data(),
        original.data()
    );

    let by_format = original.encode_to_buffer("webp").unwrap();
    assert_eq!(decode_bytes(&by_format).unwrap().data(), original.data());

    // The filename is never consulted on the way back in.
    let misnamed = dir.path().join("actually_webp.png");
    std::fs::write(&misnamed, &by_format).unwrap();
    assert_eq!(
        libviprs::decode_file(&misnamed).unwrap().data(),
        original.data()
    );
}

/// A 16-bit raster is refused rather than narrowed, and the message tells
/// the caller what to do about it. vips narrows the same input by a right
/// shift of 8 instead; the module docs record that divergence.
#[test]
fn sixteen_bit_is_refused_from_outside_the_crate() {
    let wide = Raster::zeroed(4, 3, PixelFormat::Rgb16).unwrap();
    let err = wide
        .encode_webp(webp::SaveOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Rgb16"), "{err}");
    assert!(err.contains("cast"), "{err}");
}

/// The animated entry point and its options struct resolve from outside the
/// crate, and the option shape is the one a caller writes: struct-update
/// syntax over a `Default` that is page 0 and one frame, not every frame
/// (issue #569).
#[test]
fn the_animated_load_surface_resolves_from_outside_the_crate() {
    let explicit = webp::LoadOptions {
        page: 0,
        n: Some(1),
    };
    assert_eq!(webp::LoadOptions::default(), explicit);
    let all = webp::LoadOptions {
        n: None,
        ..Default::default()
    };
    assert_eq!(all.page, 0);

    // A still is a one-page file, so both option shapes load it and page 1
    // of it is a typed refusal rather than a panic or an empty raster.
    let bytes = ramp().encode_webp(webp::SaveOptions::default()).unwrap();
    for options in [explicit, all] {
        let raster = decode_webp_with(&bytes, options, DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.pages_loaded(), 1);
    }
    let err = decode_webp_with(
        &bytes,
        webp::LoadOptions {
            page: 1,
            n: Some(1),
        },
        DecodeLimits::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            SourceError::PageOutOfRange {
                format: "WebP",
                page: 1,
                pages: 1,
                ..
            }
        ),
        "{err:?}"
    );
}
