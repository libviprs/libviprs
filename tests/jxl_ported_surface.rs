//! Pins the JPEG XL call surface from outside the crate (issues #619, #620).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_jxl`, the `encode_jxl` /
//! `save_jxl` pair on `Raster`, the `jxl::SaveOptions` struct literal with
//! `..Default::default()`, and the `.jxl` rows in the two shared
//! dispatchers (`Raster::save`'s extension route and
//! `Raster::encode_to_buffer`'s format route). Behaviour depth lives in the
//! unit tests in `src/jxl.rs`.
//!
//! The ported foreign cell for JPEG XL is `jxlsave` with a `distance` and a
//! `Q`, neither of which this crate can spell: the only encoder reachable
//! in pure Rust is lossless modular and has no rate control at all. So what
//! a caller writes against is the `Compression` axis, and the point of
//! pinning it from outside is that `Compression::Lossy { .. }` can be added
//! later without breaking any of the code below.
//!
//! The loader's refusals are `JxlError`, reached through `SourceError::Jxl`,
//! and both are declared in either build (issue #634), so the arms a caller
//! writes do not change with the feature. The encoder's refusals stay on the
//! shared `EncodeError` spine, which is where `gif`, `radiance` and `fits`
//! leave theirs too.
//!
//! The codec itself is behind the non-default `jxl` feature, so the tests
//! that move pixels carry `#[cfg(feature = "jxl")]` and CI runs this file
//! twice. What runs under both is the shape: the options struct is still
//! constructible and every entry point is still callable and still typed,
//! which is the half a caller's code depends on either way.

#[cfg(not(feature = "jxl"))]
use libviprs::EncodeError;
#[cfg(feature = "jxl")]
use libviprs::decode_bytes;
use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{JxlError, PixelFormat, Raster, decode_jxl, decode_jxl_with, jxl};

/// A 4x3 sRGB ramp, the same one `oracle-captures/foreign-jxl` is built
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

/// The same ramp with 16-bit samples, which JPEG XL holds natively and
/// WebP does not.
#[cfg(feature = "jxl")]
fn ramp16() -> Raster {
    let mut data = Vec::with_capacity(4 * 3 * 3 * 2);
    for y in 0..3u32 {
        for x in 0..4u32 {
            for m in [1013u32, 4099, 7919] {
                let v = ((x * m + y * (m * 3)) % 65536) as u16;
                data.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    Raster::new(4, 3, PixelFormat::Rgb16, data).unwrap()
}

/// The free decode entry point resolves from the crate root and reports the
/// carrier the file declares rather than a fixed one.
#[test]
#[cfg(feature = "jxl")]
fn decode_jxl_is_public_and_follows_the_files_carrier() {
    let bytes = ramp().encode_jxl(jxl::SaveOptions::default()).unwrap();
    let raster = decode_jxl(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!((raster.width(), raster.height()), (4, 3));
    assert_eq!(raster.format(), PixelFormat::Rgb8);
    assert_eq!(raster.interpretation(), libviprs::Interpretation::Srgb);

    let wide = ramp16().encode_jxl(jxl::SaveOptions::default()).unwrap();
    let back = decode_jxl(&wide, DecodeLimits::default()).unwrap();
    assert_eq!(back.format(), PixelFormat::Rgb16);
    assert_eq!(back.interpretation(), libviprs::Interpretation::Rgb16);
}

/// The options struct is a `#[non_exhaustive]`, `Default`, module-scoped type
/// a caller outside the crate builds from `default()` through the `with_*`
/// setters, and neither distance nor quality has a spelling in it. It used to
/// be a struct literal here, which is what issue #630 took away: this test
/// compiles as an external crate, so it was itself the downstream caller the
/// old "later fields can be added without a breaking change" promise would
/// have broken.
#[test]
fn save_options_are_constructible_downstream() {
    let explicit = jxl::SaveOptions::default().with_compression(jxl::Compression::Lossless);
    let partial = jxl::SaveOptions::default();
    assert_eq!(explicit, partial);
    assert_eq!(jxl::SaveOptions::default(), explicit);
    assert_eq!(explicit.compression, jxl::Compression::Lossless);
}

/// The lossless encoder is a true identity from outside the crate too,
/// through both the buffer entry point and the file one, and at both
/// integer carriers.
#[test]
#[cfg(feature = "jxl")]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn encode_and_save_round_trip_exactly() {
    let dir = tempfile::tempdir().unwrap();
    for (name, original) in [("eight", ramp()), ("sixteen", ramp16())] {
        let bytes = original.encode_jxl(jxl::SaveOptions::default()).unwrap();
        assert_eq!(&bytes[..2], b"\xff\x0a", "{name} is a bare codestream");
        assert_eq!(decode_bytes(&bytes).unwrap().data(), original.data());

        let path = dir.path().join(format!("out_{name}.jxl"));
        original
            .save_jxl(&path, jxl::SaveOptions::default())
            .unwrap();
        assert_eq!(
            libviprs::decode_file(&path).unwrap().data(),
            original.data(),
            "{name}"
        );
    }
}

/// `.jxl` is a live row in both shared dispatchers, and the content sniffer
/// routes the bytes back without help from the filename.
#[test]
#[cfg(feature = "jxl")]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_shared_dispatchers_carry_jxl() {
    let original = ramp();
    let dir = tempfile::tempdir().unwrap();

    let by_extension = dir.path().join("via_save.jxl");
    original.save(&by_extension).unwrap();
    assert_eq!(
        libviprs::decode_file(&by_extension).unwrap().data(),
        original.data()
    );

    let by_format = original.encode_to_buffer("jxl").unwrap();
    assert_eq!(decode_bytes(&by_format).unwrap().data(), original.data());

    // The filename is never consulted on the way back in.
    let misnamed = dir.path().join("actually_jxl.png");
    std::fs::write(&misnamed, &by_format).unwrap();
    assert_eq!(
        libviprs::decode_file(&misnamed).unwrap().data(),
        original.data()
    );
}

/// The loader's refusals resolve as `SourceError::Jxl` carrying a `JxlError`
/// from outside the crate, which is the half of issue #634 a caller sees:
/// the variant is reachable and matchable without the `libviprs` internals
/// and without reading a message. Both the codec's own refusal and the
/// shared decode ceilings are pinned, because they are deliberately
/// different variants.
#[test]
#[cfg(feature = "jxl")]
fn decode_refusals_are_typed_from_outside_the_crate() {
    let bytes = ramp().encode_jxl(jxl::SaveOptions::default()).unwrap();

    let err = decode_jxl(&bytes[..6], DecodeLimits::default()).unwrap_err();
    assert!(
        matches!(err, SourceError::Jxl(JxlError::Truncated { .. })),
        "{err:?}"
    );

    let err = decode_jxl(b"not a jxl at all", DecodeLimits::default()).unwrap_err();
    assert!(
        matches!(err, SourceError::Jxl(JxlError::Decode { .. })),
        "{err:?}"
    );

    // The shared geometry ceilings stay on `SourceError` itself, the way
    // every other codec in the crate reports them.
    let err = decode_jxl(&bytes, DecodeLimits::default().with_max_coord(2)).unwrap_err();
    assert!(
        matches!(
            err,
            SourceError::CoordLimitExceeded {
                width: 4,
                height: 3,
                max_coord: 2
            }
        ),
        "{err:?}"
    );
}

/// A float raster is refused rather than quantised, and the message tells
/// the caller what to do about it. vips writes float samples natively; the
/// module docs record that divergence.
#[test]
#[cfg(feature = "jxl")]
fn float_is_refused_from_outside_the_crate() {
    let wide = Raster::zeroed(4, 3, PixelFormat::RgbaF32).unwrap();
    let err = wide
        .encode_jxl(jxl::SaveOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("float"), "{err}");
    assert!(err.contains("cast"), "{err}");
}

/// The floor libviprs has and vips does not: a single-pixel axis has no
/// encoder behind it, and the refusal names the floor rather than leaking
/// the dependency's own wording.
#[test]
#[cfg(feature = "jxl")]
fn a_single_pixel_axis_is_refused_from_outside_the_crate() {
    let thin = Raster::zeroed(1, 4, PixelFormat::Rgb8).unwrap();
    let err = thin
        .encode_jxl(jxl::SaveOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("1x4"), "{err}");
    assert!(err.contains("2 pixels on each axis"), "{err}");
}

/// Without the feature the surface does not move: the free decoder, both
/// encoders and the two shared dispatchers are all still callable at the
/// same signatures from outside the crate, and every one of them reports a
/// typed refusal rather than a missing symbol or a panic. This is the pin
/// that a consumer's code compiles against either build.
#[test]
#[cfg(not(feature = "jxl"))]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn without_the_feature_the_surface_is_unchanged_and_typed() {
    let raster = ramp();

    // The variant, not the message: this is the one `JxlError` a build
    // with the feature never produces, so it is what tells a caller that
    // the build has no JPEG XL rather than that the bytes are bad.
    let err = decode_jxl(&[0xff, 0x0a], DecodeLimits::default()).unwrap_err();
    assert!(
        matches!(err, SourceError::Jxl(JxlError::FeatureNotEnabled)),
        "{err:?}"
    );
    assert!(err.to_string().contains("JPEG XL"), "{err}");

    let err = raster.encode_jxl(jxl::SaveOptions::default()).unwrap_err();
    assert!(
        matches!(err, EncodeError::Unsupported { ref format } if format == "jxl"),
        "{err:?}"
    );

    let err = raster.encode_to_buffer("jxl").unwrap_err();
    assert!(
        matches!(err, EncodeError::Unsupported { ref format } if format == "jxl"),
        "{err:?}"
    );

    let dir = tempfile::tempdir().unwrap();
    assert!(
        raster
            .save_jxl(&dir.path().join("a.jxl"), jxl::SaveOptions::default())
            .is_err()
    );
    // `.jxl` leaves the extension route entirely without an encoder behind
    // it, so it reads as an unsupported extension rather than an encode
    // failure, and the refusal names the set this build really can write.
    let err = raster.save(&dir.path().join("b.jxl")).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("unsupported save extension"), "{message}");
    assert!(!message.contains("webp, jxl"), "{message}");
}

/// The animated entry point and its options struct resolve from outside the
/// crate in **either** build, and the option shape is the one a caller
/// writes: a `Default` that is page 0 and one frame, reached through `with_*`
/// setters because the struct is `#[non_exhaustive]` and a literal does not
/// compile out here (issues #621, #630).
///
/// The refusal under test is the feature-off one,
/// `JxlError::FeatureNotEnabled`, so this runs in both builds; the page
/// refusal needs a real file and lives in the unit tests.
#[test]
fn the_animated_load_surface_resolves_from_outside_the_crate() {
    let one = jxl::LoadOptions::default();
    assert_eq!((one.page, one.n), (0, 1));
    let all = jxl::LoadOptions::default().with_page(1).with_n(-1);
    assert_eq!((all.page, all.n), (1, -1));

    // Not a JPEG XL file, so the call fails either way; what is pinned here
    // is that it is callable, that it is typed the same in both builds, and
    // that the feature-off build says so rather than reporting a bad file.
    let err = decode_jxl_with(b"not a jxl file at all", DecodeLimits::default(), all)
        .expect_err("those bytes are not JPEG XL");
    if cfg!(feature = "jxl") {
        assert!(matches!(err, SourceError::Jxl(_)), "{err:?}");
    } else {
        assert!(
            matches!(err, SourceError::Jxl(JxlError::FeatureNotEnabled)),
            "{err:?}"
        );
    }
}
