//! Pins the OpenEXR call surface from outside the crate (issue #504).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_exr`, the typed `ExrError`
//! reachable through `SourceError`, the sniff route that reaches the same
//! codec from `decode_bytes` and `decode_file` regardless of the
//! extension, and the metadata a caller needs to know what a band means.
//! Behaviour depth lives in the unit tests in `src/exr.rs`.
//!
//! There is no `exrsave` cell to reproduce and no encoder half to pin.
//! libvips has never shipped an EXR writer: `vips -l` registers
//! `openexrload` and nothing else, and `vips copy src.png out.exr` answers
//! `"out.exr" is not a known file format`. That is captured in
//! `oracle-captures/foreign-exr/oracle.json` under
//! `findings.there_is_no_exr_saver` rather than asserted from memory.

use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{ExrError, Interpretation, PixelFormat, decode_bytes, decode_exr, decode_file};

/// The RGBA half fixture, written by the OpenEXR reference implementation
/// 3.4.15. See `oracle-captures/foreign-exr/make_corpus.cpp`.
fn sample() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/oracle-captures/foreign-exr/fixtures/rgba_half_zip.exr"
    );
    std::fs::read(path).expect("the committed EXR fixture must be readable")
}

/// The free decode entry point resolves from outside the crate and returns
/// the float carrier with one band per selected channel, in that band
/// count's canonical spelling: four bands is `RgbaF32`, which is the format
/// `with_channels(4, 4)` names, and it reports alpha (issue #531).
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn decode_exr_is_public_and_returns_float_bands() {
    let raster = decode_exr(&sample(), DecodeLimits::default()).unwrap();
    assert_eq!(raster.width(), 8);
    assert_eq!(raster.height(), 4);
    assert_eq!(raster.format(), PixelFormat::RgbaF32);
    assert_eq!(
        raster.format(),
        PixelFormat::with_channels(4, 4).unwrap(),
        "the RGBA carrier must be the canonical spelling of four float bands"
    );
    assert!(raster.format().is_float());
    assert!(
        raster.format().has_alpha(),
        "an RGBA EXR carries alpha, and resize consults has_alpha to decide \
         whether to premultiply"
    );
    assert_eq!(raster.interpretation(), Interpretation::ScRgb);
}

/// A caller who does not know the bytes are EXR gets the same raster from
/// the content-sniffing entry point, and a misnamed file still decodes:
/// the container is identified from its magic and never from the path
/// extension (issue #563).
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_sniff_route_reaches_the_exr_codec_from_both_entry_points() {
    let bytes = sample();
    let direct = decode_exr(&bytes, DecodeLimits::default()).unwrap();

    let sniffed = decode_bytes(&bytes).unwrap();
    assert_eq!(sniffed.format(), direct.format());
    assert_eq!(sniffed.data(), direct.data());

    let dir = tempfile::tempdir().unwrap();
    let misnamed = dir.path().join("actually_an_exr.png");
    std::fs::write(&misnamed, &bytes).unwrap();
    let from_file = decode_file(&misnamed).unwrap();
    assert_eq!(from_file.format(), direct.format());
    assert_eq!(
        from_file.data(),
        direct.data(),
        "the extension must never pick the loader"
    );
}

/// The channel names reach the caller, which is what makes a band count
/// other than three or four readable. vips has no equivalent: it emits
/// four bands for every file and says nothing about what is in them.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn channel_names_and_compression_are_readable_downstream() {
    let raster = decode_exr(&sample(), DecodeLimits::default()).unwrap();
    assert_eq!(
        raster.get_field("exr-channels").unwrap().as_str(),
        "R,G,B,A"
    );
    assert_eq!(raster.get_field("exr-compression").unwrap().as_str(), "zip");
}

/// The typed error is reachable and matchable from outside the crate,
/// through `SourceError` as well as on its own. The UINT ceiling is the
/// variant a caller is most likely to hit, so it is the one pinned here.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_typed_error_is_matchable_downstream() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/oracle-captures/foreign-exr/fixtures/rgba_uint_zip.exr"
    );
    let bytes = std::fs::read(path).unwrap();
    let err = decode_exr(&bytes, DecodeLimits::default()).unwrap_err();
    assert!(
        matches!(
            &err,
            SourceError::Exr(ExrError::UnsupportedSampleType { channel }) if channel == "R"
        ),
        "got {err:?}"
    );

    let err = decode_exr(b"not an exr at all", DecodeLimits::default()).unwrap_err();
    assert!(matches!(&err, SourceError::Exr(ExrError::BadMagic { .. })));
}

/// A caller can lower the decode budget and have it enforced on the
/// declared data window, the same knob every other loader honours.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn decode_limits_apply_to_exr_downstream() {
    let limits = DecodeLimits::default().with_max_pixels(16);
    let err = decode_exr(&sample(), limits).unwrap_err();
    assert!(
        matches!(
            err,
            SourceError::DimensionLimitExceeded {
                width: 8,
                height: 4,
                max_pixels: 16
            }
        ),
        "got {err:?}"
    );
}
