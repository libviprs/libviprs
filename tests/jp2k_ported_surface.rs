//! The JPEG 2000 surface as a *consumer* sees it (issue #501).
//!
//! An integration test compiles as an external crate, which is what makes the
//! claims below different from the unit tests in `src/jp2k.rs`: they see only
//! the public API, so a type that stopped being public, a route that stopped
//! being wired into the shared dispatchers, or an error variant that vanished
//! behind a feature all fail here and nowhere else.
//!
//! Every fixture is embedded with `include_bytes!` rather than read from
//! disk. The Miri gate aborts the whole run on the first filesystem call its
//! isolation layer refuses, so a filesystem-touching test has to be recorded
//! in `tests/miri_fs_test_inventory.txt` and annotated; these are the same
//! bytes either way, so taking them at compile time keeps this file runnable
//! under Miri and off that ledger.

use libviprs::source::{DecodeLimits, decode_bytes, decode_bytes_with_limits};
use libviprs::{Jp2kError, PixelFormat, Raster, SourceError};

/// `oracle-captures/foreign-jp2k/fixtures/rgb_lossless.jp2`: a 4x3 sRGB ramp
/// written by `vips jp2ksave --lossless`, in the RFC 3745 JP2 container.
const JP2: &[u8] = include_bytes!("../oracle-captures/foreign-jp2k/fixtures/rgb_lossless.jp2");

/// `oracle-captures/foreign-jp2k/fixtures/depth8u.j2k`: a 5x1 greyscale bare
/// codestream written by `opj_compress`, which is the carrier `jp2ksave`
/// never writes and `jp2kload` still reads.
const J2K: &[u8] = include_bytes!("../oracle-captures/foreign-jp2k/fixtures/depth8u.j2k");

/// A 4x3 RGB ramp, the shape every encode assertion below starts from.
fn ramp() -> Raster {
    Raster::new(4, 3, PixelFormat::Rgb8, (0..36u8).collect()).expect("ramp fixture")
}

/// Issue #501. Both container forms reach the loader through the shared
/// content sniffer, not just through `decode_jp2k` directly.
///
/// This is the wiring claim and it cannot be made from inside the module: a
/// row missing from `SniffedFormat`'s route table leaves `decode_jp2k` working
/// perfectly and `decode_bytes` answering "these bytes are not an image".
/// The bare codestream is the half that would go missing first, because
/// `jp2ksave` never writes one, so nothing this crate produces would notice.
#[test]
fn both_container_forms_reach_the_loader_through_the_shared_sniffer() {
    for (name, bytes, geometry) in [("jp2", JP2, (4, 3)), ("j2k", J2K, (5, 1))] {
        let decoded = decode_bytes(bytes);
        if cfg!(feature = "jp2k") {
            let raster = decoded.unwrap_or_else(|e| panic!("{name} must decode: {e}"));
            assert_eq!((raster.width(), raster.height()), geometry, "{name}");
        } else {
            // Still routed here, which is the point: the answer is "this build
            // has no JPEG 2000" and not "these bytes are not an image".
            let err = decoded.expect_err("this build has no JPEG 2000 decoder");
            assert!(
                matches!(err, SourceError::Jp2k(Jp2kError::FeatureNotEnabled)),
                "{name}: {err:?}"
            );
        }
    }
}

/// Issue #501, and issue #634's promise that a caller's `match` has the same
/// arms in either build.
///
/// `Jp2kError` and `SourceError::Jp2k` are declared whether or not the feature
/// is on, so this compiles and runs identically both ways. Without that, a
/// consumer's error handling would change shape when some *other* crate in
/// their workspace enabled `jp2k`, because features are additive.
#[test]
fn the_error_type_has_the_same_shape_in_either_build() {
    let err = SourceError::Jp2k(Jp2kError::FeatureNotEnabled);
    assert!(
        err.to_string().contains("jp2k"),
        "the message names the format: {err}"
    );
    assert!(
        !err.is_alloc_limit(),
        "a missing feature is not an allocation refusal"
    );

    // And the budget refusal is the shared shape, reachable from out here.
    if cfg!(feature = "jp2k") {
        let bytes = ramp().encode_jp2k(Default::default()).expect("encode");
        let err =
            decode_bytes_with_limits(&bytes, DecodeLimits::default().with_max_alloc_bytes(35))
                .expect_err("36 bytes is past a 35-byte budget");
        assert!(
            matches!(err, SourceError::AllocLimitExceeded { .. }),
            "the budget must report the one shared shape (issue #686): {err:?}"
        );
        assert!(err.is_alloc_limit());
    }
}

/// Issue #501. The encoder is on `Raster` and round-trips through the public
/// decode entry point, which is the whole of what a consumer does with it.
///
/// The `Unsupported` half is the other build's claim: `encode_jp2k` keeps its
/// signature without the feature and reports the same variant every format
/// without an encoder reports, carrying `"jp2k"`.
#[test]
fn the_encoder_round_trips_through_the_public_decode_entry_point() {
    let source = ramp();
    let encoded = source.encode_jp2k(Default::default());
    if cfg!(feature = "jp2k") {
        let bytes = encoded.expect("encode");
        let back = decode_bytes(&bytes).expect("decode");
        assert_eq!(back.data(), source.data(), "lossless is an identity");
        assert_eq!(back.format(), PixelFormat::Rgb8);
    } else {
        let err = encoded.expect_err("this build has no JPEG 2000 encoder");
        assert!(
            matches!(err, libviprs::EncodeError::Unsupported { ref format } if format == "jp2k"),
            "{err:?}"
        );
    }
}

/// Issue #501. The stubs this replaced are gone from the public API.
///
/// `Raster::encode_jp2k(quality, lossless)` and `encode_jp2k_chroma` used to
/// exist in `crate::foreign_stubs` and always returned
/// `EncodeError::Unsupported`. They are not deprecated aliases, they are
/// deleted, and `encode_jp2k` now takes `jp2k::SaveOptions`: this test is
/// where a consumer reading the changelog can see the new shape compiled.
#[test]
fn the_encoder_takes_save_options_rather_than_the_old_stub_arguments() {
    let options = libviprs::jp2k::SaveOptions::default();
    assert_eq!(options.compression, libviprs::jp2k::Compression::Lossless);
    // Compiles, which is the assertion: the old two-argument stub does not.
    let _ = ramp().encode_jp2k(options);
}
