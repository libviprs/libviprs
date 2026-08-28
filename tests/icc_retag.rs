//! An ICC profile is removed when the interpretation is retagged to a space
//! the profile cannot describe (issue #720).
//!
//! libviprs kept it, so a `try_colourspace(Interpretation::Bw)` handed back a
//! one-band grey raster with a three-channel RGB profile still attached, and
//! the next `icc_transform` read that profile as if it described the samples.
//!
//! # What the oracle says
//!
//! Measured against the pinned `vips-8.18.6` (`oracle-captures/ORACLE_PIN.json`),
//! read off the binary. Three sources, each an 8x8 `.v` carrying a *real*
//! profile: 3144 bytes of sRGB from `vips icc_transform` against
//! `sRGB Profile.icc`, 2020 bytes of grey from `Generic Gray Profile.icc`, and
//! 55280 bytes of CMYK from `Generic CMYK Profile.icc`. Retag with
//! `vips copy in.v out.v --interpretation X` and read `icc-profile-data` back:
//!
//! ```text
//! target       RGB profile   grey profile   CMYK profile
//! b-w          removed       kept           removed
//! grey16       removed       kept           removed
//! cmyk         removed       removed        kept
//! srgb         kept          removed        removed
//! lab          kept          .              removed
//! rgb          kept          .              .
//! scrgb        kept          .              .
//! rgb16        kept          .              .
//! labs         kept          .              .
//! lch          kept          .              .
//! xyz          kept          .              .
//! yxy          kept          .              .
//! hsv          kept          .              .
//! fourier      kept          .              .
//! matrix       kept          .              .
//! histogram    kept          .              .
//! multiband    kept          .              .
//! ```
//!
//! Swapping the profile swaps the column, which is what makes this a rule
//! rather than a list of unlucky interpretations. The rule is the band count
//! the new tag implies against the profile's own colour space: `b-w` and
//! `grey16` imply one, `cmyk` implies four, and everything else implies three,
//! which is exactly `colour::space_bands`.
//!
//! # What the rule is not
//!
//! It does not read the image's actual band count. `vips bandmean` takes the
//! three-band source to one band and leaves the `scrgb` tag alone, and the
//! three-channel profile survives; so does `vips extract_band 0`. Only the tag
//! decides, and there is a test for that because it is the assumption a
//! band-count-based implementation would get wrong.
//!
//! # Profiles this build cannot read
//!
//! The colour space lives at bytes 16..20 of the ICC header, so a blob too
//! short to hold one, or holding a signature this build does not know, has no
//! answer. Those are **kept**: dropping an attachment because the code could
//! not parse it is worse than keeping one that may not apply, and it is the
//! same call `crate::imageio` makes for `.v` trailer fields it cannot
//! interpret (#565).

use libviprs::{Interpretation, MetadataValue, PixelFormat, Raster};

/// A profile whose header declares `space` at bytes 16..20. libviprs stores the
/// blob opaquely and reads only that field, so the rest is zeroes; the *oracle*
/// half of this was measured on three real profiles.
fn profile(space: &[u8; 4]) -> Vec<u8> {
    let mut p = vec![0u8; 132];
    p[16..20].copy_from_slice(space);
    p
}

fn tagged(bands: usize, space: &[u8; 4]) -> Raster {
    let fmt = PixelFormat::with_channels(bands, 1).unwrap();
    let data = vec![64u8; 8 * 8 * bands];
    let mut im = Raster::new(8, 8, fmt, data).unwrap();
    im.set_icc_profile(&profile(space));
    im.set_field("lane-720", MetadataValue::Str("carried".to_string()));
    im
}

/// Every interpretation, with the band count it implies.
const TARGETS: &[(Interpretation, usize)] = &[
    (Interpretation::Bw, 1),
    (Interpretation::Grey16, 1),
    (Interpretation::Cmyk, 4),
    (Interpretation::Rgb, 3),
    (Interpretation::Srgb, 3),
    (Interpretation::ScRgb, 3),
    (Interpretation::Rgb16, 3),
    (Interpretation::Lab, 3),
    (Interpretation::Labs, 3),
    (Interpretation::Lch, 3),
    (Interpretation::Xyz, 3),
    (Interpretation::Yxy, 3),
    (Interpretation::Hsv, 3),
    (Interpretation::Fourier, 3),
    (Interpretation::Matrix, 3),
    (Interpretation::Histogram, 3),
    (Interpretation::Multiband, 3),
    (Interpretation::OkLab, 3),
    (Interpretation::OkLch, 3),
    (Interpretation::Cmc, 3),
];

/// Issue #720. `Raster::copy().interpretation(..)` is the public retag surface
/// and the site the rule was measured on (`vips copy --interpretation`).
///
/// All three profile arms are swept against all twenty targets, so the
/// assertion is the rule rather than one column of it.
#[test]
fn a_retag_removes_a_profile_that_cannot_describe_the_new_space() {
    for (space, profile_bands) in [(b"RGB ", 3usize), (b"GRAY", 1), (b"CMYK", 4)] {
        let im = tagged(3, space);
        for &(target, target_bands) in TARGETS {
            let out = im.copy().interpretation(target).build();
            let kept = out.icc_profile().is_some();
            assert_eq!(
                kept,
                profile_bands == target_bands,
                "{space:?} profile retagged {target:?}: kept={kept}"
            );
            // Only the profile is revalidated; the rest of the metadata is a
            // plain carry and must not be collateral.
            assert_eq!(
                out.get_field("lane-720"),
                Some(MetadataValue::Str("carried".to_string())),
                "{space:?} profile retagged {target:?}: other fields survive"
            );
            assert_eq!(out.interpretation(), target, "the tag itself is set");
        }
    }
}

/// Issue #720. The ops that stamp an interpretation of their own go through the
/// same rule, which is the half `Raster::copy` cannot cover.
#[test]
fn colourspace_and_the_stamping_ops_apply_the_rule_too() {
    let rgb = tagged(3, b"RGB ");

    for target in [Interpretation::Bw, Interpretation::Grey16] {
        let out = rgb.try_colourspace(target).unwrap();
        assert_eq!(
            out.icc_profile(),
            None,
            "colourspace to {target:?} drops an RGB profile"
        );
        assert_eq!(
            out.get_field("lane-720"),
            Some(MetadataValue::Str("carried".to_string())),
            "colourspace to {target:?} keeps the other fields"
        );
    }

    for target in [Interpretation::Lab, Interpretation::Srgb] {
        let out = rgb.try_colourspace(target).unwrap();
        assert!(
            out.icc_profile().is_some(),
            "colourspace to {target:?} keeps an RGB profile"
        );
    }

    // `falsecolour` stamps sRGB on a one-band input, so a grey profile that was
    // valid on the way in is not on the way out.
    let mut grey = tagged(1, b"GRAY");
    grey.set_icc_profile(&profile(b"GRAY"));
    let out = grey.try_falsecolour().unwrap();
    assert_eq!(
        out.interpretation(),
        Interpretation::Srgb,
        "falsecolour tag"
    );
    assert_eq!(
        out.icc_profile(),
        None,
        "falsecolour stamps sRGB, so a grey profile goes with the tag"
    );
}

/// Issue #720. The rule reads the **tag**, not the image's band count, and this
/// is the test that says so.
///
/// Measured: `vips bandmean` and `vips extract_band 0` both take the three-band
/// source to one band, leave the `scrgb` tag alone, and keep the three-channel
/// profile. An implementation that compared the profile against
/// `format().channels()` would drop it here and pass every other test in this
/// file.
#[test]
fn the_band_count_does_not_decide_it_the_tag_does() {
    let im = tagged(3, b"RGB ")
        .copy()
        .interpretation(Interpretation::ScRgb)
        .build();
    assert!(im.icc_profile().is_some(), "control: the source has one");

    for (name, out) in [
        ("bandmean", im.try_bandmean().unwrap()),
        ("extract_band", im.try_extract_band(0).unwrap()),
    ] {
        assert_eq!(out.format().channels(), 1, "{name} really is one band");
        assert_eq!(
            out.interpretation(),
            Interpretation::ScRgb,
            "{name} keeps the tag"
        );
        assert!(
            out.icc_profile().is_some(),
            "{name} keeps the profile, because the tag still says three channels"
        );
    }
}

/// Issue #720. A profile this build cannot read is kept rather than dropped.
///
/// Dropping an attachment because the parser could not reach a verdict is worse
/// than keeping one that may not apply, and it is the same call `imageio` makes
/// for `.v` trailer values it cannot interpret (#565). It also keeps the rule
/// from silently eating a profile in a colour space a later libviprs learns
/// about.
#[test]
fn an_unreadable_profile_is_kept() {
    for blob in [
        vec![],
        vec![1u8, 2, 3, 4],
        // Long enough for a header, but the signature is not one we know.
        {
            let mut p = vec![0u8; 132];
            p[16..20].copy_from_slice(b"ZZZZ");
            p
        },
    ] {
        let mut im = Raster::new(8, 8, PixelFormat::Rgb8, vec![7u8; 8 * 8 * 3]).unwrap();
        im.set_icc_profile(&blob);
        for target in [
            Interpretation::Bw,
            Interpretation::Cmyk,
            Interpretation::Lab,
        ] {
            let out = im.copy().interpretation(target).build();
            assert_eq!(
                out.icc_profile(),
                Some(&blob[..]),
                "an unreadable {}-byte profile survives a retag to {target:?}",
                blob.len()
            );
        }
    }
}

/// Issue #720. Setting the interpretation through the **field API** keeps the
/// profile, where retagging through an **operation** drops it, and that split
/// is vips's rather than a gap in this change.
///
/// ```text
/// vipsedit rgb.v --interpretation b-w     -> b-w, icc-profile-data 3144 bytes
/// vips copy rgb.v out.v --interpretation b-w -> b-w, no icc-profile-data
/// ```
///
/// `vipsedit` writes the header in place, which is what
/// [`Raster::set_field`] is (libvips `vips_image_set`); `vips copy` runs an
/// operation, which is what [`Raster::copy`] is. A caller writing the header
/// is describing what the file already holds, so revalidating there would drop
/// a profile the file legitimately carries. The decoders assign the tag
/// directly for the same reason.
#[test]
fn the_field_api_writes_the_header_and_keeps_the_profile() {
    let im = tagged(3, b"RGB ");

    let mut by_field = im.clone();
    by_field.set_field("interpretation", MetadataValue::Str("b-w".to_string()));
    assert_eq!(
        by_field.interpretation(),
        Interpretation::Bw,
        "the tag is set"
    );
    assert!(
        by_field.icc_profile().is_some(),
        "a header write keeps the profile, as vipsedit does"
    );

    let by_op = im.copy().interpretation(Interpretation::Bw).build();
    assert_eq!(by_op.interpretation(), Interpretation::Bw, "the tag is set");
    assert_eq!(
        by_op.icc_profile(),
        None,
        "an operation drops it, as vips copy does"
    );
}
