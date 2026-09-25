//! Pins the FITS call surface from outside the crate (issue #505).
//!
//! Integration tests compile as an external crate, which is the position a
//! caller is actually in, so this file proves the module's public shape
//! resolves and behaves there: the free `decode_fits`, the `encode_fits` /
//! `save_fits` pair on `Raster`, the typed `FitsError` reachable through
//! `SourceError`, and the `"fits"` row in the shared format dispatch.
//! Behaviour depth lives in the unit tests in `src/fits.rs`.
//!
//! There is no `fitsload` / `fitssave` cell in the ported suite to
//! reproduce literally, because vips reaches cfitsio through a filename
//! rather than through a source: `vips_foreign_load_fits_build`
//! (`fitsload.c`) refuses any source that has no file behind it. libviprs
//! decodes from a byte slice like every other codec here, so what a caller
//! writes against is this crate's own surface.

use libviprs::pixel::SampleKind;
use libviprs::source::{DecodeLimits, SourceError};
use libviprs::{FitsError, Interpretation, PixelFormat, Raster, decode_fits};

/// A 4x3 single-band BITPIX 8 file, built card by card so every byte in it
/// is visible here. The pixel block is written bottom row first, which is
/// the FITS scan order.
fn sample() -> Vec<u8> {
    let mut file = Vec::new();
    for card in [
        "SIMPLE  =                    T / file does conform to FITS standard",
        "BITPIX  =                    8 / number of bits per data pixel",
        "NAXIS   =                    2 / number of data axes",
        "NAXIS1  =                    4 / length of data axis 1",
        "NAXIS2  =                    3 / length of data axis 2",
        "END",
    ] {
        file.extend_from_slice(format!("{card:<80}").as_bytes());
    }
    file.resize(2880, b' ');
    // Bottom row first: image row 2, then 1, then 0.
    file.extend_from_slice(&[56, 63, 70, 77, 28, 35, 42, 49, 0, 7, 14, 21]);
    file.resize(2880 * 2, 0);
    file
}

/// The free decode entry point resolves from outside the crate and returns
/// the 8-bit grey carrier, right way up.
#[test]
fn decode_fits_is_public_and_returns_the_byte_carrier() {
    let raster = decode_fits(&sample(), DecodeLimits::default()).unwrap();
    assert_eq!(raster.width(), 4);
    assert_eq!(raster.height(), 3);
    assert_eq!(raster.format(), PixelFormat::Gray8);
    assert_eq!(raster.interpretation(), Interpretation::Bw);
    assert_eq!(
        raster.data(),
        &[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
    );
}

/// The header cards arrive as `fits-0`, `fits-1`, ... in file order, which
/// is the shape `vips_fits_get_header` attaches them in.
#[test]
fn header_cards_are_attached_as_numbered_fields() {
    let raster = decode_fits(&sample(), DecodeLimits::default()).unwrap();
    assert_eq!(
        raster.get_field("fits-1").unwrap().as_str(),
        "BITPIX  =                    8 / number of bits per data pixel"
    );
    assert!(raster.get_field("fits-5").is_none());
}

/// `encode_fits` and `save_fits` are both reachable on `Raster`, they agree
/// byte for byte, and a decode of what they write reproduces the pixels.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn encode_and_save_round_trip_from_outside_the_crate() {
    let raster = decode_fits(&sample(), DecodeLimits::default()).unwrap();
    let encoded = raster.encode_fits().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.fits");
    raster.save_fits(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), encoded);

    let back = decode_fits(&encoded, DecodeLimits::default()).unwrap();
    assert_eq!(back.data(), raster.data());
    assert_eq!(back.format(), raster.format());
}

/// The extension-dispatched save and the shared format dispatch both carry
/// a FITS row, and all three spellings vips registers reach it.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_shared_dispatch_carries_a_fits_row() {
    let raster = decode_fits(&sample(), DecodeLimits::default()).unwrap();
    let direct = raster.encode_fits().unwrap();
    assert_eq!(raster.encode_to_buffer("fits").unwrap(), direct);

    let dir = tempfile::tempdir().unwrap();
    for suffix in ["fits", "fit", "fts"] {
        let path = dir.path().join(format!("out.{suffix}"));
        raster.save(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), direct);
    }
}

/// A BITPIX libviprs has no carrier for is refused by name rather than
/// silently narrowed, and the refusal arrives as a typed variant through
/// `SourceError`. BITPIX -64 (double) is the one that still refuses;
/// BITPIX 32 unscaled used to be this test's fixture, until issue #966
/// gave it a signed carrier, at which point
/// `an_unscaled_signed_carrier_decodes_from_outside_the_crate` below
/// became its positive control.
#[test]
fn an_unreachable_carrier_surfaces_a_typed_fits_error() {
    let mut file = sample();
    // Rewrite BITPIX 8 as BITPIX -64, which vips's own table has no row
    // for either (`fits.c:196-204`).
    let card = format!(
        "{:<80}",
        "BITPIX  =                  -64 / number of bits per data pixel"
    );
    file[80..160].copy_from_slice(card.as_bytes());
    match decode_fits(&file, DecodeLimits::default()) {
        Err(SourceError::Fits(FitsError::UnsupportedCarrier { bitpix, .. })) => {
            assert_eq!(bitpix, -64);
        }
        other => panic!("expected a typed UnsupportedCarrier, got {other:?}"),
    }
}

/// The 32-bit twin of `an_unreachable_carrier_surfaces_a_typed_fits_error`
/// used to live here, before issue #966: an unscaled BITPIX 32 array is a
/// genuine signed integer and this crate's public surface now decodes it
/// onto `Int32` instead of refusing, matching `vips fitsload`.
#[test]
fn an_unscaled_signed_carrier_decodes_from_outside_the_crate() {
    let mut file = sample();
    let card = format!(
        "{:<80}",
        "BITPIX  =                   32 / number of bits per data pixel"
    );
    file[80..160].copy_from_slice(card.as_bytes());
    // BITPIX 32 widens each sample from one byte to four, so the payload
    // needs to widen with it: reuse the same twelve values the BITPIX 8
    // fixture carries, sign-extended, big-endian i32.
    let pixels: [i32; 12] = [56, 63, 70, 77, 28, 35, 42, 49, 0, 7, 14, 21];
    let mut payload = Vec::new();
    for p in pixels {
        payload.extend_from_slice(&p.to_be_bytes());
    }
    file.truncate(2880);
    file.extend_from_slice(&payload);
    file.resize(2880 * 2, 0);

    let raster = decode_fits(&file, DecodeLimits::default()).unwrap();
    assert_eq!(
        raster.format(),
        PixelFormat::with_kind(1, SampleKind::I32).unwrap()
    );
    let want: [i32; 12] = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77];
    let want_bytes: Vec<u8> = want.iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(raster.data(), want_bytes);
}

/// The geometry ceilings are the ones every other decoder honours, checked
/// on the declared header geometry before a byte is allocated.
#[test]
fn declared_geometry_is_bounded_by_decode_limits() {
    let mut file = sample();
    let card = format!(
        "{:<80}",
        "NAXIS1  =                65536 / length of data axis 1"
    );
    file[240..320].copy_from_slice(card.as_bytes());
    let limits = DecodeLimits::default().with_max_coord(1024);
    assert!(matches!(
        decode_fits(&file, limits),
        Err(SourceError::CoordLimitExceeded { .. })
    ));
}

/// Every carrier has a FITS spelling, unsigned, signed (issue #516) and the
/// unsigned 32-bit integer one (issue #517), so the encoder is total over
/// `PixelFormat` and never has to refuse one.
#[test]
fn every_carrier_has_a_fits_spelling() {
    for (format, len) in [
        (PixelFormat::Gray8, 12),
        (PixelFormat::Rgb8, 36),
        (PixelFormat::Rgba8, 48),
        (PixelFormat::Gray16, 24),
        (PixelFormat::Rgb16, 72),
        (PixelFormat::RgbaF32, 192),
        (PixelFormat::with_kind(1, SampleKind::I8).unwrap(), 12),
        (PixelFormat::with_kind(1, SampleKind::I16).unwrap(), 24),
        (PixelFormat::with_kind(1, SampleKind::I32).unwrap(), 48),
        (PixelFormat::with_kind(1, SampleKind::U32).unwrap(), 48),
    ] {
        let raster = Raster::new(4, 3, format, vec![7u8; len]).unwrap();
        assert!(
            raster.encode_fits().is_ok(),
            "{format:?} should have a FITS spelling"
        );
    }
}
