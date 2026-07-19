//! Pins the IO / metadata / free-function call surface required by the
//! libviprs-tests ported suite (libviprs-tests issue #55:
//! `tests/ported_infrastructure.rs` metadata + tokenization + CLI
//! sections, the `save` / `get_field` / `set_field` / `set_typeof` call
//! sites in `ported_foreign.rs`, and the `save` call sites in
//! `ported_iofuncs.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: method names, argument types, `.into()`
//! conversions, and return types. Behaviour depth is covered by the unit
//! tests in `src/imageio.rs`; this file is the API contract.
//!
//! Where a ported test's setup decodes a fixture (`sample.jpg` with its
//! embedded ICC profile), the setup is reproduced with a synthetic
//! raster carrying a synthetic profile, and the metadata expressions are
//! kept literal.

use std::path::Path;

use libviprs::{MetadataValue, PixelFormat, Raster, decode_file};

/// Stand-in for the `sample.jpg` decode: a small RGB image with an ICC
/// profile and an EXIF blob attached, the two fields the keep/strip
/// ported tests read back.
fn sample_like() -> Raster {
    let mut data = vec![0u8; 32 * 32 * 3];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut im = Raster::new(32, 32, PixelFormat::Rgb8, data).unwrap();
    im.set_icc_profile(&[0, 0, 1, 44, 65, 68, 66, 69]);
    im.set_field("exif-data", MetadataValue::Blob(vec![0x49, 0x49, 42, 0]));
    im
}

/// The ported `test_keep_icc` body: save keeps the ICC profile.
#[test]
fn ported_keep_icc_call_site() {
    let im = sample_like();
    let profile = im.get_field("icc-profile-data");
    assert!(profile.is_some(), "fixture should have an ICC profile");

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("keep_icc.jpg");
    im.save(&out).unwrap();

    let im2 = decode_file(&out).unwrap();
    let profile2 = im2.get_field("icc-profile-data");
    assert!(
        profile2.is_some(),
        "ICC profile should be preserved after save"
    );
}

/// The ported `test_keep_none` body: save_stripped drops ICC and EXIF.
#[test]
fn ported_keep_none_call_site() {
    let im = sample_like();

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("stripped.jpg");
    im.save_stripped(&out).unwrap();

    let im2 = decode_file(&out).unwrap();
    assert!(
        im2.get_field("icc-profile-data").is_none(),
        "ICC should be stripped"
    );
    assert!(
        im2.get_field("exif-data").is_none(),
        "EXIF should be stripped"
    );
}

/// The ported `test_keep_custom_profile` body: attach a profile, save,
/// reload, and match the `MetadataValue::Blob` pattern on the field.
#[test]
fn ported_keep_custom_profile_call_site() {
    let mut im = sample_like();
    let srgb_icc = vec![7u8; 480];
    im.set_icc_profile(&srgb_icc);

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("custom_icc.jpg");
    im.save(&out).unwrap();

    let im2 = decode_file(&out).unwrap();
    if let Some(MetadataValue::Blob(profile)) = im2.get_field("icc-profile-data") {
        assert_eq!(
            profile.len(),
            srgb_icc.len(),
            "ICC profile size should match"
        );
        assert_eq!(profile, srgb_icc);
    } else {
        panic!("Custom ICC profile should be preserved");
    }
}

/// The ported `test_vips` body: `.v` save/load round-trip preserving
/// EXIF and pixels (`Path::join` produces the `&PathBuf` argument shape
/// the ported tests pass to `save`).
#[test]
fn ported_test_vips_call_site() {
    let im = sample_like();
    let dir = tempfile::tempdir().unwrap();
    let out_v = dir.path().join("test.v");
    im.save(&out_v).unwrap();
    let im2 = decode_file(&out_v).unwrap();
    assert_eq!(im2.width(), im.width());
    assert_eq!(im2.height(), im.height());
    assert_eq!(im.get_field("exif-data"), im2.get_field("exif-data"));

    // Part 2: synthetic 16x16 constant image round-trip.
    let data = vec![128u8; 16 * 16 * 3];
    let synth = Raster::new(16, 16, PixelFormat::Rgb8, data).unwrap();
    let out_v2 = dir.path().join("synth.v");
    synth.save(&out_v2).unwrap();
    let synth2 = decode_file(&out_v2).unwrap();
    assert_eq!(synth2.width(), 16);
    assert_eq!(synth2.height(), 16);
    assert_eq!(synth2.data(), synth.data());
}

/// The ported `test_jpegsave_exif` field expressions: `set_field` with
/// `.into()` string conversions, `get_field(...).unwrap().as_str()`, and
/// tag removal via `set_typeof(..., 0)` observed with `get_typeof`.
/// Structured EXIF embedding in the JPEG APP1 TIFF directory is deferred
/// to the foreign batch, so the round-trip here goes through `.v`, which
/// carries every attached field.
#[test]
fn ported_exif_field_call_sites() {
    let mut im = sample_like();
    im.set_field("exif-ifd2-UserComment", "Hello UserComment".into());
    im.set_field("exif-ifd0-Software", "TestSoftware".into());
    im.set_field("exif-ifd0-XPComment", "TestXPComment".into());
    let tag = "exif-ifd0-CameraOwnerName";
    im.set_field(tag, format!("test-{tag}").into());

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("exif_fields.v");
    im.save(&out).unwrap();
    let im2 = decode_file(&out).unwrap();
    assert_eq!(
        im2.get_field("exif-ifd2-UserComment").unwrap().as_str(),
        "Hello UserComment"
    );
    assert_eq!(
        im2.get_field("exif-ifd0-Software").unwrap().as_str(),
        "TestSoftware"
    );
    assert_eq!(
        im2.get_field("exif-ifd0-XPComment").unwrap().as_str(),
        "TestXPComment"
    );
    assert_eq!(im2.get_field(tag).unwrap().as_str(), format!("test-{tag}"));

    // Tag removal via typeof == 0.
    im.set_typeof("exif-ifd0-Software", 0);
    let out2 = dir.path().join("exif_removed.v");
    im.save(&out2).unwrap();
    let im3 = decode_file(&out2).unwrap();
    assert_eq!(im3.get_typeof("exif-ifd0-Software"), 0);
}

/// The ported `test_get_fields` body (from `ported_iofuncs.rs`): more
/// than 10 fields, "width" first.
#[test]
fn ported_get_fields_call_site() {
    let im = Raster::zeroed(10, 10, PixelFormat::Gray8).unwrap();
    let fields = im.get_fields();
    assert!(
        fields.len() > 10,
        "Should have more than 10 fields, got {}",
        fields.len()
    );
    assert_eq!(fields[0], "width");
}

/// The metadata getters the ported suites read alongside `get_field`:
/// width/height/bands via fields, format/interpretation/resolution/
/// offsets/orientation via both the typed getters and the field system.
#[test]
fn ported_header_getter_call_sites() {
    let im = sample_like();
    assert_eq!(im.get_field("width").unwrap().as_u32(), 32);
    assert_eq!(im.get_field("height").unwrap().as_u32(), 32);
    assert_eq!(im.get_field("bands").unwrap().as_u32(), 3);
    assert_eq!(im.get_field("format").unwrap().as_str(), "uchar");
    assert_eq!(im.get_field("interpretation").unwrap().as_str(), "srgb");
    assert_eq!(im.get_field("orientation").unwrap().as_u32(), 1);
    assert_eq!(im.get_field("xres").unwrap().as_f64(), im.xres());
    assert_eq!(im.get_field("yres").unwrap().as_f64(), im.yres());
    assert_eq!(im.get_field("xoffset").unwrap().as_i64(), 0);
    assert_eq!(im.get_field("yoffset").unwrap().as_i64(), 0);
}

/// The ported `test_token_parsing` body (13.7 Tokenization).
#[test]
fn ported_token_parsing_call_site() {
    use libviprs::tokenize;

    let result = tokenize("hello world");
    assert_eq!(result, vec!["hello", "world"]);

    let result = tokenize("\"hello world\"");
    assert_eq!(result, vec!["hello world"]);

    let result = tokenize("a \"b c\" d");
    assert_eq!(result, vec!["a", "b c", "d"]);
}

/// The ported `test_cli_thumbnail` geometry expressions (13.8 CLI).
#[test]
fn ported_thumbnail_geometry_call_site() {
    use libviprs::parse_thumbnail_geometry;

    let geom = parse_thumbnail_geometry("200");
    assert_eq!(geom.width, Some(200));

    let geom = parse_thumbnail_geometry("200x150");
    assert_eq!(geom.width, Some(200));
    assert_eq!(geom.height, Some(150));
}

/// The ported `test_cli_max_coord_flag` and `test_cli_max_coord_env` call
/// sites, migrated to the per-decode ceiling that superseded the removed
/// process-global shims (libviprs#462): the ceiling is configured with
/// [`DecodeLimits::with_max_coord`] and enforced by the decoder, and the
/// `VIPS_MAX_COORD` environment variable is read by the caller into a
/// [`DecodeLimits`] rather than a global.
#[test]
fn ported_max_coord_call_sites() {
    use libviprs::source::{DecodeLimits, SourceError, decode_bytes_with_limits};

    let limits = DecodeLimits::default().with_max_coord(1000);

    // A small image decodes regardless of the ceiling.
    let small = Raster::zeroed(500, 500, PixelFormat::Gray8)
        .unwrap()
        .encode_vips()
        .unwrap();
    let result = decode_bytes_with_limits(&small, limits).unwrap();
    assert_eq!(result.width(), 500);

    // A width past the ceiling is rejected before allocation.
    let big = Raster::zeroed(2000, 1, PixelFormat::Gray8)
        .unwrap()
        .encode_vips()
        .unwrap();
    assert!(matches!(
        decode_bytes_with_limits(&big, limits),
        Err(SourceError::CoordLimitExceeded {
            max_coord: 1000,
            ..
        })
    ));

    // The env variable is read by the caller into a `DecodeLimits`.
    // SAFETY: test-local mutation of this process's environment; the
    // variable is removed again before it is used.
    unsafe { std::env::set_var("VIPS_MAX_COORD", "500") };
    let ceiling: u32 = std::env::var("VIPS_MAX_COORD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap();
    unsafe { std::env::remove_var("VIPS_MAX_COORD") };
    assert_eq!(
        DecodeLimits::default().with_max_coord(ceiling).max_coord,
        500
    );
}

/// `Raster::save` accepts a `&Path` directly, the other argument shape
/// the ported tests use.
#[test]
fn save_accepts_path_ref() {
    let dir = tempfile::tempdir().unwrap();
    let owned = dir.path().join("as_path.png");
    let path: &Path = owned.as_path();
    sample_like().save(path).unwrap();
    assert!(path.exists());
}
