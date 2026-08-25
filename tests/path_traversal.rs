//! Regression tests for manifest-driven path traversal in the checksum
//! verifier (issue #79).
//!
//! A `manifest.json` discovered inside an untrusted checkpoint/pyramid bundle
//! carries attacker-controlled `checksums.per_tile` keys. Before the fix these
//! were joined onto the pyramid root with `Path::join` and read whole with
//! `std::fs::read`, so a key like `/etc/hostname` or `../secret.txt` let a
//! hostile bundle read arbitrary files and surface their digests (a hash /
//! existence oracle), and an oversized target was an OOM primitive.
//!
//! These tests assert that such keys are rejected *before* any filesystem
//! access, and that a well-formed relative key still verifies.

use std::fs;

use serde_json::json;

fn write_manifest(pyramid: &std::path::Path, per_tile: serde_json::Value) {
    let manifest = json!({
        "checksums": {
            "algo": "blake3",
            "per_tile": per_tile,
        }
    });
    fs::write(
        pyramid.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn verify_rejects_parent_directory_traversal() {
    let parent = tempfile::tempdir().unwrap();
    let pyramid = parent.path().join("pyramid");
    fs::create_dir(&pyramid).unwrap();

    // A secret that lives OUTSIDE the pyramid dir. The traversal key points at
    // it; a vulnerable verifier reads and hashes it.
    let secret = parent.path().join("secret.txt");
    fs::write(&secret, b"TOP SECRET CONTENTS").unwrap();

    // Deliberately-wrong digest so a vulnerable verifier would surface the file
    // in `tiles_mismatched` (proving it read it).
    write_manifest(
        &pyramid,
        json!({ "../secret.txt": "0000000000000000000000000000000000000000000000000000000000000000" }),
    );

    let res = libviprs::checksum::verify_output(&pyramid);
    assert!(res.is_err(), "traversal path must be rejected, got {res:?}");
}

#[test]
fn verify_rejects_absolute_path() {
    let parent = tempfile::tempdir().unwrap();
    let pyramid = parent.path().join("pyramid");
    fs::create_dir(&pyramid).unwrap();

    // An absolute key. `Path::join` with an absolute component replaces the
    // base entirely, so a vulnerable verifier reads the absolute target.
    write_manifest(
        &pyramid,
        json!({ "/etc/hostname": "0000000000000000000000000000000000000000000000000000000000000000" }),
    );

    let res = libviprs::checksum::verify_output(&pyramid);
    assert!(res.is_err(), "absolute path must be rejected, got {res:?}");
}

#[test]
fn verify_accepts_wellformed_relative_tile() {
    let parent = tempfile::tempdir().unwrap();
    let pyramid = parent.path().join("pyramid");
    fs::create_dir_all(pyramid.join("0")).unwrap();

    let tile_bytes = b"legitimate tile bytes";
    let tile_rel = "0/0_0.raw";
    fs::write(pyramid.join(tile_rel), tile_bytes).unwrap();

    let digest = blake3::hash(tile_bytes).to_hex().to_string();
    write_manifest(&pyramid, json!({ tile_rel: digest }));

    let report = libviprs::checksum::verify_output(&pyramid).expect("valid manifest verifies");
    assert_eq!(report.tiles_checked, 1);
    assert_eq!(report.tiles_ok, 1);
    assert!(report.tiles_mismatched.is_empty());
    assert!(report.tiles_missing.is_empty());
}

#[test]
fn verify_streams_large_tile_without_buffering_whole_file() {
    let parent = tempfile::tempdir().unwrap();
    let pyramid = parent.path().join("pyramid");
    fs::create_dir_all(pyramid.join("0")).unwrap();

    // A tile larger than the internal streaming chunk (64 KiB), to exercise the
    // multi-read streaming path and confirm the digest still matches a
    // one-shot reference hash.
    let tile_bytes = vec![0xABu8; 3 * 1024 * 1024 + 7];
    let tile_rel = "0/0_0.raw";
    fs::write(pyramid.join(tile_rel), &tile_bytes).unwrap();

    let digest = blake3::hash(&tile_bytes).to_hex().to_string();
    write_manifest(&pyramid, json!({ tile_rel: digest }));

    let report = libviprs::checksum::verify_output(&pyramid).expect("valid manifest verifies");
    assert_eq!(report.tiles_ok, 1);
    assert!(report.tiles_mismatched.is_empty());
}

// ---------------------------------------------------------------------------
// SVG `<image xlink:href>` (issue #586)
// ---------------------------------------------------------------------------
//
// `usvg`'s default `ImageHrefResolver` treats an href as a filesystem path:
// it calls `Path::exists` and then `fs::read`, and because
// `Options::resources_dir` defaults to `None` the path is taken verbatim, so
// it resolves absolutely or relative to the process working directory. That
// is the same primitive the manifest tests above guard, arriving through a
// different door: an arbitrary read (a readable PNG's bytes land in the
// output pixels) plus an existence oracle (an href that exists takes a
// different branch from one that does not).
//
// `libviprs::svg` overrides both halves of the resolver to return `None`
// unconditionally. These tests prove it, and they are written so they would
// fail loudly against the stock resolver rather than passing by accident.

/// Build a document whose only content is an `<image>` pointing at `href`.
#[cfg(feature = "svg")]
fn svg_referencing(href: &str) -> Vec<u8> {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="8" height="8"><image xlink:href="{href}" x="0" y="0" width="8" height="8"/></svg>"#
    )
    .into_bytes()
}

/// An href that certainly exists and is readable, one that certainly does
/// not, and a relative traversal, must all be indistinguishable in the
/// rendered output. Any difference is an existence oracle.
#[test]
#[cfg(feature = "svg")]
fn svg_image_href_cannot_distinguish_an_existing_file_from_a_missing_one() {
    use libviprs::{SvgOptions, decode_svg};

    let existing = decode_svg(&svg_referencing("/etc/passwd"), SvgOptions::default())
        .expect("the document itself is valid; only the href is refused");
    let missing = decode_svg(
        &svg_referencing("/nonexistent/libviprs-502-not-a-real-path/passwd"),
        SvgOptions::default(),
    )
    .expect("the document itself is valid; only the href is refused");
    let traversal = decode_svg(
        &svg_referencing("../../../../../../etc/passwd"),
        SvgOptions::default(),
    )
    .expect("the document itself is valid; only the href is refused");

    assert_eq!(
        existing.data(),
        missing.data(),
        "an href that exists must render identically to one that does not"
    );
    assert_eq!(
        existing.data(),
        traversal.data(),
        "a relative traversal must render identically to a missing path"
    );
    assert_eq!(
        (existing.width(), existing.height()),
        (8, 8),
        "geometry must come from the document, never from the referenced file"
    );
}

/// The blocked render must be *empty*, not merely equal. Two identical
/// renders of the same leaked file would also compare equal, so pin that
/// nothing was drawn at all.
#[test]
#[cfg(feature = "svg")]
fn svg_image_href_renders_nothing_at_all() {
    use libviprs::{SvgOptions, decode_svg};

    let im = decode_svg(&svg_referencing("/etc/passwd"), SvgOptions::default()).unwrap();
    assert!(
        im.data().iter().all(|&b| b == 0),
        "a refused href must leave a fully transparent raster, not partial content"
    );
}

/// A real, readable image on disk is the strongest form of the leak: with the
/// stock resolver its pixels appear in the output. Point an href at a PNG
/// this test just wrote and assert the render stays empty.
#[test]
#[cfg(feature = "svg")]
fn svg_image_href_does_not_read_a_readable_image_off_disk() {
    use libviprs::{SvgOptions, decode_svg};

    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("secret.png");
    // A 1x1 opaque magenta PNG. If the resolver reads it, the 8x8 output is
    // filled with magenta instead of staying transparent.
    let png: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x0B, 0x02, 0xA0, 0x9C, 0x99, 0x2F, 0x38, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    fs::write(&secret, png).unwrap();

    let doc = svg_referencing(&secret.display().to_string());
    let im = decode_svg(&doc, SvgOptions::default()).unwrap();
    assert!(
        im.data().iter().all(|&b| b == 0),
        "the resolver must not read {} off disk",
        secret.display()
    );
}

/// A `data:` URI is refused too. Nothing on disk is involved, but a nested
/// `image/svg+xml` payload re-enters the parser, so the data half of the
/// resolver is closed as well and the two halves are pinned together.
#[test]
#[cfg(feature = "svg")]
fn svg_image_href_refuses_data_uris_as_well() {
    use libviprs::{SvgOptions, decode_svg};

    let nested = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPjxyZWN0IHdpZHRoPSI4IiBoZWlnaHQ9IjgiIGZpbGw9IiNmZjAwZmYiLz48L3N2Zz4=";
    let im = decode_svg(&svg_referencing(nested), SvgOptions::default()).unwrap();
    assert!(
        im.data().iter().all(|&b| b == 0),
        "the data-URI half of the resolver must be closed too"
    );
}
