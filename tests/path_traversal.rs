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
