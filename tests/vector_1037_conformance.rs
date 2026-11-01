//! Mobile PMTiles conformance (issue #1037, Android slice #1393).
//!
//! The Android run itself needs an emulator, so it lives in `conformance/android/` and runs on MARS.
//! These tests hold what a plain `cargo test` can: the harness covers both modes against the shared
//! style and golden archive, the recorded result says both passed, and the docs carry the offline
//! sentence the issue asks for.

use serde_json::Value;
use std::fs;
use std::path::Path;

const OFFLINE_SENTENCE: &str = "PMTiles is not handled by MapLibre Native offline-pack caching";

fn read(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn docs_say_pmtiles_skips_offline_pack_caching() {
    for rel in ["docs/mobile-pmtiles.md", "web/vector-viewer/README.md"] {
        assert!(
            read(rel).contains(OFFLINE_SENTENCE),
            "{rel} must say: {OFFLINE_SENTENCE}"
        );
    }
}

#[test]
fn docs_tell_apps_to_download_the_whole_archive() {
    let d = read("docs/mobile-pmtiles.md");
    assert!(
        d.contains("pmtiles://file://"),
        "docs show the local URL form"
    );
    assert!(d.contains("Range"), "docs describe remote Range mode");
}

#[test]
fn harness_runs_both_modes_against_the_shared_style_and_golden() {
    let t = read(
        "conformance/android/app/src/androidTest/java/org/libviprs/conformance/Issue1037ConformanceTest.java",
    );
    for needle in [
        "remoteRangeMode",
        "localFileMode",
        "missingArchiveRendersBlank",
        "pmtiles://file://",
        "queryRenderedFeatures",
        "/__log",
    ] {
        assert!(t.contains(needle), "the Android test must cover {needle}");
    }
    let g = read("conformance/android/app/build.gradle");
    assert!(
        g.contains("src/vector/black_on_white.style.json"),
        "the app must use the shared style, not a copy"
    );
    let r = read("conformance/android/run.sh");
    assert!(
        r.contains("web/vector-viewer/e2e/server.mjs"),
        "the harness must reuse the Range/CORS server"
    );
}

#[test]
fn recorded_android_result_passed_both_modes_on_the_committed_golden() {
    let v: Value = serde_json::from_str(&read("conformance/android/evidence/result.json")).unwrap();
    let golden = fs::metadata(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("web/vector-viewer/e2e/fixtures/golden.pmtiles"),
    )
    .unwrap()
    .len();
    assert_eq!(
        v["golden_bytes"].as_u64(),
        Some(golden),
        "result is for this golden"
    );
    for mode in [
        "remoteRangeMode",
        "localFileMode",
        "missingArchiveRendersBlank",
    ] {
        assert_eq!(v["tests"][mode].as_str(), Some("passed"), "{mode}");
    }
    assert!(v["maplibre_native_android"].is_string());
}
