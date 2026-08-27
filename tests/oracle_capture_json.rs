//! Guards that every checked-in oracle capture is JSON a strict parser reads
//! (issue #674).
//!
//! The captures under `oracle-captures/` are written by Python, and
//! `json.dump` spells a non-finite float as a bare `NaN`, `Infinity` or
//! `-Infinity` by default. Python's own `json.load` reads those back as a
//! documented non-standard extension, so a capture round-trips perfectly on
//! the machine that wrote it and is rejected WHOLE by `serde_json`, `jq` and
//! `JSON.parse`. Two files had drifted that way before this test existed:
//! `foreign-radiance` (one `Infinity`) and `foreign-uhdr` (six `NaN`).
//!
//! The repair is the spelling `foreign-nifti` already used, quoting the token
//! `json.dump` would have written bare. It is the encoding a reader has to
//! agree with, so it is pinned here rather than left to each consumer:
//!
//!   * a finite value stays a JSON number,
//!   * the other three are the strings `"NaN"`, `"Infinity"` and
//!     `"-Infinity"`.
//!
//! `null` was the other candidate and is not usable: it collapses NaN, `+inf`
//! and `-inf` onto one value, and the whole point of the two records that
//! carry them is WHICH one libvips produced.

use std::fs;
use std::path::{Path, PathBuf};

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every `oracle.json` under `oracle-captures/`, sorted. The tree is two
/// levels deep in one place (`convolution/canny`), so this recurses rather
/// than reading one directory.
fn oracle_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("oracle-captures must be readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.file_name().is_some_and(|n| n == "oracle.json") {
                out.push(path);
            }
        }
    }

    let mut out = Vec::new();
    walk(&repo_root().join("oracle-captures"), &mut out);
    out.sort();
    assert!(
        out.len() >= 13,
        "expected the whole capture tree, found only {out:?}"
    );
    out
}

/// Decode one recorded float: a JSON number, or one of the three quoted
/// tokens a capture writes for a value RFC 8259 has no literal for.
fn recorded_float(v: &serde_json::Value) -> f64 {
    match v {
        serde_json::Value::Number(n) => n.as_f64().expect("a recorded float fits an f64"),
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => panic!("{other:?} is not a recorded non-finite float"),
        },
        other => panic!("{other} is not a recorded float"),
    }
}

/// The write half of the same convention, mirroring what `capture.py` does on
/// the way out.
fn record_float(v: f64) -> serde_json::Value {
    if v.is_finite() {
        return serde_json::json!(v);
    }
    if v.is_nan() {
        return serde_json::json!("NaN");
    }
    serde_json::json!(if v > 0.0 { "Infinity" } else { "-Infinity" })
}

/// The guard the two bad files walked past for as long as they existed: parse
/// every capture with a parser that implements the standard and nothing else.
///
/// It collects rather than failing on the first file, because the interesting
/// answer when this goes red is how far the rot spread, not which name sorts
/// first.
#[test]
fn every_oracle_capture_is_strict_json() {
    let mut bad = Vec::new();
    for path in oracle_files() {
        let text = fs::read_to_string(&path).expect("a capture must be readable");
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&text) {
            let rel = path.strip_prefix(repo_root()).unwrap_or(&path);
            bad.push(format!("{}: {e}", rel.display()));
        }
    }
    assert!(
        bad.is_empty(),
        "these captures are not JSON any standard parser accepts. \
         A non-finite float is written as \"NaN\", \"Infinity\" or \
         \"-Infinity\", never bare (issue #674):\n  {}",
        bad.join("\n  ")
    );
}

/// The parser the test above leans on has to actually be strict, or that test
/// is a check that cannot fail for the reason it claims to. `serde_json`
/// rejects the three bare literals; this says so out loud so a future switch
/// to a lenient parser shows up here rather than as silence.
#[test]
fn the_strict_parser_really_does_reject_bare_literals() {
    for bare in ["[NaN]", "[Infinity]", "[-Infinity]"] {
        assert!(
            serde_json::from_str::<serde_json::Value>(bare).is_err(),
            "{bare} parsed, so the strict-JSON guard is testing nothing"
        );
    }
}

/// The encoding has to survive write-then-read with the value's identity
/// intact, which is exactly what `null` would not do. NaN, `+inf` and `-inf`
/// must come back as themselves and not as each other.
#[test]
fn non_finite_floats_round_trip_through_the_recorded_spelling() {
    let values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.5,
        f64::MAX,
    ];

    let encoded: Vec<serde_json::Value> = values.iter().copied().map(record_float).collect();
    let text = serde_json::to_string(&encoded).expect("the encoding must serialise");
    assert!(
        !text.contains("NaN,") && !text.contains("[NaN") && !text.contains("Infinity,"),
        "the encoded form still carries a bare literal: {text}"
    );

    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("the encoded form must be strict JSON");
    let read_back: Vec<f64> = parsed.iter().map(recorded_float).collect();

    assert_eq!(read_back.len(), values.len());
    for (before, after) in values.iter().zip(&read_back) {
        if before.is_nan() {
            assert!(after.is_nan(), "NaN came back as {after}");
        } else {
            assert_eq!(
                before.to_bits(),
                after.to_bits(),
                "{before} came back as {after}"
            );
        }
    }

    // The three tokens are pairwise distinct on the wire, so nothing about the
    // encoding is lossy: a reader can tell which one it has.
    let tokens: Vec<&serde_json::Value> = encoded[..3].iter().collect();
    assert_ne!(tokens[0], tokens[1]);
    assert_ne!(tokens[1], tokens[2]);
    assert_ne!(tokens[0], tokens[2]);
}

/// The two records the repair touched still say what they measured. Quoting
/// the token must not have quietly turned an infinity into a string nobody
/// decodes, or into the `0.0` that sits next to it in the same row.
#[test]
fn the_repaired_captures_still_record_a_real_infinity_and_a_real_nan() {
    let radiance: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join("oracle-captures/foreign-radiance/oracle.json"))
            .expect("the radiance capture must be readable"),
    )
    .expect("the radiance capture must be strict JSON");

    // `float2rad`'s setcolr sweep: row 8 is the undefined-behaviour row, and
    // its first component is the +inf that makes it one.
    let row = &radiance["records"]["encode_setcolr"]["inputs"][8];
    let inf = recorded_float(&row[0]);
    assert!(
        inf.is_infinite() && inf.is_sign_positive(),
        "encode_setcolr input 8 must be +inf, got {inf}"
    );
    assert_eq!(recorded_float(&row[1]), 1.0);
    assert_eq!(recorded_float(&row[2]), 1.0);

    let uhdr: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join("oracle-captures/foreign-uhdr/oracle.json"))
            .expect("the uhdr capture must be readable"),
    )
    .expect("the uhdr capture must be strict JSON");

    // `log2(min) * (1.0f - gg)` is `-inf * 0` at both ends of the degenerate
    // metadata sweep, so one pixel per arm is NaN in all three bands. Which
    // pixel differs: the min arm lands on the last, the max arm on the first.
    let results = &uhdr["records"]["uhdr2scRGB_degenerate_metadata"]["results"];
    for (arm, pixel) in [("min_boost_zero", 15), ("max_boost_zero", 0)] {
        let scrgb = &results[arm]["scRGB"];
        assert_eq!(scrgb.as_array().expect("a 16-pixel row").len(), 16);
        for band in 0..3 {
            let v = recorded_float(&scrgb[pixel][band]);
            assert!(
                v.is_nan(),
                "{arm} pixel {pixel} band {band} must be NaN, got {v}"
            );
        }
    }
}
