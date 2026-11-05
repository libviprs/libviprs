//! #1471: artifact bytes differ between x86_64 and arm64 for `general-affine-chain-001` and
//! `nonuniform-scale-rotated-001`.
//!
//! The drift check (#1475) localised it. The composed matrices are identical on both architectures
//! and so are libm's `sin`, `cos`, `atan2`, `tan`, `exp` and `sqrt` (200k inputs each, glibc 2.36 in
//! both containers). Only `hypot` differs, and the only `hypot` in the path is inside PVec's
//! `Curve::transform` (`Vector2::length`, `Point2::distance`, direction normalisation), which
//! libviprs reaches through the public API and does not reimplement. `pvec-numeric` has a
//! deterministic `atan` and `atan2` and no `hypot`, so libviprs cannot make those bits reproducible
//! without copying PVec's maths. The decision is recorded in
//! `docs/adr/pvec-artifact-reproducibility.md`: artifacts are reproducible per architecture, the
//! digests are keyed per architecture, and the set of fixtures that differ is pinned here so it
//! cannot grow silently.
//!
//! These tests read files only, so they run on every architecture and under no feature flag.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn adr() -> String {
    std::fs::read_to_string(root().join("docs/adr/pvec-artifact-reproducibility.md"))
        .expect("docs/adr/pvec-artifact-reproducibility.md is the record of the #1471 decision")
}

/// The fixtures the ADR lists under its "Fixtures that differ" heading, one `- \`id\`` per line.
fn listed() -> BTreeSet<String> {
    let text = adr();
    let body = text
        .split("## Fixtures that differ")
        .nth(1)
        .expect("the ADR has a `## Fixtures that differ` section");
    body.split("\n## ")
        .next()
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("- `")?.split('`').next())
        .map(str::to_owned)
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1471_the_adr_says_artifacts_are_reproducible_per_architecture_only() {
    let text = adr();
    for needle in [
        "#1471",
        "per architecture",
        "hypot",
        "pvec-numeric",
        "libviprs/pvec#",
        "general-affine-chain-001",
        "nonuniform-scale-rotated-001",
    ] {
        assert!(text.contains(needle), "the ADR does not mention {needle}");
    }
    assert!(
        !text.contains('\u{2014}'),
        "the ADR keeps to the house style: no em dashes"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1471_the_architecture_overrides_are_exactly_the_fixtures_the_adr_lists() {
    let listed = listed();
    assert_eq!(
        listed.len(),
        2,
        "the ADR lists the two fixtures of #1471, got {listed:?}"
    );
    let gate = |name: &str| -> BTreeSet<String> {
        std::fs::read_to_string(root().join("tests/fixtures/pvec-gate").join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .lines()
            .filter_map(|l| l.split_whitespace().next())
            .map(str::to_owned)
            .collect()
    };
    // Since the pin moved to afd2fe6 (pvec#261, `detmath::hypot`) the general affine chain's artifact
    // bytes agree across architectures. Its drift digest still differs, so it stays in the other two.
    let artifact_rows: BTreeSet<String> = ["nonuniform-scale-rotated-001".to_owned()].into();
    assert_eq!(gate("determinism.aarch64.baseline"), artifact_rows);
    assert_eq!(gate("drift.aarch64.baseline"), listed);
    let dump =
        std::fs::read_to_string(root().join("tests/fixtures/pvec-drift/accepted.aarch64.dump"))
            .expect("accepted.aarch64.dump");
    let sections: BTreeSet<String> = dump
        .lines()
        .filter_map(|l| l.strip_prefix("== "))
        .map(str::to_owned)
        .collect();
    assert_eq!(sections, listed);
}

/// A libm `hypot` in libviprs's own PVec code would put the architecture into a stored number
/// that libviprs itself computed. The grid spacing in `flatten.rs` did, and takes its unit vectors
/// from basic arithmetic now. (`drift.rs` compares and stores nothing, so it is outside the scan.)
#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1471_no_geometry_libviprs_computes_goes_through_libm_hypot() {
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(root().join("src/pvec")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("drift.rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            for (n, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap();
                if code.contains(".hypot(") {
                    offenders.push(format!("{}:{}", path.display(), n + 1));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "libm hypot is architecture-dependent in the last bit: {offenders:?}"
    );
}

/// #1513: the digest rows are Linux rows. On a Mac the two bit-exact tests are ignored, and the
/// attribute says why, so the skip is a visible decision and not a quiet cfg.
#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1513_the_bit_exact_digest_tests_are_gated_to_linux() {
    for (file, name) in [
        (
            "tests/pvec_drift.rs",
            "n_1471_the_digest_of_every_converted_fixture_matches_the_committed_one",
        ),
        (
            "tests/pvec_gate_determinism.rs",
            "n2_8_1060_the_digest_of_every_conversion_matches_the_checked_in_baseline",
        ),
    ] {
        let text = std::fs::read_to_string(root().join(file)).unwrap();
        let before = text
            .split(&format!("fn {name}("))
            .next()
            .unwrap_or_else(|| panic!("{file} has no {name}"));
        let attrs: Vec<&str> = before
            .lines()
            .rev()
            .take_while(|l| l.starts_with('#'))
            .collect();
        assert!(
            attrs.iter().any(|l| l.contains(
                "cfg_attr(not(target_os = \"linux\"), ignore = \"bit-exact Linux rows, #1513\")"
            )),
            "{file}: {name} must be ignored off Linux with the reason `bit-exact Linux rows, #1513`, got {attrs:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1513_the_other_pvec_gates_are_not_gated_by_os() {
    // Only the two digest tests are bit-exact. The policy-tolerance checks keep running on a Mac.
    for file in ["tests/pvec_drift.rs", "tests/pvec_gate_determinism.rs"] {
        let text = std::fs::read_to_string(root().join(file)).unwrap();
        assert_eq!(
            text.matches("not(target_os").count(),
            1,
            "{file}: exactly one test is gated by OS"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files
fn n_1513_the_adr_records_what_a_mac_does() {
    let text = adr();
    let body = text
        .split("## macOS")
        .nth(1)
        .expect("the ADR has a `## macOS` section");
    let body = body.split("\n## ").next().unwrap();
    for needle in [
        "#1513",
        "Linux only",
        "aarch64-apple-darwin",
        "ulp",
        "elliptical-arc-001",
        "nonuniform-scale-rotated-001",
        "general-affine-chain-001",
        "libm",
    ] {
        assert!(
            body.contains(needle),
            "the macOS section does not mention {needle}"
        );
    }
}
