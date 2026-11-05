//! The release-readiness claims for PMTiles, each one decidable (issue #993).
//!
//! F2.5 closes EPIC F by saying PMTiles is ready to ship as the default, and
//! "ready" is three separate claims: it is in every build rather than behind a
//! feature, the release notes describe what actually shipped, and there is a
//! publish path that can be rehearsed without uploading. Benchmarks are an
//! optional, on-demand tool and no release claim depends on them.
//!
//! Prose makes those claims; nothing checked them. This repository has already
//! been bitten twice by exactly that shape: `merge-gate.yml` claiming the crate
//! had no `unsafe` when it had ten (issue #897), and a CHANGELOG entry claiming
//! a variant had never been released when it shipped in v0.4.0 (issue #947).
//! Both are now checked, by `tests/unsafe_inventory.rs` and
//! `tests/changelog_release_claims.rs`. This is the same treatment for the
//! three above.
//!
//! Every file it reads comes in through `include_str!`, the way
//! `tests/ci_feature_coverage.rs` does it, so editing one forces a rebuild and
//! the assertions run under Miri along with everything else.

const LIB_RS: &str = include_str!("../src/lib.rs");
const CHANGELOG: &str = include_str!("../CHANGELOG.md");
const MIGRATION: &str = include_str!("../MIGRATION.md");
const PUBLISH_YML: &str = include_str!("../.github/workflows/publish.yml");

/// The PMTiles modules ship in a default build.
///
/// "PMTiles is the default storage" is false the moment one of these sits
/// behind a `#[cfg(feature = ...)]`, and a feature gate is one line to add and
/// invisible in a review that is looking at the sink instead. The check reads
/// the line before each `pub mod` rather than the whole file, because that is
/// where a module-level `cfg` has to be.
#[test]
fn the_pmtiles_modules_are_in_every_build() {
    let lines: Vec<&str> = LIB_RS.lines().collect();
    for module in ["pmtiles", "sink_pmtiles", "pyramid_reader"] {
        let declaration = format!("pub mod {module};");
        let at = lines
            .iter()
            .position(|line| line.trim() == declaration)
            .unwrap_or_else(|| panic!("src/lib.rs no longer declares `{declaration}`"));
        let previous = lines[..at]
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with("//"))
            .copied()
            .unwrap_or("");
        assert!(
            !previous.contains("cfg(feature"),
            "`{declaration}` is gated by `{previous}`, so PMTiles is not in a default build"
        );
    }
}

/// The release notes describe the storage flip that shipped.
///
/// F2.1 (#992) wrote both of these; this holds them there. A release that
/// flipped the default and said nothing about it in either file is the
/// upgrade that breaks a consumer with no way to find out why.
#[test]
fn the_release_notes_describe_the_storage_flip() {
    for (name, text) in [("CHANGELOG.md", CHANGELOG), ("MIGRATION.md", MIGRATION)] {
        assert!(
            text.contains("PyramidStorage"),
            "{name} never names `PyramidStorage`, the type the default flip is made of"
        );
        assert!(
            text.contains("PMTiles"),
            "{name} never mentions PMTiles at all"
        );
    }
    // The specific claim a consumer acts on: which variant `default()` is.
    assert!(
        MIGRATION.contains("PyramidStorage::default()"),
        "MIGRATION.md should say what `PyramidStorage::default()` answers, since that is the \
         flip itself"
    );
    assert!(
        MIGRATION.contains("PyramidStorage::Directory"),
        "MIGRATION.md should name the variant that keeps the old behaviour"
    );
}

/// A publish can be rehearsed without uploading.
///
/// `cargo publish` is irreversible: a version yanked is still a version. The
/// workflow has a `dry_run` input for exactly that, and the point of checking
/// it here is that the input and the command that honours it are in two
/// different places in the file and either can be removed alone.
#[test]
fn a_publish_can_be_rehearsed_before_it_is_made() {
    assert!(
        PUBLISH_YML.contains("dry_run:"),
        "publish.yml no longer offers a dry-run input"
    );
    assert!(
        PUBLISH_YML.contains("cargo publish --dry-run --locked"),
        "publish.yml offers a dry-run input that no step honours"
    );
    assert!(
        PUBLISH_YML.contains("cargo publish --locked"),
        "publish.yml no longer has a real upload step"
    );
    // The real upload is the one that must stay gated.
    assert!(
        PUBLISH_YML.contains("Refuse a real upload unless the published contract is true"),
        "the gate in front of the real upload is gone"
    );
}
