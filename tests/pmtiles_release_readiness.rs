//! The release-readiness claims for PMTiles, each one decidable (issue #993).
//!
//! F2.5 closes EPIC F by saying PMTiles is ready to ship as the default, and
//! "ready" is four separate claims: it is in every build rather than behind a
//! feature, the release notes describe what actually shipped, there is a
//! publish path that can be rehearsed without uploading, and the benchmark
//! procedure is written down and runnable.
//!
//! Prose makes those claims; nothing checked them. This repository has already
//! been bitten twice by exactly that shape: `merge-gate.yml` claiming the crate
//! had no `unsafe` when it had ten (issue #897), and a CHANGELOG entry claiming
//! a variant had never been released when it shipped in v0.4.0 (issue #947).
//! Both are now checked, by `tests/unsafe_inventory.rs` and
//! `tests/changelog_release_claims.rs`. This is the same treatment for the
//! four above.
//!
//! Every file it reads comes in through `include_str!`, the way
//! `tests/ci_feature_coverage.rs` does it, so editing one forces a rebuild and
//! the assertions run under Miri along with everything else.

const LIB_RS: &str = include_str!("../src/lib.rs");
const CHANGELOG: &str = include_str!("../CHANGELOG.md");
const MIGRATION: &str = include_str!("../MIGRATION.md");
const PUBLISH_YML: &str = include_str!("../.github/workflows/publish.yml");
const BENCHMARK_DOC: &str = include_str!("../docs/pmtiles-benchmarks.md");

#[path = "common/pmtiles_bench.rs"]
mod bench;

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
        PUBLISH_YML.contains("cargo publish --dry-run"),
        "publish.yml offers a dry-run input that no step honours"
    );
    assert!(
        PUBLISH_YML.lines().any(|line| {
            let line = line.trim();
            line.starts_with("cargo publish") && !line.contains("--dry-run")
        }),
        "publish.yml no longer has a real upload step"
    );
    // The real upload is the one that must stay gated.
    assert!(
        PUBLISH_YML.contains("Refuse a real upload unless the published contract is true"),
        "the gate in front of the real upload is gone"
    );
}

/// The publish commands can run on the checkout the workflow makes.
///
/// `--locked` only means something where `Cargo.lock` is tracked. This
/// repository gitignores it, so a `--locked` publish dies before it packages
/// anything, with "cannot create the lock file because --locked was passed",
/// on the first clean checkout that ever runs it. Both branches of the publish
/// step carried the flag until #993 rehearsed the dry run and hit exactly that.
///
/// Either half can move. Start tracking the lock file and `--locked` becomes
/// correct again, which is why this asserts the **pair** rather than the flag:
/// a lock file with no `--locked` is a missed opportunity, and `--locked` with
/// no lock file is a workflow that cannot run.
///
/// # Reading the index rather than the working directory
///
/// `Cargo.lock` exists on disk after any build, so its presence says nothing.
/// What matters is whether git tracks it, and `git ls-files --error-unmatch`
/// answers exactly that: 0 for tracked, 1 for not. Anything else is a broken
/// invocation and must not read as "not tracked", so there is a positive
/// control on `Cargo.toml` first. The local mirror stages this tree's index
/// and refs without its history, which `git ls-files` is fine with and
/// `git grep <tag>` is not, so this needs no history-shaped skip the way
/// `tests/changelog_release_claims.rs` does.
#[test]
#[cfg_attr(miri, ignore)]
fn the_publish_commands_can_run_on_the_checkout_they_get() {
    let tracked = |path: &str| -> Option<bool> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(["ls-files", "--error-unmatch", "--", path])
            .output()
            .ok()?;
        match out.status.code() {
            Some(0) => Some(true),
            Some(1) => Some(false),
            _ => None,
        }
    };

    let Some(control) = tracked("Cargo.toml") else {
        eprintln!("skipping: git could not be asked what this tree tracks");
        return;
    };
    assert!(
        control,
        "positive control: git says it does not track Cargo.toml, so it is not \
         reading this repository's index and its answer about Cargo.lock means \
         nothing"
    );

    let lock_is_tracked = tracked("Cargo.lock").expect("git answered for Cargo.toml a line ago");
    let uses_locked = PUBLISH_YML
        .lines()
        .any(|line| line.trim().starts_with("cargo publish") && line.contains("--locked"));

    assert_eq!(
        lock_is_tracked, uses_locked,
        "Cargo.lock tracked: {lock_is_tracked}, publish.yml passes --locked: {uses_locked}. \
         A --locked publish with no tracked lock file cannot run at all, and a tracked lock \
         file without --locked publishes from a resolution nobody pinned."
    );
}

/// The benchmark procedure is written down, and it names commands this
/// repository can actually run.
///
/// A benchmark nobody else can reproduce is a screenshot. Each fenced command
/// the doc gives as a `cargo test --test <name>` has to name a test file that
/// exists, so a renamed test file fails here rather than in six months when
/// somebody tries to rerun the numbers.
#[test]
#[cfg_attr(miri, ignore)]
fn the_benchmark_doc_names_tests_that_exist() {
    let mut named = 0;
    for line in BENCHMARK_DOC.lines() {
        let Some(at) = line.find("--test ") else {
            continue;
        };
        let name = line[at + "--test ".len()..]
            .split_whitespace()
            .next()
            .expect("a test name follows --test")
            // The doc names some of these inside inline code spans, so the
            // closing backtick and any sentence punctuation ride along.
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join(format!("{name}.rs"));
        assert!(
            path.is_file(),
            "docs/pmtiles-benchmarks.md runs `--test {name}`, and tests/{name}.rs does not exist"
        );
        named += 1;
    }
    assert!(
        named >= 3,
        "the benchmark doc should name the benchmark, the bounded-memory and the index-only \
         suites; it names {named}"
    );
}

/// The doc's column table is the harness's field list.
///
/// The exported JSON is the hand-off to libviprs.org (issue #62), so the table
/// that tells a reader what each column means is part of the contract rather
/// than decoration. A field added to the harness and not to the table, or
/// renamed in one and not the other, fails here.
#[test]
fn the_benchmark_doc_documents_every_exported_column() {
    let documented: Vec<&str> = BENCHMARK_DOC
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("| `")?;
            rest.split('`').next()
        })
        .collect();

    for field in bench::FIELDS {
        assert!(
            documented.contains(&field),
            "docs/pmtiles-benchmarks.md does not document the `{field}` column; it documents \
             {documented:?}"
        );
    }
    for column in &documented {
        assert!(
            bench::FIELDS.contains(column),
            "docs/pmtiles-benchmarks.md documents a `{column}` column the harness does not emit"
        );
    }
}

/// The doc says which file the numbers are handed to, and in what shape.
#[test]
fn the_benchmark_doc_names_the_hand_off() {
    assert!(
        BENCHMARK_DOC.contains("scalability_results.json"),
        "the doc should name the file whose shape the export matches"
    );
    assert!(
        BENCHMARK_DOC.contains("LIBVIPRS_BENCH_JSON"),
        "the doc should say how to choose where the export lands"
    );
    assert!(
        BENCHMARK_DOC.contains("LIBVIPRS_BENCH_PROFILE"),
        "the doc should say how to select the opt-in large profile"
    );
}
