//! Every Cargo feature has to say which CI jobs run it, and CI has to actually
//! run them (issues #772, #816).
//!
//! # The hole this closes
//!
//! `svg`, `jxl`, `jp2k`, `avif`, `packfile`, `serde` and `tracing` all gate
//! code behind `#[cfg(feature = ...)]`, so a job that does not name the feature
//! compiles those bodies **out** and runs zero assertions against them. The
//! first three each needed their own issue (#502, #500, #772) to notice the
//! same thing, and the last four had nobody watching at all. Measured on
//! `d21bfd1` by diffing `cargo test -- --list` against a default run:
//!
//! | feature | tests only it runs | in `ci.yml` before #816 |
//! |---|---|---|
//! | `jxl` | 30 | yes |
//! | `jp2k` | 24 | **no** (#772) |
//! | `svg` | 17 | yes |
//! | `object-store-sink` | 9 | yes |
//! | `packfile` | 8 | **no** |
//! | `serde` | 6 | **no** |
//! | `tracing` | 5 | **no** |
//! | `avif` | 10, all of them `ignore`d without it | **no** |
//!
//! `avif` is the one that shows why counting listed tests is not enough: the
//! feature adds no *new* tests, it un-`ignore`s ten that are already there, and
//! those ten are the entire oracle comparison (bit-exactness, lossy 4:4:4,
//! chroma siting, alpha, deep samples, the decode ceilings). A count of the
//! test list moves by zero for it.
//!
//! `packfile` is the one that shows the lint half is not hypothetical either.
//! The first `cargo clippy --all-targets --features packfile` ever run on this
//! tree was **red**, on a `collapsible_if` in `src/sink_packfile.rs` that no
//! job had ever compiled.
//!
//! # A cell has to run the tests, not merely name them
//!
//! The module title says CI "has to actually run them", and until #949 the
//! check did not ask that. `job_runs` accepted a cell's command as a prefix
//! followed by a space, so changing `- run: cargo test --features jp2k` to the
//! same line plus `--no-run` left all six tests here **green** while the Test
//! job compiled and ran nothing. A trailing `-- some::filter` selecting no test
//! does the same.
//!
//! The prefix rule is not simply wrong: the Check & Lint job's lines end
//! `-- -D warnings -W clippy::incompatible_msrv -W deprecated`, and a cell has
//! to match those. So the tail a cell allows is now part of the question:
//! nothing at all for a Test or MSRV cell, and lint configuration after a `--`
//! separator for a lint one, read as flag-and-name pairs so `-- some::filter`
//! is not one.
//!
//! # Why a table rather than a rule
//!
//! There is no honest rule that derives the right job set from the feature
//! name. `pdfium` cannot be in the test job because it needs a native library
//! at runtime; `s3` is an alias with no code of its own and gets a build cell
//! instead; `test-util` gates helpers rather than assertions. So each feature
//! carries an explicit row with its reason, and the guard's real question is
//! the one a rule cannot answer: **does `[features]` in `Cargo.toml` contain
//! exactly the names this table covers?** A new feature fails here until
//! somebody decides, in writing, which jobs it belongs in.
//!
//! `Cargo.toml`, `ci.yml` and the `Makefile` are pulled in with `include_str!`
//! at compile time rather than read at runtime, so those tests stay runnable
//! under Miri and off `tests/miri_fs_test_inventory.txt` (#712).
//!
//! `test_util_is_only_ever_gated_alongside_cfg_test` is the exception, and it
//! had to become one. Its claim is crate-wide ("every gate on the feature is
//! `cfg(any(test, feature = "test-util"))`") and it was read out of one file,
//! `src/sink.rs`, so a bare `#[cfg(feature = "test-util")] pub fn` planted in
//! `src/colour.rs` was invisible to it, and code behind such a gate is
//! compiled by no CI cell at all, which is the hole this row's `why` claims to
//! close (issue #949). A crate-wide claim needs a crate-wide walk, an
//! `include_str!` of every file would be a hand-written list of exactly the
//! kind that rots, so it walks `src/` at runtime, carries
//! `#[cfg_attr(miri, ignore)]` and has a line in the inventory.

/// The manifest, at compile time.
use std::collections::BTreeSet;

#[path = "common/mask.rs"]
mod mask;

const CARGO_TOML: &str = include_str!("../Cargo.toml");
/// The only CI workflow that gates a merge (issue #585), at compile time.
const CI_YML: &str = include_str!("../.github/workflows/ci.yml");

/// The local gate, which has to lint the same set as the hosted one.
const MAKEFILE: &str = include_str!("../Makefile");

/// Which jobs a feature belongs in, and why.
struct Coverage {
    /// The `Check & Lint` job runs `cargo clippy --all-targets --features F`.
    lint: bool,
    /// The `MSRV` job runs `cargo check --all-targets --features F`.
    msrv: bool,
    /// The `Test` job runs `cargo test --features F`.
    test: bool,
    /// Why the three above are what they are. Read on failure.
    why: &'static str,
}

const fn cov(lint: bool, msrv: bool, test: bool, why: &'static str) -> Coverage {
    Coverage {
        lint,
        msrv,
        test,
        why,
    }
}

/// Every feature this crate declares, and the CI cells it needs.
///
/// The MSRV column is not taste. A feature needs that cell when it pulls in at
/// least one crate that declares no `rust-version`, because that is exactly
/// the case the MSRV-aware resolver cannot see and `Cargo.lock` is not
/// committed, so a patch release is free to raise the real floor with nothing
/// to declare it. Measured with `cargo metadata` over the resolve graph:
/// `jxl` adds 16 such crates, `svg` 12, `avif` 9, `packfile` 5 and `jp2k` 1
/// (`openjpeg2-pure-rs` 0.1.1). `serde`, `tracing` and `object-store-sink`
/// add none.
const EXPECTED: &[(&str, Coverage)] = &[
    (
        "pdfium",
        cov(
            true,
            true,
            false,
            "linted and checked, but not tested: pdfium-render needs a native \
             PDFium library present at runtime, which the hosted runner has not \
             got",
        ),
    ),
    (
        "pdfium-static",
        cov(
            false,
            false,
            false,
            "`pdfium` plus `pdfium-render/static`, which builds PDFium from \
             source. No libviprs code is gated on it, so `pdfium`'s cells cover \
             everything this crate owns, and the build is far too heavy for a \
             per-push job",
        ),
    ),
    (
        "object-store-sink",
        cov(
            true,
            false,
            true,
            "gates src/sink_object_store.rs and its unit tests (issue #382); \
             adds no crate without a rust-version",
        ),
    ),
    (
        "s3",
        cov(
            false,
            false,
            false,
            "a deprecated alias for `object-store-sink` with no code of its \
             own. It gets `cargo build --features s3` instead, which is what a \
             broken alias would fail; asserted separately below",
        ),
    ),
    (
        "tracing",
        cov(
            true,
            false,
            true,
            "gates the span instrumentation and tests/tracing_tile_spans.rs, 5 \
             tests; adds one crate and it declares a rust-version",
        ),
    ),
    (
        "avif",
        cov(
            true,
            true,
            true,
            "gates the real decode bodies in src/avif.rs and un-ignores the 10 \
             oracle-comparison tests; adds 9 crates with no rust-version",
        ),
    ),
    (
        "svg",
        cov(
            true,
            true,
            true,
            "gates src/svg.rs and the xlink:href lockdown tests (issue #502), \
             17 tests; adds 12 crates with no rust-version",
        ),
    ),
    (
        "jxl",
        cov(
            true,
            true,
            true,
            "gates src/jxl.rs, 30 tests (issue #500); adds 16 crates with no \
             rust-version",
        ),
    ),
    (
        "packfile",
        cov(
            true,
            true,
            true,
            "gates src/sink_packfile.rs and 8 tests; adds 5 crates with no \
             rust-version",
        ),
    ),
    (
        "serde",
        cov(
            true,
            false,
            true,
            "gates the Serialize/Deserialize derives and all of \
             tests/serde_wire.rs, 6 tests; adds no crate at all",
        ),
    ),
    (
        "test-util",
        cov(
            false,
            false,
            false,
            "the only feature with no cell of its own, and the only one where \
             that is right. Every gate on it is `cfg(any(test, feature = \
             \"test-util\"))`, so `--all-targets` already compiles and lints \
             the code in the test target, and it moves the test list by zero, \
             so a test cell would run the same suite twice. Asserted by \
             `test_util_is_only_ever_gated_alongside_cfg_test`",
        ),
    ),
    (
        "jp2k",
        cov(
            true,
            true,
            true,
            "gates src/jp2k.rs, 24 tests (issue #772); adds \
             `openjpeg2-pure-rs` 0.1.1, which declares no rust-version",
        ),
    ),
];

/// The feature names declared in `Cargo.toml`'s `[features]` table, in
/// declaration order, with `default` dropped.
///
/// `default = []` is the empty set rather than a feature with code behind it,
/// and giving it a row would mean asserting cells for the jobs that already run
/// bare.
fn declared_features() -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in CARGO_TOML.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            inside = trimmed == "[features]";
            continue;
        }
        if !inside || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, _)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name != "default" && !name.is_empty() {
            out.push(name);
        }
    }
    out
}

/// The `- run:` command lines inside one two-space-indented job block of
/// `ci.yml`.
///
/// Scoped to the job rather than searched over the whole file, because "the
/// feature is named somewhere in CI" is the question that passes while the
/// feature is only linted and never run, which is half of what #816 was.
fn run_lines_of_job(job: &str) -> Vec<&'static str> {
    let header = format!("  {job}:");
    let mut out = Vec::new();
    let mut inside = false;
    for line in CI_YML.lines() {
        if line == header {
            inside = true;
            continue;
        }
        if inside
            && line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_end().ends_with(':')
        {
            break;
        }
        if inside && let Some(cmd) = line.trim().strip_prefix("- run: ") {
            out.push(cmd);
        }
    }
    out
}

/// What a `- run:` line is allowed to carry after the command a cell names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tail {
    /// Nothing at all. A `cargo test` cell has to be exactly the cell:
    /// `--no-run` compiles and runs zero tests and a trailing filter can
    /// select none, and either leaves this module's "CI has to actually run
    /// them" false while every assertion here stays green (issue #949).
    Nothing,
    /// Lint configuration after a `--` separator, which is how the Check &
    /// Lint job spells `-D warnings`. Pairs only: a flag then a lint name, so
    /// `-- some::filter` is not one.
    LintFlags,
}

/// Whether one `- run:` line satisfies a cell's command.
///
/// The command has to be the whole line, or the whole line up to a tail the
/// cell allows. Anything else before a `--` separator changes what cargo does:
/// `--no-run` compiles and runs nothing, `--lib` narrows the set, another
/// `--features` widens it, and each of those leaves the Test job green over a
/// cell that ran no assertion at all (issue #949).
fn cell_matches(line: &str, command: &str, tail: Tail) -> bool {
    let Some(rest) = line.strip_prefix(command) else {
        return false;
    };
    let rest = rest.trim();
    if rest.is_empty() {
        return true;
    }
    if tail == Tail::Nothing {
        return false;
    }
    let Some(flags) = rest.strip_prefix("--") else {
        return false;
    };
    // Pairs only, a lint flag then a lint name, so `-- some::filter` is not a
    // lint tail and neither is `-- --ignored`.
    let toks: Vec<&str> = flags.split_whitespace().collect();
    !toks.is_empty()
        && toks.len().is_multiple_of(2)
        && toks
            .chunks(2)
            .all(|p| matches!(p[0], "-D" | "-W" | "-A" | "-F") && !p[1].starts_with('-'))
}

/// Whether `job` runs `command`, allowing `tail` after it.
fn job_runs_with(job: &str, command: &str, tail: Tail) -> bool {
    run_lines_of_job(job)
        .iter()
        .any(|line| cell_matches(line, command, tail))
}

/// Whether `job` runs `command`, allowing only lint configuration after it.
///
/// The lint tail is what the Check & Lint job needs, since its lines end
/// `-- -D warnings ..`. The Test and MSRV cells go through
/// [`job_runs_exactly`], which allows nothing.
fn job_runs(job: &str, command: &str) -> bool {
    job_runs_with(job, command, Tail::LintFlags)
}

/// Whether `job` runs `command` and nothing else on the line.
fn job_runs_exactly(job: &str, command: &str) -> bool {
    job_runs_with(job, command, Tail::Nothing)
}

/// The table covers exactly the declared feature set.
///
/// This is the assertion that makes the rest of the file survive a new
/// feature. Without it, a feature added to `Cargo.toml` and to no job would
/// simply not be looked at, and every other test here would stay green while
/// the thing they exist to prevent happened again.
#[test]
fn every_declared_feature_has_a_row_saying_which_jobs_run_it() {
    let declared = declared_features();
    assert!(
        declared.len() >= 10,
        "the [features] parser found only {declared:?}, which cannot be right; \
         it has to fail loudly rather than vacuously pass"
    );
    assert!(
        declared.contains(&"jp2k") && declared.contains(&"pdfium"),
        "positive control: the parser must find the features that are \
         definitely there, got {declared:?}"
    );

    for name in &declared {
        assert!(
            EXPECTED.iter().any(|(n, _)| n == name),
            "Cargo.toml declares the feature `{name}` and this table says \
             nothing about it. Add a row to EXPECTED naming which of the \
             Check & Lint, MSRV and Test jobs must run it, and why. If it \
             gates any `#[cfg(feature = \"{name}\")]` code at all, the answer \
             is at least the lint job: without it nothing in CI type-checks \
             those bodies."
        );
    }
    for (name, _) in EXPECTED {
        assert!(
            declared.contains(name),
            "this table has a row for `{name}`, which Cargo.toml no longer \
             declares. Drop the row and the CI cells with it."
        );
    }
}

/// Every cell the table claims is actually in `ci.yml`, in the right job.
#[test]
fn ci_runs_every_cell_the_table_claims() {
    let mut missing: Vec<String> = Vec::new();
    for (name, c) in EXPECTED {
        if c.lint
            && !job_runs(
                "check",
                &format!("cargo clippy --all-targets --features {name}"),
            )
        {
            missing.push(format!(
                "the Check & Lint job does not run `cargo clippy --all-targets \
                 --features {name}` ({why})",
                why = c.why
            ));
        }
        if c.msrv
            && !job_runs_exactly(
                "msrv",
                &format!("cargo check --all-targets --features {name}"),
            )
        {
            missing.push(format!(
                "the MSRV job does not run `cargo check --all-targets \
                 --features {name}` ({why})",
                why = c.why
            ));
        }
        if c.test && !job_runs_exactly("test", &format!("cargo test --features {name}")) {
            missing.push(format!(
                "the Test job does not run `cargo test --features {name}` \
                 ({why})",
                why = c.why
            ));
        }
    }
    assert!(
        missing.is_empty(),
        "CI is missing cells:\n  {}",
        missing.join("\n  ")
    );
}

/// The three job blocks exist and carry commands, and a feature that is *not*
/// named really does come back missing.
///
/// The positive control matters more here than usual: `ci_runs_every_cell_the_table_claims`
/// reports success by finding nothing, and a `run_lines_of_job` that returned
/// an empty list for a mistyped job name, or a `job_runs` that answered `true`
/// for everything, would both make it pass while checking nothing.
#[test]
fn the_workflow_scanner_would_notice_a_missing_cell() {
    for job in ["check", "msrv", "test"] {
        assert!(
            run_lines_of_job(job).len() >= 2,
            "job `{job}` should have several `- run:` steps, found {:?}",
            run_lines_of_job(job)
        );
    }
    // A command that is there.
    assert!(job_runs("test", "cargo test --features jp2k"));
    // A command that is not.
    assert!(!job_runs(
        "test",
        "cargo test --features definitely-not-a-feature"
    ));
    // A command that is there, but in a different job. This is the half that
    // "named somewhere in CI" would get wrong: `pdfium` is linted and checked
    // and deliberately not tested.
    assert!(job_runs(
        "check",
        "cargo clippy --all-targets --features pdfium"
    ));
    assert!(!job_runs("test", "cargo test --features pdfium"));
    // And the job scanner really is scoped: the MSRV job's `cargo check` lines
    // must not leak into the lint job's view.
    assert!(job_runs(
        "msrv",
        "cargo check --all-targets --features jp2k"
    ));
    assert!(!job_runs(
        "check",
        "cargo check --all-targets --features jp2k"
    ));
}

/// A `- run:` line only satisfies a cell when it runs what the cell says.
///
/// `job_runs` accepts the cell's command as a **prefix followed by a space**,
/// which the Check & Lint job needs (its lines end `-- -D warnings ..`) and
/// which the Test job must not have: changing `cargo test --features jp2k` to
/// `cargo test --features jp2k --no-run` left all six tests here green while
/// the Test job compiled and ran nothing (issue #949).
#[test]
fn a_cell_is_not_satisfied_by_a_command_that_runs_nothing() {
    const TEST_CELL: &str = "cargo test --features jp2k";
    const LINT_CELL: &str = "cargo clippy --all-targets --features svg";
    let cases: [(&str, &str, Tail, bool); 10] = [
        (TEST_CELL, TEST_CELL, Tail::Nothing, true),
        // The four ways a Test cell can be present and run nothing.
        (
            "cargo test --features jp2k --no-run",
            TEST_CELL,
            Tail::Nothing,
            false,
        ),
        (
            "cargo test --features jp2k -- imageio::tests",
            TEST_CELL,
            Tail::Nothing,
            false,
        ),
        (
            "cargo test --features jp2k --lib",
            TEST_CELL,
            Tail::Nothing,
            false,
        ),
        (
            "cargo test --features jp2k -- --skip decode",
            TEST_CELL,
            Tail::Nothing,
            false,
        ),
        // A feature name is not a prefix of a longer one.
        (
            "cargo test --features jp2king",
            TEST_CELL,
            Tail::Nothing,
            false,
        ),
        // The lint job's real spelling, which has to keep working.
        (
            "cargo clippy --all-targets --features svg -- -D warnings \
             -W clippy::incompatible_msrv -W deprecated",
            LINT_CELL,
            Tail::LintFlags,
            true,
        ),
        (LINT_CELL, LINT_CELL, Tail::LintFlags, true),
        // and a lint cell that has grown something that is not a lint flag.
        (
            "cargo clippy --all-targets --features svg --no-deps -- -D warnings",
            LINT_CELL,
            Tail::LintFlags,
            false,
        ),
        (
            "cargo clippy --all-targets --features svg -- some::filter",
            LINT_CELL,
            Tail::LintFlags,
            false,
        ),
    ];
    let mut wrong = Vec::new();
    for (line, command, tail, want) in cases {
        let got = cell_matches(line, command, tail);
        if got != want {
            wrong.push(format!(
                "{line:?} against {command:?} with {tail:?}: got {got}, want {want}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cell-match rows are wrong:\n  {}",
        wrong.len(),
        cases.len(),
        wrong.join("\n  ")
    );
}

/// `s3` has no cells of its own, so the one thing that keeps the alias
/// resolving has to be asserted rather than assumed.
#[test]
fn the_deprecated_s3_alias_is_still_built() {
    assert!(
        job_runs("check", "cargo build --features s3"),
        "`s3` is an alias with no code, so the build step is the only thing \
         standing between it and a silent break"
    );
}

/// Every `#[cfg(feature = "test-util")]` gate under `src/`, as
/// `(files scanned, bare gates, gates seen)`.
fn test_util_gates() -> (Vec<String>, Vec<String>, usize) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut scanned = Vec::new();
    let mut bare = Vec::new();
    let mut seen = 0usize;
    for (rel, path) in mask::rs_files_under(&root) {
        let rel = format!("src/{rel}");
        scanned.push(rel.clone());
        let text = std::fs::read_to_string(&path).expect("read a source file");
        for (n, line) in text.lines().enumerate() {
            if !line.contains("feature = \"test-util\"") {
                continue;
            }
            seen += 1;
            if !line.contains("any(test,") && !line.contains("doc(cfg(") {
                bare.push(format!("{rel}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    (scanned, bare, seen)
}

/// `test-util`'s row is the only one claiming a feature needs no CI cell at
/// all, so the reason behind it is asserted rather than written down.
///
/// Every gate on the feature is `cfg(any(test, feature = "test-util"))`, which
/// means `cargo clippy --all-targets` already compiles that code in the test
/// target and a cell of its own would add nothing. The day someone writes a
/// bare `#[cfg(feature = "test-util")]`, that stops being true and this fails,
/// which is the moment the row needs revisiting.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn test_util_is_only_ever_gated_alongside_cfg_test() {
    let (scanned, bare, seen) = test_util_gates();
    assert!(
        scanned.len() > 30,
        "the scan reached {} files under src/, which cannot be right; the \
         invariant is crate-wide and a scan of one file cannot hold it \
         (issue #949). Files: {scanned:?}",
        scanned.len()
    );
    for anchor in ["src/sink.rs", "src/colour.rs"] {
        assert!(
            scanned.iter().any(|f| f == anchor),
            "the scan did not reach `{anchor}`, so a bare gate planted there \
             would be invisible and no CI cell would compile the code behind \
             it"
        );
    }
    assert!(
        seen >= 3,
        "positive control: the scan must find the gates that are there, found \
         {seen}"
    );
    assert!(
        bare.is_empty(),
        "`test-util` is now gating code that `cfg(test)` does not also reach, \
         so it needs its own lint cell in ci.yml and a row change in \
         EXPECTED:\n  {}",
        bare.join("\n  ")
    );
}

/// The `Makefile`'s `LINTED_FEATURES` is exactly the `lint: true` rows above.
///
/// #816 closed the hosted half of this hole and left the local half open, and
/// the local half is the one that matters more here: the handover says plainly
/// that the local gate is authoritative and GitHub Actions is not. So `main`
/// could be red under a feature, `make clippy` could be green, and both would
/// be behaving as documented.
///
/// It was not hypothetical. `main` was red under `packfile`
/// (`sink_packfile.rs:147`, `collapsible_if`) for an unknown stretch, and the
/// only reason anyone found it was someone verifying #816's measurement by
/// hand. A sweep of all seven non-default features at that commit found
/// `packfile` was the only red one, which is exactly the shape that gets
/// dismissed as a one-off (issue #844).
///
/// Asserted as set equality rather than containment, in both directions: a
/// feature CI lints that the `Makefile` skips is the original hole, and a
/// feature the `Makefile` lints that CI skips is a local green that means more
/// than a hosted one, which is its own kind of wrong.
#[test]
fn the_makefile_lints_exactly_the_features_ci_lints() {
    let declared: BTreeSet<&str> = EXPECTED
        .iter()
        .filter(|(_, c)| c.lint)
        .map(|(name, _)| *name)
        .collect();

    let line = MAKEFILE
        .lines()
        .find(|l| l.starts_with("LINTED_FEATURES"))
        .expect("the Makefile must declare LINTED_FEATURES; see issue #844");
    let listed: BTreeSet<&str> = line
        .split_once(":=")
        .expect("LINTED_FEATURES must be a := assignment")
        .1
        .split_whitespace()
        .collect();

    assert_eq!(
        listed,
        declared,
        "`make clippy` and CI's Check & Lint job must lint the same features. \
         Missing from the Makefile: {:?}. Extra in the Makefile: {:?}. Move both \
         lists in the same change (issue #844).",
        declared.difference(&listed).collect::<Vec<_>>(),
        listed.difference(&declared).collect::<Vec<_>>(),
    );
}
