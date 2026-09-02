//! `make ci` has to *be* the CI job list, not a second copy of it (issue #982).
//!
//! # The hole this closes
//!
//! `make ci` is what README.md and CONTRIBUTING.md tell people to run before
//! they push, and until #982 it read `ci: fmt clippy test doc miri loom`: six
//! hand-written targets standing in for the eight jobs the two workflows
//! actually run. Nothing compared the two lists, so they drifted, and by the
//! time anyone did compare them the drift was in six separate places at once:
//!
//! * `MSRV (1.97)` was not run at all, so neither were its seven `cargo check`
//!   cells nor the guard that holds the four written-out MSRV claims together;
//! * `Integration Tests (libviprs-tests)` was not run at all, which is the one
//!   job that compiles the ported cells against this crate's API;
//! * `pdfium-render source audit (#149)` was not run at all;
//! * `make test` ran one of the `Test` job's nine `cargo test` cells;
//! * `make clippy` skipped the `cargo build --features s3` cell;
//! * `make loom` ran one of the `Loom` job's two invocations, missing
//!   `loom_checkpoint_dedupe`.
//!
//! Every one of those is the same failure: run the documented gate, see green,
//! push, get a red run. A bigger `ci:` line would have fixed the six and left
//! the seventh, eighth and ninth to be found the same way.
//!
//! # Why this checks a shape rather than a list
//!
//! So the fix was not a longer list. `tools/local-ci.py` reads the workflow
//! files and runs the steps that are in them, which is the property a copy can
//! never have, and `make ci` now delegates to it. What can still go wrong is
//! somebody putting a command list back into the recipe, adding a gate
//! workflow the recipe never names, or a job growing a job-level `if:` that
//! the runner refuses to evaluate and nothing else picks up. Those three are
//! what this file asserts.
//!
//! The `if:` one is worth spelling out, because a skipped check reading as a
//! passing check is a trap this repository has already written about at length
//! in `merge-gate.yml`'s own `miri` comment. The runner reports such a job
//! HELD and does not run it. That is honest, and on its own it is also a hole,
//! so every held job needs a `make` target the `ci` recipe invokes. Today that
//! is Miri, covered by `make miri` on this machine's pinned nightly.
//!
//! `Makefile` and both workflow files come in through `include_str!` at
//! compile time, the same way `tests/ci_feature_coverage.rs` reads them, so
//! most of this file runs under Miri. The one test that lists a directory
//! cannot, and carries the annotation and an inventory line.

use std::collections::BTreeSet;
use std::path::Path;

const MAKEFILE: &str = include_str!("../Makefile");
const CI_YML: &str = include_str!("../.github/workflows/ci.yml");
const MERGE_GATE_YML: &str = include_str!("../.github/workflows/merge-gate.yml");

/// The workflow files whose jobs decide whether a change is good.
const GATE_WORKFLOWS: &[(&str, &str)] = &[("ci.yml", CI_YML), ("merge-gate.yml", MERGE_GATE_YML)];

/// Workflow files that are deliberately not part of the gate, and why.
///
/// A new file in `.github/workflows/` fails
/// `the_workflow_directory_holds_only_files_this_guard_has_classified` until
/// somebody decides, in writing, which of these two lists it belongs in.
const NOT_A_GATE: &[(&str, &str)] = &[(
    "publish.yml",
    "runs on a release tag and publishes the crate. It reacts to a decision \
     that has already been made rather than making one, so a developer has \
     nothing to run locally for it.",
)];

/// A job a hosted run holds behind a job-level `if:`, and what covers it here.
///
/// The columns are the workflow file, the job name, the `make` target the `ci`
/// recipe has to invoke instead, and the reason.
const HELD: &[(&str, &str, &str, &str)] = &[(
    "merge-gate.yml",
    "Miri",
    "miri",
    "held at the release boundary because it still does not pass on a whole \
     suite invocation (#675, #652). `make miri` runs the workflow's filtered \
     command on this machine's pinned nightly, and \
     tests/miri_invocation_parity.rs holds the two invocations together.",
)];

/// The recipe lines of the `Makefile`'s `ci` target, without their tabs.
///
/// A `make` recipe is every line after the target's `:` line that starts with
/// a tab, so this needs no parser beyond that rule.
fn ci_recipe() -> Vec<&'static str> {
    let mut lines = MAKEFILE.lines().skip_while(|l| !l.starts_with("ci:"));
    let header = lines
        .next()
        .expect("the Makefile must declare a `ci` target; see issue #982");
    assert_eq!(
        header.trim(),
        "ci:",
        "`ci` must have no prerequisites. Prerequisites are how the old \
         hand-written job list was spelled, and re-adding one is re-adding the \
         second copy this guard exists to prevent (issue #982)."
    );
    lines
        .take_while(|l| l.starts_with('\t'))
        .map(|l| l.trim_start_matches('\t'))
        .filter(|l| !l.is_empty())
        .collect()
}

/// Every `make` target the `Makefile` declares.
fn declared_targets() -> BTreeSet<&'static str> {
    MAKEFILE
        .lines()
        .filter(|l| !l.starts_with([' ', '\t', '#']))
        .filter_map(|l| l.split_once(':'))
        .filter(|(_, rest)| !rest.starts_with('='))
        .map(|(name, _)| name.trim())
        .filter(|n| !n.is_empty() && !n.starts_with('.'))
        .collect()
}

/// Every job in `yml` that carries a job-level `if:`, by name.
///
/// The indentation is the whole parser and it is enough: a job key sits at two
/// spaces, a job's own keys at four, and a step's keys at eight, because steps
/// are a list whose items open with `      - `. So a four-space `if:` is a job
/// condition and an eight-space one is a step condition, and nothing else in
/// these files is at four spaces called `if`.
fn jobs_with_a_condition(yml: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut name: Option<&str> = None;
    let mut has_if = false;
    let mut in_jobs = false;
    let flush = |name: Option<&str>, has_if: bool, out: &mut BTreeSet<String>| {
        if let (Some(n), true) = (name, has_if) {
            out.insert(n.to_owned());
        }
    };
    for line in yml.lines() {
        if line == "jobs:" {
            in_jobs = true;
            continue;
        }
        if !in_jobs {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if !line.trim().is_empty() && indent == 0 {
            break;
        }
        if indent == 2 && line.trim_end().ends_with(':') && !line.trim_start().starts_with('#') {
            flush(name, has_if, &mut out);
            name = None;
            has_if = false;
            continue;
        }
        if indent == 4 {
            if let Some(rest) = line.trim_start().strip_prefix("name:") {
                if name.is_none() {
                    name = Some(rest.trim());
                }
            } else if line.trim_start().starts_with("if:") {
                has_if = true;
            }
        }
    }
    flush(name, has_if, &mut out);
    out
}

/// The `ci` recipe runs `tools/local-ci.py` over every gate workflow.
///
/// Reading the invocations rather than the whole recipe is deliberate: what
/// matters is which workflow files reach the runner, and the runner's default
/// workflow is `ci.yml`, so an invocation with no `--workflow` names that one.
#[test]
fn the_ci_recipe_runs_the_runner_over_every_gate_workflow() {
    let mut covered = BTreeSet::new();
    for line in ci_recipe() {
        let Some(args) = line.strip_prefix("tools/local-ci.py") else {
            continue;
        };
        let mut words = args.split_whitespace();
        let mut workflow = "ci.yml";
        while let Some(w) = words.next() {
            if w == "--workflow" {
                workflow = words.next().expect("--workflow takes a file name");
            }
        }
        covered.insert(workflow.to_owned());
    }

    let expected: BTreeSet<String> = GATE_WORKFLOWS
        .iter()
        .map(|(f, _)| (*f).to_owned())
        .collect();
    assert_eq!(
        covered,
        expected,
        "`make ci` has to hand every gate workflow to tools/local-ci.py, which \
         is the only thing in this repository that reads the job list rather \
         than repeating it. Not run by the recipe: {:?}. Run but not a gate \
         workflow this guard knows about: {:?}. See issue #982.",
        expected.difference(&covered).collect::<Vec<_>>(),
        covered.difference(&expected).collect::<Vec<_>>(),
    );
}

/// The `ci` recipe carries no command list of its own.
///
/// This is the regression itself rather than a proxy for it. Every one of the
/// six drifts in #982 started as somebody writing a cargo invocation into the
/// `Makefile` because it was the obvious place, and then the workflow moving
/// underneath it.
#[test]
fn the_ci_recipe_names_no_build_command_of_its_own() {
    let offenders: Vec<&str> = ci_recipe()
        .into_iter()
        .filter(|l| {
            let l = l.trim_start_matches('@');
            l.starts_with("cargo") || l.contains(" cargo ") || l.starts_with("RUSTFLAGS")
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "the `ci` recipe must not spell out build commands. The workflow files \
         are the list and tools/local-ci.py reads them; a command written here \
         is a second copy, and every second copy this repository has had went \
         stale (issue #982). Offending lines: {offenders:?}"
    );
}

/// Every job the runner holds back has a `make` target the `ci` recipe runs.
///
/// Set equality in both directions. A new `if:` on a job fails here until
/// somebody adds a row saying what covers it, and a row for a job that no
/// longer has an `if:` fails too, because that one is now running under the
/// runner and the `make` target beside it is a second invocation nobody meant.
#[test]
fn every_held_job_is_covered_by_a_make_target_the_recipe_runs() {
    let mut found = BTreeSet::new();
    for (file, text) in GATE_WORKFLOWS {
        for job in jobs_with_a_condition(text) {
            found.insert(format!("{file}:{job}"));
        }
    }
    let listed: BTreeSet<String> = HELD
        .iter()
        .map(|(file, job, _, _)| format!("{file}:{job}"))
        .collect();
    assert_eq!(
        found,
        listed,
        "a job-level `if:` is a condition tools/local-ci.py refuses to \
         evaluate, so such a job does NOT run locally and needs something else \
         covering it. Held by a workflow with no row in HELD: {:?}. Has a row \
         but no longer carries an `if:`: {:?}. See issue #982.",
        found.difference(&listed).collect::<Vec<_>>(),
        listed.difference(&found).collect::<Vec<_>>(),
    );

    let recipe = ci_recipe();
    let targets = declared_targets();
    for (file, job, target, why) in HELD {
        assert!(
            targets.contains(target),
            "HELD says `make {target}` covers {job} from {file}, and the \
             Makefile declares no such target. Reason on the row: {why}"
        );
        let call = format!("$(MAKE) {target}");
        assert!(
            recipe.iter().any(|l| l.contains(&call)),
            "HELD says `make {target}` covers {job} from {file}, so the `ci` \
             recipe has to run it: no `{call}` line in {recipe:?}. Reason on \
             the row: {why}"
        );
    }
}

/// The runner still provisions the tree from git rather than bind-mounting it.
///
/// This is a tripwire and not a proof, and it is worth saying which. The proof
/// that the git-provisioned mode sees the two bug classes and the bind-mounted
/// one does not is the falsification recorded on #982: a fixture committed
/// under one case and read under another, and a fixture that was never
/// committed at all, both of which the bind mount resolved and reported PASS
/// on. What this catches is the cheap way to lose that again, which is
/// somebody flipping the default back because a rebuild felt slow.
#[test]
fn the_runner_provisions_from_git_by_default() {
    const RUNNER: &str = include_str!("../tools/local-ci.py");
    assert!(
        RUNNER.contains(r#"mode = "worktree" if a.worktree else "git""#),
        "tools/local-ci.py must default to checking the tree out of git and \
         take --worktree to bind-mount it instead. A Docker Desktop bind mount \
         off an APFS host is case-insensitive and carries untracked files, so \
         with the default the other way the container sees a tree no runner \
         can have (#977, #979, #982)."
    );
    assert!(
        RUNNER.contains("git clone --quiet --shared --no-checkout"),
        "the git-provisioned mode clones the mounted object store rather than \
         copying it, which is what keeps it at about a second on a \
         gigabyte-scale history (issue #982)."
    );
}

/// Every file in `.github/workflows/` is one this guard has classified.
///
/// The two lists above are only worth having if nothing can appear outside
/// them. A new gate workflow that no `make` target names would reproduce #982
/// exactly, one job at a time, and the `ci` recipe would still look complete.
#[test]
#[cfg_attr(miri, ignore)] // reads a directory, which Miri's isolation refuses (#712)
fn the_workflow_directory_holds_only_files_this_guard_has_classified() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut found = BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("failed to read .github/workflows") {
        let name = entry
            .expect("failed to read a workflow directory entry")
            .file_name();
        let name = name.to_string_lossy().into_owned();
        if name.ends_with(".yml") || name.ends_with(".yaml") {
            found.insert(name);
        }
    }
    assert!(
        !found.is_empty(),
        "no workflow files under {}, so this guard has nothing to check",
        dir.display()
    );

    let classified: BTreeSet<String> = GATE_WORKFLOWS
        .iter()
        .map(|(f, _)| (*f).to_owned())
        .chain(NOT_A_GATE.iter().map(|(f, _)| (*f).to_owned()))
        .collect();
    assert_eq!(
        found,
        classified,
        "every workflow file is either a gate, and then `make ci` has to run \
         it, or explicitly not one, and then it needs a reason on the \
         NOT_A_GATE row. Unclassified: {:?}. Classified but not present: {:?}. \
         See issue #982.",
        found.difference(&classified).collect::<Vec<_>>(),
        classified.difference(&found).collect::<Vec<_>>(),
    );
}
