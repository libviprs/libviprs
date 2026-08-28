//! Keeps the two places that build the documentation denying the same
//! intra-doc-link lints, and keeps the docs job's own name true (issue #697).
//!
//! # Why this exists
//!
//! `rustdoc::private_intra_doc_links` is warn-by-default. The docs job and the
//! `doc` target in the `Makefile` both denied `rustdoc::broken_intra_doc_links`
//! and nothing else, so a public doc comment could point at a `pub(crate)` item,
//! rustdoc would drop the link and render it as inert bracketed text on
//! docs.rs, and both gates stayed green while publishing a dead pointer. It had
//! happened 33 times across 13 files by the time anyone counted.
//!
//! Turning the lint on is only half the fix. The other half is that it stays on
//! in *both* invocations. A local mirror that denies less than the hosted job
//! reports green on a machine while the thing it claims to mirror reports red,
//! and nobody reads the flags because they live two files apart. That is exactly
//! the drift `tests/miri_invocation_parity.rs` was written for after the Miri
//! invocations parted, so this guards the doc invocations the same way.
//!
//! # What it asserts
//!
//! * every lint in [`REQUIRED_DENIES`] is denied by both invocations;
//! * the two deny *sets* are equal, so tightening one file alone fails here
//!   rather than silently un-mirroring the gate;
//! * the two `cargo doc` argument lists are equal, so the mirror keeps building
//!   the same surface (`--all-features` is load-bearing: without it the gated
//!   modules are not in the graph and their links read as broken, issue #146);
//! * the docs job's `name:` names every lint it denies, because a job called
//!   "deny broken intra-doc links" that also denies private ones is a claim with
//!   nothing behind it.
//!
//! # Why it parses instead of grepping
//!
//! A grep for the flag string is satisfied by the flag appearing anywhere in
//! either file, including in a comment. This pulls the assignment out of the
//! `docs` job's merged `env:` scopes and out of the `doc` recipe, then compares
//! the parsed sets, so a deny that is present but not reaching `cargo doc` fails
//! the same as a missing one.
//!
//! # Why it runs under Miri
//!
//! Both files arrive through `include_str!` at compile time, so there is no
//! filesystem access to isolate and no `#[cfg_attr(miri, ignore)]` is needed
//! (the same reasoning as `tests/miri_invocation_parity.rs`). It also means
//! editing either file forces a rebuild of this test.

use std::collections::BTreeSet;

/// The hosted job.
const WORKFLOW: &str = include_str!("../.github/workflows/ci.yml");
/// The local mirror of it.
const MAKEFILE: &str = include_str!("../Makefile");

/// Lints both invocations have to deny. `broken_intra_doc_links` was already
/// there; `private_intra_doc_links` is what issue #697 adds.
const REQUIRED_DENIES: [&str; 2] = [
    "rustdoc::broken_intra_doc_links",
    "rustdoc::private_intra_doc_links",
];

/// One `cargo doc` invocation, however it was spelled.
#[derive(Debug, PartialEq, Eq)]
struct DocInvocation {
    /// Every `-D <lint>` in `RUSTDOCFLAGS`, as a set so ordering is free.
    denies: BTreeSet<String>,
    /// The arguments passed to `cargo doc`, in order.
    args: Vec<String>,
}

/// Pull the `docs` job out of `.github/workflows/ci.yml`.
///
/// `RUSTDOCFLAGS` is read from the workflow-level `env:` and the job-level
/// `env:` with the job winning, which is the precedence GitHub applies. The
/// `cargo doc` line comes from the job's own `run:` steps.
fn workflow_invocation() -> DocInvocation {
    let mut workflow_env: Option<String> = None;
    let mut job_env: Option<String> = None;
    let mut args: Option<Vec<String>> = None;

    // Workflow-level `env:` is the only block at column 0 named `env:`.
    let mut in_workflow_env = false;
    for line in WORKFLOW.lines() {
        if line == "env:" {
            in_workflow_env = true;
            continue;
        }
        if in_workflow_env {
            if !line.starts_with("  ") || line.trim().is_empty() {
                in_workflow_env = false;
            } else if let Some(v) = line.trim().strip_prefix("RUSTDOCFLAGS:") {
                workflow_env = Some(v.trim().to_string());
            }
        }
    }

    // The `docs:` job runs from its own header to the next job header, which is
    // any other key at that indentation.
    let mut in_job = false;
    let mut in_job_env = false;
    for line in WORKFLOW.lines() {
        if line == "  docs:" {
            in_job = true;
            continue;
        }
        if !in_job {
            continue;
        }
        // A new key at the job indentation ends this job.
        if line.starts_with("  ") && !line.starts_with("   ") && line.trim_end().ends_with(':') {
            break;
        }
        if line == "    env:" {
            in_job_env = true;
            continue;
        }
        if in_job_env {
            if !line.starts_with("      ") {
                in_job_env = false;
            } else if let Some(v) = line.trim().strip_prefix("RUSTDOCFLAGS:") {
                job_env = Some(v.trim().to_string());
            }
        }
        if let Some(run) = line.trim().strip_prefix("- run: cargo doc") {
            assert!(
                args.is_none(),
                "the docs job runs `cargo doc` more than once; this guard assumes one build",
            );
            args = Some(run.split_whitespace().map(str::to_string).collect());
        }
    }

    assert!(in_job, "no `docs:` job in .github/workflows/ci.yml");
    let flags = job_env
        .or(workflow_env)
        .expect("the docs job sets no RUSTDOCFLAGS in any scope");

    DocInvocation {
        denies: parse_denies(&flags),
        args: args.expect("the docs job never runs `cargo doc`"),
    }
}

/// Pull the `doc` recipe out of the `Makefile`.
///
/// A file-level `RUSTDOCFLAGS =` assignment is exported to every recipe, so it
/// counts unless the recipe line spells the variable itself. That is the same
/// conservative rule `tests/miri_invocation_parity.rs` applies, and for the same
/// reason: a false positive is fixed by putting the variable on the recipe line,
/// which is where it belongs.
fn makefile_invocation() -> DocInvocation {
    let file_level = MAKEFILE
        .lines()
        .find(|l| !l.starts_with('\t') && l.trim_start().starts_with("RUSTDOCFLAGS"))
        .and_then(|l| l.split_once('='))
        .map(|(_, v)| v.trim().trim_matches('"').to_string());

    let mut in_recipe = false;
    let mut recipe: Option<&str> = None;
    for line in MAKEFILE.lines() {
        if line.starts_with("doc:") {
            in_recipe = true;
            continue;
        }
        if !in_recipe {
            continue;
        }
        // A recipe ends at the first line that is not a tab-indented command.
        if !line.starts_with('\t') {
            break;
        }
        // The recipe echoes its own headline, and that echo names `cargo doc`
        // too. Only the command line counts.
        let command = line.trim_start_matches('\t').trim_start_matches('@');
        if command.starts_with("echo ") {
            continue;
        }
        if command.contains("cargo doc") {
            assert!(
                recipe.is_none(),
                "the `doc` target runs `cargo doc` more than once; this guard assumes one build",
            );
            recipe = Some(line);
        }
    }
    assert!(in_recipe, "no `doc:` target in the Makefile");
    let recipe = recipe.expect("the `doc` target never runs `cargo doc`");

    let flags = match recipe.split_once("RUSTDOCFLAGS=") {
        Some((_, rest)) => {
            let rest = rest.strip_prefix('"').expect("RUSTDOCFLAGS is not quoted");
            let (value, _) = rest.split_once('"').expect("unterminated RUSTDOCFLAGS");
            value.to_string()
        }
        None => {
            file_level.expect("the `doc` recipe sets no RUSTDOCFLAGS and neither does the file")
        }
    };

    let (_, after) = recipe.split_once("cargo doc").expect("no `cargo doc` call");
    DocInvocation {
        denies: parse_denies(&flags),
        args: after.split_whitespace().map(str::to_string).collect(),
    }
}

/// Collect every `-D <lint>` out of a `RUSTDOCFLAGS` value.
fn parse_denies(flags: &str) -> BTreeSet<String> {
    let tokens: Vec<&str> = flags.split_whitespace().collect();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "-D" | "--deny" => {
                let lint = tokens.get(i + 1).unwrap_or_else(|| {
                    panic!("`{}` with no lint after it in {flags:?}", tokens[i])
                });
                out.insert((*lint).to_string());
                i += 2;
            }
            t => {
                if let Some(lint) = t.strip_prefix("-D") {
                    out.insert(lint.to_string());
                }
                i += 1;
            }
        }
    }
    out
}

#[test]
fn both_doc_invocations_deny_the_required_lints() {
    let workflow = workflow_invocation();
    let makefile = makefile_invocation();
    for lint in REQUIRED_DENIES {
        assert!(
            workflow.denies.contains(lint),
            "the ci.yml docs job does not deny `{lint}`; it denies {:?}",
            workflow.denies,
        );
        assert!(
            makefile.denies.contains(lint),
            "the Makefile `doc` target does not deny `{lint}`; it denies {:?}",
            makefile.denies,
        );
    }
}

#[test]
fn the_local_mirror_matches_the_hosted_docs_job() {
    let workflow = workflow_invocation();
    let makefile = makefile_invocation();
    assert_eq!(
        workflow.denies, makefile.denies,
        "`make doc` and the ci.yml docs job deny different lints, so the mirror \
         is not a mirror",
    );
    assert_eq!(
        workflow.args, makefile.args,
        "`make doc` and the ci.yml docs job build different surfaces",
    );
}

#[test]
fn the_docs_job_name_states_every_lint_it_denies() {
    let name = WORKFLOW
        .lines()
        .skip_while(|l| *l != "  docs:")
        .nth(1)
        .and_then(|l| l.trim().strip_prefix("name:"))
        .map(|n| n.trim().to_string())
        .expect("the docs job has no `name:` on the line after its header");

    for lint in workflow_invocation().denies {
        // `rustdoc::broken_intra_doc_links` has to show up in the name as the
        // word "broken", `rustdoc::private_intra_doc_links` as "private".
        let word = lint
            .rsplit("::")
            .next()
            .and_then(|l| l.split('_').next())
            .expect("a lint name with no path segment");
        assert!(
            name.to_lowercase().contains(word),
            "the docs job denies `{lint}` but its name {name:?} never says `{word}`",
        );
    }
}
