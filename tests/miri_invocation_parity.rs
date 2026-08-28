//! Keeps the two places that invoke Miri saying the same thing, and keeps the
//! prose in `merge-gate.yml` about the Miri convention true (issues #675, #707).
//!
//! # Why this exists
//!
//! Miri is invoked from two files. `.github/workflows/merge-gate.yml` runs it
//! on the hosted runner, and the `miri` target in the `Makefile` is the local
//! mirror of that job. A mirror that runs different flags is worse than no
//! mirror at all: it reports green on a machine while the thing it claims to
//! mirror reports red, and nobody looks at the flags because they are two
//! files apart.
//!
//! They had already drifted before this guard existed. The workflow carried
//! `MIRIFLAGS: -Zmiri-disable-isolation` and `RUSTFLAGS: -A deprecated`; the
//! `Makefile` carried neither, so `make miri` did not even compile the crate
//! (`[lints.rust] deprecated = "deny"` plus nightly's rename of
//! `AtomicU64::fetch_update` is #643) and `make ci` had been failing at that
//! step for as long as the lint has been denied.
//!
//! # Why the counts are in here too
//!
//! The comment block on the `miri` job describes the
//! `#[cfg_attr(miri, ignore)]` convention by quoting numbers at the reader.
//! Those numbers were measured once, at `f62a56a`, and then went stale in
//! place: the workflow said 48 annotations across seven modules while the tree
//! carried 53 across eight. `tests/miri_ignore_convention.rs` noticed and left
//! a note asking whoever owns the workflow to fix it, which is the same shape
//! of defect one level up. A sentence in a workflow file that reads as
//! measured, with nothing enforcing it, is exactly what this epic keeps
//! finding.
//!
//! So the counts are asserted against `tests/miri_fs_test_inventory.txt`, and
//! the inventory is asserted against the tree by
//! `tests/miri_ignore_convention.rs`. Two links, each of them checked.
//!
//! # Why it runs under Miri
//!
//! Every file this reads is pulled in with `include_str!` at compile time
//! rather than opened at runtime, so there is no filesystem access to isolate
//! and these tests need no `#[cfg_attr(miri, ignore)]`. A guard about the Miri
//! gate that the Miri gate cannot run would be a poor joke. It also means
//! editing any of the three files forces a rebuild of this test.

use std::collections::{BTreeMap, BTreeSet};

/// The hosted job.
const WORKFLOW: &str = include_str!("../.github/workflows/merge-gate.yml");
/// The local mirror of it.
const MAKEFILE: &str = include_str!("../Makefile");
/// The ledger `tests/miri_ignore_convention.rs` keeps against the tree.
const INVENTORY: &str = include_str!("miri_fs_test_inventory.txt");

/// Environment variables that change what Miri checks, as opposed to ones that
/// only change how it prints. Both invocations have to agree on all of these,
/// including on leaving one unset.
const SIGNIFICANT_ENV: [&str; 2] = ["MIRIFLAGS", "RUSTFLAGS"];

/// One Miri invocation, however it happens to be written down.
#[derive(Debug, PartialEq, Eq)]
struct Invocation {
    /// The cargo command with any leading environment assignments stripped and
    /// its whitespace collapsed, so `Makefile` tabs and YAML indentation do not
    /// count as a difference.
    command: String,
    /// The [`SIGNIFICANT_ENV`] variables the command runs with. A variable that
    /// is not set is absent from the map rather than present and empty.
    env: BTreeMap<String, String>,
}

/// Split a shell command line into tokens, honouring single and double quotes
/// so `RUSTFLAGS='-A deprecated --cfg sha2_backend="soft"'` comes back as one
/// token with the outer quotes removed and the inner ones kept.
fn shell_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut has_token = false;
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = line.chars();

    while let Some(ch) = chars.next() {
        match ch {
            '\\' if !in_single => {
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                    has_token = true;
                }
            }
            '\'' if !in_double => {
                in_single = !in_single;
                has_token = true;
            }
            '"' if !in_single => {
                in_double = !in_double;
                has_token = true;
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if has_token {
                    tokens.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            c => {
                current.push(c);
                has_token = true;
            }
        }
    }

    assert!(
        !in_single && !in_double,
        "unbalanced quotes in shell command: {line}"
    );
    if has_token {
        tokens.push(current);
    }
    tokens
}

/// Pull the leading `KEY=value` assignments off a shell command and return them
/// alongside the command that follows.
fn split_env_prefix(tokens: &[String]) -> (BTreeMap<String, String>, String) {
    let mut env = BTreeMap::new();
    let mut rest = tokens;

    while let Some(first) = rest.first() {
        let Some((key, value)) = first.split_once('=') else {
            break;
        };
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        {
            break;
        }
        env.insert(key.to_string(), value.to_string());
        rest = &rest[1..];
    }

    (env, rest.join(" "))
}

/// The Miri invocation in the `Makefile`'s `miri` target.
fn makefile_invocation() -> Invocation {
    let mut recipe = Vec::new();
    let mut in_target = false;
    for line in MAKEFILE.lines() {
        if line.starts_with('\t') {
            if in_target {
                recipe.push(line.trim_start_matches('\t'));
            }
            continue;
        }
        in_target = line.trim_end() == "miri:";
    }
    assert!(
        !recipe.is_empty(),
        "the `Makefile` has no `miri:` target, or its recipe is empty. This guard \
         locates it by an exact `miri:` line at column zero; if the target was renamed \
         or given prerequisites, teach `tests/miri_invocation_parity.rs` about it \
         rather than letting it match nothing."
    );

    let commands: Vec<&&str> = recipe
        .iter()
        .filter(|line| {
            let bare = line.trim_start_matches(['@', '-', '+']);
            line.contains("miri test") && !bare.starts_with("echo ")
        })
        .collect();
    assert_eq!(
        commands.len(),
        1,
        "expected exactly one `miri test` command in the `Makefile`'s `miri:` recipe, \
         found {}: {commands:?}",
        commands.len()
    );

    let (mut env, command) = split_env_prefix(&shell_tokens(commands[0]));
    env.retain(|key, _| SIGNIFICANT_ENV.contains(&key.as_str()));
    Invocation { command, env }
}

/// Strip the quoting off a YAML plain or single/double quoted scalar. Nothing
/// here needs block scalars or escapes, and a value that reaches for one should
/// fail loudly rather than be half-parsed.
fn yaml_scalar(raw: &str) -> String {
    let value = raw.trim();
    assert!(
        !value.starts_with('|') && !value.starts_with('>'),
        "block scalar in the Miri step's `env:`, which this guard does not parse: {raw}"
    );
    for quote in ['\'', '"'] {
        if value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote) {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

/// Count the leading spaces on a line.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The Miri invocation in `merge-gate.yml`'s `miri` job.
fn workflow_invocation() -> Invocation {
    let lines: Vec<&str> = WORKFLOW.lines().collect();
    let steps: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            let trimmed = line.trim_start();
            trimmed.starts_with("- run:") && trimmed.contains("miri test")
        })
        .map(|(idx, _)| idx)
        .collect();
    assert_eq!(
        steps.len(),
        1,
        "expected exactly one `run:` step invoking `miri test` in `merge-gate.yml`, \
         found {}",
        steps.len()
    );

    let start = steps[0];
    let step_indent = indent_of(lines[start]);
    let command = lines[start]
        .trim_start()
        .trim_start_matches("- run:")
        .trim()
        .to_string();

    let mut env = BTreeMap::new();
    let mut env_indent = None;
    for line in &lines[start + 1..] {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if indent_of(line) <= step_indent {
            break;
        }
        if trimmed == "env:" {
            env_indent = Some(indent_of(line));
            continue;
        }
        let Some(env_indent) = env_indent else {
            continue;
        };
        if indent_of(line) <= env_indent {
            break;
        }
        let (key, value) = trimmed.split_once(':').unwrap_or_else(|| {
            panic!("unparseable line in the Miri step's `env:` block: {trimmed}")
        });
        env.insert(key.trim().to_string(), yaml_scalar(value));
    }

    env.retain(|key, _| SIGNIFICANT_ENV.contains(&key.as_str()));
    Invocation { command, env }
}

/// One parsed inventory row.
struct Row<'a> {
    annotated: bool,
    path: &'a str,
}

/// The inventory rows, skipping its comment header.
fn inventory_rows() -> Vec<Row<'static>> {
    INVENTORY
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(
                fields.len(),
                3,
                "inventory row is not `state detected path`: {line}"
            );
            let annotated = match fields[0] {
                "annotated" => true,
                "unannotated" => false,
                other => panic!("unknown annotation state `{other}` in inventory row: {line}"),
            };
            Row {
                annotated,
                path: fields[2],
            }
        })
        .collect()
}

#[test]
fn the_makefile_and_merge_gate_run_miri_the_same_way() {
    let makefile = makefile_invocation();
    let workflow = workflow_invocation();

    assert_eq!(
        makefile.command, workflow.command,
        "`make miri` and the `miri` job in `merge-gate.yml` run different commands, so \
         the local mirror does not mirror anything. Change both or neither."
    );
    assert_eq!(
        makefile.env, workflow.env,
        "`make miri` and the `miri` job in `merge-gate.yml` disagree on {SIGNIFICANT_ENV:?}, \
         so a local green and a hosted green mean different things."
    );
}

#[test]
fn neither_miri_invocation_disables_isolation() {
    for (where_, invocation) in [
        ("Makefile", makefile_invocation()),
        ("merge-gate.yml", workflow_invocation()),
    ] {
        for (key, value) in &invocation.env {
            assert!(
                !value.contains("-Zmiri-disable-isolation"),
                "{where_} sets {key} to `{value}`, which puts `-Zmiri-disable-isolation` \
                 back. That flag is what let the unannotated filesystem tests execute \
                 under the interpreter, and it is why three consecutive hosted runs were \
                 killed at the 90 minute ceiling with nothing to show (#675). The \
                 convention that replaces it is `#[cfg_attr(miri, ignore)]`, enforced by \
                 `tests/miri_ignore_convention.rs`."
            );
        }
    }
}

#[test]
fn both_miri_invocations_pin_the_sha2_backend() {
    for (where_, invocation) in [
        ("Makefile", makefile_invocation()),
        ("merge-gate.yml", workflow_invocation()),
    ] {
        let rustflags = invocation.env.get("RUSTFLAGS").map_or("", String::as_str);
        assert!(
            rustflags.contains("--cfg sha2_backend=\"soft\""),
            "{where_} does not pin sha2's portable backend, so on aarch64 the run reaches \
             sha2's NEON path and aborts on a Stacked Borrows violation about 30 seconds \
             in, before it has checked anything (#707). RUSTFLAGS is currently `{rustflags}`."
        );
    }
}

#[test]
fn merge_gate_quotes_the_miri_counts_the_inventory_records() {
    let rows = inventory_rows();
    let annotated_src: Vec<&Row<'_>> = rows
        .iter()
        .filter(|row| row.annotated && row.path.starts_with("src/"))
        .collect();
    let modules: BTreeSet<&str> = annotated_src
        .iter()
        .map(|row| row.path.split("::").next().unwrap_or(row.path))
        .collect();
    let unannotated = rows.iter().filter(|row| !row.annotated).count();

    let convention = format!(
        "{} `#[cfg_attr(miri, ignore)]` annotations across {} modules",
        annotated_src.len(),
        modules.len()
    );
    assert!(
        WORKFLOW.contains(&convention),
        "`merge-gate.yml` does not say `{convention}` on any one line. The tree moved and \
         the workflow's description of the convention did not, which is how it came to \
         claim 48 across seven while the inventory recorded 53 across eight."
    );

    let ledger = format!("{unannotated} unannotated filesystem-touching tests");
    assert!(
        WORKFLOW.contains(&ledger),
        "`merge-gate.yml` does not say `{ledger}` on any one line. That number is how far \
         the job is from completing with isolation on: each one of them aborts the run on \
         its first filesystem call, so the workflow has to keep quoting it accurately or \
         the next reader will assume the gate reports."
    );
}
