//! Keeps the two places that invoke Miri saying the same thing, and keeps the
//! claims `merge-gate.yml` makes about the Miri gate true (issues #675, #707).
//!
//! # Why this exists
//!
//! Miri is invoked from two files. `.github/workflows/merge-gate.yml` runs it
//! on the hosted runner, and the `miri` target in the `Makefile` is the local
//! mirror of that job. A mirror that runs different flags is worse than no
//! mirror at all: it reports green on a machine while the thing it claims to
//! mirror reports red, and nobody looks at the flags because they are two files
//! apart.
//!
//! They had already drifted before this guard existed. The workflow carried
//! `MIRIFLAGS: -Zmiri-disable-isolation` and `RUSTFLAGS: -A deprecated`; the
//! `Makefile` carried neither, so `make miri` did not even compile the crate
//! (`[lints.rust] deprecated = "deny"` plus nightly's rename of
//! `AtomicU64::fetch_update` is #643) and `make ci` had been failing at that
//! step for as long as the lint has been denied.
//!
//! # Every scope an environment variable can arrive from
//!
//! The first version of this guard read the step's own `env:` and the one
//! `Makefile` recipe line, and that is three quarters of a check. GitHub merges
//! `env:` from the workflow, the job and the step, in that order, and make
//! exports file-level variables to every recipe it runs. So
//! `-Zmiri-disable-isolation` could come back in four places and this only
//! looked at one; the other three were measured putting it back with the guard
//! still green.
//!
//! [`workflow_invocation`] now merges all three YAML scopes with step-beats-job
//! and job-beats-workflow precedence, and [`makefile_invocation`] treats any
//! `MIRIFLAGS`/`RUSTFLAGS` assignment anywhere in the file as reaching the
//! recipe unless the recipe line sets the same variable itself. That last rule
//! is deliberately conservative: a file-level assignment that make would not
//! actually export still fails here, and the fix for a false positive is to
//! spell the variable on the recipe line, which is where it belongs anyway.
//!
//! # Why the counts are a bound rather than a number
//!
//! An earlier version asserted the exact size of the unannotated backlog
//! against the number `merge-gate.yml` quoted. That was worse than the staleness
//! it replaced. Every pull request that adds a filesystem-touching test anywhere
//! under `src/` changes that number, so the workflow file became a
//! mandatory-edit hotspot in a repository whose batching rule is that no two
//! pull requests touch the same file, and unlike `CHANGELOG.md` it is a single
//! number rather than an append, so it is a hard conflict every time. It
//! collided twice in two days.
//!
//! So the workflow states a bound and this asserts the bound. The exact number
//! lives in `tests/miri_fs_test_inventory.txt`, which is the file that has to be
//! edited anyway, and `tests/miri_ignore_convention.rs` holds it against the
//! tree. When the backlog finally drops through the bound, this goes red once
//! and the sentence gets rewritten once, which is the right number of times.
//!
//! # Why it runs under Miri
//!
//! Every file this reads is pulled in with `include_str!` at compile time rather
//! than opened at runtime, so there is no filesystem access to isolate and these
//! tests need no `#[cfg_attr(miri, ignore)]`. A guard about the Miri gate that
//! the Miri gate cannot run would be a poor joke. It also means editing any of
//! the three files forces a rebuild of this test.

use std::collections::BTreeMap;

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

/// The bound `merge-gate.yml` states about the unannotated backlog, and the
/// words it states it in. Asserted rather than quoted exactly, for the reason in
/// the module docs.
const BACKLOG_BOUND: usize = 100;
/// How the workflow has to spell [`BACKLOG_BOUND`].
const BACKLOG_PHRASE: &str = "more than a hundred unannotated filesystem-touching tests";

/// One Miri invocation, however it happens to be written down.
#[derive(Debug, PartialEq, Eq)]
struct Invocation {
    /// The `+toolchain` selector, without its `+`, or `None` when the command
    /// takes whatever `cargo` defaults to. Compared separately from the command
    /// because the two sides cannot agree on it: the hosted job resolves
    /// `dtolnay/rust-toolchain@nightly` on the day, and the local mirror pins a
    /// date so it does not land on a nightly below the MSRV.
    toolchain: Option<String>,
    /// The cargo command with the toolchain selector and any leading environment
    /// assignments removed, and its whitespace collapsed, so `Makefile` tabs and
    /// YAML indentation do not count as a difference.
    command: String,
    /// The [`SIGNIFICANT_ENV`] variables the command runs with, merged across
    /// every scope that reaches it. A variable that is not set anywhere is
    /// absent from the map rather than present and empty.
    env: BTreeMap<String, String>,
}

/// Split a shell command line into tokens, honouring single and double quotes so
/// `RUSTFLAGS='-A deprecated --cfg sha2_backend="soft"'` comes back as one token
/// with the outer quotes removed and the inner ones kept.
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

/// Split a `KEY=value` token, accepting only shell-shaped variable names so a
/// path or a flag containing `=` is not mistaken for an assignment.
fn as_assignment(token: &str) -> Option<(String, String)> {
    let (key, value) = token.split_once('=')?;
    let shaped = !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    shaped.then(|| (key.to_string(), value.to_string()))
}

/// Pull the leading `KEY=value` assignments off a shell command, then the
/// `+toolchain` selector, and return them alongside what is left.
fn split_command(tokens: &[String]) -> (BTreeMap<String, String>, Option<String>, String) {
    let mut env = BTreeMap::new();
    let mut rest = tokens;

    while let Some(first) = rest.first() {
        let Some((key, value)) = as_assignment(first) else {
            break;
        };
        env.insert(key, value);
        rest = &rest[1..];
    }

    let toolchain = rest
        .iter()
        .find_map(|token| token.strip_prefix('+').map(str::to_string));
    let command: Vec<&str> = rest
        .iter()
        .map(String::as_str)
        .filter(|token| !token.starts_with('+'))
        .collect();

    (env, toolchain, command.join(" "))
}

/// Expand `$(NAME)` references against the `Makefile`'s own simple assignments,
/// so a recipe written in terms of `$(MIRI_TOOLCHAIN)` compares as the value it
/// resolves to.
fn expand_make_vars(line: &str) -> String {
    let mut vars: BTreeMap<&str, &str> = BTreeMap::new();
    for raw in MAKEFILE.lines() {
        if raw.starts_with('\t') || raw.trim_start().starts_with('#') {
            continue;
        }
        let trimmed = raw.trim();
        let Some((lhs, rhs)) = trimmed.split_once('=') else {
            continue;
        };
        let name = lhs.trim_end_matches([':', '?', '+']).trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        vars.insert(name, rhs.trim());
    }

    let mut out = line.to_string();
    // Two passes is enough for the one level of indirection this file uses, and
    // a third would only hide a variable defined in terms of itself.
    for _ in 0..2 {
        for (name, value) in &vars {
            out = out.replace(&format!("$({name})"), value);
        }
    }
    out
}

/// Join `Makefile` recipe lines that end in a backslash, so a multi-line shell
/// command is one string.
fn join_continuations(lines: &[&str]) -> Vec<String> {
    let mut joined: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    for line in lines {
        let body = line.strip_suffix('\\');
        let piece = body.unwrap_or(line);
        match &mut pending {
            Some(acc) => {
                acc.push(' ');
                acc.push_str(piece.trim());
            }
            None => pending = Some(piece.trim().to_string()),
        }
        if body.is_none() {
            joined.push(pending.take().unwrap_or_default());
        }
    }
    if let Some(last) = pending {
        joined.push(last);
    }
    joined
}

/// Flag variables assigned at `Makefile` scope, which make exports to every
/// recipe. Conservative on purpose: see the module docs.
fn makefile_file_scope_env() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for raw in MAKEFILE.lines() {
        if raw.starts_with('\t') || raw.trim_start().starts_with('#') {
            continue;
        }
        let trimmed = raw.trim().trim_start_matches("export ").trim();
        let Some((lhs, rhs)) = trimmed.split_once('=') else {
            continue;
        };
        let name = lhs.trim_end_matches([':', '?', '+']).trim();
        if SIGNIFICANT_ENV.contains(&name) {
            env.insert(
                name.to_string(),
                rhs.trim().trim_matches(['\'', '"']).to_string(),
            );
        }
    }
    env
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

    let statements = join_continuations(&recipe);
    let commands: Vec<&String> = statements
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

    let (mut env, toolchain, command) =
        split_command(&shell_tokens(&expand_make_vars(commands[0])));
    env.retain(|key, _| SIGNIFICANT_ENV.contains(&key.as_str()));

    // File scope first, then the recipe line, so the recipe wins where both set
    // the same variable.
    let mut merged = makefile_file_scope_env();
    merged.extend(env);

    Invocation {
        toolchain,
        command,
        env: merged,
    }
}

/// Strip the quoting off a YAML plain or single/double quoted scalar. Nothing
/// here needs block scalars or escapes, and a value that reaches for one should
/// fail loudly rather than be half-parsed.
fn yaml_scalar(raw: &str) -> String {
    let value = raw.trim();
    assert!(
        !value.starts_with('|') && !value.starts_with('>'),
        "block scalar in a Miri `env:` block, which this guard does not parse: {raw}"
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

/// Read the mapping that follows an `env:` line, stopping when the indentation
/// returns to the level of `env:` itself.
fn yaml_env_block(lines: &[&str], env_line: usize) -> BTreeMap<String, String> {
    let base = indent_of(lines[env_line]);
    let mut env = BTreeMap::new();
    for line in &lines[env_line + 1..] {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if indent_of(line) <= base {
            break;
        }
        let (key, value) = trimmed
            .split_once(':')
            .unwrap_or_else(|| panic!("unparseable line in a Miri `env:` block: {trimmed}"));
        env.insert(key.trim().to_string(), yaml_scalar(value));
    }
    env
}

/// The first `env:` line at exactly `indent`, searched in `range`.
fn find_env_line(lines: &[&str], range: std::ops::Range<usize>, indent: usize) -> Option<usize> {
    lines[range.clone()]
        .iter()
        .enumerate()
        .find(|(_, line)| line.trim() == "env:" && indent_of(line) == indent)
        .map(|(offset, _)| range.start + offset)
}

/// Where the `miri:` job's block ends: the next line indented at or above the
/// job key itself.
fn block_end(lines: &[&str], start: usize, key_indent: usize) -> usize {
    lines[start + 1..]
        .iter()
        .position(|line| !line.trim().is_empty() && indent_of(line) <= key_indent)
        .map_or(lines.len(), |offset| start + 1 + offset)
}

/// The Miri invocation in `merge-gate.yml`'s `miri` job, with `env:` merged from
/// the workflow, the job and the step in GitHub's precedence order.
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
        "expected exactly one `run:` step invoking `miri test` in `merge-gate.yml`, found {}",
        steps.len()
    );
    let step = steps[0];
    let step_indent = indent_of(lines[step]);

    let jobs_line = lines
        .iter()
        .position(|line| line.trim_end() == "jobs:")
        .expect("`merge-gate.yml` has no `jobs:` key");
    let miri_line = lines
        .iter()
        .position(|line| line.trim() == "miri:" && indent_of(line) == 2)
        .expect("`merge-gate.yml` has no `miri:` job at the expected indentation");
    let miri_end = block_end(&lines, miri_line, 2);

    // Workflow scope: an `env:` at column zero, outside `jobs:`.
    let mut env = find_env_line(&lines, 0..jobs_line, 0)
        .map(|idx| yaml_env_block(&lines, idx))
        .unwrap_or_default();
    // Job scope beats it.
    if let Some(idx) = find_env_line(&lines, miri_line..miri_end, 4) {
        env.extend(yaml_env_block(&lines, idx));
    }
    // Step scope beats both.
    let step_end = block_end(&lines, step, step_indent);
    if let Some(idx) = find_env_line(&lines, step..step_end, step_indent + 2) {
        env.extend(yaml_env_block(&lines, idx));
    }
    env.retain(|key, _| SIGNIFICANT_ENV.contains(&key.as_str()));

    let run = lines[step]
        .trim_start()
        .trim_start_matches("- run:")
        .trim()
        .to_string();
    let (inline_env, toolchain, command) = split_command(&shell_tokens(&run));
    for (key, value) in inline_env {
        if SIGNIFICANT_ENV.contains(&key.as_str()) {
            env.insert(key, value);
        }
    }

    Invocation {
        toolchain,
        command,
        env,
    }
}

/// The inventory rows, skipping its comment header, as `(annotated, path)`.
fn inventory_rows() -> Vec<(bool, &'static str)> {
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
            (annotated, fields[2])
        })
        .collect()
}

#[test]
fn the_makefile_and_merge_gate_run_miri_the_same_way() {
    let makefile = makefile_invocation();
    let workflow = workflow_invocation();

    assert_eq!(
        makefile.command, workflow.command,
        "`make miri` and the `miri` job in `merge-gate.yml` run different commands, so the \
         local mirror does not mirror anything. Change both or neither."
    );
    assert_eq!(
        makefile.env, workflow.env,
        "`make miri` and the `miri` job in `merge-gate.yml` disagree on {SIGNIFICANT_ENV:?}, \
         so a local green and a hosted green mean different things. This compares every \
         scope a variable can arrive from, not just the ones spelled next to the command."
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
                 back. That flag is what let the unannotated filesystem tests execute under \
                 the interpreter, and it is why three consecutive hosted runs were killed at \
                 the 90 minute ceiling with nothing to show (#675). The convention that \
                 replaces it is `#[cfg_attr(miri, ignore)]`, enforced by \
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
            "{where_} does not pin sha2's portable backend. On a host whose baseline has \
             the `sha2` target feature, which `aarch64-apple-darwin` does, the run reaches \
             sha2's NEON path and aborts on a Stacked Borrows violation about 30 seconds \
             in, before it has checked anything (#707). RUSTFLAGS is currently `{rustflags}`."
        );
    }
}

#[test]
fn the_local_mirror_does_not_run_on_the_bare_nightly() {
    let toolchain = makefile_invocation().toolchain.expect(
        "the `Makefile`'s miri recipe names no `+toolchain`, so it takes whatever \
                 `cargo` defaults to, which is a stable compiler that has no Miri at all",
    );
    assert_ne!(
        toolchain, "nightly",
        "the `Makefile` runs Miri on the floating `nightly`. That is what made `make miri` \
         unrunnable while the workflow claimed the opposite: `+nightly` resolved to \
         1.96.0-nightly here, below the crate's 1.97 MSRV, so cargo refused to build and \
         Miri was never reached. Pin a dated nightly and let `MIRI_TOOLCHAIN` override it."
    );
}

#[test]
fn merge_gate_states_the_backlog_as_a_bound_it_still_meets() {
    let rows = inventory_rows();
    let unannotated = rows.iter().filter(|(annotated, _)| !annotated).count();

    assert!(
        unannotated > BACKLOG_BOUND,
        "the unannotated backlog is down to {unannotated}, which is no longer \
         `{BACKLOG_PHRASE}`. That sentence in `merge-gate.yml` is now false: rewrite it \
         against the real number, and revisit `BACKLOG_BOUND` here. Reaching this point is \
         #712 nearly finishing, which is good news."
    );
    assert!(
        WORKFLOW.contains(BACKLOG_PHRASE),
        "`merge-gate.yml` no longer says `{BACKLOG_PHRASE}`. The job aborts on the first \
         unannotated filesystem test it reaches, and the workflow has to keep saying so or \
         the next reader will assume the gate reports. The live count is {unannotated}."
    );
    assert!(
        WORKFLOW.contains("tests/miri_fs_test_inventory.txt"),
        "`merge-gate.yml` no longer points at `tests/miri_fs_test_inventory.txt`. The exact \
         count deliberately lives there rather than here, so the workflow has to name it."
    );
}
