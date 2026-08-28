//! Enforces the `#[cfg_attr(miri, ignore)]` convention that keeps
//! `cargo +nightly miri test` runnable (issue #652).
//!
//! `merge-gate.yml` runs Miri as the only check this repository has on
//! undefined behaviour. Miri runs the interpreted program under an isolation
//! layer that refuses real syscalls, and it *aborts the whole run* on the first
//! unsupported operation rather than failing that one test. So a single new
//! test that reaches for `tempfile::tempdir()` takes the entire gate down, and
//! it reports as "Miri failed", which reads like undefined behaviour rather
//! than like a missing annotation.
//!
//! The convention that avoids this is a `#[cfg_attr(miri, ignore)]` on every
//! test that touches the filesystem. Nothing enforced it, so it broke:
//! `checksum::tests::hash_file_matches_in_memory_hash_for_both_algos` arrived
//! without one and disarmed the gate.
//!
//! # What this guard does
//!
//! It walks `src/` and `tests/` from the repo root, recursively, parses every
//! `#[test]` function out of every `.rs` file it finds, classifies each one as
//! filesystem-touching or not, and compares the result against
//! `tests/miri_fs_test_inventory.txt`. The comparison is an exact set equality
//! in both directions, so all four interesting edits fail the build:
//!
//! * a new filesystem-touching test appears and is not in the inventory;
//! * an existing annotation is deleted, flipping a recorded `annotated` entry
//!   to `unannotated`;
//! * an annotation is added, which is fine but must be recorded so the ledger
//!   keeps meaning what it says;
//! * a test in the inventory is deleted or renamed.
//!
//! Nothing here can be satisfied by editing a single grep pattern: the file set
//! comes from a directory walk, not from a hand-written list, and the parse
//! asserts its own shape per file (see [`scan_source`]) so a construct the
//! scanner does not understand fails loudly instead of silently shrinking the
//! window it looks at.
//!
//! # What the detector can see, and what it cannot
//!
//! It matches the substrings in [`FS_MARKERS`] against the body of each test,
//! with comments and string, byte-string and character literals masked out, so
//! a doc comment or an error message that merely mentions `std::fs` does not
//! count.
//!
//! It is a *syntactic* check on one function body, which means it cannot see:
//!
//! * filesystem access reached through a helper. Four of the annotations that
//!   predate this guard are exactly that shape and are recorded as
//!   `not-detected`: `stream_verify`'s three malformed-strip tests go through
//!   `assert_strip_layout_rejected`, and `source::tests::decode_file_not_found`
//!   goes through `decode_file`, which opens the path itself.
//! * filesystem access inside a library entry point that takes a `Path`. Any
//!   `foo(path)` that opens `path` internally reads as pure to this scanner.
//! * a filesystem call spelled through an alias (`use std::fs as f;`) or
//!   through a crate this list does not name.
//! * anything outside `src/` and `tests/`, which means the `fuzz/` member (a
//!   separate crate that `cargo miri test` on this package does not build) and
//!   the `build.rs`-less root manifest.
//!
//! The `not-detected` rows are what keeps those blind spots from being silent:
//! a test the detector cannot classify still gets pinned the moment somebody
//! annotates it, so the annotation cannot later be deleted unnoticed.
//!
//! # Both classes are refusals now, and the filesystem one keeps its rows
//!
//! Everything above was written as a ledger, back when `merge-gate.yml` ran
//! Miri with `-Zmiri-disable-isolation`: a filesystem call came back rather
//! than aborting, so an `unannotated fs-detected` test could stand. **#711
//! removed that flag**, and under isolation such a test ends the run with
//! `unsupported operation: \`open\` not available when isolation is enabled`.
//! Measured on `800c699`, plain `main`,
//! `cargo miri test --test workspace_layout` died on
//! `fuzz_crate_is_a_member_of_the_root_workspace` before running anything.
//!
//! #739 annotated 134 of the 138 rows that made that fatal, so
//! [`no_filesystem_touching_test_runs_under_miri_outside_the_named_exceptions`]
//! is an assertion now rather than a count of known debt. The four that are
//! left are in `src/resample.rs`, which had four open pull requests against it
//! while the sweep ran; they are named in [`UNANNOTATED_FS_EXCEPTIONS`] and
//! tracked by issue #756.
//!
//! The rows stay, and the set-equality check with them, because they catch a
//! different edit: an annotation being *deleted*, and an annotation being added
//! to a test the detector cannot see through (the `not-detected` rows). An
//! assertion that every filesystem test is annotated is green when somebody
//! annotates a pure test by mistake; the inventory is what makes that a new
//! `annotated not-detected` row and a red build.
//!
//! # Spawning a process, which was a refusal first
//!
//! `std::process` was different in kind before that change and is merely
//! *worse* after it. Miri supports process spawning on no target and under no
//! flag, so no `MIRIFLAGS` setting has ever made a spawning test survivable,
//! where the filesystem class was survivable until last week. Measured on
//! `120acb6`, `cargo +nightly miri test --test dependency_policy` died on
//! `every_links_key_is_on_the_allowlist` before running anything else in the
//! tree (issue #714).
//!
//! So [`no_process_spawning_test_can_run_under_miri`] is an outright assertion,
//! not a row: every test that reaches `std::process` must carry
//! `#[cfg_attr(miri, ignore)]`, full stop. There is no "record it as
//! unannotated and say why in the review" arm, because there is no
//! configuration in which such a test runs.
//!
//! # Following a helper, once, for the process case
//!
//! The filesystem detector reads one function body and stops, which is why the
//! `not-detected` rows exist. That is not good enough here, because none of the
//! eleven tests #714 found spells `Command::new` in its own body: they call
//! `cells()`, which calls `graphs()`, which spawns cargo. A body scan sees all
//! eleven as pure.
//!
//! [`process_spawning_fns`] closes that by resolving calls inside the file. It
//! parses every `fn` in the file, marks the ones whose body matches
//! [`PROCESS_MARKERS`], and then repeats to a fixed point, marking any function
//! whose body names an already-marked one. Two hops is what the real tree
//! needs; the loop takes any depth.
//!
//! Where it can, it over-approximates on purpose. A call is matched as the
//! *name*, on identifier boundaries, not as `name(`: `graphs()` reaches cargo
//! through `CELLS.iter().map(resolve)`, where the callee never sits next to a
//! paren at all, and insisting on one is how my first attempt at this missed
//! four of the ten tests #714 lists. The cost is that a method spelled the same
//! as a spawning free function counts, and so does a name that is merely
//! mentioned. Over-approximating costs an unnecessary annotation on a test Miri
//! could have run. Under-approximating costs the whole gate.
//!
//! # Where it under-approximates, which is the half that matters
//!
//! It would be comfortable to leave the paragraph above as the whole story. It
//! is not. Four shapes reach `std::process` without this seeing them, none of
//! them present in the tree today, all of them things somebody could write
//! tomorrow:
//!
//! * **An aliased import.** `use std::process::Command as Cmd;` then
//!   `Cmd::new(..)`. The `use` is at module scope rather than inside any `fn`,
//!   so [`fn_bodies`] never reads it and neither spelling matches
//!   [`PROCESS_MARKERS`]. Closing this needs the scan to resolve imports, which
//!   is a different kind of program from the one this file is.
//! * **A spawn inside a `macro_rules!` body.** Not an `fn`, so it is not in the
//!   map, and the expansion this file never sees is where the call appears.
//! * **A closure held in a `static` or a `const`.** Same reason: the body is
//!   not under an `fn` header, so nothing indexes it.
//! * **A helper in another file.** [`process_spawning_fns`] runs per file, so a
//!   spawning helper in a shared `tests/common/mod.rs` is invisible to every
//!   test that calls it. This is the one most likely to arrive by accident,
//!   because a shared test helper is an ordinary thing to write.
//!
//! [`EXPECTED_PROCESS_SPAWNING_TESTS`] is what stops that list growing in
//! silence. It pins how many tests the detector finds, so a change that makes
//! it stop seeing a whole class shows up as a count that moved, rather than as
//! an empty offender list that still reads as a pass.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The recorded state of the tree. Compiled in with `include_str!` rather than
/// read at runtime so the ledger and the binary asserting against it cannot
/// drift apart, and so editing it forces a rebuild.
const INVENTORY: &str = include_str!("miri_fs_test_inventory.txt");

/// Directories under the repo root whose `.rs` files `cargo miri test` compiles
/// and runs. Walked recursively; this is a list of *roots*, not of files.
const SCANNED_DIRS: [&str; 2] = ["src", "tests"];

/// Substrings that mean "the body of this test reaches the real filesystem".
///
/// Derived from what the 48 pre-existing annotated tests actually call:
/// `tempfile::tempdir` (97 uses), `std::fs::write` (34), `std::fs::read` (9),
/// `symlink` (8), `std::fs::create_dir_all` (7), `std::fs::read_dir` (6),
/// `std::fs::metadata` (5), then a tail of `remove_file`, `OpenOptions`,
/// `canonicalize`, `create_dir` and `File::open`. The `fs::` entries are
/// spelled with the item name attached so a local `mod fs` or an unrelated
/// `fs::Config` does not match.
const FS_MARKERS: &[&str] = &[
    // Temporary files and directories.
    "tempfile::",
    "TempDir",
    "NamedTempFile",
    // The whole of `std::fs`, however it is reached.
    "std::fs::",
    "fs::canonicalize(",
    "fs::copy(",
    "fs::create_dir(",
    "fs::create_dir_all(",
    "fs::exists(",
    "fs::hard_link(",
    "fs::metadata(",
    "fs::read(",
    "fs::read_dir(",
    "fs::read_link(",
    "fs::read_to_string(",
    "fs::remove_dir(",
    "fs::remove_dir_all(",
    "fs::remove_file(",
    "fs::rename(",
    "fs::set_permissions(",
    "fs::symlink_metadata(",
    "fs::write(",
    "fs::File",
    "fs::OpenOptions",
    // Handles and directory iteration.
    "File::open(",
    "File::create(",
    "File::create_new(",
    "OpenOptions::new(",
    "read_dir(",
    // Links, which are platform-specific and so are not spelled `std::fs::`.
    "symlink(",
    "symlink_file(",
    "symlink_dir(",
    "hard_link(",
    "std::os::unix::fs::",
    "std::os::windows::fs::",
    // `Path` methods that stat the path even though they read like accessors.
    ".canonicalize()",
    ".exists()",
    ".is_file()",
    ".is_dir()",
    ".metadata()",
    ".read_dir()",
    ".symlink_metadata()",
];

/// Substrings that mean "this function reaches `std::process`".
///
/// Short, because there is only one way to start a process from std and every
/// spelling of it goes through `Command`. `Command::new(` catches the
/// `use std::process::Command;` form and `process::Command` catches the
/// qualified one; the bare `.spawn(` / `.output(` / `.status(` methods are
/// deliberately absent, since those names collide with plenty of innocent APIs
/// and the constructor is unavoidable.
const PROCESS_MARKERS: &[&str] = &["Command::new(", "process::Command"];

/// Attribute forms of `#[test]` this scanner does not understand. Finding one
/// means the parse below would skip a real test, so it is a hard failure with
/// an instruction rather than a silent gap.
const UNSUPPORTED_TEST_ATTRS: &[&str] = &["::test]", "#[test(", "#[test_case"];

/// Floors that catch a scanner pointed at nothing, or at one file, instead of
/// letting it report a clean tree. Deliberately far below the real numbers
/// (97 files, 1834 tests at the time of writing) so ordinary churn never trips
/// them.
const MIN_FILES: usize = 50;
/// Companion floor to [`MIN_FILES`], on parsed `#[test]` functions.
const MIN_TESTS: usize = 1000;

/// Files that must turn up in the walk. Not the scan set (that is the walk
/// itself) — a canary that the walk reached both roots and is reading real
/// source rather than an empty or wrongly-rooted directory.
const ANCHOR_FILES: &[&str] = &[
    "src/lib.rs",
    "src/checksum.rs",
    "src/engine.rs",
    "tests/non_exhaustive_enums.rs",
    "tests/workspace_layout.rs",
];

/// Annotated tests under `src/`, pinned so a bulk change in either direction is
/// a deliberate edit here rather than a number that quietly drifts.
///
/// It went from 53 to 157 in a single change, which is #739's sweep: 104
/// filesystem tests across thirteen more `src/` modules picked the annotation
/// up at once, because #711 turned Miri's isolation on and made every one of
/// them fatal to the whole run rather than merely slow.
///
/// `merge-gate.yml` used to quote a count here ("48 annotations across seven
/// modules", true at `f62a56a` and stale for months afterwards). It quotes none
/// now, on purpose: an exact number in the workflow made it a file every
/// unrelated pull request had to edit, which is the reasoning written up in
/// `tests/miri_invocation_parity.rs`.
const EXPECTED_SRC_ANNOTATIONS: usize = 157;
/// Companion to [`EXPECTED_SRC_ANNOTATIONS`]: how many `src/` modules carry at
/// least one annotation.
const EXPECTED_SRC_MODULES: usize = 21;

/// How many tests in the tree reach `std::process`.
///
/// The positive control for [`no_process_spawning_test_can_run_under_miri`],
/// which on its own asserts that a set is empty and is therefore green both
/// when every spawning test is annotated and when the detector has quietly
/// stopped finding any. That is not hypothetical: matching a callee as `name(`
/// instead of as an identifier took this from 10 to 7 while every assertion
/// stayed green, and reading a `;` inside `[u8; 32]` as a bodyless declaration
/// dropped 116 functions out of the call graph the same way.
///
/// Eighteen today: five in `tests/dependency_policy.rs`, three in
/// `tests/pdfium_source_audit.rs` and two in `tests/workspace_layout.rs`, which
/// are the ten #714 is about; six in `tests/icc_lut_alloc.rs`, which spawn a
/// child on purpose to watch it abort (#693); one in `src/source.rs` that
/// shells out to `mkfifo` and was already annotated, for the filesystem reason,
/// before any of this; and one in `tests/oracle_capture_pins.rs` that runs
/// `git ls-files`, which is the one #701 brought in.
///
/// The population is wider than the ten that needed fixing, and deliberately
/// so. Pinning only the ten would go green again the moment the detector lost
/// the other seven, which is the failure this constant exists to catch. It
/// caught my own miscount the first time I ran it: I wrote eleven, having
/// forgotten the six that arrived with #693 in the commit underneath this one.
///
/// It then caught a second one, which is the better advertisement: #701 added
/// `no_compiled_python_is_tracked_under_oracle_captures`, which shells out to
/// `git ls-files`, and both PRs were green on their own branches. The count
/// only moved when they landed together, and it is the eighteenth here for
/// exactly that reason. This is a count that two file-disjoint changes can
/// both be right about and still break, so move it in the same change that
/// moves the population.
const EXPECTED_PROCESS_SPAWNING_TESTS: usize = 18;

/// The filesystem-touching tests still allowed to run under Miri, and so still
/// allowed to end the whole run on their first syscall.
///
/// Empty is the target state and an empty list is a legal one: the assertion in
/// [`no_filesystem_touching_test_runs_under_miri_outside_the_named_exceptions`]
/// reads this as an exception list, not as a floor. What it replaced was a
/// floor, `assert!(unannotated_fs > 0)`, and that is the difference #739 turned
/// on: the old form demanded the debt still exist and would have gone red on
/// the change that cleared it.
///
/// It is not empty today for a scheduling reason rather than a technical one.
/// `src/resample.rs` was held by the lane resolving #692, #704, #705, #732,
/// #733 and #736, with four pull requests open against that one file, while
/// #739's sweep ran across the other 28. Annotating these four here would have
/// been four hand-resolved conflicts in a module this change has no other
/// business in. Issue #756 carries them, and this list is how they stay
/// visible instead of becoming a quiet gap in an otherwise-enforced rule.
const UNANNOTATED_FS_EXCEPTIONS: &[&str] = &[
    "src/resample.rs::thumbnail_crop_free_fn_fills_and_crops_the_box",
    "src/resample.rs::thumbnail_file_and_buffer_agree",
    "src/resample.rs::thumbnail_free_fn_fits_the_width_box",
    "src/resample.rs::thumbnail_unknown_profile_is_typed_error",
];

/// How many tests in the tree the filesystem detector finds, annotated or not.
///
/// The positive control for
/// [`no_filesystem_touching_test_runs_under_miri_outside_the_named_exceptions`],
/// which is otherwise an assertion that a set is empty, and a detector that has
/// stopped recognising filesystem calls produces an empty set too. That is a
/// one-character edit away: [`FS_MARKERS`] is a substring list, and dropping
/// `"tempfile::"` from it alone takes this from 191 to 105 while leaving the
/// offender list empty and the check green. Measured, not reasoned: that
/// deletion is one of the mutations in #739's table.
const EXPECTED_FS_TOUCHING_TESTS: usize = 191;

/// Repo root (the directory holding the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// One parsed `#[test]` function.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TestFn {
    /// Repo-relative path with `/` separators.
    file: String,
    /// The function name.
    name: String,
    /// Whether the attribute block carries `#[cfg_attr(miri, ignore)]`.
    annotated: bool,
    /// Whether the body matched [`FS_MARKERS`].
    touches_fs: bool,
    /// Whether the body matched [`PROCESS_MARKERS`], or called something in
    /// the same file that (transitively) does. See [`process_spawning_fns`].
    spawns_process: bool,
}

impl TestFn {
    /// The canonical inventory line. Annotation state is part of the key on
    /// purpose: flipping it shows up as one removal plus one addition rather
    /// than as a silent no-op.
    fn ledger_line(&self) -> String {
        let ann = if self.annotated {
            "annotated"
        } else {
            "unannotated"
        };
        let det = if self.touches_fs {
            "fs-detected"
        } else {
            "not-detected"
        };
        let (file, name) = (&self.file, &self.name);
        format!("{ann:<11} {det:<12} {file}::{name}")
    }

    /// Whether this test belongs in the inventory at all: either the detector
    /// classified it as filesystem-touching, or somebody annotated it (which
    /// pins the helper-reached cases the detector cannot see).
    fn is_tracked(&self) -> bool {
        self.touches_fs || self.annotated
    }
}

/// Replace comments and string, byte-string, raw-string and character literals
/// with spaces, preserving newlines and character count so line numbers still
/// line up with the original.
///
/// Everything downstream (brace matching, attribute recognition, marker
/// matching) runs on the masked text, which is why a `//` comment naming
/// `std::fs::write` does not make a pure test look like a filesystem test, and
/// why a `{` inside a string literal cannot desynchronise the body scan.
fn mask_literals_and_comments(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let n = c.len();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    // Whether the previously emitted character can end an identifier, which is
    // how `br"x"` (a raw byte string) is told apart from `abr"x"` (which cannot
    // occur) and, more usefully, from an identifier ending in `r` or `b`.
    let mut prev_ident = false;

    // Blank out `[from, to)` as spaces, keeping newlines.
    fn blank(out: &mut String, c: &[char], from: usize, to: usize) {
        for &ch in &c[from..to] {
            out.push(if ch == '\n' { '\n' } else { ' ' });
        }
    }

    // Walk a normal (non-raw) quoted literal starting at the opening delimiter
    // `quote`, returning the index one past the closing delimiter.
    fn end_of_quoted(c: &[char], start: usize, quote: char) -> usize {
        let n = c.len();
        let mut j = start + 1;
        while j < n {
            if c[j] == '\\' {
                j += 2;
                continue;
            }
            if c[j] == quote {
                return j + 1;
            }
            j += 1;
        }
        n
    }

    while i < n {
        let ch = c[i];

        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            let mut j = i;
            while j < n && c[j] != '\n' {
                j += 1;
            }
            blank(&mut out, &c, i, j);
            i = j;
            prev_ident = false;
            continue;
        }

        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            let mut depth = 0usize;
            let mut j = i;
            while j < n {
                if c[j] == '/' && j + 1 < n && c[j + 1] == '*' {
                    depth += 1;
                    j += 2;
                    continue;
                }
                if c[j] == '*' && j + 1 < n && c[j + 1] == '/' {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                    continue;
                }
                j += 1;
            }
            blank(&mut out, &c, i, j);
            i = j;
            prev_ident = false;
            continue;
        }

        // `r"…"`, `r#"…"#`, `b"…"`, `br#"…"#`, `b'x'`.
        if !prev_ident && (ch == 'r' || ch == 'b') {
            let mut j = i;
            if c[j] == 'b' {
                j += 1;
            }
            let raw = j < n && c[j] == 'r';
            if raw {
                j += 1;
                let hash_start = j;
                while j < n && c[j] == '#' {
                    j += 1;
                }
                if j < n && c[j] == '"' {
                    let hashes = j - hash_start;
                    let mut k = j + 1;
                    let mut end = n;
                    while k < n {
                        if c[k] == '"' && c[k + 1..].iter().take(hashes).all(|&h| h == '#') {
                            end = (k + 1 + hashes).min(n);
                            break;
                        }
                        k += 1;
                    }
                    blank(&mut out, &c, i, end);
                    i = end;
                    prev_ident = false;
                    continue;
                }
            } else if j > i && j < n && (c[j] == '"' || c[j] == '\'') {
                let end = end_of_quoted(&c, j, c[j]);
                blank(&mut out, &c, i, end);
                i = end;
                prev_ident = false;
                continue;
            }
        }

        if ch == '"' {
            let end = end_of_quoted(&c, i, '"');
            blank(&mut out, &c, i, end);
            i = end;
            prev_ident = false;
            continue;
        }

        // A `'` is either a character literal or a lifetime. `'a` is a
        // lifetime; `'a'` and `'\n'` are literals. Getting this wrong would eat
        // the rest of the file, which the balanced-brace assertion would catch,
        // but it is cheap to get right.
        if ch == '\'' {
            let is_char_literal = (i + 1 < n && c[i + 1] == '\\')
                || (i + 2 < n && c[i + 2] == '\'')
                || (i + 1 < n && c[i + 1] == '\'');
            if is_char_literal {
                let end = end_of_quoted(&c, i, '\'');
                blank(&mut out, &c, i, end);
                i = end;
                prev_ident = false;
                continue;
            }
        }

        out.push(ch);
        prev_ident = ch.is_alphanumeric() || ch == '_';
        i += 1;
    }

    out
}

/// Strip the modifiers that may sit between an attribute block and `fn`, and
/// return the function name. `None` means the line is not a function header,
/// which the caller turns into a hard failure.
fn function_name(header: &str) -> Option<String> {
    let mut rest = header.trim();
    loop {
        let stripped = [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "async ",
            "unsafe ",
            "const ",
        ]
        .iter()
        .find_map(|kw| rest.strip_prefix(kw));
        match stripped {
            Some(s) => rest = s.trim_start(),
            None => break,
        }
    }
    let rest = rest.strip_prefix("fn ")?.trim_start();
    let name: String = rest
        .chars()
        .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Every `fn` in a masked file, by name, with its body.
///
/// A second, looser pass than the one [`scan_source`] runs: it takes any line
/// [`function_name`] can read a name off, not only the ones under `#[test]`, so
/// the helpers a test calls are in the map too. A signature that ends in `;`
/// before it opens a brace is a trait declaration with no body and is skipped,
/// which matters because consuming forward from one would swallow the next
/// function whole.
///
/// Names collide (two `mod`s can each define `helper`), and the map keeps the
/// last. That is fine for what it feeds: the caller only asks whether *some*
/// function of that name spawns, and the answer it wants on a collision is the
/// conservative one.
fn fn_bodies(masked: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = masked.lines().collect();
    let mut out: Vec<(String, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = function_name(line) else {
            continue;
        };
        let mut body = String::new();
        let mut depth = 0i64;
        // Square brackets and parens seen in the *signature*, before the body
        // opens. A `;` inside one belongs to an array type or a const-generic
        // default, not to the end of a declaration: `fn f() -> [u8; 32]` and
        // `where T: Into<[u8; 4]>` both carry one, and treating those as
        // bodyless dropped 116 real functions from the map across this tree.
        // Angle brackets need no counting of their own, because a `;` only
        // reaches the type level inside an array or a tuple and both of those
        // bring a bracket or a paren with them.
        let mut nesting = 0i64;
        let mut opened = false;
        let mut declaration = false;
        for line in &lines[i..] {
            for ch in line.chars() {
                match ch {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => depth -= 1,
                    '[' | '(' if !opened => nesting += 1,
                    ']' | ')' if !opened => nesting -= 1,
                    ';' if !opened && nesting == 0 => declaration = true,
                    _ => {}
                }
            }
            body.push_str(line);
            body.push('\n');
            if declaration || (opened && depth == 0) {
                break;
            }
        }
        if !declaration {
            out.push((name, body));
        }
    }
    out
}

/// Whether `ident` appears in `body` as a whole identifier rather than as part
/// of a longer one, so a spawning `run` is not matched by `rerun` or `run_id`.
fn mentions_ident(body: &str, ident: &str) -> bool {
    let bytes = body.as_bytes();
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    body.match_indices(ident).any(|(at, _)| {
        let before = at == 0 || !word(bytes[at - 1]);
        let end = at + ident.len();
        let after = end == bytes.len() || !word(bytes[end]);
        before && after
    })
}

/// The names of every function in `masked` that reaches `std::process`, either
/// directly or through another function in the same file.
///
/// The fixed point is what makes this useful rather than decorative: the eleven
/// tests issue #714 found call `cells()`, which calls `graphs()`, which calls
/// `Command::new(cargo())`. One hop would still see them all as pure.
fn process_spawning_fns(masked: &str) -> BTreeSet<String> {
    let bodies = fn_bodies(masked);
    let mut spawning: BTreeSet<String> = bodies
        .iter()
        .filter(|(_, body)| PROCESS_MARKERS.iter().any(|m| body.contains(m)))
        .map(|(name, _)| name.clone())
        .collect();
    loop {
        let mut grew = false;
        for (name, body) in &bodies {
            if spawning.contains(name) {
                continue;
            }
            if spawning.iter().any(|callee| mentions_ident(body, callee)) {
                spawning.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            return spawning;
        }
    }
}

/// Parse every `#[test]` function out of one file.
///
/// The parse asserts its own shape as it goes, which is the point: the failure
/// this guard is guarding against is a scanner that quietly stops seeing part
/// of a file. Every one of these is a hard panic naming the file and line.
///
/// 1. Masking must preserve length, so line numbers mean what they say.
/// 2. Braces must balance over the whole masked file, which is an end-to-end
///    check that the masker handled every literal and comment in it.
/// 3. Every `#[test]` must be followed by attributes and then a `fn` header.
/// 4. Every parsed body must open and close before end of file.
/// 5. The number of functions parsed must equal a naive count of `#[test]`
///    lines, so nothing was skipped.
/// 6. The number of tests carrying `#[cfg_attr(miri, ignore)]` must equal a
///    naive count of `cfg_attr(miri` lines, so no annotation sits somewhere the
///    parse does not look (above a multi-line attribute, on a `mod`, ...).
/// 7. No `#[test]` spelling this scanner does not understand may appear.
fn scan_source(rel: &str, src: &str) -> Vec<TestFn> {
    let masked = mask_literals_and_comments(src);
    assert_eq!(
        masked.chars().count(),
        src.chars().count(),
        "{rel}: masking changed the character count, so line numbers no longer line up"
    );

    for bad in UNSUPPORTED_TEST_ATTRS {
        assert!(
            !masked.contains(bad),
            "{rel}: found a test attribute spelled `{bad}`, which this scanner does not \
             understand. Teach `tests/miri_ignore_convention.rs` about it rather than \
             letting it skip the test."
        );
    }

    let spawning = process_spawning_fns(&masked);
    let lines: Vec<&str> = masked.lines().collect();
    let raw: Vec<&str> = src.lines().collect();

    let mut depth = 0i64;
    for (idx, line) in lines.iter().enumerate() {
        for ch in line.chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        assert!(
            depth >= 0,
            "{rel}:{}: brace depth went negative, so the masker mishandled a literal or \
             comment above this line",
            idx + 1
        );
    }
    assert_eq!(
        depth, 0,
        "{rel}: braces do not balance over the whole file, so the masker mishandled a \
         literal or comment somewhere in it"
    );

    let mut found = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i].trim() != "#[test]" {
            i += 1;
            continue;
        }

        // The attribute block: the contiguous run of attribute lines around the
        // `#[test]`, so the annotation is seen whether it sits above or below.
        let mut start = i;
        while start > 0 && lines[start - 1].trim().starts_with("#[") {
            start -= 1;
        }
        let mut j = i + 1;
        while j < lines.len() && lines[j].trim().starts_with("#[") {
            j += 1;
        }
        assert!(
            j < lines.len(),
            "{rel}:{}: `#[test]` runs to end of file with no function after it",
            i + 1
        );
        let name = function_name(lines[j]).unwrap_or_else(|| {
            panic!(
                "{rel}:{}: expected a `fn` header after `#[test]`, found `{}`. Either the \
                 attribute block uses a form this scanner does not parse, or the file layout \
                 changed; teach `tests/miri_ignore_convention.rs` about it.",
                j + 1,
                raw.get(j).unwrap_or(&"").trim()
            )
        });

        let annotated = lines[start..j]
            .iter()
            .any(|l| l.contains("cfg_attr(miri") && l.contains("ignore"));

        // The body, by brace matching over the masked text.
        let mut body = String::new();
        let mut body_depth = 0i64;
        let mut opened = false;
        let mut k = j;
        while k < lines.len() {
            for ch in lines[k].chars() {
                match ch {
                    '{' => {
                        body_depth += 1;
                        opened = true;
                    }
                    '}' => body_depth -= 1,
                    _ => {}
                }
            }
            body.push_str(lines[k]);
            body.push('\n');
            if opened && body_depth == 0 {
                break;
            }
            k += 1;
        }
        assert!(
            opened && body_depth == 0,
            "{rel}:{}: the body of `{name}` never closed before end of file",
            j + 1
        );

        let touches_fs = FS_MARKERS.iter().any(|m| body.contains(m));
        let spawns_process = spawning.contains(&name);
        found.push(TestFn {
            file: rel.to_string(),
            name,
            annotated,
            touches_fs,
            spawns_process,
        });
        i = k + 1;
    }

    let naive_tests = lines.iter().filter(|l| l.trim() == "#[test]").count();
    assert_eq!(
        naive_tests,
        found.len(),
        "{rel}: found {naive_tests} `#[test]` lines but parsed {} test functions, so the \
         parse is skipping some",
        found.len()
    );

    let naive_annotations = lines.iter().filter(|l| l.contains("cfg_attr(miri")).count();
    let parsed_annotations = found.iter().filter(|t| t.annotated).count();
    assert_eq!(
        naive_annotations, parsed_annotations,
        "{rel}: {naive_annotations} lines carry `cfg_attr(miri` but only {parsed_annotations} \
         of them landed on a parsed test. An annotation outside a `#[test]` attribute block \
         is invisible to this guard, so either move it or teach the guard about it."
    );

    found
}

/// Every `.rs` file under `dir`, recursively, as repo-relative `/`-separated
/// paths. A walk, deliberately: a hand-listed set is exactly how this kind of
/// guard stops seeing new modules.
fn rs_files_under(dir: &Path, rel_prefix: &str, out: &mut Vec<(String, PathBuf)>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut names: Vec<PathBuf> = entries
        .map(|e| e.expect("cannot read a directory entry").path())
        .collect();
    names.sort();
    for path in names {
        let name = path
            .file_name()
            .expect("directory entry with no file name")
            .to_str()
            .unwrap_or_else(|| panic!("non-UTF-8 path under {}", dir.display()))
            .to_string();
        let rel = if rel_prefix.is_empty() {
            name.clone()
        } else {
            format!("{rel_prefix}/{name}")
        };
        if path.is_dir() {
            rs_files_under(&path, &rel, out);
        } else if name.ends_with(".rs") {
            out.push((rel, path));
        }
    }
}

/// Walk both roots and parse everything.
fn scan_repo() -> Vec<TestFn> {
    let root = repo_root();
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for dir in SCANNED_DIRS {
        let path = root.join(dir);
        assert!(
            path.is_dir(),
            "{} is not a directory, so the scan is rooted wrongly",
            path.display()
        );
        rs_files_under(&path, dir, &mut files);
    }

    assert!(
        files.len() >= MIN_FILES,
        "the walk found only {} `.rs` files under {SCANNED_DIRS:?}, below the floor of \
         {MIN_FILES}. The scan is rooted wrongly or the recursion broke; it is not that the \
         repository shrank by half.",
        files.len()
    );
    let seen: BTreeSet<&str> = files.iter().map(|(rel, _)| rel.as_str()).collect();
    for anchor in ANCHOR_FILES {
        assert!(
            seen.contains(anchor),
            "the walk did not reach `{anchor}`, so it is not seeing the real source tree"
        );
    }

    let mut tests = Vec::new();
    for (rel, path) in &files {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        tests.extend(scan_source(rel, &src));
    }

    assert!(
        tests.len() >= MIN_TESTS,
        "the scan parsed only {} `#[test]` functions, below the floor of {MIN_TESTS}. \
         Something stopped the parse, it is not that the suite shrank.",
        tests.len()
    );

    tests.sort();
    tests
}

/// The inventory as a set of canonical lines, ignoring blanks and `#` comments.
fn inventory() -> BTreeSet<String> {
    INVENTORY
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| {
            let mut it = l.split_whitespace();
            let ann = it.next().expect("inventory line with no status column");
            let det = it.next().expect("inventory line with no detection column");
            let path = it.next().expect("inventory line with no test path");
            assert!(
                it.next().is_none(),
                "inventory line has more than three columns: `{l}`"
            );
            assert!(
                ann == "annotated" || ann == "unannotated",
                "inventory line has an unknown status `{ann}`: `{l}`"
            );
            assert!(
                det == "fs-detected" || det == "not-detected",
                "inventory line has an unknown detection column `{det}`: `{l}`"
            );
            format!("{ann:<11} {det:<12} {path}")
        })
        .collect()
}

/// The whole point: the set of filesystem-touching-or-annotated tests in the
/// tree must be exactly the set recorded in the inventory, annotation state
/// included.
///
/// This is the assertion that catches a new `tempfile::tempdir()` test arriving
/// without `#[cfg_attr(miri, ignore)]`, and equally catches an existing
/// annotation being deleted. Both show up as a set difference, and both name
/// the offending test.
#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn filesystem_touching_tests_match_the_recorded_inventory() {
    let live: BTreeSet<String> = scan_repo()
        .iter()
        .filter(|t| t.is_tracked())
        .map(TestFn::ledger_line)
        .collect();
    let recorded = inventory();

    if live == recorded {
        return;
    }

    let mut msg = String::new();
    msg.push_str(
        "\ntests/miri_fs_test_inventory.txt no longer describes the tree.\n\n\
         Every test listed there either touches the filesystem or carries \
         `#[cfg_attr(miri, ignore)]`. Miri aborts the whole run on the first filesystem \
         call it refuses, so an unannotated filesystem test takes down the merge gate and \
         reports as a Miri failure rather than as a missing annotation (issue #652).\n\n",
    );

    let added: Vec<&String> = live.difference(&recorded).collect();
    if !added.is_empty() {
        let _ = writeln!(
            msg,
            "{} test(s) in the tree are not recorded as written:",
            added.len()
        );
        for line in &added {
            let _ = writeln!(msg, "  + {line}");
        }
        msg.push_str(
            "\nIf a `+ unannotated` line is new, add `#[cfg_attr(miri, ignore)]` to that \
             test. Only record it as `unannotated` if it genuinely has to run under Miri, \
             and say why in the review.\n\n",
        );
    }

    let removed: Vec<&String> = recorded.difference(&live).collect();
    if !removed.is_empty() {
        let _ = writeln!(
            msg,
            "{} recorded test(s) no longer look like that:",
            removed.len()
        );
        for line in &removed {
            let _ = writeln!(msg, "  - {line}");
        }
        msg.push_str(
            "\nA `- annotated` line paired with a `+ unannotated` line for the same test \
             means somebody deleted the annotation.\n\n",
        );
    }

    msg.push_str("Regenerated inventory body:\n");
    for line in &live {
        let _ = writeln!(msg, "{line}");
    }

    panic!("{msg}");
}

/// The size of the annotated set, pinned so a bulk move in either direction is
/// a deliberate edit here rather than a number that drifts.
///
/// This used to end with `assert!(unannotated_fs > 0)`, guarding the claim that
/// the inventory was a ledger of a known gap and had to keep saying how big the
/// gap was. That assertion demanded the gap exist, so it would have gone red on
/// the change that cleared it, and its own message said as much. The claim now
/// lives in
/// [`no_filesystem_touching_test_runs_under_miri_outside_the_named_exceptions`],
/// which reads the same set and asserts the stronger, opposite thing.
#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn the_annotated_set_stays_the_size_it_is_documented_to_be() {
    let tests = scan_repo();
    let annotated: Vec<&TestFn> = tests.iter().filter(|t| t.annotated).collect();
    let src_annotated: Vec<&&TestFn> = annotated
        .iter()
        .filter(|t| t.file.starts_with("src/"))
        .collect();
    let modules: BTreeSet<&str> = src_annotated.iter().map(|t| t.file.as_str()).collect();

    assert_eq!(
        src_annotated.len(),
        EXPECTED_SRC_ANNOTATIONS,
        "`src/` carries {} annotated tests, not {EXPECTED_SRC_ANNOTATIONS}. Adding or \
         removing one is fine, but update `EXPECTED_SRC_ANNOTATIONS` and the count quoted \
         in `merge-gate.yml` with it. Modules involved: {modules:?}",
        src_annotated.len()
    );
    assert_eq!(
        modules.len(),
        EXPECTED_SRC_MODULES,
        "expected {EXPECTED_SRC_MODULES} annotated modules under `src/`, found {modules:?}"
    );
}

/// Every test that reaches `std::process` must be ignored under Miri. This is
/// an assertion rather than an inventory row, and the difference is the point.
///
/// The filesystem rows were a ledger because `-Zmiri-disable-isolation` let an
/// unannotated filesystem test run. Nothing has ever let a process spawn run:
/// Miri supports it on no target and under no flag, so the first one it reaches
/// ends the whole session with `unsupported operation: can't call foreign
/// function \`fork\`` and reports as a Miri failure rather than as a missing
/// annotation. Measured on `120acb6` with `nightly-2026-08-20`,
/// `cargo miri test --test dependency_policy` died on
/// `every_links_key_is_on_the_allowlist` having run nothing else (issue #714).
///
/// Since #711 turned isolation on the filesystem class aborts too, and #739
/// annotated it, so both classes are assertions now and the asymmetry this
/// check was built around has mostly closed. What is left of it is the reason
/// they are still two checks: the filesystem one carries a named exception list
/// ([`UNANNOTATED_FS_EXCEPTIONS`]), because a filesystem test that runs is fatal
/// only under isolation and isolation is a flag somebody could argue about,
/// while a spawning test is fatal under every flag Miri has. An exception to
/// this one would not mean anything.
#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn no_process_spawning_test_can_run_under_miri() {
    let tests = scan_repo();
    let offenders: Vec<String> = tests
        .iter()
        .filter(|t| t.spawns_process && !t.annotated)
        .map(|t| format!("  {}::{}", t.file, t.name))
        .collect();
    assert!(
        offenders.is_empty(),
        "{} test(s) reach `std::process` without `#[cfg_attr(miri, ignore)]`:\n{}\n\n\
         Any one of them ends the whole Miri run on the first `fork`, whatever \
         `MIRIFLAGS` says, so the annotation is not optional here the way it is for \
         the filesystem rows (issue #714). If a test genuinely must not carry it, the \
         fix is to stop it spawning, not to widen this check.",
        offenders.len(),
        offenders.join("\n")
    );

    // The positive control. Everything above is an assertion that a set is
    // empty, and a broken detector produces an empty set too.
    let spawning: Vec<String> = tests
        .iter()
        .filter(|t| t.spawns_process)
        .map(|t| format!("  {}::{}", t.file, t.name))
        .collect();
    assert_eq!(
        spawning.len(),
        EXPECTED_PROCESS_SPAWNING_TESTS,
        "the detector found {} process-spawning tests, not \
         {EXPECTED_PROCESS_SPAWNING_TESTS}. Adding or removing one is fine, but move \
         the constant in the same change, because the check above is satisfied by a \
         detector that has stopped finding anything at all. Found:\n{}",
        spawning.len(),
        spawning.join("\n")
    );
}

/// Every filesystem-touching test must be ignored under Miri, bar the ones
/// named in [`UNANNOTATED_FS_EXCEPTIONS`].
///
/// This is the check #739 asked for. Under isolation Miri ends the whole
/// session on the first filesystem call it refuses, so one unannotated test is
/// not one failing test, it is the gate reporting nothing at all. Measured on
/// `800c699`: `cargo miri test --test workspace_layout` died on
/// `fuzz_crate_is_a_member_of_the_root_workspace` having run nothing.
///
/// The exception list is checked in both directions. An entry that no longer
/// names an unannotated filesystem test is as much a defect as a test missing
/// from it: a stale exception is how a list like this stops describing anything
/// and starts being decoration, and it is the failure mode that arrives by
/// itself, when somebody annotates the test and leaves the row.
#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn no_filesystem_touching_test_runs_under_miri_outside_the_named_exceptions() {
    let tests = scan_repo();
    let allowed: BTreeSet<&str> = UNANNOTATED_FS_EXCEPTIONS.iter().copied().collect();

    let live: BTreeSet<String> = tests
        .iter()
        .filter(|t| t.touches_fs && !t.annotated)
        .map(|t| format!("{}::{}", t.file, t.name))
        .collect();

    let offenders: Vec<&String> = live
        .iter()
        .filter(|k| !allowed.contains(k.as_str()))
        .collect();
    assert!(
        offenders.is_empty(),
        "{} filesystem-touching test(s) have no `#[cfg_attr(miri, ignore)]` and are not \
         named in `UNANNOTATED_FS_EXCEPTIONS`:\n{}\n\n\
         Miri runs with isolation on since #711, so the first one of these the run reaches \
         ends the whole session with `unsupported operation` and the gate reports nothing \
         at all. Add the annotation. Adding a name to the exception list instead needs a \
         reason that survives review and an issue to carry it, because the cost is the \
         whole gate rather than one test.",
        offenders.len(),
        offenders
            .iter()
            .map(|k| format!("  {k}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let stale: Vec<&&str> = UNANNOTATED_FS_EXCEPTIONS
        .iter()
        .filter(|k| !live.contains(**k))
        .collect();
    assert!(
        stale.is_empty(),
        "`UNANNOTATED_FS_EXCEPTIONS` names {} test(s) that are not unannotated \
         filesystem-touching tests any more:\n{}\n\n\
         Either they were annotated, in which case delete the entry and close the issue it \
         carries, or they were renamed or deleted. An exception list nobody prunes stops \
         describing the tree and starts excusing whatever happens to match it.",
        stale.len(),
        stale
            .iter()
            .map(|k| format!("  {k}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // The positive control. An empty offender list is what a working detector
    // produces and also what a detector that has stopped recognising filesystem
    // calls produces. The stale check catches some of that by accident, since
    // an exception that stops being detected fires it, but only the four names
    // it happens to cover. This pins the whole population.
    let touching = tests.iter().filter(|t| t.touches_fs).count();
    assert_eq!(
        touching,
        EXPECTED_FS_TOUCHING_TESTS,
        "the detector found {touching} filesystem-touching tests, not \
         {EXPECTED_FS_TOUCHING_TESTS}. Adding or removing one is fine, but move the \
         constant in the same change, because the assertion above is satisfied by a \
         detector that has stopped recognising a filesystem call at all."
    );
}

/// The call-following half of the process detector, pinned on its own source
/// rather than on the tree, so it says what the classifier does instead of what
/// the tree happens to contain.
///
/// The three cells are the three cases that decide whether #714 stays fixed: a
/// direct spawn, a spawn two hops down a helper chain (which is the shape all
/// eleven of #714's tests have), and a test that touches neither.
#[test]
fn the_detector_follows_a_spawn_through_helpers() {
    let src = r#"
fn runs_cargo() -> String {
    let out = Command::new("cargo").output().unwrap();
    String::from_utf8(out.stdout).unwrap()
}

fn graph() -> String {
    runs_cargo()
}

fn passes_it_as_a_reference() -> Vec<String> {
    [0usize].iter().map(runs_cargo_of).collect()
}

fn runs_cargo_of(_: &usize) -> String {
    runs_cargo()
}

fn pure_helper() -> usize {
    41 + 1
}

fn rerun_counter() -> usize {
    7
}

mod tests {
    #[test]
    fn spawns_directly() {
        let _ = Command::new("true").status();
    }

    #[test]
    fn spawns_two_hops_down() {
        assert!(!graph().is_empty());
    }

    #[test]
    fn mentions_command_new_in_a_string_only() {
        let msg = "Command::new("cargo") failed";
        assert!(msg.contains("failed"));
    }

    #[test]
    fn takes_the_helper_as_a_function_reference() {
        assert!(!passes_it_as_a_reference().is_empty());
    }

    #[test]
    fn calls_only_a_pure_helper() {
        assert_eq!(pure_helper() + rerun_counter(), 49);
    }
}
"#;
    let found = scan_source("fixture.rs", src);
    let by_name: Vec<(&str, bool)> = found
        .iter()
        .map(|t| (t.name.as_str(), t.spawns_process))
        .collect();
    assert_eq!(
        by_name,
        vec![
            ("spawns_directly", true),
            ("spawns_two_hops_down", true),
            ("mentions_command_new_in_a_string_only", false),
            ("takes_the_helper_as_a_function_reference", true),
            ("calls_only_a_pure_helper", false),
        ],
        "the process detector must follow calls within the file, including one passed \
         as a bare function reference, and must not fire on a marker that only appears \
         inside a string literal"
    );
}

/// The three shapes the module docs list as invisible, pinned as misses.
///
/// A limitation nobody can reproduce is a limitation nobody believes, and one
/// that gets quietly fixed without the docs following is worse. Each cell here
/// reaches `std::process` and each comes back `false`. If one ever flips, that
/// is good news and this check is where you find out, so move it up into
/// [`the_detector_follows_a_spawn_through_helpers`] and delete the bullet.
///
/// The fourth shape from that list, a helper in another file, cannot be
/// written as a single-file fixture, which is the whole reason it is missed.
#[test]
fn the_documented_blind_spots_are_still_blind() {
    let cases = [
        (
            "an aliased import",
            r#"
use std::process::Command as Cmd;

fn runs() -> bool {
    Cmd::new("true").status().is_ok()
}

mod tests {
    #[test]
    fn spawns_through_an_alias() {
        assert!(runs());
    }
}
"#,
        ),
        (
            "a spawn inside a macro body",
            r#"
macro_rules! run_it {
    () => {
        Command::new("true").status()
    };
}

mod tests {
    #[test]
    fn spawns_through_a_macro() {
        let _ = run_it!();
    }
}
"#,
        ),
        (
            "a closure in a static",
            r#"
static RUNNER: fn() -> bool = || Command::new("true").status().is_ok();

mod tests {
    #[test]
    fn spawns_through_a_static_closure() {
        assert!(RUNNER());
    }
}
"#,
        ),
    ];
    for (what, src) in cases {
        let found = scan_source("fixture.rs", src);
        assert_eq!(found.len(), 1, "{what}: expected one test, got {found:?}");
        assert!(
            !found[0].spawns_process,
            "{what} is now detected, which is an improvement. Move this cell into \
             the helper-following check and drop the bullet from the module docs, \
             so the two do not disagree."
        );
    }
}

/// A `;` inside the signature's brackets is a type, not the end of a
/// declaration, and reading it as one drops the whole function from the map.
///
/// This is the sharp edge of the declaration skip, and it is a silent
/// *under*-approximation, which is the direction that costs the gate. Measured
/// across `src/` and `tests/` on the tree that introduced it, the naive `;`
/// test dropped 133 function headers where the depth-aware one drops 17: 116
/// real functions were invisible to the call graph. None of the 116 spawns
/// today, so nothing was actually missed, and that is luck rather than design.
///
/// Both spellings below are in this tree. `-> [u8; 32]` is the shape
/// `src/checksum.rs` uses, and a `where` clause carrying an array type is the
/// other way a `;` reaches a signature.
#[test]
fn a_semicolon_inside_a_signature_type_does_not_drop_the_function() {
    let src = r#"
fn fingerprint() -> [u8; 32] {
    let _ = Command::new("true").status();
    [0u8; 32]
}

fn constrained<T>(_x: T) -> usize
where
    T: Into<[u8; 4]>,
{
    let _ = Command::new("true").status();
    4
}

mod tests {
    #[test]
    fn calls_fingerprint() {
        assert_eq!(fingerprint().len(), 32);
    }

    #[test]
    fn calls_constrained() {
        assert_eq!(constrained([0u8; 4]), 4);
    }
}
"#;
    let masked = mask_literals_and_comments(src);
    let parsed: BTreeSet<String> = fn_bodies(&masked).into_iter().map(|(n, _)| n).collect();
    for want in ["fingerprint", "constrained"] {
        assert!(
            parsed.contains(want),
            "`{want}` must reach the fn map; its signature carries a `;` inside \
             brackets, not a bodyless declaration. Parsed: {parsed:?}"
        );
    }
    assert_eq!(
        scan_source("fixture.rs", src)
            .iter()
            .map(|t| (t.name.as_str(), t.spawns_process))
            .collect::<Vec<_>>(),
        vec![("calls_fingerprint", true), ("calls_constrained", true)],
        "and both callers must therefore be seen as spawning"
    );
}

/// A trait method declaration has no body, and consuming forward from one runs
/// into whatever comes next. That would file the *next* function's spawn under
/// the declaration's name, and every caller of the trait method would then be
/// flagged for a spawn it never makes.
///
/// The parser starts a fresh body at every `fn` header, so nothing is ever
/// hidden by this; the cost is entirely spurious annotations. This pins that it
/// does not happen, and it is the check that caught my first attempt at the
/// fixture, which stayed green with the skip removed because the declaration
/// and the function it swallowed shared a name.
#[test]
fn a_bodyless_declaration_does_not_borrow_the_next_function_s_spawn() {
    let src = r#"
trait Runner {
    fn run(&self) -> String;
}

fn spawns() -> String {
    let _ = Command::new("true").status();
    String::new()
}

struct R;

impl Runner for R {
    fn run(&self) -> String {
        String::new()
    }
}

mod tests {
    #[test]
    fn calls_only_the_trait_method() {
        assert!(R.run().is_empty());
    }
}
"#;
    let found = scan_source("fixture.rs", src);
    assert_eq!(
        found
            .iter()
            .map(|t| (t.name.as_str(), t.spawns_process))
            .collect::<Vec<_>>(),
        vec![("calls_only_the_trait_method", false)],
        "the declaration above `spawns` must not take its `Command::new` with it"
    );
}

/// The scanner has to be able to tell a real filesystem call from a mention of
/// one, or the inventory is noise. These are self-contained: they run the
/// detector over source text written here rather than over the tree, so they
/// pin the classifier without depending on what any module happens to contain.
#[test]
fn the_detector_distinguishes_calls_from_mentions() {
    let src = r#"
mod tests {
    /// Writes a tile with `std::fs::write` and checks it.
    #[test]
    fn mentions_fs_only_in_a_doc_comment() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn reports_a_path_in_an_error_message() {
        let msg = "std::fs::write failed for tempfile::tempdir()";
        assert!(msg.contains("failed"));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn really_writes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), b"x").unwrap();
    }

    #[test]
    fn really_writes_but_forgot_the_annotation() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), b"x").unwrap();
    }
}
"#;
    let found = scan_source("fixture.rs", src);
    let by_name: Vec<(&str, bool, bool)> = found
        .iter()
        .map(|t| (t.name.as_str(), t.touches_fs, t.annotated))
        .collect();
    assert_eq!(
        by_name,
        vec![
            ("mentions_fs_only_in_a_doc_comment", false, false),
            ("reports_a_path_in_an_error_message", false, false),
            ("really_writes", true, true),
            ("really_writes_but_forgot_the_annotation", true, false),
        ],
        "the detector must ignore comments and string literals and must read the \
         annotation off the attribute block"
    );
}

/// The annotation is found whether it sits above or below `#[test]`, because
/// both orders compile and both are things a contributor will write.
#[test]
fn the_detector_reads_the_annotation_in_either_order() {
    let src = r#"
mod tests {
    #[cfg_attr(miri, ignore)]
    #[test]
    fn annotation_above() {
        let _ = tempfile::tempdir();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn annotation_below() {
        let _ = tempfile::tempdir();
    }
}
"#;
    let found = scan_source("fixture.rs", src);
    assert_eq!(found.len(), 2, "both tests must be parsed, got {found:?}");
    assert!(
        found.iter().all(|t| t.annotated && t.touches_fs),
        "both orders must read as annotated, got {found:?}"
    );
}

/// A body containing braces inside string and character literals must not
/// desynchronise the scan, which is the failure that would silently shrink the
/// window the detector looks at.
#[test]
fn the_body_scan_survives_braces_in_literals() {
    let src = "
mod tests {
    #[test]
    fn braces_in_literals() {
        let a = \"{{{\";
        let b = '}';
        let c = r#\"} } }\"#;
        let _ = (a, b, c);
    }

    #[test]
    fn after_the_tricky_one() {
        let _ = std::fs::read_dir(\".\");
    }
}
";
    let found = scan_source("fixture.rs", src);
    assert_eq!(
        found.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        vec!["braces_in_literals", "after_the_tricky_one"],
        "a literal full of braces must not swallow the tests after it"
    );
    assert!(!found[0].touches_fs, "literals are not filesystem calls");
    assert!(found[1].touches_fs, "`std::fs::read_dir` is");
}
