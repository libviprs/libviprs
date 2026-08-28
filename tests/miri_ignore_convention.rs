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
//! * other operations Miri refuses under isolation that are not filesystem
//!   calls at all, most obviously spawning a process with `std::process`.
//! * anything outside `src/` and `tests/`, which means the `fuzz/` member (a
//!   separate crate that `cargo miri test` on this package does not build) and
//!   the `build.rs`-less root manifest.
//!
//! The `not-detected` rows are what keeps those blind spots from being silent:
//! a test the detector cannot classify still gets pinned the moment somebody
//! annotates it, so the annotation cannot later be deleted unnoticed.

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
/// `merge-gate.yml` describes the convention as "48 `#[cfg_attr(miri, ignore)]`
/// annotations across seven modules". That was true at `f62a56a` and is one
/// module out of date the moment `src/checksum.rs` adopts the convention, which
/// is what issue #652 is about. The workflow file belongs to PR #644, so the
/// correction to its comment goes with that PR, not this one.
const EXPECTED_SRC_ANNOTATIONS: usize = 51;
/// Companion to [`EXPECTED_SRC_ANNOTATIONS`]: how many `src/` modules carry at
/// least one annotation.
const EXPECTED_SRC_MODULES: usize = 8;

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
        found.push(TestFn {
            file: rel.to_string(),
            name,
            annotated,
            touches_fs,
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

/// The inventory is a ledger of a known gap, not a clean bill of health, so it
/// has to keep saying how big the gap is. This pins the annotated set against
/// the count `merge-gate.yml` quotes, so the two cannot drift apart silently
/// the way they already have once.
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

    let unannotated_fs = tests
        .iter()
        .filter(|t| t.touches_fs && !t.annotated)
        .count();
    assert!(
        unannotated_fs > 0,
        "if this ever reaches zero the ledger has stopped being a ledger: drop the \
         `unannotated` rows and make the guard assert the convention outright"
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
