//! Every path an `include_bytes!` names must be in git's index, spelled the
//! same way (issue #979).
//!
//! `main` did not compile on Linux for two months. `src/mat.rs` and
//! `src/source.rs` embedded
//! `oracle-captures/foreign-mat/fixtures/magic_MATLAB_50.mat`, a file that was
//! never committed, and every job that builds the crate's own tests failed on
//! the same `couldn't read` error. Issue #977 has the cause: a capture script
//! derived two fixture names that differed only in case, so on the
//! case-insensitive filesystem it ran on the second write landed on the first
//! one's file and only one of the two names reached the tree.
//!
//! #977 fixed that capture script and added
//! `tests/case_only_path_collisions.rs`, which fails if two tracked paths ever
//! differ only in case again. That guard is worth having and it would not have
//! caught this: at the moment `main` was broken there was no collision in the
//! index at all, just a name nothing had committed. This is the guard that
//! sees the failure itself rather than one way of causing it.
//!
//! # Why the host cannot answer this and the index can
//!
//! `include_bytes!` resolves through the filesystem, so on macOS or Windows it
//! finds `magic_matlab_50.mat` when the source asks for `magic_MATLAB_50.mat`
//! and the build succeeds. Every local gate inherits that, including the
//! containerised one: `tools/local-ci.py` hands the tree to a Linux container
//! as a bind mount, and a Docker Desktop bind mount off an APFS host is still
//! case-insensitive. So the authoritative local gate is structurally unable to
//! see this bug class, and was green throughout.
//!
//! Git's index is not. It stores the byte string, so it is the same on every
//! host, and asking it is what makes this check mean the same thing on the
//! machine that writes the fixture and the machine that builds it.
//!
//! # How it reads the sources, and how it fails rather than shrinks
//!
//! The paths come out of a scan of `src/` and `tests/` in two passes over the
//! same text. The first resolves the forms this crate actually writes:
//!
//! * `include_bytes!("relative/path")`, against the including file's directory;
//! * a `macro_rules!` whose body is
//!   `include_bytes!(concat!(<prefix parts>, $meta))`, where the prefix parts
//!   are string literals and optionally `env!("CARGO_MANIFEST_DIR")`, together
//!   with that macro's `name!("leaf")` and `name![...]` call sites in the same
//!   file;
//! * one level of nesting: an outer `macro_rules!` taking a `$stem:literal`
//!   whose body calls `inner!(concat!($stem, ".ext"))`, together with that
//!   outer macro's own call sites. `src/analyze.rs`'s `decoded!` and
//!   `refused!` are the reason that arm exists.
//!
//! The second pass is what keeps the first honest. It looks for the token
//! `include_bytes!` in the *masked* source, where comments and literal
//! contents are blanked but every byte keeps its offset, so only real code
//! survives. Every offset it finds must be one the first pass claimed. A form
//! nobody taught this scanner therefore fails the test naming its file and
//! line, instead of being quietly skipped and taking the guard's coverage down
//! with it. That is the failure mode `tests/unsafe_inventory.rs` hit in #943,
//! where a masking bug lost 2650 lines of `src/jp2k.rs` while looking green.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/scan.rs"]
mod scan;

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every path git tracks, relative to the repo root, as a set.
///
/// Anchored on paths that are certainly tracked, because an empty listing
/// would make every check below vacuously true. `tests/oracle_capture_pins.rs`
/// and `tests/case_only_path_collisions.rs` anchor their own `git ls-files`
/// the same way.
fn tracked() -> BTreeSet<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["ls-files", "-z", "--full-name"])
        .output()
        .expect("failed to spawn git ls-files");
    assert!(
        out.status.success(),
        "git ls-files failed, so this guard has nothing to check:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let files: BTreeSet<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
        .collect();
    for anchor in [
        "src/mat.rs",
        "src/source.rs",
        "oracle-captures/foreign-mat/fixtures/base_2x3_uint8.mat",
    ] {
        assert!(
            files.contains(anchor),
            "git tracks {} paths and {anchor} is not one of them, so this is \
             not the listing the guard means to read",
            files.len()
        );
    }
    files
}

/// The string literal starting at `at`, which must be an opening `"`, plus the
/// offset just past its closing quote. `None` if it never closes.
fn string_at(b: &[u8], at: usize) -> Option<(String, usize)> {
    if b.get(at) != Some(&b'"') {
        return None;
    }
    let mut i = at + 1;
    let mut s = String::new();
    while i < b.len() {
        match b[i] {
            b'\\' => {
                // No fixture path in this crate needs an escape, and reading one
                // as a literal backslash would silently produce a path that is
                // not the one rustc opens. Refuse instead.
                return None;
            }
            b'"' => return Some((s, i + 1)),
            c => {
                s.push(c as char);
                i += 1;
            }
        }
    }
    None
}

/// Skip spaces, tabs, newlines and commas from `at`.
fn skip_ws(b: &[u8], mut at: usize) -> usize {
    while at < b.len() && (b[at] as char).is_whitespace() {
        at += 1;
    }
    at
}

/// The directory an `include_bytes!` in `file` resolves relative paths against.
fn file_dir(file: &Path) -> &Path {
    file.parent().expect("a source file has a parent directory")
}

/// Normalise `dir` joined with `leaf` to a repo-relative path with `/`.
fn resolve(dir: &Path, leaf: &str) -> String {
    let joined = dir.join(leaf);
    let mut parts: Vec<String> = Vec::new();
    for c in joined.components() {
        match c {
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::CurDir => {}
            other => parts.push(other.as_os_str().to_string_lossy().into_owned()),
        }
    }
    let abs = PathBuf::from("/".to_string() + &parts.join("/"));
    let root: PathBuf = {
        let mut p = Vec::new();
        for c in repo_root().components() {
            match c {
                std::path::Component::ParentDir => {
                    p.pop();
                }
                std::path::Component::CurDir => {}
                other => p.push(other.as_os_str().to_string_lossy().into_owned()),
            }
        }
        PathBuf::from("/".to_string() + &p.join("/"))
    };
    abs.strip_prefix(&root)
        .unwrap_or(&abs)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Every repo-relative `include_bytes!` target in one file, and the byte
/// offsets in that file the first pass claimed.
///
/// `raw` keeps its literals so the paths can be read; `masked` is the same
/// bytes with comments and literal contents blanked, and is only ever used to
/// find where real code sits.
fn embeds_in(raw: &str, masked: &str, file: &Path) -> (Vec<String>, BTreeSet<usize>) {
    let rb = raw.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut claimed: BTreeSet<usize> = BTreeSet::new();

    // Pass 1a: `include_bytes!("leaf")`, and the `concat!` prefixes that
    // `macro_rules!` bodies wrap around a metavariable.
    let mut prefixes: BTreeMap<String, String> = BTreeMap::new();
    for (at, _) in raw.match_indices("include_bytes!") {
        if masked.as_bytes()[at] != b'i' {
            continue; // inside a comment or a literal
        }
        let after = skip_ws(rb, at + "include_bytes!".len());
        if rb.get(after) != Some(&b'(') {
            panic!(
                "{}: include_bytes! at byte {at} is not followed by `(`, which \
                 this scanner does not understand",
                file.display()
            );
        }
        let inner = skip_ws(rb, after + 1);
        if let Some((leaf, _)) = string_at(rb, inner) {
            out.push(resolve(file_dir(file), &leaf));
            claimed.insert(at);
            continue;
        }
        if raw[inner..].starts_with("concat!") {
            // Collect the leading constant parts of the concat: string
            // literals and `env!("CARGO_MANIFEST_DIR")`. Anything else ends
            // the prefix, and what follows is assumed to be the metavariable.
            let mut i = skip_ws(rb, inner + "concat!".len());
            if rb.get(i) != Some(&b'(') {
                panic!("{}: concat! at byte {inner} is not a call", file.display());
            }
            i = skip_ws(rb, i + 1);
            let mut prefix = String::new();
            loop {
                if raw[i..].starts_with("env!(\"CARGO_MANIFEST_DIR\")") {
                    prefix.push_str(&repo_root().to_string_lossy());
                    i = skip_ws(rb, i + "env!(\"CARGO_MANIFEST_DIR\")".len());
                } else if let Some((lit, next)) = string_at(rb, i) {
                    prefix.push_str(&lit);
                    i = skip_ws(rb, next);
                } else {
                    break;
                }
                if rb.get(i) == Some(&b',') {
                    i = skip_ws(rb, i + 1);
                }
            }
            if rb.get(i) == Some(&b')') {
                // Every part was constant, so the concat! is the whole path
                // and no macro fills anything in. `src/source.rs` splits the
                // NIfTI paths over two literals purely to fit the line.
                out.push(resolve(file_dir(file), &prefix));
                claimed.insert(at);
                continue;
            }
            // The enclosing `macro_rules!` name, if this body is one. Searching
            // backwards is enough because a fixture macro's body is the only
            // thing between its header and its `include_bytes!`.
            let name = raw[..at]
                .rfind("macro_rules!")
                .map(|m| {
                    raw[m + "macro_rules!".len()..]
                        .trim_start()
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("")
                        .to_string()
                })
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| {
                    panic!(
                        "{}: the concat! include_bytes! at byte {at} is not \
                         inside a macro_rules!, so nothing tells this scanner \
                         what fills the rest of the path",
                        file.display()
                    )
                });
            prefixes.insert(name, prefix);
            claimed.insert(at);
            continue;
        }
        panic!(
            "{}: include_bytes! at byte {at} takes neither a string literal \
             nor a concat!, which this scanner does not understand",
            file.display()
        );
    }

    // Pass 1b: the call sites of every prefix macro found above.
    for (name, prefix) in &prefixes {
        let base: PathBuf = if Path::new(prefix).is_absolute() {
            PathBuf::from(prefix)
        } else {
            file_dir(file).join(prefix)
        };
        let bang = format!("{name}!");
        for (at, _) in raw.match_indices(&bang) {
            if masked.as_bytes()[at] != bang.as_bytes()[0] {
                continue;
            }
            let open = skip_ws(rb, at + bang.len());
            match rb.get(open) {
                // `name!("leaf")`
                Some(b'(') => {
                    let i = skip_ws(rb, open + 1);
                    if let Some((leaf, _)) = string_at(rb, i) {
                        out.push(resolve(&base, &leaf));
                    }
                    // `name!(concat!($stem, ".ext"))` is resolved in pass 1c,
                    // from the outer macro's call sites, so it is not an error
                    // to find no literal here.
                }
                // `name![ "a", "b", ... ]`
                Some(b'[') => {
                    let mut i = skip_ws(rb, open + 1);
                    while let Some((leaf, next)) = string_at(rb, i) {
                        out.push(resolve(&base, &leaf));
                        i = skip_ws(rb, next);
                        if rb.get(i) == Some(&b',') {
                            i = skip_ws(rb, i + 1);
                        }
                    }
                }
                _ => {}
            }
        }

        // Pass 1c: one level of nesting. An outer macro taking `$stem:literal`
        // whose body calls `name!(concat!($stem, ".ext"))` contributes
        // `<stem><ext>` for each of its own call sites.
        for (m, _) in raw.match_indices("macro_rules!") {
            if masked.as_bytes()[m] != b'm' {
                continue;
            }
            let outer: String = raw[m + "macro_rules!".len()..]
                .trim_start()
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("")
                .to_string();
            if outer.is_empty() || outer == *name {
                continue;
            }
            // The body runs to the next `macro_rules!` or 2 KiB, whichever is
            // sooner. Fixture macros in this crate are a dozen lines.
            let end = raw[m + 1..]
                .find("macro_rules!")
                .map(|o| m + 1 + o)
                .unwrap_or(raw.len())
                .min(m + 2048);
            let body = &raw[m..end];
            let param = match body.find(":literal") {
                Some(p) => body[..p]
                    .rfind('$')
                    .map(|d| body[d + 1..p].to_string())
                    .unwrap_or_default(),
                None => continue,
            };
            if param.is_empty() {
                continue;
            }
            let needle = format!("{name}!(concat!(${param},");
            let mut suffixes: Vec<String> = Vec::new();
            for (c, _) in body.match_indices(&needle) {
                let i = skip_ws(body.as_bytes(), c + needle.len());
                if let Some((suf, _)) = string_at(body.as_bytes(), i) {
                    suffixes.push(suf);
                }
            }
            if suffixes.is_empty() {
                continue;
            }
            let call = format!("{outer}!");
            for (at, _) in raw.match_indices(&call) {
                if masked.as_bytes()[at] != call.as_bytes()[0] {
                    continue;
                }
                let open = skip_ws(rb, at + call.len());
                if rb.get(open) != Some(&b'(') {
                    continue;
                }
                let i = skip_ws(rb, open + 1);
                if let Some((stem, _)) = string_at(rb, i) {
                    for suf in &suffixes {
                        out.push(resolve(&base, &format!("{stem}{suf}")));
                    }
                }
            }
        }
    }

    (out, claimed)
}

/// Every `include_bytes!` in `src/` and `tests/` names a file git tracks,
/// spelled exactly as the source spells it.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn every_embedded_fixture_is_committed_under_the_name_the_source_uses() {
    let tracked = tracked();
    let mut sources: Vec<(String, PathBuf)> = Vec::new();
    for dir in ["src", "tests"] {
        sources.extend(scan::rs_files_under(&repo_root().join(dir)));
    }
    assert!(
        sources.len() > 50,
        "only {} .rs files found under src/ and tests/, so the walk is not \
         seeing the tree",
        sources.len()
    );

    let mut checked = 0usize;
    let mut missing: Vec<String> = Vec::new();
    let mut saw_a_mat_fixture = false;
    for (_, path) in &sources {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if !raw.contains("include_bytes!") {
            continue;
        }
        let masked = scan::mask_literals_and_comments(&raw);
        let (embeds, claimed) = embeds_in(&raw, &masked, path);

        // Nothing in real code may be left unclaimed.
        for (at, _) in masked.match_indices("include_bytes!") {
            assert!(
                claimed.contains(&at),
                "{}: the include_bytes! at byte {at} is a form this scanner \
                 did not resolve, so it would have been skipped silently",
                path.display()
            );
        }

        for embedded in embeds {
            checked += 1;
            if embedded.starts_with("oracle-captures/foreign-mat/fixtures/") {
                saw_a_mat_fixture = true;
            }
            if !tracked.contains(&embedded) {
                missing.push(format!("{embedded} (from {})", path.display()));
            }
        }
    }

    // Positive controls on the scan itself, so an empty result cannot pass as
    // a clean one. The MAT fixtures are named because they are the ones #977
    // was about, and they are reached through the two macro forms rather than
    // through a plain literal.
    assert!(
        checked > 100,
        "only {checked} embedded paths resolved, so the scan has lost its \
         window: this crate embeds well over a hundred"
    );
    assert!(
        saw_a_mat_fixture,
        "no oracle-captures/foreign-mat fixture resolved, so the macro arms \
         this guard exists for are not being reached"
    );

    assert!(
        missing.is_empty(),
        "these paths are embedded with include_bytes! but git does not track \
         them under that spelling, so the crate compiles only where the \
         filesystem is case-insensitive or the file happens to be lying \
         around untracked: {missing:#?}"
    );
}
