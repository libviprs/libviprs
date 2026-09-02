//! No two tracked paths may differ only in case (issue #977).
//!
//! Git's index is case-sensitive and most of the filesystems this repo is
//! developed on are not. When both `a.mat` and `A.mat` are tracked, a checkout
//! on macOS or Windows writes one over the other: the working tree ends up with
//! a single file, `git status` reports the loser as modified forever, and
//! whichever content landed last wins. Nothing in the normal workflow says so.
//!
//! That is not hypothetical here. `oracle-captures/foreign-mat/capture.py`
//! built two of its fixture names by slugging a label, and `"matlab 5.0"` and
//! `"MATLAB_5.0"` both slugged to the same path once case stopped mattering. On
//! the capture host the third write reopened the second one's file, so only two
//! of three fixtures were ever committed, and the one that survived carried the
//! other one's bytes under its own name. `include_bytes!` then resolved the
//! missing name through the same case-insensitive lookup, so the crate built on
//! the machine that made the mistake and failed to compile on every Linux
//! runner. `main` was red for two months on exactly this.
//!
//! # Why this is a test rather than a naming convention
//!
//! The colliding names were machine-generated, so there was no moment when a
//! person chose them and could have known better. A convention cannot reach
//! that; a check over the index can. It is repo-wide rather than scoped to
//! `oracle-captures/` because the damage has nothing to do with what the files
//! are for: any two such paths break the same way in any directory.
//!
//! # What it does not cover
//!
//! Directories whose names differ only in case. Those merge into one directory
//! on a case-insensitive checkout, which is untidy but loses no file, so it is
//! a different (and much milder) problem than this one.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every path git tracks, relative to the repo root.
///
/// This asks git rather than walking the tree, and it has to: on a
/// case-insensitive filesystem the walk cannot see the collision at all,
/// because the whole point is that only one of the two files is there. The
/// index is the only place both names exist.
///
/// The listing is what could come back empty and take the guard down with it,
/// so it is anchored on paths that are certainly tracked before any caller
/// reads a conclusion into an absence. `tests/oracle_capture_pins.rs` anchors
/// its own `git ls-files` the same way.
fn tracked_paths() -> Vec<String> {
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
    let files: Vec<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
        .collect();

    for anchor in [
        "Cargo.toml",
        "src/lib.rs",
        "tests/case_only_path_collisions.rs",
    ] {
        assert!(
            files.iter().any(|f| f == anchor),
            "git tracks {} paths and {anchor} is not one of them, so this is \
             not the listing the guard means to read",
            files.len()
        );
    }
    files
}

/// The guard itself: lowercase every tracked path and look for a duplicate.
///
/// Comparing whole paths rather than basenames is deliberate. `a/x.mat` and
/// `A/x.mat` collide just as badly as two files in one directory, and folding
/// the whole string catches both without a second pass.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn no_two_tracked_paths_differ_only_in_case() {
    let mut by_folded: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in tracked_paths() {
        by_folded.entry(path.to_lowercase()).or_default().push(path);
    }

    let collisions: Vec<&Vec<String>> = by_folded.values().filter(|v| v.len() > 1).collect();
    assert!(
        collisions.is_empty(),
        "these tracked paths differ only in case, so a checkout on a \
         case-insensitive filesystem writes one over the other and silently \
         loses the rest: {collisions:?}"
    );
}
