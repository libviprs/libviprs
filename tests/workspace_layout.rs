//! Guards for the cargo-workspace layout adopted in issue #151.
//!
//! The repo used to have two disjoint cargo roots: the `libviprs` package and
//! a `fuzz/` crate detached by an empty `[workspace]` table. The detached
//! crate resolved dependencies on its own, with its own lockfile, and never
//! saw the root pin of `pdfium-render` to the thread-safety fork (then a
//! `[patch.crates-io]` entry, now a direct git dependency). These tests
//! assert the unified layout: one workspace, one lockfile, and dependency
//! pins that apply to every member.

use std::path::Path;
use std::process::Command;

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Run `cargo metadata --no-deps` against the given manifest and parse it.
fn metadata_no_deps(manifest: &Path) -> serde_json::Value {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .arg("--manifest-path")
        .arg(manifest)
        .output()
        .expect("failed to spawn cargo metadata");
    assert!(
        out.status.success(),
        "cargo metadata failed for {}:\n{}",
        manifest.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("cargo metadata emitted invalid JSON")
}

/// The fuzz crate must be a member of the root workspace, not a detached
/// cargo root with its own lockfile and its own (patch-less) resolution.
#[test]
fn fuzz_crate_is_a_member_of_the_root_workspace() {
    let root = repo_root();
    let meta = metadata_no_deps(&root.join("fuzz").join("Cargo.toml"));

    let workspace_root = Path::new(meta["workspace_root"].as_str().unwrap());
    assert_eq!(
        workspace_root.canonicalize().unwrap(),
        root.canonicalize().unwrap(),
        "fuzz/Cargo.toml must resolve to the repo-root workspace, \
         not act as its own cargo root"
    );

    let names: Vec<&str> = meta["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    for expected in ["libviprs", "libviprs-fuzz"] {
        assert!(
            names.contains(&expected),
            "workspace members must include {expected}, got: {names:?}"
        );
    }
}

/// Both crates seen from the root manifest: same workspace, so the single
/// root lockfile and the single set of workspace dependency pins govern
/// them all.
#[test]
fn root_workspace_contains_both_crates() {
    let root = repo_root();
    let meta = metadata_no_deps(&root.join("Cargo.toml"));
    let members = meta["workspace_members"].as_array().unwrap();
    assert_eq!(
        members.len(),
        2,
        "expected exactly the libviprs and libviprs-fuzz members, got: {members:?}"
    );
}

/// The workspace must still pin `pdfium-render` to the libviprs fork
/// (per-call thread-safety locking), now as a direct git dependency rather
/// than a `[patch.crates-io]` entry. The workspace lockfile is the single
/// source of truth for that resolution.
#[test]
fn workspace_lockfile_pins_pdfium_render_to_the_fork() {
    let lock = std::fs::read_to_string(repo_root().join("Cargo.lock"))
        .expect("workspace Cargo.lock must exist at the repo root");

    // Find the [[package]] block for pdfium-render and inspect its source.
    let mut source_lines: Vec<&str> = Vec::new();
    let mut in_block = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            in_block = false;
        }
        if line == "name = \"pdfium-render\"" {
            in_block = true;
        }
        if in_block && line.starts_with("source = ") {
            source_lines.push(line);
        }
    }
    assert_eq!(
        source_lines.len(),
        1,
        "expected exactly one pdfium-render entry in Cargo.lock, got: {source_lines:?}"
    );
    let source = source_lines[0];
    assert!(
        source.contains("git+https://github.com/libviprs/pdfium-render.git"),
        "pdfium-render must resolve from the fork, got: {source}"
    );
    assert!(
        source.contains("per-call-thread-safety"),
        "pdfium-render must track the per-call-thread-safety branch, got: {source}"
    );
}
