//! Regression coverage for issue #149.
//!
//! The `pdfium-render` per-call thread-safety fork is a direct git
//! dependency of this crate (it used to be a `[patch.crates-io]` entry,
//! which only takes effect from the build root, so consumers that omitted
//! it silently linked the unpatched `pdfium-render 0.8.x` wrapper that is
//! documented to segfault under concurrent access in `src/streaming.rs`).
//! These tests exercise the release gate that detects a registry-sourced
//! wrapper, `scripts/audit-pdfium-source.sh`:
//!
//!   1. `audit_gate_accepts_patched_core` — this crate carries the fork, so
//!      the gate must pass. Guards against the pin being dropped/renamed.
//!   2. `audit_gate_rejects_unpatched_consumer` — a synthesized sibling that
//!      depends on `pdfium-render` directly from the registry, with no fork
//!      pin; the gate must reject it (this is the #149 defect).
//!   3. `patchless_downstream_consumer_resolves_the_fork` — a synthesized
//!      downstream crate that depends on `libviprs` itself, with NO `[patch]`
//!      table, must still resolve the fork. This is #149 part 2: the fork
//!      must propagate to consumers as an ordinary dependency edge instead
//!      of relying on build-root-only `[patch]` hygiene.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn audit_script() -> PathBuf {
    repo_root().join("scripts").join("audit-pdfium-source.sh")
}

fn run_audit(manifest_dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    let script = audit_script();
    assert!(
        script.exists(),
        "release gate missing: {} (issue #149)",
        script.display()
    );
    let mut cmd = Command::new("bash");
    cmd.arg(&script).arg(manifest_dir);
    if !extra.is_empty() {
        cmd.arg("--");
        cmd.args(extra);
    }
    cmd.output().expect("failed to spawn audit script")
}

#[test]
fn audit_gate_accepts_patched_core() {
    let out = run_audit(&repo_root(), &["--features", "pdfium"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "audit gate rejected the patched core crate.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("github.com/libviprs/pdfium-render"),
        "audit gate did not confirm the libviprs fork source.\nstdout: {stdout}"
    );
}

#[test]
fn audit_gate_rejects_unpatched_consumer() {
    // A minimal consumer that depends on pdfium-render the way the sibling
    // crates do today: a plain registry dependency with no [patch] fork.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\n\
         name = \"vip149-unpatched-consumer\"\n\
         version = \"0.0.0\"\n\
         edition = \"2021\"\n\
         \n\
         [dependencies]\n\
         pdfium-render = { version = \"0.8\", features = [\"sync\"] }\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src").join("lib.rs"), "").unwrap();

    let out = run_audit(dir.path(), &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Tooling error (exit 2) means we couldn't resolve the graph (e.g. no
    // network in a sandbox); don't turn that into a false failure.
    if out.status.code() == Some(2) {
        eprintln!("skipping: could not resolve consumer graph\nstderr: {stderr}");
        return;
    }
    assert_eq!(
        out.status.code(),
        Some(1),
        "audit gate should reject an unpatched consumer (issue #149).\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("does NOT resolve from the libviprs fork"),
        "audit gate rejected for the wrong reason.\nstderr: {stderr}"
    );
}

#[test]
fn patchless_downstream_consumer_resolves_the_fork() {
    // Issue #149 part 2. A downstream crate that depends on libviprs the way
    // an external git/path consumer does, with no [patch.crates-io] table of
    // its own, must still resolve the per-call-locking pdfium-render fork.
    // With the fork applied only via [patch] this fails: cargo ignores a
    // dependency's [patch] table, so the consumer links the unpatched
    // registry wrapper. Declaring the fork as a direct git dependency of
    // libviprs makes it an ordinary edge in the graph, which every git/path
    // consumer inherits.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    let libviprs_path = repo_root();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        format!(
            "[package]\n\
             name = \"vip149-patchless-consumer\"\n\
             version = \"0.0.0\"\n\
             edition = \"2021\"\n\
             \n\
             [dependencies]\n\
             libviprs = {{ path = {:?}, features = [\"pdfium\"] }}\n",
            libviprs_path.display().to_string(),
        ),
    )
    .unwrap();
    std::fs::write(dir.path().join("src").join("lib.rs"), "").unwrap();

    let out = run_audit(dir.path(), &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Tooling error (exit 2) means we couldn't resolve the graph (e.g. no
    // network in a sandbox); don't turn that into a false failure.
    if out.status.code() == Some(2) {
        eprintln!("skipping: could not resolve consumer graph\nstderr: {stderr}");
        return;
    }
    assert!(
        out.status.success(),
        "a patch-less downstream consumer of libviprs must resolve the \
         pdfium-render fork (issue #149 part 2).\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("github.com/libviprs/pdfium-render"),
        "audit did not confirm the libviprs fork source for the downstream \
         consumer.\nstdout: {stdout}"
    );
}
