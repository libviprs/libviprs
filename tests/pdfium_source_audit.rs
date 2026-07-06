//! Regression coverage for issue #149.
//!
//! The `pdfium-render` per-call thread-safety fork is applied through this
//! crate's `[patch.crates-io]` table. A `[patch]` only takes effect from the
//! build root, so any consumer that omits it silently links the unpatched
//! `pdfium-render 0.8.x` wrapper (documented to segfault under concurrent
//! access in `src/streaming.rs`). These tests exercise the release gate that
//! detects that condition, `scripts/audit-pdfium-source.sh`:
//!
//!   1. `audit_gate_accepts_patched_core` — this crate carries the fork, so
//!      the gate must pass. Guards against the patch being dropped/renamed.
//!   2. `audit_gate_rejects_unpatched_consumer` — a synthesized sibling that
//!      depends on `pdfium-render` without the patch resolves the crates.io
//!      registry crate; the gate must reject it (this is the #149 defect).

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
