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
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn audit_gate_accepts_this_crate() {
    let out = run_audit(&repo_root(), &["--features", "pdfium"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "audit gate rejected this crate.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("crates.io"),
        "audit gate did not confirm a registry source.\nstdout: {stdout}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn audit_gate_rejects_a_consumer_still_pinning_the_fork() {
    // The mirror image of what this test used to assert. A consumer that still
    // pins the git fork is now the failure case, because the fork is retired
    // (#981) and a git source cannot survive publish, so pinning it puts that
    // consumer on different code from everyone else.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\n\
         name = \"vip981-fork-pinning-consumer\"\n\
         version = \"0.0.0\"\n\
         edition = \"2021\"\n\
         \n\
         [dependencies]\n\
         pdfium-render = { version = \"0.9\", git = \"https://github.com/libviprs/pdfium-render.git\", rev = \"3b03093295b85486c2e42f514cef9647eb629c63\" }\n",
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
        "audit gate should reject a consumer that still pins the fork \
         (issue #981).\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("resolves from the libviprs fork")
            || stderr.contains("resolves from neither"),
        "audit gate rejected for the wrong reason.\nstderr: {stderr}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn patchless_downstream_consumer_resolves_the_same_crate_we_do() {
    // Issue #149 part 2, inverted by #981. The question has not changed: does a
    // downstream crate that depends on libviprs with no `[patch.crates-io]` of
    // its own end up on the same `pdfium-render` libviprs itself builds? The
    // answer used to be "only if libviprs declares the fork as a direct git
    // edge", because cargo ignores a dependency's `[patch]` table. That made
    // git and path consumers match libviprs and left crates.io consumers on
    // something else, which is the split #981 closes.
    //
    // Now the answer is simpler: everybody resolves the registry crate,
    // including a published consumer, because an ordinary version requirement
    // is the one kind of edge that survives publish.
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
        "a patch-less downstream consumer of libviprs must resolve the same \
         pdfium-render libviprs builds against (issues #149, #981).\nstdout: \
         {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("crates.io"),
        "audit did not confirm a registry source for the downstream consumer, \
         so the consumer and libviprs are on different code again.\nstdout: \
         {stdout}"
    );
}
