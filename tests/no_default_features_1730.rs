//! The library has to build with no features at all (issue #1730).
//!
//! `pvec::emf` was compiled in every build but used `pvec_numeric`, which only the
//! `pvec-spline` and `pvec-normalized` features bring in, so a dependent that sets
//! `default-features = false` (libviprs-cli does) could not build the core. This runs
//! `cargo check --lib --no-default-features` in a target directory of its own, so it
//! does not wait on the lock the test run holds.

use std::process::Command;

#[test]
fn the_library_checks_with_no_default_features() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let out = Command::new(env!("CARGO"))
        .args(["check", "--lib", "--no-default-features", "--manifest-path"])
        .arg(format!("{manifest}/Cargo.toml"))
        .env(
            "CARGO_TARGET_DIR",
            format!("{manifest}/target/no-default-features-1730"),
        )
        .output()
        .expect("cargo must run");
    assert!(
        out.status.success(),
        "cargo check --lib --no-default-features failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
