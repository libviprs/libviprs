//! Issue #1486 (part of #1488): acadrust comes from our fork, `libviprs/acadrust`, as a git
//! dependency pinned by rev, and never from a registry. Rom's rule.
//!
//! The scan reads the manifests that can name acadrust (the root, `fuzz/` and the two spike
//! harnesses) through `include_str!`, so it touches no file at run time and needs no entry in
//! the Miri inventory. `lock_complaints` is the same rule for a `Cargo.lock`; this repo commits
//! none, so for now it is only held by the planted case below, and reading a real lockfile waits
//! on the Miri inventory being free of other open PRs (libviprs#1486). A `path` dependency is fine (a spike pointing at a checkout of the fork); a version-only
//! dependency, a `registry` key, or a git URL that is not the fork is not.
//! `the_scan_catches_a_registry_acadrust` runs the same checks over planted files.

const FORK: &str = "https://github.com/libviprs/acadrust";

fn dependency_tables<'a>(v: &'a toml::Value, out: &mut Vec<&'a toml::Table>) {
    let Some(t) = v.as_table() else { return };
    for (k, v) in t {
        if k.ends_with("dependencies") {
            if let Some(d) = v.as_table() {
                out.push(d);
            }
        } else {
            dependency_tables(v, out);
        }
    }
}

fn manifest_complaints(text: &str) -> Vec<String> {
    let doc: toml::Value = text.parse().expect("a Cargo.toml parses");
    let mut tables = Vec::new();
    dependency_tables(&doc, &mut tables);
    let mut out = Vec::new();
    for deps in tables {
        for (name, spec) in deps {
            let renamed = spec.get("package").and_then(toml::Value::as_str);
            if name != "acadrust" && renamed != Some("acadrust") {
                continue;
            }
            let git = spec.get("git").and_then(toml::Value::as_str);
            match (git, spec.get("path")) {
                (_, Some(_)) => {}
                (Some(url), None) => {
                    if url.trim_end_matches(".git") != FORK {
                        out.push(format!("acadrust comes from {url}, not {FORK}"));
                    }
                    let rev = spec.get("rev").and_then(toml::Value::as_str).unwrap_or("");
                    if rev.len() != 40 || !rev.bytes().all(|b| b.is_ascii_hexdigit()) {
                        out.push("acadrust is not pinned by a full 40 digit rev".into());
                    }
                }
                (None, None) => {
                    out.push("acadrust has no git source, so it is a registry dependency".into())
                }
            }
            if spec.get("registry").is_some() {
                out.push("acadrust names a registry".into());
            }
        }
    }
    out
}

fn lock_complaints(text: &str) -> Vec<String> {
    let doc: toml::Value = text.parse().expect("a Cargo.lock parses");
    let mut out = Vec::new();
    for pkg in doc
        .get("package")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        if pkg.get("name").and_then(toml::Value::as_str) != Some("acadrust") {
            continue;
        }
        match pkg.get("source").and_then(toml::Value::as_str) {
            Some(s) if s.starts_with(&format!("git+{FORK}")) => {}
            other => out.push(format!("a lockfile resolves acadrust from {other:?}")),
        }
    }
    out
}

#[test]
fn acadrust_is_only_ever_the_forks_git_revision() {
    let manifests = [
        ("Cargo.toml", include_str!("../Cargo.toml")),
        ("fuzz/Cargo.toml", include_str!("../fuzz/Cargo.toml")),
        (
            "docs/spikes/view-context/acadrust-probe/Cargo.toml",
            include_str!("../docs/spikes/view-context/acadrust-probe/Cargo.toml"),
        ),
        (
            "docs/spikes/pdf-normalization/pair/Cargo.toml",
            include_str!("../docs/spikes/pdf-normalization/pair/Cargo.toml"),
        ),
    ];
    for (name, text) in manifests {
        let complaints = manifest_complaints(text);
        assert!(complaints.is_empty(), "{name}: {complaints:?}");
    }
    assert!(
        include_str!("../Cargo.toml")
            .contains("acadrust = { git = \"https://github.com/libviprs/acadrust\""),
        "the root manifest must name the fork"
    );
}

#[test]
fn the_scan_catches_a_registry_acadrust() {
    assert!(!manifest_complaints("[dependencies]\nacadrust = \"0.6\"\n").is_empty());
    assert!(
        !manifest_complaints(
            "[workspace.dependencies]\nacadrust = { version = \"0.6\", optional = true }\n"
        )
        .is_empty()
    );
    assert!(!manifest_complaints("[dependencies]\nacadrust = { git = \"https://github.com/hakanaktt/acadrust\", rev = \"72af48970e306ef1d50dd751b227df2b4fc3fb99\" }\n").is_empty());
    assert!(
        !manifest_complaints(
            "[target.'cfg(unix)'.dependencies]\nx = { package = \"acadrust\", version = \"1\" }\n"
        )
        .is_empty()
    );
    assert!(!manifest_complaints("[dependencies]\nacadrust = { git = \"https://github.com/libviprs/acadrust\", branch = \"main\" }\n").is_empty());
    assert!(manifest_complaints("[dependencies]\nacadrust = { git = \"https://github.com/libviprs/acadrust\", rev = \"72af48970e306ef1d50dd751b227df2b4fc3fb99\", version = \"0.5.5\" }\n").is_empty());
    assert!(
        manifest_complaints("[dependencies]\nacadrust = { path = \"../acadrust\" }\n").is_empty()
    );
    let registry = "[[package]]\nname = \"acadrust\"\nversion = \"0.6.8\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n";
    assert!(!lock_complaints(registry).is_empty());
    let fork = "[[package]]\nname = \"acadrust\"\nversion = \"0.5.5\"\nsource = \"git+https://github.com/libviprs/acadrust?rev=72af48970e306ef1d50dd751b227df2b4fc3fb99#72af48970e306ef1d50dd751b227df2b4fc3fb99\"\n";
    assert!(lock_complaints(fork).is_empty());
}
