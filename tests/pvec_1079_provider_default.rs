//! Issue #1395 (slice of #1079): acadrust is the default DWG provider, and any acadsharp-rs path is
//! opt-in and documented as such. Rom's decision D2.
//!
//! Static checks over `Cargo.toml`, `src/pvec` and `docs/pvec-providers.md`, so they run in every
//! cell with no feature and no native library. `the_checks_fail_on_a_mutated_manifest` runs the same
//! checks over a manifest that defaults to acadsharp-rs and requires them to complain.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST: &str = include_str!("../Cargo.toml");
const PROVIDER_DOC: &str = include_str!("../docs/pvec-providers.md");

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The features reachable from `default`, and the `dep:` entries they switch on.
fn default_closure(manifest: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let doc: toml::Table = manifest.parse().expect("Cargo.toml parses");
    let features = doc["features"].as_table().expect("[features]");
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut deps: BTreeSet<String> = BTreeSet::new();
    let mut todo = vec!["default".to_string()];
    while let Some(f) = todo.pop() {
        if !seen.insert(f.clone()) {
            continue;
        }
        for item in features
            .get(&f)
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            let item = item.as_str().unwrap_or_default();
            if let Some(d) = item.strip_prefix("dep:") {
                deps.insert(d.to_string());
            } else if features.contains_key(item) {
                todo.push(item.to_string());
            } else {
                deps.insert(item.split('/').next().unwrap_or(item).to_string());
            }
        }
    }
    (seen, deps)
}

/// Every complaint about `manifest`, empty when it keeps acadrust the default and acadsharp-rs opt-in.
fn manifest_complaints(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (features, deps) = default_closure(manifest);
    for provider in ["acadsharp-rs", "acadrust"] {
        if deps.contains(provider) {
            out.push(format!("the default feature set switches on {provider}"));
        }
    }
    if features.contains("dwg-acadsharp") || features.contains("dwg-acadrust") {
        out.push("the default feature set contains a DWG provider feature".into());
    }
    let doc: toml::Table = manifest.parse().unwrap();
    let dep = &doc["dependencies"]["acadsharp-rs"];
    if dep.get("optional").and_then(toml::Value::as_bool) != Some(true) {
        out.push("acadsharp-rs is not an optional dependency".into());
    }
    let enabling: Vec<&String> = doc["features"]
        .as_table()
        .unwrap()
        .iter()
        .filter(|(_, v)| {
            v.as_array().is_some_and(|a| {
                a.iter().any(|i| {
                    i.as_str()
                        .is_some_and(|s| s == "dep:acadsharp-rs" || s.starts_with("acadsharp-rs"))
                })
            })
        })
        .map(|(k, _)| k)
        .collect();
    if enabling != ["dwg-acadsharp"] {
        out.push(format!(
            "only dwg-acadsharp may enable acadsharp-rs, found {enabling:?}"
        ));
    }
    out
}

#[test]
fn cargo_defaults_leave_both_providers_off_and_acadsharp_opt_in() {
    assert_eq!(manifest_complaints(MANIFEST), Vec::<String>::new());
}

#[test]
fn the_checks_fail_on_a_mutated_manifest() {
    let defaulted = MANIFEST.replacen("default = []", "default = [\"dwg-acadsharp\"]", 1);
    assert_ne!(defaulted, MANIFEST, "the mutation must change the manifest");
    assert!(!manifest_complaints(&defaulted).is_empty());
    let leaked = MANIFEST.replacen(
        "pvec-spline = [\"dep:pvec-numeric\", \"dep:pvec-geometry\"]",
        "pvec-spline = [\"dep:pvec-numeric\", \"dep:pvec-geometry\", \"dep:acadsharp-rs\"]",
        1,
    );
    assert_ne!(leaked, MANIFEST);
    assert!(!manifest_complaints(&leaked).is_empty());
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn no_code_in_src_pvec_names_acadsharp() {
    let mut files = Vec::new();
    rust_files(&root().join("src/pvec"), &mut files);
    assert!(!files.is_empty());
    for f in files {
        for (i, line) in fs::read_to_string(&f).unwrap().lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            assert!(
                !code.to_ascii_lowercase().contains("acadsharp"),
                "{}:{} names acadsharp outside a comment: {line}",
                f.display(),
                i + 1
            );
        }
    }
}

#[test]
fn the_first_provider_is_acadrust() {
    let src = include_str!("../src/pvec/dwg_acadrust.rs");
    assert!(src.contains("const NAME: &str = \"acadrust\";"));
}

#[test]
fn the_provider_page_says_acadrust_is_the_default_and_acadsharp_is_opt_in() {
    let doc = PROVIDER_DOC
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for needle in [
        "acadrust** is the default provider",
        "acadsharp-rs** is opt-in",
        "`dwg-acadsharp`",
        "off by default",
        "feature set is empty",
        "libredwg",
    ] {
        assert!(
            doc.contains(needle),
            "docs/pvec-providers.md lost: {needle}"
        );
    }
}

// ---- no LibreDWG in the build (issue #1405, Rom's rule) ---------------------------------------
//
// libviprs has no LibreDWG dependency of any kind. Anything LibreDWG produced is committed as a
// static file with provenance. SCOPE, explicitly: the places a build, a CI row or a script could
// build or run it, which are the root `Cargo.toml`, every `build.rs`, everything under `tools/`
// and `scripts/` and `.github/`, the `Makefile`, `run-tests.sh` and any `Dockerfile*`. Out of
// scope on purpose, because they only name or store it: `docs/`, `src/` comments, `tests/` (the
// label strings of `pvec_dual_oracle.rs`, this file) and the static pairs under
// `tests/fixtures/pvec/pdf_loss/` with their `exporter.txt` provenance.

const FORBIDDEN: [&str; 4] = ["libredwg", "dwg2svg", "dwg2pdf", "dwgread"];

fn scoped_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                if !["target", ".git", "docs", "node_modules"].contains(&name.as_str()) {
                    walk(&p, out);
                }
            } else {
                out.push(p);
            }
        }
    }
    let r = root();
    let mut all = Vec::new();
    for d in [
        "tools", "scripts", ".github", "src", "fuzz", "examples", "web",
    ] {
        walk(&r.join(d), &mut all);
    }
    let mut out: Vec<PathBuf> = all
        .into_iter()
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            let top = p.strip_prefix(&r).unwrap().components().next().unwrap();
            let top = top.as_os_str().to_string_lossy().to_string();
            ["tools", "scripts", ".github"].contains(&top.as_str()) || n == "build.rs"
        })
        .collect();
    for f in ["Cargo.toml", "Makefile", "run-tests.sh"] {
        out.push(r.join(f));
    }
    if let Ok(rd) = fs::read_dir(&r) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("Dockerfile") {
                out.push(e.path());
            }
        }
    }
    out.retain(|p| p.is_file());
    out
}

fn forbidden_hits(files: &[(PathBuf, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (p, text) in files {
        let lower = text.to_ascii_lowercase();
        for w in FORBIDDEN {
            if lower.contains(w) {
                out.push(format!("{} names {w}", p.display()));
            }
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn nothing_that_builds_or_runs_libredwg_lives_in_libviprs() {
    let files: Vec<(PathBuf, String)> = scoped_files()
        .into_iter()
        .filter_map(|p| fs::read_to_string(&p).ok().map(|t| (p, t)))
        .collect();
    assert!(
        files.len() > 5,
        "the scope walk found {} files, so it would pass on nothing",
        files.len()
    );
    assert!(
        files.iter().any(|(p, _)| p.ends_with("Cargo.toml")),
        "Cargo.toml is in scope"
    );
    assert_eq!(forbidden_hits(&files), Vec::<String>::new());
}

#[test]
fn the_libredwg_guard_fails_on_a_mention_in_each_kind_of_place() {
    for place in [
        "Cargo.toml",
        "crate/build.rs",
        "tools/x.sh",
        ".github/workflows/ci.yml",
        "Makefile",
    ] {
        for word in FORBIDDEN {
            let hit =
                forbidden_hits(&[(PathBuf::from(place), format!("run {}", word.to_uppercase()))]);
            assert_eq!(hit.len(), 1, "{place} / {word}");
        }
    }
    assert!(forbidden_hits(&[(PathBuf::from("Cargo.toml"), "acadrust".into())]).is_empty());
}

#[test]
fn the_provider_page_says_libviprs_has_no_libredwg_dependency() {
    let doc = PROVIDER_DOC
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for needle in [
        "no libredwg dependency of any kind",
        "static file",
        "out-of-tree crate",
    ] {
        assert!(
            doc.contains(needle),
            "docs/pvec-providers.md lost: {needle}"
        );
    }
    assert!(
        !doc.contains("planned"),
        "the page must not call LibreDWG planned work"
    );
}
