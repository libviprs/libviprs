//! Every requirement in the manifest admits a live release, so a fresh
//! resolve of this tree succeeds (issue #1213).
//!
//! # The hole this closes
//!
//! `Cargo.lock` is not committed here, on purpose (see the `jxl-oxide` comment
//! in `Cargo.toml`): CI, libviprs-tests' gate image and libviprs-bench's Docker
//! builder all resolve the graph fresh, so a dependency family raising its MSRV
//! shows up within a day instead of in a consumer's lockfile. The flip side is
//! that the tree is only as buildable as the registry says it is, and nothing
//! here asked the registry. On 2026-10-13 the `zip` crate published 9.0.3 and
//! yanked every release before it, and `zip = "7.2.0"` (`^7.2.0`) was left
//! matching nothing cargo may select:
//!
//! ```text
//! error: failed to select a version for the requirement `zip = "^7.2.0"`
//!   version 7.2.0 is yanked
//!   version 7.3.0 is yanked
//!   version 7.4.0 is yanked
//! ```
//!
//! That error is not confined to `--features packfile`. A lockfile covers every
//! feature of a workspace member, so a fresh `cargo test` with no features at
//! all fails before compiling anything, and so does the gate image's
//! `cargo fetch` and the bench builder's `cargo fetch --locked`. A checkout with
//! a pre-yank lock keeps building, because a locked version stays allowed after
//! it is yanked, which is how the breakage went unnoticed: every tree that had
//! built before kept building, and only fresh ones died.
//!
//! # What this file checks
//!
//! Three things, each a real `cargo` resolution against the live index rather
//! than a parse of the manifest, because "does this requirement admit a live
//! release" is a question only the registry can answer:
//!
//! * [`a_fresh_resolve_of_this_workspace_succeeds_1213`] builds the same
//!   skeleton libviprs-tests' Dockerfile builds (every `Cargo.toml`, an empty
//!   `src/lib.rs` per package, an empty file per explicit `path =` target) in a
//!   temporary directory and runs `cargo generate-lockfile` there. That is the
//!   resolve CI does, with none of the real tree's lockfile to lean on.
//! * [`every_feature_s_dependency_set_resolves_on_its_own_1213`] does the same
//!   for each feature's `dep:` set alone, copying the entries out of
//!   `[workspace.dependencies]` into a one-package manifest, so a failure names
//!   the feature and the requirement rather than the whole graph.
//! * [`packfile_round_trips_against_a_live_zip_release_1213`] (under
//!   `--features packfile`) writes a DeepZoom pyramid through `PackfileSink` as
//!   a zip, reopens it with the `zip` crate this build linked, and then asks
//!   the registry whether that exact version is still a live release. A yanked
//!   one is the lockfile keeping a dead version alive, and that is this issue
//!   frozen rather than fixed.
//!
//! The two controls at the bottom pin that the probe can tell a yanked
//! requirement from a live one, so a pass above means what it says.
//!
//! # Network, and cost
//!
//! Every test here needs the crates.io index: a fresh resolve under the sparse
//! protocol revalidates the index entries it touches, so even a warm `CARGO_HOME`
//! gives a fresh answer. No crate is downloaded, only index entries. Measured on
//! MARS (native x86_64, warm index): the whole-workspace skeleton resolve in
//! about a second, each feature set well under that. `cargo test --offline`
//! fails here with cargo's own error, which is the right answer: the question
//! cannot be asked offline.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// A fresh `cargo generate-lockfile` in `dir`, against the live index. `Ok`
/// with the lockfile text, `Err` with cargo's stderr.
fn fresh_resolve(dir: &Path) -> Result<String, String> {
    let out = Command::new(cargo())
        .current_dir(dir)
        .args(["generate-lockfile"])
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn cargo generate-lockfile: {e}"));
    if out.status.success() {
        Ok(std::fs::read_to_string(dir.join("Cargo.lock")).expect("cargo wrote no Cargo.lock"))
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// The explicit `path = "..."` targets of a parsed manifest (`[[bin]]`,
/// `[[test]]`, `[[bench]]`, `[[example]]`, `[lib]`), relative to it.
fn explicit_target_paths(manifest: &toml::Value) -> Vec<String> {
    let mut out = Vec::new();
    for key in ["bin", "test", "bench", "example"] {
        if let Some(arr) = manifest.get(key).and_then(toml::Value::as_array) {
            out.extend(
                arr.iter()
                    .filter_map(|t| t.get("path").and_then(toml::Value::as_str))
                    .map(str::to_owned),
            );
        }
    }
    if let Some(p) = manifest
        .get("lib")
        .and_then(|l| l.get("path"))
        .and_then(toml::Value::as_str)
    {
        out.push(p.to_owned());
    }
    out
}

/// Lay the dependency skeleton of this workspace down under `dir`: what
/// libviprs-tests' Dockerfile copies before its `cargo fetch`, and no more.
fn workspace_skeleton(dir: &Path) {
    let root: toml::Value = read("Cargo.toml").parse().expect("Cargo.toml parses");
    let mut manifests = vec![PathBuf::from("Cargo.toml")];
    if let Some(members) = root
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
    {
        for m in members.iter().filter_map(toml::Value::as_str) {
            assert!(
                !m.contains('*'),
                "workspace member `{m}` is a glob; this skeleton copies members by name"
            );
            manifests.push(Path::new(m).join("Cargo.toml"));
        }
    }
    for rel in manifests {
        let text = read(rel.to_str().unwrap());
        let parsed: toml::Value = text
            .parse()
            .unwrap_or_else(|e| panic!("{} parses: {e}", rel.display()));
        let pkg_dir = dir.join(rel.parent().unwrap());
        std::fs::create_dir_all(pkg_dir.join("src")).unwrap();
        std::fs::write(dir.join(&rel), &text).unwrap();
        if parsed.get("package").is_some() {
            std::fs::write(pkg_dir.join("src/lib.rs"), "").unwrap();
        }
        for target in explicit_target_paths(&parsed) {
            let p = pkg_dir.join(target);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "").unwrap();
        }
    }
}

/// A one-package manifest with exactly `deps` as its dependencies, in `dir`.
fn probe_crate(dir: &Path, name: &str, deps: &BTreeMap<String, toml::Value>) {
    let mut manifest = toml::map::Map::new();
    let mut package = toml::map::Map::new();
    package.insert("name".into(), toml::Value::String(name.into()));
    package.insert("version".into(), toml::Value::String("0.0.0".into()));
    package.insert("edition".into(), toml::Value::String("2024".into()));
    manifest.insert("package".into(), toml::Value::Table(package));
    let mut table = toml::map::Map::new();
    for (k, v) in deps {
        table.insert(k.clone(), v.clone());
    }
    manifest.insert("dependencies".into(), toml::Value::Table(table));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/lib.rs"), "").unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        toml::to_string(&toml::Value::Table(manifest)).expect("manifest serialises"),
    )
    .unwrap();
}

/// One requirement on its own: `name = <req>`, resolved fresh.
fn probe_requirement(name: &str, req: &str) -> Result<String, String> {
    let dir = tempfile::tempdir().unwrap();
    let mut deps = BTreeMap::new();
    deps.insert(name.to_owned(), toml::Value::String(req.to_owned()));
    probe_crate(dir.path(), &format!("probe-{name}"), &deps);
    fresh_resolve(dir.path())
}

/// cargo's explanation, trimmed to the lines that say what failed.
fn first_error(stderr: &str) -> String {
    stderr
        .lines()
        .skip_while(|l| !l.starts_with("error"))
        .take(8)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[cfg_attr(miri, ignore)] // spawns a process and writes a skeleton, neither of which Miri allows (#714)
fn a_fresh_resolve_of_this_workspace_succeeds_1213() {
    let dir = tempfile::tempdir().unwrap();
    workspace_skeleton(dir.path());
    if let Err(stderr) = fresh_resolve(dir.path()) {
        panic!(
            "a fresh resolve of this workspace fails, so every clean clone (CI, the \
             libviprs-tests gate image, the libviprs-bench builder) fails with it (#1213):\n{}",
            first_error(&stderr)
        );
    }
}

/// `(feature, {dependency name: its [workspace.dependencies] entry})` for every
/// feature that turns a `dep:` on.
fn feature_dependency_sets() -> BTreeMap<String, BTreeMap<String, toml::Value>> {
    let root: toml::Value = read("Cargo.toml").parse().expect("Cargo.toml parses");
    let workspace_deps = root
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .and_then(toml::Value::as_table)
        .expect("[workspace.dependencies]");
    let features = root
        .get("features")
        .and_then(toml::Value::as_table)
        .expect("[features]");
    let mut out = BTreeMap::new();
    for (feature, enables) in features {
        let mut set = BTreeMap::new();
        for e in enables.as_array().into_iter().flatten() {
            let Some(dep) = e.as_str().and_then(|s| s.strip_prefix("dep:")) else {
                continue;
            };
            let entry = workspace_deps.get(dep).unwrap_or_else(|| {
                panic!("feature `{feature}` enables `dep:{dep}`, which [workspace.dependencies] does not declare")
            });
            let mut entry = entry.clone();
            if let Some(t) = entry.as_table_mut() {
                // Not meaningful in a probe crate that is not this workspace.
                t.remove("optional");
                t.remove("workspace");
            }
            set.insert(dep.to_owned(), entry);
        }
        if !set.is_empty() {
            out.insert(feature.clone(), set);
        }
    }
    assert!(
        out.contains_key("packfile"),
        "expected `packfile` among the features that enable a `dep:`, got {:?}",
        out.keys().collect::<Vec<_>>()
    );
    out
}

#[test]
#[cfg_attr(miri, ignore)] // spawns a process and writes probe crates, neither of which Miri allows (#714)
fn every_feature_s_dependency_set_resolves_on_its_own_1213() {
    let mut failures = Vec::new();
    for (feature, deps) in feature_dependency_sets() {
        let dir = tempfile::tempdir().unwrap();
        probe_crate(dir.path(), &format!("probe-{feature}"), &deps);
        if let Err(stderr) = fresh_resolve(dir.path()) {
            failures.push(format!(
                "`{feature}` ({}):\n{}",
                deps.keys().cloned().collect::<Vec<_>>().join(", "),
                first_error(&stderr)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "these features' dependency sets do not resolve against the live index (#1213):\n\n{}",
        failures.join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Under `packfile`: the code that uses the crate, against a live release of it
// ---------------------------------------------------------------------------

/// The versions of `name` this build links, out of `cargo tree` over the real
/// tree with the features that pull it in. One entry in a sane graph; the test
/// reports every one it finds.
#[cfg(feature = "packfile")]
fn linked_versions(name: &str, features: &str) -> Vec<String> {
    let out = Command::new(cargo())
        .current_dir(repo_root())
        .args([
            "tree",
            "-p",
            "libviprs",
            "-e",
            "normal",
            "--features",
            features,
            "-i",
            name,
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .output()
        .expect("failed to spawn cargo tree");
    assert!(
        out.status.success(),
        "cargo tree -i {name} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let prefix = format!("{name} v");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix(&prefix))
        .map(|v| v.split_whitespace().next().unwrap_or(v).to_owned())
        .collect()
}

#[cfg(feature = "packfile")]
#[test]
#[cfg_attr(miri, ignore)] // writes an archive and spawns cargo, neither of which Miri allows (#714)
fn packfile_round_trips_against_a_live_zip_release_1213() {
    use libviprs::pixel::PixelFormat;
    use libviprs::planner::{Layout, PyramidPlanner, TileCoord};
    use libviprs::raster::Raster;
    use libviprs::sink::{PackfileFormat, PackfileSink, Tile, TileFormat, TileSink};

    // The round trip: a small DeepZoom pyramid through the zip sink, reopened
    // with the zip crate this build linked.
    let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pyramid.zip");
    let sink = PackfileSink::builder(&out)
        .plan(plan.clone())
        .format(PackfileFormat::Zip)
        .tile_format(TileFormat::Png)
        .build()
        .unwrap();
    let mut written = 0usize;
    for level in &plan.levels {
        for y in 0..level.rows {
            for x in 0..level.cols {
                sink.write_tile(&Tile {
                    coord: TileCoord::new(level.level, x, y),
                    raster: Raster::zeroed(32, 32, PixelFormat::Rgb8).unwrap(),
                    blank: false,
                })
                .unwrap();
                written += 1;
            }
        }
    }
    sink.finish().unwrap();
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
    assert!(
        archive.len() > written,
        "the archive holds {} entries for {written} tiles, so the manifest or the .dzi is missing",
        archive.len()
    );
    archive
        .by_name("manifest.json")
        .expect("the archive carries manifest.json at its root");

    // And the release it was written and read with is one the registry still
    // offers. A yanked one only linked because a lockfile kept it alive.
    let versions = linked_versions("zip", "packfile");
    assert!(
        !versions.is_empty(),
        "cargo tree found no `zip` under --features packfile"
    );
    let mut yanked = Vec::new();
    for v in &versions {
        match probe_requirement("zip", &format!("={v}")) {
            Ok(_) => {}
            Err(stderr) if stderr.contains("is yanked") => yanked.push(v.clone()),
            Err(stderr) => panic!("probing zip ={v} failed for another reason:\n{stderr}"),
        }
    }
    assert!(
        yanked.is_empty(),
        "this build linked zip {} and the registry has yanked {}; the round trip above ran \
         against a release nobody can resolve fresh, and that is #1213 kept alive by a \
         lockfile rather than fixed",
        versions.join(", "),
        yanked.join(", ")
    );
}

// ---------------------------------------------------------------------------
// Controls: the probe can tell yanked from live
// ---------------------------------------------------------------------------

/// `zip = "=7.2.0"` is the release the manifest's requirement starts at, and
/// the registry has yanked it (2026-10-13). The probe has to say so, in cargo's
/// words, or the tests above could pass for want of looking. If this ever goes
/// red because upstream un-yanked 7.x, the manifest's requirement has become
/// satisfiable again and the file above it needs a new control.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process and writes a probe crate, neither of which Miri allows (#714)
fn control_a_requirement_with_only_yanked_releases_is_refused_1213() {
    match probe_requirement("zip", "=7.2.0") {
        Err(stderr) => assert!(
            stderr.contains("version 7.2.0 is yanked"),
            "expected cargo to name the yanked release, got:\n{stderr}"
        ),
        Ok(_) => panic!(
            "zip =7.2.0 resolved, so either 7.2.0 is live again or the probe is not asking the registry"
        ),
    }
}

/// And a requirement with a live release resolves, so a refusal above is about
/// the requirement and not about the probe or the network.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process and writes a probe crate, neither of which Miri allows (#714)
fn control_a_requirement_with_a_live_release_resolves_1213() {
    if let Err(stderr) = probe_requirement("serde", "1") {
        panic!("serde = \"1\" did not resolve, so the probe cannot reach the registry:\n{stderr}");
    }
}
