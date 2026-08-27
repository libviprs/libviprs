//! The dependency rule from `CONTRIBUTING.md`, checked against the graph cargo
//! actually resolves (issue #606).
//!
//! The rule this project used to enforce was "zero `-sys` crates and zero C
//! compiles anywhere in the tree", and it was never true: `blake3` is a direct
//! dependency and its `build.rs` compiles `c/blake3_neon.c` on every aarch64
//! build. Every obvious tightening of it is false too, which is why this file
//! keys on none of them:
//!
//! * `build.rs` is not the discriminator. 21 crates in the default host graph
//!   have one, `serde` and `libc` among them.
//! * `links =` is not the discriminator. `rayon-core` declares
//!   `links = "rayon-core"` in every default build on every target and is not
//!   a C library at all; `links` is also how cargo enforces a single version
//!   of a crate across the graph.
//! * A `-sys` suffix is not the discriminator. `libbz2-rs-sys` is pure Rust,
//!   with no `build.rs` and no `.c` file anywhere in it.
//!
//! What this file pins instead is the real property: nothing libviprs builds
//! needs a library that is not already in the dependency tree. Concretely, per
//! target and per feature set, it asserts that the set of `links` keys equals
//! a named allowlist, that no crate whose job is to *find* an installed
//! library is in the graph outside the one feature that carries one, that the
//! crates shipping compilable C or assembly are exactly the two documented
//! ones, and that the features which do reach an external library stay off by
//! default.
//!
//! # Why `cargo tree` and not `cargo metadata`
//!
//! `cargo metadata`'s resolve graph lists optional dependencies that nothing
//! enabled, so reading it directly finds `defmt` (an unused optional of
//! `chrono`, `jiff` and `tinyvec`, and a `links = "defmt"` declarer) in the
//! default build, where `cargo tree` correctly prints nothing for it.
//! `--filter-platform` does not help: it filters by target cfg, not by
//! feature. So the package set comes from `cargo tree`, which is
//! feature-aware, and `cargo metadata` is used only as a lookup table for the
//! manifest facts (`links`, build script, source directory) of the packages
//! `cargo tree` named.
//!
//! `--target` matters just as much in the other direction: without it,
//! `cargo metadata` reports every target's dependencies at once and an audit
//! run on a mac will flag `wasm-bindgen-shared`.
//!
//! # Cost
//!
//! One `cargo metadata --all-features` plus one `cargo tree` per cell below,
//! around a fifth of a second each against a warm registry. The metadata call
//! runs first on purpose: it downloads every manifest in the graph, including
//! the windows- and wasm-only ones, so the per-target trees after it resolve
//! without further network access.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;

/// Crates whose whole job is to locate a library that is already installed on
/// the build machine.
///
/// `cc` is deliberately absent. It compiles source that came down with the
/// crate rather than discovering anything installed, and that is exactly the
/// distinction the rule turns on. It is in every graph on every target.
const DISCOVERY_CRATES: &[&str] = &["pkg-config", "system-deps", "vcpkg", "cmake", "bindgen"];

/// Extensions a `build.rs` can hand to a C toolchain.
const NATIVE_SOURCE_EXTENSIONS: &[&str] = &["c", "cc", "cpp", "cxx", "s", "asm"];

/// The host triple stands in for "the machine a contributor is on". The other
/// three cover the platform-conditional edges: `core-foundation-sys` on macOS,
/// `linux-raw-sys` on linux, the windows import crates, and `wasm-bindgen` on
/// wasm32, which is the one place a second `links` key shows up.
const MACOS: &str = "aarch64-apple-darwin";
const LINUX: &str = "x86_64-unknown-linux-gnu";
const WINDOWS: &str = "x86_64-pc-windows-msvc";
const WASM: &str = "wasm32-unknown-unknown";

/// One (target, feature set) pair, with everything the rule claims about it.
struct Cell {
    /// Human name, used in failure messages.
    label: &'static str,
    target: &'static str,
    /// Extra arguments spliced into the `cargo tree` invocation.
    features: &'static [&'static str],
    /// What `-e` gets. `normal,build` is the graph a consumer of the library
    /// links; the one `normal,build,dev` cell covers what a contributor's own
    /// `cargo test` additionally compiles.
    edges: &'static str,
    /// Every `links` key expected in this graph, as (crate, key).
    links: &'static [(&'static str, &'static str)],
    /// Every [`DISCOVERY_CRATES`] entry expected in this graph.
    discovery: &'static [&'static str],
    /// Every crate expected to ship compilable native source *and* a build
    /// script able to compile it. `None` skips the scan, which the non-host
    /// cells do: the check reads crate source directories off disk and there
    /// is no reason to trust that a windows-only crate has been unpacked here.
    vendored_native: Option<&'static [&'static str]>,
}

const RAYON: (&str, &str) = ("rayon-core", "rayon-core");
const ZSTD: (&str, &str) = ("zstd-sys", "zstd");
const WASM_BINDGEN: (&str, &str) = ("wasm-bindgen-shared", "wasm_bindgen");

const CELLS: &[Cell] = &[
    // The default build, which is what most consumers get.
    Cell {
        label: "host / default",
        target: MACOS,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    // The one carve-out that compiles vendored C beyond blake3, and the only
    // place a discovery crate is allowed to appear at all.
    Cell {
        label: "host / packfile",
        target: MACOS,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: Some(&["blake3", "zstd-sys"]),
    },
    // The two codec features that cost real crate count. Both add nothing
    // here, which is the claim their `Cargo.toml` comments make.
    Cell {
        label: "host / svg",
        target: MACOS,
        features: &["--features", "svg"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    Cell {
        label: "host / jxl",
        target: MACOS,
        features: &["--features", "jxl"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    // The external-library features. Neither shows up in `links` or in a C
    // compile, which is precisely why a check keyed on those would miss them;
    // `the_external_library_features_stay_opt_in` below is what covers them.
    Cell {
        label: "host / pdfium",
        target: MACOS,
        features: &["--features", "pdfium"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    Cell {
        label: "host / pdfium-static",
        target: MACOS,
        features: &["--features", "pdfium-static"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    // Every remaining feature at once. These are the ones whose Cargo.toml
    // comments say they add no dependencies, so the cell is the check on that.
    Cell {
        label: "host / object-store-sink + tracing + serde + test-util + s3",
        target: MACOS,
        features: &["--features", "object-store-sink,tracing,serde,test-util,s3"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3"]),
    },
    Cell {
        label: "host / all features",
        target: MACOS,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: Some(&["blake3", "zstd-sys"]),
    },
    // Dev-dependencies never reach a consumer, but they do have to build on a
    // contributor's machine. `generator` is loom's coroutine crate; it
    // assembles its own vendored stack-switching `.s` files, so it lands under
    // clause 3 the same way blake3 does.
    Cell {
        label: "host / default + dev-dependencies",
        target: MACOS,
        features: &[],
        edges: "normal,build,dev",
        links: &[RAYON],
        discovery: &[],
        vendored_native: Some(&["blake3", "generator"]),
    },
    Cell {
        label: "linux / default",
        target: LINUX,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: None,
    },
    Cell {
        label: "linux / packfile",
        target: LINUX,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
    Cell {
        label: "linux / all features",
        target: LINUX,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
    Cell {
        label: "windows / default",
        target: WINDOWS,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: None,
    },
    Cell {
        label: "windows / packfile",
        target: WINDOWS,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
    Cell {
        label: "windows / all features",
        target: WINDOWS,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
    // wasm32 is where the third `links` key lives. `packfile` is what puts it
    // there: `zip` turns on `getrandom`'s `wasm_js` backend and pulls `time`'s
    // wasm clock through `js-sys`, and both reach `wasm-bindgen`.
    Cell {
        label: "wasm32 / default",
        target: WASM,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        vendored_native: None,
    },
    Cell {
        label: "wasm32 / packfile",
        target: WASM,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, WASM_BINDGEN, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
    Cell {
        label: "wasm32 / all features",
        target: WASM,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, WASM_BINDGEN, ZSTD],
        discovery: &["pkg-config"],
        vendored_native: None,
    },
];

/// Repo root (the directory holding the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

/// The manifest facts about one package that the rule cares about.
struct Manifest {
    links: Option<String>,
    manifest_path: PathBuf,
    has_build_script: bool,
}

/// Every package cargo knows about, keyed by (name, version).
///
/// Read once, with `--all-features` and no `--filter-platform`, so the table
/// covers every package any cell below can name. This is a lookup table only:
/// which packages are actually *in* a given graph comes from `cargo tree`.
fn manifests() -> &'static BTreeMap<(String, String), Manifest> {
    static ONCE: OnceLock<BTreeMap<(String, String), Manifest>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let out = Command::new(cargo())
            .current_dir(repo_root())
            .args(["metadata", "--format-version", "1", "--all-features"])
            .output()
            .expect("failed to spawn cargo metadata");
        assert!(
            out.status.success(),
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let meta: Value = serde_json::from_slice(&out.stdout).expect("cargo metadata is not JSON");

        let mut table = BTreeMap::new();
        for pkg in meta["packages"].as_array().expect("packages array") {
            let name = pkg["name"].as_str().expect("package name").to_owned();
            let version = pkg["version"].as_str().expect("package version").to_owned();
            let has_build_script = pkg["targets"]
                .as_array()
                .expect("targets array")
                .iter()
                .any(|t| {
                    t["kind"]
                        .as_array()
                        .is_some_and(|k| k.iter().any(|k| k == "custom-build"))
                });
            table.insert(
                (name, version),
                Manifest {
                    links: pkg["links"].as_str().map(str::to_owned),
                    manifest_path: PathBuf::from(
                        pkg["manifest_path"].as_str().expect("manifest_path"),
                    ),
                    has_build_script,
                },
            );
        }
        table
    })
}

/// The packages cargo would build for one cell, as (name, version).
fn resolve(cell: &Cell) -> BTreeSet<(String, String)> {
    // Force the manifest table first: it is the call that downloads every
    // manifest, so the per-target trees after it need no network.
    let _ = manifests();

    let mut args = vec![
        "tree",
        "-p",
        "libviprs",
        "-e",
        cell.edges,
        "--target",
        cell.target,
        "--prefix",
        "none",
        "--format",
        "{p}",
    ];
    args.extend_from_slice(cell.features);

    let out = Command::new(cargo())
        .current_dir(repo_root())
        .args(&args)
        .output()
        .expect("failed to spawn cargo tree");
    assert!(
        out.status.success(),
        "`cargo {}` failed for {}:\n{}",
        args.join(" "),
        cell.label,
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).expect("cargo tree emitted non-UTF-8");
    let mut packages = BTreeSet::new();
    for line in stdout.lines() {
        // `{p}` is "<name> v<version>" followed by an optional source or
        // `(proc-macro)`, and a repeated subtree is marked with a trailing
        // `(*)`. Only the first two fields matter.
        let mut fields = line.split_whitespace();
        let (Some(name), Some(version)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Some(version) = version.strip_prefix('v') else {
            continue;
        };
        packages.insert((name.to_owned(), version.to_owned()));
    }
    assert!(
        packages.iter().any(|(name, _)| name == "libviprs"),
        "cargo tree for {} did not even contain libviprs, so the parse is wrong:\n{stdout}",
        cell.label
    );
    packages
}

/// Every cell's package set, resolved once for the whole test binary.
fn graphs() -> &'static Vec<BTreeSet<(String, String)>> {
    static ONCE: OnceLock<Vec<BTreeSet<(String, String)>>> = OnceLock::new();
    ONCE.get_or_init(|| CELLS.iter().map(resolve).collect())
}

fn cells() -> impl Iterator<Item = (&'static Cell, &'static BTreeSet<(String, String)>)> {
    CELLS.iter().zip(graphs().iter())
}

fn contains(graph: &BTreeSet<(String, String)>, crate_name: &str) -> bool {
    graph.iter().any(|(name, _)| name == crate_name)
}

/// Does this package ship a file a C toolchain could compile?
fn ships_native_source(manifest: &Manifest) -> bool {
    let root = manifest
        .manifest_path
        .parent()
        .expect("a manifest path always has a parent");
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let matched = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .is_some_and(|e| NATIVE_SOURCE_EXTENSIONS.contains(&e.as_str()));
            if matched {
                return true;
            }
        }
    }
    false
}

/// Clause 2, the mechanical half: the `links` keys in every resolved graph are
/// exactly the ones `CONTRIBUTING.md` names, and a new one cannot appear
/// unnoticed.
///
/// `links` is not by itself evidence of a C library, and this test is not
/// claiming it is. It is a cheap, exact tripwire: a crate that links something
/// almost always declares one, so pinning the set to an allowlist means any
/// new candidate has to be looked at by a human before this goes green again.
#[test]
fn every_links_key_is_on_the_allowlist() {
    for (cell, graph) in cells() {
        let found: BTreeSet<(&str, &str)> = graph
            .iter()
            .filter_map(|key| {
                let manifest = manifests().get(key)?;
                let links = manifest.links.as_deref()?;
                Some((key.0.as_str(), links))
            })
            .collect();
        let expected: BTreeSet<(&str, &str)> = cell.links.iter().copied().collect();
        assert_eq!(
            found, expected,
            "`links` keys changed for {}. Anything new here needs checking against the \
             dependency rule in CONTRIBUTING.md before the allowlist moves.",
            cell.label
        );
    }
}

/// Clause 1: nothing in the tree goes looking for a library on the build
/// machine, except under `packfile`, where `pkg-config` rides along as a
/// build-dependency of `zstd-sys`.
///
/// It is dormant there. `zstd-sys` resolves with only its `std` feature, so
/// its `build.rs` takes the vendored branch and compiles `zstd/lib`. The
/// discovery branch needs either the crate's own `pkg-config` feature or the
/// `ZSTD_SYS_USE_PKG_CONFIG` environment variable, and neither is set here.
/// Present but dormant is still worth pinning: if it ever spreads to the
/// default graph, that is a change nobody should make by accident.
#[test]
fn library_discovery_is_confined_to_packfile() {
    for (cell, graph) in cells() {
        let found: BTreeSet<&str> = DISCOVERY_CRATES
            .iter()
            .copied()
            .filter(|name| contains(graph, name))
            .collect();
        let expected: BTreeSet<&str> = cell.discovery.iter().copied().collect();
        assert_eq!(
            found, expected,
            "library-discovery crates changed for {}. A crate that probes the build machine \
             for an installed library fails clause 1 of the rule in CONTRIBUTING.md.",
            cell.label
        );
    }
}

/// Clause 3: vendored C and assembly are allowed, and these are all of it.
///
/// The scan is restricted to packages that have a build script, because a
/// package with no build script cannot compile the sources it ships (`cc`
/// itself carries a `.c` fixture and is a build-dependency everywhere). That
/// makes the result a superset of what really gets compiled, which is the
/// right direction for a tripwire.
#[test]
fn vendored_native_code_is_only_the_documented_crates() {
    for (cell, graph) in cells() {
        let Some(expected) = cell.vendored_native else {
            continue;
        };
        let found: BTreeSet<&str> = graph
            .iter()
            .filter_map(|key| {
                let manifest = manifests().get(key)?;
                if manifest.has_build_script && ships_native_source(manifest) {
                    Some(key.0.as_str())
                } else {
                    None
                }
            })
            .collect();
        let expected: BTreeSet<&str> = expected.iter().copied().collect();
        assert_eq!(
            found, expected,
            "the set of crates compiling vendored native code changed for {}. \
             CONTRIBUTING.md lists them by name, so update it in the same change.",
            cell.label
        );
    }
}

/// The clause no `links` key and no C compile would ever have caught.
///
/// `pdfium-render` ships no C, declares no `links`, and still puts libviprs on
/// a library that is not in the tree: `--features pdfium` loads `libpdfium` at
/// runtime through `libloading`, and `--features pdfium-static` links it at
/// build time out of `PDFIUM_STATIC_LIB_PATH`. That is allowed and it is
/// documented, on one condition: it stays opt-in. The same goes for the
/// vendored-C carve-out, `zstd-sys` under `packfile`.
#[test]
fn the_external_library_features_stay_opt_in() {
    for (cell, graph) in cells() {
        let asked_for_pdfium = cell.features.iter().any(|f| f.contains("pdfium"))
            || cell.features.contains(&"--all-features");
        assert_eq!(
            contains(graph, "pdfium-render"),
            asked_for_pdfium,
            "{} must reach pdfium-render only when the feature asks for it. It is the one \
             dependency here on a library outside the tree, so it stays behind `pdfium` / \
             `pdfium-static` (CONTRIBUTING.md).",
            cell.label
        );

        let asked_for_packfile = cell.features.iter().any(|f| f.contains("packfile"))
            || cell.features.contains(&"--all-features");
        assert_eq!(
            contains(graph, "zstd-sys"),
            asked_for_packfile,
            "{} must reach zstd-sys only under `packfile`. It is the vendored-C carve-out, \
             and the carve-out is the feature (CONTRIBUTING.md).",
            cell.label
        );
    }
}
