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
//! library is in the graph outside the one feature that carries one, that no
//! crate whose job is to *open* one at runtime is either, that the crates
//! shipping compilable C or assembly are exactly the documented ones, and that
//! the features which do reach an external library stay off by default.
//!
//! # What this file cannot check
//!
//! Clauses 1 and 3 are mechanical and the checks below are the whole of them.
//! Clause 2, no dependency may link a third-party library somebody has to
//! install first, is not mechanical and cannot be made so: nothing in a
//! manifest separates a crate that needs an installed library from one that
//! does not. `pdfium-render` is the proof. It ships no C, declares no `links`
//! key, carries no `-sys` suffix, and it is the one dependency here that the
//! rule is really about.
//!
//! So the clause-2 checks here are proxies, and two of them are hard-coded
//! name lists. [`DISCOVERY_CRATES`] is five names, so `autotools` or
//! `metadeps` would walk straight past it, and [`RUNTIME_LOADER_CRATES`] is
//! three. A crate that finds or loads a library under a name nobody wrote down
//! goes green here. Clause 2 gets applied by hand, against the checklist in
//! `CONTRIBUTING.md`, and these tripwires exist to make that by-hand step
//! happen rather than to stand in for it.
//!
//! [`RUNTIME_LOADER_CRATES`] is here because the gap got demonstrated rather
//! than argued: a reviewer added `libloading` as an unconditional dependency,
//! a crate whose whole purpose is to `dlopen` something that is not in the
//! tree, and every cell passed. That closes one mechanism. It does not close
//! the clause.
//!
//! # Why `cargo tree` and not `cargo metadata`
//!
//! `cargo metadata`'s resolve graph lists optional dependencies that nothing
//! enabled, so a package can sit in it while no build ever compiles it.
//! `--filter-platform` does not fix that: it filters by target cfg, not by
//! feature. Measured while writing this, the `aarch64-apple-darwin` filter
//! took the resolve from 193 packages to 163, and all 30 it dropped were
//! target-gated (`wasm-bindgen` and its tail, the `windows-*` family,
//! `linux-raw-sys`, `js-sys`); the unenabled optionals came through it
//! untouched. So the package set here comes from `cargo tree`, which is
//! feature-aware, and `cargo metadata` is only a lookup table for the manifest
//! facts (`links`, build script, source directory) of the packages `cargo
//! tree` named.
//!
//! The example on that resolution was `defmt`, an unenabled optional of
//! `chrono`, `jiff` and `tinyvec` that declares `links = "defmt"`. Do not
//! expect to reproduce that particular name. `Cargo.lock` is not committed, so
//! two checkouts a day apart resolve differently, and on an older one (`jiff`
//! 0.2.23 rather than 0.2.35) `defmt` is not in the graph at all. This test
//! does not depend on it either way: it never reads the metadata resolve
//! graph, so an unenabled optional is invisible to it by construction. It
//! passes unchanged on both of those lockfiles and on a fresh CI resolve.
//!
//! `--target` matters just as much in the other direction: without it,
//! `cargo metadata` reports every target's dependencies at once and an audit
//! run on a mac will flag `wasm-bindgen-shared`.
//!
//! # Cost
//!
//! One `cargo metadata --all-features` plus one `cargo tree` per cell below.
//! Measured warm: the metadata call 0.09 s, a single tree 0.05 to 0.09 s, the
//! whole file 1.3 s over twenty-one cells, and 1.8 s with other cargo work
//! running alongside it. The number worth knowing is not that one, though: the
//! real cost is a *cold* multi-target resolve, where cargo has to fetch every
//! manifest in four target graphs before it can answer at all.
//!
//! The metadata call runs first on purpose. It downloads every manifest in the
//! graph, including the windows- and wasm-only ones no host build ever needs,
//! so the per-target trees after it resolve without further network access.
//! The flip side is that `cargo test --offline` on a machine that has never
//! fetched those manifests fails here at the `assert!(out.status.success())`
//! in [`manifests`] or [`resolve`], with cargo's error rather than a policy
//! message. Nothing is wrong with the tree when that happens; the resolver
//! just cannot answer the question offline.

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
///
/// This is a closed list of names, not a property, so it only catches the
/// crates somebody thought of. `autotools` and `metadeps` are two that are not
/// on it. See "What this file cannot check" above.
const DISCOVERY_CRATES: &[&str] = &["pkg-config", "system-deps", "vcpkg", "cmake", "bindgen"];

/// Crates whose whole job is to open a shared library at runtime, which is how
/// a dependency can need an installed library while declaring nothing at all.
///
/// This is the tripwire for the `pdfium-render` shape: pure Rust, no `links`
/// key, no `.c` file, and still unusable without a `libpdfium` on the machine.
/// The same closed-list caveat as [`DISCOVERY_CRATES`] applies, and it matters
/// more here, because a crate can `dlopen` with nothing but `libc`.
const RUNTIME_LOADER_CRATES: &[&str] = &["libloading", "dlopen", "dlopen2"];

/// Extensions a `build.rs` can hand to a C toolchain.
const NATIVE_SOURCE_EXTENSIONS: &[&str] = &["c", "cc", "cpp", "cxx", "s", "asm"];

/// Stand-in for "whatever machine this is running on", resolved by
/// [`host_triple`] out of `rustc -vV`.
///
/// It used to be spelled `aarch64-apple-darwin`, which made every cell labelled
/// "host" a lie on the linux runners: the label said host and the graph said
/// mac. The value here is deliberately not a valid triple, so anything that
/// forgets to route through [`target_of`] fails in cargo rather than quietly
/// resolving something else.
const HOST: &str = "<host>";

/// The four fixed triples, checked in addition to the host. They cover the
/// platform-conditional edges: `core-foundation-sys` on macOS, `linux-raw-sys`
/// on linux, the windows import crates, and `wasm-bindgen` on wasm32, which is
/// the one place a second `links` key shows up.
const MACOS: &str = "aarch64-apple-darwin";
const LINUX: &str = "x86_64-unknown-linux-gnu";
const WINDOWS: &str = "x86_64-pc-windows-msvc";
const WASM: &str = "wasm32-unknown-unknown";

/// One (target, feature set) pair, with everything the rule claims about it.
struct Cell {
    /// Human name, used in failure messages.
    label: &'static str,
    /// The triple to resolve for, or [`HOST`] for this machine's own.
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
    /// Every [`RUNTIME_LOADER_CRATES`] entry expected in this graph.
    runtime_loaders: &'static [&'static str],
    /// Every crate expected to ship compilable native source *and* a build
    /// script able to compile it.
    ///
    /// This check reads crate source directories off disk, and it runs on every
    /// cell rather than only the host ones. It can: [`manifests`] resolves with
    /// no `--filter-platform`, so cargo has unpacked every package any cell can
    /// name, `windows-sys` and `wasm-bindgen-shared` included. Clause 3 is the
    /// clause this whole document is built around, so checking it on one target
    /// was the wrong place to economise.
    vendored_native: &'static [&'static str],
}

const RAYON: (&str, &str) = ("rayon-core", "rayon-core");
const ZSTD: (&str, &str) = ("zstd-sys", "zstd");
const WASM_BINDGEN: (&str, &str) = ("wasm-bindgen-shared", "wasm_bindgen");

/// `blake3` is a direct dependency on every target and compiles its own C, so
/// it is in every cell's clause-3 set; `zstd-sys` joins it wherever `packfile`
/// resolves.
const BLAKE3: &[&str] = &["blake3"];
const BLAKE3_ZSTD: &[&str] = &["blake3", "zstd-sys"];

/// `pdfium-render` takes `libloading` as an unconditional dependency under
/// `cfg(not(target_arch = "wasm32"))`, so every cell that asks for a pdfium
/// feature carries it, and the wasm32 cells do not.
const LIBLOADING: &[&str] = &["libloading"];

const CELLS: &[Cell] = &[
    // The default build, which is what most consumers get.
    Cell {
        label: "host / default",
        target: HOST,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    // The one carve-out that compiles vendored C beyond blake3, and the only
    // place a discovery crate is allowed to appear at all.
    Cell {
        label: "host / packfile",
        target: HOST,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
    // The two codec features that cost real crate count. Both add nothing
    // here, which is the claim their `Cargo.toml` comments make.
    Cell {
        label: "host / svg",
        target: HOST,
        features: &["--features", "svg"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "host / jxl",
        target: HOST,
        features: &["--features", "jxl"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    // The external-library features. Neither shows up in `links` or in a C
    // compile, which is precisely why a check keyed on those would miss them.
    // `libloading` is the one mechanical trace either of them leaves, and
    // `the_external_library_features_stay_opt_in` below covers the rest.
    Cell {
        label: "host / pdfium",
        target: HOST,
        features: &["--features", "pdfium"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3,
    },
    Cell {
        label: "host / pdfium-static",
        target: HOST,
        features: &["--features", "pdfium-static"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3,
    },
    // Every remaining feature at once. These are the ones whose Cargo.toml
    // comments say they add no dependencies, so the cell is the check on that.
    Cell {
        label: "host / object-store-sink + tracing + serde + test-util + s3",
        target: HOST,
        features: &["--features", "object-store-sink,tracing,serde,test-util,s3"],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "host / all features",
        target: HOST,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3_ZSTD,
    },
    // Dev-dependencies never reach a consumer, but they do have to build on a
    // contributor's machine. `generator` is loom's coroutine crate; it
    // assembles its own vendored stack-switching `.s` files, so it lands under
    // clause 3 the same way blake3 does.
    Cell {
        label: "host / default + dev-dependencies",
        target: HOST,
        features: &[],
        edges: "normal,build,dev",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: &["blake3", "generator"],
    },
    // The four fixed triples. The host cells above already cover one of these
    // on any given machine; these are what make the answer the same whoever
    // runs it.
    Cell {
        label: "macos / default",
        target: MACOS,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "macos / packfile",
        target: MACOS,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "macos / all features",
        target: MACOS,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "linux / default",
        target: LINUX,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "linux / packfile",
        target: LINUX,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "linux / all features",
        target: LINUX,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "windows / default",
        target: WINDOWS,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "windows / packfile",
        target: WINDOWS,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "windows / all features",
        target: WINDOWS,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: LIBLOADING,
        vendored_native: BLAKE3_ZSTD,
    },
    // wasm32 is where the third `links` key lives. `packfile` is what puts it
    // there: `zip` turns on `getrandom`'s `wasm_js` backend and pulls `time`'s
    // wasm clock through `js-sys`, and both reach `wasm-bindgen`. It is also
    // the one target where the pdfium features carry no `libloading`, since
    // `pdfium-render` gates that dependency on `cfg(not(target_arch =
    // "wasm32"))` and reaches PDFium through `js-sys` instead.
    Cell {
        label: "wasm32 / default",
        target: WASM,
        features: &[],
        edges: "normal,build",
        links: &[RAYON],
        discovery: &[],
        runtime_loaders: &[],
        vendored_native: BLAKE3,
    },
    Cell {
        label: "wasm32 / packfile",
        target: WASM,
        features: &["--features", "packfile"],
        edges: "normal,build",
        links: &[RAYON, WASM_BINDGEN, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
    Cell {
        label: "wasm32 / all features",
        target: WASM,
        features: &["--all-features"],
        edges: "normal,build",
        links: &[RAYON, WASM_BINDGEN, ZSTD],
        discovery: &["pkg-config"],
        runtime_loaders: &[],
        vendored_native: BLAKE3_ZSTD,
    },
];

/// Repo root (the directory holding the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

/// This machine's target triple, out of `rustc -vV`, which is the same way the
/// shell recipe in `CONTRIBUTING.md` gets it.
fn host_triple() -> &'static str {
    static ONCE: OnceLock<String> = OnceLock::new();
    ONCE.get_or_init(|| {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let out = Command::new(&rustc)
            .arg("-vV")
            .output()
            .unwrap_or_else(|err| panic!("failed to spawn `{rustc} -vV`: {err}"));
        assert!(
            out.status.success(),
            "`{rustc} -vV` failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8(out.stdout).expect("rustc -vV emitted non-UTF-8");
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .unwrap_or_else(|| panic!("`{rustc} -vV` printed no `host:` line:\n{stdout}"))
            .trim()
            .to_owned()
    })
}

/// The triple to hand `cargo tree` for one cell.
fn target_of(cell: &Cell) -> &'static str {
    if cell.target == HOST {
        host_triple()
    } else {
        cell.target
    }
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
/// covers every package any cell below can name, and so every one of them has
/// been unpacked on disk for [`ships_native_source`] to read. This is a lookup
/// table only: which packages are actually *in* a given graph comes from
/// `cargo tree`.
///
/// (name, version) is not a package identity. Two packages can share both and
/// differ in source, and this tree already has a git source (`pdfium-render`),
/// so the collision is not hypothetical. A silent overwrite here would hide one
/// of them from every check below, so it is a panic rather than a shrug.
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

        let mut table: BTreeMap<(String, String), Manifest> = BTreeMap::new();
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
            let manifest_path =
                PathBuf::from(pkg["manifest_path"].as_str().expect("manifest_path"));
            let clash = table.insert(
                (name.clone(), version.clone()),
                Manifest {
                    links: pkg["links"].as_str().map(str::to_owned),
                    manifest_path: manifest_path.clone(),
                    has_build_script,
                },
            );
            if let Some(clash) = clash {
                panic!(
                    "cargo metadata lists {name} {version} twice, at {} and at {}. This table \
                     is keyed on (name, version) alone, so one of them would shadow the other \
                     and its `links` key and native sources would go unchecked.",
                    clash.manifest_path.display(),
                    manifest_path.display(),
                );
            }
        }
        table
    })
}

/// The manifest facts for a package `cargo tree` named.
///
/// A miss is a hard error. Dropping the package instead would hide its `links`
/// key and its native sources from the set-equality checks below, which is the
/// green direction for a tripwire whose entire job is noticing a new crate.
fn manifest_of(key: &(String, String)) -> &'static Manifest {
    manifests().get(key).unwrap_or_else(|| {
        panic!(
            "cargo tree named {key:?}, which cargo metadata did not. The lookup table is keyed \
             on (name, version) and built with --all-features and no --filter-platform, so a \
             miss means the two calls disagree and the checks below cannot see this package."
        )
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
        target_of(cell),
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
    let mut at_root = true;
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            // An unreadable crate root would answer "ships no native source",
            // which is the wrong direction to fail in: the whole point is to
            // notice a crate that started shipping some.
            Err(err) if at_root => panic!(
                "cannot read the crate root {} while checking it for native source: {err}",
                dir.display()
            ),
            // A subdirectory is different. It cannot hide the crate, only part
            // of it, and a dangling symlink in a registry checkout is not
            // evidence of anything, so walk on.
            Err(_) => continue,
        };
        at_root = false;
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

/// Clause 2, the first mechanical proxy: the `links` keys in every resolved
/// graph are exactly the ones `CONTRIBUTING.md` names, and a new one cannot
/// appear unnoticed.
///
/// `links` is not by itself evidence of a C library, and this test is not
/// claiming it is. It is a cheap, exact tripwire: a crate that links something
/// almost always declares one, so pinning the set to an allowlist means any
/// new candidate has to be looked at by a human before this goes green again.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn every_links_key_is_on_the_allowlist() {
    for (cell, graph) in cells() {
        let found: BTreeSet<(&str, &str)> = graph
            .iter()
            .filter_map(|key| {
                let links = manifest_of(key).links.as_deref()?;
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
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
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

/// Clause 2, the second mechanical proxy: nothing in the tree opens a shared
/// library at runtime except where a feature asked for exactly that.
///
/// This is the check that would have caught `pdfium-render` if `pdfium-render`
/// had arrived unannounced, and it exists because the version of this file
/// without it let an unconditional `libloading` through every cell. It says
/// nothing about *which* library gets loaded, so it is a prompt to go and read
/// the crate rather than a verdict on it.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn runtime_library_loaders_are_confined_to_the_pdfium_features() {
    for (cell, graph) in cells() {
        let found: BTreeSet<&str> = RUNTIME_LOADER_CRATES
            .iter()
            .copied()
            .filter(|name| contains(graph, name))
            .collect();
        let expected: BTreeSet<&str> = cell.runtime_loaders.iter().copied().collect();
        assert_eq!(
            found, expected,
            "runtime library-loader crates changed for {}. A crate that opens a shared library \
             at runtime needs one installed on the machine, which is clause 2 of the rule in \
             CONTRIBUTING.md, and it declares nothing that the other checks here can see.",
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
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn vendored_native_code_is_only_the_documented_crates() {
    for (cell, graph) in cells() {
        let found: BTreeSet<&str> = graph
            .iter()
            .filter(|key| {
                let manifest = manifest_of(key);
                manifest.has_build_script && ships_native_source(manifest)
            })
            .map(|key| key.0.as_str())
            .collect();
        let expected: BTreeSet<&str> = cell.vendored_native.iter().copied().collect();
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
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
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
