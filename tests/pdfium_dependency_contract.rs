//! What the `pdfium-render` dependency has to say, and why each part of it
//! (libviprs#981).
//!
//! `cargo publish` strips a git source. The published manifest keeps `version`,
//! `features` and `default-features` and nothing else, so those three are the
//! entire contract a crates.io consumer gets. Every assertion here is about
//! making that contract true, because for months it was not: the crate declared
//! `^0.9` while 0.9.0 through 0.9.3 shipped a `thread_safe` feature that gated a
//! bare `unsafe impl Send + Sync` with no serialisation behind it
//! (ajrcarey/pdfium-render#262, proven with a real crash).
//!
//! # Why this reads the manifest and not the lockfile
//!
//! `Cargo.lock` is gitignored here, so a lockfile guard only runs where somebody
//! has already built. The manifest is what ships.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The `pdfium-render = { ... }` declaration, as one line with the newlines
/// taken out, so the assertions below do not depend on how it is wrapped.
fn declaration() -> String {
    let manifest =
        std::fs::read_to_string(repo_root().join("Cargo.toml")).expect("the workspace manifest");
    let start = manifest
        .find("\npdfium-render = {")
        .map(|i| i + 1)
        .expect("a `pdfium-render = {` declaration in Cargo.toml");
    let rest = &manifest[start..];
    let end = rest
        .find("}\n")
        .map(|i| i + 1)
        .expect("the declaration is closed");
    rest[..end].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// On the `pdfium_latest` branch the dependency is the fork's `pdfium_8085`
/// branch, and nothing else.
///
/// pdfium-render 0.9.4 on crates.io tops out at `pdfium_7881`, so a build whose
/// bindings match libviprs-dep's pdfium-8085 has to come from the
/// libviprs/pdfium-render `pdfium_8085` branch (libviprs#1197). A git source is
/// a second dependency wearing the same name, which is why `main` forbids one,
/// and why this branch cannot be published: `.github/workflows/publish.yml`
/// still refuses a git source, and that refusal is the point. Publishing waits
/// until upstream pdfium-render carries 8085.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn pdfium_render_comes_from_the_pdfium_8085_fork_branch() {
    let decl = declaration();
    assert!(
        decl.contains("git = \"https://github.com/libviprs/pdfium-render\""),
        "pdfium-render has to come from the libviprs fork on this branch: {decl}"
    );
    assert!(
        decl.contains("branch = \"pdfium_8085\""),
        "pdfium-render has to track the fork's pdfium_8085 branch: {decl}"
    );
    assert!(
        !decl.contains("rev ="),
        "pdfium-render is pinned to a git rev, not the branch: {decl}"
    );
}

/// A git source carries no version floor, so the floor is the branch itself.
///
/// The fork's `pdfium_8085` branch is upstream 0.9.4 plus our fixes, and its
/// `thread_safe` serialises every method. Anything older than 0.9.4 gates a
/// bare `unsafe impl Send + Sync` with nothing behind it, so the branch has to
/// declare a version of 0.9.4 or later in its own manifest, which the fork's
/// guard pins. Here we only require that the declaration names no `version`
/// that could fall back to the registry.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn the_declaration_has_no_registry_fallback() {
    let decl = declaration();
    assert!(
        !decl.contains("version ="),
        "a `version` beside a git source is two dependencies under one name \
         (#149, #981): {decl}"
    );
}

/// The libpdfium ABI is named, not inherited.
///
/// `pdfium_latest` is a floating alias and it has moved three times inside the
/// 0.9 line: 0.9.0 bound `pdfium_7543`, 0.9.1 and 0.9.2 bound `pdfium_7763`,
/// 0.9.3 and 0.9.4 bind `pdfium_7881`. Taking defaults means a patch release of
/// the wrapper silently swaps the whole bindgen set against a `libpdfium.so`
/// this repo pins separately, and the failure is a signature mismatch at
/// runtime rather than a compile error. `Cargo.lock` is gitignored, so there is
/// no lockfile to catch it either.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn the_libpdfium_abi_is_pinned_explicitly() {
    let decl = declaration();
    assert!(
        decl.contains("default-features = false"),
        "pdfium-render takes default features, which turns on `pdfium_latest` \
         and lets a patch release move the ABI. The declaration reads: {decl}"
    );
    assert!(
        !decl.contains("pdfium_latest"),
        "pdfium_latest is a floating alias; name the milestone instead: {decl}"
    );
    let named = decl.contains("pdfium_8085");
    assert!(
        named,
        "no `pdfium_XXXX` feature is named, so nothing says which libpdfium ABI \
         these bindings are for: {decl}"
    );
}

/// `thread_safe` stays named, and it is not a stylistic choice.
///
/// libviprs keeps its `Pdfium` in a `static OnceLock<Pdfium>`, and
/// `OnceLock<T>: Sync` needs `T: Send + Sync`. In 0.9.4 those impls are behind
/// `#[cfg(feature = "thread_safe")]`, so without the feature this crate does
/// not compile at all. Measured by building without it: E0277 at
/// `src/pdf.rs`, and again in `streaming.rs` for `PdfDocument`.
///
/// What the feature does **not** do is make libviprs' calls safe. 0.9.4 locks
/// 290 of its 484 binding methods, and `FPDF_RenderPageBitmapWithMatrix`, which
/// every `.apply_matrix()` render goes through, is one of the unlocked ones.
/// libviprs is safe because it holds `pdfium_lock()` across whole operations
/// itself, not because the wrapper serialises.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn thread_safe_is_requested_because_the_singleton_needs_send_and_sync() {
    let decl = declaration();
    assert!(
        decl.contains("thread_safe"),
        "pdfium-render must request `thread_safe`: `static PDFIUM: \
         OnceLock<Pdfium>` does not compile without it. The declaration reads: \
         {decl}"
    );
}
